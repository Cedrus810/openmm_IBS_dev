#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenMM ABFE 计算核心流程管理器 (v4.0 - 生产级重构)
职责：
1. 物理预平衡 (10 ns NPT → NVT) 与轨迹保存
2. ACES 路径预优化 (单λ / 双λ 路由)
3. IBS 生产采样与全局 MBAR 分析
4. Boresch 解析修正与最终结果聚合
设计原则：
- 严格控制职责边界，不混入底层力场构建逻辑
- 统一日志、错误处理与状态管理
- 与 ibs_engine.py / abfe_preoptimizer.py 保持接口兼容
"""

import openmm
from openmm import app, unit, XmlSerializer
import numpy as np
import os
import json
import shutil
import multiprocessing as mp
import time
import logging
import builtins
import hashlib
import platform
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 项目内部模块依赖
from abfe_preoptimizer import ABFEPreOptimizer, DualLambdaPreOptimizer
from abfe_preoptimizer import generate_overlapping_windows
from abfe_preoptimizer import build_aces_probe_system, build_aces_probe_system_dual_lambda
from abfe_preoptimizer import generate_overlapping_windows   # ✅ 保留这个
from abfe_preoptimizer import refine_stage_lambda_path_from_data
from ibs_engine import (
    IBSWindowManagerDualLambda,
    GlobalMBARAnalyzer,
    solve_stage_integrated,
    REMDManager,
    TraditionalMBARAnalyzer,
    generate_overlapping_windows,
    lambda_endpoint_diagnostics,
)
from abfe_core import (
    calculate_boresch_analytical_correction,
    ACESoftcorePotential,
    BeutlerSoftcoreBuilder,
    DEXPSurrogatePotential,
    run_orbv3_dexp_fitting,
    UnitFormatter,
    TwoDimensionalLambdaPathPlanner,
    THERMODYNAMIC_CYCLE_DOC,
)
import warnings

PME_DECHARGE_MODEL_VERSION = "pme_decharge_v2_llfreeze_pmeself_20260523"

logger = logging.getLogger(__name__)


def _infer_log_level_from_message(message: str) -> int:
    if any(token in message for token in ("⚠️", "警告", "warning")):
        return logging.WARNING
    if any(token in message for token in ("🚨", "❌", "失败", "错误", "异常", "error")):
        return logging.ERROR
    return logging.INFO


def _log_print(*args, sep=" ", end="\n", file=None, flush=False):
    message = sep.join(str(arg) for arg in args)
    if end and end != "\n":
        message += end.rstrip("\n")
    if logger.handlers:
        logger.log(_infer_log_level_from_message(message), message)
    builtins.print(*args, sep=sep, end=end, file=file, flush=flush)


print = _log_print


def _resolve_alchemical_params(
    potential_type: str,
    dexp_params: Optional[Dict],
    ligand_indices: List[int],
):
    if potential_type == "dexp":
        return DEXPSurrogatePotential.from_dict(dexp_params or {})
    return ACESoftcorePotential.from_dict(
        ACESoftcorePotential.optimize_alpha(len(ligand_indices))
    )


def _pme_u_kn_meta_path(stage_output_dir: str, stage_name: str) -> str:
    return os.path.join(stage_output_dir, f"{stage_name}_pme_u_kn.meta.json")


def _lambda_signature(values: List[float]) -> List[float]:
    return [round(float(v), 8) for v in values]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _system_xml_hash(system: Optional[openmm.System]) -> Optional[str]:
    if system is None:
        return None
    return _sha256_text(XmlSerializer.serialize(system))


def _topology_hash(topology: Optional[app.Topology]) -> Optional[str]:
    if topology is None:
        return None
    atoms = [
        (
            int(atom.index),
            str(atom.name),
            str(atom.element.symbol if atom.element is not None else ""),
            str(atom.residue.name),
            int(atom.residue.index),
            str(atom.residue.chain.id),
        )
        for atom in topology.atoms()
    ]
    bonds = sorted((int(a1.index), int(a2.index)) for a1, a2 in topology.bonds())
    box = topology.getPeriodicBoxVectors()
    if box is not None:
        box_nm = []
        for vec in box:
            if hasattr(vec, "value_in_unit"):
                values = vec.value_in_unit(unit.nanometer)
            else:
                values = vec
            box_nm.append([round(float(v), 10) for v in values])
    else:
        box_nm = None
    return _sha256_text(json.dumps({"atoms": atoms, "bonds": bonds, "box_nm": box_nm}, sort_keys=True))


def _positions_hash(positions) -> Optional[str]:
    if positions is None:
        return None
    try:
        if hasattr(positions, "value_in_unit"):
            arr = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
        else:
            arr = np.asarray(positions, dtype=np.float64)
    except Exception:
        try:
            arr = np.asarray([[p.x, p.y, p.z] for p in positions], dtype=np.float64)
        except Exception:
            return None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()


def _code_hash() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    payload = {}
    for name in ("abfe_pipeline.py", "abfe_core.py", "ibs_engine.py", "abfe_preoptimizer.py"):
        path = os.path.join(base_dir, name)
        try:
            with open(path, "rb") as handle:
                payload[name] = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            payload[name] = None
    return _sha256_text(json.dumps(payload, sort_keys=True))


def _package_version(package_name: str) -> Optional[str]:
    try:
        from importlib import metadata
        return metadata.version(package_name)
    except Exception:
        return None


def _collect_pipeline_provenance(
    *,
    config: Optional[Dict],
    system: Optional[openmm.System],
    topology: Optional[app.Topology],
    positions,
    command_line: Optional[List[str]] = None,
) -> Dict:
    env_seed_keys = ("OPENMM_RANDOM_SEED", "ABFE_RANDOM_SEED", "PYTHONHASHSEED")
    return {
        "config": config or {},
        "command_line": command_line if command_line is not None else sys.argv,
        "hashes": {
            "system_xml_sha256": _system_xml_hash(system),
            "topology_sha256": _topology_hash(topology),
            "coordinates_nm_sha256": _positions_hash(positions),
            "code_sha256": _code_hash(),
        },
        "random_seeds": {
            key: os.environ.get(key)
            for key in env_seed_keys
            if os.environ.get(key) is not None
        },
        "software_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "openmm": getattr(openmm, "__version__", None),
            "numpy": getattr(np, "__version__", None),
            "pymbar": _package_version("pymbar"),
            "mdtraj": _package_version("mdtraj"),
        },
        "thermodynamic_cycle": THERMODYNAMIC_CYCLE_DOC,
    }


def _pme_u_kn_meta_payload(
    n_states: int,
    lambdas_coul: List[float],
    lambdas_vdw: List[float],
    temperature_k: float,
    system: Optional[openmm.System],
    topology: Optional[app.Topology],
    ligand_indices: Optional[List[int]],
    boresch_params: Optional[Dict],
) -> Dict:
    boresch_sig = None
    if boresch_params:
        boresch_sig = {
            "receptor_indices": [int(i) for i in boresch_params.get("receptor_indices", [])],
            "ligand_indices": [int(i) for i in boresch_params.get("ligand_indices", [])],
            "equilibrium_values": {
                str(k): round(float(v), 8)
                for k, v in (boresch_params.get("equilibrium_values") or {}).items()
            },
            "force_constants": {
                str(k): round(float(v), 8)
                for k, v in (boresch_params.get("force_constants") or {}).items()
            },
        }
    return {
        "model_version": PME_DECHARGE_MODEL_VERSION,
        "n_states": int(n_states),
        "temperature_k": round(float(temperature_k), 6),
        "lambdas_coul": _lambda_signature(lambdas_coul),
        "lambdas_vdw": _lambda_signature(lambdas_vdw),
        "n_particles": int(system.getNumParticles()) if system is not None else None,
        "n_forces": int(system.getNumForces()) if system is not None else None,
        "system_xml_sha256": _system_xml_hash(system),
        "topology_sha256": _topology_hash(topology),
        "code_sha256": _code_hash(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "openmm": getattr(openmm, "__version__", None),
            "pymbar": _package_version("pymbar"),
        },
        "ligand_indices": [int(i) for i in (ligand_indices or [])],
        "boresch": boresch_sig,
    }


def _is_pme_u_kn_cache_compatible(
    stage_output_dir: str,
    stage_name: str,
    n_states: int,
    lambdas_coul: List[float],
    lambdas_vdw: List[float],
    temperature_k: float,
    system: Optional[openmm.System],
    topology: Optional[app.Topology],
    ligand_indices: Optional[List[int]],
    boresch_params: Optional[Dict],
) -> bool:
    meta_path = _pme_u_kn_meta_path(stage_output_dir, stage_name)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta == _pme_u_kn_meta_payload(
            n_states=n_states,
            lambdas_coul=lambdas_coul,
            lambdas_vdw=lambdas_vdw,
            temperature_k=temperature_k,
            system=system,
            topology=topology,
            ligand_indices=ligand_indices,
            boresch_params=boresch_params,
        )
    except Exception:
        return False


def _write_pme_u_kn_meta(
    stage_output_dir: str,
    stage_name: str,
    n_states: int,
    lambdas_coul: List[float],
    lambdas_vdw: List[float],
    temperature_k: float,
    system: Optional[openmm.System],
    topology: Optional[app.Topology],
    ligand_indices: Optional[List[int]],
    boresch_params: Optional[Dict],
) -> None:
    meta_path = _pme_u_kn_meta_path(stage_output_dir, stage_name)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            _pme_u_kn_meta_payload(
                n_states=n_states,
                lambdas_coul=lambdas_coul,
                lambdas_vdw=lambdas_vdw,
                temperature_k=temperature_k,
                system=system,
                topology=topology,
                ligand_indices=ligand_indices,
                boresch_params=boresch_params,
            ),
            f,
            indent=2,
        )


def _has_valid_boresch_restraint(params: Optional[Dict]) -> bool:
    """仅当 Boresch 参数包含完整 3+3 锚点时才认为可启用。"""
    if not isinstance(params, dict):
        return False
    rec_idx = params.get("receptor_indices") or []
    lig_idx = params.get("ligand_indices") or []
    return len(rec_idx) == 3 and len(lig_idx) == 3


class _PipelineStateLock:
    def __init__(self, path: str, timeout_s: float = 10.0, poll_s: float = 0.05):
        self.path = path
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.fd = None

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        except PermissionError:
            return True
        return True

    def _break_stale_lock_if_needed(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = f.read().strip()
            pid = int(payload) if payload else -1
        except Exception:
            pid = -1
        if pid > 0 and self._pid_is_alive(pid):
            return
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass

    def __enter__(self):
        deadline = time.time() + self.timeout_s
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, str(os.getpid()).encode("utf-8"))
                return self
            except FileExistsError:
                self._break_stale_lock_if_needed()
                if time.time() >= deadline:
                    raise TimeoutError(f"获取状态文件锁超时: {self.path}")
                time.sleep(self.poll_s)

    def __exit__(self, exc_type, exc, tb):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
        except Exception:
            pass

#============================================================================
# 辅助函数：统一 Simulation Reporter 挂载工具 (Step 1)
#============================================================================
def attach_simulation_reporters(
    simulation: app.Simulation,
    prefix: str,
    output_dir: str,
    traj_interval: int = 5000,      # 轨迹保存间隔 (步)
    energy_interval: int = 1000,    # 能量日志间隔
    chk_interval: int = 10000,      # Checkpoint 间隔
    append_traj: bool = False
):
    """为任意 Simulation 实例统一挂载轨迹、能量、Checkpoint Reporter"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 轨迹
    dcd_path = os.path.join(output_dir, f"{prefix}_traj.dcd")
    simulation.reporters.append(app.DCDReporter(dcd_path, traj_interval, append=append_traj, enforcePeriodicBox=False))
    
    # 2. 能量日志 (包含势能、温度、密度等)
    log_path = os.path.join(output_dir, f"{prefix}_energy.log")
    simulation.reporters.append(app.StateDataReporter(
        log_path, energy_interval,
        step=True, time=True, potentialEnergy=True, temperature=True,
        volume=True, density=True, speed=True, separator="	",
        totalSteps=simulation.currentStep + 10000000  # 防截断
    ))
    
    # 3. Checkpoint
    chk_path = os.path.join(output_dir, f"{prefix}.chk")
    simulation.reporters.append(app.CheckpointReporter(chk_path, chk_interval))
    
    return dcd_path, log_path, chk_path

#============================================================================
# 辅助函数：Checkpoint 与轨迹完整性校验 (Step 3)
#============================================================================
def _is_checkpoint_valid(chk_path: str) -> bool:
    """检查 Checkpoint 是否可读且非空"""
    if not os.path.exists(chk_path) or os.path.getsize(chk_path) < 512:
        return False
    try:
        with open(chk_path, "rb") as f:
            f.seek(-8, 2)  # 跳至文件末尾
            return True
    except:
        return False

def _is_traj_valid(dcd_path: str, min_frames: int = 1) -> bool:
    """检查 DCD 轨迹是否完整 (✅ 修复：启用 min_frames 校验与结构验证)"""
    if not os.path.exists(dcd_path):
        return False
    
    file_size = os.path.getsize(dcd_path)
    # DCD 标准头 212 字节。保守估计每帧至少 64 字节 (4原子坐标+边界)
    min_required_size = 212 + (min_frames * 64)
    if file_size < min_required_size:
        return False
        
    try:
        with open(dcd_path, "rb") as f:
            # 1. 校验 DCD 魔数 (CORD) 与基础头信息
            header = f.read(212)
            if b"CORD" not in header:
                return False
                
            # 2. 尝试读取第一帧尺寸记录 (4字节) 验证流可读性
            f.seek(212)
            frame_size_bytes = f.read(4)
            if len(frame_size_bytes) < 4:
                return False
                
        return True
    except Exception:
        return False


def _expected_remd_traj_files(stage_output_dir: str, stage_name: str, n_replicas: int) -> List[str]:
    return [os.path.join(stage_output_dir, f"{stage_name}_rep{i}.dcd") for i in range(int(n_replicas))]


def _expected_remd_frame_count(n_steps: int, save_interval: int = 5000) -> int:
    if n_steps <= 0 or save_interval <= 0:
        return 0
    return int(n_steps // save_interval)


def _all_remd_trajs_valid(stage_output_dir: str, stage_name: str, n_replicas: int, min_frames: int = 1) -> bool:
    traj_files = _expected_remd_traj_files(stage_output_dir, stage_name, n_replicas)
    return all(_is_traj_valid(path, min_frames=min_frames) for path in traj_files)

def cleanup_temp_files(checkpoint_dir: str):
    """清理损坏的临时文件 (.tmp)"""
    if not os.path.exists(checkpoint_dir):
        return
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".chk.tmp") or f.endswith(".dcd.tmp"):
            try:
                os.remove(os.path.join(checkpoint_dir, f))
                print(f"  🗑️ 已清理临时文件: {f}")
            except Exception as e:
                print(f"  ⚠️ 清理失败 {f}: {e}")

#============================================================================
# 辅助函数：能量聚合 (Step 5)
#============================================================================
def aggregate_all_energies(output_dir: str):
    import glob as glob_module
    all_e = [np.load(f) for f in glob_module.glob(os.path.join(output_dir, "*_energies.npy"))]
    if not all_e: return False
    
    # ✅ 确保每张矩阵为 (K, N_frames) 格式，并沿帧维度水平拼接
    all_e = [arr.T if arr.shape[0] > arr.shape[1] else arr for arr in all_e]
    u_kn_global = np.hstack(all_e)  # 形状: (K, total_frames)
    
    np.save(os.path.join(output_dir, "full_u_kn_matrix.npy"), u_kn_global)
    print(f"  ✓ 已聚合 {len(all_e)} 个窗口能量，全局矩阵形状: {u_kn_global.shape}")
    return True

def _split_platform_spec(platform_name: str) -> Tuple[str, Optional[str]]:
    """解析平台字符串，支持 'CUDA:1' 这种显式设备写法。"""
    spec = str(platform_name or "CPU").strip()
    if ":" not in spec:
        return spec, None
    base, device = spec.split(":", 1)
    base = base.strip() or "CPU"
    device = device.strip() or None
    return base, device


def _build_platform_props(platform_name: str) -> Tuple[str, Dict[str, str]]:
    base, device = _split_platform_spec(platform_name)
    upper = base.upper()
    props: Dict[str, str] = {}
    if upper == "CUDA":
        props["Precision"] = "mixed"
        if device is not None:
            props["DeviceIndex"] = device
        if shutil.which("nvcc"):
            props["CudaCompiler"] = "nvcc"
    elif upper == "OPENCL":
        props["Precision"] = "mixed"
        if device is not None:
            props["DeviceIndex"] = device
    return base, props
#============================================================================
# 辅助类：NumpyEncoder (JSON 序列化支持)
#============================================================================
class NumpyEncoder(json.JSONEncoder):
    """🔑 支持 numpy 数组/类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


# =============================================================================
# 多进程工作函数：双阶段并行采样
# =============================================================================
def _run_stage_worker_process(
    state_dir: str,
    temperature_k: float,
    platform_name: str,
    output_dir: str,
    stage_name: str,
    fixed_lam_coul: float,
    fixed_lam_vdw: float,
    n_states: int,
    n_steps_per_window: int,
    steps_per_update: int,
    system_type: str,
    potential_type: str,
    dexp_params: Optional[Dict],
    optimized_lambdas: Optional[List[float]],
    enable_early_stop: bool,
    boresch_params: Optional[Dict],
    enable_gradual_warmup: bool,
    warmup_steps: int,
    resume: bool,
    result_file: str,
):
    """子进程工作函数：加载保存的Pipeline状态并执行一个双λ阶段"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import json as _json
    import numpy as _np
    from openmm import app as _app, unit as _unit, Vec3 as _Vec3, XmlSerializer as _XmlSerializer

    with open(os.path.join(state_dir, "system.xml")) as _f:
        _system = _XmlSerializer.deserialize(_f.read())
    _pdbx = app.PDBxFile(os.path.join(state_dir, "topology.cif"))
    _topology = _pdbx.topology
    _pos_np = _np.load(os.path.join(state_dir, "positions.npy"))
    _positions = [_Vec3(float(_v[0]), float(_v[1]), float(_v[2])) for _v in _pos_np] * _unit.nanometer
    _bv_np = _np.load(os.path.join(state_dir, "box_vectors.npy"))
    _box_vectors = [_Vec3(float(_v[0]), float(_v[1]), float(_v[2])) for _v in _bv_np] * _unit.nanometer
    with open(os.path.join(state_dir, "ligand_indices.json")) as _f:
        _ligand_indices = _json.load(_f)

    from abfe_pipeline import ABFEPipeline as _Pipeline
    _stage_ckpt_dir = os.path.join(output_dir, "checkpoints", stage_name)
    _pipeline = _Pipeline(
        system=_system,
        topology=_topology,
        positions=_positions,
        box_vectors=_box_vectors,
        ligand_indices=_ligand_indices,
        temperature=temperature_k,
        output_dir=output_dir,
        checkpoint_dir=_stage_ckpt_dir,
        platform_name=platform_name,
    )
    _result = _pipeline._run_dual_lambda_stage(
        stage_name=stage_name,
        fixed_lam_coul=fixed_lam_coul,
        fixed_lam_vdw=fixed_lam_vdw,
        n_states=n_states,
        n_steps_per_window=n_steps_per_window,
        steps_per_update=steps_per_update,
        system_type=system_type,
        resume=resume,
        potential_type=potential_type,
        dexp_params=dexp_params,
        optimized_lambdas=optimized_lambdas,
        enable_early_stop=enable_early_stop,
        boresch_params=boresch_params,
        enable_gradual_warmup=enable_gradual_warmup,
        warmup_steps=warmup_steps,
    )
    with open(result_file, "w") as _f:
        _json.dump(_result, _f, indent=2)


class ABFEPipeline:
    """ABFE 计算流程管理器"""

    def __init__(
        self,
        system: openmm.System,
        topology: app.Topology,
        positions: List[unit.Quantity],
        box_vectors: Optional[List[unit.Quantity]] = None,
        ligand_indices: List[int] = None,
        temperature: float = 300.0,
        pressure: float = 1.0,
        output_dir: str = "./output",
        checkpoint_dir: Optional[str] = None,
        platform_name: str = "CUDA",
    ):

        # 统一温度/压力单位
        self.temperature = (
            temperature * unit.kelvin
            if isinstance(temperature, (int, float))
            else temperature
        )
        self.pressure = (
            pressure * unit.bar if isinstance(pressure, (int, float)) else pressure
        )

        # 系统与拓扑状态
        self.system = system
        self.topology = topology
        self.positions = positions
        self.box_vectors = box_vectors
        self.ligand_indices = ligand_indices or []

        # 路径配置
        self.output_dir = os.path.abspath(output_dir)
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            self.output_dir, "checkpoints"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 运行状态
        self.log_file = os.path.join(self.output_dir, "pipeline.log")
        self.results = {}
        self.platform_name = platform_name

        self._log(f"{'=' * 60}")
        self._log(f"ABFE Pipeline v4.0 初始化完成 | {datetime.now().isoformat()}")
        self._log(f"输出目录: {self.output_dir}")
        self._log(
            f"配体原子数: {len(self.ligand_indices)} | 温度: {self.temperature} | 压力: {self.pressure}"
        )
        self._log(f"{'=' * 60}")

    # =========================================================================
    # 0. Native System 缓存 (XML 持久化，支持续跑跳过 GROMACS 重建)
    # =========================================================================
    # abfe_pipeline.py -> _ensure_temperature_quantity (约第 55 行)
    @staticmethod
    def _ensure_temperature_quantity(temp_input) -> unit.Quantity:
        """确保温度参数是标准的 kelvin 单位 Quantity"""
        if hasattr(temp_input, 'unit'):
            if temp_input.unit == unit.kelvin:
                return temp_input
            # ✅ 修复：移除 kelvin**2 脏分支，改为严格校验+自动转换
            try:
                val = temp_input.value_in_unit(unit.kelvin)
                print(f"  ⚠️ 温度单位非 Kelvin ({temp_input.unit})，已自动转换: {val} K")
                return val * unit.kelvin
            except Exception:
                raise ValueError(f"无法将温度转换为 Kelvin: {temp_input}")
        else:
            return float(temp_input) * unit.kelvin

    # abfe_pipeline.py -> get_device_strategy (约第 70 行)
    @staticmethod
    def get_device_strategy(n_windows: int = 1, min_free_mb: int = 2000, platform_name: str = "CUDA"):
        import warnings
        platform_base, _ = _split_platform_spec(platform_name)
        if platform_base.upper() != "CUDA":
            return {"strategy": "cpu", "devices": [], "n_gpus": 0}
        
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("Torch CUDA unavailable")
            n_gpus = torch.cuda.device_count()
            devices = list(range(n_gpus))
        except Exception:
            msg = "🚨 [设备策略] 未检测到可用 CUDA 设备，已强制降级至 CPU。请检查 GPU 队列/驱动。"
            warnings.warn(msg, UserWarning, stacklevel=2)
            print(f"\033[93m⚠️ {msg}\033[0m")
            return {"strategy": "cpu", "devices": [], "n_gpus": 0}
            
        if n_gpus >= 2 and n_windows >= 2:
            return {"strategy": "multi_gpu", "devices": devices, "n_gpus": n_gpus}
        return {"strategy": "single_gpu", "devices": [0], "n_gpus": n_gpus}

    # =========================================================================
    # 0. 基础工具
    # =========================================================================
    def _log(self, msg: str):
        """写入日志与控制台"""
        print(msg)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {msg}\n")

    # =========================================================================
    # 0.3 并行阶段状态序列化
    # =========================================================================
    def _save_state_to_dir(self, state_dir: str):
        """将 Pipeline 状态序列化至磁盘，供子进程加载"""
        os.makedirs(state_dir, exist_ok=True)
        with open(os.path.join(state_dir, "system.xml"), "w") as f:
            f.write(XmlSerializer.serialize(self.system))
        with open(os.path.join(state_dir, "topology.cif"), "w") as f:
            app.PDBxFile.writeFile(self.topology, self.positions, f)

        pos = self.positions
        if hasattr(pos, "value_in_unit"):
            pos_np = np.array([[float(v[i]) for i in range(3)] for v in pos.value_in_unit(unit.nanometer)])
        else:
            pos_np = np.asarray(pos, dtype=np.float64)
        np.save(os.path.join(state_dir, "positions.npy"), pos_np)

        if self.box_vectors is not None:
            bv = self.box_vectors
            if hasattr(bv, "value_in_unit"):
                bv_np = np.array([[float(v[i]) for i in range(3)] for v in bv.value_in_unit(unit.nanometer)])
            else:
                bv_np = np.asarray(bv, dtype=np.float64)
        else:
            bv_np = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        np.save(os.path.join(state_dir, "box_vectors.npy"), bv_np)

        with open(os.path.join(state_dir, "ligand_indices.json"), "w") as f:
            json.dump(self.ligand_indices, f)
        self._log(f"  💾 Pipeline 状态已保存至 {state_dir}")

    # =========================================================================
    # 0.5 全局状态管理 (断点续传)
    # =========================================================================
    def _get_state_file(self) -> str:
        """获取全局状态文件路径"""
        return os.path.join(self.checkpoint_dir, "pipeline_state.json")

    def _get_state_lock_file(self) -> str:
        return self._get_state_file() + ".lock"

    def _load_pipeline_state(self) -> Dict:
        """加载 Pipeline 状态"""
        state_file = self._get_state_file()
        if os.path.exists(state_file):
            try:
                with open(state_file, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_pipeline_state(self, state: Dict):
        state_file = self._get_state_file()
        tmp_file = state_file + ".tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_file, state_file)
        except Exception as e:
            self._log(f"  ⚠️ 状态保存失败: {e}")

    def _update_stage_status(self, stage: str, status: str, extra: Dict = None):
        """更新阶段状态"""
        try:
            with _PipelineStateLock(self._get_state_lock_file()):
                state = self._load_pipeline_state()
                if "stages" not in state:
                    state["stages"] = {}

                state["stages"][stage] = {
                    "status": status,
                    "timestamp": datetime.now().isoformat(),
                    **(extra or {}),
                }
                self._save_pipeline_state(state)
        except Exception as e:
            self._log(f"  ⚠️ 状态更新失败 ({stage}={status}): {e}")
            return
        self._log(f"  📝 状态更新: {stage} = {status}")

    # =========================================================================
    # 1. 物理预平衡 (10 ns) → 保存轨迹 → 提取稳态坐标
    # =========================================================================
    def pre_equilibrate(
        self,
        n_steps: int = 5_000_000,  # 10ns @ 2fs
        save_traj: bool = True,
        platform_name: str = None,  # ✅ 默认使用实例配置的 platform
        resume: bool = False,
    ) -> Dict:
        """物理预平衡 - 【修复】默认使用 GPU，仅在生产采样前清理上下文"""
        traj_file = os.path.join(self.output_dir, "pre_equilibration.dcd")
        chk_file = os.path.join(self.checkpoint_dir, "pre_equil.chk")
        
        # ✅ 修复：默认使用实例配置的 platform（通常是 CUDA）
        equil_platform = platform_name or self.platform_name
        
        self._log(f"\n[阶段 0] 启动物理预平衡 (目标: {n_steps} 步 | Platform: {equil_platform})...")
        
        # 系统深拷贝 + 强制声明 Python 所有权
        sys_xml = XmlSerializer.serialize(self.system)
        equil_sys = XmlSerializer.deserialize(sys_xml)
        equil_sys.thisown = 1
        _ = equil_sys.getNumParticles()  # 触发底层指针验证，固化状态
        
        # 添加 Barostat（如果缺失）
        has_barostat = any(
            isinstance(f, openmm.MonteCarloBarostat) for f in equil_sys.getForces()
        )
        if not has_barostat:
            equil_sys.addForce(
                openmm.MonteCarloBarostat(self.pressure, self.temperature, 25)
            )
        
        # 创建 Integrator
        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature, 1.0 / unit.picosecond, 0.002 * unit.picosecond
        )
        
        # ✅ 修复：正确初始化 Platform，支持 CUDA
        try:
            resolved_platform, props = _build_platform_props(equil_platform)
            platform = openmm.Platform.getPlatformByName(resolved_platform)
        except Exception as e:
            self._log(f"  ⚠️ Platform '{equil_platform}' 初始化失败: {e}，回退到 CPU")
            platform = openmm.Platform.getPlatformByName("CPU")
            props = {}
            # ✅ 修复 2.2：仅在初始化失败时才降级平台，避免永久污染 self.platform_name
            self.platform_name = "CPU"
            equil_platform = "CPU"
        
        # 创建 Simulation
        simulation = app.Simulation(self.topology, equil_sys, integrator, platform, props)
        
        # Resume 逻辑
        resume_from_chk = False
        if resume and os.path.exists(chk_file):
            try:
                simulation.loadCheckpoint(chk_file)
                current_step = simulation.currentStep
                steps_remaining = max(0, n_steps - current_step)
                self._log(f"  ♻️ 从 Checkpoint 恢复 | 已完成: {current_step} | 剩余: {steps_remaining}")
                resume_from_chk = True
            except Exception as e:
                self._log(f"  ⚠️ Checkpoint 加载失败 ({e})，将重新开始")
                steps_remaining = n_steps
        else:
            simulation.context.setPositions(self.positions)
            if self.box_vectors is not None:
                simulation.context.setPeriodicBoxVectors(*self.box_vectors)
            self._log("  → 能量最小化...")
            openmm.LocalEnergyMinimizer.minimize(simulation.context, maxIterations=1000)
            steps_remaining = n_steps
        
        # 添加 Reporter
        if save_traj and steps_remaining > 0:
            simulation.reporters.append(
                app.DCDReporter(traj_file, 10000, append=resume_from_chk, enforcePeriodicBox=False)
            )
            simulation.reporters.append(app.CheckpointReporter(chk_file, 100000))
        
        # 运行模拟
        if steps_remaining > 0:
            self._log(f"  → 运行 {steps_remaining} 步 ({equil_platform})...")
            simulation.step(steps_remaining)
        
        # 提取稳态坐标
        state = simulation.context.getState(
            getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True
        )
        self.positions = state.getPositions()
        self.box_vectors = state.getPeriodicBoxVectors()
        final_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        
        # ✅ 关键修复：预平衡完成后，显式清理 CUDA 上下文（避免污染后续采样）
        if equil_platform.upper() == "CUDA":
            try:
                del simulation.context
                del integrator
                del equil_sys
                import gc; gc.collect()
                # 可选：重置 CUDA 上下文（需要 PyCUDA）
                # import pycuda.driver as cuda; cuda.Context.pop()
                self._log("  ✓ CUDA 上下文已清理，后续采样可安全复用 GPU")
            except Exception as e:
                self._log(f"  ⚠️ 上下文清理警告: {e}（通常不影响后续运行）")
        
        self._log(f"  ✓ 预平衡完成 | 最终势能: {final_energy:.2f} kJ/mol")
        
        self._update_stage_status(
            "equilibration",
            "completed",
            {
                "trajectory": traj_file if save_traj else None,
                "final_energy": final_energy,
                "total_steps": n_steps,
                "platform_used": equil_platform,
            },
        )

        return {
            "positions": self.positions,
            "box_vectors": self.box_vectors,
            "trajectory_file": traj_file if save_traj else None,
            "final_energy": final_energy,
            "resumed": resume_from_chk,
            "platform": equil_platform,
        }

    def fit_dexp_parameters(
        self,
        ligand_resname: str,
        top_file: str,
        output_name: str = "dexp_fitted_params.json",
        device: Optional[str] = None,
        n_frames: int = 200,
        env_radius_nm: float = 0.85,
        env_max_atoms: Optional[int] = None,
        fit_last_ns: Optional[float] = None,
        fit_r_min: float = 0.20,
        fit_r_max: float = 0.45,
        gmx_include_dir: Optional[str] = None,
    ) -> str:
        traj_file = os.path.join(self.output_dir, "pre_equilibration.dcd")
        if not _is_traj_valid(traj_file, min_frames=1):
            raise FileNotFoundError(
                f"未找到可用预平衡轨迹: {traj_file}。请先运行 pre_equilibrate(save_traj=True)。"
            )

        output_path = os.path.join(self.output_dir, output_name)
        platform_upper = str(self.platform_name).upper()
        resolved_device = device or ("cuda" if platform_upper == "CUDA" else "cpu")
        self._log(
            f"🧪 启动 DEXP 拟合 | device={resolved_device} | "
            f"frames={n_frames} | tail_ns={fit_last_ns} | env_radius={env_radius_nm}"
        )
        generated_path = run_orbv3_dexp_fitting(
            traj_file=traj_file,
            top_file=top_file,
            ligand_resname=ligand_resname,
            output_dir=self.output_dir,
            device=resolved_device,
            n_frames=n_frames,
            env_radius_nm=env_radius_nm,
            env_max_atoms=env_max_atoms,
            fit_last_ns=fit_last_ns,
            fit_r_min=fit_r_min,
            fit_r_max=fit_r_max,
            gmx_include_dir=gmx_include_dir,
        )
        if os.path.abspath(generated_path) != os.path.abspath(output_path):
            shutil.copy(generated_path, output_path)
        self._log(f"✅ DEXP 参数已生成: {output_path}")
        return output_path

    # =========================================================================
    # 1.5 带 Boresch 限制力的再平衡
    # =========================================================================
    def _rebalance_with_boresch(
        self,
        boresch_params: Dict,
        n_steps: int = 50_000,
        platform_name: Optional[str] = None,
        resume: bool = False,
    ) -> Dict:
        from abfe_core import LambdaDependentBoreschForce
        
        cleanup_temp_files(self.checkpoint_dir)
        self._log(f"🔄 启动带 Boresch 限制力的再平衡 ({n_steps} 步)...")
        chk_path = os.path.join(self.output_dir, "rebalance.chk")
        traj_path = os.path.join(self.output_dir, "rebalance_traj.dcd")
        state_path = os.path.join(self.output_dir, "rebalance_state.json")

        if resume and os.path.exists(state_path) and _is_checkpoint_valid(chk_path) and _is_traj_valid(traj_path, min_frames=1):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    rebalance_state = json.load(f)
                if rebalance_state.get("status") == "completed" and rebalance_state.get("n_steps") == int(n_steps):
                    self._log("  ♻️ 再平衡状态已完成，且 Checkpoint/轨迹有效，跳过重复再平衡。")
                    return {
                        "positions": self.positions,
                        "box_vectors": self.box_vectors,
                        "resumed": True,
                        "skipped": True,
                    }
            except Exception as e:
                self._log(f"  ⚠️ 再平衡完成态读取失败 ({e})，继续按 Checkpoint 续跑逻辑处理")

        # ✅ 新增：初始距离检查，防止拉力过载
        if _has_valid_boresch_restraint(boresch_params):
            import numpy as np
            from openmm import unit
            
            rec_idx = boresch_params["receptor_indices"]
            lig_idx = boresch_params["ligand_indices"]
            eq = boresch_params["equilibrium_values"]
            
            # ✅ 替换原有 pos_nm 转换逻辑 (约第 480 行)
            # 1. 强制转为 (N, 3) float64 numpy 数组，彻底杜绝 object 数组与索引报错
            if hasattr(self.positions, 'value_in_unit'):
                raw = self.positions.value_in_unit(unit.nanometer)
                # 处理 Quantity 包裹的 Vec3 列表或嵌套列表
                if hasattr(raw, '__iter__') and len(raw) > 0 and hasattr(raw[0], 'x'):
                    pos_nm = np.array([[p.x, p.y, p.z] for p in raw], dtype=np.float64)
                else:
                    pos_nm = np.asarray(raw, dtype=np.float64)
            elif isinstance(self.positions, (list, tuple)):
                if len(self.positions) == 0:
                    pos_nm = np.empty((0, 3), dtype=np.float64)
                elif hasattr(self.positions[0], 'x'):
                    pos_nm = np.array([[p.x, p.y, p.z] for p in self.positions], dtype=np.float64)
                else:
                    pos_nm = np.asarray(self.positions, dtype=np.float64)
            elif isinstance(self.positions, np.ndarray):
                pos_nm = self.positions.astype(np.float64, copy=False)
            else:
                raise TypeError(f"不支持的 positions 类型: {type(self.positions)}")

            # 2. 形状矫正 (防一维扁平数组)
            if pos_nm.ndim == 1:
                pos_nm = pos_nm.reshape(-1, 3)
            elif pos_nm.ndim != 2 or pos_nm.shape[1] != 3:
                raise ValueError(f"positions 形状异常: {pos_nm.shape}，期望 (N, 3)")

            # 3. 安全索引 (将 ligand_indices 转为 numpy 整数数组)
            lig_idx_arr = np.array(self.ligand_indices, dtype=int)
            lig_com = pos_nm[lig_idx_arr].mean(axis=0)
            box_nm = np.asarray([v.value_in_unit(unit.nanometer) for v in self.box_vectors], dtype=np.float64)
            box_center = 0.5 * np.sum(box_nm, axis=0)
            box_lengths = np.linalg.norm(box_nm, axis=1)
            if np.linalg.norm(lig_com - box_center) > 0.4 * np.min(box_lengths):
                self.positions, self.box_vectors = self._wrap_ligand_to_box(self.positions, self.box_vectors)
                self._log("  📦 检测到配体偏离主周期，已自动执行 PBC 居中")            
            # 计算实际距离 (H0-L0: 最近受体锚点 - 配体首锚点)
            H0 = pos_nm[rec_idx[0]]
            L0 = pos_nm[lig_idx[0]]
            actual_dist = np.linalg.norm(H0 - L0)
            target_r0 = eq.get("r0", 1.0)  # nm
            
            if abs(actual_dist - target_r0) > 0.15:
                self._log(f"  🔧 动态校正 Boresch r0: {target_r0*10:.2f}Å → {actual_dist*10:.2f}Å (防爬坡撕裂)")
                boresch_params["equilibrium_values"]["r0"] = float(actual_dist)
        # 1. 系统深拷贝 + 强制声明 Python 所有权
        sys_xml = XmlSerializer.serialize(self.system)
        rebal_sys = XmlSerializer.deserialize(sys_xml)
        rebal_sys.thisown = 1
        _ = rebal_sys.getNumParticles()  # 触发底层指针验证，固化状态
        
        # 添加 Boresch 限制力 (fixed_lam=1.0 全程开启)
        if _has_valid_boresch_restraint(boresch_params):
            rest_force = LambdaDependentBoreschForce(
                rec_idx=boresch_params["receptor_indices"],
                lig_idx=boresch_params["ligand_indices"],
                eq=boresch_params["equilibrium_values"],
                fc=boresch_params["force_constants"],
                fixed_lam=1.0,
                sign=1.0,
                use_pbc=True,
            )
            rest_force.setForceGroup(3)  # 与采样阶段一致
            rebal_sys.addForce(rest_force)
            self._log(f"  ✓ Boresch 限制力已注入 (Group 3)")
        
        # 2. 创建 Integrator + Platform
        integrator = openmm.LangevinMiddleIntegrator(
            self._ensure_temperature_quantity(self.temperature),
            1.0 / unit.picosecond,
            0.002 * unit.picosecond
        )
        equil_platform = platform_name or self.platform_name
        
        try:
            platform = openmm.Platform.getPlatformByName(equil_platform)
            _, props = _build_platform_props(equil_platform)
        except Exception as e:
            self._log(f"  ⚠️ Platform '{equil_platform}' 初始化失败: {e}，回退到 CPU")
            platform, props = openmm.Platform.getPlatformByName("CPU"), {}
            self.platform_name = "CPU"
            equil_platform = "CPU"
        
        # 3. 创建 Simulation
        simulation = app.Simulation(self.topology, rebal_sys, integrator, platform, props)
        
        # ✅ 挂载统一 Reporter
        dcd_path, log_path, _ = attach_simulation_reporters(
            simulation, "rebalance", self.output_dir,
            traj_interval=2000, energy_interval=500, chk_interval=5000,
            append_traj=resume and _is_checkpoint_valid(chk_path),
        )
        
        # ✅ 续跑逻辑
        resume_enabled = False
        if resume and _is_checkpoint_valid(chk_path):
            self._log(f"  ♻️ 检测到再平衡 Checkpoint ({chk_path})，恢复状态...")
            try:
                simulation.loadCheckpoint(chk_path)
                steps_remaining = max(0, n_steps - simulation.currentStep)
                resume_enabled = True
            except Exception as e:
                self._log(f"  ⚠️ Checkpoint 加载失败 ({e})，重新开始")
                steps_remaining = n_steps
        else:
            steps_remaining = n_steps
            
        # 如果不是续跑，才进行初始化和最小化
        if not resume_enabled:
            simulation.context.setPositions(self.positions)
            if self.box_vectors is not None:
                simulation.context.setPeriodicBoxVectors(*self.box_vectors)
            self._log("  → 能量最小化...")
            simulation.minimizeEnergy(maxIterations=1000)
        
        # 5. 运行
        if steps_remaining > 0:
            self._log(f"  → 运行 {steps_remaining} 步再平衡...")
            simulation.step(steps_remaining)
        
        # 5. 提取稳态坐标
        state = simulation.context.getState(getPositions=True, getVelocities=True)
        new_positions = state.getPositions()
        new_box = state.getPeriodicBoxVectors()
        
        # 6. 清理上下文（防止污染后续采样）
        if equil_platform.upper() == "CUDA":
            try:
                del simulation.context
                del integrator
                del rebal_sys
                import gc; gc.collect()
            except Exception as e:
                self._log(f"  ⚠️ 上下文清理警告: {e}")

        try:
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "status": "completed",
                        "n_steps": int(n_steps),
                        "timestamp": datetime.now().isoformat(),
                        "checkpoint": chk_path,
                        "trajectory": dcd_path,
                        "log": log_path,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            self._log(f"  ⚠️ 再平衡完成态保存失败: {e}")

        self._log(f"  ✓ 再平衡完成 | 坐标已更新")
        return {"positions": new_positions, "box_vectors": new_box}

    # =========================================================================
    # 1.5 辅助方法：PBC 居中与收敛监控
    # =========================================================================
    def _wrap_ligand_to_box(self, positions, box_vectors, margin_nm: float = 0.3) -> Tuple[list, list]:
        """仅做整体刚性平移，绝不逐原子 round 包裹，避免跨 PBC 分子被撕裂。"""
        # --- 1. 将任意输入统一转换为 (N,3) 纯数值 (nm) ---
        # 步骤 A: 提取裸数值并转为二维数组
        if hasattr(positions, 'value_in_unit'):
            raw = positions.value_in_unit(unit.nanometer)
            pos = np.asarray(raw, dtype=np.float64)
        elif isinstance(positions, (list, tuple)):
            if len(positions) == 0:
                return positions, box_vectors
            # 检查第一个元素类型
            first = positions[0]
            if hasattr(first, 'x'):          # OpenMM Vec3
                pos = np.array([[v.x, v.y, v.z] for v in positions], dtype=np.float64)
            else:
                # 普通列表、元组、数组等，直接转为 numpy
                pos = np.asarray(positions, dtype=np.float64)
        elif isinstance(positions, np.ndarray):
            pos = positions.astype(np.float64, copy=False)
        else:
            raise TypeError(f"不支持的 positions 类型: {type(positions)}")

        # 步骤 B: 确保形状为 (N, 3)
        if pos.ndim == 1:
            # 一维数组，可能是 [x1, y1, z1, x2, y2, z2, ...] 格式
            if pos.size % 3 != 0:
                raise ValueError(f"positions 元素数 {pos.size} 不是 3 的倍数")
            pos = pos.reshape(-1, 3)
        elif pos.ndim == 2:
            if pos.shape[1] != 3:
                # 可能是 (3, N) 转置为 (N, 3)
                if pos.shape[0] == 3:
                    pos = pos.T
                else:
                    raise ValueError(f"positions 二维形状必须为 (N,3)，实际: {pos.shape}")
        else:
            raise ValueError(f"positions 维度异常: {pos.shape}")

        # --- 2. 盒子向量处理，同样转换为 (3,3) numpy ---
        if hasattr(box_vectors, 'value_in_unit'):
            box = box_vectors.value_in_unit(unit.nanometer)
            box = np.asarray(box, dtype=np.float64)
        elif isinstance(box_vectors, (list, tuple)):
            if len(box_vectors) != 3:
                raise ValueError("box_vectors 必须包含 3 个向量")
            first = box_vectors[0]
            if hasattr(first, 'x'):
                box = np.array([[v.x, v.y, v.z] for v in box_vectors], dtype=np.float64)
            else:
                box = np.asarray(box_vectors, dtype=np.float64)
        elif isinstance(box_vectors, np.ndarray):
            box = box_vectors.astype(np.float64, copy=False)
        else:
            raise TypeError(f"不支持的 box_vectors 类型: {type(box_vectors)}")

        if box.shape != (3, 3):
            # 尝试重塑
            box = box.reshape(3, 3)
        box_center = 0.5 * (box[0] + box[1] + box[2])

        # --- 3. 仅按配体质心做整体平移 ---
        lig_pos = pos[self.ligand_indices]
        lig_com = np.mean(lig_pos, axis=0)
        shift_cart = box_center - lig_com
        pos_shifted = pos + shift_cart

        # --- 4. 转回 OpenMM 格式（Vec3 列表 + 单位）---
        new_pos = [openmm.Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_shifted] * unit.nanometer
        return new_pos, box_vectors

    def _check_equilibration_convergence(self, pressure_hist: np.ndarray, window: int = 500, tol_bar: float = 10.0) -> bool:
        """滑动窗口压力/密度波动检查"""
        if len(pressure_hist) < window: return False
        recent = pressure_hist[-window:]
        mean_p, std_p = np.mean(recent), np.std(recent)
        # 收敛判据：均值接近目标压力 且 波动 < 阈值
        target = self.pressure.value_in_unit(unit.bar) if hasattr(self.pressure, 'value_in_unit') else self.pressure
        return abs(mean_p - target) < tol_bar and std_p < tol_bar * 0.5

    # =========================================================================
    # 2. ACES 路径预优化
    # =========================================================================
    def run_preoptimization(
        self,
        decoupling_scheme: str = "dual_lambda",
        n_states: int = 12,
        n_steps_per_lambda: int = 5000,
        system_type: str = "complex",
        **kwargs,
    ) -> Dict:
        """调用预优化器生成自适应 Lambda 路径与窗口划分"""
        self._log(f"\n[预优化] 启动 ACES 路径优化 (方案: {decoupling_scheme})...")

        softcore_obj = ACESoftcorePotential.from_dict(
            ACESoftcorePotential.optimize_alpha(len(self.ligand_indices))
        )

        probe_sys = build_aces_probe_system(
            self.system, self.ligand_indices, softcore_obj
        )
        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature, 1.0 / unit.picosecond, 0.002 * unit.picosecond
        )
        try:
            platform = openmm.Platform.getPlatformByName("CUDA")
            props = {"Precision": "mixed"}
        except Exception as e:
            self._log(f"  ⚠️ CUDA 探针平台初始化失败: {e}，回退至 CPU")
            platform = openmm.Platform.getPlatformByName("CPU")  # ✅ 强制获取 CPU 平台对象
            props = {}

        context = openmm.Context(probe_sys, integrator, platform, props)
        context.setPositions(self.positions)
        if self.box_vectors is not None:
            context.setPeriodicBoxVectors(*self.box_vectors)
        context.setParameter("lam_coul", 1.0)
        context.setParameter("lam_vdw", 1.0)
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=500)

        optimizer = ABFEPreOptimizer(
            probe_sys,
            context,
            np.linspace(1.0, 0.0, max(12, n_states)),
            self.temperature.value_in_unit(unit.kelvin),
        )
        landscape = optimizer.analyze_gradient_and_optimize_path(
            n_steps_per_state=n_steps_per_lambda
        )
        opt_lambdas = optimizer.optimize_lambda_path_adaptive(
            landscape, target_n_states=n_states
        )

        # 窗口划分
        if hasattr(optimizer, "partition_ibs_windows_fixed"):
            window_ranges = optimizer.partition_ibs_windows_fixed(
                len(opt_lambdas), n_ib_windows=2, pts_per_window=6, overlap=2
            )
        else:
            window_ranges = [(0, len(opt_lambdas))]

        del context, integrator, probe_sys

        return {
            "lambdas": opt_lambdas,
            "window_ranges": window_ranges,
            "initial_weights": np.zeros(len(opt_lambdas)),
            "softcore_params": softcore_obj,
            "boresch_params": self._setup_boresch_params(system_type),
        }

    def _setup_boresch_params(self, system_type: str) -> Optional[Dict]:
        """占位：实际应由外部传入 Orb/传统 Boresch 参数字典"""
        return None  # 保持向后兼容，实际运行时通过 run_full_pipeline 注入

    # =========================================================================
    # 2.5 二面角修正 (在预平衡前应用)
    # =========================================================================
    def apply_torsion_corrections(self, torsion_params: Optional[Dict] = None):
        """
        在预平衡前应用二面角修正力
        支持两种格式：
        1. 傅里叶格式: parameters = [offset, c1, s1, c2, s2, ...]
        2. 传统格式: k, n, phi0
        """
        if not torsion_params:
            return

        self._log("🔧 应用二面角修正力...")

        fmt = torsion_params.get("format", "traditional")
        torsions = (
            torsion_params
            if isinstance(torsion_params, list)
            else torsion_params.get("torsions", [])
        )

        if fmt == "fourier":
            self._apply_fourier_torsions(torsions)
        else:
            self._apply_traditional_torsions(torsions)

    def _apply_fourier_torsions(self, torsions: List[Dict]):
        """应用傅里叶级数格式的二面角修正"""
        from openmm import CustomTorsionForce

        max_order = 0
        for t in torsions:
            params = t.get("parameters", [])
            order = (len(params) - 1) // 2
            max_order = max(max_order, order)

        if max_order == 0:
            self._log("  ⚠️ 无有效的傅里叶二面角参数")
            return

        terms = ["offset"]
        for n in range(1, max_order + 1):
            terms.append(f"c{n}*cos({n}*theta)")
            terms.append(f"s{n}*sin({n}*theta)")

        expr = " + ".join(terms)
        self._log(f"  📐 傅里叶表达式 (阶数={max_order}): {expr}")

        force = CustomTorsionForce(expr)
        force.addPerTorsionParameter("offset")
        for n in range(1, max_order + 1):
            force.addPerTorsionParameter(f"c{n}")
            force.addPerTorsionParameter(f"s{n}")

        applied = 0
        for t in torsions:
            indices = t.get("indices", [])
            if len(indices) != 4:
                continue

            params = t.get("parameters", [])
            if len(params) < 1:
                continue

            param_values = [params[0]]
            for n in range(1, max_order + 1):
                c_idx = 2 * n - 1
                s_idx = 2 * n
                c_val = params[c_idx] if c_idx < len(params) else 0.0
                s_val = params[s_idx] if s_idx < len(params) else 0.0
                param_values.extend([c_val, s_val])

            force.addTorsion(
                indices[0], indices[1], indices[2], indices[3], param_values
            )
            applied += 1

        if applied > 0:
            self.system.addForce(force)
            self._log(
                f"  ✓ 已添加 {applied} 个傅里叶二面角修正项 (最大阶数={max_order})"
            )
        else:
            self._log("  ⚠️ 无有效的二面角修正项")

    def _apply_traditional_torsions(self, torsions: List[Dict]):
        """应用传统 k/n/phi 格式的二面角修正"""
        from openmm import CustomTorsionForce

        force = CustomTorsionForce("k * (1 + cos(n*theta - phi0))")
        force.addPerTorsionParameter("k")
        force.addPerTorsionParameter("n")
        force.addPerTorsionParameter("phi0")

        applied = 0
        for t in torsions:
            indices = t.get("indices", [])
            if len(indices) != 4:
                continue

            k = t.get("k", 0.0)
            n = t.get("n", 1)
            phi0 = t.get("phi0", 0.0)
            is_degrees = t.get("phi0_in_degrees", True)
            phi0_rad = np.radians(phi0) if is_degrees else phi0

            force.addTorsion(
                indices[0], indices[1], indices[2], indices[3], [k, n, phi0_rad]
            )
            applied += 1

        if applied > 0:
            self.system.addForce(force)
            self._log(f"  ✓ 已添加 {applied} 个传统二面角修正项")
        else:
            self._log("  ⚠️ 无有效的二面角修正项")

    # =========================================================================
    # 3. IBS 生产采样
    # =========================================================================


    # =========================================================================
    # 4. 双λ解耦专用路由
    # =========================================================================
    # ================= abfe_pipeline.py -> _run_dual_lambda_optimization =================
    # ================= abfe_pipeline.py =================
    # 替换原 _run_dual_lambda_optimization 方法
    def _run_dual_lambda_optimization(
        self,
        stage_name: str,
        n_states: int = 12,
        n_steps_per_state: int = 10000,
        potential_type: str = "softcore",
    ) -> Dict:
        from abfe_preoptimizer import DualLambdaPreOptimizer
        
        self._log(f"\n[PIPELINE] 开始优化 {stage_name} 阶段...")
        softcore_obj = ACESoftcorePotential.from_dict(ACESoftcorePotential.optimize_alpha(len(self.ligand_indices)))

        if stage_name == "decharging" or str(potential_type).lower() != "softcore":
            if str(potential_type).lower() != "softcore":
                self._log(
                    f"[PIPELINE] {stage_name} 预优化已禁用：当前 potential_type={potential_type}，"
                    "避免用 ACES-softcore 探针优化非 softcore 生产 Hamiltonian，改用线性 λ 路径。"
                )
            else:
                self._log(
                    "[PIPELINE] Stage 1 去电荷预优化已强制禁用自适应 Pathfinding："
                    "Cutoff 型 CustomNonbonded 探针无法保真 PME 长程静电，直接回退线性 λ 路径。"
                )
            self._log(
                f"[PIPELINE] {stage_name} 使用线性 λ 路径 ({n_states} 状态)。"
            )
            opt_n_states = int(n_states)
            return {
                "lambdas_var": np.linspace(1.0, 0.0, opt_n_states).tolist(),
                "n_states": opt_n_states,
                "window_ranges": generate_overlapping_windows(
                    n_states=opt_n_states,
                    n_windows=None,
                    pts_per_window=6,
                    overlap=2,
                ),
                "softcore_params": softcore_obj,
                "path_optimization_disabled_reason": (
                    "pme_decharging_probe_disabled"
                    if stage_name == "decharging"
                    else f"unsupported_probe_for_{potential_type}"
                ),
            }
        
        probe_sys = build_aces_probe_system_dual_lambda(self.system, self.ligand_indices, softcore_obj, fixed_lam_coul=0.0, fixed_lam_vdw=1.0)

        integrator = openmm.LangevinMiddleIntegrator(self.temperature, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
        try:
            platform = openmm.Platform.getPlatformByName("CUDA")
            props = {"Precision": "mixed"}
        except Exception as e:
            self._log(f"  ⚠️ CUDA 优化平台初始化失败: {e}，回退至 CPU")
            platform = openmm.Platform.getPlatformByName("CPU")  # ✅ 修复：避免传入 None
            props = {}

        self._log(f"[CONTEXT] 正在创建 Context...")
        context = openmm.Context(probe_sys, integrator, platform, props)
        context.setPositions(self.positions)
        if self.box_vectors is not None:
            context.setPeriodicBoxVectors(*self.box_vectors)
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=500)
        
        self._log(f"[CONTEXT] Context 创建完成。执行 context.getParameters()...")
        probe_params = context.getParameters()
        self._log(f"[CONTEXT] 返回值类型: {type(probe_params)}")
        self._log(f"[CONTEXT] 实际参数字典: {dict(probe_params)}")
        
        # 🔑 强制注入测试
        required = {"lam_coul": 1.0 if stage_name=="decharging" else 0.0, "lam_vdw": 1.0}
        injected = {}
        for p_name, p_val in required.items():
            if p_name not in probe_params:
                try:
                    context.setParameter(p_name, float(p_val))
                    injected[p_name] = True
                    self._log(f"[CONTEXT] ✅ 强制注入成功: {p_name}={p_val}")
                except Exception as e:
                    injected[p_name] = False
                    self._log(f"[CONTEXT] ❌ 强制注入失败 {p_name} | 报错: {e}")
        
        optimizer = DualLambdaPreOptimizer(probe_sys, context, self.temperature.value_in_unit(unit.kelvin))
        
        self._log(f"[OPTIMIZER] 初始化完成。param_coul={optimizer.param_coul}, param_vdw={optimizer.param_vdw}")
        
        try:
            if stage_name == "decharging":
                opt_res = optimizer.optimize_stage1_decharging(n_states=n_states, n_steps_per_state=n_steps_per_state)
            else:
                opt_res = optimizer.optimize_stage2_vanishing(n_states=n_states, n_steps_per_state=n_steps_per_state)
            self._log(f"[OUTPUT] 优化成功返回: keys={list(opt_res.keys())}")
        except Exception as e:
            self._log(f"[OUTPUT] 优化器异常捕获: {e}")
            try:
                import traceback
                self._log(traceback.format_exc())
            except Exception:
                pass
            opt_res = {
                "stage": stage_name,
                "lambdas_coul": np.linspace(1.0, 0.0, n_states).tolist() if stage_name=="decharging" else [0.0]*n_states,
                "lambdas_vdw": np.linspace(1.0, 0.0, n_states).tolist() if stage_name=="vanishing" else [1.0]*n_states,
                "n_states": n_states
            }
            self._log(f"[OUTPUT] 降级返回线性路径: {opt_res['n_states']} 状态")
        
        opt_n_states = int(opt_res["n_states"])
        opt_window_ranges = generate_overlapping_windows(
            n_states=opt_n_states,
            n_windows=None,
            pts_per_window=6,
            overlap=2,
        )

        del context, integrator, probe_sys
        return {
            "lambdas_var": opt_res["lambdas_coul"] if stage_name=="decharging" else opt_res["lambdas_vdw"],
            "n_states": opt_res["n_states"],
            "window_ranges": opt_window_ranges,
            "softcore_params": softcore_obj,
        }

    def _refine_lambda_path_with_medium_probe(
        self,
        stage_name: str,
        fixed_lam_coul: float,
        fixed_lam_vdw: float,
        lambdas_var: List[float],
        window_ranges: List[Tuple[int, int]],
        preopt_path: str,
        potential_type: str,
        dexp_params: Optional[Dict],
        boresch_params: Optional[Dict],
        refine_n_steps_per_window: int,
        refine_steps_per_update: int,
        max_window_span_kJ: float,
        overlap: int,
        resume: bool = False,
    ) -> Tuple[List[float], List[Tuple[int, int]]]:
        """
        用"中等步数"探针（比粗扫 optimize_stageN 贵、比正式生产便宜得多）在独立
        scratch 目录里把当前 λ 路径实采一遍，基于真实测得的 f(λ) 曲线精修 λ 分布
        与窗口边界，写回 preopt_path。

        scratch 目录必须与正式生产的 stage_output_dir 完全隔离：
        IBSWindowManagerDualLambda.run_all_windows 的 resume 断点续传只按"能量数组
        形状是否匹配当前窗口"判断是否跳过采样，不检查实际步数/样本量是否够——如果
        中等步数探针直接写进生产目录，后续生产阶段会误把这些样本量不足的数据当成
        "已采样完成"而跳过，真正的生产步数永远不会被执行。
        """
        n_states = len(lambdas_var)
        lambdas_fix = [fixed_lam_vdw if stage_name == "decharging" else fixed_lam_coul] * n_states
        stage_type = "coul" if stage_name == "decharging" else "vdw"

        scratch_dir = os.path.join(self.output_dir, f"{stage_name}_refine_probe")
        os.makedirs(scratch_dir, exist_ok=True)

        alchemical_params = _resolve_alchemical_params(
            potential_type, dexp_params, self.ligand_indices
        )
        manager = IBSWindowManagerDualLambda(
            system_template=self.system,
            topology=self.topology,
            perturbed_atom_indices=self.ligand_indices,
            lambdas_coul=lambdas_var if stage_name == "decharging" else lambdas_fix,
            lambdas_vdw=lambdas_fix if stage_name == "decharging" else lambdas_var,
            temperature=self.temperature,
            window_ranges=window_ranges,
            alchemical_params=alchemical_params,
            potential_type=potential_type,
            restraint_params=boresch_params,
            prefix="abfe_dual_refine_probe",
            platform_name=self.platform_name,
            output_dir=scratch_dir,
            checkpoint_dir=self.checkpoint_dir,
        )
        manager.output_dir = scratch_dir

        self._log(
            f"  🔬 [精修探针] {stage_name}: 中等步数采样 "
            f"({refine_n_steps_per_window} 步/窗口，独立 scratch 目录，不影响生产数据)..."
        )
        manager.run_all_windows(
            positions=self.positions,
            box_vectors=self.box_vectors,
            n_steps_per_window=refine_n_steps_per_window,
            steps_per_update=refine_steps_per_update,
            stage_type=stage_type,
            resume=resume,
        )

        result = refine_stage_lambda_path_from_data(
            stage_dir=scratch_dir,
            preopt_path=preopt_path,
            temperature_k=self.temperature.value_in_unit(unit.kelvin),
            n_states=n_states,
            max_window_span_kJ=max_window_span_kJ,
            overlap=overlap,
            stage_type=stage_type,
        )
        self._log(
            f"  ✅ [精修探针] {stage_name} λ 路径已按实测 |Δf| 精修："
            f"{result['n_states']} 个状态，{len(result['window_ranges'])} 个窗口"
        )
        return result["lambdas_var"], [tuple(r) for r in result["window_ranges"]]

    # ================= abfe_pipeline.py =================
    # 替换 _run_dual_lambda_stage 方法
    def _run_dual_lambda_stage(
        self,
        stage_name: str,
        fixed_lam_coul: float,
        fixed_lam_vdw: float,
        n_states: int,
        n_steps_per_window: int,
        steps_per_update: int,
        system_type: str,
        resume: bool,
        potential_type: str = "softcore",
        dexp_params: Optional[Dict] = None,
        optimized_lambdas: Optional[List[float]] = None,
        window_ranges: Optional[List[Tuple[int, int]]] = None,
        enable_early_stop: bool = False,
        boresch_params: Optional[Dict] = None,
        enable_gradual_warmup: bool = True,
        warmup_steps: int = 500000,
        parallel: bool = True,
        device_indices: Optional[list] = None,
        n_workers: int = None,
        **kwargs,
    ) -> Dict:
        """
        执行单个双λ阶段 (去电荷 或 去VDW)
        职责：路由采样 -> 获取结果
        """
        self._log(f"\n{'=' * 60}")
        self._log(f"[双λ阶段] {stage_name.upper()} | λ_coul={fixed_lam_coul} | λ_vdw={fixed_lam_vdw}")
        self._log(f"{'=' * 60}")

        # 1. 确定 Lambda 路径
        if optimized_lambdas is not None:
            lambdas_var = optimized_lambdas
            self._log(f"  ✓ 使用自适应优化 Lambda 路径 ({len(lambdas_var)} 个状态)")
        else:
            lambdas_var = np.linspace(1.0, 0.0, n_states).tolist()
            self._log(f"  ⚠️ 使用线性 Lambda 路径 ({n_states} 个状态)")

        n_states = len(lambdas_var)
        # 固定另一个 Lambda
        lambdas_fix = [
            fixed_lam_vdw if stage_name == "decharging" else fixed_lam_coul
        ] * n_states

        if stage_name == "decharging":
            self._log(
                "  ⚠️ Coulomb 去电荷阶段已禁用 IBS-CustomNonbondedForce；"
                "改用 NonbondedForce ParameterOffset 路径以保留 PME 长程静电。"
            )
            stage_output_dir = os.path.join(self.output_dir, stage_name)
            os.makedirs(stage_output_dir, exist_ok=True)
            lambdas_coul = lambdas_var
            lambdas_vdw = lambdas_fix
            temp_k = self.temperature.value_in_unit(unit.kelvin)
            traj_files = _expected_remd_traj_files(stage_output_dir, stage_name, len(lambdas_coul))
            u_kn_path = os.path.join(stage_output_dir, f"{stage_name}_pme_u_kn.npy")
            n_k_path = u_kn_path + ".n_k.npy"
            if resume and os.path.exists(u_kn_path) and _is_pme_u_kn_cache_compatible(
                stage_output_dir,
                stage_name,
                n_states,
                lambdas_coul,
                lambdas_vdw,
                temp_k,
                self.system,
                self.topology,
                self.ligand_indices,
                boresch_params,
            ):
                self._log("  ♻️ 检测到已有 PME u_kn，跳过 REMD 采样与重算，直接求解 MBAR")
                u_kn = np.load(u_kn_path)
                analyzer = TraditionalMBARAnalyzer(temperature=temp_k)
                if not os.path.exists(n_k_path):
                    raise RuntimeError(f"PME u_kn 缓存缺少样本数 sidecar: {n_k_path}")
                analyzer._last_n_k = np.load(n_k_path)
                res = analyzer.solve(u_kn)
                return {
                    "stage": stage_name,
                    "total_delta_G": float(res.get("delta_G", 0.0)),
                    "total_error": float(res.get("error", 0.0)),
                    "method": "PME-REMD-MBAR",
                    "n_states": int(n_states),
                    "lambda_endpoint_diagnostics": lambda_endpoint_diagnostics(lambdas_coul, lambdas_vdw),
                    "diagnostics": res.get("diagnostics", {}),
                }
            elif resume and os.path.exists(u_kn_path):
                self._log("  ♻️ 检测到旧版 PME u_kn 缓存，但模型版本不兼容；保留轨迹并重新执行离线 MBAR 重算。")

            expected_frames = max(1, _expected_remd_frame_count(n_steps_per_window))
            if resume and _all_remd_trajs_valid(
                stage_output_dir,
                stage_name,
                len(lambdas_coul),
                min_frames=expected_frames,
            ):
                self._log("  ♻️ 检测到完整 REMD DCD，视为采样已完成，跳过 REMD 继续离线 MBAR")
            else:
                remd = REMDManager(
                    system_template=self.system,
                    topology=self.topology,
                    positions=self.positions,
                    box_vectors=self.box_vectors,
                    ligand_indices=self.ligand_indices,
                    lambdas_coul=lambdas_coul,
                    lambdas_vdw=lambdas_vdw,
                    temperature=temp_k,
                    platform_name=self.platform_name,
                    output_dir=stage_output_dir,
                    boresch_params=boresch_params,
                )
                traj_files = remd.run(
                    n_steps=n_steps_per_window,
                    exchange_interval=max(1, int(steps_per_update)),
                    stage_name=stage_name,
                )

            analyzer = TraditionalMBARAnalyzer(temperature=temp_k)
            u_kn = analyzer.compute_u_kn(
                traj_files=traj_files,
                system_template=self.system,
                ligand_indices=self.ligand_indices,
                lambdas_coul=lambdas_coul,
                lambdas_vdw=lambdas_vdw,
                platform_name="CPU",
                topology=self.topology,
                reference_positions=self.positions,
                reference_box_vectors=self.box_vectors,
                boresch_params=boresch_params,
            )
            np.save(u_kn_path, u_kn)
            np.save(n_k_path, analyzer._last_n_k)
            _write_pme_u_kn_meta(
                stage_output_dir,
                stage_name,
                n_states,
                lambdas_coul,
                lambdas_vdw,
                temp_k,
                self.system,
                self.topology,
                self.ligand_indices,
                boresch_params,
            )
            res = analyzer.solve(u_kn)
            return {
                "stage": stage_name,
                "total_delta_G": float(res.get("delta_G", 0.0)),
                "total_error": float(res.get("error", 0.0)),
                "method": "PME-REMD-MBAR",
                "n_states": int(n_states),
                "lambda_endpoint_diagnostics": lambda_endpoint_diagnostics(lambdas_coul, lambdas_vdw),
                "diagnostics": res.get("diagnostics", {}),
            }

        # 2. 划分窗口
        # 🔑 支持显式传入 window_ranges（例如手动在某个窗口内插入额外 λ 点做局部
        # 加密时使用）：此前这里无条件调用 generate_overlapping_windows 重新自动
        # 划分，即使调用方（或 preopt 缓存）里已经带着 window_ranges 也会被忽略，
        # 导致"手动往某个窗口塞点"这种局部修改在实际采样时完全不生效——重新划分
        # 出来的窗口边界跟手动想要的边界（比如只让窗口0变大、其余窗口原样保留）
        # 完全不是一回事。校验：覆盖必须正好是 [0, n_states) 且按窗口起点排序。
        if window_ranges is not None:
            covered = sorted({idx for s, e in window_ranges for idx in range(s, e)})
            if covered != list(range(n_states)):
                raise RuntimeError(
                    f"显式传入的 window_ranges 覆盖范围与 n_states={n_states} 不匹配"
                    f"（覆盖 {covered[:3]}...{covered[-3:] if covered else []}，"
                    f"共 {len(covered)} 个索引），拒绝使用可能导致漏采样/越界的窗口划分。"
                )
            self._log(f"  🪟 使用显式指定的 {len(window_ranges)} 个 IBS 窗口边界: {window_ranges}")
        else:
            from abfe_preoptimizer import generate_overlapping_windows
            pts_per_window, overlap = 6, 2
            window_ranges = generate_overlapping_windows(
                n_states=n_states,
                n_windows=kwargs.get("n_windows", None),
                pts_per_window=pts_per_window,
                overlap=overlap
            )
            self._log(f"  🪟 自动划分 {len(window_ranges)} 个 IBS 窗口")

        # 3. 初始化 Manager
        stage_output_dir = os.path.join(self.output_dir, stage_name)
        os.makedirs(stage_output_dir, exist_ok=True)
        
        stage_type = "coul" if stage_name == "decharging" else "vdw"
        
        alchemical_params = _resolve_alchemical_params(
            potential_type, dexp_params, self.ligand_indices
        )
        manager = IBSWindowManagerDualLambda(
            system_template=self.system,
            topology=self.topology,
            perturbed_atom_indices=self.ligand_indices,
            lambdas_coul=lambdas_var if stage_name == "decharging" else lambdas_fix,
            lambdas_vdw=lambdas_fix if stage_name == "decharging" else lambdas_var,
            temperature=self.temperature,
            window_ranges=window_ranges,
            alchemical_params=alchemical_params,
            potential_type=potential_type,
            restraint_params=boresch_params,
            prefix="abfe_dual",
            platform_name=self.platform_name,
            output_dir=stage_output_dir,
            checkpoint_dir=self.checkpoint_dir,
        )
        
        # 🔑 关键：设置输出目录，确保 combine_results 能找到文件
        manager.output_dir = stage_output_dir

        # 4. 运行采样
        manager.run_all_windows(
            positions=self.positions,
            box_vectors=self.box_vectors,
            n_steps_per_window=n_steps_per_window,
            steps_per_update=steps_per_update,
            stage_type=stage_type,
            resume=resume,
            enable_gradual_warmup=enable_gradual_warmup,
            warmup_steps=warmup_steps,
        )


        # ✅【替换旧分析逻辑】直接调用 TMBAR 全局求解器
        # ✅ 动态计算 kT，避免 self.kt 未初始化报错
        kt_val = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(unit.kilojoule_per_mole)
        
        # ✅ 导入修复后的函数
        from ibs_engine import solve_stage_integrated
        
        window_outputs = manager.get_stage_data_for_analysis(stage_type=stage_type)
        if not window_outputs:
            raise RuntimeError(
                f"{stage_name} 阶段未找到任何窗口能量文件，无法执行全局 TMBAR。"
                "这通常意味着窗口落盘失败或输出目录异常。"
            )

        stage_result = solve_stage_integrated(
            window_outputs=window_outputs,
            kt=kt_val,
            stage_name=stage_name
        )
        if stage_result.get("error"):
            raise RuntimeError(
                f"{stage_name} 阶段全局 TMBAR 失败: {stage_result['error']}"
            )
        stage_result.setdefault("stage", stage_name)
        stage_result.setdefault("n_states", int(n_states))
        stage_result["lambda_endpoint_diagnostics"] = lambda_endpoint_diagnostics(
            manager.lambdas_coul,
            manager.lambdas_vdw,
        )
        stage_result.setdefault("diagnostics", {})
        stage_result["diagnostics"].update({
            "method": stage_result.get("method"),
            "min_overlap_proxy": stage_result.get("min_overlap"),
            "offset_error_contribution": stage_result.get("offset_error_contribution"),
            "uncertainty_note": stage_result.get("uncertainty_note"),
        })
        return stage_result

    # =========================================================================
    # 4.5 2D λ 路径采样 (对角线 / 测地线)
    # =========================================================================
    def _run_2d_lambda_stage(
        self,
        path_2d: List[Tuple[float, float]],
        label: str = "2d",
        n_steps_per_window: int = 50000,
        steps_per_update: int = 500,
        system_type: str = "complex",
        resume: bool = False,
        potential_type: str = "softcore",
        dexp_params: Optional[Dict] = None,
        enable_early_stop: bool = False,
        boresch_params: Optional[Dict] = None,
        enable_gradual_warmup: bool = True,
        warmup_steps: int = 500000,
        **kwargs,
    ) -> Dict:
        """
        执行 2D λ 路径采样 (λ_coul, λ_vdw 同时变化)
        接收预计算的 path_2d = [(lc0,lv0), (lc1,lv1), ...]
        """
        n_states = len(path_2d)
        lambdas_coul = [p[0] for p in path_2d]
        lambdas_vdw = [p[1] for p in path_2d]
        if potential_type == "dexp":
            raise NotImplementedError(
                "single_lambda / 2D 的 PME-REMD 路径当前尚未实现 DEXP Hamiltonian；"
                "请改用 IBS dual_lambda + dexp，或先切回 softcore。"
            )
        self._log(f"\n{'=' * 60}")
        self._log(f"[2D 路径] {label} | {n_states} 个状态")
        self._log(f"  λ_coul: {lambdas_coul[0]:.3f} → {lambdas_coul[-1]:.3f}")
        self._log(f"  λ_vdw:  {lambdas_vdw[0]:.3f} → {lambdas_vdw[-1]:.3f}")
        self._log(f"{'=' * 60}")

        stage_output_dir = os.path.join(self.output_dir, label)
        os.makedirs(stage_output_dir, exist_ok=True)

        traj_files = _expected_remd_traj_files(stage_output_dir, label, n_states)
        u_kn_path = os.path.join(stage_output_dir, f"{label}_pme_u_kn.npy")
        n_k_path = u_kn_path + ".n_k.npy"
        temp_k = self.temperature.value_in_unit(unit.kelvin)
        if resume and os.path.exists(u_kn_path) and _is_pme_u_kn_cache_compatible(
            stage_output_dir,
            label,
            n_states,
            lambdas_coul,
            lambdas_vdw,
            temp_k,
            self.system,
            self.topology,
            self.ligand_indices,
            boresch_params,
        ):
            self._log("  ♻️ 检测到兼容的 PME u_kn 缓存，直接求解 MBAR")
            u_kn = np.load(u_kn_path)
            analyzer = TraditionalMBARAnalyzer(
                temperature=self.temperature.value_in_unit(unit.kelvin)
            )
            if not os.path.exists(n_k_path):
                raise RuntimeError(f"PME u_kn 缓存缺少样本数 sidecar: {n_k_path}")
            analyzer._last_n_k = np.load(n_k_path)
            res = analyzer.solve(u_kn)
        else:
            expected_frames = max(1, _expected_remd_frame_count(n_steps_per_window))
            if resume and _all_remd_trajs_valid(
                stage_output_dir, label, n_states, min_frames=expected_frames
            ):
                self._log("  ♻️ 检测到完整 REMD 轨迹，跳过采样直接重算 u_kn")
            else:
                self._log("  ⚡ 2D/单λ 路径改走 PME-preserving REMD+MBAR 通路")
                remd = REMDManager(
                    system_template=self.system,
                    topology=self.topology,
                    positions=self.positions,
                    box_vectors=self.box_vectors,
                    ligand_indices=self.ligand_indices,
                    lambdas_coul=lambdas_coul,
                    lambdas_vdw=lambdas_vdw,
                    temperature=temp_k,
                    platform_name=self.platform_name,
                    output_dir=stage_output_dir,
                    boresch_params=boresch_params,
                )
                traj_files = remd.run(
                    n_steps=n_steps_per_window,
                    exchange_interval=max(1, int(steps_per_update)),
                    stage_name=label,
                )

            analyzer = TraditionalMBARAnalyzer(
                temperature=temp_k
            )
            u_kn = analyzer.compute_u_kn(
                traj_files=traj_files,
                system_template=self.system,
                ligand_indices=self.ligand_indices,
                lambdas_coul=lambdas_coul,
                lambdas_vdw=lambdas_vdw,
                platform_name="CPU",
                topology=self.topology,
                reference_positions=self.positions,
                reference_box_vectors=self.box_vectors,
                boresch_params=boresch_params,
            )
            np.save(u_kn_path, u_kn)
            np.save(n_k_path, analyzer._last_n_k)
            _write_pme_u_kn_meta(
                stage_output_dir,
                label,
                n_states,
                lambdas_coul,
                lambdas_vdw,
                temp_k,
                self.system,
                self.topology,
                self.ligand_indices,
                boresch_params,
            )
            res = analyzer.solve(u_kn)

        stage_result = {
            "stage": label,
            "total_delta_G": float(res.get("delta_G", 0.0)),
            "total_error": float(res.get("error", 0.0)),
            "method": "PME-REMD-MBAR",
            "n_states": int(n_states),
            "lambda_path": [list(map(float, p)) for p in path_2d],
            "lambda_endpoint_diagnostics": lambda_endpoint_diagnostics(lambdas_coul, lambdas_vdw),
            "diagnostics": res.get("diagnostics", {}),
        }
        self._log(
            f"  ✓ {label} 路径完成: ΔG={stage_result['total_delta_G']:.2f} ± "
            f"{stage_result['total_error']:.2f} kJ/mol"
        )
        return stage_result

    # =========================================================================
    # 5. Boresch 修正与结果聚合
    # =========================================================================
    # === 替换 apply_boresch_correction ===
    # === 替换 apply_boresch_correction ===
    @staticmethod
    def _strip_unit_suffix(key: str, target_keys: Dict[str, str]) -> Optional[str]:
        """智能剥离单位后缀"""
        if key in target_keys: return target_keys[key]
        # 移除常见后缀并匹配
        suffixes = ["_kJ_mol_nm2", "_kJ_mol_rad2", "_nm", "_rad", "_deg"]
        for suffix in suffixes:
            if key.endswith(suffix):
                base = key[:-len(suffix)]
                if base in target_keys.values(): return base
        return None


    def apply_boresch_correction(
        self,
        boresch_params: Optional[Dict] = None,
        autoload_from_disk: bool = True,
    ) -> Dict:
        """🔑 增强版：支持磁盘自动加载 + 严格单位清洗 + 异常不静默吞没"""
        if boresch_params is None:
            boresch_path = os.path.join(self.output_dir, "boresch_params.json")
            if autoload_from_disk and os.path.exists(boresch_path):
                self._log(f"  📂 参数未传入，自动从磁盘加载: {boresch_path}")
                with open(boresch_path, "r") as f:
                    boresch_params = json.load(f)
            else:
                raise RuntimeError("未提供 Boresch 参数且未找到缓存文件；拒绝以 0.0 kJ/mol 修正继续生产 ABFE。")
                
        # 1. 兼容嵌套结构提取
        fc = boresch_params.get("force_constants")
        eq = boresch_params.get("equilibrium_values")
        if not fc or not eq:
            # 尝试解包嵌套层 (兼容 Auto/Orb 输出格式)
            anchors = boresch_params.get("boresch_anchors", boresch_params)
            fc = anchors.get("force_constants", {})
            eq = anchors.get("equilibrium_values", {})
            
        if not fc or not eq:
            raise RuntimeError("Boresch 参数字典结构异常：缺失 force_constants 或 equilibrium_values。")
            
        # 2. 智能剥离单位后缀
        fc_targets = {
            "kr": "kr",
            "kthetaA": "kthetaA",
            "kthetaB": "kthetaB",
            "kphiA": "kphiA",
            "kphiB": "kphiB",
            "kphiC": "kphiC",
        }
        eq_targets = {
            "r0": "r0",
            "thetaA0": "thetaA0",
            "thetaB0": "thetaB0",
            "phiA0": "phiA0",
            "phiB0": "phiB0",
            "phiC0": "phiC0",
        }
        fc_norm = {}
        for k, v in fc.items():
            clean_k = self._strip_unit_suffix(str(k), fc_targets) or str(k)
            fc_norm[clean_k] = v
        eq_norm = {}
        for k, v in eq.items():
            clean_k = self._strip_unit_suffix(str(k), eq_targets) or str(k)
            eq_norm[clean_k] = float(v)
        fc_norm = {k: float(v) for k, v in fc_norm.items()}

        required_eq = ("r0", "thetaA0", "thetaB0", "phiA0", "phiB0", "phiC0")
        required_fc = ("kr", "kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC")
        missing_eq = [k for k in required_eq if k not in eq_norm]
        missing_fc = [k for k in required_fc if k not in fc_norm]
        if missing_eq or missing_fc:
            raise RuntimeError(
                "Boresch 参数缺失必要字段："
                f"equilibrium missing={missing_eq}, force_constants missing={missing_fc}"
            )
        if not np.all(np.isfinite([eq_norm[k] for k in required_eq] + [fc_norm[k] for k in required_fc])):
            raise RuntimeError("Boresch 参数包含 NaN/Inf；拒绝计算解析修正。")
            
        # 3. 防御性拦截
        kr_val = fc_norm.get("kr", 0)
        if kr_val <= 0:
            raise RuntimeError(f"Boresch kr={kr_val} 非正；拒绝替换为默认力常数继续。")

        thA_val = float(eq_norm.get("thetaA0", 1.5708))
        thB_val = float(eq_norm.get("thetaB0", 1.5708))
        sin_guard = min(abs(np.sin(thA_val)), abs(np.sin(thB_val)))
        if sin_guard < 0.1:
            raise RuntimeError(
                f"Boresch 平衡角接近奇点: "
                f"θA={np.degrees(thA_val):.2f}°, θB={np.degrees(thB_val):.2f}° "
                f"(min|sinθ|={sin_guard:.4f})；拒绝以 0.0 kJ/mol 修正继续。"
            )
            
        # 4. 计算“restrained decoupling → 标准态释放”的修正项；失败必须中止。
        delta_g = calculate_boresch_analytical_correction(eq=eq_norm, fc=fc_norm, T=self.temperature)
        self._log(f"[Boresch] 标准态释放修正: {delta_g:.3f} kJ/mol ({delta_g/4.184:.3f} kcal/mol)")

        # ✅ 唯一出口：参数落盘 + 返回
        boresch_json = UnitFormatter.format_boresch_json(boresch_params)
        boresch_json_path = os.path.join(self.output_dir, "boresch_params.json")
        with open(boresch_json_path, "w") as f:
            json.dump(boresch_json, f, indent=2)
        self._log(f"  ✓ Boresch 参数已保存 (JSON): {boresch_json_path}")
        
        return {
            "delta_g_rest": float(delta_g),
            "error": 0.0,
            "diagnostics": boresch_params.get("diagnostics", {}) if isinstance(boresch_params, dict) else {},
            "method": boresch_params.get("method") if isinstance(boresch_params, dict) else None,
            "force_constants_raw": boresch_params.get("force_constants_raw", {}) if isinstance(boresch_params, dict) else {},
            "force_constant_clipped": boresch_params.get("force_constant_clipped", {}) if isinstance(boresch_params, dict) else {},
            "uses_analytical_release_formula": True,
            "analytical_release_assumption": (
                "Boresch release correction assumes locally harmonic, approximately Gaussian restraint-coordinate fluctuations."
            ),
        }

    def update_boresch_from_last_frame(self, boresch_params: Optional[Dict] = None) -> Optional[Dict]:
        """🔑 生产级修复：严格拦截奇点角度与异常漂移，防止自动更新引入 NaN 隐患"""
        if not _has_valid_boresch_restraint(boresch_params):
            return boresch_params
        try:
            from abfe_core import calc_boresch_from_last_frame
            orig_eq = boresch_params["equilibrium_values"]
            orig_r0 = float(orig_eq.get("r0", 1.0))
            
            # 基于当前坐标重新计算平衡几何量
            new_eq = calc_boresch_from_last_frame(
                self.positions,
                boresch_params["receptor_indices"],
                boresch_params["ligand_indices"]
            )
            new_r0 = float(new_eq.get("r0", orig_r0))
            new_thA = float(new_eq.get("thetaA0", 1.5708))
            new_thB = float(new_eq.get("thetaB0", 1.5708))
            
            # 🔑 强校验 1：角度奇点硬拦截 (安全域: 40°~140° ≈ 0.698~2.443 rad)
            thA_deg, thB_deg = np.degrees(new_thA), np.degrees(new_thB)
            if not (40.0 <= thA_deg <= 140.0) or not (40.0 <= thB_deg <= 140.0):
                self._log(f"  ⚠️ 自动更新拦截：新角度 θA={thA_deg:.1f}°, θB={thB_deg:.1f}° 触及奇点 (<40° 或 >140°)")
                self._log(f"     保留原始安全平衡值 (r0={orig_r0*10:.2f}Å)，请检查预平衡轨迹或手动指定锚点")
                return boresch_params  # 🛑 拒绝更新，阻断 NaN 源头
                
            # 🔑 强校验 2：距离漂移拦截 (> 2.5 Å 视为配体脱离口袋或严重穿模)
            r0_drift = abs(new_r0 - orig_r0)
            if r0_drift > 0.25:
                self._log(f"  ⚠️ 自动更新拦截：r0 漂移过大 ({r0_drift*10:.2f} Å > 2.5 Å)")
                self._log(f"     保留原始平衡值，体系可能未充分弛豫")
                return boresch_params
                
            # ✅ 校验通过，安全覆盖
            boresch_params["equilibrium_values"] = new_eq
            self._log(f"  ✅ 已用最后一帧安全更新 Boresch 平衡值: r0={new_r0*10:.2f}Å, θA={thA_deg:.1f}°, θB={thB_deg:.1f}°")
        except Exception as e:
            self._log(f"  ⚠️ Boresch 平衡值更新失败: {e}，使用原始值")
        return boresch_params

    def compute_final_results(self, sampling_results: Dict, correction_results: Dict, system: openmm.System = None, decoupling_scheme: str = "dual_lambda") -> Dict:
        cons_correction = 0.0
        if system is not None and self.ligand_indices:
            try:
                from abfe_core import calculate_constraint_jacobian_correction
                cons_correction = calculate_constraint_jacobian_correction(system, self.ligand_indices, self.temperature.value_in_unit(unit.kelvin))
            except Exception as e:
                self._log(f"  ⚠️ 约束 Jacobian 修正失败: {e}")

        # 🔑 核心修复：严格累加物理自由能分量
        if decoupling_scheme == "dual_lambda":
            dg_decharge = sampling_results.get("stage1", {}).get("total_delta_G", 0.0)
            dg_vdw = sampling_results.get("stage2", {}).get("total_delta_G", 0.0)
            err_decharge = sampling_results.get("stage1", {}).get("total_error", 0.0)
            err_vdw = sampling_results.get("stage2", {}).get("total_error", 0.0)
            
            dg_phys = dg_decharge + dg_vdw
            err_phys = np.sqrt(err_decharge**2 + err_vdw**2)
            self._log(f"  🔗 双λ解耦: ΔG_charge={dg_decharge:.2f} + ΔG_vdw={dg_vdw:.2f} = {dg_phys:.2f} ± {err_phys:.2f} kJ/mol")
        else:
            dg_phys = sampling_results.get("total_delta_G", 0.0)
            err_phys = sampling_results.get("total_error", 0.0)

        # ✅ 显式加入 Boresch 修正与约束修正
        dg_boresch = correction_results.get("delta_g_rest", 0.0)
        total_dg = dg_phys + cons_correction + dg_boresch
        # 约束 Jacobian 修正是解析确定性项；没有独立采样误差时不并入方差。
        total_err = np.sqrt(err_phys**2 + correction_results.get("error", 0.0)**2)

        final = {
            "decoupling_scheme": decoupling_scheme,
            "decoupling_delta_G_kJ_mol": dg_phys,
            "constraint_correction_kJ_mol": cons_correction,
            "boresch_correction_kJ_mol": dg_boresch,
            "boresch_correction_diagnostics": {
                "method": correction_results.get("method"),
                "diagnostics": correction_results.get("diagnostics", {}),
                "force_constants_raw": correction_results.get("force_constants_raw", {}),
                "force_constant_clipped": correction_results.get("force_constant_clipped", {}),
                "uses_analytical_release_formula": bool(correction_results.get("uses_analytical_release_formula", False)),
                "analytical_release_assumption": correction_results.get("analytical_release_assumption", ""),
            },
            "lj_long_range_dispersion_correction": {
                "applied": False,
                "delta_G_kJ_mol": 0.0,
                "status": "not_implemented",
                "note": (
                    "Custom softcore VDW interaction-group forces do not include "
                    "the original NonbondedForce LJ dispersion/tail correction. "
                    "This leg result is reported without an automatic LJ LRC term."
                ),
            },
            "total_delta_G_complex_kJ_mol": float(total_dg),
            "total_delta_G_complex_kcal_mol": float(total_dg / 4.184),
            "total_error_kJ_mol": float(total_err),
            "total_error_kcal_mol": float(total_err / 4.184),
            "timestamp": datetime.now().isoformat(),
            "diagnostics": sampling_results.get("diagnostics", {}),
            "stage_diagnostics": {
                "stage1": sampling_results.get("stage1", {}).get("diagnostics", {}),
                "stage2": sampling_results.get("stage2", {}).get("diagnostics", {}),
                "stage1_lambda_endpoints": sampling_results.get("stage1", {}).get("lambda_endpoint_diagnostics", {}),
                "stage2_lambda_endpoints": sampling_results.get("stage2", {}).get("lambda_endpoint_diagnostics", {}),
            },
            "provenance": _collect_pipeline_provenance(
                config=getattr(self, "_last_run_config", {}),
                system=system or self.system,
                topology=self.topology,
                positions=self.positions,
                command_line=getattr(self, "_command_line", None),
            ),
        }
        
        out_path = os.path.join(self.output_dir, "final_results.json")
        with open(out_path, "w") as f: json.dump(final, f, indent=2, cls=NumpyEncoder)
        cycle_path = os.path.join(self.output_dir, "thermodynamic_cycle.md")
        with open(cycle_path, "w", encoding="utf-8") as f:
            f.write(THERMODYNAMIC_CYCLE_DOC + "\n")
        self._log(f"\n✅ 最终结果已保存: {out_path}")
        self._log(f"  ✓ 热力学循环说明已保存: {cycle_path}")
        self._log(UnitFormatter.format_results_human(final))
        return final

    def run_full_abfe_loop(
        self,
        decoupling_scheme="dual_lambda",
        run_solvent=True,
        solvent_gro=None,
        solvent_top=None,
        **kwargs
    ):
        """完整 ABFE 循环：复合物 + 溶剂相 → 结合自由能"""
        # 1. 复合物腿
        self._log(f"\n{'='*60}")
        self._log(f"🔬 开始复合物相 ABFE 计算...")
        self._log(f"{'='*60}")
        complex_kwargs = dict(kwargs)
        complex_kwargs.setdefault("system_type", "complex")
        complex_res = self.run_full_pipeline(decoupling_scheme=decoupling_scheme, run_equilibration=True, **complex_kwargs)
        
        # 🔑 与 runabfe.py 主流程同一处修复：标准双解耦循环给出的是
        # ΔG_bind = ΔG_solvent - ΔG_complex，不是 ΔG_complex - ΔG_solvent。这里先把
        # complex 侧取成负值，后面 solvent 侧用 += 而不是 -=，效果上等价于整体取负。
        delta_g_bind = -complex_res.get(
            "total_delta_G_complex_kJ_mol",
            complex_res.get("total_delta_G", 0.0),
        )
        total_err_bind = complex_res.get(
            "total_error_kJ_mol",
            complex_res.get("total_error", 0.0),
        )

        if run_solvent and solvent_gro and solvent_top:
            print("\n💧 启动溶剂相 (Ligand-in-Water) 计算...")
            from abfe_core import SolventLegRunner  # ✅ 修复 E10：正确导入路径
            # ✅ 修复：传递残基名称字符串而非整数索引
            ligand_resname = self.topology.atom(self.ligand_indices[0]).residue.name
            solvent_runner = SolventLegRunner(ligand_resname, platform_name=self.platform_name)
            sys_solv, top_solv, pos_solv, box_solv = solvent_runner.build_solvent_system(solvent_gro, solvent_top)
            solvent_ligand_indices = [
                atom.index for atom in top_solv.atoms()
                if atom.residue.name == ligand_resname
            ]
            
            solvent_kwargs = dict(kwargs)
            solvent_kwargs.setdefault("decoupling_scheme", decoupling_scheme)
            solvent_kwargs["system_type"] = "solvent"
            solvent_kwargs["boresch_params"] = None
            solvent_res = solvent_runner.run_solvent_decoupling(pos_solv, top_solv, solvent_ligand_indices, **solvent_kwargs)
            delta_g_bind += solvent_res.get(
                "decoupling_delta_G_kJ_mol",
                solvent_res.get("total_delta_G_complex_kJ_mol", solvent_res.get("total_delta_G", 0.0)),
            )
            total_err_bind = float(np.sqrt(
                total_err_bind**2
                + solvent_res.get("total_error_kJ_mol", solvent_res.get("total_error", 0.0))**2
            ))
            
        print(f"\n🎯 最终结合自由能 ΔG_bind = {delta_g_bind:.2f} ± {total_err_bind:.2f} kJ/mol")
        return {"delta_g_bind": delta_g_bind, "total_error": total_err_bind, "complex": complex_res}

    def _assert_stage_result_sane(self, stage_label: str, result: Dict) -> None:
        """
        🔑 熔断检查：MBAR/TMBAR 求解失败或协方差不可用时，此前的代码会把
        total_error=NaN、甚至 total_delta_G 精确等于 0.0 这类明显不可信的结果当作
        "合法完成"写入 checkpoint、标记 completed，并一路传播到最终 ΔG_bind——除了
        日志里一句容易被淹没的警告，没有任何硬性拦截（曾实测出现过 decharging 腿
        total_delta_G=0.0、total_error=NaN 仍被当正常结果使用）。这里把"完全没有
        误差棒"或"自由能/误差不是有限数"当作阶段失败处理，拒绝继续，逼迫先解决
        采样/重叠/Boresch 一致性问题，而不是让一个已知不可信的数字悄悄流入生产结果。
        """
        dg = result.get("total_delta_G")
        err = result.get("total_error")
        if dg is None or not np.isfinite(dg):
            raise RuntimeError(
                f"{stage_label} 阶段 total_delta_G={dg} 不是有限数，拒绝标记为 completed。"
            )
        if err is None or not np.isfinite(err):
            raise RuntimeError(
                f"{stage_label} 阶段 total_error={err}（非有限，通常意味着 MBAR 协方差/BAR "
                "求解在 default 和 robust 两种 solver protocol 下均失败）。这条腿的结果不可信，"
                "拒绝标记为 completed 并写入最终 ΔG_bind；请检查窗口重叠率、采样长度，或该阶段"
                "是否跨越了一次 --resume 重启导致 Boresch/restraint 基准不一致，再重新采样该阶段。"
            )

    # =========================================================================
    # 6. 主流程控制器
    # =========================================================================
    def run_full_pipeline(
        self,
        decoupling_scheme: str = "dual_lambda",
        potential_type: str = "softcore",
        dexp_params: Optional[Dict] = None,
        n_states_per_stage: int = 12,
        stage1_n_states: Optional[int] = None,
        stage2_n_states: Optional[int] = None,
        n_steps_per_window: int = 50000,
        steps_per_update: int = 500,
        system_type: str = "complex",
        boresch_params: Optional[Dict] = None,
        torsion_params: Optional[Dict] = None,
        resume: bool = False,
        run_equilibration: bool = True,
        enable_early_stop: bool = False,
        **kwargs,
    ) -> Dict:
        """完整 ABFE 计算入口 (已集成全局断点续传、势能路由、二面角修正)"""
        self._last_run_config = {
            "decoupling_scheme": decoupling_scheme,
            "potential_type": potential_type,
            "n_states_per_stage": n_states_per_stage,
            "stage1_n_states": stage1_n_states,
            "stage2_n_states": stage2_n_states,
            "n_steps_per_window": n_steps_per_window,
            "steps_per_update": steps_per_update,
            "system_type": system_type,
            "resume": resume,
            "run_equilibration": run_equilibration,
            "enable_early_stop": enable_early_stop,
            "temperature_K": self.temperature.value_in_unit(unit.kelvin),
            "platform_name": self.platform_name,
            "kwargs": {
                str(k): v for k, v in kwargs.items()
                if isinstance(v, (str, int, float, bool, type(None), list, tuple, dict))
            },
        }
        self._command_line = sys.argv
        self._log(f"\n{'#' * 60}")
        self._log(
            f"# 启动完整 ABFE 流程 | 方案: {decoupling_scheme} | 势能: {potential_type} | Resume: {resume}"
        )
        self._log(f"{'#' * 60}")

        # 自动 GPU 设备策略检测
        n_windows_for_strategy = kwargs.get("n_windows_for_strategy", 2)
        gpu_strategy = self.get_device_strategy(
            n_windows=n_windows_for_strategy,
            platform_name=self.platform_name  # ✅ 透传平台名
        )
        device_indices = gpu_strategy["devices"]
        self._log(
            f"🖥️ GPU 策略: {gpu_strategy['strategy']} | 分配设备: {device_indices}"
        )

        # 加载全局状态
        state = self._load_pipeline_state() if resume else {}
        stages = state.get("stages", {})
        stage1_states = int(stage1_n_states or n_states_per_stage)
        stage2_states = int(stage2_n_states or n_states_per_stage)

        # ✅ 在预平衡前应用二面角修正
        if torsion_params:
            self.apply_torsion_corrections(torsion_params)

        # =========================================================================
        # 1. 物理预平衡 (支持智能跳过)
        # =========================================================================
        # =========================================================================
        # 1. 物理预平衡 (支持智能跳过)
        # =========================================================================
        if run_equilibration:
            equil_traj = os.path.join(self.output_dir, "pre_equilibration.dcd")
            eq_status = stages.get("equilibration", {}).get("status")
            
            # === 前置跳过逻辑 ===
            skip_equil = False
            chk_file = os.path.join(self.checkpoint_dir, "pre_equil.chk")
            if resume and os.path.exists(equil_traj) and os.path.getsize(equil_traj) > 5000:
                if eq_status == "completed":
                    self._log("  ♻️ 预平衡状态已完成，轨迹文件有效。跳过模拟。")
                    skip_equil = True
                elif os.path.exists(chk_file) and os.path.getsize(chk_file) > 512:
                    self._log("  ⚠️ 检测到未完成状态 + 有效 Checkpoint，将断点续传...")
                    skip_equil = False  # 不跳过，让 pre_equilibrate 处理续跑
                else:
                    self._log("  ⚠️ 状态未完成且无有效 Checkpoint，重新执行预平衡...")
                    skip_equil = False
            else:
                skip_equil = False  # 非 resume 模式或文件不存在，正常执行

            if skip_equil:
                # === 1. 严格加载最后一帧坐标 ===
                try:
                    import mdtraj as md
                    from mdtraj import Topology
                    md_top = Topology.from_openmm(self.topology)
                    traj = md.load(equil_traj, top=md_top)
                    if len(traj) == 0:
                        raise ValueError("轨迹文件为空")
                        
                    self.positions = traj.xyz[-1] * unit.nanometer
                    self.box_vectors = traj.unitcell_vectors[-1] * unit.nanometer
                    self._log("  ✓ 已从预平衡轨迹加载稳态坐标")
                    
                except Exception as e:
                    self._log(f"  🚨 加载轨迹坐标失败: {e}")
                    self._log("  ⛔ 初始坐标与预平衡态偏差未知，强制重新执行预平衡！")
                    skip_equil = False  # 🔑 触发下方正常预平衡流程
                    # 不继续执行后续逻辑，直接跳至 else 分支
                    
            if not skip_equil:
                # === 正常执行预平衡 ===
                self._log("  ⏳ 预平衡状态未完成或首次运行，开始执行...")
                equil_data = self.pre_equilibrate(resume=resume)  # ✅ 透传 resume
                self.positions = equil_data["positions"]
                self.box_vectors = equil_data["box_vectors"]
                self._log("  ✓ 预平衡轨迹已保存，坐标已更新至稳态。")
                
                # === 2. 快速最小化消除残余应力（仅在新跑或续跑后执行） ===
                self._log("  🔧 执行快速最小化 (2000 步) 以消除加载坐标的残余应力...")
                try:
                    temp_sys = XmlSerializer.deserialize(XmlSerializer.serialize(self.system))
                    integrator = openmm.LangevinMiddleIntegrator(
                        self.temperature, 2.0/unit.picosecond, 0.002*unit.picosecond
                    )
                    resolved_platform, props = _build_platform_props(self.platform_name)
                    platform = openmm.Platform.getPlatformByName(resolved_platform)
                    sim = app.Simulation(self.topology, temp_sys, integrator, platform, props)
                    sim.context.setPositions(self.positions)
                    if self.box_vectors is not None:
                        sim.context.setPeriodicBoxVectors(*self.box_vectors)
                    sim.minimizeEnergy(maxIterations=2000)
                    state = sim.context.getState(getPositions=True, getEnergy=True)
                    self.positions = state.getPositions()
                    final_e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    self._log(f"  ✓ 快速最小化完成，势能: {final_e:.2f} kJ/mol")
                    del sim.context; del sim; del temp_sys
                except Exception as e:
                    self._log(f"  ⚠️ 快速最小化失败: {e}，使用当前坐标继续")

        else:
            self._log("⚠️ 跳过预平衡 (使用传入初始坐标)。")

        # =========================================================================
        # 2. PBC 居中处理（防止配体跨越周期性边界）
        # =========================================================================
        # ✅ 无论是否跳过预平衡，只要坐标/盒子有效就执行居中
        # 注：曾计划由 runabfe.py 的 center_and_wrap_molecules 完成此步，但该函数从未实现，
        # 之前用 `if False` 彻底禁用了这段逻辑，导致跨盒断裂的构型无法被修复而悄悄放行。
        if self.positions is not None and self.box_vectors is not None:
            self._log("  📦 正在执行 PBC 分子完整性修复与配体居中...")
            try:
                import mdtraj as md
                md_top = md.Topology.from_openmm(self.topology)
                # ⚠️ Quantity.value_in_unit() 在底层是 list-of-Vec3（而非 numpy 数组）时
                # 返回的仍是 Python list，没有 .reshape；必须显式再包一层 np.asarray。
                # 🚨 关键修复：mdtraj 的 Cython 扩展（含 image_molecules 内部用到的
                # geometry 例程）要求 float32（"float"）缓冲区；这里之前用 float64
                # （"double"）构造，Trajectory() 构造函数会自动转换 xyz，但直接赋值
                # unitcell_vectors 不会，导致 image_molecules() 必然抛
                # "Buffer dtype mismatch, expected 'float' but got 'double'"，
                # 每次都静默回退到只居中配体、不修复其余分子跨盒撕裂的 numpy 兜底，
                # 使得撕裂的水分子从预平衡开始就带着隐患一路进入所有窗口。
                pos_nm = np.asarray(
                    self.positions.value_in_unit(unit.nanometer)
                    if hasattr(self.positions, 'value_in_unit') else self.positions,
                    dtype=np.float32,
                )
                box_nm = np.asarray(
                    [
                        v.value_in_unit(unit.nanometer) if hasattr(v, "value_in_unit") else v
                        for v in self.box_vectors
                    ],
                    dtype=np.float32,
                )
                traj = md.Trajectory(pos_nm.reshape(1, -1, 3), md_top)
                # 不传 unitcell_vectors 时 mdtraj 完全不知道盒子形状，
                # image_molecules() 会直接报 "does not define a periodic unit cell"。
                traj.unitcell_vectors = box_nm.reshape(1, 3, 3)
                traj.image_molecules(inplace=True)
                traj.center_coordinates()
                self.positions = [openmm.Vec3(float(x), float(y), float(z)) 
                                  for x, y, z in traj.xyz[0]] * unit.nanometer
                self._log("  ✓ PBC 分子完整性已修复，体系已居中至主周期")
            except Exception as e:
                self._log(f"  ⚠️ MDTraj PBC 修复失败: {e}，回退到 numpy 质心平移")
                self.positions, self.box_vectors = self._wrap_ligand_to_box(
                    self.positions, self.box_vectors, margin_nm=0.3
                )
        # 🔒 PBC 修复到此为止：image_molecules 只按分子整体做周期平移，
        # center_coordinates/_wrap_ligand_to_box 只做整体质心平移；两者都不改变
        # 任何原子间的相对位置。此前这里还有一段"L-E 界面安全弛豫"
        # (sim.minimizeEnergy(maxIterations=500))，会真实改变原子间相对坐标——
        # 不是平移，明确不允许，已删除。若 PBC 重新成像后确实出现瞬时穿模，应在
        # 下游窗口构建时各自做能量极小化（已有此步骤），而不是在这里预先"抹平"，
        # 否则每次 resume 都会用一份被悄悄弛豫过的构型重新推导 Boresch 平衡值，
        # 导致同一条腿前后窗口的限制力基准不一致。

        # =========================================================================
        # 3. 用最后一帧更新 Boresch 平衡几何量
        # =========================================================================
        # 🔑 关键修复：此前每次调用 run_full_pipeline（包括每一次 --resume 重启）都会
        # 无条件重新从当前坐标推导 Boresch 平衡几何量。但 IBS 窗口/REMD 副本是按窗口
        # 粒度做断点续传的——一条腿（decharging/vanishing）完全可能跨越多次进程重启；
        # 如果"重启前已完成的窗口"和"重启后继续采样的窗口"被喂进两套不同的 Boresch
        # 平衡值，就是在用两个不同的哈密顿量拼接同一条自由能曲线，会在拼接处产生
        # 不属于任何真实物理过程的能量跳变（实测曾导致 vdw 腿拼接曲线单步跳变
        # ~200 kJ/mol）。因此：平衡几何量只在该条腿第一次开始采样时推导一次并落盘；
        # 之后同一条腿的任何 resume 都必须原样复用，不再重算。
        if _has_valid_boresch_restraint(boresch_params):
            committed_path = os.path.join(self.checkpoint_dir, "boresch_equilibrium_committed.json")
            if resume and os.path.exists(committed_path):
                with open(committed_path, "r") as f:
                    committed_eq = json.load(f)["equilibrium_values"]
                boresch_params = dict(boresch_params)
                boresch_params["equilibrium_values"] = committed_eq
                r0 = committed_eq.get("r0", 0) * 10  # nm → Å
                self._log(
                    f"  ♻️ 本腿此前已提交过 Boresch 平衡值 (resume)，复用缓存值 "
                    f"(r0={r0:.2f} Å)，不再从当前坐标重新锚定。"
                )
            else:
                self._log("  🔧 正在用当前坐标更新 Boresch 平衡几何量...")
                boresch_params = self.update_boresch_from_last_frame(boresch_params)
                r0 = boresch_params["equilibrium_values"].get("r0", 0) * 10  # nm → Å
                self._log(f"  ✓ Boresch 平衡值已更新: r0={r0:.2f} Å")
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                with open(committed_path, "w") as f:
                    json.dump({"equilibrium_values": boresch_params["equilibrium_values"]}, f, indent=2)
                self._log(f"  📌 Boresch 平衡值已提交落盘，本腿后续 resume 将强制复用: {committed_path}")

        # =========================================================================
        # 2. 路由采样 (支持阶段级 Resume)
        # =========================================================================
        sampling_key = f"sampling_{decoupling_scheme}"
        samp_status = stages.get(sampling_key, {}).get("status")

        if resume and samp_status == "completed":
            self._log(f"  ♻️ {decoupling_scheme} 采样已完成，跳过")
            results_file = os.path.join(self.output_dir, "final_results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    final = json.load(f)
                self.results["final"] = final
                self._log("  ✓ 已加载已有最终结果")
                return final
            else:
                self._log("  ⚠️ 状态标记为完成但未找到结果文件，重新运行采样")
        if decoupling_scheme == "dual_lambda":
            stage1_key = "sampling_dual_decharging"
            stage2_key = "sampling_dual_vanishing"
            stage1_status = stages.get(stage1_key, {}).get("status")
            stage2_status = stages.get(stage2_key, {}).get("status")
            stage1_file = os.path.join(self.checkpoint_dir, "stage1_decharging.json")
            stage2_file = os.path.join(self.checkpoint_dir, "stage2_vanishing.json")
            preopt1_file = os.path.join(
                self.checkpoint_dir, "preopt_dual_decharging.json"
            )
            preopt2_file = os.path.join(
                self.checkpoint_dir, "preopt_dual_vanishing.json"
            )
            window_ranges_1 = None
            window_ranges_2 = None

            # === Stage 1: pre-opt + resume check ===
            optimized_lambdas_1 = None
            if resume and os.path.exists(preopt1_file):
                try:
                    with open(preopt1_file, "r") as f:
                        cached = json.load(f)
                    cached_lambdas = cached["lambdas_var"]
                    if len(cached_lambdas) == stage1_states:
                        optimized_lambdas_1 = cached_lambdas
                        # 🔑 之前这里只读 lambdas_var，缓存里同时存着的 window_ranges
                        # 从未被读回来用——导致手动往缓存文件里塞自定义窗口边界（比如
                        # 只放大某一个窗口去加密局部 λ）完全不会生效，_run_dual_lambda_stage
                        # 还是会用 generate_overlapping_windows 重新自动划分。这里补上，
                        # 并校验覆盖范围与 lambdas_var 长度一致才采用，否则保留 None
                        # 让下游按默认自动划分处理。
                        cached_ranges = cached.get("window_ranges")
                        if cached_ranges:
                            covered = sorted({i for s, e in cached_ranges for i in range(s, e)})
                            if covered == list(range(len(cached_lambdas))):
                                window_ranges_1 = [tuple(r) for r in cached_ranges]
                            else:
                                self._log("  ⚠️ Stage 1 缓存里的 window_ranges 覆盖范围与 lambdas_var 不匹配，忽略并回退自动划分")
                        self._log(
                            f"  ♻️ 已加载 Stage 1 优化路径缓存 ({len(optimized_lambdas_1)} 个状态)"
                            + ("，含手动窗口边界" if window_ranges_1 else "")
                        )
                    else:
                        self._log(
                            f"  ⚠️ Stage 1 优化路径缓存状态数不匹配 "
                            f"({len(cached_lambdas)} != {stage1_states})，重新优化"
                        )
                except Exception as e:
                    self._log(f"  ⚠️ 加载 Stage 1 优化缓存失败: {e}，将重新优化")

            if optimized_lambdas_1 is None:
                try:
                    opt_res = self._run_dual_lambda_optimization(
                        "decharging",
                        n_states=stage1_states,
                        n_steps_per_state=10000,
                        potential_type=potential_type,
                    )
                    optimized_lambdas_1 = opt_res["lambdas_var"]
                    window_ranges_1 = opt_res.get("window_ranges")
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    with open(preopt1_file, "w") as f:
                        json.dump({
                            "lambdas_var": optimized_lambdas_1,
                            "window_ranges": window_ranges_1,
                            "n_states": len(optimized_lambdas_1),
                        }, f, indent=2)
                    self._log(f"  ✓ Stage 1 优化路径已缓存")
                except Exception as e:
                    raise RuntimeError(f"Stage 1 自适应优化失败，拒绝静默回退线性路径: {e}") from e

            should_run_stage1 = True
            if resume and stage1_status == "completed" and os.path.exists(stage1_file):
                try:
                    with open(stage1_file, "r") as f:
                        stage1 = json.load(f)
                    if stage1.get("n_states") == stage1_states:
                        self._log("  ♻️ 双λ Stage 1 (去电荷) 已完成，跳过")
                        should_run_stage1 = False
                    else:
                        self._log("  ⚠️ Stage 1 结果缓存状态数不匹配，重新运行")
                except Exception as e:
                    self._log(f"  ⚠️ Stage 1 缓存读取失败: {e}，重新运行")

            # === Stage 2: pre-opt + resume check ===
            optimized_lambdas_2 = None
            if resume and os.path.exists(preopt2_file):
                try:
                    with open(preopt2_file, "r") as f:
                        cached = json.load(f)
                    cached_lambdas = cached["lambdas_var"]
                    if len(cached_lambdas) == stage2_states:
                        optimized_lambdas_2 = cached_lambdas
                        # 同 Stage 1：把缓存里的 window_ranges 读回来并校验，
                        # 否则手动编辑缓存加密某个窗口不会真正生效。
                        cached_ranges = cached.get("window_ranges")
                        if cached_ranges:
                            covered = sorted({i for s, e in cached_ranges for i in range(s, e)})
                            if covered == list(range(len(cached_lambdas))):
                                window_ranges_2 = [tuple(r) for r in cached_ranges]
                            else:
                                self._log("  ⚠️ Stage 2 缓存里的 window_ranges 覆盖范围与 lambdas_var 不匹配，忽略并回退自动划分")
                        self._log(
                            f"  ♻️ 已加载 Stage 2 优化路径缓存 ({len(optimized_lambdas_2)} 个状态)"
                            + ("，含手动窗口边界" if window_ranges_2 else "")
                        )
                    else:
                        self._log(
                            f"  ⚠️ Stage 2 优化路径缓存状态数不匹配 "
                            f"({len(cached_lambdas)} != {stage2_states})，重新优化"
                        )
                except Exception as e:
                    self._log(f"  ⚠️ 加载 Stage 2 优化缓存失败: {e}，将重新优化")

            if optimized_lambdas_2 is None:
                try:
                    opt_res = self._run_dual_lambda_optimization(
                        "vanishing",
                        n_states=stage2_states,
                        n_steps_per_state=10000,
                        potential_type=potential_type,
                    )
                    optimized_lambdas_2 = opt_res["lambdas_var"]
                    window_ranges_2 = opt_res.get("window_ranges")
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    with open(preopt2_file, "w") as f:
                        json.dump({
                            "lambdas_var": optimized_lambdas_2,
                            "window_ranges": window_ranges_2,
                            "n_states": len(optimized_lambdas_2),
                        }, f, indent=2)
                    self._log(f"  ✓ Stage 2 优化路径已缓存")
                except Exception as e:
                    raise RuntimeError(f"Stage 2 自适应优化失败，拒绝静默回退线性路径: {e}") from e

            should_run_stage2 = True
            if resume and stage2_status == "completed" and os.path.exists(stage2_file):
                try:
                    with open(stage2_file, "r") as f:
                        stage2 = json.load(f)
                    if stage2.get("n_states") == stage2_states:
                        self._log("  ♻️ 双λ Stage 2 (去VDW) 已完成，跳过")
                        should_run_stage2 = False
                    else:
                        self._log("  ⚠️ Stage 2 结果缓存状态数不匹配，重新运行")
                except Exception as e:
                    self._log(f"  ⚠️ Stage 2 缓存读取失败: {e}，重新运行")

            # === Stage 2: 精修阶段（中等步数探针，基于实测 |Δf| 精修 λ 路径/窗口边界）===
            # 只对 Stage 2 (去VDW/vanishing) 生效：Stage 1 (去电荷) 走的是
            # PME-REMD-MBAR 路径（见 _run_dual_lambda_stage 里 decharging 分支），
            # 不产出 dual_window_*_coul_energies.npy，refine_stage_lambda_path_from_data
            # 无从下手。粗扫(几千步/态) → 精修(中等步数/窗口，本节) → 生产(满步数)，
            # 精修用独立 scratch 目录采样，绝不写入生产目录，避免被生产阶段的 resume
            # 形状校验误判为"已采样完成"而跳过真正的生产步数。
            if kwargs.get("enable_lambda_refine", False) and should_run_stage2:
                with open(preopt2_file, "r") as f:
                    _preopt2_cached = json.load(f)
                _already_refined = (
                    _preopt2_cached.get("provenance", {}).get("source")
                    == "refine_stage_lambda_path_from_data"
                )
                if _already_refined:
                    self._log("  ♻️ Stage 2 λ 路径已精修过（provenance 已标记），跳过中等步数探针")
                else:
                    self._log("\n[双λ] 🔬 Stage 2 精修：中等步数探针精修 λ 路径与窗口边界")
                    _refine_window_ranges_2 = window_ranges_2 or generate_overlapping_windows(
                        n_states=len(optimized_lambdas_2),
                        n_windows=None,
                        pts_per_window=6,
                        overlap=2,
                    )
                    # refine_stage_lambda_path_from_data 直接从 preopt_path 里读
                    # window_ranges 去定位能量文件，必须先把这里实际要用来采样的
                    # window_ranges 落盘，保证跟探针管理器用的边界完全一致，
                    # 否则窗口能量文件数会跟缓存里的旧 window_ranges 数对不上而报错。
                    with open(preopt2_file, "w") as f:
                        json.dump({
                            "lambdas_var": optimized_lambdas_2,
                            "window_ranges": [list(r) for r in _refine_window_ranges_2],
                            "n_states": len(optimized_lambdas_2),
                        }, f, indent=2)
                    optimized_lambdas_2, window_ranges_2 = self._refine_lambda_path_with_medium_probe(
                        "vanishing",
                        fixed_lam_coul=0.0,
                        fixed_lam_vdw=1.0,
                        lambdas_var=optimized_lambdas_2,
                        window_ranges=_refine_window_ranges_2,
                        preopt_path=preopt2_file,
                        potential_type=potential_type,
                        dexp_params=dexp_params,
                        boresch_params=boresch_params,
                        refine_n_steps_per_window=kwargs.get("refine_n_steps_per_window", 30000),
                        refine_steps_per_update=kwargs.get("refine_steps_per_update", steps_per_update),
                        max_window_span_kJ=kwargs.get("refine_max_window_span_kJ", 35.0),
                        overlap=kwargs.get("refine_overlap", 2),
                        resume=resume,
                    )
                    stage2_states = len(optimized_lambdas_2)

            # === Sampling: parallel or sequential ===
            _parallel_stages = kwargs.get("parallel_stages", False)

            if _parallel_stages and should_run_stage1 and should_run_stage2:
                self._log("\n[双λ] 🚀 并行执行 Stage 1 (去电荷) + Stage 2 (去VDW)")
                state_dir = os.path.join(self.checkpoint_dir, "parallel_state")
                self._save_state_to_dir(state_dir)

                _res_dir = os.path.join(self.checkpoint_dir, "parallel_results")
                os.makedirs(_res_dir, exist_ok=True)
                _res1 = os.path.join(_res_dir, "stage1.json")
                _res2 = os.path.join(_res_dir, "stage2.json")

                _temp_k = self.temperature.value_in_unit(unit.kelvin)
                _common = dict(
                    n_states_stage1=stage1_states,
                    n_states_stage2=stage2_states,
                    n_steps_per_window=n_steps_per_window,
                    steps_per_update=steps_per_update,
                    system_type=system_type,
                    potential_type=potential_type,
                    dexp_params=dexp_params,
                    enable_early_stop=enable_early_stop,
                    boresch_params=boresch_params,
                    enable_gradual_warmup=kwargs.get("enable_gradual_warmup", True),
                    warmup_steps=kwargs.get("warmup_steps", 500000),
                    resume=resume,
                )
                stage1_platform = self.platform_name
                stage2_platform = self.platform_name
                if str(self.platform_name).upper().startswith("CUDA"):
                    env_stage1 = os.environ.get("IBS_STAGE1_CUDA_DEVICE")
                    env_stage2 = os.environ.get("IBS_STAGE2_CUDA_DEVICE")
                    if env_stage1 is not None and env_stage2 is not None and env_stage1 != env_stage2:
                        stage1_platform = f"CUDA:{env_stage1}"
                        stage2_platform = f"CUDA:{env_stage2}"
                        self._log(f"  🔀 并行阶段将分别使用 CUDA 设备 {env_stage1} 和 {env_stage2}")
                    else:
                        self._log("  ⚠️ 检测到并行双阶段 + CUDA，但未提供两个不同 GPU；为避免上下文冲突，回退为串行执行。")
                        _parallel_stages = False

                if _parallel_stages:
                    ctx = mp.get_context("spawn")
                    p1 = ctx.Process(
                        target=_run_stage_worker_process,
                        args=(state_dir, _temp_k, stage1_platform, self.output_dir,
                              "decharging", 1.0, 1.0,
                              _common["n_states_stage1"], _common["n_steps_per_window"],
                              _common["steps_per_update"], _common["system_type"],
                              _common["potential_type"], _common["dexp_params"],
                              optimized_lambdas_1, _common["enable_early_stop"],
                              _common["boresch_params"], _common["enable_gradual_warmup"],
                              _common["warmup_steps"], _common["resume"], _res1),
                    )
                    p2 = ctx.Process(
                        target=_run_stage_worker_process,
                        args=(state_dir, _temp_k, stage2_platform, self.output_dir,
                              "vanishing", 0.0, 1.0,
                              _common["n_states_stage2"], _common["n_steps_per_window"],
                              _common["steps_per_update"], _common["system_type"],
                              _common["potential_type"], _common["dexp_params"],
                              optimized_lambdas_2, _common["enable_early_stop"],
                              _common["boresch_params"], _common["enable_gradual_warmup"],
                              _common["warmup_steps"], _common["resume"], _res2),
                    )
                    p1.start()
                    p2.start()
                    p1.join()
                    p2.join()
                else:
                    _run_stage_worker_process(
                        state_dir, _temp_k, stage1_platform, self.output_dir,
                        "decharging", 1.0, 1.0,
                        _common["n_states_stage1"], _common["n_steps_per_window"],
                        _common["steps_per_update"], _common["system_type"],
                        _common["potential_type"], _common["dexp_params"],
                        optimized_lambdas_1, _common["enable_early_stop"],
                        _common["boresch_params"], _common["enable_gradual_warmup"],
                        _common["warmup_steps"], _common["resume"], _res1,
                    )
                    _run_stage_worker_process(
                        state_dir, _temp_k, stage2_platform, self.output_dir,
                        "vanishing", 0.0, 1.0,
                        _common["n_states_stage2"], _common["n_steps_per_window"],
                        _common["steps_per_update"], _common["system_type"],
                        _common["potential_type"], _common["dexp_params"],
                        optimized_lambdas_2, _common["enable_early_stop"],
                        _common["boresch_params"], _common["enable_gradual_warmup"],
                        _common["warmup_steps"], _common["resume"], _res2,
                    )

                # Check for errors
                for _rf, _label in [(_res1, "Stage 1"), (_res2, "Stage 2")]:
                    with open(_rf) as f:
                        _r = json.load(f)
                    if "error" in _r:
                        raise RuntimeError(f"{_label} 子进程失败: {_r['error']}")

                with open(_res1) as f:
                    stage1 = json.load(f)
                with open(_res2) as f:
                    stage2 = json.load(f)

                # Save checkpoint files
                self._assert_stage_result_sane("Stage 1 (decharging)", stage1)
                _s1 = {"stage": "decharging", "total_delta_G": stage1["total_delta_G"],
                        "total_error": stage1["total_error"], "n_states": stage1_states}
                with open(stage1_file, "w") as f:
                    json.dump(_s1, f, indent=2)
                self._update_stage_status(stage1_key, "completed",
                                          {"total_delta_G": stage1["total_delta_G"]})

                self._assert_stage_result_sane("Stage 2 (vanishing)", stage2)
                _s2 = {"stage": "vanishing", "total_delta_G": stage2["total_delta_G"],
                        "total_error": stage2["total_error"], "n_states": stage2_states}
                with open(stage2_file, "w") as f:
                    json.dump(_s2, f, indent=2)
                self._update_stage_status(stage2_key, "completed",
                                          {"total_delta_G": stage2["total_delta_G"]})

            else:
                # === Sequential execution ===
                if should_run_stage1:
                    self._log("\n[双λ] Stage 1: 去电荷 (λ_coul: 1→0, λ_vdw=1)")
                    stage1 = self._run_dual_lambda_stage(
                        "decharging",
                        fixed_lam_coul=1.0,
                        fixed_lam_vdw=1.0,
                        potential_type=potential_type,
                        dexp_params=dexp_params,
                        n_states=stage1_states,
                        n_steps_per_window=n_steps_per_window,
                        steps_per_update=steps_per_update,
                        system_type=system_type,
                        resume=resume,
                        optimized_lambdas=optimized_lambdas_1,
                        window_ranges=window_ranges_1,
                        enable_early_stop=enable_early_stop,
                        boresch_params=boresch_params,
                        enable_gradual_warmup=kwargs.get("enable_gradual_warmup", True),
                        warmup_steps=kwargs.get("warmup_steps", 500000),
                    )
                    self._assert_stage_result_sane("Stage 1 (decharging)", stage1)
                    stage1_save = {
                        "stage": "decharging",
                        "total_delta_G": stage1["total_delta_G"],
                        "total_error": stage1["total_error"],
                        "n_states": stage1_states,
                    }
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    with open(stage1_file, "w") as f:
                        json.dump(stage1_save, f, indent=2)
                    self._update_stage_status(
                        stage1_key,
                        "completed",
                        {
                            "total_delta_G": stage1["total_delta_G"],
                        },
                    )

                if should_run_stage2:
                    self._log("\n[双λ] Stage 2: 去VDW (λ_coul=0, λ_vdw: 1→0)")
                    stage2 = self._run_dual_lambda_stage(
                        "vanishing",
                        fixed_lam_coul=0.0,
                        fixed_lam_vdw=1.0,
                        potential_type=potential_type,
                        dexp_params=dexp_params,
                        n_states=stage2_states,
                        n_steps_per_window=n_steps_per_window,
                        steps_per_update=steps_per_update,
                        system_type=system_type,
                        resume=resume,
                        optimized_lambdas=optimized_lambdas_2,
                        window_ranges=window_ranges_2,
                        enable_early_stop=enable_early_stop,
                        boresch_params=boresch_params,
                        enable_gradual_warmup=kwargs.get("enable_gradual_warmup", True),
                        warmup_steps=kwargs.get("warmup_steps", 500000),
                    )
                    self._assert_stage_result_sane("Stage 2 (vanishing)", stage2)
                    stage2_save = {
                        "stage": "vanishing",
                        "total_delta_G": stage2["total_delta_G"],
                        "total_error": stage2["total_error"],
                        "n_states": stage2_states,
                    }
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    with open(stage2_file, "w") as f:
                        json.dump(stage2_save, f, indent=2)
                    self._update_stage_status(
                        stage2_key,
                        "completed",
                        {
                            "total_delta_G": stage2["total_delta_G"],
                        },
                    )

            sampling = {
                "total_delta_G": stage1["total_delta_G"] + stage2["total_delta_G"],
                "total_error": np.sqrt(stage1["total_error"] ** 2 + stage2["total_error"] ** 2),
                "stage1": stage1,
                "stage2": stage2,
            }
            
            # ✅ 【修复 1】延迟状态更新：确保 Boresch 修正与结果落盘成功后再标记 completed
            if system_type == "solvent" and not _has_valid_boresch_restraint(boresch_params):
                correction = {"delta_g_rest": 0.0, "error": 0.0}
            else:
                correction = self.apply_boresch_correction(
                    boresch_params,
                    autoload_from_disk=kwargs.get("allow_disk_boresch_autoload", True),
                )
                
            final = self.compute_final_results(sampling, correction, system=self.system)
            self.results["final"] = final
            
            # 仅当最终结果成功生成后，才标记阶段完成
            self._update_stage_status(
                sampling_key,
                "completed",
                {"total_delta_G": sampling.get("total_delta_G")},
            )
            return final  # ✅ 新增：阻断落入末尾通用汇总块，避免二次写入与重复计算
        elif decoupling_scheme == "single_lambda":
            path_cache_file = os.path.join(self.checkpoint_dir, "path_single_lambda.json")
            path_1d = None
            if resume and os.path.exists(path_cache_file):
                try:
                    with open(path_cache_file) as f:
                        _cached = json.load(f)
                    path_1d = [tuple(p) for p in _cached["path"]]
                    self._log(f"  ♻️ 已加载 single_lambda 路径缓存 ({len(path_1d)} 个状态)")
                except Exception as e:
                    self._log(f"  ⚠️ 加载 single_lambda 路径缓存失败: {e}，将重新生成")

            if path_1d is None:
                lambdas = np.linspace(1.0, 0.0, n_states_per_stage).tolist()
                path_1d = [(lam, lam) for lam in lambdas]
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                with open(path_cache_file, "w") as f:
                    json.dump({"path": path_1d, "scheme": "single_lambda"}, f, indent=2)

            _samp_file = os.path.join(self.checkpoint_dir, "sampling_single_lambda.json")
            _should_run = True
            if resume:
                _key = "sampling_single_lambda"
                _status = stages.get(_key, {}).get("status")
                if _status == "completed" and os.path.exists(_samp_file):
                    try:
                        with open(_samp_file) as f:
                            sample_result = json.load(f)
                        self._log("  ♻️ single_lambda 采样已完成，跳过")
                        _should_run = False
                    except Exception:
                        pass

            if _should_run:
                sample_result = self._run_2d_lambda_stage(
                    path_2d=path_1d,
                    label="single_lambda",
                    n_steps_per_window=n_steps_per_window,
                    steps_per_update=steps_per_update,
                    system_type=system_type,
                    resume=resume,
                    potential_type=potential_type,
                    dexp_params=dexp_params,
                    enable_early_stop=enable_early_stop,
                    boresch_params=boresch_params,
                    enable_gradual_warmup=kwargs.get("enable_gradual_warmup", True),
                    warmup_steps=kwargs.get("warmup_steps", 500000),
                )
                _save = {
                    "total_delta_G": sample_result["total_delta_G"],
                    "total_error": sample_result["total_error"],
                }
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                with open(_samp_file, "w") as f:
                    json.dump(_save, f, indent=2)
                self._update_stage_status(
                    "sampling_single_lambda",
                    "completed",
                    {"total_delta_G": sample_result["total_delta_G"]},
                )

            sampling = {
                "total_delta_G": sample_result["total_delta_G"],
                "total_error": sample_result["total_error"],
            }

            if system_type == "solvent" and not _has_valid_boresch_restraint(boresch_params):
                correction = {"delta_g_rest": 0.0, "error": 0.0}
            else:
                correction = self.apply_boresch_correction(
                    boresch_params,
                    autoload_from_disk=kwargs.get("allow_disk_boresch_autoload", True),
                )

            final = self.compute_final_results(
                sampling,
                correction,
                system=self.system,
                decoupling_scheme="single_lambda",
            )
            self.results["final"] = final
            return final
        elif decoupling_scheme == "2d_diagonal":
            # === 生成对角线路径 ===
            path_cache_file = os.path.join(self.checkpoint_dir, "path_2d_diagonal.json")
            path_2d = None
            if resume and os.path.exists(path_cache_file):
                try:
                    with open(path_cache_file) as f:
                        _cached = json.load(f)
                    path_2d = [tuple(p) for p in _cached["path"]]
                    self._log(f"  ♻️ 已加载对角线 2D 路径缓存 ({len(path_2d)} 个状态)")
                except Exception as e:
                    self._log(f"  ⚠️ 加载 2D 路径缓存失败: {e}，将重新生成")

            if path_2d is None:
                planner = TwoDimensionalLambdaPathPlanner(
                    n_points=n_states_per_stage, path_type="diagonal"
                )
                path_2d = planner.generate_path()
                self._log(f"  📐 生成了对角线 2D 路径 ({len(path_2d)} 个状态)")
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                with open(path_cache_file, "w") as f:
                    json.dump({"path": path_2d, "scheme": "2d_diagonal"}, f, indent=2)

            # === 采样 ===
            _samp_file = os.path.join(self.checkpoint_dir, "sampling_2d_diagonal.json")
            _should_run = True
            if resume:
                _key = "sampling_2d_diagonal"
                _status = stages.get(_key, {}).get("status")
                if _status == "completed" and os.path.exists(_samp_file):
                    try:
                        with open(_samp_file) as f:
                            sample_result = json.load(f)
                        self._log("  ♻️ 对角线 2D 采样已完成，跳过")
                        _should_run = False
                    except Exception:
                        pass

            if _should_run:
                sample_result = self._run_2d_lambda_stage(
                    path_2d=path_2d,
                    label="2d_diagonal",
                    n_steps_per_window=n_steps_per_window,
                    steps_per_update=steps_per_update,
                    system_type=system_type,
                    resume=resume,
                    potential_type=potential_type,
                    dexp_params=dexp_params,
                    enable_early_stop=enable_early_stop,
                    boresch_params=boresch_params,
                    enable_gradual_warmup=kwargs.get("enable_gradual_warmup", True),
                    warmup_steps=kwargs.get("warmup_steps", 500000),
                )
                _save = {"total_delta_G": sample_result["total_delta_G"],
                         "total_error": sample_result["total_error"]}
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                with open(_samp_file, "w") as f:
                    json.dump(_save, f, indent=2)
                self._update_stage_status("sampling_2d_diagonal", "completed",
                                          {"total_delta_G": sample_result["total_delta_G"]})

            sampling = {"total_delta_G": sample_result["total_delta_G"],
                        "total_error": sample_result["total_error"]}

            if system_type == "solvent" and not _has_valid_boresch_restraint(boresch_params):
                correction = {"delta_g_rest": 0.0, "error": 0.0}
            else:
                correction = self.apply_boresch_correction(
                    boresch_params,
                    autoload_from_disk=kwargs.get("allow_disk_boresch_autoload", True),
                )

            final = self.compute_final_results(
                sampling, correction, system=self.system, decoupling_scheme="2d_diagonal"
            )
            self.results["final"] = final
            return final
        elif decoupling_scheme == "2d_geodesic":
            # === 测地线路径优化 ===
            path_cache_file = os.path.join(self.checkpoint_dir, "path_2d_geodesic.json")
            path_2d = None
            if resume and os.path.exists(path_cache_file):
                try:
                    with open(path_cache_file) as f:
                        _cached = json.load(f)
                    path_2d = [tuple(p) for p in _cached["path"]]
                    self._log(f"  ♻️ 已加载测地线 2D 路径缓存 ({len(path_2d)} 个状态)")
                except Exception as e:
                    self._log(f"  ⚠️ 加载测地线路径缓存失败: {e}，将重新优化")

            if path_2d is None:
                from abfe_preoptimizer import optimize_2d_geodesic_path
                path_2d = optimize_2d_geodesic_path(
                    system=self.system,
                    topology=self.topology,
                    positions=self.positions,
                    box_vectors=self.box_vectors,
                    ligand_indices=self.ligand_indices,
                    n_grid=n_states_per_stage,
                    n_steps_per_point=3000,
                    temperature=self.temperature.value_in_unit(unit.kelvin),
                    platform_name=self.platform_name,
                )
                self._log(f"  🗺️ 测地线优化完成 ({len(path_2d)} 个状态)")
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                with open(path_cache_file, "w") as f:
                    json.dump({"path": path_2d, "scheme": "2d_geodesic"}, f, indent=2)

            # === 采样 (复用 _run_2d_lambda_stage) ===
            _samp_file = os.path.join(self.checkpoint_dir, "sampling_2d_geodesic.json")
            _should_run = True
            if resume:
                _key = "sampling_2d_geodesic"
                _status = stages.get(_key, {}).get("status")
                if _status == "completed" and os.path.exists(_samp_file):
                    try:
                        with open(_samp_file) as f:
                            sample_result = json.load(f)
                        self._log("  ♻️ 测地线 2D 采样已完成，跳过")
                        _should_run = False
                    except Exception:
                        pass

            if _should_run:
                sample_result = self._run_2d_lambda_stage(
                    path_2d=path_2d,
                    label="2d_geodesic",
                    n_steps_per_window=n_steps_per_window,
                    steps_per_update=steps_per_update,
                    system_type=system_type,
                    resume=resume,
                    potential_type=potential_type,
                    dexp_params=dexp_params,
                    enable_early_stop=enable_early_stop,
                    boresch_params=boresch_params,
                    enable_gradual_warmup=kwargs.get("enable_gradual_warmup", True),
                    warmup_steps=kwargs.get("warmup_steps", 500000),
                )
                _save = {"total_delta_G": sample_result["total_delta_G"],
                         "total_error": sample_result["total_error"]}
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                with open(_samp_file, "w") as f:
                    json.dump(_save, f, indent=2)
                self._update_stage_status("sampling_2d_geodesic", "completed",
                                          {"total_delta_G": sample_result["total_delta_G"]})

            sampling = {"total_delta_G": sample_result["total_delta_G"],
                        "total_error": sample_result["total_error"]}

            if system_type == "solvent" and not _has_valid_boresch_restraint(boresch_params):
                correction = {"delta_g_rest": 0.0, "error": 0.0}
            else:
                correction = self.apply_boresch_correction(
                    boresch_params,
                    autoload_from_disk=kwargs.get("allow_disk_boresch_autoload", True),
                )

            final = self.compute_final_results(
                sampling, correction, system=self.system, decoupling_scheme="2d_geodesic"
            )
            self.results["final"] = final
            return final
        else:
            raise ValueError(f"不支持的解耦方案: {decoupling_scheme}")




# ============================================================================
# 8. 传统 ABFE-REMD 流水线 (从 traditional_abfe_remd.py 迁移)
# ============================================================================
class TraditionalABFEPipeline:
    """传统 REMD + 离线 MBAR 双阶段 ABFE 流水线"""
    def __init__(
        self,
        system: openmm.System,
        topology: app.Topology,
        positions,
        box_vectors,
        ligand_indices: List[int],
        temperature: float = 300.0,
        platform_name: str = "CUDA",
        output_dir: str = "./traditional_abfe",
    ):
        self.system = system
        self.topology = topology
        self.positions = positions
        self.box_vectors = box_vectors
        self.ligand_indices = ligand_indices
        self.temperature = temperature
        self.platform_name = platform_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @classmethod
    def from_gromacs(
        cls,
        gro_file: str,
        top_file: str,
        ligand_resname: str,
        temperature: float = 300.0,
        platform_name: str = "CUDA",
        output_dir: str = "./traditional_abfe",
        gmx_include_dir: str = None,
    ):
        gro = app.GromacsGroFile(gro_file)
        top = app.GromacsTopFile(
            top_file,
            periodicBoxVectors=gro.getPeriodicBoxVectors(),
            includeDir=gmx_include_dir,
        )
        system = top.createSystem(
            nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer,
            constraints=app.HBonds, rigidWater=True,
        )
        topology = top.topology
        positions = gro.positions
        box_vectors = gro.getPeriodicBoxVectors()
        ligand_indices = [a.index for a in topology.atoms() if a.residue.name == ligand_resname]
        return cls(
            system=system, topology=topology,
            positions=positions, box_vectors=box_vectors,
            ligand_indices=ligand_indices,
            temperature=temperature, platform_name=platform_name,
            output_dir=output_dir,
        )

    def run_leg(
        self,
        stage_name: str,
        lambdas_coul: List[float],
        lambdas_vdw: List[float],
        n_steps: int = 500000,
        exchange_interval: int = 1000,
        resume: bool = False,
        boresch_params: Optional[Dict] = None,
        potential_type: str = "softcore",
    ) -> Dict:
        print(f"\n{'='*60}\n🧪 开始 {stage_name} 腿解耦\n{'='*60}")
        if len(lambdas_coul) != len(lambdas_vdw):
            raise ValueError("传统 REMD 腿的 lambdas_coul/lambdas_vdw 长度必须一致。")
        if potential_type == "dexp":
            raise NotImplementedError(
                "traditional / PME-REMD 路径当前未实现 DEXP 或混合 softcore 替代势；"
                "如需 DEXP，请使用 IBS dual_lambda。"
            )
        stage_output_dir = os.path.join(self.output_dir, stage_name)
        os.makedirs(stage_output_dir, exist_ok=True)
        traj_files = _expected_remd_traj_files(stage_output_dir, stage_name, len(lambdas_coul))
        u_kn_path = os.path.join(self.output_dir, f"{stage_name}_u_kn.npy")
        n_k_path = u_kn_path + ".n_k.npy"

        if resume and os.path.exists(u_kn_path):
            print("  ♻️ 检测到已有 u_kn，跳过 REMD 采样与重算，直接求解 MBAR")
            u_kn = np.load(u_kn_path)
            analyzer = TraditionalMBARAnalyzer(temperature=self.temperature)
            if not os.path.exists(n_k_path):
                raise RuntimeError(f"u_kn 缓存缺少样本数 sidecar: {n_k_path}")
            analyzer._last_n_k = np.load(n_k_path)
            return analyzer.solve(u_kn)

        expected_frames = max(1, _expected_remd_frame_count(n_steps))
        if resume and _all_remd_trajs_valid(
            stage_output_dir,
            stage_name,
            len(lambdas_coul),
            min_frames=expected_frames,
        ):
            print("  ♻️ 检测到完整 REMD DCD，视为采样已完成，跳过 REMD 继续离线 MBAR")
        else:
            remd = REMDManager(
                system_template=self.system,
                topology=self.topology,
                positions=self.positions,
                box_vectors=self.box_vectors,
                ligand_indices=self.ligand_indices,
                lambdas_coul=lambdas_coul,
                lambdas_vdw=lambdas_vdw,
                temperature=self.temperature,
                platform_name=self.platform_name,
                output_dir=stage_output_dir,
                boresch_params=boresch_params,
            )
            traj_files = remd.run(
                n_steps=n_steps,
                exchange_interval=exchange_interval,
                stage_name=stage_name,
            )

        analyzer = TraditionalMBARAnalyzer(temperature=self.temperature)
        u_kn = analyzer.compute_u_kn(
            traj_files=traj_files,
            system_template=self.system,
            ligand_indices=self.ligand_indices,
            lambdas_coul=lambdas_coul,
            lambdas_vdw=lambdas_vdw,
            platform_name="CPU",
            topology=self.topology,
            reference_positions=self.positions,
            reference_box_vectors=self.box_vectors,
            boresch_params=boresch_params,
        )
        np.save(u_kn_path, u_kn)
        np.save(n_k_path, analyzer._last_n_k)
        return analyzer.solve(u_kn)

    def run_full(
        self,
        n_lambda: int = 12,
        n_steps_per_leg: int = 500000,
        boresch_correction: float = 0.0,
        boresch_params: Optional[Dict] = None,
        potential_type: str = "softcore",
    ) -> Dict:
        lambdas_coul = np.linspace(1.0, 0.0, n_lambda).tolist()
        lambdas_vdw = [1.0] * n_lambda
        res_coul = self.run_leg(
            "decharging",
            lambdas_coul,
            lambdas_vdw,
            n_steps_per_leg,
            resume=False,
            boresch_params=boresch_params,
            potential_type=potential_type,
        )

        lambdas_coul = [0.0] * n_lambda
        lambdas_vdw = np.linspace(1.0, 0.0, n_lambda).tolist()
        res_vdw = self.run_leg(
            "vanishing",
            lambdas_coul,
            lambdas_vdw,
            n_steps_per_leg,
            resume=False,
            boresch_params=boresch_params,
            potential_type=potential_type,
        )

        dg_leg = res_coul["delta_G"] + res_vdw["delta_G"]
        err_leg = np.sqrt(res_coul["error"]**2 + res_vdw["error"]**2)
        dg_total = dg_leg + boresch_correction

        final = {
            "stage_decharging": res_coul,
            "stage_vanishing": res_vdw,
            "delta_G_leg_kJ_mol": dg_leg,
            "error_leg_kJ_mol": err_leg,
            "boresch_correction_kJ_mol": boresch_correction,
            "delta_G_total_kJ_mol": dg_total,
            "delta_G_total_kcal_mol": dg_total / 4.184,
        }
        with open(os.path.join(self.output_dir, "final_results.json"), "w") as f:
            json.dump(final, f, indent=2)
        print(f"\n✅ 传统腿完成 | ΔG_leg = {dg_total:.2f} ± {err_leg:.2f} kJ/mol")
        return final
