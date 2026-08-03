#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ABFE 计算命令行入口 (v3.0 - 完全重构版)
===========================================
核心设计：
  - 内置 GROMACS → OpenMM 原生缓存转换逻辑（无需 convert_to_native.py）
  - 所有模拟强制从 output 目录的缓存文件加载，避免 GROMACS 解析的副作用
  - 支持 JSON 配置文件，命令行参数优先级最高
  - 智能断点续传：自动检测各阶段状态
  - 修复多个已知 bug：配体索引提取、坐标类型转换、Boresch 参数清洗等
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import hashlib
import glob
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import openmm
from openmm import app, unit, Vec3, XmlSerializer

# 确保只导入所需模块，避免循环依赖
# 删除 runabfe.py 中手写的 NumpyEncoder 类
# 修改导入语句：
from abfe_core import (
    ACESoftcorePotential, UnitFormatter, calculate_boresch_analytical_correction,
    calc_boresch_from_last_frame, GeometricRestraintEstimator, OrbBoreschEstimator,
    DEXPSurrogatePotential, LambdaDependentBoreschForce, ensure_owned_system,
    NumpyEncoder, THERMODYNAMIC_CYCLE_DOC, assess_boresch_harmonicity,  # ✅ 统一从 abfe_core 导入
    combine_binding_free_energy,  # [ATT-09] 热力学循环闭合的唯一实现
    solvent_box_edge_nm, SOLVENT_NONBONDED_CUTOFF_NM,  # 溶剂盒唯一实现，勿在此重复
    resolve_water_model_xml,  # 溶剂腿水模型必须从复合物 .top 反推
    resolve_membrane_protocol,  # 膜体系协议唯一解析实现（B1）
    resolve_environment_type,  # 只看 config 的环境类型规范化（不需要 topology）
    resolve_charge_treatment,  # 净电荷处理协议唯一校验实现（B2）
    CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,  # B3 生产路线；溶剂腿 builder 是 B4
    CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED,  # B4 状态，进 provenance 与开跑前告警
    resolve_dispersion_protocol,  # LJ/色散路线唯一校验实现（B6）
    resolve_forcefield_family,  # 力场族识别唯一实现（§1.1）
    acceptance_thresholds_payload,  # §13 阈值快照，进 provenance
    validate_membrane_input,  # §3.3 膜输入核对唯一实现
    resolve_membrane_quality_gate_mode,  # §9 质量门 enforce / advisory
    MEMBRANE_MIN_EQUILIBRATION_NS,  # §9/§15 膜预平衡时长下限
    resolve_gromacs_include,  # include 解析唯一实现
    parse_gromacs_topology,  # `.top` 解析（[ defaults ] / [ moleculetype ] / [ molecules ]）
    classify_system_composition,  # 组成驱动的身份判定，替代残基名硬编码
    gromacs_topology_has_funct2_pairs,  # OpenMM 不支持 [ pairs ] funct 2
    convert_gromacs_pairs_funct2,  # 带逐对等价校验的兼容性转换
    openmm_compatible_gromacs_top,  # 需要时转换，返回可交给 OpenMM 的 .top
    load_gromacs_topology_for_openmm,  # **唯一**的拓扑加载入口
    GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION,
)
from abfe_pipeline import (
    ABFEPipeline, TraditionalABFEPipeline, _collect_pipeline_provenance, _pme_u_kn_meta_payload,
    _pre_equilibration_fingerprint,
)
from ibs_engine import (
    solve_stage_integrated,
    generate_overlapping_windows,
    pme_self_correction_prefactor_kj,
    pme_self_correction_energy_kj,
    lambda_endpoint_diagnostics,
    synthetic_mbar_u_kn,
    TraditionalMBARAnalyzer,
    # 配体净电荷的**唯一**实现。B2 的校验层刻意只吃一个 float，不自己再数一遍
    # 电荷，避免出现第二套净电荷判据（abfe_core 在 ibs_engine 下层，不能反向 import）。
    _compute_ligand_net_charge,
) # ✅ 保持从 ibs_engine 导入
from abfe_preoptimizer import DualLambdaPreOptimizer, build_aces_probe_system_dual_lambda

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("runabfe")

# ---------------------------------------------------------------------------
# 常量与预设
# ---------------------------------------------------------------------------
PRESET_CONFIGS = {
    "test": {
        "n_steps_per_window": 10000,
        "steps_per_update": 500,
        "stage1_n_states": 12,
        "stage2_n_states": 17,
    },
    "production": {
        "n_steps_per_window": 250000,
        "steps_per_update": 500,
        # 去电荷腿走 PME-REMD（每个 λ 状态各建一个 replica context），
        # 状态数越多同时常驻显存的 replica 越多；8 个状态相邻窗口能量差过大、
        # 容易在交换/重加权时能量过高，12 个状态更稳妥，仍不需要和 VDW 阶段一样多。
        "stage1_n_states": 12,
        "stage2_n_states": 17,
    },
    "high_accuracy": {
        "n_steps_per_window": 500000,
        "steps_per_update": 500,
        "stage1_n_states": 24,
        "stage2_n_states": 17,
    },
}

MAIN_SYSTEM_CACHE_PROTOCOL_VERSION = 2
# version 4: 溶剂盒不再交给 addSolvent(padding=...) 自己推导。旧路径在本仓库
#   产出过 box = 2*padding 的 3.000 nm 立方盒——溶质尺寸对盒长的贡献是 0，而
#   配体最长轴 1.257 nm，每侧只剩 0.87 nm 溶剂，第二水化层直接和周期镜像重叠。
#   所有 v3 及更早的溶剂缓存都是在那个盒子里建的，必须整体作废重建。
SOLVENT_CACHE_PROTOCOL_VERSION = 4
DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR = 0.15
SOLVENT_PADDING_NM = 1.5


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_gromacs_include_path(
    include_name: str,
    including_file: str,
    gmx_include_dir: Optional[str],
) -> str:
    """薄包装：解析顺序的**唯一实现**在 `abfe_core.resolve_gromacs_include`。

    保留这个名字是因为 `_gromacs_dependency_hashes` 与主缓存指纹都在用它；
    实现收敛到一处，避免两份解析顺序日后各改一半。
    """
    return resolve_gromacs_include(include_name, including_file, gmx_include_dir)


def _gromacs_dependency_hashes(
    top_file: str,
    gmx_include_dir: Optional[str],
) -> List[Dict[str, str]]:
    """Hash the complete transitive set of quoted GROMACS ``#include`` files."""
    pending = [os.path.realpath(top_file)]
    seen = set()
    dependencies: List[Dict[str, str]] = []
    include_re = re.compile(r'^\s*#\s*include\s+"([^"]+)"')
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        if not os.path.isfile(path):
            raise FileNotFoundError(f"GROMACS 拓扑依赖不存在: {path}")
        seen.add(path)
        dependencies.append({"path": path, "sha256": _sha256_file(path)})
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = include_re.match(line)
                if match:
                    pending.append(
                        _resolve_gromacs_include_path(
                            match.group(1), path, gmx_include_dir
                        )
                    )
    return sorted(dependencies, key=lambda item: item["path"])


def _main_cache_identity(
    gro_file: Optional[str],
    top_file: Optional[str],
    ligand_resname: Optional[str],
    gmx_include_dir: Optional[str],
) -> Dict:
    if not gro_file or not os.path.isfile(gro_file):
        raise FileNotFoundError("主 System 缓存校验需要当前有效的 --gro 输入")
    if not top_file or not os.path.isfile(top_file):
        raise FileNotFoundError("主 System 缓存校验需要当前有效的 --top 输入")
    if not ligand_resname:
        raise ValueError("主 System 缓存校验需要当前配体残基名")
    payload = {
        "protocol_version": MAIN_SYSTEM_CACHE_PROTOCOL_VERSION,
        "gro": {
            "path": os.path.realpath(gro_file),
            "sha256": _sha256_file(gro_file),
        },
        "topology_dependencies": _gromacs_dependency_hashes(
            top_file, gmx_include_dir
        ),
        "ligand_resname": str(ligand_resname),
        "system_build_parameters": {
            # OpenMM 不支持 `[ pairs ]` funct 2，含它的拓扑会先经等价转换再交给
            # GromacsTopFile。指纹按**原始**输入算（topology_dependencies 上面那项），
            # 但转换逻辑本身也决定最终 System，所以它的版本号必须进指纹——
            # 否则改了转换而复用旧 System 缓存就是静默串协议。
            "gromacs_pairs_funct2_conversion_version": (
                GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION
            ),
            "nonbonded_method": "PME",
            "nonbonded_cutoff_nm": 1.0,
            "constraints": "HBonds",
            "rigid_water": True,
            "ewald_error_tolerance": 0.0005,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {"identity": payload, "identity_sha256": hashlib.sha256(encoded.encode()).hexdigest()}


def _ligand_parameter_identity(
    system: openmm.System,
    topology: app.Topology,
    ligand_indices: List[int],
    ligand_resname: str,
    top_file: Optional[str],
    ligand_ffxml: Optional[str],
    gmx_include_dir: Optional[str],
    padding_nm: float = SOLVENT_PADDING_NM,
) -> Dict:
    nb_force = next(
        (force for force in system.getForces() if isinstance(force, openmm.NonbondedForce)),
        None,
    )
    if nb_force is None:
        raise RuntimeError("无法生成溶剂腿缓存身份：complex System 缺少 NonbondedForce")
    indices = sorted(int(i) for i in ligand_indices)
    topology_ligand = [
        {
            "index": int(atom.index),
            "name": str(atom.name),
            "element": str(atom.element.symbol if atom.element is not None else ""),
            "residue": str(atom.residue.name),
        }
        for atom in topology.atoms()
        if int(atom.index) in set(indices)
    ]
    params = []
    total_charge = 0.0
    for idx in indices:
        charge, sigma, epsilon = nb_force.getParticleParameters(idx)
        q = float(charge.value_in_unit(unit.elementary_charge))
        total_charge += q
        params.append(
            [
                idx,
                round(q, 12),
                round(float(sigma.value_in_unit(unit.nanometer)), 12),
                round(float(epsilon.value_in_unit(unit.kilojoule_per_mole)), 12),
            ]
        )
    explicit_ffxml = (
        {"path": os.path.realpath(ligand_ffxml), "sha256": _sha256_file(ligand_ffxml)}
        if ligand_ffxml and os.path.isfile(ligand_ffxml)
        else None
    )
    payload = {
        "complex_system_xml_sha256": hashlib.sha256(
            XmlSerializer.serialize(system).encode("utf-8")
        ).hexdigest(),
        "ligand_resname": str(ligand_resname),
        "ligand_indices": indices,
        "ligand_atom_count": len(indices),
        "ligand_net_charge_e": round(total_charge, 10),
        "ligand_topology": topology_ligand,
        "ligand_nonbonded_parameters": params,
        "gromacs_topology_dependencies": (
            _gromacs_dependency_hashes(top_file, gmx_include_dir)
            if top_file and os.path.isfile(top_file)
            else None
        ),
        "explicit_ligand_ffxml": explicit_ffxml,
        # 从复合物 .top 反推，而不是写死——两腿换了水模型必须让缓存失效。
        "solvent_forcefield": ["amber14-all.xml", resolve_water_model_xml(top_file)[0]],
        "padding_nm": float(padding_nm),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {"identity": payload, "identity_sha256": hashlib.sha256(encoded.encode()).hexdigest()}

# ---------------------------------------------------------------------------
# 工具函数：GROMACS 力场路径探测
# ---------------------------------------------------------------------------
def find_gmx_include_dir(user_path: Optional[str] = None) -> Optional[str]:
    """智能查找 GROMACS 力场 include 目录"""
    if user_path and os.path.exists(user_path):
        return user_path

    env_path = os.environ.get("GMXDATA")
    if env_path and os.path.exists(env_path):
        top_path = os.path.join(env_path, "top")
        return top_path if os.path.exists(top_path) else env_path

    try:
        gmx_bin = (
            subprocess.check_output(["which", "gmx"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        gmx_root = os.path.abspath(os.path.join(os.path.dirname(gmx_bin), ".."))
        ff_path = os.path.join(gmx_root, "share/gromacs/top")
        if os.path.exists(ff_path):
            return ff_path
    except Exception:
        pass

    for path in [
        "/home/ruigengji/gmx25.1/share/gromacs/top",  
        "/home/ruigengji/gmx26.1_avx2/share/gromacs/top",      
        "/usr/local/gromacs/share/gromacs/top",
        "/usr/share/gromacs/top",
        "/opt/gromacs/share/gromacs/top",
    ]:
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# 状态检测函数
# ---------------------------------------------------------------------------
def system_cache_exists(
    output_dir: str,
    gro_file: Optional[str] = None,
    top_file: Optional[str] = None,
    ligand_resname: Optional[str] = None,
    gmx_include_dir: Optional[str] = None,
) -> bool:
    """Only accept a native cache bound to the complete current GROMACS input."""
    xml = os.path.join(output_dir, "system_native.xml")
    idx = os.path.join(output_dir, "ligand_indices.json")
    top = os.path.join(output_dir, "topology.cif")
    box = os.path.join(output_dir, "box_vectors.npy")
    manifest_path = os.path.join(output_dir, "system_cache_manifest.json")
    if not all(os.path.isfile(path) for path in (xml, idx, top, box, manifest_path)):
        return False
    try:
        expected = _main_cache_identity(
            gro_file, top_file, ligand_resname, gmx_include_dir
        )
        with open(manifest_path, encoding="utf-8") as handle:
            recorded = json.load(handle)
    except Exception as exc:
        log.warning("⚠️ 主 System 缓存身份无法校验 (%s)，将重建", exc)
        return False
    matches = (
        recorded.get("protocol_version") == MAIN_SYSTEM_CACHE_PROTOCOL_VERSION
        and recorded.get("identity_sha256") == expected["identity_sha256"]
        and recorded.get("system_xml_sha256") == _sha256_file(xml)
        and recorded.get("ligand_indices_sha256") == _sha256_file(idx)
        and recorded.get("topology_sha256") == _sha256_file(top)
        and recorded.get("box_vectors_sha256") == _sha256_file(box)
    )
    if not matches:
        log.warning("⚠️ 主 System 缓存与当前 GRO/TOP/include/配体/构建参数不匹配，将重建")
    return bool(matches)


def solvent_cache_exists(
    output_dir: str,
    ionic_strength_molar: float = DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR,
    expected_identity: Optional[Dict] = None,
) -> bool:
    """Only accept solvent caches built with the requested explicit salt."""
    xml = os.path.join(output_dir, "system_solvent.xml")
    idx = os.path.join(output_dir, "ligand_indices_solvent.json")
    top = os.path.join(output_dir, "topology_solvent.cif")
    manifest_path = os.path.join(output_dir, "solvent_cache_manifest.json")
    if not all(os.path.isfile(path) for path in (xml, idx, top, manifest_path)):
        return False
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception as exc:
        log.warning("⚠️ 溶剂腿缓存 manifest 不可读 (%s)，将重建", exc)
        return False
    expected_strength = float(ionic_strength_molar)
    matches = (
        manifest.get("protocol_version") == SOLVENT_CACHE_PROTOCOL_VERSION
        and abs(float(manifest.get("ionic_strength_molar", -1.0)) - expected_strength)
        <= 1.0e-12
        and bool(manifest.get("neutralize"))
        and expected_identity is not None
        and manifest.get("identity_sha256") == expected_identity.get("identity_sha256")
        and manifest.get("system_xml_sha256") == _sha256_file(xml)
        and manifest.get("ligand_indices_sha256") == _sha256_file(idx)
        and manifest.get("topology_sha256") == _sha256_file(top)
    )
    if expected_strength > 0.0:
        matches = matches and int(manifest.get("na_count", 0)) > 0
        matches = matches and int(manifest.get("cl_count", 0)) > 0
    if not matches:
        log.warning(
            "⚠️ 已有溶剂腿缓存不是当前显式盐协议 "
            "(v%d, %.3f M NaCl)，将自动重建",
            SOLVENT_CACHE_PROTOCOL_VERSION,
            expected_strength,
        )
    return bool(matches)


def equilibrium_is_done(output_dir: str, expected_fingerprint: Optional[str] = None) -> bool:
    """判断预平衡是否完成（存在轨迹和 checkpoint）。

    expected_fingerprint（可选）：由
    abfe_pipeline._pre_equilibration_fingerprint(system, ligand_indices,
    temperature, pressure) 算出的当前 system/config 指纹。此前这里只按文件
    存在与否判断，同一个 --output 目录换了 gro/top/ligand/温度/压力再跑
    （没有 --reset）时，会把上一次配置留下的旧轨迹静默当作"已完成"复用。
    传入这个参数后，还会核对 pre_equilibration_fingerprint.json（由
    pre_equilibrate() 写出）里记录的指纹是否与当前配置一致；不传时保持旧的
    纯文件存在性判断（向后兼容，供尚未持有 system/ligand_indices 的调用点
    使用）。
    """
    traj = os.path.join(output_dir, "pre_equilibration.dcd")
    chk = os.path.join(output_dir, "checkpoints", "pre_equil.chk")
    if not (os.path.isfile(traj) and os.path.getsize(traj) > 10000 and os.path.isfile(chk)):
        return False
    if expected_fingerprint is None:
        return True
    fp_file = os.path.join(output_dir, "pre_equilibration_fingerprint.json")
    if not os.path.isfile(fp_file):
        log.warning(
            "⚠️ 预平衡轨迹/Checkpoint 存在，但缺少 pre_equilibration_fingerprint.json，"
            "无法确认是否匹配当前 system/config（可能来自本次修复之前的旧运行），"
            "保守视为未完成，将重新执行预平衡。"
        )
        return False
    try:
        with open(fp_file) as f:
            recorded = json.load(f).get("fingerprint")
    except Exception as e:
        log.warning("⚠️ 读取 pre_equilibration_fingerprint.json 失败 (%s)，保守视为未完成", e)
        return False
    if recorded != expected_fingerprint:
        log.warning(
            "⚠️ 已有预平衡轨迹的指纹与当前 system/config 不匹配（可能是换了 gro/top/ligand/"
            "温度或目标步数但复用了同一个 --output 目录），拒绝复用，将重新执行预平衡。"
        )
        return False

    # ---- 真的跑完了吗（MEM-09，2026-08-02）----
    #
    # ⚠️ 到这里为止的判据全部是 **fail-open** 的：轨迹与 checkpoint 只要存在就算数，
    # 而 `pre_equilibration_fingerprint.json` 是 `pre_equilibrate()` 在**第一步之前**
    # 就写下的（`abfe_pipeline.py`，为的是让被中断的运行下次能认出自己的身份），
    # 它记的 `n_steps` 是**目标**步数，不是已完成步数。
    #
    # 于是一次被中断的运行（例如 100 ns 只跑到 40 ns）在下一次调用里会被判成
    # "已完成"，`pre_equilibrate()` 整段跳过 —— 连同它内部的 §9 膜质量门一起跳过，
    # 而 provenance 与 fingerprint 都写着 100 ns。这是"短平衡冒充长平衡"，
    # 且 `enforce` 模式根本没机会拦。
    #
    # 唯一可信的完成标记是 `pipeline_state.json` 里 equilibration 的
    # `status=completed` —— 它只在步进真正结束之后才写。
    state_file = os.path.join(output_dir, "checkpoints", "pipeline_state.json")
    if not os.path.isfile(state_file):
        log.warning(
            "⚠️ 预平衡轨迹/Checkpoint/指纹都在，但缺少 checkpoints/pipeline_state.json，"
            "无法确认预平衡**真的跑完了**（指纹是第一步之前就写的，只代表目标）。"
            "保守视为未完成。加 --resume 可从 Checkpoint 续跑，不会白烧已完成的部分。"
        )
        return False
    try:
        with open(state_file, encoding="utf-8") as f:
            equil_state = (json.load(f).get("stages") or {}).get("equilibration") or {}
    except Exception as e:
        log.warning("⚠️ 读取 pipeline_state.json 失败 (%s)，保守视为预平衡未完成", e)
        return False
    if equil_state.get("status") != "completed":
        log.warning(
            "⚠️ 上一次预平衡**没有跑完**（pipeline_state.json 里 equilibration.status=%r）。"
            "拒绝把它当成已完成的平衡复用 —— 那会让一段短平衡冒充目标时长，"
            "并连带跳过 §9 膜质量门。加 --resume 可从 Checkpoint 续跑。",
            equil_state.get("status"),
        )
        return False
    try:
        with open(fp_file, encoding="utf-8") as handle:
            recorded_target = int(json.load(handle).get("n_steps", 0))
    except Exception:
        recorded_target = 0
    achieved = int(equil_state.get("total_steps") or 0)
    if recorded_target and achieved < recorded_target:
        log.warning(
            "⚠️ 已完成的预平衡只有 %d 步，而当前目标是 %d 步。拒绝复用，"
            "加 --resume 可从 Checkpoint 续跑剩下的部分。",
            achieved, recorded_target,
        )
        return False
    return True


def boresch_params_ready(output_dir: str, source: str) -> bool:
    """检查 Boresch 参数文件是否已生成"""
    if source in ("auto", "simple", "orb_simple", "fluctuation"):
        return os.path.isfile(os.path.join(output_dir, f"boresch_{source}.json"))
    return False


def _cache_paths(output_dir: str, phase: str = "complex") -> Dict[str, str]:
    if phase == "solvent":
        return {
            "xml": os.path.join(output_dir, "system_solvent.xml"),
            "idx": os.path.join(output_dir, "ligand_indices_solvent.json"),
            "box": os.path.join(output_dir, "box_vectors_solvent.npy"),
            "top": os.path.join(output_dir, "topology_solvent.cif"),
            "runtime_dir": os.path.join(output_dir, "solvent_leg"),
        }
    return {
        "xml": os.path.join(output_dir, "system_native.xml"),
        "idx": os.path.join(output_dir, "ligand_indices.json"),
        "box": os.path.join(output_dir, "box_vectors.npy"),
        "top": os.path.join(output_dir, "topology.cif"),
        "runtime_dir": output_dir,
    }


def _get_residue_name_by_atom_index(topology: app.Topology, atom_index: int) -> str:
    atoms = list(topology.atoms())
    if atom_index < 0 or atom_index >= len(atoms):
        raise IndexError(f"原子索引越界: {atom_index}")
    return atoms[atom_index].residue.name


def resolve_ligand_ffxml(output_dir: str, ligand_resname: str, explicit_path: Optional[str] = None) -> Optional[str]:
    """定位配体力场 XML/FFXML，优先用户显式指定，其次尝试 output 下常见命名。"""
    candidates: List[Optional[str]] = [
        explicit_path,
        os.path.join(output_dir, "ligand.xml"),
        os.path.join(output_dir, "ligand.ffxml"),
        os.path.join(output_dir, f"{ligand_resname}.xml"),
        os.path.join(output_dir, f"{ligand_resname}.ffxml"),
        os.path.join(output_dir, "ffxml", "ligand.xml"),
        os.path.join(output_dir, "ffxml", "ligand.ffxml"),
        os.path.join(output_dir, "ffxml", f"{ligand_resname}.xml"),
        os.path.join(output_dir, "ffxml", f"{ligand_resname}.ffxml"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _build_stable_ligand_atom_names(gmx_top: app.GromacsTopFile, ligand_resname: str) -> Dict[int, str]:
    """基于 .top 的原始原子顺序生成稳定且唯一的配体原子名映射。"""
    lig_atoms = [a for a in gmx_top.topology.atoms() if a.residue.name == ligand_resname]
    if not lig_atoms:
        raise ValueError(f"未在拓扑中找到配体残基: {ligand_resname}")

    raw_names = [atom.name.strip() or atom.element.symbol for atom in lig_atoms]
    if len(set(raw_names)) == len(raw_names):
        return {atom.index: raw for atom, raw in zip(lig_atoms, raw_names)}

    elem_counter: Dict[str, int] = {}
    stable_names: Dict[int, str] = {}
    for atom in lig_atoms:
        elem = atom.element.symbol if atom.element is not None else "X"
        elem_counter[elem] = elem_counter.get(elem, 0) + 1
        stable_names[atom.index] = f"{elem}{elem_counter[elem]}"
    return stable_names


def _resolve_mdtraj_topology_input(output_dir: str, gro_file: Optional[str], top_file: Optional[str]) -> str:
    """为 mdtraj.load 解析最稳妥的拓扑输入，优先缓存的 mmCIF。"""
    cif_path = os.path.join(output_dir, "topology.cif")
    if os.path.isfile(cif_path):
        return cif_path
    if gro_file and os.path.isfile(gro_file):
        return gro_file
    if top_file and os.path.isfile(top_file):
        return top_file
    raise FileNotFoundError(
        "无法为轨迹加载解析拓扑：未找到 topology.cif 缓存，也未提供有效的 --gro/--top。"
    )


def generate_ligand_xml_from_top(
    top_file: str,
    ligand_resname: str,
    output_dir: str,
    gmx_include_dir: Optional[str] = None,
) -> str:
    """从 GROMACS .top 中抽取配体参数，生成可被 OpenMM ForceField 加载的 ligand_only.xml。"""
    log.info("🔹 从 GROMACS 拓扑抽取配体参数: 生成 ligand_only.xml")
    # 走唯一入口（见 abfe_core.load_gromacs_topology_for_openmm）。
    top = load_gromacs_topology_for_openmm(top_file, includeDir=gmx_include_dir)
    lig_res = next((r for r in top.topology.residues() if r.name == ligand_resname), None)
    if lig_res is None:
        raise ValueError(f"未在拓扑中找到配体残基: {ligand_resname}")

    lig_idx = [a.index for a in lig_res.atoms()]
    stable_names = _build_stable_ligand_atom_names(top, ligand_resname)
    global_to_local = {g: l for l, g in enumerate(lig_idx)}
    atoms = list(top.topology.atoms())
    lig_set = set(lig_idx)
    bond_neighbors: Dict[int, set] = {idx: set() for idx in lig_idx}
    for bond in top.topology.bonds():
        i, j = bond.atom1.index, bond.atom2.index
        if i in lig_set and j in lig_set:
            bond_neighbors[i].add(j)
            bond_neighbors[j].add(i)

    # 不再依赖 GromacsTopFile 的私有字段；直接用 createSystem() 后的真实 OpenMM 力参数。
    extracted_system = top.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None,
        rigidWater=False,
    )
    nb_force = next(
        (f for f in extracted_system.getForces() if isinstance(f, openmm.NonbondedForce)),
        None,
    )
    bond_force = next(
        (f for f in extracted_system.getForces() if isinstance(f, openmm.HarmonicBondForce)),
        None,
    )
    angle_force = next(
        (f for f in extracted_system.getForces() if isinstance(f, openmm.HarmonicAngleForce)),
        None,
    )
    torsion_force = next(
        (f for f in extracted_system.getForces() if isinstance(f, openmm.PeriodicTorsionForce)),
        None,
    )
    if nb_force is None:
        raise RuntimeError("GROMACS 拓扑构建出的 System 中未找到 NonbondedForce，无法抽取配体参数")

    def classify_torsion(p1: int, p2: int, p3: int, p4: int) -> Tuple[str, Tuple[int, int, int, int]]:
        atoms4 = (p1, p2, p3, p4)
        if (
            p2 in bond_neighbors.get(p1, ())
            and p3 in bond_neighbors.get(p2, ())
            and p4 in bond_neighbors.get(p3, ())
        ):
            return "Proper", atoms4

        for center in atoms4:
            outer = [x for x in atoms4 if x != center]
            if all(x in bond_neighbors.get(center, ()) for x in outer):
                return "Improper", (center, outer[0], outer[1], outer[2])

        return "Proper", atoms4

    unique_names: Dict[int, str] = {int(i): stable_names[int(i)] for i in lig_idx}

    xml_lines = ["<ForceField>", "  <AtomTypes>"]
    for i in lig_idx:
        elem = atoms[i].element
        mass = elem.mass.value_in_unit(unit.dalton) if hasattr(elem, "mass") else 12.0
        xml_lines.append(f'    <Type name="T{i}" class="T{i}" element="{elem.symbol}" mass="{mass:.6f}"/>')
    xml_lines += ["  </AtomTypes>", "  <Residues>", f'    <Residue name="{ligand_resname}">']
    for i in lig_idx:
        charge = nb_force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
        xml_lines.append(f'      <Atom name="{unique_names[i]}" type="T{i}" charge="{charge:.8f}"/>')

    for bond in top.topology.bonds():
        i, j = bond.atom1.index, bond.atom2.index
        if i in global_to_local and j in global_to_local:
            xml_lines.append(
                f'      <Bond from="{global_to_local[i]}" to="{global_to_local[j]}"/>'
            )
    xml_lines += ["    </Residue>", "  </Residues>"]

    xml_lines.append("  <HarmonicBondForce>")
    if bond_force is not None:
        for bidx in range(bond_force.getNumBonds()):
            p1, p2, length, k = bond_force.getBondParameters(bidx)
            p1 = int(p1)
            p2 = int(p2)
            if p1 in unique_names and p2 in unique_names:
                xml_lines.append(
                    f'    <Bond class1="T{p1}" class2="T{p2}" '
                    f'length="{length.value_in_unit(unit.nanometer):.8f}" '
                    f'k="{k.value_in_unit(unit.kilojoule_per_mole / unit.nanometer**2):.8f}"/>'
                )
    xml_lines.append("  </HarmonicBondForce>")

    xml_lines.append("  <HarmonicAngleForce>")
    if angle_force is not None:
        for aidx in range(angle_force.getNumAngles()):
            p1, p2, p3, angle, k = angle_force.getAngleParameters(aidx)
            p1, p2, p3 = int(p1), int(p2), int(p3)
            if all(x in unique_names for x in (p1, p2, p3)):
                xml_lines.append(
                    f'    <Angle class1="T{p1}" class2="T{p2}" class3="T{p3}" '
                    f'angle="{angle.value_in_unit(unit.radian):.8f}" '
                    f'k="{k.value_in_unit(unit.kilojoule_per_mole / unit.radian**2):.8f}"/>'
                )
    xml_lines.append("  </HarmonicAngleForce>")

    xml_lines.append("  <PeriodicTorsionForce>")
    if torsion_force is not None:
        for tidx in range(torsion_force.getNumTorsions()):
            p1, p2, p3, p4, periodicity, phase, k = torsion_force.getTorsionParameters(tidx)
            p1, p2, p3, p4 = int(p1), int(p2), int(p3), int(p4)
            if all(x in unique_names for x in (p1, p2, p3, p4)):
                torsion_tag, ordered = classify_torsion(p1, p2, p3, p4)
                t1, t2, t3, t4 = ordered
                xml_lines.append(
                    f'    <{torsion_tag} class1="T{t1}" class2="T{t2}" class3="T{t3}" class4="T{t4}" '
                    f'periodicity="{int(periodicity)}" '
                    f'phase="{phase.value_in_unit(unit.radian):.8f}" '
                    f'k="{k.value_in_unit(unit.kilojoule_per_mole):.8f}"/>'
                )
    xml_lines.append("  </PeriodicTorsionForce>")

    xml_lines.append('  <NonbondedForce coulomb14scale="0.833333" lj14scale="0.5">')
    for i in lig_idx:
        charge, sigma, epsilon = nb_force.getParticleParameters(i)
        xml_lines.append(
            f'    <Atom type="T{i}" '
            f'charge="{charge.value_in_unit(unit.elementary_charge):.8f}" '
            f'sigma="{sigma.value_in_unit(unit.nanometer):.8f}" '
            f'epsilon="{epsilon.value_in_unit(unit.kilojoule_per_mole):.8f}"/>'
        )
    xml_lines.append("  </NonbondedForce>")
    xml_lines.append("</ForceField>")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ligand_only.xml")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(xml_lines))
    log.info("✅ 已生成配体 XML: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# 原生系统缓存读写
# ---------------------------------------------------------------------------
def diagnose_14_scaling(system: openmm.System) -> None:
    """打印 NonbondedForce 的 1-4 缩放因子，验证 GROMACS 继承正确性。"""
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nb = force
            break
    else:
        log.warning("  ⚠️ 未找到 NonbondedForce，无法诊断 1-4 缩放")
        return

    # 收集所有异常对的参数
    charge_prods = []
    lj_scales = []
    for i in range(nb.getNumExceptions()):
        p1, p2, cp, sigma, eps = nb.getExceptionParameters(i)
        # 获取两个原子的原始参数
        q1, sig1, eps1 = nb.getParticleParameters(int(p1))
        q2, sig2, eps2 = nb.getParticleParameters(int(p2))
        q_prod_raw = (q1 * q2).value_in_unit(unit.elementary_charge**2)
        if abs(q_prod_raw) > 1e-10:
            charge_prods.append(cp.value_in_unit(unit.elementary_charge**2) / q_prod_raw)
        # LJ 缩放近似为 epsilon_14 / (sqrt(eps1*eps2))
        eps_cross = np.sqrt(eps1.value_in_unit(unit.kilojoule_per_mole) * eps2.value_in_unit(unit.kilojoule_per_mole))
        if eps_cross > 1e-10:
            lj_scales.append(eps.value_in_unit(unit.kilojoule_per_mole) / eps_cross)

    fudgeQQ = np.mean(charge_prods) if charge_prods else 0.0
    fudgeLJ = np.mean(lj_scales) if lj_scales else 0.0
    log.info("  🔍 1-4 缩放诊断: fudgeQQ=%.4f (应 ~0.8333), fudgeLJ=%.4f (应 ~0.5)", fudgeQQ, fudgeLJ)

def save_native_system(
    output_dir,
    system,
    topology,
    ligand_indices,
    positions,
    box_vectors,
    cache_identity: Optional[Dict] = None,
):
    """持久化原生 System XML 与辅助数据至 output_dir（内置 convert 功能）"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. System XML (强制 Python 所有权)
    native_sys = ensure_owned_system(system)
    xml_path = os.path.join(output_dir, "system_native.xml")
    with open(xml_path, "w") as f:
        f.write(XmlSerializer.serialize(native_sys))
    log.info("  [缓存] System XML 已保存: %s", xml_path)

    # 2. Ligand indices
    lig_path = os.path.join(output_dir, "ligand_indices.json")
    with open(lig_path, "w") as f:
        json.dump({"ligand_indices": [int(i) for i in ligand_indices]}, f, indent=2)
    log.info("  [缓存] 配体索引已保存: %s", lig_path)

    # 3. Box vectors
    if box_vectors is not None:
        box_path = os.path.join(output_dir, "box_vectors.npy")
        if hasattr(box_vectors, 'value_in_unit'):
            box_nm = box_vectors.value_in_unit(unit.nanometer)
        else:
            box_nm = box_vectors
        np.save(box_path, np.asarray(box_nm, dtype=np.float64))
        log.info("  [缓存] 盒子向量已保存: %s", box_path)

    # 4. Topology 缓存 (🔑 改用 mmCIF 格式，彻底解决 PDB >99999 原子序列号溢出与截断问题)
    top_path = os.path.join(output_dir, "topology.cif")
    try:
        app.PDBxFile.writeFile(topology, positions, top_path)
        log.info("  [缓存] 参考拓扑已保存 (mmCIF): %s", top_path)
    except Exception as e:
        log.warning("  [缓存] 拓扑 mmCIF 保存失败: %s", e)

    if cache_identity is None:
        raise ValueError("拒绝写入无输入身份的主 System 缓存")
    manifest = {
        "protocol_version": MAIN_SYSTEM_CACHE_PROTOCOL_VERSION,
        **cache_identity,
        "system_xml_sha256": _sha256_file(xml_path),
        "ligand_indices_sha256": _sha256_file(lig_path),
        "topology_sha256": _sha256_file(top_path) if os.path.isfile(top_path) else None,
        "box_vectors_sha256": (
            _sha256_file(os.path.join(output_dir, "box_vectors.npy"))
            if os.path.isfile(os.path.join(output_dir, "box_vectors.npy"))
            else None
        ),
    }
    manifest_path = os.path.join(output_dir, "system_cache_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    log.info("  [缓存] 输入身份 manifest 已保存: %s", manifest_path)

# ================= runabfe.py =================
# 完整替换 build_and_cache_solvent_leg 函数
def build_and_cache_solvent_leg(
    output_dir,
    topology,
    positions,
    ligand_indices,
    ligand_resname,
    ligand_ffxml: Optional[str] = None,
    top_file: Optional[str] = None,
    gmx_include_dir: Optional[str] = None,
    ionic_strength_molar: float = DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR,
    cache_identity: Optional[Dict] = None,
    padding_nm: float = SOLVENT_PADDING_NM,
    charge_treatment: Optional[str] = None,
):
    """
    🔑 终极纯净版：彻底抛弃 mmCIF，直接从原始 .top 提取配体并自动加水。

    ⚠️ [B4 未落地] 本 builder 只产 `ligand + water + 普通盐` 的盒子。§4.1 要求
    charge-transfer 的溶剂腿还必须显式产出 **reserved co-alchemical ion**，
    并把三类电荷来源（配体形式电荷 / 中和+盐浓度的普通离子 / reserved co-ion）
    分开登记（§4.3）。那是 B4，尚未实现，所以这里 fail closed。
    """
    # 唯一的 B4 门：放在真正建溶剂盒的这一处，而不是再在上游写一遍判据。
    if charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        raise NotImplementedError(
            "charge_treatment=co_alchemical_charge_transfer 的**溶剂腿 builder 尚未实现**"
            "（memtodolist Phase B4）。\n"
            "    复合物腿的 charging Hamiltonian 已经可用（B3，2026-08-04），可以用它做"
            "pilot / λ 阶梯重估（§6.4 明确要求用 pilot 重定 stage1 窗口数）；\n"
            "    但溶剂腿里没有 reserved co-ion，ΔG_solv 与 ΔG_cplx 的 co-ion 项就不对消，"
            "热力学循环闭不上 ⟹ **不得报出 ΔG_bind**。\n"
            "    需要的是 §4.1 那个一次性产出 "
            "`ligand + water + ordinary salt/counterions + reserved co-alchemical ion` "
            "并返回 `coion_indices` / `ordinary_ion_indices` 的 builder。\n"
            "    ⚠️ 不要为了跑通而在这里临时挑一个盐离子当 co-ion —— 那既不是"
            "charge-transfer（λ=1 端必须是中性 dummy），也会让两腿离子强度口径分叉（§4.3）。"
        )
    padding_nm = float(padding_nm)
    log.info("💧 正在构建溶剂相 (配体腿) 系统并生成缓存...")
    from openmm.app import Modeller, ForceField
    os.makedirs(output_dir, exist_ok=True)

    if cache_identity is None:
        raise ValueError("拒绝构建无 complex-leg/配体参数身份的溶剂腿缓存")
    topology_ligand_count = sum(
        1 for atom in topology.atoms() if atom.residue.name == ligand_resname
    )
    if topology_ligand_count != len(ligand_indices):
        raise ValueError(
            f"complex topology 中配体原子数 ({topology_ligand_count}) 与索引数 "
            f"({len(ligand_indices)}) 不一致"
        )

    # 1. 提取配体坐标，仅用于确认输入坐标/索引有效。盒子由 OpenMM padding
    # 语义生成，确保配体每一侧都有指定厚度的溶剂。
    if hasattr(positions, "value_in_unit"):
        pos_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    else:
        pos_nm = np.asarray(positions, dtype=np.float64)
    
    lig_coords = pos_nm[ligand_indices]
    if lig_coords.ndim != 2 or lig_coords.shape[1] != 3 or not np.all(np.isfinite(lig_coords)):
        raise ValueError("配体坐标必须是有限的 (N, 3) nm 数组")

    # 2. 🔑 降维打击：彻底抛弃 mmCIF，直接从原始 .top 读取 100% 纯净的配体拓扑
    if not top_file or not os.path.isfile(top_file):
        log.error("❌ 构建溶剂相必须提供原始复合物 .top 文件以提取纯净配体键连接")
        return False
        
    log.info("  🧬 彻底抛弃 mmCIF，从原始 .top 重建 100% 纯净配体拓扑...")
    # 走唯一入口：溶剂腿也会读复合物 .top，同样可能撞上 `[ pairs ]` funct 2。
    # （首次修 funct-2 时只接了 build_system_from_gromacs，这里就漏了。）
    gmx_top = load_gromacs_topology_for_openmm(top_file, includeDir=gmx_include_dir)
    stable_names = _build_stable_ligand_atom_names(gmx_top, ligand_resname)
    
    # 获取 .top 中的配体原子索引
    top_lig_indices = [a.index for a in gmx_top.topology.atoms() if a.residue.name == ligand_resname]
    if not top_lig_indices:
        log.error(f"❌ 在 .top 文件中未找到配体残基 {ligand_resname}")
        return False
        
    # 使用 .top 的绝对正确拓扑 + 全局坐标创建 Modeller
    modeller = Modeller(gmx_top.topology, positions)
    
    # 删掉所有环境原子（蛋白、水、离子），只留配体
    atoms_to_delete = [a for a in modeller.topology.atoms() if a.index not in top_lig_indices]
    modeller.delete(atoms_to_delete)
    log.info("  ✅ 纯净配体拓扑构建完成 (键连接 100% 忠于 .top，无假键)")

    # 3. 用 .top 的稳定名字映射重命名，确保拓扑与 XML 完全同序对齐
    lig_atoms_in_mod = [a for a in modeller.topology.atoms() if a.residue.name == ligand_resname]
    if len(lig_atoms_in_mod) != len(top_lig_indices):
        log.error("❌ 配体原子数不匹配：Modeller=%d, Topology=%d", len(lig_atoms_in_mod), len(top_lig_indices))
        return False
    for atom, top_idx in zip(lig_atoms_in_mod, top_lig_indices):
        atom.name = stable_names[int(top_idx)]
    log.info("  ✅ 配体原子名已按 .top 稳定映射重命名并对齐 XML")

    # 4. 复用/生成配体 XML + TIP3P 水
    ffxml_path = resolve_ligand_ffxml(output_dir, ligand_resname, explicit_path=ligand_ffxml)
    if ffxml_path is None:
        ffxml_path = generate_ligand_xml_from_top(
            top_file=top_file,
            ligand_resname=ligand_resname,
            output_dir=output_dir,
            gmx_include_dir=gmx_include_dir,
        )
    if ffxml_path is None:
        log.error("❌ 溶剂相构建失败：未找到或无法生成配体力场 XML")
        return False

    try:
        # 🔑 水模型不硬编码，从复合物 .top 反推，保证两腿同模型。
        water_xml, water_itp = resolve_water_model_xml(top_file)
        log.info(
            "  🔍 使用 Amber14 + %s (继承自 %s) + 配体 XML 构建溶剂相: %s",
            water_xml,
            water_itp,
            ffxml_path,
        )
        ff = ForceField("amber14-all.xml", water_xml, ffxml_path)
        
        ionic_strength_molar = float(ionic_strength_molar)
        if ionic_strength_molar < 0.0:
            raise ValueError("solvent_ionic_strength_molar 不能为负")
        # 🔑 盒子显式给出，不用 padding=。addSolvent 的 padding 分支曾经产出
        # box = 2*padding 的 3.000 nm 盒（溶质尺寸贡献为 0），而且 OpenMM 7.7+
        # 的 padding 默认给菱形十二面体、不是立方——两者都不是这里想要的。
        box_edge_nm, lig_extent_nm = solvent_box_edge_nm(
            lig_coords, padding_nm=padding_nm
        )
        log.info(
            "  📦 溶剂盒: 配体最长轴 %.4f nm + 2×%.2f nm padding → 立方盒 %.4f nm",
            lig_extent_nm,
            padding_nm,
            box_edge_nm,
        )
        # Match the ionized complex leg's ~0.15 M NaCl instead of silently
        # building a pure-water solvent leg. OpenMM adds salt pairs for the
        # requested ionic strength and any extra counterions needed to neutralize.
        modeller.addSolvent(
            ff,
            boxSize=Vec3(box_edge_nm, box_edge_nm, box_edge_nm) * unit.nanometer,
            positiveIon="Na+",
            negativeIon="Cl-",
            ionicStrength=ionic_strength_molar * unit.molar,
            neutralize=True,
        )

        # fail closed：确认 OpenMM 真的按我们给的盒子建，而不是又自己推了一个。
        realized_vecs = modeller.topology.getPeriodicBoxVectors()
        if realized_vecs is None:
            raise RuntimeError("addSolvent 之后拓扑没有周期盒向量")
        realized_edges = np.linalg.norm(
            np.array(
                [v.value_in_unit(unit.nanometer) for v in realized_vecs], dtype=float
            ),
            axis=1,
        )
        if not np.allclose(realized_edges, box_edge_nm, atol=1.0e-6):
            raise RuntimeError(
                f"溶剂盒构建结果与请求不符：请求 {box_edge_nm:.6f} nm 立方，"
                f"实际边长 {[round(float(x), 6) for x in realized_edges]} nm"
            )

        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=SOLVENT_NONBONDED_CUTOFF_NM * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )
    except Exception as e:
        log.error("❌ 溶剂相构建失败 (配体 XML 未正确接入或模板不匹配): %s", e)
        log.error("💡 当前使用的配体 XML: %s", ffxml_path)
        return False

    # 5. 获取溶剂相中的新配体索引并保存缓存
    new_lig_indices = [atom.index for atom in modeller.topology.atoms() if atom.residue.name == ligand_resname]
    residue_names = [str(residue.name).upper() for residue in modeller.topology.residues()]
    na_count = sum(name in {"NA", "NA+", "SOD"} for name in residue_names)
    cl_count = sum(name in {"CL", "CL-", "CLA"} for name in residue_names)
    if ionic_strength_molar > 0.0 and (na_count == 0 or cl_count == 0):
        log.error(
            "❌ 请求 %.3f M NaCl，但生成拓扑中离子计数异常: Na=%d, Cl=%d",
            ionic_strength_molar,
            na_count,
            cl_count,
        )
        return False
    
    sol_xml = os.path.join(output_dir, "system_solvent.xml")
    sol_cif = os.path.join(output_dir, "topology_solvent.cif")
    sol_idx = os.path.join(output_dir, "ligand_indices_solvent.json")
    sol_box = os.path.join(output_dir, "box_vectors_solvent.npy")
    sol_manifest = os.path.join(output_dir, "solvent_cache_manifest.json")
    
    with open(sol_xml, "w") as f:
        f.write(XmlSerializer.serialize(ensure_owned_system(system)))
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, sol_cif)
    with open(sol_idx, "w") as f:
        json.dump({"ligand_indices": new_lig_indices}, f)
    
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    if box_vecs:
        np.save(sol_box, np.array([v.value_in_unit(unit.nanometer) for v in box_vecs]))
    with open(sol_manifest, "w", encoding="utf-8") as f:
        json.dump(
            {
                "protocol_version": SOLVENT_CACHE_PROTOCOL_VERSION,
                **cache_identity,
                "padding_nm": padding_nm,
                "ligand_longest_axis_nm": lig_extent_nm,
                "box_edge_nm": box_edge_nm,
                "nonbonded_cutoff_nm": SOLVENT_NONBONDED_CUTOFF_NM,
                "ionic_strength_molar": ionic_strength_molar,
                "positive_ion": "Na+",
                "negative_ion": "Cl-",
                "neutralize": True,
                "na_count": int(na_count),
                "cl_count": int(cl_count),
                "system_xml_sha256": _sha256_file(sol_xml),
                "topology_sha256": _sha256_file(sol_cif),
                "ligand_indices_sha256": _sha256_file(sol_idx),
            },
            f,
            indent=2,
        )
        
    box_lengths_nm = [
        float(np.linalg.norm(v.value_in_unit(unit.nanometer))) for v in box_vecs
    ]
    log.info(
        "✅ 溶剂相缓存已保存 (padding %.2f nm, 盒长 %s nm, 原子数 %d, Na=%d, Cl=%d, %.3f M)",
        padding_nm,
        [round(x, 3) for x in box_lengths_nm],
        system.getNumParticles(),
        na_count,
        cl_count,
        ionic_strength_molar,
    )
    return True

def load_native_system(
    output_dir,
    gro_file=None,
    top_file=None,
    gmx_include_dir=None,
    phase="complex",
    prefer_equilibrated: bool = True,
    expected_pre_equilibration_fingerprint: Optional[str] = None,
    require_bonded_topology: bool = False,
):
    """从 output_dir 的缓存文件加载系统（跳过 GROMACS 解析）

    ## ⚠️ `topology.cif` 会丢掉非标准残基的键

    `app.PDBxFile.writeFile` **不写任何键记录**（写入端没有 `struct_conn` /
    `chem_comp_bond`），而读取端只能靠 `createStandardBonds()` 补**标准残基**
    （氨基酸/核酸/水）的键。于是 mmCIF 往返之后：

      * 脂质（`POPC` = `PA`+`PC`+`OL`）、配体（`MOL`）、离子的键**全部丢失**；
      * 而这里原本只校验**原子数**，键数不符照样通过。

    后果是实测出来的：`pre_equilibrate` 之前的「PBC 分子完整性修复」靠 topology 的
    键判断"什么算一个分子"，键丢了就把跨周期边界的脂质**逐段撕开** ——
    最小化后 PE = 4.1e13 kJ/mol、max|F| = 3.7e9（落在脂质尾链氢 `PA334/H8S`），
    几千步后变成一个没有上下文的 `Particle coordinate is NaN`。
    从 `.top` 重建拓扑（有键）时同一体系完全正常（PE ≈ −6.5e5）。

    `require_bonded_topology=True`（膜体系必须传）会：
      1. **优先从 `.top` 重建**拓扑，mmCIF 只在没有 `.top` 时兜底；
      2. 校验键数，覆盖不足即 fail closed。

    可溶路径默认仍走原逻辑（mmCIF 优先）——它的配体 `MOL` 同样丢键，属存量隐患，
    但改动会移动现有生产基线的预平衡起点（§7.7 / R7），单独立项评估。
    """
    paths = _cache_paths(output_dir, phase=phase)
    xml_path = paths["xml"]
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Native XML 缓存不存在: {xml_path}")
    log.info("♻️ 从原生缓存加载系统 (%s): %s", phase, output_dir)

    # 1. System
    with open(xml_path, "r") as f:
        xml_str = f.read()
    system = ensure_owned_system(XmlSerializer.deserialize(xml_str))
    log.info("  ✓ System 恢复 | 原子数: %d", system.getNumParticles())

    # 2. Ligand indices
    lig_path = paths["idx"]
    with open(lig_path, "r") as f:
        lig_data = json.load(f)
    ligand_indices = lig_data["ligand_indices"]
    log.info("  ✓ 配体索引恢复: %d 原子", len(ligand_indices))

    # 3. Box vectors
    box_path = paths["box"]
    box_vectors = None
    if os.path.exists(box_path):
        box_nm = np.load(box_path)
        box_vectors = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in box_nm] * unit.nanometer
        log.info("  ✓ 盒子向量恢复")
        try:
            system.setDefaultPeriodicBoxVectors(*box_vectors)
        except Exception as e:
            log.warning("  ⚠️ 设置默认盒子失败: %s", e)

    # 4. Topology 恢复
    #
    # 膜体系（require_bonded_topology）**优先从 .top 重建**：mmCIF 不保存非标准残基
    # 的键，而 PBC 分子完整性修复依赖键。详见本函数 docstring。
    top_path = paths["top"]
    topology = None
    cif = None

    if require_bonded_topology and top_file and os.path.exists(top_file):
        try:
            inc_dir = gmx_include_dir or find_gmx_include_dir()
            _rebuilt = load_gromacs_topology_for_openmm(
                top_file, includeDir=inc_dir, periodicBoxVectors=box_vectors
            )
            topology = _rebuilt.topology
            log.info(
                "  ✓ 拓扑从原始 TOP 重建（%d 原子, %d 键）—— 膜体系不用 mmCIF，"
                "因为它不保存非标准残基的键",
                topology.getNumAtoms(), topology.getNumBonds(),
            )
        except Exception as e:
            raise RuntimeError(
                f"膜体系要求从 .top 重建带键拓扑，但失败了：{e}\n"
                "   不能退回 mmCIF —— 它丢掉脂质/配体的键，会让 PBC 分子完整性修复"
                "把跨边界的分子撕开（实测最小化后 PE 到 1e13、随后 NaN）。"
            ) from e

    if topology is None and os.path.exists(top_path):
        try:
            cif = app.PDBxFile(top_path)
            topology = cif.topology
            n_cif = topology.getNumAtoms()
            n_sys = system.getNumParticles()
            if n_cif != n_sys:
                log.warning("  ⚠️ mmCIF 拓扑原子数 (%d) 与 System (%d) 不匹配，已丢弃缓存", n_cif, n_sys)
                topology = None
            else:
                # 🔑 键数校验：只校验原子数是不够的。mmCIF 不保存非标准残基的键，
                # 读回时只有 createStandardBonds() 能补标准残基，脂质/配体/离子的键
                # 会静默丢失，而 PBC 分子完整性修复正是靠键判断分子边界。
                n_cif_bonds = topology.getNumBonds()
                n_sys_bonds = 0
                for _f in system.getForces():
                    if isinstance(_f, openmm.HarmonicBondForce):
                        n_sys_bonds = max(n_sys_bonds, _f.getNumBonds())
                # System 的键数 = HarmonicBondForce 键数 + 被 constraints 吸收的 X–H 键，
                # 所以 topology 键数应当 ≥ HarmonicBondForce 键数。明显更少就是丢了。
                if n_sys_bonds > 0 and n_cif_bonds < n_sys_bonds:
                    log.warning(
                        "  ⚠️ mmCIF 拓扑只有 %d 个键，而 System 的 HarmonicBondForce 有 %d 个"
                        " —— mmCIF 不保存非标准残基（脂质/配体/离子）的键。"
                        "PBC 分子完整性修复依赖键，键丢了会把跨边界分子撕开。"
                        "已丢弃 mmCIF 拓扑，改从 .top 重建。",
                        n_cif_bonds, n_sys_bonds,
                    )
                    topology = None
                else:
                    log.info(
                        "  ✓ 拓扑从 mmCIF 缓存恢复 (原子数 %d, 键数 %d)",
                        n_cif, n_cif_bonds,
                    )
        except Exception as e:
            log.warning("  ⚠️ mmCIF 拓扑加载失败: %s", e)

    # 降级方案：从原始 TOP 文件重建拓扑
    if topology is None and phase == "complex" and top_file and os.path.exists(top_file):
        try:
            inc_dir = gmx_include_dir or find_gmx_include_dir()
            top = load_gromacs_topology_for_openmm(top_file, includeDir=inc_dir)
            topology = top.topology
            log.info("  ✓ 拓扑从原始 TOP 文件重建")
        except Exception as e:
            log.warning("  ⚠️ TOP 拓扑重建失败: %s", e)

    if topology is None:
        raise RuntimeError(
            "无法恢复拓扑：mmCIF 缓存损坏且无有效 TOP 文件。\n"
            "   💡 解决方案：请确保命令中包含 --top 参数，或删除 output 目录后重新运行。"
        )

    # 🔑 无论拓扑从哪条路径恢复，都必须带上周期盒矢量。
    #
    # `app.DCDFile` 写每帧的 unitcell 时读的是 **topology** 的盒矢量
    # （`dcdfile.py:155`：`boxVectors = self._topology.getPeriodicBoxVectors()`；
    # 为 None 就整段不写，header 的 boxFlag 也是 0）。拓扑没盒矢量 → DCD 没 unitcell
    # → 下游任何要算 APL / 盒序列的分析（§9 膜质量门）直接失效。
    #
    # mmCIF 恢复的拓扑自带 `_cell`，所以历史上没暴露；但从 `.top` 重建时
    # `GromacsTopFile` 只在显式传 `periodicBoxVectors` 时才设盒——这是一处
    # **fail-open**：少传一个参数，产出的轨迹静默缺少盒信息。
    if box_vectors is not None and topology.getPeriodicBoxVectors() is None:
        topology.setPeriodicBoxVectors(box_vectors)
        log.info("  ✓ 已把盒矢量写入拓扑（DCD 的 unitcell 依赖它）")

    # 5. Positions (保持不变)
    positions = None
    equil_dcd = os.path.join(paths["runtime_dir"], "pre_equilibration.dcd")
    equil_cache_trusted = bool(
        prefer_equilibrated
        and expected_pre_equilibration_fingerprint is not None
        and equilibrium_is_done(
            paths["runtime_dir"],
            expected_fingerprint=expected_pre_equilibration_fingerprint,
        )
    )
    if prefer_equilibrated and expected_pre_equilibration_fingerprint is None:
        log.info(
            "  ℹ️ 未提供完整预平衡 fingerprint，不自动读取 DCD；"
            "先恢复身份绑定的初始坐标，由 pipeline 决定是否续用预平衡"
        )
    if equil_cache_trusted and os.path.exists(equil_dcd) and os.path.getsize(equil_dcd) > 212:
        try:
            import mdtraj as md
            md_top = md.Topology.from_openmm(topology)
            traj = md.load(equil_dcd, top=md_top)
            if len(traj) > 0:
                positions = traj.xyz[-1] * unit.nanometer
                if len(traj.unitcell_vectors) > 0:
                    box_vectors = traj.unitcell_vectors[-1] * unit.nanometer
                    system.setDefaultPeriodicBoxVectors(*box_vectors)
                log.info("  ✓ 坐标从预平衡 DCD 最后一帧恢复")
        except Exception as e:
            log.warning("  ⚠️ DCD 坐标加载失败: %s", e)

    if positions is None and cif is not None:
        try:
            positions = cif.positions
            log.info("  ✓ 坐标从 mmCIF 缓存恢复")
        except Exception as e:
            log.warning("  ⚠️ mmCIF 坐标恢复失败: %s", e)

    if positions is None and phase == "complex" and gro_file and os.path.exists(gro_file):
        gro = app.GromacsGroFile(gro_file)
        positions = gro.positions
        log.info("  ⚠️ 坐标回退到 GRO 初始值")

    if positions is None:
        raise RuntimeError("无法恢复坐标，请检查输入文件或预平衡结果")

    log.info("✅ 原生缓存加载完成")
    return system, topology, positions, box_vectors, ligand_indices

# ---------------------------------------------------------------------------
# GROMACS 系统构建（仅首次使用，之后都会被缓存替代）
# ---------------------------------------------------------------------------
def build_system_from_gromacs(
    gro_file: str,
    top_file: str,
    ligand_resname: str,
    gmx_include_dir: Optional[str] = None,
) -> Tuple[openmm.System, app.Topology, list, list, List[int]]:
    """从 GROMACS 文件构建 OpenMM 系统（仅用于首次创建缓存）"""
    if not os.path.exists(gro_file):
        raise FileNotFoundError(f"GRO 文件不存在: {gro_file}")
    if not os.path.exists(top_file):
        raise FileNotFoundError(f"TOP 文件不存在: {top_file}")

    log.info("📥 读取 GROMACS 文件: %s | %s", gro_file, top_file)

    # 解析坐标与盒子
    gro = app.GromacsGroFile(gro_file)
    positions = gro.positions
    box_vectors = gro.getPeriodicBoxVectors()

    # 解析拓扑
    include_dir = gmx_include_dir or find_gmx_include_dir()
    if include_dir:
        log.info("  🔍 使用 GROMACS include 目录: %s", include_dir)
    else:
        log.warning("  ⚠️ 未找到 include 目录，若拓扑含 #include 可能失败")

    try:
        top = load_gromacs_topology_for_openmm(
            top_file,
            periodicBoxVectors=box_vectors,
            includeDir=include_dir,
        )
    except ValueError as e:
        if "Could not locate #include file" in str(e):
            log.error("拓扑解析失败: %s", e)
            log.error("请通过 --gmx-path 指定包含 .ff 文件夹的父目录")
            sys.exit(1)
        raise

    omm_top = top.topology

    # 创建 System（保留自定义电荷与力常数）
    log.info("  ⚙️ 构建 OpenMM System (保留自定义 RESP/Hessian 参数)...")
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )
    if box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*box_vectors)

    # 提取配体索引（修复：正确使用 atom.index）
    all_atoms = list(omm_top.atoms())
    ligand_info = [(atom.index, atom.residue.index) for atom in all_atoms if atom.residue.name == ligand_resname]
    ligand_indices = [idx for idx, _ in ligand_info]

    if not ligand_indices:
        raise ValueError(f"未在拓扑中找到配体残基: {ligand_resname}")

    # 防御性处理同名残基
    res_indices = set(res_idx for _, res_idx in ligand_info)
    if len(res_indices) > 1:
        target_res = min(res_indices)
        ligand_indices = [idx for idx, res_idx in ligand_info if res_idx == target_res]
        log.warning("⚠️ 发现多个同名残基，仅保留残基索引 %d", target_res)

    log.info("  ✅ 系统构建完成 | 原子数: %d | 配体: %d 原子", system.getNumParticles(), len(ligand_indices))

    return system, omm_top, positions, box_vectors, ligand_indices

def center_system_rigidly(
    positions: unit.Quantity,
    box_vectors: unit.Quantity,
    ligand_indices: List[int],
) -> Tuple[unit.Quantity, unit.Quantity]:
    """
    🔑 绝对刚性居中协议 (No-Wrap Guarantee)
    1. 彻底禁用原子级 PBC Wrap，杜绝跨盒分子被撕裂。
    2. 仅计算配体质心，将整个系统作为刚体平移至盒子几何中心。
    3. 配合 CMMotionRemover，确保生产采样期间体系死死钉在盒子中央。
    """
    import numpy as np
    from openmm import Vec3
    
    pos = np.array(positions.value_in_unit(unit.nanometer))
    box = np.array([v.value_in_unit(unit.nanometer) for v in box_vectors])
    
    # 计算配体质心与盒子中心
    lig_com = np.mean(pos[ligand_indices], axis=0)
    box_center = 0.5 * np.sum(box, axis=0)
    
    # 整体刚性平移
    shift = box_center - lig_com
    pos += shift
    
    log.info("  🔒 体系已刚性锁定至盒子中心 (No-Wrap, 仅整体平移)")
    new_pos = [Vec3(float(x), float(y), float(z)) for x, y, z in pos] * unit.nanometer
    return new_pos, box_vectors
# ---------------------------------------------------------------------------
# 配置管理
# ---------------------------------------------------------------------------
# membrane.* 的 CLI 子键 → argparse 属性名。RunConfig 与 run_prepare_command 共用，
# 不各写一份（`--membrane-*` 的 argparse 默认值全是 None，所以 "is not None"
# 等价于"用户显式给了"，不需要再去翻 argv）。
_MEMBRANE_CLI_ARG_NAMES = {
    "normal_axis": "membrane_normal_axis",
    "surface_tension_bar_nm": "membrane_surface_tension_bar_nm",
    "xy_mode": "membrane_xy_mode",
    "z_mode": "membrane_z_mode",
    "barostat_frequency": "membrane_barostat_frequency",
}


def _membrane_overrides_from_args(args) -> Dict:
    """从 argparse 命名空间取出显式给出的 membrane.* 子键。"""
    overrides = {}
    for key, attr in _MEMBRANE_CLI_ARG_NAMES.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides[key] = value
    return overrides


def _load_config(config_path: str) -> dict:
    """加载配置文件，支持 .json / .yaml / .yml 格式"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"显式配置文件不存在或不是普通文件: {config_path}")
    ext = os.path.splitext(config_path)[1].lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except ImportError:
            raise ImportError(
                "读取 .yaml 配置文件需要 pyyaml，请运行: pip install pyyaml\n"
                "或使用 .json 格式的配置文件。"
            )
    else:
        with open(config_path) as f:
            config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"配置文件顶层必须是 JSON/YAML 对象，实际为 {type(config).__name__}: "
            f"{config_path}"
        )
    return config


def _load_json_object_file(path: str, label: str) -> Dict:
    """Load an explicitly requested JSON object without silent fallback."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"显式 {label} 文件不存在或不是普通文件: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取或解析 {label} 文件 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"{label} 文件顶层必须是 JSON 对象，实际为 {type(value).__name__}: {path}"
        )
    return value


def _load_dexp_params_fail_closed(
    potential_type: str,
    params_path: Optional[str],
) -> Optional[Dict]:
    """Load the narrow production DEXP contract, or return None for other potentials."""
    if str(potential_type).strip().lower() != "dexp":
        return None
    if not params_path:
        raise ValueError(
            "potential='dexp' 必须显式提供 --dexp-params JSON；"
            "拒绝静默使用默认参数。"
        )
    dexp_dict = _load_json_object_file(params_path, "DEXP 参数")
    return DEXPSurrogatePotential.from_dict(dexp_dict).get_parameters_dict()


class RunConfig:
    """统一运行时配置，优先级：命令行 > 配置文件 > 预设"""

    def __init__(self, args: argparse.Namespace):
        argv = sys.argv[1:]

        def _flag_present(*flags: str) -> bool:
            for flag in flags:
                if flag in argv:
                    return True
                prefix = f"{flag}="
                if any(token.startswith(prefix) for token in argv):
                    return True
            return False

        # 1. 载入预设
        preset = PRESET_CONFIGS.get(args.preset, PRESET_CONFIGS["production"]).copy()

        # 2. 合并配置文件（支持 .json / .yaml / .yml）
        if hasattr(args, "config") and args.config:
            file_conf = _load_config(args.config)
            for k, v in file_conf.items():
                if k.startswith("_"):
                    continue
                preset[k] = v
            log.info("📄 已合并配置文件: %s", args.config)

        # 3. 仅当命令行显式提供参数时才覆盖配置文件，避免 parser 默认值反向污染配置。
        if _flag_present("--resume"):
            preset["resume"] = bool(args.resume)
        if _flag_present("--reset"):
            preset["reset"] = bool(args.reset)
        if _flag_present("--n-steps-per-window"):
            preset["n_steps_per_window"] = args.n_steps_per_window
        if _flag_present("--steps-per-update"):
            preset["steps_per_update"] = args.steps_per_update
        if _flag_present("--n-states-per-stage"):
            preset["stage1_n_states"] = args.n_states_per_stage
            preset["stage2_n_states"] = args.n_states_per_stage
        # 独立覆盖，优先级高于 --n-states-per-stage（去电荷/去VDW通常需要不同的状态数）
        if _flag_present("--stage1-n-states"):
            preset["stage1_n_states"] = args.stage1_n_states
        if _flag_present("--stage2-n-states"):
            preset["stage2_n_states"] = args.stage2_n_states
        if _flag_present("--temperature"):
            preset["temperature"] = args.temperature
        if _flag_present("--solvent-ionic-strength-molar"):
            preset["solvent_ionic_strength_molar"] = (
                args.solvent_ionic_strength_molar
            )
        if _flag_present("--platform"):
            preset["platform"] = args.platform
        if _flag_present("--output"):
            preset["output"] = args.output
        if _flag_present("--mode"):
            preset["mode"] = args.mode
        if _flag_present("--decoupling"):
            preset["decoupling"] = args.decoupling
        if _flag_present("--potential"):
            preset["potential"] = args.potential
        if _flag_present("--dexp-params"):
            preset["dexp_params"] = args.dexp_params
        if _flag_present("--decharge-method"):
            preset["decharge_method"] = args.decharge_method
        if _flag_present("--gro"):
            preset["gro"] = args.gro
        if _flag_present("--top"):
            preset["top"] = args.top
        if _flag_present("--ligand"):
            preset["ligand"] = args.ligand
        if _flag_present("--ligand-xml"):
            preset["ligand_xml"] = args.ligand_xml
        if _flag_present("--gmx-path"):
            preset["gmx_path"] = args.gmx_path
        if _flag_present("--torsion-params"):
            preset["torsion_params"] = args.torsion_params
        if _flag_present("--boresch", "--no-boresch"):
            preset["boresch"] = args.boresch
        if _flag_present("--boresch-source"):
            preset["boresch_source"] = args.boresch_source
        if _flag_present("--boresch-anchors"):
            preset["boresch_anchors"] = args.boresch_anchors
        if _flag_present("--boresch-orb"):
            preset["boresch_orb"] = args.boresch_orb
        if _flag_present("--boresch-batch"):
            preset["boresch_batch"] = args.boresch_batch
        if _flag_present("--boresch-select"):
            preset["boresch_select"] = args.boresch_select
        if _flag_present("--enable-early-stop"):
            preset["enable_early_stop"] = bool(args.enable_early_stop)
        if _flag_present("--disable-warmup"):
            preset["enable_gradual_warmup"] = False
        elif _flag_present("--enable-gradual-warmup"):
            preset["enable_gradual_warmup"] = True
        if _flag_present("--warmup-steps"):
            preset["warmup_steps"] = args.warmup_steps
        if _flag_present("--rebalance-steps"):
            preset["rebalance_steps"] = args.rebalance_steps
        if _flag_present("--skip-rebalance"):
            preset["skip_rebalance"] = bool(args.skip_rebalance)
        if _flag_present("--n-workers"):
            preset["n_workers"] = args.n_workers
        if _flag_present("--parallel-stages"):
            preset["parallel_stages"] = bool(args.parallel_stages)
        if _flag_present("--n-lambda"):
            preset["n_lambda"] = args.n_lambda
        if _flag_present("--apbs-correction-kj-mol"):
            preset["apbs_correction_kJ_mol"] = args.apbs_correction_kj_mol
        if _flag_present("--apbs-correction-note"):
            preset["apbs_correction_note"] = args.apbs_correction_note
        if _flag_present("--charging-rerun-dir"):
            preset["charging_rerun_dir"] = args.charging_rerun_dir
        if _flag_present("--attachment-rerun-dir"):
            preset["attachment_rerun_dir"] = args.attachment_rerun_dir
        if _flag_present("--attachment-n-steps-per-state"):
            preset["attachment_n_steps_per_state"] = int(
                args.attachment_n_steps_per_state
            )
        if _flag_present("--attachment-n-seeds"):
            preset["attachment_n_seeds"] = int(args.attachment_n_seeds)
        if _flag_present("--attachment-lambdas"):
            preset["attachment_lambdas"] = [
                float(x) for x in str(args.attachment_lambdas).split(",") if x.strip()
            ]
        if _flag_present("--charging-max-resident-contexts"):
            preset["charging_max_resident_contexts"] = (
                args.charging_max_resident_contexts
            )
        if _flag_present("--only-complex-charging"):
            preset["only_complex_charging"] = bool(args.only_complex_charging)
        if _flag_present("--only-boresch-attachment"):
            preset["only_boresch_attachment"] = bool(args.only_boresch_attachment)

        # ---- 膜体系（B1）。配置键按 memtodolist §3.1：顶层 system_type + 嵌套 membrane.* ----
        if _flag_present("--system-type"):
            preset["system_type"] = args.system_type
        if _flag_present("--charge-treatment"):
            preset["charge_treatment"] = args.charge_treatment
        if _flag_present("--co-alchemical-ion"):
            preset["co_alchemical_ion"] = args.co_alchemical_ion
        if _flag_present("--apbs-evidence"):
            preset["apbs_evidence"] = args.apbs_evidence
        if _flag_present("--dispersion-protocol"):
            preset["dispersion_protocol"] = args.dispersion_protocol
        if _flag_present("--forcefield-family"):
            preset["forcefield_family"] = args.forcefield_family
        if _flag_present("--membrane-input-declaration"):
            preset["membrane_input_declaration"] = args.membrane_input_declaration
        if _flag_present("--force-switch-deviation-evidence"):
            preset["force_switch_deviation_evidence"] = (
                args.force_switch_deviation_evidence
            )
        if _flag_present("--membrane-quality-gate"):
            preset["membrane_quality_gate"] = args.membrane_quality_gate
        if _flag_present("--confirm-soluble-with-lipids"):
            preset["confirm_soluble_with_lipids"] = bool(
                args.confirm_soluble_with_lipids
            )
        _membrane_overrides = _membrane_overrides_from_args(args)
        if _membrane_overrides:
            # 命令行只覆盖显式给出的子键，不清空配置文件里的其余 membrane.* 设置。
            merged_membrane = dict(preset.get("membrane") or {})
            merged_membrane.update(_membrane_overrides)
            preset["membrane"] = merged_membrane

        defaults = {
            "resume": False,
            "reset": False,
            "temperature": 300.0,
            "platform": "CUDA",
            "output": "./output",
            "mode": "ibs",
            "decoupling": "dual_lambda",
            "potential": "softcore",
            "decharge_method": "pme",
            "boresch_batch": 0,
            "boresch_select": 1,
            "enable_early_stop": False,
            "enable_gradual_warmup": True,
            "warmup_steps": 500000,
            "rebalance_steps": 50000,
            "skip_rebalance": False,
            "parallel_stages": False,
            "n_lambda": 12,
            "apbs_correction_kJ_mol": 0.0,
            "apbs_correction_note": "",
            "solvent_ionic_strength_molar": DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR,
            "only_complex_charging": False,
            "charging_rerun_dir": None,
            "charging_max_resident_contexts": None,
            # [P1-17] Boresch attachment 腿 A′→A
            "only_boresch_attachment": False,
            "attachment_rerun_dir": None,
            "attachment_lambdas": None,
            "attachment_n_steps_per_state": 250000,
            "attachment_equil_steps_per_state": 50000,
            "attachment_steps_per_sample": 1000,
            "attachment_seed": 20260728,
            "attachment_n_seeds": 1,
            # 膜功能默认关闭（§7.7）：不声明时与改动前逐位一致。
            "system_type": None,
            "membrane": None,
            "confirm_soluble_with_lipids": False,
            # §9 质量门默认 enforce（严格）。
            "membrane_quality_gate": None,
            # 净电荷协议默认由配体净电荷推导（§1.2 生产默认值）。
            "charge_treatment": None,
            "co_alchemical_ion": None,
            "apbs_evidence": None,
            # LJ/色散：soluble 不声明 = legacy_uniform_density_lrc（行为不变）。
            "dispersion_protocol": None,
            "forcefield_family": None,
            "force_switch_deviation_evidence": None,
            "membrane_input_declaration": None,
        }
        for key, value in defaults.items():
            preset.setdefault(key, value)

        # 复合物腿默认启用 Boresch；若用户未显式指定来源，则默认走自动估算。
        if preset.get("boresch") is None:
            preset["boresch"] = True
        if preset.get("boresch_source") in (None, "", "traditional") and preset.get("boresch", False):
            if getattr(args, "boresch_source", None) is None and not preset.get("boresch_anchors"):
                preset["boresch_source"] = "auto"

        self.data = preset
        self.args = args  # 保留原始 args 供其它函数使用

    def __getattr__(self, item):
        # 优先从配置数据获取
        if item in self.data:
            return self.data[item]
        # 其次从命令行参数获取
        return getattr(self.args, item, None)

    def get(self, key, default=None):
        """兼容字典式获取"""
        if key in self.data:
            return self.data[key]
        return getattr(self.args, key, default)

    def as_dict(self) -> Dict:
        """Return a JSON-serializable snapshot of the resolved runtime config."""
        return dict(self.data)


def _write_run_provenance(
    output_dir: str,
    config: RunConfig,
    system: Optional[openmm.System] = None,
    topology: Optional[app.Topology] = None,
    positions=None,
    barostat_protocol: Optional[Dict] = None,
    charge_protocol: Optional[Dict] = None,
    dispersion_protocol: Optional[Dict] = None,
    forcefield_family: Optional[Dict] = None,
    membrane_input: Optional[Dict] = None,
    pairs_conversion: Optional[Dict] = None,
    membrane_quality_gate_mode: Optional[str] = None,
) -> Dict:
    provenance = _collect_pipeline_provenance(
        config=config.as_dict(),
        system=system,
        topology=topology,
        positions=positions,
        command_line=sys.argv,
    )
    # memtodolist §10：system_type / barostat_protocol 必须落进 provenance，
    # 且 §3.2 要求 XY/Z 模式与膜法向可审计。这里写的是**已解析**的协议，
    # 不是原始配置字段——原始字段已在 provenance["config"] 里，两者可对照。
    if barostat_protocol is not None:
        provenance["barostat_protocol"] = barostat_protocol
        provenance["system_type"] = barostat_protocol.get("system_type")
    # memtodolist §10：charge_treatment / ligand_net_charge / coion_identity /
    # apbs_applicable-applied 必须落盘；§1.2 要求"最终说明"能直接回答电荷怎么处理。
    if dispersion_protocol is not None:
        provenance["dispersion_protocol"] = dispersion_protocol
    if forcefield_family is not None:
        provenance["forcefield_family"] = forcefield_family
    if pairs_conversion is not None:
        provenance["gromacs_pairs_funct2_conversion"] = pairs_conversion
    # §3.3 / §10：膜输入核对结果（含实测叶片数/水/离子计数与声明来源）落盘。
    if membrane_quality_gate_mode is not None:
        # advisory 放行过的运行必须在 provenance 里认得出来——否则事后无从判断
        # 那个 ΔG_bind 有没有生产资格。
        provenance["membrane_quality_gate_mode"] = membrane_quality_gate_mode
    if membrane_input is not None:
        provenance["membrane_input"] = membrane_input
        provenance["membrane_composition"] = (membrane_input.get("declared") or {}).get(
            "membrane_composition"
        )
    # §13：阈值必须进 provenance，否则"当时用的哪套阈值"无从追溯。
    provenance["acceptance_thresholds"] = acceptance_thresholds_payload()
    if charge_protocol is not None:
        provenance["charge_protocol"] = charge_protocol
        provenance["charge_treatment"] = charge_protocol.get("charge_treatment")
        provenance["ligand_net_charge"] = charge_protocol.get("ligand_net_charge_e")
        provenance["coion_identity"] = charge_protocol.get("co_alchemical_ion")
        provenance["apbs_applicable"] = charge_protocol.get("apbs_applicable")
        provenance["apbs_applied"] = charge_protocol.get("apbs_applied")
        if charge_protocol.get("experimental_not_for_production"):
            provenance["experimental_not_for_production"] = True
        # B3/B4 实现状态如实落盘：一条声明 charge-transfer 的运行到底闭不闭合循环，
        # 事后必须能从 provenance 直接读出来，而不是靠回忆当时代码到哪一步了。
        if charge_protocol.get("closes_thermodynamic_cycle") is False:
            provenance["closes_thermodynamic_cycle"] = False
            provenance["must_not_report_delta_g_bind"] = True
            provenance["incomplete_cycle_reason"] = (
                "charge_transfer_solvent_leg_builder_not_implemented_phase_b4"
            )
    provenance["input_files"] = {
        "gro": config.get("gro"),
        "top": config.get("top"),
        "ligand_xml": config.get("ligand_xml"),
        "config": config.get("config"),
        "torsion_params": config.get("torsion_params"),
        "dexp_params": config.get("dexp_params"),
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_provenance.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2, cls=NumpyEncoder)
    return provenance


def run_self_tests() -> int:
    """Lightweight built-in tests for physics conventions and cache signatures."""
    failures = []
    skipped = []

    def check(name: str, condition: bool, detail: str = ""):
        if condition:
            log.info("self-test PASS | %s", name)
        else:
            failures.append(f"{name}: {detail}")
            log.error("self-test FAIL | %s | %s", name, detail)

    eq = {"r0": 1.0, "thetaA0": 1.5708, "thetaB0": 1.5708}
    fc = {"kr": 418.4, "kthetaA": 41.84, "kthetaB": 41.84, "kphiA": 41.84, "kphiB": 41.84, "kphiC": 41.84}
    dg = calculate_boresch_analytical_correction(eq, fc, 300.0)
    check("Boresch analytical reference", abs(dg - (-22.382123)) < 1e-3, f"got {dg:.6f} kJ/mol")

    # ⚠️ 这只是纯数学工具函数 pme_self_correction_energy_kj 自身的符号单元测试，
    # 不代表生产路径会应用它：apply_pme_self_correction 在生产代码里永远是 False
    # （见 ibs_engine.py 附近注释与 PHYSICS_DEFECTS.md #撤销 +C*lambda^2 修正），
    # OpenMM 的 NonbondedForce ParameterOffset 已经在每个 λ 态里正确算出完整
    # PME 自能，不需要也不应该再手动叠加这个修正。这两个函数只作为历史遗留
    # 工具保留，供未来若要做“已验证的带电荷体系协同炼金中和循环”时复用。
    pref = pme_self_correction_prefactor_kj(alpha_ewald_inv_nm=3.0, ligand_charge_square_sum=2.0)
    corr_lam1 = pme_self_correction_energy_kj(1.0, pref)
    corr_lam0 = pme_self_correction_energy_kj(0.0, pref)
    check(
        "PME self correction helper sign (legacy math util, not applied in production)",
        corr_lam1 > 0.0 and corr_lam0 == 0.0,
        f"pref={pref}, lam1={corr_lam1}",
    )

    endpoints = lambda_endpoint_diagnostics([1.0, 0.5, 0.0], [1.0, 0.5, 0.0])
    bad_endpoints = lambda_endpoint_diagnostics([1.0, 0.5], [1.0, 0.5])
    check("lambda endpoint diagnostics", endpoints["ok"] and not bad_endpoints["ok"], f"{endpoints} / {bad_endpoints}")

    sys_a = openmm.System()
    sys_a.addParticle(12.0)
    sys_b = openmm.System()
    sys_b.addParticle(13.0)
    meta_a = _pme_u_kn_meta_payload(2, [1.0, 0.0], [1.0, 1.0], 300.0, sys_a, None, [0], None)
    meta_b = _pme_u_kn_meta_payload(2, [1.0, 0.0], [1.0, 1.0], 300.0, sys_b, None, [0], None)
    check(
        "resume/cache invalidation system hash",
        meta_a["system_xml_sha256"] != meta_b["system_xml_sha256"],
        "same cache signature for different particle masses",
    )

    try:
        u_kn, n_k = synthetic_mbar_u_kn(delta_f_kT=1.25, n_per_state=200)
        analyzer = TraditionalMBARAnalyzer(temperature=300.0)
        analyzer._last_n_k = n_k
        res = analyzer.solve(u_kn)
        expected = 1.25 * analyzer.kt
        check("MBAR synthetic data", abs(res["delta_G"] - expected) < 1.0, f"got {res['delta_G']:.3f}, expected {expected:.3f}")
    except ImportError as exc:
        skipped.append(f"MBAR synthetic data: {exc}")
        log.warning("self-test SKIP | MBAR synthetic data | %s", exc)

    check("thermodynamic cycle doc present", "PME/self correction" in THERMODYNAMIC_CYCLE_DOC, "missing PME section")

    if skipped:
        log.warning("self-test skipped: %s", skipped)
    if failures:
        log.error("self-test failures: %s", failures)
        return 1
    log.info("self-test completed successfully")
    return 0


# ---------------------------------------------------------------------------
# Boresch 参数统一管理
# ---------------------------------------------------------------------------
def _sanitize_boresch_params(params: Dict) -> Dict:
    """清洗 Boresch 参数字典，去除单位后缀，解包嵌套"""
    if params is None:
        return None

    # 尝试解包嵌套结构
    anchors = params.get("boresch_anchors") or params
    eq = anchors.get("equilibrium_values", params.get("equilibrium_values", {}))
    fc = anchors.get("force_constants", params.get("force_constants", {}))

    eq_mapping = {
        "r0_nm": "r0", "thetaA0_rad": "thetaA0", "thetaB0_rad": "thetaB0",
        "phiA0_rad": "phiA0", "phiB0_rad": "phiB0", "phiC0_rad": "phiC0"
    }
    fc_mapping = {
        "kr_kJ_mol_nm2": "kr", "kthetaA_kJ_mol_rad2": "kthetaA",
        "kthetaB_kJ_mol_rad2": "kthetaB", "kphiA_kJ_mol_rad2": "kphiA",
        "kphiB_kJ_mol_rad2": "kphiB", "kphiC_kJ_mol_rad2": "kphiC"
    }

    try:
        clean_eq = {eq_mapping.get(k, k): float(v) for k, v in eq.items()}
        clean_fc = {fc_mapping.get(k, k): float(v) for k, v in fc.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Boresch 参数包含非数值字段: {exc}") from exc

    cleaned = {
        "receptor_indices": anchors.get("receptor_indices", params.get("receptor_indices", [])),
        "ligand_indices": anchors.get("ligand_indices", params.get("ligand_indices", [])),
        "equilibrium_values": clean_eq,
        "force_constants": clean_fc,
        "is_fallback": params.get("is_fallback", False),
    }
    for key in (
        "method",
        "diagnostics",
        "force_constants_raw",
        "force_constant_clip_ranges",
        "force_constant_clipped",
        "equilibrium_update_error",
    ):
        if isinstance(params, dict) and key in params:
            cleaned[key] = params[key]
        elif isinstance(anchors, dict) and key in anchors:
            cleaned[key] = anchors[key]
    return cleaned


def _has_valid_boresch_anchors(params: Optional[Dict]) -> bool:
    """判定 Boresch 参数是否包含完整 3+3 锚点，避免键存在但值为空时静默失效。"""
    if not isinstance(params, dict):
        return False
    rec_idx = params.get("receptor_indices") or []
    lig_idx = params.get("ligand_indices") or []
    return len(rec_idx) == 3 and len(lig_idx) == 3


def _sanitize_boresch_params_strict(params: Dict) -> Dict:
    """严格清洗并校验 Boresch 参数，发现锚点缺失时立即报错。"""
    cleaned = _sanitize_boresch_params(params)
    if cleaned is None:
        return None
    if not _has_valid_boresch_anchors(cleaned):
        anchors = (params.get("boresch_anchors") or params) if isinstance(params, dict) else {}
        raise ValueError(
            "Boresch 参数缺失有效锚点：需要 3 个 receptor_indices 和 3 个 ligand_indices。"
            f" 当前解析结果 receptor_indices={anchors.get('receptor_indices', params.get('receptor_indices', [])) if isinstance(params, dict) else []},"
            f" ligand_indices={anchors.get('ligand_indices', params.get('ligand_indices', [])) if isinstance(params, dict) else []}"
        )
    required_eq = ("r0", "thetaA0", "thetaB0", "phiA0", "phiB0", "phiC0")
    required_fc = ("kr", "kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC")
    eq = cleaned.get("equilibrium_values") or {}
    fc = cleaned.get("force_constants") or {}
    missing_eq = [k for k in required_eq if k not in eq]
    missing_fc = [k for k in required_fc if k not in fc]
    if missing_eq or missing_fc:
        raise ValueError(
            "Boresch 参数缺失必要字段："
            f"equilibrium missing={missing_eq}, force_constants missing={missing_fc}"
        )
    values = [float(eq[k]) for k in required_eq] + [float(fc[k]) for k in required_fc]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Boresch 参数包含 NaN/Inf: {values}")
    if float(eq["r0"]) <= 0.0 or any(float(fc[k]) <= 0.0 for k in required_fc):
        raise ValueError("Boresch r0 和所有力常数必须为正值")
    return cleaned


BORESCH_GEOMETRY_CONVENTION_VERSION = 2


def resolve_boresch_restraint(config: RunConfig, pipeline: ABFEPipeline) -> Optional[Dict]:
    """统一获取 Boresch 参数。来源：
    - traditional/orb_ml: 加载外部参数文件
    - simple/fluctuation: 纯几何涨落估算 (GeometricRestraintEstimator)，不依赖任何力场/ML 势
    - orb_simple: 基于 ORB/MACE 口袋力投影的单候选估算 (原先的 "simple"，需要 MACE-OFF 许可证/模型)
    - auto: 基于 ORB/MACE 的多候选枚举估算
    """
    if not config.boresch:
        return None

    source = config.boresch_source
    output_dir = config.output

    # 🔑 基线预平衡必须无条件先跑一次，不能只在需要用轨迹估算 Boresch 的来源
    # （auto/simple/fluctuation）才触发。之前 traditional/orb_ml 直接读外部
    # 参数文件返回，从来不会走到这里，导致后续 `_rebalance_with_boresch()`
    # 直接从未平衡的原始坐标开始加 Boresch 限制力——变成"原始坐标 → Boresch
    # rebalance → IBS"，而不是正确的"基线预平衡一次 → Boresch 参数生成/加载 →
    # 带 Boresch rebalance"。这里的判断本身沿用原有的 resume/指纹缓存逻辑，
    # 只是把触发时机提到了 source 分支之前，让所有来源都必然先有这一步。
    _n_equil_steps = config.get("n_equil_steps", 5_000_000)
    _equil_fingerprint = _pre_equilibration_fingerprint(
        pipeline.system, pipeline.ligand_indices, pipeline.temperature, pipeline.pressure,
        positions=pipeline.positions,
        box_vectors=pipeline.box_vectors,
        requested_steps=_n_equil_steps,
        barostat_protocol=pipeline.barostat_protocol,
    )
    if not config.reset and equilibrium_is_done(output_dir, expected_fingerprint=_equil_fingerprint):
        log.info("♻️ 基线预平衡已完成，复用已有轨迹")
    else:
        log.info("▶️ 执行基线预平衡")
        pipeline.pre_equilibrate(
            n_steps=_n_equil_steps,
            save_traj=True,
            resume=config.resume and not config.reset,
        )

    # 传统文件来源
    if source in ("traditional", "orb_ml"):
        path = config.boresch_orb if source == "orb_ml" else config.boresch_anchors
        if not path or not os.path.exists(path):
            raise ValueError(f"Boresch 参数文件不存在: {path}")
        with open(path) as f:
            params = json.load(f)
        log.info("✅ 加载外部 Boresch 参数: %s", path)
        return _sanitize_boresch_params_strict(params)

    # 自动估算来源 (auto / simple / fluctuation)
    boresch_file = os.path.join(output_dir, f"boresch_{source}.json")
    if config.resume and not config.reset and os.path.exists(boresch_file):
        log.info("♻️ 从缓存加载 Boresch 参数: %s", boresch_file)
        with open(boresch_file) as f:
            params = json.load(f)
        if source == "auto" and isinstance(params, dict) and isinstance(params.get("candidates"), list):
            candidates = params["candidates"]
            select_idx = int(getattr(config, "boresch_select", 1) or 1) - 1
            if select_idx < 0 or select_idx >= len(candidates):
                raise ValueError(
                    f"boresch_select={select_idx + 1} 超出缓存候选范围：共 {len(candidates)} 个候选"
                )
            params = candidates[select_idx]
        cache_convention = int(params.get("geometry_convention_version", 0) or 0)
        if (
            source in ("simple", "fluctuation")
            and cache_convention != BORESCH_GEOMETRY_CONVENTION_VERSION
        ):
            log.warning(
                "⚠️ Boresch 缓存几何约定版本 %d != %d；旧缓存可能含末帧二面角反号，"
                "将从无约束预平衡轨迹重新生成。",
                cache_convention,
                BORESCH_GEOMETRY_CONVENTION_VERSION,
            )
        else:
            return _sanitize_boresch_params_strict(params)

    traj_file = os.path.join(output_dir, "pre_equilibration.dcd")
    if not os.path.exists(traj_file):
        raise RuntimeError("预平衡轨迹不存在，无法估算 Boresch 参数")
    traj_top = _resolve_mdtraj_topology_input(output_dir, config.gro, config.top)

    # 根据来源调用不同估算器
    if source == "auto":
        from abfe_core import OrbBoreschEstimator
        estimator = OrbBoreschEstimator(temperature=config.temperature)
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        # 自动估算（默认使用最后 5ns）
        n_candidates = max(int(getattr(config, "boresch_batch", 0) or 1), 1)
        candidates = estimator.estimate_multiple_anchors_from_trajectory(
            traj, config.ligand, n_candidates=n_candidates, output_path=boresch_file, use_last_ns=5.0
        )
        if not candidates:
            raise RuntimeError("自动 Boresch 估算失败，未找到合格候选")
        select_idx = int(getattr(config, "boresch_select", 1) or 1) - 1
        if select_idx < 0 or select_idx >= len(candidates):
            raise ValueError(
                f"boresch_select={select_idx + 1} 超出候选范围：共生成 {len(candidates)} 个候选"
            )
        boresch = candidates[select_idx]

    elif source == "orb_simple":
        # 🔁 原来的 "simple"：基于 ORB/MACE-OFF 口袋力投影的单候选估算，
        # 需要加载 MACE-OFF 模型 (受限许可证)。改名以避免跟下面纯几何的
        # "simple" 混淆——如果不想用 MACE，请用 "simple" 或 "fluctuation"。
        from abfe_core import OrbBoreschEstimator
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        estimator = OrbBoreschEstimator(temperature=config.temperature)
        boresch = estimator.estimate_from_trajectory(traj, config.ligand, output_path=boresch_file)

    elif source in ("simple", "fluctuation"):
        # ✅ 传统方法：纯几何涨落估算 (mdtraj 距离/角度/二面角方差 + 等配分定理)，
        # 不加载任何力场/ML 势，不需要 MACE-OFF 许可证。"simple" 是 "fluctuation" 的别名。
        estimator = GeometricRestraintEstimator(temperature=config.temperature)
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        boresch = estimator.estimate_from_trajectory(traj, config.ligand, output_path=boresch_file)
        boresch["geometry_convention_version"] = BORESCH_GEOMETRY_CONVENTION_VERSION
    else:
        raise ValueError(f"未识别的 Boresch 来源: {source}")

    # 用最后一帧更新平衡几何量
    try:
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        protein_sel = traj.topology.select("protein and backbone")
        if len(protein_sel) > 0:
            traj.superpose(traj, 0, atom_indices=protein_sel)
        traj.center_coordinates()
        traj.image_molecules(inplace=True)
        last_frame_pos = traj.xyz[-1] * unit.nanometer
        new_eq = calc_boresch_from_last_frame(
            last_frame_pos, boresch["receptor_indices"], boresch["ligand_indices"]
        )
        if source in ("simple", "fluctuation"):
            log.info(
                "✅ Boresch 平衡值保留无约束预平衡轨迹系综均值"
                "（geometry convention v%d；不做末帧重锚）",
                BORESCH_GEOMETRY_CONVENTION_VERSION,
            )
        else:
            boresch["equilibrium_values"] = new_eq
            log.info(
                "✅ Boresch 平衡值已用最后一帧更新: r0=%.3f nm", new_eq.get("r0", 0)
            )
    except Exception as e:
        # ⚠️ 不能只打个警告就悄悄继续：平衡值仍是估算器给出的、未经最后一帧精修的旧值。
        # 显式标记 is_fallback，使其经 _sanitize_boresch_params_strict 透传进最终
        # provenance/结果 JSON，保证一次隐性失败可以被事后追溯，而不是产出一个
        # 查不出来的错误 ΔG_bind。
        log.warning("⚠️ 更新 Boresch 平衡值失败: %s，将使用估算器原始平衡值（未经最后一帧精修）", e)
        boresch["is_fallback"] = True
        boresch["equilibrium_update_error"] = str(e)

    # ✅ 谐振/高斯假设的模型无关校验：不管来源是 auto/orb_simple/simple/fluctuation
    # 哪一种估算器，都直接用锁定的 6 个锚点在轨迹上重算涨落分布并做非谐性判定，
    # 结果写入 diagnostics 并在不满足假设时显式报警（原先只有 fluctuation 来源
    # 才有涨落诊断，且从不影响是否报警；scan_boresch_1d_pes 需要 ML 势且只实现
    # r 坐标，从未被任何流程调用，这里用不依赖 ML 势的轨迹统计校验替代其职责）。
    try:
        harmonicity = assess_boresch_harmonicity(
            traj, boresch["receptor_indices"], boresch["ligand_indices"]
        )
        boresch.setdefault("diagnostics", {})
        boresch["diagnostics"]["boresch_harmonicity_check"] = harmonicity
        if harmonicity.get("ok") and not harmonicity.get("harmonic_assumption_ok", True):
            log.warning("⚠️ %s", harmonicity.get("warning", ""))
            boresch["diagnostics"].setdefault("warnings", [])
            if harmonicity["warning"] not in boresch["diagnostics"]["warnings"]:
                boresch["diagnostics"]["warnings"].append(harmonicity["warning"])
        elif not harmonicity.get("ok"):
            log.info("ℹ️ Boresch 谐振性校验未执行: %s", harmonicity.get("reason"))
    except Exception as e:
        log.warning("⚠️ Boresch 谐振性校验失败（不影响主流程，但意味着本次没有该项诊断）: %s", e)
        boresch.setdefault("diagnostics", {})
        boresch["diagnostics"]["boresch_harmonicity_check"] = {"ok": False, "reason": f"exception:{e}"}

    # 裁剪力常数到安全范围
    if "force_constants" in boresch:
        fc = boresch["force_constants"]
        # 🔑 这套范围必须跟 abfe_core.py 里另外两处对力常数的处理保持完全一致，
        # 否则同一个物理量在不同阶段被裁到不同的上限，等于悄悄改写了估计器
        # 给出的合法结果：
        #   - GeometricRestraintEstimator.estimate_from_trajectory 自己的
        #     force_constant_ranges 已经把 kthetaA/B、kphiA/B/C 裁到 [10, 1000]
        #     （simple/fluctuation 来源在到达这里之前就已经clip过一次）；
        #   - calculate_boresch_analytical_correction 对传入的 kthetaA/kthetaB
        #     做的硬性 ValueError 校验也是 [10, 1000]（其注释明确写着要跟前者
        #     一致，见该函数）。
        # 之前这里角度类力常数的上限是 200，比上面两处都窄——一个已经通过
        # estimator 自身 [10,1000] clip 的合法值（比如 500）会在这里被第二次
        # 悄悄裁到 200，且默认来源正是 simple/fluctuation，等于默认路径下的
        # Boresch restraint 被静默软化。统一成同一套 [10, 1000]，不再有第二次
        # 更紧的裁剪。kr 范围本来就与 estimator 自身一致（[100, 2000]），维持不变。
        post_clip_ranges = {
            "kr": [100.0, 2000.0],
            "kthetaA": [10.0, 1000.0],
            "kthetaB": [10.0, 1000.0],
            "kphiA": [10.0, 1000.0],
            "kphiB": [10.0, 1000.0],
            "kphiC": [10.0, 1000.0],
        }
        post_clip_raw = dict(boresch.get("force_constants_raw", {}))
        post_clip_flags = dict(boresch.get("force_constant_clipped", {}))
        for key, (lower, upper) in post_clip_ranges.items():
            if key not in fc:
                continue
            raw_value = float(fc[key])
            post_clip_raw.setdefault(key, raw_value)
            clipped_value = float(np.clip(raw_value, lower, upper))
            fc[key] = clipped_value
            post_clip_flags[key] = bool(post_clip_flags.get(key, False) or abs(clipped_value - raw_value) > 1e-8)
        boresch["force_constants_raw"] = post_clip_raw
        boresch["force_constant_clip_ranges"] = post_clip_ranges
        boresch["force_constant_clipped"] = post_clip_flags
        boresch.setdefault("diagnostics", {})
        if any(post_clip_flags.values()):
            boresch["diagnostics"].setdefault("warnings", [])
            warning = "Boresch force constants were clipped during final sanitation."
            if warning not in boresch["diagnostics"]["warnings"]:
                boresch["diagnostics"]["warnings"].append(warning)
    # 保存
    with open(boresch_file, "w") as f:
        json.dump(boresch, f, indent=2, cls=NumpyEncoder)
    log.info("💾 Boresch 参数已保存: %s", boresch_file)
    return _sanitize_boresch_params_strict(boresch)


# ---------------------------------------------------------------------------
# 命令行解析
# ---------------------------------------------------------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ABFE 计算流程控制器 (v3.0 重构版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 首次运行（自动创建缓存并运行）
  python runabfe.py --gro complex.gro --top complex.top --ligand MOL \\
      --output ./output --boresch --boresch-source auto --preset production

  # 续跑（自动检测缓存）
  python runabfe.py --output ./output --ligand MOL --resume

  # 使用配置文件
  python runabfe.py --config params.json --output ./output --ligand MOL
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    prep_parser = subparsers.add_parser("prepare", help="生成预处理文件 (Boresch/DEXP等)")
    prep_parser.add_argument("--gro", required=True)
    prep_parser.add_argument("--top", required=True)
    prep_parser.add_argument("--ligand", required=True)
    prep_parser.add_argument("--ligand-xml", default=None, help="配体力场 XML/FFXML，用于溶剂腿构建")
    prep_parser.add_argument("--gmx-path", default=None)
    prep_parser.add_argument("--output-dir", default="./prep_output")
    prep_parser.add_argument("--save-boresch", default=None, help="保存 Boresch 文件")
    prep_parser.add_argument("--temperature", type=float, default=300.0)
    prep_parser.add_argument("--platform", default="CUDA")
    prep_parser.add_argument("--n-steps", type=int, default=5_000_000)
    subparsers.add_parser("self-test", help="运行轻量内置单元测试/物理约定检查")
    refine_parser = subparsers.add_parser(
        "refine-lambda-path",
        help="用某个 stage 已经采集到的窗口能量数据，按累积|Δf|重新设计λ分布与窗口边界（不是手写数字）",
    )
    refine_parser.add_argument("--stage-dir", required=True, help="该 stage 的窗口能量文件目录，如 output/vanishing")
    refine_parser.add_argument("--preopt-file", required=True, help="要读取并覆盖写回的 preopt_dual_*.json 路径")
    refine_parser.add_argument("--stage-type", default="vdw", choices=["vdw", "coul"], help="窗口文件名里的阶段类型后缀")
    refine_parser.add_argument("--temperature", type=float, default=300.0)
    refine_parser.add_argument("--n-states", type=int, default=None, help="重新分布后的总态数；默认保持原状态数不变")
    refine_parser.add_argument("--max-window-span-kj", type=float, default=35.0, help="单窗口允许的累积|Δf|预算上限 (kJ/mol)")
    refine_parser.add_argument("--overlap", type=int, default=2, help="相邻窗口之间共享的状态点数")
    # 基本输入
    parser.add_argument("--gro", default=None, help="GROMACS 结构文件 (首次运行时必需)")
    parser.add_argument("--top", default=None, help="GROMACS 拓扑文件 (首次运行时必需)")
    parser.add_argument("--ligand", default=None, help="配体残基名称 (如 MOL)")
    parser.add_argument("--ligand-xml", default=None, help="配体力场 XML/FFXML，用于溶剂腿构建")
    parser.add_argument("--gmx-path", default=None, help="GROMACS 力场 include 目录")
    parser.add_argument("--output", default="./output", help="输出目录 (同时也是缓存目录)")
    parser.add_argument("--config", default=None, help="JSON 配置文件，可覆盖预设")

    # 运行模式
    parser.add_argument("--resume", action="store_true", help="从 Checkpoint 恢复运行")
    parser.add_argument("--reset", action="store_true", help="忽略所有缓存，强制重新开始")

    # 策略选择
    parser.add_argument("--mode", default="ibs", choices=["ibs", "traditional"],
                        help="采样引擎: ibs (默认) 或 traditional-REMD")
    parser.add_argument(
        "--decoupling",
        default="dual_lambda",
        choices=["dual_lambda", "single_lambda", "2d_diagonal", "2d_geodesic"],
    )
    parser.add_argument("--potential", default="softcore", choices=["softcore", "dexp"])
    parser.add_argument("--dexp-params", default=None, help="DEXP 参数文件 (JSON)")
    # 仅影响 --decoupling=dual_lambda 的 Stage 1 (去电荷)；Stage 2 (去VDW)/
    # single_lambda/2d_diagonal/2d_geodesic 不受影响，接口保持原样。
    #   - "pme"（默认，原有接口/行为不变）：IBS-CustomNonbondedForce 对
    #     λ_coul≠0 会硬性拒绝（CustomNonbondedForce 不支持 PME，真实电荷放进
    #     去会被截断成 cutoff 库仑，物理不可信），所以去电荷阶段走的是
    #     NonbondedForce ParameterOffset + 传统 REMD/MBAR，保留完整 PME 长程
    #     静电。这是本仓库当前推荐路线（见 README「当前推荐路线」）。
    #   - "shadow_ibs"（实验性，尚未经独立物理验证，生产结果不要用它）：改走
    #     Shadow-Coulomb IBS——用 erfc(alpha*r)/r 短程"影子"库仑 CV（Ewald 实
    #     空间形式，可以合法塞进 CustomNonbondedForce）做真正的 IBS 多态偏置
    #     采样，再加一段独立的 Bridge 腿（几个窗口的传统 REMD）把体系从"真实
    #     PME 满电荷"搭桥到"Shadow 满电荷"端点，两段 ΔG 相加得到完整去电荷
    #     自由能。只支持电中性配体（带净电配体会在构建系统时直接报错，不会
    #     静默算错），且暂不支持 --parallel-stages。详见
    #     ibs_engine.py::run_shadow_bridge_leg /
    #     ibs_engine.py::IBSWindowManagerShadowCoul。
    parser.add_argument(
        "--decharge-method",
        default="pme",
        choices=["pme", "shadow_ibs"],
        help="dual_lambda 去电荷阶段的实现方式：pme(默认，生产路线) 或 "
             "shadow_ibs(实验性 Shadow-Coulomb IBS，尚未经物理验证)",
    )

    # Boresch 相关
    parser.add_argument("--boresch", action=argparse.BooleanOptionalAction, default=None,
                        help="是否启用 Boresch 限制力；复合物腿默认启用，可用 --no-boresch 显式关闭")
    parser.add_argument("--boresch-source", default=None,
                        choices=["traditional", "orb_ml", "simple", "orb_simple", "auto", "fluctuation"],
                        help="Boresch 来源：traditional/orb_ml=加载外部参数文件；"
                             "simple/fluctuation=纯几何涨落估算(不需要MACE)；"
                             "orb_simple=ORB/MACE口袋力投影单候选估算(需要MACE-OFF模型)；"
                             "auto=ORB/MACE多候选枚举估算(需要MACE-OFF模型)")
    parser.add_argument("--boresch-anchors", default=None, help="传统 Boresch 锚点文件")
    parser.add_argument("--boresch-orb", default=None, help="Orb ML 预测 Boresch 文件")
    parser.add_argument("--boresch-batch", type=int, default=0, help="批量估算候选数量")
    parser.add_argument("--boresch-select", type=int, default=1, help="选择第 N 个候选")
    parser.add_argument("--skip-rebalance", action="store_true", help="跳过带 Boresch 的再平衡")
    parser.add_argument("--rebalance-steps", type=int, default=50000, help="再平衡步数")

    # 采样控制
    parser.add_argument("--preset", default="production", choices=PRESET_CONFIGS.keys())
    parser.add_argument("--n-steps-per-window", type=int, default=None)
    parser.add_argument("--steps-per-update", type=int, default=None)
    parser.add_argument("--n-states-per-stage", type=int, default=None, help="每阶段 λ 状态数（同时设置两阶段）")
    parser.add_argument("--stage1-n-states", type=int, default=None, help="去电荷阶段 λ 状态数，优先级高于 --n-states-per-stage")
    parser.add_argument("--stage2-n-states", type=int, default=None, help="去VDW阶段 λ 状态数，优先级高于 --n-states-per-stage")
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument(
        "--solvent-ionic-strength-molar",
        type=float,
        default=None,
        help="溶剂腿 NaCl 浓度 (M)，默认 0.15；另加必要反离子保持中性",
    )
    parser.add_argument("--platform", default="CUDA", choices=["CUDA", "OpenCL", "CPU"])

    # ---- 膜受体体系（memtodolist §3.1/§3.2，B1）。默认 soluble，不传等于关闭 ----
    # ⚠️ 这里的 --system-type 是**环境类型**（soluble/membrane），决定用哪种
    # barostat。它与 run_full_pipeline(system_type="complex"|"solvent") 那个
    # 同名的**腿身份**参数是两个不相干的轴，不要混。
    parser.add_argument(
        "--system-type",
        default=None,
        choices=["soluble", "membrane"],
        help="体系环境类型；membrane 启用 MonteCarloMembraneBarostat。默认 soluble（不改变任何现有行为）",
    )
    parser.add_argument(
        "--membrane-normal-axis",
        default=None,
        choices=["x", "y", "z"],
        help="膜法向轴；OpenMM 的膜 barostat 只支持 z，其它值 fail closed",
    )
    parser.add_argument(
        "--membrane-surface-tension-bar-nm",
        type=float,
        default=None,
        help="膜表面张力 (bar·nm)，默认 0.0",
    )
    parser.add_argument(
        "--membrane-xy-mode",
        default=None,
        choices=["isotropic", "anisotropic"],
        help="XY 平面缩放模式，默认 isotropic（XY 等比例）",
    )
    parser.add_argument(
        "--membrane-z-mode",
        default=None,
        choices=["free", "fixed", "constant_volume"],
        help="Z 轴模式，默认 free（Z 独立变化）",
    )
    parser.add_argument(
        "--membrane-barostat-frequency",
        type=int,
        default=None,
        help="膜 barostat 体积移动频率，默认 25",
    )
    parser.add_argument(
        "--membrane-quality-gate",
        default=None,
        choices=["enforce", "advisory"],
        help=(
            "§9 膜质量门模式。enforce（默认）门未过即阻断；"
            "advisory 照样计算并落盘报告、失败大声 warning 但不阻断"
            "（用于先把管路跑通；**不是生产资格**）"
        ),
    )
    parser.add_argument(
        "--confirm-soluble-with-lipids",
        action="store_true",
        help="确认 system_type=soluble 但拓扑含大量脂质残基（留记录，不静默放行）",
    )

    # ---- 净电荷处理协议（memtodolist §1.2，B2）----
    # 不传时按配体净电荷推导：中性 → neutral（当前生产路径），带电 → charge-transfer。
    parser.add_argument(
        "--charge-treatment",
        default=None,
        choices=[
            "neutral",
            "co_alchemical_charge_transfer",
            "rocklin_apbs_neutralizing_plasma",
            "co_annihilation_experimental",
        ],
        help=(
            "净电荷处理路线。co-ion 路线与 Rocklin/APBS 是二选一，同时用即重复修正、"
            "会 fail closed。co_annihilation_experimental 仅供方法对照，膜体系禁用"
        ),
    )
    parser.add_argument(
        "--co-alchemical-ion",
        default=None,
        help="co-alchemical ion 身份/参数/restraint 的 JSON 文件路径（§3.4 字段）",
    )
    parser.add_argument(
        "--apbs-evidence",
        default=None,
        help="rocklin_apbs_neutralizing_plasma 路线的 APBS 证据 JSON（manifest/result/介电图/脂质电荷图）",
    )

    # ---- LJ/色散路线与力场族（memtodolist §1.1 / §1.3，B6）----
    parser.add_argument(
        "--dispersion-protocol",
        default=None,
        choices=[
            "legacy_uniform_density_lrc",
            "ff_native_isotropic_lrc",
            "ff_native_force_switch_no_lrc",
            "lj_pme",
            "membrane_inhomogeneous",
        ],
        help=(
            "LJ/色散路线。soluble 不传 = legacy_uniform_density_lrc（行为不变）；"
            "membrane 不传即 fail closed。lj_pme / membrane_inhomogeneous 是路线 B/C，未实现"
        ),
    )
    parser.add_argument(
        "--forcefield-family",
        default=None,
        choices=["amber", "charmm"],
        help="显式覆盖从 .top 自动识别的力场族；覆盖会留记录，不静默",
    )
    parser.add_argument(
        "--membrane-input-declaration",
        default=None,
        help=(
            "膜输入声明 JSON（§3.3）：build_tool / build_parameters / "
            "final_equilibration_job / source_structure_id / conformational_state / "
            "membrane_composition / binding_site_solvent_exposure / "
            "leaflet_assignment_basis，可选 n_atoms / n_upper / n_lower / n_water / ion_counts。"
            "system_type=membrane 时必填"
        ),
    )
    parser.add_argument(
        "--force-switch-deviation-evidence",
        default=None,
        help=(
            "charmm 分支放行所需的定量偏差论证 JSON（APL / 膜厚 / 单点能对照）。"
            "OpenMM 没有 force-switch，无此证据一律 fail closed"
        ),
    )

    # 高级选项
    parser.add_argument("--torsion-params", default=None)
    parser.add_argument("--enable-early-stop", action="store_true")
    parser.add_argument("--enable-gradual-warmup", action="store_true")
    parser.add_argument("--disable-warmup", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=500000)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--analyze-only", action="store_true", help="仅分析已有 .npy")
    parser.add_argument("--parallel-stages", action="store_true", help="并行执行去电荷和去VDW阶段")
    parser.add_argument(
        "--only-complex-charging",
        action="store_true",
        help=(
            "隔离重跑复合物腿 Stage 1 PME charging；跳过 Stage 2 路径优化、"
            "VDW 采样和溶剂腿，并用现有 Stage 2/solvent 结果生成候选汇总"
        ),
    )
    parser.add_argument(
        "--only-boresch-attachment",
        action="store_true",
        help=(
            "[P1-17] 只跑 Boresch attachment 腿 ΔG(A′→A)（限制从关到开），"
            "复用冻结的 stage1/stage2/solvent 生成闭合循环的候选汇总。"
            "stage1/stage2 本就是受约束系综里测的，补这一项不改变它们"
        ),
    )
    parser.add_argument(
        "--attachment-rerun-dir",
        default=None,
        help="attachment 腿的隔离输出目录（默认 <output>/attachment_rerun/<时间戳>）",
    )
    parser.add_argument(
        "--attachment-n-steps-per-state",
        type=int,
        default=None,
        help="attachment 腿每个 λ 态的生产步数（默认 250000）",
    )
    parser.add_argument(
        "--attachment-n-seeds",
        type=int,
        default=None,
        help=(
            "attachment 腿跑几条独立轨迹（默认 1）。>1 时误差棒改用**跨 seed SEM**，"
            "而不是 MBAR 渐近协方差——后者在本体系已证明低估约 7 倍"
        ),
    )
    parser.add_argument(
        "--attachment-lambdas",
        default=None,
        help="逗号分隔的 λ 表，必须严格升序且从 0 到 1（默认 0.0,0.1,0.35,1.0）",
    )
    parser.add_argument(
        "--charging-rerun-dir",
        default=None,
        help=(
            "charging-only 隔离输出目录；默认在 OUTPUT/charging_rerun/ 下创建"
            "带时间戳的新目录，现有非空目录会被拒绝"
        ),
    )
    parser.add_argument(
        "--charging-max-resident-contexts",
        type=int,
        default=None,
        help=(
            "charging REMD 允许同时驻留的 Context 数；默认使用安全策略（CUDA "
            "会回退 CPU）。仅在显存足够时显式设为 Stage 1 状态数"
        ),
    )
    parser.add_argument("--n-lambda", type=int, default=12, help="传统 REMD 模式的 λ 状态数")
    parser.add_argument(
        "--apbs-correction-kj-mol",
        type=float,
        default=None,
        help="外部 APBS 长程/连续介质修正，单位 kJ/mol；最终 ΔG_bind 会加上该项",
    )
    parser.add_argument(
        "--apbs-correction-note",
        default=None,
        help="APBS 修正来源说明，例如网格、介电常数、输入文件或结果文件",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# 分析模式（简化版，调用原有后处理）
# ---------------------------------------------------------------------------
def run_post_analysis(args):
    output_dir = args.output
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")

    log.info("进入后处理分析模式...")
    temp = args.temperature * unit.kelvin
    kt = (unit.MOLAR_GAS_CONSTANT_R * temp).value_in_unit(unit.kilojoule_per_mole)
    def _load_boresch_params(base_dir: str):
        for json_name in [
            "boresch_params.json",
            "boresch_auto.json",
            "boresch_simple.json",
            "boresch_fluctuation.json",
        ]:
            path = os.path.join(base_dir, json_name)
            if os.path.exists(path):
                with open(path) as f:
                    return _sanitize_boresch_params(json.load(f))
        return None

    def _analyze_dual_leg(base_dir: str) -> Dict:
        stages = ["coul", "vdw"]
        stage_name_map = {"coul": "decharging", "vdw": "vanishing"}
        total_dg = 0.0
        total_err_sq = 0.0
        for stage in stages:
            stage_name = stage_name_map[stage]
            stage_dir = os.path.join(base_dir, stage_name)
            if not os.path.exists(stage_dir):
                raise FileNotFoundError(f"缺少阶段目录: {stage_dir}")
            window_data = []

            stage_checkpoint = os.path.join(
                base_dir,
                "checkpoints",
                "stage1_decharging.json" if stage == "coul" else "stage2_vanishing.json",
            )
            if os.path.exists(stage_checkpoint):
                with open(stage_checkpoint) as f:
                    cached_stage = json.load(f)
                # 🔑 之前用 .get("total_delta_G", 0.0)/.get("total_error", 0.0)
                # 静默补 0——一份损坏或旧格式（缺字段）的 checkpoint 会让整段腿
                # 被悄悄算成 0 kJ/mol 而不是报错，且没有下游门槛会发现这个 0 是
                # 假的。这里要求两个字段都必须存在、是数值类型、且有限，否则
                # fail closed；同时校验落盘时记录的 stage 名与当前正在分析的
                # stage_name 一致，防止一份属于另一阶段的 checkpoint 被张冠李戴
                # 读取（例如目录被手动拷贝/合并过）。
                if cached_stage.get("stage") not in (None, stage_name):
                    raise RuntimeError(
                        f"阶段 checkpoint {stage_checkpoint} 记录的 stage="
                        f"{cached_stage.get('stage')!r} 与当前分析的阶段 "
                        f"{stage_name!r} 不一致，拒绝把它当作本阶段结果使用。"
                    )
                dg_raw = cached_stage.get("total_delta_G")
                err_raw = cached_stage.get("total_error")
                if not isinstance(dg_raw, (int, float)) or not isinstance(err_raw, (int, float)):
                    raise RuntimeError(
                        f"阶段 checkpoint {stage_checkpoint} 缺少或类型错误的 "
                        f"total_delta_G/total_error（{dg_raw!r}/{err_raw!r}）；拒绝"
                        "用默认值 0.0 静默把这段腿算成 0，请检查该 checkpoint 是否"
                        "损坏或来自旧格式。"
                    )
                dg_raw = float(dg_raw)
                err_raw = float(err_raw)
                if not (np.isfinite(dg_raw) and np.isfinite(err_raw) and err_raw >= 0.0):
                    raise RuntimeError(
                        f"阶段 checkpoint {stage_checkpoint} 的 total_delta_G/total_error "
                        f"非有限或不确定度为负（{dg_raw}/{err_raw}）；拒绝静默使用。"
                    )
                total_dg += dg_raw
                total_err_sq += err_raw ** 2
                continue

            preopt_file = os.path.join(base_dir, "checkpoints", f"preopt_dual_{stage_name}.json")
            window_ranges = []
            if os.path.exists(preopt_file):
                try:
                    with open(preopt_file) as f:
                        window_ranges = json.load(f).get("window_ranges", [])
                except Exception as e:
                    log.warning("读取 preopt 缓存失败 (%s): %s", preopt_file, e)

            e_files_raw = glob.glob(os.path.join(stage_dir, f"dual_window_*_{stage}_energies.npy"))
            if not e_files_raw and not os.path.exists(stage_checkpoint):
                raise FileNotFoundError(f"阶段 {stage_name} 缺少可分析的能量文件: {stage_dir}")
            # 🔑 之前这里对 glob 结果做普通字符串排序、再用 enumerate 的位置索引当
            # window_idx——窗口数达到两位数时 window_10/window_11 会排到
            # window_1/window_2 前面，中间缺一个文件时后续全部错位。改为从文件名
            # 正则解析真实窗口编号（跟 abfe_pipeline.py 里清理窗口产物时用的同一
            # 套正则 dual_window_(\d+)_{stage_type}_energies\.npy$ 一致），按数值
            # 排序，并要求编号从 0 连续到 N-1，缺一个都拒绝继续（不能悄悄错配）。
            _window_idx_re = re.compile(rf"dual_window_(\d+)_{stage}_energies\.npy$")
            indexed_e_files = []
            for e_file in e_files_raw:
                m = _window_idx_re.search(os.path.basename(e_file))
                if not m:
                    raise RuntimeError(f"无法从文件名解析窗口编号: {e_file}")
                indexed_e_files.append((int(m.group(1)), e_file))
            indexed_e_files.sort(key=lambda pair: pair[0])
            parsed_indices = [idx for idx, _ in indexed_e_files]
            if parsed_indices and parsed_indices != list(range(len(parsed_indices))):
                raise RuntimeError(
                    f"阶段 {stage_name} 的窗口能量文件编号不连续或有缺失"
                    f"（解析得到 {parsed_indices}），拒绝在窗口缺失/错位的情况下"
                    "继续分析（此前会静默按字符串排序位置错配窗口）。"
                )
            for w_idx, e_file in indexed_e_files:
                u_kn = np.load(e_file)
                bias_path = e_file.replace("_energies.npy", "_bias.npy")
                base_path = e_file.replace("_energies.npy", "_base.npy")
                if not os.path.exists(bias_path):
                    raise RuntimeError(f"缺少 IBS bias 能量文件: {bias_path}")
                if not os.path.exists(base_path):
                    raise RuntimeError(f"缺少 base 能量文件: {base_path}")
                if w_idx < len(window_ranges):
                    start, end = window_ranges[w_idx]
                    real_indices = list(range(start, end))
                else:
                    real_indices = list(range(u_kn.shape[0]))
                window_data.append(
                    {
                        "u_kn": u_kn,
                        "bias_energies": np.load(bias_path),
                        "base_energies": np.load(base_path),
                        "lambda_indices": real_indices,
                    }
                )
            if window_data:
                res = solve_stage_integrated(window_data, kt, stage_name=stage)
                # 🔑 之前这里不管 solve_stage_integrated 是否报告 error 或
                # converged is not True，都直接用 total_delta_G 的默认值 0.0
                # 继续拼装并写出 final_results_postprocess.json——一个失败或
                # 只解出部分窗口的求解结果会被静默当成真实答案。回退路径本身
                # 就是"缺正式结果时的粗略核查"，更不能再放行一个明确失败的解。
                if res.get("error"):
                    raise RuntimeError(
                        f"阶段 {stage_name} 的 solve_stage_integrated 报告错误: "
                        f"{res.get('error')!r}；拒绝在回退分析路径下静默产出结果。"
                    )
                if res.get("converged") is not True:
                    raise RuntimeError(
                        f"阶段 {stage_name} 的 solve_stage_integrated 未收敛"
                        f"（converged={res.get('converged')!r}）；拒绝在回退分析"
                        "路径下静默产出部分/不可信结果。"
                    )
                total_dg += float(res.get("total_delta_G", 0.0))
                total_err_sq += float(res.get("total_error", 999.0)) ** 2
        return {
            "decoupling_delta_G_kJ_mol": float(total_dg),
            "total_error_kJ_mol": float(np.sqrt(total_err_sq)),
        }

    def _load_leg_result(base_dir: str) -> Dict:
        result_path = os.path.join(base_dir, "final_results.json")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"缺少腿结果文件: {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)

    complex_boresch = _load_boresch_params(output_dir)
    dg_boresch = 0.0
    if complex_boresch and complex_boresch.get("force_constants") and complex_boresch.get("equilibrium_values"):
        # 🔑 之前这里计算失败只打 warning，dg_boresch 保持初始化的 0.0 继续往下走——
        # 在没有正式 final_results.json 的原始窗口回退场景（见下面
        # boresch_included_in_complex_dg=False 的分支），这个 0.0 会被直接减进
        # delta_g_bind_uncorrected 的公式，等于悄悄漏掉整项
        # restraint release 修正。已经找到 Boresch 参数（force_constants/
        # equilibrium_values 都在）说明这个体系确实需要这项修正，计算失败就必须
        # fail closed；只有从一开始就找不到 Boresch 参数（上面 if 判断为 False，
        # 即体系本身没有做 Boresch restraint）才允许 dg_boresch 维持 0.0。
        dg_boresch = calculate_boresch_analytical_correction(
            eq=complex_boresch["equilibrium_values"],
            fc=complex_boresch["force_constants"],
            T=temp,
        )

    # 追踪 dg_boresch 这一物理量本身是否已经被烘焙进 dg_complex
    # （total_delta_G_complex_kJ_mol）。这既是下游消费者判断"能不能对
    # complex_delta_G_kJ_mol 再扣一次 Boresch"的依据，也是
    # combine_binding_free_energy 决定减不减第二次的唯一开关。
    #
    # 🔑 [ATT-09] 这里原来还有一个并行的 `dg_boresch_term`（在同样两个分支里被置 0），
    # 与这个布尔量是同一件事的两份记账。公式统一进 helper 之后它已无人读取，
    # 直接删掉——留着一个"曾经控制公式、现在没人读"的变量正是日后被错误接回去的引信。
    boresch_included_in_complex_dg = False
    if args.mode == "traditional":
        complex_leg = _load_leg_result(os.path.join(output_dir, "traditional_complex"))
        solvent_leg = _load_leg_result(os.path.join(output_dir, "traditional_solvent"))
        dg_complex = float(complex_leg["delta_G_total_kJ_mol"])
        dg_solvent = float(solvent_leg["delta_G_total_kJ_mol"])
        err_complex = float(complex_leg["error_leg_kJ_mol"])
        err_solvent = float(solvent_leg["error_leg_kJ_mol"])
    elif args.decoupling == "dual_lambda":
        _complex_final_path = os.path.join(output_dir, "final_results.json")
        _solvent_final_path = os.path.join(output_dir, "solvent_leg", "final_results.json")
        if os.path.exists(_complex_final_path) and os.path.exists(_solvent_final_path):
            # ✅ 正式流程已经把 PME 自能修正、Boresch 释放等修正项烘焙进
            # total_delta_G_complex_kJ_mol 里落盘为 final_results.json；这里优先复用
            # 该权威口径，而不是用 _analyze_dual_leg 从原始窗口能量重新拼出一份可能
            # 遗漏这些修正的"影子"总量，避免 --analyze-only 与正式流程结果对不上。
            complex_leg = _load_leg_result(output_dir)
            solvent_leg = _load_leg_result(os.path.join(output_dir, "solvent_leg"))
            dg_complex = float(complex_leg["total_delta_G_complex_kJ_mol"])
            dg_solvent = float(solvent_leg["total_delta_G_complex_kJ_mol"])
            err_complex = float(complex_leg["total_error_kJ_mol"])
            err_solvent = float(solvent_leg["total_error_kJ_mol"])
            # ✅ 同下方 "else" 分支的既有约定：total_delta_G_complex_kJ_mol 已经把
            # Boresch 释放修正烘焙进去了（见 abfe_pipeline.py total_dg = dg_phys +
            # cons_correction + dg_boresch），所以循环闭合时绝不能再减一次，
            # 否则同一个 Boresch 修正会被计入两遍。
            boresch_included_in_complex_dg = True
        else:
            log.warning(
                "未找到正式 final_results.json，回退为从原始窗口能量文件重新估算 "
                "decoupling_delta_G_kJ_mol（该值不含正式流程里烘焙的 PME 自能/长程修正等项，"
                "仅供粗略核查，不等价于正式 total/APBS 组装口径）。"
            )
            complex_leg = _analyze_dual_leg(output_dir)
            solvent_leg = _analyze_dual_leg(os.path.join(output_dir, "solvent_leg"))
            dg_complex = float(complex_leg["decoupling_delta_G_kJ_mol"])
            dg_solvent = float(solvent_leg["decoupling_delta_G_kJ_mol"])
            err_complex = float(complex_leg["total_error_kJ_mol"])
            err_solvent = float(solvent_leg["total_error_kJ_mol"])
    else:
        complex_leg = _load_leg_result(output_dir)
        solvent_leg = _load_leg_result(os.path.join(output_dir, "solvent_leg"))
        dg_complex = float(complex_leg["total_delta_G_complex_kJ_mol"])
        dg_solvent = float(solvent_leg["total_delta_G_complex_kJ_mol"])
        err_complex = float(complex_leg["total_error_kJ_mol"])
        err_solvent = float(solvent_leg["total_error_kJ_mol"])
        boresch_included_in_complex_dg = True

    # 🔑 [ATT-09] 循环闭合统一走 abfe_core.combine_binding_free_energy。
    # 这条路径上 `boresch_included_in_complex_dg` 就是 helper 要的那个开关：
    # 上面三个分支已经分别判定过 complex 腿有没有烘焙释放项，helper 据此决定
    # 减不减第二次。原来这里手写 `- dg_boresch_term`，与那个布尔量是两份记账，
    # 现在合成一份。
    # 🔑 之前这里读 getattr(args, "apbs_correction_kj_mol", None)——小写 j，是
    # argparse 从 --apbs-correction-kj-mol 派生的 CLI dest 名，只在这次
    # --analyze-only 调用本身显式带了这个参数时才非 None；正式跑
    # (RunConfig.data 存的是大写 J 的 "apbs_correction_kJ_mol"，main() 里
    # config.get("apbs_correction_kJ_mol", ...) 读的是这个) 用过的值从未被
    # 读取——不带 CLI 参数重跑 --analyze-only 会静默把 APBS 修正算成 0，
    # 且不报错。正式跑一定会在早期写一份 run_provenance.json
    # (_write_run_provenance，含 config.as_dict() 的完整快照)，这里改为优先
    # 读它；只有这次 --analyze-only 调用显式重新传了 --apbs-correction-kj-mol
    # 才用它覆盖（跟 RunConfig.__init__ 里 _flag_present 的判断方式一致，不
    # 依赖 CLI 参数默认值是否恰好为 None 这种脆弱判断）。
    _argv = sys.argv[1:]
    def _flag_present(*flags):
        for flag in flags:
            if flag in _argv:
                return True
            prefix = f"{flag}="
            if any(token.startswith(prefix) for token in _argv):
                return True
        return False
    if _flag_present("--apbs-correction-kj-mol"):
        apbs_correction = float(getattr(args, "apbs_correction_kj_mol", None) or 0.0)
    else:
        apbs_correction = 0.0
        _provenance_path = os.path.join(output_dir, "run_provenance.json")
        if os.path.exists(_provenance_path):
            with open(_provenance_path) as f:
                _provenance = json.load(f)
            apbs_correction = float(
                _provenance.get("config", {}).get("apbs_correction_kJ_mol", 0.0) or 0.0
            )
    cycle = combine_binding_free_energy(
        dg_complex_kJ_mol=dg_complex,
        dg_solvent_kJ_mol=dg_solvent,
        err_complex_kJ_mol=err_complex,
        err_solvent_kJ_mol=err_solvent,
        dg_boresch_kJ_mol=dg_boresch,
        boresch_already_included_in_complex=bool(boresch_included_in_complex_dg),
        apbs_correction_kJ_mol=apbs_correction,
    )
    delta_g_bind_uncorrected = cycle["delta_G_bind_uncorrected_kJ_mol"]
    final_dg = cycle["delta_G_bind_kJ_mol"]
    final_err = cycle["total_error_kJ_mol"]
    log.info("ABFE 后处理完成: ΔG_bind = %.2f ± %.2f kJ/mol", final_dg, final_err)

    result = {
        "complex_leg_delta_G_kJ_mol": float(dg_complex),
        "solvent_leg_delta_G_kJ_mol": float(dg_solvent),
        # ✅ 报告 Boresch 修正的真实物理量级 (dg_boresch)，而不是公式里实际被减掉
        # 的那一项（已内含时为 0，见 cycle["boresch_term_subtracted_kJ_mol"]）。
        # 同时显式标记它是否已经烘焙进 complex_leg_delta_G_kJ_mol，避免下游
        # 消费者误以为这里恒为独立可加项而对 complex_delta_G 二次扣减。
        "boresch_correction_kJ_mol": float(dg_boresch),
        "boresch_correction_already_included_in_complex_delta_G": bool(boresch_included_in_complex_dg),
        "boresch_correction_note": (
            "boresch_correction_kJ_mol 已经烘焙进 complex_leg_delta_G_kJ_mol，"
            "且已完整体现在 delta_G_bind_kJ_mol 中；下游不要再对二者中任何一个二次扣减 Boresch 修正。"
            if boresch_included_in_complex_dg else
            "boresch_correction_kJ_mol 未包含在 complex_leg_delta_G_kJ_mol 中，"
            "而是作为独立项已经减到 delta_G_bind_kJ_mol 里；下游不要再对 delta_G_bind_kJ_mol 二次扣减。"
        ),
        "delta_G_bind_uncorrected_kJ_mol": float(delta_g_bind_uncorrected),
        "apbs_correction_kJ_mol": float(apbs_correction),
        "apbs_correction_note": getattr(args, "apbs_correction_note", None) or "",
        "delta_G_bind_kJ_mol": float(final_dg),
        "delta_G_bind_kcal_mol": float(final_dg / 4.184),
        "total_error_kJ_mol": final_err,
        "thermodynamic_cycle_terms": cycle,
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "decoupling": args.decoupling,
    }
    out_path = os.path.join(output_dir, "final_results_postprocess.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    log.info("结果已保存: %s", out_path)

def run_prepare_command(args):
    """简化版预处理命令：生成系统缓存 + 可选 Boresch / DEXP 估算"""
    log.info("🔧 执行 prepare 子命令")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. 构建系统（与 main 中首次构建一致）
    include_dir = find_gmx_include_dir(args.gmx_path)
    cache_identity = _main_cache_identity(
        args.gro, args.top, args.ligand, include_dir
    )
    system, topology, positions, box_vectors, ligand_indices = build_system_from_gromacs(
        args.gro, args.top, args.ligand, include_dir
    )
    diagnose_14_scaling(system)
    save_native_system(
        output_dir,
        system,
        topology,
        ligand_indices,
        positions,
        box_vectors,
        cache_identity=cache_identity,
    )

    # 2. 从缓存重载（保证一致性）
    system, topology, positions, box_vectors, ligand_indices = load_native_system(
        output_dir, gro_file=args.gro, top_file=args.top, gmx_include_dir=include_dir
    )

    # 3. PBC 居中（与主流程完全一致）
    if hasattr(positions, "value_in_unit"):
        pos_nm = positions.value_in_unit(unit.nanometer)
    else:
        pos_nm = positions
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    if pos_nm.ndim == 1:
        pos_nm = pos_nm.reshape(-1, 3)
    positions = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_nm] * unit.nanometer
    positions, box_vectors = center_system_rigidly(positions, box_vectors, ligand_indices)

    # 4. 初始化 Pipeline（只做预平衡估算用）
    # 这里也走膜协议：prepare 会调 pre_equilibrate()，膜体系必须用膜 barostat，
    # 否则 prepare 产出的预平衡轨迹是各向同性缩放下的，APL 已经跑掉了。
    pipeline = ABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=args.temperature,
        output_dir=output_dir,
        platform_name=args.platform,
        environment_type=getattr(args, "system_type", None),
        membrane=(_membrane_overrides_from_args(args) or None),
        confirm_soluble_with_lipids=bool(
            getattr(args, "confirm_soluble_with_lipids", False)
        ),
        dispersion_protocol=getattr(args, "dispersion_protocol", None),
        forcefield_family=getattr(args, "forcefield_family", None),
        # [B3] 净电荷路线：两条腿必须传同一个（§6.1）。它只决定 co-ion 身份怎么冻结，
        # 中性配体（当前生产体系）根本不进 co-ion 分支，行为逐位不变。
        charge_treatment=getattr(args, "charge_treatment", None),
    )

    # 5. 预平衡生成轨迹
    equil_data = pipeline.pre_equilibrate(
        n_steps=args.n_steps,
        save_traj=True,
        platform_name=args.platform,
    )
    traj_file = equil_data["trajectory_file"]

    # 6. Boresch 估算（默认 fluctuation 模式）
    if args.save_boresch:
        import mdtraj as md
        traj = md.load(traj_file, top=args.gro)
        estimator = GeometricRestraintEstimator(temperature=args.temperature)
        boresch = estimator.estimate_from_trajectory(
            traj, args.ligand,
            output_path=os.path.join(output_dir, args.save_boresch)
        )
        log.info("Boresch 参数已保存至 %s", args.save_boresch)

    log.info("✅ prepare 完成，文件已输出至 %s", output_dir)

# ---------------------------------------------------------------------------
# 传统 ABFE-REMD 模式
# ---------------------------------------------------------------------------
def run_traditional_mode(config: RunConfig):
    """传统双阶段 λ-REMD：分别计算复合物腿与溶剂腿并汇总结合自由能。"""
    log.info("🔧 启动传统 ABFE-REMD 模式")
    output_dir = config.output
    os.makedirs(output_dir, exist_ok=True)

    if not config.gro or not config.top:
        raise ValueError("traditional 模式必须提供 gro/top 输入，或在配置文件中定义 gro/top。")

    system, topology, positions, box_vectors, ligand_indices = build_system_from_gromacs(
        config.gro,
        config.top,
        config.ligand,
        find_gmx_include_dir(config.gmx_path),
    )
    diagnose_14_scaling(system)

    if hasattr(positions, "value_in_unit"):
        pos_nm = positions.value_in_unit(unit.nanometer)
    else:
        pos_nm = positions
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    if pos_nm.ndim == 1:
        pos_nm = pos_nm.reshape(-1, 3)
    positions = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_nm] * unit.nanometer
    positions, box_vectors = center_system_rigidly(positions, box_vectors, ligand_indices)

    ligand_resname = _get_residue_name_by_atom_index(topology, ligand_indices[0])
    solvent_identity = _ligand_parameter_identity(
        system,
        topology,
        ligand_indices,
        ligand_resname,
        config.top,
        getattr(config, "ligand_xml", None),
        find_gmx_include_dir(config.gmx_path),
    )
    if config.reset or not solvent_cache_exists(
        output_dir, config.solvent_ionic_strength_molar, solvent_identity
    ):
        log.info("💧 traditional 模式准备溶剂腿缓存...")
        if not build_and_cache_solvent_leg(
            output_dir,
            topology,
            positions,
            ligand_indices,
            ligand_resname,
            ligand_ffxml=getattr(config, "ligand_xml", None),
            top_file=config.top,
            gmx_include_dir=find_gmx_include_dir(config.gmx_path),
            ionic_strength_molar=config.solvent_ionic_strength_molar,
            cache_identity=solvent_identity,
            charge_treatment=config.get("charge_treatment"),
        ):
            raise RuntimeError("traditional 模式自动构建溶剂腿缓存失败。")

    dg_boresch = 0.0
    boresch_restraint = None
    if config.boresch:
        boresch_pipeline = ABFEPipeline(
            system=system,
            topology=topology,
            positions=positions,
            box_vectors=box_vectors,
            ligand_indices=ligand_indices,
            temperature=config.temperature,
            output_dir=output_dir,
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
            platform_name=config.platform,
            # 复合物腿 → 走膜协议（Boresch 估算会触发预平衡）。
            environment_type=config.get("system_type"),
            membrane=config.get("membrane"),
            confirm_soluble_with_lipids=bool(
                config.get("confirm_soluble_with_lipids", False)
            ),
            dispersion_protocol=config.get("dispersion_protocol"),
            forcefield_family=config.get("forcefield_family"),
            charge_treatment=config.get("charge_treatment"),
        )
        boresch_restraint = resolve_boresch_restraint(config, boresch_pipeline)
        if boresch_restraint:
            dg_boresch = boresch_pipeline.apply_boresch_correction(
                boresch_restraint,
                autoload_from_disk=False,
            ).get("delta_g_rest", 0.0)
            positions = boresch_pipeline.positions
            box_vectors = boresch_pipeline.box_vectors

    complex_pipeline = TraditionalABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=config.temperature,
        platform_name=config.platform,
        output_dir=os.path.join(output_dir, "traditional_complex"),
    )

    log.info("🔄 开始传统复合物腿 REMD + MBAR (n_lambda=%d, n_steps=%d)",
             config.n_lambda, config.n_steps_per_window or 500000)
    complex_results = complex_pipeline.run_full(
        n_lambda=config.n_lambda,
        n_steps_per_leg=config.n_steps_per_window or 500000,
        boresch_correction=0.0,
        boresch_params=boresch_restraint,
        potential_type=config.potential,
        resume=config.resume and not config.reset,
    )

    sys_solv, top_solv, pos_solv, box_solv, lig_idx_solv = load_native_system(
        output_dir,
        phase="solvent",
        prefer_equilibrated=not config.reset,
    )
    pos_solv, box_solv = center_system_rigidly(pos_solv, box_solv, lig_idx_solv)
    solvent_pipeline = TraditionalABFEPipeline(
        system=sys_solv,
        topology=top_solv,
        positions=pos_solv,
        box_vectors=box_solv,
        ligand_indices=lig_idx_solv,
        temperature=config.temperature,
        platform_name=config.platform,
        output_dir=os.path.join(output_dir, "traditional_solvent"),
    )
    log.info("🔄 开始传统溶剂腿 REMD + MBAR (n_lambda=%d, n_steps=%d)",
             config.n_lambda, config.n_steps_per_window or 500000)
    solvent_results = solvent_pipeline.run_full(
        n_lambda=config.n_lambda,
        n_steps_per_leg=config.n_steps_per_window or 500000,
        boresch_correction=0.0,
        boresch_params=None,
        potential_type=config.potential,
        resume=config.resume and not config.reset,
    )

    dg_complex = float(complex_results["delta_G_total_kJ_mol"])
    dg_solvent = float(solvent_results["delta_G_total_kJ_mol"])
    err_complex = float(complex_results["error_leg_kJ_mol"])
    err_solvent = float(solvent_results["error_leg_kJ_mol"])
    # 🔑 [ATT-09] 循环闭合统一走 abfe_core.combine_binding_free_energy。
    # 这条路径 `TraditionalABFEPipeline.run_full` 是以 boresch_correction=0.0 调的，
    # 所以 complex 腿**不含**释放项，already_included=False，由 helper 减一次。
    #
    # 🔑 [ATT-09] 同时补上此前完全缺失的 APBS 项：main() 和 run_post_analysis()
    # 一直在读 config["apbs_correction_kJ_mol"]，只有 traditional 这条路径从来没读，
    # 等于对带电配体静默漏掉整项有限尺寸静电修正。这个标量由离线的
    # apbs_correction.py（prepare → run → collect）算出，collect 的输出里直接给了
    # 要传的 --apbs-correction-kj-mol；这里只是消费同一个值，不在流程内调 APBS。
    apbs_correction = float(config.get("apbs_correction_kJ_mol", 0.0) or 0.0)
    cycle = combine_binding_free_energy(
        dg_complex_kJ_mol=dg_complex,
        dg_solvent_kJ_mol=dg_solvent,
        err_complex_kJ_mol=err_complex,
        err_solvent_kJ_mol=err_solvent,
        dg_boresch_kJ_mol=dg_boresch,
        boresch_already_included_in_complex=False,
        apbs_correction_kJ_mol=apbs_correction,
    )
    delta_g_bind = cycle["delta_G_bind_kJ_mol"]
    total_err_bind = cycle["total_error_kJ_mol"]

    final = {
        "complex_leg": complex_results,
        "solvent_leg": solvent_results,
        "complex_delta_G_kJ_mol": dg_complex,
        "solvent_delta_G_kJ_mol": dg_solvent,
        "boresch_correction_kJ_mol": float(dg_boresch),
        # ✅ TraditionalABFEPipeline.run_full 是以 boresch_correction=0.0 调用的，
        # 因此 complex_delta_G_kJ_mol 不含 Boresch 释放修正；该修正只作为独立项
        # 已经减到 delta_G_bind_kJ_mol 里（见上方 delta_g_bind 公式），下游不要
        # 再对 complex_delta_G_kJ_mol 或 delta_G_bind_kJ_mol 二次扣减。
        "boresch_correction_already_included_in_complex_delta_G": False,
        "delta_G_bind_uncorrected_kJ_mol": float(
            cycle["delta_G_bind_uncorrected_kJ_mol"]
        ),
        "apbs_correction_kJ_mol": float(apbs_correction),
        "delta_G_bind_kJ_mol": float(delta_g_bind),
        "delta_G_bind_kcal_mol": float(delta_g_bind / 4.184),
        "total_error_kJ_mol": total_err_bind,
        "thermodynamic_cycle_terms": cycle,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(output_dir, "final_binding_results_traditional.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, cls=NumpyEncoder)
    log.info("✅ 传统 ABFE 完成: ΔG_bind = %.2f ± %.2f kJ/mol", delta_g_bind, total_err_bind)
    log.info("💾 传统模式最终结果已保存: %s", out_path)
    return final

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 复合物 charging-only 隔离重跑
# ---------------------------------------------------------------------------
def _load_frozen_stage_result(path: str, expected_stage: str) -> Dict:
    """Load a completed stage as a read-only input for an isolated rerun."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少冻结的 {expected_stage} 结果: {path}")
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("stage") != expected_stage:
        raise RuntimeError(
            f"冻结结果 stage={result.get('stage')!r}，期望 {expected_stage!r}: {path}"
        )
    for key in ("total_delta_G", "total_error"):
        value = result.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise RuntimeError(f"冻结结果 {path} 的 {key}={value!r} 非法")
    return result


def _load_frozen_stage2_boresch(output_dir: str) -> Dict:
    """Use the exact restraint Hamiltonian that produced frozen Stage 2."""
    stage2_path = os.path.join(
        os.path.abspath(output_dir),
        "checkpoints",
        "stage2_vanishing.json",
    )
    stage2 = _load_frozen_stage_result(stage2_path, "vanishing")
    params = (
        ((stage2.get("protocol_key") or {}).get("payload") or {}).get(
            "boresch_params"
        )
    )
    if not isinstance(params, dict):
        raise RuntimeError(
            f"冻结 Stage 2 未记录 Boresch Hamiltonian，无法安全重跑 charging: "
            f"{stage2_path}"
        )
    return _sanitize_boresch_params_strict(params)


def _boresch_core_signature(params: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(params, dict):
        return None
    params = _sanitize_boresch_params(params)
    return {
        "receptor_indices": [int(v) for v in params.get("receptor_indices", [])],
        "ligand_indices": [int(v) for v in params.get("ligand_indices", [])],
        "equilibrium_values": {
            str(k): round(float(v), 8)
            for k, v in (params.get("equilibrium_values") or {}).items()
        },
        "force_constants": {
            str(k): round(float(v), 8)
            for k, v in (params.get("force_constants") or {}).items()
        },
    }


def _prepare_charging_rerun_dir(
    output_dir: str, requested: Optional[str], label: str = "charging"
) -> str:
    """为「只跑某一条腿」的隔离重跑准备一个空目录。

    label 只影响默认子目录名与报错措辞；attachment 腿复用同一套隔离/防覆盖规则。
    """
    if requested:
        rerun_dir = os.path.abspath(requested)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rerun_dir = os.path.abspath(
            os.path.join(output_dir, f"{label}_rerun", stamp)
        )
    source_dir = os.path.abspath(output_dir)
    if rerun_dir == source_dir:
        raise RuntimeError(f"{label}-only 输出目录不能等于主 output 目录")
    if os.path.isdir(rerun_dir) and os.listdir(rerun_dir):
        raise FileExistsError(
            f"{label}-only 输出目录已存在且非空，拒绝覆盖: {rerun_dir}"
        )
    os.makedirs(rerun_dir, exist_ok=True)
    return rerun_dir


def _charging_mbar_crosschecks(
    u_kn_path: str,
    n_k_path: str,
    temperature_k: float,
    production_result: Dict,
) -> Dict:
    """Re-solve one u_kn with raw MBAR and adjacent BAR as diagnostics."""
    u_kn = np.load(u_kn_path, allow_pickle=False)
    n_k = np.load(n_k_path, allow_pickle=False).astype(int)
    analyzer = TraditionalMBARAnalyzer(temperature=temperature_k)
    analyzer._last_n_k = n_k
    raw = analyzer.solve(u_kn, decorrelate=False)

    from pymbar import other_estimators

    kt = 0.008314462618 * float(temperature_k)
    offsets = np.concatenate(([0], np.cumsum(n_k)))
    bar_delta_f = []
    bar_variance = []
    for k in range(len(n_k) - 1):
        start_k, end_k = int(offsets[k]), int(offsets[k + 1])
        start_j, end_j = int(offsets[k + 1]), int(offsets[k + 2])
        w_forward = u_kn[k + 1, start_k:end_k] - u_kn[k, start_k:end_k]
        w_reverse = u_kn[k, start_j:end_j] - u_kn[k + 1, start_j:end_j]
        edge = other_estimators.bar(
            w_forward,
            w_reverse,
            compute_uncertainty=True,
        )
        bar_delta_f.append(float(edge["Delta_f"]))
        bar_variance.append(float(edge["dDelta_f"]) ** 2)
    bar_dg = float(np.sum(bar_delta_f) * kt)
    bar_error = float(np.sqrt(np.sum(bar_variance)) * kt)

    prod_dg = float(production_result["total_delta_G"])
    prod_error = float(production_result["total_error"])
    tolerance = max(
        2.0,
        3.0 * max(prod_error, float(raw["error"]), bar_error),
    )
    return {
        "production_decorrelated_mbar": {
            "delta_G_kJ_mol": prod_dg,
            "error_kJ_mol": prod_error,
        },
        "raw_all_frame_mbar": {
            "delta_G_kJ_mol": float(raw["delta_G"]),
            "error_kJ_mol": float(raw["error"]),
        },
        "adjacent_bar_sum": {
            "delta_G_kJ_mol": bar_dg,
            "error_kJ_mol": bar_error,
            "uncertainty_note": (
                "相邻 BAR 方差按独立边近似相加，仅作为交叉检查。"
            ),
        },
        "acceptance_tolerance_kJ_mol": tolerance,
        "consistent": bool(
            abs(float(raw["delta_G"]) - prod_dg) <= tolerance
            and abs(bar_dg - prod_dg) <= tolerance
        ),
    }


def _run_boresch_attachment_only(
    config: RunConfig,
    source_pipeline: ABFEPipeline,
    boresch_restraint: Dict,
) -> str:
    """[P1-17] 只跑 Boresch attachment 腿 A′→A，复用冻结的 stage1/stage2/solvent。

    stage1/stage2 是在**受约束**系综里测的，补上 attachment 项不会改变它们，
    所以这条腿可以作为增量跑（~2–3 h），不必重跑整条复合物腿（~7 h）。
    """
    if config.reset:
        raise RuntimeError(
            "--only-boresch-attachment 不能与 --reset 同用；该模式必须只读复用"
            "现有 stage1/stage2 与 solvent 结果"
        )
    if config.decoupling != "dual_lambda":
        raise RuntimeError("--only-boresch-attachment 只支持 --decoupling dual_lambda")
    # [§9 / MEM-14] 增量重跑同样是在**消费**那次预平衡，所以门必须先过。
    # 否则"门失败 → 用 --only-* 重跑"就成了另一条绕过路径。可溶体系直接短路。
    source_pipeline.ensure_membrane_quality_gate_passed()
    # 运行时解析出的这组只是个参考——真正要用的是冻结腿 payload 里那组（见下）。
    # 所以这里不拦，解析不出来也能继续。
    if not _has_valid_boresch_anchors(boresch_restraint):
        log.warning(
            "本次运行没有解析出有效的 Boresch 锚点；将直接采用冻结 stage1/stage2 采样时用的那组"
        )

    from ibs_engine import run_boresch_attachment_leg

    source_dir = os.path.abspath(config.output)
    stage1 = _load_frozen_stage_result(
        os.path.join(source_dir, "checkpoints", "stage1_decharging.json"), "decharging"
    )
    stage2 = _load_frozen_stage_result(
        os.path.join(source_dir, "checkpoints", "stage2_vanishing.json"), "vanishing"
    )
    solvent_path = os.path.join(source_dir, "solvent_leg", "final_results.json")
    if not os.path.isfile(solvent_path):
        raise FileNotFoundError(f"缺少冻结的 solvent 最终结果: {solvent_path}")
    with open(solvent_path, encoding="utf-8") as handle:
        solvent = json.load(handle)
    for key in ("total_delta_G_complex_kJ_mol", "total_error_kJ_mol"):
        value = solvent.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise RuntimeError(f"冻结 solvent 结果的 {key}={value!r} 非法")

    # attachment 腿必须挂**冻结腿实际采样时用的那组**限制，而不是本次运行时解析
    # 出来的那组。两者经常不同且属正常：`--boresch-source simple` 解析的是
    # `boresch_simple.json`，而生产采样走 P0-10 的 commit 机制，用的是
    # `checkpoints/boresch_equilibrium_committed.json` 里重新锚定过的平衡值
    # （本例中 r0 0.42916 vs 0.43081、thetaA0 83.52° vs 83.69°）。
    # 挂错那组 = 给循环补的是另一个限制势的 attachment 项，循环照样不闭合。
    # 所以这里**不报错**，直接采用冻结值并把差异打出来。
    frozen_boresch = {}
    frozen_sigs = {}
    for name, frozen in (("stage1", stage1), ("stage2", stage2)):
        payload_bp = (
            ((frozen.get("protocol_key") or {}).get("payload") or {}).get("boresch_params")
        )
        if payload_bp is None:
            raise RuntimeError(f"冻结 {name} 的 protocol_key 里没有 boresch_params，无法确定它采样时用的限制势")
        frozen_boresch[name] = _sanitize_boresch_params(payload_bp)
        frozen_sigs[name] = _boresch_core_signature(payload_bp)

    # 这一条才是真正的硬矛盾：两条解耦腿如果用了不同的限制势，现有结果本身就废了。
    if frozen_sigs["stage1"] != frozen_sigs["stage2"]:
        raise RuntimeError(
            "冻结 stage1 与 stage2 用的 Boresch 限制势不同，现有复合物腿本身不自洽，"
            "拒绝在其上补 attachment 腿"
        )

    sampled_restraint = frozen_boresch["stage2"]

    # 🔑 三处必须是同一份 restraint signature，否则整条腿不是「同端点闭合实验」：
    #   (1) 传进来的 boresch_restraint —— rebalance 用它平衡了起始坐标
    #   (2) 冻结 stage1/stage2 采样时用的那组
    #   (3) 下面 apply_boresch_correction 算解析释放时用的那组
    # (3) 直接复用 (2)，所以这里只需把 (1) 和 (2) 对上。
    # 主流程已改成在 rebalance **之前**就加载冻结值，所以正常路径下这里必然一致；
    # 一旦不一致，说明有人改动了加载顺序 —— 那是硬错误，不能只打警告，
    # 因为起始系综会来自另一个限制势下的平衡态（λ=1 的短平衡消不掉它）。
    if _boresch_core_signature(boresch_restraint) != frozen_sigs["stage2"]:
        cur_eq = (_sanitize_boresch_params(boresch_restraint) or {}).get("equilibrium_values", {})
        frz_eq = sampled_restraint.get("equilibrium_values", {})
        diffs = []
        for key in sorted(set(cur_eq) | set(frz_eq)):
            a, b = cur_eq.get(key), frz_eq.get(key)
            if a is None or b is None or abs(float(a) - float(b)) > 1e-9:
                diffs.append(f"{key}: rebalance 用 {a} vs 冻结腿用 {b}")
        raise RuntimeError(
            "rebalance 用的 Boresch 参数与冻结 stage1/stage2 采样时用的那组不一致，"
            "起始坐标来自另一个限制势下的平衡态，attachment 腿不是同端点闭合实验。"
            "主流程应在 rebalance 之前就加载冻结值（见 runabfe.py 第 4 节）。\n  "
            + "\n  ".join(diffs)
        )
    log.info("🔒 rebalance / attachment / 解析释放 三处 Boresch signature 一致")
    boresch_restraint = sampled_restraint

    rerun_dir = _prepare_charging_rerun_dir(
        source_dir, config.get("attachment_rerun_dir"), label="attachment"
    )
    rerun_pipeline = ABFEPipeline(
        system=source_pipeline.system,
        topology=source_pipeline.topology,
        positions=source_pipeline.positions,
        box_vectors=source_pipeline.box_vectors,
        ligand_indices=source_pipeline.ligand_indices,
        temperature=config.temperature,
        output_dir=rerun_dir,
        checkpoint_dir=os.path.join(rerun_dir, "checkpoints"),
        platform_name=config.platform,
        # 🔑 rerun **继承 source pipeline 已解析的协议**，不重读 config——
        # 增量重跑必须与被复用的那次用同一个恒压器集合，而 config 可能已经改过。
        environment_type=source_pipeline.environment_type,
        membrane=source_pipeline.barostat_protocol.get("membrane"),
        confirm_soluble_with_lipids=bool(
            source_pipeline.barostat_protocol.get("soluble_with_lipids_confirmed", False)
        ),
        # 色散路线同样继承：换了它 u_kn 口径就变，增量段与被复用段不可比。
        dispersion_protocol=source_pipeline.dispersion_protocol,
        forcefield_family=source_pipeline.dispersion_protocol_info.get(
            "forcefield_family"
        ),
        # 净电荷路线同样继承：换了它 co-ion 身份与 charging 哈密顿量就变。
        charge_treatment=getattr(source_pipeline, "charge_treatment", None),
    )
    run_provenance = _write_run_provenance(
        rerun_dir,
        config,
        rerun_pipeline.system,
        rerun_pipeline.topology,
        rerun_pipeline.positions,
        barostat_protocol=rerun_pipeline.barostat_protocol,
    )

    stage0 = run_boresch_attachment_leg(
        source_pipeline.system,
        source_pipeline.topology,
        source_pipeline.positions,
        source_pipeline.box_vectors,
        boresch_restraint,
        temperature_k=float(config.temperature),
        lambdas=config.get("attachment_lambdas"),
        n_steps_per_state=int(config.get("attachment_n_steps_per_state", 250_000)),
        equil_steps_per_state=int(config.get("attachment_equil_steps_per_state", 50_000)),
        steps_per_sample=int(config.get("attachment_steps_per_sample", 1_000)),
        platform_name=config.platform,
        seed=int(config.get("attachment_seed", 20260728)),
        n_seeds=int(config.get("attachment_n_seeds", 1)),
        output_dir=os.path.join(rerun_dir, "attachment"),
        log=log.info,
    )
    os.makedirs(rerun_pipeline.checkpoint_dir, exist_ok=True)
    with open(
        os.path.join(rerun_pipeline.checkpoint_dir, "stage0_attachment.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(stage0, handle, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    correction = rerun_pipeline.apply_boresch_correction(
        boresch_restraint, autoload_from_disk=False
    )
    rerun_pipeline._last_run_config = {
        "potential_type": config.potential,
        "mode": "only_boresch_attachment",
    }
    complex_candidate = rerun_pipeline.compute_final_results(
        {"stage0": stage0, "stage1": stage1, "stage2": stage2},
        correction,
        system=rerun_pipeline.system,
        decoupling_scheme="dual_lambda",
    )
    cycle = combine_binding_free_energy(
        dg_complex_kJ_mol=float(complex_candidate["total_delta_G_complex_kJ_mol"]),
        dg_solvent_kJ_mol=float(solvent["total_delta_G_complex_kJ_mol"]),
        err_complex_kJ_mol=float(complex_candidate["total_error_kJ_mol"]),
        err_solvent_kJ_mol=float(solvent["total_error_kJ_mol"]),
        dg_boresch_kJ_mol=float(complex_candidate["boresch_correction_kJ_mol"]),
        boresch_already_included_in_complex=True,
        apbs_correction_kJ_mol=float(config.get("apbs_correction_kJ_mol", 0.0) or 0.0),
    )

    candidate = {
        "mode": "only_boresch_attachment_candidate",
        "promoted_to_primary_results": False,
        "attachment_delta_G_kJ_mol": float(stage0["attachment_delta_G_kJ_mol"]),
        "attachment_error_kJ_mol": float(stage0["attachment_error_kJ_mol"]),
        "attachment_delta_G_kcal_mol": float(stage0["attachment_delta_G_kJ_mol"]) / 4.184,
        "complex_delta_G_kJ_mol": float(complex_candidate["total_delta_G_complex_kJ_mol"]),
        "solvent_delta_G_kJ_mol": float(solvent["total_delta_G_complex_kJ_mol"]),
        "delta_G_bind_kJ_mol": float(cycle["delta_G_bind_kJ_mol"]),
        "delta_G_bind_kcal_mol": float(cycle["delta_G_bind_kJ_mol"]) / 4.184,
        "total_error_kJ_mol": float(cycle["total_error_kJ_mol"]),
        "thermodynamic_cycle_terms": cycle,
        "attachment_stage": stage0,
        "frozen_inputs": {
            "stage1_total_delta_G": float(stage1["total_delta_G"]),
            "stage2_total_delta_G": float(stage2["total_delta_G"]),
            "solvent_final_results": solvent_path,
        },
        "provenance": run_provenance,
        "timestamp": datetime.now().isoformat(),
    }
    candidate_path = os.path.join(rerun_dir, "final_binding_results_candidate.json")
    with open(candidate_path, "w", encoding="utf-8") as handle:
        json.dump(candidate, handle, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    log.info(
        "ΔG_attach = %.4f ± %.4f kJ/mol (%.4f kcal/mol)；参考 result.txt 的对应项是 +0.442 kcal/mol",
        candidate["attachment_delta_G_kJ_mol"],
        candidate["attachment_error_kJ_mol"],
        candidate["attachment_delta_G_kcal_mol"],
    )
    log.info(
        "ΔG_bind 候选 = %.4f kJ/mol = %.4f kcal/mol（原 −2.121）",
        candidate["delta_G_bind_kJ_mol"],
        candidate["delta_G_bind_kcal_mol"],
    )
    return candidate_path


def _run_complex_charging_only(
    config: RunConfig,
    source_pipeline: ABFEPipeline,
    boresch_restraint: Dict,
) -> str:
    if config.reset:
        raise RuntimeError(
            "--only-complex-charging 不能与 --reset 同用；该模式必须只读复用"
            "现有 Stage 2 与 solvent 结果"
        )
    if config.decoupling != "dual_lambda":
        raise RuntimeError("--only-complex-charging 只支持 --decoupling dual_lambda")
    if config.get("decharge_method", "pme") != "pme":
        raise RuntimeError("--only-complex-charging 只支持生产级 PME charging 路径")
    if not isinstance(boresch_restraint, dict):
        raise RuntimeError("charging-only 需要现有 complex Boresch 参数")
    # [§9 / MEM-14] 同上：增量重跑也在消费那次预平衡，门必须先过。
    source_pipeline.ensure_membrane_quality_gate_passed()

    source_dir = os.path.abspath(config.output)
    stage2_path = os.path.join(source_dir, "checkpoints", "stage2_vanishing.json")
    solvent_path = os.path.join(source_dir, "solvent_leg", "final_results.json")
    stage2 = _load_frozen_stage_result(stage2_path, "vanishing")
    if not os.path.isfile(solvent_path):
        raise FileNotFoundError(f"缺少冻结的 solvent 最终结果: {solvent_path}")
    with open(solvent_path, encoding="utf-8") as handle:
        solvent = json.load(handle)
    for key in ("total_delta_G_complex_kJ_mol", "total_error_kJ_mol"):
        value = solvent.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise RuntimeError(f"冻结 solvent 结果的 {key}={value!r} 非法")

    stage2_boresch = (
        ((stage2.get("protocol_key") or {}).get("payload") or {}).get(
            "boresch_params"
        )
    )
    if _boresch_core_signature(stage2_boresch) != _boresch_core_signature(
        boresch_restraint
    ):
        raise RuntimeError(
            "当前 Boresch 参数与冻结 Stage 2 的协议不一致；拒绝拼接不同限制势的腿"
        )

    rerun_dir = _prepare_charging_rerun_dir(
        source_dir, config.get("charging_rerun_dir")
    )
    rerun_pipeline = ABFEPipeline(
        system=source_pipeline.system,
        topology=source_pipeline.topology,
        positions=source_pipeline.positions,
        box_vectors=source_pipeline.box_vectors,
        ligand_indices=source_pipeline.ligand_indices,
        temperature=config.temperature,
        output_dir=rerun_dir,
        checkpoint_dir=os.path.join(rerun_dir, "checkpoints"),
        platform_name=config.platform,
        # 🔑 rerun **继承 source pipeline 已解析的协议**，不重读 config——
        # 增量重跑必须与被复用的那次用同一个恒压器集合，而 config 可能已经改过。
        environment_type=source_pipeline.environment_type,
        membrane=source_pipeline.barostat_protocol.get("membrane"),
        confirm_soluble_with_lipids=bool(
            source_pipeline.barostat_protocol.get("soluble_with_lipids_confirmed", False)
        ),
        # 色散路线同样继承：换了它 u_kn 口径就变，增量段与被复用段不可比。
        dispersion_protocol=source_pipeline.dispersion_protocol,
        forcefield_family=source_pipeline.dispersion_protocol_info.get(
            "forcefield_family"
        ),
        # 净电荷路线同样继承：换了它 co-ion 身份与 charging 哈密顿量就变。
        charge_treatment=getattr(source_pipeline, "charge_treatment", None),
    )
    run_provenance = _write_run_provenance(
        rerun_dir,
        config,
        rerun_pipeline.system,
        rerun_pipeline.topology,
        rerun_pipeline.positions,
        barostat_protocol=rerun_pipeline.barostat_protocol,
    )

    n_states = int(config.get("stage1_n_states", 12))
    max_contexts = config.get("charging_max_resident_contexts")
    if max_contexts is not None and int(max_contexts) < 1:
        raise RuntimeError("--charging-max-resident-contexts 必须至少为 1")
    lambdas_coul = np.linspace(1.0, 0.0, n_states).tolist()
    lambdas_vdw = [1.0] * n_states
    expected_meta = _pme_u_kn_meta_payload(
        n_states=n_states,
        lambdas_coul=lambdas_coul,
        lambdas_vdw=lambdas_vdw,
        temperature_k=float(config.temperature),
        system=rerun_pipeline.system,
        topology=rerun_pipeline.topology,
        ligand_indices=rerun_pipeline.ligand_indices,
        boresch_params=boresch_restraint,
    )
    manifest_path = os.path.join(rerun_dir, "charging_sampling_manifest.json")
    manifest = {
        "mode": "only_complex_charging",
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "pid": os.getpid(),
        "fresh_sampling_required": True,
        "resume": False,
        "sampling_and_analysis_same_process": True,
        "remd_execution": {
            "requested_platform": str(config.platform),
            "max_resident_contexts": (
                None if max_contexts is None else int(max_contexts)
            ),
        },
        "expected_pme_u_kn_metadata": expected_meta,
        "frozen_inputs": {
            "stage2_vanishing": {
                "path": stage2_path,
                "sha256": _sha256_file(stage2_path),
                "delta_G_kJ_mol": float(stage2["total_delta_G"]),
                "error_kJ_mol": float(stage2["total_error"]),
            },
            "solvent_final": {
                "path": solvent_path,
                "sha256": _sha256_file(solvent_path),
                "delta_G_kJ_mol": float(
                    solvent["total_delta_G_complex_kJ_mol"]
                ),
                "error_kJ_mol": float(solvent["total_error_kJ_mol"]),
            },
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, cls=NumpyEncoder)

    try:
        stage1 = rerun_pipeline._run_dual_lambda_stage(
            "decharging",
            fixed_lam_coul=1.0,
            fixed_lam_vdw=1.0,
            n_states=n_states,
            n_steps_per_window=int(config.n_steps_per_window),
            steps_per_update=int(config.steps_per_update),
            system_type="complex",
            resume=False,
            potential_type=config.potential,
            dexp_params=_load_dexp_params_fail_closed(
                config.potential,
                config.dexp_params,
            ),
            optimized_lambdas=lambdas_coul,
            window_ranges=None,
            enable_early_stop=False,
            boresch_params=boresch_restraint,
            remd_max_resident_contexts=max_contexts,
        )
        rerun_pipeline._assert_stage_result_sane(
            "isolated complex decharging", stage1
        )

        decharging_dir = os.path.join(rerun_dir, "decharging")
        meta_path = os.path.join(
            decharging_dir, "decharging_pme_u_kn.meta.json"
        )
        u_kn_path = os.path.join(decharging_dir, "decharging_pme_u_kn.npy")
        n_k_path = u_kn_path + ".n_k.npy"
        with open(meta_path, encoding="utf-8") as handle:
            actual_meta = json.load(handle)
        if actual_meta != expected_meta:
            raise RuntimeError(
                "charging 采样前 manifest 与离线 u_kn Hamiltonian 指纹不一致"
            )

        crosschecks = _charging_mbar_crosschecks(
            u_kn_path,
            n_k_path,
            float(config.temperature),
            stage1,
        )
        if not crosschecks["consistent"]:
            raise RuntimeError(
                "charging 的去相关 MBAR、全帧 MBAR 与相邻 BAR 未通过一致性检查"
            )

        correction = rerun_pipeline.apply_boresch_correction(
            boresch_restraint, autoload_from_disk=False
        )
        rerun_pipeline._last_run_config = {
            "potential_type": config.potential,
            "mode": "only_complex_charging",
        }
        complex_candidate = rerun_pipeline.compute_final_results(
            {"stage1": stage1, "stage2": stage2},
            correction,
            system=rerun_pipeline.system,
            decoupling_scheme="dual_lambda",
        )
        dg_complex = float(
            complex_candidate["total_delta_G_complex_kJ_mol"]
        )
        err_complex = float(complex_candidate["total_error_kJ_mol"])
        dg_solvent = float(solvent["total_delta_G_complex_kJ_mol"])
        err_solvent = float(solvent["total_error_kJ_mol"])
        dg_boresch = float(complex_candidate["boresch_correction_kJ_mol"])
        apbs = float(config.get("apbs_correction_kJ_mol", 0.0) or 0.0)
        cycle = combine_binding_free_energy(
            dg_complex_kJ_mol=dg_complex,
            dg_solvent_kJ_mol=dg_solvent,
            err_complex_kJ_mol=err_complex,
            err_solvent_kJ_mol=err_solvent,
            dg_boresch_kJ_mol=dg_boresch,
            boresch_already_included_in_complex=True,
            apbs_correction_kJ_mol=apbs,
        )
        binding_candidate = {
            "mode": "only_complex_charging_candidate",
            "promoted_to_primary_results": False,
            "complex_delta_G_kJ_mol": dg_complex,
            "solvent_delta_G_kJ_mol": dg_solvent,
            "boresch_correction_kJ_mol": dg_boresch,
            "boresch_correction_already_included_in_complex_delta_G": True,
            "delta_G_bind_kJ_mol": float(cycle["delta_G_bind_kJ_mol"]),
            "delta_G_bind_kcal_mol": float(
                cycle["delta_G_bind_kJ_mol"] / 4.184
            ),
            "total_error_kJ_mol": float(cycle["total_error_kJ_mol"]),
            "thermodynamic_cycle_terms": cycle,
            "charging_crosschecks": crosschecks,
            "frozen_inputs": manifest["frozen_inputs"],
            "provenance": run_provenance,
            "timestamp": datetime.now().isoformat(),
        }
        binding_path = os.path.join(
            rerun_dir, "final_binding_results_candidate.json"
        )
        with open(binding_path, "w", encoding="utf-8") as handle:
            json.dump(binding_candidate, handle, indent=2, cls=NumpyEncoder)

        dcd_inventory = []
        for path in sorted(glob.glob(os.path.join(decharging_dir, "*.dcd"))):
            stat_result = os.stat(path)
            dcd_inventory.append(
                {
                    "path": path,
                    "size_bytes": int(stat_result.st_size),
                    "mtime_ns": int(stat_result.st_mtime_ns),
                }
            )
        exchange_path = os.path.join(
            decharging_dir, "decharging_exchange_diagnostics.json"
        )
        exchange_diagnostics = None
        if os.path.isfile(exchange_path):
            with open(exchange_path, encoding="utf-8") as handle:
                exchange_diagnostics = json.load(handle)
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "actual_pme_u_kn_metadata": actual_meta,
                "metadata_match": True,
                "dcd_inventory": dcd_inventory,
                "exchange_diagnostics": exchange_diagnostics,
                "charging_result": stage1,
                "crosschecks": crosschecks,
                "candidate_binding_result": binding_path,
            }
        )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, cls=NumpyEncoder)
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now().isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, cls=NumpyEncoder)
        raise

    log.info("✅ complex charging-only 候选结果完成: %s", binding_path)
    return binding_path


# 主入口
# ---------------------------------------------------------------------------
def main():
    args = parse_arguments()

    if args.command == "self-test":
        sys.exit(run_self_tests())

    if args.command == "prepare":
        run_prepare_command(args)
        return

    if args.command == "refine-lambda-path":
        from abfe_preoptimizer import refine_stage_lambda_path_from_data
        result = refine_stage_lambda_path_from_data(
            stage_dir=args.stage_dir,
            preopt_path=args.preopt_file,
            temperature_k=args.temperature,
            n_states=args.n_states,
            max_window_span_kJ=args.max_window_span_kj,
            overlap=args.overlap,
            stage_type=args.stage_type,
        )
        log.info(
            "✅ 已按累积|Δf|重新设计 λ 路径：%d 个状态，%d 个窗口（旧文件备份为 %s.bak）",
            result["n_states"], len(result["window_ranges"]), args.preopt_file,
        )
        log.info("下次 --resume 时，形状不匹配的窗口会被自动判定为需要重新采样。")
        return

    # 创建配置对象
    config = RunConfig(args)
    dexp_params = _load_dexp_params_fail_closed(
        config.potential,
        config.dexp_params,
    )
    if not config.ligand:
        log.error("未提供配体残基名称。请通过 --ligand 或配置文件中的 ligand 指定。")
        sys.exit(2)
    if config.only_complex_charging and config.reset:
        raise RuntimeError(
            "--only-complex-charging 不能与 --reset 同用；拒绝在读取冻结结果前"
            "重建或覆盖主输出缓存"
        )
    if config.only_complex_charging and not config.resume:
        raise RuntimeError(
            "--only-complex-charging 必须与 --resume 同用，以只读加载现有 committed "
            "Boresch、Stage 2 和 solvent 结果"
        )
    if config.only_boresch_attachment and config.only_complex_charging:
        raise RuntimeError(
            "--only-boresch-attachment 与 --only-complex-charging 互斥"
        )
    if config.only_boresch_attachment and config.reset:
        raise RuntimeError(
            "--only-boresch-attachment 不能与 --reset 同用；拒绝在读取冻结结果前"
            "重建或覆盖主输出缓存"
        )
    if config.only_boresch_attachment and not config.resume:
        raise RuntimeError(
            "--only-boresch-attachment 必须与 --resume 同用，以只读加载现有 committed "
            "Boresch、stage1/stage2 和 solvent 结果"
        )
    # 分析模式单独处理
    if args.analyze_only:
        run_post_analysis(config)
        return
    # 传统模式单独处理
    if config.mode == "traditional":
        run_traditional_mode(config)
        return

    # 准备输出目录
    output_dir = config.output
    os.makedirs(output_dir, exist_ok=True)

    # ----- 1. 系统加载：优先从缓存，否则 GROMACS 构建并立即落盘 -----
    #
    # 环境类型必须在**加载 System 之前**就知道：膜体系要求 `load_native_system`
    # 返回**带键**的拓扑（mmCIF 不保存脂质/配体的键，而 PBC 分子完整性修复靠键）。
    # 这里只用 config 判环境类型——完整的膜协议解析（要 topology 做脂质交叉检查）
    # 仍在下面 `resolve_membrane_protocol` 那一处，两者取值一致由下方断言保证。
    _environment_type = resolve_environment_type(config.get("system_type"))
    _require_bonded_topology = _environment_type == "membrane"

    include_dir = find_gmx_include_dir(config.gmx_path)
    main_cache_identity = _main_cache_identity(
        config.gro, config.top, config.ligand, include_dir
    )
    if not config.reset and system_cache_exists(
        output_dir,
        config.gro,
        config.top,
        config.ligand,
        include_dir,
    ):
        system, topology, positions, box_vectors, ligand_indices = load_native_system(
            output_dir, 
            gro_file=config.gro, 
            top_file=config.top, 
            gmx_include_dir=include_dir,
            # 膜体系必须拿**带键**的拓扑：mmCIF 不保存脂质/配体的键，而
            # PBC 分子完整性修复靠键判断分子边界，键丢了会把跨边界脂质撕开。
            require_bonded_topology=_require_bonded_topology,
        )
        log.info("♻️ 从缓存加载 System 完成")
        # 缓存命中时不需要再转换（缓存里存的是转换后构建出来的 System）。
        # 转换版本已进主缓存指纹，所以指纹一致就代表转换口径一致。
        _pairs_conversion = None
    else:
        if not config.gro or not config.top:
            log.error("未提供 --gro/--top 且无缓存，无法构建系统")
            sys.exit(1)
        # OpenMM 的 GromacsTopFile 只接受 `[ pairs ]` funct 1；CHARMM-GUI 的
        # AMBER FF-Converter 会对部分对写 funct 2（多出 fudgeQQ/q1/q2 三列）。
        # 那三列经**逐对校验**若与全局 fudgeQQ 和 `[ atoms ]` 电荷一致，就是冗余重述，
        # 可等价转成 funct 1；不一致则 fail closed（硬转会静默改变哈密顿量）。
        # 转换写到 output_dir 下的独立目录，**原始输入一个字节都不动**。
        # 主路径显式指定转换目录，让产物落在 output_dir 里便于审计；
        # 下游各加载点即使自己再调一次，也会命中同一份（幂等 + 内容寻址）。
        _top_for_openmm, _pairs_conversion = openmm_compatible_gromacs_top(
            config.top,
            include_dir,
            compat_dir=os.path.join(output_dir, "gromacs_openmm_compat"),
        )
        if _pairs_conversion:
            log.info(
                "🔧 [ pairs ] funct 2（OpenMM 不支持）已等价转换 → %s"
                "（%s 条；原始输入未修改）",
                _top_for_openmm,
                _pairs_conversion.get("n_pairs_converted", "复用已有"),
            )

        # 从 GROMACS 构建（build_system_from_gromacs 内部也走唯一入口，这里传
        # 转换后的路径只是为了让 provenance 记录的就是实际用的那份）。
        system, topology, positions, box_vectors, ligand_indices = build_system_from_gromacs(
            config.gro, _top_for_openmm, config.ligand,
            include_dir
        )
        diagnose_14_scaling(system)
        # 立即保存为原生缓存
        save_native_system(
            output_dir,
            system,
            topology,
            ligand_indices,
            positions,
            box_vectors,
            cache_identity=main_cache_identity,
        )
        log.info("✅ 原生缓存已生成")
        # 重新从缓存加载，确保后续所有对象都来自落盘文件
        # 替换原有的 load_native_system 调用为：
        # ⚠️ 注意这一步：即使是**全新构建**，这里也会立刻从落盘文件重新读回，
        # 所以生产的每一次运行（首跑或缓存命中）都会拿到 `load_native_system` 的
        # 拓扑。膜体系必须要求带键拓扑，否则首跑同样会被 mmCIF 丢键坑到——
        # 这正是"直接用 build_system_from_gromacs 复现不出 NaN"的原因。
        system, topology, positions, box_vectors, ligand_indices = load_native_system(
            output_dir, 
            gro_file=config.gro, 
            top_file=config.top, 
            gmx_include_dir=include_dir,
            prefer_equilibrated=False,
            require_bonded_topology=_require_bonded_topology,
        )
        log.info("🔄 已从缓存重新加载 System (使用落盘后对象)")

    # 坐标安全处理：转换为纯 numpy 再转为 Vec3 列表（防止类型问题）
    if hasattr(positions, "value_in_unit"):
        pos_nm = positions.value_in_unit(unit.nanometer)
    else:
        pos_nm = positions
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    if pos_nm.ndim == 1:
        pos_nm = pos_nm.reshape(-1, 3)
    positions = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_nm] * unit.nanometer
    # ========== PBC 确定性居中 ==========
    positions, box_vectors = center_system_rigidly(
        positions, box_vectors, ligand_indices
    )
    # 🔑 [P1-14] 这里**只是**整体质心平移。原来这行写的是"分子完整性修复完毕"，
    # 是假的：center_system_rigidly 的 docstring 自己就写着"彻底禁用原子级 PBC
    # Wrap"，它不碰拓扑、修不了任何跨盒断裂的分子。真正的整分子周期平移是
    # ABFEPipeline.repair_pbc_molecule_integrity()，现已提前到第一次建 Context /
    # 最小化 / 预平衡之前执行（pre_equilibrate 开头），并在失败时 fail closed。
    log.info("  ✅ 配体已居中（仅整体质心平移；整分子 PBC 修复在 pre_equilibrate 之前执行）")
    # 膜协议在这里先解析一次：provenance 写在 Pipeline 构建之前，而 §6.1 要求
    # 协议组合的 fail-closed 检查在创建任何 Context 之前完成。下面 ABFEPipeline
    # 会用同一个纯函数、同一份输入再解析一次，结果必然一致（单一实现，无第二套判据）。
    _barostat_protocol = resolve_membrane_protocol(
        config.get("system_type"),
        membrane_config=config.get("membrane"),
        topology=topology,
        confirm_soluble_with_lipids=bool(
            config.get("confirm_soluble_with_lipids", False)
        ),
    )
    # 两处环境类型解析必须一致（前者只看 config，后者还做脂质交叉检查）。
    # 不一致意味着有人给其中一条加了额外来源，那会让"要不要带键拓扑"与
    # "用哪种 barostat"基于不同的判断，属静默不一致。
    if _barostat_protocol["system_type"] != _environment_type:
        raise SystemExit(
            f"环境类型解析不一致：加载 System 前判为 {_environment_type!r}，"
            f"完整协议解析判为 {_barostat_protocol['system_type']!r}。"
            "两者必须同源（都来自 config 的 system_type）。"
        )

    if _barostat_protocol["system_type"] != "soluble":
        log.info(
            "🧫 环境类型=%s，复合物腿将使用 %s",
            _barostat_protocol["system_type"],
            _barostat_protocol["barostat_class"],
        )

    # 净电荷处理协议（memtodolist §1.2，B2）。同样在建任何 Context 之前判死，
    # 且**自动计算配体净电荷并与配置交叉核对**（§6.1）。
    # 净电荷用 ibs_engine 里既有的唯一实现算，不另造一套判据。
    _ligand_net_charge_e = _compute_ligand_net_charge(system, ligand_indices)
    _charge_protocol = resolve_charge_treatment(
        config.get("charge_treatment"),
        ligand_net_charge_e=_ligand_net_charge_e,
        apbs_correction_kJ_mol=config.get("apbs_correction_kJ_mol", 0.0),
        co_alchemical_ion=(
            _load_json_object_file(config.get("co_alchemical_ion"), "co-alchemical ion 规格")
            if config.get("co_alchemical_ion")
            else None
        ),
        environment_type=config.get("system_type"),
        apbs_evidence=(
            _load_json_object_file(config.get("apbs_evidence"), "APBS 证据")
            if config.get("apbs_evidence")
            else None
        ),
    )
    if _charge_protocol["charge_treatment"] != "neutral":
        log.info(
            "⚡ 净电荷协议=%s（配体净电荷 %+d e，APBS %s）",
            _charge_protocol["charge_treatment"],
            _charge_protocol["ligand_net_charge_e"],
            "适用" if _charge_protocol["apbs_applicable"] else "不适用",
        )
        if _charge_protocol["experimental_not_for_production"]:
            log.warning(
                "⚠️ charge_treatment=co_annihilation_experimental 是**实验对照专用**，"
                "其数值不得进入任何 ΔG_bind 汇总（memtodolist MEM-00a-2）。"
            )
        if (
            _charge_protocol["charge_treatment"]
            == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
            and not CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED
        ):
            # 开跑前就说清楚，而不是让人烧几小时复合物腿再在溶剂腿撞墙。
            # 真正的门在 `build_and_cache_solvent_leg`（唯一一处），这里只负责告知。
            log.warning(
                "⚠️ charge-transfer 的**溶剂腿 builder 尚未落地（B4）**：本次运行可以跑"
                "复合物腿（B3 的 charging 哈密顿量已实现，适合做 pilot / λ 阶梯重估），"
                "但热力学循环闭不上 —— **不得报出 ΔG_bind**。到溶剂腿会 fail closed。"
            )

    # 力场族与 LJ/色散路线（memtodolist §1.1 / §1.3，B6）。同样在建 Context 之前判死。
    # 力场族识别只在**膜体系或用户显式声明了色散路线/力场族**时才强制——可溶体系
    # 走 legacy_uniform_density_lrc，与识别结果无关，不能因为识别不出就挡住现有运行。
    _forcefield_family_info = None
    _needs_forcefield_family = (
        _barostat_protocol["system_type"] != "soluble"
        or config.get("dispersion_protocol")
        or config.get("forcefield_family")
    )
    if _needs_forcefield_family:
        _forcefield_family_info = resolve_forcefield_family(
            top_path=config.get("top"),
            explicit_family=config.get("forcefield_family"),
        )
        log.info(
            "🧬 力场族=%s（来源 %s）",
            _forcefield_family_info["family"],
            _forcefield_family_info["source"],
        )
    _dispersion_protocol = resolve_dispersion_protocol(
        config.get("dispersion_protocol"),
        environment_type=config.get("system_type"),
        forcefield_family=(
            _forcefield_family_info["family"] if _forcefield_family_info else None
        ),
        force_switch_deviation_evidence=(
            _load_json_object_file(
                config.get("force_switch_deviation_evidence"),
                "force-switch 偏差论证",
            )
            if config.get("force_switch_deviation_evidence")
            else None
        ),
    )
    if not _dispersion_protocol["was_defaulted"]:
        log.info(
            "🧪 色散路线=%s（均匀密度 LRC %s）",
            _dispersion_protocol["dispersion_protocol"],
            "生效" if _dispersion_protocol["uniform_density_lrc_active"] else "关闭",
        )

    # ---- §3.3 膜输入核对 + §9/§15 预平衡时长门（只在膜体系生效）----
    _membrane_input_report = None
    if _barostat_protocol["system_type"] == "membrane":
        _declaration_path = config.get("membrane_input_declaration")
        if not _declaration_path:
            raise SystemExit(
                "system_type=membrane 需要 --membrane-input-declaration：§3.3 要求输入是"
                "已完成膜构建和主要平衡的体系，且构建工具/参数/最终平衡作业/来源结构"
                "可追溯。缺这份声明就无法核对，拒绝继续。"
            )
        _membrane_declared = _load_json_object_file(_declaration_path, "膜输入声明")

        # 身份一律以 `.top` 的 [ molecules ] + [ moleculetype ] 为准，不靠残基名猜。
        # 实测理由：CHARMM-GUI 的 AMBER 体系里一个 POPC 是三个残基（PA+PC+OL）、
        # 水叫 TP3、离子叫 Na+/Cl-，靠残基名会把脂质数数成 3 倍、水和离子数成 0。
        _system_composition = classify_system_composition(
            parse_gromacs_topology(
                config.get("top"), find_gmx_include_dir(config.get("gmx_path"))
            ),
            ligand_molecule_name=_membrane_declared.get("ligand_molecule_name"),
            declared_roles=_membrane_declared.get("molecule_roles"),
        )
        log.info(
            "🧾 体系组成（来自 .top）：%s",
            {
                name: f"{role}×{_system_composition['molecule_counts'][name]}"
                for name, role in _system_composition["roles"].items()
            },
        )
        if _system_composition["n_atoms_total"] != topology.getNumAtoms():
            raise SystemExit(
                f".top 展开得到 {_system_composition['n_atoms_total']} 个原子，"
                f"但拓扑有 {topology.getNumAtoms()} 个。两者必须一致，否则原子索引"
                "对不上，后续所有按索引的选择都会静默错位。"
            )

        _membrane_input_report = validate_membrane_input(
            topology,
            positions,
            box_vectors,
            declared=_membrane_declared,
            normal_axis=(_barostat_protocol["membrane"] or {}).get("normal_axis", "z"),
            composition=_system_composition,
        )
        log.info(
            "🧫 膜输入核对通过：%d 原子，上叶 %d / 下叶 %d 脂质（不平衡 %.1f%%），"
            "水 %d，离子 %s",
            _membrane_input_report["n_atoms"],
            _membrane_input_report["leaflets"]["n_upper"],
            _membrane_input_report["leaflets"]["n_lower"],
            100.0 * _membrane_input_report["leaflets"]["imbalance_fraction"],
            _membrane_input_report["n_water"],
            _membrane_input_report["ion_counts"],
        )

        # §9/§15：把"拿可溶体系的 10 ns 默认值去跑膜"挡在配置阶段，而不是等质量门
        # 事后否掉——后者要先烧掉整轮采样才知道。
        _equil_steps = int(config.get("n_equil_steps", 5_000_000))
        _timestep_ps = float(config.get("timestep_ps", 0.002))
        _equil_ns = _equil_steps * _timestep_ps / 1000.0
        # 上游平衡的两条合法表述已由 validate_membrane_input 裁决过，这里只用结论。
        _upstream_ns = _membrane_input_report["upstream_equilibration_ns"]
        _nominal_precheck = _membrane_input_report[
            "nominal_equilibration_precheck_applicable"
        ]
        _shortfall_note = _membrane_declared.get("equilibration_shortfall_justification")
        if not _nominal_precheck:
            # 上游生产已完成但时长不可考 → 标称时长预检不适用。
            # §9 的实测质量门（末段漂移 + 脂质横向弛豫时间尺度）仍会独立判定。
            _total_equil_ns = None
            log.info(
                "🧫 上游平衡：%s（证据 %s）→ 标称时长预检不适用；"
                "本流程再平衡 %.1f ns。§9 实测质量门仍为硬门。",
                _membrane_input_report["upstream_equilibration_status"],
                _membrane_declared.get("final_equilibration_job"),
                _equil_ns,
            )
        else:
            _total_equil_ns = float(_upstream_ns) + _equil_ns
            log.info(
                "🧫 平衡时长：上游 %.1f ns + 本流程 %.1f ns = %.1f ns（下限 %.0f ns）",
                _upstream_ns, _equil_ns, _total_equil_ns, MEMBRANE_MIN_EQUILIBRATION_NS,
            )
        if _total_equil_ns is not None and _total_equil_ns < MEMBRANE_MIN_EQUILIBRATION_NS:
            if not _shortfall_note:
                raise SystemExit(
                    f"膜体系总平衡时长 {_total_equil_ns:.1f} ns"
                    f"（上游声明 {_upstream_ns:.1f} + 本流程 {_equil_ns:.1f}）"
                    f"低于下限 {MEMBRANE_MIN_EQUILIBRATION_NS:.0f} ns。"
                    "§9：通用 10 ns 不是膜平衡充分性的证明；§15：膜体系预平衡 ≥ 100 ns。"
                    "正解是调大 n_equil_steps 或延长上游平衡。"
                    "确有理由放行时，在膜输入声明里填 "
                    "equilibration_shortfall_justification 写明依据（会进 provenance）——"
                    "但注意这**只**放开这道标称时长预检，"
                    "§9 基于实测漂移与脂质横向弛豫时间的质量门仍然是硬门，不受影响。"
                )
            log.warning(
                "⚠️ 膜平衡标称时长不足（%.1f < %.0f ns），已按声明的理由放行：%s。"
                "§9 的实测质量门仍会独立判定。",
                _total_equil_ns, MEMBRANE_MIN_EQUILIBRATION_NS, _shortfall_note,
            )

    run_provenance = None
    if not config.only_complex_charging:
        run_provenance = _write_run_provenance(
            output_dir, config, system, topology, positions,
            barostat_protocol=_barostat_protocol,
            charge_protocol=_charge_protocol,
            dispersion_protocol=_dispersion_protocol,
            forcefield_family=_forcefield_family_info,
            membrane_input=_membrane_input_report,
            pairs_conversion=_pairs_conversion,
            membrane_quality_gate_mode=(
                resolve_membrane_quality_gate_mode(config.get("membrane_quality_gate"))
                if _barostat_protocol["system_type"] == "membrane"
                else None
            ),
        )
        log.info(
            "🧾 运行 provenance 已保存: %s",
            os.path.join(output_dir, "run_provenance.json"),
        )

    # ----- 1.5 自动构建溶剂腿缓存 -----
    if config.only_complex_charging:
        log.info("⏭️ charging-only：跳过溶剂腿缓存构建与校验")
    else:
        ligand_resname = _get_residue_name_by_atom_index(
            topology, ligand_indices[0]
        )
        solvent_identity = _ligand_parameter_identity(
            system,
            topology,
            ligand_indices,
            ligand_resname,
            config.top,
            getattr(config, "ligand_xml", None),
            include_dir,
        )
        if config.reset or not solvent_cache_exists(
            output_dir, config.solvent_ionic_strength_molar, solvent_identity
        ):
            log.info(
                "💧 溶剂腿缓存缺失或盐协议不匹配，开始构建 %.3f M NaCl 配体体系...",
                config.solvent_ionic_strength_molar,
            )
            if not build_and_cache_solvent_leg(
                output_dir,
                topology,
                positions,
                ligand_indices,
                ligand_resname,
                ligand_ffxml=getattr(config, "ligand_xml", None),
                top_file=config.top,
                gmx_include_dir=include_dir,
                ionic_strength_molar=config.solvent_ionic_strength_molar,
                cache_identity=solvent_identity,
                charge_treatment=_charge_protocol["charge_treatment"],
            ):
                log.error("❌ 自动构建溶剂腿失败，无法继续一键 ABFE")
                sys.exit(1)
        else:
            log.info("♻️ 检测到已有溶剂腿缓存，跳过重建")

    # ----- 2. 初始化 Pipeline -----
    pipeline = ABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=config.temperature,
        output_dir=output_dir,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        platform_name=config.platform,
        # 膜体系只影响复合物腿（memtodolist §3.2 / docs/TODO.md 的膜受体前置条）。
        # 协议在这里解析，即在创建任何 Context 之前完成 fail-closed 检查（§6.1）。
        environment_type=config.get("system_type"),
        membrane=config.get("membrane"),
        confirm_soluble_with_lipids=bool(
            config.get("confirm_soluble_with_lipids", False)
        ),
        dispersion_protocol=_dispersion_protocol["dispersion_protocol"],
        forcefield_family=_dispersion_protocol["forcefield_family"],
        charge_treatment=_charge_protocol["charge_treatment"],
        # §9 质量门需要的显式输入，从膜输入声明里取（可溶体系为空 dict，不进该分支）。
        membrane_quality_inputs=(
            {
                key: (_membrane_input_report["declared"] or {}).get(key)
                for key in (
                    "pocket_atom_indices",
                    "coion_atom_index",
                    "literature_apl_nm2",
                    # 诊断专用参考值：只落盘偏差，不判门（MEM-12）。与
                    # `literature_apl_nm2` 刻意分成两个字段，见
                    # `abfe_core.evaluate_membrane_quality_gate` 里的说明。
                    "pure_lipid_reference_apl_nm2",
                    "equilibration_length_ns",
                    "ligand_resname",
                )
            }
            | {
                "composition": _system_composition,
                # [MEM-17 已移除 2026-08-03] 这里曾传 `expected_pre_equilibration_frames`
                # 给 §9 质量门做帧数对账。resume 追加的重复帧是真的（实测那条 100 ns
                # 多 1 帧），但对账拦的是主线而不是根因 —— 根因在
                # `abfe_pipeline.pre_equilibrate` 的 `DCDReporter(append=True)`
                # 没有先把 DCD 截断到 checkpoint 帧边界。按用户决定删除，别再加回来。
                # §9 的判据吃的是**总**平衡时长（上游 + 本流程），不是只看本流程。
                # 上游时长不可考时传 None，提取器会退回用轨迹自身跨度——那是我们
                # 唯一能实测到的时长，比塞一个编出来的数诚实。
                "equilibration_length_ns": _total_equil_ns,
            }
            if _membrane_input_report is not None
            else None
        ),
        membrane_quality_gate_mode=config.get("membrane_quality_gate"),
    )

    # ----- 3. 加载可选参数 -----
    # 二面角修正
    torsion_params = (
        _load_json_object_file(config.torsion_params, "torsion 参数")
        if config.torsion_params
        else None
    )

    # DEXP 势能
    # ----- 4. Boresch 参数获取（可能触发预平衡估算） -----
    if config.boresch:
        log.info("🧷 复合物腿 Boresch 已启用 | source=%s", config.boresch_source)
    else:
        log.info("🧷 复合物腿 Boresch 已显式关闭")
    # 🔑 [2026-07-28] 只跑单条腿的增量模式**必须在 rebalance 之前**就换成冻结腿
    # 采样时用的那组 Boresch，否则 rebalance 会在另一个限制势下平衡坐标，
    # 而后续计算用的是冻结那组——起始系综与被求值的 Hamiltonian 对不上。
    #
    # attachment 腿此前漏了这一步（只在 `_run_boresch_attachment_only` 内部换，
    # 那已经是 rebalance 之后），实测留下证据：
    #     rebalance : r0=0.429 nm, θA=83.5°, θB=69.5°   ← boresch_simple.json
    #     attachment: r0=0.431 nm, θA=83.7°, θB=74.68°  ← committed
    # λ=1 那 50k 步平衡不能替代它：本体系的瓶颈已经确认是**慢构象弛豫**，
    # 短平衡消不掉起始系综的记忆。
    #
    # 参考实现没有这个断点：`pipeline.zsh` 里 charging-complex(749) 与
    # restraint(763) 都从同一个 `prerun.run` 出发，而 restr_pull.mdp 与
    # restr_pull_decouple.mdp 由**同一份** restrinfo 生成(678)。
    if config.only_complex_charging or config.only_boresch_attachment:
        _mode = "charging-only" if config.only_complex_charging else "attachment-only"
        boresch_restraint = _load_frozen_stage2_boresch(output_dir)
        log.info(
            "🔒 %s：rebalance 之前就采用冻结 Stage 2 的 Boresch Hamiltonian "
            "(r0=%.6f nm)，不读取/重估 boresch_simple.json",
            _mode,
            float(boresch_restraint["equilibrium_values"]["r0"]),
        )
    else:
        boresch_restraint = resolve_boresch_restraint(config, pipeline)

    # ----- 5. 带限制力再平衡（如果启用） -----
    if boresch_restraint and not config.skip_rebalance:
        log.info("🔧 执行带 Boresch 限制力的再平衡...")
        rebal_data = pipeline._rebalance_with_boresch(
            boresch_params=boresch_restraint,
            n_steps=config.rebalance_steps,
            resume=config.resume and not config.reset,
        )
        pipeline.positions = rebal_data["positions"]
        pipeline.box_vectors = rebal_data["box_vectors"]
        log.info("✓ 再平衡完成，坐标已更新")

    if config.only_boresch_attachment:
        candidate_path = _run_boresch_attachment_only(
            config,
            pipeline,
            boresch_restraint,
        )
        log.info(
            "attachment-only 已结束；stage1/stage2/solvent 未运行，主结果未覆盖。候选文件: %s",
            candidate_path,
        )
        return

    if config.only_complex_charging:
        candidate_path = _run_complex_charging_only(
            config,
            pipeline,
            boresch_restraint,
        )
        log.info(
            "charging-only 已结束；Stage 2 与 solvent 未运行，主结果未覆盖。候选文件: %s",
            candidate_path,
        )
        return

    # ----- 6. 运行复合物腿主流程 -----
    log.info("🔄 启动复合物腿主采样流程 (%s)", config.decoupling)
    complex_results = pipeline.run_full_pipeline(
        decoupling_scheme=config.decoupling,
        potential_type=config.potential,
        dexp_params=dexp_params,
        boresch_params=boresch_restraint,
        torsion_params=torsion_params,
        resume=config.resume and not config.reset,
        run_equilibration=not equilibrium_is_done(
            output_dir,
            expected_fingerprint=_pre_equilibration_fingerprint(
                pipeline.system, pipeline.ligand_indices, pipeline.temperature, pipeline.pressure,
                positions=pipeline.positions,
                box_vectors=pipeline.box_vectors,
                requested_steps=config.get("n_equil_steps", 5_000_000),
                barostat_protocol=pipeline.barostat_protocol,
            ),
        ) or config.reset,
        # 注意：这个 system_type 是**腿身份**（complex/solvent），与膜协议的
        # 环境类型（soluble/membrane）是两个不同的轴，后者走
        # ABFEPipeline(environment_type=...)。
        system_type="complex",
        n_steps_per_window=config.n_steps_per_window,
        steps_per_update=config.steps_per_update,
        n_states_per_stage=config.get("stage1_n_states", 16),
        stage1_n_states=config.get("stage1_n_states", 16),
        stage2_n_states=config.get("stage2_n_states", config.get("stage1_n_states", 16)),
        enable_early_stop=config.enable_early_stop,
        enable_gradual_warmup=config.enable_gradual_warmup,
        warmup_steps=config.warmup_steps,
        n_workers=config.n_workers,
        parallel_stages=config.parallel_stages,
        decharge_method=config.get("decharge_method", "pme"),
        allow_disk_boresch_autoload=True,
        enable_lambda_refine=config.get("enable_lambda_refine", False),
        refine_n_steps_per_window=config.get("refine_n_steps_per_window", 30000),
        refine_steps_per_update=config.get("refine_steps_per_update", config.steps_per_update),
        refine_max_window_span_kJ=config.get("refine_max_window_span_kJ", 35.0),
        pilot_finite_difference_delta=config.get("pilot_finite_difference_delta", 0.01),
        pilot_n_steps_per_state=config.get("pilot_n_steps_per_state", 10000),
        ibs_lse_log_residual_tolerance=config.get(
            "ibs_lse_log_residual_tolerance", 0.5
        ),
        min_bias_updates=config.get("min_bias_updates", 12),
        max_bias_updates=config.get("max_bias_updates", 50),
        required_consecutive_bias_updates=config.get(
            "required_consecutive_bias_updates", 3
        ),
        max_bias_warmup_steps=config.get("max_bias_warmup_steps", 500000),
        # 🔑 [2026-07-27] 此前这个参数只在 --only-complex-charging 那条路径接通，
        # 完整 dual_lambda 链路根本没传，REMDManager 于是用默认上限、在建任何 GPU
        # Context 前预防性回退 CPU——整个 decharging 阶段慢约两个数量级，而且那条
        # 告警只 print 到终端、pipeline.log 里看不见，表现得像卡死。
        charging_max_resident_contexts=config.get("charging_max_resident_contexts"),
        attachment_lambdas=config.get("attachment_lambdas"),
        attachment_n_steps_per_state=config.get("attachment_n_steps_per_state", 250_000),
        attachment_equil_steps_per_state=config.get("attachment_equil_steps_per_state", 50_000),
        attachment_steps_per_sample=config.get("attachment_steps_per_sample", 1_000),
        attachment_seed=config.get("attachment_seed", 20260728),
        attachment_n_seeds=config.get("attachment_n_seeds", 1),
    )
    
    dg_complex = complex_results.get("total_delta_G_complex_kJ_mol", complex_results.get("decoupling_delta_G_kJ_mol", 0.0))
    err_complex = complex_results.get("total_error_kJ_mol", 0.0)
    dg_boresch = complex_results.get("boresch_correction_kJ_mol", 0.0)
    attachment_result = complex_results.get("boresch_attachment", {})

    # ----- 7. 自动加载溶剂相缓存并运行溶剂腿 (ABFE 必选项) -----
    log.info("\n" + "="*70)
    log.info("💧 启动溶剂相 (Ligand-in-Water) 配体腿计算 (自动加载缓存)...")
    log.info("="*70)
    
    try:
        sys_solv, top_solv, pos_solv, box_solv, lig_idx_solv = load_native_system(
            output_dir,
            phase="solvent",
            prefer_equilibrated=not config.reset,
        )
    except FileNotFoundError:
        log.error("❌ 未找到溶剂相缓存，自动构建步骤未成功完成")
        sys.exit(1)
        
    pos_solv, box_solv = center_system_rigidly(pos_solv, box_solv, lig_idx_solv)
    
    solvent_out_dir = os.path.join(output_dir, "solvent_leg")
    # 🔑 溶剂腿有两条**方向相反**的规矩，别搞混：
    #
    #   - **恒压器**：刻意不接 environment_type / membrane。memtodolist §3.2
    #     "溶剂腿继续使用普通 MonteCarloBarostat"——溶剂腿是配体在体相水里，与膜无关，
    #     即使复合物腿跑 membrane，这里也必须保持各向同性恒压器。不要"为了一致"
    #     把 config 的 system_type 传进来，那会给纯水盒装上膜 barostat。
    #   - **色散路线**：必须与复合物腿**完全相同**。§1.3 路线 A 与 §7.5 要求
    #     "复合物腿和溶剂腿使用同一套 ligand–environment 非键定义"。若复合物腿
    #     关掉了炼金均匀密度 LRC 而溶剂腿还留着，两腿的 ligand–environment 口径
    #     就不一致，ΔG_bind = ΔG_solv − ΔG_cplx 的差值里会混进一个协议差。
    pipeline_solv = ABFEPipeline(
        system=sys_solv, topology=top_solv, positions=pos_solv, box_vectors=box_solv,
        ligand_indices=lig_idx_solv, temperature=config.temperature,
        output_dir=solvent_out_dir, checkpoint_dir=os.path.join(solvent_out_dir, "checkpoints"),
        platform_name=config.platform,
        dispersion_protocol=_dispersion_protocol["dispersion_protocol"],
        forcefield_family=_dispersion_protocol["forcefield_family"],
        # ⚠️ 与恒压器方向相反：色散路线与净电荷路线**必须**两腿相同（§1.3 路线 A / §6.1），
        # 只有 environment_type/membrane 是溶剂腿刻意不接的。
        charge_treatment=_charge_protocol["charge_treatment"],
    )
    
    # 运行溶剂腿 (🔑 强制关闭 Boresch)
    solv_results = pipeline_solv.run_full_pipeline(
        decoupling_scheme=config.decoupling,
        potential_type=config.potential,
        dexp_params=dexp_params,
        boresch_params=None,  # 绝对不加 Boresch
        torsion_params=torsion_params,
        resume=config.resume and not config.reset,
        run_equilibration=not equilibrium_is_done(
            solvent_out_dir,
            expected_fingerprint=_pre_equilibration_fingerprint(
                pipeline_solv.system, pipeline_solv.ligand_indices, pipeline_solv.temperature,
                pipeline_solv.pressure,
                positions=pipeline_solv.positions,
                box_vectors=pipeline_solv.box_vectors,
                requested_steps=config.get("n_equil_steps", 5_000_000),
                # 溶剂腿永远是 soluble（§3.2：溶剂腿继续用普通 MonteCarloBarostat），
                # 所以这里的 payload 恒为 None，指纹与改动前一致。
                barostat_protocol=pipeline_solv.barostat_protocol,
            ),
        ) or config.reset,
        system_type="solvent",
        n_steps_per_window=config.n_steps_per_window,
        steps_per_update=config.steps_per_update,
        n_states_per_stage=config.get("stage1_n_states", 16),
        stage1_n_states=config.get("stage1_n_states", 16),
        stage2_n_states=config.get("stage2_n_states", config.get("stage1_n_states", 16)),
        enable_early_stop=config.enable_early_stop,
        enable_gradual_warmup=config.enable_gradual_warmup,
        warmup_steps=config.warmup_steps,
        n_workers=config.n_workers,
        parallel_stages=config.parallel_stages,
        decharge_method=config.get("decharge_method", "pme"),
        allow_disk_boresch_autoload=False,
        enable_lambda_refine=config.get("enable_lambda_refine", False),
        refine_n_steps_per_window=config.get("refine_n_steps_per_window", 30000),
        refine_steps_per_update=config.get("refine_steps_per_update", config.steps_per_update),
        refine_max_window_span_kJ=config.get("refine_max_window_span_kJ", 35.0),
        pilot_finite_difference_delta=config.get("pilot_finite_difference_delta", 0.01),
        pilot_n_steps_per_state=config.get("pilot_n_steps_per_state", 10000),
        ibs_lse_log_residual_tolerance=config.get(
            "ibs_lse_log_residual_tolerance", 0.5
        ),
        min_bias_updates=config.get("min_bias_updates", 12),
        max_bias_updates=config.get("max_bias_updates", 50),
        required_consecutive_bias_updates=config.get(
            "required_consecutive_bias_updates", 3
        ),
        max_bias_warmup_steps=config.get("max_bias_warmup_steps", 500000),
        charging_max_resident_contexts=config.get("charging_max_resident_contexts"),
    )
    
    dg_solvent = solv_results.get("total_delta_G_complex_kJ_mol", solv_results.get("decoupling_delta_G_kJ_mol", 0.0))
    err_solvent = solv_results.get("total_error_kJ_mol", 0.0)
    
    # ----- 8. 计算最终结合自由能 ΔG_bind -----
    # 🔑 关键修复：标准双解耦循环推导（态 A=复合物真实结合；态 D=去耦+释放限制力后
    # 的 apo 蛋白+标准态配体；G(D)=G(A)+ΔG_complex）给出的是
    # ΔG_bind = G(A) - G(apo) - G(配体,溶液,coupled) = ΔG_solvent - ΔG_complex，
    # 不是 ΔG_complex - ΔG_solvent。原因很直观：对一个真实结合的配体，口袋里去耦
    # 花的自由能应该比溶液里去耦花的更多（口袋相互作用更强，这正是它愿意结合的
    # 原因），所以 ΔG_complex > ΔG_solvent；正确公式 ΔG_solvent-ΔG_complex 应为负
    # （有利结合），之前 ΔG_complex-ΔG_solvent 会给出正值（看起来"不利结合"），
    # 这与本次用参考方法交叉验证时符号持续相反的现象完全吻合。
    # 🔑 [ATT-09] 循环闭合统一走 abfe_core.combine_binding_free_energy，
    # 不再在这里手写公式。这里 dg_complex 取自 total_delta_G_complex_kJ_mol，
    # Boresch 释放项已经烘焙在里面（abfe_pipeline: total_dg = dg_phys +
    # cons_correction + dg_boresch），所以 already_included=True，不再减第二次。
    apbs_correction = float(config.get("apbs_correction_kJ_mol", 0.0) or 0.0)
    cycle = combine_binding_free_energy(
        dg_complex_kJ_mol=dg_complex,
        dg_solvent_kJ_mol=dg_solvent,
        err_complex_kJ_mol=err_complex,
        err_solvent_kJ_mol=err_solvent,
        dg_boresch_kJ_mol=dg_boresch,
        boresch_already_included_in_complex=True,
        apbs_correction_kJ_mol=apbs_correction,
    )
    delta_g_bind_uncorrected = cycle["delta_G_bind_uncorrected_kJ_mol"]
    delta_g_bind = cycle["delta_G_bind_kJ_mol"]
    total_err_bind = cycle["total_error_kJ_mol"]
    
    log.info("\n" + "="*70)
    log.info("🎯 ABFE 最终结合自由能计算结果:")
    log.info("   复合物腿 (膜/蛋白+水) ΔG_cplx  = %.2f ± %.2f kJ/mol", dg_complex, err_complex)
    log.info("   溶剂腿   (纯水)       ΔG_solv  = %.2f ± %.2f kJ/mol", dg_solvent, err_solvent)
    if attachment_result.get("present"):
        log.info(
            "   Boresch attachment     ΔG_attach = %.2f ± %.2f kJ/mol",
            float(attachment_result.get("delta_G_kJ_mol", 0.0)),
            float(attachment_result.get("error_kJ_mol", 0.0)),
        )
    log.info("   Boresch 解析修正      ΔG_rest  = %.2f kJ/mol", dg_boresch)
    log.info("   APBS 外部长程修正     ΔG_APBS  = %.2f kJ/mol", apbs_correction)
    log.info("   --------------------------------------------------------")
    log.info("   结合自由能 ΔG_bind           = %.2f ± %.2f kJ/mol", delta_g_bind, total_err_bind)
    log.info("                              = %.2f ± %.2f kcal/mol", delta_g_bind/4.184, total_err_bind/4.184)
    log.info("="*70)
    
    # 保存最终结合结果
    final_bind_result = {
        "complex_delta_G_kJ_mol": float(dg_complex),
        "solvent_delta_G_kJ_mol": float(dg_solvent),
        "boresch_correction_kJ_mol": float(dg_boresch),
        "boresch_attachment": attachment_result,
        # ✅ complex_delta_G_kJ_mol 来自 total_delta_G_complex_kJ_mol，已经在
        # abfe_pipeline.py 里把 Boresch 释放修正烘焙进去 (total_dg = dg_phys +
        # cons_correction + dg_boresch)；下面 delta_g_bind_uncorrected 没有再单独
        # 减一次 dg_boresch。这里显式标记，避免下游看到 boresch_correction_kJ_mol
        # 和 complex_delta_G_kJ_mol 并列就误以为还需要再手动扣一次。
        "boresch_correction_already_included_in_complex_delta_G": True,
        "delta_G_bind_uncorrected_kJ_mol": float(delta_g_bind_uncorrected),
        "apbs_correction_kJ_mol": float(apbs_correction),
        "delta_G_bind_kJ_mol": float(delta_g_bind),
        "delta_G_bind_kcal_mol": float(delta_g_bind / 4.184),
        "total_error_kJ_mol": float(total_err_bind),
        # [ATT-09] 循环闭合的完整记账，由唯一实现给出。
        "thermodynamic_cycle_terms": cycle,
        "timestamp": datetime.now().isoformat(),
        "provenance": run_provenance,
        "thermodynamic_cycle": THERMODYNAMIC_CYCLE_DOC,
        "external_corrections": {
            "apbs": {
                "delta_G_kJ_mol": float(apbs_correction),
                "applied_to": "final_binding_free_energy",
                "note": config.get("apbs_correction_note", ""),
                "status": "applied" if abs(apbs_correction) > 0.0 else "not_applied",
                "helper": "apbs_correction.py",
                "protocol_warning": (
                    "APBS correction is treated as an explicit external final term. "
                    "Use only when the generated APBS cycle matches the intended thermodynamic correction."
                ),
            }
        },
        "diagnostics": {
            "complex": complex_results.get("diagnostics", {}),
            "complex_stage_diagnostics": complex_results.get("stage_diagnostics", {}),
            "solvent": solv_results.get("diagnostics", {}),
            "solvent_stage_diagnostics": solv_results.get("stage_diagnostics", {}),
            "boresch": {
                "method": boresch_restraint.get("method") if isinstance(boresch_restraint, dict) else None,
                "diagnostics": boresch_restraint.get("diagnostics", {}) if isinstance(boresch_restraint, dict) else {},
                "force_constants_raw": boresch_restraint.get("force_constants_raw", {}) if isinstance(boresch_restraint, dict) else {},
                "force_constant_clipped": boresch_restraint.get("force_constant_clipped", {}) if isinstance(boresch_restraint, dict) else {},
                "uses_analytical_release_formula": bool(config.boresch and boresch_restraint),
                "analytical_release_assumption": (
                    "Boresch release correction assumes locally harmonic, approximately Gaussian restraint-coordinate fluctuations."
                ),
                "analytical_release_assumption_checked": (
                    isinstance(boresch_restraint, dict)
                    and bool(boresch_restraint.get("diagnostics", {}).get("boresch_harmonicity_check", {}).get("ok"))
                ),
                "analytical_release_reliable": (
                    boresch_restraint.get("diagnostics", {}).get("boresch_harmonicity_check", {}).get("harmonic_assumption_ok")
                    if isinstance(boresch_restraint, dict)
                    else None
                ),
            },
            "independent_repeats": {
                "performed": False,
                "note": "Independent repeat runs are not launched automatically; run the same config with distinct random seeds and compare final_binding_results.json.",
            },
        },
    }
    bind_out_path = os.path.join(output_dir, "final_binding_results.json")
    with open(bind_out_path, "w") as f:
        json.dump(final_bind_result, f, indent=2, cls=NumpyEncoder)
    log.info("💾 最终结合自由能结果已保存: %s", bind_out_path)

    # ----- 7. 输出最终结果 -----
    log.info("✅ ABFE 计算完成")
    log.info("结果已生成，见 %s", bind_out_path)


if __name__ == "__main__":
    main()
