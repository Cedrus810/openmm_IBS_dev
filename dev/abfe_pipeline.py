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
import glob
import json
import shutil
import multiprocessing as mp
import time
import logging
import builtins
import hashlib
import platform
import sys
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 项目内部模块依赖
from abfe_preoptimizer import ABFEPreOptimizer, DualLambdaPreOptimizer
from abfe_preoptimizer import generate_overlapping_windows
from abfe_preoptimizer import build_aces_probe_system, build_aces_probe_system_dual_lambda
from abfe_preoptimizer import generate_overlapping_windows   # ✅ 保留这个
from abfe_preoptimizer import refine_stage_lambda_path_from_data
from abfe_preoptimizer import (
    refine_stage_lambda_path_by_overlap,
    split_window_from_warmup_failure,
    insert_lambda_from_overlap_failure,
    plan_vdw_overlap_repair_targets,
    canonicalize_window_ranges,
    split_window_from_ibs_lse_failure,
    insert_thermodynamic_midpoint_from_ibs_lse_failure,
    redistribute_vanishing_lambda_subdomains,
    vanishing_subdomain_ranges_from_lambdas,
    human_vanishing_initial_lambdas,
    quadratic_vanishing_base_lambdas,
    validate_vanishing_lambda_path_invariants,
    blended_metric_vanishing_lambdas,
    VANISHING_FINAL_STATE_COUNT,
    validate_single_shared_boundary_ranges,
    VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
    THERMODYNAMIC_PATH_PROTOCOL_VERSION,
)
from ibs_engine import (
    IBSWindowManagerDualLambda,
    IBSWindowManagerShadowCoul,
    GlobalMBARAnalyzer,
    solve_stage_integrated,
    REMDManager,
    TraditionalMBARAnalyzer,
    generate_overlapping_windows,
    lambda_endpoint_diagnostics,
    run_shadow_bridge_leg,
    WCA_ACCOUNTING_VERSION,
    IBS_BIAS_PROTOCOL_VERSION,
    TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
    FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS,
    IBSWarmupConvergenceError,
    IBSFrozenCalibrationValidationError,
    probe_bidirectional_overlap,
    probe_bidirectional_overlap_for_bias_calibration,
    _resolve_periodic_box_vectors,
    _build_platform_properties,
    _system_has_global_parameter,
    _atomic_write_json,
    _invalidate_production_window_checkpoint,
    _load_validated_window_data_triplet,
    _assert_expected_windows_all_loaded,
    IBSIncompleteStageCoverageError,
    ibs_lj_tail_lrc_is_applicable,
    ibs_lj_tail_lrc_inapplicable_reason,
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
    combine_binding_free_energy,  # [ATT-09] 热力学循环闭合的唯一实现
)
import warnings

PME_DECHARGE_MODEL_VERSION = "pme_decharge_v2_llfreeze_pmeself_20260523"

logger = logging.getLogger(__name__)


def _stage_lambda_endpoint_diagnostics(
    stage_name: str,
    lambdas_coul,
    lambdas_vdw,
) -> Dict:
    """Apply the endpoint contract for one half of the dual-lambda path."""
    expected = {
        "decharging": ((1.0, 1.0), (0.0, 1.0)),
        "vanishing": ((0.0, 1.0), (0.0, 0.0)),
    }
    if stage_name not in expected:
        raise ValueError(f"未知 dual-lambda stage: {stage_name!r}")
    expected_start, expected_end = expected[stage_name]
    return lambda_endpoint_diagnostics(
        lambdas_coul,
        lambdas_vdw,
        expected_start=expected_start,
        expected_end=expected_end,
    )


def _json_safe(obj):
    """🔑 [P1-15] 递归转成 `json.dump` 直接可写的原生类型。

    `_atomic_write_json`（ibs_engine）用的是不带 `cls=NumpyEncoder` 的
    `json.dump`，所以任何混进 payload 的 numpy 标量/数组都会 `TypeError`
    并让整个 checkpoint 写失败。`solve_stage_integrated` 返回的
    `window_overlap_diagnostics` / `f_k` / `coverage_diagnostics` 恰恰都含 numpy，
    把它们纳入落盘范围之前必须先过这一道。

    非有限值（NaN/±Inf）转成 `None`：`json.dump` 默认会写出 `NaN`/`Infinity`
    这种非标准 JSON，别的工具读不了。
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    return obj


# 🔑 [BORESCH-COMMIT] `boresch_equilibrium_committed.json` 的 schema 版本。
# v1 = 只有 {"equilibrium_values": {...}} 的裸格式（无任何身份信息）。
BORESCH_COMMITTED_SCHEMA_VERSION = 2

# 复用已提交平衡值时，允许它与当前坐标实测几何相差多少个 σ。
# σ_i = sqrt(kT/k_i)：六个自由度的限制势分别是 0.5*kr*(d-r0)^2 与
# k*(1-cos(Δ))，小偏离下方差都是 kT/k，所以这是限制势自身的热涨落宽度。
# 与单帧比较本身就带 ~1σ 噪声，取 4σ 既能放过正常涨落，也能抓住
# "平衡值根本不描述当前构象"这类错误（实测那次是 4.0-23.6σ）。
BORESCH_COMMITTED_MAX_DEVIATION_SIGMA = 4.0

# 告警带。硬门为什么不能再压低：committed 值来自过去某一帧、current 来自当前帧，
# 两个独立单帧之差的宽度是 √2·σ，所以 4σ ≈ 2.8 个"差值 σ"，每个自由度误报率
# ~0.5%、六个约 2.8%。压到 3σ 会让单次运行误报率升到约 19%，resume 直接没法用。
# 代价是纯 thetaA/thetaB 对调（实测 3.70 σ）刚好落在硬门下面——但真实的标签错位
# 必然同时打乱二面角（它们共用同一批原子），实测那次 phiA0 就到了 23.65 σ。
# 这个区间的东西不阻断，但必须大声打出来。
BORESCH_COMMITTED_WARN_DEVIATION_SIGMA = 2.5

_BORESCH_EQ_TO_FORCE_CONSTANT = (
    ("r0", "kr", False),
    ("thetaA0", "kthetaA", False),
    ("thetaB0", "kthetaB", False),
    ("phiA0", "kphiA", True),
    ("phiB0", "kphiB", True),
    ("phiC0", "kphiC", True),
)


def _wrap_to_pi(value: float) -> float:
    """把角度差折回 **[-π, π)**，二面角比较必须先做这一步。

    注意区间是左闭右开：`_wrap_to_pi(math.pi)` 返回 `-π` 而不是 `+π`。
    在 ±π 这个对跖点上符号本来就是任意的（同一个角距离的两种写法），
    而下游 `boresch_committed_deviation_sigma` 只取 `abs(delta)` 来判门，
    所以取哪一侧不影响任何判定。
    """
    return float((float(value) + math.pi) % (2.0 * math.pi) - math.pi)


def boresch_committed_deviation_sigma(
    committed_eq: Dict,
    current_eq: Dict,
    force_constants: Dict,
    temperature_k: float,
) -> Dict[str, Dict[str, float]]:
    """逐自由度算"已提交平衡值"与"当前坐标实测几何"相差几个 σ。

    🔑 为什么需要这个检查：`run_full_pipeline` 有一条刻意的保护——平衡几何量只在
    一条腿第一次采样时推导一次并落盘，之后任何 resume 都原样复用，绝不重算
    （避免同一条腿的前后窗口用两套哈密顿量拼接自由能曲线）。动机是对的，但它
    **没有任何一致性校验**：只要文件存在就复用。

    实测后果（2026-07-27 定位）：`boresch_equilibrium_committed.json` 写于
    2026-07-10 18:51，而体系在 2026-07-26 被整体重新平衡过。那份 17 天前的
    平衡值里 thetaA0/thetaB0 是对调的、三个二面角完全错乱：

        thetaA0  committed 2.0361  vs  轨迹实测 1.5634   (4.2 σ)
        thetaB0  committed 1.5338  vs  轨迹实测 1.9770   (4.0 σ)
        phiA0    committed -2.1285 vs  轨迹实测 1.5130   (23.6 σ)

    限制力因此把配体从自己的 pose 上拽走 3.4 Å（无约束预平衡只漂 0.60 Å），
    方向性氢键丢失 → 复合物腿去电荷偏低约 25 kJ/mol、解析释放修正也是错的。
    唯一没受影响的是 vdW——它对取向不敏感，这也正是当时唯一对得上参考值的那一项。

    注意：`update_boresch_from_last_frame` 已有的两道门（角度 40-140°、r0 漂移
    <2.5 Å）对这组错值**全部放行**（2.0361 rad = 116.7° 在安全域内，r0 也没漂），
    所以必须用"与当前几何的偏离"这个正交判据。
    """
    kt = 8.31446261815324e-3 * float(temperature_k)
    report: Dict[str, Dict[str, float]] = {}
    for eq_key, k_key, is_dihedral in _BORESCH_EQ_TO_FORCE_CONSTANT:
        if eq_key not in committed_eq or eq_key not in current_eq:
            continue
        k = float(force_constants.get(k_key, 0.0) or 0.0)
        if k <= 0.0 or not math.isfinite(k):
            continue
        sigma = math.sqrt(kt / k)
        delta = float(current_eq[eq_key]) - float(committed_eq[eq_key])
        if is_dihedral:
            delta = _wrap_to_pi(delta)
        report[eq_key] = {
            "committed": float(committed_eq[eq_key]),
            "current": float(current_eq[eq_key]),
            "delta": delta,
            "sigma": sigma,
            "deviation_sigma": abs(delta) / sigma if sigma > 0 else float("inf"),
        }
    return report


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


def _pre_equilibration_fingerprint(
    system: Optional[openmm.System],
    ligand_indices: Optional[List[int]],
    temperature,
    pressure=None,
    positions=None,
    box_vectors=None,
    requested_steps: Optional[int] = None,
) -> str:
    """Content fingerprint for one pre-equilibration run: system Hamiltonian +
    which atoms are the ligand + temperature + barostat pressure.

    ``equilibrium_is_done()`` (runabfe.py) previously only checked that
    ``pre_equilibration.dcd``/``pre_equil.chk`` exist and are non-trivially
    sized -- it had no way to tell whether those files actually correspond to
    the *current* system/config or are stale leftovers from a differently
    configured run reusing the same --output directory without --reset. This
    fingerprint is written alongside the trajectory by ``pre_equilibrate()``
    and can be compared against a freshly recomputed one before trusting the
    cache.

    The initial coordinates, periodic box and requested step budget are part
    of the identity. A different docking pose, box, or a short smoke-test
    equilibration must never satisfy a later production request.

    ``pressure`` is optional only for call-site convenience (old callers that
    haven't been updated pass ``None``, which is distinguishable from any real
    pressure value in the hash) -- omitting it silently would mean a changed
    barostat pressure (different target density/ensemble) could reuse a
    stale equilibration undetected.
    """
    temp_k = (
        temperature.value_in_unit(unit.kelvin)
        if hasattr(temperature, "value_in_unit")
        else float(temperature)
    )
    pressure_bar = (
        pressure.value_in_unit(unit.bar)
        if hasattr(pressure, "value_in_unit")
        else (float(pressure) if pressure is not None else None)
    )
    payload = {
        "system_xml_hash": _system_xml_hash(system),
        "ligand_indices": sorted(int(i) for i in (ligand_indices or [])),
        "temperature_K": round(float(temp_k), 6),
        "pressure_bar": round(float(pressure_bar), 6) if pressure_bar is not None else None,
        "positions_sha256": _positions_hash(positions),
        "box_vectors_sha256": _box_vectors_hash(box_vectors),
        "requested_steps": int(requested_steps) if requested_steps is not None else None,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True))


def _rebalance_fingerprint(system: Optional[openmm.System], boresch_params: Optional[Dict]) -> str:
    """Content fingerprint for one Boresch-restrained rebalance run: system
    Hamiltonian + the actual Boresch anchors/equilibrium values/force
    constants used to build the restraint.

    ``rebalance_state.json`` previously only recorded ``status``/``n_steps``
    -- a completed rebalance from a run with different Boresch anchors or
    force constants (different receptor/ligand restraint atoms, a
    re-estimated r0, a different force constant clip) would still read as
    "completed" and get its stale ``rebalance.chk``/``rebalance_traj.dcd``
    loaded, silently mismatched against the *current* restraint. The
    checkpoint itself is tied to a specific System (including the injected
    ``LambdaDependentBoreschForce`` parameters), so loading it under a
    different Boresch configuration doesn't error -- it just resumes from a
    state that was never equilibrated under the restraint now in effect.
    """
    if not _has_valid_boresch_restraint(boresch_params):
        boresch_payload = None
    else:
        boresch_payload = {
            "receptor_indices": [int(i) for i in boresch_params["receptor_indices"]],
            "ligand_indices": [int(i) for i in boresch_params["ligand_indices"]],
            "equilibrium_values": {
                k: round(float(v), 8) for k, v in sorted(boresch_params["equilibrium_values"].items())
            },
            "force_constants": {
                k: round(float(v), 8) for k, v in sorted(boresch_params["force_constants"].items())
            },
        }
    payload = {
        "system_xml_hash": _system_xml_hash(system),
        "boresch": boresch_payload,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, default=str))


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


def _box_vectors_nm_array(box_vectors) -> Optional[np.ndarray]:
    if box_vectors is None:
        return None
    try:
        rows = []
        for vec in box_vectors:
            values = (
                vec.value_in_unit(unit.nanometer)
                if hasattr(vec, "value_in_unit")
                else vec
            )
            rows.append([float(x) for x in values])
        box = np.asarray(rows, dtype=np.float64)
    except Exception as exc:
        raise ValueError(f"无法解析周期盒向量: {exc}") from exc
    if box.shape != (3, 3) or not np.all(np.isfinite(box)):
        raise ValueError(f"周期盒向量必须是有限 (3, 3) 数组，实际为 {box.shape}")
    if abs(float(np.linalg.det(box))) <= 1.0e-12:
        raise ValueError("周期盒向量奇异，无法计算 minimum-image 位移")
    return box


def _box_vectors_hash(box_vectors) -> Optional[str]:
    box = _box_vectors_nm_array(box_vectors)
    if box is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(box).tobytes()).hexdigest()


def _minimum_image_displacement(displacement, box_vectors) -> np.ndarray:
    """Return the triclinic minimum-image displacement for row-vector boxes."""
    delta = np.asarray(displacement, dtype=np.float64)
    box = _box_vectors_nm_array(box_vectors)
    if box is None:
        raise ValueError("周期体系缺少 box vectors，无法计算 Boresch minimum-image 距离")
    fractional = delta @ np.linalg.inv(box)
    fractional -= np.round(fractional)
    return fractional @ box


_CODE_HASH_CACHE: Optional[str] = None
_DEBUG_CODE_HASH_WARNED = False


def _debug_code_hash_frozen() -> Optional[str]:
    """开发/调试专用逃生舱：设置环境变量 ABFE_DEBUG_FREEZE_CODE_HASH=1 时，
    `_code_hash()`/`_preopt_code_hash()` 都返回同一个固定常量，而不是真的
    读盘算哈希——这样反复修改 abfe_pipeline.py/ibs_engine.py 等文件本身
    （调试修复循环/收敛逻辑时几乎每轮都要改代码）不会连带让 stage 完成
    缓存/fixed-H 探针缓存/IBS 状态/λ 路径预优化缓存全部失效重算。

    🔑 这不是"删掉哈希校验"——system_xml/topology/坐标/potential_type/
    Boresch 参数以及各个手动维护的 protocol version 常量（IBS_BIAS_
    PROTOCOL_VERSION 等）仍然正常参与比较，真正改变了物理/协议含义的
    改动照样会让缓存失效。只有"我又手改了一遍 Python 代码本身"这一件事
    被冻结掉，代价是：这段时间里，如果某次代码改动恰好改了会影响已采样
    结果正确性的逻辑（不只是修 bug/重构控制流），旧缓存不会自动感知到。
    这是有意识的、显式 opt-in 的临时状态，只应该在同一次调试会话内密集
    改代码、跑收敛时打开；确认收敛后做最终会计入结果的正式运行前，必须
    取消设置这个环境变量，让哈希校验真正生效一次，排除"这段时间内某次
    改动其实动了物理逻辑却被跳过检查"的可能。
    """
    global _DEBUG_CODE_HASH_WARNED
    if os.environ.get("ABFE_DEBUG_FREEZE_CODE_HASH") != "1":
        return None
    if not _DEBUG_CODE_HASH_WARNED:
        print(
            "  🚨 [DEBUG] ABFE_DEBUG_FREEZE_CODE_HASH=1：code_sha256/preopt_code_sha256 "
            "已冻结为固定常量，不反映当前磁盘上的代码改动——仅供调试收敛逻辑时使用，"
            "正式出结果前必须取消设置这个环境变量并至少完整跑一次真正的哈希校验。"
        )
        _DEBUG_CODE_HASH_WARNED = True
    return "DEBUG_CODE_HASH_FROZEN_ABFE_DEBUG_FREEZE_CODE_HASH"


def _code_hash() -> str:
    """进程级代码指纹，只在本进程第一次调用时读盘计算一次并缓存。

    🔑 [live-edit 指纹漂移 bug] 之前每次调用都重新读盘哈希这几个源文件——
    但一个长时间运行的进程，其真正在执行的字节码在 import 时就已经固定，
    源文件之后被编辑（这个项目里开发时经常发生，同一个会话里就多次边跑
    边改）不会让运行中的进程重新加载。之前的写法会让同一个进程在不同时刻
    算出不同的 code_sha256——纯粹因为磁盘上的文件变了，跟这个进程实际在跑
    的代码毫无关系，会污染 stage/probe 缓存的失效判断（可能把同一进程内
    本该复用的旧结果误判为"协议不匹配"而白白重算，或者反过来掩盖两次不同
    进程之间代码其实不同）。改成只在本进程第一次用到时读盘一次、后续调用
    直接返回缓存值，让这个指纹真正反映"这个进程从 import 起就固定的代码"，
    而不是"调用这一刻磁盘上恰好是什么"。
    """
    debug_frozen = _debug_code_hash_frozen()
    if debug_frozen is not None:
        return debug_frozen
    global _CODE_HASH_CACHE
    if _CODE_HASH_CACHE is not None:
        return _CODE_HASH_CACHE
    base_dir = os.path.dirname(os.path.abspath(__file__))
    payload = {}
    for name in ("abfe_pipeline.py", "abfe_core.py", "ibs_engine.py", "abfe_preoptimizer.py"):
        path = os.path.join(base_dir, name)
        try:
            with open(path, "rb") as handle:
                payload[name] = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            payload[name] = None
    _CODE_HASH_CACHE = _sha256_text(json.dumps(payload, sort_keys=True))
    return _CODE_HASH_CACHE


_PREOPT_CODE_HASH_CACHE: Optional[str] = None


def _preopt_code_hash() -> str:
    """λ 路径预优化（thermodynamic-length 逐点扫描）专用的、范围更窄的代码指纹。

    🔑 [预优化被无关 bug 修复连带失效] `_code_hash()` 把 abfe_pipeline.py/
    abfe_core.py/ibs_engine.py/abfe_preoptimizer.py 四个文件哈希成一个整体，
    任何一处改动（哪怕只是修 ibs_engine.py 里窗口修复循环/reseed_resample
    续采这类跟预优化完全无关的 bug）都会让它变化，进而让 _stage_protocol_key
    里嵌的 code_sha256 跟着变，连带把 Stage 1/Stage 2 那份跑一次要几个小时的
    λ 路径预优化缓存也判定失效、逼着重新跑一遍——预优化（
    optimize_stage1_decharging/optimize_stage2_vanishing，见 abfe_preoptimizer.py）
    只是在探针 Context 上测 dU/dλ 的方差来定 λ 路径，完全不涉及 IBS bias/f_k/
    窗口修复循环，本不该被这些代码的改动连带作废。这里只哈希预优化真正会
    执行到的代码：abfe_preoptimizer.py 本身，以及它复用的力场/软核势构建代码
    abfe_core.py（AlchemicalPotentialFactory/ensure_owned_system/
    create_ligand_internal_force/sync_all_exclusions，见 abfe_preoptimizer.py
    顶部 import）——不包含 abfe_pipeline.py/ibs_engine.py，所以修复窗口管理/
    修复循环/production checkpoint 续采这类 bug 不会再连带让预优化缓存失效。
    如果将来预优化本身用到的代码（这两个文件）真的改了，这份指纹会正确
    变化，缓存依然会失效重算——收窄范围不等于放弃校验。
    """
    debug_frozen = _debug_code_hash_frozen()
    if debug_frozen is not None:
        return debug_frozen
    global _PREOPT_CODE_HASH_CACHE
    if _PREOPT_CODE_HASH_CACHE is not None:
        return _PREOPT_CODE_HASH_CACHE
    base_dir = os.path.dirname(os.path.abspath(__file__))
    payload = {}
    for name in ("abfe_preoptimizer.py", "abfe_core.py"):
        path = os.path.join(base_dir, name)
        try:
            with open(path, "rb") as handle:
                payload[name] = hashlib.sha256(handle.read()).hexdigest()
        except OSError:
            payload[name] = None
    _PREOPT_CODE_HASH_CACHE = _sha256_text(json.dumps(payload, sort_keys=True))
    return _PREOPT_CODE_HASH_CACHE


PROTOCOL_FINGERPRINT_SCHEMA_VERSION = 1


def _file_sha256(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return None


def _canonical_protocol_value(value):
    """Convert protocol inputs to deterministic, JSON-safe values."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"协议指纹不接受 NaN/Inf: {value!r}")
        return float(value)
    if isinstance(value, np.ndarray):
        return _canonical_protocol_value(value.tolist())
    if isinstance(value, dict):
        return {
            str(key): _canonical_protocol_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_protocol_value(item) for item in value]
    if isinstance(value, set):
        canonical = [_canonical_protocol_value(item) for item in value]
        return sorted(
            canonical,
            key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False),
        )
    if hasattr(value, "unit") and hasattr(value, "value_in_unit"):
        raw = value.value_in_unit(value.unit)
        return {
            "value": _canonical_protocol_value(raw),
            "unit": str(value.unit),
        }
    raise TypeError(
        f"协议指纹遇到不支持的值类型 {type(value).__name__}: {value!r}"
    )


def _protocol_fingerprint(payload: Dict) -> Dict:
    canonical = _canonical_protocol_value(payload)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return {
        "schema_version": PROTOCOL_FINGERPRINT_SCHEMA_VERSION,
        "sha256": _sha256_text(encoded),
        # Payload is deliberately retained: a mismatch is auditable instead of
        # being an opaque hash-only cache miss.
        "payload": canonical,
    }


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
    window_ranges: Optional[List[Tuple[int, int]]],
    enable_early_stop: bool,
    boresch_params: Optional[Dict],
    enable_gradual_warmup: bool,
    warmup_steps: int,
    min_bias_updates: int,
    max_bias_updates: int,
    required_consecutive_bias_updates: int,
    max_bias_warmup_steps: int,
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
        window_ranges=window_ranges,
        enable_early_stop=enable_early_stop,
        boresch_params=boresch_params,
        enable_gradual_warmup=enable_gradual_warmup,
        warmup_steps=warmup_steps,
        min_bias_updates=min_bias_updates,
        max_bias_updates=max_bias_updates,
        required_consecutive_bias_updates=required_consecutive_bias_updates,
        max_bias_warmup_steps=max_bias_warmup_steps,
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

        # 🔑 双重预平衡防护：这两个标记只反映"本进程这个 pipeline 实例这次调用
        # 里是否已经跑过"，不是磁盘缓存状态（那由 equilibrium_is_done() 单独判断）。
        # pre_equilibrate() 在跑完（无论是否带 Boresch 限制力）后置位第一个；
        # _rebalance_with_boresch() 在其每一个 return 分支（含"已完成、跳过"分支）
        # 都会置位第二个，因为两种情况下 self.positions 都已经是可信的、带 Boresch
        # 限制力平衡过的坐标。run_full_pipeline() 的内部预平衡块用它们短路，
        # 避免用一次无约束的预平衡覆盖掉刚做完的 Boresch 再平衡坐标——这个覆盖
        # 之前在 --reset 或外部 Boresch 参数的全新运行里都会真实发生。
        self._pre_equilibration_done_this_process = False
        self._boresch_rebalance_done_this_process = False

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
    def repair_pbc_molecule_integrity(self, *, context: str = "") -> bool:
        """🔑 [P1-14] 按拓扑把每个连通分子做整分子周期平移，再整体居中。

        **必须在第一次创建 Context / 最小化 / 预平衡之前调用。** 此前这段逻辑只在
        `run_full_pipeline` 第 2 节出现，也就是 `pre_equilibrate()`（内含
        `LocalEnergyMinimizer.minimize` 与 NPT 步进）**之后**；而
        `runabfe.center_system_rigidly()` 只做整体质心平移，却有调用点紧接着打印
        "分子完整性修复完毕"。结果是：GRO/缓存里本来就跨边界断裂的分子，会带着
        断裂先进最小化和 NPT——最小化会真实改变原子间相对坐标，把断裂"焊"进构型。

        只做两件事，都不改变任何分子内相对坐标：
          1. `image_molecules()` —— 按连通分子整体做周期平移；
          2. `center_coordinates()` —— 全体系整体平移。
        不旋转、不缩放。

        失败时 **fail closed**。此前的回退是 `_wrap_ligand_to_box()`（自己的
        docstring 就写着"仅做整体刚性平移"），它根本修不了跨盒断裂，却让流程
        带着未修复的构型继续——等于把"修复失败"降级成"看起来修好了"。
        """
        if self.positions is None or self.box_vectors is None:
            self._log(f"  ⏭️ 无坐标/盒矢量，跳过 PBC 分子完整性修复{('（' + context + '）') if context else ''}")
            return False

        self._log(f"  📦 正在执行 PBC 分子完整性修复与配体居中{('（' + context + '）') if context else ''}...")
        try:
            import mdtraj as md
            md_top = md.Topology.from_openmm(self.topology)
            # ⚠️ Quantity.value_in_unit() 在底层是 list-of-Vec3（而非 numpy 数组）时
            # 返回的仍是 Python list，没有 .reshape；必须显式再包一层 np.asarray。
            # 🚨 mdtraj 的 Cython 扩展（含 image_molecules 内部用到的 geometry 例程）
            # 要求 float32（"float"）缓冲区；用 float64 构造时 Trajectory() 会自动
            # 转换 xyz，但直接赋值 unitcell_vectors 不会，于是 image_molecules() 必然抛
            # "Buffer dtype mismatch, expected 'float' but got 'double'"。历史上这个
            # dtype bug 让每次都静默回退到只居中配体的 numpy 兜底，撕裂的水分子
            # 从预平衡开始就一路带进所有窗口。
            pos_nm = np.asarray(
                self.positions.value_in_unit(unit.nanometer)
                if hasattr(self.positions, "value_in_unit") else self.positions,
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
            self.positions = [
                openmm.Vec3(float(x), float(y), float(z)) for x, y, z in traj.xyz[0]
            ] * unit.nanometer
            self._log("  ✓ PBC 分子完整性已修复，体系已居中至主周期")
            self._pbc_integrity_repaired = True
            return True
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"PBC 分子完整性修复失败：{exc}\n"
                "  拒绝以未修复的构型继续（fail closed）。此前这里会回退到 "
                "_wrap_ligand_to_box()，但那只是整体质心平移，**修不了跨盒断裂的分子**，\n"
                "  只会让断裂构型静默进入最小化/NPT，并被最小化固化进相对坐标。\n"
                "  常见原因：拓扑与坐标原子数不一致、盒矢量非法、mdtraj 未安装。"
            ) from exc

    def pre_equilibrate(
        self,
        n_steps: int = 5_000_000,  # 10ns @ 2fs
        save_traj: bool = True,
        platform_name: str = None,  # ✅ 默认使用实例配置的 platform
        resume: bool = False,
    ) -> Dict:
        """物理预平衡 - 【修复】默认使用 GPU，仅在生产采样前清理上下文"""
        # 🔑 [P1-14] 在建 Context / 最小化 / NPT **之前**做整分子 PBC 修复。
        # 这段此前只在 run_full_pipeline 第 2 节出现，也就是本函数**之后**——
        # 于是输入里本来就跨盒断裂的分子会先进最小化，被固化进相对坐标。
        # 幂等：已经修过就跳过（run_full_pipeline 里那次仍会执行，用于修预平衡
        # 步进过程中新产生的跨盒，两者不冲突）。
        if not getattr(self, "_pbc_integrity_repaired", False):
            self.repair_pbc_molecule_integrity(context="pre_equilibrate 之前")

        traj_file = os.path.join(self.output_dir, "pre_equilibration.dcd")
        chk_file = os.path.join(self.checkpoint_dir, "pre_equil.chk")
        fp_file = os.path.join(self.output_dir, "pre_equilibration_fingerprint.json")
        requested_fingerprint = _pre_equilibration_fingerprint(
            self.system,
            self.ligand_indices,
            self.temperature,
            self.pressure,
            positions=self.positions,
            box_vectors=self.box_vectors,
            requested_steps=n_steps,
        )

        # A binary OpenMM checkpoint is only meaningful for the exact initial
        # pose/box/Hamiltonian and requested budget that created it.
        if resume and os.path.exists(chk_file):
            try:
                with open(fp_file, encoding="utf-8") as handle:
                    recorded_fingerprint = json.load(handle).get("fingerprint")
            except Exception:
                recorded_fingerprint = None
            if recorded_fingerprint != requested_fingerprint:
                self._log(
                    "  ⚠️ 预平衡 checkpoint 的坐标/盒子/System/步数指纹不匹配，"
                    "拒绝恢复并从当前输入重新开始"
                )
                resume = False
        if save_traj:
            # Persist the identity before the first step so an interrupted run
            # has a checkpoint identity available on its very next resume.
            with open(fp_file, "w", encoding="utf-8") as handle:
                json.dump(
                    {"fingerprint": requested_fingerprint, "n_steps": int(n_steps)},
                    handle,
                    indent=2,
                )
        
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
        if save_traj:
            try:
                with open(fp_file, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "fingerprint": requested_fingerprint,
                            "n_steps": int(n_steps),
                        },
                        f,
                        indent=2,
                    )
            except Exception as e:
                self._log(f"  ⚠️ 预平衡指纹写入失败（不影响本次结果，但下次 resume 无法校验缓存是否匹配当前系统）: {e}")
        # 🔑 标记本进程已经做过一次预平衡——run_full_pipeline() 内部的预平衡块
        # 会用这个标记短路，不再重新跑一次并覆盖这里刚更新的 self.positions。
        self._pre_equilibration_done_this_process = True

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
        # ✅ 初始距离检查，防止拉力过载（必须在计算/比对 fingerprint 之前做，
        # 否则下面 _rebalance_fingerprint() 用的是校正前的 r0，而实际注入
        # LambdaDependentBoreschForce、写进 rebalance.chk 的却是这里校正后的
        # r0——fingerprint 描述的与 checkpoint 里真正生效的限制力对不上，缓存
        # 校验形同虚设）。
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
                pos_nm = np.asarray(
                    self.positions.value_in_unit(unit.nanometer)
                    if hasattr(self.positions, "value_in_unit")
                    else self.positions,
                    dtype=np.float64,
                )
            # 计算实际距离 (H0-L0: 最近受体锚点 - 配体首锚点)
            H0 = pos_nm[rec_idx[0]]
            L0 = pos_nm[lig_idx[0]]
            h0_l0 = _minimum_image_displacement(H0 - L0, self.box_vectors)
            actual_dist = float(np.linalg.norm(h0_l0))
            target_r0 = eq.get("r0", 1.0)  # nm

            if abs(actual_dist - target_r0) > 0.15:
                self._log(f"  🔧 动态校正 Boresch r0: {target_r0*10:.2f}Å → {actual_dist*10:.2f}Å (防爬坡撕裂)")
                boresch_params["equilibrium_values"]["r0"] = float(actual_dist)

        # ✅ fingerprint 必须在上面的动态 r0 校正之后计算：它要描述的是"最终
        # 实际会被注入 LambdaDependentBoreschForce、写进 rebalance.chk 的那组
        # Boresch 参数"，而不是校正前的原始值。
        expected_rebalance_fingerprint = _rebalance_fingerprint(self.system, boresch_params)

        # 🔑 rebalance_state.json 之前只记录 status/n_steps，不记录 system/
        # Boresch 锚点/力常数指纹。rebalance.chk 是绑定到具体 System（含注入的
        # LambdaDependentBoreschForce 参数）的二进制状态；如果 Boresch 锚点、
        # 平衡值或力常数变了（重新估算过、或上面的动态 r0 校正给出了
        # 不同值），加载旧 checkpoint 不会报错，只会从一个"从未在当前限制力下
        # 平衡过"的状态继续——必须显式比对指纹，不一致就整体视为缓存失效，
        # 不能只看 status/n_steps 是否对得上。
        rebalance_cache_trusted = True
        rebalance_cache_reject_reason = None
        if resume and os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    _cached_rebalance_state_for_fp = json.load(f)
                if _cached_rebalance_state_for_fp.get("fingerprint") != expected_rebalance_fingerprint:
                    rebalance_cache_trusted = False
                    rebalance_cache_reject_reason = "system/Boresch 锚点或力常数指纹不匹配"
            except Exception:
                rebalance_cache_trusted = False
                rebalance_cache_reject_reason = "rebalance_state.json 读取失败"
        if not rebalance_cache_trusted:
            self._log(
                f"  ⚠️ 再平衡缓存视为无效（{rebalance_cache_reject_reason}），将忽略已有 "
                "Checkpoint/轨迹，从当前坐标重新做一次完整的 Boresch 限制力再平衡。"
            )

        # 🔑 之前这里检测到 rebalance_state.json 标记为 completed 时会直接
        # `return self.positions/self.box_vectors`——但 self.positions 此时是
        # pipeline 构造时加载的坐标（通常来自基线预平衡 pre_equilibration.dcd
        # 的最后一帧，见 runabfe.py::load_native_system 的 prefer_equilibrated
        # 逻辑），根本不是 rebalance.chk/rebalance_traj.dcd 里真正带 Boresch
        # 限制力平衡过的那一帧——等于把还没做完 Boresch 限制力再平衡的坐标当成
        # 已经做完的结果直接返回。真正正确的"跳过"不是提前 return，而是让下面
        # 的续跑逻辑（resume_enabled=True + loadCheckpoint + steps_remaining=
        # max(0, n_steps-currentStep)==0）自然处理：steps_remaining<=0 时不会
        # 再步进，但仍会从刚加载的 checkpoint 里正确提取出已经完成再平衡的
        # 坐标/盒子。这里只做提示性日志，不再提前返回。
        if (
            resume and rebalance_cache_trusted and os.path.exists(state_path)
            and _is_checkpoint_valid(chk_path) and _is_traj_valid(traj_path, min_frames=1)
        ):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    rebalance_state = json.load(f)
                if rebalance_state.get("status") == "completed" and rebalance_state.get("n_steps") == int(n_steps):
                    self._log("  ♻️ 再平衡状态已完成，Checkpoint/轨迹有效，将从 Checkpoint 加载真实再平衡坐标（不重新步进）。")
            except Exception as e:
                self._log(f"  ⚠️ 再平衡完成态读取失败 ({e})，继续按 Checkpoint 续跑逻辑处理")

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
        
        # ✅ 挂载统一 Reporter（rebalance_cache_trusted=False 时同 resume 一样
        # 不追加旧轨迹——那份轨迹是在不同的 Boresch 限制力下产生的，续接只会
        # 拼出一条物理上不连续的假轨迹）
        dcd_path, log_path, _ = attach_simulation_reporters(
            simulation, "rebalance", self.output_dir,
            traj_interval=2000, energy_interval=500, chk_interval=5000,
            append_traj=resume and rebalance_cache_trusted and _is_checkpoint_valid(chk_path),
        )

        # ✅ 续跑逻辑（rebalance_cache_trusted=False 时强制不加载旧 checkpoint，
        # 即使文件本身格式有效——它是在不同的 Boresch 限制力下产生的状态）
        resume_enabled = False
        if resume and rebalance_cache_trusted and _is_checkpoint_valid(chk_path):
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
                        "fingerprint": expected_rebalance_fingerprint,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:
            self._log(f"  ⚠️ 再平衡完成态保存失败: {e}")

        self._log(f"  ✓ 再平衡完成 | 坐标已更新")
        # 标记本进程已完成 Boresch 再平衡（见 __init__ 里的说明和
        # run_full_pipeline() 内部预平衡块对这个标记的使用）。
        self._boresch_rebalance_done_this_process = True
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
        finite_difference_delta: float = 0.01,
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
            if stage_name == "vanishing":
                # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=21] 无 pilot 度规可用的兜底
                # 分支：度规驱动布点在这里无从谈起，退回纯几何的平方调度，直接生成
                # 最终态数（不再走「17 基础 + 4 人工 + 2 bridge」那条链——那条链的
                # 存在理由是补救被错误 alpha 压缩到 λ≈1 的度规，已随 alpha 修正消失）。
                _fallback_path = quadratic_vanishing_base_lambdas(
                    VANISHING_FINAL_STATE_COUNT
                )
                validate_vanishing_lambda_path_invariants(_fallback_path)
                _human_ranges = vanishing_subdomain_ranges_from_lambdas(
                    _fallback_path,
                    first_ensemble_target_intervals=(
                        VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS
                    ),
                )
                validate_single_shared_boundary_ranges(
                    _human_ranges, len(_fallback_path)
                )
                return {
                    "lambdas_var": _fallback_path.tolist(),
                    "n_states": len(_fallback_path),
                    "window_ranges": _human_ranges,
                    "softcore_params": softcore_obj,
                    "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
                    "path_diagnostics": {
                        "lambda_placement_method": "quadratic_geometric_fallback_v21",
                        "probe_controls_base_lambda_placement": False,
                        "adaptive_metric_disabled": True,
                    },
                    "path_optimization_disabled_reason": (
                        f"unsupported_probe_for_{potential_type}"
                    ),
                }
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
                opt_res = optimizer.optimize_stage2_vanishing(
                    n_states=n_states,
                    n_steps_per_state=n_steps_per_state,
                    finite_difference_delta=finite_difference_delta,
                )
            self._log(f"[OUTPUT] 优化成功返回: keys={list(opt_res.keys())}")
        except Exception as e:
            self._log(f"[OUTPUT] 优化器异常捕获: {e}")
            try:
                import traceback
                self._log(traceback.format_exc())
            except Exception:
                pass
            raise RuntimeError(
                f"{stage_name} 热力学路径 pilot 失败；拒绝静默回退线性 λ 路径: {e}"
            ) from e
        
        opt_n_states = int(opt_res["n_states"])
        opt_window_ranges = opt_res.get("window_ranges")
        if opt_window_ranges is None:
            if stage_name == "vanishing":
                raise RuntimeError(
                    "vanishing 优化器未返回论文双子区间 window_ranges；"
                    "拒绝回退 overlap=2 自动划窗。"
                )
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
            "path_protocol_version": opt_res.get("path_protocol_version"),
            "path_diagnostics": opt_res.get("path_diagnostics", {}),
        }

    def _retired_overlapping_vdw_schedule_design(
        self,
        lambdas_var: List[float],
        window_ranges: List[Tuple[int, int]],
        path_diagnostics: Dict,
        potential_type: str,
        dexp_params: Optional[Dict],
        boresch_params: Optional[Dict],
        probe_max_bias_updates: int = 15,
        probe_max_warmup_steps: int = 150000,
        probe_required_consecutive: int = 2,
        lse_log_residual_tolerance: float = 0.5,
        max_window_splits: int = 32,
        max_lambda_insertions: int = 4,
        max_insertions_per_initial_edge: int = 2,
    ) -> Tuple[List[float], List[Tuple[int, int]], Dict]:
        """Retired v9 scratch designer; retained only for failure-record archaeology.

        Lambda placement and IBS ensemble width are separate decisions.  The
        pilot thermodynamic metric supplies the initial lambda grid.  Short
        runs of the actual production IBS Hamiltonian then test the paper's
        Log-Sum-Exp fixed point.  K>=3 failures split the ensemble using only
        existing nodes.  A K=2 failure is irreducible, so and only so may one
        thermodynamic-length midpoint be inserted; its two replacement IBS
        ensembles must be tested again.  All probes live under a scratch tree
        and never mutate production data or invoke fixed-H/MBAR overlap.
        """
        raise RuntimeError(
            "retired in thermodynamic-path protocol v10: vanishing is one "
            "integrated [0:K] IBS ensemble and cannot enter recursive window design"
        )
        lambdas = [float(x) for x in lambdas_var]
        ranges = [(int(s), int(e)) for s, e in window_ranges]
        pilot_lambdas = path_diagnostics.get("pilot_lambdas")
        pilot_cumulative = path_diagnostics.get(
            "pilot_cumulative_thermodynamic_length"
        )
        if not pilot_lambdas or not pilot_cumulative:
            raise RuntimeError(
                "Stage-2 LSE schedule design 缺少 pilot lambda/累计热力学长度，"
                "拒绝用算术 lambda 中点替代。"
            )

        edge_roots = list(range(len(lambdas) - 1))
        insertions_by_root = [0] * len(edge_roots)
        total_insertions = 0
        total_splits = 0
        probe_counter = 0
        validated_signatures = set()
        history = []
        scratch_root = os.path.join(
            self.output_dir, "schedule_design", "vanishing_ibs_lse"
        )
        os.makedirs(scratch_root, exist_ok=True)
        alchemical_params = _resolve_alchemical_params(
            potential_type, dexp_params, self.ligand_indices
        )

        def _signature(start: int, end: int) -> Tuple[float, ...]:
            return tuple(round(float(x), 14) for x in lambdas[start:end])

        while True:
            pending = [
                (s, e) for s, e in ranges
                if _signature(s, e) not in validated_signatures
            ]
            if not pending:
                break
            start, end = pending[0]
            probe_counter += 1
            probe_dir = os.path.join(scratch_root, f"probe_{probe_counter:03d}")
            probe_checkpoint_dir = os.path.join(probe_dir, "checkpoints")
            os.makedirs(probe_checkpoint_dir, exist_ok=True)
            manager = IBSWindowManagerDualLambda(
                system_template=self.system,
                topology=self.topology,
                perturbed_atom_indices=self.ligand_indices,
                lambdas_coul=[0.0] * len(lambdas),
                lambdas_vdw=lambdas,
                temperature=self.temperature,
                window_ranges=[(start, end)],
                alchemical_params=alchemical_params,
                potential_type=potential_type,
                restraint_params=boresch_params,
                prefix=f"abfe_dual_design_{probe_counter}",
                platform_name=self.platform_name,
                output_dir=probe_dir,
                checkpoint_dir=probe_checkpoint_dir,
            )
            try:
                result = manager.run_all_windows(
                    positions=self.positions,
                    box_vectors=self.box_vectors,
                    n_steps_per_window=0,
                    steps_per_update=500,
                    stage_type="vdw",
                    resume=False,
                    enable_gradual_warmup=True,
                    warmup_steps=int(probe_max_warmup_steps),
                    min_bias_updates=min(6, int(probe_max_bias_updates)),
                    max_bias_updates=int(probe_max_bias_updates),
                    required_consecutive_bias_updates=int(
                        probe_required_consecutive
                    ),
                    max_bias_warmup_steps=int(probe_max_warmup_steps),
                    mbar_calibration_reserved_steps=0,
                    repair_policy="non_mutating_v1",
                    lse_log_residual_tolerance=float(
                        lse_log_residual_tolerance
                    ),
                    warmup_only=True,
                )
            except IBSWarmupConvergenceError as error:
                diagnostics = dict(error.diagnostics or {})
                diagnostics["global_state_range"] = [int(start), int(end)]
                history.append({
                    "probe": int(probe_counter),
                    "range": [int(start), int(end)],
                    "lambdas_vdw": [float(x) for x in lambdas[start:end]],
                    "passed": False,
                    "lse_balance": diagnostics.get("lse_balance"),
                })
                if end - start >= 3:
                    if total_splits >= int(max_window_splits):
                        raise RuntimeError(
                            "IBS LSE schedule design 达到拆窗上限仍未稳定；"
                            "拒绝提交 schedule，请检查构象/pose。"
                        ) from error
                    ranges, feedback = split_window_from_ibs_lse_failure(
                        ranges, diagnostics, len(lambdas)
                    )
                    total_splits += 1
                    history[-1]["action"] = feedback
                    continue

                root = int(edge_roots[start])
                if (
                    total_insertions >= int(max_lambda_insertions)
                    or insertions_by_root[root]
                    >= int(max_insertions_per_initial_edge)
                ):
                    raise RuntimeError(
                        "不可再拆的两态 IBS ensemble 在热力学中点加密达到上限后仍未满足 "
                        "Log-Sum-Exp 自洽方程；这不是继续增加 lambda 能可靠修复的问题，"
                        "拒绝提交 schedule，请转 structural diagnosis / pose audit。"
                    ) from error
                lambdas, ranges, feedback = (
                    insert_thermodynamic_midpoint_from_ibs_lse_failure(
                        lambdas,
                        ranges,
                        diagnostics,
                        pilot_lambdas,
                        pilot_cumulative,
                    )
                )
                edge_roots[start:start + 1] = [root, root]
                insertions_by_root[root] += 1
                total_insertions += 1
                history[-1]["action"] = feedback
                continue

            if not result or len(result) != 1:
                raise RuntimeError(
                    f"IBS LSE design probe 未返回唯一窗口诊断: {result}"
                )
            validated_signatures.add(_signature(start, end))
            history.append({
                "probe": int(probe_counter),
                "range": [int(start), int(end)],
                "lambdas_vdw": [float(x) for x in lambdas[start:end]],
                "passed": True,
                "lse_balance": result[0].get("lse_balance"),
            })

        return lambdas, ranges, {
            "status": "passed",
            "criterion": "ibs_log_sum_exp_fixed_point",
            "lse_log_residual_tolerance": float(lse_log_residual_tolerance),
            "initial_n_states": int(len(edge_roots) + 1 - total_insertions),
            "final_n_states": int(len(lambdas)),
            "total_window_splits": int(total_splits),
            "total_lambda_insertions": int(total_insertions),
            "insertions_by_initial_edge": [int(x) for x in insertions_by_root],
            "final_window_ranges": [list(r) for r in ranges],
            "probe_history": history,
            "scratch_root": os.path.abspath(scratch_root),
            "fixed_h_overlap_used": False,
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
        decharge_method: str = "pme",
        shadow_bridge_lambdas: Optional[List[float]] = None,
        shadow_bridge_n_steps: int = 200000,
        shadow_bridge_exchange_interval: int = 1000,
        production_step_overrides: Optional[Dict[int, int]] = None,
        frozen_validation_step_overrides: Optional[Dict[int, int]] = None,
        frozen_validation_is_final_rung: Optional[Dict[int, bool]] = None,
        pilot_lambdas: Optional[List[float]] = None,
        pilot_mean_dU_dlambda: Optional[List[float]] = None,
        allow_partial_vanishing_rescue: bool = False,
        stage_output_dir_override: Optional[str] = None,
        checkpoint_dir_override: Optional[str] = None,
        remd_max_resident_contexts: Optional[int] = None,
        **kwargs,
    ) -> Dict:
        """
        执行单个双λ阶段 (去电荷 或 去VDW)
        职责：路由采样 -> 获取结果

        pilot_lambdas/pilot_mean_dU_dlambda: [IBS_BIAS_PROTOCOL_VERSION=20]
            可选的 Stage 2 pilot 探针真实测出的 λ 网格与对应的平均梯度
            （mean_dU_dlambda_kJ_mol，不是用于摆 λ 点密度的方差代理
            metric_g），透传给 IBSWindowManagerDualLambda 供
            run_all_windows 给每个全新（非 resume）窗口的 f_k 做 TI 热启动，
            而不是从 0.0 冷启动。只有 vanishing/vdw 阶段有意义；decharging
            传 None 即可（走 REMD/PME-MBAR，不消费这两个参数）。

        production_step_overrides: 可选的 {window_idx: 该窗口实际生产步数} 覆盖表，
            透传给 ibs_engine.py::run_all_windows（仅 vdw/"vanishing" 阶段的
            manager.run_all_windows 调用会用到；decharging 走 REMD/PME-MBAR 分支，
            不消费这个参数）。供 _run_stage_with_overlap_autorepair 的
            production ESS sampling-repair 分支使用，真正延长某个窗口的采样，
            而不是删旧样本后用同样步数重采一次。

        frozen_validation_step_overrides: 可选的 {window_idx: 该窗口冻结验证的
            累计目标预算步数} 覆盖表 [IBS_BIAS_PROTOCOL_VERSION=12]，透传给
            ibs_engine.py::run_all_windows。fixed-H overlap 全通过、MBAR 校准
            探针也已达标，但冻结验证没能在默认的 mbar_calibration_reserved_steps
            内通过覆盖门时使用；供 _run_stage_with_overlap_autorepair 的
            calibration_pending_validation 分支使用，按 50k→150k→300k 阶梯延长，
            resume 时跳过 SGD 和 fixed-H 探针/重新校准，只续验累计预算里还没跑
            完的差值（ibs_engine.py 内部按 frozen_validation_cumulative_steps
            记账，不会把这个字段当成"这次要新跑多少步"）。

        frozen_validation_is_final_rung: 可选的 {window_idx: 这次是否已经是
            冻结验证阶梯的最后一档} 表，透传给 ibs_engine.py::run_all_windows。
            标记为 True 后如果仍未通过独立验证，该窗口会被直接判定为终态失败
            （bias_status="calibrated_validation_failed"，raise
            IBSFrozenCalibrationValidationError(terminal=True)），不再落盘为
            "calibrated_pending_validation"、不再暗示还能自动继续延长。

        decharge_method 只影响 stage_name == "decharging"：
          - "pme" (默认，原有接口/行为不变)：NonbondedForce ParameterOffset +
            传统 REMD/MBAR，保留完整 PME 长程静电。
          - "shadow_ibs" (实验性，尚未经物理验证)：改走 Shadow-Coulomb IBS，
            见 _run_shadow_ibs_decharging_leg。
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

        if stage_name == "decharging" and decharge_method == "shadow_ibs":
            return self._run_shadow_ibs_decharging_leg(
                lambdas_shadow_coul=lambdas_var,
                n_steps_per_window=n_steps_per_window,
                steps_per_update=steps_per_update,
                resume=resume,
                boresch_params=boresch_params,
                window_ranges=window_ranges,
                enable_gradual_warmup=enable_gradual_warmup,
                warmup_steps=warmup_steps,
                shadow_bridge_lambdas=shadow_bridge_lambdas or [0.0, 0.5, 1.0],
                shadow_bridge_n_steps=shadow_bridge_n_steps,
                shadow_bridge_exchange_interval=shadow_bridge_exchange_interval,
            )

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
                    "lambda_endpoint_diagnostics": _stage_lambda_endpoint_diagnostics(
                        stage_name, lambdas_coul, lambdas_vdw
                    ),
                    "converged": res.get("converged"),
                    "min_overlap": res.get("min_overlap"),
                    "min_overlap_threshold": (
                        res.get("diagnostics", {}).get("min_overlap_threshold")
                    ),
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
                    max_resident_contexts=remd_max_resident_contexts,
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
                "lambda_endpoint_diagnostics": _stage_lambda_endpoint_diagnostics(
                    stage_name, lambdas_coul, lambdas_vdw
                ),
                "converged": res.get("converged"),
                "min_overlap": res.get("min_overlap"),
                "min_overlap_threshold": (
                    res.get("diagnostics", {}).get("min_overlap_threshold")
                ),
                "diagnostics": res.get("diagnostics", {}),
            }

        # 2. 划分 IBS ensembles
        # Vanishing preserves every immutable human anchor while the Fisher
        # probe may insert additional states. Neighbors share one endpoint.
        if stage_name == "vanishing" and not allow_partial_vanishing_rescue:
            validate_vanishing_lambda_path_invariants(lambdas_var)
            expected_subdomain_ranges = vanishing_subdomain_ranges_from_lambdas(
                lambdas_var,
                first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
            )
            normalized_ranges = [
                tuple(int(x) for x in r) for r in (window_ranges or [])
            ]
            if normalized_ranges != expected_subdomain_ranges:
                raise RuntimeError(
                    "vanishing v12 只接受热力学坐标上的 few-state IBS 子区间："
                    f"expected={expected_subdomain_ranges}, got={normalized_ranges}. "
                    "禁止共享两个节点（从而重复一条 λ interval）的 legacy overlap=2 "
                    "或滑动窗口布局。"
                )
            validate_single_shared_boundary_ranges(
                expected_subdomain_ranges, len(lambdas_var)
            )
            window_ranges = expected_subdomain_ranges
        if window_ranges is not None:
            covered = sorted({idx for s, e in window_ranges for idx in range(s, e)})
            if not allow_partial_vanishing_rescue and covered != list(range(n_states)):
                raise RuntimeError(
                    f"显式传入的 window_ranges 覆盖范围与 n_states={n_states} 不匹配"
                    f"（覆盖 {covered[:3]}...{covered[-3:] if covered else []}，"
                    f"共 {len(covered)} 个索引），拒绝使用可能导致漏采样/越界的窗口划分。"
                )
            _edge_ranges = [(int(s), int(e) - 1) for s, e in window_ranges]
            _shared_boundary_nodes = [
                int(window_ranges[i][0])
                for i in range(1, len(window_ranges))
                if int(window_ranges[i - 1][1]) - 1 == int(window_ranges[i][0])
            ]
            self._log(
                f"  🪟 {len(window_ranges)} 个 IBS ensemble："
                f"λ 节点范围（半开）={window_ranges}；"
                f"λ interval 分区（半开、互不重复）={_edge_ranges}；"
                f"复用的共同边界节点={_shared_boundary_nodes}"
            )
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
        stage_output_dir = (
            str(stage_output_dir_override)
            if stage_output_dir_override is not None
            else os.path.join(self.output_dir, stage_name)
        )
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
            prefix=("abfe_dual_rescue" if allow_partial_vanishing_rescue else "abfe_dual"),
            platform_name=self.platform_name,
            output_dir=stage_output_dir,
            checkpoint_dir=(
                str(checkpoint_dir_override)
                if checkpoint_dir_override is not None
                else self.checkpoint_dir
            ),
            pilot_lambdas=pilot_lambdas,
            pilot_mean_dU_dlambda=pilot_mean_dU_dlambda,
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
            min_bias_updates=kwargs.get("min_bias_updates", 12),
            max_bias_updates=kwargs.get("max_bias_updates", 50),
            required_consecutive_bias_updates=kwargs.get(
                "required_consecutive_bias_updates", 3
            ),
            max_bias_warmup_steps=kwargs.get("max_bias_warmup_steps", 500000),
            production_step_overrides=production_step_overrides,
            frozen_validation_step_overrides=frozen_validation_step_overrides,
            frozen_validation_is_final_rung=frozen_validation_is_final_rung,
            # 🔑 [non_mutating_v1] 显式声明非变异策略：预热完成一次足额 fixed-f
            # attempt 后即锁定 f_k 进入独立生产；不跑 fixed-H 探针、不就地重校准。
            # 最终可用性由生产后的 overlap/ESS/去相关样本/不确定度硬门判断。
            repair_policy="non_mutating_v1",
            lse_log_residual_tolerance=kwargs.get(
                "ibs_lse_log_residual_tolerance", 0.5
            ),
            # 🔑 之前 enable_early_stop 在这里是纯空转——_run_dual_lambda_stage
            # 接受这个参数，但从未转发给 run_all_windows，run_all_windows 也
            # 从未定义对应参数，打开这个开关不会有任何效果。现在真正接入；
            # 默认仍是 False，且阈值尚未做离线轨迹回放校准（见
            # ibs_engine.py::EARLY_STOP_PROTOCOL_VERSION 定义处），不要在生产
            # 配置里显式打开它。
            enable_early_stop=enable_early_stop,
            early_stop_min_steps=kwargs.get("early_stop_min_steps", 100000),
            early_stop_check_interval_steps=kwargs.get("early_stop_check_interval_steps", 20000),
            early_stop_required_consecutive_passes=kwargs.get(
                "early_stop_required_consecutive_passes", 3
            ),
            early_stop_min_ess_ratio=kwargs.get("early_stop_min_ess_ratio", 0.05),
            early_stop_min_absolute_ess=kwargs.get("early_stop_min_absolute_ess", 50.0),
            early_stop_min_decorrelated_samples=kwargs.get("early_stop_min_decorrelated_samples", 20),
            early_stop_max_delta_g_drift_kJ_mol=kwargs.get("early_stop_max_delta_g_drift_kJ_mol", 0.5),
            early_stop_max_uncertainty_kJ_mol=kwargs.get("early_stop_max_uncertainty_kJ_mol", 1.0),
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
            stage_name=stage_name,
            # 🔑 [P1 修复] 最终收敛门四项阈值，跟 run_full_pipeline 里构建
            # _final_gate_thresholds 时用的 kwargs.get(..., 默认值) 必须完全
            # 一致，否则协议指纹记录的阈值和这里真正生效的阈值会对不上。
            final_min_ess_ratio=kwargs.get("final_min_ess_ratio", 0.05),
            final_min_absolute_ess=kwargs.get("final_min_absolute_ess", 50.0),
            final_min_decorrelated_samples=kwargs.get("final_min_decorrelated_samples", 20),
            final_max_uncertainty_kJ_mol=kwargs.get("final_max_uncertainty_kJ_mol", 1.0),
        )
        if stage_result.get("error"):
            raise RuntimeError(
                f"{stage_name} 阶段全局 TMBAR 失败: {stage_result['error']}"
            )
        stage_result.setdefault("stage", stage_name)
        stage_result.setdefault("n_states", int(n_states))
        stage_result["lambda_endpoint_diagnostics"] = _stage_lambda_endpoint_diagnostics(
            stage_name,
            manager.lambdas_coul,
            manager.lambdas_vdw,
        )
        self._populate_stage_diagnostics(stage_result)
        return stage_result

    @staticmethod
    def _populate_stage_diagnostics(stage_result: Dict) -> Dict:
        """🔑 [P1-15] 把 stage 求解结果里的收敛/覆盖证据汇进 `diagnostics`。

        抽成独立方法是因为**它此前只在 `_run_ibs_stage` 里被调用**，
        而 vanishing rescue 合并那条路径直接调 `solve_stage_integrated`、绕过了
        `_run_ibs_stage`，于是 `stage_result["diagnostics"]` 从来没被填过。
        后果是 `_build_stage_cache_payload` 存下 `diagnostics={}`、
        `final_results.json` 的 `stage_diagnostics.stage2` 也是空的——
        2026-07-27 那次 `ΔG_vdw = 145.908 ± 1.384` 因此**没有任何审计痕迹**：
        逐段 ΔG、逐窗 σ、converged、ESS、乃至"发生过 rescue"全部无处可查。

        这里只做汇总与搬运，不改任何数值。
        """
        stage_result.setdefault("diagnostics", {})
        stage_result["diagnostics"].update({
            "method": stage_result.get("method"),
            # 🔑 曾经是自由能间距的假代理指标，现在是真实的重加权有效样本比例
            # （见 ibs_engine.py::GlobalMBARAnalyzer.solve_stage_integrated，审查报告 #2）。
            "min_overlap": stage_result.get("min_overlap"),
            "min_overlap_threshold": stage_result.get("min_overlap_threshold"),
            "min_overlap_method": stage_result.get("min_overlap_method"),
            "window_overlap_diagnostics": stage_result.get("window_overlap_diagnostics"),
            "statistical_inefficiency_per_window": stage_result.get("statistical_inefficiency_per_window"),
            "offset_error_contribution": stage_result.get("offset_error_contribution"),
            "uncertainty_note": stage_result.get("uncertainty_note"),
            # 🔑 [P1-15] 以下这些原本只存在于 solve_stage_integrated 的返回顶层，
            # 从不进 diagnostics、也就从不落盘。光看 total_delta_G 无法复核为何放行。
            "converged": stage_result.get("converged"),
            "coverage_diagnostics": stage_result.get("coverage_diagnostics"),
            "covariance_chain_segments": stage_result.get("covariance_chain_segments"),
            "total_error_method": stage_result.get("total_error_method"),
            "ess_gate_protocol_version": stage_result.get("ess_gate_protocol_version"),
            "min_occupancy_normalized": stage_result.get("min_occupancy_normalized"),
            "min_occupancy_normalized_threshold": stage_result.get("min_occupancy_normalized_threshold"),
            "min_decorrelated_samples": stage_result.get("min_decorrelated_samples"),
            "min_decorrelated_samples_threshold": stage_result.get("min_decorrelated_samples_threshold"),
            "max_endpoint_uncertainty_kJ_mol": stage_result.get("max_endpoint_uncertainty_kJ_mol"),
            "max_endpoint_uncertainty_kJ_mol_threshold": stage_result.get(
                "max_endpoint_uncertainty_kJ_mol_threshold"
            ),
            "min_absolute_ess": stage_result.get("min_absolute_ess"),
            "min_absolute_ess_threshold": stage_result.get("min_absolute_ess_threshold"),
            "raw_min_overlap": stage_result.get("raw_min_overlap"),
            "max_common_mode_log_sigma_kT": stage_result.get("max_common_mode_log_sigma_kT"),
            # rescue provenance：只有走过 rescue 的 stage 才有，普通 stage 为 None。
            "immutable_bridge_rescue": stage_result.get("immutable_bridge_rescue"),
            "production_rescue_targets": stage_result.get("production_rescue_targets"),
        })
        return stage_result

    # =========================================================================
    # 4.4 Shadow-Coulomb IBS 去电荷 (实验性备选路径，见 decharge_method="shadow_ibs")
    # =========================================================================
    def _run_shadow_ibs_decharging_leg(
        self,
        lambdas_shadow_coul: List[float],
        n_steps_per_window: int,
        steps_per_update: int,
        resume: bool,
        boresch_params: Optional[Dict],
        window_ranges: Optional[List[Tuple[int, int]]],
        enable_gradual_warmup: bool,
        warmup_steps: int,
        shadow_bridge_lambdas: List[float],
        shadow_bridge_n_steps: int,
        shadow_bridge_exchange_interval: int,
    ) -> Dict:
        """
        Shadow-Coulomb IBS 去电荷完整入口 (实验性，尚未经物理验证)。

        两段独立的热力学循环腿，ΔG 直接相加、误差按平方和开根合并：
          1. Bridge 腿：真实 PME 满电荷 -> Shadow 满电荷 端点转换
             (run_shadow_bridge_leg，传统 REMD+MBAR，只有几个窗口)。
          2. Shadow-IBS 腿：Shadow 满电荷 -> Shadow 去电荷
             (IBSWindowManagerShadowCoul，真正的 IBS 多态偏置采样)。

        与默认的 decharge_method="pme" (NonbondedForce ParameterOffset +
        REMD/MBAR) 相比，这条路径的电荷维度全程留在 IBS 偏置框架内，代价是
        多引入一段 Bridge 腿，且目前没有独立物理验证，只应视为实验性尝试。
        """
        stage_name = "decharging"
        stage_output_dir = os.path.join(self.output_dir, stage_name)
        os.makedirs(stage_output_dir, exist_ok=True)
        temp_k = self.temperature.value_in_unit(unit.kelvin)

        self._log(f"\n{'=' * 60}")
        self._log(f"[双λ阶段] {stage_name.upper()} | decharge_method=shadow_ibs (实验性)")
        self._log(f"{'=' * 60}")

        # ---------- 1. Bridge 腿：PME 满电荷 <-> Shadow 满电荷 ----------
        bridge_result_file = os.path.join(stage_output_dir, "shadow_bridge_result.json")
        if resume and os.path.exists(bridge_result_file):
            self._log("  ♻️ 检测到已有 Shadow-Bridge 结果缓存，跳过重新采样")
            with open(bridge_result_file) as f:
                bridge_result = json.load(f)
        else:
            self._log(f"  🌉 [Bridge 腿] 运行 {len(shadow_bridge_lambdas)} 个窗口 (PME↔Shadow 满电荷端点)...")
            bridge_result = run_shadow_bridge_leg(
                system=self.system,
                topology=self.topology,
                positions=self.positions,
                box_vectors=self.box_vectors,
                perturbed_indices=self.ligand_indices,
                lambdas_bridge_s=shadow_bridge_lambdas,
                temperature_k=temp_k,
                platform_name=self.platform_name,
                output_dir=os.path.join(stage_output_dir, "shadow_bridge"),
                n_steps_per_state=shadow_bridge_n_steps,
                exchange_interval=shadow_bridge_exchange_interval,
                restraint_params=boresch_params,
            )
            with open(bridge_result_file, "w") as f:
                json.dump(bridge_result, f, indent=2)
        # 🔑 之前这里（以及缓存命中分支）从不检查 Bridge 腿自己的 converged/
        # min_overlap——run_shadow_bridge_leg 内部的 TraditionalMBARAnalyzer.solve()
        # 早就算出了这两项（min_overlap_threshold=0.03 硬门），但只要没抛异常就
        # 被当作"腿完成"直接往下用，一条重叠度低到不可信的 Bridge 腿也能悄悄
        # 混进最终 Stage1 结果。这里和下面的 Shadow-IBS 腿一样，执行与正式
        # stage（_assert_stage_result_sane）同等级的硬门。
        if bridge_result.get("converged") is False:
            raise RuntimeError(
                f"Shadow-Bridge 腿报告 converged=False（min_overlap="
                f"{bridge_result.get('min_overlap')}，阈值="
                f"{bridge_result.get('min_overlap_threshold')}）；拒绝把这条子腿的"
                "结果并入 Stage1 去电荷总量。"
            )
        self._log(
            f"  ✓ Bridge 腿完成: ΔG={bridge_result['total_delta_G']:.2f} ± "
            f"{bridge_result['total_error']:.2f} kJ/mol"
        )

        # ---------- 2. Shadow-IBS 腿：Shadow 满电荷 -> Shadow 去电荷 ----------
        n_states = len(lambdas_shadow_coul)
        if window_ranges is not None:
            covered = sorted({idx for s, e in window_ranges for idx in range(s, e)})
            if covered != list(range(n_states)):
                raise RuntimeError(
                    f"显式传入的 window_ranges 覆盖范围与 n_states={n_states} 不匹配，"
                    "拒绝使用可能导致漏采样/越界的窗口划分。"
                )
        else:
            window_ranges = generate_overlapping_windows(
                n_states=n_states, n_windows=None, pts_per_window=6, overlap=2
            )
            self._log(f"  🪟 自动划分 {len(window_ranges)} 个 Shadow-IBS 窗口")

        manager = IBSWindowManagerShadowCoul(
            system_template=self.system,
            topology=self.topology,
            perturbed_atom_indices=self.ligand_indices,
            lambdas_shadow_coul=lambdas_shadow_coul,
            temperature=self.temperature,
            window_ranges=window_ranges,
            restraint_params=boresch_params,
            prefix="abfe_shadow",
            platform_name=self.platform_name,
            output_dir=stage_output_dir,
            checkpoint_dir=self.checkpoint_dir,
        )
        manager.output_dir = stage_output_dir

        manager.run_all_windows(
            positions=self.positions,
            box_vectors=self.box_vectors,
            n_steps_per_window=n_steps_per_window,
            steps_per_update=steps_per_update,
            stage_type="shadow_coul",
            resume=resume,
            enable_gradual_warmup=enable_gradual_warmup,
            warmup_steps=warmup_steps,
        )

        kt_val = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(unit.kilojoule_per_mole)
        window_outputs = manager.get_stage_data_for_analysis(stage_type="shadow_coul")
        if not window_outputs:
            raise RuntimeError(
                "Shadow-IBS 去电荷阶段未找到任何窗口能量文件，无法执行全局 TMBAR。"
            )
        shadow_ibs_result = solve_stage_integrated(
            window_outputs=window_outputs, kt=kt_val, stage_name="decharging_shadow_ibs"
        )
        if shadow_ibs_result.get("error"):
            raise RuntimeError(f"Shadow-IBS 去电荷阶段全局 TMBAR 失败: {shadow_ibs_result['error']}")
        # 🔑 之前这里只检查了 error 字段，不检查 converged/min_overlap——
        # solve_stage_integrated 已经实现了同正式 stage 一样的 ESS ratio/绝对
        # ESS/去相关样本数/不确定度四项硬门（见 GlobalMBARAnalyzer.
        # solve_stage_integrated），一个重叠不足但没有报错的 Shadow-IBS 腿会
        # 被当作成功结果直接并入 Stage1 总量。两条子腿必须都执行同等级的硬门，
        # 任一失败即整腿失败。
        if shadow_ibs_result.get("converged") is not True:
            raise RuntimeError(
                f"Shadow-IBS 去电荷子腿未收敛（converged="
                f"{shadow_ibs_result.get('converged')!r}，min_overlap="
                f"{shadow_ibs_result.get('min_overlap')}，阈值="
                f"{shadow_ibs_result.get('min_overlap_threshold')}）；拒绝把这条"
                "子腿的结果并入 Stage1 去电荷总量。"
            )
        self._log(
            f"  ✓ Shadow-IBS 腿完成: ΔG={shadow_ibs_result['total_delta_G']:.2f} ± "
            f"{shadow_ibs_result['total_error']:.2f} kJ/mol"
        )

        # ---------- 3. 合并两段腿 ----------
        total_delta_G = float(bridge_result["total_delta_G"]) + float(shadow_ibs_result["total_delta_G"])
        total_error = float(np.sqrt(bridge_result["total_error"] ** 2 + shadow_ibs_result["total_error"] ** 2))
        self._log(f"  Σ 去电荷总计: ΔG={total_delta_G:.2f} ± {total_error:.2f} kJ/mol")

        # 🔑 两条子腿到这里都已经各自通过了硬门（上面任一失败已经 raise），但
        # 组合后的 Stage1 结果此前从不带顶层 converged/min_overlap/
        # min_overlap_threshold——_assert_stage_result_sane 对这三个字段的检查是
        # "只在结果带有这些诊断字段时才检查"，字段缺失时会被直接跳过，一个已经
        # 用不可信数据拼出来的 Stage1 结果本该被这道门拦下，却因为字段缺失静默
        # 放行。这里显式带上"两条子腿共同的最差重叠度"，让最终写入 checkpoint
        # 的结果也过一遍同样的硬门（此处理论上恒为 converged=True，因为不满足
        # 就已经在上面 raise 了；这里落盘纯粹是让 _assert_stage_result_sane 和
        # 未来任何读这份 checkpoint 的代码都能看到这两条子腿真实的重叠度，而不是
        # 只能看到最终拼接后的 delta_G/error）。
        both_min_overlaps = [
            m for m in (bridge_result.get("min_overlap"), shadow_ibs_result.get("min_overlap"))
            if m is not None
        ]
        return {
            "stage": stage_name,
            "total_delta_G": total_delta_G,
            "total_error": total_error,
            "method": "Shadow-Bridge+Shadow-IBS-TMBAR",
            "n_states": int(n_states),
            "converged": True,
            "min_overlap": float(min(both_min_overlaps)) if both_min_overlaps else None,
            "min_overlap_threshold": shadow_ibs_result.get("min_overlap_threshold"),
            "lambda_endpoint_diagnostics": _stage_lambda_endpoint_diagnostics(
                "decharging", lambdas_shadow_coul, [1.0] * n_states
            ),
            "diagnostics": {
                "experimental": True,
                "physically_validated": False,
                "note": (
                    "decharge_method=shadow_ibs 是实验性路径，尚未经过独立物理验证，"
                    "生产结果请优先使用默认的 decharge_method=pme。"
                ),
                "bridge_leg": bridge_result,
                "shadow_ibs_leg": shadow_ibs_result,
            },
        }

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

    def _assert_committed_boresch_still_matches_pose(
        self,
        *,
        committed_doc: Dict,
        committed_eq: Dict,
        boresch_params: Dict,
        committed_path: str,
    ) -> None:
        """🔑 [BORESCH-COMMIT] 复用已提交平衡值前，验证它仍描述当前构象。

        两道检查：

        1. **锚点身份**：committed 文件记录的 receptor/ligand 锚点必须与本次
           `boresch_params` 一致。锚点变了，平衡值就不是同一个几何量。
           （v1 裸格式没有这些字段，跳过这道，仍走第 2 道。）
        2. **几何一致性**：用当前坐标重算六个自由度，与 committed 值逐个比较，
           偏离以限制势自身的热涨落宽度 σ_i = sqrt(kT/k_i) 为单位。

        第 2 道是主判据，也是唯一能抓住"平衡值比结构还老"这类错误的。
        `update_boresch_from_last_frame` 已有的两道门（角度 40-140°、r0 漂移）
        与它正交，且对实测那组错值全部放行。

        不一致时 **fail closed**，不静默重新锚定：这条 resume 保护存在的理由
        就是防止一条腿中途换哈密顿量，自动重锚会把它变成另一个 bug。
        """
        rec = [int(i) for i in boresch_params.get("receptor_indices", [])]
        lig = [int(i) for i in boresch_params.get("ligand_indices", [])]

        schema = committed_doc.get("schema_version")
        if schema is not None:
            rec_c = [int(i) for i in committed_doc.get("receptor_indices", [])]
            lig_c = [int(i) for i in committed_doc.get("ligand_indices", [])]
            if rec_c and lig_c and (rec_c != rec or lig_c != lig):
                raise RuntimeError(
                    f"已提交的 Boresch 平衡值锚点与本次不一致，拒绝复用：\n"
                    f"  文件 {committed_path}\n"
                    f"  committed receptor={rec_c} ligand={lig_c}\n"
                    f"  本次     receptor={rec}   ligand={lig}\n"
                    "锚点变了，平衡值描述的就不是同一个几何量。"
                )
        else:
            self._log(
                f"  ⚠️ {os.path.basename(committed_path)} 是无身份信息的旧格式"
                f"（schema_version 缺失），无法核对锚点来源；仍将执行几何一致性校验。"
            )

        force_constants = boresch_params.get("force_constants") or {}
        if len(rec) != 3 or len(lig) != 3 or not force_constants:
            self._log("  ⚠️ 缺锚点或力常数，跳过 Boresch 平衡值几何一致性校验。")
            return

        try:
            from abfe_core import calc_boresch_from_last_frame
            current_eq = calc_boresch_from_last_frame(self.positions, rec, lig)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"无法用当前坐标重算 Boresch 几何以校验已提交平衡值：{exc}。"
                "拒绝在未经校验的情况下复用——这正是 2026-07-10 那组错值能沿用 17 天的原因。"
            ) from exc

        report = boresch_committed_deviation_sigma(
            committed_eq, current_eq, force_constants,
            self.temperature.value_in_unit(unit.kelvin),
        )
        if not report:
            self._log("  ⚠️ 没有可比较的自由度，跳过 Boresch 平衡值几何一致性校验。")
            return

        worst_key = max(report, key=lambda k: report[k]["deviation_sigma"])
        worst = report[worst_key]["deviation_sigma"]
        threshold = float(BORESCH_COMMITTED_MAX_DEVIATION_SIGMA)

        if worst <= threshold:
            warn_at = float(BORESCH_COMMITTED_WARN_DEVIATION_SIGMA)
            if worst > warn_at:
                self._log(
                    f"  ⚠️ 已提交 Boresch 平衡值与当前坐标偏离 {worst:.2f} σ @ {worst_key}"
                    f"（告警带 {warn_at:.1f}-{threshold:.1f} σ，未阻断）。"
                    "单帧比较本身带 ~√2σ 噪声，偶发一次可以忽略；但若每次 resume 都出现、"
                    "或多个自由度同时进入该带，说明平衡值已经在偏离当前构象，"
                    "请核对是否该重新锚定。逐自由度："
                )
                for k, v in sorted(report.items(), key=lambda kv: -kv[1]["deviation_sigma"]):
                    self._log(
                        f"       {k:>8s} committed={v['committed']:+.6f} "
                        f"current={v['current']:+.6f} ({v['deviation_sigma']:.2f} σ)"
                    )
            else:
                self._log(
                    f"  ✓ 已提交 Boresch 平衡值与当前坐标一致（最大偏离 "
                    f"{worst:.2f} σ @ {worst_key}，阈值 {threshold:.1f} σ）"
                )
            return

        lines = [
            f"    {k:>8s}  committed={v['committed']:+10.6f}  current={v['current']:+10.6f}"
            f"  Δ={v['delta']:+9.6f}  ({v['deviation_sigma']:6.2f} σ)"
            for k, v in sorted(
                report.items(), key=lambda kv: -kv[1]["deviation_sigma"]
            )
        ]
        raise RuntimeError(
            "已提交的 Boresch 平衡值已经不描述当前构象，拒绝复用（fail closed）。\n"
            f"  文件 {committed_path}\n"
            f"  最大偏离 {worst:.2f} σ @ {worst_key}，阈值 {threshold:.1f} σ\n"
            "  逐自由度（σ = sqrt(kT/k)，即该限制势自身的热涨落宽度）：\n"
            + "\n".join(lines)
            + "\n\n"
            "  这意味着限制力会把配体从它当前的构象上拽走：静电（方向性氢键）会因此\n"
            "  严重偏低，而 vdW 因为对取向不敏感看起来仍然正常——2026-07-27 实测过\n"
            "  这个组合，复合物腿去电荷偏低约 25 kJ/mol。\n\n"
            "  处理方式（二选一，不要直接放宽阈值）：\n"
            f"    A. 若该腿尚无有效采样数据：删除 {committed_path}，\n"
            "       下次运行会用当前坐标重新锚定并带身份信息落盘。\n"
            "    B. 若该腿已有采样数据：那些数据是在这组平衡值下采的，与当前结构不匹配，\n"
            "       必须连同该腿的采样输出一起作废重跑，不能只换平衡值继续拼接。"
        )

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

        # 🔑 [P2-14] LJ 长程尾项到底有没有被附加，必须来自生产者，不能写死 True。
        # 两条生产路径各有自己的真相来源：
        #   * 传统 REMD / single_lambda：TraditionalMBARAnalyzer 已经按正确模式维护
        #     `lj_lrc_metadata`（初始化 applied=False，真算完系数才翻 True），经
        #     `result["diagnostics"]["traditional_lj_lrc"]` 上来——直接读它；
        #   * 双 λ / IBS：真相在 `ibs_wrapper.lj_tail_lrc_coeff_kj_mol is not None`，
        #     该对象不出 ibs_engine，所以与生产者共用同一个谓词
        #     `ibs_lj_tail_lrc_is_applicable`（见其 docstring：共用而不是复写，
        #     DEXP 尾项公式一旦被验证/替换只需改一处）。
        _lj_lrc_potential_type = str(
            (getattr(self, "_last_run_config", {}) or {}).get("potential_type", "softcore")
        )
        _traditional_lrc = (
            (sampling_results.get("diagnostics") or {}).get("traditional_lj_lrc")
            if isinstance(sampling_results, dict)
            else None
        )
        if isinstance(_traditional_lrc, dict) and "applied" in _traditional_lrc:
            _lj_lrc_applicable = bool(_traditional_lrc.get("applied"))
            _lj_lrc_truth_source = "traditional_mbar_analyzer_lj_lrc_metadata"
        else:
            _lj_lrc_applicable = bool(
                ibs_lj_tail_lrc_is_applicable(_lj_lrc_potential_type)
            )
            _lj_lrc_truth_source = "ibs_lj_tail_lrc_is_applicable(potential_type)"

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
            # ✅ total_delta_G_complex_kJ_mol (见下方 total_dg) 恒等于
            # dg_phys + cons_correction + dg_boresch，即 Boresch 释放修正已经烘焙
            # 进 total_delta_G_complex_kJ_mol。runabfe.py 等下游消费者读取本文件时，
            # 不应再对 total_delta_G_complex_kJ_mol 或最终 ΔG_bind 二次扣减
            # boresch_correction_kJ_mol，否则会重复计入该修正项。
            "boresch_correction_already_included_in_total_delta_G": True,
            "boresch_correction_diagnostics": {
                "method": correction_results.get("method"),
                "diagnostics": correction_results.get("diagnostics", {}),
                "force_constants_raw": correction_results.get("force_constants_raw", {}),
                "force_constant_clipped": correction_results.get("force_constant_clipped", {}),
                "uses_analytical_release_formula": bool(correction_results.get("uses_analytical_release_formula", False)),
                "analytical_release_assumption": correction_results.get("analytical_release_assumption", ""),
            },
            "lj_long_range_dispersion_correction": {
                # 🔑 [P2-14] 这里原来是裸字面量 True，对 DEXP 运行无条件说谎。
                # 现在与生产者 build_ibs_dual_system 共用同一个谓词
                # （ibs_engine.ibs_lj_tail_lrc_is_applicable），所以 DEXP 的解析
                # 尾项公式一旦被验证/替换，行为和报告会一起动，不会再分叉。
                "applicable": _lj_lrc_applicable,
                "applied": _lj_lrc_applicable,
                "potential_type": _lj_lrc_potential_type,
                "truth_source": _lj_lrc_truth_source,
                "not_applied_reason": (
                    None if _lj_lrc_applicable
                    else ibs_lj_tail_lrc_inapplicable_reason(_lj_lrc_potential_type)
                ),
                "delta_G_kJ_mol": None,
                "status": (
                    "implemented_analytic_mean_field_switching_softcore_aware"
                    if _lj_lrc_applicable
                    else "not_applied_tail_formula_unvalidated_for_this_potential"
                ),
                "protocol_version": TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
                "note": ("" if _lj_lrc_applicable else (
                    f"NOT APPLIED for potential_type={_lj_lrc_potential_type!r}: "
                    f"{ibs_lj_tail_lrc_inapplicable_reason(_lj_lrc_potential_type)}. "
                    "ibs_engine.py::build_ibs_dual_system leaves "
                    "ibs_wrapper.lj_tail_lrc_coeff_kj_mol = None, and every consumer "
                    "short-circuits to zeros, so the reported free energy contains NO "
                    "long-range dispersion correction. The description that follows "
                    "documents what the correction does when it IS applied; it did not "
                    "run for this result. "
                )) + (
                    "OpenMM's native CustomNonbondedForce.setUseLongRangeCorrection cannot be used here: "
                    "the softcore VDW CV expression bundles LJ and Coulomb into one CustomNonbondedForce, "
                    "and OpenMM's analytic tail integral diverges (and was empirically observed to crash "
                    "the CUDA backend) once real nonzero charges are present in that combined expression "
                    "(see test_lrc_interaction_group_compat.py Q3). Instead, a hand-derived analytic "
                    "correction is precomputed once per window in ibs_engine.py::build_ibs_dual_system "
                    "(_lj_tail_lrc_coefficients_kj_mol) and added per-frame, per-lambda_vdw-state inside "
                    "IBSSampler.collect_energies() before MBAR sees the energies. As of protocol version 2, "
                    "this is a real switching-aware + softcore-aware numerical integral -- it restores both "
                    "the energy the 1.0-1.2nm switching function removes AND the standard tail beyond the "
                    "1.2nm cutoff, integrated against the actual softcore denominator D(r) = "
                    "alpha_lj*(1-lambda_vdw)^m_lj + r^6 (not a bare r^6), and includes both the attractive "
                    "r^-6 and repulsive r^-12 moments (S6, S12) rather than dispersion-only. (Protocol "
                    "version 1, superseded, only integrated the plain r^-6 tail beyond the hard cutoff and "
                    "ignored the switching region entirely.) Because it is folded into the raw interaction "
                    "energies feeding MBAR rather than applied as a separate additive scalar afterward, it "
                    "is not separable into its own delta_G_kJ_mol contribution the way the Boresch "
                    "correction is -- it is already included in total_delta_G_complex_kJ_mol below. Shared "
                    "by three consumers of the same per-lambda coefficient array "
                    "(ibs_wrapper.lj_tail_lrc_coeff_kj_mol): IBSSampler production sampling, the fixed-H "
                    "overlap probe, and (via a parallel construction using BeutlerSoftcoreBuilder's default "
                    "alpha_lj/power_lj) TraditionalMBARAnalyzer.compute_u_kn for the "
                    "BeutlerSoftcoreBuilder/--decoupling single_lambda REMD path, which now has an "
                    "equivalent correction rather than none."
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
        
        # 🔑 [ATT-09] 循环闭合统一走 abfe_core.combine_binding_free_energy（见下方
        # 溶剂腿分支）。这里只先取出 complex 侧的两个量；没有溶剂腿时保持既有行为，
        # 但必须说清楚那不是一个结合自由能。
        dg_complex = complex_res.get(
            "total_delta_G_complex_kJ_mol",
            complex_res.get("total_delta_G", 0.0),
        )
        err_complex = complex_res.get(
            "total_error_kJ_mol",
            complex_res.get("total_error", 0.0),
        )
        delta_g_bind = -dg_complex
        total_err_bind = err_complex
        cycle = None

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
            # ✅ solvent_runner 委托给同一个 run_full_pipeline，其口径与 complex 侧一致，
            # 主键是 total_delta_G_complex_kJ_mol；decoupling_delta_G_kJ_mol 只是
            # _analyze_dual_leg 等旧辅助函数使用的口径，这里只作为兜底，避免误取旧字段。
            dg_solvent = solvent_res.get(
                "total_delta_G_complex_kJ_mol",
                solvent_res.get("decoupling_delta_G_kJ_mol", solvent_res.get("total_delta_G", 0.0)),
            )
            err_solvent = solvent_res.get(
                "total_error_kJ_mol", solvent_res.get("total_error", 0.0)
            )
            # 🔑 [ATT-09] 这里此前是 `delta_g_bind += dg_solvent` 手写闭合，且
            # **完全没有 APBS 项**——runabfe.main() 和 run_post_analysis() 一直在读
            # config["apbs_correction_kJ_mol"]，只有这条路径和 traditional 路径没读，
            # 对带电配体等于静默漏掉整项有限尺寸静电修正。现在与其余三处共用
            # abfe_core.combine_binding_free_energy，并从 run kwargs 读同一个标量
            # （由离线 apbs_correction.py 的 collect 步骤算出并经
            # --apbs-correction-kj-mol 传入；本流程内不调 APBS）。
            #
            # complex 侧取的是 total_delta_G_complex_kJ_mol，Boresch 释放项已烘焙其中
            # （abfe_pipeline: total_dg = dg_phys + cons_correction + dg_boresch），
            # 所以 already_included=True，不再减第二次。
            cycle = combine_binding_free_energy(
                dg_complex_kJ_mol=dg_complex,
                dg_solvent_kJ_mol=dg_solvent,
                err_complex_kJ_mol=err_complex,
                err_solvent_kJ_mol=err_solvent,
                dg_boresch_kJ_mol=complex_res.get("boresch_correction_kJ_mol", 0.0),
                boresch_already_included_in_complex=True,
                apbs_correction_kJ_mol=float(
                    kwargs.get("apbs_correction_kJ_mol", 0.0) or 0.0
                ),
            )
            delta_g_bind = cycle["delta_G_bind_kJ_mol"]
            total_err_bind = cycle["total_error_kJ_mol"]

        if cycle is None:
            print(
                "\n⚠️ 未运行溶剂腿：下面这个数只是 -ΔG_complex，**不是**结合自由能，"
                "热力学循环没有闭合。"
            )
        print(f"\n🎯 最终结合自由能 ΔG_bind = {delta_g_bind:.2f} ± {total_err_bind:.2f} kJ/mol")
        return {
            "delta_g_bind": delta_g_bind,
            "total_error": total_err_bind,
            "complex": complex_res,
            # [ATT-09] 循环闭合的完整记账；未跑溶剂腿时为 None，明确表示循环未闭合。
            "thermodynamic_cycle_terms": cycle,
            "cycle_closed": cycle is not None,
        }

    @staticmethod
    def _load_ibs_window_outputs_from_dir(
        output_dir: str,
        ranges: List[Tuple[int, int]],
        lambdas_coul: List[float],
        lambdas_vdw: List[float],
        *,
        checkpoint_dir: str,
        stage_type: str = "vdw",
        window_index_offset: int = 0,
        window_label_prefix: str = "window",
        excluded_local_windows: Optional[set] = None,
    ) -> List[Dict]:
        outputs: List[Dict] = []
        excluded = {int(x) for x in (excluded_local_windows or set())}
        # 🔑 [P0-8] 见 ibs_engine._assert_expected_windows_all_loaded 的说明：
        # 缺首/末窗口不会被协方差链挡下，会静默产出截断的 ΔG 且报 converged=True。
        # 合法的部分分析（rescue ensemble 取代原始窗口）必须走显式的
        # excluded_local_windows，不能靠"文件恰好不存在"来隐式决定覆盖范围。
        expected_windows = [
            int(i) for i in range(len(ranges)) if int(i) not in excluded
        ]
        missing_windows: List[Dict] = []
        for local_idx, (start, end) in enumerate(ranges):
            if int(local_idx) in excluded:
                continue
            energy_path = os.path.join(
                output_dir, f"dual_window_{local_idx}_{stage_type}_energies.npy"
            )
            bias_path = os.path.join(
                output_dir, f"dual_window_{local_idx}_{stage_type}_bias.npy"
            )
            base_path = os.path.join(
                output_dir, f"dual_window_{local_idx}_{stage_type}_base.npy"
            )
            convergence_path = os.path.join(
                output_dir,
                f"dual_window_{local_idx}_{stage_type}_convergence.json",
            )
            absent = [
                label
                for label, path in (
                    ("energies", energy_path),
                    ("bias", bias_path),
                    ("base", base_path),
                )
                if not os.path.isfile(path)
            ]
            if absent:
                missing_windows.append({
                    "window_index": int(local_idx),
                    "lambda_range": [int(start), int(end)],
                    "missing_files": absent,
                })
                continue
            if not os.path.isfile(convergence_path):
                raise FileNotFoundError(
                    f"窗口 {local_idx} 有 energies 但缺少 convergence manifest"
                )
            with open(convergence_path, encoding="utf-8") as handle:
                convergence = json.load(handle)
            u_kn, bias, base = _load_validated_window_data_triplet(
                energy_path,
                bias_path,
                base_path,
                convergence,
            )
            if u_kn.ndim != 2 or u_kn.shape[1] == 0:
                raise ValueError(f"窗口 {local_idx} 没有有效 IBS 帧")
            n_frames = u_kn.shape[1]
            # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 混合覆盖度 ESS 门需要该窗口冻结进生产
            # 的 f_k，从对应 checkpoint 目录的 ibs_state_*.json 读（与
            # IBSWindowManagerDualLambda.get_stage_data_for_analysis 同一策略，同样
            # fail closed）。注意本函数会被 original/rescue 两个不同 output_dir 各调
            # 一次，checkpoint_dir 必须与 output_dir 配对传入，不能共用一个。
            state_path = os.path.join(
                checkpoint_dir, f"ibs_state_{stage_type}_window_{local_idx}.json"
            )
            if not os.path.isfile(state_path):
                raise FileNotFoundError(
                    f"窗口 {local_idx} 有 energies 但缺少 ibs_state checkpoint "
                    f"({state_path})；ESS 门需要冻结的 f_k，拒绝在缺判据的情况下继续分析"
                )
            with open(state_path, encoding="utf-8") as handle:
                ibs_state = json.load(handle)
            f_k_window = np.asarray(ibs_state.get("f_k", []), dtype=np.float64).ravel()
            if f_k_window.size != u_kn.shape[0] or not np.all(np.isfinite(f_k_window)):
                raise ValueError(
                    f"窗口 {local_idx} ibs_state 的 f_k（长度 {f_k_window.size}）与能量"
                    f"矩阵态数（{u_kn.shape[0]}）不符或含非有限值，拒绝用于 ESS 门"
                )
            outputs.append({
                "window_index": int(window_index_offset + local_idx),
                "window_label": f"{window_label_prefix}_{local_idx}",
                "window_range": [int(start), int(end)],
                "u_kn": u_kn[:, :n_frames],
                "bias_energies": bias[:n_frames],
                "base_energies": base[:n_frames],
                "lambda_indices": list(range(int(start), int(end))),
                "lambdas_coul": [float(x) for x in lambdas_coul[start:end]],
                "lambdas_vdw": [float(x) for x in lambdas_vdw[start:end]],
                "f_k": f_k_window,
                "sampled_distribution_row": 0,
            })
        _assert_expected_windows_all_loaded(
            expected_windows=expected_windows,
            loaded_windows=[
                int(o["window_index"]) - int(window_index_offset) for o in outputs
            ],
            missing_windows=missing_windows,
            source=f"{output_dir} (stage_type={stage_type})",
        )
        return outputs

    @staticmethod
    def _build_vanishing_rescue_ranges(
        failing_window_indices: List[int],
        base_ranges: List[Tuple[int, int]],
    ) -> List[Tuple[int, int]]:
        """Replace each failed ensemble by smaller overlapping ensembles.

        The lambda grid is immutable.  These are new sampling ensembles over
        existing states, kept in a separate rescue directory.
        """
        rescue_ranges: List[Tuple[int, int]] = []
        for window_idx in sorted(set(int(x) for x in failing_window_indices)):
            start, end = (int(x) for x in base_ranges[window_idx])
            n_states = end - start
            if n_states <= 2:
                rescue_ranges.append((start, end))
                continue
            midpoint = start + (n_states - 1) // 2
            left = (start, midpoint + 1)
            right = (midpoint, end)
            for child in (left, right):
                if child[1] - child[0] >= 2 and child not in rescue_ranges:
                    rescue_ranges.append(child)
        return rescue_ranges

    @staticmethod
    def _stage_quality_failure_details(result: Dict) -> List[Dict]:
        """Return per-window final-gate failures with the exact worst state."""
        diagnostics = result.get("window_overlap_diagnostics") or []
        ratio_threshold = result.get("min_overlap_threshold")
        absolute_threshold = result.get("min_absolute_ess_threshold")
        decorrelated_threshold = result.get("min_decorrelated_samples_threshold")
        uncertainty_threshold = result.get(
            "max_endpoint_uncertainty_kJ_mol_threshold"
        )
        failures: List[Dict] = []
        for record in diagnostics:
            reasons = []
            ratio = record.get("min_ess_ratio")
            absolute = record.get("absolute_ess")
            n_decorrelated = record.get("n_frames_decorrelated")
            uncertainty = record.get("endpoint_diff_uncertainty_kJ_mol")
            if ratio_threshold is not None and (
                ratio is None or float(ratio) < float(ratio_threshold)
            ):
                reasons.append("ess_ratio")
            if absolute_threshold is not None and (
                absolute is None or float(absolute) < float(absolute_threshold)
            ):
                reasons.append("absolute_ess")
            if decorrelated_threshold is not None and (
                n_decorrelated is None
                or int(n_decorrelated) < int(decorrelated_threshold)
            ):
                reasons.append("decorrelated_samples")
            if uncertainty_threshold is not None and (
                uncertainty is None
                or not np.isfinite(float(uncertainty))
                or float(uncertainty) > float(uncertainty_threshold)
            ):
                reasons.append("endpoint_uncertainty")
            if not reasons:
                continue

            per_lambda = record.get("ess_ratio_per_lambda") or {}
            worst_state = None
            worst_ratio = None
            if per_lambda:
                worst_key, worst_value = min(
                    per_lambda.items(), key=lambda item: float(item[1])
                )
                worst_state = int(worst_key)
                worst_ratio = float(worst_value)
            global_states = [int(x) for x in (record.get("lambdas") or [])]
            state_position = (
                global_states.index(worst_state)
                if worst_state is not None and worst_state in global_states
                else None
            )
            lambdas_coul = [float(x) for x in (record.get("lambdas_coul") or [])]
            lambdas_vdw = [float(x) for x in (record.get("lambdas_vdw") or [])]
            failures.append({
                "window_index": int(record.get("window_index", -1)),
                "window_label": record.get("window_label"),
                "window_range": record.get("window_range"),
                "global_states": global_states,
                "lambdas_coul": lambdas_coul,
                "lambdas_vdw": lambdas_vdw,
                "worst_global_state": worst_state,
                "worst_lambda_coul": (
                    lambdas_coul[state_position]
                    if state_position is not None and state_position < len(lambdas_coul)
                    else None
                ),
                "worst_lambda_vdw": (
                    lambdas_vdw[state_position]
                    if state_position is not None and state_position < len(lambdas_vdw)
                    else None
                ),
                "worst_ess_ratio": worst_ratio,
                "min_ess_ratio": ratio,
                "absolute_ess": absolute,
                "n_frames_decorrelated": n_decorrelated,
                "endpoint_uncertainty_kJ_mol": uncertainty,
                "failed_gates": reasons,
            })
        return failures

    @staticmethod
    def _format_stage_quality_failure_details(details: List[Dict]) -> str:
        if not details:
            return "未能从 window_overlap_diagnostics 定位具体窗口"
        rows = []
        for item in details:
            label = item.get("window_label") or f"window {item['window_index']}"
            rows.append(
                f"{label} range={item.get('window_range')} states={item.get('global_states')} "
                f"lambda_vdw={item.get('lambdas_vdw')}；worst_state="
                f"{item.get('worst_global_state')} (lambda_vdw={item.get('worst_lambda_vdw')})，"
                f"ESS_ratio={item.get('min_ess_ratio')}, absolute_ESS="
                f"{item.get('absolute_ess')}, N_decorrelated="
                f"{item.get('n_frames_decorrelated')}, endpoint_sigma="
                f"{item.get('endpoint_uncertainty_kJ_mol')} kJ/mol，failed="
                f"{item.get('failed_gates')}"
            )
        return " | ".join(rows)

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

        # 🔑 修复（审查报告 #2）：此前 GlobalMBARAnalyzer/TraditionalMBARAnalyzer
        # 返回的 converged/min_overlap 字段从未被这里检查过——即使它们现在已经是
        # 真实的重叠/收敛诊断（见 ibs_engine.py 的 solve / solve_stage_integrated），
        # 一个重叠度低到不可信的阶段仍然会被当作"合法完成"写进最终 ΔG_bind。这里
        # 补上硬性检查：只在结果里带有这些诊断字段时才检查（兼容不提供这些字段的
        # 旧路径），重叠度低于阶段自己报告的阈值就直接拒绝，而不是把一个已知不可靠
        # 的数字悄悄传下去。
        converged = result.get("converged")
        min_overlap = result.get("min_overlap")
        min_overlap_threshold = result.get("min_overlap_threshold")
        if converged is False:
            failure_details = self._stage_quality_failure_details(result)
            raise RuntimeError(
                f"{stage_label} 阶段报告 converged=False"
                + (f"，min_overlap={min_overlap:.4g}（阈值 {min_overlap_threshold:.4g}）" if min_overlap is not None and min_overlap_threshold is not None else "")
                + "。Reweighting-quality gate failed. Preserve data and run "
                "rescue/coverage analysis; do not mutate the sampling grid in place. "
                f"具体瓶颈：{self._format_stage_quality_failure_details(failure_details)}。"
                "（完整诊断见 window_overlap_diagnostics / statistical_inefficiency；由 rescue/"
                "coverage 审计决定是否需要新 ensemble，不在原地拆窗/插 λ/重校准 f_k。）"
            )
        if min_overlap is not None and min_overlap_threshold is not None and min_overlap < min_overlap_threshold:
            raise RuntimeError(
                f"{stage_label} 阶段单参考重要性 ESS 比值 min_overlap={min_overlap:.4g} 低于阈值 "
                f"{min_overlap_threshold:.4g}（此处 min_overlap 是 compute_effective_sample_number "
                "重要性 ESS 比值，非 fixed-H adjacent overlap），拒绝标记为 completed。"
                "Reweighting-quality gate failed. Preserve data and run rescue/coverage analysis; "
                "do not mutate the sampling grid in place."
            )

        # 🔑 [P1 修复] 最终收敛门此前只检查 ESS ratio；样本总数很少时，即使绝对
        # 有效样本数只有个位数，只要比例超过阈值仍会被判定 completed。这里对
        # ibs_engine.py::GlobalMBARAnalyzer.solve_stage_integrated 新增的三项
        # 硬门槛做同样的镜像检查（同 min_overlap 的模式：只在结果带这些字段时
        # 才检查，兼容不提供这些诊断的旧/其它求解路径）。
        min_absolute_ess = result.get("min_absolute_ess")
        min_absolute_ess_threshold = result.get("min_absolute_ess_threshold")
        if (
            min_absolute_ess is not None
            and min_absolute_ess_threshold is not None
            and min_absolute_ess < min_absolute_ess_threshold
        ):
            raise RuntimeError(
                f"{stage_label} 阶段最小绝对有效样本数 min_absolute_ess="
                f"{min_absolute_ess:.4g} 低于阈值 {min_absolute_ess_threshold:.4g}，拒绝标记为 "
                "completed。ESS ratio 达标不代表绝对样本数足够，请延长重叠最差窗口的采样。"
            )
        min_decorrelated_samples = result.get("min_decorrelated_samples")
        min_decorrelated_samples_threshold = result.get("min_decorrelated_samples_threshold")
        if (
            min_decorrelated_samples is not None
            and min_decorrelated_samples_threshold is not None
            and min_decorrelated_samples < min_decorrelated_samples_threshold
        ):
            raise RuntimeError(
                f"{stage_label} 阶段最少去相关样本数 min_decorrelated_samples="
                f"{min_decorrelated_samples} 低于阈值 {min_decorrelated_samples_threshold}，"
                "拒绝标记为 completed，请延长采样。"
            )
        max_endpoint_uncertainty_kJ_mol = result.get("max_endpoint_uncertainty_kJ_mol")
        max_endpoint_uncertainty_kJ_mol_threshold = result.get("max_endpoint_uncertainty_kJ_mol_threshold")
        if (
            max_endpoint_uncertainty_kJ_mol is not None
            and max_endpoint_uncertainty_kJ_mol_threshold is not None
            and (
                not np.isfinite(max_endpoint_uncertainty_kJ_mol)
                or max_endpoint_uncertainty_kJ_mol > max_endpoint_uncertainty_kJ_mol_threshold
            )
        ):
            raise RuntimeError(
                f"{stage_label} 阶段最大端点自由能差不确定度 "
                f"max_endpoint_uncertainty_kJ_mol={max_endpoint_uncertainty_kJ_mol:.4g} kJ/mol "
                f"高于阈值 {max_endpoint_uncertainty_kJ_mol_threshold:.4g} kJ/mol，拒绝标记为 "
                "completed，请延长采样或检查窗口重叠。"
            )

    def _assert_reusable_stage_cache_sane(
        self,
        stage_label: str,
        result: Dict,
    ) -> None:
        """Re-run the scientific gates before a completed stage is reused.

        Stage checkpoints keep detailed gate values inside ``diagnostics`` to
        preserve the public result shape.  Rehydrate those values temporarily
        at the top level because ``_assert_stage_result_sane`` is also used on
        fresh in-memory solver results and therefore reads the top-level form.
        """
        if result.get("converged") is not True:
            raise RuntimeError(
                f"{stage_label} 缓存缺少明确的 converged=True 证据，拒绝复用。"
            )
        if result.get("stage") == "vanishing" and not isinstance(
            result.get("coverage_diagnostics"), dict
        ):
            raise RuntimeError(
                f"{stage_label} 缓存缺少 coverage_diagnostics，拒绝复用。"
            )
        diagnostics = result.get("diagnostics")
        if not isinstance(diagnostics, dict):
            raise RuntimeError(f"{stage_label} 缓存 diagnostics 非法，拒绝复用。")
        for key in (
            "min_overlap",
            "min_overlap_threshold",
            "min_absolute_ess",
            "min_absolute_ess_threshold",
            "min_decorrelated_samples",
            "min_decorrelated_samples_threshold",
            "max_endpoint_uncertainty_kJ_mol",
            "max_endpoint_uncertainty_kJ_mol_threshold",
            "window_overlap_diagnostics",
        ):
            if key not in result and key in diagnostics:
                result[key] = diagnostics[key]
        self._assert_stage_result_sane(stage_label, result)

    @staticmethod
    def _is_overlap_failure(result: Dict) -> bool:
        """结果没通过 sanity check 的原因是不是"重叠不足"（而不是 NaN/求解失败）——
        只有这种失败才是 refine_stage_lambda_path_by_overlap 有能力自动修的。"""
        min_overlap = result.get("min_overlap")
        min_overlap_threshold = result.get("min_overlap_threshold")
        dg = result.get("total_delta_G")
        err = result.get("total_error")
        if dg is None or not np.isfinite(dg):
            return False
        if err is None or not np.isfinite(err):
            return False
        if (
            min_overlap is None
            or min_overlap_threshold is None
            or not np.isfinite(min_overlap)
            or not np.isfinite(min_overlap_threshold)
            or min_overlap >= min_overlap_threshold
        ):
            return False

        # converged=False alone is not an overlap failure: MBAR solver failure,
        # missing windows, NaN, and warmup failure must keep their own failure
        # types.  Auto-insertion is allowed only when every reported window has
        # the ESS detail needed to locate a concrete lambda bottleneck.
        diagnostics = result.get("window_overlap_diagnostics")
        if not isinstance(diagnostics, list) or not diagnostics:
            return False
        for record in diagnostics:
            if not isinstance(record, dict):
                return False
            ratio = record.get("min_ess_ratio")
            per_lambda = record.get("ess_ratio_per_lambda")
            lambdas = record.get("lambdas")
            if (
                ratio is None
                or not np.isfinite(ratio)
                or not isinstance(per_lambda, dict)
                or not per_lambda
                or not isinstance(lambdas, list)
                or len(lambdas) < 2
            ):
                return False
        return True

    @staticmethod
    def _window_lambda_key(lambdas: List[float], lo: int, hi: int) -> Tuple[float, ...]:
        """一个窗口的内容指纹：它包含的 λ 值（四舍五入避免浮点噪声导致误判不等）。
        跟 window_idx 无关——路径变了之后，同一段 λ 完全可能挪到不同的 idx 上，
        这个指纹才是判断"是不是同一个窗口"唯一可信的依据。原是
        `_invalidate_stage_window_files` 内部的私有嵌套函数，提出来给
        `_remap_window_by_lambda_content` 共用，避免两处各自维护一份同样的逻辑。
        """
        return tuple(round(float(x), 8) for x in lambdas[lo:hi])

    def _remap_window_by_lambda_content(
        self,
        old_range: Tuple[int, int],
        old_lambdas: List[float],
        new_lambdas: List[float],
        new_ranges: List[Tuple[int, int]],
    ) -> Optional[Tuple[int, int]]:
        """给定一个窗口在旧方案里的 (start,end)，按它包含的 λ 值集合在新方案
        new_ranges 里找到内容完全一致的新 (start,end)——不依赖 window_idx（拆窗/
        插 λ 之后会整体重排）。找不到（比如被 canonicalize_window_ranges 归约掉、
        或被相邻拆窗的邻窗重排吞并）时返回 None，调用方必须把这种情况当成"这个
        窗口本轮暂时定位不到"来处理（跳过/推迟），不能假装还能找到一个近似匹配。
        """
        target_key = self._window_lambda_key(old_lambdas, old_range[0], old_range[1])
        for (ns, ne) in new_ranges:
            if self._window_lambda_key(new_lambdas, ns, ne) == target_key:
                return (ns, ne)
        return None

    def _apply_already_good_repairs(
        self,
        stage_name: str,
        stage_type: str,
        stage_label: str,
        threshold: float,
        repair_round: int,
        entries: List[Tuple[Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]],
        probe_results: Dict[Tuple[int, int], List[Dict]],
        pending_step_overrides: Dict[int, int],
        n_steps_per_window: Optional[int],
        resample_step_growth_factor: float,
        max_resample_step_multiplier: float,
    ) -> List[Dict]:
        """对一批 fixed-H 全通过但 production ESS 低的窗口跑
        `_diagnose_and_repair_all_pass_low_ess_window` 并落盘/打印结果。

        `entries` 是 `(probe_key, lookup_range, lookup_ranges)` 三元组列表：
        `probe_key` 用于取 `probe_results[probe_key]`（探针结果永远按探测时的
        原始 λ 内容为键，不随窗口重排变化）；`lookup_range`/`lookup_ranges` 是
        实际要传给 `_diagnose_and_repair_all_pass_low_ess_window` 的窗口范围/
        全量范围列表——路径本轮没变时就是原始 se/`effective_old_ranges`，路径
        本轮变了就必须是重映射之后的新 (start,end)/`new_ranges`（否则函数内部
        `new_ranges.index(...)` 式的 window_idx 推导会找错窗口，或找不到）。
        两条调用路径（未变路径 vs 已变路径重映射后）共用这同一段诊断/落盘/
        打印逻辑，避免维护两份几乎一样的代码。
        """
        repair_actions = []
        for probe_key, lookup_range, lookup_ranges in entries:
            action = self._diagnose_and_repair_all_pass_low_ess_window(
                stage_name, stage_type, lookup_range, probe_results[probe_key], lookup_ranges,
            )
            repair_actions.append(action)
        if not repair_actions:
            return repair_actions
        repair_file = self._persist_sampling_repair_actions(stage_name, repair_round, repair_actions)
        for action in repair_actions:
            se = tuple(action["window_range"])
            if action["decision"] == "recalibrate_f_k":
                bad_edges = [i for i, m in enumerate(action["mismatched_edges"]) if m]
                self._log(
                    f"  🎯 {stage_label}: 窗口 {se} fixed-H 全通过但 production ESS 低于阈值 "
                    f"{threshold:.4g}；相邻边 {bad_edges} 的生产冻结 f_k 增量与 fixed-H "
                    f"BAR/MBAR ΔF 差异超出各自噪声阈值（max|Δ|="
                    f"{action['max_abs_edge_diff_kJ_mol']:.3f} kJ/mol，阈值="
                    f"max({action['f_k_edge_mismatch_floor_kJ_mol']:.2f}, "
                    f"{action['f_k_edge_mismatch_sigma_multiplier']:.1f}×σ_probe)）——判定 "
                    "warmup 学到的偏置不准。已用 fixed-H 校准的 f_k 覆盖该窗口的 ibs_state "
                    "缓存（bias_converged=True），只清空该窗口自己的 production 产物，下一轮"
                    "只对它重新做冻结 burn-in + 只读验证 + 生产重采样，不影响其它窗口。"
                )
            elif action["decision"] == "reseed_resample":
                window_idx = action["window_idx"]
                if n_steps_per_window:
                    # 🔑 真正延长，而不是原地打转：每次这个窗口触发
                    # reseed_resample，把它下一轮的生产步数在当前覆盖值
                    # （首次为默认 n_steps_per_window）基础上乘以
                    # resample_step_growth_factor，封顶
                    # max_resample_step_multiplier 倍默认步数，避免
                    # 一个持续不收敛的窗口无界烧 GPU。
                    current_steps = pending_step_overrides.get(window_idx, int(n_steps_per_window))
                    current_multiplier = current_steps / float(n_steps_per_window)
                    new_multiplier = min(
                        current_multiplier * resample_step_growth_factor,
                        max_resample_step_multiplier,
                    )
                    new_steps = int(round(n_steps_per_window * new_multiplier))
                    pending_step_overrides[window_idx] = new_steps
                    self._log(
                        f"  🔁 {stage_label}: 窗口 {se} fixed-H 全通过但 production ESS 低于阈值 "
                        f"{threshold:.4g}；每条相邻边的生产冻结 f_k 增量与 fixed-H BAR/MBAR ΔF 都"
                        f"在各自噪声阈值内一致（max|Δ|={action['max_abs_edge_diff_kJ_mol']:.3f} "
                        "kJ/mol）——偏置本身没问题，更可能是构象弛豫慢/采样太短；只清空该窗口"
                        f"自己的 production 产物，下一轮用 {new_steps} 步（默认的 "
                        f"{new_multiplier:.2f}×）真正延长该窗口的生产采样，不影响其它窗口。"
                    )
                else:
                    self._log(
                        f"  🔁 {stage_label}: 窗口 {se} fixed-H 全通过但 production ESS 低于阈值 "
                        f"{threshold:.4g}；每条相邻边的生产冻结 f_k 增量与 fixed-H BAR/MBAR ΔF 都"
                        f"在各自噪声阈值内一致（max|Δ|={action['max_abs_edge_diff_kJ_mol']:.3f} "
                        "kJ/mol）——偏置本身没问题，更可能是构象弛豫慢/采样太短；但本次调用未"
                        "提供 n_steps_per_window 基准，无法计算延长后的步数，只能清空该窗口的 "
                        "production 产物，下一轮用默认步数重采一批独立样本（不是延长，只是换"
                        "一批同样长度的样本），不影响其它窗口。"
                    )
            else:
                self._log(
                    f"  ⚠️ {stage_label}: 窗口 {se} fixed-H 全通过但 production ESS 低于阈值 "
                    f"{threshold:.4g}，且无法自动诊断/修复（{action.get('note')}）——详情见 "
                    f"{repair_file}，需要人工检查；不插 λ、不拆窗，production 结果保留不动。"
                )
        return repair_actions

    def _invalidate_stage_window_files(
        self,
        stage_name: str,
        stage_type: str,
        old_lambdas: Optional[List[float]] = None,
        old_ranges: Optional[List[Tuple[int, int]]] = None,
        new_lambdas: Optional[List[float]] = None,
        new_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        """λ 路径/窗口边界被自动修复改变后，旧的每窗口产物（能量/偏置/基准能量、
        收敛诊断、IBS 断点状态）按新方案的窗口编号可能对不上号——window_idx 相同
        不代表覆盖同一段 λ，必须显式清掉，不能依赖隐式的形状校验（ibs_engine.py
        run_all_windows 的 resume 判断现在会额外校验 convergence.json 里记的真实
        λ 值，见该文件改动，但这里如果留着"位置对得上、λ 对不上"的旧文件，至少也
        是白占磁盘、且容易在别处被误读）。

        但"λ 路径变了"不代表*每个*窗口都变了：插值加密通常只让重叠不足的那一段
        λ 区间受影响，其余窗口在新方案里往往覆盖的还是完全相同的一段 λ（可能挪到
        了不同的 window_idx）。如果调用方提供了 old_lambdas/old_ranges/new_lambdas/
        new_ranges，就按"这个窗口包含的 λ 值集合"逐一比对旧窗口与新窗口——集合
        完全相同的直接重命名旧产物到新 window_idx（不重新采样），只有真正内容变了
        的窗口才清掉、留给重新采样。不提供这些参数时退化为原来的"全部清空"行为。
        """
        stage_dir = os.path.join(self.output_dir, stage_name)
        _window_key = self._window_lambda_key

        def _paths_for(idx: int) -> List[str]:
            return [
                os.path.join(stage_dir, f"dual_window_{idx}_{stage_type}_energies.npy"),
                os.path.join(stage_dir, f"dual_window_{idx}_{stage_type}_bias.npy"),
                os.path.join(stage_dir, f"dual_window_{idx}_{stage_type}_base.npy"),
                os.path.join(stage_dir, f"dual_window_{idx}_{stage_type}_convergence.json"),
                os.path.join(self.checkpoint_dir, f"ibs_state_{stage_type}_window_{idx}.json"),
            ]

        def _old_window_accounting_ok(old_idx: int) -> bool:
            """旧窗口的 base/bias 力组切分口径（WCA_ACCOUNTING_VERSION）、IBS 偏置
            预热/冻结协议（IBS_BIAS_PROTOCOL_VERSION）和 LJ 长程尾项修正公式版本
            （TRADITIONAL_LJ_LRC_PROTOCOL_VERSION，均见 ibs_engine.py）是否都跟
            当前一致——λ 集合再怎么完全相同，任一口径变了就是不同/不可信的数值，
            绝不能复用。读不到 convergence.json 或缺字段的一律保守地判"不一致"。"""
            conv_path = os.path.join(stage_dir, f"dual_window_{old_idx}_{stage_type}_convergence.json")
            try:
                with open(conv_path, "r", encoding="utf-8") as f:
                    conv = json.load(f)
            except Exception:
                return False
            return (
                conv.get("wca_accounting_version") == WCA_ACCOUNTING_VERSION
                and conv.get("ibs_bias_protocol_version") == IBS_BIAS_PROTOCOL_VERSION
                and conv.get("lj_tail_lrc_protocol_version") == TRADITIONAL_LJ_LRC_PROTOCOL_VERSION
                # 🔑 [non_mutating_v1] 旧变异策略缓存（f_k 可能被就地重校准过）不得复用。
                and conv.get("sampling_repair_policy") == "non_mutating_v1"
            )

        reuse_map: Dict[int, int] = {}  # new_idx -> old_idx，λ 集合 + 记账口径完全一致
        if old_lambdas is not None and old_ranges is not None and new_lambdas is not None and new_ranges is not None:
            old_keys = {
                old_idx: _window_key(old_lambdas, lo, hi)
                for old_idx, (lo, hi) in enumerate(old_ranges)
                if _old_window_accounting_ok(old_idx)
            }
            new_keys = {new_idx: _window_key(new_lambdas, lo, hi) for new_idx, (lo, hi) in enumerate(new_ranges)}
            used_old = set()
            for new_idx, key in new_keys.items():
                for old_idx, okey in old_keys.items():
                    if old_idx in used_old:
                        continue
                    if okey == key:
                        reuse_map[new_idx] = old_idx
                        used_old.add(old_idx)
                        break

        # 第一阶段：把要复用的旧窗口产物先挪到临时文件名，避免 new_idx/old_idx
        # 数字发生交换（比如新窗口1要用旧窗口3的数据，同时新窗口3要用旧窗口1的
        # 数据）时互相覆盖。
        pending_moves = []  # (tmp_path, final_path, new_idx)
        for new_idx, old_idx in reuse_map.items():
            if old_idx == new_idx:
                continue
            for old_path, new_path in zip(_paths_for(old_idx), _paths_for(new_idx)):
                if os.path.exists(old_path):
                    tmp_path = old_path + ".reuse_tmp"
                    os.replace(old_path, tmp_path)
                    pending_moves.append((tmp_path, new_path, new_idx))

        for tmp_path, final_path, new_idx in pending_moves:
            os.replace(tmp_path, final_path)
            if final_path.endswith("_convergence.json"):
                try:
                    with open(final_path, "r", encoding="utf-8") as f:
                        conv = json.load(f)
                    conv["window_idx"] = int(new_idx)
                    with open(final_path, "w", encoding="utf-8") as f:
                        json.dump(conv, f, indent=2)
                except Exception:
                    pass  # 元数据字段更新失败不影响正确性，文件名/内容才是真正依据

        reused_new_idx = set(reuse_map.keys())

        # 第二阶段：清掉所有不在"本轮被复用"名单里的窗口产物——包括真正内容变了
        # 的窗口，以及没被任何新窗口认领的旧编号残留。
        import re as _re
        removed = []
        patterns_with_regex = (
            (os.path.join(stage_dir, f"dual_window_*_{stage_type}_energies.npy"), rf"dual_window_(\d+)_{stage_type}_energies\.npy$"),
            (os.path.join(stage_dir, f"dual_window_*_{stage_type}_bias.npy"), rf"dual_window_(\d+)_{stage_type}_bias\.npy$"),
            (os.path.join(stage_dir, f"dual_window_*_{stage_type}_base.npy"), rf"dual_window_(\d+)_{stage_type}_base\.npy$"),
            (os.path.join(stage_dir, f"dual_window_*_{stage_type}_convergence.json"), rf"dual_window_(\d+)_{stage_type}_convergence\.json$"),
            (os.path.join(self.checkpoint_dir, f"ibs_state_{stage_type}_window_*.json"), rf"ibs_state_{stage_type}_window_(\d+)\.json$"),
        )
        for glob_pattern, regex in patterns_with_regex:
            for path in glob.glob(glob_pattern):
                m = _re.search(regex, path)
                idx = int(m.group(1)) if m else None
                if idx is not None and idx in reused_new_idx:
                    continue
                os.remove(path)
                removed.append(path)

        if reused_new_idx:
            self._log(
                f"  ♻️  λ 路径已变更，其中 {len(reused_new_idx)} 个窗口覆盖的 λ 集合与之前完全一致，"
                "直接复用旧产物（未重新采样）"
            )
        if removed:
            self._log(f"  🧹 λ 路径已变更，清理 {len(removed)} 个受影响的旧窗口产物，强制重新采样")

    def _load_window_bias_warmup_status(
        self,
        stage_name: str,
        stage_type: str,
        window_ranges: List[Tuple[int, int]],
        window_range: Tuple[int, int],
    ) -> Optional[str]:
        """Read one window's own bias_warmup.status from its convergence.json.

        The on-disk window index is its position in ``window_ranges`` -- that
        list IS exactly ``self.ranges`` as passed to the manager for this run
        (``effective_old_ranges`` at the call site), so this matches the same
        enumeration ``run_all_windows`` used when naming
        ``dual_window_{idx}_{stage_type}_convergence.json``. Returns None if
        the window isn't found, the file is missing, or it can't be parsed --
        callers must treat None as "not confirmed", not as a pass.
        """
        try:
            window_idx = window_ranges.index(tuple(window_range))
        except ValueError:
            return None
        conv_path = os.path.join(
            self.output_dir, stage_name, f"dual_window_{window_idx}_{stage_type}_convergence.json"
        )
        try:
            with open(conv_path, "r", encoding="utf-8") as f:
                conv = json.load(f)
            return conv.get("bias_warmup", {}).get("status")
        except Exception:
            return None

    @staticmethod
    def _merge_overlapping_ranges_into_components(
        ranges: List[Tuple[int, int]],
    ) -> List[List[Tuple[int, int]]]:
        """Group ranges into connected components by state-index overlap.

        Two windows are "connected" here iff their global lambda-index spans
        overlap -- true by construction for any two IBS-adjacent windows,
        since neighboring windows always share states. Windows failing in one
        contiguous stretch of the path form one component; windows failing in
        separate, non-adjacent stretches (with passing windows in between)
        form separate components.
        """
        ordered = sorted(tuple(r) for r in ranges)
        components: List[List[Tuple[int, int]]] = []
        current: List[Tuple[int, int]] = []
        current_end = None
        for (s, e) in ordered:
            if current and s >= current_end:
                components.append(current)
                current = []
                current_end = None
            current.append((s, e))
            current_end = e if current_end is None else max(current_end, e)
        if current:
            components.append(current)
        return components

    def _fixed_h_probe_file(self, stage_name: str) -> str:
        return os.path.join(self.output_dir, stage_name, "production_fixed_h_overlap.json")

    def _fixed_h_probe_fingerprint(
        self,
        protocol_key: Optional[Dict],
        window_range: Tuple[int, int],
        current_lambdas: List[float],
        threshold: float = 0.03,
    ) -> str:
        """Content-based cache key for one window's fixed-H overlap probe.

        Keyed on the window's actual lambda_vdw values (not its (start,end)
        global-state indices, which shift whenever a lambda is inserted
        elsewhere in the path) plus the stage protocol fingerprint and the
        probe threshold, so a cached result is only reused when the physical
        Hamiltonian and window content are unchanged -- including across a
        fresh process (resume), where recomputing every probe from scratch
        would otherwise burn the same expensive burn-in + sampling per edge
        again for no reason.
        """
        start, end = window_range
        payload = {
            "protocol_key": protocol_key,
            "lambda_vdw_window": [round(float(x), 10) for x in current_lambdas[start:end]],
            "probe_threshold": float(threshold),
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def _load_fixed_h_probe_cache(self, stage_name: str) -> Dict[str, Dict]:
        probe_file = self._fixed_h_probe_file(stage_name)
        if not os.path.exists(probe_file):
            return {}
        try:
            with open(probe_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("windows", {})
        except Exception:
            return {}

    def _persist_fixed_h_probe_edge(
        self,
        stage_name: str,
        stage_type: str,
        attempt: int,
        fingerprint: str,
        window_range: Tuple[int, int],
        pairs_so_far: List[Dict],
        complete: bool,
    ) -> str:
        """Write one window's fixed-H probe entry to disk immediately after
        each edge finishes (called as the per-edge callback from
        ``_probe_vdw_window_fixed_overlap``), keyed by the content fingerprint
        computed in ``_fixed_h_probe_fingerprint`` so a later resume/round can
        find and reuse it.

        Each edge burns real GPU time (independent NVT burn-in + sampling);
        previously all edges for a window only lived in a local list held by
        the caller, so a crash on (say) the last edge of a multi-edge window
        threw away every earlier edge's result too. Writing after every edge
        means a crash never loses more than the one in-flight edge.
        """
        probe_file = self._fixed_h_probe_file(stage_name)
        os.makedirs(os.path.dirname(probe_file), exist_ok=True)
        payload = {"stage_name": stage_name, "stage_type": stage_type, "windows": {}}
        if os.path.exists(probe_file):
            try:
                with open(probe_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                pass
        windows = payload.setdefault("windows", {})
        entry = windows.setdefault(fingerprint, {
            "window_range": [int(window_range[0]), int(window_range[1])],
            "pairs": [],
            "all_passed": None,
            "complete": False,
            "first_round": int(attempt) + 1,
            "rounds_seen": [],
        })
        entry["window_range"] = [int(window_range[0]), int(window_range[1])]
        entry["pairs"] = pairs_so_far
        entry["complete"] = bool(complete)
        if complete:
            entry["all_passed"] = bool(pairs_so_far) and all(p.get("passed") for p in pairs_so_far)
        if (int(attempt) + 1) not in entry["rounds_seen"]:
            entry["rounds_seen"].append(int(attempt) + 1)
        # 🔑 每条边都要落盘一次，中途被杀不能损坏整份 JSON——否则下一次加载
        # 失败后又会以空 payload 覆写，丢掉其它窗口已经跑完的昂贵探针结果。
        # 用临时文件+os.replace 保证要么是旧的完整版本，要么是新的完整版本。
        _atomic_write_json(probe_file, payload)
        return probe_file

    def _invalidate_single_window_production(
        self,
        stage_name: str,
        stage_type: str,
        window_idx: int,
        keep_ibs_state: bool = False,
        keep_production_data: bool = False,
    ) -> None:
        """Force exactly one window (by its current on-disk index) to
        resample, without touching any other window's cached production
        files, renumbering anything, or going through the λ-content
        reuse-mapping in ``_invalidate_stage_window_files`` (which is for
        whole-path λ/window changes, not a single targeted resample).

        Used by the fixed-H-all-pass / production-ESS-low sampling repair
        path below, which only ever targets one already-identified window
        index and must leave the rest of the stage's already-good production
        data completely alone.

        🔑 [reseed_resample 真续算修复] ``keep_production_data=True`` is for
        the ``reseed_resample`` decision specifically: that branch's whole
        premise is "the frozen f_k is confirmed correct (matches fixed-H
        BAR/MBAR ΔF within noise), production ESS is just low because
        sampling hasn't run long enough yet" -- the physical Hamiltonian and
        sampling distribution have NOT changed, so the already-collected
        ``energies``/``bias``/``base``/``convergence.json`` are still valid
        data from the SAME distribution, not stale data from a different one.
        Deleting them (the old behaviour, still the default here) meant every
        "extend the step budget" repair actually threw away everything
        already sampled and resampled an independent batch from scratch at
        the new (larger) budget -- not a real extension, just a bigger
        from-scratch resample every time. With ``keep_production_data=True``,
        these files are left alone; ``run_all_windows``'s production-entry
        checkpoint-continuation logic (see ``_build_production_window_checkpoint_manifest``/
        ``_production_window_checkpoint_is_usable`` in ``ibs_engine.py``) then
        detects the still-present files + a still-valid, content-matching
        production checkpoint and genuinely continues the same trajectory
        instead of resampling. ``recalibrate_f_k`` must keep the default
        ``False`` -- there, the frozen f_k is being OVERWRITTEN because it was
        found wrong, so any production data sampled under the old (wrong) f_k
        really is from a different distribution and must be discarded; the
        production checkpoint is invalidated alongside it for the same reason
        (a checkpoint's saved state is a snapshot taken under the old f_k).
        """
        stage_dir = os.path.join(self.output_dir, stage_name)
        if not keep_production_data:
            for suffix in ("energies", "bias", "base"):
                p = os.path.join(stage_dir, f"dual_window_{window_idx}_{stage_type}_{suffix}.npy")
                if os.path.exists(p):
                    os.remove(p)
            conv_path = os.path.join(stage_dir, f"dual_window_{window_idx}_{stage_type}_convergence.json")
            if os.path.exists(conv_path):
                os.remove(conv_path)
            _invalidate_production_window_checkpoint(self.checkpoint_dir, stage_type, window_idx)
        if not keep_ibs_state:
            ibs_path = os.path.join(self.checkpoint_dir, f"ibs_state_{stage_type}_window_{window_idx}.json")
            if os.path.exists(ibs_path):
                os.remove(ibs_path)

    def _diagnose_and_repair_all_pass_low_ess_window(
        self,
        stage_name: str,
        stage_type: str,
        window_range: Tuple[int, int],
        pairs: List[Dict],
        effective_old_ranges: List[Tuple[int, int]],
        f_k_edge_mismatch_floor_kJ_mol: float = 1.0,
        f_k_edge_mismatch_sigma_multiplier: float = 2.0,
    ) -> Dict:
        """For one window whose fixed-H probe passed but production ESS is
        still low, decide whether the SGD-learned production f_k is actually
        wrong or the bias is fine and the problem is short sampling/slow
        relaxation, then trigger the corresponding repair. "fixed-H all
        passed" only means the λ grid has reached the minimum connectivity
        bar (threshold 0.03) -- it is not evidence the production f_k (learned
        under a different, possibly-still-oscillating SGD threshold of 0.05)
        is actually correct, so this must not be silently treated as "stage
        converged" while the final free energy still comes from low-ESS
        production data.

        Decision rule: compare PER-EDGE increments, not absolute (de-meaned)
        state values -- ``diff(production_f_k)[i]`` (the increment the
        production-frozen f_k implies for edge i) vs. that edge's own
        ``delta_f_bias_kJ_mol`` from ``pairs`` (the WCA-preserving, no-LRC
        bias-calibration sub-probe that ``_probe_vdw_window_fixed_overlap``
        now runs on every edge whose path-overlap probe already passed --
        NOT ``delta_f_kJ_mol``, which comes from the path-overlap probe
        itself: WCA-less dynamics, LRC-inclusive energy, i.e. a different
        ensemble/energy than what the production Group 1 bias CV's f_k
        actually needs to reproduce. Using ``delta_f_kJ_mol`` here used to be
        the same "wrong ensemble, wrong energy" bug the warmup-time
        calibration probe had [see IBS_BIAS_PROTOCOL_VERSION=10] -- it just
        surfaced during production-ESS repair instead of during warmup, and
        directly overwrote production f_k with values derived from it).
        Comparing diffs rather than absolute f_k makes the test invariant to
        the arbitrary additive constant either vector is defined up to,
        without needing to force a shared zero-point convention first. The
        mismatch threshold is per-edge and noise-aware: ``max(
        f_k_edge_mismatch_floor_kJ_mol, f_k_edge_mismatch_sigma_multiplier *
        delta_f_bias_uncertainty_kJ_mol)`` -- a flat floor alone (e.g. 1.0
        kJ/mol) is too tight when the probe itself is noisy (real fixed-H ΔF
        uncertainties observed here run ~2-3 kJ/mol/edge), which would make
        recalibration trigger on probe noise almost every time; scaling by
        the probe's own reported uncertainty keeps the comparison honest
        about how well the "independent ground truth" is actually known. Any
        edge exceeding its own threshold is enough to call the warmup bias
        unreliable.

        Every edge in ``pairs`` must have ``bias_calibration_sufficient is
        True`` (the calibration sub-probe met its decorrelated-sample-count
        and ΔF-uncertainty gates, extending sampling up to 3 times first) --
        if any edge is missing this or has it False/None, this function
        refuses to compare or recalibrate at all
        (``decision="skipped_insufficient_bias_calibration"``): with no
        trustworthy independent ΔF for at least one edge, neither branch of
        the decision (recalibrate vs. reseed) is safe to take automatically.

        If any edge mismatches: overwrite the cached ibs_state with the
        f_k implied by cumulating the probe's edge deltas (f_0=0, de-meaned
        for reporting only -- the decision itself never depended on this
        choice of zero-point), bias_converged=True, so the next repair round
        skips learning and goes straight to frozen burn-in + read-only
        validation on the better f_k, then invalidate only this window's
        production files so it is the only one resampled
        (``decision="recalibrate_f_k"``). If every edge agrees within its
        threshold, the bias is judged fine and the likely cause is slow
        relaxation/short sampling: keep the existing ibs_state as-is and
        call ``_invalidate_single_window_production`` with
        ``keep_production_data=True`` (``decision="reseed_resample"``).

        🔑 [production checkpoint 续采 fix] This USED to be a true reseed, not
        a longer run: the old code path deleted the existing production
        files unconditionally and ``_run_dual_lambda_stage``'s window
        ``Simulation``/``setVelocitiesToTemperature`` calls never pass an
        explicit seed, so a from-scratch resample drew an independent batch
        at the (possibly larger, via the caller's step-count growth factor)
        target budget -- not an extension, a discard-and-redo every time.
        That is fixed now: ``keep_production_data=True`` leaves
        ``energies``/``bias``/``base``/``convergence.json`` and the
        production-window OpenMM checkpoint in place (the physical
        Hamiltonian/sampling distribution is unchanged here, unlike
        ``recalibrate_f_k``), and ``run_all_windows``'s production-entry
        checkpoint-continuation logic (see
        ``_build_production_window_checkpoint_manifest``/
        ``_production_window_checkpoint_is_usable`` in ``ibs_engine.py``)
        detects the still-present, content-matching checkpoint and genuinely
        continues the same trajectory from where it left off, appending new
        samples instead of resampling independently from scratch.

        Caller contract [starvation fix]: this MAY be called in the same
        round as a split/insert, but only *after* ``_invalidate_stage_window_
        files`` has fully completed its rename/purge pass for that round --
        it depends on each window's ``convergence.json`` still existing at
        the position it runs at, and this function's own repair branches
        delete that same file, so running it first would get the just-
        recalibrated window caught in that same purge (misread as "unmatched
        leftover"). ``window_range``/``effective_old_ranges`` must be this
        window's CURRENT (post-split/insert) position/range list, resolved
        via ``_remap_window_by_lambda_content`` against its pre-round
        position if the path changed this round -- not blindly the pre-round
        tuple, which may no longer exist positionally. ``pairs`` (probe
        results) stays keyed by the window's pre-round identity regardless
        (probe results describe sampled lambda content, not a position).
        This function never slices any lambda array by position -- only
        ``effective_old_ranges.index(tuple(window_range))`` (to locate the
        on-disk ``ibs_state_*.json``) and a size check against ``pairs`` --
        so passing the post-remap range/ranges list here needs no other
        change on the caller's part.

        Never touches any window other than ``window_range``. Returns a dict
        describing what was decided/done, for the caller to log/persist.
        """
        start, end = window_range
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=10] fail closed：任何一条边的 bias 校准
        # 子探针没跑（None，path-overlap 本身没过）或跑了但延长采样 3 次后仍
        # 不够精确（False）都拒绝继续——两个分支（recalibrate_f_k/
        # reseed_resample）都需要对每条边的独立 ΔF 有信心，其中任何一条边缺失
        # 都会让下面的比较不完整、决策不可信。这不应该发生：本函数的调用点
        # 只在 all(p["passed"] for p in probe_results[se]) 之后才调用，也就是
        # 每条边的 path-overlap 都通过了，calibration 子探针理应都跑过；这里
        # 仍显式校验，防止调用契约将来被违反时静默用错误字段。
        insufficient = [
            i for i, p in enumerate(pairs) if p.get("bias_calibration_sufficient") is not True
        ]
        if insufficient:
            return {
                "window_range": [int(start), int(end)],
                "decision": "skipped_insufficient_bias_calibration",
                "note": (
                    f"边 {insufficient} 的 bias 校准子探针未达标（去相关样本数/ΔF"
                    "不确定度门槛，延长采样 3 次后仍不够，或 path-overlap 本身未通过），"
                    "拒绝用其 delta_f_bias_kJ_mol 做 f_k 比较/覆盖，跳过自动修复，需人工检查。"
                ),
            }
        delta_f_edges = np.asarray([float(p["delta_f_bias_kJ_mol"]) for p in pairs], dtype=np.float64)
        delta_f_sigmas = np.asarray([float(p["delta_f_bias_uncertainty_kJ_mol"]) for p in pairs], dtype=np.float64)
        # 🔑 fail closed，不静默传播：ibs_engine.py::_compute_bidirectional_overlap_from_u_kn
        # 已经对这个不确定度做了 isfinite/>=0 校验并 fail closed，这里是第二道防线
        # （防止将来有其它 probe 实现路径绕过那处校验）。若放过 NaN/负值，下面
        # edge_mismatch_thresholds = max(floor, sigma_multiplier * sigma) 会是 NaN，
        # Python 的 `>` 比较对 NaN 恒为 False，会让任何真实差异都被误判为"在噪声
        # 阈值内"、错误地把该 recalibrate_f_k 的窗口放行成 reseed_resample。
        if not np.all(np.isfinite(delta_f_sigmas)) or np.any(delta_f_sigmas < 0.0):
            return {
                "window_range": [int(start), int(end)],
                "decision": "skipped_invalid_probe_uncertainty",
                "note": (
                    f"fixed-H 校准探针返回的 delta_f_bias_uncertainty_kJ_mol 存在非有限或负值"
                    f"（{delta_f_sigmas.tolist()}），拒绝用它做噪声感知阈值比较，跳过自动"
                    "修复，需人工检查探针本身。"
                ),
            }
        f_calibrated = np.concatenate(([0.0], np.cumsum(delta_f_edges)))
        f_calibrated = f_calibrated - np.mean(f_calibrated)

        base_info = {
            "window_range": [int(start), int(end)],
            "calibrated_f_k_kJ_mol": f_calibrated.tolist(),
        }
        try:
            window_idx = effective_old_ranges.index(tuple(window_range))
        except ValueError:
            return {
                **base_info,
                "decision": "skipped_window_index_not_found",
                "note": "窗口范围在当前 effective_old_ranges 里找不到匹配的 window_idx，无法定位 ibs_state 缓存。",
            }

        ibs_state_path = os.path.join(
            self.checkpoint_dir, f"ibs_state_{stage_type}_window_{window_idx}.json"
        )
        cached_state = None
        production_f_k = None
        if os.path.exists(ibs_state_path):
            try:
                with open(ibs_state_path, "r", encoding="utf-8") as f:
                    cached_state = json.load(f)
                cached_f_k = cached_state.get("f_k")
                if cached_state.get("bias_converged") and cached_f_k is not None and len(cached_f_k) == (end - start):
                    production_f_k = np.asarray(cached_f_k, dtype=np.float64)
            except Exception:
                cached_state = None
                production_f_k = None

        if production_f_k is None:
            return {
                **base_info,
                "window_idx": int(window_idx),
                "decision": "skipped_missing_production_f_k",
                "note": (
                    f"无法从 {ibs_state_path} 读取该窗口生产冻结的 f_k"
                    "（文件缺失/不是 bias_converged 状态/态数不匹配），无法比较，跳过自动修复，需人工检查。"
                ),
            }

        production_edge_diffs = np.diff(production_f_k)
        edge_abs_diffs = np.abs(production_edge_diffs - delta_f_edges)
        edge_thresholds = np.maximum(
            float(f_k_edge_mismatch_floor_kJ_mol),
            float(f_k_edge_mismatch_sigma_multiplier) * delta_f_sigmas,
        )
        mismatched_edges = edge_abs_diffs > edge_thresholds
        decision = "recalibrate_f_k" if bool(np.any(mismatched_edges)) else "reseed_resample"

        if decision == "recalibrate_f_k":
            new_state = dict(cached_state)
            new_state["f_k"] = f_calibrated.tolist()
            new_state["bias_converged"] = True
            new_state["t"] = 0
            with open(ibs_state_path, "w", encoding="utf-8") as f:
                json.dump(new_state, f, indent=2)
            # 🔑 f_k 真的被覆盖了——旧 production 数据是在错误 f_k 下采的，
            # 不是同一个采样分布，必须丢弃（keep_production_data 默认 False）。
            self._invalidate_single_window_production(stage_name, stage_type, window_idx, keep_ibs_state=True)
        else:
            # 🔑 [reseed_resample 真续算修复] f_k 已确认没问题（跟 fixed-H
            # BAR/MBAR ΔF 在噪声阈值内一致），production ESS 低只是采样还不够
            # 长——物理 Hamiltonian/采样分布完全没变，不能再像以前一样把已经
            # 采到的样本整批扔掉重采。保留 energies/bias/base/convergence.json
            # 和生产 checkpoint，run_all_windows 的续算检测会用它们真正接着跑，
            # 而不是从头开始独立重采一批。
            self._invalidate_single_window_production(
                stage_name, stage_type, window_idx, keep_ibs_state=True, keep_production_data=True,
            )

        return {
            **base_info,
            "window_idx": int(window_idx),
            "decision": decision,
            "production_f_k_kJ_mol": production_f_k.tolist(),
            "production_edge_diffs_kJ_mol": production_edge_diffs.tolist(),
            "probe_delta_f_bias_edges_kJ_mol": delta_f_edges.tolist(),
            "probe_delta_f_bias_sigmas_kJ_mol": delta_f_sigmas.tolist(),
            "edge_abs_diffs_kJ_mol": edge_abs_diffs.tolist(),
            "edge_mismatch_thresholds_kJ_mol": edge_thresholds.tolist(),
            "mismatched_edges": mismatched_edges.tolist(),
            "max_abs_edge_diff_kJ_mol": float(np.max(edge_abs_diffs)),
            "f_k_edge_mismatch_floor_kJ_mol": float(f_k_edge_mismatch_floor_kJ_mol),
            "f_k_edge_mismatch_sigma_multiplier": float(f_k_edge_mismatch_sigma_multiplier),
        }

    def _persist_sampling_repair_actions(
        self, stage_name: str, attempt: int, actions: List[Dict],
    ) -> str:
        """Append this round's fixed-H-all-pass/low-ESS diagnosis+repair
        decisions to disk (recalibrate_f_k / reseed_resample / skipped_*, with
        the per-edge diffs/thresholds and calibrated vs. production f_k
        values that drove the decision), alongside the raw probe results, so
        the reasoning behind each repair is auditable without re-deriving it
        from ibs_state snapshots later.
        """
        repair_file = os.path.join(self.output_dir, stage_name, "sampling_repair_decisions.json")
        os.makedirs(os.path.dirname(repair_file), exist_ok=True)
        payload = {"stage_name": stage_name, "rounds": []}
        if os.path.exists(repair_file):
            try:
                with open(repair_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                pass
        payload.setdefault("rounds", []).append({"round": int(attempt) + 1, "actions": actions})
        # 🔑 同上（见 _persist_fixed_h_probe_edge）：原子写，避免写入中途被杀
        # 损坏整份 JSON、下一次加载失败后又以空 payload 覆写掉之前几轮的记录。
        _atomic_write_json(repair_file, payload)
        return repair_file

    def _probe_vdw_window_fixed_overlap(
        self,
        window_range: Tuple[int, int],
        lambdas_vdw_full: List[float],
        potential_type: str,
        dexp_params,
        boresch_params,
        threshold: float = 0.03,
        on_edge_done=None,
        resume_pairs: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Measure real fixed-Hamiltonian bidirectional overlap on every adjacent
        pair of an already-completed vdw window, for production-time ESS
        auto-repair (mirrors the warmup-time probe in ibs_engine.py's
        ``run_all_windows``, which cannot be reused directly here because by the
        time production ESS diagnostics exist, that window's live Simulation is
        long gone -- ``run_all_windows`` explicitly deletes it, ibs_engine.py
        ~3944-3951).

        Two-phase edge processing (see inline comments below for the full
        rationale): phase 1 runs every edge's cheap path-overlap probe first,
        with NO calibration attempted yet; only if *every* edge in the window
        passes path-overlap does phase 2 run the expensive bias-calibration
        probe per edge. Any single failing edge means the window is very
        likely about to be split/have a lambda inserted, so calibration on
        the edges that did pass would be wasted GPU time -- skip it entirely
        for the whole window rather than partially calibrating.

        ``on_edge_done``, if given, is called with each edge's result dict as
        soon as that edge's *current phase* result is ready (path result in
        phase 1, then again with the final calibration result in phase 2 if
        the window proceeds to phase 2) -- so it may be called twice for the
        same edge (identified by its ``global_edge`` field), once per phase.
        The caller must upsert by ``global_edge`` (replace an existing entry
        with the same ``global_edge``, not blindly append) to avoid
        duplicating that edge in its accumulated list -- see
        ``_run_stage_with_overlap_autorepair``'s ``_on_edge_done`` closure.
        Persisting after every phase-1 edge means a crash partway through the
        (cheap) path-probe loop only loses the one in-flight edge, not
        anything computed so far; a crash during phase 2 similarly only
        loses the one in-flight edge's calibration attempt, with every
        edge's phase-1 path result already safely on disk.

        ``resume_pairs``, if given, seeds the result with edges already
        computed in an earlier (possibly interrupted) call for this exact
        window content -- every edge is independent (all start from the same
        ``relaxed_positions``/``relaxed_box`` computed once below, regardless
        of edge order), so skipping already-cached leading edges and only
        computing the remaining ones is exactly equivalent to computing all
        of them in one pass. Without this, a crash after edge 5 of 9 would
        recompute all 9 edges from scratch on the next attempt even though
        the caller's fingerprint-keyed cache still has edges 0-4 on disk.
        Resumed pairs whose ``bias_calibration_sufficient`` is still ``None``
        (phase 1 only, from an interrupted earlier call, or because that
        earlier call's window failed path-overlap) are re-examined by phase 2
        exactly like freshly-computed phase-1 pairs -- if the window later
        turns out all-passed, they get calibrated; if not, they stay ``None``.

        No per-window final configuration is ever persisted to disk -- every
        window, including the original production run, starts from the same
        stage-wide ``self.positions``/``self.box_vectors`` and relaxes
        internally (confirmed: the resume path rebuilds from the same stage-wide
        input too, never from a saved per-window frame). So this rebuilds the
        window via the identical ``IBSWindowManagerDualLambda`` construction
        ``_run_dual_lambda_stage`` uses for the vdw stage (same kwargs, same
        ``alchemical_params``/``potential_type``/``restraint_params``), which
        guarantees the same Hamiltonian rather than an accidentally-different
        one, then does its own short minimization before handing frames to
        ``probe_bidirectional_overlap``.
        """
        start, end = window_range
        lc_win = [0.0] * (end - start)
        lv_win = list(lambdas_vdw_full[start:end])
        alchemical_params = _resolve_alchemical_params(
            potential_type, dexp_params, self.ligand_indices
        )
        stage_output_dir = os.path.join(self.output_dir, "vanishing")
        manager = IBSWindowManagerDualLambda(
            system_template=self.system,
            topology=self.topology,
            perturbed_atom_indices=self.ligand_indices,
            lambdas_coul=[0.0] * len(lambdas_vdw_full),
            lambdas_vdw=list(lambdas_vdw_full),
            temperature=self.temperature,
            window_ranges=[window_range],
            alchemical_params=alchemical_params,
            potential_type=potential_type,
            restraint_params=boresch_params,
            prefix="abfe_dual",
            platform_name=self.platform_name,
            output_dir=stage_output_dir,
            checkpoint_dir=self.checkpoint_dir,
        )
        resolved_box = _resolve_periodic_box_vectors(
            self.box_vectors, topology=self.topology, system=self.system
        )
        win_sys, ibs_wrap = manager._build_window_system(
            lc_win, lv_win, resolved_box, self.positions
        )

        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)
        integrator = openmm.LangevinMiddleIntegrator(
            self.temperature, 1.0 / unit.picosecond, 0.001 * unit.picoseconds
        )
        relax_sim = app.Simulation(self.topology, win_sys, integrator, platform, props)
        relax_sim.context.setPeriodicBoxVectors(*resolved_box)
        relax_sim.context.setPositions(self.positions)
        # 🔑 [live crash fix] lambda_shield 必须在最小化*之前*就同步成这个窗口
        # 校准探针实际会用的 mean(lv_win)——否则 Group 4 WCA 力的
        # addGlobalParameter 默认值 0.0 会让最小化在"防护壳完全关闭"的势能面上
        # 进行，允许溶剂原子逼近到防护壳生效距离以内；随后
        # probe_bidirectional_overlap_for_bias_calibration 用同一份
        # relaxed_positions 但把 lambda_shield 显式设成 mean(lv_win)（非零），
        # 防护壳在动力学开始的第一步就从 0 跳到全强度，对着一个从未在该力下
        # 弛豫过的构型——真实运行中在窗口 (5,9) 边 [7,8] 上正是这样炸成
        # "Particle coordinate is NaN"。run_all_windows 里对应的活体 warmup
        # 路径（ibs_engine.py ~4918-4921）在最小化前就做了这个同步，这里必须
        # 保持一致。
        lam_vdw_shield_for_calibration = float(np.mean(lv_win))
        if _system_has_global_parameter(win_sys, "lambda_shield"):
            relax_sim.context.setParameter("lambda_shield", lam_vdw_shield_for_calibration)
        # The stage-wide starting configuration was equilibrated under the fully
        # interacting force field, not this window's (possibly heavily
        # decoupled) lambda -- unlike the warmup-time probe, which reuses an
        # already-relaxed live configuration, this one must minimize first or
        # the fixed-H burn-in below can start from a clashing configuration.
        relax_sim.minimizeEnergy(maxIterations=2000)
        relaxed_state = relax_sim.context.getState(getPositions=True)
        relaxed_positions = relaxed_state.getPositions()
        relaxed_box = relaxed_state.getPeriodicBoxVectors()
        del relax_sim, integrator

        n_edges = len(lv_win) - 1
        # Truncate defensively if a corrupted/stale cache entry somehow has
        # more entries than this window actually has edges.
        pairs = list(resume_pairs)[:n_edges] if resume_pairs else []
        resume_count = len(pairs)

        # =====================================================================
        # 阶段一：先把这个窗口全部 path edge 的双向 overlap 探针跑完（不碰任何
        # bias calibration）。理由：只要窗口里*任意*一条边 path-overlap 没过，
        # 这个窗口大概率马上要被拆分/插 λ，此时花在其它已经通过的边上的
        # 校准探针（burn-in + 最多 3 次翻倍重试的采样，比 path-overlap 贵得多）
        # 就白烧了——先把全部边的 path 结果收集齐，既给下游拆窗/插 λ 的诊断
        # 提供完整信息，又避免在窗口即将改变形状时提前投入昂贵的校准算力。
        # 每条边算完仍然立刻通过 on_edge_done 落盘（crash 只丢这一条 path
        # 探针，不丢calibration，因为 calibration 这一阶段还没开始）。
        # =====================================================================
        for local_i in range(resume_count, n_edges):
            pair = probe_bidirectional_overlap(
                topology=self.topology,
                common_system_xml=ibs_wrap._common_system_xml,
                ibs_wrapper=ibs_wrap,
                state_i=local_i,
                state_j=local_i + 1,
                positions=relaxed_positions,
                box_vectors=relaxed_box,
                temperature=self.temperature,
                platform_name=self.platform_name,
                threshold=threshold,
            )
            pair["global_edge"] = [int(start + local_i), int(start + local_i + 1)]
            # 占位：这一阶段还不知道窗口是否会进入 calibration，先标 None
            # （跟"path 本身没过、calibration 无意义"用的是同一个值，但含义
            # 不同——None 在这里只表示"calibration 阶段尚未处理"，阶段二
            # 会按需要改写成 True/False，或者在 all_passed=False 时保持
            # None 作为最终值）。
            pair["bias_calibration_sufficient"] = None
            pairs.append(pair)
            if on_edge_done is not None:
                on_edge_done(pair)

        # 🔑 只有全部 path edge 都通过，才进入阶段二的 bias calibration；任意
        # 一条边失败，立刻返回，完全跳过整个窗口的 calibration。
        all_path_edges_passed = bool(pairs) and all(p.get("passed") for p in pairs)
        if not all_path_edges_passed:
            return pairs

        # =====================================================================
        # 阶段二：全部 path edge 都通过后，才对每条边做 bias calibration。
        # 跳过已经有真实校准结果（True/False，来自更早一轮已完整跑过 calibration
        # 的缓存）的边，只对仍是 None 占位的边补算——这样 resume 时不会重复烧
        # 已经算好的 calibration。
        # =====================================================================
        for local_i in range(n_edges):
            pair = pairs[local_i]
            if pair.get("bias_calibration_sufficient") is not None:
                continue
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=10] pair["delta_f_kJ_mol"] 上面来自
            # probe_bidirectional_overlap：WCA-less 动力学 + LRC-inclusive 能量——
            # 这正是校准 f_k 时用错的 ensemble/能量定义（见该函数与
            # probe_bidirectional_overlap_for_bias_calibration 的 docstring）。
            # _diagnose_and_repair_all_pass_low_ess_window 会用这个窗口每条边
            # 的探针 ΔF 去判断是否要覆盖生产冻结的 f_k，这跟 warmup 阶段校准
            # f_k 是同一件事、同一个正确性要求，必须用同一个 WCA-preserving、
            # 不含 LRC 的校准探针，不能继续用 path-overlap 探针的 delta_f_kJ_mol。
            # 不达标（去相关样本数/ΔF 不确定度）就延长采样重试，最多 3 次
            # （每次采样步数翻倍），仍不达标就把 bias_calibration_sufficient
            # 标为 False，交给调用方决定如何处理，绝不用不够精确的估计去
            # 覆盖 f_k。
            calib_pair = None
            attempt_sample_steps = 20000
            for _extend_attempt in range(3):
                calib_pair = probe_bidirectional_overlap_for_bias_calibration(
                    topology=self.topology,
                    common_plus_wca_system_xml=ibs_wrap._common_plus_wca_system_xml,
                    ibs_wrapper=ibs_wrap,
                    state_i=local_i,
                    state_j=local_i + 1,
                    positions=relaxed_positions,
                    box_vectors=relaxed_box,
                    temperature=self.temperature,
                    platform_name=self.platform_name,
                    lambda_shield=lam_vdw_shield_for_calibration,
                    sample_steps=attempt_sample_steps,
                )
                if calib_pair is not None:
                    break
                attempt_sample_steps *= 2
            if calib_pair is not None:
                pair["delta_f_bias_kJ_mol"] = calib_pair["delta_f_bias_kJ_mol"]
                pair["delta_f_bias_uncertainty_kJ_mol"] = calib_pair["delta_f_bias_uncertainty_kJ_mol"]
                pair["bias_calibration_sufficient"] = True
                pair["bias_calibration_lambda_shield"] = lam_vdw_shield_for_calibration
                pair["bias_calibration_n_k_decorrelated"] = calib_pair.get("n_k_decorrelated")
            else:
                pair["bias_calibration_sufficient"] = False
                pair["bias_calibration_lambda_shield"] = lam_vdw_shield_for_calibration
            if on_edge_done is not None:
                on_edge_done(pair)
        return pairs

    def _run_stage_with_overlap_autorepair(
        self,
        stage_label: str,
        stage_name: str,
        preopt_file: str,
        n_states: int,
        lambdas_var: List[float],
        window_ranges: Optional[List[Tuple[int, int]]],
        run_once,
        protocol_key: Optional[Dict] = None,
        max_repair_rounds: int = 5,
        probe_window_overlap_fn=None,
        n_steps_per_window: Optional[int] = None,
        preopt_protocol_key: Optional[Dict] = None,
    ) -> Tuple[Dict, int, List[float], Optional[List[Tuple[int, int]]]]:
        """运行一个双 lambda 阶段；如果失败原因是 ESS 重叠不足，就用这次采样
        自己算出的 window_overlap_diagnostics（而不是任何手填的数字）自动在
        重叠最差的 λ 区间插点、重新划分窗口、清掉旧窗口产物，然后重新采样，
        最多重复 max_repair_rounds 轮。

        probe_window_overlap_fn: Optional[Callable[[Tuple[int,int], List[float], Optional[Callable]], List[Dict]]]。
        只有 vdw（"vanishing"）阶段的调用方会传入这个闭包（见 run_full_pipeline
        里的 _probe_stage2_window_overlap，它闭包住了跟 run_once 完全相同的
        potential_type/dexp_params/boresch_params，保证探针重建的是同一个
        Hamiltonian）。第二个参数是调用时刻的 current_lambdas（每轮修复都可能
        变，必须显式传，不能让闭包捕获修复循环开始前的旧值）。第三个可选参数
        是逐边完成回调（on_edge_done），本方法用它在每条边算完后立即落盘，不
        依赖窗口整体算完才持久化。传入后，
        production ESS 低的分支改用「先拆窗、最小窗口才做 fixed-H 双向 overlap
        探针、探针证实真有缺口才插 λ」，跟 warmup 失败分支统一；不传（目前是
        coul/"decharging" 阶段）则维持 refine_stage_lambda_path_by_overlap 的旧
        行为，因为 fixed-H 探针依赖的 ibs_wrap._common_system_xml 目前只有 vdw
        builder 会构造。

        max_repair_rounds 是有限的，不是"重试到天荒地老"：如果 5 轮加密过后
        重叠依然不达标，说明问题很可能不是 λ 密度不够（比如 IBS 偏置本身没
        收敛、构象弛豫太慢、restraint 有问题），继续无限加密只会无意义地烧
        GPU 时间，此时应该交回人工检查，而不是让流水线在一个真实的采样问题
        上假装"自动修复"、无限重跑。

        run_once(n_states, lambdas_var, window_ranges, production_step_overrides=None,
        frozen_validation_step_overrides=None, frozen_validation_is_final_rung=None)
        -> stage_result dict，由调用方决定具体怎么跑（串行直接调用
        _run_dual_lambda_stage，并行则走子进程 worker），本方法只负责判断失败
        原因、生成新方案、清理产物、更新 preopt 缓存并重试。第四、五、六个参数
        只有 vdw/"vanishing" 阶段会用到（分别对应生产 ESS 的 reseed_resample
        分支、warmup 阶段 MBAR 校准后冻结验证的 calibration_pending_validation
        分支 [IBS_BIAS_PROTOCOL_VERSION=12]、该分支的阶梯终态标记），decharging
        的 run_once 接受但忽略它们。

        返回最终通过（或耗尽重试后抛出异常前）的
        (stage_result, n_states, lambdas_var, window_ranges)，调用方应当用
        返回的 n_states/lambdas_var/window_ranges（而不是调用前的原值）去写
        stage 结果缓存，因为它们可能已经被自动修复改写过。

        n_steps_per_window: 本阶段每窗口默认生产步数，只用作 reseed_resample
        判定"该延长多少步"的基准（乘上增长倍数后写入下一轮的
        production_step_overrides）；不传时 reseed_resample 只能退化为按默认
        步数重采一次（不真正延长），并会打印一条说明。

        preopt_protocol_key: [预优化缓存范围收窄] 落盘到 preopt_file 里
        "protocol_key" 字段用的指纹——必须是 `_preopt_protocol_key()` 算出的窄
        指纹，不能是这里的 `protocol_key` 参数本身（那是宽指纹，服务
        `_fixed_h_probe_fingerprint` 这个探针结果缓存，两者用途不同：探针结果
        依赖实际探针构建代码，理应在代码变了之后失效；但 preopt_file 存的是
        λ 路径/窗口边界这份纯数据，不应该因为修了窗口修复循环/reseed_resample
        续采这类无关代码就连带失效）。不传时退化为跟 protocol_key 相同（向后
        兼容旧调用方），但两个真正的调用点（stage1/stage2）都应显式传入窄
        指纹。
        """
        if preopt_protocol_key is None:
            preopt_protocol_key = protocol_key
        stage_type = "vdw" if stage_name == "vanishing" else "coul"
        current_n_states = n_states
        current_lambdas = list(lambdas_var)
        current_ranges = window_ranges

        # ════════════════════════════════════════════════════════════════
        # [deprecated_non_mutating_policy]  IBS non-mutating stage policy.
        #
        # IBS recovers each state's ΔG by reweighting ONE frozen integrated
        # mixture → each target state; adjacent-state fixed-H overlap is NOT a
        # correctness criterion for it. The stage therefore runs ONCE:
        #   • f_k convergence (the frozen OCCUPATION gate) is validated inside
        #     run_once (ibs_engine.run_all_windows, repair_policy="non_mutating_v1");
        #     if it fails it raises IBSWarmupConvergenceError → surfaced for
        #     human / rescue-audit review (nothing mutated).
        #   • _assert_stage_result_sane then applies ONLY the legitimate hard
        #     gates on the frozen production: single-reference importance-ESS
        #     ratio (result["min_overlap"] == compute_effective_sample_number
        #     ratio from GlobalMBARAnalyzer, NOT fixed-H overlap), absolute ESS,
        #     decorrelated samples, endpoint uncertainty.
        # On any hard-gate failure all data is preserved, the stage is NOT
        # marked completed, and the failure is surfaced — λ / f_k / ensemble
        # fingerprint are never touched.
        #
        # ALL ensemble-MUTATING auto-repair below (split window, insert
        # sampling λ, tail renumber, invalidate/remap production,
        # recalibrate_f_k, auto-extend/re-run) is DISABLED: the entire
        # `while True` loop that follows is now UNREACHABLE. It is kept verbatim
        # for review and will be excised in a separate change after code review
        # + one maintainer-run GPU verification. Adjacent fixed-H overlap and
        # the asymmetric-overlap / ΔF-slope arbiter live only inside that dead
        # loop (and inside run_all_windows' now-gated branch), so they are
        # hereby demoted to unused diagnostics.
        # Plan: a-relative-binding-free-energy-framewor-immutable-starlight.md
        # ════════════════════════════════════════════════════════════════
        result = run_once(current_n_states, current_lambdas, current_ranges)
        self._assert_stage_result_sane(stage_label, result)
        return result, current_n_states, current_lambdas, current_ranges

        # 🔑 熔断器：加密只应该在"确实能改善重叠"的前提下继续。之前发现的真实
        # 案例是 min_overlap 一路 0.01553→0.01948→0.01328→0.01266→0.007631→
        # 0.007973→0.003479，不是收敛趋势，是噪声里夹杂着系统性变差——真正的
        # 瓶颈（IBS 偏置未收敛）不会被插点修好，continue 只会一轮轮重复同样的
        # 失败还烧更多 GPU 时间。一旦某一轮加密后 min_overlap 没有改善（含打平
        # 或变差），立即停止，不再等到 max_repair_rounds 耗尽。
        previous_min_overlap = None
        # 🔑 window_idx -> 生产步数覆盖，供 reseed_resample 真正"延长"采样（而不是
        # 只是删旧样本用同样步数重采）。只在纯 sampling-repair 轮（λ 路径不变）
        # 里被写入/消费，见 _diagnose_and_repair_all_pass_low_ess_window 调用点；
        # 键是这一轮 effective_old_ranges 里的 window_idx，只有在写入它的那一轮和
        # 紧接着消费它的下一轮之间 λ 路径确实没变时才有效——这正是
        # path_will_change 门控保证的前提。
        pending_step_overrides: Dict[int, int] = {}
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] window_idx -> 这个窗口冻结验证的
        # 累计目标预算（50k/150k/300k 阶梯）。只在 IBSFrozenCalibrationValidationError
        # 诊断报告 calibration_pending_validation=True 时写入/消费——不同于
        # pending_step_overrides（服务生产 ESS 的 reseed_resample），这个只服务
        # "MBAR 校准好但冻结验证没在累计预算内通过"这一种失败模式。
        frozen_validation_step_overrides: Dict[int, int] = {}
        # 🔑 [四处修复之一] window_idx -> 下一次调用时这个窗口的目标预算是否已经
        # 是阶梯（50k/150k/300k）的最后一档——true 时若仍未通过，
        # ibs_engine.py::run_all_windows 会把该窗口判定为终态失败
        # （calibrated_validation_failed），不会再落盘成"pending"。跟
        # frozen_validation_step_overrides 一样按 window_idx 累积，同一个窗口
        # 一旦被标记过就不会撤销（阶梯只会前进不会后退）。
        frozen_validation_is_final_rung: Dict[int, bool] = {}
        # 🔑 reseed_resample 每次触发时，把该窗口下一轮的步数在当前覆盖倍数上
        # 乘以这个增长因子，封顶下面这个最大倍数——避免一个持续不收敛的窗口
        # 无界烧 GPU；如果封顶后仍不收敛，交给下一轮的 max_repair_rounds 熔断
        # 或人工检查，而不是继续无限加码。
        resample_step_growth_factor = 2.0
        max_resample_step_multiplier = 4.0

        # 🔑 [ladder 独立预算修复] 之前用 for attempt in range(max_repair_rounds+1)，
        # 冻结验证阶梯的 continue（下面 except IBSFrozenCalibrationValidationError
        # 分支）跟拆窗/插λ/production ESS 修复的 continue 共用同一个 attempt 计数器
        # ——阶梯自己的耗尽判据（schedule_idx+1>=len(schedule)）虽然不再依赖
        # attempt>=max_repair_rounds，但如果本轮之前的拆窗/插λ/ESS 修复已经消耗
        # 了大部分共享迭代次数，阶梯可能还没跑到 300k 这一档，for 循环就已经耗尽，
        # 落到下面"自动修复循环异常退出"的兜底错误——阶梯并未真正获得独立预算。
        # 现在改成 while True + 独立的 repair_round 计数器：只有拆窗/插λ/production
        # ESS 修复才会递增 repair_round（语义与之前的 attempt 完全一致，
        # >=max_repair_rounds 时终止），冻结验证阶梯的 continue 完全不触碰它，
        # 阶梯的进退真正只由它自己的 3 档 schedule 长度决定。
        repair_round = 0
        while True:
            try:
                result = run_once(
                    current_n_states, current_lambdas, current_ranges,
                    dict(pending_step_overrides) if pending_step_overrides else None,
                    dict(frozen_validation_step_overrides) if frozen_validation_step_overrides else None,
                    dict(frozen_validation_is_final_rung) if frozen_validation_is_final_rung else None,
                )
            except IBSFrozenCalibrationValidationError as calib_exc:
                # 🔑 [四处修复] 这个 except 分支必须独立于下面
                # `except IBSWarmupConvergenceError` 用的 attempt>=max_repair_rounds
                # 门槛，且必须先于它检查（两者是同级 RuntimeError 子类，不是父子
                # 类型，顺序由代码本身决定，不是异常类型系统自动决定的）。之前的
                # bug：calibration_pending_validation 分支写在
                # `except IBSWarmupConvergenceError` 内部、`attempt>=max_repair_rounds`
                # 判断之后——如果这一轮之前的拆窗/插λ/production ESS 修复已经把
                # 共享的 attempt 计数器耗尽，冻结验证阶梯（本身只有 3 档、有自己
                # 独立的耗尽判据）从未获得执行机会，会直接把原始异常原样抛出，
                # 根本到不了"延长预算续验"或"封顶转人工检查"的分支。现在冻结
                # 验证阶梯的进退完全由它自己的 schedule 长度决定，不受拆窗/插λ/
                # production ESS 修复轮数影响。
                diagnostics = calib_exc.diagnostics
                if calib_exc.terminal:
                    # ibs_engine.py 已经把这个窗口判定为终态失败
                    # （bias_status="calibrated_validation_failed"）并带着
                    # terminal=True 抛出——不再自动重试，直接向上传播这个
                    # 语义清晰的异常（不是被误当成"偏置预热失败"的
                    # IBSWarmupConvergenceError，也不再被拆窗/插λ的逻辑捕获）。
                    raise
                window_idx = diagnostics.get("window_index")
                # 🔑 [跨文件单一数据源修复] 之前这里自己硬编码一份 (50_000, 150_000,
                # 300_000)，跟 ibs_engine.py 内部用于"调用方未提供覆盖字典时"的
                # 阶梯回退逻辑各自维护一份 tuple——两处必须永远保持一致，否则两侧
                # 对"第几档""是否最后一档"的理解会不一致。现在改成从 ibs_engine.py
                # 导入同一个常量，只有一处定义。
                schedule = FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS
                prev_budget = frozen_validation_step_overrides.get(window_idx, schedule[0])
                try:
                    schedule_idx = schedule.index(prev_budget)
                except ValueError:
                    schedule_idx = 0
                if schedule_idx + 1 >= len(schedule):
                    # 阶梯理应已经在上一轮把这个窗口标记为最后一档
                    # （frozen_validation_is_final_rung[window_idx]=True），
                    # ibs_engine.py 那一侧应该已经带着 terminal=True 抛出、被
                    # 上面的分支处理掉，不应该走到这里。真的走到这里说明两侧
                    # 阶梯状态不一致，直接兜底报错，不静默重试、也不假装还能
                    # 继续延长。
                    raise RuntimeError(
                        f"窗口 {window_idx} 的冻结验证阶梯已经在 {schedule[-1]} 步耗尽，"
                        "但收到的异常未标记为 terminal——ibs_engine.py 与 "
                        "abfe_pipeline.py 的阶梯状态不一致，需要人工检查（不应该发生，"
                        "属于代码 bug 而不是正常的采样失败）。"
                    ) from calib_exc
                next_budget = schedule[schedule_idx + 1]
                frozen_validation_step_overrides[window_idx] = next_budget
                is_next_final = (schedule_idx + 1 == len(schedule) - 1)
                frozen_validation_is_final_rung[window_idx] = is_next_final
                self._log(
                    f"  ⏳ {stage_label}: 窗口 {window_idx} 的 MBAR 校准 f_k 冻结验证在累计 "
                    f"{prev_budget} 步预算内未通过，延长累计预算至 {next_budget} 步续验"
                    + ("（这是最后一档，若仍不通过将判定为终态失败 calibrated_validation_failed，"
                       "不再自动重试）" if is_next_final else "")
                    + "（不回 SGD、不重跑 fixed-H overlap/校准探针，只续验累计预算里还没跑完的"
                    "差值，不是重新烧一遍）。"
                )
                continue
            except IBSWarmupConvergenceError as warmup_exc:
                if stage_name != "vanishing" or repair_round >= max_repair_rounds:
                    raise
                cached_preopt = {}
                if os.path.exists(preopt_file):
                    with open(preopt_file, "r") as f:
                        cached_preopt = json.load(f)
                effective_old_ranges = current_ranges or generate_overlapping_windows(
                    current_n_states, pts_per_window=6, overlap=2
                )
                diagnostics = warmup_exc.diagnostics
                # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=7] 边界条件曾经写错：
                # split_window_from_warmup_failure 要求两个孩子各自至少 3 个态、
                # 共享 1 个态，父窗口因此至少需要 3+3-1=5 个态才能这样拆；旧代码
                # 用 >=4 判断"能不能拆"，导致 K=4 的窗口（如 [2,6)）被拆成
                # K=2+K=3（如 [2,4)+[3,6)），产出一个明知统计脆弱的两态窗口。
                # K=4 现在和 K<=3 一样，直接走下面的 fixed-H 双向 overlap 探针，
                # 不再盲拆。
                if int(diagnostics.get("n_states", 0)) >= 5:
                    new_lambdas, new_ranges, feedback = split_window_from_warmup_failure(
                        current_lambdas, effective_old_ranges, diagnostics
                    )
                    self._log(
                        f"  ✂️ {stage_label}: warmup 在窗口 "
                        f"{diagnostics.get('window_index')} coverage 失败；不插 λ，"
                        f"只拆为 {feedback['child_ranges']}，共享旧态 "
                        f"{feedback['shared_global_state']}"
                    )
                else:
                    fixed_probe = diagnostics.get("bidirectional_overlap_probe", {})
                    if not fixed_probe.get("pairs"):
                        raise RuntimeError(
                            "最小 IBS 窗口 coverage 失败，但缺少 fixed-H 双向 overlap 诊断；"
                            "拒绝回退到 Delta-u/算术二分插点。"
                        ) from warmup_exc
                    asymmetric = fixed_probe.get("passed_but_asymmetric_bottleneck")
                    if bool(fixed_probe.get("all_passed", False)) and not (
                        asymmetric and asymmetric.get("qualified")
                    ):
                        raise RuntimeError(
                            "最小 IBS 窗口 coverage 失败，但所有相邻 fixed-H 双向 overlap 均已通过，"
                            "且没有检测到显著局部热力学瓶颈；拒绝自动插点。"
                        ) from warmup_exc
                    new_lambdas, new_ranges, feedback = insert_lambda_from_overlap_failure(
                        current_lambdas, effective_old_ranges, diagnostics
                    )
                    self._log(
                        f"  🧪 {stage_label}: 最小窗口 fixed-H 双向 overlap="
                        f"{feedback['measured_min_bidirectional_overlap']:.5f} < "
                        f"{feedback['overlap_threshold']:.5f}；仅在实测失败边 "
                        f"{feedback['failed_global_edge']} 插入待重测 λ="
                        f"{feedback['inserted_lambda']:.8f}"
                    )
                # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=7] 这条 warmup 失败修复
                # 路径此前直接落盘 new_ranges，从未 canonicalize 过——split 只替换
                # 失败的父窗口，未拆的旧邻窗原样保留，产出的孩子完全可能被邻窗
                # 严格包含（真实案例：[2,4) 完全落在旧邻窗 [3,9) 里，一次采样都是
                # 白跑）。落盘前统一归约一次，跟 production ESS 分支保持一致。
                new_ranges = canonicalize_window_ranges(new_ranges, len(new_lambdas))
                self._invalidate_stage_window_files(
                    stage_name,
                    stage_type,
                    old_lambdas=current_lambdas,
                    old_ranges=effective_old_ranges,
                    new_lambdas=new_lambdas,
                    new_ranges=new_ranges,
                )
                # 🔑 同上（见下面 production ESS 拆窗/插 λ 分支）：warmup coverage
                # 失败触发的拆窗/插 λ 同样重排 window_idx，旧的生产步数覆盖必须
                # 清空，不能带着旧编号进入下一轮。
                if pending_step_overrides:
                    self._log(
                        f"  🧹 {stage_label}: warmup coverage 修复改变了窗口编号，"
                        f"清空 {len(pending_step_overrides)} 条旧的生产步数覆盖 "
                        f"{sorted(pending_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                    )
                    pending_step_overrides.clear()
                # 🔑 [window_idx 陈旧覆盖修复] frozen_validation_step_overrides/
                # frozen_validation_is_final_rung 跟 pending_step_overrides 一样按
                # window_idx 写入，同样会被这里的拆窗/插 λ 重排废掉——不清空的话，
                # 重排后编号恰好相同的新窗口会直接继承旧窗口已经烧到的阶梯预算
                # （甚至"已是最后一档"标记），从未做过冻结验证就被判定终态失败。
                if frozen_validation_step_overrides or frozen_validation_is_final_rung:
                    self._log(
                        f"  🧹 {stage_label}: warmup coverage 修复改变了窗口编号，"
                        f"清空 {len(frozen_validation_step_overrides)} 条旧的冻结验证阶梯覆盖 "
                        f"{sorted(frozen_validation_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                    )
                    frozen_validation_step_overrides.clear()
                    frozen_validation_is_final_rung.clear()
                path_diagnostics = dict(cached_preopt.get("path_diagnostics", {}))
                if feedback.get("thermodynamic_lengths_invalidated", False):
                    # Do not repeat the old 0.5L + 0.5L fiction.  The inserted
                    # coordinate is a hypothesis and its two new edges have no
                    # thermodynamic length until they are sampled.
                    path_diagnostics.pop("optimized_edge_thermodynamic_lengths", None)
                    path_diagnostics["requires_pilot_remeasurement"] = True
                path_diagnostics.setdefault("warmup_feedback_history", []).append(feedback)
                os.makedirs(os.path.dirname(preopt_file), exist_ok=True)
                with open(preopt_file, "w") as f:
                    json.dump({
                        "lambdas_var": new_lambdas,
                        "window_ranges": [list(r) for r in new_ranges],
                        "n_states": len(new_lambdas),
                        "protocol_key": preopt_protocol_key,
                        "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
                        "path_diagnostics": path_diagnostics,
                        "provenance": feedback,
                    }, f, indent=2)
                current_lambdas = new_lambdas
                current_ranges = new_ranges
                current_n_states = len(new_lambdas)
                repair_round += 1
                continue

            if not self._is_overlap_failure(result):
                # 要么通过，要么失败原因不是重叠（NaN/求解失败等）——两种情况
                # 都不该走自动修复，直接交给 _assert_stage_result_sane 决定
                # （通过就直接返回，失败就按原有语义硬性报错）。
                self._assert_stage_result_sane(stage_label, result)
                return result, current_n_states, current_lambdas, current_ranges

            min_overlap = result.get("min_overlap")
            threshold = result.get("min_overlap_threshold")
            diagnostics = result.get("window_overlap_diagnostics")

            # This stage-wide circuit breaker only makes sense for the legacy
            # arithmetic-midpoint path (probe_window_overlap_fn is None): there,
            # every repair round bisects the single worst lambda edge across the
            # whole path, so a non-improving global min_overlap really does mean
            # continued bisection is unlikely to help. The new split-first /
            # fixed-H-probe path (below) can legitimately NOT improve the
            # stage-wide worst ESS on a round that only split a large window
            # (splitting doesn't insert any lambda, so it isn't expected to
            # move the global minimum yet) or on a round that fixed one
            # genuine gap while a different, not-yet-processed window still
            # holds the global worst value. That path already has its own
            # fail-closed gates (an all-passed fixed-H probe or a missing
            # probe result raises immediately), so it doesn't need this
            # cross-round, cross-window comparison to stay safe.
            if probe_window_overlap_fn is None and (
                previous_min_overlap is not None
                and min_overlap is not None
                and min_overlap <= previous_min_overlap
            ):
                raise RuntimeError(
                    f"{stage_label} 阶段自动加密 λ 路径未能改善重叠度：上一轮 "
                    f"min_overlap={previous_min_overlap:.4g}，本轮加密后是 {min_overlap:.4g}"
                    f"（阈值 {threshold:.4g}，未改善或变差）。继续插点不太可能修好它——"
                    "这通常说明真正的瓶颈不是 λ 密度不够，而是 IBS 偏置未收敛/构象弛豫"
                    "过慢/restraint 有问题，请检查 window_overlap_diagnostics 与各窗口"
                    "convergence.json 里的 bias_warmup 状态，而不是继续自动加密 λ 硬跑。"
                )
            previous_min_overlap = min_overlap

            if repair_round >= max_repair_rounds:
                raise RuntimeError(
                    f"{stage_label} 阶段 min_overlap={min_overlap:.4g} 低于阈值 {threshold:.4g}，"
                    f"自动加密 λ 路径并重新采样已连续尝试 {max_repair_rounds} 轮仍未通过。"
                    "这通常说明问题不是 λ 密度不够，而是采样本身有结构性问题（IBS 偏置未收敛、"
                    "构象陷阱、restraint 不一致等），请人工检查 window_overlap_diagnostics 与"
                    "bias_warmup 状态，而不是继续加密 λ 硬跑。"
                )

            effective_old_ranges = current_ranges or generate_overlapping_windows(
                current_n_states, pts_per_window=6, overlap=2
            )

            if probe_window_overlap_fn is not None:
                # Unified with the warmup-failure branch above: a whole window's
                # low ESS is not evidence a specific lambda edge is too wide (a
                # saturated bias/slow relaxation depresses every state equally),
                # so failing windows are only ever split here; a real lambda
                # insertion still requires a measured fixed-H overlap gap.
                to_split, to_probe = plan_vdw_overlap_repair_targets(
                    effective_old_ranges, diagnostics, threshold, min_states_before_split=5,
                )
                if not to_split and not to_probe:
                    raise RuntimeError(
                        f"{stage_label} 阶段 min_overlap={min_overlap:.4g} 低于阈值 {threshold:.4g}，"
                        "但自动修复逻辑未能从 window_overlap_diagnostics 里定位到需要处理的窗口"
                        "（lambdas/min_ess_ratio 明细缺失，或窗口范围对不上当前方案），拒绝盲目重试。"
                    )

                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=7] 之前只按"失败窗口占比 > 50%"硬停止,
                # 太粗糙：(a) 从不实际读取每个失败窗口自己的 warmup 是否真的通过了
                # frozen validation,只是假设"进了 production 就等于 converged 属实";
                # (b) 一个局部坏边可能同时污染两个重叠窗口——总共只有 3 个窗口时,
                # 2/3 已经超过 50%,但根因仍可能只是同一处局部 λ gap,不该被误判成
                # 全局问题。改成两步更精确的判据：
                #   1) 先核实每个失败窗口自己的 convergence.json 是否真的记录了
                #      bias_warmup.status == "frozen_validation_converged"——不确认
                #      直接硬停止,不能把"warmup JSON 写着 converged"当充分证据。
                #   2) 把失败窗口按全局 λ 区间重叠关系分组：IBS 相邻窗口按设计一定
                #      重叠,所以同一段连续失败的窗口自然会被分进同一个 connected
                #      component；只有当失败窗口分散在多个互不相邻的区域时,才说明
                #      这不是某一处局部 gap,而更可能是全局采样协议问题,才硬停止。
                #      单一 connected component（哪怕包含全部窗口）仍按局部问题处理，
                #      继续走下面的拆窗/probe 逻辑。
                failing_windows = to_split + to_probe
                # 🔑 warmup 冻结验证成功有两种落盘状态：SGD 学习本身收敛的
                # "frozen_validation_converged"，以及 fixed-H overlap 全通过后用
                # BAR/MBAR 校准 f_k 再验证通过的"frozen_validation_converged_
                # after_mbar_calibration"（见 ibs_engine.py run_all_windows）。
                # 之前只认第一种字面量，任何被 MBAR 校准修好的窗口都会在这里被
                # 误判成"未确认收敛"而硬停止——必须两者都算作已确认收敛。
                valid_bias_warmup_statuses = {
                    "frozen_validation_converged",
                    "frozen_validation_converged_after_mbar_calibration",
                }
                unvalidated = [
                    (se, self._load_window_bias_warmup_status(stage_name, stage_type, effective_old_ranges, se))
                    for se in failing_windows
                ]
                unvalidated = [(se, status) for se, status in unvalidated if status not in valid_bias_warmup_statuses]
                if unvalidated:
                    raise RuntimeError(
                        f"{stage_label}: 窗口 {unvalidated} 的 convergence.json 里 bias_warmup.status "
                        f"不属于 {sorted(valid_bias_warmup_statuses)}（或读取失败/缺失）——无法确认这些"
                        "窗口自身的偏置真的通过了冻结验证，也就无法判断 production ESS 低究竟是局部 "
                        "λ 密度问题还是 warmup/IBS 偏置协议本身有系统性缺陷，拒绝继续自动拆窗/插点。"
                    )
                # 🔑 之前"失败窗口分散在多个互不相邻区域"直接硬停止在这里执行，
                # 早于下面逐窗口的 fixed-H 分类（哪些窗口真正 fixed-H 失败、哪些
                # 全通过只是 production ESS/f_k 问题）。这个全局判断和"逐窗口独立
                # 分类、互不一票否决"的设计冲突：区域分散本身不能证明是系统性
                # 协议问题，也可能只是恰好有多处独立的局部 λ gap，或多处独立的
                # f_k/采样问题，都能被下面的逐窗口分类正确处理。降级为警告，不再
                # 在探针之前全局拦截；仍然打印出来供人工关注。
                failure_components = self._merge_overlapping_ranges_into_components(failing_windows)
                if len(failure_components) > 1:
                    self._log(
                        f"  ⚠️ {stage_label}: production ESS 低于阈值 {threshold:.4g} 的窗口分散在 "
                        f"{len(failure_components)} 个互不相邻的区域：{failure_components}。各失败窗口"
                        "自身的 warmup 都已确认通过冻结验证；不再因为区域分散就整体硬停止——继续走下面"
                        "逐窗口的 fixed-H 分类，每个窗口按自己的探针结果独立判断是插 λ、拆窗，还是"
                        "fixed-H 通过但 production ESS 低需要 f_k/采样诊断。"
                    )

                # Probe every to_probe window, reusing a cached result whenever
                # its content fingerprint (protocol_key + this window's actual
                # lambda_vdw values + probe threshold) already has a complete
                # entry on disk -- otherwise a fresh process (resume) would
                # burn the same expensive burn-in + sampling per edge again
                # for windows that were already fully probed in an earlier
                # round or an earlier run. Freshly computed edges are
                # persisted one at a time via the on_edge_done callback (not
                # after the whole window finishes), so a crash partway
                # through a multi-edge window only loses the one in-flight
                # edge, not everything computed so far. All results are used
                # (not just the first all-passed one) because each to_probe
                # window is classified and acted on independently below --
                # unlike the old global veto, nothing here is thrown away, so
                # there is no early-stop shortcut to take.
                probe_results = {}
                probe_file = self._fixed_h_probe_file(stage_name)
                for se in to_probe:
                    fingerprint = self._fixed_h_probe_fingerprint(protocol_key, se, current_lambdas)
                    cached_entry = self._load_fixed_h_probe_cache(stage_name).get(fingerprint)
                    if cached_entry is not None and cached_entry.get("complete"):
                        probe_results[se] = cached_entry["pairs"]
                        self._log(
                            f"  ♻️ {stage_label}: 窗口 {se} 复用已缓存的 fixed-H 探针结果"
                            f"（λ 内容/协议指纹匹配，跳过重新采样；见 {probe_file}）"
                        )
                        self._persist_fixed_h_probe_edge(
                            stage_name, stage_type, repair_round, fingerprint, se, cached_entry["pairs"], complete=True,
                        )
                        continue

                    # 🔑 之前只在 complete=True 时才复用缓存；complete=False 的
                    # 部分结果（比如上次跑到第 5/9 条边时进程崩溃）会被整体忽略，
                    # 从第 0 条边重新算——每条边都是独立的固定 Hamiltonian burn-in
                    # + 采样（用同一份 relaxed_positions/relaxed_box 起跑，边与边
                    # 之间不互相依赖），恢复已缓存的前几条边、只补算剩余边，跟
                    # 一次性算完全部边等价，不是近似。
                    resume_pairs = (
                        list(cached_entry["pairs"])
                        if cached_entry is not None and cached_entry.get("pairs")
                        else []
                    )
                    if resume_pairs:
                        self._log(
                            f"  ♻️ {stage_label}: 窗口 {se} 从已缓存的 {len(resume_pairs)} 条边续算 fixed-H "
                            f"探针（λ 内容/协议指纹匹配，只补算剩余边；见 {probe_file}）"
                        )
                    collected_pairs = list(resume_pairs)

                    def _on_edge_done(pair, _se=se, _fp=fingerprint, _pairs=collected_pairs):
                        # 🔑 [两阶段探针] _probe_vdw_window_fixed_overlap 现在可能对
                        # 同一条边调用两次 on_edge_done——阶段一给出 path-only 结果
                        # （bias_calibration_sufficient=None 占位），窗口全部 path
                        # edge 都通过后，阶段二再对同一条边补上真正的 calibration
                        # 结果。按 global_edge 原地覆盖而不是盲目 append，否则同一条
                        # 边会在 _pairs 里出现两次、破坏顺序和长度。
                        edge_key = pair.get("global_edge")
                        replaced = False
                        for _idx, _existing in enumerate(_pairs):
                            if _existing.get("global_edge") == edge_key:
                                _pairs[_idx] = pair
                                replaced = True
                                break
                        if not replaced:
                            _pairs.append(pair)
                        self._log(
                            f"    · 窗口 {_se} fixed-H edge {pair.get('global_edge')}: "
                            f"min_overlap={pair.get('min_bidirectional_overlap', float('nan')):.5f} "
                            f"(阈值 {pair.get('threshold', float('nan')):.5f}, "
                            f"passed={pair.get('passed')}), ΔF="
                            f"{pair.get('delta_f_kJ_mol', float('nan')):.3f}±"
                            f"{pair.get('delta_f_uncertainty_kJ_mol', float('nan')):.3f} kJ/mol, "
                            f"N_decorrelated={pair.get('n_k_decorrelated')}"
                            + (
                                f", bias_calibration_sufficient={pair.get('bias_calibration_sufficient')}"
                                if pair.get("bias_calibration_sufficient") is not None
                                else ""
                            )
                        )
                        self._persist_fixed_h_probe_edge(
                            stage_name, stage_type, repair_round, _fp, _se, list(_pairs), complete=False,
                        )

                    pairs = probe_window_overlap_fn(
                        se, current_lambdas, on_edge_done=_on_edge_done, resume_pairs=resume_pairs,
                    )
                    probe_file = self._persist_fixed_h_probe_edge(
                        stage_name, stage_type, repair_round, fingerprint, se, pairs, complete=True,
                    )
                    probe_results[se] = pairs

                missing = [se for se in to_probe if not probe_results[se]]
                if missing:
                    raise RuntimeError(
                        f"{stage_label}: 窗口 {missing} production ESS 低，但 fixed-H overlap 探针"
                        "未能返回结果；拒绝回退到旧的按 ESS-per-lambda 算术二分插点。"
                    )

                # 🔑 之前一旦 to_probe 里任何一个窗口的 fixed-H 双向 overlap 全部
                # 通过，就把整个 to_probe（包括真正 fixed-H 失败、理应插 λ 的窗口）
                # 一起硬停止——真实案例：production ESS 低的窗口 [2,6)/[5,9)/[14,18)
                # 里，[2,6)/[14,18) 的 fixed-H 探针全通过，[5,9) 至少有一条边未通过，
                # 旧代码却整体 raise，[5,9) 的真实缺口从未被处理，也从未插过 λ。
                # 现在逐窗口分类：fixed-H 确有失败边的窗口照常留在 to_probe 里，
                # 按原逻辑插入待重测 λ；fixed-H 全通过的窗口不插 λ、不拆窗——但也
                # 不能就此把 production ESS 低当作"跟 lambda 无关，忽略"就结束：
                # 两个门槛本来就不同（production ESS 0.05 vs fixed-H overlap
                # 0.03），"fixed-H 全通过"只表示 λ 边已达到最低连通标准，自动插点
                # 缺少证据支持；最终自由能仍然来自这批低 ESS 的 production 数据，
                # 所以改为真正诊断+修复：用相邻边的 BAR/MBAR ΔF 累计出一份独立
                # f_k，跟生产冻结的 f_k 比较——差异明显就用校准 f_k 重新冻结验证，
                # 差异不明显就只重采这个窗口（见
                # _diagnose_and_repair_all_pass_low_ess_window），只针对这一个
                # 窗口生效，不影响 to_split/still_failing 的处理。
                already_good = [
                    se for se in to_probe
                    if all(p.get("passed") for p in probe_results[se])
                ]
                still_failing = [se for se in to_probe if se not in already_good]

                # 🔑 [starvation 修复] 之前只要 to_split/still_failing 非空
                # （path_will_change=True），already_good 窗口的校准/重采样修复
                # 就整轮推迟到"下一轮路径稳定之后"——但一条 18-20 态的路径几乎
                # 每轮都会在别处新冒出一条失败边，导致 already_good 窗口被反复
                # 推迟、永远轮不到，同时白白消耗共享的 repair_round 预算（真实
                # 案例：窗口 (0,3)/(2,6) 被推迟两轮以上，直到 5 轮预算耗尽直接
                # 硬停止，从未真正被修过）。现在分成两条路：路径本轮不变
                # （path_will_change=False）时按原逻辑立即处理；路径本轮会变时
                # 不再"整轮跳过"，而是在下面完成拆窗/插 λ/_invalidate_stage_
                # window_files 之后，按 λ 内容把每个 already_good 窗口重新定位到
                # 新的 (start,end)，同一轮内就把它们的校准/重采样也做掉——
                # _invalidate_stage_window_files 的重命名/清理必须先跑完
                # （它依赖每个窗口的 convergence.json 还在），这份修复才安全。
                path_will_change = bool(to_split or still_failing)
                if not path_will_change:
                    repair_actions = self._apply_already_good_repairs(
                        stage_name, stage_type, stage_label, threshold, repair_round,
                        [(se, se, effective_old_ranges) for se in already_good],
                        probe_results, pending_step_overrides, n_steps_per_window,
                        resample_step_growth_factor, max_resample_step_multiplier,
                    )
                else:
                    # 本轮路径会变，already_good 的修复推迟到下面拆窗/插 λ/
                    # _invalidate_stage_window_files 完成之后，在同一轮内按新
                    # (start,end) 应用——不再是"推迟到下一轮"。
                    repair_actions = []
                    if already_good:
                        self._log(
                            f"  ⏳ {stage_label}: 窗口 {already_good} fixed-H 全通过但 production ESS 低于"
                            f"阈值 {threshold:.4g}；本轮还有需要拆分/插 λ 的窗口，λ 路径即将变化——先完成"
                            "拆窗/插λ/窗口重映射，再在同一轮内按重映射后的新窗口范围对它们做 f_k 校准/"
                            "重采样修复，不再推迟到下一轮。"
                        )

                acted_on = [
                    tuple(a["window_range"]) for a in repair_actions
                    if a["decision"] in ("recalibrate_f_k", "reseed_resample")
                ]
                if not to_split and not still_failing and not acted_on:
                    raise RuntimeError(
                        f"{stage_label}: 本轮除 {already_good} 外没有其它可自动处理的窗口，且它们的"
                        "fixed-H 通过但 production ESS 低都无法自动诊断/修复（既无需要拆分的大窗口，"
                        "也无实测确认存在失败边的 fixed-H 探针，也没有可信的生产冻结 f_k 可供比较）；"
                        f"重跑只会得到相同结果，拒绝盲目重试。请参考上面的诊断信息、{probe_file} 里的"
                        "实测 overlap/ΔF 数值人工判断下一步。"
                    )
                to_probe = still_failing

                if not to_split and not to_probe and acted_on:
                    # Pure sampling-repair round: the λ path/window ranges are
                    # completely unchanged (nothing to split, nothing to
                    # insert). Do NOT fall through to
                    # _invalidate_stage_window_files()/preopt rewrite below --
                    # there is no path change for it to reconcile, and running
                    # it anyway would re-derive its reuse map from each
                    # window's convergence.json, which the sampling repair
                    # above just deleted for every acted-on window; that would
                    # flag them as "unmatched" and purge the ibs_state
                    # overwrite/keep this step just made.
                    # _invalidate_single_window_production() already did the
                    # only invalidation this round needs, scoped to exactly
                    # the touched windows -- just retry next round.
                    continue

                new_ranges = list(effective_old_ranges)
                split_feedback_list = []
                # Splitting never changes lambda count/global indices, so every
                # failing large window can be split in the same round with no
                # index-shift hazard. Must process highest-start-first though:
                # split_window_from_warmup_failure now also reflows the failed
                # window's immediate NEXT neighbor down to single-state overlap
                # (see its docstring) -- if a lower-start window were processed
                # before a higher-start one that's also in to_split, the later
                # call's own recorded (s, e) could already have been shifted by
                # the earlier call's neighbor-reflow and no longer match what's
                # in new_ranges. Processing right-to-left means any window that
                # could touch a given window's neighbor slot is handled after
                # that window itself, so each call's own (s, e) is always still
                # exactly what's on file at the time it's processed.
                for (s, e) in sorted(to_split, key=lambda se: -se[0]):
                    _, new_ranges, split_feedback = split_window_from_warmup_failure(
                        current_lambdas, new_ranges, {"window_index": -1, "global_state_range": [s, e]},
                    )
                    split_feedback_list.append(split_feedback)
                    self._log(
                        f"  ✂️ {stage_label}: production ESS 整窗低（窗口 [{s},{e})，态数={e - s}）；"
                        f"不插 λ，只拆为 {split_feedback['child_ranges']}，"
                        f"共享旧态 {split_feedback['shared_global_state']}"
                    )

                if to_split:
                    # Multiple overlapping parent windows split independently
                    # can produce a child that lands entirely inside a
                    # NEIGHBORING parent's span (IBS windows overlap by
                    # design) -- e.g. parents (0,6)/(3,9) each splitting on
                    # their own midpoint leaves (3,6) strictly contained in
                    # (2,6). Coverage is unaffected (the contained window adds
                    # no lambda index its superset doesn't already have), but
                    # canonicalizing here removes the redundant extra
                    # sampling before it gets persisted/resampled.
                    pre_canonical_count = len(new_ranges)
                    new_ranges = canonicalize_window_ranges(new_ranges, len(current_lambdas))
                    if len(new_ranges) != pre_canonical_count:
                        self._log(
                            f"  🧹 {stage_label}: 批量拆窗产生 {pre_canonical_count} 个窗口，"
                            f"归约掉 {pre_canonical_count - len(new_ranges)} 个被相邻窗口严格包含的"
                            f"冗余窗口，剩 {len(new_ranges)} 个"
                        )

                new_lambdas = current_lambdas
                insert_feedback_list = []
                if to_probe:
                    # 🔑 [批量插边修复] 之前每轮只处理 to_probe 里"最差窗口"的一条
                    # 失败边，哪怕同一轮里其它窗口也有失败边——导致一条边一条边
                    # 排队修，占满 repair_round 预算的同时，让 already_good 窗口
                    # （真正需要 f_k 重新校准/重采样的窗口）因为 path_will_change
                    # 被反复推迟，长期得不到修复（真实案例：窗口 (0,3)/(2,6) 早在
                    # 好几轮之前就被诊断出该修，却因为总有别的窗口这一轮还有失败边
                    # 一直没轮到，直到 5 轮预算耗尽直接硬停止）。现在收集这一轮所有
                    # still_failing 窗口各自的最差失败边，按全局边索引去重、从大到
                    # 小依次插入——insert_lambda_from_overlap_failure 每次插入都会
                    # 平移它所拿到的整份 ranges 列表，处理顺序从大到小保证已经处理
                    # 过的边不会再被后面的插入影响，只需要手动同步更新"尚未处理"的
                    # 窗口自己的 (start,end)。
                    #
                    # 拆窗（to_split）阶段可能已经移动了某个 to_probe 窗口的
                    # (start,end)（拆窗会把失败窗口的紧邻下一个窗口的 start 前移到
                    # 单态重叠）——先按 λ 内容把每个 to_probe 窗口重新定位到
                    # new_ranges 里对应的新范围，不能继续假设 effective_old_ranges
                    # 里的旧 (start,end) 仍然有效，否则会从 insert_lambda_from_
                    # overlap_failure 内部直接 raise（"failed_range not in ranges"）。
                    pending = []
                    unmatched_to_probe = []
                    for se in to_probe:
                        cur_range = self._remap_window_by_lambda_content(
                            se, current_lambdas, new_lambdas, new_ranges,
                        )
                        if cur_range is None:
                            unmatched_to_probe.append(se)
                            continue
                        failed_pairs = [p for p in probe_results[se] if not p.get("passed")]
                        worst_pair = min(
                            failed_pairs,
                            key=lambda p: float(p.get("min_bidirectional_overlap", np.inf)),
                        )
                        pending.append({
                            "orig_se": se,
                            "cur_range": cur_range,
                            "pair": worst_pair,
                            "global_edge": int(worst_pair["global_edge"][0]),
                        })
                    if unmatched_to_probe:
                        self._log(
                            f"  ⚠️ {stage_label}: 窗口 {unmatched_to_probe} 在本轮拆窗/归约后找不到"
                            "λ 内容匹配的新窗口范围（可能被 canonicalize_window_ranges 归约掉，或被"
                            "相邻拆窗的邻窗重排吞并）；本轮跳过它们的插 λ 处理，下一轮重新分类/探测。"
                        )

                    by_edge: Dict[int, List[Dict]] = {}
                    for item in pending:
                        by_edge.setdefault(item["global_edge"], []).append(item)
                    ordered_edges = sorted(by_edge.keys(), reverse=True)

                    for edge_pos, edge in enumerate(ordered_edges):
                        group = by_edge[edge]
                        # 同一条全局边可能同时是多个重叠窗口各自的最差失败边——只
                        # 需要真正插一次；挑测得更差的那个窗口作为
                        # insert_lambda_from_overlap_failure 的"失败窗口"（决定
                        # 拆成两个孩子的是哪个窗口），其余窗口会被它内部通用的
                        # 平移分支自动一并修好，不重复插点。
                        primary = min(
                            group,
                            key=lambda it: (
                                float(it["pair"]["min_bidirectional_overlap"]),
                                it["cur_range"][0],
                            ),
                        )
                        others = [it for it in group if it is not primary]
                        diag = {
                            "window_index": -1,
                            "global_state_range": list(primary["cur_range"]),
                            "bidirectional_overlap_probe": {"pairs": [primary["pair"]]},
                        }
                        new_lambdas, new_ranges, insert_feedback = insert_lambda_from_overlap_failure(
                            new_lambdas, new_ranges, diag,
                        )
                        insert_feedback_list.append(insert_feedback)
                        insert_at = int(insert_feedback["failed_global_edge"][1])
                        failed_range = tuple(primary["cur_range"])
                        windows_fixed_for_free = [it["orig_se"] for it in others]
                        self._log(
                            f"  🧪 {stage_label}: 窗口 {primary['orig_se']} fixed-H 双向 overlap="
                            f"{insert_feedback['measured_min_bidirectional_overlap']:.5f} < "
                            f"{insert_feedback['overlap_threshold']:.5f}；在实测失败边 "
                            f"{insert_feedback['failed_global_edge']} 插入待重测 λ="
                            f"{insert_feedback['inserted_lambda']:.8f}"
                            + (f"；同一条边同时覆盖窗口 {windows_fixed_for_free}，一并解决，不重复插点"
                               if windows_fixed_for_free else "")
                        )
                        # 同步更新尚未处理窗口的 (start,end)——跟
                        # insert_lambda_from_overlap_failure 内部完全一致的 4 分支
                        # 位移规则（原地匹配失败窗口/整体在插入点左侧/整体在插入点
                        # 右侧/跨插入点），保证下一次迭代里这些窗口自己的
                        # cur_range 仍然精确对应 new_ranges 里的实际内容。
                        for other_edge in ordered_edges[edge_pos + 1:]:
                            for it in by_edge[other_edge]:
                                s, e = it["cur_range"]
                                if (s, e) == failed_range:
                                    it["cur_range"] = (s, insert_at + 1)
                                elif e <= insert_at:
                                    pass
                                elif s >= insert_at:
                                    it["cur_range"] = (s + 1, e + 1)
                                else:
                                    it["cur_range"] = (s, e + 1)

                # Final canonicalization pass right before anything is
                # persisted, independent of whether the split loop above
                # already ran one -- cheap and idempotent on an already-
                # canonical list, and it's the one check that must hold no
                # matter which combination of split/insert produced new_ranges.
                new_ranges = canonicalize_window_ranges(new_ranges, len(new_lambdas))

                self._invalidate_stage_window_files(
                    stage_name,
                    stage_type,
                    old_lambdas=current_lambdas,
                    old_ranges=effective_old_ranges,
                    new_lambdas=new_lambdas,
                    new_ranges=new_ranges,
                )
                # 🔑 split/insert 之后 window_idx 的编号会整体重排（拆窗新增窗口、
                # 插 λ 改变全局态编号），pending_step_overrides 是按上一轮的
                # window_idx 位置写入的，路径变了就不再对应同一个物理窗口——
                # 留着会把延长步数的覆盖值发给这一轮里编号恰好相同、但其实是
                # 另一个窗口的目标，白烧 GPU 且让真正欠采样的窗口拿不到延长。
                # 直接清空，下一轮任何窗口需要 reseed_resample 时都从默认步数
                # 重新开始按倍数增长，不去猜哪个旧 key 还对得上。
                if pending_step_overrides:
                    self._log(
                        f"  🧹 {stage_label}: λ 路径本轮拆窗/插 λ 改变了窗口编号，"
                        f"清空 {len(pending_step_overrides)} 条旧的生产步数覆盖 "
                        f"{sorted(pending_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                    )
                    pending_step_overrides.clear()
                # 🔑 [window_idx 陈旧覆盖修复] 同上：frozen_validation_step_overrides/
                # frozen_validation_is_final_rung 一样按 window_idx 写入，必须跟
                # pending_step_overrides 一起清空，理由同下面 legacy 分支的注释。
                if frozen_validation_step_overrides or frozen_validation_is_final_rung:
                    self._log(
                        f"  🧹 {stage_label}: λ 路径本轮拆窗/插 λ 改变了窗口编号，"
                        f"清空 {len(frozen_validation_step_overrides)} 条旧的冻结验证阶梯覆盖 "
                        f"{sorted(frozen_validation_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                    )
                    frozen_validation_step_overrides.clear()
                    frozen_validation_is_final_rung.clear()
                # 🔑 [starvation 修复，slow lane] 上面已经完成拆窗/插 λ/窗口
                # 重映射、_invalidate_stage_window_files，以及本轮路径变化触发
                # 的 pending_step_overrides/frozen_validation_* 陈旧覆盖清空——
                # 这里才第一次安全地处理本轮被 path_will_change 挡住的
                # already_good 窗口：按 λ 内容把每个窗口重新定位到 new_ranges 里
                # 对应的新 (start,end)，同一轮内立即做校准/重采样修复，不再拖到
                # 下一轮。必须放在陈旧覆盖清空之后运行——若提前到清空之前，
                # reseed_resample 分支这一轮刚写入 pending_step_overrides 的全新
                # （按新 window_idx 编号的）延长步数覆盖会被紧接着的"清空陈旧
                # 覆盖"逻辑一并冲掉，下一轮读不到。同理也必须在
                # _invalidate_stage_window_files 完成之后运行（它依赖每个窗口的
                # convergence.json 还在原位置；_diagnose_and_repair_all_pass_
                # low_ess_window 的修复分支会删除这个文件，顺序反了会被上面的
                # 重用判断误当成"未匹配、该清理"）。probe_results 仍按探测时的
                # 原始 se 为键（探针结果只跟 λ 内容有关，不跟位置有关），只有
                # 传给诊断函数的窗口范围/全量范围列表要用重映射后的新值。
                if already_good:
                    already_good_entries = []
                    unmatched_already_good = []
                    for se in already_good:
                        new_se = self._remap_window_by_lambda_content(
                            se, current_lambdas, new_lambdas, new_ranges,
                        )
                        if new_se is None:
                            unmatched_already_good.append(se)
                            continue
                        already_good_entries.append((se, new_se, new_ranges))
                    if unmatched_already_good:
                        self._log(
                            f"  ⚠️ {stage_label}: 窗口 {unmatched_already_good} fixed-H 全通过，"
                            "但本轮拆窗/插 λ 后找不到 λ 内容匹配的新窗口范围（可能被 "
                            "canonicalize_window_ranges 归约掉，或被相邻拆窗的邻窗重排吞并）；"
                            "暂缓其 f_k 校准/重采样修复，下一轮重新分类/按需重新探测。"
                        )
                    self._apply_already_good_repairs(
                        stage_name, stage_type, stage_label, threshold, repair_round,
                        already_good_entries,
                        probe_results, pending_step_overrides, n_steps_per_window,
                        resample_step_growth_factor, max_resample_step_multiplier,
                    )
                cached_preopt = {}
                if os.path.exists(preopt_file):
                    with open(preopt_file, "r") as f:
                        cached_preopt = json.load(f)
                path_diagnostics = dict(cached_preopt.get("path_diagnostics", {}))
                if insert_feedback_list:
                    # Do not repeat the old 0.5L + 0.5L fiction -- the inserted
                    # coordinate is a hypothesis and its two new edges have no
                    # thermodynamic length until they are sampled.
                    path_diagnostics.pop("optimized_edge_thermodynamic_lengths", None)
                    path_diagnostics["requires_pilot_remeasurement"] = True
                path_diagnostics.setdefault("production_repair_history", []).extend(
                    split_feedback_list + insert_feedback_list
                )
                os.makedirs(os.path.dirname(preopt_file), exist_ok=True)
                with open(preopt_file, "w") as f:
                    json.dump({
                        "lambdas_var": new_lambdas,
                        "window_ranges": [list(r) for r in new_ranges],
                        "n_states": len(new_lambdas),
                        "protocol_key": preopt_protocol_key,
                        "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
                        "path_diagnostics": path_diagnostics,
                        "provenance": {
                            "source": "production_overlap_repair_split_then_probe",
                            "round": repair_round + 1,
                            "prior_min_overlap": min_overlap,
                            "prior_min_overlap_threshold": threshold,
                        },
                    }, f, indent=2)
                current_lambdas, current_ranges = new_lambdas, new_ranges
                current_n_states = len(new_lambdas)
                repair_round += 1
                continue

            new_lambdas, new_ranges = refine_stage_lambda_path_by_overlap(
                current_lambdas,
                effective_old_ranges,
                diagnostics,
                threshold,
            )
            if new_lambdas is None:
                raise RuntimeError(
                    f"{stage_label} 阶段 min_overlap={min_overlap:.4g} 低于阈值 {threshold:.4g}，"
                    "但自动修复逻辑未能从 window_overlap_diagnostics 里定位到需要加密的 λ 区间"
                    "（缺少 ess_ratio_per_lambda 明细，或诊断结构异常），拒绝盲目重试。"
                )

            self._log(
                f"  🔧 {stage_label}: min_overlap={min_overlap:.4g} < {threshold:.4g}，"
                f"第 {repair_round + 1}/{max_repair_rounds} 轮自动加密 λ 路径 "
                f"({len(current_lambdas)} -> {len(new_lambdas)} 个态)，按 λ 内容比对复用旧窗口产物"
            )
            self._invalidate_stage_window_files(
                stage_name,
                stage_type,
                old_lambdas=current_lambdas,
                old_ranges=effective_old_ranges,
                new_lambdas=new_lambdas,
                new_ranges=new_ranges,
            )
            # 🔑 同上（见拆窗/插 λ 分支）：加密 λ 路径同样会重排 window_idx，
            # pending_step_overrides 里按旧编号写入的延长步数覆盖不再对应
            # 同一个物理窗口，必须清空，不能带着旧编号进入下一轮。
            if pending_step_overrides:
                self._log(
                    f"  🧹 {stage_label}: λ 路径本轮加密改变了窗口编号，"
                    f"清空 {len(pending_step_overrides)} 条旧的生产步数覆盖 "
                    f"{sorted(pending_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                )
                pending_step_overrides.clear()
            # 🔑 [window_idx 陈旧覆盖修复] frozen_validation_step_overrides/
            # frozen_validation_is_final_rung 同样按 window_idx 写入，加密 λ 路径
            # 重排编号后必须一起清空，理由同上（见拆窗/插 λ 分支的对应注释）：
            # 否则重排后编号恰好相同的新窗口会继承旧窗口的阶梯预算/终态标记。
            if frozen_validation_step_overrides or frozen_validation_is_final_rung:
                self._log(
                    f"  🧹 {stage_label}: λ 路径本轮加密改变了窗口编号，"
                    f"清空 {len(frozen_validation_step_overrides)} 条旧的冻结验证阶梯覆盖 "
                    f"{sorted(frozen_validation_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                )
                frozen_validation_step_overrides.clear()
                frozen_validation_is_final_rung.clear()
            os.makedirs(os.path.dirname(preopt_file), exist_ok=True)
            with open(preopt_file, "w") as f:
                json.dump(
                    {
                        "lambdas_var": new_lambdas,
                        "window_ranges": [list(r) for r in new_ranges],
                        "n_states": len(new_lambdas),
                        # 🔑 落这个协议指纹，是为了让 resume 时能安全地信任一份
                        # n_states 已经不等于最初请求值、但确实是本协议下已验证过的
                        # 自动加密结果的缓存（见 run_full_pipeline 里的 preopt resume
                        # 读取逻辑），而不是盲目要求 n_states 精确等于最初的猜测值。
                        "protocol_key": preopt_protocol_key,
                        "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
                        "provenance": {
                            "source": "auto_repair_by_overlap",
                            "round": repair_round + 1,
                            "prior_n_states": current_n_states,
                            "prior_min_overlap": min_overlap,
                            "prior_min_overlap_threshold": threshold,
                            "prior_window_overlap_diagnostics": diagnostics,
                        },
                    },
                    f,
                    indent=2,
                )
            current_lambdas, current_ranges = new_lambdas, new_ranges
            current_n_states = len(new_lambdas)
            repair_round += 1

        raise RuntimeError(f"{stage_label}: 自动修复循环异常退出（不应到达这里）")

    def _stage_protocol_key(
        self,
        stage_name: str,
        potential_type: str,
        boresch_params: Optional[Dict],
        decharge_method: str = "pme",
        n_states: Optional[int] = None,
        dexp_params: Optional[Dict] = None,
        final_gate_thresholds: Optional[Dict] = None,
    ) -> Dict:
        """
        双λ阶段结果缓存的"协议指纹"。

        之前 stage1/stage2 的 resume 判定只比较 n_states，同一个 output 目录里
        换了 decharge_method（比如从默认 pme 切到实验性 shadow_ibs）、potential_type
        或 Boresch 开关之后再 --resume，n_states 不变的话会直接复用旧协议算出来的
        stage 结果，而不会报错或重算——这是一个真实的静默串协议 bug，不是假设的
        边界情况。这里把决定"这个 stage 结果是用什么协议算出来的"的关键字段收敛
        成一个可比较的 dict，连同 n_states 一起写进 stage 缓存文件，resume 时逐字
        段比较，任何一项不一致都视为缓存失效、强制重新采样。
        """
        run_config = dict(getattr(self, "_last_run_config", {}) or {})
        run_config.pop("resume", None)
        run_config.pop("run_equilibration", None)
        return _protocol_fingerprint({
            "kind": "dual_lambda_stage",
            "stage_name": stage_name,
            "potential_type": str(potential_type),
            "dexp_params": dexp_params,
            "boresch_params": boresch_params,
            "decharge_method": str(decharge_method) if stage_name == "decharging" else "n/a",
            "requested_n_states": None if n_states is None else int(n_states),
            "run_config": run_config,
            "temperature_K": self.temperature.value_in_unit(unit.kelvin),
            "pressure_bar": self.pressure.value_in_unit(unit.bar),
            "ligand_indices": [int(i) for i in self.ligand_indices],
            "system_xml_sha256": _system_xml_hash(self.system),
            "topology_sha256": _topology_hash(self.topology),
            "coordinates_nm_sha256": _positions_hash(self.positions),
            "code_sha256": _code_hash(),
            "aces_softcore_params": ACESoftcorePotential.optimize_alpha(
                len(self.ligand_indices)
            ),
            # 🔑 stage1/stage2 的"completed"缓存是聚合过的最终 ΔG/误差，命中时完全
            # 不会碰每窗口的 base/bias 文件，也就绕过了 run_all_windows 里的
            # wca_accounting_version 校验。必须把这个版本号也编进协议指纹，否则
            # WCA base/bias 记账口径变了之后，旧的"completed"缓存还会被当成有效
            # 结果直接复用，静默沿用旧口径算出来的 ΔG。
            "wca_accounting_version": WCA_ACCOUNTING_VERSION,
            # 🔑 同理：IBS 偏置预热/冻结协议（是否满足严格收敛判据才放行进生产、
            # 生产阶段是否冻结 f_k）变了之后，旧协议下产出的 stage 结果同样不能
            # 被当成有效缓存直接复用。
            "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
            # 🔑 [non_mutating_v1] 采样修复策略必须进入协议指纹：旧的变异策略
            # （fixed-H 探针 + 就地重校准 f_k + 拆窗/插 λ）产出的 stage 结果，其
            # f_k/λ 网格来源与非变异策略不同一个参考系，绝不能被非变异策略的
            # run 静默复用；也防止旧策略下写好的 "completed" 缓存在进入非变异
            # early-return 之前就把 stage 跳过。
            "sampling_repair_policy": "non_mutating_v1",
            "thermodynamic_path_protocol_version": (
                THERMODYNAMIC_PATH_PROTOCOL_VERSION
                if stage_name == "vanishing"
                else "n/a"
            ),
            # 🔑 [P1 修复] 最终收敛门（GlobalMBARAnalyzer.solve_stage_integrated
            # 的 final_min_ess_ratio/final_min_absolute_ess/
            # final_min_decorrelated_samples/final_max_uncertainty_kJ_mol）的
            # 实际生效阈值必须编进协议指纹——这些是决定"这个 stage 结果算不算
            # 收敛"的关键参数，改了阈值（无论是用户显式传参还是代码默认值本身
            # 变化）而不让旧的"completed"缓存失效，等于允许一个用旧的、更宽松
            # 门槛判定过关的结果继续被当成通过了新门槛。传入调用方实际解析出的
            # 阈值字典（而不是只看用户是否显式传参），这样即使只是代码默认值本身
            # 变了，缓存也会正确失效。
            "final_gate_thresholds": final_gate_thresholds,
        })

    def _preopt_protocol_key(
        self,
        stage_name: str,
        potential_type: str,
        boresch_params: Optional[Dict],
        decharge_method: str = "pme",
        dexp_params: Optional[Dict] = None,
    ) -> Dict:
        """λ 路径预优化（optimize_stage1_decharging/optimize_stage2_vanishing）
        专用的、范围更窄的协议指纹——只保留真正影响"这次预优化测到的 dU/dλ
        路径还有效吗"的字段，故意不包含 `_stage_protocol_key` 里那些只跟
        *实际生产采样*相关、预优化根本不会碰到的字段：
          - `wca_accounting_version`：管 base/bias 力组切分记账口径，预优化
            没有这个概念（探针系统只有一个 group1 力 + 有限差分，没有
            base/bias 拆分）。
          - `ibs_bias_protocol_version`：管 IBS SGD 学习/冻结验证协议，预优化
            完全不涉及 IBS bias/f_k。
          - `final_gate_thresholds`：管最终聚合 ΔG 是否通过收敛门，是 stage
            完成之后才有意义的判据，预优化阶段还没有任何 ΔG 可言。
          - 完整版 `code_sha256`（四文件合并哈希）：换成范围更窄的
            `_preopt_code_hash()`（只哈希 abfe_preoptimizer.py + abfe_core.py），
            这样修 ibs_engine.py/abfe_pipeline.py 里跟预优化无关的 bug
            （窗口修复循环、production checkpoint 续采等）不会连带让这份
            要跑好几个小时的预优化缓存失效——这正是收窄这份指纹的直接原因。
        `thermodynamic_path_protocol_version` 仍然保留：它是预优化算法本身
        的版本号，理应让这份指纹随之变化。
        """
        run_config = dict(getattr(self, "_last_run_config", {}) or {})
        run_config.pop("resume", None)
        run_config.pop("run_equilibration", None)
        return _protocol_fingerprint({
            "kind": "dual_lambda_preopt",
            "stage_name": stage_name,
            "potential_type": str(potential_type),
            "dexp_params": dexp_params,
            "boresch_params": boresch_params,
            "decharge_method": str(decharge_method) if stage_name == "decharging" else "n/a",
            "run_config": run_config,
            "temperature_K": self.temperature.value_in_unit(unit.kelvin),
            "pressure_bar": self.pressure.value_in_unit(unit.bar),
            "ligand_indices": [int(i) for i in self.ligand_indices],
            "system_xml_sha256": _system_xml_hash(self.system),
            "topology_sha256": _topology_hash(self.topology),
            "coordinates_nm_sha256": _positions_hash(self.positions),
            "preopt_code_sha256": _preopt_code_hash(),
            "aces_softcore_params": ACESoftcorePotential.optimize_alpha(
                len(self.ligand_indices)
            ),
            "thermodynamic_path_protocol_version": (
                THERMODYNAMIC_PATH_PROTOCOL_VERSION
                if stage_name == "vanishing"
                else "n/a"
            ),
        })

    @staticmethod
    def _preopt_cache_matches_ignoring_code_hash(
        cached_protocol: Optional[Dict], fresh_preopt_key: Dict
    ) -> bool:
        """[预优化缓存 schema 迁移兼容] 判断磁盘上缓存的 protocol_key——无论是
        `_preopt_protocol_key` 上线之前的旧宽指纹（`_stage_protocol_key`，含
        `code_sha256`/`wca_accounting_version`/`ibs_bias_protocol_version`/
        `final_gate_thresholds`）还是现在的新窄指纹——除了这几个字段本身之外，
        物理相关的其余字段是否跟当前这次运行完全一致。

        目的：旧缓存文件的 protocol_key 是按宽指纹写的，跟新窄指纹连顶层键
        集合都不一样，永远不可能逐字节相等——直接比较会把"物理输入完全没变、
        只是指纹 schema 换了"误判成"协议不一致"，逼着重新跑一遍代价高达数
        小时的 λ 路径预优化。只要除了那几个刻意跟预优化物理内容无关的字段
        之外，其余每一项都完全一致，就认为这份旧缓存仍然可信；调用方应据此
        直接复用 lambdas_var/window_ranges，并把 protocol_key 原地重新盖成
        新窄指纹（自愈式迁移一次即可，之后的 resume 都是正常的窄指纹比较）。

        任何一步比较失败（缺字段、类型不对、payload 结构变了）都保守地返回
        False，交由调用方按"协议不一致"的原有逻辑处理——这个函数只放行
        "确认物理输入没变"的情况，不放行"看起来大概没变"。
        """
        if not isinstance(cached_protocol, dict) or not isinstance(fresh_preopt_key, dict):
            return False
        cached_payload = cached_protocol.get("payload")
        fresh_payload = fresh_preopt_key.get("payload")
        if not isinstance(cached_payload, dict) or not isinstance(fresh_payload, dict):
            return False
        legacy_only_fields = ("code_sha256", "wca_accounting_version", "ibs_bias_protocol_version")
        if not all(field in cached_payload for field in legacy_only_fields):
            return False
        code_identity_fields = {"kind", "preopt_code_sha256"}
        for key, fresh_value in fresh_payload.items():
            if key in code_identity_fields:
                continue
            if key not in cached_payload or cached_payload[key] != fresh_value:
                return False
        return True

    @staticmethod
    def _lambda_path_fingerprint(lambdas_var, window_ranges) -> Dict:
        lambda_values = [] if lambdas_var is None else list(lambdas_var)
        ranges = [] if window_ranges is None else list(window_ranges)
        return _protocol_fingerprint({
            "lambdas_var": _lambda_signature(lambda_values),
            "window_ranges": [list(map(int, r)) for r in ranges],
        })

    @staticmethod
    def _build_stage_cache_payload(
        stage_name: str,
        result: Dict,
        n_states: int,
        protocol_key: Dict,
        lambdas_var,
        window_ranges,
    ) -> Dict:
        """
        构造 stage1/stage2 缓存文件的完整落盘内容。

        之前这里只存 stage/total_delta_G/total_error/n_states 四个字段，resume
        命中时 method/diagnostics/lambda_endpoint_diagnostics 全部丢失，导致
        compute_final_results 里的 stage_diagnostics 在 resumed run 上被静默
        清空（overlap 诊断、PME self-correction 诊断、Shadow-IBS 的 experimental
        标记等全部看不到）。这里把 _run_dual_lambda_stage 返回的完整结果落盘，
        resume 命中时可以原样当作 stage1/stage2 使用，不用再退化成一个裸数字。

        🔑 [P1-15] 顶层的 `converged` / `coverage_diagnostics` 也一并落盘。
        它们此前只在内存里存在，归档结果无法复核"为什么放行"。

        ⚠️ 写入方 `_atomic_write_json` 用的是 `json.dump(payload, handle, indent=2)`，
        **没有 `cls=NumpyEncoder`**。而 `window_overlap_diagnostics` / `f_k` /
        `coverage_diagnostics` 里混着 numpy 标量与数组，直接塞进去会 `TypeError`
        并让整个 checkpoint 写失败。所以这里统一过一遍 `_json_safe`。
        """
        return {
            "stage": stage_name,
            "total_delta_G": float(result["total_delta_G"]),
            "total_error": float(result["total_error"]),
            "n_states": n_states,
            # ⚠️ protocol_key / lambda_path_fingerprint 刻意**不过** _json_safe：
            # 它们参与缓存身份比对，任何形状变换（tuple→list、int key→str key）
            # 都可能让 resume 误判。它们本来也不含 numpy。
            "protocol_key": protocol_key,
            "lambda_path_fingerprint": ABFEPipeline._lambda_path_fingerprint(
                lambdas_var, window_ranges
            ),
            "method": result.get("method"),
            # 只有这几项是新纳入落盘、且确实含 numpy 的，逐个过 _json_safe。
            "diagnostics": _json_safe(result.get("diagnostics", {})),
            "lambda_endpoint_diagnostics": _json_safe(
                result.get("lambda_endpoint_diagnostics", {})
            ),
            # 顶层收敛与覆盖证据：resume 命中时 _assert_stage_result_sane 需要它们，
            # 事后审计也需要。
            "converged": _json_safe(result.get("converged")),
            "coverage_diagnostics": _json_safe(result.get("coverage_diagnostics")),
        }

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
        # 🔑 双重预平衡防护：run_equilibration 是调用方（runabfe.py）根据磁盘上
        # equilibrium_is_done()/config.reset 算出来的，它并不知道这个 pipeline
        # 实例在*本次调用*里是否已经在别处（resolve_boresch_restraint() 里的
        # pre_equilibrate()、或 _rebalance_with_boresch()）做过预平衡/Boresch
        # 再平衡。真实 bug 场景：--reset 时 run_equilibration 会被强制置 True；
        # 外部 Boresch 参数来源（traditional/orb_ml）从不写
        # pre_equilibration.dcd，所以 equilibrium_is_done() 恒为 False、
        # run_equilibration 恒为 True。两种情况下，如果本进程已经做过 Boresch
        # 再平衡，这里再跑一次无约束预平衡会直接覆盖 self.positions，把刚做完
        # 的、带限制力的平衡坐标扔掉——不是"重复浪费"而是"结果被静默改写"。
        # 这两个标记只反映本进程真实发生过什么，比 run_equilibration 这个
        # 磁盘/config 推导值更可信，因此在这里短路，而不是依赖调用方把
        # run_equilibration 算对。
        if self._boresch_rebalance_done_this_process:
            self._log(
                "  ⏭️ 本进程已完成 Boresch 限制力再平衡，跳过 run_full_pipeline 内部的"
                "预平衡块（避免用无约束预平衡覆盖刚完成的 Boresch 平衡坐标）。"
            )
        elif self._pre_equilibration_done_this_process:
            self._log(
                "  ⏭️ 本进程已完成一次预平衡（未带 Boresch），跳过 run_full_pipeline "
                "内部重复的预平衡块。"
            )
        elif run_equilibration:
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
        self.repair_pbc_molecule_integrity(context="run_full_pipeline 第 2 节")
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
                    committed_doc = json.load(f)
                committed_eq = committed_doc["equilibrium_values"]

                # 🔑 [BORESCH-COMMIT] 复用之前必须验证它还描述当前构象。
                # 这条保护此前**只检查文件是否存在**，于是一份 2026-07-10 写下的
                # 平衡值（thetaA0/thetaB0 对调、二面角错乱）在体系于 07-26 被整体
                # 重新平衡之后，仍被沿用了 17 天：限制力把配体从自己的 pose 上拽走
                # 3.4 Å（无约束预平衡只漂 0.60 Å），复合物腿去电荷因此偏低约
                # 25 kJ/mol，解析释放修正同样是错的。详见
                # boresch_committed_deviation_sigma 的 docstring。
                self._assert_committed_boresch_still_matches_pose(
                    committed_doc=committed_doc,
                    committed_eq=committed_eq,
                    boresch_params=boresch_params,
                    committed_path=committed_path,
                )

                boresch_params = dict(boresch_params)
                boresch_params["equilibrium_values"] = committed_eq
                r0 = committed_eq.get("r0", 0) * 10  # nm → Å
                self._log(
                    f"  ♻️ 本腿此前已提交过 Boresch 平衡值 (resume)，复用缓存值 "
                    f"(r0={r0:.2f} Å)，不再从当前坐标重新锚定。"
                    "（已校验其与当前坐标的几何一致性）"
                )
            else:
                self._log("  🔧 正在用当前坐标更新 Boresch 平衡几何量...")
                boresch_params = self.update_boresch_from_last_frame(boresch_params)
                r0 = boresch_params["equilibrium_values"].get("r0", 0) * 10  # nm → Å
                self._log(f"  ✓ Boresch 平衡值已更新: r0={r0:.2f} Å")
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                # 🔑 [BORESCH-COMMIT] 带上身份信息落盘。裸的
                # {"equilibrium_values": ...} 无法判断它是从哪个锚点/哪套力常数、
                # 什么时候推出来的——正是它让上面那组错值活过了 17 天。
                _atomic_write_json(committed_path, _json_safe({
                    "schema_version": BORESCH_COMMITTED_SCHEMA_VERSION,
                    "equilibrium_values": boresch_params["equilibrium_values"],
                    "receptor_indices": list(boresch_params.get("receptor_indices", [])),
                    "ligand_indices": list(boresch_params.get("ligand_indices", [])),
                    "force_constants": dict(boresch_params.get("force_constants", {})),
                    "temperature_K": self.temperature.value_in_unit(unit.kelvin),
                    "derived_at": datetime.now().isoformat(),
                    "note": (
                        "本腿后续 resume 强制复用这组平衡值；复用前会用 "
                        "boresch_committed_deviation_sigma 校验它是否仍描述当前构象。"
                    ),
                }))
                self._log(f"  📌 Boresch 平衡值已提交落盘，本腿后续 resume 将强制复用: {committed_path}")

        # =========================================================================
        # 2. 路由采样 (支持阶段级 Resume)
        # =========================================================================
        # 🔑 [P0] 这是整条 resume 链路里最先被检查的一环——status=="completed" 就
        # 直接把整份 final_results.json 读回来当最终结果 return，连下面 stage1/2
        # 的 protocol_key 校验都不会走到。之前只判断 status，不比对协议指纹，导致
        # 任何"记账口径变了但状态还标着 completed"的旧 output_dir（比如 WCA
        # e_base/e_bias 切分方式改变之前跑完的结果）会被无条件当成最终答案直接
        # 返回。这里必须先算出本次运行的协议指纹，任何环节缺失/不匹配都不能走这条
        # 早退路径。
        def _build_top_level_protocol_key() -> Dict:
            config = dict(self._last_run_config)
            config.pop("resume", None)
            config.pop("run_equilibration", None)
            return _protocol_fingerprint({
                "kind": "abfe_final_result",
                "run_config": config,
                "potential_type": potential_type,
                "dexp_params": dexp_params,
                "boresch_params": boresch_params,
                "torsion_params": torsion_params,
                "decharge_method": kwargs.get("decharge_method", "pme"),
                "system_xml_sha256": _system_xml_hash(self.system),
                "topology_sha256": _topology_hash(self.topology),
                "coordinates_nm_sha256": _positions_hash(self.positions),
                "code_sha256": _code_hash(),
                "ligand_indices": [int(i) for i in self.ligand_indices],
                "temperature_K": self.temperature.value_in_unit(unit.kelvin),
                "pressure_bar": self.pressure.value_in_unit(unit.bar),
                "aces_softcore_params": ACESoftcorePotential.optimize_alpha(
                    len(self.ligand_indices)
                ),
                "wca_accounting_version": WCA_ACCOUNTING_VERSION,
                "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
                "thermodynamic_path_protocol_version": (
                    THERMODYNAMIC_PATH_PROTOCOL_VERSION
                ),
                "traditional_lj_lrc_protocol_version": (
                    TRADITIONAL_LJ_LRC_PROTOCOL_VERSION
                ),
                # The actual optimized/refined lambda contents are committed in
                # these files.  Hash their bytes so a same-length edited path can
                # never hit the top-level completed-result cache.
                "preopt_cache_sha256": {
                    "decharging": _file_sha256(os.path.join(
                        self.checkpoint_dir, "preopt_dual_decharging.json"
                    )),
                    "vanishing": _file_sha256(os.path.join(
                        self.checkpoint_dir, "preopt_dual_vanishing.json"
                    )),
                },
            })

        _top_level_protocol_key = _build_top_level_protocol_key()

        sampling_key = f"sampling_{decoupling_scheme}"
        samp_status = stages.get(sampling_key, {}).get("status")

        if resume and samp_status == "completed":
            results_file = os.path.join(self.output_dir, "final_results.json")
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    final = json.load(f)
                cached_top_key = final.get("protocol_key")
                if cached_top_key is None:
                    self._log(
                        "  ⚠️ 已有 final_results.json 缺少协议指纹（旧版本产物，可能产生于本次"
                        "WCA 记账口径修复之前），拒绝直接复用，将重新校验/运行各阶段"
                    )
                elif cached_top_key != _top_level_protocol_key:
                    self._log(
                        f"  ⚠️ 已有 final_results.json 协议指纹不匹配"
                        f"（缓存={cached_top_key}, 当前={_top_level_protocol_key}），"
                        "拒绝直接复用，将重新校验/运行各阶段"
                    )
                else:
                    self._log(f"  ♻️ {decoupling_scheme} 采样已完成，协议指纹一致，跳过")
                    self.results["final"] = final
                    self._log("  ✓ 已加载已有最终结果")
                    return final
            else:
                self._log("  ⚠️ 状态标记为完成但未找到结果文件，重新运行采样")
        if decoupling_scheme == "dual_lambda":
            _decharge_method = kwargs.get("decharge_method", "pme")
            # 🔑 实际生效的最终收敛门阈值（不是只看用户是否显式传参）——
            # 跟下面 _run_dual_lambda_stage/solve_stage_integrated 调用时解析
            # 出来的默认值必须完全一致，否则协议指纹和真正生效的判据会对不上。
            _final_gate_thresholds = {
                "final_min_ess_ratio": kwargs.get("final_min_ess_ratio", 0.05),
                "final_min_absolute_ess": kwargs.get("final_min_absolute_ess", 50.0),
                "final_min_decorrelated_samples": kwargs.get("final_min_decorrelated_samples", 20),
                "final_max_uncertainty_kJ_mol": kwargs.get("final_max_uncertainty_kJ_mol", 1.0),
            }
            _stage1_protocol_key = self._stage_protocol_key(
                "decharging", potential_type, boresch_params, _decharge_method,
                n_states=stage1_states, dexp_params=dexp_params,
                final_gate_thresholds=_final_gate_thresholds,
            )
            _stage2_protocol_key = self._stage_protocol_key(
                "vanishing", potential_type, boresch_params, _decharge_method,
                n_states=stage2_states, dexp_params=dexp_params,
                final_gate_thresholds=_final_gate_thresholds,
            )
            # 🔑 [预优化缓存范围收窄] λ 路径预优化（跑一次要几个小时）用这份
            # 单独的、更窄的指纹判定缓存有效性，不用上面完整的
            # _stage1_protocol_key/_stage2_protocol_key——否则任何跟预优化无关
            # 的代码修复（窗口修复循环、production checkpoint 续采等）都会
            # 通过完整 code_sha256 连带让预优化缓存失效，见 _preopt_protocol_key
            # 的完整说明。
            _stage1_preopt_key = self._preopt_protocol_key(
                "decharging", potential_type, boresch_params, _decharge_method,
                dexp_params=dexp_params,
            )
            _stage2_preopt_key = self._preopt_protocol_key(
                "vanishing", potential_type, boresch_params, _decharge_method,
                dexp_params=dexp_params,
            )
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
            # [IBS_BIAS_PROTOCOL_VERSION=20] Stage 2 pilot's real measured mean
            # gradient (mean_dU_dlambda_kJ_mol per pilot lambda point) -- set
            # below whichever path optimized_lambdas_2 comes from (cache hit or
            # fresh generation), consumed by _run_stage2_once further down to
            # let run_all_windows TI-warm-start f_k instead of cold-starting at
            # 0.0. Stays None (no warm-start, today's behavior) for stale
            # caches predating this field, or for decharging (unused there).
            stage2_pilot_lambdas = None
            stage2_pilot_mean_dU_dlambda = None

            # === Stage 1: pre-opt + resume check ===
            optimized_lambdas_1 = None
            if resume and os.path.exists(preopt1_file):
                try:
                    with open(preopt1_file, "r") as f:
                        cached = json.load(f)
                    cached_lambdas = cached["lambdas_var"]
                    # 🔑 [P0] n_states 相等只是"缓存态数没变"的一种情况；另一种合法情况
                    # 是本协议下的自动加密（refine_stage_lambda_path_by_overlap）已经把
                    # 态数从最初的猜测值（stage1_states）真实地涨到了更高的值——那是
                    # 花了真实 GPU production 才验证出来的更好起点，不能因为进程重启后
                    # n_states 对不上就当成"不匹配"整个丢弃、逼着从最初的猜测值重来。
                    # 只有当 provenance.source=="auto_repair_by_overlap" 且协议指纹
                    # （potential_type/Boresch/decharge_method/WCA/IBS 偏置协议版本）
                    # 跟当前完全一致时，才信任这份态数已变的缓存并采用它的态数。
                    cached_protocol = cached.get("protocol_key")
                    cached_source = cached.get("provenance", {}).get("source")
                    # 🔑 [P1 修复] 之前只有 is_verified_auto_repair（态数变了的分支）
                    # 才会检查协议指纹，态数恰好没变的分支完全不检查
                    # cached_protocol == _stage1_protocol_key——切换势函数/Boresch/
                    # 体系坐标/softcore 变体/代码版本后，只要凑巧态数相同，就会静默
                    # 复用旧协议优化出的 λ 路径和窗口划分（stage 结果缓存本身仍会
                    # 因指纹不匹配正确重跑，所以不会导致 ΔG 静默算错，但会在错误的
                    # 路径上浪费 GPU、提高 overlap 失败概率）。现在两个分支都先要求
                    # 协议指纹严格一致，态数是否变化只决定"指纹一致后还要不要额外
                    # 检查态数变化本身是否来自已验证的自动加密"。
                    # 🔑 [预优化缓存范围收窄] 这里比对的是 _stage1_preopt_key（窄
                    # 指纹），不是 _stage1_protocol_key（宽指纹，含完整四文件
                    # code_sha256）——预优化缓存的有效性不应该被 stage 采样/
                    # 窗口修复循环相关的代码修复连带作废，见 _preopt_protocol_key。
                    protocol_match = cached_protocol is not None and cached_protocol == _stage1_preopt_key
                    if not protocol_match and self._preopt_cache_matches_ignoring_code_hash(
                        cached_protocol, _stage1_preopt_key
                    ):
                        # 🔑 [预优化缓存 schema 迁移] 缓存是 _preopt_protocol_key 上线
                        # 之前写的旧宽指纹，跟新窄指纹逐字节比较必然不相等——但物理
                        # 输入逐项核对完全一致，判定为纯 schema 迁移，不重新优化，
                        # 原地把 protocol_key 重新盖成新窄指纹（自愈一次即可）。
                        self._log(
                            "  🩹 Stage 1 预优化缓存是旧 schema（_preopt_protocol_key 上线之前写入的"
                            "宽指纹），但物理输入（potential_type/Boresch/温度/压力/坐标/预优化代码本身"
                            "等）逐项核对完全一致——判定为 schema 迁移，不重新优化，原地重盖 protocol_key。"
                        )
                        cached["protocol_key"] = _stage1_preopt_key
                        with open(preopt1_file, "w") as f:
                            json.dump(cached, f, indent=2)
                        protocol_match = True
                    is_verified_auto_repair = (
                        cached_source == "auto_repair_by_overlap"
                        and protocol_match
                    )
                    if protocol_match and (len(cached_lambdas) == stage1_states or is_verified_auto_repair):
                        optimized_lambdas_1 = cached_lambdas
                        if len(cached_lambdas) != stage1_states:
                            self._log(
                                f"  ♻️ Stage 1 缓存态数 ({len(cached_lambdas)}) 与初始请求 "
                                f"({stage1_states}) 不同，但协议指纹一致且来自本协议下已验证的"
                                "自动加密结果——采用缓存态数，而不是丢弃重新优化。"
                            )
                            stage1_states = len(cached_lambdas)
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
                    elif not protocol_match:
                        self._log(
                            "  ⚠️ Stage 1 预优化缓存协议指纹不一致（potential_type/Boresch/"
                            "decharge_method/WCA/IBS 偏置协议版本等已变化），将重新优化完整 "
                            "λ 路径（此前版本会在态数恰好相同时静默复用旧协议下的缓存）。"
                        )
                    else:
                        self._log(
                            f"  ⚠️ Stage 1 优化路径缓存状态数不匹配 "
                            f"({len(cached_lambdas)} != {stage1_states})，且非本协议下已验证的自动加密结果，重新优化"
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
                            # 🔑 [预优化缓存范围收窄] 存窄指纹 _stage1_preopt_key，
                            # 不是宽指纹 _stage1_protocol_key——否则下次加载时会跟
                            # 上面改成比对窄指纹的读取逻辑对不上，写完立刻判定
                            # "不匹配"。
                            "protocol_key": _stage1_preopt_key,
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
                    cached_protocol_1 = stage1.get("protocol_key")
                    if stage1.get("n_states") != stage1_states:
                        self._log("  ⚠️ Stage 1 结果缓存状态数不匹配，重新运行")
                    elif cached_protocol_1 is None:
                        # 🔑 [P0] 之前这里是"信任并跳过"——旧格式缓存(在 protocol_key
                        # 字段存在之前生成，包括 WCA e_base/e_bias 记账口径修复之前的
                        # 所有历史结果)会被直接当成有效结果复用。缺指纹就是没法校验，
                        # 必须 fail closed 视为缓存失效，而不是"大概率没变"。
                        self._log("  ⚠️ Stage 1 结果缓存无协议指纹（旧版本产物），视为缓存失效，重新运行")
                    elif cached_protocol_1 != _stage1_protocol_key:
                        self._log(
                            f"  ⚠️ Stage 1 结果缓存协议指纹不匹配 (缓存={cached_protocol_1}, "
                            f"当前={_stage1_protocol_key})，拒绝静默复用，重新运行"
                        )
                    elif stage1.get("lambda_path_fingerprint") != self._lambda_path_fingerprint(
                        optimized_lambdas_1, window_ranges_1
                    ):
                        self._log("  ⚠️ Stage 1 的 λ 内容/窗口边界指纹不匹配，重新运行")
                    else:
                        self._assert_reusable_stage_cache_sane(
                            "Stage 1 (decharging)", stage1
                        )
                        self._log("  ♻️ 双λ Stage 1 (去电荷) 已完成，跳过")
                        should_run_stage1 = False
                except Exception as e:
                    self._log(f"  ⚠️ Stage 1 缓存读取失败: {e}，重新运行")

            # === Stage 2: pre-opt + resume check ===
            optimized_lambdas_2 = None
            if resume and os.path.exists(preopt2_file):
                try:
                    with open(preopt2_file, "r") as f:
                        cached = json.load(f)
                    cached_lambdas = cached["lambdas_var"]
                    # 🔑 [P0] 同 Stage 1：n_states 不等于最初请求值，不代表缓存无效——
                    # 可能是本协议下的自动加密已经把它真实地涨到了更高的、花了真实
                    # GPU production 验证过的值。只有 provenance.source 属于下方受信
                    # 白名单且协议指纹一致时才信任并采用其态数。白名单曾经只认
                    # "auto_repair_by_overlap"（旧的按 ESS-per-lambda 算术二分插点），
                    # 但 warmup 失败分支现在写出 "fixed_hamiltonian_bidirectional_overlap"、
                    # production ESS 修复分支现在写出
                    # "production_overlap_repair_split_then_probe"——态数变化时这两种
                    # 缓存会被错误当成"未验证"丢弃，逼着已经用真实 GPU production 验证
                    # 过 fixed-H overlap 的插点重新跑一遍。fixed-H 全通过但检测到局部
                    # 热力学瓶颈的分支写出 "fixed_hamiltonian_passed_but_asymmetric_
                    # bottleneck"——同样是真实 GPU production 验证过的插点，遗漏这一条
                    # 会让态数增长后的缓存被判定为"未验证"整体丢弃，逼着从头重新优化。
                    _VERIFIED_STAGE2_REPAIR_SOURCES = {
                        "fisher_metric_blended_with_geometric_floor_v21",
                        "quadratic_geometric_fallback_v21",
                        "human_anchors_no_probe_fallback",
                    }
                    cached_protocol = cached.get("protocol_key")
                    cached_source = cached.get("provenance", {}).get("source")
                    # 🔑 [P1 修复] path_protocol_version 只是 _stage2_protocol_key
                    # 完整指纹里的一个子字段（thermodynamic_path_protocol_version
                    # 已经折叠进 _stage_protocol_key），之前把它单独当顶层门槛，
                    # 态数恰好没变时完整的 cached_protocol == _stage2_protocol_key
                    # 从未被检查——跟 Stage 1 是同一个漏洞。现在顶层门槛改为完整
                    # 协议指纹匹配；path_protocol_version 的检查保留在
                    # is_verified_auto_repair 内部作为额外防御，不再单独作为
                    # 顶层旁路条件（否则相当于重新开一个可以绕过完整指纹检查的
                    # 后门）。
                    path_protocol_match = (
                        cached.get("path_protocol_version")
                        == THERMODYNAMIC_PATH_PROTOCOL_VERSION
                    )
                    anchor_contract_match = False
                    try:
                        validate_vanishing_lambda_path_invariants(cached_lambdas)
                        anchor_contract_match = True
                    except Exception as anchor_exc:
                        self._log(
                            "  ⚠️ Stage 2 缓存违反 v21 vanishing 路径不变量："
                            f"{anchor_exc}"
                        )
                    # 🔑 [预优化缓存范围收窄] 同 Stage 1：比对 _stage2_preopt_key
                    # （窄指纹），不是 _stage2_protocol_key（宽指纹，含完整四文件
                    # code_sha256）——理由同上，见 _preopt_protocol_key。
                    protocol_match = cached_protocol is not None and cached_protocol == _stage2_preopt_key
                    # 🔑 [TEMPORARY, 2026-07-19 深夜] 用户当前没有计算节点排队时间，
                    # 显式要求跳过 Stage 2 指纹检查、直接复用已用
                    # repair_stage2_window0_regroup.py 重新算过分组的缓存，不重跑
                    # ~20 分钟的 pilot。只在显式设置环境变量时生效，默认行为完全不变。
                    # 用完记得 unset，这不是永久修复。
                    if os.environ.get("ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT") == "1":
                        if not protocol_match:
                            self._log(
                                "  🩹 [ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT=1] 临时跳过 Stage 2 "
                                "预优化缓存指纹检查，直接复用磁盘上的 lambdas_var/window_ranges "
                                "——这是显式请求的临时旁路，不是永久行为，用完请 unset。"
                            )
                        protocol_match = True
                    if not protocol_match and self._preopt_cache_matches_ignoring_code_hash(
                        cached_protocol, _stage2_preopt_key
                    ):
                        # 🔑 [预优化缓存 schema 迁移] 同 Stage 1：缓存是旧宽指纹，物理
                        # 输入逐项核对完全一致时判定为纯 schema 迁移，不重新优化，原地
                        # 重盖 protocol_key。
                        self._log(
                            "  🩹 Stage 2 预优化缓存是旧 schema（_preopt_protocol_key 上线之前写入的"
                            "宽指纹），但物理输入（potential_type/Boresch/温度/压力/坐标/预优化代码本身"
                            "等）逐项核对完全一致——判定为 schema 迁移，不重新优化，原地重盖 protocol_key。"
                        )
                        cached["protocol_key"] = _stage2_preopt_key
                        with open(preopt2_file, "w") as f:
                            json.dump(cached, f, indent=2)
                        protocol_match = True
                    is_verified_auto_repair = (
                        cached_source in _VERIFIED_STAGE2_REPAIR_SOURCES
                        and path_protocol_match
                        and protocol_match
                        and anchor_contract_match
                    )
                    if protocol_match and anchor_contract_match and (
                        len(cached_lambdas) == stage2_states or is_verified_auto_repair
                    ):
                        optimized_lambdas_2 = cached_lambdas
                        if len(cached_lambdas) != stage2_states:
                            self._log(
                                f"  ♻️ Stage 2 缓存态数 ({len(cached_lambdas)}) 与初始请求 "
                                f"({stage2_states}) 不同，但协议指纹一致且来自本协议下已验证的"
                                "自动加密结果——采用缓存态数，而不是丢弃重新优化。"
                            )
                            stage2_states = len(cached_lambdas)
                        cached_ranges = cached.get("window_ranges")
                        expected_subdomain_ranges = (
                            vanishing_subdomain_ranges_from_lambdas(
                                cached_lambdas,
                                first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
                            )
                        )
                        normalized_cached_ranges = (
                            [tuple(int(x) for x in r) for r in cached_ranges]
                            if cached_ranges else []
                        )
                        if normalized_cached_ranges != expected_subdomain_ranges:
                            self._log(
                                "  ⚠️ Stage 2 缓存不是 vanishing v12 的热力学 few-state 子区间布局；"
                                "拒绝单一 [0:K]、overlap=2 或滑动窗口缓存并重新优化。"
                            )
                            optimized_lambdas_2 = None
                            window_ranges_2 = None
                        else:
                            window_ranges_2 = expected_subdomain_ranges
                            self._log(
                                f"  ♻️ 已加载 Stage 2 热力学 few-state IBS 子区间 "
                                f"({len(optimized_lambdas_2)} 个状态, ranges={window_ranges_2})"
                            )
                            # [IBS_BIAS_PROTOCOL_VERSION=20] Old caches
                            # predating pilot_points won't have this -- stays
                            # None, estimate_f_k_from_pilot_ti() itself also
                            # treats missing/short data as "no seed".
                            _cached_diag = cached.get("path_diagnostics") or {}
                            stage2_pilot_lambdas = _cached_diag.get("pilot_lambdas")
                            _cached_pilot_points = _cached_diag.get("pilot_points")
                            if _cached_pilot_points:
                                stage2_pilot_mean_dU_dlambda = [
                                    p.get("mean_dU_dlambda_kJ_mol") for p in _cached_pilot_points
                                ]
                    elif not protocol_match:
                        self._log(
                            "  ⚠️ Stage 2 预优化缓存协议指纹不一致（potential_type/Boresch/"
                            "decharge_method/WCA/IBS 偏置协议版本/热力学路径协议版本等已变化），"
                            "将重新优化完整 λ 路径（此前版本会在态数恰好相同时静默复用旧协议下"
                            "的缓存）。"
                        )
                    else:
                        self._log(
                            f"  ⚠️ Stage 2 优化路径缓存状态数不匹配 "
                            f"({len(cached_lambdas)} != {stage2_states})，"
                            "且非本协议下已验证的自动加密结果，重新优化"
                        )
                except Exception as e:
                    self._log(f"  ⚠️ 加载 Stage 2 优化缓存失败: {e}，将重新优化")

            if optimized_lambdas_2 is None:
                try:
                    opt_res = self._run_dual_lambda_optimization(
                        "vanishing",
                        n_states=stage2_states,
                        # 🔑 2026-07-19: window 0 (lambda_vdw->1 端点) 反复
                        # IBSWarmupConvergenceError，诊断发现 pilot 用 10000 步的
                        # 有限差分探针系统性低估了该区域由稀有/发作性事件主导的
                        # beta^2*Var[dU/dlambda]，导致这段真实热力学长度极大的
                        # 区域分不到足够密度。改为可配置并默认拉长，而不是保持
                        # 硬编码的短探针窗口。
                        n_steps_per_state=int(kwargs.get("pilot_n_steps_per_state", 10000)),
                        potential_type=potential_type,
                        finite_difference_delta=kwargs.get("pilot_finite_difference_delta", 0.01),
                    )
                    optimized_lambdas_2 = opt_res["lambdas_var"]
                    validate_vanishing_lambda_path_invariants(optimized_lambdas_2)
                    # [IBS_BIAS_PROTOCOL_VERSION=20] Capture the pilot's real
                    # measured mean gradient alongside its lambda placement,
                    # for run_all_windows to TI-warm-start f_k with later.
                    _fresh_diag = opt_res.get("path_diagnostics") or {}
                    stage2_pilot_lambdas = _fresh_diag.get("pilot_lambdas")
                    _fresh_pilot_points = _fresh_diag.get("pilot_points")
                    if _fresh_pilot_points:
                        stage2_pilot_mean_dU_dlambda = [
                            p.get("mean_dU_dlambda_kJ_mol") for p in _fresh_pilot_points
                        ]
                    # Lambda density is set by measured thermodynamic length;
                    # consecutive thermodynamic intervals are then grouped into
                    # few-state ensembles.  No fixed 0.5 or overlap=2 cut.
                    window_ranges_2 = vanishing_subdomain_ranges_from_lambdas(
                        optimized_lambdas_2,
                        first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
                    )
                    opt_res["window_ranges"] = window_ranges_2
                    opt_res.setdefault("path_diagnostics", {})[
                        "ibs_ensemble_layout"
                    ] = "few_state_thermodynamic_subdomains"
                    opt_res["path_diagnostics"][
                        "schedule_design_probes_used"
                    ] = bool(
                        _fresh_diag.get("probe_controls_base_lambda_placement", False)
                    )
                    opt_res["n_states"] = len(optimized_lambdas_2)
                    _placement_source = _fresh_diag.get(
                        "lambda_placement_method",
                        "unknown_vanishing_lambda_placement",
                    )
                    opt_res["provenance"] = {
                        "source": _placement_source,
                        "fixed_h_overlap_used": False,
                        "overlap_two_windowing_used": False,
                        "shared_endpoint_states_per_neighbor": 1,
                    }
                    stage2_states = len(optimized_lambdas_2)
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                    with open(preopt2_file, "w") as f:
                        json.dump({
                            "lambdas_var": optimized_lambdas_2,
                            "window_ranges": window_ranges_2,
                            "n_states": len(optimized_lambdas_2),
                            # 🔑 [预优化缓存范围收窄] 同 Stage 1：存窄指纹
                            # _stage2_preopt_key，不是宽指纹 _stage2_protocol_key。
                            "protocol_key": _stage2_preopt_key,
                            "path_protocol_version": opt_res.get("path_protocol_version"),
                            "path_diagnostics": opt_res.get("path_diagnostics", {}),
                            "provenance": opt_res.get("provenance", {}),
                        }, f, indent=2)
                    self._log(f"  ✓ Stage 2 优化路径已缓存")
                except Exception as e:
                    raise RuntimeError(f"Stage 2 自适应优化失败，拒绝静默回退线性路径: {e}") from e

            should_run_stage2 = True
            if resume and stage2_status == "completed" and os.path.exists(stage2_file):
                try:
                    with open(stage2_file, "r") as f:
                        stage2 = json.load(f)
                    cached_protocol_2 = stage2.get("protocol_key")
                    if stage2.get("n_states") != stage2_states:
                        self._log("  ⚠️ Stage 2 结果缓存状态数不匹配，重新运行")
                    elif cached_protocol_2 is None:
                        # 🔑 [P0] 同 Stage 1：缺协议指纹一律 fail closed，不再"信任并跳过"。
                        self._log("  ⚠️ Stage 2 结果缓存无协议指纹（旧版本产物），视为缓存失效，重新运行")
                    elif cached_protocol_2 != _stage2_protocol_key:
                        self._log(
                            f"  ⚠️ Stage 2 结果缓存协议指纹不匹配 (缓存={cached_protocol_2}, "
                            f"当前={_stage2_protocol_key})，拒绝静默复用，重新运行"
                        )
                    elif stage2.get("lambda_path_fingerprint") != self._lambda_path_fingerprint(
                        optimized_lambdas_2, window_ranges_2
                    ):
                        self._log("  ⚠️ Stage 2 的 λ 内容/窗口边界指纹不匹配，重新运行")
                    else:
                        self._assert_reusable_stage_cache_sane(
                            "Stage 2 (vanishing)", stage2
                        )
                        self._log("  ♻️ 双λ Stage 2 (去VDW) 已完成，跳过")
                        should_run_stage2 = False
                except Exception as e:
                    self._log(f"  ⚠️ Stage 2 缓存读取失败: {e}，重新运行")

            # === Stage 2: 精修阶段（中等步数探针，基于实测 |Δf| 精修 λ 路径/窗口边界）===
            # 只对 Stage 2 (去VDW/vanishing) 生效：Stage 1 (去电荷) 走的是
            # PME-REMD-MBAR 路径（见 _run_dual_lambda_stage 里 decharging 分支），
            # 不产出 dual_window_*_coul_energies.npy，refine_stage_lambda_path_from_data
            # 无从下手。粗扫(几千步/态) → 精修(中等步数/窗口，本节) → 生产(满步数)，
            # 精修用独立 scratch 目录采样，绝不写入生产目录，避免被生产阶段的 resume
            # 形状校验误判为"已采样完成"而跳过真正的生产步数。
            if kwargs.get("enable_lambda_refine", False):
                raise RuntimeError(
                    "enable_lambda_refine 的旧实现按 |Δf| 重排，会覆盖新的 "
                    "beta^2 Var[dU/dlambda] 双物理子区间路径，并重新引入 "
                    "refine_overlap=2；vanishing v12 明确禁止启用。"
                )

            # === Sampling: parallel or sequential ===
            _parallel_stages = kwargs.get("parallel_stages", False)
            if _parallel_stages:
                self._log(
                    "  ⚠️ 热力学路径协议 v1 需要在父进程捕获结构化 warmup 失败并立即"
                    "反馈重切路径；跨进程异常反馈尚未序列化，当前自动回退串行阶段执行。"
                )
                _parallel_stages = False
            if _decharge_method == "shadow_ibs" and _parallel_stages:
                raise RuntimeError(
                    "decharge_method='shadow_ibs' 暂不支持 --parallel-stages"
                    "（子进程 worker 尚未对接 Shadow-Bridge/Shadow-IBS 两段腿）；"
                    "请去掉 --parallel-stages 后串行运行。"
                )

            if _parallel_stages and should_run_stage1 and should_run_stage2:
                self._log("\n[双λ] 🚀 并行执行 Stage 1 (去电荷) + Stage 2 (去VDW)")
                state_dir = os.path.join(self.checkpoint_dir, "parallel_state")
                self._save_state_to_dir(state_dir)

                _res_dir = os.path.join(self.checkpoint_dir, "parallel_results")
                os.makedirs(_res_dir, exist_ok=True)
                _res1 = os.path.join(_res_dir, "stage1.json")
                _res2 = os.path.join(_res_dir, "stage2.json")
                # ✅ 清空上一轮遗留的 stage1.json/stage2.json：若本轮 worker 子进程
                # 崩溃/被杀而未写出新结果，下面 open(_res1)/open(_res2) 必须直接报
                # FileNotFoundError，而不是静默读到上一次运行的旧结果当作本轮成功。
                for _stale in (_res1, _res2):
                    if os.path.exists(_stale):
                        os.remove(_stale)

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
                    min_bias_updates=kwargs.get("min_bias_updates", 12),
                    max_bias_updates=kwargs.get("max_bias_updates", 50),
                    required_consecutive_bias_updates=kwargs.get(
                        "required_consecutive_bias_updates", 3
                    ),
                    max_bias_warmup_steps=kwargs.get("max_bias_warmup_steps", 500000),
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
                              optimized_lambdas_1, window_ranges_1, _common["enable_early_stop"],
                              _common["boresch_params"], _common["enable_gradual_warmup"],
                              _common["warmup_steps"], _common["min_bias_updates"],
                              _common["max_bias_updates"], _common["required_consecutive_bias_updates"],
                              _common["max_bias_warmup_steps"], _common["resume"], _res1),
                    )
                    p2 = ctx.Process(
                        target=_run_stage_worker_process,
                        args=(state_dir, _temp_k, stage2_platform, self.output_dir,
                              "vanishing", 0.0, 1.0,
                              _common["n_states_stage2"], _common["n_steps_per_window"],
                              _common["steps_per_update"], _common["system_type"],
                              _common["potential_type"], _common["dexp_params"],
                              optimized_lambdas_2, window_ranges_2, _common["enable_early_stop"],
                              _common["boresch_params"], _common["enable_gradual_warmup"],
                              _common["warmup_steps"], _common["min_bias_updates"],
                              _common["max_bias_updates"], _common["required_consecutive_bias_updates"],
                              _common["max_bias_warmup_steps"], _common["resume"], _res2),
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
                        optimized_lambdas_1, window_ranges_1, _common["enable_early_stop"],
                        _common["boresch_params"], _common["enable_gradual_warmup"],
                        _common["warmup_steps"], _common["min_bias_updates"],
                        _common["max_bias_updates"], _common["required_consecutive_bias_updates"],
                        _common["max_bias_warmup_steps"], _common["resume"], _res1,
                    )
                    _run_stage_worker_process(
                        state_dir, _temp_k, stage2_platform, self.output_dir,
                        "vanishing", 0.0, 1.0,
                        _common["n_states_stage2"], _common["n_steps_per_window"],
                        _common["steps_per_update"], _common["system_type"],
                        _common["potential_type"], _common["dexp_params"],
                        optimized_lambdas_2, window_ranges_2, _common["enable_early_stop"],
                        _common["boresch_params"], _common["enable_gradual_warmup"],
                        _common["warmup_steps"], _common["min_bias_updates"],
                        _common["max_bias_updates"], _common["required_consecutive_bias_updates"],
                        _common["max_bias_warmup_steps"], _common["resume"], _res2,
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
                _s1 = self._build_stage_cache_payload(
                    "decharging", stage1, stage1_states, _stage1_protocol_key,
                    optimized_lambdas_1, window_ranges_1,
                )
                with open(stage1_file, "w") as f:
                    json.dump(_s1, f, indent=2)
                self._update_stage_status(stage1_key, "completed",
                                          {"total_delta_G": stage1["total_delta_G"]})

                self._assert_stage_result_sane("Stage 2 (vanishing)", stage2)
                _s2 = self._build_stage_cache_payload(
                    "vanishing", stage2, stage2_states, _stage2_protocol_key,
                    optimized_lambdas_2, window_ranges_2,
                )
                with open(stage2_file, "w") as f:
                    json.dump(_s2, f, indent=2)
                self._update_stage_status(stage2_key, "completed",
                                          {"total_delta_G": stage2["total_delta_G"]})

            else:
                # === Sequential execution ===
                if should_run_stage1:
                    self._log("\n[双λ] Stage 1: 去电荷 (λ_coul: 1→0, λ_vdw=1)")

                    def _run_stage1_once(_n_states, _lambdas, _ranges, _production_step_overrides=None,
                                          _frozen_validation_step_overrides=None,
                                          _frozen_validation_is_final_rung=None):
                        # decharging has no probe_window_overlap_fn / sampling-repair /
                        # IBS-bias-calibration branch, so _production_step_overrides and
                        # _frozen_validation_step_overrides/_frozen_validation_is_final_rung
                        # are always None here; accepted (and ignored) only so run_once()
                        # can be called with a uniform signature regardless of stage.
                        return self._run_dual_lambda_stage(
                            "decharging",
                            fixed_lam_coul=1.0,
                            fixed_lam_vdw=1.0,
                            potential_type=potential_type,
                            dexp_params=dexp_params,
                            n_states=_n_states,
                            n_steps_per_window=n_steps_per_window,
                            steps_per_update=steps_per_update,
                            system_type=system_type,
                            resume=resume,
                            optimized_lambdas=_lambdas,
                            window_ranges=_ranges,
                            enable_early_stop=enable_early_stop,
                            boresch_params=boresch_params,
                            enable_gradual_warmup=kwargs.get("enable_gradual_warmup", True),
                            warmup_steps=kwargs.get("warmup_steps", 500000),
                            min_bias_updates=kwargs.get("min_bias_updates", 12),
                            max_bias_updates=kwargs.get("max_bias_updates", 50),
                            required_consecutive_bias_updates=kwargs.get(
                                "required_consecutive_bias_updates", 3
                            ),
                            max_bias_warmup_steps=kwargs.get("max_bias_warmup_steps", 500000),
                            decharge_method=_decharge_method,
                            shadow_bridge_lambdas=kwargs.get("shadow_bridge_lambdas"),
                            shadow_bridge_n_steps=kwargs.get("shadow_bridge_n_steps", 200000),
                            shadow_bridge_exchange_interval=kwargs.get("shadow_bridge_exchange_interval", 1000),
                        )

                    stage1, stage1_states, optimized_lambdas_1, window_ranges_1 = (
                        self._run_stage_with_overlap_autorepair(
                            "Stage 1 (decharging)",
                            "decharging",
                            preopt1_file,
                            stage1_states,
                            optimized_lambdas_1,
                            window_ranges_1,
                            _run_stage1_once,
                            protocol_key=_stage1_protocol_key,
                            # 🔑 [defense-in-depth] 5->8：批量插边(§4)/消除
                            # already_good 饥饿(§5) 修好之后，理论上不该再需要
                            # 靠加大轮次上限硬扛——但一条严重碎片化的 λ 路径仍
                            # 可能需要超过 5 轮才能彻底稳定，这里只是留个安全
                            # 余量，不是替代上面两处真正的修复（用户明确否决了
                            # 直接调到 10-15：那只会在真正的 bug 上继续烧 GPU）。
                            max_repair_rounds=kwargs.get("max_overlap_repair_rounds", 8),
                            preopt_protocol_key=_stage1_preopt_key,
                        )
                    )
                    stage1_save = self._build_stage_cache_payload(
                        "decharging", stage1, stage1_states, _stage1_protocol_key,
                        optimized_lambdas_1, window_ranges_1,
                    )
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

                    def _run_stage2_once(_n_states, _lambdas, _ranges, _production_step_overrides=None,
                                          _frozen_validation_step_overrides=None,
                                          _frozen_validation_is_final_rung=None,
                                          _resume_override=None):
                        return self._run_dual_lambda_stage(
                            "vanishing",
                            fixed_lam_coul=0.0,
                            fixed_lam_vdw=1.0,
                            potential_type=potential_type,
                            dexp_params=dexp_params,
                            n_states=_n_states,
                            n_steps_per_window=n_steps_per_window,
                            steps_per_update=steps_per_update,
                            system_type=system_type,
                            resume=(
                                resume
                                if _resume_override is None
                                else bool(_resume_override)
                            ),
                            optimized_lambdas=_lambdas,
                            window_ranges=_ranges,
                            enable_early_stop=enable_early_stop,
                            boresch_params=boresch_params,
                            enable_gradual_warmup=kwargs.get("enable_gradual_warmup", True),
                            warmup_steps=kwargs.get("warmup_steps", 500000),
                            min_bias_updates=kwargs.get("min_bias_updates", 12),
                            max_bias_updates=kwargs.get("max_bias_updates", 50),
                            required_consecutive_bias_updates=kwargs.get(
                                "required_consecutive_bias_updates", 3
                            ),
                            max_bias_warmup_steps=kwargs.get("max_bias_warmup_steps", 500000),
                            ibs_lse_log_residual_tolerance=kwargs.get(
                                "ibs_lse_log_residual_tolerance", 0.5
                            ),
                            production_step_overrides=_production_step_overrides,
                            frozen_validation_step_overrides=_frozen_validation_step_overrides,
                            frozen_validation_is_final_rung=_frozen_validation_is_final_rung,
                            pilot_lambdas=stage2_pilot_lambdas,
                            pilot_mean_dU_dlambda=stage2_pilot_mean_dU_dlambda,
                            final_min_ess_ratio=kwargs.get("final_min_ess_ratio", 0.05),
                            final_min_absolute_ess=kwargs.get("final_min_absolute_ess", 50.0),
                            final_min_decorrelated_samples=kwargs.get(
                                "final_min_decorrelated_samples", 20
                            ),
                            final_max_uncertainty_kJ_mol=kwargs.get(
                                "final_max_uncertainty_kJ_mol", 1.0
                            ),
                        )

                    expected_vanishing_ranges = (
                        vanishing_subdomain_ranges_from_lambdas(
                            optimized_lambdas_2,
                            first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
                        )
                    )
                    validate_vanishing_lambda_path_invariants(
                        optimized_lambdas_2
                    )
                    validate_single_shared_boundary_ranges(
                        expected_vanishing_ranges, len(optimized_lambdas_2)
                    )
                    normalized_vanishing_ranges = [
                        tuple(int(x) for x in r) for r in (window_ranges_2 or [])
                    ]
                    if normalized_vanishing_ranges != expected_vanishing_ranges:
                        raise RuntimeError(
                            "vanishing v12 要求热力学 few-state IBS 子区间: "
                            f"expected={expected_vanishing_ranges}, got={normalized_vanishing_ranges}. "
                            "拒绝共享两个节点（从而重复一条 λ interval）的 legacy overlap=2、"
                            "滑动窗口或单一 [0:K] ensemble。"
                        )
                    vanishing_edge_ranges = [
                        (int(start), int(end) - 1)
                        for start, end in expected_vanishing_ranges
                    ]
                    vanishing_boundary_nodes = [
                        int(expected_vanishing_ranges[i][0])
                        for i in range(1, len(expected_vanishing_ranges))
                    ]
                    self._log(
                        "  🧩 Vanishing few-state："
                        f"λ interval 分区（半开、互不重复）={vanishing_edge_ranges}；"
                        f"各 ensemble 使用的 λ 节点范围（半开）={expected_vanishing_ranges}；"
                        f"共同边界节点={vanishing_boundary_nodes}。"
                        "边界节点会在相邻 ensemble 中各出现一次以对齐自由能参考；"
                        "未使用固定 λ=0.5，也未使用会重复一条 λ interval 的 legacy overlap=2，"
                        "基础 ensemble 不原地拆窗/插点；若生产 coverage 补采仍失败，"
                        "只会在独立目录新建使用现有 λ 节点的 rescue ensembles"
                    )
                    stage2 = _run_stage2_once(
                        stage2_states,
                        optimized_lambdas_2,
                        expected_vanishing_ranges,
                    )
                    # Production-quality rescue is deliberately separate from
                    # warmup: only the failing production windows are extended,
                    # from their existing production checkpoint, under the same
                    # already-frozen f_k.  Good windows remain cache hits.  This
                    # never changes the lambda grid or carries warmup frames into
                    # production.
                    production_rescue_targets: Dict[int, int] = {}
                    production_rescue_rounds = max(
                        0, int(kwargs.get("stage2_production_rescue_rounds", 2))
                    )
                    production_rescue_growth = max(
                        1.1, float(kwargs.get("stage2_production_rescue_growth", 2.0))
                    )
                    for rescue_round in range(1, production_rescue_rounds + 1):
                        if stage2.get("converged") is True:
                            break
                        failure_details = self._stage_quality_failure_details(stage2)
                        failing_windows = sorted({
                            int(item["window_index"])
                            for item in failure_details
                            if 0 <= int(item.get("window_index", -1))
                            < len(expected_vanishing_ranges)
                        })
                        if not failing_windows:
                            break
                        for failing_window in failing_windows:
                            old_target = int(
                                production_rescue_targets.get(
                                    failing_window, n_steps_per_window
                                )
                            )
                            production_rescue_targets[failing_window] = int(
                                math.ceil(old_target * production_rescue_growth)
                            )
                        self._log(
                            f"  🛟 Stage 2 production coverage rescue round "
                            f"{rescue_round}/{production_rescue_rounds}: 仅追加窗口 "
                            f"{failing_windows}，累计生产目标={production_rescue_targets}。"
                            "沿用各窗口 production checkpoint 与已锁定 f_k；不重新学习、"
                            "不修改 f_k、不丢弃已有生产帧。当前瓶颈："
                            f"{self._format_stage_quality_failure_details(failure_details)}"
                        )
                        stage2 = _run_stage2_once(
                            stage2_states,
                            optimized_lambdas_2,
                            expected_vanishing_ranges,
                            _production_step_overrides=dict(production_rescue_targets),
                            _resume_override=True,
                        )
                    stage2["production_rescue_targets"] = dict(
                        production_rescue_targets
                    )
                    # If extra samples do not improve an ESS *ratio*, the
                    # bottleneck is usually the ensemble span rather than raw
                    # frame count.  Build new, smaller overlapping ensembles on
                    # the existing immutable lambda grid in a separate rescue
                    # directory.  Original production files remain untouched;
                    # the failed ensemble is replaced only in the combined
                    # analysis cover.
                    if (
                        stage2.get("converged") is not True
                        and bool(kwargs.get("stage2_enable_bridge_rescue", True))
                    ):
                        failure_details = self._stage_quality_failure_details(stage2)
                        failing_windows = sorted({
                            int(item["window_index"])
                            for item in failure_details
                            if 0 <= int(item.get("window_index", -1))
                            < len(expected_vanishing_ranges)
                        })
                        rescue_ranges = self._build_vanishing_rescue_ranges(
                            failing_windows, expected_vanishing_ranges
                        )
                        if rescue_ranges:
                            rescue_plan_id = hashlib.sha256(
                                json.dumps(rescue_ranges, sort_keys=True).encode("utf-8")
                            ).hexdigest()[:12]
                            rescue_output_dir = os.path.join(
                                self.output_dir,
                                "vanishing_rescue",
                                rescue_plan_id,
                            )
                            rescue_checkpoint_dir = os.path.join(
                                self.checkpoint_dir,
                                "vanishing_rescue",
                                rescue_plan_id,
                            )
                            rescue_steps = int(
                                kwargs.get(
                                    "stage2_bridge_production_steps",
                                    n_steps_per_window,
                                )
                            )
                            self._log(
                                "  🌉 Stage 2 追加采样后仍未通过：保留原生产数据，"
                                f"为失败窗口 {failing_windows} 新建 immutable rescue "
                                f"ensembles={rescue_ranges}（仅使用现有 λ 节点，"
                                f"{rescue_steps} production steps/ensemble）。"
                                "新 ensemble 各自预热一次后锁定自己的 f_k；生产阶段"
                                "仍禁止修改 f_k。"
                            )
                            self._run_dual_lambda_stage(
                                "vanishing_rescue",
                                fixed_lam_coul=0.0,
                                fixed_lam_vdw=1.0,
                                potential_type=potential_type,
                                dexp_params=dexp_params,
                                n_states=stage2_states,
                                n_steps_per_window=rescue_steps,
                                steps_per_update=steps_per_update,
                                system_type=system_type,
                                resume=True,
                                optimized_lambdas=optimized_lambdas_2,
                                window_ranges=rescue_ranges,
                                enable_early_stop=enable_early_stop,
                                boresch_params=boresch_params,
                                enable_gradual_warmup=kwargs.get(
                                    "enable_gradual_warmup", True
                                ),
                                warmup_steps=kwargs.get("warmup_steps", 500000),
                                min_bias_updates=kwargs.get("min_bias_updates", 12),
                                max_bias_updates=kwargs.get("max_bias_updates", 50),
                                required_consecutive_bias_updates=kwargs.get(
                                    "required_consecutive_bias_updates", 3
                                ),
                                max_bias_warmup_steps=kwargs.get(
                                    "max_bias_warmup_steps", 500000
                                ),
                            ibs_lse_log_residual_tolerance=kwargs.get(
                                "ibs_lse_log_residual_tolerance", 0.5
                                ),
                                pilot_lambdas=stage2_pilot_lambdas,
                                pilot_mean_dU_dlambda=stage2_pilot_mean_dU_dlambda,
                                final_min_ess_ratio=kwargs.get(
                                    "final_min_ess_ratio", 0.05
                                ),
                                final_min_absolute_ess=kwargs.get(
                                    "final_min_absolute_ess", 50.0
                                ),
                                final_min_decorrelated_samples=kwargs.get(
                                    "final_min_decorrelated_samples", 20
                                ),
                                final_max_uncertainty_kJ_mol=kwargs.get(
                                    "final_max_uncertainty_kJ_mol", 1.0
                                ),
                                allow_partial_vanishing_rescue=True,
                                stage_output_dir_override=rescue_output_dir,
                                checkpoint_dir_override=rescue_checkpoint_dir,
                            )

                            full_lambdas_coul = [0.0] * len(optimized_lambdas_2)
                            original_outputs = self._load_ibs_window_outputs_from_dir(
                                os.path.join(self.output_dir, "vanishing"),
                                expected_vanishing_ranges,
                                full_lambdas_coul,
                                optimized_lambdas_2,
                                checkpoint_dir=self.checkpoint_dir,
                                excluded_local_windows=set(failing_windows),
                                window_label_prefix="original_window",
                            )
                            rescue_outputs = self._load_ibs_window_outputs_from_dir(
                                rescue_output_dir,
                                rescue_ranges,
                                full_lambdas_coul,
                                optimized_lambdas_2,
                                checkpoint_dir=rescue_checkpoint_dir,
                                window_index_offset=10_000,
                                window_label_prefix="rescue_window",
                            )
                            combined_outputs = original_outputs + rescue_outputs
                            stage2 = solve_stage_integrated(
                                window_outputs=combined_outputs,
                                kt=(
                                    unit.MOLAR_GAS_CONSTANT_R * self.temperature
                                ).value_in_unit(unit.kilojoule_per_mole),
                                stage_name="vanishing",
                                final_min_ess_ratio=kwargs.get(
                                    "final_min_ess_ratio", 0.05
                                ),
                                final_min_absolute_ess=kwargs.get(
                                    "final_min_absolute_ess", 50.0
                                ),
                                final_min_decorrelated_samples=kwargs.get(
                                    "final_min_decorrelated_samples", 20
                                ),
                                final_max_uncertainty_kJ_mol=kwargs.get(
                                    "final_max_uncertainty_kJ_mol", 1.0
                                ),
                            )
                            stage2["production_rescue_targets"] = dict(
                                production_rescue_targets
                            )
                            stage2["immutable_bridge_rescue"] = {
                                "replaced_original_windows_in_analysis": failing_windows,
                                "rescue_ranges": [list(x) for x in rescue_ranges],
                                "output_dir": rescue_output_dir,
                                "plan_id": rescue_plan_id,
                            }
                            stage2["lambda_endpoint_diagnostics"] = (
                                _stage_lambda_endpoint_diagnostics(
                                    "vanishing",
                                    full_lambdas_coul,
                                    optimized_lambdas_2,
                                )
                            )
                            stage2["n_states"] = int(stage2_states)
                            # 🔑 [P1-15] 这条路径绕过了 _run_ibs_stage，此前从不填
                            # diagnostics，导致合并结果落盘时 diagnostics={}，
                            # 逐段 ΔG/σ、converged、ESS 与 rescue provenance 全部丢失。
                            self._populate_stage_diagnostics(stage2)
                            self._log(
                                "  🧮 Stage 2 rescue 合并求解完成："
                                f"ΔG={stage2.get('total_delta_G', float('nan')):.4f} ± "
                                f"{stage2.get('total_error', float('nan')):.4f} kJ/mol，"
                                f"converged={stage2.get('converged')}；"
                                f"被 rescue 取代的原始窗口={failing_windows}，"
                                f"rescue ranges={rescue_ranges}；"
                                f"min_overlap={stage2.get('min_overlap')}，"
                                f"min_occupancy_normalized={stage2.get('min_occupancy_normalized')}，"
                                f"min_decorrelated_samples={stage2.get('min_decorrelated_samples')}，"
                                f"max_endpoint_uncertainty="
                                f"{stage2.get('max_endpoint_uncertainty_kJ_mol')} kJ/mol"
                            )
                            for _seg in (stage2.get("covariance_chain_segments") or []):
                                self._log(
                                    "     · 段 src_window="
                                    f"{_seg.get('source_window_index', _seg.get('window_index'))} "
                                    f"λ[{_seg.get('join_lambda_index')}→{_seg.get('end_lambda_index')}] "
                                    f"ΔG={_seg.get('delta_G_kJ_mol'):.4f} "
                                    f"σ={_seg.get('uncertainty_kJ_mol'):.4f} kJ/mol"
                                )
                    window_ranges_2 = expected_vanishing_ranges
                    # Direct few-state subdomain execution bypasses the retired
                    # overlap-autorepair wrapper, so retain its non-mutating
                    # postcondition gate explicitly.
                    self._assert_stage_result_sane("Stage 2 (vanishing)", stage2)
                    stage2_save = self._build_stage_cache_payload(
                        "vanishing", stage2, stage2_states, _stage2_protocol_key,
                        optimized_lambdas_2, window_ranges_2,
                    )
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
            # 🔑 落盘的 final_results.json 必须带上本次运行的协议指纹，否则下次
            # --resume 时最上层的 "status==completed" 早退检查（见上面）无从校验，
            # 只能保守地拒绝复用——这里补写，就能在协议不变的情况下正常复用。
            final["protocol_key"] = _build_top_level_protocol_key()
            with open(os.path.join(self.output_dir, "final_results.json"), "w") as f:
                json.dump(final, f, indent=2, cls=NumpyEncoder)
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
            final["protocol_key"] = _build_top_level_protocol_key()
            with open(os.path.join(self.output_dir, "final_results.json"), "w") as f:
                json.dump(final, f, indent=2, cls=NumpyEncoder)
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
            final["protocol_key"] = _build_top_level_protocol_key()
            with open(os.path.join(self.output_dir, "final_results.json"), "w") as f:
                json.dump(final, f, indent=2, cls=NumpyEncoder)
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
            final["protocol_key"] = _build_top_level_protocol_key()
            with open(os.path.join(self.output_dir, "final_results.json"), "w") as f:
                json.dump(final, f, indent=2, cls=NumpyEncoder)
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
        u_kn_meta_path = u_kn_path + ".meta.json"
        remd_meta_path = os.path.join(stage_output_dir, f"{stage_name}_remd.meta.json")
        sampling_fingerprint = _protocol_fingerprint({
            "kind": "traditional_remd_sampling",
            "stage_name": stage_name,
            "system_xml_sha256": _system_xml_hash(self.system),
            "topology_sha256": _topology_hash(self.topology),
            "initial_positions_sha256": _positions_hash(self.positions),
            "ligand_indices": [int(i) for i in self.ligand_indices],
            "lambdas_coul": _lambda_signature(lambdas_coul),
            "lambdas_vdw": _lambda_signature(lambdas_vdw),
            "temperature_K": float(self.temperature),
            "n_steps": int(n_steps),
            "exchange_interval": int(exchange_interval),
            "boresch_params": boresch_params,
            "potential_type": potential_type,
            "code_sha256": _code_hash(),
        })
        analysis_fingerprint = _protocol_fingerprint({
            "kind": "traditional_mbar_u_kn",
            "sampling_fingerprint_sha256": sampling_fingerprint["sha256"],
            "traditional_lj_lrc_protocol_version": (
                TRADITIONAL_LJ_LRC_PROTOCOL_VERSION
            ),
            "pme_decharge_model_version": PME_DECHARGE_MODEL_VERSION,
        })

        if resume and os.path.exists(u_kn_path):
            cached_meta = None
            if os.path.exists(u_kn_meta_path):
                try:
                    with open(u_kn_meta_path, "r", encoding="utf-8") as handle:
                        cached_meta = json.load(handle)
                except Exception as exc:
                    print(f"  ⚠️ u_kn 元数据不可读 ({exc})，拒绝复用")
            if (
                cached_meta is not None
                and cached_meta.get("analysis_fingerprint") == analysis_fingerprint
                and os.path.exists(n_k_path)
            ):
                u_kn = np.load(u_kn_path)
                n_k = np.load(n_k_path)
                if (
                    u_kn.ndim == 2
                    and u_kn.shape[0] == len(lambdas_coul)
                    and int(np.sum(n_k)) == u_kn.shape[1]
                    and np.all(np.isfinite(u_kn))
                ):
                    print("  ♻️ u_kn 完整协议指纹一致，跳过 REMD 与能量重算")
                    analyzer = TraditionalMBARAnalyzer(temperature=self.temperature)
                    analyzer._last_n_k = n_k
                    analyzer._last_lj_lrc_metadata = cached_meta.get(
                        "traditional_lj_lrc", {}
                    )
                    result = analyzer.solve(u_kn)
                    result.setdefault("diagnostics", {})["traditional_lj_lrc"] = (
                        analyzer._last_lj_lrc_metadata
                    )
                    return result
                print("  ⚠️ u_kn/n_k 内容或形状无效，拒绝复用")
            else:
                print("  ⚠️ u_kn 缺少或不匹配完整协议指纹，拒绝复用")

        expected_frames = max(1, _expected_remd_frame_count(n_steps))
        remd_fingerprint_matches = False
        if resume and os.path.exists(remd_meta_path):
            try:
                with open(remd_meta_path, "r", encoding="utf-8") as handle:
                    remd_fingerprint_matches = (
                        json.load(handle).get("sampling_fingerprint")
                        == sampling_fingerprint
                    )
            except Exception as exc:
                print(f"  ⚠️ REMD 元数据不可读 ({exc})，拒绝复用轨迹")
        if resume and remd_fingerprint_matches and _all_remd_trajs_valid(
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
            with open(remd_meta_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {"sampling_fingerprint": sampling_fingerprint},
                    handle,
                    indent=2,
                    ensure_ascii=False,
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
        with open(u_kn_meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "analysis_fingerprint": analysis_fingerprint,
                    "sampling_fingerprint": sampling_fingerprint,
                    "traditional_lj_lrc": analyzer._last_lj_lrc_metadata,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        result = analyzer.solve(u_kn)
        result.setdefault("diagnostics", {})["traditional_lj_lrc"] = (
            analyzer._last_lj_lrc_metadata
        )
        return result

    def run_full(
        self,
        n_lambda: int = 12,
        n_steps_per_leg: int = 500000,
        boresch_correction: float = 0.0,
        boresch_params: Optional[Dict] = None,
        potential_type: str = "softcore",
        resume: bool = False,
    ) -> Dict:
        lambdas_coul = np.linspace(1.0, 0.0, n_lambda).tolist()
        lambdas_vdw = [1.0] * n_lambda
        res_coul = self.run_leg(
            "decharging",
            lambdas_coul,
            lambdas_vdw,
            n_steps_per_leg,
            resume=resume,
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
            resume=resume,
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
            # ✅ dg_total = dg_leg + boresch_correction：本函数的 delta_G_total_kJ_mol
            # 恒已把传入的 boresch_correction 加总在内。调用方（如 runabfe.py 的
            # traditional 模式）若显式传 boresch_correction=0.0 并在外层单独扣减
            # Boresch，则本文件里的 boresch_correction_kJ_mol 会是 0，如实反映"未包含"；
            # 但只要调用方传入非零值，此标记即为 True，不应再对 delta_G_total_kJ_mol 二次扣减。
            "boresch_correction_already_included_in_total_delta_G": True,
            "delta_G_total_kJ_mol": dg_total,
            "delta_G_total_kcal_mol": dg_total / 4.184,
        }
        with open(os.path.join(self.output_dir, "final_results.json"), "w") as f:
            json.dump(final, f, indent=2)
        print(f"\n✅ 传统腿完成 | ΔG_leg = {dg_total:.2f} ± {err_leg:.2f} kJ/mol")
        return final
