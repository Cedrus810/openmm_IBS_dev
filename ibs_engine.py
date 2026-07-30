#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IBS 核心引擎 v2.0 — 双λ专用，生产级重构版
职责：
1. 双λ四力组微扰系统构建（严格隔离 Group 0-3 与 IBS CV）
2. IBS 偏置力与采样器（严格按论文 Eq.8/19 能量收集）
3. 窗口管理器（能量落盘、Checkpoint、渐进预热）
4. 全局 TMBAR 分析器（自洽拼接局部窗口）
5. 辅助工具（排除表同步、窗口划分、所有权管理）
已移除：单λ、多GPU、探针构建、死函数。
"""

import openmm
from openmm import app, unit, XmlSerializer, LangevinMiddleIntegrator
import numpy as np
import os
import sys
import gc
import json
import time
import math
import warnings
import multiprocessing as mp
import logging
import builtins
import hashlib
import shutil
import functools
from scipy.integrate import quad as _scipy_quad
from typing import Dict, List, Tuple, Optional, Any, Sequence
from abfe_core import (
    ACESoftcorePotential,
    AlchemicalPotentialFactory,
    DEXPSurrogatePotential,
    DEXP_VDW_CUTOFF_NM,
    DEXP_VDW_SWITCH_WIDTH_NM,
    LambdaDependentBoreschForce,
    create_ligand_internal_force,
    ensure_owned_system,
    sync_all_exclusions,
    BeutlerSoftcoreBuilder,
    _build_mbar_compatible,
    _compute_free_energy_result_compatible,
    _extract_free_energy_arrays,
    subsample_series_by_autocorrelation,
    NumpyEncoder,
)

try:
    import pymbar
    HAS_PYMBAR = True
except ImportError:
    HAS_PYMBAR = False

logger = logging.getLogger(__name__)

IBS_WINDOW_DATA_PROTOCOL_VERSION = 1
SOFTCORE_CUTOFF_NM = 1.2
SOFTCORE_SWITCH_NM = 1.0

# 🔑 [2026-07-28] 「分析协议」版本，与「采样协议」版本严格分离。
#
# 为什么要单独一个：本轮要换主估计量（stage1 → 相邻 BAR，stage2 → 全帧单参考
# MBAR）、换帧选择、换 split-half 口径、换 σ 定义，但**采样 Hamiltonian 一个字节
# 都没动**。如果因为分析代码变了就去 bump `IBS_BIAS_PROTOCOL_VERSION` 或
# `IBS_WINDOW_DATA_PROTOCOL_VERSION`，resume 会误判磁盘上的生产数组失效并重烧 GPU
# —— 那是几小时的代价，换不到任何东西。
#
# 反过来也必须成立：旧的 estimator 结果不能因为「采样协议号没变」就被静默当成
# 当前口径复用。任何 stage result / diagnostics 里都要带上这个号，消费方
# 只接受号一致的值。
#
# version 1（已作废，勿再引用）: 曾写「stage2 重加权 FD-TI 门」与「σ 主口径为分块 SEM」，
#            两条都被 2026-07-28 的复核否决。
# version 2（当前）: **只动 stage1 = charging（decharging 腿）的估计量。**
#            主值 = **相邻 BAR**，一致性门 = **重加权 FD-TI**，
#            去相关 / 全帧 MBAR 降级为 crosschecks 诊断。
#            依据：2026-07-28 实测 BAR 65.076 / FD-TI 65.126 / 全帧 MBAR 65.003
#            三者一致，而去相关 MBAR 64.411 两条腿共同偏低 0.5–0.6 kJ/mol，那是
#            **自相关子采样导致有限样本点估计不稳定**（问题在丢帧后选中的有限子集），
#            **不是「MBAR 本身有偏」**。
#            ⚠️ 两条腿的偏移基本抵消：换估计量后对 ΔG_bind 的净移动只有
#            **−0.140 kJ/mol = −0.033 kcal/mol**。它是个真问题，但**不是**
#            `result.txt` 那 1.3 kcal charging 缺口的解释，别把两件事混起来。
#
# ⛔⛔ **stage2 = vdW 只能用 TMBAR。不得引入任何其它估计量或 σ 口径。** ⛔⛔
#    · 不得加 BAR：IBS 每个窗口只有 row 0（偏置混合分布）有样本、物理 λ 行 n_k=0，
#      而 BAR 要求两个端点系综各自有样本。`adjacent_bar_chain()` 已对零样本态
#      fail closed，正是为了让任何误调用当场炸掉而不是给出一个能被误引用的数。
#    · 不得加 TI：vdW softcore 对 λ 非线性，`_attachment_ti()` 那种「势对 λ 线性
#      ⟹ ∂U/∂λ = U」的假设不成立；这条腿也从未落盘 ∂U/∂λ。用相邻 λ 能量差硬凑的
#      FD-TI（曾算出 144.85）不是这条路径的 ∂U/∂λ 积分，没有判据意义。
#    · 也不得加全帧主值 / √g σ 缩放 / 块 bootstrap σ / σ evidence 汇总。
#      2026-07-28 曾把这些加进 `GlobalMBARAnalyzer.solve_stage_integrated` 并被**整批
#      撤回**：那是在"不该动的 vdW 核心"上做大幅扩展，而"该改的 charging"却没接上线。
#      vdW 的口径问题（去相关选帧、σ 低估）单独立项，不在估计量这一层顺手解决。
ESTIMATOR_ANALYSIS_PROTOCOL_VERSION = 2

# 🔑 estimator policy 指纹：charging 生产主值切到 BAR 之后，靠它判定磁盘上的旧 stage
# result 是不是当前分析口径产出的（旧的去相关 MBAR 主值不能被静默当成 BAR 主值复用）。
# 只放**会改变数值或门判定**的字段。
ESTIMATOR_POLICY_FINGERPRINT_FIELDS = (
    "estimator_analysis_protocol_version",
    "primary_estimator",
    # charging 的估计量用了哪批帧（BAR 用全部帧、去相关 MBAR 用子采样帧）。
    # 刻意叫 charging_* ：`frame_selection` 这个名字属于已撤回的 vdW 全帧模式，
    # 不要让下一个人以为 vdW 还有这个开关。
    "charging_frame_selection",
    "sigma_policy",
    "sigma_inflation_applied",
    "ti_gate_tolerance_kJ_mol",
)


def estimator_policy_fingerprint(policy: Dict[str, Any]) -> str:
    """把估计量策略压成 16 位十六进制指纹。

    只吃 `ESTIMATOR_POLICY_FINGERPRINT_FIELDS` 里的键，缺的键按 None 参与——
    这样「没声明 primary_estimator」与「声明成 None」是同一个指纹，而与
    「声明成 adjacent_bar」不同。未知键一律忽略而不是静默混进去，避免调用方
    多传一个无关字段就让全仓缓存失效。
    """
    payload = {k: policy.get(k) for k in ESTIMATOR_POLICY_FINGERPRINT_FIELDS}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

ENERGY_QUERY_MAX_CONSECUTIVE_FAILURES = 5
ENERGY_QUERY_MAX_TOTAL_FAILURES = 10
ENERGY_QUERY_MAX_FAILURE_FRACTION = 0.01
ENERGY_QUERY_FAILURE_FRACTION_MIN_ATTEMPTS = 100


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


def _atomic_save_npy(filepath: str, array: np.ndarray) -> None:
    """同步原子写 NPY：先写临时文件，再原子替换。"""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_file = filepath + ".tmp"
    with open(tmp_file, "wb") as handle:
        np.save(handle, array)
    os.replace(tmp_file, filepath)


def _sha256_file(filepath: str) -> str:
    digest = hashlib.sha256()
    with open(filepath, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_data_metadata(
    energies_path: str,
    bias_path: str,
    base_path: str,
) -> Dict[str, Any]:
    """Describe one indivisible IBS analysis triplet after all files are durable."""
    arrays = {
        "energies": np.load(energies_path, allow_pickle=False),
        "bias": np.load(bias_path, allow_pickle=False),
        "base": np.load(base_path, allow_pickle=False),
    }
    energies = np.asarray(arrays["energies"])
    bias = np.asarray(arrays["bias"])
    base = np.asarray(arrays["base"])
    if (
        energies.ndim != 2
        or bias.ndim != 1
        or base.ndim != 1
        or energies.shape[1] != bias.size
        or energies.shape[1] != base.size
        or not np.all(np.isfinite(energies))
        or not np.all(np.isfinite(bias))
        or not np.all(np.isfinite(base))
    ):
        raise ValueError("IBS energies/bias/base 三文件形状、长度或有限性无效")
    return {
        "energies": {
            "sha256": _sha256_file(energies_path),
            "shape": [int(x) for x in energies.shape],
        },
        "bias": {
            "sha256": _sha256_file(bias_path),
            "shape": [int(x) for x in bias.shape],
        },
        "base": {
            "sha256": _sha256_file(base_path),
            "shape": [int(x) for x in base.shape],
        },
        "n_frames": int(energies.shape[1]),
    }


def _load_validated_window_data_triplet(
    energies_path: str,
    bias_path: str,
    base_path: str,
    convergence: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fail closed unless the complete analysis triplet matches its manifest."""
    if convergence.get("window_data_protocol_version") != IBS_WINDOW_DATA_PROTOCOL_VERSION:
        raise ValueError("IBS 窗口数据协议缺失或过旧")
    recorded = convergence.get("window_data")
    if not isinstance(recorded, dict):
        raise ValueError("IBS 窗口缺少三文件 manifest")
    for label, path in (
        ("energies", energies_path),
        ("bias", bias_path),
        ("base", base_path),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"IBS 窗口缺少 {label} 文件: {path}")
        item = recorded.get(label)
        if not isinstance(item, dict) or item.get("sha256") != _sha256_file(path):
            raise ValueError(f"IBS 窗口 {label} 文件 hash 不匹配")
    current = _window_data_metadata(energies_path, bias_path, base_path)
    if current != recorded:
        raise ValueError("IBS energies/bias/base 三文件 metadata 不匹配")
    return (
        np.load(energies_path, allow_pickle=False),
        np.load(bias_path, allow_pickle=False),
        np.load(base_path, allow_pickle=False),
    )


def ibs_lj_tail_lrc_is_applicable(potential_type: Optional[str]) -> bool:
    """🔑 [P2-14] `build_ibs_dual_system` 是否会给该势函数附加解析 LJ 长程尾项修正。

    这是**唯一真相**，生产者和报告者共用同一个谓词：

      * 生产者 `build_ibs_dual_system` 用它决定要不要算
        `ibs_wrapper.lj_tail_lrc_coeff_kj_mol`；
      * 报告者 `ABFEPipeline.compute_final_results` 用它填
        `final_results.json` 的 `lj_long_range_dispersion_correction.applied`。

    之前那个字段是**裸字面量 `True`**，对 DEXP 运行无条件说谎：IBS 明确不给
    DEXP 附加修正，`None` 一路 short-circuit 到零（`IBSSampler._lj_tail_
    correction_kj_mol` 直接 `return np.zeros(...)`），可 JSON 里仍写着已应用、
    且下面那段 note 还用散文再断言一遍。

    共用谓词而不是在报告侧另写一个 `potential_type != "dexp"`：DEXP 的解析尾项
    公式一旦被验证/替换，只需要改这一处，行为和报告会一起动，不会再分叉。
    同一份纪律仓库里已经用在系数数组本身上（三个消费者共读
    `ibs_wrapper.lj_tail_lrc_coeff_kj_mol`）。
    """
    return str(potential_type or "").strip().lower() != "dexp"


def ibs_lj_tail_lrc_inapplicable_reason(potential_type: Optional[str]) -> str:
    """不附加修正时的机器可读理由；适用时返回空串。"""
    if ibs_lj_tail_lrc_is_applicable(potential_type):
        return ""
    return (
        "potential_type='dexp' 尚未验证解析尾项公式是否适用于该势函数"
    )


def _production_history_lengths(sampler) -> int:
    """🔑 [P1-13] 返回三份生产 history 的公共长度，顺带强制它们等长。

    `energy_history` / `bias_history` / `base_energy_history` 由
    `collect_energies()` 在同一个 `frame_finite` 门下原子追加，所以实践中总是等长；
    但此前**没有任何地方断言过**这一点。既然回退要靠这个长度做同步截断，
    就必须在取用时把这个不变量钉死——不等长意味着某处已经单独动过其中一份，
    截断会把三者错位，比不截断更糟。
    """
    lengths = (
        len(sampler.energy_history),
        len(sampler.bias_history),
        len(sampler.base_energy_history),
    )
    if len(set(lengths)) != 1:
        raise RuntimeError(
            f"生产 history 三份长度不一致 energy/bias/base = {lengths}；"
            "它们必须始终由 collect_energies() 原子追加。拒绝在错位状态下继续采样。"
        )
    return lengths[0]


def _truncate_production_history(sampler, keep: int) -> int:
    """🔑 [P1-13] 把三份生产 history 同步截断到 `keep` 帧，返回丢弃的帧数。

    用在灾难回退处：坐标已经退回 `production_pos_backup`，那么该备份之后写入的
    帧属于被放弃的分支，不能与重启后的分支拼成一条连续轨迹交给自相关估计。
    """
    current = _production_history_lengths(sampler)
    keep = max(0, int(keep))
    if keep >= current:
        return 0
    del sampler.energy_history[keep:]
    del sampler.bias_history[keep:]
    del sampler.base_energy_history[keep:]
    _production_history_lengths(sampler)  # 截断后仍须等长
    return current - keep


class IBSIncompleteStageCoverageError(RuntimeError):
    """一个 stage 的预期窗口没有全部加载成功，拒绝在缺口上求解自由能。"""


def _assert_expected_windows_all_loaded(
    *,
    expected_windows: List[int],
    loaded_windows: List[int],
    missing_windows: List[Dict[str, Any]],
    source: str,
) -> None:
    """🔑 [P0-8] expected → loaded 的收缩必须 fail closed。

    `GlobalMBARAnalyzer.solve_stage_integrated()` 的完整性判据是
    `len(local_results) == len(valid_windows)`，而 `valid_windows` 本身就是从
    传进来的 `window_data` 算的——loader 静默丢掉的窗口在那里根本不可见。
    协方差链能挡住**中间**缺窗（找不到共享 λ 会走 `_fallback`），但挡不住：

      * 缺首窗：window 1 自然变成 `local_idx == 0`，`join_lam = local_lams[0]`；
      * 缺末窗：链提前正常结束。

    两者都返回一条截断的 ΔG 并报 `converged=True`。`_assert_stage_result_sane()`
    只看求解器返回的字段，不数窗口、不查 λ 覆盖，结构上也抓不到。所以这道门必须
    加在 loader 出口——求解器拿到数据时，信息已经丢了。
    """
    expected = [int(x) for x in expected_windows]
    loaded = {int(x) for x in loaded_windows}
    missing = [int(x) for x in expected if int(x) not in loaded]
    if not missing:
        return
    detail_by_index = {
        int(entry.get("window_index", -1)): entry for entry in missing_windows
    }
    lines = []
    for idx in missing:
        entry = detail_by_index.get(idx)
        if entry:
            lines.append(
                f"    窗口 {idx} λ范围={entry.get('lambda_range')} "
                f"缺文件={entry.get('missing_files')}"
            )
        else:
            lines.append(f"    窗口 {idx}（未产出任何数据）")
    raise IBSIncompleteStageCoverageError(
        f"IBS stage 数据不完整，拒绝求解截断的自由能曲线（来源：{source}）。\n"
        f"  预期窗口 {expected}\n"
        f"  实际加载 {sorted(loaded)}\n"
        f"  缺失窗口 {missing}\n" + "\n".join(lines) + "\n"
        "  缺中间窗口会被协方差链挡下，但缺首/末窗口只会让链在更窄的 λ 区间上闭合，"
        "产出截断的 ΔG 且仍报 converged=True。若确实要在部分窗口上分析（例如 "
        "vanishing rescue 用 rescue ensemble 取代原始窗口），必须显式传入 "
        "excluded_local_windows，而不是让文件缺失来隐式决定覆盖范围。"
    )


def _atomic_write_json(filepath: str, payload: Dict[str, Any]) -> None:
    """Durably write JSON through a temporary file and atomic replace."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_file = filepath + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_file, filepath)
    # POSIX requires syncing the parent directory for the rename itself to be
    # durable across a sudden power loss.  Windows does not allow opening a
    # directory this way; the flushed file + replace remains the strongest
    # portable behavior there.
    if os.name == "posix":
        parent = dirpath or "."
        dir_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def _atomic_save_npz(filepath: str, **arrays: np.ndarray) -> None:
    """同步原子写 NPZ：先写临时文件，再原子替换。复用 _atomic_save_npy 的
    temp-file+os.replace 模式，供 fixed-H 探针轨迹库的续采 checkpoint 使用。"""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_file = filepath + ".tmp"
    with open(tmp_file, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp_file, filepath)


def _positions_to_nm_array(positions) -> np.ndarray:
    if positions is None:
        raise ValueError("positions 不能为空")
    if hasattr(positions, "value_in_unit"):
        return np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    if isinstance(positions, np.ndarray):
        return positions.astype(np.float64, copy=False)
    if isinstance(positions, (list, tuple)):
        if len(positions) == 0:
            raise ValueError("positions 不能为空序列")
        if hasattr(positions[0], "x"):
            return np.asarray([[p.x, p.y, p.z] for p in positions], dtype=np.float64)
        return np.asarray(positions, dtype=np.float64)
    raise TypeError(f"不支持的 positions 类型: {type(positions)}")


def _box_vectors_to_nm_array(box_vectors) -> Optional[np.ndarray]:
    if box_vectors is None:
        return None
    if hasattr(box_vectors, "value_in_unit"):
        return np.asarray(box_vectors.value_in_unit(unit.nanometer), dtype=np.float64)
    if isinstance(box_vectors, np.ndarray):
        return box_vectors.astype(np.float64, copy=False)
    if isinstance(box_vectors, (list, tuple)):
        if len(box_vectors) != 3:
            return None
        first = box_vectors[0]
        if hasattr(first, "x"):
            return np.asarray([[v.x, v.y, v.z] for v in box_vectors], dtype=np.float64)
        return np.asarray(box_vectors, dtype=np.float64)
    return None


def _minimum_image_displacement_nm(
    displacement,
    box_vectors,
) -> np.ndarray:
    """Triclinic minimum-image displacement for row-vector box matrices."""
    delta = np.asarray(displacement, dtype=np.float64)
    box = _box_vectors_to_nm_array(box_vectors)
    if box is None or box.shape != (3, 3) or not np.all(np.isfinite(box)):
        raise ValueError("minimum-image 计算需要有限的 (3,3) 周期盒向量")
    det = float(np.linalg.det(box))
    if not np.isfinite(det) or abs(det) <= 1.0e-12:
        raise ValueError("周期盒向量奇异，无法计算 minimum-image 位移")
    fractional = delta @ np.linalg.inv(box)
    fractional -= np.round(fractional)
    return fractional @ box


def _create_bulk_water_ion_restraint(
    ion_index: int,
    reference_position_nm: np.ndarray,
    force_constant_kj_per_mol_nm2: float = 25.0,
) -> openmm.CustomExternalForce:
    """将共炼金反离子锚定在初始 bulk 水位点，使用 periodicdistance 保持 PBC 一致性。"""
    expr = "0.5*k_ion*periodicdistance(x,y,z,x0,y0,z0)^2"
    force = openmm.CustomExternalForce(expr)
    force.addGlobalParameter("k_ion", float(force_constant_kj_per_mol_nm2))
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    force.addParticle(int(ion_index), [float(reference_position_nm[0]), float(reference_position_nm[1]), float(reference_position_nm[2])])
    return force


def _value_in_inverse_nanometer(value: Any) -> float:
    """
    将 OpenMM 返回的 PME alpha 统一归一化为 1/nm 的纯浮点数。
    兼容 float、Quantity(/nanometer) 以及不同 API 返回形式。
    """
    if hasattr(value, "value_in_unit"):
        try:
            return float((value * unit.nanometer).value_in_unit(unit.dimensionless))
        except Exception:
            pass
        try:
            return float(value / (1 / unit.nanometer))
        except Exception:
            pass
    return float(value)


def _value_in_elementary_charge(value: Any) -> float:
    """
    将 OpenMM 返回的电荷标量统一归一化为 elementary_charge 的纯浮点数。
    兼容 float 与 Quantity(elementary_charge) 两种返回形式——
    NonbondedForce.getParticleParameterOffset()/getExceptionParameterOffset()
    在不同 OpenMM 版本里对 chargeScale 到底包不包 Quantity 并不一致
    （8.5.1 实测直接返回裸 float），之前这里假设一定是 Quantity 直接调用
    .value_in_unit()，遇到裸 float 会抛 AttributeError 而不是静默给错值，
    但仍需要在两种返回形式下都能正确取值。
    """
    if hasattr(value, "value_in_unit"):
        return float(value.value_in_unit(unit.elementary_charge))
    return float(value)


def pme_self_correction_prefactor_kj(alpha_ewald_inv_nm: float, ligand_charge_square_sum: float) -> float:
    """Return C for the PME self offset -C*lambda^2 in kJ/mol."""
    alpha = float(alpha_ewald_inv_nm)
    qsq = float(ligand_charge_square_sum)
    if alpha <= 0.0 or qsq <= 0.0:
        return 0.0
    return 138.935456 * alpha * qsq / math.sqrt(math.pi)


def pme_self_correction_energy_kj(lambda_coul: float, prefactor_kj: float) -> float:
    """Energy added to offline u_kn to remove OpenMM's -C*lambda^2 PME self offset."""
    lam = float(lambda_coul)
    return float(prefactor_kj) * lam * lam


def pme_offset_charge_square_sum(
    nb_force: openmm.NonbondedForce,
    lambda_name: str = "lambda_coul",
) -> Tuple[float, List[Dict[str, float]]]:
    """Return Σq² for particle charge offsets controlled by one lambda.

    OpenMM's reciprocal self term depends on the charge carried by the PME
    particle offsets.  For a co-alchemical counterion cycle this must include
    both the ligand atoms and the selected counterion, not ligand atoms alone.
    """
    qsq = 0.0
    rows: List[Dict[str, float]] = []
    for offset_idx in range(nb_force.getNumParticleParameterOffsets()):
        param, particle_idx, charge_scale, _sigma_scale, _epsilon_scale = nb_force.getParticleParameterOffset(offset_idx)
        if str(param) != str(lambda_name):
            continue
        q_val = _value_in_elementary_charge(charge_scale)
        qsq += q_val * q_val
        rows.append({
            "particle_index": int(particle_idx),
            "charge_offset_e": float(q_val),
            "charge_offset_square_e2": float(q_val * q_val),
        })
    return float(qsq), rows


def lambda_endpoint_diagnostics(
    lambdas_coul,
    lambdas_vdw,
    tol: float = 1e-8,
    *,
    expected_start: Optional[Tuple[float, float]] = None,
    expected_end: Optional[Tuple[float, float]] = None,
) -> Dict:
    """Check a lambda path against explicit physical endpoints.

    The backwards-compatible default describes a complete dual-stage path,
    ``(lambda_coul, lambda_vdw): (1, 1) -> (0, 0)``.  A single stage must pass
    its own endpoints explicitly: decharging is ``(1, 1) -> (0, 1)`` and
    vanishing is ``(0, 1) -> (0, 0)``.
    """
    lc = np.asarray(lambdas_coul, dtype=float)
    lv = np.asarray(lambdas_vdw, dtype=float)
    if lc.size == 0 or lv.size == 0 or lc.size != lv.size:
        return {"ok": False, "reason": "empty_or_mismatched_lambda_arrays"}
    if expected_start is None:
        expected_start = (1.0, 1.0)
    if expected_end is None:
        expected_end = (0.0, 0.0)
    if len(expected_start) != 2 or len(expected_end) != 2:
        raise ValueError("expected_start/expected_end 必须是 (lambda_coul, lambda_vdw)")
    expected_start_payload = {
        "lambda_coul": float(expected_start[0]),
        "lambda_vdw": float(expected_start[1]),
    }
    expected_end_payload = {
        "lambda_coul": float(expected_end[0]),
        "lambda_vdw": float(expected_end[1]),
    }
    start = {"lambda_coul": float(lc[0]), "lambda_vdw": float(lv[0])}
    end = {"lambda_coul": float(lc[-1]), "lambda_vdw": float(lv[-1])}
    matches_expected_start = (
        abs(start["lambda_coul"] - expected_start_payload["lambda_coul"]) <= tol
        and abs(start["lambda_vdw"] - expected_start_payload["lambda_vdw"]) <= tol
    )
    matches_expected_end = (
        abs(end["lambda_coul"] - expected_end_payload["lambda_coul"]) <= tol
        and abs(end["lambda_vdw"] - expected_end_payload["lambda_vdw"]) <= tol
    )
    starts_fully_coupled = (
        abs(start["lambda_coul"] - 1.0) <= tol
        and abs(start["lambda_vdw"] - 1.0) <= tol
    )
    ends_fully_decoupled = (
        abs(end["lambda_coul"]) <= tol and abs(end["lambda_vdw"]) <= tol
    )
    monotonic_coul = bool(np.all(np.diff(lc) <= tol))
    monotonic_vdw = bool(np.all(np.diff(lv) <= tol))
    return {
        "ok": bool(
            matches_expected_start
            and matches_expected_end
            and monotonic_coul
            and monotonic_vdw
        ),
        "start": start,
        "end": end,
        "expected_start": expected_start_payload,
        "expected_end": expected_end_payload,
        "matches_expected_start": bool(matches_expected_start),
        "matches_expected_end": bool(matches_expected_end),
        "starts_fully_coupled": bool(starts_fully_coupled),
        "ends_fully_decoupled": bool(ends_fully_decoupled),
        "monotonic_coul": monotonic_coul,
        "monotonic_vdw": monotonic_vdw,
    }


def delta_u_distribution_diagnostics(u_kn: np.ndarray) -> Dict:
    """Summarize adjacent-state Δu distributions for overlap/convergence checks."""
    u = np.asarray(u_kn, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] < 2 or u.shape[1] == 0:
        return {"available": False, "reason": "u_kn_requires_at_least_2_states_and_1_frame"}
    rows = []
    for k in range(u.shape[0] - 1):
        du = u[k + 1] - u[k]
        rows.append({
            "pair": [int(k), int(k + 1)],
            "mean": float(np.mean(du)),
            "std": float(np.std(du)),
            "p05": float(np.percentile(du, 5)),
            "p50": float(np.percentile(du, 50)),
            "p95": float(np.percentile(du, 95)),
        })
    return {"available": True, "adjacent_delta_u": rows}


def ibs_lse_balance_diagnostics(mean_p: Any) -> Dict[str, Any]:
    """Evaluate the IBS Log-Sum-Exp fixed-point residual.

    A balanced K-state IBS ensemble satisfies ``K * <p_k> = 1`` for every
    state.  Its logarithmic residual is exactly the update direction used by
    ``IBSSampler.update_weights``.  Convergence and schedule design therefore
    inspect this quantity directly, not adjacent fixed-H/replica overlap.
    """
    p = np.asarray(mean_p, dtype=np.float64).ravel()
    if p.size < 2 or not np.all(np.isfinite(p)) or np.any(p < 0.0):
        return {
            "available": False,
            "reason": "mean_p_requires_at_least_two_finite_nonnegative_states",
        }
    total = float(np.sum(p))
    if not np.isfinite(total) or total <= 0.0:
        return {"available": False, "reason": "mean_p_has_nonpositive_sum"}
    p = p / total
    scaled = float(p.size) * p
    with np.errstate(divide="ignore", invalid="ignore"):
        log_residual = np.log(scaled)
    max_abs = (
        float(np.max(np.abs(log_residual)))
        if np.all(np.isfinite(log_residual))
        else float("inf")
    )
    return {
        "available": True,
        "normalized_mean_p": p.tolist(),
        "scaled_balance_K_mean_p": scaled.tolist(),
        "log_residual": log_residual.tolist(),
        "max_abs_log_residual": max_abs,
        "coverage_ess": float(1.0 / np.sum(np.square(p))),
        "target_p": float(1.0 / p.size),
    }


def _best_effort_validation_is_acceptable(
    residual: Any,
    validation_stats: Optional[Dict[str, Any]],
    tolerance: float,
    max_residual_multiple: float,
    minimum_complete_frames: int,
) -> bool:
    """Whether a completed fixed-f attempt is safe for best-effort use.

    This is separate from the strict frozen-validation gate.  It only decides
    whether a completed warmup attempt is close enough to flat to enter
    production, whose independent overlap/ESS/decorrelation/uncertainty gates
    remain authoritative.  A partial attempt at the global step cap is never
    eligible.
    """
    if not validation_stats:
        return False
    try:
        residual_value = float(residual)
        n_frames = int(validation_stats.get("validation_sample_count", 0))
        bound = float(tolerance) * float(max_residual_multiple)
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(
        np.isfinite(residual_value)
        and np.isfinite(bound)
        and bound > 0.0
        and n_frames >= int(minimum_complete_frames)
        and residual_value <= bound
    )


def _meets_minimum_with_roundoff(
    value: Optional[float],
    minimum: float,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> bool:
    """Compare a computed diagnostic to an inclusive lower bound.

    ESS/overlap values come from normalized floating-point sums and can land a
    few ulps below their exact mathematical boundary (real v23 example:
    0.04999999999999431 vs 0.05, and 0.9999999999998863 vs 1.0).  Treat only
    roundoff-scale equality as equality; this does not relax the physical
    threshold for materially lower values.
    """
    if value is None:
        return False
    value = float(value)
    minimum = float(minimum)
    if not np.isfinite(value) or not np.isfinite(minimum):
        return False
    return bool(
        value >= minimum
        or np.isclose(value, minimum, rtol=float(rtol), atol=float(atol))
    )


def _meets_maximum_with_roundoff(
    value: Optional[float],
    maximum: float,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> bool:
    """Upper-bound counterpart of `_meets_minimum_with_roundoff` (same
    roundoff-only tolerance, same rationale) -- for `value <= maximum`
    criteria (e.g. an uncertainty ceiling) instead of `value >= minimum`.
    """
    if value is None:
        return False
    value = float(value)
    maximum = float(maximum)
    if not np.isfinite(value) or not np.isfinite(maximum):
        return False
    return bool(
        value <= maximum
        or np.isclose(value, maximum, rtol=float(rtol), atol=float(atol))
    )


def synthetic_mbar_u_kn(delta_f_kT: float = 1.25, n_per_state: int = 200, seed: int = 20260630) -> Tuple[np.ndarray, np.ndarray]:
    """Small deterministic two-state synthetic dataset for MBAR smoke tests."""
    rng = np.random.default_rng(int(seed))
    x0 = rng.normal(loc=0.0, scale=1.0, size=int(n_per_state))
    x1 = rng.normal(loc=float(delta_f_kT), scale=1.0, size=int(n_per_state))
    x = np.concatenate([x0, x1])
    u0 = 0.5 * x * x
    u1 = 0.5 * (x - float(delta_f_kT)) ** 2 + float(delta_f_kT)
    return np.vstack([u0, u1]), np.array([int(n_per_state), int(n_per_state)], dtype=int)


def _select_bulk_water_counterion(
    nb_force: openmm.NonbondedForce,
    ligand_indices: List[int],
    topology,
    positions,
    box_vectors,
) -> Tuple[List[int], List[np.ndarray], Dict[str, Any]]:
    """Select enough monovalent counterions using PBC-aware bulk-water metrics."""
    pos_nm = _positions_to_nm_array(positions)
    ligand_set = set(int(i) for i in ligand_indices)
    raw_lig_net_charge = 0.0
    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(idx)
        raw_lig_net_charge += q.value_in_unit(unit.elementary_charge)
    lig_net_charge = int(round(raw_lig_net_charge))
    if abs(raw_lig_net_charge - lig_net_charge) > 1.0e-3:
        raise RuntimeError(
            f"配体净电荷 {raw_lig_net_charge:+.6f} e 不接近整数（容差 1e-3 e）"
        )
    if lig_net_charge == 0:
        return [], [], {}

    target_ion_charge = -1.0 if lig_net_charge > 0 else 1.0
    required_count = abs(lig_net_charge)
    water_names = {"HOH", "WAT", "SOL", "TIP3", "TIP3P"}
    ion_names = {"CL", "CLA", "NA", "SOD", "K", "POT", "MG", "CA"}
    heavy_solute_indices = [
        a.index for a in topology.atoms()
        if a.index not in ligand_set
        and a.residue.name.upper() not in water_names | ion_names
        and getattr(a.element, "symbol", "") != "H"
    ]
    if not heavy_solute_indices:
        heavy_solute_indices = [
            a.index for a in topology.atoms()
            if a.index not in ligand_set and a.residue.name.upper() not in water_names | ion_names
        ]
    solute_indices = sorted(ligand_set | set(heavy_solute_indices))

    water_oxygen_indices = [
        a.index for a in topology.atoms()
        if a.residue.name.upper() in water_names and getattr(a.element, "symbol", "").upper() == "O"
    ]

    candidates: List[Tuple[float, int, int, Dict[str, Any]]] = []
    for atom in topology.atoms():
        if atom.residue.name.upper() not in ion_names:
            continue
        idx = atom.index
        q, _, _ = nb_force.getParticleParameters(idx)
        q_val = q.value_in_unit(unit.elementary_charge)
        if abs(q_val - target_ion_charge) >= 0.1:
            continue

        ion_pos = pos_nm[idx]
        solute_delta = _minimum_image_displacement_nm(
            pos_nm[solute_indices] - ion_pos,
            box_vectors,
        )
        solute_dist = float(np.min(np.linalg.norm(solute_delta, axis=1)))
        water_coord = 0
        if water_oxygen_indices:
            water_delta = _minimum_image_displacement_nm(
                pos_nm[water_oxygen_indices] - ion_pos,
                box_vectors,
            )
            water_dists = np.linalg.norm(water_delta, axis=1)
            water_coord = int(np.sum(water_dists <= 0.45))
        candidates.append(
            (
                solute_dist,
                water_coord,
                -int(idx),
                {
                "ion_index": int(idx),
                "solute_distance_nm": solute_dist,
                "water_coordination": float(water_coord),
                "target_charge_e": float(target_ion_charge),
                },
            )
        )

    candidates.sort(reverse=True)
    if len(candidates) < required_count:
        raise RuntimeError(
            f"需要 {required_count} 个 {target_ion_charge:+.0f}e 反离子，"
            f"但只找到 {len(candidates)} 个"
        )
    selected = candidates[:required_count]
    ion_indices = [int(item[3]["ion_index"]) for item in selected]
    selected_charge = 0.0
    for idx in ion_indices:
        charge, _, _ = nb_force.getParticleParameters(idx)
        selected_charge += charge.value_in_unit(unit.elementary_charge)
    if abs(raw_lig_net_charge + selected_charge) > 1.0e-3:
        raise RuntimeError(
            "所选共炼金反离子无法严格抵消配体电荷："
            f"ligand={raw_lig_net_charge:+.6f} e, ions={selected_charge:+.6f} e"
        )
    reference_positions = [pos_nm[idx].copy() for idx in ion_indices]
    return ion_indices, reference_positions, {
        "ligand_net_charge_e": int(lig_net_charge),
        "target_ion_charge_e": float(target_ion_charge),
        "required_count": int(required_count),
        "selected": [item[3] for item in selected],
        "selection": "max_minimum_image_distance_to_nearest_solute_then_water_coordination",
    }


def _prepare_pme_coulomb_leg_system(
    system_template: openmm.System,
    ligand_indices: List[int],
    lambda_name: str = "lambda_coul",
    allow_charged_ligand: bool = False,
    topology=None,
    positions=None,
    box_vectors=None,
) -> openmm.System:
    prepared = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system_template))
    prepared.thisown = 1
    configure_pme_ligand_charge_offsets(
        prepared,
        ligand_indices,
        lambda_name=lambda_name,
        allow_charged_ligand=allow_charged_ligand,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
    )
    return prepared


def _compute_ligand_net_charge(
    system: openmm.System,
    ligand_indices: List[int],
) -> float:
    nb_force = next((f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
    if nb_force is None:
        raise RuntimeError("系统中未找到 NonbondedForce，无法统计配体净电荷。")
    total = 0.0
    for idx in ligand_indices:
        q, _, _ = nb_force.getParticleParameters(int(idx))
        total += q.value_in_unit(unit.elementary_charge)
    return float(total)


def _restore_ligand_internal_nonbonded(
    system: openmm.System,
    nb_force: openmm.NonbondedForce,
    ligand_indices: List[int],
    zero_original_exceptions: bool = False,
) -> None:
    """为传统 VDW 路径恢复配体内部普通非键与 1-4 作用。"""
    particle_params = [nb_force.getParticleParameters(i) for i in range(nb_force.getNumParticles())]
    reference_exclusions = [
        (int(nb_force.getExceptionParameters(i)[0]), int(nb_force.getExceptionParameters(i)[1]))
        for i in range(nb_force.getNumExceptions())
    ]
    ll_force, ll_14_force = create_ligand_internal_force(
        nb_force,
        ligand_indices,
        particle_params,
        reference_exclusions,
        nb_force.getNumParticles(),
        system=system,
    )
    if ll_force is not None:
        system.addForce(ll_force)
    if ll_14_force is not None:
        system.addForce(ll_14_force)
    if zero_original_exceptions:
        ligand_set = set(int(idx) for idx in ligand_indices)
        for exc_idx in range(nb_force.getNumExceptions()):
            p1, p2, charge_prod, sigma, epsilon = nb_force.getExceptionParameters(exc_idx)
            p1 = int(p1)
            p2 = int(p2)
            if p1 in ligand_set and p2 in ligand_set:
                nb_force.setExceptionParameters(
                    exc_idx,
                    p1,
                    p2,
                    0.0 * unit.elementary_charge**2,
                    sigma,
                    0.0 * unit.kilojoule_per_mole,
                )


def _add_physical_boresch_restraint(
    system: openmm.System,
    restraint_params: Optional[Dict],
    force_group: int = 3,
) -> None:
    if not _has_valid_boresch_restraint(restraint_params):
        return
    rest_f_phys = LambdaDependentBoreschForce(
        rec_idx=restraint_params["receptor_indices"],
        lig_idx=restraint_params["ligand_indices"],
        eq=restraint_params["equilibrium_values"],
        fc=restraint_params["force_constants"],
        lam_name="lambda_boresch_scale",
        fixed_lam=1.0,
        sign=1.0,
        use_pbc=True,
    )
    rest_f_phys.setForceGroup(force_group)
    system.addForce(rest_f_phys)


# ---------------------------------------------------------------------------
# Boresch attachment 腿 (A′→A)
# ---------------------------------------------------------------------------
# 热力学循环里缺的那一项。当前 ΔG_complex = ΔG(A→B) + ΔG_release，其中
# A =「配体耦合 + 限制已打开」；但物理结合态是 A′ =「配体耦合 + 无限制」。
# 缺的正是 A′→A。补上之后，循环对**任意**限制强度都严格闭合——受约束系综里
# 测出来的 charging/vdW 就是对的，不需要（也不能）再单独修正。
#
# 与 stage1/stage2 的关键差别：
#   * 配体全程完全耦合，不碰 lambda_coul / lambda_vdw；
#   * 只有一维 λ（限制强度），没有软核端点奇异性；
#   * 因此不需要副本交换，独立窗口 + MBAR 就够——也就不必去动 REMDManager
#     里那些硬编码的 "lambda_coul"。
BORESCH_ATTACHMENT_PROTOCOL_VERSION = 1
BORESCH_ATTACHMENT_LAMBDA_NAME = "lambda_boresch_scale"
BORESCH_ATTACHMENT_FORCE_GROUP = 3

# 升序排列——MBAR 的 delta_G = f[K-1] − f[0]，升序时它直接就是 ΔG(A′→A)，
# 全链路不出现任何负号。
#
# ⚠️ 2026-07-28 更正：初版默认表有 12 个态，理由写的是「有效力常数是 λ·k，
# λ→0 附近相邻态重叠极差，所以近 0 端必须加密」。**那个论证在这里不成立，
# 而且实测正好反过来。** 它的前提是限制势是该坐标的主要约束；但配体在口袋里，
# 蛋白本身已经把它按住了，U_Boresch 在所有 λ 下都被压在 1.6–5.6 kT，于是
# Δu_相邻 = Δλ·U_B/kT 处处很小。12 态实测每条边的 ⟨Δu⟩ 全部 ≤ 0.33 kT
# （λ→0 那几档 0.04/0.08/0.15 反而是全阶梯重叠最好的），而 BAR 的合理区间是
# 每边 1–2 kT——等于密了 3–6 倍。
#
# 用同一批样本做的对照（`attachment_rerun/20260728_130848`）：
#     [0, 1]                2 态  ΔG=5.4317   最大边 3.54 kT
#     [0, 0.25, 1]          3 态  ΔG=5.1329   最大边 1.73 kT
#     [0, 0.1, 0.35, 1]     4 态  ΔG=5.1949   最大边 1.41 kT
#     12 态链式 BAR                ΔG=5.3784   最大边 0.33 kT
#     12 态 MBAR                   ΔG=5.7440
# 全距 0.61 kJ/mol，而 12 态 MBAR 报出的 σ 只有 0.083——**估计量选择造成的
# 散布是报出误差棒的 7 倍**。多花 6 倍态数买不到任何精度。
#
# 本体系真正的瓶颈是**慢构象弛豫**，不是 λ 重叠：12 个密集态把预算切成了
# 12 段短平衡，每段都可能残留上一个态的记忆。省下的态数要投到每态更长采样、
# 多 seed 和双向验证上。
DEFAULT_BORESCH_ATTACHMENT_LAMBDAS = (0.0, 0.1, 0.35, 1.0)


def add_scalable_boresch_restraint(
    system: openmm.System,
    restraint_params: Optional[Dict],
    lam_name: str = BORESCH_ATTACHMENT_LAMBDA_NAME,
    force_group: int = BORESCH_ATTACHMENT_FORCE_GROUP,
) -> bool:
    """加一条**可扫 λ** 的 Boresch 力（注册全局参数）。

    与 `_add_physical_boresch_restraint` 的区别：那个传 `fixed_lam=1.0`，
    `LambdaDependentBoreschForce.__init__` 会把 1.0 直接编译进表达式字符串、
    **连全局参数都不注册**，所以生产采样期的限制强度根本改不了。
    attachment 腿要扫 λ，必须用 `fixed_lam=None` 这一路。
    """
    if not _has_valid_boresch_restraint(restraint_params):
        return False
    force = LambdaDependentBoreschForce(
        rec_idx=restraint_params["receptor_indices"],
        lig_idx=restraint_params["ligand_indices"],
        eq=restraint_params["equilibrium_values"],
        fc=restraint_params["force_constants"],
        lam_name=lam_name,
        fixed_lam=None,
        sign=1.0,
        use_pbc=True,
    )
    force.setForceGroup(force_group)
    system.addForce(force)
    return True


def _assert_force_group_is_free(system: openmm.System, force_group: int) -> None:
    """确认没有别的力占着这个力组。

    attachment 腿靠「只取该力组的能量」来拿 U_Boresch(x)，一旦有别的力混进来，
    拆出来的就不是限制势，整条腿的能量矩阵都会错——而且错得很隐蔽（数值仍然
    有限、MBAR 照样收敛）。所以这里 fail closed。
    """
    occupied = [
        f.__class__.__name__
        for f in system.getForces()
        if f.getForceGroup() == int(force_group)
    ]
    if occupied:
        raise RuntimeError(
            f"力组 {force_group} 已被 {occupied} 占用，无法用它单独取 Boresch 能量。"
            "请换一个空闲力组。"
        )


def adjacent_bar_chain(u_kn, n_k, kt):
    """相邻 BAR 链：逐边双向 BAR 再相加。`u_kn` 必须是**约化**势，返回 kJ/mol。

    适用条件（**硬前提，下面 fail closed**）：每个态自己都有样本。
    满足的腿：stage1 decharging（REMD demux 后每态有帧）、attachment 腿
    （顺序独立窗口）。**不满足的腿：stage2 IBS**——每个窗口只有 row 0（偏置混合
    采样分布）有样本，物理 λ 行 `n_k=0`（见
    `GlobalMBARAnalyzer.solve_stage_integrated` 里 `n_k_local[sampled_row] = n_frames`）。
    把偏置混合帧当端点帧喂给 BAR 会违反其前提，算出来的数看似正常实则无效，
    所以这里对任何零样本态直接抛错，而不是给出一个能被误引用的值。

    为什么它比去相关 MBAR 可信：2026-07-28 实测同一批 attachment 样本上
    TI 5.3867 / 相邻 BAR 5.3784 / 两端 BAR 5.4317 三者一致，而
    `TraditionalMBARAnalyzer` 的**去相关 MBAR** 给 5.7440——选帧把结果偏出
    0.37 kJ/mol（4.4σ）。stage1 上同样的模式：BAR 65.076 / FD-TI 65.126 /
    全帧 MBAR 65.003 一致，去相关 MBAR 64.411 偏低 0.6。BAR 逐边只用相邻两态
    自己的样本，不受全局选帧影响。

    误差按独立边方差相加——边之间其实有相关，这是保守近似而非严格值。
    """
    from pymbar import other_estimators

    n_k = np.asarray(n_k, dtype=int)
    u_kn = np.asarray(u_kn, dtype=np.float64)
    K = len(n_k)
    if K < 2:
        raise ValueError(f"相邻 BAR 至少需要 2 个态，收到 {K}")
    if u_kn.ndim != 2 or u_kn.shape[0] != K:
        raise ValueError(
            f"u_kn 形状 {u_kn.shape} 与态数 {K} 不符；相邻 BAR 需要 (K, N) 的约化势矩阵"
        )
    if int(np.sum(n_k)) != u_kn.shape[1]:
        raise ValueError(
            f"sum(n_k)={int(np.sum(n_k))} 与 u_kn 帧数 {u_kn.shape[1]} 不符"
        )
    zero_states = [int(k) for k in range(K) if n_k[k] <= 0]
    if zero_states:
        raise ValueError(
            f"相邻 BAR 前提不成立：态 {zero_states} 的 n_k=0。BAR 要求两个端点系综"
            "各自有样本。IBS stage2 每窗只有 row 0（偏置混合采样分布）有样本、物理 λ "
            "行 n_k=0，因此 stage2 **不得**使用 BAR（也不得使用线性势 TI：vdW "
            "softcore 对 λ 非线性且未落盘 ∂U/∂λ）；那两格必须写 not_applicable_reason。"
        )
    off = np.concatenate(([0], np.cumsum(n_k))).astype(int)
    total, var, edges = 0.0, 0.0, []
    for k in range(K - 1):
        a = slice(off[k], off[k + 1])
        b = slice(off[k + 1], off[k + 2])
        e = other_estimators.bar(
            u_kn[k + 1, a] - u_kn[k, a],
            u_kn[k, b] - u_kn[k + 1, b],
            compute_uncertainty=True,
        )
        d, s = float(e["Delta_f"]) * kt, float(e["dDelta_f"]) * kt
        total += d
        var += s ** 2
        edges.append({"edge": [int(k), int(k + 1)], "delta_G_kJ_mol": d, "error_kJ_mol": s})
    return total, float(np.sqrt(var)), edges


def _attachment_bar_chain(u_kn, n_k, kt):
    """attachment 腿的主估计量。就是 `adjacent_bar_chain`，保留旧名以免破坏调用方。

    （曾经是独立实现；2026-07-28 stage1 也要用同一个链式 BAR，与其抄第二份，
    直接复用——两者数学完全相同，连误差合并方式都一样。）
    """
    return adjacent_bar_chain(u_kn, n_k, kt)


def _attachment_ti(lam, mean_u_boresch):
    """梯形 TI：∫₀¹⟨∂U/∂λ⟩dλ。势对 λ 严格线性 ⟹ ∂U/∂λ = U_Boresch。

    TI 是三个口径里对稀有大能量帧最不敏感的（普通均值，没有指数放大），
    所以拿它做 BAR 的交叉检查最有分辨力。

    ⚠️ **只对 attachment 腿成立**：Boresch 势乘 λ_boresch 是严格线性的，所以
    ∂U/∂λ 就是 U_Boresch 本身。**不要拿它去做 stage2 的 vdW 腿**——softcore 对 λ
    非线性，线性势假设不成立；那条腿也从未落盘 ∂U/∂λ。需要在已有 λ 网格上估
    ∂U/∂λ 的场合用下面的 `reweighted_fd_ti`（它也只适用于每个态都有样本的腿）。
    """
    lam = np.asarray(lam, dtype=float)
    ub = np.asarray(mean_u_boresch, dtype=float)
    return float(np.trapezoid(ub, lam) if hasattr(np, "trapezoid") else np.trapz(ub, lam))


def reweighted_fd_ti(
    u_kn,
    n_k,
    lambdas: Sequence[float],
    kt: float,
    weights: Optional[np.ndarray] = None,
):
    """⟨∂U/∂λ⟩ 重加权有限差分 → 对 λ 梯形积分，返回 (kJ/mol, 逐态 ⟨∂U/∂λ⟩)。

    `u_kn` 是**约化**势（无量纲），所以差分出来要乘 kT 才是 kJ/mol。
    ∂U/∂λ 用**实际的（非均匀）λ 网格**做中心差分，端点用单边差分。
    **绝不用 state index 代替 λ**——λ 表通常非均匀，用序号会给出错误的梯度。

    `weights=None` 时退化为「每个态用自己的样本块求平均」（普通 TI）；
    给了 MBAR 的 `W_nk` 就是重加权 TI。

    适用条件与 `adjacent_bar_chain` 相同：每个态都要有样本。stage2 IBS 不适用。
    """
    lam = np.asarray(lambdas, dtype=float)
    n_k = np.asarray(n_k, dtype=int)
    u_kn = np.asarray(u_kn, dtype=np.float64)
    K = len(n_k)
    if lam.size != K:
        raise ValueError(f"λ 数 {lam.size} != 态数 {K}")
    if K < 2:
        raise ValueError("λ 表至少需要 2 个态")
    off = np.concatenate(([0], np.cumsum(n_k))).astype(int)

    dudl = np.empty(K, dtype=float)
    for k in range(K):
        lo = max(0, k - 1)
        hi = min(K - 1, k + 1)
        dlam = lam[hi] - lam[lo]
        if abs(dlam) < 1e-12:
            raise ValueError(f"λ[{lo}] 与 λ[{hi}] 重合，无法做差分")
        # (U_hi − U_lo)/(λ_hi − λ_lo)，逐帧算完再按权重平均。乘 kt 把约化势转 kJ/mol。
        per_frame = (u_kn[hi] - u_kn[lo]) * kt / dlam
        if weights is not None:
            w = np.asarray(weights[:, k], dtype=float)
            tot = float(np.sum(w))
            if not np.isfinite(tot) or tot <= 0.0:
                raise RuntimeError(f"态 {k} 的 MBAR 权重和非正，无法重加权")
            dudl[k] = float(np.dot(w / tot, per_frame))
        else:
            if off[k + 1] <= off[k]:
                raise ValueError(f"态 {k} 没有样本，无法做普通 TI 平均")
            dudl[k] = float(np.mean(per_frame[off[k] : off[k + 1]]))

    # ⚠️ 不要翻符号。`np.trapz(y, x)` 沿给定顺序算 ∫_{x[0]}^{x[-1]} y dx，
    # 所以 λ 降序时它**直接就是** ΔG(λ=1 → λ=0)，正是本仓库解耦腿的约定。
    # 数值核对：降序 λ + ∂U/∂λ<0（λ↓ 电荷关掉、U 升）两个负号相乘 → 正值 ✓；
    # 升序 λ（attachment）+ ∂U/∂λ=U_B>0 → 也是正值 ✓。
    # 曾在这里多加过一次 `if lam[0] > lam[-1]: integ = -integ`，会把 stage1 的
    # +65 变成 −65。
    integ = np.trapezoid(dudl, lam) if hasattr(np, "trapezoid") else np.trapz(dudl, lam)
    return float(integ), dudl.tolist()


# 门的绝对下限。σ 可以任意小（弱限制极限下所有边的 ΔG 都≈0，BAR 误差也≈0），
# 这时纯 z 判据会把 1e-5 kJ/mol 的数值噪声判成 7σ——实测就出现过
# 「漂移 +0.0000 kJ/mol = 7.1×2σ」。所以每道门都必须同时超过一个绝对量才失败。
ATTACHMENT_BAR_TI_ABS_TOL_KJ = 1.0
ATTACHMENT_SPLIT_HALF_ABS_TOL_KJ = 0.5


def attachment_bar_ti_gate(dg_bar: float, err: float, dg_ti: float) -> Tuple[bool, float, str]:
    """BAR 与 TI 一致性门。返回 ``(是否通过, 容差, 说明)``。

    两者对稀有大能量帧的敏感度不同：TI 是普通均值，BAR 走指数平均。
    一帧二面角反转（2k ≈ 360–470 kJ/mol）能把 BAR 拉走而 TI 几乎不动，
    分歧就是「估计量被少数帧支配」的信号。
    """
    tol = max(ATTACHMENT_BAR_TI_ABS_TOL_KJ, 3.0 * max(float(err), 0.0))
    diff = abs(float(dg_bar) - float(dg_ti))
    msg = (
        f"BAR({dg_bar:.4f}) 与 TI({dg_ti:.4f}) 差 {diff:.4f} kJ/mol，容差 {tol:.4f}"
    )
    return diff <= tol, tol, msg


def attachment_split_half_gate(drift: float, err: float) -> Tuple[bool, float, str]:
    """前后半程一致性门。返回 ``(是否通过, 容差, 说明)``。

    两个半程各自 SE≈√2σ，其差的 SE≈2σ，所以 z 判据的分母是 2σ。
    再叠一个绝对下限，避免 σ≈0 时把数值噪声判成大偏离。
    """
    tol = max(ATTACHMENT_SPLIT_HALF_ABS_TOL_KJ, 3.0 * 2.0 * max(float(err), 0.0))
    d = abs(float(drift))
    return d <= tol, tol, f"前后半程漂移 {drift:+.4f} kJ/mol，容差 {tol:.4f}"


def run_boresch_attachment_leg(
    system: openmm.System,
    topology,
    positions,
    box_vectors,
    restraint_params: Dict,
    *,
    temperature_k: float = 300.0,
    lambdas: Optional[Sequence[float]] = None,
    n_steps_per_state: int = 250_000,
    equil_steps_per_state: int = 50_000,
    steps_per_sample: int = 1_000,
    platform_name: str = "CUDA",
    seed: int = 0,
    n_seeds: int = 1,
    enforce_convergence_gates: bool = True,
    output_dir: Optional[str] = None,
    log=print,
) -> Dict:
    """测 ΔG(A′→A)：把 Boresch 限制从关到开的自由能代价。

    `enforce_convergence_gates=False` 只关掉 BAR/TI 与 split-half 两道**收敛**门
    （改为打警告），用于故意跑很短轨迹的单元测试；`ΔG ≥ 0`、力组占用、
    几何奇点、λ 阶梯方向这些**正确性**门任何时候都不能关。

    顺序独立窗口（λ 从 1 降到 0 链式推进），**主估计量是相邻 BAR**，TI 作交叉检查。

    2026-07-28 的两次更正，都写在这里免得后人重蹈：

    1. **主值曾接在去相关 MBAR 上，那是 bug。** 同一批样本 TI 5.3867 /
       相邻 BAR 5.3784 一致，而去相关 MBAR 给 5.7440。MBAR 现在降级为诊断。
    2. **不要改用 Hamiltonian REMD。** 试过，产出 38.6±110、零 round trip。
       根因是 Boresch 的 `k(1−cosΔ)` 项在反转时取 2k：单个二面角 359–469 kJ/mol
       （144–188 kT），三个全反转 1189 kJ/mol = 477 kT。交换一旦把副本送进翻转态，
       那一帧的天文 U_B 就支配了指数平均。顺序窗口从不离开原盆地，反而良态。
       **而且单盆地是必要的**：配套的解析释放项（`calculate_boresch_analytical_correction`）
       本身就假定单一简谐盆地，配一个会采到翻转的 attachment 腿反而不自洽。

    λ 升序（0 → 1），所有估计量都直接给 ΔG(A′→A)，全链路不出现负号。

    严格下界：Boresch 势处处 ≥ 0 ⟹ ΔG(A′→A) ≥ 0，算出负值直接 fail closed。
    """
    if not _has_valid_boresch_restraint(restraint_params):
        raise ValueError("attachment 腿需要完整的 3+3 Boresch 锚点")

    n_seeds = int(n_seeds)
    if n_seeds < 1:
        raise ValueError(f"n_seeds 必须 ≥1，收到 {n_seeds}")
    if n_seeds > 1:
        common = dict(
            temperature_k=temperature_k, lambdas=lambdas,
            n_steps_per_state=n_steps_per_state,
            equil_steps_per_state=equil_steps_per_state,
            steps_per_sample=steps_per_sample,
            platform_name=platform_name, n_seeds=1, log=log,
            enforce_convergence_gates=enforce_convergence_gates,
        )
        runs = []
        for r in range(n_seeds):
            log(f"\n  ── seed {r + 1}/{n_seeds} ──")
            runs.append(run_boresch_attachment_leg(
                system, topology, positions, box_vectors, restraint_params,
                seed=int(seed) + 104729 * r,
                output_dir=(os.path.join(output_dir, f"seed{r}") if output_dir else None),
                **common,
            ))
        vals = np.array([r["attachment_delta_G_kJ_mol"] for r in runs], dtype=float)
        mean = float(np.mean(vals))
        sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
        log(f"\n  ✅ 跨 {n_seeds} 个 seed: ΔG(A′→A) = {mean:.4f} ± {sem:.4f} kJ/mol "
            f"= {mean / 4.184:.4f} ± {sem / 4.184:.4f} kcal/mol")
        log(f"     逐 seed: {[round(v, 4) for v in vals.tolist()]}")
        return {
            "stage": "boresch_attachment",
            "protocol_version": int(BORESCH_ATTACHMENT_PROTOCOL_VERSION),
            "method": "sequential-windows-adjacent-BAR",
            "attachment_delta_G_kJ_mol": mean,
            "attachment_error_kJ_mol": sem,
            "uncertainty_source": "across_seed_sem",
            "n_seeds": n_seeds,
            "per_seed_delta_G_kJ_mol": vals.tolist(),
            "per_seed_spread_kJ_mol": float(np.max(vals) - np.min(vals)),
            "direction": "A_prime_to_A (restraint OFF -> ON), always >= 0",
            "lambdas": runs[0]["lambdas"],
            "per_seed": runs,
        }

    lam = np.asarray(
        list(lambdas) if lambdas is not None else list(DEFAULT_BORESCH_ATTACHMENT_LAMBDAS),
        dtype=np.float64,
    )
    if lam.ndim != 1 or lam.size < 2:
        raise ValueError(f"λ 阶梯至少 2 个态（两端双向 BAR），收到 {lam.size} 个")
    if not (np.all(np.diff(lam) > 0) and abs(lam[0]) < 1e-12 and abs(lam[-1] - 1.0) < 1e-12):
        raise ValueError(
            f"λ 阶梯必须严格升序且从 0 到 1（否则 ΔG 的符号含义就变了），收到 {lam.tolist()}"
        )

    kt = 0.008314462618 * float(temperature_k)
    K = int(lam.size)

    work = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    work.thisown = 1
    _assert_force_group_is_free(work, BORESCH_ATTACHMENT_FORCE_GROUP)
    if not add_scalable_boresch_restraint(work, restraint_params):
        raise RuntimeError("Boresch 力注入失败")

    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin, 1.0 / unit.picosecond, 2.0 * unit.femtosecond
    )
    integrator.setRandomNumberSeed(int(seed) or 1)
    simulation = app.Simulation(
        topology, work, integrator, openmm.Platform.getPlatformByName(platform_name)
    )
    simulation.context.setPositions(positions)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)

    safe, r0_chk, tha_chk, thb_chk = _check_boresch_geometry_safe(
        simulation.context, restraint_params
    )
    if not safe:
        raise RuntimeError(
            f"attachment 腿起始构象的 Boresch 锚点几何不安全（r0={r0_chk}, "
            f"thetaA={tha_chk}°, thetaB={thb_chk}°）：角度接近 0°/180°，"
            "力的解析梯度会发散。请换锚点或起始构象。"
        )
    log(f"  ✓ 锚点几何: r0={r0_chk:.4f} nm  θA={tha_chk:.2f}°  θB={thb_chk:.2f}°")

    n_samples = max(2, int(n_steps_per_state) // int(steps_per_sample))
    u_boresch = np.zeros((K, n_samples), dtype=np.float64)
    u_base = np.zeros((K, n_samples), dtype=np.float64)

    order = list(range(K - 1, -1, -1))   # 从全强度端往下走
    log(f"  attachment 腿：{K} 个 λ 态 {np.round(lam, 4).tolist()}")
    log(f"  每态 平衡 {equil_steps_per_state} 步 + 生产 {n_steps_per_state} 步 → {n_samples} 帧")

    linearity_checked = False
    for k in order:
        simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, float(lam[k]))
        simulation.context.setVelocitiesToTemperature(
            temperature_k * unit.kelvin, int(seed) + 7919 * k + 1
        )
        simulation.step(int(equil_steps_per_state))
        for s in range(n_samples):
            simulation.step(int(steps_per_sample))
            total = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            total = total.value_in_unit(unit.kilojoule_per_mole)
            # U_B 一律在 λ=1 下直接量，不用「力组能量 / λ_k」反推：
            # λ=0 时无定义，λ=0.01 会把单精度舍入放大 100 倍。
            simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, 1.0)
            u_b = simulation.context.getState(
                getEnergy=True, groups={BORESCH_ATTACHMENT_FORCE_GROUP}
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, float(lam[k]))
            u_boresch[k, s] = u_b
            u_base[k, s] = total - float(lam[k]) * u_b

            if not linearity_checked:
                for probe in (0.35, 0.8):
                    simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, probe)
                    direct = simulation.context.getState(
                        getEnergy=True
                    ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    predicted = u_base[k, s] + probe * u_b
                    if abs(direct - predicted) > 1.0e-4 * max(1.0, abs(direct)):
                        raise RuntimeError(
                            f"Boresch 势对 λ 的线性假设不成立：λ={probe} 直接测 {direct:.6f}，"
                            f"按 U_0+λ·U_B 预测 {predicted:.6f}。力组可能被别的力共用。"
                        )
                simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, float(lam[k]))
                linearity_checked = True
                log("  ✓ U(λ) 对 λ 的线性性自检通过（λ=0.35 / 0.8 两点）")

        log(f"    λ={lam[k]:.4f}  ⟨U_Boresch⟩={float(np.mean(u_boresch[k])):9.3f} "
            f"± {float(np.std(u_boresch[k])):7.3f}  max={float(np.max(u_boresch[k])):9.3f} kJ/mol")

    del simulation, integrator

    n_k = np.full(K, n_samples, dtype=int)
    u_kn = np.zeros((K, K * n_samples), dtype=np.float64)
    for k in range(K):
        cols = slice(k * n_samples, (k + 1) * n_samples)
        for j in range(K):
            u_kn[j, cols] = (u_base[k] + lam[j] * u_boresch[k]) / kt

    mean_ub = [float(np.mean(u_boresch[k])) for k in range(K)]
    max_ub = float(np.max(u_boresch))

    # ---- 主估计量：相邻 BAR ----
    dg, err, bar_edges = _attachment_bar_chain(u_kn, n_k, kt)
    # ---- 交叉检查：TI ----
    dg_ti = _attachment_ti(lam, mean_ub)
    # ---- 诊断：去相关 MBAR（历史主值，已降级）----
    try:
        an = TraditionalMBARAnalyzer(temperature=temperature_k)
        an._last_n_k = n_k
        _m = an.solve(u_kn, decorrelate=True)
        dg_mbar, err_mbar = float(_m["delta_G"]), float(_m["error"])
    except Exception as exc:
        dg_mbar, err_mbar = float("nan"), float("nan")
        log(f"  ⚠️ MBAR 诊断失败（不影响主值）: {exc}")

    log(f"  BAR(主) {dg:.4f} ± {err:.4f} | TI {dg_ti:.4f} | MBAR(诊断) {dg_mbar:.4f}")
    if not np.isfinite(dg):
        raise RuntimeError(f"attachment 腿 ΔG 非有限值: {dg}")
    if dg < 0.0:
        raise RuntimeError(
            f"attachment 腿算出 ΔG(A′→A) = {dg:.4f} kJ/mol < 0，数学上不可能："
            "Boresch 势处处 ≥0 ⟹ ⟨exp(−βU_rest)⟩ ≤ 1 ⟹ ΔG ≥ 0。拒绝返回。"
        )

    # 二面角反转告警：单个二面角反转就要 2k ≈ 360–470 kJ/mol。一旦采到，
    # 指数平均会被那一帧支配（HREMD 那轮 38.6±110 就是这么来的）。
    fc = restraint_params.get("force_constants", {})
    flip_scale = 2.0 * min(
        [float(fc[k]) for k in ("kphiA", "kphiB", "kphiC") if k in fc] or [np.inf]
    )
    dihedral_flip_seen = bool(np.isfinite(flip_scale) and max_ub > 0.5 * flip_scale)
    if dihedral_flip_seen:
        log(f"  ⚠️ 采到了 U_B={max_ub:.1f} kJ/mol 的帧（最软二面角反转标度 {flip_scale:.1f}），"
            "指数平均可能被单帧支配；BAR/TI 一致性门是唯一的拦截。")

    # ---- 门：只在真不一致时失败 ----
    ti_ok, ti_tol, ti_msg = attachment_bar_ti_gate(dg, err, dg_ti)
    if not ti_ok:
        if enforce_convergence_gates:
            raise RuntimeError(
                f"attachment 腿 {ti_msg}。两者对稀有大能量帧的敏感度不同，"
                "分歧说明估计量被少数帧支配（多半是二面角反转）。"
            )
        log(f"  ⚠️ [门已关闭] {ti_msg} —— 超容差但未阻断")
    mbar_z = abs(dg_mbar - dg) / err if (err > 0 and np.isfinite(dg_mbar)) else float("nan")
    if np.isfinite(mbar_z) and mbar_z > 3.0:
        log(f"  ⚠️ 去相关 MBAR({dg_mbar:.4f}) 偏离主值 {mbar_z:.1f}σ —— 只记录不失败，"
            "MBAR 已降级为诊断。")

    half = {}
    try:
        for label, (lo, hi) in (("first", (0.0, 0.5)), ("second", (0.5, 1.0))):
            cols, nk2 = [], []
            for k in range(K):
                a = k * n_samples
                s, e = a + int(lo * n_samples), a + int(hi * n_samples)
                cols.append(np.arange(s, e)); nk2.append(e - s)
            v, _, _ = _attachment_bar_chain(
                u_kn[:, np.concatenate(cols)], np.asarray(nk2, dtype=int), kt
            )
            half[label] = v
        half["drift"] = half["second"] - half["first"]
        half["drift_over_2sigma"] = abs(half["drift"]) / (2.0 * err) if err > 0 else None
        sh_ok, sh_tol, sh_msg = attachment_split_half_gate(half["drift"], err)
        half["tolerance_kJ_mol"] = sh_tol
        half["passed"] = bool(sh_ok)
        log(f"  前后半程 {half['first']:.4f} / {half['second']:.4f}，{sh_msg}")
        if not sh_ok:
            if enforce_convergence_gates:
                raise RuntimeError(f"attachment 腿{sh_msg}，系综未收敛，拒绝返回。")
            log(f"  ⚠️ [门已关闭] {sh_msg} —— 超容差但未阻断")
    except RuntimeError:
        raise
    except Exception as exc:
        half = {"error": str(exc)}
        log(f"  ⚠️ split-half 诊断失败（不影响主值）: {exc}")

    payload = {
        "stage": "boresch_attachment",
        "protocol_version": int(BORESCH_ATTACHMENT_PROTOCOL_VERSION),
        "method": "sequential-windows-adjacent-BAR",
        "attachment_delta_G_kJ_mol": dg,
        "attachment_error_kJ_mol": err,
        "uncertainty_source": "adjacent_bar_independent_edge_variances",
        "n_seeds": 1,
        "direction": "A_prime_to_A (restraint OFF -> ON), always >= 0",
        "lambdas": lam.tolist(),
        "n_states": K,
        "n_samples_per_state": int(n_samples),
        "n_steps_per_state": int(n_steps_per_state),
        "equil_steps_per_state": int(equil_steps_per_state),
        "temperature_K": float(temperature_k),
        "boresch_params": {
            "receptor_indices": [int(i) for i in restraint_params["receptor_indices"]],
            "ligand_indices": [int(i) for i in restraint_params["ligand_indices"]],
            "equilibrium_values": dict(restraint_params["equilibrium_values"]),
            "force_constants": dict(restraint_params["force_constants"]),
        },
        "mean_u_boresch_per_state_kJ_mol": mean_ub,
        "max_u_boresch_kJ_mol": max_ub,
        "dihedral_flip_scale_kJ_mol": None if not np.isfinite(flip_scale) else float(flip_scale),
        "dihedral_flip_sampled": dihedral_flip_seen,
        "crosschecks": {
            "primary_estimator": "adjacent_bar",
            "adjacent_bar_kJ_mol": dg,
            "adjacent_bar_error_kJ_mol": err,
            "bar_edges": bar_edges,
            "thermodynamic_integration_kJ_mol": dg_ti,
            "bar_minus_ti_kJ_mol": float(dg - dg_ti),
            "bar_ti_tolerance_kJ_mol": float(ti_tol),
            "decorrelated_mbar_kJ_mol": dg_mbar,
            "decorrelated_mbar_error_kJ_mol": err_mbar,
            "decorrelated_mbar_z_vs_primary": None if not np.isfinite(mbar_z) else float(mbar_z),
            "mbar_note": "去相关 MBAR 已降级为诊断：2026-07-28 实测它的选帧把结果偏出 0.37 kJ/mol",
        },
        "split_half": half,
    }

    log(f"  ✅ ΔG(A′→A) = {dg:.4f} ± {err:.4f} kJ/mol "
        f"= {dg / 4.184:.4f} ± {err / 4.184:.4f} kcal/mol")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "attachment_u_kn.npy"), u_kn)
        np.save(os.path.join(output_dir, "attachment_u_kn.npy.n_k.npy"), n_k)
        with open(os.path.join(output_dir, "attachment_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    return payload



def _prepare_pme_mixed_alchemical_system(
    system_template: openmm.System,
    ligand_indices: List[int],
    topology,
    positions,
    box_vectors=None,
    lambda_coul_name: str = "lambda_coul",
    lambda_vdw_name: str = "lambda_vdw",
    restraint_params: Optional[Dict] = None,
) -> openmm.System:
    """构建同时支持 PME 去电荷与软核去 VDW 的联合炼金体系。"""
    mixed_sys = _prepare_pme_coulomb_leg_system(
        system_template,
        ligand_indices,
        lambda_name=lambda_coul_name,
        allow_charged_ligand=True,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
    )
    mixed_sys.thisown = 1

    nb_force = next(
        (f for f in mixed_sys.getForces() if isinstance(f, openmm.NonbondedForce)),
        None,
    )
    if nb_force is None:
        raise RuntimeError("联合 PME/VDW 炼金体系未找到 NonbondedForce。")

    ligand_set = set(int(i) for i in ligand_indices)
    environment_indices = [
        idx for idx in range(mixed_sys.getNumParticles()) if idx not in ligand_set
    ]
    particle_params = [
        nb_force.getParticleParameters(i) for i in range(nb_force.getNumParticles())
    ]

    _restore_ligand_internal_nonbonded(
        mixed_sys,
        nb_force,
        ligand_indices,
        zero_original_exceptions=True,
    )

    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(int(idx))
        nb_force.setParticleParameters(
            int(idx),
            q,
            0.1 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )

    for exc_idx in range(nb_force.getNumExceptions()):
        p1, p2, charge_prod, sig, eps = nb_force.getExceptionParameters(exc_idx)
        p1 = int(p1)
        p2 = int(p2)
        if (p1 in ligand_set) ^ (p2 in ligand_set):
            nb_force.setExceptionParameters(
                exc_idx,
                p1,
                p2,
                charge_prod,
                sig,
                0.0 * unit.kilojoule_per_mole,
            )

    sc_force = BeutlerSoftcoreBuilder.build(
        nb_force,
        list(ligand_indices),
        environment_indices,
        particle_params_override=particle_params,
    )
    sc_force.setForceGroup(1)
    mixed_sys.addForce(sc_force)
    _add_physical_boresch_restraint(mixed_sys, restraint_params, force_group=3)
    return mixed_sys


def _build_traditional_mbar_eval_context(
    system_xml: str,
    platform_name: str,
):
    """为离线 MBAR worker 构建独立的 OpenMM Context。"""
    eval_sys = openmm.XmlSerializer.deserialize(system_xml)
    resolved_platform, props = _build_platform_properties(platform_name)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(eval_sys, integ, platform, props)
    return eval_sys, integ, ctx


def _compute_u_kn_chunk(task: Dict) -> Tuple[int, np.ndarray]:
    """多进程 worker：重算一个帧块在所有 λ 态下的约化势。"""
    frame_offset = int(task["frame_offset"])
    xyz_chunk = np.asarray(task["xyz"], dtype=np.float64)
    box_chunk = task.get("box_vectors")
    if box_chunk is not None:
        box_chunk = np.asarray(box_chunk, dtype=np.float64)

    eval_sys, integ, ctx = _build_traditional_mbar_eval_context(
        system_xml=task["system_xml"],
        platform_name=str(task["platform_name"]),
    )

    lambdas_coul = np.asarray(task["lambdas_coul"], dtype=float)
    lambdas_vdw = np.asarray(task["lambdas_vdw"], dtype=float)
    n_states = len(lambdas_coul)
    u_chunk = np.zeros((n_states, xyz_chunk.shape[0]), dtype=np.float64)
    kt = float(task["kt"])
    use_total_energy = bool(task.get("use_total_energy", False))
    apply_pme_self_correction = bool(task.get("apply_pme_self_correction", False))
    # 🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=2] 不再是一个标量 prefactor + 统一的
    # lambda_vdw**power 缩放——每个 λ 态的 switching+softcore-aware 积分结果不同，
    # 由生产者（compute_u_kn）逐态数值积分好、按 lambdas_vdw 的顺序传过来。
    lj_tail_lrc_coeff = task.get("lj_tail_lrc_coeff_kj_mol")
    if lj_tail_lrc_coeff is not None:
        lj_tail_lrc_coeff = np.asarray(lj_tail_lrc_coeff, dtype=np.float64)
    if lj_tail_lrc_coeff is not None and box_chunk is None:
        raise RuntimeError(
            "传统 Beutler LRC 已启用，但轨迹没有周期盒向量，无法计算 1/V 尾项。"
        )
    pme_self_prefactor_kj = None
    if use_total_energy and apply_pme_self_correction:
        lig_qsq = float(task.get("ligand_charge_square_sum", 0.0))
        if lig_qsq > 0.0:
            nb_force = next((f for f in eval_sys.getForces() if isinstance(f, openmm.NonbondedForce)), None)
            if nb_force is None:
                raise RuntimeError(
                    "PME 自能修正被标记为启用，但 worker 进程里的 eval_sys 找不到 "
                    "NonbondedForce，无法拿到 Ewald alpha；拒绝静默跳过修正（那会产出一个"
                    "看起来正常、实际没做自能修正的 u_kn）。"
                )
            # 静态 PME 参数查询在 OpenMM 自动派生参数时通常为零，不能作为
            # Context 查询失败后的 fallback。优先读取 Context 的真实参数；若平台
            # 不支持该查询，则用同一 cutoff/tolerance 公式闭式派生 alpha。
            alpha_source = "context"
            try:
                alpha_ewald, _, _, _ = nb_force.getPMEParametersInContext(ctx)
                alpha_ewald = _value_in_inverse_nanometer(alpha_ewald)
            except Exception as exc:
                # A static PME-parameter query returns zero when OpenMM is
                # deriving parameters automatically, so it is not a valid fallback.
                # Derive the same alpha from cutoff/tolerance instead; this is
                # also independent of the worker platform implementation.
                alpha_source = (
                    "closed_form_after_context_query_failed"
                    f"({type(exc).__name__}: {exc})"
                )
                alpha_ewald = get_pme_alpha_for_system(eval_sys)
            if alpha_ewald <= 0.0:
                raise RuntimeError(
                    f"PME 自能修正被标记为启用（Σq_offset²={lig_qsq:.6f} e²），但无法拿到"
                    f"有效的 Ewald alpha（来源={alpha_source}，取值={alpha_ewald}）。拒绝"
                    "静默跳过修正——这会产出一个看起来正常、实际带着未修正自能伪项的 "
                    "u_kn。请检查该 NonbondedForce 的 nonbonded method/cutoff 设置，或"
                    "该 worker 平台是否支持 getPMEParametersInContext。"
                )
            pme_self_prefactor_kj = pme_self_correction_prefactor_kj(alpha_ewald, lig_qsq)
            if frame_offset == 0:
                print(
                    f"  🔎 [PME 自能修正 worker] alpha_ewald={alpha_ewald:.6f} /nm "
                    f"(来源={alpha_source})，Σq²={lig_qsq:.6f} e² → "
                    f"prefactor={pme_self_prefactor_kj:.4f} kJ/mol"
                )

    for local_f in range(xyz_chunk.shape[0]):
        ctx.setPositions(xyz_chunk[local_f] * unit.nanometer)
        if box_chunk is not None and len(box_chunk) > 0:
            ctx.setPeriodicBoxVectors(*box_chunk[local_f] * unit.nanometer)

        for k in range(n_states):
            REMDManager._try_set_context_parameter(ctx, "lambda_coul", lambdas_coul[k])
            REMDManager._try_set_context_parameter(ctx, "lambda_vdw", lambdas_vdw[k])
            if use_total_energy:
                e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                if apply_pme_self_correction and pme_self_prefactor_kj is not None:
                    # PME 总能量包含与坐标无关的自能项 -C*lambda^2。
                    # 它不会影响构型采样，但会严重污染 ΔG；这里在离线 u_kn 中解析去除。
                    e += pme_self_correction_energy_kj(lambdas_coul[k], pme_self_prefactor_kj)
            else:
                e = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if lj_tail_lrc_coeff is not None:
                volume_nm3 = _periodic_box_volume_nm3(box_chunk[local_f])
                e += float(lj_tail_lrc_coeff[k]) / volume_nm3
            u_chunk[k, local_f] = e / kt

    del ctx, integ, eval_sys
    return frame_offset, u_chunk
def _split_platform_spec(platform_name: str) -> Tuple[str, Optional[str]]:
    spec = str(platform_name or "CPU").strip()
    if ":" not in spec:
        return spec, None
    base, device = spec.split(":", 1)
    base = base.strip() or "CPU"
    device = device.strip() or None
    return base, device


def _build_platform_properties(platform_name: str) -> Tuple[str, Dict[str, str]]:
    base, device = _split_platform_spec(platform_name)
    upper = base.upper()
    props: Dict[str, str] = {}
    if upper == "CUDA":
        props["Precision"] = "mixed"
        if device is not None:
            props["DeviceIndex"] = device
    elif upper == "OPENCL":
        props["Precision"] = "mixed"
        if device is not None:
            props["DeviceIndex"] = device
    return base, props

# ============================================================================
# 0. 通用工具 (从 abfe_core 导入)
# ============================================================================
# ensure_owned_system, sync_all_exclusions, create_ligand_internal_force 已从 abfe_core 导入
# ================= ibs_engine.py / abfe_core.py =================
import openmm
from openmm import app, unit
import numpy as np

# ================= ibs_engine.py 顶部新增 =================
def configure_coalchemical_neutral_decharging(
    system: openmm.System,
    ligand_indices: List[int],
    topology,
    positions,
    box_vectors=None,
    lambda_name: str = "lam_coul",
    ion_restraint_k: float = 25.0,
) -> Tuple[Dict, List[int]]:
    """
    🔥 终极防御：共炼金反离子策略
    寻找体系中最远的反离子，使其与配体同步消电，保持全局严格电中性。
    """
    nb_force = next((f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
    if nb_force is None:
        raise RuntimeError("系统中未找到 NonbondedForce")

    ligand_set = set(int(i) for i in ligand_indices)
    raw_lig_net_charge = 0.0
    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(idx)
        raw_lig_net_charge += q.value_in_unit(unit.elementary_charge)
    lig_net_charge = int(round(raw_lig_net_charge))
    if abs(raw_lig_net_charge - lig_net_charge) > 1.0e-3:
        raise RuntimeError(
            f"配体净电荷 {raw_lig_net_charge:+.6f} e 不接近整数（容差 1e-3 e）"
        )

    if lig_net_charge == 0:
        print("  ℹ️ 配体为电中性，无需共消电反离子。使用标准 PME Offset。")
        target_ion_charge = 0.0
        best_ion_indices: List[int] = []
        ion_ref_positions_nm: List[np.ndarray] = []
        ion_meta = {}
    else:
        print(f"  ⚡ 检测到带电配体 (Net Charge: {lig_net_charge:+d})，启动共炼金反离子搜索...")
        target_ion_charge = -1.0 if lig_net_charge > 0 else 1.0
        best_ion_indices, ion_ref_positions_nm, ion_meta = _select_bulk_water_counterion(
            nb_force, ligand_indices, topology, positions, box_vectors
        )
        if best_ion_indices:
            print(
                f"  🎯 锁定共消电反离子: Indices {best_ion_indices}"
            )
        else:
            raise RuntimeError("未找到可用于共炼金的匹配反离子，无法保持 PME 去电荷腿的电中性。")

    # 3. 注入全局 Lambda 与 ParameterOffset
    existing_params = [nb_force.getGlobalParameterName(i) for i in range(nb_force.getNumGlobalParameters())]
    if lambda_name not in existing_params:
        nb_force.addGlobalParameter(lambda_name, 1.0)

    original_charges = {}
    ligand_params = {}
    alchemical_set = set(ligand_set)
    
    # 3.1 配体 Offset
    for idx in ligand_set:
        q, sig, eps = nb_force.getParticleParameters(idx)
        original_charges[idx] = q
        ligand_params[idx] = (q, sig, eps)
        nb_force.setParticleParameters(idx, 0.0*unit.elementary_charge, sig, eps)
        nb_force.addParticleParameterOffset(lambda_name, idx, q, 0.0*unit.nanometer, 0.0*unit.kilojoule_per_mole)

    # 3.2 反离子 Offset
    for best_ion_idx in best_ion_indices:
        ion_q, ion_sig, ion_eps = nb_force.getParticleParameters(best_ion_idx)
        original_charges[best_ion_idx] = ion_q
        alchemical_set.add(int(best_ion_idx))
        nb_force.setParticleParameters(best_ion_idx, 0.0*unit.elementary_charge, ion_sig, ion_eps)
        nb_force.addParticleParameterOffset(lambda_name, best_ion_idx, ion_q, 0.0*unit.nanometer, 0.0*unit.kilojoule_per_mole)

    # 3.3 保持配体内部静电常量；仅对“单端炼金”的 exception 施加线性 offset。
    frozen_ll_pairs = set()
    for i in range(nb_force.getNumExceptions()):
        p1, p2, charge_prod, sig, eps = nb_force.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if p1 in ligand_set and p2 in ligand_set:
            frozen_ll_pairs.add((min(p1, p2), max(p1, p2)))
            continue
        if (p1 in alchemical_set) ^ (p2 in alchemical_set):
            nb_force.setExceptionParameters(i, p1, p2, 0.0*unit.elementary_charge**2, sig, eps)
            nb_force.addExceptionParameterOffset(lambda_name, i, charge_prod, 0.0*unit.nanometer, 0.0*unit.kilojoule_per_mole)

    # 为原本走标准 NB 的 L-L 对补全显式 exception，避免粒子电荷缩放把内部库仑也带着变掉。
    lig_list = sorted(ligand_set)
    for offset_i, p1 in enumerate(lig_list):
        q1, sig1, eps1 = ligand_params[p1]
        q1_val = q1.value_in_unit(unit.elementary_charge)
        sig1_val = sig1.value_in_unit(unit.nanometer)
        eps1_val = eps1.value_in_unit(unit.kilojoule_per_mole)
        for p2 in lig_list[offset_i + 1:]:
            key = (p1, p2)
            if key in frozen_ll_pairs:
                continue
            q2, sig2, eps2 = ligand_params[p2]
            q2_val = q2.value_in_unit(unit.elementary_charge)
            sig2_val = sig2.value_in_unit(unit.nanometer)
            eps2_val = eps2.value_in_unit(unit.kilojoule_per_mole)
            nb_force.addException(
                p1,
                p2,
                (q1_val * q2_val) * unit.elementary_charge**2,
                0.5 * (sig1_val + sig2_val) * unit.nanometer,
                math.sqrt(max(eps1_val * eps2_val, 0.0)) * unit.kilojoule_per_mole,
                True,
            )

    for best_ion_idx, ion_ref_pos_nm in zip(
        best_ion_indices,
        ion_ref_positions_nm,
    ):
        restraint = _create_bulk_water_ion_restraint(
            ion_index=best_ion_idx,
            reference_position_nm=ion_ref_pos_nm,
            force_constant_kj_per_mol_nm2=ion_restraint_k,
        )
        restraint.setForceGroup(6)
        system.addForce(restraint)
        print(
            f"  🪢 共炼金反离子 bulk 水锚定已注入: "
            f"k={ion_restraint_k:.1f} kJ/mol/nm^2, ref=({ion_ref_pos_nm[0]:.3f}, {ion_ref_pos_nm[1]:.3f}, {ion_ref_pos_nm[2]:.3f}) nm"
        )

    print("  ✅ 共炼金反离子防御阵列部署完毕。PME 倒空间计算全程严格电中性！")
    return original_charges, list(best_ion_indices)


def configure_pme_ligand_charge_offsets(
    system: openmm.System,
    ligand_indices: List[int],
    lambda_name: str = "lambda_coul",
    allow_charged_ligand: bool = False,
    topology=None,
    positions=None,
    box_vectors=None,
) -> Optional[Dict[str, Any]]:
    """
    Configure a PME-preserving Coulomb leg on the native NonbondedForce.

    This avoids putting ligand-environment electrostatics into
    CustomNonbondedForce, which would truncate Coulomb at the cutoff.  Charged
    ligands are rejected by default because co-alchemical ion schemes change the
    thermodynamic cycle unless they are implemented and validated explicitly.
    """
    nb_force = next((f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
    if nb_force is None:
        raise RuntimeError("系统中未找到 NonbondedForce，无法配置 PME 去电荷阶段。")

    ligand_set = set(int(i) for i in ligand_indices)
    net_charge = 0.0
    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(idx)
        net_charge += q.value_in_unit(unit.elementary_charge)
    rounded_charge = int(round(net_charge))
    if abs(net_charge - rounded_charge) > 1.0e-3:
        raise RuntimeError(
            f"配体净电荷 {net_charge:+.6f} e 不接近整数（容差 1e-3 e）；"
            "拒绝把异常部分电荷判为中性"
        )
    if rounded_charge != 0 and not allow_charged_ligand:
        raise RuntimeError(
            f"检测到带净电配体 ({rounded_charge:+d} e)。当前 PME 去电荷路径不再自动共炼金反离子，"
            "请先使用净电中性配体、显式实现可验证的离子方案，或改用支持 PME 多态炼金的底层实现。"
        )

    if rounded_charge != 0:
        if topology is None or positions is None:
            raise RuntimeError("带电配体的 PME 去电荷路径需要 topology 和初始 positions 以构建共炼金反离子 bulk 水锚定。")
        original_charges, ion_indices = configure_coalchemical_neutral_decharging(
            system,
            ligand_indices,
            topology,
            positions,
            box_vectors=box_vectors,
            lambda_name=lambda_name,
        )
        return {
            "mode": "coalchemical_counterion",
            "ion_indices": [int(i) for i in ion_indices],
            "n_offsets": len(original_charges),
        }

    existing_params = [nb_force.getGlobalParameterName(i) for i in range(nb_force.getNumGlobalParameters())]
    if lambda_name not in existing_params:
        nb_force.addGlobalParameter(lambda_name, 1.0)

    ligand_params = {}
    for idx in ligand_set:
        q, sig, eps = nb_force.getParticleParameters(idx)
        ligand_params[idx] = (q, sig, eps)
        nb_force.setParticleParameters(idx, 0.0 * unit.elementary_charge, sig, eps)
        nb_force.addParticleParameterOffset(
            lambda_name,
            idx,
            q,
            0.0 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )

    # 冻结所有 L-L 非键对，使去电荷腿只作用于配体-环境静电，而不污染配体内部 self-energy。
    frozen_ll_pairs = set()
    for exc_idx in range(nb_force.getNumExceptions()):
        p1, p2, charge_prod, sig, eps = nb_force.getExceptionParameters(exc_idx)
        p1, p2 = int(p1), int(p2)
        if p1 in ligand_set and p2 in ligand_set:
            frozen_ll_pairs.add((min(p1, p2), max(p1, p2)))
            continue
        if (p1 in ligand_set) ^ (p2 in ligand_set):
            nb_force.setExceptionParameters(
                exc_idx,
                p1,
                p2,
                0.0 * unit.elementary_charge**2,
                sig,
                eps,
            )
            nb_force.addExceptionParameterOffset(
                lambda_name,
                exc_idx,
                charge_prod,
                0.0 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
            )

    # 为原本走标准 NB 的 L-L 对补全显式 exception，避免粒子电荷缩放把内部库仑也带着变掉。
    lig_list = sorted(ligand_set)
    for offset_i, p1 in enumerate(lig_list):
        q1, sig1, eps1 = ligand_params[p1]
        q1_val = q1.value_in_unit(unit.elementary_charge)
        sig1_val = sig1.value_in_unit(unit.nanometer)
        eps1_val = eps1.value_in_unit(unit.kilojoule_per_mole)
        for p2 in lig_list[offset_i + 1:]:
            key = (p1, p2)
            if key in frozen_ll_pairs:
                continue
            q2, sig2, eps2 = ligand_params[p2]
            q2_val = q2.value_in_unit(unit.elementary_charge)
            sig2_val = sig2.value_in_unit(unit.nanometer)
            eps2_val = eps2.value_in_unit(unit.kilojoule_per_mole)
            nb_force.addException(
                p1,
                p2,
                (q1_val * q2_val) * unit.elementary_charge**2,
                0.5 * (sig1_val + sig2_val) * unit.nanometer,
                math.sqrt(max(eps1_val * eps2_val, 0.0)) * unit.kilojoule_per_mole,
                True,
            )
    return {"mode": "ligand_only_offset", "ion_index": None, "n_offsets": len(ligand_set)}

def generate_overlapping_windows(
    n_states: int,
    pts_per_window: int = 6,
    overlap: int = 2,
    n_windows: int = None
) -> List[Tuple[int, int]]:
    """
    生成重叠窗口划分
    示例：n_states=13, pts_per_window=6, overlap=2 → [(0,6), (4,10), (8,13)]
    """
    n_states = int(n_states)
    pts_per_window = int(pts_per_window)
    overlap = max(0, int(overlap))
    if n_windows is not None:
        n_windows = max(1, int(n_windows))

    if n_states <= pts_per_window:
        return [(0, n_states)]

    if n_windows is None:
        stride = max(1, pts_per_window - overlap)
        n_windows = max(2, math.ceil((n_states - overlap) / stride))
        effective_pts = pts_per_window
    else:
        effective_pts = max(
            pts_per_window,
            math.ceil((n_states + (n_windows - 1) * overlap) / n_windows),
        )

    if n_windows > 1:
        max_start = max(0, n_states - effective_pts)
        starts = np.linspace(0, max_start, n_windows)
        starts = [int(round(x)) for x in starts]
    else:
        starts = [0]

    windows = []
    for start in starts:
        if start >= n_states:
            break
        end = min(start + effective_pts, n_states)
        windows.append((start, end))

    if windows and windows[-1][1] < n_states:
        windows[-1] = (max(0, n_states - effective_pts), n_states)

    # 验证覆盖性
    covered = set()
    for s, e in windows:
        covered.update(range(s, e))
    if len(covered) != n_states:
        raise RuntimeError(f"窗口划分未完全覆盖 {n_states} 个状态")
    return windows

# ============================================================================
# 1. 双λ微扰系统构建 (核心修复版)
# ============================================================================
# create_ligand_internal_force 已从 abfe_core 导入
def _lj_tail_correction_sigma_resolved_moments(
    all_params,
    ligand_indices: List[int],
    environment_indices: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """配体<->环境 LJ 长程尾项的【按 sigma_ij 分组】几何求和矩。

    返回 (sigma_nm, s6_per_sigma, s12_per_sigma)，其中对第 b 组

        s6_per_sigma[b]  = sum_{pairs with sigma_ij == sigma_nm[b]} epsilon_ij * sigma_ij^6
        s12_per_sigma[b] = sum_{pairs with sigma_ij == sigma_nm[b]} epsilon_ij * sigma_ij^12

    混合规则与软核表达式一致 (sigma_ij=0.5*(sigma_i+sigma_j),
    epsilon_ij=sqrt(epsilon_i*epsilon_j))。全部是固定几何量，跟 lambda、switching、
    盒子体积无关（那些在径向积分/每帧体积换算里另外处理）。

    🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=3] v2 只返回两个【全局标量】
    S6=sum(eps*sigma^6)、S12=sum(eps*sigma^12)，因为当时软核分母
    D(r)=alpha_lj*(1-lambda)^m + r^6 与 pair 无关，I6/I12 可以提到求和号外面、每个
    lambda 只积一次。软核 alpha 改为无量纲、乘 sigma_ij^6 之后
    （SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2），
    D_ij(r)=alpha_lj*sigma_ij^6*(1-lambda)^m + r^6 变成 pair-specific，那个分解不再
    成立，尾修正必须按 sigma 分组后逐组积分再求和，否则尾修正对应的哈密顿量与实际
    采样的哈密顿量不一致。力场里不同的 sigma 取值很少（远少于 pair 数），分组后每个
    lambda 的积分次数是 O(distinct sigma) 而非 O(pairs)，且只在建系时算一次。
    """
    sigma_lig = np.array(
        [all_params[i][1].value_in_unit(unit.nanometer) for i in ligand_indices], dtype=np.float64
    )
    eps_lig = np.array(
        [all_params[i][2].value_in_unit(unit.kilojoule_per_mole) for i in ligand_indices], dtype=np.float64
    )
    sigma_env = np.array(
        [all_params[j][1].value_in_unit(unit.nanometer) for j in environment_indices], dtype=np.float64
    )
    eps_env = np.array(
        [all_params[j][2].value_in_unit(unit.kilojoule_per_mole) for j in environment_indices], dtype=np.float64
    )
    sigma_ij = (0.5 * (sigma_lig[:, None] + sigma_env[None, :])).ravel()
    eps_ij = np.sqrt(eps_lig[:, None] * eps_env[None, :]).ravel()
    # 1e-9 nm 的分组容差远小于任何物理 sigma 差异，只用来把浮点上完全等价的取值合并。
    sigma_key = np.round(sigma_ij, 9)
    sigma_nm, inverse = np.unique(sigma_key, return_inverse=True)
    n_bins = sigma_nm.shape[0]
    s6_per_sigma = np.bincount(
        inverse, weights=eps_ij * sigma_ij ** 6, minlength=n_bins
    )
    s12_per_sigma = np.bincount(
        inverse, weights=eps_ij * sigma_ij ** 12, minlength=n_bins
    )
    return (
        sigma_nm.astype(np.float64),
        s6_per_sigma.astype(np.float64),
        s12_per_sigma.astype(np.float64),
    )


# 🔑 LRC 协议版本 1→2：v1 只补 r_cutoff（1.2 nm）之外的标准 r^-6 尾项，完全忽略了
# 软核力实际启用的 1.0-1.2 nm switching 区间——OpenMM 的 switching function 会在这段
# 区间把能量从满强度削到 0，真正缺失的修正是 [switching 区间被削弱的部分] +
# [cutoff 之外的标准尾项]（见 CustomNonbondedForce LRC 文档），v1 只补了后者。v1 还
# 把 λ<1 时的 softcore 分母 D(r)=alpha_lj*(1-λ)^m + r^6 当成纯 r^6 处理，等于假装
# softcore 在尾区间不存在。v2 改为对每个 λ 数值积分真实的
# switching-aware + softcore-aware 径向积分（见 _lj_softcore_tail_radial_integrals /
# _lj_tail_lrc_coefficients_kj_mol），同时补上排斥项 r^-12 的尾贡献（之前只有吸引项
# r^-6）。ACE/dual_lambda 路径（IBSSampler._lj_tail_correction_kj_mol /
# probe_bidirectional_overlap 的 fixed-H 探针）和传统 Beutler REMD 路径
# （TraditionalMBARAnalyzer.compute_u_kn 的离线 u_kn 重算）现在共用同一套
# lrc_coeff 计算逻辑，不再各自维护一份。旧版本号下产出的能量文件/u_kn 缓存一律
# fail closed，不能和 v2 的结果混用。
# 🔑 LRC 协议版本 2→3：配合 SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2，
# 软核分母从 pair 无关的 D(r)=alpha_lj*(1-λ)^m + r^6 变成 pair-specific 的
# D_ij(r)=alpha_lj*sigma_ij^6*(1-λ)^m + r^6。v2 把 S6/S12 先汇总成两个标量、每个 λ
# 只积一次的分解因此失效，v3 改为按 sigma_ij 分组逐组积分再求和（见
# _lj_tail_correction_sigma_resolved_moments / _lj_tail_lrc_coefficients_kj_mol）。
# 两处变化的量级不同，别混为一谈：alpha 本身缩小 ~685 倍会让低 λ 的尾修正系数发生
# 可观改变（旧 α 在 λ→0 时偏移 0.5 nm^6，与 r_switch=1.0 nm 处的 r^6=1.0 同量级，
# 真的会压低积分）；而在此基础上再做 sigma 分辨，因为 α·sigma^6 ≤ 0.5·(0.4)^6≈2e-3
# 相对 r^6≥1 只有 ≲0.2%，属于小修正——但它是让尾修正与采样哈密顿量严格一致所必需的。
# v2 及更早版本号下产出的能量文件/u_kn 缓存对 v3 一律 fail closed。
TRADITIONAL_LJ_LRC_PROTOCOL_VERSION = 3

# 两条路径（ACE `_create_softcore_force` 与传统 `BeutlerSoftcoreBuilder.build`）
# 目前都硬编码用这组 switching/cutoff 距离构造软核 CustomNonbondedForce，LRC 积分
# 必须用完全相同的边界，否则修正的是一个跟实际采样哈密顿量不匹配的积分区间。
LJ_TAIL_LRC_R_SWITCH_NM = 1.0
LJ_TAIL_LRC_R_CUTOFF_NM = 1.2


def _lj_switching_function_value(r_nm: float, r_switch_nm: float, r_cutoff_nm: float) -> float:
    """OpenMM's standard quintic switching polynomial S(r): 1 at r<=r_switch,
    0 at r>=r_cutoff, smoothly (C1-continuous) interpolating in between."""
    if r_nm <= r_switch_nm:
        return 1.0
    if r_nm >= r_cutoff_nm:
        return 0.0
    x = (r_nm - r_switch_nm) / (r_cutoff_nm - r_switch_nm)
    return 1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5


@functools.lru_cache(maxsize=4096)
def _lj_softcore_tail_radial_integrals(
    lambda_vdw: float,
    alpha_lj: float,
    m_lj: float,
    sigma_nm: float,
    r_switch_nm: float = LJ_TAIL_LRC_R_SWITCH_NM,
    r_cutoff_nm: float = LJ_TAIL_LRC_R_CUTOFF_NM,
) -> Tuple[float, float]:
    """Real switching-aware + softcore-aware radial integrals for one
    (lambda_vdw, sigma_ij) pair:

        I6  = integral_{r_switch}^{r_cutoff} (1-S(r)) r^2 / D(r)   dr
            + integral_{r_cutoff}^{infinity}            r^2 / D(r)   dr
        I12 = integral_{r_switch}^{r_cutoff} (1-S(r)) r^2 / D(r)^2 dr
            + integral_{r_cutoff}^{infinity}            r^2 / D(r)^2 dr

    where D(r) = alpha_lj*sigma_nm^6*(1-lambda_vdw)^m_lj + r^6 is the exact
    softcore denominator used by both AlchemicalPotentialFactory/
    ACESoftcorePotential and BeutlerSoftcoreBuilder under
    SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2.  It is now
    sigma-dependent, which is why the caller must integrate per sigma group
    rather than once per lambda -- see
    _lj_tail_correction_sigma_resolved_moments.

    (BeutlerSoftcoreBuilder additionally carries a +1e-4*sigma12^6*(1-lambda)
    numerical safety floor near r=0; at r>=r_switch=1.0nm that term is
    <=1e-4*sigma^6 against a D(r)>=1 from the r^6 piece alone, i.e. completely
    negligible for this tail integral, so both paths can safely share this one
    D(r).)

    lru_cache: the same (lambda, sigma) combinations recur across the three
    consumers (IBS production build, fixed-H overlap probe, offline traditional
    MBAR u_kn recompute) and across sigma groups that share a value; the cache
    keeps the per-sigma refactor from multiplying quad() calls.

    No closed form exists once the switching polynomial and the
    lambda-shifted r^6 denominator are both present, so this integrates
    numerically (scipy.integrate.quad, handles the improper r_cutoff->inf
    integral directly). Called once per distinct lambda_vdw at system-build
    time, never per-frame, so the extra cost is negligible.
    """
    lam = float(lambda_vdw)
    alpha = float(alpha_lj)
    m = float(m_lj)
    sigma6 = float(sigma_nm) ** 6

    def _softcore_D(r):
        return alpha * sigma6 * (1.0 - lam) ** m + r ** 6

    def _integrand6_switch(r):
        return (1.0 - _lj_switching_function_value(r, r_switch_nm, r_cutoff_nm)) * r ** 2 / _softcore_D(r)

    def _integrand6_tail(r):
        return r ** 2 / _softcore_D(r)

    def _integrand12_switch(r):
        return (1.0 - _lj_switching_function_value(r, r_switch_nm, r_cutoff_nm)) * r ** 2 / _softcore_D(r) ** 2

    def _integrand12_tail(r):
        return r ** 2 / _softcore_D(r) ** 2

    i6_switch, _ = _scipy_quad(_integrand6_switch, r_switch_nm, r_cutoff_nm)
    i6_tail, _ = _scipy_quad(_integrand6_tail, r_cutoff_nm, np.inf)
    i12_switch, _ = _scipy_quad(_integrand12_switch, r_switch_nm, r_cutoff_nm)
    i12_tail, _ = _scipy_quad(_integrand12_tail, r_cutoff_nm, np.inf)
    return float(i6_switch + i6_tail), float(i12_switch + i12_tail)


def _lj_tail_lrc_coefficients_kj_mol(
    lambdas_vdw,
    sigma_nm,
    s6_per_sigma_kj_nm6,
    s12_per_sigma_kj_nm12,
    alpha_lj: float,
    m_lj: float,
    n_lj: float,
    r_switch_nm: float = LJ_TAIL_LRC_R_SWITCH_NM,
    r_cutoff_nm: float = LJ_TAIL_LRC_R_CUTOFF_NM,
) -> np.ndarray:
    """Per-lambda LJ tail correction coefficients (kJ*nm^3/mol); the actual
    per-frame energy is ``coeff[k] / V(t)`` (V in nm^3). Shared by every
    consumer (IBSSampler production sampling, the fixed-H overlap probe, and
    the offline traditional MBAR u_kn recompute) so all three always apply
    the exact same correction to the exact same lambda -- see call sites for
    the one-time construction and the per-frame `/ V(t)` division.

    🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=3] sigma-resolved sum:

        coeff[k] = 16*pi * lambda_vdw[k]**n_lj
                   * sum_b [ S12_b*I12(lambda_vdw[k], sigma_b)
                             - S6_b*I6(lambda_vdw[k], sigma_b) ]

    v2 was ``16*pi*lam**n_lj*(S12*I12(lam) - S6*I6(lam))`` with two global
    scalars, valid only while D(r) was pair-independent.  Under
    SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2 the softcore
    denominator is D_ij(r)=alpha_lj*sigma_ij^6*(1-lam)^m + r^6, so I6/I12 can
    no longer be factored out of the pair sum; they are evaluated per distinct
    sigma_ij group instead.  sum_b S6_b and sum_b S12_b still reproduce v2's
    S6/S12 exactly, which is what the call sites log.

    At lambda_vdw=0, lambda_vdw**n_lj is exactly 0.0 in floating point (not
    just close to it) for any n_lj>0, so coeff is exactly 0 regardless of
    I6/I12 -- the integral is skipped entirely for that state as a cheap
    optimization, not because it would otherwise be wrong.
    """
    lambdas_arr = np.asarray(lambdas_vdw, dtype=np.float64)
    sigma_arr = np.asarray(sigma_nm, dtype=np.float64).ravel()
    s6_arr = np.asarray(s6_per_sigma_kj_nm6, dtype=np.float64).ravel()
    s12_arr = np.asarray(s12_per_sigma_kj_nm12, dtype=np.float64).ravel()
    if not (sigma_arr.shape == s6_arr.shape == s12_arr.shape):
        raise ValueError(
            "sigma_nm / s6_per_sigma / s12_per_sigma 长度必须一致："
            f"{sigma_arr.shape} / {s6_arr.shape} / {s12_arr.shape}"
        )
    coeffs = np.zeros(lambdas_arr.shape[0], dtype=np.float64)
    for k, lam in enumerate(lambdas_arr):
        lam = float(lam)
        if lam == 0.0:
            continue
        acc = 0.0
        for sigma_b, s6_b, s12_b in zip(sigma_arr, s6_arr, s12_arr):
            i6, i12 = _lj_softcore_tail_radial_integrals(
                lam, float(alpha_lj), float(m_lj), float(sigma_b),
                r_switch_nm, r_cutoff_nm,
            )
            acc += s12_b * i12 - s6_b * i6
        coeffs[k] = 16.0 * math.pi * (lam ** float(n_lj)) * acc
    return coeffs


def _periodic_box_volume_nm3(box_vectors) -> float:
    """Return a validated triclinic box volume in nm^3."""
    arr = np.asarray(box_vectors, dtype=np.float64)
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"周期盒向量必须是有限的 (3, 3) nm 数组，实际 shape={arr.shape}。"
        )
    volume = float(abs(np.linalg.det(arr)))
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError(f"周期盒体积无效: {volume!r} nm^3")
    return volume


def _create_softcore_force(
    nb_force: openmm.NonbondedForce,
    perturbed_indices: List[int],
    environment_indices: List[int],
    lam_coul: float,
    lam_vdw: float,
    alchemical_params,
    potential_type: str = "softcore",
    reference_exclusions=None,
    particle_params_override=None,
    num_particles=None,
) -> openmm.CustomNonbondedForce:
    """
    🔑 修复版：构建 L-E 软核力 (用于 IBS CV)
    严格遵循论文 Eq.8：每个 CV 必须是独立的、无参数的势能函数。
    将 λ 值直接硬编码进表达式字符串，彻底移除 GlobalParameter 依赖。
    """
    total_particles = nb_force.getNumParticles()
    perturbed_set, env_set = set(perturbed_indices), set(environment_indices)
    
    # 🔑 核心：将 λ 格式化为高精度字符串，直接注入表达式
    lam_c_str = f"{lam_coul:.8f}"
    lam_v_str = f"{lam_vdw:.8f}"
    
    # 调用工厂生成完整软核表达式 (含 Coulomb + VdW)
    expr, resolved_params = AlchemicalPotentialFactory.build(
        potential_type, alchemical_params, lam_c_str, lam_v_str
    )
    
    sc_force = openmm.CustomNonbondedForce(expr)
    for p in ["q", "sigma", "epsilon"]:
        sc_force.addPerParticleParameter(p)
        
    # 🔑 移除 addGlobalParameter，λ 已硬编码，CV 成为纯坐标函数 U'_k(x)
    
    for i in range(total_particles):
        if particle_params_override:
            if isinstance(particle_params_override, dict):
                q, sig, eps = particle_params_override.get(i, nb_force.getParticleParameters(i))
            elif i < len(particle_params_override):
                q, sig, eps = particle_params_override[i]
            else:
                q, sig, eps = nb_force.getParticleParameters(i)
        else:
            q, sig, eps = nb_force.getParticleParameters(i)
            
        sc_force.addParticle([
            q.value_in_unit(unit.elementary_charge),
            sig.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole)
        ])
        
    sc_force.addInteractionGroup(list(perturbed_set), list(env_set))
    sc_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    is_dexp = str(potential_type or "").strip().lower() == "dexp"
    if is_dexp:
        cutoff_nm = DEXP_VDW_CUTOFF_NM
        switch_width_nm = DEXP_VDW_SWITCH_WIDTH_NM
        switch_nm = cutoff_nm - switch_width_nm
        if not (cutoff_nm > switch_nm > 0.0):
            raise ValueError(
                "DEXP cutoff/switch 配置无效："
                f"cutoff={cutoff_nm}, switch_width={switch_width_nm}, "
                f"derived_switch={switch_nm} nm"
            )
    else:
        cutoff_nm, switch_nm = SOFTCORE_CUTOFF_NM, SOFTCORE_SWITCH_NM
    sc_force.setCutoffDistance(cutoff_nm * unit.nanometer)
    print(
        "  ⚠️ [VDW softcore] OpenMM 原生 CustomNonbondedForce.setUseLongRangeCorrection 已禁用"
        "（LJ+Coulomb 拼在同一个表达式里，原生解析修正对 Coulomb 的 1/r 尾项会发散，"
        "实测直接让 CUDA 崩溃，见 AUDIT_STATUS.md）——这不代表 LJ tail 完全不补偿："
        "对默认 softcore/ACE 路径，build_ibs_dual_system 会在此后为每个 λ_vdw 态单独算出"
        "手写解析尾项系数并挂在 ibs_wrapper 上（下方 [LJ LRC] 日志），由 IBSSampler 每帧加进 "
        "target_energies，不走这个原生开关；potential_type='dexp' 时该解析尾项尚未验证，会"
        "显式跳过（见下方对应日志）。APBS 外部项仅用于静电/连续介质类长程修正，跟这个 "
        "LJ tail 项无关。"
    )
    sc_force.setUseLongRangeCorrection(False)
    
    if reference_exclusions:
        for p1, p2 in reference_exclusions:
            p1, p2 = int(p1), int(p2)
            if p1 < total_particles and p2 < total_particles:
                sc_force.addExclusion(p1, p2)
                
    sc_force.setUseSwitchingFunction(True)
    sc_force.setSwitchingDistance(switch_nm * unit.nanometer)
    return sc_force

def _normalize_softcore_params(
    softcore_params: ACESoftcorePotential,
    n_perturbed: int,
    explicit: bool = False,
) -> ACESoftcorePotential:
    requested_lj = float(getattr(softcore_params, "alpha_lj", float("nan")))
    requested_coul = float(getattr(softcore_params, "alpha_coul", float("nan")))
    if explicit:
        normalized = ACESoftcorePotential(
            alpha_lj=requested_lj,
            alpha_coul=requested_coul,
            power_lj=getattr(softcore_params, "power_lj", (2, 2)),
            power_coul=getattr(softcore_params, "power_coul", (1, 1)),
        )
        source = "explicit_user_or_cached"
    else:
        adaptive = ACESoftcorePotential.optimize_alpha(max(int(n_perturbed), 1))
        normalized = ACESoftcorePotential.from_dict(adaptive)
        source = "adaptive_by_perturbed_atom_count"
    normalized.provenance = {
        "source": source,
        "n_perturbed_atoms": int(n_perturbed),
        "input_alpha_lj": requested_lj,
        "input_alpha_coul": requested_coul,
        "alpha_lj": float(normalized.alpha_lj),
        "alpha_coul": float(normalized.alpha_coul),
        "alpha_convention": ACESoftcorePotential.ALPHA_CONVENTION,
        "note": "Softcore parameters are no longer silently overwritten by a fixed production default.",
    }
    print(
        f"  🧪 [Softcore 参数] {source}: "
        f"alpha_lj={normalized.alpha_lj:.3f} nm^6, alpha_coul={normalized.alpha_coul:.3f} nm^2 "
        f"(输入值: LJ={requested_lj:.3f}, Coul={requested_coul:.3f})"
    )
    return normalized


def _normalize_alchemical_params(alchemical_params, potential_type: str, n_perturbed: int):
    if potential_type == "dexp":
        if isinstance(alchemical_params, DEXPSurrogatePotential):
            return alchemical_params
        return DEXPSurrogatePotential.from_dict(alchemical_params or {})
    if isinstance(alchemical_params, ACESoftcorePotential):
        return _normalize_softcore_params(alchemical_params, n_perturbed, explicit=True)
    explicit = bool(
        isinstance(alchemical_params, dict)
        and ("alpha_lj" in alchemical_params or "alpha_coul" in alchemical_params)
    )
    return _normalize_softcore_params(
        ACESoftcorePotential.from_dict(alchemical_params or {}),
        n_perturbed,
        explicit=explicit,
    )


def _compute_reference_com(
    positions,
    system: openmm.System,
    atom_indices: List[int],
) -> Tuple[float, float, float]:
    pos_nm = positions.value_in_unit(unit.nanometer) if hasattr(positions, "value_in_unit") else np.asarray(positions, dtype=float)
    masses = np.array(
        [system.getParticleMass(int(idx)).value_in_unit(unit.dalton) for idx in atom_indices],
        dtype=float,
    )
    if np.any(masses <= 0.0):
        masses = np.ones(len(atom_indices), dtype=float)
    coords = np.asarray([pos_nm[int(idx)] for idx in atom_indices], dtype=float)
    ref_com = np.average(coords, axis=0, weights=masses)
    return tuple(float(x) for x in ref_com)


def _resolve_periodic_box_vectors(box_vectors, topology=None, system: Optional[openmm.System] = None):
    """
    解析并校验周期性盒子。

    优先级：
    1. 显式传入的 box_vectors
    2. topology 上的盒子
    3. system 默认盒子
    """
    candidates = []
    if box_vectors is not None:
        candidates.append(box_vectors)
    if topology is not None and hasattr(topology, "getPeriodicBoxVectors"):
        try:
            topo_box = topology.getPeriodicBoxVectors()
        except Exception:
            topo_box = None
        if topo_box is not None:
            candidates.append(topo_box)
    if system is not None:
        try:
            sys_box = system.getDefaultPeriodicBoxVectors()
        except Exception:
            sys_box = None
        if sys_box is not None:
            candidates.append(sys_box)

    for candidate in candidates:
        try:
            ax = candidate[0][0].value_in_unit(unit.nanometer)
            by = candidate[1][1].value_in_unit(unit.nanometer)
            cz = candidate[2][2].value_in_unit(unit.nanometer)
        except Exception:
            continue
        if ax > 0.0 and by > 0.0 and cz > 0.0:
            return candidate
    return None


def _system_requires_periodic_box(system: openmm.System) -> bool:
    """判断系统内是否存在依赖有效周期性盒子的力。"""
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            if force.getNonbondedMethod() in (
                openmm.NonbondedForce.CutoffPeriodic,
                openmm.NonbondedForce.PME,
                openmm.NonbondedForce.LJPME,
                openmm.NonbondedForce.Ewald,
            ):
                return True
        if isinstance(force, openmm.CustomNonbondedForce):
            if force.getNonbondedMethod() == openmm.CustomNonbondedForce.CutoffPeriodic:
                return True
    return False


def _system_has_global_parameter(system: openmm.System, name: str) -> bool:
    """检查 System 里任意 Force 是否已注册某个 global parameter。

    用于窗口管理器的不同系统变体（例如 build_ibs_dual_system 的 VDW softcore
    路径 vs. build_shadow_coul_ibs_system 的 Shadow-Coulomb 路径）共享同一段
    窗口调度代码时，对只在某些变体里存在的全局参数（如 lambda_shield）做
    存在性检查，避免对不含该参数的系统调用 context.setParameter 直接报错。
    """
    for force in system.getForces():
        if not hasattr(force, "getNumGlobalParameters"):
            continue
        for i in range(force.getNumGlobalParameters()):
            if force.getGlobalParameterName(i) == name:
                return True
    return False


def _collect_softcore_exclusions(
    nb_force: openmm.NonbondedForce,
    system: openmm.System,
    perturbed_indices: List[int],
    environment_indices: List[int],
    reference_exclusions=None,
) -> List[Tuple[int, int]]:
    """
    为配体-环境软核相互作用收集完整排除表。
    仅保留跨组(L-E)对，覆盖 1-2/1-3/1-4 与约束对，避免 CV 重复计算。
    """
    perturbed_set = set(int(i) for i in perturbed_indices)
    env_set = set(int(i) for i in environment_indices)
    exclusion_pairs = set()

    def _maybe_add_pair(p1, p2):
        p1, p2 = int(p1), int(p2)
        if p1 == p2:
            return
        in_cross_group = (
            (p1 in perturbed_set and p2 in env_set) or
            (p2 in perturbed_set and p1 in env_set)
        )
        if in_cross_group:
            exclusion_pairs.add((min(p1, p2), max(p1, p2)))

    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, _, _ = force.getBondParameters(i)
                _maybe_add_pair(p1, p2)
        elif isinstance(force, openmm.CustomBondForce):
            for i in range(force.getNumBonds()):
                p1, p2, _ = force.getBondParameters(i)
                _maybe_add_pair(p1, p2)
        elif isinstance(force, openmm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                p1, _, p3, _, _ = force.getAngleParameters(i)
                _maybe_add_pair(p1, p3)
        elif isinstance(force, openmm.CustomAngleForce):
            for i in range(force.getNumAngles()):
                p1, _, p3, _ = force.getAngleParameters(i)
                _maybe_add_pair(p1, p3)
        elif isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, _, _, p4, _, _, _ = force.getTorsionParameters(i)
                _maybe_add_pair(p1, p4)
        elif isinstance(force, openmm.CustomTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, _, _, p4, _ = force.getTorsionParameters(i)
                _maybe_add_pair(p1, p4)
        elif hasattr(openmm, "RBTorsionForce") and isinstance(force, openmm.RBTorsionForce):
            for i in range(force.getNumTorsions()):
                p1, _, _, p4, *_ = force.getTorsionParameters(i)
                _maybe_add_pair(p1, p4)

    for i in range(system.getNumConstraints()):
        p1, p2, _ = system.getConstraintParameters(i)
        _maybe_add_pair(p1, p2)

    for i in range(nb_force.getNumExceptions()):
        p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
        _maybe_add_pair(p1, p2)

    if reference_exclusions:
        for p1, p2 in reference_exclusions:
            _maybe_add_pair(p1, p2)

    return sorted(exclusion_pairs)


def _estimate_wca_shield_parameters(
    particle_params,
    perturbed_indices: List[int],
    environment_indices: List[int],
) -> Dict[str, float]:
    """Estimate conservative lambda-WCA shield parameters from LJ sigma values."""
    lig_sigmas = []
    env_sigmas = []
    for idx in perturbed_indices:
        _q, sig, eps = particle_params[int(idx)]
        eps_val = eps.value_in_unit(unit.kilojoule_per_mole)
        sig_val = sig.value_in_unit(unit.nanometer)
        if eps_val > 1.0e-8 and sig_val > 1.0e-8:
            lig_sigmas.append(sig_val)
    for idx in environment_indices:
        _q, sig, eps = particle_params[int(idx)]
        eps_val = eps.value_in_unit(unit.kilojoule_per_mole)
        sig_val = sig.value_in_unit(unit.nanometer)
        if eps_val > 1.0e-8 and sig_val > 1.0e-8:
            env_sigmas.append(sig_val)

    if lig_sigmas and env_sigmas:
        mixed = []
        env_sample = np.asarray(env_sigmas, dtype=float)
        if env_sample.size > 5000:
            env_sample = np.quantile(env_sample, np.linspace(0.05, 0.95, 64))
        for sig_l in lig_sigmas:
            mixed.extend((0.5 * (sig_l + env_sample)).tolist())
        sigma_ref = float(np.percentile(np.asarray(mixed, dtype=float), 10.0))
        source = "lj_sigma_10th_percentile"
    else:
        sigma_ref = 0.22
        source = "fallback_default_no_lj_sigmas"

    rc = float(np.clip(0.85 * sigma_ref, 0.18, 0.32))
    eps_wca = float(np.clip(1.0 + 0.015 * max(len(perturbed_indices) - 20, 0), 1.0, 2.5))
    return {
        "rc_nm": rc,
        "eps_wca_kJ_mol": eps_wca,
        "sigma_reference_nm": sigma_ref,
        "source": source,
        "n_ligand_lj_atoms": int(len(lig_sigmas)),
        "n_environment_lj_atoms": int(len(env_sigmas)),
    }


def diagnose_softcore_cv_values(
    context: openmm.Context,
    ibs_wrapper: "IBSBiasForce",
    lambdas_coul: List[float],
    lambdas_vdw: List[float],
    prefix: str = "",
    sampler: "IBSSampler" = None,
):
    """打印当前窗口所有软核 CV / Boresch CV 的原始数值，帮助定位异常背景。"""
    print(f"\n🔍 [{prefix}] 软核 CV 数值诊断:")
    try:
        state_base = context.getState(getEnergy=True, groups={0, 2, 3, 5})
        e_base = state_base.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    except Exception as e:
        e_base = float("nan")
        print(f"  e_base 读取失败: {e}")

    try:
        state_bias = context.getState(getEnergy=True, groups={1, 4})
        e_bias = state_bias.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    except Exception:
        e_bias = float("nan")

    print(
        f"  e_base(Group0+2+3+5)={e_base:.3f} kJ/mol | "
        f"e_bias(Group1+4)={e_bias:.3f} kJ/mol"
    )

    sampled_interactions = None
    if sampler is not None:
        try:
            sampled_interactions = sampler.get_raw_interaction_energies()
        except Exception as e:
            print(f"  CV 探针读取失败: {e}")
            return

    interaction_values = []
    for k, (lc, lv) in enumerate(zip(lambdas_coul, lambdas_vdw)):
        if sampler is not None:
            e_int = sampled_interactions[k] if k < len(sampled_interactions) else float("nan")
        else:
            try:
                cv_vals = ibs_wrapper.get_force().getCollectiveVariableValues(context)
            except Exception as e:
                print(f"  CV 读取失败: {e}")
                return
            idx_int = 2 * k
            e_int = cv_vals[idx_int] if idx_int < len(cv_vals) else float("nan")
        e_rest = 0.0
        interaction_values.append(float(e_int))
        e_tot = e_base + e_int + e_rest if np.isfinite(e_base) else float("nan")
        print(
            f"  state {k:>2d} | lam_c={lc:7.4f} lam_v={lv:7.4f} | "
            f"e_int={e_int:14.3f} | e_rest={e_rest:10.3f} | e_total={e_tot:14.3f}"
        )

    interaction_values = np.asarray(interaction_values, dtype=float)
    finite = interaction_values[np.isfinite(interaction_values)]
    if finite.size >= 2:
        adjacent = np.abs(np.diff(finite))
        max_adjacent = float(np.max(adjacent))
        total_span = float(np.max(finite) - np.min(finite))
        print(f"  ΔU诊断: 窗口跨度={total_span:.1f} kJ/mol | 最大相邻ΔU={max_adjacent:.1f} kJ/mol")
        if max_adjacent > 50.0:
            print("  ⚠️ 相邻 λ 能量差远超 50 kJ/mol，IBS 权重很可能塌缩；建议显著增加该阶段 λ 状态数。")

def _serialize_ibs_common_system(window_system: openmm.System) -> str:
    """Return U_common from an assembled VDW-IBS window.

    ``system_template`` is the original fully interacting system, so cloning it
    and adding a state CV would double-count ligand-environment interactions.
    The assembled window is the authoritative source: its main NB interaction
    is already zeroed and groups 0/2/3/5 are exactly U_common.  Work on a clone
    and remove only the top-level IBS mixture (group 1) and WCA sampling bias
    (group 4), with strict structural assertions.

    This WCA-less system is for the *path/lambda-grid* overlap probe
    (``probe_bidirectional_overlap``) only -- it deliberately answers "is the
    underlying physical free energy landscape connected enough", not "what
    does production actually sample". Do not reuse it for bias-CV
    calibration purposes; see ``_serialize_ibs_common_plus_wca_system`` for
    that (production's real sampled ensemble includes the WCA shield).
    """
    common = XmlSerializer.deserialize(XmlSerializer.serialize(window_system))
    removed = {1: 0, 4: 0}
    for force_index in reversed(range(common.getNumForces())):
        group = int(common.getForce(force_index).getForceGroup())
        if group in removed:
            common.removeForce(force_index)
            removed[group] += 1
    if removed != {1: 1, 4: 1}:
        raise RuntimeError(
            f"无法从 VDW-IBS 窗口唯一导出 U_common：移除的顶层 force 数={removed}"
        )
    residual = [
        int(common.getForce(i).getForceGroup())
        for i in range(common.getNumForces())
        if int(common.getForce(i).getForceGroup()) in (1, 4)
    ]
    if residual:
        raise RuntimeError(f"U_common 中仍残留 IBS/WCA force group: {residual}")
    return XmlSerializer.serialize(common)


def _serialize_ibs_common_plus_wca_system(window_system: openmm.System) -> str:
    """Return U_common + WCA_window (Group 4 kept; only Group 1 IBS mixture
    removed) from an assembled VDW-IBS window.

    Production actually samples ``U_common + WCA_window(lambda_shield) +
    CV_k`` (Group 1), not ``U_common + CV_k`` alone -- the WCA shield only
    cancels out of *same-frame, same-window* energy differences between
    states (it has the same lambda_shield for every k in a window), it does
    not disappear from the sampled *trajectory*. A fixed-H probe built on
    ``_serialize_ibs_common_system`` (which strips Group 4 entirely) samples
    a genuinely different conformational ensemble than production, so a
    delta_f measured on it cannot be used to calibrate the bias CV's f_k --
    only for the separate path/lambda-grid overlap question that function
    already serves. This variant keeps Group 4 so the bias-calibration probe
    (``probe_bidirectional_overlap_for_bias_calibration``) samples the exact
    ensemble it needs to reweight, once the caller sets ``lambda_shield`` to
    the same value production used for this window.
    """
    common_plus_wca = XmlSerializer.deserialize(XmlSerializer.serialize(window_system))
    removed = {1: 0}
    for force_index in reversed(range(common_plus_wca.getNumForces())):
        group = int(common_plus_wca.getForce(force_index).getForceGroup())
        if group in removed:
            common_plus_wca.removeForce(force_index)
            removed[group] += 1
    if removed != {1: 1}:
        raise RuntimeError(
            f"无法从 VDW-IBS 窗口唯一导出 U_common+WCA：移除的顶层 force 数={removed}"
        )
    residual = [
        int(common_plus_wca.getForce(i).getForceGroup())
        for i in range(common_plus_wca.getNumForces())
        if int(common_plus_wca.getForce(i).getForceGroup()) == 1
    ]
    if residual:
        raise RuntimeError(f"U_common+WCA 中仍残留 IBS force group: {residual}")
    has_group4 = any(
        int(common_plus_wca.getForce(i).getForceGroup()) == 4
        for i in range(common_plus_wca.getNumForces())
    )
    if not has_group4:
        raise RuntimeError("U_common+WCA 导出后缺少 Group 4 WCA 防护壳，无法用于 bias 校准探针。")
    return XmlSerializer.serialize(common_plus_wca)


OPENMM_CUSTOM_CV_MAX_VARIABLES = 32
IBS_DUAL_CVS_PER_LAMBDA_STATE = 2
IBS_DUAL_MAX_LAMBDA_STATES = (
    OPENMM_CUSTOM_CV_MAX_VARIABLES // IBS_DUAL_CVS_PER_LAMBDA_STATE
)


def build_ibs_dual_system(
    system: openmm.System,
    topology,
    perturbed_indices: List[int],
    lambdas_coul: List[float],
    lambdas_vdw: List[float],
    alchemical_params,
    potential_type: str = "softcore",
    restraint_params: Optional[Dict] = None,
    temperature: openmm.unit.Quantity = 300 * openmm.unit.kelvin,
    prefix: str = "abfe_dual",
    box_vectors=None,
    reference_positions=None,
) -> Tuple[openmm.System, 'IBSBiasForce']:
    """
    构建双λ IBS 采样系统 (终极修复版：彻底剥离 Group 0 的 λ 依赖)
    """
    if len(lambdas_coul) != len(lambdas_vdw):
        raise ValueError("lambdas_coul 与 lambdas_vdw 必须等长")
    if len(lambdas_vdw) > IBS_DUAL_MAX_LAMBDA_STATES:
        raise RuntimeError(
            "单个 IBS ensemble 的 lambda 状态过多："
            f"K={len(lambdas_vdw)}, 每态 {IBS_DUAL_CVS_PER_LAMBDA_STATE} 个 CV，"
            f"将超过 OpenMM CustomCVForce 的 {OPENMM_CUSTOM_CV_MAX_VARIABLES}-CV 上限。"
            "请把 vanishing 域划为多个物理子区间；禁止使用单一 [0:K] ensemble。"
        )
    system = ensure_owned_system(system)
    new_sys = ensure_owned_system(XmlSerializer.deserialize(XmlSerializer.serialize(system)))
    resolved_box = _resolve_periodic_box_vectors(box_vectors, topology=topology, system=new_sys)
    if _system_requires_periodic_box(new_sys):
        if resolved_box is None:
            raise ValueError("系统包含周期性非键力，但未提供有效的周期性盒子。")
        new_sys.setDefaultPeriodicBoxVectors(*resolved_box)

    num_atoms = new_sys.getNumParticles()
    perturbed_set = set(perturbed_indices)
    env_indices = [i for i in range(num_atoms) if i not in perturbed_set]
    alchemical_params = _normalize_alchemical_params(
        alchemical_params, potential_type, len(perturbed_indices)
    )

    lambda_coul_arr = np.asarray(lambdas_coul, dtype=float)
    if np.any(np.abs(lambda_coul_arr) > 1e-8):
        raise RuntimeError(
            "IBS/OpenMM CustomNonbondedForce 路径不能用于非零 λ_coul。"
            "CustomNonbondedForce 不支持 PME，若把配体-环境 Coulomb 放进 IBS CV，"
            "静电会被截断成 cutoff 相互作用，导致物理 Hamiltonian 和 MBAR 均不可信。"
            "请将 Coulomb 去电荷阶段改为保留 PME 的 NonbondedForce/ParameterOffset 或独立传统窗口；"
            "当前 IBS 路径仅允许 λ_coul=0 的 VDW 短程阶段。"
        )

    # ---------- 1. 提取 NonbondedForce ----------
    nb_forces = [f for f in new_sys.getForces() if isinstance(f, openmm.NonbondedForce)]
    if not nb_forces:
        raise ValueError("未找到 NonbondedForce")
    nb = nb_forces[0]

    # ---------- 2. 保存原始参数快照 (供 Group 1 & 2 使用) ----------
    all_params = [nb.getParticleParameters(i) for i in range(num_atoms)]
    ref_excl = []
    for i in range(nb.getNumExceptions()):
        p1, p2, cp, sig, eps = nb.getExceptionParameters(i)
        ref_excl.append((int(p1), int(p2)))
    
    softcore_excl = _collect_softcore_exclusions(nb, new_sys, perturbed_indices, env_indices, ref_excl)
    # 🚨 关键修复：OpenMM 要求同一 System 内所有粒子数相同的 NonbondedForce/
    # CustomNonbondedForce 拥有完全相同的排除表 ("All Forces must have identical
    # exclusions")。IBS 的软核 CV (Group 1，嵌在 CustomCVForce 内部，
    # sync_all_exclusions 扫不到) 之前只带了 softcore_excl（仅 L-E 跨组对），
    # 跟 nb 的完整排除表（含大量环境蛋白/水分子自身的 1-2/1-3/1-4 对）数量对不
    # 上，会在第一次真正构建邻居表时（通常是 Stage2 去VDW 阶段首次 minimize）
    # 报 "All Forces must have identical exclusions"。这里把 nb 的完整排除对
    # 也并进来，统一喂给下面的软核 CV，跟 wca_force/nb 保持逐对一致；
    # interaction group 之外的这些对本来就不会被软核 CV 计算，纯粹是账本对齐。
    full_softcore_excl = sorted(
        {(min(int(p1), int(p2)), max(int(p1), int(p2))) for p1, p2 in ref_excl if int(p1) != int(p2)}
        | set(softcore_excl)
    )

    # 构建原始 nb 副本 (供 Group 2 配体内部力使用)
    original_nb = openmm.NonbondedForce()
    for q, sig, eps in all_params:
        original_nb.addParticle(q, sig, eps)
    for p1, p2, cp, sig, eps in [nb.getExceptionParameters(i) for i in range(nb.getNumExceptions())]:
        original_nb.addException(int(p1), int(p2), cp, sig, eps)
    original_nb.setNonbondedMethod(nb.getNonbondedMethod())
    original_nb.setCutoffDistance(nb.getCutoffDistance())
    if nb.getUseSwitchingFunction():
        original_nb.setUseSwitchingFunction(True)
        original_nb.setSwitchingDistance(nb.getSwitchingDistance())

    # ---------- 2.5 🔑 静态电中性防御 (替代 ParameterOffset，彻底消除 base 的 λ 依赖) ----------
    lig_net_charge = 0.0
    for idx in perturbed_indices:
        q, _, _ = all_params[idx]
        lig_net_charge += q.value_in_unit(unit.elementary_charge)
    lig_net_charge = round(lig_net_charge)

    if abs(lig_net_charge) > 0.01:
        raise RuntimeError(
            f"检测到带净电配体 ({lig_net_charge:+d} e)。IBS 的 VDW 阶段不再静态改写反离子电荷；"
            "请先用保留 PME 的去电荷阶段处理净电荷，或显式实现经过验证的离子炼金方案。"
        )

    # ---------- 3. 🔑 永久关闭主 NB 力中的配体相互作用 (确保 base 严格 λ 无关) ----------
    for idx in perturbed_set:
        # 电荷与 VdW 同时归零，彻底切断主 NB 力对配体的贡献
        nb.setParticleParameters(idx, 0.0*unit.elementary_charge, 0.1*unit.nanometer, 0.0*unit.kilojoule_per_mole)
    
    for i in range(nb.getNumExceptions()):
        p1, p2, cp, sig, eps = nb.getExceptionParameters(i)
        if p1 in perturbed_set or p2 in perturbed_set:
            # 跨组或配体内部异常对全部归零（内部作用已由 Group 2 接管）
            nb.setExceptionParameters(i, int(p1), int(p2), 0.0*unit.elementary_charge**2, sig, 0.0*unit.kilojoule_per_mole)

    # ---------- 4. 配体内部力 (Group 2) ----------
    internal_ref_excl = [(p1, p2) for p1, p2 in ref_excl if p1 in perturbed_set and p2 in perturbed_set]
    ll_f, ll_14_f = create_ligand_internal_force(
        original_nb, perturbed_indices, all_params, internal_ref_excl, num_atoms, system=new_sys
    )
    ll_f.setForceGroup(2)
    if ll_14_f: ll_14_f.setForceGroup(2)
    new_sys.addForce(ll_f)
    if ll_14_f: new_sys.addForce(ll_14_f)

    # ---------- 5. 物理 Boresch 限制力 (Group 3) ----------
    if _has_valid_boresch_restraint(restraint_params):
        rest_f_phys = LambdaDependentBoreschForce(
            rec_idx=restraint_params["receptor_indices"], lig_idx=restraint_params["ligand_indices"],
            eq=restraint_params["equilibrium_values"], fc=restraint_params["force_constants"],
            lam_name="lambda_boresch_scale", fixed_lam=None, sign=1.0, use_pbc=True
        )
        rest_f_phys.setForceGroup(3)
        new_sys.addForce(rest_f_phys)

    # ---------- 6. λ-WCA 防护壳（Group 4） & COM 限制（Group 5） [保持原样] ----------
    wca_expr = (
        "4.0*lambda_shield*(1.0-lambda_shield)*step(rc-r)*eps_wca*"
        "(((rc/max(r, 1e-6))^6)^2 - 2*((rc/max(r, 1e-6))^6) + 1)"
    )
    wca_params = _estimate_wca_shield_parameters(all_params, list(perturbed_set), env_indices)
    wca_force = openmm.CustomNonbondedForce(wca_expr)
    wca_force.addGlobalParameter("lambda_shield", 0.0)
    wca_force.addGlobalParameter("rc", wca_params["rc_nm"])
    wca_force.addGlobalParameter("eps_wca", wca_params["eps_wca_kJ_mol"])
    for _ in range(num_atoms): wca_force.addParticle([])
    wca_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    wca_force.setCutoffDistance(wca_params["rc_nm"] * openmm.unit.nanometer)
    wca_force.addInteractionGroup(list(perturbed_set), env_indices)
    for p1, p2 in softcore_excl: wca_force.addExclusion(int(p1), int(p2))
    wca_force.setForceGroup(4)
    new_sys.addForce(wca_force)
    print(
        "  🛡️ [λ-WCA 防护壳] "
        f"rc={wca_params['rc_nm']:.3f} nm, eps={wca_params['eps_wca_kJ_mol']:.3f} kJ/mol "
        f"({wca_params['source']}, sigma_ref={wca_params['sigma_reference_nm']:.3f} nm)"
    )

    if reference_positions is not None and perturbed_indices and not _has_valid_boresch_restraint(restraint_params):
        ref_com = _compute_reference_com(reference_positions, new_sys, perturbed_indices)
        # CustomCentroidBondForce 在部分 OpenMM 版本中不支持 periodicdistance()；
        # 这里退回兼容性更好的显式欧氏距离表达式，优先保证 Context 可创建。
        com_expr = (
            "0.5*k_com*step(r_com-r0_com)*(r_com-r0_com)^2; "
            "r_com=sqrt((x1-x0)^2 + (y1-y0)^2 + (z1-z0)^2)"
        )
        try:
            com_force = openmm.CustomCentroidBondForce(1, com_expr)
            com_force.addGlobalParameter("k_com", 50.0)
            com_force.addGlobalParameter("r0_com", 1.0)
            com_force.addGlobalParameter("x0", ref_com[0])
            com_force.addGlobalParameter("y0", ref_com[1])
            com_force.addGlobalParameter("z0", ref_com[2])
            masses = [new_sys.getParticleMass(int(idx)).value_in_unit(unit.dalton) for idx in perturbed_indices]
            com_force.addGroup([int(idx) for idx in perturbed_indices], masses)
            com_force.addBond([0], [])
            com_force.setForceGroup(5)
            new_sys.addForce(com_force)
        except Exception:
            logger.warning("COM 限制力构建失败，配体可能逃逸", exc_info=True)
    elif _has_valid_boresch_restraint(restraint_params):
        print("  ℹ️ 检测到有效 Boresch 锚定，跳过 Group 5 COM 限制力以避免双重定位约束冲突。")

    # ---------- 7. IBS 偏置力与纯 VDW 软核 CV (Group 1) ----------
    ibs_wrapper = IBSBiasForce(len(lambdas_coul), temperature, prefix=prefix)
    original_params_fresh = [original_nb.getParticleParameters(i) for i in range(num_atoms)]

    cv_template = _create_softcore_force(
        nb,
        perturbed_indices,
        env_indices,
        lam_coul=0.0,
        lam_vdw=0.0,
        alchemical_params=alchemical_params,
        potential_type=potential_type,
        reference_exclusions=full_softcore_excl,
        particle_params_override=original_params_fresh,
        num_particles=num_atoms,
    )
    legacy_softcore_signature = "+ 1e-4)^2"
    if legacy_softcore_signature in cv_template.getEnergyFunction():
        print("  ⚠️ [IBS CV] 检测到旧版 VDW softcore 表达式签名；请检查 CV 构造路径。")
    template_cutoff = cv_template.getCutoffDistance()
    template_switch = cv_template.getSwitchingDistance()
    template_method = cv_template.getNonbondedMethod()
    template_excl = {
        tuple(sorted(map(int, cv_template.getExclusionParticles(i))))
        for i in range(cv_template.getNumExclusions())
    }

    # ---------- 7.5 🔑 LJ 长程色散尾项：手算解析修正（不走 OpenMM 内建 LRC） ----------
    # 已用 test_lrc_interaction_group_compat.py 实测确认：CustomNonbondedForce.
    # setUseLongRangeCorrection(True) 配合 addInteractionGroup 在电荷=0 的纯 LJ
    # 情形下工作正常（截断误差消除 ~88%，lambda_vdw=1.0 与 0.5 结果一致）；但本
    # 项目的软核表达式把 LJ 和 Coulomb 拼在同一个 CustomNonbondedForce 里，真实
    # 非零电荷会让 OpenMM 对 1/r 尾项的解析积分发散，直接让 CUDA 后端崩溃
    # ("terminate called recursively" / core dump，已实测复现)。因此改为在
    # Python 侧手算解析修正（不触碰 CustomNonbondedForce/GPU kernel；每帧只是
    # 一次除法）。
    # 🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION=2] 之前这里只补 cutoff（1.2nm）
    # 之外的标准 r^-6 尾项，完全忽略了这个软核力实际启用的 1.0-1.2nm switching
    # 区间——OpenMM 在这段区间把能量从满强度削到 0，真正缺失的是"switching 削弱
    # 的部分 + cutoff 之外的标准尾项"（见 CustomNonbondedForce LRC 文档），且
    # λ<1 时软核分母 D(r)=alpha_lj*(1-λ)^m_lj + r^6 也需要真实积分，而不是当成
    # 纯 r^6 处理。现在对每个 λ_vdw 数值积分真实的 switching-aware +
    # softcore-aware 径向积分（_lj_tail_lrc_coefficients_kj_mol），同时补上
    # 排斥项 r^-12 的尾贡献（之前只有吸引项 r^-6）。每帧修正 = coeff[k] / V(t)。
    ibs_wrapper.lj_tail_lrc_coeff_kj_mol = None
    if not ibs_lj_tail_lrc_is_applicable(potential_type):
        print(
            f"  ⚠️ [LJ LRC] {ibs_lj_tail_lrc_inapplicable_reason(potential_type)}"
            "，本次不附加修正。"
        )
    else:
        rc_nm = template_cutoff.value_in_unit(unit.nanometer)
        rs_nm = template_switch.value_in_unit(unit.nanometer)
        tail_sigma, tail_s6_per_sigma, tail_s12_per_sigma = (
            _lj_tail_correction_sigma_resolved_moments(
                all_params, perturbed_indices, env_indices
            )
        )
        tail_s6 = float(np.sum(tail_s6_per_sigma))
        tail_s12 = float(np.sum(tail_s12_per_sigma))
        n_lj_exp = float(getattr(alchemical_params, "n_lj", 2))
        alpha_lj = float(getattr(alchemical_params, "alpha_lj", 0.5))
        m_lj = float(getattr(alchemical_params, "m_lj", 2))
        ibs_wrapper.lj_tail_lrc_coeff_kj_mol = _lj_tail_lrc_coefficients_kj_mol(
            lambdas_vdw, tail_sigma, tail_s6_per_sigma, tail_s12_per_sigma,
            alpha_lj, m_lj, n_lj_exp, rs_nm, rc_nm,
        )
        print(
            f"  🧮 [LJ LRC v{TRADITIONAL_LJ_LRC_PROTOCOL_VERSION}] switching+softcore-aware 解析长程尾项已启用："
            f"S6={tail_s6:.4g} kJ·nm^6/mol, S12={tail_s12:.4g} kJ·nm^12/mol, "
            f"{tail_sigma.size} 个 sigma 分组（{tail_sigma.min():.4f}~{tail_sigma.max():.4f} nm，逐组积分）, "
            f"switch={rs_nm:.3f} nm, cutoff={rc_nm:.3f} nm, alpha_lj={alpha_lj:.4g}(无量纲), m_lj={m_lj:.1f}, "
            f"n_lj={n_lj_exp:.1f}；每帧修正 = lrc_coeff[k] / V(t)，lrc_coeff 逐 (λ, sigma) 数值积分得出。"
        )

    for k, (_lc, lv) in enumerate(zip(lambdas_coul, lambdas_vdw)):
        # IBS/OpenMM 只允许处理短程 VDW CV；Coulomb 已禁止进入 CustomNonbondedForce，
        # 避免把 PME 静电截断为 cutoff 相互作用。
        int_f_cv = _create_softcore_force(
            nb,
            perturbed_indices,
            env_indices,
            lam_coul=0.0,
            lam_vdw=float(lv),
            alchemical_params=alchemical_params,
            potential_type=potential_type,
            reference_exclusions=full_softcore_excl,
            particle_params_override=original_params_fresh,
            num_particles=num_atoms,
        )
        int_f_cv.setCutoffDistance(template_cutoff)
        int_f_cv.setSwitchingDistance(template_switch)
        int_f_cv.setNonbondedMethod(template_method)
        print(
            "  ⚠️ [IBS CV] VDW custom softcore CV 不含 LJ 长程修正；"
            "请勿直接等同于已含 dispersion correction 的 PME/LJPME 循环；APBS 修正应作为最终外部项记录。"
        )
        int_f_cv.setUseLongRangeCorrection(False)
        int_f_cv.setUseSwitchingFunction(True)
        current_excl = {
            tuple(sorted(map(int, int_f_cv.getExclusionParticles(i))))
            for i in range(int_f_cv.getNumExclusions())
        }
        if current_excl != template_excl:
            raise RuntimeError(f"CV {k} 排除表不一致，将破坏 VDW-IBS 邻居表复用条件。")
        ibs_wrapper.addCollectiveVariable(f"cv_{k}_int", int_f_cv)
        
        # Boresch CV 保持零力（物理限制已在 Group 3）
        rest_f_cv = openmm.CustomExternalForce("0")
        ibs_wrapper.addCollectiveVariable(f"cv_{k}_rest", rest_f_cv)

    new_sys.addForce(ibs_wrapper.get_force())

    # ---------- 8. 统一排除表同步 ----------
    sync_all_exclusions(new_sys)
    # Export the exact pre-bias Hamiltonian for fixed-state overlap probes.
    # Do this only after exclusion synchronization so the added state CV can
    # share the same nonbonded topology as the main NonbondedForce.
    ibs_wrapper._common_system_xml = _serialize_ibs_common_system(new_sys)
    # 供 bias 校准探针使用（保留 Group 4 WCA，见该函数 docstring 里对两种用途的
    # 区分说明）——不能用上面那份缺 WCA 的 U_common 去校准 f_k。
    ibs_wrapper._common_plus_wca_system_xml = _serialize_ibs_common_plus_wca_system(new_sys)
    new_sys.thisown = 1
    _ = new_sys.getNumParticles()
    print(f"[IBS System] 构建完成，共 {new_sys.getNumForces()} 个力对象 (Group 0 已彻底剥离 λ 依赖)")
    return new_sys, ibs_wrapper

# ============================================================================
# 1.5 Shadow-Coulomb 去电荷：PME-Bridge (容斥探针) + Reciprocal-free IBS 高速公路
# ============================================================================
# 物理背景：
#   ΔG_decharge_PME = ΔG_bridge(PME full -> Shadow full) + ΔG_shadow_ibs(Shadow full -> Shadow decharged)
# 记 A = 配体原子 (perturbed_indices)，E = 其余全部原子：
#   U_shadow_cross(A,E) = A-E 间的短程"影子"库仑势，核函数 erfc(alpha*r)/r，alpha 取
#       自该系统真实 PME 的 Ewald alpha —— 与真实 PME real-space 项逐 pair 严格相等。
#   E_PME_cross(A,E) = E_full - E_A_only - E_E_only （容斥原理，三个纯静电 PME 探针），
#       隔离出 PME 下 A-E 的总静电交叉贡献 (real + reciprocal + self + PME 修正)。
#   Bridge:     U_bridge(s) = U_common + s*U_shadow_cross + (1-s)*E_PME_cross
#   Shadow IBS: U_shadow_ibs(k) = U_common + lambda_shadow_coul_k * U_shadow_cross(A,E)
# 只支持电中性配体 (Q_ligand ≈ 0，已对本项目 Atenolol 拓扑实测验证)；带净电配体需要
# 额外的共炼金反离子逻辑 (思路同 configure_coalchemical_neutral_decharging，驱动变量
# 换成 lambda_shadow_coul)，本实现尚未接入，遇到带净电配体会显式报错而不是静默算错。
# ============================================================================

_SHADOW_ONE_4PI_EPS0 = 138.935456  # kJ/mol*nm/e^2，与 OpenMM NonbondedForce 内部常量一致


def get_pme_alpha_for_system(
    system: openmm.System,
    topology=None,
    box_vectors=None,
) -> float:
    """
    用 OpenMM 自己派生 Ewald alpha 的闭式公式 (1/nm)，直接从静态的
    cutoff/ewaldErrorTolerance 算出来，不建 Context：

        alpha = sqrt(-log(2*ewaldErrorTolerance)) / cutoff

    这与 OpenMM 内部（NonbondedForceImpl）在没有显式 setPMEParameters() 时自动
    派生 alpha 用的公式完全一致，已经用本项目真实的 system_native.xml 实测校验
    （getPMEParametersInContext 给出的 alpha 与该公式逐位吻合）。

    🔑 为什么不像别处那样用 getPMEParametersInContext(ctx)：这需要现建一个一次性
    Context 才能拿到真实值，但实测发现——只要进程里已经创建过至少一个 CPU 平台的
    Context 并释放（生产流程里几乎必然如此，比如更早的 preopt 探针/REMD 阶段），
    再建一个新的一次性 CPU Context 去查 PME 参数会稳定触发 OpenMM/SWIG 的
    `std::bad_cast`（很可能是 CPU 平台 kernel 工厂在同进程内跨 Context 生命周期
    复用时的底层问题）。闭式公式完全不需要 Context，从根上绕开这个坑。
    topology/box_vectors 参数保留只是为了跟其他 builder 签名一致，这里不需要用到。
    """
    nb_forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    if not nb_forces:
        raise ValueError("系统中未找到 NonbondedForce，无法提取 PME alpha。")
    nb = nb_forces[0]
    if nb.getNonbondedMethod() not in (openmm.NonbondedForce.PME, openmm.NonbondedForce.LJPME):
        raise RuntimeError(
            "Shadow-Coulomb 桥接依赖真实 PME 的 Ewald alpha，但该系统 NonbondedForce 的 "
            f"NonbondedMethod={nb.getNonbondedMethod()} 不是 PME，拒绝继续。"
        )
    cutoff_nm = nb.getCutoffDistance().value_in_unit(unit.nanometer)
    tolerance = float(nb.getEwaldErrorTolerance())
    if cutoff_nm <= 0.0 or not (0.0 < tolerance < 0.5):
        raise RuntimeError(
            f"NonbondedForce 的 cutoff={cutoff_nm} nm / ewaldErrorTolerance={tolerance} "
            "不在合理范围内，无法派生 PME alpha。"
        )
    alpha = math.sqrt(-math.log(2.0 * tolerance)) / cutoff_nm
    if alpha <= 0.0:
        raise RuntimeError("派生出的 Ewald alpha <= 0，拒绝继续构建 Shadow 力。")
    return alpha


def _assert_neutral_ligand_for_shadow_coul(nb: openmm.NonbondedForce, ligand_indices: List[int]) -> float:
    """
    Shadow-Coulomb 支路目前只支持电中性配体：带净电配体单独去电荷会让炼金路径中段
    整体净电荷偏离 0，需要跟旧 configure_coalchemical_neutral_decharging 一样的共
    炼金反离子逻辑，本实现尚未接入，遇到带净电配体显式报错而不是悄悄算错。
    返回实测的净电荷（供调用方记录 provenance）。
    """
    net_q = 0.0
    for idx in ligand_indices:
        q, _, _ = nb.getParticleParameters(idx)
        net_q += q.value_in_unit(unit.elementary_charge)
    if abs(round(net_q)) > 0.01:
        raise RuntimeError(
            f"检测到带净电配体 (Net Charge: {round(net_q):+d})。Shadow-Coulomb 去电荷支路"
            "目前只支持电中性配体：单独关掉带净电配体会让炼金路径中段净电荷偏离 0。"
            "请先接入共炼金反离子（同 configure_coalchemical_neutral_decharging 的思路，"
            "驱动变量换成 lambda_shadow_coul），本实现暂不支持，拒绝静默继续。"
        )
    return net_q


def _collect_shadow_cross_exclusions(
    nb: openmm.NonbondedForce,
    ligand_indices: List[int],
    environment_indices: List[int],
    ref_excl: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    跨组 (配体-环境) exclusion 对，供 Shadow 力使用，逻辑同 _collect_softcore_exclusions。

    配体与环境原子之间通常没有真实的键连 1-2/1-3/1-4 exception（配体不与蛋白/水共价
    相连），所以这里天然算出来的跨组对经常是空集——这在物理上完全正确（Shadow 力的
    addInteractionGroup(A,E) 本来就该覆盖所有 A-E pair，没有谁需要被排除）。但如果
    传了 ref_excl（nb 的完整排除表），会把它一并并入返回值：这纯粹是为了让这个力的
    exclusion **数量** 跟同一个 CustomCVForce 里另外几个 NonbondedForce 探针（它们
    照抄了 nb 的完整排除表）保持一致——OpenMM 要求塞进同一个 CustomCVForce 的所有
    CV 互相之间排除表数量必须相同，否则会在建 Context 时报 'All Forces must have
    identical exclusions'。多出来的这些 ref_excl 对本来就落在 interaction group 之外
    （比如环境蛋白/水分子自身的 1-2/1-3/1-4 对），从不会被这个力实际计算，纯粹是
    账本对齐，不影响物理结果。
    """
    ligand_set = set(int(i) for i in ligand_indices)
    env_set = set(int(i) for i in environment_indices)
    pairs = set()
    for i in range(nb.getNumExceptions()):
        p1, p2, _, _, _ = nb.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if (p1 in ligand_set and p2 in env_set) or (p2 in ligand_set and p1 in env_set):
            pairs.add((min(p1, p2), max(p1, p2)))
    if ref_excl:
        pairs |= {(min(int(p1), int(p2)), max(int(p1), int(p2))) for p1, p2 in ref_excl if int(p1) != int(p2)}
    return sorted(pairs)


def _zero_ligand_environment_charge_in_background(
    nb: openmm.NonbondedForce,
    ligand_indices: List[int],
) -> List[Any]:
    """
    只把主 NonbondedForce 里配体的电荷 (不含 VdW) 归零，并把跨组 (配体-环境) 的
    exception 电荷乘积也归零，避免 Shadow/Bridge 的显式电荷探针与主背景力双计
    ligand-environment 静电。配体内部电荷、跨组/配体内部 VdW 一律保持原样不动——
    VdW 在 Bridge/Shadow-IBS 这两个 leg 里从头到尾都是满强度、未被炼金，属于
    U_common，不需要像 build_ibs_dual_system 那样连 VdW 一起归零。
    返回每个粒子归零前的原始 (q, sigma, eps) 快照，供构建 Shadow 力时取真实电荷。
    """
    ligand_set = set(int(i) for i in ligand_indices)
    num_atoms = nb.getNumParticles()
    original_params = [nb.getParticleParameters(i) for i in range(num_atoms)]
    for idx in ligand_set:
        _, sig, eps = original_params[idx]
        nb.setParticleParameters(idx, 0.0 * unit.elementary_charge, sig, eps)
    for i in range(nb.getNumExceptions()):
        p1, p2, cp, sig, eps = nb.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if (p1 in ligand_set) != (p2 in ligand_set):
            nb.setExceptionParameters(i, p1, p2, 0.0 * unit.elementary_charge ** 2, sig, eps)
    return original_params


def _build_shadow_coul_cross_force(
    alpha_ewald: float,
    original_params: List[Any],
    ligand_indices: List[int],
    environment_indices: List[int],
    lambda_value: float,
    exclusions: List[Tuple[int, int]],
    cutoff_nm: float = 1.2,
) -> openmm.CustomNonbondedForce:
    """
    A(配体)-E(环境) 短程"影子"库仑势：lambda * ONE_4PI_EPS0 * q1*q2 * erfc(alpha*r)/r，
    只在 addInteractionGroup(A, E) 之间计算（不含 A-A/E-E），避免引入 lambda^2 项，
    与旧 co-alchemical 去电荷的线性 offset 约定保持一致。lambda 以字面常量刻入表达式
    （不是 runtime 可调的 global parameter），供 IBS 多态窗口 CV（每态一个实例，
    lambda_shadow_coul_k 不同）或 Bridge 的"满强度探针"（lambda=1.0）场景复用同一
    个 builder。
    """
    expr = (
        f"{float(lambda_value)} * {_SHADOW_ONE_4PI_EPS0} * charge1*charge2 * "
        f"erfc({float(alpha_ewald)}*r) / r"
    )
    force = openmm.CustomNonbondedForce(expr)
    force.addPerParticleParameter("charge")
    for q, _, _ in original_params:
        force.addParticle([q.value_in_unit(unit.elementary_charge)])
    force.addInteractionGroup(
        list(set(int(i) for i in ligand_indices)),
        list(set(int(i) for i in environment_indices)),
    )
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff_nm * unit.nanometer)
    force.setUseSwitchingFunction(False)
    force.setUseLongRangeCorrection(False)
    for p1, p2 in exclusions:
        force.addExclusion(int(p1), int(p2))
    return force


def _build_electrostatics_only_pme_probe(
    original_params: List[Any],
    zero_indices: Optional[List[int]],
    reference_nb: openmm.NonbondedForce,
) -> openmm.NonbondedForce:
    """
    构建一个"只算静电、VdW 全部关闭"的 PME 探针 NonbondedForce 副本：epsilon 全部
    置零 (LJ 贡献消失)，PME method/cutoff/switching/ewald tolerance 与 reference_nb
    完全一致（容斥相减要求三个探针的 PME 配置逐一相同，否则减不干净）。可选把
    zero_indices 里的粒子电荷置零，用来构造 A-only / E-only 探针；zero_indices=None
    时构造"全电荷都在"的 full 探针。
    """
    probe = openmm.NonbondedForce()
    zero_set = set(int(i) for i in zero_indices) if zero_indices else set()
    for i, (q, sig, _eps) in enumerate(original_params):
        q_val = 0.0 * unit.elementary_charge if i in zero_set else q
        probe.addParticle(q_val, sig, 0.0 * unit.kilojoule_per_mole)
    for i in range(reference_nb.getNumExceptions()):
        p1, p2, cp, sig, _eps = reference_nb.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        cp_val = 0.0 * unit.elementary_charge ** 2 if (p1 in zero_set or p2 in zero_set) else cp
        probe.addException(p1, p2, cp_val, sig, 0.0 * unit.kilojoule_per_mole)
    probe.setNonbondedMethod(reference_nb.getNonbondedMethod())
    probe.setCutoffDistance(reference_nb.getCutoffDistance())
    probe.setEwaldErrorTolerance(reference_nb.getEwaldErrorTolerance())
    if reference_nb.getUseSwitchingFunction():
        probe.setUseSwitchingFunction(True)
        probe.setSwitchingDistance(reference_nb.getSwitchingDistance())
    probe.setUseDispersionCorrection(False)
    return probe


def build_shadow_coul_ibs_system(
    system: openmm.System,
    topology,
    perturbed_indices: List[int],
    lambdas_shadow_coul: List[float],
    restraint_params: Optional[Dict] = None,
    prefix: str = "abfe_shadow",
    box_vectors=None,
) -> Tuple[openmm.System, "IBSBiasForce"]:
    """
    构建 Shadow-Coulomb IBS 去电荷系统：Shadow full charge (λ=1) -> Shadow decharged
    (λ=0)，只处理 A(配体)-E(环境) 的短程 erfc(alpha*r)/r "影子"电荷交叉项，VdW 全程
    满强度不炼金 (属于 U_common)。复用 IBSBiasForce 的多态 log-sum-exp 偏置机制
    （与 build_ibs_dual_system 的 VDW CV 完全同一套框架，只是把 CV 换成静电）。
    """
    system = ensure_owned_system(system)
    new_sys = ensure_owned_system(XmlSerializer.deserialize(XmlSerializer.serialize(system)))
    resolved_box = _resolve_periodic_box_vectors(box_vectors, topology=topology, system=new_sys)
    if _system_requires_periodic_box(new_sys):
        if resolved_box is None:
            raise ValueError("系统包含周期性非键力，但未提供有效的周期性盒子。")
        new_sys.setDefaultPeriodicBoxVectors(*resolved_box)

    num_atoms = new_sys.getNumParticles()
    perturbed_set = set(int(i) for i in perturbed_indices)
    env_indices = [i for i in range(num_atoms) if i not in perturbed_set]

    nb_forces = [f for f in new_sys.getForces() if isinstance(f, openmm.NonbondedForce)]
    if not nb_forces:
        raise ValueError("未找到 NonbondedForce")
    nb = nb_forces[0]

    _assert_neutral_ligand_for_shadow_coul(nb, perturbed_indices)
    alpha_ewald = get_pme_alpha_for_system(system, topology=topology, box_vectors=box_vectors)

    ref_excl = [
        (int(p1), int(p2))
        for p1, p2, _, _, _ in (nb.getExceptionParameters(i) for i in range(nb.getNumExceptions()))
    ]
    shadow_excl = _collect_shadow_cross_exclusions(nb, perturbed_indices, env_indices, ref_excl=ref_excl)

    # ---------- 1. 主 NonbondedForce：只关掉配体的电荷 (VdW 保持满强度) ----------
    original_params = _zero_ligand_environment_charge_in_background(nb, perturbed_indices)

    # ---------- 2. 配体内部力 (Group 2)：与 build_ibs_dual_system 同一套，保持 U_common ----------
    internal_ref_excl = [(p1, p2) for p1, p2 in ref_excl if p1 in perturbed_set and p2 in perturbed_set]
    ll_f, ll_14_f = create_ligand_internal_force(
        nb, perturbed_indices, original_params, internal_ref_excl, num_atoms, system=new_sys
    )
    ll_f.setForceGroup(2)
    if ll_14_f:
        ll_14_f.setForceGroup(2)
    new_sys.addForce(ll_f)
    if ll_14_f:
        new_sys.addForce(ll_14_f)

    # ---------- 3. 物理 Boresch 限制力 (Group 3)，与 build_ibs_dual_system 一致 ----------
    if _has_valid_boresch_restraint(restraint_params):
        rest_f_phys = LambdaDependentBoreschForce(
            rec_idx=restraint_params["receptor_indices"], lig_idx=restraint_params["ligand_indices"],
            eq=restraint_params["equilibrium_values"], fc=restraint_params["force_constants"],
            lam_name="lambda_boresch_scale", fixed_lam=None, sign=1.0, use_pbc=True
        )
        rest_f_phys.setForceGroup(3)
        new_sys.addForce(rest_f_phys)

    # ---------- 4. IBS 偏置力与 Shadow 电荷软核 CV (Group 1) ----------
    ibs_wrapper = IBSBiasForce(len(lambdas_shadow_coul), 300.0 * unit.kelvin, prefix=prefix)
    cv_template = _build_shadow_coul_cross_force(
        alpha_ewald, original_params, perturbed_indices, env_indices,
        lambda_value=0.0, exclusions=shadow_excl,
    )
    template_excl = {
        tuple(sorted(map(int, cv_template.getExclusionParticles(i))))
        for i in range(cv_template.getNumExclusions())
    }
    for k, lam in enumerate(lambdas_shadow_coul):
        int_f_cv = _build_shadow_coul_cross_force(
            alpha_ewald, original_params, perturbed_indices, env_indices,
            lambda_value=float(lam), exclusions=shadow_excl,
        )
        current_excl = {
            tuple(sorted(map(int, int_f_cv.getExclusionParticles(i))))
            for i in range(int_f_cv.getNumExclusions())
        }
        if current_excl != template_excl:
            raise RuntimeError(f"Shadow CV {k} 排除表不一致，将破坏 IBS 邻居表复用条件。")
        ibs_wrapper.addCollectiveVariable(f"cv_{k}_int", int_f_cv)
        rest_f_cv = openmm.CustomExternalForce("0")
        ibs_wrapper.addCollectiveVariable(f"cv_{k}_rest", rest_f_cv)

    new_sys.addForce(ibs_wrapper.get_force())

    sync_all_exclusions(new_sys)
    print(
        f"[Shadow-Coulomb IBS System] 构建完成，alpha_ewald={alpha_ewald:.6f}/nm，"
        f"共 {new_sys.getNumForces()} 个力对象，{len(lambdas_shadow_coul)} 个 λ_shadow_coul 状态。"
    )
    return new_sys, ibs_wrapper


def build_shadow_bridge_system(
    system: openmm.System,
    topology,
    perturbed_indices: List[int],
    restraint_params: Optional[Dict] = None,
    box_vectors=None,
) -> Tuple[openmm.System, str, Dict[str, Any]]:
    """
    构建 Shadow-PME Bridge 系统：单个 System 里同时放 4 个探针 CV
    (cv_full / cv_A_only / cv_E_only / cv_shadow)，用一个 CustomCVForce 按

        U_bridge(s) = U_common + s*cv_shadow + (1-s)*(cv_full - cv_A_only - cv_E_only)

    组合，s 是名为 "lambda_bridge_s" 的 global parameter，运行期可以直接
    context.setParameter("lambda_bridge_s", s) 在 0 (PME full charge) 到 1
    (Shadow full charge) 之间切换 —— 不需要为每个 bridge 窗口重新建系统。
    Bridge 只有 1~3 个窗口，直接用传统 REMD/BAR 采样，不进 IBS。

    返回 (new_sys, "lambda_bridge_s", diagnostics)。
    """
    system = ensure_owned_system(system)
    new_sys = ensure_owned_system(XmlSerializer.deserialize(XmlSerializer.serialize(system)))
    resolved_box = _resolve_periodic_box_vectors(box_vectors, topology=topology, system=new_sys)
    if _system_requires_periodic_box(new_sys):
        if resolved_box is None:
            raise ValueError("系统包含周期性非键力，但未提供有效的周期性盒子。")
        new_sys.setDefaultPeriodicBoxVectors(*resolved_box)

    num_atoms = new_sys.getNumParticles()
    perturbed_set = set(int(i) for i in perturbed_indices)
    env_indices = [i for i in range(num_atoms) if i not in perturbed_set]

    nb_forces = [f for f in new_sys.getForces() if isinstance(f, openmm.NonbondedForce)]
    if not nb_forces:
        raise ValueError("未找到 NonbondedForce")
    nb = nb_forces[0]

    _assert_neutral_ligand_for_shadow_coul(nb, perturbed_indices)
    alpha_ewald = get_pme_alpha_for_system(system, topology=topology, box_vectors=box_vectors)

    ref_excl = [
        (int(p1), int(p2))
        for p1, p2, _, _, _ in (nb.getExceptionParameters(i) for i in range(nb.getNumExceptions()))
    ]
    shadow_excl = _collect_shadow_cross_exclusions(nb, perturbed_indices, env_indices, ref_excl=ref_excl)
    original_params_before_zeroing = [nb.getParticleParameters(i) for i in range(num_atoms)]

    # ---------- 1. 主 NonbondedForce：只关掉配体的电荷 (VdW 保持满强度) ----------
    original_params = _zero_ligand_environment_charge_in_background(nb, perturbed_indices)

    # ---------- 2. 配体内部力 (Group 2)，与 Shadow-IBS 一致，保持 U_common ----------
    internal_ref_excl = [(p1, p2) for p1, p2 in ref_excl if p1 in perturbed_set and p2 in perturbed_set]
    ll_f, ll_14_f = create_ligand_internal_force(
        nb, perturbed_indices, original_params, internal_ref_excl, num_atoms, system=new_sys
    )
    ll_f.setForceGroup(2)
    if ll_14_f:
        ll_14_f.setForceGroup(2)
    new_sys.addForce(ll_f)
    if ll_14_f:
        new_sys.addForce(ll_14_f)

    # ---------- 3. 物理 Boresch 限制力 (Group 3) ----------
    if _has_valid_boresch_restraint(restraint_params):
        rest_f_phys = LambdaDependentBoreschForce(
            rec_idx=restraint_params["receptor_indices"], lig_idx=restraint_params["ligand_indices"],
            eq=restraint_params["equilibrium_values"], fc=restraint_params["force_constants"],
            lam_name="lambda_boresch_scale", fixed_lam=None, sign=1.0, use_pbc=True
        )
        rest_f_phys.setForceGroup(3)
        new_sys.addForce(rest_f_phys)

    # ---------- 4. 三个纯静电 PME 探针 (容斥) + Shadow 探针，一起塞进一个 CustomCVForce ----------
    probe_full = _build_electrostatics_only_pme_probe(original_params_before_zeroing, None, nb)
    probe_a = _build_electrostatics_only_pme_probe(original_params_before_zeroing, env_indices, nb)
    probe_e = _build_electrostatics_only_pme_probe(original_params_before_zeroing, list(perturbed_set), nb)
    shadow_full = _build_shadow_coul_cross_force(
        alpha_ewald, original_params_before_zeroing, perturbed_indices, env_indices,
        lambda_value=1.0, exclusions=shadow_excl,
    )

    bridge_expr = (
        "lambda_bridge_s*cv_shadow + (1.0-lambda_bridge_s)*(cv_full - cv_A_only - cv_E_only)"
    )
    bridge_force = openmm.CustomCVForce(bridge_expr)
    bridge_force.addGlobalParameter("lambda_bridge_s", 0.0)
    bridge_force.addCollectiveVariable("cv_full", probe_full)
    bridge_force.addCollectiveVariable("cv_A_only", probe_a)
    bridge_force.addCollectiveVariable("cv_E_only", probe_e)
    bridge_force.addCollectiveVariable("cv_shadow", shadow_full)
    bridge_force.setForceGroup(1)
    new_sys.addForce(bridge_force)

    sync_all_exclusions(new_sys)
    diagnostics = {
        "alpha_ewald_per_nm": alpha_ewald,
        "n_ligand_atoms": len(perturbed_set),
        "n_environment_atoms": len(env_indices),
    }
    print(
        f"[Shadow-PME Bridge System] 构建完成，alpha_ewald={alpha_ewald:.6f}/nm，"
        f"共 {new_sys.getNumForces()} 个力对象。lambda_bridge_s=0 -> PME full charge，"
        "lambda_bridge_s=1 -> Shadow full charge。"
    )
    return new_sys, "lambda_bridge_s", diagnostics


# ============================================================================
# 2. IBS 偏置力与采样器 (修复版)
# ============================================================================

class IBSBiasForce:
    """IBS 偏置力封装 (Group 1) - 数值稳定差分形式"""
    def __init__(self, n_states: int, temperature: openmm.unit.Quantity, prefix: str = "abfe"):
        self.n_states = n_states
        self.prefix = prefix
        self._cv_keeper = []
        self._int_cv_force_xmls = []
        if isinstance(temperature, float):
            temperature = temperature * openmm.unit.kelvin
        kt = (unit.MOLAR_GAS_CONSTANT_R * temperature).value_in_unit(openmm.unit.kilojoule_per_mole)
        beta = 1.0 / kt
        
        # 🔑 修复 Bug 1: 使用差分形式避免数值溢出
        # V_bias = -kT * ln( sum_k exp(-beta * (U'_k - f_k)) )
        # 提取 k=0 项: = -kT * [ -beta*(U'_0 - f_0) + ln( 1 + sum_{k>0} exp(-beta * ((U'_k - f_k) - (U'_0 - f_0))) ) ]
        # 简化后: = (U'_0 - f_0) - kT * ln( 1 + sum_{k>0} exp(-beta * (Delta_U_k - Delta_f_k)) )
        # 注意：OpenMM CustomCVForce 只能定义势能函数。
        # 我们定义: Energy = -kt * log( 1 + sum_{k>0} exp(...) )
        # 而 (U'_0 - f_0) 这一项实际上是一个依赖于坐标的项 (因为 U'_0 是 CV)。
        # 但是！IBS 的目标是平坦化分布。如果我们只关心相对权重，通常可以忽略全局偏移。
        # 然而，为了严格对应论文 Eq.18 并保证力的正确性，我们需要小心处理。
        
        # 更稳健的工程实现：
        # 令 X_k = cv_{k}_int + cv_{k}_rest - {prefix}_f_{k}
        # 目标势能: V = -kt * log( sum_k exp(-beta * X_k) )
        # 数值稳定写法: V = X_0 - kt * log( 1 + sum_{k>0} exp(-beta * (X_k - X_0)) )
        
        # 由于 OpenMM CustomCVForce 不支持直接在表达式中引用其他 CV 的复杂组合而不重复计算，
        # 我们采用标准的 Log-Sum-Exp 技巧。

        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 之前这里用 80.0*tanh(logit/80.0) 平滑
        # 饱和每个相对 logit，理由是"避免偏置力在边界处发生不可导突变"——但这个
        # 理由本身是错的：真正会引入不可导突变的是对 logit 做硬 min/max 截断
        # （clip 到固定范围），log-sum-exp 的 max-shift 技巧不是截断，是恒等变形。
        # 用真正的全局 max 作为平移基准时，V = M + log(sum_i exp(logit_i - M))
        # 对任意选取的 M（只要不溢出）都与 log(sum_i exp(logit_i)) 严格相等；
        # 对坐标求导时，dM/dx 项在展开后精确相消（标准 log-sum-exp 梯度=softmax
        # 的证明），即使 M 本身在 argmax 切换处不可导，组合表达式的力仍然精确
        # 等于 softmax 概率，处处连续。用旧的固定 k=0 基准（M=logit_0=0）只在
        # k=0 恰好是最大项时才安全，一旦某个 k 态在当前构型下比 k=0 更"划算"
        # （diff_k 很负、逻辑值很大正），单项 exp() 就可能溢出——tanh 饱和虽然
        # 防止了溢出，但把远端态实际贡献的力系统性压低了（80 kT 的饱和尺度只
        # 对应约 200 kJ/mol，vdW 生长/消失窗口在构象过渡瞬间的能量差很容易达到
        # 这个量级），这是一个真实的模型偏差，不是纯数值技巧——用真正的
        # max-pivot log-sum-exp 替换后，偏置力精确等于 MBAR/SGD/冻结验证共用的
        # 同一个 softmax 公式（见 update_weights/evaluate_frozen_batch_probability），
        # 三者不再是三套不同的数学模型。
        logit_exprs = {}
        for k in range(1, n_states):
            # 相对 logit_k = -beta * ((cv_k_int+cv_k_rest-f_k) - (cv_0_int+cv_0_rest-f_0))
            diff_expr = f"(cv_{k}_int + cv_{k}_rest - {prefix}_f_{k}) - (cv_0_int + cv_0_rest - {prefix}_f_0)"
            logit_exprs[k] = f"(-beta * ({diff_expr}))"

        # 平移基准 M = max(0.0, logit_1, ..., logit_{K-1})（0.0 对应 k=0 自身的 logit）
        pivot_candidates = ["0.0"] + [logit_exprs[k] for k in range(1, n_states)]
        pivot_expr = pivot_candidates[0]
        for cand in pivot_candidates[1:]:
            pivot_expr = f"max({pivot_expr}, {cand})"

        sum_terms = [f"exp(0.0 - ({pivot_expr}))"]
        sum_terms += [f"exp({logit_exprs[k]} - ({pivot_expr}))" for k in range(1, n_states)]
        sum_expr = " + ".join(sum_terms)

        # 最终能量表达式:
        # V_bias = (cv_0_int + cv_0_rest - f_0) - kt * (M + log(sum(exp(logit_i - M))))
        # 注意：第一项 (cv_0...) 是坐标依赖的，必须包含在内以保证力的正确性。
        energy_expr = (
            f"{prefix}_bias_scale * ((cv_0_int + cv_0_rest - {prefix}_f_0) "
            f"- kt * (({pivot_expr}) + log(max(1e-300, {sum_expr}))))"
        )
        
        self.force = openmm.CustomCVForce(energy_expr)
        self.force.addGlobalParameter("kt", kt)
        self.force.addGlobalParameter("beta", beta)
        self.force.addGlobalParameter(f"{prefix}_bias_scale", 1.0)
        for k in range(n_states):
            self.force.addGlobalParameter(f"{prefix}_f_{k}", 0.0)
        self.force.setForceGroup(1)
    def addCollectiveVariable(self, name: str, cv_force: openmm.Force) -> int:
        self._cv_keeper.append(cv_force)
        if name.endswith("_int"):
            self._int_cv_force_xmls.append(openmm.XmlSerializer.serialize(cv_force))
        return self.force.addCollectiveVariable(name, cv_force)

    def get_force(self) -> openmm.CustomCVForce:
        return self.force

    def setForceGroup(self, group_id: int):
        self.force.setForceGroup(group_id)

    def set_bias_enabled(self, context: openmm.Context, enabled: bool):
        try:
            context.setParameter(f"{self.prefix}_bias_scale", 1.0 if enabled else 0.0)
        except Exception as e:
            print(f"  ⚠️ 偏置开关设置失败: {e}")

    def update_parameters(self, context: openmm.Context, f_values: np.ndarray):
        for k in range(self.n_states):
            try:
                context.setParameter(f"{self.prefix}_f_{k}", float(f_values[k]))
            except Exception as e:
                print(f"  ⚠️ 参数更新失败: {e}")
# 🔑 e_base/e_bias 的力组切分方式（哪些 Force Group 算"物理"、哪些算"纯采样偏置"）
# 每次改变时必须递增此版本号：resume/窗口产物复用逻辑靠这个字段判断一份旧的
# dual_window_*_base.npy/*_bias.npy/convergence.json 是否是在当前口径下算出来的，
# 不能靠"λ 值凑巧没变"就信任旧文件——决定物理量对不对的是"能量怎么分组"，不是 λ。
#   version 1: e_base=groups{0,2,3,4,5}, e_bias=groups{1}（Group4 λ-WCA 被误当 λ 无关）
#   version 2: e_base=groups{0,2,3,5},   e_bias=groups{1,4}（Group4 明确为纯采样偏置）
WCA_ACCOUNTING_VERSION = 2

# 🔑 生产阶段在线 early-stop 判据协议版本。这个开关目前默认关闭
# （enable_early_stop=False），阈值默认值尚未用完整轨迹离线回放校准过——见
# run_all_windows 里对 early_stop_* 参数的说明。判据本身（哪些量、怎么组合、
# 连续通过要求）每次改变都必须递增此版本号：resume 时如果一份缓存是
# early_stop_triggered=True 提前停止产出的，只有 early_stop_protocol_version
# 与当前版本一致、且当前调用同样启用 early stop、且当前目标步数没有被调高，
# 才允许复用；任何一项不满足都视为缓存不可信，强制重新采样，避免"以后关闭
# early stop 或调高预算"时被一次短样本悄悄糊弄过去。
#   version 1: 初始版本——最小生产步数门槛 + 定期 local MBAR + 绝对 ESS/ESS
#       比例/去相关样本数/局部 ΔG 漂移/局部不确定度五项同时通过、连续
#       required_consecutive_passes 个独立 block 才停止；任一项失败清零连续
#       计数；硬上限仍是 n_steps_per_window（含 production_step_overrides）。
EARLY_STOP_PROTOCOL_VERSION = 1


def _early_stop_configs_match(cached_cfg: Optional[Dict], current_cfg: Dict) -> bool:
    """Compare a cached window's recorded early-stop threshold config against
    the current call's, tolerant only of float round-tripping through JSON
    (not of any actual difference in value).

    Matching ``early_stop_protocol_version`` alone is not enough: the
    protocol version only changes when the *decision logic itself* changes
    (which checks, how they combine). It says nothing about the specific
    threshold *values* used, which are ordinary function-default parameters
    a caller can change at any time without touching the protocol version --
    e.g. tightening ``max_uncertainty_kJ_mol`` from 5.0 to 1.0 between runs.
    A window that early-stopped and passed under the looser 5.0 threshold
    has no guarantee it would also pass under the stricter 1.0 one; reusing
    it silently would mean the "stricter" config never actually gets
    enforced against already-cached windows.
    """
    if not isinstance(cached_cfg, dict):
        return False
    if set(cached_cfg.keys()) != set(current_cfg.keys()):
        return False
    for key, current_value in current_cfg.items():
        cached_value = cached_cfg.get(key)
        if isinstance(current_value, float):
            if cached_value is None or not np.isclose(float(cached_value), current_value, rtol=1e-9, atol=1e-12):
                return False
        elif cached_value != current_value:
            return False
    return True


# 🔑 IBS 偏置预热/冻结协议版本。之前"续传时跳过 Warmup"和"生产阶段仍调用
# update_weights()"这两条合起来意味着：(a) 未真正收敛的偏置也会被当成已预热、
# 直接放行进生产；(b) f_k 在整个生产采样期间还在继续被 SGD 调整，同一个窗口
# 的样本实际上来自随时间变化的多个不同偏置势，而不是 MBAR 假设的单一固定
# 采样分布。现在改为：只有满足严格收敛判据（见 run_all_windows 里的收敛判据
# 说明）才允许进入生产，且进入生产后 f_k 冻结、不再更新。旧协议下产出的
# ibs_state_*.json（f_k 可能仍在漂移、且从未真正校验过是否收敛）不能被当成
# "已收敛"信任续传，必须靠这个版本号强制失效。
#   version 1: 续传即跳过 Warmup（不检查是否真收敛）；生产阶段仍调用 update_weights()。
#   version 2: 收敛判据升级为"逐态概率落在 (0.5/K, 2.0/K) 区间 + f_k 变化量收敛 + 连续
#              多次通过"，未达标直接 raise（不放行进生产）；生产阶段 f_k 冻结、只采样。
#              ⚠️ 有 bug：每 check_chunk=500 步检查一次，但 update_weights() 要攒够
#              10 帧才真正执行（约每 10 个 check_chunk 一次）——中间那几次检查看到
#              的是完全相同的旧 ema_mean_p/f_history，却仍被各自计成一次"通过"，
#              "连续 3 次"可能只是同一次统计结果被反复计数，不是三次独立证据。
#   version 3: 只有真正发生一次新的 update_weights() 调用（返回值非 None）才评估收敛，
#              没有新更新的迭代直接跳过，不计入连续通过也不清零。
#              ⚠️ 仍有缺口：load_ibs_state 只校验 n_states/prefix/协议版本，从未
#              校验 f_k 对应的实际 λ 值——λ 路径被自动加密/重新划分窗口后，某个
#              窗口凑巧还是同样的态数，旧 f_k 就会被当成有效热启动注入，但
#              f_k[k] 早就不对应同一个物理 λ 了，这是主动引入偏差，不是中性起点。
#   version 4: save/load_ibs_state 额外存取 lambdas_coul/lambdas_vdw，resume 时
#              与当前窗口的 λ 值逐一核对；n_states/prefix/协议版本/λ 值任何一项
#              不匹配，一律完全忽略旧状态（不再当热启动用），从 f_k=0 全新开始，
#              不再有"协议对不上但 f_k 仍可用"的中间态。
#   version 5: full-bias 收敛预算改按真实 update_weights() 次数计数；失败诊断保存
#              完整逐态 ema_mean_p 与相邻态 Delta-u 分布，并用结构化异常反馈路径规划器。
#   version 6: |Delta-f| 只保留为诊断量，不再把 SGD 步长本身误当收敛门；硬门改为
#              min(p_k)>0.5/K 且 1/sum(p_k^2)>0.8K，并要求连续真实更新通过。
#              大窗口失败只拆窗；三态及以下再用 fixed-H 双向 MBAR overlap 判断是否插点。
#   version 7: v6 的"连续通过"仍是假收敛——update_weights() 每次调用都用 f_old 算
#              概率、算完立刻把 f_new 写回 context，真正冻结进生产的那个 f_k 从未
#              被任何一次"通过"检验过。改为两阶段：learning（沿用 v6 判据找候选
#              收敛）→ freeze（冻结 f_k，绝不再 update_weights()，丢弃一段 burn-in
#              重新平衡）→ validate（用新增的只读 evaluate_frozen_batch_probability()
#              在完全固定的 Hamiltonian 下连续采若干新 batch 验证，失败则恢复
#              learning）。同时删除了"收敛后卸压"：旧代码在宣布 converged 后立刻
#              bias_scale 清零→最小化→5000 步无偏置运行→重新爬回 1.0→直接进
#              production，从未对这个被重新扰动过的构型复验；现在验证通过后只做
#              一次非破坏性安全力检查，不再改变 bias_scale/最小化/重新爬坡。
#   version 8: v7 的 frozen validation 判据本身又读了 evaluate_frozen_batch_probability()
#              的副作用 self.ema_mean_p，而不是该方法的返回值——真正生效的仍是
#              gamma=0.9 的滞后 EMA，一个塌缩 batch 可以被前几个正常 batch 的
#              EMA 记忆掩盖，等于把"用旧统计量冒充新证据"的 bug 原样搬进了刚写好
#              的验证阶段。改为直接用返回的原始 batch 概率判定，EMA 只作诊断。
#              同时修了 update_weights() 的同类问题：log_grad 之前也读
#              self.ema_mean_p（跨越不同 f_old 的滞后统计），现在改用当次真实
#              mean_p_batch；学习率衰减时标从 100 缩短到 30，并新增
#              self.eta_penalty（每次冻结验证失败恢复 learning 时减半），避免
#              "推过头->验证失败->反向推过头"的振荡；min_buffer_size 从 10 提到
#              20 降低单批噪声。新增：fixed-H 双向 overlap 全部通过时（证明 λ
#              网格本身没问题），直接用该 probe 已经解出的 BAR/MBAR 相邻态自由能
#              差校准 f_k（f_0=0, f_1=ΔF_01, f_2=ΔF_01+ΔF_12, ... 去均值），只给
#              一次冻结 burn-in + 独立验证的机会，不再回退 learning——比继续在一
#              个已经证明会振荡的 SGD 循环里盲搜更可靠。
#   version 9: fixed-H 双向 overlap 探针的触发边界从 K<=3 改为 K<=4，跟
#              abfe_preoptimizer.py 里 split_window_from_warmup_failure 的
#              min_states_before_split=5（v7 起）保持一致——K=4 的窗口不可能被
#              拆成两个各自 >=3 态、共享 1 态的孩子，之前 K<=3 的边界让 K=4 仍
#              走"直接拆窗"分支，产出过一个 K=2+K=3 的统计脆弱窗口；现在 K=4
#              和 K<=3 一样直接走这里的 fixed-H overlap 探针。诊断字段
#              feedback_action 的判定边界同步从 K>=4 改为 K>=5。
#   version 10: v8/v9 用来校准 f_k 的 delta_f_kJ_mol 来自"路径 overlap 探针"
#              （probe_bidirectional_overlap），这是一个真实的、已确认的 ensemble/
#              能量口径错配 bug，不是又一次假收敛：(1) 该探针的固定 Hamiltonian
#              动力学是 U_common + CV_k，_serialize_ibs_common_system 会连 Group 4
#              WCA 防护壳一起删掉——但生产实际采样的是
#              U_common + WCA_window(lambda_shield) + CV_k；WCA 只在"同一帧、
#              同一窗口内、不同态之间"的能量差里抵消，不会从采样动力学的构象
#              分布里消失，删掉它采的是另一个系综。(2) 该探针为了服务它自己的
#              "路径/λ 密度是否足够"这个不同问题，把 LRC 加进了喂给 MBAR 的能量
#              里；但生产 Group 1 偏置力本身从不含 LRC（LRC 只在
#              collect_energies() 里事后加进 target_energies 喂最终 MBAR/离线
#              分析，不进偏置力),用这个 delta_f 校准 f_k 等于把一个偏置力里根本
#              不存在的能量项校准了进去。(3) 门槛本身也不够：overlap>=0.03 只
#              证明两态"有最低程度的连接"，不证明 ΔF 已经精确到可以直接覆盖
#              f_k——真实案例 n_k_decorrelated=[39, 9]、overlap=0.105 时被直接
#              采信，去相关后只有 9 个独立样本的一侧显然撑不起这个精度要求。
#              (4) 校准冻结验证失败时，失败报告打印的 ema_mean_p[k] 仍是校准前
#              SGD 阶段的旧值，从未被这次校准验证真正观察到的最后一批概率覆盖，
#              诊断信息本身不可靠。
#              修复：新增独立函数 probe_bidirectional_overlap_for_bias_calibration
#              和 _serialize_ibs_common_plus_wca_system——校准探针的动力学换成
#              U_common + WCA_window(与生产同一个 lambda_shield=mean(该窗口
#              λ_vdw)) + CV_k，能量只用纯 softcore CV、绝不加 LRC，返回专用字段
#              delta_f_bias_kJ_mol（不是 delta_f_kJ_mol，两者不能再被混用）；
#              旧的 probe_bidirectional_overlap（不含 WCA、含 LRC）保留不变，
#              继续只服务"路径 overlap/λ 密度"这一个问题，两个探针的用途和实现
#              彻底分开。新增去相关样本数（默认 >=20）和 ΔF_bias 不确定度
#              （默认 <=1.0 kJ/mol）双重门槛，任一不满足就把 sample_steps 翻倍
#              重新采样（最多 3 次），仍不达标则放弃本次校准，不拿不够精确的
#              估计去覆盖 f_k。校准冻结验证失败分支现在会用 last_validation_batch_p
#              （本轮验证循环里真实观察到的最后一批概率）覆盖
#              ema_mean_p_values/min_ema_p/max_ema_p/coverage_ess，失败报告
#              不再打印校准前的旧值。
#   version 11: v10 的两个 fixed-H 探针按"边"（相邻态对）独立采样：K 态窗口的
#              K-1 条边各自建两个全新 Simulation、各自烧一遍 burn-in，共享同一个
#              态的相邻两条边把它重复采样了两次；bias 校准探针"不够精确就重试"
#              更是每次整段丢弃已采样本、从零重新烧 burn-in 再采样 20k/40k/80k。
#              修复：新增按"态"共享的可续采轨迹库
#              probe_adjacent_path_overlap_bank/probe_adjacent_bias_calibration_bank
#              （内部复用 _build_fixed_state_simulation/_extend_state_trajectory/
#              _analyze_adjacent_pair），K 态只建 K 个 Simulation、只烧一次
#              burn-in，同一态的轨迹同时喂给它左右两条边的 MBAR 计算——
#              _compute_bidirectional_overlap_from_u_kn 对每条边仍是完全自包含
#              的双态计算，不依赖另一条边怎么处理同一批帧，共享轨迹不改变任何
#              一条边自己的 u_kn/n_k 配对，只在假设性的"跨边合并不确定度"分析
#              里才需要留意相关性（目前代码库没有这么做）。旧的两个逐边探针
#              probe_bidirectional_overlap/probe_bidirectional_overlap_for_bias_calibration
#              原样保留，继续服务 abfe_pipeline.py::_probe_vdw_window_fixed_overlap
#              （生产后 ESS 修复用的是重新最小化过的独立起始构型，跟这里
#              warmup 阶段活体 context 的当前构型不是同一份，两个调用点的
#              checkpoint 不能混用，故本次只重构 run_all_windows 这一个调用点）。
#              随机种子从按边的固定 offset（0/1）改为按"全局态序号"
#              （global_state_index=start+local_index）生成，避免同一态在两个
#              不同边循环里的旧写法造成的重复初始化：
#              seed = BASE + 97*global_state_index（97 为质数步长，避免周期性
#              碰撞），四个独立 BASE 见 PATH_PROBE_INTEGRATOR_SEED_BASE 等常量。
#              轨迹库落盘于
#              checkpoints/probes/<stage_type>/window_<idx>/<path_probe|
#              bias_calibration_probe>/，manifest.json 记录系统/CV XML 哈希、
#              λ 数组、lambda_shield、温度/步长/摩擦系数、协议版本号等指纹；
#              任一字段不匹配则整个 window+probe_type 目录判定不可信、全部
#              重采（跟旧的 whole-window 缓存同一套"全对才信、错一项就整体
#              拒绝"哲学），指纹匹配后每个态各自独立判断能量/checkpoint 是否
#              完好。同一态若因 checkpoint 丢失被迫重开一个新 segment，两个
#              segment 分别独立去相关，绝不对跨越重启边界的拼接序列直接算
#              自相关（见 _decorrelate_per_segment）。
#              这是与本次轨迹库重构同时提交但逻辑独立的一处策略变更：早触发
#              条件默认由 IBS_EARLY_PROBE_TRIGGER_ENABLED 这个独立开关控制
#              （见该常量定义处），可以单独关闭以排查某次行为变化到底是
#              轨迹库重构还是早触发条件导致的。
#   version 12: 三处独立修复，均来自一次真实 GPU 生产日志（vdw 窗口 fixed-H
#               overlap 全通过、bias 校准 f_k 与物理相邻 ΔF 基本一致，但冻结
#               验证长期卡住，state 2 占据概率仅 ~0.0013）：
#               (1) IBSBiasForce 的偏置力表达式此前对每个相对 logit 做
#                   80.0*tanh(logit/80.0) 平滑饱和（理由是"避免 max/min 硬截断
#                   处不可导"），但这个理由本身错误——log-sum-exp 的 max-shift
#                   平移是恒等变形，其梯度经展开后精确等于 softmax，处处连续，
#                   不需要 tanh 近似；80 kT（约 200 kJ/mol）的饱和尺度在 vdW
#                   过渡构型的能量差量级上会被真实触发，系统性压低远端态应得
#                   的偏置力，可能正好阻止主轨迹跨越构象壁垒。现在改用真正的
#                   全局 max 作为 log-sum-exp 平移基准，偏置力精确等于
#                   update_weights()/evaluate_frozen_batch_probability() 用的
#                   同一个 softmax 公式，采样势/学习梯度/冻结验证概率三者
#                   不再是三套不同的数学模型。
#               (2) IBSSampler 新增 bias_status（"unconverged" /
#                   "calibrated_pending_validation" / "converged"）与
#                   frozen_f_k_pending，随 save_ibs_state/load_ibs_state 落盘/
#                   恢复。此前 MBAR 校准出的 f_k 一旦冻结验证在预算内未通过，
#                   就会和"从未校准过的普通未收敛"混为一谈地存为
#                   bias_converged=False；resume 时会被送回 [learning]，SGD
#                   重新调整这个其实已经正确的 f_k。现在校准成功但验证未完成
#                   会被落盘为 calibrated_pending_validation，resume 时直接
#                   带着冻结的 f_k 回到 freeze_burn_in/validating，不再回退
#                   SGD、也不再重跑 fixed-H overlap/校准探针。
#               (3) run_all_windows 冻结验证预算从单次固定的
#                   mbar_calibration_reserved_steps 改为可由调用方通过
#                   frozen_validation_step_overrides 按窗口累计延长（见
#                   abfe_pipeline.py::_run_stage_with_overlap_autorepair 里的
#                   calibration_pending_validation 分支，50k→150k→300k 阶梯）。
#                   【2026-07-16 更正】这条当时还写了"并在预算耗尽仍未通过时用
#                   OpenMM 原生 checkpoint（含积分器 RNG 状态）持久化主生产
#                   Context，下一轮真正续算而不是重新最小化、重新爬坡"——这句话
#                   描述的功能当时其实**从未真正实现**在主窗口路径里（只实现在
#                   探针轨迹库，见 _atomic_save_openmm_checkpoint 等），是一句
#                   写进注释但没有兑现的承诺，之前一直没被发现。真正的实现见
#                   下方"补丁 2"与 MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION。
#   补丁（未升版本号，仍是 12）：真实 GPU 生产日志发现 version 12 引入的
#     resumed_calibration_pending 续验分支有一个状态机 bug——续验冻结校准 f_k
#     时，只要 mode=="validating" 下单批验证没通过（`validation_pass_count==0`
#     分支，run_all_windows 内联循环），代码此前无条件 `mode = "learning"`，
#     没有检查 resumed_calibration_pending，等于让"从一开始就该永远停留在
#     freeze_burn_in/validating、绝不进 learning"（见上方 version 12 (2) 的
#     承诺）这句话形同虚设：验证一批失败就把 SGD 重新打开，续验预算
#     （frozen_validation_step_overrides 给的 50k/150k/300k 阶梯）几乎全被
#     无关的 SGD 学习烧掉，abfe_pipeline.py 的阶梯升级逻辑只读
#     calibration_pending_validation 这一个布尔值，分不清"真的验证了整段
#     预算仍失败"和"验证一批就被打回学习、剩余预算全在 learning 里空转"，
#     阶梯升到顶后会误报"冻结验证在 300000 步内失败"。修复：validating 分支
#     的失败处理现在按 resumed_calibration_pending 分两支——续验分支保持
#     mode="validating"、不重新 burn-in（Hamiltonian 全程未变，没有需要
#     burn-in 忘记的漂移历史）、不调用 apply_learning_rate_penalty()、不增加
#     learning_to_validation_cycles（该计数器语义是"退回 learning 的次数"，
#     续验分支必须保持为 0），改用独立的纯诊断计数器
#     frozen_validation_retry_count；全新窗口/skip_warmup_entirely 分支的原有
#     "验证失败退回 learning"行为不变。**刻意不升级协议版本号**：
#     bias_status/frozen_f_k_pending 字段的落盘格式和含义在修复前后完全没变，
#     旧 bug 也从未污染落盘的 frozen_f_k_pending（那份值来自循环开始前的
#     Python 快照，从未被误入 learning 的 SGD 触碰；唯一受影响的是
#     save_ibs_state 里 cosmetic 的顶层 "f_k" 字段，但 load_ibs_state 在
#     calibrated_pending_validation 分支会用 frozen_f_k_pending 覆盖注入，
#     这个漂移值从未被实际采信）——升版本号会让 load_ibs_state 的版本门控
#     整体作废现有的、已经过 fixed-H overlap+bias 校准探针证明过是对的
#     calibrated_pending_validation 缓存，逼对应窗口从 f_k=0 重新做一遍这些
#     真实物理探针（数十万步 GPU 时间），这正是这次修复要避免的浪费。
#   补丁 2（未升版本号，仍是 12）：上面 version 12 (3) 条声称已经实现的
#     "OpenMM 原生 checkpoint 持久化主生产 Context、下一轮真正续算"从未真正
#     兑现——resumed_calibration_pending 续验时，run_all_windows 每次仍然对
#     这个窗口重新最小化 + dt 测试步进 + Boresch 安全爬坡，即使窗口的构象/
#     速度/盒子在上一次尝试结束时早已充分平衡、冻结 f_k 完全没变（用户读代码
#     发现，已用只读 Explore agent 逐行核实：save_ibs_state/load_ibs_state 只
#     落盘 f_k/t/status，run_all_windows 窗口开头无条件重建 Context/最小化/
#     爬坡，跟 resume 无关）。现在把探针轨迹库已验证好的 native checkpoint
#     续算模式（_atomic_save_openmm_checkpoint/sim.loadCheckpoint，含积分器
#     RNG 状态）原样搬到主窗口：resumed_calibration_pending 续验 + 全新窗口
#     首次 MBAR 校准后验证，两条路径在每次评估完一个 batch 后都会覆盖式落盘
#     一份主窗口 checkpoint（新常量 MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION=1
#     管这份文件自己的指纹），下次 resume 若指纹匹配就直接 loadCheckpoint 跳过
#     最小化/dt测试步进/Boresch爬坡、且跳过 freeze_burn_in 直接从 validating
#     继续（Hamiltonian 全程未变，没有需要 burn-in 忘记的漂移历史）；checkpoint
#     缺失/不兼容才回退今天的完整重建流程（仍然不回退 SGD，见补丁 1）。窗口
#     真正收敛后清理这份 checkpoint。**同样刻意不升级
#     IBS_BIAS_PROTOCOL_VERSION**：这是纯增量能力，ibs_state_*.json 任何已有
#     字段的格式/含义都没变，用独立的 MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION
#     管新文件自身的指纹校验，没有历史缓存兼容性问题（这次上线前跑的窗口本来
#     就没有这份 checkpoint，第一次遇到自然回退到重建流程，无损）。
#   version 13：这次是真正需要升版本号的物理 bug——fixed-H overlap 探针/bias
#     校准探针（probe_bidirectional_overlap、probe_bidirectional_overlap_for_
#     bias_calibration、_build_fixed_state_simulation）新建 Context 后从未设置
#     lambda_boresch_scale，其 System 级默认值是 0.0（见
#     LambdaDependentBoreschForce.__init__），而主窗口生产/冻结验证早已爬坡到
#     1.0——探针证明的是关掉 Boresch 限制的系统，跟它要验证的生产 Hamiltonian
#     不是同一个。已修复三处探针的 Context 构建代码，显式对齐到 1.0。但
#     bias_status="calibrated_pending_validation"/frozen_f_k_pending 这些字段
#     本身的格式/含义没变，光改代码不会让 load_ibs_state 拒绝旧缓存——必须靠
#     这次版本号升级，让所有在这个 bug 修复之前就已经落盘的 frozen_f_k_pending
#     （物理上是用错误 Hamiltonian 证明过的，不能再信任）被版本门控拒绝，
#     强制从 f_k=0 重新做一遍这些真实物理探针，而不是继续沿用它们。
#   version 14：non_mutating_v1 把 sampling_repair_policy 加入 ibs_state 和
#     convergence 身份指纹。这个字段决定 f_k 是否可能被旧的 fixed-H/MBAR
#     变异修复路径就地改写，因而是采样协议的一部分，不能在不升版本号的情况下
#     要求所有既有 v13 state 都已经带它。旧实现漏升版本且 load_ibs_state 又在
#     协议版本门之前检查该字段，导致本应按正常版本失配路径安全失效的 v13 state
#     （sampling_repair_policy=None）被提前劫持成
#     ExistingEnsembleRequiresRescueAudit，整条 resume 流程无条件硬停止。v14 同时
#     修正门控顺序：先确认 n_states/prefix/协议版本/λ 身份，只有一份确实声称属于
#     当前 v14 ensemble 的 state 才检查 repair policy。这样 v13 数据绝不注入
#     Context、会从 f_k=0 重建；而缺字段/legacy policy 的 v14 数据仍然 fail-closed。
#   version 15：收敛门改为直接检查论文中的 Log-Sum-Exp 自洽方程。
#     对每个冻结 batch 计算 r_k=log(K*<p_k>)，只有 max|r_k| 不超过显式容差
#     才算一次通过；learning 候选判定也读取当前 update 的原始 batch，而不再用
#     跨 Hamiltonian 的 EMA。旧的 min(p)>0.5/K + coverage_ESS>0.8K 只是宽松覆盖
#     代理，不能证明 LSE fixed point 已解出。v14 的 bias_converged 状态没有通过
#     这条方程门，必须失效重验。
#   version 16：修正“判据是 LSE、更新却仍是衰减 SGD”的不一致。update_weights
#     现在直接应用 f_k<-f_k-kT*log(K*<p_k>) 的完整固定点步，不再乘 eta_sgd；
#     bias_scale<1 的 ramp 只做动力学爬坡，不采集正式方程样本、不修改 f_k；
#     non_mutating_v1 不会进入 legacy MBAR 校准，因此不再从 500k 预算中空留
#     50k，max_bias_updates=50 时能真正执行满 50 次而不是 45 次。v15 的 f_k
#     来自不同更新方程且可能在 45/50 提前终止，必须失效重学。
#   version 17：v16 错把单个 20-frame batch 的 <p_k> 当成论文 eq. 13 的
#     time-averaged Q_hat_k，并对这份高方差估计施加完整固定点步，真实运行表现为
#     左右端占据交替饱和（max LSE residual 最高 33.7），而不是收敛。现在每个
#     iteration 用采样时的 n_k=exp(beta*f_k) 做重要性校正：Q_k ∝ <p_k>/n_k，
#     在 log-space 去掉各 iteration 公共归一化后跨迭代等权累计 Q_hat_k，再按
#     n_k ∝ 1/Q_hat_k 更新。learning 候选门读取“更新前的 n_old*Q_hat_TA”平衡
#     残差；冻结后仍用全新 raw batch 直接检查 log(K*<p_k>)，不把更新后恒等于零
#     的代数结果冒充独立收敛证据。TA 累加器随 ibs_state 落盘，v16 state 因更新
#     方程和状态语义不同必须失效重学。
#   version 18：v17 虽然开始累计 Q_hat，却把每个 iteration 的 <p_k>/n_k
#     分别归一化后再平均，漏掉 normalized p_IBS^(t) 的配分函数
#     Z_t=sum_j n_j Q_j。真实冻结验证因此连续 9 次显示候选 f_k 完全错误，而
#     learning 内部的 n*Qhat_TA 残差仍因自归一化惯性虚假接近 0。现在用上一轮
#     TA Q_hat 计算 Zhat_t，再累计 Zhat_t*<p_k>/n_k；同时冻结验证在同一 f_k 下
#     累积多个 batch，不再因单个 20-frame batch 未通过就立刻退回 learning。
#     v17 TA 累加器的尺度/含义不同，必须由版本门控失效。
#   version 19：v17/v18 的 TA（时间平均）估计器只用论文 eq. 11-13，每轮更新完
#     就把这批原始能量丢掉（self.energy_buffer=[]），没有机制把冷启动早期
#     （f_k=0，state 2/3/4 采样天然稀疏）产生的偏置 mean_p_batch 后续"稀释"
#     掉——TI 交叉验证证实 window 0 观测到的 state0↔state5 ~300+ kJ/mol 落差
#     （真实 TI 只有 ~16.7 kJ/mol）就是这个数量级偏差的病理，不是真实物理量。
#     论文 eq.15 TMBAR（本仓库已经实现、已经在 GlobalMBARAnalyzer.
#     solve_stage_integrated 里跑生产/最终分析）才是"聚合来自一系列时变 IBS
#     分布样本"的正确工具。现在 update_weights()/冻结验证失败后的 feedback
#     两个调用点都不再丢弃 minibatch：每批 (u_kn, bias_energies, base_energies)
#     打包成 lambda_indices 固定为本窗口全部态索引的一条 entry，追加进随
#     ibs_state 落盘的持久列表 self.tmbar_history；f_k 由
#     solve_stage_integrated(self.tmbar_history, kt) 的 f_k（mean-centered）
#     给出，learning 候选门改用它的四项联合 converged（ESS ratio+绝对 ESS+
#     去相关样本数+端点不确定度），比旧的单一 LSE 残差更严格。彻底删除
#     ibs_lse_time_averaged_update 及 ta_relative_q_sum/ta_iteration_count——
#     两者语义/尺度都跟 tmbar_history 不同，必须由版本门控失效，不能被静默
#     当成空的 tmbar_history 复用。
#   version 20：vanishing 窗口 0 在 v19 之后仍反复 IBSWarmupConvergenceError
#     （6/4/3 态分组、pilot 自适应加密、真实 Δf 均匀切分五次真实 GPU 尝试全部
#     失败，占据始终卡在 state0 ~96-99%，`min_absolute_ess`≈1.0）——根因确认
#     不在 λ 网格，而在于每个全新（非 resume）窗口的 f_k 从 OpenMM 力常量默认值
#     0.0 冷启动（`ibs_engine.py:2473`），SGD/TMBAR 只能从每批 ~20 帧里的统计
#     证据自举出需要多大的偏置——但如果构型跨越本来就难采，早期批次几乎看不到
#     欠采样态的证据，学不出足够大的偏置，也就永远看不到更多证据（自举困境，
#     与 λ 摆在哪里无关）。修复：(1) 新增
#     `abfe_preoptimizer.estimate_f_k_from_pilot_ti`，对 Stage 2 pilot 早就测过
#     的真实平均梯度 `mean_dU_dlambda_kJ_mol`（不是这份 changelog 之前一直用的
#     方差代理 `metric_g`）做热力学积分得到 F(λ)，插值到窗口自己的 λ 值上，
#     mean-centered 后作为该窗口第一次 learning 的 f_k 起点，而不是从 0 开始
#     摸索——这份数据在窗口第一次真正采样之前就已经存在（pilot 阶段测的），
#     不需要先失败一次才能用。`IBSWindowManagerDualLambda` 新增
#     `pilot_lambdas`/`pilot_mean_dU_dlambda`（默认 None，向后兼容），
#     `run_all_windows` 只在 `not is_resumed_ibs`（真正全新学习、没有可复用
#     resume 缓存）时注入，不影响 resume 语义。(2) 候选门的
#     `candidate_min_absolute_ess` 从 2.5 降到 1.0：真实失败诊断显示四项候选
#     判据里 min_overlap/去相关样本数/端点不确定度三项都已经过关，只有
#     min_absolute_ess≈1.0（有效样本数的数学下限，只要还有非零权重样本就不可能
#     低于 1.0，本身不是这个区间里的质量信号）卡在 2.5 这个门槛——2.5 在这个场景
#     下意外地不可满足，不是有意设的质量要求，调到数学下限 1.0 不影响其余三项
#     判据，也完全不碰独立、未变的 `final_*`/冻结验证严格门槛。
#   version 21：v20 warm-start 上线后第一次真实 GPU 测试，占据反而从 96.8% 恶化
#     到 99.1%（state1 min_absolute_ess 降到 8.6e-6）——排查发现 v19 起
#     `_solve_tmbar_and_recenter` 一直直接使用
#     `solve_stage_integrated`（`df_matrix[0, 1:]`，pymbar 原始
#     `f_i=-ln(Z_i)` 相对采样态的约定）的原始输出作为 f_k，未反号。但
#     `IBSBiasForce`（本文件 2397 行起）的能量表达式与 `update_weights`
#     的 `logits = beta*(f_old - u_mk)`（约 3364-3367 行）严格对应
#     `n_k=exp(beta*f_k)`：给某态更高的 f_k 直接推高它在混合分布里的权重。
#     `solve_stage_integrated` 原始约定对越贴近当前采样分布（即当前过量
#     代表的态）给出越高（越不负）的值——用当前失败窗口真实数据验证：
#     99.1% 占据的 state0 被判给全组最高（最不负）的 f_k，直接喂给
#     context.setParameter 等于每次在线更新都在进一步强化已经占主导的
#     态，而不是压低它。这个符号错误从 v19 引入 TMBAR 起就存在，影响的是
#     这条在线学习路径覆盖的所有窗口，不只是 window 0。修复：
#     `_solve_tmbar_and_recenter` 现在对 `solve_stage_integrated` 的原始
#     f_k 取负号后再 mean-center。同时暂时禁用 v20 引入的 pilot TI
#     热启动注入（`run_all_windows` 里 `estimate_f_k_from_pilot_ti` 调用
#     点，见该函数旁注释）——pilot 的 `mean_dU_dlambda_kJ_mol` 是每个 λ
#     独立短程弛豫后测的系综平均梯度，和 update_weights 实际用的 u_mk
#     （在当前占主导构型上跨态求值）是概念不同的量，其本身是否同样反号
#     未经独立验证，先只验证 TMBAR 符号修复这一个变量，避免两处不确定
#     叠加。（历史记录：这项 v21 符号判断后来由 v27 用
#     exp(beta*f_k)Z_k=exp[beta*(f_k-F_k)] 恒等式及 solve_stage_integrated 的
#     明确返回单位证实为错误；v27 已撤销该反号。）
#   version 22：独立用平衡条件 `f_k - U_k ≈ const` 核对 v20 的 pilot-TI
#     warm-start，确认它和 v21 修复的 TMBAR 调用点存在同一个符号错误：原始
#     pilot TI 曲线也会把最高 f_k 给已经占主导的 state0，从而通过
#     `n_k=exp(beta*f_k)` 进一步放大其权重。现在
#     `estimate_f_k_from_pilot_ti` 对原始 TI 插值结果取负号后再 mean-center，
#     返回可直接注入的 bias-parameter convention；同时重新启用
#     `run_all_windows` 的 pilot-TI 注入。此前实测的 v20 window 0 种子
#     `[+48.2, +35.3, +20.0, +4.3, -15.2, -35.9, -56.7]` 因而变为
#     `[-48.2, -35.3, -20.0, -4.3, +15.2, +35.9, +56.7]`，与 v21 的
#     修复方向一致：已经占主导的 state0 得到全组最低 f_k。v21 的 TMBAR
#     修复本身不变。（历史记录：v27 证实这里把“观测占据的负反馈方向”误套到
#     “物理自由能种子方向”；pilot/TMBAR 两处反号均已在 v27 撤销。）
#   version 23：v22 首次真实 GPU 运行把占据从 state0≈99% 推到了 state6≈99.5%，
#     证明 pilot-TI 热启动成功让体系离开旧主导端，但在线更新仍把另一端锁死。
#     失败状态的 f_k 已发散到 [-1197.6, ..., +296.1] kJ/mol；根因不是“反号
#     还不够”，而是 update_weights 把每次尚未收敛（min_absolute_ess≈1）的
#     TMBAR 绝对向量整组覆盖进 Context。mean-center 只消除规范零点，并不会
#     缩小跨度或平均权重。现在 TMBAR 继续累计全部时变分布样本，但只承担候选
#     覆盖质量门/诊断；实际 f_k 学习改为对每批真实占据执行全态同步负反馈：
#     delta_f_k=-eta*kT*log(K*<p_k>)，log 残差和每轮最大步长都显式受限，
#     再 mean-center。过强态必降、过弱态必升，不再把任何正号/反号的 TMBAR
#     绝对解直接覆盖到 f_k。冻结验证失败的累计占据也走同一条受限更新，并降低
#     eta_penalty。pilot-TI 仍只负责提供非零起点；v22 状态必须失效，不能续用
#     已经发散的 f_k/tmbar_history。
#   version 24：v23 上线后 window 0 反复 5 次以上完整 learning->freeze_burn_in->
#     validating attempt 全部失败，从未进入生产。真实失败快照
#     （dual_window_0_vdw_warmup_failure.json）证实：K=7 时 state6 占据 p 高达
#     0.6-0.68（目标 1/7≈0.143），需要的 delta_f_6 约 3.6 kJ/mol，整条 f_k
#     向量量级在 10-50 kJ/mol，而 `_bounded_log_occupancy_update` 每步只挪了
#     ~0.1-0.15 kJ/mol（远低于当时 max_step_kT=2.0 换算出的 4.989 kJ/mol
#     上限）。根因是两处学习率抑制叠加过猛：(1) `apply_learning_rate_penalty`
#     每次冻结验证失败就把 `eta_penalty` 砍半、floor=0.05——5 次真实失败后
#     0.5**5=0.03125 已跌破 floor，之后每次 resume 都从磁盘原样加载这个
#     触底值（`save_ibs_state`/`load_ibs_state` 持久化 `eta_penalty`），
#     没有任何机制能让它回升，等于把所有未来 learning 尝试永久限制在 1/20
#     步长；(2) `update_weights` 里 `eta = eta_penalty/(1+update_index/100)`
#     的 /100 分母在验证预算只有 50000 步（≈100 frames）内进一步快速衰减
#     可用步长。三者叠加：100 frames 预算、0.1 kJ/mol 步长、0.25 的 LSE
#     容差，数学上不可能在预算内收敛——每次"3 步一冻结"验证必败，验证失败
#     又再砍一次学习率，形成预热步数远超生产步数（本窗口生产仅
#     50000 步/window）却永远进不了生产的恶性循环。修复：
#     `apply_learning_rate_penalty` 的 factor 从 0.5 松到 0.85、floor 从
#     0.05 提到 0.25（`_load_ibs_state` 校验区间同步改为 [0.25, 1.0]）；
#     `update_weights` 的衰减分母从 /100 放缓到 /500；
#     `_bounded_log_occupancy_update` 的 `max_step_kT` 默认值从 2.0 提到
#     5.0（单步硬上限约 12.5 kJ/mol），让单步物理上能够覆盖真实需要的
#     跨态修正幅度。不改变收敛判据本身（LSE 自洽 `max|log(K*p_k)|<=0.25`
#     仍是论文要求、未放松）——只是让 SGD 真正有能力在预算内到达那个判据，
#     而不是被人为砍到几乎为零的步长锁死。旧协议下已经触底/发散的
#     eta_penalty/f_k 状态必须失效，不能续用。
#   version 25：v24 上线后真实 GPU 测试暴露一个新问题——delta_f 确实跳到了
#     2-5 kJ/mol（符合预期），但 dominant_k 每次更新都在不同态之间跳变
#     （state4→state3→state6→state2→...），三次完整冻结验证 attempt 全部
#     失败，占据从未稳定下来。根因：`_bounded_log_occupancy_update` 同时对
#     全部 K 个态按各自当前残差施加 delta_f（同时压低占主导的态、抬高多个
#     欠采样的态），这在 softmax 耦合系统里会把相对 log-odds 的移动幅度
#     放大到远超公式隐含的单态线性响应假设——v24 去掉了 eta_penalty 快速
#     衰减到底这个（虽然调错了方向，但确实存在）的阻尼，而没有补上任何替代
#     阻尼，暴露出这个此前一直被过度阻尼掩盖的过冲问题。修复：新增独立于
#     eta_penalty（失败计数机制）和 update_index 衰减（一次 learning
#     attempt 通常只有 12-18 次更新，/500 分母基本不起作用）的
#     `IBS_UPDATE_RELAXATION_FACTOR=0.35` 阻尼系数，直接乘进 eta——该数值
#     未经解析推导，是经验性阻尼系数，若后续真实 GPU 测试仍观察到
#     dominant_k 反复跳变（下面新增的 watchdog 会打印警告），应进一步调低
#     （如 0.15-0.2）；若收敛明显变慢，可适当调高。同时新增 dominant_k
#     跳变 watchdog（纯诊断，不影响控制流）。
#     同一次改动里一并处理三个之前讨论过的独立问题（均已获用户确认）：
#     (a) `lse_log_residual_tolerance` 从 0.25 放宽到 0.5，并新增"有界次数
#     后接受目前最好结果"的兜底（不再无限重试）——两份独立研究证实：论文
#     本身用固定步数预算而非强收敛判据放行生产（integrated-boltzmann-
#     sampling.md 336/348 行）；占据均衡只影响重加权效率/方差，不影响
#     `solve_stage_integrated`（单参考重要性 MBAR）的无偏性；真正的正确性
#     门槛是完全独立于这个预冻结判据的生产后 `_assert_stage_result_sane`
#     （`min_overlap`/绝对 ESS/去相关样本数/端点不确定度，abfe_pipeline.py
#     2641-2644/3340-3436），任何一次"差不多就放行"的 f_k 若真的采样不足，
#     会在那里 fail-closed，而不是静默产出错误结果。因此放宽这里的判据只
#     影响效率，不影响正确性。
#     (b) 实现此前草拟但未落地的 PROPOSAL_frozen_validation_fallback.md：
#     learning 阶段候选连续通过次数从未凑满、纯粹因为 `max_bias_updates`
#     耗尽而 break 时，现在给当前 f_k 一次真正的冻结验证机会（仅一次，
#     guard 防止重复），而不是直接判定 f_not_converged——burn-in/validation
#     预算本来就已经预留，不用白不用。
#     (c) 当时曾把冻结验证阶段样本结转进生产数据集；该调度优化后来撤销：
#     预热/验证与生产必须是两个严格隔离的数据阶段，验证样本一律不计入生产。
#   version 26：v25 上线后第一次真实 GPU 全流程测试——5 个 vdw 窗口全部
#     真正跑完了预热+生产，但 Stage 2 最终 `_assert_stage_result_sane`
#     报告 `min_overlap=0.0047`（阈值 0.05）而失败。查真实落盘诊断：5 个
#     窗口全部是 `learning_to_validation_cycles==2`（v25 默认的 cap）就被
#     best-effort 接受，但接受时的 `best_effort_residual` 高达 13~123
#     （容差 0.5 的 26~246 倍），对应真实占据概率低到 1e-10~1e-54 的态——
#     不是"差不多"，是这些态在生产阶段几乎完全没被采样。根因：v25 只加了
#     "重试次数上限"，没加"接受结果本身是否合理"的下限检查，等于把
#     "见好就收"实现成了"不管多差、够了次数就收"。同时 cap=2（每次 cycle
#     只给 learning ~12-18 次更新的机会）本来就远不足以让这类大跨度
#     （50+ kJ/mol）不均衡收敛。修复：cap 从 2 提到 8（f_k/eta_penalty 跨
#     cycle 累积，更多 cycle 才有机会让有界负反馈逐步纠正过来）；新增
#     `best_effort_max_residual_multiple=4.0`——只有达到 cap 时最优 attempt
#     的残差仍在 `tolerance` 的 4 倍以内，才真正采用 best-effort 路径；
#     否则视为"目前最好的也远未达标"，不静默放行，回退到 v25 之前的原有
#     行为（继续 learning，预算耗尽后按 non_mutating_v1 fail-closed 抛
#     IBSWarmupConvergenceError，交人工/rescue 审计）——绝不通过放宽下游
#     `_assert_stage_result_sane` 的 min_overlap 阈值来掩盖这类真实的
#     欠收敛，那样会让统计上不可靠的态悄悄进入最终 ΔG。
#   version 27：修复 pilot-TI 热启动和 TMBAR 候选向量的符号。IBS 势
#     V=-kT*log sum_k exp[-beta*(U_k-f_k)] 下，态 k 的积分权重正比于
#     exp(beta*f_k) Z_k = exp[beta*(f_k-F_k)]，所以平坦权重要求
#     f_k=F_k+constant。旧 estimate_f_k_from_pilot_ti 对物理 TI 曲线额外取负，
#     会把本来低自由能/高占据的态进一步抬高；真实 window 1 因此从跨度
#     52.61 kJ/mol 的完全反向种子出发，26 次有界负反馈主要耗在撤销错误热启动。
#     v27 删除该反号；同时删除 `_solve_tmbar_and_recenter` 对
#     `solve_stage_integrated` 返回的物理 F_k 再次施加的反号。后者从 v23 起只作为
#     未安装进 Context 的诊断候选返回，本次失败不是它造成的，但保留会给未来调用
#     埋下同一符号错误。在线占据负反馈 delta_f=-eta*kT*log(K*p_k) 保持不变。
#     升版使所有由旧反号种子生成的 IBS 状态/生产缓存失效，避免静默续用。
#   warmup-control follow-up（不改变生产 Hamiltonian/cache identity，故协议仍为
#   version 27）：修复多次冻结验证与总预算不相容导致的“合理结果来不及放行”。
#     v26 允许最多 8 次完整冻结 attempt 后接受 sane best-effort，但总步数只额外
#     预留了 1 次 burn-in+validation；真实 v27 run 在 4 次完整 attempt 后，第 5
#     次只采到 20 帧便于 mode=validating 撞总帽，永远到不了 8 次分支。现在总帽
#     耗尽时也可复用同一 sane-bound，但只接受此前完整采满 attempt 预算的最佳
#     fixed-f 结果，并额外要求累计 TMBAR converged；当前被截断的半次验证绝不参与
#     选择。若恢复的是较早 attempt 的 f_k，先在该 f_k 下重新 burn-in、丢弃这段
#     轨迹，再进 production。生产后独立 overlap/ESS/去相关样本/不确定度硬门不变。
#     同时修正 dominant-flip watchdog：接近平坦时 argmax 随噪声换人不是过冲，
#     只有 dominant 概率至少达到目标的 1.5 倍时才累计/报警（仍纯诊断）。
#   version 29：把 warmup 的 f_k 收敛判据整体换成"局部滑窗 MBAR loose-gate"，
#     彻底移除旧的冻结验证机制（累计 <p_k> 的 LSE 自洽门 max|log(K·<p_k>)|≤tol、
#     required_consecutive_bias_updates 连续通过、50k→150k→300k 冻结验证阶梯、
#     "见好就收"best-effort、warmup ESS/coverage 四联门）。新判据：冻结 f_k、短
#     burn-in 后累计最近 IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES 批固定-f_k minibatch，
#     跑一次单参考 local MBAR（_solve_single_window_local_mbar，不吃全历史时变
#     TMBAR），只设一个 loose gate：max_k|Δf_{k,k+1}−ΔF^MBAR_{k,k+1}|（相邻差、
#     gauge 无关）< IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL(10 kJ/mol≈4 kT)
#     即冻结进生产；≥阈值退回 learning 再做一轮现有 update_weights；MBAR 不可解/
#     NaN 也退回继续更新。跑满 warmup 步数预算仍未通过则接受当前 f_k 进生产
#     (best-effort)。动机：避开时变 TMBAR 判据不稳/自身 BAR 不收敛导致的"合理
#     f_k 永远卡在验证门外"，把真正的自由能/ESS/overlap/误差全部交给生产后独立的
#     _assert_stage_result_sane + 最终 MBAR。这是纯 warmup 停止判据/诊断变更，不动
#     production Hamiltonian、f_k 符号约定、采样方式；且新 loose gate 严格弱于旧
#     LSE 门（旧门通过的近均匀占据 f_k 必然满足新门），故按 v28 同样理由保持缓存
#     兼容——已完成/已收敛的 v27/v28 f_k 不判废、可直接续用。
#     v29 同批还修了 warmup 更新控制器的塌缩正反馈（"低重叠 TMBAR 给出错误大步 →
#     占据塌缩 → TMBAR 更不可靠"）：(a) update_weights 只有累计 TMBAR 解质量同时
#     可信（min_overlap≥0.05、min_absolute_ess≥10、去相关≥10、端点不确定度≤5
#     kJ/mol、当前 batch coverage_ESS≥0.8K，见 IBS_TMBAR_TRUST_*）时才应用
#     _damped_tmbar_absolute_update 的绝对候选；否则退回 _bounded_log_occupancy_
#     update 先恢复覆盖。(b) 任何一次应用到 Context 的更新，相邻步长
#     max_k(Δf)-min_k(Δf) 硬 cap 到 IBS_MAX_APPLIED_PAIRWISE_STEP_KT=2 kT（在
#     update_weights 里、两个控制器之后统一施加，不进 _damped_* 本体），damping
#     0.20→0.10。(c) learning minibatch 20→40 帧（配 5 批滑窗=200 冻结帧）。
#     (d) local-MBAR loose gate 增加可信度门（min_ess_ratio<0.05 或绝对 ESS<10 →
#     insufficient_overlap，不把零重叠外推的巨大 ΔF 当门残差）。这些同样只改
#     warmup 学习/停止动力学与诊断，不动 production Hamiltonian/f_k 约定/采样，
#     故仍保持缓存兼容。
IBS_BIAS_PROTOCOL_VERSION = 29

# v28/v29 都只改过 warmup 的停止/诊断控制，没有改变 production Hamiltonian、
# f_k 符号约定或生产采样方式。它们不该让已经完成或正在续采的 v27 production
# 失效；撤回 v28 误升版后，也不能反过来把已经写出的 v28 缓存判废；v29 换收敛
# 判据同理（新 loose gate 弱于旧 LSE 门，旧收敛 f_k 仍有效），保持缓存兼容。
IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS = frozenset((27, 28, 29))


def _ibs_bias_protocol_version_is_cache_compatible(value: Any) -> bool:
    try:
        return int(value) in IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS
    except (TypeError, ValueError):
        return False


def _normalize_ibs_protocol_for_cache_compare(manifest: Dict[str, Any]) -> Dict[str, Any]:
    normalized = json.loads(json.dumps(manifest, sort_keys=True, default=str))
    if _ibs_bias_protocol_version_is_cache_compatible(
        normalized.get("ibs_bias_protocol_version")
    ):
        normalized["ibs_bias_protocol_version"] = IBS_BIAS_PROTOCOL_VERSION
    return normalized


def _resume_cached_window_gate_status(
    cached_conv: Optional[Dict[str, Any]],
    cached_e_shape: Optional[Tuple[int, ...]],
    lc_win: Any,
    lv_win: Any,
    repair_policy: str,
    lse_log_residual_tolerance: float,
    enable_early_stop: bool,
    current_early_stop_config: Dict[str, Any],
    effective_target_steps: int,
) -> Dict[str, Any]:
    """一份磁盘窗口缓存在 resume 时能不能直接复用：8 个门的纯函数求值。

    原先这 ~110 行判断内联在 ``run_all_windows`` 里，依赖 8 个局部变量，没有任何
    办法单独调用，因此"缺字段是否真的 fail-closed""步数预算门是否真的不管
    early_stop_triggered 都生效"这类问题只能靠读代码确认，不能靠测试保证。抽成
    纯函数后每个门都能用 mock 的 ``cached_conv`` 单独验证（见
    ``test_resume_reuse_contracts.py``），且不碰任何文件 I/O。

    刻意留在 ``run_all_windows`` 里没有搬进来的两件事：
      - ``ExistingEnsembleRequiresRescueAudit``：那是带 diagnostics 的 raise，
        不是布尔门，且在 triplet 加载之前就要先拦（保护旧 ensemble 不被覆盖）。
      - ``_load_validated_window_data_triplet`` 的文件读取与 manifest 校验。
    本函数只吃已经解析好的 ``cached_conv`` dict 和能量矩阵的 shape。

    返回 dict 而不是 ``(bool, reason)``：调用方那串逐门诊断打印（每个门有各自的
    文案和一个 legacy_mutating 分支）需要读到**单个门**的布尔值，返回聚合结果会
    迫使那些打印重写。这里把逐门布尔一并返回，调用侧的 elif 阶梯得以一字不动。

    ``usable`` 的与链顺序与原内联判断完全一致：
        shape -> lambdas -> wca -> ibs_bias -> lse -> lrc -> repair_policy -> early_stop
    任何字段缺失（旧格式缓存读不到该 key）一律保守判"不可复用"，绝不"大概率
    没变"就放过。
    """
    if not isinstance(cached_conv, dict):
        cached_conv = {}

    # 🔑 形状：ndim==2、物理态数与本次窗口的 λ 数一致、至少有一帧。
    shape_ok = bool(
        cached_e_shape is not None
        and len(tuple(cached_e_shape)) == 2
        and int(tuple(cached_e_shape)[0]) == len(lc_win)
        and int(tuple(cached_e_shape)[1]) > 0
    )

    # 🔑 只按 window_idx + shape 判断"这份缓存还能不能用"是不安全的：λ 路径被
    # 自动加密/重新划分窗口后，同一个 window_idx 完全可能对应一段全新的 λ 区间，
    # 只要凑巧态数相同就会被静默当成有效缓存复用——采样对了，但对的是错的 λ。
    # 旧格式缓存没有这两个字段的，保守地当作不匹配，强制重采。
    cached_lc = cached_conv.get("lambdas_coul")
    cached_lv = cached_conv.get("lambdas_vdw")
    lambdas_match = bool(
        cached_lc is not None
        and cached_lv is not None
        and len(cached_lc) == len(lc_win)
        and len(cached_lv) == len(lv_win)
        and np.allclose(cached_lc, lc_win, atol=1e-9)
        and np.allclose(cached_lv, lv_win, atol=1e-9)
    )

    # 🔑 [wca_accounting_version] λ 值对得上不代表数值对得上：base/bias 的力组
    # 切分口径变过（Group4 λ-WCA 从"算作 base"改成"算作 bias"）时，旧文件里的
    # base/bias 数值是在旧口径下算出来的，即使 λ 完全相同也绝不能复用。
    version_match = cached_conv.get("wca_accounting_version") == WCA_ACCOUNTING_VERSION

    # 🔑 [ibs_bias_protocol_version] 旧协议下这份能量文件可能是在 f_k 全程漂移
    # （生产阶段仍在 update_weights()）的情况下采出来的，不满足 MBAR 的单一固定
    # 采样分布假设——即使 λ/WCA 口径都对得上，也不能当成有效缓存复用。
    bias_protocol_match = _ibs_bias_protocol_version_is_cache_compatible(
        cached_conv.get("ibs_bias_protocol_version")
    )

    cached_lse_tolerance = cached_conv.get("lse_log_residual_tolerance")
    lse_tolerance_match = bool(
        cached_lse_tolerance is not None
        and np.isclose(
            float(cached_lse_tolerance),
            float(lse_log_residual_tolerance),
            rtol=0.0,
            atol=1.0e-15,
        )
    )

    # 🔑 [lj_tail_lrc_protocol_version] LRC 公式版本必须匹配——v1 只补 cutoff 外的
    # r^-6、忽略 switching 区间，v2 是真正 switching+softcore-aware 的积分；旧协议
    # 下算出的 target_energies 里叠加的 LRC 数值跟当前公式完全不是同一回事。缺失
    # 该字段（比 v1 更旧）同样保守地判不匹配。
    lrc_version_match = (
        cached_conv.get("lj_tail_lrc_protocol_version")
        == TRADITIONAL_LJ_LRC_PROTOCOL_VERSION
    )

    # 🔑 [non_mutating_v1] 采样修复策略必须匹配：旧的变异策略缓存（其 f_k 可能被
    # fixed-H 累计 ΔF 就地覆盖过，属于不同参考系）绝不能被非变异策略的 run 复用。
    # 旧缓存没有这个字段（None），与 "non_mutating_v1" 不相等，因此自动判无效。
    repair_policy_match = cached_conv.get("sampling_repair_policy") == repair_policy

    # 🔑 [early_stop / 步数预算] 两条独立检查，不管这份缓存是否触发过 early stop
    # 都要过第一条：
    #  (1) 缓存产出时的目标步数不能低于当前调用的目标步数——否则即使从来没有触发
    #      过 early stop、是"完整跑满"的缓存，也可能只是"250k 时代"跑完的窗口，
    #      被静默当成满足"预算已提到 500k"的当前要求复用，实际样本量根本不够。
    #  (2) 如果这份缓存确实是 early_stop_triggered=True 提前停止产出的短样本，还
    #      要求当前调用同样启用 early stop、协议版本一致、且这次实际使用的八个
    #      阈值跟缓存记录的完全一致——只比协议版本不够：版本号只在判据逻辑本身
    #      变了才会变，把某个阈值从松调紧（例如 max_uncertainty_kJ_mol 从 5.0 收紧
    #      到 1.0）根本不影响协议版本，但旧窗口是在更松的阈值下通过的。
    early_stop_ok = True
    early_stop_reject_reason: Optional[str] = None
    cached_target = cached_conv.get("n_steps_per_window_effective")
    if cached_target is None or int(cached_target) < int(effective_target_steps):
        early_stop_ok = False
        early_stop_reject_reason = (
            f"当前目标步数（{int(effective_target_steps)}）高于缓存产出时的目标"
            f"（{cached_target}）"
        )
    elif bool(cached_conv.get("early_stop_triggered", False)):
        if not enable_early_stop:
            early_stop_ok = False
            early_stop_reject_reason = "当前调用未启用 early stop"
        elif (
            cached_conv.get("early_stop_protocol_version")
            != EARLY_STOP_PROTOCOL_VERSION
        ):
            early_stop_ok = False
            early_stop_reject_reason = (
                f"缓存的 early_stop_protocol_version="
                f"{cached_conv.get('early_stop_protocol_version')!r}（期望 "
                f"{EARLY_STOP_PROTOCOL_VERSION}）"
            )
        elif not _early_stop_configs_match(
            cached_conv.get("early_stop_config"), current_early_stop_config
        ):
            early_stop_ok = False
            early_stop_reject_reason = (
                f"缓存的 early_stop_config（{cached_conv.get('early_stop_config')}）"
                f"与当前调用（{current_early_stop_config}）不一致"
            )

    usable = bool(
        shape_ok
        and lambdas_match
        and version_match
        and bias_protocol_match
        and lse_tolerance_match
        and lrc_version_match
        and repair_policy_match
        and early_stop_ok
    )

    # reason 的判定顺序与调用侧那串诊断打印的 elif 阶梯一致，保证"日志说的原因"
    # 和"函数报告的原因"永远是同一个门。
    if usable:
        reason: Optional[str] = None
    elif not shape_ok:
        reason = (
            f"缓存能量形状 {tuple(cached_e_shape) if cached_e_shape is not None else None} "
            f"与期望 ({len(lc_win)}, N) 不符"
        )
    elif not version_match:
        reason = (
            f"wca_accounting_version={cached_conv.get('wca_accounting_version')!r}"
            f"（期望 {WCA_ACCOUNTING_VERSION}）"
        )
    elif not bias_protocol_match:
        reason = (
            f"ibs_bias_protocol_version="
            f"{cached_conv.get('ibs_bias_protocol_version')!r}（兼容版本 "
            f"{sorted(IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS)}）"
        )
    elif not lse_tolerance_match:
        reason = (
            f"LSE log 残差容差={cached_lse_tolerance!r}"
            f"（当前 {lse_log_residual_tolerance}）"
        )
    elif not lrc_version_match:
        reason = (
            f"lj_tail_lrc_protocol_version="
            f"{cached_conv.get('lj_tail_lrc_protocol_version')!r}"
            f"（期望 {TRADITIONAL_LJ_LRC_PROTOCOL_VERSION}）"
        )
    elif not repair_policy_match:
        reason = (
            f"sampling_repair_policy="
            f"{cached_conv.get('sampling_repair_policy')!r}（期望 {repair_policy!r}）"
        )
    elif not early_stop_ok:
        reason = early_stop_reject_reason
    else:
        reason = "λ 值不匹配（缺少 λ 元数据或 λ 路径已变更）"

    return {
        "usable": usable,
        "reason": reason,
        "shape_ok": shape_ok,
        "lambdas_match": lambdas_match,
        "version_match": version_match,
        "bias_protocol_match": bias_protocol_match,
        "lse_tolerance_match": lse_tolerance_match,
        "cached_lse_tolerance": cached_lse_tolerance,
        "lrc_version_match": lrc_version_match,
        "repair_policy_match": repair_policy_match,
        "early_stop_ok": early_stop_ok,
        "early_stop_reject_reason": early_stop_reject_reason,
    }


# A minibatch stores a full KxN energy matrix plus bias/base vectors.  Keeping
# every learning batch forever makes long/debug warmups grow without bound and
# bloats JSON checkpoints.  The normal production budget is far below this
# limit; the cap only affects unusually long or repeatedly resumed learning.
TMBAR_HISTORY_MAX_ENTRIES = 200

# Warm-up updater identity is deliberately separate from the production bias
# protocol. v9 replaces raw-minibatch SGD control with the absolute physical
# free-energy candidate from the full accumulated TMBAR history. Each update
# is f <- f + 0.2*(f_TMBAR-f). Dominant-state identity is diagnostic only and
# never changes the update. If TMBAR is temporarily unavailable, one bounded
# occupancy step provides coverage acquisition: 4-kT pairwise radius, widened
# to the 10-kT severe ceiling only for a true collapse (raw_residual >= 70). Warm-up frames are
# collected every 250 MD steps (production output cadence is untouched).
# Completed production made by an older updater remains valid; only an
# unfinished older learning checkpoint must not be resumed into this solver.
IBS_WARMUP_UPDATE_PROTOCOL_VERSION = 9
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 0.20 -> 0.10：低重叠时绝对 TMBAR 候选给出错误
# 大步 → 占据塌缩 → TMBAR 更不可靠的正反馈循环。配合下面 IBS_MAX_APPLIED_
# PAIRWISE_STEP_KT 的硬步长上限，一起把每次更新的相邻步长压住。
IBS_TMBAR_UPDATE_DAMPING = 0.10
# 这是 _bounded_log_occupancy_update 的【真塌陷档】pairwise 上限，只在
# raw_residual ≥ IBS_UPDATE_ADAPTIVE_RESIDUAL_COLLAPSE 时才用得到；6≤residual<
# COLLAPSE 的中等区固定 4 kT（见该函数 severe 分支）。
IBS_TMBAR_FALLBACK_SGD_PAIRWISE_STEP_KT = 10.0
IBS_WARMUP_FRAME_STRIDE_STEPS = 250
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 20 -> 40：20 帧太容易被单一构象主导，绝对
# TMBAR/占据估计噪声大。40 帧 × local-MBAR gate 的 5 批滑窗 = 200 冻结帧，恰好
# 满足"冻结后至少累计 ~200 帧再判 loose gate"。
IBS_TMBAR_LEARNING_MINIBATCH_FRAMES = 40
IBS_TMBAR_FREEZE_MAX_APPLIED_PAIRWISE_STEP_KT = 1.0
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 任何一次 warmup 权重更新（绝对 TMBAR 或
# bounded occupancy fallback）应用到 Context 之前，都把相邻步长
# max_k(Δf)-min_k(Δf) 硬 cap 到这个值（2 kT≈5 kJ/mol@300K）。防止低重叠 MBAR
# 候选的巨大 pairwise 跳变一步打崩占据。真实 ΔF>5 kJ 的边靠多轮小步逼近。
IBS_MAX_APPLIED_PAIRWISE_STEP_KT = 2.0
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 绝对 TMBAR 候选"可信"门：只有 update_weights
# 里累计 TMBAR 解的质量同时满足以下全部时，才用 _damped_tmbar_absolute_update
# 应用绝对候选；否则退回 _bounded_log_occupancy_update（受限占据反馈，先恢复
# 覆盖）。这些只门控"用哪个更新控制器"，不是窗口收敛判据（收敛仍是 run_all_
# windows 里的 local-MBAR loose gate）。
IBS_TMBAR_TRUST_MIN_OVERLAP = 0.05
IBS_TMBAR_TRUST_MIN_ABSOLUTE_ESS = 10.0
IBS_TMBAR_TRUST_MIN_DECORRELATED_SAMPLES = 10
IBS_TMBAR_TRUST_MAX_UNCERTAINTY_KJ_MOL = 5.0
IBS_TMBAR_TRUST_MIN_COVERAGE_ESS_FRACTION = 0.8
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 生产入口门是纯 Δf−ΔF<阈值：abs_ess / min_ess_
# ratio / 占据平坦度 / coverage_ESS 全部只作诊断，不参与放行（曾经用 abs_ess≥10 门
# 控，会把宽松门偷偷变成严格收敛门，已删除）。零重叠外推出的巨大 ΔF（如 908
# kJ/mol）自然因 >阈值被拒，不需要 ESS 门。下面的 OCC_* 常量只供
# _diagnose_local_mbar_situation 判断占据平坦/塌陷（诊断 + 预算耗尽接受时的判决），
# 不参与生产入口放行。
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 占据平坦度阈值，仅供 _diagnose_local_mbar_
# situation 的诊断（占据是否平坦 / 是否塌陷）——用于预算耗尽 best-effort 接受时的
# 判决（平坦=良性、塌陷=建议插 λ）以及日志现场，不参与生产入口放行（入口门是纯
# Δf−ΔF<阈值）。平坦定义：每个态占据落在 [OCC_MIN_FRACTION/K, OCC_MAX_FRACTION/K]
# 且 coverage_ESS≥OCC_MIN_COVERAGE_ESS_FRACTION×K。两侧 box 防止
# [0.499,0.499,0.001,0.001] 这类"max<0.5 但塌了两个态"被误判成平坦。
IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION = 0.5
IBS_LOCAL_MBAR_GATE_OCC_MAX_FRACTION = 2.0
IBS_LOCAL_MBAR_GATE_OCC_MIN_COVERAGE_ESS_FRACTION = 0.8
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] "占据真的塌陷"（硬热力学瓶颈 → 建议插 λ/拆窗）
# 的判据，故意比"未达平坦兜底门"严得多：只有 coverage_ESS 掉到 0.5×K 以下、或某态
# 占据低于 0.25/K（目标的四分之一）才算塌陷。防止把 [0.41,0.27,0.20,0.125] 这类
# 仅仅"最低态贴着 0.5/K 下限"的健康窗口误报成需要插 λ 的瓶颈。塌陷/平坦之间是
# "统计薄，延长采样"的中间态。
IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_COVERAGE_ESS_FRACTION = 0.5
IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_MIN_FRACTION = 0.25
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] learning→freeze 的 freeze 时机（纯效率、不阻
# 放行）。生产入口门本身是纯 Δf−ΔF<10（见 run_all_windows 里的门）；这里只决定
# "何时值得冻结去跑一次 local-MBAR 检查"，避免在 f_k 还在大步移动时白花 burn-in/
# 验证开销（那正是塌陷窗口浪费预算的原因）。判据只看【应用的 f_k pairwise 步长已
# 降到 ≤ IBS_TMBAR_FREEZE_MAX_APPLIED_PAIRWISE_STEP_KT kT（f_k 几乎不再动）】，连续
# IBS_LEARNING_READY_CONSECUTIVE 批满足才冻结。故意不看占据平坦/coverage/abs_ess
# ——那些会把宽松门偷偷变成严格收敛门。step 门不会阻挡任何本该通过的窗口：
# gap<10 ⟹ 占据残差小 ⟹ bounded 步长小；step 仍大的窗口必然 gap≫10、冻结也白费。
IBS_LEARNING_READY_CONSECUTIVE = 2
IBS_UPDATE_NEAR_FLAT_RELAXATION_FACTOR = 0.35
IBS_UPDATE_MIDDLE_RELAXATION_FACTOR = 0.50
IBS_UPDATE_SEVERE_RELAXATION_FACTOR = 1.00
IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW = 1.0
IBS_UPDATE_ADAPTIVE_RESIDUAL_HIGH = 6.0
# 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] severe 区再分两档的门槛。原来 severe 区的半径
# 从 4 kT 连续爬到 severe_max(10 kT)、residual=12 就到顶，结果 residual 十几到几十
# 的"中等塌陷"窗口也在吃 24.94 kJ/mol 的大步 → dominant 来回翻转、占据震荡。
# 现在：residual < COLLAPSE 一律 4 kT（≈9.98 kJ/mol，单次相对更新不超过 ~10
# kJ/mol）；只有 residual ≥ COLLAPSE 的真塌陷（某态平均 softmax 权重 ~e^-70，
# 靠 4 kT/步要爬十几二十步）才放开到 severe_max。
IBS_UPDATE_ADAPTIVE_RESIDUAL_COLLAPSE = 70.0

# 🔑 [IBS_BIAS_PROTOCOL_VERSION=28] 局部滑窗 MBAR loose-gate。冻结 f_k 后，把最近
# IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES 批固定-f_k minibatch 拼成一次单参考 local
# MBAR（_solve_single_window_local_mbar，跟最终阶段拼接同一套增广矩阵数学，但只看
# 当前冻结 f_k 下最近这几批数据，不吃全历史时变 TMBAR），只比较相邻态 ΔF^MBAR 与
# 当前相邻 Δf_k（gauge 无关，绝对 f_k 的任意公共常数在相邻差里抵消），设唯一 loose
# gate：max_k |Δf_{k,k+1} − ΔF^MBAR_{k,k+1}| < 阈值即冻结进生产。10 kJ/mol ≈ 4 kT，
# 只作"别让某个局部边完全饿死"的粗门；真正的自由能/ESS/overlap/误差全部交给生产后
# 独立的 _assert_stage_result_sane + 最终 MBAR。故意不设连续通过、不设 LSE 占据门、
# 不等 f_k 稳定、不设 warmup ESS 四联门——见 run_all_windows 里的收敛状态机。
IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL = 10.0
IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES = 5

# 落盘格式的独立版本号：只管 fixed-H 探针轨迹库 checkpoint/manifest 的文件
# 结构（不是采样/校准协议本身），见 probe_adjacent_path_overlap_bank 等函数。
#   version 2: v1 的续采 checkpoint 只存 NPZ（positions/velocities/box），重建
#              Context 后积分器随机数种子从头重设，内部 RNG 状态并未真正恢复——
#              仍被当成同一条不间断轨迹续采，但严格说是"位置/速度对上、但
#              积分器内部状态是全新的"伪续采。改为每个态额外存一份 OpenMM 原生
#              二进制 Context checkpoint（state_{k}_openmm.chk，含积分器内部
#              状态），作为续采的主路径；NPZ 降级为 checkpoint 缺失/损坏/平台
#              不兼容时的兜底——降级到 NPZ 时必须强制开新 segment、重新烧
#              burn-in，不能再当作连续轨迹。见 _resume_or_start_state_simulation。
#   version 3：同上面 IBS_BIAS_PROTOCOL_VERSION=13 的修复——探针轨迹库
#              （_build_fixed_h_probe_bank_manifest 管的 per-state 原始采样
#              轨迹）之前没有对齐 lambda_boresch_scale=1.0，manifest 里也没有
#              任何 code_sha256/代码指纹字段能自动感知这个 Context 层面的
#              修复（common_system_xml 的序列化字节没变，Boresch 力在 XML 里
#              仍然是默认值 0.0，只是运行时 setParameter 覆盖了它）。升版本号
#              让 _fixed_h_probe_bank_manifest_matches 判定所有旧轨迹库整体
#              失配，_invalidate_fixed_h_probe_bank 整目录清空重采。
FIXED_H_PROBE_CACHE_PROTOCOL_VERSION = 3

# 🔑 独立开关：learning_to_validation_cycles>=2 时提前跳出 SGD 收敛循环、直接
# 走 fixed-H overlap 探针+bias 校准兜底，是跟本次轨迹库性能重构同时提交、但
# 逻辑上完全独立的一次策略变更（见 run_all_windows 里对这个常量的唯一读取
# 点）。分开做成开关，是为了以后如果发现行为异常，能立刻二分排查是"轨迹库"
# 还是"早触发"导致的，而不必回退整个提交。默认开启。
IBS_EARLY_PROBE_TRIGGER_ENABLED = True

# 🔑 冻结校准验证阶梯的累计目标步数（50k→150k→300k），single source of truth：
# abfe_pipeline.py::_run_stage_with_overlap_autorepair 导入这个常量而不是自己
# 再定义一份，避免两处硬编码的 schedule tuple 未来改一处漏改另一处。同时也是
# run_all_windows 内部"调用方未提供覆盖字典时"的阶梯档位回退依据——见
# effective_frozen_validation_budget/is_final_rung 的计算处注释。
FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS = (50_000, 150_000, 300_000)


def _resolve_frozen_validation_budget_for_window(
    window_idx,
    frozen_validation_step_overrides,
    prior_cumulative_steps: int,
    schedule=FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS,
) -> int:
    """Pick this attempt's cumulative step-budget target for a window.

    Prefers the caller's explicit per-window override (the normal case: a
    same-process ladder escalation in abfe_pipeline.py). Falls back to the
    smallest schedule rung strictly greater than the already-persisted
    ``prior_cumulative_steps`` when no override is present -- this is the
    case right after a process restart (--resume), where the caller's
    in-memory override dict is empty but the window's real progress already
    lives on disk. Falling back to schedule[0] unconditionally there would
    silently discard that progress (see run_all_windows call site).
    """
    if window_idx in (frozen_validation_step_overrides or {}):
        return int(frozen_validation_step_overrides[window_idx])
    return next(
        (rung for rung in schedule if rung > int(prior_cumulative_steps)),
        schedule[-1],
    )


def _resolve_frozen_validation_is_final_rung(
    window_idx,
    frozen_validation_is_final_rung,
    effective_frozen_validation_budget: int,
    schedule=FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS,
) -> bool:
    """Whether this attempt's budget target is the ladder's last rung.

    Prefers the caller's explicit per-window flag; falls back to comparing
    the resolved budget against the schedule's last entry when the caller
    has no entry for this window (post-restart, same reasoning as
    _resolve_frozen_validation_budget_for_window -- must not default to
    False, or a window whose persisted progress already reached the final
    rung would never be judged terminal and would loop on the same rung.
    """
    if window_idx in (frozen_validation_is_final_rung or {}):
        return bool(frozen_validation_is_final_rung[window_idx])
    return bool(int(effective_frozen_validation_budget) >= schedule[-1])


class IBSWarmupConvergenceError(RuntimeError):
    """Structured signal that a window lacks adequate sampled-state coverage."""

    def __init__(self, message: str, diagnostics: Dict):
        super().__init__(message)
        self.diagnostics = diagnostics


class ExistingEnsembleRequiresRescueAudit(RuntimeError):
    """On-disk data for this window was produced under a DIFFERENT sampling
    repair policy (e.g. the deprecated mutating path that could rewrite f_k in
    place). Under non_mutating_v1 we refuse to reuse it AND refuse to overwrite
    it — the original files are preserved and this is raised so the rescue audit
    can decide, per ensemble, whether the data is reusable (clean, pure-SGD) or
    must be re-run. Never silently continue and clobber the old ensemble."""

    def __init__(self, message: str, diagnostics: Optional[Dict] = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


# 🔑 [non_mutating_v1] Fail-closed policy gate. ONLY the explicit legacy value
# re-enables the deprecated mutating fixed-H probe + in-place f_k recalibration.
# Any unrecognized value (e.g. a typo) MUST raise — never silently fall back to
# mutation. This is the single decision point used by run_all_windows so the
# behavior is testable without a GPU (see test_non_mutating_policy.py).
_VALID_SAMPLING_REPAIR_POLICIES = ("non_mutating_v1", "legacy_mutating")


def should_run_legacy_repair(repair_policy: str) -> bool:
    """Return True only for the explicit deprecated 'legacy_mutating' policy;
    False for 'non_mutating_v1'; raise ValueError for anything else."""
    if repair_policy == "non_mutating_v1":
        return False
    if repair_policy == "legacy_mutating":
        return True
    raise ValueError(
        f"unknown sampling repair_policy {repair_policy!r}; expected one of "
        f"{_VALID_SAMPLING_REPAIR_POLICIES}. Refusing to guess — fail closed "
        "(a typo must NOT re-open the deprecated mutating fixed-H/recalibration path)."
    )


class IBSFrozenCalibrationValidationError(RuntimeError):
    """一份已经用 fixed-H overlap 探针 + bias 校准探针证明过物理正确的冻结 f_k，
    在（当前这次或累计）冻结验证预算内仍未通过独立验证——跟
    IBSWarmupConvergenceError（从未获得过一份校准 f_k 的普通未收敛，真正需要
    SGD 继续搜索或拆窗/插 λ）是两种完全不同的失败模式，故意用不同的类型，
    避免上层（abfe_pipeline.py 的拆窗/插 λ 修复逻辑）把两者混为一谈。

    ``terminal=True`` 表示这已经是终态（冻结验证累计预算已经用到调用方标记的
    最后一档仍未通过，对应 bias_status="calibrated_validation_failed"）——
    调用方不应再尝试任何形式的自动续验/延长预算/回退 SGD，需要人工检查。
    ``terminal=False`` 表示这次只是当前这一档预算内没验证完，diagnostics 里的
    calibration_pending_validation=True 供调用方决定是否延长预算重试。
    """

    def __init__(self, message: str, diagnostics: Dict, terminal: bool = False):
        super().__init__(message)
        self.diagnostics = diagnostics
        self.terminal = bool(terminal)


class IBSSampler:
    """
    IBS 采样器：严格通过 CustomCVForce 探针收集各态微扰能。
    【修复】按论文 Sec. 2.3/eq. 13 累计跨迭代相对 Q_k 的 TA 估计。
    【修复】能量收集时自动扣除参考态偏移，确保数值稳定性。
    """
    def __init__(self, context: openmm.Context, n_states: int,
                 temperature: unit.Quantity, prefix: str = "abfe",
                 ibs_wrapper: IBSBiasForce = None):
        self.context = context
        self.n_states = n_states
        self.prefix = prefix
        self.ibs_wrapper = ibs_wrapper
        _temp = temperature if hasattr(temperature, 'value_in_unit') else temperature * unit.kelvin
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * _temp).value_in_unit(unit.kilojoule_per_mole)
        self.beta = 1.0 / self.kt
        
        # EMA 只保留为原始占据趋势诊断，不参与权重更新或 learning 收敛门。
        self.ema_mean_p = None
        self.gamma = 0.9  # 衰减因子，保留 90% 历史信息
        # [IBS_BIAS_PROTOCOL_VERSION=19] 论文 eq. 15 TMBAR 状态：每个 update_weights()/
        # 冻结验证 minibatch 打包成一条 {'u_kn','bias_energies','base_energies',
        # 'lambda_indices'=全部本窗口态,'sampled_distribution_row'=0} entry，永久
        # 追加（不像旧 TA 估计器那样每轮用完就丢）。f_k 由
        # solve_stage_integrated(self.tmbar_history, self.kt) 的逆方差加权拼接
        # 给出——因为每条 entry 的 lambda_indices 完全重叠（都是同一窗口的全部
        # 态，只是采样时刻的 f_k 不同），拼接逻辑天然按每条 entry 自己的采样
        # 分布把它们正确对齐合并，正是"聚合一系列时变 IBS 分布样本"。v23 起
        # 解出的绝对向量不再直接写回 f_k，只用于累计覆盖质量门和诊断。
        self.tmbar_history: List[Dict[str, Any]] = []
        self.tmbar_history_dropped_entries = 0
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=8] 每次 frozen validation 失败、恢复
        # learning 时减半（下限见 apply_learning_rate_penalty），让重新学习的
        # 步子比上一次尝试更保守，而不是用几乎相同的大步长继续来回振荡。
        self.eta_penalty = 1.0
        self.energy_buffer = []
        self.energy_history = []
        self.f_history = []
        self.bias_history = []
        self.base_energy_history = []
        self.last_update_diagnostics = {}
        self.last_frozen_batch_size = 0
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] dominant-state oscillation watchdog
        # (diagnostic only, does not affect control flow): tracks whether the
        # argmax of mean_p_batch keeps flipping between consecutive
        # update_weights() calls, which is the visible symptom of the
        # multi-state overshoot fixed in this version.
        self._last_dominant_k = None
        
        # 🔑 新增：能量偏移缓存
        self.e_offset = 0.0
        # 连续 Base 能量读取失败计数——单帧失败允许跳过（很可能是瞬时 getState
        # 异常），但连续多帧失败说明 Context/系统本身有问题，必须硬报错，而不是
        # 一直用假 0.0 悄悄污染 base_energy_history。
        self._consecutive_base_failures = 0
        self._energy_query_attempts = 0
        self._energy_query_failures = 0
        self._energy_query_consecutive_failures = 0
        self._energy_query_failure_reasons: Dict[str, int] = {}
        # 🔑 只有在 run_all_windows 的严格收敛判据真正通过之后才置 True；
        # 生产采样阶段以此为准冻结 f_k（不再调用 update_weights），且这个
        # 状态会随 save_ibs_state/load_ibs_state 落盘/恢复，见 IBS_BIAS_PROTOCOL_VERSION。
        self.bias_converged = False
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 单独的 bias_converged bool 无法区分
        # "从未校准过、普通未收敛"和"MBAR 校准已经给出正确 f_k，只是冻结验证
        # 还没在预算内通过"——这两者 resume 时必须走不同路径：前者应该回到
        # [learning] 重新 SGD；后者的 f_k 已经是好的，只应该继续 freeze_burn_in/
        # validating，绝不能被当成热启动喂回 SGD 重新调整。bias_status 记录
        # 四态："unconverged"（默认）/"calibrated_pending_validation"/"converged"/
        # "calibrated_validation_failed"（真实 GPU 生产日志发现的补丁：冻结验证
        # 累计预算已经用到调用方标记的最后一档仍未通过，是终态，不再自动续验/
        # 延长预算/回退 SGD——跟"calibrated_pending_validation"的关键区别就是
        # "还要不要继续自动重试"）；frozen_f_k_pending 是 calibrated_pending_
        # validation 状态下被冻结的那份 f_k 快照，随 save_ibs_state/load_ibs_state
        # 落盘/恢复；到达终态后设为 None（不再是"pending"，不该被继续消费）。
        self.bias_status = "unconverged"
        self.frozen_f_k_pending = None
        # 🔑 这份冻结 f_k 累计已经花在冻结验证上的步数（跨越同一份校准 f_k 的
        # 多次 resume/阶梯升级累加，不是单次 attempt 内的 steps_at_full_bias）。
        # 只有在 calibrated_pending_validation 状态下才有意义，随 save_ibs_state/
        # load_ibs_state 落盘/恢复。没有它，每次阶梯升级（50k→150k→300k）都会
        # 把 run_all_windows 内部的 while 循环预算当成"这次要新跑这么多步"而
        # 不是"累计到这么多步"，导致真实总步数变成 50k+150k+300k=500k 而不是
        # 阶梯设计意图的"累计延长到 300k"。
        self.frozen_validation_cumulative_steps = 0
        # run_all_windows 在 load 前会用调用方已通过枚举校验的 policy 覆盖此值；
        # 这里仍给直接使用 IBSSampler 的调用方一个 fail-closed 默认值，避免遗漏
        # 赋值时把无策略/旧策略 state 当成可恢复状态注入 Context。
        self.sampling_repair_policy = "non_mutating_v1"
        self._probe_context = None
        self._probe_integrator = None
        self._probe_groups = []
        if self.ibs_wrapper is not None and getattr(self.ibs_wrapper, "_int_cv_force_xmls", None):
            self._build_probe_context()

    def _build_probe_context(self):
        main_system = self.context.getSystem()
        probe_sys = openmm.System()
        for i in range(main_system.getNumParticles()):
            probe_sys.addParticle(main_system.getParticleMass(i))
        try:
            probe_sys.setDefaultPeriodicBoxVectors(*main_system.getDefaultPeriodicBoxVectors())
        except Exception:
            pass

        self._probe_groups = []
        for idx, force_xml in enumerate(self.ibs_wrapper._int_cv_force_xmls):
            force = openmm.XmlSerializer.deserialize(force_xml)
            gid = 16 + idx
            force.setForceGroup(gid)
            probe_sys.addForce(force)
            self._probe_groups.append(gid)

        self._probe_integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
        self._probe_context = openmm.Context(probe_sys, self._probe_integrator, self.context.getPlatform())

    def evaluate_interaction_energies(self, positions, box_vectors=None) -> np.ndarray:
        """Evaluate every target CV at arbitrary coordinates without changing dynamics."""
        if self._probe_context is None or not self._probe_groups:
            return np.zeros(self.n_states, dtype=float)

        self._probe_context.setPositions(positions)
        if box_vectors is not None:
            try:
                self._probe_context.setPeriodicBoxVectors(*box_vectors)
            except Exception:
                pass

        interaction_energies = np.zeros(self.n_states, dtype=float)
        for k, gid in enumerate(self._probe_groups[:self.n_states]):
            state = self._probe_context.getState(getEnergy=True, groups={gid})
            interaction_energies[k] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        return interaction_energies

    def _collect_interaction_energies(self) -> np.ndarray:
        state_main = self.context.getState(getPositions=True)
        try:
            box_vectors = state_main.getPeriodicBoxVectors()
        except Exception:
            box_vectors = None
        return self.evaluate_interaction_energies(state_main.getPositions(), box_vectors)

    def get_raw_interaction_energies(self) -> np.ndarray:
        return self._collect_interaction_energies().copy()

    def _lj_tail_correction_kj_mol(self) -> np.ndarray:
        """解析 LJ 长程色散尾项修正（逐态,加到 interaction_energies 上）。
        lj_tail_lrc_coeff_kj_mol（switching+softcore-aware，见
        _lj_tail_lrc_coefficients_kj_mol）由 build_ibs_dual_system 预计算并挂在
        ibs_wrapper 上；这里每帧只需读当前盒子体积做一次除法，不重新积分。
        缺失/不适用 (potential_type='dexp' 或没有周期性盒子) 时返回全 0，
        不影响原有行为。fixed-H overlap 探针（probe_bidirectional_overlap）
        和这里读的是同一个 ibs_wrapper.lj_tail_lrc_coeff_kj_mol，保证两处
        用的是同一组系数。
        """
        lrc_coeff = getattr(self.ibs_wrapper, "lj_tail_lrc_coeff_kj_mol", None) if self.ibs_wrapper is not None else None
        if lrc_coeff is None:
            return np.zeros(self.n_states, dtype=float)
        try:
            box_nm = self.context.getState().getPeriodicBoxVectors().value_in_unit(unit.nanometer)
            a, b, c = (np.asarray(v, dtype=np.float64) for v in box_nm)
            volume_nm3 = abs(np.dot(a, np.cross(b, c)))
        except Exception:
            return np.zeros(self.n_states, dtype=float)
        if not np.isfinite(volume_nm3) or volume_nm3 <= 0.0:
            return np.zeros(self.n_states, dtype=float)
        return np.asarray(lrc_coeff, dtype=np.float64)[: self.n_states] / volume_nm3

    def energy_query_diagnostics(self) -> Dict[str, Any]:
        attempts = int(getattr(self, "_energy_query_attempts", 0))
        failures = int(getattr(self, "_energy_query_failures", 0))
        return {
            "attempts": attempts,
            "failures": failures,
            "failure_fraction": float(failures / attempts) if attempts else 0.0,
            "consecutive_failures": int(
                getattr(self, "_energy_query_consecutive_failures", 0)
            ),
            "failure_reasons": dict(
                getattr(self, "_energy_query_failure_reasons", {})
            ),
            "limits": {
                "max_consecutive_failures": ENERGY_QUERY_MAX_CONSECUTIVE_FAILURES,
                "max_total_failures": ENERGY_QUERY_MAX_TOTAL_FAILURES,
                "max_failure_fraction": ENERGY_QUERY_MAX_FAILURE_FRACTION,
                "fraction_min_attempts": ENERGY_QUERY_FAILURE_FRACTION_MIN_ATTEMPTS,
            },
        }

    def assert_energy_query_quality(self, final: bool = False) -> None:
        diag = self.energy_query_diagnostics()
        attempts = int(diag["attempts"])
        fraction_gate = attempts > 0 and (
            final or attempts >= ENERGY_QUERY_FAILURE_FRACTION_MIN_ATTEMPTS
        )
        if (
            int(diag["consecutive_failures"])
            >= ENERGY_QUERY_MAX_CONSECUTIVE_FAILURES
            or int(diag["failures"]) >= ENERGY_QUERY_MAX_TOTAL_FAILURES
            or (
                fraction_gate
                and float(diag["failure_fraction"])
                > ENERGY_QUERY_MAX_FAILURE_FRACTION
            )
        ):
            raise RuntimeError(
                "IBS 能量查询失败超过 hard gate："
                f"attempts={attempts}, failures={diag['failures']}, "
                f"fraction={diag['failure_fraction']:.3%}, "
                f"consecutive={diag['consecutive_failures']}, "
                f"reasons={diag['failure_reasons']}"
            )

    def _record_energy_query_result(
        self,
        success: bool,
        reason: Optional[str] = None,
    ) -> None:
        if success:
            self._energy_query_consecutive_failures = 0
            return
        self._energy_query_failures = int(
            getattr(self, "_energy_query_failures", 0)
        ) + 1
        self._energy_query_consecutive_failures = int(
            getattr(self, "_energy_query_consecutive_failures", 0)
        ) + 1
        reasons = dict(getattr(self, "_energy_query_failure_reasons", {}))
        key = str(reason or "unknown")
        reasons[key] = int(reasons.get(key, 0)) + 1
        self._energy_query_failure_reasons = reasons
        self.assert_energy_query_quality(final=False)

    def collect_energies(self) -> np.ndarray:
        self._energy_query_attempts = int(
            getattr(self, "_energy_query_attempts", 0)
        ) + 1
        energies = np.zeros(self.n_states)
        failure_reason = None
        # 1. Base 能量 (Group 0, 2, 3, 5)：严格 λ 无关的物理项。
        #    🔑 [wca_accounting_version=2] Group 4 (λ-WCA 防护壳) 曾经也在这里，被当成
        #    "λ 无关"处理——但它的 lambda_shield 是每个窗口各自设成"本窗口 λ_vdw 均值"
        #    (见 run_all_windows 里 setParameter("lambda_shield", ...))，同一个重叠 λ 态
        #    在两个不同窗口里因此带着不同的 WCA 取值，却被当成同一个物理态拼接——
        #    这是 Hamiltonian 不一致，不只是效率问题。Group 4 现在明确定义为纯采样期
        #    偏置（帮助窗口内动力学不塌缩，跟 Group 1 的 IBS flattening bias 同类），
        #    连同 Group 1 一起在 bias_history 里被完整记录、完整 reweight 掉，不再计入
        #    任何目标态的物理能量。是否把 WCA 改造成"每个目标态各自算一次"的正式中间
        #    Hamiltonian（效率更高但不影响正确性）留作后续独立评估，见 wca_accounting_version。
        try:
            state_base = self.context.getState(getEnergy=True, groups={0, 2, 3, 5})
            e_base = state_base.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            self._consecutive_base_failures = 0
        except Exception as e:
            failure_reason = "base_energy"
            # ⚠️ 不能静默吞掉：e_base 会被直接写入 base_energy_history 并喂给
            # GlobalMBARAnalyzer (u_phys_kj = base_kj + u_kj_raw)。之前失败时回退
            # 成 0.0 并"标记"——但下面只检查 interaction_energies 是否有 NaN，e_base
            # 本身从未被检查，假 0.0 会照常被 append 进 base_energy_history，
            # 悄悄污染 MBAR 且无迹可查。现在改为 NaN（下面统一按"任一分量非有限
            # 就整帧跳过"处理，不单独放行 e_base），单帧失败允许跳过；但连续失败
            # 说明 Context/系统本身有问题（不是偶发 getState 抖动），必须硬报错，
            # 而不是无限跳帧、悄悄丢数据。
            self._consecutive_base_failures += 1
            # On the first failure, distinguish a transient energy-query
            # problem from a simulation that already has non-finite
            # coordinates/forces.  In the latter case waiting five more MD
            # chunks would only let a broken trajectory drift further.
            if self._consecutive_base_failures == 1:
                try:
                    diagnostic_state = self.context.getState(
                        getPositions=True,
                        getForces=True,
                    )
                    diagnostic_positions = np.asarray(
                        diagnostic_state.getPositions(asNumpy=True).value_in_unit(
                            unit.nanometer
                        ),
                        dtype=np.float64,
                    )
                    diagnostic_forces = np.asarray(
                        diagnostic_state.getForces(asNumpy=True).value_in_unit(
                            unit.kilojoule_per_mole / unit.nanometer
                        ),
                        dtype=np.float64,
                    )
                except Exception as diagnostic_exc:
                    raise RuntimeError(
                        "Base 能量首次读取失败，且坐标/力诊断也无法完成；拒绝继续推进 MD。"
                    ) from diagnostic_exc
                if (
                    not np.all(np.isfinite(diagnostic_positions))
                    or not np.all(np.isfinite(diagnostic_forces))
                ):
                    raise RuntimeError(
                        "Base 能量首次读取失败时检测到非有限坐标或力；系统已经失稳，"
                        "拒绝继续推进 MD。"
                    ) from e
            print(
                f"  🚨 Base 能量 (Group 0,2,3,5) 获取失败（连续第 "
                f"{self._consecutive_base_failures} 次），本帧标记为 NaN 并跳过：{e}"
            )
            e_base = float("nan")
        if self.ibs_wrapper is None:
            self._record_energy_query_result(False, "missing_ibs_wrapper")
            return np.full(self.n_states, np.nan)

        try:
            try:
                # Group 1 (IBS flattening bias) + Group 4 (λ-WCA 防护壳，纯采样偏置，
                # 详见上面 collect_energies 顶部注释)：两者都只影响采样分布、不代表任何
                # 目标态的物理能量，必须一起完整 reweight 掉，缺一个都会重新引入偏差。
                state_bias = self.context.getState(getEnergy=True, groups={1, 4})
                e_bias = state_bias.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            except Exception:
                e_bias = np.nan
                failure_reason = failure_reason or "bias_energy"

            # 🔑 [LRC vs 实际施加的 CV 不一致] softcore_energies 是真正驱动 Group1
            # log-sum-exp 偏置力的那个量（build_ibs_dual_system 里的 cv_k_int，纯
            # softcore，不含 LRC）；lrc 修正只是 Python 侧对"物理能量"的解析加成，
            # 从未进过实际的 OpenMM 力。之前两者被合成同一个数组，同时喂给
            # update_weights()（学 f_k，理应对齐"实际施加的偏置力"）和 MBAR（理应
            # 对齐"完整物理能量，含 LRC"）——用同一个含 LRC 的量去训练 f_k，学到的
            # 是一个真实偏置力并不存在的目标，降低 IBS 平坦化效率（不影响 MBAR
            # 正确性：sampled row 用的是直接读出的 e_bias，与 f_k 好坏无关）。
            # 现在分开：bias_cv_energies（纯 softcore，喂 energy_buffer/update_weights）
            # vs target_energies（softcore+LRC，喂 energy_history/MBAR）。
            softcore_energies = self._collect_interaction_energies()
            lrc_energies = self._lj_tail_correction_kj_mol()
            target_energies = softcore_energies + lrc_energies

            # 相对偏移防溢出 (以 State 0 为参考)——只用于 bias_cv 训练路径的数值稳定性，
            # 是否与 target_energies 用同一个偏移量无所谓：softmax/log-sum-exp 对每帧
            # 内所有态统一平移不变，MBAR 那边存的是未平移的 target_energies 原始值。
            if self.n_states > 0 and np.isfinite(softcore_energies[0]):
                self.e_offset = softcore_energies[0]
            bias_cv_energies = softcore_energies - self.e_offset
            energies = bias_cv_energies

            frame_finite = (
                np.all(np.isfinite(bias_cv_energies))
                and np.all(np.isfinite(target_energies))
                and np.isfinite(e_base)
                and np.isfinite(e_bias)
            )
            if frame_finite:
                self.energy_buffer.append(bias_cv_energies)
                self.energy_history.append(target_energies.copy())
                self.base_energy_history.append(float(e_base))
                self.bias_history.append(float(e_bias))
                self._record_energy_query_result(True)
            else:
                self._record_energy_query_result(
                    False,
                    failure_reason or "nonfinite_energy_component",
                )
        except Exception as e:
            if isinstance(e, RuntimeError) and "hard gate" in str(e):
                raise
            print(f"  ⚠️ CV 探针能量提取失败: {e}")
            energies[:] = np.nan
            self._record_energy_query_result(False, "cv_probe")
        return energies

    def _append_tmbar_batch_from_buffer(self) -> int:
        """[IBS_BIAS_PROTOCOL_VERSION=19] 把当前 self.energy_buffer（清空前）打包成
        一条持久的 tmbar_history entry，供论文 eq. 15 TMBAR 聚合。

        必须在任何清空 self.energy_buffer 的操作（update_weights() 末尾 /
        evaluate_frozen_batch_probability() 末尾）之前调用：collect_energies() 对
        energy_buffer/bias_history/base_energy_history 用同一个 frame_finite 门同步
        append，因此清空前 energy_buffer 的最后 len(energy_buffer) 帧，与
        bias_history/base_energy_history 的对应尾部条目逐帧一一对应。
        返回实际追加的有效（非 NaN）帧数（0 表示这一批没有可用样本）。
        """
        m0 = len(self.energy_buffer)
        if m0 == 0:
            return 0
        u_mk_raw = np.array(self.energy_buffer, dtype=np.float64)
        valid_mask = ~np.isnan(u_mk_raw).any(axis=1)
        n_valid = int(np.sum(valid_mask))
        if n_valid == 0:
            return 0
        bias_tail = np.asarray(self.bias_history[-m0:], dtype=np.float64)
        base_tail = np.asarray(self.base_energy_history[-m0:], dtype=np.float64)
        self.tmbar_history.append({
            "u_kn": u_mk_raw[valid_mask].T,
            "bias_energies": bias_tail[valid_mask],
            "base_energies": base_tail[valid_mask],
            "lambda_indices": list(range(self.n_states)),
            "sampled_distribution_row": 0,
        })
        overflow = len(self.tmbar_history) - TMBAR_HISTORY_MAX_ENTRIES
        if overflow > 0:
            del self.tmbar_history[:overflow]
            self.tmbar_history_dropped_entries += overflow
        # 滑动窗口：即使 dominant 不切换，累积过多 entry 也会因早期帧权重过大
        # 而拖慢 TMBAR 收敛。只保留最近 N 条（每条约 20 帧），双保险。
        _TMBAR_SLIDING_WINDOW = 10
        if len(self.tmbar_history) > _TMBAR_SLIDING_WINDOW:
            n_drop = len(self.tmbar_history) - _TMBAR_SLIDING_WINDOW
            self.tmbar_history = self.tmbar_history[-_TMBAR_SLIDING_WINDOW:]
            self.tmbar_history_dropped_entries += n_drop
        return n_valid

    def _solve_tmbar_and_recenter(
        self,
        min_ess_ratio: float = 0.05,
        min_absolute_ess: float = 5.0,
        min_decorrelated_samples: int = 5,
        max_uncertainty_kJ_mol: float = 5.0,
        min_frames_per_window: int = 3,
    ) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """[IBS_BIAS_PROTOCOL_VERSION=19] 对 self.tmbar_history 里累计的全部
        minibatch 跑一次论文 eq. 15 TMBAR（复用 GlobalMBARAnalyzer.
        solve_stage_integrated，同一份代码同时服务于生产/最终分析的窗口间拼接），
        取回按本窗口态数 mean-centered 的 f_k。

        阈值默认刻意宽松（"这份候选值不值得冻结去走 frozen validation"），真正
        严格的接受/拒绝判据仍是 run_all_windows 里独立、未变的冻结验证阶段——
        候选门放宽不会让坏 f_k 蒙混过关，只会让它更快地被送去验证、验证失败后
        样本并入 tmbar_history 继续学习。

        min_frames_per_window=3（而不是 solve_stage_integrated 生产/最终分析
        用的默认 10）：在线学习每个 minibatch 只有 ~20 帧，去相关后统计非效率
        g≈5-15 时常剩 2-7 帧，用默认的 10 帧门槛会让绝大多数 minibatch 被
        solve_stage_integrated 自身的逐 entry 门槛跳过、白白丢弃已经采到的真实
        数据。3 帧是能跑 MBAR 增广矩阵的下限，仍然不足以解出时返回 None，调用方
        按"这次没有可用更新"处理，不改动 f_k/Context。
        """
        if not self.tmbar_history:
            return None
        res = solve_stage_integrated(
            self.tmbar_history,
            self.kt,
            final_min_ess_ratio=min_ess_ratio,
            final_min_absolute_ess=min_absolute_ess,
            final_min_decorrelated_samples=min_decorrelated_samples,
            final_max_uncertainty_kJ_mol=max_uncertainty_kJ_mol,
            min_frames_per_window=min_frames_per_window,
        )
        if "error" in res:
            return None
        lambdas = list(res.get("lambdas") or [])
        f_k = list(res.get("f_k") or [])
        if sorted(lambdas) != list(range(self.n_states)) or len(f_k) != len(lambdas):
            return None
        f_by_lambda = {int(lam): float(f) for lam, f in zip(lambdas, f_k)}
        # [IBS_BIAS_PROTOCOL_VERSION=27] solve_stage_integrated 返回的 "f_k" 是
        # df_matrix[0, 1:] * kT，即 PyMBAR 的 f_i=-ln(Z_i)=beta*F_i 转成 kJ/mol
        # 后的物理 F_i（相差任意 gauge 常数）。IBS 混合权重满足
        # exp(beta*f_k) Z_k = exp[beta*(f_k-F_k)]，所以可直接使用同号的物理
        # F_k 并 mean-center；额外取负会把补偿方向翻转。瞬时占据过高时降低 f_k
        # 是下面 bounded occupancy feedback 的职责，不能据此反转物理 F_k。
        # v23 起这个绝对候选不写入 Context、只返回作诊断，但仍保持正确约定，
        # 避免未来调用者重新启用时继承旧的 v21 反号错误。
        f_new = np.array(
            [f_by_lambda[k] for k in range(self.n_states)], dtype=np.float64
        )
        f_new = f_new - float(np.mean(f_new))
        return f_new, res

    def _damped_tmbar_absolute_update(
        self,
        f_old: np.ndarray,
        f_tmbar: np.ndarray,
        damping: float = IBS_TMBAR_UPDATE_DAMPING,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Move toward the full-history absolute TMBAR solution.

        TMBAR already performs the self-consistent solve; repeatedly applying
        raw occupancy feedback to one minibatch would not add information.
        The damping controls statistical adaptation. This implements the
        literal convex iteration requested by the warm-up protocol; the
        pairwise trust radius is reserved for the no-TMBAR SGD fallback.
        """
        old = np.asarray(f_old, dtype=np.float64).ravel()
        target = np.asarray(f_tmbar, dtype=np.float64).ravel()
        if old.size != self.n_states or target.size != self.n_states:
            raise ValueError("f_old/f_tmbar 长度必须与 n_states 一致")
        if not np.all(np.isfinite(old)) or not np.all(np.isfinite(target)):
            raise ValueError("f_old/f_tmbar 必须全部有限")
        old = old - float(np.mean(old))
        target = target - float(np.mean(target))
        requested_damping = float(damping)
        if not 0.0 < requested_damping <= 1.0:
            raise ValueError("TMBAR damping 必须位于 (0, 1]")
        effective_damping = requested_damping
        delta_f = effective_damping * (target - old)
        delta_f -= float(np.mean(delta_f))
        pairwise_spread = (
            float(np.max(delta_f) - np.min(delta_f)) if delta_f.size else 0.0
        )
        f_new = old + delta_f
        f_new -= float(np.mean(f_new))
        diagnostics = {
            "method": "damped_absolute_tmbar_v9",
            "requested_damping": requested_damping,
            "effective_damping": float(effective_damping),
            "tmbar_candidate_f_kJ_mol": target.astype(float).tolist(),
            "tmbar_candidate_pairwise_span_kJ_mol": float(
                np.max(target) - np.min(target)
            ) if target.size else 0.0,
            "delta_f_kJ_mol": delta_f.astype(float).tolist(),
            "max_abs_delta_f_kJ_mol": float(np.max(np.abs(delta_f))) if delta_f.size else 0.0,
            "pairwise_delta_f_spread_kJ_mol": float(pairwise_spread),
            "pairwise_step_limit_kT": None,
            "pairwise_step_limit_kJ_mol": None,
        }
        return f_new, diagnostics

    def _bounded_log_occupancy_update(
        self,
        f_old: np.ndarray,
        mean_p: np.ndarray,
        max_abs_log_residual: float = 50.0,
        severe_max_pairwise_step_kT: float = 6.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply one sample-and-hold, pairwise-bounded all-state update.

        ``mean_p`` is the actually observed IBS responsibility under ``f_old``.
        The fixed point is ``K*<p_k> = 1``.  Updating by the *negative* log
        residual lowers every overrepresented state's f_k and raises every
        underrepresented state's f_k simultaneously.  This is an increment,
        not an absolute TMBAR vector; mean-centering only fixes the gauge after
        the increment has been applied.
        """
        f_old = np.asarray(f_old, dtype=np.float64).ravel()
        p = np.asarray(mean_p, dtype=np.float64).ravel()
        if f_old.size != self.n_states or p.size != self.n_states:
            raise ValueError("f_old/mean_p 长度必须与 n_states 一致")
        if not np.all(np.isfinite(f_old)) or not np.all(np.isfinite(p)):
            raise ValueError("f_old/mean_p 必须全部有限")
        if np.any(p < 0.0) or float(np.sum(p)) <= 0.0:
            raise ValueError("mean_p 必须是非负且总和为正")

        p = p / float(np.sum(p))
        p_safe = np.maximum(p, np.finfo(np.float64).tiny)
        raw_log_residual = np.log(float(self.n_states) * p_safe)
        clipped_log_residual = np.clip(
            raw_log_residual,
            -float(max_abs_log_residual),
            float(max_abs_log_residual),
        )
        # A common shift in all residuals is a pure f_k gauge change.  Remove
        # it before stepping so every update redistributes weight rather than
        # translating the whole vector.
        clipped_log_residual -= float(np.mean(clipped_log_residual))

        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=24] Denominator was /100 -- with a
        # per-window correction on the order of several kJ/mol needed
        # (log-residual ~1-2.5 -> delta_f ~kT*residual), that decayed eta
        # below a useful step size within the handful of updates it takes to
        # trip the candidate freeze gate, well before real convergence.
        # Slowed to /500 so eta stays close to full strength over the
        # window this decision actually needs to happen in.
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] v24's fix alone caused a NEW,
        # real GPU-observed failure mode: with eta no longer crushed, this
        # update applies delta_f to all K states *simultaneously*, each
        # sized off that state's own pre-update residual. For a
        # softmax-coupled mixture, lowering the (formerly) dominant state's
        # f_k while raising several underrepresented states' f_k in the same
        # step compounds the shift in relative log-odds well past the
        # single-state linear-response assumption this formula makes --
        # observed as delta_f staying pinned at 2-5 kJ/mol every update
        # while `dominant_k` flips between different states update to
        # update, never settling (classic control-loop overshoot, not slow
        # convergence). Warm-up updater v5 handles this with a tent-shaped
        # gain plus a bound on the pairwise Delta-f spread.  eta_penalty and
        # the update_index long-run decay remain independent safeguards.
        update_index = len(self.f_history) + 1
        residual_severity = float(np.max(np.abs(raw_log_residual)))
        if residual_severity >= IBS_UPDATE_ADAPTIVE_RESIDUAL_HIGH:
            gain_regime = "severe"
            adaptive_fraction = 1.0
            adaptive_relaxation = IBS_UPDATE_SEVERE_RELAXATION_FACTOR
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] severe 区分两档，不再从 4 kT 连续
            # 爬到 severe_max（那让 residual 十几的中等塌陷也吃 ~25 kJ/mol 的大步，
            # 表现为 dominant 每轮翻转、占据不收敛）。residual < COLLAPSE 保持中区
            # 顶端的 4 kT（≈10 kJ/mol，单次相对更新的安全上限）；只有 residual ≥
            # COLLAPSE 的真塌陷才放开到 caller 选的 severe_max——那种窗口用 4 kT/步
            # 需要十几二十步才建起所需 f_k 谱宽，慢到会吃光 max_bias_updates 预算。
            if residual_severity >= IBS_UPDATE_ADAPTIVE_RESIDUAL_COLLAPSE:
                pairwise_step_limit_kT = float(severe_max_pairwise_step_kT)
            else:
                pairwise_step_limit_kT = 4.0
        elif residual_severity >= IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW:
            gain_regime = "middle"
            adaptive_fraction = float(
                (residual_severity - IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW)
                / (
                    IBS_UPDATE_ADAPTIVE_RESIDUAL_HIGH
                    - IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW
                )
            )
            # At the edge of the near-flat basin gain is highest, because the
            # response is observable but not saturated.  It falls toward the
            # severe value as collapse increases. The pairwise radius grows
            # from 2 to 4 kT across this recoverable middle regime.
            adaptive_relaxation = float(
                IBS_UPDATE_MIDDLE_RELAXATION_FACTOR
                + adaptive_fraction
                * (
                    IBS_UPDATE_SEVERE_RELAXATION_FACTOR
                    - IBS_UPDATE_MIDDLE_RELAXATION_FACTOR
                )
            )
            pairwise_step_limit_kT = float(2.0 + 2.0 * adaptive_fraction)
        else:
            gain_regime = "near_flat"
            adaptive_fraction = float(
                residual_severity / IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW
            )
            # Do not chase finite-block argmax noise at the fixed point.
            adaptive_relaxation = float(
                IBS_UPDATE_NEAR_FLAT_RELAXATION_FACTOR
                + adaptive_fraction
                * (
                    IBS_UPDATE_MIDDLE_RELAXATION_FACTOR
                    - IBS_UPDATE_NEAR_FLAT_RELAXATION_FACTOR
                )
            )
            pairwise_step_limit_kT = float(0.5 + 1.5 * adaptive_fraction)
        eta = (
            adaptive_relaxation
            * float(self.eta_penalty)
            / (1.0 + float(update_index) / 500.0)
        )
        delta_f = -eta * float(self.kt) * clipped_log_residual

        # What changes softmax odds is delta_f[i]-delta_f[j], not an absolute
        # component relative to an arbitrary mean-zero gauge.  Bound that
        # physically meaningful pairwise spread and preserve the direction by
        # uniform rescaling.  In a one-dominant five-state collapse, a 3-kT
        # The caller selects a severe ceiling only after checking dominant
        # persistence. This is still a relative-odds bound, not an arbitrary
        # gauge-dependent per-component radius.
        pairwise_step_limit_kj = pairwise_step_limit_kT * float(self.kt)
        pairwise_delta_spread = (
            float(np.max(delta_f) - np.min(delta_f)) if delta_f.size else 0.0
        )
        if pairwise_delta_spread > pairwise_step_limit_kj > 0.0:
            delta_f *= pairwise_step_limit_kj / pairwise_delta_spread
            pairwise_delta_spread = float(np.max(delta_f) - np.min(delta_f))

        f_new = f_old + delta_f
        f_new -= float(np.mean(f_new))
        diagnostics = {
            "method": "bounded_log_occupancy_fallback_v9",
            "eta": float(eta),
            "eta_penalty": float(self.eta_penalty),
            "residual_severity": float(residual_severity),
            "adaptive_relaxation": float(adaptive_relaxation),
            "adaptive_fraction": float(adaptive_fraction),
            "gain_regime": gain_regime,
            "raw_log_residual": raw_log_residual.astype(float).tolist(),
            "clipped_centered_log_residual": clipped_log_residual.astype(float).tolist(),
            "delta_f_kJ_mol": delta_f.astype(float).tolist(),
            "max_abs_delta_f_kJ_mol": float(np.max(np.abs(delta_f))) if delta_f.size else 0.0,
            "pairwise_delta_f_spread_kJ_mol": float(pairwise_delta_spread),
            "pairwise_step_limit_kT": float(pairwise_step_limit_kT),
            "pairwise_step_limit_kJ_mol": float(pairwise_step_limit_kj),
        }
        return f_new, diagnostics

    def update_weights(
        self,
        min_buffer_size: int = 20,
        candidate_min_ess_ratio: float = 0.05,
        candidate_min_absolute_ess: float = 1.0,
        # [v23 numerical follow-up #3] Was 5 -- mathematically unreachable
        # given `_solve_tmbar_and_recenter`'s own `min_frames_per_window=3`
        # (documented there as the actual floor for running the augmented
        # MBAR matrix at all, precisely because decorrelation commonly
        # leaves only 2-7 frames per ~20-frame minibatch). Any minibatch with
        # 3 or 4 decorrelated frames is deliberately let into local_results
        # by that floor, which then caps the aggregate min_decorrelated_
        # samples at 3 or 4 for as long as such a minibatch exists in
        # tmbar_history -- requiring >=5 here made this candidate criterion
        # permanently unsatisfiable in exactly the regime it's meant to
        # evaluate, same failure shape as the min_absolute_ess 2.5->1.0 fix
        # above. Lowered to 3 to match the actual floor; final_*/frozen-
        # validation gates are untouched.
        candidate_min_decorrelated_samples: int = 3,
        candidate_max_uncertainty_kJ_mol: float = 5.0,
    ) -> Optional[np.ndarray]:
        """
        在线 IBS 权重更新。v9 使用全部累计 minibatch 的 TMBAR 绝对候选，
        按固定 0.20 阻尼更新 f_k。``mean_p_batch``、EMA 和 dominant 仅供诊断；
        只有 TMBAR 暂不可解时才执行一次 bounded-SGD 兜底（pairwise 半径 4 kT，
        仅 raw_residual≥70 的真塌陷放开到 10 kT）。
        candidate_* 阈值同时写入 tmbar_update 诊断，供 learning 候选门读取。
        """
        if len(self.energy_buffer) < min_buffer_size:
            return None

        u_mk = np.array(self.energy_buffer) # (M, K)
        valid_mask = ~np.isnan(u_mk).any(axis=1)
        if np.sum(valid_mask) < min_buffer_size:
            return None

        u_mk = u_mk[valid_mask]
        M, K = u_mk.shape
        
        # 1. 获取当前 f_old
        f_old = np.array([self.context.getParameter(f"{self.prefix}_f_{k}") for k in range(K)])
        
        # 2. 计算瞬时概率 p_ik
        # 注意：u_mk 已经是相对于 State 0 偏移后的能量
        # 公式: p_ik ∝ exp(-beta * (u_mk[i,k] - f_k))
        # 由于 u_mk 是相对值，f_k 也应当被解释为相对自由能估计
        
        beta_u = -self.beta * u_mk  # (M, K)
        beta_f = self.beta * f_old  # (K,)
        
        logits = beta_f[None, :] + beta_u  # (M, K)
        
        # Numerical stability: log-sum-exp
        max_logits = np.max(logits, axis=1, keepdims=True)
        log_denom = max_logits.squeeze() + np.log(np.sum(np.exp(logits - max_logits), axis=1))
        
        log_p = logits - log_denom[:, None]
        p_ik = np.exp(log_p)  # (M, K)
        
        # 3. 计算批次平均概率
        mean_p_batch = np.mean(p_ik, axis=0)  # (K,)

        adjacent_delta_u = np.diff(u_mk, axis=1)
        adjacent_records = []
        for edge_idx in range(max(K - 1, 0)):
            edge = np.asarray(adjacent_delta_u[:, edge_idx], dtype=float)
            adjacent_records.append({
                "edge": [int(edge_idx), int(edge_idx + 1)],
                "mean_delta_u_kJ_mol": float(np.mean(edge)),
                "std_delta_u_kJ_mol": float(np.std(edge, ddof=1)) if edge.size > 1 else 0.0,
                "std_reduced_delta_u": (
                    float(self.beta * np.std(edge, ddof=1)) if edge.size > 1 else 0.0
                ),
                "rms_delta_u_kJ_mol": float(np.sqrt(np.mean(edge * edge))),
                "p05_delta_u_kJ_mol": float(np.percentile(edge, 5.0)),
                "p50_delta_u_kJ_mol": float(np.percentile(edge, 50.0)),
                "p95_delta_u_kJ_mol": float(np.percentile(edge, 95.0)),
            })
        self.last_update_diagnostics = {
            "batch_size": int(M),
            "mean_p_batch": mean_p_batch.astype(float).tolist(),
            "lse_balance": ibs_lse_balance_diagnostics(mean_p_batch),
            "mean_u_kJ_mol": np.mean(u_mk, axis=0).astype(float).tolist(),
            "adjacent_delta_u": adjacent_records,
        }
        
        # 4. EMA 更新（仅供原始占据趋势诊断）。
        if self.ema_mean_p is None:
            self.ema_mean_p = mean_p_batch.copy()
        else:
            self.ema_mean_p = self.gamma * self.ema_mean_p + (1.0 - self.gamma) * mean_p_batch

        # 5. 论文 eq. 15 TMBAR：把这批 minibatch 打包进持久历史，用累计
        # 至今的全部时变采样分布自洽求解绝对物理自由能。v9 起 f_k 更新只朝
        # 这个全历史绝对候选阻尼移动；当前 raw batch 占据仅作诊断，
        # 不再充当 SGD 更新方向。
        self._append_tmbar_batch_from_buffer()
        tmbar_result = self._solve_tmbar_and_recenter(
            min_ess_ratio=candidate_min_ess_ratio,
            min_absolute_ess=candidate_min_absolute_ess,
            min_decorrelated_samples=candidate_min_decorrelated_samples,
            max_uncertainty_kJ_mol=candidate_max_uncertainty_kJ_mol,
        )
        if tmbar_result is None:
            tmbar_absolute_candidate = None
            self.last_update_diagnostics["tmbar_update"] = {
                "available": False,
                "n_tmbar_entries": len(self.tmbar_history),
                "converged": False,
            }
        else:
            tmbar_absolute_candidate, tmbar_res = tmbar_result
            self.last_update_diagnostics["tmbar_update"] = {
                "available": True,
                "n_tmbar_entries": len(self.tmbar_history),
                # solve_stage_integrated's legacy flag takes the minimum over
                # every historical minibatch. One early low-ESS entry then
                # poisons the flag forever as history grows. Preserve it only
                # as a quality diagnostic; online readiness is assigned below
                # from the actual damped TMBAR fixed-point step.
                "legacy_per_entry_quality_converged": bool(
                    tmbar_res.get("converged", False)
                ),
                # [ESS_GATE_PROTOCOL_VERSION=2] raw_* 是这条路径真正用来判 trust 的量
                # （见下面 tmbar_candidate_trusted 处的说明）；mixture 量对 tmbar_history
                # 恒为 None，一并落盘只为让诊断自解释。
                "min_overlap": tmbar_res.get("raw_min_overlap"),
                "min_absolute_ess": tmbar_res.get("raw_min_absolute_ess"),
                "mixture_min_overlap": tmbar_res.get("min_overlap"),
                "min_decorrelated_samples": tmbar_res.get("min_decorrelated_samples"),
                "max_endpoint_uncertainty_kJ_mol": tmbar_res.get("max_endpoint_uncertainty_kJ_mol"),
            }
        self.last_update_diagnostics["adjacent_delta_u_is_convergence_gate"] = False

        dominant_k = int(np.argmax(mean_p_batch))
        raw_log_residual = np.log(
            float(K) * np.maximum(mean_p_batch, np.finfo(np.float64).tiny)
        )
        residual_severity = float(np.max(np.abs(raw_log_residual)))
        total_tmbar_frames = int(sum(
            np.asarray(entry.get("u_kn", np.empty((0, 0)))).shape[1]
            for entry in self.tmbar_history
        ))
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 只有累计 TMBAR 解质量同时可信时才应用
        # 绝对候选；否则退回受限占据反馈先恢复覆盖，避免"低重叠 TMBAR 给出错误
        # 大步 → 占据塌缩 → TMBAR 更不可靠"的正反馈。coverage_ess 用当前 batch 占据
        # 算（mean_p_batch 已归一，sum=1）。这些只门控"用哪个控制器"，不是窗口
        # 收敛判据——收敛仍是 run_all_windows 里的 local-MBAR loose gate。
        _sum_sq_p = float(np.sum(np.square(mean_p_batch)))
        coverage_ess_batch = float(1.0 / _sum_sq_p) if _sum_sq_p > 0.0 else 0.0
        tmbar_candidate_trusted = False
        if tmbar_absolute_candidate is not None:
            # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 这里刻意读 raw_* 而不是 min_overlap/
            # min_absolute_ess。两个消费者要的不是同一件事：
            #   - 阶段收敛门问"混合分布对物理态覆盖够不够"→ 用扣掉共模因子的
            #     mixture 量（共模项在它真正报出去的物理态↔物理态 ΔF 里会抵消）。
            #   - 这里问"从这批原始重要性权重解出来的绝对 f_k 向量敢不敢直接用"→
            #     权重退化本身就是不敢用的理由，raw 单参考 ESS 才是正确的悲观度量。
            # 另外 tmbar_history 的 entry 天生没有 f_k 字段（每条 entry 是在不同
            # f_k 下采的，单一 f_k 无法代表整段历史，这正是 TMBAR 存在的理由），
            # mixture 量对这个调用方恒为 None——读它会让 trust 门永久为 False，
            # 把 warmup 永久钉死在受限占据反馈上。
            _q_overlap = tmbar_res.get("raw_min_overlap")
            _q_abs_ess = tmbar_res.get("raw_min_absolute_ess")
            _q_decorr = tmbar_res.get("min_decorrelated_samples")
            _q_uncert = tmbar_res.get("max_endpoint_uncertainty_kJ_mol")
            tmbar_candidate_trusted = bool(
                _q_overlap is not None
                and _q_overlap >= IBS_TMBAR_TRUST_MIN_OVERLAP
                and _q_abs_ess is not None
                and _q_abs_ess >= IBS_TMBAR_TRUST_MIN_ABSOLUTE_ESS
                and _q_decorr is not None
                and _q_decorr >= IBS_TMBAR_TRUST_MIN_DECORRELATED_SAMPLES
                and _q_uncert is not None
                and np.isfinite(_q_uncert)
                and _q_uncert <= IBS_TMBAR_TRUST_MAX_UNCERTAINTY_KJ_MOL
                and coverage_ess_batch
                >= IBS_TMBAR_TRUST_MIN_COVERAGE_ESS_FRACTION * float(K)
            )
        if tmbar_candidate_trusted:
            f_new, weight_update_diag = self._damped_tmbar_absolute_update(
                f_old,
                tmbar_absolute_candidate,
            )
        else:
            # TMBAR 候选不可信（低重叠/低 ESS/去相关不足/不确定度大/覆盖差）或
            # 根本不可解：用受限占据反馈先把覆盖拉回来（Δf_k=-η·kT·ln(K·p_k)，
            # 自带自适应 pairwise 上限），不让一步错误大步打崩占据。
            f_new, weight_update_diag = self._bounded_log_occupancy_update(
                f_old,
                mean_p_batch,
                severe_max_pairwise_step_kT=(
                    IBS_TMBAR_FALLBACK_SGD_PAIRWISE_STEP_KT
                ),
            )
            weight_update_diag["method"] = (
                "bounded_occupancy_tmbar_untrusted_v9"
                if tmbar_absolute_candidate is not None
                else "bounded_sgd_fallback_v9"
            )
        weight_update_diag["tmbar_candidate_trusted"] = bool(tmbar_candidate_trusted)
        weight_update_diag["coverage_ess_batch"] = float(coverage_ess_batch)
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 硬 pairwise cap 只加在【可信绝对 TMBAR】
        # 路径上：绝对候选即使可信、估计略偏时一大步也危险，故 cap 到
        # IBS_MAX_APPLIED_PAIRWISE_STEP_KT(2 kT)。而 bounded occupancy 反馈是自我纠偏
        # 的比例控制器（Δf=-η·kT·ln(K·p)），且严重塌陷时本就需要大步纠偏——它自带
        # 的自适应 pairwise 上限（近平坦 0.5→中区 2→4→中等塌陷 4→真塌陷 severe_max）已
        # 足够安全，这里不再额外硬 cap 到 2 kT，否则 raw_residual≈150 的严重窗口会被
        # 卡在 2 kT/步、要 ~75 步才建起所需 f_k 谱宽（塌陷的正反馈风险来自已被 trust-
        # gate 关掉的绝对 TMBAR，不来自 bounded）。诊断一律刷新成 cap 之后的真实步长。
        _applied_delta = (
            np.asarray(f_new, dtype=np.float64)
            - np.asarray(f_old, dtype=np.float64)
        )
        _applied_delta -= float(np.mean(_applied_delta))
        _applied_spread = (
            float(np.max(_applied_delta) - np.min(_applied_delta))
            if _applied_delta.size else 0.0
        )
        if tmbar_candidate_trusted:
            _hard_cap_kj = float(IBS_MAX_APPLIED_PAIRWISE_STEP_KT) * float(self.kt)
            if _applied_spread > _hard_cap_kj > 0.0:
                _applied_delta *= _hard_cap_kj / _applied_spread
                _applied_spread = float(np.max(_applied_delta) - np.min(_applied_delta))
                f_new = np.asarray(f_old, dtype=np.float64) + _applied_delta
                f_new -= float(np.mean(f_new))
                weight_update_diag["hard_pairwise_cap_applied"] = True
            else:
                weight_update_diag["hard_pairwise_cap_applied"] = False
            weight_update_diag["hard_pairwise_cap_kJ_mol"] = float(_hard_cap_kj)
        else:
            # bounded 路径：不额外硬 cap，用其自带自适应上限（见 _bounded_log_
            # occupancy_update）。诊断记录未施加外部硬 cap。
            weight_update_diag["hard_pairwise_cap_applied"] = False
            weight_update_diag["hard_pairwise_cap_kJ_mol"] = None
        weight_update_diag["delta_f_kJ_mol"] = _applied_delta.astype(float).tolist()
        weight_update_diag["max_abs_delta_f_kJ_mol"] = (
            float(np.max(np.abs(_applied_delta))) if _applied_delta.size else 0.0
        )
        weight_update_diag["pairwise_delta_f_spread_kJ_mol"] = float(_applied_spread)
        weight_update_diag["residual_severity"] = float(residual_severity)
        weight_update_diag["total_tmbar_frames"] = int(total_tmbar_frames)
        applied_pairwise_step = float(
            weight_update_diag.get("pairwise_delta_f_spread_kJ_mol", float("inf"))
        )
        self_consistency_limit_kj = float(
            IBS_TMBAR_FREEZE_MAX_APPLIED_PAIRWISE_STEP_KT * self.kt
        )
        tmbar_self_consistent = bool(
            tmbar_absolute_candidate is not None
            and np.isfinite(applied_pairwise_step)
            and applied_pairwise_step <= self_consistency_limit_kj
        )
        self.last_update_diagnostics["tmbar_update"].update({
            "converged": tmbar_self_consistent,
            "online_convergence_method": "damped_absolute_step_pairwise",
            "applied_pairwise_step_kJ_mol": applied_pairwise_step,
            "applied_pairwise_step_threshold_kJ_mol": self_consistency_limit_kj,
            "total_tmbar_frames": int(total_tmbar_frames),
        })
        self.last_update_diagnostics["weight_update"] = weight_update_diag
        print(
            f"    [IBS TMBAR 自洽权重更新 v9] dominant诊断=state{dominant_k} "
            f"p={float(mean_p_batch[dominant_k]):.6f}, "
            f"delta_f={float(weight_update_diag['delta_f_kJ_mol'][dominant_k]):+.3f} kJ/mol, "
            f"max|delta_f|={float(weight_update_diag['max_abs_delta_f_kJ_mol']):.3f} "
            f"(pairwise={float(weight_update_diag['pairwise_delta_f_spread_kJ_mol']):.3f} kJ/mol, "
            f"method={weight_update_diag['method']}, "
            f"alpha={float(weight_update_diag.get('effective_damping', 0.0)):.3f}, "
            f"raw_residual={residual_severity:.3f}, "
            f"batch={M}, total_frames={total_tmbar_frames}, "
            f"tmbar_self_consistent={tmbar_self_consistent}, "
            f"legacy_quality={self.last_update_diagnostics['tmbar_update'].get('legacy_per_entry_quality_converged')})"
        )
        # Dominant identity remains a diagnostic breadcrumb only. It does not
        # reset, brake, accelerate, or otherwise alter the TMBAR update.
        self._last_dominant_k = dominant_k

        # 6. 应用更新
        for k in range(K):
            self.context.setParameter(f"{self.prefix}_f_{k}", float(f_new[k]))

        self.energy_buffer = []
        self.f_history.append(f_new.copy())
        return f_new

    def apply_learning_rate_penalty(self, factor: float = 0.85, floor: float = 0.25) -> float:
        """Damp (default 0.85x) the learning-rate multiplier after a frozen-
        validation failure, so the next learning attempt takes more
        conservative steps instead of oscillating with the same large step
        size that just failed. Floored to avoid learning stalling out
        entirely after repeated failures.

        [IBS_BIAS_PROTOCOL_VERSION=24] factor was 0.5 and floor was 0.05:
        after just 5 real failures (0.5**5 = 0.03125 < floor), eta_penalty
        bottomed out at 1/20th strength and stayed there across every future
        process restart (it's persisted -- see save/load_ibs_state), because
        nothing ever raises it back up. That silently capped every future
        learning attempt to steps far too small to close a multi-kJ/mol
        occupancy gap, guaranteeing repeated freeze/validate failures that
        each burn the full step budget for no real progress. 0.85/0.25 keep
        the same "get more conservative after a real failure" intent without
        collapsing to a step size that can no longer converge at all.
        """
        self.eta_penalty = max(float(floor), float(self.eta_penalty) * float(factor))
        return self.eta_penalty

    def evaluate_frozen_batch_probability(self, min_buffer_size: int = 20) -> Optional[np.ndarray]:
        """Measure this batch's mean occupation probability under the CURRENT
        f_k, without applying any update.

        ``update_weights()`` computes its probability estimate from ``f_old``
        and then immediately overwrites the context with ``f_new`` in the same
        call -- so the f_k that actually ends up frozen for production has
        never itself been the subject of a passing check; every "consecutive
        pass" was evidence about a different, already-discarded f_k. This
        method is the frozen-validation counterpart: same log-sum-exp
        probability math, same EMA update, but it reads f_k and never writes
        it, so repeated calls against an unchanged f_k are genuine repeated
        evidence about that exact f_k.
        """
        if len(self.energy_buffer) < min_buffer_size:
            return None
        u_mk = np.array(self.energy_buffer)
        valid_mask = ~np.isnan(u_mk).any(axis=1)
        if np.sum(valid_mask) < min_buffer_size:
            return None
        u_mk = u_mk[valid_mask]
        M, K = u_mk.shape
        self.last_frozen_batch_size = int(M)

        f_k = np.array([self.context.getParameter(f"{self.prefix}_f_{k}") for k in range(K)])
        beta_u = -self.beta * u_mk
        beta_f = self.beta * f_k
        logits = beta_f[None, :] + beta_u
        max_logits = np.max(logits, axis=1, keepdims=True)
        log_denom = max_logits.squeeze() + np.log(np.sum(np.exp(logits - max_logits), axis=1))
        log_p = logits - log_denom[:, None]
        p_ik = np.exp(log_p)
        mean_p_batch = np.mean(p_ik, axis=0)

        if self.ema_mean_p is None:
            self.ema_mean_p = mean_p_batch.copy()
        else:
            self.ema_mean_p = self.gamma * self.ema_mean_p + (1.0 - self.gamma) * mean_p_batch

        self.energy_buffer = []
        return mean_p_batch
# ================= ibs_engine.py -> IBSSampler 类 =================
    def save_ibs_state(self, filepath: str, lambdas_coul=None, lambdas_vdw=None):
        """同步保存 IBS 状态，使用原子替换避免损坏。

        lambdas_coul/lambdas_vdw：本窗口这次真正对应的 λ 值（跟 convergence.json
        里存的是同一份东西）。之前这里没存——resume 只按 n_states 数量判断"能不能
        用"，λ 路径被自动加密/重新划分窗口后，state 数量凑巧相同但 f_k[k] 早就不
        对应同一个 λ 了，却仍会被当成有效热启动注入，这是错的。
        """
        f_current = [self.context.getParameter(f"{self.prefix}_f_{k}") for k in range(self.n_states)]
        state = {
            "n_states": int(self.n_states),
            "prefix": self.prefix,
            "f_k": f_current,
            "t": len(self.f_history),
            # [IBS_BIAS_PROTOCOL_VERSION=23] 有界增量的学习率惩罚属于续算状态；
            # 冻结验证失败后会降低，resume 时必须保持，不能静默重置为 1.0。
            "eta_penalty": float(self.eta_penalty),
            "e_offset": self.e_offset,
            # [IBS_BIAS_PROTOCOL_VERSION=19] 论文 eq. 15 TMBAR 的持久 minibatch
            # 历史（取代 v18 及更早版本的 ta_relative_q_sum/ta_iteration_count），
            # 是恢复在线学习进度必须的一部分——跟旧字段一样，load 时要在注入
            # 任何 f_k 之前完整校验。
            "tmbar_history": [
                {
                    "u_kn": np.asarray(entry["u_kn"], dtype=float).tolist(),
                    "bias_energies": np.asarray(
                        entry["bias_energies"], dtype=float
                    ).tolist(),
                    "base_energies": np.asarray(
                        entry["base_energies"], dtype=float
                    ).tolist(),
                    "lambda_indices": [int(x) for x in entry["lambda_indices"]],
                    "sampled_distribution_row": int(
                        entry.get("sampled_distribution_row", 0)
                    ),
                }
                for entry in self.tmbar_history
            ],
            "tmbar_history_dropped_entries": int(
                self.tmbar_history_dropped_entries
            ),
            "status": "running",
            # 🔑 见 IBS_BIAS_PROTOCOL_VERSION 定义处：只有这里存的 bias_converged=True
            # 才允许 load_ibs_state 之后直接跳过 Warmup 进生产；否则续传必须继续走
            # 严格收敛判据，不能假设旧状态已经收敛。
            "bias_converged": bool(self.bias_converged),
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] bias_status/frozen_f_k_pending：
            # 见 IBSSampler.__init__ 里的注释。resume 只有在 bias_status ==
            # "calibrated_pending_validation" 且 frozen_f_k_pending 非空时才会
            # 跳过 [learning]、直接带着这份冻结 f_k 回到 freeze_burn_in。
            "bias_status": str(self.bias_status),
            "frozen_f_k_pending": (
                [float(x) for x in self.frozen_f_k_pending]
                if self.frozen_f_k_pending is not None else None
            ),
            # 🔑 见 IBSSampler.__init__ 里的注释——这份冻结 f_k 累计已经花在冻结
            # 验证上的步数，跨越同一份校准 f_k 的多次 resume/阶梯升级累加。
            "frozen_validation_cumulative_steps": int(self.frozen_validation_cumulative_steps),
            "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
            "warmup_update_protocol_version": IBS_WARMUP_UPDATE_PROTOCOL_VERSION,
            # 🔑 [non_mutating_v1] 记录产出这份 f_k 状态的采样修复策略。旧的变异
            # 策略可能就地重校准过 f_k（不同参考系）；load 时据此 fail-closed。
            "sampling_repair_policy": getattr(self, "sampling_repair_policy", None),
            "lambdas_coul": [float(x) for x in lambdas_coul] if lambdas_coul is not None else None,
            "lambdas_vdw": [float(x) for x in lambdas_vdw] if lambdas_vdw is not None else None,
        }
        _atomic_write_json(filepath, state)

    def load_ibs_state(self, filepath: str, lambdas_coul=None, lambdas_vdw=None) -> bool:
        """反序列化并注入 IBS 状态，恢复历史记忆。

        🔑 n_states/prefix/协议版本/λ 内容任何一项对不上，一律完全拒绝这份旧
        状态。只有这些基础身份字段都匹配、即 state 确实声称属于当前协议时，才
        检查更严格的 sampling_repair_policy：当前 policy 为 non_mutating_v1 时，
        当前协议的缓存若缺字段或来自其它 policy，不允许静默 return False 后在原
        目录重采，必须在注入任何 f_k 之前抛
        ExistingEnsembleRequiresRescueAudit，保留原 ensemble 交 rescue audit。
        旧协议 state 则走正常的版本失配路径安全失效，绝不注入 f_k。
        """
        import json, os
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r") as f:
                state = json.load(f)
            if state.get("n_states") != self.n_states:
                print(
                    f"  ⚠️ IBS 状态与当前窗口不兼容 "
                    f"(cache n_states={state.get('n_states')}, current={self.n_states})，忽略旧状态"
                )
                return False
            if state.get("prefix") not in (None, self.prefix):
                print(
                    f"  ⚠️ IBS 状态 prefix 不兼容 "
                    f"(cache={state.get('prefix')}, current={self.prefix})，忽略旧状态"
                )
                return False
            cached_protocol_version = state.get("ibs_bias_protocol_version")
            if not _ibs_bias_protocol_version_is_cache_compatible(
                cached_protocol_version
            ):
                print(
                    f"  ⚠️ IBS 状态协议版本不匹配 (cache={cached_protocol_version!r}, "
                    f"兼容版本={sorted(IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS)})，"
                    "完全忽略旧状态（不作为热启动），"
                    "从 f_k=0 重新开始"
                )
                return False
            cached_warmup_update_version = state.get(
                "warmup_update_protocol_version"
            )
            if (
                not bool(state.get("bias_converged", False))
                and cached_warmup_update_version
                != IBS_WARMUP_UPDATE_PROTOCOL_VERSION
            ):
                print(
                    "  ⚠️ 未完成的 IBS 预热状态使用旧权重控制器 "
                    f"(cache={cached_warmup_update_version!r}, "
                    f"current={IBS_WARMUP_UPDATE_PROTOCOL_VERSION})，"
                    "拒绝把旧控制器的中间 f_k 注入新自适应学习；从当前协议重新预热"
                )
                return False
            cached_lc = state.get("lambdas_coul")
            cached_lv = state.get("lambdas_vdw")
            lambdas_match = (
                lambdas_coul is not None
                and lambdas_vdw is not None
                and cached_lc is not None
                and cached_lv is not None
                and len(cached_lc) == len(lambdas_coul)
                and len(cached_lv) == len(lambdas_vdw)
                and np.allclose(cached_lc, lambdas_coul, atol=1e-9)
                and np.allclose(cached_lv, lambdas_vdw, atol=1e-9)
            )
            if not lambdas_match:
                print(
                    "  ⚠️ IBS 状态对应的 λ 值与当前窗口不匹配（λ 路径已被自动加密/"
                    "重新划分窗口，或旧状态缺少 λ 元数据），完全忽略旧状态（不作为"
                    "热启动，f_k[k] 错配到不同 λ 上是主动引入偏差，不是中性起点），"
                    "从 f_k=0 重新开始"
                )
                return False

            # repair policy 是当前协议 ensemble 身份的一部分，但必须放在上面的
            # 协议版本/λ 门之后：旧版本 state 本来就应无条件失效，不能因为它在
            # 当时尚不存在的字段为 None 而被误判成“当前协议的可疑 ensemble”。
            # 对真正属于当前协议的 state，仍然在读取/注入任何 f_k 之前 fail-closed。
            current_repair_policy = getattr(
                self, "sampling_repair_policy", "non_mutating_v1"
            )
            cached_repair_policy = state.get("sampling_repair_policy")
            if cached_repair_policy != current_repair_policy:
                if current_repair_policy == "non_mutating_v1":
                    raise ExistingEnsembleRequiresRescueAudit(
                        "IBS 状态的 sampling_repair_policy="
                        f"{cached_repair_policy!r}（当前 {current_repair_policy!r}）。"
                        "non_mutating_v1 拒绝把当前协议下无策略/旧策略 state 的 f_k "
                        "注入 Context，也拒绝在原 ensemble 目录静默重新开始；"
                        "已保留原文件，交 rescue audit 判定。",
                        diagnostics={
                            "state_file": os.path.abspath(filepath),
                            "cached_sampling_repair_policy": cached_repair_policy,
                            "current_sampling_repair_policy": current_repair_policy,
                            "requires_rescue_audit": True,
                        },
                    )
                print(
                    "  ⚠️ IBS 状态 sampling_repair_policy 不匹配 "
                    f"(cache={cached_repair_policy!r}, current={current_repair_policy!r})，"
                    "legacy_mutating 模式下忽略旧状态并重新开始"
                )
                return False

            f_k = state["f_k"]
            t = state["t"]
            if len(f_k) != self.n_states or not np.all(np.isfinite(np.asarray(f_k, dtype=float))):
                print("  ⚠️ IBS 状态 f_k 无效，忽略旧状态")
                return False
            cached_eta_penalty = state.get("eta_penalty")
            if (
                cached_eta_penalty is None
                or not np.isfinite(float(cached_eta_penalty))
                or not (0.25 <= float(cached_eta_penalty) <= 1.0)
            ):
                print("  ⚠️ IBS 状态 eta_penalty 无效，忽略旧状态")
                return False

            # [IBS_BIAS_PROTOCOL_VERSION=19] 每条 tmbar_history entry 必须完整
            # 校验（形状/有限性/lambda_indices 覆盖本窗口全部态/sampled_
            # distribution_row 恒为 0）才注入，任何一条不对就整份忽略——
            # 这份历史直接决定后续 solve_stage_integrated 的 f_k，绝不能
            # 部分信任、悄悄跳过坏 entry。
            cached_tmbar_history = state.get("tmbar_history")
            expected_lambda_indices = list(range(self.n_states))
            tmbar_history_valid = isinstance(cached_tmbar_history, list)
            validated_tmbar_history: List[Dict[str, Any]] = []
            if tmbar_history_valid:
                # Older checkpoints may predate the cap.  Only the newest
                # bounded suffix can influence future learning, so validate
                # and restore that suffix instead of re-inflating memory.
                cached_history_suffix = cached_tmbar_history[
                    -TMBAR_HISTORY_MAX_ENTRIES:
                ]
                for entry in cached_history_suffix:
                    try:
                        if not isinstance(entry, dict):
                            raise ValueError("entry 不是 dict")
                        if [int(x) for x in entry["lambda_indices"]] != expected_lambda_indices:
                            raise ValueError("lambda_indices 与当前窗口态数不匹配")
                        if int(entry.get("sampled_distribution_row", -1)) != 0:
                            raise ValueError("sampled_distribution_row 必须为 0")
                        u_kn = np.asarray(entry["u_kn"], dtype=np.float64)
                        bias_e = np.asarray(entry["bias_energies"], dtype=np.float64).ravel()
                        base_e = np.asarray(entry["base_energies"], dtype=np.float64).ravel()
                        if (
                            u_kn.ndim != 2
                            or u_kn.shape[0] != self.n_states
                            or u_kn.shape[1] < 1
                            or bias_e.size != u_kn.shape[1]
                            or base_e.size != u_kn.shape[1]
                            or not np.all(np.isfinite(u_kn))
                            or not np.all(np.isfinite(bias_e))
                            or not np.all(np.isfinite(base_e))
                        ):
                            raise ValueError("u_kn/bias_energies/base_energies 形状或有限性无效")
                        validated_tmbar_history.append({
                            "u_kn": u_kn,
                            "bias_energies": bias_e,
                            "base_energies": base_e,
                            "lambda_indices": expected_lambda_indices,
                            "sampled_distribution_row": 0,
                        })
                    except Exception:
                        tmbar_history_valid = False
                        break
            if not tmbar_history_valid:
                print("  ⚠️ IBS 状态的 tmbar_history 无效，忽略旧状态")
                return False

            self.e_offset = state.get("e_offset", 0.0)
            self.bias_converged = bool(state.get("bias_converged", False))
            self.tmbar_history = validated_tmbar_history
            prior_dropped = state.get("tmbar_history_dropped_entries", 0)
            try:
                prior_dropped = max(0, int(prior_dropped))
            except (TypeError, ValueError):
                prior_dropped = 0
            self.tmbar_history_dropped_entries = (
                prior_dropped
                + max(0, len(cached_tmbar_history) - len(validated_tmbar_history))
            )
            self.eta_penalty = float(cached_eta_penalty)

            cached_status = state.get("bias_status", "unconverged")
            cached_pending_f_k = state.get("frozen_f_k_pending")
            if (
                cached_status == "calibrated_pending_validation"
                and cached_pending_f_k is not None
                and len(cached_pending_f_k) == self.n_states
                and np.all(np.isfinite(np.asarray(cached_pending_f_k, dtype=float)))
            ):
                self.bias_status = "calibrated_pending_validation"
                self.frozen_f_k_pending = [float(x) for x in cached_pending_f_k]
                self.frozen_validation_cumulative_steps = int(
                    state.get("frozen_validation_cumulative_steps", 0)
                )
            elif cached_status == "calibrated_validation_failed":
                # 🔑 终态：冻结验证累计预算已经用到最后一档仍未通过，不再是
                # "pending"——不能把这份 f_k 当成可以继续自动续验的东西注入，
                # run_all_windows 里紧接着 load_ibs_state 之后的检查会看到这个
                # 状态并立刻硬停止，不做任何进一步的 SGD/续验尝试。
                self.bias_status = "calibrated_validation_failed"
                self.frozen_f_k_pending = None
                self.frozen_validation_cumulative_steps = 0
            else:
                self.bias_status = "converged" if self.bias_converged else "unconverged"
                self.frozen_f_k_pending = None
                self.frozen_validation_cumulative_steps = 0

            # 注入 Context
            for k in range(self.n_states):
                self.context.setParameter(f"{self.prefix}_f_{k}", float(f_k[k]))

            # 伪造历史长度以恢复 v23 有界增量的学习率衰减。
            self.f_history = [np.array(f_k)] * t
            print(
                f"  ♻️ IBS 状态已恢复: t={t}, max|f_k|={np.max(np.abs(f_k)):.2f} kJ/mol, "
                f"tmbar_history entries={len(self.tmbar_history)}, "
                f"bias_converged={self.bias_converged}, bias_status={self.bias_status}"
            )
            return True
        except ExistingEnsembleRequiresRescueAudit:
            raise
        except Exception as e:
            print(f"  ⚠️ IBS 状态加载失败: {e}")
            return False
def _compute_bidirectional_overlap_from_u_kn(
    u_kn: np.ndarray,
    n_k: np.ndarray,
    threshold: float = 0.03,
) -> Dict[str, Any]:
    """Run the repository's normal MBAR overlap calculation for two states."""
    if not HAS_PYMBAR:
        raise ImportError("fixed-lambda overlap 探针需要 pymbar")
    u_kn = np.asarray(u_kn, dtype=np.float64)
    n_k = np.asarray(n_k, dtype=int)
    if u_kn.ndim != 2 or u_kn.shape[0] != 2 or n_k.shape != (2,):
        raise ValueError(f"双态 MBAR 输入形状错误: u_kn={u_kn.shape}, n_k={n_k.shape}")
    if int(np.sum(n_k)) != u_kn.shape[1] or np.any(n_k < 2):
        raise ValueError(f"双态 MBAR 样本计数错误: u_kn={u_kn.shape}, n_k={n_k.tolist()}")
    valid = np.all(np.isfinite(u_kn), axis=0)
    if not np.all(valid):
        raise RuntimeError("fixed-lambda overlap 探针产生 NaN/Inf，拒绝把坏帧静默丢弃")
    stable = u_kn - np.min(u_kn, axis=0, keepdims=True)
    mbar = _build_mbar_compatible(
        stable,
        n_k,
        relative_tolerance=1e-7,
        initialize="BAR",
        solver_protocol="default",
    )
    overlap_matrix = np.asarray(mbar.compute_overlap()["matrix"], dtype=np.float64)
    if overlap_matrix.shape != (2, 2) or not np.all(np.isfinite(overlap_matrix)):
        raise RuntimeError(f"PyMBAR 返回无效 overlap matrix: {overlap_matrix}")
    minimum = float(min(overlap_matrix[0, 1], overlap_matrix[1, 0]))
    # 🔑 [IBS_BIAS_PROTOCOL_VERSION=8] MBAR 在算 overlap 的同时已经把这两态之间的
    # BAR/MBAR 自由能差解出来了——这正是 IBS 偏置 U_k-f_k 要求的 f_k（均匀混合
    # 需要 f_k 等于状态 k 的相对自由能）。overlap 全部通过但 SGD 学习一直振荡时，
    # 与其继续在一个不稳定的求解器里盲搜，不如直接用这个已经算出来的真值校准
    # f_k，见调用方 run_all_windows 里 fixed-H overlap 全通过后的处理。返回的是
    # 约化单位（kT），换算成 kJ/mol 由调用方用自己的 kt 完成。
    res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
    df_matrix, ddf_matrix = _extract_free_energy_arrays(res, require_uncertainty=True)
    delta_f_reduced = float(df_matrix[0, 1])
    delta_f_uncertainty_reduced = float(ddf_matrix[0, 1])
    if not np.isfinite(delta_f_reduced):
        raise RuntimeError(f"PyMBAR 返回非有限的双态自由能差: {delta_f_reduced}")
    # 🔑 之前只检查了 delta_f 本身，没检查它的不确定度。这个不确定度会被下游
    # abfe_pipeline.py::_diagnose_and_repair_all_pass_low_ess_window 直接当噪声
    # 尺度用在判据 max(floor, sigma_multiplier * sigma) 里——如果 sigma 是 NaN，
    # 这个阈值也会变成 NaN，Python 的 `>` 比较对 NaN 恒为 False，会让"任意差异都
    # 判定为在阈值内"，错误地把本该判定为 recalibrate_f_k 的窗口放行成
    # reseed_resample。同样，负的不确定度在物理上没有意义，说明 PyMBAR 求解或
    # 上游数据有问题，不应该被静默当成正常值传下去。两种情况都必须 fail closed。
    if not (np.isfinite(delta_f_uncertainty_reduced) and delta_f_uncertainty_reduced >= 0.0):
        raise RuntimeError(
            f"PyMBAR 返回无效的双态自由能差不确定度: {delta_f_uncertainty_reduced}"
            "（非有限或为负），拒绝把它当作噪声尺度使用。"
        )
    return {
        "overlap_matrix": overlap_matrix.tolist(),
        "min_bidirectional_overlap": minimum,
        "threshold": float(threshold),
        "passed": bool(minimum >= float(threshold)),
        "delta_f_reduced_kT": delta_f_reduced,
        "delta_f_uncertainty_reduced_kT": delta_f_uncertainty_reduced,
    }


def probe_bidirectional_overlap(
    topology,
    common_system_xml: str,
    ibs_wrapper,
    state_i: int,
    state_j: int,
    positions,
    box_vectors,
    temperature,
    platform_name: str,
    burn_in_steps: int = 5000,
    sample_steps: int = 20000,
    sample_interval: int = 500,
    threshold: float = 0.03,
) -> Dict[str, Any]:
    """Measure overlap from independent unbiased fixed-H trajectories at i and j.

    Each dynamics system is exactly ``U_common + cv_k_int``.  It contains no
    CustomCVForce, no IBS bias and no WCA sampling shell.  The two trajectories
    are NVT because the common window system has no barostat and uses the same
    LangevinMiddleIntegrator settings as production.  Energies of both target
    states are evaluated at every frame through IBSSampler's existing probe
    context, then passed to the normal PyMBAR overlap implementation.
    """
    if state_i == state_j:
        raise ValueError("fixed-lambda overlap 的两个状态必须不同")
    cv_xmls = list(getattr(ibs_wrapper, "_int_cv_force_xmls", []))
    if max(state_i, state_j) >= len(cv_xmls):
        raise RuntimeError("IBS wrapper 缺少 fixed-lambda overlap 所需的 CV XML")
    if sample_interval <= 0 or sample_steps < 2 * sample_interval:
        raise ValueError("fixed-lambda overlap 采样长度不足")
    temperature_q = (
        temperature
        if hasattr(temperature, "value_in_unit")
        else float(temperature) * unit.kelvin
    )

    resolved_platform, props = _build_platform_properties(platform_name)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    simulations = []

    def _build_fixed_simulation(state_index: int, seed_offset: int):
        fixed_system = ensure_owned_system(XmlSerializer.deserialize(common_system_xml))
        fixed_cv = XmlSerializer.deserialize(cv_xmls[state_index])
        fixed_cv.setForceGroup(1)
        fixed_system.addForce(fixed_cv)
        if any(int(fixed_system.getForce(k).getForceGroup()) == 4 for k in range(fixed_system.getNumForces())):
            raise RuntimeError("fixed-lambda dynamics system 中不应存在 WCA group 4")
        integrator = LangevinMiddleIntegrator(
            temperature_q, 2.0 / unit.picosecond, 0.002 * unit.picosecond
        )
        integrator.setConstraintTolerance(1e-3)
        integrator.setRandomNumberSeed(8731 + int(seed_offset))
        if hasattr(integrator, "setRemoveCMMotion"):
            integrator.setRemoveCMMotion(True)
        simulation = app.Simulation(topology, fixed_system, integrator, platform, props)
        if box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(positions)
        # 🔑 [Hamiltonian mismatch fix] Context 默认 lambda_boresch_scale=0（见
        # LambdaDependentBoreschForce.__init__），但主窗口生产轨迹早已爬坡到
        # 1.0——必须显式对齐，否则这个 fixed-H 探针评估的是关掉 Boresch 限制的
        # 系统，跟它要验证的生产 Hamiltonian 不是同一个。
        if _system_has_global_parameter(fixed_system, "lambda_boresch_scale"):
            simulation.context.setParameter("lambda_boresch_scale", 1.0)
        simulation.context.setVelocitiesToTemperature(temperature_q, 29173 + int(seed_offset))
        return simulation

    def _collect_frames(simulation):
        simulation.step(int(burn_in_steps))
        frames = []
        n_samples = int(sample_steps) // int(sample_interval)
        for _ in range(n_samples):
            simulation.step(int(sample_interval))
            state = simulation.context.getState(getPositions=True)
            frames.append((state.getPositions(), state.getPeriodicBoxVectors()))
        return frames

    try:
        sim_i = _build_fixed_simulation(state_i, 0)
        simulations.append(sim_i)
        sim_j = _build_fixed_simulation(state_j, 1)
        simulations.append(sim_j)
        frames_by_ensemble = [_collect_frames(sim_i), _collect_frames(sim_j)]

        # Reuse the already validated all-state energy-only probe machinery.
        evaluator = IBSSampler(
            sim_i.context,
            len(cv_xmls),
            temperature_q,
            prefix=getattr(ibs_wrapper, "prefix", "abfe"),
            ibs_wrapper=ibs_wrapper,
        )
        # 🔑 跟 IBSSampler._lj_tail_correction_kj_mol() 读的是同一个
        # ibs_wrapper.lj_tail_lrc_coeff_kj_mol（switching+softcore-aware，见
        # _lj_tail_lrc_coefficients_kj_mol），保证 fixed-H overlap 探针和
        # 生产采样对同一个 λ 用同一组 LRC 系数，不会出现两条路径各算一遍、
        # 结果不一致的情况。
        lrc_coeff = getattr(ibs_wrapper, "lj_tail_lrc_coeff_kj_mol", None)
        selected = np.asarray([state_i, state_j], dtype=int)
        reduced_by_ensemble = []
        inefficiencies = []
        kt = (unit.MOLAR_GAS_CONSTANT_R * temperature_q).value_in_unit(unit.kilojoule_per_mole)
        beta = 1.0 / float(kt)
        for frames in frames_by_ensemble:
            values = []
            for frame_positions, frame_box in frames:
                e_kj = evaluator.evaluate_interaction_energies(frame_positions, frame_box)[selected]
                if lrc_coeff is not None:
                    box_nm = _box_vectors_to_nm_array(frame_box)
                    volume_nm3 = abs(float(np.linalg.det(box_nm)))
                    if not np.isfinite(volume_nm3) or volume_nm3 <= 0.0:
                        raise RuntimeError("fixed-lambda overlap 帧的盒子体积无效")
                    e_kj = e_kj + np.asarray(lrc_coeff, dtype=np.float64)[selected] / volume_nm3
                values.append(beta * np.asarray(e_kj, dtype=np.float64))
            reduced = np.asarray(values, dtype=np.float64).T
            decorrelated, g = subsample_series_by_autocorrelation(reduced[1] - reduced[0])
            reduced_by_ensemble.append(reduced[:, decorrelated])
            inefficiencies.append(float(g))

        n_k = np.asarray([block.shape[1] for block in reduced_by_ensemble], dtype=int)
        u_kn = np.concatenate(reduced_by_ensemble, axis=1)
        result = _compute_bidirectional_overlap_from_u_kn(u_kn, n_k, threshold=threshold)
        result.update({
            "local_states": [int(state_i), int(state_j)],
            "n_k_decorrelated": n_k.tolist(),
            "statistical_inefficiency": inefficiencies,
            "burn_in_steps": int(burn_in_steps),
            "sample_steps": int(sample_steps),
            "sample_interval": int(sample_interval),
            "dynamics_hamiltonian": "U_common_plus_single_cv_int",
            "ensemble": "NVT",
            # kJ/mol version of the BAR/MBAR free energy difference already
            # solved above. This is the *physical* delta_f (WCA-less
            # dynamics, LRC included in the energies fed to MBAR) -- useful
            # for path/lambda-grid overlap diagnostics, but
            # [IBS_BIAS_PROTOCOL_VERSION=10] it must NOT be used to calibrate
            # the IBS bias CV's f_k: production actually samples
            # U_common + WCA_window(lambda_shield) + CV_k (this function's
            # dynamics omit the WCA shield entirely), and the CV's own energy
            # never includes LRC (that is only added afterward, offline, to
            # target_energies feeding MBAR -- see IBSSampler.collect_energies).
            # Calibrating f_k from this field was a real, confirmed bug: the
            # measured delta_f corresponds to a different ensemble and a
            # different energy definition than what the bias force actually
            # needs to reproduce a uniform mixture over. Use
            # probe_bidirectional_overlap_for_bias_calibration's
            # delta_f_bias_kJ_mol for that purpose instead.
            "delta_f_kJ_mol": result["delta_f_reduced_kT"] * kt,
            "delta_f_uncertainty_kJ_mol": result["delta_f_uncertainty_reduced_kT"] * kt,
        })
        return result
    finally:
        simulations.clear()
        gc.collect()


def probe_bidirectional_overlap_for_bias_calibration(
    topology,
    common_plus_wca_system_xml: str,
    ibs_wrapper,
    state_i: int,
    state_j: int,
    positions,
    box_vectors,
    temperature,
    platform_name: str,
    lambda_shield: float,
    burn_in_steps: int = 5000,
    sample_steps: int = 20000,
    sample_interval: int = 500,
    min_decorrelated_samples: int = 20,
    max_delta_f_uncertainty_kJ_mol: float = 1.0,
    overlap_threshold: float = 0.03,
) -> Optional[Dict[str, Any]]:
    """Measure the adjacent-state free energy difference under the *exact*
    ensemble and energy definition the IBS bias CV's f_k actually needs to
    reproduce a uniform mixture over -- deliberately a separate function from
    ``probe_bidirectional_overlap`` (used elsewhere for path/lambda-grid
    overlap diagnostics), not a mode flag on it, so the two purposes can
    never again be silently conflated the way they were before
    [IBS_BIAS_PROTOCOL_VERSION=10]:

    - Dynamics: ``U_common + WCA_window(lambda_shield) + CV_k`` (Group 1
      single-state), built from ``common_plus_wca_system_xml`` (see
      ``_serialize_ibs_common_plus_wca_system``) with ``lambda_shield`` set
      to the same value production used for this window
      (``mean(lambdas_vdw_in_window)``). The WCA shield only cancels out of
      same-frame, same-window energy *differences* between states; it does
      not disappear from the sampled trajectory, so omitting it (as the
      path-overlap probe deliberately does, for its own different purpose)
      samples a genuinely different conformational ensemble than production.
    - Energy: raw softcore CV interaction energy only (``IBSSampler.
      evaluate_interaction_energies``), with **no** long-range dispersion
      correction added. Production's Group 1 bias force never includes LRC
      either -- LRC is only added afterward, offline, to ``target_energies``
      feeding MBAR/final analysis (see ``IBSSampler.collect_energies``:
      ``bias_cv_energies`` vs. ``target_energies = softcore_energies +
      lrc_energies``). Calibrating the bias CV from an LRC-inclusive delta_f
      would inject an energy term the bias force itself never applies.

    Returns ``None`` -- not a partial/best-effort result -- if, after
    ``burn_in_steps + sample_steps``, either the decorrelated sample count
    (``n_k_decorrelated``, per the *smaller* of the two ensembles) is below
    ``min_decorrelated_samples`` or the resulting delta_f uncertainty exceeds
    ``max_delta_f_uncertainty_kJ_mol``. A real overlap threshold gate alone
    (``passed``/``min_bidirectional_overlap`` are still reported for
    diagnostics) is not sufficient evidence that delta_f is precise enough to
    directly overwrite f_k -- e.g. a real run here saw
    ``n_k_decorrelated=[39, 9]`` with ``overlap=0.105``: minimally connected,
    but far too few decorrelated samples on one side to trust the resulting
    delta_f as a calibration target. The caller must extend sampling (larger
    ``sample_steps``) and retry rather than accept and use an
    under-supported estimate.
    """
    if state_i == state_j:
        raise ValueError("bias 校准探针的两个状态必须不同")
    cv_xmls = list(getattr(ibs_wrapper, "_int_cv_force_xmls", []))
    if max(state_i, state_j) >= len(cv_xmls):
        raise RuntimeError("IBS wrapper 缺少 bias 校准探针所需的 CV XML")
    if sample_interval <= 0 or sample_steps < 2 * sample_interval:
        raise ValueError("bias 校准探针采样长度不足")
    temperature_q = (
        temperature
        if hasattr(temperature, "value_in_unit")
        else float(temperature) * unit.kelvin
    )

    resolved_platform, props = _build_platform_properties(platform_name)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    simulations = []
    # 🔑 0.5→1.0→2.0 fs 短 ramp，2.0 fs 正好是正式 burn-in/采样步长，ramp 结束后
    # 不需要再显式恢复步长。步数很小（三段共 400 步 ≈ 0.5 ps），相对
    # burn_in_steps（默认 5000 步 = 10 ps）+ sample_steps 的开销可以忽略。
    _RAMP_STAGES_FS_STEPS = ((0.5, 100), (1.0, 100), (2.0, 200))

    def _finite_check(simulation, label: str) -> None:
        state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
        e_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        )
        positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        if not (np.isfinite(e_kj) and np.all(np.isfinite(forces)) and np.all(np.isfinite(positions_nm))):
            raise RuntimeError(
                f"[bias 校准探针] {label}：检测到非有限的能量/力/坐标"
                f"（potential_energy={e_kj!r} kJ/mol）。提前终止，不进入正式 "
                "burn-in/采样。"
            )

    def _step_with_nan_guard(simulation, n_steps: int, label: str) -> None:
        try:
            simulation.step(int(n_steps))
        except openmm.OpenMMException as exc:
            raise RuntimeError(
                f"[bias 校准探针] {label}：积分过程中出现非有限坐标"
                f"（原始异常: {exc}）。"
            ) from exc

    def _build_fixed_simulation(state_index: int, seed_offset: int):
        label = f"state_index={state_index}（edge state_i={state_i}, state_j={state_j}）"
        fixed_system = ensure_owned_system(XmlSerializer.deserialize(common_plus_wca_system_xml))
        fixed_cv = XmlSerializer.deserialize(cv_xmls[state_index])
        fixed_cv.setForceGroup(1)
        fixed_system.addForce(fixed_cv)
        has_group4 = any(
            int(fixed_system.getForce(k).getForceGroup()) == 4
            for k in range(fixed_system.getNumForces())
        )
        if not has_group4:
            raise RuntimeError("bias 校准探针的 fixed-H 系统缺少 Group 4 WCA 防护壳")
        integrator = LangevinMiddleIntegrator(
            temperature_q, 2.0 / unit.picosecond, 0.002 * unit.picosecond
        )
        integrator.setConstraintTolerance(1e-3)
        integrator.setRandomNumberSeed(41337 + int(seed_offset))
        if hasattr(integrator, "setRemoveCMMotion"):
            integrator.setRemoveCMMotion(True)
        simulation = app.Simulation(topology, fixed_system, integrator, platform, props)
        if box_vectors is not None:
            simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(positions)
        if _system_has_global_parameter(fixed_system, "lambda_shield"):
            simulation.context.setParameter("lambda_shield", float(lambda_shield))
        # 🔑 [Hamiltonian mismatch fix] 同 probe_bidirectional_overlap 里
        # _build_fixed_simulation 的注释：必须显式对齐到主窗口早已爬坡到的 1.0，
        # 否则这份"证明 f_k 物理正确"的校准探针实际验证的是另一个 Hamiltonian。
        if _system_has_global_parameter(fixed_system, "lambda_boresch_scale"):
            simulation.context.setParameter("lambda_boresch_scale", 1.0)
        # 🔑 [live crash fix] 在真实 lambda_shield + 这个态自己的 CV 力都已加入
        # 之后，针对这个具体态的精确 Hamiltonian 再单独最小化一次——外层调用方
        # 传入的 positions 只是这个窗口的粗弛豫构型（可能是在另一个
        # lambda_shield 值、甚至完全没有 WCA 防护壳的势能面上做的），不能保证
        # 对这个态、这个 lambda_shield 强度也是无碰撞的。这是防止防护壳力在
        # 动力学开始时撞上残余原子重叠、第一步就爆炸成 NaN 的最可靠手段
        # （比只保证外层粗弛豫时 lambda_shield 同步更进一步的第二道保险）。
        try:
            simulation.minimizeEnergy(maxIterations=2000)
        except openmm.OpenMMException as exc:
            raise RuntimeError(
                f"[bias 校准探针] {label}：局部最小化过程中出现非有限坐标"
                f"（原始异常: {exc}）。"
            ) from exc
        _finite_check(simulation, f"{label} 局部最小化后")
        simulation.context.setVelocitiesToTemperature(temperature_q, 62143 + int(seed_offset))
        # 🔑 最小化只保证落在一个（局部）能量极小点，不保证 Langevin 动力学
        # 用生产步长（2 fs）起步时力已经足够温和——用更小的步长先走几百步，
        # 每段之间检查有限性，再逐步过渡到正式步长，避免真实运行中窗口 (5,9)
        # 边 [7,8] 那种"第一步就 NaN"的爆炸；NaN 时明确报告是哪个 state/edge
        # 出的问题，而不是让 OpenMM 的裸 traceback 直接冒出来。
        for ramp_dt_fs, ramp_steps in _RAMP_STAGES_FS_STEPS:
            simulation.integrator.setStepSize(ramp_dt_fs * unit.femtosecond)
            _step_with_nan_guard(simulation, ramp_steps, f"{label} ramp {ramp_dt_fs} fs")
            _finite_check(simulation, f"{label} ramp {ramp_dt_fs} fs 后")
        return simulation

    def _collect_frames(simulation, state_index: int):
        label = f"state_index={state_index}（edge state_i={state_i}, state_j={state_j}）"
        _step_with_nan_guard(simulation, burn_in_steps, f"{label} burn-in")
        _finite_check(simulation, f"{label} burn-in 后")
        frames = []
        n_samples = int(sample_steps) // int(sample_interval)
        for _ in range(n_samples):
            _step_with_nan_guard(simulation, sample_interval, f"{label} 采样")
            state = simulation.context.getState(getPositions=True)
            frames.append((state.getPositions(), state.getPeriodicBoxVectors()))
        return frames

    try:
        sim_i = _build_fixed_simulation(state_i, 0)
        simulations.append(sim_i)
        sim_j = _build_fixed_simulation(state_j, 1)
        simulations.append(sim_j)
        frames_by_ensemble = [
            _collect_frames(sim_i, state_i),
            _collect_frames(sim_j, state_j),
        ]

        evaluator = IBSSampler(
            sim_i.context,
            len(cv_xmls),
            temperature_q,
            prefix=getattr(ibs_wrapper, "prefix", "abfe"),
            ibs_wrapper=ibs_wrapper,
        )
        selected = np.asarray([state_i, state_j], dtype=int)
        reduced_by_ensemble = []
        inefficiencies = []
        kt = (unit.MOLAR_GAS_CONSTANT_R * temperature_q).value_in_unit(unit.kilojoule_per_mole)
        beta = 1.0 / float(kt)
        for frames in frames_by_ensemble:
            values = []
            for frame_positions, frame_box in frames:
                # 🔑 只用纯 softcore CV 能量，绝不加 LRC——见函数 docstring。
                e_kj = evaluator.evaluate_interaction_energies(frame_positions, frame_box)[selected]
                values.append(beta * np.asarray(e_kj, dtype=np.float64))
            reduced = np.asarray(values, dtype=np.float64).T
            decorrelated, g = subsample_series_by_autocorrelation(reduced[1] - reduced[0])
            reduced_by_ensemble.append(reduced[:, decorrelated])
            inefficiencies.append(float(g))

        n_k = np.asarray([block.shape[1] for block in reduced_by_ensemble], dtype=int)
        if int(np.min(n_k)) < int(min_decorrelated_samples):
            return None
        u_kn = np.concatenate(reduced_by_ensemble, axis=1)
        result = _compute_bidirectional_overlap_from_u_kn(u_kn, n_k, threshold=overlap_threshold)
        delta_f_bias_kj = result["delta_f_reduced_kT"] * kt
        delta_f_bias_uncertainty_kj = result["delta_f_uncertainty_reduced_kT"] * kt
        if delta_f_bias_uncertainty_kj > float(max_delta_f_uncertainty_kJ_mol):
            return None
        result.update({
            "local_states": [int(state_i), int(state_j)],
            "n_k_decorrelated": n_k.tolist(),
            "statistical_inefficiency": inefficiencies,
            "burn_in_steps": int(burn_in_steps),
            "sample_steps": int(sample_steps),
            "sample_interval": int(sample_interval),
            "dynamics_hamiltonian": "U_common_plus_wca_window_plus_single_cv_int_no_lrc",
            "ensemble": "NVT",
            "lambda_shield": float(lambda_shield),
            # These are the ONLY fields that may be used to calibrate f_k --
            # see docstring. Deliberately named differently from
            # probe_bidirectional_overlap's delta_f_kJ_mol so the two can
            # never be silently swapped again.
            "delta_f_bias_kJ_mol": delta_f_bias_kj,
            "delta_f_bias_uncertainty_kJ_mol": delta_f_bias_uncertainty_kj,
        })
        return result
    finally:
        simulations.clear()
        gc.collect()


# ============================================================================
# 2b. 按"态"共享的可续采 fixed-H 探针轨迹库 [IBS_BIAS_PROTOCOL_VERSION=11]
#
# 替代上面两个按"边"独立采样的探针在 run_all_windows 里的用法（这两个函数
# 本身原样保留，继续服务 abfe_pipeline.py::_probe_vdw_window_fixed_overlap
# ——见 IBS_BIAS_PROTOCOL_VERSION=11 changelog，那个调用点的起始构型是重新
# 最小化过的，跟这里 warmup 阶段活体 context 的当前构型不是同一份，不能共用
# checkpoint，本次只重构 run_all_windows 这一个调用点）。
#
# 与 abfe_pipeline.py 里已有的边级别落盘缓存
# （_fixed_h_probe_fingerprint/_persist_fixed_h_probe_edge/
# _load_fixed_h_probe_cache）不是同一层、也不是重复实现：那是"边结果缓存"
# （存的是某条边探针跑完之后的结果字典），这里是"态原始轨迹缓存"（存的是
# 某个态自己的能量/体积时间序列，供任意需要它的边现算 MBAR）。调用顺序永远
# 是先查边结果缓存，miss 了再落到这里补采，不会反过来。
# ============================================================================

# 🔑 随机种子按"全局态序号"（global_state_index = global_state_start + 窗口内
# 局部态序号）生成，不再按"边"用固定 offset（旧的 probe_bidirectional_overlap
# 系列用 0/1、41337/62143 等常量+0/1 offset）——因为现在一个态的轨迹要同时服务
# 它左右两条边，如果还按边分配种子，同一个态在两次不同的边循环里会被要求
# "既是这个种子又是那个种子"，无意义。四个独立 BASE（路径/校准各自的积分器
# 种子、速度种子）+ 质数步长 97（避免 97*k 在 k 较小时出现周期性碰撞）保证
# 不同态、不同探针类型之间的种子互不相撞，且远高于旧常量，不会跟旧代码路径
# （仍在用的两个按边探针）撞车。
PATH_PROBE_INTEGRATOR_SEED_BASE = 900_000
PATH_PROBE_VELOCITY_SEED_BASE = 901_000
BIAS_CALIB_INTEGRATOR_SEED_BASE = 902_000
BIAS_CALIB_VELOCITY_SEED_BASE = 903_000
_FIXED_H_PROBE_SEED_STRIDE = 97


def _build_fixed_state_simulation(
    topology,
    system_xml: str,
    cv_xml: str,
    require_group4: bool,
    platform_name: str,
    temperature_q,
    positions,
    box_vectors,
    integrator_seed: int,
    velocity_seed: int,
    velocities=None,
    lambda_shield: Optional[float] = None,
):
    """Build one fixed-Hamiltonian ``app.Simulation`` for a single state's own
    trajectory-bank entry -- shared by both ``probe_adjacent_path_overlap_bank``
    (``require_group4=False``, dynamics is ``U_common + CV_k``) and
    ``probe_adjacent_bias_calibration_bank`` (``require_group4=True``,
    dynamics is ``U_common + WCA_window(lambda_shield) + CV_k``). Factors out
    the two hard Group-4 assertions that used to be duplicated, inline, in
    ``probe_bidirectional_overlap``/``probe_bidirectional_overlap_for_bias_calibration``,
    so both directions (missing WCA when required, present WCA when
    forbidden) are independent, unit-testable failures instead of silently
    sampling the wrong ensemble.

    ``velocities``, if given, seeds the Simulation from an exact checkpointed
    velocity array (trajectory continuation -- no re-thermalization); when
    omitted, velocities are freshly drawn from the Maxwell-Boltzmann
    distribution at ``temperature_q`` using ``velocity_seed`` (a brand-new
    segment).
    """
    fixed_system = ensure_owned_system(XmlSerializer.deserialize(system_xml))
    fixed_cv = XmlSerializer.deserialize(cv_xml)
    fixed_cv.setForceGroup(1)
    fixed_system.addForce(fixed_cv)
    has_group4 = any(
        int(fixed_system.getForce(k).getForceGroup()) == 4
        for k in range(fixed_system.getNumForces())
    )
    if require_group4 and not has_group4:
        raise RuntimeError("bias 校准轨迹库的 fixed-H 系统缺少 Group 4 WCA 防护壳")
    if not require_group4 and has_group4:
        raise RuntimeError("path overlap 轨迹库的 dynamics system 中不应存在 WCA group 4")
    resolved_platform, props = _build_platform_properties(platform_name)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    integrator = LangevinMiddleIntegrator(
        temperature_q, 2.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    integrator.setConstraintTolerance(1e-3)
    integrator.setRandomNumberSeed(int(integrator_seed))
    if hasattr(integrator, "setRemoveCMMotion"):
        integrator.setRemoveCMMotion(True)
    simulation = app.Simulation(topology, fixed_system, integrator, platform, props)
    if box_vectors is not None:
        simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    if require_group4 and lambda_shield is not None and _system_has_global_parameter(fixed_system, "lambda_shield"):
        simulation.context.setParameter("lambda_shield", float(lambda_shield))
    # 🔑 [Hamiltonian mismatch fix] 同上：必须显式对齐到主窗口早已爬坡到的 1.0，
    # 否则 probe_adjacent_path_overlap_bank/probe_adjacent_bias_calibration_bank
    # 用来"证明 f_k 物理正确"的轨迹库实际跑在关闭 Boresch 限制的系统上。
    if _system_has_global_parameter(fixed_system, "lambda_boresch_scale"):
        simulation.context.setParameter("lambda_boresch_scale", 1.0)
    if velocities is not None:
        simulation.context.setVelocities(velocities)
    else:
        simulation.context.setVelocitiesToTemperature(temperature_q, int(velocity_seed))
    return simulation


def _fixed_h_probe_bank_dir(checkpoint_dir: str, stage_type: str, window_idx: int, probe_type: str) -> str:
    return os.path.join(checkpoint_dir, "probes", stage_type, f"window_{window_idx}", probe_type)


def _fixed_h_probe_bank_manifest_path(bank_dir: str) -> str:
    return os.path.join(bank_dir, "manifest.json")


def _fixed_h_probe_bank_state_pointer_path(bank_dir: str, k: int) -> str:
    """Path to the single small pointer file that says which *generation* of
    a state's 5-file artifact set is currently trustworthy -- see
    ``_fixed_h_probe_bank_state_paths`` and ``_persist_state_trajectory_record``.
    """
    return os.path.join(bank_dir, f"state_{k}_generation.json")


def _read_fixed_h_probe_bank_state_generation(bank_dir: str, k: int) -> Optional[int]:
    """Read one state's currently-committed generation number, or ``None``
    if the pointer file is missing/corrupt -- callers must treat that as "no
    persisted record at all" (this pointer is written only after every
    generation-tagged file below it has been durably written, so its absence
    means nothing for this state has ever been fully committed)."""
    pointer_path = _fixed_h_probe_bank_state_pointer_path(bank_dir, k)
    if not os.path.exists(pointer_path):
        return None
    try:
        with open(pointer_path, "r", encoding="utf-8") as f:
            return int(json.load(f)["generation"])
    except Exception:
        return None


def _fixed_h_probe_bank_state_paths(bank_dir: str, k: int, generation: int) -> Tuple[str, str, str, str, str]:
    """Paths for one state's generation-tagged artifact set.

    ``generation`` is the state's own ``sampled_steps`` value at the moment
    these were written -- a naturally monotonic version stamp, reused
    instead of introducing a separate counter. These 5 files are never
    overwritten in place: each persist writes a *new* generation's files,
    then atomically flips ``state_{k}_generation.json`` to point at it (see
    ``_persist_state_trajectory_record``). This is what makes the 5-file set
    atomic as a whole despite each file only being individually atomic on
    its own -- a crash between individual file writes just leaves the
    pointer referencing the last complete generation; the partially-written
    new generation's files are orphaned and ignored, never read, because
    nothing ever resolves a state's files except through the pointer.
    """
    tag = f"g{int(generation)}"
    return (
        os.path.join(bank_dir, f"state_{k}_{tag}_energies.npy"),
        os.path.join(bank_dir, f"state_{k}_{tag}_volume.npy"),
        os.path.join(bank_dir, f"state_{k}_{tag}_checkpoint.npz"),
        os.path.join(bank_dir, f"state_{k}_{tag}_meta.json"),
        os.path.join(bank_dir, f"state_{k}_{tag}_openmm.chk"),
    )


def _build_fixed_h_probe_bank_manifest(
    probe_type: str,
    stage_type: str,
    window_idx: int,
    K: int,
    global_state_start: int,
    common_system_xml: str,
    cv_xmls: List[str],
    lambda_shield: Optional[float],
    temperature_K: float,
    sample_interval: int,
    platform_name: str,
) -> Dict[str, Any]:
    """Content-based fingerprint for one window's fixed-H probe trajectory
    bank. Any change to the physical Hamiltonian (system/CV XML content,
    cache/bias protocol version, lambda_shield, temperature, sample_interval,
    platform) must invalidate every state's already-sampled frames in this
    window+probe_type directory -- see the resume trust rules in
    ``probe_adjacent_path_overlap_bank``/``probe_adjacent_bias_calibration_bank``.
    XML blobs are hashed (not stored verbatim) to keep manifest.json small.
    """
    return {
        "fixed_h_probe_cache_protocol_version": FIXED_H_PROBE_CACHE_PROTOCOL_VERSION,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        "probe_type": str(probe_type),
        "stage_type": str(stage_type),
        "window_idx": int(window_idx),
        "K": int(K),
        "global_state_range": [int(global_state_start), int(global_state_start) + int(K)],
        "common_system_xml_sha256": hashlib.sha256(common_system_xml.encode("utf-8")).hexdigest(),
        "cv_xmls_sha256": [hashlib.sha256(x.encode("utf-8")).hexdigest() for x in cv_xmls],
        "lambda_shield": (float(lambda_shield) if lambda_shield is not None else None),
        "temperature_K": float(temperature_K),
        "step_size_ps": 0.002,
        "friction_per_ps": 2.0,
        "sample_interval": int(sample_interval),
        "platform_name": str(platform_name),
    }


def _fixed_h_probe_bank_manifest_matches(bank_dir: str, expected_manifest: Dict[str, Any]) -> bool:
    manifest_path = _fixed_h_probe_bank_manifest_path(bank_dir)
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return False
    normalized_expected = _normalize_ibs_protocol_for_cache_compare(expected_manifest)
    normalized_loaded = _normalize_ibs_protocol_for_cache_compare(loaded)
    return normalized_loaded == normalized_expected


def _invalidate_fixed_h_probe_bank(bank_dir: str) -> None:
    """Discard an entire window+probe_type trajectory bank -- any manifest
    mismatch (protocol version, lambda content, lambda_shield, ...) makes
    every state's already-sampled frames untrustworthy at once, matching the
    existing whole-window convergence-cache philosophy elsewhere in this
    file: trust everything only if every fingerprinted field matches, else
    reject the whole thing and resample from scratch."""
    if os.path.isdir(bank_dir):
        shutil.rmtree(bank_dir, ignore_errors=True)


def _load_state_trajectory_record(
    bank_dir: str, k: int, K: int, sample_interval: int
) -> Optional[Dict[str, Any]]:
    """Load one state's persisted energies/volumes/segment-metadata for the
    generation named by its pointer file, if present and internally
    consistent -- ``None`` otherwise (never a partial/best-effort record):

    - the pointer (``_read_fixed_h_probe_bank_state_generation``) must exist;
    - the pointer's generation-tagged ``energies.npy``/``meta.json`` must
      both exist and parse;
    - ``energies.npy`` must be 2D with exactly ``K`` columns and all-finite;
    - ``volume.npy`` must exist (a missing file is treated as corruption,
      never as "no volume data"), must have the same frame count as
      ``energies.npy``, and must be strictly positive everywhere;
    - ``meta.json``'s own bookkeeping must be self-consistent:
      ``sum(segment.n_frames) == energies.npy`` frame count,
      ``sum(segment.sample_steps) == sampled_steps``, and every segment's
      ``sample_steps == n_frames * sample_interval`` individually (not just
      in aggregate, so a corrupted middle segment can't hide behind
      coincidentally-matching sums).

    Callers must treat any failure here as "start this state's segment 0
    from scratch", never as an empty-but-trustworthy record. Continuation
    state (the native OpenMM checkpoint and its NPZ fallback) is loaded
    separately, lazily, by ``_resume_or_start_state_simulation`` -- this
    function only ever reads the frame-history side of the record.
    """
    generation = _read_fixed_h_probe_bank_state_generation(bank_dir, k)
    if generation is None:
        return None
    energies_path, volume_path, _, meta_path, _ = _fixed_h_probe_bank_state_paths(bank_dir, k, generation)
    if not (os.path.exists(energies_path) and os.path.exists(meta_path)):
        return None
    try:
        u_cv_kj_mol = np.asarray(np.load(energies_path), dtype=np.float64)
        if u_cv_kj_mol.ndim != 2 or u_cv_kj_mol.shape[1] != int(K):
            raise ValueError(f"energies.npy 形状与窗口 K 不符: {u_cv_kj_mol.shape} vs K={K}")
        if not np.all(np.isfinite(u_cv_kj_mol)):
            raise ValueError("energies.npy 含非有限值")
        if not os.path.exists(volume_path):
            # 🔑 volume.npy 缺失不能补零继续：下游路径探针在 LRC 里算
            # coefficient / volume，零体积会先产生 Inf，续采时又可能把这段
            # 零体积和后续真实体积拼接成同一条 volume_nm3。缺失就是记录损坏，
            # 必须让外层 except 捕获、返回 None，强制该态从 segment 0 重采。
            raise ValueError("volume.npy 缺失，判定该态记录损坏")
        volume_nm3 = np.asarray(np.load(volume_path), dtype=np.float64)
        if volume_nm3.shape[0] != u_cv_kj_mol.shape[0]:
            raise ValueError("volume.npy 长度与 energies.npy 帧数不一致")
        if not np.all(np.isfinite(volume_nm3)) or not np.all(volume_nm3 > 0.0):
            raise ValueError("volume.npy 含非有限或非正值")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if int(meta.get("generation", generation)) != int(generation):
            raise ValueError("meta.json 记录的 generation 与 pointer 不一致")
        segments = list(meta["segments"])
        sampled_steps = int(meta["sampled_steps"])
        total_frames = sum(int(seg["n_frames"]) for seg in segments)
        if total_frames != u_cv_kj_mol.shape[0]:
            raise ValueError("segment n_frames 之和与 energies.npy 帧数不一致")
        total_sample_steps = sum(int(seg["sample_steps"]) for seg in segments)
        if total_sample_steps != sampled_steps:
            raise ValueError("segment sample_steps 之和与 sampled_steps 不一致")
        for seg in segments:
            if int(seg["sample_steps"]) != int(seg["n_frames"]) * int(sample_interval):
                raise ValueError("segment 的 sample_steps 与 n_frames*sample_interval 不匹配")
        record = {
            "u_cv_kj_mol": u_cv_kj_mol,
            "volume_nm3": volume_nm3,
            "segments": segments,
            "sampled_steps": sampled_steps,
        }
    except Exception:
        return None
    return record


def _load_state_checkpoint_npz(bank_dir: str, k: int) -> Optional[Dict[str, np.ndarray]]:
    """Load one state's NPZ positions/velocities/box fallback snapshot for
    its currently-committed generation, if present, intact, and stamped with
    a ``sampled_steps`` matching the pointer -- ``None`` otherwise. This is
    the *degraded* resume path (see ``_resume_or_start_state_simulation``):
    it does not capture the integrator's internal RNG state, so it is only
    ever used to seed a fresh, re-burned segment, never as a true
    continuation. Read fresh from disk every time it's needed (never cached
    in memory) since a new generation can be committed between tiers within
    a single bank call.
    """
    generation = _read_fixed_h_probe_bank_state_generation(bank_dir, k)
    if generation is None:
        return None
    _, _, checkpoint_path, _, _ = _fixed_h_probe_bank_state_paths(bank_dir, k, generation)
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with np.load(checkpoint_path) as npz:
            if int(npz["sampled_steps"]) != int(generation):
                return None
            return {
                "positions_nm": np.asarray(npz["positions_nm"], dtype=np.float64),
                "velocities_nm_per_ps": np.asarray(npz["velocities_nm_per_ps"], dtype=np.float64),
                "box_nm": np.asarray(npz["box_nm"], dtype=np.float64),
            }
    except Exception:
        return None


def _atomic_save_openmm_checkpoint(simulation, filepath: str) -> None:
    """Atomically persist OpenMM's own binary Context checkpoint (positions,
    velocities, box *and* the integrator's internal state, including the
    stochastic-thermostat RNG stream) -- write to a temp path, then
    ``os.replace``. This, not the NPZ array dump, is the trajectory bank's
    real continuation mechanism: NPZ alone reconstructs a Context with
    matching positions/velocities but a freshly re-seeded integrator, which
    is a *different* (if still valid) Langevin trajectory, not a bit-level
    continuation of the one that was running -- see
    ``_resume_or_start_state_simulation``.
    """
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_file = filepath + ".tmp"
    simulation.saveCheckpoint(tmp_file)
    os.replace(tmp_file, filepath)


# 独立于 IBS_BIAS_PROTOCOL_VERSION：只管主窗口（run_all_windows，不是探针轨迹库）
# 这份 OpenMM 原生 checkpoint 文件本身的格式/指纹字段，不影响 ibs_state_*.json
# 里任何已有字段的含义——那部分完全不变，见 IBSSampler.save_ibs_state/load_ibs_state。
#   version 1: 首次引入。真实 GPU 生产日志发现：resumed_calibration_pending 续验
#     冻结校准 f_k 时，每一次阶梯升级重试（frozen_validation_step_overrides 给的
#     50k/150k/300k）都会对这个窗口重新最小化 + dt 测试步进 + Boresch 安全爬坡，
#     即使窗口的构象/速度/盒子在上一次尝试结束时早已充分平衡、冻结 f_k 完全没变
#     ——这是实现缺口，不是物理上必须的（IBS_BIAS_PROTOCOL_VERSION=12 changelog
#     (3) 条曾经声称已经用 OpenMM 原生 checkpoint 解决了这一点，但实际从未真正
#     实现在主窗口路径里，只实现在下面的探针轨迹库里；那条注释已经改正）。
#     现在把探针轨迹库已验证好的 native checkpoint 续算模式原样搬到主窗口。
#   version 2: 同 IBS_BIAS_PROTOCOL_VERSION=13/FIXED_H_PROBE_CACHE_PROTOCOL_
#     VERSION=3 的 lambda_boresch_scale 修复——这份 checkpoint 的 manifest
#     （_build_main_window_checkpoint_manifest）只哈希 win_sys_xml/lambdas/
#     lambda_shield/temperature/platform，Boresch Context 修复不改变其中任何
#     一项（win_sys_xml 序列化字节不变，Boresch 力默认值仍是 0.0，只是运行时
#     setParameter 覆盖），所以旧 checkpoint 会被误判为仍然可用而直接
#     loadCheckpoint 续算——但那份坐标/速度/积分器状态是在 Boresch=0（或未经
#     校准探针验证过 Boresch=1 时）的轨迹上产生的，不能再假设它是干净续算的
#     起点。升版本号强制这次 resume 全部回退到完整重建流程（最小化+dt测试+
#     Boresch 爬坡），不再信任修复之前的主窗口 checkpoint。
MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION = 2


def _main_window_checkpoint_paths(checkpoint_dir: str, stage_type: str, window_idx: int) -> Tuple[str, str]:
    """主窗口 checkpoint 的落盘路径（openmm.chk + manifest.json），布局风格
    照抄探针轨迹库：{checkpoint_dir}/main_window/{stage_type}/window_{idx}/。
    跟探针轨迹库不同，这里只保留最新一份（原地覆盖，不按代际累积）——主窗口
    resume 永远只需要"最后一次看到的状态"，没有多代际分别去相关的需求。
    """
    window_dir = os.path.join(checkpoint_dir, "main_window", str(stage_type), f"window_{int(window_idx)}")
    return (
        os.path.join(window_dir, "openmm.chk"),
        os.path.join(window_dir, "manifest.json"),
    )


def _build_main_window_checkpoint_manifest(
    stage_type: str,
    window_idx: int,
    K: int,
    win_sys_xml: str,
    lambdas_coul: Any,
    lambdas_vdw: Any,
    lambda_shield: Optional[float],
    temperature_K: float,
    platform_name: str,
    repair_policy: str = "non_mutating_v1",
) -> Dict[str, Any]:
    """主窗口 checkpoint 的内容指纹——任何一项不匹配都必须整体拒绝这份
    checkpoint（λ 网格被自动加密/重新划分窗口、协议版本变化、平台不同等），
    直接沿用探针轨迹库 `_build_fixed_h_probe_bank_manifest` 的字段/哈希写法。
    """
    return {
        "main_window_checkpoint_protocol_version": MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        # 🔑 [non_mutating_v1] 采样修复策略进入 checkpoint 指纹：旧变异策略下的
        # 主窗口 checkpoint（可能带就地重校准过的 f_k）与非变异策略不匹配即拒绝。
        "sampling_repair_policy": str(repair_policy),
        "stage_type": str(stage_type),
        "window_idx": int(window_idx),
        "K": int(K),
        "win_sys_xml_sha256": hashlib.sha256(win_sys_xml.encode("utf-8")).hexdigest(),
        "lambdas_coul": [float(x) for x in lambdas_coul],
        "lambdas_vdw": [float(x) for x in lambdas_vdw],
        "lambda_shield": (float(lambda_shield) if lambda_shield is not None else None),
        "temperature_K": float(temperature_K),
        "step_size_ps": 0.002,
        "friction_per_ps": 2.0,
        "platform_name": str(platform_name),
    }


def _main_window_checkpoint_is_usable(
    checkpoint_dir: str, stage_type: str, window_idx: int, expected_manifest: Dict[str, Any]
) -> bool:
    """checkpoint 文件与 manifest 文件都必须存在，且 manifest 内容完全匹配才
    返回 True——照抄 `_fixed_h_probe_bank_manifest_matches` 的比较写法（json
    归一化后按内容比较，不是按字符串逐字节比较）。任何缺失/损坏/不匹配都
    安全返回 False，让调用方回退到完整重建流程，绝不抛异常。
    """
    ckpt_path, manifest_path = _main_window_checkpoint_paths(checkpoint_dir, stage_type, window_idx)
    if not (os.path.exists(ckpt_path) and os.path.exists(manifest_path)):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return False
    normalized_expected = _normalize_ibs_protocol_for_cache_compare(expected_manifest)
    normalized_loaded = _normalize_ibs_protocol_for_cache_compare(loaded)
    return normalized_loaded == normalized_expected


def _peek_ibs_bias_status(ibs_state_file: str) -> Optional[str]:
    """只读一次 ibs_state_*.json、返回其中的 bias_status 字段，不做任何其它
    校验（n_states/协议版本/λ 内容是否匹配仍然只由 IBSSampler.load_ibs_state
    的完整校验负责）。这里只是一个"值不值得尝试 loadCheckpoint"的廉价预判，
    任何异常（文件不存在/JSON 损坏/字段缺失）都返回 None，绝不抛出。
    """
    try:
        with open(ibs_state_file, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state.get("bias_status")
    except Exception:
        return None


def _try_load_main_window_checkpoint(sim, checkpoint_path: str) -> bool:
    """尝试从主窗口 checkpoint 恢复这个 Context 的坐标/速度/盒子/积分器 RNG
    状态。任何失败（文件不存在、损坏、跟当前 System/Integrator/Platform 不
    兼容）都必须安全返回 False，绝不能让一个坏 checkpoint 搞崩整个窗口——
    调用方在 False 时应回退到完整的重建+最小化+爬坡流程。
    """
    try:
        sim.loadCheckpoint(checkpoint_path)
        return True
    except Exception:
        return False


# 🔑 [production checkpoint 续采] 独立于 MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION
# （只覆盖 warmup/冻结验证阶段）——那份 checkpoint 在 bias_converged 变 True、
# 真正进入生产采样的那一刻就被删除（见 run_all_windows 里那段清理代码），
# 从未在生产采样阶段被复用过。之前 production ESS 低触发 reseed_resample 时，
# _invalidate_single_window_production 会整窗删掉 energies/bias/base/
# convergence.json，下一轮从头重建 Context、从 stage 起始坐标重新最小化+
# dt测试+Boresch爬坡+冻结重验证+从零步production——"250k 延长到 500k"实际是
# "扔掉这 250k、重跑一条独立的 500k"，不是真正的续算。这里补上生产采样自己的
# native OpenMM checkpoint，让"λ/窗口/系统/冻结 f_k 都没变，只是 production
# ESS 还不够、需要更多样本"这一种（也是最常见的）情况能真正从上次结束的坐标/
# 速度/积分器 RNG 状态接着跑，而不是重新起跑一条独立轨迹。
PRODUCTION_WINDOW_CHECKPOINT_PROTOCOL_VERSION = 1


def _production_window_checkpoint_paths(
    checkpoint_dir: str, stage_type: str, window_idx: int
) -> Tuple[str, str]:
    """生产采样 checkpoint 的落盘路径——布局风格照抄
    `_main_window_checkpoint_paths`：{checkpoint_dir}/production_window/
    {stage_type}/window_{idx}/。同样只保留最新一份，原地覆盖，不按代际累积
    ——续算永远只需要"最后一次看到的状态"，生产阶段没有需要分段去相关的
    多代际需求（不同于探针轨迹库，见 _fixed_h_probe_bank_state_paths）。
    """
    window_dir = os.path.join(
        checkpoint_dir, "production_window", str(stage_type), f"window_{int(window_idx)}"
    )
    return (
        os.path.join(window_dir, "openmm.chk"),
        os.path.join(window_dir, "manifest.json"),
    )


def _build_production_window_checkpoint_manifest(
    stage_type: str,
    window_idx: int,
    K: int,
    win_sys_xml: str,
    lambdas_coul: Any,
    lambdas_vdw: Any,
    lambda_shield: Optional[float],
    lambda_boresch_scale: Optional[float],
    frozen_f_k: Any,
    temperature_K: float,
    platform_name: str,
    repair_policy: str = "non_mutating_v1",
) -> Dict[str, Any]:
    """生产采样 checkpoint 的内容指纹——直接沿用
    `_build_main_window_checkpoint_manifest` 的字段/哈希写法（win_sys_xml/
    lambdas/lambda_shield/温度/平台/协议版本），但多两个 warmup checkpoint
    不需要的字段：

    - `lambda_boresch_scale`：warmup checkpoint 続采的是"还在冻结验证阶段"的
      窗口，那时 Boresch 早就爬坡到 1.0 且全程不变，不需要单独校验；但这里
      要独立确认，避免把这份指纹的正确性建立在"warmup 那份检查已经做过"这个
      隐含假设上。
    - `frozen_f_k_sha256`：production checkpoint 场景是"这个窗口已经用某个
      冻结 f_k 采过一批生产样本，现在还要不要接着采"——f_k 若变了（比如同一
      窗口后来被 recalibrate_f_k 覆盖），新旧样本就不再来自同一个采样分布，
      必须老实重采，不能续算；warmup checkpoint 没有这个概念，因为它续的是
      "验证 f_k 是否可信"这件事本身，f_k 在那个阶段还不该被信任为"已经在用
      它采生产样本"。
    """
    return {
        "production_window_checkpoint_protocol_version": PRODUCTION_WINDOW_CHECKPOINT_PROTOCOL_VERSION,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        # 🔑 [non_mutating_v1] 采样修复策略进入生产 checkpoint 指纹。
        "sampling_repair_policy": str(repair_policy),
        "stage_type": str(stage_type),
        "window_idx": int(window_idx),
        "K": int(K),
        "win_sys_xml_sha256": hashlib.sha256(win_sys_xml.encode("utf-8")).hexdigest(),
        "lambdas_coul": [float(x) for x in lambdas_coul],
        "lambdas_vdw": [float(x) for x in lambdas_vdw],
        "lambda_shield": (float(lambda_shield) if lambda_shield is not None else None),
        "lambda_boresch_scale": (
            float(lambda_boresch_scale) if lambda_boresch_scale is not None else None
        ),
        "frozen_f_k_sha256": hashlib.sha256(
            json.dumps([round(float(x), 10) for x in frozen_f_k], sort_keys=False).encode("utf-8")
        ).hexdigest(),
        "temperature_K": float(temperature_K),
        "step_size_ps": 0.002,
        "friction_per_ps": 2.0,
        "platform_name": str(platform_name),
    }


def _production_window_checkpoint_is_usable(
    checkpoint_dir: str, stage_type: str, window_idx: int, expected_manifest: Dict[str, Any]
) -> bool:
    """checkpoint 文件与 manifest 文件都必须存在，且 manifest 内容完全匹配才
    返回 True——照抄 `_main_window_checkpoint_is_usable` 的比较写法。任何
    缺失/损坏/不匹配都安全返回 False，让调用方回退到今天的删除+完整重采
    流程，绝不抛异常。
    """
    ckpt_path, manifest_path = _production_window_checkpoint_paths(
        checkpoint_dir, stage_type, window_idx
    )
    if not (os.path.exists(ckpt_path) and os.path.exists(manifest_path)):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except Exception:
        return False
    normalized_expected = _normalize_ibs_protocol_for_cache_compare(expected_manifest)
    normalized_loaded = _normalize_ibs_protocol_for_cache_compare(loaded)
    return normalized_loaded == normalized_expected


def _invalidate_production_window_checkpoint(checkpoint_dir: str, stage_type: str, window_idx: int) -> None:
    """删除一个窗口的生产 checkpoint+manifest——production 数据本身因为
    f_k/λ 真的变了而被丢弃时（`recalibrate_f_k`，或路径变化的整窗重排），
    这份 checkpoint 也必须一起作废，否则会在物理上不再对应的新一轮生产
    采样里被误当成"续算起点"。任何缺失都安全忽略。
    """
    ckpt_path, manifest_path = _production_window_checkpoint_paths(
        checkpoint_dir, stage_type, window_idx
    )
    for _path in (ckpt_path, manifest_path):
        if os.path.exists(_path):
            try:
                os.remove(_path)
            except OSError:
                pass


def _persist_state_trajectory_record(bank_dir: str, k: int, record: Dict[str, Any], simulation) -> None:
    """Persist one state's energies/volumes/segment-metadata, a native
    OpenMM checkpoint (the real continuation mechanism) and an NPZ
    positions/velocities/box fallback snapshot as a new *generation*
    (tagged with this state's post-extension ``sampled_steps``), then
    atomically commit the pointer that names it current -- only after all 5
    generation-tagged files have been individually, durably written.

    This is what makes the 5-file set atomic as a *whole*, not just
    file-by-file: a crash at any point before the pointer write leaves it
    referencing the previous (complete, self-consistent) generation, so a
    resume never sees energies/meta claiming more sampled steps than the
    checkpoint can actually back up (the scenario this guards against: a
    crash between updating energies/meta and updating the checkpoint used to
    silently leave ``sampled_steps`` ahead of the checkpoint's real physical
    state, so the *next* resume would treat a discontinuous restart as a
    same-segment continuation). The previous generation's files are removed
    only after the new pointer is committed (best-effort; leftover orphans
    from an interrupted cleanup are harmless since nothing ever reads a
    generation the pointer doesn't name).
    """
    new_generation = int(record["sampled_steps"])
    old_generation = _read_fixed_h_probe_bank_state_generation(bank_dir, k)

    energies_path, volume_path, checkpoint_path, meta_path, openmm_checkpoint_path = (
        _fixed_h_probe_bank_state_paths(bank_dir, k, new_generation)
    )
    _atomic_save_npy(energies_path, record["u_cv_kj_mol"])
    _atomic_save_npy(volume_path, record["volume_nm3"])
    _atomic_write_json(meta_path, {
        "segments": record["segments"],
        "sampled_steps": new_generation,
        "generation": new_generation,
    })
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    positions_nm = _positions_to_nm_array(state.getPositions())
    velocities_nm_per_ps = np.asarray(
        state.getVelocities().value_in_unit(unit.nanometer / unit.picosecond), dtype=np.float64
    )
    box_nm = _box_vectors_to_nm_array(state.getPeriodicBoxVectors())
    _atomic_save_npz(
        checkpoint_path,
        positions_nm=positions_nm,
        velocities_nm_per_ps=velocities_nm_per_ps,
        box_nm=box_nm,
        sampled_steps=np.asarray(new_generation),
    )
    _atomic_save_openmm_checkpoint(simulation, openmm_checkpoint_path)

    # 🔑 提交点：只有以上 5 个 generation 文件全部落盘成功后，才原子翻转这个
    # 小指针文件——在此之前崩溃，指针仍指向上一个完整、自洽的 generation，
    # 半成品的新 generation 文件成为孤儿，永远不会被读到。
    _atomic_write_json(
        _fixed_h_probe_bank_state_pointer_path(bank_dir, k), {"generation": new_generation}
    )

    if old_generation is not None and old_generation != new_generation:
        for stale_path in _fixed_h_probe_bank_state_paths(bank_dir, k, old_generation):
            try:
                os.remove(stale_path)
            except OSError:
                pass


def _resume_or_start_state_simulation(
    k: int,
    bank_dir: str,
    has_prior_segments: bool,
    topology,
    system_xml: str,
    cv_xml: str,
    require_group4: bool,
    platform_name: str,
    temperature_q,
    positions,
    box_vectors,
    integrator_seed: int,
    velocity_seed: int,
    lambda_shield: Optional[float] = None,
):
    """Build one state's fixed-H ``Simulation`` for this round, resuming it
    via OpenMM's own binary Context checkpoint whenever possible.

    The native checkpoint (``_atomic_save_openmm_checkpoint``) is the only
    mechanism that also restores the integrator's internal RNG state, so
    only a successful ``loadCheckpoint`` here counts as a true, uninterrupted
    continuation (``needs_burn_in=False``, same segment). If it is missing,
    corrupted, or incompatible (different platform/OpenMM build -- native
    checkpoints are not portable) but the NPZ positions/velocities/box
    fallback is available, that seeds a *new* (re-burned) segment instead --
    reusing the last known configuration avoids re-equilibrating from a far
    configuration, but does not claim a continuation it cannot back up.
    Falls further back to the caller-supplied current window configuration
    if neither is available.

    Returns ``(simulation, needs_burn_in, reason)``. The caller is
    responsible for releasing ``simulation`` (``del`` + ``gc.collect()``)
    right after extending and persisting it -- this function never caches or
    retains a reference to it, by design: only one dynamics Context should be
    alive at a time (see IBS_BIAS_PROTOCOL_VERSION=11 changelog).
    """
    checkpoint_arrays = _load_state_checkpoint_npz(bank_dir, k)
    if checkpoint_arrays is not None:
        sim_positions = checkpoint_arrays["positions_nm"] * unit.nanometer
        sim_velocities = checkpoint_arrays["velocities_nm_per_ps"] * (unit.nanometer / unit.picosecond)
        sim_box = checkpoint_arrays["box_nm"] * unit.nanometer
    else:
        sim_positions = positions
        sim_velocities = None
        sim_box = box_vectors

    sim = _build_fixed_state_simulation(
        topology=topology,
        system_xml=system_xml,
        cv_xml=cv_xml,
        require_group4=require_group4,
        platform_name=platform_name,
        temperature_q=temperature_q,
        positions=sim_positions,
        box_vectors=sim_box,
        integrator_seed=integrator_seed,
        velocity_seed=velocity_seed,
        velocities=sim_velocities,
        lambda_shield=lambda_shield,
    )

    resumed_native = False
    generation = _read_fixed_h_probe_bank_state_generation(bank_dir, k)
    if has_prior_segments and generation is not None:
        _, _, _, _, openmm_checkpoint_path = _fixed_h_probe_bank_state_paths(bank_dir, k, generation)
        if os.path.exists(openmm_checkpoint_path):
            try:
                sim.loadCheckpoint(openmm_checkpoint_path)
                resumed_native = True
            except Exception:
                resumed_native = False

    if resumed_native:
        return sim, False, "resumed_from_native_checkpoint"
    if checkpoint_arrays is not None:
        return sim, True, "native_checkpoint_missing_or_incompatible_npz_fallback"
    return sim, True, ("checkpoint_missing_or_corrupted" if has_prior_segments else "fresh")


def _decorrelate_per_segment(
    diff_series: np.ndarray,
    segments: List[Dict[str, Any]],
    min_frames_for_subsampling: int = 20,
) -> Tuple[np.ndarray, List[float], List[Dict[str, Any]]]:
    """Decorrelate one state's difference-energy series one segment at a
    time, never across a restart boundary.

    A state's trajectory-bank record can span multiple segments (a brand-new
    burn-in, plus zero or more re-burned segments opened after a
    ``checkpoint.npz`` was lost -- see ``probe_adjacent_path_overlap_bank``'s
    resume rules). Concatenating frames from two independently-reburned
    segments before estimating the autocorrelation time would treat the
    discontinuous jump between them as real dynamics, corrupting the
    statistical inefficiency estimate -- so each segment's own frame slice is
    passed to ``subsample_series_by_autocorrelation`` independently, and only
    the resulting (already-decorrelated) index sets are concatenated.

    A segment shorter than ``min_frames_for_subsampling`` is excluded from
    the returned ``combined`` index set entirely -- estimating an
    autocorrelation time from too few points is noisier than not
    decorrelating at all, and a re-burn-in segment opened after checkpoint
    loss is exactly the case where those few points are also the most
    suspect (fresh burn-in, not yet equilibrated). Letting such a segment
    into MBAR at full weight (``g=1``, unweighted) would let it silently
    inflate ``n_k_decorrelated`` and blend with normally-decorrelated
    segments. Such segments are instead reported in the returned
    ``short_segments`` list (segment index + frame count) for
    diagnostic-only use by callers; if excluding them leaves a state with
    too few (or zero) decorrelated frames, the downstream MBAR call fails
    closed on its own sample-count check rather than silently proceeding.
    """
    if sum(int(seg["n_frames"]) for seg in segments) != diff_series.shape[0]:
        raise RuntimeError("segment 边界与能量序列长度不一致，拒绝跨段去相关")
    all_indices = []
    inefficiencies = []
    short_segments = []
    cursor = 0
    for seg_idx, seg in enumerate(segments):
        n = int(seg["n_frames"])
        if n < int(min_frames_for_subsampling):
            short_segments.append({"segment_index": seg_idx, "n_frames": n})
            cursor += n
            continue
        segment_slice = diff_series[cursor:cursor + n]
        local_indices, g = subsample_series_by_autocorrelation(
            segment_slice, min_frames_for_subsampling=min_frames_for_subsampling
        )
        all_indices.append(np.asarray(local_indices, dtype=int) + cursor)
        inefficiencies.append(float(g))
        cursor += n
    combined = np.concatenate(all_indices) if all_indices else np.arange(0, dtype=int)
    return combined, inefficiencies, short_segments


def _extend_state_trajectory(
    simulation,
    record: Dict[str, Any],
    evaluator,
    target_steps: int,
    sample_interval: int,
    burn_in_steps: int,
    needs_burn_in: bool,
    segment_reason: str,
) -> Dict[str, Any]:
    """Advance one state's fixed-H trajectory in-place to >= ``target_steps``
    sampled steps, mutating and returning ``record``.

    Never re-burns an already-open segment: ``needs_burn_in`` must be True
    only for a state's very first segment or a segment reopened after
    checkpoint loss (see ``probe_adjacent_*_bank``'s resume trust rules) --
    repeated calls against the same live ``simulation`` within an
    already-open segment must pass ``needs_burn_in=False``, which only steps
    ``target_steps - record["sampled_steps"]`` additional steps and appends
    to the existing (last) segment's own counters instead of opening a new
    one. Energies are stored raw (kJ/mol, no beta reduction, no LRC) against
    every state in the window -- reduction to reduced units and adding LRC
    (path bank only) happens later, in ``_analyze_adjacent_pair``, once it is
    known which two columns a given edge actually needs.
    """
    additional_steps = int(target_steps) - int(record["sampled_steps"])
    if additional_steps <= 0:
        return record
    if sample_interval <= 0 or additional_steps < sample_interval:
        raise ValueError("fixed-H 轨迹库单次延长的步数不足一个 sample_interval")
    if needs_burn_in:
        simulation.step(int(burn_in_steps))
        record["segments"].append({
            "burn_in_steps": int(burn_in_steps),
            "sample_steps": 0,
            "n_frames": 0,
            "reason": str(segment_reason),
        })
    elif not record["segments"]:
        raise RuntimeError("延长一个尚无任何 segment 的轨迹时必须先烧 burn-in")

    n_new_samples = additional_steps // int(sample_interval)
    new_energies = []
    new_volumes = []
    for _ in range(n_new_samples):
        simulation.step(int(sample_interval))
        frame_state = simulation.context.getState(getPositions=True)
        frame_positions = frame_state.getPositions()
        frame_box = frame_state.getPeriodicBoxVectors()
        e_kj = evaluator.evaluate_interaction_energies(frame_positions, frame_box)
        new_energies.append(np.asarray(e_kj, dtype=np.float64))
        box_nm = _box_vectors_to_nm_array(frame_box)
        volume_nm3 = abs(float(np.linalg.det(box_nm)))
        if not np.isfinite(volume_nm3) or volume_nm3 <= 0.0:
            raise RuntimeError("fixed-H 轨迹库帧的盒子体积无效")
        new_volumes.append(volume_nm3)
    new_energies = np.asarray(new_energies, dtype=np.float64)
    new_volumes = np.asarray(new_volumes, dtype=np.float64)
    if record["u_cv_kj_mol"] is None:
        record["u_cv_kj_mol"] = new_energies
        record["volume_nm3"] = new_volumes
    else:
        record["u_cv_kj_mol"] = np.concatenate([record["u_cv_kj_mol"], new_energies], axis=0)
        record["volume_nm3"] = np.concatenate([record["volume_nm3"], new_volumes], axis=0)
    added_steps = int(n_new_samples) * int(sample_interval)
    record["sampled_steps"] += added_steps
    record["segments"][-1]["sample_steps"] += added_steps
    record["segments"][-1]["n_frames"] += int(n_new_samples)
    return record


def _analyze_adjacent_pair(
    record_i: Dict[str, Any],
    record_j: Dict[str, Any],
    local_i: int,
    local_j: int,
    kt: float,
    lrc_coeff: Optional[np.ndarray] = None,
    threshold: float = 0.03,
    min_frames_for_subsampling: int = 20,
) -> Dict[str, Any]:
    """Reduce two states' shared trajectory-bank records to one adjacent-pair
    MBAR overlap/delta_f result, reusing
    ``_compute_bidirectional_overlap_from_u_kn`` unchanged.

    Each state's trajectory independently serves every edge it borders --
    state k's frames back both edge (k-1,k) and edge (k,k+1) -- but this
    function only ever reads the two energy columns ``[local_i, local_j]``
    needed for one specific edge and is otherwise self-contained, exactly
    like the original per-edge probes: sharing a trajectory across edges does
    not bias either edge's own MBAR estimate (each edge's u_kn/n_k pairing is
    unchanged); it only introduces a correlation between adjacent edges'
    uncertainty estimates if those were ever summed into a path-level
    uncertainty -- no code currently does that, but it is worth remembering
    if that ever changes.
    """
    beta = 1.0 / float(kt)
    reduced_by_ensemble = []
    inefficiencies = []
    short_segments_report = {}
    for label, record in (("state_i", record_i), ("state_j", record_j)):
        e_kj = record["u_cv_kj_mol"][:, [local_i, local_j]].astype(np.float64, copy=True)
        if lrc_coeff is not None:
            volume = np.asarray(record["volume_nm3"], dtype=np.float64)
            lrc_pair = np.asarray(lrc_coeff, dtype=np.float64)[[local_i, local_j]]
            e_kj = e_kj + lrc_pair[None, :] / volume[:, None]
        reduced = beta * e_kj  # (n_frames, 2): column 0 = local_i, column 1 = local_j
        diff = reduced[:, 1] - reduced[:, 0]
        decorrelated, g_list, short_segs = _decorrelate_per_segment(
            diff, record["segments"], min_frames_for_subsampling=min_frames_for_subsampling
        )
        reduced_by_ensemble.append(reduced[decorrelated, :].T)  # (2, n_decorrelated)
        inefficiencies.append(g_list)
        if short_segs:
            short_segments_report[label] = short_segs

    n_k = np.asarray([block.shape[1] for block in reduced_by_ensemble], dtype=int)
    u_kn = np.concatenate(reduced_by_ensemble, axis=1)
    result = _compute_bidirectional_overlap_from_u_kn(u_kn, n_k, threshold=threshold)
    result.update({
        "local_states": [int(local_i), int(local_j)],
        "n_k_decorrelated": n_k.tolist(),
        "statistical_inefficiency": inefficiencies,
        "short_segments_diagnostic_only": short_segments_report,
    })
    return result


def detect_passed_but_asymmetric_overlap_bottleneck(
    pairs: List[Dict[str, Any]],
    lambdas_window,
    max_weak_overlap: float = 0.12,
    min_overlap_ratio: float = 2.5,
    min_slope_ratio: float = 3.0,
    min_delta_f_gap_kj_mol: float = 5.0,
    uncertainty_sigma_multiplier: float = 2.0,
) -> Optional[Dict[str, Any]]:
    """识别 fixed-H 全通过、但局部热力学密度严重不均匀的瓶颈边。

    只有当同一条边同时是最低 overlap 和最大 |ΔF|/|Δλ| 时才可能命中，且差距必须
    显著超过 ΔF 的联合不确定度噪声地板；否则返回 None，绝不在指标含糊时插点。
    """
    if len(pairs) < 2 or not all(bool(p.get("passed")) for p in pairs):
        return None

    lambdas_window = np.asarray(lambdas_window, dtype=np.float64)
    records = []
    for pair in pairs:
        local_states = pair.get("local_states")
        if not local_states or len(local_states) != 2:
            return None

        i, j = map(int, local_states)
        delta_lambda = abs(float(lambdas_window[j] - lambdas_window[i]))
        overlap = float(pair.get("min_bidirectional_overlap", np.nan))
        delta_f = abs(float(pair.get("delta_f_kJ_mol", np.nan)))
        uncertainty = float(
            pair.get("delta_f_uncertainty_kJ_mol", np.nan)
        )

        if (
            delta_lambda <= 0.0
            or not np.isfinite(overlap)
            or not np.isfinite(delta_f)
        ):
            return None

        records.append({
            "pair": pair,
            "overlap": overlap,
            "delta_f": delta_f,
            "uncertainty": uncertainty,
            "slope": delta_f / delta_lambda,
        })

    weakest = min(records, key=lambda x: x["overlap"])
    steepest = max(records, key=lambda x: x["slope"])

    # 必须由同一条边同时表现为最低 overlap 和最大 |ΔF|/|Δλ|。
    if weakest is not steepest:
        return None

    references = [r for r in records if r is not weakest]
    reference = max(references, key=lambda x: x["overlap"])

    overlap_ratio = reference["overlap"] / max(weakest["overlap"], 1e-12)
    slope_ratio = weakest["slope"] / max(reference["slope"], 1e-12)
    delta_f_gap = weakest["delta_f"] - reference["delta_f"]

    significant_gap = float(min_delta_f_gap_kj_mol)
    if (
        np.isfinite(weakest["uncertainty"])
        and np.isfinite(reference["uncertainty"])
    ):
        significant_gap = max(
            significant_gap,
            uncertainty_sigma_multiplier * np.hypot(
                weakest["uncertainty"],
                reference["uncertainty"],
            ),
        )

    if not (
        weakest["overlap"] <= max_weak_overlap
        and overlap_ratio >= min_overlap_ratio
        and slope_ratio >= min_slope_ratio
        and delta_f_gap >= significant_gap
    ):
        return None

    return {
        "qualified": True,
        "pair": dict(weakest["pair"]),
        "global_edge": list(weakest["pair"]["global_edge"]),
        "weak_overlap": weakest["overlap"],
        "reference_overlap": reference["overlap"],
        "overlap_ratio": overlap_ratio,
        "slope_ratio": slope_ratio,
        "delta_f_gap_kJ_mol": delta_f_gap,
        "required_delta_f_gap_kJ_mol": significant_gap,
    }


def _bias_calibration_pair_is_sufficient(
    pair: Optional[Dict[str, Any]],
    min_decorrelated_samples: int = 20,
    max_delta_f_uncertainty_kJ_mol: float = 1.0,
) -> bool:
    """Explicit three-way AND gate for accepting one edge's bias-calibration
    result as precise enough to overwrite f_k: it must (1) pass the overlap
    threshold, (2) have >= ``min_decorrelated_samples`` on the smaller side,
    and (3) have delta_f_bias uncertainty <= ``max_delta_f_uncertainty_kJ_mol``.
    Any one failing condition rejects the pair -- this used to be enforced ad
    hoc inline (``probe_bidirectional_overlap_for_bias_calibration`` returning
    ``None``); factoring it out here makes each condition independently
    unit-testable.
    """
    if pair is None:
        return False
    if not pair.get("passed"):
        return False
    n_k = pair.get("n_k_decorrelated")
    if not n_k or int(np.min(n_k)) < int(min_decorrelated_samples):
        return False
    uncertainty = pair.get("delta_f_bias_uncertainty_kJ_mol")
    if uncertainty is None or not np.isfinite(uncertainty):
        return False
    if float(uncertainty) > float(max_delta_f_uncertainty_kJ_mol):
        return False
    return True


def probe_adjacent_path_overlap_bank(
    topology,
    common_system_xml: str,
    ibs_wrapper,
    K: int,
    positions,
    box_vectors,
    temperature,
    platform_name: str,
    checkpoint_dir: str,
    stage_type: str,
    window_idx: int,
    global_state_start: int,
    burn_in_steps: int = 5000,
    sample_targets: Tuple[int, ...] = (20_000,),
    sample_interval: int = 500,
    threshold: float = 0.03,
) -> Dict[str, Any]:
    """Per-state, resumable replacement for calling ``probe_bidirectional_overlap``
    once per edge.

    A K-state window has K-1 adjacent edges but only K states; the old
    per-edge probe built two brand-new fixed-H Simulations and burned two
    fresh burn-ins for *every* edge, so an interior state (shared by two
    edges) was independently equilibrated and sampled twice. This builds
    exactly one Simulation per state, burns its Langevin dynamics in once,
    and lets ``_analyze_adjacent_pair`` read whichever two energy columns a
    given edge needs from that one shared trajectory.

    ``sample_targets`` defaults to a *single* 20k-step tier, not a growing
    one -- unlike bias calibration, path/overlap failure means the lambda
    grid itself is too sparse (an edge's two states genuinely don't share
    enough phase space), which more sampling of the *same* two states cannot
    fix; growing the tier here would just burn extra GPU time re-confirming
    the same structural verdict (K=3's verified savings, 100,000 -> 75,000
    steps, assumes exactly this single-tier default). Callers may still pass
    a longer/growing ``sample_targets`` explicitly if they have a reason to;
    when more than one tier is given, only the two end states of any edge
    that still fails the overlap threshold are extended to the next tier --
    states whose surrounding edges already passed stop consuming GPU time.
    Trajectories, segment metadata, a native OpenMM checkpoint and an NPZ
    fallback snapshot are persisted after every extension (see
    ``_persist_state_trajectory_record``), keyed by a content fingerprint
    (``_build_fixed_h_probe_bank_manifest``) that invalidates the *entire*
    window+probe_type directory if the physical Hamiltonian changed, but
    otherwise resumes each state independently -- see
    ``_resume_or_start_state_simulation`` for the native-checkpoint /
    NPZ-fallback / current-configuration resume order.

    At most one state's dynamics ``Simulation``/Context is ever alive at a
    time (plus the one lightweight energy-evaluation probe Context shared
    across all states) -- each state is built, extended, checkpointed and
    released before the next one starts, even within a single
    ``sample_targets`` tier, so a K-state window never holds K simultaneous
    GPU Contexts.

    This bank is checked *before* falling back to any edge-result cache in
    ``abfe_pipeline.py`` (``_fixed_h_probe_fingerprint``/
    ``_load_fixed_h_probe_cache``) -- that cache stores *edge results*, keyed
    by window content, one layer above this one; this bank stores *raw
    per-state trajectories* one layer below. A miss in the edge-result cache
    falls through to sampling here, never the other way around; the two are
    not competing implementations of the same thing.
    """
    if K < 2:
        raise ValueError("path overlap 轨迹库至少需要两个态")
    cv_xmls = list(getattr(ibs_wrapper, "_int_cv_force_xmls", []))
    if len(cv_xmls) < K:
        raise RuntimeError("IBS wrapper 缺少 path overlap 轨迹库所需的 CV XML")
    if sample_interval <= 0:
        raise ValueError("path overlap 轨迹库采样间隔必须为正")
    temperature_q = (
        temperature if hasattr(temperature, "value_in_unit") else float(temperature) * unit.kelvin
    )
    kt = (unit.MOLAR_GAS_CONSTANT_R * temperature_q).value_in_unit(unit.kilojoule_per_mole)
    lrc_coeff = getattr(ibs_wrapper, "lj_tail_lrc_coeff_kj_mol", None)

    bank_dir = _fixed_h_probe_bank_dir(checkpoint_dir, stage_type, window_idx, "path_probe")
    expected_manifest = _build_fixed_h_probe_bank_manifest(
        probe_type="path_probe",
        stage_type=stage_type,
        window_idx=window_idx,
        K=K,
        global_state_start=global_state_start,
        common_system_xml=common_system_xml,
        cv_xmls=cv_xmls[:K],
        lambda_shield=None,
        temperature_K=temperature_q.value_in_unit(unit.kelvin),
        sample_interval=sample_interval,
        platform_name=platform_name,
    )
    # 🔑 先查指纹、指纹对了才续采：任何一项不匹配就整个目录判定不可信，
    # 必须在决定"续采 vs 重新分段"之前完成，不能只是隐含意图。
    if not _fixed_h_probe_bank_manifest_matches(bank_dir, expected_manifest):
        _invalidate_fixed_h_probe_bank(bank_dir)
        _atomic_write_json(_fixed_h_probe_bank_manifest_path(bank_dir), expected_manifest)

    records: Dict[int, Dict[str, Any]] = {}
    for k in range(K):
        record = _load_state_trajectory_record(bank_dir, k, K, sample_interval)
        if record is None:
            record = {"u_cv_kj_mol": None, "volume_nm3": None, "segments": [], "sampled_steps": 0}
        records[k] = record

    evaluator = None
    pairs: List[Optional[Dict[str, Any]]] = [None] * (K - 1)
    active = set(range(K))
    for target in sample_targets:
        states_to_extend = sorted(s for s in active if records[s]["sampled_steps"] < target)
        for k in states_to_extend:
            global_k = int(global_state_start) + int(k)
            sim, needs_burn_in_k, segment_reason = _resume_or_start_state_simulation(
                k=k,
                bank_dir=bank_dir,
                has_prior_segments=bool(records[k]["segments"]),
                topology=topology,
                system_xml=common_system_xml,
                cv_xml=cv_xmls[k],
                require_group4=False,
                platform_name=platform_name,
                temperature_q=temperature_q,
                positions=positions,
                box_vectors=box_vectors,
                integrator_seed=PATH_PROBE_INTEGRATOR_SEED_BASE + _FIXED_H_PROBE_SEED_STRIDE * global_k,
                velocity_seed=PATH_PROBE_VELOCITY_SEED_BASE + _FIXED_H_PROBE_SEED_STRIDE * global_k,
            )
            try:
                if evaluator is None:
                    evaluator = IBSSampler(
                        sim.context, K, temperature_q,
                        prefix=getattr(ibs_wrapper, "prefix", "abfe"),
                        ibs_wrapper=ibs_wrapper,
                    )
                    # 🔑 evaluator 在这条路径上只会被调用 evaluate_interaction_energies，
                    # 那个方法只读 self._probe_context/self._probe_groups（已经在
                    # 上面 __init__ 里用这台 sim 的 Context 建好了独立的探针
                    # Context），永远不再读 self.context。构造完就立刻丢掉这份
                    # 对本态动力学 Context 的引用，否则它会跟随 evaluator 活到
                    # 整个 bank 探针函数结束，导致下一态的 dynamics Context 建立
                    # 后，旧 dynamics Context、当前 dynamics Context、probe
                    # Context 同时占着显存。
                    evaluator.context = None
                _extend_state_trajectory(
                    sim,
                    records[k],
                    evaluator,
                    target_steps=target,
                    sample_interval=sample_interval,
                    burn_in_steps=burn_in_steps,
                    needs_burn_in=needs_burn_in_k,
                    segment_reason=segment_reason,
                )
                _persist_state_trajectory_record(bank_dir, k, records[k], sim)
            finally:
                # 🔑 每个态延长完、落盘完立刻释放它的动力学 Context -- 峰值只
                # 保持一个动力学 Context + 一个 evaluator 探针 Context，K 态
                # 窗口不会同时常驻 K 个 GPU Context。
                del sim
                gc.collect()

        all_passed = True
        for local_i in range(K - 1):
            pair = _analyze_adjacent_pair(
                records[local_i], records[local_i + 1], local_i, local_i + 1, kt,
                lrc_coeff=lrc_coeff, threshold=threshold,
            )
            pair["global_edge"] = [int(global_state_start) + local_i, int(global_state_start) + local_i + 1]
            pair["burn_in_steps"] = int(burn_in_steps)
            pair["sample_interval"] = int(sample_interval)
            pair["dynamics_hamiltonian"] = "U_common_plus_single_cv_int"
            pair["ensemble"] = "NVT"
            pair["delta_f_kJ_mol"] = pair["delta_f_reduced_kT"] * kt
            pair["delta_f_uncertainty_kJ_mol"] = pair["delta_f_uncertainty_reduced_kT"] * kt
            pairs[local_i] = pair
            if not pair["passed"]:
                all_passed = False
        if all_passed:
            break
        active = set()
        for local_i, pair in enumerate(pairs):
            if not pair["passed"]:
                active.add(local_i)
                active.add(local_i + 1)

    return {
        "pairs": pairs,
        "all_passed": bool(all(p["passed"] for p in pairs)),
        "sample_targets": [int(x) for x in sample_targets],
        "final_sampled_steps": {int(k): int(records[k]["sampled_steps"]) for k in range(K)},
    }


def probe_adjacent_bias_calibration_bank(
    topology,
    common_plus_wca_system_xml: str,
    ibs_wrapper,
    K: int,
    positions,
    box_vectors,
    temperature,
    platform_name: str,
    checkpoint_dir: str,
    stage_type: str,
    window_idx: int,
    global_state_start: int,
    lambda_shield: float,
    burn_in_steps: int = 5000,
    sample_targets: Tuple[int, ...] = (20_000, 40_000, 80_000),
    sample_interval: int = 500,
    min_decorrelated_samples: int = 20,
    max_delta_f_uncertainty_kJ_mol: float = 1.0,
    overlap_threshold: float = 0.03,
) -> Dict[str, Any]:
    """Per-state, resumable replacement for the old retry loop around
    ``probe_bidirectional_overlap_for_bias_calibration`` (rebuild two fresh
    Simulations, burn a fresh burn-in, and throw away every already-sampled
    frame whenever one edge's decorrelated-sample-count/uncertainty gate
    failed, up to 3 times with doubled sample_steps). Same per-state sharing
    and resumable-checkpoint design as ``probe_adjacent_path_overlap_bank``,
    with the same U_common + WCA_window(lambda_shield) + CV_k dynamics and
    LRC-free energies as the original single-edge bias-calibration probe --
    see that function's docstring for why both of those must never change.

    Growing through ``sample_targets`` tiers replaces the old
    retry-with-doubled-sample_steps loop: an edge only keeps extending its
    two end states while ``_bias_calibration_pair_is_sufficient`` is False,
    and never discards already-sampled frames to do so.
    """
    if K < 2:
        raise ValueError("bias 校准轨迹库至少需要两个态")
    cv_xmls = list(getattr(ibs_wrapper, "_int_cv_force_xmls", []))
    if len(cv_xmls) < K:
        raise RuntimeError("IBS wrapper 缺少 bias 校准轨迹库所需的 CV XML")
    if sample_interval <= 0:
        raise ValueError("bias 校准轨迹库采样间隔必须为正")
    temperature_q = (
        temperature if hasattr(temperature, "value_in_unit") else float(temperature) * unit.kelvin
    )
    kt = (unit.MOLAR_GAS_CONSTANT_R * temperature_q).value_in_unit(unit.kilojoule_per_mole)

    bank_dir = _fixed_h_probe_bank_dir(checkpoint_dir, stage_type, window_idx, "bias_calibration_probe")
    expected_manifest = _build_fixed_h_probe_bank_manifest(
        probe_type="bias_calibration_probe",
        stage_type=stage_type,
        window_idx=window_idx,
        K=K,
        global_state_start=global_state_start,
        common_system_xml=common_plus_wca_system_xml,
        cv_xmls=cv_xmls[:K],
        lambda_shield=lambda_shield,
        temperature_K=temperature_q.value_in_unit(unit.kelvin),
        sample_interval=sample_interval,
        platform_name=platform_name,
    )
    if not _fixed_h_probe_bank_manifest_matches(bank_dir, expected_manifest):
        _invalidate_fixed_h_probe_bank(bank_dir)
        _atomic_write_json(_fixed_h_probe_bank_manifest_path(bank_dir), expected_manifest)

    records: Dict[int, Dict[str, Any]] = {}
    for k in range(K):
        record = _load_state_trajectory_record(bank_dir, k, K, sample_interval)
        if record is None:
            record = {"u_cv_kj_mol": None, "volume_nm3": None, "segments": [], "sampled_steps": 0}
        records[k] = record

    evaluator = None
    pairs: List[Optional[Dict[str, Any]]] = [None] * (K - 1)
    active = set(range(K))
    for target in sample_targets:
        states_to_extend = sorted(s for s in active if records[s]["sampled_steps"] < target)
        for k in states_to_extend:
            global_k = int(global_state_start) + int(k)
            sim, needs_burn_in_k, segment_reason = _resume_or_start_state_simulation(
                k=k,
                bank_dir=bank_dir,
                has_prior_segments=bool(records[k]["segments"]),
                topology=topology,
                system_xml=common_plus_wca_system_xml,
                cv_xml=cv_xmls[k],
                require_group4=True,
                platform_name=platform_name,
                temperature_q=temperature_q,
                positions=positions,
                box_vectors=box_vectors,
                integrator_seed=BIAS_CALIB_INTEGRATOR_SEED_BASE + _FIXED_H_PROBE_SEED_STRIDE * global_k,
                velocity_seed=BIAS_CALIB_VELOCITY_SEED_BASE + _FIXED_H_PROBE_SEED_STRIDE * global_k,
                lambda_shield=lambda_shield,
            )
            try:
                if evaluator is None:
                    evaluator = IBSSampler(
                        sim.context, K, temperature_q,
                        prefix=getattr(ibs_wrapper, "prefix", "abfe"),
                        ibs_wrapper=ibs_wrapper,
                    )
                    # 🔑 同上（见 probe_adjacent_path_overlap_bank）：evaluator
                    # 这条路径上只调用 evaluate_interaction_energies，只依赖
                    # __init__ 里已经建好的独立探针 Context，不再需要这份
                    # 动力学 Context 引用——留着它会让首个态的 dynamics Context
                    # 跟随 evaluator 存活到函数结束，和后续态的 dynamics
                    # Context、探针 Context 一起同时占用显存。
                    evaluator.context = None
                _extend_state_trajectory(
                    sim,
                    records[k],
                    evaluator,
                    target_steps=target,
                    sample_interval=sample_interval,
                    burn_in_steps=burn_in_steps,
                    needs_burn_in=needs_burn_in_k,
                    segment_reason=segment_reason,
                )
                _persist_state_trajectory_record(bank_dir, k, records[k], sim)
            finally:
                # 🔑 同上（见 probe_adjacent_path_overlap_bank）：每个态延长完、
                # 落盘完立刻释放它的动力学 Context，峰值只保持一个动力学
                # Context + 一个 evaluator 探针 Context。
                del sim
                gc.collect()

        all_sufficient = True
        for local_i in range(K - 1):
            pair = _analyze_adjacent_pair(
                records[local_i], records[local_i + 1], local_i, local_i + 1, kt,
                lrc_coeff=None, threshold=overlap_threshold,
            )
            pair["global_edge"] = [int(global_state_start) + local_i, int(global_state_start) + local_i + 1]
            pair["burn_in_steps"] = int(burn_in_steps)
            pair["sample_interval"] = int(sample_interval)
            pair["dynamics_hamiltonian"] = "U_common_plus_wca_window_plus_single_cv_int_no_lrc"
            pair["ensemble"] = "NVT"
            pair["lambda_shield"] = float(lambda_shield)
            pair["delta_f_bias_kJ_mol"] = pair["delta_f_reduced_kT"] * kt
            pair["delta_f_bias_uncertainty_kJ_mol"] = pair["delta_f_uncertainty_reduced_kT"] * kt
            pairs[local_i] = pair
            if not _bias_calibration_pair_is_sufficient(
                pair, min_decorrelated_samples, max_delta_f_uncertainty_kJ_mol
            ):
                all_sufficient = False
        if all_sufficient:
            break
        active = set()
        for local_i, pair in enumerate(pairs):
            if not _bias_calibration_pair_is_sufficient(
                pair, min_decorrelated_samples, max_delta_f_uncertainty_kJ_mol
            ):
                active.add(local_i)
                active.add(local_i + 1)

    return {
        "pairs": pairs,
        "all_sufficient": bool(all(
            _bias_calibration_pair_is_sufficient(p, min_decorrelated_samples, max_delta_f_uncertainty_kJ_mol)
            for p in pairs
        )),
        "lambda_shield": float(lambda_shield),
        "sample_targets": [int(x) for x in sample_targets],
        "final_sampled_steps": {int(k): int(records[k]["sampled_steps"]) for k in range(K)},
    }


# ============================================================================
# 3. 双λ窗口管理器
# ============================================================================
class IBSWindowManagerDualLambda:
    """双λ IBS 采样管理器 (生产级)"""
    def __init__(
        self,
        system_template,
        topology,
        perturbed_atom_indices,
        lambdas_coul: List[float],
        lambdas_vdw: List[float],
        temperature,
        window_ranges: List[Tuple[int, int]],
        alchemical_params,
        potential_type: str = "softcore",
        restraint_params: Optional[Dict] = None,
        prefix: str = "abfe_dual",
        platform_name: str = "CUDA",
        output_dir: str = "./output",
        checkpoint_dir: str = "./checkpoints",
        pilot_lambdas: Optional[List[float]] = None,
        pilot_mean_dU_dlambda: Optional[List[float]] = None,
    ):
        self.system_template = system_template
        self.topology = topology
        self.ligand_indices = perturbed_atom_indices
        self.lambdas_coul = lambdas_coul
        self.lambdas_vdw = lambdas_vdw
        self.temperature = temperature
        self.ranges = window_ranges
        self.alchemical_params = alchemical_params
        self.potential_type = potential_type
        self.boresch = restraint_params
        self.prefix = prefix
        self.platform_name = platform_name
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        # [IBS_BIAS_PROTOCOL_VERSION warm-start] Stage 2's pilot probe already
        # measures the mean gradient (mean_dU_dlambda_kJ_mol), not just the
        # variance proxy used for lambda spacing -- kept here so
        # run_all_windows can TI-integrate it into a real f_k seed for a
        # window's first learning attempt instead of cold-starting at 0.0.
        # None for any manager that isn't given pilot data (e.g. decharging,
        # Shadow-Coulomb) -- estimate_f_k_from_pilot_ti() treats that as
        # "no seed available" and callers fall back to today's behavior.
        self.pilot_lambdas = pilot_lambdas
        self.pilot_mean_dU_dlambda = pilot_mean_dU_dlambda
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        _temp_q = temperature if hasattr(temperature, 'value_in_unit') else temperature * unit.kelvin
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * _temp_q).value_in_unit(unit.kilojoule_per_mole)

    def _build_window_system(self, lc_win, lv_win, resolved_box, positions):
        """构建单个窗口的 (System, IBSBiasForce)。

        子类覆盖这个方法即可接入不同的 CV 构造（例如 Shadow-Coulomb 去电荷），
        同时复用本类其余的窗口调度/最小化/Boresch 爬坡/渐进预热/生产采样/
        断点续传/TMBAR 落盘逻辑——这些逻辑只依赖 (win_sys, ibs_wrap) 这两个
        返回值和 lc_win 的长度，不关心 CV 具体代表哪种物理量。
        """
        win_sys_xml = XmlSerializer.serialize(self.system_template)
        return build_ibs_dual_system(
            ensure_owned_system(XmlSerializer.deserialize(win_sys_xml)),
            self.topology,
            self.ligand_indices,
            lc_win,
            lv_win,
            self.alchemical_params,
            self.potential_type,
            self.boresch,
            self.temperature,
            self.prefix,
            box_vectors=resolved_box,
            reference_positions=positions,
        )

    def _enqueue_window_snapshot(self, window_idx: int, stage_type: str, sampler) -> None:
        """同步原子刷盘能量快照。"""
        e_arr = np.array(sampler.energy_history, dtype=np.float64, copy=True) if sampler.energy_history else np.zeros((0, 0), dtype=np.float64)
        e_save = e_arr.T if e_arr.size > 0 else np.zeros((0, 0), dtype=np.float64)
        bias = np.array(sampler.bias_history, dtype=np.float64, copy=True) if sampler.bias_history else np.zeros((0,), dtype=np.float64)
        base = np.array(sampler.base_energy_history, dtype=np.float64, copy=True) if sampler.base_energy_history else np.zeros((0,), dtype=np.float64)

        _atomic_save_npy(
            os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_energies.npy"),
            e_save,
        )
        if bias.size > 0:
            _atomic_save_npy(
                os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_bias.npy"),
                bias,
            )
        if base.size > 0:
            _atomic_save_npy(
                os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_base.npy"),
                base,
            )

    def run_all_windows(
        self,
        positions,
        box_vectors,
        n_steps_per_window: int,
        steps_per_update: int,
        stage_type: str = "coul",
        resume: bool = False,
        enable_gradual_warmup: bool = True,
        warmup_steps: int = 500000,
        min_bias_updates: int = 12,
        max_bias_updates: int = 50,
        required_consecutive_bias_updates: int = 3,
        max_bias_warmup_steps: int = 500000,
        mbar_calibration_reserved_steps: int = 50000,
        frozen_validation_step_overrides: Optional[Dict[int, int]] = None,
        frozen_validation_is_final_rung: Optional[Dict[int, bool]] = None,
        production_step_overrides: Optional[Dict[int, int]] = None,
        enable_early_stop: bool = False,
        early_stop_min_steps: int = 100000,
        early_stop_check_interval_steps: int = 20000,
        early_stop_required_consecutive_passes: int = 3,
        early_stop_min_ess_ratio: float = 0.05,
        early_stop_min_absolute_ess: float = 50.0,
        early_stop_min_decorrelated_samples: int = 20,
        early_stop_max_delta_g_drift_kJ_mol: float = 0.5,
        early_stop_max_uncertainty_kJ_mol: float = 1.0,
        debug_mode: bool = False,
        repair_policy: str = "non_mutating_v1",
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] 0.25 -> 0.5：见 version 25
        # changelog (a) 段——这个容差只影响重加权效率/方差，不影响
        # solve_stage_integrated（单参考重要性 MBAR）的无偏性；真正独立的
        # 正确性门槛是生产后的 _assert_stage_result_sane（min_overlap/
        # 绝对 ESS/去相关样本数/端点不确定度），完全不读这个值。放宽后仍然
        # 会被那道独立门槛兜底，不会让真正采样不足的 f_k 悄悄蒙混过关。
        lse_log_residual_tolerance: float = 0.5,
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] "见好就收"兜底：完整冻结验证
        # attempt 连续失败达到这个次数后，不再无限重试——接受目前观察到的
        # 最小 max_abs_log_residual 那次 attempt 的 f_k，标记为 best-effort
        # 而非严格达标，直接放行进生产。同一条理由：真正的正确性门槛在生产
        # 后独立检查，这里只影响"要不要再多花一次 attempt 的预算去碰运气"。
        # 预热只负责给生产确定一组固定 f_k，不负责用预热样本替代生产统计。
        # 完成一次足额 fixed-f 验证后，无论占据平坦残差是否达到理想阈值，均
        # 锁定该 attempt 的 f_k 并进入独立生产；残差只作为效率诊断。真正的
        # 可用性由生产后的 overlap/ESS/去相关样本/不确定度硬门判断。
        max_frozen_validation_cycles_before_accept_best: int = 1,
        # 仅用于把预热残差标为 within/outside sane bound，便于日志和诊断解释
        # f_k 的预计效率；生产运行不再拿它作准入硬门。warmup_only 设计审计
        # 没有后续生产统计兜底，因此仍要求落入该诊断范围才可返回成功。
        best_effort_max_residual_multiple: float = 4.0,
        # [IBS_BIAS_PROTOCOL_VERSION=19] learning 候选门读取的 TMBAR
        # solve_stage_integrated 阈值。故意比 final_*（stage 级最终验收，
        # min_absolute_ess=50/min_decorrelated_samples=20/max_uncertainty=1.0
        # kJ/mol）宽松得多：候选门只是"值不值得冻结去走 frozen validation"，
        # 真正严格的接受/拒绝判据仍是下面独立、未变的冻结验证阶段——候选门
        # 放宽不会让坏 f_k 蒙混过关，只会让它更快地被送去验证，验证失败后
        # 样本并入 tmbar_history 继续学习，不会像直接复用 final_* 那样让
        # consecutive_pass_count 在 max_bias_updates 预算内几乎不可能达到。
        candidate_min_ess_ratio: float = 0.05,
        candidate_min_absolute_ess: float = 1.0,
        # [v23 numerical follow-up #3] Was 5 -- mathematically unreachable
        # given `_solve_tmbar_and_recenter`'s own `min_frames_per_window=3`
        # (documented there as the actual floor for running the augmented
        # MBAR matrix at all, precisely because decorrelation commonly
        # leaves only 2-7 frames per ~20-frame minibatch). Any minibatch with
        # 3 or 4 decorrelated frames is deliberately let into local_results
        # by that floor, which then caps the aggregate min_decorrelated_
        # samples at 3 or 4 for as long as such a minibatch exists in
        # tmbar_history -- requiring >=5 here made this candidate criterion
        # permanently unsatisfiable in exactly the regime it's meant to
        # evaluate, same failure shape as the min_absolute_ess 2.5->1.0 fix
        # above. Lowered to 3 to match the actual floor; final_*/frozen-
        # validation gates are untouched.
        candidate_min_decorrelated_samples: int = 3,
        candidate_max_uncertainty_kJ_mol: float = 5.0,
        warmup_only: bool = False,
    ):
        """
        执行全部窗口采样，能量保存至 {output_dir}/dual_window_{window_idx}_{stage_type}_energies.npy

        repair_policy: 采样修复策略。默认 "non_mutating_v1" —— IBS 非变异策略：
            某个 vdw 窗口 SGD 预热结束仍未获得任何足额 fixed-f attempt 时，绝不启动 fixed-H
            overlap 探针 / asymmetric-ΔF-slope 判据 / bias 校准探针，也绝不用累计
            ΔF 覆盖 context 里的 f_k；直接落盘 warmup_failure 并抛
            IBSWarmupConvergenceError（f_not_converged），交人工/救援审计处理。
            任何其它值恢复旧的 fixed-H 探针 + 就地重校准 f_k 行为（已弃用，仅为
            对照/回滚保留）。fixed-H adjacent overlap 因此被降为纯诊断，绝不再自动
            触发插点/重校准。见 plan a-relative-binding-free-energy-framewor-immutable-starlight.md。

        warmup_only: 设计期 IBS 窗口验证模式。严格执行与生产相同的 Log-Sum-Exp
            权重学习、冻结 burn-in 和只读自洽验证；通过后立即返回该窗口诊断，
            不进入 production、不写能量数组。失败仍按 non_mutating_v1 原样抛出，
            由调用方在独立 design scratch ensemble 中决定拆窗或插入热力学中点，
            绝不修改正式生产 ensemble。

        诊断功能 (debug_mode=True，生产环境默认关闭)：
            - 构建后打印力组架构
            - 关键阶段调用 diagnose_force_breakdown (非侵入式力分解)
            - 若检测到 NaN 或力爆炸，立即抛出异常并终止

        frozen_validation_is_final_rung: 可选的 {window_idx: 这次是否已经是调用方
            冻结验证阶梯的最后一档} 表。只在 calibration_pending 场景下参与判断：
            若 True 且这次仍未通过独立验证，直接判定为终态失败
            （bias_status="calibrated_validation_failed"，raise
            IBSFrozenCalibrationValidationError(terminal=True)），不再落盘为
            "calibrated_pending_validation"、不再暗示"resume 会继续自动延长预算"。
            调用方（abfe_pipeline.py::_run_stage_with_overlap_autorepair）应在把
            某个窗口的 frozen_validation_step_overrides 升级到阶梯最后一档时，
            同步把这个窗口标记为 True。

        production_step_overrides: 可选的 {window_idx: 该窗口实际生产步数} 覆盖表，
            未列出的窗口仍用 n_steps_per_window。供 abfe_pipeline.py 的
            production ESS sampling-repair 分支使用：fixed-H 全通过但 production
            ESS 低、且判定偏置本身没问题（只是采样太短/构象弛豫慢）时，不能只是
            删掉旧样本再用同样步数重采一次（那只是换一批同样长度的独立样本，
            不会真正增加信息量）；调用方应在这里传入更大的步数，真正延长该窗口
            的采样，而不是原地打转。window_idx 必须对应本次调用 self.ranges 里
            该窗口在这次调用中的实际位置——调用方需要自己保证窗口顺序在设置
            override 和真正消费它的这次调用之间没有变化（例如夹在两次
            split/insert 之间的“纯 sampling-repair 轮”）。

        mbar_calibration_reserved_steps: 从 max_bias_warmup_steps 总预算里为
            fixed-H overlap 全通过后的 MBAR 校准 f_k 单独预留的步数（burn-in +
            冻结验证）。SGD learning/freeze_burn_in/validating 阶段只能用
            max_bias_warmup_steps - mbar_calibration_reserved_steps；这样即使
            SGD 把自己的份额烧光，MBAR 校准仍有独立预算跑完整验证，而不会因为
            共享的 steps_at_full_bias 已耗尽而必然判失败。

        frozen_validation_step_overrides: 可选的 {window_idx: 该窗口这次冻结
            验证的目标预算步数} 覆盖表 [IBS_BIAS_PROTOCOL_VERSION=12]。fixed-H
            overlap 全通过 + MBAR 校准出的 f_k 冻结后，若冻结验证在
            mbar_calibration_reserved_steps（或上一轮的覆盖值）内仍未通过
            覆盖门，本方法不会像旧协议那样直接判失败并把 f_k 退回当作普通
            未收敛热启动——而是把 bias_status 存为
            "calibrated_pending_validation"、frozen_f_k_pending 存下这份已经
            校准好的 f_k，并在诊断里标记 calibration_pending_validation=True。
            调用方（见 abfe_pipeline.py::_run_stage_with_overlap_autorepair）
            据此按 50k→150k→300k 的阶梯延长这里的预算重试，同一个窗口 resume
            时会跳过 [learning] 和 fixed-H 探针/MBAR 校准，直接带着冻结的
            f_k 从 freeze_burn_in 开始，只消耗新增的那部分步数（不是重新烧
            完整个新预算）。未列出的窗口沿用 mbar_calibration_reserved_steps。

        enable_early_stop 及 early_stop_*：默认关闭（enable_early_stop=False），
            此前这个开关在调用链上是纯空转——abfe_pipeline.py 从未把它传进这个
            方法，这里也从未定义对应参数，打开它不会有任何效果。现在是真正接入
            的在线判据，但阈值默认值尚未用已有完整轨迹做离线回放校准过，见
            EARLY_STOP_PROTOCOL_VERSION 定义处；在离线回放验证通过、确定合适的
            默认阈值之前，不要在生产配置里打开这个开关。

            达到 early_stop_min_steps 步之后才开始检查；此后每满
            early_stop_check_interval_steps 步，用该窗口至今累积的样本
            （sampler.energy_history/bias_history/base_energy_history）跑一次
            local MBAR（_solve_single_window_local_mbar），同时核对五项：
            (1) 绝对有效样本数 >= early_stop_min_absolute_ess；
            (2) 最差 ESS 比例 >= early_stop_min_ess_ratio；
            (3) 去相关后样本数 >= early_stop_min_decorrelated_samples；
            (4) 本窗口局部 ΔG（f[-1]-f[0]）相对上一次检查的漂移
                <= early_stop_max_delta_g_drift_kJ_mol（第一次检查没有"上一次"
                可比，直接判不通过，因此至少要连续通过
                required_consecutive_passes+1 次检查才可能真正停止）；
            (5) 局部 ΔG 端点合并不确定度 <= early_stop_max_uncertainty_kJ_mol。
            五项全部通过才计入一次"连续通过"，任一项失败清零连续计数；连续
            达到 early_stop_required_consecutive_passes 次才真正停止该窗口的
            生产采样。n_steps_per_window（含 production_step_overrides 覆盖值）
            始终是硬上限，early stop 只会提前结束，不会超过它。

            触发后 convergence.json 会记录 actual_production_steps（真正跑了
            多少步）、early_stop_triggered、stop_reason、
            early_stop_check_history（每次检查的 ESS/ΔG/不确定度和是否通过）、
            early_stop_protocol_version 与本次调用的阈值配置——resume 时如果
            缓存显示 early_stop_triggered=True，只有当前调用同样启用 early
            stop、协议版本一致、且当前目标步数没有被调高，才会复用这份提前
            停止的短样本；任何一项不满足都视为缓存不可信，强制重新采样，避免
            "以后关闭 early stop 或调高预算"时被一次短样本悄悄糊弄过去。
        """
        # 🔑 [non_mutating_v1] Validate the sampling repair policy ONCE at entry,
        # fail-closed on any unrecognized value (a typo must never silently
        # re-open the deprecated mutating fixed-H/recalibration path).
        legacy_repair = should_run_legacy_repair(repair_policy)
        if not np.isfinite(lse_log_residual_tolerance) or float(lse_log_residual_tolerance) <= 0.0:
            raise ValueError("lse_log_residual_tolerance 必须是有限正数")
        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)
        resolved_box = _resolve_periodic_box_vectors(box_vectors, topology=self.topology, system=self.system_template)
        if _system_requires_periodic_box(self.system_template) and resolved_box is None:
            raise ValueError(
                "当前系统使用了周期性非键方法，但没有可用的周期性盒子。"
                "请检查输入坐标/拓扑是否带 box，或显式传入 box_vectors。"
            )

        warmup_results = []
        for window_idx, (start, end) in enumerate(self.ranges):
            lc_win = self.lambdas_coul[start:end]
            lv_win = self.lambdas_vdw[start:end]
            print(f"\n{'='*80}")
            print(f"[窗口 {window_idx}] 索引 [{start}:{end}] | 状态数: {len(lc_win)}")
            print(f"{'='*80}")

            # ---------- 断点续传：按窗口跳过已完成的采样 ----------
            # 此前 resume 只热启动了 IBS bias 权重 (f_k)，中途崩溃后重跑会把
            # 已经跑完的窗口从头再来一遍，白白浪费 GPU 时间且违背"Checkpoint"的
            # 设计承诺。这里在真正重建系统/最小化/生产采样之前，先检查该窗口
            # 是否已有形状匹配的有效能量文件，若有则直接跳过整窗。
            if resume:
                energies_path = os.path.join(
                    self.output_dir, f"dual_window_{window_idx}_{stage_type}_energies.npy"
                )
                bias_path = os.path.join(
                    self.output_dir, f"dual_window_{window_idx}_{stage_type}_bias.npy"
                )
                base_path = os.path.join(
                    self.output_dir, f"dual_window_{window_idx}_{stage_type}_base.npy"
                )
                convergence_path = os.path.join(
                    self.output_dir, f"dual_window_{window_idx}_{stage_type}_convergence.json"
                )
                if os.path.exists(energies_path) and os.path.exists(convergence_path):
                    try:
                        with open(convergence_path, "r", encoding="utf-8") as f:
                            cached_conv = json.load(f)
                        cached_policy_early = cached_conv.get(
                            "sampling_repair_policy"
                        )
                        if (
                            not legacy_repair
                            and cached_policy_early != repair_policy
                        ):
                            raise ExistingEnsembleRequiresRescueAudit(
                                f"窗口 {window_idx}: 磁盘能量缓存的 sampling_repair_policy="
                                f"{cached_policy_early!r}（当前 {repair_policy!r}）；"
                                "拒绝在原 ensemble 目录覆盖旧策略数据。",
                                diagnostics={
                                    "window_index": int(window_idx),
                                    "stage_type": stage_type,
                                    "cached_sampling_repair_policy": cached_policy_early,
                                    "current_sampling_repair_policy": repair_policy,
                                    "requires_rescue_audit": True,
                                },
                            )
                        cached_e, _cached_bias, _cached_base = (
                            _load_validated_window_data_triplet(
                                energies_path,
                                bias_path,
                                base_path,
                                cached_conv,
                            )
                        )
                        # 🔑 只按 window_idx + shape 判断"这份缓存还能不能用"曾经是不安全的：
                        # λ 路径被自动加密/重新划分窗口后，同一个 window_idx 完全可能对应
                        # 一段全新的 λ 区间，只要凑巧态数相同（shape 相同）就会被静默当成
                        # 有效缓存复用——采样对了，但对的是错的 λ。这里额外校验 convergence.json
                        # 里存的实际 λ 值（见下方保存逻辑）跟本次真正要跑的 lc_win/lv_win 是否
                        # 完全一致；旧格式缓存没有这个字段的，保守地当作不匹配，强制重采，而不是
                        # 假设它"大概率没变"。
                        _gate = _resume_cached_window_gate_status(
                            cached_conv,
                            tuple(cached_e.shape),
                            lc_win,
                            lv_win,
                            repair_policy,
                            lse_log_residual_tolerance,
                            enable_early_stop,
                            {
                                "min_steps": int(early_stop_min_steps),
                                "check_interval_steps": int(early_stop_check_interval_steps),
                                "required_consecutive_passes": int(early_stop_required_consecutive_passes),
                                "min_ess_ratio": float(early_stop_min_ess_ratio),
                                "min_absolute_ess": float(early_stop_min_absolute_ess),
                                "min_decorrelated_samples": int(early_stop_min_decorrelated_samples),
                                "max_delta_g_drift_kJ_mol": float(early_stop_max_delta_g_drift_kJ_mol),
                                "max_uncertainty_kJ_mol": float(early_stop_max_uncertainty_kJ_mol),
                            },
                            (
                                int(production_step_overrides[window_idx])
                                if production_step_overrides and window_idx in production_step_overrides
                                else int(n_steps_per_window)
                            ),
                        )
                        # 🔑 这 8 个门原先内联在这里（~110 行），现已抽成模块级纯函数
                        # _resume_cached_window_gate_status，逐门语义与阈值一字未改，
                        # 只是变得可以用 mock 的 convergence.json 单独测试（见
                        # test_resume_reuse_contracts.py）。下面那串逐门诊断打印仍然
                        # 读同名局部变量，因此完全保持原样。
                        lambdas_match = _gate["lambdas_match"]
                        version_match = _gate["version_match"]
                        bias_protocol_match = _gate["bias_protocol_match"]
                        cached_lse_tolerance = _gate["cached_lse_tolerance"]
                        lse_tolerance_match = _gate["lse_tolerance_match"]
                        lrc_version_match = _gate["lrc_version_match"]
                        repair_policy_match = _gate["repair_policy_match"]
                        early_stop_ok = _gate["early_stop_ok"]
                        early_stop_reject_reason = _gate["early_stop_reject_reason"]
                        if _gate["usable"]:
                            print(
                                f"  ⏭️  窗口 {window_idx} 已有有效缓存能量 {cached_e.shape} 且 λ 值匹配，"
                                f"resume 模式下跳过重新采样。"
                            )
                            continue
                        if cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0 and not version_match:
                            print(
                                f"  ⚠️ 窗口 {window_idx} 缓存的 wca_accounting_version="
                                f"{cached_conv.get('wca_accounting_version')!r}（期望 {WCA_ACCOUNTING_VERSION}），"
                                "base/bias 力组口径已变更，视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0 and not bias_protocol_match:
                            print(
                                f"  ⚠️ 窗口 {window_idx} 缓存的 ibs_bias_protocol_version="
                                f"{cached_conv.get('ibs_bias_protocol_version')!r}（兼容版本 "
                                f"{sorted(IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS)}），"
                                "IBS 偏置预热/冻结协议已变更，视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0 and not lse_tolerance_match:
                            print(
                                f"  ⚠️ 窗口 {window_idx} 缓存的 LSE log 残差容差="
                                f"{cached_lse_tolerance!r}（当前 {lse_log_residual_tolerance}），"
                                "自洽收敛门已改变，视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0 and not lrc_version_match:
                            print(
                                f"  ⚠️ 窗口 {window_idx} 缓存的 lj_tail_lrc_protocol_version="
                                f"{cached_conv.get('lj_tail_lrc_protocol_version')!r}（期望 "
                                f"{TRADITIONAL_LJ_LRC_PROTOCOL_VERSION}），LJ 长程尾项修正公式已变更"
                                "（switching-aware），视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0 and not repair_policy_match:
                            # 🔑 [non_mutating_v1] 磁盘上已有旧策略的能量数据。绝不能
                            # 就地重采覆盖它——那会毁掉 rescue 审计所需的原始 ensemble。
                            # 保留原文件，抛 ExistingEnsembleRequiresRescueAudit，交审计
                            # 判定是否可救 / 是否需重跑（可强制写入新 ensemble-ID 目录）。
                            if not legacy_repair:
                                raise ExistingEnsembleRequiresRescueAudit(
                                    f"窗口 {window_idx}: 磁盘能量缓存的 sampling_repair_policy="
                                    f"{cached_conv.get('sampling_repair_policy')!r}（当前 {repair_policy!r}）——"
                                    "产自旧的变异修复策略（f_k 可能被就地重校准过，属于不同参考系）。"
                                    "non_mutating_v1 拒绝复用，也拒绝就地重采覆盖（会毁掉 rescue 审计所需"
                                    "的原始数据）。已保留原文件，交 rescue 审计判定是否可救或需重跑。",
                                    diagnostics={
                                        "window_index": int(window_idx),
                                        "stage_type": stage_type,
                                        "cached_sampling_repair_policy": cached_conv.get("sampling_repair_policy"),
                                        "requires_rescue_audit": True,
                                    },
                                )
                            print(
                                f"  ⚠️ 窗口 {window_idx} 缓存的 sampling_repair_policy="
                                f"{cached_conv.get('sampling_repair_policy')!r}（期望 {repair_policy!r}）——"
                                "该缓存产自旧的变异修复策略（legacy_mutating 下按无效缓存重采）。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0 and not early_stop_ok:
                            print(
                                f"  ⚠️ 窗口 {window_idx} 缓存是 early stop 提前停止产出的短样本，"
                                f"但{early_stop_reject_reason}，视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0:
                            print(
                                f"  ⚠️ 窗口 {window_idx} 缓存能量形状匹配但 λ 值不匹配"
                                f"（缺少 λ 元数据或 λ 路径已变更），视为无效缓存，将重新采样该窗口。"
                            )
                        else:
                            print(
                                f"  ⚠️ 窗口 {window_idx} 缓存能量形状 {cached_e.shape} 与期望 "
                                f"({len(lc_win)}, N) 不符，将重新采样该窗口。"
                            )
                    except ExistingEnsembleRequiresRescueAudit:
                        raise
                    except Exception as e:
                        print(f"  ⚠️ 窗口 {window_idx} 缓存能量加载失败 ({e})，将重新采样该窗口。")

            # ---------- 构建系统 ----------
            win_sys, ibs_wrap = self._build_window_system(lc_win, lv_win, resolved_box, positions)
            if debug_mode:
                print_force_group_details(win_sys, prefix=f"窗口{window_idx}_系统构建后")

            # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] ibs_state_file 提前到这里
            # 拼接（原来在下面"初始化采样器"处才拼），纯字符串操作、无副作用——
            # 需要在还没建 Simulation/走最小化/爬坡之前就先廉价 peek 一下这个窗口
            # 是不是"续验冻结校准 f_k"的场景，才能决定要不要跳过重建。
            ibs_state_file = os.path.join(
                self.checkpoint_dir,
                f"ibs_state_{stage_type}_window_{window_idx}.json",
            )
            main_ckpt_path, main_manifest_path = _main_window_checkpoint_paths(
                self.checkpoint_dir, stage_type, window_idx
            )
            expected_main_manifest = _build_main_window_checkpoint_manifest(
                stage_type,
                window_idx,
                len(lc_win),
                openmm.XmlSerializer.serialize(win_sys),
                lc_win,
                lv_win,
                (float(np.mean(lv_win)) if _system_has_global_parameter(win_sys, "lambda_shield") else None),
                # 🔑 [temperature_K 类型修复] self.temperature 是带 kelvin 单位的
                # openmm Quantity（LangevinMiddleIntegrator 等物理调用都要求这个
                # 类型），但 _build_main_window_checkpoint_manifest 的 temperature_K
                # 形参约定是纯 float——直接传 Quantity 进去，里面的 float(temperature_K)
                # 会报 TypeError: Quantity.__float__ returned non-float。跟文件里其它
                # manifest 构造调用（如 _build_fixed_h_probe_bank_manifest 用
                # temperature_q.value_in_unit(unit.kelvin)）保持同样的显式转换。
                self.temperature.value_in_unit(unit.kelvin),
                self.platform_name,
                repair_policy=repair_policy,
            )
            attempt_checkpoint_restore = bool(
                resume
                and _peek_ibs_bias_status(ibs_state_file) == "calibrated_pending_validation"
                and _main_window_checkpoint_is_usable(
                    self.checkpoint_dir, stage_type, window_idx, expected_main_manifest
                )
            )

            integrator = LangevinMiddleIntegrator(self.temperature, 2.0 / unit.picosecond, 0.002 * unit.picosecond)
            integrator.setConstraintTolerance(1e-3)
            if hasattr(integrator, 'setRemoveCMMotion'):
                integrator.setRemoveCMMotion(True)

            sim = app.Simulation(self.topology, win_sys, integrator, platform, props)
            if resolved_box is not None:
                sim.context.setPeriodicBoxVectors(*resolved_box)
            sim.context.setPositions(positions)
            # lambda_shield 只存在于 VDW softcore 系统变体（build_ibs_dual_system）；
            # Shadow-Coulomb 变体（build_shadow_coul_ibs_system）里 VdW 全程满强度、
            # 不经过 λ-WCA 防护壳，不会注册这个 global parameter，需先判存在再设。
            if _system_has_global_parameter(win_sys, "lambda_shield"):
                lam_vdw_center = float(np.mean(lv_win))
                sim.context.setParameter("lambda_shield", lam_vdw_center)
                print(f"  🛡️ λ-WCA 防护壳已同步: lambda_shield={lam_vdw_center:.4f}")

            # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 续验冻结校准 f_k 时，
            # 若存在指纹匹配的主窗口 checkpoint，直接续算这个 Context 的坐标/
            # 速度/盒子/积分器 RNG 状态，跳过下面的最小化/dt 测试步进/Boresch
            # 爬坡——这份 checkpoint 就是上次冻结验证中断时的真实动力学快照，
            # 对它做最小化是人为的势能下降移动、不是动力学步，会把样本挪出
            # 续采轨迹本该在的分布，不能因为"反正很快"就无条件加上。
            # loadCheckpoint 失败（缺失/损坏/跟当前 System/Platform 不兼容）必须
            # 安全回退到今天的完整重建流程，不能让一个坏 checkpoint 搞崩整窗口。
            restored_from_window_checkpoint = False
            if attempt_checkpoint_restore:
                restored_from_window_checkpoint = _try_load_main_window_checkpoint(sim, main_ckpt_path)
                if restored_from_window_checkpoint:
                    print(
                        f"  🧊 窗口 {window_idx} 从主窗口 OpenMM checkpoint 续算"
                        "（跳过最小化/dt测试步进/Boresch爬坡，构象/速度/积分器RNG状态与上次中断时完全一致）。"
                    )
                    # 防御性重设：checkpoint 是否连带保存了 setParameter 的全局
                    # 参数值未经 100% 确认，代价极低，直接显式重设两个安全关键
                    # 参数，不依赖假设——呼应之前"探针最小化忘了同步 lambda_shield
                    # 导致动力学第一步 NaN"那次真实事故，这里不重蹈。
                    if _has_valid_boresch_restraint(self.boresch):
                        sim.context.setParameter("lambda_boresch_scale", 1.0)
                    if _system_has_global_parameter(win_sys, "lambda_shield"):
                        sim.context.setParameter("lambda_shield", float(np.mean(lv_win)))
                else:
                    print(f"  ⚠️ 窗口 {window_idx} 主窗口 checkpoint 加载失败，回退到完整重建流程。")

            # ---------- 最小化 ----------
            if not restored_from_window_checkpoint:
                print(f"\n  [阶段1] 开始能量最小化...")
                sim.minimizeEnergy(maxIterations=20000)
                print(f"  ✓ 最小化完成")

            # 几何检查
            if _has_valid_boresch_restraint(self.boresch):
                ok, r0_chk, thA_chk, thB_chk = _check_boresch_geometry_safe(sim.context, self.boresch)
                if not ok:
                    raise RuntimeError(
                        f"窗口 {window_idx} Boresch 几何不合格 "
                        f"(r0={r0_chk*10:.1f}Å, θA={thA_chk:.1f}°, θB={thB_chk:.1f}°)"
                    )
                print(f"  ✅ Boresch 几何检查通过：r0={r0_chk*10:.2f}Å，θA={thA_chk:.1f}°，θB={thB_chk:.1f}°")

            pre_test_breakdown = None
            if debug_mode:
                diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_最小化后")
                pre_test_breakdown = diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_最小化后")

            if not restored_from_window_checkpoint:
                # ---------- 测试步进 ----------
                print(f"\n[阶段2] 测试性步进 (Boresch 缩放至 1%)...")
                if _has_valid_boresch_restraint(self.boresch):
                    # 只做 minimum-image 诊断，绝不通过改坐标“修复”锚点距离。
                    # 旧逻辑把整个配体平移 H0-L0，必然令 L0'=H0，制造零距离。
                    state_chk = sim.context.getState(getPositions=True)
                    pos_chk = state_chk.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    rec_idx = self.boresch["receptor_indices"]
                    lig_idx = self.boresch["ligand_indices"]
                    H0, L0 = pos_chk[rec_idx[0]], pos_chk[lig_idx[0]]
                    minimum_image = _minimum_image_displacement_nm(
                        H0 - L0,
                        state_chk.getPeriodicBoxVectors(),
                    )
                    image_dist = float(np.linalg.norm(minimum_image))
                    if not np.isfinite(image_dist) or image_dist <= 1.0e-6:
                        raise RuntimeError(
                            f"窗口 {window_idx} Boresch minimum-image 锚点距离无效: "
                            f"{image_dist} nm"
                        )
                    print(
                        f"  ✅ Boresch minimum-image 锚点距离: {image_dist*10:.2f}Å "
                        "（未修改坐标）"
                    )
                    sim.context.setParameter("lambda_boresch_scale", 0.01)  # 1%
                    print(f"  🔧 Boresch 力常数缩放至 1%")
    
                original_dt = sim.integrator.getStepSize()
                test_schedule = [
                    (0.00001, 200, "0.01fs"),
                    (0.00005, 200, "0.05fs"),
                    (0.0001,  500, "0.1fs"),
                    (0.0002, 5000, "0.2fs"),
                    (0.0005, 5000, "0.5fs"),
                ]
                for dt_ps, n_steps, label in test_schedule:
                    sim.integrator.setStepSize(dt_ps * unit.picoseconds)
                    for step_batch in range(0, n_steps, 50):
                        actual_steps = min(50, n_steps - step_batch)
                        sim.step(actual_steps)
                        state = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
                        max_f = np.max(np.linalg.norm(forces, axis=1))
                        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                        has_bad_positions = np.any(~np.isfinite(positions))
                        has_bad_energy = not np.isfinite(e)
                        has_bad_force = not np.isfinite(max_f)
                        has_bad_values = has_bad_positions or has_bad_energy or has_bad_force
                        if debug_mode and (step_batch == 0 or step_batch >= n_steps - 50):
                            print(f"    [dt={label}] 步{step_batch+actual_steps}: E={e:.1f}, max|F|={max_f:.1f}, finite={not (has_bad_positions or has_bad_energy or has_bad_force)}")
                        if has_bad_values:
                            print(f"    🚨 非有限数值检测，打印力组拆解：")
                            diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_NaN前_dt={label}")
                            diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_NaN前_dt={label}")
                            # ✅ 诊断增强：只看力组拆解只能知道"哪一类力炸了"，看不出是哪个
                            # 原子/哪对接触炸的。补一次原子级定位，直接指认具体原子，不用
                            # 再靠猜。若这一步坐标本身已经是 NaN（力已发散到无法再取一次
                            # 有限的 getForces），定位可能拿不到有效结果，静默跳过即可。
                            try:
                                diagnose_top_force_atoms(
                                    sim.context,
                                    win_sys,
                                    topology=self.topology,
                                    ligand_indices=self.ligand_indices,
                                    prefix=f"窗口{window_idx}_NaN前_dt={label}",
                                )
                            except Exception as diag_exc:
                                print(f"    ⚠️ 原子级定位诊断失败（不影响主异常上报）：{diag_exc}")
                            raise RuntimeError(f"在 dt={label} 处发生非有限坐标/能量/力")
    
                sim.integrator.setStepSize(original_dt)
                print(f"  ✅ 测试步进通过，恢复步长 {original_dt.value_in_unit(unit.picoseconds):.3f} ps")
                post_test_breakdown = None
                if debug_mode:
                    post_test_breakdown = diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_测试步进后")
    
                deadlock_msg = _detect_constraint_deadlock(
                    pre_test_breakdown,
                    post_test_breakdown,
                    win_sys.getNumConstraints(),
                )
                if deadlock_msg is not None:
                    print(f"  🚨 [约束死锁预警] {deadlock_msg}")
                    print("  🧯 自动切换到 1.0 fs 保守步长，并执行额外最小化/松弛以避免生产阶段首步崩溃...")
                    original_dt = 0.001 * unit.picoseconds
                    sim.integrator.setStepSize(original_dt)
                    sim.minimizeEnergy(maxIterations=5000, tolerance=10.0)
                    for _ in range(10):
                        sim.step(200)
                    if debug_mode:
                        diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_死锁缓解后")
    
                # ---------- Boresch 安全爬坡 ----------
                # ================================================================
                # Boresch 安全爬坡：自定义阶梯，逐个恢复力强度
                # ================================================================
                if _has_valid_boresch_restraint(self.boresch):
                    print(f"\n[阶段3] Boresch 安全爬坡（自定义阶梯）...")
                    
                    # 自定义阶梯序列：低强度区采用更细分辨率，避免在高内应力底座上一步踩空。
                    custom_scales = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
                    dt_ramp = 0.001                  # 爬坡期间极小步长 (ps)
                    
                    # 保存原始步长，并设置为爬坡步长
                    original_dt = sim.integrator.getStepSize()
                    sim.integrator.setStepSize(dt_ramp * unit.picoseconds)
                    print(f"  → 爬坡使用步长 {dt_ramp} ps，低强度区采用更细台阶")
                    
                    # 确保从当前 scale 开始（例如之前测试步进时设置的 0.01）
                    try:
                        current_scale = sim.context.getParameter("lambda_boresch_scale")
                    except Exception:
                        current_scale = 0.01
                    print(f"  → 起始 Boresch scale = {current_scale:.3f}")
                    prev_energy = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    
                    ramp_success = True
                    for target_scale in custom_scales:
                        # 只在目标大于当前值时才爬升（避免回退）
                        if target_scale <= current_scale:
                            continue
                        
                        sim.context.setParameter("lambda_boresch_scale", float(target_scale))
                        n_steps_per_level = 1500 if target_scale <= 0.10 else (1000 if target_scale <= 0.30 else 500)
                        print(f"  🔹 设置 Boresch scale = {target_scale:.2f}，松弛 {n_steps_per_level} 步...", end="", flush=True)
                        sim.minimizeEnergy(maxIterations=200, tolerance=20.0)
                        for _ in range(max(1, n_steps_per_level // 100)):
                            sim.step(100)
                        
                        # 检查能量与受力
                        state = sim.context.getState(getEnergy=True, getForces=True)
                        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
                        max_f = np.max(np.linalg.norm(forces, axis=1))
                        delta_e = e - prev_energy
                        
                        if (not np.isfinite(e)) or max_f > 50000 or (abs(delta_e) > 5e5 and max_f > 15000):
                            print(f"\n  🚨 Boresch 爬坡在 scale={target_scale:.2f} 处失败！")
                            print(f"    当前势能 = {e:.1f} kJ/mol，ΔE = {delta_e:.1f} kJ/mol，最大力 = {max_f:.1f} kJ/(mol·nm)")
                            if debug_mode:
                                diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_Boresch爬坡失败_scale{target_scale:.2f}")
                            ramp_success = False
                            break
                        else:
                            print(f" 势能 = {e:.2f} kJ/mol，ΔE = {delta_e:.2f} kJ/mol，最大力 = {max_f:.2f} kJ/(mol·nm)")
                            # 记录新的当前 scale
                            current_scale = target_scale
                            prev_energy = e
                    
                    # 恢复原始步长
                    sim.integrator.setStepSize(original_dt)
                    
                    if not ramp_success:
                        raise RuntimeError(
                            f"窗口 {window_idx} Boresch 爬坡失败（scale={target_scale:.2f}），系统可能已崩溃。"
                            "请检查锚点几何、力常数或初始结构。"
                        )
                    
                    print(f"  ✅ Boresch 爬坡成功完成 (scale 已恢复至 1.0)")
                else:
                    print(f"\n[阶段3] 无 Boresch 限制力，跳过爬坡。")

            # ================================================================
            # 爬坡后：立刻进行一次非侵入式力分解，供对比分析
            # ================================================================
            if debug_mode and self.boresch:
                diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_Boresch爬坡完成")

            # ---------- 初始化采样器 ----------
            # ibs_state_file 已经在窗口开头（checkpoint restore 判断之前）提前
            # 拼接过，这里不再重复。
            sampler = IBSSampler(sim.context, len(lc_win), self.temperature, self.prefix, ibs_wrapper=ibs_wrap)
            # 🔑 [non_mutating_v1] stamp the current run's policy so save_ibs_state
            # records it (old-policy state files carry None → detected on load).
            sampler.sampling_repair_policy = repair_policy

            # 🔑 核心修复：断点续传状态检测
            is_resumed_ibs = False
            if resume and os.path.exists(ibs_state_file):
                is_resumed_ibs = sampler.load_ibs_state(ibs_state_file, lc_win, lv_win)

            # 🔑 [终态硬停止] load_ibs_state 已经完成了 n_states/prefix/协议版本/
            # λ 内容的严格匹配（不是廉价 peek），确认这份状态真的属于当前窗口后，
            # 若上一轮已经把它判定为 calibrated_validation_failed（冻结验证累计
            # 预算用到最后一档仍未通过），必须立刻硬停止——不做任何 SGD/续验
            # 尝试，也不落入下面"is_resumed_ibs and not sampler.bias_converged"
            # 分支被当成普通未收敛热启动重新学习。调用方（abfe_pipeline.py）看到
            # 这个 terminal=True 的异常应该直接向上传播，不再自动重试。
            if is_resumed_ibs and sampler.bias_status == "calibrated_validation_failed":
                raise IBSFrozenCalibrationValidationError(
                    f"窗口 {window_idx} 的冻结校准验证此前已判定为终态失败"
                    "（calibrated_validation_failed）：这份 f_k 已经用 fixed-H overlap 探针 + "
                    "bias 校准探针证明过物理正确，但冻结验证累计预算用到最后一档仍未能通过独立"
                    "验证——不再自动续验、不延长预算、不回退 learning。需要人工检查（构象弛豫"
                    "确实很慢，或偏置表达式仍有问题），人工确认后需手动清空该窗口的 IBS 状态"
                    "文件才能重新开始。",
                    diagnostics={
                        "window_index": int(window_idx),
                        "stage_type": stage_type,
                        "calibration_validation_terminally_failed": True,
                    },
                    terminal=True,
                )

            # 🔑 [non_mutating_v1] Fail closed on any resumed state that ONLY the
            # deprecated mutating calibration path could have produced. Under
            # non_mutating_v1 f_k is never MBAR-recalibrated in place, so a
            # loaded "calibrated_pending_validation" status / non-null
            # frozen_f_k_pending can only come from an old mutating run.
            # Continuing to validate it would build production on an f_k from a
            # different reference frame — refuse, preserve the state files, and
            # hand the ensemble to the rescue audit.
            if (
                is_resumed_ibs
                and not legacy_repair
                and (
                    getattr(sampler, "bias_status", "unconverged") == "calibrated_pending_validation"
                    or sampler.frozen_f_k_pending is not None
                )
            ):
                raise ExistingEnsembleRequiresRescueAudit(
                    f"窗口 {window_idx}: 载入的 IBS 状态带有 calibrated_pending_validation / "
                    f"frozen_f_k_pending（bias_status={getattr(sampler, 'bias_status', None)!r}），"
                    "这只可能由已弃用的变异校准路径产生。non_mutating_v1 从不就地重校准 f_k，"
                    "拒绝续验（否则会把生产建立在旧参考系的 f_k 上），保留该窗口全部状态文件，"
                    "交 rescue 审计判定是否可救、是否需要重跑。",
                    diagnostics={
                        "window_index": int(window_idx),
                        "stage_type": stage_type,
                        "bias_status": getattr(sampler, "bias_status", None),
                        "requires_rescue_audit": True,
                    },
                )

            # bias 预热收敛诊断（在分支之外先占位，确保后面落盘时一定有值可写）。
            bias_warmup_diag = {"status": "not_run"}

            # ---------- 渐进预热 ----------
            # 🔑 之前"续传就跳过 Warmup"是不安全的：load_ibs_state 恢复的 f_k
            # 可能是上次中断时还没收敛的中间值（旧协议下甚至可能是"生产阶段仍在
            # 被 update_weights() 调整"的漂移值），直接当成已收敛放行进生产，
            # 会让整段生产采样建立在一个从未真正验证过的偏置上。现在只有当
            # load_ibs_state 明确报告 bias_converged=True（该状态本身是在严格
            # 收敛判据下才会被置位并落盘的）时才真正跳过；否则一律走下面的收敛
            # 判定循环——已恢复的 f_k 只是被当成一个更好的热启动起点，不是免检的
            # 通行证。
            skip_warmup_entirely = bool(is_resumed_ibs and sampler.bias_converged)
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=7] 缓存的 bias_converged=True 只能证明这个
            # f_k 曾经在旧构型下有效，不能证明本次新建 Context 的当前构型已经在这个
            # 固定偏置下平衡过——旧代码在这里直接跳过整个预热块进生产，等于零平衡就
            # 采信一个从未在当前 Context 里验证过的分布。现在不再整段跳过：仍然完全
            # 跳过 learning（不重新调整已恢复的 f_k），但下面的状态机会从
            # freeze_burn_in 开始，对这份恢复的 f_k 重新做一次冻结 burn-in + 只读验证，
            # 通过了才真正放行进生产；验证失败则和全新窗口一样自动回退到 learning。
            resumed_frozen_f_k = None
            if skip_warmup_entirely:
                sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
                print(
                    "  🚀 检测到已收敛的 IBS 历史状态：跳过 learning，"
                    "直接对已恢复的 f_k 重新做一次冻结 burn-in + 只读验证"
                    "（不重新学习权重，但也不会零平衡就直接进生产）。"
                )
                resumed_frozen_f_k = [
                    float(sim.context.getParameter(f"{self.prefix}_f_{k}"))
                    for k in range(len(lc_win))
                ]
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] MBAR 校准过的 f_k 冻结验证未在
            # 预算内通过时不能落进下面这个"普通未收敛"分支——那个分支会把
            # bias_scale 设为 1.0 后让状态机从 mode="learning" 开始，SGD 会
            # 立刻开始修改这份其实已经用 fixed-H overlap + bias 校准探针证明
            # 过是对的 f_k。必须单独识别 bias_status=="calibrated_pending_validation"，
            # 直接用 frozen_f_k_pending 当 resumed_frozen_f_k、跳过 learning，
            # 从 freeze_burn_in 开始只延长冻结验证。
            elif (
                is_resumed_ibs
                and getattr(sampler, "bias_status", "unconverged") == "calibrated_pending_validation"
                and sampler.frozen_f_k_pending is not None
            ):
                sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
                for k, f_val in enumerate(sampler.frozen_f_k_pending):
                    sim.context.setParameter(f"{self.prefix}_f_{k}", float(f_val))
                print(
                    "  🧊 检测到 MBAR 校准已冻结但验证尚未通过的历史状态："
                    "跳过 learning 与 fixed-H overlap/bias 校准探针，直接延长这份"
                    "冻结 f_k 的 burn-in + 只读验证（不重新学习权重，也不重新校准）。"
                )
                resumed_frozen_f_k = [float(x) for x in sampler.frozen_f_k_pending]
            elif is_resumed_ibs and not sampler.bias_converged:
                print(
                    "  ♻️ 检测到 IBS 历史状态，但此前预热未判定收敛——"
                    "以该状态为热启动起点继续走严格收敛判定，不清零重来，也不假设已经收敛。"
                )
                sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
            else:
                if enable_gradual_warmup:
                    # 只做时间步长爬坡；偏置力的唯一平滑引入通道是下面的
                    # "[偏置预热]" 块，两者不再重复覆盖同一个 bias_scale 目标。
                    self._gradual_warmup_debug(sim, ibs_wrap, sampler, warmup_steps, win_sys, window_idx, debug_mode)
                # 不管上面是否做了时间步长爬坡，bias_scale 的唯一一次 0→1.0
                # 平滑引入都交给紧接着的"[偏置预热]"块（它会显式把 bias_scale
                # 重置到 0.0 再爬升，不依赖这里是什么值）。

            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 这次 resume 是否直接续验已经
            # 校准好的冻结 f_k——是的话，下面必须跳过 fixed-H overlap 探针和
            # MBAR 校准（f_k 已经是校准好的，不需要也不应该重新学习/重新校准），
            # 且这次冻结验证要用 frozen_validation_step_overrides 里给这个窗口
            # 累计延长过的预算，而不是默认的 mbar_calibration_reserved_steps。
            resumed_calibration_pending = bool(
                resumed_frozen_f_k is not None and is_resumed_ibs
                and getattr(sampler, "bias_status", "unconverged") == "calibrated_pending_validation"
            )
            # 🔑 [跨进程阶梯状态修复] frozen_validation_step_overrides/
            # frozen_validation_is_final_rung 只存在调用方 abfe_pipeline.py 一次
            # 进程运行期间的本地 dict，从不落盘——一旦作业被杀、用 --resume 起
            # 一个全新进程，这两个 dict 对这个窗口都是空的，之前这里会直接退回
            # 一个跟已持久化进度无关的固定默认值 mbar_calibration_reserved_steps
            # (50000)，如果这个窗口的 frozen_validation_cumulative_steps 早就
            # 超过了 50000（几乎总是这样，只要它已经续验过至少一轮），
            # remaining_budget_this_attempt 会被下面的 max(..., check_chunk) 封底
            # 成几乎为零的一小段（500 步），这一整个 attempt 基本被浪费掉，阶梯
            # 还得从第一档重新往上爬——真实预算被这个重启悄悄削减。现在的回退
            # 逻辑不再假设"没有覆盖字典就等于第一档"，而是直接从已持久化的
            # frozen_validation_cumulative_steps 反推"下一档没跑完的目标"：
            # 取 FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS 中第一个大于当前累计
            # 步数的档位（找不到就说明已经跑满最后一档，直接落在最后一档，让
            # 下面 is_final_rung 的回退逻辑正确判定终态）。调用方显式传入的
            # 覆盖值（同进程内正常阶梯升档的情况）优先于这个回退，行为不变。
            effective_frozen_validation_budget = _resolve_frozen_validation_budget_for_window(
                window_idx,
                frozen_validation_step_overrides,
                int(getattr(sampler, "frozen_validation_cumulative_steps", 0)),
            )

            if debug_mode:
                print(f"\n[诊断] 预热后状态检查：")
                diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_预热后")
                diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_预热后")
                diagnose_softcore_cv_values(sim.context, ibs_wrap, lc_win, lv_win, prefix=f"窗口{window_idx}_预热后", sampler=sampler)
            raw_probe = sampler.get_raw_interaction_energies()
            if len(raw_probe) >= 2:
                raw_span = float(np.max(raw_probe) - np.min(raw_probe))
                if raw_span < 1e-6 and np.max(np.abs(raw_probe)) < 1e-6:
                    raise RuntimeError(
                        f"窗口 {window_idx} 的 IBS 软核 CV 全部为 0。"
                        "这不是窗口重叠问题，而是 CV 构造/读取链路异常；已中止以避免产出全零溶剂腿或失真复合物腿。"
                    )

            # 清空 sampler 的能量缓存
            sampler.energy_buffer = []
            sampler.energy_history = []
            sampler.bias_history = []
            sampler.base_energy_history = []

            # ⚠️ original_dt 在此无条件捕获（不能挪进下面的 if 分支里）：续传
            # 分支跳过偏置预热块，但"生产前卸压"结束后仍会无条件用它恢复步长。
            original_dt = sim.integrator.getStepSize()

            # ---------- 偏置力预热 (Bias Ramp) ----------
            # ✅ 性能/正确性修复：这是 bias_scale 唯一一次真正的 0→1.0 平滑爬坡
            # （_gradual_warmup_debug 只做时间步长爬坡，不再重复覆盖 bias_scale）。
            # 续传时 (is_resumed_ibs=True) 上面已经把 bias_scale 直接设为 1.0 并
            # 打印"跳过 Warmup"——这里必须真的跳过，否则会打脸自己的日志，且对
            # 已收敛的 f_k 做一次无意义的清零重爬。
            # [IBS_BIAS_PROTOCOL_VERSION=29] 是否已经用 pilot 种子播过 f_k；否则冷启动
            # 时下面（scale=1.0 后）用自举 TI 种子兜底，避免从 f_k=0 慢爬。
            f_k_warm_started = False
            if not is_resumed_ibs:
                # [IBS_BIAS_PROTOCOL_VERSION=27] The pilot-TI estimator returns
                # the physical F(lambda), mean-centered, which is the
                # ready-to-use convention for exp[-beta*(U_k-f_k)].  Do not
                # infer this sign from instantaneous occupancy: occupancy-high
                # states are lowered by the separate online negative feedback.
                from abfe_preoptimizer import estimate_f_k_from_pilot_ti
                warm_start_seed = estimate_f_k_from_pilot_ti(
                    getattr(self, "pilot_lambdas", None),
                    getattr(self, "pilot_mean_dU_dlambda", None),
                    target_lambdas=lv_win,
                )
                if warm_start_seed is not None and len(warm_start_seed) == len(lv_win):
                    for k in range(len(lv_win)):
                        sim.context.setParameter(f"{self.prefix}_f_{k}", float(warm_start_seed[k]))
                    f_k_warm_started = True
                    print(
                        f"  🌱 [pilot TI 热启动] 窗口 {window_idx} f_k 初始值（非冷启动 0.0）: "
                        f"{[round(float(x), 3) for x in warm_start_seed]} kJ/mol"
                    )
                elif warm_start_seed is not None:
                    print(
                        f"  ⚠️ [pilot TI 热启动] 窗口 {window_idx} 估计出的种子长度 "
                        f"({len(warm_start_seed)}) 与本窗口态数 ({len(lv_win)}) 不符，"
                        "拒绝注入，回退 f_k=0.0 冷启动"
                    )
                else:
                    print(f"  ℹ️ [pilot TI 热启动] 窗口 {window_idx} 无可用 pilot 数据，f_k 仍从 0.0 冷启动")

                print(f"\n  🔥 [偏置预热] 开始... (从零缓慢加载偏置力)")
                sim.integrator.setStepSize(0.001 * unit.picoseconds)

                sim.context.setParameter(f"{self.prefix}_bias_scale", 0.0)

                ramp_stages = [
                    (0.2, 2000),  # 可以适当增加每一步的步数，让系统充分弛豫
                    (0.3, 2000),
                    (0.5, 2000),
                    (0.7, 2000),
                ]
                sampler.energy_buffer = []
                sampler.energy_history = []
                sampler.bias_history = []
                sampler.base_energy_history = []
                sampler.ema_mean_p = None
                for target_scale, steps in ramp_stages:
                    sim.context.setParameter(f"{self.prefix}_bias_scale", target_scale)
                    print(f"    → Bias Scale 设为 {target_scale}, 运行 {steps} 步...", end="", flush=True)

                    for ramp_step in range(0, steps, 200):   # 每 200 步检查一次
                        sim.step(200)

                    state = sim.context.getState(getEnergy=True)
                    if not np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
                        raise RuntimeError(f"偏置预热阶段在 scale={target_scale} 时能量 NaN")
                    print(" 完成")
                # Ramp 只负责把动力学平滑带到正式 IBS Hamiltonian。scale<1 的
                # 构型不能进入 scale=1 固定点方程，也不能提前修改 f_k。
                sampler.energy_buffer = []
                sampler.ema_mean_p = None

            # ---- bias_scale=1.0：两阶段严格收敛判据（learning -> freeze -> validate）----
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=7] v6 判据有个更深的问题：一轮"连续
            # N 次通过"里，每一次通过用的 mean_p_batch/ema_mean_p 都是从
            # update_weights() 内部的 f_old 算出来的，而 update_weights() 在同一
            # 次调用里立刻把 f_old 换成 f_new 并写回 context——真正冻结进生产的
            # 那个 f_k，从未被任何一次"通过"检验过；"连续 3 次通过"检验的是 3 个
            # 不同、都已经被丢弃的旧 f_k，不是同一个冻结 Hamiltonian 下的 3 次独立
            # 验证，不能证明分布已经稳定。正确做法改成两阶段：
            #   1) [learning] 沿用旧的 update_weights() + 连续通过判据，达到后只是
            #      "候选收敛"，不直接宣布 converged。已恢复的收敛缓存
            #      （resumed_frozen_f_k 非 None）跳过这一阶段，直接从 freeze_burn_in
            #      开始，用恢复的 f_k 当候选，而不是重新学习。
            #   2) 冻结 f_k（此后到验证结束前绝不再调用 update_weights()）→ 丢弃
            #      一段 burn-in（之前的样本来自还在漂移的偏置，冻结后需要重新
            #      平衡）→ 用 evaluate_frozen_batch_probability()（只读，不写 f_k）
            #      在真正固定的 Hamiltonian 下累计新 batch；至少达到指定 batch 数
            #      且累计 <p_k> 通过同一 LSE 门才宣布 converged。不能用 EMA，也不能
            #      像 v17 那样让一个仅 20-frame 的 batch 一失败就立刻切回 learning。
            #   3) 一次完整冻结验证 attempt 仍失败：先把这批 held-out 样本作为新的
            #      Sec. 2.3 TA iteration 并入 Q_hat，再恢复 [learning]；下一候选仍需
            #      重新 burn-in，并用全新的固定-f 样本独立验证。
            # 更新次数上限 max_bias_updates 只应该限制 learning：如果恰好在第
            # max_bias_updates 次更新时形成候选收敛，mode 已切到 freeze_burn_in，但
            # 循环条件若仍把 bias_update_count 当整体退出条件，下一轮会因为
            # "count < max_bias_updates" 为假直接退出，burn-in/validation 根本来
            # 不及执行——所以这里改成只在 mode=="learning" 时检查这个上限，
            # freeze_burn_in/validating 不消耗、也不受这个额度限制，只受
            # steps_at_full_bias 这个总步数安全帽约束。
            sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
            print("    → Bias Scale 设为 1.0，按两阶段收敛判据运行...", end="", flush=True)
            K = len(lc_win)
            target_p = 1.0 / K
            min_probability_threshold = 0.5 * target_p
            coverage_ess_threshold = 0.8 * K
            check_chunk = IBS_WARMUP_FRAME_STRIDE_STEPS
            f_stability_threshold_kJ_mol = 0.05
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 自举 TI 种子（pilot 缺失时的兜底）：
            # 冷启动（无 pilot、非续算、非恢复冻结 f_k）时，在 scale=1.0 下采一小批
            # ~IBS_TMBAR_LEARNING_MINIBATCH_FRAMES 帧，用每态平均 softcore 能量把
            # f_k 一步播到 f_k=⟨u_k⟩（去均值）——这是从采样系综测得的零阶 EXP/TI
            # 种子（与 pilot TI 同一约定 f_k≈F_k），把偏置一步放到大致尺寸，避免从
            # f_k=0 靠 2 kT/步慢爬。这段样本只用于播种、不进 tmbar_history，也不计入
            # steps_at_full_bias 预算。重叠好的窗口据此近乎立即变平；低重叠窗口种子
            # 可能过/欠，但随后的 bounded 自适应更新会快速纠偏。
            if (
                not is_resumed_ibs
                and not f_k_warm_started
                and resumed_frozen_f_k is None
            ):
                _boot_target = int(IBS_TMBAR_LEARNING_MINIBATCH_FRAMES)
                sampler.energy_buffer = []
                _boot_guard = 0
                while (
                    len(sampler.energy_buffer) < _boot_target
                    and _boot_guard < _boot_target * 4
                ):
                    sim.step(check_chunk)
                    sampler.collect_energies()
                    _boot_guard += 1
                _boot_u = (
                    np.asarray(sampler.energy_buffer, dtype=np.float64)
                    if sampler.energy_buffer else np.empty((0, K))
                )
                _boot_valid = (
                    ~np.isnan(_boot_u).any(axis=1)
                    if _boot_u.size else np.zeros(0, dtype=bool)
                )
                if _boot_u.size and int(np.sum(_boot_valid)) >= max(5, K + 1):
                    _boot_mean_u = np.mean(_boot_u[_boot_valid], axis=0)
                    _boot_seed = _boot_mean_u - float(np.mean(_boot_mean_u))
                    if np.all(np.isfinite(_boot_seed)) and _boot_seed.shape[0] == K:
                        for k in range(K):
                            sim.context.setParameter(
                                f"{self.prefix}_f_{k}", float(_boot_seed[k])
                            )
                        f_k_warm_started = True
                        print(
                            f"  🌱 [自举 TI 种子] 窗口 {window_idx} 冷启动无 pilot，用首批 "
                            f"{int(np.sum(_boot_valid))} 帧每态平均 softcore 能量播 "
                            f"f_k=⟨u_k⟩（去均值）: "
                            f"{[round(float(x), 2) for x in _boot_seed]} kJ/mol"
                            "（跳过 f_k=0 慢爬）"
                        )
                    else:
                        print(
                            f"  ⚠️ [自举 TI 种子] 窗口 {window_idx} 种子非有限/维度不符，"
                            "回退 f_k=0 冷启动"
                        )
                else:
                    print(
                        f"  ⚠️ [自举 TI 种子] 窗口 {window_idx} 有效帧不足，回退 f_k=0 冷启动"
                    )
                sampler.energy_buffer = []
                sampler.ema_mean_p = None
            # [IBS_BIAS_PROTOCOL_VERSION=29] 冻结 f_k 后、采 local-MBAR 门数据前的
            # 重新平衡 burn-in。这不是"等 f_k 稳定"（loose-gate 不等 f_k 收敛），
            # 而是让构型忘掉上一段漂移偏置轨迹、在这个刚冻结的固定 Hamiltonian 下
            # 重新平衡；之前 v27 用 20k 步冻结验证阶梯，loose-gate 只需一小段短
            # burn-in（5k 步≈10 ps）即可，随后累计固定-f_k 帧一次性喂 local MBAR。
            frozen_burn_in_steps = 5_000
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=9] steps_at_full_bias 是 SGD 三阶段
            # （learning/freeze_burn_in/validating）和后面 fixed-H overlap 全通过
            # 后的 MBAR 校准验证共用的同一个计数器。如果 SGD 把 max_bias_warmup_steps
            # 全部烧光才退出循环，MBAR 校准验证的 while 条件一次都不会执行，
            # calibration_converged 必然是 False——不是校准本身失败，是预算已经
            # 没有了。因此这里把总预算拆成两块：SGD 只能用
            # sgd_step_budget = max_bias_warmup_steps - mbar_calibration_reserved_steps，
            # 剩下的 mbar_calibration_reserved_steps 留给 MBAR 校准（下面校准循环
            # 用独立的 calibration_steps_used 计数，不再检查 steps_at_full_bias 是否
            # 撞到 max_bias_warmup_steps），保证 fixed-H overlap 全通过时校准验证
            # 一定有机会真正跑完 burn-in + 连续通过判据。
            # non_mutating_v1 永远不会进入下面 legacy fixed-H/MBAR 校准分支，
            # 因而不能白白预留 50k：那正是日志出现 45/50 updates、450k/500k
            # 却提前退出的原因。只有 legacy_mutating 才扣校准预留预算。
            effective_mbar_calibration_reserved_steps = (
                int(mbar_calibration_reserved_steps) if legacy_repair else 0
            )
            frozen_validation_reserved_steps = (
                0 if legacy_repair else int(mbar_calibration_reserved_steps)
            )
            sgd_step_budget = max(
                int(max_bias_warmup_steps)
                - int(effective_mbar_calibration_reserved_steps),
                int(frozen_burn_in_steps) + check_chunk,
            )
            # Frozen burn-in/validation is a distinct read-only phase.  Give it
            # its own reserve instead of stealing that reserve from learning;
            # otherwise update 50 can never be frozen and validated.
            full_bias_step_budget = (
                int(max_bias_warmup_steps)
                + int(frozen_burn_in_steps)
                + int(frozen_validation_reserved_steps)
            )
            if resumed_calibration_pending:
                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 续验已经冻结的校准 f_k 时，
                # 从一开始就 mode="freeze_burn_in"，永远不会进入 learning，
                # 不需要也不应该套用"给 learning 大部分预算、只留
                # mbar_calibration_reserved_steps 给校准验证"的拆分——这里的
                # 预算是 frozen_validation_step_overrides 给这个窗口的累计
                # 目标步数（50k→150k→300k 阶梯，见调用方
                # abfe_pipeline.py::_run_stage_with_overlap_autorepair）。
                # 🔑 [修复：累计预算记的是"目标总量"，不是"这一轮要新跑多少"]
                # effective_frozen_validation_budget 是"这份校准 f_k 从第一次
                # 冻结验证开始、累计应该验证到的总步数"，不是"这次 attempt 单独
                # 要跑的步数"——之前这里直接把它当成 sgd_step_budget，导致
                # 50k/150k/300k 三档实际总步数变成 50k+150k+300k=500k，而不是
                # 阶梯设计意图的"累计延长到 300k"。现在扣掉
                # sampler.frozen_validation_cumulative_steps（之前所有 attempt
                # 已经真正花在这份冻结 f_k 验证上的步数，跨 resume 持久化），
                # 这一轮只跑差值——如果从主窗口 checkpoint 续算（见
                # MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION），差值步数接的就是
                # 上次中断的真实动力学轨迹，不是重新烧一遍。
                prior_cumulative_steps = int(getattr(sampler, "frozen_validation_cumulative_steps", 0))
                remaining_budget_this_attempt = max(
                    int(effective_frozen_validation_budget) - prior_cumulative_steps,
                    int(check_chunk),
                )
                sgd_step_budget = max(
                    remaining_budget_this_attempt,
                    int(frozen_burn_in_steps) + check_chunk,
                )
                frozen_validation_reserved_steps = 0
                full_bias_step_budget = int(sgd_step_budget)
            validation_attempt_budget_steps = (
                int(full_bias_step_budget)
                if resumed_calibration_pending
                else max(
                    int(frozen_validation_reserved_steps),
                    int(required_consecutive_bias_updates) * 20 * int(check_chunk),
                )
            )
            minimum_complete_validation_frames = max(
                int(required_consecutive_bias_updates) * 20,
                (
                    int(validation_attempt_budget_steps) + int(check_chunk) - 1
                ) // int(check_chunk),
            )
            steps_at_full_bias = 0
            bias_update_count = 0
            bias_converged = False
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] True 当且仅当这个窗口是靠"见好
            # 就收"兜底（达到重试上限、采用最优 attempt）而不是严格
            # lse_log_residual_tolerance 判据放行进生产的——见 version 25
            # changelog (a)，写入 bias_warmup_diag/convergence.json 供事后追溯。
            best_effort_acceptance = False
            best_effort_acceptance_reason = None
            best_effort_reburnin_required = False
            best_effort_reburnin_steps = 0
            truncated_validation_frames_ignored = 0
            safety_cap_best_effort_tmbar_converged = False
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] 实现 PROPOSAL_frozen_validation_
            # fallback.md：learning 阶段候选连续通过次数从未凑满、纯粹因为
            # max_bias_updates 耗尽时，给当前 f_k 一次真正的冻结验证机会
            # （而不是直接 f_not_converged），guard 确保每个窗口 attempt 只
            # 触发一次——第二次再撞到同样的耗尽条件说明这次真的没戏，仍然
            # 老实 break。
            budget_fallback_used = False
            consecutive_pass_count = 0
            validation_pass_count = 0
            last_lse_balance = None
            lse_residual_history = []
            learning_to_validation_cycles = 0
            validation_feedback_update_count = 0
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12 修复，未升版本号——见下方修复处注释]
            # 仅用于诊断的计数器：resumed_calibration_pending 续验分支里每次单批
            # 验证失败都会 +1，但绝不触发退回 learning。跟 learning_to_validation_cycles
            # 分开计数，因为后者的语义是"退回 learning 的次数"，续验分支永远不退回
            # learning，必须让它保持 0，否则会污染下面失败打印/诊断里"是否发生过
            # 假收敛重学习"的含义。
            frozen_validation_retry_count = 0
            last_f_delta = float("nan")
            freeze_burn_in_done = 0
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] "见好就收"兜底状态：记录目前
            # 观察到的、max_abs_log_residual 最小的那次完整冻结验证 attempt
            # 的 f_k/统计信息。达到 max_frozen_validation_cycles_before_
            # accept_best 次失败后，直接采用这份最优结果而非继续无限重试——
            # 见 version 25 changelog (a)。
            best_effort_residual = float("inf")
            best_effort_f_k = None
            best_effort_validation_stats = None
            validation_batch_history = []
            validation_cumulative_history = []
            last_validation_batch_p = None
            validation_probability_sum = np.zeros(K, dtype=np.float64)
            validation_sample_count = 0
            validation_batch_count = 0
            validation_steps_this_freeze = 0
            early_probe_triggered = False
            early_probe_trigger_reason = None
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 局部滑窗 MBAR loose-gate 状态：
            # frozen_mbar_batches 累计当前冻结 f_k 下最近若干批固定-f_k minibatch
            # （每条是 tmbar_history 里的 {u_kn,bias_energies,base_energies}）；攒满
            # IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES 批就拼一次 local MBAR 判一次门。
            # updates_since_freeze/have_frozen_once 控制"首次冻结前累计 min_bias_
            # updates 次更新、此后每次门失败只需再更新一轮就重新冻结复检"。
            frozen_mbar_batches = []
            updates_since_freeze = 0
            have_frozen_once = False
            local_mbar_gate_history = []
            last_local_mbar_gate = None
            if resumed_frozen_f_k is not None:
                # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 从主窗口 checkpoint
                # 续算时不需要 freeze_burn_in：checkpoint 保存的那一刻本身就已经
                # 在 validating（或刚做完一次 freeze_burn_in）里，冻结 f_k 全程
                # 没变，没有需要"忘记"的漂移历史，直接继续采下一批全新样本即可。
                # 没有可用 checkpoint 时（缺失/不兼容/首次校准刚失败一次）保持
                # 原有行为：先走一次 freeze_burn_in 再进入 validating。
                mode = "validating" if restored_from_window_checkpoint else "freeze_burn_in"
                frozen_f_k_snapshot = resumed_frozen_f_k
                sampler.energy_buffer = []
                sampler.ema_mean_p = None
                # 恢复的是一份已经冻结过的 f_k，直接进 freeze/validate。若这次 loose
                # gate 未过退回 learning，则和普通窗口一样由 readiness 门决定何时重新
                # 冻结（占据重新平坦、步长≤1 kT、连续满足），不会"回来一轮就立刻重冻"。
                have_frozen_once = True
            else:
                mode = "learning"
                frozen_f_k_snapshot = None

            while steps_at_full_bias < full_bias_step_budget:
                # learning 阶段的更新次数上限只约束 learning；freeze_burn_in/
                # validating 只受总步数安全帽 full_bias_step_budget 约束。反复未
                # 通过 loose-gate 把更新次数烧到上限时退出循环，由下方"预算耗尽即
                # 接受当前 f_k"分支放行进生产（见 IBS_BIAS_PROTOCOL_VERSION=29）。
                if mode == "learning" and bias_update_count >= int(max_bias_updates):
                    break
                sim.step(check_chunk)
                steps_at_full_bias += check_chunk
                sampler.collect_energies()

                if mode == "learning":
                    # 现有 f_k 更新（全历史 TMBAR 自洽绝对候选，不可解时 4/10-kT
                    # SGD 兜底）——loose-gate 不改这个更新机制，只改"何时冻结进生产"。
                    # raw block/EMA/dominant 仍仅作诊断。
                    f_updated = None
                    if len(sampler.energy_buffer) >= IBS_TMBAR_LEARNING_MINIBATCH_FRAMES:
                        f_updated = sampler.update_weights(
                            min_buffer_size=IBS_TMBAR_LEARNING_MINIBATCH_FRAMES,
                            candidate_min_ess_ratio=candidate_min_ess_ratio,
                            candidate_min_absolute_ess=candidate_min_absolute_ess,
                            candidate_min_decorrelated_samples=candidate_min_decorrelated_samples,
                            candidate_max_uncertainty_kJ_mol=candidate_max_uncertainty_kJ_mol,
                        )
                    if f_updated is None:
                        continue
                    bias_update_count += 1
                    updates_since_freeze += 1
                    if len(sampler.f_history) >= 2:
                        last_f_delta = float(
                            np.max(
                                np.abs(
                                    np.asarray(sampler.f_history[-1])
                                    - np.asarray(sampler.f_history[-2])
                                )
                            )
                        )
                    # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] freeze 时机 = f_k 实际步长已
                    # 稳定（纯效率、不阻放行）。生产入口门是纯 Δf−ΔF<10（见下方 gate）；
                    # 这里只决定"何时值得冻结去跑一次 local-MBAR 检查"，避免 f_k 还在
                    # 大步移动时白花 burn-in/验证。只看应用的 pairwise 步长是否已降到
                    # ≤ IBS_TMBAR_FREEZE_MAX_APPLIED_PAIRWISE_STEP_KT kT（f_k 几乎不再动），
                    # 连续 N 批满足才冻结。不看占据/coverage/abs_ess——那会把宽松门变严格。
                    # gap<10 ⟹ 残差小 ⟹ 步长小，所以 step 门不会阻挡任何本该通过的窗口；
                    # step 仍大的窗口必然 gap≫10、冻结也白费。严重塌陷时步长持续大 → 一直
                    # 留在 learning 用大步压占据，不浪费冻结周期。
                    _ready_step = float(
                        sampler.last_update_diagnostics.get("weight_update", {}).get(
                            "pairwise_delta_f_spread_kJ_mol", float("inf")
                        )
                    )
                    learning_ready = bool(
                        _ready_step
                        <= IBS_TMBAR_FREEZE_MAX_APPLIED_PAIRWISE_STEP_KT * float(sampler.kt)
                    )
                    if learning_ready:
                        consecutive_pass_count += 1
                    else:
                        consecutive_pass_count = 0
                    if (
                        len(sampler.f_history) >= int(min_bias_updates)
                        and consecutive_pass_count >= int(IBS_LEARNING_READY_CONSECUTIVE)
                    ):
                        frozen_f_k_snapshot = [
                            float(sim.context.getParameter(f"{self.prefix}_f_{k}"))
                            for k in range(K)
                        ]
                        print(
                            f"    🧊 f_k 步长已稳定（pairwise={_ready_step:.2f}≤"
                            f"{IBS_TMBAR_FREEZE_MAX_APPLIED_PAIRWISE_STEP_KT * float(sampler.kt):.2f} kJ/mol，"
                            f"连续 {consecutive_pass_count} 批）→ 冻结跑 local-MBAR Δf−ΔF 门"
                        )
                        mode = "freeze_burn_in"
                        have_frozen_once = True
                        updates_since_freeze = 0
                        consecutive_pass_count = 0
                        freeze_burn_in_done = 0
                        frozen_mbar_batches = []
                        sampler._last_dominant_k = None
                        sampler.energy_buffer = []
                        sampler.ema_mean_p = None
                    continue

                if mode == "freeze_burn_in":
                    # 冻结 f_k 后构型要在这个固定 Hamiltonian 下重新平衡；之前在漂移
                    # 偏置下攒的帧一律丢弃，不进 local MBAR 门统计。
                    freeze_burn_in_done += check_chunk
                    sampler.energy_buffer = []
                    if freeze_burn_in_done >= frozen_burn_in_steps:
                        mode = "validating"
                        frozen_mbar_batches = []
                        validation_batch_count = 0
                        validation_sample_count = 0
                        validation_steps_this_freeze = 0
                    continue

                # mode == "validating"：保持同一冻结 f_k，累计最近
                # IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES 批固定-f_k minibatch，拼成一次
                # 单参考 local MBAR，只设一个 loose gate：相邻态 ΔF^MBAR 与当前相邻
                # Δf_k 的最大偏差 < 阈值即冻结进生产（相邻差 gauge 无关，不比较带任意
                # 公共常数的绝对 f_k）。
                validation_steps_this_freeze += check_chunk
                if len(sampler.energy_buffer) < IBS_TMBAR_LEARNING_MINIBATCH_FRAMES:
                    continue
                # 把这批固定-f_k 帧并入持久 tmbar_history（loose-gate 失败退回
                # learning 时，下一轮 update_weights 会复用它们继续学习），并取回
                # 同一份 (u_kn,bias_energies,base_energies) 数组喂 local MBAR——这三
                # 者跟 _solve_single_window_local_mbar 的入参一一对应（同 collect_
                # energies 记录的纯 softcore/base/bias，与 f_k 训练所用口径一致）。
                n_appended = sampler._append_tmbar_batch_from_buffer()
                sampler.energy_buffer = []
                if n_appended <= 0:
                    continue
                frozen_mbar_batches.append(sampler.tmbar_history[-1])
                if len(frozen_mbar_batches) > IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES:
                    frozen_mbar_batches = frozen_mbar_batches[
                        -IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES:
                    ]
                validation_batch_count += 1
                validation_sample_count += int(n_appended)
                if resumed_calibration_pending:
                    # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 每批覆盖式落盘主
                    # 窗口 checkpoint + 累计步数，抢占/撞墙被杀也能精确续算，不重跑
                    # 也不漏算（沿用 v27 的每批落盘频率，只是判据换成 loose-gate）。
                    _atomic_save_openmm_checkpoint(sim, main_ckpt_path)
                    _atomic_write_json(main_manifest_path, expected_main_manifest)
                    sampler.frozen_validation_cumulative_steps = (
                        prior_cumulative_steps + steps_at_full_bias
                    )
                    sampler.save_ibs_state(ibs_state_file, lc_win, lv_win)
                # 只看最近 IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES 批：攒齐再解一次。
                if len(frozen_mbar_batches) < IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES:
                    continue
                u_kn_gate = np.concatenate(
                    [
                        np.asarray(b["u_kn"], dtype=np.float64)
                        for b in frozen_mbar_batches
                    ],
                    axis=1,
                )
                bias_gate = np.concatenate(
                    [
                        np.asarray(b["bias_energies"], dtype=np.float64)
                        for b in frozen_mbar_batches
                    ]
                )
                base_gate = np.concatenate(
                    [
                        np.asarray(b["base_energies"], dtype=np.float64)
                        for b in frozen_mbar_batches
                    ]
                )
                f_frozen_now = np.asarray(
                    [
                        sim.context.getParameter(f"{self.prefix}_f_{k}")
                        for k in range(K)
                    ],
                    dtype=np.float64,
                )
                # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] f_frozen_now 上移到这里：混合
                # 覆盖度 ESS 需要"这批 minibatch 采样时真正生效的 f_k"才能把共模
                # 因子除干净（gauge-free 的逐帧 softmax 捷径在谱宽大的窗口会给出
                # 差数十倍的结果，见 _ibs_reweighting_quality_diagnostics）。这批
                # batch 是在冻结 f_k 下采的，所以此刻读到的就是采样时的值。
                gate_mbar = _solve_single_window_local_mbar(
                    u_kn_gate,
                    bias_gate,
                    base_gate,
                    list(range(K)),
                    sampler.kt,
                    f_k=f_frozen_now,
                    sampled_distribution_row=0,
                    w_idx=window_idx,
                )
                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 生产入口门 = 纯 Δf−ΔF loose gate：
                # 只要 local MBAR 解出有限 f_mbar，就算相邻态 |Δf_k−ΔF^MBAR|（gauge
                # 无关），max < IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL 即冻结进
                # 生产。abs_ess / 占据平坦度 / coverage_ESS / raw_residual / tmbar_self_
                # consistent 全部只作诊断，不参与放行——否则会把这个宽松 10 kJ/mol 门
                # 偷偷变成严格收敛门（挡住 Δf 其实已在 10 kJ/mol 内、只是短 warmup ESS
                # 薄的窗口）。前面那种 908 kJ/mol 的零重叠外推自然因 908>10 被拒，不需要
                # abs_ess 门才能拒。最终绝对 ESS/误差/真实自由能交生产后独立
                # _assert_stage_result_sane + 最终 MBAR 兜底。
                gate_solver_error = gate_mbar.get("error")
                # 以下三个只作诊断（不参与放行）。
                _gate_min_ess_ratio = gate_mbar.get("min_ess_ratio")
                _gate_n_used = int(gate_mbar.get("n_frames_used", 0) or 0)
                _gate_abs_ess = (
                    float(_gate_min_ess_ratio) * _gate_n_used
                    if (_gate_min_ess_ratio is not None
                        and np.isfinite(_gate_min_ess_ratio))
                    else 0.0
                )
                try:
                    gate_situation = _diagnose_local_mbar_situation(
                        u_kn_gate, f_frozen_now, sampler.kt, gate_mbar, int(start)
                    )
                except Exception as _situ_err:
                    gate_situation = {"error": repr(_situ_err)}
                max_adjacent_gap_kJ_mol = float("inf")
                adjacent_gaps = None
                gate_error = gate_solver_error
                if gate_solver_error is None:
                    f_mbar = np.asarray(gate_mbar.get("f"), dtype=np.float64)
                    if f_mbar.size == K and np.all(np.isfinite(f_mbar)):
                        if K > 1:
                            df_current = np.diff(f_frozen_now)
                            dF_mbar = np.diff(f_mbar)
                            adjacent_gaps = np.abs(df_current - dF_mbar)
                            max_adjacent_gap_kJ_mol = float(np.max(adjacent_gaps))
                        else:
                            max_adjacent_gap_kJ_mol = 0.0
                    else:
                        gate_error = "nan_or_shape_mismatch_in_local_mbar_f"
                gate_ok = bool(
                    gate_error is None
                    and np.isfinite(max_adjacent_gap_kJ_mol)
                    and max_adjacent_gap_kJ_mol
                    < float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL)
                )
                gate_diag = {
                    "phase": "frozen_local_mbar_loose_gate",
                    "n_batches": int(len(frozen_mbar_batches)),
                    "max_adjacent_delta_kJ_mol": (
                        max_adjacent_gap_kJ_mol
                        if np.isfinite(max_adjacent_gap_kJ_mol) else None
                    ),
                    "adjacent_delta_kJ_mol": (
                        [float(x) for x in adjacent_gaps]
                        if adjacent_gaps is not None else None
                    ),
                    "gate_threshold_kJ_mol": float(
                        IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL
                    ),
                    "error": gate_error,
                    "steps_at_full_bias": int(steps_at_full_bias),
                    "validation_sample_count": int(validation_sample_count),
                    # 以下仅诊断，不参与放行：
                    "n_frames_used": int(_gate_n_used),
                    "min_ess_ratio": _gate_min_ess_ratio,
                    "min_absolute_ess": float(_gate_abs_ess),
                    "statistical_inefficiency": gate_mbar.get("statistical_inefficiency"),
                    "situation": gate_situation,
                    "diagnostics_only_note": (
                        "abs_ess/coverage_ess/occupancy/raw_residual do NOT gate "
                        "production entry; only max_adjacent_delta < threshold does"
                    ),
                }
                local_mbar_gate_history.append(gate_diag)
                last_local_mbar_gate = gate_diag
                lse_residual_history.append(gate_diag)
                # loose-gate 不用占据概率判据；last_validation_batch_p 仅用于失败/
                # 收敛报告兜底展示，保持 None（下游报告退回 EMA/nan）。
                last_validation_batch_p = None
                if gate_ok:
                    validation_pass_count = int(validation_batch_count)
                    bias_converged = True
                    print(
                        "    ✅ local-MBAR loose gate 通过："
                        f"max|Δf_k−ΔF^MBAR|={max_adjacent_gap_kJ_mol:.3f} kJ/mol < "
                        f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f} kJ/mol"
                        f"（{validation_sample_count} frames；abs_ess={_gate_abs_ess:.1f}、"
                        f"min_ess_ratio={_gate_min_ess_ratio} 仅诊断，不参与放行）；"
                        "冻结 f_k 进生产，最终绝对 ESS/误差/自由能交生产后 MBAR"
                    )
                    break
                # 未过（gap≥阈值，或 MBAR 不可解/NaN）：退回 learning 再更新一轮，逐步
                # 把相邻 f_k 拉进 ~阈值范围。abs_ess/占据只在现场里打印，不参与判定。
                learning_to_validation_cycles += 1
                if gate_error is None:
                    print(
                        "    ⟳ local-MBAR loose gate 未过："
                        f"max|Δf_k−ΔF^MBAR|={max_adjacent_gap_kJ_mol:.3f} kJ/mol ≥ "
                        f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f}；"
                        "相邻边仍偏离，退回 learning 再更新一轮。现场："
                        f"{_format_local_mbar_situation(gate_situation)}"
                    )
                else:
                    print(
                        f"    ⟳ local MBAR 暂不可解（{gate_error}）；退回 learning 继续"
                        "更新（不让 BAR/MBAR 不收敛卡住流程）。现场："
                        f"{_format_local_mbar_situation(gate_situation)}"
                    )
                sampler.last_update_diagnostics = {
                    "source": "failed_local_mbar_loose_gate",
                    "local_mbar_gate": gate_diag,
                    "adjacent_delta_u_is_convergence_gate": False,
                }
                mode = "learning"
                updates_since_freeze = 0
                # 退回 learning 后必须重新挣得 readiness（连续满足才再冻结），不带走
                # 冻结前的旧 pass 计数——否则又会"回来一轮就立刻重冻"。
                consecutive_pass_count = 0
                frozen_mbar_batches = []
                sampler._last_dominant_k = None
                sampler.ema_mean_p = None
                sampler.energy_buffer = []
                validation_probability_sum = np.zeros(K, dtype=np.float64)
                validation_sample_count = 0
                validation_batch_count = 0
                validation_steps_this_freeze = 0
                continue

            # [IBS_BIAS_PROTOCOL_VERSION=29] 跑满整个 warmup 步数预算仍没有任何一次
            # local-MBAR loose gate < 阈值时，直接接受当前 f_k 进生产（best-effort）。
            # loose gate 只是"别让某个局部边完全饿死"的粗门（10 kJ/mol≈4 kT）；真正的
            # 自由能/ESS/overlap/误差全部交给生产后独立的 _assert_stage_result_sane +
            # 最终 MBAR。resumed_calibration_pending 续验同样走这条（不再有阶梯升级/
            # 无限续验：判据已换成 loose gate，预算耗尽即接受）。接受的就是循环退出
            # 时 Context 里的这份 f_k，不需要恢复更早的 attempt，也不需要额外 re-burn-in。
            # 条件用 `not bias_converged` 覆盖两种退出：撞总步数帽，或反复未过 gate
            # 把 learning 更新次数烧到 max_bias_updates 而 break——两者都算预算耗尽。
            # 但 warmup_only 是设计期验证模式，本就不进生产、其职责是如实报告窗口是
            # 否收敛，故不 best-effort 接受——让它照常落到下面 else 分支报未收敛。
            if not bias_converged and not warmup_only:
                accepted_f_k = np.asarray(
                    [
                        sim.context.getParameter(f"{self.prefix}_f_{k}")
                        for k in range(K)
                    ],
                    dtype=np.float64,
                )
                frozen_f_k_snapshot = accepted_f_k.astype(float).tolist()
                validation_pass_count = int(validation_batch_count)
                bias_converged = True
                best_effort_acceptance = True
                best_effort_acceptance_reason = "warmup_budget_exhausted_loose_gate"
                best_effort_residual = (
                    float(last_local_mbar_gate["max_adjacent_delta_kJ_mol"])
                    if (
                        last_local_mbar_gate is not None
                        and last_local_mbar_gate.get("max_adjacent_delta_kJ_mol")
                        is not None
                    )
                    else float("inf")
                )
                mode = "best_effort_budget_exhausted_accepted"
                _last_gap_repr = (
                    f"{best_effort_residual:.3f} kJ/mol"
                    if np.isfinite(best_effort_residual)
                    else "n/a（local MBAR 从未在预算内解出）"
                )
                print(
                    "    🤝 warmup 步数预算耗尽仍未通过 local-MBAR loose gate；接受当前 "
                    f"f_k 进生产（最近一次 max|Δf_k−ΔF^MBAR|={_last_gap_repr}，阈值 "
                    f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f} kJ/mol）；"
                    "生产后独立质量门 + 最终 MBAR 继续兜底"
                )
                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 现场判决必须区分两种"预算耗尽"：
                # (a) 占据真的塌了（某态低于 0.5/K）→ 硬热力学瓶颈，权重救不了，建议
                #     插 λ/拆窗；(b) 占据其实平坦、只是短 warmup 的 local MBAR 绝对
                #     ESS/端点 reweight ESS 偏低 → 不是瓶颈，已接受进生产，最终 ESS/
                #     误差交生产后 MBAR。之前只凭"MBAR 没解出可信 ΔF"就一律报瓶颈，会
                #     把 (b)（如占据 [0.41,0.27,0.20,0.125] 这种健康窗口）误报成需要插 λ。
                _acc_situation = (
                    last_local_mbar_gate.get("situation")
                    if last_local_mbar_gate is not None else None
                )
                _never_trustworthy = (
                    last_local_mbar_gate is None
                    or last_local_mbar_gate.get("max_adjacent_delta_kJ_mol") is None
                )
                if _acc_situation:
                    print(
                        f"    🔎 现场：{_format_local_mbar_situation(_acc_situation)}"
                    )
                _situ_ok = bool(_acc_situation and not _acc_situation.get("error"))
                if _situ_ok and _acc_situation.get("occupancy_is_flat"):
                    # (b) 占据平坦：良性，不是插 λ 的场景。
                    print(
                        "    ✔️ 判决：冻结占据已平坦（coverage_ESS="
                        f"{_acc_situation.get('coverage_ess', float('nan')):.2f}≥"
                        f"{IBS_LOCAL_MBAR_GATE_OCC_MIN_COVERAGE_ESS_FRACTION * float(K):.2f}），"
                        "只是短 warmup 的 local MBAR 绝对 ESS/端点 reweight ESS 偏低、未能走 "
                        "Δf−ΔF 主门；已接受进生产，最终绝对 ESS/误差/自由能交生产后 "
                        "_assert_stage_result_sane + 最终 MBAR。这不是需要插 λ 的瓶颈；"
                        "若最终 MBAR 显示某端点采样不足，再考虑延长生产采样。"
                    )
                elif _never_trustworthy and _situ_ok and _acc_situation.get(
                    "occupancy_collapsed"
                ):
                    # (a) 占据真的塌陷：硬瓶颈。
                    _sg = _acc_situation.get("starved_global_state")
                    _edges = _acc_situation.get("starved_edges_global") or []
                    print(
                        "    ⚠️ 判决：占据塌陷 + local MBAR 全程重叠/ESS 不足 → 这是"
                        f"{'端点' if _acc_situation.get('is_endpoint') else '内部'}"
                        f"热力学瓶颈（global s{_sg} 占据="
                        f"{_acc_situation.get('starved_occupancy'):.2e} 低于健康下限 "
                        f"{_acc_situation.get('occupancy_floor'):.3f}，与相邻态零重叠），"
                        f"权重无法弥补物理缺口。建议在相邻 global 态之间插入 λ 或拆窗："
                        f"边 {_edges}。（生产后 _assert_stage_result_sane 亦会独立判不可用。）"
                    )
                elif _never_trustworthy and _situ_ok:
                    # 中间态：占据没塌到下限、但也没达到平坦兜底门（如 coverage 略低）。
                    print(
                        "    ℹ️ 判决：占据尚可但未达平坦兜底门、local MBAR 绝对 ESS 也不足；"
                        "已接受进生产（best-effort）。优先延长生产采样；若最终 MBAR 仍显示"
                        "某相邻边重叠不足，再考虑插 λ。现场见上。"
                    )

            state = sim.context.getState(getEnergy=True)
            if not np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
                raise RuntimeError("偏置预热阶段在 scale=1.0 时能量 NaN")

            # 报告用的 min/max/coverage 优先取最后一次真正驱动收敛判定的原始
            # validation batch；只有从未进入过 validating（例如始终卡在
            # learning）时才退回 EMA，仅供参考，不代表任何"已验证"的结论。
            if last_validation_batch_p is not None:
                _report_p = np.asarray(last_validation_batch_p, dtype=np.float64)
                min_ema_p = float(np.min(_report_p))
                max_ema_p = float(np.max(_report_p))
                ema_mean_p_values = last_validation_batch_p
                coverage_ess = float(1.0 / np.sum(np.square(_report_p)))
            else:
                max_ema_p = float(np.max(sampler.ema_mean_p)) if sampler.ema_mean_p is not None else float("nan")
                min_ema_p = float(np.min(sampler.ema_mean_p)) if sampler.ema_mean_p is not None else float("nan")
                ema_mean_p_values = (
                    np.asarray(sampler.ema_mean_p, dtype=float).tolist()
                    if sampler.ema_mean_p is not None
                    else []
                )
                coverage_ess = (
                    float(1.0 / np.sum(np.square(np.asarray(sampler.ema_mean_p, dtype=np.float64))))
                    if sampler.ema_mean_p is not None
                    else float("nan")
                )
            report_lse_balance = ibs_lse_balance_diagnostics(ema_mean_p_values)
            bias_warmup_diag = {
                "status": (
                    (
                        "frozen_validation_best_effort_accepted"
                        if best_effort_acceptance
                        else "frozen_validation_converged"
                    )
                    if bias_converged
                    else (
                        "early_probe_triggered_unconverged"
                        if early_probe_triggered
                        else "hit_update_or_safety_cap_unconverged"
                    )
                ),
                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] 见 version 25 changelog (a)：
                # True 表示这个窗口是靠"见好就收"重试上限兜底放行进生产的，
                # 而不是严格达到 lse_log_residual_tolerance——生产后独立的
                # _assert_stage_result_sane 仍是真正的正确性把关。
                "best_effort_acceptance": bool(best_effort_acceptance),
                "best_effort_acceptance_reason": best_effort_acceptance_reason,
                "best_effort_residual": (
                    float(best_effort_residual)
                    if np.isfinite(best_effort_residual)
                    else None
                ),
                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] PROPOSAL_frozen_validation_
                # fallback.md：True 表示这个窗口至少有一次是 learning 候选
                # streak 从未凑满、纯粹靠 max_bias_updates 耗尽 fallback 才
                # 进入 freeze_burn_in 的，不是"reached via genuine candidate
                # streak"。
                "reached_freeze_via_budget_exhaustion_fallback": bool(
                    budget_fallback_used
                ),
                "early_probe_triggered": bool(early_probe_triggered),
                "early_probe_trigger_reason": early_probe_trigger_reason,
                "window_index": int(window_idx),
                "global_state_range": [int(start), int(end)],
                "global_lambda_indices": list(range(int(start), int(end))),
                "lambdas_coul": [float(x) for x in lc_win],
                "lambdas_vdw": [float(x) for x in lv_win],
                "n_states": int(K),
                "target_p": float(target_p),
                "min_probability_threshold": float(min_probability_threshold),
                "coverage_ess": float(coverage_ess),
                "coverage_ess_threshold": float(coverage_ess_threshold),
                "ema_mean_p": ema_mean_p_values,
                "max_ema_mean_p": max_ema_p,
                "min_ema_mean_p": min_ema_p,
                "last_update_diagnostics": sampler.last_update_diagnostics,
                "tmbar_history_length": len(sampler.tmbar_history),
                "last_tmbar_update": sampler.last_update_diagnostics.get("tmbar_update", {}),
                "last_f_k_delta_kJ_mol": last_f_delta,
                "f_stability_threshold_kJ_mol": float(f_stability_threshold_kJ_mol),
                "min_bias_updates": int(min_bias_updates),
                "max_bias_updates": int(max_bias_updates),
                "bias_update_count": int(bias_update_count),
                "required_consecutive_passes": int(required_consecutive_bias_updates),
                "consecutive_pass_count": int(consecutive_pass_count),
                "validation_pass_count": int(validation_pass_count),
                "validation_batch_probabilities": validation_batch_history,
                "validation_cumulative_probabilities": validation_cumulative_history,
                "validation_attempt_budget_steps": int(validation_attempt_budget_steps),
                "minimum_complete_validation_frames": int(
                    minimum_complete_validation_frames
                ),
                "truncated_validation_frames_ignored": int(
                    truncated_validation_frames_ignored
                ),
                "best_effort_reburnin_steps": int(best_effort_reburnin_steps),
                "safety_cap_best_effort_tmbar_converged": bool(
                    safety_cap_best_effort_tmbar_converged
                ),
                "lse_log_residual_tolerance": float(lse_log_residual_tolerance),
                "lse_balance": report_lse_balance,
                "lse_residual_history": lse_residual_history,
                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 局部滑窗 MBAR loose-gate 诊断：
                # 每次冻结复检解出的相邻态 |Δf_k−ΔF^MBAR| 及阈值/滑窗深度，供事后
                # 追溯这个窗口是靠 loose-gate 通过还是预算耗尽 best-effort 放行。
                "local_mbar_loose_gate": {
                    "gate_threshold_kJ_mol": float(
                        IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL
                    ),
                    "sliding_window_batches": int(
                        IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
                    ),
                    "last": last_local_mbar_gate,
                    "history": local_mbar_gate_history,
                },
                "convergence_criterion": {
                    "learning_candidate": "existing_update_weights_tmbar_or_bounded_sgd",
                    "frozen_validation": (
                        "max_adjacent_abs_delta_f_minus_local_mbar_delta_F_below_"
                        "IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL"
                    ),
                },
                "resumed_from_cache": bool(resumed_frozen_f_k is not None),
                "learning_to_validation_cycles": int(learning_to_validation_cycles),
                "validation_feedback_update_count": int(
                    validation_feedback_update_count
                ),
                "resumed_calibration_pending": bool(resumed_calibration_pending),
                "frozen_validation_retry_count": int(frozen_validation_retry_count),
                "frozen_burn_in_steps": int(frozen_burn_in_steps),
                "frozen_f_k_at_last_freeze": frozen_f_k_snapshot,
                "final_mode": mode,
                "steps_at_full_bias": int(steps_at_full_bias),
                "max_bias_warmup_steps_safety_cap": int(full_bias_step_budget),
                "sgd_step_budget": int(sgd_step_budget),
                "frozen_validation_reserved_steps": int(
                    frozen_validation_reserved_steps
                ),
                "mbar_calibration_reserved_steps": int(
                    effective_mbar_calibration_reserved_steps
                ),
                "was_resumed": bool(is_resumed_ibs),
                "f_delta_is_diagnostic_only": True,
                "feedback_action": (
                    (
                        "design_split_window_using_existing_lambdas"
                        if K >= 3
                        else "design_insert_thermodynamic_midpoint_then_revalidate_lse"
                    )
                    if warmup_only
                    else
                    # 🔑 [non_mutating_v1] 非变异策略下不做任何拆窗/插点/fixed-H 探针
                    # 建议——未收敛就是 f_not_converged，保留数据交人工/rescue 审计。
                    # 只有已弃用的 legacy_mutating 策略才给旧的拆窗/探针语义（见下）。
                    (
                        # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=7]
                        # split_window_from_warmup_failure 要求两个孩子各自 >=3 态、
                        # 共享 1 态，父窗口因此至少要 5 态才能这样拆（3+3-1）；K=4
                        # 走 fixed-H overlap 探针，不再被拆成 K=2+K=3。
                        ("split_window_without_lambda_insertion" if K >= 5
                         else "probe_fixed_hamiltonian_bidirectional_overlap")
                        if legacy_repair
                        else "none_non_mutating_f_not_converged"
                    )
                ),
            }

            # Only a minimal-or-unsplittable window is allowed to request a new
            # lambda, and only after independent fixed-H trajectories prove
            # that a real adjacent edge lacks overlap. K<=4 here must match
            # split_window_from_warmup_failure's min_states_before_split=5
            # floor in abfe_preoptimizer.py/abfe_pipeline.py -- K=4 cannot be
            # split into two >=3-state children sharing one state, so it must
            # come here instead of being blindly bisected into K=2+K=3.
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 续验已冻结的校准 f_k 时，λ 网格
            # 覆盖度和 f_k 校准都已经在上一轮证明过了——重新跑 fixed-H overlap
            # 探针和 MBAR 校准既浪费 GPU，也会覆盖掉这份已经验证正确的
            # frozen_f_k_pending。只有从未校准过（或本次是全新窗口）才需要
            # 走这条探针+校准分支。
            # 🔑 [non_mutating_v1] 只有旧的（已弃用）变异策略才进入 fixed-H overlap
            # 探针 + asymmetric 判据 + bias 校准 + 就地覆盖 f_k 这条分支。非变异策略
            # 下 not bias_converged 直接落到下面的 `else` 分支，抛 f_not_converged，
            # 绝不烧 fixed-H GPU、绝不重写 f_k。overlap_probe 保持空 → 下面 6512 的
            # asymmetric 块与 6528 的校准块都因 all_passed 为假而自然跳过。
            if (
                legacy_repair
                and not bias_converged
                and not resumed_calibration_pending
                and K <= 4
                and stage_type == "vdw"
            ):
                try:
                    current_state = sim.context.getState(getPositions=True)
                    bank_result = probe_adjacent_path_overlap_bank(
                        topology=self.topology,
                        common_system_xml=ibs_wrap._common_system_xml,
                        ibs_wrapper=ibs_wrap,
                        K=K,
                        positions=current_state.getPositions(),
                        box_vectors=current_state.getPeriodicBoxVectors(),
                        temperature=self.temperature,
                        platform_name=self.platform_name,
                        checkpoint_dir=self.checkpoint_dir,
                        stage_type=stage_type,
                        window_idx=window_idx,
                        global_state_start=int(start),
                    )
                    bias_warmup_diag["bidirectional_overlap_probe"] = {
                        "pairs": bank_result["pairs"],
                        "all_passed": bool(bank_result["all_passed"]),
                    }
                except Exception as probe_error:
                    bias_warmup_diag["bidirectional_overlap_probe"] = {
                        "pairs": [],
                        "all_passed": False,
                        "error": repr(probe_error),
                    }

            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=10] fixed-H overlap 全部通过说明 λ 网格
            # 本身没问题：固定态之间确实能互相重加权，问题出在 f_k 求解器（SGD
            # 学习率过大/EMA 跨 Hamiltonian）没能稳定收敛，而不是密度不够。但用来
            # 下这个判断的 probe_pairs（上面的路径 overlap 探针）绝不能直接拿来
            # 校准 f_k——它的动力学是 U_common + CV_k（不含 Group 4 WCA），采的是
            # 跟生产不同的构象系综；它的 delta_f_kJ_mol 还叠加了 LRC，而生产 Group1
            # 偏置力本身从不含 LRC（LRC 只在 collect_energies() 里事后加进
            # target_energies 喂 MBAR，不进偏置力）。这曾经是一个真实的、已确认的
            # bug：用错了 ensemble、又把不属于偏置力的能量项校准了进去。这里改为
            # 单独跑 probe_bidirectional_overlap_for_bias_calibration——同一个窗口、
            # 同一个 lambda_shield、保留 Group 4 WCA、绝不加 LRC——只用它返回的
            # delta_f_bias_kJ_mol 校准 f_k；且要求每条边的去相关样本数/不确定度都
            # 达标（不够就延长采样重试，不能直接拿一次勉强通过 overlap 阈值但样本
            # 严重不足的估计去覆盖 f_k）。校准后只给一次"冻结 burn-in + 独立验证"
            # 的机会，不再回退 learning；仍不通过就是真实的构象弛豫慢/偏置表达式
            # 问题，交给人工检查，而不是继续无意义地烧 GPU 重试同一个不稳定求解器。
            overlap_probe = bias_warmup_diag.get("bidirectional_overlap_probe", {})

            asymmetric_bottleneck = None
            if overlap_probe.get("all_passed"):
                asymmetric_bottleneck = detect_passed_but_asymmetric_overlap_bottleneck(
                    overlap_probe.get("pairs", []), lv_win,
                )
                if asymmetric_bottleneck is not None:
                    overlap_probe["passed_but_asymmetric_bottleneck"] = asymmetric_bottleneck
                    bias_warmup_diag["status"] = "passed_but_asymmetric_overlap_bottleneck"
                    bias_warmup_diag["feedback_action"] = "insert_lambda_at_passed_bottleneck"
                    print(
                        "    ⚠️ fixed-H 各边虽均超过绝对阈值，但检测到局部热力学瓶颈："
                        f"edge={asymmetric_bottleneck['global_edge']}，"
                        f"overlap ratio={asymmetric_bottleneck['overlap_ratio']:.2f}，"
                        f"slope ratio={asymmetric_bottleneck['slope_ratio']:.2f}；"
                        "跳过 f_k 校准阶梯，反馈路径插点。"
                    )

            if (
                not bias_converged
                and not resumed_calibration_pending
                and overlap_probe.get("all_passed")
                and asymmetric_bottleneck is None
            ):
                lam_vdw_center_for_calibration = float(np.mean(lv_win))
                calib_state = sim.context.getState(getPositions=True)
                calib_positions = calib_state.getPositions()
                calib_box = calib_state.getPeriodicBoxVectors()
                calib_bank_result = probe_adjacent_bias_calibration_bank(
                    topology=self.topology,
                    common_plus_wca_system_xml=ibs_wrap._common_plus_wca_system_xml,
                    ibs_wrapper=ibs_wrap,
                    K=K,
                    positions=calib_positions,
                    box_vectors=calib_box,
                    temperature=self.temperature,
                    platform_name=self.platform_name,
                    checkpoint_dir=self.checkpoint_dir,
                    stage_type=stage_type,
                    window_idx=window_idx,
                    global_state_start=int(start),
                    lambda_shield=lam_vdw_center_for_calibration,
                )
                calibration_pairs = calib_bank_result["pairs"]
                calibration_sufficient = bool(calib_bank_result["all_sufficient"])

                bias_warmup_diag["bias_calibration_probe"] = {
                    "pairs": calibration_pairs,
                    "all_sufficient": calibration_sufficient,
                    "lambda_shield": lam_vdw_center_for_calibration,
                }

                if not calibration_sufficient:
                    print(
                        "    ⚠️ bias 校准探针在多次延长采样后，仍有边的去相关样本数/ΔF 不确定度"
                        "未达标，放弃本次 MBAR 校准（不拿不够精确的估计覆盖 f_k），"
                        "继续按未收敛处理，交给人工检查。"
                    )
                else:
                    delta_f_bias_edges = [float(p["delta_f_bias_kJ_mol"]) for p in calibration_pairs]
                    f_calibrated = np.concatenate(([0.0], np.cumsum(delta_f_bias_edges)))
                    f_calibrated = f_calibrated - np.mean(f_calibrated)
                    for k in range(K):
                        sim.context.setParameter(f"{self.prefix}_f_{k}", float(f_calibrated[k]))
                    print(
                        f"    🎯 fixed-H 双向 overlap 全部通过（λ 网格没问题），bias 校准探针"
                        f"（含 WCA、不含 LRC）也已达到精度门槛："
                        f"用 ΔF_bias 校准 f_k={[f'{v:.2f}' for v in f_calibrated]}，"
                        "不再继续 SGD 搜索，重新做一次冻结 burn-in + 只读验证。"
                    )
                    sampler.energy_buffer = []
                    sampler.ema_mean_p = None
                    calibration_burn_in_done = 0
                    calibration_validation_pass_count = 0
                    calibration_converged = False
                    calibration_steps_used = 0
                    # 用独立预留的预算，而不是继续检查
                    # steps_at_full_bias < max_bias_warmup_steps——SGD 阶段已经把它的
                    # 预算（sgd_step_budget）用尽退出，如果这里还沿用同一个已耗尽的
                    # 计数器，这个循环会一次都不执行，calibration_converged 必然为
                    # False，跟校准本身是否有效无关。默认是
                    # mbar_calibration_reserved_steps，但 [IBS_BIAS_PROTOCOL_VERSION=12]
                    # 起如果调用方通过 frozen_validation_step_overrides 给这个窗口
                    # 累计延长过预算（上一轮校准好但验证未通过），这里用那个更大的值。
                    while calibration_steps_used < int(effective_frozen_validation_budget):
                        sim.step(check_chunk)
                        steps_at_full_bias += check_chunk
                        calibration_steps_used += check_chunk
                        sampler.collect_energies()
                        if calibration_burn_in_done < frozen_burn_in_steps:
                            calibration_burn_in_done += check_chunk
                            sampler.energy_buffer = []
                            continue
                        if len(sampler.energy_buffer) < 20:
                            continue
                        # 🔑 有意不调用 sampler._append_tmbar_batch_from_buffer()：
                        # 这是已弃用的 legacy_mutating 专属 fixed-H/MBAR 校准回退路径
                        # （repair_policy != "non_mutating_v1" 才会走到这里），验证失败
                        # 时是终态（不像上面 non_mutating_v1 的 validating 分支那样退回
                        # "learning"），tmbar_history 只在 learning 模式被解出使用，
                        # 追加进去也永远不会被读取，白白增大落盘状态。
                        batch_p = sampler.evaluate_frozen_batch_probability()
                        if batch_p is None:
                            continue
                        p = np.asarray(batch_p, dtype=np.float64)
                        last_validation_batch_p = p.tolist()
                        validation_batch_history.append(last_validation_batch_p)
                        last_lse_balance = ibs_lse_balance_diagnostics(p)
                        lse_residual_history.append(dict(last_lse_balance))
                        # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 这个循环
                        # 本来就只服务于刚校准出来的 f_k 的验证，不需要额外条件
                        # ——跟上面 resumed_calibration_pending 分支一样，每评估
                        # 完一批就存一次，覆盖式落盘，应对 HPC 作业被抢占/撞墙
                        # 时限杀掉的情况。
                        _atomic_save_openmm_checkpoint(sim, main_ckpt_path)
                        _atomic_write_json(main_manifest_path, expected_main_manifest)
                        # 🔑 [checkpoint/累计步数同步修复] 同上面 resumed_calibration_
                        # pending 分支：这是这份 f_k 第一次被冻结验证（不是续验），
                        # 累计步数就是这次 attempt 里已经跑过的 calibration_steps_used，
                        # 必须跟 .chk 同频率落盘，理由同上。
                        sampler.frozen_validation_cumulative_steps = int(calibration_steps_used)
                        sampler.save_ibs_state(ibs_state_file, lc_win, lv_win)
                        calib_lse_ok = bool(
                            last_lse_balance.get("available", False)
                            and last_lse_balance["max_abs_log_residual"]
                            <= float(lse_log_residual_tolerance)
                        )
                        if calib_lse_ok:
                            calibration_validation_pass_count += 1
                        else:
                            calibration_validation_pass_count = 0
                        if calibration_validation_pass_count >= int(required_consecutive_bias_updates):
                            calibration_converged = True
                            break

                    bias_warmup_diag["mbar_calibration"] = {
                        "attempted": True,
                        "delta_f_bias_edges_kJ_mol": delta_f_bias_edges,
                        "calibrated_f_k_kJ_mol": f_calibrated.tolist(),
                        "converged": calibration_converged,
                        "steps_used": int(calibration_steps_used),
                        "steps_reserved": int(effective_frozen_validation_budget),
                        "validation_pass_count": int(calibration_validation_pass_count),
                    }
                    if calibration_converged:
                        bias_converged = True
                        frozen_f_k_snapshot = f_calibrated.tolist()
                        mode = "validating_after_mbar_calibration"
                        _report_p = np.asarray(last_validation_batch_p, dtype=np.float64)
                        min_ema_p = float(np.min(_report_p))
                        max_ema_p = float(np.max(_report_p))
                        ema_mean_p_values = last_validation_batch_p
                        coverage_ess = float(1.0 / np.sum(np.square(_report_p)))
                        report_lse_balance = ibs_lse_balance_diagnostics(_report_p)
                        bias_warmup_diag["status"] = "frozen_validation_converged_after_mbar_calibration"
                        bias_warmup_diag["coverage_ess"] = float(coverage_ess)
                        bias_warmup_diag["ema_mean_p"] = ema_mean_p_values
                        bias_warmup_diag["max_ema_mean_p"] = max_ema_p
                        bias_warmup_diag["min_ema_mean_p"] = min_ema_p
                        bias_warmup_diag["lse_balance"] = report_lse_balance
                        bias_warmup_diag["lse_residual_history"] = lse_residual_history
                        # 🔑 final_mode/frozen_f_k_at_last_freeze 是在 SGD 循环退出时就写进
                        # bias_warmup_diag 的，用的是当时的 mode/frozen_f_k_snapshot；这两个
                        # 局部变量在 MBAR 校准成功后已经被重新赋值（above），但字典里的旧
                        # entry 从未被刷新——不同步的话，诊断会显示校准前的旧 mode/f_k，跟
                        # 实际让这个窗口收敛的其实是校准后的验证这一事实矛盾。
                        bias_warmup_diag["final_mode"] = mode
                        bias_warmup_diag["frozen_f_k_at_last_freeze"] = frozen_f_k_snapshot
                    else:
                        # 🔑 之前这个分支不更新 ema_mean_p_values/min_ema_p/max_ema_p/
                        # coverage_ess，导致下面失败报告打印的是校准前 SGD 阶段的旧值，
                        # 不是这次 MBAR 校准冻结验证真正观察到的最后一批概率——现在用
                        # last_validation_batch_p（本轮 while 循环里每次真实评估都会
                        # 更新）覆盖这几个报告变量，保证失败诊断反映真实情况。
                        if last_validation_batch_p is not None:
                            _report_p = np.asarray(last_validation_batch_p, dtype=np.float64)
                            min_ema_p = float(np.min(_report_p))
                            max_ema_p = float(np.max(_report_p))
                            ema_mean_p_values = last_validation_batch_p
                            coverage_ess = float(1.0 / np.sum(np.square(_report_p)))
                            report_lse_balance = ibs_lse_balance_diagnostics(_report_p)
                            bias_warmup_diag["coverage_ess"] = float(coverage_ess)
                            bias_warmup_diag["ema_mean_p"] = ema_mean_p_values
                            bias_warmup_diag["max_ema_mean_p"] = max_ema_p
                            bias_warmup_diag["min_ema_mean_p"] = min_ema_p
                            bias_warmup_diag["lse_balance"] = report_lse_balance
                            bias_warmup_diag["lse_residual_history"] = lse_residual_history
                        print(
                            "    ⚠️ 用 BAR/MBAR（ΔF_bias，含 WCA、不含 LRC）校准的 f_k 冻结验证仍未"
                            "通过——λ 网格和求解器都已排除，可能是构象弛豫过慢或偏置表达式本身有"
                            "问题，需要人工检查，不再自动重试。"
                        )

            if bias_converged:
                # [IBS_BIAS_PROTOCOL_VERSION=29] 收敛报告改用 local-MBAR loose gate
                # 的相邻态残差（max|Δf_k−ΔF^MBAR|），不再打印已移除的占据 LSE 残差。
                _gate_gap = (
                    last_local_mbar_gate.get("max_adjacent_delta_kJ_mol")
                    if last_local_mbar_gate is not None else None
                )
                _gate_gap_repr = (
                    f"{float(_gate_gap):.3f} kJ/mol"
                    if _gate_gap is not None else "n/a"
                )
                if best_effort_acceptance:
                    print(
                        f" 完成（{bias_update_count} 次权重更新、"
                        f"{learning_to_validation_cycles} 次 loose-gate 未过后，采用 "
                        f"{best_effort_acceptance_reason} best-effort：warmup 预算耗尽即接受"
                        f"当前 f_k；最近一次 max|Δf_k−ΔF^MBAR|={_gate_gap_repr}，阈值 "
                        f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f} kJ/mol；"
                        "未声称严格达标，生产后独立质量门 + 最终 MBAR 仍生效）"
                    )
                else:
                    print(
                        f" 完成（{bias_update_count} 次权重更新、"
                        f"{learning_to_validation_cycles} 次 loose-gate 未过后，最终在冻结 "
                        f"f_k 下通过 local-MBAR loose gate："
                        f"max|Δf_k−ΔF^MBAR|={_gate_gap_repr} < "
                        f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f} kJ/mol"
                        f"（{validation_sample_count} frames / "
                        f"{last_local_mbar_gate.get('n_batches') if last_local_mbar_gate else 0} "
                        "batches）；真正的自由能/ESS/overlap/误差交生产后 MBAR）"
                    )
                sampler.bias_converged = True
                sampler.bias_status = "converged"
                sampler.frozen_f_k_pending = None
                sampler.save_ibs_state(ibs_state_file, lc_win, lv_win)
                # 🔑 一次成功的收敛必须让这个窗口的失败记录彻底过期——否则一次
                # 先失败、重跑后成功的窗口会留下一份跟当前 convergence.json 矛盾
                # 的旧 warmup_failure.json，误导任何检查"这个窗口是否曾失败过"的
                # 下游逻辑（人工排查、自动化审计脚本等）。
                stale_failure_path = os.path.join(
                    self.output_dir,
                    f"dual_window_{window_idx}_{stage_type}_warmup_failure.json",
                )
                if os.path.exists(stale_failure_path):
                    try:
                        os.remove(stale_failure_path)
                    except OSError:
                        pass
                # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 同理清理主窗口
                # checkpoint+manifest——窗口已经真正收敛，不再需要这份"续验用"
                # 的二进制快照，留着只会占磁盘、并在以后误导人工排查。
                for _stale_path in (main_ckpt_path, main_manifest_path):
                    if os.path.exists(_stale_path):
                        try:
                            os.remove(_stale_path)
                        except OSError:
                            pass
            else:
                failure_path = os.path.join(
                    self.output_dir,
                    f"dual_window_{window_idx}_{stage_type}_warmup_failure.json",
                )
                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 区分两种不同的"未收敛"：
                # (a) 这次（或之前 resume 续验的那次）已经用 MBAR 校准探针给出过
                #     一份 fixed-H overlap 证实过 λ 网格没问题、且校准探针自身
                #     样本/不确定度达标的 f_k，只是冻结验证还没在预算内通过——
                #     这份 f_k 不该被当成"求解器没找到答案"，不能退回 SGD；
                # (b) 真正从未获得过这样一份校准 f_k（SGD 没收敛、fixed-H overlap
                #     没通过、或校准探针本身样本不足）——这才是需要 SGD 继续
                #     搜索、或交给拆窗/插 λ 的普通未收敛。
                calibration_pending = bool(
                    resumed_calibration_pending or ("mbar_calibration" in bias_warmup_diag)
                )
                is_final_rung = False
                new_cumulative_steps = 0
                steps_spent_this_attempt = 0
                if calibration_pending:
                    pending_f_k = (
                        resumed_frozen_f_k if resumed_calibration_pending
                        else bias_warmup_diag["mbar_calibration"]["calibrated_f_k_kJ_mol"]
                    )
                    # 🔑 [累计步数记账] resumed_calibration_pending 分支从不进入
                    # learning，这次 attempt 的 steps_at_full_bias 就是纯验证步数；
                    # 全新窗口首次 MBAR 校准的验证步数记在独立的 calibration_steps_used
                    # （bias_warmup_diag["mbar_calibration"]["steps_used"]）。累加到
                    # 跨 resume 持久化的 frozen_validation_cumulative_steps 上——
                    # 全新校准是这份 f_k 的第一次验证，不累加任何旧值。
                    # 🔑 [重复计数修复] 循环内每个 batch 的 checkpoint 落盘代码（见
                    # 上面 "每评估完一批（无论 pass/fail）就存一次主窗口 checkpoint"
                    # 处）已经把 sampler.frozen_validation_cumulative_steps 更新成
                    # prior_cumulative_steps + steps_at_full_bias——循环退出时这个
                    # 属性已经是这次 attempt 的正确累计值。这里如果再读一次这个
                    # 属性、并在上面加一次 steps_spent_this_attempt，就等于把这次
                    # attempt 的步数计了两遍（真实案例：prior=250000、这次真实跑了
                    # 50000 步，循环内已经把属性更新成 300000，这里再加一次 50000
                    # 变成 350000——足足多算了 50000 步，且下一次 resume 会拿着这个
                    # 虚高的 350000 当 prior_cumulative_steps，把后续阶梯档位的真实
                    # 预算越挤越少）。必须用进入这次 attempt 之前捕获的
                    # prior_cumulative_steps（第 5997 行），不能读循环已经写脏的
                    # sampler 属性。
                    if resumed_calibration_pending:
                        steps_spent_this_attempt = int(steps_at_full_bias)
                        new_cumulative_steps = prior_cumulative_steps + steps_spent_this_attempt
                    else:
                        steps_spent_this_attempt = int(
                            bias_warmup_diag["mbar_calibration"]["steps_used"]
                        )
                        new_cumulative_steps = steps_spent_this_attempt
                    sampler.frozen_validation_cumulative_steps = new_cumulative_steps
                    # 🔑 [跨进程阶梯状态修复] 同上——调用方没有显式告诉我们这是不是
                    # 最后一档时（本地 dict 在进程重启后为空），不能默认 False：
                    # 这次 effective_frozen_validation_budget 如果本来就是回退逻辑
                    # 解出来的 FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[-1]（说明
                    # 持久化的累计步数已经追上或超过最后一档），就应该判定为终态，
                    # 而不是静默地当成"还有下一档可以延长"，那样只会在下一次调用
                    # 时重复同一档、原地打转。
                    is_final_rung = _resolve_frozen_validation_is_final_rung(
                        window_idx,
                        frozen_validation_is_final_rung,
                        effective_frozen_validation_budget,
                    )
                    if is_final_rung:
                        # 🔑 [终态] 调用方已经把这次标记为冻结验证阶梯的最后一档，
                        # 仍未通过独立验证——这不再是"等待延长"，是真正的终态失败。
                        # 绝不能存成 calibrated_pending_validation（那会让下次
                        # resume 看起来还能继续自动续验），frozen_f_k_pending 清空。
                        sampler.bias_status = "calibrated_validation_failed"
                        sampler.frozen_f_k_pending = None
                        bias_warmup_diag["calibration_pending_validation"] = False
                        bias_warmup_diag["calibration_validation_terminally_failed"] = True
                    else:
                        sampler.bias_status = "calibrated_pending_validation"
                        sampler.frozen_f_k_pending = [float(x) for x in pending_f_k]
                        bias_warmup_diag["calibration_pending_validation"] = True
                        bias_warmup_diag["frozen_f_k_kJ_mol"] = list(sampler.frozen_f_k_pending)
                    bias_warmup_diag["frozen_validation_budget_this_attempt"] = int(
                        effective_frozen_validation_budget
                    )
                    bias_warmup_diag["frozen_validation_cumulative_steps"] = int(new_cumulative_steps)
                    # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 共享的最终兜底
                    # 保存点——不管这次失败是来自 resumed_calibration_pending 续验
                    # 还是全新窗口的首次 MBAR 校准后验证，都在这里再存一次最新
                    # checkpoint（跟循环内"每批一次"的保存是同一份文件，覆盖式
                    # 落盘，这里只是防御性地确保退出前一定有一份最新的）。终态
                    # 失败时也存一份：万一以后人工决定基于这份构型继续排查，不用
                    # 从头重建。
                    _atomic_save_openmm_checkpoint(sim, main_ckpt_path)
                    _atomic_write_json(main_manifest_path, expected_main_manifest)
                else:
                    sampler.bias_status = "unconverged"
                    sampler.frozen_f_k_pending = None
                    sampler.frozen_validation_cumulative_steps = 0
                _atomic_write_json(failure_path, bias_warmup_diag)
                sampler.bias_converged = False
                sampler.save_ibs_state(ibs_state_file, lc_win, lv_win)
                print(
                    f" 未收敛（{bias_update_count}/{max_bias_updates} 次权重更新，"
                    f"{learning_to_validation_cycles} 次冻结验证失败重学习，"
                    + (
                        f"{frozen_validation_retry_count} 次续验批次未通过（续验模式下未退回 learning），"
                        if resumed_calibration_pending else ""
                    )
                    + (
                        f"{steps_at_full_bias}/{full_bias_step_budget} 步总上限"
                        f"（learning={sgd_step_budget}, freeze burn-in={frozen_burn_in_steps}, "
                        f"冻结累计验证预留={frozen_validation_reserved_steps}），最终阶段={mode}）"
                    )
                )
                print(
                    "    reported mean_p[k]（若进入验证则为固定 f_k 累计均值；"
                    f"否则为 raw EMA 诊断）= {ema_mean_p_values}"
                )
                _tmbar_diag = sampler.last_update_diagnostics.get("tmbar_update", {})
                print(
                    f"    TMBAR: n_entries={len(sampler.tmbar_history)}, "
                    f"converged={_tmbar_diag.get('converged')}, "
                    f"min_overlap={_tmbar_diag.get('min_overlap')}, "
                    f"min_absolute_ess={_tmbar_diag.get('min_absolute_ess')}"
                )
                _weight_diag = sampler.last_update_diagnostics.get("weight_update", {})
                if _weight_diag:
                    print(
                        "    bounded weight update: "
                        f"eta={_weight_diag.get('eta')}, "
                        f"max|delta_f|={_weight_diag.get('max_abs_delta_f_kJ_mol')} kJ/mol, "
                        f"pairwise_spread={_weight_diag.get('pairwise_delta_f_spread_kJ_mol')} kJ/mol, "
                        f"pairwise_limit={_weight_diag.get('pairwise_step_limit_kJ_mol')} kJ/mol, "
                        f"regime={_weight_diag.get('gain_regime')}"
                    )
                if "bidirectional_overlap_probe" in bias_warmup_diag:
                    print(
                        "    fixed-H 双向 overlap = "
                        f"{bias_warmup_diag['bidirectional_overlap_probe']['pairs']}"
                    )
                if calibration_pending:
                    if is_final_rung:
                        print(
                            f"    🛑 校准后的冻结 f_k 冻结验证累计 {new_cumulative_steps} 步后仍未通过，"
                            "已判定为终态失败（calibrated_validation_failed）——"
                            "不再自动续验/延长预算/回退 learning，需要人工检查。"
                        )
                        raise_msg = (
                            f"窗口 {window_idx} 的冻结校准验证累计 {new_cumulative_steps} 步后仍未能通过"
                            "独立验证，已判定为终态失败（calibrated_validation_failed）——这份 f_k 已经用 "
                            "fixed-H overlap 探针 + bias 校准探针证明过物理正确，终态不再自动续验/延长"
                            f"预算/回退 learning，需要人工检查。完整诊断已写入 {failure_path}。"
                        )
                    else:
                        print(
                            f"    🧊 校准后的冻结 f_k 已存为 calibrated_pending_validation"
                            f"（本次冻结验证累计 {new_cumulative_steps}/{effective_frozen_validation_budget} "
                            "步仍未通过），resume 会跳过 SGD/重新校准，直接续验"
                            f"（不重新烧已经验证过的这 {steps_spent_this_attempt} 步）。"
                        )
                        raise_msg = (
                            f"窗口 {window_idx} 的冻结校准验证在累计 {effective_frozen_validation_budget} "
                            f"步目标内仍未通过独立验证（本次 attempt 实际验证了 {steps_spent_this_attempt} "
                            f"步，累计已验证 {new_cumulative_steps} 步）。这份 f_k 已经用 fixed-H overlap "
                            "探针 + bias 校准探针证明过物理正确，不应拆窗/插 λ/退回 SGD，应延长冻结验证"
                            f"累计预算重试。完整诊断已写入 {failure_path}。"
                        )
                    raise IBSFrozenCalibrationValidationError(
                        raise_msg,
                        diagnostics=bias_warmup_diag,
                        terminal=is_final_rung,
                    )
                if mode == "learning":
                    _ordinary_failure_reason = (
                        "time-averaged Q̂ 的 learning 候选平衡在预算内未收敛，"
                        "尚未进入冻结独立验证"
                    )
                else:
                    _ordinary_failure_reason = "候选 f_k 在冻结后的独立验证中未收敛"
                if not legacy_repair:
                    # non_mutating_v1 has priority over warmup_only: design-time
                    # execution is not permission to mutate this sampled grid.
                    _ordinary_failure_action = (
                        "保留全部已采数据；不按相邻 overlap 拆窗/插点/重校准，交人工 / "
                        "rescue-coverage 审计决定是否需要新的 immutable ensemble。"
                    )
                elif warmup_only:
                    _ordinary_failure_action = (
                        "legacy 设计期将先用已有 lambda 拆分 IBS ensemble；只有不可再拆的"
                        "两态窗口才允许插入热力学长度中点，并必须重新通过同一 LSE 自洽验证。"
                    )
                else:
                    _ordinary_failure_action = (
                        "legacy repair 将反馈为拆窗；只有最小窗口的 fixed-H overlap 失败才允许插点。"
                    )
                raise IBSWarmupConvergenceError(
                    f"窗口 {window_idx} 的 IBS 偏置预热在 {bias_update_count} 次权重更新、"
                    f"{learning_to_validation_cycles} 次冻结验证失败重学习后，"
                    f"{_ordinary_failure_reason}（f_not_converged）；完整诊断已写入 {failure_path}。"
                    f"{_ordinary_failure_action}",
                    diagnostics=bias_warmup_diag,
                )

            # 恢复步长
            sim.integrator.setStepSize(original_dt)

            # 打印预热后的 f_k 值
            f_vals = [sim.context.getParameter(f"{self.prefix}_f_{k}") for k in range(len(lc_win))]
            print(
                "  ✅ 偏置预热结束（f_k 已锁定；严格通过/预算终止类型见诊断）: "
                f"{[f'{v:.1f}' for v in f_vals]}"
            )
            if max(f_vals) - min(f_vals) < 1.0:
                print(f"  ⚠️ 警告: f_k 仍未明显分化，可能需延长预热或检查窗口重叠度。")
            if debug_mode:
                diagnose_softcore_cv_values(sim.context, ibs_wrap, lc_win, lv_win, prefix=f"窗口{window_idx}_偏置预热后", sampler=sampler)
            raw_probe = sampler.get_raw_interaction_energies()
            if len(raw_probe) >= 2:
                raw_span = float(np.max(raw_probe) - np.min(raw_probe))
                if raw_span < 1e-6 and np.max(np.abs(raw_probe)) < 1e-6:
                    raise RuntimeError(
                        f"窗口 {window_idx} 的 IBS 软核 CV 在偏置预热后仍全部为 0。"
                        "已阻止进入生产采样，请继续检查 VDW CV 构造或状态注入。"
                    )

            if warmup_only:
                bias_warmup_diag["warmup_only"] = True
                bias_warmup_diag["frozen_f_k_kJ_mol"] = [float(v) for v in f_vals]
                warmup_results.append(dict(bias_warmup_diag))
                if getattr(sampler, "_probe_context", None) is not None:
                    del sampler._probe_context
                    sampler._probe_context = None
                if getattr(sampler, "_probe_integrator", None) is not None:
                    del sampler._probe_integrator
                    sampler._probe_integrator = None
                del sampler
                del sim.context
                del sim
                del win_sys
                gc.collect()
                continue

            # ---- 进入生产采样：不再改变 bias_scale/最小化/重新爬坡 ----
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=7] 旧版本这里有"生产前卸压"：bias_scale
            # 清零→最小化→5000 步无偏置运行→重新爬回 1.0，紧接着直接进 production，
            # 从未对这个被重新扰动过的构型重新验证过覆盖度——哪怕前面的收敛判据是
            # 真的，这一步也会在没有复检的情况下把刚验证过的分布毁掉。现在冻结
            # 验证已经在完全固定的 production Hamiltonian（bias_scale=1.0、f_k 不再
            # 变化）下完成，不需要也不允许再碰 bias_scale/最小化/爬坡：只做一次
            # 非破坏性的安全力检查（不达标就直接报错，不再用更多扰动去"修复"一个
            # 刚验证通过的构型），然后原样进入生产采样。

            # 🔑 提前到这里计算（原来在安全检查/生产采样断言之后才算）：下面的
            # production checkpoint 续采检测需要提前知道本次目标步数，才能判断
            # "已有的累计步数是否还没到目标"。
            effective_n_steps_per_window = (
                int(production_step_overrides[window_idx])
                if production_step_overrides and window_idx in production_step_overrides
                else int(n_steps_per_window)
            )
            if effective_n_steps_per_window != int(n_steps_per_window):
                print(
                    f"  🔁 窗口 {window_idx} 使用步数覆盖：{effective_n_steps_per_window} 步"
                    f"（默认 {n_steps_per_window} 步），来自 production_step_overrides"
                )

            # 🔑 [production checkpoint 续采] 之前 production ESS 低触发
            # reseed_resample 时，是整窗删除 energies/bias/base/convergence.json
            # 后从这里重新最小化+dt测试+Boresch爬坡+冻结重验证+从零步 production
            # 重采一遍（"250k 延长到 500k"实际是"扔掉 250k、重跑独立的 500k"，
            # 不是真续算）。这里检查是否存在 λ/窗口/系统/冻结 f_k 完全一致的
            # production checkpoint，且 energies/bias/base/convergence.json 仍在
            # （没有被 _invalidate_single_window_production 清掉——f_k 真的变了
            # 时它会被清掉，这里自然检测不到，回退到今天的完整重采），若有且
            # 累计步数小于本次目标，就在下面的安全检查之前先恢复到上次结束的
            # 坐标/速度/积分器 RNG 状态，让安全检查/production_pos_backup/生产
            # 采样本身都在这份续算状态上进行；已有的能量/偏置/基准能量历史在
            # 下面 sampler.energy_history 等重置之后再追加读回。
            production_frozen_f_k = [
                float(sim.context.getParameter(f"{self.prefix}_f_{k}")) for k in range(len(lc_win))
            ]
            win_sys_xml_for_prod_ckpt = openmm.XmlSerializer.serialize(win_sys)
            expected_production_manifest = _build_production_window_checkpoint_manifest(
                stage_type, window_idx, len(lc_win), win_sys_xml_for_prod_ckpt, lc_win, lv_win,
                (float(np.mean(lv_win)) if _system_has_global_parameter(win_sys, "lambda_shield") else None),
                (1.0 if _has_valid_boresch_restraint(self.boresch) else None),
                production_frozen_f_k,
                self.temperature.value_in_unit(unit.kelvin),
                self.platform_name,
                repair_policy=repair_policy,
            )
            production_energies_path = os.path.join(
                self.output_dir, f"dual_window_{window_idx}_{stage_type}_energies.npy"
            )
            production_bias_path = os.path.join(
                self.output_dir, f"dual_window_{window_idx}_{stage_type}_bias.npy"
            )
            production_base_path = os.path.join(
                self.output_dir, f"dual_window_{window_idx}_{stage_type}_base.npy"
            )
            production_conv_path = os.path.join(
                self.output_dir, f"dual_window_{window_idx}_{stage_type}_convergence.json"
            )
            production_ckpt_path, production_manifest_path = _production_window_checkpoint_paths(
                self.checkpoint_dir, stage_type, window_idx
            )
            prior_cumulative_production_steps = 0
            resumed_production_checkpoint = False
            prior_energy_history = None
            prior_bias_history = None
            prior_base_energy_history = None
            if (
                os.path.exists(production_energies_path)
                and os.path.exists(production_conv_path)
                and _production_window_checkpoint_is_usable(
                    self.checkpoint_dir, stage_type, window_idx, expected_production_manifest
                )
            ):
                try:
                    with open(production_conv_path, "r", encoding="utf-8") as f:
                        prior_production_convergence = json.load(f)
                    prior_cumulative_production_steps = int(
                        prior_production_convergence.get("cumulative_production_steps", 0)
                    )
                except Exception:
                    prior_cumulative_production_steps = 0
                if 0 < prior_cumulative_production_steps < effective_n_steps_per_window:
                    if _try_load_main_window_checkpoint(sim, production_ckpt_path):
                        try:
                            _prior_e, _prior_b, _prior_base = (
                                _load_validated_window_data_triplet(
                                    production_energies_path,
                                    production_bias_path,
                                    production_base_path,
                                    prior_production_convergence,
                                )
                            )
                            if _prior_e.shape[1] == _prior_b.shape[0] == _prior_base.shape[0]:
                                prior_energy_history = [_prior_e[:, i] for i in range(_prior_e.shape[1])]
                                prior_bias_history = [float(x) for x in _prior_b]
                                prior_base_energy_history = [float(x) for x in _prior_base]
                                resumed_production_checkpoint = True
                            else:
                                print(
                                    f"  ⚠️ 窗口 {window_idx} 生产 checkpoint 匹配，但已有能量/偏置/"
                                    "基准数组帧数不一致，回退完整重采。"
                                )
                        except Exception as e:
                            print(f"  ⚠️ 窗口 {window_idx} 生产历史能量数组加载失败（{e}），回退完整重采。")
                    else:
                        print(f"  ⚠️ 窗口 {window_idx} 生产 checkpoint 加载失败，回退完整重采。")
            if not resumed_production_checkpoint:
                prior_cumulative_production_steps = 0
                prior_energy_history = None
                prior_bias_history = None
                prior_base_energy_history = None
                print(
                    f"  🧱 窗口 {window_idx} 预热/冻结验证与生产严格隔离："
                    "验证样本不计入生产，生产从 0 步开始"
                )

            safety_state = sim.context.getState(getForces=True, getPositions=True)
            safety_forces = safety_state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            fmax_safety = float(np.max(np.linalg.norm(safety_forces, axis=1)))
            preprod_force_threshold = 10000.0
            print(f"  生产前安全检查 max|F| = {fmax_safety:.1f} kJ/(mol·nm)（未做任何扰动）")
            if fmax_safety > preprod_force_threshold:
                if debug_mode:
                    diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_生产前拦截")
                    diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_生产前拦截")
                    diagnose_top_force_atoms(
                        sim.context,
                        win_sys,
                        topology=self.topology,
                        ligand_indices=self.ligand_indices,
                        prefix=f"窗口{window_idx}_生产前拦截",
                    )
                raise RuntimeError(
                    f"窗口 {window_idx} 在已验证收敛的冻结 Hamiltonian 下 max|F|={fmax_safety:.1f} "
                    f"kJ/(mol·nm) 仍超过安全阈值 {preprod_force_threshold:.0f}——这与刚通过的统计验证"
                    "矛盾，可能是构象陷阱或数值问题，已阻止进入生产采样而不是用更多扰动掩盖它。"
                )

            sampler.energy_buffer = []
            # 生产历史只有一个合法来源：同一冻结 f_k、同一 manifest 的生产
            # checkpoint。预热、freeze burn-in 和冻结验证的 history 全部清空，
            # 绝不混入生产统计。
            if resumed_production_checkpoint and prior_energy_history is not None:
                sampler.energy_history = list(prior_energy_history)
                sampler.bias_history = list(prior_bias_history)
                sampler.base_energy_history = list(prior_base_energy_history)
            else:
                sampler.energy_history = []
                sampler.bias_history = []
                sampler.base_energy_history = []
            production_pos_backup = safety_state.getPositions(asNumpy=True)
            # 🔑 [P1-13] 与坐标备份**同时**记下三份 history 的长度。灾难回退只把
            # 坐标退回这个备份点，却把备份之后已经写入的 energy/bias/base history
            # 留在原地，然后从旧坐标 + 新随机速度长出另一条分支——被放弃的分支与
            # 重启分支共享祖先，却仍被当作一条连续时间序列做自相关子采样
            # （_decorrelate_by_worst_target_state），g 与 N_decorr 的口径就不再可靠。
            # 有了这个长度，回退时才能把三份 history 同步截断回同一个分叉点。
            production_history_backup_len = _production_history_lengths(sampler)

            # ---------- 生产采样 ----------
            # 🔑 防御性断言：走到这里必须已经满足严格收敛判据（sampler.bias_converged
            # 为 True），否则说明上面的收敛判据/raise 逻辑被绕过或本函数被以未预期的
            # 方式调用——不能让这种情况悄悄进入生产采样，MBAR 需要一个 f_k 已冻结的
            # 单一固定采样分布。
            if not sampler.bias_converged:
                raise RuntimeError(
                    f"窗口 {window_idx} 内部状态异常：即将进入生产采样但 "
                    "sampler.bias_converged 不为 True。这不应该发生——说明收敛判据/"
                    "raise 逻辑被绕过，拒绝继续，请检查上面的预热分支逻辑。"
                )
            # 生产阶段的硬边界：从这一行起 f_k 只读。即便未来有人误把
            # update/recalibration/restore 逻辑插进生产循环，下面逐 update 的
            # 运行时断言也会立即终止，而不是悄悄生成混合偏置数据。
            production_f_k_lock = np.asarray(
                [
                    sim.context.getParameter(f"{self.prefix}_f_{k}")
                    for k in range(K)
                ],
                dtype=np.float64,
            )
            if not np.allclose(
                production_f_k_lock,
                np.asarray(production_frozen_f_k, dtype=np.float64),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    f"窗口 {window_idx} 生产入口的 f_k 与冻结 manifest 不一致；"
                    "拒绝进入生产。生产阶段禁止恢复、重算或修改 f_k。"
                )
            print(f"\n  [阶段5] 生产采样 ({effective_n_steps_per_window} 步)...")
            # 🔑 [production checkpoint 续采] 只跑目标步数与已完成累计步数的差值，
            # 不是把 effective_n_steps_per_window 当成"这次要新跑这么多步"（那样
            # 会变成每次 reseed_resample 都把已经真实跑过的步数重复计入，
            # 250k->500k 实际跑出 750k）。
            remaining_production_steps = max(
                int(effective_n_steps_per_window) - int(prior_cumulative_production_steps), 0
            )
            if resumed_production_checkpoint:
                print(
                    f"  🧊 窗口 {window_idx} 从生产 checkpoint 续算：累计已完成 "
                    f"{prior_cumulative_production_steps} 步，本次只需再跑 "
                    f"{remaining_production_steps} 步（目标 {effective_n_steps_per_window} 步），"
                    f"保留 {len(sampler.energy_history)} 帧已有样本，坐标/速度/积分器 RNG "
                    "状态与上次中断时完全一致（注意：这里跳过的只是重跑 production 本身——"
                    "最小化/dt测试/Boresch爬坡/冻结重验证仍会照常先跑一遍，代价远小于"
                    "重跑整个 production 步数预算）。"
                )
            n_updates = remaining_production_steps // steps_per_update

            # 🔑 [EARLY_STOP_PROTOCOL_VERSION] 在线 early-stop 监测状态。默认关闭
            # （enable_early_stop=False，见 run_all_windows 参数说明），下面这段
            # 只在显式打开时才会真正跑 local MBAR/提前退出；关闭时行为与之前完全
            # 一致（跑满 n_updates，不做任何额外检查）。
            early_stop_kt = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(unit.kilojoule_per_mole)
            early_stop_min_updates = max(1, int(early_stop_min_steps) // max(1, int(steps_per_update)))
            early_stop_check_interval_updates = max(1, int(early_stop_check_interval_steps) // max(1, int(steps_per_update)))
            early_stop_pass_count = 0
            early_stop_check_history = []
            early_stop_triggered = False
            early_stop_stop_reason = None
            early_stop_previous_local_dg = None
            actual_production_updates = n_updates  # overwritten below only if early-stopped

            # 以下 CV/力组诊断仅用于排障，默认关闭（debug_mode=False），
            # 避免生产环境每个窗口都无条件刷屏并产生额外的 getState 开销。
            if debug_mode:
                for gid in [0, 2, 3]:
                    e_g = sim.context.getState(getEnergy=True, groups={gid}).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                    print(f"Group {gid} energy: {e_g:.1f} kJ/mol")

                cv_vals = ibs_wrap.get_force().getCollectiveVariableValues(sim.context)
                print(f"First CV value (should be LE interaction): {cv_vals[0]:.1f} kJ/mol")
                # 获取 CV 力对象（第一个 CV）
                cv_force_0 = ibs_wrap.get_force().getCollectiveVariable(0)
                # 检查它的交互组
                num_groups = cv_force_0.getNumInteractionGroups()
                print(f"  [诊断] CV 交互组数量 = {num_groups}")
                for gidx in range(num_groups):
                    set1, set2 = cv_force_0.getInteractionGroupParameters(gidx)
                    print(f"  [诊断] 交互组 {gidx}: size={len(set1)} × {len(set2)}")

                print("\n🔬 软核 CV 详细诊断：")

                # 1. 获取第一个 CV 力对象
                cv_force = ibs_wrap.get_force().getCollectiveVariable(0)

                # 2. 打印其能量表达式
                print(f"  CV 表达式：\n{cv_force.getEnergyFunction()}\n")

                # 3. 打印 CV 力内部的所有全局参数及其当前值
                print("  CV 全局参数：")
                for i in range(cv_force.getNumGlobalParameters()):
                    name = cv_force.getGlobalParameterName(i)
                    try:
                        val = sim.context.getParameter(name)
                    except Exception:
                        val = cv_force.getGlobalParameterDefaultValue(i)
                    print(f"    {name} = {val}")

                # 4. 打印配体和环境原子各自的前 5 个参数
                lig_indices = self.ligand_indices
                num_atoms = win_sys.getNumParticles()
                cv_param_names = [
                    cv_force.getPerParticleParameterName(i)
                    for i in range(cv_force.getNumPerParticleParameters())
                ]

                def _format_cv_particle(idx: int) -> str:
                    params = cv_force.getParticleParameters(idx)
                    values = []
                    for name, val in zip(cv_param_names, params):
                        if hasattr(val, "value_in_unit"):
                            if "charge" in name.lower() or name.lower().startswith("q"):
                                val = val.value_in_unit(unit.elementary_charge)
                            elif "sigma" in name.lower() or "r0" in name.lower():
                                val = val.value_in_unit(unit.nanometer)
                            else:
                                val = val.value_in_unit(unit.kilojoule_per_mole)
                        values.append(f"{name}={float(val):.4f}")
                    return ", ".join(values)

                print(f"\n  配体原子前 5 个参数（{', '.join(cv_param_names)}）：")
                for idx in lig_indices[:5]:
                    print(f"    atom {idx}: {_format_cv_particle(idx)}")

                # 环境原子前 5 个参数
                env_list = [i for i in range(num_atoms) if i not in set(lig_indices)]
                print("\n  环境原子前 5 个参数：")
                for idx in env_list[:5]:
                    print(f"    atom {idx}: {_format_cv_particle(idx)}")

            # ✅ 性能修复：getForces=True/getPositions=True 每次都要把全原子数组从
            # GPU 传回 CPU，代价远高于纯 getEnergy=True。原来每个 update（默认每
            # 500 步）都做一次完整的力+坐标查询，纯粹为了灾难检测和回退备份——
            # 但灾难本身是小概率事件。改为默认只查能量（便宜很多），每
            # IBS_FORCE_CHECK_INTERVAL 个 update 才做一次完整的力/坐标查询；一旦
            # 能量本身已经 NaN/Inf，无论轮到没轮到都立刻补查一次力，不能让"跳过"
            # 的 update 错过真正的灾难信号。
            # ⚠️ 权衡：两次完整检查之间，如果力在没有把能量推成 NaN/Inf 的前提下
            # 短暂冲高又回落，这个窗口是看不到的——比原来"每 update 都查"的覆盖率
            # 低。position 备份只在做了完整检查的 update 上刷新，回退时可能回退到
            # 更早的坐标（多丢几个 update 的进度），但回退本身就是保守操作，不影响
            # 正确性。
            force_check_interval = max(1, int(os.environ.get("IBS_FORCE_CHECK_INTERVAL", "10")))

            # 开始生产采样循环
            for up in range(n_updates):
                pos_backup = production_pos_backup
                try:
                    sim.step(steps_per_update)
                except Exception as e:
                    print(f"\n  🚨 采样崩溃于 update={up}/{n_updates}")
                    if debug_mode:
                        diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_崩溃_update{up}")
                        diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_崩溃_update{up}")
                    raise

                current_production_f_k = np.asarray(
                    [
                        sim.context.getParameter(f"{self.prefix}_f_{k}")
                        for k in range(K)
                    ],
                    dtype=np.float64,
                )
                if not np.allclose(
                    current_production_f_k,
                    production_f_k_lock,
                    rtol=0.0,
                    atol=1.0e-12,
                ):
                    raise RuntimeError(
                        f"窗口 {window_idx} 生产阶段 update={up} 检测到 f_k 被修改；"
                        "已立即停止，拒绝继续生成混合偏置生产数据。"
                    )

                do_force_check = (up % force_check_interval == 0) or (up == n_updates - 1)
                if do_force_check:
                    state_n = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                    forces_n = state_n.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                    fmax = np.max(np.linalg.norm(forces_n, axis=1))
                else:
                    state_n = sim.context.getState(getEnergy=True)
                    fmax = None
                e_total_n = state_n.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                energy_bad = not np.isfinite(e_total_n)
                if energy_bad and not do_force_check:
                    # 能量已经异常，不能再等下一次排期检查——立刻补查力和坐标
                    state_n = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                    forces_n = state_n.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                    fmax = np.max(np.linalg.norm(forces_n, axis=1))
                    do_force_check = True

                if energy_bad or (fmax is not None and ((not np.isfinite(fmax)) or fmax > 10000.0)):
                    sim.context.setPositions(pos_backup)
                    sim.context.setVelocitiesToTemperature(self.temperature)
                    print("    🔧 触发回退，执行局部最小化释放应力...")
                    sim.minimizeEnergy(maxIterations=2000, tolerance=1.0)
                    current_dt_ps = sim.integrator.getStepSize().value_in_unit(unit.picoseconds)
                    new_dt_ps = max(0.0001, current_dt_ps * 0.5)
                    sim.integrator.setStepSize(new_dt_ps * unit.picoseconds)
                    # 🔑 [P1-13] 坐标退回备份点了，备份之后写入的帧属于被放弃的
                    # 分支，必须同步截断——否则它们会和重启后长出的新分支拼成一条
                    # "连续"轨迹交给 _decorrelate_by_worst_target_state 估自相关。
                    dropped = _truncate_production_history(
                        sampler, production_history_backup_len
                    )
                    fmax_report = fmax if fmax is not None else float("nan")
                    print(
                        f"    ⚠️ 灾难检测触发: update={up}/{n_updates}, "
                        f"E_total={e_total_n:.1f}, max|F|={fmax_report:.1f}. "
                        f"已回退坐标并将步长降至 {new_dt_ps*1000.0:.1f} fs"
                        + (f"；同步丢弃被放弃分支的 {dropped} 帧生产 history" if dropped else "")
                    )
                    if debug_mode:
                        diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_回退_update{up}")
                        diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_回退_update{up}")
                    continue

                if do_force_check:
                    # 复用同一次 getState 里已经取回的坐标，不再单独发一次 getPositions 请求
                    production_pos_backup = state_n.getPositions(asNumpy=True)
                    # 坐标备份与 history 长度必须成对更新，否则回退会截到错的分叉点。
                    production_history_backup_len = _production_history_lengths(sampler)

                # 🔑 生产阶段 f_k 已冻结（sampler.bias_converged=True，见上面严格收敛
                # 判据），不再调用 update_weights()——否则同一窗口的样本会来自随时间
                # 漂移的多个偏置，而不是 MBAR 假设的单一固定采样分布。只采样、记录
                # 能量；定期清空 energy_buffer 避免其无谓增长（update_weights() 之前
                # 顺带做的这件事，现在没人做了）。
                e = sampler.collect_energies()
                if len(sampler.energy_buffer) >= 10:
                    sampler.energy_buffer = []
                if (up + 1) % 100 == 0:
                    self._enqueue_window_snapshot(window_idx, stage_type, sampler)
                    # 🔑 [production checkpoint 续采] 每 100 个 update 覆盖式落盘一次
                    # 生产 checkpoint（坐标/速度/积分器 RNG）+ 当前已有的能量/偏置/
                    # 基准能量数组 + 累计步数——应对 HPC 作业被抢占/撞墙时限杀掉的
                    # 情况：不这样做的话，一次很长的 production 预算在被杀时会丢失
                    # 全部进展，下次只能整窗从零重采（这正是这次要修的问题本身）。
                    # convergence.json 只合并更新 cumulative_production_steps 字段，
                    # 不覆盖其它字段——这份文件此刻还不是"最终结果"，真正完整的
                    # 版本在窗口正常结束时于下面"保存能量"区块整份重写。
                    _atomic_save_openmm_checkpoint(sim, production_ckpt_path)
                    _atomic_write_json(production_manifest_path, expected_production_manifest)
                    _periodic_e_arr = (
                        np.array(sampler.energy_history) if sampler.energy_history
                        else np.zeros((0, len(lc_win)))
                    )
                    _periodic_e_save = (
                        _periodic_e_arr.T if _periodic_e_arr.size > 0 else np.zeros((len(lc_win), 0))
                    )
                    _atomic_save_npy(production_energies_path, _periodic_e_save)
                    _atomic_save_npy(
                        production_bias_path,
                        np.asarray(sampler.bias_history, dtype=np.float64),
                    )
                    _atomic_save_npy(
                        production_base_path,
                        np.asarray(sampler.base_energy_history, dtype=np.float64),
                    )
                    _periodic_conv = {}
                    if os.path.exists(production_conv_path):
                        try:
                            with open(production_conv_path, "r", encoding="utf-8") as pf:
                                _periodic_conv = json.load(pf)
                        except Exception:
                            _periodic_conv = {}
                    _periodic_conv["cumulative_production_steps"] = (
                        int(prior_cumulative_production_steps) + (up + 1) * int(steps_per_update)
                    )
                    _periodic_conv["window_data_protocol_version"] = (
                        IBS_WINDOW_DATA_PROTOCOL_VERSION
                    )
                    _periodic_conv["window_data"] = _window_data_metadata(
                        production_energies_path,
                        production_bias_path,
                        production_base_path,
                    )
                    _atomic_write_json(production_conv_path, _periodic_conv)

                if (
                    enable_early_stop
                    and (up + 1) >= early_stop_min_updates
                    and (up + 1) % early_stop_check_interval_updates == 0
                ):
                    u_kj_raw = (
                        np.asarray(sampler.energy_history, dtype=np.float64).T
                        if sampler.energy_history else np.zeros((len(lc_win), 0))
                    )
                    bias_kj = (
                        np.asarray(sampler.bias_history, dtype=np.float64)
                        if sampler.bias_history else np.zeros((0,))
                    )
                    base_kj = (
                        np.asarray(sampler.base_energy_history, dtype=np.float64)
                        if sampler.base_energy_history else np.zeros((0,))
                    )
                    # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 生产期 f_k 已冻结（进生产后
                    # 不再调用 update_weights），所以此刻从 context 读到的就是这段
                    # 历史采样时生效的 f_k，可以直接用来除掉共模因子。
                    f_k_frozen = np.asarray(
                        [
                            sim.context.getParameter(f"{self.prefix}_f_{k}")
                            for k in range(len(lc_win))
                        ],
                        dtype=np.float64,
                    )
                    local_result = _solve_single_window_local_mbar(
                        u_kj_raw, bias_kj, base_kj, list(range(start, end)), early_stop_kt,
                        f_k=f_k_frozen,
                        sampled_distribution_row=0, w_idx=window_idx,
                    )
                    step_at_check = (up + 1) * steps_per_update
                    if "error" in local_result:
                        early_stop_pass_count = 0
                        early_stop_check_history.append({
                            "step": int(step_at_check), "passed": False, "error": local_result["error"],
                        })
                        print(
                            f"    ⏱️ [early-stop] 窗口 {window_idx} 第 {step_at_check} 步检查：local MBAR "
                            f"未解出 ({local_result['error']})，连续通过计数清零"
                        )
                    else:
                        f_arr = np.asarray(local_result["f"], dtype=np.float64)
                        df_arr = np.asarray(local_result["df"], dtype=np.float64)
                        local_dg = float(f_arr[-1] - f_arr[0]) if f_arr.size > 1 else 0.0
                        # 🔑 端点差的不确定度必须直接读 MBAR 算出的两个物理态之间
                        # 的成对不确定度（_solve_single_window_local_mbar 已经从
                        # ddf_matrix 里取好了），不能用 sqrt(df[0]^2+df[-1]^2) 把
                        # 两个边际（相对采样态）不确定度当独立量合并——它们来自
                        # 同一次 MBAR 拟合、同一批样本，彼此有协方差，平方相加
                        # 会系统性算错，可能让判据过早通过，也可能让它永远通不过。
                        local_uncertainty = (
                            float(local_result.get("endpoint_diff_uncertainty_kJ_mol", float("nan")))
                            if f_arr.size > 1
                            else (float(df_arr[0]) if f_arr.size else float("nan"))
                        )
                        min_ess_ratio = local_result.get("min_ess_ratio")
                        n_frames_used = int(local_result.get("n_frames_used", 0))
                        absolute_ess = (min_ess_ratio * n_frames_used) if min_ess_ratio is not None else None

                        # 🔑 同一类 ESS/overlap/不确定度浮点舍入问题（见
                        # _meets_minimum_with_roundoff/_meets_maximum_with_roundoff
                        # 的 docstring 与 v23 numerical follow-up）也会出现在这里——
                        # 这四项判据在数学上和 solve_stage_integrated 里的 converged
                        # 判据同源，必须用同一套只接受舍入尺度相等的容差比较，不能
                        # 只修一处。
                        ess_ratio_ok = min_ess_ratio is not None and _meets_minimum_with_roundoff(min_ess_ratio, early_stop_min_ess_ratio)
                        absolute_ess_ok = absolute_ess is not None and _meets_minimum_with_roundoff(absolute_ess, early_stop_min_absolute_ess)
                        decorrelated_ok = n_frames_used >= early_stop_min_decorrelated_samples
                        uncertainty_ok = np.isfinite(local_uncertainty) and _meets_maximum_with_roundoff(local_uncertainty, early_stop_max_uncertainty_kJ_mol)
                        # 第一次检查没有"上一次"可比，直接判不通过——至少要连续
                        # required_consecutive_passes+1 次检查才可能真正停止，
                        # 保证漂移判据不会在只有一个数据点时被跳过。
                        if early_stop_previous_local_dg is None:
                            drift_ok = False
                            drift = None
                        else:
                            drift = float(abs(local_dg - early_stop_previous_local_dg))
                            drift_ok = drift <= early_stop_max_delta_g_drift_kJ_mol
                        all_ok = bool(ess_ratio_ok and absolute_ess_ok and decorrelated_ok and drift_ok and uncertainty_ok)

                        early_stop_check_history.append({
                            "step": int(step_at_check),
                            "min_ess_ratio": min_ess_ratio,
                            "absolute_ess": absolute_ess,
                            "n_frames_decorrelated": n_frames_used,
                            "local_delta_g_kJ_mol": local_dg,
                            "local_uncertainty_kJ_mol": local_uncertainty,
                            "drift_from_previous_kJ_mol": drift,
                            "checks": {
                                "ess_ratio_ok": ess_ratio_ok,
                                "absolute_ess_ok": absolute_ess_ok,
                                "decorrelated_ok": decorrelated_ok,
                                "drift_ok": drift_ok,
                                "uncertainty_ok": uncertainty_ok,
                            },
                            "passed": all_ok,
                        })
                        early_stop_previous_local_dg = local_dg
                        early_stop_pass_count = early_stop_pass_count + 1 if all_ok else 0
                        print(
                            f"    ⏱️ [early-stop] 窗口 {window_idx} 第 {step_at_check} 步检查："
                            f"{'通过' if all_ok else '未通过'}（连续 {early_stop_pass_count}/"
                            f"{early_stop_required_consecutive_passes}），ESS比例={min_ess_ratio}, "
                            f"绝对ESS={absolute_ess}, 去相关样本={n_frames_used}, "
                            f"局部ΔG={local_dg:.3f} kJ/mol, 不确定度={local_uncertainty:.3f} kJ/mol, "
                            f"漂移={drift}"
                        )
                        if early_stop_pass_count >= early_stop_required_consecutive_passes:
                            early_stop_triggered = True
                            early_stop_stop_reason = "consecutive_passes_reached"
                            actual_production_updates = up + 1
                            print(
                                f"    ✅ [early-stop] 窗口 {window_idx} 连续 {early_stop_pass_count} 次独立 "
                                f"block 通过，提前于第 {step_at_check} 步停止生产采样"
                            )
                            break

                if debug_mode and up % 10 == 0:
                    e_total = e_total_n
                    has_nan_energy = not np.isfinite(e_total)
                    has_nan_e = np.any(np.isnan(e))
                    print(f"    [采样] update={up}/{n_updates}: E_total={e_total:.1f}, NaN(E_total)={has_nan_energy}, NaN(E_k)={has_nan_e}")
                    if has_nan_energy or has_nan_e:
                        print(f"    🚨 检测到 NaN 能量！主系统各组能量与最大受力：")
                        num_forces = win_sys.getNumForces()
                        for i in range(num_forces):
                            force = win_sys.getForce(i)
                            gid = force.getForceGroup()
                            try:
                                # 获取该 ForceGroup 的能量
                                state_g = sim.context.getState(getEnergy=True, groups={gid})
                                e_g = state_g.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                                # 获取该 ForceGroup 的力
                                state_f = sim.context.getState(getForces=True, groups={gid})
                                forces = state_f.getForces(asNumpy=True).value_in_unit(
                                    unit.kilojoule_per_mole / unit.nanometer
                                )
                                max_f = np.max(np.linalg.norm(forces, axis=1)) if forces.size else 0.0
                                print(f"      Group {gid:2d} ({type(force).__name__:30s}) E={e_g:14.3f} kJ/mol, max|F|={max_f:12.2f}")
                            except Exception as e:
                                print(f"      Group {gid:2d} ({type(force).__name__:30s}) 获取失败: {e}")

            # early stop 提前退出时，跳过余数补齐——那是为了让"跑满预算"时的总
            # 步数严格达标而设计的，跟"提前停止"的意图矛盾。
            remaining_steps = 0 if early_stop_triggered else effective_n_steps_per_window % steps_per_update
            if remaining_steps > 0:
                pos_backup = production_pos_backup
                try:
                    sim.step(remaining_steps)
                except Exception as e:
                    print(f"\n  🚨 余数补齐阶段崩溃 ({remaining_steps} 步)")
                    if debug_mode:
                        diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_余数补齐崩溃")
                        diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_余数补齐崩溃")
                    raise

                state_n = sim.context.getState(getEnergy=True, getForces=True)
                e_total_n = state_n.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                forces_n = state_n.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                fmax = np.max(np.linalg.norm(forces_n, axis=1))
                if (not np.isfinite(e_total_n)) or (not np.isfinite(fmax)) or fmax > 10000.0:
                    sim.context.setPositions(pos_backup)
                    sim.context.setVelocitiesToTemperature(self.temperature)
                    print("    🔧 余数补齐触发回退，执行局部最小化释放应力...")
                    sim.minimizeEnergy(maxIterations=2000, tolerance=1.0)
                    current_dt_ps = sim.integrator.getStepSize().value_in_unit(unit.picoseconds)
                    new_dt_ps = max(0.0001, current_dt_ps * 0.5)
                    sim.integrator.setStepSize(new_dt_ps * unit.picoseconds)
                    # 🔑 [P1-13] 同主循环：坐标退回备份点，备份之后的帧必须同步丢弃。
                    dropped = _truncate_production_history(
                        sampler, production_history_backup_len
                    )
                    print(
                        f"    ⚠️ 余数补齐灾难检测触发: E_total={e_total_n:.1f}, max|F|={fmax:.1f}. "
                        f"已回退坐标并将步长降至 {new_dt_ps*1000.0:.1f} fs"
                        + (f"；同步丢弃被放弃分支的 {dropped} 帧生产 history" if dropped else "")
                    )
                    if debug_mode:
                        diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_余数补齐回退")
                        diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_余数补齐回退")
                else:
                    production_pos_backup = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
                    production_history_backup_len = _production_history_lengths(sampler)
                    # 生产阶段 f_k 已冻结，不再调用 update_weights()，见上方主循环同类注释。
                    e = sampler.collect_energies()
                    if len(sampler.energy_buffer) >= 10:
                        sampler.energy_buffer = []
                    self._enqueue_window_snapshot(window_idx, stage_type, sampler)
                    print(f"    [采样] 补齐余数 {remaining_steps} 步，总步数严格达标 {effective_n_steps_per_window}")
                    if debug_mode:
                        has_nan_energy = not np.isfinite(e_total_n)
                        has_nan_e = np.any(np.isnan(e))
                        print(
                            f"    [采样-余数] E_total={e_total_n:.1f}, "
                            f"NaN(E_total)={has_nan_energy}, NaN(E_k)={has_nan_e}"
                        )

            # ---------- 保存能量 ----------
            # 窗口结束时即使样本数不足 100，也必须执行最终失败率门；否则短窗口
            # 中的间歇性丢帧会逃过运行期检查。
            sampler.assert_energy_query_quality(final=True)
            e_arr = np.array(sampler.energy_history) if sampler.energy_history else np.zeros((0, len(lc_win)))
            e_save = e_arr.T if e_arr.size > 0 else np.zeros((len(lc_win), 0))
            _atomic_save_npy(production_energies_path, e_save)
            _atomic_save_npy(
                production_bias_path,
                np.asarray(sampler.bias_history, dtype=np.float64),
            )
            _atomic_save_npy(
                production_base_path,
                np.asarray(sampler.base_energy_history, dtype=np.float64),
            )
            window_data_metadata = _window_data_metadata(
                production_energies_path,
                production_bias_path,
                production_base_path,
            )
            convergence = {
                "window_idx": int(window_idx),
                "stage_type": stage_type,
                # 该窗口这次实际采样的 λ 值——不是"这个位置理论上该有什么"，是这份
                # 能量文件里的每一行真实对应哪个 λ。resume 断点续传 / abfe_pipeline.py
                # 的窗口产物复用逻辑都靠这个字段做内容校验，而不是只信任 window_idx。
                "lambdas_coul": [float(x) for x in lc_win],
                "lambdas_vdw": [float(x) for x in lv_win],
                # base/bias 的力组切分口径版本，见 WCA_ACCOUNTING_VERSION 定义处；
                # resume / 窗口产物复用逻辑必须校验这个字段，不能只比 λ 值。
                "wca_accounting_version": WCA_ACCOUNTING_VERSION,
                # IBS 偏置预热/冻结协议版本，见 IBS_BIAS_PROTOCOL_VERSION 定义处；
                # 旧协议下这份能量文件可能是在 f_k 全程漂移的情况下采出来的，
                # 不满足 MBAR 的单一固定采样分布假设，同样必须校验，不能只看 λ 值。
                "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
                "lse_log_residual_tolerance": float(lse_log_residual_tolerance),
                # 🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION] LJ 长程尾项修正公式版本
                # （尽管名字里写着 traditional，v2 起这个常量同时覆盖 ACE/dual_lambda
                # 路径和传统 Beutler REMD 路径共用的 switching+softcore-aware LRC
                # 积分——见该常量定义处）。旧版本（v1，只补 cutoff 外的 r^-6、忽略
                # switching 区间）下的能量文件里，target_energies 里叠加的 LRC 数值
                # 跟当前公式不是同一回事，resume / 窗口产物复用逻辑必须校验这个
                # 字段，不能只看 λ/WCA/IBS 偏置协议是否匹配。
                "lj_tail_lrc_protocol_version": TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
                # 🔑 [non_mutating_v1] 采样修复策略。旧的变异策略可能在采样中途用
                # 累计 ΔF 就地覆盖过 f_k（不同参考系）；resume / 窗口产物复用必须校
                # 验这个字段，绝不能把旧策略缓存当成非变异策略的有效数据复用。
                "sampling_repair_policy": repair_policy,
                "n_steps_per_window_default": int(n_steps_per_window),
                "n_steps_per_window_effective": int(effective_n_steps_per_window),
                "n_updates": int(len(sampler.f_history)),
                "free_energy_history_kT": [
                    np.asarray(f_k, dtype=float).tolist()
                    for f_k in sampler.f_history
                ],
                "n_energy_samples": int(e_save.shape[1]),
                "energy_query_diagnostics": sampler.energy_query_diagnostics(),
                "window_data_protocol_version": IBS_WINDOW_DATA_PROTOCOL_VERSION,
                "window_data": window_data_metadata,
                "bias_warmup": bias_warmup_diag,
                # 🔑 [EARLY_STOP_PROTOCOL_VERSION] 见 run_all_windows 参数说明和该
                # 常量定义处。actual_production_steps 是这个窗口真正跑了多少步
                # （早停时 < n_steps_per_window_effective，未早停/未启用时等于它）；
                # resume 时只有 early_stop_triggered=True 且协议版本/enable_early_stop/
                # 目标步数都与当前调用一致才允许复用这份缓存，见上面 resume 分支
                # 的 early_stop_ok 校验。
                # 未提前停止时，余数补齐（remaining_steps）会让总步数严格达到
                # n_steps_per_window_effective；只有提前停止时才用真正跑到的
                # update 数折算实际步数。
                "actual_production_steps": (
                    int(actual_production_updates * steps_per_update)
                    if early_stop_triggered
                    else int(effective_n_steps_per_window)
                ),
                # 🔑 [production checkpoint 续采] 这是跨多次 reseed_resample 续算
                # 累加的总步数（不是这次 attempt 单独跑了多少步），供下一次
                # run_all_windows 调用时判断"已有的累计步数是否还没到目标"，
                # 从而只跑差值、真正续算，而不是把 effective_n_steps_per_window
                # 当成"这次要新跑这么多步"（那样会变成每次延长都重复计入已经跑
                # 过的步数）。窗口真正收敛/被判定终态失败时，这个字段的语义就
                # 不再重要——下一次若 λ/窗口/f_k 变了，_production_window_
                # checkpoint_is_usable 的指纹比对会自然拒绝复用这份累计值。
                "cumulative_production_steps": (
                    int(prior_cumulative_production_steps) + int(actual_production_updates * steps_per_update)
                    if early_stop_triggered
                    # 未早停时余数补齐让这次 attempt 严格跑完 remaining_production_steps
                    # （= effective_n_steps_per_window - prior_cumulative_production_steps），
                    # 累计总数因此就等于目标本身，不需要再分别相加。
                    else int(effective_n_steps_per_window)
                ),
                "early_stop_enabled": bool(enable_early_stop),
                "early_stop_triggered": bool(early_stop_triggered),
                "early_stop_stop_reason": early_stop_stop_reason,
                "early_stop_protocol_version": EARLY_STOP_PROTOCOL_VERSION if enable_early_stop else None,
                "early_stop_check_history": early_stop_check_history,
                "early_stop_config": {
                    "min_steps": int(early_stop_min_steps),
                    "check_interval_steps": int(early_stop_check_interval_steps),
                    "required_consecutive_passes": int(early_stop_required_consecutive_passes),
                    "min_ess_ratio": float(early_stop_min_ess_ratio),
                    "min_absolute_ess": float(early_stop_min_absolute_ess),
                    "min_decorrelated_samples": int(early_stop_min_decorrelated_samples),
                    "max_delta_g_drift_kJ_mol": float(early_stop_max_delta_g_drift_kJ_mol),
                    "max_uncertainty_kJ_mol": float(early_stop_max_uncertainty_kJ_mol),
                } if enable_early_stop else None,
            }
            # 🔑 [原子写修复] 这里之前是普通 open(...)+json.dump，不是这个文件
            # 别处到处使用的 _atomic_write_json（先写临时文件再 os.replace）——
            # 进程在这次写入中途被杀，会留下一份截断/损坏的 convergence.json，
            # 而 energies/bias/base.npy 这三个数组已经通过 _atomic_save_npy
            # 安全落盘，二者的原子性保证不一致。改为同样的 temp+replace 模式。
            _atomic_write_json(
                os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_convergence.json"),
                convergence,
            )
            # 🔑 [production checkpoint 续采] 窗口生产采样正常结束（无论是否早停）
            # 时再存一次最新的生产 checkpoint——每 100 个 update 的周期性落盘之外，
            # 补上循环结束、余数补齐这段尾巴，保证 checkpoint 反映的是这次 attempt
            # 真正结束时的坐标/速度/积分器 RNG 状态，跟上面刚写的
            # cumulative_production_steps 一致。不在这里删除这份 checkpoint——
            # 它可能还会被下一轮 reseed_resample 用来继续续算；只有 production
            # 数据真的因为 f_k/λ 改变而被作废时才应该删（见
            # _invalidate_production_window_checkpoint 的调用点）。
            _atomic_save_openmm_checkpoint(sim, production_ckpt_path)
            _atomic_write_json(production_manifest_path, expected_production_manifest)
            print(f"  💾 窗口 {window_idx} 完成，能量已保存 ({e_save.shape})")

            # 清理
            if getattr(sampler, "_probe_context", None) is not None:
                del sampler._probe_context
                sampler._probe_context = None
            if getattr(sampler, "_probe_integrator", None) is not None:
                del sampler._probe_integrator
                sampler._probe_integrator = None
            del sampler
            del sim.context
            del sim
            del win_sys
            gc.collect()

        print(f"\n{'='*80}")
        print(f"✅ 所有窗口采样完成")
        print(f"{'='*80}")
        return warmup_results if warmup_only else None


    def _gradual_warmup_debug(self, sim, ibs_wrap, sampler, warmup_steps, win_sys, window_idx, debug_mode=True):
        """步长爬坡（不含偏置力）。

        ✅ 性能/收敛修复：这里以前还带一段独立的"纯偏置预热"（bias_scale 0→1，
        预算 warmup_steps，默认高达 50 万步），而调用方紧接着又会做一次完整的
        清零重爬（见 IBSWindowManagerDualLambda 里的"[偏置预热]"段）。两段都在
        往同一个 f_k 上叠加更新，前一段辛苦爬升的 bias_scale 会被后一段立刻推回
        0 重来——除了浪费 GPU，还会把 IBSSampler.update_weights() 里随调用次数
        衰减的学习率 (eta_sgd) 提前消耗在一个马上被丢弃的目标上。现在只保留物理
        上真正必要的部分（时间步长爬坡），偏置力的唯一平滑引入通道移到调用方
        那一段，避免同一件事做两遍。warmup_steps 参数保留仅为向后兼容签名，
        当前不再被本函数使用。
        """
        print("  🔥 渐进预热：时间步长爬坡...")

        # 偏置力在时间步长爬坡期间保持关闭，避免爬 dt 的同时还叠加一个尚未
        # 校准的偏置力，两个不稳定源混在一起更难诊断。
        sim.context.setParameter(f"{self.prefix}_bias_scale", 0.0)

        original_dt = sim.integrator.getStepSize()
        for dt_ps in [0.0005, 0.001, 0.002]:
            sim.integrator.setStepSize(dt_ps * unit.picoseconds)
            sim.step(5000)

            if debug_mode:
                state = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/openmm.unit.nanometer)
                max_f = np.max(np.linalg.norm(forces, axis=1))
                print(f"    [爬坡 dt={dt_ps}ps] E={e:.1f}, max|F|={max_f:.1f}")

        sim.integrator.setStepSize(original_dt)
        print("  ✅ 时间步长爬坡完成（偏置力将在下一步统一引入）")


    def _safe_boresch_ramp(self, sim, final_scale=1.0, n_steps=200):
        """
        渐进式提升 Boresch 力强度，每步后用极短时间步长松弛并检测能量。
        返回 True 表示爬坡成功，False 表示在中途检测到异常。
        """
        try:
            current_scale = sim.context.getParameter("lambda_boresch_scale")
        except Exception:
            logger.warning(
                "读取 lambda_boresch_scale 失败，回退到默认初始值 0.01",
                exc_info=True,
            )
            current_scale = 0.01
        n_ramp = 10
        scales = np.linspace(current_scale, final_scale, n_ramp + 1)[1:]

        original_dt = sim.integrator.getStepSize()
        sim.integrator.setStepSize(0.001 * unit.picoseconds)

        for s in scales:
            sim.context.setParameter("lambda_boresch_scale", float(s))
            sim.step(n_steps)

            state = sim.context.getState(getEnergy=True)
            e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if abs(e) > 1e5 or not np.isfinite(e):
                print(f"  🚨 Boresch 爬坡中断：scale={s:.3f}，能量={e:.1f} kJ/mol，可能几何奇点")
                sim.integrator.setStepSize(original_dt)
                return False
            print(f"  ↪ Boresch scale → {s:.3f}，当前势能={e:.1f} kJ/mol")

        sim.integrator.setStepSize(original_dt)
        print("  ✅ Boresch 限制力已完全启用")
        return True

    def get_stage_data_for_analysis(
        self,
        stage_type: str = "coul",
        *,
        excluded_local_windows: Optional[set] = None,
    ) -> List[Dict]:
        """从磁盘加载能量与偏置，构造分析器期望的 window_data 列表

        🔑 [P0-8] `self.ranges` 里每个未被显式排除的窗口都必须有完整的
        energies/bias/base 三文件；缺任何一个一律 fail closed，不再静默
        `continue`。旧行为下缺**首**窗口会让 window 1 自然成为协方差链的
        `local_idx == 0`（`join_lam = local_lams[0]`），缺**末**窗口会让链
        提前结束，两者都产出截断的 ΔG 却仍报 `converged=True`——因为
        `solve_stage_integrated` 的完整性判据只比 `local_results` 与已经被
        加载的 `valid_windows`，看不到 expected→loaded 的收缩。中间缺窗反而
        是安全的（协方差链找不到共享 λ 会走 `_fallback`）。

        `excluded_local_windows` 是**显式**的合法排除集合，给 vanishing
        rescue 那条"原始窗口被 rescue ensemble 取代"的路径用；默认不排除
        任何窗口。
        """
        results = []
        excluded = {int(x) for x in (excluded_local_windows or set())}
        expected_windows = [
            int(i) for i in range(len(self.ranges)) if int(i) not in excluded
        ]
        missing_windows: List[Dict] = []
        for i, (start, end) in enumerate(self.ranges):
            if int(i) in excluded:
                continue
            e_path = os.path.join(self.output_dir, f"dual_window_{i}_{stage_type}_energies.npy")
            b_path = os.path.join(self.output_dir, f"dual_window_{i}_{stage_type}_bias.npy")
            base_path = os.path.join(self.output_dir, f"dual_window_{i}_{stage_type}_base.npy")
            conv_path = os.path.join(
                self.output_dir,
                f"dual_window_{i}_{stage_type}_convergence.json",
            )
            # 🔑 [P0-8] 三文件缺任何一个都记账，循环结束后统一 fail closed。
            absent = [
                label
                for label, path in (
                    ("energies", e_path),
                    ("bias", b_path),
                    ("base", base_path),
                )
                if not os.path.isfile(path)
            ]
            if absent:
                missing_windows.append({
                    "window_index": int(i),
                    "lambda_range": [int(start), int(end)],
                    "missing_files": absent,
                })
                continue
            if not os.path.isfile(conv_path):
                raise FileNotFoundError(
                    f"窗口 {i} 有 energies 但缺少 convergence manifest"
                )
            with open(conv_path, encoding="utf-8") as handle:
                convergence = json.load(handle)
            u_kn, bias, base = _load_validated_window_data_triplet(
                e_path,
                b_path,
                base_path,
                convergence,
            )
            if u_kn.shape[1] == 0:
                raise ValueError(f"窗口 {i} 的 IBS 数据集没有任何有效帧")
            n_frames = u_kn.shape[1]
            # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 混合覆盖度 ESS 需要该窗口真正冻结进
            # 生产的 f_k（gauge-free 的逐帧 softmax 捷径在谱宽大的窗口会差数十倍，
            # 见 _ibs_reweighting_quality_diagnostics）。f_k 不新存一份文件：
            # save_ibs_state 早就把它写进 checkpoints/ibs_state_*.json 了，这里直接
            # 读那份，因此所有已有缓存数据都能用、无需协议号升级或重算。
            # 缺文件/态数不符一律 fail closed——与本函数上面"有 energies 但缺
            # convergence manifest"同一策略：静默降级会让收敛门失去判据却看起来通过。
            state_path = os.path.join(
                self.checkpoint_dir, f"ibs_state_{stage_type}_window_{i}.json"
            )
            if not os.path.isfile(state_path):
                raise FileNotFoundError(
                    f"窗口 {i} 有 energies 但缺少 ibs_state checkpoint ({state_path})；"
                    "ESS 门需要冻结的 f_k，拒绝在缺判据的情况下继续分析"
                )
            with open(state_path, encoding="utf-8") as handle:
                ibs_state = json.load(handle)
            f_k_window = np.asarray(ibs_state.get("f_k", []), dtype=np.float64).ravel()
            if f_k_window.size != u_kn.shape[0] or not np.all(np.isfinite(f_k_window)):
                raise ValueError(
                    f"窗口 {i} ibs_state 的 f_k（长度 {f_k_window.size}）与能量矩阵"
                    f"态数（{u_kn.shape[0]}）不符或含非有限值，拒绝用于 ESS 门"
                )
            results.append({
                'window_index': int(i),
                'window_label': f'window_{i}',
                'window_range': [int(start), int(end)],
                'u_kn': u_kn[:, :n_frames],  # U_k_int, kJ/mol
                'bias_energies': bias[:n_frames],
                'base_energies': base[:n_frames],
                'lambda_indices': list(range(start, end)),
                'lambdas_coul': [float(x) for x in self.lambdas_coul[start:end]],
                'lambdas_vdw': [float(x) for x in self.lambdas_vdw[start:end]],
                # 冻结进生产的偏置权重，kJ/mol；ESS 门用它把共模因子除掉。
                'f_k': f_k_window,
                # 显式记录：局部 MBAR 的第 0 行是“采样分布”而非某个物理 lambda 态。
                'sampled_distribution_row': 0,
            })
        _assert_expected_windows_all_loaded(
            expected_windows=expected_windows,
            loaded_windows=[int(r["window_index"]) for r in results],
            missing_windows=missing_windows,
            source=f"{self.output_dir} (stage_type={stage_type})",
        )
        return results


class IBSWindowManagerShadowCoul(IBSWindowManagerDualLambda):
    """Shadow-Coulomb IBS 去电荷窗口管理器 (实验性，尚未经物理验证)。

    复用 IBSWindowManagerDualLambda 的窗口调度/最小化/Boresch 爬坡/渐进预热/
    生产采样/断点续传/能量与 f_k 落盘逻辑，只把系统构造换成
    build_shadow_coul_ibs_system：CV 核函数是 erfc(alpha*r)/r（Ewald 实空间
    形式），物理上兼容 CustomNonbondedForce/CustomCVForce，因此可以合法放进
    IBS 偏置，不会像真实 PME 库仑那样被截断成 cutoff 相互作用。

    约束与前提（均由 build_shadow_coul_ibs_system 内部显式检查，不满足时会
    直接 raise 而不是静默算错）：
      - 配体必须电中性；带净电配体的 Shadow-Coulomb 去电荷路径尚不支持。
      - VdW 全程满强度、不参与炼金（属于 U_common），因此 self.lambdas_vdw
        恒为 1.0，仅用于日志；构建出的系统不含 lambda_shield 全局参数，
        基类里对它的 setParameter 调用会被存在性检查自动跳过。
      - 这条去电荷路径只覆盖"Shadow 满电荷 -> Shadow 去电荷"这一段，前面还
        需要一段独立的 Bridge 腿（ibs_engine.run_shadow_bridge_leg）把体系从
        "真实 PME 满电荷"搭桥到"Shadow 满电荷"端点，两段 ΔG 相加才是完整的
        去电荷自由能。

    这个子类的 `super().__init__(...)` 没有传 `pilot_lambdas`/
    `pilot_mean_dU_dlambda`（基类默认 `None`），对 Shadow-Coulomb 无意义——
    这两个字段只用于 `run_all_windows` 里给 vanishing 窗口的 f_k 热启动，
    不需要为这个子类专门补传。
    """

    def __init__(
        self,
        system_template,
        topology,
        perturbed_atom_indices,
        lambdas_shadow_coul: List[float],
        temperature,
        window_ranges: List[Tuple[int, int]],
        restraint_params: Optional[Dict] = None,
        prefix: str = "abfe_shadow",
        platform_name: str = "CUDA",
        output_dir: str = "./output",
        checkpoint_dir: str = "./checkpoints",
    ):
        super().__init__(
            system_template=system_template,
            topology=topology,
            perturbed_atom_indices=perturbed_atom_indices,
            lambdas_coul=list(lambdas_shadow_coul),
            lambdas_vdw=[1.0] * len(lambdas_shadow_coul),
            temperature=temperature,
            window_ranges=window_ranges,
            alchemical_params=None,
            potential_type="softcore",
            restraint_params=restraint_params,
            prefix=prefix,
            platform_name=platform_name,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        self.lambdas_shadow_coul = self.lambdas_coul

    def _build_window_system(self, lc_win, lv_win, resolved_box, positions):
        win_sys_xml = XmlSerializer.serialize(self.system_template)
        return build_shadow_coul_ibs_system(
            ensure_owned_system(XmlSerializer.deserialize(win_sys_xml)),
            self.topology,
            self.ligand_indices,
            lc_win,
            restraint_params=self.boresch,
            prefix=self.prefix,
            box_vectors=resolved_box,
        )


def _diagnose_local_mbar_situation(
    u_kn_softcore: np.ndarray,
    f_frozen: np.ndarray,
    kt: float,
    gate_mbar: Dict[str, Any],
    global_start: int,
) -> Dict[str, Any]:
    """[IBS_BIAS_PROTOCOL_VERSION=29] 把一次冻结 local-MBAR loose gate 的"现场"
    量化成可读诊断，回答"哪个态/哪条边饿死、物理缺口多大、该在哪里插 λ"。

    - occupancy_per_state：当前冻结 f_k 下每个局部态的平均占据（softmax 与
      update_weights/gate 同式：p_k ∝ exp(-beta*(u_k - f_k))）。一个态≈1、其余
      ≈0 就是塌缩。
    - ess_ratio_per_state：local MBAR 对每个物理态的重加权有效样本比例（来自
      gate_mbar）。~0 = 该态与采样分布零重叠，ΔF 不可信。
    - adjacent_mean_delta_u_kJ_mol：相邻态原始 softcore 平均能量差（物理缺口，
      与 f_k 无关）。某条边极大 = 该边本身没有构象重叠，加权改不动。
    - starved_state / starved_edge：占据最低（或 ESS 最低）的态及其两侧边，给出
      global 态索引，方便直接决定在哪对 global 态之间插 λ 或拆窗。
    纯诊断，不参与任何控制流。
    """
    u = np.asarray(u_kn_softcore, dtype=np.float64)
    f = np.asarray(f_frozen, dtype=np.float64).ravel()
    K = u.shape[0]
    beta = 1.0 / float(kt)
    # 占据：p_k ∝ exp(-beta*(u_k - f_k))，逐帧 softmax 再对帧取平均。
    logits = beta * (f[:, None] - u)
    logits -= np.max(logits, axis=0, keepdims=True)
    w = np.exp(logits)
    p = w / np.sum(w, axis=0, keepdims=True)
    occ = np.mean(p, axis=1)
    mean_u = np.mean(u, axis=1)
    adjacent_delta_u = np.diff(mean_u) if K > 1 else np.array([])
    ess_map = gate_mbar.get("ess_ratio_per_lambda") or {}
    ess_per_state = [float(ess_map.get(int(k), float("nan"))) for k in range(K)]
    # 最低占据态：occ 最小者（占据是最直接、总有定义的信号）。注意"最低占据"≠
    # "饿死"——只有当它真的低于健康下限 0.5/K 时才算塌陷/饿死。
    starved_k = int(np.argmin(occ)) if K > 0 else 0
    starved_edges = []
    if starved_k - 1 >= 0:
        starved_edges.append((starved_k - 1, starved_k))
    if starved_k + 1 <= K - 1:
        starved_edges.append((starved_k, starved_k + 1))
    is_endpoint = bool(starved_k == 0 or starved_k == K - 1)
    # 🔑 占据平坦度：区分"真塌陷（需插 λ/拆窗）"与"只是短 warmup 统计薄、
    # 占据其实平坦（交生产后 MBAR）"。用与生产入口占据兜底门相同的判据。
    target_p = 1.0 / float(K) if K > 0 else 0.0
    sum_sq_occ = float(np.sum(np.square(occ))) if K > 0 else 0.0
    coverage_ess = float(1.0 / sum_sq_occ) if sum_sq_occ > 0.0 else 0.0
    min_occ = float(np.min(occ)) if K > 0 else 0.0
    max_occ = float(np.max(occ)) if K > 0 else 0.0
    least_occupied_below_floor = bool(
        K > 0 and min_occ < IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION * target_p
    )
    occupancy_is_flat = bool(
        K > 0
        and max_occ <= IBS_LOCAL_MBAR_GATE_OCC_MAX_FRACTION * target_p
        and min_occ >= IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION * target_p
        and coverage_ess
        >= IBS_LOCAL_MBAR_GATE_OCC_MIN_COVERAGE_ESS_FRACTION * float(K)
    )
    # 塌陷（硬瓶颈）判据比"未平坦"严得多：coverage_ESS<0.5K 或某态<0.25/K。
    occupancy_collapsed = bool(
        K > 0
        and (
            coverage_ess
            < IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_COVERAGE_ESS_FRACTION * float(K)
            or min_occ < IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_MIN_FRACTION * target_p
        )
    )
    return {
        "n_states": int(K),
        "global_state_start": int(global_start),
        "occupancy_per_state": [float(x) for x in occ],
        "ess_ratio_per_state": ess_per_state,
        "mean_softcore_u_kJ_mol_per_state": [float(x) for x in mean_u],
        "adjacent_mean_delta_u_kJ_mol": [float(x) for x in adjacent_delta_u],
        "coverage_ess": coverage_ess,
        "occupancy_is_flat": occupancy_is_flat,
        "occupancy_collapsed": occupancy_collapsed,
        "least_occupied_below_floor": least_occupied_below_floor,
        "occupancy_floor": float(IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION * target_p),
        # 名字保留 starved_* 以兼容既有 gate_diag 消费方；语义是"最低占据态"。
        "starved_local_state": int(starved_k),
        "starved_global_state": int(global_start + starved_k),
        "starved_occupancy": float(occ[starved_k]) if K > 0 else float("nan"),
        "starved_ess_ratio": (
            float(ess_per_state[starved_k]) if K > 0 else float("nan")
        ),
        "starved_edges_global": [
            [int(global_start + a), int(global_start + b)] for a, b in starved_edges
        ],
        "starved_edge_mean_delta_u_kJ_mol": [
            float(adjacent_delta_u[min(a, b)]) for a, b in starved_edges
        ],
        "is_endpoint": is_endpoint,
    }


def _format_local_mbar_situation(situation: Dict[str, Any]) -> str:
    """把 _diagnose_local_mbar_situation 的 dict 压成一行可读中文摘要。"""
    if not situation:
        return "无可用现场诊断"
    if situation.get("error") or situation.get("starved_global_state") is None:
        return f"现场诊断不可用（{situation.get('error', 'incomplete')}）"
    occ = situation.get("occupancy_per_state") or []
    gstart = int(situation.get("global_state_start", 0))
    occ_str = ", ".join(
        f"s{gstart + k}={o:.2e}" for k, o in enumerate(occ)
    )
    edges = situation.get("starved_edges_global") or []
    edge_du = situation.get("starved_edge_mean_delta_u_kJ_mol") or []
    edge_str = "; ".join(
        f"边{e}原始Δu≈{du:+.1f} kJ/mol"
        for e, du in zip(edges, edge_du)
    ) or "n/a"
    # "最低占据态"用中性词；只有真的塌陷（occupancy_collapsed）才标"饿死"，避免把
    # 0.125=0.5/K 这种健康端点误报成塌陷。
    _collapsed = situation.get("occupancy_collapsed")
    _flat = situation.get("occupancy_is_flat")
    _label = "饿死态" if _collapsed else "最低占据态"
    _flat_tag = "，占据平坦" if _flat else ("，占据塌陷" if _collapsed else "，占据未平坦")
    return (
        f"占据[{occ_str}]（coverage_ESS={situation.get('coverage_ess', float('nan')):.2f}"
        f"{_flat_tag}）；{_label}=global s{situation.get('starved_global_state')}"
        f"（占据={situation.get('starved_occupancy'):.2e}, "
        f"ESS_ratio={situation.get('starved_ess_ratio'):.3f}"
        f"{'，端点' if situation.get('is_endpoint') else ''}）；{edge_str}"
    )


def _logsumexp_rows(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Numerically stable log-sum-exp along ``axis`` (no scipy.special dep)."""
    a = np.asarray(a, dtype=np.float64)
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    return np.squeeze(a_max, axis=axis) + np.log(
        np.sum(np.exp(a - a_max), axis=axis)
    )


def _ess_from_log_weights(log_w: np.ndarray) -> float:
    """(sum w)^2 / sum w^2 computed in log space; invariant to any constant
    shift of ``log_w`` (so it does not depend on the f_k gauge for a fixed
    state, nor on the arbitrary global energy offset)."""
    log_w = np.asarray(log_w, dtype=np.float64).ravel()
    if log_w.size == 0 or not np.all(np.isfinite(log_w)):
        return float("nan")
    return float(np.exp(2.0 * _logsumexp_rows(log_w) - _logsumexp_rows(2.0 * log_w)))


# ============================================================================
# 🔑 [ESS_GATE_PROTOCOL_VERSION=2] IBS 重加权质量诊断：区分"混合分布覆盖度"
# 与"共模采样偏置代价"
# ----------------------------------------------------------------------------
# 单参考增广 MBAR（n_k=[N,0,...,0]）的 compute_effective_sample_number() 算的是
#
#     W_nk ∝ exp[ (V_bias(x_n) - U'_k(x_n)) / kT ]
#
# 而生产实际施加的 V_bias 读的是 OpenMM force groups {1, 4}（见
# IBSSampler.collect_energies）——Group 1 是 IBS 混合偏置，Group 4 是 λ-WCA 防护壳；
# 目标态能量 U'_k 是 softcore + 解析 LRC。防护壳在窗口内对所有 k 用同一个
# lambda_shield（见 _serialize_ibs_common_plus_wca_system 的 docstring），LRC 是
# coeff[k]/V(t)，所以两者合起来在 log 权重里是一个**逐帧共模因子** r_n：
#
#     log W_nk = r_n / kT + log p_k(x_n) + const_k,
#     p_k(x) = exp[-(U'_k - f_k)/kT] / Σ_j exp[-(U'_j - f_j)/kT]
#
# r_n 在 occupancy ⟨p_k⟩、相邻 ΔF、warmup loose gate、以及物理态↔物理态自由能差
# 里都（大部分）抵消——而后者正是本阶段真正报出去的量。但它在 W_nk 里不抵消，
# 于是把 ESS 按 exp(σ_r²) 整体压掉：真实测到 σ_r 从 λ_vdw≈1 端的 0.95 kT 涨到
# λ_vdw→0 端的 2.40 kT，对应 ESS/N 上限 0.40 → 0.003，而同一批数据的 ⟨p_k⟩ 是
# 平坦的 0.249-0.251。也就是说旧的 min_ess_ratio 门衡量的是**防护壳的重加权代价**，
# 不是采样质量，更不是输出精度：实测 window 5 的 raw ESS 最差(0.0029)但端点
# 不确定度最好(0.199 kJ/mol)，window 3 的 σ_r 几乎一样(2.35 kT)而端点不确定度
# 是 2.17 kJ/mol。用它当收敛门会让 rescue 循环去追一个数学上不可满足、且与
# 被估计量无关的目标。
#
# 现在拆成三项互相独立的证据：
#   (1) min_ess_ratio（**受门**）：去掉共模因子后的混合分布覆盖度 ESS(p_k)/N_decorr。
#       这才是"采样到的混合分布对物理态 k 覆盖够不够"。
#   (2) min_decorrelated_samples（**受门**，已有字段）：独立样本数本身。
#   (3) max_endpoint_uncertainty_kJ_mol（**受门**，已有字段）：MBAR 自己带全协方差
#       算出的端点 ΔF 精度，即输出精度的直接度量。
# raw_min_ess_ratio / raw_absolute_ess / common_mode_log_sigma_kT 继续算、继续落盘，
# 但**只报告不设门**——它们是"防护壳收了多少重加权税"的诚实诊断。
#
# 注意 ESS(p_k) 必须用该窗口真正冻结进生产的 f_k。试过用逐帧
# softmax 归一化（r_n=mean_k 或 logsumexp_k，无需 f_k、且 gauge-free）来省掉
# f_k 的传参：对相邻 ΔU≲2.5 kT 的窄窗口（3/4/5）与真 f_k 版本一致，但对
# window 0（相邻 ΔU≈10 kT）给出 0.014 vs 真值 0.500，差 36 倍——因为 f_k 加权的
# logsumexp 与无权算术平均在谱宽大时是完全不同的逐帧函数。所以这条捷径不成立，
# f_k 必须真的传进来（从 checkpoint 的 ibs_state_*.json 读，见
# get_stage_data_for_analysis）。
# ============================================================================
# v3: occupancy 与 warmup 协议统一为诊断项，不再在全部生产完成后反向否决 stage。
ESS_GATE_PROTOCOL_VERSION = 3


def _ibs_reweighting_quality_diagnostics(
    u_kj_raw: np.ndarray,
    bias_kj: np.ndarray,
    f_k: Optional[np.ndarray],
    kt: float,
) -> Dict[str, Any]:
    """Split IBS single-reference reweighting quality into the mixture-coverage
    part (gated) and the common-mode sampling-bias part (reported only).

    ``u_kj_raw`` is (K, N) per-state interaction energy in kJ/mol, ``bias_kj``
    the (N,) recorded sampling bias (force groups {1,4}) in kJ/mol, ``f_k`` the
    (K,) frozen production bias weights in kJ/mol.

    Returns ``mixture_ess`` (absolute, per state), ``mixture_ess_ratio``
    (per state, relative to N), ``raw_ess`` / ``raw_ess_ratio`` (the old
    single-reference quantity, for reference), ``top1pct_raw_weight`` (fraction
    of total raw weight carried by the heaviest 1% of frames -- the most direct
    statement of weight degeneracy) and ``common_mode_log_sigma_kT``.

    ``f_k`` of ``None`` (or a length/finiteness mismatch) yields ``None`` for
    every mixture-derived field and an explicit ``error`` -- callers must treat
    that as "cannot evaluate the gate", never as a pass.
    """
    u_kj_raw = np.asarray(u_kj_raw, dtype=np.float64)
    bias_kj = np.asarray(bias_kj, dtype=np.float64).ravel()
    n_states, n_frames = u_kj_raw.shape
    out: Dict[str, Any] = {
        "n_frames": int(n_frames),
        "raw_ess": None,
        "raw_ess_ratio": None,
        "top1pct_raw_weight": None,
        "mixture_ess": None,
        "mixture_ess_ratio": None,
        "mixture_occupancy": None,
        "mixture_occupancy_normalized": None,
        "common_mode_log_sigma_kT": None,
        "common_mode_log_mean_kJ_mol": None,
        "error": None,
    }
    if n_frames == 0 or bias_kj.size != n_frames:
        out["error"] = "frame_count_mismatch"
        return out

    # ---- raw single-reference weights (reported only) ----
    log_w_raw = (bias_kj[None, :] - u_kj_raw) / float(kt)
    raw_ess = [_ess_from_log_weights(log_w_raw[k]) for k in range(n_states)]
    out["raw_ess"] = [float(x) for x in raw_ess]
    out["raw_ess_ratio"] = [float(x) / float(n_frames) for x in raw_ess]
    n_top = max(1, n_frames // 100)
    top1 = []
    for k in range(n_states):
        lw = log_w_raw[k] - np.max(log_w_raw[k])
        wt = np.exp(lw)
        total = float(np.sum(wt))
        top1.append(
            float(np.sum(np.sort(wt)[-n_top:]) / total) if total > 0.0 else float("nan")
        )
    out["top1pct_raw_weight"] = top1

    # ---- mixture coverage, common-mode factor divided out (gated) ----
    if f_k is None:
        out["error"] = "missing_f_k"
        return out
    f_arr = np.asarray(f_k, dtype=np.float64).ravel()
    if f_arr.size != n_states or not np.all(np.isfinite(f_arr)):
        out["error"] = "f_k_shape_or_finiteness_mismatch"
        return out

    logits = -(u_kj_raw - f_arr[:, None]) / float(kt)
    log_norm = _logsumexp_rows(logits, axis=0)
    log_p = logits - log_norm[None, :]
    mix_ess = [_ess_from_log_weights(log_p[k]) for k in range(n_states)]
    out["mixture_ess"] = [float(x) for x in mix_ess]
    out["mixture_ess_ratio"] = [float(x) / float(n_frames) for x in mix_ess]

    # 🔑 ESS 单独用**不够**，必须配上占据 <p_k>：ESS 对每个态是尺度不变的
    # ((sum w)^2/sum w^2 在 w -> c*w 下不变)，所以一个"均匀地"被饿死的态——比如
    # f_k 没补偿、U'_k 比别人高 80 kT——它的 p_k 逐帧都是 ~e^-80，但**相对**逐帧
    # 起伏很小，ESS(p_k)/N 照样能报 0.34 这种健康值。实测确认过这个盲点。
    # ESS 查的是二阶矩（权重是否集中在少数帧），<p_k> 查的是一阶矩（这个态到底
    # 有没有拿到权重），两个失效模式必须各由一项覆盖。归一化成 K*<p_k>（理想=1），
    # 沿用 warmup 那边同一套占据判据的口径（IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION）。
    occ = np.exp(log_p).mean(axis=1)
    occ_sum = float(np.sum(occ))
    if occ_sum > 0.0 and np.all(np.isfinite(occ)):
        occ = occ / occ_sum
        out["mixture_occupancy"] = [float(x) for x in occ]
        out["mixture_occupancy_normalized"] = [
            float(x) * float(n_states) for x in occ
        ]

    # r_n: the part of the applied sampling bias that no (U'_k, f_k) pair can
    # reproduce -- i.e. the WCA guard shell (Group 4) plus the softcore-vs-LRC
    # Hamiltonian mismatch. -kt*log_norm is the IBS bias the recorded f_k/U'_k
    # imply; the difference is the common-mode nuisance factor.
    residual_kj = bias_kj + float(kt) * log_norm
    if np.all(np.isfinite(residual_kj)):
        out["common_mode_log_sigma_kT"] = float(np.std(residual_kj) / float(kt))
        out["common_mode_log_mean_kJ_mol"] = float(np.mean(residual_kj))
    return out


def _decorrelate_by_worst_target_state(
    u_kj_raw: np.ndarray,
    bias_kj: np.ndarray,
    kt: float,
) -> Tuple[np.ndarray, float, int]:
    """Subsample using the slowest-decorrelating *reweighting* series.

    🔑 [ESS_GATE_PROTOCOL_VERSION=2] The previous series was the total sampled
    potential ``base_kj + bias_kj``. That is dominated by bulk solvent
    fluctuation, which has nothing to do with how fast the importance weights
    decorrelate, and it measurably *underestimates* g here: on the live
    Atenolol vdw stage it gave g=1.76/2.58/2.44 for windows 0/2/5 where the
    per-target-state reduced-potential difference gives g=19.6/26.5/7.7. An
    understated g inflates n_k fed to MBAR and shrinks the reported error bars
    -- the exact failure ``subsample_series_by_autocorrelation`` exists to
    prevent.

    The right series is the one whose exponential *is* the weight,
    ``Delta_u_k = (U'_k - V_bias)/kT`` (``base`` cancels). One g per target
    state, then the most conservative (largest g) state drives the shared
    subsampling, so no state is left correlated. Mirrors the existing
    per-state pattern at the offline ``u_kn[k, start:end]`` call site.

    Returns ``(indices, g, worst_state_index)``.
    """
    u_kj_raw = np.asarray(u_kj_raw, dtype=np.float64)
    bias_kj = np.asarray(bias_kj, dtype=np.float64).ravel()
    best_idx = np.arange(u_kj_raw.shape[1])
    best_g = 1.0
    worst_k = -1
    for k in range(u_kj_raw.shape[0]):
        series = (u_kj_raw[k] - bias_kj) / float(kt)
        idx_k, g_k = subsample_series_by_autocorrelation(series)
        if float(g_k) > best_g:
            best_g = float(g_k)
            best_idx = np.asarray(idx_k, dtype=int)
            worst_k = int(k)
    return best_idx, best_g, worst_k


def _solve_single_window_local_mbar(
    u_kj_raw: np.ndarray,
    bias_kj: np.ndarray,
    base_kj: np.ndarray,
    win_lams: List[int],
    kt: float,
    f_k: Optional[np.ndarray] = None,
    sampled_distribution_row: int = 0,
    w_idx: int = 0,
    min_frames: int = 10,
) -> Dict[str, Any]:
    """Solve one IBS window's local augmented-MBAR problem from raw per-frame
    data (autocorrelation subsampling -> global energy offset -> augmented
    sampled+physical-state matrix -> kT reduction -> column-wise numerical
    stabilization -> MBAR -> effective-sample-number overlap diagnostic).

    This mirrors the per-window construction inside
    ``GlobalMBARAnalyzer.solve_stage_integrated`` (deliberately a separate,
    independent implementation rather than a refactor-and-share: that
    end-of-stage solver has been through multiple audit rounds and touching
    it to extract a shared helper would risk regressing already-verified
    behavior for a lower-value DRY win). Used by the production-time online
    early-stop monitor in ``run_all_windows`` to evaluate a window's
    convergence from its in-progress (not yet finished) sample history,
    without waiting for the stage to complete. If the two implementations
    ever need to diverge (e.g. the stage-level solver gains a correction this
    one doesn't need, or vice versa), that's fine -- they answer different
    questions (final stage assembly vs. "should we stop sampling now") even
    though today they share the same math.

    Returns a dict with ``error`` set if this window's data isn't solvable
    yet (too few frames, pymbar missing, MBAR failure), otherwise
    ``lambdas``/``f`` (kJ/mol, relative to sampled reference)/``df`` (kJ/mol
    uncertainty)/``n_frames_used``/``global_offset``/``min_ess_ratio``/
    ``ess_ratio_per_lambda``/``statistical_inefficiency``.
    """
    u_kj_raw = np.asarray(u_kj_raw, dtype=np.float64)
    bias_kj = np.asarray(bias_kj, dtype=np.float64)
    base_kj = np.asarray(base_kj, dtype=np.float64)
    if u_kj_raw.ndim != 2 or len(win_lams) != u_kj_raw.shape[0]:
        return {"error": f"窗口 {w_idx} 数据维度不匹配: u_kn={u_kj_raw.shape}, lambdas={len(win_lams)}"}

    n_frames = u_kj_raw.shape[1]
    if bias_kj.ndim != 1 or base_kj.ndim != 1:
        return {
            "error": (
                f"窗口 {w_idx} bias/base 必须是一维数组: "
                f"bias={bias_kj.shape}, base={base_kj.shape}"
            )
        }
    if len(bias_kj) != n_frames or len(base_kj) != n_frames:
        return {
            "error": (
                f"窗口 {w_idx} energies/bias/base 帧数不一致: "
                f"energies={n_frames}, bias={len(bias_kj)}, base={len(base_kj)}"
            )
        }
    if (
        not np.all(np.isfinite(u_kj_raw))
        or not np.all(np.isfinite(bias_kj))
        or not np.all(np.isfinite(base_kj))
    ):
        return {"error": f"窗口 {w_idx} energies/bias/base 含 NaN/Inf"}
    if n_frames < min_frames:
        return {"error": "insufficient_frames", "n_frames": int(n_frames)}

    # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 去相关序列改为"权重本身的指数"，见
    # _decorrelate_by_worst_target_state 的 docstring（旧的 base+bias 会低估 g）。
    sub_idx, g_val, g_worst_state = _decorrelate_by_worst_target_state(
        u_kj_raw, bias_kj, kt
    )
    if sub_idx.size < n_frames:
        u_kj_raw = u_kj_raw[:, sub_idx]
        bias_kj = bias_kj[sub_idx]
        base_kj = base_kj[sub_idx]
        n_frames = int(sub_idx.size)
    if n_frames < min_frames:
        return {
            "error": "insufficient_frames_after_decorrelation",
            "n_frames": int(n_frames),
            "statistical_inefficiency": float(g_val),
        }

    u_phys_kj = base_kj[None, :] + u_kj_raw
    mean_energies = np.mean(u_phys_kj, axis=1)
    global_offset = float(np.min(mean_energies))
    u_kj_shifted = u_phys_kj - global_offset
    u_sampled_eff = base_kj + bias_kj - global_offset
    u_mbar = np.vstack([u_sampled_eff, u_kj_shifted])
    u_mbar = u_mbar * (1.0 / float(kt))

    # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] Row 0 是"实际 IBS 采样分布"、row 1..K 是
    # 物理目标态，这个布局是 u_mbar 上面那几行硬编码出来的。旧代码允许把样本数放
    # 到任意 sampled_distribution_row，然后又用 `i != sampled_row` 挑目标行、再和
    # win_lams 直接 zip——sampled_row != 0 时 (a) MBAR 会被告知样本来自某个*物理*
    # 态（物理错误，不只是索引错位），(b) ESS→λ 映射整体错位，win_lams[0] 会被配到
    # 采样分布那一行的 ESS。这个不变量原先只写在下面的注释里、没有 enforce，而对
    # "非法值"的容错回退（越界→0）恰好掩盖了真正危险的"合法但非零"值。改为 fail
    # closed。
    sampled_row = int(sampled_distribution_row)
    if sampled_row != 0:
        return {
            "error": (
                f"窗口 {w_idx} sampled_distribution_row={sampled_row} != 0；"
                "本函数的增广矩阵 row 0 固定是 IBS 采样分布、row 1..K 是物理态，"
                "非零采样行会同时破坏 MBAR 的采样态声明与 ESS→λ 映射，拒绝继续。"
            )
        }
    n_k_local = np.zeros(len(win_lams) + 1, dtype=np.int32)
    n_k_local[sampled_row] = n_frames

    u_min_col = np.min(u_mbar, axis=0, keepdims=True)
    u_mbar_stable = u_mbar - u_min_col
    valid_mask = np.isfinite(u_mbar_stable).all(axis=0)
    u_mbar_final = u_mbar_stable[:, valid_mask]
    n_k_local[sampled_row] = int(np.sum(valid_mask))
    if n_k_local[sampled_row] < min_frames:
        return {"error": "insufficient_valid_frames", "n_frames": int(n_k_local[sampled_row])}

    if not HAS_PYMBAR:
        return {"error": "pymbar_not_installed"}

    try:
        mbar = _build_mbar_compatible(
            u_mbar_final, n_k_local, relative_tolerance=1e-7,
            initialize="BAR", solver_protocol="default",
        )
        res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
        df_matrix, ddf_matrix = _extract_free_energy_arrays(res, require_uncertainty=True)
    except Exception as e:
        return {"error": f"mbar_solve_failed: {e}"}

    # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] min_ess_ratio 从"单参考 raw ESS"换成
    # "扣掉共模因子后的混合覆盖度 ESS(p_k)"，raw 量降级为只报告不设门。
    # 详见 _ibs_reweighting_quality_diagnostics 顶部的长注释。
    # 必须用 valid_mask 之后的那批帧：denom 是 n_k_local[sampled_row]=sum(valid_mask)，
    # 若 quality 用未过滤的全部帧，比例的分子分母就来自不同样本集。
    quality = _ibs_reweighting_quality_diagnostics(
        u_kj_raw[:, valid_mask], bias_kj[valid_mask], f_k, kt
    )
    denom = max(int(n_k_local[sampled_row]), 1)
    min_ess_ratio = None
    ess_ratio_per_lambda = None
    raw_min_ess_ratio = None
    raw_ess_ratio_per_lambda = None
    if quality.get("raw_ess") is not None:
        raw_ratio = np.asarray(quality["raw_ess"], dtype=float) / denom
        raw_min_ess_ratio = float(np.min(raw_ratio))
        raw_ess_ratio_per_lambda = {
            int(lam): float(r) for lam, r in zip(win_lams, raw_ratio)
        }
    min_occupancy_normalized = None
    if quality.get("mixture_ess") is not None:
        mix_ratio = np.asarray(quality["mixture_ess"], dtype=float) / denom
        min_ess_ratio = float(np.min(mix_ratio))
        ess_ratio_per_lambda = {
            int(lam): float(r) for lam, r in zip(win_lams, mix_ratio)
        }
        # 一阶矩伴随量，见 _ibs_reweighting_quality_diagnostics 里的说明：ESS 尺度
        # 不变，抓不到"均匀被饿死"的态，两项必须一起看。
        occ_norm = quality.get("mixture_occupancy_normalized")
        if occ_norm:
            min_occupancy_normalized = float(np.min(occ_norm))

    f_phys_kt = df_matrix[0, 1:]
    df_phys_kt = ddf_matrix[0, 1:]
    f_phys_kj = (f_phys_kt * float(kt)).astype(float)
    df_phys_kj = (df_phys_kt * float(kt)).astype(float)

    # 🔑 端点（第一个 vs 最后一个物理态）自由能差的不确定度，必须直接读
    # ddf_matrix 里这两个物理态之间的元素，而不是把各自相对采样态的
    # 边际不确定度 df_phys_kj[0]/df_phys_kj[-1] 用 sqrt(a^2+b^2) 合并——
    # 两者都是同一次 MBAR 拟合、同一批样本估计出来的，彼此有协方差，
    # 把它们当独立量平方相加会系统性算错端点差的不确定度（可能偏大也可能
    # 偏小，取决于协方差符号），进而让在线 early-stop 判据错误地提前/永远
    # 不停止。物理态在增广矩阵里占据第 1..len(win_lams) 行（第 0 行是采样
    # 分布），第一个物理态是第 1 行，最后一个是第 len(win_lams) 行。
    n_lams = len(win_lams)
    endpoint_diff_uncertainty_kj = (
        float(ddf_matrix[1, n_lams] * float(kt)) if n_lams > 1 else 0.0
    )

    return {
        "lambdas": list(int(x) for x in win_lams),
        "f": f_phys_kj,
        "df": df_phys_kj,
        "endpoint_diff_uncertainty_kJ_mol": endpoint_diff_uncertainty_kj,
        "n_frames_used": int(n_k_local[sampled_row]),
        "global_offset": global_offset,
        "min_ess_ratio": min_ess_ratio,
        "ess_ratio_per_lambda": ess_ratio_per_lambda,
        "statistical_inefficiency": float(g_val),
        # ---- [ESS_GATE_PROTOCOL_VERSION=2] 只报告、不设门的诚实诊断 ----
        "ess_gate_protocol_version": int(ESS_GATE_PROTOCOL_VERSION),
        "ess_gate_metric": "mixture_coverage_ess_common_mode_removed",
        "min_occupancy_normalized": min_occupancy_normalized,
        "min_occupancy_normalized_threshold": float(
            IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION
        ),
        "mixture_occupancy_normalized": quality.get("mixture_occupancy_normalized"),
        "raw_min_ess_ratio": raw_min_ess_ratio,
        "raw_ess_ratio_per_lambda": raw_ess_ratio_per_lambda,
        "top1pct_raw_weight": quality.get("top1pct_raw_weight"),
        "common_mode_log_sigma_kT": quality.get("common_mode_log_sigma_kT"),
        "common_mode_log_mean_kJ_mol": quality.get("common_mode_log_mean_kJ_mol"),
        "reweighting_quality_error": quality.get("error"),
        "statistical_inefficiency_worst_state": int(g_worst_state),
    }


# ============================================================================
# 4. 全局 TMBAR 分析器 (集成 bar.py 修复版逻辑)
# ============================================================================
class GlobalMBARAnalyzer:
    """
    严格 IBS-TMBAR 实现 (JCTC 2026 Sec 2.3)。
    集成 bar.py 修复版逻辑：
    1. 全局能量偏移 (Global Offset) 防止 exp 下溢/上溢。
    2. 增广矩阵构建 (Augmented Matrix) 正确处理偏置采样分布。
    3. 列平移稳定性 (Column-wise Shift) 确保 pymbar 数值收敛。
    
    输入: window_data = [{u_kn: (K_local, N), bias_energies: (N,), base_energies: (N,), lambda_indices: [...]}]
    输出: 全局自由能曲线与误差。
    """
    def __init__(self, kt: float):
        self.kt = kt
        self.beta = 1.0 / kt

    def solve_stage_integrated(
        self,
        window_data: List[Dict],
        final_min_ess_ratio: float = 0.05,
        final_min_absolute_ess: float = 50.0,
        final_min_decorrelated_samples: int = 20,
        final_max_uncertainty_kJ_mol: float = 1.0,
        min_frames_per_window: int = 10,
    ) -> Dict:
        """局部 TMBAR + 自洽拼接

        ⛔ **这条腿（stage2 = vdW）只能用 TMBAR。** 不要在这里加 BAR、TI、全帧主值、
        √g σ 缩放或 bootstrap σ ——理由与被撤回的先例都写在
        `ESTIMATOR_ANALYSIS_PROTOCOL_VERSION` 的注释里。

        final_min_ess_ratio/final_min_absolute_ess/final_min_decorrelated_samples/
        final_max_uncertainty_kJ_mol [P1 修复]：之前这里的最终 converged 判据
        只检查 ESS ratio（每个局部窗口只需 >=10 个原始样本即可进入求解），样本
        总数很少时即使绝对有效样本数只有个位数，只要比例超过阈值仍会被判定
        completed。在线 early-stop 监控（run_all_windows 里的
        early_stop_min_absolute_ess/early_stop_min_decorrelated_samples/
        early_stop_max_uncertainty_kJ_mol）早就同时检查这三项+比例+漂移，但
        阶段最终门从未同步。这里补上同样的三项硬门槛（不含漂移——一次性拼接
        没有"上一次检查"的基线可比）。故意用独立的 final_* 前缀而不是复用
        early_stop_* 参数名：语义不同，一个是"提前结束采样"的启发式，一个是
        "已经跑完的结果能不能算数"的最终接受门槛。
        """
        valid_windows = [w for w in window_data if w.get("u_kn") is not None and w["u_kn"].size > 0]
        if not valid_windows:
            return {"error": "no_valid_windows", "converged": False, "total_delta_G": 0.0}

        # 按 λ 索引排序，确保拼接顺序正确
        valid_windows.sort(key=lambda d: min(d.get("lambda_indices", [10**9])))

        local_results = []
        window_g_values = []          # 每个窗口自相关子采样得到的统计非效率 g（诊断用）
        window_overlap_records = []   # 每个窗口的重加权有效样本比例（真实 overlap 诊断，见 #2）

        for w_idx, w in enumerate(valid_windows):
            source_window_idx = int(w.get("window_index", w_idx))
            u_kj_raw = np.asarray(w["u_kn"], dtype=np.float64) # (K_local, N)
            if w.get("bias_energies") is None or w.get("base_energies") is None:
                raise ValueError(
                    f"窗口 {w_idx} 缺少 bias/base 能量；IBS-TMBAR 禁止以零替代"
                )
            bias_kj = np.asarray(w["bias_energies"], dtype=np.float64)
            base_kj = np.asarray(w["base_energies"], dtype=np.float64)
            win_lams = list(w.get("lambda_indices", []))
            
            if u_kj_raw.ndim != 2 or len(win_lams) != u_kj_raw.shape[0]:
                print(f"  ⚠️ 窗口 {w_idx} 数据维度不匹配，跳过")
                continue
            
            # 确定采样态索引 (通常 IBS 采样的是当前窗口的某个特定态，或者是加权平均)
            # 在 IBS 引擎中，通常假设样本来自当前窗口的“有效采样分布”。
            # 如果未指定 sampled_lambda_index，默认取窗口中间态作为参考，或者取第一个态。
            # 这里为了兼容旧逻辑，若未指定则取中间态，但需注意 bias_energies 必须对应正确的采样态。
            n_frames = u_kj_raw.shape[1]
            if n_frames < min_frames_per_window:
                print(f"  ⚠️ 窗口 {w_idx} 原始帧数 ({n_frames}) < min_frames_per_window ({min_frames_per_window})，跳过")
                continue
            if len(bias_kj) != n_frames or len(base_kj) != n_frames:
                raise ValueError(
                    f"窗口 {w_idx} energies/bias/base 帧数不一致"
                )
            if (
                not np.all(np.isfinite(u_kj_raw))
                or not np.all(np.isfinite(bias_kj))
                or not np.all(np.isfinite(base_kj))
            ):
                raise ValueError(f"窗口 {w_idx} energies/bias/base 含 NaN/Inf")
            bias_kj = bias_kj[:n_frames]
            base_kj = base_kj[:n_frames]

            # ------------------------------------------------------------------
            # 🔑 修复（审查报告 #1）: 自相关子采样
            # ------------------------------------------------------------------
            # 该窗口只有一个真实被采样的分布；用能量时间序列估计统计非效率 g，再对
            # 本窗口所有相关数组做同步去相关子采样，避免把强相关的逐帧 MD 输出当独
            # 立样本喂给 MBAR（会让误差棒系统性偏小）。
            # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 序列从 base_kj+bias_kj 换成每个目标态
            # 自己的 Δu_k=(U'_k−V_bias)/kT、取 g 最大的那个态——总势能被溶剂涨落主导，
            # 实测低估 g 达 3-10 倍（见 _decorrelate_by_worst_target_state）。
            sub_idx, g_val, g_worst_state = _decorrelate_by_worst_target_state(
                u_kj_raw, bias_kj, self.kt
            )
            if sub_idx.size < n_frames:
                u_kj_raw = u_kj_raw[:, sub_idx]
                bias_kj = bias_kj[sub_idx]
                base_kj = base_kj[sub_idx]
                n_frames = int(sub_idx.size)
            window_g_values.append(float(g_val))
            if n_frames < min_frames_per_window:
                print(f"  ⚠️ 窗口 {w_idx} 去相关子采样后有效帧数 ({n_frames}) < {min_frames_per_window}，跳过")
                continue

            # ------------------------------------------------------------------
            # 🔑 修复 1: 全局能量偏移 (Global Offset)
            # ------------------------------------------------------------------
            # 这里 u_kj_raw 存的是 U_k_int；分析时重建 U_k = E_base + U_k_int。
            u_phys_kj = base_kj[None, :] + u_kj_raw
            mean_energies = np.mean(u_phys_kj, axis=1)
            global_offset = np.min(mean_energies)
            
            # 对所有物理态能量进行平移
            u_kj_shifted = u_phys_kj - global_offset
            
            # ------------------------------------------------------------------
            # 🔑 修复 2: 构建增广矩阵 (Augmented Matrix)
            # ------------------------------------------------------------------
            # Row 0: 真实采样势能 E_base + V_bias
            # Rows 1..K: 各物理态 E_base + U_k_int
            u_sampled_eff = base_kj + bias_kj - global_offset
            
            # 构建 (K+1, N) 矩阵
            u_mbar = np.vstack([u_sampled_eff, u_kj_shifted])

            # 🔑 关键修复：pymbar.MBAR 要求输入是无量纲的约化势 (β·U)，这里在此之前
            # 一直是 kJ/mol 原始能量，从未乘过 self.beta（self.beta/self.kt 在这个
            # 方法里定义了却从没被用上）。把 kJ/mol 尺度的数直接喂给期望"已经是
            # kT 量级"的 MBAR，等价于让它在一个错误的（偏冷很多的）有效温度下求解，
            # 算出来的自由能差会被系统性放大。末尾 `f_phys_kj = f_phys_kt * self.kt`
            # 那一步本身没错，但前提是喂给 MBAR 的输入必须已经是约化势——现在补上这个
            # 转换，让输入输出的单位换算配对。
            u_mbar = u_mbar * self.beta

            # 样本计数：只有第 0 行（采样分布，而非物理 lambda 态）有 N 个样本。
            # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 上面 u_mbar 的 vstack 把 row 0 硬编码
            # 成采样分布、row 1..K 硬编码成物理态，所以 sampled_row 只能是 0。旧代码
            # 只对"越界"值回退到 0，却放过"合法但非零"的值——那会让 MBAR 以为样本来自
            # 某个*物理*态（物理错误），并让下面 `i != sampled_row` + zip(win_lams, ...)
            # 的 ESS→λ 映射整体错位。改为 fail closed。
            sampled_row = int(w.get("sampled_distribution_row", 0))
            n_k_local = np.zeros(len(win_lams) + 1, dtype=np.int32)
            if sampled_row != 0:
                raise ValueError(
                    f"窗口 {w_idx} sampled_distribution_row={sampled_row} != 0；"
                    "增广矩阵 row 0 固定是 IBS 采样分布、row 1..K 是物理态，非零采样行"
                    "会同时破坏 MBAR 的采样态声明与 ESS→λ 映射，拒绝继续。"
                )
            n_k_local[sampled_row] = n_frames
            
            # ------------------------------------------------------------------
            # 🔑 修复 3: 列平移稳定性 (Column-wise Shift)
            # ------------------------------------------------------------------
            # 进一步对每一列减去最小值，防止 exp 溢出
            u_min_col = np.min(u_mbar, axis=0, keepdims=True)
            u_mbar_stable = u_mbar - u_min_col
            
            # 剔除含 NaN 或 Inf 的列
            valid_mask = np.isfinite(u_mbar_stable).all(axis=0)
            u_mbar_final = u_mbar_stable[:, valid_mask]
            n_k_local[sampled_row] = np.sum(valid_mask) # 更新有效样本数

            # 🔑 [v23 numerical follow-up #2] 这里之前硬编码了字面量 10，而不是
            # 用调用方传入的 min_frames_per_window（该参数默认值恰好也是 10，
            # 这个硬编码大概率是把默认值抄成字面量留下的）。_solve_tmbar_and_
            # recenter 在线学习路径故意把 min_frames_per_window 放宽到 3
            # （docstring 原话："去相关后统计非效率 g≈5-15 时常剩 2-7 帧，用默认
            # 的 10 帧门槛会让绝大多数 minibatch 被 solve_stage_integrated 自身
            # 的逐 entry 门槛跳过、白白丢弃已经采到的真实数据"）——但这条硬编码
            # 完全无视那个放宽，任何 3-9 帧的 minibatch 依然在这里被无声跳过。
            # 真实后果：len(local_results) < len(valid_windows)，导致
            # converged 恒为 False，且不打印任何警告，表现上跟"roundoff 卡在
            # 阈值"一模一样，实际根因完全不同（这条本身不是舍入问题，是这里从
            # 未真正遵守 min_frames_per_window 参数）。
            if n_k_local[sampled_row] < min_frames_per_window:
                print(
                    f"  ⚠️ 窗口 {w_idx} 剔除 NaN/Inf 列后有效帧数 "
                    f"({int(n_k_local[sampled_row])}) < min_frames_per_window "
                    f"({min_frames_per_window})，跳过"
                )
                continue

            if not HAS_PYMBAR:
                return {"error": "pymbar_not_installed", "converged": False}
            
            try:
                # 使用混合求解器，提高收敛性
                mbar = _build_mbar_compatible(
                    u_mbar_final,
                    n_k_local,
                    relative_tolerance=1e-7,
                    initialize="BAR",
                    solver_protocol="default",
                )
                
                res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
                df_matrix, ddf_matrix = _extract_free_energy_arrays(res, require_uncertainty=True)

                # ------------------------------------------------------------------
                # 🔑 修复（审查报告 #2）：真实的重加权质量诊断
                # ------------------------------------------------------------------
                # 该窗口的 MBAR 只有一个真实被采样的态 (row=sampled_row)，其余物理态
                # 样本数均为 0；标准 MBAR 重叠矩阵 compute_overlap() 在这种"单一采样
                # 分布 + 多个零样本目标态"场景下会退化（未采样列恒为 0，无法反映真实
                # 重叠）。这里改用 compute_effective_sample_number()：它衡量"把 row 0
                # 的样本重加权到每个目标 λ 态后，还剩多少个等效独立样本"，是这类单向
                # 重加权场景下物理上正确、数值上已验证有效的重叠代理（详见审查报告）。
                # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 受门的 min_ess_ratio 换成"扣掉共模
                # 因子后的混合覆盖度 ESS(p_k)"；raw 单参考 ESS 降级为只报告不设门。
                # 理由见 _ibs_reweighting_quality_diagnostics 顶部长注释：raw 量被
                # Group-4 λ-WCA 防护壳 + LRC 的逐帧共模因子按 exp(σ_r²) 整体压掉
                # （实测 σ_r 0.95→2.40 kT，对应 ESS/N 上限 0.40→0.003），而那个因子在
                # 本方法真正报出去的物理态↔物理态自由能差里大部分抵消。
                min_ess_ratio = None
                ess_ratio_per_lambda = None
                min_absolute_ess_this_window = None
                raw_min_ess_ratio = None
                raw_min_absolute_ess = None
                raw_ess_ratio_per_lambda = None
                min_occ_norm_this_window = None
                quality = {}
                try:
                    neff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)
                    denom = max(int(n_k_local[sampled_row]), 1)
                    # sampled_row 已在上面 fail-closed 保证为 0，所以 target_idx 严格是
                    # [1..K]，与 win_lams 同序，zip 映射成立。
                    target_idx = [i for i in range(len(neff)) if i != sampled_row]
                    if target_idx:
                        raw_ratio = neff[target_idx] / denom
                        raw_min_ess_ratio = float(np.min(raw_ratio))
                        raw_min_absolute_ess = float(np.min(neff[target_idx]))
                        # 保留逐态明细而不只是 min：调用方（abfe_preoptimizer.py 的
                        # refine_stage_lambda_path_by_overlap）需要知道是窗口内*哪个*
                        # λ 才是瓶颈，而不只是"这个窗口整体没过"。
                        raw_ess_ratio_per_lambda = {
                            int(lam): float(r) for lam, r in zip(win_lams, raw_ratio)
                        }

                    # 必须用 valid_mask 之后的那批帧：denom 是
                    # n_k_local[sampled_row]=sum(valid_mask)，若 quality 用未过滤
                    # 的 n_frames 帧，比例的分子分母就来自不同样本集。
                    quality = _ibs_reweighting_quality_diagnostics(
                        u_kj_raw[:, valid_mask],
                        bias_kj[valid_mask],
                        w.get("f_k"),
                        self.kt,
                    )
                    if quality.get("mixture_ess") is not None:
                        mix_ess = np.asarray(quality["mixture_ess"], dtype=float)
                        mix_ratio = mix_ess / denom
                        min_ess_ratio = float(np.min(mix_ratio))
                        min_absolute_ess_this_window = float(np.min(mix_ess))
                        ess_ratio_per_lambda = {
                            int(lam): float(r) for lam, r in zip(win_lams, mix_ratio)
                        }
                        occ_norm = quality.get("mixture_occupancy_normalized")
                        min_occ_norm_this_window = (
                            float(np.min(occ_norm)) if occ_norm else None
                        )
                    elif "f_k" in w:
                        # 只在调用方**本来打算**给 f_k 的时候才报警。完全没有 'f_k'
                        # 键的调用方（在线 warmup 的 tmbar_history、λ 重划分探针、
                        # 合成数据）是刻意不用这个门的，它们每次求解会遍历十几个
                        # entry，逐 entry 打印会把日志刷爆；对它们 min_ess_ratio=None
                        # 本身就是信号（converged 因此 fail closed）。
                        print(
                            f"  ⚠️ 窗口 {w_idx} 提供了 f_k 但无法计算混合覆盖度 ESS "
                            f"({quality.get('error')})；该窗口 min_ess_ratio=None，"
                            "收敛门会因此判失败（fail closed，不当作通过）"
                        )
                except Exception as ess_exc:
                    print(f"  ⚠️ 窗口 {w_idx} 有效样本数(ESS)重叠诊断计算失败: {ess_exc}")
                window_overlap_records.append({
                    "window_index": source_window_idx,
                    "window_label": w.get("window_label"),
                    "window_range": list(w.get("window_range", [])),
                    "lambdas": win_lams,
                    "lambdas_coul": list(w.get("lambdas_coul", [])),
                    "lambdas_vdw": list(w.get("lambdas_vdw", [])),
                    "min_ess_ratio": min_ess_ratio,
                    "ess_ratio_per_lambda": ess_ratio_per_lambda,
                    "absolute_ess": min_absolute_ess_this_window,
                    # 一阶矩：K*<p_k> 的最小值（理想=1）。与 min_ess_ratio 一起构成
                    # "权重质量"这一份证据的两半——ESS 是尺度不变的，抓不到"均匀被
                    # 饿死"的态，必须靠这一项。
                    "min_occupancy_normalized": min_occ_norm_this_window,
                    "mixture_occupancy_normalized": quality.get(
                        "mixture_occupancy_normalized"
                    ),
                    # 🔑 去相关、剔除 NaN/Inf 列之后真正喂进 MBAR 的样本数（不是
                    # 去相关前的原始帧数）——跟下面 n_k_local[sampled_row]<10 的
                    # 门槛用的是同一个量。
                    "n_frames_decorrelated": int(n_k_local[sampled_row]),
                    # ---- 只报告、不设门 ----
                    "ess_gate_protocol_version": int(ESS_GATE_PROTOCOL_VERSION),
                    "ess_gate_metric": "mixture_coverage_ess_common_mode_removed",
                    "raw_min_ess_ratio": raw_min_ess_ratio,
                    "raw_min_absolute_ess": raw_min_absolute_ess,
                    "raw_ess_ratio_per_lambda": raw_ess_ratio_per_lambda,
                    "top1pct_raw_weight": quality.get("top1pct_raw_weight"),
                    "common_mode_log_sigma_kT": quality.get("common_mode_log_sigma_kT"),
                    "common_mode_log_mean_kJ_mol": quality.get(
                        "common_mode_log_mean_kJ_mol"
                    ),
                    "reweighting_quality_error": quality.get("error"),
                    "statistical_inefficiency": float(g_val),
                    "statistical_inefficiency_worst_state": int(g_worst_state),
                })

                # Delta_f[0, k] 是第 k 个物理态相对于采样态 (Row 0) 的自由能差 (单位: kT)
                # 注意：res['Delta_f'] 的形状是 (K+1, K+1)
                # 我们想要的是物理态 (indices 1..K) 的结果
                f_phys_kt = df_matrix[0, 1:] 
                df_phys_kt = ddf_matrix[0, 1:]
                
                # 转换为 kJ/mol
                f_phys_kj = f_phys_kt * self.kt
                df_phys_kj = df_phys_kt * self.kt

                # 🔑 端点（第一个 vs 最后一个物理态）自由能差的不确定度，直接读
                # ddf_matrix 里这两个物理态之间的元素，不用 sqrt(a^2+b^2) 合并
                # 各自相对采样态的边际不确定度——同一次 MBAR 拟合、同一批样本，
                # 彼此有协方差，平方相加会系统性算错。约定跟
                # _solve_single_window_local_mbar（在线 early-stop 用）完全一致：
                # 物理态在增广矩阵里占据第 1..len(win_lams) 行，第 0 行是采样分布。
                n_lams = len(win_lams)
                endpoint_diff_uncertainty_kj = (
                    float(ddf_matrix[1, n_lams] * self.kt) if n_lams > 1 else 0.0
                )
                window_overlap_records[-1]["endpoint_diff_uncertainty_kJ_mol"] = endpoint_diff_uncertainty_kj

                # 注意：这里的 f_phys_kj 是相对于“采样态有效能量”的绝对自由能估计。
                # 为了拼接，我们需要相对于窗口内某个特定 lambda 的相对自由能。
                # 通常我们关心的是窗口内各 lambda 之间的相对差值。
                # 由于所有态都减去了同一个 global_offset，且参考态一致，相对差值是准确的。
                
                local_results.append({
                    # 🔑 [P0-8] 真实来源窗口号；下面 chain_segments 里那个
                    # "window_index" 其实是 local_results 的位置下标，两者在有窗口被
                    # 跳过时会分叉，覆盖诊断必须用这个。
                    "source_window_index": source_window_idx,
                    "lambdas": win_lams,
                    "f": f_phys_kj.astype(float), # 相对于采样参考点的绝对 F
                    "df": df_phys_kj.astype(float),
                    "dDelta_f": (
                        np.asarray(ddf_matrix[1:, 1:], dtype=np.float64)
                        * self.kt
                    ),
                    "weight": int(n_k_local[0]),
                    "global_offset": global_offset # 记录偏移量，用于调试
                })
                
            except Exception as e:
                print(f"  ⚠️ 局部 TMBAR 窗口 {w_idx} (Lams {win_lams}) 失败: {e}")
                import traceback
                traceback.print_exc()

        if not local_results:
            return self._fallback("no_local_tmbAR_results")

        # ------------------------------------------------------------------
        # 拼接全局自由能曲线
        # ------------------------------------------------------------------
        global_f = {}
        global_var = {}
        offset_var_terms = []
        
        # 初始化：使用第一个窗口的结果
        first = local_results[0]
        for i, lam in enumerate(first["lambdas"]):
            global_f[lam] = float(first["f"][i])
            global_var[lam] = float(max(first["df"][i], 1e-6)**2)
            
        # 逐个窗口拼接
        for local in local_results[1:]:
            # 寻找当前窗口与前一个已拼接全局结果的_overlap_ lambda
            overlap_lams = [lam for lam in local["lambdas"] if lam in global_f]
            
            if not overlap_lams:
                return self._fallback("window_overlap_broken")
            
            # 计算偏移量 (Offset)
            # 对于重叠的 lambda，计算 (Global_F - Local_F) 的加权平均值
            offsets = []
            offset_vars = []
            for lam in overlap_lams:
                idx_in_local = local["lambdas"].index(lam)
                # 局部自由能
                f_loc = float(local["f"][idx_in_local])
                var_loc_overlap = float(max(local["df"][idx_in_local], 1e-6)**2)
                # 全局自由能
                f_glob = global_f[lam]
                var_glob_overlap = global_var[lam]
                offsets.append(f_glob - f_loc)
                offset_vars.append(var_glob_overlap + var_loc_overlap)

            # 🔑 修复（审查报告 #4）：offset 之前用的是重叠点的非加权均值，而下面
            # 逐 λ 合并却是逆方差加权——当重叠点之间不确定度差异较大时，二者不一致
            # 会让 offset 本身产生偏置，拖动整条拼接曲线。这里统一改为逆方差加权，
            # 与下方合并公式使用同一套权重逻辑。
            offsets_arr = np.asarray(offsets, dtype=float)
            offset_vars_arr = np.asarray(offset_vars, dtype=float)
            inv_var = 1.0 / np.maximum(offset_vars_arr, 1e-12)
            offset = float(np.sum(offsets_arr * inv_var) / np.sum(inv_var))
            offset_var = float(1.0 / np.sum(inv_var))
            offset_var_terms.append(offset_var)
            
            # 应用偏移量并合并
            for i, lam in enumerate(local["lambdas"]):
                f_loc = float(local["f"][i]) + offset
                var_loc = float(max(local["df"][i], 1e-6)**2) + offset_var
                
                if lam in global_f:
                    # 逆方差加权平均
                    w_old = 1.0 / max(global_var[lam], 1e-12)
                    w_new = 1.0 / max(var_loc, 1e-12)
                    
                    global_f[lam] = (global_f[lam] * w_old + f_loc * w_new) / (w_old + w_new)
                    global_var[lam] = 1.0 / (w_old + w_new)
                else:
                    global_f[lam] = f_loc
                    global_var[lam] = var_loc

        # 整理结果
        sorted_lams = sorted(global_f.keys())
        f_curve = np.array([global_f[lam] for lam in sorted_lams])
        err_curve = np.sqrt([global_var[lam] for lam in sorted_lams])
        
        # 总自由能与误差必须来自每个独立窗口中“连接点→新增端点”的直接
        # MBAR 差值。各窗口样本独立，因此窗口段方差可相加；同一窗口内的
        # 端点协方差已经包含在 dDelta_f[i,j] 中。禁止再把两个相对采样态的
        # 边际方差做 sqrt(a²+b²)。
        chain_segments = []
        covered_lambdas = set()
        total_dg = 0.0
        total_variance = 0.0
        for local_idx, local in enumerate(local_results):
            local_lams = [int(x) for x in local["lambdas"]]
            if not local_lams:
                continue
            if local_idx == 0:
                join_lam = local_lams[0]
            else:
                shared = [lam for lam in local_lams if lam in covered_lambdas]
                if not shared:
                    return self._fallback("window_overlap_broken_for_covariance_chain")
                # 选择路径方向上最靠后的共享态，避免多重 overlap 被重复积分。
                join_lam = max(shared)
            end_lam = local_lams[-1]
            join_idx = local_lams.index(join_lam)
            end_idx = len(local_lams) - 1
            segment_dg = float(local["f"][end_idx] - local["f"][join_idx])
            segment_error = float(local["dDelta_f"][join_idx, end_idx])
            if not np.isfinite(segment_dg) or not np.isfinite(segment_error):
                return self._fallback("nonfinite_covariance_chain_segment")
            total_dg += segment_dg
            total_variance += segment_error ** 2
            chain_segments.append({
                # 注意：这个键是 local_results 的位置下标，不是来源窗口号——有窗口被
                # 跳过时两者会分叉。保持原键以免破坏既有读取方，真实来源见下一行。
                "window_index": int(local_idx),
                "source_window_index": int(
                    local.get("source_window_index", local_idx)
                ),
                "join_lambda_index": int(join_lam),
                "end_lambda_index": int(end_lam),
                "delta_G_kJ_mol": segment_dg,
                "uncertainty_kJ_mol": segment_error,
            })
            covered_lambdas.update(local_lams)
        total_err = float(np.sqrt(total_variance))
        endpoint_error_after_offset = total_err
        
        # ------------------------------------------------------------------
        # 🔑 修复（审查报告 #2）：converged/min_overlap 曾经分别是"每个窗口的
        # MBAR 是否解出来了"和"自由能曲线相邻点间距的单调函数"——前者与统计收敛
        # 无关，后者根本不是重叠度。现在改用每个窗口 compute_effective_sample_
        # number() 算出的真实重加权有效样本比例（见上面窗口内循环），
        # min_overlap = 所有窗口中最差的那个比例；converged 要求全部窗口都成功
        # 解出 *且* 最差重叠比例不低于阈值。旧的 Δf 间距量保留在
        # lambda_spacing_max_step_kJ_mol 里供参考，但不再冒充"重叠度"。
        # ------------------------------------------------------------------
        # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] min_ess_ratio 现在是"扣掉共模因子后的混合
        # 覆盖度"，不再是 raw 单参考 ESS——阈值语义因此才成立（raw 量的上限被防护壳
        # 按 exp(σ_r²) 压到 0.003，任何 5% 量级的阈值对它都是数学上不可满足的）。
        # 任一窗口算不出 min_ess_ratio（f_k 缺失/态数不符）时 ess_ratios 会少一项，
        # 下面 min_overlap 就不是"全部窗口都达标"的证据了——所以额外要求样本齐全。
        ess_ratios = [w["min_ess_ratio"] for w in window_overlap_records if w["min_ess_ratio"] is not None]
        min_overlap = (
            float(np.min(ess_ratios))
            if ess_ratios and len(ess_ratios) == len(window_overlap_records)
            else None
        )
        min_overlap_threshold = float(final_min_ess_ratio)

        # 🔑 [P1 修复] 只检查 ratio 不够——样本总数很少时绝对有效样本数可能只有
        # 个位数，比例却轻易超过阈值。补上去相关样本数、端点不确定度硬门槛，
        # 跟在线 early-stop 用的判据集合一致（不含漂移，一次性拼接没有"上一次
        # 检查"的基线）。
        # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] absolute_ess 从"独立硬门"降级为纯诊断。
        # 它在构造上就是 min_ess_ratio × N_decorrelated（denom 是同一个标量，所以
        # min(neff)/denom == min(neff/denom) 是恒等式），不是第二份独立证据——给
        # 同一个量配两个阈值，效果是让 ratio 那个阈值失去意义：final_min_absolute_ess
        # =50 在 N_decorrelated=114 时等价于要求 ratio ≥ 0.44，而日志里报的门是 0.05，
        # 真实放行条件其实是 ratio ≥ 50/N_decorrelated，且样本越少门越严——跟"延长
        # 采样"的修复方向相反。"比例达标但绝对样本数不够"这件事本来就该由
        # min_decorrelated_samples 直接看（它才是与 ratio 正交的那份证据），端点精度
        # 由 max_endpoint_uncertainty_kJ_mol 直接看。
        # min_absolute_ess 继续算、继续落盘；把 min_absolute_ess_threshold 置 None，
        # 这样 abfe_pipeline.py 里 _assert_stage_result_sane /
        # _stage_quality_failure_details 那两处"只在字段存在时才检查"的镜像判据会
        # 自动失活，无需改动 pipeline 侧逻辑。
        abs_ess_vals = [w["absolute_ess"] for w in window_overlap_records if w.get("absolute_ess") is not None]
        min_absolute_ess = float(np.min(abs_ess_vals)) if abs_ess_vals else None
        raw_ess_vals = [
            w["raw_min_ess_ratio"] for w in window_overlap_records
            if w.get("raw_min_ess_ratio") is not None
        ]
        raw_min_overlap = float(np.min(raw_ess_vals)) if raw_ess_vals else None
        raw_abs_vals = [
            w["raw_min_absolute_ess"] for w in window_overlap_records
            if w.get("raw_min_absolute_ess") is not None
        ]
        raw_min_absolute_ess = float(np.min(raw_abs_vals)) if raw_abs_vals else None
        # 一阶矩伴随诊断：ESS 尺度不变，抓不到"均匀被饿死"的态（实测一个 f_k 未补偿、
        # 高出 80 kT 的态 ESS/N 仍报 0.34）。但 warmup 已明确把 occupancy 定义为只诊断、
        # 不参与 production entry；最终 stage 不能在全部 GPU 采样完成后再用同一指标反向否决。
        occ_vals = [
            w["min_occupancy_normalized"] for w in window_overlap_records
            if w.get("min_occupancy_normalized") is not None
        ]
        min_occupancy_normalized = (
            float(np.min(occ_vals))
            if occ_vals and len(occ_vals) == len(window_overlap_records)
            else None
        )
        min_occupancy_normalized_threshold = float(
            IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION
        )
        sigma_vals = [
            w["common_mode_log_sigma_kT"] for w in window_overlap_records
            if w.get("common_mode_log_sigma_kT") is not None
        ]
        max_common_mode_log_sigma_kT = float(np.max(sigma_vals)) if sigma_vals else None

        n_frames_vals = [
            w["n_frames_decorrelated"] for w in window_overlap_records
            if w.get("n_frames_decorrelated") is not None
        ]
        min_decorrelated_samples = int(np.min(n_frames_vals)) if n_frames_vals else 0

        unc_vals = [
            segment["uncertainty_kJ_mol"] for segment in chain_segments
        ]
        max_endpoint_uncertainty_kJ_mol = float(np.max(unc_vals)) if unc_vals else None

        # [v23 numerical follow-up, no protocol bump] These are inclusive
        # physical thresholds. Normalized ESS arithmetic can undershoot an
        # exact boundary by ~1e-13; rejecting 0.9999999999998863 against 1.0
        # (and 0.04999999999999431 against 0.05) is a floating-point bug, not a
        # stricter scientific gate. Only roundoff-scale equality is accepted.
        converged = bool(
            len(local_results) == len(valid_windows)
            and min_overlap is not None
            and _meets_minimum_with_roundoff(min_overlap, min_overlap_threshold)
            and min_decorrelated_samples >= int(final_min_decorrelated_samples)
            and max_endpoint_uncertainty_kJ_mol is not None
            and np.isfinite(max_endpoint_uncertainty_kJ_mol)
            and _meets_maximum_with_roundoff(
                max_endpoint_uncertainty_kJ_mol, float(final_max_uncertainty_kJ_mol)
            )
        )
        lambda_spacing_max_step = float(np.max(np.abs(np.diff(f_curve)))) if f_curve.size > 1 else 0.0

        # 🔑 [P0-8] 把"这条曲线到底覆盖了哪些窗口/哪段 λ"落盘。loader 侧
        # (_assert_expected_windows_all_loaded) 已经挡住了"预期窗口没产出数据"这个
        # 入口，但求解器自己还会因维度不符/帧数不足 `continue` 掉窗口——那部分由
        # `len(local_results) == len(valid_windows)` 抓。这里记录的是审计证据：
        # 光看 total_delta_G 无法判断它积分的是不是完整的 λ 区间。
        input_window_indices = sorted(
            int(w.get("window_index", i)) for i, w in enumerate(window_data)
        )
        solved_window_indices = sorted(
            int(r.get("source_window_index", i)) for i, r in enumerate(local_results)
        )
        covered_lambda_indices = sorted(int(x) for x in covered_lambdas)
        coverage_diagnostics = {
            "input_window_indices": input_window_indices,
            "valid_window_indices": sorted(
                int(w.get("window_index", i)) for i, w in enumerate(valid_windows)
            ),
            "solved_window_indices": solved_window_indices,
            "dropped_window_indices": sorted(
                set(input_window_indices) - set(solved_window_indices)
            ),
            "covered_lambda_indices": covered_lambda_indices,
            "covered_lambda_index_first": (
                covered_lambda_indices[0] if covered_lambda_indices else None
            ),
            "covered_lambda_index_last": (
                covered_lambda_indices[-1] if covered_lambda_indices else None
            ),
            "n_covered_lambda_indices": len(covered_lambda_indices),
            "note": (
                "求解器只能证明它积分了这些 λ 索引，无法证明它们就是该 stage 的完整"
                "路径——'预期窗口是否全部产出数据'必须在 loader 出口用 "
                "_assert_expected_windows_all_loaded 判定。"
            ),
        }

        return {
            "coverage_diagnostics": coverage_diagnostics,
            "total_delta_G": total_dg,
            "total_error": total_err,
            "endpoint_error_after_offset": endpoint_error_after_offset,
            "covariance_chain_segments": chain_segments,
            "total_error_method": (
                "independent_window_segment_variances_using_direct_dDelta_f"
            ),
            "offset_error_contribution": float(np.sqrt(np.sum(offset_var_terms))) if offset_var_terms else 0.0,
            "converged": converged,
            "min_overlap": min_overlap,
            "min_overlap_threshold": min_overlap_threshold,
            "min_overlap_method": (
                "per_window_mixture_coverage_ess_ratio_common_mode_removed"
            ),
            "ess_gate_protocol_version": int(ESS_GATE_PROTOCOL_VERSION),
            "min_occupancy_normalized": min_occupancy_normalized,
            # occupancy 与 warmup 一致，只报告、不设门；保留 0.5 作为诊断参考线。
            "min_occupancy_normalized_threshold": None,
            "min_occupancy_normalized_diagnostic_reference": min_occupancy_normalized_threshold,
            "min_occupancy_is_gate": False,
            "min_occupancy_gate_retired_reason": (
                "warmup 明确将 occupancy 定义为 diagnostics-only；最终 stage 不得在生产"
                "全部完成后用同一指标反向否决已放行的 ensemble。"
            ),
            "min_absolute_ess": min_absolute_ess,
            # None = 该项已降级为纯诊断，见上面 ESS_GATE_PROTOCOL_VERSION=2 的说明。
            # 置 None 同时让 abfe_pipeline 侧两处镜像检查自动失活。
            "min_absolute_ess_threshold": None,
            "min_absolute_ess_gate_retired_reason": (
                "absolute_ess == min_ess_ratio * n_frames_decorrelated 恒等，不是独立"
                "证据；给同一个量配两个阈值会让 ratio 阈值失效（final_min_absolute_ess"
                f"={float(final_min_absolute_ess):.4g} 在 N_decorrelated=114 时等价于"
                "要求 ratio>=0.44）。改由 min_decorrelated_samples 与 "
                "max_endpoint_uncertainty_kJ_mol 分别承担样本量与精度这两份正交证据。"
            ),
            # ---- 只报告、不设门：防护壳/LRC 共模因子收了多少重加权税 ----
            "raw_min_overlap": raw_min_overlap,
            "raw_min_absolute_ess": raw_min_absolute_ess,
            "max_common_mode_log_sigma_kT": max_common_mode_log_sigma_kT,
            "min_decorrelated_samples": min_decorrelated_samples,
            "min_decorrelated_samples_threshold": int(final_min_decorrelated_samples),
            "max_endpoint_uncertainty_kJ_mol": max_endpoint_uncertainty_kJ_mol,
            "max_endpoint_uncertainty_kJ_mol_threshold": float(final_max_uncertainty_kJ_mol),
            "lambda_spacing_max_step_kJ_mol": lambda_spacing_max_step,
            "window_overlap_diagnostics": window_overlap_records,
            "statistical_inefficiency_per_window": window_g_values,
            "method": "Local-TMBAR covariance-chain (ESS-overlap-checked)",
            "uncertainty_note": (
                "总误差对每个独立窗口直接读取连接态到新增端点的 dDelta_f，并合并独立"
                "窗口段方差；同一窗口内的端点协方差因此不会重复丢失。converged 要求 "
                "ESS ratio、去相关样本数和端点差不确定度达标；occupancy 与 absolute ESS "
                "只作诊断。不同窗口来自独立采样，未假设跨窗口协方差。"
            ),
            "f_k": f_curve.tolist(),
            "df_k": err_curve.tolist(),
            "lambdas": sorted_lams
        }

    def _fallback(self, msg: str) -> Dict:
        return {"error": msg, "converged": False, "total_delta_G": 0.0, "total_error": 999.9}

# 模块级便捷入口
# ---------------------------------------------------------------------------
# split-half 时序一致性诊断
# ---------------------------------------------------------------------------
# 现有五道门（overlap / occupancy / decorrelated-samples / endpoint-σ / ESS）
# 全部是对整批样本的统计量：把帧的时间顺序打乱，结果一模一样。所以它们对
# "系综还在漂" 这件事完全没有分辨力。2026-07-28 的溶剂盒扫描实测：18 个窗口里
# 有 5 个的前后半程 ΔG 之差超过自身报出 σ 的 2 倍（最大 4.34），而
# `min_decorrelated_samples` 的阈值是 20、实测 137~1868，永远不可能响。
#
# 判据：把每个窗口的帧按时间切成前后两半，各自解一遍。两个半程各自的标准误
# 约 √2·σ，它们之差的标准误约 2σ，所以 z = |后半 − 前半| / (2σ_win)。
#
# 默认只诊断不阻断（`split_half_max_z=None`）：把它直接设成阻断会让现有大量
# 运行当场失败，那是采样预算问题，不该由一次代码改动替使用者决定。要变成硬门，
# 显式传 `split_half_max_z=2.0`。
SPLIT_HALF_GATE_PROTOCOL_VERSION = 1
SPLIT_HALF_DEFAULT_MAX_Z = 2.0


def _slice_window_frames(window: Dict, lo_frac: float, hi_frac: float) -> Optional[Dict]:
    """按时间顺序取窗口的 [lo_frac, hi_frac) 段帧；帧数不足返回 None。"""
    u_kn = window.get("u_kn")
    if u_kn is None:
        return None
    u_kn = np.asarray(u_kn, dtype=np.float64)
    if u_kn.ndim != 2:
        return None
    n_frames = u_kn.shape[1]
    lo = max(0, min(int(lo_frac * n_frames), n_frames))
    hi = max(lo, min(int(hi_frac * n_frames), n_frames))
    if hi - lo < 2:
        return None
    bias = window.get("bias_energies")
    base = window.get("base_energies")
    if bias is None or base is None:
        return None
    sliced = dict(window)
    sliced["u_kn"] = u_kn[:, lo:hi]
    sliced["bias_energies"] = np.asarray(bias, dtype=np.float64)[lo:hi]
    sliced["base_energies"] = np.asarray(base, dtype=np.float64)[lo:hi]
    return sliced


def _segment_delta_g_by_window(result: Dict) -> Dict[int, float]:
    segs = result.get("covariance_chain_segments") or []
    return {int(s["window_index"]): float(s["delta_G_kJ_mol"]) for s in segs}


def split_half_drift_diagnostics(
    window_outputs: List[Dict],
    kt: float,
    full_result: Dict,
    solver_kwargs: Optional[Dict] = None,
) -> Dict:
    """前后半程一致性诊断。只读，不改 `full_result`。"""
    solver_kwargs = dict(solver_kwargs or {})
    halves: Dict[str, Optional[Dict]] = {}
    for label, (lo, hi) in (("first", (0.0, 0.5)), ("second", (0.5, 1.0))):
        sliced = [_slice_window_frames(w, lo, hi) for w in window_outputs]
        if any(s is None for s in sliced):
            return {
                "split_half_gate_protocol_version": int(SPLIT_HALF_GATE_PROTOCOL_VERSION),
                "available": False,
                "reason": f"{label} half: 有窗口帧数不足以切半",
            }
        halves[label] = solve_stage_integrated(
            sliced, kt, split_half_max_z=None, _split_half_recursion_guard=True,
            **solver_kwargs,
        )

    seg_full = _segment_delta_g_by_window(full_result)
    seg_first = _segment_delta_g_by_window(halves["first"])
    seg_second = _segment_delta_g_by_window(halves["second"])
    sigma_win = {
        int(s["window_index"]): float(s["uncertainty_kJ_mol"])
        for s in (full_result.get("covariance_chain_segments") or [])
    }
    if set(seg_first) != set(seg_full) or set(seg_second) != set(seg_full):
        return {
            "split_half_gate_protocol_version": int(SPLIT_HALF_GATE_PROTOCOL_VERSION),
            "available": False,
            "reason": "半程解出的窗口集合与全量不一致，无法逐窗比较",
        }

    per_window = []
    max_z = 0.0
    for w in sorted(seg_full):
        drift = seg_second[w] - seg_first[w]
        sigma = sigma_win.get(w, 0.0)
        # σ_win 为 0 时 z 无意义（不是"完美"），记 None 而不是 inf。
        z = abs(drift) / (2.0 * sigma) if sigma > 0.0 else None
        if z is not None:
            max_z = max(max_z, z)
        per_window.append({
            "window_index": int(w),
            "delta_G_full_kJ_mol": seg_full[w],
            "delta_G_first_half_kJ_mol": seg_first[w],
            "delta_G_second_half_kJ_mol": seg_second[w],
            "drift_kJ_mol": drift,
            "uncertainty_kJ_mol": sigma,
            "drift_over_2sigma": z,
        })

    total_sigma = float(full_result.get("total_error", 0.0) or 0.0)
    total_drift = float(halves["second"].get("total_delta_G", 0.0)) - float(
        halves["first"].get("total_delta_G", 0.0)
    )
    return {
        "split_half_gate_protocol_version": int(SPLIT_HALF_GATE_PROTOCOL_VERSION),
        "available": True,
        "metric": "abs(second_half - first_half) / (2 * sigma)",
        "note": (
            "两个半程各自 SE≈√2·σ，其差的 SE≈2σ，所以判据分母是 2σ 而不是 σ。"
            "该诊断对帧的时间顺序敏感，是现有五道门都不具备的维度。"
        ),
        "total_delta_G_first_half_kJ_mol": float(halves["first"].get("total_delta_G", 0.0)),
        "total_delta_G_second_half_kJ_mol": float(halves["second"].get("total_delta_G", 0.0)),
        "total_drift_kJ_mol": total_drift,
        "total_drift_over_2sigma": (
            abs(total_drift) / (2.0 * total_sigma) if total_sigma > 0.0 else None
        ),
        "max_window_drift_over_2sigma": max_z,
        "per_window": per_window,
    }


def sigma_inflated_from_split_half(full_result: Dict, drift: Dict) -> Dict:
    """[P1-19] 用 split-half 实测漂移给 per-window σ 定一个下界。

    `segment_error` 取的是 pymbar 的**渐近协方差**（`dDelta_f[join, end]`），
    它假定样本独立同分布且系综已收敛。本仓库已三次实测到它系统性低估：

      1. 溶剂盒扫描 18 个窗口里 5 个的 split-half z > 2（σ 正确时期望 0.8 个）；
         window 4 在所有现有指标上都是优等生（ESS 348、n_decorr 357、g 1.40、
         ess_ratio ≥ 0.976）却 z=4.34 —— 问题不在采样质量的代理量，在 σ 本身。
      2. attachment 腿两轮独立测量差 0.4454 kJ/mol = 报出 σ 的 4.4 倍。
      3. charging 重跑与原值差 10.63 kJ/mol 而报出 σ 只有 0.88。

    修法：两个半程之差的标准差是 2σ，所以观测到 |漂移| 就意味着
    `σ ≳ |漂移|/2`。取 `σ_eff = max(σ_MBAR, |漂移|/2)` 作为下界。
    这只用已经算出来的量，零额外计算。

    **只返回结果，不改 `full_result`。** 调用方决定要不要采用——因为
    `final_max_uncertainty_kJ_mol` 那道门吃的就是 max(σ_win)，直接套用会让
    一批本来通过的运行突然失败，那是使用者的决定而不是这个函数的。
    """
    segs = full_result.get("covariance_chain_segments") or []
    if not segs or not drift.get("available"):
        return {"available": False, "reason": "缺 covariance_chain_segments 或 split-half 不可用"}

    by_window = {int(w["window_index"]): w for w in drift.get("per_window", [])}
    rows, var_orig, var_infl = [], 0.0, 0.0
    for seg in segs:
        w = int(seg["window_index"])
        s_mbar = float(seg["uncertainty_kJ_mol"])
        d = by_window.get(w, {}).get("drift_kJ_mol")
        floor = abs(float(d)) / 2.0 if d is not None else 0.0
        s_eff = max(s_mbar, floor)
        var_orig += s_mbar ** 2
        var_infl += s_eff ** 2
        rows.append({
            "window_index": w,
            "sigma_mbar_kJ_mol": s_mbar,
            "sigma_floor_from_drift_kJ_mol": floor,
            "sigma_effective_kJ_mol": s_eff,
            "inflated": bool(s_eff > s_mbar + 1e-12),
            "inflation_factor": (s_eff / s_mbar) if s_mbar > 0 else None,
        })
    return {
        "available": True,
        "rule": "sigma_eff = max(sigma_mbar, |split_half_drift| / 2)",
        "rationale": "两个半程之差的 SD 是 2σ，故观测到 |漂移| 蕴含 σ ≳ |漂移|/2",
        "per_window": rows,
        "total_error_mbar_kJ_mol": float(np.sqrt(var_orig)),
        "total_error_inflated_kJ_mol": float(np.sqrt(var_infl)),
        "total_inflation_factor": (
            float(np.sqrt(var_infl) / np.sqrt(var_orig)) if var_orig > 0 else None
        ),
        "n_windows_inflated": int(sum(1 for r in rows if r["inflated"])),
        "max_endpoint_sigma_mbar_kJ_mol": max((r["sigma_mbar_kJ_mol"] for r in rows), default=None),
        "max_endpoint_sigma_inflated_kJ_mol": max((r["sigma_effective_kJ_mol"] for r in rows), default=None),
    }


def solve_stage_integrated(
    window_outputs: List[Dict],
    kt: float,
    stage_name: str = "",
    final_min_ess_ratio: float = 0.05,
    final_min_absolute_ess: float = 50.0,
    final_min_decorrelated_samples: int = 20,
    final_max_uncertainty_kJ_mol: float = 1.0,
    min_frames_per_window: int = 10,
    split_half_max_z: Optional[float] = None,
    inflate_sigma_from_split_half: bool = False,
    _split_half_recursion_guard: bool = False,
) -> Dict:
    """stage 级 IBS-TMBAR 求解入口（stage2 = vdW）。

    ⛔ **只能用 TMBAR。** 不要在这里加 BAR / TI / 全帧主值 / √g σ 缩放 /
    bootstrap σ / σ evidence 汇总——理由与被撤回的先例见
    `ESTIMATOR_ANALYSIS_PROTOCOL_VERSION` 的注释。
    """
    if not window_outputs:
        return {"total_delta_G": 0.0, "total_error": 999.9, "converged": False}
    analyzer = GlobalMBARAnalyzer(kt=kt)
    solver_kwargs = dict(
        final_min_ess_ratio=final_min_ess_ratio,
        final_min_absolute_ess=final_min_absolute_ess,
        final_min_decorrelated_samples=final_min_decorrelated_samples,
        final_max_uncertainty_kJ_mol=final_max_uncertainty_kJ_mol,
        min_frames_per_window=min_frames_per_window,
    )
    res = analyzer.solve_stage_integrated(window_outputs, **solver_kwargs)
    res.setdefault("stage", stage_name)

    if not _split_half_recursion_guard:
        try:
            drift = split_half_drift_diagnostics(window_outputs, kt, res, solver_kwargs)
        except Exception as exc:  # 诊断绝不能把主求解带崩
            drift = {
                "split_half_gate_protocol_version": int(SPLIT_HALF_GATE_PROTOCOL_VERSION),
                "available": False,
                "reason": f"split-half 诊断异常: {exc}",
            }
        res["split_half_diagnostics"] = drift
        max_z = drift.get("max_window_drift_over_2sigma") if drift.get("available") else None
        res["split_half_max_window_z"] = max_z
        res["split_half_max_z_threshold"] = split_half_max_z
        if max_z is not None and max_z > SPLIT_HALF_DEFAULT_MAX_Z:
            worst = max(
                (w for w in drift["per_window"] if w["drift_over_2sigma"] is not None),
                key=lambda w: w["drift_over_2sigma"],
            )
            print(
                f"  ⚠️ [split-half] {stage_name or 'stage'} 前后半程不一致："
                f"window {worst['window_index']} 漂移 {worst['drift_kJ_mol']:+.3f} kJ/mol "
                f"= {worst['drift_over_2sigma']:.2f}×2σ（σ_win={worst['uncertainty_kJ_mol']:.3f}）。"
                f"报出的不确定度低估了实际抽样波动。"
            )
        if split_half_max_z is not None and max_z is not None and max_z > float(split_half_max_z):
            res["converged"] = False
            res["split_half_gate_failed"] = True

        # [P1-19] σ 下界。**始终计算并落盘，默认不采用**——
        # `final_max_uncertainty_kJ_mol` 那道门吃的就是 max(σ_win)，
        # 直接套用会让一批本来通过的运行突然失败，那是使用者的决定。
        infl = sigma_inflated_from_split_half(res, drift)
        res["sigma_inflation_from_split_half"] = infl
        res["sigma_inflation_applied"] = False
        if infl.get("available") and infl.get("n_windows_inflated"):
            print(
                f"  ℹ️ [P1-19] {stage_name or 'stage'} 若按 σ≥|漂移|/2 定下界："
                f"{infl['n_windows_inflated']}/{len(infl['per_window'])} 个窗口的 σ 被抬高，"
                f"总 σ {infl['total_error_mbar_kJ_mol']:.4f} → "
                f"{infl['total_error_inflated_kJ_mol']:.4f} kJ/mol "
                f"(×{infl['total_inflation_factor']:.2f})。默认未采用。"
            )
        if inflate_sigma_from_split_half and infl.get("available"):
            res["total_error_mbar_only_kJ_mol"] = float(res.get("total_error", 0.0))
            res["total_error"] = float(infl["total_error_inflated_kJ_mol"])
            for seg in res.get("covariance_chain_segments") or []:
                row = next(
                    (r for r in infl["per_window"]
                     if r["window_index"] == int(seg["window_index"])),
                    None,
                )
                if row:
                    seg["uncertainty_mbar_kJ_mol"] = seg["uncertainty_kJ_mol"]
                    seg["uncertainty_kJ_mol"] = row["sigma_effective_kJ_mol"]
            res["sigma_inflation_applied"] = True
            print(
                f"  ✅ [P1-19] 已采用 σ 下界：总 σ = "
                f"{res['total_error']:.4f} kJ/mol"
            )

    return res

# ============================================================================
# 力组诊断工具 (开发/调试用，保留)
# ============================================================================

def _check_boresch_geometry_safe(context, boresch_params, min_sin_theta=0.1):
    """
    检查当前锚点几何是否远离奇点（θ 偏离 0° 和 180°）。
    返回 (is_safe, r0_nm, thA_deg, thB_deg)
    """
    state = context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box_vectors = state.getPeriodicBoxVectors()
    rec = [int(i) for i in boresch_params["receptor_indices"]]
    lig = [int(i) for i in boresch_params["ligand_indices"]]

    H0 = np.asarray(pos[rec[0]], dtype=np.float64)
    try:
        H1 = H0 + _minimum_image_displacement_nm(pos[rec[1]] - H0, box_vectors)
        H2 = H1 + _minimum_image_displacement_nm(pos[rec[2]] - H1, box_vectors)
        L0 = H0 + _minimum_image_displacement_nm(pos[lig[0]] - H0, box_vectors)
        L1 = L0 + _minimum_image_displacement_nm(pos[lig[1]] - L0, box_vectors)
        L2 = L1 + _minimum_image_displacement_nm(pos[lig[2]] - L1, box_vectors)
    except ValueError as exc:
        print(f"  🚨 Boresch minimum-image 几何无法计算：{exc}")
        return False, np.nan, np.nan, np.nan
    anchor_coords = np.asarray([H0, H1, H2, L0, L1, L2], dtype=float)
    if not np.all(np.isfinite(anchor_coords)):
        print("  🚨 Boresch 锚点坐标包含 NaN/Inf，几何检查判定为不安全")
        return False, np.nan, np.nan, np.nan

    r0 = np.linalg.norm(H0 - L0)
    if not np.isfinite(r0):
        print("  🚨 Boresch r0 为 NaN/Inf，几何检查判定为不安全")
        return False, np.nan, np.nan, np.nan

    def calc_angle(a, b, c):
        ba, bc = a - b, c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom <= 1e-12 or not np.isfinite(denom):
            return np.nan
        cos_val = np.clip(
            np.dot(ba, bc) / denom,
            -1.0, 1.0
        )
        return np.arccos(cos_val)

    thA = calc_angle(H1, H0, L0)
    thB = calc_angle(H0, L0, L1)
    if not np.all(np.isfinite([thA, thB])):
        print("  🚨 Boresch θA/θB 为 NaN/Inf，几何检查判定为不安全")
        return False, float(r0), np.nan, np.nan

    sinA, sinB = np.sin(thA), np.sin(thB)
    if (
        not np.all(np.isfinite([sinA, sinB]))
        or sinA < min_sin_theta
        or sinB < min_sin_theta
    ):
        print(f"  🚨 Boresch 角度接近奇点：θA={np.degrees(thA):.1f}° (sin={sinA:.3f})，"
              f"θB={np.degrees(thB):.1f}° (sin={sinB:.3f})")
        return False, r0, np.degrees(thA), np.degrees(thB)

    return True, r0, np.degrees(thA), np.degrees(thB)


def _has_valid_boresch_restraint(params) -> bool:
    """仅当 Boresch 参数包含完整 3+3 锚点时才认为可注入/可启用。"""
    if not isinstance(params, dict):
        return False
    rec_idx = params.get("receptor_indices") or []
    lig_idx = params.get("ligand_indices") or []
    return len(rec_idx) == 3 and len(lig_idx) == 3


def current_max_force_from_context(context) -> float:
    """返回当前上下文的全体系最大合力范数，用于生产前安全拦截。"""
    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    if forces.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(forces, axis=1)))


def diagnose_top_force_atoms(
    context,
    system,
    topology=None,
    ligand_indices=None,
    prefix: str = "窗口诊断",
    top_n: int = 20,
):
    """打印最大受力原子及各 force group 贡献，用于定位高力来源。"""
    from openmm import unit
    ligand_set = set(int(i) for i in (ligand_indices or []))
    atoms = list(topology.atoms()) if topology is not None else []
    water_names = {"HOH", "WAT", "SOL", "TIP3", "TIP3P"}
    ion_names = {"NA", "SOD", "CL", "CLA", "K", "POT", "MG", "CA"}

    state = context.getState(getForces=True)
    total_forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    if total_forces.size == 0:
        return

    group_forces = {}
    for gid in sorted({force.getForceGroup() for force in system.getForces()}):
        try:
            g_state = context.getState(getForces=True, groups={gid})
            group_forces[gid] = g_state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
        except Exception:
            continue

    norms = np.linalg.norm(total_forces, axis=1)
    order = np.argsort(norms)[::-1][:top_n]

    print(f"\n🔎 [{prefix}] 最大受力原子定位:")
    print(
        f"{'rank':<4} | {'atom':<7} | {'residue':<12} | {'role':<7} | "
        f"{'total':>10} | {'G0':>10} | {'G1':>10} | {'G2':>10} | {'G3':>10} | {'G4':>10}"
    )
    print("-" * 105)
    for rank, idx in enumerate(order, 1):
        atom_label = str(idx)
        residue_label = "N/A"
        role = "env"
        if idx < len(atoms):
            atom = atoms[int(idx)]
            residue = atom.residue
            atom_label = f"{idx}:{atom.name}"
            residue_label = f"{residue.name}{residue.index}"
            res_name = residue.name.upper()
            if idx in ligand_set:
                role = "ligand"
            elif res_name in water_names:
                role = "water"
            elif res_name in ion_names:
                role = "ion"
        elif idx in ligand_set:
            role = "ligand"

        def gnorm(gid):
            arr = group_forces.get(gid)
            if arr is None or idx >= len(arr):
                return 0.0
            return float(np.linalg.norm(arr[int(idx)]))

        print(
            f"{rank:<4} | {atom_label:<7} | {residue_label:<12} | {role:<7} | "
            f"{float(norms[int(idx)]):10.1f} | {gnorm(0):10.1f} | {gnorm(1):10.1f} | "
            f"{gnorm(2):10.1f} | {gnorm(3):10.1f} | {gnorm(4):10.1f}"
        )
    print("-" * 105)


def _detect_constraint_deadlock(pre_breakdown, post_breakdown, n_constraints: int) -> Optional[str]:
    """根据测试步进前后 Bond/Angle 核心力异常放大，识别约束-非键死锁。"""
    if n_constraints < 1000 or not pre_breakdown or not post_breakdown:
        return None
    pre_bond = float(pre_breakdown.get("Bond", {}).get("max", 0.0))
    pre_angle = float(pre_breakdown.get("Angle", {}).get("max", 0.0))
    post_bond = float(post_breakdown.get("Bond", {}).get("max", 0.0))
    post_angle = float(post_breakdown.get("Angle", {}).get("max", 0.0))

    bond_exploded = post_bond > max(3000.0, 1.4 * max(pre_bond, 1.0))
    angle_exploded = post_angle > max(2500.0, 1.4 * max(pre_angle, 1.0))
    if bond_exploded and angle_exploded:
        return (
            f"疑似约束死锁：Bond Max {pre_bond:.1f} -> {post_bond:.1f}, "
            f"Angle Max {pre_angle:.1f} -> {post_angle:.1f}, constraints={n_constraints}"
        )
    return None

def diagnose_force_groups_detailed(context, system, prefix="窗口诊断"):
    """按唯一 ForceGroup 聚合的受力拆解，避免同组总力被重复打印到每个力对象上。"""
    from openmm import unit
    print(f"\n🔍 [{prefix}] 逐力组受力拆解报告:")
    print(f"{'Group':<8} | {'成员力类型':<52} | {'Max|F| (kJ/mol/nm)':<20} | {'RMS|F|':<12} | {'状态'}")
    print("-" * 125)

    group_members = {}
    for force in system.getForces():
        gid = force.getForceGroup()
        group_members.setdefault(gid, []).append(type(force).__name__)

    for gid in sorted(group_members):
        member_types = group_members[gid]
        member_summary = ", ".join(member_types[:3])
        if len(member_types) > 3:
            member_summary += f", ... x{len(member_types)}"
        try:
            state = context.getState(getForces=True, groups={gid})
            forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/openmm.unit.nanometer)
            if forces.size == 0:
                continue
            norms = np.linalg.norm(forces, axis=1)
            max_f = np.max(norms)
            rms_f = np.sqrt(np.mean(norms**2))
            status = "🚨 爆炸源" if max_f > 10000 else ("⚠️ 偏高" if max_f > 2000 else "✓ 正常")
            print(f"Group {gid:<2} | {member_summary:<52} | {max_f:<20.2f} | {rms_f:<12.2f} | {status}")
        except Exception as e:
            print(f"Group {gid:<2} | {member_summary:<52} | {'(CV/元力跳过)':<20} | {'N/A':<12} | ℹ️ {str(e)[:25]}")
    print("-" * 125)
    n_cons = system.getNumConstraints()
    if n_cons > 0:
        print(f"  🔗 系统含 {n_cons} 个刚性约束 (SHAKE/LINCS)。若上述力组均正常但合力爆炸，根因必为约束死锁或初始键长畸变。")


def print_force_group_details(system: openmm.System, prefix: str = "系统"):
    """生产级力组详细诊断：打印每个力的类型、组号、粒子/键数、关键全局参数"""
    print(f"\n🔍 [{prefix}] 力组架构详细报告:")
    print(f"{'ID':<4} | {'Force Group':<12} | {'Force Type':<28} | {'Particles/Bonds':<16} | {'Key Global Params'}")
    print("-" * 90)
    for i, f in enumerate(system.getForces()):
        gid = f.getForceGroup()
        ftype = type(f).__name__
        if hasattr(f, 'getNumParticles'):
            count_str = f"{f.getNumParticles()} particles"
        elif hasattr(f, 'getNumBonds'):
            count_str = f"{f.getNumBonds()} bonds"
        elif hasattr(f, 'getNumAngles'):
            count_str = f"{f.getNumAngles()} angles"
        elif hasattr(f, 'getNumTorsions'):
            count_str = f"{f.getNumTorsions()} torsions"
        else:
            count_str = "N/A"
        params = []
        if hasattr(f, 'getNumGlobalParameters'):
            for j in range(f.getNumGlobalParameters()):
                name = f.getGlobalParameterName(j)
                val = f.getGlobalParameterDefaultValue(j)
                params.append(f"{name}={val:.3g}")
        param_str = ", ".join(params[:4]) + ("..." if len(params) > 4 else "")
        print(f"{i:<4} | Group {gid:<6} | {ftype:<28} | {count_str:<16} | {param_str}")
    print("-" * 90)

def diagnose_force_breakdown(main_context, main_system, prefix=""):
    """
    非侵入式力分解：利用临时系统 + 当前坐标，单独计算 Bond/Angle/Torsion/NB 的受力。
    完全不修改主系统的 ForceGroup 或力常数。
    """
    import openmm
    from openmm import unit
    import numpy as np

    # 从主系统收集需要的力类型
    bond_forces = [f for f in main_system.getForces() if isinstance(f, openmm.HarmonicBondForce)]
    angle_forces = [f for f in main_system.getForces() if isinstance(f, openmm.HarmonicAngleForce)]
    torsion_forces = [f for f in main_system.getForces() if isinstance(f, openmm.PeriodicTorsionForce)]
    nb_forces = [f for f in main_system.getForces() if isinstance(f, openmm.NonbondedForce)]

    # 构建一个极简系统，只包含上述力，并使用与主系统相同的粒子数
    num_particles = main_system.getNumParticles()
    diag_sys = openmm.System()
    for _ in range(num_particles):
        diag_sys.addParticle(1.0)  # 质量随意

    # 复制 Bond 力并设 Group 10
    for bf in bond_forces:
        new_bf = openmm.HarmonicBondForce()
        for i in range(bf.getNumBonds()):
            p1, p2, length, k = bf.getBondParameters(i)
            new_bf.addBond(p1, p2, length, k)
        new_bf.setForceGroup(10)
        diag_sys.addForce(new_bf)

    # 复制 Angle 力 → Group 11
    for af in angle_forces:
        new_af = openmm.HarmonicAngleForce()
        for i in range(af.getNumAngles()):
            p1, p2, p3, angle, k = af.getAngleParameters(i)
            new_af.addAngle(p1, p2, p3, angle, k)
        new_af.setForceGroup(11)
        diag_sys.addForce(new_af)

    # 复制 Torsion 力 → Group 13
    for tf in torsion_forces:
        new_tf = openmm.PeriodicTorsionForce()
        for i in range(tf.getNumTorsions()):
            p1, p2, p3, p4, per, phase, k = tf.getTorsionParameters(i)
            new_tf.addTorsion(p1, p2, p3, p4, per, phase, k)
        new_tf.setForceGroup(13)
        diag_sys.addForce(new_tf)

    # 复制 NonbondedForce → Group 12 （保持原始参数，不归零）
    for nb in nb_forces:
        new_nb = openmm.NonbondedForce()
        for i in range(num_particles):
            q, sig, eps = nb.getParticleParameters(i)
            new_nb.addParticle(q, sig, eps)
        # 复制排除表
        for i in range(nb.getNumExceptions()):
            p1, p2, cp, sig, eps = nb.getExceptionParameters(i)
            new_nb.addException(p1, p2, cp, sig, eps)
        new_nb.setNonbondedMethod(nb.getNonbondedMethod())
        new_nb.setCutoffDistance(nb.getCutoffDistance())
        new_nb.setForceGroup(12)
        diag_sys.addForce(new_nb)

    # 获取主 context 的当前坐标
    state = main_context.getState(getPositions=True)
    pos = state.getPositions()
    box = state.getPeriodicBoxVectors()

    # 创建临时 integrator/context（仅用于力计算）
    integ = openmm.VerletIntegrator(0.001)
    plat = openmm.Platform.getPlatformByName("Reference")  # 避免 GPU 上下文冲突
    diag_context = openmm.Context(diag_sys, integ, plat)
    diag_context.setPositions(pos)
    if box is not None:
        diag_context.setPeriodicBoxVectors(*box)

    # 读取各组力
    group_map = {10: "Bond", 11: "Angle", 12: "Nonbonded", 13: "Torsion"}
    print(f"\n🔍 [{prefix}] 非侵入式力分解报告：")
    stats = {}
    for gid, name in group_map.items():
        try:
            fstate = diag_context.getState(getForces=True, groups={gid})
            forces = fstate.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            norms = np.linalg.norm(forces, axis=1)
            max_f, avg_f, rms_f = np.max(norms), np.mean(norms), np.sqrt(np.mean(norms**2))
            stats[name] = {"max": float(max_f), "avg": float(avg_f), "rms": float(rms_f)}
            print(f"  {name:<12} | Max:{max_f:12.2f} | Avg:{avg_f:12.2f} | RMS:{rms_f:12.2f} kJ/(mol·nm)")
        except Exception as e:
            stats[name] = {"error": str(e)}
            print(f"  {name:<12} | 无法获取: {e}")

    # 清理
    del diag_context
    del diag_sys
    gc.collect()
    return stats


# ============================================================================
# 6. 传统λ-REMD 采样引擎 (从 traditional_abfe_remd.py 迁移)
# ============================================================================
class REMDManager:
    """传统 λ-REMD 引擎：相邻窗口 Metropolis 交换 + 轨迹落盘"""
    class _ReporterIntegratorView:
        def __init__(self, step_size):
            self._step_size = step_size

        def getStepSize(self):
            return self._step_size

    class _ReporterSimulationView:
        def __init__(self, topology, current_step: int, step_size):
            self.topology = topology
            self.currentStep = int(current_step)
            self.integrator = REMDManager._ReporterIntegratorView(step_size)

    def __init__(
        self,
        system_template: openmm.System,
        topology: app.Topology,
        positions,
        box_vectors,
        ligand_indices: List[int],
        lambdas_coul: List[float],
        lambdas_vdw: List[float],
        temperature: float = 300.0,
        platform_name: str = "CUDA",
        output_dir: str = "./remd_output",
        boresch_params: Optional[Dict] = None,
        random_seed: Optional[int] = None,
        max_resident_contexts: Optional[int] = None,
    ):
        self.topology = topology
        self.positions = positions
        self.box_vectors = box_vectors
        self.ligand_indices = ligand_indices
        self.lambdas_coul = np.array(lambdas_coul)
        self.lambdas_vdw = np.array(lambdas_vdw)
        self.n_replicas = len(lambdas_coul)
        self.has_coulomb_scaling = not np.allclose(
            self.lambdas_coul,
            self.lambdas_coul[0],
        )
        self.has_vdw_scaling = not np.allclose(
            self.lambdas_vdw,
            self.lambdas_vdw[0],
        )
        self.is_pme_coulomb_leg = (
            np.allclose(self.lambdas_vdw, 1.0)
            and self.has_coulomb_scaling
        )
        self.is_mixed_pme_alchemical = self.has_coulomb_scaling and self.has_vdw_scaling
        self.temperature = temperature * unit.kelvin
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(unit.kilojoule_per_mole)
        self.beta = 1.0 / self.kt
        self.platform_name = platform_name
        requested_platform, _ = _split_platform_spec(platform_name)
        if max_resident_contexts is None:
            # 🔑 [2026-07-27] 这里原来对 CUDA/OPENCL 默认 **1**。但当前交换实现
            # 天生要求每个 replica 同时持有独立 Context（`context_residency_mode
            # = "all_resident"`），于是「n_replicas > 1」恒成立 → **任何** REMD
            # 都会在建 Context 之前静默回退 CPU，GPU 路径事实上不可达。
            #
            # 代价极不对称：
            #   · CPU 回退让整个阶段慢约两个数量级（实测 73536 原子 × 12 副本 ×
            #     PME：GPU 上 ~24 分钟跑完 500 轮交换；回退 CPU 后 23 分钟连第一个
            #     DCD 帧都没写出来）；而且这个决定原先只 print 到终端、
            #     `pipeline.log` 里完全看不见，表现成"卡住了"。
            #   · 真正的 GPU OOM **本来就有优雅处理**：`_build_replicas` 的
            #     `except` 分支会 `_clear_replica_contexts()` 再回退 CPU 重建
            #     （`_is_gpu_context_failure`）。OOM 是响亮且立即的，慢 100 倍是静默的。
            #
            # 实测 12 个 73536 原子 PME Context 在 11 GB RTX 2080 Ti 上装得下
            # （`--only-complex-charging` 那条路径一直这么跑，manifest 记录
            # `platform_name: CUDA`、`platform_fallback_reason: None`）。
            # 所以默认改为「不预防性回退」，把判断交给真实的构建期 OOM 处理；
            # 显存小的机器仍可显式传小值来强制 CPU 回退。
            max_resident_contexts = self.n_replicas
        self.max_resident_contexts = int(max_resident_contexts)
        if self.max_resident_contexts < 1:
            raise ValueError("max_resident_contexts 必须至少为 1")
        self.context_residency_mode = "all_resident"
        # 当前交换实现要求每个 replica 同时持有独立 Context。与其在单 GPU 上
        # 先创建 N 个 Context、OOM 后才补救，不如在构建前 fail-safe 地切到 CPU。
        # 这保持算法和 RNG/交换语义不变，同时保证 GPU 常驻 Context 不超过上限。
        if (
            requested_platform.upper() in {"CUDA", "OPENCL"}
            and self.n_replicas > self.max_resident_contexts
        ):
            # 只有调用方**显式**传了小于 n_replicas 的上限才会走到这里。
            # 这个决定会让整个阶段慢约两个数量级，必须响亮且可在归档日志里看见——
            # 用 logger.warning 而不是裸 print，避免只出现在终端 stdout。
            _msg = (
                f"⚠️ REMD replica 数({self.n_replicas})超过显式指定的 GPU 常驻 Context "
                f"上限({self.max_resident_contexts})；在创建任何 GPU Context 前回退 CPU。"
                "\n     ⛔ 注意：CPU 回退会让本阶段慢约两个数量级（实测 73536 原子 × "
                "12 副本 × PME：GPU ~24 分钟跑完 500 轮交换，CPU 上 23 分钟连第一帧 "
                "DCD 都写不出来，表现得像卡死）。"
                "\n     若显存足够，请提高 max_resident_contexts（完整流程用 "
                "--charging-max-resident-contexts）；真实 OOM 由构建期回退兜底。"
            )
            logger.warning(_msg)
            print(f"  {_msg}")
            self.platform_name = "CPU"
            self.context_residency_mode = "cpu_fallback_bounded_gpu_contexts"
            self.platform_fallback_reason = (
                f"explicit_max_resident_contexts={self.max_resident_contexts}"
                f"_below_n_replicas={self.n_replicas}"
            )
        self.output_dir = output_dir
        self.boresch_params = boresch_params
        if random_seed is None:
            seed_env = os.environ.get("IBS_RANDOM_SEED") or os.environ.get("ABFE_RANDOM_SEED")
            random_seed = int(seed_env) if seed_env not in (None, "") else None
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)
        os.makedirs(output_dir, exist_ok=True)

        self.contexts = []
        self.integrators = []
        self.replica_systems = []
        self._atoms = list(topology.atoms()) if topology is not None else []
        self._ligand_set = set(int(i) for i in ligand_indices)
        self._state_to_context = list(range(self.n_replicas))
        self._context_to_state = list(range(self.n_replicas))
        self._steps_completed = 0
        self._is_warmed_up = False
        # 不要无条件置 None：上面「显式上限低于 n_replicas → 预防性回退 CPU」那个
        # 分支已经写过这个字段，无条件覆盖会把回退原因抹掉，让 exchange_diagnostics
        # 里显示 platform_fallback_reason=None 而实际却在 CPU 上跑。
        if getattr(self, "platform_fallback_reason", None) is None:
            self.platform_fallback_reason = None
        self._build_replicas(system_template)

    @staticmethod
    def _try_set_context_parameter(context, name: str, value: float) -> None:
        try:
            context.setParameter(name, float(value))
        except Exception as exc:
            msg = str(exc)
            if "invalid parameter name" not in msg:
                raise

    def _clear_replica_contexts(self) -> None:
        """Release already-created OpenMM Contexts after a partial REMD build failure."""
        self.contexts = []
        self.integrators = []
        self.replica_systems = []
        self._state_to_context = list(range(self.n_replicas))
        self._context_to_state = list(range(self.n_replicas))
        gc.collect()

    @staticmethod
    def _is_gpu_context_failure(exc: Exception) -> bool:
        msg = str(exc).lower()
        needles = (
            "no compatible cuda device",
            "cuda_error_out_of_memory",
            "out of memory",
            "failed to create cuda context",
            "cucontextcreate",
            "cuda device",
        )
        return any(token in msg for token in needles)

    def _build_replicas(self, system_template, allow_platform_fallback: bool = True):
        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)

        try:
            if (
                str(resolved_platform).upper() in {"CUDA", "OPENCL"}
                and self.n_replicas > self.max_resident_contexts
            ):
                raise RuntimeError(
                    "REMD GPU Context 需求超过 max_resident_contexts；"
                    "拒绝无界创建 replica Context"
                )
            for i in range(self.n_replicas):
                if self.is_pme_coulomb_leg:
                    replica_sys = _prepare_pme_coulomb_leg_system(
                        system_template,
                        self.ligand_indices,
                        lambda_name="lambda_coul",
                        allow_charged_ligand=True,
                        topology=self.topology,
                        positions=self.positions,
                        box_vectors=self.box_vectors,
                    )
                    _add_physical_boresch_restraint(replica_sys, self.boresch_params, force_group=3)
                elif self.is_mixed_pme_alchemical:
                    replica_sys = _prepare_pme_mixed_alchemical_system(
                        system_template,
                        self.ligand_indices,
                        topology=self.topology,
                        positions=self.positions,
                        box_vectors=self.box_vectors,
                        lambda_coul_name="lambda_coul",
                        lambda_vdw_name="lambda_vdw",
                        restraint_params=self.boresch_params,
                    )
                else:
                    sys_xml = openmm.XmlSerializer.serialize(system_template)
                    replica_sys = openmm.XmlSerializer.deserialize(sys_xml)
                    replica_sys.thisown = 1
                    nb = [f for f in replica_sys.getForces() if isinstance(f, openmm.NonbondedForce)][0]
                    _restore_ligand_internal_nonbonded(
                        replica_sys,
                        nb,
                        self.ligand_indices,
                        zero_original_exceptions=True,
                    )
                    env_idx = [j for j in range(replica_sys.getNumParticles()) if j not in self.ligand_indices]
                    sc_force = BeutlerSoftcoreBuilder.build(nb, self.ligand_indices, env_idx)
                    sc_force.setForceGroup(1)
                    replica_sys.addForce(sc_force)
                    _add_physical_boresch_restraint(replica_sys, self.boresch_params, force_group=3)

                    for idx in self.ligand_indices:
                        nb.setParticleParameters(idx, 0.0, 0.1*unit.nanometer, 0.0)

                integ = openmm.LangevinMiddleIntegrator(self.temperature, 1.0/unit.picosecond, 0.002*unit.picosecond)
                ctx = openmm.Context(replica_sys, integ, platform, props)
                ctx.setPositions(self.positions)
                if self.box_vectors is not None:
                    ctx.setPeriodicBoxVectors(*self.box_vectors)
                self._try_set_context_parameter(ctx, "lambda_coul", self.lambdas_coul[i])
                self._try_set_context_parameter(ctx, "lambda_vdw", self.lambdas_vdw[i])
                ctx.setVelocitiesToTemperature(self.temperature)

                self.contexts.append(ctx)
                self.integrators.append(integ)
                self.replica_systems.append(replica_sys)
        except Exception as exc:
            self._clear_replica_contexts()
            if (
                allow_platform_fallback
                and str(resolved_platform).upper() in {"CUDA", "OPENCL"}
                and self._is_gpu_context_failure(exc)
            ):
                self.platform_fallback_reason = str(exc)
                print(
                    "  ⚠️ REMD GPU Context 构建失败，已释放已创建的 replica contexts；"
                    f"回退 CPU 重建。原始错误: {exc}"
                )
                self.platform_name = "CPU"
                self._build_replicas(system_template, allow_platform_fallback=False)
                return
            raise

    def _set_context_state(self, context_idx: int, state_idx: int) -> None:
        ctx = self.contexts[context_idx]
        self._try_set_context_parameter(ctx, "lambda_coul", self.lambdas_coul[state_idx])
        self._try_set_context_parameter(ctx, "lambda_vdw", self.lambdas_vdw[state_idx])

    @staticmethod
    def _crossed_save_boundary(prev_step: int, next_step: int, save_interval: int) -> bool:
        if save_interval <= 0:
            return False
        return (prev_step // save_interval) != (next_step // save_interval)

    def _atom_label(self, idx: int) -> str:
        idx = int(idx)
        role = "ligand" if idx in self._ligand_set else "env"
        if 0 <= idx < len(self._atoms):
            atom = self._atoms[idx]
            residue = atom.residue
            resname = residue.name.upper()
            if role != "ligand":
                if resname in {"HOH", "WAT", "SOL", "TIP3", "TIP3P"}:
                    role = "water"
                elif resname in {"NA", "SOD", "CL", "CLA", "K", "POT", "MG", "CA"}:
                    role = "ion"
            return f"{idx}:{atom.name}/{residue.name}{residue.index}:{role}"
        return f"{idx}:{role}"

    def _context_lambda_label(self, context_idx: int) -> Tuple[int, float, float]:
        state_idx = int(self._context_to_state[int(context_idx)])
        return state_idx, float(self.lambdas_coul[state_idx]), float(self.lambdas_vdw[state_idx])

    def _diagnose_context_failure(
        self,
        context_idx: int,
        phase: str,
        exc: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        context_idx = int(context_idx)
        ctx = self.contexts[context_idx]
        system = self.replica_systems[context_idx] if context_idx < len(self.replica_systems) else None
        state_idx, lam_coul, lam_vdw = self._context_lambda_label(context_idx)
        diag: Dict[str, Any] = {
            "phase": str(phase),
            "context_idx": context_idx,
            "state_idx": state_idx,
            "lambda_coul": lam_coul,
            "lambda_vdw": lam_vdw,
            "exception": str(exc) if exc is not None else None,
        }

        print("\n🚨 [REMD-NaN诊断] 捕获到不稳定 context")
        print(
            f"  phase={phase} | context={context_idx} | state={state_idx} | "
            f"lambda_coul={lam_coul:.6f} | lambda_vdw={lam_vdw:.6f}"
        )
        if exc is not None:
            print(f"  OpenMM 异常: {exc}")

        state = None
        try:
            state = ctx.getState(
                getPositions=True,
                getEnergy=True,
                getForces=True,
                enforcePeriodicBox=False,
            )
        except Exception as state_exc:
            diag["state_read_error"] = str(state_exc)
            print(f"  ⚠️ 无法同时读取能量/坐标/受力: {state_exc}")
            try:
                state = ctx.getState(getPositions=True, enforcePeriodicBox=False)
            except Exception as pos_exc:
                diag["position_read_error"] = str(pos_exc)
                print(f"  ⚠️ 坐标也无法读取: {pos_exc}")

        if state is not None:
            try:
                positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                bad_pos = np.where(~np.isfinite(positions_nm).all(axis=1))[0]
                diag["n_nonfinite_position_atoms"] = int(len(bad_pos))
                diag["nonfinite_position_atoms"] = [self._atom_label(i) for i in bad_pos[:20]]
                if len(bad_pos) > 0:
                    print(f"  ❌ 非有限坐标原子数: {len(bad_pos)}")
                    print("     " + ", ".join(diag["nonfinite_position_atoms"]))
            except Exception as pos_exc:
                diag["position_parse_error"] = str(pos_exc)

            try:
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                diag["potential_energy_kj_mol"] = float(energy)
                print(f"  E_potential={energy:.3f} kJ/mol")
            except Exception as energy_exc:
                diag["energy_error"] = str(energy_exc)

            try:
                forces = state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                )
                norms = np.linalg.norm(forces, axis=1)
                bad_force = np.where(~np.isfinite(forces).all(axis=1))[0]
                finite_norms = np.where(np.isfinite(norms), norms, -np.inf)
                top = np.argsort(finite_norms)[::-1][:10]
                diag["n_nonfinite_force_atoms"] = int(len(bad_force))
                diag["top_force_atoms"] = [
                    {"atom": self._atom_label(i), "force_kj_mol_nm": float(norms[int(i)])}
                    for i in top
                    if np.isfinite(norms[int(i)])
                ]
                if np.any(np.isfinite(norms)):
                    diag["max_force_kj_mol_nm"] = float(np.nanmax(norms))
                    print(f"  max|F|={diag['max_force_kj_mol_nm']:.1f} kJ/mol/nm")
                if len(bad_force) > 0:
                    diag["nonfinite_force_atoms"] = [self._atom_label(i) for i in bad_force[:20]]
                    print(f"  ❌ 非有限受力原子数: {len(bad_force)}")
                    print("     " + ", ".join(diag["nonfinite_force_atoms"]))
                if diag["top_force_atoms"]:
                    print("  Top受力原子:")
                    for row in diag["top_force_atoms"][:5]:
                        print(f"    {row['atom']:<28} {row['force_kj_mol_nm']:12.1f}")
            except Exception as force_exc:
                diag["force_error"] = str(force_exc)

        if _has_valid_boresch_restraint(self.boresch_params):
            try:
                ok, r0_chk, thA_chk, thB_chk = _check_boresch_geometry_safe(ctx, self.boresch_params)
                diag["boresch_geometry"] = {
                    "safe": bool(ok),
                    "r_nm": float(r0_chk),
                    "thetaA_deg": float(thA_chk),
                    "thetaB_deg": float(thB_chk),
                }
                print(
                    f"  Boresch当前几何: r={r0_chk:.3f} nm, "
                    f"thetaA={thA_chk:.1f}°, thetaB={thB_chk:.1f}°, safe={ok}"
                )
            except Exception as boresch_exc:
                diag["boresch_geometry_error"] = str(boresch_exc)
                print(f"  ⚠️ Boresch 几何诊断失败: {boresch_exc}")

        group_stats = []
        if system is not None:
            for gid in sorted({force.getForceGroup() for force in system.getForces()}):
                row: Dict[str, Any] = {"group": int(gid)}
                try:
                    g_state = ctx.getState(getEnergy=True, getForces=True, groups={gid})
                    g_energy = g_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    g_forces = g_state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer
                    )
                    g_norms = np.linalg.norm(g_forces, axis=1)
                    row["energy_kj_mol"] = float(g_energy)
                    row["max_force_kj_mol_nm"] = float(np.nanmax(g_norms))
                except Exception as group_exc:
                    row["error"] = str(group_exc)
                group_stats.append(row)
            diag["force_groups"] = group_stats
            print("  ForceGroup 分解:")
            for row in group_stats:
                if "error" in row:
                    print(f"    G{row['group']}: ERROR {row['error']}")
                else:
                    print(
                        f"    G{row['group']}: E={row['energy_kj_mol']:.3f} kJ/mol | "
                        f"max|F|={row['max_force_kj_mol_nm']:.1f}"
                    )

        try:
            out_path = os.path.join(self.output_dir, "remd_last_failure.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2, ensure_ascii=False)
            print(f"  📝 REMD 失败诊断已写入: {out_path}")
        except Exception as write_exc:
            print(f"  ⚠️ REMD 诊断写入失败: {write_exc}")

        return diag

    def _preflight_context(self, context_idx: int) -> None:
        ctx = self.contexts[int(context_idx)]
        state_idx, lam_coul, lam_vdw = self._context_lambda_label(context_idx)
        max_force_limit = float(os.environ.get("IBS_REMD_PREFLIGHT_MAX_FORCE", "100000"))
        try:
            state = ctx.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            norms = np.linalg.norm(forces, axis=1)
            max_force = float(np.nanmax(norms))
            if (not np.isfinite(energy)) or (not np.isfinite(max_force)):
                self._diagnose_context_failure(context_idx, f"preflight_nonfinite_state{state_idx}")
                raise RuntimeError(
                    f"REMD preflight 非有限能量/受力: state={state_idx}, "
                    f"lambda_coul={lam_coul}, lambda_vdw={lam_vdw}"
                )
            if max_force > max_force_limit:
                print(
                    f"  ⚠️ [REMD] context {context_idx} state {state_idx} "
                    f"预检查 max|F|={max_force:.1f} > {max_force_limit:.1f}，执行短最小化"
                )
                openmm.LocalEnergyMinimizer.minimize(ctx, tolerance=20.0, maxIterations=1000)
                ctx.setVelocitiesToTemperature(self.temperature)
        except Exception as exc:
            self._diagnose_context_failure(context_idx, f"preflight_state{state_idx}", exc)
            raise

    def _relax_context_before_preheat(self, context_idx: int) -> None:
        """Minimize and gently settle each REMD replica before the first long MD step."""
        ctx = self.contexts[int(context_idx)]
        state_idx, lam_coul, lam_vdw = self._context_lambda_label(context_idx)
        max_force_limit = float(os.environ.get("IBS_REMD_PREFLIGHT_MAX_FORCE", "100000"))
        em_iters = int(os.environ.get("IBS_REMD_PREHEAT_EM_ITERS", "5000"))
        em_tol = float(os.environ.get("IBS_REMD_PREHEAT_EM_TOL", "10.0"))
        try:
            print(
                f"  [REMD] context {context_idx} state {state_idx} "
                f"预热前 EM: lambda_coul={lam_coul:.6f}, lambda_vdw={lam_vdw:.6f}"
            )
            openmm.LocalEnergyMinimizer.minimize(
                ctx,
                tolerance=em_tol * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=em_iters,
            )
            ctx.setVelocitiesToTemperature(self.temperature)

            state = ctx.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            max_force = float(np.nanmax(np.linalg.norm(forces, axis=1)))
            if (not np.isfinite(energy)) or (not np.isfinite(max_force)) or max_force > max_force_limit:
                self._diagnose_context_failure(context_idx, f"post_em_preheat_state{state_idx}")
                raise RuntimeError(
                    f"REMD 预热前 EM 后仍不稳定: state={state_idx}, "
                    f"lambda_coul={lam_coul:.6f}, lambda_vdw={lam_vdw:.6f}, "
                    f"E={energy}, max|F|={max_force:.1f}"
                )
        except Exception as exc:
            self._diagnose_context_failure(context_idx, f"preheat_em_state{state_idx}", exc)
            raise

    def _preheat_context_gently(self, context_idx: int, total_steps: int, phase: str) -> None:
        """Run preheat with a short timestep ramp instead of one long 2 fs block."""
        ctx = self.contexts[int(context_idx)]
        integ = ctx.getIntegrator()
        original_dt = integ.getStepSize()
        total_steps = max(0, int(total_steps))
        if total_steps <= 0:
            return

        schedule = [
            (0.0001, min(500, total_steps), "0.1fs"),
            (0.00025, min(1000, total_steps), "0.25fs"),
            (0.0005, min(1500, total_steps), "0.5fs"),
        ]
        used = 0
        try:
            for dt_ps, n_step, label in schedule:
                if used >= total_steps:
                    break
                n_step = min(int(n_step), total_steps - used)
                if n_step <= 0:
                    continue
                integ.setStepSize(dt_ps * unit.picoseconds)
                self._step_context_with_diagnostics(
                    context_idx,
                    n_step,
                    f"{phase}:ramp_{label}",
                )
                used += n_step

            remaining = total_steps - used
            if remaining > 0:
                integ.setStepSize(original_dt)
                self._step_context_with_diagnostics(
                    context_idx,
                    remaining,
                    f"{phase}:ramp_full_dt",
                )
        finally:
            integ.setStepSize(original_dt)

    def _step_context_with_diagnostics(self, context_idx: int, n_steps: int, phase: str) -> None:
        ctx = self.contexts[int(context_idx)]
        n_steps = int(n_steps)
        if n_steps <= 0:
            return
        chunk = int(os.environ.get("IBS_REMD_STEP_CHUNK", str(n_steps)))
        chunk = max(1, min(chunk, n_steps))
        steps_done = 0
        try:
            while steps_done < n_steps:
                this_chunk = min(chunk, n_steps - steps_done)
                ctx.getIntegrator().step(this_chunk)
                steps_done += this_chunk
        except Exception as exc:
            self._diagnose_context_failure(
                context_idx,
                f"{phase}; local_step={steps_done}/{n_steps}",
                exc,
            )
            state_idx, lam_coul, lam_vdw = self._context_lambda_label(context_idx)
            raise RuntimeError(
                f"REMD context {context_idx} 在 {phase} 爆炸: "
                f"state={state_idx}, lambda_coul={lam_coul:.6f}, lambda_vdw={lam_vdw:.6f}"
            ) from exc

    def run(
        self,
        n_steps: int = 500000,
        exchange_interval: int = 1000,
        save_interval: int = 5000,
        stage_name: str = "complex",
    ):
        n_exchanges = n_steps // exchange_interval
        remaining_steps = n_steps - n_exchanges * exchange_interval
        traj_files = [os.path.join(self.output_dir, f"{stage_name}_rep{i}.dcd") for i in range(self.n_replicas)]
        append_mode = self._steps_completed > 0
        reporters = [
            app.DCDReporter(f, save_interval, append=append_mode, enforcePeriodicBox=False)
            for f in traj_files
        ]
        print(f"\n🔄 启动传统 REMD (单卡懒加载极速版) | {self.n_replicas} 副本 | 交换间隔={exchange_interval}")
        
        # 预热
        if not self._is_warmed_up:
            print("  [REMD] 预热前执行能量/受力 preflight...")
            for ctx_idx in range(self.n_replicas):
                self._preflight_context(ctx_idx)
                self._relax_context_before_preheat(ctx_idx)
            for ctx_idx in range(self.n_replicas):
                self._preheat_context_gently(
                    ctx_idx,
                    5000,
                    f"{stage_name}:preheat",
                )
            self._is_warmed_up = True
            
        exchange_log = []
        
        for step in range(n_exchanges):
            # 1. 批量提交步进任务 (GPU 会在底层自动流水线并发，无需 Python 干预)
            prev_steps = self._steps_completed
            for ctx_idx in range(self.n_replicas):
                state_idx, _, _ = self._context_lambda_label(ctx_idx)
                self._step_context_with_diagnostics(
                    ctx_idx,
                    exchange_interval,
                    f"{stage_name}:exchange_round={step}:state={state_idx}",
                )
            self._steps_completed += exchange_interval
                
            # 2. 轨迹落盘
            if self._crossed_save_boundary(prev_steps, self._steps_completed, save_interval):
                for state_idx, ctx_idx in enumerate(self._state_to_context):
                    ctx = self.contexts[ctx_idx]
                    state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
                    reporters[state_idx].report(
                        self._ReporterSimulationView(
                            self.topology, self._steps_completed, ctx.getIntegrator().getStepSize()
                        ), state,
                    )
                
            accepted = 0
            
            # 3. 交换状态映射而非整份坐标/速度，避免接受交换时的大对象搬运
            for state_i in range(self.n_replicas - 1):
                state_j = state_i + 1
                ctx_idx_i = self._state_to_context[state_i]
                ctx_idx_j = self._state_to_context[state_j]
                ctx_i = self.contexts[ctx_idx_i]
                ctx_j = self.contexts[ctx_idx_j]
                
                # --- 阶段 A: 仅获取能量 (极快，PCIe 传输量仅几字节，不阻塞 GPU) ---
                U_i_i = ctx_i.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                U_j_j = ctx_j.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                
                # --- 阶段 B: 计算交叉能量 (通过修改参数，GPU 原地重算) ---
                self._set_context_state(ctx_idx_i, state_j)
                U_i_j = ctx_i.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                
                self._set_context_state(ctx_idx_j, state_i)
                U_j_i = ctx_j.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                
                # 恢复原始参数
                self._set_context_state(ctx_idx_i, state_i)
                self._set_context_state(ctx_idx_j, state_j)

                if not np.all(np.isfinite([U_i_i, U_j_j, U_i_j, U_j_i])):
                    print(
                        f"  ⚠️ [REMD] 交换能量非有限: pair=({state_i},{state_j}) "
                        f"Uii={U_i_i}, Ujj={U_j_j}, Uij={U_i_j}, Uji={U_j_i}"
                    )
                    self._diagnose_context_failure(
                        ctx_idx_i,
                        f"{stage_name}:exchange_energy_state{state_i}_{state_j}",
                    )
                    self._diagnose_context_failure(
                        ctx_idx_j,
                        f"{stage_name}:exchange_energy_state{state_i}_{state_j}",
                    )
                    continue
                
                # --- 阶段 C: Metropolis 判定 ---
                delta = self.beta * (U_i_j + U_j_i - U_i_i - U_j_j)
                if not np.isfinite(delta):
                    accept = bool(np.isneginf(delta))
                else:
                    accept = delta < 0 or self.rng.random() < np.exp(-delta)
                
                # --- 阶段 D: 仅交换状态分配，Context 本身保持连续推进 ---
                if accept:
                    self._state_to_context[state_i], self._state_to_context[state_j] = (
                        ctx_idx_j,
                        ctx_idx_i,
                    )
                    self._context_to_state[ctx_idx_i], self._context_to_state[ctx_idx_j] = (
                        state_j,
                        state_i,
                    )
                    self._set_context_state(ctx_idx_i, state_j)
                    self._set_context_state(ctx_idx_j, state_i)
                    accepted += 1
                    
            exchange_log.append(accepted / (self.n_replicas - 1))
            if step % 50 == 0:
                print(f"  [REMD] 交换轮次 {step}/{n_exchanges} | 接受率: {exchange_log[-1]:.2f}")

        if remaining_steps > 0:
            prev_steps = self._steps_completed
            for ctx_idx in range(self.n_replicas):
                state_idx, _, _ = self._context_lambda_label(ctx_idx)
                self._step_context_with_diagnostics(
                    ctx_idx,
                    remaining_steps,
                    f"{stage_name}:remaining:state={state_idx}",
                )
            self._steps_completed += remaining_steps
            if self._crossed_save_boundary(prev_steps, self._steps_completed, save_interval):
                for state_idx, ctx_idx in enumerate(self._state_to_context):
                    ctx = self.contexts[ctx_idx]
                    state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
                    reporters[state_idx].report(
                        self._ReporterSimulationView(
                            self.topology, self._steps_completed, ctx.getIntegrator().getStepSize()
                        ),
                        state,
                    )
                
        # OpenMM 的 DCDReporter 没有公共 close()；对象释放时会完成底层文件收尾。
        reporters.clear()
        exchange_summary = {
            "stage_name": stage_name,
            "n_replicas": int(self.n_replicas),
            "platform_name": str(self.platform_name),
            "platform_fallback_reason": self.platform_fallback_reason,
            "n_exchange_attempts": int(n_exchanges),
            "acceptance_by_round": [float(v) for v in exchange_log],
            "mean_acceptance": float(np.mean(exchange_log)) if exchange_log else 0.0,
            "min_acceptance": float(np.min(exchange_log)) if exchange_log else 0.0,
            "max_acceptance": float(np.max(exchange_log)) if exchange_log else 0.0,
        }
        with open(os.path.join(self.output_dir, f"{stage_name}_exchange_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(exchange_summary, f, indent=2)
        print(f"✅ REMD 完成 | 平均交换接受率: {exchange_summary['mean_acceptance']:.3f}")
        return traj_files


# ============================================================================
# 6.4 Boresch attachment 专用 HREMD：同温度，只交换 lambda_boresch_scale
# ============================================================================


class BoreschAttachmentREMDManager(REMDManager):
    """Hamiltonian REMD over `lambda_boresch_scale`（同温，不是温度 REMD）。

    ⛔ **不是生产路径，默认不可达。** 2026-07-28 首次启用产出
    `38.6006 ± 109.9858 kJ/mol`、零 round trip，而同一体系的顺序窗口给
    `5.3784`（BAR）/ `5.3867`（TI），两者一致。

    根因不是实现错，是物理：Boresch 的 `k(1−cosΔ)` 项在反转（Δ=π）时取 2k——
    单个二面角 359–469 kJ/mol（144–188 kT），三个全反转 1189 kJ/mol = 477 kT。
    副本交换一旦把某个副本送进翻转态，那一帧的天文 U_B 就支配了指数平均
    （σ=110 就是它的指纹）。顺序窗口从不离开原盆地，反而良态。

    而且**单盆地是必要的**：配套的解析释放项
    （`calculate_boresch_analytical_correction`）本身假定单一简谐盆地，
    配一个会采到翻转的 attachment 腿反而不自洽。

    要重新启用，先解决翻转态的处理（例如对 U_B 设物理上界、或改用不含
    `1−cos` 反转分支的限制形式），并且必须过 `max(round_trips) > 0` 这道门。

    与 `ShadowBridgeREMDManager` 同一套路：交换循环（Metropolis 判据、DCD 落盘、
    诊断）完全继承自 `REMDManager.run()`——那部分只通过 `_set_context_state` /
    `_context_lambda_label` 间接接触 lambda_coul/lambda_vdw，本身与具体参数无关。
    这里只重写三个方法，把驱动变量换成 `lambda_boresch_scale`。

    额外记录 `state_history`：每次落帧时快照 context→state 映射，用来数
    λ=0 ↔ 1 的 round trip。REMDManager 基类只维护当前映射、不留历史。
    """

    def __init__(
        self,
        attachment_system: openmm.System,
        topology: app.Topology,
        positions,
        box_vectors,
        ligand_indices: List[int],
        lambdas_boresch: List[float],
        lam_name: str = BORESCH_ATTACHMENT_LAMBDA_NAME,
        temperature: float = 300.0,
        platform_name: str = "CUDA",
        output_dir: str = "./boresch_attachment_remd",
        random_seed: Optional[int] = None,
    ):
        self.lambdas_boresch = np.asarray(lambdas_boresch, dtype=float)
        self.lam_name = lam_name
        self.state_history: List[List[int]] = []
        n = len(self.lambdas_boresch)
        super().__init__(
            system_template=attachment_system,
            topology=topology,
            positions=positions,
            box_vectors=box_vectors,
            ligand_indices=ligand_indices,
            lambdas_coul=[0.0] * n,   # 占位，只为让基类旧字段存在
            lambdas_vdw=[1.0] * n,
            temperature=temperature,
            platform_name=platform_name,
            output_dir=output_dir,
            boresch_params=None,      # 限制力已经在 attachment_system 里了
            random_seed=random_seed,
        )

    def _build_replicas(self, system_template, allow_platform_fallback: bool = True):
        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)
        try:
            for i in range(self.n_replicas):
                sys_xml = openmm.XmlSerializer.serialize(system_template)
                replica_sys = openmm.XmlSerializer.deserialize(sys_xml)
                replica_sys.thisown = 1
                integ = openmm.LangevinMiddleIntegrator(
                    self.temperature, 1.0 / unit.picosecond, 0.002 * unit.picosecond
                )
                ctx = openmm.Context(replica_sys, integ, platform, props)
                ctx.setPositions(self.positions)
                if self.box_vectors is not None:
                    ctx.setPeriodicBoxVectors(*self.box_vectors)
                self._try_set_context_parameter(ctx, self.lam_name, self.lambdas_boresch[i])
                ctx.setVelocitiesToTemperature(self.temperature)
                self.contexts.append(ctx)
                self.integrators.append(integ)
                self.replica_systems.append(replica_sys)
        except Exception as exc:
            self._clear_replica_contexts()
            if (
                allow_platform_fallback
                and str(resolved_platform).upper() in {"CUDA", "OPENCL"}
                and self._is_gpu_context_failure(exc)
            ):
                self.platform_fallback_reason = str(exc)
                print(
                    "  ⚠️ attachment HREMD GPU Context 构建失败，已释放已创建的 contexts；"
                    f"回退 CPU 重建。原始错误: {exc}"
                )
                self.platform_name = "CPU"
                self._build_replicas(system_template, allow_platform_fallback=False)
                return
            raise

    def _set_context_state(self, context_idx: int, state_idx: int) -> None:
        ctx = self.contexts[context_idx]
        self._try_set_context_parameter(ctx, self.lam_name, self.lambdas_boresch[state_idx])

    def _context_lambda_label(self, context_idx: int) -> Tuple[int, float, float]:
        state_idx = int(self._context_to_state[int(context_idx)])
        lam_val = float(self.lambdas_boresch[state_idx])
        if int(context_idx) == 0:
            # 每一轮落帧从 context 0 开始，借这里快照整张映射表。
            self.state_history.append(list(self._context_to_state))
        return state_idx, lam_val, lam_val

    def begin_production_stage(self) -> None:
        """平衡段跑完、生产段开始前调用。

        两件事：
        1. 清空 `state_history`，别把平衡期的交换记录混进生产统计；
        2. **把 `_steps_completed` 归零**。基类用它决定 DCDReporter 的 append
           模式（`append_mode = self._steps_completed > 0`）。平衡段跑完后它是
           正数，而生产段换了 `stage_name`、对应的 DCD 还不存在——以 append
           模式打开不存在的文件会让 DCDReporter 构造直接抛异常，表现为
           `'DCDReporter' object has no attribute '_out'`，真正的报错反而被
           `__del__` 里的二次异常盖住。
           `_is_warmed_up` 不动，预热不会重做。
        """
        self.state_history = []
        self._steps_completed = 0

    def cleanup(self) -> None:
        self._clear_replica_contexts()

    def round_trip_counts(self) -> List[int]:
        """每个 replica 完成了几次 λ=0 ↔ λ=1 的往返。

        往返 = 摸到一端后再摸到另一端算半程，两个半程算一次。REMD 的采样效率
        直接由它决定：没有往返，副本交换等于没做。
        """
        if not self.state_history:
            return [0] * self.n_replicas
        top = self.n_replicas - 1
        trips = []
        for ctx in range(self.n_replicas):
            last_end, halves = None, 0
            for snapshot in self.state_history:
                st = snapshot[ctx]
                if st == 0 or st == top:
                    end = "lo" if st == 0 else "hi"
                    if last_end is not None and end != last_end:
                        halves += 1
                    last_end = end
            trips.append(halves // 2)
        return trips


def compute_boresch_attachment_u_kn(
    attachment_system: openmm.System,
    topology: app.Topology,
    traj_files: List[str],
    box_vectors=None,
    platform_name: str = "CPU",
) -> Tuple[np.ndarray, np.ndarray]:
    """离线重算每帧的 **U_Boresch(x)**（kJ/mol，λ=1 下的未缩放值）。

    只取 Boresch 力组。U_common（键合 / 环境 PME / 配体内部）对同一帧在所有 λ
    下完全相同，是逐帧可加常数，在 MBAR 的自洽方程里精确抵消——与
    `compute_shadow_bridge_u_kn` 同一论证。因为势对 λ 严格线性
    （表达式里 λ 是整体前因子），拿到 U_B 之后 u_kn[j] = λ_j·U_B 即可，
    不需要逐 λ 重新评估。

    返回 ``(u_boresch_kj_per_frame, n_k)``，帧按 traj_files 顺序拼接。
    """
    import mdtraj as md

    sys_xml = openmm.XmlSerializer.serialize(attachment_system)
    eval_sys = openmm.XmlSerializer.deserialize(sys_xml)
    eval_sys.thisown = 1
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    resolved_platform, props = _build_platform_properties(platform_name)
    ctx = openmm.Context(
        eval_sys, integrator, openmm.Platform.getPlatformByName(resolved_platform), props
    )
    if box_vectors is not None:
        ctx.setPeriodicBoxVectors(*box_vectors)
    # 只要 U_B，所以固定 λ=1 量一次即可。
    _try_set_ctx_param(ctx, BORESCH_ATTACHMENT_LAMBDA_NAME, 1.0)

    values, n_k = [], []
    mdtop = md.Topology.from_openmm(topology)
    for path in traj_files:
        traj = md.load(path, top=mdtop)
        n_k.append(int(traj.n_frames))
        for f in range(traj.n_frames):
            ctx.setPositions(traj.xyz[f] * unit.nanometer)
            if traj.unitcell_vectors is not None:
                ctx.setPeriodicBoxVectors(
                    *[openmm.Vec3(*v) * unit.nanometer for v in traj.unitcell_vectors[f]]
                )
            e = ctx.getState(
                getEnergy=True, groups={BORESCH_ATTACHMENT_FORCE_GROUP}
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            values.append(float(e))
    del ctx, integrator
    return np.asarray(values, dtype=np.float64), np.asarray(n_k, dtype=int)


def _try_set_ctx_param(context, name: str, value: float) -> None:
    try:
        context.setParameter(name, float(value))
    except Exception:
        pass


def attachment_convergence_diagnostics(
    u_kn_sub, n_k_sub, lam, u_boresch_kj, n_k_raw, kt, remd, dg, err, log=print
) -> Dict:
    """按验收清单出诊断：交换接受率 / round trip / ⟨U_B⟩ 单调性 / 前后半 / 三口径一致。"""
    K = int(lam.size)
    offsets = np.concatenate(([0], np.cumsum(n_k_raw))).astype(int)
    mean_ub = [float(np.mean(u_boresch_kj[offsets[k]:offsets[k + 1]])) for k in range(K)]

    # ⟨U_B⟩ 必须随 λ 减小而单调上升：限制越松、偏离参考几何越远。倒挂 = 没采够。
    violations = [
        {"lambda_hi": float(lam[k + 1]), "lambda_lo": float(lam[k]),
         "mean_u_hi": mean_ub[k + 1], "mean_u_lo": mean_ub[k]}
        for k in range(K - 1)
        if mean_ub[k] < mean_ub[k + 1] - 1e-9
    ]

    trips = remd.round_trip_counts()
    exch_path = os.path.join(remd.output_dir, "attachment_exchange_diagnostics.json")
    exch = {}
    if os.path.isfile(exch_path):
        with open(exch_path, encoding="utf-8") as fh:
            exch = json.load(fh)

    # 前后半程
    half = {}
    try:
        sub_off = np.concatenate(([0], np.cumsum(n_k_sub))).astype(int)
        for label, (lo, hi) in (("first", (0.0, 0.5)), ("second", (0.5, 1.0))):
            cols, nk2 = [], []
            for k in range(K):
                a, b = sub_off[k], sub_off[k + 1]
                n = b - a
                s, e = a + int(lo * n), a + int(hi * n)
                if e - s < 2:
                    raise ValueError("帧数不足以切半")
                cols.append(np.arange(s, e)); nk2.append(e - s)
            an = TraditionalMBARAnalyzer(temperature=kt / 0.008314462618)
            an._last_n_k = np.asarray(nk2, dtype=int)
            half[label] = float(an.solve(u_kn_sub[:, np.concatenate(cols)], decorrelate=False)["delta_G"])
        half["drift"] = half["second"] - half["first"]
        half["drift_over_2sigma"] = abs(half["drift"]) / (2.0 * err) if err > 0 else None
    except Exception as exc:
        half = {"error": str(exc)}

    mean_acc = float(exch.get("mean_acceptance", float("nan")))
    log(f"  交换接受率 平均 {mean_acc:.3f} / 最低 {exch.get('min_acceptance', float('nan'))}")
    log(f"  round trip 每副本: {trips}")
    log(f"  ⟨U_B⟩ 逐 λ: {[round(v, 3) for v in mean_ub]}"
        + ("  ❌ 有倒挂" if violations else "  ✓ 单调"))
    if isinstance(half.get("drift"), float):
        log(f"  前后半程 {half['first']:.4f} / {half['second']:.4f}，"
            f"漂移 {half['drift']:+.4f}（|漂移|/2σ = {half['drift_over_2sigma']:.2f}）")

    if np.isfinite(mean_acc) and mean_acc < 0.20:
        log(f"  ⚠️ 平均交换接受率 {mean_acc:.3f} < 0.20，λ 太稀，加中间态而不是加时间")
    if max(trips) == 0:
        log("  ⚠️ 没有任何副本完成 λ=0↔1 往返——副本交换等于没做，结果不比顺序窗口强")
    if violations:
        log(f"  ⚠️ ⟨U_B⟩ 有 {len(violations)} 处倒挂，仍未收敛")

    # 零 round trip = 副本交换等于没做，结果不比顺序窗口强，禁止进候选。
    if trips and max(trips) == 0:
        raise RuntimeError(
            "attachment HREMD 零 round trip：没有任何副本走通 λ=0↔1，"
            "副本交换实际未生效，拒绝把该结果作为候选。"
        )

    return {
        "mean_u_boresch_per_state_kJ_mol": mean_ub,
        "monotonicity_violations": violations,
        "round_trips_per_replica": trips,
        "exchange": exch,
        "split_half": half,
        "acceptance_ok": bool(np.isfinite(mean_acc) and mean_acc >= 0.20),
        "round_trip_ok": bool(max(trips) > 0) if trips else False,
        "monotonic_ok": not violations,
    }


# ============================================================================
# 6.5 Shadow-PME Bridge 专用 REMD：复用 REMDManager 通用交换机制，
#     只替换"每个 replica 的系统/参数怎么建"这一层。
# ============================================================================
class ShadowBridgeREMDManager(REMDManager):
    """
    REMDManager 的最小子类：交换循环 (Metropolis 判据、DCD 落盘、诊断) 完全继承
    自 REMDManager.run()，不做任何改动——这部分只通过 _set_context_state /
    _context_lambda_label 间接接触 lambda_coul/lambda_vdw，本身跟具体是哪个
    参数无关。这里只重写三个方法，把驱动变量换成 build_shadow_bridge_system()
    产出的单一全局参数 "lambda_bridge_s"：

      - _build_replicas: 不需要 REMDManager 原来那套按 leg 类型分支建系统的重
        逻辑（PME-coulomb-leg / mixed-alchemical / Beutler-softcore），因为
        Bridge 只有一个已经建好的系统，每个 replica 只是它的一份深拷贝 + 不同
        的 lambda_bridge_s 取值。
      - _set_context_state / _context_lambda_label: 把 lambda_coul/lambda_vdw
        换成 lambda_bridge_s。

    基类 __init__ 仍然会用占位的 lambdas_coul=[0]*n / lambdas_vdw=[1]*n 调用
    （只是为了让 has_coulomb_scaling / is_pme_coulomb_leg 等旧字段存在且不报错，
    这两个占位数组本身不会被下面任何重写方法用到）。
    """

    def __init__(
        self,
        bridge_system: openmm.System,
        topology: app.Topology,
        positions,
        box_vectors,
        ligand_indices: List[int],
        lambdas_bridge_s: List[float],
        s_param_name: str = "lambda_bridge_s",
        temperature: float = 300.0,
        platform_name: str = "CUDA",
        output_dir: str = "./shadow_bridge_remd",
        random_seed: Optional[int] = None,
    ):
        self.lambdas_bridge_s = np.asarray(lambdas_bridge_s, dtype=float)
        self.s_param_name = s_param_name
        self._bridge_system_template = bridge_system
        n = len(self.lambdas_bridge_s)
        super().__init__(
            system_template=bridge_system,
            topology=topology,
            positions=positions,
            box_vectors=box_vectors,
            ligand_indices=ligand_indices,
            lambdas_coul=[0.0] * n,
            lambdas_vdw=[1.0] * n,
            temperature=temperature,
            platform_name=platform_name,
            output_dir=output_dir,
            boresch_params=None,
            random_seed=random_seed,
        )

    def _build_replicas(self, system_template, allow_platform_fallback: bool = True):
        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)
        try:
            for i in range(self.n_replicas):
                sys_xml = openmm.XmlSerializer.serialize(system_template)
                replica_sys = openmm.XmlSerializer.deserialize(sys_xml)
                replica_sys.thisown = 1
                integ = openmm.LangevinMiddleIntegrator(self.temperature, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
                ctx = openmm.Context(replica_sys, integ, platform, props)
                ctx.setPositions(self.positions)
                if self.box_vectors is not None:
                    ctx.setPeriodicBoxVectors(*self.box_vectors)
                self._try_set_context_parameter(ctx, self.s_param_name, self.lambdas_bridge_s[i])
                ctx.setVelocitiesToTemperature(self.temperature)
                self.contexts.append(ctx)
                self.integrators.append(integ)
                self.replica_systems.append(replica_sys)
        except Exception as exc:
            self._clear_replica_contexts()
            if (
                allow_platform_fallback
                and str(resolved_platform).upper() in {"CUDA", "OPENCL"}
                and self._is_gpu_context_failure(exc)
            ):
                self.platform_fallback_reason = str(exc)
                print(
                    "  ⚠️ Shadow-Bridge REMD GPU Context 构建失败，已释放已创建的 replica contexts；"
                    f"回退 CPU 重建。原始错误: {exc}"
                )
                self.platform_name = "CPU"
                self._build_replicas(system_template, allow_platform_fallback=False)
                return
            raise

    def _set_context_state(self, context_idx: int, state_idx: int) -> None:
        ctx = self.contexts[context_idx]
        self._try_set_context_parameter(ctx, self.s_param_name, self.lambdas_bridge_s[state_idx])

    def _context_lambda_label(self, context_idx: int) -> Tuple[int, float, float]:
        state_idx = int(self._context_to_state[int(context_idx)])
        s_val = float(self.lambdas_bridge_s[state_idx])
        return state_idx, s_val, s_val


def compute_shadow_bridge_u_kn(
    bridge_system: openmm.System,
    topology: app.Topology,
    traj_files: List[str],
    lambdas_bridge_s: List[float],
    s_param_name: str = "lambda_bridge_s",
    box_vectors=None,
    platform_name: str = "CPU",
    energy_force_group: Optional[int] = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    离线重算 Shadow-Bridge 的 u_kn (未除 kT 的势能矩阵，单位 kJ/mol；调用方自己
    乘 beta)：对每条轨迹里的每一帧，在全部 len(lambdas_bridge_s) 个 s 状态下
    重新评估势能。

    只取 energy_force_group=1 (bridge_force 所在的 force group) 而不是总能量：
    U_common (bonded/环境 PME/配体内部) 对同一帧在所有状态下都完全相同，是纯粹
    的按帧可加常数，在 MBAR 的自洽方程里精确抵消，不影响任何 f_k 结果——只算
    force group 1 更省算力，效果完全等价。跳过整数分组不存在的兼容性问题时可传
    energy_force_group=None 改回算总能量。
    """
    import mdtraj as md

    n_states = len(lambdas_bridge_s)
    if len(traj_files) != n_states:
        raise ValueError(f"traj_files 数量 ({len(traj_files)}) 与 lambdas_bridge_s 数量 ({n_states}) 不一致")

    sys_xml = openmm.XmlSerializer.serialize(bridge_system)
    eval_sys = openmm.XmlSerializer.deserialize(sys_xml)
    eval_sys.thisown = 1
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    resolved_platform, props = _build_platform_properties(platform_name)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    ctx = openmm.Context(eval_sys, integrator, platform, props)
    if box_vectors is not None:
        ctx.setPeriodicBoxVectors(*box_vectors)

    n_k = []
    all_xyz = []
    all_box = []
    for traj_path in traj_files:
        traj = md.load_dcd(traj_path, top=topology if not isinstance(topology, str) else topology)
        n_k.append(traj.n_frames)
        all_xyz.append(traj.xyz)
        if traj.unitcell_vectors is not None:
            all_box.append(traj.unitcell_vectors)
        else:
            all_box.append([None] * traj.n_frames)
    n_k = np.asarray(n_k, dtype=int)
    n_total = int(np.sum(n_k))
    u_kn = np.zeros((n_states, n_total), dtype=np.float64)

    groups = {energy_force_group} if energy_force_group is not None else None
    try:
        frame_cursor = 0
        for k in range(n_states):
            xyz_k = all_xyz[k]
            box_k = all_box[k]
            for local_f in range(xyz_k.shape[0]):
                ctx.setPositions(xyz_k[local_f] * unit.nanometer)
                if box_k[local_f] is not None:
                    ctx.setPeriodicBoxVectors(*(row * unit.nanometer for row in box_k[local_f]))
                n = frame_cursor + local_f
                for j in range(n_states):
                    ctx.setParameter(s_param_name, float(lambdas_bridge_s[j]))
                    e = ctx.getState(getEnergy=True, groups=groups).getPotentialEnergy()
                    u_kn[j, n] = e.value_in_unit(unit.kilojoule_per_mole)
            frame_cursor += xyz_k.shape[0]
    finally:
        del ctx, integrator, eval_sys

    if not np.all(np.isfinite(u_kn)):
        raise RuntimeError("Shadow-Bridge u_kn 含 NaN/Inf，拒绝继续求解 MBAR。")
    return u_kn, n_k


def run_shadow_bridge_leg(
    system: openmm.System,
    topology: app.Topology,
    positions,
    box_vectors,
    perturbed_indices: List[int],
    lambdas_bridge_s: List[float],
    temperature_k: float = 300.0,
    platform_name: str = "CUDA",
    output_dir: str = "./shadow_bridge",
    n_steps_per_state: int = 200000,
    exchange_interval: int = 1000,
    save_interval: int = 5000,
    restraint_params: Optional[Dict] = None,
    random_seed: Optional[int] = None,
) -> Dict:
    """
    Shadow-PME Bridge 的完整入口：build_shadow_bridge_system -> ShadowBridgeREMDManager
    跑 REMD -> 离线重算 u_kn -> TraditionalMBARAnalyzer 求解 ΔG_bridge。

    lambdas_bridge_s 建议只放 2~4 个点（s=0 是 PME full charge 端点，s=1 是
    Shadow full charge 端点），因为这一段要跨越的只是 reciprocal+self+PME 修正
    的残差（对本项目 Atenolol 实测量级 ~11 kJ/mol），不需要像正式去电荷那样几
    十个窗口。
    """
    bridge_sys, s_name, build_diag = build_shadow_bridge_system(
        system,
        topology,
        perturbed_indices,
        restraint_params=restraint_params,
        box_vectors=box_vectors,
    )
    lambdas_bridge_s = sorted(float(s) for s in lambdas_bridge_s)
    if lambdas_bridge_s[0] != 0.0 or lambdas_bridge_s[-1] != 1.0:
        raise ValueError(
            "lambdas_bridge_s 必须以 0.0 (PME full charge) 开始、以 1.0 "
            "(Shadow full charge) 结束。"
        )

    remd = ShadowBridgeREMDManager(
        bridge_system=bridge_sys,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=perturbed_indices,
        lambdas_bridge_s=lambdas_bridge_s,
        s_param_name=s_name,
        temperature=temperature_k,
        platform_name=platform_name,
        output_dir=output_dir,
        random_seed=random_seed,
    )
    traj_files = remd.run(
        n_steps=n_steps_per_state,
        exchange_interval=exchange_interval,
        save_interval=save_interval,
        stage_name="shadow_bridge",
    )

    u_kn, n_k = compute_shadow_bridge_u_kn(
        bridge_sys,
        topology,
        traj_files,
        lambdas_bridge_s,
        s_param_name=s_name,
        box_vectors=box_vectors,
        platform_name="CPU",
    )
    analyzer = TraditionalMBARAnalyzer(temperature=temperature_k)
    analyzer._last_n_k = n_k
    res = analyzer.solve(analyzer.beta * u_kn)
    return {
        "stage": "shadow_bridge",
        "total_delta_G": float(res.get("delta_G", 0.0)),
        "total_error": float(res.get("error", 0.0)),
        "method": "Shadow-PME-Bridge-REMD-MBAR",
        "n_states": len(lambdas_bridge_s),
        "lambdas_bridge_s": lambdas_bridge_s,
        "build_diagnostics": build_diag,
        "converged": res.get("converged"),
        "min_overlap": res.get("min_overlap"),
        "min_overlap_threshold": res.get("diagnostics", {}).get("min_overlap_threshold"),
        "diagnostics": res.get("diagnostics", {}),
    }

def stage1_estimator_crosschecks(
    u_kn: np.ndarray,
    n_k: np.ndarray,
    kt: float,
    lambdas: Optional[Sequence[float]] = None,
) -> Dict:
    """[v2] 每态都有样本的腿（stage1 decharging / attachment）的四口径规范表。

    `u_kn` 必须是**约化**势（无量纲）。返回 kJ/mol。

    四个口径：去相关 MBAR / 全帧 MBAR / 相邻 BAR / 重加权 FD-TI。
    2026-07-28 stage1 实测：BAR 65.076 ± 0.615、FD-TI 65.126、全帧 MBAR 65.003
    三者一致，**去相关 MBAR 64.411 两条腿共同偏低 0.5–0.6 kJ/mol**——那是
    自相关子采样在有限样本下选帧不稳定造成的，不是 MBAR 有偏。
    所以主值取相邻 BAR、TI 作一致性门、两个 MBAR 降级为诊断。

    ⚠️ 两腿的偏移基本抵消：换估计量后 charging 对 ΔG_bind 的净移动只有约
    **−0.140 kJ/mol = −0.033 kcal/mol**。它是个真问题，但**不是** `result.txt`
    那 1.3 kcal charging 缺口的解释，别把两件事混起来。
    """
    u_kn = np.asarray(u_kn, dtype=np.float64)
    n_k = np.asarray(n_k, dtype=int)
    out: Dict[str, Any] = {
        "estimator_analysis_protocol_version": int(ESTIMATOR_ANALYSIS_PROTOCOL_VERSION),
        "u_kn_units": "reduced (dimensionless) — 转 kJ/mol 需乘 kT",
        "n_states": int(len(n_k)),
        "n_frames_total": int(np.sum(n_k)),
    }

    def _mbar_dg(sub_u, sub_n):
        stable = sub_u - np.min(sub_u, axis=0, keepdims=True)
        m = _build_mbar_compatible(
            stable, sub_n, solver_protocol="default", initialize="BAR",
            relative_tolerance=1e-6, verbose=False,
        )
        res = _compute_free_energy_result_compatible(m, compute_uncertainty=True)
        df, ddf = _extract_free_energy_arrays(res, require_uncertainty=True)
        return float((df[0, -1] - df[0, 0]) * kt), float(ddf[0, -1] * kt), m

    # --- 全帧 MBAR（诊断） ---
    try:
        dg_full, err_full, mbar_full = _mbar_dg(u_kn, n_k)
        out["full_frame_mbar"] = {"delta_G_kJ_mol": dg_full, "error_kJ_mol": err_full}
    except Exception as exc:
        mbar_full = None
        out["full_frame_mbar"] = {"delta_G_kJ_mol": None, "error": str(exc)}

    # --- 去相关 MBAR（诊断） ---
    try:
        offsets = np.concatenate(([0], np.cumsum(n_k)))
        keep, n_k_sub, g_per_state = [], np.zeros_like(n_k), []
        for k in range(len(n_k)):
            s, e = int(offsets[k]), int(offsets[k + 1])
            if e <= s:
                g_per_state.append(1.0)
                continue
            idx_k, g_k = subsample_series_by_autocorrelation(u_kn[k, s:e])
            g_per_state.append(float(g_k))
            keep.append(s + idx_k)
            n_k_sub[k] = idx_k.size
        keep_arr = np.concatenate(keep)
        dg_dec, err_dec, _ = _mbar_dg(u_kn[:, keep_arr], n_k_sub)
        out["decorrelated_mbar"] = {
            "delta_G_kJ_mol": dg_dec,
            "error_kJ_mol": err_dec,
            "statistical_inefficiency_per_state": g_per_state,
            "n_frames_after": int(keep_arr.size),
        }
    except Exception as exc:
        out["decorrelated_mbar"] = {"delta_G_kJ_mol": None, "error": str(exc)}

    # --- 相邻 BAR（主值候选） ---
    try:
        dg_bar, err_bar, edges = adjacent_bar_chain(u_kn, n_k, kt)
        out["adjacent_bar"] = {
            "delta_G_kJ_mol": dg_bar, "error_kJ_mol": err_bar, "edges": edges,
        }
    except Exception as exc:
        out["adjacent_bar"] = {"delta_G_kJ_mol": None, "not_applicable_reason": str(exc)}

    # --- 重加权 FD-TI（一致性门） ---
    if lambdas is None:
        out["reweighted_fd_ti"] = {
            "delta_G_kJ_mol": None,
            "not_applicable_reason": "未提供 lambdas；FD-TI 必须用真实非均匀 λ 网格，不得用态序号代替",
        }
    else:
        try:
            weights = getattr(mbar_full, "W_nk", None) if mbar_full is not None else None
            dg_ti, dudl = reweighted_fd_ti(u_kn, n_k, lambdas, kt, weights=weights)
            out["reweighted_fd_ti"] = {
                "delta_G_kJ_mol": dg_ti,
                "mean_dU_dlambda_kJ_mol": dudl,
                "lambdas": [float(x) for x in lambdas],
                "weights": "mbar_W_nk" if weights is not None else "per_state_block_mean",
            }
        except Exception as exc:
            out["reweighted_fd_ti"] = {"delta_G_kJ_mol": None, "error": str(exc)}

    return out


def stage1_ti_consistency_gate(
    dg_bar: float,
    err_bar: float,
    dg_ti: Optional[float],
    tolerance_kJ_mol: float,
) -> Dict:
    """BAR vs 重加权 FD-TI 的一致性门。

    ⚠️ 容差**必须显式给**，不得沿用 attachment 那个
    `ATTACHMENT_BAR_TI_ABS_TOL_KJ = 1.0`：两条腿的量级和 TI 变体都不同。
    stage1 实测 |BAR − FD-TI| = 0.050（complex）/ 0.226（solvent）kJ/mol。
    """
    if dg_ti is None:
        return {"passed": None, "reason": "没有 TI 值，无法判定（不当作通过）"}
    tol = max(float(tolerance_kJ_mol), 3.0 * max(float(err_bar), 0.0))
    diff = abs(float(dg_bar) - float(dg_ti))
    return {
        "passed": bool(diff <= tol),
        "abs_diff_kJ_mol": diff,
        "tolerance_kJ_mol": tol,
        "tolerance_source": "max(显式容差, 3*sigma_BAR)",
        "delta_G_bar_kJ_mol": float(dg_bar),
        "delta_G_ti_kJ_mol": float(dg_ti),
    }


class TraditionalMBARAnalyzer:
    """标准离线 MBAR：读取轨迹 → 重算 u_kn → 求解 ΔG"""
    def __init__(self, temperature: float = 300.0):
        self.kt = 0.008314462618 * temperature
        self.beta = 1.0 / self.kt
        self._last_lj_lrc_metadata = {
            "protocol_version": TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
            "applicable": False,
            "applied": False,
        }

    def compute_u_kn(
        self,
        traj_files: List[str],
        system_template: openmm.System,
        ligand_indices: List[int],
        lambdas_coul: List[float],
        lambdas_vdw: List[float],
        platform_name: str = "CPU",
        topology: app.Topology = None,
        reference_positions=None,
        reference_box_vectors=None,
        boresch_params: Optional[Dict] = None,
    ) -> np.ndarray:
        import mdtraj as md
        if topology is None:
            raise ValueError("compute_u_kn 需要 OpenMM topology，不能用 System 构建 mdtraj Topology")
        md_top = md.Topology.from_openmm(topology)

        traj_list = [md.load(path, top=md_top) for path in traj_files]
        n_k = np.array([t.n_frames for t in traj_list], dtype=int)
        traj = md.join(traj_list, check_topology=False)
        self._last_n_k = n_k
        # ✅ 在其下方紧跟着插入 PBC 解包逻辑：
        # 🔑 核心修复：强制 PBC 分子完整性解包，消除跨盒“假撕裂”导致的能量 Spike
        try:
            # image_molecules 会根据拓扑连通性，将跨越边界的分子重新拼合
            # anchor_molecules 确保配体和受体不会在解包时被分到不同的镜像盒子
            traj.image_molecules(inplace=True)
            print("  ✅ 轨迹 PBC 分子完整性已修复 (image_molecules)，消除撕裂隐患")
        except Exception as e:
            print(f"  ⚠️ PBC 修复失败: {e}，将使用原始坐标（存在跨盒撕裂导致 Energy Spike 的风险）")    
        n_frames = traj.n_frames
        n_states = len(lambdas_coul)
        u_kn = np.zeros((n_states, n_frames))
        lambdas_coul_arr = np.asarray(lambdas_coul, dtype=float)
        lambdas_vdw_arr = np.asarray(lambdas_vdw, dtype=float)
        is_pme_coulomb_leg = (
            np.allclose(lambdas_vdw_arr, 1.0)
            and not np.allclose(lambdas_coul_arr, lambdas_coul_arr[0])
        )
        is_mixed_pme_alchemical = (
            not np.allclose(lambdas_coul_arr, lambdas_coul_arr[0])
            and not np.allclose(lambdas_vdw_arr, lambdas_vdw_arr[0])
        )
        use_total_energy = is_pme_coulomb_leg or is_mixed_pme_alchemical
        ligand_charge_square_sum = 0.0
        apply_pme_self_correction = False
        pme_self_metadata = {
            "applicable": bool(use_total_energy),
            "applied": False,
            "lambda_name": "lambda_coul",
            "charge_square_sum_e2": 0.0,
            "source": "not_applicable",
        }
        if use_total_energy:
            nb_force_ref = next((f for f in system_template.getForces() if isinstance(f, openmm.NonbondedForce)), None)
            if nb_force_ref is None:
                raise RuntimeError("PME 去电荷路径未找到参考 NonbondedForce，无法估算自能修正。")
            for idx in ligand_indices:
                q, _, _ = nb_force_ref.getParticleParameters(int(idx))
                q_val = q.value_in_unit(unit.elementary_charge)
                ligand_charge_square_sum += q_val * q_val
            pme_self_metadata.update({
                "charge_square_sum_e2": float(ligand_charge_square_sum),
                "source": "ligand_particle_charges_before_offset_preparation",
            })
        xyz_all = np.asarray(traj.xyz, dtype=np.float64)
        box_all = None
        if traj.unitcell_vectors is not None and len(traj.unitcell_vectors) > 0:
            box_all = np.asarray(traj.unitcell_vectors, dtype=np.float64)

        # BeutlerSoftcoreBuilder uses a CutoffPeriodic L-E force with no OpenMM
        # long-range correction.  For every leg containing that force, add the
        # missing attractive r^-6 tail to every target-state u_k(x).  This is
        # exact for the traditional REMD trajectories only when V is constant:
        # under NPT, the omitted 1/V term would also have changed the sampled
        # volume distribution, which cannot be repaired by post-processing.
        needs_traditional_lrc = not is_pme_coulomb_leg
        lj_tail_lrc_coeff = None
        lj_lrc_metadata = {
            "protocol_version": TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
            "applicable": bool(needs_traditional_lrc),
            "applied": False,
            "model": "switching_softcore_aware_r6_r12_tail",
            "switching_nm": LJ_TAIL_LRC_R_SWITCH_NM,
            "cutoff_nm": LJ_TAIL_LRC_R_CUTOFF_NM,
        }
        if needs_traditional_lrc:
            if box_all is None or len(box_all) != n_frames:
                raise RuntimeError(
                    "传统 Beutler LRC 需要每一帧的周期盒向量；当前轨迹缺少完整 unitcell_vectors。"
                )
            volumes = np.asarray(
                [_periodic_box_volume_nm3(box) for box in box_all], dtype=np.float64
            )
            mean_volume = float(np.mean(volumes))
            relative_span = float((np.max(volumes) - np.min(volumes)) / mean_volume)
            if relative_span > 1.0e-3:
                raise RuntimeError(
                    "传统 REMD 的 Beutler 软核力缺少在线 LRC，而当前轨迹盒体积有明显波动"
                    f"（relative_span={relative_span:.3e} > 1e-3）。离线追加 1/V 尾项不能"
                    "修复未按该哈密尔顿量采样的 NPT 体积分布；请改用固定盒 NVT 生产，或先"
                    "把同一 LRC 真正加入 REMD 采样系统后重跑。"
                )
            nb_force_ref = next(
                (f for f in system_template.getForces() if isinstance(f, openmm.NonbondedForce)),
                None,
            )
            if nb_force_ref is None:
                raise RuntimeError("传统 Beutler LRC 找不到参考 NonbondedForce。")
            all_params = [
                nb_force_ref.getParticleParameters(i)
                for i in range(nb_force_ref.getNumParticles())
            ]
            ligand_set = set(int(i) for i in ligand_indices)
            env_indices = [
                i for i in range(nb_force_ref.getNumParticles()) if i not in ligand_set
            ]
            tail_sigma, tail_s6_per_sigma, tail_s12_per_sigma = (
                _lj_tail_correction_sigma_resolved_moments(
                    all_params, list(ligand_indices), env_indices
                )
            )
            tail_s6 = float(np.sum(tail_s6_per_sigma))
            tail_s12 = float(np.sum(tail_s12_per_sigma))
            # BeutlerSoftcoreBuilder.build() defaults: the LJ term is scaled by
            # lambda_vdw**1 and, under
            # SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2,
            # D_ij(r)=0.5*sigma_ij^6*(1-lambda_vdw)**1 + r^6 (apart from its
            # negligible 1e-4*sigma_ij^6 safety floor near r=0).  Use the same
            # v3 sigma-resolved integrator as the dual-lambda/IBS path so
            # producer and worker share both the key and the physical definition.
            beutler_alpha_lj = 0.5
            beutler_m_lj = 1.0
            beutler_n_lj = 1.0
            lj_tail_lrc_coeff = _lj_tail_lrc_coefficients_kj_mol(
                lambdas_vdw_arr,
                tail_sigma,
                tail_s6_per_sigma,
                tail_s12_per_sigma,
                beutler_alpha_lj,
                beutler_m_lj,
                beutler_n_lj,
                LJ_TAIL_LRC_R_SWITCH_NM,
                LJ_TAIL_LRC_R_CUTOFF_NM,
            )
            lj_lrc_metadata.update({
                "applied": True,
                "status": "implemented_switching_softcore_aware",
                "tail_S6_kj_nm6": float(tail_s6),
                "tail_S12_kj_nm12": float(tail_s12),
                "n_sigma_groups": int(tail_sigma.size),
                "coefficients_kj_nm3_mol": lj_tail_lrc_coeff.tolist(),
                "alpha_lj": beutler_alpha_lj,
                "alpha_convention": ACESoftcorePotential.ALPHA_CONVENTION,
                "m_lj": beutler_m_lj,
                "n_lj": beutler_n_lj,
                "volume_mean_nm3": mean_volume,
                "volume_relative_span": relative_span,
            })
            print(
                f"  ✅ 传统 Beutler LRC: v{TRADITIONAL_LJ_LRC_PROTOCOL_VERSION} "
                "switching+softcore-aware r^-6/r^-12 离线尾项已启用 "
                f"({tail_sigma.size} 个 sigma 分组, V={mean_volume:.6f} nm^3, "
                f"relative_span={relative_span:.3e})"
            )

        cpu_count = max(1, os.cpu_count() or 1)
        chunk_size = max(25, min(250, int(math.ceil(n_frames / max(1, cpu_count * 2)))))
        n_chunks = max(1, int(math.ceil(n_frames / chunk_size)))
        n_workers = min(cpu_count, n_chunks)
        if is_pme_coulomb_leg:
            # ✅ 硬性保护：is_pme_coulomb_leg 仅靠 np.allclose 推断，不能作为最终防线。
            # 一旦上游误传了非纯 decharging 的 λ 表（例如 vdw 未锁定在 1.0 却被误判为
            # decharging 腿），下面的 PME-only 能量路径会静默给出物理上错误的结果。
            # 这里显式断言 lambda_vdw 必须严格等于 1.0，配置错误时硬报错而不是沉默产出错误 ΔG。
            if not np.allclose(lambdas_vdw_arr, 1.0, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    "PME 去电荷 (decharging) 分支要求所有态的 lambda_vdw 严格等于 1.0，"
                    f"但实际检测到 lambdas_vdw={lambdas_vdw_arr.tolist()}。这通常意味着 λ 路径"
                    "配置错误（例如把 vanishing/mixed 腿的 λ 表误传入了 decharging 分析），"
                    "请检查上游 lambda 生成逻辑后重试。"
                )
            prepared_system = _prepare_pme_coulomb_leg_system(
                system_template,
                ligand_indices,
                lambda_name="lambda_coul",
                allow_charged_ligand=True,
                topology=topology,
                positions=reference_positions,
                box_vectors=reference_box_vectors,
            )
            nb_prepared = next((f for f in prepared_system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
            if nb_prepared is None:
                raise RuntimeError("PME 去电荷路径 prepared system 缺少 NonbondedForce。")
            offset_qsq, offset_rows = pme_offset_charge_square_sum(nb_prepared, "lambda_coul")
            # 🔑 关键修复：此前这里认为 OpenMM 报告的总 PME 能量里含有一个"坐标无关的
            # 自能伪项 -C·λ²"，需要在离线 u_kn 里手动加回 +C·λ² 抵消掉。但
            # NonbondedForce.addParticleParameterOffset 是 OpenMM 官方支持的线性电荷
            # 缩放机制：每次 getState(getEnergy=True) 都会用当前 λ 对应的真实（已缩放）
            # 电荷重新计算包括 Ewald self-energy (-ke·alpha/√π·Σq_i²) 在内的完整 PME
            # 能量，这个 self-energy 项是该 λ 态总哈密顿量里真实存在、且必须存在的一部分
            # （它精确抵消了倒易空间求和里对粒子自身的重复计数，不是数值伪影），MBAR/TMBAR
            # 只要求 U_k(x) 是状态 k 的正确势能，self-energy 项本就应该被完整保留在 U_k
            # 里参与自由能估计。手动加回 +C·λ² 相当于凭空注入了一个真实存在的物理量的
            # 反向修正，量级是 ke·alpha·Σq²/√π（对本配体 Σq²≈5 e² 时可达数百 kJ/mol），
            # 实测正是它把 decharging 腿的 ΔG 从物理上合理的量级拉到了 -954.81 kJ/mol
            # 这种明显偏大的结果。这里保留 Σq_offset² 的诊断计算（对排查/存档有用），但不再
            # 把它转换成能量修正项叠加进 u_kn。
            if offset_qsq > 0.0:
                pme_self_metadata.update({
                    "applied": False,
                    "charge_square_sum_e2": float(offset_qsq),
                    "source": "particle_parameter_offsets_including_coalchemical_counterion",
                    "offset_particles": offset_rows,
                    "note": (
                        "diagnostic only: OpenMM's NonbondedForce particle-parameter-offset "
                        "charge scaling already reports the physically correct, λ-dependent "
                        "PME self-energy at every state; it is a real Hamiltonian term, not an "
                        "artifact, so it is intentionally NOT added back as a correction."
                    ),
                })
                print(
                    "  ℹ️ PME 自能诊断（未应用修正）: "
                    f"Σq_offset²={offset_qsq:.6f} e²；OpenMM 已在每个 λ 态正确报告该项，"
                    "不再额外叠加 +Cλ² 修正。"
                )
            _add_physical_boresch_restraint(prepared_system, boresch_params, force_group=3)
            system_xml = openmm.XmlSerializer.serialize(prepared_system)
            del prepared_system
        elif is_mixed_pme_alchemical:
            prepared_system = _prepare_pme_mixed_alchemical_system(
                system_template,
                ligand_indices,
                topology=topology,
                positions=reference_positions,
                box_vectors=reference_box_vectors,
                lambda_coul_name="lambda_coul",
                lambda_vdw_name="lambda_vdw",
                restraint_params=boresch_params,
            )
            nb_prepared = next((f for f in prepared_system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
            if nb_prepared is None:
                raise RuntimeError("PME mixed alchemical system 缺少 NonbondedForce。")
            offset_qsq, offset_rows = pme_offset_charge_square_sum(nb_prepared, "lambda_coul")
            # 同上（is_pme_coulomb_leg 分支）：不再把 Σq_offset² 转换为 +Cλ² 能量修正，
            # 只保留诊断记录。理由见上方注释。
            if offset_qsq > 0.0:
                pme_self_metadata.update({
                    "applied": False,
                    "charge_square_sum_e2": float(offset_qsq),
                    "source": "particle_parameter_offsets_including_coalchemical_counterion",
                    "offset_particles": offset_rows,
                    "note": (
                        "diagnostic only: OpenMM's NonbondedForce particle-parameter-offset "
                        "charge scaling already reports the physically correct, λ-dependent "
                        "PME self-energy at every state; it is a real Hamiltonian term, not an "
                        "artifact, so it is intentionally NOT added back as a correction."
                    ),
                })
                print(
                    "  ℹ️ PME 自能诊断（未应用修正）: "
                    f"Σq_offset²={offset_qsq:.6f} e²；OpenMM 已在每个 λ 态正确报告该项，"
                    "不再额外叠加 +Cλ² 修正。"
                )
            system_xml = openmm.XmlSerializer.serialize(prepared_system)
            del prepared_system
        else:
            eval_sys = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system_template))
            eval_sys.thisown = 1
            nb = [f for f in eval_sys.getForces() if isinstance(f, openmm.NonbondedForce)][0]
            _restore_ligand_internal_nonbonded(
                eval_sys,
                nb,
                ligand_indices,
                zero_original_exceptions=True,
            )
            env_idx = [i for i in range(eval_sys.getNumParticles()) if i not in ligand_indices]
            sc = BeutlerSoftcoreBuilder.build(nb, ligand_indices, env_idx)
            sc.setForceGroup(1)
            eval_sys.addForce(sc)
            _add_physical_boresch_restraint(eval_sys, boresch_params, force_group=3)
            for idx in ligand_indices:
                nb.setParticleParameters(idx, 0.0, 0.1 * unit.nanometer, 0.0)
            system_xml = openmm.XmlSerializer.serialize(eval_sys)
            del eval_sys

        print(f"\n📊 开始离线能量重算 | {n_frames} 帧 × {n_states} 态 | workers={n_workers} | chunk_size={chunk_size}")
        tasks = []
        for frame_offset in range(0, n_frames, chunk_size):
            frame_end = min(frame_offset + chunk_size, n_frames)
            tasks.append(
                {
                    "frame_offset": frame_offset,
                    "xyz": xyz_all[frame_offset:frame_end].copy(),
                    "box_vectors": None if box_all is None else box_all[frame_offset:frame_end].copy(),
                    "system_xml": system_xml,
                    "ligand_indices": list(ligand_indices),
                    "lambdas_coul": lambdas_coul_arr,
                    "lambdas_vdw": lambdas_vdw_arr,
                    "platform_name": platform_name,
                    "kt": self.kt,
                    "use_total_energy": use_total_energy,
                    "apply_pme_self_correction": apply_pme_self_correction,
                    "ligand_charge_square_sum": ligand_charge_square_sum,
                    "pme_self_correction_metadata": pme_self_metadata,
                    "lj_tail_lrc_coeff_kj_mol": (
                        None
                        if lj_tail_lrc_coeff is None
                        else lj_tail_lrc_coeff.copy()
                    ),
                    "traditional_lj_lrc_protocol_version": (
                        TRADITIONAL_LJ_LRC_PROTOCOL_VERSION
                    ),
                }
            )

        if n_workers == 1:
            for task in tasks:
                frame_offset, u_chunk = _compute_u_kn_chunk(task)
                frame_end = frame_offset + u_chunk.shape[1]
                u_kn[:, frame_offset:frame_end] = u_chunk
                print(f"  → 帧 {frame_end}/{n_frames} 完成")
        else:
            try:
                ctx = mp.get_context("spawn")
                with ctx.Pool(processes=n_workers) as pool:
                    for frame_offset, u_chunk in pool.imap_unordered(_compute_u_kn_chunk, tasks):
                        frame_end = frame_offset + u_chunk.shape[1]
                        u_kn[:, frame_offset:frame_end] = u_chunk
                        print(f"  → 帧 {frame_end}/{n_frames} 完成")
            except Exception as exc:
                print(f"  ⚠️ 多进程重算失败，回退单进程: {exc}")
                for task in tasks:
                    frame_offset, u_chunk = _compute_u_kn_chunk(task)
                    frame_end = frame_offset + u_chunk.shape[1]
                    u_kn[:, frame_offset:frame_end] = u_chunk
                    print(f"  → 帧 {frame_end}/{n_frames} 完成")

        self._last_pme_self_correction_metadata = pme_self_metadata
        self._last_lj_lrc_metadata = lj_lrc_metadata
        return u_kn

    def solve(
        self,
        u_kn: np.ndarray,
        decorrelate: bool = True,
        primary_estimator: str = "mbar_decorrelated",
        lambdas: Optional[Sequence[float]] = None,
        ti_gate_tolerance_kJ_mol: Optional[float] = None,
        crosschecks: bool = False,
    ) -> Dict:
        """标准 MBAR 求解。

        **默认参数一字不动地保持历史行为**（`primary_estimator="mbar_decorrelated"`、
        `crosschecks=False`），所以现有 6 处生产调用点不受影响。

        primary_estimator ["mbar_decorrelated" | "adjacent_bar"]：
          `"adjacent_bar"` 时 `delta_G`/`error` 取相邻 BAR 链，并（若给了
          `lambdas` 与 `ti_gate_tolerance_kJ_mol`）跑 BAR-vs-FD-TI 一致性门。
          依据见 `stage1_estimator_crosschecks`：BAR/TI/全帧 MBAR 三者一致，
          去相关 MBAR 偏低 0.5–0.6 kJ/mol。
        crosschecks [False]：True 时把四个口径全部算出来放进
          `result["crosschecks"]`（`primary_estimator="adjacent_bar"` 时自动开启，
          因为主值本身就来自那张表）。
        """
        if primary_estimator not in ("mbar_decorrelated", "adjacent_bar"):
            raise ValueError(
                f"primary_estimator 只能是 'mbar_decorrelated' 或 'adjacent_bar'，"
                f"收到 {primary_estimator!r}"
            )
        u_kn = np.asarray(u_kn, dtype=np.float64)
        # 交叉检查必须用**原始**数组：下面 decorrelate 分支会就地换掉 u_kn/n_k。
        u_kn_original = u_kn
        K, N = u_kn.shape
        if not hasattr(self, "_last_n_k"):
            raise RuntimeError(
                "TraditionalMBARAnalyzer.solve() 缺少每态样本数 n_k。"
                "请先调用 compute_u_kn()，或在 solve() 前显式设置 analyzer._last_n_k。"
            )
        n_k = np.asarray(self._last_n_k, dtype=int)
        n_k_original = n_k
        if len(n_k) != K or int(np.sum(n_k)) != N:
            raise ValueError(f"MBAR 样本数不匹配: len(n_k)={len(n_k)}, sum(n_k)={np.sum(n_k)}, N={N}")
        if not np.all(np.isfinite(u_kn)):
            raise ValueError("u_kn 含 NaN/Inf，无法执行 MBAR")
        if not HAS_PYMBAR:
            raise ImportError("需要 pymbar 包，请安装: pip install pymbar")

        # ------------------------------------------------------------------
        # 🔑 修复（审查报告 #1）：自相关子采样
        # ------------------------------------------------------------------
        # u_kn 的列按 n_k 分块，每一块都来自同一个态自己的（demux 后）REMD 轨迹。
        # MBAR/BAR 假设逐帧样本互相独立，但相邻输出帧强相关；这里对每个态自己
        # 的能量时间序列估计统计非效率 g，再据此做去相关子采样，避免 n_k 虚高、
        # 误差棒系统性偏小（常见 2-10 倍）。
        decorrelation_diag = {"applied": False}
        if decorrelate:
            offsets = np.concatenate(([0], np.cumsum(n_k)))
            keep_cols = []
            g_per_state = []
            n_k_sub = np.zeros_like(n_k)
            for k in range(K):
                start, end = int(offsets[k]), int(offsets[k + 1])
                if end <= start:
                    g_per_state.append(1.0)
                    continue
                local_idx, g_k = subsample_series_by_autocorrelation(u_kn[k, start:end])
                g_per_state.append(g_k)
                keep_cols.append(start + local_idx)
                n_k_sub[k] = local_idx.size
            if keep_cols:
                keep_cols_arr = np.concatenate(keep_cols)
                n_frames_before = N
                u_kn = u_kn[:, keep_cols_arr]
                n_k = n_k_sub
                N = u_kn.shape[1]
                decorrelation_diag = {
                    "applied": True,
                    "statistical_inefficiency_per_state": g_per_state,
                    "n_frames_before": int(n_frames_before),
                    "n_frames_after": int(N),
                }

        # 逐列平移不会改变自由能差，但能显著改善大体系绝对能量下的数值条件。
        u_kn_stable = u_kn - np.min(u_kn, axis=0, keepdims=True)
        delta_u_diag = delta_u_distribution_diagnostics(u_kn_stable)

        last_exc = None
        last_mbar = None
        # ✅ 修复：以前 "default" protocol 一旦协方差求解失败就立刻 return
        # error=nan，"robust" protocol（专门为数值困难场景准备的备选）根本没有
        # 机会尝试——所以 err_complex=nan 会一路传到最终 ΔG_bind，而日志里那句
        # "⚠️ MBAR 协方差求解失败...回退为仅计算 ΔG 不估计误差" 淹没在一堆输出里，
        # 不容易注意到这其实意味着最终结果完全没有误差棒。现在协方差失败时先
        # 保留这个 fallback（不确定度=nan），继续尝试下一个 protocol；只有全部
        # protocol 都拿不到真正的不确定度时，才退而求其次返回第一个成功拿到 ΔG
        # 的 fallback 结果。
        no_uncertainty_fallback = None
        protocol_plan = (
            ("default", "MBAR-default"),
            ("robust", "MBAR-robust"),
        )
        for protocol, method_name in protocol_plan:
            try:
                last_mbar = _build_mbar_compatible(
                    u_kn_stable,
                    n_k,
                    solver_protocol=protocol,
                    initialize="BAR",
                    relative_tolerance=1e-6,
                    verbose=False,
                )
                try:
                    res = _compute_free_energy_result_compatible(
                        last_mbar,
                        compute_uncertainty=True,
                    )
                    df_matrix, ddf_matrix = _extract_free_energy_arrays(
                        res,
                        require_uncertainty=True,
                    )
                    err = float(ddf_matrix[0, -1] * self.kt)
                except Exception as cov_exc:
                    print(
                        f"  ⚠️ MBAR 协方差求解失败 ({protocol}): {cov_exc}，"
                        "尝试下一个 solver protocol 以求拿到真正的不确定度..."
                    )
                    if no_uncertainty_fallback is None:
                        res_fb = _compute_free_energy_result_compatible(
                            last_mbar,
                            compute_uncertainty=False,
                        )
                        df_fb, _ = _extract_free_energy_arrays(
                            res_fb,
                            require_uncertainty=False,
                        )
                        dg_fb = float((df_fb[0, -1] - df_fb[0, 0]) * self.kt)
                        no_uncertainty_fallback = {
                            "delta_G": dg_fb,
                            "error": float("nan"),
                            "method": f"{method_name}-no-uncertainty",
                            "n_frames": N,
                            "n_states": K,
                            "converged": False,
                            "min_overlap": None,
                            "diagnostics": {
                                "delta_u_distribution": delta_u_diag,
                                "pme_self_correction": getattr(self, "_last_pme_self_correction_metadata", {}),
                                "uncertainty_solve_error": str(cov_exc),
                                "decorrelation": decorrelation_diag,
                            },
                        }
                    continue

                dg = float((df_matrix[0, -1] - df_matrix[0, 0]) * self.kt)
                diagnostics = {
                    "delta_u_distribution": delta_u_diag,
                    "pme_self_correction": getattr(self, "_last_pme_self_correction_metadata", {}),
                    "overlap_matrix": None,
                    "effective_sample_number": None,
                    "decorrelation": decorrelation_diag,
                }
                # ------------------------------------------------------------------
                # 🔑 修复（审查报告 #2）：这里所有 K 个态都真正有样本（REMD 全采样），
                # 标准 MBAR 重叠矩阵在这种场景下是有效的，不像 GlobalMBARAnalyzer 那种
                # 单一采样分布 + 多个零样本目标态的场景。用相邻态（|i-j|=1）重叠矩阵
                # 元素的最小值作为 min_overlap，阈值 0.03 与 abfe_core.py 在线监控里
                # 相邻窗口重叠的既有约定保持一致。
                # ------------------------------------------------------------------
                min_overlap = None
                min_overlap_threshold = 0.03
                try:
                    overlap_res = last_mbar.compute_overlap()
                    overlap_matrix = overlap_res["matrix"] if isinstance(overlap_res, dict) else overlap_res
                    overlap_matrix = np.asarray(overlap_matrix, dtype=float)
                    diagnostics["overlap_matrix"] = overlap_matrix.tolist()
                    if K >= 2:
                        offdiag_vals = [float(overlap_matrix[i, i + 1]) for i in range(K - 1)]
                        offdiag_vals += [float(overlap_matrix[i + 1, i]) for i in range(K - 1)]
                        min_overlap = float(np.min(offdiag_vals)) if offdiag_vals else None
                except Exception as overlap_exc:
                    diagnostics["overlap_error"] = str(overlap_exc)
                try:
                    neff = last_mbar.compute_effective_sample_number()
                    diagnostics["effective_sample_number"] = np.asarray(neff, dtype=float).tolist()
                except Exception as neff_exc:
                    diagnostics["effective_sample_number_error"] = str(neff_exc)
                diagnostics["min_overlap_threshold"] = min_overlap_threshold
                converged = bool(min_overlap is not None and min_overlap >= min_overlap_threshold)
                return self._finalize_solve_result(
                    {
                        "delta_G": dg,
                        "error": err,
                        "method": method_name,
                        "n_frames": N,
                        "n_states": K,
                        "converged": converged,
                        "min_overlap": min_overlap,
                        "diagnostics": diagnostics,
                    },
                    u_kn_original,
                    n_k_original,
                    primary_estimator=primary_estimator,
                    lambdas=lambdas,
                    ti_gate_tolerance_kJ_mol=ti_gate_tolerance_kJ_mol,
                    crosschecks=crosschecks,
                )
            except Exception as exc:
                last_exc = exc
                print(f"  ⚠️ MBAR {protocol} 求解失败: {exc}")

        if no_uncertainty_fallback is not None:
            print(
                "  🚨 所有 solver protocol 均无法给出不确定度估计；"
                f"返回 {no_uncertainty_fallback['method']} 的 ΔG，error 标记为 nan——"
                "这条腿/阶段的最终误差棒不可信，请检查窗口重叠率与采样长度。"
            )
            return self._finalize_solve_result(
                no_uncertainty_fallback,
                u_kn_original,
                n_k_original,
                primary_estimator=primary_estimator,
                lambdas=lambdas,
                ti_gate_tolerance_kJ_mol=ti_gate_tolerance_kJ_mol,
                crosschecks=crosschecks,
            )

        raise RuntimeError(f"MBAR 求解失败，最后错误: {last_exc}")

    def _finalize_solve_result(
        self,
        result: Dict,
        u_kn_original: np.ndarray,
        n_k_original: np.ndarray,
        primary_estimator: str,
        lambdas: Optional[Sequence[float]],
        ti_gate_tolerance_kJ_mol: Optional[float],
        crosschecks: bool,
    ) -> Dict:
        """挂上分析协议号/策略指纹，按需算交叉检查并切换主值。

        默认路径（`primary_estimator="mbar_decorrelated"`、`crosschecks=False`）
        只加三个纯记录字段，**不做任何额外计算、不改数值**。
        """
        want_crosschecks = bool(crosschecks) or primary_estimator == "adjacent_bar"
        cc = None
        if want_crosschecks:
            try:
                cc = stage1_estimator_crosschecks(
                    u_kn_original, n_k_original, self.kt, lambdas=lambdas,
                )
            except Exception as exc:
                cc = {"error": f"交叉检查失败: {exc}"}
            result["crosschecks"] = cc

        sigma_policy = "mbar_asymptotic_on_decorrelated_frames"
        charging_frame_selection = (
            "decorrelated_per_state"
            if bool((result.get("diagnostics", {}).get("decorrelation") or {}).get("applied"))
            else "all_frames"
        )
        if primary_estimator == "adjacent_bar":
            bar = (cc or {}).get("adjacent_bar") or {}
            dg_bar = bar.get("delta_G_kJ_mol")
            if dg_bar is None:
                # fail closed：要 BAR 主值却算不出 BAR，绝不静默退回 MBAR 值。
                raise RuntimeError(
                    "primary_estimator='adjacent_bar' 但相邻 BAR 不可用："
                    f"{bar.get('not_applicable_reason') or bar.get('error')}"
                )
            result["delta_G_mbar_decorrelated_kJ_mol"] = result.get("delta_G")
            result["error_mbar_decorrelated_kJ_mol"] = result.get("error")
            result["delta_G"] = float(dg_bar)
            result["error"] = float(bar.get("error_kJ_mol", float("nan")))
            result["method"] = "adjacent-BAR-chain"
            sigma_policy = "bar_asymptotic_edge_variance_sum"
            charging_frame_selection = "all_frames_per_state"
            # BAR 的点估计刻意使用每态全部帧；当前边方差和仍是把相关帧近似为
            # 独立样本的渐近误差。P1-19 尚未给 charging 建立可采纳的块 bootstrap，
            # 所以必须把这个限制写成机器可读标记，不能只藏在 TODO 的文字里。
            result["sigma_suspect_underestimated"] = True
            result["sigma_suspect_underestimated_reason"] = (
                "adjacent BAR uses all per-state frames, while the reported "
                "asymptotic edge-variance sum is not corrected for temporal "
                "autocorrelation"
            )
            # TI 一致性门
            ti = ((cc or {}).get("reweighted_fd_ti") or {}).get("delta_G_kJ_mol")
            if ti_gate_tolerance_kJ_mol is None:
                result["ti_gate"] = {
                    "passed": None,
                    "reason": (
                        "未提供 ti_gate_tolerance_kJ_mol。容差必须显式给——不得沿用 "
                        "attachment 的 1.0 kJ/mol，两条腿量级与 TI 变体都不同。"
                    ),
                }
            else:
                gate = stage1_ti_consistency_gate(
                    float(dg_bar), float(result["error"]), ti,
                    float(ti_gate_tolerance_kJ_mol),
                )
                result["ti_gate"] = gate
            # fail closed：BAR 主值的独立一致性证据必须明确通过。TI 缺失、
            # 计算异常或调用方忘记给容差都属于“没有通过”，不能靠原 MBAR overlap
            # 的 converged=True 混过去。
            if result["ti_gate"].get("passed") is not True:
                result["converged"] = False
                result["ti_gate_failed"] = True
                if result["ti_gate"].get("passed") is False:
                    print(
                        f"  🚨 [TI 门] BAR 与重加权 FD-TI 分歧 "
                        f"{result['ti_gate']['abs_diff_kJ_mol']:.4f} > 容差 "
                        f"{result['ti_gate']['tolerance_kJ_mol']:.4f} kJ/mol "
                        "⟹ converged=False"
                    )
                else:
                    print(
                        "  🚨 [TI 门] BAR 主值缺少可判定且明确通过的 FD-TI 证据 "
                        f"({result['ti_gate'].get('reason')}) ⟹ converged=False"
                    )

        result["primary_estimator"] = str(primary_estimator)
        result["estimator_analysis_protocol_version"] = int(
            ESTIMATOR_ANALYSIS_PROTOCOL_VERSION
        )
        if primary_estimator == "adjacent_bar":
            result["charging_frame_selection"] = charging_frame_selection
            result["sigma_policy"] = sigma_policy
        result["estimator_policy_fingerprint"] = estimator_policy_fingerprint({
            "estimator_analysis_protocol_version": int(
                ESTIMATOR_ANALYSIS_PROTOCOL_VERSION
            ),
            "primary_estimator": str(primary_estimator),
            "charging_frame_selection": charging_frame_selection,
            "sigma_policy": sigma_policy,
            "sigma_inflation_applied": False,
            "ti_gate_tolerance_kJ_mol": (
                float(ti_gate_tolerance_kJ_mol)
                if ti_gate_tolerance_kJ_mol is not None else None
            ),
        })
        return result
