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
import inspect
import shutil
import functools
from contextlib import contextmanager
from scipy.integrate import quad as _scipy_quad
from typing import Dict, List, Tuple, Optional, Any, Sequence
from abfe_core import (
    ACESoftcorePotential,
    # §3.0 空腔填充迟滞：正反向 / 双起点 stage2 的 ΔF 差 <= 2σ。
    # 这个常量此前**只被写进 provenance、从未被任何代码执行过**；
    # endpoint_wet_dry_hysteresis_gate 是它的第一个真实执行点。
    # 绝不要在本文件另立一份同义常量——那会让"文档记录的阈值"和"实际生效的阈值"
    # 分叉，正是这个常量一开始变成死常量的同一类问题。
    STAGE2_HYSTERESIS_MAX_SIGMA,
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
    # attachment 腿起点体检（MEM-06）。两个都**复用**已有实现：
    # 受力门与预平衡共用一份；六个几何量走 BOR-01 之后唯一正确的那份计算
    # （手写第 5 份二面角副本正是 BOR-01 反号事故的成因）。
    assert_starting_state_is_sane,
    calc_boresch_from_last_frame,
    # MEM-00c：co-ion 身份冻结。选择只允许发生一次，之后所有入口只核对不重选。
    build_co_alchemical_ion_identity,
    verify_co_alchemical_ion_identity,
    CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
    # [B6-FIX] 目标色散路线 × 该腿环境 → 加不加炼金解析尾项，唯一实现在 abfe_core。
    resolve_leg_dispersion_implementation,
    # MEM-00d + B3：co-ion restraint 形式与 charging λ 电荷映射的**唯一**实现。
    # 这一层只负责把它们写进 OpenMM 对象，不重新定义任何形式或映射。
    CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION,
    CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP,
    CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM,
    CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL,
    CO_ALCHEMICAL_ION_RESIDUE_NAMES,
    COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2,
    COION_FLAT_BOTTOM_RADIUS_NM,
    LIGAND_CHARGE_LAMBDA_TOLERANCE_E,
    LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E,
    TOTAL_CHARGE_CONSERVATION_TOLERANCE_E,
    charge_at_lambda,
    co_alchemical_charge_offset_plan,
    minimum_image_displacement_nm,
)

try:
    import pymbar
    HAS_PYMBAR = True
except ImportError:
    HAS_PYMBAR = False

logger = logging.getLogger(__name__)

IBS_WINDOW_DATA_PROTOCOL_VERSION = 1

# EXP-030 residual sampling is an optional production Hamiltonian extension.
# Keep its identity separate from the estimator/analysis modules so a caller
# can fail closed before constructing an OpenMM Context.
# v2 freezes the durable sampling-state ledger layout as frames x states.
# Version 1 artifacts were written as states x frames by an earlier producer;
# they are intentionally not auto-transposed because the loader must never
# guess an ambiguous legacy orientation.
IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION = 2

# EXP-019 seed wiring contract.  This is deliberately implemented in the
# backend (rather than in the launcher) so the value passed to OpenMM is the
# value that gets recorded by provenance.  The hash-based derivation is stable
# across processes and does not depend on Python's randomized hash().
EXP019_SEED_WIRING_PROTOCOL_VERSION = 1
_EXP019_OPENMM_SEED_MAX = 2_147_483_646


def derive_exp019_seed(
    repeat_seed: int,
    leg: str,
    phase: str,
    stage: str,
    window: Any,
    stream: str,
    attempt: int = 0,
) -> int:
    """Derive one stable, domain-separated OpenMM/NumPy seed.

    ``window`` and ``attempt`` are part of the domain on purpose: a recovery
    velocity draw must never silently reuse a production or another-window
    random stream.  OpenMM accepts a positive signed 32-bit seed; zero is
    excluded because it has special/default-like behavior in several APIs.
    """
    if int(repeat_seed) <= 0:
        raise ValueError("EXP-019 repeat_seed must be a positive integer")
    fields = (
        "EXP-019",
        f"v{EXP019_SEED_WIRING_PROTOCOL_VERSION}",
        str(int(repeat_seed)),
        str(leg),
        str(phase),
        str(stage),
        str(window),
        str(stream),
        str(int(attempt)),
    )
    digest = hashlib.sha256("\x1f".join(fields).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") % _EXP019_OPENMM_SEED_MAX
    return int(value or 1)


class Exp019SeedLedger:
    """Record every derived seed actually requested by a baseline run."""

    def __init__(self, repeat_seed: int, leg: str):
        self.repeat_seed = int(repeat_seed)
        if self.repeat_seed <= 0:
            raise ValueError("EXP-019 repeat_seed must be positive")
        self.leg = str(leg)
        if not self.leg:
            raise ValueError("EXP-019 leg must be non-empty")
        self.values: Dict[str, int] = {}

    @staticmethod
    def _key(phase: str, stage: str, window: Any, stream: str, attempt: int) -> str:
        return "/".join(
            (str(phase), str(stage), str(window), str(stream), str(int(attempt)))
        )

    def derive(
        self,
        phase: str,
        stage: str,
        window: Any,
        stream: str,
        attempt: int = 0,
    ) -> int:
        key = self._key(phase, stage, window, stream, attempt)
        value = derive_exp019_seed(
            self.repeat_seed,
            self.leg,
            phase,
            stage,
            window,
            stream,
            attempt,
        )
        previous = self.values.get(key)
        if previous is not None and previous != value:
            raise RuntimeError(f"EXP-019 seed ledger collision for {key!r}")
        self.values[key] = int(value)
        return int(value)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "protocol_version": EXP019_SEED_WIRING_PROTOCOL_VERSION,
            "repeat_seed": self.repeat_seed,
            "leg": self.leg,
            "derived_seed_map": {
                key: int(self.values[key]) for key in sorted(self.values)
            },
        }

# 🔑 [MEM-00h，2026-08-06] 统一到物理力场的 1.0 nm、无 switching——不是 1.2 nm。
# 之前这里跟基础 `NonbondedForce`（PME cutoff=1.0 nm，`SOLVENT_NONBONDED_CUTOFF_NM`/
# `runabfe.py` 的建系）不一致：softcore CV 在 1.0-1.2 nm 这段还有非零 switched
# 相互作用，基础力却已经硬截断为零，导致 λ=1（完全耦合端点）跟"关掉炼金描述、
# 直接用普通力场"的真实系统不严格等价。1.0 nm 是与 Amber 系力场原始参数化匹配的
# 截断（§1.3），1.2 这组值找不到独立物理依据，所以收敛方向是让 softcore 让步，
# 不是把基础力拉长。同步关闭 switching：基础力从来没开过 switching
# （实测 System XML：cutoff=1、switchingDistance=-1、useSwitchingFunction=0），
# switching 只是 softcore 这条线自己多加的一段，没有独立存在的理由。
# ⚠️ 只影响 `potential_type="softcore"`（ACE 传统路径，当前生产配置）；
# `potential_type="dexp"` 用完全独立的 DEXP_VDW_CUTOFF_NM=0.70/
# DEXP_VDW_SWITCH_WIDTH_NM=0.20，不受这条改动影响，见 _create_softcore_force。
SOFTCORE_CUTOFF_NM = 1.0

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


def _stage_window_sampling_identity(protocol_key):
    """Compare sampled Hamiltonians independently of separately checked budgets/gates.

    Stage results still require the complete protocol key. Window trajectories can
    survive a longer requested run or stricter final analysis: their actual budget
    and early-stop evidence are validated by _resume_cached_window_gate_status.
    Native continuation additionally checks the full window System and frozen f_k.
    """
    if not isinstance(protocol_key, dict) or not isinstance(protocol_key.get("payload"), dict):
        return protocol_key
    payload = json.loads(json.dumps(protocol_key["payload"]))
    payload.pop("code_sha256", None)
    payload.pop("final_gate_thresholds", None)
    config = payload.get("run_config") or {}
    config.pop("n_steps_per_window", None)
    config.pop("enable_early_stop", None)
    kwargs = config.get("kwargs") or {}
    for key in (
        "final_min_ess_ratio", "final_min_absolute_ess", "final_min_decorrelated_samples",
        "final_max_uncertainty_kJ_mol", "final_min_target_absolute_ess", "final_max_top1pct_raw_weight",
        "early_stop_min_steps", "early_stop_check_interval_steps", "early_stop_required_consecutive_passes",
        "early_stop_min_ess_ratio", "early_stop_min_absolute_ess", "early_stop_min_decorrelated_samples",
        "early_stop_max_delta_g_drift_kJ_mol", "early_stop_max_uncertainty_kJ_mol",
    ):
        kwargs.pop(key, None)
    # Preserve the original schema's presence/absence of unrelated fields.
    if "kwargs" in config:
        config["kwargs"] = kwargs
    return payload


def load_ibs_window_outputs_from_dir(
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
    current_sampling_score_sha256: Optional[str] = None,
    expected_window_protocol: Optional[Dict] = None,
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
        if u_kn.shape[0] != int(end) - int(start):
            raise ValueError(f"窗口 {local_idx} 状态数与 window_ranges 不符")
        lc, lv = list(lambdas_coul[start:end]), list(lambdas_vdw[start:end])
        if convergence.get("lambdas_coul") != lc or convergence.get("lambdas_vdw") != lv:
            raise ValueError(f"窗口 {local_idx} lambda 内容与当前路径不匹配")
        if stage_type == "vdw" and convergence.get("vdw_nonbonded_protocol_version") != VDW_NONBONDED_PROTOCOL_VERSION:
            raise ValueError(f"窗口 {local_idx} vdW 非键协议版本不匹配")
        if expected_window_protocol is not None:
            gate = _resume_cached_window_gate_status(
                convergence, u_kn.shape, lc, lv, stage_type=stage_type,
                current_sampling_score_sha256=current_sampling_score_sha256,
                **expected_window_protocol,
            )
            if not gate["usable"]:
                raise ValueError(f"窗口 {local_idx} 未通过生产协议校验: {gate}")
        joint_ledgers = _load_validated_joint_score_ledgers(
            output_dir,
            local_idx,
            stage_type,
            convergence,
            n_states=u_kn.shape[0],
            n_frames=n_frames,
            current_sampling_score_sha256=current_sampling_score_sha256,
        )
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
        marker = np.asarray(ibs_state.get("production_entry_f_k", []), dtype=float)
        if marker.shape != f_k_window.shape or not np.array_equal(marker, f_k_window):
            raise ValueError(f"窗口 {local_idx} f_k 与冻结生产入口不一致")
        _, manifest_path = _production_window_checkpoint_paths(checkpoint_dir, stage_type, local_idx)
        with open(manifest_path, encoding="utf-8") as handle:
            production_manifest = json.load(handle)
        frozen_hash = hashlib.sha256(
            json.dumps([round(float(x), 10) for x in f_k_window], sort_keys=False).encode("utf-8")
        ).hexdigest()
        manifest_identity = {
            "stage_type": stage_type, "window_idx": int(local_idx), "K": int(end-start),
            "lambdas_coul": lc, "lambdas_vdw": lv,
            "production_window_checkpoint_protocol_version": PRODUCTION_WINDOW_CHECKPOINT_PROTOCOL_VERSION,
            "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        }
        if stage_type == "vdw":
            manifest_identity["vdw_nonbonded_protocol_version"] = VDW_NONBONDED_PROTOCOL_VERSION
        if expected_window_protocol is not None:
            manifest_identity["stage_protocol_key"] = expected_window_protocol.get("current_stage_protocol_key")
            manifest_identity["coion_identity"] = expected_window_protocol.get("current_coion_identity")
            manifest_identity["sampling_repair_policy"] = expected_window_protocol["repair_policy"]
            parent_protocol = (expected_window_protocol.get("current_stage_protocol_key") or {}).get("payload", {})
            if "temperature_K" in parent_protocol:
                manifest_identity["temperature_K"] = parent_protocol["temperature_K"]
            if "platform_name" in (parent_protocol.get("run_config") or {}):
                manifest_identity["platform_name"] = parent_protocol["run_config"]["platform_name"]
        for field, expected in manifest_identity.items():
            observed = production_manifest.get(field)
            if field == "stage_protocol_key":
                observed = _stage_window_sampling_identity(observed)
                expected = _stage_window_sampling_identity(expected)
            if observed != expected:
                raise ValueError(f"窗口 {local_idx} production manifest {field} 不匹配")
        if production_manifest.get("frozen_f_k_sha256") != frozen_hash:
            raise ValueError(f"窗口 {local_idx} 冻结 f_k 与 production manifest 不一致")
        segments = _validate_production_segments(convergence.get("production_segments"), n_frames)
        outputs.append({
            "production_segments": segments,
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
        if joint_ledgers is not None:
            outputs[-1].update(joint_ledgers)
    _assert_expected_windows_all_loaded(
        expected_windows=expected_windows,
        loaded_windows=[
            int(o["window_index"]) - int(window_index_offset) for o in outputs
        ],
        missing_windows=missing_windows,
        source=f"{output_dir} (stage_type={stage_type})",
    )
    return outputs


def _load_validated_joint_score_ledgers(
    output_dir: str,
    window_index: int,
    stage_type: str,
    convergence: Dict[str, Any],
    *,
    n_states: int,
    n_frames: int,
    current_sampling_score_sha256: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Load and validate the two residual-aware ledgers when they are declared.

    The physical ``energies/bias/base`` triplet remains the primary estimator
    input.  When a residual score identity is present, however, its sampling
    state energies and residual basis are part of the same frame-by-frame
    evidence and may not be silently omitted on resume or analysis.  Returning
    ``None`` for the legacy ``score=None`` case preserves the old baseline file
    layout and behavior.
    """
    cached_score = convergence.get("sampling_score_sha256")
    if cached_score != current_sampling_score_sha256:
        raise ValueError(
            "sampling_score_sha256 mismatch while loading joint score ledgers: "
            f"cached={cached_score!r}, current={current_sampling_score_sha256!r}"
        )
    if cached_score is None:
        return None

    if convergence.get("residual_sampling_protocol_version") != IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION:
        raise ValueError(
            f"窗口 {window_index} 的 residual sampling protocol="
            f"{convergence.get('residual_sampling_protocol_version')!r}，当前期望 "
            f"{IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION}；拒绝加载旧 residual ledger"
        )

    joint = convergence.get("joint_score_window_data")
    if not isinstance(joint, dict):
        raise ValueError(
            f"窗口 {window_index} 缺少 residual-aware joint_score_window_data"
        )

    if joint.get("protocol_version") != IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION:
        raise ValueError(
            f"窗口 {window_index} 的 residual ledger protocol="
            f"{joint.get('protocol_version')!r}，当前期望 "
            f"{IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION}；拒绝猜测旧 ledger 的轴方向"
        )

    if joint.get("physical_target_excludes_residual") is not True:
        raise ValueError(
            f"窗口 {window_index} 的 physical target 未明确排除 sampling residual"
        )
    if joint.get("sampling_state_definition") != (
        "softcore_U_k_plus_A_k_times_residual_basis_minus_offset"
    ):
        raise ValueError(
            f"窗口 {window_index} 的 sampling state 定义未知或不匹配"
        )

    sampling_path = os.path.join(
        output_dir,
        f"dual_window_{window_index}_{stage_type}_sampling_states.npy",
    )
    residual_path = os.path.join(
        output_dir,
        f"dual_window_{window_index}_{stage_type}_residual_basis.npy",
    )
    arrays = {}
    for name, path, expected_shape in (
        (
            "sampling_states",
            sampling_path,
            (int(n_frames), int(n_states)),
        ),
        ("residual_basis", residual_path, (int(n_frames),)),
    ):
        metadata = joint.get(name)
        if not isinstance(metadata, dict):
            raise ValueError(
                f"窗口 {window_index} 的 {name} ledger metadata 缺失"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"窗口 {window_index} 缺少 {name} ledger: {path}"
            )
        if metadata.get("sha256") != _sha256_file(path):
            raise ValueError(f"窗口 {window_index} 的 {name} ledger hash 不匹配")
        array = np.asarray(np.load(path, allow_pickle=False))
        if tuple(array.shape) != expected_shape:
            raise ValueError(
                f"窗口 {window_index} 的 {name} ledger shape={tuple(array.shape)}，"
                f"期望 {expected_shape}"
            )
        recorded_shape = metadata.get("shape")
        if recorded_shape != [int(x) for x in expected_shape]:
            raise ValueError(
                f"窗口 {window_index} 的 {name} ledger metadata shape 不匹配"
            )
        if metadata.get("dtype") != str(array.dtype):
            raise ValueError(
                f"窗口 {window_index} 的 {name} ledger dtype 不匹配"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"窗口 {window_index} 的 {name} ledger 含非有限值")
        arrays[name] = array

    return {
        # Production writes sampling states as (frames, states) for durable
        # row-wise storage; analysis consumes the existing (states, frames)
        # convention used by physical ``u_kn``.
        "sampling_state_energies": arrays["sampling_states"].T.copy(),
        "residual_basis": arrays["residual_basis"].copy(),
    }


def ibs_lj_tail_lrc_is_applicable(
    potential_type: Optional[str],
    dispersion_protocol: Optional[str] = None,
    environment_type: Optional[str] = None,
) -> bool:
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

    ---
    🔑 [B6 / memtodolist §1.3 / §6.4] 第二个否决维度：`dispersion_protocol`。

    这里的 `lj_tail_lrc_coeff[k] / V(t)` 假设配体周围是**均匀体相密度**。配体埋在
    脂双层口袋里时这个假设直接不成立——局域密度既不是水也不是体相脂质。所以
    §1.3 明文要求膜体系"关闭当前 `lrc_coeff/V`，metadata 写
    `disabled_by_membrane_forcefield_protocol`，**不能写成遗漏**"。

    注意这**不是**"膜体系一律禁用长程色散修正"。环境–环境那部分色散仍然按所选
    脂质力场的原始参数化条件走（Amber Lipid21 就是开着各向同性 LRC 拟合的，
    关掉才是错的）——那是基础 `NonbondedForce` 的事。被关掉的只有**炼金
    ligand–environment** 这一项，因为只有它的均匀密度假设在膜里失效。

    扩展这个谓词而不是在别处加第二道门，正是它 docstring 一直强调的纪律：
    生产者与报告者必须共用同一个真相，否则 `final_results.json` 会声称
    "已应用"而实际没有（或反之）。

    `dispersion_protocol=None` 或 `legacy_uniform_density_lrc` 时行为与本改动前
    **逐位一致**，当前可溶体系生产路径不受影响。

    ---
    ⚠️ [B6-FIX 2026-08-04] 第三个维度：`environment_type`，**这条腿**的环境。

    此前这里是一个全局布尔（`protocol != legacy → False`），没有环境维度，于是
    "膜口袋里局域密度不均匀"这个**对复合物腿正确**的理由，被原样套到了同一次运行的
    **纯水溶剂腿**上——那条腿里配体周围恰恰就是均匀体相水，尾项修正完全成立。
    实测代价：同一个配体的溶剂腿 vanishing 96.96 → 83.83 kJ/mol（−13.1），
    而 `final_results.json` 里还写着 `disabled_by_membrane_forcefield_protocol`。

    分层判据挪到 `abfe_core.resolve_leg_dispersion_implementation()`（唯一实现）：
    目标（力场原始参数化条件）× 该腿环境 → 加不加解析尾项。本谓词只是它的
    "potential_type 一票否决 + 布尔投影"，保持生产者/报告者共用同一真相的纪律。
    """
    if str(potential_type or "").strip().lower() == "dexp":
        return False
    return bool(
        resolve_leg_dispersion_implementation(
            dispersion_protocol, environment_type
        )["alchemical_uniform_density_lrc"]
    )


def ibs_lj_tail_lrc_inapplicable_reason(
    potential_type: Optional[str],
    dispersion_protocol: Optional[str] = None,
    environment_type: Optional[str] = None,
) -> str:
    """不附加修正时的机器可读理由；适用时返回空串。"""
    if ibs_lj_tail_lrc_is_applicable(
        potential_type, dispersion_protocol, environment_type
    ):
        return ""
    if str(potential_type or "").strip().lower() == "dexp":
        return (
            "potential_type='dexp' 尚未验证解析尾项公式是否适用于该势函数"
        )
    # 理由字符串同样来自唯一实现（§1.3 指定的机器可读措辞在那里，不要在这里改写）。
    return str(
        resolve_leg_dispersion_implementation(
            dispersion_protocol, environment_type
        )["reason"]
    )


@contextmanager
def _timed(bucket: Dict[str, float], key: str):
    """把一段代码的墙钟耗时累加进 `bucket[key]`（性能分档计时用）。

    `time.perf_counter()` 单次开销约 50-100ns，控制面每 500 步一次的循环里
    加这一层可忽略不计，因此默认常开——目的是把"积分 vs guard vs CV/probe
    vs ledger I/O"的耗时占比变成可测量的数字，而不是靠猜测决定优化方向。
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        bucket[key] = bucket.get(key, 0.0) + (time.perf_counter() - t0)


PRODUCTION_SEGMENT_PROTOCOL_VERSION = 1


def _validate_production_segments(segments, n_frames):
    if not isinstance(segments, list) or (n_frames and not segments):
        raise ValueError("缺少 production segments；不能假定跨进程轨迹连续")
    cursor = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("production segment 必须是对象")
        start, end = segment.get("start_frame"), segment.get("end_frame")
        if (type(start) is not int or type(end) is not int
                or start != cursor or end <= start or end > n_frames
                or segment.get("n_frames") != end - start
                or not isinstance(segment.get("session_id"), str) or not segment["session_id"]
                or not isinstance(segment.get("reason"), str) or not segment["reason"]):
            raise ValueError("production segment 边界非法、不连续或与帧数不匹配")
        cursor = end
    if cursor != n_frames:
        raise ValueError("production segments 未完整覆盖能量帧")
    return segments


def _start_production_segment(sampler, reason):
    start = len(sampler.energy_history)
    previous = getattr(sampler, "production_segment_starts", [])
    sampler.production_segment_starts = [dict(seg) for seg in previous if seg["start_frame"] < start]
    sampler.production_segment_starts.append({
        "start_frame": start, "reason": str(reason),
        "session_id": f"{os.getpid()}:{time.time_ns()}",
    })


def _production_segments_snapshot(sampler):
    n_frames = len(sampler.energy_history)
    starts = getattr(sampler, "production_segment_starts", [])
    segments = []
    for i, start in enumerate(starts):
        end = starts[i + 1]["start_frame"] if i + 1 < len(starts) else n_frames
        if end > start["start_frame"]:
            segments.append(dict(start, end_frame=end, n_frames=end-start["start_frame"]))
    return _validate_production_segments(segments, n_frames)


def _production_history_lengths(sampler) -> int:
    """返回全部逐帧生产 ledgers 的公共长度，并强制原子对齐。

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
    has_sampling_state = hasattr(sampler, "sampling_state_energy_history")
    has_residual_basis = hasattr(sampler, "residual_basis_history")
    if has_sampling_state != has_residual_basis:
        raise RuntimeError(
            "生产 history 的 residual ledgers 不完整：sampling_state/residual_basis "
            "必须同时存在"
        )
    if has_sampling_state:
        lengths += (
            len(sampler.sampling_state_energy_history),
            len(sampler.residual_basis_history),
        )
    if len(set(lengths)) != 1:
        raise RuntimeError(
            f"三份长度不一致（含可选 residual ledgers）energy/bias/base/sampling_state/residual = {lengths}；"
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
    if hasattr(sampler, "production_segment_starts"):
        sampler.production_segment_starts = [seg for seg in sampler.production_segment_starts
                                             if seg["start_frame"] < keep]
    del sampler.energy_history[keep:]
    del sampler.bias_history[keep:]
    del sampler.base_energy_history[keep:]
    if hasattr(sampler, "sampling_state_energy_history"):
        del sampler.sampling_state_energy_history[keep:]
        del sampler.residual_basis_history[keep:]
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
    """Triclinic minimum-image displacement for row-vector box matrices.

    薄包装：本层只负责把 OpenMM Quantity / list-of-Vec3 归一成 (3,3) 数组，
    minimum-image 的数学只有一份，在 `abfe_core.minimum_image_displacement_nm`
    （co-ion 的 §13.1 几何判据在那一层就要用，而 abfe_core 不能反向 import 本模块）。
    """
    box = _box_vectors_to_nm_array(box_vectors)
    if box is None:
        raise ValueError("minimum-image 计算需要有限的 (3,3) 周期盒向量")
    return minimum_image_displacement_nm(displacement, box)


def _create_co_alchemical_ion_restraint(
    ion_index: int,
    anchor_atom_index: int,
    reference_displacement_nm: Sequence[float],
    force_constant_kj_per_mol_nm2: float = COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2,
    flat_bottom_radius_nm: float = COION_FLAT_BOTTOM_RADIUS_NM,
) -> openmm.CustomCompoundBondForce:
    """[MEM-00d] flat-bottom + 锚点相对的 co-ion 位置限制。

    井心 = 锚点原子的**当前**位置 + 冻结位移 d0，所以它随体系一起被 barostat 缩放；
    平坦区内无力，出了平坦区才有 `0.5·k·(r−r₀)²` 的软墙。

    为什么是 `CustomCompoundBondForce` + `pointdistance`（而不是
    `CustomExternalForce` + `periodicdistance`）：
      * `periodicdistance` **只在 CustomExternalForce 里存在**（实测
        CustomCompoundBondForce / CustomCentroidBondForce 都报 unknown function），
        而 CustomExternalForce 只能吃绝对参考点 —— 那正是 MEM-00d 要退役的形式；
      * CustomCompoundBondForce 打开 PBC 后会把 bond 内的粒子平移到与第一个粒子
        相同的周期镜像，所以其中的 `pointdistance` **就是** minimum-image 距离
        （实测：离子 z=0.2、锚点 z=9.4、盒 z=12 → 0.2 nm，不是 9.2 nm）。

    表达式与力常数/半径都来自 `abfe_core` 的常量，与落进身份指纹的那份 restraint
    描述同源；两者若分叉，`verify_co_alchemical_ion_identity` 会当场拦下。
    """
    force = openmm.CustomCompoundBondForce(2, CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION)
    force.addGlobalParameter("k_ion", float(force_constant_kj_per_mol_nm2))
    force.addGlobalParameter("r0_ion", float(flat_bottom_radius_nm))
    for name in ("dx0", "dy0", "dz0"):
        force.addPerBondParameter(name)
    displacement = [float(v) for v in list(reference_displacement_nm)[:3]]
    if len(displacement) != 3:
        raise ValueError(
            f"reference_displacement_nm 需要 3 个分量，收到 {reference_displacement_nm!r}"
        )
    if int(ion_index) == int(anchor_atom_index):
        raise ValueError(
            f"co-ion 与 restraint 锚点是同一个粒子（index={ion_index}）——"
            "那样这个 restraint 恒等于常数，什么也限制不住。"
        )
    force.addBond([int(ion_index), int(anchor_atom_index)], displacement)
    force.setUsesPeriodicBoundaryConditions(True)
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


# 离子残基名的**唯一**一份表（`_select_bulk_water_counterion` 与
# `_identify_reserved_neutral_co_ions` 共用）。两处都还要按电荷交叉核对，
# 所以名字表只是"候选集"，判身份不靠它单独说话。
#
# ⚠️ §0.5.4 的教训在这儿仍然成立：**靠残基名判身份，换一套体系就错**
# （CHARMM-GUI 的 AMBER 转换器把离子写成 `Na+` / `Cl-`，所以带符号的形式也在表里）。
# 真正的修法是走 `abfe_core.classify_system_composition()` 的组成驱动判定；
# 这里先把两处收敛为一份，避免"补一个漏一片"，彻底收口单独立项。
# Candidate residue names live in abfe_core so the builder-identity helper and
# the runtime selector share one table.  The actual identity check still also
# requires a zero charge; residue names alone are never sufficient.


def _select_bulk_water_counterion(
    nb_force: openmm.NonbondedForce,
    ligand_indices: List[int],
    topology,
    positions,
    box_vectors,
) -> Tuple[List[int], List[np.ndarray], Dict[str, Any]]:
    """Select enough monovalent counterions using PBC-aware bulk-water metrics.

    ⚠️ **[MEM-00c] 只允许 `select_co_alchemical_ion_once()` 调用本函数。**
    它按传入坐标当场排序挑离子（主键是"到最近溶质的 minimum-image 距离"这个连续量），
    所以**坐标变了结果就可能变**——实测 0.05 nm 位移即可翻转（见
    `tests/test_coalchemical_ion_identity.py`）。任何"动力学 / REMD 副本 / u_kn
    各自调一次"的用法都会造成 u_kn 与动力学 Hamiltonian 用了不同粒子的静默错误。
    下游一律走 `abfe_core.verify_co_alchemical_ion_identity()` 只读核对。
    """
    pos_nm = _positions_to_nm_array(positions)
    ligand_set = set(int(i) for i in ligand_indices)
    raw_lig_net_charge = 0.0
    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(idx)
        raw_lig_net_charge += q.value_in_unit(unit.elementary_charge)
    if not np.isfinite(raw_lig_net_charge):
        raise RuntimeError("配体净电荷为 NaN/Inf，拒绝选择共炼金反离子")
    lig_net_charge = int(round(raw_lig_net_charge))
    if abs(raw_lig_net_charge - lig_net_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
        raise RuntimeError(
            f"配体净电荷 {raw_lig_net_charge:+.6f} e 不接近整数（容差 1e-3 e）"
        )
    if lig_net_charge == 0:
        return [], [], {}

    target_ion_charge = -1.0 if lig_net_charge > 0 else 1.0
    required_count = abs(lig_net_charge)
    water_names = {"HOH", "WAT", "SOL", "TIP3", "TIP3P"}
    ion_names = set(CO_ALCHEMICAL_ION_RESIDUE_NAMES)
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


def _identify_reserved_neutral_co_ions(
    nb_force: openmm.NonbondedForce,
    topology,
    required_count: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """[B3] 找出建系时预留的 **中性 ion-shaped dummy** 粒子（charge-transfer 路线）。

    判据只有两条，**都与坐标无关**：残基名在离子名集合里，且电荷严格为 0。
    这不是"挑一个"，而是"认出来"——真实体系里的离子都带电，所以一个**电荷为 0 的
    离子残基**只可能是建系时专门留出来的 co-ion。于是 MEM-00c 那个"坐标一动选择就翻转"
    的失效模式在 charge-transfer 路线上**结构上不存在**（没有排序、没有连续量主键）。

    数量必须恰好等于 |q_L|（§2.2：每个 co-ion 最多接过一个单位电荷），
    不足或多余都 fail closed —— 多余尤其危险：那说明建系时留了不止一个 dummy，
    随便挑一个就又回到"按坐标选"的老路上了。
    """
    ion_names = set(CO_ALCHEMICAL_ION_RESIDUE_NAMES)
    candidates: List[Dict[str, Any]] = []
    for atom in topology.atoms():
        if atom.residue.name.upper() not in ion_names:
            continue
        q, sigma, epsilon = nb_force.getParticleParameters(atom.index)
        q_val = q.value_in_unit(unit.elementary_charge)
        if abs(q_val) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            continue
        candidates.append(
            {
                "ion_index": int(atom.index),
                "residue_name": str(atom.residue.name),
                "element": str(getattr(atom.element, "symbol", "") or ""),
                "charge_e": float(q_val),
                "sigma_nm": float(sigma.value_in_unit(unit.nanometer)),
                "epsilon_kj_mol": float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
            }
        )
    if len(candidates) != int(required_count):
        raise RuntimeError(
            f"charge-transfer 需要建系时预留 {int(required_count)} 个**电荷为 0 的**"
            f"ion-shaped dummy 粒子（§2.2 / §4.3），实测找到 {len(candidates)} 个"
            f"：{[c['ion_index'] for c in candidates]}\n"
            "    · 少了：输入体系里没有 reserved co-ion。复合物腿的输入拓扑必须自带；"
            "溶剂腿由 B4 的 builder 生成。**不要**拿一个已经带电的物理离子顶上——"
            "那会让 λ=1 端的总电荷不再等于物理体系的总电荷。\n"
            "    · 多了：预留了不止一个，那就得靠坐标去挑，MEM-00c 的漂移风险原地复活。\n"
            f"    离子残基名判据：{sorted(ion_names)}；"
            f"电荷零判据容差 {TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:g} e。"
        )
    indices = sorted(int(c["ion_index"]) for c in candidates)
    return indices, {
        "selection": "reserved_neutral_ion_shaped_dummy_identified_by_zero_charge",
        "coordinate_independent": True,
        "required_count": int(required_count),
        "candidates": candidates,
    }


def select_co_alchemical_ion_once(
    system: openmm.System,
    ligand_indices: List[int],
    topology,
    positions,
    box_vectors,
    charge_treatment: str = CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
    ion_restraint_k: Optional[float] = None,
    flat_bottom_radius_nm: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """[MEM-00c] **唯一**允许发生 co-ion 选择的入口；返回可落盘的身份 spec。

    调用它**一次**（首跑、拿到最终预平衡坐标之后），把返回的 spec 落盘；
    此后动力学 System 构建、REMD 副本、`compute_u_kn`、resume 全部只读消费这份 spec，
    不再有第二次选择。中性配体返回 `None`（那条路径根本不需要 co-ion）。

    为什么这条边界必须由代码强制、而不是靠约定：co-annihilation 选择器的排序主键是
    坐标的连续函数，而首跑与 resume 喂进来的坐标本来就不同（首跑在预平衡输出上又叠了
    2000 步最小化，resume 直接读 DCD 末帧）。靠"记得只选一次"是守不住的。

    两条路线的身份来源不同，这是**有意的**：

    * `co_annihilation_experimental`：从既有盐里按 bulk-water 判据挑一个**异号**物理
      反离子（`_select_bulk_water_counterion`，坐标相关 ⟹ 必须冻结）；
    * `co_alchemical_charge_transfer`：认出建系时预留的**中性** ion-shaped dummy
      （`_identify_reserved_neutral_co_ions`，坐标无关）。
    """
    nb_force = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None
    )
    if nb_force is None:
        raise RuntimeError("选择 co-ion 需要 NonbondedForce，但 System 里没有。")

    raw_lig_net_charge = 0.0
    for idx in {int(i) for i in ligand_indices}:
        q, _, _ = nb_force.getParticleParameters(idx)
        raw_lig_net_charge += q.value_in_unit(unit.elementary_charge)
    if not np.isfinite(raw_lig_net_charge):
        raise RuntimeError("配体净电荷为 NaN/Inf，拒绝冻结 co-ion 身份")
    lig_net_charge = int(round(raw_lig_net_charge))
    if abs(raw_lig_net_charge - lig_net_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
        raise RuntimeError(
            f"配体净电荷 {raw_lig_net_charge:+.6f} e 不接近整数（容差 1e-3 e）"
        )
    if lig_net_charge == 0:
        return None

    if charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        ion_indices, ion_meta = _identify_reserved_neutral_co_ions(
            nb_force, topology, abs(lig_net_charge)
        )
    else:
        ion_indices, _ion_ref_positions_nm, ion_meta = _select_bulk_water_counterion(
            nb_force, ligand_indices, topology, positions, box_vectors
        )
    if not ion_indices:
        raise RuntimeError(
            "配体带净电荷但没能确定 co-ion 身份 —— 无法保持 PME 去电荷腿逐 λ 电荷守恒。"
        )

    return build_co_alchemical_ion_identity(
        system=system,
        topology=topology,
        ion_atom_indices=ion_indices,
        ligand_indices=list(ligand_indices),
        positions_nm=_positions_to_nm_array(positions),
        box_vectors=_box_vectors_to_nm_array(box_vectors),
        ligand_net_charge_e=lig_net_charge,
        charge_treatment=charge_treatment,
        selection_provenance=ion_meta,
        k_kj_per_mol_nm2=ion_restraint_k,
        flat_bottom_radius_nm=flat_bottom_radius_nm,
    )


def _prepare_pme_coulomb_leg_system(
    system_template: openmm.System,
    ligand_indices: List[int],
    lambda_name: str = "lambda_coul",
    allow_charged_ligand: bool = False,
    topology=None,
    positions=None,
    box_vectors=None,
    co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
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
        # [MEM-00c] 每个 replica / 每次 u_kn 重算都拿同一份冻结身份。
        co_alchemical_ion_spec=co_alchemical_ion_spec,
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
# attachment 腿的监控落盘间隔（步）。2 fs × 500 = 1 ps 一行，一条腿约 2400 行，
# 写盘开销可忽略。这条腿此前**一帧都不写**，2026-08-03 的 NaN 因此没有任何现场证据。
ATTACHMENT_MONITOR_INTERVAL = 500
# 头若干步用细粒度：2026-08-03 定位 MEM-15 时，NaN 发生在第一个 500 步分块内，
# 粗粒度只留下 step 0 一行、夹不住死亡时刻。根因修掉之后细粒度就只剩成本了 ——
# 每行要两次 `getState`（一次 `getForces` = GPU 同步 + 45354×3 个 float 下载，
# 一次取 Boresch 力组能量），25 步一行相当于每 50 fs 打断一次 GPU。
# 放宽到 250 步（0.5 ps）、只覆盖头 2500 步：既留住"起跑阶段出事能夹住"的能力，
# 又不再按 50 fs 停。要重新细化时把这两个数调小即可，逻辑不用动。
ATTACHMENT_MONITOR_FINE_INTERVAL = 250
ATTACHMENT_MONITOR_FINE_STEPS = 2500
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

    try:
        temperature_k = float(temperature_k)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("attachment 温度必须是正有限数") from exc
    if not np.isfinite(temperature_k) or temperature_k <= 0.0:
        raise ValueError("attachment 温度必须是正有限数")

    n_seeds = int(n_seeds)
    if n_seeds < 1:
        raise ValueError(f"n_seeds 必须 ≥1，收到 {n_seeds}")
    try:
        n_steps_per_state = int(n_steps_per_state)
        equil_steps_per_state = int(equil_steps_per_state)
        steps_per_sample = int(steps_per_sample)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "attachment 步数参数必须是整数"
        ) from exc
    if n_steps_per_state < 2:
        raise ValueError("attachment n_steps_per_state 必须至少为 2")
    if equil_steps_per_state < 0:
        raise ValueError("attachment equil_steps_per_state 不能为负")
    if steps_per_sample <= 0 or steps_per_sample > n_steps_per_state:
        raise ValueError(
            "attachment steps_per_sample 必须为正且不超过 n_steps_per_state"
        )
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
            "converged": bool(all(r.get("converged") is True for r in runs)),
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
    resolved_platform, platform_props = _build_platform_properties(platform_name)
    simulation = app.Simulation(
        topology,
        work,
        integrator,
        openmm.Platform.getPlatformByName(resolved_platform),
        platform_props,
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
    log(
        f"  ⚠️ 扫描顺序是**从全强度端往下**：第一个实际跑的态是 "
        f"λ={float(lam[order[0]]):.4f}，不是列表里的第一个 λ。"
    )
    log(f"  每态 平衡 {equil_steps_per_state} 步 + 生产 {n_steps_per_state} 步 → {n_samples} 帧")

    # ---- 起点体检（MEM-06）：把"跑了几步的无上下文 NaN"变成"起点就坏，坏在哪" ----
    #
    # 实测背景：memtest 膜体系 2026-08-02 在这里出 `Particle coordinate is NaN`。
    # 那次第一个跑的态是 λ=1.0（**全强度** Boresch 限制力，kr=946.7 kJ/mol/nm²），
    # 起点坐标刚经过两次 PBC 修复 + 两次锚点重推导（r0 0.705 → 0.767 nm、
    # θA 61.3° → 63.3°）。所以"限制力参数"与"起点坐标"都是嫌疑，
    # 而当时的日志两样都没留下 —— 下面把它们全部量出来再落盘。
    #
    # ⚠️ 这段**只读**：不最小化、不改坐标/速度、不改任何 global parameter
    # （λ 会在下面的循环里由 order[0] 重新设成同一个值）。所以对既有可溶生产路径
    # 的数值逐位无影响（§7.7）。
    simulation.context.setParameter(
        BORESCH_ATTACHMENT_LAMBDA_NAME, float(lam[order[0]])
    )
    log(
        "  🔎 Boresch 力常数: "
        + ", ".join(
            f"{name}={float(restraint_params['force_constants'][name]):.4g}"
            for name in ("kr", "kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC")
            if name in (restraint_params.get("force_constants") or {})
        )
    )
    try:
        _measured = calc_boresch_from_last_frame(
            simulation.context.getState(getPositions=True).getPositions(asNumpy=True),
            restraint_params["receptor_indices"],
            restraint_params["ligand_indices"],
        )
        _eq = restraint_params.get("equilibrium_values") or {}
        log(
            "  🔎 起点实测几何 vs 已提交平衡值: "
            + ", ".join(
                f"{key}: {float(_measured[key]):.4f} / {float(_eq[key]):.4f}"
                for key in ("r0", "thetaA0", "thetaB0", "phiA0", "phiB0", "phiC0")
                if key in _measured and key in _eq
            )
        )
    except Exception as exc:  # noqa: BLE001
        # 诊断失败不该顶替真正的失败，但也不许静默 —— 说出来，然后继续体检受力。
        log(f"  ⚠️ 起点几何诊断未能完成（不阻断）：{type(exc).__name__}: {exc}")
    _group_energies = {}
    for _group in (0, BORESCH_ATTACHMENT_FORCE_GROUP):
        try:
            _group_energies[_group] = float(
                simulation.context.getState(getEnergy=True, groups={_group})
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )
        except Exception:  # noqa: BLE001
            pass
    if _group_energies:
        log(
            "  🔎 逐 force group 能量 (kJ/mol): "
            + ", ".join(f"g{g}={e:.6g}" for g, e in sorted(_group_energies.items()))
        )
    # ---- 入口快照 + 在跑中监控（2026-08-03）----
    #
    # 为什么必须有：这条腿此前**一帧轨迹、一行监控都不写**，所以 2026-08-03 的
    # `Particle coordinate is NaN` 除了一个 traceback 什么证据都没留下。
    # 而离线忠实重放（同起点、同种子、同步数，走完整条 λ 序列 2.4 ns）
    # **不复现** —— 也就是说生产与重放之间有差异，但没有任何落盘的东西能拿来 diff。
    # §0.5.7 的教训原话是"离线再猜性价比很低，必须在跑中留下轨迹"。
    _diag_dir = output_dir or "."
    try:
        os.makedirs(_diag_dir, exist_ok=True)
        _forces = []
        for _i, _f in enumerate(work.getForces()):
            _entry = {
                "index": _i,
                "class": _f.__class__.__name__,
                "force_group": int(_f.getForceGroup()),
            }
            if hasattr(_f, "usesPeriodicBoundaryConditions"):
                try:
                    _entry["uses_pbc"] = bool(_f.usesPeriodicBoundaryConditions())
                except Exception:  # noqa: BLE001
                    pass
            for _attr in ("getNonbondedMethod", "getCutoffDistance", "getNumParticles"):
                if hasattr(_f, _attr):
                    try:
                        _v = getattr(_f, _attr)()
                        _entry[_attr[3:]] = (
                            float(_v.value_in_unit(unit.nanometer))
                            if hasattr(_v, "value_in_unit") else int(_v)
                        )
                    except Exception:  # noqa: BLE001
                        pass
            _forces.append(_entry)
        _pos_nm = np.asarray(
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
        _box_nm = np.asarray(
            [v.value_in_unit(unit.nanometer)
             for v in simulation.context.getState().getPeriodicBoxVectors()],
            dtype=np.float64,
        )
        # 锚点的 raw 距离 vs minimum-image 距离：两者不等就说明锚点分处不同周期镜像，
        # 而 `CustomCompoundBondForce` 的 angle()/dihedral() 不做 minimum-image。
        _rec = [int(i) for i in restraint_params["receptor_indices"]]
        _lig = [int(i) for i in restraint_params["ligand_indices"]]
        _anchor_pairs = {}
        for _tag, _a, _b in (
            ("R0-L0", _rec[0], _lig[0]), ("R1-R0", _rec[1], _rec[0]),
            ("R2-R1", _rec[2], _rec[1]), ("L1-L0", _lig[1], _lig[0]),
            ("L2-L1", _lig[2], _lig[1]),
        ):
            _d = _pos_nm[_a] - _pos_nm[_b]
            # Use the same closest-lattice implementation as all production
            # geometry paths.  The former diagonal ``round`` approximation was
            # silently wrong for a triclinic box and made this diagnostic
            # disagree with the actual OpenMM periodic geometry.
            _mi = minimum_image_displacement_nm(_d, _box_nm)
            _anchor_pairs[_tag] = {
                "raw_nm": float(np.linalg.norm(_d)),
                "minimum_image_nm": float(np.linalg.norm(_mi)),
                "differs": bool(
                    abs(np.linalg.norm(_d) - np.linalg.norm(_mi)) > 1.0e-6
                ),
            }
        _snapshot = {
            "n_particles": int(work.getNumParticles()),
            "n_constraints": int(work.getNumConstraints()),
            "forces": _forces,
            "box_vectors_nm": _box_nm.tolist(),
            "positions_sha256": hashlib.sha256(
                np.ascontiguousarray(_pos_nm, dtype=np.float64).tobytes()
            ).hexdigest(),
            "positions_min_nm": _pos_nm.min(axis=0).tolist(),
            "positions_max_nm": _pos_nm.max(axis=0).tolist(),
            "anchor_pair_distances": _anchor_pairs,
            "lambdas": [float(v) for v in lam],
            "scan_order_k": [int(k) for k in order],
            "seed": int(seed),
            "equil_steps_per_state": int(equil_steps_per_state),
            "n_steps_per_state": int(n_steps_per_state),
            "steps_per_sample": int(steps_per_sample),
            "restraint": {
                "equilibrium_values": restraint_params.get("equilibrium_values"),
                "force_constants": restraint_params.get("force_constants"),
                "receptor_indices": _rec,
                "ligand_indices": _lig,
            },
        }
        _snap_path = os.path.join(_diag_dir, "stage0_attachment_inputs.json")
        with open(_snap_path, "w", encoding="utf-8") as _h:
            json.dump(_snapshot, _h, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        log(f"  🧾 attachment 腿入口快照已落盘: {_snap_path}")
        # ⚠️ **坐标本身**也要存，不能只存 SHA256。
        # 2026-08-03 的教训：只有哈希时，离线既无法复现也无法逐力分解 ——
        # 那次 NaN 发生在第一个 500 步分块内（< 1 ps），而用 rebalance 末帧做的
        # 离线重放（同种子、同步数）走完 2.4 ns 都不崩，两者 PE 差 31 MJ/mol。
        # 差异只能来自坐标，而坐标没落盘，于是每查一步都要重跑一次生产。
        # 1.1 MB 换"能离线确定性复现"，永远值。
        _start_path = os.path.join(_diag_dir, "stage0_attachment_start.npz")
        np.savez_compressed(
            _start_path, positions_nm=_pos_nm, box_vectors_nm=_box_nm
        )
        log(f"  🧾 attachment 腿起点坐标已落盘（可离线复现）: {_start_path}")
        _torn = [t for t, v in _anchor_pairs.items() if v["differs"]]
        if _torn:
            log(
                f"  ⚠️ 锚点对 {_torn} 的 raw 距离与 minimum-image 距离不同 —— "
                "说明锚点分处不同周期镜像。`CustomCompoundBondForce` 的 "
                "angle()/dihedral() **不做** minimum-image，这会让被约束的几何量"
                "不是你以为的那个。"
            )
            raise RuntimeError(
                "attachment Boresch 锚点跨越周期边界；angle/dihedral 的 PBC 几何"
                "无法与 minimum-image 参考一致，拒绝在撕裂的锚点几何上采样。"
            )
    except RuntimeError:
        raise
    except Exception as _exc:  # noqa: BLE001
        log(f"  ⚠️ 入口快照落盘失败（不阻断）: {type(_exc).__name__}: {_exc}")

    _monitor_path = os.path.join(_diag_dir, "stage0_attachment_monitor.csv")
    _monitor = None
    try:
        _monitor = open(_monitor_path, "w", encoding="utf-8", buffering=1)
        _monitor.write("cumulative_step,lambda_index_k,lambda,phase,"
                       "potential_kJ_mol,temperature_K,max_force_kJ_mol_nm,"
                       "max_force_atom_index,boresch_energy_kJ_mol\n")
        log(f"  📈 attachment 腿监控已启用（每 {ATTACHMENT_MONITOR_INTERVAL} 步）: "
            f"{_monitor_path}")
    except Exception as _exc:  # noqa: BLE001
        log(f"  ⚠️ 监控无法写入（不阻断）: {_exc}")
        _monitor = None

    _n_dof = 3 * work.getNumParticles() - work.getNumConstraints()

    def _monitor_row(cumulative_step, k_index, lam_value, phase, simulation_obj=simulation):
        """写一行监控。**崩之前的最后几行就是唯一的现场证据。**"""
        if _monitor is None:
            return
        try:
            st = simulation_obj.context.getState(getEnergy=True, getForces=True)
            pe = st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            ke = st.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
            temp = 2.0 * ke / (_n_dof * 0.008314462618) if _n_dof > 0 else float("nan")
            fmag = np.linalg.norm(
                np.asarray(st.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer), dtype=float),
                axis=1,
            )
            worst = int(np.argmax(fmag)) if fmag.size else -1
            ub = simulation_obj.context.getState(
                getEnergy=True, groups={BORESCH_ATTACHMENT_FORCE_GROUP}
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            _monitor.write(
                f"{int(cumulative_step)},{int(k_index)},{float(lam_value):.6f},"
                f"{phase},{pe:.6f},{temp:.4f},{float(fmag.max()):.6f},{worst},"
                f"{ub:.6f}\n"
            )
        except Exception:  # noqa: BLE001
            pass

    assert_starting_state_is_sane(
        simulation.context,
        topology,
        label=f"attachment 腿起点 λ={float(lam[order[0]]):.4f}",
        remediation=(
            "    这一步**没有**做最小化：起点坐标是上游交给本腿的（预平衡 → 带 Boresch\n"
            "    的 rebalance → PBC 修复 → 锚点重推导）。所以要分两种情况看：\n"
            "      * 上面 g0（基础力）已经异常 → 坏的是起点坐标本身，与 Boresch 无关，\n"
            "        查 PBC 修复是否撕开了分子（§0.5.7）以及拓扑有没有键；\n"
            "      * 只有 Boresch 力组异常 → 查上面那行「实测几何 vs 已提交平衡值」，\n"
            "        以及力常数是否过硬。`--boresch-source simple` 是纯几何涨落估算，\n"
            "        可与 `auto`（MACE 打分、偏好大 kr）对照。"
        ),
        log=log,
    )

    linearity_checked = False
    _cumulative = 0
    for k in order:
        simulation.context.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, float(lam[k]))
        simulation.context.setVelocitiesToTemperature(
            temperature_k * unit.kelvin, int(seed) + 7919 * k + 1
        )
        _monitor_row(_cumulative, k, lam[k], "state_start")
        # 平衡段分块步进，只为了能在崩之前留下监控行；总步数与分块前完全相同。
        #
        # 头 `ATTACHMENT_MONITOR_FINE_STEPS` 步用**细**粒度：2026-08-03 那次 NaN
        # 就发生在第一个 500 步分块内，粗粒度只留下了 step 0 一行，等于没夹住。
        _remaining = int(equil_steps_per_state)
        while _remaining > 0:
            _interval = (
                ATTACHMENT_MONITOR_FINE_INTERVAL
                if _cumulative < ATTACHMENT_MONITOR_FINE_STEPS
                else ATTACHMENT_MONITOR_INTERVAL
            )
            _chunk = min(_interval, _remaining)
            simulation.step(_chunk)
            _remaining -= _chunk
            _cumulative += _chunk
            _monitor_row(_cumulative, k, lam[k], "equil")
        for s in range(n_samples):
            simulation.step(int(steps_per_sample))
            _cumulative += int(steps_per_sample)
            if s % max(1, ATTACHMENT_MONITOR_INTERVAL // int(steps_per_sample)) == 0:
                _monitor_row(_cumulative, k, lam[k], "sample")
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

    if _monitor is not None:
        try:
            _monitor.close()
        except Exception:  # noqa: BLE001
            pass

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
        # When convergence gates are deliberately disabled for a short pilot,
        # preserve their outcome in the result instead of claiming completion.
        # Production callers leave the gates enabled and will already have
        # raised on a failed check.
        "converged": bool(
            ti_ok and isinstance(half, dict) and half.get("passed") is True
        ),
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

    # [P0-REMD-CUDA] 显式释放 attachment 腿的 CUDA Context。
    #
    # 本函数整段原来**没有任何 teardown**：`simulation` 是局部变量，只靠函数返回时的
    # 引用计数回收。这在"下一步立刻要建 12 个 replica Context"的场景下太脆——
    # 引用一旦被 payload / 闭包 / 异常 traceback 之一勾住，显存就一直挂着，而
    # Stage 1 实测开跑前卡上已被占 12197/16303 MiB（只剩 3646，12 个 Context 需 3804）。
    # 与预平衡那段同一口径：显式断链 + gc，并把释放前后的显存打出来，对不上账就看得见。
    #
    # ⚠️ 量级要诚实：这里只有**一个** Context ≈ 317 MiB，所以它**不足以**解释那
    # 12.2 GB 里对不上账的约 10 GB。修它是因为它本来就该修，不是因为它是元凶。
    _vram_before_release = _gpu_memory_mib()
    try:
        del simulation.context
    except Exception:  # noqa: BLE001
        pass
    try:
        del simulation, integrator
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    _vram_after_release = _gpu_memory_mib()
    if _vram_before_release and _vram_after_release:
        log(
            f"  🧹 attachment 腿 Context 已释放 | 显存 used "
            f"{_vram_before_release[0]} → {_vram_after_release[0]} MiB "
            f"(free {_vram_before_release[1]} → {_vram_after_release[1]} MiB)"
        )
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
    co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
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
        co_alchemical_ion_spec=co_alchemical_ion_spec,
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
    cpu_threads: Optional[int] = None,
):
    """为离线 MBAR worker 构建独立的 OpenMM Context。

    `cpu_threads`：见 `_build_platform_properties`——离线 MBAR 常常是
    `n_workers` 个 CPU 进程同时跑，每个进程的 Context 如果不设线程上限，
    默认会各自尝试用满所有物理核，造成进程数×线程数的过度并行。调用方
    （见下方 `_mbar_worker_init`/`compute_u_kn`）按 `physical_cores //
    n_workers` 算好预算后传进来。
    """
    eval_sys = openmm.XmlSerializer.deserialize(system_xml)
    resolved_platform, props = _build_platform_properties(platform_name, cpu_threads=cpu_threads)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(eval_sys, integ, platform, props)
    return eval_sys, integ, ctx


# 🔑 [性能修复：worker 级 Context 复用] 只有被带 initializer 的 multiprocessing
# Pool 派生出的 worker 子进程才会调用 _mbar_worker_init 把这个填上；直接调用
# _compute_u_kn_chunk 的两条串行路径（n_workers==1 / 多进程失败后的单进程回退，
# 见 compute_u_kn）永远在主进程里跑，这个全局量在主进程里永远是 None，因此这两
# 条路径的行为跟本次改动前逐位一致（各自独立反序列化 System、建 Context）。
_MBAR_WORKER_CTX: Optional[Tuple[Any, Any, Any]] = None


def _mbar_worker_init(system_xml: str, platform_name: str, cpu_threads: Optional[int]) -> None:
    """`multiprocessing.Pool(initializer=...)` 钩子：worker 进程启动时只反
    序列化一次 System、建一次 Context，此后由这个 worker 通过
    `imap_unordered` 拉到的每个 chunk 任务都复用同一份，不再像之前那样
    `_compute_u_kn_chunk` 每次调用都重新反序列化+重新建 Context。只在一次
    `compute_u_kn` 调用内部临时创建的 Pool 生命周期内有效——Pool 退出时
    worker 进程被销毁，不会跨调用残留旧的 System/Context。
    """
    global _MBAR_WORKER_CTX
    _MBAR_WORKER_CTX = _build_traditional_mbar_eval_context(
        system_xml=system_xml,
        platform_name=platform_name,
        cpu_threads=cpu_threads,
    )


def _compute_u_kn_chunk(task: Dict) -> Tuple[int, np.ndarray]:
    """多进程 worker：重算一个帧块在所有 λ 态下的约化势。

    [性能修复] 如果这个进程是被带 `initializer=_mbar_worker_init` 的 Pool
    派生出来的，`_MBAR_WORKER_CTX` 在这个 worker 进程里已经被填好，直接复用
    （同一个 worker 处理的所有 chunk 共享一份 System/Context，不再每个 chunk
    都反序列化+重建）；否则（`n_workers==1` 的串行路径、或多进程失败后的
    单进程回退路径，两者都在主进程里跑，主进程从不调用
    `_mbar_worker_init`，`_MBAR_WORKER_CTX` 在那里恒为 None）保持原来的行为：
    从 `task["system_xml"]`/`task["platform_name"]` 独立建一份、用完即删。
    """
    frame_offset = int(task["frame_offset"])
    xyz_chunk = np.asarray(task["xyz"], dtype=np.float64)
    box_chunk = task.get("box_vectors")
    if box_chunk is not None:
        box_chunk = np.asarray(box_chunk, dtype=np.float64)

    reuse_worker_ctx = _MBAR_WORKER_CTX is not None
    if reuse_worker_ctx:
        eval_sys, integ, ctx = _MBAR_WORKER_CTX
    else:
        eval_sys, integ, ctx = _build_traditional_mbar_eval_context(
            system_xml=task["system_xml"],
            platform_name=str(task["platform_name"]),
            cpu_threads=task.get("cpu_threads"),
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

    if not reuse_worker_ctx:
        # 复用 worker 缓存的 Context 时不能删——同一个 worker 处理的下一个
        # chunk 还要用它。ctx 上的 lambda 参数/positions/box 在下一个 chunk
        # 的循环开头就会被完整覆盖（见上面 for local_f/k 循环），不存在跨
        # chunk 的状态残留风险。
        del ctx, integ, eval_sys
    return frame_offset, u_chunk
def _split_platform_spec(platform_name: str) -> Tuple[str, Optional[str]]:
    spec = str(platform_name or "CPU").strip()
    if ":" not in spec:
        return spec, None
    base, device = spec.split(":", 1)
    base = base.strip() or "CPU"
    device = device.strip() or None
    if device is None:
        raise ValueError(
            f"平台设备索引不能为空：{platform_name!r}；请使用 CUDA、CUDA:N、"
            "OpenCL 或 OpenCL:N"
        )
    return base, device


# [P0-REMD-CUDA] 判 OOM 的下限：一个 45354 原子 PME Context 实测约 315–338 MiB
# （`memtest/probe_remd_context_capacity.py`）。留一倍余量当"还装得下一个"的门槛。
_REMD_CONTEXT_VRAM_FLOOR_MIB = 700.0


def _gpu_memory_mib() -> Optional[Tuple[int, int, int]]:
    """(used, free, total) MiB；拿不到就返回 None。

    刻意用 `nvidia-smi` 子进程而不是 torch/pynvml：这条路径**不能**为了打个日志就
    在本进程里初始化 CUDA（ATT-04 的教训——`abfe_core` 模块级那行 torch 调用曾让
    每个 spawn 子进程在 import 期各抓一个 CUDA context）。诊断绝不能改变被诊断的状态。
    """
    try:
        import subprocess

        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout.strip().splitlines()
        if not out:
            return None
        used, free, total = (int(v.strip()) for v in out[0].split(","))
        return used, free, total
    except Exception:
        # 拿不到显存读数不是错误——诊断失败不该让生产跑挂。
        return None


def _build_platform_properties(
    platform_name: str,
    cpu_threads: Optional[int] = None,
) -> Tuple[str, Dict[str, str]]:
    """`cpu_threads`：仅在 CPU 平台生效，对应 OpenMM CPU 平台的 "Threads"
    属性。默认 None 时保持原来的行为（不设置，OpenMM 自行决定，通常是用满
    所有核）——只有明确知道"这个 Context 只应该用几个线程"的调用方（比如
    离线 MBAR 多进程 worker pool，见 _mbar_worker_init）才会传非 None 值，
    其余调用点（生产/warmup 用的主 CUDA Context 等）不受影响。
    """
    base, device = _split_platform_spec(platform_name)
    upper = base.upper()
    props: Dict[str, str] = {}
    if device is not None:
        if upper not in {"CUDA", "OPENCL"}:
            raise ValueError(
                f"平台 {base!r} 不支持设备索引写法 {platform_name!r}；"
                "只有 CUDA:N/OpenCL:N 合法"
            )
        if not device.isascii() or not device.isdigit():
            raise ValueError(f"平台设备索引必须是非负整数：{platform_name!r}")
    if upper == "CUDA":
        props["Precision"] = "mixed"
        if device is not None:
            props["DeviceIndex"] = device
    elif upper == "OPENCL":
        props["Precision"] = "mixed"
        if device is not None:
            props["DeviceIndex"] = device
    elif upper == "CPU" and cpu_threads is not None:
        props["Threads"] = str(int(cpu_threads))
    return base, props

# ============================================================================
# 0. 通用工具 (从 abfe_core 导入)
# ============================================================================
# ensure_owned_system, sync_all_exclusions, create_ligand_internal_force 已从 abfe_core 导入
# ================= ibs_engine.py / abfe_core.py =================
import openmm
from openmm import app, unit
import numpy as np

def charging_charge_conservation_report(
    nb_force: openmm.NonbondedForce,
    lambda_name: str,
    *,
    ligand_indices: Optional[Sequence[int]] = None,
    co_ion_indices: Optional[Sequence[int]] = None,
    ligand_net_charge_e: Optional[float] = None,
    lambdas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Dict[str, Any]:
    """[§7.2] 逐 λ 电荷账目：总电荷是否恒定、配体/co-ion 是否走在该走的路上。

    直接读**已经配置好的** `NonbondedForce`（粒子基电荷 + 该 λ 名下的 ParameterOffset），
    所以它核对的是"实际会跑的那个哈密顿量"，不是我们打算写进去的那个。

    总电荷守恒是一次代数证明而不是抽查：`Σq(λ) = Σq_base + λ·Σq_scale`，
    所以只要 `Σq_scale = 0`，**所有** λ（含中间态）的总电荷都相同。逐 λ 的数值仍然
    一并算出来落盘，因为它是最容易被人读懂的证据。
    """
    n = nb_force.getNumParticles()
    base = np.zeros(n, dtype=np.float64)
    for i in range(n):
        q, _, _ = nb_force.getParticleParameters(i)
        base[i] = q.value_in_unit(unit.elementary_charge)
    scale = np.zeros(n, dtype=np.float64)
    n_offsets = 0
    for offset_idx in range(nb_force.getNumParticleParameterOffsets()):
        param, particle_idx, charge_scale, _s, _e = nb_force.getParticleParameterOffset(
            offset_idx
        )
        if str(param) != str(lambda_name):
            continue
        scale[int(particle_idx)] += _value_in_elementary_charge(charge_scale)
        n_offsets += 1

    base_sum = float(base.sum())
    scale_sum = float(scale.sum())
    totals = {
        f"{float(lam):.6g}": float(charge_at_lambda(base_sum, scale_sum, float(lam)))
        for lam in lambdas
    }
    report: Dict[str, Any] = {
        "lambda_name": str(lambda_name),
        "n_particle_offsets": int(n_offsets),
        "base_sum_e": base_sum,
        "scale_sum_e": scale_sum,
        "total_charge_by_lambda_e": totals,
        "total_charge_is_lambda_independent": bool(
            abs(scale_sum) <= TOTAL_CHARGE_CONSERVATION_TOLERANCE_E
        ),
        "tolerance_e": float(TOTAL_CHARGE_CONSERVATION_TOLERANCE_E),
    }
    if ligand_indices is not None:
        lig = np.asarray(sorted({int(i) for i in ligand_indices}), dtype=int)
        report["ligand_charge_by_lambda_e"] = {
            f"{float(lam):.6g}": float(
                charge_at_lambda(base[lig].sum(), scale[lig].sum(), float(lam))
            )
            for lam in lambdas
        }
        if ligand_net_charge_e is not None:
            report["ligand_charge_matches_lambda_times_qL"] = all(
                abs(
                    charge_at_lambda(base[lig].sum(), scale[lig].sum(), float(lam))
                    - float(lam) * float(ligand_net_charge_e)
                )
                <= LIGAND_CHARGE_LAMBDA_TOLERANCE_E
                for lam in lambdas
            )
    if co_ion_indices is not None and len(list(co_ion_indices)):
        ions = np.asarray(sorted({int(i) for i in co_ion_indices}), dtype=int)
        report["co_ion_charge_by_lambda_e"] = {
            f"{float(lam):.6g}": float(
                charge_at_lambda(base[ions].sum(), scale[ions].sum(), float(lam))
            )
            for lam in lambdas
        }
    if not report["total_charge_is_lambda_independent"]:
        raise RuntimeError(
            "charging 腿的总电荷随 λ 变化："
            f"Σq_scale = {scale_sum:+.6e} e（容差 {TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:g} e）。\n"
            f"    逐 λ 总电荷：{totals}\n"
            "    PME 会用一个逐 λ 变化的中和背景电荷把这件事掩盖掉，ΔG 静默出错（§7.2）。"
        )
    return report


def _inject_co_alchemical_ion_restraints(
    system: openmm.System,
    co_alchemical_ion_spec: Dict[str, Any],
    log_prefix: str = "  🪢",
) -> List[Dict[str, Any]]:
    """[MEM-00d] 按冻结 spec 注入每个 co-ion 的 flat-bottom 锚点相对位置限制。

    两条路线共用这一处（§4.4 要求复合物腿与溶剂腿"同一函数形式和力常数"，
    co-annihilation 与 charge-transfer 同理没有理由各写一份）。
    所有参数都来自 spec，**不看当前坐标** —— 参考量属于身份的一部分。
    """
    # The restraint stores a Cartesian anchor→ion displacement as a per-bond
    # parameter.  OpenMM barostats scale the box/coordinates but cannot scale
    # that parameter, so using this Hamiltonian in NPT would move the well
    # centre relative to the physical fractional position.  Refuse every
    # barostat variant instead of silently producing a box-size-dependent
    # carrier reservoir.  A future fractional-coordinate implementation must
    # use a new restraint box_model/protocol version.
    barostats = [
        force.__class__.__name__
        for force in system.getForces()
        if "Barostat" in force.__class__.__name__
    ]
    if barostats:
        raise RuntimeError(
            "co-ion Cartesian anchor-relative restraint 仅支持固定盒 NVT；"
            f"检测到 barostat={barostats}。请在注入前移除 barostat，或实现并验证分数坐标 tether。"
        )
    injected: List[Dict[str, Any]] = []
    for ion in co_alchemical_ion_spec["ions"]:
        restraint_spec = ion["restraint"]
        if restraint_spec.get("form") != CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM:
            raise RuntimeError(
                f"co-ion restraint 形式 {restraint_spec.get('form')!r} 不是当前实现的 "
                f"{CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM!r}（MEM-00d）。"
                "旧 spec 不可复用，请重新选择 co-ion 并落盘。"
            )
        if restraint_spec.get("box_model") != CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL:
            raise RuntimeError(
                "co-ion restraint spec 未声明固定盒 box_model；拒绝将旧笛卡尔 tether 注入当前 Hamiltonian。"
            )
        force = _create_co_alchemical_ion_restraint(
            ion_index=int(ion["atom_index"]),
            anchor_atom_index=int(restraint_spec["anchor_atom_index"]),
            reference_displacement_nm=restraint_spec["reference_displacement_nm"],
            force_constant_kj_per_mol_nm2=float(restraint_spec["k_kj_per_mol_nm2"]),
            flat_bottom_radius_nm=float(restraint_spec["flat_bottom_radius_nm"]),
        )
        # §6.4 末条：restraint 逐 λ 相同，且必须待在自己的 force group 里，
        # 不许混进任何 λ 相关的能量分解或 u_kn 差值。
        force.setForceGroup(int(restraint_spec.get(
            "force_group", CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP
        )))
        system.addForce(force)
        d0 = [float(v) for v in restraint_spec["reference_displacement_nm"]]
        injected.append(
            {
                "ion_index": int(ion["atom_index"]),
                "anchor_atom_index": int(restraint_spec["anchor_atom_index"]),
                "reference_displacement_nm": d0,
                "k_kj_per_mol_nm2": float(restraint_spec["k_kj_per_mol_nm2"]),
                "flat_bottom_radius_nm": float(restraint_spec["flat_bottom_radius_nm"]),
                "force_group": int(force.getForceGroup()),
            }
        )
        print(
            f"{log_prefix} co-ion flat-bottom 位置限制已注入: 粒子 {ion['atom_index']} "
            f"↔ 锚点 {restraint_spec['anchor_atom_index']}, "
            f"k={float(restraint_spec['k_kj_per_mol_nm2']):.1f} kJ/mol/nm², "
            f"r₀={float(restraint_spec['flat_bottom_radius_nm']):.2f} nm, "
            f"d0=({d0[0]:.3f}, {d0[1]:.3f}, {d0[2]:.3f}) nm"
        )
    return injected


def configure_charge_transfer_decharging(
    system: openmm.System,
    ligand_indices: List[int],
    topology,
    lambda_name: str = "lam_coul",
    co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, List[int]]:
    """[B3] charge-transfer 的 charging Hamiltonian：ligand q→0 与 co-ion 0→q。

    §2.1 的 λ 定义（`lam_coul` 1 → 0，即 `lambda_q`）：

        q_lig_i(λ) = λ · q_i                    base 0,       scale  q_i
        q_coion(λ) = (1 − λ) · share            base share,   scale −share

    于是 `Σq_lig(λ) + Σq_coion(λ) = q_L` 对**所有** λ 恒成立，全盒总电荷与 λ=1
    的物理体系逐 λ 严格相同（§2.1 末句）。这一点由
    `charging_charge_conservation_report` 在配置完成后**读回真实 Force** 证明，
    而不是靠这段注释。

    与 co-annihilation（`configure_coalchemical_neutral_decharging`）的区别只在
    co-ion 那两行 base/scale，但物理含义相反：那条是"同时湮灭一对异号电荷"，
    这条是"把电荷从结合位点搬到体相水"。两者不可混用，spec 里记着
    `charge_treatment` 并在此处强制核对。

    电荷映射本身来自 `abfe_core.co_alchemical_charge_offset_plan()`——这一层只负责
    把它写进 OpenMM，不重新推导一遍（写歪了会自己对上自己）。
    """
    nb_force = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None
    )
    if nb_force is None:
        raise RuntimeError("系统中未找到 NonbondedForce")

    ligand_set = {int(i) for i in ligand_indices}
    raw_lig_net_charge = 0.0
    ligand_params: Dict[int, Any] = {}
    for idx in sorted(ligand_set):
        q, sig, eps = nb_force.getParticleParameters(idx)
        ligand_params[idx] = (q, sig, eps)
        raw_lig_net_charge += q.value_in_unit(unit.elementary_charge)
    if not np.isfinite(raw_lig_net_charge):
        raise RuntimeError("配体净电荷为 NaN/Inf，拒绝配置 charge-transfer Hamiltonian")
    lig_net_charge = int(round(raw_lig_net_charge))
    if abs(raw_lig_net_charge - lig_net_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
        raise RuntimeError(
            f"配体净电荷 {raw_lig_net_charge:+.6f} e 不接近整数（容差 1e-3 e）"
        )
    if lig_net_charge == 0:
        raise RuntimeError(
            "charge-transfer 被调用，但配体净电荷为 0。中性配体应当走 "
            "`charge_treatment=neutral` 的普通 ligand-only offset 路径 —— "
            "给中性配体造 co-ion 只会凭空加一个不必要的炼金粒子。"
        )
    if co_alchemical_ion_spec is None:
        raise RuntimeError(
            f"检测到带电配体 (Net Charge: {lig_net_charge:+d}) 且路线为 charge-transfer，"
            "但没有传入 `co_alchemical_ion_spec`。\n"
            "    [MEM-00c] co-ion 身份必须由 `select_co_alchemical_ion_once()` 选一次并"
            "落盘，动力学 / REMD 副本 / compute_u_kn / resume 全部只读消费。\n"
            "    **不要**在这里恢复自动选择。"
        )

    pinned = verify_co_alchemical_ion_identity(
        co_alchemical_ion_spec,
        system=system,
        topology=topology,
        charge_treatment=CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=lig_net_charge,
        context="configure_charge_transfer_decharging",
    )
    ion_indices = [int(i) for i in pinned]
    if ligand_set & set(ion_indices):
        raise RuntimeError(
            f"co-ion 与配体原子重叠：{sorted(ligand_set & set(ion_indices))}。"
            "co-ion 必须是体相水里另一个粒子。"
        )

    plan = co_alchemical_charge_offset_plan(
        charge_treatment=CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=lig_net_charge,
        ligand_charges_e={
            idx: ligand_params[idx][0].value_in_unit(unit.elementary_charge)
            for idx in sorted(ligand_set)
        },
        co_ion_physical_charges_e={
            idx: nb_force.getParticleParameters(idx)[0].value_in_unit(
                unit.elementary_charge
            )
            for idx in ion_indices
        },
    )

    existing_params = [
        nb_force.getGlobalParameterName(i)
        for i in range(nb_force.getNumGlobalParameters())
    ]
    if lambda_name not in existing_params:
        nb_force.addGlobalParameter(lambda_name, 1.0)

    original_charges: Dict[int, Any] = {}
    alchemical_set = set(ligand_set) | set(ion_indices)
    for idx, (base_e, scale_e) in sorted(plan["offsets"].items()):
        q, sig, eps = nb_force.getParticleParameters(idx)
        original_charges[idx] = q
        nb_force.setParticleParameters(idx, base_e * unit.elementary_charge, sig, eps)
        nb_force.addParticleParameterOffset(
            lambda_name,
            idx,
            scale_e * unit.elementary_charge,
            0.0 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )

    # co-ion 的 exception：单原子离子在 GROMACS/OpenMM 拓扑里**不该有**任何
    # 1-2/1-3/1-4 exception。真有的话说明它不是单原子离子，而"base 非 0 的粒子
    # 走 exception offset"需要 (base·q_other, −base·q_other) 那一对系数，
    # 与下面配体那段的 (0, chargeProd) 不是同一个式子 —— 所以这里 fail closed
    # 而不是套用配体的写法。
    for exc_idx in range(nb_force.getNumExceptions()):
        p1, p2, _cp, _s, _e = nb_force.getExceptionParameters(exc_idx)
        if int(p1) in ion_indices or int(p2) in ion_indices:
            raise RuntimeError(
                f"co-ion（粒子 {int(p1)} / {int(p2)} 之一）带有 NonbondedForce exception。"
                "charge-transfer 的第一版只支持**单原子** co-ion（§2.2：只改 charge，"
                "mass/sigma/epsilon 逐 λ 不变）；多原子 co-ion 的 exception offset 系数"
                "与配体那套不同，必须单独实现并验证，不许套用。"
            )

    # 配体内部静电的 λ 口径（P0-01 之后的 v3 annihilation）：既有 L–L exception
    # （chargeProd 是独立参数、不读粒子电荷）逐 λ 恒定；普通 L–L 对随粒子 offset
    # 线性湮灭。不再把普通对补成显式 exception——那会改写真实 PME 的 λ=1 端点。
    frozen_ll_pairs = set()
    for exc_idx in range(nb_force.getNumExceptions()):
        p1, p2, charge_prod, sig, eps = nb_force.getExceptionParameters(exc_idx)
        p1, p2 = int(p1), int(p2)
        if p1 in ligand_set and p2 in ligand_set:
            frozen_ll_pairs.add((min(p1, p2), max(p1, p2)))
            continue
        if (p1 in alchemical_set) ^ (p2 in alchemical_set):
            nb_force.setExceptionParameters(
                exc_idx, p1, p2, 0.0 * unit.elementary_charge**2, sig, eps
            )
            nb_force.addExceptionParameterOffset(
                lambda_name,
                exc_idx,
                charge_prod,
                0.0 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
            )

    # [P0-01, 2026-08-30] 同 configure_pme_ligand_charge_offsets：不再把普通
    # L–L 对补成 exception（会改写真实 PME 的 λ=1 物理端点）。既有 L–L
    # exception 冻结、普通 L–L 对随粒子 offset 线性湮灭，口径见上一处注释。

    _inject_co_alchemical_ion_restraints(system, co_alchemical_ion_spec)

    report = charging_charge_conservation_report(
        nb_force,
        lambda_name,
        ligand_indices=sorted(ligand_set),
        co_ion_indices=ion_indices,
        ligand_net_charge_e=lig_net_charge,
    )
    print(
        f"  ⚡ [B3] charge-transfer charging 已配置: 配体 {lig_net_charge:+d} e → 0，"
        f"co-ion {ion_indices} 0 → {lig_net_charge:+d} e（每粒子 ≤ 1 单位电荷）；"
        f"逐 λ 总电荷恒为 {report['base_sum_e']:+.6f} e "
        f"(Σscale={report['scale_sum_e']:+.2e} e)"
    )
    return original_charges, ion_indices


# ================= ibs_engine.py 顶部新增 =================
def configure_coalchemical_neutral_decharging(
    system: openmm.System,
    ligand_indices: List[int],
    topology,
    positions,
    box_vectors=None,
    lambda_name: str = "lam_coul",
    co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, List[int]]:
    """共炼金反离子策略：反离子与配体同步消电，保持全局严格电中性。

    ⚠️ 这是 **co-annihilation**（异号反离子跟着一起消电），**不是** §0 选定的
    charge-transfer（ligand q→0 / co-ion 0→q）。生产路线见 MEM-00a / B3
    （`configure_charge_transfer_decharging`）。本函数按 MEM-00a-2 降级为
    **实验对照专用**，只允许用于水盒 / lipid slab 的方法对照。

    ⚠️ **[MEM-00c] 配体带净电荷时必须传 `co_alchemical_ion_spec`。**
    本函数**不再自己选离子**：身份由 `select_co_alchemical_ion_once()` 选一次、
    落盘，这里只做 `verify_co_alchemical_ion_identity()` 只读核对。
    没有 spec 就 fail closed —— 因为"每个入口自己选一次"在跨进程 resume 下
    会静默选中不同粒子，让 u_kn 与动力学用上不同的 Hamiltonian。
    """
    nb_force = next((f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
    if nb_force is None:
        raise RuntimeError("系统中未找到 NonbondedForce")

    ligand_set = set(int(i) for i in ligand_indices)
    raw_lig_net_charge = 0.0
    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(idx)
        raw_lig_net_charge += q.value_in_unit(unit.elementary_charge)
    if not np.isfinite(raw_lig_net_charge):
        raise RuntimeError("配体净电荷为 NaN/Inf，拒绝配置 co-annihilation Hamiltonian")
    lig_net_charge = int(round(raw_lig_net_charge))
    if abs(raw_lig_net_charge - lig_net_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
        raise RuntimeError(
            f"配体净电荷 {raw_lig_net_charge:+.6f} e 不接近整数（容差 1e-3 e）"
        )

    if lig_net_charge == 0:
        print("  ℹ️ 配体为电中性，无需共消电反离子。使用标准 PME Offset。")
        best_ion_indices: List[int] = []
        ion_meta: Dict[str, Any] = {}
    else:
        # [MEM-00c] 只核对，不重选。没有 spec 就停 —— 见本函数 docstring。
        if co_alchemical_ion_spec is None:
            raise RuntimeError(
                f"检测到带电配体 (Net Charge: {lig_net_charge:+d})，但没有传入 "
                "`co_alchemical_ion_spec`。\n"
                "    [MEM-00c] co-ion 身份必须由 `select_co_alchemical_ion_once()` "
                "**选一次并落盘**，动力学 / REMD 副本 / compute_u_kn / resume 全部只读消费。\n"
                "    本函数以前会在这里自己调一次选择器；因为选择结果是坐标的连续函数，"
                "而首跑（预平衡输出 + 2000 步最小化）与 resume（直接读 DCD 末帧）"
                "喂进来的坐标不同，于是同一条腿的动力学与 u_kn 可能选中**不同粒子** —— "
                "ΔG 会错且没有任何异常现象。\n"
                "    修法是在上游选一次、把 spec 传下来；**不要**在这里恢复自动选择。"
            )
        pinned = verify_co_alchemical_ion_identity(
            co_alchemical_ion_spec,
            system=system,
            topology=topology,
            ligand_net_charge_e=lig_net_charge,
            context="configure_coalchemical_neutral_decharging",
        )
        if (
            co_alchemical_ion_spec.get("charge_treatment")
            == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
        ):
            raise RuntimeError(
                "传进来的是 charge-transfer 的 co-ion 身份 spec，但调用的是 "
                "co-annihilation 的 charging builder。两者的哈密顿量相反"
                "（q_phys→0 vs 0→q_L），请改调 `configure_charge_transfer_decharging`。"
            )
        best_ion_indices = list(pinned)
        # restraint 参数全部取自 spec，不取当前坐标（MEM-00c/00d：参考量属于身份的
        # 一部分）。注入由 `_inject_co_alchemical_ion_restraints` 统一负责，
        # 与 charge-transfer 共用同一份形式（§4.4）。
        ion_meta = dict(co_alchemical_ion_spec.get("selection_provenance") or {})
        print(
            f"  🔒 [MEM-00c] 复用已冻结的共炼金反离子身份: Indices {best_ion_indices} "
            f"(fingerprint {str(co_alchemical_ion_spec.get('fingerprint'))[:12]}…)"
        )

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

    # [P0-01, 2026-08-30] 配体内部普通 L–L 对（≥1-5、无 exception）**不再**补成
    # 显式 exception。原因：OpenMM 的 exception 从不使用普通非键的 cutoff/PME
    # 处理（API 文档 "cutoffs are never applied to exceptions"；OpenMM #2310 也
    # 记录了把参数相同的普通对改成 exception 时 PME 能量/力会变），补对会让
    # λ=1 的物理端点不再是原始 System——tests/test_pme_decharge_endpoint_
    # equivalence.py 在 26 粒子最小 PME 体系上实测能量偏差 ~1e-2 kJ/mol。
    # 新口径（PME_DECHARGE_MODEL_VERSION → v4，annihilation）：
    #   - 既有 L–L exception（1-2/1-3 排除、1-4 缩放）的 chargeProd 是独立参数、
    #     不读粒子电荷 → 逐 λ 恒定（frozen_ll_pairs 的冻结语义不变）；
    #   - 普通 L–L 对随粒子 offset 线性缩放 → 配体内部库仑随 λ 湮灭。
    # complex/solvent 两腿的配体内部 Hamiltonian 完全相同，湮灭项在 ΔG_bind 里
    # 严格相消，热力学循环闭合。

    if best_ion_indices:
        _inject_co_alchemical_ion_restraints(system, co_alchemical_ion_spec)
        # §7.2：逐 λ 电荷账目由同一个函数核对（与 charge-transfer 共用），
        # 不靠"异号反离子应该会抵消"这句话。
        charging_charge_conservation_report(
            nb_force,
            lambda_name,
            ligand_indices=sorted(ligand_set),
            co_ion_indices=best_ion_indices,
            ligand_net_charge_e=lig_net_charge,
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
    co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
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
    if not np.isfinite(net_charge):
        raise RuntimeError("配体净电荷为 NaN/Inf，拒绝配置 PME 去电荷阶段")
    rounded_charge = int(round(net_charge))
    if abs(net_charge - rounded_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
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
        if topology is None:
            raise RuntimeError(
                "带电配体的 PME 去电荷路径需要 topology 才能核对冻结的 co-ion 身份。"
            )
        # 走哪条共炼金路线由**冻结 spec 里记录的 charge_treatment** 决定，不是由这里
        # 再猜一次。理由：spec 的端点电荷已经按某一条路线算好了（q_phys→0 还是 0→q_L），
        # 用另一条路线的 builder 消费它就是"声明一种哈密顿量、实际跑另一种"。
        # spec 缺失时保持既有 co-annihilation 分支的 fail-closed 报错口径（MEM-00c）。
        treatment = (co_alchemical_ion_spec or {}).get("charge_treatment")
        if treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
            original_charges, ion_indices = configure_charge_transfer_decharging(
                system,
                ligand_indices,
                topology,
                lambda_name=lambda_name,
                co_alchemical_ion_spec=co_alchemical_ion_spec,
            )
            mode = "co_alchemical_charge_transfer"
        else:
            if positions is None:
                raise RuntimeError(
                    "co-annihilation 的 PME 去电荷路径需要初始 positions。"
                )
            original_charges, ion_indices = configure_coalchemical_neutral_decharging(
                system,
                ligand_indices,
                topology,
                positions,
                box_vectors=box_vectors,
                lambda_name=lambda_name,
                # [MEM-00c] 身份只读透传；为 None 时下游 fail closed，不再自动选。
                co_alchemical_ion_spec=co_alchemical_ion_spec,
            )
            mode = "coalchemical_counterion"
        return {
            "mode": mode,
            "charge_treatment": treatment,
            "ion_indices": [int(i) for i in ion_indices],
            "n_offsets": len(original_charges),
            "co_alchemical_ion_fingerprint": (
                (co_alchemical_ion_spec or {}).get("fingerprint")
            ),
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

    # [P0-01, 2026-08-30] 配体内部普通 L–L 对（≥1-5、无 exception）**不再**补成
    # 显式 exception。原因：OpenMM 的 exception 从不使用普通非键的 cutoff/PME
    # 处理（API 文档 "cutoffs are never applied to exceptions"；OpenMM #2310 也
    # 记录了把参数相同的普通对改成 exception 时 PME 能量/力会变），补对会让
    # λ=1 的物理端点不再是原始 System——tests/test_pme_decharge_endpoint_
    # equivalence.py 在 26 粒子最小 PME 体系上实测能量偏差 ~1e-2 kJ/mol。
    # 新口径（PME_DECHARGE_MODEL_VERSION → v4，annihilation）：
    #   - 既有 L–L exception（1-2/1-3 排除、1-4 缩放）的 chargeProd 是独立参数、
    #     不读粒子电荷 → 逐 λ 恒定（frozen_ll_pairs 的冻结语义不变）；
    #   - 普通 L–L 对随粒子 offset 线性缩放 → 配体内部库仑随 λ 湮灭。
    # complex/solvent 两腿的配体内部 Hamiltonian 完全相同，湮灭项在 ΔG_bind 里
    # 严格相消，热力学循环闭合。
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

# 🔑 [MEM-00h，2026-08-06] vdW/softcore 非键协议版本——单独一个版本号，只覆盖
# softcore cutoff/switching（1.2nm+switch → 1.0nm 无switch）这一次改动本身，
# 不复用 TRADITIONAL_LJ_LRC_PROTOCOL_VERSION。
#
# 为什么不直接 bump TRADITIONAL_LJ_LRC_PROTOCOL_VERSION：那个字段目前是
# `stage_type in {"coul","vdw"}` 都写、都校验（见 convergence 字典构造与
# _resume_cached_window_gate_status）——即便 LRC 概念上只跟 vdW 有关。沿用它会
# 把 Stage 1 charging 的缓存也一起判废，而 Stage 1 charging 完全不受这次改动
# 影响（它不构造软核 CV，SOFTCORE_CUTOFF_NM/switching 跟它无关）。这是本仓库
# 明确要求的边界：只作废 Stage 2 vdW window 与对应 u_kn，Stage 1 charging、
# Boresch attachment、预平衡、C1 的 charging 轨迹一律不许重跑。
#
# 因此这个新版本号只在 stage_type=="vdw" 时写入 convergence.json、只在
# stage_type=="vdw" 时参与 resume 门判定（见 _resume_cached_window_gate_status
# 的 stage_type 形参与 vdw_nb_version_match）；stage_type=="coul" 的窗口完全
# 不受影响，缺这个字段也不会被判无效。
VDW_NONBONDED_PROTOCOL_VERSION = 1

# ============================================================================
# [LIGAND_COM_RESTRAINT_PROTOCOL_VERSION]
#   version 1: Group 5 = CustomCentroidBondForce，非周期绝对锚点
#              r_com=sqrt((x1-x0)^2+(y1-y0)^2+(z1-z0)^2)，usesPBC=False，
#              锚点取自输入构型的配体 COM。**已确认为 P0 缺陷，见
#              4W53/STAGE2_GROUP5_CUDA_PBC_ROOT_CAUSE_2026-08-29.md。**
#   version 2: 移除该力（当前）。
#
# 缺陷的准确定义：固定锚点使用未折叠周期像，而 CUDA 动力学中的 centroid 被折叠到
# 主盒；非周期绝对距离把两个不同周期像当成真实距离，形成永久激活、跨边界不连续的
# 错误外力。注意"锚点在主盒外"本身不是错误——CPU 上锚点与 centroid 成像一致，
# E5 恒为 0、F5 恒为 0，行为完全正常。错的是**两侧成像规则不一致**。
#
# 实测（同一生产 System，λ_vdw=0，初始化后不再 setPositions，enforcePeriodicBox=False）：
#     CUDA  err_fold 恒为 0.000，|F5| 12–256 kJ/mol/nm，COM 未折叠位移 110 ps 内
#           3.9 → 30.9 nm 单调增长（自由扩散 RMS 仅约 3.0 nm），跨盒 10 次且力方向跳变
#     CPU   err_raw  恒为 0.000，|F5| 恒为 0.00，COM 位移与自由扩散一致
# 后果：CUDA 轨迹是带定向粒子流的非平衡过程，不满足 MBAR 的"样本来自已知平衡分布"
# 前提。静态 setPositions() 测试与任何 CPU 测试都**检不出**它。
#
# 为什么是移除而不是改成周期最小像：该力只在**没有 Boresch 锚定**时才添加，即只存在
# 于溶剂腿；而均匀溶剂中它不提供任何必要约束。B/C 受控对照（4 个态 × 湿干 × 3 种子）
# 显示：删除组与最小像修复组的湿/干结构统计与局部 BAR 自由能在 ±0.25 kJ/mol 预设
# 等价界内一致（6→8 局部总自由能只差 0.001 kJ/mol），而最小像组在 6→7 边的正向 ESS
# 显著更低（24.8 vs 177.1）。即：保留它没有热力学收益，还可能降低有限时间混合效率。
# 复合物腿本来就跳过它（依赖 Boresch），因此本次改动不影响复合物腿的 Hamiltonian。
#
# ⚠️ 若将来确需 COM 限制，**不要**再用绝对锚点。本文件 `build_co_alchemical_ion_restraint`
# 的注释里已记录可用写法：`periodicdistance` 只存在于 CustomExternalForce；
# `CustomCompoundBondForce` 打开 PBC 后其中的 `pointdistance` 就是 minimum-image 距离。
# 复合物腿还必须先判断它是否与 Boresch 限制重复计入。
# ============================================================================
LIGAND_COM_RESTRAINT_PROTOCOL_VERSION = 2

# 两条路径（ACE `_create_softcore_force` 与传统 `BeutlerSoftcoreBuilder.build`）
# 目前都硬编码用这组 switching/cutoff 距离构造软核 CustomNonbondedForce，LRC 积分
# 必须用完全相同的边界，否则修正的是一个跟实际采样哈密顿量不匹配的积分区间。
#
# [MEM-00h，2026-08-06] 统一到 1.0 nm、无 switching 后，switch==cutoff，下面积分
# 公式里 [r_switch, r_cutoff] 那段"switching 壳层补偿"区间宽度精确为 0（自动贡献
# 0，不需要为"switching 已关闭"另写一条积分分支），只剩 [r_cutoff, ∞) 的标准
# r^-6/r^-12 尾项——这正是"解析 LRC 不再补偿 1.0-1.2nm switching 壳层，只积分
# 1.0nm→∞"这句话在代码里的落地方式。这两个值仅作函数默认参数；实际调用（见
# build_ibs_dual_system）都会传入从活的 softcore force 读回的 template_cutoff/
# template_switch，因此哪怕只改了 _create_softcore_force 忘了改这里，行为也不会
# 悄悄分叉——但两处都改是为了让默认值本身也如实反映现状，不留一个说谎的常量。
LJ_TAIL_LRC_R_SWITCH_NM = 1.0
LJ_TAIL_LRC_R_CUTOFF_NM = 1.0


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
        use_switching = True
    else:
        # [MEM-00h] 统一到基础力场的 1.0 nm、无 switching——见 SOFTCORE_CUTOFF_NM
        # 定义处的理由。switch_nm 特意设成等于 cutoff_nm（而不是干脆不设）：
        # 下游 build_ibs_dual_system 会读 cv_template.getSwitchingDistance() 去算
        # LJ 解析尾项的积分下限（见 _lj_softcore_tail_radial_integrals），switch==
        # cutoff 让那段"switching 壳层"积分区间宽度精确为 0，自动退化成纯
        # cutoff→∞ 的标准尾项，不需要为"switching 已关闭"另开一条积分公式分支。
        cutoff_nm = SOFTCORE_CUTOFF_NM
        switch_nm = SOFTCORE_CUTOFF_NM
        use_switching = False
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
                
    sc_force.setUseSwitchingFunction(use_switching)
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
    """质量加权 COM。

    ⚠️ [LIGAND_COM_RESTRAINT_PROTOCOL_VERSION=2] 唯一的调用点（Group 5 COM 限制力）
    已被移除，本函数目前无调用方，保留仅为将来可能需要的**周期**COM 限制。
    若要重新启用：绝不可再把它的返回值当作非周期绝对锚点使用——那正是被移除的
    P0 缺陷（见该常量的注释）。可用写法见 `build_co_alchemical_ion_restraint`。
    """
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
    dispersion_protocol: Optional[str] = None,
    # [B6-FIX] 这条腿的环境（soluble/membrane）。非 legacy 色散路线必须给，
    # 否则 `resolve_leg_dispersion_implementation` 会 raise —— 刻意不设可猜的默认值。
    environment_type: Optional[str] = None,
    *,
    # EXP-025 G4 Layer-2 (2026-08-13): optional, disabled-by-default native
    # shared residual basis, threaded straight into IBSBiasForce (see that
    # class's docstring for the exact contract). Every EXISTING call site
    # that never passes these keeps IBSBiasForce.residual_enabled=False and
    # therefore the byte-identical Group-1 expression -- verified by
    # scripts/test_exp025_g4_ibsbiasforce_native_residual.py, not just
    # asserted here.
    residual_basis_force: Optional[openmm.Force] = None,
    residual_state_coefficients: Optional[Sequence[float]] = None,
    residual_energy_offset_kj_mol: float = 0.0,
) -> Tuple[openmm.System, 'IBSBiasForce']:
    """
    构建双λ IBS 采样系统 (终极修复版：彻底剥离 Group 0 的 λ 依赖)
    """
    if len(lambdas_coul) != len(lambdas_vdw):
        raise ValueError("lambdas_coul 与 lambdas_vdw 必须等长")
    if len(lambdas_coul) < 2:
        raise ValueError("IBS dual-lambda ensemble 至少需要两个状态")
    if len(lambdas_vdw) > IBS_DUAL_MAX_LAMBDA_STATES:
        raise RuntimeError(
            "单个 IBS ensemble 的 lambda 状态过多："
            f"K={len(lambdas_vdw)}, 每态 {IBS_DUAL_CVS_PER_LAMBDA_STATE} 个 CV，"
            f"将超过 OpenMM CustomCVForce 的 {OPENMM_CUSTOM_CV_MAX_VARIABLES}-CV 上限。"
            "请把 vanishing 域划为多个物理子区间；禁止使用单一 [0:K] ensemble。"
            "\n\n⚠️ 这条限制来自 **Group-1 IBS 混合偏置力**（CustomCVForce 一次最多引用 "
            f"{OPENMM_CUSTOM_CV_MAX_VARIABLES} 个 CV），**不是** λ 态数本身的物理上限。"
            "逐态独立固定-λ 采样（INDEPENDENT_ENDPOINT_PROTOCOL_VERSION）根本不用这个"
            "混合偏置力——它只需要 λ 无关的 U_common 和逐态的 _int_cv_force_xmls[k]。"
            "若你要的是独立采样而非 IBS，请按 <=%d 态分块调用本函数、再把各块的 "
            "CV/LRC 拼起来（重叠态可用来校验同一 λ 在不同块建出的 CV 逐字节相同），"
            "不要为了绕过这条限制去改 IBS 偏置力的结构。" % IBS_DUAL_MAX_LAMBDA_STATES
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
    lambda_vdw_arr = np.asarray(lambdas_vdw, dtype=float)
    if (
        lambda_coul_arr.ndim != 1
        or lambda_vdw_arr.ndim != 1
        or not np.all(np.isfinite(lambda_coul_arr))
        or not np.all(np.isfinite(lambda_vdw_arr))
    ):
        raise ValueError("IBS lambda 数组必须是一维且全部为有限数")
    if (
        np.any((lambda_coul_arr < 0.0) | (lambda_coul_arr > 1.0))
        or np.any((lambda_vdw_arr < 0.0) | (lambda_vdw_arr > 1.0))
    ):
        raise ValueError("IBS lambda 必须位于 [0, 1]")
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
    lig_net_charge_raw = 0.0
    for idx in perturbed_indices:
        q, _, _ = all_params[idx]
        lig_net_charge_raw += q.value_in_unit(unit.elementary_charge)
    if not np.isfinite(lig_net_charge_raw):
        raise RuntimeError("配体净电荷为 NaN/Inf，拒绝构建 IBS Hamiltonian")
    lig_net_charge = int(round(lig_net_charge_raw))
    if (
        abs(lig_net_charge_raw - lig_net_charge)
        > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E
        or lig_net_charge != 0
    ):
        raise RuntimeError(
            f"检测到带净电或部分净电配体 ({lig_net_charge_raw:+.6f} e)。"
            "净电荷必须先接近整数并且为 0；"
            "IBS 的 VDW 阶段不再静态改写反离子电荷；"
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
    # PME decharging v4 uses annihilation for ordinary ligand–ligand
    # Coulomb pairs.  At the charging endpoint (lambda_coul=0) those particle
    # charges are zero while pre-existing 1-4 exception chargeProd terms stay
    # frozen.  Rebuild U_common with exactly that endpoint convention so the
    # Stage-1/Stage-2 seam is an identity, rather than reintroducing the raw
    # ligand charges here and leaving a constant but real Hamiltonian jump.
    group2_params = list(all_params)
    for idx in perturbed_set:
        _q, _sig, _eps = group2_params[idx]
        group2_params[idx] = (0.0 * unit.elementary_charge, _sig, _eps)
    internal_ref_excl = [(p1, p2) for p1, p2 in ref_excl if p1 in perturbed_set and p2 in perturbed_set]
    ll_f, ll_14_f = create_ligand_internal_force(
        original_nb, perturbed_indices, group2_params, internal_ref_excl, num_atoms, system=new_sys
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

    # ---------- Group 5（配体 COM 限制）已移除 ----------
    # [LIGAND_COM_RESTRAINT_PROTOCOL_VERSION=2] 详见文件上方该常量的长注释。
    # 旧实现是非周期绝对锚点的 CustomCentroidBondForce；在 CUDA 上 centroid 被折叠进
    # 主盒而锚点没有，绝对距离把两个周期像当成真实距离，产生永久激活、跨边界跳变的
    # 外力，实测让配体以远超自由扩散的速度定向漂移（110 ps 内 30.9 nm），使轨迹不再
    # 满足 MBAR 的平衡采样前提。CPU 上不复现，静态 setPositions() 测试也检不出。
    #
    # 该力本来就只在**没有 Boresch 锚定**时才添加（复合物腿一直跳过它），所以它只存在
    # 于溶剂腿，而均匀溶剂中它不提供任何必要约束。B/C 受控对照证明删除它与改成周期
    # 最小像在结构统计与局部自由能上实用等价，且删除的单向重加权效率更好。
    #
    # 结论：不再添加任何 Group 5 力。**Group 5 现为空组**——`e_base = groups{0,2,3,5}`
    # 的记账口径（WCA_ACCOUNTING_VERSION=2）不受影响，该组只是贡献 0。
    if _has_valid_boresch_restraint(restraint_params):
        print("  ℹ️ 复合物腿由 Boresch 锚定定位配体；Group 5 COM 限制力已于"
              " LIGAND_COM_RESTRAINT_PROTOCOL_VERSION=2 全局移除。")
    else:
        print("  ℹ️ 溶剂腿不添加 Group 5 COM 限制力"
              " (LIGAND_COM_RESTRAINT_PROTOCOL_VERSION=2)：旧的非周期绝对锚点实现在 CUDA 上"
              "产生定向拖拽，均匀溶剂中该限制亦无必要。")

    # ---------- 7. IBS 偏置力与纯 VDW 软核 CV (Group 1) ----------
    if residual_basis_force is None:
        ibs_wrapper = IBSBiasForce(len(lambdas_coul), temperature, prefix=prefix)
    else:
        ibs_wrapper = IBSBiasForce.with_residual_basis(
            len(lambdas_coul),
            temperature,
            prefix=prefix,
            residual_basis_force=residual_basis_force,
            residual_state_coefficients=residual_state_coefficients,
            residual_energy_offset_kj_mol=residual_energy_offset_kj_mol,
        )
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
    template_use_switching = cv_template.getUseSwitchingFunction()
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
    if not ibs_lj_tail_lrc_is_applicable(
        potential_type, dispersion_protocol, environment_type
    ):
        print(
            "  ⚠️ [LJ LRC] "
            f"{ibs_lj_tail_lrc_inapplicable_reason(potential_type, dispersion_protocol, environment_type)}"
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
        # [MEM-00h，2026-08-06] 之前这里无条件写 True，会把 _create_softcore_force
        # 按 potential_type 已经算好的 use_switching 悄悄覆盖回"总是开 switching"——
        # 对当前生产的 softcore（非 dexp）路径来说，那会让每个 λ_vdw 态实际使用的
        # CV 力switching 状态跟 cv_template（同一份构造逻辑理应产出同样的状态）不
        # 一致。改成跟随 template 的真实状态，而不是硬编码常量。
        int_f_cv.setUseSwitchingFunction(template_use_switching)
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

    # EXP-025 G4 Layer-2: fail closed HERE, before the Force ever enters the
    # System/before any Context can be created, if the per-state loop above
    # ever left a cv_k_int/cv_k_rest (or, when residual_basis_force is set,
    # the shared exp025_residual_basis) unregistered or duplicated. This is
    # not a hypothetical -- see g4_layer1_oracle_report.json's
    # real_bug_found_and_fixed_during_this_gate for the exact silent-wrong-
    # energy failure mode this closes.
    ibs_wrapper.validate_wiring()
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
    # OpenMM returns the explicitly committed PME parameters here when the
    # caller used setPMEParameters().  Prefer those values: deriving alpha
    # from the tolerance is only equivalent for OpenMM's automatic-parameter
    # path and can silently disagree with a manually frozen grid/alpha.
    try:
        explicit_alpha, nx, ny, nz = nb.getPMEParameters()
        alpha_explicit = (
            float(explicit_alpha.value_in_unit(1 / unit.nanometer))
            if hasattr(explicit_alpha, "value_in_unit")
            else float(explicit_alpha)
        )
        grid = (int(nx), int(ny), int(nz))
    except Exception:
        alpha_explicit = 0.0
        grid = (0, 0, 0)
    if alpha_explicit > 0.0 or any(v != 0 for v in grid):
        if (
            not math.isfinite(alpha_explicit)
            or alpha_explicit <= 0.0
            or any(v <= 0 for v in grid)
        ):
            raise RuntimeError(
                "NonbondedForce 的显式 PME alpha/grid 不完整或非法；"
                "拒绝用 cutoff/tolerance 猜测另一个 Hamiltonian。"
            )
        return alpha_explicit

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


def _pme_static_parameter_signature(nb: openmm.NonbondedForce) -> Dict[str, Any]:
    """Return the committed PME alpha/grid for provenance and cache checks.

    OpenMM reports zeros when alpha/grid are automatic.  In that case we still
    record the fact explicitly; callers that need an actual alpha use
    :func:`get_pme_alpha_for_system`, which applies the same closed-form
    derivation as OpenMM.  Keeping this helper separate avoids silently
    dropping an explicitly frozen grid from the Shadow bridge diagnostics.
    """
    try:
        alpha, nx, ny, nz = nb.getPMEParameters()
        alpha_nm = _value_in_inverse_nanometer(alpha)
        grid = [int(nx), int(ny), int(nz)]
    except Exception:
        alpha_nm = 0.0
        grid = [0, 0, 0]
    explicit = bool(alpha_nm > 0.0 or any(v != 0 for v in grid))
    return {
        "alpha_per_nm": float(alpha_nm),
        "grid": grid,
        "explicit": explicit,
    }


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
    # Do not round before testing: +0.4e is not a neutral ligand and must not
    # pass merely because round(+0.4) == 0. Use the central integer-charge
    # tolerance as the single source of truth for this experimental path.
    if not np.isfinite(net_q):
        raise RuntimeError(
            f"配体净电荷为 NaN/Inf ({net_q!r})，Shadow-Coulomb 只支持有限中性配体"
        )
    nearest_integer = int(round(net_q))
    if (
        abs(net_q - nearest_integer) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E
        or nearest_integer != 0
    ):
        raise RuntimeError(
            f"检测到非中性配体 (Net Charge: {net_q:+.6f} e)。Shadow-Coulomb 去电荷支路"
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
    cutoff_nm: float,
) -> openmm.CustomNonbondedForce:
    """
    A(配体)-E(环境) 短程"影子"库仑势：lambda * ONE_4PI_EPS0 * q1*q2 * erfc(alpha*r)/r，
    只在 addInteractionGroup(A, E) 之间计算（不含 A-A/E-E），避免引入 lambda^2 项，
    与旧 co-alchemical 去电荷的线性 offset 约定保持一致。lambda 以字面常量刻入表达式
    （不是 runtime 可调的 global parameter），供 IBS 多态窗口 CV（每态一个实例，
    lambda_shadow_coul_k 不同）或 Bridge 的"满强度探针"（lambda=1.0）场景复用同一
    个 builder。
    """
    cutoff_nm = float(cutoff_nm)
    if not math.isfinite(cutoff_nm) or cutoff_nm <= 0.0:
        raise ValueError(f"Shadow-Coulomb cutoff 必须为正有限数，收到 {cutoff_nm!r}")
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
    # Explicit PME alpha/grid settings are part of the Hamiltonian. Merely
    # copying method/cutoff/tolerance can make OpenMM choose a different grid,
    # breaking the inclusion-exclusion identity.
    if reference_nb.getNonbondedMethod() in (
        openmm.NonbondedForce.PME,
        openmm.NonbondedForce.LJPME,
    ):
        try:
            alpha, nx, ny, nz = reference_nb.getPMEParameters()
            probe.setPMEParameters(alpha, nx, ny, nz)
        except Exception:
            # Older OpenMM versions may report automatic (zero) parameters.
            pass
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
    temperature=None,
) -> Tuple[openmm.System, "IBSBiasForce"]:
    """
    构建 Shadow-Coulomb IBS 去电荷系统：Shadow full charge (λ=1) -> Shadow decharged
    (λ=0)，只处理 A(配体)-E(环境) 的短程 erfc(alpha*r)/r "影子"电荷交叉项，VdW 全程
    满强度不炼金 (属于 U_common)。复用 IBSBiasForce 的多态 log-sum-exp 偏置机制
    （与 build_ibs_dual_system 的 VDW CV 完全同一套框架，只是把 CV 换成静电）。
    """
    if not lambdas_shadow_coul:
        raise ValueError("Shadow-Coulomb 至少需要一个 lambda 状态")
    if len(lambdas_shadow_coul) < 2:
        raise ValueError("Shadow-Coulomb 至少需要两个 lambda 状态以定义去电荷腿")
    if len(lambdas_shadow_coul) > IBS_DUAL_MAX_LAMBDA_STATES:
        raise ValueError(
            f"Shadow-Coulomb lambda 状态数 {len(lambdas_shadow_coul)} 超过 CustomCVForce "
            f"32 变量上限对应的 {IBS_DUAL_MAX_LAMBDA_STATES} 态上限"
        )
    lambda_values = np.asarray(lambdas_shadow_coul, dtype=float)
    if not np.all(np.isfinite(lambda_values)):
        raise ValueError("Shadow-Coulomb lambda 必须全部为有限数")
    if np.any((lambda_values < 0.0) | (lambda_values > 1.0)):
        raise ValueError("Shadow-Coulomb lambda 必须位于 [0, 1]")
    if not np.isclose(lambda_values[0], 1.0) or not np.isclose(lambda_values[-1], 0.0):
        raise ValueError(
            "Shadow-Coulomb lambda 必须以 1.0（满电荷）开始、以 0.0（去电荷）结束"
        )
    # The production Shadow-Coulomb leg is defined as full charge (1) ->
    # decharged (0).  Keep that orientation explicit so endpoint labels and
    # bridge bookkeeping cannot be silently reversed.
    lambda_diffs = np.diff(lambda_values)
    if np.any(lambda_diffs > 0.0) and np.any(lambda_diffs < 0.0):
        raise ValueError("Shadow-Coulomb lambda 必须单调（允许 1 -> 0 或 0 -> 1）")
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
    if temperature is None:
        raise ValueError("Shadow-Coulomb IBS 必须显式提供采样温度")
    temp_quantity = (
        temperature if hasattr(temperature, "value_in_unit") else float(temperature) * unit.kelvin
    )
    ibs_wrapper = IBSBiasForce(len(lambdas_shadow_coul), temp_quantity, prefix=prefix)
    cutoff_nm = nb.getCutoffDistance().value_in_unit(unit.nanometer)
    cv_template = _build_shadow_coul_cross_force(
        alpha_ewald, original_params, perturbed_indices, env_indices,
        lambda_value=0.0, exclusions=shadow_excl, cutoff_nm=cutoff_nm,
    )
    template_excl = {
        tuple(sorted(map(int, cv_template.getExclusionParticles(i))))
        for i in range(cv_template.getNumExclusions())
    }
    for k, lam in enumerate(lambdas_shadow_coul):
        int_f_cv = _build_shadow_coul_cross_force(
            alpha_ewald, original_params, perturbed_indices, env_indices,
            lambda_value=float(lam), exclusions=shadow_excl, cutoff_nm=cutoff_nm,
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
    pme_static_signature = _pme_static_parameter_signature(nb)

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
        cutoff_nm=nb.getCutoffDistance().value_in_unit(unit.nanometer),
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
        "reference_pme_parameters": pme_static_signature,
        "reference_pme_cutoff_nm": float(
            nb.getCutoffDistance().value_in_unit(unit.nanometer)
        ),
        "shadow_coulomb_cutoff_nm": float(
            nb.getCutoffDistance().value_in_unit(unit.nanometer)
        ),
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
    """IBS 偏置力封装 (Group 1) - 数值稳定差分形式

    EXP-025 G4 Layer-2 (2026-08-13): optional, disabled-by-default native
    shared residual basis. When ``residual_basis_force`` is None (the
    default), this class's behavior is BYTE-IDENTICAL to before this
    addition -- same expression string, same CVs, same globals, same force
    group. This is deliberate and load-bearing: production callers that
    never pass the new keyword-only arguments must see zero behavior
    change, verified by an XML-diff regression test (see
    scripts/test_exp025_g4_ibsbiasforce_native_residual.py), not just
    argued informally.

    When enabled, each state's discriminant argument becomes
    ``X_k = cv_k_int + cv_k_rest + A_k*(exp025_residual_basis - U_offset) - f_k``
    with the shared basis Force registered as exactly ONE collective
    variable (never duplicated per state) inside the SAME Group-1
    CustomCVForce -- the plugin executes once per force evaluation, and the
    max-pivot log-sum-exp discriminant is otherwise unchanged.
    """

    @classmethod
    def with_residual_basis(
        cls,
        n_states: int,
        temperature: openmm.unit.Quantity,
        prefix: str = "abfe",
        *,
        residual_basis_force: Optional[openmm.Force] = None,
        residual_state_coefficients: Optional[Sequence[float]] = None,
        residual_energy_offset_kj_mol: float = 0.0,
    ) -> "IBSBiasForce":
        """Construct the opt-in residual variant without changing the legacy API."""
        instance = cls.__new__(cls)
        instance._residual_constructor_config = {
            "residual_basis_force": residual_basis_force,
            "residual_state_coefficients": residual_state_coefficients,
            "residual_energy_offset_kj_mol": residual_energy_offset_kj_mol,
        }
        cls.__init__(instance, n_states, temperature, prefix)
        return instance

    def __init__(
        self,
        n_states: int,
        temperature: openmm.unit.Quantity,
        prefix: str = "abfe",
        **residual_kwargs,
    ):
        residual_config = getattr(self, "_residual_constructor_config", None)
        if residual_kwargs:
            allowed_residual_kwargs = {
                "residual_basis_force",
                "residual_state_coefficients",
                "residual_energy_offset_kj_mol",
            }
            unexpected = sorted(set(residual_kwargs) - allowed_residual_kwargs)
            if unexpected:
                raise TypeError(
                    "IBSBiasForce.__init__() got unexpected keyword argument(s): "
                    f"{', '.join(unexpected)}"
                )
            if residual_config is not None:
                raise TypeError(
                    "IBSBiasForce: residual configuration was supplied both through "
                    "with_residual_basis() and direct constructor keywords"
                )
            # Preserve the EXP-025 direct keyword path for existing callers. New
            # production wiring uses the named classmethod, while this path keeps
            # older scripts and downstream integrations behaviorally unchanged.
            residual_config = {
                "residual_basis_force": residual_kwargs.get("residual_basis_force"),
                "residual_state_coefficients": residual_kwargs.get("residual_state_coefficients"),
                "residual_energy_offset_kj_mol": residual_kwargs.get(
                    "residual_energy_offset_kj_mol", 0.0
                ),
            }
        if residual_config is None:
            residual_basis_force = None
            residual_state_coefficients = None
            residual_energy_offset_kj_mol = 0.0
        else:
            residual_basis_force = residual_config["residual_basis_force"]
            residual_state_coefficients = residual_config[
                "residual_state_coefficients"
            ]
            residual_energy_offset_kj_mol = residual_config[
                "residual_energy_offset_kj_mol"
            ]
        self.n_states = n_states
        self.prefix = prefix
        self._cv_keeper = []
        self._int_cv_force_xmls = []
        # [性能修复：主 Context 一次读全部 CV] 与 _int_cv_force_xmls 平行的索引表——
        # 每个 "_int" CV 在主 CustomCVForce (self.force) 里的真实注册索引，供
        # IBSSampler._evaluate_interaction_energies_live() 用
        # getCollectiveVariableValues() 一次性取全部值，不必重建 probe System。
        self._int_cv_indices: List[int] = []
        self._residual_basis_cv_index: Optional[int] = None
        self.residual_enabled = residual_basis_force is not None
        if not self.residual_enabled:
            if residual_state_coefficients is not None:
                raise ValueError(
                    "IBSBiasForce: residual_state_coefficients must be None when residual_basis_force is None "
                    "(both or neither -- see class docstring)"
                )
            if residual_energy_offset_kj_mol != 0.0:
                raise ValueError(
                    "IBSBiasForce: residual_energy_offset_kj_mol must be 0.0 when residual_basis_force is None "
                    "(both or neither -- see class docstring)"
                )
        else:
            if residual_state_coefficients is None or len(residual_state_coefficients) != n_states:
                raise ValueError(
                    f"IBSBiasForce: residual_state_coefficients must have exactly n_states={n_states} entries "
                    f"when residual_basis_force is provided (got "
                    f"{None if residual_state_coefficients is None else len(residual_state_coefficients)})"
                )
            if not all(math.isfinite(float(c)) for c in residual_state_coefficients):
                raise ValueError("IBSBiasForce: residual_state_coefficients must all be finite")
            if not math.isfinite(float(residual_energy_offset_kj_mol)):
                raise ValueError("IBSBiasForce: residual_energy_offset_kj_mol must be finite")
            # Fail closed on the CV budget HERE, before any addCollectiveVariable()
            # call -- do not rely on OpenMM to reject an over-budget CustomCVForce
            # at some later, less predictable point (matches the EXP-025 G4
            # Layer-1 lesson: a wiring contract violation must be caught by an
            # explicit check, not "it didn't throw so it must be fine").
            required_cv_count = 2 * n_states + 1
            if required_cv_count > OPENMM_CUSTOM_CV_MAX_VARIABLES:
                raise ValueError(
                    f"IBSBiasForce: enabling the native residual basis needs {required_cv_count} collective "
                    f"variables (2*n_states+1) for n_states={n_states}, exceeding the hard ceiling of "
                    f"{OPENMM_CUSTOM_CV_MAX_VARIABLES} -- i.e. n_states must be <= "
                    f"{(OPENMM_CUSTOM_CV_MAX_VARIABLES - 1) // 2} with the residual basis enabled"
                )
        self.residual_state_coefficients = (
            [float(c) for c in residual_state_coefficients] if self.residual_enabled else None
        )
        self.residual_energy_offset_kj_mol = float(residual_energy_offset_kj_mol)
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
        # EXP-025 G4 Layer-2: when residual_enabled, state k's parenthesized
        # term gains "+ (A_k)*(exp025_residual_basis-(offset))" before the
        # "- {prefix}_f_{k}" close. When NOT enabled, _state_expr(k) returns
        # EXACTLY the original "(cv_{k}_int + cv_{k}_rest - {prefix}_f_{k})"
        # string, byte-for-byte -- verified by
        # scripts/test_exp025_g4_ibsbiasforce_native_residual.py, not just
        # asserted here. Coefficients formatted with 17 significant digits
        # (round-trips a double exactly), matching the existing
        # OuterLambdaIBSBiasForce/OuterLambdaResidualBiasForce convention;
        # a state whose OWN coefficient is exactly 0.0 (e.g. a genuine
        # physical endpoint) omits the term entirely rather than adding a
        # numerically-inert "+0.0*(...)" -- also matching that convention.
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=30] 新增 {prefix}_s_residual：独立于
        # bias_scale 的残差耦合开关，默认 1.0（不显式调用 setParameter 时行为
        # 与 v29 逐字节相同）。原因见 2026-08-25 EXP-030 window_0 诊断：EM 阶段
        # bias_scale 仍是构造时默认值 1.0，candidate 在冷启动 f_k=0 时就已经用
        # 全强度残差力做无温控 L-BFGS 最小化，配对 baseline/candidate 密度探针
        # 显示两者在同一起点逐点一致，candidate 之后局部环境原子数雪崩、baseline
        # 全程稳定——但把 bias_scale 直接压到 0 会连 baseline 本来就在正常使用、
        # EM 时稳定贡献真实力的物理 softcore-state 混合项一起关掉，不是同一个
        # Hamiltonian。s_residual 只切残差项，EM 时物理软核混合力保持跟 baseline
        # 一样活跃。
        def _state_expr(k: int) -> str:
            if self.residual_enabled and self.residual_state_coefficients[k] != 0.0:
                coeff = self.residual_state_coefficients[k]
                offset = self.residual_energy_offset_kj_mol
                return (
                    f"(cv_{k}_int + cv_{k}_rest + {prefix}_s_residual*({coeff:.17g})*"
                    f"(exp025_residual_basis - ({offset:.17g})) - {prefix}_f_{k})"
                )
            return f"(cv_{k}_int + cv_{k}_rest - {prefix}_f_{k})"

        logit_exprs = {}
        for k in range(1, n_states):
            # 相对 logit_k = -beta * (X_k - X_0)
            diff_expr = f"{_state_expr(k)} - {_state_expr(0)}"
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
        # V_bias = X_0 - kt * (M + log(sum(exp(logit_i - M))))
        # 注意：第一项 (X_0...) 是坐标依赖的，必须包含在内以保证力的正确性。
        # _state_expr(0) already carries its own parens, matching exactly
        # what the original literal "(cv_0_int + cv_0_rest - {prefix}_f_0)"
        # substring looked like when residual_enabled is False.
        energy_expr = (
            f"{prefix}_bias_scale * ({_state_expr(0)} "
            f"- kt * (({pivot_expr}) + log(max(1e-300, {sum_expr}))))"
        )

        self.force = openmm.CustomCVForce(energy_expr)
        self.force.addGlobalParameter("kt", kt)
        self.force.addGlobalParameter("beta", beta)
        self.force.addGlobalParameter(f"{prefix}_bias_scale", 1.0)
        if self.residual_enabled:
            self.force.addGlobalParameter(f"{prefix}_s_residual", 1.0)
        for k in range(n_states):
            self.force.addGlobalParameter(f"{prefix}_f_{k}", 0.0)
        self.force.setForceGroup(1)
        if self.residual_enabled:
            # Registered here (inside __init__, not left to the caller) --
            # unlike cv_k_int/cv_k_rest (which the caller must still register
            # via addCollectiveVariable(), since only the caller has the
            # per-state softcore Forces to hand), this ONE shared CV is fully
            # determined by the constructor's own arguments, so there is no
            # reason to defer it and risk the exact "caller forgot to
            # register it" class of bug that G4 Layer-1 hit.
            self._cv_keeper.append(residual_basis_force)
            self._residual_basis_cv_index = self.force.addCollectiveVariable(
                "exp025_residual_basis", residual_basis_force
            )

    def addCollectiveVariable(self, name: str, cv_force: openmm.Force) -> int:
        if name == "exp025_residual_basis":
            raise ValueError(
                "IBSBiasForce: the 'exp025_residual_basis' collective variable name is reserved "
                "(registered automatically by __init__ when residual_basis_force is provided)"
            )
        self._cv_keeper.append(cv_force)
        # 索引必须是 self.force.addCollectiveVariable() 的**真实返回值**，不能事后按
        # 位置算（比如 2*k）——residual_enabled 时 __init__ 已经先注册了
        # exp025_residual_basis，会把后续所有索引整体偏移 1（第 3934 行那个调试分支
        # 用 idx_int=2*k 硬算就是这个坑，新代码不能重犯）。
        idx = self.force.addCollectiveVariable(name, cv_force)
        if name.endswith("_int"):
            self._int_cv_force_xmls.append(openmm.XmlSerializer.serialize(cv_force))
            # [性能修复：主 Context 一次读全部 CV] 记住这个 "_int" CV 在主
            # CustomCVForce 里的索引，供 IBSSampler._evaluate_interaction_energies_live()
            # 直接用 getCollectiveVariableValues() 一次性取全部值，不必再重建一个
            # 完全独立的 probe System/Context 逐个 force group 查询。
            self._int_cv_indices.append(idx)
        return idx

    def get_force(self) -> openmm.CustomCVForce:
        return self.force


    def get_residual_basis_energy(self, context) -> float:
        """Return the exact residual CV value used by the Group-1 score."""
        if not self.residual_enabled:
            return 0.0
        values = self.force.getCollectiveVariableValues(context)
        value = float(values[self._residual_basis_cv_index])
        if not math.isfinite(value):
            raise RuntimeError("non-finite residual basis energy")
        return value

    def get_sampling_state_coefficients(self) -> np.ndarray:
        values = self.residual_state_coefficients or [0.0] * self.n_states
        return np.asarray(values, dtype=np.float64)

    def setForceGroup(self, group_id: int):
        self.force.setForceGroup(group_id)

    def validate_wiring(self) -> None:
        """Fail closed BEFORE any Context is created if this Force's
        CustomCVForce is missing a collective variable or global parameter
        its own constructed expression actually references, or has a
        duplicate name registered. See
        outer_lambda_neural_basis.OuterLambdaResidualBiasForce.validate_wiring
        for why this exists (EXP-025 G4 Layer-1, 2026-08-13): a missing
        cv_k_int/cv_k_rest registration on that sibling class silently
        produced a wrong, finite energy with no exception at all. This
        class's cv_k_int/cv_k_rest are STILL the caller's responsibility
        (see addCollectiveVariable() above) -- only exp025_residual_basis is
        self-registered -- so callers must call this after registering all
        per-state CVs, before creating any Context.
        """
        expected_cv_names = {f"cv_{k}_int" for k in range(self.n_states)}
        expected_cv_names |= {f"cv_{k}_rest" for k in range(self.n_states)}
        if self.residual_enabled:
            expected_cv_names.add("exp025_residual_basis")

        actual_cv_names = [
            self.force.getCollectiveVariableName(i) for i in range(self.force.getNumCollectiveVariables())
        ]
        if len(actual_cv_names) != len(set(actual_cv_names)):
            duplicates = sorted({name for name in actual_cv_names if actual_cv_names.count(name) > 1})
            raise ValueError(f"IBSBiasForce.validate_wiring: duplicate collective variable name(s) {duplicates}")
        actual_cv_set = set(actual_cv_names)
        if actual_cv_set != expected_cv_names:
            missing = sorted(expected_cv_names - actual_cv_set)
            unexpected = sorted(actual_cv_set - expected_cv_names)
            raise ValueError(
                "IBSBiasForce.validate_wiring: collective variable set does not match this Force's own "
                f"contract -- missing={missing}, unexpected={unexpected}. Every cv_{{k}}_int/cv_{{k}}_rest "
                "for k in range(n_states) must be registered by the caller via addCollectiveVariable() before "
                "creating any Context."
            )

        expected_global_names = {"kt", "beta", f"{self.prefix}_bias_scale"}
        if self.residual_enabled:
            expected_global_names.add(f"{self.prefix}_s_residual")
        expected_global_names |= {f"{self.prefix}_f_{k}" for k in range(self.n_states)}
        actual_global_names = {
            self.force.getGlobalParameterName(i) for i in range(self.force.getNumGlobalParameters())
        }
        missing_globals = expected_global_names - actual_global_names
        if missing_globals:
            raise ValueError(f"IBSBiasForce.validate_wiring: missing required global parameter(s) {sorted(missing_globals)}")

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


# The four-argument constructor is the stable public API. The runtime
# ``**residual_kwargs`` compatibility path above is intentionally kept out of
# introspection so existing API-contract checks and callers see that surface
# unchanged.
IBSBiasForce.__init__.__signature__ = inspect.Signature(
    parameters=[
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("n_states", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("temperature", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter("prefix", inspect.Parameter.POSITIONAL_OR_KEYWORD, default="abfe"),
    ]
)

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
# v30（2026-08-25）：新增 {prefix}_s_residual，最小化阶段对 residual_enabled
# 的 arm 临时关闭残差耦合（EM 时 bias_scale 仍是构造默认值 1.0，candidate 在
# 冷启动 f_k=0 下会用全强度残差力做无温控 L-BFGS 最小化；配对 baseline/candidate
# 密度探针显示同一起点逐点一致、candidate 之后局部环境原子数雪崩撞上
# max_environment_atoms 硬上限、baseline 全程稳定，见 EXP-030 window_0 诊断），
# 最小化后（四条 is_resumed_ibs 分支的任意一条、或没走这四条分支时紧跟着的
# "[偏置预热]"爬坡块）统一恢复到 1.0。这是真实的 production Hamiltonian 变化
# （EM 阶段的有效势能变了，不是纯 warmup 停止判据/诊断），不能跟 v27-29 一样
# 保持缓存兼容——residual_enabled 的窗口如果在 v30 之前已经跑过 EM，那次 EM
# 是在全强度残差力下做的，跟 v30 的物理过程不是同一件事，旧 f_k 不能被当成
# v30 的有效热启动/冻结验证结果直接续用。baseline（residual_enabled=False）
# 不受影响：s_residual 只在 residual_enabled 时才注册和被设置。
# v31（2026-08-26）：修复 f_k 在线学习目标与实际部署目标不一致的 bug——
# collect_energies() 里喂给 self.energy_buffer（进而 update_weights()/
# _solve_tmbar_and_recenter() 的 u_mk）的量此前一直是纯物理 softcore_energies，
# 完全不含残差项；但实际驱动采样动力学的 Group-1 偏置力公式是
# X_k = U_k + s_residual·A_k·B_φ − f_k（v30 引入 s_residual 时改的
# _state_expr(k)）。residual_enabled 的窗口因此在拿一份不含残差的训练信号去
# 学习如何拉平一个含残差的分布，学出来的 f_k 系统性学不对——EXP-030 生产数据
# 三个独立 repeat 的 candidate 臂占据分布系统性偏斜（跟 baseline 的近似均匀
# 形成鲜明对比）证实了这一点。修复后改用已经算出来但此前只存进
# self.sampling_state_energy_history、从未接入训练链路的 sampling_state_energies
# （含残差，full-strength，不随 bias_scale/s_residual 爬坡缩放——这是有意的，
# 物理项那边训练目标本来就不跟着 bias_scale 爬坡缩放）。这是真实的训练目标
# 变化：v30 之前用 residual_enabled 学出来的所有 f_k，都是在错误目标下学的，
# 不能被当成 v31 的有效热启动/冻结验证结果直接续用。baseline
# （residual_enabled=False）不受影响：sampling_state_energies 在这种情况下
# 就是 softcore_energies 的原样拷贝，这次改动对 baseline 逐字节不变，但为了
# 不悄悄放过 candidate 窗口，版本号检查本身不区分 arm，直接把兼容集合收窄成
# 只有 v31 自己，代价只是 baseline 也会多冷启动一次（跟 v30 断点时的处理方式
# 一致）。
IBS_BIAS_PROTOCOL_VERSION = 31

# v28/v29 都只改过 warmup 的停止/诊断控制，没有改变 production Hamiltonian、
# f_k 符号约定或生产采样方式，所以那次是把 27/28/29 都放进同一个兼容集合。
# v30/v31 都是真正的断裂点（分别见上面各自的版本说明），缓存兼容集合只留最新
# 版本自己。
IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS = frozenset((31,))


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
    current_coion_identity: Optional[Dict[str, Any]] = None,
    stage_type: str = "coul",
    current_sampling_score_sha256: Optional[str] = None,
    current_stage_protocol_key: Optional[Dict] = None,
) -> Dict[str, Any]:
    """一份磁盘窗口缓存在 resume 时能不能直接复用：10 个门的纯函数求值。

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

    ``usable`` 的与链顺序与原内联判断完全一致，并额外绑定本窗口实际采样的
    co-ion runtime identity 与联合评分身份：
        shape -> lambdas -> wca -> ibs_bias -> lse -> lrc -> repair_policy
        -> coion_identity -> score_identity -> early_stop
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

    # 🔑 [ligand_com_restraint_protocol_version] v1 的 Group 5 配体 COM 限制力在
    # CUDA 上产生永久激活、跨边界跳变的定向拖拽（锚点用未折叠像、centroid 被折叠，
    # 非周期绝对距离把两个周期像当成真实距离），使轨迹不满足 MBAR 的平衡采样前提。
    # 见 4W53/STAGE2_GROUP5_CUDA_PBC_ROOT_CAUSE_2026-08-29.md 与
    # LIGAND_COM_RESTRAINT_PROTOCOL_VERSION 定义处。
    #
    # ⚠️ 这道门不能只挂在 stage 级指纹上：stage 结果因 v1→v2 失效后会重新进入
    # run_all_windows，而**逐窗口**的 energies/bias/base 复用发生在建 Context 之前，
    # 走的是这个门。少了它，旧 Group5 窗口的能量数组仍会被直接复用。
    # 该力只在没有 Boresch 锚定时添加（溶剂腿 stage1/stage2 都有），所以这道门对
    # coul 与 vdw **都**生效，不做 stage_type 豁免。
    # 缺该字段（v1 及更早的缓存根本不写）一律判不可复用。
    com_restraint_version_match = (
        cached_conv.get("ligand_com_restraint_protocol_version")
        == LIGAND_COM_RESTRAINT_PROTOCOL_VERSION
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

    # 🔑 [vdw_nonbonded_protocol_version，MEM-00h] softcore cutoff/switching 只
    # 影响 Stage 2 vdW window（构造软核 CV 的那条路径），跟 Stage 1 charging
    # 完全无关——charging 阶段不建软核 CV，SOFTCORE_CUTOFF_NM 对它没有意义。
    # 所以这道门只在 stage_type=="vdw" 时才生效；stage_type=="coul" 一律
    # pass-through（True），不因为这个字段缺失就把充电阶段的缓存判废。
    vdw_nb_version_match = (
        True
        if stage_type != "vdw"
        else cached_conv.get("vdw_nonbonded_protocol_version")
        == VDW_NONBONDED_PROTOCOL_VERSION
    )

    # 🔑 [non_mutating_v1] 采样修复策略必须匹配：旧的变异策略缓存（其 f_k 可能被
    # fixed-H 累计 ΔF 就地覆盖过，属于不同参考系）绝不能被非变异策略的 run 复用。
    # 旧缓存没有这个字段（None），与 "non_mutating_v1" 不相等，因此自动判无效。
    repair_policy_match = cached_conv.get("sampling_repair_policy") == repair_policy

    # EXP-030: completed-window reuse happens before IBSSampler.load_ibs_state,
    # so the immutable score-family identity must be checked at this early gate.
    sampling_score_match = cached_conv.get("sampling_score_sha256") == current_sampling_score_sha256
    # A matching score hash is not enough if the runtime contract itself was
    # changed.  Baseline caches intentionally stay compatible when no score
    # identity is in use.
    residual_protocol_match = (
        current_sampling_score_sha256 is None
        or cached_conv.get("residual_sampling_protocol_version")
        == IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
    )

    # 能量/偏置/base 三件套来自具体 co-ion Hamiltonian 的采样分布；仅检查
    # checkpoint manifest 太晚，因为完整能量文件会在建 Context 前被提前复用。
    # neutral 的旧缓存没有该字段，None == None 保持兼容；任何当前 identity
    # 非 None 的运行都要求 convergence.json 有完全相等的 payload。
    cached_coion_identity = cached_conv.get("coion_identity")
    coion_identity_match = cached_coion_identity == current_coion_identity

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

    segment_metadata_match = False
    if shape_ok and cached_conv.get("production_segment_protocol_version") == PRODUCTION_SEGMENT_PROTOCOL_VERSION:
        try:
            _validate_production_segments(cached_conv.get("production_segments"), cached_e_shape[1])
            segment_metadata_match = True
        except (ValueError, TypeError):
            pass
    stage_protocol_match = (
        current_stage_protocol_key is None
        or _stage_window_sampling_identity(cached_conv.get("stage_protocol_key"))
        == _stage_window_sampling_identity(current_stage_protocol_key)
    )
    usable = bool(
        segment_metadata_match
        and stage_protocol_match
        and shape_ok
        and lambdas_match
        and version_match
        and bias_protocol_match
        and com_restraint_version_match
        and lse_tolerance_match
        and lrc_version_match
        and vdw_nb_version_match
        and repair_policy_match
        and sampling_score_match
        and residual_protocol_match
        and coion_identity_match
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
    elif not com_restraint_version_match:
        reason = (
            f"ligand_com_restraint_protocol_version="
            f"{cached_conv.get('ligand_com_restraint_protocol_version')!r}"
            f"（期望 {LIGAND_COM_RESTRAINT_PROTOCOL_VERSION}）；v1 的 Group 5 COM "
            "限制力在 CUDA 上产生定向拖拽，该窗口轨迹不满足平衡采样前提"
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
    elif not vdw_nb_version_match:
        reason = (
            f"vdw_nonbonded_protocol_version="
            f"{cached_conv.get('vdw_nonbonded_protocol_version')!r}"
            f"（期望 {VDW_NONBONDED_PROTOCOL_VERSION}，stage_type={stage_type!r}）"
        )
    elif not repair_policy_match:
        reason = (
            f"sampling_repair_policy="
            f"{cached_conv.get('sampling_repair_policy')!r}（期望 {repair_policy!r}）"
        )
    elif not sampling_score_match:
        reason = (
            "sampling_score_sha256 mismatch: "
            f"cached={cached_conv.get('sampling_score_sha256')!r}, "
            f"current={current_sampling_score_sha256!r}"
        )
    elif not residual_protocol_match:
        reason = (
            "residual_sampling_protocol_version mismatch: "
            f"cached={cached_conv.get('residual_sampling_protocol_version')!r}, "
            f"current={IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION!r}"
        )
    elif not coion_identity_match:
        reason = (
            "co-ion runtime identity 不匹配或缺失："
            f"缓存={cached_coion_identity!r}，当前={current_coion_identity!r}"
        )
    elif not segment_metadata_match:
        reason = "production segment metadata 缺失、版本错误或帧范围不完整"
    elif not stage_protocol_match:
        reason = "stage_protocol_key 与当前采样身份不匹配"
    elif not early_stop_ok:
        reason = early_stop_reject_reason
    else:
        reason = "λ 值不匹配（缺少 λ 元数据或 λ 路径已变更）"

    return {
        "usable": usable,
        "reason": reason,
        "segment_metadata_match": segment_metadata_match,
        "stage_protocol_match": stage_protocol_match,
        "shape_ok": shape_ok,
        "lambdas_match": lambdas_match,
        "version_match": version_match,
        "bias_protocol_match": bias_protocol_match,
        "com_restraint_version_match": com_restraint_version_match,
        "lse_tolerance_match": lse_tolerance_match,
        "cached_lse_tolerance": cached_lse_tolerance,
        "lrc_version_match": lrc_version_match,
        "vdw_nb_version_match": vdw_nb_version_match,
        "repair_policy_match": repair_policy_match,
        "sampling_score_match": sampling_score_match,
        "residual_protocol_match": residual_protocol_match,
        "coion_identity_match": coion_identity_match,
        "cached_coion_identity": cached_coion_identity,
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
# [Candidate-first, Validate-or-Learn v1] 9 -> 10：LEARN 不再调用
# update_weights()/全历史 TMBAR 选择器，改为纯 bounded-occupancy 反馈；冻结
# 时机也从"固定 min_bias_updates 批 + 连续通过"改成"raw residual 一旦
# ≤ IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW 立即冻结"。语义变化足以让旧协议下
# 未完成的 in-flight learning 缓存被新逻辑误读，必须让它们版本失配后
# 老实重新预热，不能被当成新语义下的中间状态注入。
IBS_WARMUP_UPDATE_PROTOCOL_VERSION = 10
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
        # Exact Group-1 state energies and residual basis, distinct from the
        # physical target ledger which excludes the sampling-only residual.
        self.sampling_state_energy_history = []
        self.residual_basis_history = []
        # New checkpoints preserve real f trajectories; old ones still load
        # through the update-count fallback below.
        self.sampling_score_family_sha256 = None
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
        # 🔑 [2026-08-27，见 EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md]
        # 只在 run_all_windows 真正进入生产采样这一刻被设置一次，随
        # save_ibs_state/load_ibs_state 落盘/恢复。None 表示这份冻结 f_k
        # 还没有被观测到过"第一次进入生产"的时刻——续采时绝不能用当前
        # attempt 重新读到的 Context 值覆盖已恢复的旧值，那会用"现在"冒充
        # "最初"。
        self.production_entry_f_k = None
        # [Candidate-first, Validate-or-Learn v1] 纯附加元数据，不参与
        # run_all_windows 的任何控制流分支。seed_source 由实际生效的 SEED
        # 路径写入：load_ibs_state() 成功即写 "resume"；pilot/bootstrap
        # 热启动在 run_all_windows 里写 "pilot"/"bootstrap"；LEARN 第一次
        # 真正改动 f_k 时改写为 "learned"。
        self.seed_source: Optional[str] = None
        self.validation_attempts = 0
        self.last_failure_reason: Optional[str] = None
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
        # 直接调用、不吞异常——probe context 用来算的 CV 里有基于 PBC 的距离项，
        # 如果这里静默失败，probe_sys 会留着 OpenMM System 的默认盒子（跟
        # main_system 完全不一致），后面 evaluate_interaction_energies() 算出来
        # 的相互作用能就会是错的盒子下的错误值，而它直接喂给在线 IBS bias 更新
        # 和 TMBAR，没有任何信号说这里出过问题。main_system 的盒矢量本身已经在
        # 真实 Context 里跑通，正常情况下这一步不该失败；真失败了就该在这里炸。
        probe_sys.setDefaultPeriodicBoxVectors(*main_system.getDefaultPeriodicBoxVectors())

        # 🔑 [构建前检查] 探针 Context 给每个 λ 态分配一个独立 force group，
        # 起始号 PROBE_FORCE_GROUP_BASE=16，而 OpenMM 的 force group 上限是 31
        # ⟹ 单个探针 Context 最多容纳 32-16 = 16 个 λ 态。
        # 不做这个检查的话，第 17 个态会在 setForceGroup(32) 上抛
        # `OpenMMException: Force group must be between 0 and 31`——那句话既不说明
        # 是哪一层的限制，也不告诉调用方该怎么办。
        # ⚠️ 这是**探针 Context 的**限制，与 build_ibs_dual_system 里那条
        # IBS_DUAL_MAX_LAMBDA_STATES（CustomCVForce 的 32-CV 上限）是两回事：
        # 后者约束 IBS 混合偏置力，独立固定态采样根本不用它。
        n_cv = len(self.ibs_wrapper._int_cv_force_xmls)
        if n_cv > PROBE_MAX_LAMBDA_STATES:
            raise RuntimeError(
                f"探针 Context 无法容纳 {n_cv} 个 λ 态：每态占用一个 force group，"
                f"起始号 {PROBE_FORCE_GROUP_BASE}，OpenMM 上限 "
                f"{OPENMM_MAX_FORCE_GROUP} ⟹ 单次最多 {PROBE_MAX_LAMBDA_STATES} 个态。"
                "请把 λ 区间拆成多段分别求值；不要通过降低 PROBE_FORCE_GROUP_BASE "
                "来腾空间——16 以下的组号被生产 Hamiltonian 的力占用（0/1/2/3/4/5）。"
            )
        self._probe_groups = []
        for idx, force_xml in enumerate(self.ibs_wrapper._int_cv_force_xmls):
            force = openmm.XmlSerializer.deserialize(force_xml)
            gid = PROBE_FORCE_GROUP_BASE + idx
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
            # 同上：不吞异常。调用方显式传了这一帧的盒子，说明它认为盒子形状变了
            # （NPT/膜 barostat 之后常见），这里静默失败就意味着 probe context
            # 继续用上一次成功设置的旧盒子去算这一帧的相互作用能——这是一个
            # 实实在在喂进采样/统计的物理量，不是诊断边角料，不能猜、不能吞。
            self._probe_context.setPeriodicBoxVectors(*box_vectors)

        interaction_energies = np.zeros(self.n_states, dtype=float)
        for k, gid in enumerate(self._probe_groups[:self.n_states]):
            state = self._probe_context.getState(getEnergy=True, groups={gid})
            interaction_energies[k] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

        return interaction_energies

    def _evaluate_interaction_energies_live(self) -> np.ndarray:
        """[性能修复：主 Context 一次读全部 CV] `collect_energies()` 的热路径永远读
        的是 `self.context` **此刻**的活体状态（生产/warmup 主循环紧跟
        `sim.step()` 之后调用，中间没有任何 `setPositions`）——而每个 `cv_k_int`
        早在 `build_ibs_dual_system`/`build_shadow_coul_ibs_system` 里就已经被
        注册成了 `self.ibs_wrapper.force`（Group 1，驱动真实动力学的那个
        CustomCVForce）的 collective variable。既然主 Context 自己就知道全部
        K 个 CV 的值，`getCollectiveVariableValues()` 一次调用就能拿到全部，
        完全不需要 `evaluate_interaction_energies()` 那条路径——重新反序列化
        K 个力、建一个独立 probe System/Context、再逐个 force group 发起
        `getState()`——的 K 次 GPU 同步。

        ⚠️ 只能用在"当前坐标就是 self.context 此刻坐标"的场景。`evaluate_
        interaction_energies(positions, box_vectors)` 仍然是**唯一**给任意
        历史/候选帧（fixed-H probe bank、bidirectional overlap 探针）求值的入口，
        原样保留，不受这次改动影响。
        """
        indices = self.ibs_wrapper._int_cv_indices[: self.n_states]
        cv_vals = self.ibs_wrapper.get_force().getCollectiveVariableValues(self.context)
        return np.asarray([cv_vals[i] for i in indices], dtype=float)

    def _collect_interaction_energies(
        self,
        reuse_positions=None,
        reuse_box_vectors=None,
    ) -> Tuple[np.ndarray, Any]:
        """返回 (interaction_energies, box_vectors)。

        [性能修复：主 Context 一次读全部 CV] 这个方法的两个调用方
        (`collect_energies()`/`get_raw_interaction_energies()`) 在整个仓库里
        从不传显式坐标——读的永远是 `self.context` 此刻的活体状态，也就是说
        `positions` 参数本身从未真正需要被喂给一个独立 probe context（那只是
        旧实现为了"逐 λ 查询"绕的路）。现在直接从主 Context 的 CustomCVForce
        读全部 CV（`_evaluate_interaction_energies_live()`），`reuse_positions`
        参数保留只是为了不用改调用方签名，不再被使用。`reuse_box_vectors` 仍然
        有效：省掉一次只为了拿盒子而单独发起的 `getState()`，返回值同时喂给
        `_lj_tail_correction_kj_mol` 算体积。
        """
        if reuse_box_vectors is not None:
            box_vectors = reuse_box_vectors
        else:
            try:
                box_vectors = self.context.getState().getPeriodicBoxVectors()
            except Exception:
                box_vectors = None
        return self._evaluate_interaction_energies_live(), box_vectors

    def get_raw_interaction_energies(self) -> np.ndarray:
        energies, _ = self._collect_interaction_energies()
        return energies.copy()

    def _lj_tail_correction_kj_mol(self, reuse_box_vectors=None) -> np.ndarray:
        """解析 LJ 长程色散尾项修正（逐态,加到 interaction_energies 上）。
        lj_tail_lrc_coeff_kj_mol（switching+softcore-aware，见
        _lj_tail_lrc_coefficients_kj_mol）由 build_ibs_dual_system 预计算并挂在
        ibs_wrapper 上；这里每帧只需读当前盒子体积做一次除法，不重新积分。
        缺失/不适用 (potential_type='dexp' 或没有周期性盒子) 时返回全 0，
        不影响原有行为。fixed-H overlap 探针（probe_bidirectional_overlap）
        和这里读的是同一个 ibs_wrapper.lj_tail_lrc_coeff_kj_mol，保证两处
        用的是同一组系数。

        [性能修复] `reuse_box_vectors` 允许调用方传入本帧已经取到的 box
        vectors（例如 `_collect_interaction_energies()` 刚返回的那一份），
        省掉一次只为了拿盒子体积就单独发起的 getState() 调用——之前这里
        无条件独立查询一次，跟同一次 collect_energies() 里稍早已经查过的
        盒子完全重复。默认 None 时保持原来独立查询的行为。
        """
        lrc_coeff = getattr(self.ibs_wrapper, "lj_tail_lrc_coeff_kj_mol", None) if self.ibs_wrapper is not None else None
        if lrc_coeff is None:
            return np.zeros(self.n_states, dtype=float)
        try:
            if reuse_box_vectors is not None:
                box_vectors = reuse_box_vectors
            else:
                box_vectors = self.context.getState().getPeriodicBoxVectors()
            if box_vectors is None:
                raise ValueError("LRC 需要周期性盒向量，但当前 Context 未提供")
            box_nm = box_vectors.value_in_unit(unit.nanometer)
            a, b, c = (np.asarray(v, dtype=np.float64) for v in box_nm)
            volume_nm3 = abs(np.dot(a, np.cross(b, c)))
        except Exception as exc:
            raise RuntimeError("LJ 长程尾项体积计算失败，拒绝以零修正继续") from exc
        if not np.isfinite(volume_nm3) or volume_nm3 <= 0.0:
            raise RuntimeError(f"LJ 长程尾项盒体积非法: {volume_nm3!r}")
        coeff = np.asarray(lrc_coeff, dtype=np.float64).ravel()
        if coeff.size != self.n_states:
            raise RuntimeError(
                "LJ 长程尾项系数长度与 sampler 状态数不符："
                f"{coeff.size} != {self.n_states}"
            )
        if not np.all(np.isfinite(coeff)):
            raise RuntimeError("LJ 长程尾项系数含 NaN/Inf")
        return coeff / volume_nm3

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

    def collect_energies(
        self,
        *,
        reuse_positions=None,
        reuse_box_vectors=None,
        reuse_e_base=None,
        reuse_e_bias=None,
    ) -> np.ndarray:
        """[性能修复] `reuse_positions`/`reuse_box_vectors`：生产控制面的
        guard 代码块（见 `run_all_windows` 主循环）在 `do_force_check=True`
        的 update 上已经做过一次 `getState(getPositions=True, ...)`，把结果
        传进来可以省掉这里独立再查一次同一个 context、同一时刻的
        positions/box——两者读到的是完全相同的物理状态，中间没有任何动力学
        演化。默认 None 时保持原来独立查询的行为，其余调用点（冷启动/warmup/
        legacy 校准循环等）不用改。

        [性能修复：合并 guard 查询] `reuse_e_base`/`reuse_e_bias`：guard 弱分支
        （`do_force_check=False`）现在自己直接查 `groups={0,2,3,5}`/`{1,4}`
        （见 `run_all_windows`），因为生产窗口系统（`build_ibs_dual_system`/
        `build_shadow_coul_ibs_system`）的力只落在 Group 0-5，`e_base+e_bias`
        恒等于不过滤的 `getState(getEnergy=True)` 总能量——guard 用这两个数
        既能判"总能量是否有限"，又能把它们喂给这里，省掉这里本来还要再查一遍
        同一个瞬时状态的两次 `getState()`。guard 只有在两者都有限时才会走到
        这个调用（否则在 guard 里就已经 rollback + continue，不会进这个函数），
        所以下面两段"非有限"分支理论上不会触发，仍然保留是为了不静默吞掉任何
        未来误用（例如新调用点传了没检查过的值）。
        """
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
        if reuse_e_base is not None:
            # guard 已经用 groups={0,2,3,5} 查过这个瞬时状态的 e_base，且只有在
            # 它和 e_bias 都有限时才会走到这个调用（见上面 docstring）——正常
            # 情况下这里恒真。仍然显式检查有限性，不静默信任传入值。
            e_base = float(reuse_e_base)
            if np.isfinite(e_base):
                self._consecutive_base_failures = 0
            else:
                failure_reason = "base_energy"
                self._consecutive_base_failures += 1
                print(
                    f"  🚨 Base 能量 (Group 0,2,3,5，来自 guard 复用值) 非有限"
                    f"（连续第 {self._consecutive_base_failures} 次），本帧标记为 NaN 并跳过"
                )
        else:
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
            if reuse_e_bias is not None:
                # 同上：guard 已经用 groups={1,4} 查过同一瞬时状态，且只有在它和
                # e_base 都有限时才会走到这里——正常情况下恒真，仍显式检查。
                e_bias = float(reuse_e_bias)
                if not np.isfinite(e_bias):
                    failure_reason = failure_reason or "bias_energy"
            else:
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
            softcore_energies, interaction_box_vectors = self._collect_interaction_energies(
                reuse_positions=reuse_positions,
                reuse_box_vectors=reuse_box_vectors,
            )
            lrc_energies = self._lj_tail_correction_kj_mol(interaction_box_vectors)
            target_energies = softcore_energies + lrc_energies

            residual_basis_energy = self.ibs_wrapper.get_residual_basis_energy(
                self.context
            )
            if self.ibs_wrapper.residual_enabled:
                residual_coefficients = (
                    self.ibs_wrapper.get_sampling_state_coefficients()
                )
                residual_offset = float(
                    self.ibs_wrapper.residual_energy_offset_kj_mol
                )
                sampling_state_energies = (
                    softcore_energies
                    + residual_coefficients
                    * (residual_basis_energy - residual_offset)
                )
            else:
                sampling_state_energies = softcore_energies.copy()

            # 🔑 [2026-08-26 bug fix] 之前这里用的是纯物理 softcore_energies，
            # 完全不含残差项。但驱动实际采样动力学的 Group-1 偏置力公式是
            # X_k = U_k + s_residual·A_k·B_φ − f_k（见 _state_expr(k)，
            # IBS_BIAS_PROTOCOL_VERSION=30 引入 s_residual 时改的那处）——f_k
            # 在线学习（update_weights()/_solve_tmbar_and_recenter()）读的
            # self.energy_buffer 如果只喂纯物理量，训练目标和实际部署目标完全
            # 对不上：f_k 学的是"如何拉平不含残差的分布"，却被拿去拉平"含残差
            # 的分布"，残差项越大的态天然学不对。已经在候选臂三个独立 repeat
            # 的生产数据里实测到系统性占据分布偏斜（跟这个不对齐完全吻合）。
            # sampling_state_energies（上面已经算出来，含残差、full-strength，
            # 不随 bias_scale/s_residual 爬坡缩放——这是有意的：bias_scale/
            # s_residual 只是部署期的爬坡机制，f_k 训练目标应该始终是满强度的
            # 最终混合分布，物理项这边一直就是这么处理的，没有把 bias_scale
            # 也乘进 u_mk）才是训练该用的目标；baseline 臂 residual_enabled=False
            # 时 sampling_state_energies 就是 softcore_energies 的原样拷贝，
            # 这一改动对 baseline 逐字节不变。
            # 相对偏移防溢出 (以 State 0 为参考)——只用于 bias_cv 训练路径的数值稳定性，
            # 是否与 target_energies 用同一个偏移量无所谓：softmax/log-sum-exp 对每帧
            # 内所有态统一平移不变，MBAR 那边存的是未平移的 target_energies 原始值。
            if self.n_states > 0 and np.isfinite(sampling_state_energies[0]):
                self.e_offset = sampling_state_energies[0]
            bias_cv_energies = sampling_state_energies - self.e_offset
            energies = bias_cv_energies

            frame_finite = (
                np.all(np.isfinite(bias_cv_energies))
                and np.all(np.isfinite(target_energies))
                and np.isfinite(e_base)
                and np.isfinite(e_bias)
                and np.all(np.isfinite(sampling_state_energies))
                and np.isfinite(residual_basis_energy)
            )
            if frame_finite:
                self.energy_buffer.append(bias_cv_energies)
                self.energy_history.append(target_energies.copy())
                self.base_energy_history.append(float(e_base))
                self.bias_history.append(float(e_bias))
                self.sampling_state_energy_history.append(
                    sampling_state_energies.copy()
                )
                self.residual_basis_history.append(float(residual_basis_energy))
                self._record_energy_query_result(True)
            else:
                self._record_energy_query_result(
                    False,
                    failure_reason or "nonfinite_energy_component",
                )
        except Exception as e:
            if isinstance(e, RuntimeError) and (
                "hard gate" in str(e)
                or "LJ 长程尾项" in str(e)
            ):
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
            # [P1-19] self.tmbar_history 是容量固定的滑动 minibatch 历史，列表
            # 位置不是物理 λ 窗口——split-half 诊断的"window 0"在这条在线学习
            # 调用路径下没有实际含义，只会在滑窗塞满前对同一条最早 minibatch
            # 反复报出逐字节相同的告警。真正有意义的 split-half 审计在整段
            # 生产/最终分析的另外两个调用点上仍然照常运行。见
            # P1-19_ONLINE_SLIDING_WINDOW_SPLITHALF_MISMATCH.md。
            skip_split_half_diagnostics=True,
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

    @staticmethod
    def _apply_pairwise_cap(
        f_old: np.ndarray,
        f_candidate: np.ndarray,
        cap_kt: float,
        kt: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Hard-cap the pairwise (max-min) spread of (f_candidate - f_old) to
        ``cap_kt*kt`` kJ/mol, preserving direction via uniform rescaling, then
        mean-center. Pure function -- no sampler state.

        [Candidate-first, Validate-or-Learn v1] Extracted from the hard-cap
        block previously inlined in ``update_weights()`` (bit-identical math)
        so both the trusted-absolute-TMBAR path there and the VALIDATE-
        failure damped-correction retry path in ``run_all_windows`` share one
        implementation instead of two copies that can silently drift apart.
        """
        old = np.asarray(f_old, dtype=np.float64).ravel()
        candidate = np.asarray(f_candidate, dtype=np.float64).ravel()
        delta = candidate - old
        delta -= float(np.mean(delta)) if delta.size else 0.0
        spread = float(np.max(delta) - np.min(delta)) if delta.size else 0.0
        cap_kj = float(cap_kt) * float(kt)
        capped = False
        if spread > cap_kj > 0.0:
            delta = delta * (cap_kj / spread)
            spread = float(np.max(delta) - np.min(delta))
            capped = True
        f_new = old + delta
        f_new -= float(np.mean(f_new))
        return f_new, {
            "hard_pairwise_cap_applied": capped,
            "hard_pairwise_cap_kJ_mol": cap_kj,
            "delta_f_kJ_mol": delta.astype(float).tolist(),
            "pairwise_delta_f_spread_kJ_mol": float(spread),
        }

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
        if tmbar_candidate_trusted:
            # [Candidate-first, Validate-or-Learn v1] 硬 cap 逻辑已提取为
            # 独立纯函数 _apply_pairwise_cap（跟 VALIDATE 失败重试路径共用
            # 同一实现），行为与之前内联版本逐字节一致。
            f_new, _cap_diag = self._apply_pairwise_cap(
                f_old, f_new, IBS_MAX_APPLIED_PAIRWISE_STEP_KT, self.kt
            )
            weight_update_diag["hard_pairwise_cap_applied"] = _cap_diag["hard_pairwise_cap_applied"]
            weight_update_diag["hard_pairwise_cap_kJ_mol"] = _cap_diag["hard_pairwise_cap_kJ_mol"]
            _applied_delta = np.asarray(_cap_diag["delta_f_kJ_mol"], dtype=np.float64)
            _applied_spread = float(_cap_diag["pairwise_delta_f_spread_kJ_mol"])
        else:
            # bounded 路径：不额外硬 cap，用其自带自适应上限（见 _bounded_log_
            # occupancy_update）。诊断记录未施加外部硬 cap。
            _applied_delta = (
                np.asarray(f_new, dtype=np.float64)
                - np.asarray(f_old, dtype=np.float64)
            )
            _applied_delta -= float(np.mean(_applied_delta))
            _applied_spread = (
                float(np.max(_applied_delta) - np.min(_applied_delta))
                if _applied_delta.size else 0.0
            )
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
        # [P1-19] `effective_damping` 只在走了可信 TMBAR 绝对更新
        # （_damped_tmbar_absolute_update）时才被写进 weight_update_diag；
        # occupancy/SGD fallback 路径没有"阻尼系数"这个概念。此前用
        # `.get(..., 0.0)` 兜底会在 fallback 路径把 alpha 打印成 0.000，
        # 看起来像"阻尼到零但 delta_f 仍非零"，实为字段不存在，不是阻尼系数
        # 真的是 0。fallback 时改打印 "n/a" 并显式带上 method 是否可信。
        _alpha = weight_update_diag.get("effective_damping")
        _alpha_str = f"{float(_alpha):.3f}" if _alpha is not None else "n/a(fallback)"
        print(
            f"    [IBS TMBAR 自洽权重更新 v9] dominant诊断=state{dominant_k} "
            f"p={float(mean_p_batch[dominant_k]):.6f}, "
            f"delta_f={float(weight_update_diag['delta_f_kJ_mol'][dominant_k]):+.3f} kJ/mol, "
            f"max|delta_f|={float(weight_update_diag['max_abs_delta_f_kJ_mol']):.3f} "
            f"(pairwise={float(weight_update_diag['pairwise_delta_f_spread_kJ_mol']):.3f} kJ/mol, "
            f"method={weight_update_diag['method']}, trusted={tmbar_candidate_trusted}, "
            f"alpha={_alpha_str}, "
            f"raw_residual={residual_severity:.3f}, "
            f"batch={M}, sliding_window_frames={total_tmbar_frames}, "
            # [P1-19] 这个字段只是"本次实际应用的 pairwise 步长 ≤ 自洽阈值"，
            # 与 method/trusted 是否走了可信 TMBAR 绝对更新正交——method=
            # bounded_occupancy...untrusted 时它仍可能是 True（fallback 自己的
            # 步长恰好也小），不代表 TMBAR 候选本身可信/被采用，改名避免
            # 读成"TMBAR 自洽"。
            f"applied_step_within_selfconsistency_limit={tmbar_self_consistent}, "
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
    def save_ibs_state(
        self,
        filepath: str,
        lambdas_coul=None,
        lambdas_vdw=None,
        stage_type: Optional[str] = None,
    ):
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
            "f_history_kj_mol": [
                np.asarray(values, dtype=float).tolist()
                for values in self.f_history
            ],
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
            "sampling_score_sha256": getattr(self, "sampling_score_sha256", None),
            "residual_sampling_protocol_version": (
                IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
                if getattr(self, "sampling_score_sha256", None) is not None
                else None
            ),
            "lambdas_coul": [float(x) for x in lambdas_coul] if lambdas_coul is not None else None,
            "lambdas_vdw": [float(x) for x in lambdas_vdw] if lambdas_vdw is not None else None,
            # [Candidate-first, Validate-or-Learn v1] 纯附加元数据，绝不参与
            # load_ibs_state 的任何 fail-closed/版本判定。seed_source 只是
            # "这份 f_k 是怎么来的"的事后可读标签；validation_attempts 只数
            # 真正跑完一次 local-MBAR gate 评估的次数（不含 burn-in/早退）；
            # last_failure_reason 是最近一次未收敛/失败的具体原因，供事后
            # 排障，不驱动任何 resume 分支。learning_updates 是 "t" 的只读
            # 别名，只在保存时写一次，绝不在 load 时独立读回——避免出现两个
            # 可能漂移的"学习次数"来源。
            "seed_source": getattr(self, "seed_source", None),
            "validation_attempts": int(getattr(self, "validation_attempts", 0) or 0),
            "last_failure_reason": getattr(self, "last_failure_reason", None),
            "learning_updates": len(self.f_history),
            # 🔑 [2026-08-27，见 EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md
            # 补验收] 只在 run_all_windows 真正进入生产采样那一刻被设置一次
            # （见该处注释），之后永不再被赋值——所以每次 save_ibs_state 落盘的
            # 都是同一份"生产开始时"的快照，不会被生产期间的任何操作污染。
            # None 表示这个窗口还没跑到生产采样阶段（纯校准/学习期间的存档）。
            "production_entry_f_k": (
                [float(x) for x in getattr(self, "production_entry_f_k")]
                if getattr(self, "production_entry_f_k", None) is not None else None
            ),
        }
        if stage_type == "vdw":
            state["vdw_nonbonded_protocol_version"] = VDW_NONBONDED_PROTOCOL_VERSION
        if getattr(self, "sampling_score_sha256", None) is not None:
            state["residual_sampling"] = {
                "feature": getattr(
                    self, "residual_feature_name", "Outer-Lambda Local Residual for IBS"
                ),
                "em_policy": getattr(self, "residual_em_policy", "no_residual_twin"),
                "plugin_identity": getattr(self, "residual_plugin_identity", None),
            }
        _atomic_write_json(filepath, state)

    def load_ibs_state(
        self,
        filepath: str,
        lambdas_coul=None,
        lambdas_vdw=None,
        stage_type: Optional[str] = None,
    ) -> bool:
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

            current_score_sha256 = getattr(self, "sampling_score_sha256", None)
            cached_score_sha256 = state.get("sampling_score_sha256")
            if current_score_sha256 is not None and cached_score_sha256 != current_score_sha256:
                raise ExistingEnsembleRequiresRescueAudit(
                    "IBS state score identity mismatch: cached "
                    f"sampling_score_sha256={cached_score_sha256!r}, current="
                    f"{current_score_sha256!r}. Refusing to inject f_k or overwrite the "
                    "existing ensemble; route it to rescue/provenance audit.",
                    diagnostics={
                        "state_file": os.path.abspath(filepath),
                        "cached_sampling_score_sha256": cached_score_sha256,
                        "current_sampling_score_sha256": current_score_sha256,
                        "requires_rescue_audit": True,
                    },
                )
            if (
                current_score_sha256 is not None
                and state.get("residual_sampling_protocol_version")
                != IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
            ):
                raise ExistingEnsembleRequiresRescueAudit(
                    "IBS state residual sampling protocol version is missing or "
                    "incompatible; refusing to inject f_k into the current ensemble.",
                    diagnostics={
                        "state_file": os.path.abspath(filepath),
                        "cached_residual_sampling_protocol_version": state.get(
                            "residual_sampling_protocol_version"
                        ),
                        "current_residual_sampling_protocol_version": (
                            IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
                        ),
                        "requires_rescue_audit": True,
                    },
                )
            if current_score_sha256 is not None:
                cached_residual = state.get("residual_sampling")
                if not isinstance(cached_residual, dict) or cached_residual.get(
                    "em_policy"
                ) != "no_residual_twin":
                    raise ExistingEnsembleRequiresRescueAudit(
                        "IBS state residual EM/plugin identity is missing or incompatible; "
                        "refusing to inject f_k into the current ensemble.",
                        diagnostics={
                            "state_file": os.path.abspath(filepath),
                            "cached_residual_sampling": cached_residual,
                            "requires_rescue_audit": True,
                        },
                    )

            # The persisted tmbar_history is the actual u_kn history used to
            # continue bias learning.  A Stage 2 history sampled with the old
            # softcore Hamiltonian must never seed the current Hamiltonian;
            # Stage 1 charging states intentionally do not carry this gate.
            if (
                stage_type == "vdw"
                and state.get("vdw_nonbonded_protocol_version")
                != VDW_NONBONDED_PROTOCOL_VERSION
            ):
                print(
                    "  ⚠️ Stage 2 IBS 状态的 "
                    f"vdw_nonbonded_protocol_version={state.get('vdw_nonbonded_protocol_version')!r} "
                    f"与当前 {VDW_NONBONDED_PROTOCOL_VERSION} 不匹配，"
                    "拒绝恢复旧 Hamiltonian 的 f_k/u_kn 历史"
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
            elif cached_status == "failed":
                # [Candidate-first, Validate-or-Learn v1] 新协议下的终态失败
                # 值——必须原样保留，不能落进下面的 else 分支被静默改写成
                # "unconverged"，否则 run_all_windows 里那一处专门识别
                # "failed" 为终态的检查（10586 行附近）永远看不到这个值，
                # 一份已经判定终态失败的窗口会被当成普通未收敛热启动重新学习。
                self.bias_status = "failed"
                self.frozen_f_k_pending = None
                self.frozen_validation_cumulative_steps = 0
            else:
                self.bias_status = "converged" if self.bias_converged else "unconverged"
                self.frozen_f_k_pending = None
                self.frozen_validation_cumulative_steps = 0

            # [Candidate-first, Validate-or-Learn v1] 纯附加元数据恢复，缺失
            # 时安全回退默认值，绝不新增 fail-closed 规则。一次成功的
            # load_ibs_state 按定义就是一次 resume——seed_source 在这里写
            # "resume"，而不是在本次刻意不重写的 10586-10689 resume 路由分支
            # 里，两者等价，但避免了对那段控制流的任何改动。
            self.seed_source = "resume"
            self.validation_attempts = int(state.get("validation_attempts", 0) or 0)
            self.last_failure_reason = state.get("last_failure_reason")

            # 🔑 [2026-08-27] 恢复生产入口标记。缺失（旧文件）或形状/有限性
            # 校验不过时保持 None（"历史入口未知"），绝不能拿这次 resume
            # 之后才会算出的值冒充最初进入生产时的值——那正是这个字段存在
            # 的意义。
            cached_production_entry_f_k = state.get("production_entry_f_k")
            if (
                cached_production_entry_f_k is not None
                and len(cached_production_entry_f_k) == self.n_states
                and np.all(np.isfinite(np.asarray(cached_production_entry_f_k, dtype=float)))
            ):
                self.production_entry_f_k = [float(x) for x in cached_production_entry_f_k]
            else:
                self.production_entry_f_k = None

            # 注入 Context
            for k in range(self.n_states):
                self.context.setParameter(f"{self.prefix}_f_{k}", float(f_k[k]))

            cached_f_history = state.get("f_history_kj_mol")
            valid_f_history = []
            if isinstance(cached_f_history, list) and len(cached_f_history) == int(t):
                for values in cached_f_history:
                    array = np.asarray(values, dtype=np.float64)
                    if array.shape != (self.n_states,) or not np.all(np.isfinite(array)):
                        valid_f_history = []
                        break
                    valid_f_history.append(array.copy())
            # Backward compatible: old checkpoints preserve the learning-rate
            # update count; new checkpoints retain the actual trajectory for
            # restart continuity and EXP-030 audit.
            self.f_history = (
                valid_f_history if len(valid_f_history) == int(t)
                else [np.array(f_k, dtype=np.float64)] * int(t)
            )
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
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Content-based fingerprint for one window's fixed-H probe trajectory
    bank. Any change to the physical Hamiltonian (system/CV XML content,
    cache/bias protocol version, lambda_shield, temperature, sample_interval,
    platform) must invalidate every state's already-sampled frames in this
    window+probe_type directory -- see the resume trust rules in
    ``probe_adjacent_path_overlap_bank``/``probe_adjacent_bias_calibration_bank``.
    XML blobs are hashed (not stored verbatim) to keep manifest.json small.
    """
    manifest = {
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
    if coion_identity is not None:
        manifest["coion_identity"] = coion_identity
    if stage_type == "vdw":
        manifest["vdw_nonbonded_protocol_version"] = VDW_NONBONDED_PROTOCOL_VERSION
    return manifest


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
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """主窗口 checkpoint 的内容指纹——任何一项不匹配都必须整体拒绝这份
    checkpoint（λ 网格被自动加密/重新划分窗口、协议版本变化、平台不同等），
    直接沿用探针轨迹库 `_build_fixed_h_probe_bank_manifest` 的字段/哈希写法。
    """
    manifest = {
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
    if coion_identity is not None:
        manifest["coion_identity"] = coion_identity
    if stage_type == "vdw":
        manifest["vdw_nonbonded_protocol_version"] = VDW_NONBONDED_PROTOCOL_VERSION
    return manifest


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
    coion_identity: Optional[Dict[str, Any]] = None,
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
    manifest = {
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
    if coion_identity is not None:
        manifest["coion_identity"] = coion_identity
    if stage_type == "vdw":
        manifest["vdw_nonbonded_protocol_version"] = VDW_NONBONDED_PROTOCOL_VERSION
    return manifest


def _resolve_production_entry_marker(
    current_marker: Optional[List[float]],
    current_f_k: np.ndarray,
    resumed_production_checkpoint: bool,
) -> Tuple[str, Optional[List[float]]]:
    """决定 production_entry_f_k 在"即将真正开始生产采样"这一刻该怎么处理
    （checkpoint 恢复/续采判断、安全检查、f_k 与冻结 manifest 一致性断言全部
    已经通过之后调用）。纯函数，不碰 sampler/磁盘，方便独立测试；调用方
    （run_all_windows）据返回值决定是否真的写 sampler.production_entry_f_k
    和调用 save_ibs_state()。

    返回 (action, marker)：
      "keep"   —— 原样保留 sampler 当前的 production_entry_f_k，不写、不存。
      "set"    —— 调用方应把 production_entry_f_k 设为 marker 并 save_ibs_state()。
      "refuse" —— 调用方应拒绝继续生产（真实数据矛盾），marker 为 None。

    🔑 [续跑边界，2026-08-27] current_marker 为 None 不能一律当成"这是第一次
    进入生产"直接写 current_f_k：如果 resumed_production_checkpoint 为
    True——已经存在一份真正驱动过采样、且 manifest 已确认属于这份冻结 f_k
    的旧生产历史（energies/bias/base.npy + 匹配的 checkpoint）——那份历史
    最初进入生产时的 f_k 本来就是真的不知道（旧版本 ibs_engine.py 从没记过
    这个字段）；这时候写一个"现在"的值会把一段入口未知的旧历史悄悄升级成
    "生产入口标记验证通过"，正是这个字段本该防止的事——必须继续保持 None，
    让下游 reconcile 走 degraded 路径。只有 resumed_production_checkpoint 为
    False（这次生产确实是从 0 步重新开始，没有沿用任何旧生产样本）时，
    marker=None 才代表"这次的生产历史确实还没有入口标记"，可以第一次写入。

    同理，marker 非 None 但与 current_f_k 对不上：如果 resumed_production_
    checkpoint 为 True，manifest 已经确认这份 checkpoint 的生产历史就是
    current_f_k 驱动出来的，那这份不匹配的旧 marker 就是真正的数据不一致
    （标记与它本该对应的历史对不上），必须拒绝，不能用当前值静默覆盖掉一份
    可能记录着真实矛盾的标记；resumed_production_checkpoint 为 False 时，
    这份不匹配的 marker 只是来自另一份已作废的校准，可以安全覆盖。
    """
    current_arr = np.asarray(current_f_k, dtype=np.float64)
    if current_marker is None:
        if resumed_production_checkpoint:
            return ("keep", None)
        return ("set", [float(x) for x in current_arr])
    matches = (
        len(current_marker) == len(current_arr)
        and np.allclose(
            np.asarray(current_marker, dtype=np.float64), current_arr, rtol=0.0, atol=1.0e-12
        )
    )
    if matches:
        return ("keep", None)
    if resumed_production_checkpoint:
        return ("refuse", None)
    return ("set", [float(x) for x in current_arr])


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
    for manifest in (normalized_expected, normalized_loaded):
        if "stage_protocol_key" in manifest:
            manifest["stage_protocol_key"] = _stage_window_sampling_identity(manifest["stage_protocol_key"])
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
    frame_observer=None,
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

    ``frame_observer``, if given, is called as ``frame_observer(positions,
    box_vectors)`` once per stored frame and its return value is appended to
    ``record["observer_values"]`` -- used by the independent-endpoint
    production sampler (INDEPENDENT_ENDPOINT_PROTOCOL_VERSION) to record the
    per-frame cavity water count alongside the energies. It is deliberately a
    callback rather than a second sampling pass: the structural observable
    must come from *exactly* the frames that feed MBAR, not from a separately
    strided re-read of the trajectory.
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
    new_observations = []
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
        if frame_observer is not None:
            new_observations.append(frame_observer(frame_positions, frame_box))
    new_energies = np.asarray(new_energies, dtype=np.float64)
    new_volumes = np.asarray(new_volumes, dtype=np.float64)
    if frame_observer is not None:
        record["observer_values"] = (
            list(record.get("observer_values") or []) + new_observations
        )
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
    coion_identity: Optional[Dict[str, Any]] = None,
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
        coion_identity=coion_identity,
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
    coion_identity: Optional[Dict[str, Any]] = None,
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
        coion_identity=coion_identity,
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
# ============================================================================
# [INDEPENDENT_ENDPOINT_PROTOCOL_VERSION=1] 端点态独立固定-λ 生产采样
#
# 根因见 STAGE2_ROOT_CAUSE_2026-08-28.md。IBS 的 stage2 每个窗口只跑**一条**
# 轨迹，窗口内所有 λ 态都从这一条轨迹重加权出来。那条轨迹里配体（哪怕软核）
# 几乎总占着体积，"水填满配体空腔"的构型概率 ≈ 0——**重加权造不出没采到的
# 构型，再多帧也造不出来**。而 ΔG 的主要来源恰恰是这个溶剂重组：4W53 溶剂腿
# window 2（λ_vdw 0.398→0.000）真值 -14.85 kJ/mol 里 TΔS ≈ +21 kJ/mol 全部
# 来自水塌进空腔，能量差几乎为零。生产给 +4.64，错 +19.49。
#
# 决定性的一点：window 2 的相邻 <ΔU> 只有 0.4~0.6 kT，**任何基于能量的重叠
# 判据都会说"完美"**。所以这不是"窗口太宽/重叠不足/统计噪声"，加窗、插 λ、
# 延长采样、乃至任何形式的重加权诊断都治不了它。
#
# 本模块实现的修法：对 λ_vdw 接近 0 的关键态**各自建立真正独立的固定-λ 生产
# 轨迹**——每个态有自己的 System、自己的 Context、自己的平衡和采样，绝不从
# 某条窗口混合轨迹重加权出来。这样 λ=0 附近的态从一开始就在"配体不与环境
# 相互作用"的哈密顿量下演化，水本来就能进空腔，塌缩系综是被**采到**的而不是
# 被重加权"造"出来的。
#
# 五条必须同时成立的口径要求（少一条这套东西就没有意义）：
#
#   1. **正式采样不得保留 Group-4 WCA 防护壳。** 传进来的 system XML 必须是
#      `_serialize_ibs_common_system` 的产物（group 1 与 group 4 都已移除），
#      `_build_fixed_state_simulation(require_group4=False)` 会硬断言系统里
#      不存在 group 4。WCA 只准用于建初始构型/预平衡，不准进生产动力学——
#      它把水挡在空腔外，正是本模块要采的那个构型。
#   2. **其余一切与生产完全同一套。** system XML 直接来自已组装好的生产窗口
#      系统，所以 ACE softcore 形式、PME、cutoff/switching 全部逐字节相同；
#      每个态的 CV 力用的是生产自己的 `ibs_wrapper._int_cv_force_xmls[k]`；
#      LRC 用的是 `ibs_wrapper.lj_tail_lrc_coeff_kj_mol`，逐帧按
#      `lrc_coeff[k] / V(t)` 加上——与 `IBSSampler.collect_energies` 里
#      `target_energies = softcore_energies + lrc_energies` 是同一个式子、
#      同一组系数。绝不在这里另算一份。
#   3. **湿空腔与干空腔分别起 walker。** 同一个态至少两个独立起点：干空腔
#      （配体耦合端平衡出来的构型，空腔里没有水）和湿空腔（在完全解耦端长
#      平衡、让水灌进空腔之后的构型）。两组各自跑多个 walker。若两组停在
#      各自的模态里出不来，MBAR 的统计误差可以很小而结果是错的——所以
#      两组的 ΔF 之差必须过 2σ 门（见 `endpoint_wet_dry_hysteresis_gate`）。
#   4. **逐态、逐 segment 时间去相关**（`_decorrelate_per_segment`），MBAR 吃
#      的是有效独立样本数，不是原始帧数。
#   5. **结构性证据必须落盘**：每帧的空腔水分子数、湿/干占比、水进出空腔的
#      转换次数、每个 walker 各自的占据分布。统计量对"该采的构型一次都没采
#      到"这个失效模式是失明的（4W53 的 split-half 前后半程一致地错），只有
#      结构量能直接回答"水到底进没进去过"。
#
# 与既有 fixed-H 探针的关系：`probe_adjacent_path_overlap_bank` 早就在用
# `_build_fixed_state_simulation` 做按态独立采样，但它是**诊断探针**（回答
# "λ 网格连不连得上"），采样长度和产物都不用于生产自由能。本模块复用同一套
# 底层构件，但产出的是**生产数据**：能量矩阵直接喂 MBAR 当 stage2 端点段的
# 主值。两者的 checkpoint 目录、指纹、采样长度都彼此独立，不共享缓存。
# ============================================================================
INDEPENDENT_ENDPOINT_PROTOCOL_VERSION = 1

# 空腔水判据：水的氧原子到**任一配体重原子**的最小像距离小于这个半径，就算它
# 侵入了配体本该占据的体积。0.24 nm 明显小于 O–O 第一峰(0.28 nm)，配体真正
# 耦合时任何水都不可能待在这个距离上——所以计数 > 0 是"空腔被水占据"的直接
# 结构证据，而不是一个需要调参的软指标。
CAVITY_PROBE_RADIUS_NM = 0.24
# 一帧里有多少个这样的水就算"湿"。默认 1：一个水进到配体体积里就已经是那个
# 被 IBS 单轨迹漏掉的构型。
CAVITY_WET_MIN_WATERS = 1
# 湿/干双起点 ΔF 之差的验收门，单位是合并 σ 的倍数。
# ⚠️ 直接**别名** abfe_core.STAGE2_HYSTERESIS_MAX_SIGMA，不另立数值：
# 该常量此前只写进 provenance、从未被任何代码执行过（死常量）。本模块是它的第一个
# 真实执行点。若这里写成独立的 2.0，就会出现"provenance 记录的阈值"与"实际生效的
# 阈值"两份定义，改一个忘另一个 —— 那正是它最初变成死常量的同一类问题。
ENDPOINT_WET_DRY_MAX_SIGMA = STAGE2_HYSTERESIS_MAX_SIGMA

INDEPENDENT_ENDPOINT_INTEGRATOR_SEED_BASE = 611_501
INDEPENDENT_ENDPOINT_VELOCITY_SEED_BASE = 733_907
# 质数步长，避免 (态, 模态, walker) 三重循环里不同组合撞到同一个种子。
_INDEPENDENT_ENDPOINT_SEED_STRIDE = 97
_INDEPENDENT_ENDPOINT_MODE_STRIDE = 10_007
_INDEPENDENT_ENDPOINT_WALKER_STRIDE = 1_009

INDEPENDENT_ENDPOINT_INIT_MODES = ("dry", "wet")


def water_oxygen_indices(topology) -> np.ndarray:
    """拓扑里所有水氧原子的索引。

    只认真正的水残基（HOH/WAT/TIP3/SOL/H2O）里的氧，不靠"元素是 O"来猜——
    蛋白骨架和配体上也全是氧。
    """
    water_resnames = {"HOH", "WAT", "TIP3", "TIP3P", "TIP4", "SOL", "H2O", "T3P"}
    indices = []
    for residue in topology.residues():
        if str(residue.name).upper() not in water_resnames:
            continue
        for atom in residue.atoms():
            element = getattr(atom, "element", None)
            symbol = getattr(element, "symbol", None) if element is not None else None
            if symbol == "O":
                indices.append(int(atom.index))
    return np.asarray(indices, dtype=int)


def ligand_heavy_atom_indices(topology, ligand_indices) -> np.ndarray:
    """配体重原子（非氢）索引。空腔判据只用重原子——氢的范德华半径太小，
    把它算进去会让"侵入"判据依赖于配体的具体质子化细节。"""
    atoms = list(topology.atoms())
    heavy = []
    for idx in np.asarray(ligand_indices, dtype=int).ravel():
        atom = atoms[int(idx)]
        element = getattr(atom, "element", None)
        symbol = getattr(element, "symbol", None) if element is not None else None
        if symbol != "H":
            heavy.append(int(idx))
    if not heavy:
        raise ValueError("配体没有任何重原子，无法定义空腔水判据")
    return np.asarray(heavy, dtype=int)


def _minimum_image_displacements(delta_nm: np.ndarray, box_nm: np.ndarray) -> np.ndarray:
    """OpenMM 归约形式（reduced form）三斜盒子的最小像位移。

    必须按 c→b→a 的顺序依次减，这是归约形式成立的前提；对正交盒子退化成
    逐轴 round，结果与常见的 `dr -= L*round(dr/L)` 完全一致。
    """
    return minimum_image_displacement_nm(delta_nm, box_nm)


def count_cavity_waters(
    positions_nm: np.ndarray,
    box_nm: np.ndarray,
    ligand_heavy_idx: np.ndarray,
    water_oxygen_idx: np.ndarray,
    probe_radius_nm: float = CAVITY_PROBE_RADIUS_NM,
) -> int:
    """一帧里"侵入配体体积"的水分子数：氧到任一配体重原子的最小像距离 < 半径。

    这是 stage2 唯一能直接回答"水到底进没进空腔"的量。所有基于能量的统计门
    对这件事都是失明的（4W53 实测：g≈1.4、split-half 一致，稳定地收敛到错值）。
    """
    pos = np.asarray(positions_nm, dtype=np.float64)
    lig = pos[np.asarray(ligand_heavy_idx, dtype=int)]
    wat = pos[np.asarray(water_oxygen_idx, dtype=int)]
    if wat.size == 0:
        return 0
    delta = wat[:, None, :] - lig[None, :, :]
    delta = _minimum_image_displacements(delta, box_nm)
    dist2 = np.sum(delta * delta, axis=-1)
    r2 = float(probe_radius_nm) ** 2
    return int(np.count_nonzero(np.min(dist2, axis=1) < r2))


def cavity_occupancy_diagnostics(
    counts,
    wet_min_waters: int = CAVITY_WET_MIN_WATERS,
) -> Dict[str, Any]:
    """把一条轨迹的逐帧空腔水数压成验收用的结构性指标。

    `n_wet_dry_transitions` 是最关键的一项：它 > 0 才说明这条轨迹**真的**在
    湿态和干态之间来回过。如果湿起点的 walker 全程 wet_fraction=1、干起点的
    全程 =0、两边转换次数都是 0，那就是两个互不连通的模态各自被采了一遍——
    此时 MBAR 报出来的 σ 再小也不可信，必须由 `endpoint_wet_dry_hysteresis_gate`
    判失败。
    """
    arr = np.asarray(counts, dtype=np.float64).ravel()
    if arr.size == 0:
        return {
            "n_frames": 0, "mean_cavity_waters": None, "max_cavity_waters": None,
            "wet_fraction": None, "dry_fraction": None,
            "n_wet_dry_transitions": None, "wet_min_waters": int(wet_min_waters),
        }
    wet = arr >= float(wet_min_waters)
    transitions = int(np.count_nonzero(wet[1:] != wet[:-1])) if wet.size > 1 else 0
    return {
        "n_frames": int(arr.size),
        "mean_cavity_waters": float(np.mean(arr)),
        "max_cavity_waters": float(np.max(arr)),
        "wet_fraction": float(np.mean(wet)),
        "dry_fraction": float(1.0 - np.mean(wet)),
        "n_wet_dry_transitions": transitions,
        "wet_min_waters": int(wet_min_waters),
    }


def _independent_endpoint_bank_dir(checkpoint_dir: str, stage_type: str) -> str:
    return os.path.join(checkpoint_dir, "independent_endpoint", str(stage_type))


def _independent_endpoint_record_path(
    bank_dir: str, global_k: int, mode: str, walker: int
) -> str:
    return os.path.join(bank_dir, f"state_{int(global_k)}_{mode}_w{int(walker)}.npz")


def build_independent_endpoint_manifest(
    stage_type: str,
    state_indices: Sequence[int],
    common_system_xml: str,
    cv_xmls: Sequence[str],
    temperature_K: float,
    sample_interval: int,
    sample_steps: int,
    burn_in_steps: int,
    n_walkers_per_mode: int,
    platform_name: str,
    cavity_probe_radius_nm: float,
    cavity_wet_min_waters: int,
    init_modes: Optional[Sequence[str]] = None,
    minimize_iterations: int = 2_000,
    integrator_seed_base: Optional[int] = None,
    velocity_seed_base: Optional[int] = None,
    seed_identity: Optional[Dict[str, Any]] = None,
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """独立端点采样库的内容指纹。

    任何一项不匹配 → 整个目录判定不可信、全部重采（与 fixed-H 探针库同一套
    "全对才信、错一项就整体拒绝"哲学）。`common_system_xml` 的哈希覆盖了 ACE
    softcore 形式、PME/cutoff/switching 以及"有没有 Group-4 WCA"——所以把
    WCA 重新塞回生产动力学会立刻让所有已采数据失效，不可能被静默复用。
    """
    return {
        "kind": "independent_endpoint_states",
        "protocol_version": int(INDEPENDENT_ENDPOINT_PROTOCOL_VERSION),
        "stage_type": str(stage_type),
        "state_indices": [int(x) for x in state_indices],
        "common_system_sha256": hashlib.sha256(
            common_system_xml.encode("utf-8")
        ).hexdigest(),
        "cv_xml_sha256": [
            hashlib.sha256(x.encode("utf-8")).hexdigest() for x in cv_xmls
        ],
        "temperature_K": float(temperature_K),
        "sample_interval": int(sample_interval),
        "sample_steps": int(sample_steps),
        "burn_in_steps": int(burn_in_steps),
        "n_walkers_per_mode": int(n_walkers_per_mode),
        # 实际使用的起点模态。干-only 与 干+湿 是两份不同的数据集，缓存不得混用。
        "init_modes": list(init_modes if init_modes is not None
                           else INDEPENDENT_ENDPOINT_INIT_MODES),
        "platform_name": str(platform_name),
        "cavity_probe_radius_nm": float(cavity_probe_radius_nm),
        "cavity_wet_min_waters": int(cavity_wet_min_waters),
        # 起始构型的制备方式变了，已采数据就不该被复用。
        "minimize_iterations": int(minimize_iterations),
        "wet_seeding_scheme": "wet_cavity_ladder_v1",
        # 🔑 [P1] 种子身份进指纹：换了 repeat_seed 就必须重采，不能复用旧 bank。
        "integrator_seed_base": (
            int(integrator_seed_base) if integrator_seed_base is not None else None
        ),
        "velocity_seed_base": (
            int(velocity_seed_base) if velocity_seed_base is not None else None
        ),
        "seed_identity": seed_identity,
        "coion_identity": coion_identity,
    }


def _independent_endpoint_manifest_matches(bank_dir: str, expected: Dict[str, Any]) -> bool:
    path = os.path.join(bank_dir, "manifest.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle) == expected
    except Exception:
        return False


def prepare_wet_cavity_seed(
    topology,
    common_system_xml: str,
    cv_xml_decoupled: str,
    positions,
    box_vectors,
    temperature,
    platform_name: str,
    ligand_heavy_idx: np.ndarray,
    water_oxygen_idx: np.ndarray,
    equilibration_steps: int = 500_000,
    check_interval: int = 5_000,
    cavity_probe_radius_nm: float = CAVITY_PROBE_RADIUS_NM,
    cavity_wet_min_waters: int = CAVITY_WET_MIN_WATERS,
    integrator_seed: int = INDEPENDENT_ENDPOINT_INTEGRATOR_SEED_BASE - 1,
    velocity_seed: int = INDEPENDENT_ENDPOINT_VELOCITY_SEED_BASE - 1,
) -> Dict[str, Any]:
    """在**完全解耦**态上长平衡出一个"湿空腔"起始构型。

    这是湿起点 walker 的来源。在 λ_vdw=0 上配体与环境没有相互作用，水本来就
    可以自由进入配体占据的体积——所以这一步是在一个水**能**进空腔的哈密顿量
    上等它进去，而不是靠重加权去"造"那个构型。返回的构型随后被喂给**所有**
    端点态（包括还有相互作用的那些）当作湿起点。

    动力学同样不含 Group-4 WCA（`require_group4=False` 会硬断言）——防护壳的
    全部作用就是把水挡在外面，用它来平衡湿空腔是自相矛盾的。

    返回值里带上逐次检查的空腔水数序列：若到最后 `reached_wet` 仍是 False，
    调用方必须当作**失败**处理，而不是拿一个干构型冒充湿起点——那会让湿/干
    双起点退化成两组干起点，2σ 门看起来通过、实际什么都没验证。
    """
    temperature_q = (
        temperature if hasattr(temperature, "value_in_unit")
        else float(temperature) * unit.kelvin
    )
    simulation = _build_fixed_state_simulation(
        topology=topology,
        system_xml=common_system_xml,
        cv_xml=cv_xml_decoupled,
        require_group4=False,
        platform_name=platform_name,
        temperature_q=temperature_q,
        positions=positions,
        box_vectors=box_vectors,
        integrator_seed=int(integrator_seed),
        velocity_seed=int(velocity_seed),
    )
    try:
        counts = []
        steps_done = 0
        best_state = None
        while steps_done < int(equilibration_steps):
            chunk = min(int(check_interval), int(equilibration_steps) - steps_done)
            simulation.step(int(chunk))
            steps_done += chunk
            state = simulation.context.getState(getPositions=True, getVelocities=True)
            pos_nm = np.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=np.float64,
            )
            box_nm = _box_vectors_to_nm_array(state.getPeriodicBoxVectors())
            n_wat = count_cavity_waters(
                pos_nm, box_nm, ligand_heavy_idx, water_oxygen_idx,
                probe_radius_nm=cavity_probe_radius_nm,
            )
            counts.append(int(n_wat))
            # 保留**最湿**的那一帧，而不是最后一帧：最后一帧可能恰好处在水
            # 短暂退出的瞬间，用它当湿起点会白白浪费这次长平衡。
            if best_state is None or n_wat >= max(counts[:-1] or [0]):
                if n_wat >= int(cavity_wet_min_waters) or best_state is None:
                    best_state = state
        diagnostics = cavity_occupancy_diagnostics(counts, cavity_wet_min_waters)
        reached = bool(counts) and max(counts) >= int(cavity_wet_min_waters)
        final = best_state if best_state is not None else simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        return {
            "reached_wet": reached,
            "positions": final.getPositions(),
            "box_vectors": final.getPeriodicBoxVectors(),
            "cavity_water_counts": [int(x) for x in counts],
            "cavity_diagnostics": diagnostics,
            "equilibration_steps": int(steps_done),
            "dynamics_hamiltonian": "U_common_plus_single_cv_int_no_wca",
        }
    finally:
        del simulation
        gc.collect()


def _independent_endpoint_seed(global_k: int, mode: str, walker: int, base: int) -> int:
    mode_index = INDEPENDENT_ENDPOINT_INIT_MODES.index(str(mode))
    return int(
        base
        + _INDEPENDENT_ENDPOINT_SEED_STRIDE * int(global_k)
        + _INDEPENDENT_ENDPOINT_MODE_STRIDE * int(mode_index)
        + _INDEPENDENT_ENDPOINT_WALKER_STRIDE * int(walker)
    )


def run_independent_endpoint_states(
    topology,
    common_system_xml: str,
    ibs_wrapper,
    state_indices: Sequence[int],
    dry_seed: Dict[str, Any],
    wet_seed: Dict[str, Any],
    temperature,
    platform_name: str,
    checkpoint_dir: str,
    stage_type: str,
    ligand_indices,
    global_state_indices: Optional[Sequence[int]] = None,
    lambdas_vdw_for_states: Optional[Sequence[float]] = None,
    # 🔑 [P1] 种子身份必须由调用方（pipeline 的 _seed_for / seed_ledger）派生并
    # 传入。写死常量基数意味着：不同 repeat_seed 的两次运行会产生**逐字节相同**的
    # 端点随机流（等于把独立重复变成同一条轨迹跑两遍），而且改了 repeat_seed 之后
    # 旧的采样库仍然指纹匹配、被直接复用。两者都必须堵死。
    integrator_seed_base: Optional[int] = None,
    velocity_seed_base: Optional[int] = None,
    seed_identity: Optional[Dict[str, Any]] = None,
    burn_in_steps: int = 100_000,
    sample_steps: int = 500_000,
    sample_interval: int = 1_000,
    n_walkers_per_mode: int = 2,
    cavity_probe_radius_nm: float = CAVITY_PROBE_RADIUS_NM,
    cavity_wet_min_waters: int = CAVITY_WET_MIN_WATERS,
    minimize_iterations: int = 2_000,
    coion_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """为 `state_indices` 里的每一个 λ 态建立**真正独立**的固定-λ 生产轨迹。

    每个 (态, 起点模态, walker) 组合都有自己的 System、自己的 Context、自己的
    平衡和自己的采样——没有任何一条轨迹的构型是从别的态重加权出来的。这是
    与 IBS "一条窗口轨迹重加权出全窗"的根本区别，也是本模块存在的全部理由。

    `common_system_xml` 必须已经移除 Group 1（IBS 混合偏置）与 Group 4
    （λ-WCA 防护壳），即 `_serialize_ibs_common_system` 的产物；
    `_build_fixed_state_simulation(require_group4=False)` 会在建每一个 Context
    时硬断言系统里不存在 Group 4，所以"生产里还藏着 WCA"这件事不可能悄悄
    发生——它会直接 RuntimeError。

    同一时刻最多只有一个动力学 Context 存活（外加一个共享的轻量能量探针
    Context）：每个 walker 建好、跑完、落盘后立即释放。

    **湿空腔阶梯（wet cavity ladder）**：湿构型是在**完全解耦**态上平衡出来的，
    空腔里坐着好几个水。把它直接丢给一个强耦合的 λ 态，水正好压在完全显形的
    配体上——2026-08-29 实测在 λ_vdw=1.0 上 `Particle coordinate is NaN`，
    minimize 2000 步也救不回来（水氧到碳只有零点几埃，LJ 能量太大，L-BFGS 出不去）。
    加大最小化步数是治标；真正的问题是**在 λ_vdw≈1 上"湿空腔"根本不是一个亚稳态**，
    硬塞一个湿构型不是物理检验，只是碰撞。

    所以湿 walker 按 λ_vdw **从小到大**（最解耦 → 最耦合）依次跑，每个态用**上一个
    更解耦的态**的末构型作起点（只取 walker 0 的末构型，保证阶梯与 walker 数无关）。
    相邻两态的哈密顿量差别很小，湿空腔要么被平滑地带上去，要么在某个 λ 上自然干掉——
    后者正是**物理答案**，会如实反映在 `per_walker_cavity` 的 wet_fraction 上。

    干 walker **不**走阶梯：干构型对任何 λ 都相容（配体占着体积本来就是耦合端的常态），
    每个干 walker 都从同一个独立的干种子起。所以湿/干双起点门比的仍然是"两条来源
    完全不同的路径"——阶梯只影响湿侧的**起始构型**，每个 walker 的 Context、速度种子、
    burn-in 和采样都还是各自独立的。

    ⚠️ **每个 walker 在 burn-in 之前还会先在它自己的态上最小化**（
    `minimize_iterations`）。湿起点是在**完全解耦**态上平衡出来的，空腔里坐着
    好几个水；把那个构型直接丢给一个强耦合的 λ 态（λ_vdw→1），水正好压在完全
    显形的配体上，第一步积分就 `Particle coordinate is NaN`——2026-08-29 实测在
    λ_vdw=1.0 上确实炸了。这与 `_probe_vdw_window_fixed_overlap` 里记的那段
    live crash 是同一类错误（在一个势能面上弛豫、到另一个势能面上跑），只是方向
    相反。阶梯负责把跨度变小，最小化负责挤掉残余的不相容，两者缺一不可。

    返回的 `records` 里每条都带逐帧空腔水数；`per_walker_cavity` 汇总每个
    walker 自己的湿/干占比与转换次数——这是判断"湿/干两组是不是各自卡在自己
    的模态里"的第一手证据，`endpoint_wet_dry_hysteresis_gate` 要用它。
    """
    state_indices = [int(x) for x in state_indices]
    # `state_indices` 索引的是 `ibs_wrapper._int_cv_force_xmls`（即"这个窗口
    # 系统里的第几个 λ 态"）。当调用方只为端点块建了一个窗口系统时，这些是
    # **局部**索引，与整条 λ 路径上的全局索引不同——`global_state_indices` 用来
    # 记录后者，拼接时要靠它对齐接点。两者混用会让拼接在错误的 λ 上接合，而
    # 接错的自由能看起来完全正常。
    global_state_indices = (
        [int(x) for x in global_state_indices]
        if global_state_indices is not None else list(state_indices)
    )
    if len(global_state_indices) != len(state_indices):
        raise ValueError("global_state_indices 与 state_indices 长度必须一致")
    if len(state_indices) < 2:
        raise ValueError("独立端点采样至少需要两个 λ 态")
    if sorted(set(state_indices)) != sorted(state_indices):
        raise ValueError("state_indices 不能重复")
    if int(n_walkers_per_mode) < 1:
        raise ValueError("每个起点模态至少需要一个 walker")
    if sample_interval <= 0 or sample_steps < sample_interval:
        raise ValueError("独立端点采样长度不足一个 sample_interval")

    cv_xmls = list(getattr(ibs_wrapper, "_int_cv_force_xmls", []))
    if max(state_indices) >= len(cv_xmls):
        raise RuntimeError("IBS wrapper 缺少独立端点采样所需的 CV XML")

    temperature_q = (
        temperature if hasattr(temperature, "value_in_unit")
        else float(temperature) * unit.kelvin
    )
    ligand_heavy_idx = ligand_heavy_atom_indices(topology, ligand_indices)
    water_oxygen_idx = water_oxygen_indices(topology)
    if water_oxygen_idx.size == 0:
        raise RuntimeError(
            "拓扑里找不到任何水分子；空腔水结构诊断是本模块的验收前提，拒绝"
            "在没有结构证据的情况下产出端点自由能"
        )

    if not dry_seed or dry_seed.get("positions") is None:
        raise ValueError("缺少 dry 起点构型")
    # ------------------------------------------------------------------
    # 🔑 [2026-08-31] 湿起点是**条件性诊断**，不是前置条件。
    #
    # 这里曾经在 reached_wet=False 时直接 raise。那是错的：湿/干双起点是为溶剂腿
    # 的空腔灌水问题引入的遍历性检验手段，而 T4 L99A 这类**本来就干的埋藏疏水腔**
    # 根本不存在湿盆——要求必须找到它，逻辑上不成立，且会让整条腿崩掉
    # （实测：复合物腿 λ_vdw=0 平衡 1 ns，空腔水数 100 次检查全为 0）。
    #
    # 不能把「水」这个针对溶剂腿引入的概念，变成整个 ABFE 管线的普遍物理假设。
    # 真正通用的硬门与具体物理盆无关：raw ESS、top-1% 权重、独立 walker/种子
    # 一致性、BAR–MBAR 一致性、样本数与去相关要求。
    #
    # 现在的行为：构造不到湿起点就只跑干起点，并如实记录「该诊断未评估」。
    # 不 raise、不要求调用方预先知道体系是干腔还是湿腔。
    # ------------------------------------------------------------------
    wet_available = bool(
        wet_seed and wet_seed.get("positions") is not None
        and wet_seed.get("reached_wet") is not False
    )
    active_modes = tuple(INDEPENDENT_ENDPOINT_INIT_MODES) if wet_available else ("dry",)
    seeds = {"dry": dry_seed}
    if wet_available:
        seeds["wet"] = wet_seed

    bank_dir = _independent_endpoint_bank_dir(checkpoint_dir, stage_type)
    expected_manifest = build_independent_endpoint_manifest(
        stage_type=stage_type,
        state_indices=state_indices,
        common_system_xml=common_system_xml,
        cv_xmls=[cv_xmls[k] for k in state_indices],
        temperature_K=temperature_q.value_in_unit(unit.kelvin),
        sample_interval=sample_interval,
        sample_steps=sample_steps,
        burn_in_steps=burn_in_steps,
        n_walkers_per_mode=n_walkers_per_mode,
        platform_name=platform_name,
        cavity_probe_radius_nm=cavity_probe_radius_nm,
        cavity_wet_min_waters=cavity_wet_min_waters,
        init_modes=list(active_modes),
        minimize_iterations=minimize_iterations,
        integrator_seed_base=integrator_seed_base,
        velocity_seed_base=velocity_seed_base,
        seed_identity=seed_identity,
        coion_identity=coion_identity,
    )
    if not _independent_endpoint_manifest_matches(bank_dir, expected_manifest):
        if os.path.isdir(bank_dir):
            shutil.rmtree(bank_dir, ignore_errors=True)
        os.makedirs(bank_dir, exist_ok=True)
        _atomic_write_json(os.path.join(bank_dir, "manifest.json"), expected_manifest)
    os.makedirs(bank_dir, exist_ok=True)

    records: Dict[str, Dict[str, Any]] = {}
    evaluator = None
    n_all_states = len(cv_xmls)

    # ------------------------------------------------------------------
    # 湿起点必须沿 λ 阶梯走（见 docstring 的"湿空腔阶梯"一段）。
    # 干起点不需要：干构型对任何 λ 都是相容的（配体占着体积本来就是耦合端的
    # 常态），所以每个干 walker 都从同一个独立的干种子起。
    # ------------------------------------------------------------------
    if lambdas_vdw_for_states is not None:
        lam_for = {
            int(k): float(v)
            for k, v in zip(state_indices, lambdas_vdw_for_states)
        }
        wet_order = sorted(state_indices, key=lambda k: lam_for[int(k)])
    else:
        # 缺 λ 值时退回本仓库的约定：state_indices 升序 == λ_vdw 降序，
        # 所以倒序就是"从最解耦走向最耦合"。
        wet_order = list(reversed(state_indices))

    visit: List[Tuple[int, str, int]] = []
    for mode in active_modes:
        order = list(state_indices) if mode == "dry" else wet_order
        for gk in order:
            for w in range(int(n_walkers_per_mode)):
                visit.append((int(gk), str(mode), int(w)))

    # 每个模态"当前的"起始构型。干的自始至终不变；湿的沿阶梯逐态往前传。
    current_seed: Dict[str, Dict[str, Any]] = {"dry": dict(dry_seed)}
    if wet_available:
        current_seed["wet"] = dict(wet_seed)

    for global_k, mode, walker in visit:
        key = f"{global_k}|{mode}|{walker}"
        path = _independent_endpoint_record_path(bank_dir, global_k, mode, walker)
        if os.path.isfile(path):
            with np.load(path, allow_pickle=False) as data:
                records[key] = {
                    "u_cv_kj_mol": np.asarray(data["u_cv_kj_mol"]),
                    "volume_nm3": np.asarray(data["volume_nm3"]),
                    "cavity_waters": np.asarray(data["cavity_waters"]),
                    "energy_column_indices": [
                        int(x) for x in np.asarray(data["energy_column_indices"])
                    ],
                    "segments": [{
                        "burn_in_steps": int(burn_in_steps),
                        "sample_steps": int(data["sampled_steps"]),
                        "n_frames": int(np.asarray(data["volume_nm3"]).size),
                        "reason": "resumed_from_disk",
                    }],
                    "sampled_steps": int(data["sampled_steps"]),
                    "global_state": int(global_k),
                    "init_mode": str(mode),
                    "walker": int(walker),
                }
                # 续跑时也要把湿阶梯接上，否则后面那些更耦合的态又会拿到最
                # 解耦端的湿构型（也就是这次修掉的那个 NaN）。
                if mode == "wet" and walker == 0 and "final_positions_nm" in data:
                    current_seed["wet"] = {
                        "positions": np.asarray(data["final_positions_nm"]) * unit.nanometer,
                        "box_vectors": np.asarray(data["final_box_nm"]) * unit.nanometer,
                        "origin": f"wet_ladder_resumed_from_state_{global_k}",
                    }
            continue

        seed = current_seed[mode]
        sim = _build_fixed_state_simulation(
            topology=topology,
            system_xml=common_system_xml,
            cv_xml=cv_xmls[global_k],
            require_group4=False,
            platform_name=platform_name,
            temperature_q=temperature_q,
            positions=seed["positions"],
            box_vectors=seed.get("box_vectors"),
            integrator_seed=_independent_endpoint_seed(
                global_k, mode, walker,
                INDEPENDENT_ENDPOINT_INTEGRATOR_SEED_BASE
                if integrator_seed_base is None else int(integrator_seed_base),
            ),
            velocity_seed=_independent_endpoint_seed(
                global_k, mode, walker,
                INDEPENDENT_ENDPOINT_VELOCITY_SEED_BASE
                if velocity_seed_base is None else int(velocity_seed_base),
            ),
        )
        try:
            if evaluator is None:
                evaluator = IBSSampler(
                    sim.context, n_all_states, temperature_q,
                    prefix=getattr(ibs_wrapper, "prefix", "abfe"),
                    ibs_wrapper=ibs_wrapper,
                )
                # 同 probe_adjacent_path_overlap_bank：evaluator 只用它自己的
                # 探针 Context，立刻丢掉对本 walker 动力学 Context 的引用，
                # 避免多个 Context 同时占显存。
                evaluator.context = None

            def _observe(frame_positions, frame_box):
                pos_nm = np.asarray(
                    frame_positions.value_in_unit(unit.nanometer)
                    if hasattr(frame_positions, "value_in_unit")
                    else frame_positions,
                    dtype=np.float64,
                )
                return count_cavity_waters(
                    pos_nm,
                    _box_vectors_to_nm_array(frame_box),
                    ligand_heavy_idx,
                    water_oxygen_idx,
                    probe_radius_nm=cavity_probe_radius_nm,
                )

            # 阶梯之后相邻两态的哈密顿量差别已经很小，但仍然先在**本态自己的**
            # 势能面上弛豫一下，把残余的不相容挤掉。
            if int(minimize_iterations) > 0:
                sim.minimizeEnergy(maxIterations=int(minimize_iterations))
            record = {
                "u_cv_kj_mol": None, "volume_nm3": None,
                "segments": [], "sampled_steps": 0,
            }
            try:
                _extend_state_trajectory(
                    sim, record, evaluator,
                    target_steps=int(sample_steps),
                    sample_interval=int(sample_interval),
                    burn_in_steps=int(burn_in_steps),
                    needs_burn_in=True,
                    segment_reason=f"independent_endpoint_{mode}_w{walker}",
                    frame_observer=_observe,
                )
            except openmm.OpenMMException as exc:
                raise RuntimeError(
                    f"独立端点采样在 λ 态 {global_k}（{mode} 起点, walker "
                    f"{walker}）上积分失败: {exc}。起始构型与本态的哈密顿量不相容。"
                    "湿起点已经沿 λ 阶梯逐态传递（wet ladder），若仍然炸，通常说明"
                    "相邻两个 λ 态之间跨度太大、或 minimize_iterations 不足。请加密"
                    "该段 λ 布点、增大 minimize_iterations，或把独立采样范围限制在"
                    "软核足够软的那一段。"
                ) from exc
            record["global_state"] = int(global_k)
            record["init_mode"] = str(mode)
            record["walker"] = int(walker)
            # evaluator 是用**全部**态建的，所以每帧能量含所有 λ 态的列。
            # 显式落盘这个布局，下游按物理 λ 索引取列，不靠位置约定。
            record["energy_column_indices"] = list(range(n_all_states))
            record["cavity_waters"] = np.asarray(
                record.pop("observer_values"), dtype=np.float64
            )
            final_state = sim.context.getState(getPositions=True)
            final_pos_nm = np.asarray(
                final_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=np.float64,
            )
            final_box_nm = _box_vectors_to_nm_array(
                final_state.getPeriodicBoxVectors()
            )
            np.savez(
                path,
                u_cv_kj_mol=record["u_cv_kj_mol"],
                volume_nm3=record["volume_nm3"],
                cavity_waters=record["cavity_waters"],
                energy_column_indices=np.asarray(
                    record["energy_column_indices"], dtype=np.int64
                ),
                sampled_steps=np.int64(record["sampled_steps"]),
                final_positions_nm=final_pos_nm,
                final_box_nm=final_box_nm,
            )
            records[key] = record
            # 只用 walker 0 的末构型往前传，保证阶梯是确定性的（walker 数量
            # 变化不改变后面每个态拿到的湿起点）。
            if mode == "wet" and walker == 0:
                current_seed["wet"] = {
                    "positions": final_state.getPositions(),
                    "box_vectors": final_state.getPeriodicBoxVectors(),
                    "origin": f"wet_ladder_from_state_{global_k}",
                }
        finally:
            del sim
            gc.collect()

    per_walker_cavity = {
        key: dict(
            cavity_occupancy_diagnostics(rec["cavity_waters"], cavity_wet_min_waters),
            global_state=int(rec["global_state"]),
            init_mode=str(rec["init_mode"]),
            walker=int(rec["walker"]),
        )
        for key, rec in records.items()
    }
    return {
        "protocol_version": int(INDEPENDENT_ENDPOINT_PROTOCOL_VERSION),
        "state_indices": list(state_indices),
        "global_state_indices": list(global_state_indices),
        "records": records,
        "per_walker_cavity": per_walker_cavity,
        "bank_dir": bank_dir,
        "dynamics_hamiltonian": "U_common_plus_single_cv_int_no_wca",
        "wca_present_in_production_dynamics": False,
        "n_walkers_per_mode": int(n_walkers_per_mode),
        "init_modes": list(active_modes),
        # 湿盆是否在预算内被观察到。False 时湿/干门是**未评估**，不是失败。
        "wet_basin_found": bool(wet_available),
        "wet_seed_diagnostics": (wet_seed.get("cavity_diagnostics") if wet_seed else None),
        "wet_seed_equilibration_steps": (
            wet_seed.get("equilibration_steps") if wet_seed else None),
    }


def _reduced_energies_for_record(
    record: Dict[str, Any],
    state_indices: Sequence[int],
    kt: float,
    lrc_coeff: Optional[np.ndarray],
) -> np.ndarray:
    """一条独立轨迹在 `state_indices` 全部态上的约化能量 (n_frames, K)。

    列的物理身份由 `record["energy_column_indices"]` **显式**声明，绝不靠
    "列号恰好等于全局 λ 索引"这种隐式约定去猜——那种约定在能量矩阵只存了窗口
    子集时会静默取错列，而取错列的自由能看起来完全正常。

    LRC 用 `lrc_coeff[k] / V(t)` 逐帧加上——与 `IBSSampler.collect_energies` 的
    `target_energies = softcore_energies + lrc_energies` 是同一个式子、同一组
    系数（`ibs_wrapper.lj_tail_lrc_coeff_kj_mol`）。这里绝不另算一份 LRC，
    否则独立端点段和 IBS 段就不是同一个热力学量，拼接无意义。
    """
    columns = [int(c) for c in record["energy_column_indices"]]
    position_of = {c: i for i, c in enumerate(columns)}
    missing = [int(k) for k in state_indices if int(k) not in position_of]
    if missing:
        raise KeyError(
            f"独立端点轨迹的能量矩阵没有这些 λ 态的列: {missing}；"
            f"该记录记录的列布局是 {columns}"
        )
    idx = np.asarray([position_of[int(k)] for k in state_indices], dtype=int)
    e_kj = np.asarray(record["u_cv_kj_mol"], dtype=np.float64)[:, idx].copy()
    if lrc_coeff is not None:
        volume = np.asarray(record["volume_nm3"], dtype=np.float64)
        if volume.shape[0] != e_kj.shape[0]:
            raise RuntimeError("独立端点轨迹的体积序列与能量帧数不一致")
        e_kj = e_kj + np.asarray(lrc_coeff, dtype=np.float64)[idx][None, :] / volume[:, None]
    return e_kj / float(kt)


def _decorrelate_independent_record(
    record: Dict[str, Any],
    reduced: np.ndarray,
    own_column: int,
    min_frames_for_subsampling: int = 20,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """对一条独立轨迹做逐 segment 时间去相关，并在两个候选序列里取**更差**的那个。

    候选一是这个态自己的约化势能；候选二是逐帧空腔水数。这不是可有可无的
    保险：本模块要采的慢模态是**结构性**的（水进出配体空腔），而
    STAGE2_ROOT_CAUSE_2026-08-28.md §3.2 已经实测出这个模态在能量上几乎不
    可见（window 2 相邻 <ΔU> 仅 0.4~0.6 kT，却贡献 ~21 kJ/mol 的 TΔS）。只用
    能量序列估自相关时间会系统性低估 g，从而高估独立样本数——正是"稳定地
    收敛到错值"那类失效的统计学版本。取两者中样本更少的那个，宁可保守。
    """
    candidates = {"reduced_potential": np.asarray(reduced[:, own_column], dtype=np.float64)}
    cavity = record.get("cavity_waters")
    if cavity is not None:
        cavity = np.asarray(cavity, dtype=np.float64).ravel()
        if cavity.size == reduced.shape[0] and float(np.std(cavity)) > 0.0:
            candidates["cavity_water_count"] = cavity

    best_key = None
    best_indices = None
    report: Dict[str, Any] = {}
    for key, series in candidates.items():
        indices, g_list, short_segs = _decorrelate_per_segment(
            series, record["segments"],
            min_frames_for_subsampling=min_frames_for_subsampling,
        )
        report[key] = {
            "n_decorrelated": int(indices.size),
            "statistical_inefficiency": [float(x) for x in g_list],
            "short_segments_diagnostic_only": short_segs,
        }
        if best_indices is None or indices.size < best_indices.size:
            best_indices, best_key = indices, key
    report["selected_series"] = best_key
    report["selected_reason"] = "两个候选序列里去相关后样本更少（自相关更慢）的那个"
    return best_indices, report


def solve_independent_endpoint_states(
    bank: Dict[str, Any],
    kt: float,
    lrc_coeff: Optional[np.ndarray] = None,
    init_modes: Optional[Sequence[str]] = None,
    min_decorrelated_samples: int = 20,
    min_frames_for_subsampling: int = 20,
) -> Dict[str, Any]:
    """把独立端点轨迹解成 λ 态之间的自由能曲线。

    与 IBS 的局部 TMBAR 有一个**结构性**区别：这里每一个物理态都有 `n_k > 0`
    的真实样本，不存在"一个采样分布 + 一堆零样本目标态"的单向重加权。所以
    `raw` 支撑度天然就是健康的，TARGET_SUPPORT_GATE 那道门在这条路径上是
    自然满足而不是被绕过——这正是修复的意义所在。

    `init_modes` 限定只用某一组起点（"wet" 或 "dry"）的 walker 求解，供
    `endpoint_wet_dry_hysteresis_gate` 做双起点对照。默认用全部。
    """
    state_indices = [int(x) for x in bank["state_indices"]]
    K = len(state_indices)
    wanted_modes = (
        set(str(m) for m in init_modes) if init_modes is not None
        else set(INDEPENDENT_ENDPOINT_INIT_MODES)
    )
    position_of = {k: i for i, k in enumerate(state_indices)}

    blocks: List[List[np.ndarray]] = [[] for _ in range(K)]
    decorrelation_report: Dict[str, Any] = {}
    used_records: List[Dict[str, Any]] = []
    for key, record in sorted(bank["records"].items()):
        if str(record["init_mode"]) not in wanted_modes:
            continue
        own = position_of[int(record["global_state"])]
        reduced = _reduced_energies_for_record(record, state_indices, kt, lrc_coeff)
        indices, report = _decorrelate_independent_record(
            record, reduced, own, min_frames_for_subsampling=min_frames_for_subsampling
        )
        decorrelation_report[key] = report
        if indices.size == 0:
            continue
        blocks[own].append(reduced[indices, :].T)  # (K, n_decorrelated)
        used_records.append(record)

    n_k = np.asarray([sum(b.shape[1] for b in blk) for blk in blocks], dtype=int)
    if np.any(n_k <= 0):
        return {
            "error": "independent_endpoint_state_without_samples",
            "n_k": n_k.tolist(),
            "state_indices": state_indices,
            "converged": False,
        }
    if int(np.min(n_k)) < int(min_decorrelated_samples):
        return {
            "error": "independent_endpoint_insufficient_decorrelated_samples",
            "n_k": n_k.tolist(),
            "min_decorrelated_samples_threshold": int(min_decorrelated_samples),
            "state_indices": state_indices,
            "converged": False,
        }
    if not HAS_PYMBAR:
        return {"error": "pymbar_not_installed", "converged": False}

    u_kn = np.concatenate([np.concatenate(blk, axis=1) for blk in blocks], axis=1)
    u_kn = u_kn - np.min(u_kn, axis=0, keepdims=True)
    mbar = _build_mbar_compatible(
        u_kn, n_k, relative_tolerance=1e-7, initialize="BAR", solver_protocol="default"
    )
    res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
    df_matrix, ddf_matrix = _extract_free_energy_arrays(res, require_uncertainty=True)

    f_kj = (df_matrix[0, :] * float(kt)).astype(float)
    df_kj = (ddf_matrix[0, :] * float(kt)).astype(float)
    delta_g = float(df_matrix[0, K - 1] * float(kt))
    delta_g_sigma = float(ddf_matrix[0, K - 1] * float(kt))

    neff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)
    cavity_by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for record in used_records:
        cavity_by_mode.setdefault(str(record["init_mode"]), []).append(
            cavity_occupancy_diagnostics(record["cavity_waters"])
        )
    return {
        "protocol_version": int(INDEPENDENT_ENDPOINT_PROTOCOL_VERSION),
        "state_indices": state_indices,
        "global_state_indices": [
            int(x) for x in (bank.get("global_state_indices") or state_indices)
        ],
        "init_modes_used": sorted(wanted_modes),
        "f_kJ_mol": f_kj.tolist(),
        "df_kJ_mol": df_kj.tolist(),
        "delta_G_kJ_mol": delta_g,
        "delta_G_sigma_kJ_mol": delta_g_sigma,
        "n_k_decorrelated": n_k.tolist(),
        "min_decorrelated_samples": int(np.min(n_k)),
        "min_decorrelated_samples_threshold": int(min_decorrelated_samples),
        "effective_sample_number": neff.tolist(),
        "min_effective_sample_number": float(np.min(neff)),
        "decorrelation": decorrelation_report,
        "cavity_by_init_mode": cavity_by_mode,
        "estimator": "multi_state_MBAR_all_states_sampled",
        "sampling_note": (
            "每个 λ 态都有自己的独立固定-λ 轨迹（n_k>0），不存在单向重加权；"
            "动力学不含 Group-1 IBS 偏置与 Group-4 WCA 防护壳。"
        ),
        "converged": True,
    }


def endpoint_wet_dry_hysteresis_gate(
    result_wet: Dict[str, Any],
    result_dry: Dict[str, Any],
    per_walker_cavity: Dict[str, Any],
    max_sigma: float = ENDPOINT_WET_DRY_MAX_SIGMA,
    wet_arm_available: Optional[bool] = None,
) -> Dict[str, Any]:
    """湿/干双起点验收门。abfe_core.STAGE2_HYSTERESIS_MAX_SIGMA 的第一个真实执行点。

    两道**都必须过**的判据：

      (1) 数值一致：|ΔF_wet - ΔF_dry| <= max_sigma * sqrt(σ_wet² + σ_dry²)。
          两组来自各自独立的种子与起始构型，样本互不重叠，所以 σ 按独立量
          平方相加；若将来改成共享 walker，必须把协方差补进来。
      (2) 模态确实连通：至少要有一个 walker 真的在湿态和干态之间转换过
          （n_wet_dry_transitions > 0），**或者**湿组与干组的 wet_fraction
          区间有重叠。若湿组全程 wet_fraction=1、干组全程 =0、两边转换次数
          都是 0，那就是两个互不连通的模态各自被采了一遍——此时 (1) 即使
          通过也毫无意义（两个都错、且错得一样，差值照样是 0）。这一条是
          STAGE2_ROOT_CAUSE_2026-08-28.md §4 "所有门测的都是已采到的构型之间
          的散布" 那个教训的直接产物：必须有一项直接证明"该采的构型采到了"。
    """
    # ------------------------------------------------------------------
    # 🔑 [2026-08-31] 湿/干双起点是**条件性诊断**，不是普遍适用的硬门。
    # 只有真的构造出两类起点时才评估；构造不到（例如 T4 L99A 这类本来就干的
    # 埋藏疏水腔，预算内根本观察不到湿盆）就返回 evaluated=False / passed=None。
    #   passed=None 既不算失败、也不算通过，**不参与 converged 合取**，
    #   但必须把「该诊断未评估」如实带进结果与 provenance。
    # 真正通用的硬门与具体物理盆无关：raw ESS、top-1% 权重、独立 walker/种子
    # 一致性、BAR–MBAR 一致性、样本数与去相关要求。
    # ------------------------------------------------------------------
    if wet_arm_available is False or not result_wet:
        return {
            "evaluated": False,
            "passed": None,
            "reason": "wet_start_not_observed_within_budget",
            "note": ("预算内未观察到湿盆，已只跑干起点。湿/干双起点门未评估——"
                     "既不算通过也不算失败，不参与 converged 判定。"
                     "该端点段的遍历性未经双起点检验。"),
            "max_sigma": float(max_sigma),
        }
    failures: List[str] = []
    for label, result in (("wet", result_wet), ("dry", result_dry)):
        if not result or result.get("converged") is not True:
            failures.append(f"{label}_solve_failed")
    if failures:
        return {
            "evaluated": True,
            "passed": False, "failed_checks": failures,
            "failure_reason": "wet_dry_hysteresis_unavailable",
            "max_sigma": float(max_sigma),
        }

    dg_wet = float(result_wet["delta_G_kJ_mol"])
    dg_dry = float(result_dry["delta_G_kJ_mol"])
    s_wet = float(result_wet["delta_G_sigma_kJ_mol"])
    s_dry = float(result_dry["delta_G_sigma_kJ_mol"])
    delta = abs(dg_wet - dg_dry)
    sigma_combined = float(np.hypot(s_wet, s_dry))
    numeric_ok = bool(
        np.isfinite(delta) and np.isfinite(sigma_combined) and sigma_combined > 0.0
        and delta <= float(max_sigma) * sigma_combined
    )
    if not numeric_ok:
        failures.append("wet_dry_delta_exceeds_sigma_gate")

    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for entry in per_walker_cavity.values():
        by_mode.setdefault(str(entry.get("init_mode")), []).append(entry)
    transitions = [
        int(e.get("n_wet_dry_transitions") or 0)
        for entries in by_mode.values() for e in entries
    ]
    wet_fracs = {
        mode: [float(e["wet_fraction"]) for e in entries
               if e.get("wet_fraction") is not None]
        for mode, entries in by_mode.items()
    }
    any_transition = bool(transitions) and max(transitions) > 0
    overlap = False
    if wet_fracs.get("wet") and wet_fracs.get("dry"):
        overlap = bool(
            min(wet_fracs["wet"]) <= max(wet_fracs["dry"])
            and min(wet_fracs["dry"]) <= max(wet_fracs["wet"])
        )
    ergodicity_ok = bool(any_transition or overlap)
    if not ergodicity_ok:
        failures.append("wet_and_dry_walkers_never_exchanged_modes")

    return {
        "evaluated": True,
        "passed": not failures,
        "failed_checks": failures,
        "failure_reason": (
            None if not failures else "stage2_wet_dry_hysteresis_failed"
        ),
        "delta_G_wet_kJ_mol": dg_wet,
        "delta_G_dry_kJ_mol": dg_dry,
        "delta_kJ_mol": delta,
        "sigma_wet_kJ_mol": s_wet,
        "sigma_dry_kJ_mol": s_dry,
        "sigma_combined_kJ_mol": sigma_combined,
        "max_sigma": float(max_sigma),
        "allowed_delta_kJ_mol": float(max_sigma) * sigma_combined,
        "max_wet_dry_transitions": max(transitions) if transitions else 0,
        "wet_fraction_by_init_mode": wet_fracs,
        "modes_are_connected": ergodicity_ok,
        "sigma_combination": "independent_seeds_quadrature",
    }


def combine_ibs_and_independent_endpoint(
    ibs_result: Dict[str, Any],
    endpoint_result: Dict[str, Any],
    wet_dry_gate: Dict[str, Any],
    n_states: int,
    # 默认值在**调用时**解析：TARGET_SUPPORT_MIN_ABSOLUTE_ESS 定义在本文件更靠后
    # 的位置（紧邻它所服务的 ESS 门），def 时求值会 NameError。
    target_support_min_absolute_ess: Optional[float] = None,
) -> Dict[str, Any]:
    """把 IBS 段与独立端点段拼成一个 stage2 结果。

    λ 路径被切成两段：低 λ 索引那段仍由 IBS 窗口重加权求解（那里配体还实实在在
    占着体积，不存在"水塌进空腔"这个未被采到的构型），高 λ 索引那段（λ_vdw→0）
    改由每个态自己的独立固定-λ 轨迹求解。两段在**同一个** λ 索引上相接。

    方向约定必须一致才能相加：`solve_stage_integrated` 的 `total_delta_G` 是沿
    λ 索引**升序**逐段累加的（见协方差链那段），而 `solve_independent_endpoint_
    states` 的 `delta_G_kJ_mol` 是 `f[K-1] - f[0]`、`state_indices` 同样升序——
    两者同向，接点处的 f 抵消，直接相加即为全程。

    σ 按独立量平方相加：两段来自**完全不相干**的轨迹（不同 System 变体、不同
    Context、不同种子），样本没有任何重叠，所以没有需要补的协方差项。这与同一
    窗口内两个端点之间必须直接读 dDelta_f（有协方差）是两回事，不要混淆。

    `converged` 是三者的合取：IBS 段自己的全部硬门、独立端点段解出且样本足够、
    以及湿/干双起点门。任何一项没有明确通过都判失败——尤其 `wet_dry_gate` 缺失
    时绝不当作"没测到就放行"。
    """
    if target_support_min_absolute_ess is None:
        target_support_min_absolute_ess = TARGET_SUPPORT_MIN_ABSOLUTE_ESS
    ibs_lams = [int(x) for x in (ibs_result.get("lambdas") or [])]
    # 拼接必须用**全局** λ 索引；`state_indices` 在只为端点块建窗口时是局部的。
    endpoint_states = [
        int(x) for x in (
            endpoint_result.get("global_state_indices")
            or endpoint_result.get("state_indices")
            or []
        )
    ]
    if not ibs_lams or not endpoint_states:
        return {
            "error": "combine_missing_segment_lambdas",
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }
    if ibs_lams != sorted(ibs_lams) or endpoint_states != sorted(endpoint_states):
        return {
            "error": "combine_segments_not_ascending",
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }
    join = endpoint_states[0]
    if ibs_lams[-1] != join:
        return {
            "error": (
                f"combine_join_mismatch: IBS 段止于 λ 索引 {ibs_lams[-1]}，"
                f"独立端点段起于 {join}，两段必须在同一个 λ 索引相接"
            ),
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }
    covered = sorted(set(ibs_lams) | set(endpoint_states))
    if covered != list(range(int(n_states))):
        return {
            "error": (
                f"combine_incomplete_lambda_coverage: 两段合起来覆盖 {covered}，"
                f"期望 [0,{int(n_states)})"
            ),
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }

    ibs_dg = float(ibs_result.get("total_delta_G", 0.0))
    ibs_sigma = float(ibs_result.get("total_error", 0.0))
    end_dg = float(endpoint_result.get("delta_G_kJ_mol", 0.0))
    end_sigma = float(endpoint_result.get("delta_G_sigma_kJ_mol", 0.0))
    if not all(np.isfinite(x) for x in (ibs_dg, ibs_sigma, end_dg, end_sigma)):
        return {
            "error": "combine_nonfinite_segment",
            "converged": False, "total_delta_G": 0.0, "total_error": 999.9,
        }

    endpoint_min_ess = endpoint_result.get("min_effective_sample_number")
    endpoint_support_ok = bool(
        endpoint_min_ess is not None
        and np.isfinite(endpoint_min_ess)
        and _meets_minimum_with_roundoff(
            float(endpoint_min_ess), float(target_support_min_absolute_ess)
        )
    )
    ibs_gate = ibs_result.get("target_support_gate")
    ibs_support_ok = bool(isinstance(ibs_gate, dict) and ibs_gate.get("passed") is True)
    failed_checks: List[str] = []
    if not ibs_support_ok:
        failed_checks.append("ibs_segment_target_support")
    if not endpoint_support_ok:
        failed_checks.append("independent_endpoint_effective_sample_number")
    combined_target_support = {
        "protocol_version": int(TARGET_SUPPORT_GATE_PROTOCOL_VERSION),
        "passed": not failed_checks,
        "failure_reason": (
            "insufficient_target_support" if failed_checks else None
        ),
        "failed_checks": failed_checks,
        "ibs_segment_gate": ibs_gate,
        "independent_endpoint_min_effective_sample_number": (
            float(endpoint_min_ess) if endpoint_min_ess is not None else None
        ),
        "raw_min_absolute_ess_threshold": float(target_support_min_absolute_ess),
        "segment_note": (
            "独立端点段每个物理态都有 n_k>0 的真实样本，不存在单向重加权，"
            "所以它的支撑度用 MBAR 自己的有效样本数直接判定；IBS 段仍按"
            "raw 重加权支撑度判定。"
        ),
    }

    # 🔑 [2026-08-31] 湿/干门是条件性诊断：
    #   passed is True  -> 通过
    #   passed is False -> 失败，参与合取（真的检出了迟滞）
    #   passed is None  -> **未评估**（预算内没观察到湿盆），不参与合取，
    #                      但必须在结果里留下明确警告。
    _wd = wet_dry_gate if isinstance(wet_dry_gate, dict) else {}
    _wd_passed = _wd.get("passed", False) if _wd else False
    wet_dry_blocks = (_wd_passed is False)      # None 不阻塞
    wet_dry_unevaluated = (_wd_passed is None)
    converged = bool(
        ibs_result.get("converged") is True
        and endpoint_result.get("converged") is True
        and combined_target_support["passed"]
        and not wet_dry_blocks
    )
    return {
        "stage": ibs_result.get("stage", "vanishing"),
        "total_delta_G": ibs_dg + end_dg,
        "total_error": float(np.hypot(ibs_sigma, end_sigma)),
        "total_error_method": "independent_segments_quadrature_ibs_plus_endpoint",
        "converged": converged,
        "join_lambda_index": int(join),
        "n_states": int(n_states),
        "ibs_segment": {
            "lambda_indices": ibs_lams,
            "delta_G_kJ_mol": ibs_dg,
            "uncertainty_kJ_mol": ibs_sigma,
            "converged": ibs_result.get("converged"),
        },
        "independent_endpoint_segment": {
            "lambda_indices": endpoint_states,
            "delta_G_kJ_mol": end_dg,
            "uncertainty_kJ_mol": end_sigma,
            "converged": endpoint_result.get("converged"),
            "n_k_decorrelated": endpoint_result.get("n_k_decorrelated"),
            "min_effective_sample_number": endpoint_min_ess,
        },
        "target_support_gate": combined_target_support,
        "target_support_gate_protocol_version": int(
            TARGET_SUPPORT_GATE_PROTOCOL_VERSION
        ),
        "wet_dry_hysteresis_gate": wet_dry_gate,
        "wet_dry_hysteresis_evaluated": (not wet_dry_unevaluated),
        # 未评估时必须有一条显式警告随结果走，不能只是 passed=None 一个静默的值。
        "warnings": ([
            "湿/干双起点门未评估（预算内未观察到湿盆，已只跑干起点）——"
            "该端点段的遍历性未经双起点检验。"
        ] if wet_dry_unevaluated else []),
        "independent_endpoint_protocol_version": int(
            INDEPENDENT_ENDPOINT_PROTOCOL_VERSION
        ),
        "min_overlap": ibs_result.get("min_overlap"),
        "min_overlap_threshold": ibs_result.get("min_overlap_threshold"),
        "raw_min_absolute_ess": ibs_result.get("raw_min_absolute_ess"),
        "raw_min_absolute_ess_threshold": ibs_result.get(
            "raw_min_absolute_ess_threshold"
        ),
        "max_top1pct_raw_weight": ibs_result.get("max_top1pct_raw_weight"),
        "max_top1pct_raw_weight_threshold": ibs_result.get(
            "max_top1pct_raw_weight_threshold"
        ),
        "window_overlap_diagnostics": ibs_result.get("window_overlap_diagnostics"),
        "min_decorrelated_samples": ibs_result.get("min_decorrelated_samples"),
        "min_decorrelated_samples_threshold": ibs_result.get(
            "min_decorrelated_samples_threshold"
        ),
        "max_endpoint_uncertainty_kJ_mol": ibs_result.get(
            "max_endpoint_uncertainty_kJ_mol"
        ),
        "max_endpoint_uncertainty_kJ_mol_threshold": ibs_result.get(
            "max_endpoint_uncertainty_kJ_mol_threshold"
        ),
        "coverage_diagnostics": {
            "covered_lambda_indices": covered,
            "n_covered_lambda_indices": len(covered),
            "ibs_lambda_indices": ibs_lams,
            "independent_endpoint_lambda_indices": endpoint_states,
            "note": (
                "stage2 由两段组成：IBS 重加权段 + 独立固定-λ 端点段，"
                f"在 λ 索引 {int(join)} 相接。"
            ),
        },
        "method": "IBS-TMBAR + independent fixed-lambda endpoint states",
    }



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
        dispersion_protocol: Optional[str] = None,
        # [B6-FIX] 这条腿的环境类型。溶剂腿构造时天然是 soluble（runabfe 刻意不给
        # 溶剂腿传 environment_type/membrane，B1 的接线契约测试钉着），所以纯水盒
        # 里那条**合法**的均匀体相尾项修正会被正确保留。
        environment_type: Optional[str] = None,
        restraint_params: Optional[Dict] = None,
        prefix: str = "abfe_dual",
        platform_name: str = "CUDA",
        output_dir: str = "./output",
        checkpoint_dir: str = "./checkpoints",
        pilot_lambdas: Optional[List[float]] = None,
        pilot_mean_dU_dlambda: Optional[List[float]] = None,
        # 🔑 [窗口预热状态机重构 Stage 0] pilot 网格早就在 _sample_scalar_metric
        # 里算出了 std_dU_dlambda_kJ_mol/n_derivative_samples，只是之前没被
        # 提取、没被传到这里——没有它们就没法判断某个 pilot TI 种子对某个
        # 窗口够不够精确（见 pilot_ti_seed_trust_diagnostics）。默认 None
        # 向后兼容：旧调用点/旧 preopt cache 不传这两个字段时，行为不变。
        pilot_std_dU_dlambda: Optional[List[float]] = None,
        pilot_n_dU_dlambda_samples: Optional[List[int]] = None,
        coion_identity: Optional[Dict[str, Any]] = None,
        seed_ledger: Optional[Exp019SeedLedger] = None,
        leg_name: Optional[str] = None,
        # EXP-030: return a fresh Force for every System/window. OpenMM Force
        # objects cannot be owned by multiple Systems. The coefficient factory
        # receives this window's local lambda vectors and returns one A_k/state.
        residual_basis_force_factory: Optional[Any] = None,
        residual_state_coefficients_factory: Optional[Any] = None,
        residual_energy_offset_kj_mol: float = 0.0,
        # Canonical JointStateScoreSpec hash, persisted as resume identity.
        sampling_score_sha256: Optional[str] = None,
        residual_plugin_identity: Optional[Dict[str, Any]] = None,
        residual_em_policy: str = "no_residual_twin",
        residual_feature_name: str = "Outer-Lambda Local Residual for IBS",
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
        # [B6] 膜体系用非 legacy 色散路线时，炼金 ligand–environment 的均匀密度
        # LRC 必须关闭（memtodolist §1.3）。None → 与改动前逐位一致。
        self.dispersion_protocol = dispersion_protocol
        self.environment_type = environment_type
        self.boresch = restraint_params
        self.prefix = prefix
        self.platform_name = platform_name
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        # B5: the caller resolves/freeze-checks this once; manifests only
        # record the payload and never read the spec or select an ion.
        self.coion_identity = coion_identity
        self.seed_ledger = seed_ledger
        self.leg_name = str(leg_name) if leg_name is not None else None
        if self.seed_ledger is not None and self.leg_name != self.seed_ledger.leg:
            raise ValueError("IBS manager leg_name 与 EXP-019 seed ledger 不一致")
        residual_enabled = residual_basis_force_factory is not None
        if residual_enabled != (residual_state_coefficients_factory is not None):
            raise ValueError(
                "residual_basis_force_factory 与 residual_state_coefficients_factory "
                "必须同时提供或同时省略"
            )
        if residual_enabled and not callable(residual_basis_force_factory):
            raise TypeError("residual_basis_force_factory 必须可调用")
        if residual_enabled and not callable(residual_state_coefficients_factory):
            raise TypeError("residual_state_coefficients_factory 必须可调用")
        if not math.isfinite(float(residual_energy_offset_kj_mol)):
            raise ValueError("residual_energy_offset_kj_mol 必须有限")
        if not residual_enabled and float(residual_energy_offset_kj_mol) != 0.0:
            raise ValueError("未启用 residual 时 energy offset 必须为 0")
        if residual_enabled and sampling_score_sha256 is None:
            raise ValueError("residual-enabled manager 必须绑定 sampling_score_sha256")
        if sampling_score_sha256 is not None:
            score_hash = str(sampling_score_sha256).lower()
            if len(score_hash) != 64 or any(ch not in "0123456789abcdef" for ch in score_hash):
                raise ValueError("sampling_score_sha256 必须是 64 位十六进制 SHA-256")
            self.sampling_score_sha256 = score_hash
        else:
            self.sampling_score_sha256 = None
        self.residual_basis_force_factory = residual_basis_force_factory
        self.residual_state_coefficients_factory = residual_state_coefficients_factory
        self.residual_energy_offset_kj_mol = float(residual_energy_offset_kj_mol)
        self.residual_plugin_identity = (
            dict(residual_plugin_identity) if residual_plugin_identity is not None else None
        )
        self.residual_em_policy = str(residual_em_policy)
        self.residual_feature_name = str(residual_feature_name)
        if residual_enabled and self.residual_em_policy != "no_residual_twin":
            raise ValueError(
                "residual-enabled manager 必须使用 no_residual_twin EM 策略"
            )
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
        # 🔑 [窗口预热状态机重构 Stage 0] 见上面参数说明；同样 None 表示"这个
        # manager 没有精度数据"，消费方（未来的 pilot-first 信任度判断）必须
        # 按 pilot_ti_seed_trust_diagnostics() 的约定把它当"不可信"处理，
        # 不能当异常。
        self.pilot_std_dU_dlambda = pilot_std_dU_dlambda
        self.pilot_n_dU_dlambda_samples = pilot_n_dU_dlambda_samples
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        _temp_q = temperature if hasattr(temperature, 'value_in_unit') else temperature * unit.kelvin
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * _temp_q).value_in_unit(unit.kilojoule_per_mole)

    def _seed_for(
        self,
        phase: str,
        stage: str,
        window: Any,
        stream: str,
        attempt: int = 0,
    ) -> Optional[int]:
        if self.seed_ledger is None:
            return None
        return self.seed_ledger.derive(phase, stage, window, stream, attempt)

    def _build_window_system(self, lc_win, lv_win, resolved_box, positions):
        """构建单个窗口的 (System, IBSBiasForce)。

        子类覆盖这个方法即可接入不同的 CV 构造（例如 Shadow-Coulomb 去电荷），
        同时复用本类其余的窗口调度/最小化/Boresch 爬坡/渐进预热/生产采样/
        断点续传/TMBAR 落盘逻辑——这些逻辑只依赖 (win_sys, ibs_wrap) 这两个
        返回值和 lc_win 的长度，不关心 CV 具体代表哪种物理量。
        """
        win_sys_xml = XmlSerializer.serialize(self.system_template)
        residual_kwargs = {}
        if self.residual_basis_force_factory is not None:
            coefficients = [
                float(value)
                for value in self.residual_state_coefficients_factory(lc_win, lv_win)
            ]
            if len(coefficients) != len(lc_win):
                raise ValueError(
                    "residual coefficient factory 返回长度与当前窗口状态数不一致："
                    f"{len(coefficients)} != {len(lc_win)}"
                )
            if not all(math.isfinite(value) for value in coefficients):
                raise ValueError("residual coefficient factory 返回了非有限 A_k")
            residual_force = self.residual_basis_force_factory()
            if residual_force is None:
                raise ValueError("residual_basis_force_factory 返回了 None")
            residual_kwargs = {
                "residual_basis_force": residual_force,
                "residual_state_coefficients": coefficients,
                "residual_energy_offset_kj_mol": self.residual_energy_offset_kj_mol,
            }
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
            dispersion_protocol=self.dispersion_protocol,
            environment_type=self.environment_type,
            **residual_kwargs,
        )

    def _enqueue_window_snapshot(self, window_idx: int, stage_type: str, sampler) -> None:
        """同步原子刷盘能量快照。"""
        e_arr = np.array(sampler.energy_history, dtype=np.float64, copy=True) if sampler.energy_history else np.zeros((0, 0), dtype=np.float64)
        e_save = e_arr.T if e_arr.size > 0 else np.zeros((0, 0), dtype=np.float64)
        bias = np.array(sampler.bias_history, dtype=np.float64, copy=True) if sampler.bias_history else np.zeros((0,), dtype=np.float64)
        base = np.array(sampler.base_energy_history, dtype=np.float64, copy=True) if sampler.base_energy_history else np.zeros((0,), dtype=np.float64)
        sampling_states = np.asarray(sampler.sampling_state_energy_history, dtype=np.float64)
        # Durable residual ledger contract: frames x states.  Analysis returns
        # states x frames below in _load_validated_joint_score_ledgers, but the
        # on-disk producer/consumer contract stays row-wise and unambiguous.
        sampling_states_save = sampling_states if sampling_states.size else np.zeros((0, 0), dtype=np.float64)
        residual_basis = np.asarray(sampler.residual_basis_history, dtype=np.float64)

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
        if sampling_states_save.size > 0:
            _atomic_save_npy(
                os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_sampling_states.npy"),
                sampling_states_save,
            )
        if residual_basis.size > 0:
            _atomic_save_npy(
                os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_residual_basis.npy"),
                residual_basis,
            )

    def _production_disaster_rollback(
        self,
        sim,
        pos_backup,
        sampler,
        production_history_backup_len: int,
        stage_type: str,
        window_idx: int,
        attempt: int,
        e_total_n: float,
        fmax,
        win_sys,
        debug_mode: bool,
        label_prefix: str = "",
        progress_note: str = "",
        diagnose_prefix: str = "",
    ) -> int:
        """灾难回滚共享实现（生产主循环 + 余数补齐块此前各自内联一份几乎
        相同的代码，见 run_all_windows）。回退坐标 → 用同一套种子规则重设
        速度 → 局部最小化释放应力 → 步长减半 → 同步截断生产 history 的顺序
        和参数在两处必须逐字节一致，抽成一个方法只把两处真正不同的部分
        （attempt 编号、print 文案前缀、诊断文件名前缀）作为参数传入，不改
        choreography 本身。返回同步丢弃的 history 帧数（供调用方打日志用）。
        两个生产异常入口都汇聚到同一个 ``_truncate_production_history(`` 截断点。
        """
        sim.context.setPositions(pos_backup)
        recovery_velocity_seed = self._seed_for(
            "recovery", stage_type, window_idx, "velocity", attempt=attempt
        )
        if recovery_velocity_seed is None:
            sim.context.setVelocitiesToTemperature(self.temperature)
        else:
            sim.context.setVelocitiesToTemperature(
                self.temperature, recovery_velocity_seed
            )
        print(f"    🔧 {label_prefix}触发回退，执行局部最小化释放应力...")
        sim.minimizeEnergy(maxIterations=2000, tolerance=1.0)
        current_dt_ps = sim.integrator.getStepSize().value_in_unit(unit.picoseconds)
        new_dt_ps = max(0.0001, current_dt_ps * 0.5)
        sim.integrator.setStepSize(new_dt_ps * unit.picoseconds)
        # 🔑 [P1-13] 坐标退回备份点了，备份之后写入的帧属于被放弃的分支，
        # 必须同步截断——否则它们会和重启后长出的新分支拼成一条"连续"轨迹
        # 交给 _decorrelate_by_worst_target_state 估自相关。
        dropped = _truncate_production_history(sampler, production_history_backup_len)
        _start_production_segment(sampler, "catastrophe_rollback_rebuild")
        fmax_report = fmax if fmax is not None else float("nan")
        print(
            f"    ⚠️ {label_prefix}灾难检测触发: {progress_note}"
            f"E_total={e_total_n:.1f}, max|F|={fmax_report:.1f}. "
            f"已回退坐标并将步长降至 {new_dt_ps*1000.0:.1f} fs"
            + (f"；同步丢弃被放弃分支的 {dropped} 帧生产 history" if dropped else "")
        )
        if debug_mode:
            diagnose_force_groups_detailed(sim.context, win_sys, prefix=diagnose_prefix)
            diagnose_force_breakdown(sim.context, win_sys, prefix=diagnose_prefix)
        return dropped

    def run_all_windows(
        self,
        positions,
        box_vectors,
        n_steps_per_window: int,
        steps_per_update: int,
        stage_type: str = "coul",
        initial_velocities=None,
        resume: bool = False,
        enable_gradual_warmup: bool = True,
        warmup_steps: int = 500000,
        min_bias_updates: int = 12,
        max_bias_updates: int = 50,
        required_consecutive_bias_updates: int = 3,
        # 🔑 [窗口预热状态机重构 Stage 1a] 改名 max_bias_warmup_steps →
        # max_bias_learning_steps：这个参数从来控制的都是"learning 阶段的步数
        # 预算"（sgd_step_budget/full_bias_step_budget 都从它派生），跟真正的
        # "爬坡"（dt/bias ramp）完全是两回事，旧名字容易让人以为它管的是
        # 预热爬坡步数。语义/默认值不变，纯改名。外部配置面（abfe_config.json
        # 的 "max_bias_warmup_steps" 键、exp029_protocol.py 的协议 schema、
        # abfe_pipeline.py 里 kwargs.get("max_bias_warmup_steps", ...) 的查找
        # 键名）不跟着改——那些是已经落盘/已经在用的外部契约，只有这个函数
        # 自己的 Python 形参改名。
        max_bias_learning_steps: int = 500000,
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
        # [Candidate-first, Validate-or-Learn v1] Best-effort acceptance has
        # no place in a production protocol: an unvalidated f_k may not enter
        # production merely because the calibration budget was exhausted.
        # Default flipped True->False: EXP-030 already passed False
        # explicitly; this flip makes that the default for every production
        # caller too. Only smoke/debug entrypoints should opt back in
        # explicitly. The fallback code path itself is left in place.
        allow_best_effort_warmup: bool = False,
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

        mbar_calibration_reserved_steps: 从 max_bias_learning_steps 总预算里为
            fixed-H overlap 全通过后的 MBAR 校准 f_k 单独预留的步数（burn-in +
            冻结验证）。SGD learning/freeze_burn_in/validating 阶段只能用
            max_bias_learning_steps - mbar_calibration_reserved_steps；这样即使
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
                            current_coion_identity=self.coion_identity,
                            stage_type=stage_type,
                            current_sampling_score_sha256=self.sampling_score_sha256,
                            current_stage_protocol_key=getattr(self, "stage_protocol_key", None),
                        )
                        # 🔑 这 10 个门原先内联在这里（~110 行），现已抽成模块级纯函数
                        # _resume_cached_window_gate_status，逐门语义与阈值一字未改，
                        # 只是变得可以用 mock 的 convergence.json 单独测试（见
                        # test_resume_reuse_contracts.py）。下面那串逐门诊断打印仍然
                        # 读同名局部变量，因此完全保持原样。
                        lambdas_match = _gate["lambdas_match"]
                        version_match = _gate["version_match"]
                        bias_protocol_match = _gate["bias_protocol_match"]
                        com_restraint_version_match = _gate["com_restraint_version_match"]
                        cached_lse_tolerance = _gate["cached_lse_tolerance"]
                        lse_tolerance_match = _gate["lse_tolerance_match"]
                        lrc_version_match = _gate["lrc_version_match"]
                        vdw_nb_version_match = _gate["vdw_nb_version_match"]
                        repair_policy_match = _gate["repair_policy_match"]
                        coion_identity_match = _gate["coion_identity_match"]
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
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0 and not com_restraint_version_match:
                            print(
                                f"  ⚠️ 窗口 {window_idx} 缓存的 ligand_com_restraint_protocol_version="
                                f"{cached_conv.get('ligand_com_restraint_protocol_version')!r}"
                                f"（期望 {LIGAND_COM_RESTRAINT_PROTOCOL_VERSION}）：v1 的 Group 5 "
                                "COM 限制力在 CUDA 上产生永久激活、跨边界跳变的定向拖拽，该窗口"
                                "轨迹不满足 MBAR 的平衡采样前提，视为无效缓存，将重新采样该窗口。"
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
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0 and not vdw_nb_version_match:
                            print(
                                f"  ⚠️ 窗口 {window_idx}（stage_type={stage_type!r}）缓存的 "
                                f"vdw_nonbonded_protocol_version="
                                f"{cached_conv.get('vdw_nonbonded_protocol_version')!r}（期望 "
                                f"{VDW_NONBONDED_PROTOCOL_VERSION}），vdW softcore cutoff/switching "
                                "协议已变更（MEM-00h：1.2nm+switch → 1.0nm 无switch），"
                                "视为无效缓存，将重新采样该窗口。"
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
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0 and not coion_identity_match:
                            print(
                                f"  ⚠️ 窗口 {window_idx} convergence.json 的 co-ion runtime identity="
                                f"{cached_conv.get('coion_identity')!r} 与当前"
                                f" {self.coion_identity!r} 不一致或缺失，拒绝复用能量缓存，"
                                "将重新采样该窗口。"
                            )
                        elif _gate["shape_ok"] and (not _gate["segment_metadata_match"] or not _gate["stage_protocol_match"]):
                            print(f"  ⚠️ 窗口 {window_idx} 缓存不能复用: {_gate['reason']}；将重新采样该窗口。")
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
                coion_identity=self.coion_identity,
            )
            attempt_checkpoint_restore = bool(
                resume
                and _peek_ibs_bias_status(ibs_state_file) == "calibrated_pending_validation"
                and _main_window_checkpoint_is_usable(
                    self.checkpoint_dir, stage_type, window_idx, expected_main_manifest
                )
            )

            integrator = LangevinMiddleIntegrator(self.temperature, 2.0 / unit.picosecond, 0.002 * unit.picosecond)
            window_integrator_seed = self._seed_for(
                "sampling", stage_type, window_idx, "integrator"
            )
            if window_integrator_seed is not None:
                integrator.setRandomNumberSeed(window_integrator_seed)
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

            # 🔑 [2026-08-05 修，排序 bug] Boresch 必须在最小化**之前**就已经在
            # scale=1.0（生产强度），不能先无限制力自由最小化、再事后爬坡拽回来
            # ——旧顺序正是下面"阶段3"（已注释掉）存在的唯一理由，也是它仍会在
            # 某些窗口爬坡中途发散（Particle coordinate is NaN）的根因。
            if _has_valid_boresch_restraint(self.boresch):
                sim.context.setParameter("lambda_boresch_scale", 1.0)

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
                    # 🔑 [IBS_BIAS_PROTOCOL_VERSION=30] 续算的 checkpoint 已经跑过
                    # 最小化，残差力理应早就是全强度——同样按防御性重设原则显式
                    # 写回 1.0，不依赖 checkpoint 是否连带保存了这个新参数。
                    if ibs_wrap.residual_enabled:
                        sim.context.setParameter(f"{ibs_wrap.prefix}_s_residual", 1.0)
                else:
                    print(f"  ⚠️ 窗口 {window_idx} 主窗口 checkpoint 加载失败，回退到完整重建流程。")

            # ---------- 最小化 ----------
            if not restored_from_window_checkpoint:
                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=30] 2026-08-25 EXP-030 window_0
                # 诊断：EM 时 bias_scale 仍是构造默认值 1.0，candidate 在冷启动
                # f_k=0 下就已经用全强度残差力做无温控 L-BFGS 最小化；配对
                # baseline/candidate 密度探针显示同一起点逐点一致，candidate 之后
                # 局部环境原子数雪崩、baseline 全程稳定（不能只归咎于弱化的
                # softcore vdW）。原则同 2026-08-05 Boresch 排序修复：该在最小化
                # 前定好的状态不能等最小化后再纠正。这里只关残差项，不像
                # bias_scale=0 那样连 baseline 也在正常使用的物理 softcore-state
                # 混合力一起关掉。
                if ibs_wrap.residual_enabled:
                    sim.context.setParameter(f"{ibs_wrap.prefix}_s_residual", 0.0)
                print(f"\n  [阶段1] 开始能量最小化...")
                sim.minimizeEnergy(maxIterations=20000)
                print(f"  ✓ 最小化完成")
                initial_velocity_seed = self._seed_for(
                    "sampling", stage_type, window_idx, "velocity"
                )
                if initial_velocities is not None:
                    # EXP-030 paired arms consume the exact velocity array
                    # from their shared, hash-frozen source checkpoint.
                    sim.context.setVelocities(initial_velocities)
                elif initial_velocity_seed is not None:
                    sim.context.setVelocitiesToTemperature(
                        self.temperature, initial_velocity_seed
                    )

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
                print(f"\n[阶段2] 测试性步进（Boresch 全程 scale=1.0）...")
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
                    # 🔑 [2026-08-05 注释掉，排序 bug 修复的一部分] 不再把 scale
                    # 缩到 1%——它现在已经在 setPositions 之后、最小化之前被设成
                    # 1.0（生产强度），全程保持，不需要在这里往下调。
                    # sim.context.setParameter("lambda_boresch_scale", 0.01)  # 1%
                    # print(f"  🔧 Boresch 力常数缩放至 1%")

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
    
                # 🔑 [2026-08-05 整段注释掉，排序 bug 修复] 原"阶段3 Boresch 安全
                # 爬坡"存在的唯一理由是补偿上面已经修掉的排序 bug：旧代码先在
                # scale=0 下自由最小化（配体漂离 committed r0 达 ~0.2nm），再靠
                # 这段 16 级自定义阶梯把它硬拽回限制力平衡点——拽的过程本身在
                # 某些窗口（如这次的窗口 5）会在 sim.minimizeEnergy() 内部就已经
                # 发散，抛出 Particle coordinate is NaN，而这段爬坡自己的能量/力
                # 检查全部是"minimizeEnergy 返回之后"才做，根本挡不住这一种失败
                # 模式。现在 Boresch 从最小化第一步起就已经在 scale=1.0，最小化
                # 收敛点本身就满足限制力平衡，不再有"漂开再拽回"这回事，这整段
                # 机制随之失去存在理由。整段注释掉而不是删除，留作历史参考；
                # 不要因为将来某个窗口在满强度最小化下不稳定，就把这段爬坡重新
                # 启用——那只是把同一个问题从"爬坡中途"挪回"最小化中途"，该查
                # 的是那个窗口本身的起始构象/λ 组合，不是重新引入分级 ramp。
                # ---------- Boresch 安全爬坡 ----------
                # ================================================================
                # Boresch 安全爬坡：自定义阶梯，逐个恢复力强度
                # ================================================================
                # if _has_valid_boresch_restraint(self.boresch):
                #     print(f"\n[阶段3] Boresch 安全爬坡（自定义阶梯）...")
                #
                #     # 自定义阶梯序列：低强度区采用更细分辨率，避免在高内应力底座上一步踩空。
                #     custom_scales = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
                #     dt_ramp = 0.001                  # 爬坡期间极小步长 (ps)
                #
                #     # 保存原始步长，并设置为爬坡步长
                #     original_dt = sim.integrator.getStepSize()
                #     sim.integrator.setStepSize(dt_ramp * unit.picoseconds)
                #     print(f"  → 爬坡使用步长 {dt_ramp} ps，低强度区采用更细台阶")
                #
                #     # 确保从当前 scale 开始（例如之前测试步进时设置的 0.01）
                #     try:
                #         current_scale = sim.context.getParameter("lambda_boresch_scale")
                #     except Exception:
                #         current_scale = 0.01
                #     print(f"  → 起始 Boresch scale = {current_scale:.3f}")
                #     prev_energy = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                #
                #     ramp_success = True
                #     for target_scale in custom_scales:
                #         # 只在目标大于当前值时才爬升（避免回退）
                #         if target_scale <= current_scale:
                #             continue
                #
                #         sim.context.setParameter("lambda_boresch_scale", float(target_scale))
                #         n_steps_per_level = 1500 if target_scale <= 0.10 else (1000 if target_scale <= 0.30 else 500)
                #         print(f"  🔹 设置 Boresch scale = {target_scale:.2f}，松弛 {n_steps_per_level} 步...", end="", flush=True)
                #         sim.minimizeEnergy(maxIterations=200, tolerance=20.0)
                #         for _ in range(max(1, n_steps_per_level // 100)):
                #             sim.step(100)
                #
                #         # 检查能量与受力
                #         state = sim.context.getState(getEnergy=True, getForces=True)
                #         e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                #         forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
                #         max_f = np.max(np.linalg.norm(forces, axis=1))
                #         delta_e = e - prev_energy
                #
                #         if (not np.isfinite(e)) or max_f > 50000 or (abs(delta_e) > 5e5 and max_f > 15000):
                #             print(f"\n  🚨 Boresch 爬坡在 scale={target_scale:.2f} 处失败！")
                #             print(f"    当前势能 = {e:.1f} kJ/mol，ΔE = {delta_e:.1f} kJ/mol，最大力 = {max_f:.1f} kJ/(mol·nm)")
                #             if debug_mode:
                #                 diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_Boresch爬坡失败_scale{target_scale:.2f}")
                #             ramp_success = False
                #             break
                #         else:
                #             print(f" 势能 = {e:.2f} kJ/mol，ΔE = {delta_e:.2f} kJ/mol，最大力 = {max_f:.2f} kJ/(mol·nm)")
                #             # 记录新的当前 scale
                #             current_scale = target_scale
                #             prev_energy = e
                #
                #     # 恢复原始步长
                #     sim.integrator.setStepSize(original_dt)
                #
                #     if not ramp_success:
                #         raise RuntimeError(
                #             f"窗口 {window_idx} Boresch 爬坡失败（scale={target_scale:.2f}），系统可能已崩溃。"
                #             "请检查锚点几何、力常数或初始结构。"
                #         )
                #
                #     print(f"  ✅ Boresch 爬坡成功完成 (scale 已恢复至 1.0)")
                # else:
                #     print(f"\n[阶段3] 无 Boresch 限制力，跳过爬坡。")

            # ================================================================
            # 热化完成（Boresch 全程 scale=1.0，阶段3 爬坡已注释掉不再需要）：
            # 立刻进行一次非侵入式力分解，供对比分析
            # ================================================================
            if debug_mode and self.boresch:
                diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_热化完成")

            # ---------- 初始化采样器 ----------
            # ibs_state_file 已经在窗口开头（checkpoint restore 判断之前）提前
            # 拼接过，这里不再重复。
            sampler = IBSSampler(sim.context, len(lc_win), self.temperature, self.prefix, ibs_wrapper=ibs_wrap)
            sampler.sampling_score_sha256 = self.sampling_score_sha256
            sampler.residual_plugin_identity = self.residual_plugin_identity
            sampler.residual_em_policy = self.residual_em_policy
            sampler.residual_feature_name = self.residual_feature_name
            # 🔑 [non_mutating_v1] stamp the current run's policy so save_ibs_state
            # records it (old-policy state files carry None → detected on load).
            sampler.sampling_repair_policy = repair_policy

            # 🔑 核心修复：断点续传状态检测
            is_resumed_ibs = False
            if resume and os.path.exists(ibs_state_file):
                is_resumed_ibs = sampler.load_ibs_state(
                    ibs_state_file, lc_win, lv_win, stage_type=stage_type
                )

            # 🔑 [终态硬停止] load_ibs_state 已经完成了 n_states/prefix/协议版本/
            # λ 内容的严格匹配（不是廉价 peek），确认这份状态真的属于当前窗口后，
            # 若上一轮已经把它判定为 calibrated_validation_failed（冻结验证累计
            # 预算用到最后一档仍未通过），必须立刻硬停止——不做任何 SGD/续验
            # 尝试，也不落入下面"is_resumed_ibs and not sampler.bias_converged"
            # 分支被当成普通未收敛热启动重新学习。调用方（abfe_pipeline.py）看到
            # 这个 terminal=True 的异常应该直接向上传播，不再自动重试。
            # [Candidate-first, Validate-or-Learn v1] 唯一在这段区间内做的改动：
            # 新协议下 VALIDATE 耗尽预算后的终态失败写 "failed"（不再写
            # calibrated_validation_failed），必须在这里同样被识别为终态，
            # 否则会被下面 `elif is_resumed_ibs and not sampler.bias_converged`
            # 分支误当成普通未收敛热启动重新学习。除这一处 in (...) 之外，
            # 10586-10689 区间的其余分支/语义本次刻意不做任何改动。
            if is_resumed_ibs and sampler.bias_status in (
                "calibrated_validation_failed", "failed",
            ):
                raise IBSFrozenCalibrationValidationError(
                    f"窗口 {window_idx} 的冻结校准验证此前已判定为终态失败"
                    f"（bias_status={sampler.bias_status!r}）：VALIDATE 累计预算已耗尽仍未能"
                    "通过独立验证——不再自动续验、不延长预算、不回退 learning。需要人工检查"
                    "（构象弛豫确实很慢，或偏置表达式仍有问题），人工确认后需手动清空该窗口的 "
                    "IBS 状态文件才能重新开始。",
                    diagnostics={
                        "window_index": int(window_idx),
                        "stage_type": stage_type,
                        "bias_status": sampler.bias_status,
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

            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=30] 到这里为止（这一整段仍在
            # `not restored_from_window_checkpoint` 内，即本窗口这次确实跑过
            # 最小化）四个分支都可能已经把 bias_scale 直接设成 1.0 并跳过下面
            # 的"[偏置预热]"爬坡块（skip_warmup_entirely / calibrated_pending_
            # validation / is_resumed_ibs-but-not-converged 三种），或者完全没碰
            # bias_scale、交给下面的爬坡块处理——无论走了哪一条，最小化阶段临时
            # 关闭的残差项都必须在这里统一恢复到 1.0，不能只在爬坡块里恢复
            # （否则前三个分支会让残差力从这个窗口开始永久关闭）。
            if ibs_wrap.residual_enabled:
                sim.context.setParameter(f"{ibs_wrap.prefix}_s_residual", 1.0)

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
                    sampler.seed_source = "pilot"
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
                        sampler.seed_source = "bootstrap"
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
            # 后的 MBAR 校准验证共用的同一个计数器。如果 SGD 把 max_bias_learning_steps
            # 全部烧光才退出循环，MBAR 校准验证的 while 条件一次都不会执行，
            # calibration_converged 必然是 False——不是校准本身失败，是预算已经
            # 没有了。因此这里把总预算拆成两块：SGD 只能用
            # sgd_step_budget = max_bias_learning_steps - mbar_calibration_reserved_steps，
            # 剩下的 mbar_calibration_reserved_steps 留给 MBAR 校准（下面校准循环
            # 用独立的 calibration_steps_used 计数，不再检查 steps_at_full_bias 是否
            # 撞到 max_bias_learning_steps），保证 fixed-H overlap 全通过时校准验证
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
                int(max_bias_learning_steps)
                - int(effective_mbar_calibration_reserved_steps),
                int(frozen_burn_in_steps) + check_chunk,
            )
            # Frozen burn-in/validation is a distinct read-only phase.  Give it
            # its own reserve instead of stealing that reserve from learning;
            # otherwise update 50 can never be frozen and validated.
            full_bias_step_budget = (
                int(max_bias_learning_steps)
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
            # [Candidate-first, Validate-or-Learn v1] validate_direct_retry_pending
            # limits the damped+pairwise-capped VALIDATE-failure retry (see the
            # gate-failure router below) to exactly one direct re-validation per
            # freeze cycle -- a second consecutive gap failure falls back to
            # bounded-occupancy LEARN instead of chaining retries indefinitely.
            # ever_completed_a_validate_attempt/last_failure_reason distinguish a
            # real, completed local-MBAR gate failure (evidence against this f_k,
            # -> terminal "failed" on exhaustion) from "never got that far"
            # (still eligible for another resume, -> "unconverged").
            validate_direct_retry_pending = False
            ever_completed_a_validate_attempt = False
            last_failure_reason = None
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

            # 🔑 [性能计时] warmup/learning 控制面没有 guard、也没有周期性
            # checkpoint（见下方循环体），桶比生产循环少；同样默认常开，循环
            # 结束时打印一次汇总，作为"这一段到底花在积分/CV-probe/权重更新
            # 上多少时间"的实测证据。
            warmup_timers: Dict[str, float] = {}
            while steps_at_full_bias < full_bias_step_budget:
                # learning 阶段的更新次数上限只约束 learning；freeze_burn_in/
                # validating 只受总步数安全帽 full_bias_step_budget 约束。反复未
                # 通过 loose-gate 把更新次数烧到上限时退出循环，由下方"预算耗尽即
                # 接受当前 f_k"分支放行进生产（见 IBS_BIAS_PROTOCOL_VERSION=29）。
                if mode == "learning" and bias_update_count >= int(max_bias_updates):
                    break
                with _timed(warmup_timers, "integration_s"):
                    sim.step(check_chunk)
                steps_at_full_bias += check_chunk
                with _timed(warmup_timers, "cv_probe_s"):
                    sampler.collect_energies()

                if mode == "learning":
                    # [Candidate-first, Validate-or-Learn v1] LEARN controls
                    # occupancy only -- no full-history TMBAR solve, no
                    # trusted/untrusted/self_consistent/legacy_quality branch
                    # selection (update_weights()/_solve_tmbar_and_recenter()
                    # are left fully defined but no longer called from this
                    # loop), no fixed min_bias_updates batch count. Sole
                    # update rule: bounded log-occupancy feedback
                    # (Δf_k=-η·kT·log(K·p_k)). Freeze the instant the raw
                    # residual drops to/below IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW
                    # -- VALIDATE (the unchanged real-Hamiltonian local-MBAR
                    # loose gate below) is the sole production entry proof,
                    # not a batch counter or a consecutive-pass streak.
                    f_old = np.array(
                        [sim.context.getParameter(f"{self.prefix}_f_{k}") for k in range(K)]
                    )
                    with _timed(warmup_timers, "weight_update_s"):
                        mean_p_batch = sampler.evaluate_frozen_batch_probability(
                            min_buffer_size=IBS_TMBAR_LEARNING_MINIBATCH_FRAMES
                        )
                    if mean_p_batch is None:
                        continue
                    f_new, learn_diag = sampler._bounded_log_occupancy_update(
                        f_old,
                        mean_p_batch,
                        severe_max_pairwise_step_kT=IBS_TMBAR_FALLBACK_SGD_PAIRWISE_STEP_KT,
                    )
                    for k in range(K):
                        sim.context.setParameter(f"{self.prefix}_f_{k}", float(f_new[k]))
                    sampler.f_history.append(f_new.copy())
                    sampler.last_update_diagnostics = {
                        "source": "learn_bounded_log_occupancy",
                        "weight_update": learn_diag,
                        "adjacent_delta_u_is_convergence_gate": False,
                    }
                    if sampler.seed_source != "learned":
                        sampler.seed_source = "learned"
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
                    residual_severity = float(
                        learn_diag.get("residual_severity", float("inf"))
                    )
                    if residual_severity <= IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW:
                        frozen_f_k_snapshot = [
                            float(sim.context.getParameter(f"{self.prefix}_f_{k}"))
                            for k in range(K)
                        ]
                        print(
                            f"    🧊 占据 raw_residual={residual_severity:.3f}≤"
                            f"{float(IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW):.2f} → 冻结跑 "
                            "local-MBAR Δf−ΔF 门（VALIDATE 是唯一生产入口证明，"
                            "不再要求固定 min_bias_updates 批数）"
                        )
                        mode = "freeze_burn_in"
                        have_frozen_once = True
                        updates_since_freeze = 0
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
                    sampler.save_ibs_state(
                        ibs_state_file, lc_win, lv_win, stage_type=stage_type
                    )
                # [Candidate-first, Validate-or-Learn v1] 攒满 IBS_LOCAL_MBAR_GATE_
                # SLIDING_BATCHES 批之前的读诊断早退：占据用跟 _diagnose_local_mbar_
                # situation 完全相同的公式/阈值判断是否已现塌陷迹象，若是则不等烧满
                # 全部 ~200 帧验证预算即可判定、退回 bounded-occupancy learning。
                # 纯路由决策——不设置 gate_ok/bias_converged，占据本身对生产入口门
                # 依旧只是诊断，不参与放行。
                if len(frozen_mbar_batches) < IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES:
                    _f_frozen_probe = np.asarray(
                        [
                            sim.context.getParameter(f"{self.prefix}_f_{k}")
                            for k in range(K)
                        ],
                        dtype=np.float64,
                    )
                    _u_probe = np.concatenate(
                        [
                            np.asarray(b["u_kn"], dtype=np.float64)
                            for b in frozen_mbar_batches
                        ],
                        axis=1,
                    )
                    _occ_probe = _softmax_occupancy_per_state(
                        _u_probe, _f_frozen_probe, sampler.kt
                    )
                    _target_p_probe = 1.0 / float(K) if K > 0 else 0.0
                    _sum_sq_probe = (
                        float(np.sum(np.square(_occ_probe))) if K > 0 else 0.0
                    )
                    _coverage_ess_probe = (
                        float(1.0 / _sum_sq_probe) if _sum_sq_probe > 0.0 else 0.0
                    )
                    _min_occ_probe = float(np.min(_occ_probe)) if K > 0 else 0.0
                    _early_exit_collapsed = bool(
                        K > 0
                        and (
                            _coverage_ess_probe
                            < IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_COVERAGE_ESS_FRACTION
                            * float(K)
                            or _min_occ_probe
                            < IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_MIN_FRACTION
                            * _target_p_probe
                        )
                    )
                    if _early_exit_collapsed:
                        print(
                            "    ⟳ VALIDATE 早退：第 "
                            f"{len(frozen_mbar_batches)}/{IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES} "
                            f"批已现占据塌陷迹象（coverage_ESS={_coverage_ess_probe:.2f}, "
                            f"min_occ={_min_occ_probe:.3f}），不等攒满全部验证预算即退回 "
                            "bounded-occupancy learning（纯路由决策，不参与 gate_ok）"
                        )
                        sampler.last_update_diagnostics = {
                            "source": "validate_early_exit_occupancy_collapse",
                            "coverage_ess": _coverage_ess_probe,
                            "min_occ": _min_occ_probe,
                            "adjacent_delta_u_is_convergence_gate": False,
                        }
                        last_failure_reason = "occupancy_collapse_early_exit"
                        mode = "learning"
                        validate_direct_retry_pending = False
                        updates_since_freeze = 0
                        frozen_mbar_batches = []
                        sampler._last_dominant_k = None
                        sampler.ema_mean_p = None
                        sampler.energy_buffer = []
                        validation_probability_sum = np.zeros(K, dtype=np.float64)
                        validation_sample_count = 0
                        validation_batch_count = 0
                        validation_steps_this_freeze = 0
                        continue
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
                # [Candidate-first, Validate-or-Learn v1] This is a real,
                # completed local-MBAR gate evaluation (pass or fail) --
                # counted once here regardless of outcome, distinct from the
                # cheap early-exit probe above which never reaches this line.
                ever_completed_a_validate_attempt = True
                sampler.validation_attempts = (
                    int(getattr(sampler, "validation_attempts", 0)) + 1
                )
                if gate_ok:
                    validation_pass_count = int(validation_batch_count)
                    bias_converged = True
                    last_failure_reason = None
                    print(
                        "    ✅ local-MBAR loose gate 通过："
                        f"max|Δf_k−ΔF^MBAR|={max_adjacent_gap_kJ_mol:.3f} kJ/mol < "
                        f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f} kJ/mol"
                        f"（{validation_sample_count} frames；abs_ess={_gate_abs_ess:.1f}、"
                        f"min_ess_ratio={_gate_min_ess_ratio} 仅诊断，不参与放行）；"
                        "冻结 f_k 进生产，最终绝对 ESS/误差/自由能交生产后 MBAR"
                    )
                    break
                # 未过（gap≥阈值，或 MBAR 不可解/NaN）。abs_ess/占据只在现场里打印，
                # 不参与判定——gate_ok 的定义完全不变。
                learning_to_validation_cycles += 1
                if gate_error is None:
                    print(
                        "    ⟳ local-MBAR loose gate 未过："
                        f"max|Δf_k−ΔF^MBAR|={max_adjacent_gap_kJ_mol:.3f} kJ/mol ≥ "
                        f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f}；"
                        "现场："
                        f"{_format_local_mbar_situation(gate_situation)}"
                    )
                else:
                    print(
                        f"    ⟳ local MBAR 暂不可解（{gate_error}）；不作为 f_k 的反面证据，"
                        "继续 bounded-occupancy learning。现场："
                        f"{_format_local_mbar_situation(gate_situation)}"
                    )
                sampler.last_update_diagnostics = {
                    "source": "failed_local_mbar_loose_gate",
                    "local_mbar_gate": gate_diag,
                    "adjacent_delta_u_is_convergence_gate": False,
                }
                # [Candidate-first, Validate-or-Learn v1] 3-way VALIDATE-failure
                # router. occupancy_collapsed/gate_error remain diagnostics-only
                # for gate_ok itself (unchanged above) -- they only choose the
                # recovery path here:
                #   (i)   occupancy_collapsed -> bounded-occupancy LEARN.
                #   (ii)  occupancy OK, gate_error is None, gap failed, no
                #         retry spent yet this freeze cycle -> ONE damped +
                #         pairwise-capped correction using this attempt's
                #         local-MBAR candidate, then re-VALIDATE directly
                #         (skip LEARN for this one retry).
                #   (iii) a second consecutive gap failure after that one
                #         retry, or gate_error is not None (local-MBAR
                #         unsolvable, not evidence against f_k) -> LEARN.
                _occ_collapsed_now = bool(gate_situation.get("occupancy_collapsed"))
                if _occ_collapsed_now:
                    last_failure_reason = "occupancy_collapse"
                    mode = "learning"
                    validate_direct_retry_pending = False
                elif gate_error is None and not validate_direct_retry_pending:
                    f_mbar_retry = np.asarray(gate_mbar.get("f"), dtype=np.float64)
                    f_damped, _damp_diag = sampler._damped_tmbar_absolute_update(
                        f_frozen_now, f_mbar_retry, damping=IBS_TMBAR_UPDATE_DAMPING
                    )
                    f_capped, _cap_diag = sampler._apply_pairwise_cap(
                        f_frozen_now,
                        f_damped,
                        IBS_MAX_APPLIED_PAIRWISE_STEP_KT,
                        sampler.kt,
                    )
                    for k in range(K):
                        sim.context.setParameter(f"{self.prefix}_f_{k}", float(f_capped[k]))
                    frozen_f_k_snapshot = f_capped.astype(float).tolist()
                    print(
                        "    🔧 occupancy 尚可但 local-MBAR gap 未过：对冻结 f_k 应用一次"
                        "阻尼+pairwise-capped 修正（用本次 local-MBAR 候选），直接重新"
                        "验证（跳过 learning，本轮仅此一次）"
                    )
                    last_failure_reason = "local_mbar_gap_exceeded_retry_applied"
                    validate_direct_retry_pending = True
                    mode = "freeze_burn_in"
                elif gate_error is None:
                    last_failure_reason = "local_mbar_gap_exceeded_after_retry"
                    mode = "learning"
                    validate_direct_retry_pending = False
                else:
                    last_failure_reason = "local_mbar_unsolvable"
                    mode = "learning"
                    validate_direct_retry_pending = False
                updates_since_freeze = 0
                freeze_burn_in_done = 0
                frozen_mbar_batches = []
                sampler._last_dominant_k = None
                sampler.ema_mean_p = None
                sampler.energy_buffer = []
                validation_probability_sum = np.zeros(K, dtype=np.float64)
                validation_sample_count = 0
                validation_batch_count = 0
                validation_steps_this_freeze = 0
                continue

            # 🔑 [性能计时] warmup/learning 循环结束，打印一次耗时拆分汇总——
            # 只是诊断输出，不落盘、不参与任何 resume/协议比较。
            print(
                "    ⏱️ [warmup 计时] "
                + ", ".join(f"{k}={v:.1f}s" for k, v in warmup_timers.items())
            )

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
            if not bias_converged and not warmup_only and allow_best_effort_warmup:
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
                "max_bias_learning_steps_safety_cap": int(full_bias_step_budget),
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
                        coion_identity=self.coion_identity,
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
                    coion_identity=self.coion_identity,
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
                    # steps_at_full_bias < max_bias_learning_steps——SGD 阶段已经把它的
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
                        sampler.save_ibs_state(
                            ibs_state_file, lc_win, lv_win, stage_type=stage_type
                        )
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
                sampler.last_failure_reason = None
                sampler.save_ibs_state(
                    ibs_state_file, lc_win, lv_win, stage_type=stage_type
                )
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
                    # [Candidate-first, Validate-or-Learn v1] Only writes
                    # "failed"/"unconverged" going forward (never the legacy
                    # calibrated_* values) -- see the calibration_pending
                    # branch above, which is legacy-only and left untouched.
                    # "failed" is reserved for a real, completed local-MBAR
                    # gate rejection (gate ran cleanly, gap stayed too big
                    # even after the one damped+capped retry): genuine
                    # evidence against this candidate. Everything else
                    # (never reached a completed gate check, occupancy
                    # collapsing, or local-MBAR numerically unsolvable) is
                    # "unconverged" -- worth another resume with more budget,
                    # not a terminal verdict.
                    if last_failure_reason in (
                        "local_mbar_gap_exceeded_retry_applied",
                        "local_mbar_gap_exceeded_after_retry",
                    ):
                        sampler.bias_status = "failed"
                    else:
                        sampler.bias_status = "unconverged"
                    sampler.frozen_f_k_pending = None
                    sampler.frozen_validation_cumulative_steps = 0
                sampler.last_failure_reason = last_failure_reason
                _atomic_write_json(failure_path, bias_warmup_diag)
                sampler.bias_converged = False
                sampler.save_ibs_state(
                    ibs_state_file, lc_win, lv_win, stage_type=stage_type
                )
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

            # 🔑 [2026-08-27，见 EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md 补验收]
            # 生产入口标记（production_entry_f_k）不再在这里赋值——这里是
            # checkpoint 续采判断/恢复**之前**，早于真正确定这次生产是否续算
            # 同一份冻结 f_k。真正落盘的赋值挪到下面 production_f_k_lock 断言
            # 通过之后（[阶段5] 打印之前），那才是"checkpoint 恢复与入口调整都
            # 已完成、第一步生产采样之前"的正确时刻；同时也在那里补上写入
            # ibs_state_file 的 save_ibs_state() 调用，否则这份标记只存在于
            # 内存里，续跑和下游冻结检查都读不到它。

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
                coion_identity=self.coion_identity,
            )
            expected_production_manifest["stage_protocol_key"] = getattr(self, "stage_protocol_key", None)
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
            production_sampling_states_path = os.path.join(
                self.output_dir, f"dual_window_{window_idx}_{stage_type}_sampling_states.npy"
            )
            production_residual_basis_path = os.path.join(
                self.output_dir, f"dual_window_{window_idx}_{stage_type}_residual_basis.npy"
            )
            prior_cumulative_production_steps = 0
            resumed_production_checkpoint = False
            prior_energy_history = None
            prior_bias_history = None
            prior_sampling_state_history = None
            prior_residual_basis_history = None
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
                            prior_segments = _validate_production_segments(
                                prior_production_convergence.get("production_segments"), _prior_e.shape[1]
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
            if resumed_production_checkpoint:
                try:
                    if self.sampling_score_sha256 is not None and (
                        prior_production_convergence.get(
                            "residual_sampling_protocol_version"
                        ) != IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
                    ):
                        raise ValueError(
                            "joint-score production ledger protocol is missing or old; "
                            "refusing to guess its axis orientation"
                        )
                    _prior_sampling_states = np.load(
                        production_sampling_states_path, allow_pickle=False
                    )
                    _prior_residual_basis = np.load(
                        production_residual_basis_path, allow_pickle=False
                    )
                    expected_frames = len(prior_energy_history)
                    if (
                        _prior_sampling_states.shape != (expected_frames, len(lc_win))
                        or _prior_residual_basis.shape != (expected_frames,)
                        or not np.all(np.isfinite(_prior_sampling_states))
                        or not np.all(np.isfinite(_prior_residual_basis))
                    ):
                        raise ValueError("joint-score production ledgers are invalid")
                    prior_sampling_state_history = [
                        _prior_sampling_states[i, :].copy()
                        for i in range(expected_frames)
                    ]
                    prior_residual_basis_history = [
                        float(value) for value in _prior_residual_basis
                    ]
                except Exception as exc:
                    print(f"  ⚠️ joint-score production ledgers unavailable ({exc}); full resample required")
                    resumed_production_checkpoint = False
            if not resumed_production_checkpoint:
                prior_cumulative_production_steps = 0
                prior_energy_history = None
                prior_sampling_state_history = None
                prior_residual_basis_history = None
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
                sampler.sampling_state_energy_history = list(prior_sampling_state_history)
                sampler.residual_basis_history = list(prior_residual_basis_history)
            else:
                sampler.energy_history = []
                sampler.bias_history = []
                sampler.base_energy_history = []
                sampler.sampling_state_energy_history = []
                sampler.residual_basis_history = []
            sampler.production_segment_starts = (
                [{key: seg[key] for key in ("start_frame", "reason", "session_id")}
                 for seg in prior_segments] if resumed_production_checkpoint else []
            )
            _start_production_segment(
                sampler, "cross_process_resume" if resumed_production_checkpoint else "fresh_or_rebuilt"
            )
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

            # 🔑 [2026-08-27，见 EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md
            # 补验收] 真正的生产入口标记：checkpoint 恢复/续采判断、安全检查、
            # f_k 与冻结 manifest 一致性断言全部通过之后，第一步生产采样之前。
            # load_ibs_state 可能已经从上一次 attempt 的 IBS JSON 里恢复出这份
            # 冻结 f_k 的入口标记——如果数值上就是这一次 production_f_k_lock，
            # 说明这是同一份冻结 f_k 的续采，必须原样保留旧标记，不能用"现在"
            # 重新读到的 Context 值覆盖"最初"的值；否则（None，或数值对不上、
            # 说明标记来自另一份已作废的校准）才在这里第一次写入，并立即调用
            # save_ibs_state() 落盘——不这样做的话这份标记只活在内存里，进程
            # 一退出、续跑重新构造 sampler 时就彻底丢了，下游冻结检查永远读到
            # None，静默退化成旧的单段比较。
            #
            # 🔑 [续跑边界补丁] marker 为 None 时不能一律当成"第一次进入生产"
            # 直接写现在的值：如果 resumed_production_checkpoint 为 True（已有
            # 之前 attempt 留下的、真正被用来驱动过采样的生产历史——energies/
            # bias/base.npy + 匹配的 checkpoint），那份历史最初进入生产时的
            # f_k 就是真的不知道（旧版本从没记过），现在写一个"现在"的值会把
            # 一段其实入口未知的旧历史悄悄升级成"生产入口标记验证通过"，这正是
            # 这个字段本该防止的事情——必须继续保持 None，让下游 reconcile
            # 走 degraded 路径。只有 resumed_production_checkpoint 为 False
            # （这次生产是从 0 步真正重新开始，没有沿用任何旧生产样本）时，
            # marker=None 才代表"这次的生产历史确实还没有入口标记"，可以在这
            # 里第一次写入。
            #
            # 判定逻辑本身在 _resolve_production_entry_marker()（纯函数，
            # 独立于 sampler/磁盘，见其 docstring）——这里只按返回值执行写/
            # 存/拒绝三种动作，保证真正跑生产用的分支判断和离线单元测试用的
            # 是同一份代码，而不是各自维护一份容易漂移的复制。
            _entry_marker_action, _entry_marker_value = _resolve_production_entry_marker(
                sampler.production_entry_f_k, production_f_k_lock, resumed_production_checkpoint
            )
            if _entry_marker_action == "refuse":
                raise RuntimeError(
                    f"窗口 {window_idx} 已恢复的生产入口标记与本次冻结 f_k 驱动的"
                    "生产 checkpoint 不一致——这份 checkpoint 的 manifest 已确认"
                    "生产历史属于当前冻结 f_k，但保存的入口标记却对不上，是真实"
                    "数据矛盾，不能用当前值静默覆盖。已阻止继续生产，请人工核查"
                    "这份 IBS 状态文件与生产 checkpoint 的一致性。"
                )
            elif _entry_marker_action == "set":
                sampler.production_entry_f_k = _entry_marker_value
                sampler.save_ibs_state(
                    ibs_state_file, lc_win, lv_win, stage_type=stage_type
                )
            # else "keep": sampler.production_entry_f_k 原样不动，不写盘。

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
            # 🔑 [性能计时] 按"积分/guard/CV-probe/ledger-IO"四档累加墙钟耗时，
            # 默认常开（perf_counter 单次开销可忽略）——用来把后续任何"是否
            # 值得继续优化"的判断建立在实测数字上，而不是猜测。跟余数补齐块
            # 共用同一份累加器，因为它是同一窗口生产采样的尾巴。
            loop_timers: Dict[str, float] = {}
            for up in range(n_updates):
                pos_backup = production_pos_backup
                with _timed(loop_timers, "integration_s"):
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

                with _timed(loop_timers, "guard_s"):
                    do_force_check = (up % force_check_interval == 0) or (up == n_updates - 1)
                    # 🔑 [性能修复] guard 在 do_force_check=True 时已经把 positions/box
                    # 一并取回；下面 collect_energies() 会复用这份数据，不再独立
                    # 重新查一次同一个 context、同一时刻的状态。
                    guard_positions = None
                    guard_box_vectors = None
                    # [性能修复：合并 guard 查询，PLAN 2026-08-26] 弱分支（大多数
                    # update）不需要 forces——fmax 只在强分支才非 None——只需要
                    # "总能量是否有限"这一个数。生产窗口系统（build_ibs_dual_system/
                    # build_shadow_coul_ibs_system，已逐一核实两者的力只落在
                    # Group 0-5，且 _build_window_system() 之后没有第三方再往
                    # win_sys 加力）里，e_base(groups={0,2,3,5}) + e_bias(groups=
                    # {1,4}) 恒等于不过滤的 getState(getEnergy=True) 总能量——不是
                    # 近似，是这套系统构造保证的精确恒等式。直接查这两组既够判
                    # disaster，又能把这两个数喂给下面 collect_energies()，省掉它
                    # 原本还要对同一瞬时状态重新查一遍的两次 getState()。
                    guard_e_base = None
                    guard_e_bias = None
                    if do_force_check:
                        state_n = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                        forces_n = state_n.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                        fmax = np.max(np.linalg.norm(forces_n, axis=1))
                        guard_positions = state_n.getPositions(asNumpy=True)
                        guard_box_vectors = state_n.getPeriodicBoxVectors()
                        e_total_n = state_n.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    else:
                        # 各自包一层 try/except，语义对齐 collect_energies() 内部
                        # 原来对这两组的处理——查询失败不让整个 update 崩，标 NaN
                        # 交给下面统一的 disaster 判据处理。
                        try:
                            guard_e_base = (
                                sim.context.getState(getEnergy=True, groups={0, 2, 3, 5})
                                .getPotentialEnergy()
                                .value_in_unit(unit.kilojoule_per_mole)
                            )
                        except Exception:
                            guard_e_base = float("nan")
                        try:
                            guard_e_bias = (
                                sim.context.getState(getEnergy=True, groups={1, 4})
                                .getPotentialEnergy()
                                .value_in_unit(unit.kilojoule_per_mole)
                            )
                        except Exception:
                            guard_e_bias = float("nan")
                        e_total_n = guard_e_base + guard_e_bias
                        fmax = None

                    energy_bad = not np.isfinite(e_total_n)
                    if energy_bad and not do_force_check:
                        # 能量已经异常，不能再等下一次排期检查——立刻补查力和坐标。
                        # 升级到强分支后立刻 continue（见下面的 disaster 判据），
                        # guard_e_base/guard_e_bias 不会被用到，不需要重新算。
                        state_n = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                        forces_n = state_n.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                        fmax = np.max(np.linalg.norm(forces_n, axis=1))
                        guard_positions = state_n.getPositions(asNumpy=True)
                        guard_box_vectors = state_n.getPeriodicBoxVectors()
                        do_force_check = True

                    if energy_bad or (fmax is not None and ((not np.isfinite(fmax)) or fmax > 10000.0)):
                        self._production_disaster_rollback(
                            sim, pos_backup, sampler, production_history_backup_len,
                            stage_type, window_idx, attempt=up + 1,
                            e_total_n=e_total_n, fmax=fmax, win_sys=win_sys, debug_mode=debug_mode,
                            label_prefix="", progress_note=f"update={up}/{n_updates}, ",
                            diagnose_prefix=f"窗口{window_idx}_回退_update{up}",
                        )
                        continue

                    if do_force_check:
                        # 复用同一次 getState 里已经取回的坐标（guard_positions），
                        # 不再单独发一次 getPositions 请求
                        production_pos_backup = guard_positions
                        # 坐标备份与 history 长度必须成对更新，否则回退会截到错的分叉点。
                        production_history_backup_len = _production_history_lengths(sampler)

                # 🔑 生产阶段 f_k 已冻结（sampler.bias_converged=True，见上面严格收敛
                # 判据），不再调用 update_weights()——否则同一窗口的样本会来自随时间
                # 漂移的多个偏置，而不是 MBAR 假设的单一固定采样分布。只采样、记录
                # 能量；定期清空 energy_buffer 避免其无谓增长（update_weights() 之前
                # 顺带做的这件事，现在没人做了）。
                with _timed(loop_timers, "cv_probe_s"):
                    e = sampler.collect_energies(
                        reuse_positions=guard_positions,
                        reuse_box_vectors=guard_box_vectors,
                        reuse_e_base=guard_e_base,
                        reuse_e_bias=guard_e_bias,
                    )
                if len(sampler.energy_buffer) >= 10:
                    sampler.energy_buffer = []
                if (up + 1) % 100 == 0:
                    with _timed(loop_timers, "ledger_io_s"):
                        self._enqueue_window_snapshot(window_idx, stage_type, sampler)
                        # 🔑 [production checkpoint 续采] 每 100 个 update 覆盖式落盘一次
                        # 生产 checkpoint（坐标/速度/积分器 RNG）+ 累计步数——应对 HPC
                        # 作业被抢占/撞墙时限杀掉的情况：不这样做的话，一次很长的
                        # production 预算在被杀时会丢失全部进展，下次只能整窗从零重采
                        # （这正是这次要修的问题本身）。
                        # 🔑 [性能修复：重复落盘] 能量/偏置/基准能量/采样态/residual
                        # 五份数组已经在上面 _enqueue_window_snapshot() 里原子写过一次
                        # 了——production_energies_path 等变量本来就是跟
                        # _enqueue_window_snapshot 写的 dual_window_*.npy 完全相同的
                        # 路径，此前这里又把同一份 sampler.*_history 重新序列化一遍
                        # 写到同一批文件，是逐字节相同、纯浪费的第二次落盘，故删除。
                        # convergence.json 只合并更新 cumulative_production_steps 字段，
                        # 不覆盖其它字段——这份文件此刻还不是"最终结果"，真正完整的
                        # 版本在窗口正常结束时于下面"保存能量"区块整份重写。
                        _atomic_save_openmm_checkpoint(sim, production_ckpt_path)
                        _atomic_write_json(production_manifest_path, expected_production_manifest)
                        _periodic_conv = {}
                        if os.path.exists(production_conv_path):
                            try:
                                with open(production_conv_path, "r", encoding="utf-8") as pf:
                                    _periodic_conv = json.load(pf)
                            except Exception:
                                _periodic_conv = {}
                        _periodic_conv["stage_protocol_key"] = getattr(self, "stage_protocol_key", None)
                        _periodic_conv["production_segment_protocol_version"] = PRODUCTION_SEGMENT_PROTOCOL_VERSION
                        _periodic_conv["production_segments"] = _production_segments_snapshot(sampler)
                        _periodic_conv["cumulative_production_steps"] = (
                            int(prior_cumulative_production_steps) + (up + 1) * int(steps_per_update)
                        )
                        _periodic_conv["window_data_protocol_version"] = (
                            IBS_WINDOW_DATA_PROTOCOL_VERSION
                        )
                        if stage_type == "vdw":
                            _periodic_conv["vdw_nonbonded_protocol_version"] = (
                                VDW_NONBONDED_PROTOCOL_VERSION
                            )
                        _periodic_conv["window_data"] = _window_data_metadata(
                            production_energies_path,
                            production_bias_path,
                            production_base_path,
                        )
                        if self.sampling_score_sha256 is not None:
                            _periodic_conv["residual_sampling_protocol_version"] = (
                                IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
                            )
                            _periodic_conv["joint_score_window_data"] = {
                                "protocol_version": IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION,
                                "sampling_states": {
                                    "shape": [
                                        len(sampler.sampling_state_energy_history),
                                        len(lc_win),
                                    ],
                                    "dtype": "float64",
                                    "sha256": _sha256_file(
                                        production_sampling_states_path
                                    ),
                                },
                                "residual_basis": {
                                    "shape": [len(sampler.residual_basis_history)],
                                    "dtype": "float64",
                                    "sha256": _sha256_file(
                                        production_residual_basis_path
                                    ),
                                },
                                "sampling_state_definition": (
                                    "softcore_U_k_plus_A_k_times_residual_basis_minus_offset"
                                ),
                                "physical_target_excludes_residual": True,
                            }
                        _periodic_conv["loop_timing_s"] = dict(loop_timers)
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
                        production_segments=_production_segments_snapshot(sampler),
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
                with _timed(loop_timers, "integration_s"):
                    try:
                        sim.step(remaining_steps)
                    except Exception as e:
                        print(f"\n  🚨 余数补齐阶段崩溃 ({remaining_steps} 步)")
                        if debug_mode:
                            diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_余数补齐崩溃")
                            diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_余数补齐崩溃")
                        raise

                with _timed(loop_timers, "guard_s"):
                    # 🔑 [性能修复] 补上 getPositions=True，跟下面 else 分支复用同一次
                    # State 里的坐标/box，不再像之前那样再单独发一次 getState(getPositions=True)。
                    state_n = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                    e_total_n = state_n.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    forces_n = state_n.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                    fmax = np.max(np.linalg.norm(forces_n, axis=1))
                    guard_positions = state_n.getPositions(asNumpy=True)
                    guard_box_vectors = state_n.getPeriodicBoxVectors()
                    if (not np.isfinite(e_total_n)) or (not np.isfinite(fmax)) or fmax > 10000.0:
                        self._production_disaster_rollback(
                            sim, pos_backup, sampler, production_history_backup_len,
                            stage_type, window_idx, attempt=n_updates + 1,
                            e_total_n=e_total_n, fmax=fmax, win_sys=win_sys, debug_mode=debug_mode,
                            label_prefix="余数补齐", progress_note="",
                            diagnose_prefix=f"窗口{window_idx}_余数补齐回退",
                        )
                        remainder_disaster = True
                    else:
                        production_pos_backup = guard_positions
                        production_history_backup_len = _production_history_lengths(sampler)
                        remainder_disaster = False

                if not remainder_disaster:
                    # 生产阶段 f_k 已冻结，不再调用 update_weights()，见上方主循环同类注释。
                    with _timed(loop_timers, "cv_probe_s"):
                        e = sampler.collect_energies(
                            reuse_positions=guard_positions,
                            reuse_box_vectors=guard_box_vectors,
                        )
                    if len(sampler.energy_buffer) >= 10:
                        sampler.energy_buffer = []
                    # 🔑 [性能修复：重复落盘] 不在这里再调 _enqueue_window_snapshot()——
                    # 紧接着下面无条件执行的"保存能量"区块几行之后会把同一批
                    # sampler.*_history 原样重写到完全相同的 dual_window_*.npy 路径，
                    # 这里再写一次纯属逐字节重复。
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
            sampling_states_arr = np.asarray(
                sampler.sampling_state_energy_history, dtype=np.float64
            )
            _atomic_save_npy(
                production_sampling_states_path,
                sampling_states_arr,
            )
            _atomic_save_npy(
                production_residual_basis_path,
                np.asarray(sampler.residual_basis_history, dtype=np.float64),
            )
            window_data_metadata = _window_data_metadata(
                production_energies_path,
                production_bias_path,
                production_base_path,
            )
            convergence = {
                "production_segment_protocol_version": PRODUCTION_SEGMENT_PROTOCOL_VERSION,
                "production_segments": _production_segments_snapshot(sampler),
                "stage_protocol_key": getattr(self, "stage_protocol_key", None),
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
                # 🔑 [ligand_com_restraint_protocol_version] Group 5 配体 COM 限制力
                # 的协议版本。v1 的非周期绝对锚点实现在 CUDA 上产生定向拖拽，那种
                # 轨迹不满足 MBAR 的平衡采样前提；stage 级指纹拦不住**逐窗口**的
                # 能量数组复用（它发生在建 Context 之前），所以必须逐窗口落盘并在
                # _resume_cached_window_gate_status 里校验。
                "ligand_com_restraint_protocol_version": (
                    LIGAND_COM_RESTRAINT_PROTOCOL_VERSION
                ),
                "lse_log_residual_tolerance": float(lse_log_residual_tolerance),
                # 🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION] LJ 长程尾项修正公式版本
                # （尽管名字里写着 traditional，v2 起这个常量同时覆盖 ACE/dual_lambda
                # 路径和传统 Beutler REMD 路径共用的 switching+softcore-aware LRC
                # 积分——见该常量定义处）。旧版本（v1，只补 cutoff 外的 r^-6、忽略
                # switching 区间）下的能量文件里，target_energies 里叠加的 LRC 数值
                # 跟当前公式不是同一回事，resume / 窗口产物复用逻辑必须校验这个
                # 字段，不能只看 λ/WCA/IBS 偏置协议是否匹配。
                "lj_tail_lrc_protocol_version": TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
                # 🔑 [vdw_nonbonded_protocol_version，MEM-00h] softcore cutoff/
                # switching 协议版本（1.2nm+switch → 1.0nm 无switch）。只在
                # stage_type=="vdw" 时写真实版本号——Stage 1 charging 不构造软核
                # CV，这个字段对它没有意义，写 None 也不会被 _resume_cached_
                # window_gate_status 拿来判它无效（该门对 stage_type!="vdw" 恒为
                # True，见该函数定义处）。
                "vdw_nonbonded_protocol_version": (
                    VDW_NONBONDED_PROTOCOL_VERSION if stage_type == "vdw" else None
                ),
                # 🔑 [non_mutating_v1] 采样修复策略。旧的变异策略可能在采样中途用
                # 累计 ΔF 就地覆盖过 f_k（不同参考系）；resume / 窗口产物复用必须校
                # 验这个字段，绝不能把旧策略缓存当成非变异策略的有效数据复用。
                "sampling_repair_policy": repair_policy,
                "sampling_score_sha256": self.sampling_score_sha256,
                "residual_sampling_protocol_version": (
                    IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
                    if self.sampling_score_sha256 is not None
                    else None
                ),
                # [B5] Completed-window energy reuse happens before checkpoint
                # manifests are consulted, so bind the energy triplet itself to
                # the frozen runtime co-ion identity.
                "coion_identity": self.coion_identity,
                "n_steps_per_window_default": int(n_steps_per_window),
                "n_steps_per_window_effective": int(effective_n_steps_per_window),
                "n_updates": int(len(sampler.f_history)),
                # [P1-19] 命名修正：sampler.f_history 存的是 f_new（物理 F_k，
                # 单位 kJ/mol，见 _solve_tmbar_and_recenter 的说明），不是无量纲
                # 的 kT 归一化值——Context 里 `{prefix}_f_{k}` 全局参数与
                # cv_k_int/cv_k_rest 同单位相减（见 IBSBiasWrapper 的 CV 表达式
                # 构造），本来就是 kJ/mol。旧字段名 `free_energy_history_kT`
                # 只是记错了单位，从未被任何 resume/复用逻辑读取，改名安全。
                "free_energy_history_kJ_mol": [
                    np.asarray(f_k, dtype=float).tolist()
                    for f_k in sampler.f_history
                ],
                "n_energy_samples": int(e_save.shape[1]),
                "energy_query_diagnostics": sampler.energy_query_diagnostics(),
                # 🔑 [性能计时，纯诊断字段] 整窗累计的"积分/guard/CV-probe/
                # ledger-IO"墙钟耗时（秒）。不参与任何 resume/协议指纹比较——
                # resume 逻辑只按字段名 .get(...) 读取自己关心的键
                # （cumulative_production_steps 等），新增诊断字段不会破坏它。
                "loop_timing_s": dict(loop_timers),
                "window_data_protocol_version": IBS_WINDOW_DATA_PROTOCOL_VERSION,
                "window_data": window_data_metadata,
                "bias_warmup": bias_warmup_diag,
                # 🔑 [EARLY_STOP_PROTOCOL_VERSION] 见 run_all_windows 参数说明和该
                # 常量定义处。actual_production_steps 是这个窗口真正跑了多少步
                # （早停时 < n_steps_per_window_effective，未早停/未启用时等于它）；
                # resume 时只有 early_stop_triggered=True 且协议版本/enable_early_stop/
                # 目标步数都与当前调用一致才允许复用这份缓存，见上面 resume 分支
                "joint_score_window_data": {
                    "sampling_states": {
                        "shape": list(sampling_states_arr.shape),
                        "dtype": str(sampling_states_arr.dtype),
                        "sha256": _sha256_file(production_sampling_states_path),
                    },
                    "protocol_version": IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION,
                    "residual_basis": {
                        "shape": [len(sampler.residual_basis_history)],
                        "dtype": "float64",
                        "sha256": _sha256_file(production_residual_basis_path),
                    },
                    "sampling_state_definition": "softcore_U_k_plus_A_k_times_residual_basis_minus_offset",
                    "physical_target_excludes_residual": True,
                },
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
            if self.sampling_score_sha256 is not None:
                convergence["residual_sampling"] = {
                    "feature": getattr(
                        self,
                        "residual_feature_name",
                        "Outer-Lambda Local Residual for IBS",
                    ),
                    "em_policy": getattr(
                        self, "residual_em_policy", "no_residual_twin"
                    ),
                    "plugin_identity": getattr(
                        self, "residual_plugin_identity", None
                    ),
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
            # 🔑 [2026-08-27，见 EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md
            # 补验收] 生产结束时必须再存一次 IBS JSON：这是下游冻结检查唯一会
            # 读取的文件（见 exp030_window_state_machine.py 的 state_path），
            # 之前只存 OpenMM checkpoint + manifest，IBS JSON 从生产开始前
            # （12461 附近那次 save）起就再没刷新过，导致 final_state["f_k"]
            # 其实是生产开始前的值，不是生产结束时的值。save_ibs_state()
            # 内部从当前（仍未 del 的）sim.context 重新读 f_k 写入 "f_k" 字段，
            # 同时把上面已经落盘、生产期间从未再赋值过的 production_entry_f_k
            # 原样带出——JSON 里因此同时有"生产开始时"和"生产结束时"两份值，
            # 供下游做真正的两段比较，而不是只有构造出来测试用的假数据。
            sampler.save_ibs_state(
                ibs_state_file, lc_win, lv_win, stage_type=stage_type
            )
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
        self, stage_type: str = "coul", *, excluded_local_windows: Optional[set] = None,
    ) -> List[Dict]:
        return load_ibs_window_outputs_from_dir(
            self.output_dir, self.ranges, self.lambdas_coul, self.lambdas_vdw,
            checkpoint_dir=self.checkpoint_dir, stage_type=stage_type,
            excluded_local_windows=excluded_local_windows,
            current_sampling_score_sha256=self.sampling_score_sha256,
        )


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
        coion_identity: Optional[Dict[str, Any]] = None,
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
            coion_identity=coion_identity,
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
            temperature=self.temperature,
        )


def _softmax_occupancy_per_state(
    u_kn_softcore: np.ndarray, f_frozen: np.ndarray, kt: float
) -> np.ndarray:
    """Per-state mean occupancy under a FIXED f_k, from raw softcore
    energies: p_k ∝ exp(-beta*(u_k - f_k)), softmax per frame then averaged
    over frames.

    [Candidate-first, Validate-or-Learn v1] Extracted so both
    ``_diagnose_local_mbar_situation`` and the VALIDATE-phase cheap
    early-exit occupancy probe (``run_all_windows``, before the full
    IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES-batch local-MBAR gate is run) read
    the exact same occupancy definition instead of two copies that could
    silently drift apart.
    """
    u = np.asarray(u_kn_softcore, dtype=np.float64)
    f = np.asarray(f_frozen, dtype=np.float64).ravel()
    beta = 1.0 / float(kt)
    logits = beta * (f[:, None] - u)
    logits -= np.max(logits, axis=0, keepdims=True)
    w = np.exp(logits)
    p = w / np.sum(w, axis=0, keepdims=True)
    return np.mean(p, axis=1)


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
    # 占据：p_k ∝ exp(-beta*(u_k - f_k))，逐帧 softmax 再对帧取平均。
    occ = _softmax_occupancy_per_state(u, f, kt)
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

# ---------------------------------------------------------------------------
# 探针 Context 的 force-group 预算。IBSSampler._build_probe_context 给每个 λ 态
# 分配一个独立 force group，以便一次 setPositions 后逐组读出每个态的相互作用能。
# 起始号必须避开生产 Hamiltonian 已占用的 0/1/2/3/4/5。
# ---------------------------------------------------------------------------
OPENMM_MAX_FORCE_GROUP = 31
PROBE_FORCE_GROUP_BASE = 16
PROBE_MAX_LAMBDA_STATES = OPENMM_MAX_FORCE_GROUP - PROBE_FORCE_GROUP_BASE + 1  # = 16

# ============================================================================
# [TARGET_SUPPORT_GATE_PROTOCOL_VERSION=1] 面向"无防护壳物理目标"的硬门
#
# 上面那套受门的 min_ess_ratio 是**扣掉共模因子之后**的混合覆盖度。它回答的是
# "已经采到的这批帧，在同一个窗口的各个 λ 态之间分得开不开"。它**不**回答
# "这批被 Group-4 λ-WCA 防护壳偏置过、且整窗只有一条轨迹的采样，能不能重加权
# 到没有防护壳的真实物理系综"——共模因子恰恰就是在那一步被除掉的。
#
# 4W53 实测（STAGE2_ROOT_CAUSE_2026-08-28.md）是这个盲点的直接证据：
#     溶剂腿  mixture min_overlap = 0.4684  → 判定通过
#             raw     min_overlap = 0.0196  → 对真实物理目标几乎不能重加权
#     converged=True，stage2 报 +35.61 kJ/mol，独立端点参考是 -6.29 kJ/mol。
# 逐窗口的 raw 绝对 ESS 是 [85.9, 8.5, 9.6]（N_decorr=[351, 432, 332]），
# 复合物腿是 [10.5, 8.1, 4.2]——两条腿都只剩个位数到十位数的等效独立样本在
# 支撑物理目标态，而 mixture 门报的是 0.46/0.89 这种"健康"数字。
#
# 所以 raw 量不能继续"只报告不设门"。但也不能把 raw_min_ess_ratio 直接拿去和
# 0.05 比：raw ratio 的上限被共模因子 exp(sigma_r^2) 整体压住（实测 sigma_r
# 0.48~1.24 kT），它是一个跟采样长度无关的比例量，用固定比例阈值卡它既可能
# 数学上不可满足、又会被"多采几帧"稀释。这里改用两项与"物理目标态到底有多少
# 真实支撑"直接对应、且彼此正交的量：
#
#   (1) raw_min_absolute_ess（**受门**）：重加权到无防护壳物理目标之后还剩多少
#       个等效独立样本。绝对数，不随总帧数自动变好；它低就是"目标系综没被采
#       到"。延长采样是能让它变好的正确方向，与 ratio 阈值"样本越少门越严"的
#       病理相反。阈值沿用 final_min_decorrelated_samples 的 20 这个口径：
#       支撑一个目标态的等效独立样本数不该低于我们对采样态本身的要求。
#   (2) max_top1pct_raw_weight（**受门**）：最重的 1% 帧占了全部 raw 权重的多少。
#       ESS 是二阶矩，权重塌缩到个别帧时它掉得慢；这一项直接看权重集中度，是
#       "少数几帧撑起整个估计"这一失效模式的第一手证据。理想值 ~0.01；实测坏
#       结果里是 0.31~0.55。0.35 = 比理想集中 35 倍，再高就没有可信的重加权。
#
# 任一项不达标、或算不出来（证据不全）→ converged=False，
# failure_reason = "insufficient_target_support"。
#
# ⚠️ 这道门只**拦得住**错值，治不了病。STAGE2_ROOT_CAUSE_2026-08-28.md §3.2 的
# window 2 是决定性反例：它相邻 <ΔU> 只有 0.4~0.6 kT，任何基于能量的重叠判据
# 都会说"完美"，却错得最多（+19.49 kJ/mol）——因为失效模式是"水塌进配体空腔
# 这个构型一次都没采到"，不是"采到的构型之间散布不够"。真正的修法在采样设计
# 层（端点态独立采样 / 窗口内真实 replica exchange / 把 lambda_WCA 变成显式热
# 力学维度并让 lambda_WCA=0 真的有样本），见该文档 §8.2。这道门的作用是让那
# 类结果 fail-closed，而不是静默流入 ΔG_bind。
# ============================================================================
TARGET_SUPPORT_GATE_PROTOCOL_VERSION = 1
TARGET_SUPPORT_MIN_ABSOLUTE_ESS = 20.0
TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT = 0.35


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
    segments: Optional[List[Dict[str, Any]]] = None,
    segment_diagnostics: Optional[List[Dict[str, Any]]] = None,
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
    if segments is not None:
        _validate_production_segments(segments, u_kj_raw.shape[1])
        indices, g_values, worst_states = [], [], []
        for segment in segments:
            start, end = segment["start_frame"], segment["end_frame"]
            # Short restart fragments cannot establish their autocorrelation.
            if end - start < 20:
                if segment_diagnostics is not None:
                    segment_diagnostics.append(dict(segment, n_decorrelated=0,
                        statistical_inefficiency=None, reason_excluded="too_short"))
                continue
            idx, g, worst = _decorrelate_by_worst_target_state(
                u_kj_raw[:, start:end], bias_kj[start:end], kt
            )
            if segment_diagnostics is not None:
                segment_diagnostics.append(dict(segment, n_decorrelated=int(idx.size),
                    statistical_inefficiency=float(g), worst_target_state=int(worst)))
            indices.append(idx + start)
            g_values.append(g)
            worst_states.append(worst)
        if not indices:
            return np.array([], dtype=int), float("inf"), -1
        worst = int(np.argmax(g_values))
        return np.concatenate(indices), float(g_values[worst]), worst_states[worst]
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
    production_segments: Optional[List[Dict[str, Any]]] = None,
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
        u_kj_raw, bias_kj, kt, segments=production_segments
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
        final_min_target_absolute_ess: float = TARGET_SUPPORT_MIN_ABSOLUTE_ESS,
        final_max_top1pct_raw_weight: float = TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT,
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
            segment_diagnostics = []
            sub_idx, g_val, g_worst_state = _decorrelate_by_worst_target_state(
                u_kj_raw, bias_kj, self.kt, segments=w.get("production_segments"),
                segment_diagnostics=segment_diagnostics,
            )
            for segment in segment_diagnostics:
                start = segment["start_frame"]
                segment["base_energy_jump_kJ_mol"] = float(base_kj[start] - base_kj[start-1]) if start else None
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
                    "production_segments": segment_diagnostics,
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
        # [TARGET_SUPPORT_GATE_PROTOCOL_VERSION=1] 权重集中度：每个窗口取它自己
        # 最差（最集中）的那个目标态，再对窗口取最大。与 raw 绝对 ESS 正交——
        # ESS 是二阶矩，抓不住"权重塌缩到个别帧"塌得还不够彻底的中间状态。
        top1pct_per_window = []
        for w in window_overlap_records:
            vals = w.get("top1pct_raw_weight")
            if not vals:
                continue
            finite = [
                float(x) for x in vals
                if x is not None and np.isfinite(float(x))
            ]
            if finite:
                top1pct_per_window.append(float(np.max(finite)))
        max_top1pct_raw_weight = (
            float(np.max(top1pct_per_window)) if top1pct_per_window else None
        )
        # 证据必须**每个窗口都有**才算数：少一个窗口，min/max 就不再是"全部窗口
        # 都达标"的证据（与 min_overlap 那里同一套 fail-closed 逻辑）。
        target_support_evidence_complete = bool(
            len(raw_abs_vals) == len(window_overlap_records)
            and len(top1pct_per_window) == len(window_overlap_records)
            and window_overlap_records
        )
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
        # ------------------------------------------------------------------
        # [TARGET_SUPPORT_GATE_PROTOCOL_VERSION=1] 无防护壳物理目标的支撑度硬门。
        # 见文件上方 TARGET_SUPPORT_GATE_PROTOCOL_VERSION 的长注释：mixture 覆盖度
        # 把共模因子除掉了，它证明不了"这批轨迹能重加权到真实物理系综"。
        # 证据不全 == 不通过（fail closed），绝不当作通过。
        # ------------------------------------------------------------------
        target_support_min_abs_ess_threshold = float(final_min_target_absolute_ess)
        target_support_max_top1pct_threshold = float(final_max_top1pct_raw_weight)
        target_support_failed: List[str] = []
        if not target_support_evidence_complete:
            target_support_failed.append("incomplete_evidence")
        if raw_min_absolute_ess is None or not np.isfinite(raw_min_absolute_ess):
            target_support_failed.append("raw_absolute_ess_unavailable")
        elif not _meets_minimum_with_roundoff(
            raw_min_absolute_ess, target_support_min_abs_ess_threshold
        ):
            target_support_failed.append("raw_absolute_ess_below_threshold")
        if max_top1pct_raw_weight is None or not np.isfinite(max_top1pct_raw_weight):
            target_support_failed.append("top1pct_raw_weight_unavailable")
        elif not _meets_maximum_with_roundoff(
            max_top1pct_raw_weight, target_support_max_top1pct_threshold
        ):
            target_support_failed.append("top1pct_raw_weight_above_threshold")
        target_support_gate = {
            "protocol_version": int(TARGET_SUPPORT_GATE_PROTOCOL_VERSION),
            "passed": not target_support_failed,
            "failure_reason": (
                "insufficient_target_support" if target_support_failed else None
            ),
            "failed_checks": list(target_support_failed),
            "raw_min_absolute_ess": raw_min_absolute_ess,
            "raw_min_absolute_ess_threshold": target_support_min_abs_ess_threshold,
            "max_top1pct_raw_weight": max_top1pct_raw_weight,
            "max_top1pct_raw_weight_threshold": target_support_max_top1pct_threshold,
            "raw_min_ess_ratio": raw_min_overlap,
            "max_common_mode_log_sigma_kT": max_common_mode_log_sigma_kT,
            "evidence_complete": bool(target_support_evidence_complete),
            "n_windows_with_evidence": len(top1pct_per_window),
            "n_windows": len(window_overlap_records),
            "per_window_raw_min_absolute_ess": [
                w.get("raw_min_absolute_ess") for w in window_overlap_records
            ],
            "per_window_max_top1pct_raw_weight": list(top1pct_per_window),
            "gate_measures": (
                "重加权到**没有 Group-4 防护壳**的物理目标态之后剩下的等效独立样本数"
                "与权重集中度；与受门的 mixture 覆盖度正交（后者把共模因子除掉了）。"
            ),
            "not_sufficient_note": (
                "通过这道门只说明目标态有起码的重要性采样支撑，**不**说明结果正确："
                "STAGE2_ROOT_CAUSE_2026-08-28.md §3.2 的 window 2 相邻 <dU> 仅 0.4~0.6 kT、"
                "任何能量重叠判据都判优，却错 +19.49 kJ/mol，因为失效模式是"
                "'该采的构型一次都没采到'。根治见该文档 §8.2。"
            ),
        }

        converged = bool(
            len(local_results) == len(valid_windows)
            and min_overlap is not None
            and _meets_minimum_with_roundoff(min_overlap, min_overlap_threshold)
            and bool(target_support_gate["passed"])
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
            # ---- [TARGET_SUPPORT_GATE_PROTOCOL_VERSION=1] 受门：物理目标支撑度 ----
            # raw_* 从"只报告不设门"升级为硬门，理由与实测数据见文件上方
            # TARGET_SUPPORT_GATE_PROTOCOL_VERSION 的注释。
            "target_support_gate_protocol_version": int(
                TARGET_SUPPORT_GATE_PROTOCOL_VERSION
            ),
            "target_support_gate": target_support_gate,
            "raw_min_absolute_ess_threshold": target_support_min_abs_ess_threshold,
            "max_top1pct_raw_weight": max_top1pct_raw_weight,
            "max_top1pct_raw_weight_threshold": target_support_max_top1pct_threshold,
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
    if "production_segments" in window:
        _validate_production_segments(window["production_segments"], n_frames)
        sliced["production_segments"] = [
            dict(segment, start_frame=max(segment["start_frame"], lo)-lo,
                 end_frame=min(segment["end_frame"], hi)-lo,
                 n_frames=min(segment["end_frame"], hi)-max(segment["start_frame"], lo))
            for segment in window["production_segments"]
            if max(segment["start_frame"], lo) < min(segment["end_frame"], hi)
        ]
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
    final_min_target_absolute_ess: float = TARGET_SUPPORT_MIN_ABSOLUTE_ESS,
    final_max_top1pct_raw_weight: float = TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT,
    min_frames_per_window: int = 10,
    split_half_max_z: Optional[float] = None,
    inflate_sigma_from_split_half: bool = False,
    _split_half_recursion_guard: bool = False,
    skip_split_half_diagnostics: bool = False,
) -> Dict:
    """stage 级 IBS-TMBAR 求解入口（stage2 = vdW）。

    ⛔ **只能用 TMBAR。** 不要在这里加 BAR / TI / 全帧主值 / √g σ 缩放 /
    bootstrap σ / σ evidence 汇总——理由与被撤回的先例见
    `ESTIMATOR_ANALYSIS_PROTOCOL_VERSION` 的注释。

    `skip_split_half_diagnostics`：[P1-19] split-half 前后半程诊断把
    `window_outputs` 的列表位置当成物理 λ 窗口编号（"window 0" = 列表第 0
    个元素）。整段生产/最终分析里两者重合，诊断有意义；但在线 TMBAR
    学习（`IBSSampler._solve_tmbar_and_recenter`）传入的是一个容量固定的
    滑动 minibatch 历史（`self.tmbar_history`），列表位置只是"当前还留在
    窗口里第几旧的 minibatch"，与任何物理 λ 窗口无关。这种调用方应该传
    `skip_split_half_diagnostics=True`，避免刷出"window 0 漂移 X"这种在
    该场景下没有实际信息量的告警（同一条 minibatch 在滑窗塞满前会被反复
    诊断出完全相同的数字），也省下这次多余的求解开销。见
    `P1-19_ONLINE_SLIDING_WINDOW_SPLITHALF_MISMATCH.md`。
    """
    if not window_outputs:
        return {"total_delta_G": 0.0, "total_error": 999.9, "converged": False}
    analyzer = GlobalMBARAnalyzer(kt=kt)
    solver_kwargs = dict(
        final_min_ess_ratio=final_min_ess_ratio,
        final_min_absolute_ess=final_min_absolute_ess,
        final_min_decorrelated_samples=final_min_decorrelated_samples,
        final_max_uncertainty_kJ_mol=final_max_uncertainty_kJ_mol,
        final_min_target_absolute_ess=final_min_target_absolute_ess,
        final_max_top1pct_raw_weight=final_max_top1pct_raw_weight,
        min_frames_per_window=min_frames_per_window,
    )
    res = analyzer.solve_stage_integrated(window_outputs, **solver_kwargs)
    res.setdefault("stage", stage_name)

    if not _split_half_recursion_guard and not skip_split_half_diagnostics:
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
            # ---- [P1-23] 抬高了 σ，端点 σ 门与 converged 必须一起重算 ----
            #
            # 原先只替换 `total_error` 与逐段 `uncertainty_kJ_mol`，却**不动**
            # `max_endpoint_uncertainty_kJ_mol`、也不重判 `converged` ——
            # 等于"σ 抬上去了，而门还在读抬高前的小 σ"。σ 抬高的**全部意义**就是
            # 让门看见真实的不确定度；门读旧值就是 fail-open：一个本该被端点 σ 门
            # 拦下的 stage 会带着 `converged=True` 通过。
            #
            # 这里只重算「端点 σ」这一项并据此重判，不碰 overlap / 独立样本数
            # 那两项（它们与 σ 口径正交，值没变）。原始值保留成
            # `*_mbar_only_*`，两套口径都在报告里可查。
            _seg_unc = [
                float(seg["uncertainty_kJ_mol"])
                for seg in (res.get("covariance_chain_segments") or [])
                if seg.get("uncertainty_kJ_mol") is not None
            ]
            if _seg_unc:
                _old_max = res.get("max_endpoint_uncertainty_kJ_mol")
                _new_max = float(np.max(_seg_unc))
                _thr = res.get("max_endpoint_uncertainty_kJ_mol_threshold")
                res["max_endpoint_uncertainty_kJ_mol_mbar_only"] = _old_max
                res["max_endpoint_uncertainty_kJ_mol"] = _new_max
                if _thr is not None and np.isfinite(_new_max):
                    _endpoint_ok = _meets_maximum_with_roundoff(_new_max, float(_thr))
                    res["max_endpoint_uncertainty_gate_passed"] = bool(_endpoint_ok)
                    if not _endpoint_ok and res.get("converged"):
                        res["converged"] = False
                        res["converged_revoked_by_sigma_inflation"] = True
                        print(
                            f"  ⛔ [P1-23] σ 抬高后端点 σ = {_new_max:.4f} kJ/mol "
                            f"超过门限 {float(_thr):.4f}，`converged` 由 True 改判为 "
                            "False（此前这里是 fail-open：σ 抬上去了而门还在读旧的小 σ）"
                        )
                if _old_max is not None:
                    print(
                        f"  ℹ️ [P1-23] 端点 σ 随之更新：{float(_old_max):.4f} → "
                        f"{_new_max:.4f} kJ/mol"
                    )
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
        co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
        seed_ledger: Optional[Exp019SeedLedger] = None,
        seed_stage: Optional[str] = None,
        seed_leg: Optional[str] = None,
    ):
        self.topology = topology
        self.positions = positions
        self.box_vectors = box_vectors
        self.ligand_indices = ligand_indices
        # [MEM-00c] 冻结的 co-ion 身份。带电配体时必须由调用方传入
        # （`ibs_engine.select_co_alchemical_ion_once()` 选一次的产物）；
        # 中性配体为 None。所有 replica 共用它，不各自重选。
        self.co_alchemical_ion_spec = co_alchemical_ion_spec
        self.seed_ledger = seed_ledger
        self.seed_stage = str(seed_stage) if seed_stage is not None else "remd"
        self.seed_leg = str(seed_leg) if seed_leg is not None else None
        if self.seed_ledger is not None and self.seed_leg != self.seed_ledger.leg:
            raise ValueError("REMD seed_leg 与 EXP-019 seed ledger 不一致")
        self.lambdas_coul = np.array(lambdas_coul)
        self.lambdas_vdw = np.array(lambdas_vdw)
        if self.lambdas_coul.ndim != 1 or self.lambdas_vdw.ndim != 1:
            raise ValueError("REMD lambda arrays must be one-dimensional")
        if len(self.lambdas_coul) < 2:
            # This engine performs adjacent-state exchanges.  A one-state
            # object is not REMD and used to reach the exchange-rate reduction
            # with a 0/0 denominator; reject it at construction instead of
            # silently reporting a non-exchange simulation as REMD.
            raise ValueError("REMD 至少需要两个 replica/lambda 状态")
        if len(self.lambdas_coul) != len(self.lambdas_vdw):
            raise ValueError("lambdas_coul 与 lambdas_vdw 长度必须一致")
        if not np.all(np.isfinite(self.lambdas_coul)) or not np.all(np.isfinite(self.lambdas_vdw)):
            raise ValueError("REMD lambda 必须全部为有限数")
        if np.any((self.lambdas_coul < 0.0) | (self.lambdas_coul > 1.0)):
            raise ValueError("lambdas_coul 必须位于 [0, 1]")
        if np.any((self.lambdas_vdw < 0.0) | (self.lambdas_vdw > 1.0)):
            raise ValueError("lambdas_vdw 必须位于 [0, 1]")
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
        temp_k = (
            float(temperature.value_in_unit(unit.kelvin))
            if hasattr(temperature, "value_in_unit")
            else float(temperature)
        )
        if not np.isfinite(temp_k) or temp_k <= 0.0:
            raise ValueError(f"REMD temperature 必须为正有限数，收到 {temperature!r}")
        self.temperature = temp_k * unit.kelvin
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
        if self.seed_ledger is not None and self.random_seed is None:
            self.random_seed = self.seed_ledger.derive(
                "charging", self.seed_stage, "exchange", "numpy"
            )
        self.rng = np.random.default_rng(self.random_seed)
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

    def _seed_for(
        self,
        phase: str,
        stage: str,
        window: Any,
        stream: str,
        attempt: int = 0,
    ) -> Optional[int]:
        if self.seed_ledger is None:
            return None
        return self.seed_ledger.derive(phase, stage, window, stream, attempt)

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

        # [P0-REMD-CUDA] 建 Context 前后记显存。
        #
        # 为什么必须记：`No compatible CUDA device is available` 是 OpenMM 在**所有**
        # 设备都初始化失败时给的**通用文案**，真 OOM 也长这样。没有"第几个 replica
        # 断的 / 断时剩多少显存"这两个数，"显存不够"和"其它初始化失败"就区分不了，
        # 只能反复猜。离线探针（`memtest/probe_remd_context_capacity.py`）已证明
        # 12 × 45354 原子的 Context 单进程能建满、每个 ≈ 338 MiB，
        # 连"先跑一遍 λ 预优化"都复现不了失败 —— 所以差异只可能在生产路径上，
        # 必须就地打点。
        _vram_baseline = _gpu_memory_mib()
        if _vram_baseline and str(resolved_platform).upper() == "CUDA":
            logger.info(
                "[REMD] 建 %d 个 %s Context 之前显存: used=%d free=%d total=%d MiB "
                "(props=%s)",
                self.n_replicas, resolved_platform,
                _vram_baseline[0], _vram_baseline[1], _vram_baseline[2], props,
            )

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
                        # [MEM-00c] 所有 replica 共用同一份冻结身份，不各自重选。
                        co_alchemical_ion_spec=self.co_alchemical_ion_spec,
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
                        co_alchemical_ion_spec=self.co_alchemical_ion_spec,
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
                integrator_seed = self._seed_for(
                    "charging", self.seed_stage, i, "integrator"
                )
                if integrator_seed is not None:
                    integ.setRandomNumberSeed(integrator_seed)
                ctx = openmm.Context(replica_sys, integ, platform, props)
                ctx.setPositions(self.positions)
                if self.box_vectors is not None:
                    ctx.setPeriodicBoxVectors(*self.box_vectors)
                self._try_set_context_parameter(ctx, "lambda_coul", self.lambdas_coul[i])
                self._try_set_context_parameter(ctx, "lambda_vdw", self.lambdas_vdw[i])
                velocity_seed = self._seed_for(
                    "charging", self.seed_stage, i, "velocity"
                )
                if velocity_seed is None:
                    ctx.setVelocitiesToTemperature(self.temperature)
                else:
                    ctx.setVelocitiesToTemperature(self.temperature, velocity_seed)

                self.contexts.append(ctx)
                self.integrators.append(integ)
                self.replica_systems.append(replica_sys)

                # [P0-REMD-CUDA] 逐个记，断在第几个一看就知道。
                _vram = _gpu_memory_mib()
                if _vram and str(resolved_platform).upper() == "CUDA":
                    _per_ctx = (
                        (_vram[0] - _vram_baseline[0]) / len(self.contexts)
                        if _vram_baseline else float("nan")
                    )
                    logger.info(
                        "[REMD] Context %d/%d 建成: used=%d free=%d MiB "
                        "(平均每 Context ≈ %.0f MiB)",
                        len(self.contexts), self.n_replicas,
                        _vram[0], _vram[1], _per_ctx,
                    )
        except Exception as exc:
            # ⚠️ 显存必须在 `_clear_replica_contexts()` **之前**读——释放之后再读就
            # 只剩一个"失败后已回收"的数，判不了当时到底是不是不够用。
            _vram_at_failure = _gpu_memory_mib()
            _n_built = len(self.contexts)
            self._clear_replica_contexts()
            if (
                allow_platform_fallback
                and str(resolved_platform).upper() in {"CUDA", "OPENCL"}
                and self._is_gpu_context_failure(exc)
            ):
                self.platform_fallback_reason = str(exc)
                # 判 OOM / 非 OOM：断点还剩得下一个 Context 的量 ⟹ **不是**显存不够，
                # 别按 OOM 修（把 λ 数减小只是掩盖）。阈值取实测的每 Context 量级。
                _diagnosis = "显存读数不可用（无 nvidia-smi？）"
                if _vram_at_failure:
                    _per_ctx = (
                        (_vram_at_failure[0] - _vram_baseline[0]) / max(1, _n_built)
                        if _vram_baseline and _n_built else float("nan")
                    )
                    _looks_like_oom = _vram_at_failure[1] < max(
                        _REMD_CONTEXT_VRAM_FLOOR_MIB, _per_ctx if _per_ctx == _per_ctx else 0.0
                    )
                    _diagnosis = (
                        f"失败瞬间 used={_vram_at_failure[0]} free={_vram_at_failure[1]} "
                        f"total={_vram_at_failure[2]} MiB，已建成 {_n_built} 个"
                        + (f"（平均每 Context ≈ {_per_ctx:.0f} MiB）" if _per_ctx == _per_ctx else "")
                        + "。判定: "
                    )
                    if not _looks_like_oom:
                        _diagnosis += (
                            "**不像 OOM** —— 剩余显存仍足够再建一个 Context，"
                            "所以这是别的初始化失败；减 λ 数只会掩盖它，别按 OOM 修。"
                        )
                    else:
                        # 🔑 显存不够有**两种**完全不同的原因，别混成一句"减 λ 就行"：
                        #   (a) REMD 自己的 N 个 Context 就把卡吃满了 ⟹ 减 λ 是对的；
                        #   (b) **开跑前**卡上已经被占掉一大块 ⟹ 真问题是那个占用者
                        #       （同卡上的别的作业 / 上一次没杀干净的进程 / 本进程前面
                        #       某个 CUDA 阶段没释放）。这时减 λ 只是把症状盖住，
                        #       下次窗口一多又会撞上。
                        # 实测教训（2026-08-04）：`vram_before=[12197, 3646, 16303]`，
                        # 12 个 Context 只需 3804 MiB 却只剩 3646 MiB —— 差 158 MiB。
                        # 本条腿之前的 CUDA 阶段（预平衡 1 + attachment 4 + λ 预优化 1）
                        # 约 2 GB，剩下约 10 GB 来源不明。若当时只报"减 λ"，
                        # 那 10 GB 就永远不会被查。
                        _need_all = _per_ctx * self.n_replicas if _per_ctx == _per_ctx else float("nan")
                        _diagnosis += "**像 OOM**（剩余显存已不足一个 Context）。"
                        if _vram_baseline:
                            _baseline_used, _baseline_free, _baseline_total = _vram_baseline
                            _diagnosis += (
                                f" 但注意开跑**之前**就已 used={_baseline_used} "
                                f"free={_baseline_free} MiB"
                            )
                            if _need_all == _need_all:
                                _diagnosis += (
                                    f"；{self.n_replicas} 个 Context 共需 ≈ {_need_all:.0f} MiB"
                                )
                            if _baseline_used > 0.25 * _baseline_total:
                                _diagnosis += (
                                    f"。⚠️ 开跑前的占用已达全卡 "
                                    f"{100.0 * _baseline_used / _baseline_total:.0f}% —— "
                                    "**先查是谁占的**（`nvidia-smi` 看 PID：同卡上的别的作业？"
                                    "上一次没杀干净的进程？本进程前面某个 CUDA 阶段没释放？）。"
                                    "只减 λ 数能让这次跑起来，但会把这块占用永久掩盖，"
                                    "窗口一多又会撞上。"
                                )
                            else:
                                _diagnosis += (
                                    "。开跑前占用不大 ⟹ 确实是 REMD 自己的 Context 数"
                                    "超过了这张卡，减少 λ 状态数"
                                    "（= replica 数 = 常驻 Context 数）是对的修法。"
                                )
                        else:
                            _diagnosis += (
                                " 拿不到开跑前的显存基线，无法区分"
                                "'REMD 自己吃满'与'开跑前已被占'——先补这个数再决定要不要减 λ。"
                            )
                _message = (
                    "⚠️ REMD GPU Context 构建失败，已释放已创建的 replica contexts；"
                    f"回退 CPU 重建。\n"
                    f"     原始错误 [{type(exc).__name__}]: {exc}\n"
                    f"     {_diagnosis}\n"
                    "     ⛔ CPU 回退会让本阶段慢约两个数量级（实测 45354 原子 × 12 副本："
                    "第 0 轮交换用了 29 分钟，500 轮约 10 天），表现得像卡死。"
                )
                logger.warning(_message)

                # 🔑 当场落盘，不等阶段跑完。
                #
                # 这条告警此前只到终端：`logger` 没有 FileHandler，而 `pipeline.log`
                # 是 `ABFEPipeline._log` 自己写的另一条通路，所以归档日志里**一行都没有**
                # （2026-08-03/04 因此绕了很多轮：只能靠 `nvidia-smi` 看到卡是空的，
                # 才推断出它退了 CPU）。而 `platform_fallback_reason` 只在阶段**跑完**时
                # 进 `*_exchange_diagnostics.json` —— 一旦像这次那样在 CPU 上磨到被杀，
                # 那份文件永远不会出现，证据就彻底没了。
                try:
                    os.makedirs(self.output_dir, exist_ok=True)
                    _fallback_path = os.path.join(
                        self.output_dir, "remd_platform_fallback.json"
                    )
                    with open(_fallback_path, "w", encoding="utf-8") as _handle:
                        json.dump(
                            {
                                "requested_platform": str(resolved_platform),
                                "platform_properties": dict(props),
                                "n_replicas": int(self.n_replicas),
                                "max_resident_contexts": int(self.max_resident_contexts),
                                "n_contexts_built_before_failure": int(_n_built),
                                "exception_type": type(exc).__name__,
                                "exception": str(exc),
                                "vram_before_mib": (
                                    list(_vram_baseline) if _vram_baseline else None
                                ),
                                "vram_at_failure_mib": (
                                    list(_vram_at_failure) if _vram_at_failure else None
                                ),
                                "diagnosis": _diagnosis,
                            },
                            _handle,
                            indent=2,
                            ensure_ascii=False,
                        )
                    print(f"  🧾 平台回退证据已落盘: {_fallback_path}")
                except Exception as _write_exc:  # noqa: BLE001
                    logger.warning("平台回退证据落盘失败: %s", _write_exc)

                print(f"  {_message}")
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
                velocity_seed = self._seed_for(
                    "recovery", self.seed_stage, state_idx, "velocity", attempt=1
                )
                if velocity_seed is None:
                    ctx.setVelocitiesToTemperature(self.temperature)
                else:
                    ctx.setVelocitiesToTemperature(self.temperature, velocity_seed)
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
            velocity_seed = self._seed_for(
                "recovery", self.seed_stage, state_idx, "velocity", attempt=2
            )
            if velocity_seed is None:
                ctx.setVelocitiesToTemperature(self.temperature)
            else:
                ctx.setVelocitiesToTemperature(self.temperature, velocity_seed)

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
        n_steps = int(n_steps)
        exchange_interval = int(exchange_interval)
        save_interval = int(save_interval)
        if n_steps < 0:
            raise ValueError("REMD n_steps 不能为负")
        if exchange_interval <= 0:
            raise ValueError("REMD exchange_interval 必须为正")
        if save_interval <= 0:
            raise ValueError("REMD save_interval 必须为正")
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
                    
            # A one-replica REMD run is valid (it is simply a single-state
            # simulation), but has no exchange pairs. Avoid 0/0 and record a
            # well-defined zero acceptance rate.
            exchange_log.append(accepted / max(1, self.n_replicas - 1))
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
        seed_ledger: Optional[Exp019SeedLedger] = None,
        seed_stage: Optional[str] = None,
        seed_leg: Optional[str] = None,
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
            seed_ledger=seed_ledger,
            seed_stage=seed_stage or "boresch_attachment",
            seed_leg=seed_leg,
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
                integrator_seed = self._seed_for(
                    "attachment", self.seed_stage, i, "integrator"
                )
                if integrator_seed is not None:
                    integ.setRandomNumberSeed(integrator_seed)
                ctx = openmm.Context(replica_sys, integ, platform, props)
                ctx.setPositions(self.positions)
                if self.box_vectors is not None:
                    ctx.setPeriodicBoxVectors(*self.box_vectors)
                self._try_set_context_parameter(ctx, self.lam_name, self.lambdas_boresch[i])
                velocity_seed = self._seed_for(
                    "attachment", self.seed_stage, i, "velocity"
                )
                if velocity_seed is None:
                    ctx.setVelocitiesToTemperature(self.temperature)
                else:
                    ctx.setVelocitiesToTemperature(self.temperature, velocity_seed)
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
    #
    # 直接调用、不吞异常——这是本文件唯一一处会给 setParameter 包 try/except 的
    # 地方（其它调用点，如 1778/1993/2023/2027/2033/2043 行，全都直接调用让异常
    # 传出去）。这个函数的唯一任务就是在 λ=1 下重算整条 attachment leg 每一帧的
    # U_Boresch，喂给 MBAR 的 u_kn；如果 eval_sys 上真的没有这个全局参数（比如
    # 未来改名/序列化路径变了），静默吞掉意味着后面每一帧都在"某个默认 λ"下算，
    # 而不是 λ=1，整条 leg 的自由能贡献会悄悄错掉且零信号——必须让它在这里就炸。
    ctx.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, 1.0)

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
        seed_ledger: Optional[Exp019SeedLedger] = None,
        seed_stage: Optional[str] = None,
        seed_leg: Optional[str] = None,
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
            seed_ledger=seed_ledger,
            seed_stage=seed_stage or "shadow_bridge",
            seed_leg=seed_leg,
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
                integrator_seed = self._seed_for(
                    "bridge", self.seed_stage, i, "integrator"
                )
                if integrator_seed is not None:
                    integ.setRandomNumberSeed(integrator_seed)
                ctx = openmm.Context(replica_sys, integ, platform, props)
                ctx.setPositions(self.positions)
                if self.box_vectors is not None:
                    ctx.setPeriodicBoxVectors(*self.box_vectors)
                self._try_set_context_parameter(ctx, self.s_param_name, self.lambdas_bridge_s[i])
                velocity_seed = self._seed_for(
                    "bridge", self.seed_stage, i, "velocity"
                )
                if velocity_seed is None:
                    ctx.setVelocitiesToTemperature(self.temperature)
                else:
                    ctx.setVelocitiesToTemperature(self.temperature, velocity_seed)
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
    seed_ledger: Optional[Exp019SeedLedger] = None,
    seed_stage: Optional[str] = None,
    seed_leg: Optional[str] = None,
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
    if len(lambdas_bridge_s) < 2:
        raise ValueError("Shadow-Bridge 至少需要 2 个 s 状态")
    if not np.all(np.isfinite(lambdas_bridge_s)):
        raise ValueError("Shadow-Bridge s 必须全部为有限数")
    if any(s < 0.0 or s > 1.0 for s in lambdas_bridge_s):
        raise ValueError("Shadow-Bridge s 必须位于 [0, 1]")
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
        seed_ledger=seed_ledger,
        seed_stage=seed_stage or "shadow_bridge",
        seed_leg=seed_leg,
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
        co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
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
        # 🔑 [性能修复：进程×线程过度并行] 每个 worker 进程的 OpenMM CPU
        # Context 之前不设线程上限，默认想用满所有物理核——n_workers 个进程
        # 同时这么干就是经典的过度并行。这里按"总物理核数 / worker 进程数"
        # 给每个 Context 分配线程预算，保证 n_workers × cpu_threads_per_worker
        # ≤ cpu_count。n_workers==1 时预算等于 cpu_count，跟原来隐式"用满
        # 所有核"的行为一致，不改变单进程路径的性能特征。
        cpu_threads_per_worker = max(1, cpu_count // max(1, n_workers))
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
                # [MEM-00c] 同上：只读消费冻结身份，不按 reference_positions 重选。
                co_alchemical_ion_spec=co_alchemical_ion_spec,
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
                # [MEM-00c] 重算必须用**动力学当时冻结的那个粒子**。
                # `reference_positions` 在 resume 进程里与首跑不同，
                # 以前这里会据此重新选一个离子 —— 那正是静默不一致的来源。
                co_alchemical_ion_spec=co_alchemical_ion_spec,
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

        print(
            f"\n📊 开始离线能量重算 | {n_frames} 帧 × {n_states} 态 | workers={n_workers} | "
            f"chunk_size={chunk_size} | cpu_threads_per_worker={cpu_threads_per_worker}"
        )
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
                    "cpu_threads": cpu_threads_per_worker,
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
                # 🔑 [性能修复：worker Context 复用 + 不重复传输 System XML]
                # 用 initializer 让每个 worker 进程启动时只反序列化一次
                # System、建一次 Context（_mbar_worker_init），同一个 worker
                # 后续通过 imap_unordered 拉到的所有 chunk 都复用它——不再
                # 像之前那样每个 chunk 都重新反序列化+重建。既然 Context 已
                # 经在 worker 启动时用 system_xml/platform_name/cpu_threads
                # 建好了，pool_tasks 就不用再让每个 chunk 各自带一份（这三
                # 个字段在一次 compute_u_kn 调用里对所有 chunk 都相同）——
                # 省掉一份不必要的重复 IPC 序列化开销。任务失败回退到下面
                # except 分支时用的还是原来完整的 `tasks`（带 system_xml），
                # 保证串行回退路径行为不变。
                pool_tasks = [
                    {k: v for k, v in task.items() if k not in ("system_xml", "platform_name", "cpu_threads")}
                    for task in tasks
                ]
                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=n_workers,
                    initializer=_mbar_worker_init,
                    initargs=(system_xml, platform_name, cpu_threads_per_worker),
                ) as pool:
                    for frame_offset, u_chunk in pool.imap_unordered(_compute_u_kn_chunk, pool_tasks):
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
