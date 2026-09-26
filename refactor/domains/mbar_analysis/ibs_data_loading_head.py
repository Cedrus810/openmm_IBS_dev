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
from step_guard import guarded_step, step_with_chunk_rollback
from abfe_core import (
    # 整分子周期回卷：连通性只信 System，不信 topology 的键（唯一实现）。
    image_molecules_by_system,
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
    create_ligand_internal_coulomb_force,
    LIGAND_INTERNAL_COULOMB_FORCE_NAME,
    ensure_owned_system,
    sync_all_exclusions,
    BeutlerSoftcoreBuilder,
    _build_mbar_compatible,
    _compute_free_energy_result_compatible,
    _extract_free_energy_arrays,
    subsample_series_by_autocorrelation,
    # [2026-09-15] 唯一不依赖 MBAR 渐近性的精度判据。**不要在本文件另立一份实现**
    # ——那正是 `max_endpoint_uncertainty_kJ_mol ≤ 1.0` 这个魔法阈值的成因。
    cross_repeat_precision,
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
    box_vectors_to_nm_array,
    unwrap_boresch_anchors_nm,
    # [CLI-01, 2026-09-02] pymbar 的可用性探测统一在 abfe_core 里做（惰性、
    # memoized）。本模块以前在这里 eager `import pymbar`，让每个 import
    # ibs_engine 的入口都付一次 JAX 初始化；实际用到 pymbar 的只有
    # `from pymbar import other_estimators` 那处函数内 import。
    has_pymbar,
)

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
        """完整快照：含 `derived_seed_map`。

        ⚠️ **只能用于 provenance 记录，不能进任何缓存/协议指纹。**
        `derived_seed_map` 记的是"本进程到此刻为止实际派生过哪些种子"，它随执行路径
        增长 —— 首跑时预平衡真跑过、派生了 equilibration/* 的种子；resume 时预平衡走
        checkpoint 没派生，同一份配置下快照就不同。把它放进缓存键会让键依赖**执行历史**
        而不是**配置身份**，于是每次 resume 都判失配、整段重算。
        做缓存身份请用 `identity()`。
        """
        return {
            "protocol_version": EXP019_SEED_WIRING_PROTOCOL_VERSION,
            "repeat_seed": self.repeat_seed,
            "leg": self.leg,
            "derived_seed_map": {
                key: int(self.values[key]) for key in sorted(self.values)
            },
        }

    def identity(self) -> Dict[str, Any]:
        """稳定身份：只含决定"种子从哪来"的量，不含已派生记录。

        换 repeat_seed / 换腿 ⇒ 变；同一配置下重跑或 resume ⇒ 逐位不变。
        这才是能进缓存键的那一份。
        """
        return {
            "protocol_version": EXP019_SEED_WIRING_PROTOCOL_VERSION,
            "repeat_seed": self.repeat_seed,
            "leg": self.leg,
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


print = _log_print


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
    # 🔑 [2026-09-16] 冻结验证的批数上限与 `final_*` 门槛同类：它决定**判定**，
    # 不决定这个窗口采的是什么。冻结验证的样本与生产严格隔离（"验证样本不计入
    # 生产，生产从 0 步开始"），所以窗口轨迹对它免疫、可以照常复用；
    # 而 stage **结果**仍要求完整 protocol key，那一层会因它失配 —— 正是要的效果。
    payload.pop("ibs_local_mbar_gate_max_batches", None)
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
    # 🔑 [2026-09-10] 路径形状键同理：它们决定**有哪些窗口**，不决定**某个窗口采什么**。
    # 一个窗口采样的哈密顿量完全由它自己的 λ 值 + 该窗口 System 决定，而这两者都
    # 已经被单独校验（`_resume_cached_window_gate_status` 里 `lambdas_match` 是
    # `usable` 的 AND 项；原生续算另查完整 window System 与冻结 f_k）。把它们留在
    # 窗口采样身份里的后果是：在末段插一个 λ 时，前面 λ 值逐位未变、物理上完全可
    # 复用的窗口也会被判失配而整窗重采（实测代价见 `_stage_protocol_key` 注释：
    # Stage 1 约 28 分钟 + 6 个 vanishing 窗口全部重采样）。这与本函数已经摘掉
    # 预算/门槛是同一条原则，只是原来没贯彻到路径形状上。
    # ⚠️ stage **结果**缓存不受影响：它仍要求完整 protocol key，另外还叠了
    # `lambda_path_fingerprint` 全路径门（abfe_pipeline.py stage2 resume 处），
    # 所以路径一变，总 ΔG 绝不会被静默复用。这里放宽的只有窗口轨迹的复用。
    # ⚠️ 本放宽的安全性**依赖** `lambdas_match` 继续是 `usable` 的 AND 项；
    # 若哪天它被摘掉，这几个键必须同时放回来。见 test_resume_reuse_contracts.py。
    for key in (
        "stage2_final_n_states",
        "stage2_free_energy_densify_points",
        "stage2_window_min_states",
        "stage2_window_max_states",
        "stage2_refine_extra_points_per_segment",
        # 采样修复策略同理：它决定路径**怎么演化**，不决定某个窗口采什么。窗口层面
        # 另有 `repair_policy_match` 按等价类单独判（non_mutating_v1 与
        # path_evolution_v1 同类，legacy_mutating 自成一类）。
        "sampling_repair_policy",
        # 路径演化 / f_k 重标定的**执行策略**：它们决定要不要演化路径、要不要另开
        # 一段采样，不决定某个已落盘窗口采的是什么。被重标定的那段写在独立目录
        # （vanishing_2/）里、自带 checkpoint 树，原目录的窗口逐帧未变。
        # 不摘掉的后果：任何人一旦打开这两个开关，就会让**已有的全部窗口缓存失配、
        # 静默重采**（实测代价见 `_stage_protocol_key` 注释）。
        "max_path_insertions",
        "stage2_recalibrate_f_k_on_rescue",
        "stage2_f_k_recalibration_min_shift",
    ):
        kwargs.pop(key, None)
    # Preserve the original schema's presence/absence of unrelated fields.
    if "kwargs" in config:
        config["kwargs"] = kwargs
    return payload


def _resolve_analysis_f_k(ibs_state, convergence, live_f_k, local_idx):
    """分析要用的 f_k = **生成这批生产帧的那一份**，不是状态文件里活的那份。

    🔑🔑 [2026-09-14] 先前这里断言 `ibs_state["f_k"] == production_entry_f_k`，
    对不上就 `窗口 N f_k 与冻结生产入口不一致` 直接拒。但这两个字段**语义不同**：

      · `production_entry_f_k` —— 生产真正开始那一刻锁住的值，
        `_resolve_production_entry_marker` 写一次、**生产期间从不再赋值**；
      · `f_k`                  —— 学习器**活的**状态，续跑时窗口再进一次 warmup
        就会动（`was_resumed=True` / `final_mode=learning`）。

    所以"活的 f_k 漂了"是**合法状态**，不是数据矛盾。真机 `cyclod_ligand1/rep3`
    win0：预算在上一跑耗尽（`no_budget_remaining_on_resume`）⟹ 本次续跑没验成、
    判 `indeterminate`，活的 f_k 与入口标记差 0.03~0.28 kJ/mol，而生产帧明明
    是入口那一份驱动出来的 —— 整条分析被一个**用错字段**的断言挡死。

    ESS 门自己的注释写的就是"该窗口**冻结进生产**的 f_k"。这里改成直接取它，
    并且**加一道原来没有的横向核对**：入口标记必须与 convergence 里独立记录的
    `bias_warmup.frozen_f_k_at_last_freeze` 一致。两份独立记录对不上才是真矛盾，
    照旧 fail-closed。缺标记同样照旧拒绝。
    """
    marker = np.asarray(
        ibs_state.get("production_entry_f_k") or [], dtype=np.float64
    ).ravel()
    if marker.shape != live_f_k.shape or not np.all(np.isfinite(marker)):
        raise ValueError(
            f"窗口 {local_idx} 缺少可用的生产入口标记 production_entry_f_k"
            f"（长度 {marker.size}，期望 {live_f_k.size}）⟹ 拿不到「生成这批帧的"
            "那份 f_k」，拒绝在缺判据的情况下继续分析"
        )
    if np.array_equal(marker, live_f_k):
        return live_f_k

    frozen = np.asarray(
        ((convergence.get("bias_warmup") or {}).get("frozen_f_k_at_last_freeze")
         or []), dtype=np.float64
    ).ravel()
    if frozen.shape != marker.shape or not np.allclose(
        frozen, marker, rtol=0.0, atol=1.0e-9
    ):
        raise ValueError(
            f"窗口 {local_idx} 的生产入口标记与 convergence 里独立记录的"
            f"`frozen_f_k_at_last_freeze` 对不上 ⟹ **两份本该相同的记录矛盾**，"
            "拒绝继续分析（这与「活的 f_k 漂了」不是一回事）"
        )
    print(
        f"  [INFO] 窗口 {local_idx} 状态文件里活的 f_k 与生产入口标记不同"
        f"（最大差 {float(np.max(np.abs(live_f_k - marker))):.4g} kJ/mol）——"
        "续跑时该窗口又进过一次 warmup，属合法状态。分析用**生产入口**那一份"
        "（已与 convergence 的 frozen_f_k_at_last_freeze 交叉核对一致）。"
    )
    return marker


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
        # 🔑 [2026-09-12] **在量化栅格上比，不做浮点精确相等。**
        # λ 路径落盘时被 `lambda_path_versions._q()` 量化到 LAMBDA_DECIMALS 位，
        # 而窗口 convergence.json 记的是**实际采样用的**那个未量化值。两者因此
        # 天然差半个栅格（真机实测 win4：路径 0.31005333 vs 采样 0.310053335，
        # 差 5e-9），于是：
        #   · 采样侧 resume 门用 np.allclose ⟹ 判"λ 匹配、跳过重采"
        #   · 这里用精确相等        ⟹ 判"λ 不匹配"、抛 ValueError
        # 同一个不变量两套严格度 ⟹ 那个窗口**永远采不了也永远读不了**，每次启动必崩。
        #
        # 这不是放宽判据：λ 既然量化到 8 位，任何**真实**的 λ 差异都 ≥ 1e-8，
        # 而量化往返噪声 ≤ 5e-9。把两边都归一到栅格上再精确比较，既消掉噪声，
        # 又保持对真实差异的 fail-closed。
        def _grid(vals):
            return [round(float(x), LAMBDA_GRID_DECIMALS) + 0.0 for x in (vals or [])]

        if (_grid(convergence.get("lambdas_coul")) != _grid(lc)
                or _grid(convergence.get("lambdas_vdw")) != _grid(lv)):
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
        f_k_window = _resolve_analysis_f_k(
            ibs_state, convergence, f_k_window, local_idx
        )
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
            elif field in ("lambdas_coul", "lambdas_vdw"):
                # 与上面同一条理由：λ 路径落盘时被量化到 LAMBDA_GRID_DECIMALS 位，
                # manifest 记的是实际采样值，两者天然差半个栅格。归一到栅格上
                # 再精确比较 —— 消掉量化噪声，保持对真实 λ 差异的 fail-closed。
                observed = [round(float(x), LAMBDA_GRID_DECIMALS) + 0.0
                            for x in (observed or [])]
                expected = [round(float(x), LAMBDA_GRID_DECIMALS) + 0.0
                            for x in (expected or [])]
            if observed != expected:
                raise ValueError(f"窗口 {local_idx} production manifest {field} 不匹配")
        if production_manifest.get("frozen_f_k_sha256") != frozen_hash:
            raise ValueError(f"窗口 {local_idx} 冻结 f_k 与 production manifest 不一致")
        segments = _validate_production_segments(convergence.get("production_segments"), n_frames)
        outputs.append({
            "production_segments": segments,
            # 🔑 [2026-09-15] 这个窗口的帧是不是在一份**从未通过冻结验证**的 f_k
            # 下采的。必须跟着窗口一路传到 stage result —— 「这份 ΔG 里含临时生产
            # 窗口」这件事只活在日志里等于没有。老产物缺键 ⟹ False。
            "provisional_production": bool(
                convergence.get("provisional_production", False)),
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
            # [CTL-09] 自检那一份 provenance 落在这个文件里。求解器（GlobalMBARAnalyzer）
            # 只拿得到 window_data、拿不到 output_dir/stage_type，所以路径必须在**这里**
            # 随窗口一起传下去，否则同输入对照在生产里根本无从发起。
            # ⚠️ 多段合并（abfe_pipeline 的 base = dict(parts[0])）会继承**第 0 段**的
            # 路径；那份自检只覆盖第 0 段的帧，与合并后的帧集本来就不是同一份输入 ——
            # 对照会如实判"输入不同"，这正是要的结果，不是 bug。
            "self_support_path": window_self_support_path(
                output_dir, stage_type, local_idx
            ),
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
        # 🔑 [2026-09-09] 显式标记"这个窗口属于 residual 臂"。
        #
        # 下游 `solve_stage_integrated` 原来是拿
        # `sampling_gauge_required = sampling_kj is not None` 当守卫 —— 用**被守护
        # 的那个数组自己**推导要不要守护它，是个恒真的循环，永远不会触发。
        # 它要防的情形（residual 臂的窗口拿不到 sampling_state_energies，于是 ESS /
        # occupancy / common_mode 全部退回物理规范却照样过门）恰好是它测不出来的
        # 那一种。在线调用方一直是对的（从 `ibs_wrapper.residual_enabled` 读），
        # 只有 stage-final 这条不是。这里给它一个不依赖数组本身的判据。
        "residual_sampling_arm": True,
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


class IBSControlPlaneError(RuntimeError):
    """引擎抛出的**决定**，不是崩溃。所有这类异常必须继承它。

    🔑🔑🔑 [2026-09-18] **「引擎抛的可路由异常」先前是一张手写白名单，漏一个就是
    整跑以 traceback 收场。**

    实测：`ExistingEnsembleRequiresRescueAudit` 的消息自己写着 `route it to
    rescue/provenance audit`，而 `abfe_pipeline` 的两处 handler
    （自治循环、`_run_stage2_with_path_evolution._guarded_once`）都只列了
    `IBSValidationBudgetIndeterminateError` + `IBSWarmupConvergenceError`
    ⟹ 它落进 `except Exception` ⟹ 一个**已知、可分类**的控制器结局被伪装成
    执行器崩溃。而它的四个 `non_mutating_v1` 守卫恰好是自治模式的常态策略。

    为什么要一个基类而不是只加一个 dict：dict 里**没有**的条目照旧静默落进
    `except Exception`，"下一个不会漏"这句话就不成立。有了基类，
    `IBSControlPlaneError.__subclasses__()` 是一个**可枚举的唯一集合**，
    `abfe_pipeline` 在 import 时拿它跟注册表对账，缺一项立刻抛。

    ⚠️ **「刻意不路由」也必须显式登记**（注册表里有 `NOT_ROUTED` 档）。
    靠缺席表达"决定不路由"读不出来 —— 它和"忘了"长得一模一样。这与
    `EXIT_SPECS` 里 `NO_FEASIBLE_ACTION` 的 `default_scope=None` 是同一个手法。

    ⚠️ 它**仍然是 `RuntimeError` 子类**：插在中间不改变任何既有 `except
    RuntimeError` / `except <具体类>` 的行为，纯增量。
    """


class IBSIncompleteStageCoverageError(IBSControlPlaneError):
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


# 唯一实现在 abfe_core（`unwrap_boresch_anchors_nm` 与 co-ion 几何判据在那一层就要用）。
_box_vectors_to_nm_array = box_vectors_to_nm_array


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

    🔑 [0831issue P2] 这里只累加 `scale²`，也就是隐含假设**每个带 offset 的粒子的
    base 电荷为 0**：完整式子是
    ``Σq(λ)² = Σq_base² + 2λ·Σ(q_base·scale) + λ²·Σscale²``。
    当前三个 builder 都先 `setParticleParameters(idx, 0.0, ...)` 把 base 电荷清零、
    再挂 offset，所以前提成立、只留 `λ²Σscale²` 一项是对的。但这个前提以前**没有
    任何断言守护** —— 一旦有人改成分段去电荷（base 电荷不清零），离线 u_kn 的 PME
    自能修正会残留一个假能量项，而且两条腿同错、循环闭合里不会互相抵消，是个静默
    的自由能偏差。现在把前提就地断言掉。
    """
    qsq = 0.0
    rows: List[Dict[str, float]] = []
    for offset_idx in range(nb_force.getNumParticleParameterOffsets()):
        param, particle_idx, charge_scale, _sigma_scale, _epsilon_scale = nb_force.getParticleParameterOffset(offset_idx)
        if str(param) != str(lambda_name):
            continue
        q_base, _sigma_base, _eps_base = nb_force.getParticleParameters(int(particle_idx))
        q_base_e = _value_in_elementary_charge(q_base)
        if abs(q_base_e) > 1.0e-9:
            raise RuntimeError(
                f"pme_offset_charge_square_sum 假设带 {lambda_name} 电荷 offset 的粒子 "
                f"base 电荷为 0，但粒子 {int(particle_idx)} 的 base 电荷是 {q_base_e:.12g} e。"
                "此时 Σq(λ)² 还有 Σq_base² 与 2λΣ(q_base·scale) 两项，只算 λ²Σscale² 会让"
                "离线 u_kn 的 PME 自能修正残留假能量项（两条腿同错、循环里不抵消）。"
                "请先把完整公式实现出来再放开这条路径。"
            )
        q_val = _value_in_elementary_charge(charge_scale)
        qsq += q_val * q_val
        rows.append({
            "particle_index": int(particle_idx),
            "charge_offset_e": float(q_val),
            "charge_offset_square_e2": float(q_val * q_val),
            "base_charge_e": float(q_base_e),
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
    prepared = openmm.XmlSerializer.clone(system_template)
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


