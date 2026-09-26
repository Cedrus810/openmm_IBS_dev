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
import gc
import numpy as np
import re
import os
import glob
import json
import contextlib
import shutil
import multiprocessing as mp
import time
import logging
import builtins
import hashlib
import platform
import socket
import sys
import math
import atexit
from datetime import datetime
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple, Optional

# 项目内部模块依赖
from abfe_preoptimizer import (
    ABFEPreOptimizer,
    DualLambdaPreOptimizer,
    recompute_vanishing_path_from_cached_pilot,
    segment_index_of_dir,
)
from abfe_preoptimizer import build_aces_probe_system, build_aces_probe_system_dual_lambda
from abfe_preoptimizer import refine_stage_lambda_path_from_data
from abfe_preoptimizer import (
    refine_stage_lambda_path_by_overlap,
    split_window_from_warmup_failure,
    insert_lambda_from_overlap_failure,
    plan_vdw_overlap_repair_targets,
    canonicalize_window_ranges,
    split_window_from_ibs_lse_failure,
    feasible_repair_actions,
    vanishing_rescue_ranges,
    tail_overflow_exemption_active,
    Stage2RepairController,
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
    run_boresch_attachment_leg,
    DEFAULT_BORESCH_ATTACHMENT_LAMBDAS,
    WCA_ACCOUNTING_VERSION,
    IBS_BIAS_PROTOCOL_VERSION,
    # [TARGET_SUPPORT_GATE_PROTOCOL_VERSION=1] 无防护壳物理目标支撑度硬门的
    # 阈值与协议版本；求解器侧的定义见 ibs_engine.py 同名常量的长注释。
    TARGET_SUPPORT_GATE_PROTOCOL_VERSION,
    TARGET_SUPPORT_MIN_ABSOLUTE_ESS,
    TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT,
    # [INDEPENDENT_ENDPOINT_PROTOCOL_VERSION=1] 端点态独立固定-λ 生产采样。
    INDEPENDENT_ENDPOINT_PROTOCOL_VERSION,
    # [LIGAND_COM_RESTRAINT_PROTOCOL_VERSION=2] Group 5 配体 COM 限制力已移除。
    LIGAND_COM_RESTRAINT_PROTOCOL_VERSION,
    TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
    VDW_NONBONDED_PROTOCOL_VERSION,
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
    load_ibs_window_outputs_from_dir,
    _stage_window_sampling_identity,
    PRODUCTION_SEGMENT_PROTOCOL_VERSION,
    _load_validated_joint_score_ledgers,
    _assert_expected_windows_all_loaded,
    IBSControlPlaneError,
    IBSIncompleteStageCoverageError,
    _compute_ligand_net_charge,
    ibs_lj_tail_lrc_is_applicable,
    ibs_lj_tail_lrc_inapplicable_reason,
    # [MEM-00c] co-ion 身份：唯一的选择入口（此外任何地方都不许再选）。
    select_co_alchemical_ion_once,
    # [P0-REMD-CUDA] 显存读数唯一实现（走 nvidia-smi 子进程，绝不在本进程初始化 CUDA）。
    _gpu_memory_mib,
    # [Stage2 handoff] charge-transfer 配体的 vanishing 输入需要先走一遍
    # charging 配置（复用同一个生产入口，不重新发明）。
    configure_pme_ligand_charge_offsets,
    Exp019SeedLedger,
    EXP019_SEED_WIRING_PROTOCOL_VERSION,
    IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION,
)
from step_guard import guarded_step
from abfe_core import (
    frames_per_chunk,
    calculate_boresch_analytical_correction,
    constraint_identity_fingerprint,
    minimum_image_displacement_nm,
    ACESoftcorePotential,
    BeutlerSoftcoreBuilder,
    DEXPSurrogatePotential,
    UnitFormatter,
    TwoDimensionalLambdaPathPlanner,
    THERMODYNAMIC_CYCLE_DOC,
    combine_binding_free_energy,  # [ATT-09] 热力学循环闭合的唯一实现
    # [P1-12] final-result 统一 schema/sanity gate：cache 早退复用前必须过这道门。
    validate_final_leg_result,
    validate_boresch_attachment_result,
    FinalResultValidationError,
    # [2026-09-15] 精度只由**跨独立重复**的样本 SD 判定，唯一实现在 abfe_core。
    # 单次 run 结构上只有一个重复 ⟹ 恒为 UNMEASURED（不是达标也不是不达标）。
    cross_repeat_precision,
    PRECISION_UNMEASURED,
    PRECISION_MEETS,
    MIN_INDEPENDENT_REPEATS,
    # 膜体系协议（memtodolist §3.1/§3.2，B1）。默认 soluble，行为与改动前一致。
    # 注意：这里的 soluble/membrane 是**环境类型**，与本文件里既有的
    # `system_type="complex"|"solvent"`（腿身份）是两个互不相干的轴。
    ENVIRONMENT_TYPE_MEMBRANE,
    DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC,
    resolve_dispersion_protocol,
    # [P0-12a] 配体构象诊断（§3.0 末条）：度量与汇总的唯一实现在 abfe_core，
    # 本层只负责把 replica 轨迹切出来喂给它。
    ligand_conformer_metrics,
    ligand_conformer_summary,
    LIGAND_INTERNAL_POLAR_MIN_BOND_SEPARATION,
    CO_ALCHEMICAL_ION_ANCHOR_MIN_MASS_AMU,
    CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E,
    # §9 质量门：提取 + 判定 + 落盘的接线只有 `run_membrane_quality_gate` 一份，
    # 本文件不再直接调 `membrane_observables_from_trajectory` /
    # `evaluate_membrane_quality_gate`（那会变成第二套判据，见 §0.5.7 的教训）。
    run_membrane_quality_gate,
    MEMBRANE_QUALITY_GATE_MODE_ADVISORY,
    resolve_membrane_quality_gate_mode,
    # 预平衡轨迹时间轴：写轨迹的 reporter/integrator 与判 §9 的一侧共用这两个数。
    PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS,
    PRE_EQUILIBRATION_TIMESTEP_PS,
    pre_equilibration_frame_interval_ps,
    # 预平衡 block 收敛早停（PLAN 2026-08-26）。
    PRE_EQUILIBRATION_CONVERGENCE_MIN_STEPS,
    PRE_EQUILIBRATION_CONVERGENCE_BLOCK_STEPS,
    PRE_EQUILIBRATION_CONVERGENCE_STABLE_BLOCKS_REQUIRED,
    PRE_EQUILIBRATION_CONVERGENCE_TEMP_DRIFT_TOL_K,
    PRE_EQUILIBRATION_CONVERGENCE_VOLUME_DRIFT_TOL_PERCENT,
    PRE_EQUILIBRATION_CONVERGENCE_DENSITY_DRIFT_TOL_PERCENT,
    PRE_EQUILIBRATION_CONVERGENCE_PE_DRIFT_TOL_KJ_PER_MOL_PER_ATOM,
    PRE_EQUILIBRATION_CONVERGENCE_RMSD_DRIFT_TOL_NM,
    PRE_EQUILIBRATION_CONVERGENCE_POSE_DRIFT_TOL_NM,
    PRE_EQUILIBRATION_LIGAND_POSE_ESCAPE_TOL_NM,
    # 起始态体检（PE / max|F|）唯一实现；attachment 腿共用它。
    assert_starting_state_is_sane,
    # 整分子周期回卷：连通性只信 System，不信 topology 的键（唯一实现）。
    image_molecules_by_system,
    load_gromacs_topology_for_openmm,
    barostat_fingerprint_payload,
    ensure_barostat_for_protocol,
    resolve_membrane_protocol,
    # [MEM-00c] 只读核对冻结的 co-ion 身份（唯一实现，不在 pipeline 里另写一份）。
    verify_co_alchemical_ion_identity,
    co_alchemical_ion_cache_identity_payload,
    # [Stage2 handoff，2026-08-11] 把某个 GlobalParameter 在给定端点上的取值
    # 固化成静态 NonbondedForce 参数并彻底删除该参数——charge-transfer 配体的
    # charging→vanishing 交接就是靠它，不是靠"调用方记得设 λ=0"。
    bake_global_parameter_into_fixed_nonbonded_force,
    ensure_owned_system,
    CHARGE_TRANSFER_VANISHING_HANDOFF_PROTOCOL_VERSION,
    charge_treatment_qualification_payload,
    charge_transfer_result_markers,
    validate_charge_transfer_reservoir_correction,
)
import warnings

# [P0-01, 2026-08-30] v2 → v4：v2 会把配体内部普通 L–L 对补成显式 exception 来
# "冻结"内部库仑；但 OpenMM 的 exception 不走普通非键的 cutoff/PME 处理，这会
# 改写 λ=1 物理端点（tests/test_pme_decharge_endpoint_equivalence.py 实测复现）。
# v4 撤销补对并冻结 Stage1/Stage2 seam：既有 L–L exception 冻结、普通 L–L 对随粒子 offset 线性湮灭
# （annihilation 口径）。去电荷腿的逐腿 ΔG 数值会变（内部库仑湮灭项进腿），
# 但 complex/solvent 两腿内部 Hamiltonian 相同、湮灭项在 ΔG_bind 中严格相消；
# 协议身份必须换新，旧 Stage 1 artifact 不再与新结果混用。
PME_DECHARGE_MODEL_VERSION = "pme_decharge_v5_ll_exception_frozen_intramolecular_coulomb_decoupled_20260914"

# 膜体系预平衡的速度初始化种子。固定值使同一输入可复现；只影响膜路径
# （可溶路径不初始化速度，以保持既有生产基线逐位一致）。
MEMBRANE_EQUILIBRATION_VELOCITY_SEED = 20260730

# 膜预平衡的监控落盘间隔（步）。5000 步 = 10 ps，10 ns 会得到 1000 行，
# 足够看清体积/温度/能量往哪走，写盘开销可忽略。
MEMBRANE_EQUILIBRATION_MONITOR_INTERVAL = 5000

# 最小化后允许的最大单原子受力（kJ/mol/nm）。超过即判定"起始态坏了"并 fail closed。
#
# 定这个数的依据是实测：memtest 体系（45354 原子）**正常**时最小化后
# max|F| ≈ 2.5e3 且落在水上；而一个坏掉的 System（缓存里的 system_native.xml）
# 给出 max|F| = 3.7e9、PE = 4.1e13 —— 相差 6 个数量级。
# 1e6 留了 400 倍余量，正常体系不可能触发，坏体系必定触发。
#
# 为什么必须拦：坏的起始态会在几千步后变成一个**没有上下文的** `Particle
# coordinate is NaN`，然后要花好几轮去猜。拦在这里，报的是"起点就坏"，
# 并且能指出是哪个原子。
#
# 判门与报数的实现在 `abfe_core.assert_starting_state_is_sane()`（唯一一份，
# Boresch attachment 腿的起点体检共用它）。这里只是这个调用点的阈值别名 ——
# 值等于 `abfe_core.STARTING_STATE_MAX_FORCE_KJ_PER_MOL_NM`，改要一起改。
MEMBRANE_POST_MINIMIZATION_MAX_FORCE_KJ_PER_MOL_NM = 1.0e6

logger = logging.getLogger(__name__)

# 预优化协议版本只在探针 Hamiltonian、有限差分测量或路径优化算法发生
# 物理变化时递增。它不是 Python 源文件的 hash：在 abfe_core.py 增加与
# 预优化无关的 cache/provenance helper 不应让数小时级 pilot 失效。
PREOPT_HAMILTONIAN_PROTOCOL_VERSION = 2  # ligand 1-4 terms counted exactly once
GEODESIC_PATH_PROTOCOL_VERSION = 1
# The co-ion-aware pilot changed the charged probe Hamiltonian (native B3
# offsets/restraint plus PME in the pilot energy group).  Keep this version
# separate so neutral Atenolol preopt/geodesic caches remain reusable.
COION_PREOPT_HAMILTONIAN_PROTOCOL_VERSION = 1


# ═══════════════════════════════════════════════════════════════════════════
# 引擎控制面异常 → 控制器语义。**唯一注册表**（`EXIT_SPECS` 的对侧）
# ═══════════════════════════════════════════════════════════════════════════
#
# 🔑🔑🔑 [2026-09-18] **先前这是两处手写的 `except (A, B)` 元组，而引擎有 5 个
# 控制面异常。**
#
# 漏掉的那个是 `ExistingEnsembleRequiresRescueAudit` —— 它的消息自己写着
# `route it to rescue/provenance audit`，七个抛出点全部 fail-closed
# （`ibs_engine` 的 `load_ibs_state` 四处 + `run_all_windows` 三处），而且四条
# `non_mutating_v1` 守卫恰好是**自治模式的常态策略**。它落进 `except Exception`
# ⟹ `_finish("FAILED") + raise` ⟹ 一个**已知、可分类**的控制器结局被伪装成
# 执行器崩溃、整跑 traceback 收场。
#
# ⚠️ 补一个 dict **不足以**保证"下一个不会漏"：没登记的新异常照旧落进
# `except Exception`。所以配套在 `ibs_engine` 立了 `IBSControlPlaneError` 基类，
# 下面那道 import 期对账拿 `__subclasses__()` 跟本表逐项比 —— 缺一项立刻抛。
#
# ⚠️ **「刻意不路由」也必须登记**（`NOT_ROUTED`）。靠缺席表达"决定不路由"与
# "忘了登记"在代码里长得一模一样。同 `EXIT_SPECS` 里 `NO_FEASIBLE_ACTION` 的
# `default_scope=None`：强迫发出点做一次显式决定。
class EngineSignalSpec(NamedTuple):
    handler: str          # REDECIDE / SEAL_REFUTED_CANDIDATE / HALT / NOT_ROUTED
    signal: Optional[str]  # 记进 history 的 routing_signal
    exit_: Optional[str]   # HALT 档才有：控制器出口码（必须在 EXIT_SPECS 里）
    why: str


ENGINE_SIGNAL_SPECS: Dict[str, EngineSignalSpec] = {
    "IBSValidationBudgetIndeterminateError": EngineSignalSpec(
        "REDECIDE", "LOCAL_VALIDATION_CAP", None,
        "单周期验证批次打满、对这份 f_k 无结论 ⟹ 「没测出来」≠「f_k 错了」"
        "≠「该停」。回顶层重判，控制器有对症动作（PROVISIONAL_PRODUCTION）。"),
    "IBSWarmupConvergenceError": EngineSignalSpec(
        "REDECIDE", "WARMUP_F_K_NOT_CONVERGED", None,
        "一个 bias 压不平这个跨度 ⟹ 回顶层重判，证据类型→动作的映射由 "
        "`decide()` 负责，不在 handler 里发明。"),
    "IBSFrozenCalibrationValidationError": EngineSignalSpec(
        "SEAL_REFUTED_CANDIDATE", "FK_CANDIDATE_REFUTED", None,
        "有统计功效的否决 ⟹ 只终止**这份候选**，不终止整个 Stage-2："
        "封存候选与指纹、永不续验，然后回顶层重判。"),
    "ExistingEnsembleRequiresRescueAudit": EngineSignalSpec(
        "HALT", "EXISTING_ENSEMBLE_REQUIRES_RESCUE_AUDIT", "HALT_INVALID_INPUT",
        "盘上的 ensemble 与当前采样身份/协议不相容：不能复用、也不能原地覆盖，"
        "必须整份留给 rescue/provenance audit。"
        "⚠️ **不走 `REDECIDE`**：它不是「这一轮没测出来」，是「这条路径依赖的"
        "ensemble 身份已经不成立」。继续下一轮产不出新证据 —— 控制器会再发同一个"
        "动作、再撞同一个异常，盘面一字节不变 ⟹ no-op 台账把它挡掉 ⟹ 最后以 "
        "`NO_FEASIBLE_ACTION` 收场。**不是死循环，是把「身份无效」歪曲成「动作"
        "推不动」**，而那是人工接手时最需要分清的一件事。"
        "出口 `HALT_INVALID_INPUT` 的默认 scope 就是 `PATH_INVALID`"
        "（见 `EXIT_SPECS`：绕过去采下游无意义），所以也不会被退役换窗绕开。"),
    "IBSIncompleteStageCoverageError": EngineSignalSpec(
        "NOT_ROUTED", None, None,
        "**刻意不路由。** 缺窗在控制器侧有专门分支（缺窗 ⟹ RUN_PRODUCTION），"
        "而这个异常来自加载器：合并求解已经按段显式传 `excluded_local_windows` "
        "把「部分段」这个常态摘掉了，剩下真能触发它的情形就是「全体覆盖度不成立」"
        "—— 那必须炸，不许在缺口上求解一条截断的 ΔG。"
        "登记在这里是为了让「决定不路由」和「忘了登记」在代码里长得不一样。"),
}


def _assert_engine_signal_registry_complete() -> None:
    """引擎声明的控制面异常必须**逐个**在本表里有处置。缺一项立刻抛。

    这是 `EXIT_SPECS` 那套「漏项立即暴露」在异常这一侧的对应物。放在 import 期：
    它只可能因为**改代码**而失败（加了新异常没登记），不会因为盘面/数据失败。
    """
    declared = {c.__name__ for c in IBSControlPlaneError.__subclasses__()}
    registered = set(ENGINE_SIGNAL_SPECS)
    missing = sorted(declared - registered)
    stale = sorted(registered - declared)
    if missing or stale:
        raise RuntimeError(
            "引擎控制面异常的路由注册表与 `ibs_engine.IBSControlPlaneError` 的子类"
            f"集合不一致：未登记 {missing}、表里多余 {stale}。"
            "未登记的异常会静默落进 `except Exception`，被当成执行器崩溃 —— "
            "而它们表达的是**控制器的一个结局**。请在 `ENGINE_SIGNAL_SPECS` 里"
            "补一条（确实不该路由就登记 `NOT_ROUTED`，别靠缺席表达）。"
        )
    for _name, _spec in ENGINE_SIGNAL_SPECS.items():
        if _spec.handler == "HALT" and not _spec.exit_:
            raise RuntimeError(f"{_name} 是 HALT 档但没给 `exit_`")


_assert_engine_signal_registry_complete()


def _stage_lambda_endpoint_diagnostics(
    stage_name: str,
    lambdas_coul,
    lambdas_vdw,
) -> Dict:
    """Apply the endpoint contract for one half of the dual-lambda path."""
    # ``vanishing_rescue`` is an internal sampling label for a partial
    # replacement of failed vanishing windows.  Thermodynamically it is still
    # the vanishing leg, so it must use the same endpoint contract.
    diagnostic_stage_name = (
        "vanishing" if stage_name == "vanishing_rescue" else stage_name
    )
    expected = {
        "decharging": ((1.0, 1.0), (0.0, 1.0)),
        "vanishing": ((0.0, 1.0), (0.0, 0.0)),
    }
    if diagnostic_stage_name not in expected:
        raise ValueError(f"未知 dual-lambda stage: {stage_name!r}")
    expected_start, expected_end = expected[diagnostic_stage_name]
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


class _StdoutTeeToFile:
    """把裸 `print()` 也复制一份进 `pipeline.log`，不用逐个改 print 调用点。

    ## 为什么需要这个（2026-08-24）

    `ibs_engine.py`/`abfe_core.py`/`abfe_preoptimizer.py` 里绝大多数运行时输出走的是
    裸 `print()`，只到 stdout；`ABFEPipeline._log()` 才会同时写进 `pipeline.log`
    （`print(msg)` + 手动 `open(self.log_file, "a").write(...)` 两步都做，见下方
    `_log` 定义）。这正是 `tests/test_remd_gpu_context_budget.py` 记录过的那次真实
    事故的同一类问题（"决定只 print 到终端，归档日志里查不到"）——那次事故只在
    `REMDManager.__init__` 一处补了 `logger.warning`，全仓库其余 300+ 处裸 print()
    从来没有统一修过，归档的 `pipeline.log` 因此长期缺一大块运行时输出。

    这里选择在一个地方包一层 `sys.stdout`，而不是把 300+ 处裸 print() 逐一改成
    `self._log()`——那样的机械大改动风险远高于在一个地方接一层 tee，而且很多
    print() 调用点在 `ibs_engine.py`/`abfe_core.py` 里，根本拿不到
    `ABFEPipeline` 实例、无法直接调 `self._log()`。装好之后，任何后续 print()
    都会同时出现在终端和 `pipeline.log` 里，用跟 `_log()` 完全一致的
    `"%Y-%m-%d %H:%M:%S | 内容"` 格式，按行落盘（多行 write 会先缓冲，遇到
    `\n` 才落一行，避免半行被误打上时间戳）。
    """

    def __init__(self, original_stdout, log_path: str):
        self._original = original_stdout
        self._log_path = log_path
        self._line_buffer = ""
        # 🔑 [性能修复] 之前每一行都单独 open(path, "a")...write()...close()——
        # 一次 open/close 系统调用对。这个 tee 是进程级单例、装一次用到进程
        # 结束（见 _install_process_wide_log_file），所以直接常驻打开一个句柄
        # 复用，用 flush() 保持跟原来"每行都落盘可见"等价的崩溃可见性（原来
        # 的 with-block close 本身也不做 fsync，flush() 在这一点上是等价的）。
        # 句柄打开失败时保持原来的静默降级行为：不让日志落盘问题反过来打断
        # 终端输出。
        try:
            self._fh = open(self._log_path, "a", encoding="utf-8")
        except OSError:
            self._fh = None

    def write(self, s: str) -> int:
        n = self._original.write(s)
        # 🔑 [2026-09-09] 转发给真 stdout 的这一份也必须逐行 flush。
        #
        # `_append_line()` 对**文件**那份每行都 flush，而这里原本什么都不做 ——
        # stdout 被重定向到文件时是块缓冲（8 KB），于是同一批输出：
        #   `pipeline.log` 逐行可见，`launch.log`（= `> launch.log 2>&1`）落后
        #   几 KB，进程被杀/崩掉就整块丢，丢的正好是崩因那一段。
        # 实测 cyclod_ligand1/rep1：18:44:17→18:50:32 那 6 分钟（banner、PBC 修复、
        # Stage 0 attachment 逐行）在 pipeline.log 里一行不少，launch.log 里一行没有；
        # 两个文件 mtime 18:50:37 vs 18:53:30 也对得上。
        # 而 logging 走 stderr（不缓冲），所以只剩带时间戳的行能看见 —— 表现成
        # "print 的输出全没了"。
        if "\n" in s:
            try:
                self._original.flush()
            except (OSError, ValueError):
                # 落盘/终端写失败不能反过来打断流程，与 `_append_line` 同样降级。
                pass
        self._line_buffer += s
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._append_line(line)
        return n

    def _append_line(self, line: str) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {line}\n")
            self._fh.flush()
        except OSError:
            # 落盘失败不能反过来把终端输出也搞挂——静默丢这一行到文件，
            # 终端那一行已经通过上面 self._original.write(s) 正常写出去了。
            pass

    def retarget(self, log_path: str) -> None:
        """把这个 tee 指向新的日志文件。

        🔑 [0831issue P2] 同进程跑第二条腿时，旧代码用
        `if not isinstance(sys.stdout, _StdoutTeeToFile)` 短路，于是第二条腿**根本
        没装 tee**：它的裸 print() 全部继续写进第一条腿的 `pipeline.log`，第二条腿
        自己的日志文件几乎是空的。这里改成"已经是我们的 tee 就换目标"，既不套娃
        （套娃会让每行在多个文件里重复），也不丢腿。
        """
        if log_path == self._log_path and self._fh is not None:
            return
        # 换文件前把半行缓冲落到**旧**文件里，避免一行被切到两个文件。
        if self._line_buffer:
            self._append_line(self._line_buffer)
            self._line_buffer = ""
        self.close()
        self._log_path = log_path
        try:
            self._fh = open(self._log_path, "a", encoding="utf-8")
        except OSError:
            self._fh = None

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def flush(self) -> None:
        self._original.flush()
        if self._fh is not None:
            try:
                self._fh.flush()
            except OSError:
                pass

    def isatty(self) -> bool:
        return bool(getattr(self._original, "isatty", lambda: False)())

    def __getattr__(self, name):
        return getattr(self._original, name)


_PROCESS_WIDE_LOG_FILE: Optional[str] = None
# [0831issue P2] 进程级单例句柄，供 _install_process_wide_log_file 换目标时摘旧挂新。
_PROCESS_WIDE_LOG_HANDLER: Optional[logging.Handler] = None
_PROCESS_WIDE_LOG_TEE: Optional["_StdoutTeeToFile"] = None


def _install_process_wide_log_file(log_file_path: str) -> None:
    """把这个进程往后的所有输出（`logging` 模块 + 裸 `print()`）都接进同一份
    `pipeline.log`，只装一次，同一个文件重复调用是空操作。

    两个来源分别接：
    1. `logging` 模块（`runabfe.py` 的 `log.info`/`log.warning` 等）默认只有
       `logging.basicConfig()` 装的 `StreamHandler`（写到 **stderr**，不是
       stdout，也不是任何文件）——额外挂一个 `FileHandler` 在根 logger 上，
       所有子 logger（`"runabfe"` 等）不设自己的 handler 时都会往根传播，
       挂在根上这一份就能接住全部。
    2. 裸 `print()`（`ibs_engine.py` 等）只到 stdout——用上面的
       `_StdoutTeeToFile` 包一层。
    """
    global _PROCESS_WIDE_LOG_FILE, _PROCESS_WIDE_LOG_HANDLER, _PROCESS_WIDE_LOG_TEE
    if _PROCESS_WIDE_LOG_FILE == log_file_path:
        return

    # 🔑 [0831issue P2] 同进程第二条腿（abfe_core 会在同一进程里新建第二个
    # pipeline）以前会踩两个坑：
    #   (a) `logging.getLogger().addHandler(...)` 每次调用都**追加**一个 FileHandler，
    #       旧的从不摘掉 —— 第 N 条腿的每行 logging 输出会被写 N 遍，而且第一条腿的
    #       日志文件会一直继续收第二条腿的行；
    #   (b) stdout tee 用 `isinstance` 短路，第二条腿根本不装 tee，它的裸 print()
    #       全进了第一条腿的文件，自己的 `pipeline.log` 几乎是空的。
    # 现在两者都按"进程级单例 + 换目标"处理：先摘旧 handler 再挂新的，tee 直接
    # retarget（不套娃）。
    root_logger = logging.getLogger()
    if _PROCESS_WIDE_LOG_HANDLER is not None:
        try:
            root_logger.removeHandler(_PROCESS_WIDE_LOG_HANDLER)
            _PROCESS_WIDE_LOG_HANDLER.close()
        except Exception:
            pass
        _PROCESS_WIDE_LOG_HANDLER = None

    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)
    _PROCESS_WIDE_LOG_HANDLER = file_handler

    if isinstance(sys.stdout, _StdoutTeeToFile):
        sys.stdout.retarget(log_file_path)
        _PROCESS_WIDE_LOG_TEE = sys.stdout
    else:
        tee = _StdoutTeeToFile(sys.stdout, log_file_path)
        sys.stdout = tee
        _PROCESS_WIDE_LOG_TEE = tee
        # 常驻句柄需要在进程退出时显式收尾（原来的每行 open/close 不需要，
        # 因为从不跨语句持有句柄）；atexit 保证正常退出路径都会 flush+close。
        # 只注册一次：retarget 路径复用同一个对象，重复注册会让 close() 被调多次。
        atexit.register(tee.close)

    _PROCESS_WIDE_LOG_FILE = log_file_path


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _system_xml_hash(system: Optional[openmm.System]) -> Optional[str]:
    # ===================== [2026-09-09] 已删除 =====================
    # 恒返回 None。此前它哈希的是**我们自己生成的中间产物 / 本仓库代码**，不是
    # 用户输入。每次它进入缓存身份，结果都一样：我们改一行代码、或做一个按设计
    # 修坐标的修复（PBC-01），用户已经烧掉的 GPU 时间就全部作废。
    #
    # 复发记录：code_sha256（2026-08-24）、system_xml_hash + positions_sha256
    # （2026-09-09，cyclod_ligand1/rep1 实测：pipeline 前一秒说"预平衡已完成、
    # 跳过重跑"，后一秒说"拒绝判定为已完成"）。
    #
    # 返回 None ⟹ 所有 payload 该字段为 None、所有比较 None == None 恒真，
    # 不再作废任何缓存，也不会 KeyError。这些 payload 本来就有显式协议版本号
    # 承担"算法变了"的信号，sha256 是多余的第二套机制。
    # 用户输入（gro/top/ligand）的绑定在 runabfe._sha256_file，与此无关。
    return None
    # ===============================================================
    if system is None:
        return None
    return _sha256_text(XmlSerializer.serialize(system))


def _topology_hash(topology: Optional[app.Topology]) -> Optional[str]:
    # ===================== [2026-09-09] 已删除 =====================
    # 恒返回 None。此前它哈希的是**我们自己生成的中间产物 / 本仓库代码**，不是
    # 用户输入。每次它进入缓存身份，结果都一样：我们改一行代码、或做一个按设计
    # 修坐标的修复（PBC-01），用户已经烧掉的 GPU 时间就全部作废。
    #
    # 复发记录：code_sha256（2026-08-24）、system_xml_hash + positions_sha256
    # （2026-09-09，cyclod_ligand1/rep1 实测：pipeline 前一秒说"预平衡已完成、
    # 跳过重跑"，后一秒说"拒绝判定为已完成"）。
    #
    # 返回 None ⟹ 所有 payload 该字段为 None、所有比较 None == None 恒真，
    # 不再作废任何缓存，也不会 KeyError。这些 payload 本来就有显式协议版本号
    # 承担"算法变了"的信号，sha256 是多余的第二套机制。
    # 用户输入（gro/top/ligand）的绑定在 runabfe._sha256_file，与此无关。
    return None
    # ===============================================================
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


# 🔑 [2026-09-01] 坐标进指纹时必须先量化。`_positions_hash` 哈希的是原始 float64
# 字节，**1e-15 的差也换哈希**：同一套坐标经 GRO 读入与经 mmCIF 缓存读回相差
# ~1.8e-15（实测），于是首跑与 resume 的指纹永远不同。0.001 Å 远细于任何真实的
# "换了个 pose/盒子"，但足以吸收往返浮点噪声。
# ⚠️ 诚实的局限：量化有**边界**——若某个坐标恰好落在舍入边界上，1e-15 的噪声也能把它
# 推过去、从而改变哈希。这不是保证，是概率：以实测噪声 ~1.8e-15、量子 1e-4 nm、
# 5 万原子计，单次比较误失配概率约 1e-15/1e-4 × 1.5e5 ≈ 1.5e-6。
# 相对"当前 100% 必然失配"是决定性改善，但**不要当成"永不误判"**。
# 1e-4 nm = 0.001 Å，仍比任何真实的"换了个 pose"细两个量级（实测 0.002 Å 就能抓到）。
# 真要确定性，得存坐标而非存哈希、按容差比较，那是另一个量级的改动。
_PRE_EQUIL_IDENTITY_COORD_DECIMALS_NM = 4


def _quantize_positions_for_identity(positions):
    if positions is None:
        return None
    try:
        arr = (
            np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
            if hasattr(positions, "value_in_unit")
            else np.asarray(positions, dtype=np.float64)
        )
    except Exception:
        return positions
    return np.round(arr, _PRE_EQUIL_IDENTITY_COORD_DECIMALS_NM) * unit.nanometer


def _pre_equilibration_fingerprint(
    system: Optional[openmm.System],
    ligand_indices: Optional[List[int]],
    temperature,
    pressure=None,
    positions=None,
    box_vectors=None,
    requested_steps: Optional[int] = None,
    barostat_protocol: Optional[Dict] = None,
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
        # [2026-09-09] `system_xml_hash` 已从身份里移除。
        #
        # 它哈希的是**当次在内存里重建出来的 System 的序列化字节**，而
        # `save_native_system()`（runabfe.py:4810）是**无条件**调用的：每次非
        # cache-only 运行都会重建并覆盖 system_native.xml。于是任何只改字节、
        # 不改物理的改动都会让它翻脸 —— 例如把排除表灌入顺序从 set 序换成
        # `sorted(missing)`（能量逐比特不变）。后果是 resume 被判定失配。
        #
        # 这与 `code_sha256` 那次是同一个形状：已经有语义正确的完成信号
        # （`pipeline_state.json` 的 `equilibration.status == "completed"`，
        # 见 runabfe.py `equilibrium_is_done()` 里自己的注释），再叠一套按字节
        # 比对的机制是多余的第二套判据，而且是会误伤的那套。
        #
        # 真正的身份仍然在下面：配体索引、温度、压力、初始坐标、盒、目标步数、
        # barostat 协议 —— 换 pose / 换盒 / 换温度 / 换步数都仍然会失配。
        "ligand_indices": sorted(int(i) for i in (ligand_indices or [])),
        "temperature_K": round(float(temp_k), 6),
        "pressure_bar": round(float(pressure_bar), 6) if pressure_bar is not None else None,
        # [2026-09-09] `positions_sha256` / `box_vectors_sha256` 已从身份里移除。
        #
        # 坐标是**可变状态**，而且本项目自己会合法地改它：PBC-01（同日）按设计把被
        # 假键撕开的分子拼回去，连带 ~0.005 nm 的整体质心平移。量化只挡得住 1e-15
        # 的往返浮点噪声，挡不住这种真实位移 ⟹ **PBC-01 一上线，所有已有 run 目录的
        # 预平衡指纹全部失配、resume 全部失效。** 这不是假设，是 2026-09-09
        # cyclod_ligand1/rep1 的实测（日志里 18:00:28 刚说"基线预平衡已完成，跳过重跑"，
        # 18:00:29 就说"拒绝把它判定为已完成"）。
        #
        # 更根本的问题：把坐标/序列化字节塞进缓存身份，等于让**任何修 bug 的改动**
        # 都作废用户已经烧掉的 GPU 时间。同一个形状已经反复出现（`code_sha256`、
        # `system_xml_hash`）。不要再往身份里加 sha256。
        #
        # 换体系/换 pose 由 `ligand_indices` + `requested_steps` + barostat 协议，
        # 以及 runabfe 侧的 `system_cache_exists()`（绑定当前 gro/top/ligand 输入）
        # 负责；真正的"跑完没有"只看 pipeline_state.json 的 equilibration.status。
        "requested_steps": int(requested_steps) if requested_steps is not None else None,
    }
    # memtodolist §3.2：barostat 类型/压力/表面张力/XY-Z 模式/频率进入预平衡
    # fingerprint，改任一项都必须让旧 checkpoint 失效。
    #
    # 但 §7.7 同时要求"不传 system_type 时行为与改动前逐位一致"。这两条靠
    # barostat_fingerprint_payload() 调和：它对 legacy soluble 协议
    # （MonteCarloBarostat + 频率 25，即改动前唯一存在的协议）返回 None，
    # 于是**一个键都不加**，json.dumps 的结果与改动前逐字节相同，已有的生产预平衡
    # checkpoint 继续有效。只要 system_type 变成 membrane 或频率偏离 25，
    # payload 非 None，指纹立刻改变，旧 checkpoint 自动失效。
    barostat_payload = barostat_fingerprint_payload(barostat_protocol)
    if barostat_payload is not None:
        payload["barostat_protocol"] = barostat_payload
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
    # ===================== [2026-09-09] 已删除 =====================
    # 恒返回 None。此前它哈希的是**我们自己生成的中间产物 / 本仓库代码**，不是
    # 用户输入。每次它进入缓存身份，结果都一样：我们改一行代码、或做一个按设计
    # 修坐标的修复（PBC-01），用户已经烧掉的 GPU 时间就全部作废。
    #
    # 复发记录：code_sha256（2026-08-24）、system_xml_hash + positions_sha256
    # （2026-09-09，cyclod_ligand1/rep1 实测：pipeline 前一秒说"预平衡已完成、
    # 跳过重跑"，后一秒说"拒绝判定为已完成"）。
    #
    # 返回 None ⟹ 所有 payload 该字段为 None、所有比较 None == None 恒真，
    # 不再作废任何缓存，也不会 KeyError。这些 payload 本来就有显式协议版本号
    # 承担"算法变了"的信号，sha256 是多余的第二套机制。
    # 用户输入（gro/top/ligand）的绑定在 runabfe._sha256_file，与此无关。
    return None
    # ===============================================================
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
    # ===================== [2026-09-09] 已删除 =====================
    # 恒返回 None。此前它哈希的是**我们自己生成的中间产物 / 本仓库代码**，不是
    # 用户输入。每次它进入缓存身份，结果都一样：我们改一行代码、或做一个按设计
    # 修坐标的修复（PBC-01），用户已经烧掉的 GPU 时间就全部作废。
    #
    # 复发记录：code_sha256（2026-08-24）、system_xml_hash + positions_sha256
    # （2026-09-09，cyclod_ligand1/rep1 实测：pipeline 前一秒说"预平衡已完成、
    # 跳过重跑"，后一秒说"拒绝判定为已完成"）。
    #
    # 返回 None ⟹ 所有 payload 该字段为 None、所有比较 None == None 恒真，
    # 不再作废任何缓存，也不会 KeyError。这些 payload 本来就有显式协议版本号
    # 承担"算法变了"的信号，sha256 是多余的第二套机制。
    # 用户输入（gro/top/ligand）的绑定在 runabfe._sha256_file，与此无关。
    return None
    # ===============================================================
    box = _box_vectors_nm_array(box_vectors)
    if box is None:
        return None
    return hashlib.sha256(np.ascontiguousarray(box).tobytes()).hexdigest()


def _minimum_image_displacement(displacement, box_vectors) -> np.ndarray:
    """Compatibility wrapper around the single closest-lattice implementation."""
    delta = np.asarray(displacement, dtype=np.float64)
    box = _box_vectors_nm_array(box_vectors)
    if box is None:
        raise ValueError("周期体系缺少 box vectors，无法计算 Boresch minimum-image 距离")
    return minimum_image_displacement_nm(delta, box)


_CODE_HASH_CACHE: Optional[str] = None
_DEBUG_CODE_HASH_WARNED = False


def _debug_code_hash_frozen() -> Optional[str]:
    """开发/调试专用逃生舱：设置环境变量 ABFE_DEBUG_FREEZE_CODE_HASH=1 时，
    `_code_hash()` 返回固定常量，而不是真的
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
            "  [ERR] [DEBUG] ABFE_DEBUG_FREEZE_CODE_HASH=1：code_sha256 已冻结为固定常量，"
            "不反映当前磁盘上的代码改动——仅供调试收敛逻辑时使用，正式出结果前必须"
            "取消设置这个环境变量并至少完整跑一次真正的哈希校验。"
        )
        _DEBUG_CODE_HASH_WARNED = True
    return "DEBUG_CODE_HASH_FROZEN_ABFE_DEBUG_FREEZE_CODE_HASH"


def _code_hash() -> str:
    # ===================== [2026-09-09] 已删除 =====================
    # 恒返回 None。此前它哈希的是**我们自己生成的中间产物 / 本仓库代码**，不是
    # 用户输入。每次它进入缓存身份，结果都一样：我们改一行代码、或做一个按设计
    # 修坐标的修复（PBC-01），用户已经烧掉的 GPU 时间就全部作废。
    #
    # 复发记录：code_sha256（2026-08-24）、system_xml_hash + positions_sha256
    # （2026-09-09，cyclod_ligand1/rep1 实测：pipeline 前一秒说"预平衡已完成、
    # 跳过重跑"，后一秒说"拒绝判定为已完成"）。
    #
    # 返回 None ⟹ 所有 payload 该字段为 None、所有比较 None == None 恒真，
    # 不再作废任何缓存，也不会 KeyError。这些 payload 本来就有显式协议版本号
    # 承担"算法变了"的信号，sha256 是多余的第二套机制。
    # 用户输入（gro/top/ligand）的绑定在 runabfe._sha256_file，与此无关。
    return None
    # ===============================================================
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


PROTOCOL_FINGERPRINT_SCHEMA_VERSION = 1


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


def _protocol_fingerprint_ignoring_code_hash(key):
    """比较两个 `_protocol_fingerprint(...)` 结果时忽略其中的 `code_sha256`。

    ## 为什么需要这个共享 helper（2026-08-24 事故）

    "不相关的 Python 文件改动不该让 resume 强制重算"这条理由，在这个文件里
    已经独立出现、独立修过至少两次——stage0 attachment（本文件另一处，`payload`
    里 pop 掉 `code_sha256` 后重算 `sha256`）和 `_pme_u_kn_meta_matches`（同样
    pop 掉再比较，注释原话："the old global code hash caused every resume
    after any pipeline fix to recompute u_kn from all DCD frames"）。

    但 `_build_top_level_protocol_key`/stage1·stage2 的 `_build_stage_protocol_key`/
    traditional REMD 的 sampling fingerprint 这三处都各自漏掉了同一个修法，
    直接拿含 `code_sha256` 的 payload 做 `==`/`!=` 比较。2026-08-24 这次就是
    因为一次跟这几处协议指纹完全无关的代码修复（`ibs_engine.py` 里一处吞异常的
    bug）改了 `abfe_core.py`/`abfe_pipeline.py`/`ibs_engine.py` 的文件内容，
    `code_sha256` 随之改变，导致用户正在跑的 production resume 在这三处
    全部判定"协议指纹不匹配"，整段重新校验/运行——这正是前两次修复的注释里
    明确警告过的同一个失效模式，只是换了三个新的触发点。

    这里做成一个共享 helper，而不是继续在新的比较点各自重新发明一次同样的
    "pop payload 里的 code_sha256、重算 sha256" 逻辑——那正是这次会漏掉的原因：
    同一件事有多处独立实现，补一个漏一片。

    只对**存在** `payload["code_sha256"]` 的一侧生效；不存在时原样返回，
    不引入任何变化（对没有这个字段的新写法是无操作）。
    """
    if not isinstance(key, dict):
        return key
    payload = key.get("payload")
    if not isinstance(payload, dict) or "code_sha256" not in payload:
        return key
    stripped_payload = dict(payload)
    stripped_payload.pop("code_sha256", None)
    stripped_key = dict(key)
    stripped_key["payload"] = stripped_payload
    stripped_key["sha256"] = _sha256_text(
        json.dumps(
            _canonical_protocol_value(stripped_payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return stripped_key


class _Missing:
    """`_walk` 里表示"这一侧根本没有这个键"的哨兵。

    不用字符串 `"<missing>"`：那样一份真的存了字符串 `"<missing>"` 的 payload 会跟
    "键不存在"无法区分，而且 `repr` 出来带引号、读日志的人会以为值就是那串字。
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<缺失>"


_MISSING = _Missing()

_MISMATCH_VALUE_REPR_LIMIT = 120


def _brief_mismatch_value(value: Any, limit: int = _MISMATCH_VALUE_REPR_LIMIT) -> str:
    """把一个 diff 叶子压成**有界长度**的字符串。

    没有这一层，一条"缓存作废、重新跑"的日志能吐出几页文本。成因是 `_walk` 只在
    **两边都是 dict** 时才递归：只要有一边整个缺了某个键（典型是旧缓存里有
    `boresch_params.diagnostics`、新指纹里没有），另一边那棵完整的字典就会落到
    `a != b` 分支、被当成叶子原样 `repr` 出来。实测那条日志 1151 字符，而两边都有
    该键时只有 72 字符。

    截断后给的是**结构摘要**（有几个键 / 几项、键名前几个）而不是掐断的 `repr`：
    掐断的字典字面量既读不出全貌又不能复制粘贴，结构摘要至少能回答"是整块没了，
    还是里面某个字段变了"。完整 payload 仍逐字留在磁盘缓存文件里。
    """
    text = repr(value)
    if len(text) <= limit:
        return text
    if isinstance(value, dict):
        keys = sorted(map(str, value))
        head = ", ".join(keys[:6])
        more = f", …共 {len(keys)} 个键" if len(keys) > 6 else ""
        return f"<dict {len(keys)} 键: {head}{more}>"
    if isinstance(value, (list, tuple)):
        return f"<{type(value).__name__} {len(value)} 项>"
    return text[: max(1, limit - 1)] + "…"


def _summarize_protocol_mismatch(
    cached: Optional[Dict], current: Dict, max_diffs: int = 8
) -> str:
    """把两份 `_protocol_fingerprint(...)` 结果的差异压缩成一行摘要，而不是把
    整棵 payload（可能带着 Boresch harmonicity 诊断那种上千字符的嵌套 list）
    原样甩进日志。

    2026-08-27 之前，三处"协议指纹不匹配"的日志都是直接
    `f"缓存={cached_key}, 当前={current_key}"`——`cached_key`/`current_key`
    整个是 `{'schema_version', 'sha256', 'payload': {...}}`，`payload` 里嵌着
    完整的 Boresch 诊断分布（每个自由度的 fluctuation_distribution 明细）。
    结果就是一条"缓存作废，重新跑"的日志能吐出几页文本，读的人还是看不出
    到底是哪个字段变了。

    这里只递归 diff 两边 `payload` 里实际不同的叶子路径，值相同的子树完全
    跳过；超过 `max_diffs` 条差异时截断并注明还剩多少条没列出来。完整的
    payload 仍然逐字保留在磁盘上的缓存文件（`final_results.json` 等）里，
    "可审计"这一点没有丢——只是不再无差别地塞进控制台日志。
    """
    if not isinstance(cached, dict):
        return f"缓存协议指纹缺失或格式异常（{cached!r}）"

    def _walk(path: Tuple[str, ...], a: Any, b: Any, out: List[Tuple[str, Any, Any]]) -> None:
        if len(out) >= max_diffs:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if len(out) >= max_diffs:
                    return
                _walk(path + (str(key),), a.get(key, _MISSING), b.get(key, _MISSING), out)
        elif (
            isinstance(a, (list, tuple))
            and isinstance(b, (list, tuple))
            and len(a) == len(b)
        ):
            # 也要进 list：`boresch_harmonicity_check.fluctuation_distribution` 就是
            # 一个 list of dict，不进去的话一有差异就把整段甩进日志（而且指不出是哪个
            # 自由度的哪个统计量变了）。长度不同就没法逐项对齐，退化成只报长度。
            for index, (item_a, item_b) in enumerate(zip(a, b)):
                if len(out) >= max_diffs:
                    return
                _walk(path + (f"[{index}]",), item_a, item_b, out)
        elif a != b:
            out.append((".".join(path) or "<root>", a, b))

    diffs: List[Tuple[str, Any, Any]] = []
    _walk((), cached.get("payload", cached), current.get("payload", current), diffs)

    if not diffs:
        # payload 逐叶子比对完全一样，说明差异只可能在 schema_version 或哈希
        # 序列化逻辑本身，不是物理内容变了——这种情况本身就值得单独说清楚。
        return (
            "payload 内容逐项比对一致，但顶层 sha256 不同"
            f"（缓存={cached.get('sha256')!r}, 当前={current.get('sha256')!r}）"
            "——多半是 schema_version 或哈希序列化逻辑变了，不是物理内容变了"
        )

    truncated = len(diffs) >= max_diffs
    lines = [
        f"{p}: 缓存={_brief_mismatch_value(a)} -> 当前={_brief_mismatch_value(b)}"
        for p, a, b in diffs
    ]
    if truncated:
        lines.append(f"...（已截断，只列出前 {max_diffs} 条差异）")
    return "; ".join(lines)


_PREOPT_EQ_KEYS = (
    "r0", "thetaA0", "thetaB0", "phiA0", "phiB0", "phiC0",
)
_PREOPT_FC_KEYS = (
    "kr", "kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC",
)
_PREOPT_EQ_ALIASES = {
    **{key: key for key in _PREOPT_EQ_KEYS},
    "r0_nm": "r0",
    "thetaA0_rad": "thetaA0",
    "thetaB0_rad": "thetaB0",
    "phiA0_rad": "phiA0",
    "phiB0_rad": "phiB0",
    "phiC0_rad": "phiC0",
}
_PREOPT_FC_ALIASES = {
    **{key: key for key in _PREOPT_FC_KEYS},
    "kr_kJ_mol_nm2": "kr",
    "kthetaA_kJ_mol_rad2": "kthetaA",
    "kthetaB_kJ_mol_rad2": "kthetaB",
    "kphiA_kJ_mol_rad2": "kphiA",
    "kphiB_kJ_mol_rad2": "kphiB",
    "kphiC_kJ_mol_rad2": "kphiC",
}


def _preopt_boresch_protocol_payload(boresch_params: Optional[Dict]) -> Optional[Dict]:
    """Return only Boresch fields that can identify the restraint Hamiltonian.

    Diagnostics, scores, provenance and timestamps are deliberately excluded:
    changing an explanation of the same restraint must not rerun λ pilot scans.
    """
    if not isinstance(boresch_params, dict):
        return None
    anchors = boresch_params.get("boresch_anchors", boresch_params)
    if not isinstance(anchors, dict):
        return None

    rec_idx = anchors.get("receptor_indices") or boresch_params.get(
        "receptor_indices", []
    )
    lig_idx = anchors.get("ligand_indices") or boresch_params.get(
        "ligand_indices", []
    )
    eq = anchors.get("equilibrium_values") or boresch_params.get(
        "equilibrium_values", {}
    )
    fc = anchors.get("force_constants") or boresch_params.get(
        "force_constants", {}
    )
    if not isinstance(eq, dict) or not isinstance(fc, dict):
        return None

    def _normalize(source, aliases):
        result = {}
        for raw_key, value in source.items():
            key = aliases.get(str(raw_key))
            if key is not None and key not in result:
                result[key] = value
        return result

    payload = {
        "receptor_indices": [int(i) for i in rec_idx],
        "ligand_indices": [int(i) for i in lig_idx],
        "equilibrium_values": _normalize(eq, _PREOPT_EQ_ALIASES),
        "force_constants": _normalize(fc, _PREOPT_FC_ALIASES),
    }
    if not any(payload[key] for key in (
        "receptor_indices", "ligand_indices", "equilibrium_values", "force_constants"
    )):
        return None
    return payload


def _protocol_key_differences(cached_protocol: Optional[Dict], current_protocol: Dict) -> List[str]:
    """Return concise, recursive field differences for cache diagnostics."""
    missing = object()
    old = cached_protocol.get("payload") if isinstance(cached_protocol, dict) else None
    new = current_protocol.get("payload") if isinstance(current_protocol, dict) else None
    if not isinstance(old, dict) or not isinstance(new, dict):
        return ["protocol payload: <missing or invalid> -> current fingerprint"]

    differences: List[str] = []

    def _walk(path: str, left, right):
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right), key=str):
                _walk(
                    f"{path}.{key}" if path else str(key),
                    left.get(key, missing),
                    right.get(key, missing),
                )
            return
        if left is missing or right is missing or left != right:
            left_text = "<missing>" if left is missing else repr(left)
            right_text = "<missing>" if right is missing else repr(right)
            differences.append(f"{path}: {left_text} -> {right_text}")

    _walk("", old, new)
    return differences


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
    env_seed_keys = (
        "OPENMM_RANDOM_SEED",
        "ABFE_RANDOM_SEED",
        "IBS_RANDOM_SEED",
        "PYTHONHASHSEED",
    )
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
    coion_identity: Optional[Dict] = None,
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
    payload = {
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
    if coion_identity is not None:
        payload["coion_identity"] = coion_identity
    return payload


# 🔑 [ESTIMATOR_ANALYSIS_PROTOCOL_VERSION=2, 2026-07-28] charging（decharging 腿）
# 的估计量口径。**只用于 stage1 = charging**，vdW/stage2 与此无关（那条腿只能用
# TMBAR，见 ibs_engine.ESTIMATOR_ANALYSIS_PROTOCOL_VERSION 的注释）。
#
# 主值改成相邻 BAR 的依据（2026-07-28 实测同一批 u_kn）：
#   相邻 BAR 65.0762±0.6148 / 重加权 FD-TI 65.1262 / 全帧 MBAR 65.0032 三者一致，
#   而**去相关 MBAR 64.4113** 偏低 0.6649；溶剂腿同样偏低 0.5252。
#   根因是自相关子采样在有限样本下选帧不稳定（丢帧后那个子集不稳），**不是 MBAR
#   有偏**。BAR 逐边只用相邻两态自己的样本，不受全局选帧影响。
# ⚠️ 两条腿的偏移基本抵消：净移动只有 (0.5252−0.6649)/4.184 = **−0.033 kcal/mol**。
#   这是个真问题，但**不是** result.txt 那 1.3 kcal charging 缺口的解释。
CHARGING_TI_GATE_TOL_KJ_MOL = 0.5


def _CHARGING_ESTIMATOR_KWARGS(lambdas_coul: Sequence[float]) -> Dict:
    """decharging 腿两处 `analyzer.solve()` 共用的估计量参数。

    抽成一个函数而不是在两处各写一遍：缓存命中路径与正常路径必须用**完全相同**的
    口径，否则 resume 与首次运行会报出两个不同定义的 ΔG。

    `ti_gate_tolerance_kJ_mol` 必须显式给：实测 |BAR−FD-TI| = 0.050（复合物）/
    0.226（溶剂）kJ/mol。0.5 是绝对下限；实际判据是
    `max(0.5, 3*sigma_BAR)`，当前两腿约为 1.84 / 1.98 kJ/mol。
    **刻意不沿用 attachment 的 `ATTACHMENT_BAR_TI_ABS_TOL_KJ = 1.0`**——两条腿的
    量级与 TI 变体都不同。
    """
    return dict(
        primary_estimator="adjacent_bar",
        # 真实（非均匀）λ 表。FD-TI 的中心差分必须用它，不能用态序号。
        lambdas=[float(x) for x in lambdas_coul],
        ti_gate_tolerance_kJ_mol=CHARGING_TI_GATE_TOL_KJ_MOL,
        # 四个口径全部落盘：主值换了，旧口径必须留下可比的痕迹。
        crosschecks=True,
    )


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
    coion_identity: Optional[Dict] = None,
) -> bool:
    meta_path = _pme_u_kn_meta_path(stage_output_dir, stage_name)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        expected = _pme_u_kn_meta_payload(
            n_states=n_states,
            lambdas_coul=lambdas_coul,
            lambdas_vdw=lambdas_vdw,
            temperature_k=temperature_k,
            system=system,
            topology=topology,
            ligand_indices=ligand_indices,
            boresch_params=boresch_params,
            coion_identity=coion_identity,
        )
        # The explicit PME model version is the cache invalidation contract.
        # Do not invalidate expensive offline energy caches merely because an
        # unrelated Python file changed (the old global code hash caused every
        # resume after any pipeline fix to recompute u_kn from all DCD frames).
        # Keep accepting legacy metadata that still contains the old hash.
        meta.pop("code_sha256", None)
        expected.pop("code_sha256", None)
        if coion_identity is not None and not isinstance(
            meta.get("coion_identity"), dict
        ):
            return False
        return meta == expected
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
    coion_identity: Optional[Dict] = None,
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
                coion_identity=coion_identity,
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
    """`pipeline_state.json` 的短时互斥锁。

    🔑 [RELEASE_READINESS R2 + 0831issue P2] 三处修正，全部朝"只在**能证明**锁已经
    没有主人时才敢删"的方向收紧：

    1. **`PermissionError` 分支原来不可达。** 它是 `OSError` 的子类，写在 `except
       OSError` 之后永远轮不到，于是"权限不足、无法判断"被当成"进程已退出"，
       stale-lock 清理可能删掉仍在使用的锁。现在按 `ProcessLookupError`（确认不
       存在）/ `PermissionError`（存在但不属于本用户）/ 其它 `OSError`（判不了）
       三类分开处理，只有第一类算 stale。
    2. **建锁与写 PID 之间有竞态。** `O_CREAT|O_EXCL` 成功之后才写 PID，另一个进程
       在这个窗口里读到空文件会得出 `pid=-1` 并把 A 的锁删掉 → 双持锁。现在读到
       空/不可解析的 payload 时不立即判 stale，而是要求该文件的 mtime 已经老于
       `_EMPTY_PAYLOAD_GRACE_S`；正常竞态窗口是微秒级，宽限期足以覆盖。
    3. **共享文件系统上 PID 不可跨节点比较。** payload 改为 JSON
       `{"pid": ..., "hostname": ...}`（仍然兼容旧的裸整数写法）。hostname 与本机
       不一致时一律视为"还活着"，绝不删别的节点的锁。

    这些都不改变正常路径的行为（无竞争时照常获取/释放），只改变"判不了"时的默认。
    """

    # 建锁与写 payload 之间的竞态窗口是微秒级；给足宽限期，避免把正在建的锁
    # 当成残留锁删掉。同时仍然能清理"进程建了锁就被 kill -9"留下的空文件。
    _EMPTY_PAYLOAD_GRACE_S = 30.0

    def __init__(self, path: str, timeout_s: float = 10.0, poll_s: float = 0.05):
        self.path = path
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.fd = None

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        """只有**确认该 PID 不存在**时才返回 False；判不了一律 True（保守）。"""
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            # 唯一能确认"这个 PID 已经没了"的信号。
            return False
        except PermissionError:
            # 进程存在，只是不属于本用户 —— 必须算活着。
            return True
        except OSError:
            # 其它 errno（EINVAL 等）说明我们判不了，不能据此删锁。
            return True
        return True

    @staticmethod
    def _own_hostname() -> str:
        try:
            return socket.gethostname()
        except Exception:
            return ""

    def _lock_payload(self) -> str:
        return json.dumps(
            {"pid": int(os.getpid()), "hostname": self._own_hostname()},
            ensure_ascii=False,
        )

    def _read_lock_owner(self) -> Tuple[Optional[int], Optional[str], bool]:
        """返回 `(pid, hostname, payload_present)`。

        `payload_present=False` 表示文件存在但内容为空或无法解析 —— 可能是竞态中的
        锁，也可能是被 kill -9 留下的空壳，由调用方结合 mtime 决定。
        兼容旧格式：裸十进制整数即 PID，hostname 未知（None）。
        """
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = f.read().strip()
        except OSError:
            return None, None, False
        if not payload:
            return None, None, False
        try:
            parsed = json.loads(payload)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            try:
                pid = int(parsed.get("pid", -1))
            except (TypeError, ValueError):
                return None, None, False
            hostname = parsed.get("hostname")
            return pid, (str(hostname) if hostname else None), True
        # 旧格式：裸 PID，没有 hostname 信息。
        try:
            return int(payload), None, True
        except ValueError:
            return None, None, False

    def _break_stale_lock_if_needed(self) -> None:
        pid, hostname, payload_present = self._read_lock_owner()

        if not payload_present:
            # 竞态窗口：锁刚被 O_EXCL 建立、payload 还没写完。只有明显老于宽限期
            # 的空文件才当残留处理。
            try:
                age = time.time() - os.path.getmtime(self.path)
            except OSError:
                return
            if age < self._EMPTY_PAYLOAD_GRACE_S:
                return
        else:
            own_host = self._own_hostname()
            if hostname and own_host and hostname != own_host:
                # 共享文件系统：别的节点的 PID 在本机毫无意义，绝不据此删锁。
                return
            if pid is not None and pid > 0 and self._pid_is_alive(pid):
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
                os.write(self.fd, self._lock_payload().encode("utf-8"))
                # payload 必须真的落到文件里，别人才判断得了归属；这条锁只活几十
                # 毫秒，一次 fsync 的代价可以忽略。
                try:
                    os.fsync(self.fd)
                except OSError:
                    pass
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


class TerminationRequested(BaseException):
    """收到 SIGTERM/SIGINT。用异常而不是 `sys.exit`，好让 `finally` 链跑完。

    🔑 [2026-09-10] 基类必须是 `BaseException`，不能是 `Exception`/`RuntimeError`。

    本仓有 ~220 处 `except Exception` 的 fail-open 兜底（ibs_engine 100、
    abfe_pipeline 76、runabfe 21、abfe_core 19、abfe_preoptimizer 7），全仓
    `except KeyboardInterrupt` 则是 **0** 处 —— 也就是说这些兜底从来没打算接住
    "用户要停"。改动前 Ctrl-C 抛的 `KeyboardInterrupt` 是 `BaseException` 子类，
    能干净地穿透它们；一旦让 `TerminationRequested` 变成 `Exception` 子类，
    这 220 处会**逐个把停机信号当成一次普通失败吞掉**。实测过两个具体后果：

      * `_is_traj_valid` 的 `except Exception: return False` —— Ctrl-C 恰好打在
        读 DCD 时，一条**完好**的轨迹被判成损坏，下游重跑该窗口；
      * `get_device_strategy` 的 `except Exception` —— Ctrl-C 打在 CUDA 探测的
        那几毫秒里，整条生产**静默降级到 CPU** 跑完，还会警告"CUDA 平台不可用"。

    净效果是既停不下来、又新增了一批"按了 Ctrl-C 反而算错"的路径。
    `BaseException` 保留了原本的穿透语义，`finally` / `with` 链照样跑。
    """


@contextlib.contextmanager
def graceful_termination(log=print):
    """把 SIGTERM / SIGINT 变成一次**有记录、会清理**的退出（ATT-23 / issue #142）。

    ## 它做什么

    默认的 SIGTERM 处理是直接终止进程：所有 `finally` 一个都不跑 —— 输出目录锁
    不释放、Context 不销毁、DCDReporter 的收尾不执行、日志里连"我被信号杀了"
    这句都没有。2026-09-09 那次作业就是这样消失的：两个 log 停在同一行，
    没有 traceback，事后完全分不清是崩了、挂了、还是被杀了。

    改成抛 `TerminationRequested`：Python 会在下一个字节码边界抛出，既有的
    `finally` / `with` 链照常执行，退出前留下"第几号信号、什么时候"。

    ## 它**不**做什么

    **不新增任何 stage 中途的 checkpoint。** issue 里"graceful checkpoint-on-stop"
    的字面目标需要在采样循环内部插入落盘点，那是另一件事。这里只保证：
    已经写好的那些 `finally`（释放锁、销毁 Context、reporter 收尾）真的会跑，
    以及这次死亡在日志里留痕。别把它当成"随时可以 kill 而不丢进度"。

    ## 为什么捕不到 OOM killer

    OOM killer 发的是 **SIGKILL**，内核直接回收，任何语言都装不上处理器。
    要防的是"内存被吃光"本身（见 `_host_memory_mib()` 打点与
    `compute_u_kn` 的 worker 内存预算），不是在这里。
    """
    previous = install_termination_handlers(log)
    try:
        yield
    finally:
        import signal as _signal

        for sig, handler in previous.items():
            try:
                _signal.signal(sig, handler)
            except (ValueError, OSError):
                pass


def install_termination_handlers(log=print) -> Dict:
    """装上 SIGTERM/SIGINT 处理器，返回被替换掉的旧处理器（供还原）。

    语义与 `graceful_termination` 完全一致，只是不自动还原 —— 给 `main()` 这种
    "装上就用到进程结束"的场景用，免得为了一个 `with` 把整个主函数重缩进。
    """
    import signal as _signal

    previous = {}

    def _handler(signum, _frame):
        name = _signal.Signals(signum).name
        log(f"  [信号] 收到 {name}（{signum}）——开始有序退出，执行既有清理链。")
        # 措辞要准：这条信息是在**抛出的那一刻**打的，清理链还没跑；而且输出目录
        # 独占锁已经不存在了（见 guard_run_directory）。原来那句"已执行既有清理
        # （释放输出目录锁、…）"两处都是假的，读日志的人会据此以为锁被正常释放过。
        raise TerminationRequested(
            f"收到 {name}；开始有序退出，既有的 finally / with 清理链"
            "（销毁 Context、reporter 收尾）将照常执行。"
            "注意：本次退出**没有**新增 stage 中途 checkpoint，未落盘的采样进度会丢。"
        )

    for sig in (_signal.SIGTERM, _signal.SIGINT):
        try:
            previous[sig] = _signal.signal(sig, _handler)
        except (ValueError, OSError):
            # 非主线程 / 平台不支持：不装就是了，不该因此拦住整次运行。
            pass
    return previous


def guard_run_directory(output_dir: str, log=print) -> None:
    """接管 SIGTERM/SIGINT，让既有清理链跑完。

    这里曾经还取一把 `--output` 目录独占锁。删了：进程崩掉或被 SIGKILL 时锁文件
    残留，而 stale 判定在共享盘上判不出来（PID 跨节点无意义、hostname 不同一律
    当活着），于是 `--resume` 永远进不来 —— 锁挡住的是续跑本身。
    """
    install_termination_handlers(log)
    log(f"  [作业防护] 已接管 SIGTERM/SIGINT: {output_dir}")
    return None


def estimate_stage_trajectory_bytes(
    n_states: int, n_steps: int, save_interval: int, n_atoms: int
) -> int:
    """一个 stage 的 DCD 大概要占多少字节。

    DCD 每帧每原子 3 个 float32 = 12 B，外加每帧几十字节的头，这里按 12 B/原子
    算再加 5% 余量 —— 预检只需要量级对，不需要精确。
    """
    frames_per_state = max(1, int(n_steps) // max(1, int(save_interval)))
    per_frame = max(1, int(n_atoms)) * 3 * 4
    return int(max(1, int(n_states)) * frames_per_state * per_frame * 1.05)


def ensure_free_disk_for_stage(
    output_dir: str, required_bytes: int, label: str, log=print
) -> None:
    """stage 开跑前检查剩余空间；不够就**在烧 GPU 之前**失败（ATT-23 / issue #142）。

    原来全仓只有 `abfe_diagnostics.py` 的 `doctor` 只读体检里出现过
    `shutil.disk_usage`，运行前没有任何预检 —— 盘满是在跑了几小时之后、
    DCDReporter 写到一半时才发现的，那时轨迹已经截断、checkpoint 可能半截。

    要求 `free >= required * 2`：留一倍余量给 checkpoint、能量日志、
    以及同一个盘上别的作业。
    """
    try:
        usage = shutil.disk_usage(output_dir)
    except OSError as exc:
        log(f"  [WARN] {label}：读不到 {output_dir} 的磁盘用量（{exc}），跳过预检。")
        return
    needed = int(required_bytes) * 2
    if usage.free >= needed:
        log(
            f"  [磁盘预检] {label}: 预计产出 {required_bytes / 2**30:.2f} GiB，"
            f"可用 {usage.free / 2**30:.1f} GiB —— 通过。"
        )
        return
    raise RuntimeError(
        f"{label}：磁盘空间不足，拒绝开跑（fail closed）。\n"
        f"    目录 {output_dir}\n"
        f"    预计本 stage 产出 {required_bytes / 2**30:.2f} GiB，"
        f"按 2× 余量需要 {needed / 2**30:.2f} GiB，实际可用 "
        f"{usage.free / 2**30:.2f} GiB。\n"
        "  在这里失败，比跑几小时之后 DCD 写到一半被截断要好 —— 那种截断"
        "还会让下次 resume 把半截轨迹当成'已存在'。"
    )


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
# ---------------------------------------------------------------------------
# 跨 platform 的 checkpoint 迁移
# ---------------------------------------------------------------------------
# 🔑 [2026-09-01] "这个 checkpoint 是别的 platform 写的" 和 "这个 checkpoint 坏了"
# 是两件事，此前被同一个 `except Exception: return False` 吞成一件。
#
# 4W53 实测：`checkpoints/pre_equil.chk` 文件头是 `CPUB`，OpenMM 报
#   loadCheckpoint: Checkpoint was created with a different Platform: CPU
# 成因是那次运行开始时 CUDA 没被检测到、强制降级到 CPU 跑完了预平衡（该误降级
# 已于同日修复）。现在 CUDA 认得出来了，于是这份**完好**的 checkpoint 载不进
# CUDA context，`equilibrium_is_done` 判成"没做完"，要重跑 5,000,000 步预平衡。
#
# 迁移做法：用同一个 System + 同一个（反序列化出来的）Integrator，在 checkpoint
# 自己的 platform 上建一个一次性 Simulation 载入，取出 State 再 setState 到目标
# Context，并搬运 step count 与 time。
#
# **迁移得到什么、丢什么，说清楚：**
#   搬过去了：坐标、速度、周期盒、全部全局参数（含 λ、Boresch scale 等）、
#             已完成步数、模拟时间 —— 也就是全部**物理**状态。
#   搬不过去：Integrator 内部的随机数流（State 不含它）。
#   代价评估：Langevin 的 RNG 流不是体系的物理状态；从同一个 (x, v) 用新的随机
#             流继续积分，仍然是同一个系综的合法续算，丢的是**逐位可复现性**，
#             不是正确性。而不迁移的代价是整段轨迹作废、从初始坐标重跑 —— 那才
#             是真正丢状态。所以迁移严格优于现状。
#   迁移发生时会打醒目警告并写进 `checkpoint_platform_migration` 供 provenance
#   追溯，绝不静默。
_CHECKPOINT_PLATFORM_MISMATCH_RE = re.compile(
    r"created with a different Platform:\s*([A-Za-z0-9_]+)"
)


def _checkpoint_source_platform(exc) -> Optional[str]:
    """从 OpenMM 的报错里认出 checkpoint 自己的 platform；不是这个原因则 None。"""
    m = _CHECKPOINT_PLATFORM_MISMATCH_RE.search(str(exc))
    return m.group(1) if m else None


def load_checkpoint_with_platform_migration(simulation, chk_path: str) -> str:
    """把 checkpoint 载入 ``simulation``，必要时跨 platform 迁移。

    返回 ``"native"`` 或 ``"migrated:<源 platform>"``。
    只有 platform 不匹配这一种失败会走迁移；其它异常原样抛出（损坏、System 不符
    等仍然 fail-closed）。
    """
    try:
        simulation.loadCheckpoint(chk_path)
        return "native"
    except Exception as exc:  # noqa: BLE001 - 需要看异常内容才能分类
        source_platform = _checkpoint_source_platform(exc)
        if source_platform is None:
            raise

    import openmm.app as _app

    probe_system = openmm.XmlSerializer.clone(simulation.system)
    probe_integrator = openmm.XmlSerializer.clone(simulation.integrator)
    probe = _app.Simulation(
        simulation.topology,
        probe_system,
        probe_integrator,
        openmm.Platform.getPlatformByName(source_platform),
    )
    probe.loadCheckpoint(chk_path)
    state = probe.context.getState(
        getPositions=True,
        getVelocities=True,
        getParameters=True,
        enforcePeriodicBox=False,
    )
    # ⚠️ 这里**不能**用 `context.setState(state)` 一把梭：它会把 checkpoint 里的
    # 全局参数整表往目标 context 上灌，目标少任何一个就抛
    # "Called setParameter() with invalid parameter name: ..." —— 而更危险的是
    # 反过来：如果哪天它改成宽容模式，一个被静默丢掉的全局参数（λ_vdw、
    # λ_shield、Boresch scale 都是全局参数）就是一个静默的正确性 bug。
    # 所以逐项搬，并且**缺任何一个参数就 fail-closed**，把名字列出来。
    source_parameters = dict(state.getParameters())
    target_parameters = set(simulation.context.getParameters().keys())
    missing = sorted(set(source_parameters) - target_parameters)
    if missing:
        raise RuntimeError(
            f"checkpoint（{source_platform} 平台写入）带有目标 Context 不认识的全局参数 "
            f"{missing}；这说明两边的 System 不是同一个（例如缺 barostat 或缺某个 "
            "λ 控制的力），拒绝迁移 —— 静默丢掉全局参数会直接算错。"
        )
    simulation.context.setPositions(state.getPositions())
    simulation.context.setVelocities(state.getVelocities())
    simulation.context.setPeriodicBoxVectors(*state.getPeriodicBoxVectors())
    for name, value in source_parameters.items():
        simulation.context.setParameter(name, value)
    # 上面这些都不搬步数/时间，但它们决定"还剩多少步"，必须一起搬。
    simulation.context.setStepCount(probe.context.getStepCount())
    simulation.context.setTime(probe.context.getTime())
    simulation.currentStep = probe.context.getStepCount()
    del probe
    return f"migrated:{source_platform}"


def _is_checkpoint_valid(
    chk_path: str,
    simulation=None,
    load_checkpoint=None,
) -> bool:
    """Return whether a checkpoint can be loaded by its target ``Simulation``.

    OpenMM checkpoints are opaque, platform/version dependent binary state.
    File size, a seek, or a magic prefix cannot establish that they belong to
    the current System/Context.  Consequently this helper is deliberately
    fail-closed when no target loader is supplied.  Callers that need to make
    a resume/append decision must call ``simulation.loadCheckpoint`` on the
    target Simulation and use that successful call as the authority.
    """
    if not os.path.isfile(chk_path):
        return False
    try:
        if os.path.getsize(chk_path) <= 0:
            return False
    except OSError:
        return False

    if load_checkpoint is None:
        load_checkpoint = getattr(simulation, "loadCheckpoint", None)
    if not callable(load_checkpoint):
        return False
    try:
        load_checkpoint(chk_path)
    except Exception as exc:  # noqa: BLE001 - 需要区分"别的 platform 写的"与"坏了"
        # platform 不匹配不等于无效：只要能在源 platform 上载入并把 State 迁过来，
        # 这份 checkpoint 就是完好且可用的。simulation 为 None（调用方只给了裸
        # load_checkpoint 函数）时无法迁移，维持原来的 fail-closed。
        if simulation is None or _checkpoint_source_platform(exc) is None:
            return False
        try:
            load_checkpoint_with_platform_migration(simulation, chk_path)
        except Exception:
            return False
    return True


def _is_traj_valid(
    dcd_path: str,
    min_frames: int = 1,
    topology=None,
) -> bool:
    """Validate a DCD with MDTraj's real low-level reader, fail-closed.

    ``DCDTrajectoryFile`` is used lazily so a topology is not required merely
    to authenticate a DCD stream.  Its read operation must consume all frames
    through EOF; reader failures (including truncated records), missing MDTraj,
    and files with fewer than ``min_frames`` are invalid.  A supplied topology
    is used only for an optional atom-count check.  There is deliberately no
    magic-byte, file-size, or dependency-free parser fallback here: those
    checks cannot establish that an opaque DCD is complete.
    """
    if not os.path.isfile(dcd_path):
        return False
    try:
        required_frames = max(0, int(min_frames))
        file_size = os.path.getsize(dcd_path)
    except (OSError, TypeError, ValueError):
        return False
    if file_size <= 0:
        return False

    # Keep MDTraj lazy so importing this module does not make it a hard
    # dependency.  The low-level reader does not require a topology and, unlike
    # a header/size heuristic, validates every record while reading to EOF.
    try:
        import mdtraj as md

        formats = getattr(md, "formats", None)
        reader_factory = getattr(formats, "DCDTrajectoryFile", None)
        if not callable(reader_factory):
            return False

        expected_atoms = None
        if topology is not None:
            get_num_atoms = getattr(topology, "getNumAtoms", None)
            if callable(get_num_atoms):
                expected_atoms = int(get_num_atoms())
            elif hasattr(topology, "n_atoms"):
                expected_atoms = int(topology.n_atoms)

        # 🔑 [2026-09-09] 原来是一次裸 `active_reader.read()` —— 为了"证明这条 DCD
        # 能读到 EOF"，把**整条轨迹**拉进宿主内存。这个函数在一次运行里被调很多次
        # （`_all_remd_trajs_valid` 每个 replica 一次，decharging 的 resume 路径整套
        # 走两遍；`runabfe.equilibrium_is_done` 还会对 pre_equilibration.dcd 调），
        # 而它紧挨着 Stage 1 建 replica —— 正是 std::bad_alloc 出现的窗口。
        # 分块读语义完全一样：仍然逐记录读到 EOF，截断记录照样在 read() 里抛，
        # 只是峰值从 O(整条轨迹) 降到 O(一块)。
        # 一次读多少帧按内存预算自适应，用 `abfe_core.frames_per_chunk`（与回卷复查、
        # LJ 尾项共用同一套预算算术，不在这里另搓一份）。每帧 float32 =
        # n_atoms × 3 × 4 B，1 GiB 预算下 30710 原子约 2850 帧/块、75000 原子约 1160 帧/块。
        # 下限 500 帧：小体系别退化成一堆小 read。读不到 n_atoms 就用它。
        def _frames_per_read(active_reader) -> int:
            n_atoms_hint = getattr(active_reader, "n_atoms", None)
            if not n_atoms_hint:
                return 500
            return frames_per_chunk(int(n_atoms_hint) * 3 * 4, minimum=500)

        def _consume_to_eof(active_reader):
            frames_per_read = _frames_per_read(active_reader)
            frames_seen = 0
            atoms_seen = None
            # 不是所有 reader 都接 `n_frames`（测试里的低层 reader 替身就只有
            # 无参 `read()`）。第一次遇到 TypeError 就退回"一次读完"，
            # 行为与分块之前完全一致。
            chunked = True
            while True:
                if chunked:
                    try:
                        payload = active_reader.read(n_frames=frames_per_read)
                    except TypeError:
                        chunked = False
                        payload = active_reader.read()
                else:
                    payload = active_reader.read()
                coordinates = (
                    payload[0] if isinstance(payload, (tuple, list)) and payload else payload
                )
                if coordinates is None:
                    break
                shape = getattr(coordinates, "shape", None)
                if shape is not None:
                    if len(shape) < 1:
                        break
                    n_here = int(shape[0])
                    if atoms_seen is None and len(shape) >= 2:
                        atoms_seen = int(shape[1])
                else:
                    n_here = len(coordinates)
                    if atoms_seen is None and n_here:
                        atoms_seen = len(coordinates[0])
                if n_here <= 0:
                    break
                frames_seen += n_here
                if not chunked or n_here < frames_per_read:
                    break
            return frames_seen, atoms_seen

        reader = reader_factory(dcd_path)
        has_context_manager = callable(getattr(reader, "__enter__", None))
        if has_context_manager:
            with reader as active_reader:
                frame_count, atom_count = _consume_to_eof(active_reader)
                reader_atom_count = getattr(active_reader, "n_atoms", None)
        else:
            try:
                frame_count, atom_count = _consume_to_eof(reader)
                reader_atom_count = getattr(reader, "n_atoms", None)
            finally:
                close = getattr(reader, "close", None)
                if callable(close):
                    close()

        if atom_count is None:
            atom_count = reader_atom_count

        if expected_atoms is not None and atom_count is not None:
            if int(atom_count) != expected_atoms:
                return False
        if frame_count < required_frames:
            return False
        # [P1-07 补强, 2026-08-30] 头部声明帧数 vs 实际读到帧数。
        #
        # 低层 reader 对"头部声称 N 帧、文件只有 N-1 完整帧 + 半帧"（崩溃瞬间的
        # 典型文件形态）只向 stderr 发一条 warning，然后**静默返回短读** ——
        # 上面的 read() 不抛异常，frame_count 只是"能读出来的帧数"。DCD 头部
        # 偏移 8..12 是 NSET（声明帧数；OpenMM 的 DCDFile 每写一帧都回写更新
        # 该字段），所以 NSET > 0 且 frame_count != NSET ⟹ 文件被截断或头部
        # 与数据不一致，一律拒绝。
        # 头部不可解析（非 OpenMM/CHARMM 布局、大端字节序、伪造内容）时置
        # None：无从约束，保持上面的真实 reader 判据，不发明新的假判据。
        # （内联解析而不是抽 helper：契约测试会把本函数单独编译、globals 只有
        # `os`，任何跨函数调用都会变成 NameError。）
        declared_frames = None
        try:
            import struct as _struct

            with open(dcd_path, "rb") as _header_handle:
                _prefix = _header_handle.read(16)
            if len(_prefix) >= 12:
                _magic = _struct.unpack("<i", _prefix[0:4])[0]
                if _magic == 84 and bytes(_prefix[4:8]) == b"CORD":
                    declared_frames = int(_struct.unpack("<i", _prefix[8:12])[0])
        except Exception:
            declared_frames = None
        if declared_frames is not None and declared_frames > 0:
            if frame_count != declared_frames:
                return False
        return True
    except Exception:
        return False


PRE_EQUILIBRATION_SEGMENT_PROTOCOL_VERSION = 1
PRE_EQUILIBRATION_SEGMENT_MANIFEST = "pre_equilibration_segments.json"


def _pre_equilibration_segment_manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, PRE_EQUILIBRATION_SEGMENT_MANIFEST)


def _begin_pre_equilibration_resume_segment(
    output_dir: str,
    trajectory_path: str,
    monitor_path: Optional[str],
    checkpoint_step: int,
    expected_end_step: int,
    fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    """Rotate old pre-equilibration outputs before a checkpoint continuation.

    DCD records are opaque to this layer.  We therefore never seek/truncate a
    guessed byte offset and never ask OpenMM to append to a file whose last
    frame may be newer than the checkpoint.  The old files are moved to an
    indexed segment and the canonical paths are deliberately reopened with
    ``append=False`` by ``pre_equilibrate``.  The manifest is only a provenance
    index; consumers continue to read the canonical active segment, which is a
    single valid chronological stream starting at the successfully loaded
    checkpoint state.

    The small transaction below restores moved files if manifest persistence
    fails before the new reporter is constructed, so an interrupted rotation
    cannot silently discard the previous segment.
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = _pre_equilibration_segment_manifest_path(output_dir)
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("segment manifest is not an object")
    except Exception:
        manifest = {
            "protocol_version": PRE_EQUILIBRATION_SEGMENT_PROTOCOL_VERSION,
            "kind": "pre_equilibration_segments",
            "segments": [],
        }
    manifest["protocol_version"] = PRE_EQUILIBRATION_SEGMENT_PROTOCOL_VERSION
    manifest["kind"] = "pre_equilibration_segments"
    segments = manifest.get("segments")
    if not isinstance(segments, list):
        segments = []
    manifest["segments"] = segments

    used_indices = set()
    for entry in segments:
        if isinstance(entry, dict):
            try:
                used_indices.add(int(entry.get("segment_index")))
            except (TypeError, ValueError):
                continue
    segment_index = 1
    while segment_index in used_indices:
        segment_index += 1

    trajectory_name = os.path.basename(trajectory_path)
    trajectory_stem, trajectory_ext = os.path.splitext(trajectory_name)
    archived_trajectory = os.path.join(
        output_dir, f"{trajectory_stem}.segment-{segment_index:04d}{trajectory_ext}"
    )
    archived_monitor = None
    if monitor_path:
        monitor_name = os.path.basename(monitor_path)
        monitor_stem, monitor_ext = os.path.splitext(monitor_name)
        archived_monitor = os.path.join(
            output_dir, f"{monitor_stem}.segment-{segment_index:04d}{monitor_ext}"
        )

    moved = []
    try:
        if os.path.exists(trajectory_path):
            os.replace(trajectory_path, archived_trajectory)
            moved.append((archived_trajectory, trajectory_path))
        if monitor_path and os.path.exists(monitor_path):
            os.replace(monitor_path, archived_monitor)
            moved.append((archived_monitor, monitor_path))

        archived_record = {
            "segment_index": int(segment_index),
            "status": "archived",
            "trajectory": os.path.relpath(archived_trajectory, output_dir),
            "monitor": (
                os.path.relpath(archived_monitor, output_dir)
                if archived_monitor and os.path.exists(archived_monitor)
                else None
            ),
            "checkpoint_step_before_rotation": int(checkpoint_step),
            "end_checkpoint_step": int(checkpoint_step),
        }
        # Retire the previous active record in place when the manifest already
        # knows about it.  Keeping one record per physical segment avoids a
        # misleading duplicate where the same canonical path appears once as
        # active and again as archived after the next resume.
        previous_active = manifest.get("active_segment")
        previous_index = None
        if isinstance(previous_active, dict):
            try:
                previous_index = int(previous_active.get("segment_index"))
            except (TypeError, ValueError):
                previous_index = None
        retired_previous = False
        if previous_index is not None:
            for entry in segments:
                if not isinstance(entry, dict):
                    continue
                try:
                    entry_index = int(entry.get("segment_index"))
                except (TypeError, ValueError):
                    continue
                if entry_index != previous_index:
                    continue
                entry.update(archived_record)
                retired_previous = True
                break
        # A segment record is useful even when one of the optional files did
        # not exist: it documents that this continuation intentionally started
        # a new generation rather than appending to an unknown prefix.
        if not retired_previous:
            segments.append(archived_record)
        active_record = {
            "segment_index": int(segment_index) + 1,
            "status": "active",
            "trajectory": os.path.relpath(trajectory_path, output_dir),
            "monitor": (
                os.path.relpath(monitor_path, output_dir) if monitor_path else None
            ),
            "start_checkpoint_step": int(checkpoint_step),
            "expected_end_step": int(expected_end_step),
            "fingerprint": fingerprint,
        }
        manifest["active_segment"] = active_record
        manifest["segments"].append(active_record)
        manifest["updated_at"] = datetime.now().isoformat()
        _atomic_write_json(manifest_path, manifest)
    except Exception:
        for moved_path, original_path in reversed(moved):
            try:
                if os.path.exists(moved_path) and not os.path.exists(original_path):
                    os.replace(moved_path, original_path)
            except Exception:
                pass
        raise
    return active_record


def _expected_remd_traj_files(stage_output_dir: str, stage_name: str, n_replicas: int) -> List[str]:
    return [os.path.join(stage_output_dir, f"{stage_name}_rep{i}.dcd") for i in range(int(n_replicas))]


# ============================================================================
# [P1-11] single/2D 采样结果的收敛 sanity gate（写缓存与缓存命中共用同一份）
# ============================================================================
def _endpoint_analysis_artifact(checkpoint_dir, endpoint=None, *, stage_protocol_key, lambda_path_fingerprint):
    """Persist/recover an endpoint solve bound to its complete fixed-state bank."""
    path = os.path.join(checkpoint_dir, "independent_endpoint_analysis.json")
    if endpoint is not None:
        bank_dir = endpoint["diagnostics"]["bank_dir"]
        names = sorted(name for name in os.listdir(bank_dir) if name.endswith(".npz") or name == "manifest.json")
        if "manifest.json" not in names or not any(name.endswith(".npz") for name in names):
            raise ValueError("独立端点 bank 缺少 manifest 或轨迹记录")
        payload = dict(stage_protocol_key=stage_protocol_key,
                       lambda_path_fingerprint=lambda_path_fingerprint,
                       endpoint=endpoint,
                       bank_subdir=os.path.relpath(bank_dir, checkpoint_dir),
                       bank_files={name: _file_sha256(os.path.join(bank_dir, name)) for name in names})
        _atomic_write_json(path, _protocol_fingerprint(payload))
    else:
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        if doc != _protocol_fingerprint(doc.get("payload")):
            raise ValueError("独立端点分析 fingerprint 不匹配")
        payload = doc["payload"]
        if (_stage_window_sampling_identity(payload.get("stage_protocol_key")) != _stage_window_sampling_identity(stage_protocol_key)
                or payload.get("lambda_path_fingerprint") != lambda_path_fingerprint):
            raise ValueError("独立端点分析协议或 lambda 路径不匹配")
        endpoint = payload["endpoint"]
        subdir = payload.get("bank_subdir")
        if not isinstance(subdir, str) or os.path.isabs(subdir) or ".." in subdir.split(os.sep):
            raise ValueError("独立端点 bank 位置缺失或越界")
        bank_dir = os.path.join(checkpoint_dir, subdir)
        files = payload.get("bank_files") or {}
        if "manifest.json" not in files or not any(name.endswith(".npz") for name in files):
            raise ValueError("独立端点分析缺少完整 bank manifest")
        for name, expected_hash in files.items():
            if os.path.basename(name) != name or not expected_hash or _file_sha256(os.path.join(bank_dir, name)) != expected_hash:
                raise ValueError(f"独立端点 bank 文件缺失或内容改变: {name}")
    return endpoint


def _stage_analysis_protocol_versions(stage_name):
    versions = {
        "wca_accounting_version": WCA_ACCOUNTING_VERSION,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        "production_segment_protocol_version": PRODUCTION_SEGMENT_PROTOCOL_VERSION,
        "ligand_com_restraint_protocol_version": LIGAND_COM_RESTRAINT_PROTOCOL_VERSION,
        "pme_decharge_model_version": PME_DECHARGE_MODEL_VERSION,
        "thermodynamic_path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION if stage_name == "vanishing" else "n/a",
    }
    if stage_name == "vanishing":
        versions.update(
            vdw_nonbonded_protocol_version=VDW_NONBONDED_PROTOCOL_VERSION,
            independent_endpoint_protocol_version=INDEPENDENT_ENDPOINT_PROTOCOL_VERSION,
        )
    return versions


# ---------------------------------------------------------------------------
# [2026-09-15 用户拍板] stage 结果的 `converged` 键**已删除**，换成两个正交状态。
#
#   analysis_status  —— "求解器有没有在一条完整路径上解出有限的数"。
#                       **只含硬不变量**：路径完整（无缺窗无跳窗）+ 数值有限 +
#                       结构/身份自洽。**不含任何拟合阈值。**
#   precision_status —— "这个数准不准"。只由 `abfe_core.cross_repeat_precision()`
#                       跨**独立重复**的样本 SD 判定。单次 run 恒为 UNMEASURED。
#
# 故意删键而不是同名改义：同名改义是静默失效（别的文件照旧读到一个布尔），
# 删掉之后 `result.get("converged")` 当场拿到 None，是 fail-loud。
# **不要给 `converged` 做兼容别名。**
#
# `min_overlap` / `target_support_gate` / `min_decorrelated_samples` /
# `max_endpoint_uncertainty_kJ_mol` 及各自阈值**原样留在结果里**，但只是报告：
# 不进 `analysis_status`，也**不得**触发任何自动补帧 / rescue / top-up。
# ---------------------------------------------------------------------------
ANALYSIS_COMPLETE = "ANALYSIS_COMPLETE"
ANALYSIS_INCOMPLETE = "ANALYSIS_INCOMPLETE"

# 🔑🔑 [2026-09-15 用户拍板，第一位的要求] **假门驱动的补帧一律停用。**
#
# `min_overlap` / `target_support_gate` / `min_decorrelated_samples` /
# `max_endpoint_uncertainty_kJ_mol` 没够到阈值，**不是**自动批 GPU 的理由：
#   · 四个阈值全无标定（1.0 那个是抄自一个默认关闭的 early-stop 默认形参）；
#   · 它们是逐段量，科学目标是两腿合成的 σ_bind；
#   · MBAR 的 σ 是渐近估计，对「该采的构型一次都没采到」结构性失明
#     （4W53 +32 kJ/mol、decharging −88 kJ/mol，两次 σ 都很小都没报警）；
#   · 实测：同一个偏斜分布里加帧，N 涨 g 也涨，ESS **比值**不动甚至倒退。
# 诊断照报、照落盘（`stage_quality_failures` / `window_overlap_diagnostics`），
# 但**不得**据此触发 rescue / top-up / 重采样。
# **不要**把这个常量翻成 True 来"恢复功能"—— 要补帧请走自治控制器的
# `decide()`，它的理由是路径不完整，不是一个没标定过的阈值。
AUTO_RESAMPLE_ON_FITTED_GATES = False


def _finite_thermo_rejection_reason(result) -> Optional[str]:
    """ΔG/σ 的硬不变量：dict、两个字段都在、都有限、σ 非负。返回 None 表示通过。"""
    if not isinstance(result, dict):
        return f"结果不是 dict 而是 {type(result).__name__}"
    for field in ("total_delta_G", "total_error"):
        if field not in result:
            return f"缺少 {field}"
        try:
            value = float(result[field])
        except (TypeError, ValueError):
            return f"{field} 不是数值: {result[field]!r}"
        if not math.isfinite(value):
            return f"{field} 非有限: {value!r}"
    if float(result["total_error"]) < 0.0:
        return f"total_error 为负: {float(result['total_error'])!r}"
    return None


def _stage_analysis_rejection_reason(result) -> Optional[str]:
    """stage 结果的**硬不变量**门。返回 None = 结构上完整，可以进下一步 / 当缓存完成标记。

    ⚠️ 这道门**不判精度**。"能不能作为已验收结果发布"另看 `precision_status`
    （见 `_stage_precision_rejection_reason`）。

    缺 `analysis_status` 键 ⟹ **fail-closed**：那是 2026-09-15 之前的老产物
    （当时写的是已删除的 `converged`），它的硬不变量从来没有被这套口径判过。
    """
    reason = _finite_thermo_rejection_reason(result)
    if reason is not None:
        return reason
    status = result.get("analysis_status")
    if status is None:
        return (
            "缺少 analysis_status —— 这是 2026-09-15 `converged` 删除之前的老产物，"
            "它的硬不变量（路径完整/数值有限/结构自洽）从未按当前口径判过。"
            "fail-closed：不当成通过。请用当前代码重新求解该阶段"
            "（能量矩阵已落盘时不需要重新采样）。"
        )
    if status != ANALYSIS_COMPLETE:
        return (
            f"analysis_status={status!r}（硬不变量未满足）："
            f"{result.get('analysis_incomplete_reasons')}"
        )
    return None


def _stage_precision_rejection_reason(result) -> Optional[str]:
    """**发布**门：要作为已验收结果对外，跨重复精度必须实测达标。

    单次 run 结构上只有一个重复 ⟹ `precision_status` 恒为 `UNMEASURED` ⟹
    这道门恒拒。`UNMEASURED` **不是达标也不是不达标**，它是"没测"。
    """
    status = (result or {}).get("precision_status")
    if status == PRECISION_MEETS:
        return None
    if status is None or status == PRECISION_UNMEASURED:
        return (
            f"precision_status={status!r} —— 跨重复精度**未测**（单次 run 只有一个"
            f"重复，至少需要 {MIN_INDEPENDENT_REPEATS} 个独立重复）。"
            "「未测」不是达标也不是不达标，**不得**当作已验收结果发布。"
        )
    return (
        f"precision_status={status!r} —— 跨重复样本 SD 超过目标"
        f"（{(result.get('precision_evidence') or {}).get('sample_stddev_kcal_per_mol')!r} > "
        f"{(result.get('precision_evidence') or {}).get('threshold_kcal_per_mol')!r} kcal/mol）。"
    )


def _single_run_precision_payload() -> Dict[str, Any]:
    """单次 run 的诚实精度证据：一个重复 ⟹ UNMEASURED。不发明任何新阈值。"""
    return cross_repeat_precision([])


def _pme_stage_status_fields(res: Dict[str, Any]) -> Dict[str, Any]:
    """PME/REMD stage（`TraditionalMBARAnalyzer.solve`）结果的新契约状态字段。

    这条路径的 λ 覆盖由 REMD 构造保证（每个 λ 一个 replica，全部求解），所以剩下
    的硬不变量只有"数值有限"。求解器自己那个 `converged`（min_overlap 阈值合取）
    **是拟合阈值** ⟹ 降级为报告，原样留在 `min_overlap`/`min_overlap_threshold`。
    """
    reason = _finite_thermo_rejection_reason({
        "total_delta_G": res.get("delta_G"),
        "total_error": res.get("error"),
    })
    return {
        "analysis_status": ANALYSIS_COMPLETE if reason is None else ANALYSIS_INCOMPLETE,
        "analysis_incomplete_reasons": [] if reason is None else [reason],
        "precision_status": PRECISION_UNMEASURED,
        "precision_evidence": _single_run_precision_payload(),
    }


def _sampling_result_convergence_rejection_reason(result) -> Optional[str]:
    """**traditional 腿专用**（TraditionalMBARAnalyzer.solve 的结果）：返回 None 表示通过。

    ⚠️ 这条路径的 `converged` **没有**被 2026-09-15 的删键波及 —— 删的是
    `ibs_engine.solve_stage_integrated` 的返回契约。IBS/stage 结果请改用
    `_stage_analysis_rejection_reason`，不要复用本函数。

    判据：dict、total_delta_G/total_error 存在且为有限数值、误差非负、
    converged 显式为 True。converged 缺席不算通过——本修复之后所有新写
    的采样缓存都带该字段，旧缓存缺席即视为旧协议产物、拒绝复用。
    """
    reason = _finite_thermo_rejection_reason(result)
    if reason is not None:
        return reason
    if result.get("converged") is not True:
        return (
            f"converged={result.get('converged')!r}"
            f"（min_overlap={result.get('min_overlap')!r}, threshold="
            f"{result.get('min_overlap_threshold')!r}）—— solver 返回了有限 ΔG "
            "但重叠率不足，不允许标成完成"
        )
    return None


def _assert_sampling_result_converged(result, *, context: str) -> None:
    """[P1-11] 硬检查：不通过就 raise，绝不写完成缓存/产出最终 ΔG。"""
    reason = _sampling_result_convergence_rejection_reason(result)
    if reason is not None:
        raise RuntimeError(
            f"[{context}] 采样结果未通过收敛 sanity gate：{reason}（P1-11）"
        )


def _assert_or_warn_sampling_converged(
    result, *, context: str, allow_untrusted: bool, log=print
) -> Optional[str]:
    """`_assert_sampling_result_converged` 的显式放行版，给 traditional 腿用。

    默认（`allow_untrusted=False`）行为与 `_assert_sampling_result_converged`
    **逐字相同**：不通过就抛。只有调用方显式传 True 才降级成一条刺眼的 WARN，
    并把拒绝理由**返回**给调用方落盘 —— 放行过的结果必须在产物里留痕，
    否则下一个人看到的就是一个没有任何标记的"正常" ΔG。
    """
    reason = _sampling_result_convergence_rejection_reason(result)
    if reason is None:
        return None
    if not allow_untrusted:
        raise RuntimeError(
            f"[{context}] 采样结果未通过收敛 sanity gate：{reason}（P1-11）"
        )
    log(
        f"  [WARN] [{context}] 采样结果未通过收敛 sanity gate：{reason}；"
        "因 allow_untrusted_stage_results=True 显式放行。"
        "[WARN] 本腿 ΔG 只能作为中间量，不得作为可发布结果。"
    )
    return reason


def _expected_remd_frame_count(n_steps: int, save_interval: int = 5000) -> int:
    if n_steps <= 0 or save_interval <= 0:
        return 0
    return int(n_steps // save_interval)


def _all_remd_trajs_valid(stage_output_dir: str, stage_name: str, n_replicas: int, min_frames: int = 1) -> bool:
    traj_files = _expected_remd_traj_files(stage_output_dir, stage_name, n_replicas)
    # 🔑 `all([])` 是 True ⟹ `n_replicas == 0`（或期望文件列表为空）时会报
    # 「REMD 轨迹全部完整」，`sampling_cache_ok` 因此成立、整段采样被跳过。
    # 四个 resume 调用点都吃这个返回值。现实里 n_replicas=0 不可达，属潜伏项，
    # 但「空集合上的全称命题为真」正是本仓反复栽的那类静默通过，封死成本一行。
    return bool(traj_files) and all(
        _is_traj_valid(path, min_frames=min_frames) for path in traj_files)


def _remd_sampling_meta_path(stage_output_dir: str, stage_name: str) -> str:
    """Metadata binding a REMD DCD set to the sampling Hamiltonian/protocol."""
    return os.path.join(stage_output_dir, f"{stage_name}_sampling.meta.json")


def _remd_sampling_fingerprint(
    *,
    stage_name: str,
    system: openmm.System,
    topology: app.Topology,
    ligand_indices: List[int],
    lambdas_coul: List[float],
    lambdas_vdw: List[float],
    temperature_K: float,
    n_steps: int,
    exchange_interval: int,
    boresch_params: Optional[Dict],
    potential_type: str,
    platform_name: str,
    coion_identity: Optional[Dict] = None,
    max_resident_contexts: Optional[int] = None,
    # [P1-18] 有效随机源身份：动力学用了哪个 repeat-seed 派生 seed，采样
    # 缓存身份就必须包含它，否则换 repeat_seed 的"独立重复"会命中并复用
    # 旧 DCD/u_kn。repeat_seed 未启用时为 None（payload 里仍是显式 None，
    # 与"没记"有区别）。
    seed_contract: Optional[Dict] = None,
) -> Dict:
    """Fingerprint the distribution that produced a REMD trajectory set.

    A complete DCD is not sufficient for resume: if the co-ion identity or any
    sampling Hamiltonian input changed, reusing those frames for a new u_kn is
    an invalid change of ensemble.  Keep this separate from the u_kn analysis
    fingerprint so an analysis-only change can still reuse valid trajectories.
    """
    payload = {
        "kind": "remd_sampling",
        "pme_decharge_model_version": PME_DECHARGE_MODEL_VERSION,
        "stage_name": str(stage_name),
        "system_xml_sha256": _system_xml_hash(system),
        "topology_sha256": _topology_hash(topology),
        "ligand_indices": [int(i) for i in ligand_indices],
        "lambdas_coul": _lambda_signature(lambdas_coul),
        "lambdas_vdw": _lambda_signature(lambdas_vdw),
        "temperature_K": round(float(temperature_K), 8),
        "n_steps": int(n_steps),
        "exchange_interval": int(exchange_interval),
        # 🔑 [2026-09-01] 同 `_stage_protocol_key`：只有收窄后的四项决定限制力
        # Hamiltonian；诊断/评分/provenance 进键会让"同一个限制力的不同说明"触发重算。
        "boresch_params": _preopt_boresch_protocol_payload(boresch_params),
        "potential_type": str(potential_type),
        "platform_name": str(platform_name),
        "max_resident_contexts": (
            int(max_resident_contexts)
            if max_resident_contexts is not None
            else None
        ),
        "coion_identity": coion_identity,
        "seed_contract": seed_contract,
    }
    return _protocol_fingerprint(payload)


def _remd_sampling_metadata_matches(
    stage_output_dir: str,
    stage_name: str,
    expected_fingerprint: Dict,
) -> bool:
    path = _remd_sampling_meta_path(stage_output_dir, stage_name)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return False
    return metadata.get("sampling_fingerprint") == expected_fingerprint


def _write_remd_sampling_metadata(
    stage_output_dir: str,
    stage_name: str,
    sampling_fingerprint: Dict,
) -> None:
    _atomic_write_json(
        _remd_sampling_meta_path(stage_output_dir, stage_name),
        {"sampling_fingerprint": sampling_fingerprint},
    )

def cleanup_temp_files(checkpoint_dir: str):
    """清理损坏的临时文件 (.tmp)"""
    if not os.path.exists(checkpoint_dir):
        return
    for f in os.listdir(checkpoint_dir):
        if f.endswith(".chk.tmp") or f.endswith(".dcd.tmp"):
            try:
                os.remove(os.path.join(checkpoint_dir, f))
                print(f"  已清理临时文件: {f}")
            except Exception as e:
                print(f"  [WARN] 清理失败 {f}: {e}")

# [2026-09-09] `aggregate_all_energies()` 已删除（GitHub #66 的三个「零调用遗留点」之一）。
# 它用 `arr.T if arr.shape[0] > arr.shape[1] else arr` **按长短猜** (K, N) 方向——
# 窗口的 λ 态数多于帧数时会静默转置错，而且把不同窗口的矩阵直接 hstack 成一个
# "全局 u_kn"，那在分窗口 IBS 下本来就不是一个合法的 MBAR 输入。
# 删除前复核：全仓零调用方，它写的 `full_u_kn_matrix.npy` 也零消费者（只有写没有读）。
# 真正在用的聚合走 `solve_stage_integrated` 的逐窗口链式拼接。

def _split_platform_spec(platform_name: str) -> Tuple[str, Optional[str]]:
    """解析平台字符串，支持 'CUDA:1' 这种显式设备写法。"""
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


def _resolve_nvcc() -> Optional[str]:
    """找 `nvcc` 的**绝对路径**。🔑🔑 **不能只问 PATH。**

    真机（计算节点，2026-09-15）：`nvcc` 只存在于 mamba env 里，而作业的启动方式
    是**用绝对路径调 env 的 python**：

        /…/miniforge3/envs/openmm_dev/bin/python  /…/runabfe.py …

    那样启动**不会**把 env 的 `bin/` 放进 PATH ⟹ `shutil.which("nvcc")` 返回
    None ⟹ 不设 `CudaCompiler` ⟹ OpenMM 退回 NVRTC ⟹ 那段生成源码编译失败
    ⟹ 被 `_create_context_with_local_cpu_fallback` 静默接住、**整个阶段跑 CPU**
    （实测 pilot 87.6 s/λ）。同一份 `launch.log` 里 vanishing pilot 连起三次，
    前两次（从激活过 env 的 shell 起）`CudaCompiler: 'nvcc'` 跑 CUDA，第三次没有
    ⟹ `default_program(325)` 编译错误 ⟹ CPU。差别只在 PATH，不在代码、不在体系。

    所以查找顺序按"**可靠性**"排，PATH 排在后面：
      1. `OPENMM_CUDA_COMPILER` —— 使用者显式指定，最高优先级；
      2. **运行中解释器的同级目录**（`sys.executable` 的 bin/）—— conda/mamba env
         里 nvcc 与 python 是邻居，这条与"有没有 activate"完全无关，正是上面
         那个失败形状的对症解；
      3. `CONDA_PREFIX/bin`；
      4. PATH（原来唯一的那条，留作兜底）；
      5. `CUDA_HOME` / `CUDA_PATH` 的 `bin/`。

    找不到返回 None —— 调用方负责**明说**，不要静默。
    """
    import sys as _sys

    explicit = os.environ.get("OPENMM_CUDA_COMPILER")
    if explicit and os.path.isfile(explicit) and os.access(explicit, os.X_OK):
        return explicit
    cands = [
        os.path.join(os.path.dirname(os.path.abspath(_sys.executable)), "nvcc"),
    ]
    if os.environ.get("CONDA_PREFIX"):
        cands.append(os.path.join(os.environ["CONDA_PREFIX"], "bin", "nvcc"))
    for var in ("CUDA_HOME", "CUDA_PATH"):
        if os.environ.get(var):
            cands.append(os.path.join(os.environ[var], "bin", "nvcc"))
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return shutil.which("nvcc")


def _build_platform_props(platform_name: str) -> Tuple[str, Dict[str, str]]:
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
            raise ValueError(
                f"平台设备索引必须是非负整数：{platform_name!r}"
            )
    if upper == "CUDA":
        props["Precision"] = "mixed"
        if device is not None:
            props["DeviceIndex"] = device
        # 🔑 [2026-09-15] 原来是 `shutil.which("nvcc")` —— 只问 PATH。
        # 绝对路径调 env 的 python 时 env 的 bin 不在 PATH ⟹ 找不到 ⟹ 退 NVRTC
        # ⟹ 编译失败 ⟹ 静默跑 CPU。见 `_resolve_nvcc` 的 docstring（真机实证）。
        # 这里写**绝对路径**而不是 "nvcc"：OpenMM 接受全路径，而且这样它不依赖
        # 子进程的 PATH（`CudaCompiler` 最终是被 system() 调起来的）。
        _nvcc = _resolve_nvcc()
        if _nvcc:
            props["CudaCompiler"] = _nvcc
    elif upper == "OPENCL":
        props["Precision"] = "mixed"
        if device is not None:
            props["DeviceIndex"] = device
    return base, props


def _create_context_with_local_cpu_fallback(
    system,
    integrator_factory,
    platform_name: str,
    log=print,
):
    """Create an OpenMM Context for an explicit platform/device specification.

    Platform lookup alone is not a usable-device probe: CUDA may be registered
    while Context construction fails because the selected device is busy,
    absent, or out of memory.  The requested Context is therefore constructed
    inside the guarded block.  A fresh Integrator is used for the CPU retry so
    a partially initialized Context can never leave the original Integrator
    bound.  The fallback is local and does not mutate a pipeline's configured
    platform.
    """
    requested_base, requested_props = _build_platform_props(platform_name)
    try:
        integrator = integrator_factory()
        platform = openmm.Platform.getPlatformByName(requested_base)
        context = openmm.Context(system, integrator, platform, requested_props)
        return context, integrator, requested_base, requested_props
    except Exception as exc:
        if requested_base.upper() == "CPU":
            raise
        # 🔑🔑 [2026-09-15 真机] **编译错误和"卡忙/显存不够"不是一类失败。**
        #
        # 原来一个 `except Exception` 全接住、一律回退 CPU。但：
        #   · 资源类（设备被占 / OOM / 卡不在）—— 换台机、等一会儿就好，回退合理；
        #   · **编译错误** —— 确定性的。回退只是把一个每次都会复现的 bug 变成一次
        #     **静默的百倍减速**：真机 vanishing pilot 掉到 CPU 后 87.6 s/λ，
        #     20 个 λ 点光 pilot 就半小时，生产阶段更不用说，而且跑完还得重来。
        #
        # ⚠️ 顺带一个更要紧的事实：**OpenMM 的 `CudaCompiler` 默认值是空串，
        # 也就是默认走 NVRTC**，nvcc 是可选旧路。先前 `shutil.which("nvcc")` 能不能
        # 命中纯看启动 shell 的 PATH —— 命中了就绕开 NVRTC，没命中就撞上这个编译
        # 错误。所以「以前一直好好的」是**碰巧**，不是代码正确。
        # 真正的 bug 在生成的 kernel 源码里（`0 }`：一个裸常数后面直接跟 `}`，
        # 缺分号）。要定位它必须拿到那段源码 —— 所以这里把完整编译错误原样抛出，
        # 而不是吞掉换 CPU。
        _msg = str(exc)
        _is_compile = ("Error compiling program" in _msg
                       or "error: expected" in _msg
                       or "nvrtc" in _msg.lower())
        if _is_compile:
            raise RuntimeError(
                f"平台 {platform_name!r} 的 kernel **编译失败**（不是资源问题）："
                f"{_msg}\n"
                "这类失败是确定性的：回退 CPU 只会把它变成一次静默的百倍减速"
                "（真机实测 pilot 87.6 s/λ），跑完还得重来，所以这里不回退。\n"
                "处置（三选一）：\n"
                "  · 指定 nvcc 绕开 NVRTC：export OPENMM_CUDA_COMPILER=<nvcc 绝对路径>\n"
                "    （conda/mamba 环境里它就在解释器隔壁；本代码也会自动去那儿找）\n"
                "  · 真要用 CPU 就显式传 --platform CPU，让它成为一个决定而不是意外\n"
                "  · 把上面那段编译错误连同 kernel 源码留档 —— 那才是根因所在"
            ) from exc
        log(
            f"  [WARN] 平台 {platform_name!r} 的 Context 初始化失败（判为**资源类**，"
            f"非编译错误）: {exc}，本阶段局部回退到 CPU"
        )
        integrator = integrator_factory()
        platform = openmm.Platform.getPlatformByName("CPU")
        context = openmm.Context(system, integrator, platform, {})
        return context, integrator, "CPU", {}
# =============================================================================
# 多进程工作函数：双阶段并行采样
# =============================================================================


# 🔑 [2026-09-11] 残差采样**实际生效**的 stage，唯一定义。
# `_run_dual_lambda_stage` 的 `residual_for_stage` 与 `_stage_protocol_key` 的缓存
# 身份必须读同一个集合：这两处一旦漂开，就是"开开关会作废压根没被残差碰过的
# stage 缓存"那个 bug（decharging 整段约 28 分钟白重跑）。
RESIDUAL_SAMPLING_STAGES = frozenset({"vanishing", "vanishing_rescue"})

# 🔑 [2026-09-15] Stage-2 控制器的**执行策略**键：它们决定控制器什么时候停手，
# 不改 Hamiltonian、不改任何被算出来的数（同一份轨迹在任意取值下 ΔG 逐位相同），
# 所以和 `resume` / `allow_untrusted_stage_results` 一样必须被排除在缓存身份之外。
# 不排除的后果是已经踩过的那条：`_last_run_config["kwargs"]` 逐字段进指纹，
# 新加一个 kwargs 键就让全部 stage 缓存失配（Stage 1 约 28 分钟 + 6 个 vanishing
# 窗口全部重采样）—— 于是"把预算接上去"和"不重跑 GPU"变成二选一。
# ⚠️ 这两个键**今天的指纹里本来就没有**（上游从没传过），所以剔除它们对既有
# 缓存是逐位 no-op；这是它能无痛落地的前提，别往这个集合里塞已经在传的键。
_NON_IDENTITY_KWARGS = (
    "stage2_production_budget_steps",
    "stage2_max_production_blocks_per_window",
)


def _strip_non_identity_kwargs(run_config: Dict) -> Dict:
    """把执行策略键从 run_config["kwargs"] 里摘掉（不改入参）。"""
    kw = run_config.get("kwargs")
    if isinstance(kw, dict) and any(k in kw for k in _NON_IDENTITY_KWARGS):
        kw = {k: v for k, v in kw.items() if k not in _NON_IDENTITY_KWARGS}
        run_config = dict(run_config)
        run_config["kwargs"] = kw
    return run_config

# 全局拼接失败里，**成因是「有窗口数据不够被跳过」**的那几个 reason。
# 它们是路由信号（SKIPPED_WINDOW / INSUFFICIENT_DATA），不是终态：动作是补采，
# 不是炸出流水线。字符串必须与 `ibs_engine` 里 `self._fallback("...")` 的字面量
# 逐字一致（`no_local_tmbAR_results` 的大小写就是那么怪），
# 由 tests/test_routing_signal_not_terminal.py 对着 ibs_engine 源码钉住。
STAGE_SOLVE_ROUTING_ERRORS = frozenset({
    "no_local_tmbAR_results",
    "window_overlap_broken",
    "window_overlap_broken_for_covariance_chain",
})


def _normalize_residual_sampling_runtime(
    enabled: bool,
    residual_basis_force_factory: Optional[Any],
    residual_state_coefficients_factory: Optional[Any],
    residual_energy_offset_kj_mol: float,
    sampling_score_sha256: Optional[str],
) -> Dict[str, Any]:
    """Validate the optional production residual-sampling injection point.

    This is deliberately resolved in ``ABFEPipeline.__init__``.  A residual
    model is a sampling-Hamiltonian input, so accepting a partial/anonymous
    configuration after an OpenMM Context has already been created would make
    resume identity and physical accounting ambiguous.  The default-disabled
    branch returns the legacy values unchanged.
    """
    if not isinstance(enabled, bool):
        raise TypeError("residual_sampling_enabled 必须是 bool")
    if isinstance(residual_energy_offset_kj_mol, bool):
        raise TypeError("residual_energy_offset_kj_mol 必须是有限数值")
    try:
        offset = float(residual_energy_offset_kj_mol)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "residual_energy_offset_kj_mol 必须是有限数值"
        ) from exc
    if not math.isfinite(offset):
        raise ValueError("residual_energy_offset_kj_mol 必须是有限数值")

    factories_present = (
        residual_basis_force_factory is not None
        or residual_state_coefficients_factory is not None
    )
    if not enabled:
        if factories_present or offset != 0.0 or sampling_score_sha256 is not None:
            raise ValueError(
                "residual sampling 未启用时不得提供 residual factory、energy offset 或 "
                "sampling_score_sha256；请显式设置 residual_sampling_enabled=True "
                "（正式用户入口为 outer_lambda_local_residual_ibs=true）"
            )
        return {
            "enabled": False,
            "basis_force_factory": None,
            "state_coefficients_factory": None,
            "energy_offset_kj_mol": 0.0,
            "sampling_score_sha256": None,
        }

    if not callable(residual_basis_force_factory):
        raise TypeError(
            "启用 residual sampling 时必须提供可调用的 residual_basis_force_factory"
        )
    if not callable(residual_state_coefficients_factory):
        raise TypeError(
            "启用 residual sampling 时必须提供可调用的 "
            "residual_state_coefficients_factory"
        )
    if not isinstance(sampling_score_sha256, str):
        raise ValueError(
            "启用 residual sampling 时必须提供 sampling_score_sha256（64 位 SHA-256）"
        )
    score_hash = sampling_score_sha256.strip().lower()
    if len(score_hash) != 64 or any(
        character not in "0123456789abcdef" for character in score_hash
    ):
        raise ValueError("sampling_score_sha256 必须是 64 位十六进制 SHA-256")
    return {
        "enabled": True,
        "basis_force_factory": residual_basis_force_factory,
        "state_coefficients_factory": residual_state_coefficients_factory,
        "energy_offset_kj_mol": offset,
        "sampling_score_sha256": score_hash,
    }


# ---------------------------------------------------------------------------
# preopt 缓存的两层划分（2026-09-10）
# ---------------------------------------------------------------------------
# 第 1 层「原始 pilot 测量」：Hamiltonian、采样步数、差分步长、遍历顺序、
#   加密探针协议。变了 ⟹ 必须重跑 pilot（GPU）。
# 第 2 层「派生路径」：final states、densify、window min/max。只影响从同一份
#   pilot 测量出发怎么布点分窗 ⟹ 变了可以**离线重算**，不必重烧 GPU。
#
# 下面这个集合就是第 2 层的全部键；`_preopt_protocol_key()` 里除它以外的一切
# 都属于第 1 层。**不要**往这里加任何会影响采样本身的东西。
_PREOPT_DERIVED_PATH_KEYS = (
    "stage2_final_n_states",
    "stage2_refine_extra_points_per_segment",
    "stage2_window_min_states",
    "stage2_window_max_states",
    # 🔑🔑 [2026-09-15] 见 `derived_path_params` 处的注释：这两个键同样决定布局。
    "stage2_window_partition",
    "stage2_first_window_max_states",
    # 🔑 [2026-09-18] 末窗上界同样**决定布局**（它决定末窗长到多大才拆）⟹ 必须
    # 进派生路径身份。漏了它的后果是改了这个数**不作废旧窗口缓存**，resume 直接
    # 拿到一份按另一个上界分出来的布局 —— 这正是 09-11 `max_bias_warmup_steps`
    # 那次的形状。
    "stage2_last_window_max_states",
    "stage2_n_windows",
    "stage2_free_energy_densify_points",
)

# pilot 的采样语义标签（第 1 层）。改变遍历顺序 / 加密点如何起步时**必须**换掉
# 这个字符串——它是旧缓存失配的唯一依据，而不是协议版本号。
PILOT_TRAVERSAL_SEMANTICS = "high_lambda_endpoint_restart_v1"

# 旧缓存（2026-09-10 之前）没有 `pilot_traversal` 键。它们的加密点是从主 pilot
# 结束时的 λ=0 构型直接跳回高 λ 的，语义确实不同，所以读成这个值、并因此失配。
_LEGACY_PILOT_TRAVERSAL = "legacy_jump_from_decoupled_endpoint"


def _cached_pilot_traversal(cached_preopt) -> str:
    """这份缓存的 pilot 是按哪种采样语义测出来的。

    旧缓存（2026-09-10 之前）没有 `pilot_traversal` 键。但遍历语义**只在加密点
    上起作用** —— `_refine_pilot_grid_in_steep_segments` 改的是"加密点从哪个构型
    起步"，主网格那 N 个点的测法一个字没变。

    所以：如果这份 pilot 的 `path_diagnostics.pilot_points` 里**一个
    `is_refinement_point=True` 都没有**（加密从未触发，因为最陡段占比没超过阈值），
    那么新旧语义测到的就是同一批点，这份测量与当前语义等价，没有理由重烧 GPU。

    反之——有加密点、或者干脆读不到 `pilot_points`（无法证明）——一律按 legacy 处理，
    fail closed。
    """
    protocol_key = (cached_preopt or {}).get("protocol_key")
    payload = protocol_key.get("payload") if isinstance(protocol_key, dict) else None
    recorded = (payload or {}).get("pilot_traversal")
    if recorded:
        return str(recorded)

    points = ((cached_preopt or {}).get("path_diagnostics") or {}).get("pilot_points")
    if not isinstance(points, list) or not points:
        return _LEGACY_PILOT_TRAVERSAL
    if any(bool(point.get("is_refinement_point")) for point in points):
        return _LEGACY_PILOT_TRAVERSAL
    # 没有任何加密点 ⟹ 遍历语义从未起作用 ⟹ 与当前语义等价。
    return PILOT_TRAVERSAL_SEMANTICS


def _split_preopt_protocol_key(protocol_key):
    """把一份 preopt 指纹拆成 (第 1 层 采样, 第 2 层 派生路径)。

    对**旧缓存**做两件补全，好让比较有意义而不是"缺字段所以全不等"：

    * 缺 `pilot_traversal` ⟹ 补成 `_LEGACY_PILOT_TRAVERSAL`。它会与当前值不等，
      **这是对的**：那些缓存的采样语义真的不同。
    * 缺第 2 层的键 ⟹ 补成 `None`。旧缓存本来就没记过这些配置，
      所以"派生层未知"，只能按不匹配处理（但第 1 层若匹配，仍可离线重算）。

    `protocol_key` 为 None / 非 dict 时返回两个空 dict —— 调用方按不匹配处理。

    🔑 [2026-09-10] 入参可以是 `_protocol_fingerprint()` 的**外壳**
    `{schema_version, sha256, payload}`，也可以是裸的 payload dict，两种都认。

    这里原来只按裸 payload 处理，而两个生产调用点（`_preopt_protocol_key()` 的
    返回值）传进来的全是外壳 ⟹ 五个派生键在顶层一个都取不到、`derived` 恒为
    5 个 `None` ⟹ `_derived_match` **恒 True**；同时 `sampling` 里混进了整个
    `sha256`（派生键一变它就变）、而 `pilot_traversal` 在顶层取不到又被
    `setdefault` 补成 legacy ⟹ `_sampling_match` 对任何新格式缓存**恒 False**。
    两头相反方向地失效，"第 1 层匹配、只有第 2 层变了"这个条件永远不成立，
    整套两层拆分是死代码。单元测试没抓到是因为它们喂的是裸 payload。
    """
    if not isinstance(protocol_key, dict):
        return {}, {}
    if isinstance(protocol_key.get("payload"), dict):
        protocol_key = protocol_key["payload"]
    derived = {
        name: protocol_key.get(name) for name in _PREOPT_DERIVED_PATH_KEYS
    }
    sampling = {
        key: value
        for key, value in protocol_key.items()
        if key not in _PREOPT_DERIVED_PATH_KEYS
    }
    sampling.setdefault("pilot_traversal", _LEGACY_PILOT_TRAVERSAL)
    if sampling.get("pilot_traversal") is None:
        sampling["pilot_traversal"] = _LEGACY_PILOT_TRAVERSAL
    return sampling, derived



def _q8(values):
    """把一组 λ 量化到路径记录的比较口径（8 位小数），用于集合差集比较。"""
    return [round(float(v), 8) + 0.0 for v in values]

def _ie_gate_budget_for_protocol_key() -> int:
    """`IBS_LOCAL_MBAR_GATE_MAX_BATCHES` 的当前值，供协议指纹使用。

    惰性读取：`ibs_engine` 在本模块别处也是按需 import 的，不在顶层硬绑。
    读不到就 fail closed —— 指纹里放一个猜出来的数比不放更糟。
    """
    import ibs_engine as _ie
    return int(_ie.IBS_LOCAL_MBAR_GATE_MAX_BATCHES)


