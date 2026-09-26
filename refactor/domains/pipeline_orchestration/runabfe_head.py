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

import faulthandler
import os
import sys
import json
import argparse

# 🔑 [2026-09-10] SEGV 必须留下现场。
#
# 本进程里绝大部分时间跑在 OpenMM 的 C++/CUDA 里，那边一旦踩坏内存，Python 侧
# 得到的是一个信号：终端上只会多一行 `SEGV`，**没有 traceback、没有行号、
# 什么都没有**。2026-09-10 那次就是这样——从"跑不起来"到定位到
# `XmlSerializer.serialize(win_sys)` 这一行，花了一整天，而全部信息量就是
# "日志停在哪一条 print"。
#
# `faulthandler` 是 stdlib，enable() 之后 SIGSEGV/SIGBUS/SIGFPE/SIGABRT 会把
# 当时的 Python 调用栈打到 stderr 再让进程死。它**不改变任何行为**：不拦截、
# 不恢复、不吞信号，进程照样以同样的方式退出，只是多一段栈。零运行开销。
faulthandler.enable()
import logging
import shutil
import hashlib
import glob
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import openmm
from openmm import app, unit, Vec3, XmlSerializer

# 确保只导入所需模块，避免循环依赖
# 删除 runabfe.py 中手写的 NumpyEncoder 类
# 修改导入语句：
from abfe_core import (
    preflight_physical_inputs,
    # [PBC-01] mmCIF 往返造出来的假键：唯一实现。
    prune_topology_bonds_unsupported_by_system,
    # 整分子周期回卷：连通性只信 System，不信 topology 的键（唯一实现）。
    image_molecules_by_system,
    ACESoftcorePotential, UnitFormatter, calculate_boresch_analytical_correction,
    calc_boresch_from_last_frame, GeometricRestraintEstimator, OrbBoreschEstimator,
    DEXPSurrogatePotential, LambdaDependentBoreschForce, ensure_owned_system,
    NumpyEncoder, THERMODYNAMIC_CYCLE_DOC, assess_boresch_harmonicity,  # ✅ 统一从 abfe_core 导入
    # [P1-12] final-result 统一 schema/sanity gate（三处共用：pipeline cache
    # 早退、本文件两腿汇总/回退分析、combine_binding_free_energy）。
    validate_final_leg_result,
    FinalResultValidationError,
    combine_binding_free_energy,  # [ATT-09] 热力学循环闭合的唯一实现
    constraint_identity_fingerprint,
    ligand_conformer_fingerprint,  # [P0-12b] 起始构象进溶剂腿缓存身份
    solvent_box_edge_nm, SOLVENT_NONBONDED_CUTOFF_NM,  # 溶剂盒唯一实现，勿在此重复
    resolve_water_model_xml,  # 溶剂腿水模型必须从复合物 .top 反推
    resolve_membrane_protocol,  # 膜体系协议唯一解析实现（B1）
    resolve_environment_type,  # 只看 config 的环境类型规范化（不需要 topology）
    resolve_charge_treatment,  # 净电荷处理协议唯一校验实现（B2）
    validate_charge_transfer_reservoir_correction,
    CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,  # 已闭环实验路线；C4/C5 前不生产验收
    CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
    CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED,  # 溶剂腿能力状态，进 provenance
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
    co_alchemical_ion_builder_identity_payload,
    COION_COION_MIN_IMAGE_INITIAL_NM,  # §2.2 多个 reserved co-ion 彼此的安全边距
    CO_ALCHEMICAL_ION_RESIDUE_NAMES,  # reserved co-ion dummy 的残基名判据
    TOTAL_CHARGE_CONSERVATION_TOLERANCE_E,  # 派生 co-ion 后的总电荷守恒容差
    charge_treatment_qualification_payload,
    LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E,
    WATER_MOLECULE_NAMES,
    minimum_image_displacement_nm,
    ensure_barostat_for_protocol,
    PRE_EQUILIBRATION_TIMESTEP_PS,
    # [2026-09-03] equilibrium_is_done() 判断"合法收敛早停 vs 被中断"要用到收敛地板。
    PRE_EQUILIBRATION_CONVERGENCE_MIN_STEPS,
)
# PLAN_openmm_8_6_remd_backend.md P0：REMD 后端选择器。本模块不 import openmm
# 顶层，旧环境仍可导入。
import free_energy_engine

from abfe_pipeline import (
    guard_run_directory,
    ABFEPipeline, TraditionalABFEPipeline, _collect_pipeline_provenance, _pme_u_kn_meta_payload,
    _pre_equilibration_fingerprint,
    # [P1-07] 外层完成判断的严格校验 helper：真实 DCD parser + 目标 Simulation
    # loadCheckpoint。与 pipeline 内部 resume 用同一套判据，见 equilibrium_is_done。
    _is_checkpoint_valid, _is_traj_valid, _build_platform_props,
)
from ibs_engine import (
    solve_stage_integrated,
    generate_overlapping_windows,
    pme_self_correction_prefactor_kj,
    pme_self_correction_energy_kj,
    lambda_endpoint_diagnostics,
    synthetic_mbar_u_kn,
    TraditionalMBARAnalyzer,
    # [P1-04] analyze-only 回退路径与生产 loader 共用的窗口完整性 helper。
    _assert_expected_windows_all_loaded,
    IBSIncompleteStageCoverageError,
    # 配体净电荷的**唯一**实现。B2 的校验层刻意只吃一个 float，不自己再数一遍
    # 电荷，避免出现第二套净电荷判据（abfe_core 在 ibs_engine 下层，不能反向 import）。
    _compute_ligand_net_charge,
) # ✅ 保持从 ibs_engine 导入
from abfe_preoptimizer import DualLambdaPreOptimizer, build_aces_probe_system_dual_lambda
from local_residual.openmm_plugin import (
    FEATURE_NAME as OUTER_LAMBDA_RESIDUAL_FEATURE_NAME,
    build_outer_lambda_local_residual_runtime,
)
from local_residual.em_no_residual import (
    install as install_outer_lambda_em_policy,
    uninstall as uninstall_outer_lambda_em_policy,
)

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# ---------------------------------------------------------------------------
# 两条腿的配体约束/HMR 身份比较
# ---------------------------------------------------------------------------
# 🔑 [2026-09-01] 质量比较必须带容差，且容差要有物理依据。
#
# 起因（4W53 实测）：两条腿的配体是同一个分子、约束对逐位相同，但质量表来源不同 ——
#   complex 腿走 GROMACS 拓扑（LIG.itp，AMBER 表）：C = 12.010736, H = 1.007941
#   solvent 腿走 OpenMM 元素表                    ：C = 12.010780, H = 1.007947
# 相对差 3.7e-6 / 6.0e-6。`comparison_sha256` 是逐位哈希，于是整个热力学循环
# 在最后一步被拦下，两条腿都已经算完了。
#
# 为什么 5e-4 amu 这个门槛是安全的（不是为了让它变绿凑出来的）：
#   * 这道门要抓的是**同位素级/HMR 级**的改动。HMR 把 H 从 1.008 抬到 3.0~4.0
#     （Δ ≈ 2 amu），重原子相应扣减（C 12.011 → ~5.96，Δ ≈ 6 amu）；氘代 Δ ≈ 1 amu。
#     最小的真实改动也是 **1 amu 量级**。
#   * 质量表来源差异是 **5e-5 amu 量级**。
#   两者相差 4 个数量级，5e-4 amu 落在正中间，两边各有 10× 和 2000× 余量。
#
# 约束**对与距离**仍然逐位比较，不给任何容差：约束集合变了就是不同的
# constrained Hamiltonian，与质量表来源无关。
#
# 物理上为什么可以放行：配体质量在同一条腿内不随 λ 变化，动量部分与
# constrained-metric 因子在腿内精确抵消；腿间残留的敏感度 ~ kT × 4e-6，
# 即 ~1e-5 kJ/mol 量级，比这条腿本身的统计误差（0.6 kJ/mol）小五个数量级。
_LIGAND_MASS_COMPARISON_DECIMALS = 3   # 5e-4 amu


def _ligand_constraint_comparable_view(identity: Dict) -> Dict:
    """把一份 constraint_identity 化成**腿间可比**的形式。

    两条腿的全局原子编号必然不同，所以约束用配体内局部序号表示；质量按
    `_LIGAND_MASS_COMPARISON_DECIMALS` 量化。对新算出来的和已落盘的 identity
    都适用（只吃 ligand_indices / ligand_masses_amu / ligand_constraints 三个字段）。
    """
    indices = [int(i) for i in (identity.get("ligand_indices") or [])]
    local = {g: k for k, g in enumerate(indices)}
    masses = [
        round(float(m), _LIGAND_MASS_COMPARISON_DECIMALS)
        for _, m in (identity.get("ligand_masses_amu") or [])
    ]
    constraints = sorted(
        (
            str(local.get(int(a), "external")),
            str(local.get(int(b), "external")),
            round(float(d), 12),
        )
        for a, b, d in (identity.get("ligand_constraints") or [])
    )
    return {"n_atoms": len(indices), "masses": masses, "constraints": constraints}


def _describe_constraint_identity_mismatch(
    complex_identity: Dict, solvent_identity: Dict
) -> str:
    """说清楚到底差在哪 —— 光打两个 sha256 对定位毫无帮助。"""
    cv = _ligand_constraint_comparable_view(complex_identity)
    sv = _ligand_constraint_comparable_view(solvent_identity)
    lines = []
    if cv["n_atoms"] != sv["n_atoms"]:
        lines.append(f"配体原子数不同: complex={cv['n_atoms']} solvent={sv['n_atoms']}")
    if cv["constraints"] != sv["constraints"]:
        lines.append(
            f"配体约束集合不同（局部序号）: complex={cv['constraints']} "
            f"solvent={sv['constraints']}"
        )
    c_raw = [float(m) for _, m in (complex_identity.get("ligand_masses_amu") or [])]
    s_raw = [float(m) for _, m in (solvent_identity.get("ligand_masses_amu") or [])]
    if len(c_raw) == len(s_raw) and c_raw:
        worst = max(
            range(len(c_raw)),
            key=lambda k: abs(c_raw[k] - s_raw[k]) / max(abs(s_raw[k]), 1e-30),
        )
        d_abs = abs(c_raw[worst] - s_raw[worst])
        d_rel = d_abs / max(abs(s_raw[worst]), 1e-30)
        lines.append(
            f"质量最大偏差在配体局部原子 {worst}: complex={c_raw[worst]!r} "
            f"solvent={s_raw[worst]!r}（绝对 {d_abs:.3e} amu，相对 {d_rel:.3e}；"
            f"容差 {10 ** -_LIGAND_MASS_COMPARISON_DECIMALS:.0e} amu）"
        )
        if d_abs < 10 ** -_LIGAND_MASS_COMPARISON_DECIMALS:
            lines.append(
                "  ↑ 这个量级是**质量表来源差异**（如 GROMACS 拓扑 vs OpenMM 元素表），"
                "不是 HMR/同位素；已在容差内，不应由这道门拦下。"
            )
    return "；".join(lines) if lines else "（未能定位到具体差异字段）"


def _assert_matching_ligand_constraint_identity(
    complex_system,
    complex_ligand_indices,
    solvent_system,
    solvent_ligand_indices,
    *,
    context: str = "ABFE two-leg cycle",
):
    """Require both legs to use the same ligand constrained Hamiltonian.

    The global atom numbering can differ between complex and solvent systems,
    so comparison uses the ligand-local ``comparison_sha256`` emitted by
    :func:`constraint_identity_fingerprint`.  A mismatch means that the
    nominally cancelling constrained configurational factor (including HMR
    masses) is not the same on both legs; no pseudo-Jacobian constant may be
    guessed in its place.
    """
    complex_identity = constraint_identity_fingerprint(
        complex_system, complex_ligand_indices
    )
    solvent_identity = constraint_identity_fingerprint(
        solvent_system, solvent_ligand_indices
    )
    # 逐位相同是快路径；不同的时候再按物理判据看差异是否在容差内。
    if (
        complex_identity.get("comparison_sha256")
        != solvent_identity.get("comparison_sha256")
    ):
        if _ligand_constraint_comparable_view(
            complex_identity
        ) != _ligand_constraint_comparable_view(solvent_identity):
            raise RuntimeError(
                f"{context} 的配体约束/HMR 身份不一致；拒绝用伪 Jacobian 修正掩盖差异。"
                f" {_describe_constraint_identity_mismatch(complex_identity, solvent_identity)}"
                f" [complex={complex_identity.get('comparison_sha256')},"
                f" solvent={solvent_identity.get('comparison_sha256')}]"
            )
        log.warning(
            "[WARN] 两条腿的配体质量表来源不同但在容差内，按同一 constrained Hamiltonian "
            "处理：%s",
            _describe_constraint_identity_mismatch(complex_identity, solvent_identity),
        )
    return complex_identity


def _assert_matching_result_constraint_identity(
    complex_result: Dict,
    solvent_result: Dict,
    *,
    context: str,
) -> Dict:
    """Validate the persisted constraint identity before post-processing.

    A result written before the identity field was introduced cannot prove
    that its constrained Hamiltonian matches the other leg.  Treat that as an
    invalid/legacy artifact instead of silently applying the retired
    mass-based Jacobian correction.
    """
    complex_identity = complex_result.get("constraint_identity")
    solvent_identity = solvent_result.get("constraint_identity")
    if not isinstance(complex_identity, dict) or not isinstance(solvent_identity, dict):
        raise RuntimeError(
            f"{context} 缺少 constraint_identity；旧结果无法证明两条腿的约束/HMR 身份一致，"
            "拒绝继续汇总。"
        )
    c_hash = complex_identity.get("comparison_sha256")
    s_hash = solvent_identity.get("comparison_sha256")
    if not c_hash or not s_hash:
        raise RuntimeError(
            f"{context} 的持久化 constraint_identity 缺少 comparison_sha256；"
            "无法证明两条腿的约束/HMR 身份一致，拒绝汇总。"
        )
    if c_hash != s_hash:
        if _ligand_constraint_comparable_view(
            complex_identity
        ) != _ligand_constraint_comparable_view(solvent_identity):
            raise RuntimeError(
                f"{context} 的持久化配体约束/HMR 身份不一致；拒绝汇总。"
                f" {_describe_constraint_identity_mismatch(complex_identity, solvent_identity)}"
                f" [complex={c_hash!r}, solvent={s_hash!r}]"
            )
        log.warning(
            "[WARN] 持久化身份的配体质量表来源不同但在容差内，按同一 constrained "
            "Hamiltonian 处理：%s",
            _describe_constraint_identity_mismatch(complex_identity, solvent_identity),
        )
    return complex_identity

# ---------------------------------------------------------------------------
# 常量与预设
# ---------------------------------------------------------------------------
PRESET_CONFIGS = {
    "test": {
        "n_steps_per_window": 10000,
        "steps_per_update": 500,
        "stage1_n_states": 12,
        "stage2_n_states": 17,
        # 见 production 预设里这两个键的说明（审计 #31）。
        "stage2_production_budget_steps": None,
        "stage2_max_production_blocks_per_window": 4,
        # 见 production 预设里 `stage2_first_window_max_states` 的说明（2026-09-16 拍板 / BM-04）。
        "stage2_first_window_max_states": 4,
    },
    "production": {
        "n_steps_per_window": 250000,
        "steps_per_update": 500,
        # 去电荷腿走 PME-REMD（每个 λ 状态各建一个 replica context），
        # 状态数越多同时常驻显存的 replica 越多。
        #
        # 🔑 [2026-09-03] 12 → 8。原注释断言「8 个状态相邻窗口能量差过大、容易在
        # 交换/重加权时能量过高」——那是一条**未经测量的先验判断**，实测数据方向
        # 相反。把已有运行的 decharging_exchange_diagnostics.json 里的平均交换
        # 接受率 P_acc 按高斯近似反解成每边热力学长度
        #     δ = 2·erfc⁻¹(P_acc)          （⟨P_acc⟩ = erfc(δ/2)）
        # 得到 4W53/甲苯：
        #     复合物腿 N=12 → P=0.959~0.967 → δ≈0.06~0.07 → L_coul≈0.64~0.80
        #     复合物腿 N=8  → P=0.948       → δ≈0.09      → L_coul≈0.64
        #     溶剂腿   N=12 → P=0.850       → δ≈0.27      → L_coul≈2.93
        #     溶剂腿   N=8  → P=0.773       → δ≈0.41      → L_coul≈2.86
        # L 跨 N=8/N=12 不变（体系性质，不随离散化变），这是反解自洽性的检验。
        # 按「相邻态 δ≈1（σ_ΔU≈1 kT）」的判据，这个体系复合物腿只需 2 态、溶剂腿
        # 只需 4 态；N=8 已经是留了余量的保守值，N=12 是明显超采样。
        # N=8 的实跑（output/）P_acc=0.948、最差轮 0.714，没有出现原注释担心的
        # 「能量过高」，交换反而过易。
        #
        # ⚠️ 这个 8 是**当前基准体系的经验值，不是体系无关的默认值**。L_coul 随
        # 配体电荷分布走（∝√Σq²，带净电荷还要加 Born 项），甲苯几乎无电荷所以这条
        # 腿近乎 no-op。极性配体、尤其带电配体必须调大——与 fepsuite
        # (rundir_template/para_conf.zsh) 的 "NCHARGE=12, for large molecules &
        # strongly charged molecules we recommend greater values" 是同一条经验。
        # 换体系时请读该体系自己 decharging_exchange_diagnostics.json 的 P_acc
        # 重新定 N，不要沿用这个数。
        "stage1_n_states": 8,
        "stage2_n_states": 17,
        # 🔑 [2026-09-14 审计 #31] Stage-2 控制器的三道预算闸，此前**全仓只有读侧**
        # （`Stage2RepairController.__init__` 从 run_provenance.json 的 config 里读），
        # 配置/预设/CLI 全都没有 ⟹ provenance 里永远没有这几个键 ⟹ 生产预算闸
        # `cap_known=False` 从未生效过。默认值严格等于今天的硬编码兜底，行为不变：
        #   · stage2_production_budget_steps = None —— 上限**未知**，不是 0。
        #     控制器写死「未知时不拦」，所以 null 就是今天的行为；**不要**随手拍一个
        #     具体数字，那等于给所有现存 run 引入一个没人验证过的终止条件。
        #   · stage2_max_production_blocks_per_window = 4 —— 同 preoptimizer 兜底。
        "stage2_production_budget_steps": None,
        "stage2_max_production_blocks_per_window": 4,
        # 🔑🔑 [2026-09-16 维护者拍板 / BM-04] **耦合端（win0）专用的态数上限。**
        # 此前这个键**只存在于 `abfe_config.json`**，而 `abfe_config.json` 不会被
        # 自动加载（`--config` 的 argparse 默认是 None），预设与 CLI 也都没有它
        # ⟹ 除非有人显式把那份样例传进来，否则**它从未在任何一次真实运行里生效过**。
        # 实测代价：benchmark 14/14 config、3/3 预设都没有这个键，于是 win0 照样
        # 能拿到 8 个态；而 win0 一旦不可信，`tail_repartition_anchor` 恒返回 None
        # （它之前没有可冻结的窗口）⟹ 缩跨度一族动作在**构造上**全部不可行 ⟹
        # 只剩插 λ ⟹ 插到末窗顶满就 `NO_FEASIBLE_ACTION`。真机 4 个 run 死在这里。
        # ⚠️ 本键在 `_PREOPT_DERIVED_PATH_KEYS` 里（**第 2 层**派生路径）：
        # 加上它会让**全部现存 run 的 Stage-2 窗口缓存失配重采**；
        # 第 1 层（pilot 采样语义）不变 ⟹ **pilot/preopt 与 Stage 0/1 全部保留**，
        # 布局离线重算。维护者已按「跑得准优先」接受这个代价。
        # 要回到旧行为：把本键设为 None（不是删掉——删掉会退回"预设里没有"的老坑）。
        "stage2_first_window_max_states": 4,
    },
    "high_accuracy": {
        "n_steps_per_window": 500000,
        "steps_per_update": 500,
        "stage1_n_states": 24,
        "stage2_n_states": 17,
        # 见 production 预设里这两个键的说明（审计 #31）。
        "stage2_production_budget_steps": None,
        "stage2_max_production_blocks_per_window": 4,
        # 见 production 预设里 `stage2_first_window_max_states` 的说明（2026-09-16 拍板 / BM-04）。
        "stage2_first_window_max_states": 4,
    },
}

MAIN_SYSTEM_CACHE_PROTOCOL_VERSION = 3
# version 4: 溶剂盒不再交给 addSolvent(padding=...) 自己推导。旧路径在本仓库
#   产出过 box = 2*padding 的 3.000 nm 立方盒——溶质尺寸对盒长的贡献是 0，而
#   配体最长轴 1.257 nm，每侧只剩 0.87 nm 溶剂，第二水化层直接和周期镜像重叠。
#   所有 v3 及更早的溶剂缓存都是在那个盒子里建的，必须整体作废重建。
# 4 → 5（2026-08-04，P0-12b）：身份里加入配体**起始构象**指纹。
# 实测同一个分子换一个起始构象，溶剂腿去电荷从 62.80 变成 191.05 kJ/mol（P0-12），
# 而旧口径里这两次都判"缓存有效"——起始构象决定了溶剂腿采到哪个构象族，它必须进身份。
# 5 → 6（2026-08-05，B4）：charge-transfer 路线的溶剂腿 builder 落地——现在会额外
# 插入 reserved co-ion dummy，manifest 加了 `charge_treatment` /
# `reserved_coion_*` 字段（§4.5）。旧缓存不含这些粒子，必须 fail closed 重建，
# 不能被静默当成"已经有 co-ion"复用（那会让下游 `select_co_alchemical_ion_once`
# 在一个没有 dummy 的盒子里报错，或更糟——如果盒子恰好凑巧有别的零电荷粒子，
# 静默选错身份）。
# ⚠️ 升版本会让**所有**已有溶剂腿盒缓存重建（盒构建只要几十秒；真正贵的 stage
# 采样缓存由 `_stage_protocol_key` 单独把关，不受本版本号影响）。
SOLVENT_CACHE_PROTOCOL_VERSION = 7
DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR = 0.15
SOLVENT_PADDING_NM = 1.5


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
    builder_identity: Optional[Dict] = None,
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
    if builder_identity is not None:
        payload["reserved_coion_builder_identity"] = builder_identity
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
    positions=None,
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
        # ===================== [2026-09-10] 已删除 =====================
        # 原来这里是：
        #     "complex_system_xml_sha256": hashlib.sha256(
        #         XmlSerializer.serialize(system).encode("utf-8")).hexdigest()
        #
        # 为了写一个**全仓零读者**的字段，把整个 System（本体系 30710 原子）
        # 序列化成一份多兆字节的字符串。grep 确认过：除了这一行的写入，
        # 没有任何代码读 `complex_system_xml_sha256`。
        #
        # 它还与本文件自己的契约矛盾——见下方 `_main_cache_identity` 附近那条
        # 注释：「身份只认用户输入（identity_sha256 来自 gro/top/config）」。
        # 这里哈的却是**当次在内存里重建出来的 System**，是自产产物。
        # `abfe_pipeline._system_xml_hash` 已经因为同一个理由被改成恒返回 None
        # （见那里的说明：协议版本号已经承担"算法变了"的信号，sha256 是多余的
        # 第二套机制），这一处是当时漏掉的同类。
        #
        # 真正的身份下面全都在：配体残基名 / 原子索引 / 原子数 / 净电荷 /
        # 逐原子非键参数 / 用户 .top 与 include 文件的哈希。
        #
        # ⚠️ 代价：payload 少一个键 ⟹ `identity_sha256` 变 ⟹ **已有的溶剂腿缓存
        # 会失配一次、需要重建**。这是一次性的。
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
    # [P0-12b] 配体**起始构象**进身份。用内部距离矩阵指纹（刚体平移/旋转不敏感），
    # 所以只有构象真的变了才让盒缓存失效。positions 缺省时明确记 None 而不是静默略过。
    if positions is not None:
        pos_nm = (
            np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
            if hasattr(positions, "value_in_unit")
            else np.asarray(positions, dtype=np.float64)
        )
        heavy = [
            int(atom.index)
            for atom in topology.atoms()
            if int(atom.index) in set(indices)
            and atom.element is not None
            and float(atom.element.mass.value_in_unit(unit.dalton)) >= 2.5
        ]
        payload["ligand_start_conformer"] = ligand_conformer_fingerprint(
            pos_nm, indices, heavy_indices=heavy or None
        )
    else:
        payload["ligand_start_conformer"] = None
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {"identity": payload, "identity_sha256": hashlib.sha256(encoded.encode()).hexdigest()}

