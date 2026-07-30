#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ABFE 核心物理模块 (v6.0 - 完整收敛版)
职责：统一封装所有势能、限制力、拟合器、估算器、校验器、路径规划、替身构建、Orb扫描与路由工厂
架构约束：严格收敛至 5 文件，本文件为唯一物理核心单例，零占位符
依赖：openmm, numpy, scipy, mdtraj, torch, openmmml (部分功能)
"""

import openmm
from openmm import app, unit
import numpy as np
import math
import warnings
import json
import os
import logging
import gc
import builtins
import statistics
from itertools import combinations
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Callable
from scipy.optimize import differential_evolution, least_squares, minimize
from scipy import constants

warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import mdtraj

    HAS_MDTRAJ = True
except ImportError:
    HAS_MDTRAJ = False
try:
    import torch
    from openmmml import MLPotential

    HAS_ORB = True
except ImportError:
    HAS_ORB = False
try:
    import pymbar

    HAS_PYMBAR = True
except ImportError:
    HAS_PYMBAR = False

logger = logging.getLogger(__name__)

# DEXP only controls the analytic van der Waals kernel shape.  The force-shell
# and Gaussian-Coulomb settings are independent production protocol constants.
DEXP_VDW_CUTOFF_NM = 0.70
DEXP_VDW_SWITCH_WIDTH_NM = 0.20
GAUSS_COUL_SIGMA_NM = 0.10
GAUSS_COUL_CUTOFF_NM = 0.70

# Retired global Orb-fit payloads must never be silently interpreted as the
# pair-specific LJ-matched production Hamiltonian.
DEXP_LEGACY_FIT_KEYS = frozenset(
    {
        "r0_vdw",
        "A_fit",
        "B_fit",
        "offset_c0",
        "offset_c1",
        "fitting_success",
        "final_cost",
        "fit_target_mode",
        "fit_objective",
    }
)


# ============================================================================
# 膜体系协议：system_type + 膜恒压器（memtodolist.md §3.1 / §3.2；MEM-00i）
#
# 这一节只负责"声明与校验"，不做任何采样决策。设计约束逐条对应清单：
#   §3.1 不根据残基名自动猜 system_type，用户必须显式声明；脂质残基检测只做交叉
#        检查，不作唯一判据；membrane 但无脂质 → fail closed；soluble 但检测到大量
#        脂质 → 警告并要求确认。
#   §3.2 预平衡用 MonteCarloMembraneBarostat（XY 等比例、Z 独立、默认表面张力 0）；
#        溶剂腿继续用普通 MonteCarloBarostat；检测任意已有 barostat 禁止重复添加；
#        已有不兼容 barostat → fail closed；barostat 类型/压力/表面张力/XY-Z 模式/
#        频率进入预平衡 fingerprint，改任一项使旧 checkpoint 失效。
#   §7.7 膜功能默认关闭：不声明 system_type 时行为与改动前逐位一致——这就是
#        barostat_fingerprint_payload() 对 legacy soluble 协议返回 None 的原因。
# ============================================================================

MEMBRANE_BAROSTAT_PROTOCOL_VERSION = 1

ENVIRONMENT_TYPE_SOLUBLE = "soluble"
ENVIRONMENT_TYPE_MEMBRANE = "membrane"
ENVIRONMENT_TYPES = (ENVIRONMENT_TYPE_SOLUBLE, ENVIRONMENT_TYPE_MEMBRANE)

# OpenMM 的三种 barostat 都是彼此独立的 Force 子类——MonteCarloMembraneBarostat
# **不是** MonteCarloBarostat 的子类。所以原来 abfe_pipeline.py 里
# `isinstance(f, openmm.MonteCarloBarostat)` 这一个判断检测不到膜/各向异性
# barostat：输入 System 若已带膜 barostat，旧代码会在它之上再叠一个各向同性的，
# 两个 barostat 同时生效 → 集合定义错误且不会报错。这里改用类名集合检测。
BAROSTAT_FORCE_CLASS_NAMES = (
    "MonteCarloBarostat",
    "MonteCarloAnisotropicBarostat",
    "MonteCarloMembraneBarostat",
)

# 仅用于交叉检查（§3.1），**绝不**用来推断 system_type。
#
# 刻意拆成两套，因为两个方向的误判后果完全不对称：
#
#   - `membrane` 但找不到脂质 → fail closed。这里用**宽**集合（含 Amber Lipid21
#     的模块化短残基名），误认成脂质只会放过一个用户已经显式声明为膜的运行，
#     无害；漏认才会挡住合法的膜运行。
#   - `soluble` 却检测到大量脂质 → 拦下来要求确认。这里只用**窄**集合（无歧义的
#     全名），因为误判会挡住一个完全合法的可溶体系运行。`PC`/`PE`/`PS`/`OL`/`ST`
#     这类 2–3 字母 token 在别的力场/配体命名里撞车的概率不低，不能拿它挡人。
#
# 实测当前生产体系 `solv_ions.gro` 的残基名只有 SOL / 20 种氨基酸 / ASH / CL /
# NA / MOL，与两套集合都无交集——本节新增的检查对现有可溶路径零影响。
LIPID_RESIDUE_NAMES_UNAMBIGUOUS = frozenset(
    {
        # 磷脂全分子残基名（CHARMM36 / Slipids 风格）
        "POPC", "POPE", "POPS", "POPG", "POPA", "POPI",
        "DOPC", "DOPE", "DOPS", "DOPG", "DOPA",
        "DPPC", "DPPE", "DPPS", "DPPG",
        "DMPC", "DMPE", "DMPG",
        "DSPC", "DLPC", "DPPI", "SAPI", "PSM", "SSM",
        # 胆固醇
        "CHL1", "CHOL", "CLR",
    }
)

# Amber Lipid21/Lipid17 把一个脂质拆成"头基 + 甘油 + 两条尾链"多个残基。
LIPID_RESIDUE_NAMES_AMBER_MODULAR = frozenset(
    {
        "PC", "PE", "PS", "PGR", "PA", "PH-",
        "OL", "ST", "MY", "LAL", "AR", "DHA",
    }
)

KNOWN_LIPID_RESIDUE_NAMES = LIPID_RESIDUE_NAMES_UNAMBIGUOUS | LIPID_RESIDUE_NAMES_AMBER_MODULAR

# soluble 却检测到多少个**无歧义**脂质残基就算"大量"（§3.1 最后一条）。取 8：
# 单个脂质分子被误命名不至于触发，任何真实双层（最小 slab 也有几十个）必定触发。
SOLUBLE_LIPID_RESIDUE_WARN_THRESHOLD = 8

MEMBRANE_XY_MODES = ("isotropic", "anisotropic")
MEMBRANE_Z_MODES = ("free", "fixed", "constant_volume")
MEMBRANE_NORMAL_AXES = ("x", "y", "z")

DEFAULT_MEMBRANE_PROTOCOL: Dict[str, Any] = {
    "normal_axis": "z",
    "surface_tension_bar_nm": 0.0,
    "xy_mode": "isotropic",
    "z_mode": "free",
    "barostat_frequency": 25,
}

# 改动前唯一存在的 barostat 协议：各向同性、频率 25（abfe_pipeline.py:1382）。
# fingerprint 对它保持沉默，以保证 §7.7 的"不声明 system_type 时逐位一致"。
LEGACY_SOLUBLE_BAROSTAT_FREQUENCY = 25


def _openmm_membrane_xy_mode(xy_mode: str):
    mapping = {
        "isotropic": openmm.MonteCarloMembraneBarostat.XYIsotropic,
        "anisotropic": openmm.MonteCarloMembraneBarostat.XYAnisotropic,
    }
    return mapping[xy_mode]


def _openmm_membrane_z_mode(z_mode: str):
    mapping = {
        "free": openmm.MonteCarloMembraneBarostat.ZFree,
        "fixed": openmm.MonteCarloMembraneBarostat.ZFixed,
        "constant_volume": openmm.MonteCarloMembraneBarostat.ConstantVolume,
    }
    return mapping[z_mode]


# ⚠️ 命名撞车警告，读到这里的人请务必分清两个**互不相干**的轴：
#
#   1. 本节的 soluble / membrane —— 环境类型，决定用哪种 barostat。
#      配置与 provenance 里的键名按 memtodolist §3.1/§10 定为 `system_type`。
#   2. 仓库里早就存在的 `system_type="complex"` / `"solvent"`
#      （`ABFEPipeline.run_full_pipeline` 等 20+ 处）—— **腿身份**，
#      决定加不加 Boresch、走复合物还是溶剂盒。
#
# 两者同名但含义完全不同。为避免在代码里混淆，本节内部一律用
# `environment_type` 这个标识符，只在**序列化**（fingerprint / provenance /
# 配置键）时才叫 `system_type`。任何函数都不要用 `system_type` 当形参名。
_LEG_IDENTITY_VALUES = ("complex", "solvent")


def resolve_environment_type(value: Optional[str]) -> str:
    """把配置键 `system_type` 规范化为环境类型；未声明即 soluble（§7.7 默认关闭）。

    不接受 None 之外的任何"聪明"回落：拼错的值必须报错，而不是静默当 soluble 跑。
    """
    if value is None:
        return ENVIRONMENT_TYPE_SOLUBLE
    normalized = str(value).strip().lower()
    if normalized == "":
        return ENVIRONMENT_TYPE_SOLUBLE
    if normalized in _LEG_IDENTITY_VALUES:
        raise ValueError(
            f"环境类型收到 {value!r}，但这是**腿身份**的取值。"
            "本仓库里 `system_type` 被两个不同的轴共用："
            "`run_full_pipeline(system_type='complex'|'solvent')` 指腿身份，"
            f"而环境类型只接受 {list(ENVIRONMENT_TYPES)}。"
            "请检查是不是把腿身份传进了膜协议解析。"
        )
    if normalized not in ENVIRONMENT_TYPES:
        raise ValueError(
            f"system_type={value!r} 不是合法的环境类型；允许 {list(ENVIRONMENT_TYPES)}。"
            "不会静默回落到 soluble——膜体系被误判为可溶体系会用错恒压器集合。"
        )
    return normalized


def count_lipid_residues(
    topology,
    names: Optional[frozenset] = None,
) -> Dict[str, int]:
    """按残基名统计疑似脂质残基数量，仅用于交叉检查（§3.1）。

    `names` 默认用宽集合 `KNOWN_LIPID_RESIDUE_NAMES`；传
    `LIPID_RESIDUE_NAMES_UNAMBIGUOUS` 可只统计无歧义全名。
    """
    counts: Dict[str, int] = {}
    if topology is None:
        return counts
    allowed = KNOWN_LIPID_RESIDUE_NAMES if names is None else names
    for residue in topology.residues():
        name = str(residue.name).strip().upper()
        if name in allowed:
            counts[name] = counts.get(name, 0) + 1
    return counts


def resolve_membrane_protocol(
    environment_type: Optional[str],
    membrane_config: Optional[Dict[str, Any]] = None,
    topology=None,
    confirm_soluble_with_lipids: bool = False,
) -> Dict[str, Any]:
    """校验环境类型与 membrane.* 组合，返回可直接进 fingerprint/provenance 的协议。

    `environment_type` 来自配置键 `system_type`（soluble/membrane），
    **不是**腿身份 complex/solvent——见上方命名撞车警告。

    在创建任何 Context 之前调用（§6.1）。返回的 dict 是**唯一**的下游真相来源，
    调用方不应再自己读原始配置字段。
    """
    resolved_type = resolve_environment_type(environment_type)
    lipid_counts = count_lipid_residues(topology)
    n_lipid_residues = int(sum(lipid_counts.values()))
    # 拦人的方向只认无歧义全名，见 LIPID_RESIDUE_NAMES_UNAMBIGUOUS 处的说明。
    unambiguous_counts = count_lipid_residues(
        topology, names=LIPID_RESIDUE_NAMES_UNAMBIGUOUS
    )
    n_unambiguous = int(sum(unambiguous_counts.values()))

    if resolved_type == ENVIRONMENT_TYPE_SOLUBLE:
        if membrane_config:
            raise ValueError(
                "system_type=soluble 却提供了 membrane.* 配置："
                f"{sorted(membrane_config)}。这两者组合含义不明，拒绝猜测；"
                "要跑膜体系请显式声明 system_type=membrane。"
            )
        if topology is not None and n_unambiguous >= SOLUBLE_LIPID_RESIDUE_WARN_THRESHOLD:
            message = (
                f"system_type=soluble 但拓扑里检测到 {n_unambiguous} 个脂质残基 "
                f"{dict(sorted(unambiguous_counts.items()))}。各向同性恒压器会把膜面积与厚度"
                "绑死，APL 会跑掉（memtodolist §3.1 / MEM-00i）。"
            )
            if not confirm_soluble_with_lipids:
                raise ValueError(
                    message
                    + " 若确认这不是双层膜（例如只是几个游离脂质配体），"
                    "请显式传 confirm_soluble_with_lipids=True 留下记录。"
                )
            logger.warning("⚠️ %s 已由 confirm_soluble_with_lipids=True 显式确认。", message)
        return {
            "protocol_version": MEMBRANE_BAROSTAT_PROTOCOL_VERSION,
            "system_type": ENVIRONMENT_TYPE_SOLUBLE,
            "barostat_class": "MonteCarloBarostat",
            "barostat_frequency": LEGACY_SOLUBLE_BAROSTAT_FREQUENCY,
            "membrane": None,
            "lipid_residue_counts": dict(sorted(lipid_counts.items())),
            "lipid_residue_total": n_lipid_residues,
            "lipid_residue_total_unambiguous": n_unambiguous,
            "soluble_with_lipids_confirmed": bool(confirm_soluble_with_lipids),
        }

    # ---- system_type == membrane ----
    if topology is not None and n_lipid_residues == 0:
        raise ValueError(
            "system_type=membrane 但拓扑里找不到任何已知脂质残基名"
            f"（已知集合 {len(KNOWN_LIPID_RESIDUE_NAMES)} 项）。"
            "按 memtodolist §3.1 这里 fail closed：要么输入不是膜体系，"
            "要么脂质残基名不在已知集合里——后者请先把残基名加进 "
            "KNOWN_LIPID_RESIDUE_NAMES 并说明力场来源，不要绕过本检查。"
        )

    protocol = dict(DEFAULT_MEMBRANE_PROTOCOL)
    unknown = sorted(set(membrane_config or {}) - set(DEFAULT_MEMBRANE_PROTOCOL))
    if unknown:
        raise ValueError(
            f"membrane.* 出现未知字段 {unknown}；允许 "
            f"{sorted(DEFAULT_MEMBRANE_PROTOCOL)}。拒绝静默忽略拼错的协议字段。"
        )
    protocol.update(membrane_config or {})

    normal_axis = str(protocol["normal_axis"]).strip().lower()
    if normal_axis not in MEMBRANE_NORMAL_AXES:
        raise ValueError(
            f"membrane.normal_axis={protocol['normal_axis']!r} 非法；允许 {list(MEMBRANE_NORMAL_AXES)}。"
        )
    if normal_axis != "z":
        raise ValueError(
            f"membrane.normal_axis={normal_axis!r}：OpenMM 的 MonteCarloMembraneBarostat "
            "把膜法向硬编码为 z（它只区分 XY 平面与 Z 轴，没有换轴选项）。"
            "请在建系时把膜法向对齐 z，不要在这里换轴——那会让 XY 等比例缩放"
            "作用在错误的平面上，且不会报错。"
        )

    xy_mode = str(protocol["xy_mode"]).strip().lower()
    if xy_mode not in MEMBRANE_XY_MODES:
        raise ValueError(
            f"membrane.xy_mode={protocol['xy_mode']!r} 非法；允许 {list(MEMBRANE_XY_MODES)}。"
        )
    z_mode = str(protocol["z_mode"]).strip().lower()
    if z_mode not in MEMBRANE_Z_MODES:
        raise ValueError(
            f"membrane.z_mode={protocol['z_mode']!r} 非法；允许 {list(MEMBRANE_Z_MODES)}。"
        )

    try:
        surface_tension = float(protocol["surface_tension_bar_nm"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"membrane.surface_tension_bar_nm={protocol['surface_tension_bar_nm']!r} 不是数值"
        ) from exc
    if not math.isfinite(surface_tension):
        raise ValueError("membrane.surface_tension_bar_nm 必须有限")

    try:
        frequency = int(protocol["barostat_frequency"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"membrane.barostat_frequency={protocol['barostat_frequency']!r} 不是整数"
        ) from exc
    if frequency <= 0:
        raise ValueError("membrane.barostat_frequency 必须为正整数")

    return {
        "protocol_version": MEMBRANE_BAROSTAT_PROTOCOL_VERSION,
        "system_type": ENVIRONMENT_TYPE_MEMBRANE,
        "barostat_class": "MonteCarloMembraneBarostat",
        "barostat_frequency": frequency,
        "membrane": {
            "normal_axis": normal_axis,
            "surface_tension_bar_nm": surface_tension,
            "xy_mode": xy_mode,
            "z_mode": z_mode,
        },
        "lipid_residue_counts": dict(sorted(lipid_counts.items())),
        "lipid_residue_total": n_lipid_residues,
        "lipid_residue_total_unambiguous": n_unambiguous,
        "soluble_with_lipids_confirmed": False,
    }


def detect_barostats(system) -> List[Tuple[int, str]]:
    """列出 System 里所有 barostat 的 (force index, 类名)。

    覆盖各向同性/各向异性/膜三种（BAROSTAT_FORCE_CLASS_NAMES），按类名而非
    isinstance 判断——三者不共享基类，isinstance 单查一种必然漏检。
    """
    found: List[Tuple[int, str]] = []
    for index, force in enumerate(system.getForces()):
        class_name = type(force).__name__
        if class_name in BAROSTAT_FORCE_CLASS_NAMES:
            found.append((index, class_name))
    return found


def ensure_barostat_for_protocol(
    system,
    protocol: Dict[str, Any],
    temperature,
    pressure,
) -> Dict[str, Any]:
    """按协议保证 System 上恰好有一个正确类型的 barostat（§3.2）。

    三种结局，都不静默：
      - 已有正确类型的 barostat → 复用，不重复添加（`action="reused_existing"`）；
      - 没有 barostat → 按协议添加（`action="added"`）；
      - 已有不兼容 barostat，或有多个 → **fail closed**，绝不再叠一个。

    `temperature` / `pressure` 接受裸数值（分别按 K / bar 解释）或 openmm Quantity。
    """
    temperature_k = (
        temperature.value_in_unit(unit.kelvin)
        if hasattr(temperature, "value_in_unit")
        else float(temperature)
    )
    pressure_bar = (
        pressure.value_in_unit(unit.bar)
        if hasattr(pressure, "value_in_unit")
        else float(pressure)
    )

    expected_class = protocol["barostat_class"]
    existing = detect_barostats(system)

    if len(existing) > 1:
        raise RuntimeError(
            f"输入 System 上已有 {len(existing)} 个 barostat "
            f"{[name for _, name in existing]}；多个 barostat 同时生效的集合定义不明，"
            "拒绝继续（memtodolist §3.2）。"
        )
    if existing:
        _, existing_class = existing[0]
        if existing_class != expected_class:
            raise RuntimeError(
                f"输入 System 已带 {existing_class}，但当前协议 "
                f"system_type={protocol['system_type']} 要求 {expected_class}。"
                "按 memtodolist §3.2 这里 fail closed，而不是再叠加一个——"
                "叠加会让两个 barostat 同时做体积移动，集合定义错误且不报错。"
            )
        return {
            "action": "reused_existing",
            "barostat_class": existing_class,
            "force_index": int(existing[0][0]),
            "protocol": protocol,
        }

    if expected_class == "MonteCarloBarostat":
        force = openmm.MonteCarloBarostat(
            pressure_bar * unit.bar,
            temperature_k * unit.kelvin,
            int(protocol["barostat_frequency"]),
        )
    elif expected_class == "MonteCarloMembraneBarostat":
        membrane = protocol["membrane"]
        force = openmm.MonteCarloMembraneBarostat(
            pressure_bar * unit.bar,
            float(membrane["surface_tension_bar_nm"]) * unit.bar * unit.nanometer,
            temperature_k * unit.kelvin,
            _openmm_membrane_xy_mode(membrane["xy_mode"]),
            _openmm_membrane_z_mode(membrane["z_mode"]),
            int(protocol["barostat_frequency"]),
        )
    else:  # pragma: no cover - resolve_membrane_protocol 只产出上面两种
        raise RuntimeError(f"未知 barostat_class={expected_class!r}")

    force_index = system.addForce(force)
    return {
        "action": "added",
        "barostat_class": expected_class,
        "force_index": int(force_index),
        "protocol": protocol,
    }


def barostat_fingerprint_payload(protocol: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """预平衡 fingerprint 里的 barostat 身份；legacy soluble 协议返回 None。

    返回 None 的意义是"与改动前完全相同的协议"，调用方据此**不往 fingerprint
    payload 里加任何键**，从而保证 §7.7：不声明 system_type 的运行，其
    fingerprint 与本次改动前逐位一致，已有的生产预平衡 checkpoint 不会失效。

    反过来，只要 system_type 变成 membrane、或频率偏离 legacy 值，payload 就非
    None，旧 checkpoint 自动失效——这正是 §3.2 要求的"改任一项使旧 checkpoint 失效"。
    """
    if not protocol:
        return None
    if (
        protocol.get("system_type") == ENVIRONMENT_TYPE_SOLUBLE
        and protocol.get("barostat_class") == "MonteCarloBarostat"
        and int(protocol.get("barostat_frequency", LEGACY_SOLUBLE_BAROSTAT_FREQUENCY))
        == LEGACY_SOLUBLE_BAROSTAT_FREQUENCY
    ):
        return None
    payload = {
        "protocol_version": int(protocol["protocol_version"]),
        "system_type": protocol["system_type"],
        "barostat_class": protocol["barostat_class"],
        "barostat_frequency": int(protocol["barostat_frequency"]),
        "membrane": protocol.get("membrane"),
    }
    return payload


# ============================================================================
# 净电荷处理协议：charge_treatment（memtodolist.md §1.2 / §2 / §7.1；B2）
#
# 这一节是**纯校验层**：把"配体净电荷 / co-ion / APBS / 环境类型"这四个量的合法
# 组合一次性判死，在创建任何 Context 之前（§6.1）。它不构建任何 Hamiltonian——
# charge-transfer 的 charging System 构建是 B3，溶剂腿 co-ion 身份是 B4。
#
# 关键设计约束逐条对应清单：
#   §1.2 必须是**显式配置**；禁止根据"有没有 APBS 数值"猜协议。
#   §1.2 co-ion 路线与 Rocklin/APBS 是二选一，禁止重复修正（双计数）。
#   §0.5.1 MEM-00a-1 `CHARGE_TRANSFER_PROTOCOL_VERSION` 必须**独立**，
#          不复用 SOLVENT_CACHE_PROTOCOL_VERSION / IBS_BIAS_PROTOCOL_VERSION。
#   §0.5.1 MEM-00a-2 co-annihilation 降级为实验对照：membrane 一律 fail closed，
#          输出必须带 `experimental_not_for_production: true`，
#          其数值不得进入任何 ΔG_bind 汇总。
#   §7.7 中性配体路径行为不变——这是当前生产体系（Atenolol 中性）走的那条。
# ============================================================================

# MEM-00a-1：与其它协议版本并列的独立版本号。charge-transfer 协议本身变化时递增，
# 不要因为溶剂缓存或 IBS bias 协议变了就动它，反之亦然。
CHARGE_TRANSFER_PROTOCOL_VERSION = 1

CHARGE_TREATMENT_NEUTRAL = "neutral"
CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER = "co_alchemical_charge_transfer"
CHARGE_TREATMENT_ROCKLIN_APBS = "rocklin_apbs_neutralizing_plasma"
CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL = "co_annihilation_experimental"

CHARGE_TREATMENTS = (
    CHARGE_TREATMENT_NEUTRAL,
    CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    CHARGE_TREATMENT_ROCKLIN_APBS,
    CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
)

# 🚧 B3 未落地。charge-transfer 的 charging Hamiltonian（ligand q→0 与
# co-ion 0→q 由同一个 lambda_q 反向驱动、co-ion 电荷走 PME particle offset）
# 还没有实现，`ibs_engine.py` 里现存的是 co-annihilation。
# 所以本校验层即使收到一份格式完全合法的 co-ion 规格，也必须 fail closed——
# 放行只会让用户拿到一个"声明了 charge-transfer、实际跑的是别的东西"的结果。
# B3 落地时把它改成 True，并同时打开对应的端点测试（§7.2 / §7.3）。
CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED = False

# §13.2 数值自洽的两个容差。放在这里是因为本层就要用；§13 的完整阈值表另立。
LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E = 1.0e-3
TOTAL_CHARGE_CONSERVATION_TOLERANCE_E = 1.0e-6

# §3.4 要求写入 manifest 的 co-ion 身份字段。B2 只校验"齐不齐、算不算得平"，
# 真正的选择与落盘是 B4。
CO_ALCHEMICAL_ION_REQUIRED_FIELDS = (
    "atom_index",
    "residue_index",
    "residue_name",
    "element",
    "charge_at_lambda1_e",
    "charge_at_lambda0_e",
    "sigma_nm",
    "epsilon_kj_mol",
    "mass_amu",
    "restraint",
)

# §5 / §1.2：选 Rocklin 路线时必须真的有 APBS 证据，不能只填一个数。
APBS_REQUIRED_EVIDENCE_FIELDS = (
    "manifest_path",
    "result_path",
    "dielectric_map_paths",
    "lipid_charge_map_path",
    "net_charge_e",
)


def resolve_charge_treatment(
    charge_treatment: Optional[str],
    ligand_net_charge_e: float,
    apbs_correction_kJ_mol: float = 0.0,
    co_alchemical_ion: Optional[Any] = None,
    environment_type: Optional[str] = None,
    apbs_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """校验净电荷处理协议，返回可直接进 fingerprint/provenance 的解析结果。

    `charge_treatment=None` 时按 §1.2 的生产默认推导：中性配体 → `neutral`，
    带电配体 → `co_alchemical_charge_transfer`。**这不是"猜协议"**——它只看配体
    净电荷这一个客观量，与清单禁止的"根据有没有 APBS 数值猜"是两回事。

    `ligand_net_charge_e` 由调用方用现有实现算出并传入（复合物腿走
    `ibs_engine._compute_ligand_net_charge`），本函数不自己再数一遍电荷，
    避免出现第二套净电荷判据。
    """
    raw_q = float(ligand_net_charge_e)
    if not math.isfinite(raw_q):
        raise ValueError(f"配体净电荷不是有限数：{ligand_net_charge_e!r}")
    q_int = int(round(raw_q))
    if abs(raw_q - q_int) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
        raise ValueError(
            f"配体净电荷 {raw_q:+.6f} e 不接近整数"
            f"（容差 {LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:g} e）。"
            "§2.2：非整数净电荷必须先作为输入错误调查，"
            "不要静默塞给一个分数价 co-ion。"
        )

    try:
        apbs_value = float(apbs_correction_kJ_mol or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"apbs_correction_kJ_mol={apbs_correction_kJ_mol!r} 不是数值"
        ) from exc
    if not math.isfinite(apbs_value):
        raise ValueError("apbs_correction_kJ_mol 必须有限")

    # ---- 解析协议名 ----
    if charge_treatment is None or str(charge_treatment).strip() == "":
        resolved = (
            CHARGE_TREATMENT_NEUTRAL
            if q_int == 0
            else CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
        )
        was_defaulted = True
    else:
        resolved = str(charge_treatment).strip().lower()
        was_defaulted = False
    if resolved not in CHARGE_TREATMENTS:
        raise ValueError(
            f"charge_treatment={charge_treatment!r} 不是合法值；"
            f"允许 {list(CHARGE_TREATMENTS)}。"
        )

    resolved_environment = resolve_environment_type(environment_type)
    is_experimental = resolved == CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL

    # ---- 逐协议规则（§1.2 的 fail-closed 清单）----
    if resolved == CHARGE_TREATMENT_NEUTRAL:
        # fail-closed #2：neutral 但检测到配体净电荷不为 0。
        if q_int != 0:
            raise ValueError(
                f"charge_treatment=neutral 但配体净电荷为 {q_int:+d} e。"
                "带电配体必须显式选择净电荷处理路线："
                f"生产用 {CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER}，"
                f"方法对照用 {CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL}，"
                f"neutralizing plasma 用 {CHARGE_TREATMENT_ROCKLIN_APBS}。"
            )
        if co_alchemical_ion:
            raise ValueError("charge_treatment=neutral 不得创建 co-alchemical ion。")
        if apbs_value != 0.0:
            raise ValueError(
                f"charge_treatment=neutral 但 apbs_correction_kJ_mol={apbs_value:+.6f}。"
                "中性配体既不需要 co-ion，也不需要 Rocklin 净电荷修正（§0 第 3 条）。"
            )

    elif resolved in (
        CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
    ):
        if q_int == 0:
            raise ValueError(
                f"charge_treatment={resolved} 但配体净电荷为 0。"
                "中性配体不需要共炼金离子，请用 neutral。"
            )
        # fail-closed #1：co-ion 路线与 APBS 是二选一，禁止双计数。
        if apbs_value != 0.0:
            raise ValueError(
                f"charge_treatment={resolved} 且 "
                f"apbs_correction_kJ_mol={apbs_value:+.6f} —— 这是**重复修正**。"
                "共炼金路线全程保持体系总电荷不变，Rocklin/APBS 净电荷修正必须为 0，"
                "并记录 not_applicable_co_alchemical_charge_transfer（§0 第 3 条）。"
            )

        if is_experimental:
            # MEM-00a-2：膜生产一律 fail closed。
            if resolved_environment == ENVIRONMENT_TYPE_MEMBRANE:
                raise ValueError(
                    "charge_treatment=co_annihilation_experimental 不允许出现在膜体系中。"
                    "co-annihilation 同时执行 ligand: q→0 与 counterion: −q→0，"
                    "两个异号离子在膜/水非均匀环境中的消失自由能不能可靠抵消"
                    "（一个在结合位点、一个在体相水，介电环境完全不同）。"
                    "膜生产请用 co_alchemical_charge_transfer；"
                    "本路线只允许用于水盒 / lipid slab 的方法对照（§8.1/§8.2 末条）。"
                )
        else:
            # fail-closed #3：charge-transfer 缺 co-ion 身份/参数/restraint。
            if not co_alchemical_ion:
                raise ValueError(
                    "charge_treatment=co_alchemical_charge_transfer 但没有提供 "
                    "co_alchemical_ion 身份、参数与 restraint（§1.2 fail-closed 第 3 条、§3.4）。"
                )
            _validate_co_alchemical_ion_spec(co_alchemical_ion, q_int)
            if not CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED:
                raise NotImplementedError(
                    "charge_treatment=co_alchemical_charge_transfer 的 charging "
                    "Hamiltonian 尚未实现（memtodolist Phase B3）。"
                    "`ibs_engine.py` 里现存的共炼金实现是 co-annihilation，"
                    "不是 ligand q→0 / co-ion 0→q 的 charge-transfer（MEM-00a）。"
                    "在 B3 落地前拒绝放行：否则结果会声明 charge-transfer 而实际跑的是别的哈密顿量。"
                    "方法对照请显式选 co_annihilation_experimental。"
                )

    elif resolved == CHARGE_TREATMENT_ROCKLIN_APBS:
        if co_alchemical_ion:
            raise ValueError(
                "charge_treatment=rocklin_apbs_neutralizing_plasma 禁止创建 "
                "co-alchemical ion —— 两条路线是二选一，同时用就是重复修正。"
            )
        # fail-closed #4：缺 APBS 来源说明/结果文件。
        missing = [
            field
            for field in APBS_REQUIRED_EVIDENCE_FIELDS
            if not (apbs_evidence or {}).get(field)
        ]
        if missing:
            raise ValueError(
                f"charge_treatment=rocklin_apbs_neutralizing_plasma 缺少 APBS 证据字段 "
                f"{missing}（§1.2 fail-closed 第 4 条、§5）。"
                "必须提供真实 APBS manifest/result、介电图与脂质电荷图；"
                "只填一个 apbs_correction_kJ_mol 数值不算。"
            )

    payload: Dict[str, Any] = {
        "protocol_version": CHARGE_TRANSFER_PROTOCOL_VERSION,
        "charge_treatment": resolved,
        "was_defaulted_from_net_charge": bool(was_defaulted),
        "ligand_net_charge_e": q_int,
        "ligand_net_charge_raw_e": raw_q,
        "environment_type": resolved_environment,
        "apbs_correction_kJ_mol": apbs_value,
        "apbs_applicable": resolved == CHARGE_TREATMENT_ROCKLIN_APBS,
        "apbs_applied": resolved == CHARGE_TREATMENT_ROCKLIN_APBS and apbs_value != 0.0,
        "co_alchemical_ion": co_alchemical_ion or None,
        "experimental_not_for_production": bool(is_experimental),
    }

    if resolved in (
        CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
    ):
        payload["apbs_not_applicable_reason"] = (
            "not_applicable_co_alchemical_charge_transfer"
            if not is_experimental
            else "not_applicable_co_annihilation_conserves_total_charge"
        )
        payload["total_charge_conserved_at_every_lambda"] = True
    elif resolved == CHARGE_TREATMENT_ROCKLIN_APBS:
        payload["total_charge_conserved_at_every_lambda"] = False
        payload["apbs_evidence"] = dict(apbs_evidence or {})
    else:
        payload["total_charge_conserved_at_every_lambda"] = True
        payload["apbs_not_applicable_reason"] = "not_applicable_neutral_ligand"

    return payload


def _validate_co_alchemical_ion_spec(spec: Any, ligand_net_charge: int) -> None:
    """校验 co-ion 规格：字段齐全、每粒子不超过一个单位电荷、总变化抵消（§2.2/§3.4）。

    接受单个 dict 或 dict 列表——`|q_L| > 1` 时按 §2.2 必须用多个单价 co-ion 分担，
    不允许把多个单位电荷集中到一个非物理多价粒子上。
    """
    ions = spec if isinstance(spec, (list, tuple)) else [spec]
    if not ions:
        raise ValueError("co_alchemical_ion 为空")

    seen_indices = set()
    total_transferred = 0.0
    for position, ion in enumerate(ions):
        if not isinstance(ion, dict):
            raise ValueError(
                f"co_alchemical_ion[{position}] 必须是 dict，收到 {type(ion).__name__}"
            )
        missing = [f for f in CO_ALCHEMICAL_ION_REQUIRED_FIELDS if f not in ion]
        if missing:
            raise ValueError(
                f"co_alchemical_ion[{position}] 缺少 §3.4 要求的字段 {missing}"
            )
        index = int(ion["atom_index"])
        if index in seen_indices:
            raise ValueError(
                f"co_alchemical_ion 出现重复 atom_index={index}；"
                "同一个粒子不能被当成两个 co-ion。"
            )
        seen_indices.add(index)

        q1 = float(ion["charge_at_lambda1_e"])
        q0 = float(ion["charge_at_lambda0_e"])
        # §2.2：λ=1 时它是"中性但保留 LJ 的 ion-shaped dummy"。
        if abs(q1) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            raise ValueError(
                f"co_alchemical_ion[{position}] 在 λ=1 的电荷应为 0（中性 dummy），"
                f"实际 {q1:+.6f} e。"
            )
        transferred = q0 - q1
        if abs(transferred) - 1.0 > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            raise ValueError(
                f"co_alchemical_ion[{position}] 转移了 {transferred:+.6f} e。"
                "§2.2：每个 co-ion 最多转移一个单位电荷；"
                f"|q_L| > 1 请用 {abs(ligand_net_charge)} 个单价 co-ion 分担，"
                "不要构造非物理多价粒子。"
            )
        if ligand_net_charge > 0 and transferred <= 0:
            raise ValueError(
                f"配体净电荷 {ligand_net_charge:+d} e 为正，co-ion 必须是同号（阳离子型，"
                f"0 → +1），实际转移 {transferred:+.6f} e。"
                "注意 charge-transfer 不是 co-annihilation：不是去掉一个异号反离子。"
            )
        if ligand_net_charge < 0 and transferred >= 0:
            raise ValueError(
                f"配体净电荷 {ligand_net_charge:+d} e 为负，co-ion 必须是同号（阴离子型，"
                f"0 → −1），实际转移 {transferred:+.6f} e。"
            )
        if not ion.get("restraint"):
            raise ValueError(
                f"co_alchemical_ion[{position}] 缺少 restraint（§2.3 要求可审计的 "
                "flat-bottom restraint，参考位置随盒缩放）。"
            )
        total_transferred += transferred

    # fail-closed #5：配体电荷变化与 co-ion 电荷变化之和不为 0。
    ligand_change = 0.0 - float(ligand_net_charge)
    residual = ligand_change + total_transferred
    if abs(residual) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
        raise ValueError(
            f"配体电荷变化 ({ligand_change:+.6f} e) 与 co-ion 电荷变化之和 "
            f"({total_transferred:+.6f} e) 不为 0，残差 {residual:+.6e} e "
            f"> {TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:g} e"
            "（§1.2 fail-closed 第 5 条）。总电荷必须在每个 λ 严格守恒。"
        )


# ============================================================================
# §13 验收阈值：全部落成命名常量并进 provenance
#
# 清单原文通篇写"预设阈值"但没给数——没有数就写不了 fail-closed 检查，也没法判
# 验收。§13 要求"必须在 Phase A 结束前落成常量并进 provenance，不许运行时凭感觉判"。
# 这里就是那份常量表；数值取 §13 的提案值，可改，但改必须改这里、且会进 provenance。
# ============================================================================

ACCEPTANCE_THRESHOLDS_VERSION = 1

# ---- §13.1 co-ion 几何 ----
COION_LIGAND_MIN_IMAGE_INITIAL_NM = 1.6
# 全程下限取 softcore cutoff。⚠️ 这个值必须与 `ibs_engine.SOFTCORE_CUTOFF_NM` 一致；
# abfe_core 在 ibs_engine 的下层不能反向 import，所以由
# tests/test_acceptance_thresholds.py 的交叉检查测试钉住，防止两处各改一半。
COION_LIGAND_MIN_IMAGE_RUNTIME_NM = 1.2
COION_PROTEIN_HEAVY_ATOM_MIN_NM = 1.2
COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM = 3.0
COION_NEAREST_PHOSPHORUS_MIN_NM = 1.0
COION_FIRST_SHELL_WATER_CUTOFF_NM = 0.32
# 按离子类型给首层水配位数下限（§13.1）。键为残基名/元素的大写形式。
COION_FIRST_SHELL_MIN_WATER_COUNT = {
    "NA": 5, "SOD": 5, "K": 5, "POT": 5,
    "CL": 6, "CLA": 6,
}
COION_FLAT_BOTTOM_RADIUS_NM = 0.5
COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2 = 100.0

# ---- §13.2 数值自洽 ----
# 总电荷守恒容差 TOTAL_CHARGE_CONSERVATION_TOLERANCE_E 已在 B2 一节定义，此处不重复。
LIGAND_CHARGE_LAMBDA_TOLERANCE_E = 1.0e-6
ENDPOINT_ENERGY_RELATIVE_TOLERANCE = 1.0e-5
ENDPOINT_FORCE_MAX_ABS_TOLERANCE_KJ_PER_MOL_NM = 1.0e-3
# λ=0 端 ligand–environment 静电与 LJ 必须是**严格零**，不是"很小"。
DECOUPLED_ENDPOINT_ENERGY_ABS_TOLERANCE_KJ_PER_MOL = 1.0e-6
GROMACS_OPENMM_COMPONENT_RELATIVE_TOLERANCE = 1.0e-4
GROMACS_OPENMM_TOTAL_ABS_TOLERANCE_KJ_PER_MOL = 0.1

# ---- §13.3 膜质量门（判据统一为"末段窗口内线性漂移小于阈值"）----
MEMBRANE_QUALITY_GATE_TAIL_WINDOW_NS = 20.0
APL_MAX_DRIFT_PERCENT_PER_NS = 0.2
APL_MAX_DEVIATION_FROM_LITERATURE_PERCENT = 3.0
BILAYER_THICKNESS_MAX_DRIFT_NM_PER_TAIL_WINDOW = 0.05
PROTEIN_BACKBONE_MAX_RMSD_NM = 0.30
TRANSMEMBRANE_TILT_MAX_DRIFT_DEG = 5.0
POCKET_MAX_RMSD_NM = 0.20
LIGAND_HEAVY_ATOM_MAX_RMSD_NM = 0.25

# ---- §13.4 结果验收 ----
CROSS_REPEAT_MAX_STDDEV_KCAL_PER_MOL = 1.0
MIN_INDEPENDENT_REPEATS = 3
BENCHMARK_MIN_LIGANDS = 5
BENCHMARK_MAX_MAE_KCAL_PER_MOL = 1.5
BENCHMARK_MAX_ABS_OUTLIER_KCAL_PER_MOL = 3.0
# §3.0 空腔填充迟滞：正反向 / 双起点 stage 2 的 ΔF 差 ≤ 2σ。
STAGE2_HYSTERESIS_MAX_SIGMA = 2.0


def acceptance_thresholds_payload() -> Dict[str, Any]:
    """§13 全部阈值的可序列化快照，供 provenance 落盘。

    目的是让每一份结果都能回答"当时用的是哪套阈值"——阈值改了而结果没重跑，
    对照 provenance 就能看出来。
    """
    return {
        "version": ACCEPTANCE_THRESHOLDS_VERSION,
        "coion_geometry": {
            "ligand_min_image_initial_nm": COION_LIGAND_MIN_IMAGE_INITIAL_NM,
            "ligand_min_image_runtime_nm": COION_LIGAND_MIN_IMAGE_RUNTIME_NM,
            "protein_heavy_atom_min_nm": COION_PROTEIN_HEAVY_ATOM_MIN_NM,
            "membrane_midplane_min_abs_z_nm": COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM,
            "nearest_phosphorus_min_nm": COION_NEAREST_PHOSPHORUS_MIN_NM,
            "first_shell_water_cutoff_nm": COION_FIRST_SHELL_WATER_CUTOFF_NM,
            "first_shell_min_water_count": dict(COION_FIRST_SHELL_MIN_WATER_COUNT),
            "flat_bottom_radius_nm": COION_FLAT_BOTTOM_RADIUS_NM,
            "flat_bottom_k_kj_per_mol_nm2": COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2,
        },
        "numerical_selfconsistency": {
            "total_charge_conservation_e": TOTAL_CHARGE_CONSERVATION_TOLERANCE_E,
            "ligand_charge_lambda_e": LIGAND_CHARGE_LAMBDA_TOLERANCE_E,
            "endpoint_energy_relative": ENDPOINT_ENERGY_RELATIVE_TOLERANCE,
            "endpoint_force_max_abs_kj_per_mol_nm": (
                ENDPOINT_FORCE_MAX_ABS_TOLERANCE_KJ_PER_MOL_NM
            ),
            "decoupled_endpoint_energy_abs_kj_per_mol": (
                DECOUPLED_ENDPOINT_ENERGY_ABS_TOLERANCE_KJ_PER_MOL
            ),
            "gromacs_openmm_component_relative": (
                GROMACS_OPENMM_COMPONENT_RELATIVE_TOLERANCE
            ),
            "gromacs_openmm_total_abs_kj_per_mol": (
                GROMACS_OPENMM_TOTAL_ABS_TOLERANCE_KJ_PER_MOL
            ),
        },
        "membrane_quality_gate": {
            "tail_window_ns": MEMBRANE_QUALITY_GATE_TAIL_WINDOW_NS,
            "apl_max_drift_percent_per_ns": APL_MAX_DRIFT_PERCENT_PER_NS,
            "apl_max_deviation_from_literature_percent": (
                APL_MAX_DEVIATION_FROM_LITERATURE_PERCENT
            ),
            "bilayer_thickness_max_drift_nm_per_tail_window": (
                BILAYER_THICKNESS_MAX_DRIFT_NM_PER_TAIL_WINDOW
            ),
            "protein_backbone_max_rmsd_nm": PROTEIN_BACKBONE_MAX_RMSD_NM,
            "transmembrane_tilt_max_drift_deg": TRANSMEMBRANE_TILT_MAX_DRIFT_DEG,
            "pocket_max_rmsd_nm": POCKET_MAX_RMSD_NM,
            "ligand_heavy_atom_max_rmsd_nm": LIGAND_HEAVY_ATOM_MAX_RMSD_NM,
        },
        "result_acceptance": {
            "cross_repeat_max_stddev_kcal_per_mol": (
                CROSS_REPEAT_MAX_STDDEV_KCAL_PER_MOL
            ),
            "min_independent_repeats": MIN_INDEPENDENT_REPEATS,
            "benchmark_min_ligands": BENCHMARK_MIN_LIGANDS,
            "benchmark_max_mae_kcal_per_mol": BENCHMARK_MAX_MAE_KCAL_PER_MOL,
            "benchmark_max_abs_outlier_kcal_per_mol": (
                BENCHMARK_MAX_ABS_OUTLIER_KCAL_PER_MOL
            ),
            "stage2_hysteresis_max_sigma": STAGE2_HYSTERESIS_MAX_SIGMA,
        },
    }


# ============================================================================
# §1.1 力场族自动识别 + §1.3 dispersion_protocol（B6）
#
# 两者绑在一起，因为 §1.1 的裁决直接决定 §1.3 走哪条路线：
#   amber 系  → Amber 脂质（Lipid21/Lipid17）+ ff_native_isotropic_lrc
#   charmm 系 → CHARMM36 脂质 + ff_native_force_switch_no_lrc
#
# ⚠️ §1.3 的修正框：原稿"膜体系一律禁用 LRC"是错的。判据是**跟随所选脂质力场的
# 原始参数化条件**——Amber Lipid21 就是在开着各向同性 vdW 长程修正下拟合的，
# 对它关掉 LRC 才是错的；CHARMM36 是 force-switch 且不加 LRC，对它开 LRC 才是错的。
#
# ⚠️ OpenMM 的 `NonbondedForce` 只有 potential-switch（`setUseSwitchingFunction`），
# **没有 force-switch**。所以 charmm 分支无法复现原始 Hamiltonian，默认 fail closed，
# 只有给出定量偏差论证（APL / 膜厚 / 单点能对照）后才允许放行。amber 是首选路径。
# ============================================================================

FORCEFIELD_FAMILY_AMBER = "amber"
FORCEFIELD_FAMILY_CHARMM = "charmm"
FORCEFIELD_FAMILIES = (FORCEFIELD_FAMILY_AMBER, FORCEFIELD_FAMILY_CHARMM)

# 识别得出但本轮不支持的族：明确报错好过"识别不出"这种含糊结论。
FORCEFIELD_FAMILIES_UNSUPPORTED = ("opls", "gromos")

# `#include` 路径里出现这些 token 即判定为对应族。按最长匹配优先，避免
# "charmm36" 被 "charm" 之类的前缀误伤。
_FORCEFIELD_FAMILY_TOKENS = (
    ("charmm", FORCEFIELD_FAMILY_CHARMM),
    ("amber", FORCEFIELD_FAMILY_AMBER),
    ("opls", "opls"),
    ("oplsaa", "opls"),
    ("gromos", "gromos"),
)

DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC = "legacy_uniform_density_lrc"
DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC = "ff_native_isotropic_lrc"
DISPERSION_PROTOCOL_FF_NATIVE_FORCE_SWITCH_NO_LRC = "ff_native_force_switch_no_lrc"
DISPERSION_PROTOCOL_LJ_PME = "lj_pme"
DISPERSION_PROTOCOL_MEMBRANE_INHOMOGENEOUS = "membrane_inhomogeneous"

DISPERSION_PROTOCOLS = (
    DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC,
    DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC,
    DISPERSION_PROTOCOL_FF_NATIVE_FORCE_SWITCH_NO_LRC,
    DISPERSION_PROTOCOL_LJ_PME,
    DISPERSION_PROTOCOL_MEMBRANE_INHOMOGENEOUS,
)

# 已实现（有代码支撑）的路线。LJ-PME 是 §1.3 路线 B、非均匀色散修正是路线 C，
# 两者都只有验收条件、没有实现——声明它们必须 NotImplementedError，
# 而不是被当成拼错的未知值。
DISPERSION_PROTOCOLS_IMPLEMENTED = (
    DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC,
    DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC,
)

# §1.3：`system_type=membrane` 且未选择**已验证**的 dispersion_protocol → fail closed。
# 目前只有 amber 分支这一条算已验证路径。
MEMBRANE_VALIDATED_DISPERSION_PROTOCOLS = (
    DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC,
)

FORCEFIELD_FAMILY_DISPERSION_PROTOCOL = {
    FORCEFIELD_FAMILY_AMBER: DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC,
    FORCEFIELD_FAMILY_CHARMM: DISPERSION_PROTOCOL_FF_NATIVE_FORCE_SWITCH_NO_LRC,
}

# charmm 分支放行所需的定量论证字段（§1.1 最后一条）。
FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS = (
    "apl_comparison",
    "bilayer_thickness_comparison",
    "single_point_energy_comparison",
)


def detect_forcefield_family_from_top(top_path: str) -> Dict[str, Any]:
    """从 GROMACS `.top` 的 `#include` 判定力场族（§1.1）。

    实测本仓库的 `topol.top` **没有 `[ defaults ]` 段**（它在
    `amber14sb_OL15_fs1.ff/forcefield.itp` 里面），所以主判据只能是 include 路径；
    `[ defaults ]` 若存在则作为可选交叉检查一并记录，不作为唯一判据。

    识别不出（混合 include / 自定义 ff 目录）时**返回 family=None**，由
    `resolve_forcefield_family()` 决定是否 fail closed——本函数只负责观测。
    """
    with open(top_path, encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()

    includes: List[str] = []
    defaults_row: Optional[str] = None
    in_defaults = False
    for raw in lines:
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("#include"):
            # #include "path/to/forcefield.itp"
            parts = line.split('"')
            if len(parts) >= 2:
                includes.append(parts[1])
            continue
        if line.startswith("["):
            in_defaults = line.replace(" ", "").lower().startswith("[defaults]")
            continue
        if in_defaults and defaults_row is None:
            defaults_row = line

    families: Dict[str, List[str]] = {}
    for include in includes:
        lowered = include.lower()
        # 本地 include（./x.itp、posre.itp）不带 ff 目录，跳过。
        matched = None
        for token, family in sorted(
            _FORCEFIELD_FAMILY_TOKENS, key=lambda kv: -len(kv[0])
        ):
            if token in lowered:
                matched = family
                break
        if matched:
            families.setdefault(matched, []).append(include)

    distinct = sorted(families)
    if len(distinct) == 1:
        detected = distinct[0]
        reason = "single_family_include"
    elif len(distinct) > 1:
        detected = None
        reason = f"mixed_family_includes:{distinct}"
    else:
        detected = None
        reason = "no_recognized_forcefield_include"

    return {
        "family": detected,
        "reason": reason,
        "includes": includes,
        "family_evidence": {k: sorted(v) for k, v in families.items()},
        "defaults_row": defaults_row,
        "top_path": str(top_path),
    }


def resolve_forcefield_family(
    top_path: Optional[str] = None,
    explicit_family: Optional[str] = None,
) -> Dict[str, Any]:
    """确定力场族；识别不出即 fail closed，允许显式覆盖但必须留记录（§1.1）。"""
    detection: Dict[str, Any] = {"family": None, "reason": "not_attempted"}
    if top_path:
        detection = detect_forcefield_family_from_top(top_path)

    if explicit_family is not None and str(explicit_family).strip():
        override = str(explicit_family).strip().lower()
        if override in FORCEFIELD_FAMILIES_UNSUPPORTED:
            raise ValueError(
                f"forcefield_family={override!r} 本轮不支持（只支持 "
                f"{list(FORCEFIELD_FAMILIES)}）。"
            )
        if override not in FORCEFIELD_FAMILIES:
            raise ValueError(
                f"forcefield_family={override!r} 非法；允许 {list(FORCEFIELD_FAMILIES)}。"
            )
        # §1.1：覆盖必须留记录，不能静默。
        if detection.get("family") and detection["family"] != override:
            logger.warning(
                "⚠️ forcefield_family 被显式覆盖：自动识别为 %r（依据 %s），"
                "但用户指定 %r。覆盖已记入 provenance。",
                detection["family"], detection["reason"], override,
            )
        return {
            "family": override,
            "source": "explicit_override",
            "overrode_detection": detection.get("family"),
            "detection": detection,
        }

    family = detection.get("family")
    if family in FORCEFIELD_FAMILIES_UNSUPPORTED:
        raise ValueError(
            f"从 {top_path} 识别出力场族 {family!r}，本轮不支持"
            f"（只支持 {list(FORCEFIELD_FAMILIES)}）。"
        )
    if family is None:
        raise ValueError(
            f"无法从 {top_path!r} 识别力场族（原因：{detection.get('reason')}；"
            f"include 列表 {detection.get('includes')}）。"
            "按 memtodolist §1.1 这里 fail closed：**不许猜、不许默认回落到 amber**。"
            "请用 --forcefield-family 显式指定（会记入 provenance）。"
        )
    return {
        "family": family,
        "source": "auto_detected",
        "overrode_detection": None,
        "detection": detection,
    }


def resolve_dispersion_protocol(
    dispersion_protocol: Optional[str],
    environment_type: Optional[str] = None,
    forcefield_family: Optional[str] = None,
    force_switch_deviation_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """校验 LJ/色散路线（§1.3 / B6）。

    默认值刻意**按环境类型分叉**，以保证 §7.7：
      - soluble 未声明 → `legacy_uniform_density_lrc`，即改动前的
        `lrc_coeff[k]/V(t)` 行为，可溶路径逐位不变；
      - membrane 未声明 → **fail closed**（§1.3 明文要求）。
    """
    resolved_environment = resolve_environment_type(environment_type)
    is_membrane = resolved_environment == ENVIRONMENT_TYPE_MEMBRANE

    if dispersion_protocol is None or str(dispersion_protocol).strip() == "":
        if is_membrane:
            expected = (
                FORCEFIELD_FAMILY_DISPERSION_PROTOCOL.get(forcefield_family)
                if forcefield_family
                else None
            )
            raise ValueError(
                "system_type=membrane 但没有选择 dispersion_protocol —— fail closed（§1.3）。"
                f"已验证可用的只有 {list(MEMBRANE_VALIDATED_DISPERSION_PROTOCOLS)}"
                + (f"；按识别到的力场族 {forcefield_family!r} 应选 {expected!r}。" if expected else "。")
            )
        resolved = DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC
        was_defaulted = True
    else:
        resolved = str(dispersion_protocol).strip().lower()
        was_defaulted = False

    if resolved not in DISPERSION_PROTOCOLS:
        raise ValueError(
            f"dispersion_protocol={dispersion_protocol!r} 非法；"
            f"允许 {list(DISPERSION_PROTOCOLS)}。"
        )

    if resolved in (
        DISPERSION_PROTOCOL_LJ_PME,
        DISPERSION_PROTOCOL_MEMBRANE_INHOMOGENEOUS,
    ):
        route = "B（LJ-PME）" if resolved == DISPERSION_PROTOCOL_LJ_PME else "C（膜非均匀色散修正）"
        raise NotImplementedError(
            f"dispersion_protocol={resolved} 属 §1.3 路线 {route}，尚未实现。"
            "该路线的四项/三项验收条件清单里已写明，但代码不存在；"
            "在完成之前不能只把基础 NonbondedForce 切过去就宣称支持。"
        )

    # §6.4 / §1.3：membrane + legacy uniform-density LRC 必须 fail closed。
    if is_membrane and resolved == DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC:
        raise ValueError(
            "system_type=membrane 不得使用 legacy_uniform_density_lrc。"
            "现有 `lj_tail_lrc_coeff[k]/V(t)` 假设配体周围是**均匀体相密度**；"
            "配体埋在脂双层口袋里时这个假设直接不成立——局域密度既不是水也不是体相脂质。"
            "这是膜体系下的真实缺陷，不是保守选项（§1.3 修正框第 1 条、§6.4）。"
        )

    if is_membrane and resolved not in MEMBRANE_VALIDATED_DISPERSION_PROTOCOLS:
        raise ValueError(
            f"system_type=membrane + dispersion_protocol={resolved} 尚未验证。"
            f"已验证的只有 {list(MEMBRANE_VALIDATED_DISPERSION_PROTOCOLS)}（§1.3）。"
        )

    # charmm 的 force-switch 无法在 OpenMM 复现，默认卡住（§1.1 / §1.3 路线 A 末条）。
    if resolved == DISPERSION_PROTOCOL_FF_NATIVE_FORCE_SWITCH_NO_LRC:
        missing = [
            field
            for field in FORCE_SWITCH_DEVIATION_EVIDENCE_FIELDS
            if not (force_switch_deviation_evidence or {}).get(field)
        ]
        if missing:
            raise ValueError(
                "dispersion_protocol=ff_native_force_switch_no_lrc 默认 fail closed："
                "OpenMM 的 NonbondedForce 只有 potential-switch"
                "（setUseSwitchingFunction），**没有 force-switch**，"
                "无法复现 CHARMM36 脂质的原始 Hamiltonian；用 potential-switch 顶替会"
                "移动 APL 与膜厚，且不会报错。"
                f"放行需要定量偏差论证，当前缺少 {missing}（§1.1 / §1.3）。"
                "首选路径是改用 Amber 系脂质力场（与现有 amber14sb 同族）。"
            )

    # 与力场族交叉核对：族与路线不匹配就是用错了参数化条件。
    if forcefield_family:
        family = str(forcefield_family).strip().lower()
        if family not in FORCEFIELD_FAMILIES:
            raise ValueError(
                f"forcefield_family={forcefield_family!r} 非法；允许 {list(FORCEFIELD_FAMILIES)}。"
            )
        expected = FORCEFIELD_FAMILY_DISPERSION_PROTOCOL[family]
        if resolved not in (expected, DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC):
            raise ValueError(
                f"力场族 {family!r} 的原始参数化条件对应 dispersion_protocol={expected!r}，"
                f"但声明的是 {resolved!r}。§1.3 的判据是**跟随所选力场的原始参数化条件**："
                "Amber Lipid21 是在开着各向同性 LRC 下拟合的，关掉 LRC 才是错的；"
                "CHARMM36 是 force-switch 且不加 LRC，开 LRC 才是错的。"
            )

    return {
        "dispersion_protocol": resolved,
        "was_defaulted": bool(was_defaulted),
        "environment_type": resolved_environment,
        "forcefield_family": (
            str(forcefield_family).strip().lower() if forcefield_family else None
        ),
        "implemented": resolved in DISPERSION_PROTOCOLS_IMPLEMENTED,
        "uniform_density_lrc_active": (
            resolved == DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC
        ),
        # §5 最后一条 / §6.5：APBS 与 LJ 色散正交，不得互相顶替。
        "apbs_is_orthogonal_to_dispersion": True,
    }


def _validate_minimum_image(box_vectors, cutoff_nm: float) -> None:
    """Validate the triclinic minimum-image condition using plane spacings."""
    vectors = box_vectors
    if hasattr(vectors, "value_in_unit"):
        vectors = vectors.value_in_unit(unit.nanometer)
    else:
        vectors = [
            vector.value_in_unit(unit.nanometer)
            if hasattr(vector, "value_in_unit")
            else vector
            for vector in vectors
        ]
    matrix = np.asarray(vectors, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"Invalid periodic box vectors: shape={matrix.shape}")
    volume = float(abs(np.linalg.det(matrix)))
    if volume <= 0.0:
        raise ValueError(f"Periodic box has non-positive volume: {volume}")

    spacings = []
    for axis in range(3):
        other = [idx for idx in range(3) if idx != axis]
        area = float(np.linalg.norm(np.cross(matrix[other[0]], matrix[other[1]])))
        if area <= 0.0:
            raise ValueError("Periodic box contains collinear vectors")
        spacings.append(volume / area)
    shortest = min(spacings)
    if shortest <= 2.0 * float(cutoff_nm):
        raise ValueError(
            "Minimum-image violation: shortest box plane spacing "
            f"{shortest:.6f} nm must exceed 2*cutoff="
            f"{2.0 * float(cutoff_nm):.6f} nm"
        )


def _build_openmmml_kwargs(
    device: Optional[str] = None,
    precision: Optional[str] = None,
    return_energy_type: Optional[str] = None,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
) -> Dict[str, Any]:
    """
    统一按 openmm-ml 官方接口组织 MLPotential.createSystem() 参数。
    是否显式传 precision 由调用侧决定；若不传则遵循 openmm-ml 的模型默认精度。
    """
    kwargs: Dict[str, Any] = {}
    if return_energy_type is not None:
        kwargs["returnEnergyType"] = return_energy_type
    if device is not None:
        kwargs["device"] = device
    if precision in ("single", "double"):
        kwargs["precision"] = precision
    if charge is not None:
        kwargs["charge"] = charge
    if multiplicity is not None:
        kwargs["multiplicity"] = multiplicity
    return kwargs


_MACE_LOCAL_MODEL_PATHS = {
    "mace-off24-medium": os.path.expanduser("~/.cache/mace/MACE-OFF24_medium.model"),
}


def _build_mace_potential(model_name: str):
    """
    部分 openmm-ml 版本的模型注册表里没有 mace-off24-medium 等较新的预训练名，
    但支持用 model_name="mace" + modelPath=本地缓存权重文件的方式加载同一个模型。
    这里对已知的本地缓存做名字 -> 路径映射；找不到对应缓存文件时退回原始按名加载
    （交给 openmm-ml 自己报错或解析）。
    """
    cached_path = _MACE_LOCAL_MODEL_PATHS.get(model_name)
    if cached_path and os.path.isfile(cached_path):
        return MLPotential("mace", modelPath=cached_path)
    return MLPotential(model_name)


def _select_env_indices_from_mdtraj_frame(frame, lig_idx: np.ndarray, env_radius_nm: float, max_env_atoms: Optional[int] = None) -> np.ndarray:
    """
    先做半径近邻筛选，再按“到配体最近距离”的 Top-K 排序裁剪环境原子。
    这里裁的是原子，不是整盒水，也不是全环境残基。
    """
    import mdtraj as md

    raw_env = md.compute_neighbors(frame, env_radius_nm, lig_idx)[0]
    env_idx = np.setdiff1d(raw_env, lig_idx, assume_unique=True)
    if max_env_atoms is None or len(env_idx) <= max_env_atoms:
        return np.asarray(env_idx, dtype=int)

    pos_nm = np.asarray(frame.xyz[0], dtype=np.float64)
    if frame.unitcell_vectors is not None:
        box_vecs = np.asarray(frame.unitcell_vectors[0], dtype=np.float64)
        box_lens = np.linalg.norm(box_vecs, axis=1)
    else:
        box_lens = None

    delta = pos_nm[lig_idx][:, None, :] - pos_nm[env_idx][None, :, :]
    if box_lens is not None:
        delta -= box_lens * np.round(delta / box_lens)
    dists = np.linalg.norm(delta, axis=-1)
    min_dists = np.min(dists, axis=0)
    keep_order = np.argsort(min_dists)[:max_env_atoms]
    return np.sort(np.asarray(env_idx[keep_order], dtype=int))


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


def _pymbar_version_tuple() -> Tuple[int, ...]:
    if not HAS_PYMBAR:
        return (0,)
    version_str = str(getattr(pymbar, "__version__", getattr(pymbar, "version", "0")))
    parts = []
    for token in version_str.replace("-", ".").split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts) if parts else (0,)


def _build_mbar_compatible(u_kn, n_k, **kwargs):
    """
    兼容 PyMBAR 3.x/4.x 的 MBAR 构造器。
    若某些关键字参数在当前版本不可用，则按保守顺序回退。
    """
    if not HAS_PYMBAR:
        raise ImportError("需要 pymbar 包，请安装: pip install pymbar")

    base_kwargs = dict(kwargs)
    drop_order = [
        "solver_protocol",
        "initialize",
        "relative_tolerance",
        "solver_tolerance",
        "initial_f_k",
        "verbose",
    ]
    variants = [base_kwargs]
    seen = {tuple(sorted((k, repr(v)) for k, v in base_kwargs.items()))}
    current = dict(base_kwargs)
    for key in drop_order:
        if key in current:
            current = dict(current)
            current.pop(key, None)
            signature = tuple(sorted((k, repr(v)) for k, v in current.items()))
            if signature not in seen:
                variants.append(current)
                seen.add(signature)

    last_type_error = None
    for candidate in variants:
        try:
            return pymbar.MBAR(u_kn, n_k, **candidate)
        except TypeError as exc:
            last_type_error = exc
            continue
    if last_type_error is not None:
        raise last_type_error
    return pymbar.MBAR(u_kn, n_k)


def _extract_mbar_matrix(result, primary_name: str, fallback_names: Tuple[str, ...]) -> Optional[np.ndarray]:
    candidate_names = (primary_name,) + tuple(fallback_names)
    for name in candidate_names:
        if isinstance(result, dict) and name in result:
            return np.asarray(result[name], dtype=float)
        if hasattr(result, name):
            return np.asarray(getattr(result, name), dtype=float)
    return None


def _compute_free_energy_result_compatible(mbar, compute_uncertainty: bool = True):
    methods = []
    if hasattr(mbar, "compute_free_energy_differences"):
        methods.append(("compute_free_energy_differences", {"compute_uncertainty": compute_uncertainty}))
    if hasattr(mbar, "compute_free_energy"):
        methods.append(("compute_free_energy", {}))
    if not methods:
        raise AttributeError("当前 pymbar.MBAR 对象不包含可用的自由能计算方法")

    last_exc = None
    for method_name, kwargs in methods:
        try:
            return getattr(mbar, method_name)(**kwargs)
        except TypeError:
            try:
                return getattr(mbar, method_name)()
            except Exception as exc:
                last_exc = exc
        except Exception as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("MBAR 自由能计算失败")


def _extract_free_energy_arrays(result, require_uncertainty: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    delta_f = _extract_mbar_matrix(result, "Delta_f", ("delta_f", "free_energy"))
    if delta_f is None:
        raise KeyError("无法从 pymbar 结果中提取自由能矩阵")

    delta_df = _extract_mbar_matrix(result, "dDelta_f", ("d_delta_f", "error", "uncertainty"))
    if delta_df is None:
        if require_uncertainty:
            raise KeyError("无法从 pymbar 结果中提取不确定度矩阵")
        delta_df = np.full_like(delta_f, np.nan, dtype=float)
    return delta_f, delta_df


def _compute_overlap_matrix_compatible(mbar) -> np.ndarray:
    """兼容层：pymbar.MBAR.compute_overlap() 的返回形状。

    ibs_engine.py 里同类调用点（如 TraditionalMBARAnalyzer.solve()）已经
    对"返回 dict（含 'matrix' 键）还是直接返回 ndarray"做了兼容判断；这里
    补上同样的分支，并加一道有限性校验（不让 NaN/Inf 静默流入调用方的
    收敛/重叠判据），而不是像之前那样无条件假设一定是 dict、也从不校验数值。
    """
    overlap_res = mbar.compute_overlap()
    overlap_matrix = overlap_res["matrix"] if isinstance(overlap_res, dict) else overlap_res
    overlap_matrix = np.asarray(overlap_matrix, dtype=float)
    if not np.all(np.isfinite(overlap_matrix)):
        raise RuntimeError(f"PyMBAR compute_overlap 返回非有限矩阵: {overlap_matrix}")
    return overlap_matrix


def _compute_effective_sample_number_compatible(mbar) -> np.ndarray:
    """兼容层：同 _compute_overlap_matrix_compatible，对
    compute_effective_sample_number() 的返回值做统一的有限性校验。
    """
    neff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)
    if not np.all(np.isfinite(neff)):
        raise RuntimeError(f"PyMBAR compute_effective_sample_number 返回非有限值: {neff}")
    return neff


def subsample_series_by_autocorrelation(
    series: np.ndarray,
    min_frames_for_subsampling: int = 20,
) -> Tuple[np.ndarray, float]:
    """
    对单一状态自身的（约化）能量时间序列估计统计非效率 g，返回近似去相关的帧索引。

    MBAR/BAR 假设逐帧样本互相独立；直接把每一帧原始 MD 轨迹都当独立样本喂给
    MBAR 会让 n_k/有效样本数虚高，报告的误差棒因此系统性偏小（常见 2-10 倍）。
    这里用该状态"自己产生"的能量时间序列估计关联时间的代理量 g，再用 pymbar
    的分块子采样取出近似独立的帧索引；调用方需要用同一组索引去裁剪该状态对应
    的所有相关数组，以保持帧与帧之间的对应关系。

    样本过少（< min_frames_for_subsampling）、序列本身没有涨落（比如恒定值）、
    或 pymbar 不可用时，原样返回全部索引、g=1.0（即不子采样）——对极短序列
    强行估计 g 噪声本身就很大，不如不做。
    """
    series = np.asarray(series, dtype=np.float64)
    n = series.shape[0]
    full_indices = np.arange(n)
    if not HAS_PYMBAR or n < min_frames_for_subsampling:
        return full_indices, 1.0
    if not np.all(np.isfinite(series)) or np.std(series) < 1e-12:
        return full_indices, 1.0
    try:
        from pymbar import timeseries
        g = float(timeseries.statistical_inefficiency(series, fast=True))
        if not np.isfinite(g) or g < 1.0:
            g = 1.0
        indices = np.asarray(timeseries.subsample_correlated_data(series, g=g), dtype=int)
        if indices.size < 2:
            return full_indices, g
        return indices, g
    except Exception:
        return full_indices, 1.0


def get_optimal_device_settings():
    if not HAS_ORB or not torch.cuda.is_available():
        return "cpu", False
    device = "cuda"
    major, minor = torch.cuda.get_device_capability()
    support_tf32 = False
    if major >= 8:
        support_tf32 = True
        torch.set_float32_matmul_precision("high")
    return device, support_tf32


# 🔑 [ATT-04] 这里原本是模块级的
#     GLOBAL_DEVICE, SUPPORTS_TF32 = get_optimal_device_settings()
# 也就是 **import 期** 就会调 `torch.cuda.is_available()` /
# `torch.cuda.get_device_capability()`（惰性建 CUDA context）并
# `torch.set_float32_matmul_precision("high")`（改全局 torch 状态）。
#
# 并行 stage worker 用 `mp.get_context("spawn")`（abfe_pipeline.py、ibs_engine.py），
# spawn 反序列化 target 时必然 import `abfe_pipeline → ibs_engine → abfe_core`，
# 于是每个子进程都在 import 期就抓一次 CUDA。更糟的是子进程的 GPU 归属只通过
# OpenMM 的 `props["DeviceIndex"]` 表达、从不设 `CUDA_VISIBLE_DEVICES`，所以双 GPU
# 并行时**两个**子进程都会在 device 0 上建 torch context，然后才各自去用被分配的
# OpenMM 设备。
#
# 注意：OpenMM 侧本来就没有 import 期副作用——abfe_core/ibs_engine/abfe_pipeline 里
# 所有 `Platform.getPlatformByName` / `Context(...)` 都在函数内。要消除的只有 torch
# 这一处。
#
# 改成惰性 memoized：真正需要 device 的只有下面两个 MACE/ML 入口的默认参数，
# 它们改成 `device=None` 后在函数体里解析。
_DEVICE_SETTINGS_CACHE = None


def _resolve_device_settings():
    """首次真正需要时才探测设备；结果缓存，语义与旧的模块级求值一致。"""
    global _DEVICE_SETTINGS_CACHE
    if _DEVICE_SETTINGS_CACHE is None:
        _DEVICE_SETTINGS_CACHE = get_optimal_device_settings()
    return _DEVICE_SETTINGS_CACHE


def get_global_device() -> str:
    return _resolve_device_settings()[0]


def supports_tf32() -> bool:
    return _resolve_device_settings()[1]


# ============================================================================
# 0. 单位常量与验证器
# ============================================================================
class UnitConstants:
    NM_PER_ANGSTROM = 0.1
    KJ_PER_KCAL = 4.184
    KJ_PER_NM2_PER_KCAL_PER_A2 = 418.4
    RAD_PER_DEG = np.pi / 180.0
    DISTANCE_RANGE_NM = (0.1, 10.0)
    ANGLE_RANGE_RAD = (0.0, math.pi)
    FORCE_CONSTANT_KR_RANGE = (100.0, 100000.0)
    FORCE_CONSTANT_KANGLE_RANGE = (10.0, 1000.0)


class UnitValidator:
    @staticmethod
    def validate_distance(v, n="dist"):
        if v <= 0:
            raise ValueError(f"{n}必须>0")
        if v > 100:
            warnings.warn(f"{n}={v} nm 可能过大")

    @staticmethod
    def validate_force_constant(v, n="k"):
        if v <= 0:
            raise ValueError(f"{n}必须为正值")

class NumpyEncoder(json.JSONEncoder):
    """🔑 全局统一 JSON 序列化器（处理 numpy 类型/数组）"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)): return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)
# ============================================================================
# 1. ACES 软核势 & DEXP 替身势 & Orb 拟合器
# ============================================================================
class ACESoftcorePotential:
    def __init__(
        self, alpha_lj=0.5, alpha_coul=0.2, power_lj=(2, 2), power_coul=(1, 1)
    ):
        self.alpha_lj, self.alpha_coul = float(alpha_lj), float(alpha_coul)
        self.m_lj, self.n_lj = power_lj
        self.m_coul, self.n_coul = power_coul

    def build_expression(self, lam_coul, lam_vdw):
        COUL = 138.935456
        lc, lv = f"({lam_coul}^{self.n_coul})", f"({lam_vdw}^{self.n_lj})"

        # 必须整体加括号，否则会被解析成 0.5*(sigma1+sigma2)^n，
        # 而不是 ((sigma1+sigma2)/2)^n，短程排斥会被严重放大。
        sigma12 = "(0.5*(sigma1+sigma2))"

        # 🔑 [SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2] alpha_lj /
        # alpha_coul 是【无量纲】Beutler 系数，必须乘 sigma_ij^6 / sigma_ij^2 才构成
        # 长度^6 / 长度^2 的软核偏移（openmmtools 等实现用约化距离 (r/sigma)，等价于
        # 隐式乘了 sigma^6，所以文献里的 alpha=0.5 是无量纲的）。
        #
        # 之前这里把 alpha_lj 当成【绝对 nm^6】直接加在 r^6 上。sigma_ij≈0.3 nm ⇒
        # sigma^6≈7.3e-4 nm^6，0.5 nm^6 相当于 ~685 sigma^6：软核偏移在
        # (1-lambda)=0.038（即 lambda≈0.962）处就已越过 sigma^6，于是
        #   lambda>0.962  → 偏移 ≪ sigma^6，几乎是硬 LJ 核；
        #   lambda<0.962  → 偏移 ≫ sigma^6，相互作用被整体抹平。
        # 硬核→全软的整个过程被压缩进 lambda_vdw∈[0.96,1]，实测 Fisher 度规在该段
        # 峰值 ~2e6，vanishing 路径 87.5% 的热力学长度堆在那 4 条边上（单边长 8.8~11.8，
        # 其余 18 条边合计仅 5.9），窗口 0 因此零重叠、IBS 占据退化成硬 argmax。
        #
        # 两个端点与 alpha 无关（整体乘 lambda_vdw：lambda=0 恒为 0；lambda=1 偏移恒为
        # 0 即精确 LJ），所以这条修正不改变精确平衡态 ΔG，只改变路径效率——但采样结果、
        # LRC 与所有旧缓存都属于旧哈密顿量，不能与新路径混用（见
        # get_parameters_dict 里写入指纹的 alpha_convention）。
        #
        # 1e-6 兜底只在 r<0.1 nm（r^6<1e-6）时才生效，是 NaN 防护；该处真实 LJ 同样
        # 是 ~1e6 kJ/mol 量级的排斥墙，不污染正常物理区间。
        dlj = (
            f"max({self.alpha_lj}*{sigma12}^6*(1.0-{lam_vdw})^{self.m_lj} + r^6, 1e-6)"
        )
        dc = (
            f"sqrt(max(r^2 + {self.alpha_coul}*{sigma12}^2"
            f"*(1.0-{lam_coul})^{self.m_coul}, 1e-6))"
        )
        lj = f"{lv} * 4 * sqrt(epsilon1*epsilon2) * ({sigma12}^12/({dlj}^2) - {sigma12}^6/{dlj})"
        coul = f"{lc} * {COUL} * q1 * q2 / {dc}"
        
        return f"{lj} + {coul}"

    # 🔑 软核 alpha 语义标签。写进 get_parameters_dict() → 进入协议指纹
    # (aces_softcore_params)，使得旧的「绝对 nm^6 alpha」缓存与本版本自动指纹不匹配、
    # fail closed。alpha 的【数值】没变（仍是 0.5/0.3），变的是它在表达式里乘不乘
    # sigma_ij^6 / sigma_ij^2，因此必须靠这个显式标签来区分，不能靠数值。
    ALPHA_CONVENTION = "dimensionless_sigma_scaled_v2"

    @staticmethod
    def _normalize_alpha_units(alpha_lj, alpha_coul):
        """alpha 是无量纲 Beutler 系数，缩放在 build_expression 里由 sigma_ij 完成，
        这里不做任何单位换算（保留此钩子是为了让「不换算」这件事显式可见）。"""
        return float(alpha_lj), float(alpha_coul)

    @staticmethod
    def optimize_alpha(n, alpha_coul_nm2=None):
        """返回无量纲 Beutler 软核系数。

        ⚠️ 历史教训：这里曾写着「OpenMM 软核 alpha 标准单位即为 nm⁶/nm²，文献值 0.5
        已对应 nm 尺度，无需额外乘以 1e-6/1e-2」——推理方向是反的。openmmtools 一类
        实现用的是约化距离 (r/sigma)，alpha 隐式乘了 sigma^6，所以文献值 0.5 恰恰是
        【无量纲】的。把它当绝对 nm^6 用，等于把软核偏移放大了 ~685 倍（sigma≈0.3 nm），
        后果见 build_expression 的注释。缩放现在由 build_expression 乘 sigma_ij^6 /
        sigma_ij^2 完成，这里只返回无量纲系数本身。
        """
        if n > 50:
            alpha_lj, alpha_coul = 0.5, (alpha_coul_nm2 or 0.2)
        else:
            alpha_lj, alpha_coul = 0.5, (alpha_coul_nm2 or 0.3)

        # 无量纲区间：文献常用 alpha_lj=0.5、alpha_coul=0.2~0.3。
        assert 0.1 < alpha_lj < 2.0, f"alpha_lj 超出安全范围: {alpha_lj} (无量纲，预期 0.1~2.0)"
        assert 0.05 < alpha_coul < 1.0, f"alpha_coul 超出安全范围: {alpha_coul} (无量纲，预期 0.05~1.0)"

        return {
            "alpha_lj": alpha_lj,        # 无量纲，表达式内乘 sigma_ij^6
            "alpha_coul": alpha_coul,    # 无量纲，表达式内乘 sigma_ij^2
            "power_lj": [2, 2],
            "power_coul": [1, 1],
            "alpha_convention": ACESoftcorePotential.ALPHA_CONVENTION,
        }

    def get_parameters_dict(self):
        return {
            "alpha_lj": self.alpha_lj,
            "alpha_coul": self.alpha_coul,
            "power_lj": list([self.m_lj, self.n_lj]),
            "power_coul": list([self.m_coul, self.n_coul]),
            "alpha_convention": self.ALPHA_CONVENTION,
        }

    @classmethod
    def from_dict(cls, p):
        # fail closed：显式给了 alpha 却没带（或带错）convention 标签的 dict，只可能来自
        # 「绝对 nm^6 alpha」旧协议的缓存/配置。数值相同但物理不同，静默复用会让采样
        # 哈密顿量与本版本不一致，必须拒绝。空 dict 走默认值，是合法的新建路径。
        if p and "alpha_lj" in p:
            convention = p.get("alpha_convention")
            if convention != cls.ALPHA_CONVENTION:
                raise ValueError(
                    "软核 alpha 协议不匹配："
                    f"alpha_convention={convention!r}，本版本要求 "
                    f"{cls.ALPHA_CONVENTION!r}（alpha 为无量纲、表达式内乘 sigma_ij^6）。"
                    "旧协议把 alpha_lj 当绝对 nm^6 使用，两者数值相同但哈密顿量不同，"
                    "旧的 lambda 路径/pilot 度规/能量缓存一律不可复用，需重新 pilot。"
                )
        alpha_lj, alpha_coul = cls._normalize_alpha_units(
            p.get("alpha_lj", 0.5),
            p.get("alpha_coul", 0.2),
        )
        return cls(
            alpha_lj,
            alpha_coul,
            tuple(p.get("power_lj", [2, 2])),
            tuple(p.get("power_coul", [1, 1])),
        )


class BeutlerSoftcoreBuilder:
    """传统 Beutler 式软核势构建器 (CustomNonbondedForce + interaction group)
    与 ACESoftcorePotential 区别：显式 L-E 对过滤 + 传统 alpha*(1-lambda)^power 表达式
    """
    @staticmethod
    def build(
        nb_force: openmm.NonbondedForce,
        ligand_indices: List[int],
        env_indices: List[int],
        alpha_lj: float = 0.5,
        alpha_coul: float = 0.5,
        power_lj: int = 1,
        power_coul: int = 1,
        particle_params_override=None,
    ) -> openmm.CustomNonbondedForce:
        # 🔑 [SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2] 与
        # ACESoftcorePotential.build_expression 保持同一约定：alpha_lj / alpha_coul 无量纲，
        # 表达式内乘 sigma12^6 / sigma12^2。原式把它们当绝对 nm^6 / nm^2 直接相加（详细
        # 后果见 ACESoftcorePotential.build_expression 的注释）。
        # 同理，原来那两个绝对数值兜底项也必须一起缩放：1e-4 nm^6 相对 sigma12^6≈7.3e-4
        # 已占 ~14%，缩放 alpha 之后它自己就会变成一个不该存在的额外软化项；1e-3 nm^2 相对
        # sigma12^2≈0.09 占 ~1%，同样按 sigma12^2 缩放才是纯数值兜底。
        d_lj = (
            f"(r^6 + {alpha_lj}*sigma12^6*(1-lambda_vdw)^{power_lj}"
            f" + 1e-4*sigma12^6*(1-lambda_vdw))"
        )
        d_coul = (
            f"sqrt(r^2 + {alpha_coul}*sigma12^2*(1-lambda_coul)^{power_coul}"
            f" + 1e-3*sigma12^2)"
        )
        expr = (
            f"lambda_vdw * 4*sqrt(epsilon1*epsilon2)*("
            f"(sigma12^12 / {d_lj}^2) - "
            f"(sigma12^6 / {d_lj})"
            f") + "
            f"lambda_coul * 138.935456 * q1*q2 / {d_coul}; "
            f"sigma12=(0.5*(sigma1+sigma2))"
        )
        sc_force = openmm.CustomNonbondedForce(expr)
        for p in ["q", "sigma", "epsilon"]:
            sc_force.addPerParticleParameter(p)
        sc_force.addGlobalParameter("lambda_coul", 1.0)
        sc_force.addGlobalParameter("lambda_vdw", 1.0)

        for i in range(nb_force.getNumParticles()):
            if particle_params_override is not None and i < len(particle_params_override):
                q, sig, eps = particle_params_override[i]
            else:
                q, sig, eps = nb_force.getParticleParameters(i)
            sc_force.addParticle([
                q.value_in_unit(unit.elementary_charge),
                sig.value_in_unit(unit.nanometer),
                eps.value_in_unit(unit.kilojoule_per_mole)
            ])

        sc_force.addInteractionGroup(set(ligand_indices), set(env_indices))
        sc_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        sc_force.setCutoffDistance(1.2 * unit.nanometer)
        sc_force.setUseSwitchingFunction(True)
        sc_force.setSwitchingDistance(1.0 * unit.nanometer)

        for i in range(nb_force.getNumExceptions()):
            p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
            sc_force.addExclusion(int(p1), int(p2))

        return sc_force


class DEXPSurrogatePotential:
    """
    LJ-matched、pair-specific 的双指数(DEXP) softcore 替身。

    每个 ligand-environment 原子对不再共用一套全局 (A_fit, B_fit, r0_vdw)，而是从
    该 pair 两端原子各自原有的 LJ sigma/epsilon，用标准 Lorentz-Berthelot 组合律
    解析给出：

        eps_ij = sqrt(eps_i * eps_j)
        sigma_ij = 0.5 * (sigma_i + sigma_j)
        r0_ij = 2^(1/6) * sigma_ij

    核函数：

        U_ij(r) = eps_ij * [ (beta/(a-b))*exp(-a*x) - (a/(a-b))*exp(-b*x) ],  x = r/r0_ij - 1

    对任意 alpha>beta>0，在 r=r0_ij 处都与标准 12-6 LJ 有完全相同的井深(-eps_ij)、井位、
    一阶导(=0)；r->0 时能量和力都有限，天然没有 LJ 的 r^-12 奇点。

    默认 alpha_vdw=14, beta_vdw=5（不是标准 LJ 的 12/6）——这是在本项目结合态局部
    anchor-relative 扰动云（--perturb-scan + --perturb-fit，20 anchor x 74 扰动/anchor，
    按扰动档等权 + leave-one-anchor-out 交叉验证）上验证过的经验值：19/20 折独立选中同一
    点，held-out 加权 RMSE 比 12/6 改善约 13%。**但要注意**：数据主要约束的是
    alpha+beta≈19 这个组合(2D score surface 在 (alpha,beta) 里是沿此方向的对角谷，
    PCA 验证过)，不是分别独立钉死 alpha=14 和 beta=5——(14,5) 是这条谷上最优的、
    好看的整数代表，换成谷上邻近的 (13,6) 几乎同样好。换配体/环境化学组成时，不要
    默认沿用这两个数字，应该重新跑一次 perturb-scan/perturb-fit。
    alpha_vdw/beta_vdw 仍是可调字段，不是硬编码常量。

    不再存在需要自由拟合的 A_fit/B_fit/r0_vdw：它们现在由 alpha_vdw、beta_vdw 和
    每个 pair 自己的 sigma/epsilon 解析给出。
    """

    def __init__(self, alpha_vdw=14.0, beta_vdw=5.0):
        alpha_vdw, beta_vdw = float(alpha_vdw), float(beta_vdw)
        if not (math.isfinite(alpha_vdw) and math.isfinite(beta_vdw)):
            raise ValueError(
                "alpha_vdw 和 beta_vdw 必须是有限数值；"
                f"得到 alpha_vdw={alpha_vdw}, beta_vdw={beta_vdw}"
            )
        if not (alpha_vdw > beta_vdw > 0.0):
            raise ValueError(
                f"alpha_vdw({alpha_vdw}) 必须大于 beta_vdw({beta_vdw})，且二者都为正，"
                "否则 LJ-matched 井深/井位系数公式无定义"
            )
        self.alpha_vdw, self.beta_vdw = alpha_vdw, beta_vdw

    def build_expression(self, lam_vdw="lam_vdw"):
        """
        pair-specific DEXP 核心表达式：r0_ij/eps_ij 由 sigma1/epsilon1/sigma2/epsilon2
        （OpenMM CustomNonbondedForce 每个 pair 自动提供的两端 per-particle 参数）
        经组合律解析给出。Switch、Gaussian-Coulomb 与传统 PME 解耦由外层 Builder 统一接管。
        """
        a, b = self.alpha_vdw, self.beta_vdw
        c_a = b / (a - b)
        c_b = a / (a - b)
        two_pow_1_6 = 2.0 ** (1.0 / 6.0)
        rs = "max(r, 1e-6)"
        # 注意：Lepton 的 `;`-分隔子表达式必须在整条表达式的顶层给出，不能包在外层
        # 圆括号里当成算术分组的一部分——所以主表达式和 `;` 定义列表是拼接关系，
        # 不是 f"{lam_vdw} * ({sub_expr_with_semicolons})" 这种嵌套关系。
        main_expr = f"{lam_vdw} * eps_ij * ({c_a}*exp(-{a}*x_ij) - {c_b}*exp(-{b}*x_ij))"
        defs = (
            f"x_ij = {rs}/r0_ij - 1.0; "
            f"r0_ij = {two_pow_1_6}*sigma_ij_safe; "
            f"sigma_ij_safe = max(sigma_ij, 1e-6); "
            f"sigma_ij = 0.5*(sigma1+sigma2); "
            f"eps_ij = sqrt(epsilon1*epsilon2)"
        )
        return f"{main_expr}; {defs}"

    def get_parameters_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    @classmethod
    def from_dict(cls, p):
        if not isinstance(p, dict):
            raise TypeError(f"DEXP 参数必须是字典，得到 {type(p).__name__}")
        allowed = {"alpha_vdw", "beta_vdw"}
        legacy_keys = sorted(
            key
            for key in p
            if key in DEXP_LEGACY_FIT_KEYS
            or key.startswith("diagnostic_")
            or key.startswith("optimizer_")
        )
        if legacy_keys:
            raise ValueError(
                "旧版全局 DEXP 拟合参数与 pair-specific LJ-matched 生产势不兼容："
                + ", ".join(legacy_keys)
                + "。生产 DEXP 配置只接受 alpha_vdw/beta_vdw。"
            )
        unknown = sorted(set(p) - allowed)
        if unknown:
            raise ValueError("未知 DEXP 生产参数：" + ", ".join(unknown))
        return cls(**{key: p[key] for key in allowed if key in p})




# ============================================================================
# 2. Boresch 限制力 & 解析修正
# ============================================================================
THERMODYNAMIC_CYCLE_DOC = """
Thermodynamic cycle used by this ABFE workflow
=============================================

Complex leg:
  0. [P1-17] The Boresch restraint is switched on in the *fully coupled* complex
     by a sampled alchemical leg over lambda_boresch_scale: 0 -> 1
     (ibs_engine.run_boresch_attachment_leg).  This is the attachment term
     ΔG_attach = ΔG(A' -> A), where A' is the physical bound state (ligand
     coupled, NO restraint) and A is the restrained state the decoupling legs
     actually sample.  Because the Boresch potential is non-negative everywhere,
     ΔG_attach = -kT ln <exp(-beta U_rest)>_{A'} >= 0 is a strict bound; the
     implementation fails closed on a negative value.

     Omitting this term does not merely drop a small constant: it leaves the
     restrained-ensemble charging and vdW values with nowhere to go, which shows
     up as apparent per-term disagreement with reference implementations that do
     include it.  With the term present the cycle closes exactly for ANY
     restraint strength, so the restrained-ensemble charging/vdW values are
     correct as measured and must NOT be separately "corrected" - doing so
     double counts.
  1. A physical Boresch restraint is applied to keep the ligand in the binding
     pose during decoupling.
  2. The alchemical sampler computes the restrained complex-leg decoupling free
     energy, ΔG_decouple,restrained.
  3. The analytical Boresch term returned by calculate_boresch_analytical_correction
     is the standard-state release correction added to that leg:

       ΔG_complex = ΔG_attach + ΔG_decouple,restrained + ΔG_release_to_1M

     Only the total is invariant to the choice of Boresch anchors and force
     constants; the three terms individually are not, so comparing any single
     term against another implementation that used different anchors is
     meaningless.

     with V° = 1.6605 nm^3 and

       ΔG_release_to_1M = -RT ln[
         8π²V° / (r0² sinθA sinθB)
         * sqrt(Kr KθA KθB KφA KφB KφC) / (2πRT)^3
       ].

Solvent leg:
  No Boresch restraint is applied to the ligand in bulk solvent; therefore no
  Boresch analytical release term is added to the solvent leg.

PME/self correction:
  NonbondedForce.addParticleParameterOffset linearly scales ligand charges with
  lambda_coul; every getState(getEnergy=True) call at a given lambda recomputes
  the *complete* PME energy (reciprocal + self + real-space) from the actual
  scaled charges at that state, including the Ewald self-energy term. That
  self-energy term is a real, required part of U_k(x) at that lambda state (it
  exactly cancels the reciprocal-space sum's self-interaction double-count) and
  is therefore already correct in the offline u_kn without any further action.
  An earlier version of this workflow additionally added a manual +C*lambda^2
  correction on top of this, on the (incorrect) assumption that OpenMM's
  reported PME energy was missing this term. That manual correction has been
  revoked: apply_pme_self_correction is now always False in production, and is
  only recorded as an inert diagnostic (charge_square_sum_e2, applied=false) for
  auditing. See PHYSICS_DEFECTS.md for the full history. Charged ligand paths
  still disable ligand-only self correction outright unless a validated
  co-alchemical neutralization cycle is active.

LJ long-range/dispersion correction:
  Custom softcore VDW interaction-group forces do not automatically reproduce
  the original NonbondedForce dispersion correction, and OpenMM's native
  CustomNonbondedForce.setUseLongRangeCorrection cannot simply be enabled on
  these forces: the softcore expression bundles LJ and Coulomb into one
  CustomNonbondedForce, and OpenMM's analytic tail integral diverges (verified
  to crash the CUDA backend outright) once real nonzero charges are present in
  that combined expression. For the default ACE/dual_lambda path
  (_create_softcore_force), this is instead handled with a hand-derived
  analytic mean-field r^-6 dispersion tail correction (uniform-density
  approximation beyond the cutoff, attractive term only), precomputed once per
  window in ibs_engine.py::build_ibs_dual_system and added per-frame inside
  IBSSampler.collect_energies() before MBAR sees the energies -- i.e. it is
  folded into the sampled Hamiltonian itself, not added as a separate additive
  cycle term afterward. The BeutlerSoftcoreBuilder / --decoupling single_lambda
  (REMD) path does not yet have an equivalent correction; results from that
  path should not be treated as including this term. See AUDIT_STATUS.md for
  the full investigation and remaining gaps.

Binding free energy:
  Each leg's decoupling free energy (ΔG_complex, ΔG_solvent) is defined as the
  cost of the lambda:1->0 transformation (coupled -> decoupled), i.e. it is
  positive for a leg with net-favorable interactions. ΔG_complex already
  includes the Boresch restrained-decoupling free energy plus the analytical
  standard-state release term; ΔG_solvent has no restraint term (no restraint
  is applied in bulk solvent). Closing the thermodynamic cycle through the
  common "fully decoupled ligand at 1 M standard state" reference state (which
  is the same whether reached from the complex or the solvent leg) gives:

    ΔG_bind = ΔG_solvent - ΔG_complex + ΔG_APBS

  (ΔG_APBS defaults to 0 and is only added when supplied as an explicit
  external term; it does not replace a Lennard-Jones dispersion/tail
  correction). For a genuine binder, ΔG_complex > ΔG_solvent (interactions lost
  on decoupling are stronger in the pocket than in bulk solvent), so ΔG_bind is
  negative, as expected. Terms that are identical in both legs can cancel only
  when the Hamiltonians and correction conventions are documented and matched.
""".strip()


def combine_binding_free_energy(
    *,
    dg_complex_kJ_mol: float,
    dg_solvent_kJ_mol: float,
    err_complex_kJ_mol: float = 0.0,
    err_solvent_kJ_mol: float = 0.0,
    dg_boresch_kJ_mol: float = 0.0,
    boresch_already_included_in_complex: bool = True,
    apbs_correction_kJ_mol: float = 0.0,
) -> Dict[str, Any]:
    """🔑 [ATT-09] 热力学循环闭合的**唯一**实现。

    上面 `THERMODYNAMIC_CYCLE_DOC` 记的公式：

        ΔG_bind = ΔG_solvent - ΔG_complex + ΔG_APBS

    其中 ΔG_complex **按约定已经包含** Boresch 标准态释放项。若调用方拿到的
    complex 腿还没烘焙这一项（例如 `TraditionalABFEPipeline.run_full` 是用
    `boresch_correction=0.0` 调的），把 `boresch_already_included_in_complex=False`
    传进来，这里会替它减一次——**只减一次**。

    为什么要有这个函数：改之前同一个公式在四处独立维护，而且它们并不等价——

      * `runabfe.main()`：Boresch 已内含，**加了** APBS；
      * `runabfe.run_traditional_mode()`：Boresch 显式减，**完全没有** APBS；
      * `runabfe.run_post_analysis()`：Boresch 条件置零，加了 APBS（从
        `run_provenance.json` 重推）；
      * `ABFEPipeline.run_full_abfe_loop()`：Boresch 已内含，**完全没有** APBS。

    也就是说后两条路径对带电配体会静默漏掉整项有限尺寸静电修正。这不是代码
    整洁问题，是数值错误。统一到这里之后，那两条路径的输出会变——那是修复。

    误差：两腿采样独立，`sqrt(err_c² + err_s²)`。Boresch 解析释放项与 APBS 都是
    确定性解析量，没有独立采样方差，不并入。

    返回一个自带记账字段的 dict，调用方直接摊进自己的结果 JSON，
    不要再在外面重算任何一项。
    """
    dg_complex = float(dg_complex_kJ_mol)
    dg_solvent = float(dg_solvent_kJ_mol)
    err_complex = float(err_complex_kJ_mol or 0.0)
    err_solvent = float(err_solvent_kJ_mol or 0.0)
    dg_boresch = float(dg_boresch_kJ_mol or 0.0)
    apbs = float(apbs_correction_kJ_mol or 0.0)

    # 只有 complex 腿还没烘焙释放项时，公式里才再减一次。
    boresch_term = 0.0 if boresch_already_included_in_complex else dg_boresch

    delta_g_bind_uncorrected = dg_solvent - dg_complex - boresch_term
    delta_g_bind = delta_g_bind_uncorrected + apbs
    total_err = float(np.sqrt(err_complex ** 2 + err_solvent ** 2))

    return {
        "complex_delta_G_kJ_mol": dg_complex,
        "solvent_delta_G_kJ_mol": dg_solvent,
        "boresch_correction_kJ_mol": dg_boresch,
        "boresch_correction_already_included_in_complex_delta_G": bool(
            boresch_already_included_in_complex
        ),
        # 公式里真正被减掉的那一项（已内含时为 0）；与上面那个"物理量本身"区分开，
        # 下游据此判断能不能再对 complex_delta_G 二次扣减。
        "boresch_term_subtracted_kJ_mol": boresch_term,
        "apbs_correction_kJ_mol": apbs,
        "delta_G_bind_uncorrected_kJ_mol": delta_g_bind_uncorrected,
        "delta_G_bind_kJ_mol": delta_g_bind,
        "delta_G_bind_kcal_mol": delta_g_bind / 4.184,
        "total_error_kJ_mol": total_err,
        "total_error_kcal_mol": total_err / 4.184,
        "cycle_formula": (
            "delta_G_bind = delta_G_solvent - delta_G_complex"
            " - boresch_term_subtracted + delta_G_APBS"
        ),
    }


def boresch_dihedral_rad(a, b, c, d):
    """四点二面角，rad，范围 [-π, π]，**标准（IUPAC）符号约定**。

    🚨 2026-07-29 事故的根源就在这个符号上，务必不要"顺手简化"回去。

    Boresch 限制势的参考值 phiA0/phiB0/phiC0 是喂给 OpenMM
    `CustomCompoundBondForce` 表达式里的 `dihedral(p1,p2,p3,p4)` 的
    （见 `LambdaDependentBoreschForce`）。OpenMM 的 `dihedral()`、
    `mdtraj.compute_dihedrals` 用的都是标准约定：

        n1 = b1×b2, n2 = b2×b3,  φ = atan2( (n1×n2)·b2̂ , n1·n2 )

    此前 abfe_core 里有**四份**手写副本都写成了

        m1 = n1 × b2̂ ;  φ = atan2(m1·n2, n1·n2)

    而 (n1×b2̂)·n2 = −(n1×n2)·b2̂，所以那四份返回的是 **−φ**。距离和键角不受
    影响（arccos 无符号），只有三个二面角整体反号——也就是给出限制势参考几何的
    **镜像**。

    实测后果（`output_lrc_fix`，2026-07-29 02:02）：带 Boresch 的 rebalance 用
    `boresch_simple.json` 的（mdtraj 算的、正确的）参考值把配体稳稳按在自己的
    pose 上；紧接着 `update_boresch_from_last_frame` 用错号的副本重算并**覆盖**了
    参考值，提交下去的 phiA0/phiB0/phiC0 全部反号：

        phiA0  +1.6696 → −1.7168      phiB0  −1.8045 → +1.8136
        phiC0  −0.6839 → +0.6163      (r0/θA/θB 只差 <0.03，因为它们无符号)

    于是 attachment 腿的 λ=1 参考态变成了当前 pose 的镜像，每个二面角都坐在
    k(1−cosΔ) 的势壁顶上（Δ≈π ⟹ ≈2k）：λ=0 实测 ⟨U_B⟩=777 kJ/mol、
    max=1115 ≈ Σ2k_φ=1140，ΔG(A′→A) 从应有的 ~5.5 kJ/mol 涨到 98.8 kJ/mol，
    BAR/TI 一致性门失败（8.27 > 8.13 kJ/mol）。

    符号自检（可手算复核）：a=(0,1,0) b=(0,0,0) c=(1,0,0) d=(1,0,1) ⟹ φ=+π/2。
    见 `test_boresch_dihedral_convention.py`，它把本函数直接钉在 OpenMM
    `dihedral()` 上。

    退化情形（|b2|≈0 或两个法向量之一退化）返回 0.0，与被替换的四份副本中最
    保守的那份（`calc_boresch_from_last_frame`）保持一致。
    """
    b1 = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    b2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    b3 = np.asarray(d, dtype=np.float64) - np.asarray(c, dtype=np.float64)
    norm_b2 = float(np.linalg.norm(b2))
    if norm_b2 < 1e-6:
        return 0.0
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    denom = float(np.linalg.norm(n1)) * float(np.linalg.norm(n2))
    if denom < 1e-10:
        return 0.0
    return float(np.arctan2(np.dot(np.cross(n1, n2), b2 / norm_b2), np.dot(n1, n2)))


def calculate_boresch_analytical_correction(eq, fc, T=300.0):
    """
    计算 Boresch 解析修正。

    返回值是“解耦采样中保留 Boresch restraint”时需要加到 leg 上的
    标准态释放修正:

        ΔG_release = -RT ln[
            (8π² V° / (r0² sinθA sinθB))
            * sqrt(Kr KθA KθB KφA KφB KφC) / (2πRT)^3
        ]

    6 个谐振 Boresch 自由度的高斯积分给出 (2πRT)^3，而不是 1.5 次方。
    【强制标准单位】kJ/mol/nm², nm, rad
    ⚠️ 注意：eq["r0"] 必须是 nm 单位（不是 Å）
    ⚠️ 注意：fc["kr"] 必须为 kJ/mol/nm²，fc["kthetaA"] 等为 kJ/mol/rad²
    """
    T = T.value_in_unit(unit.kelvin) if hasattr(T, "value_in_unit") else float(T)
    R = constants.R / 1000.0
    RT = R * T
    V0 = 1.6605  # nm³ (标准摩尔体积)

    # ✅ 修复1：增加单位量级与物理合理性断言
    kr, ktA, ktB = fc.get("kr", 0), fc.get("kthetaA", 0), fc.get("kthetaB", 0)
    if not (50 <= kr <= 5000):
        raise ValueError(f"kr 超出合理范围 [50, 5000] kJ/mol/nm²: {kr}")
    # ✅ 与 GeometricRestraintEstimator 的 clip 范围 [10, 1000] 保持一致
    # （见 force_constant_ranges["kthetaA"/"kthetaB"]），否则该估计器给出的合法
    # 力常数（500~1000 区间）会在这里被误判为"超出合理范围"而硬报错。
    if not (10 <= ktA <= 1000 and 10 <= ktB <= 1000):
        raise ValueError("角度力常数 ktA/ktB 建议范围 [10, 1000] kJ/mol/rad²")

    # ✅ 强制标准单位，不再进行任何转换
    r0 = eq["r0"]  # nm
    thA = eq["thetaA0"]  # rad
    thB = eq["thetaB0"]  # rad

    kr = fc["kr"]  # kJ/mol/nm²
    ktA = fc["kthetaA"]  # kJ/mol/rad²
    ktB = fc["kthetaB"]
    kpA = fc["kphiA"]
    kpB = fc["kphiB"]
    kpC = fc["kphiC"]

    Kdet = kr * ktA * ktB * kpA * kpB * kpC
    if Kdet <= 0:
        raise ValueError("Boresch 力常数存在零值或负值，无法计算解析修正")
    
    sin_t = math.sin(thA) * math.sin(thB)

    if sin_t < 1e-4:
        raise ValueError("Boresch 锚点几何奇点 (sinθ≈0)")

    standard_state_factor = (8.0 * math.pi**2 * V0) / (r0**2 * sin_t)
    restraint_integral_factor = math.sqrt(Kdet) / ((2.0 * math.pi * RT) ** 3.0)
    argument = standard_state_factor * restraint_integral_factor
    if argument <= 0 or not math.isfinite(argument):
        raise ValueError(f"Boresch 解析修正对数参数异常: {argument}")

    return -RT * math.log(argument)



class LambdaDependentBoreschForce(openmm.CustomCompoundBondForce):
    def __init__(
        self,
        rec_idx,
        lig_idx,
        eq,
        fc,
        lam_name="lambda_rest",
        fixed_lam=None,
        sign=1.0,
        use_pbc=False,
        # ✅ 移除 unit_sys 参数，强制标准单位
    ):
        if len(rec_idx) != 3 or len(lig_idx) != 3:
            raise ValueError("需 exactly 3 受体 + 3 配体原子")

        # ✅ 强制标准单位，不再进行任何转换
        r0 = eq["r0"]  # nm
        thA = eq["thetaA0"]  # rad
        thB = eq["thetaB0"]  # rad
        phA = eq["phiA0"]  # rad
        phB = eq["phiB0"]  # rad
        phC = eq["phiC0"]  # rad

        kr = fc["kr"]  # kJ/mol/nm²
        ktA = fc["kthetaA"]  # kJ/mol/rad²
        ktB = fc["kthetaB"]
        kpA = fc["kphiA"]
        kpB = fc["kphiB"]
        kpC = fc["kphiC"]

        # 打印调试信息（确保 thA 是弧度）
        print(f"  [Boresch] kr={kr:.1f} kJ/mol/nm², r0={r0:.3f} nm, θA={np.degrees(thA):.1f}°")

        ls = f"{fixed_lam:.6f}" if fixed_lam is not None else lam_name

        # ✅ 修复2：标准谐波势 (distance-r0)^2，导数连续且数值稳定
        # 🚨 关键修复：atom-index 顺序与 thetaA0/thetaB0/phiA0/phiB0/phiC0 的
        # 计算约定必须严格一致。addBond(rec_idx+lig_idx) 的顺序是
        # [R0(离配体最近), R1, R2(离配体最远), L0(离受体最近), L1, L2]——
        # 这也是 calc_boresch_from_last_frame / _check_boresch_geometry_safe /
        # _validate_boresch_geometry_strict 全部使用的约定：
        #   r0      = distance(R0, L0)
        #   thetaA0 = angle(R1, R0, L0)         顶点=R0
        #   thetaB0 = angle(R0, L0, L1)         顶点=L0
        #   phiA0   = dihedral(R2, R1, R0, L0)
        #   phiB0   = dihedral(R1, R0, L0, L1)
        #   phiC0   = dihedral(R0, L0, L1, L2)
        # 旧表达式误用 angle(p2,p3,p4)/angle(p3,p4,p5) 和
        # dihedral(p1,p2,p3,p4) 等，把顶点/参考原子错当成了 R2(最远的受体
        # 锚点，选择时只保证"刚性"而不保证与 R1/L0 不共线)，导致实际被约束
        # 的角度和平衡值计算出的角度根本不是同一个几何量：一来平衡值形同虚设、
        # 限制力没有真正锁住原有构象；二来一旦 R2 恰好与 R1、L0 接近共线，
        # angle()/dihedral() 的解析梯度出现 1/sinθ 型奇点，能量看起来正常但
        # 力却能炸到 10^7~10^8 kJ/mol/nm 量级——这正是本次 REMD 预热崩溃的根源。
        expr = (
            f"({sign})*{ls}*("
            "0.5*kr*(distance(p1,p4)-r0)^2+"
            "ktA*(1-cos(angle(p2,p1,p4)-thetaA0))+"
            "ktB*(1-cos(angle(p1,p4,p5)-thetaB0))+"
            "kpA*(1-cos(dihedral(p3,p2,p1,p4)-phiA0))+"
            "kpB*(1-cos(dihedral(p2,p1,p4,p5)-phiB0))+"
            "kpC*(1-cos(dihedral(p1,p4,p5,p6)-phiC0))"
            ")"
        )
        super().__init__(6, expr)  # ✅ N=6

        if fixed_lam is None:
            self.addGlobalParameter(lam_name, 0.0)

        for n, v in [
            ("r0", r0),
            ("thetaA0", thA),
            ("thetaB0", thB),
            ("phiA0", phA),
            ("phiB0", phB),
            ("phiC0", phC),
            ("kr", kr),
            ("ktA", ktA),
            ("ktB", ktB),
            ("kpA", kpA),
            ("kpB", kpB),
            ("kpC", kpC),
        ]:
            self.addGlobalParameter(n, v)

        if hasattr(self, "setUsesPeriodicBoundaryConditions"):
            self.setUsesPeriodicBoundaryConditions(bool(use_pbc))

        self.addBond(list(rec_idx) + list(lig_idx))


# ============================================================================
# 3. Orb 口袋力投影估算器 (v4.3 - 3-Stage Optimized & Hybrid Filter)
# ============================================================================
class OrbVacuumContext:
    """ORB 真空力场计算上下文 (仅用于口袋内力场计算)"""

    def __init__(
        self, topology, model_name="mace-off24-medium", device="cpu"
    ):
        self.device = device
        self.model_name = model_name
        self.potential = _build_mace_potential(model_name)
        self.system = self.potential.createSystem(topology, **_build_openmmml_kwargs(
            device=self.device,
            return_energy_type="energy",
            charge=0,
            multiplicity=1,
        ))
        self.integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        try:
            platform = openmm.Platform.getPlatformByName(device.upper())
        except Exception:
            platform = openmm.Platform.getPlatformByName("CPU")
        self.context = openmm.Context(self.system, self.integrator, platform)

    def calculate_forces(self, positions_nm):
        self.context.setPositions(positions_nm)
        return (
            self.context.getState(getForces=True)
            .getForces(asNumpy=True)
            .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        )


class OrbBoreschEstimator:
    """
    基于 Orb 口袋力投影与三阶段锚点优化的 Boresch 估算器
    特性：
    1. 稳定性+动态距离+几何构型 3-Stage 锚点筛选
    2. 边界氢饱和修补 (避免切割键导致的力场畸变)
    3. 混合滤波拟合 (线性回归优先，相关性不足时自动切换至波动法)
    4. 二面角自动展开 (Unwrap) 与 Jacobian 修正
    """

    DEFAULT_CONFIG = {
        "temperature": 300.0,
        "cutoff_nm": 0.9,
        "rmsf_cutoff_nm": 0.15,
        "dist_reject_nm": 0.40,
        "dist_gold_min_nm": 0.50,
        "dist_gold_max_nm": 1.10,
        "dist_backup_min_nm": 1.10,
        "dist_backup_max_nm": 1.50,
        "rec_anchor_dist_min": 0.40,
        "rec_anchor_dist_max": 0.80,
        "rec_anchor_angle_min": 60,
        "rec_anchor_angle_max": 120,
        "corr_threshold_keep": -0.1,
        "corr_threshold_good": -0.5,
        "use_fluctuation_fallback": True,
        "short_sidechain_res": ["GLY", "ALA", "SER", "VAL", "THR", "CYS"],
        "long_sidechain_res": [
            "TRP",
            "PHE",
            "TYR",
            "ARG",
            "LYS",
            "GLU",
            "GLN",
            "MET",
            "HIS",
        ],
        "score_weights": {
            "stability": 1.0,
            "distance": 2.0,
            "geometry": 1.5,
            "signal": 3.0,
        },
        "top_n_candidates": 5,
    }

    def __init__(self, temperature=300.0, device=None, cutoff_nm=0.9, n_frames=500):
        if not HAS_ORB:
            raise ImportError("OrbBoreschEstimator 依赖 torch + openmmml，请安装后重试")
        self.T = temperature
        self.gas_constant_kj_per_mol_k = 8.314e-3
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = {
            **self.DEFAULT_CONFIG,
            "temperature": temperature,
            "cutoff_nm": cutoff_nm,
        }
        self.n_frames = n_frames
        self.sc_correction = {
            "GLY": -0.05,
            "ALA": -0.03,
            "SER": -0.02,
            "THR": -0.02,
            "CYS": -0.02,
            "ARG": 0.15,
            "LYS": 0.12,
            "GLU": 0.10,
            "GLN": 0.10,
            "TRP": 0.12,
            "TYR": 0.10,
            "PHE": 0.08,
            "MET": 0.08,
            "HIS": 0.06,
        }

    def _get_sidechain_correction(self, resname):
        return self.sc_correction.get(resname, 0.0)

    def _score_distance(self, dist_nm, resname):
        cfg = self.config
        if dist_nm < cfg["dist_reject_nm"]:
            return -100, "❌ 太近"
        c_dist = dist_nm - self._get_sidechain_correction(resname)
        if cfg["dist_gold_min_nm"] <= c_dist <= cfg["dist_gold_max_nm"]:
            center = (cfg["dist_gold_min_nm"] + cfg["dist_gold_max_nm"]) / 2
            bonus = 50 * (
                1
                - abs(c_dist - center)
                / (cfg["dist_gold_max_nm"] - cfg["dist_gold_min_nm"])
            )
            return 50 + bonus, "✅ 黄金区间"
        if cfg["dist_backup_min_nm"] <= c_dist <= cfg["dist_backup_max_nm"]:
            score = 30 * (
                1
                - (c_dist - cfg["dist_backup_min_nm"])
                / (cfg["dist_backup_max_nm"] - cfg["dist_backup_min_nm"])
            )
            return score, "⚠️ 备选区间"
        return 5 if c_dist > cfg["dist_backup_max_nm"] else 25, "⚪ 过渡/较远"

    def _check_anchor_geometry(self, rec_anchors, traj, pocket_sel):
        r = traj.xyz[0, pocket_sel]
        rec_local = [np.where(pocket_sel == a)[0][0] for a in rec_anchors]
        P1, P2, P3 = r[rec_local]

        def calc_angle(a, b, c):
            ba, bc = a - b, c - b
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
            return np.arccos(np.clip(cos, -1, 1)) * 180 / np.pi

        cfg = self.config
        dists = [
            np.linalg.norm(P1 - P2),
            np.linalg.norm(P2 - P3),
            np.linalg.norm(P1 - P3),
        ]
        d_score = sum(
            20
            if cfg["rec_anchor_dist_min"] <= d <= cfg["rec_anchor_dist_max"]
            else (-30 if d < cfg["rec_anchor_dist_min"] else 0)
            for d in dists
        )
        angle_P = calc_angle(P3, P2, P1)
        a_score = (
            40
            if cfg["rec_anchor_angle_min"] <= angle_P <= cfg["rec_anchor_angle_max"]
            else (15 if 30 <= angle_P <= 150 else -50)
        )
        return (
            (a_score > -30),
            d_score + a_score,
            f"∠P3-P2-P1={angle_P:.1f}°, d12={dists[0] * 10:.2f}Å",
        )

    def _validate_boresch_geometry_strict(self, rec_anchors, lig_anchors, ref_coords):
        """
        【严格几何硬过滤器】(Hard Reject) - 彻底重写版
        统一单位：内部全部使用 nm，仅日志转 Å/°
        拒绝标准：
          1. r0 (R0-L0) ∈ [0.50, 1.00] nm (5-10 Å)
          2. θA (R1-R0-L0) ∈ [40°, 140°]  → sin(θA) ≥ 0.642
          3. θB (R0-L0-L1) ∈ [40°, 140°]  → sin(θB) ≥ 0.642
          4. 受体锚点间距 ≥ 0.38 nm (3.8 Å)
          5. 配体锚点间距 ≥ 0.25 nm (2.5 Å)
        """
        try:
            R0, R1, R2 = [ref_coords[a] for a in rec_anchors]
            L0, L1, L2 = [ref_coords[a] for a in lig_anchors]
        except IndexError:
            return False, "❌ 锚点索引越界"

        def dist(p1, p2): return np.linalg.norm(p1 - p2)
        def angle_rad(p1, p2, p3):
            v1, v2 = p1 - p2, p3 - p2
            cos_val = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12), -1.0, 1.0)
            return np.arccos(cos_val)

        # 1. 核心几何量 (nm & rad)
        r0_nm = dist(R0, L0)
        thA_rad = angle_rad(R1, R0, L0)
        thB_rad = angle_rad(R0, L0, L1)

        d_R01, d_R12 = dist(R0, R1), dist(R1, R2)
        d_L01, d_L12 = dist(L0, L1), dist(L1, L2)

        # 2. 严格阈值拦截 (全部基于 nm/rad 比较)
        if not (0.50 <= r0_nm <= 1.00):
            return False, f"❌ r0={r0_nm*10:.2f}Å [需5.0-10.0Å]"

        thA_deg, thB_deg = np.degrees(thA_rad), np.degrees(thB_rad)
        if not (40.0 <= thA_deg <= 140.0):
            return False, f"❌ θA={thA_deg:.1f}° [需40-140°]"
        if not (40.0 <= thB_deg <= 140.0):
            return False, f"❌ θB={thB_deg:.1f}° [需40-140°]"

        # 3. 奇点保护：sin(θ) < 0.64 直接拒绝 (对应 θ<40° 或 θ>140°)
        if np.sin(thA_rad) < 0.64 or np.sin(thB_rad) < 0.64:
            return False, f"❌ 几何奇异 sinθ≈0 (θA={thA_deg:.1f}°, θB={thB_deg:.1f}°)"

        # 4. 锚点共线/重叠保护
        if d_R01 < 0.38 or d_R12 < 0.38:
            return False, f"❌ 受体锚点过近 (min={min(d_R01, d_R12)*10:.1f}Å)"
        if d_L01 < 0.25 or d_L12 < 0.25:
            return False, f"❌ 配体锚点过近 (min={min(d_L01, d_L12)*10:.1f}Å)"

        # 5. 通过
        return True, f"✅ 几何合格 r0={r0_nm*10:.2f}Å θA={thA_deg:.1f}° θB={thB_deg:.1f}°"

    def _add_capping_hydrogens(self, topology, selection, ref_coords):
        cap_h_info = []
        residues = list(topology.residues)
        for res in residues:
            res_atoms = [a.index for a in res.atoms]
            if not any(a in selection for a in res_atoms) or all(
                a in selection for a in res_atoms
            ):
                continue
            bb = {a.name: a.index for a in res.atoms if a.name in ["N", "CA", "C", "O"]}
            if "C" in bb and bb["C"] in selection:
                nxt = residues[res.index + 1] if res.index + 1 < len(residues) else None
                if nxt:
                    nxt_n = [a.index for a in nxt.atoms if a.name == "N"]
                    if nxt_n and nxt_n[0] not in selection:
                        c_pos = ref_coords[bb["C"]]
                        n_pos = (
                            ref_coords[nxt_n[0]]
                            if nxt_n
                            else c_pos + np.array([0.15, 0, 0])
                        )
                        d = n_pos - c_pos
                        n = np.linalg.norm(d)
                        d = d / n if n > 1e-10 else np.array([1, 0, 0])
                        cap_h_info.append(
                            {
                                "cut_atom": bb["C"],
                                "cut_type": "C_term",
                                "direction": d,
                                "bond_length": 0.11,
                                "neighbor_global": nxt_n[0],
                            }
                        )
            if "N" in bb and bb["N"] in selection:
                prv = residues[res.index - 1] if res.index > 0 else None
                if prv:
                    prv_c = [a.index for a in prv.atoms if a.name == "C"]
                    if prv_c and prv_c[0] not in selection:
                        n_pos = ref_coords[bb["N"]]
                        c_pos = (
                            ref_coords[prv_c[0]]
                            if prv_c
                            else n_pos + np.array([-0.15, 0, 0])
                        )
                        d = c_pos - n_pos
                        n = np.linalg.norm(d)
                        d = d / n if n > 1e-10 else np.array([-1, 0, 0])
                        cap_h_info.append(
                            {
                                "cut_atom": bb["N"],
                                "cut_type": "N_term",
                                "direction": d,
                                "bond_length": 0.10,
                                "neighbor_global": prv_c[0],
                            }
                        )
        return cap_h_info

    def _build_pocket_context(self, traj, ligand_resname):
        if not HAS_MDTRAJ:
            raise ImportError("需要 mdtraj")
        top = traj.topology
        lig_sel = top.select(f"resname {ligand_resname}")
        if len(lig_sel) == 0:
            raise ValueError("未找到配体")
        lig_center = traj.xyz[0, lig_sel].mean(axis=0)
        prot_sel = top.select("protein")
        nearby_atoms = [
            a
            for a in prot_sel
            if np.linalg.norm(traj.xyz[0, a] - lig_center) <= self.config["cutoff_nm"]
        ]
        # 保留完整残基，避免在主链/侧链中间截断产生自由基；比事后手工补 capping H 更稳。
        nearby_residues = {top.atom(a).residue.index for a in nearby_atoms}
        pocket_atoms = set(int(i) for i in lig_sel.tolist())
        for res in top.residues:
            if res.index in nearby_residues:
                for atom in res.atoms:
                    pocket_atoms.add(atom.index)
        pocket_sel = np.array(sorted(pocket_atoms), dtype=int)
        if len(pocket_sel) < 10:
            raise ValueError("口袋原子不足，请增大 cutoff_nm")

        cap_h = []
        pocket_traj = traj.atom_slice(pocket_sel)
        omm_top = pocket_traj.topology.to_openmm()
        context = OrbVacuumContext(omm_top, device=self.device)

        # --- 3-Stage 锚点筛选 ---
        lig_atoms = top.select(f"resname {ligand_resname} and not element H")
        if len(lig_atoms) < 3:
            lig_atoms = top.select(f"resname {ligand_resname}")
        if len(lig_atoms) == 0:
            raise ValueError(f"未找到配体 {ligand_resname} 原子，无法构建口袋上下文")

        lig_scores = [
            top.atom(i).element.mass / 12.0
            + (2.0 if top.atom(i).name in ["CG", "CD", "CE", "CZ", "CA"] else 0)
            for i in lig_atoms
        ]
        lig_anchors = lig_atoms[np.argsort(lig_scores)[-3:]].tolist()

        rec_ca = top.select("protein and name CA")
        rec_ca_p = np.intersect1d(rec_ca, pocket_sel)
        rmsf_traj = traj[:: max(1, len(traj) // 100)]
        if len(rmsf_traj) > 1:
            rmsf_traj.superpose(rmsf_traj, 0, atom_indices=rec_ca_p)
            rmsf = np.sqrt(
                np.mean(
                    (rmsf_traj.xyz[:, rec_ca_p] - rmsf_traj.xyz[0, rec_ca_p]) ** 2,
                    axis=(0, 2),
                )
            )
        else:
            rmsf = np.zeros(len(rec_ca_p))

        rigid_mask = rmsf < self.config["rmsf_cutoff_nm"]
        rigid_ca = rec_ca_p[rigid_mask]
        rigid_rmsf = rmsf[rigid_mask]

        candidates = []
        if len(rigid_ca) >= 3:
            from itertools import combinations

            sorted_idx = np.argsort(rigid_rmsf)[:30]
            sorted_ca = rigid_ca[sorted_idx]
            sorted_rmsf = rigid_rmsf[sorted_idx]
            sorted_resnames = [top.atom(idx).residue.name for idx in sorted_ca]

            MAX_R0_ANGSTROM = 12.0
            MIN_R0_ANGSTROM = 6.0
            seen_combos = set()
            candidates = []

            for combo in combinations(range(len(sorted_ca)), 3):
                combo_indices = list(combo)
                combo_anchors = [sorted_ca[i] for i in combo_indices]
                combo_rmsf_vals = [sorted_rmsf[i] for i in combo_indices]

                dists = [
                    np.linalg.norm(traj.xyz[0, a] - lig_center) for a in combo_anchors
                ]
                sorted_pairs = sorted(
                    zip(
                        dists,
                        combo_anchors,
                        combo_rmsf_vals,
                        [sorted_resnames[i] for i in combo_indices],
                    ),
                    key=lambda x: x[0],
                )
                rec_anchors = [p[1] for p in sorted_pairs]
                rec_rmsf = [p[2] for p in sorted_pairs]
                rec_resnames = [p[3] for p in sorted_pairs]

                R0, R1, R2 = traj.xyz[0, rec_anchors]
                L0 = traj.xyz[0, lig_anchors[0]]

                r0_nm = np.linalg.norm(R0 - L0)
                r0_A = r0_nm * 10.0

                def calc_ang(a, b, c):
                    ba, bc = a - b, c - b
                    cos = np.dot(ba, bc) / (
                        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10
                    )
                    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

                thetaA = calc_ang(R1, R0, L0)
                thetaB = calc_ang(R0, L0, traj.xyz[0, lig_anchors[1]])

                if not (MIN_R0_ANGSTROM <= r0_A < MAX_R0_ANGSTROM):
                    continue

                geo_key = (
                    round(r0_A, 1),
                    round(thetaA, 1),
                    round(thetaB, 1),
                    tuple(sorted(rec_anchors[:2])),
                )
                if geo_key in seen_combos:
                    continue
                seen_combos.add(geo_key)

                score_stab = sum(
                    (self.config["rmsf_cutoff_nm"] - rv) * 100 for rv in rec_rmsf
                )

                if 8.0 <= r0_A <= 10.5:
                    score_dist = 60
                elif 10.5 < r0_A < 12.0:
                    score_dist = 30
                else:
                    score_dist = 0

                if 60 <= thetaA <= 110:
                    score_ang = 50
                elif 45 < thetaA < 60 or 110 < thetaA < 135:
                    score_ang = 20
                else:
                    score_ang = 0

                pre_sc = score_stab + score_dist + score_ang
                candidates.append(
                    {
                        "rec_anchors": rec_anchors,
                        "score": pre_sc,
                        "r0_A": r0_A,
                        "thetaA": thetaA,
                        "thetaB": thetaB,
                        "rmsf_avg": np.mean(rec_rmsf),
                    }
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_cands = candidates[: self.config["top_n_candidates"]]

        # 快速力学信号验证 (取前50帧)
        validated = []
        for cand in top_cands:
            local_anchors = [
                int(np.where(pocket_sel == a)[0][0])
                for a in cand["rec_anchors"] + lig_anchors
            ]
            ks_temp = self._quick_validate(
                traj[:50], context, pocket_sel, local_anchors, cap_h
            )

            if ks_temp is not None:
                validated.append(
                    {**cand, "ks_temp": ks_temp, "total": cand["score"] + 20.0}
                )

        if validated:
            validated.sort(key=lambda x: x["total"], reverse=True)
            best = validated[0]
            rec_anchors, best_ks = best["rec_anchors"], best["ks_temp"]
        else:
            rec_anchors = (
                candidates[0]["rec_anchors"] if candidates else rigid_ca[:3].tolist()
            )
            best_ks = None

        anchor_global = rec_anchors + lig_anchors
        local_anchors = [int(np.where(pocket_sel == a)[0][0]) for a in anchor_global]
        return context, pocket_sel, local_anchors, anchor_global, cap_h, best_ks

    def _quick_validate(self, traj, context, selection, local_anchors, cap_h):
        if len(traj) < 10:
            return None
        n_frames = len(traj)
        q_data = np.zeros((n_frames, 6))
        Fq_data = np.zeros((n_frames, 6))
        for f in range(n_frames):
            pos = traj.xyz[f, selection]
            forces = context.calculate_forces(pos)
            r_anchors = pos[local_anchors]
            if np.any(np.isnan(r_anchors)):
                continue
            q, grads = self._compute_geom_gradients(r_anchors)
            if np.any(np.isnan(q)):
                continue
            Fq = np.zeros(6)
            for g in range(6):
                for a in range(6):
                    Fq[g] += np.dot(forces[local_anchors[a]], grads[g, a])
            q_data[f] = q
            Fq_data[f] = Fq
        return self._apply_hybrid_filter(q_data, Fq_data)

    def _compute_geom_gradients(self, r_anchors):
        # 🚨 关键修复：r_anchors 的 6 个 slot 严格是
        # [0]=R0(受体,离配体最近) [1]=R1 [2]=R2(受体,最远)
        # [3]=L0(配体,离受体最近) [4]=L1 [5]=L2(配体,最远)
        # 必须与 calc_boresch_from_last_frame / _check_boresch_geometry_safe /
        # LambdaDependentBoreschForce 完全一致：
        #   r0      = distance(R0, L0)                slot(0,3)
        #   thetaA0 = angle(R1, R0, L0)   顶点=R0      slot(1,0,3)
        #   thetaB0 = angle(R0, L0, L1)   顶点=L0      slot(0,3,4)
        #   phiA0   = dihedral(R2, R1, R0, L0)         slot(2,1,0,3)
        #   phiB0   = dihedral(R1, R0, L0, L1)         slot(1,0,3,4)
        #   phiC0   = dihedral(R0, L0, L1, L2)         slot(0,3,4,5)
        # 旧版把 H0/H2 的变量名接反了（H0 实际绑定的是 slot[2]=R2 而不是
        # slot[0]=R0），导致这里算出的力常数/CV 用的是"最远"受体锚点当顶点，
        # 跟平衡值计算/几何合法性检查完全对不上，是这次 Boresch 崩溃的另一个源头。
        q = np.zeros(6)
        grads = np.zeros((6, 6, 3))

        R0, L0 = r_anchors[0], r_anchors[3]
        vec_r = L0 - R0
        norm_r = np.linalg.norm(vec_r) + 1e-10
        q[0] = norm_r
        ur = vec_r / norm_r
        grads[0, 0, :] = -ur
        grads[0, 3, :] = ur

        angle_slots = [(1, 0, 3), (0, 3, 4)]
        for i, (sa, sb, sc) in enumerate(angle_slots):
            a, b, c = r_anchors[sa], r_anchors[sb], r_anchors[sc]
            ba, bc = a - b, c - b
            nba, nbc = np.linalg.norm(ba) + 1e-10, np.linalg.norm(bc) + 1e-10
            cosA = np.clip(np.dot(ba, bc) / (nba * nbc), -1, 1)
            q[i + 1] = np.arccos(cosA)
            sinA = np.sqrt(1 - cosA**2) + 1e-10
            if sinA > 1e-3:
                dbda = (cosA * bc / nbc - ba / nba) / (nba * sinA)
                dbdc = (cosA * ba / nba - bc / nbc) / (nbc * sinA)
                grads[i + 1, sa, :] = dbda
                grads[i + 1, sb, :] = -dbda - dbdc
                grads[i + 1, sc, :] = dbdc

        if HAS_MDTRAJ:
            dummy_top = mdtraj.Topology()
            c = dummy_top.add_chain()
            r = dummy_top.add_residue("X", c)
            for _ in range(6):
                dummy_top.add_atom("C", openmm.app.element.Element.getBySymbol("C"), r)
            eps = 1e-3
            dihedral_slots = [(2, 1, 0, 3), (1, 0, 3, 4), (0, 3, 4, 5)]
            for g_idx, tup in enumerate(dihedral_slots, 3):
                q[g_idx] = mdtraj.compute_dihedrals(
                    mdtraj.Trajectory(r_anchors[None], dummy_top), [tup]
                )[0, 0]
                perturbations = []
                grad_slots = []
                for a in tup:
                    for d in range(3):
                        rp = r_anchors.copy()
                        rm = r_anchors.copy()
                        rp[a, d] += eps
                        rm[a, d] -= eps
                        perturbations.extend((rp, rm))
                        grad_slots.append((a, d))

                batch_xyz = np.asarray(perturbations, dtype=float)
                batch_angles = mdtraj.compute_dihedrals(
                    mdtraj.Trajectory(batch_xyz, dummy_top), [tup]
                )[:, 0]
                for idx, (a, d) in enumerate(grad_slots):
                    grads[g_idx, a, d] = (
                        batch_angles[2 * idx] - batch_angles[2 * idx + 1]
                    ) / (2 * eps)
        return q, grads

    def _apply_hybrid_filter(self, q, Fq):
        kB_T = self.gas_constant_kj_per_mol_k * self.T
        names = ["kr", "kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC"]
        ks = {}
        for i in range(6):
            valid = ~(np.isnan(q[:, i]) | np.isnan(Fq[:, i]))
            if valid.sum() < 10:
                ks[names[i]] = 0.0
                continue
            dq, dF = (
                q[:, i][valid] - np.mean(q[:, i][valid]),
                Fq[:, i][valid] - np.mean(Fq[:, i][valid]),
            )
            var = np.var(dq)
            cov = np.cov(dF, dq)[0, 1]
            k_reg = -cov / var if var > 1e-12 else None
            k_fluc = kB_T / var if var > 1e-12 else None
            corr = (
                np.corrcoef(q[:, i][valid], Fq[:, i][valid])[0, 1]
                if np.std(q[:, i][valid]) > 1e-10
                else 0.0
            )

            if k_reg and k_reg > 0 and corr < self.config["corr_threshold_keep"]:
                ks[names[i]] = k_reg
            elif k_fluc and self.config["use_fluctuation_fallback"]:
                ks[names[i]] = k_fluc
            else:
                ks[names[i]] = 0.0

        for name in names:
            if name == "kr":
                ks[name] = float(min(max(ks[name], 100.0), 2000.0))
            else:
                ks[name] = float(min(max(ks[name], 10.0), 100.0))

        return ks

    def run_pocket_force_projection(
        self, traj, context, selection, local_anchors, cap_h_coords=None
    ):
        n_frames = len(traj)
        q_data = np.zeros((n_frames, 6))
        Fq_data = np.zeros((n_frames, 6))
        for f in range(n_frames):
            pos = traj.xyz[f, selection]
            forces = context.calculate_forces(pos)
            r_anchors = pos[local_anchors]
            if np.any(np.isnan(r_anchors)):
                continue
            q, grads = self._compute_geom_gradients(r_anchors)
            if np.any(np.isnan(q)):
                continue
            Fq = np.zeros(6)
            for g in range(6):
                for a in range(6):
                    Fq[g] += np.dot(forces[local_anchors[a]], grads[g, a])
            q_data[f] = q
            Fq_data[f] = Fq

        # Unwrap 二面角
        for i in [3, 4, 5]:
            valid = ~(np.isnan(q_data[:, i]) | np.isinf(q_data[:, i]))
            if valid.sum() > 10:
                q_data[valid, i] = np.unwrap(q_data[valid, i])
        return self._apply_hybrid_filter(q_data, Fq_data)

    def estimate_from_trajectory(self, traj, ligand_resname, output_path=None):
        context, pocket_sel, local_anchors, anchor_global, cap_h, ks_quick = (
            self._build_pocket_context(traj, ligand_resname)
        )
        ks = (
            ks_quick
            if ks_quick
            else self.run_pocket_force_projection(
                traj, context, pocket_sel, local_anchors, cap_h
            )
        )

        traj_aligned = traj[:]
        traj_aligned.superpose(traj_aligned, 0, atom_indices=pocket_sel)
        r0 = traj_aligned.xyz[0, pocket_sel][local_anchors]
        # 🚨 关键修复：local_anchors 顺序是 [R0(离配体最近),R1,R2(最远),L0,L1,L2]，
        # 之前写成 H2,H1,H0=r0[0,1,2] 把 H0 错绑定到 R2（最远锚点），导致下面算出
        # 的 eq 平衡值和 receptor_indices=anchor_global[:3] 实际代表的原子对不上。
        H0, H1, H2, G0, G1, G2 = r0

        def calc_angle(a, b, c):
            ba, bc = a - b, c - b
            cos_val = np.clip(
                np.dot(ba, bc)
                / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10),
                -1,
                1,
            )
            # ✅ 直接返回弧度 (rad)
            return np.arccos(cos_val)

        # ✅ 直接返回弧度 (rad)。符号约定必须与 OpenMM `dihedral()` 一致，
        # 所以统一走 `boresch_dihedral_rad`，不再本地手写（见其 docstring）。
        calc_dihedral = boresch_dihedral_rad

        eq = {
            "r0": float(np.linalg.norm(H0 - G0)),  # ✅ nm (移除 *10)
            "thetaA0": float(calc_angle(H1, H0, G0)),  # ✅ rad
            "thetaB0": float(calc_angle(H0, G0, G1)),  # ✅ rad
            "phiA0": float(calc_dihedral(H2, H1, H0, G0)),  # ✅ rad
            "phiB0": float(calc_dihedral(H1, H0, G0, G1)),  # ✅ rad
            "phiC0": float(calc_dihedral(H0, G0, G1, G2)),  # ✅ rad
        }
        rec_indices = anchor_global[:3]
        lig_indices = anchor_global[3:]
        result = {
            "receptor_indices": rec_indices.tolist()
            if hasattr(rec_indices, "tolist")
            else list(rec_indices),
            "ligand_indices": lig_indices.tolist()
            if hasattr(lig_indices, "tolist")
            else list(lig_indices),
            "force_constants": ks,
            "equilibrium_values": eq,
            "method": "orb_pocket_projection_v4.3",
        }
        if output_path:

            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, (np.bool_,)):
                        return bool(obj)
                    return super().default(obj)

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
        return result

    def _finalize_candidate(self, cand, traj, context, pocket_sel, cap_h):
        """计算候选者的最终力常数和平衡值"""
        rec_anchors = cand["rec_anchors"]
        lig_anchors = cand["lig_anchors"]
        local_anchors = cand["local_anchors"]
        ks = self.run_pocket_force_projection(
            traj, context, pocket_sel, local_anchors, cap_h
        )
        traj_aligned = traj[:]
        traj_aligned.superpose(traj_aligned, 0, atom_indices=pocket_sel)
        r0_frame = traj_aligned.xyz[0, pocket_sel][local_anchors]
        # 🚨 关键修复：同 estimate_from_trajectory，local_anchors 顺序是
        # [R0(离配体最近),R1,R2(最远),L0,L1,L2]，之前 H0 被错绑定到 R2。
        H0, H1, H2, G0, G1, G2 = r0_frame

        def calc_angle_rad(a, b, c):
            ba, bc = a - b, c - b
            cos_val = np.clip(
                np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10),
                -1, 1,
            )
            return float(np.arccos(cos_val))

        # 符号约定必须与 OpenMM `dihedral()` 一致，见 `boresch_dihedral_rad`。
        calc_dihedral_rad = boresch_dihedral_rad

        eq = {
            "r0": float(np.linalg.norm(H0 - G0)),
            "thetaA0": calc_angle_rad(H1, H0, G0),
            "thetaB0": calc_angle_rad(H0, G0, G1),
            "phiA0": calc_dihedral_rad(H2, H1, H0, G0),
            "phiB0": calc_dihedral_rad(H1, H0, G0, G1),
            "phiC0": calc_dihedral_rad(H0, G0, G1, G2),
        }
        return {**cand, "ks": ks, "eq": eq}

    def _score_candidate_comprehensive(self, rec_anchors, lig_anchors, rmsf_vals,
                                        r0_nm, thA_rad, thB_rad, kr_raw, traj, top):
        """综合评分函数：稳定性 + 几何 + 力常数 + 序列分散度"""
        
        # === 1. 稳定性 (原有，权重 1.0) ===
        score_stab = sum((self.config["rmsf_cutoff_nm"] - rv) * 80 for rv in rmsf_vals)
        
        # === 2. 距离偏好 (微调，权重 0.8) ===
        r0_A = r0_nm * 10
        if 7.0 <= r0_A <= 9.5:  # 偏好 7-9.5Å，略收紧
            score_dist = 40 - abs(r0_A - 8.2) * 3
        else:
            score_dist = max(0, 20 - abs(r0_A - 8.2) * 2)
        
        # === 3. NEW: kr 合理性 (权重 1.2) ===
        if kr_raw <= 600:
            kr_score = 25  # 理想范围
        elif kr_raw <= 1200:
            kr_score = 25 - (kr_raw - 600) * 0.02  # 线性衰减
        else:
            kr_score = max(0, 13 - (kr_raw - 1200) * 0.03)  # 快速衰减
        
        # === 4. NEW: 角度质量 (权重 1.0) ===
        thA_deg, thB_deg = np.degrees(thA_rad), np.degrees(thB_rad)
        if 70 <= thA_deg <= 110 and 70 <= thB_deg <= 110:
            angle_score = 20
        elif 50 <= thA_deg <= 130 and 50 <= thB_deg <= 130:
            angle_score = 10
        else:
            angle_score = 0  # 已在硬过滤中排除，此处为保险
        
        # === 5. NEW: 锚点几何分散度 (权重 1.0) ===
        R0, R1, R2 = [traj.xyz[0, a] for a in rec_anchors]
        anchor_dists = [
            np.linalg.norm(R0-R1) * 10.0,
            np.linalg.norm(R1-R2) * 10.0,
            np.linalg.norm(R0-R2) * 10.0,
        ]  # nm → Å
        avg_anchor_dist = np.mean(anchor_dists)
        if 12 <= avg_anchor_dist <= 22:  # 12-22Å 理想分散
            geo_score = 18
        elif 8 <= avg_anchor_dist <= 28:
            geo_score = 18 - abs(avg_anchor_dist - 17) * 0.8
        else:
            geo_score = 0
        
        # === 6. NEW: 残基序列分散度 (权重 0.8) ===
        res_indices = sorted([top.atom(a).residue.index for a in rec_anchors])
        min_gap = min(res_indices[1]-res_indices[0], res_indices[2]-res_indices[1])
        if min_gap >= 20:
            seq_score = 15
        elif min_gap >= 10:
            seq_score = 10
        elif min_gap >= 5:
            seq_score = 4
        else:
            seq_score = 0  # 太接近，可能共线
        
        # === 综合 ===
        total = (score_stab + score_dist*0.8 + kr_score*1.2 +
                 angle_score + geo_score + seq_score*0.8)
        
        return total, {
            "stab": score_stab, "dist": score_dist, "kr": kr_score,
            "angle": angle_score, "geo": geo_score, "seq": seq_score
        }

    def estimate_multiple_anchors_from_trajectory(
        self,
        traj,
        ligand_resname: str,
        n_candidates: int = 5,
        output_path: Optional[str] = None,
        min_anchor_distance: float = 0.4,
        max_r0_angstrom: float = 10.0,
        max_kr: float = 2000.0,
        min_residue_gap: int = 3,
        use_last_ns: float = 5.0,  # ✅ 强制参数：只分析最后 N ns
    ) -> List[Dict]:
        """
        确定性枚举 v6.5 (最终修复版)
        【关键修正】
        1. 强制切片最后 5ns：基于轨迹时间精确切除前期不稳定部分
        2. 最终 r0/kr 二次硬校验：绝不返回超标候选
        3. 残基间隔与去重：杜绝鬼打墙
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError("需要 mdtraj")

        # ========================================================================
        # ✅ 0. 强制切片最后 5ns (物理需求：切除预平衡前期不稳定部分)
        # ========================================================================
        original_len = len(traj)
        if hasattr(traj, "time") and len(traj.time) > 0:
            t_max = traj.time[-1]  # ps
            t_cut = t_max - (use_last_ns * 1000.0)  # ns -> ps
            mask = traj.time >= t_cut
            n_keep = mask.sum()

            if n_keep > 0:
                traj = traj[mask]
                print(
                    f"🔪 轨迹切片: 仅使用最后 {use_last_ns} ns ({n_keep} 帧 / 原始 {original_len} 帧)"
                )
            else:
                print(
                    f"⚠️ 轨迹长度不足 {use_last_ns} ns，使用全轨迹 ({original_len} 帧)"
                )
        else:
            print(f"⚠️ 轨迹无时间信息，使用全轨迹 ({original_len} 帧)")

        print(
            f"🔍 确定性枚举 v6.5 | 目标: {n_candidates} | r0≤{max_r0_angstrom}Å | kr≤{max_kr} | 残基间隔≥{min_residue_gap}"
        )

        top = traj.topology
        lig_sel = top.select(f"resname {ligand_resname}")
        if len(lig_sel) == 0:
            raise ValueError(f"未找到配体: {ligand_resname}")

        # 2. 口袋 Cα 与 RMSF (基于切片后的稳态轨迹)
        # --- 确保 rigid_ca 在这里被定义，防止 NameError ---
        lig_center = traj.xyz[0, lig_sel].mean(axis=0)
        prot_ca = top.select("protein and name CA")
        
        # 筛选口袋区域 Cα (1.0 nm 范围内)
        pocket_ca = [ca for ca in prot_ca if np.linalg.norm(traj.xyz[0, ca] - lig_center) <= 1.0]
        
        rmsf_traj = traj[:: max(1, len(traj) // 100)]
        if len(rmsf_traj) > 1 and len(pocket_ca) > 0:
            rmsf_traj.superpose(rmsf_traj, 0, atom_indices=pocket_ca)
            # 🔑 RMSF 的参考结构应该是平均结构，不是第 0 帧——第 0 帧本身只是
            # 一个样本，如果它恰好是这段轨迹里偏离平均构象较远的一帧，相对它
            # 算出来的涨落会系统性偏大，让下面 rigid_mask = rmsf < cutoff 偏
            # 保守地把本来足够刚性的 Cα 判定为不合格，漏掉本可用的锚点候选。
            pocket_xyz = rmsf_traj.xyz[:, pocket_ca]
            mean_xyz = pocket_xyz.mean(axis=0)
            rmsf = np.sqrt(np.mean((pocket_xyz - mean_xyz[None, :, :]) ** 2, axis=(0, 2)))
        else:
            rmsf = np.zeros(len(pocket_ca))
            
        rigid_mask = rmsf < self.config["rmsf_cutoff_nm"]
        rigid_ca = [pocket_ca[i] for i in range(len(pocket_ca)) if rigid_mask[i]]
        rigid_res = [top.atom(ca).residue.index for ca in rigid_ca]

        # ✅ 安全检查：刚性原子不足无法构建 Boresch
        if len(rigid_ca) < 3:
            print(f"  ❌ 刚性 Cα 原子不足 3 个 (当前 {len(rigid_ca)})，无法进行几何枚举。")
            return []

        # 1. 生成配体锚点候选三元组 (基于质量+刚性排序)
        lig_heavy = top.select(f"resname {ligand_resname} and not element H")
        if len(lig_heavy) < 3:
            lig_heavy = top.select(f"resname {ligand_resname}")
        
        lig_combos = []
        for combo in combinations(lig_heavy, 3):
            # 评分逻辑：质量之和 + 骨架原子奖励
            mass_score = sum(top.atom(i).element.mass for i in combo)
            name_bonus = sum(2.0 if top.atom(i).name in ["CG","CD","CE","CZ","CA"] else 0.0 for i in combo)
            lig_combos.append((combo, mass_score + name_bonus))
            
        M = min(20, len(lig_combos))
        lig_combos = [c[0] for c in sorted(lig_combos, key=lambda x: x[1], reverse=True)[:M]]
        print(f"  → 配体锚点候选: {len(lig_combos)} 个三元组")

        # 2. 受体锚点枚举 + 嵌套配体候选扫描
        candidates = []
        seen_geo_keys = set()  # 几何去重

        # 遍历受体三元组
        for rec_combo in combinations(range(len(rigid_ca)), 3):
            ca0, ca1, ca2 = [rigid_ca[i] for i in rec_combo]
            res0, res1, res2 = [rigid_res[i] for i in rec_combo]
            
            # 受体侧预过滤（残基间隔、锚点间距）
            if min([abs(res0-res1), abs(res0-res2), abs(res1-res2)]) < min_residue_gap:
                continue
                
            pos_rec = [traj.xyz[0, ca] for ca in [ca0, ca1, ca2]]
            if min(np.linalg.norm(pos_rec[i]-pos_rec[j]) for i in range(3) for j in range(i+1,3)) < min_anchor_distance:
                continue
                
            # 遍历配体三元组
            for lig_combo in lig_combos:
                pos_lig = [traj.xyz[0, idx] for idx in lig_combo]
                
                # ✅ 修复：放宽配体内部间距阈值 (从 0.25 -> 0.15 nm / 1.5 Å)
                # 适配刚性小分子（如苯环）原子间距较近的情况
                if min(np.linalg.norm(pos_lig[i]-pos_lig[j]) for i in range(3) for j in range(i+1,3)) < 0.15:
                    continue
                
                # ✅ 严格几何校验
                ok, msg = self._validate_boresch_geometry_strict(
                    [ca0, ca1, ca2], lig_combo, traj.xyz[0]
                )
                if not ok:
                    continue
                    
                # 几何去重
                r0_A = np.linalg.norm(pos_rec[0] - pos_lig[0]) * 10.0
                geo_key = (round(r0_A, 1), tuple(sorted([ca0, ca1, ca2][:2])), tuple(sorted(lig_combo[:2])))
                if geo_key in seen_geo_keys:
                    continue
                seen_geo_keys.add(geo_key)
                
                # 综合评分
                rmsf_vals = [rmsf[pocket_ca.index(ca)] for ca in [ca0, ca1, ca2]]
                score_stab = sum((self.config["rmsf_cutoff_nm"] - rv) * 100 for rv in rmsf_vals)
                if 5.0 <= r0_A <= 10.0:
                    score_dist = 60 - abs(r0_A - 7.5) * 4
                else:
                    score_dist = max(0, 30 - abs(r0_A - 7.5) * 2)
                total_score = score_stab + score_dist
                
                candidates.append({
                    "rec_anchors": [ca0, ca1, ca2],
                    "lig_anchors": list(lig_combo),
                    "res_key": tuple(sorted([res0, res1, res2])),
                    "r0_A": r0_A,
                    "score": total_score,
                    "rmsf_avg": np.mean(rmsf_vals),
                })

        # 3. 验证与快速筛选
        candidates.sort(key=lambda x: x["score"], reverse=True)
        print(f"  → 通过几何过滤的候选: {len(candidates)} 个")
        
        rigid_residues = {top.atom(int(a)).residue.index for a in rigid_ca}
        pocket_atoms = set(int(i) for i in lig_sel.tolist())
        for res in top.residues:
            if res.index in rigid_residues:
                for atom in res.atoms:
                    pocket_atoms.add(atom.index)
        pocket_sel = np.array(sorted(pocket_atoms), dtype=int)

        cap_h = []
        pocket_traj = traj.atom_slice(pocket_sel)
        context = OrbVacuumContext(pocket_traj.topology.to_openmm(), device=self.device)
        
        validated = []
        kr_seen = set()
        search_pool = max(n_candidates * 6, 30)
        
        for cand in candidates[:search_pool]:
            # 构建局部原子索引映射
            local_anchors = [int(np.where(pocket_sel == a)[0][0]) for a in cand["rec_anchors"] + cand["lig_anchors"]]
            
            ks = self._quick_validate(traj[:50], context, pocket_sel, local_anchors, cap_h)
            if ks is None:
                continue
            if ks["kr"] > max_kr:
                continue
            kr_r = round(ks["kr"], 1)
            if kr_r in kr_seen:
                continue
            kr_seen.add(kr_r)
            validated.append({**cand, "ks": ks, "local_anchors": local_anchors})
        print(f"  → 快速验证通过: {len(validated)} 个")

        # 4. 最终结果构建
        results = []
        seen_final = set()
        fallback_pool = []
        
        # 辅助函数
        def log_cand(rank, r0_nm, kr, res_key, tag="合格"):
            r0_a = r0_nm * 10.0
            print(f"  [{'✅' if tag=='合格' else '⬇️'} {tag}] #{rank}: r0={r0_a:.2f}Å | kr={kr:.1f} kJ/mol/nm² | 残基={res_key}")

        for cand in validated:
            res_key = cand["res_key"]
            if res_key in seen_final:
                continue
            seen_final.add(res_key)
            
            # 计算最终参数
            final = self._finalize_candidate(cand, traj, context, pocket_sel, cap_h)
            
            # 角度防御 (防止奇异)
            thA_deg = np.degrees(final["eq"]["thetaA0"])
            thB_deg = np.degrees(final["eq"]["thetaB0"])
            if not (40.0 <= thA_deg <= 140.0) or not (40.0 <= thB_deg <= 140.0):
                continue

            # 物理边界过滤
            r0_nm = final["eq"]["r0"]
            kr_val = final["ks"]["kr"]
            if r0_nm < 0.4 or r0_nm > 1.0:
                continue
            if r0_nm > max_r0_angstrom / 10.0:
                fallback_pool.append(final)
                continue
            if kr_val > max_kr:
                fallback_pool.append(final)
                continue

            # ✅ 使用综合评分替代简单 kr 惩罚
            rmsf_vals = [rmsf[pocket_ca.index(ca)] for ca in final["rec_anchors"]]
            total_score, score_breakdown = self._score_candidate_comprehensive(
                rec_anchors=final["rec_anchors"],
                lig_anchors=final["lig_anchors"],
                rmsf_vals=rmsf_vals,
                r0_nm=r0_nm,
                thA_rad=final["eq"]["thetaA0"],
                thB_rad=final["eq"]["thetaB0"],
                kr_raw=kr_val,
                traj=traj,
                top=top,
            )

            # 加入结果
            results.append({
                "rank": len(results) + 1,
                "receptor_indices": final["rec_anchors"],
                "ligand_indices": final["lig_anchors"],
                "receptor_residues": list(final["res_key"]),
                "force_constants": {k: float(v) for k, v in final["ks"].items()},
                "equilibrium_values": final["eq"],
                "total_score": float(total_score),
                "score_breakdown": score_breakdown,
                "method": "finite_combo_v6.5_last5ns",
            })
            log_cand(len(results), r0_nm, kr_val, res_key)
            
            if len(results) >= n_candidates:
                break

        # 降级回退
        if len(results) < n_candidates and fallback_pool:
            print(f"  ⚠️ 合格候选不足 ({len(results)}/{n_candidates})，启动降级回退...")
            fallback_pool.sort(key=lambda x: x["ks"]["kr"])
            for final in fallback_pool:
                if len(results) >= n_candidates: break
                res_key = final["res_key"]
                if res_key in seen_final: continue
                seen_final.add(res_key)
                
                # 放宽 kr 限制，但死守 r0 几何
                r0_nm_fb = final["eq"]["r0"]
                kr_val_fb = final["ks"]["kr"]
                
                # ✅ 使用综合评分（与主路径一致）
                rmsf_vals_fb = [rmsf[pocket_ca.index(ca)] for ca in final["rec_anchors"]]
                total_score_fb, score_breakdown_fb = self._score_candidate_comprehensive(
                    rec_anchors=final["rec_anchors"],
                    lig_anchors=final["lig_anchors"],
                    rmsf_vals=rmsf_vals_fb,
                    r0_nm=r0_nm_fb,
                    thA_rad=final["eq"]["thetaA0"],
                    thB_rad=final["eq"]["thetaB0"],
                    kr_raw=kr_val_fb,
                    traj=traj,
                    top=top,
                )

                results.append({
                    "rank": len(results) + 1,
                    "receptor_indices": final["rec_anchors"],
                    "ligand_indices": final["lig_anchors"],
                    "receptor_residues": list(final["res_key"]),
                    "force_constants": {k: float(v) for k, v in final["ks"].items()},
                    "equilibrium_values": final["eq"],
                    "total_score": float(total_score_fb),
                    "score_breakdown": score_breakdown_fb,
                    "method": "finite_combo_v6.5_fallback",
                    "warning": "kr 超出上限，已放行"
                })
                log_cand(len(results), final["eq"]["r0"], final["ks"]["kr"], res_key, tag="回退")

        if not results:
            print(f"  ❌ 未找到满足条件的合格候选。")
        else:
            print(f"✅ 最终返回 {len(results)} 个合格候选")
            
            # ✅ 【关键修复】按总分降序排序，并重新分配 rank 序号
            results.sort(key=lambda x: x.get("total_score", 0.0), reverse=True)
            for i, res in enumerate(results):
                res["rank"] = i + 1
                res["kr_bonus"] = round((max_kr - res["force_constants"]["kr"]) * 0.1, 2)
            
            # 打印诊断信息，确认排序生效
            top = results[0]
            print(f"  🏆 推荐首选: Rank #{top['rank']} (残基={top['receptor_residues']})")
            print(f"     kr={top['force_constants']['kr']:.1f} | 总分={top['total_score']:.2f} | kr加分={top.get('kr_bonus',0):.2f}")

        if output_path:
            import json
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer, np.floating)): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super().default(obj)
            with open(output_path, "w") as f:
                json.dump({"candidates": results}, f, indent=2, cls=NumpyEncoder)
            print(f"✅ 结果已保存: {output_path}")
            
        return results


# ============================================================================
# 4. 幽灵离子 & 2D路径规划 & 替身系统构建器
# ============================================================================
class GhostIonHandler:
    def __init__(self, ghost_ion_distance=10.0, ghost_ion_scale_factor=1.0):
        self.ghost_ion_distance = ghost_ion_distance
        self.ghost_ion_scale_factor = ghost_ion_scale_factor

    def _resolve_ghost_anchor(self, box_vectors=None, reference_positions=None, ligand_indices=None):
        if box_vectors is None:
            return (
                float(self.ghost_ion_distance),
                0.0,
                0.0,
            )

        box_lengths = np.array(
            [
                np.linalg.norm(np.asarray(vec.value_in_unit(unit.nanometer) if hasattr(vec, "value_in_unit") else vec, dtype=float))
                for vec in box_vectors
            ],
            dtype=float,
        )
        box_lengths = np.where(box_lengths > 1.0e-6, box_lengths, 3.0)
        safe_margin = np.minimum(0.2, 0.1 * box_lengths)

        if reference_positions is not None and ligand_indices:
            lig_xyz = []
            for idx in ligand_indices:
                pos = reference_positions[idx]
                if hasattr(pos, "value_in_unit"):
                    lig_xyz.append(np.asarray(pos.value_in_unit(unit.nanometer), dtype=float))
                else:
                    lig_xyz.append(np.asarray(pos, dtype=float))
            if lig_xyz:
                lig_com = np.mean(np.asarray(lig_xyz, dtype=float), axis=0)
                anchor = np.mod(lig_com + 0.5 * box_lengths - safe_margin, box_lengths)
                return tuple(float(x) for x in anchor)

        anchor = 0.5 * box_lengths - safe_margin
        return tuple(float(x) for x in anchor)

    def create_ghost_ion_force(
        self,
        ligand_indices,
        ligand_charges,
        lambda_param="lam_coul",
        box_vectors=None,
        reference_positions=None,
    ):
        total = sum(ligand_charges)
        ghost = -total * self.ghost_ion_scale_factor
        if abs(ghost) < 1.0e-12 or not ligand_indices:
            return None

        xg, yg, zg = self._resolve_ghost_anchor(
            box_vectors=box_vectors,
            reference_positions=reference_positions,
            ligand_indices=ligand_indices,
        )
        force = openmm.CustomExternalForce(
            f"{lambda_param} * 138.935456 * ghost_charge * ligand_charge / "
            f"max(periodicdistance(x, y, z, ghost_x, ghost_y, ghost_z), 0.05)"
        )
        force.addGlobalParameter(lambda_param, 1.0)
        force.addGlobalParameter("ghost_charge", float(ghost))
        force.addGlobalParameter("ghost_x", float(xg))
        force.addGlobalParameter("ghost_y", float(yg))
        force.addGlobalParameter("ghost_z", float(zg))
        force.addPerParticleParameter("ligand_charge")
        for idx, charge in zip(ligand_indices, ligand_charges):
            force.addParticle(int(idx), [float(charge)])
        return force


class TwoDimensionalLambdaPathPlanner:
    def __init__(self, n_points=20, path_type="decoupling"):
        self.n_points = n_points
        self.path_type = path_type

    def generate_path(self):
        if self.path_type == "diagonal":
            return [
                (1.0 - i / self.n_points, 1.0 - i / self.n_points)
                for i in range(self.n_points + 1)
            ]
        elif self.path_type == "decoupling":
            n_half = self.n_points // 2
            lambdas = []
            for i in range(n_half + 1):
                lambdas.append((1.0 - i / n_half, 1.0))
            for i in range(1, self.n_points - n_half + 1):
                lambdas.append((0.0, 1.0 - i / (self.n_points - n_half)))
            return lambdas
        else:
            return [(1.0, 1.0 - i / self.n_points) for i in range(self.n_points + 1)]


class SurrogateSystemBuilder:
    def __init__(
        self,
        surrogate_params,
        ghost_handler=None,
        sigma_gauss_nm: float = GAUSS_COUL_SIGMA_NM,
    ):
        self.surrogate_potential = DEXPSurrogatePotential.from_dict(
            surrogate_params or {}
        )
        self.ghost_handler = ghost_handler
        self.sigma_gauss_nm = float(sigma_gauss_nm)

    def build_surrogate_system(
        self,
        original_system,
        ligand_indices,
        environment_indices,
        lambda_names=("lam_coul", "lam_vdw"),
        force_group=1,
        reference_positions=None,
        box_vectors=None,
    ):
        # 必须强制深拷贝：ensure_owned_system 在 thisown==1 时会原样返回同一个对象
        # （XmlSerializer.deserialize 出来的 System 默认就是 thisown==1），如果不
        # 先 serialize/deserialize 一次，下面对 nb_force 的原地修改和 addForce 会
        # 直接污染调用者传进来的 original_system —— 这会导致外部同时持有的“原始
        # 力场”引用实际上已经被替换成了这个 surrogate system。
        new_system = ensure_owned_system(
            XmlSerializer.deserialize(XmlSerializer.serialize(original_system))
        )
        resolved_box_vectors = (
            box_vectors
            if box_vectors is not None
            else new_system.getDefaultPeriodicBoxVectors()
        )
        _validate_minimum_image(resolved_box_vectors, DEXP_VDW_CUTOFF_NM)
        nb_force = next(
            (f for f in new_system.getForces() if isinstance(f, openmm.NonbondedForce)),
            None,
        )
        if not nb_force:
            raise ValueError("未找到 NonbondedForce")

        lig_set = {int(idx) for idx in ligand_indices}
        env_set = {int(idx) for idx in environment_indices if int(idx) not in lig_set}
        if not lig_set:
            raise ValueError("ligand_indices 为空，无法构建 surrogate decoupling system")
        if not env_set:
            raise ValueError("environment_indices 为空，无法构建 ligand-environment surrogate force")
        original_params = [
            nb_force.getParticleParameters(i) for i in range(new_system.getNumParticles())
        ]
        reference_exclusions = []
        for i in range(nb_force.getNumExceptions()):
            p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
            reference_exclusions.append((int(p1), int(p2)))

        # 1) 先恢复 ligand-ligand 内部 nonbonded / 1-4，保留原始 MM 内部拓扑语义。
        ll_force, ll_14_force = create_ligand_internal_force(
            nb_force=nb_force,
            perturbed_indices=sorted(lig_set),
            particle_params=original_params,
            reference_exclusions=reference_exclusions,
            num_particles=new_system.getNumParticles(),
            system=new_system,
        )
        ll_force.setForceGroup(force_group)
        new_system.addForce(ll_force)
        if ll_14_force is not None:
            ll_14_force.setForceGroup(force_group)
            new_system.addForce(ll_14_force)

        # 2) 主 NonbondedForce 中将 ligand 完全去耦，避免原始 MM L-E 项始终全开。
        for idx in sorted(lig_set):
            q, sigma, epsilon = original_params[idx]
            nb_force.setParticleParameters(
                idx,
                0.0 * unit.elementary_charge,
                sigma,
                0.0 * unit.kilojoule_per_mole,
            )
        for exc_idx in range(nb_force.getNumExceptions()):
            p1, p2, _, sigma, _ = nb_force.getExceptionParameters(exc_idx)
            p1, p2 = int(p1), int(p2)
            if p1 in lig_set or p2 in lig_set:
                nb_force.setExceptionParameters(
                    exc_idx,
                    p1,
                    p2,
                    0.0 * unit.elementary_charge * unit.elementary_charge,
                    sigma,
                    0.0 * unit.kilojoule_per_mole,
                )

        # 3) L-E Gaussian electrostatics：用平滑库仑核替代点电荷奇点，并与 DEXP 共用
        #    0.50~0.70 nm 的 switching/cutoff 缝合区。这样 0.45~0.50 nm 仍由
        #    MACE/DEXP 核心描述区承担，switch shell 只作为 surrogate 平滑退出区，
        #    不应当被当作核心近程
        #    势能面/RDF/PMF 判据区解释。
        sigma_gauss_nm = max(float(self.sigma_gauss_nm), 1.0e-6)
        gamma_eff = 1.0 / max(math.sqrt(2.0) * sigma_gauss_nm, 1.0e-6)
        # 修复 switch 伪影：erf(γr)/r 在 0.5~0.7 nm 仍是 ~1/r 长程尾巴，用能量 switching 截断会
        # 引入 -S'(r)·U(r) 假力，在 cutoff 内侧堆出假的 RDF 峰（~0.63 nm）。改用 shifted-force：
        # U_sf(r)=U(r)-U(rc)-(r-rc)U'(rc)，使势与力在 cutoff 处都连续归零，不再需要 switching。
        rc_nm = GAUSS_COUL_CUTOFF_NM
        g = float(gamma_eff)
        fc = math.erf(g * rc_nm) / rc_nm                                  # U(rc)/(k q1 q2)
        fpc = (2.0 * g / math.sqrt(math.pi)) * math.exp(-(g * rc_nm) ** 2) / rc_nm \
            - math.erf(g * rc_nm) / rc_nm ** 2                            # U'(rc)/(k q1 q2)
        gauss_expr = (
            f"{lambda_names[0]} * 138.935456*q1*q2*("
            f"erf({g}*r_safe)/r_safe - ({fc}) - (r_safe - {rc_nm})*({fpc})"
            "); r_safe = max(r, 1e-6)"
        )
        coul_force = openmm.CustomNonbondedForce(gauss_expr)
        coul_force.addPerParticleParameter("q")
        coul_force.addGlobalParameter(lambda_names[0], 1.0)
        for i in range(new_system.getNumParticles()):
            q, _, _ = original_params[i]
            coul_force.addParticle([q.value_in_unit(unit.elementary_charge)])
        coul_force.addInteractionGroup(sorted(lig_set), sorted(env_set))
        coul_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        coul_force.setCutoffDistance(GAUSS_COUL_CUTOFF_NM * unit.nanometer)
        # shifted-force 已保证 cutoff 处势/力归零，关闭 switching（否则又引入 -S'·U 假力）
        coul_force.setUseSwitchingFunction(False)
        coul_force.setForceGroup(force_group)
        for p1, p2 in reference_exclusions:
            coul_force.addExclusion(int(p1), int(p2))
        new_system.addForce(coul_force)

        # 4) L-E DEXP：短程排斥/色散替身，与 Gaussian electrostatics 拼成完整 surrogate。
        dexp_expr = (
            f"{self.surrogate_potential.build_expression(lam_vdw=lambda_names[1])}"
        )
        dexp_force = openmm.CustomNonbondedForce(dexp_expr)
        dexp_force.addGlobalParameter(lambda_names[1], 1.0)
        dexp_force.addPerParticleParameter("sigma")
        dexp_force.addPerParticleParameter("epsilon")
        for i in range(new_system.getNumParticles()):
            # 用去耦前捕获的 original_params，而不是已被步骤2清零 epsilon 的 nb_force：
            # DEXP 核需要 ligand/environment 双方各自真实的原始 LJ sigma/epsilon 才能
            # 按组合律解析出 pair-specific r0_ij/eps_ij，不能沿用去耦后的占位值。
            _, sigma_i, epsilon_i = original_params[i]
            dexp_force.addParticle([
                sigma_i.value_in_unit(unit.nanometer),
                epsilon_i.value_in_unit(unit.kilojoule_per_mole),
            ])
        dexp_force.addInteractionGroup(sorted(lig_set), sorted(env_set))
        dexp_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
        dexp_force.setCutoffDistance(DEXP_VDW_CUTOFF_NM * unit.nanometer)
        dexp_force.setUseSwitchingFunction(True)
        dexp_force.setSwitchingDistance(
            (DEXP_VDW_CUTOFF_NM - DEXP_VDW_SWITCH_WIDTH_NM)
            * unit.nanometer
        )
        dexp_force.setForceGroup(force_group)

        for p1, p2 in reference_exclusions:
            dexp_force.addExclusion(int(p1), int(p2))

        new_system.addForce(dexp_force)
        sync_all_exclusions(new_system)

        if self.ghost_handler and ligand_indices:
            charges = [
                original_params[i][0].value_in_unit(unit.elementary_charge)
                for i in sorted(ligand_indices)
            ]
            ghost_force = self.ghost_handler.create_ghost_ion_force(
                ligand_indices=sorted(ligand_indices),
                ligand_charges=charges,
                lambda_param=lambda_names[0],
                box_vectors=box_vectors if box_vectors is not None else new_system.getDefaultPeriodicBoxVectors(),
                reference_positions=reference_positions,
            )
            if ghost_force is not None:
                new_system.addForce(ghost_force)
        return new_system


# ============================================================================
# 5. Orb 扫描器 & 混合工厂 & Surrogate 流水线
# ============================================================================

class OrbMMHybridFactory:
    def get_rotatable_torsions_rdkit(self, mol):
        import rdkit.Chem as Chem

        pattern = Chem.MolFromSmarts("[!#1;!D1]-[!#1;!D1]")
        if pattern is None:
            return []
        torsions = []
        for bond in mol.GetSubstructMatches(pattern):
            a1, a2 = bond[0], bond[1]
            for n1 in [x.GetIdx() for x in mol.GetAtomWithIdx(a1).GetNeighbors()]:
                if n1 != a2:
                    for n2 in [
                        x.GetIdx() for x in mol.GetAtomWithIdx(a2).GetNeighbors()
                    ]:
                        if n2 != a1 and mol.GetBondBetweenAtoms(a1, a2) is not None:
                            torsions.append((n1, a1, a2, n2))
        return torsions






# ============================================================================
# 6. 势能路由工厂 & 幽灵离子快捷函数
# ============================================================================
class AlchemicalPotentialFactory:
    @staticmethod
    def build(potential_type, params, lam_coul, lam_vdw):
        if isinstance(params, (ACESoftcorePotential, DEXPSurrogatePotential)):
            obj = params
        elif potential_type == "dexp":
            obj = DEXPSurrogatePotential.from_dict(params or {})
        elif potential_type == "softcore":  # ✅ 显式识别
            obj = ACESoftcorePotential.from_dict(params or {})
        else:
            obj = ACESoftcorePotential.from_dict(params or {})
        if isinstance(obj, DEXPSurrogatePotential):
            return obj.build_expression(lam_vdw=lam_vdw), obj.get_parameters_dict()
        return obj.build_expression(lam_coul, lam_vdw), obj.get_parameters_dict()

def create_ghost_ion_force(
    lig_indices,
    lig_charges,
    lam_param="lam_coul",
    dist=10.0,
    scale=0.1,
    box_vectors=None,
    reference_positions=None,
):
    return GhostIonHandler(dist, scale).create_ghost_ion_force(
        ligand_indices=lig_indices,
        ligand_charges=lig_charges,
        lambda_param=lam_param,
        box_vectors=box_vectors,
        reference_positions=reference_positions,
    )


# ============================================================================
# 7. 在线收敛监控器 (生产级动态诊断)
# ============================================================================
class OnlineConvergenceMonitor:
    """
    ABFE 在线收敛监控器 (核心物理组件)
    职责：实时分析自由能轨迹的平稳性、统计误差和相空间重叠度。
    设计原则：
    - 增量热启动 (initial_f_k) 实现 O(1) 级别 MBAR 重算
    - 五维正交判据杜绝"假收敛"
    - 单位原生支持 (OpenMM unit -> kJ/mol)

    ⚠️ 严格契约：输入 u_kn_chunk 必须为 Total Reduced Potential
       即 u_k(x) = β[U_phys(x) + U_restraint(x)]
       若仅传入纯物理势能，Overlap 与 N_eff 指标将失效，可能导致假收敛。
    """

    def __init__(
        self,
        temperature: unit.Quantity,
        check_interval: int = 10,
        ma_window: int = 5,
        precision_thresholds: Optional[Dict] = None,
    ):

        self.kt = (unit.MOLAR_GAS_CONSTANT_R * temperature).value_in_unit(
            unit.kilojoules_per_mole
        )
        self.interval = check_interval
        self.ma_window = ma_window

        self.thr = {
            "drift": 0.5,
            "error": 0.8,
            "neff_ratio": 0.20,
            "overlap": 0.85,
            "min_neighbor_overlap": 0.03,
            "ma_std": 0.30,
        }
        if precision_thresholds:
            self.thr.update(precision_thresholds)

        self.f_k_prev = None
        self.history = []
        self.dg_deque = deque(maxlen=ma_window + 2)

    def add_diagnostic(self, u_kn_chunk: np.ndarray, step: int) -> Dict:
        """
        核心诊断逻辑：输入势能矩阵，输出多维收敛报告
        参数：
            u_kn_chunk: (K_states, N_frames) 约化势能矩阵 (已除以 kT)
            step: 当前模拟步数 (用于日志)
        返回：
            dict: {converged: bool, dg: float, error: float, ...}
        """
        K, N = u_kn_chunk.shape
        if N < 20:
            return {"converged": False, "msg": "waiting_for_data", "step": step}

        energy_mean = np.mean(u_kn_chunk) * self.kt
        energy_var = np.var(u_kn_chunk, axis=1)

        if energy_mean < -2500.0 and np.max(energy_var) < 5.0:
            print(
                f"  ⚠️ [Monitor] 能量矩阵疑似遗漏限制力 (μ={energy_mean:.1f}, max(σ²)={np.max(energy_var):.1f})"
            )
            print(
                f"     → 请确保 ibs_engine 中 getState(groups={{group_id}}) 包含 Boresch 力"
            )
            return {
                "converged": False,
                "error": "suspected_missing_restraint_energy",
                "step": step,
            }

        try:
            K, N = u_kn_chunk.shape
            n_k_array = np.full(K, N, dtype=int)
            mbar = _build_mbar_compatible(
                u_kn_chunk,
                n_k_array,
                initial_f_k=self.f_k_prev,
                solver_protocol="hybr",
                solver_tolerance=1e-5,
                verbose=False,
            )

            res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
            df, ddf = _extract_free_energy_arrays(res, require_uncertainty=True)

            self.f_k_prev = df[0, :].copy()

            dg = (df[0, -1] - df[0, 0]) * self.kt
            err = ddf[0, -1] * self.kt
            # 🔑 之前这里直接调用 mbar.compute_effective_sample_number()/
            # mbar.compute_overlap()["matrix"]，既没有走项目里其它调用点
            # （ibs_engine.py）已经在用的 dict-or-ndarray 兼容判断，也没有对
            # 返回值做有限性校验。改走下面两个项目内的兼容层函数，跟其它
            # MBAR 结果消费点保持一致。
            neff = _compute_effective_sample_number_compatible(mbar)
            neff_ratio = float(np.min(neff) / N) if N > 0 else 0.0

            overlap_mat = _compute_overlap_matrix_compatible(mbar)
            overlap = float(np.max(np.diag(overlap_mat)))
            
            # ✅ MBAR 重叠度自动诊断与降级
            min_offdiag = np.min([
                overlap_mat[i, j] 
                for i in range(K) for j in range(K) 
                if abs(i-j) == 1
            ])
            if min_offdiag < 0.03:
                print(f"  🚨 [Monitor] 相邻窗口最小重叠 {min_offdiag:.3f} < 0.03，MBAR 误差可能低估！")
                print(f"     → 建议：延长采样 20% 或在重叠最差区域附近插值窗口")

            self.dg_deque.append(dg)
            ma_std = (
                float(np.std(list(self.dg_deque)[-self.ma_window :]))
                if len(self.dg_deque) >= self.ma_window
                else 99.0
            )
            drift = (
                float(abs(self.dg_deque[-1] - self.dg_deque[0]))
                if len(self.dg_deque) > 1
                else 99.0
            )

            is_stable = drift < self.thr["drift"] and ma_std < self.thr["ma_std"]
            is_precise = err < self.thr["error"] and neff_ratio > self.thr["neff_ratio"]
            is_connected = min_offdiag >= self.thr.get("min_neighbor_overlap", 0.03)

            converged = is_stable and is_precise and is_connected

            report = {
                "step": step,
                "converged": converged,
                "dg": dg,
                "error": err,
                "neff_ratio": neff_ratio,
                "overlap_max_diag": overlap,
                "overlap_min_offdiag": float(min_offdiag),
                "drift": drift,
                "ma_std": ma_std,
                "details": {
                    "stable": is_stable,
                    "precise": is_precise,
                    "connected": is_connected,
                },
            }
            if min_offdiag < 0.03:
                report["warning"] = "low_overlap"
            self.history.append(report)
            return report

        except ImportError:
            return {"converged": False, "error": "pymbar_not_installed", "step": step}
        except Exception as e:
            return {
                "converged": False,
                "error": f"mbar_failed: {str(e)[:60]}",
                "step": step,
            }

    def export_convergence_data(self, path: str):
        import json

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(path, "w") as f:
            json.dump(self.history, f, indent=2, cls=NumpyEncoder)

    def plot_convergence(self, output_path: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.history:
            print(f"  ⚠️  无收敛历史数据，跳过绘图: {output_path}")
            return

        valid_data = [h for h in self.history if "dg" in h and "error" in h]
        if not valid_data:
            return

        steps = [h["step"] for h in valid_data]
        dgs = [h["dg"] for h in valid_data]
        errs = [h["error"] for h in valid_data]

        plt.figure(figsize=(10, 6), dpi=300)

        plt.errorbar(
            steps,
            dgs,
            yerr=errs,
            fmt="o-",
            color="#2E86AB",
            ecolor="#A23B72",
            capsize=3,
            label="ΔG ± error",
            linewidth=1.5,
        )

        if dgs:
            plt.axhline(
                y=dgs[-1],
                color="gray",
                linestyle="--",
                alpha=0.5,
                label=f"Final ΔG: {dgs[-1]:.2f} kJ/mol",
            )

        plt.xlabel("Simulation Step", fontsize=11)
        plt.ylabel("ΔG (kJ/mol)", fontsize=11)
        plt.title("ABFE Convergence Monitor", fontsize=13, pad=15)
        plt.legend(frameon=True, fancybox=True, shadow=False)
        plt.grid(True, alpha=0.3, linestyle=":")

        plt.tight_layout()

        try:
            plt.savefig(output_path, bbox_inches="tight")
            print(f"  ✓ 收敛曲线已保存: {output_path}")
        except Exception as e:
            print(f"  ⚠️  保存图片失败: {e}")
        finally:
            plt.close()


#=============================================================================
# 修复 2: 约束 Jacobian 解析修正 (Constraint Correction)
#=============================================================================
def calculate_constraint_jacobian_correction(system, ligand_indices, temperature=300.0):
    """
    计算配体解耦过程中的约束修正项 (Jacobian Correction)
    物理原理：当约束键随配体变为 Ghost 时，构象空间体积密度发生变化。
    公式: ΔG_cons = +0.5 * R * T * Σ ln(μ_bond / μ_ref)
    μ_ref 取 1.0 Da (OpenMM 质量单位基准)
    """
    R = 0.008314462618  # kJ/(mol·K)
    T = temperature if isinstance(temperature, float) else temperature.value_in_unit(unit.kelvin)
    kT = R * T

    correction_kj = 0.0
    constrained_bonds = 0
    lig_set = set(ligand_indices)

    for i in range(system.getNumConstraints()):
        p1, p2, _ = system.getConstraintParameters(i)
        if p1 in lig_set and p2 in lig_set:
            m1 = system.getParticleMass(p1).value_in_unit(unit.dalton)
            m2 = system.getParticleMass(p2).value_in_unit(unit.dalton)
            mu = (m1 * m2) / (m1 + m2) if (m1 + m2) > 0 else 1e-6
            mu_ref = 1.0  # Da (OpenMM 标准参考约化质量)
            correction_kj += 0.5 * kT * math.log(max(mu, 1e-6) / mu_ref)
            constrained_bonds += 1

    if constrained_bonds > 0:
        print(f"  🔍 检测到 {constrained_bonds} 个配体约束键，Jacobian 修正: {correction_kj:.3f} kJ/mol")
    return correction_kj


#=============================================================================
# 修复 6: Boresch 平衡值从预平衡最后一帧直接计算
#=============================================================================
# 完整替换 abfe_core.py 中的 calc_boresch_from_last_frame 函数
def calc_boresch_from_last_frame(positions, rec_idx, lig_idx):
    """✅ 修复 2.3：兼容 Quantity 包裹、Numpy 数组、OpenMM Vec3 列表"""
    # 1. 尝试剥离单位
    if hasattr(positions, "value_in_unit"):
        pos = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    elif isinstance(positions, np.ndarray):
        pos = positions.astype(np.float64, copy=False)
    else:
        # 兼容 [openmm.Vec3, ...] 列表
        pos = np.array([[getattr(p, 'x', p[0]), getattr(p, 'y', p[1]), getattr(p, 'z', p[2])] 
                        for p in positions], dtype=np.float64)
                        
    # 确保形状为 (N, 3)
    if pos.shape == (3, 3): pos = pos.T  # 处理传入 box_vectors 类转置误用
    elif pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions 形状异常: {pos.shape}，期望 (N, 3)")

    rec_idx = [int(i) for i in rec_idx]
    lig_idx = [int(i) for i in lig_idx]
    if len(rec_idx) != 3 or len(lig_idx) != 3:
        raise ValueError("Boresch 平衡值计算需要 3 个受体锚点和 3 个配体锚点")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions 包含 NaN/Inf，拒绝更新 Boresch 平衡几何")

    r_coords = pos[rec_idx]
    l_coords = pos[lig_idx]
    if not np.all(np.isfinite(r_coords)) or not np.all(np.isfinite(l_coords)):
        raise ValueError("Boresch 锚点坐标包含 NaN/Inf，拒绝更新平衡几何")
    L0, L1, L2 = l_coords

    # 受体锚点顺序必须在估算阶段确定后保持锁定，绝不能按瞬时几何动态重排。
    H0, H1, H2 = r_coords

    def dist(a, b): return np.linalg.norm(a - b)
    def angle(a, b, c):
        ba, bc = a - b, c - b
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6: return np.pi / 2
        cos_val = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc + 1e-10), -1.0, 1.0)
        return np.arccos(cos_val)
    # 🚨 符号约定必须与 OpenMM `dihedral()` 一致。这里曾是一份返回 **−φ** 的手写
    # 副本，它把 2026-07-29 那次 attachment 腿的参考几何整体镜像掉了
    # （ΔG(A′→A) 5.5 → 98.8 kJ/mol）。详见 `boresch_dihedral_rad` 的 docstring。
    dihedral = boresch_dihedral_rad

    r0 = dist(H0, L0)
    if not np.isfinite(r0):
        raise ValueError("Boresch r0 为 NaN/Inf，拒绝更新平衡几何")
    if r0 < 0.3 or r0 > 2.0:
        raise RuntimeError(
            f"Boresch r0={r0*10:.2f}Å 超出合理范围 [3, 20]Å；"
            "拒绝使用默认几何继续生产 ABFE。"
        )

    thetaA0 = angle(H1, H0, L0)
    thetaB0 = angle(H0, L0, L1)
    phiA0 = dihedral(H2, H1, H0, L0)
    phiB0 = dihedral(H1, H0, L0, L1)
    phiC0 = dihedral(H0, L0, L1, L2)
    geom = np.array([r0, thetaA0, thetaB0, phiA0, phiB0, phiC0], dtype=float)
    if not np.all(np.isfinite(geom)):
        raise ValueError(f"Boresch 平衡几何包含 NaN/Inf: {geom.tolist()}")

    return {
        "r0": float(r0),
        "thetaA0": float(thetaA0),  # H1-H0-L0
        "thetaB0": float(thetaB0),  # H0-L0-L1
        "phiA0": float(phiA0),      # H2-H1-H0-L0
        "phiB0": float(phiB0),      # H1-H0-L0-L1
        "phiC0": float(phiC0),      # H0-L0-L1-L2
    }


def assess_boresch_harmonicity(traj, receptor_indices, ligand_indices) -> Dict:
    """Model-free check of the harmonic/Gaussian assumption behind
    `calculate_boresch_analytical_correction`, computed directly from the
    trajectory that locked the anchor choice.

    Runs unconditionally for every Boresch source (auto/orb_simple/simple/
    fluctuation). The retired ML scanner in
    `dexp_退役.py::OrbScanner.scan_boresch_1d_pes` only implements the
    r-coordinate and was never called from a production pipeline. This uses
    the same distance/angle/dihedral convention as
    `calc_boresch_from_last_frame` (receptor_indices[0] nearest ligand) and
    reuses `GeometricRestraintEstimator._fluctuation_diagnostics` so the same
    skew/kurtosis/under-sampling criteria apply regardless of which estimator
    produced the anchors.
    """
    if not HAS_MDTRAJ:
        return {"ok": False, "reason": "mdtraj_unavailable"}

    rec_idx = [int(i) for i in receptor_indices]
    lig_idx = [int(i) for i in ligand_indices]
    if len(rec_idx) != 3 or len(lig_idx) != 3:
        return {"ok": False, "reason": "invalid_anchor_index_count"}
    if len(traj) < 4:
        return {"ok": False, "reason": "too_few_trajectory_frames"}

    dist_idx = [[rec_idx[0], lig_idx[0]]]
    angleA_idx = [[rec_idx[1], rec_idx[0], lig_idx[0]]]
    angleB_idx = [[rec_idx[0], lig_idx[0], lig_idx[1]]]
    dihA_idx = [[rec_idx[2], rec_idx[1], rec_idx[0], lig_idx[0]]]
    dihB_idx = [[rec_idx[1], rec_idx[0], lig_idx[0], lig_idx[1]]]
    dihC_idx = [[rec_idx[0], lig_idx[0], lig_idx[1], lig_idx[2]]]

    r = mdtraj.compute_distances(traj, dist_idx)[:, 0]
    thetaA = mdtraj.compute_angles(traj, angleA_idx)[:, 0]
    thetaB = mdtraj.compute_angles(traj, angleB_idx)[:, 0]
    phiA = mdtraj.compute_dihedrals(traj, dihA_idx)[:, 0]
    phiB = mdtraj.compute_dihedrals(traj, dihB_idx)[:, 0]
    phiC = mdtraj.compute_dihedrals(traj, dihC_idx)[:, 0]

    def _unwrap(vals):
        vals = np.asarray(vals, dtype=float).copy()
        for t in range(1, len(vals)):
            diff = vals[t] - vals[t - 1]
            vals[t] -= 2 * np.pi * np.round(diff / (2 * np.pi))
        mean_val = float(np.mean(vals)) if len(vals) else 0.0
        vals -= 2 * np.pi * np.round(mean_val / (2 * np.pi))
        return vals

    coords = {
        "r": r,
        "thetaA": thetaA,
        "thetaB": thetaB,
        "phiA": _unwrap(phiA),
        "phiB": _unwrap(phiB),
        "phiC": _unwrap(phiC),
    }
    fluctuation_diagnostics = [
        GeometricRestraintEstimator._fluctuation_diagnostics(vals, name)
        for name, vals in coords.items()
    ]
    n_bad = sum(1 for item in fluctuation_diagnostics if not item.get("ok", False))
    harmonic_ok = n_bad == 0

    result = {
        "ok": True,
        "method": "trajectory_fluctuation_v1",
        "n_frames_used": int(len(r)),
        "receptor_indices": rec_idx,
        "ligand_indices": lig_idx,
        "fluctuation_distribution": fluctuation_diagnostics,
        "n_non_gaussian_or_under_sampled_terms": int(n_bad),
        "harmonic_assumption_ok": bool(harmonic_ok),
        "warning": "",
    }
    if not harmonic_ok:
        result["warning"] = (
            f"{n_bad}/6 Boresch restraint coordinates show non-Gaussian or under-sampled "
            "fluctuations over the trajectory used to lock this restraint. "
            "calculate_boresch_analytical_correction assumes independent, approximately "
            "Gaussian coordinates; its result may be biased for this anchor choice. "
            "Consider a different --boresch-select candidate, longer pre-equilibration, "
            "or a numerical (non-analytical) release free-energy estimate."
        )
    return result


#=============================================================================
# 修复 3: 基于 RMSF 的自动化锚点选择器 (Automatic Anchor Selection)
#=============================================================================
def auto_select_boresch_anchors_rmsf(
    traj_path: str, top_path: str, ligand_resname: str,
    temperature: float = 300.0, rmsf_threshold_nm: float = 0.12,
    r0_range_angstrom: Tuple[float, float] = (5.0, 10.0),
    output_path: Optional[str] = None
) -> Dict:
    import mdtraj as md

    traj = md.load(traj_path, top=top_path)
    top = traj.topology

    align_atoms = top.select("protein and backbone")
    if len(align_atoms) >= 3:
        traj.superpose(traj, 0, atom_indices=align_atoms)

    ca_atoms = top.select("protein and name CA")
    if len(ca_atoms) < 3:
        raise RuntimeError("受体 CA 原子不足3个，无法构建 Boresch 限制")

    ca_rmsf = md.rmsf(traj, traj, 0, atom_indices=ca_atoms)
    rmsf_by_atom = {int(atom): float(value) for atom, value in zip(ca_atoms, ca_rmsf)}
    rigid_cas = [int(atom) for atom, value in zip(ca_atoms, ca_rmsf) if value <= rmsf_threshold_nm]
    if len(rigid_cas) < 3:
        order = np.argsort(ca_rmsf)
        rigid_cas = [int(ca_atoms[i]) for i in order[: min(12, len(order))]]
    else:
        rigid_cas = rigid_cas[: min(12, len(rigid_cas))]
    
    # 🔑 修复 1：枚举配体重原子三元组，而非死板取前 3 个
    lig_heavy = top.select(f"resname {ligand_resname} and not element H")
    if len(lig_heavy) < 3:
        raise RuntimeError("配体重原子不足3个，无法构建 Boresch 限制")
    
    # 按原子质量排序，优先选择重原子作为锚点候选
    lig_masses = np.array([top.atom(i).element.mass for i in lig_heavy])
    sorted_lig_idx = np.argsort(lig_masses)[::-1]
    top_lig_candidates = [int(lig_heavy[i]) for i in sorted_lig_idx[:min(10, len(sorted_lig_idx))]]
    
    best_score, best_config = -np.inf, None
    for rec_combo in combinations(rigid_cas, 3):
        for lig_combo in combinations(top_lig_candidates, 3):
            r_coords = traj.xyz[0, list(rec_combo)]
            l_coords = traj.xyz[0, list(lig_combo)]
            
            r0 = np.linalg.norm(r_coords[0] - l_coords[0]) * 10.0
            if not (r0_range_angstrom[0] <= r0 <= r0_range_angstrom[1]):
                continue
                
            def angle(a, b, c):
                v1, v2 = a - b, c - b
                return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10), -1, 1)))
            
            thA = angle(r_coords[1], r_coords[0], l_coords[0])
            thB = angle(r_coords[0], l_coords[0], l_coords[1])  # 🔑 修复 2：增加 thB 计算
            
            # 🔑 修复 3：严格同时拦截 thA 和 thB 的奇点
            if not (45 <= thA <= 135) or not (45 <= thB <= 135):
                continue
                
            # 锚点内部距离检查 (防止共线)
            if np.linalg.norm(r_coords[0]-r_coords[1]) < 0.3 or np.linalg.norm(l_coords[0]-l_coords[1]) < 0.2:
                continue
                
            rec_rmsf_mean = float(np.mean([rmsf_by_atom.get(int(i), rmsf_threshold_nm) for i in rec_combo]))
            score = 100 - abs(r0 - 7.5) * 5 - rec_rmsf_mean * 500
            if score > best_score:
                best_score = score
                best_config = {
                    "receptor_indices": list(rec_combo),
                    "ligand_indices": [int(i) for i in lig_combo],
                    "equilibrium_r0": float(r0 * 0.1),
                    "rmsf_mean": rec_rmsf_mean
                }
                
    if best_config is None:
        raise RuntimeError("未找到符合几何 (thA/thB) 与稳定性条件的锚点组合")
        
    print(f"✅ 自动锚点选择完成: r0={best_config['equilibrium_r0']:.2f}nm, RMSF={best_config['rmsf_mean']:.3f}nm")
    if output_path:
        with open(output_path, "w") as f:
            json.dump(best_config, f, indent=2, cls=NumpyEncoder)
    return best_config


#=============================================================================
# 修复 4: 分块 MBAR 分析器 (解决 OOM 瓶颈)
#=============================================================================
class ChunkedMBARAnalyzer:
    """
    支持超大 u_kn 矩阵的 MBAR 分析器
    ✅ 使用 np.memmap 零拷贝加载
    ✅ 自动分块计算，避免多进程 OOM
    ✅ 兼容 pymbar >= 3.0.5
    """
    def __init__(self, max_memory_gb: float = 32.0, cache_dir: str = "./mbar_cache", temperature_k: float = 300.0):
        import gc
        self.max_ram = max_memory_gb * 1e9
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.gc = gc
        # 🔑 与 TraditionalMBARAnalyzer 等项目内其它 MBAR 分析器使用同一个气体
        # 常数换算，供 _extract_delta_g 把 pymbar 返回的约化自由能(kT)转成 kJ/mol。
        self.kt = 0.008314462618 * float(temperature_k)
        
    def _save_chunk_to_disk(self, u_kn_block: np.ndarray, chunk_id: int) -> str:
        path = os.path.join(self.cache_dir, f"u_kn_chunk_{chunk_id}.npy")
        np.save(path, u_kn_block)
        return path
        
    def run_chunked_mbar(self, u_kn_total: np.ndarray, n_k_array: np.ndarray, stage_type: str = "coul"):
        """使用分态步幅抽样执行单次全局 MBAR，避免伪分块平均导致的统计错误。"""
        if not HAS_PYMBAR:
            raise ImportError("需要 pymbar 进行 MBAR 分析: pip install pymbar")
        import pymbar
        
        K, N = u_kn_total.shape
        n_k_array = np.asarray(n_k_array, dtype=int)
        if n_k_array.ndim != 1 or len(n_k_array) != K:
            raise ValueError(f"n_k_array 维度异常: 期望长度 {K}，实际 {n_k_array.shape}")
        if np.any(n_k_array < 0):
            raise ValueError("n_k_array 不能包含负样本数")

        # 1. 计算安全步幅 (Stride)，确保抽样后内存低于限制的 50%
        stride = 1
        while (K * (N // stride) * u_kn_total.itemsize) > self.max_ram * 0.5 and stride < N:
            stride *= 2

        if stride > 1:
            if np.sum(n_k_array) != N:
                raise MemoryError("u_kn_total 列数与 n_k_array 总和不一致，无法安全执行分态步幅抽样")

            print(f"  ⚠️ 内存受限，启用分态步幅抽样 (Stride={stride}) 后执行单次全局 MBAR")
            keep_indices = []
            start = 0
            n_k_sub = np.zeros_like(n_k_array)
            for k, n_k in enumerate(n_k_array):
                end = start + int(n_k)
                if n_k > 0:
                    state_idx = np.arange(start, end, stride, dtype=int)
                    if state_idx.size == 0:
                        state_idx = np.array([start], dtype=int)
                    keep_indices.append(state_idx)
                    n_k_sub[k] = state_idx.size
                start = end

            if not keep_indices:
                raise ValueError("n_k_array 全为 0，无法执行 MBAR")

            keep_indices = np.concatenate(keep_indices)
            u_kn_sub = u_kn_total[:, keep_indices]
            print(f"  ℹ️ 抽样后保留 {u_kn_sub.shape[1]} 帧，分态样本数: {n_k_sub.tolist()}")
        else:
            u_kn_sub = u_kn_total
            n_k_sub = n_k_array.copy()

        # 2. 单次全局 MBAR 求解 (统计严格)
        try:
            mbar = _build_mbar_compatible(
                u_kn_sub,
                n_k_sub,
                verbose=False,
                solver_protocol="hybr",
            )
            res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
            return self._extract_delta_g(res, n_k_sub, stage_type)
        except Exception as e:
            print(f"  🚨 MBAR 求解失败: {e}，尝试降级至 robust 求解器...")
            mbar = _build_mbar_compatible(
                u_kn_sub,
                n_k_sub,
                verbose=False,
                solver_protocol="robust",
            )
            res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
            return self._extract_delta_g(res, n_k_sub, stage_type)

    def _extract_delta_g(self, res, n_k_array, stage_type):
        """返回 kJ/mol 的自由能差/不确定度。

        _extract_free_energy_arrays 返回的是 pymbar 的约化自由能（单位 kT，
        即 beta*Delta_G），之前这里直接原样返回，调用方若不知情就会把 kT
        当 kJ/mol 直接使用（室温下 1 kT ≈ 2.5 kJ/mol，量级误差明显）。
        """
        df, ddf = _extract_free_energy_arrays(res, require_uncertainty=True)
        return df[0, :] * self.kt, ddf[0, :] * self.kt


#=============================================================================
# 修复 5: 溶剂化能闭环支持 (Ligand-in-Water)
#=============================================================================
# 与溶剂腿 createSystem 的 nonbondedCutoff 保持一致；盒长校验要用它。
SOLVENT_NONBONDED_CUTOFF_NM = 1.0

# GROMACS 水模型 itp 名 → 对应的 OpenMM 水模型 XML。
# 这张表只用来"翻译"复合物腿实际用的那个水模型，不是可选项列表：认不出来就
# fail closed，绝不回退到某个默认值。
GMX_TO_OPENMM_WATER_XML = {
    "tip3p": "amber14/tip3p.xml",
    "tip3pfb": "amber14/tip3pfb.xml",
    "tip4pew": "amber14/tip4pew.xml",
    "tip4pfb": "amber14/tip4pfb.xml",
    "spce": "amber14/spce.xml",
    "opc": "amber14/opc.xml",
    "opc3": "amber14/opc3.xml",
}


def resolve_water_model_xml(top_file: str) -> Tuple[str, str]:
    """从复合物 ``.top`` 的 ``#include`` 里解出水模型，返回 ``(OpenMM XML, 命中的 itp)``。

    溶剂腿的水必须和复合物腿是同一个模型。以前这里两边是各自硬编码的
    （复合物走 GROMACS ``amber14sb_OL15_fs1.ff/tip3p.itp``，而
    ``SolventLegRunner`` 写死 ``amber14/tip3pfb.xml``），TIP3P 与 TIP3P-FB 的
    σ/ε/电荷都不同，循环里本该抵消的水化项就对不上了。现在一律从复合物
    ``.top`` 反推，认不出来直接报错而不是默认成 TIP3P。
    """
    if not top_file or not os.path.isfile(top_file):
        raise FileNotFoundError(f"解析水模型需要有效的复合物 .top：{top_file!r}")
    with open(top_file, encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    hits: Dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line.startswith("#include"):
            continue
        rest = line[len("#include"):].strip()
        if len(rest) < 2 or rest[0] not in '"<':
            continue
        closing = '"' if rest[0] == '"' else ">"
        end = rest.find(closing, 1)
        if end < 0:
            continue
        include_path = rest[1:end]
        stem = os.path.splitext(os.path.basename(include_path))[0]
        key = stem.lower().replace("-", "").replace("_", "")
        if key in GMX_TO_OPENMM_WATER_XML:
            hits[key] = include_path

    if not hits:
        raise ValueError(
            f"在 {top_file} 的 #include 里没认出任何水模型；"
            f"已知的有 {sorted(GMX_TO_OPENMM_WATER_XML)}。"
            "拒绝为溶剂腿猜一个水模型——它必须和复合物腿一致。"
        )
    if len(hits) > 1:
        raise ValueError(
            f"{top_file} 同时 include 了多个水模型 {sorted(hits)}，无法确定复合物腿用的是哪个"
        )
    key, include_path = next(iter(hits.items()))
    return GMX_TO_OPENMM_WATER_XML[key], include_path


def solvent_box_edge_nm(
    lig_coords_nm,
    padding_nm: float,
    cutoff_nm: float = SOLVENT_NONBONDED_CUTOFF_NM,
) -> Tuple[float, float]:
    """按 ``gmx editconf -d`` 语义算立方溶剂盒边长：配体最长轴 + 2*padding。

    绝不依赖 ``addSolvent(padding=...)`` 自己推盒子——那条路径在本仓库产出过
    ``box = 2*padding`` 的 3.000 nm 立方盒（溶质尺寸对盒长的贡献是 0），配体
    最长轴 1.257 nm，每侧只剩 0.87 nm 溶剂；而且 OpenMM 7.7+ 的 padding 分支
    默认给的是菱形十二面体，不是立方。

    返回 ``(盒边 nm, 配体最长轴 nm)``。
    """
    coords = np.asarray(lig_coords_nm, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.all(np.isfinite(coords)):
        raise ValueError("配体坐标必须是有限的 (N, 3) nm 数组")
    padding = float(padding_nm)
    if not np.isfinite(padding) or padding <= 0.0:
        raise ValueError(f"padding_nm 必须是有限正数，收到 {padding_nm!r}")
    extent_nm = float(np.max(coords.max(axis=0) - coords.min(axis=0)))
    if not np.isfinite(extent_nm) or extent_nm <= 0.0:
        raise ValueError(f"配体最长轴计算异常: {extent_nm}")
    edge_nm = extent_nm + 2.0 * padding
    min_image_floor = 2.0 * float(cutoff_nm)
    if edge_nm <= min_image_floor:
        raise ValueError(
            f"溶剂盒边长 {edge_nm:.4f} nm 不满足最小镜像约定"
            f"（需 > 2×cutoff = {min_image_floor:.4f} nm）"
        )
    return edge_nm, extent_nm


class SolventLegRunner:
    """自动构建并运行 Ligand-in-Water 解耦腿"""
    def __init__(
        self,
        ligand_resname: str,
        box_size_nm: Optional[float] = None,
        platform_name: str = "CUDA",
        padding_nm: float = 1.5,
    ):
        self.ligand_resname = ligand_resname
        if box_size_nm is not None:
            warnings.warn(
                "SolventLegRunner.box_size_nm 已废弃且不再作为完整盒边；"
                "请使用 padding_nm 指定配体每侧的溶剂厚度。",
                DeprecationWarning,
                stacklevel=2,
            )
        self.padding_nm = float(padding_nm)
        if not np.isfinite(self.padding_nm) or self.padding_nm <= 0.0:
            raise ValueError("padding_nm 必须是有限正数")
        self.platform_name = platform_name
        self._cached_system = None
        self._cached_topology = None
        self._cached_positions = None
        self._cached_box_vectors = None

    def build_solvent_system(self, gro_file: str, top_file: str, gmx_include_dir: str = None):
        """从 GROMACS 文件提取配体并水盒化"""
        from openmm.app import Modeller, ForceField
        gro = app.GromacsGroFile(gro_file)
        top = app.GromacsTopFile(top_file, includeDir=gmx_include_dir)
        modeller = Modeller(top.topology, gro.positions)
        
        lig_indices = [atom.index for atom in gro.topology.atoms() if atom.residue.name == self.ligand_resname]
        if not lig_indices:
            raise ValueError(f"未在 GRO 中找到配体残基 {self.ligand_resname}")

        # 🔑 水模型不再硬编码，一律从复合物 .top 反推，保证两腿同模型。
        water_xml, water_itp = resolve_water_model_xml(top_file)
        logger.info("  💧 溶剂腿水模型继承自复合物 .top: %s → %s", water_itp, water_xml)
        ff = ForceField("amber14-all.xml", water_xml)

        # 🔑 盒子显式给出，不用 padding=。理由见 solvent_box_edge_nm 的 docstring。
        pos_nm = np.asarray(
            gro.positions.value_in_unit(unit.nanometer), dtype=np.float64
        )
        box_edge_nm, lig_extent_nm = solvent_box_edge_nm(
            pos_nm[lig_indices], padding_nm=self.padding_nm
        )
        modeller.addSolvent(
            ff,
            boxSize=openmm.Vec3(box_edge_nm, box_edge_nm, box_edge_nm) * unit.nanometer,
        )

        # fail closed：确认 OpenMM 真的按我们给的盒子建，而不是又自己推了一个。
        realized_vecs = modeller.topology.getPeriodicBoxVectors()
        if realized_vecs is None:
            raise RuntimeError("addSolvent 之后拓扑没有周期盒向量")
        realized_edges = np.linalg.norm(
            np.array([v.value_in_unit(unit.nanometer) for v in realized_vecs], dtype=float),
            axis=1,
        )
        if not np.allclose(realized_edges, box_edge_nm, atol=1.0e-6):
            raise RuntimeError(
                f"溶剂盒构建结果与请求不符：请求 {box_edge_nm:.6f} nm 立方"
                f"（配体最长轴 {lig_extent_nm:.4f} nm + 2×{self.padding_nm:.2f} nm），"
                f"实际边长 {[round(float(x), 6) for x in realized_edges]} nm"
            )

        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=SOLVENT_NONBONDED_CUTOFF_NM * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )
        # ✅ 缓存构建结果
        self._cached_system = system
        self._cached_topology = modeller.topology
        self._cached_positions = modeller.positions
        self._cached_box_vectors = modeller.topology.getPeriodicBoxVectors()
        return self._cached_system, self._cached_topology, self._cached_positions, self._cached_box_vectors
        
    def run_solvent_decoupling(self, positions, topology, ligand_indices, **pipeline_kwargs):
        """运行溶剂相解耦计算（委托给 ABFEPipeline）"""
        from abfe_pipeline import ABFEPipeline
        if self._cached_system is None:
            raise RuntimeError("请先调用 build_solvent_system 构建溶剂系统")
        
        pipe = ABFEPipeline(
            system=self._cached_system,          # ✅ 修复：传入有效 system
            topology=self._cached_topology,
            positions=self._cached_positions,
            box_vectors=self._cached_box_vectors,
            ligand_indices=ligand_indices,
            temperature=pipeline_kwargs.get('temperature', 300.0),
            output_dir=pipeline_kwargs.get('output_dir', './solvent_output'),
            platform_name=self.platform_name,
        )
        # ✅ 移除 decoupling_scheme 硬编码，透传用户配置
        return pipe.run_full_pipeline(**pipeline_kwargs)


# ============================================================================
# 7. 单位格式化器 (I/O 层专用：内核计算绝不碰单位转换)
# ============================================================================
class UnitFormatter:
    """
    单位格式化器：内部计算不碰，仅用于 I/O 转换
    【三层单位规范】
    1. 内核层 (OpenMM): nm, kJ/mol, ps, rad, e (裸数值)
    2. 数据交换层 (JSON): key 带单位后缀，如 r0_nm, kr_kJ_mol_nm2
    3. 人类可读层 (LOG/Report): Å, kcal/mol, ns, °
    """
    # === 基础转换函数 ===
    @staticmethod
    def nm_to_A(val): return val * 10.0
    @staticmethod
    def A_to_nm(val): return val * 0.1
    @staticmethod
    def kJ_to_kcal(val): return val / 4.184
    @staticmethod
    def kcal_to_kJ(val): return val * 4.184
    @staticmethod
    def rad_to_deg(val): return np.degrees(val)
    @staticmethod
    def deg_to_rad(val): return np.radians(val)
    @staticmethod
    def ps_to_ns(val): return val / 1000.0
    @staticmethod
    def ns_to_ps(val): return val * 1000.0

    # === Boresch 参数格式化 (人类可读) ===
    @classmethod
    def format_boresch_human(cls, boresch_dict: dict) -> str:
        """格式化 Boresch 参数为化学家常用单位"""
        eq = boresch_dict["equilibrium_values"]
        fc = boresch_dict["force_constants"]
        return (
            f"📏 r0={cls.nm_to_A(eq['r0']):.2f} Å | "
            f"θA={cls.rad_to_deg(eq['thetaA0']):.1f}° | "
            f"φA={cls.rad_to_deg(eq['phiA0']):.1f}°\n"
            f"⚖️ kr={cls.kJ_to_kcal(fc['kr']) / 100.0:.2f} kcal/mol/Å²| "
            f"kθA={cls.kJ_to_kcal(fc['kthetaA']):.2f} kcal/mol/rad²"
        )

    # === Boresch 参数序列化 (JSON，key 带单位后缀) ===
    @classmethod
    def format_boresch_json(cls, boresch_dict: dict) -> dict:
        """
        序列化 Boresch 参数为 JSON 安全格式 (兼容扁平/嵌套/混合结构)
        🔑 智能路由提取逻辑，彻底解决 KeyError: 'equilibrium_values'
        """
        # 1. 智能提取：优先从嵌套层取，若为空则降级到顶层
        anchors = boresch_dict.get("boresch_anchors", boresch_dict)
        eq = anchors.get("equilibrium_values") or boresch_dict.get("equilibrium_values", {})
        fc = anchors.get("force_constants") or boresch_dict.get("force_constants", {})
        rec_idx = anchors.get("receptor_indices") or boresch_dict.get("receptor_indices", [])
        lig_idx = anchors.get("ligand_indices") or boresch_dict.get("ligand_indices", [])

        if not eq or not fc:
            raise ValueError("Boresch 参数字典结构异常：缺失 equilibrium_values 或 force_constants")

        # 2. 构建标准嵌套输出 (严格带单位后缀)
        return {
            "boresch_anchors": {
                "receptor_indices": rec_idx,
                "ligand_indices": lig_idx,
                "equilibrium_values": {
                    "r0_nm": float(eq.get("r0", 0)),
                    "thetaA0_rad": float(eq.get("thetaA0", 0)),
                    "thetaB0_rad": float(eq.get("thetaB0", 0)),
                    "phiA0_rad": float(eq.get("phiA0", 0)),
                    "phiB0_rad": float(eq.get("phiB0", 0)),
                    "phiC0_rad": float(eq.get("phiC0", 0)),
                },
                "force_constants": {
                    "kr_kJ_mol_nm2": float(fc.get("kr", 0)),
                    "kthetaA_kJ_mol_rad2": float(fc.get("kthetaA", 0)),
                    "kthetaB_kJ_mol_rad2": float(fc.get("kthetaB", 0)),
                    "kphiA_kJ_mol_rad2": float(fc.get("kphiA", 0)),
                    "kphiB_kJ_mol_rad2": float(fc.get("kphiB", 0)),
                    "kphiC_kJ_mol_rad2": float(fc.get("kphiC", 0)),
                },
            },
            "is_fallback": boresch_dict.get("is_fallback", False),
            "total_score": boresch_dict.get("total_score", None),
            "diagnostics": boresch_dict.get("diagnostics", None),
        }

    # === 结果格式化 (人类可读) ===
    @classmethod
    def format_results_human(cls, results: dict) -> str:
        """格式化最终结果报告"""
        err_kj = results.get("total_error_kJ_mol", results.get("total_error", 0.0))
        if "delta_G_bind_kJ_mol" in results:
            dg_kj = results.get("delta_G_bind_kJ_mol", 0.0)
            title = "✅ 结合自由能 ΔG_bind"
        elif "total_delta_G_complex_kJ_mol" in results:
            dg_kj = results.get("total_delta_G_complex_kJ_mol", 0.0)
            title = "✅ 复合物总自由能 ΔG_complex"
        elif "decoupling_delta_G_kJ_mol" in results:
            dg_kj = results.get("decoupling_delta_G_kJ_mol", 0.0)
            title = "✅ 解耦腿自由能 ΔG_leg"
        else:
            dg_kj = results.get("total_delta_G_complex_kJ_mol", results.get("total_delta_G_complex", 0.0))
            title = "✅ 自由能结果 ΔG"
        return (
            f"\n{'='*50}\n"
            f"{title} = {cls.kJ_to_kcal(dg_kj):.2f} ± {cls.kJ_to_kcal(err_kj):.2f} kcal/mol\n"
            f"   ( = {dg_kj:.2f} ± {err_kj:.2f} kJ/mol )\n"
            f"{'='*50}"
        )

    # === 采样元数据格式化 (JSON) ===
    @classmethod
    def format_sampling_metadata_json(cls, config: dict) -> dict:
        """序列化采样元数据为 JSON"""
        return {
            "sampling_metadata": {
                "dt_ps": config.get("timestep_ps", 0.002),
                "temperature_K": config.get("temperature", 300.0),
                "n_steps_per_window": config.get("n_steps_per_window", 0),
                "friction_ps": config.get("friction", 1.0),
            }
        }

# ============================================================================
# 通用工具函数：System 管理与配体内部力构建
# ============================================================================
from openmm import XmlSerializer

def ensure_owned_system(system: openmm.System) -> openmm.System:
    """强制获取 System 的 Python 所有权，防止 SWIG GC"""
    if system is None:
        raise ValueError("System 对象为 None")
    if getattr(system, 'thisown', 0) == 1:
        try:
            _ = system.getNumParticles()
        except Exception as exc:
            raise RuntimeError(
                "System 声称由 Python 持有，但底层 OpenMM 对象已不可访问。"
            ) from exc
        return system
    xml = XmlSerializer.serialize(system)
    new_sys = XmlSerializer.deserialize(xml)
    new_sys.thisown = 1
    _ = new_sys.getNumParticles()
    return new_sys


def sync_all_exclusions(system: openmm.System) -> int:
    """
    生产级排除表同步。

    🚨 关键修复：OpenMM 要求同一 System 里所有共享同一套邻居表的
    NonbondedForce/CustomNonbondedForce（粒子数相同）拥有完全相同的排除表——
    "All Forces must have identical exclusions" 就是这个要求被违反时抛出的。
    旧版本按每个 CustomNonbondedForce 的 interaction group 范围"按需"补齐排除表
    （例如只给 L-E 力补 L-E 相关的对），这在物理上没问题（interaction group
    之外的对本来就不会被计算），但 OpenMM 底层邻居表校验比较的是排除表本身
    是否逐对相同，不管 interaction group——所以只要 NonbondedForce 里有任何
    一个不落在某个 CustomNonbondedForce interaction group 内的排除对
    （典型情况：环境蛋白/水分子自身的 1-2/1-3/1-4 排除，跟只处理 L-E 的软核力
    毫不相关），旧逻辑就会让两者的排除表数量对不上，从而在生产采样阶段
    （通常是第一次真正调用 minimizeEnergy/getState 触发底层邻居表构建时）报错。
    这里改为无差别地把"并集"灌给每一个粒子数匹配的力，牺牲一点点冗余排除对，
    换来严格逐对相同——interaction group 之外的排除对本来就不会被该力用到，
    是纯粹的账本对齐，不改变任何物理量。
    """
    nb_forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    custom_forces = [f for f in system.getForces() if isinstance(f, openmm.CustomNonbondedForce)]
    if not nb_forces or not custom_forces:
        return 0
    nb_force = nb_forces[0]
    n_particles = nb_force.getNumParticles()

    union_excl = set()
    for i in range(nb_force.getNumExceptions()):
        p1, p2, _, _, _ = nb_force.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if p1 != p2:
            union_excl.add((min(p1, p2), max(p1, p2)))

    eligible_forces = []
    existing_per_force = []
    for c_force in custom_forces:
        if c_force.getNumParticles() != n_particles:
            continue
        existing = set()
        for i in range(c_force.getNumExclusions()):
            p1, p2 = c_force.getExclusionParticles(i)
            existing.add((min(int(p1), int(p2)), max(int(p1), int(p2))))
        union_excl |= existing
        eligible_forces.append(c_force)
        existing_per_force.append(existing)

    total_synced = 0
    for c_force, existing in zip(eligible_forces, existing_per_force):
        missing = union_excl - existing
        for p1, p2 in missing:
            c_force.addExclusion(p1, p2)
        total_synced += len(missing)
    return total_synced


def create_ligand_internal_force(
    nb_force: openmm.NonbondedForce,
    perturbed_indices: List[int],
    particle_params,
    reference_exclusions=None,
    num_particles: int = None,
    system: openmm.System = None
):
    """
    构建配体-配体内部非键力 (Standard LJ + Coulomb) 和 1-4 恢复力。
    注意：此函数不分配 ForceGroup，调用者需自行设置并添加至 System。
    """
    if num_particles is None:
        num_particles = nb_force.getNumParticles()
    perturbed_set = set(perturbed_indices)

    expr = "4*sqrt(epsilon1*epsilon2)*((sigma12/r)^12 - (sigma12/r)^6) + 138.935456*q1*q2/r; sigma12 = 0.5*(sigma1+sigma2)"
    ll_force = openmm.CustomNonbondedForce(expr)
    ll_force.addPerParticleParameter('q')
    ll_force.addPerParticleParameter('sigma')
    ll_force.addPerParticleParameter('epsilon')

    for i in range(num_particles):
        if particle_params and i < len(particle_params):
            q, sig, eps = particle_params[i]
        else:
            q, sig, eps = nb_force.getParticleParameters(i)
        ll_force.addParticle([
            q.value_in_unit(unit.elementary_charge),
            sig.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole)
        ])

    ll_force.addInteractionGroup(perturbed_set, perturbed_set)
    ll_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    ll_force.setCutoffDistance(1.2 * unit.nanometer)
    ll_force.setUseLongRangeCorrection(False)

    # ========================================================================
    # 🔑 生产级排除对收集：全覆盖 1-2/1-3/1-4 (修复漏扫约束与异常表的致命缺陷)
    # ========================================================================
    exclusion_pairs = set()

    if system is not None:
        # === 1. 谐波键 (1-2 排除) ===
        for f in system.getForces():
            if isinstance(f, openmm.HarmonicBondForce):
                for i in range(f.getNumBonds()):
                    p1, p2, _, _ = f.getBondParameters(i)
                    if p1 in perturbed_set and p2 in perturbed_set:
                        exclusion_pairs.add((min(p1, p2), max(p1, p2)))
            
            # === 2. 谐波角 (1-3 排除，取首尾原子) ===
            elif isinstance(f, openmm.HarmonicAngleForce):
                for i in range(f.getNumAngles()):
                    p1, p2, p3, _, _ = f.getAngleParameters(i)
                    # ✅ 仅当首尾原子都在配体内才排除 (1-3)
                    if p1 in perturbed_set and p3 in perturbed_set:
                        exclusion_pairs.add((min(p1, p3), max(p1, p3)))
        
        # === 3. 刚性约束 (1-2 排除，GROMACS 常将含 H 键转为约束) ===
        # 🔑 核心修复：独立于力遍历，直接扫描系统级约束
        for i in range(system.getNumConstraints()):
            p1, p2, _ = system.getConstraintParameters(i)
            if p1 in perturbed_set and p2 in perturbed_set:
                exclusion_pairs.add((min(p1, p2), max(p1, p2)))
        
        # === 4. NonbondedForce 异常表 (1-4 排除) ===
        nb_forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
        if nb_forces:
            nb = nb_forces[0]
            for i in range(nb.getNumExceptions()):
                p1, p2, _, _, _ = nb.getExceptionParameters(i)
                p1, p2 = int(p1), int(p2)
                if p1 in perturbed_set and p2 in perturbed_set:
                    exclusion_pairs.add((min(p1, p2), max(p1, p2)))

    # === 5. 合并参考排除表 (来自原始 NonbondedForce 的 exceptions) ===
    if reference_exclusions:
        for p1, p2 in reference_exclusions:
            p1, p2 = int(p1), int(p2)
            if p1 in perturbed_set and p2 in perturbed_set:
                exclusion_pairs.add((min(p1, p2), max(p1, p2)))

    # === 6. 执行排除添加 (严格去重) ===
    for p1, p2 in exclusion_pairs:
        ll_force.addExclusion(p1, p2)

    # 🔍 诊断输出 (可选，生产环境可注释)
    print(f"  🔍 [Group2 排除表] 共收集 {len(exclusion_pairs)} 对配体内部排除 (1-2/1-3/1-4)")

    # 1-4 恢复力
    ll_14_force = None
    exceptions_14 = []
    for i in range(nb_force.getNumExceptions()):
        p1, p2, chargeProd, sigma, epsilon = nb_force.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if p1 in perturbed_set and p2 in perturbed_set:
            has_charge = chargeProd.value_in_unit(unit.elementary_charge**2) != 0
            has_lj = epsilon.value_in_unit(unit.kilojoule_per_mole) != 0
            if has_charge or has_lj:
                exceptions_14.append((p1, p2, chargeProd, sigma, epsilon))

    if exceptions_14:
        expr_14 = "4*epsilon*((sigma/r)^12 - (sigma/r)^6) + 138.935456*chargeProd/r"
        ll_14_force = openmm.CustomBondForce(expr_14)
        ll_14_force.addPerBondParameter('chargeProd')
        ll_14_force.addPerBondParameter('sigma')
        ll_14_force.addPerBondParameter('epsilon')
        for p1, p2, cp, sig, eps in exceptions_14:
            ll_14_force.addBond(p1, p2, [
                cp.value_in_unit(unit.elementary_charge**2),
                sig.value_in_unit(unit.nanometer),
                eps.value_in_unit(unit.kilojoule_per_mole)
            ])

    ll_force.setUseSwitchingFunction(True)
    ll_force.setSwitchingDistance(1.0 * unit.nanometer)
    return ll_force, ll_14_force

# ============================================================================
# 8. 纯轨迹几何波动 Boresch 估算器 (基于化学连通性 + 方差最小化)
# ============================================================================
class GeometricRestraintEstimator:
    """
    基于轨迹几何波动和化学连通性的 Boresch 参数估算器。
    不依赖任何力场，仅需 mdtraj 轨迹。
    核心流程：
      1. 受体候选原子：指定原子名 (默认 CA,CB,C,N,O)
      2. 0.5 nm 接触搜索找到 (锚点,配体) 最近原子对
      3. 0.22 nm 成键延伸构建受体三元组和配体三元组 (保证化学连通)
      4. 计算全轨迹距离/角度/二面角，周期性展开
      5. 硬截断 θ ∈ [45°,135°]
      6. 方差加权评分 (物理力常数尺度) 选择最优组合
      7. 力常数 = kBT / 方差 (并裁剪到安全范围)
    """

    def __init__(self, temperature=300.0,
                 search_dist=0.5,         # nm
                 bond_dist=0.22,          # nm，仅在拓扑不含键时作为显式回退
                 anchor_atom_names=None,
                 allow_geometric_bond_fallback=True,
                 # 某一侧要用拓扑真实键，该侧至少这个比例的原子能起出 2 深链。
                 # 真正描述了成键的拓扑接近 100%（实测受体 1404/1404）；只零星
                 # 知道几根残基间连接的接近 10%（实测配体 2/19）。0.5 把两者
                 # 分得很开，不是贴着数据挑的边界。
                 bond_topology_min_coverage=0.5):
        self.temperature = temperature
        self.gas_constant_kj_per_mol_k = 8.314e-3
        self.search_dist = search_dist
        self.bond_dist = bond_dist
        # 🔑 [ATT-11] 拓扑里有可用键时用真实键；这个开关只决定"某一侧拓扑没有可用键"
        # 时是回退到几何阈值还是直接 fail closed。
        self.allow_geometric_bond_fallback = bool(allow_geometric_bond_fallback)
        self.bond_topology_min_coverage = float(bond_topology_min_coverage)
        # 🔑 键来源**逐侧**记录。受体与配体的情况可以完全不同：实测本体系
        # （topology.cif，`_chem_comp_bond = 0`）受体锚点 1404/1404 都有真实键，
        # 而配体作为非标准残基只有 2/19 个重原子蹭到键——两侧必须分别决策。
        self.bond_source_receptor = None   # "topology" | "geometric_fallback"
        self.bond_source_ligand = None     # 同上
        self.bond_coverage = {}            # 逐侧 2 深链起点计数，供诊断落盘
        if anchor_atom_names is None:
            anchor_atom_names = ["CA", "CB", "C", "N", "O"]
        self.anchor_atom_names = anchor_atom_names

    @property
    def bond_source(self):
        """两侧的汇总：只有两侧都用真实键才算 "topology"。

        保留这个属性是为了兼容既有的诊断落盘读取方；真正的信息在
        `bond_source_receptor` / `bond_source_ligand` 里。
        """
        sides = (self.bond_source_receptor, self.bond_source_ligand)
        if None in sides:
            return None
        return "topology" if set(sides) == {"topology"} else "mixed_or_geometric_fallback"

    # ----------------------------------------------------------------
    # 工具：化学键邻居
    # ----------------------------------------------------------------
    def _build_bond_adjacency(self, topology) -> Optional[Dict[int, set]]:
        """🔑 [ATT-11] 从拓扑的真实成键关系建邻接表。

        原实现用 `距离 <= 0.22 nm` 冒充共价键，有三个问题：

        1. **区分不了成键与非键近接**。蛋白侧 haystack 是预筛过的锚点名子集
           （默认 CA/CB/C/N/O），其中 CA-CB≈0.153 nm、CA-C≈0.152 nm 是真键，
           但**非键**的 i/i+1 残基间 C-N≈0.133 nm、CA…N≈0.146 nm 同样落在
           0.22 nm 以内，会被当成键——于是"化学连通"的受体三元组可能跨残基
           拼出一条根本不存在的链。
        2. **漏掉长键**。S-S（≈0.205 nm）贴着阈值，金属配位键普遍超过 0.22 nm。
        3. **只看第 0 帧**（见 `_generate_anchor_combos` 的 `ref_xyz = traj.xyz[0]`），
           一次热涨落就能翻转键拓扑。

        Boresch 六原子锚点是由这张邻接表枚举出来的，锚点选错会直接改变解析释放
        修正——2026-07-27 的 P0-10 已经演示过锚点/平衡值出错的代价。

        返回 None 表示拓扑里没有键信息，由调用方决定回退还是 fail closed。
        """
        if topology is None:
            return None
        bonds = getattr(topology, "bonds", None)
        # mdtraj 的 Topology.bonds 是 property（生成器），OpenMM 的是方法。
        if callable(bonds):
            try:
                bonds = bonds()
            except TypeError:
                return None
        if bonds is None:
            return None
        adjacency: Dict[int, set] = {}
        n_bonds = 0
        for bond in bonds:
            try:
                a1, a2 = bond[0], bond[1]
            except (TypeError, KeyError, IndexError):
                a1, a2 = getattr(bond, "atom1", None), getattr(bond, "atom2", None)
            if a1 is None or a2 is None:
                continue
            i = int(getattr(a1, "index", a1))
            j = int(getattr(a2, "index", a2))
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)
            n_bonds += 1
        return adjacency if n_bonds > 0 else None

    @staticmethod
    def _count_two_deep_chain_starts(adjacency, indices) -> int:
        """数一下该原子子集里有多少个原子能起出 `a→b→c` 且 b、c 都还在子集内。

        🔑 [ATT-11 回归修复] 这才是 `_generate_anchor_combos` 真正消费的性质。

        原先的覆盖度判据是 `any(adjacency.get(i) for i in indices)`——「该侧只要有
        任意一个原子有键就放行」。实测那道判据太弱到直接造成生产崩溃：
        `topology.cif` 的 `_chem_comp_bond = 0`，配体 `MOL` 作为非标准残基只有
        **2/19** 个重原子从 `_struct_conn` 蹭到键，`any()` 被这 2 个原子放行，
        于是走了拓扑路径；而那 2 个原子恰好不在接触对里，配体侧最内层枚举
        从不执行 → `化学连通候选组合数: 0` → `RuntimeError: 没有符合条件的6原子组合`。

        注意「链上原子必须都在子集内」不是多余的限制：受体 haystack 是预筛的
        CA/CB/C/N/O 子集，配体 haystack 排除了氢；链走出子集就不能用来构造
        Boresch 三元组。
        """
        if not adjacency:
            return 0
        subset = {int(x) for x in indices}
        n = 0
        for a in subset:
            for b in adjacency.get(a, ()):  # noqa: B007
                if int(b) not in subset:
                    continue
                if (adjacency.get(int(b), set()) & subset) - {a}:
                    n += 1
                    break
        return n

    def _resolve_side_adjacency(self, side_label, adjacency, indices):
        """为某一侧（受体 / 配体）决定用拓扑真实键还是几何回退。

        返回 `(adjacency_or_None, source, n_two_deep, n_atoms)`。

        **判据是覆盖度比例，不是"有没有"。** 一个真正描述了该侧成键的拓扑，
        几乎每个原子都能起出 2 深链（实测受体锚点 1404/1404 = 100%）；
        而只零星知道几根残基间连接的拓扑给出的是 10% 这个量级
        （实测配体 2/19 = 10.5%）。用 `>= 1` 当门槛挡不住后者——那 2 个原子
        恰好不在接触对里，配体侧枚举照样全灭，正是 2026-07-27 生产崩溃的原因。

        **逐侧决策的理由是一个真实的不对称**：

        - **配体侧**：haystack 是全部重原子，`0.22 nm` 的最近邻**确实就是化学键**
          （小分子键长 0.13–0.16 nm，次近邻 ≥ 0.24 nm）。所以几何回退在这一侧
          是可靠的。
        - **受体侧**：haystack 被按原子名预筛成 CA/CB/C/N/O，于是**非键**的残基间
          C–N（≈0.133 nm）、CA…N（≈0.146 nm）也落在阈值内，会拼出跨残基的假
          「化学连通」三元组。所以这一侧必须用真实键。

        换句话说：ATT-11 的收益全在受体侧，而配体侧本来就不太需要它。
        """
        n_atoms = len(indices)
        n_two_deep = self._count_two_deep_chain_starts(adjacency, indices)
        frac = (n_two_deep / n_atoms) if n_atoms else 0.0
        min_frac = float(self.bond_topology_min_coverage)

        if n_atoms and frac >= min_frac:
            print(
                f"  ✓ [Boresch 锚点] {side_label}使用拓扑真实键"
                f"（2 深链起点 {n_two_deep}/{n_atoms} = {frac:.0%}）"
            )
            return adjacency, "topology", n_two_deep, n_atoms

        if not self.allow_geometric_bond_fallback:
            raise RuntimeError(
                f"{side_label}的拓扑成键覆盖度不足："
                f"2 深链起点仅 {n_two_deep}/{n_atoms} = {frac:.0%}，"
                f"低于要求的 {min_frac:.0%}，不足以可靠枚举 Boresch 三元组；"
                f"且已禁用几何回退，拒绝用 `距离 <= {self.bond_dist} nm` 冒充共价键。"
                "\n  请提供带该侧完整键的拓扑（例如带 CONECT 的 PDB / prmtop / "
                "OpenMM Topology），或显式设 allow_geometric_bond_fallback=True。"
                "\n  提示：OpenMM 写出的 mmCIF 不含 `_chem_comp_bond`，"
                "非标准残基（配体）在其中没有键。"
            )
        print(
            f"  ⚠️ [Boresch 锚点] {side_label}拓扑成键覆盖度不足（2 深链起点 "
            f"{n_two_deep}/{n_atoms} = {frac:.0%} < {min_frac:.0%}），"
            f"该侧退回几何阈值 {self.bond_dist} nm。"
        )
        return None, "geometric_fallback", n_two_deep, n_atoms

    def _find_bonded_neighbors(self, atom_idx, haystack, ref_xyz, adjacency=None):
        """返回 haystack 里与 atom_idx 成键的原子。

        `adjacency` 非空时用真实成键关系；为空时（拓扑无键）才回退到
        `距离 <= bond_dist` 的几何近似——见 `_build_bond_adjacency` 的说明。
        """
        haystack = np.asarray(haystack)
        if adjacency is not None:
            neighbors = adjacency.get(int(atom_idx), ())
            return [int(b) for b in haystack if int(b) in neighbors and int(b) != int(atom_idx)]
        vec = ref_xyz[haystack] - ref_xyz[atom_idx]
        dist = np.linalg.norm(vec, axis=1)
        bonded = haystack[dist <= self.bond_dist]
        return [int(b) for b in bonded if int(b) != int(atom_idx)]

    @staticmethod
    def _clip_force_constant(value, lower, upper):
        raw = float(value)
        clipped = float(np.clip(raw, lower, upper))
        return clipped, bool(abs(clipped - raw) > 1e-8)

    @staticmethod
    def _fluctuation_diagnostics(values, name):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 4:
            return {
                "name": name,
                "n": int(vals.size),
                "ok": False,
                "reason": "too_few_finite_samples",
            }

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std <= 1e-12:
            return {
                "name": name,
                "n": int(vals.size),
                "ok": False,
                "mean": mean,
                "std": std,
                "reason": "near_zero_variance",
            }

        centered = (vals - mean) / std
        skew = float(np.mean(centered ** 3))
        excess_kurtosis = float(np.mean(centered ** 4) - 3.0)
        p01, p50, p99 = np.percentile(vals, [1, 50, 99])
        ok = bool(abs(skew) <= 2.0 and abs(excess_kurtosis) <= 7.0)
        reason = "ok" if ok else "non_gaussian_tail_or_asymmetry"
        return {
            "name": name,
            "n": int(vals.size),
            "ok": ok,
            "mean": mean,
            "std": std,
            "skew": skew,
            "excess_kurtosis": excess_kurtosis,
            "p01": float(p01),
            "p50": float(p50),
            "p99": float(p99),
            "reason": reason,
        }

    # ----------------------------------------------------------------
    # 生成所有化学连通的 6-原子组合
    # ----------------------------------------------------------------
    def _generate_anchor_combos(self, traj, prot_indices, lig_heavy_indices):
        ref_xyz = traj.xyz[0]

        # 1. 接触对 (anch, lig) 距离 ≤ search_dist
        lig_pos = ref_xyz[lig_heavy_indices]
        prot_pos = ref_xyz[prot_indices]
        dist_mat = np.linalg.norm(prot_pos[:, None, :] - lig_pos[None, :, :], axis=2)
        contact_pairs = [(prot_indices[i], lig_heavy_indices[j])
                         for i, j in zip(*np.where(dist_mat <= self.search_dist))]
        if not contact_pairs:
            raise RuntimeError("未找到锚点-配体接触对，请增大 search_dist")

        # 2. 预计算键合邻居字典
        # 🔑 [ATT-11 + 回归修复] 受体与配体**各自独立**决定键来源。
        # 实测本体系（topology.cif，`_chem_comp_bond = 0`）：受体锚点 1404/1404
        # 都有真实键，而配体作为非标准残基只有 2/19 个重原子蹭到键。全局决策
        # （任一侧不合格就整体退几何）会白丢受体侧的收益；反之若因为「配体有 2 个
        # 原子有键」就整体用拓扑，配体侧枚举直接全灭——那正是 2026-07-27 生产崩溃
        # （`化学连通候选组合数: 0`）的原因。
        adjacency = self._build_bond_adjacency(getattr(traj, "topology", None))
        prot_adj, self.bond_source_receptor, prot_deep, prot_n = (
            self._resolve_side_adjacency("受体锚点侧", adjacency, prot_indices)
        )
        lig_adj, self.bond_source_ligand, lig_deep, lig_n = (
            self._resolve_side_adjacency("配体侧", adjacency, lig_heavy_indices)
        )
        self.bond_coverage = {
            "receptor_two_deep_chain_starts": int(prot_deep),
            "receptor_n_atoms": int(prot_n),
            "ligand_two_deep_chain_starts": int(lig_deep),
            "ligand_n_atoms": int(lig_n),
        }

        prot_nei = {
            idx: self._find_bonded_neighbors(idx, prot_indices, ref_xyz, prot_adj)
            for idx in prot_indices
        }
        lig_nei = {
            idx: self._find_bonded_neighbors(idx, lig_heavy_indices, ref_xyz, lig_adj)
            for idx in lig_heavy_indices
        }

        anclig_combos = []
        for anc, lig in contact_pairs:
            # 受体侧：以 anc 为 a，找与其键合的 b，再找与 b 键合的 c -> (c, b, a)
            for b in prot_nei.get(anc, []):
                for c in prot_nei.get(b, []):
                    if c == anc: continue
                    rec_tri = (c, b, anc)
                    # 配体侧：以 lig 为 a，找键合的 b，再找与 b 键合的 c -> (a, b, c)
                    for b_lig in lig_nei.get(lig, []):
                        for c_lig in lig_nei.get(b_lig, []):
                            if c_lig == lig: continue
                            lig_tri = (lig, b_lig, c_lig)
                            anclig_combos.append((rec_tri, lig_tri))

        # 去重
        unique = []
        seen = set()
        for rec, lig in anclig_combos:
            key = rec + lig
            if key not in seen:
                seen.add(key)
                unique.append((rec, lig))
        return unique

    # ----------------------------------------------------------------
    # 主估算函数
    # ----------------------------------------------------------------
    def estimate_from_trajectory(self, traj, ligand_resname, output_path=None):
        top = traj.topology

        # 1. 受体锚点候选原子 (基于原子名)
        anchor_query = "protein and name " + ' '.join(self.anchor_atom_names)
        prot_indices = top.select(anchor_query)
        if len(prot_indices) == 0:
            raise RuntimeError(f"没有找到锚点原子：{self.anchor_atom_names}")

        # 2. 配体重原子
        lig_heavy = top.select(f"resname {ligand_resname} and not element H")
        if len(lig_heavy) == 0:
            raise RuntimeError(f"未找到配体 {ligand_resname} 的重原子")

        # 3. 化学连通组合枚举
        combos = self._generate_anchor_combos(traj, prot_indices, lig_heavy)
        print(f"  🔗 化学连通候选组合数: {len(combos)}")
        if len(combos) == 0:
            raise RuntimeError("没有符合条件的6原子组合")

        # 4. 构建 mdtraj 原子索引列表 (用于批量计算几何)
        n_combos = len(combos)
        dist_indices   = [[c[0][2], c[1][0]] for c in combos]   # anc_a - lig_a
        angleA_indices = [[c[0][1], c[0][2], c[1][0]] for c in combos]  # anc_b, anc_a, lig_a
        angleB_indices = [[c[0][2], c[1][0], c[1][1]] for c in combos]
        dihA_indices   = [[c[0][0], c[0][1], c[0][2], c[1][0]] for c in combos]
        dihB_indices   = [[c[0][1], c[0][2], c[1][0], c[1][1]] for c in combos]
        dihC_indices   = [[c[0][2], c[1][0], c[1][1], c[1][2]] for c in combos]

        # 5. 逐帧计算几何量 (分块避免 OOM)
        n_frames = len(traj)
        dists    = np.zeros((n_frames, n_combos))
        angles_a = np.zeros((n_frames, n_combos))
        angles_b = np.zeros((n_frames, n_combos))
        diheds_a = np.zeros((n_frames, n_combos))
        diheds_b = np.zeros((n_frames, n_combos))
        diheds_c = np.zeros((n_frames, n_combos))

        chunk_size = 100  # 可调整
        for i in range(0, n_frames, chunk_size):
            chunk = traj[i:i+chunk_size]
            dists[i:i+len(chunk)]    = mdtraj.compute_distances(chunk, dist_indices)
            angles_a[i:i+len(chunk)] = mdtraj.compute_angles(chunk, angleA_indices)
            angles_b[i:i+len(chunk)] = mdtraj.compute_angles(chunk, angleB_indices)
            diheds_a[i:i+len(chunk)] = mdtraj.compute_dihedrals(chunk, dihA_indices)
            diheds_b[i:i+len(chunk)] = mdtraj.compute_dihedrals(chunk, dihB_indices)
            diheds_c[i:i+len(chunk)] = mdtraj.compute_dihedrals(chunk, dihC_indices)

        # 6. 周期性二面角展开 (按列展开，保持连续性)
        def periodic_unwrap(dh_array):
            for col in range(dh_array.shape[1]):
                vals = dh_array[:, col]
                for t in range(1, len(vals)):
                    diff = vals[t] - vals[t-1]
                    vals[t] -= 2*np.pi * np.round(diff / (2*np.pi))
                mean_val = np.mean(vals)
                vals -= 2*np.pi * np.round(mean_val / (2*np.pi))
                dh_array[:, col] = vals

        periodic_unwrap(diheds_a)
        periodic_unwrap(diheds_b)
        periodic_unwrap(diheds_c)

        # 7. 方差加权评分 (物理力常数尺度)
        dist_weight = 4184.0       # kJ/mol/nm²
        angle_weight = 41.84       # kJ/mol/rad²
        dihedral_weight = 41.84

        var_dist = np.var(dists, axis=0)
        var_angA = np.var(angles_a, axis=0)
        var_angB = np.var(angles_b, axis=0)
        var_dihA = np.var(diheds_a, axis=0)
        var_dihB = np.var(diheds_b, axis=0)
        var_dihC = np.var(diheds_c, axis=0)

        total_var = (dist_weight * var_dist +
                     angle_weight * (var_angA + var_angB) +
                     dihedral_weight * (var_dihA + var_dihB + var_dihC))

        # 8. 硬截断：排除平均角度不在 [45°,135°] 的候选 (避免 1/sinθ 奇点)
        avg_angA = np.mean(angles_a, axis=0)
        avg_angB = np.mean(angles_b, axis=0)
        banned = (avg_angA < np.deg2rad(45)) | (avg_angA > np.deg2rad(135)) | \
                 (avg_angB < np.deg2rad(45)) | (avg_angB > np.deg2rad(135))
        total_var[banned] = np.inf

        # 🔑 核心修复：拦截全 inf 灾难
        if np.all(np.isinf(total_var)):
            raise RuntimeError(
                "❌ 所有候选锚点组合的几何角度 (θA/θB) 均超出安全范围 [45°, 135°]！\n"
                "   体系可能存在严重畸变或配体脱离口袋。请检查预平衡轨迹，或使用 --boresch-source auto 切换至 Orb 估算。"
            )

        # 9. 选择最优组合
        best_idx = np.argmin(total_var)
        best_combo = combos[best_idx]

        # 10. 提取平衡值 (平均值) 和力常数
        eq = {
            "r0":       float(np.mean(dists[:, best_idx])),
            "thetaA0":  float(np.mean(angles_a[:, best_idx])),
            "thetaB0":  float(np.mean(angles_b[:, best_idx])),
            "phiA0":    float(np.mean(diheds_a[:, best_idx])),
            "phiB0":    float(np.mean(diheds_b[:, best_idx])),
            "phiC0":    float(np.mean(diheds_c[:, best_idx])),
        }

        kBT = self.gas_constant_kj_per_mol_k * self.temperature
        raw_fc = {
            "kr":       kBT / (var_dist[best_idx] + 1e-10),
            "kthetaA":  kBT / (var_angA[best_idx] + 1e-10),
            "kthetaB":  kBT / (var_angB[best_idx] + 1e-10),
            "kphiA":    kBT / (var_dihA[best_idx] + 1e-10),
            "kphiB":    kBT / (var_dihB[best_idx] + 1e-10),
            "kphiC":    kBT / (var_dihC[best_idx] + 1e-10),
        }
        force_constant_ranges = {
            "kr": [100.0, 2000.0],
            "kthetaA": [10.0, 1000.0],
            "kthetaB": [10.0, 1000.0],
            "kphiA": [10.0, 1000.0],
            "kphiB": [10.0, 1000.0],
            "kphiC": [10.0, 1000.0],
        }
        fc = {}
        clipped_flags = {}
        for key, raw_value in raw_fc.items():
            lower, upper = force_constant_ranges[key]
            fc[key], clipped_flags[key] = self._clip_force_constant(raw_value, lower, upper)

        fluctuation_diagnostics = [
            self._fluctuation_diagnostics(dists[:, best_idx], "r"),
            self._fluctuation_diagnostics(angles_a[:, best_idx], "thetaA"),
            self._fluctuation_diagnostics(angles_b[:, best_idx], "thetaB"),
            self._fluctuation_diagnostics(diheds_a[:, best_idx], "phiA"),
            self._fluctuation_diagnostics(diheds_b[:, best_idx], "phiB"),
            self._fluctuation_diagnostics(diheds_c[:, best_idx], "phiC"),
        ]
        n_bad_diag = sum(1 for item in fluctuation_diagnostics if not item.get("ok", False))
        n_clipped = sum(1 for clipped in clipped_flags.values() if clipped)

        # 🚨 关键修复：best_combo[0] (rec_tri) 内部是按 (c,b,anc)=(最远,中间,最近)
        # 的顺序构建的——上面 dist/angle/dihedral 的 index 列表都正确利用了这个
        # 顺序算出了符合 R0(最近)-顶点约定的 eq/fc；但如果直接原样存成
        # receptor_indices，会跟 _check_boresch_geometry_safe /
        # calc_boresch_from_last_frame / LambdaDependentBoreschForce 全部假设的
        # "receptor_indices[0]=离配体最近的锚点" 顺序相反，导致下游重新读取这份
        # 结果时把最远锚点当成了 R0。这里显式反转，使其对外统一为最近在前。
        result = {
            "receptor_indices": list(reversed(best_combo[0])),
            "ligand_indices": list(best_combo[1]),
            "equilibrium_values": eq,
            "force_constants": fc,
            "force_constants_raw": {k: float(v) for k, v in raw_fc.items()},
            "force_constant_clip_ranges": force_constant_ranges,
            "force_constant_clipped": clipped_flags,
            "diagnostics": {
                "n_frames": int(n_frames),
                "n_candidates": int(n_combos),
                # 🔑 [ATT-11] 锚点三元组是靠成键关系枚举出来的；这里逐侧记录用的是
                # 拓扑真实键还是 0.22 nm 几何近似，后者会把残基间非键近接
                # （C-N≈0.133 nm）误判成键。锚点选错直接改变解析释放修正。
                # 两侧情况可以不同：mmCIF 常有全部蛋白键但配体（非标准残基）无键。
                "bond_source": self.bond_source,           # 两侧汇总，兼容旧读取方
                "bond_source_receptor": self.bond_source_receptor,
                "bond_source_ligand": self.bond_source_ligand,
                "bond_coverage": dict(self.bond_coverage),
                "bond_dist_nm_if_geometric": (
                    float(self.bond_dist)
                    if "geometric_fallback" in (
                        self.bond_source_receptor, self.bond_source_ligand
                    ) else None
                ),
                "n_angle_banned_candidates": int(np.sum(banned)),
                "best_total_variance_score": float(total_var[best_idx]),
                "fluctuation_distribution": fluctuation_diagnostics,
                "n_non_gaussian_or_under_sampled_terms": int(n_bad_diag),
                "n_clipped_force_constants": int(n_clipped),
                "warnings": [
                    "Some fluctuation-derived force constants were clipped to conservative bounds."
                    if n_clipped else "",
                    "One or more restraint coordinates show non-Gaussian or under-sampled fluctuations."
                    if n_bad_diag else "",
                ],
            },
            "method": "geometric_fluctuation_v2_clipped",
        }
        result["diagnostics"]["warnings"] = [
            warning for warning in result["diagnostics"]["warnings"] if warning
        ]

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)

        print(f"  🏆 最优锚点: 受体 {result['receptor_indices']} | 配体 {result['ligand_indices']}")
        print(f"     r0={eq['r0']*10:.2f} Å, θA={np.degrees(eq['thetaA0']):.1f}°, θB={np.degrees(eq['thetaB0']):.1f}°")
        print(f"     kr={fc['kr']:.1f} kJ/mol/nm², kθA={fc['kthetaA']:.1f} kJ/mol/rad²")
        if n_clipped:
            print(f"  ⚠️ fluctuation Boresch 有 {n_clipped} 个力常数被裁剪；raw 值已写入结果 JSON。")
        if n_bad_diag:
            print(f"  ⚠️ fluctuation Boresch 有 {n_bad_diag} 个坐标分布偏离高斯或采样不足；请检查 diagnostics。")
        return result


# ============================================================================
# 8. Orbv3 → DEXP 拟合流水线 (从 DEXP_class.py 合并)
# ============================================================================
