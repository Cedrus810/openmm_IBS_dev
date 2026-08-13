#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ABFE 核心物理模块 (v6.0 - 完整收敛版)
职责：统一封装所有势能、限制力、拟合器、估算器、校验器、路径规划、替身构建、Orb扫描与路由工厂
架构约束：严格收敛至 5 文件，本文件为唯一物理核心单例，零占位符
依赖：openmm, numpy, scipy, mdtraj, torch, openmmml (部分功能)
"""

import os

# ============================================================================
# [P0-REMD-CUDA] 必须在**任何** pymbar/JAX import 之前执行。
#
# pymbar 4 的后端是 JAX，而 JAX 的默认行为是一碰 GPU 就**预分配整卡的 75%**
# (`XLA_PYTHON_CLIENT_PREALLOCATE=true` + `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`)。
#
# 实测后果（2026-08-04，`memtest/output_membrane_100ns`）：attachment 腿末尾用
# pymbar 解 BAR/MBAR，日志里 `JAX 64-bit mode is now on!` 之后紧跟着
#     📊 [显存] Stage 0 attachment 结束: used=12197 free=3646 total=16303 MiB
# 12197 / 16303 = **74.8%**，就是那个 0.75。于是 Stage 1 只剩 3646 MiB，
# 而 12 个 replica Context 需要 12 × 317 = 3804 MiB —— 建满 11 个后第 12 个抛
# `No compatible CUDA device is available`，整个 decharging 阶段静默退 CPU
# （慢约两个数量级：第 0 轮交换 29 分钟，500 轮约 10 天）。
#
# 这解释了此前所有反直觉现象：离线探针能建满 12 个（它不调 pymbar，JAX 从未初始化）、
# 更大的可溶体系反而成功、换全新进程照样失败（每个进程都会重新预分配）、
# 以及"约 10 GB 对不上账"——那 10 GB 就是 JAX 的预分配，不是 Context 泄漏。
#
# 只关预分配、**不**把 MBAR 挪到 CPU：JAX 仍在 GPU 上按需申请，数值路径不变
# （避免动已落盘的基线）。想更彻底地把显存全留给 OpenMM，可在外部导出
# `JAX_PLATFORMS=cpu` —— 那会改变 MBAR 的执行设备，属于需要单独验证的改动。
# 用 `setdefault`，所以外部显式设置的值优先。
# ============================================================================
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import openmm
from openmm import app, unit
import numpy as np
import math
import re
import warnings
import json
import logging
import gc
import builtins
import statistics
from itertools import combinations
from collections import deque
from typing import Dict, List, Tuple, Optional, Any, Callable, Sequence
from typing import OrderedDict as OrderedDictType
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

# [MEM-00h，2026-08-06] 传统 Beutler REMD 路径（BeutlerSoftcoreBuilder）的
# softcore cutoff——必须与 `ibs_engine.SOFTCORE_CUTOFF_NM` 一致（同一次决策：
# 统一到基础 NonbondedForce 的 1.0 nm、关闭 switching，不是拉长到 1.2）。
# abfe_core 在 ibs_engine 的下层不能反向 import，所以由
# tests/test_dispersion_and_forcefield_protocol.py 的交叉检查测试钉住，防止
# 两处各改一半。
BEUTLER_SOFTCORE_CUTOFF_NM = 1.0

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

# ✅ B3 已落地（2026-08-04）：charge-transfer 的 charging Hamiltonian
# （ligand q→0 与 co-ion 0→q 由同一个 `lam_coul` 反向驱动、co-ion 电荷走 PME
# particle offset）实现在 `ibs_engine.configure_charge_transfer_decharging`，
# restraint 换成 MEM-00d 的 flat-bottom 锚点相对形式。
# 证据：`tests/test_charge_transfer_hamiltonian.py`（§7.2 逐 λ 电荷守恒 / §7.3 co-ion 物理）。
CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED = True

# ✅ B4 已落地（2026-08-05）：`runabfe.build_and_cache_solvent_leg` 现在会在
# `charge_treatment=co_alchemical_charge_transfer` 且配体带净电荷时，额外插入
# `|q_L|` 个建系时预留的中性 ion-shaped dummy（§4.1），让
# `ibs_engine.select_co_alchemical_ion_once` 能在溶剂腿里认出它——复合物腿和
# 溶剂腿从此走同一条身份识别路径，热力学循环闭得上了。
#
# ⚠️ 这个 True 只代表"builder 会产出满足判据的粒子"，不代表已经在真实带电配体
# 体系上跑通验证过（本仓库当前的生产体系 Atenolol 净电荷为 0，这条路径测不出来）；
# §4.2/§4.4 的盒子尺寸敏感性、平衡稳定性仍待真正带电配体上机验证。
CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED = True

# C4/C5 是带电膜 charge-transfer 路线获得生产资格前的强制验收。底层
# Hamiltonian/双腿 builder 已实现并不等于这两项科学验收已经完成。
CHARGE_TRANSFER_FEATURE_STATUS = "experimental"
CHARGE_TRANSFER_PRODUCTION_QUALIFICATION_PROTOCOL_VERSION = 1
CHARGE_TRANSFER_C4_PASSED = False
CHARGE_TRANSFER_C5_PASSED = False

# [Stage2 handoff，2026-08-11] charge-transfer 配体的 charging→vanishing
# 交接：vanishing 阶段的输入 System 现在会先用
# `bake_global_parameter_into_fixed_nonbonded_force` 把一份独立的 charging
# 配置固化到 λ_coul=0 端点（配体 0 电荷、co-ion 满电、配体内部 exception
# 保留物理 chargeProd），再喂给 `build_ibs_dual_system`——不是直接把原始
# `self.system` 传过去（那样带净电配体会被 `build_ibs_dual_system` 自己的
# 电中性防御拒绝）。见 `STAGE2_CHARGE_TRANSFER_HANDOFF_PROPOSAL.md`。
#
# 这个协议版本号只在 `ABFEPipeline._charge_transfer_vanishing_handoff_active()`
# 返回 True（即 charge_treatment 是 charge-transfer 且配体净电荷非零）时才
# 写入 Stage 2 的缓存指纹——中性配体（当前唯一生产路径，Atenolol 净电荷为 0）
# 完全不受影响，指纹不变，旧缓存不会被误判失效。
CHARGE_TRANSFER_VANISHING_HANDOFF_PROTOCOL_VERSION = 1

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

# Candidate residue names used by both the solvent builder and the frozen
# identity layer.  A residue name alone never proves that a particle is a
# reserved dummy; callers must also verify its zero lambda=1 charge.
CO_ALCHEMICAL_ION_RESIDUE_NAMES = frozenset(
    {"CL", "CLA", "CL-", "NA", "NA+", "SOD", "K", "K+", "POT", "MG", "CA"}
)

CO_ALCHEMICAL_ION_BUILDER_IDENTITY_SCHEMA_VERSION = 1

# §5 / §1.2：选 Rocklin 路线时必须真的有 APBS 证据，不能只填一个数。
APBS_REQUIRED_EVIDENCE_FIELDS = (
    "manifest_path",
    "result_path",
    "dielectric_map_paths",
    "lipid_charge_map_path",
    "net_charge_e",
)


def charge_treatment_qualification_payload(charge_treatment: Optional[str]) -> Dict[str, Any]:
    """Return the single-source production-qualification payload.

    Neutral and Rocklin paths keep their legacy result shape. Charged
    charge-transfer is merged as an experimental capability: C4/C5 are not
    complete, so constructing a closed cycle cannot silently promote the
    numerical result to production-qualified.
    """
    resolved = str(charge_treatment or "").strip().lower()
    if resolved == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        c4_passed = bool(CHARGE_TRANSFER_C4_PASSED)
        c5_passed = bool(CHARGE_TRANSFER_C5_PASSED)
        return {
            "qualification_protocol_version": (
                CHARGE_TRANSFER_PRODUCTION_QUALIFICATION_PROTOCOL_VERSION
            ),
            "feature_status": CHARGE_TRANSFER_FEATURE_STATUS,
            "production_qualified": bool(c4_passed and c5_passed),
            "c4_passed": c4_passed,
            "c5_passed": c5_passed,
            "production_qualification_reason": (
                "charged_membrane_charge_transfer_requires_c4_and_c5"
            ),
        }
    if resolved == CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL:
        return {
            "qualification_protocol_version": (
                CHARGE_TRANSFER_PRODUCTION_QUALIFICATION_PROTOCOL_VERSION
            ),
            "feature_status": "experimental_method_comparison_only",
            "production_qualified": False,
            "c4_passed": False,
            "c5_passed": False,
            "production_qualification_reason": "co_annihilation_not_for_production",
        }
    return {}


def resolve_charge_treatment(
    charge_treatment: Optional[str],
    ligand_net_charge_e: float,
    apbs_correction_kJ_mol: float = 0.0,
    co_alchemical_ion: Optional[Any] = None,
    require_co_alchemical_ion: bool = False,
    environment_type: Optional[str] = None,
    apbs_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """校验净电荷处理协议，返回可直接进 fingerprint/provenance 的解析结果。

    `charge_treatment=None` 时按 §1.2 的生产默认推导：中性配体 → `neutral`，
    带电配体 → `co_alchemical_charge_transfer`。**这不是"猜协议"**——它只看配体
    净电荷这一个客观量，与清单禁止的"根据有没有 APBS 数值猜"是两回事。

    `co_alchemical_ion` 现在只接受一个**可选的兼容性 override**：真正的 B3–B5
    运行身份由 complex/solvent 两条 pipeline 各自选择并冻结，不能靠一个全局
    atom-index spec 作为前置放行条件。若调用方确实要在这个纯校验层验证一份
    外部 spec，可显式传入它；旧的“必须在前置阶段提供全局 spec”行为只有在
    `require_co_alchemical_ion=True` 时才启用。

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
            # B5：前置解析不再依赖一个全局 atom-index spec；真正的
            # co-ion 身份/参数/restraint 会在每条 pipeline 冻结后由 runtime
            # verifier fail closed。显式 override 仍可在这里做兼容性校验。
            if require_co_alchemical_ion and co_alchemical_ion is None:
                raise ValueError(
                    "charge_treatment=co_alchemical_charge_transfer 但没有提供 "
                    "co_alchemical_ion 身份、参数与 restraint（§1.2 fail-closed 第 3 条、§3.4）。"
                )
            if co_alchemical_ion is not None:
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
        payload["charging_hamiltonian_implemented"] = bool(
            CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED
        )
        # B4 状态如实落 provenance：一条 charge-transfer 运行现在能跑复合物腿（pilot），
        # 但溶剂腿 builder 没落地 ⟹ 循环闭不上 ⟹ 不得报出 ΔG_bind。
        payload["solvent_leg_builder_implemented"] = bool(
            CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED
        )
        payload["closes_thermodynamic_cycle"] = bool(
            CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED
        ) or is_experimental
        payload.update(charge_treatment_qualification_payload(resolved))
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
# MEM-00c：co-ion 身份冻结（B3 的前置条件，不是 B5 的缓存细节）
#
# 为什么必须冻结，而不是"每个入口自己选一次就行"：
# `ibs_engine._select_bulk_water_counterion` 按**传入坐标**当场排序挑离子
# （主键 = 到最近溶质的 minimum-image 距离，一个连续量），没有持久化身份、
# 也没有只读模式。而喂给它的坐标在**跨进程 resume** 时会变：
#   · 首跑：`pre_equilibrate()` 的输出**再叠 2000 步快速最小化**；
#   · resume（`skip_equil`）：直接读 `pre_equilibration.dcd` **末帧**，不做最小化。
# `tests/test_coalchemical_ion_identity.py` 实测 **0.05 nm 位移就足以翻转选择结果**，
# 而 2000 步最小化的原子位移正是 0.01–0.1 nm 量级。于是首跑用粒子 A 跑动力学、
# resume 进程用粒子 B 重算 u_kn —— u_kn 与动力学 Hamiltonian 静默不一致，
# ΔG 会错而没有任何异常现象。
#
# 落地形态（顺序不可颠倒）：
#     选一次 → 落成 spec（含指纹） → dynamics / replicas / u_kn / resume 全部只读消费
#     → 任何下游重选路径都不可达
#
# 字段直接复用 §3.4 的 `CO_ALCHEMICAL_ION_REQUIRED_FIELDS`，不另造 schema。
# ============================================================================

# v1 → v2（2026-08-04，MEM-00d + B3）：restraint 形式从"绝对笛卡尔参考点的纯谐振子"
# 换成"锚点相对的 flat-bottom"（见下方 §2.3/MEM-00d 一节），restraint 字典的键因此
# 整体改变。v1 的 spec 一律拒绝复用——它记的 `reference_frame` 是
# `absolute_cartesian_at_selection_time`，那个参考点在膜半各向异性 NPT 下会把离子拖向膜。
CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION = 2

# 指纹覆盖的字段 = "任一项变化都必须拒绝旧缓存"的那张表。少一项就等于允许
# 一个静默不一致的 resume：例如只钉 atom_index 而不钉 sigma/epsilon，
# 换了离子类型（Cl⁻→Br⁻）却仍复用旧 u_kn。
CO_ALCHEMICAL_ION_IDENTITY_FINGERPRINT_FIELDS = (
    "atom_index",       # 粒子 index
    "residue_index",    # 残基 index（index 相同但残基变了 = 拓扑换了）
    "residue_name",     # 离子类型
    "element",          # 离子类型
    "mass_amu",
    "sigma_nm",
    "epsilon_kj_mol",
    "charge_at_lambda1_e",  # 端点电荷
    "charge_at_lambda0_e",
    "restraint",        # restraint 形式 + 参考位置
)


# ============================================================================
# MEM-00d：co-ion restraint 的形式（§2.3 / §13.1）
#
# 旧形式（v1，已退役）：`0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2`，k = 25，
# 参考点是**选中那一刻的绝对笛卡尔坐标**。它有两个毛病：
#   1. 没有平坦区 —— §2.3 要求"平坦区足够大的 flat-bottom，避免把 co-ion 锁死在
#      一个异常水构象"；
#   2. 参考点在膜半各向异性 NPT 下不随盒缩放 —— Z 方向盒长变而参考点不动，
#      离子会被系统性地拖向膜。
#
# 新形式：**锚点相对**的 flat-bottom。参考点 = 锚点原子当前位置 + 冻结的位移向量 d0，
# 所以它跟着体系一起被 barostat 缩放（§2.3"可随盒缩放的定义"），而不是钉在盒坐标系里。
#
# 为什么用 `CustomCompoundBondForce` + `pointdistance` 而不是别的：
#   * `periodicdistance` **只存在于 CustomExternalForce**——实测
#     `CustomCentroidBondForce` 与 `CustomCompoundBondForce` 都报
#     `unknown function: periodicdistance`（2026-08-04 在 OpenMM 8.5.1 上逐个试过）。
#     而 CustomExternalForce 只能吃**绝对**参考点，正是要退役的那个形式。
#   * `CustomCompoundBondForce` 打开 PBC 后会把同一个 bond 里的粒子平移到与第一个
#     粒子相同的周期镜像再求值，所以 `pointdistance` 在其中**就是** minimum-image
#     距离。实测：离子 z=0.2、锚点 z=9.4、盒 z=12 → 得 0.2 nm（不是 9.2 nm）。
#   * 三斜/各向异性盒都走同一条 minimum-image 逻辑，无需自己写盒矩阵运算。
#
# 锚点取**配体重原子中离配体质心最近的那一个**，两条腿用同一条规则：
#   * 它随体系一起缩放，消掉 MEM-00d 的系统性拖拽；
#   * 它让"co-ion ↔ 配体 minimum-image 距离"在结构上被 restraint 本身钉住
#     （§13.1 的全程下限从"事后诊断"变成"构造时可证"）；
#   * 两条腿同一个锚点规则、同一个 k/r₀ ⟹ 可用体积相同，restraint 的自由能在
#     ΔG_solv − ΔG_cplx 里对消（§2.3 末条要求的说明，见 docs 的 MEM-00e 记录）；
#   * 取"离质心最近"而不是随便一个重原子：配体转动时该原子位移最小，井心抖动最小。
# ============================================================================

CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM = "flat_bottom_anchor_relative"
# ⚠️ 这个表达式与 `ibs_engine._create_co_alchemical_ion_restraint` 里实际注入的
# 必须逐字符相同 —— 它进身份指纹，所以"记录的形式"与"跑的形式"分叉就等于
# 让一份 spec 描述了另一个哈密顿量。有契约测试钉住两者相等。
CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION = (
    "0.5*k_ion*max(0, pointdistance(x1,y1,z1, x2+dx0, y2+dy0, z2+dz0) - r0_ion)^2"
)
CO_ALCHEMICAL_ION_RESTRAINT_REFERENCE_FRAME = "anchor_atom_relative_displacement"
# restraint 与 Boresch（force group 3）分开，且**逐 λ 完全相同**（§2.3、§6.4）。
CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP = 6
# 判"是重原子"的质量下限：H 是 1.008，D 是 2.014，最轻的重原子 C 是 12。
CO_ALCHEMICAL_ION_ANCHOR_MIN_MASS_AMU = 2.5

# 走出平坦区之后墙很软（k=100 时 0.316 nm 才 2 kT），所以"全程 ≥ 1.2 nm"这条
# 不能只按平坦区半径算，必须再留一段热涨落余量。取 2 kT 对应的位移：
#     0.5 * k * d² = 2 kT(300 K) = 4.988 kJ/mol  ⟹  d = 0.316 nm
CO_ALCHEMICAL_ION_RESTRAINT_THERMAL_MARGIN_KT = 2.0
CO_ALCHEMICAL_ION_RESTRAINT_REFERENCE_TEMPERATURE_K = 300.0


def co_alchemical_ion_restraint_wall_margin_nm(
    k_kj_per_mol_nm2: Optional[float] = None,
    margin_kt: float = CO_ALCHEMICAL_ION_RESTRAINT_THERMAL_MARGIN_KT,
    temperature_k: float = CO_ALCHEMICAL_ION_RESTRAINT_REFERENCE_TEMPERATURE_K,
) -> float:
    """平坦区之外还能走多远才付得起 `margin_kt` 个 kT（nm）。

    用途是把 §13.1 的"co-ion ↔ 配体全程 ≥ 1.2 nm"变成**构造时可证的几何条件**，
    而不是只靠事后逐帧诊断去发现它已经被违反了。

    `k_kj_per_mol_nm2=None` 时取 §13.1 的 `COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2`
    ——不写成默认参数值是因为那个常量在本文件里定义得更晚（§13 一节），
    默认参数在 def 执行时求值会直接 NameError。
    """
    k = (
        COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2
        if k_kj_per_mol_nm2 is None
        else float(k_kj_per_mol_nm2)
    )
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"restraint 力常数必须为正有限数，收到 {k_kj_per_mol_nm2!r}")
    kt = 0.008314462618 * float(temperature_k)
    return math.sqrt(2.0 * float(margin_kt) * kt / k)


def minimum_image_displacement_nm(displacement, box_vectors) -> np.ndarray:
    """三斜盒的 minimum-image 位移（行向量盒矩阵，nm）。

    这是本仓库 minimum-image 数学的**唯一**实现；`ibs_engine._minimum_image_displacement_nm`
    是它的薄包装（那一层负责把 OpenMM Quantity / list-of-Vec3 归一成 (3,3) 数组）。
    放在 abfe_core 是因为 co-ion 的几何判据（§13.1）在这一层就要用，而 abfe_core
    在 ibs_engine 下层、不能反向 import。
    """
    delta = np.asarray(displacement, dtype=np.float64)
    box = np.asarray(box_vectors, dtype=np.float64)
    if box.shape != (3, 3) or not np.all(np.isfinite(box)):
        raise ValueError("minimum-image 计算需要有限的 (3,3) 周期盒向量")
    det = float(np.linalg.det(box))
    if not np.isfinite(det) or abs(det) <= 1.0e-12:
        raise ValueError("周期盒向量奇异，无法计算 minimum-image 位移")
    fractional = delta @ np.linalg.inv(box)
    fractional -= np.round(fractional)
    return fractional @ box


def co_alchemical_ion_anchor_atom_index(
    *,
    system: Any,
    ligand_indices: Sequence[int],
    positions_nm: Sequence[Sequence[float]],
    box_vectors,
) -> Dict[str, Any]:
    """选 restraint 锚点原子：**离配体质心最近的配体重原子**（MEM-00d）。

    两条腿用同一条规则，规则本身与坐标无关（只有"哪一个最近"这一步看坐标），
    结果连同 minimum-image 位移一起冻结进 spec，此后只读核对。

    返回 dict：`anchor_atom_index` / `anchor_position_nm` / `ligand_extent_nm`
    （配体任一原子到锚点的最大 minimum-image 距离，供 §13.1 的几何余量判据用）。
    """
    indices = [int(i) for i in ligand_indices]
    if not indices:
        raise ValueError("配体原子列表为空，无法确定 co-ion restraint 锚点。")
    pos = np.asarray(positions_nm, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions_nm 形状非法：{pos.shape}")

    heavy = [
        i
        for i in indices
        if float(system.getParticleMass(i).value_in_unit(unit.dalton))
        >= CO_ALCHEMICAL_ION_ANCHOR_MIN_MASS_AMU
    ]
    if not heavy:
        raise ValueError(
            "配体没有质量 ≥ "
            f"{CO_ALCHEMICAL_ION_ANCHOR_MIN_MASS_AMU} amu 的重原子，无法选 restraint 锚点。"
        )

    # 质心用 minimum-image 相对第一个重原子累加，避免配体跨周期边界时质心跑到盒中间。
    origin = pos[heavy[0]]
    offsets = minimum_image_displacement_nm(pos[heavy] - origin, box_vectors)
    masses = np.asarray(
        [float(system.getParticleMass(i).value_in_unit(unit.dalton)) for i in heavy],
        dtype=np.float64,
    )
    centroid = origin + (offsets * masses[:, None]).sum(axis=0) / masses.sum()
    distances = np.linalg.norm(
        minimum_image_displacement_nm(pos[heavy] - centroid, box_vectors), axis=1
    )
    anchor_index = int(heavy[int(np.argmin(distances))])
    extent = float(
        np.max(
            np.linalg.norm(
                minimum_image_displacement_nm(
                    pos[indices] - pos[anchor_index], box_vectors
                ),
                axis=1,
            )
        )
    )
    return {
        "anchor_atom_index": anchor_index,
        "anchor_position_nm": [round(float(v), 6) for v in pos[anchor_index]],
        "ligand_centroid_nm": [round(float(v), 6) for v in centroid],
        "ligand_extent_from_anchor_nm": extent,
        "anchor_selection_rule": "ligand_heavy_atom_nearest_ligand_center_of_mass",
        "ligand_heavy_atom_count": int(len(heavy)),
    }


def co_alchemical_ion_restraint_spec(
    *,
    anchor: Dict[str, Any],
    reference_displacement_nm: Sequence[float],
    flat_bottom_radius_nm: Optional[float] = None,
    k_kj_per_mol_nm2: Optional[float] = None,
) -> Dict[str, Any]:
    """MEM-00d 的 restraint 描述（进身份指纹 ⟹ 改形式即作废旧 spec/缓存）。

    `reference_displacement_nm` 是"锚点 → co-ion"的 minimum-image 位移，在选择那一刻
    冻结。井心 = 锚点当前位置 + 该位移，所以它随体系一起被 barostat 缩放。
    """
    radius = (
        COION_FLAT_BOTTOM_RADIUS_NM
        if flat_bottom_radius_nm is None
        else float(flat_bottom_radius_nm)
    )
    k = (
        COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2
        if k_kj_per_mol_nm2 is None
        else float(k_kj_per_mol_nm2)
    )
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"flat-bottom 半径必须为正有限数，收到 {radius!r}")
    displacement = [round(float(v), 6) for v in list(reference_displacement_nm)[:3]]
    if len(displacement) != 3:
        raise ValueError(
            f"reference_displacement_nm 需要 3 个分量，收到 {reference_displacement_nm!r}"
        )
    return {
        "form": CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM,
        "expression": CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION,
        "reference_frame": CO_ALCHEMICAL_ION_RESTRAINT_REFERENCE_FRAME,
        "k_kj_per_mol_nm2": k,
        "flat_bottom_radius_nm": radius,
        "force_group": CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP,
        "anchor_atom_index": int(anchor["anchor_atom_index"]),
        "anchor_selection_rule": str(anchor["anchor_selection_rule"]),
        "reference_displacement_nm": displacement,
    }


def validate_co_alchemical_ion_placement(
    *,
    restraint: Dict[str, Any],
    ligand_extent_from_anchor_nm: float,
    strict: bool,
) -> Dict[str, Any]:
    """§13.1：flat-bottom 井是否**构造性地**保证"co-ion ↔ 配体全程 ≥ 1.2 nm"。

    判据（全部量都是 minimum-image）：

        |d0| − r₀ − wall_margin − ligand_extent  ≥  COION_LIGAND_MIN_IMAGE_RUNTIME_NM

    `|d0|` 是锚点到 co-ion 的冻结位移长度，`ligand_extent` 是配体任一原子到锚点的
    最大距离（锚点不是配体最外缘，所以必须减掉它），`wall_margin` 是走出平坦区
    2 kT 对应的位移（墙很软，只算平坦区会高估约束力）。

    `strict=False` 时只返回报告不 raise —— 那是 co-annihilation 实验对照专用路径
    （MEM-00a-2：只能用于水盒/lipid slab 方法对照，其数值不得进入任何 ΔG_bind 汇总），
    它的反离子是从既有盐里挑的物理离子、不是我们摆的，几何余量由诊断如实记录。
    ⚠️ charge-transfer 生产路线一律 `strict=True`。
    """
    d0 = np.asarray(restraint["reference_displacement_nm"], dtype=np.float64)
    d0_norm = float(np.linalg.norm(d0))
    radius = float(restraint["flat_bottom_radius_nm"])
    margin = co_alchemical_ion_restraint_wall_margin_nm(
        restraint.get("k_kj_per_mol_nm2")
    )
    extent = float(ligand_extent_from_anchor_nm)
    guaranteed = d0_norm - radius - margin - extent
    report = {
        "anchor_to_coion_distance_nm": d0_norm,
        "flat_bottom_radius_nm": radius,
        "wall_margin_nm": margin,
        "wall_margin_kt": CO_ALCHEMICAL_ION_RESTRAINT_THERMAL_MARGIN_KT,
        "ligand_extent_from_anchor_nm": extent,
        "guaranteed_min_ligand_distance_nm": guaranteed,
        "required_min_ligand_distance_nm": COION_LIGAND_MIN_IMAGE_RUNTIME_NM,
        "initial_threshold_nm": COION_LIGAND_MIN_IMAGE_INITIAL_NM,
        "satisfies_runtime_threshold": bool(
            guaranteed >= COION_LIGAND_MIN_IMAGE_RUNTIME_NM
        ),
        "satisfies_initial_threshold": bool(
            d0_norm - extent >= COION_LIGAND_MIN_IMAGE_INITIAL_NM
        ),
        "enforced": bool(strict),
    }
    if strict and not (
        report["satisfies_runtime_threshold"] and report["satisfies_initial_threshold"]
    ):
        raise ValueError(
            "co-ion 摆放不满足 §13.1 的几何余量：\n"
            f"    锚点↔co-ion 冻结距离 |d0| = {d0_norm:.3f} nm\n"
            f"    配体最外缘到锚点     = {extent:.3f} nm\n"
            f"    平坦区半径 r₀        = {radius:.3f} nm\n"
            f"    2 kT 软墙余量        = {margin:.3f} nm\n"
            f"    ⟹ 可保证的最小配体距离 = {guaranteed:.3f} nm"
            f"（要求 ≥ {COION_LIGAND_MIN_IMAGE_RUNTIME_NM} nm 全程、"
            f"初始 ≥ {COION_LIGAND_MIN_IMAGE_INITIAL_NM} nm）\n"
            "    正解是把 reserved co-ion 摆得更远（并保证 minimum-image 安全，§4.2），"
            "不是放宽本判据或缩小 flat-bottom 半径 —— 后者只是把违反藏进事后诊断。"
        )
    return report


# ============================================================================
# B3：charging λ 的逐粒子电荷映射（§2.1 / §2.4 / §7.2）
#
# OpenMM 的 `NonbondedForce.addParticleParameterOffset` 给出的是
#     q(λ) = q_base + λ · q_scale
# 而 `ibs_engine` 的 λ 约定是 `lam_coul` 从 1 → 0。于是两条共炼金路线只差
# "base/scale 怎么填"：
#
#   co-annihilation（实验对照，MEM-00a-2）：配体与**异号**反离子同步消电
#       ligand i : base 0,        scale q_i          ⟹ q(λ) = λ q_i
#       counterion: base 0,       scale q_phys       ⟹ q(λ) = λ q_phys
#
#   charge-transfer（生产，§2.1）：电荷从配体**搬到**体相水里的同号 co-ion
#       ligand i : base 0,        scale q_i          ⟹ q(λ) = λ q_i
#       co-ion j : base share_j,  scale −share_j     ⟹ q(λ) = (1−λ) share_j
#
# 两者的总电荷守恒条件是同一句话：**Σ scale = 0**。因为
#     Σq(λ) = Σq_base + λ · Σq_scale
# 所以 Σscale = 0 ⟺ 总电荷与 λ 无关 —— 这不是在几个 λ 点上抽查，而是对**所有** λ
# （含中间态，§7.2 要求）的一次代数证明。co-annihilation 满足它的方式是
# Σscale = q_L + (−q_L) = 0（异号反离子）；charge-transfer 是 q_L + (−q_L) = 0
# （配体升、co-ion 降）。
# ============================================================================


def co_alchemical_charge_offset_plan(
    *,
    charge_treatment: str,
    ligand_net_charge_e: int,
    ligand_charges_e: Dict[int, float],
    co_ion_physical_charges_e: Dict[int, float],
) -> Dict[str, Any]:
    """把"哪个粒子在 λ 上怎么变"算成 `{index: (base, scale)}`，并证明总电荷守恒。

    纯数学层：不碰 OpenMM 对象，所以可以在没有 GPU、没有真实体系的情况下逐 λ 验。
    `ibs_engine` 负责把结果写进 `NonbondedForce`，并在写完之后用同一份 plan 核对
    ——生产者与校验者共用同一份真相，不各写一套。
    """
    q_l = int(ligand_net_charge_e)
    if charge_treatment not in (
        CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
    ):
        raise ValueError(
            f"charge_treatment={charge_treatment!r} 不涉及共炼金离子，"
            "不该为它构造 charging 电荷计划。"
        )
    if q_l == 0:
        raise ValueError("配体净电荷为 0：中性路径不需要共炼金电荷计划。")

    plan: Dict[int, Tuple[float, float]] = {}
    for idx, q in ligand_charges_e.items():
        plan[int(idx)] = (0.0, float(q))

    ion_indices = sorted(int(i) for i in co_ion_physical_charges_e)
    if charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        if len(ion_indices) != abs(q_l):
            raise ValueError(
                f"charge-transfer 需要 {abs(q_l)} 个单价 co-ion（配体净电荷 {q_l:+d} e，"
                f"§2.2 禁止把多个单位电荷集中到一个非物理多价粒子上），"
                f"实际拿到 {len(ion_indices)} 个：{ion_indices}"
            )
        share = 1.0 if q_l > 0 else -1.0
        for idx in ion_indices:
            physical = float(co_ion_physical_charges_e[idx])
            if abs(physical) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
                raise ValueError(
                    f"reserved co-ion（粒子 {idx}）在输入体系里的电荷是 "
                    f"{physical:+.6f} e，不是 0。\n"
                    "    charge-transfer 要求 λ=1 端它是**中性但保留 LJ 的 "
                    "ion-shaped dummy**（§2.2），配体的净电荷则由**普通离子**在建系时"
                    "配平（§4.3）。\n"
                    "    换句话说：不能把一个已经带电的物理离子拿来当 co-ion —— 那样"
                    "λ=1 端的总电荷就不再是物理体系的总电荷了。\n"
                    "    正解是在建系时额外加入 "
                    f"{abs(q_l)} 个电荷为 0 的 ion-shaped 粒子（B4 的溶剂腿 builder "
                    "会这么做；复合物腿的输入拓扑必须自带）。"
                )
            plan[idx] = (share, -share)
    else:
        for idx in ion_indices:
            plan[idx] = (0.0, float(co_ion_physical_charges_e[idx]))

    base_sum = float(sum(base for base, _ in plan.values()))
    scale_sum = float(sum(scale for _, scale in plan.values()))
    if abs(scale_sum) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
        raise ValueError(
            f"charging 电荷计划的 Σscale = {scale_sum:+.6e} e ≠ 0，"
            f"容差 {TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:g} e。\n"
            "    Σq(λ) = Σq_base + λ·Σq_scale，所以 Σscale ≠ 0 就意味着总电荷随 λ 变 —— "
            "PME 会用一个逐 λ 变化的中和背景电荷把它掩盖掉，ΔG 静默出错（§7.2）。"
        )
    return {
        "charge_treatment": str(charge_treatment),
        "ligand_net_charge_e": q_l,
        "offsets": plan,
        "base_sum_e": base_sum,
        "scale_sum_e": scale_sum,
        "co_ion_indices": ion_indices,
        "total_charge_is_lambda_independent": True,
    }


def charge_at_lambda(base_e: float, scale_e: float, lam: float) -> float:
    """OpenMM particle-parameter-offset 的电荷模型：`q(λ) = base + λ·scale`。

    单独抽出来是为了让"期望值"只有一处定义 —— 测试、诊断与断言都调它，
    不各自手写一遍 `base + lam*scale`（写歪了就会自己对上自己）。
    """
    return float(base_e) + float(lam) * float(scale_e)


def _canonical_fingerprint(payload: Any) -> str:
    """稳定的 canonical-JSON sha256。dict 顺序、浮点格式都不能影响结果。"""
    import hashlib

    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _co_alchemical_ion_endpoint_charges(
    charge_treatment: str,
    physical_charge_e: float,
    ligand_net_charge_e: int,
    share_e: Optional[float] = None,
) -> Tuple[float, float]:
    """按所选电荷路线给出 co-ion 在 λ=1 / λ=0 的电荷。

    λ 约定与 `ibs_engine` 一致：`lam_coul` 从 **1 → 0**，粒子电荷走
    `addParticleParameterOffset`，即 q(λ) = λ · q_offset。

    * `co_annihilation_experimental`（`ibs_engine` 现存实现）：反离子是**异号**的，
      与配体**同步**消电 ⟹ λ=1 时是它的物理电荷、λ=0 时为 0。
    * `co_alchemical_charge_transfer`（B3 目标）：co-ion 是**同号**的中性 dummy，
      电荷由配体**转移**过来 ⟹ λ=1 时为 0、λ=0 时为 +q_L。

    两条路线的端点电荷不同，所以它进指纹：一份 co-annihilation 的 spec
    绝不能被一次声明 charge-transfer 的运行拿去复用。
    """
    if charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        # 中性 dummy → 接过配体电荷中属于**这一个粒子**的那一份。
        # `share_e` 由调用方按 §2.2 拆成 |q_L| 个单价份额；不传时退回单 co-ion 情形
        # （此时份额就是整份净电荷）。⚠️ 不要在这里默默把 |q_L|>1 塞给一个粒子——
        # 那正是 §2.2 禁止的非物理多价 co-ion。
        if share_e is None:
            return 0.0, float(ligand_net_charge_e)
        return 0.0, float(share_e)
    if charge_treatment == CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL:
        return float(physical_charge_e), 0.0
    raise ValueError(
        f"charge_treatment={charge_treatment!r} 不涉及 co-alchemical ion，"
        "不应该为它构造 co-ion 身份 spec。"
    )


def build_co_alchemical_ion_identity(
    *,
    system: Any,
    topology: Any,
    ion_atom_indices: Sequence[int],
    ligand_indices: Sequence[int],
    positions_nm: Sequence[Sequence[float]],
    box_vectors: Any,
    ligand_net_charge_e: int,
    charge_treatment: str,
    lambda_direction: str = "lam_coul_1_to_0",
    selection_provenance: Optional[Dict[str, Any]] = None,
    flat_bottom_radius_nm: Optional[float] = None,
    k_kj_per_mol_nm2: Optional[float] = None,
    enforce_placement_thresholds: Optional[bool] = None,
) -> Dict[str, Any]:
    """把"这一次选出来的 co-ion"固化成可落盘、可跨进程核对的身份 spec。

    只在**选择发生的那一次**调用。此后所有消费者（动力学 System 构建、REMD 副本、
    `compute_u_kn`、resume）都只读这份 spec，不再调用选择器。

    restraint 的**形式不接受调用方指定**：它由 `co_alchemical_ion_restraint_spec()`
    唯一生成（MEM-00d 的 flat-bottom + 锚点相对），调用方只能覆盖 k 与平坦区半径。
    这样就不可能出现"spec 里记着一种形式、实际注入另一种"的分叉。参考量（锚点
    index + 冻结位移）属于身份的一部分，所以整份 restraint 进指纹。

    `enforce_placement_thresholds` 默认按路线取：charge-transfer（生产）强制 §13.1
    几何余量，co-annihilation（实验对照）只记录诊断。见
    `validate_co_alchemical_ion_placement` 的 docstring。
    """
    import openmm

    nb_force = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None
    )
    if nb_force is None:
        raise RuntimeError("构造 co-ion 身份需要 NonbondedForce，但 System 里没有。")

    indices = [int(i) for i in ion_atom_indices]
    if not indices:
        raise ValueError(
            "ion_atom_indices 为空。中性配体不需要 co-ion —— 那种情况不该走到这里"
            "（调用方应当在 ligand_net_charge_e == 0 时完全跳过身份构造）。"
        )
    if len(indices) != len(set(indices)):
        raise ValueError(f"ion_atom_indices 有重复：{indices}")

    q_l = int(ligand_net_charge_e)
    if charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER and len(
        indices
    ) != abs(q_l):
        raise ValueError(
            f"charge-transfer 需要 {abs(q_l)} 个单价 co-ion（配体净电荷 {q_l:+d} e，"
            f"§2.2），实际给了 {len(indices)} 个：{indices}"
        )

    pos_nm = np.asarray(positions_nm, dtype=np.float64)
    anchor = co_alchemical_ion_anchor_atom_index(
        system=system,
        ligand_indices=ligand_indices,
        positions_nm=pos_nm,
        box_vectors=box_vectors,
    )
    strict = (
        charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
        if enforce_placement_thresholds is None
        else bool(enforce_placement_thresholds)
    )
    share = (1.0 if q_l > 0 else -1.0) if q_l else 0.0

    atoms = list(topology.atoms())
    ions: List[Dict[str, Any]] = []
    for position, index in enumerate(indices):
        if not 0 <= index < len(atoms):
            raise ValueError(
                f"co-ion atom_index={index} 越界（拓扑只有 {len(atoms)} 个原子）。"
            )
        atom = atoms[index]
        q, sigma, epsilon = nb_force.getParticleParameters(index)
        physical_charge = q.value_in_unit(unit.elementary_charge)
        q1, q0 = _co_alchemical_ion_endpoint_charges(
            charge_treatment, physical_charge, q_l, share_e=share
        )
        displacement = minimum_image_displacement_nm(
            pos_nm[index] - np.asarray(anchor["anchor_position_nm"], dtype=np.float64),
            box_vectors,
        )
        ion_restraint = co_alchemical_ion_restraint_spec(
            anchor=anchor,
            reference_displacement_nm=displacement,
            flat_bottom_radius_nm=flat_bottom_radius_nm,
            k_kj_per_mol_nm2=k_kj_per_mol_nm2,
        )
        # 选择那一刻的绝对坐标：**只作审计记录**，不再被任何 restraint 消费。
        # 刻意不叫 `reference_position_nm`（v1 的键名）—— 那样一个还在读旧键的消费者
        # 会静默拿到已退役的绝对参考点；改了名字它会 KeyError，这是有意的。
        ion_restraint["selection_time_absolute_position_nm"] = [
            round(float(v), 6) for v in pos_nm[index]
        ]
        placement = validate_co_alchemical_ion_placement(
            restraint=ion_restraint,
            ligand_extent_from_anchor_nm=anchor["ligand_extent_from_anchor_nm"],
            strict=strict,
        )
        ions.append(
            {
                "atom_index": int(index),
                "placement_diagnostics": placement,
                "residue_index": int(atom.residue.index),
                "residue_name": str(atom.residue.name),
                "element": str(getattr(atom.element, "symbol", "") or ""),
                "charge_at_lambda1_e": float(q1),
                "charge_at_lambda0_e": float(q0),
                "sigma_nm": float(sigma.value_in_unit(unit.nanometer)),
                "epsilon_kj_mol": float(
                    epsilon.value_in_unit(unit.kilojoule_per_mole)
                ),
                "mass_amu": float(
                    system.getParticleMass(index).value_in_unit(unit.dalton)
                ),
                "restraint": ion_restraint,
                # 诊断字段（不进指纹）：物理电荷用于事后核对端点推导是否合理。
                "physical_charge_e": float(physical_charge),
                "spec_position": int(position),
            }
        )

    spec: Dict[str, Any] = {
        "protocol_version": CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION,
        "charge_treatment": str(charge_treatment),
        "lambda_direction": str(lambda_direction),
        "ligand_net_charge_e": int(ligand_net_charge_e),
        "ions": ions,
        "selection_provenance": dict(selection_provenance or {}),
    }
    spec["fingerprint"] = co_alchemical_ion_identity_fingerprint(spec)
    return spec


def co_alchemical_ion_identity_fingerprint(spec: Dict[str, Any]) -> str:
    """身份指纹：只吃"变了就必须作废旧缓存"的那些字段。

    刻意**不吃** `selection_provenance`（当时的排序距离、水配位数等诊断量）——
    那些数每次读坐标都会变一点，进指纹会让每次 resume 都误判成身份漂移。
    """
    payload = {
        "protocol_version": int(spec["protocol_version"]),
        "charge_treatment": str(spec["charge_treatment"]),
        "lambda_direction": str(spec["lambda_direction"]),
        "ligand_net_charge_e": int(spec["ligand_net_charge_e"]),
        "ions": [
            {field: ion[field] for field in CO_ALCHEMICAL_ION_IDENTITY_FINGERPRINT_FIELDS}
            for ion in spec["ions"]
        ],
    }
    return _canonical_fingerprint(payload)


def verify_co_alchemical_ion_identity(
    spec: Dict[str, Any],
    *,
    system: Any,
    topology: Any,
    charge_treatment: Optional[str] = None,
    ligand_net_charge_e: Optional[int] = None,
    context: str = "",
) -> List[int]:
    """按 spec 核对当前 System/拓扑，通过则返回被钉住的 atom_index 列表。

    **只读**：任何不符都 raise，绝不"重新选一个能对上的"。这就是 MEM-00c 的修法——
    把"每个入口自己选"换成"选一次 + 处处核对"。

    返回的 index 列表可直接交给 `ibs_engine` 的 offset 注入路径使用。
    """
    import openmm

    where = f"（{context}）" if context else ""
    if not isinstance(spec, dict) or "ions" not in spec:
        raise ValueError(f"co-ion 身份 spec 结构非法{where}：{type(spec).__name__}")

    version = int(spec.get("protocol_version", -1))
    if version != CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION:
        raise ValueError(
            f"co-ion 身份 spec 的 protocol_version={version}{where}，"
            f"当前实现是 {CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION}。"
            "协议版本变了就意味着字段含义可能变了，拒绝复用旧 spec；请重新选择并落盘。"
        )

    recomputed = co_alchemical_ion_identity_fingerprint(spec)
    if recomputed != spec.get("fingerprint"):
        raise ValueError(
            f"co-ion 身份 spec 自身指纹不符{where}："
            f"记录 {spec.get('fingerprint')}，重算 {recomputed}。"
            "spec 被手工改过或写坏了 —— 不接受，请重新选择并落盘。"
            "（不要手改 spec 让它对上；那等于把不一致藏起来。）"
        )

    if charge_treatment is not None and str(charge_treatment) != spec["charge_treatment"]:
        raise ValueError(
            f"co-ion 身份 spec 记录的电荷路线是 {spec['charge_treatment']!r}，"
            f"本次运行声明的是 {charge_treatment!r}{where}。"
            "两条路线的端点电荷不同（co-annihilation 是 q_phys→0，"
            "charge-transfer 是 0→q_L），spec 不可跨路线复用。"
        )
    if (
        ligand_net_charge_e is not None
        and int(ligand_net_charge_e) != int(spec["ligand_net_charge_e"])
    ):
        raise ValueError(
            f"co-ion 身份 spec 是为配体净电荷 {spec['ligand_net_charge_e']:+d} e 选的，"
            f"当前体系是 {int(ligand_net_charge_e):+d} e{where}。"
        )

    nb_force = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None
    )
    if nb_force is None:
        raise RuntimeError(f"核对 co-ion 身份需要 NonbondedForce，但 System 里没有{where}。")

    atoms = list(topology.atoms())
    pinned: List[int] = []
    for ion in spec["ions"]:
        index = int(ion["atom_index"])
        if not 0 <= index < len(atoms):
            raise ValueError(
                f"co-ion atom_index={index} 越界{where}（拓扑只有 {len(atoms)} 个原子）——"
                "拓扑变了，旧身份不可用。"
            )
        atom = atoms[index]
        q, sigma, epsilon = nb_force.getParticleParameters(index)

        def _check(field: str, actual: Any, tolerance: Optional[float] = None) -> None:
            expected = ion[field]
            if tolerance is None:
                same = str(expected) == str(actual)
            else:
                same = abs(float(expected) - float(actual)) <= tolerance
            if not same:
                raise ValueError(
                    f"co-ion 身份漂移{where}：atom_index={index} 的 {field} "
                    f"记录为 {expected!r}，当前 System/拓扑给的是 {actual!r}。\n"
                    "    这正是 MEM-00c 要拦的那类静默不一致（u_kn 与动力学"
                    "Hamiltonian 用了不同粒子/参数）。\n"
                    "    正解是**重跑**这条腿并重新选择 co-ion；"
                    "**不要**放宽本核对或改 spec 让它对上。"
                )

        _check("residue_index", int(atom.residue.index))
        _check("residue_name", str(atom.residue.name))
        _check("element", str(getattr(atom.element, "symbol", "") or ""))
        _check("sigma_nm", sigma.value_in_unit(unit.nanometer), 1.0e-9)
        _check(
            "epsilon_kj_mol", epsilon.value_in_unit(unit.kilojoule_per_mole), 1.0e-9
        )
        _check(
            "mass_amu",
            system.getParticleMass(index).value_in_unit(unit.dalton),
            1.0e-6,
        )
        # 端点电荷：核对的是"当前 System 里这个粒子的电荷仍然支持记录的端点"。
        # 注意此处的 System 可能已经被 offset 改写过——那时粒子的**基**电荷是：
        #   * co-annihilation：0（真实电荷整份挪进 ParameterOffset）；
        #   * charge-transfer：share（= λ=0 端电荷，offset 里放 −share）。
        # 所以合法读数是一个小集合，逐项列出来而不是"够小就放过"。
        current_q = q.value_in_unit(unit.elementary_charge)
        expected_physical = float(ion.get("physical_charge_e", current_q))
        acceptable = [expected_physical, 0.0]
        if spec["charge_treatment"] == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
            acceptable.append(float(ion["charge_at_lambda0_e"]))
            # charge-transfer 的 co-ion 必须是**中性 dummy**（§2.2）。这一项不在指纹里，
            # 所以只有在这儿核对才能拦住手改过的 spec。
            if abs(expected_physical) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
                raise ValueError(
                    f"co-ion 身份非法{where}：atom_index={index} 记录的物理电荷是 "
                    f"{expected_physical:+.6f} e，而 charge-transfer 要求 λ=1 端它是"
                    "中性但保留 LJ 的 ion-shaped dummy（§2.2）。"
                    "带电的物理离子不能当 co-ion —— 那样 λ=1 端总电荷就不是物理体系的了。"
                )
        if all(abs(current_q - value) > 1.0e-6 for value in acceptable):
            raise ValueError(
                f"co-ion 身份漂移{where}：atom_index={index} 的物理电荷记录为 "
                f"{expected_physical:+.6f} e，当前 System 给的是 {current_q:+.6f} e"
                f"（本路线可接受的基电荷读数：{acceptable}）。"
                "离子类型变了 —— 旧 u_kn 与旧动力学都不可复用。"
            )

        # restraint：spec 记录的形式必须就是当前实现会注入的那一个（MEM-00d）。
        # 形式对不上意味着这份 spec 描述的是另一个哈密顿量。
        restraint = ion.get("restraint") or {}
        if restraint.get("form") != CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM:
            raise ValueError(
                f"co-ion restraint 形式不符{where}：spec 记的是 {restraint.get('form')!r}，"
                f"当前实现是 {CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM!r}（MEM-00d："
                "旧的『绝对笛卡尔参考点纯谐振子』已退役，它在膜半各向异性 NPT 下会把"
                "离子系统性拖向膜）。请重新选择 co-ion 并落盘新 spec。"
            )
        if restraint.get("expression") != CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION:
            raise ValueError(
                f"co-ion restraint 表达式不符{where}：spec 记的是 "
                f"{restraint.get('expression')!r}。它必须与实际注入的逐字符相同。"
            )
        anchor_index = int(restraint.get("anchor_atom_index", -1))
        if not 0 <= anchor_index < len(atoms):
            raise ValueError(
                f"co-ion restraint 锚点 atom_index={anchor_index} 越界{where}"
                f"（拓扑只有 {len(atoms)} 个原子）——拓扑变了，旧身份不可用。"
            )
        pinned.append(index)

    if len(pinned) != len(set(pinned)):
        raise ValueError(f"co-ion 身份 spec 出现重复 atom_index{where}：{pinned}")
    return pinned


def co_alchemical_ion_cache_identity_payload(
    spec: Optional[Dict[str, Any]],
    *,
    system: Any,
    topology: Any,
    leg: str,
    spec_relative_path: str,
    charge_treatment: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the minimal frozen co-ion identity consumed by cache protocols.

    This is deliberately a thin read-only adapter around the one canonical
    spec verifier/fingerprint.  It never selects an ion, never reads
    coordinates, and never includes ``selection_provenance`` or absolute
    positions.  Neutral legs return ``None`` so their legacy cache semantics
    remain unchanged.
    """
    requested_treatment = (
        None if charge_treatment is None else str(charge_treatment)
    )
    if spec is None:
        if requested_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
            raise ValueError(
                "co-ion runtime identity 缺少 charge-transfer spec："
                f"leg={leg!r}, path={spec_relative_path!r}"
            )
        return None
    if not isinstance(spec, dict):
        raise ValueError(
            f"co-ion runtime identity for leg={leg!r} has invalid spec type "
            f"{type(spec).__name__}; spec={spec_relative_path}"
        )
    resolved_leg = str(leg)
    if resolved_leg not in {"complex", "solvent"}:
        raise ValueError(f"co-ion runtime identity has unknown leg={resolved_leg!r}")
    try:
        q_l = int(spec["ligand_net_charge_e"])
        treatment = str(spec["charge_treatment"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"co-ion runtime identity spec is incomplete for leg={resolved_leg!r}: "
            f"{spec_relative_path}"
        ) from exc
    if requested_treatment is not None and requested_treatment != treatment:
        raise ValueError(
            "co-ion runtime identity charge_treatment 不匹配："
            f"当前运行={requested_treatment!r}，spec={treatment!r}，"
            f"leg={resolved_leg!r}, path={spec_relative_path!r}"
        )
    verify_co_alchemical_ion_identity(
        spec,
        system=system,
        topology=topology,
        # Always pass the route requested by the current caller.  When a
        # legacy direct caller omits it, None deliberately means "no route
        # assertion"; never substitute the spec's self-declaration here.
        charge_treatment=requested_treatment,
        ligand_net_charge_e=q_l,
        context=f"B5 runtime identity leg={resolved_leg} path={spec_relative_path}",
    )
    return {
        "schema_version": 1,
        "leg": resolved_leg,
        "charge_treatment": treatment,
        "identity_protocol_version": int(spec["protocol_version"]),
        "fingerprint": str(spec["fingerprint"]),
        "ligand_net_charge_e": q_l,
        "lambda_direction": str(spec["lambda_direction"]),
        "ion_atom_indices": [int(ion["atom_index"]) for ion in spec["ions"]],
        "spec_relative_path": os.path.normpath(str(spec_relative_path)),
    }


def co_alchemical_ion_builder_identity_payload(
    *,
    system: Any,
    topology: Any,
    charge_treatment: Optional[str],
    ligand_net_charge_e: int,
) -> Optional[Dict[str, Any]]:
    """Recompute the coordinate-free identity of reserved co-ion dummies.

    The identity is derived from the current base System/Topology, never
    copied back from a manifest.  It is intentionally separate from the
    runtime spec fingerprint: the builder identity answers *which dummy
    particles were built*, while the runtime identity answers *which frozen
    spec was used by a leg*.
    """
    treatment = None if charge_treatment is None else str(charge_treatment)
    if treatment != CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        return None
    q_l = int(ligand_net_charge_e)
    if q_l == 0:
        return None
    try:
        import openmm
        nb_force = next(
            force for force in system.getForces()
            if isinstance(force, openmm.NonbondedForce)
        )
    except (StopIteration, AttributeError) as exc:
        raise RuntimeError(
            "无法重算 reserved co-ion builder identity：System 缺少 NonbondedForce"
        ) from exc

    atoms = list(topology.atoms())
    candidates = []
    for atom in atoms:
        if str(atom.residue.name).upper() not in CO_ALCHEMICAL_ION_RESIDUE_NAMES:
            continue
        charge, sigma, epsilon = nb_force.getParticleParameters(int(atom.index))
        charge_e = float(charge.value_in_unit(unit.elementary_charge))
        if abs(charge_e) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            continue
        candidates.append(
            {
                "atom_index": int(atom.index),
                "residue_index": int(atom.residue.index),
                "residue_name": str(atom.residue.name),
                "element": str(getattr(atom.element, "symbol", "") or ""),
                "charge_at_lambda1_e": charge_e,
                "sigma_nm": float(sigma.value_in_unit(unit.nanometer)),
                "epsilon_kj_mol": float(
                    epsilon.value_in_unit(unit.kilojoule_per_mole)
                ),
                "mass_amu": float(
                    system.getParticleMass(int(atom.index)).value_in_unit(unit.dalton)
                ),
            }
        )
    candidates.sort(key=lambda item: item["atom_index"])
    expected_count = abs(q_l)
    if len(candidates) != expected_count:
        raise ValueError(
            "reserved co-ion builder identity 不满足 charge-transfer 数量契约："
            f"leg system 中找到 {len(candidates)} 个中性 ion-shaped dummy，"
            f"但配体净电荷 {q_l:+d} e 需要 {expected_count} 个。"
        )
    return {
        "schema_version": CO_ALCHEMICAL_ION_BUILDER_IDENTITY_SCHEMA_VERSION,
        "charge_treatment": treatment,
        "ligand_net_charge_e": q_l,
        "reserved_coion_count": expected_count,
        "ions": candidates,
        "restraint_protocol": {
            "protocol_version": int(CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION),
            "form": CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM,
            "reference_frame": CO_ALCHEMICAL_ION_RESTRAINT_REFERENCE_FRAME,
            "expression": CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION,
            "default_k_kj_per_mol_nm2": float(COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2),
            "default_r0_nm": float(COION_FLAT_BOTTOM_RADIUS_NM),
            "force_group": int(CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP),
        },
    }


# Descriptive alias used by cache/provenance callers and tests.
reserved_coion_builder_identity_payload = co_alchemical_ion_builder_identity_payload


# ============================================================================
# §13 验收阈值：全部落成命名常量并进 provenance
#
# 清单原文通篇写"预设阈值"但没给数——没有数就写不了 fail-closed 检查，也没法判
# 验收。§13 要求"必须在 Phase A 结束前落成常量并进 provenance，不许运行时凭感觉判"。
# 这里就是那份常量表；数值取 §13 的提案值，可改，但改必须改这里、且会进 provenance。
# ============================================================================

ACCEPTANCE_THRESHOLDS_VERSION = 2

# ---- §13.1 co-ion 几何 ----
COION_LIGAND_MIN_IMAGE_INITIAL_NM = 1.6
# [MEM-00h，2026-08-06 复核后拍板] 这个值**不再**要求等于
# `ibs_engine.SOFTCORE_CUTOFF_NM`——早先的注释/交叉检查测试把它当成 softcore
# cutoff 的派生量，但用户明确决定两者解耦：这是一条独立的、保守的几何安全
# 门（防止 co-ion 在动力学里游到离配体太近的地方），不是"必须等于 softcore
# cutoff 才有意义"的量。softcore cutoff 这次统一收敛到 1.0 nm（见
# ibs_engine.SOFTCORE_CUTOFF_NM），但这条门槛保持 1.2 nm 不变——没必要为了
# 那次 cutoff 收敛顺带把这里也降到 1.0，1.2 本来就更保守。
# 见 tests/test_dispersion_and_forcefield_protocol.py 里的对应测试
# （已改成断言"独立常量、不要求相等"，不再断言相等）。
COION_LIGAND_MIN_IMAGE_RUNTIME_NM = 1.2
# §2.2 多个单价 co-ion 分摊（|q_L| ≥ 2）：这些 dummy 彼此之间也必须留够安全边距。
# 2026-08-06 用 Ca²⁺(+2) 在真实水盒里第一次测这条路径就抓到：
# `runabfe._insert_reserved_coalchemical_ion_dummies` 原来只按"离配体质心最远"
# 独立给每个 dummy 打分，方盒里"离中心最远的 N 个点"天然会挤在同一个远角——
# 实测两个 dummy 相距 0.18~0.43 nm，λ→0 时两者同号各带 +1e，几乎贴脸的静电
# 排斥直接把 charging MBAR 拖到不收敛（ΔG≈660 kJ/mol，min_overlap<0.02）。
# 阈值取与 COION_LIGAND_MIN_IMAGE_INITIAL_NM 相同量级，不是拍脑袋另设一档。
COION_COION_MIN_IMAGE_INITIAL_NM = 1.6
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

# ---- 预平衡轨迹的时间轴（生产者与消费者共用同一组数）----
#
# 为什么必须共用：mdtraj 读 DCD 时**不传播真实步长**，`traj.time` 是整数帧号，
# 所以 §9 的时间轴只能由"reporter 保存间隔 × integrator 步长"重建。这两个数
# 一个写在 `pre_equilibrate()` 的 DCDReporter 里、一个写在它的 Integrator 里，
# 各自散着就必然错开（实测 memtest 那条 10 ns 轨迹被当成 0.499 ns，差 20 倍）。
PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS = 10000
PRE_EQUILIBRATION_TIMESTEP_PS = 0.002


def pre_equilibration_frame_interval_ps(
    timestep_ps: Optional[float] = None,
    interval_steps: Optional[int] = None,
) -> float:
    """预平衡 DCD 每帧对应多少 ps。默认值即生产设置（10000 × 2 fs = 20 ps）。"""
    dt = PRE_EQUILIBRATION_TIMESTEP_PS if timestep_ps is None else float(timestep_ps)
    steps = (
        PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS
        if interval_steps is None
        else int(interval_steps)
    )
    if not np.isfinite(dt) or dt <= 0.0 or steps <= 0:
        raise ValueError(
            f"预平衡帧间隔参数非法：timestep_ps={dt!r}, interval_steps={steps!r}"
        )
    return dt * steps


# ---- §13.3 膜质量门（判据统一为"末段窗口内线性漂移小于阈值"）----
MEMBRANE_QUALITY_GATE_TAIL_WINDOW_NS = 20.0
APL_MAX_DRIFT_PERCENT_PER_NS = 0.2
APL_MAX_DEVIATION_FROM_LITERATURE_PERCENT = 3.0

# ---- 含蛋白膜的 APL：必须先扣掉蛋白横截面才能与纯脂文献值比 ----
#
# 实测 memtest（PROA 1 + POPC 90，08-02 那条 10 ns 轨迹）：
# raw APL = 横向面积 / 每叶脂质数 = 0.807 nm²，而 POPC 纯脂文献值 ≈ 0.645。
# 差值来自跨膜蛋白占掉的横向面积，**不是**体系有问题。所以 §13.3 那条
# 「与文献值差 ≤ 3%」若拿 raw APL 去比，会把物理上正常的含蛋白膜判失败；
# 先前的应对是干脆不设 literature_apl_nm2，代价是这道门整条缺席。
#
# ## 为什么是"最近原子归属"而不是"蛋白外扩一个探针半径"
#
# 第一版用的是"每个蛋白重原子外扩 0.17 nm 求并集"。实测它**系统性高估蛋白面积**：
# 外扩会沿蛋白周长加一圈宽 0.17 nm 的边（周长约 10 nm → 约 1.7 nm²），
# 于是校正后 APL = 0.564，比文献值低 12.6% —— 门会因为**方法偏差**而不过，
# 而不是因为体系有问题。那种门迟早会被"调参调绿"，正是本仓库反复吃亏的模式。
#
# 现在改为**无探针半径**的定义（Voronoi 式，APL@Voro 的同一思路）：把该叶片
# slab 内的脂质重原子与蛋白重原子放在一起，横向栅格的每个格子归给**最近的**那个
# 原子，蛋白面积 = 归给蛋白原子的格子面积之和。边界自动落在两者中间，没有可调的
# 半径；唯一的方法参数是栅格边长，而它只带来无偏的离散化误差。
PROTEIN_CROSS_SECTION_GRID_NM = 0.05
# 栅格边长的敏感性检查（用 2× 粗栅格复算若干帧，确认离散化误差可忽略）。
PROTEIN_CROSS_SECTION_GRID_SENSITIVITY_FACTOR = 2.0
# 敏感性只在若干等间隔帧上算（主序列仍逐帧），避免为一个诊断量把成本翻倍。
PROTEIN_CROSS_SECTION_SENSITIVITY_MAX_FRAMES = 25
# 校正后 APL 的观测量名。**不进** REQUIRED_MEMBRANE_QUALITY_OBSERVABLES：
# 判定层要能吃老报告和手工构造的观测量；缺它时绝对值门会退回未校正值，
# 但会把 criterion 写成 `..._uncorrected`，事后分得清当时比的是什么。
APL_PROTEIN_CORRECTED_OBSERVABLE = "apl_protein_corrected_nm2"

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
            "coion_coion_min_image_initial_nm": COION_COION_MIN_IMAGE_INITIAL_NM,
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
            # 蛋白横截面校正的方法参数：绝对值门比的是校正后的 APL，
            # 而校正结果依赖这三个数，所以它们必须与阈值一起进 provenance。
            "protein_cross_section_grid_nm": PROTEIN_CROSS_SECTION_GRID_NM,
            "protein_cross_section_grid_sensitivity_factor": (
                PROTEIN_CROSS_SECTION_GRID_SENSITIVITY_FACTOR
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


# `[ defaults ]` 的 1-4 缩放约定是比 include 路径**更可靠**的力场族判据：
#   Amber  : comb-rule 2, gen-pairs yes, fudgeLJ 0.5, fudgeQQ 0.8333
#   CHARMM : comb-rule 2, gen-pairs yes, fudgeLJ 1.0, fudgeQQ 1.0
# 实测 memtest/ 那套 CHARMM-GUI FF-Converter 产的 AMBER 体系里，include 路径全是
# `toppar/*.itp`（**不含 amber 字样**），唯一可靠证据就在
# `toppar/forcefield.itp` 的 `[ defaults ]`：`1 2 yes 0.500000 0.833333`。
# 所以 §1.1 原文写的是"从 `#include` **与** `[ defaults ]` 判定"——两者都要。
_FORCEFIELD_FAMILY_FUDGE_SIGNATURES = (
    (FORCEFIELD_FAMILY_AMBER, 0.5, 1.0 / 1.2),
    (FORCEFIELD_FAMILY_CHARMM, 1.0, 1.0),
)
_FUDGE_MATCH_TOLERANCE = 1.0e-3


def resolve_gromacs_include(
    include_name: str,
    including_file: str,
    gmx_include_dir: Optional[str] = None,
) -> str:
    """解析一个 GROMACS `#include` 到真实路径。

    这是**唯一实现**：`runabfe._resolve_gromacs_include_path` 是它的薄包装，
    避免两处各写一份解析顺序（相对包含文件的目录优先，其次 GROMACS include 目录）。
    """
    candidates = [os.path.join(os.path.dirname(including_file), include_name)]
    if gmx_include_dir:
        candidates.append(os.path.join(gmx_include_dir, include_name))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.realpath(candidate)
    raise FileNotFoundError(
        f"GROMACS include 无法解析: {include_name!r}（来自 {including_file}）"
    )


def _iter_topology_lines(top_path: str, gmx_include_dir: Optional[str] = None):
    """按 include 展开顺序产出 (来源文件, 去注释后的行)。

    深度优先、就地展开，与 GROMACS 预处理器语义一致——这一点对 `[ molecules ]`
    这类"位置有意义"的段落是必需的。重复 include 只展开一次。
    """
    seen = set()

    def _walk(path: str):
        real = os.path.realpath(path)
        if real in seen:
            return
        seen.add(real)
        with open(real, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.split(";", 1)[0].strip()
                if not line:
                    continue
                if line.startswith("#include"):
                    parts = line.split('"')
                    if len(parts) >= 2:
                        try:
                            resolved = resolve_gromacs_include(
                                parts[1], real, gmx_include_dir
                            )
                        except FileNotFoundError:
                            # 解析不到的 include 记录下来，由调用方决定是否致命：
                            # 有些体系的 posre.itp 是可选的。
                            yield (real, f"#unresolved_include {parts[1]}")
                            continue
                        yield from _walk(resolved)
                    continue
                yield (real, line)

    yield from _walk(top_path)


def parse_gromacs_topology(
    top_path: str,
    gmx_include_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """解析 GROMACS 拓扑，返回**权威**的体系组成。

    为什么需要它：靠残基名硬编码集合判身份，在换一套体系时会静默判错。实测
    `memtest/` 那套体系里——

      * 一个 POPC 是**一个 `[ moleculetype ]`、三个残基**（PA + PC + OL），
        按残基计数会数出 270 个"脂质"，APL 直接错 3 倍；
      * 水叫 `TP3`（mdtraj 的水表里只有 `TIP3`），离子叫 `Na+` / `Cl-`（带符号），
        两者都会让计数**静默变成 0**。

    而 `.top` 里本来就写着权威答案：

        [ molecules ]
        PROA 1 / POPC 90 / Na+ 25 / Cl- 36 / TP3 9542 / Atenolol-rank11 1

    所以身份一律以 `[ molecules ]` + `[ moleculetype ]` 为准。
    """
    defaults: Optional[Dict[str, Any]] = None
    moleculetypes: "OrderedDictType[str, Dict[str, Any]]" = {}
    molecules: List[Tuple[str, int]] = []
    atomtypes: Dict[str, Dict[str, Any]] = {}
    unresolved_includes: List[str] = []
    files: List[str] = []

    section = None
    current_moltype: Optional[str] = None
    for source, line in _iter_topology_lines(top_path, gmx_include_dir):
        if source not in files:
            files.append(source)
        if line.startswith("#unresolved_include"):
            unresolved_includes.append(line.split(None, 1)[1])
            continue
        if line.startswith("#"):
            continue  # #define / #ifdef 等预处理指令，本解析器不求完整
        if line.startswith("["):
            section = line.strip("[] ").strip().lower()
            if section == "moleculetype":
                current_moltype = None
            continue

        fields = line.split()
        if section == "defaults" and defaults is None:
            # nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ
            defaults = {
                "raw": line,
                "nbfunc": int(fields[0]) if len(fields) > 0 else None,
                "comb_rule": int(fields[1]) if len(fields) > 1 else None,
                "gen_pairs": fields[2] if len(fields) > 2 else None,
                "fudge_lj": float(fields[3]) if len(fields) > 3 else None,
                "fudge_qq": float(fields[4]) if len(fields) > 4 else None,
                "source": source,
            }
        elif section == "atomtypes" and len(fields) >= 7:
            # name at.num mass charge ptype sigma epsilon
            # （CHARMM-GUI 还会在注释里附 sigma_14/epsilon_14，注释已被剥掉。）
            try:
                atomtypes[fields[0]] = {
                    "atomic_number": int(float(fields[1])),
                    "mass": float(fields[2]),
                    "charge": float(fields[3]),
                    "ptype": fields[4],
                    "sigma_nm": float(fields[5]),
                    "epsilon_kj_mol": float(fields[6]),
                }
            except ValueError:
                # 少数力场的 [ atomtypes ] 列序不同（例如省略 at.num）。跳过而不是猜，
                # 由用它的地方 fail closed。
                pass
        elif section == "moleculetype" and current_moltype is None:
            current_moltype = fields[0]
            moleculetypes.setdefault(
                current_moltype,
                {
                    "name": current_moltype,
                    "n_atoms": 0,
                    "residue_names": [],
                    "atoms": [],
                },
            )
        elif section == "atoms" and current_moltype:
            # nr type resnr residue atom cgnr charge mass
            if len(fields) >= 5:
                entry = moleculetypes[current_moltype]
                entry["n_atoms"] += 1
                resname = fields[3]
                if resname not in entry["residue_names"]:
                    entry["residue_names"].append(resname)
                entry.setdefault("atoms", []).append(
                    {
                        "nr": int(fields[0]),
                        "type": fields[1],
                        "residue_name": resname,
                        "atom_name": fields[4],
                        "charge": float(fields[6]) if len(fields) >= 7 else None,
                        "mass": float(fields[7]) if len(fields) >= 8 else None,
                    }
                )
        elif section == "molecules":
            if len(fields) >= 2:
                try:
                    molecules.append((fields[0], int(fields[1])))
                except ValueError:
                    continue

    return {
        "top_path": str(top_path),
        "files": files,
        "defaults": defaults,
        "moleculetypes": moleculetypes,
        "molecules": molecules,
        "atomtypes": atomtypes,
        "unresolved_includes": unresolved_includes,
    }


# ---------------------------------------------------------------------------
# 组成驱动的身份判定（替代残基名硬编码集合）
# ---------------------------------------------------------------------------

# 水的 moleculetype 名。加 `TP3` 是因为 CHARMM-GUI 的 AMBER 转换器就叫它 TP3，
# 而 mdtraj 的 `_WATER_RESIDUES` 只有 `TIP3` —— 靠 mdtraj 的 `water` 关键字会
# **静默选出 0 个水**，疏水核内水与 co-ion 首层水配位数就都变成 0 而不报错。
WATER_MOLECULE_NAMES = frozenset(
    {
        "SOL", "HOH", "WAT", "H2O", "OH2",
        "TIP", "TIP3", "TIP3P", "TP3", "T3P",
        "TIP4", "TIP4P", "TIP4PEW", "TP4",
        "SPC", "SPCE", "OPC", "OPC3",
    }
)

# 单原子离子的 moleculetype 名（已归一化：去掉 +/-/数字后再比）。
MONOATOMIC_ION_NAMES = frozenset(
    {"NA", "CL", "K", "SOD", "CLA", "POT", "MG", "CA", "ZN", "LI", "BR", "I", "CS", "RB", "F"}
)

# 标准 + Amber 常见变体氨基酸残基名。自带一份而不是用 mdtraj 的
# `_AMINO_ACID_CODES`：实测那张表里 `HID` / `HIE` / `ASH` / `CYX` / `NTRP` /
# `CCYS` 全都**不在**（只有 `HIP` 在），于是 `select("protein")` 会静默漏掉这些残基。
STANDARD_AMINO_ACID_CODES = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        # Amber 质子化/氧化态变体
        "HID", "HIE", "HIP", "ASH", "GLH", "CYX", "CYM", "LYN", "TYM", "ARN",
        # 封端基团
        "ACE", "NME", "NHE", "NH2",
    }
)


def normalize_protein_residue_name(name: str) -> Optional[str]:
    """把 `NTRP` / `CCYS` 这类 N-/C-端变体归一到标准三字母码；不是氨基酸则返回 None。"""
    upper = str(name).strip().upper()
    if upper in STANDARD_AMINO_ACID_CODES:
        return upper
    if len(upper) == 4 and upper[0] in ("N", "C") and upper[1:] in STANDARD_AMINO_ACID_CODES:
        return upper[1:]
    return None


def _normalize_ion_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().upper() if ch.isalpha())


def molecule_atom_ranges(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    """由 `[ molecules ]` 顺序与各 moleculetype 的原子数算出每个分子的原子区间。

    GROMACS 的原子顺序就是 `[ molecules ]` 的展开顺序，所以这个映射是**精确**的，
    不需要靠键连通性或残基编号（实测那套体系里所有脂质的 `.gro` 残基编号都是 1，
    根本没法用残基编号分子）。

    已用 memtest/ 实测对齐：`PROA` 4566 个原子 → 第一个 POPC 原子在 index 4566，
    与 `step7_production.gro` 里第一个 `PA` 出现的位置逐一致。
    """
    ranges: List[Dict[str, Any]] = []
    cursor = 0
    for name, count in parsed["molecules"]:
        moltype = parsed["moleculetypes"].get(name)
        if moltype is None:
            raise ValueError(
                f"[ molecules ] 里的 {name!r} 找不到对应的 [ moleculetype ]，"
                "拓扑不自洽或有未解析的 include："
                f"{parsed.get('unresolved_includes')}"
            )
        size = int(moltype["n_atoms"])
        if size <= 0:
            raise ValueError(f"moleculetype {name!r} 的原子数为 {size}")
        for ordinal in range(int(count)):
            ranges.append(
                {
                    "molecule_name": name,
                    "ordinal": ordinal,
                    "start": cursor,
                    "stop": cursor + size,
                }
            )
            cursor += size
    return ranges


def classify_system_composition(
    parsed: Dict[str, Any],
    ligand_molecule_name: Optional[str] = None,
    declared_roles: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """把每个 moleculetype 判定为 protein / lipid / water / ion / ligand。

    判据一律来自 `.top` 本身（moleculetype 的残基构成与原子数），不用残基名硬编码
    集合去猜整个体系。任何**判不出角色**的 moleculetype 一律 fail closed——
    静默归入 "other" 就等于让它在所有选择里消失。

    `declared_roles` 允许显式覆盖（例如非常规命名的脂质），覆盖会留记录。
    """
    declared_roles = {
        str(k): str(v).strip().lower() for k, v in (declared_roles or {}).items()
    }
    valid_roles = {"protein", "lipid", "water", "ion", "ligand"}
    bad = {k: v for k, v in declared_roles.items() if v not in valid_roles}
    if bad:
        raise ValueError(f"declared_roles 出现非法角色 {bad}；允许 {sorted(valid_roles)}")

    roles: Dict[str, str] = {}
    evidence: Dict[str, str] = {}
    for name, moltype in parsed["moleculetypes"].items():
        if name in declared_roles:
            roles[name] = declared_roles[name]
            evidence[name] = "declared_override"
            continue
        residues = [str(r).strip().upper() for r in moltype["residue_names"]]
        upper = str(name).strip().upper()

        if ligand_molecule_name and name == ligand_molecule_name:
            roles[name] = "ligand"
            evidence[name] = "matches_ligand_molecule_name"
        elif upper in WATER_MOLECULE_NAMES or (
            residues and all(r in WATER_MOLECULE_NAMES for r in residues)
        ):
            roles[name] = "water"
            evidence[name] = "water_molecule_name"
        elif moltype["n_atoms"] == 1 and _normalize_ion_name(name) in MONOATOMIC_ION_NAMES:
            roles[name] = "ion"
            evidence[name] = "monoatomic_ion_name"
        elif residues and all(r in KNOWN_LIPID_RESIDUE_NAMES for r in residues):
            # POPC = PA + PC + OL：三个残基全在脂质名表里 → 整个分子是脂质。
            roles[name] = "lipid"
            evidence[name] = "all_residues_are_lipid_residues"
        elif residues and any(
            normalize_protein_residue_name(r) is not None for r in residues
        ):
            roles[name] = "protein"
            evidence[name] = "contains_amino_acid_residues"
        else:
            raise ValueError(
                f"无法判定 moleculetype {name!r} 的角色"
                f"（残基 {residues[:8]}，{moltype['n_atoms']} 原子）。"
                "按 memtodolist 的纪律这里 fail closed，不静默归入 other——"
                "那会让它从所有原子选择里消失。请用 declared_roles 显式指定"
                "（protein / lipid / water / ion / ligand）。"
            )

    ranges = molecule_atom_ranges(parsed)
    by_role: Dict[str, List[int]] = {role: [] for role in valid_roles}
    molecules_by_role: Dict[str, List[Dict[str, Any]]] = {
        role: [] for role in valid_roles
    }
    for entry in ranges:
        role = roles[entry["molecule_name"]]
        by_role[role].extend(range(entry["start"], entry["stop"]))
        molecules_by_role[role].append(entry)

    counts = {}
    for name, count in parsed["molecules"]:
        counts[name] = counts.get(name, 0) + int(count)

    return {
        "roles": roles,
        "role_evidence": evidence,
        "molecule_counts": counts,
        "atom_indices_by_role": {k: sorted(v) for k, v in by_role.items()},
        "molecules_by_role": molecules_by_role,
        "n_atoms_total": ranges[-1]["stop"] if ranges else 0,
        "declared_roles": declared_roles,
    }


# ---------------------------------------------------------------------------
# `[ pairs ]` funct 2 → funct 1（OpenMM 兼容性转换）
# ---------------------------------------------------------------------------

GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION = 1


def gromacs_topology_files(
    top_path: str,
    gmx_include_dir: Optional[str] = None,
) -> List[str]:
    """按 include 展开顺序返回拓扑树里全部文件的真实路径（去重、深度优先）。"""
    seen: List[str] = []
    include_re = re.compile(r'^\s*#\s*include\s+"([^"]+)"')

    def _walk(path: str):
        real = os.path.realpath(path)
        if real in seen:
            return
        seen.append(real)
        with open(real, encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                match = include_re.match(raw.split(";", 1)[0])
                if not match:
                    continue
                try:
                    _walk(resolve_gromacs_include(match.group(1), real, gmx_include_dir))
                except FileNotFoundError:
                    continue

    _walk(top_path)
    return seen


def _scan_moleculetype_charges(text: str) -> Dict[str, Dict[int, float]]:
    """逐 moleculetype 收集 `[ atoms ]` 的电荷（按该 moleculetype 内的局部原子号）。"""
    charges: Dict[str, Dict[int, float]] = {}
    section = None
    current: Optional[str] = None
    expecting_name = False
    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line.strip("[] ").strip().lower()
            expecting_name = section == "moleculetype"
            continue
        fields = line.split()
        if expecting_name:
            current = fields[0]
            charges.setdefault(current, {})
            expecting_name = False
        elif section == "atoms" and current and len(fields) >= 7:
            charges[current][int(fields[0])] = float(fields[6])
    return charges


def convert_gromacs_pairs_funct2(
    top_path: str,
    out_dir: str,
    gmx_include_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """把 `[ pairs ]` 的 funct 2 改写成 funct 1，写到 `out_dir`，**不改原始输入**。

    ## 为什么需要

    OpenMM 的 `app.GromacsTopFile` 只接受 `[ pairs ]` funct 1
    （`gromacstopfile.py::_processPair` 显式 `if fields[2] != '1': raise`）。
    CHARMM-GUI 的 AMBER FF-Converter 会对**部分**对写 funct 2：

        ai  aj  2  fudgeQQ  q1  q2  sigma14  epsilon14

    多出来的三列（fudgeQQ / q1 / q2）是**冗余重述**——OpenMM 算 1-4 exception 的电荷
    用的是「粒子电荷 × 全局 fudgeQQ」（`gromacstopfile.py` 里
    `atom1params[0]*atom2params[0]*fudgeQQ`），所以只要

        1. 逐对 fudgeQQ == `[ defaults ]` 的全局 fudgeQQ；
        2. q1/q2 == 该 moleculetype `[ atoms ]` 里的真实电荷，

    funct 2 与 `ai aj 1 sigma14 epsilon14` 就**严格等价**：sigma/eps 正好落在 OpenMM
    读取的 `fields[3:5]` 上。**这两个条件逐对校验，任一不成立就 fail closed**——
    不成立意味着该对真的覆盖了电荷或缩放因子，硬转会静默改变哈密顿量。

    ## 做法

    把整棵 include 树**逐文件拷贝**到 `out_dir`（保持相对路径，所以 `#include` 仍能
    解析），只改写含 funct-2 的文件里那几行。其余内容逐字节原样保留——包括
    `#ifdef POSRES` / `#ifdef DIHRES` 这类未激活的预处理块，不做展开、不做丢弃。

    原始输入一个字节都不动；缓存指纹仍应基于原始文件（本函数返回两侧的 SHA256
    以便一并落进 provenance）。
    """
    import hashlib
    import shutil

    parsed = parse_gromacs_topology(top_path, gmx_include_dir)
    defaults = parsed["defaults"] or {}
    global_fudge_qq = defaults.get("fudge_qq")
    if global_fudge_qq is None:
        raise ValueError(
            f"{top_path} 的拓扑树里找不到 `[ defaults ]` 的 fudgeQQ，"
            "无法校验 funct-2 pairs 是否与全局缩放一致 —— 拒绝盲转。"
        )

    files = gromacs_topology_files(top_path, gmx_include_dir)
    top_dir = os.path.dirname(os.path.realpath(top_path))
    outside = [f for f in files if not os.path.realpath(f).startswith(top_dir + os.sep)
               and os.path.realpath(f) != os.path.realpath(top_path)]
    if outside:
        raise ValueError(
            f"拓扑树里有文件位于 {top_dir} 之外：{outside[:3]}。"
            "逐文件拷贝无法保持它们的相对 include 路径，拒绝转换（需要先把它们收拢到"
            "拓扑目录下，或改用不含 funct-2 pairs 的拓扑）。"
        )

    os.makedirs(out_dir, exist_ok=True)
    patched: List[Dict[str, Any]] = []
    total_converted = 0

    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    for path in files:
        real = os.path.realpath(path)
        relative = (
            os.path.basename(real)
            if real == os.path.realpath(top_path)
            else os.path.relpath(real, top_dir)
        )
        destination = os.path.join(out_dir, relative)
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)

        text = open(real, encoding="utf-8", errors="replace").read()
        charges_by_moltype = _scan_moleculetype_charges(text)

        section = None
        current_moltype: Optional[str] = None
        expecting_name = False
        converted_here = 0
        out_lines: List[str] = []
        for raw in text.splitlines(keepends=True):
            body, _, comment = raw.partition(";")
            stripped = body.strip()
            if stripped.startswith("["):
                section = stripped.strip("[] ").strip().lower()
                expecting_name = section == "moleculetype"
                out_lines.append(raw)
                continue
            if not stripped or stripped.startswith("#"):
                out_lines.append(raw)
                continue
            fields = stripped.split()
            if expecting_name:
                current_moltype = fields[0]
                expecting_name = False
                out_lines.append(raw)
                continue
            if section != "pairs" or len(fields) < 3 or fields[2] != "2":
                out_lines.append(raw)
                continue

            if len(fields) < 8:
                raise ValueError(
                    f"{real} 的 funct-2 `[ pairs ]` 行字段不足 8 个，无法转换：{stripped!r}。"
                    "预期 `ai aj 2 fudgeQQ q1 q2 sigma14 epsilon14`。"
                )
            ai, aj = int(fields[0]), int(fields[1])
            pair_fudge, q1, q2 = float(fields[3]), float(fields[4]), float(fields[5])
            sigma14, epsilon14 = fields[6], fields[7]

            # 条件 1：逐对 fudgeQQ 必须等于全局值。
            if abs(pair_fudge - float(global_fudge_qq)) > 1.0e-6:
                raise ValueError(
                    f"{real} 的 pair ({ai},{aj}) 覆盖了 fudgeQQ："
                    f"逐对 {pair_fudge} vs 全局 {global_fudge_qq}。"
                    "OpenMM 的 1-4 exception 只用全局 fudgeQQ，硬转会**静默改变**"
                    "该对的静电缩放，拒绝转换。"
                )
            # 条件 2：q1/q2 必须等于该 moleculetype 的真实原子电荷。
            table = charges_by_moltype.get(current_moltype or "", {})
            for index, declared_q in ((ai, q1), (aj, q2)):
                actual = table.get(index)
                if actual is None:
                    raise ValueError(
                        f"{real} 的 moleculetype {current_moltype!r} 里找不到原子 {index} 的电荷，"
                        "无法校验 funct-2 pair 的等价性。"
                    )
                if abs(declared_q - actual) > 1.0e-6:
                    raise ValueError(
                        f"{real} 的 pair ({ai},{aj}) 覆盖了原子 {index} 的电荷："
                        f"pair 里写 {declared_q} 而 `[ atoms ]` 是 {actual}。"
                        "OpenMM 的 exception 用的是粒子电荷，硬转会静默改变哈密顿量，拒绝转换。"
                    )

            new_body = f"{fields[0]:>5s} {fields[1]:>5s}     1  {sigma14}  {epsilon14}"
            note = (
                "; [pairs funct 2 -> 1] 原始: "
                f"funct=2 fudgeQQ={fields[3]} q1={fields[4]} q2={fields[5]}"
                "（已校验与全局 fudgeQQ 及 [ atoms ] 电荷一致，故等价）"
            )
            out_lines.append(f"{new_body} {note}\n")
            converted_here += 1

        if converted_here:
            with open(destination, "w", encoding="utf-8") as handle:
                handle.write("".join(out_lines))
            patched.append(
                {
                    "source": real,
                    "destination": destination,
                    "n_pairs_converted": converted_here,
                    "source_sha256": _sha256(real),
                    "destination_sha256": _sha256(destination),
                }
            )
            total_converted += converted_here
        else:
            shutil.copy2(real, destination)

    converted_top = os.path.join(out_dir, os.path.basename(os.path.realpath(top_path)))
    return {
        "conversion_version": GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION,
        "converted_top_path": converted_top,
        "n_pairs_converted": total_converted,
        "patched_files": patched,
        "global_fudge_qq": float(global_fudge_qq),
        "n_files_copied": len(files),
        "note": (
            "funct 2 的 fudgeQQ/q1/q2 三列经逐对校验与全局 fudgeQQ 及 [ atoms ] 电荷一致，"
            "属冗余重述，转成 funct 1 后 sigma14/epsilon14 落在 OpenMM 读取的 fields[3:5]，"
            "哈密顿量不变。原始输入未被修改。"
        ),
    }


def gromacs_file_has_funct2_pairs(path: str) -> bool:
    """**单个文件**自身是否含 `[ pairs ]` funct 2（不跟随 include）。

    与 `gromacs_topology_has_funct2_pairs`（整棵树）分开：定位"哪个 .itp 带"要用
    这个，判断"要不要走兼容性转换"要用那个。两者共用同一份扫描逻辑，只是作用域
    不同——不要各写一份，否则口径会分叉。
    """
    section = None
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.split(";", 1)[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                section = line.strip("[] ").strip().lower()
                continue
            if section == "pairs":
                fields = line.split()
                if len(fields) >= 3 and fields[2] == "2":
                    return True
    return False


def gromacs_topology_has_funct2_pairs(
    top_path: str,
    gmx_include_dir: Optional[str] = None,
) -> bool:
    """**整棵**拓扑树里是否存在 `[ pairs ]` funct 2（决定要不要走兼容性转换）。

    ⚠️ 递归语义：对顶层 `.top` 调用它会因为 include 了带 funct-2 的 `.itp` 而返回
    True。想知道"哪个文件自己带"请用 `gromacs_file_has_funct2_pairs`。
    """
    return any(
        gromacs_file_has_funct2_pairs(path)
        for path in gromacs_topology_files(top_path, gmx_include_dir)
    )


def openmm_compatible_gromacs_top(
    top_file: str,
    gmx_include_dir: Optional[str] = None,
    compat_dir: Optional[str] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """返回 `(可直接交给 OpenMM 的 .top 路径, 转换信息或 None)`。**幂等、按内容缓存。**

    没有 funct-2 pairs 时原路返回，`None` 表示"未做任何转换"。

    `compat_dir` 不给时用系统临时目录下的**内容寻址**缓存
    （key = 整棵拓扑树各文件 sha256 + 转换版本号），所以：
      - 同一份输入多次调用只转换一次；
      - 输入变了 key 就变，不会复用过期的转换产物；
      - 不往用户的输入目录里写任何东西。

    `runabfe` 主路径会显式传 `output_dir/gromacs_openmm_compat` 以便审计。
    """
    import hashlib
    import tempfile

    if not gromacs_topology_has_funct2_pairs(top_file, gmx_include_dir):
        return top_file, None

    if compat_dir is None:
        digest = hashlib.sha256()
        digest.update(f"v{GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION}\n".encode())
        top_dir = os.path.dirname(os.path.realpath(top_file))
        for path in gromacs_topology_files(top_file, gmx_include_dir):
            with open(path, "rb") as handle:
                file_digest = hashlib.sha256(handle.read()).hexdigest()
            relative = os.path.relpath(os.path.realpath(path), top_dir)
            digest.update(f"{relative}:{file_digest}\n".encode())
        compat_dir = os.path.join(
            tempfile.gettempdir(),
            "abfe_gromacs_openmm_compat",
            digest.hexdigest()[:16],
        )

    candidate = os.path.join(
        compat_dir, os.path.basename(os.path.realpath(top_file))
    )
    manifest_path = os.path.join(compat_dir, "conversion_manifest.json")

    # 复用之前必须核对**整棵树**是否完整且与当前输入一致。
    #
    # ⚠️ 早先这里只检查"顶层 top 存在 && 树里没有 funct-2"就直接复用——那是
    # fail-open：一次被中断的转换会留下半成品（部分文件已拷、部分没拷，或某个
    # `.itp` 被截断），而这两个条件仍可能成立。于是 OpenMM 读到一棵**不自洽**的
    # 拓扑，建出一个参数错乱的 System，最小化后 PE 到 1e13、几千步后变成一个
    # 没有上下文的 `Particle coordinate is NaN`。
    # 所以改成按 manifest 逐文件核对 sha256，任何不符就重转。
    if os.path.isfile(candidate) and os.path.isfile(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception:
            manifest = None
        if manifest and _conversion_manifest_matches(
            manifest, top_file, gmx_include_dir, compat_dir
        ):
            return candidate, {
                "reused_existing_conversion": True,
                "converted_top_path": candidate,
                "conversion_version": GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION,
                "manifest_path": manifest_path,
            }
        logger.warning(
            "⚠️ %s 里已有的 funct-2 转换与当前输入不符（或不完整），重新转换。",
            compat_dir,
        )

    result = convert_gromacs_pairs_funct2(top_file, compat_dir, gmx_include_dir)
    _write_conversion_manifest(
        manifest_path, top_file, gmx_include_dir, compat_dir, result
    )
    result["manifest_path"] = manifest_path
    return result["converted_top_path"], result


def _conversion_tree_fingerprint(
    top_file: str,
    gmx_include_dir: Optional[str],
    compat_dir: str,
) -> Dict[str, Any]:
    """源树与转换产物的逐文件 sha256，用于核对复用是否安全。"""
    import hashlib

    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    top_dir = os.path.dirname(os.path.realpath(top_file))
    source: Dict[str, str] = {}
    converted: Dict[str, Optional[str]] = {}
    for path in gromacs_topology_files(top_file, gmx_include_dir):
        real = os.path.realpath(path)
        relative = (
            os.path.basename(real)
            if real == os.path.realpath(top_file)
            else os.path.relpath(real, top_dir)
        )
        source[relative] = _sha256(real)
        destination = os.path.join(compat_dir, relative)
        converted[relative] = (
            _sha256(destination) if os.path.isfile(destination) else None
        )
    return {
        "conversion_version": GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION,
        "source": source,
        "converted": converted,
    }


def _write_conversion_manifest(
    manifest_path: str,
    top_file: str,
    gmx_include_dir: Optional[str],
    compat_dir: str,
    result: Dict[str, Any],
) -> None:
    fingerprint = _conversion_tree_fingerprint(top_file, gmx_include_dir, compat_dir)
    fingerprint["n_pairs_converted"] = int(result.get("n_pairs_converted", 0))
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(fingerprint, handle, indent=2, ensure_ascii=False)


def _conversion_manifest_matches(
    manifest: Dict[str, Any],
    top_file: str,
    gmx_include_dir: Optional[str],
    compat_dir: str,
) -> bool:
    if int(manifest.get("conversion_version", -1)) != (
        GROMACS_PAIRS_FUNCT2_CONVERSION_VERSION
    ):
        return False
    current = _conversion_tree_fingerprint(top_file, gmx_include_dir, compat_dir)
    # 源文件必须逐一相同（输入变了就不能复用旧转换）。
    if manifest.get("source") != current["source"]:
        return False
    # 转换产物必须**全部存在**且与当初写下的一致（挡住半成品 / 被改过的产物）。
    recorded = manifest.get("converted") or {}
    if set(recorded) != set(current["converted"]):
        return False
    for relative, digest in recorded.items():
        if digest is None or current["converted"].get(relative) != digest:
            return False
    return True


def load_gromacs_topology_for_openmm(
    top_file: str,
    includeDir: Optional[str] = None,
    compat_dir: Optional[str] = None,
    **kwargs,
):
    """**唯一**的 GROMACS 拓扑加载入口：需要时先做 funct-2 等价转换，再交给 OpenMM。

    ## 为什么必须只有一个入口

    OpenMM 的 `GromacsTopFile` 不支持 `[ pairs ]` funct 2。首次修这个问题时只在
    `build_system_from_gromacs` 一处接了转换，结果溶剂腿的
    `build_and_cache_solvent_leg`（另一个直接调 `app.GromacsTopFile` 的地方）
    照样炸——**这与 B1 当初只接了 1 个 `ABFEPipeline` 构造点是同一个毛病**：
    同一件事有多个入口，补一个漏一片。

    现在全仓所有加载点都走这里，并有契约测试禁止裸调 `app.GromacsTopFile`
    （见 `tests/test_gromacs_pairs_funct2_conversion.py`）。
    """
    resolved, _ = openmm_compatible_gromacs_top(top_file, includeDir, compat_dir)
    return app.GromacsTopFile(resolved, includeDir=includeDir, **kwargs)


def _family_from_defaults(defaults: Optional[Dict[str, Any]]) -> Optional[str]:
    if not defaults:
        return None
    fudge_lj = defaults.get("fudge_lj")
    fudge_qq = defaults.get("fudge_qq")
    if fudge_lj is None or fudge_qq is None:
        return None
    for family, expect_lj, expect_qq in _FORCEFIELD_FAMILY_FUDGE_SIGNATURES:
        if (
            abs(float(fudge_lj) - expect_lj) <= _FUDGE_MATCH_TOLERANCE
            and abs(float(fudge_qq) - expect_qq) <= _FUDGE_MATCH_TOLERANCE
        ):
            return family
    return None


def detect_forcefield_family_from_top(
    top_path: str,
    gmx_include_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """从 GROMACS `.top` 判定力场族（§1.1）：`[ defaults ]` 为主，`#include` 为辅。

    **主判据是 `[ defaults ]` 的 1-4 缩放约定**（Amber 0.5/0.8333 vs CHARMM 1.0/1.0），
    递归跟随 include 去找它——`[ defaults ]` 通常不在顶层 `.top` 里，而在
    `forcefield.itp` 内。实测 CHARMM-GUI FF-Converter 产出的 AMBER 体系其 include
    路径完全不含 `amber` 字样，只有 `[ defaults ]` 能判对。

    include 路径 token 作为**次要**信号：两者一致则 `agree`，只有其一则用它，
    冲突则返回 `family=None` 让上层 fail closed（这种情况必须人工裁决）。
    """
    parsed = parse_gromacs_topology(top_path, gmx_include_dir)

    includes: List[str] = []
    for source, line in _iter_topology_lines(top_path, gmx_include_dir):
        if line.startswith("#unresolved_include"):
            includes.append(line.split(None, 1)[1])
    # 顶层与各级文件里出现过的 include 名（用于路径 token 判据）。
    with open(top_path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            stripped = raw.split(";", 1)[0].strip()
            if stripped.startswith("#include"):
                parts = stripped.split('"')
                if len(parts) >= 2:
                    includes.append(parts[1])
    for path in parsed["files"]:
        includes.append(os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path))

    families: Dict[str, List[str]] = {}
    for include in includes:
        lowered = include.lower()
        for token, family in sorted(
            _FORCEFIELD_FAMILY_TOKENS, key=lambda kv: -len(kv[0])
        ):
            if token in lowered:
                families.setdefault(family, []).append(include)
                break

    path_families = sorted(families)
    path_family = path_families[0] if len(path_families) == 1 else None
    defaults_family = _family_from_defaults(parsed["defaults"])

    if defaults_family and path_family:
        if defaults_family == path_family:
            detected, reason = defaults_family, "defaults_and_include_agree"
        else:
            detected, reason = None, (
                f"conflict:defaults={defaults_family},include={path_family}"
            )
    elif defaults_family:
        detected, reason = defaults_family, "defaults_1_4_scaling_signature"
    elif path_family:
        detected, reason = path_family, "single_family_include"
    elif len(path_families) > 1:
        detected, reason = None, f"mixed_family_includes:{path_families}"
    else:
        detected, reason = None, "no_defaults_signature_and_no_recognized_include"

    return {
        "family": detected,
        "reason": reason,
        "includes": sorted(set(includes)),
        "family_evidence": {k: sorted(set(v)) for k, v in families.items()},
        "defaults": parsed["defaults"],
        "defaults_row": (parsed["defaults"] or {}).get("raw"),
        "defaults_family": defaults_family,
        "include_family": path_family,
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
        # ⚠️ 这个键说的是"**目标**是不是 legacy 那条均匀密度 LRC 路线"，
        # **不是**"某一条腿实际加不加解析尾项"。后者要按腿的环境判，见
        # `resolve_leg_dispersion_implementation()`。两者曾被混为一谈（B6-FIX）。
        "uniform_density_lrc_active": (
            resolved == DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC
        ),
        # §5 最后一条 / §6.5：APBS 与 LJ 色散正交，不得互相顶替。
        "apbs_is_orthogonal_to_dispersion": True,
    }


# ============================================================================
# B6-FIX（2026-08-04）：把"目标 Hamiltonian"与"每条腿实现该目标的环境专用长程处理"
# 拆开。
#
# 起因是一条实测出来的自相矛盾：膜运行的**溶剂腿**（4.05 nm 纯水盒）在
# `final_results.json` 里带着这句理由——
#
#     disabled_by_membrane_forcefield_protocol: …配体所在口袋的局域密度既不是水
#     也不是体相脂质
#
# 而那条腿里配体周围**恰恰就是**均匀体相水。根因是判据只有一个全局布尔：
#
#     apply_lrc = (dispersion_protocol == "legacy_uniform_density_lrc")
#
# 它没有环境维度，于是"复合物腿口袋里不均匀"这个正确理由被原样套到了水盒上，
# 把一条**合法**的 bulk-water 尾项修正一起关掉了。实测代价：同一个配体
# （Atenolol，41 原子）的溶剂腿 vanishing 从 96.96 变成 83.83 kJ/mol（−13.1）。
#
# 正确的分层：
#
#     dispersion_protocol  = **目标**：所选力场原始参数化时的色散条件
#                            （Amber Lipid21 = 开各向同性 LRC；CHARMM36 = force-switch 不加 LRC）
#     ↓  每条腿在**自己的环境**里怎么达成这个目标
#     实现              = f(目标, 该腿配体所处环境)
#
# 环境维度取 `system_type`，那是**用户在输入文件里显式声明**的值（B1 的规矩是
# 「不许按残基名猜 system_type」，不是「不许用声明出来的 system_type 分派」），
# 所以按它自动切换实现是合法的、可审计的，不是运行时猜测。
#
# 溶剂腿天然落在 soluble 一侧：`runabfe` 构造溶剂腿 pipeline 时**刻意不传**
# `environment_type`/`membrane`（B1 的接线契约测试钉着这一点），于是它解析出来
# 就是 soluble —— 不需要为这条腿新造任何标记。
# ============================================================================

# 实现名（进 provenance / final_results.json，机器可读，别改措辞）。
DISPERSION_IMPL_UNIFORM_BULK_ANALYTIC_TAIL = "uniform_bulk_density_analytic_tail"
DISPERSION_IMPL_TRUNCATED_NO_TAIL = "truncated_no_analytic_tail"
DISPERSION_IMPL_FORCE_SWITCH_NO_TAIL = "force_switch_no_tail_by_forcefield_design"


def resolve_leg_dispersion_implementation(
    dispersion_protocol: Optional[str],
    environment_type: Optional[str] = None,
) -> Dict[str, Any]:
    """给**一条腿**定"炼金 ligand–environment 的长程色散怎么处理"。

    返回里 `alchemical_uniform_density_lrc` 就是那个唯一的布尔判据；
    `target_met` 说明这条腿有没有真正达成力场参数化条件所要求的目标
    —— 膜复合物腿目前**达不成**（需要 §1.3 路线 C 的非均匀修正，未实现），
    这一点必须如实写进结果，而不是把"关掉了"记成"处理好了"。

    | 目标 dispersion_protocol | 该腿环境 | 炼金 LRC | target_met |
    | --- | --- | --- | --- |
    | `legacy_uniform_density_lrc` | soluble | 开 | 是（改动前唯一行为，逐位不变）|
    | `ff_native_isotropic_lrc`    | soluble（含膜运行的溶剂腿）| **开** | 是 |
    | `ff_native_isotropic_lrc`    | membrane | 关 | **否**（路线 C 未实现）|
    | `ff_native_force_switch_no_lrc` | 任意 | 关 | 是（力场本身就不加 LRC）|

    ⚠️ 非 legacy 目标**必须显式给出 `environment_type`**：这一维决定加不加修正，
    缺省猜哪一边都会静默出错（猜 soluble → 膜口袋里加上无效修正；猜 membrane →
    又把水盒的合法修正关掉，即本次要修的 bug）。所以缺就 raise。
    """
    protocol = str(dispersion_protocol or "").strip().lower()
    if not protocol:
        protocol = DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC

    if protocol == DISPERSION_PROTOCOL_LEGACY_UNIFORM_LRC:
        # legacy 是本改动之前唯一存在的路线，且 membrane+legacy 在
        # `resolve_dispersion_protocol` 就已 fail closed，所以这里不需要环境维度。
        return {
            "dispersion_protocol": protocol,
            "environment_type": resolve_environment_type(environment_type),
            "ligand_environment_is_uniform_bulk": True,
            "alchemical_uniform_density_lrc": True,
            "implementation": DISPERSION_IMPL_UNIFORM_BULK_ANALYTIC_TAIL,
            "target_met": True,
            "reason": "",
        }

    if environment_type is None:
        raise ValueError(
            f"dispersion_protocol={protocol!r} 是非 legacy 路线，必须显式给出这条腿的 "
            "environment_type。它决定炼金 ligand–environment 的均匀密度尾项加不加：\n"
            "    · 纯水/可溶腿：配体周围就是均匀体相 ⟹ 该修正**成立**，关掉是错的；\n"
            "    · 膜复合物腿：口袋局域密度既不是水也不是体相脂质 ⟹ 该修正不成立。\n"
            "    缺省猜任何一边都会静默产出错误的 ΔG（B6-FIX 修的正是猜错方向那一版）。"
        )
    resolved_env = resolve_environment_type(environment_type)
    is_membrane = resolved_env == ENVIRONMENT_TYPE_MEMBRANE

    if protocol == DISPERSION_PROTOCOL_FF_NATIVE_FORCE_SWITCH_NO_LRC:
        return {
            "dispersion_protocol": protocol,
            "environment_type": resolved_env,
            "ligand_environment_is_uniform_bulk": not is_membrane,
            "alchemical_uniform_density_lrc": False,
            "implementation": DISPERSION_IMPL_FORCE_SWITCH_NO_TAIL,
            # 力场原始参数化就不带 LRC，所以"不加"正是达成目标。
            "target_met": True,
            "reason": (
                "no_analytic_tail_by_forcefield_design: "
                f"dispersion_protocol={protocol!r} 的力场原始参数化条件是 force-switch "
                "且**不加**长程色散修正，因此炼金 ligand–environment 也不加解析尾项"
            ),
        }

    if protocol == DISPERSION_PROTOCOL_FF_NATIVE_ISOTROPIC_LRC:
        if is_membrane:
            return {
                "dispersion_protocol": protocol,
                "environment_type": resolved_env,
                "ligand_environment_is_uniform_bulk": False,
                "alchemical_uniform_density_lrc": False,
                "implementation": DISPERSION_IMPL_TRUNCATED_NO_TAIL,
                # ⚠️ 达不成目标：力场是在开着各向同性色散修正的条件下拟合的，
                # 而这条腿的炼金 ligand–environment 项现在是**截断**的。
                # 正解是 §1.3 路线 C（膜非均匀色散修正），尚未实现。
                "target_met": False,
                # §1.3 指定的机器可读字符串，措辞不要改。
                "reason": (
                    "disabled_by_membrane_forcefield_protocol: "
                    f"dispersion_protocol={protocol!r} 且该腿 environment_type='membrane' 时，"
                    "炼金 ligand–environment 的均匀体相密度 LRC 不适用"
                    "（配体所在口袋的局域密度既不是水也不是体相脂质）；"
                    "环境–环境色散仍按所选力场的原始参数化条件由基础 NonbondedForce 处理。"
                    "达成目标需要 §1.3 路线 C 的非均匀色散修正（未实现），"
                    "所以本腿记为 target_met=false 而不是"
                    "「已正确处理」"
                ),
            }
        return {
            "dispersion_protocol": protocol,
            "environment_type": resolved_env,
            "ligand_environment_is_uniform_bulk": True,
            "alchemical_uniform_density_lrc": True,
            "implementation": DISPERSION_IMPL_UNIFORM_BULK_ANALYTIC_TAIL,
            "target_met": True,
            "reason": "",
        }

    raise ValueError(
        f"dispersion_protocol={protocol!r} 没有对应的每腿实现映射；"
        f"合法值 {list(DISPERSION_PROTOCOLS)}，其中路线 B/C 在 "
        "`resolve_dispersion_protocol` 就已 NotImplementedError。"
    )


# ============================================================================
# §3.3 膜输入核对 + §9 膜质量门（memtodolist.md §3.3 / §9 / §13.3）
#
# 这两节的共同前提，清单说得很直接：
#   §3.3 输入必须是**已经完成膜构建和主要平衡**的体系，
#        不依赖当前通用 10 ns 预平衡去完成脂质重排或蛋白插膜。
#   §9   通用 10 ns **不是**膜平衡充分性的证明；每个量都要给时间序列与
#        末段窗口的漂移斜率，不能只报平均值；判据统一为
#        "末段 ≥ 20 ns 内线性漂移小于阈值"（阈值见 §13.3）。
#        质量门失败时回到膜体系平衡，**不允许靠增加 ABFE 窗口掩盖**。
#
# 归档部分（.top / 全部 .itp / 位置限制 / 力场 include 的递归 SHA256）复用
# `runabfe._gromacs_dependency_hashes()`，本节不再造第二套。
# ============================================================================

MEMBRANE_INPUT_PROTOCOL_VERSION = 1
# v2（2026-08-02）：
#   * 新增 `apl_protein_corrected_nm2` 观测量，§13.3 的 APL 绝对值门改为比校正后的值
#     （含蛋白膜拿 raw APL 比纯脂文献值必然偏大，实测 0.826 vs 0.645）；
#   * 时间轴改为由 `frame_interval_ps` 显式重建，不再吃 mdtraj 给 DCD 的整数帧号
#     （v1 下那条 10 ns 轨迹被当成 0.499 ns，所有 per-ns 判据都错 20 倍）。
# v1 的报告与 v2 的**不可直接比较**：per-ns 斜率与 APL 绝对值口径都变了。
# v3（2026-08-02，同日）：
#   * MEM-10：`superpose` 原地污染修掉。此前倾角、蛋白横截面/校正后 APL、疏水核内水、
#     水层间隙、密度分布、脂质横向弛豫都在"对齐到蛋白骨架"的坐标上算，而 slab 边界
#     取自对齐前坐标。实测 τ 因此被放大 12 倍（11.57 → 139.36 ns）、倾角漂移被压掉
#     （0.477 → 1.274 °/window）。**v2 的这些数字全部作废。**
#   * MEM-11：弛豫时间尺度改时间平均 MSD + 声明 lag 窗口；该判据从 `checks`
#     降级为 `statistics.equilibration_vs_relaxation` 诊断。
#   * MEM-13：口袋/配体 RMSD 改成"对齐骨架后不重拟合"的 pose 漂移
#     （0.0760 → 0.0857、0.0493 → 0.0833 nm）。
#   * MEM-12：新增 `statistics.apl_vs_pure_lipid_literature` 诊断（不判门）。
MEMBRANE_QUALITY_GATE_PROTOCOL_VERSION = 3

# §9 质量门的执行模式。
#
#   "enforce"（默认）—— 门未过即阻断。这是 §9 的原意：
#       "质量门失败时回到膜体系平衡，不允许靠增加 ABFE 窗口掩盖。"
#   "advisory"        —— 照样计算、照样落盘报告、失败照样大声 WARNING 并写进
#       provenance，但**不阻断**。用于"先把管路跑通"的探索阶段。
#
# 为什么提供 advisory 而不是让人注释掉调用：门被注释掉就**没有记录**，
# 事后无从知道当时到底过没过。advisory 下报告仍然完整落盘、
# `membrane_quality_gate_mode` 进 provenance，所以"当时是放行跑的"这件事赖不掉。
#
# ⚠️ advisory **不是**生产资格。任何要报出的 ΔG_bind 都必须在 enforce 下通过。
MEMBRANE_QUALITY_GATE_MODE_ENFORCE = "enforce"
MEMBRANE_QUALITY_GATE_MODE_ADVISORY = "advisory"
MEMBRANE_QUALITY_GATE_MODES = (
    MEMBRANE_QUALITY_GATE_MODE_ENFORCE,
    MEMBRANE_QUALITY_GATE_MODE_ADVISORY,
)


def resolve_membrane_quality_gate_mode(value: Optional[str]) -> str:
    """规范化 §9 质量门模式；未声明即 `enforce`（默认严格）。"""
    if value is None or str(value).strip() == "":
        return MEMBRANE_QUALITY_GATE_MODE_ENFORCE
    normalized = str(value).strip().lower()
    if normalized not in MEMBRANE_QUALITY_GATE_MODES:
        raise ValueError(
            f"membrane_quality_gate={value!r} 非法；允许 {list(MEMBRANE_QUALITY_GATE_MODES)}。"
            "不会静默回落——拼错的值被当成 enforce（或反之）都会让人误判结果的资格。"
        )
    return normalized

# §3.3：这些是"记录"类要求，缺一项就说明输入来源不可追溯 → fail closed。
# §9/§15 的预平衡时长门是对**总平衡时长**的要求，不是"本流程必须自己再跑 100 ns"。
# 输入若已由上游（CHARMM-GUI step6/step7、外部 slurm 作业）平衡过，那段时长应当计入。
#
# 但实践里常见的情形是"上游确实跑完了生产，可时长没记下来"——例如手上只有一个
# `step7_production.gro` 末帧。为此声明里提供两条**互斥**的合法表述，二者必居其一：
#
#   1. `upstream_equilibration_ns`: 正数 —— 明确的上游平衡时长，计入总时长；
#   2. `upstream_equilibration_status = "completed_length_unrecorded"` —— 上游生产已
#      完成但时长不可考。此时**跳过标称时长预检**，并要求 `final_equilibration_job`
#      指向证据（作业 ID / 路径），理由进 provenance。
#
# 两者都不给就 fail closed：沉默等于让未充分平衡的体系混进来。
#
# ⚠️ 无论走哪条，§9 那道**实测**质量门（末段漂移 + 脂质横向弛豫时间尺度）都不受影响，
# 它才是真正的判据；标称时长只是一道便宜的早期护栏。
MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD = "upstream_equilibration_ns"
MEMBRANE_UPSTREAM_STATUS_FIELD = "upstream_equilibration_status"
MEMBRANE_UPSTREAM_STATUS_COMPLETED_UNRECORDED = "completed_length_unrecorded"

MEMBRANE_INPUT_REQUIRED_PROVENANCE_FIELDS = (
    "build_tool",
    "build_parameters",
    "final_equilibration_job",
    "source_structure_id",
    # §1.1：SERT 这类转运体必须记录构象态，不同构象态的 S1 可及性不同、不可混用。
    "conformational_state",
    "membrane_composition",
    # §3.0：结合位点溶剂暴露程度决定空腔填充迟滞的风险等级，必须在选体系时写死。
    "binding_site_solvent_exposure",
    # §9：上下叶脂质数"如何确定"必须有依据，不是随手对半分。
    "leaflet_assignment_basis",
)

# §1.1 要求记录构象态，理由是"不同构象态的 S1 可及性和水化程度不同，不可混用"。
# 但那条是针对转运体（SERT 的 outward-open / occluded / inward-open）写的；换成
# GPCR 或作者本来就不区分构象态的体系时，硬要填一个值只会得到编造的记录。
#
# 所以接受这个显式哨兵：字段仍然**必填**（不能静默缺失），但可以填 `unspecified`，
# provenance 里会明确记着"未声明"而不是假装填过。跨构象态比较的可追溯性由输入文件
# 的递归 SHA256（`runabfe._gromacs_dependency_hashes`）兜底——同一份坐标/拓扑就是
# 同一个构象，换了输入指纹就变。
MEMBRANE_CONFORMATIONAL_STATE_UNSPECIFIED = "unspecified"

MEMBRANE_BINDING_SITE_EXPOSURE_LEVELS = (
    "solvent_accessible",
    "interfacial",
    "lipid_exposed",
    "buried_in_hydrophobic_core",
)

# 常见磷脂头基参考原子（判定脂质分子属于上叶还是下叶）。胆固醇无 P，
# 用其羟基氧兜底。识别不到参考原子的脂质会被单独报出来，不静默丢弃。
LIPID_HEAD_REFERENCE_ATOM_NAMES = ("P", "P1", "P8", "P31", "PO4")
LIPID_HEAD_FALLBACK_ATOM_NAMES = ("O3", "O1", "OH", "ROH")

# §9 要求"保存并审查"且 §13.3 给了数值阈值的量 —— 缺一个就 fail closed。
REQUIRED_MEMBRANE_QUALITY_OBSERVABLES = (
    "apl_nm2",
    "bilayer_thickness_nm",
    "lipid_tail_order_parameter",
    "protein_backbone_rmsd_nm",
    "transmembrane_tilt_deg",
    "pocket_rmsd_nm",
    "ligand_heavy_atom_rmsd_nm",
    "box_xy_area_nm2",
    "box_z_nm",
    "box_volume_nm3",
)

# §9 同样要求保存、但属于分布/计数类（§13.3 没给标量阈值）的诊断量。
# 不判阈值不等于可以不存——缺了就没法回答"膜到底平衡没平衡"。
REQUIRED_MEMBRANE_QUALITY_DIAGNOSTICS = (
    "density_profile_along_normal",
    "leaflet_composition",
    "anomalous_pocket_water_count",
    "membrane_periodic_image_contacts",
    "membrane_undulation_or_residual_tension",
    "lipid_lateral_relaxation_timescale_ns",
    "equilibration_length_ns",
)

# 只在共炼金离子路线下额外要求（§9 末两条）。
COION_MEMBRANE_QUALITY_OBSERVABLES = (
    "coion_abs_z_from_midplane_nm",
    "coion_ligand_min_image_distance_nm",
    "coion_protein_heavy_atom_distance_nm",
    "coion_nearest_phosphorus_distance_nm",
    "coion_first_shell_water_count",
)
COION_MEMBRANE_QUALITY_DIAGNOSTICS = ("coion_z_histogram",)

# §9 要求用脂质横向弛豫时间尺度**论证**预平衡时长够。§13 没给这个倍数，
# 这里取 1.0 作为最低门槛（预平衡至少覆盖一个横向弛豫时间），并明确标注它是
# 本实现补的、不是清单原有阈值——改它要同时改这条注释。
MEMBRANE_EQUILIBRATION_MIN_RELAXATION_MULTIPLE = 1.0

# 膜体系预平衡时长下限。依据 §9（"通用 10 ns 几乎肯定不够"）与 §15
# （"膜体系这几个数都会变大：预平衡 ≥ 100 ns"）。用于**配置时**就把
# "拿可溶体系的 10 ns 默认值去跑膜"挡住，而不是等质量门事后否掉——
# 后者要先烧掉整轮采样才知道。
MEMBRANE_MIN_EQUILIBRATION_NS = 100.0

# 膜与其周期镜像之间至少要留多厚的水层。§9 只要求"检查是否存在膜与周期镜像的
# 异常接触"、没给数；这里取 2.0 nm（≈ 两倍 1.0 nm cutoff），并标注为本实现补的。
MEMBRANE_MIN_WATER_SLAB_NM = 2.0

# 判定"脂质尾链进入疏水核"的头基裕度：|z − 中面| < 半膜厚 − 该裕度即算核内。
MEMBRANE_HYDROPHOBIC_CORE_HEADGROUP_MARGIN_NM = 0.5

# 估计脂质横向弛豫时间尺度时使用的特征位移（约一个脂质直径）。
LIPID_LATERAL_RELAXATION_REFERENCE_DISPLACEMENT_NM = 0.8

# 时间平均 MSD 的线性拟合 lag 窗口（ns）。
#
# 依据是实测（memtest 100 ns 膜蛋白体系）：1–30 ns 区间 MSD ~ t^0.80，
# 也就是短 lag 仍处在亚扩散/笼振区，从 5 ns 起才接近线性；上限 30 ns 是为了
# 保留足够多的独立时间原点（100 ns 轨迹在 30 ns lag 上还有约 70 ns 的原点可用）。
# 换窗口会改变 D 与 τ，所以实际用的窗口**必须**随报告落盘（见 details 里的
# `fit_lag_window_ns` / `fit_lag_window_source`）——"换窗口把数调好看"藏不住。
LIPID_LATERAL_MSD_FIT_LAG_MIN_NS = 5.0
LIPID_LATERAL_MSD_FIT_LAG_MAX_NS = 30.0

# 纯 POPC 双层的横向扩散系数参考值（nm²/ns），用于判读上面那个 τ 的量级：
# 文献量级 D ≈ 0.008 nm²/ns（≈ 0.8e-7 cm²/s）→ 位移 0.8 nm 需 τ ≈ 20 ns。
# ⚠️ 仅作**诊断锚点**，不是阈值：含蛋白膜的脂质会被蛋白减速（annular lipid），
# 小膜片还有有限尺寸效应，所以偏离它不构成"体系有问题"的证据。
PURE_POPC_REFERENCE_LATERAL_DIFFUSION_NM2_PER_NS = 0.008


def linear_drift_per_ns(times_ns, values) -> Dict[str, Any]:
    """对时间序列做一次线性拟合，返回每 ns 漂移斜率。

    §9 明确要求"每个量都必须给出时间序列和末段窗口的漂移斜率，不能只报平均值"。
    """
    t = np.asarray(times_ns, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.shape != y.shape or t.ndim != 1:
        raise ValueError(f"时间与数值序列形状不匹配：{t.shape} vs {y.shape}")
    if t.size < 2:
        raise ValueError("线性漂移至少需要 2 个点")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError("时间序列含非有限值，拒绝拟合")
    span = float(t[-1] - t[0])
    if span <= 0.0:
        raise ValueError(f"时间序列跨度非正：{span}")
    slope, intercept = np.polyfit(t, y, 1)
    return {
        "slope_per_ns": float(slope),
        "intercept": float(intercept),
        "mean": float(np.mean(y)),
        "std": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
        "n_points": int(y.size),
        "span_ns": span,
        "t_start_ns": float(t[0]),
        "t_end_ns": float(t[-1]),
    }


def _tail_window(times_ns, values, window_ns: float):
    """取末段 `window_ns` 的子序列；覆盖不足则抛错（由调用方转成 fail closed）。"""
    t = np.asarray(times_ns, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size != y.size:
        raise ValueError(f"时间与数值长度不一致：{t.size} vs {y.size}")
    if t.size < 2:
        raise ValueError("时间序列至少需要 2 个点")
    total_span = float(t[-1] - t[0])
    if total_span + 1.0e-9 < window_ns:
        raise ValueError(
            f"序列总跨度 {total_span:.3f} ns 覆盖不了要求的末段窗口 {window_ns:.3f} ns。"
            "§9：通用 10 ns 不是膜平衡充分性的证明——请延长预平衡，"
            "不要缩短判据窗口来让门变绿。"
        )
    cutoff = float(t[-1]) - float(window_ns)
    mask = t >= cutoff - 1.0e-9
    if int(np.count_nonzero(mask)) < 2:
        raise ValueError(
            f"末段 {window_ns:.3f} ns 内只有 {int(np.count_nonzero(mask))} 个采样点，"
            "无法拟合漂移斜率。请提高该量的保存频率。"
        )
    return t[mask], y[mask]


def _series_pair(name: str, series: Any):
    if not isinstance(series, dict):
        raise ValueError(
            f"观测量 {name!r} 必须是 {{'times_ns': [...], 'values': [...]}} 形式的时间序列，"
            f"收到 {type(series).__name__}。§9 不接受只报平均值。"
        )
    for key in ("times_ns", "values"):
        if key not in series:
            raise ValueError(f"观测量 {name!r} 缺少 {key!r}")
    return series["times_ns"], series["values"]


def evaluate_membrane_quality_gate(
    observables: Dict[str, Any],
    diagnostics: Optional[Dict[str, Any]] = None,
    literature_apl_nm2: Optional[float] = None,
    require_coion: bool = False,
    tail_window_ns: float = MEMBRANE_QUALITY_GATE_TAIL_WINDOW_NS,
    pure_lipid_reference_apl_nm2: Optional[float] = None,
) -> Dict[str, Any]:
    """按 §9 + §13.3 判膜质量门。

    返回结构化报告；`passed=False` 时**唯一正解是回到膜体系平衡**，
    不允许靠增加 ABFE 窗口或放宽阈值掩盖（§9 末句，已写进 `remediation`）。
    """
    diagnostics = dict(diagnostics or {})
    required_obs = list(REQUIRED_MEMBRANE_QUALITY_OBSERVABLES)
    required_diag = list(REQUIRED_MEMBRANE_QUALITY_DIAGNOSTICS)
    if require_coion:
        required_obs += list(COION_MEMBRANE_QUALITY_OBSERVABLES)
        required_diag += list(COION_MEMBRANE_QUALITY_DIAGNOSTICS)

    missing_obs = [name for name in required_obs if name not in (observables or {})]
    missing_diag = [name for name in required_diag if diagnostics.get(name) is None]
    if missing_obs or missing_diag:
        raise ValueError(
            "膜质量门缺少必须保存的量 —— fail closed（§9）。"
            f"缺时间序列 {missing_obs}；缺诊断量 {missing_diag}。"
        )

    # 阈值判据表：名字 → (判什么, 阈值, 说明)
    checks: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {}
    optional_obs = [
        name
        for name in (APL_PROTEIN_CORRECTED_OBSERVABLE,)
        if name in (observables or {})
    ]
    for name in required_obs + optional_obs:
        times, values = _series_pair(name, observables[name])
        full = linear_drift_per_ns(times, values)
        tail_t, tail_y = _tail_window(times, values, tail_window_ns)
        tail = linear_drift_per_ns(tail_t, tail_y)
        stats[name] = {"full": full, "tail": tail}

    def _add(name, ok, measured, threshold, criterion):
        checks.append(
            {
                "observable": name,
                "criterion": criterion,
                "measured": float(measured),
                "threshold": float(threshold),
                "passed": bool(ok),
            }
        )

    # ---- APL：漂移 ≤ 0.2 %/ns，且与该脂质力场文献值差 ≤ 3% ----
    apl = stats["apl_nm2"]["tail"]
    apl_drift_pct = abs(apl["slope_per_ns"]) / abs(apl["mean"]) * 100.0
    _add(
        "apl_nm2",
        apl_drift_pct <= APL_MAX_DRIFT_PERCENT_PER_NS,
        apl_drift_pct,
        APL_MAX_DRIFT_PERCENT_PER_NS,
        "tail_drift_percent_per_ns",
    )
    # 与文献值比**必须用蛋白横截面校正后的 APL**（见
    # `PROTEIN_CROSS_SECTION_GRID_NM` 的注释）：raw APL 把蛋白占掉的横向
    # 面积也摊给了脂质，含蛋白膜拿它比纯脂文献值必然偏大。
    # 漂移判据仍留在 raw APL 上——那测的是盒面积有没有平衡，掺进蛋白面积的逐帧
    # 噪声只会让它变糊。
    if literature_apl_nm2 is not None:
        corrected = stats.get(APL_PROTEIN_CORRECTED_OBSERVABLE)
        if corrected is not None:
            reference_name = APL_PROTEIN_CORRECTED_OBSERVABLE
            criterion = "deviation_from_literature_percent"
            measured_mean = corrected["tail"]["mean"]
        else:
            # 提取器没给校正序列（老报告 / 手工构造的观测量）——照样判，但把
            # "用的是未校正值"写进 criterion，否则事后分不清这道门当时比的是什么。
            reference_name = "apl_nm2"
            criterion = "deviation_from_literature_percent_uncorrected"
            measured_mean = apl["mean"]
        deviation = abs(measured_mean - float(literature_apl_nm2)) / abs(
            float(literature_apl_nm2)
        ) * 100.0
        _add(
            reference_name,
            deviation <= APL_MAX_DEVIATION_FROM_LITERATURE_PERCENT,
            deviation,
            APL_MAX_DEVIATION_FROM_LITERATURE_PERCENT,
            criterion,
        )

    # ---- 与纯脂文献 APL 的对照：**诊断，不判**（MEM-12，2026-08-02）----
    #
    # `pure_lipid_reference_apl_nm2` 与 `literature_apl_nm2` **刻意是两个不同的字段**：
    # 后者是"要判这道门"的开关，前者只是"记下参考值"。名字分开是为了避免有人
    # 顺手把诊断值填进开关里，从而给含蛋白膜套上一道物理上站不住的门。
    #
    # 为什么含蛋白膜不判与纯脂文献值的 3% 偏差（实测 memtest：校正后 APL 0.5907
    # vs POPC 0.645，低 8.4%）：
    #   * annular lipid 被跨膜蛋白减速并重排，其面积本就不等于体相脂质；
    #   * 蛋白在本体系占约 24% 的横向面积（8.5–9.2 nm² / 36.3 nm²），
    #     "扣掉蛋白面积再除以脂质数"无论怎么定义都残留方法依赖；
    #   * 90 脂小膜片还有有限尺寸效应。
    # 也就是说差百分之几**不构成"体系有问题"的证据**，拿它当门只会制造假阴性。
    #
    # 这道 3% 门应当在**无蛋白 POPC slab** 上启用（memtodolist §8.2 的 lipid slab
    # 工作），那里它才有定义 —— 这不是把门删掉，是把它放到能判的地方。
    if pure_lipid_reference_apl_nm2 is not None:
        corrected = stats.get(APL_PROTEIN_CORRECTED_OBSERVABLE)
        measured_apl = (corrected or {}).get("tail", apl)["mean"]
        reference = float(pure_lipid_reference_apl_nm2)
        stats["apl_vs_pure_lipid_literature"] = {
            "measured_apl_nm2": float(measured_apl),
            "apl_caliber": (
                APL_PROTEIN_CORRECTED_OBSERVABLE if corrected else "apl_nm2"
            ),
            "pure_lipid_reference_apl_nm2": reference,
            "deviation_percent": float(
                abs(measured_apl - reference) / abs(reference) * 100.0
            ),
            "is_gate": False,
            "not_judged_reason": (
                "含蛋白膜与纯脂文献 APL 的偏差不构成体系缺陷证据："
                "annular lipid 被蛋白减速重排、蛋白占本体系约 24% 横向面积、"
                "小膜片有限尺寸效应。这道 3% 门留给无蛋白 POPC slab（§8.2）。"
            ),
        }

    # ---- 双层厚度：漂移 ≤ 0.05 nm / 末段窗口 ----
    thick = stats["bilayer_thickness_nm"]["tail"]
    thick_drift = abs(thick["slope_per_ns"]) * float(tail_window_ns)
    _add(
        "bilayer_thickness_nm",
        thick_drift <= BILAYER_THICKNESS_MAX_DRIFT_NM_PER_TAIL_WINDOW,
        thick_drift,
        BILAYER_THICKNESS_MAX_DRIFT_NM_PER_TAIL_WINDOW,
        "tail_drift_nm_per_window",
    )

    # ---- 跨膜倾角：漂移 ≤ 5° / 末段窗口 ----
    tilt = stats["transmembrane_tilt_deg"]["tail"]
    tilt_drift = abs(tilt["slope_per_ns"]) * float(tail_window_ns)
    _add(
        "transmembrane_tilt_deg",
        tilt_drift <= TRANSMEMBRANE_TILT_MAX_DRIFT_DEG,
        tilt_drift,
        TRANSMEMBRANE_TILT_MAX_DRIFT_DEG,
        "tail_drift_deg_per_window",
    )

    # ---- RMSD 三项：末段均值上限 ----
    for name, limit in (
        ("protein_backbone_rmsd_nm", PROTEIN_BACKBONE_MAX_RMSD_NM),
        ("pocket_rmsd_nm", POCKET_MAX_RMSD_NM),
        ("ligand_heavy_atom_rmsd_nm", LIGAND_HEAVY_ATOM_MAX_RMSD_NM),
    ):
        mean_tail = stats[name]["tail"]["mean"]
        _add(name, mean_tail <= limit, mean_tail, limit, "tail_mean_nm")

    # ---- co-ion 几何（§13.1）----
    # 这几项 §13.1 写的是"**全程** ≥ ..."，所以判的是整条序列的最小值，不是末段、
    # 更不是均值。只看均值会漏掉"末段掉进膜里"这种致命情形（均值仍然漂亮）。
    if require_coion:
        for name, limit in (
            ("coion_ligand_min_image_distance_nm", COION_LIGAND_MIN_IMAGE_RUNTIME_NM),
            ("coion_protein_heavy_atom_distance_nm", COION_PROTEIN_HEAVY_ATOM_MIN_NM),
            ("coion_abs_z_from_midplane_nm", COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM),
            ("coion_nearest_phosphorus_distance_nm", COION_NEAREST_PHOSPHORUS_MIN_NM),
        ):
            _, values = _series_pair(name, observables[name])
            worst = float(np.min(np.asarray(values, dtype=float)))
            _add(name, worst >= limit, worst, limit, "full_series_min_nm_lower_bound")

    # ---- §9：用脂质横向弛豫时间尺度**论证**预平衡时长（诊断，不是门）----
    #
    # ⚠️ **这一项刻意不进 `checks`**（MEM-11，2026-08-02）。理由，按重要性排序：
    #
    # 1. **§9 原文只要求"记录…并用它论证"**，没有要求"≥ 1 倍"。那个倍数
    #    （`MEMBRANE_EQUILIBRATION_MIN_RELAXATION_MULTIPLE`）是本实现自己加的，
    #    当时的注释里就写着"§13 未给此倍数"。
    # 2. **常规膜蛋白平衡的判据不是这个**：看的是 APL / 膜厚 / 序参量 / RMSD 走平
    #    （上面那些 check 就是干这个的，本体系实测余量 2–6 倍），
    #    而不是要求脂质完成一次横向扩散位移。
    # 3. **它是方法依赖量**：τ 由 MSD 拟合窗口、体系拥挤度、盒尺寸共同决定
    #    （见 `_lipid_lateral_relaxation_timescale_ns` 的 docstring：同一条 100 ns
    #    轨迹在旧估计器下 τ 从 30 跳到 11 ns）。用一个方法依赖量当硬门，
    #    会出现"体系没问题、方法波动把它挡住"的假阴性。
    #
    # 与本仓库把 ESS `min_occupancy_normalized` 退役为 diagnostics-only 是同一先例
    # （`docs/TODO.md` TEST-GATE-01）。
    #
    # 🚫 **降级不等于不记录**：下面照样把 τ、比值、与纯 POPC 文献 D 的对照全部落盘，
    #    "永不弛豫的膜"会在这里以极大的 τ 与 < 1 的比值被如实报出（有专门测试钉住）。
    # 🚫 **不得为了让某次运行通过而把它重新塞回 `checks`。** 要重新当门，必须先给出
    #    "这个 τ 估计在跨体系/跨轨迹长度上稳定"的证据，并在 §13 里写下倍数的依据。
    relax_ns = float(diagnostics["lipid_lateral_relaxation_timescale_ns"])
    equil_ns = float(diagnostics["equilibration_length_ns"])
    stats["equilibration_vs_relaxation"] = {
        "equilibration_length_ns": equil_ns,
        "lipid_lateral_relaxation_timescale_ns": relax_ns,
        "ratio_equilibration_over_relaxation": (
            float(equil_ns / relax_ns) if relax_ns > 0 else float("inf")
        ),
        "is_gate": False,
        "retired_reason": (
            "§9 只要求记录并论证，未给倍数；常规膜平衡判据是观测量走平；"
            "τ 是方法依赖量（拟合窗口/拥挤度/盒尺寸），当硬门会产生假阴性。"
            "退役为 diagnostics-only，不得为让运行变绿而塞回 checks。"
        ),
        "lipid_lateral_diffusion": diagnostics.get("lipid_lateral_diffusion"),
    }

    # ---- §9：膜与周期镜像的异常接触必须为 0 ----
    contacts = int(diagnostics["membrane_periodic_image_contacts"])
    checks.append(
        {
            "observable": "membrane_periodic_image_contacts",
            "criterion": "must_be_zero",
            "measured": float(contacts),
            "threshold": 0.0,
            "passed": contacts == 0,
        }
    )

    failed = [c for c in checks if not c["passed"]]
    return {
        "protocol_version": MEMBRANE_QUALITY_GATE_PROTOCOL_VERSION,
        "thresholds_version": ACCEPTANCE_THRESHOLDS_VERSION,
        "tail_window_ns": float(tail_window_ns),
        "passed": not failed,
        "checks": checks,
        "failed_checks": [c["observable"] for c in failed],
        "statistics": stats,
        "diagnostics": diagnostics,
        "coion_required": bool(require_coion),
        "remediation": (
            "质量门失败时回到膜体系平衡（延长预平衡 / 检查建系组成与叶片数），"
            "不允许靠增加 ABFE 窗口、放宽阈值或缩短末段窗口掩盖（§9 末句）。"
        ),
    }


MEMBRANE_QUALITY_GATE_REPORT_FILENAME = "membrane_quality_gate.json"


def run_membrane_quality_gate(
    traj_path: str,
    openmm_topology,
    membrane_quality_inputs: Dict[str, Any],
    *,
    mode: str,
    normal_axis: str = "z",
    ligand_indices=None,
    output_dir: Optional[str] = None,
    frame_interval_ps: Optional[float] = None,
    log=None,
) -> Dict[str, Any]:
    """轨迹 → §9 观测量 → 判定 → 落盘，**唯一实现**（§9 / §6.2）。

    ## 为什么是模块级函数而不是 `ABFEPipeline` 的方法

    原先这段接线只存在于 `ABFEPipeline._evaluate_membrane_quality_gate_after_equilibration`
    里，于是"验证质量门"唯一的办法是**重烧一遍预平衡**（膜体系 10–100 ns，
    5 h 起）。§0.5.7 已经因为"离线重建与生产路径不一致"白花过好几轮，
    所以这里不允许再出现第二份接线：`ABFEPipeline` 与
    `tools/diagnostics/evaluate_membrane_quality_gate.py` 都调这一个函数。

    ## 两种模式（`mode`）

    * `enforce`（默认）——门没过就 raise。这是 §9 末句的原意：
      "质量门失败时回到膜体系平衡，不允许靠增加 ABFE 窗口掩盖。"
    * `advisory`——照样计算、照样落盘报告、失败照样大声 WARNING，但**不阻断**。
      用于"先把管路跑通"的探索阶段。

    为什么提供 advisory 而不是让人注释掉调用：门被注释掉就**没有记录**，
    事后无从知道当时到底过没过。advisory 下报告仍完整落盘、模式进 provenance。
    ⚠️ advisory **不是**生产资格；要报出的 ΔG_bind 必须在 `enforce` 下通过。

    需要的口袋定义 / co-ion 索引由 `membrane_quality_inputs` 显式给出，
    不做运行时推断——同一体系两次跑必须用同一个口袋定义（MEM-00c 那类漂移的教训）。

    `frame_interval_ps` 默认取 `pre_equilibration_frame_interval_ps()`（= 生产设置
    10000 步 × 2 fs = 20 ps/帧）。**必须显式重建时间轴**：mdtraj 读 DCD 给的
    `traj.time` 是整数帧号，直接用会让 §9 的时间轴错 20 倍（详见提取器的 docstring）。
    """
    import mdtraj as md

    emit = log if callable(log) else (lambda message: None)
    mode = resolve_membrane_quality_gate_mode(mode)
    advisory = mode == MEMBRANE_QUALITY_GATE_MODE_ADVISORY

    def _write(report: Dict[str, Any], observables) -> Optional[str]:
        if not output_dir:
            return None
        summary_path = os.path.join(output_dir, MEMBRANE_QUALITY_GATE_REPORT_FILENAME)
        payload: Dict[str, Any] = {"report": report}
        if observables is not None:
            payload["observables"] = observables
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        emit(f"  ✓ 膜质量门摘要已保存: {summary_path}")
        return summary_path

    def _blocked(message: str, observables=None) -> Dict[str, Any]:
        """enforce 下 raise；advisory 下大声记录并落一个"未评估"的报告。

        `observables` 已经算出来时**一定要一起落盘**：观测量是烧了几小时 GPU 才有的，
        判不了门不等于这些数字没价值（例如"跨度覆盖不了末段窗口"时，APL / 膜厚
        的实测值仍然是判断要不要延长平衡的唯一依据）。
        """
        if not advisory:
            raise RuntimeError(message)
        emit(f"  ⚠️ [膜质量门 advisory] 未能完成评估：{message}")
        report = {
            "protocol_version": MEMBRANE_QUALITY_GATE_PROTOCOL_VERSION,
            "mode": MEMBRANE_QUALITY_GATE_MODE_ADVISORY,
            "evaluated": False,
            "passed": None,
            "blocked_reason": message,
        }
        _write(report, observables=observables)
        return report

    inputs = dict(membrane_quality_inputs or {})
    pocket = inputs.get("pocket_atom_indices")
    if not pocket:
        return _blocked(
            "膜体系必须在 membrane_quality_inputs 里给出 pocket_atom_indices："
            "口袋定义直接决定 §9 的 pocket_rmsd 这一道门，不接受运行时推断。"
        )
    ligand_resname = inputs.get("ligand_resname")
    if not ligand_resname:
        if not ligand_indices:
            return _blocked(
                "既没有 ligand_resname，也没有 ligand_indices，无法定位配体。"
                "请在膜输入声明里填 ligand_resname。"
            )
        ligand_resname = openmm_topology.atom(int(ligand_indices[0])).residue.name

    emit(f"\n[膜质量门 · {mode}] 读取预平衡轨迹并计算 §9 观测量...")
    try:
        traj = md.load(traj_path, top=md.Topology.from_openmm(openmm_topology))
        # [MEM-17 已移除] 曾经在这里对账「帧数 == n_steps // reporter_interval」，
        # resume 追加的重复帧会让它 fail closed。判据本身算得没错（重复帧是真的），
        # 但它拦的是主线而不是根因：根因在 `abfe_pipeline.pre_equilibrate` 里
        # `DCDReporter(append=True)` 不截断到 checkpoint 帧边界。
        # 2026-08-03 用户决定删掉这道对账，**不要再加回来**；要修就修上游截断。
        observables, diagnostics = membrane_observables_from_trajectory(
            traj,
            ligand_resname=ligand_resname,
            normal_axis=normal_axis,
            pocket_atom_indices=[int(i) for i in pocket],
            coion_atom_index=inputs.get("coion_atom_index"),
            equilibration_length_ns=inputs.get("equilibration_length_ns"),
            # 身份以 `.top` 组成为准：脂质按分子分叶、水/离子/蛋白用权威原子集合，
            # 不靠残基名（TP3 / Na+ / Cl- / HID / NTRP 都会被残基名判据漏掉）。
            composition=inputs.get("composition"),
            # 时间轴显式重建，不吃 mdtraj 给 DCD 的整数帧号。
            frame_interval_ps=(
                pre_equilibration_frame_interval_ps()
                if frame_interval_ps is None
                else frame_interval_ps
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return _blocked(f"{type(exc).__name__}: {exc}")

    try:
        report = evaluate_membrane_quality_gate(
            observables,
            diagnostics,
            literature_apl_nm2=inputs.get("literature_apl_nm2"),
            require_coion=inputs.get("coion_atom_index") is not None,
            # 诊断专用参考值（不判门），见 evaluate_membrane_quality_gate 里的说明。
            pure_lipid_reference_apl_nm2=inputs.get("pure_lipid_reference_apl_nm2"),
        )
    except Exception as exc:  # noqa: BLE001
        # 判定层报错（最典型：轨迹跨度覆盖不了 20 ns 末段窗口）。观测量已经算出来了，
        # 连同"为什么判不了"一起落盘 —— 那些数字是决定要不要延长平衡的依据。
        return _blocked(f"{type(exc).__name__}: {exc}", observables=observables)
    report["mode"] = mode
    report["evaluated"] = True
    summary_path = _write(report, observables)

    for check in report["checks"]:
        flag = "✓" if check["passed"] else "✗"
        emit(
            f"    {flag} {check['observable']} [{check['criterion']}] "
            f"{check['measured']:.6g} vs 阈值 {check['threshold']:.6g}"
        )

    if not report["passed"]:
        message = (
            f"膜质量门未通过，失败项：{report['failed_checks']}。"
            f"{report['remediation']}"
            + (f" 详情见 {summary_path}。" if summary_path else "")
        )
        if not advisory:
            raise RuntimeError(message)
        emit(
            f"  ⚠️ [膜质量门 advisory] {message}\n"
            "     以 advisory 模式放行继续 —— **这不是生产资格**，"
            "要报出的 ΔG_bind 必须在 enforce 下通过。"
        )
        return report
    emit(f"  ✅ 膜质量门通过（模式 {mode}）")
    return report


# ============================================================================
# 起始态体检：把"跑了几千步的无上下文 NaN"变成"起点就坏，且坏在哪个原子"
#
# 实测教训（§0.5.7）：一个坏掉的 System 最小化后 PE = 4.1e13 kJ/mol、
# max|F| = 3.7e9（落在脂质尾链氢 PA334/H8S），但代码什么都不报，几千步后只给出
# `Particle coordinate is NaN`，于是花了好几轮去猜。正常时 PE ≈ −6.5e5、
# max|F| ≈ 2.5e3 且落在水上 —— 相差 6 个数量级。
#
# ⚠️ **只有一份实现**：预平衡与 Boresch attachment 腿都调这个函数。
# 同一道门写两遍，迟早会有一处漏改（本仓库的 §0.5.5 就是这个毛病）。
# ============================================================================

# 起始态允许的最大单原子受力（kJ/mol/nm）。1e6 相对正常量级留了约 400 倍余量：
# 正常体系不可能触发，坏体系必定触发。
STARTING_STATE_MAX_FORCE_KJ_PER_MOL_NM = 1.0e6


def _assert_periodic_images_are_consistent(context, *, label: str, log=None):
    """排除对与约束对是否都在**同一周期镜像**内。不是就 raise（MEM-15）。

    ## 为什么必须单独查

    OpenMM 的 PME **要求排除/exception 对比 cutoff 近** —— 排除修正是按两原子的
    实际位移去减倒空间贡献的，一旦这对原子跨了盒，修正就完全算错。
    刚性水的 O–H 又只以**约束**存在（实测 memtest：`topology.bonds()` 里涉及水的
    键数 = 0，而约束 28626 个），所以靠 topology 键归组的 PBC 修复会把跨边界的水
    逐原子回卷、撕开 —— 而这件事对**所有**常规诊断都是隐形的：

      * 水没有键力项 ⟹ 键能与最大键长完全正常；
      * PME 的误差是平滑长程项 ⟹ 势能只是偏移，`max|F|` 也正常（实测 5292）；
      * 崩的是**约束求解器**（要在 5.9–12.4 nm 的 O/H 间满足 0.0957 nm）
        ⟹ 不到 1 ps 就 `Particle coordinate is NaN`，且没有任何上下文。

    所以这道检查不是冗余的：它是唯一能在起点看见这类损坏的量。
    """
    import openmm as _mm

    emit = log if callable(log) else (lambda message: None)
    system = context.getSystem()
    state = context.getState(getPositions=True)
    pos = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(_unit_nanometer()),
        dtype=np.float64,
    )
    box = np.asarray(
        [v.value_in_unit(_unit_nanometer())
         for v in state.getPeriodicBoxVectors()],
        dtype=np.float64,
    )
    lengths = np.diag(box)
    if not np.all(lengths > 0):
        return {"checked": False, "reason": "box_vectors_degenerate"}

    # 允许的上界：cutoff（有 PME 时）否则半个最短盒边。超了就说明跨镜像。
    cutoff = None
    for force in system.getForces():
        if isinstance(force, _mm.NonbondedForce):
            try:
                cutoff = float(
                    force.getCutoffDistance().value_in_unit(_unit_nanometer())
                )
            except Exception:  # noqa: BLE001
                cutoff = None
            break
    limit = float(cutoff) if cutoff else 0.5 * float(lengths.min())

    findings = {}
    for kind, pairs in (
        ("nonbonded_exceptions", _nonbonded_exception_pairs(system)),
        ("constraints", _constraint_pairs(system)),
    ):
        if pairs.size == 0:
            findings[kind] = {"n_pairs": 0, "n_over_limit": 0, "max_nm": 0.0}
            continue
        d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
        over = int(np.count_nonzero(d > limit))
        findings[kind] = {
            "n_pairs": int(pairs.shape[0]),
            "n_over_limit": over,
            "max_nm": float(d.max()),
            "limit_nm": limit,
        }
        if over:
            worst = pairs[int(np.argmax(d))]
            raise RuntimeError(
                f"[{label}] {over} 个 {kind} 对跨了周期镜像"
                f"（最远 {d.max():.3f} nm，上限 {limit:.3f} nm，"
                f"最坏的一对是原子 {int(worst[0])}–{int(worst[1])}）。\n"
                "    OpenMM 的 PME 要求排除对比 cutoff 近；约束对更是必须在同一镜像内，"
                "否则约束求解器无法收敛，几百步内就会给出一个**没有上下文的** "
                "`Particle coordinate is NaN`。\n"
                "    最常见原因：PBC 分子完整性修复按 topology 的**键**归组分子，"
                "而刚性水的 O–H 只以**约束**存在（`topology.bonds()` 里 0 个水键），"
                "于是跨边界的水被逐原子回卷、撕开。\n"
                "    修法见 `ABFEPipeline.repair_pbc_molecule_integrity`（MEM-15）："
                "把 System 的约束补成键再交给 `image_molecules()`。\n"
                "    ⚠️ 这类损坏对键能 / 最大键长 / max|F| 全部隐形，只有本检查看得见。"
            )
    emit(
        f"  ✓ 镜像一致性: 排除对 {findings['nonbonded_exceptions']['n_pairs']} 个"
        f"（最远 {findings['nonbonded_exceptions']['max_nm']:.3f} nm）、"
        f"约束对 {findings['constraints']['n_pairs']} 个"
        f"（最远 {findings['constraints']['max_nm']:.3f} nm），上限 {limit:.3f} nm"
    )
    findings["checked"] = True
    return findings


def _unit_nanometer():
    from openmm import unit as _u

    return _u.nanometer


def _nonbonded_exception_pairs(system):
    import openmm as _mm

    for force in system.getForces():
        if isinstance(force, _mm.NonbondedForce):
            n = force.getNumExceptions()
            if n == 0:
                return np.empty((0, 2), dtype=int)
            return np.asarray(
                [force.getExceptionParameters(i)[:2] for i in range(n)], dtype=int
            )
    return np.empty((0, 2), dtype=int)


def _constraint_pairs(system):
    n = system.getNumConstraints()
    if n == 0:
        return np.empty((0, 2), dtype=int)
    return np.asarray(
        [system.getConstraintParameters(i)[:2] for i in range(n)], dtype=int
    )


def assert_starting_state_is_sane(
    context,
    topology,
    *,
    label: str,
    max_force_kj_per_mol_nm: float = STARTING_STATE_MAX_FORCE_KJ_PER_MOL_NM,
    remediation: str = "",
    log=None,
) -> Dict[str, Any]:
    """量一次势能与最大受力，异常即 raise。**只读**，不改坐标/速度/参数。

    `label` 进日志（例如 "最小化后" / "attachment 腿起点 λ=1"）。
    `remediation` 是调用点专属的排查提示，会附在超限报错里。

    返回实测数字，供调用方落盘。
    """
    from openmm import unit as _unit

    emit = log if callable(log) else (lambda message: None)
    state = context.getState(getEnergy=True, getForces=True)
    potential = state.getPotentialEnergy().value_in_unit(_unit.kilojoule_per_mole)
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            _unit.kilojoule_per_mole / _unit.nanometer
        ),
        dtype=float,
    )
    magnitudes = np.linalg.norm(forces, axis=1)
    atoms = _topology_atoms(topology)

    def _atom_label(index: int) -> str:
        atom = atoms[int(index)]
        return f"{atom.residue.name}{atom.residue.index}/{atom.name}"

    finite = np.isfinite(magnitudes)
    if not finite.all():
        raise RuntimeError(
            f"[{label}] 已有 {int((~finite).sum())} 个原子受力为 NaN/Inf —— "
            "起始态就是坏的，继续跑动力学没有意义。"
            "请检查输入坐标与拓扑是否对应（原子顺序、缓存是否串了体系）。"
            + (f"\n{remediation}" if remediation else "")
        )
    worst = int(np.argmax(magnitudes))
    max_force = float(magnitudes.max())
    emit(
        f"  📐 {label}: PE = {potential:.6g} kJ/mol, "
        f"max|F| = {max_force:.4g} kJ/mol/nm "
        f"(idx={worst} {_atom_label(worst)}), "
        f"中位数 {np.median(magnitudes):.4g}"
    )
    if max_force > float(max_force_kj_per_mol_nm):
        order = np.argsort(magnitudes)[::-1][:10]
        worst_list = "\n".join(
            f"      idx={int(i)} {_atom_label(i)}  |F| = {magnitudes[int(i)]:.4g}"
            for i in order
        )
        raise RuntimeError(
            f"[{label}] max|F| = {max_force:.4g} kJ/mol/nm 超过上限 "
            f"{float(max_force_kj_per_mol_nm):.4g}（PE = {potential:.6g} kJ/mol）"
            "——**起始态就是坏的**，继续跑动力学只会得到一个没有上下文的 "
            "`Particle coordinate is NaN`。\n"
            f"    受力最大的 10 个原子：\n{worst_list}\n"
            "    实测参考量级：正常时 PE ≈ −6.5e5 kJ/mol、max|F| ≈ 2.5e3 且落在水上。\n"
            + (f"{remediation}\n" if remediation else "")
        )
    # ---- 镜像一致性（MEM-15）：受力检查看不见的那一类损坏 ----
    #
    # 2026-08-03 实测：243 个刚性水的 O/H 被 PBC 修复放进了不同周期镜像，于是
    # 729 个 PME 排除对跨盒（最远 13.76 nm）。而 **上面那些量全部正常**：
    # PE 只是多了个平滑的长程误差、max|F| = 5292（水没有键力项，所以键能与最大
    # 键长也完全正常）。跑起来后是**约束求解器**先崩 → 不到 1 ps 就 NaN。
    # 也就是说"力看起来正常"根本不能证明起点没坏，必须单独查这一项。
    image_report = _assert_periodic_images_are_consistent(
        context, label=label, log=emit
    )

    return {
        "label": label,
        "potential_energy_kj_per_mol": potential,
        "max_force_kj_per_mol_nm": max_force,
        "periodic_image_consistency": image_report,
        "median_force_kj_per_mol_nm": float(np.median(magnitudes)),
        "max_force_atom_index": worst,
        "max_force_atom": _atom_label(worst),
        "max_force_threshold_kj_per_mol_nm": float(max_force_kj_per_mol_nm),
    }


def _topology_atoms(topology):
    """原子迭代器，兼容 OpenMM（`atoms()` 方法）与 mdtraj（`atoms` 属性）。"""
    attr = topology.atoms
    return list(attr() if callable(attr) else attr)


def _topology_residues(topology):
    attr = topology.residues
    return list(attr() if callable(attr) else attr)


def assign_lipid_leaflets(
    topology,
    positions,
    normal_axis: str = "z",
    lipid_molecules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """按头基参考原子相对膜中面的位置把脂质分到上/下叶（§3.3 / §9）。

    **不假设对半分**：返回实测的每叶计数与组成，以及识别不到头基参考原子的单位
    列表（不静默丢弃）。中面取所有头基参考原子坐标在膜法向上的均值。

    `lipid_molecules`（来自 `classify_system_composition()` 的
    `molecules_by_role["lipid"]`）给出时**按分子**分叶——这是唯一正确的口径：
    Amber Lipid21 把一个 POPC 拆成 `PA` + `PC` + `OL` **三个残基**，按残基数会数出
    3 倍的"脂质数"，APL 直接错 3 倍；而且尾链残基没有磷原子，按残基找头基会
    直接报"找不到头基参考原子"。不给时退回按残基分（只适用于一残基一脂质的体系）。
    """
    axis = {"x": 0, "y": 1, "z": 2}[str(normal_axis).strip().lower()]
    pos_nm = np.asarray(
        positions.value_in_unit(unit.nanometer)
        if hasattr(positions, "value_in_unit")
        else positions,
        dtype=float,
    )

    heads: List[Tuple[int, str, float]] = []
    unassignable: List[Dict[str, Any]] = []
    head_candidates = LIPID_HEAD_REFERENCE_ATOM_NAMES + LIPID_HEAD_FALLBACK_ATOM_NAMES

    if lipid_molecules is not None:
        # 权威口径：按分子。原子区间来自 `.top` 的 [ molecules ] 展开顺序。
        all_atoms = _topology_atoms(topology)
        for entry in lipid_molecules:
            start, stop = int(entry["start"]), int(entry["stop"])
            atom_by_name = {}
            for atom in all_atoms[start:stop]:
                atom_by_name.setdefault(str(atom.name).strip().upper(), int(atom.index))
            ref_index = next(
                (atom_by_name[c] for c in head_candidates if c in atom_by_name), None
            )
            label = str(entry.get("molecule_name", "LIPID")).strip().upper()
            if ref_index is None:
                unassignable.append(
                    {
                        "molecule_name": label,
                        "ordinal": int(entry.get("ordinal", -1)),
                        "atom_range": [start, stop],
                    }
                )
                continue
            heads.append((start, label, float(pos_nm[ref_index][axis])))
    else:
        for residue in _topology_residues(topology):
            name = str(residue.name).strip().upper()
            if name not in KNOWN_LIPID_RESIDUE_NAMES:
                continue
            residue_atoms = residue.atoms
            residue_atoms = list(
                residue_atoms() if callable(residue_atoms) else residue_atoms
            )
            atom_by_name = {
                str(atom.name).strip().upper(): int(atom.index) for atom in residue_atoms
            }
            ref_index = next(
                (atom_by_name[c] for c in head_candidates if c in atom_by_name), None
            )
            if ref_index is None:
                unassignable.append(
                    {"residue_index": int(residue.index), "residue_name": name}
                )
                continue
            heads.append((int(residue.index), name, float(pos_nm[ref_index][axis])))

    if not heads:
        raise ValueError(
            "找不到任何可用的脂质头基参考原子"
            f"（尝试过 {LIPID_HEAD_REFERENCE_ATOM_NAMES + LIPID_HEAD_FALLBACK_ATOM_NAMES}）。"
            "无法判定上下叶——请检查脂质残基/原子命名，不要跳过本检查。"
        )

    coords = np.asarray([h[2] for h in heads], dtype=float)
    midplane = float(np.mean(coords))
    upper: Dict[str, int] = {}
    lower: Dict[str, int] = {}
    for residue_index, name, coordinate in heads:
        bucket = upper if coordinate > midplane else lower
        bucket[name] = bucket.get(name, 0) + 1

    n_upper = int(sum(upper.values()))
    n_lower = int(sum(lower.values()))
    total = n_upper + n_lower
    return {
        "normal_axis": str(normal_axis).strip().lower(),
        "midplane_coordinate_nm": midplane,
        "upper_leaflet_counts": dict(sorted(upper.items())),
        "lower_leaflet_counts": dict(sorted(lower.items())),
        "n_upper": n_upper,
        "n_lower": n_lower,
        "n_total": total,
        "imbalance_fraction": (
            abs(n_upper - n_lower) / total if total else 0.0
        ),
        "unassignable_lipid_residues": unassignable,
        "grouping": "per_molecule" if lipid_molecules is not None else "per_residue",
    }


def verify_membrane_normal_axis(
    topology,
    positions,
    declared_axis: str = "z",
    lipid_molecules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """从坐标**实测**膜法向，并核对与声明的轴一致（§3.3「蛋白跨膜方向」）。

    为什么必须实测：`MonteCarloMembraneBarostat` 把膜法向**硬编码为 z**——它按
    XY 等比例、Z 独立地缩放。若坐标里的双层其实垂直于 x 或 y，barostat 会沿着膜
    平面内的一个方向单独缩放、同时把法向和一个面内方向绑在一起，膜会被压坏，
    而且**不会报任何错**。只看盒子形状（哪条边最长）是不够的：盒子长边与膜法向
    未必一致。

    判据：沿真正的法向，头基参考原子应当分成**两个清晰的簇**（上下叶），
    两簇计数接近相等且簇心间距 ≈ P–P 膜厚（2–5 nm）；面内两个方向上头基是
    弥散分布。所以取"二分后两簇计数最平衡且簇心间距最大"的轴为法向。
    """
    axis_names = ("x", "y", "z")
    declared = str(declared_axis).strip().lower()
    if declared not in axis_names:
        raise ValueError(f"declared_axis={declared_axis!r} 非法；允许 {list(axis_names)}")

    pos_nm = np.asarray(
        positions.value_in_unit(unit.nanometer)
        if hasattr(positions, "value_in_unit")
        else positions,
        dtype=float,
    )
    head_candidates = LIPID_HEAD_REFERENCE_ATOM_NAMES + LIPID_HEAD_FALLBACK_ATOM_NAMES
    heads: List[int] = []
    if lipid_molecules:
        all_atoms = _topology_atoms(topology)
        for entry in lipid_molecules:
            by_name = {}
            for atom in all_atoms[int(entry["start"]):int(entry["stop"])]:
                by_name.setdefault(str(atom.name).strip().upper(), int(atom.index))
            reference = next((by_name[c] for c in head_candidates if c in by_name), None)
            if reference is not None:
                heads.append(reference)
    else:
        for residue in _topology_residues(topology):
            if str(residue.name).strip().upper() not in KNOWN_LIPID_RESIDUE_NAMES:
                continue
            residue_atoms = residue.atoms
            residue_atoms = list(
                residue_atoms() if callable(residue_atoms) else residue_atoms
            )
            by_name = {
                str(a.name).strip().upper(): int(a.index) for a in residue_atoms
            }
            reference = next((by_name[c] for c in head_candidates if c in by_name), None)
            if reference is not None:
                heads.append(reference)
    if len(heads) < 4:
        raise ValueError(
            f"只找到 {len(heads)} 个脂质头基参考原子，无法实测膜法向。"
        )

    head_array = np.asarray(heads, dtype=int)
    per_axis: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(axis_names):
        values = pos_nm[head_array][:, index]
        split = 0.5 * (float(values.min()) + float(values.max()))
        lower = values[values < split]
        upper = values[values >= split]
        if lower.size == 0 or upper.size == 0:
            per_axis[name] = {
                "n_lower": int(lower.size),
                "n_upper": int(upper.size),
                "separation_nm": 0.0,
                "balance": 0.0,
            }
            continue
        total = float(lower.size + upper.size)
        per_axis[name] = {
            "n_lower": int(lower.size),
            "n_upper": int(upper.size),
            # 1.0 = 完美对半；越小越不像双层。
            "balance": 1.0 - abs(lower.size - upper.size) / total,
            "separation_nm": float(np.mean(upper) - np.mean(lower)),
        }

    # 双层的法向：两簇最平衡；平衡度相同时取簇心间距更大的。
    measured = max(
        axis_names,
        key=lambda name: (per_axis[name]["balance"], per_axis[name]["separation_nm"]),
    )
    report = {
        "declared_axis": declared,
        "measured_axis": measured,
        "n_head_atoms": int(head_array.size),
        "per_axis": per_axis,
        "agrees": measured == declared,
    }
    if measured != declared:
        raise ValueError(
            f"声明膜法向为 {declared!r}，但从坐标实测最像双层法向的是 {measured!r}。"
            f"逐轴证据（二分后两簇计数 / 平衡度 / 簇心间距）：{per_axis}。"
            "OpenMM 的 MonteCarloMembraneBarostat 把法向硬编码为 z——轴错了它会沿"
            "膜平面内单独缩放、把法向与一个面内方向绑死，膜会被压坏且不报错。"
            "请在建系时把膜法向对齐 z。"
        )
    logger.info(
        "🧫 膜法向实测确认为 %s（%d 个头基原子；上下叶 %d/%d，簇心间距 %.3f nm）",
        measured, report["n_head_atoms"],
        per_axis[measured]["n_upper"], per_axis[measured]["n_lower"],
        per_axis[measured]["separation_nm"],
    )
    return report


def validate_membrane_input(
    topology,
    positions,
    box_vectors,
    declared: Optional[Dict[str, Any]] = None,
    normal_axis: str = "z",
    composition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """§3.3 膜输入核对：结构一致性 + 来源可追溯性，任一不符即 fail closed。

    归档（.top / 全部 .itp / 位置限制 / 力场 include 的递归 SHA256）由
    `runabfe._gromacs_dependency_hashes()` 负责，本函数只做结构与声明的交叉核对。
    """
    declared = dict(declared or {})

    missing = [
        field
        for field in MEMBRANE_INPUT_REQUIRED_PROVENANCE_FIELDS
        if not declared.get(field)
    ]
    if missing:
        raise ValueError(
            f"膜输入缺少 §3.3 要求记录的字段 {missing}。"
            "输入必须是已经完成膜构建和主要平衡的体系，且来源可追溯——"
            "不依赖通用预平衡去完成脂质重排或蛋白插膜。"
        )

    # §9/§15：上游平衡的两条合法表述，二者必居其一（见上方常量处的说明）。
    upstream_ns = declared.get(MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD)
    upstream_status = str(declared.get(MEMBRANE_UPSTREAM_STATUS_FIELD) or "").strip().lower()
    if upstream_ns is not None:
        try:
            upstream_ns = float(upstream_ns)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD}={declared[MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD]!r} 不是数值"
            ) from exc
        if not math.isfinite(upstream_ns) or upstream_ns < 0.0:
            raise ValueError(f"{MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD} 必须是非负有限数")
    elif upstream_status == MEMBRANE_UPSTREAM_STATUS_COMPLETED_UNRECORDED:
        if not declared.get("final_equilibration_job"):
            raise ValueError(
                f"{MEMBRANE_UPSTREAM_STATUS_FIELD}="
                f"{MEMBRANE_UPSTREAM_STATUS_COMPLETED_UNRECORDED!r} 时必须给 "
                "final_equilibration_job 指向上游平衡的证据（作业 ID / 路径）。"
            )
    else:
        raise ValueError(
            "膜输入必须说明上游平衡情况，二者必居其一："
            f"给出 {MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD}（正数，计入总平衡时长），"
            f"或声明 {MEMBRANE_UPSTREAM_STATUS_FIELD}="
            f"{MEMBRANE_UPSTREAM_STATUS_COMPLETED_UNRECORDED!r}"
            "（上游生产已完成但时长不可考，需同时给 final_equilibration_job）。"
            "沉默等于让未充分平衡的体系混进来。"
            "⚠️ 无论走哪条，§9 的实测质量门都不受影响。"
        )

    conformational_state = str(declared["conformational_state"]).strip()
    conformational_state_declared = (
        conformational_state.lower() != MEMBRANE_CONFORMATIONAL_STATE_UNSPECIFIED
    )
    if not conformational_state_declared:
        logger.warning(
            "⚠️ conformational_state=%r：构象态未声明，已如实记入 provenance。"
            "§1.1 的本意是防止跨构象态混用——本次运行的构象由输入文件指纹唯一确定，"
            "但若日后要与其它构象态的结果比较，这一项必须补上。",
            MEMBRANE_CONFORMATIONAL_STATE_UNSPECIFIED,
        )

    exposure = str(declared["binding_site_solvent_exposure"]).strip().lower()
    if exposure not in MEMBRANE_BINDING_SITE_EXPOSURE_LEVELS:
        raise ValueError(
            f"binding_site_solvent_exposure={declared['binding_site_solvent_exposure']!r} 非法；"
            f"允许 {list(MEMBRANE_BINDING_SITE_EXPOSURE_LEVELS)}（§3.0 决定迟滞风险等级）。"
        )

    # ---- 坐标 / 拓扑原子数一致 ----
    pos_nm = np.asarray(
        positions.value_in_unit(unit.nanometer)
        if hasattr(positions, "value_in_unit")
        else positions,
        dtype=float,
    )
    n_topology_atoms = int(topology.getNumAtoms())
    if pos_nm.shape[0] != n_topology_atoms:
        raise ValueError(
            f"坐标原子数 {pos_nm.shape[0]} 与拓扑原子数 {n_topology_atoms} 不一致（§3.3）。"
        )
    if declared.get("n_atoms") is not None and int(declared["n_atoms"]) != n_topology_atoms:
        raise ValueError(
            f"声明原子数 {declared['n_atoms']} 与拓扑 {n_topology_atoms} 不一致。"
        )

    # ---- 盒型必须是长方体，膜法向对齐 z（§1.1）----
    box_nm = np.asarray(
        box_vectors.value_in_unit(unit.nanometer)
        if hasattr(box_vectors, "value_in_unit")
        else [
            v.value_in_unit(unit.nanometer) if hasattr(v, "value_in_unit") else v
            for v in box_vectors
        ],
        dtype=float,
    )
    off_diagonal = float(np.max(np.abs(box_nm - np.diag(np.diag(box_nm)))))
    if off_diagonal > 1.0e-6:
        raise ValueError(
            f"膜体系盒型必须是长方体（rectangular），实测最大非对角元 {off_diagonal:.6g} nm。"
            "不得用截角八面体/十二面体（§1.1）。"
        )

    # ---- 上下叶脂质数：实测 + 与声明交叉核对，不假设对半分 ----
    # ⚠️ 顺序有意义：**先**叶片划分再核对法向。
    # 叶片划分会报出"哪些脂质找不到头基参考原子"这条**更具体、更可操作**的错；
    # 而法向实测在头基不足时只会报"无法实测膜法向"——那是前者的下游后果，
    # 先报下游会把人引到错误的地方去查。
    leaflets = assign_lipid_leaflets(
        topology,
        positions,
        normal_axis=normal_axis,
        lipid_molecules=(
            (composition or {}).get("molecules_by_role", {}).get("lipid")
        ),
    )
    if leaflets["unassignable_lipid_residues"]:
        raise ValueError(
            f"有 {len(leaflets['unassignable_lipid_residues'])} 个脂质单位找不到头基参考原子，"
            "无法判定所属叶片。请补充原子命名映射，不要让它们静默不计入。"
        )
    # §3.3：膜法向必须**实测**核对，不能只信盒子形状（长边未必是法向）。
    # 轴错了 MonteCarloMembraneBarostat 会沿膜平面内单独缩放、把法向与一个面内
    # 方向绑死，膜被压坏且不报错。
    normal_axis_report = verify_membrane_normal_axis(
        topology,
        positions,
        declared_axis=normal_axis,
        lipid_molecules=(composition or {}).get("molecules_by_role", {}).get("lipid"),
    )

    for key, measured in (("n_upper", leaflets["n_upper"]), ("n_lower", leaflets["n_lower"])):
        if declared.get(key) is not None and int(declared[key]) != measured:
            raise ValueError(
                f"声明 {key}={declared[key]} 与实测 {measured} 不一致（§9：叶片数必须有依据）。"
            )

    # ---- 水 / 离子计数 ----
    # 有 `.top` 组成时以它为准：残基名扫描会在 `TP3` / `Na+` / `Cl-` 这类命名上
    # **静默数出 0**（实测 memtest 体系正是如此），而 `[ molecules ]` 是权威的。
    n_water = 0
    ion_counts: Dict[str, int] = {}
    if composition:
        roles = composition["roles"]
        for molecule_name, count in composition["molecule_counts"].items():
            role = roles.get(molecule_name)
            if role == "water":
                n_water += int(count)
            elif role == "ion":
                key = str(molecule_name).strip().upper()
                ion_counts[key] = ion_counts.get(key, 0) + int(count)
    else:
        for residue in _topology_residues(topology):
            name = str(residue.name).strip().upper()
            if name in WATER_MOLECULE_NAMES:
                n_water += 1
            elif _normalize_ion_name(name) in MONOATOMIC_ION_NAMES:
                ion_counts[name] = ion_counts.get(name, 0) + 1
    if declared.get("n_water") is not None and int(declared["n_water"]) != n_water:
        raise ValueError(f"声明水分子数 {declared['n_water']} 与实测 {n_water} 不一致。")
    if declared.get("ion_counts"):
        expected_ions = {
            str(k).upper(): int(v) for k, v in dict(declared["ion_counts"]).items()
        }
        if expected_ions != ion_counts:
            raise ValueError(
                f"声明离子计数 {expected_ions} 与实测 {ion_counts} 不一致。"
            )

    return {
        "protocol_version": MEMBRANE_INPUT_PROTOCOL_VERSION,
        "n_atoms": n_topology_atoms,
        "box_lengths_nm": [float(np.linalg.norm(row)) for row in box_nm],
        "box_is_rectangular": True,
        "membrane_normal_axis": normal_axis_report,
        "leaflets": leaflets,
        "n_water": n_water,
        "ion_counts": dict(sorted(ion_counts.items())),
        "declared": declared,
        "binding_site_solvent_exposure": exposure,
        "conformational_state": conformational_state,
        "conformational_state_declared": conformational_state_declared,
        "upstream_equilibration_ns": upstream_ns,
        "upstream_equilibration_status": (
            upstream_status or ("declared_ns" if upstream_ns is not None else None)
        ),
        "nominal_equilibration_precheck_applicable": upstream_ns is not None,
        "composition": (
            {
                "roles": composition["roles"],
                "role_evidence": composition["role_evidence"],
                "molecule_counts": composition["molecule_counts"],
                "n_atoms_total": composition["n_atoms_total"],
            }
            if composition
            else None
        ),
    }


# ============================================================================
# §9 观测量提取器：轨迹 → evaluate_membrane_quality_gate() 的输入
#
# 判定层（evaluate_membrane_quality_gate）此前是就绪但拿不到输入的。这一节负责把
# 一条膜轨迹算成它要的时间序列与诊断量。
#
# 设计原则：**宁可 fail closed 也不猜**。
#   - 序参量需要脂质尾链的成键关系。拓扑里没有键就直接报错，**不**用距离阈值
#     冒充共价键——仓库已经在 ATT-11 上吃过这个亏
#     （`GeometricRestraintEstimator._build_bond_adjacency` 的 docstring 记着）。
#   - 口袋定义、co-ion 索引这类无法从轨迹自动可靠推断的，一律要求调用方显式给，
#     给不出就报错，不用"离配体最近的 N 个残基"之类的默契。
#
# ⚠️ 本节的数值口径需要用真实膜体系复核一遍再上生产：APL / 膜厚 / 盒序列是
# 直接几何量，风险低；序参量（用 C–C 键向量而非 C–H，见下）、口袋异常水、
# 横向弛豫时间尺度都是"等价结构指标"，量级对不对必须实测对照文献值。
# ============================================================================


def _require_mdtraj(what: str):
    if not HAS_MDTRAJ:
        raise RuntimeError(
            f"{what} 需要 mdtraj，但当前环境没装。请在 openmm_dev 环境里运行。"
        )
    import mdtraj

    return mdtraj


def _lipid_residue_indices(topology) -> List[int]:
    return [
        int(residue.index)
        for residue in topology.residues
        if str(residue.name).strip().upper() in KNOWN_LIPID_RESIDUE_NAMES
    ]


def _lipid_head_atom_indices_by_residue(topology) -> Dict[int, int]:
    """每个脂质残基的头基参考原子 index（mdtraj topology）。"""
    result: Dict[int, int] = {}
    for residue in topology.residues:
        if str(residue.name).strip().upper() not in KNOWN_LIPID_RESIDUE_NAMES:
            continue
        by_name = {str(a.name).strip().upper(): int(a.index) for a in residue.atoms}
        for candidate in (
            LIPID_HEAD_REFERENCE_ATOM_NAMES + LIPID_HEAD_FALLBACK_ATOM_NAMES
        ):
            if candidate in by_name:
                result[int(residue.index)] = by_name[candidate]
                break
    return result


def _lipid_carbon_carbon_bond_pairs(topology) -> List[Tuple[int, int]]:
    """脂质残基内部的 C–C 成键对，用于序参量。

    只用**拓扑里真实的键**。没有键信息时返回空列表，由调用方 fail closed——
    不用距离阈值冒充共价键。
    """
    lipid_residues = set(_lipid_residue_indices(topology))
    pairs: List[Tuple[int, int]] = []
    for atom_a, atom_b in topology.bonds:
        if int(atom_a.residue.index) not in lipid_residues:
            continue
        if int(atom_b.residue.index) != int(atom_a.residue.index):
            continue
        sym_a = getattr(atom_a.element, "symbol", "") or ""
        sym_b = getattr(atom_b.element, "symbol", "") or ""
        if sym_a.upper() == "C" and sym_b.upper() == "C":
            pairs.append((int(atom_a.index), int(atom_b.index)))
    return pairs


def _series(times_ns, values) -> Dict[str, List[float]]:
    return {
        "times_ns": [float(t) for t in np.asarray(times_ns, dtype=float)],
        "values": [float(v) for v in np.asarray(values, dtype=float)],
    }


def membrane_observables_from_trajectory(
    traj,
    ligand_resname: Optional[str] = None,
    normal_axis: str = "z",
    pocket_atom_indices: Optional[List[int]] = None,
    coion_atom_index: Optional[int] = None,
    equilibration_length_ns: Optional[float] = None,
    composition: Optional[Dict[str, Any]] = None,
    frame_interval_ps: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """把一条膜轨迹算成 `evaluate_membrane_quality_gate()` 需要的观测量与诊断量。

    `traj` 是一个已载入的 mdtraj Trajectory（带 unitcell 与键信息）。
    返回 `(observables, diagnostics)`，可直接喂给判定层。

    ## 时间轴：DCD 必须显式给 `frame_interval_ps`

    §9 的判据全部定义在真实时间轴上（"末段 ≥ 20 ns"、"漂移 ≤ x/ns"），所以时间轴
    错一个倍数，两道门会往**相反**方向坏掉：末段窗口变得过严，而"预平衡 ≥ 一个脂质
    横向弛豫时间"变得过松（MSD 拟合出的 D 被同一个倍数放大）。

    ⚠️ **mdtraj 读 DCD 时不传播真实步长**：`DCDTrajectoryFile.read_as_traj` 给出的
    `traj.time` 是**整数帧号** `[0, 1, 2, …]`。实测 memtest 那条 10 ns / 500 帧
    （10000 步 × 2 fs = 20 ps/帧）的轨迹，`traj.time` 就是 `[0…499]`，
    于是时间轴被当成 0.499 ns —— 比真实值小 20 倍。
    原先这里只校验"存在且单调递增"，帧号完全满足，所以这条守卫对 DCD 是 fail-open。

    因此：
      * 传了 `frame_interval_ps` → 时间轴由它构造（`i × interval`），忽略 `traj.time`。
        这是生产路径的做法，数值来自 `pre_equilibration_frame_interval_ps()`，
        与写轨迹的 reporter/integrator 共用同一组常量。
      * 没传 → 用 `traj.time`，但**整数 dtype 一律拒绝**（那就是上面那个帧号签名）。
    实际采用了哪一条会写进 `diagnostics["time_axis_source"]`，事后查得出来。
    """
    md = _require_mdtraj("膜质量门观测量提取")
    axis = {"x": 0, "y": 1, "z": 2}[str(normal_axis).strip().lower()]
    lateral_axes = [i for i in (0, 1, 2) if i != axis]
    top = traj.topology

    if traj.unitcell_lengths is None:
        raise ValueError(
            "轨迹没有 unitcell 信息，无法计算 APL / 盒序列。\n"
            "    `app.DCDFile` 写每帧 unitcell 时读的是**topology**的盒矢量"
            "（`dcdfile.py:155`：`boxVectors = self._topology.getPeriodicBoxVectors()`），"
            "为 None 就整段不写、header 的 boxFlag 也是 0。\n"
            "    所以这不是轨迹格式问题，而是**写轨迹时那个 topology 没有盒矢量**。"
            "最常见原因：从 `.top` 重建拓扑时没传 `periodicBoxVectors`"
            "（`GromacsTopFile` 只在显式传参时才设盒）。\n"
            "    修法：加载/重建拓扑后 `topology.setPeriodicBoxVectors(box)`，然后重跑预平衡。"
        )
    if frame_interval_ps is not None:
        interval_ps = float(frame_interval_ps)
        if not np.isfinite(interval_ps) or interval_ps <= 0.0:
            raise ValueError(f"frame_interval_ps 必须是正有限值，收到 {frame_interval_ps!r}")
        times = np.arange(traj.n_frames, dtype=float) * interval_ps
        time_axis_source = "declared_frame_interval"
    else:
        raw_time = np.asarray(traj.time)
        if np.issubdtype(raw_time.dtype, np.integer):
            raise ValueError(
                "轨迹的 time 数组是**整数**，这正是 mdtraj 读 DCD 时给出的**帧号**"
                f"（实测形如 [0, 1, 2, …]，本条 {raw_time.size} 帧）。"
                "§9 的判据定义在真实时间轴上（末段 ≥ 20 ns、漂移 ≤ x/ns），"
                "用帧号冒充 ps 会让末段窗口过严、同时让"
                "「预平衡 ≥ 一个脂质横向弛豫时间」过松（D 被同一倍数放大）。\n"
                "    修法：传 `frame_interval_ps`（= reporter 保存间隔 × integrator 步长，"
                "生产路径用 `pre_equilibration_frame_interval_ps()`），不要改判据窗口。"
            )
        times = raw_time.astype(float)
        time_axis_source = "trajectory_time_field"
        interval_ps = None
    if times.size != traj.n_frames or not np.all(np.diff(times) > 0):
        raise ValueError(
            "轨迹的 time 数组缺失或非单调递增。§9 的判据定义在真实时间轴上"
            "（末段 ≥ 20 ns），不允许用帧号冒充 ns。"
        )
    times_ns = times / 1000.0

    lengths = np.asarray(traj.unitcell_lengths, dtype=float)  # (n_frames, 3) nm
    lateral_area = lengths[:, lateral_axes[0]] * lengths[:, lateral_axes[1]]
    normal_length = lengths[:, axis]
    volume = lateral_area * normal_length

    # ---- 叶片划分（用第 0 帧，身份此后固定，不逐帧重选）----
    # 有 `.top` 组成时**按分子**分叶。Amber Lipid21 把一个 POPC 拆成
    # PA + PC + OL 三个残基，按残基会数出 3 倍脂质数、APL 错 3 倍，且尾链残基
    # 没有磷原子会直接报"找不到头基参考原子"（实测 memtest 体系正是如此）。
    #
    # ⚠️ 两条分支**必须**收敛到同一个 `head_units`：`(单元标签, 头基原子 index)`。
    # 下游（`head_indices`、`leaflet_composition`）只许从它派生，不许再有分支专属变量。
    # 原先分子分支产 `head_list`、残基分支产 `head_by_residue`，而下面的
    # `leaflet_composition` 只读后者 —— 分子分支下它从未绑定，于是真实膜体系上
    # 整个 §9 质量门崩在 `UnboundLocalError`（memtest 2026-07-31 与 08-02 各一次，
    # 报告落成 `{"evaluated": false}`）。根因是"身份口径改成按分子了，但有个消费点
    # 没跟着改"，与 §0.5.5 记的"同一件事多个入口，补一个漏一片"同源。
    lipid_molecules = (composition or {}).get("molecules_by_role", {}).get("lipid")
    head_candidates = LIPID_HEAD_REFERENCE_ATOM_NAMES + LIPID_HEAD_FALLBACK_ATOM_NAMES
    head_units: List[Tuple[str, int]] = []
    if lipid_molecules:
        all_atoms = _topology_atoms(top)
        missing_units: List[Dict[str, Any]] = []
        for entry in lipid_molecules:
            start, stop = int(entry["start"]), int(entry["stop"])
            by_name = {}
            for atom in all_atoms[start:stop]:
                by_name.setdefault(str(atom.name).strip().upper(), int(atom.index))
            ref = next((by_name[c] for c in head_candidates if c in by_name), None)
            if ref is None:
                missing_units.append({"atom_range": [start, stop]})
            else:
                # 标签取 moleculetype 名（`POPC`），**不是**构成残基名（PA/PC/OL）——
                # 用残基名会让 leaflet_composition 又数出 3 倍脂质数（§0.5.4 那个坑）。
                head_units.append(
                    (str(entry["molecule_name"]).strip().upper(), ref)
                )
        if missing_units:
            raise ValueError(
                f"{len(missing_units)} 个脂质分子找不到头基参考原子"
                f"（示例原子区间 {missing_units[:3]}）。请补充原子命名，不要静默不计入。"
            )
        lipid_unit_label = "molecule"
    else:
        head_by_residue = _lipid_head_atom_indices_by_residue(top)
        lipid_residues = _lipid_residue_indices(top)
        missing_heads = sorted(set(lipid_residues) - set(head_by_residue))
        if not lipid_residues:
            raise ValueError("轨迹里找不到任何已知脂质残基，这不是膜体系。")
        if missing_heads:
            raise ValueError(
                f"{len(missing_heads)} 个脂质残基找不到头基参考原子"
                f"（残基 index 示例 {missing_heads[:5]}）。请补充原子命名，不要让它们静默不计入。"
            )
        head_units = [
            (str(top.residue(r).name).strip().upper(), head_by_residue[r])
            for r in sorted(head_by_residue)
        ]
        lipid_unit_label = "residue"
    head_indices = np.asarray([idx for _, idx in head_units], dtype=int)
    if head_indices.size == 0:
        raise ValueError("找不到任何脂质头基参考原子，这不是膜体系。")
    head_coords_frame0 = traj.xyz[0][head_indices][:, axis]
    midplane0 = float(np.mean(head_coords_frame0))
    upper_mask = head_coords_frame0 > midplane0
    n_upper = int(np.count_nonzero(upper_mask))
    n_lower = int(head_indices.size - n_upper)
    if n_upper == 0 or n_lower == 0:
        raise ValueError(
            f"叶片划分退化：上叶 {n_upper} / 下叶 {n_lower}。"
            "第 0 帧可能没有居中到膜中面，或者体系不是双层。"
        )

    # ---- 逐帧：中面、APL、膜厚 ----
    head_z_all = traj.xyz[:, head_indices, axis]  # (n_frames, n_heads)
    midplane = head_z_all.mean(axis=1)
    upper_z = head_z_all[:, upper_mask].mean(axis=1)
    lower_z = head_z_all[:, ~upper_mask].mean(axis=1)
    thickness = upper_z - lower_z
    # 每叶面积每脂：横向面积 / 该叶脂质数。上下叶不等时按各自计数分别算再取均值，
    # 不用"总数/2"（§9：叶片数不是随手对半分）。
    apl = 0.5 * (lateral_area / n_upper + lateral_area / n_lower)

    # ---- 序参量：脂质残基内 C–C 键向量相对膜法向 ----
    # ⚠️ 这是 §9 允许的"等价结构指标"，**不是** S_CD。S_CD 需要 C–H 向量，
    # 而 Amber/GROMACS 拓扑里氢是显式的但尾链 C–H 配对需要额外规则；
    # C–C 键向量序参量与 S_CD 同向、量级不同，跨力场比较必须用同一定义。
    cc_pairs = _lipid_carbon_carbon_bond_pairs(top)
    if not cc_pairs:
        raise ValueError(
            "拓扑里没有脂质残基内部的 C–C 键，无法计算尾链序参量。"
            "请提供带键信息的拓扑（GromacsTopFile 产出的 OpenMM topology 有键）；"
            "**不会**用距离阈值冒充共价键（见 ATT-11 的教训）。"
        )
    pair_a = np.asarray([p[0] for p in cc_pairs], dtype=int)
    pair_b = np.asarray([p[1] for p in cc_pairs], dtype=int)
    vectors = traj.xyz[:, pair_b, :] - traj.xyz[:, pair_a, :]
    norms = np.linalg.norm(vectors, axis=2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_theta = vectors[:, :, axis] / norms
    order_parameter = 0.5 * (3.0 * np.nanmean(cos_theta**2, axis=1) - 1.0)

    # ---- RMSD 三项 + 倾角 ----
    ligand_from_composition = (
        (composition or {}).get("atom_indices_by_role", {}).get("ligand")
    )
    if ligand_from_composition:
        ligand_set = set(int(i) for i in ligand_from_composition)
        ligand_heavy = np.asarray(
            sorted(
                int(atom.index)
                for atom in _topology_atoms(top)
                if int(atom.index) in ligand_set
                and (getattr(atom.element, "symbol", "") or "").upper() != "H"
            ),
            dtype=int,
        )
    else:
        if not ligand_resname:
            raise ValueError("未提供 composition 时必须给 ligand_resname。")
        ligand_heavy = top.select(f"resname {ligand_resname} and not element H")
    if ligand_heavy.size == 0:
        raise ValueError(f"找不到配体 {ligand_resname!r} 的重原子。")

    # mdtraj 的 `protein` 关键字按残基名查表，而它的表里 **没有** HID / HIE /
    # ASH / CYX / NTRP / CCYS（只有 HIP），实测 memtest 体系会静默漏掉 85 个原子。
    # 有 `.top` 组成时用组成给的蛋白原子集合，并只取骨架原子名。
    protein_atoms_from_composition = (
        (composition or {}).get("atom_indices_by_role", {}).get("protein")
    )
    protein_heavy = np.asarray([], dtype=int)
    if protein_atoms_from_composition:
        protein_set = set(int(i) for i in protein_atoms_from_composition)
        backbone_names = {"N", "CA", "C", "O"}
        protein_backbone = np.asarray(
            sorted(
                int(atom.index)
                for atom in _topology_atoms(top)
                if int(atom.index) in protein_set
                and str(atom.name).strip().upper() in backbone_names
            ),
            dtype=int,
        )
        # 蛋白横截面用**全部重原子**（不只骨架）：占掉脂质面积的是侧链外表面。
        protein_heavy = np.asarray(
            sorted(
                int(atom.index)
                for atom in _topology_atoms(top)
                if int(atom.index) in protein_set
                and (getattr(atom.element, "symbol", "") or "").upper() != "H"
            ),
            dtype=int,
        )
    else:
        protein_backbone = top.select("protein and backbone")
        # 没有组成兜底时，至少不能静默少选：把"既不是水/脂质/离子/配体、
        # 又没被 mdtraj 认成蛋白"的残基报出来。
        omitted = sorted(
            {
                str(res.name).strip().upper()
                for res in top.residues
                if str(res.name).strip().upper() not in WATER_MOLECULE_NAMES
                and str(res.name).strip().upper() not in KNOWN_LIPID_RESIDUE_NAMES
                and _normalize_ion_name(res.name) not in MONOATOMIC_ION_NAMES
                and str(res.name).strip().upper() != str(ligand_resname or "").upper()
                and not res.is_protein
            }
        )
        if omitted:
            raise ValueError(
                f"这些残基既不是水/脂质/离子/配体，也没被 mdtraj 认成蛋白：{omitted}。"
                "mdtraj 的 protein 残基名表不含 HID/HIE/ASH/CYX/N-/C- 端变体，"
                "直接用它会静默少选骨架原子。请传入 composition"
                "（`classify_system_composition()` 的产出）以使用权威的蛋白原子集合。"
            )
        protein_heavy = np.asarray(
            sorted(int(i) for i in top.select("protein and not element H")), dtype=int
        )
    if protein_backbone.size == 0:
        raise ValueError("找不到蛋白骨架原子，无法计算骨架 RMSD 与跨膜倾角。")
    if pocket_atom_indices is None:
        raise ValueError(
            "必须显式提供 pocket_atom_indices。口袋定义直接决定 pocket_rmsd 这一道门，"
            "不接受「离配体最近的若干残基」之类的运行时默契——那会让同一体系两次跑"
            "用不同的口袋定义（这正是 MEM-00c 那类静默漂移的成因）。"
        )
    pocket_indices = np.asarray(sorted(int(i) for i in pocket_atom_indices), dtype=int)
    if pocket_indices.size == 0:
        raise ValueError("pocket_atom_indices 为空。")

    # ---- RMSD 三项：在**子集副本**上做骨架对齐，绝不动传入的 traj（MEM-10）----
    #
    # ⚠️ `mdtraj.Trajectory.superpose()` **原地修改 `traj.xyz` 并返回 self**。
    # 原先这里写的是 `aligned = traj.superpose(traj, 0, atom_indices=protein_backbone)`，
    # 于是这一行之后所有读 `traj.xyz` 的量都在用"对齐到蛋白骨架"的坐标，而
    # `midplane` / `upper_z` / `lower_z` 是在上面、对齐**之前**算的 —— 两者不在同一
    # 坐标系。实测后果（memtest 100 ns，2026-08-02）：
    #   * 脂质横向弛豫时间尺度 τ 从 11.57 ns 被放大到 **139.36 ns**（12 倍），
    #     直接把 §9 质量门判失败（门里报的正是 139.362）；
    #   * 跨膜倾角在对齐帧里测 → 漂移被系统性压掉；
    #   * 蛋白横截面 / 校正后 APL、疏水核内水、水层间隙、密度分布全部口径错配。
    #
    # 而那次 superpose 对它本来要服务的三个 RMSD **毫无作用**：
    # `md.rmsd(..., atom_indices=X)` 内部会自己在 X 上重新做最优拟合，所以先对齐
    # 不改变返回值（实测 pocket 0.069400 / 0.069400、ligand 0.050201 / 0.050201）。
    # 也就是说它是**纯有害**的一行。
    #
    # 现在只对 backbone ∪ pocket ∪ ligand 这个子集建副本（`atom_slice` 返回新对象），
    # 旋转矩阵只由骨架坐标决定，所以骨架 RMSD 与改前数值等价；内存约为全轨迹的
    # 1/20（实测 1128 + 148 + 41 vs 45354 原子）。
    rmsd_subset = np.unique(
        np.concatenate([protein_backbone, pocket_indices, ligand_heavy])
    )
    _subset_position = {int(g): i for i, g in enumerate(rmsd_subset)}
    _backbone_in_subset = np.asarray(
        [_subset_position[int(i)] for i in protein_backbone], dtype=int
    )
    _pocket_in_subset = np.asarray(
        [_subset_position[int(i)] for i in pocket_indices], dtype=int
    )
    _ligand_in_subset = np.asarray(
        [_subset_position[int(i)] for i in ligand_heavy], dtype=int
    )
    ref = traj.atom_slice(rmsd_subset)
    backbone_rmsd = md.rmsd(ref, ref, 0, atom_indices=_backbone_in_subset)
    ref.superpose(ref, 0, atom_indices=_backbone_in_subset)  # 只改副本

    # 口袋 / 配体是 **pose 漂移**（§9："口袋 RMSD、配体 RMSD/关键相互作用"），
    # 所以对齐蛋白骨架之后**不再重新拟合**，直接量位移（MEM-13）。
    # 原先用 `md.rmsd(..., atom_indices=pocket)`，它会在口袋/配体自身上再做一次
    # 最优拟合 —— 测到的是内部构象变化，而不是相对受体的 pose 漂移。
    # 实测差别（memtest 100 ns 末段 20 ns）：不重拟合 0.0857 / 0.0833 nm，
    # 重拟合 0.0760 / 0.0493 nm（配体差 1.7 倍）。阈值 0.20 / 0.25 不变。
    def _pose_drift_nm(subset_indices):
        delta = ref.xyz[:, subset_indices, :] - ref.xyz[0, subset_indices, :][None]
        return np.sqrt(np.mean(np.sum(delta**2, axis=2), axis=1))

    pocket_rmsd = _pose_drift_nm(_pocket_in_subset)
    ligand_rmsd = _pose_drift_nm(_ligand_in_subset)

    # 跨膜倾角：蛋白骨架坐标的第一主轴与膜法向的夹角（0–90°）。
    tilt_deg = np.empty(traj.n_frames, dtype=float)
    normal_vector = np.zeros(3)
    normal_vector[axis] = 1.0
    for frame in range(traj.n_frames):
        coords = traj.xyz[frame][protein_backbone]
        centered = coords - coords.mean(axis=0)
        # 最大奇异值对应的右奇异向量 = 第一主轴。
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        principal = vt[0]
        cosine = abs(float(np.dot(principal, normal_vector)))
        tilt_deg[frame] = math.degrees(math.acos(min(1.0, max(0.0, cosine))))

    # ---- APL 的蛋白横截面校正（§13.3 的绝对值门只有校正后才有意义）----
    # raw APL 把蛋白占掉的横向面积也摊到脂质头上，含蛋白膜因此系统性偏大
    # （实测 memtest 0.826 vs POPC 纯脂文献 0.645）。校正 = 先从横向面积里扣掉
    # 该叶片 slab 内的蛋白占据面积，再除以该叶脂质数。
    # 脂质原子集合：组成优先，名字兜底。下面的"膜与周期镜像水层间隙"复用同一集合。
    lipid_from_composition = (
        (composition or {}).get("atom_indices_by_role", {}).get("lipid")
    )
    if lipid_from_composition:
        lipid_atoms = np.asarray(sorted(int(i) for i in lipid_from_composition), dtype=int)
    else:
        lipid_atoms = np.asarray(
            [
                int(atom.index)
                for atom in _topology_atoms(top)
                if str(atom.residue.name).strip().upper() in KNOWN_LIPID_RESIDUE_NAMES
            ],
            dtype=int,
        )
    if lipid_atoms.size == 0:
        raise ValueError("找不到任何脂质原子，无法判断膜与周期镜像的水层间隙。")
    _lipid_atom_set = set(int(i) for i in lipid_atoms.tolist())
    lipid_heavy = np.asarray(
        [
            int(atom.index)
            for atom in _topology_atoms(top)
            if int(atom.index) in _lipid_atom_set
            and (getattr(atom.element, "symbol", "") or "").upper() != "H"
        ],
        dtype=int,
    )
    cross_upper, cross_lower = _protein_leaflet_cross_sections_nm2(
        traj, protein_heavy, lipid_heavy, axis, lateral_axes, midplane, upper_z, lower_z
    )
    apl_corrected = 0.5 * (
        (lateral_area - cross_upper) / n_upper
        + (lateral_area - cross_lower) / n_lower
    )
    # 栅格边长是**唯一**的方法参数（最近原子归属没有探针半径），所以敏感性检查
    # 就是"换成 2× 粗栅格复算若干帧"。判读绝对值门必须连它一起看。
    sens_frames = np.unique(
        np.linspace(
            0,
            traj.n_frames - 1,
            min(traj.n_frames, PROTEIN_CROSS_SECTION_SENSITIVITY_MAX_FRAMES),
        ).astype(int)
    )
    apl_sensitivity: Dict[str, Any] = {
        "method": "nearest_reference_atom_partition",
        "grid_nm": PROTEIN_CROSS_SECTION_GRID_NM,
        "coarse_grid_nm": (
            PROTEIN_CROSS_SECTION_GRID_NM
            * PROTEIN_CROSS_SECTION_GRID_SENSITIVITY_FACTOR
        ),
        "n_frames_sampled": int(sens_frames.size),
        "n_protein_reference_atoms": int(protein_heavy.size),
        "n_lipid_reference_atoms": int(lipid_heavy.size),
    }
    up_s, lo_s = _protein_leaflet_cross_sections_nm2(
        traj,
        protein_heavy,
        lipid_heavy,
        axis,
        lateral_axes,
        midplane,
        upper_z,
        lower_z,
        grid_nm=(
            PROTEIN_CROSS_SECTION_GRID_NM
            * PROTEIN_CROSS_SECTION_GRID_SENSITIVITY_FACTOR
        ),
        frames=sens_frames,
    )
    area_s = lateral_area[sens_frames]
    apl_sensitivity["apl_nm2_fine_grid"] = float(
        np.mean(apl_corrected[sens_frames])
    )
    apl_sensitivity["apl_nm2_coarse_grid"] = float(
        np.mean(0.5 * ((area_s - up_s) / n_upper + (area_s - lo_s) / n_lower))
    )
    apl_sensitivity["protein_cross_section_nm2_fine_grid"] = float(
        np.mean(0.5 * (cross_upper[sens_frames] + cross_lower[sens_frames]))
    )
    apl_sensitivity["protein_cross_section_nm2_coarse_grid"] = float(
        np.mean(0.5 * (up_s + lo_s))
    )

    observables: Dict[str, Any] = {
        "apl_nm2": _series(times_ns, apl),
        "apl_protein_corrected_nm2": _series(times_ns, apl_corrected),
        "bilayer_thickness_nm": _series(times_ns, thickness),
        "lipid_tail_order_parameter": _series(times_ns, order_parameter),
        "protein_backbone_rmsd_nm": _series(times_ns, backbone_rmsd),
        "transmembrane_tilt_deg": _series(times_ns, tilt_deg),
        "pocket_rmsd_nm": _series(times_ns, pocket_rmsd),
        "ligand_heavy_atom_rmsd_nm": _series(times_ns, ligand_rmsd),
        "box_xy_area_nm2": _series(times_ns, lateral_area),
        "box_z_nm": _series(times_ns, normal_length),
        "box_volume_nm3": _series(times_ns, volume),
    }

    # ---- 疏水核内异常水 ----
    water_oxygens = _resolve_water_oxygen_indices(top, composition)
    half_core = np.maximum(
        0.0, 0.5 * thickness - MEMBRANE_HYDROPHOBIC_CORE_HEADGROUP_MARGIN_NM
    )
    water_z = traj.xyz[:, water_oxygens, axis]
    inside = np.abs(water_z - midplane[:, None]) < half_core[:, None]
    core_water = inside.sum(axis=1)

    # ---- 膜与周期镜像的水层间隙 ----
    # `lipid_atoms` 在上面（APL 蛋白横截面校正处）已经解析过，两处共用同一集合。
    lipid_z = traj.xyz[:, lipid_atoms, axis]
    water_gap = normal_length - (lipid_z.max(axis=1) - lipid_z.min(axis=1))
    image_contact_frames = int(np.count_nonzero(water_gap < MEMBRANE_MIN_WATER_SLAB_NM))

    # ---- 沿法向的质量密度分布（按组分）----
    density_profile = _density_profile_along_normal(traj, axis, midplane)

    # ---- 脂质横向弛豫时间尺度（由头基横向 MSD 估计）----
    relaxation_ns, relaxation_details = _lipid_lateral_relaxation_timescale_ns(
        traj, head_indices, lateral_axes, times_ns
    )

    # 从 `head_units` 派生，与 `head_indices` / 叶片 mask 同源。
    # 不要在这里重新按残基名找头基——那正是本函数崩过两次的地方。
    leaflet_composition = {"upper": {}, "lower": {}}
    for label, atom_index in head_units:
        bucket = (
            "upper"
            if traj.xyz[0][atom_index][axis] > midplane0
            else "lower"
        )
        leaflet_composition[bucket][label] = (
            leaflet_composition[bucket].get(label, 0) + 1
        )

    diagnostics: Dict[str, Any] = {
        "density_profile_along_normal": density_profile,
        "leaflet_composition": leaflet_composition,
        "anomalous_pocket_water_count": int(np.max(core_water)),
        "membrane_periodic_image_contacts": image_contact_frames,
        "membrane_undulation_or_residual_tension": {
            "midplane_std_nm": float(np.std(midplane)),
            "min_water_slab_nm": float(np.min(water_gap)),
            "water_slab_threshold_nm": MEMBRANE_MIN_WATER_SLAB_NM,
        },
        "lipid_lateral_relaxation_timescale_ns": relaxation_ns,
        "lipid_lateral_diffusion": relaxation_details,
        "equilibration_length_ns": (
            float(equilibration_length_ns)
            if equilibration_length_ns is not None
            else float(times_ns[-1] - times_ns[0])
        ),
        "n_upper": n_upper,
        "n_lower": n_lower,
        "core_water_per_frame_mean": float(np.mean(core_water)),
        "lipid_unit": lipid_unit_label,
        # 时间轴来自哪里必须可追溯：mdtraj 读 DCD 给的是帧号，用错了两道门会往
        # 相反方向坏（末段窗口过严 / 弛豫时间过松），事后只看数字分辨不出来。
        "time_axis_source": time_axis_source,
        "frame_interval_ps": interval_ps,
        "trajectory_span_ns": float(times_ns[-1] - times_ns[0]),
        "protein_cross_section_upper_nm2_mean": float(np.mean(cross_upper)),
        "protein_cross_section_lower_nm2_mean": float(np.mean(cross_lower)),
        "n_protein_heavy_atoms": int(protein_heavy.size),
        "apl_protein_cross_section_sensitivity": apl_sensitivity,
        "apl_correction_definition": (
            "apl_protein_corrected_nm2 = mean over leaflets of "
            "(lateral box area - protein-owned lateral area in that leaflet slab) "
            "/ (number of lipid units in that leaflet). The protein-owned area is a "
            "Voronoi-style nearest-reference-atom partition of the lateral plane on a "
            f"{PROTEIN_CROSS_SECTION_GRID_NM} nm periodic grid: every grid cell is "
            "assigned to the closest heavy atom among the protein and lipid heavy "
            "atoms whose normal-axis coordinate lies inside that leaflet slab. There "
            "is NO probe radius - an outward-dilation definition adds a rim along the "
            "protein perimeter and measurably overestimates the protein area. The grid "
            "spacing is the only method parameter; see "
            "apl_protein_cross_section_sensitivity for the coarse-grid cross-check. "
            "Never compare a corrected APL against an uncorrected one, nor values "
            "produced by different partition definitions."
        ),
        "order_parameter_definition": (
            "intra-lipid C–C bond-vector order parameter relative to the membrane "
            "normal; an §9-permitted equivalent structural indicator, NOT S_CD "
            "(which needs C–H vectors). Do not compare across definitions."
        ),
    }

    if coion_atom_index is not None:
        observables.update(
            _coion_observables_from_trajectory(
                traj, int(coion_atom_index), axis, midplane, times_ns, ligand_heavy,
                composition=composition,
            )
        )
        diagnostics["coion_z_histogram"] = _coion_z_histogram(
            traj, int(coion_atom_index), axis, midplane
        )

    return observables, diagnostics


def _resolve_water_oxygen_indices(topology, composition: Optional[Dict[str, Any]]):
    """水氧原子索引；解析不到就报错，**绝不返回空集**。

    实测 memtest 体系的水叫 `TP3`，而 mdtraj 的 `_WATER_RESIDUES` 只有 `TIP3`——
    `select("water and element O")` 会**静默返回 0 个原子**，于是疏水核内异常水
    与 co-ion 首层水配位数都变成 0，两道门形同虚设且不报错。
    """
    from_composition = (
        (composition or {}).get("atom_indices_by_role", {}).get("water")
    )
    atoms = _topology_atoms(topology)
    if from_composition:
        water_set = set(int(i) for i in from_composition)
        indices = [
            int(atom.index)
            for atom in atoms
            if int(atom.index) in water_set
            and (getattr(atom.element, "symbol", "") or "").upper() == "O"
        ]
    else:
        indices = [
            int(atom.index)
            for atom in atoms
            if str(atom.residue.name).strip().upper() in WATER_MOLECULE_NAMES
            and (getattr(atom.element, "symbol", "") or "").upper() == "O"
        ]
    if not indices:
        present = sorted({str(a.residue.name).strip().upper() for a in atoms})
        raise ValueError(
            "找不到任何水氧原子。膜体系必然有水，所以这是识别失败而不是事实——"
            f"拓扑里出现的残基名：{present[:20]}。"
            f"已知水名：{sorted(WATER_MOLECULE_NAMES)}。"
            "请传入 composition（`classify_system_composition()` 的产出）"
            "或把该水模型名加进 WATER_MOLECULE_NAMES。"
        )
    return np.asarray(sorted(indices), dtype=int)


def _nearest_owner_area_nm2(
    protein_xy,
    lipid_xy,
    lateral_lengths,
    grid_nm: float = PROTEIN_CROSS_SECTION_GRID_NM,
) -> float:
    """横向平面上"离蛋白原子比离任何脂质原子都近"的面积，nm²（周期性）。

    Voronoi 式划分：把横向平面打成边长 `grid_nm` 的栅格，每个格心归给最近的那个
    参考原子（蛋白重原子 ∪ 脂质重原子），返回归给蛋白的面积。
    **没有探针半径**——边界自动落在两类原子中间，所以不存在"外扩多少"这个可调量
    （见 `PROTEIN_CROSS_SECTION_GRID_NM` 的注释：外扩法会沿周长多算一圈）。

    周期性靠把参考原子在横向 3×3 复制一遍实现（格心只取主胞），
    这样贴边的蛋白/脂质在最近邻判断里不会被盒边截断。
    """
    from scipy.spatial import cKDTree

    protein_xy = np.asarray(protein_xy, dtype=float).reshape(-1, 2)
    lipid_xy = np.asarray(lipid_xy, dtype=float).reshape(-1, 2)
    if protein_xy.shape[0] == 0:
        return 0.0
    lx = float(lateral_lengths[0])
    ly = float(lateral_lengths[1])
    nx = max(1, int(round(lx / grid_nm)))
    ny = max(1, int(round(ly / grid_nm)))
    cell_x, cell_y = lx / nx, ly / ny
    if lipid_xy.shape[0] == 0:
        # 该 slab 里没有脂质原子 → 整个横向面积都不归脂质。这不是正常膜体系，
        # 但也不该在这里猜；调用方会因为 APL 明显异常而察觉。
        return lx * ly

    refs = np.vstack([protein_xy, lipid_xy])
    is_protein = np.zeros(refs.shape[0], dtype=bool)
    is_protein[: protein_xy.shape[0]] = True
    # 横向 3×3 周期镜像
    shifts = np.array(
        [(i * lx, j * ly) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float
    )
    refs_periodic = (refs[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
    owner_periodic = np.tile(is_protein, shifts.shape[0])

    centers_x = (np.arange(nx) + 0.5) * cell_x
    centers_y = (np.arange(ny) + 0.5) * cell_y
    grid = np.stack(
        np.meshgrid(centers_x, centers_y, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    _, nearest = cKDTree(refs_periodic).query(grid, k=1)
    return float(owner_periodic[nearest].sum()) * cell_x * cell_y


def _protein_leaflet_cross_sections_nm2(
    traj,
    protein_heavy,
    lipid_heavy,
    axis: int,
    lateral_axes,
    midplane,
    upper_z,
    lower_z,
    grid_nm: float = PROTEIN_CROSS_SECTION_GRID_NM,
    frames=None,
):
    """逐帧、逐叶给出蛋白在该叶片 slab 内占掉的横向面积（nm²）。

    上叶 slab = [中面, 上叶头基平面]，下叶 slab = [下叶头基平面, 中面]。
    只取落在该 slab 内的原子——蛋白在膜外的胞内/胞外结构域不占脂质面积，
    把它们算进来会高估蛋白横截面。
    """
    protein_heavy = np.asarray(protein_heavy, dtype=int)
    lipid_heavy = np.asarray(lipid_heavy, dtype=int)
    lengths = np.asarray(traj.unitcell_lengths, dtype=float)
    frame_list = (
        list(range(traj.n_frames)) if frames is None else [int(f) for f in frames]
    )
    upper = np.zeros(len(frame_list), dtype=float)
    lower = np.zeros(len(frame_list), dtype=float)
    if protein_heavy.size == 0:
        return upper, lower
    for out_i, frame in enumerate(frame_list):
        cell = lengths[frame][lateral_axes]
        p_coords = traj.xyz[frame][protein_heavy]
        l_coords = (
            traj.xyz[frame][lipid_heavy]
            if lipid_heavy.size
            else np.empty((0, 3), dtype=float)
        )
        p_z, l_z = p_coords[:, axis], l_coords[:, axis]
        for out, lo, hi in (
            (upper, midplane[frame], upper_z[frame]),
            (lower, lower_z[frame], midplane[frame]),
        ):
            out[out_i] = _nearest_owner_area_nm2(
                p_coords[(p_z >= lo) & (p_z <= hi)][:, lateral_axes],
                l_coords[(l_z >= lo) & (l_z <= hi)][:, lateral_axes]
                if l_coords.size
                else np.empty((0, 2), dtype=float),
                cell,
                grid_nm=grid_nm,
            )
    return upper, lower


def _density_profile_along_normal(traj, axis: int, midplane) -> Dict[str, Any]:
    """按组分给出沿膜法向的质量密度分布（相对膜中面）。"""
    top = traj.topology
    groups: Dict[str, List[int]] = {"water": [], "lipid": [], "protein": [], "ion": []}
    # 用与其它地方相同的名表/归一化，避免这里又出现第三套判据
    # （`TP3` 不在旧的 water 集合、`Na+`/`Cl-` 带符号不在旧的 ion 集合，
    #  HID/NTRP/CCYS 不在 mdtraj 的蛋白表——三处都会静默漏掉）。
    ion_names = MONOATOMIC_ION_NAMES
    water_names = WATER_MOLECULE_NAMES
    for atom in _topology_atoms(top):
        name = str(atom.residue.name).strip().upper()
        if name in water_names:
            groups["water"].append(int(atom.index))
        elif name in KNOWN_LIPID_RESIDUE_NAMES:
            groups["lipid"].append(int(atom.index))
        elif _normalize_ion_name(name) in ion_names:
            groups["ion"].append(int(atom.index))
        elif normalize_protein_residue_name(name) is not None:
            groups["protein"].append(int(atom.index))

    bins = np.linspace(-5.0, 5.0, 101)
    centers = 0.5 * (bins[:-1] + bins[1:])
    profile: Dict[str, Any] = {"bin_centers_nm": centers.tolist()}
    for group, indices in groups.items():
        if not indices:
            profile[group] = [0.0] * centers.size
            continue
        idx = np.asarray(indices, dtype=int)
        masses = np.asarray(
            [float(getattr(top.atom(int(i)).element, "mass", 0.0) or 0.0) for i in idx]
        )
        relative_z = traj.xyz[:, idx, axis] - midplane[:, None]
        hist = np.zeros(centers.size, dtype=float)
        for frame in range(traj.n_frames):
            counts, _ = np.histogram(relative_z[frame], bins=bins, weights=masses)
            hist += counts
        profile[group] = (hist / max(1, traj.n_frames)).tolist()
    return profile


def _lipid_lateral_relaxation_timescale_ns(
    traj, head_indices, lateral_axes, times_ns
) -> Tuple[float, Dict[str, Any]]:
    """由头基横向 MSD 估计脂质横向弛豫时间尺度（§9）。返回 `(tau_ns, details)`。

    做法：**时间平均** MSD —— 对每个 lag `L` 用**所有时间原点**
    `msd[L] = mean_t |x(t+L) − x(t)|²`，再在一个**声明的 lag 窗口**内做带截距的
    线性拟合得 2D 扩散系数（MSD = 4·D·Δt），最后换算成"位移一个脂质直径所需时间"。
    这是一个**量级估计**，用于论证预平衡时长，不是精确扩散系数测量。

    ## 为什么不能用单一参考帧 + 过原点拟合全部 lag（MEM-11）

    原实现取 `reference = lateral[0]`（每个 lag 只有**一个**样本）并用过原点最小
    二乘拟合**全部** lag（权重 ∝ lag²，长 lag 主导）。两个问题：

    * 脂质横向 MSD 在短 lag 是**亚扩散**的（实测 memtest 100 ns，1–30 ns 区间
      MSD ~ t^0.80），过原点拟合把这段一起吃进去必然偏；
    * 长 lag 只有极少数独立样本，却被 lag² 权重放大。

    实测后果：同一条 100 ns 轨迹，只改"用到前多少 ns"，τ 就是
    30.1 → 38.0 → 24.1 → 13.2 → 10.8 → 11.6 ns（10/20/40/60/80/100 ns），
    **非单调乱跳** —— 这样的量不该当硬门（见 `evaluate_membrane_quality_gate`
    里 `equilibration_vs_relaxation` 那段说明）。
    改成时间平均 + 5–30 ns 窗口后同一条轨迹给 D = 0.008664 nm²/ns → τ = 18.467 ns，
    与 POPC 文献 D ≈ 0.008 nm²/ns 给出的 τ ≈ 20 ns 吻合。
    """
    if times_ns.size < 3:
        raise ValueError("估计脂质横向弛豫时间尺度至少需要 3 帧")
    diffs = np.diff(times_ns)
    if not np.allclose(diffs, diffs[0], rtol=1e-6, atol=1e-12):
        raise ValueError(
            "时间轴不是等间隔的，时间平均 MSD 需要等间隔帧。"
            "请提供等间隔轨迹或显式给出 frame_interval_ps。"
        )
    dt_ns = float(diffs[0])
    span_ns = float(times_ns[-1] - times_ns[0])

    lag_lo, lag_hi = (
        LIPID_LATERAL_MSD_FIT_LAG_MIN_NS,
        LIPID_LATERAL_MSD_FIT_LAG_MAX_NS,
    )
    window_source = "declared"
    if span_ns < 2.0 * lag_lo:
        # 轨迹太短装不下声明窗口：按跨度缩放，并**如实记录**用了哪个窗口。
        # 不静默套用声明窗口（那会拟合到根本没有的 lag），也不报错（短轨迹在
        # 合成测试与早期诊断里是合法用法）。
        lag_lo, lag_hi = 0.1 * span_ns, 0.6 * span_ns
        window_source = "scaled_to_trajectory_span"

    lag_min_frames = max(1, int(round(lag_lo / dt_ns)))
    lag_max_frames = min(traj.n_frames - 1, int(round(lag_hi / dt_ns)))
    if lag_max_frames <= lag_min_frames:
        lag_min_frames = 1
        lag_max_frames = max(2, traj.n_frames - 1)
        window_source = "degenerate_span_used_all_lags"
    # 拟合点数上限：MSD 是 O(n_frames × n_heads) 一个 lag，30 个点足够定斜率。
    lag_frames = np.unique(
        np.linspace(lag_min_frames, lag_max_frames, 30).astype(int)
    )

    lateral = traj.xyz[:, head_indices, :][:, :, lateral_axes].astype(np.float64)
    msd = np.empty(lag_frames.size, dtype=float)
    for i, lag in enumerate(lag_frames):
        delta = lateral[int(lag):] - lateral[: -int(lag)]
        msd[i] = float(np.mean(np.sum(delta**2, axis=2)))
    lag_ns = lag_frames * dt_ns

    if lag_frames.size >= 2:
        slope, intercept = np.polyfit(lag_ns, msd, 1)
    else:
        slope, intercept = msd[0] / lag_ns[0], 0.0
    # 亚扩散指数（诊断用）：MSD ~ t^alpha，纯扩散 alpha = 1。
    positive = msd > 0
    if int(np.count_nonzero(positive)) >= 2:
        alpha = float(
            np.polyfit(np.log(lag_ns[positive]), np.log(msd[positive]), 1)[0]
        )
    else:
        alpha = float("nan")

    d_lateral = max(float(slope) / 4.0, 1.0e-12)  # nm²/ns
    reference_displacement = LIPID_LATERAL_RELAXATION_REFERENCE_DISPLACEMENT_NM
    tau_ns = float(reference_displacement**2 / (4.0 * d_lateral))
    details = {
        "method": "time_averaged_msd_multiple_origins",
        "lateral_diffusion_nm2_per_ns": float(d_lateral),
        "msd_power_law_exponent": alpha,
        "fit_lag_window_ns": [float(lag_ns[0]), float(lag_ns[-1])],
        "fit_lag_window_source": window_source,
        "n_fit_points": int(lag_frames.size),
        "fit_intercept_nm2": float(intercept),
        "reference_displacement_nm": float(reference_displacement),
        "trajectory_span_ns": span_ns,
        # 判读锚点：POPC 纯脂文献 D ≈ 0.008 nm²/ns → τ ≈ 20 ns。
        "pure_popc_reference_diffusion_nm2_per_ns": (
            PURE_POPC_REFERENCE_LATERAL_DIFFUSION_NM2_PER_NS
        ),
    }
    return tau_ns, details


def _coion_observables_from_trajectory(
    traj, coion_index: int, axis: int, midplane, times_ns, ligand_heavy,
    composition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """co-ion 的 §13.1 几何量时间序列。"""
    top = traj.topology
    coion_xyz = traj.xyz[:, coion_index, :]
    abs_z = np.abs(coion_xyz[:, axis] - midplane)

    def _min_distance(indices):
        if len(indices) == 0:
            return np.full(traj.n_frames, np.inf)
        delta = traj.xyz[:, np.asarray(indices, dtype=int), :] - coion_xyz[:, None, :]
        # 逐帧 minimum image（长方体盒；膜体系已强制 rectangular）。
        lengths = np.asarray(traj.unitcell_lengths, dtype=float)[:, None, :]
        delta -= lengths * np.round(delta / lengths)
        return np.min(np.linalg.norm(delta, axis=2), axis=1)

    protein_heavy = top.select("protein and not element H")
    phosphorus = top.select("element P")
    water_oxygens = _resolve_water_oxygen_indices(top, composition)

    if water_oxygens.size:
        delta = traj.xyz[:, water_oxygens, :] - coion_xyz[:, None, :]
        lengths = np.asarray(traj.unitcell_lengths, dtype=float)[:, None, :]
        delta -= lengths * np.round(delta / lengths)
        distances = np.linalg.norm(delta, axis=2)
        first_shell = (distances <= COION_FIRST_SHELL_WATER_CUTOFF_NM).sum(axis=1)
    else:
        first_shell = np.zeros(traj.n_frames, dtype=int)

    return {
        "coion_abs_z_from_midplane_nm": _series(times_ns, abs_z),
        "coion_ligand_min_image_distance_nm": _series(
            times_ns, _min_distance(ligand_heavy)
        ),
        "coion_protein_heavy_atom_distance_nm": _series(
            times_ns, _min_distance(protein_heavy)
        ),
        "coion_nearest_phosphorus_distance_nm": _series(
            times_ns, _min_distance(phosphorus)
        ),
        "coion_first_shell_water_count": _series(times_ns, first_shell),
    }


def _coion_z_histogram(traj, coion_index: int, axis: int, midplane) -> Dict[str, Any]:
    """§9 末条：co-ion 的 z 分布直方图，不只是瞬时距离。"""
    relative_z = traj.xyz[:, coion_index, axis] - midplane
    bins = np.linspace(-6.0, 6.0, 61)
    counts, edges = np.histogram(relative_z, bins=bins)
    return {
        "bin_centers_nm": (0.5 * (edges[:-1] + edges[1:])).tolist(),
        "counts": counts.astype(int).tolist(),
        "min_abs_z_nm": float(np.min(np.abs(relative_z))),
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
        # [MEM-00h，2026-08-06] 统一到基础力场的 1.0 nm、无 switching——理由见
        # BEUTLER_SOFTCORE_CUTOFF_NM 定义处。switchingDistance 仍显式设成等于
        # cutoff（而不是干脆不调用）：保持跟 ibs_engine._create_softcore_force
        # 同样的写法，任何读 getSwitchingDistance() 的下游代码都不会读到一个
        # 语义不明的哨兵值。
        sc_force.setCutoffDistance(BEUTLER_SOFTCORE_CUTOFF_NM * unit.nanometer)
        sc_force.setUseSwitchingFunction(False)
        sc_force.setSwitchingDistance(BEUTLER_SOFTCORE_CUTOFF_NM * unit.nanometer)

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


# ============================================================================
# P0-12a：配体构象诊断 + 跨腿构象一致性门（memtodolist §3.0 末条）
#
# §3.0 早就写着：「若配体本身亲脂，溶剂腿（纯水）里可能出现构象塌缩或自聚集；
# 记录溶剂腿配体回转半径与内部氢键随 λ 的变化。」**那条诊断一直没实现**，
# 于是 2026-08-04 那轮膜运行里它真的发生了、跑完了、汇总了，全程没有任何报警：
#
#     配体重原子最大内距（去电荷 12 个 replica、600 帧汇总）
#       膜运行 复合物腿  p5–p95 = 1.34–1.44 nm   （口袋撑着，伸展）
#       膜运行 溶剂腿    p5–p95 = 0.62–0.71 nm   ← 塌缩，且 12 个 replica 零涨落
#       可溶   复合物腿  p5–p95 = 1.34–1.44 nm
#       可溶   溶剂腿    p5–p95 = 0.73–1.39 nm   ← 健康：从塌缩到伸展都采到
#
# 后果不是"某个数偏了"，而是**两条腿在给不同的构象族做热力学循环**：
# 塌缩构象把极性基团聚拢 ⟹ 配体–水静电耦合强 3 倍
# （⟨U_lig-env⟩ = −569 ± 90 vs −190 ± 34 kJ/mol）⟹ 去电荷 191.05 vs 62.80 kJ/mol。
# ΔG_bind = ΔG_solv − ΔG_cplx 在这种情况下没有意义。
#
# 判据为什么取「两条腿的 [p5, p95] 必须重叠」而不是某个绝对阈值：
#   · 它是**物理要求**——循环有意义的前提是两条腿采的是同一个分子的同一个构象族，
#     溶剂腿的分布应当**覆盖**复合物腿采到的那些构象（通常还更宽）；
#   · 它不是为了让哪次运行变绿而调出来的数：实测可溶基线 1.34–1.44 vs 0.73–1.39
#     **有重叠**（1.34–1.39）而膜运行 1.34–1.44 vs 0.62–0.71 **完全不重叠**，
#     两者是被同一条判据分开的，没有留任何可调旋钮。
# ⚠️ 不要为了让某次运行通过而把百分位放宽成 [p0, p100] 或改成"均值差 < 某个 nm"。
#    不重叠的正解是修采样（双起点 / 加构象采样维度），不是放宽门。
# ============================================================================

LIGAND_CONFORMER_DIAGNOSTICS_VERSION = 1
# 判重叠用的百分位区间。取 p5/p95 而不是 min/max：端点单帧极值噪声大，
# 用它判"分布是否重叠"会把一次偶然的伸展当成充分采样。
LIGAND_CONFORMER_OVERLAP_PERCENTILES = (5.0, 95.0)
# 内部极性接触（§3.0 的"内部氢键"代理量）：N/O 重原子对，键路径隔 ≥ 4 键。
# 用重原子距离而不是显式 H 几何，是为了不依赖氢的命名/位置，换体系不会挂。
LIGAND_INTERNAL_POLAR_CONTACT_NM = 0.35
LIGAND_INTERNAL_POLAR_MIN_BOND_SEPARATION = 4


def ligand_conformer_metrics(
    xyz_nm: Any,
    heavy_local_indices: Sequence[int],
    masses_amu: Optional[Sequence[float]] = None,
    polar_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Dict[str, np.ndarray]:
    """逐帧算配体构象度量。`xyz_nm` 形状 (n_frames, n_ligand_atoms, 3)。

    索引都是**配体局部**索引（0..n_ligand_atoms-1），不是全体系索引 ——
    调用方只需要把配体那几十个原子的坐标切出来，不用搬整条轨迹。

    返回三个逐帧数组：
      * `max_internal_heavy_distance_nm`：重原子间最大距离。**主判据**，
        对"塌缩 vs 伸展"最敏感（实测 0.66 vs 1.28 nm）。
      * `radius_of_gyration_nm`：§3.0 明确要求记录的那个量（质量加权）。
      * `internal_polar_contact_count`：§3.0 的"内部氢键"代理量。
    """
    xyz = np.asarray(xyz_nm, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"xyz_nm 形状非法：{xyz.shape}，期望 (n_frames, n_atoms, 3)")
    heavy = np.asarray(list(heavy_local_indices), dtype=int)
    if heavy.size < 2:
        raise ValueError(
            f"配体重原子少于 2 个（{heavy.size}），无法定义构象度量。"
        )

    h = xyz[:, heavy, :]
    d = np.linalg.norm(h[:, :, None, :] - h[:, None, :, :], axis=-1)
    max_internal = d.max(axis=(1, 2))

    if masses_amu is None:
        w = np.ones(heavy.size, dtype=np.float64)
    else:
        w = np.asarray([float(masses_amu[i]) for i in heavy], dtype=np.float64)
    com = (h * w[None, :, None]).sum(axis=1) / w.sum()
    rg = np.sqrt(
        ((np.linalg.norm(h - com[:, None, :], axis=-1) ** 2) * w[None, :]).sum(axis=1)
        / w.sum()
    )

    if polar_pairs:
        pairs = np.asarray(polar_pairs, dtype=int)
        pd = np.linalg.norm(xyz[:, pairs[:, 0], :] - xyz[:, pairs[:, 1], :], axis=-1)
        contacts = (pd <= LIGAND_INTERNAL_POLAR_CONTACT_NM).sum(axis=1)
    else:
        contacts = np.zeros(xyz.shape[0], dtype=int)

    return {
        "max_internal_heavy_distance_nm": max_internal,
        "radius_of_gyration_nm": rg,
        "internal_polar_contact_count": contacts.astype(float),
    }


def _series_summary(values: Any) -> Dict[str, Any]:
    v = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = LIGAND_CONFORMER_OVERLAP_PERCENTILES
    return {
        "n": int(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        f"p{lo:g}": float(np.percentile(v, lo)),
        "p50": float(np.percentile(v, 50.0)),
        f"p{hi:g}": float(np.percentile(v, hi)),
        "max": float(np.max(v)),
    }


def ligand_conformer_summary(
    metrics: Dict[str, Any],
    *,
    leg: str,
    source: str = "",
) -> Dict[str, Any]:
    """把逐帧度量汇总成可落盘、可跨腿比较的 summary。"""
    return {
        "protocol_version": LIGAND_CONFORMER_DIAGNOSTICS_VERSION,
        "leg": str(leg),
        "source": str(source),
        "overlap_percentiles": list(LIGAND_CONFORMER_OVERLAP_PERCENTILES),
        "observables": {
            name: _series_summary(values) for name, values in metrics.items()
        },
    }


def ligand_conformer_fingerprint(
    positions_nm: Any,
    ligand_indices: Sequence[int],
    heavy_indices: Optional[Sequence[int]] = None,
    decimals: int = 3,
) -> Dict[str, Any]:
    """[P0-12b] 配体**起始构象**的指纹，进溶剂腿缓存身份。

    用内部距离矩阵（排序后取整）而不是原始坐标：刚体平移/旋转不该让缓存失效，
    **构象**变了才该失效。而构象确实必须进身份 —— 实测同一个分子换一个起始构象，
    溶剂腿去电荷从 62.80 变成 191.05 kJ/mol（P0-12），旧口径里这两次却都判"缓存有效"。
    """
    import hashlib

    pos = np.asarray(positions_nm, dtype=np.float64)
    idx = np.asarray(sorted(int(i) for i in ligand_indices), dtype=int)
    heavy = (
        np.asarray(sorted(int(i) for i in heavy_indices), dtype=int)
        if heavy_indices is not None
        else idx
    )
    h = pos[heavy]
    d = np.linalg.norm(h[:, None, :] - h[None, :, :], axis=-1)
    iu = np.triu_indices(len(heavy), k=1)
    rounded = np.round(np.sort(d[iu]), int(decimals))
    blob = ",".join(f"{v:.{int(decimals)}f}" for v in rounded)
    return {
        "sha256": hashlib.sha256(blob.encode("utf-8")).hexdigest(),
        "n_heavy_atoms": int(len(heavy)),
        "decimals": int(decimals),
        "max_internal_heavy_distance_nm": float(d.max()),
        "radius_of_gyration_nm": float(
            np.sqrt(np.mean(np.linalg.norm(h - h.mean(axis=0), axis=-1) ** 2))
        ),
        "invariant_under": "rigid_translation_and_rotation",
    }


def evaluate_cross_leg_conformer_consistency(
    complex_summary: Optional[Dict[str, Any]],
    solvent_summary: Optional[Dict[str, Any]],
    observable: str = "max_internal_heavy_distance_nm",
) -> Dict[str, Any]:
    """[P0-12a] 两条腿的配体构象分布是否重叠；不重叠即不许汇总 ΔG_bind。

    缺任何一侧的 summary 时**不判**（记 `not_evaluated` + 原因），因为判不了门
    与"门过了"必须能区分开；这条路径留给 traditional / 后处理模式（它们没有
    replica 轨迹可读）。IBS 生产路径两条腿都会给出 summary。
    """
    lo, hi = LIGAND_CONFORMER_OVERLAP_PERCENTILES
    lo_key, hi_key = f"p{lo:g}", f"p{hi:g}"
    report: Dict[str, Any] = {
        "protocol_version": LIGAND_CONFORMER_DIAGNOSTICS_VERSION,
        "observable": observable,
        "overlap_percentiles": [lo, hi],
        "evaluated": False,
        "passed": None,
        "reason": "",
    }
    if not complex_summary or not solvent_summary:
        missing = [
            name
            for name, value in (("complex", complex_summary), ("solvent", solvent_summary))
            if not value
        ]
        report["reason"] = (
            f"not_evaluated_missing_conformer_summary_for_{'_and_'.join(missing)}"
        )
        return report
    try:
        c = complex_summary["observables"][observable]
        s = solvent_summary["observables"][observable]
    except (KeyError, TypeError):
        report["reason"] = f"not_evaluated_observable_{observable}_absent"
        return report

    c_lo, c_hi = float(c[lo_key]), float(c[hi_key])
    s_lo, s_hi = float(s[lo_key]), float(s[hi_key])
    overlap = min(c_hi, s_hi) - max(c_lo, s_lo)
    union = max(c_hi, s_hi) - min(c_lo, s_lo)
    report.update(
        {
            "evaluated": True,
            "complex_interval_nm": [c_lo, c_hi],
            "solvent_interval_nm": [s_lo, s_hi],
            "overlap_nm": float(overlap),
            "union_nm": float(union),
            "overlap_fraction_of_union": (
                float(overlap / union) if union > 0 else None
            ),
            "complex_mean_nm": float(c["mean"]),
            "solvent_mean_nm": float(s["mean"]),
            "complex_std_nm": float(c["std"]),
            "solvent_std_nm": float(s["std"]),
            "passed": bool(overlap > 0.0),
        }
    )
    if not report["passed"]:
        report["reason"] = (
            "cross_leg_conformer_ensembles_do_not_overlap: "
            f"复合物腿 {observable} 的 [p{lo:g}, p{hi:g}] = [{c_lo:.3f}, {c_hi:.3f}] nm，"
            f"溶剂腿 = [{s_lo:.3f}, {s_hi:.3f}] nm，两者不相交。"
            "两条腿采的不是同一个构象族 ⟹ ΔG_bind = ΔG_solv − ΔG_cplx 没有意义（§3.0）。"
            "⚠️ 严格说不重叠有两种读法：(1) 某条腿的构象系综**没收敛**（被困在一个 basin）；"
            "(2) 两相的构象偏好**真的**差这么多，那个差值本该是 ΔG_bind 的一部分。"
            "在当前每窗口 0.5 ns 的采样下这两者分辨不开 —— 看 "
            "`per_replica_mean_max_internal_heavy_distance_nm`：若各 replica 挤在同一个"
            "窄区间（实测 0.657–0.672 nm，σ=0.005）就是 (1)。所以门保守阻断，"
            "由**溶剂腿双起点验证**（折叠/伸展各跑一遍，ΔG 差 ≤ 2σ）来区分，"
            "**不是**放宽本判据的百分位区间。"
        )
    return report


def assert_cross_leg_conformer_consistency(report: Dict[str, Any]) -> None:
    """门：不重叠就 raise。判不了门（`evaluated=False`）不阻断，但会如实记录。"""
    if report.get("evaluated") and not report.get("passed"):
        raise ValueError(
            "配体构象跨腿一致性门未通过（P0-12a / §3.0）：\n    "
            + str(report.get("reason"))
        )


def combine_binding_free_energy(
    *,
    dg_complex_kJ_mol: float,
    dg_solvent_kJ_mol: float,
    err_complex_kJ_mol: float = 0.0,
    err_solvent_kJ_mol: float = 0.0,
    dg_boresch_kJ_mol: float = 0.0,
    boresch_already_included_in_complex: bool = True,
    apbs_correction_kJ_mol: float = 0.0,
    complex_conformer_summary: Optional[Dict[str, Any]] = None,
    solvent_conformer_summary: Optional[Dict[str, Any]] = None,
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

    # [P0-12a / §3.0] 汇总 ΔG_bind 之前先判跨腿构象一致性：不重叠就不许汇总。
    # 门放在这里（热力学循环闭合的唯一实现）而不是各调用点，理由与 ATT-09 相同 ——
    # 公式只有一份，门也只能有一份。
    conformer_report = evaluate_cross_leg_conformer_consistency(
        complex_conformer_summary, solvent_conformer_summary
    )
    assert_cross_leg_conformer_consistency(conformer_report)

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
        # [P0-12a] 循环闭合的前提是两条腿采的是同一个构象族。判不了门时
        # `evaluated=False` 会如实记录，不会伪装成"通过"。
        "ligand_conformer_cross_leg": conformer_report,
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


# 参数匹配水模型时的容差。O 的 σ/ε 与电荷在各模型间的差别远大于此
# （例如 TIP3P σ=0.315075 vs SPC/E σ=0.316572），所以这个容差既能吸收
# 力场文件与 XML 的有效位差异，又不会把两个模型混为一谈。
WATER_MODEL_MATCH_CHARGE_TOLERANCE_E = 1.0e-4
WATER_MODEL_MATCH_SIGMA_TOLERANCE_NM = 1.0e-6
WATER_MODEL_MATCH_EPSILON_TOLERANCE_KJ_MOL = 1.0e-4


def water_model_parameters_from_topology(
    parsed: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """从已解析的 GROMACS 拓扑里取出水的 O/H 电荷与 O 的 σ/ε。

    找不到水 moleculetype 或缺 atomtype 参数时返回 None（由调用方 fail closed）。
    """
    water_name = next(
        (
            name
            for name in parsed["moleculetypes"]
            if str(name).strip().upper() in WATER_MOLECULE_NAMES
        ),
        None,
    )
    if water_name is None:
        return None
    atoms = parsed["moleculetypes"][water_name].get("atoms") or []
    atomtypes = parsed.get("atomtypes") or {}

    oxygen = next(
        (a for a in atoms if str(a.get("atom_name", "")).strip().upper().startswith("O")),
        None,
    )
    hydrogens = [
        a for a in atoms if str(a.get("atom_name", "")).strip().upper().startswith("H")
    ]
    if oxygen is None or not hydrogens:
        return None
    o_type = atomtypes.get(oxygen["type"])
    if o_type is None:
        return None
    return {
        "moleculetype": water_name,
        "n_sites": len(atoms),
        "charge_o_e": float(oxygen["charge"]),
        "charge_h_e": float(hydrogens[0]["charge"]),
        "sigma_o_nm": float(o_type["sigma_nm"]),
        "epsilon_o_kj_mol": float(o_type["epsilon_kj_mol"]),
    }


def openmm_water_model_parameters(xml_relative_path: str) -> Optional[Dict[str, Any]]:
    """从 OpenMM 自带的水模型 XML 里取同一组参数，用于与拓扑比对。

    刻意**读 XML 而不是硬编码一张参数表**：这样"选出的 XML"与"比对用的参数"
    构造性地来自同一处，不会出现表抄错或版本漂移。
    """
    import xml.etree.ElementTree as ET

    data_dir = os.path.join(os.path.dirname(app.__file__), "data")
    path = os.path.join(data_dir, xml_relative_path)
    if not os.path.isfile(path):
        return None
    root = ET.parse(path).getroot()

    residue = next(
        (r for r in root.iter("Residue") if r.get("name") in ("HOH", "WAT")), None
    )
    if residue is None:
        return None
    site_types: Dict[str, str] = {}
    charges: Dict[str, float] = {}
    for atom in residue:
        name, atom_type, charge = atom.get("name"), atom.get("type"), atom.get("charge")
        if name is None or atom_type is None or charge is None:
            continue
        site_types[name] = atom_type
        charges[name] = float(charge)
    if not site_types:
        return None

    sigma_epsilon: Dict[str, Tuple[float, float]] = {}
    for atom in root.iter("Atom"):
        atom_type = atom.get("type")
        sigma, epsilon = atom.get("sigma"), atom.get("epsilon")
        if atom_type in set(site_types.values()) and sigma is not None and epsilon is not None:
            sigma_epsilon[atom_type] = (float(sigma), float(epsilon))

    oxygen_name = next((n for n in site_types if n.upper().startswith("O")), None)
    hydrogen_name = next((n for n in site_types if n.upper().startswith("H")), None)
    if oxygen_name is None or hydrogen_name is None:
        return None
    o_params = sigma_epsilon.get(site_types[oxygen_name])
    if o_params is None:
        return None
    return {
        "xml": xml_relative_path,
        "n_sites": len(site_types),
        "charge_o_e": charges[oxygen_name],
        "charge_h_e": charges[hydrogen_name],
        "sigma_o_nm": o_params[0],
        "epsilon_o_kj_mol": o_params[1],
    }


def _water_models_match(topology_params, xml_params) -> bool:
    return (
        int(topology_params["n_sites"]) == int(xml_params["n_sites"])
        and abs(topology_params["charge_o_e"] - xml_params["charge_o_e"])
        <= WATER_MODEL_MATCH_CHARGE_TOLERANCE_E
        and abs(topology_params["charge_h_e"] - xml_params["charge_h_e"])
        <= WATER_MODEL_MATCH_CHARGE_TOLERANCE_E
        and abs(topology_params["sigma_o_nm"] - xml_params["sigma_o_nm"])
        <= WATER_MODEL_MATCH_SIGMA_TOLERANCE_NM
        and abs(topology_params["epsilon_o_kj_mol"] - xml_params["epsilon_o_kj_mol"])
        <= WATER_MODEL_MATCH_EPSILON_TOLERANCE_KJ_MOL
    )


def identify_water_model_by_parameters(
    top_file: str,
    gmx_include_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """按**实际参数**（O/H 电荷 + O 的 σ/ε + 位点数）识别水模型。

    为什么需要：原实现只看 `#include` 的文件名词干。那对
    `amber14sb_OL15_fs1.ff/tip3p.itp` 有效，但 CHARMM-GUI 的 AMBER 转换器把
    TIP3P 的水叫 **`TP3`**（`toppar/TP3.itp`），文件名匹配直接 fail closed。

    这是同一类根因的第三次出现（前两次是脂质按残基计数、水/离子计数静默为 0）：
    **靠名字判身份在换一套体系时就会错**。参数是权威的，且比对用的候选参数直接
    从 OpenMM 自带 XML 读出，不硬编码。

    返回 `{"xml", "matched", "topology_params", "candidates"}`；
    `matched=False` 时由调用方决定报错措辞。
    """
    parsed = parse_gromacs_topology(top_file, gmx_include_dir)
    topology_params = water_model_parameters_from_topology(parsed)
    if topology_params is None:
        return {
            "xml": None,
            "matched": False,
            "reason": "topology_water_parameters_unavailable",
            "topology_params": None,
            "candidates": {},
        }

    candidates: Dict[str, Any] = {}
    matches: List[str] = []
    for key, xml_relative_path in GMX_TO_OPENMM_WATER_XML.items():
        xml_params = openmm_water_model_parameters(xml_relative_path)
        candidates[key] = xml_params
        if xml_params and _water_models_match(topology_params, xml_params):
            matches.append(key)

    if len(matches) == 1:
        return {
            "xml": GMX_TO_OPENMM_WATER_XML[matches[0]],
            "matched": True,
            "reason": "parameter_fingerprint",
            "model_key": matches[0],
            "topology_params": topology_params,
            "candidates": candidates,
        }
    return {
        "xml": None,
        "matched": False,
        "reason": (
            f"ambiguous_parameter_match:{sorted(matches)}"
            if matches
            else "no_parameter_match"
        ),
        "topology_params": topology_params,
        "candidates": candidates,
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
        # 文件名词干认不出来时，按**实际参数**识别（O/H 电荷 + O 的 σ/ε + 位点数）。
        # 实测理由：CHARMM-GUI 的 AMBER 转换器把 TIP3P 的水命名为 `TP3`
        # （`toppar/TP3.itp`），文件名匹配必然落空，但它的参数
        # （q_O=-0.834, q_H=+0.417, σ_O=0.315075240658 nm, ε_O=0.635968 kJ/mol）
        # 与 `amber14/tip3p.xml` 逐位吻合。参数是权威的，名字不是。
        identification = identify_water_model_by_parameters(top_file)
        if identification["matched"]:
            params = identification["topology_params"]
            logger.info(
                "💧 水模型按参数识别为 %s（moleculetype=%s, %d 位点, "
                "q_O=%+.6f, q_H=%+.6f, σ_O=%.9f nm, ε_O=%.6f kJ/mol）——"
                "`#include` 文件名词干认不出，故用参数判定。",
                identification["xml"], params["moleculetype"], params["n_sites"],
                params["charge_o_e"], params["charge_h_e"],
                params["sigma_o_nm"], params["epsilon_o_kj_mol"],
            )
            return identification["xml"], f"parameter_match:{params['moleculetype']}"
        detail = ""
        if identification["topology_params"]:
            p = identification["topology_params"]
            detail = (
                f" 拓扑里的水是 moleculetype={p['moleculetype']!r}、{p['n_sites']} 位点、"
                f"q_O={p['charge_o_e']:+.6f}, q_H={p['charge_h_e']:+.6f}, "
                f"σ_O={p['sigma_o_nm']:.9f} nm, ε_O={p['epsilon_o_kj_mol']:.6f} kJ/mol，"
                f"与已知模型都不匹配（{identification['reason']}）。"
            )
        raise ValueError(
            f"在 {top_file} 的 #include 里没认出任何水模型，按参数也没匹配上；"
            f"已知的有 {sorted(GMX_TO_OPENMM_WATER_XML)}。{detail}"
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
        # 走唯一入口：拓扑若含 `[ pairs ]` funct 2（OpenMM 不支持）会先做等价转换。
        top = load_gromacs_topology_for_openmm(top_file, includeDir=gmx_include_dir)
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
                # Anchor indices may originate from OpenMM/NumPy arrays.  Cast
                # them here so the formatter's documented JSON output is
                # directly serializable even before the final write boundary.
                "receptor_indices": [int(i) for i in rec_idx],
                "ligand_indices": [int(i) for i in lig_idx],
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


_NONBONDED_FORCE_SCALAR_PROPERTIES = (
    # (getter, setter) —— 与电荷/λ 无关的 NonbondedForce 配置，烘焙时必须
    # 原样保留，否则烘焙本身就悄悄改了 Hamiltonian（2026-08-11 用户审阅指出）。
    ("getName", "setName"),
    ("getForceGroup", "setForceGroup"),
    ("getReciprocalSpaceForceGroup", "setReciprocalSpaceForceGroup"),
    ("getNonbondedMethod", "setNonbondedMethod"),
    ("getCutoffDistance", "setCutoffDistance"),
    ("getUseSwitchingFunction", "setUseSwitchingFunction"),
    ("getSwitchingDistance", "setSwitchingDistance"),
    ("getUseDispersionCorrection", "setUseDispersionCorrection"),
    ("getReactionFieldDielectric", "setReactionFieldDielectric"),
    ("getEwaldErrorTolerance", "setEwaldErrorTolerance"),
    ("getExceptionsUsePeriodicBoundaryConditions", "setExceptionsUsePeriodicBoundaryConditions"),
    ("getIncludeDirectSpace", "setIncludeDirectSpace"),
)
_NONBONDED_FORCE_TUPLE_PROPERTIES = (
    # 这两个 getter 返回一个元组（alpha, nx, ny, nz），要展开传给对应 setter。
    ("getPMEParameters", "setPMEParameters"),
    ("getLJPMEParameters", "setLJPMEParameters"),
)


def _copy_nonbonded_force_settings(
    src: openmm.NonbondedForce, dst: openmm.NonbondedForce
) -> List[str]:
    """把 `src` 上与 particle/exception 参数无关的所有配置复制到 `dst`。

    用 `getattr` 逐个探测而不是硬编码假设全部存在——不同 OpenMM 版本这组
    属性不完全一样；缺哪个就跳过并如实报告，不假装复制了。返回跳过的
    属性名列表（供调用方决定要不要打日志/报错）。
    """
    skipped: List[str] = []
    for getter_name, setter_name in _NONBONDED_FORCE_SCALAR_PROPERTIES:
        getter = getattr(src, getter_name, None)
        setter = getattr(dst, setter_name, None)
        if getter is None or setter is None:
            skipped.append(getter_name)
            continue
        setter(getter())
    for getter_name, setter_name in _NONBONDED_FORCE_TUPLE_PROPERTIES:
        getter = getattr(src, getter_name, None)
        setter = getattr(dst, setter_name, None)
        if getter is None or setter is None:
            skipped.append(getter_name)
            continue
        setter(*getter())
    if skipped:
        print(
            "  ⚠️ [bake_global_parameter_into_fixed_nonbonded_force] 当前 OpenMM "
            f"版本缺少以下 NonbondedForce 属性，未复制（版本差异，不是烘焙逻辑本身"
            f"的缺陷）：{skipped}"
        )
    return skipped


def _scan_forces_referencing_global_parameter(
    system: openmm.System, parameter_name: str
) -> List[Tuple[int, Any]]:
    """返回 System 里所有把 `parameter_name` 声明为自己 GlobalParameter 的
    `(force_index, force)`——不只看 NonbondedForce，任何 Custom*Force 都可能
    引用同名参数。
    """
    hits: List[Tuple[int, Any]] = []
    for idx in range(system.getNumForces()):
        force = system.getForce(idx)
        get_num = getattr(force, "getNumGlobalParameters", None)
        get_name = getattr(force, "getGlobalParameterName", None)
        if get_num is None or get_name is None:
            continue
        for p in range(get_num()):
            if get_name(p) == parameter_name:
                hits.append((idx, force))
                break
    return hits


def bake_global_parameter_into_fixed_nonbonded_force(
    system: openmm.System,
    parameter_name: str,
    lambda_value: float,
) -> openmm.System:
    """把某个 GlobalParameter 在给定端点上的取值烘焙成 `NonbondedForce` 的
    静态参数，并把这个 GlobalParameter 从 System 里彻底删除——不是靠"调用方
    记得把它设成这个值"这种纪律来保证安全，是让"忘了设"这条路径在结构上
    就不存在。

    背景（`STAGE2_CHARGE_TRANSFER_HANDOFF_PROPOSAL.md`）：C3-1 会话诊断
    发现的真实 bug——`ibs_engine.build_ibs_dual_system` 会把 charging 配置
    完成后 System 上的 `lam_coul`（默认值 **1.0**）连同它的
    ParticleParameterOffset/ExceptionParameterOffset 原样克隆过去；后续
    求值只要忘了显式 `context.setParameter("lam_coul", 0.0)`，OpenMM 就用
    默认值 1.0，把"配体 0 电荷、co-ion 满电"的 λ=0 端点悄悄翻成"配体满电、
    co-ion 中性"。这个函数就是那条缺失的、结构性的"安全交接"步骤。

    契约（2026-08-11 用户审阅后钉死）：

    1. 只删除 `parameter_name` 这一个 GlobalParameter；`NonbondedForce` 上
       其它 GlobalParameter、挂在其它参数名下的 offset 原样保留。
    2. 同一个粒子/exception 上如果有**多个**挂在 `parameter_name` 上的
       offset，fail closed——**不是**"先把 scale 累加再烘焙一次"。这一条
       在审阅时最初写反了：`parameter = base + Σ(global_i × scale_i)` 里的
       Σ 说的是"同一个粒子上挂了多个不同 GlobalParameter 各自的 offset 要
       相加"，不是"同一个 (parameter, particle) 重复挂多条也相加"。用真实
       Context 实测过（2026-08-11）：对同一个 (parameter, particle) 追加两条
       offset，`getNumParticleParameterOffsets()` 确实报出两条，但 Context
       求值时只认**最后一条**，不是两条的和。这是没有文档、容易被误用的
       OpenMM 行为；真实生产代码从不对同一个粒子重复调用
       `addParticleParameterOffset`，所以这里选择不去复现"取最后一条"这个
       隐藏规则，遇到真的重复就直接报错，不猜语义。
    3. 新建的 `NonbondedForce` 完整复制原 force 的非 particle/exception 配置
       （见 `_copy_nonbonded_force_settings`），当前 OpenMM 版本缺哪个属性
       就跳过并打印警告，不假装复制了。
    4. 若 `parameter_name` 还被 System 里其它（非目标）Force 引用，fail
       closed——本函数只烘焙 `NonbondedForce`，不能宣称整个 System 已经不再
       有这个活参数。
    5. charge、sigma、epsilon 三个分量分别用同一个通用公式
       `base + lambda_value * scale` 烘焙，不借用 `charge_at_lambda`
       （那是电荷专用命名，这里是通用工具，同一个公式对三个分量都适用）。
    6. `lambda_value` 必须是精确的 `0.0` 或 `1.0`——C3 的容差体系只在端点上
       有意义，非端点值拒绝。
    """
    lambda_value = float(lambda_value)
    if lambda_value not in (0.0, 1.0):
        raise ValueError(
            f"lambda_value={lambda_value} 不是精确的 0.0 或 1.0——烘焙只定义在端点上。"
        )

    system = ensure_owned_system(
        XmlSerializer.deserialize(XmlSerializer.serialize(system))
    )

    hits = _scan_forces_referencing_global_parameter(system, parameter_name)
    nb_hits = [(idx, f) for idx, f in hits if isinstance(f, openmm.NonbondedForce)]
    other_hits = [(idx, f) for idx, f in hits if not isinstance(f, openmm.NonbondedForce)]
    if not nb_hits:
        raise RuntimeError(
            f"{parameter_name!r} 不是这个 System 里任何 NonbondedForce 的 "
            "GlobalParameter，无法烘焙（可能是名字拼错了，或者这个参数根本不在"
            "这个 System 上）。"
        )
    if len(nb_hits) > 1:
        raise RuntimeError(
            f"{parameter_name!r} 出现在 {len(nb_hits)} 个 NonbondedForce 上——"
            "当前实现只支持单个 NonbondedForce，多个的语义未定义，拒绝继续。"
        )
    if other_hits:
        other_types = [type(f).__name__ for _idx, f in other_hits]
        raise RuntimeError(
            f"{parameter_name!r} 还被以下非 NonbondedForce 引用：{other_types}——"
            "本函数只烘焙 NonbondedForce 的 offset，其它 Force 里这个参数不会被"
            "处理。不能假装整个 System 已经不再有这个活参数；请先确认这些 Force "
            "是否也需要烘焙（当前未实现），或者改用别的参数名把它们隔离开。"
        )

    nb_index, nb = nb_hits[0]
    num_particles = nb.getNumParticles()

    def _value(x, target_unit):
        """SWIG 对"恰好是 0"的 offset scale 有时会返回裸 `float` 而不是
        `Quantity`（已实测：`addParticleParameterOffset(..., 0.0*nm, ...)` 传
        进去，读回来可能就是裸 `0.0`），返回值类型不稳定。这里统一转换成给定
        单位下的裸 float，后续全部用 float 做累加，只在最后写回 System 时
        才重新套上单位——不依赖 OpenMM 返回值本身的类型。
        """
        if unit.is_quantity(x):
            return x.value_in_unit(target_unit)
        return float(x)

    # ---- particle offsets：同一 (parameter, particle) 上重复出现就 fail closed ----
    #
    # 2026-08-11 用户审阅时给的契约原本要求"多条 offset 先求和"（引用 OpenMM
    # 文档里 `parameter = base + Σ(global_i × scale_i)` 的公式）。实测直接
    # 用 Context 验证发现：**OpenMM 对同一个 (parameter, particle) 上的多条
    # `ParticleParameterOffset` 并不求和**——`getNumParticleParameterOffsets()`
    # 确实报出两条，但 Context 求值时只认最后一条（0.3 与 0.2 两条追加，
    # 结果对应的是 0.2，不是 0.5；exception 同理，0.01+0.02 两条追加，结果
    # 对应 0.02，不是 0.03）。那条公式里的 Σ 说的是"同一个粒子上挂了多个
    # 不同 GlobalParameter 各自的 offset 要相加"，不是"同一个 (parameter,
    # particle) 重复挂多条"。真实生产代码（`configure_charge_transfer_
    # decharging` 等）从不对同一个粒子重复调用
    # `addParticleParameterOffset`——每个粒子只出现一次。所以这里选择**不**
    # 复现这个没有文档、容易被误用的"后者覆盖前者"行为，改成 fail closed：
    # 一旦真的出现重复，说明调用方的假设已经出了问题，直接报错比"悄悄按
    # OpenMM 的隐藏规则取最后一条"更安全。
    target_particle_scale: Dict[int, Tuple[Any, Any, Any]] = {}
    kept_particle_offsets: List[Tuple[str, int, Any, Any, Any]] = []
    for i in range(nb.getNumParticleParameterOffsets()):
        pname, particle, q_scale, sigma_scale, eps_scale = nb.getParticleParameterOffset(i)
        particle = int(particle)
        if pname != parameter_name:
            kept_particle_offsets.append((pname, particle, q_scale, sigma_scale, eps_scale))
            continue
        if particle in target_particle_scale:
            raise RuntimeError(
                f"粒子 {particle} 上有多条挂在 {parameter_name!r} 下的 "
                "ParticleParameterOffset——OpenMM 对这种重复不做加法（实测确认，"
                "只认最后一条），这个函数拒绝猜测应该按哪种语义处理，请先在源头"
                "去重。"
            )
        target_particle_scale[particle] = (
            _value(q_scale, unit.elementary_charge),
            _value(sigma_scale, unit.nanometer),
            _value(eps_scale, unit.kilojoule_per_mole),
        )

    # ---- exception offsets，同理：重复就 fail closed，不猜语义 ----
    target_exception_scale: Dict[int, Tuple[Any, Any, Any]] = {}
    kept_exception_offsets: List[Tuple[str, int, Any, Any, Any]] = []
    for i in range(nb.getNumExceptionParameterOffsets()):
        pname, exc_index, cp_scale, sigma_scale, eps_scale = nb.getExceptionParameterOffset(i)
        exc_index = int(exc_index)
        if pname != parameter_name:
            kept_exception_offsets.append((pname, exc_index, cp_scale, sigma_scale, eps_scale))
            continue
        if exc_index in target_exception_scale:
            raise RuntimeError(
                f"exception {exc_index} 上有多条挂在 {parameter_name!r} 下的 "
                "ExceptionParameterOffset——同上，OpenMM 对这种重复不做加法，"
                "拒绝猜测语义，请先在源头去重。"
            )
        target_exception_scale[exc_index] = (
            _value(cp_scale, unit.elementary_charge**2),
            _value(sigma_scale, unit.nanometer),
            _value(eps_scale, unit.kilojoule_per_mole),
        )

    # ---- 建一个干净的新 NonbondedForce，完整复制配置 ----
    new_nb = openmm.NonbondedForce()
    _copy_nonbonded_force_settings(nb, new_nb)

    # ---- 逐粒子烘焙：没有目标 offset 的粒子原样复制 ----
    for idx in range(num_particles):
        q, sigma, epsilon = nb.getParticleParameters(idx)
        if idx in target_particle_scale:
            q_scale, sigma_scale, eps_scale = target_particle_scale[idx]
            q = q + (lambda_value * q_scale) * unit.elementary_charge
            sigma = sigma + (lambda_value * sigma_scale) * unit.nanometer
            epsilon = epsilon + (lambda_value * eps_scale) * unit.kilojoule_per_mole
        new_nb.addParticle(q, sigma, epsilon)

    # ---- 逐 exception 烘焙 ----
    for idx in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sigma, epsilon = nb.getExceptionParameters(idx)
        if idx in target_exception_scale:
            cp_scale, sigma_scale, eps_scale = target_exception_scale[idx]
            charge_prod = charge_prod + (lambda_value * cp_scale) * unit.elementary_charge**2
            sigma = sigma + (lambda_value * sigma_scale) * unit.nanometer
            epsilon = epsilon + (lambda_value * eps_scale) * unit.kilojoule_per_mole
        new_nb.addException(int(p1), int(p2), charge_prod, sigma, epsilon)

    # ---- 重新声明其它 GlobalParameter，重新挂回其它 offset（原样保留） ----
    # kept_*_offsets 里的 scale 值同样可能是裸 float（同一个 SWIG 问题），
    # 重新写回前用 _value 统一转成明确单位的 Quantity，不能假设它们已经是
    # Quantity——那正是刚才炸掉的那个假设。
    for gi in range(nb.getNumGlobalParameters()):
        gname = nb.getGlobalParameterName(gi)
        if gname == parameter_name:
            continue
        new_nb.addGlobalParameter(gname, nb.getGlobalParameterDefaultValue(gi))
    for pname, particle, q_scale, sigma_scale, eps_scale in kept_particle_offsets:
        new_nb.addParticleParameterOffset(
            pname, particle,
            _value(q_scale, unit.elementary_charge) * unit.elementary_charge,
            _value(sigma_scale, unit.nanometer) * unit.nanometer,
            _value(eps_scale, unit.kilojoule_per_mole) * unit.kilojoule_per_mole,
        )
    for pname, exc_index, cp_scale, sigma_scale, eps_scale in kept_exception_offsets:
        new_nb.addExceptionParameterOffset(
            pname, exc_index,
            _value(cp_scale, unit.elementary_charge**2) * unit.elementary_charge**2,
            _value(sigma_scale, unit.nanometer) * unit.nanometer,
            _value(eps_scale, unit.kilojoule_per_mole) * unit.kilojoule_per_mole,
        )

    system.removeForce(nb_index)
    system.addForce(new_nb)

    remaining = _scan_forces_referencing_global_parameter(system, parameter_name)
    if remaining:
        raise RuntimeError(
            f"内部错误：烘焙后 {parameter_name!r} 仍出现在 "
            f"{[type(f).__name__ for _idx, f in remaining]} 上——烘焙没有做干净。"
        )
    return system


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
    # [MEM-00h，2026-08-06] 统一到基础力场的 1.0 nm、无 switching——理由见
    # BEUTLER_SOFTCORE_CUTOFF_NM 定义处；这个力此前用的是跟 softcore CV 一样的
    # 1.2nm+switch，跟基础 NonbondedForce 的 1.0nm 不一致，是同一个 MEM-00h
    # 存量问题的一部分。
    ll_force.setCutoffDistance(BEUTLER_SOFTCORE_CUTOFF_NM * unit.nanometer)
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

    ll_force.setUseSwitchingFunction(False)
    ll_force.setSwitchingDistance(BEUTLER_SOFTCORE_CUTOFF_NM * unit.nanometer)
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
