#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ABFE 核心物理模块 (v6.0 - 完整收敛版)
职责：统一封装所有势能、限制力、拟合器、估算器、校验器、路径规划、替身构建、Orb扫描与路由工厂
架构约束：严格收敛至 5 文件，本文件为唯一物理核心单例，零占位符
依赖：openmm, numpy, scipy, mdtraj, torch, openmmml (部分功能)
"""

import os
import sys
import contextlib
import functools
import tempfile

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

# ============================================================================
# [MBAR-JAX-HOST-MEM] 同样必须在**任何** pymbar import 之前执行。
#
# 上面那行只关掉了 JAX 的**显存**预分配。JAX 后端本身对这个仓库的问题规模是
# 纯负担：每个 MBAR 解都要付一次 XLA 编译（λ 表/帧数一变形状就变，编译缓存
# 命不中），编译产物和 BFC 池又**从不归还**。
#
# 实测（2026-09-09，本机 openmm_dev，K=12 态 × N=9600 帧，
# `relative_tolerance=1e-7, initialize="BAR", solver_protocol="default"`，
# 即生产 11 个构造点用的同一组参数）：
#
#            wall     RSS       mmap 数    f_k
#   JAX      6.30 s   887 MiB   1441       —
#   numpy    0.37 s   120 MiB    891       max|Δf_k| = 8.9e-16
#
# 也就是 **慢 17 倍、胖 7 倍，答案在 double 舍入内逐位相同**。而且不可逆：
# 一次 import + 一次解就把 RSS 抬到 ~860 MiB、VSZ 抬到 18.4 GiB，之后每解一次
# 再涨 ~14 MiB，解完一个都不还（numpy 后端 12 次解 RSS 只从 100 涨到 109 MiB，
# mmap 数 835→836，完全平的）。
#
# 后果就是宿主内存被解算器啃光，然后由**下游任何一次大分配**替它抛错——
# 2026-09-09 cyclod_ligand1/rep1 是 Stage 1 第一个 replica 的
# `XmlSerializer.deserializeSystem` 抛 `std::bad_alloc`（7.4 MB 的 XML、93 GB
# 的机器、`ulimit -v unlimited`），离线复现那个崩点完全正常；同一天离线 MBAR
# 的 16 个 spawn worker 则表现成三个互相矛盾的 `Unknown property` 解析错
# （'sig' in 'Particle' / 'p3' in 'Angle' / 'eps' in 'Exception'）加
# `double free or corruption` 加 `std::bad_alloc`——同一份 XML 在不同 worker
# 里报不同的错，就是堆已经不可信的形状，不是 XML 的内容问题。
#
# 用 `setdefault`：想要回 JAX 的人显式导出 `PYMBAR_DISABLE_JAX=0` 即可
# （pymbar 只认 "true"/"yes"/"1" 为真）。
# ============================================================================
os.environ.setdefault("PYMBAR_DISABLE_JAX", "1")

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
# [CLI-01, 2026-09-02] `torch` / `openmmml` / `pymbar` 以前在这里 eager import，
# 于是**任何** import abfe_core 的入口都要付它们的代价：实测 `runabfe.py --help`
# 4.3 s 里 torch 1.31 s + pymbar 1.12 s（`python -X importtime`），而 --help、
# self-test、配置诊断这些路径一个都用不到它们。
#
# 改成惰性 memoized 探测，语义与旧的 try/except 完全一致：
#   * `has_orb()` / `has_pymbar()` 仍然只回答"装没装"，不抛异常；
#   * `HAS_ORB` / `HAS_PYMBAR` 作为模块属性继续可读（见文件末尾的
#     `__getattr__`），外部 `abfe_core.HAS_PYMBAR` 和测试里的 monkeypatch
#     都不受影响；
#   * 真正要用模块对象的地方走 `_require_torch()` / `_require_pymbar()`。
# 这是 [ATT-04] 那次"torch 不得有 import 期副作用"的同一条思路的延伸——那次
# 消掉的是 import 期 `torch.cuda` 调用，这次消掉 import 本身。
_TORCH_MODULE = None
_MLPOTENTIAL_CLASS = None
_ORB_PROBED = False
_PYMBAR_MODULE = None
_PYMBAR_PROBED = False


def _probe_orb():
    """首次调用时才 import torch/openmmml；返回 (torch, MLPotential) 或 (None, None)。"""
    global _TORCH_MODULE, _MLPOTENTIAL_CLASS, _ORB_PROBED
    if not _ORB_PROBED:
        try:
            import torch as _torch
            from openmmml import MLPotential as _MLPotential

            _TORCH_MODULE, _MLPOTENTIAL_CLASS = _torch, _MLPotential
        except ImportError:
            _TORCH_MODULE, _MLPOTENTIAL_CLASS = None, None
        _ORB_PROBED = True
    return _TORCH_MODULE, _MLPOTENTIAL_CLASS


def has_orb() -> bool:
    """torch + openmmml 是否可用。`HAS_ORB` 的模块级覆盖（monkeypatch）优先。"""
    override = globals().get("HAS_ORB")
    if override is not None:
        return bool(override)
    return _probe_orb()[0] is not None


def _require_torch():
    torch_module, _ = _probe_orb()
    if torch_module is None:
        raise ImportError("此路径需要 torch + openmmml（Orb/MACE ML 势），当前环境未安装")
    return torch_module


def _require_mlpotential():
    _, mlpotential = _probe_orb()
    if mlpotential is None:
        raise ImportError("此路径需要 openmmml.MLPotential，当前环境未安装 openmm-ml")
    return mlpotential


def _probe_pymbar():
    """首次调用时才 import pymbar（连带 JAX）；返回模块或 None。"""
    global _PYMBAR_MODULE, _PYMBAR_PROBED
    if not _PYMBAR_PROBED:
        try:
            import pymbar as _pymbar

            _PYMBAR_MODULE = _pymbar
        except ImportError:
            _PYMBAR_MODULE = None
        _PYMBAR_PROBED = True
    return _PYMBAR_MODULE


def has_pymbar() -> bool:
    """pymbar 是否可用。`HAS_PYMBAR` 的模块级覆盖（monkeypatch）优先。"""
    override = globals().get("HAS_PYMBAR")
    if override is not None:
        return bool(override)
    return _probe_pymbar() is not None


def _require_pymbar():
    pymbar_module = _probe_pymbar()
    if pymbar_module is None:
        raise ImportError("此路径需要 pymbar（MBAR/TMBAR 估计器），当前环境未安装")
    return pymbar_module

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
