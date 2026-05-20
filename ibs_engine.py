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
from typing import Dict, List, Tuple, Optional
from abfe_core import (
    ACESoftcorePotential,
    AlchemicalPotentialFactory,
    LambdaDependentBoreschForce,
    create_ligand_internal_force,
    ensure_owned_system,
    sync_all_exclusions,
    BeutlerSoftcoreBuilder,
)

try:
    import pymbar
    HAS_PYMBAR = True
except ImportError:
    HAS_PYMBAR = False


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
    lambda_name: str = "lam_coul"
) -> Tuple[Dict, Optional[int]]:
    """
    🔥 终极防御：共炼金反离子策略
    寻找体系中最远的反离子，使其与配体同步消电，保持全局严格电中性。
    """
    nb_force = next((f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
    if nb_force is None:
        raise RuntimeError("系统中未找到 NonbondedForce")

    # 1. 计算配体净电荷
    lig_net_charge = 0.0
    for idx in ligand_indices:
        q, _, _ = nb_force.getParticleParameters(idx)
        lig_net_charge += q.value_in_unit(unit.elementary_charge)
    lig_net_charge = round(lig_net_charge)

    if abs(lig_net_charge) < 0.01:
        print("  ℹ️ 配体为电中性，无需共消电反离子。使用标准 PME Offset。")
        target_ion_charge = 0.0
    else:
        print(f"  ⚡ 检测到带电配体 (Net Charge: {lig_net_charge:+d})，启动共炼金反离子搜索...")
        target_ion_charge = -1.0 if lig_net_charge > 0 else 1.0

    # 2. 寻找距离“蛋白+配体”质心最远的匹配离子
    best_ion_idx = None
    if target_ion_charge != 0.0 and positions is not None:
        pos_nm = np.array([p.value_in_unit(unit.nanometer) if hasattr(p, 'value_in_unit') else p for p in positions])
        heavy_solute_indices = [
            a.index for a in topology.atoms() 
            if a.residue.name not in ['HOH', 'WAT', 'SOL', 'CL', 'CLA', 'NA', 'SOD', 'K', 'POT', 'MG'] 
            and a.element.symbol != 'H'
        ]
        if heavy_solute_indices:
            solute_com = np.mean(pos_nm[heavy_solute_indices], axis=0)
            max_dist = -1.0
            for atom in topology.atoms():
                if atom.residue.name.upper() in ['CL', 'CLA', 'NA', 'SOD', 'K', 'POT', 'MG']:
                    idx = atom.index
                    q, _, _ = nb_force.getParticleParameters(idx)
                    if abs(q.value_in_unit(unit.elementary_charge) - target_ion_charge) < 0.1:
                        dist = np.linalg.norm(pos_nm[idx] - solute_com)
                        if dist > max_dist:
                            max_dist = dist
                            best_ion_idx = idx
            if best_ion_idx is not None:
                print(f"  🎯 锁定共消电反离子: Index {best_ion_idx}, 距离溶质核心 {max_dist:.2f} nm")

    # 3. 注入全局 Lambda 与 ParameterOffset
    existing_params = [nb_force.getGlobalParameterName(i) for i in range(nb_force.getNumGlobalParameters())]
    if lambda_name not in existing_params:
        nb_force.addGlobalParameter(lambda_name, 1.0)

    original_charges = {}
    
    # 3.1 配体 Offset
    for idx in ligand_indices:
        q, sig, eps = nb_force.getParticleParameters(idx)
        original_charges[idx] = q
        nb_force.setParticleParameters(idx, 0.0*unit.elementary_charge, sig, eps)
        nb_force.addParticleParameterOffset(lambda_name, idx, q, 0.0*unit.nanometer, 0.0*unit.kilojoule_per_mole)

    # 3.2 反离子 Offset
    if best_ion_idx is not None:
        ion_q, ion_sig, ion_eps = nb_force.getParticleParameters(best_ion_idx)
        original_charges[best_ion_idx] = ion_q
        nb_force.setParticleParameters(best_ion_idx, 0.0*unit.elementary_charge, ion_sig, ion_eps)
        nb_force.addParticleParameterOffset(lambda_name, best_ion_idx, ion_q, 0.0*unit.nanometer, 0.0*unit.kilojoule_per_mole)

    # 3.3 屏蔽主系统中的 L-L 静电 (防止与 Group 2 双重计数)
    ll_pairs = set((i, j) for i in ligand_indices for j in ligand_indices if i < j)
    for i in range(nb_force.getNumExceptions()):
        p1, p2, cp, sig, eps = nb_force.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if (min(p1, p2), max(p1, p2)) in ll_pairs:
            nb_force.setExceptionParameters(i, p1, p2, 0.0*unit.elementary_charge**2, sig, eps)
            ll_pairs.remove((min(p1, p2), max(p1, p2)))
    for p1, p2 in ll_pairs:
        nb_force.addException(p1, p2, 0.0*unit.elementary_charge**2, 0.1*unit.nanometer, 0.0*unit.kilojoule_per_mole)

    # 3.4 处理 1-4 静电 Offset
    for i in range(nb_force.getNumExceptions()):
        p1, p2, cp, sig, eps = nb_force.getExceptionParameters(i)
        if p1 in ligand_indices or p2 in ligand_indices:
            q1 = original_charges.get(p1, nb_force.getParticleParameters(p1)[0])
            q2 = original_charges.get(p2, nb_force.getParticleParameters(p2)[0])
            nominal_cp = q1 * q2
            nb_force.setExceptionParameters(i, p1, p2, 0.0*unit.elementary_charge**2, sig, eps)
            nb_force.addExceptionParameterOffset(lambda_name, i, nominal_cp, 0.0*unit.nanometer, 0.0*unit.kilojoule_per_mole)

    print("  ✅ 共炼金反离子防御阵列部署完毕。PME 倒空间计算全程严格电中性！")
    return original_charges, best_ion_idx

def _create_pure_vdw_softcore_force(
    nb_force: openmm.NonbondedForce,
    perturbed_indices: List[int],
    environment_indices: List[int],
    lam_vdw: float,
    softcore_params: ACESoftcorePotential,
    reference_exclusions=None,
    particle_params_override=None,
    num_particles=None,
    use_global_lambda=False
) -> openmm.CustomNonbondedForce:
    """纯 vdW 软核力 (彻底剔除静电项，交由主系统 PME 处理)"""
    total_particles = nb_force.getNumParticles()
    perturbed_set, env_set = set(perturbed_indices), set(environment_indices)
    
    lam_v_str = "lam_vdw" if use_global_lambda else f"{lam_vdw:.6f}"
    safe_lam_v = f"max(1.0 - {lam_v_str}, 0.01)" if not use_global_lambda else "max(1.0 - lam_vdw, 0.01)"
    
    alpha_lj = softcore_params.alpha_lj
    m_lj = softcore_params.m_lj
    expr = (
        f"{lam_v_str} * 4 * sqrt(epsilon1*epsilon2) * ("
        f"(sigma12^12 / (r^6 + {alpha_lj}*(1.0-{lam_v_str}+1e-9)^{m_lj} + 1e-6)^2) - "
        f"(sigma12^6 / (r^6 + {alpha_lj}*(1.0-{lam_v_str}+1e-9)^{m_lj} + 1e-6))"
        f");"
        f"sigma12=0.5*(sigma1+sigma2)"
    )
    import re
    expr = re.sub(r'\(1\.0\s*-\s*' + re.escape(lam_v_str) + r'(?:\s*\+\s*1e-9)?\)', safe_lam_v, expr)
    expr = re.sub(r'\(1\s*-\s*' + re.escape(lam_v_str) + r'(?:\s*\+\s*1e-9)?\)', safe_lam_v, expr)

    sc_force = openmm.CustomNonbondedForce(expr)
    sc_force.addPerParticleParameter("sigma")
    sc_force.addPerParticleParameter("epsilon")

    if use_global_lambda:
        sc_force.addGlobalParameter("lam_vdw", lam_vdw)

    for i in range(total_particles):
        if particle_params_override and i < len(particle_params_override):
            _, sig, eps = particle_params_override[i]
        else:
            _, sig, eps = nb_force.getParticleParameters(i)
        sc_force.addParticle([
            sig.value_in_unit(openmm.unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole)
        ])

    sc_force.addInteractionGroup(list(perturbed_set), list(env_set))
    sc_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    sc_force.setCutoffDistance(1.2 * openmm.unit.nanometer)
    
    if reference_exclusions:
        for p1, p2 in reference_exclusions:
            p1, p2 = int(p1), int(p2)
            if p1 < total_particles and p2 < total_particles:
                sc_force.addExclusion(p1, p2)
                
    sc_force.setUseSwitchingFunction(True)
    sc_force.setSwitchingDistance(1.0 * openmm.unit.nanometer)
    return sc_force
    
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
    if n_states <= pts_per_window:
        return [(0, n_states)]

    if n_windows is None:
        n_windows = max(2, math.ceil((n_states - overlap) / (pts_per_window - overlap)))
    if n_windows > 1:
        step = math.ceil((n_states - pts_per_window) / (n_windows - 1))
        step = max(overlap + 1, step)
    else:
        step = n_states

    windows = []
    for i in range(n_windows):
        start = i * step
        if start >= n_states:
            break
        end = min(start + pts_per_window, n_states)
        windows.append((start, end))

    if windows and windows[-1][1] < n_states:
        windows[-1] = (max(0, n_states - pts_per_window), n_states)

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
def _create_softcore_force(
    nb_force: openmm.NonbondedForce,
    perturbed_indices: List[int],
    environment_indices: List[int],
    lam_coul: float,
    lam_vdw: float,
    softcore_params: ACESoftcorePotential,
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
    
    # 软核分母安全保护 (1-λ) 项，防止 λ→1 时 α*(1-λ)^m 归零导致 r→0 奇点
    safe_1mlam_v = max(1.0 - lam_vdw, 0.01)
    safe_1mlam_v_str = f"{safe_1mlam_v:.8f}"
    
    # 调用工厂生成完整软核表达式 (含 Coulomb + VdW)
    expr, _ = AlchemicalPotentialFactory.build("softcore", softcore_params, lam_c_str, lam_v_str)
    
    # 🔧 替换表达式中的 (1-λ) 软核项为安全常数 (OpenMM 编译期会自动折叠常数运算)
    import re
    expr = re.sub(r'\(1\.0\s*-\s*' + re.escape(lam_v_str) + r'(?:\s*\+\s*1e-9)?\)', safe_1mlam_v_str, expr)
    expr = re.sub(r'\(1\s*-\s*' + re.escape(lam_v_str) + r'(?:\s*\+\s*1e-9)?\)', safe_1mlam_v_str, expr)
    
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
    sc_force.setCutoffDistance(1.2 * unit.nanometer)
    sc_force.setUseLongRangeCorrection(False)
    
    if reference_exclusions:
        for p1, p2 in reference_exclusions:
            p1, p2 = int(p1), int(p2)
            if p1 < total_particles and p2 < total_particles:
                sc_force.addExclusion(p1, p2)
                
    sc_force.setUseSwitchingFunction(True)
    sc_force.setSwitchingDistance(1.0 * unit.nanometer)
    return sc_force

def _normalize_softcore_params(softcore_params: ACESoftcorePotential, n_perturbed: int) -> ACESoftcorePotential:
    _ = n_perturbed
    requested_lj = float(getattr(softcore_params, "alpha_lj", float("nan")))
    requested_coul = float(getattr(softcore_params, "alpha_coul", float("nan")))
    normalized = ACESoftcorePotential(alpha_lj=0.7, alpha_coul=0.5, power_lj=(2, 2), power_coul=(1, 1))
    print(
        "  🔒 [Softcore 参数] 生产模式固定 alpha_lj=0.700 nm^6, alpha_coul=0.500 nm^2 "
        f"(输入值: LJ={requested_lj:.3f}, Coul={requested_coul:.3f})"
    )
    return normalized


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


def diagnose_softcore_cv_values(
    context: openmm.Context,
    ibs_wrapper: "IBSBiasForce",
    lambdas_coul: List[float],
    lambdas_vdw: List[float],
    prefix: str = "",
):
    """打印当前窗口所有软核 CV / Boresch CV 的原始数值，帮助定位异常背景。"""
    print(f"\n🔍 [{prefix}] 软核 CV 数值诊断:")
    try:
        state_base = context.getState(getEnergy=True, groups={0, 2, 3})
        e_base = state_base.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    except Exception as e:
        e_base = float("nan")
        print(f"  e_base 读取失败: {e}")

    try:
        state_bias = context.getState(getEnergy=True, groups={1})
        e_bias = state_bias.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    except Exception:
        e_bias = float("nan")

    print(f"  e_base(Group2+3)={e_base:.3f} kJ/mol | e_bias(Group1)={e_bias:.3f} kJ/mol")

    try:
        cv_vals = ibs_wrapper.get_force().getCollectiveVariableValues(context)
    except Exception as e:
        print(f"  CV 读取失败: {e}")
        return

    for k, (lc, lv) in enumerate(zip(lambdas_coul, lambdas_vdw)):
        idx_int = 2 * k
        idx_rest = 2 * k + 1
        e_int = cv_vals[idx_int] if idx_int < len(cv_vals) else float("nan")
        e_rest = cv_vals[idx_rest] if idx_rest < len(cv_vals) else float("nan")
        e_tot = e_base + e_int + e_rest if np.isfinite(e_base) else float("nan")
        print(
            f"  state {k:>2d} | lam_c={lc:7.4f} lam_v={lv:7.4f} | "
            f"e_int={e_int:14.3f} | e_rest={e_rest:10.3f} | e_total={e_tot:14.3f}"
        )

def build_ibs_dual_system(
    system: openmm.System,
    topology,
    perturbed_indices: List[int],
    lambdas_coul: List[float],
    lambdas_vdw: List[float],
    softcore_params: ACESoftcorePotential,
    restraint_params: Optional[Dict] = None,
    temperature: openmm.unit.Quantity = 300 * openmm.unit.kelvin,
    prefix: str = "abfe_dual",
    box_vectors=None,
    reference_positions=None,
) -> Tuple[openmm.System, 'IBSBiasForce']:
    """
    构建双λ IBS 采样系统 (终极修复版：彻底剥离 Group 0 的 λ 依赖)
    """
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
    softcore_params = _normalize_softcore_params(softcore_params, len(perturbed_indices))

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
        print(f"  ⚡ 检测到带电配体 (Net Charge: {lig_net_charge:+d})，启动静态反离子中和...")
        target_ion_charge = -1.0 if lig_net_charge > 0 else 1.0
        pos_nm = np.array([p.value_in_unit(unit.nanometer) if hasattr(p, 'value_in_unit') else p for p in reference_positions]) if reference_positions is not None else None
        
        best_ion_idx = None
        if pos_nm is not None:
            heavy_solute_indices = [a.index for a in topology.atoms() if a.residue.name not in ['HOH', 'WAT', 'SOL', 'CL', 'NA'] and a.element.symbol != 'H']
            solute_com = np.mean(pos_nm[heavy_solute_indices], axis=0) if heavy_solute_indices else np.mean(pos_nm, axis=0)
            max_dist = -1.0
            for atom in topology.atoms():
                if atom.residue.name.upper() in ['CL', 'CLA', 'NA', 'SOD', 'K', 'POT', 'MG']:
                    idx = atom.index
                    q_ion, _, _ = all_params[idx]
                    if abs(q_ion.value_in_unit(unit.elementary_charge) - target_ion_charge) < 0.1:
                        dist = np.linalg.norm(pos_nm[idx] - solute_com)
                        if dist > max_dist:
                            max_dist = dist
                            best_ion_idx = idx
        
        if best_ion_idx is not None:
            ion_q, ion_sig, ion_eps = all_params[best_ion_idx]
            new_ion_q = (ion_q.value_in_unit(unit.elementary_charge) - lig_net_charge) * unit.elementary_charge
            # 直接修改主 NB 中的反离子电荷 (静态，不依赖 λ)
            nb.setParticleParameters(best_ion_idx, new_ion_q, ion_sig, ion_eps)
            print(f"  🎯 静态中和: 反离子 {best_ion_idx} 电荷调整为 {new_ion_q.value_in_unit(unit.elementary_charge):.2f} e (PME 严格中性)")
        else:
            print(f"  ⚠️ 未找到合适反离子，OpenMM PME 将使用均匀背景电荷中和系统。")

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
    if restraint_params and "receptor_indices" in restraint_params:
        rest_f_phys = LambdaDependentBoreschForce(
            rec_idx=restraint_params["receptor_indices"], lig_idx=restraint_params["ligand_indices"],
            eq=restraint_params["equilibrium_values"], fc=restraint_params["force_constants"],
            lam_name="lambda_boresch_scale", fixed_lam=None, sign=1.0
        )
        rest_f_phys.setForceGroup(3)
        new_sys.addForce(rest_f_phys)

    # ---------- 6. λ-WCA 防护壳（Group 4） & COM 限制（Group 5） [保持原样] ----------
    wca_expr = (
        "4.0*lambda_shield*(1.0-lambda_shield)*step(rc-r)*eps_wca*"
        "(((rc/max(r, 1e-6))^6)^2 - 2*((rc/max(r, 1e-6))^6) + 1)"
    )
    wca_force = openmm.CustomNonbondedForce(wca_expr)
    wca_force.addGlobalParameter("lambda_shield", 0.0)
    wca_force.addGlobalParameter("rc", 0.22)
    wca_force.addGlobalParameter("eps_wca", 1.0)
    for _ in range(num_atoms): wca_force.addParticle([])
    wca_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    wca_force.setCutoffDistance(0.22 * openmm.unit.nanometer)
    wca_force.addInteractionGroup(list(perturbed_set), env_indices)
    for p1, p2 in softcore_excl: wca_force.addExclusion(int(p1), int(p2))
    wca_force.setForceGroup(4)
    new_sys.addForce(wca_force)

    if reference_positions is not None and perturbed_indices:
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
            pass # 降级处理省略

    # ---------- 7. IBS 偏置力与完整软核 CV (Group 1) 🔑 核心修复 ----------
    ibs_wrapper = IBSBiasForce(len(lambdas_coul), temperature, prefix=prefix)
    original_params_fresh = [original_nb.getParticleParameters(i) for i in range(num_atoms)]
    
    for k, (lc, lv) in enumerate(zip(lambdas_coul, lambdas_vdw)):
        # 🔑 修复：直接传入浮点数值，硬编码至 CV 表达式，彻底解除全局参数绑定
        # CV 现在包含了完整的配体-环境 静电 + VdW 软核相互作用
        int_f_cv = _create_softcore_force(
            nb, perturbed_indices, env_indices,
            lam_coul=float(lc), lam_vdw=float(lv),
            softcore_params=softcore_params,
            reference_exclusions=softcore_excl,
            particle_params_override=original_params_fresh,
            num_particles=num_atoms
        )
        ibs_wrapper.addCollectiveVariable(f"cv_{k}_int", int_f_cv)
        
        # Boresch CV 保持零力（物理限制已在 Group 3）
        rest_f_cv = openmm.CustomExternalForce("0")
        ibs_wrapper.addCollectiveVariable(f"cv_{k}_rest", rest_f_cv)
        
    new_sys.addForce(ibs_wrapper.get_force())

    # ---------- 8. 统一排除表同步 ----------
    sync_all_exclusions(new_sys)
    new_sys.thisown = 1
    _ = new_sys.getNumParticles()
    print(f"[IBS System] 构建完成，共 {new_sys.getNumForces()} 个力对象 (Group 0 已彻底剥离 λ 依赖)")
    return new_sys, ibs_wrapper

# ============================================================================
# 2. IBS 偏置力与采样器 (修复版)
# ============================================================================

class IBSBiasForce:
    """IBS 偏置力封装 (Group 1) - 数值稳定差分形式"""
    def __init__(self, n_states: int, temperature: openmm.unit.Quantity, prefix: str = "abfe"):
        self.n_states = n_states
        self.prefix = prefix
        self._cv_keeper = []
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
        # 我们采用标准的 Log-Sum-Exp 技巧，但强制以 k=0 为基准进行平移。
        
        terms = []
        # k=0 的项作为基准，显式写出 exp(0) = 1
        # 对于 k>0，计算相对指数
        for k in range(1, n_states):
            # 差分项: (cv_k_int + cv_k_rest - f_k) - (cv_0_int + cv_0_rest - f_0)
            diff_expr = f"(cv_{k}_int + cv_{k}_rest - {prefix}_f_{k}) - (cv_0_int + cv_0_rest - {prefix}_f_0)"
            # 限制指数参数范围 [-50, 50] 防止 exp 溢出/下溢 (比之前的 500 更严格，更安全)
            safe_diff = f"max(-50.0, min(50.0, -beta * ({diff_expr})))"
            terms.append(f"exp({safe_diff})")
        
        # 总和: 1 + sum(exp(...))
        sum_term = "1.0 + " + " + ".join(terms) if terms else "1.0"
        
        # 最终能量表达式:
        # V_bias = (cv_0_int + cv_0_rest - f_0) - kt * log(sum_term)
        # 注意：第一项 (cv_0...) 是坐标依赖的，必须包含在内以保证力的正确性。
        energy_expr = f"{prefix}_bias_scale * ((cv_0_int + cv_0_rest - {prefix}_f_0) - kt * log(max(1e-15, {sum_term})))"
        
        self.force = openmm.CustomCVForce(energy_expr)
        self.force.addGlobalParameter("kt", kt)
        self.force.addGlobalParameter("beta", beta)
        self.force.addGlobalParameter(f"{prefix}_bias_scale", 1.0)
        for k in range(n_states):
            self.force.addGlobalParameter(f"{prefix}_f_{k}", 0.0)
        self.force.setForceGroup(1)
    def addCollectiveVariable(self, name: str, cv_force: openmm.Force) -> int:
        self._cv_keeper.append(cv_force)
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


class IBSSampler:
    """
    IBS 采样器：严格通过 CustomCVForce 探针收集各态微扰能。
    【修复】采用 EMA (指数移动平均) 实现时间平均估计，对齐论文 TA 精神。
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
        
        # 🔑 新增：EMA 累积统计量
        self.ema_mean_p = None
        self.gamma = 0.9  # 衰减因子，保留 90% 历史信息
        self.energy_buffer = []
        self.energy_history = []
        self.f_history = []
        self.bias_history = []
        self.base_energy_history = []
        
        # 🔑 新增：能量偏移缓存
        self.e_offset = 0.0

    def collect_energies(self) -> np.ndarray:
        energies = np.zeros(self.n_states)
        # 1. Base 能量 (Group 0, 2, 3, 4, 5)：现已严格 λ 无关
        try:
            state_base = self.context.getState(getEnergy=True, groups={0, 2, 3, 4, 5})
            e_base = state_base.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        except Exception:
            e_base = 0.0
        self.base_energy_history.append(float(e_base))
        
        if self.ibs_wrapper is None:
            return np.full(self.n_states, np.nan)
            
        try:
            cv_vals = self.ibs_wrapper.get_force().getCollectiveVariableValues(self.context)
            interaction_energies = np.zeros(self.n_states)
            for k in range(self.n_states):
                idx_int = 2 * k
                e_int = cv_vals[idx_int] if idx_int < len(cv_vals) else 0.0
                interaction_energies[k] = e_int
                
            # 相对偏移防溢出 (以 State 0 为参考)
            if self.n_states > 0 and np.isfinite(interaction_energies[0]):
                self.e_offset = interaction_energies[0]
            energies = interaction_energies - self.e_offset
            
            if not np.any(np.isnan(energies)):
                self.energy_buffer.append(energies)
                self.energy_history.append(interaction_energies.copy())
        except Exception as e:
            print(f"  ⚠️ CV 探针能量提取失败: {e}")
            energies[:] = np.nan
        return energies

    def update_weights(self, min_buffer_size: int = 10) -> Optional[np.ndarray]:
        """
        在线 IBS 权重更新 (EMA Time-Averaged Estimator + Log-Gradient SGD)
        严格对齐论文 Eq. 13 & Sec 2.3
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
        
        # 4. EMA 更新
        if self.ema_mean_p is None:
            self.ema_mean_p = mean_p_batch.copy()
        else:
            self.ema_mean_p = self.gamma * self.ema_mean_p + (1.0 - self.gamma) * mean_p_batch
            
        # 5. 对数梯度更新
        target_p = 1.0 / K
        log_grad = np.log(self.ema_mean_p + 1e-30) - np.log(target_p)
        log_grad = np.clip(log_grad, -2.0, 2.0)
        
        t = len(self.f_history) + 1
        eta_sgd = 1.0 / (1.0 + t / 100.0)
        
        f_new = f_old - eta_sgd * self.kt * log_grad
        
        # 约束防止发散
        f_new = np.clip(f_new, -10000, 10000)
        
        # 6. 应用更新
        for k in range(K):
            self.context.setParameter(f"{self.prefix}_f_{k}", float(f_new[k]))
            
        self.energy_buffer = []
        self.f_history.append(f_new.copy())
        return f_new
# ================= ibs_engine.py -> IBSSampler 类 =================
    def save_ibs_state(self, filepath: str):
        """序列化 IBS 核心状态 (f_k 权重与 SGD 步数)"""
        import json
        f_current = [self.context.getParameter(f"{self.prefix}_f_{k}") for k in range(self.n_states)]
        state = {
            "f_k": f_current,
            "t": len(self.f_history),
            "e_offset": self.e_offset
        }
        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

    def load_ibs_state(self, filepath: str) -> bool:
        """反序列化并注入 IBS 状态，恢复历史记忆"""
        import json, os
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, "r") as f:
                state = json.load(f)
            f_k = state["f_k"]
            t = state["t"]
            self.e_offset = state.get("e_offset", 0.0)
            
            # 注入 Context
            for k in range(self.n_states):
                self.context.setParameter(f"{self.prefix}_f_{k}", float(f_k[k]))
            
            # 伪造历史长度以恢复 SGD 学习率衰减 (eta_sgd)
            self.f_history = [np.array(f_k)] * t 
            print(f"  ♻️ IBS 状态已恢复: t={t}, max|f_k|={np.max(np.abs(f_k)):.2f} kJ/mol")
            return True
        except Exception as e:
            print(f"  ⚠️ IBS 状态加载失败: {e}")
            return False
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
        softcore_params: ACESoftcorePotential,
        restraint_params: Optional[Dict] = None,
        prefix: str = "abfe_dual",
        platform_name: str = "CUDA",
        output_dir: str = "./output",
        checkpoint_dir: str = "./checkpoints",
    ):
        self.system_template = system_template
        self.topology = topology
        self.ligand_indices = perturbed_atom_indices
        self.lambdas_coul = lambdas_coul
        self.lambdas_vdw = lambdas_vdw
        self.temperature = temperature
        self.ranges = window_ranges
        self.softcore = softcore_params
        self.boresch = restraint_params
        self.prefix = prefix
        self.platform_name = platform_name
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        _temp_q = temperature if hasattr(temperature, 'value_in_unit') else temperature * unit.kelvin
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * _temp_q).value_in_unit(unit.kilojoule_per_mole)

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
        debug_mode: bool = True,
    ):
        """
        执行全部窗口采样，能量保存至 {output_dir}/dual_window_{window_idx}_{stage_type}_energies.npy

        诊断功能 (debug_mode=True)：
            - 构建后打印力组架构
            - 关键阶段调用 diagnose_force_breakdown (非侵入式力分解)
            - 若检测到 NaN 或力爆炸，立即抛出异常并终止
        """
        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)
        resolved_box = _resolve_periodic_box_vectors(box_vectors, topology=self.topology, system=self.system_template)
        if _system_requires_periodic_box(self.system_template) and resolved_box is None:
            raise ValueError(
                "当前系统使用了周期性非键方法，但没有可用的周期性盒子。"
                "请检查输入坐标/拓扑是否带 box，或显式传入 box_vectors。"
            )

        for window_idx, (start, end) in enumerate(self.ranges):
            lc_win = self.lambdas_coul[start:end]
            lv_win = self.lambdas_vdw[start:end]
            print(f"\n{'='*80}")
            print(f"[窗口 {window_idx}] 索引 [{start}:{end}] | 状态数: {len(lc_win)}")
            print(f"{'='*80}")

            # ---------- 构建系统 ----------
            win_sys_xml = XmlSerializer.serialize(self.system_template)
            win_sys, ibs_wrap = build_ibs_dual_system(
                ensure_owned_system(XmlSerializer.deserialize(win_sys_xml)),
                self.topology,
                self.ligand_indices,
                lc_win,
                lv_win,
                self.softcore,
                self.boresch,
                self.temperature,
                self.prefix,
                box_vectors=resolved_box,
                reference_positions=positions,
            )
            if debug_mode:
                print_force_group_details(win_sys, prefix=f"窗口{window_idx}_系统构建后")

            integrator = LangevinMiddleIntegrator(self.temperature, 2.0 / unit.picosecond, 0.002 * unit.picosecond)
            integrator.setConstraintTolerance(1e-3)
            if hasattr(integrator, 'setRemoveCMMotion'):
                integrator.setRemoveCMMotion(True)

            sim = app.Simulation(self.topology, win_sys, integrator, platform, props)
            if resolved_box is not None:
                sim.context.setPeriodicBoxVectors(*resolved_box)
            sim.context.setPositions(positions)
            lam_vdw_center = float(np.mean(lv_win))
            sim.context.setParameter("lambda_shield", lam_vdw_center)
            print(f"  🛡️ λ-WCA 防护壳已同步: lambda_shield={lam_vdw_center:.4f}")

            # ---------- 最小化 ----------
            print(f"\n  [阶段1] 开始能量最小化...")
            sim.minimizeEnergy(maxIterations=20000)
            print(f"  ✓ 最小化完成")

            # 几何检查
            if self.boresch and "receptor_indices" in self.boresch:
                ok, r0_chk, thA_chk, thB_chk = _check_boresch_geometry_safe(sim.context, self.boresch)
                if not ok:
                    raise RuntimeError(
                        f"窗口 {window_idx} Boresch 几何不合格 "
                        f"(r0={r0_chk*10:.1f}Å, θA={thA_chk:.1f}°, θB={thB_chk:.1f}°)"
                    )
                print(f"  ✅ Boresch 几何检查通过：r0={r0_chk*10:.2f}Å，θA={thA_chk:.1f}°，θB={thB_chk:.1f}°")

            if debug_mode:
                diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_最小化后")
                diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_最小化后")

            # ---------- 测试步进 ----------
            print(f"\n[阶段2] 测试性步进 (Boresch 缩放至 1%)...")
            if self.boresch and "receptor_indices" in self.boresch:
                # PBC 跨盒修复 (保留原逻辑)
                state_chk = sim.context.getState(getPositions=True)
                pos_chk = state_chk.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                rec_idx = self.boresch["receptor_indices"]
                lig_idx = self.boresch["ligand_indices"]
                H0, L0 = pos_chk[rec_idx[0]], pos_chk[lig_idx[0]]
                raw_dist = np.linalg.norm(H0 - L0)
                if raw_dist > 1.5:
                    shift = H0 - L0
                    lig_mask = np.zeros(len(pos_chk), dtype=bool)
                    lig_mask[self.ligand_indices] = True
                    pos_chk[lig_mask] += shift
                    sim.context.setPositions([openmm.Vec3(*p) for p in pos_chk] * unit.nanometer)
                    print(f"  🔧 已平移配体修复跨盒，新距离={np.linalg.norm(H0 - (L0+shift))*10:.2f}Å")
                else:
                    print(f"  ✅ Boresch 锚点距离正常: {raw_dist*10:.2f}Å")
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
                    if debug_mode and (step_batch == 0 or step_batch >= n_steps - 50):
                        state = sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                        e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
                        max_f = np.max(np.linalg.norm(forces, axis=1))
                        has_nan = np.any(np.isnan(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)))
                        print(f"    [dt={label}] 步{step_batch+actual_steps}: E={e:.1f}, max|F|={max_f:.1f}, NaN={has_nan}")
                        if has_nan or np.isnan(e):
                            print(f"    🚨 NaN 检测，打印力组拆解：")
                            diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_NaN前_dt={label}")
                            diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_NaN前_dt={label}")
                            raise RuntimeError(f"在 dt={label} 处发生坐标 NaN")

            sim.integrator.setStepSize(original_dt)
            print(f"  ✅ 测试步进通过，恢复步长 {original_dt.value_in_unit(unit.picoseconds):.3f} ps")
            if debug_mode:
                diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_测试步进后")

            # ---------- Boresch 安全爬坡 ----------
            # ================================================================
            # Boresch 安全爬坡：自定义阶梯，逐个恢复力强度
            # ================================================================
            if self.boresch and "receptor_indices" in self.boresch:
                print(f"\n[阶段3] Boresch 安全爬坡（自定义阶梯）...")
                
                # 自定义阶梯序列：从 1% 逐步恢复到 100%
                custom_scales = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                n_steps_per_level = 500          # 每个台阶松弛的步数
                dt_ramp = 0.001                  # 爬坡期间极小步长 (ps)
                
                # 保存原始步长，并设置为爬坡步长
                original_dt = sim.integrator.getStepSize()
                sim.integrator.setStepSize(dt_ramp * unit.picoseconds)
                print(f"  → 爬坡使用步长 {dt_ramp} ps，每台阶 {n_steps_per_level} 步")
                
                # 确保从当前 scale 开始（例如之前测试步进时设置的 0.01）
                try:
                    current_scale = sim.context.getParameter("lambda_boresch_scale")
                except Exception:
                    current_scale = 0.01
                print(f"  → 起始 Boresch scale = {current_scale:.3f}")
                
                ramp_success = True
                for target_scale in custom_scales:
                    # 只在目标大于当前值时才爬升（避免回退）
                    if target_scale <= current_scale:
                        continue
                    
                    sim.context.setParameter("lambda_boresch_scale", float(target_scale))
                    print(f"  🔹 设置 Boresch scale = {target_scale:.2f}，松弛 {n_steps_per_level} 步...", end="", flush=True)
                    sim.step(n_steps_per_level)
                    
                    # 检查能量与受力
                    state = sim.context.getState(getEnergy=True, getForces=True)
                    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
                    max_f = np.max(np.linalg.norm(forces, axis=1))
                    
                    if abs(e) > 1e6 or not np.isfinite(e) or max_f > 50000:
                        print(f"\n  🚨 Boresch 爬坡在 scale={target_scale:.2f} 处失败！")
                        print(f"    当前势能 = {e:.1f} kJ/mol，最大力 = {max_f:.1f} kJ/(mol·nm)")
                        if debug_mode:
                            diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_Boresch爬坡失败_scale{target_scale:.2f}")
                        ramp_success = False
                        break
                    else:
                        print(f" 势能 = {e:.2f} kJ/mol，最大力 = {max_f:.2f} kJ/(mol·nm)")
                        # 记录新的当前 scale
                        current_scale = target_scale
                
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
            sampler = IBSSampler(sim.context, len(lc_win), self.temperature, self.prefix, ibs_wrapper=ibs_wrap)
            ibs_state_file = os.path.join(self.checkpoint_dir, f"ibs_state_window_{window_idx}.json")
            
            # 🔑 核心修复：断点续传状态检测
            is_resumed_ibs = False
            if resume and os.path.exists(ibs_state_file):
                is_resumed_ibs = sampler.load_ibs_state(ibs_state_file)

            # ---------- 渐进预热 ----------
            # ---------- 渐进预热 (仅在非续传时执行) ----------
            if not is_resumed_ibs:
                if enable_gradual_warmup:
                    # ... (保留原有的 warmup 逻辑) ...
                    self._gradual_warmup_debug(sim, ibs_wrap, sampler, warmup_steps, win_sys, window_idx, debug_mode)
                else:
                    sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
            else:
                # 🔑 续传时：直接开启偏置，跳过 Warmup，防止二次叠加
                sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
                print(f"  🚀 检测到 IBS 历史状态，跳过 Warmup，直接进入生产采样")


            if debug_mode:
                print(f"\n[诊断] 预热后状态检查：")
                diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_预热后")
                diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_预热后")
                diagnose_softcore_cv_values(sim.context, ibs_wrap, lc_win, lv_win, prefix=f"窗口{window_idx}_预热后")

            # 清空 sampler 的能量缓存
            sampler.energy_buffer = []
            sampler.energy_history = []
            sampler.bias_history = []
            sampler.base_energy_history = []
                
            # ---------- 偏置力预热 (Bias Ramp) ----------
            print(f"\n  🔥 [偏置预热] 开始... (从零缓慢加载偏置力)")
            original_dt = sim.integrator.getStepSize()
            sim.integrator.setStepSize(0.001 * unit.picoseconds)

            # 🔑 关键：先将偏置力完全关闭，再纯偏置爬坡
            sim.context.setParameter(f"{self.prefix}_bias_scale", 0.0)

            ramp_stages = [
                (0.2, 2000),  # 可以适当增加每一步的步数，让系统充分弛豫
                (0.3, 2000), 
                (0.5, 2000),
                (0.7, 2000),
                (1.0, 5000)
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
                    sampler.collect_energies()
                    # 只在偏置力足够大时才更新 f_k，避免弱偏置下噪声主导
                    if target_scale >= 0.5 and len(sampler.energy_buffer) >= 10:
                        sampler.update_weights()
                
                state = sim.context.getState(getEnergy=True)
                if not np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
                    raise RuntimeError(f"偏置预热阶段在 scale={target_scale} 时能量 NaN")
                print(" 完成")


            # 恢复步长
            sim.integrator.setStepSize(original_dt)
            
            # 打印预热后的 f_k 值
            f_vals = [sim.context.getParameter(f"{self.prefix}_f_{k}") for k in range(len(lc_win))]
            print(f"  ✅ 偏置预热完成，f_k 当前值: {[f'{v:.1f}' for v in f_vals]}")
            if max(f_vals) - min(f_vals) < 1.0:
                print(f"  ⚠️ 警告: f_k 仍未明显分化，可能需延长预热或检查窗口重叠度。")
            if debug_mode:
                diagnose_softcore_cv_values(sim.context, ibs_wrap, lc_win, lv_win, prefix=f"窗口{window_idx}_偏置预热后")

            print("\n  🧯 [生产前卸压] 开始...")
            sim.context.setParameter(f"{self.prefix}_bias_scale", 0.0)
            sim.minimizeEnergy(maxIterations=2000)
            for _ in range(20):
                sim.step(250)
            for scale in np.linspace(0.1, 1.0, 15):
                sim.context.setParameter(f"{self.prefix}_bias_scale", float(scale))
                sim.step(1000)

            state_relax = sim.context.getState(getForces=True, getPositions=True)
            forces_relax = state_relax.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            fmax_relax = np.max(np.linalg.norm(forces_relax, axis=1))
            print(f"  卸压后 max|F| = {fmax_relax:.1f} kJ/(mol·nm)")
            if fmax_relax > 3000.0:
                print("  ⚠️ 力仍偏高，建议延长驰豫或进一步缩短积分步长。")

            sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
            sampler.energy_buffer = []
            sampler.energy_history = []
            sampler.bias_history = []
            sampler.base_energy_history = []
            production_pos_backup = state_relax.getPositions(asNumpy=True)

            # ---------- 生产采样 ----------
            print(f"\n  [阶段5] 生产采样 ({n_steps_per_window} 步)...")
            n_updates = n_steps_per_window // steps_per_update
            state_all = sim.context.getState(getEnergy=True, groups=set(range(32)))
            E_total_all = state_all.getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)

            for gid in [0, 2, 3]:
                e_g = sim.context.getState(getEnergy=True, groups={gid}).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                print(f"Group {gid} energy: {e_g:.1f} kJ/mol")

            cv_vals = ibs_wrap.get_force().getCollectiveVariableValues(sim.context)
            print(f"First CV value (should be LE interaction): {cv_vals[0]:.1f} kJ/mol")
            # 获取 CV 力对象（第一个 CV）
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

            print("\n  配体原子前 5 个参数（电荷, sigma, epsilon）：")
            for idx in lig_indices[:5]:
                q, sig, eps = cv_force.getParticleParameters(idx)
                q_val = q.value_in_unit(unit.elementary_charge) if hasattr(q, "value_in_unit") else float(q)
                sig_val = sig.value_in_unit(unit.nanometer) if hasattr(sig, "value_in_unit") else float(sig)
                eps_val = eps.value_in_unit(unit.kilojoule_per_mole) if hasattr(eps, "value_in_unit") else float(eps)
                print(f"    atom {idx}: q={q_val:.3f} e, sigma={sig_val:.4f} nm, epsilon={eps_val:.4f} kJ/mol")

            # 环境原子前 5 个参数
            env_list = [i for i in range(num_atoms) if i not in set(lig_indices)]
            print("\n  环境原子前 5 个参数：")
            for idx in env_list[:5]:
                q, sig, eps = cv_force.getParticleParameters(idx)
                q_val = q.value_in_unit(unit.elementary_charge) if hasattr(q, "value_in_unit") else float(q)
                sig_val = sig.value_in_unit(unit.nanometer) if hasattr(sig, "value_in_unit") else float(sig)
                eps_val = eps.value_in_unit(unit.kilojoule_per_mole) if hasattr(eps, "value_in_unit") else float(eps)
                print(f"    atom {idx}: q={q_val:.3f} e, sigma={sig_val:.4f} nm, epsilon={eps_val:.4f} kJ/mol")

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

                state_n = sim.context.getState(getEnergy=True, getForces=True)
                e_total_n = state_n.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                forces_n = state_n.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                fmax = np.max(np.linalg.norm(forces_n, axis=1))
                if (not np.isfinite(e_total_n)) or (not np.isfinite(fmax)) or fmax > 7000.0:
                    sim.context.setPositions(pos_backup)
                    sim.context.setVelocitiesToTemperature(self.temperature)
                    print("    🔧 触发回退，执行局部最小化释放应力...")
                    sim.minimizeEnergy(maxIterations=2000, tolerance=1.0)
                    current_dt_ps = sim.integrator.getStepSize().value_in_unit(unit.picoseconds)
                    new_dt_ps = max(0.0001, current_dt_ps * 0.5)
                    sim.integrator.setStepSize(new_dt_ps * unit.picoseconds)
                    print(
                        f"    ⚠️ 灾难检测触发: update={up}/{n_updates}, "
                        f"E_total={e_total_n:.1f}, max|F|={fmax:.1f}. "
                        f"已回退坐标并将步长降至 {new_dt_ps*1000.0:.1f} fs"
                    )
                    if debug_mode:
                        diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_回退_update{up}")
                        diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_回退_update{up}")
                    continue

                production_pos_backup = sim.context.getState(getPositions=True).getPositions(asNumpy=True)

                e = sampler.collect_energies()
                if len(sampler.energy_buffer) >= 10:
                    sampler.update_weights()
                    # 🔑 实时落盘 IBS 状态，防止意外中断导致历史遗失
                    sampler.save_ibs_state(ibs_state_file)

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

            remaining_steps = n_steps_per_window % steps_per_update
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
                if (not np.isfinite(e_total_n)) or (not np.isfinite(fmax)) or fmax > 7000.0:
                    sim.context.setPositions(pos_backup)
                    sim.context.setVelocitiesToTemperature(self.temperature)
                    print("    🔧 余数补齐触发回退，执行局部最小化释放应力...")
                    sim.minimizeEnergy(maxIterations=2000, tolerance=1.0)
                    current_dt_ps = sim.integrator.getStepSize().value_in_unit(unit.picoseconds)
                    new_dt_ps = max(0.0001, current_dt_ps * 0.5)
                    sim.integrator.setStepSize(new_dt_ps * unit.picoseconds)
                    print(
                        f"    ⚠️ 余数补齐灾难检测触发: E_total={e_total_n:.1f}, max|F|={fmax:.1f}. "
                        f"已回退坐标并将步长降至 {new_dt_ps*1000.0:.1f} fs"
                    )
                    if debug_mode:
                        diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_余数补齐回退")
                        diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_余数补齐回退")
                else:
                    production_pos_backup = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
                    e = sampler.collect_energies()
                    if len(sampler.energy_buffer) >= 10:
                        sampler.update_weights()
                        sampler.save_ibs_state(ibs_state_file)
                    print(f"    [采样] 补齐余数 {remaining_steps} 步，总步数严格达标 {n_steps_per_window}")
                    if debug_mode:
                        has_nan_energy = not np.isfinite(e_total_n)
                        has_nan_e = np.any(np.isnan(e))
                        print(
                            f"    [采样-余数] E_total={e_total_n:.1f}, "
                            f"NaN(E_total)={has_nan_energy}, NaN(E_k)={has_nan_e}"
                        )

            # ---------- 保存能量 ----------
            e_arr = np.array(sampler.energy_history) if sampler.energy_history else np.zeros((0, len(lc_win)))
            e_save = e_arr.T if e_arr.size > 0 else np.zeros((len(lc_win), 0))
            np.save(os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_energies.npy"), e_save)

            if sampler.bias_history:
                np.save(os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_bias.npy"),
                        np.array(sampler.bias_history))
            if sampler.base_energy_history:
                np.save(os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_base.npy"),
                        np.array(sampler.base_energy_history))
            print(f"  💾 窗口 {window_idx} 完成，能量已保存 ({e_save.shape})")

            # 清理
            del sim.context
            del sim
            del win_sys
            gc.collect()

        print(f"\n{'='*80}")
        print(f"✅ 所有窗口采样完成")
        print(f"{'='*80}")


    def _gradual_warmup_debug(self, sim, ibs_wrap, sampler, warmup_steps, win_sys, window_idx, debug_mode=True):
        """带诊断的渐进预热"""
        print("  🔥 渐进预热 (Debug 模式)...")
        
        # 简化版步长爬坡（带诊断）
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

        # ================================================================
        # 阶段 4：纯偏置预热
        # ================================================================
        print("  🔥 渐进预热 (Bias-Ramp 模式)...")
        total_steps = 20000
        chunk = 500

        for s in range(0, total_steps, chunk):
            progress = min(1.0, (s + chunk) / total_steps)
            
            # λ-WCA 防护壳由窗口中心 lambda_shield 固定控制，预热阶段仅爬升偏置
            sim.context.setParameter(f"{self.prefix}_bias_scale", progress)
            
            sim.step(chunk)
            sampler.collect_energies()
            if len(sampler.energy_buffer) >= 10:
                sampler.update_weights()
                
            if debug_mode and s % 5000 == 0:
                state = sim.context.getState(getEnergy=True)
                e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                print(f"    [预热 progress={progress:.2f}] E={e:.1f}, bias_scale={progress:.2f}")

        # 确保最终状态
        sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
        print(f"  ✅ 纯偏置预热完成 (bias_scale=1.0)")
                
        if debug_mode:
            print(f"  ✅ 渐进预热完成")


    def _safe_boresch_ramp(self, sim, final_scale=1.0, n_steps=200):
        """
        渐进式提升 Boresch 力强度，每步后用极短时间步长松弛并检测能量。
        返回 True 表示爬坡成功，False 表示在中途检测到异常。
        """
        try:
            current_scale = sim.context.getParameter("lambda_boresch_scale")
        except Exception:
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

    def get_stage_data_for_analysis(self, stage_type: str = "coul") -> List[Dict]:
        """从磁盘加载能量与偏置，构造分析器期望的 window_data 列表"""
        results = []
        for i, (start, end) in enumerate(self.ranges):
            e_path = os.path.join(self.output_dir, f"dual_window_{i}_{stage_type}_energies.npy")
            b_path = os.path.join(self.output_dir, f"dual_window_{i}_{stage_type}_bias.npy")
            base_path = os.path.join(self.output_dir, f"dual_window_{i}_{stage_type}_base.npy")
            if not os.path.exists(e_path):
                continue
            u_kn = np.load(e_path)  # 预期 (K, N)
            if u_kn.shape[1] == 0:
                continue
            bias = np.load(b_path) if os.path.exists(b_path) else np.zeros(u_kn.shape[1])
            base = np.load(base_path) if os.path.exists(base_path) else np.zeros(u_kn.shape[1])
            n_frames = min(u_kn.shape[1], len(bias), len(base))
            if n_frames == 0:
                continue
            results.append({
                'u_kn': u_kn[:, :n_frames],  # U_k_int, kJ/mol
                'bias_energies': bias[:n_frames],
                'base_energies': base[:n_frames],
                'lambda_indices': list(range(start, end))
            })
        return results

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

    def solve_stage_integrated(self, window_data: List[Dict]) -> Dict:
        """局部 TMBAR + 自洽拼接"""
        valid_windows = [w for w in window_data if w.get("u_kn") is not None and w["u_kn"].size > 0]
        if not valid_windows:
            return {"error": "no_valid_windows", "converged": False, "total_delta_G": 0.0}

        # 按 λ 索引排序，确保拼接顺序正确
        valid_windows.sort(key=lambda d: min(d.get("lambda_indices", [10**9])))

        local_results = []
        
        for w_idx, w in enumerate(valid_windows):
            u_kj_raw = np.asarray(w["u_kn"], dtype=np.float64) # (K_local, N)
            bias_kj = np.asarray(w.get("bias_energies"), dtype=np.float64) if w.get("bias_energies") is not None else np.zeros(u_kj_raw.shape[1])
            base_kj = np.asarray(w.get("base_energies"), dtype=np.float64) if w.get("base_energies") is not None else np.zeros(u_kj_raw.shape[1])
            win_lams = list(w.get("lambda_indices", []))
            
            if u_kj_raw.ndim != 2 or len(win_lams) != u_kj_raw.shape[0]:
                print(f"  ⚠️ 窗口 {w_idx} 数据维度不匹配，跳过")
                continue
            
            # 确定采样态索引 (通常 IBS 采样的是当前窗口的某个特定态，或者是加权平均)
            # 在 IBS 引擎中，通常假设样本来自当前窗口的“有效采样分布”。
            # 如果未指定 sampled_lambda_index，默认取窗口中间态作为参考，或者取第一个态。
            # 这里为了兼容旧逻辑，若未指定则取中间态，但需注意 bias_energies 必须对应正确的采样态。
            n_frames = u_kj_raw.shape[1]
            if n_frames < 10:
                continue
            n_frames = min(n_frames, len(bias_kj), len(base_kj))
            u_kj_raw = u_kj_raw[:, :n_frames]
            bias_kj = bias_kj[:n_frames]
            base_kj = base_kj[:n_frames]

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
            
            # 样本计数：只有第 0 行（采样分布）有 N 个样本
            n_k_local = np.zeros(len(win_lams) + 1, dtype=np.int32)
            n_k_local[0] = n_frames
            
            # ------------------------------------------------------------------
            # 🔑 修复 3: 列平移稳定性 (Column-wise Shift)
            # ------------------------------------------------------------------
            # 进一步对每一列减去最小值，防止 exp 溢出
            u_min_col = np.min(u_mbar, axis=0, keepdims=True)
            u_mbar_stable = u_mbar - u_min_col
            
            # 剔除含 NaN 或 Inf 的列
            valid_mask = np.isfinite(u_mbar_stable).all(axis=0)
            u_mbar_final = u_mbar_stable[:, valid_mask]
            n_k_local[0] = np.sum(valid_mask) # 更新有效样本数
            
            if n_k_local[0] < 10:
                continue

            if not HAS_PYMBAR:
                return {"error": "pymbar_not_installed", "converged": False}
            
            try:
                # 使用混合求解器，提高收敛性
                mbar = pymbar.MBAR(u_mbar_final, n_k_local, relative_tolerance=1e-7,
                                   initialize='BAR', solver_protocol='hybr')
                
                res = mbar.compute_free_energy_differences()
                
                # Delta_f[0, k] 是第 k 个物理态相对于采样态 (Row 0) 的自由能差 (单位: kT)
                # 注意：res['Delta_f'] 的形状是 (K+1, K+1)
                # 我们想要的是物理态 (indices 1..K) 的结果
                f_phys_kt = res['Delta_f'][0, 1:] 
                df_phys_kt = res['dDelta_f'][0, 1:]
                
                # 转换为 kJ/mol
                f_phys_kj = f_phys_kt * self.kt
                df_phys_kj = df_phys_kt * self.kt
                
                # 注意：这里的 f_phys_kj 是相对于“采样态有效能量”的绝对自由能估计。
                # 为了拼接，我们需要相对于窗口内某个特定 lambda 的相对自由能。
                # 通常我们关心的是窗口内各 lambda 之间的相对差值。
                # 由于所有态都减去了同一个 global_offset，且参考态一致，相对差值是准确的。
                
                local_results.append({
                    "lambdas": win_lams,
                    "f": f_phys_kj.astype(float), # 相对于采样参考点的绝对 F
                    "df": df_phys_kj.astype(float),
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
            # 对于重叠的 lambda，计算 (Global_F - Local_F) 的平均值
            offsets = []
            for lam in overlap_lams:
                idx_in_local = local["lambdas"].index(lam)
                # 局部自由能
                f_loc = float(local["f"][idx_in_local])
                # 全局自由能
                f_glob = global_f[lam]
                offsets.append(f_glob - f_loc)
            
            offset = float(np.mean(offsets))
            
            # 应用偏移量并合并
            for i, lam in enumerate(local["lambdas"]):
                f_loc = float(local["f"][i]) + offset
                var_loc = float(max(local["df"][i], 1e-6)**2)
                
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
        
        # 总自由能变化 (从第一个 lambda 到最后一个)
        total_dg = float(f_curve[-1] - f_curve[0])
        total_err = float(np.sqrt(err_curve[0]**2 + err_curve[-1]**2))
        
        # 简单的重叠度代理指标
        min_overlap_proxy = 1.0 / (1.0 + float(np.max(np.abs(np.diff(f_curve)))) + 1e-6)
        
        return {
            "total_delta_G": total_dg,
            "total_error": total_err,
            "converged": len(local_results) == len(valid_windows),
            "min_overlap": min_overlap_proxy,
            "method": "Local-TMBAR-Stitched (Fixed)",
            "f_k": f_curve.tolist(),
            "df_k": err_curve.tolist(),
            "lambdas": sorted_lams
        }

    def _fallback(self, msg: str) -> Dict:
        return {"error": msg, "converged": False, "total_delta_G": 0.0, "total_error": 999.9}

# 模块级便捷入口
def solve_stage_integrated(window_outputs: List[Dict], kt: float, stage_name: str = "") -> Dict:
    if not window_outputs:
        return {"total_delta_G": 0.0, "total_error": 999.9, "converged": False}
    analyzer = GlobalMBARAnalyzer(kt=kt)
    res = analyzer.solve_stage_integrated(window_outputs)
    res.setdefault("stage", stage_name)
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
    rec = boresch_params["receptor_indices"]
    lig = boresch_params["ligand_indices"]

    H0, H1, H2 = pos[rec[0]], pos[rec[1]], pos[rec[2]]
    L0, L1, L2 = pos[lig[0]], pos[lig[1]], pos[lig[2]]

    r0 = np.linalg.norm(H0 - L0)

    def calc_angle(a, b, c):
        ba, bc = a - b, c - b
        cos_val = np.clip(
            np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10),
            -1.0, 1.0
        )
        return np.arccos(cos_val)

    thA = calc_angle(H1, H0, L0)
    thB = calc_angle(H0, L0, L1)

    sinA, sinB = np.sin(thA), np.sin(thB)
    if sinA < min_sin_theta or sinB < min_sin_theta:
        print(f"  🚨 Boresch 角度接近奇点：θA={np.degrees(thA):.1f}° (sin={sinA:.3f})，"
              f"θB={np.degrees(thB):.1f}° (sin={sinB:.3f})")
        return False, r0, np.degrees(thA), np.degrees(thB)

    return True, r0, np.degrees(thA), np.degrees(thB)

def diagnose_force_groups_detailed(context, system, prefix="窗口诊断"):
    """生产级逐力组受力拆解：精准定位哪个 ForceGroup 导致 NaN/爆炸"""
    from openmm import unit
    print(f"\n🔍 [{prefix}] 逐力组受力拆解报告:")
    print(f"{'ID':<4} | {'Group':<8} | {'Force Type':<28} | {'Max|F| (kJ/mol/nm)':<20} | {'RMS|F|':<12} | {'状态'}")
    print("-" * 95)
    for i, force in enumerate(system.getForces()):
        gid = force.getForceGroup()
        ftype = type(force).__name__
        try:
            state = context.getState(getForces=True, groups={gid})
            forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/openmm.unit.nanometer)
            if forces.size == 0:
                continue
            norms = np.linalg.norm(forces, axis=1)
            max_f = np.max(norms)
            rms_f = np.sqrt(np.mean(norms**2))
            status = "🚨 爆炸源" if max_f > 10000 else ("⚠️ 偏高" if max_f > 2000 else "✓ 正常")
            print(f"{i:<4} | Group {gid:<4} | {ftype:<28} | {max_f:<20.2f} | {rms_f:<12.2f} | {status}")
        except Exception as e:
            print(f"{i:<4} | Group {gid:<4} | {ftype:<28} | {'(CV/元力跳过)':<20} | {'N/A':<12} | ℹ️ {str(e)[:25]}")
    print("-" * 95)
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
    for gid, name in group_map.items():
        try:
            fstate = diag_context.getState(getForces=True, groups={gid})
            forces = fstate.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            norms = np.linalg.norm(forces, axis=1)
            max_f, avg_f, rms_f = np.max(norms), np.mean(norms), np.sqrt(np.mean(norms**2))
            print(f"  {name:<12} | Max:{max_f:12.2f} | Avg:{avg_f:12.2f} | RMS:{rms_f:12.2f} kJ/(mol·nm)")
        except Exception as e:
            print(f"  {name:<12} | 无法获取: {e}")

    # 清理
    del diag_context
    del diag_sys
    gc.collect()


# ============================================================================
# 6. 传统λ-REMD 采样引擎 (从 traditional_abfe_remd.py 迁移)
# ============================================================================
class REMDManager:
    """传统 λ-REMD 引擎：相邻窗口 Metropolis 交换 + 轨迹落盘"""
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
    ):
        self.topology = topology
        self.positions = positions
        self.box_vectors = box_vectors
        self.ligand_indices = ligand_indices
        self.lambdas_coul = np.array(lambdas_coul)
        self.lambdas_vdw = np.array(lambdas_vdw)
        self.n_replicas = len(lambdas_coul)
        self.temperature = temperature * unit.kelvin
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(unit.kilojoule_per_mole)
        self.beta = 1.0 / self.kt
        self.platform_name = platform_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.contexts = []
        self.integrators = []
        self._build_replicas(system_template)

    def _build_replicas(self, system_template):
        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)

        for i in range(self.n_replicas):
            sys_xml = openmm.XmlSerializer.serialize(system_template)
            replica_sys = openmm.XmlSerializer.deserialize(sys_xml)
            replica_sys.thisown = 1

            nb = [f for f in replica_sys.getForces() if isinstance(f, openmm.NonbondedForce)][0]
            env_idx = [j for j in range(replica_sys.getNumParticles()) if j not in self.ligand_indices]
            sc_force = BeutlerSoftcoreBuilder.build(nb, self.ligand_indices, env_idx)
            sc_force.setForceGroup(1)
            replica_sys.addForce(sc_force)

            for idx in self.ligand_indices:
                nb.setParticleParameters(idx, 0.0, 0.1*unit.nanometer, 0.0)

            integ = openmm.LangevinMiddleIntegrator(self.temperature, 1.0/unit.picosecond, 0.002*unit.picosecond)
            ctx = openmm.Context(replica_sys, integ, platform, props)
            ctx.setPositions(self.positions)
            if self.box_vectors:
                ctx.setPeriodicBoxVectors(*self.box_vectors)
            ctx.setParameter("lambda_coul", float(self.lambdas_coul[i]))
            ctx.setParameter("lambda_vdw", float(self.lambdas_vdw[i]))

            self.contexts.append(ctx)
            self.integrators.append(integ)

    def run(
        self,
        n_steps: int = 500000,
        exchange_interval: int = 1000,
        save_interval: int = 5000,
        stage_name: str = "complex",
    ):
        n_exchanges = n_steps // exchange_interval
        traj_files = [os.path.join(self.output_dir, f"{stage_name}_rep{i}.dcd") for i in range(self.n_replicas)]
        reporters = [app.DCDReporter(f, save_interval, enforcePeriodicBox=False) for f in traj_files]

        print(f"\n🔄 启动传统 REMD | {self.n_replicas} 副本 | {n_steps} 步 | 交换间隔={exchange_interval}")

        for ctx in self.contexts:
            ctx.getIntegrator().step(5000)

        exchange_log = []
        for step in range(n_exchanges):
            for ctx in self.contexts:
                ctx.getIntegrator().step(exchange_interval)

            for i, ctx in enumerate(self.contexts):
                state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
                reporters[i].report(app.Simulation(self.topology, ctx.getSystem(), ctx.getIntegrator()), state)

            accepted = 0
            for i in range(self.n_replicas - 1):
                ctx_i, ctx_j = self.contexts[i], self.contexts[i+1]
                state_i = ctx_i.getState(getEnergy=True, getPositions=True, getVelocities=True)
                state_j = ctx_j.getState(getEnergy=True, getPositions=True, getVelocities=True)

                U_i_i = state_i.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                U_j_j = state_j.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                ctx_i.setParameter("lambda_coul", float(self.lambdas_coul[i+1]))
                ctx_i.setParameter("lambda_vdw", float(self.lambdas_vdw[i+1]))
                ctx_i.setPositions(state_j.getPositions())
                U_i_j = ctx_i.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                ctx_j.setParameter("lambda_coul", float(self.lambdas_coul[i]))
                ctx_j.setParameter("lambda_vdw", float(self.lambdas_vdw[i]))
                ctx_j.setPositions(state_i.getPositions())
                U_j_i = ctx_j.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

                ctx_i.setParameter("lambda_coul", float(self.lambdas_coul[i]))
                ctx_i.setParameter("lambda_vdw", float(self.lambdas_vdw[i]))
                ctx_j.setParameter("lambda_coul", float(self.lambdas_coul[i+1]))
                ctx_j.setParameter("lambda_vdw", float(self.lambdas_vdw[i+1]))

                delta = self.beta * (U_i_j + U_j_i - U_i_i - U_j_j)
                if delta < 0 or np.random.rand() < np.exp(-delta):
                    ctx_i.setPositions(state_j.getPositions())
                    ctx_i.setVelocities(state_j.getVelocities())
                    ctx_j.setPositions(state_i.getPositions())
                    ctx_j.setVelocities(state_i.getVelocities())
                    accepted += 1

            exchange_log.append(accepted / (self.n_replicas - 1))
            if step % 50 == 0:
                print(f"  [REMD] 交换轮次 {step}/{n_exchanges} | 接受率: {exchange_log[-1]:.2f}")

        for rep in reporters:
            rep.close()
        print(f"✅ REMD 完成 | 平均交换接受率: {np.mean(exchange_log):.3f}")
        return traj_files


# ============================================================================
# 7. 传统离线 MBAR 分析器 (从 traditional_abfe_remd.py 迁移)
# ============================================================================
class TraditionalMBARAnalyzer:
    """标准离线 MBAR：读取轨迹 → 重算 u_kn → 求解 ΔG"""
    def __init__(self, temperature: float = 300.0):
        self.kt = 0.008314462618 * temperature
        self.beta = 1.0 / self.kt

    def compute_u_kn(
        self,
        traj_files: List[str],
        system_template: openmm.System,
        ligand_indices: List[int],
        lambdas_coul: List[float],
        lambdas_vdw: List[float],
        platform_name: str = "CPU",
    ) -> np.ndarray:
        import mdtraj as md
        md_top = md.Topology.from_openmm(system_template)

        traj = md.load(traj_files, top=md_top)
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

        platform = openmm.Platform.getPlatformByName(platform_name)
        eval_sys = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system_template))
        nb = [f for f in eval_sys.getForces() if isinstance(f, openmm.NonbondedForce)][0]
        env_idx = [i for i in range(eval_sys.getNumParticles()) if i not in ligand_indices]
        sc = BeutlerSoftcoreBuilder.build(nb, ligand_indices, env_idx)
        sc.setForceGroup(1)
        eval_sys.addForce(sc)
        for idx in ligand_indices:
            nb.setParticleParameters(idx, 0.0, 0.1*unit.nanometer, 0.0)

        integ = openmm.VerletIntegrator(0.001)
        ctx = openmm.Context(eval_sys, integ, platform)

        print(f"\n📊 开始离线能量重算 | {n_frames} 帧 × {n_states} 态")
        for f in range(n_frames):
            ctx.setPositions(traj.xyz[f] * unit.nanometer)
            if traj.unitcell_vectors is not None and len(traj.unitcell_vectors) > 0:
                ctx.setPeriodicBoxVectors(*traj.unitcell_vectors[f] * unit.nanometer)

            for k in range(n_states):
                ctx.setParameter("lambda_coul", float(lambdas_coul[k]))
                ctx.setParameter("lambda_vdw", float(lambdas_vdw[k]))
                e = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                u_kn[k, f] = e / self.kt

            if f % 100 == 0:
                print(f"  → 帧 {f}/{n_frames} 完成")

        del ctx, integ, eval_sys
        return u_kn

    def solve(self, u_kn: np.ndarray) -> Dict:
        K, N = u_kn.shape
        n_k = np.full(K, N, dtype=int)
        if not HAS_PYMBAR:
            raise ImportError("需要 pymbar 包，请安装: pip install pymbar")
        mbar = pymbar.MBAR(u_kn, n_k, solver_protocol="hybr", verbose=False)
        res = mbar.compute_free_energy_differences()

        dg = (res["Delta_f"][0, -1] - res["Delta_f"][0, 0]) * self.kt
        err = res["dDelta_f"][0, -1] * self.kt
        return {"delta_G": dg, "error": err, "method": "MBAR", "n_frames": N, "n_states": K}
