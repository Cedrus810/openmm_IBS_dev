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
from typing import Dict, List, Tuple, Optional, Any
from abfe_core import (
    ACESoftcorePotential,
    AlchemicalPotentialFactory,
    DEXPSurrogatePotential,
    LambdaDependentBoreschForce,
    create_ligand_internal_force,
    ensure_owned_system,
    sync_all_exclusions,
    BeutlerSoftcoreBuilder,
    _build_mbar_compatible,
    _compute_free_energy_result_compatible,
    _extract_free_energy_arrays,
)

try:
    import pymbar
    HAS_PYMBAR = True
except ImportError:
    HAS_PYMBAR = False

logger = logging.getLogger(__name__)


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
    logger.log(_infer_log_level_from_message(message), message)
    if flush and not logger.handlers:
        builtins.print(message, end=end, file=file, flush=flush)


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


def _atomic_write_json(filepath: str, payload: Dict[str, Any]) -> None:
    """同步原子写 JSON：先写临时文件，再原子替换。"""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_file = filepath + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
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


def _select_bulk_water_counterion(
    nb_force: openmm.NonbondedForce,
    ligand_indices: List[int],
    topology,
    positions,
) -> Tuple[Optional[int], Optional[np.ndarray], Dict[str, float]]:
    """选择位于 bulk 水区域的匹配反离子。"""
    pos_nm = _positions_to_nm_array(positions)
    ligand_set = set(int(i) for i in ligand_indices)
    lig_net_charge = 0.0
    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(idx)
        lig_net_charge += q.value_in_unit(unit.elementary_charge)
    lig_net_charge = round(lig_net_charge)
    if abs(lig_net_charge) < 0.01:
        return None, None, {}

    target_ion_charge = -1.0 if lig_net_charge > 0 else 1.0
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
    solute_anchor = np.mean(pos_nm[list(ligand_set) + heavy_solute_indices], axis=0) if heavy_solute_indices else np.mean(pos_nm[list(ligand_set)], axis=0)

    water_oxygen_indices = [
        a.index for a in topology.atoms()
        if a.residue.name.upper() in water_names and getattr(a.element, "symbol", "").upper() == "O"
    ]

    best_ion_idx = None
    best_score = -1e18
    best_meta: Dict[str, float] = {}
    for atom in topology.atoms():
        if atom.residue.name.upper() not in ion_names:
            continue
        idx = atom.index
        q, _, _ = nb_force.getParticleParameters(idx)
        q_val = q.value_in_unit(unit.elementary_charge)
        if abs(q_val - target_ion_charge) >= 0.1:
            continue

        ion_pos = pos_nm[idx]
        solute_dist = float(np.linalg.norm(ion_pos - solute_anchor))
        water_coord = 0
        if water_oxygen_indices:
            water_dists = np.linalg.norm(pos_nm[water_oxygen_indices] - ion_pos, axis=1)
            water_coord = int(np.sum(water_dists <= 0.45))
        score = solute_dist + 0.15 * water_coord
        if score > best_score:
            best_score = score
            best_ion_idx = idx
            best_meta = {
                "solute_distance_nm": solute_dist,
                "water_coordination": float(water_coord),
                "target_charge_e": float(target_ion_charge),
            }

    if best_ion_idx is None:
        return None, None, {}
    return best_ion_idx, pos_nm[best_ion_idx].copy(), best_meta


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


def _prepare_pme_mixed_alchemical_system(
    system_template: openmm.System,
    ligand_indices: List[int],
    topology,
    positions,
    box_vectors=None,
    lambda_coul_name: str = "lambda_coul",
    lambda_vdw_name: str = "lambda_vdw",
) -> openmm.System:
    """构建同时支持 PME 去电荷与软核去 VDW 的联合炼金体系。"""
    mixed_sys = openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(system_template)
    )
    mixed_sys.thisown = 1
    _prepare_pme_coulomb_leg_system(
        mixed_sys,
        ligand_indices,
        lambda_name=lambda_coul_name,
        allow_charged_ligand=True,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
    )

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
    )
    sc_force.setForceGroup(1)
    mixed_sys.addForce(sc_force)
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
    pme_self_prefactor_kj = None
    if use_total_energy and apply_pme_self_correction:
        lig_qsq = float(task.get("ligand_charge_square_sum", 0.0))
        if lig_qsq > 0.0:
            nb_force = next((f for f in eval_sys.getForces() if isinstance(f, openmm.NonbondedForce)), None)
            if nb_force is not None:
                try:
                    alpha_ewald, _, _, _ = nb_force.getPMEParametersInContext(ctx)
                    alpha_ewald = _value_in_inverse_nanometer(alpha_ewald)
                except Exception:
                    alpha_q, _, _, _ = nb_force.getPMEParameters()
                    alpha_ewald = _value_in_inverse_nanometer(alpha_q)
                if alpha_ewald > 0.0:
                    pme_self_prefactor_kj = 138.935456 * alpha_ewald * lig_qsq / math.sqrt(math.pi)

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
                    e += pme_self_prefactor_kj * (lambdas_coul[k] ** 2)
            else:
                e = ctx.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
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
) -> Tuple[Dict, Optional[int]]:
    """
    🔥 终极防御：共炼金反离子策略
    寻找体系中最远的反离子，使其与配体同步消电，保持全局严格电中性。
    """
    nb_force = next((f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
    if nb_force is None:
        raise RuntimeError("系统中未找到 NonbondedForce")

    ligand_set = set(int(i) for i in ligand_indices)
    lig_net_charge = 0.0
    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(idx)
        lig_net_charge += q.value_in_unit(unit.elementary_charge)
    lig_net_charge = round(lig_net_charge)

    if abs(lig_net_charge) < 0.01:
        print("  ℹ️ 配体为电中性，无需共消电反离子。使用标准 PME Offset。")
        target_ion_charge = 0.0
        best_ion_idx = None
        ion_ref_pos_nm = None
        ion_meta = {}
    else:
        print(f"  ⚡ 检测到带电配体 (Net Charge: {lig_net_charge:+d})，启动共炼金反离子搜索...")
        target_ion_charge = -1.0 if lig_net_charge > 0 else 1.0
        best_ion_idx, ion_ref_pos_nm, ion_meta = _select_bulk_water_counterion(
            nb_force, ligand_indices, topology, positions
        )
        if best_ion_idx is not None:
            print(
                f"  🎯 锁定共消电反离子: Index {best_ion_idx}, "
                f"距溶质 {ion_meta.get('solute_distance_nm', float('nan')):.2f} nm, "
                f"水配位={int(ion_meta.get('water_coordination', 0))}"
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
    if best_ion_idx is not None:
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

    if best_ion_idx is not None and ion_ref_pos_nm is not None:
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
    return original_charges, best_ion_idx


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
    rounded_charge = round(net_charge)
    if abs(rounded_charge) > 0.01 and not allow_charged_ligand:
        raise RuntimeError(
            f"检测到带净电配体 ({rounded_charge:+d} e)。当前 PME 去电荷路径不再自动共炼金反离子，"
            "请先使用净电中性配体、显式实现可验证的离子方案，或改用支持 PME 多态炼金的底层实现。"
        )

    if abs(rounded_charge) > 0.01:
        if topology is None or positions is None:
            raise RuntimeError("带电配体的 PME 去电荷路径需要 topology 和初始 positions 以构建共炼金反离子 bulk 水锚定。")
        original_charges, ion_idx = configure_coalchemical_neutral_decharging(
            system,
            ligand_indices,
            topology,
            positions,
            box_vectors=box_vectors,
            lambda_name=lambda_name,
        )
        return {
            "mode": "coalchemical_counterion",
            "ion_index": ion_idx,
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
        f"(sigma12^12 / (r^6 + {alpha_lj}*(1.0-{lam_v_str}+1e-9)^{m_lj} + 1e-4)^2) - "
        f"(sigma12^6 / (r^6 + {alpha_lj}*(1.0-{lam_v_str}+1e-9)^{m_lj} + 1e-4))"
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
    expr, _ = AlchemicalPotentialFactory.build(
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


def _normalize_alchemical_params(alchemical_params, potential_type: str, n_perturbed: int):
    if potential_type == "dexp":
        if isinstance(alchemical_params, DEXPSurrogatePotential):
            return alchemical_params
        return DEXPSurrogatePotential.from_dict(alchemical_params or {})
    if isinstance(alchemical_params, ACESoftcorePotential):
        return _normalize_softcore_params(alchemical_params, n_perturbed)
    return _normalize_softcore_params(
        ACESoftcorePotential.from_dict(alchemical_params or {}),
        n_perturbed,
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
    sampler: "IBSSampler" = None,
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
        reference_exclusions=softcore_excl,
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
            reference_exclusions=softcore_excl,
            particle_params_override=original_params_fresh,
            num_particles=num_atoms,
        )
        int_f_cv.setCutoffDistance(template_cutoff)
        int_f_cv.setSwitchingDistance(template_switch)
        int_f_cv.setNonbondedMethod(template_method)
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
        # 我们采用标准的 Log-Sum-Exp 技巧，但强制以 k=0 为基准进行平移。
        
        terms = []
        # k=0 的项作为基准，显式写出 exp(0) = 1
        # 对于 k>0，计算相对指数
        for k in range(1, n_states):
            # 差分项: (cv_k_int + cv_k_rest - f_k) - (cv_0_int + cv_0_rest - f_0)
            diff_expr = f"(cv_{k}_int + cv_{k}_rest - {prefix}_f_{k}) - (cv_0_int + cv_0_rest - {prefix}_f_0)"
            # 使用平滑饱和替代 max/min 硬截断，避免偏置力在边界处发生不可导突变。
            smooth_diff = f"(80.0*tanh(((-beta * ({diff_expr})))/80.0))"
            terms.append(f"exp({smooth_diff})")
        
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

    def _collect_interaction_energies(self) -> np.ndarray:
        if self._probe_context is None or not self._probe_groups:
            return np.zeros(self.n_states, dtype=float)

        state_main = self.context.getState(getPositions=True)
        self._probe_context.setPositions(state_main.getPositions())
        try:
            self._probe_context.setPeriodicBoxVectors(*self.context.getState().getPeriodicBoxVectors())
        except Exception:
            pass

        interaction_energies = np.zeros(self.n_states, dtype=float)
        for k, gid in enumerate(self._probe_groups[:self.n_states]):
            state = self._probe_context.getState(getEnergy=True, groups={gid})
            interaction_energies[k] = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        return interaction_energies

    def get_raw_interaction_energies(self) -> np.ndarray:
        return self._collect_interaction_energies().copy()

    def collect_energies(self) -> np.ndarray:
        energies = np.zeros(self.n_states)
        # 1. Base 能量 (Group 0, 2, 3, 4, 5)：现已严格 λ 无关
        try:
            state_base = self.context.getState(getEnergy=True, groups={0, 2, 3, 4, 5})
            e_base = state_base.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        except Exception:
            e_base = 0.0
        
        if self.ibs_wrapper is None:
            return np.full(self.n_states, np.nan)
            
        try:
            try:
                state_bias = self.context.getState(getEnergy=True, groups={1})
                e_bias = state_bias.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            except Exception:
                e_bias = np.nan

            interaction_energies = self._collect_interaction_energies()
                
            # 相对偏移防溢出 (以 State 0 为参考)
            if self.n_states > 0 and np.isfinite(interaction_energies[0]):
                self.e_offset = interaction_energies[0]
            energies = interaction_energies - self.e_offset
            
            if not np.any(np.isnan(energies)):
                self.energy_buffer.append(energies)
                self.energy_history.append(interaction_energies.copy())
                self.base_energy_history.append(float(e_base))
                self.bias_history.append(float(e_bias))
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

        if K > 2 and np.std(mean_p_batch[1:]) < 1e-4 and mean_p_batch[0] > 3.0 * target_p:
            mean_u = np.mean(u_mk, axis=0)
            adjacent_span = np.abs(np.diff(mean_u))
            max_adjacent_span = float(np.max(adjacent_span)) if adjacent_span.size else 0.0
            total_span = float(np.max(mean_u) - np.min(mean_u))
            print(
                "  ⚠️ IBS 权重覆盖疑似坍缩：state 0 占据显著偏高，其余状态概率几乎一致。"
                " 这通常意味着偏置过陡、窗口能量饱和，或体系仍未充分松弛。"
            )
            print(
                f"     → 当前窗口平均 ΔU 跨度={total_span:.1f} kJ/mol，"
                f"最大相邻 ΔU={max_adjacent_span:.1f} kJ/mol；"
                "若远大于 50 kJ/mol，请增加该阶段 λ 状态数或缩小窗口。"
            )
        
        t = len(self.f_history) + 1
        eta_sgd = 1.0 / (1.0 + t / 100.0)
        
        f_new = f_old - eta_sgd * self.kt * log_grad
        # 去除任意零点漂移，避免 f_k 仅因规范选择而表现为整体阶跃平移。
        f_new = f_new - np.mean(f_new)
        
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
        """同步保存 IBS 状态，使用原子替换避免损坏。"""
        f_current = [self.context.getParameter(f"{self.prefix}_f_{k}") for k in range(self.n_states)]
        state = {
            "n_states": int(self.n_states),
            "prefix": self.prefix,
            "f_k": f_current,
            "t": len(self.f_history),
            "e_offset": self.e_offset,
            "status": "running",
        }
        _atomic_write_json(filepath, state)

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
            if len(f_k) != self.n_states or not np.all(np.isfinite(np.asarray(f_k, dtype=float))):
                print("  ⚠️ IBS 状态 f_k 无效，忽略旧状态")
                return False
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
        alchemical_params,
        potential_type: str = "softcore",
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
        self.alchemical_params = alchemical_params
        self.potential_type = potential_type
        self.boresch = restraint_params
        self.prefix = prefix
        self.platform_name = platform_name
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        _temp_q = temperature if hasattr(temperature, 'value_in_unit') else temperature * unit.kelvin
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * _temp_q).value_in_unit(unit.kilojoule_per_mole)

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
                self.alchemical_params,
                self.potential_type,
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

            # ---------- 测试步进 ----------
            print(f"\n[阶段2] 测试性步进 (Boresch 缩放至 1%)...")
            if _has_valid_boresch_restraint(self.boresch):
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
            sampler = IBSSampler(sim.context, len(lc_win), self.temperature, self.prefix, ibs_wrapper=ibs_wrap)
            ibs_state_file = os.path.join(
                self.checkpoint_dir,
                f"ibs_state_{stage_type}_window_{window_idx}.json",
            )
            
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
                diagnose_softcore_cv_values(sim.context, ibs_wrap, lc_win, lv_win, prefix=f"窗口{window_idx}_偏置预热后", sampler=sampler)
            raw_probe = sampler.get_raw_interaction_energies()
            if len(raw_probe) >= 2:
                raw_span = float(np.max(raw_probe) - np.min(raw_probe))
                if raw_span < 1e-6 and np.max(np.abs(raw_probe)) < 1e-6:
                    raise RuntimeError(
                        f"窗口 {window_idx} 的 IBS 软核 CV 在偏置预热后仍全部为 0。"
                        "已阻止进入生产采样，请继续检查 VDW CV 构造或状态注入。"
                    )

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
            preprod_force_threshold = 7000.0
            if fmax_relax > preprod_force_threshold:
                print("  🚨 [安全警报] 卸压后合力依然超标！触发应急深度最小化与局部松弛...")
                emergency_dt = 0.0005 * unit.picoseconds
                sim.integrator.setStepSize(emergency_dt)
                sim.context.setParameter(f"{self.prefix}_bias_scale", 0.0)
                sim.minimizeEnergy(maxIterations=5000, tolerance=10.0)
                for _ in range(20):
                    sim.step(500)
                for scale in np.linspace(0.05, 1.0, 20):
                    sim.context.setParameter(f"{self.prefix}_bias_scale", float(scale))
                    sim.step(500)
                fmax_relax = current_max_force_from_context(sim.context)
                state_relax = sim.context.getState(getForces=True, getPositions=True)
                print(f"  🩺 应急松弛后 max|F| = {fmax_relax:.1f} kJ/(mol·nm)")
                if fmax_relax > preprod_force_threshold:
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
                        f"窗口 {window_idx} 卸压后最大合力仍为 {fmax_relax:.1f} kJ/(mol·nm)，已阻止进入生产采样。"
                    )
                elif fmax_relax > 2500.0:
                    print(
                        f"  ⚠️ 卸压后最大合力 {fmax_relax:.1f} kJ/(mol·nm) 偏高但低于生产灾难阈值 "
                        f"{preprod_force_threshold:.0f}，允许进入生产采样并交由运行时回退监控。"
                    )
            else:
                print("  ✅ 卸压后合力通过阈值检查，允许进入生产采样。")

            sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
            sim.integrator.setStepSize(original_dt)
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
                if (up + 1) % 100 == 0:
                    self._enqueue_window_snapshot(window_idx, stage_type, sampler)

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
                    self._enqueue_window_snapshot(window_idx, stage_type, sampler)
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
            _atomic_save_npy(os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_energies.npy"), e_save)

            if sampler.bias_history:
                _atomic_save_npy(
                    os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_bias.npy"),
                    np.array(sampler.bias_history),
                )
            if sampler.base_energy_history:
                _atomic_save_npy(
                    os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_base.npy"),
                    np.array(sampler.base_energy_history),
                )
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
                'lambda_indices': list(range(start, end)),
                # 显式记录：局部 MBAR 的第 0 行是“采样分布”而非某个物理 lambda 态。
                'sampled_distribution_row': 0,
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
            
            # 样本计数：只有第 0 行（采样分布，而非物理 lambda 态）有 N 个样本。
            sampled_row = int(w.get("sampled_distribution_row", 0))
            n_k_local = np.zeros(len(win_lams) + 1, dtype=np.int32)
            if not (0 <= sampled_row < len(n_k_local)):
                print(f"  ⚠️ 窗口 {w_idx} sampled_distribution_row={sampled_row} 非法，回退为 0")
                sampled_row = 0
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
            
            if n_k_local[sampled_row] < 10:
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
                
                # Delta_f[0, k] 是第 k 个物理态相对于采样态 (Row 0) 的自由能差 (单位: kT)
                # 注意：res['Delta_f'] 的形状是 (K+1, K+1)
                # 我们想要的是物理态 (indices 1..K) 的结果
                f_phys_kt = df_matrix[0, 1:] 
                df_phys_kt = ddf_matrix[0, 1:]
                
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
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.contexts = []
        self.integrators = []
        self._state_to_context = list(range(self.n_replicas))
        self._context_to_state = list(range(self.n_replicas))
        self._steps_completed = 0
        self._is_warmed_up = False
        self._build_replicas(system_template)

    @staticmethod
    def _try_set_context_parameter(context, name: str, value: float) -> None:
        try:
            context.setParameter(name, float(value))
        except Exception as exc:
            msg = str(exc)
            if "invalid parameter name" not in msg:
                raise

    def _build_replicas(self, system_template):
        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)

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
            elif self.is_mixed_pme_alchemical:
                replica_sys = _prepare_pme_mixed_alchemical_system(
                    system_template,
                    self.ligand_indices,
                    topology=self.topology,
                    positions=self.positions,
                    box_vectors=self.box_vectors,
                    lambda_coul_name="lambda_coul",
                    lambda_vdw_name="lambda_vdw",
                )
            else:
                sys_xml = openmm.XmlSerializer.serialize(system_template)
                replica_sys = openmm.XmlSerializer.deserialize(sys_xml)
                replica_sys.thisown = 1
                nb = [f for f in replica_sys.getForces() if isinstance(f, openmm.NonbondedForce)][0]
                _restore_ligand_internal_nonbonded(replica_sys, nb, self.ligand_indices)
                env_idx = [j for j in range(replica_sys.getNumParticles()) if j not in self.ligand_indices]
                sc_force = BeutlerSoftcoreBuilder.build(nb, self.ligand_indices, env_idx)
                sc_force.setForceGroup(1)
                replica_sys.addForce(sc_force)

                for idx in self.ligand_indices:
                    nb.setParticleParameters(idx, 0.0, 0.1*unit.nanometer, 0.0)

            integ = openmm.LangevinMiddleIntegrator(self.temperature, 1.0/unit.picosecond, 0.002*unit.picosecond)
            ctx = openmm.Context(replica_sys, integ, platform, props)
            ctx.setPositions(self.positions)
            if self.box_vectors is not None:
                ctx.setPeriodicBoxVectors(*self.box_vectors)
            self._try_set_context_parameter(ctx, "lambda_coul", self.lambdas_coul[i])
            self._try_set_context_parameter(ctx, "lambda_vdw", self.lambdas_vdw[i])

            self.contexts.append(ctx)
            self.integrators.append(integ)

    def _set_context_state(self, context_idx: int, state_idx: int) -> None:
        ctx = self.contexts[context_idx]
        self._try_set_context_parameter(ctx, "lambda_coul", self.lambdas_coul[state_idx])
        self._try_set_context_parameter(ctx, "lambda_vdw", self.lambdas_vdw[state_idx])

    @staticmethod
    def _crossed_save_boundary(prev_step: int, next_step: int, save_interval: int) -> bool:
        if save_interval <= 0:
            return False
        return (prev_step // save_interval) != (next_step // save_interval)

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
            for ctx in self.contexts:
                ctx.getIntegrator().step(5000)
            self._is_warmed_up = True
            
        exchange_log = []
        
        for step in range(n_exchanges):
            # 1. 批量提交步进任务 (GPU 会在底层自动流水线并发，无需 Python 干预)
            prev_steps = self._steps_completed
            for ctx in self.contexts:
                ctx.getIntegrator().step(exchange_interval)
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
                
                # --- 阶段 C: Metropolis 判定 ---
                delta = self.beta * (U_i_j + U_j_i - U_i_i - U_j_j)
                accept = delta < 0 or np.random.rand() < np.exp(-delta)
                
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
            for ctx in self.contexts:
                ctx.getIntegrator().step(remaining_steps)
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
        topology: app.Topology = None,
        reference_positions=None,
        reference_box_vectors=None,
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
        if use_total_energy:
            nb_force_ref = next((f for f in system_template.getForces() if isinstance(f, openmm.NonbondedForce)), None)
            if nb_force_ref is None:
                raise RuntimeError("PME 去电荷路径未找到参考 NonbondedForce，无法估算自能修正。")
            ligand_net_charge = _compute_ligand_net_charge(system_template, ligand_indices)
            for idx in ligand_indices:
                q, _, _ = nb_force_ref.getParticleParameters(int(idx))
                q_val = q.value_in_unit(unit.elementary_charge)
                ligand_charge_square_sum += q_val * q_val
            if abs(ligand_net_charge) < 0.01:
                apply_pme_self_correction = True
            else:
                print(
                    "  ⚠️ 检测到带电配体的共炼金 PME 去电荷路径；"
                    "已禁用 ligand-only 解析自能补偿，避免引入数百 kJ/mol 的错误偏移。"
                )
        xyz_all = np.asarray(traj.xyz, dtype=np.float64)
        box_all = None
        if traj.unitcell_vectors is not None and len(traj.unitcell_vectors) > 0:
            box_all = np.asarray(traj.unitcell_vectors, dtype=np.float64)

        cpu_count = max(1, os.cpu_count() or 1)
        chunk_size = max(25, min(250, int(math.ceil(n_frames / max(1, cpu_count * 2)))))
        n_chunks = max(1, int(math.ceil(n_frames / chunk_size)))
        n_workers = min(cpu_count, n_chunks)
        if is_pme_coulomb_leg:
            prepared_system = _prepare_pme_coulomb_leg_system(
                system_template,
                ligand_indices,
                lambda_name="lambda_coul",
                allow_charged_ligand=True,
                topology=topology,
                positions=reference_positions,
                box_vectors=reference_box_vectors,
            )
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
            )
            system_xml = openmm.XmlSerializer.serialize(prepared_system)
            del prepared_system
        else:
            eval_sys = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system_template))
            eval_sys.thisown = 1
            nb = [f for f in eval_sys.getForces() if isinstance(f, openmm.NonbondedForce)][0]
            _restore_ligand_internal_nonbonded(eval_sys, nb, ligand_indices)
            env_idx = [i for i in range(eval_sys.getNumParticles()) if i not in ligand_indices]
            sc = BeutlerSoftcoreBuilder.build(nb, ligand_indices, env_idx)
            sc.setForceGroup(1)
            eval_sys.addForce(sc)
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

        return u_kn

    def solve(self, u_kn: np.ndarray) -> Dict:
        u_kn = np.asarray(u_kn, dtype=np.float64)
        K, N = u_kn.shape
        n_k = getattr(self, "_last_n_k", np.full(K, N // K, dtype=int))
        if len(n_k) != K or int(np.sum(n_k)) != N:
            raise ValueError(f"MBAR 样本数不匹配: len(n_k)={len(n_k)}, sum(n_k)={np.sum(n_k)}, N={N}")
        if not np.all(np.isfinite(u_kn)):
            raise ValueError("u_kn 含 NaN/Inf，无法执行 MBAR")
        if not HAS_PYMBAR:
            raise ImportError("需要 pymbar 包，请安装: pip install pymbar")

        # 逐列平移不会改变自由能差，但能显著改善大体系绝对能量下的数值条件。
        u_kn_stable = u_kn - np.min(u_kn, axis=0, keepdims=True)

        last_exc = None
        last_mbar = None
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
                    print(f"  ⚠️ MBAR 协方差求解失败 ({protocol}): {cov_exc}，回退为仅计算 ΔG 不估计误差")
                    res = _compute_free_energy_result_compatible(
                        last_mbar,
                        compute_uncertainty=False,
                    )
                    df_matrix, _ = _extract_free_energy_arrays(
                        res,
                        require_uncertainty=False,
                    )
                    err = float("nan")

                dg = float((df_matrix[0, -1] - df_matrix[0, 0]) * self.kt)
                return {
                    "delta_G": dg,
                    "error": err,
                    "method": method_name,
                    "n_frames": N,
                    "n_states": K,
                }
            except Exception as exc:
                last_exc = exc
                print(f"  ⚠️ MBAR {protocol} 求解失败: {exc}")

        raise RuntimeError(f"MBAR 求解失败，最后错误: {last_exc}")
