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
from scipy.integrate import quad as _scipy_quad
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
    subsample_series_by_autocorrelation,
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


def _atomic_write_json(filepath: str, payload: Dict[str, Any]) -> None:
    """同步原子写 JSON：先写临时文件，再原子替换。"""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    tmp_file = filepath + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_file, filepath)


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


def lambda_endpoint_diagnostics(lambdas_coul, lambdas_vdw, tol: float = 1e-8) -> Dict:
    """Check whether a lambda path has clear physical endpoints."""
    lc = np.asarray(lambdas_coul, dtype=float)
    lv = np.asarray(lambdas_vdw, dtype=float)
    if lc.size == 0 or lv.size == 0 or lc.size != lv.size:
        return {"ok": False, "reason": "empty_or_mismatched_lambda_arrays"}
    start = {"lambda_coul": float(lc[0]), "lambda_vdw": float(lv[0])}
    end = {"lambda_coul": float(lc[-1]), "lambda_vdw": float(lv[-1])}
    ok_start = abs(start["lambda_coul"] - 1.0) <= tol and abs(start["lambda_vdw"] - 1.0) <= tol
    ok_end = abs(end["lambda_coul"]) <= tol and abs(end["lambda_vdw"]) <= tol
    monotonic_coul = bool(np.all(np.diff(lc) <= tol))
    monotonic_vdw = bool(np.all(np.diff(lv) <= tol))
    return {
        "ok": bool(ok_start and ok_end and monotonic_coul and monotonic_vdw),
        "start": start,
        "end": end,
        "starts_fully_coupled": bool(ok_start),
        "ends_fully_decoupled": bool(ok_end),
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
            # ✅ 修复：nb_force.getPMEParameters()（不带 context 的静态查询）按 OpenMM
            # 惯例只有显式调用过 setPMEParameters() 才会非零，否则永远返回 (0,0,0,0)——
            # 真正的 alpha 是 OpenMM 建 Context 时才按 cutoff/ewaldErrorTolerance 自动
            # 算出来的，只能从 getPMEParametersInContext(ctx) 拿。之前这里一旦
            # getPMEParametersInContext 抛异常（例如某些平台上的兼容性问题），就静默
            # 摔进这个必然返回 0 的 fallback，导致 alpha_ewald=0、修正被整批跳过——
            # 而父进程早就打印过"✅ PME 自能修正启用"，子进程里的这次静默失败完全没有
            # 任何提示，产出的 u_kn 会带着未修正的自能伪项，看起来正常实则错误。现在
            # 拿不到有效 alpha 就直接报错，不再假装修正生效。
            alpha_source = "context"
            try:
                alpha_ewald, _, _, _ = nb_force.getPMEParametersInContext(ctx)
                alpha_ewald = _value_in_inverse_nanometer(alpha_ewald)
            except Exception as exc:
                alpha_source = f"static_fallback_after_context_query_failed({exc})"
                alpha_q, _, _, _ = nb_force.getPMEParameters()
                alpha_ewald = _value_in_inverse_nanometer(alpha_q)
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
def _lj_tail_correction_moments_kj_nm6_nm12(
    all_params,
    ligand_indices: List[int],
    environment_indices: List[int],
) -> Tuple[float, float]:
    """配体<->环境 LJ 长程尾项的两个几何求和矩：
    S6  = sum_{i in ligand} sum_{j in environment} epsilon_ij * sigma_ij^6   (色散/吸引项系数)
    S12 = sum_{i in ligand} sum_{j in environment} epsilon_ij * sigma_ij^12  (排斥项系数)
    混合规则与软核表达式一致 (sigma_ij=0.5*(sigma_i+sigma_j), epsilon_ij=sqrt(epsilon_i*epsilon_j))。
    两者都是固定的几何量，跟 lambda、switching、盒子体积无关（那些在调用处的径向积分/
    每帧体积换算里另外处理）。
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
    sigma_ij = 0.5 * (sigma_lig[:, None] + sigma_env[None, :])
    eps_ij = np.sqrt(eps_lig[:, None] * eps_env[None, :])
    s6 = float(np.sum(eps_ij * sigma_ij ** 6))
    s12 = float(np.sum(eps_ij * sigma_ij ** 12))
    return s6, s12


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
TRADITIONAL_LJ_LRC_PROTOCOL_VERSION = 2

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


def _lj_softcore_tail_radial_integrals(
    lambda_vdw: float,
    alpha_lj: float,
    m_lj: float,
    r_switch_nm: float = LJ_TAIL_LRC_R_SWITCH_NM,
    r_cutoff_nm: float = LJ_TAIL_LRC_R_CUTOFF_NM,
) -> Tuple[float, float]:
    """Real switching-aware + softcore-aware radial integrals for one lambda_vdw:

        I6  = integral_{r_switch}^{r_cutoff} (1-S(r)) r^2 / D(r)   dr
            + integral_{r_cutoff}^{infinity}            r^2 / D(r)   dr
        I12 = integral_{r_switch}^{r_cutoff} (1-S(r)) r^2 / D(r)^2 dr
            + integral_{r_cutoff}^{infinity}            r^2 / D(r)^2 dr

    where D(r) = alpha_lj*(1-lambda_vdw)^m_lj + r^6 is the exact softcore
    denominator used by both AlchemicalPotentialFactory/ACESoftcorePotential
    and BeutlerSoftcoreBuilder (BeutlerSoftcoreBuilder additionally adds a
    +1e-4*(1-lambda_vdw) numerical safety floor near r=0; at r>=r_switch=1.0nm
    that term is <=1e-4 against a D(r)>=1 from the r^6 piece alone, i.e.
    completely negligible for this tail integral, so both paths can safely
    share this one D(r)).

    No closed form exists once the switching polynomial and the
    lambda-shifted r^6 denominator are both present, so this integrates
    numerically (scipy.integrate.quad, handles the improper r_cutoff->inf
    integral directly). Called once per distinct lambda_vdw at system-build
    time, never per-frame, so the extra cost is negligible.
    """
    lam = float(lambda_vdw)
    alpha = float(alpha_lj)
    m = float(m_lj)

    def _softcore_D(r):
        return alpha * (1.0 - lam) ** m + r ** 6

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
    s6_kj_nm6: float,
    s12_kj_nm12: float,
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

        coeff[k] = 16*pi * lambda_vdw[k]**n_lj * (S12*I12(lambda_vdw[k]) - S6*I6(lambda_vdw[k]))

    At lambda_vdw=0, lambda_vdw**n_lj is exactly 0.0 in floating point (not
    just close to it) for any n_lj>0, so coeff is exactly 0 regardless of
    I6/I12 -- the integral is skipped entirely for that state as a cheap
    optimization, not because it would otherwise be wrong.
    """
    lambdas_arr = np.asarray(lambdas_vdw, dtype=np.float64)
    coeffs = np.zeros(lambdas_arr.shape[0], dtype=np.float64)
    for k, lam in enumerate(lambdas_arr):
        lam = float(lam)
        if lam == 0.0:
            continue
        i6, i12 = _lj_softcore_tail_radial_integrals(lam, alpha_lj, m_lj, r_switch_nm, r_cutoff_nm)
        coeffs[k] = 16.0 * math.pi * (lam ** float(n_lj)) * (s12_kj_nm12 * i12 - s6_kj_nm6 * i6)
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
    sc_force.setSwitchingDistance(1.0 * unit.nanometer)
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
    if potential_type == "dexp":
        print("  ⚠️ [LJ LRC] potential_type='dexp' 尚未验证解析尾项公式是否适用于该势函数，本次不附加修正。")
    else:
        rc_nm = template_cutoff.value_in_unit(unit.nanometer)
        rs_nm = template_switch.value_in_unit(unit.nanometer)
        tail_s6, tail_s12 = _lj_tail_correction_moments_kj_nm6_nm12(all_params, perturbed_indices, env_indices)
        n_lj_exp = float(getattr(alchemical_params, "n_lj", 2))
        alpha_lj = float(getattr(alchemical_params, "alpha_lj", 0.5))
        m_lj = float(getattr(alchemical_params, "m_lj", 2))
        ibs_wrapper.lj_tail_lrc_coeff_kj_mol = _lj_tail_lrc_coefficients_kj_mol(
            lambdas_vdw, tail_s6, tail_s12, alpha_lj, m_lj, n_lj_exp, rs_nm, rc_nm,
        )
        print(
            f"  🧮 [LJ LRC v{TRADITIONAL_LJ_LRC_PROTOCOL_VERSION}] switching+softcore-aware 解析长程尾项已启用："
            f"S6={tail_s6:.4g} kJ·nm^6/mol, S12={tail_s12:.4g} kJ·nm^12/mol, "
            f"switch={rs_nm:.3f} nm, cutoff={rc_nm:.3f} nm, alpha_lj={alpha_lj:.4g}, m_lj={m_lj:.1f}, "
            f"n_lj={n_lj_exp:.1f}；每帧修正 = lrc_coeff[k] / V(t)，lrc_coeff 逐 λ 数值积分得出。"
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
IBS_BIAS_PROTOCOL_VERSION = 13

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


class IBSWarmupConvergenceError(RuntimeError):
    """Structured signal that a window lacks adequate sampled-state coverage."""

    def __init__(self, message: str, diagnostics: Dict):
        super().__init__(message)
        self.diagnostics = diagnostics


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
        
        # 🔑 新增：EMA 累积统计量（仅供诊断趋势观察；update_weights() 的梯度已改
        # 用当次真实 mean_p_batch，不再读这个跨 Hamiltonian 的滞后平均量）。
        self.ema_mean_p = None
        self.gamma = 0.9  # 衰减因子，保留 90% 历史信息
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
        
        # 🔑 新增：能量偏移缓存
        self.e_offset = 0.0
        # 连续 Base 能量读取失败计数——单帧失败允许跳过（很可能是瞬时 getState
        # 异常），但连续多帧失败说明 Context/系统本身有问题，必须硬报错，而不是
        # 一直用假 0.0 悄悄污染 base_energy_history。
        self._consecutive_base_failures = 0
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

    def collect_energies(self) -> np.ndarray:
        energies = np.zeros(self.n_states)
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
        _MAX_CONSECUTIVE_BASE_FAILURES = 5
        try:
            state_base = self.context.getState(getEnergy=True, groups={0, 2, 3, 5})
            e_base = state_base.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            self._consecutive_base_failures = 0
        except Exception as e:
            # ⚠️ 不能静默吞掉：e_base 会被直接写入 base_energy_history 并喂给
            # GlobalMBARAnalyzer (u_phys_kj = base_kj + u_kj_raw)。之前失败时回退
            # 成 0.0 并"标记"——但下面只检查 interaction_energies 是否有 NaN，e_base
            # 本身从未被检查，假 0.0 会照常被 append 进 base_energy_history，
            # 悄悄污染 MBAR 且无迹可查。现在改为 NaN（下面统一按"任一分量非有限
            # 就整帧跳过"处理，不单独放行 e_base），单帧失败允许跳过；但连续失败
            # 说明 Context/系统本身有问题（不是偶发 getState 抖动），必须硬报错，
            # 而不是无限跳帧、悄悄丢数据。
            self._consecutive_base_failures += 1
            print(
                f"  🚨 Base 能量 (Group 0,2,3,5) 获取失败（连续第 "
                f"{self._consecutive_base_failures} 次），本帧标记为 NaN 并跳过：{e}"
            )
            e_base = float("nan")
            if self._consecutive_base_failures >= _MAX_CONSECUTIVE_BASE_FAILURES:
                raise RuntimeError(
                    f"Base 能量 (Group 0,2,3,5) 连续 {self._consecutive_base_failures} 帧读取失败，"
                    "拒绝继续用 NaN/假值悄悄跳帧——这通常说明 Context 或系统本身已经损坏，"
                    "需要人工检查，而不是让采样继续假装正常。"
                ) from e

        if self.ibs_wrapper is None:
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
        except Exception as e:
            print(f"  ⚠️ CV 探针能量提取失败: {e}")
            energies[:] = np.nan
        return energies

    def update_weights(self, min_buffer_size: int = 20) -> Optional[np.ndarray]:
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
            "mean_u_kJ_mol": np.mean(u_mk, axis=0).astype(float).tolist(),
            "adjacent_delta_u": adjacent_records,
        }
        
        # 4. EMA 更新（仅供诊断趋势观察，下面的梯度不再读它——见第 5 步注释）。
        if self.ema_mean_p is None:
            self.ema_mean_p = mean_p_batch.copy()
        else:
            self.ema_mean_p = self.gamma * self.ema_mean_p + (1.0 - self.gamma) * mean_p_batch

        # 5. 对数梯度更新
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=8] 之前这里读 self.ema_mean_p（gamma=0.9，
        # 保留 90% 历史）而不是这次真实采出的 mean_p_batch——但 f_old 是"这次"的
        # 权重，EMA 里混着若干次更新之前、不同 f_k 下测出的旧概率，梯度因此永远
        # 落后于当前真正的 f_old，导致更新方向/幅度和实际需要的修正量不一致，
        # 容易出现"推过头 -> 冻结验证失败 -> 反向推过头"的振荡（真实案例：50 次
        # 更新后 3 态窗口仍塌缩到 [0.12,0.0017,0.88]，但同一组 λ 的 fixed-H 双向
        # overlap 全部通过，说明问题出在这个求解器而不是 λ 密度）。梯度必须用
        # mean_p_batch（在当前 f_old 下、这一批新采样的真实概率），EMA 只做趋势
        # 参考，不进入任何决策路径。
        target_p = 1.0 / K
        log_grad = np.log(mean_p_batch + 1e-30) - np.log(target_p)
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

        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=8] 衰减时标从 100 缩短到 30（第 50 次更新
        # 时 eta 从旧的 0.667 降到约 0.158），并乘上 self.eta_penalty——每次冻结
        # 验证失败恢复 learning 都会把它减半（见 apply_learning_rate_penalty），
        # 让重新学习一次比一次保守，而不是拿几乎相同的大步长继续来回振荡。
        t = len(self.f_history) + 1
        eta_sgd = (1.0 / (1.0 + t / 30.0)) * self.eta_penalty

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

    def apply_learning_rate_penalty(self, factor: float = 0.5, floor: float = 0.05) -> float:
        """Halve (default) the learning-rate multiplier after a frozen-validation
        failure, so the next learning attempt takes more conservative steps
        instead of oscillating with the same large step size that just failed.
        Floored to avoid learning stalling out entirely after repeated failures.
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
            "e_offset": self.e_offset,
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
            "lambdas_coul": [float(x) for x in lambdas_coul] if lambdas_coul is not None else None,
            "lambdas_vdw": [float(x) for x in lambdas_vdw] if lambdas_vdw is not None else None,
        }
        _atomic_write_json(filepath, state)

    def load_ibs_state(self, filepath: str, lambdas_coul=None, lambdas_vdw=None) -> bool:
        """反序列化并注入 IBS 状态，恢复历史记忆。

        🔑 [ibs_bias_protocol_version=4] n_states/prefix/协议版本/λ 内容
        （lambdas_coul/lambdas_vdw）任何一项对不上，一律完全拒绝这份旧状态——
        不再有"协议对不上但 f_k 仍可以当热启动"的中间态。旧协议下的 f_k 是在
        一个可能已经不存在的 λ 网格上学出来的，把它注入一个不同的 λ 网格不是
        "起点差一点"，是把状态 k 的旧 f_k 错配到状态 k 的新 λ 上，主动引入偏差，
        不是中性的。只有 n_states/prefix/协议版本/λ 内容全部严格一致时才真正
        采用（含 bias_converged 标记），否则整体忽略，从 f_k=0 全新开始。
        """
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
            cached_protocol_version = state.get("ibs_bias_protocol_version")
            if cached_protocol_version != IBS_BIAS_PROTOCOL_VERSION:
                print(
                    f"  ⚠️ IBS 状态协议版本不匹配 (cache={cached_protocol_version!r}, "
                    f"当前={IBS_BIAS_PROTOCOL_VERSION})，完全忽略旧状态（不作为热启动），"
                    "从 f_k=0 重新开始"
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

            self.e_offset = state.get("e_offset", 0.0)
            self.bias_converged = bool(state.get("bias_converged", False))

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

            # 伪造历史长度以恢复 SGD 学习率衰减 (eta_sgd)
            self.f_history = [np.array(f_k)] * t
            print(
                f"  ♻️ IBS 状态已恢复: t={t}, max|f_k|={np.max(np.abs(f_k)):.2f} kJ/mol, "
                f"bias_converged={self.bias_converged}, bias_status={self.bias_status}"
            )
            return True
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
    normalized_expected = json.loads(json.dumps(expected_manifest, sort_keys=True, default=str))
    return loaded == normalized_expected


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
) -> Dict[str, Any]:
    """主窗口 checkpoint 的内容指纹——任何一项不匹配都必须整体拒绝这份
    checkpoint（λ 网格被自动加密/重新划分窗口、协议版本变化、平台不同等），
    直接沿用探针轨迹库 `_build_fixed_h_probe_bank_manifest` 的字段/哈希写法。
    """
    return {
        "main_window_checkpoint_protocol_version": MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
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
    normalized_expected = json.loads(json.dumps(expected_manifest, sort_keys=True, default=str))
    return loaded == normalized_expected


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
    normalized_expected = json.loads(json.dumps(expected_manifest, sort_keys=True, default=str))
    return loaded == normalized_expected


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
    ):
        """
        执行全部窗口采样，能量保存至 {output_dir}/dual_window_{window_idx}_{stage_type}_energies.npy

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

            # ---------- 断点续传：按窗口跳过已完成的采样 ----------
            # 此前 resume 只热启动了 IBS bias 权重 (f_k)，中途崩溃后重跑会把
            # 已经跑完的窗口从头再来一遍，白白浪费 GPU 时间且违背"Checkpoint"的
            # 设计承诺。这里在真正重建系统/最小化/生产采样之前，先检查该窗口
            # 是否已有形状匹配的有效能量文件，若有则直接跳过整窗。
            if resume:
                energies_path = os.path.join(
                    self.output_dir, f"dual_window_{window_idx}_{stage_type}_energies.npy"
                )
                convergence_path = os.path.join(
                    self.output_dir, f"dual_window_{window_idx}_{stage_type}_convergence.json"
                )
                if os.path.exists(energies_path) and os.path.exists(convergence_path):
                    try:
                        cached_e = np.load(energies_path)
                        with open(convergence_path, "r", encoding="utf-8") as f:
                            cached_conv = json.load(f)
                        # 🔑 只按 window_idx + shape 判断"这份缓存还能不能用"曾经是不安全的：
                        # λ 路径被自动加密/重新划分窗口后，同一个 window_idx 完全可能对应
                        # 一段全新的 λ 区间，只要凑巧态数相同（shape 相同）就会被静默当成
                        # 有效缓存复用——采样对了，但对的是错的 λ。这里额外校验 convergence.json
                        # 里存的实际 λ 值（见下方保存逻辑）跟本次真正要跑的 lc_win/lv_win 是否
                        # 完全一致；旧格式缓存没有这个字段的，保守地当作不匹配，强制重采，而不是
                        # 假设它"大概率没变"。
                        cached_lc = cached_conv.get("lambdas_coul")
                        cached_lv = cached_conv.get("lambdas_vdw")
                        lambdas_match = (
                            cached_lc is not None
                            and cached_lv is not None
                            and len(cached_lc) == len(lc_win)
                            and len(cached_lv) == len(lv_win)
                            and np.allclose(cached_lc, lc_win, atol=1e-9)
                            and np.allclose(cached_lv, lv_win, atol=1e-9)
                        )
                        # 🔑 [wca_accounting_version] λ 值对得上不代表数值对得上：base/bias
                        # 的力组切分口径变过（Group4 λ-WCA 从"算作 base"改成"算作 bias"，
                        # 见 WCA_ACCOUNTING_VERSION），旧版本文件里的 base/bias 数值是在
                        # 旧口径下算出来的，即使 λ 完全相同也绝不能当成同一份数据复用。
                        version_match = cached_conv.get("wca_accounting_version") == WCA_ACCOUNTING_VERSION
                        # 🔑 [ibs_bias_protocol_version] 旧协议下这份能量文件可能是在
                        # f_k 全程漂移（生产阶段仍在 update_weights()）的情况下采出来的，
                        # 不满足 MBAR 的单一固定采样分布假设——即使 λ/WCA 口径都对得上，
                        # 也不能当成有效缓存复用。
                        bias_protocol_match = cached_conv.get("ibs_bias_protocol_version") == IBS_BIAS_PROTOCOL_VERSION
                        # 🔑 [early_stop / 步数预算] 两条独立检查，不管这份缓存是否
                        # 触发过 early stop 都要过第一条：
                        #  (1) 缓存产出时的目标步数（n_steps_per_window_effective）
                        #      不能低于当前调用的目标步数——否则即使从来没有触发
                        #      过 early stop、是"完整跑满"的缓存，也可能只是"250k
                        #      时代"跑完的窗口，被静默当成满足"预算已提到 500k"的
                        #      当前要求复用，实际样本量根本不够。
                        #  (2) 如果这份缓存确实是 early_stop_triggered=True 提前
                        #      停止产出的短样本，还要求当前调用同样启用 early
                        #      stop、协议版本一致、且这次实际使用的八个阈值
                        #      （early_stop_config）跟缓存记录的完全一致——只比协议
                        #      版本不够：版本号只在判据逻辑本身变了才会变，把某个
                        #      阈值从松调紧（例如 max_uncertainty_kJ_mol 从 5.0
                        #      收紧到 1.0）根本不影响协议版本，但旧窗口是在更松的
                        #      阈值下通过的，不能假设它在新阈值下也一定通过。
                        # 任何一项不满足都视为缓存不可信，强制重新采样该窗口。
                        early_stop_ok = True
                        early_stop_reject_reason = None
                        _effective_target_for_resume_check = (
                            int(production_step_overrides[window_idx])
                            if production_step_overrides and window_idx in production_step_overrides
                            else int(n_steps_per_window)
                        )
                        _cached_target = cached_conv.get("n_steps_per_window_effective")
                        if _cached_target is None or int(_cached_target) < _effective_target_for_resume_check:
                            early_stop_ok = False
                            early_stop_reject_reason = (
                                f"当前目标步数（{_effective_target_for_resume_check}）高于缓存产出时的目标"
                                f"（{_cached_target}）"
                            )
                        elif bool(cached_conv.get("early_stop_triggered", False)):
                            if not enable_early_stop:
                                early_stop_ok = False
                                early_stop_reject_reason = "当前调用未启用 early stop"
                            elif cached_conv.get("early_stop_protocol_version") != EARLY_STOP_PROTOCOL_VERSION:
                                early_stop_ok = False
                                early_stop_reject_reason = (
                                    f"缓存的 early_stop_protocol_version="
                                    f"{cached_conv.get('early_stop_protocol_version')!r}（期望 "
                                    f"{EARLY_STOP_PROTOCOL_VERSION}）"
                                )
                            else:
                                _current_early_stop_config = {
                                    "min_steps": int(early_stop_min_steps),
                                    "check_interval_steps": int(early_stop_check_interval_steps),
                                    "required_consecutive_passes": int(early_stop_required_consecutive_passes),
                                    "min_ess_ratio": float(early_stop_min_ess_ratio),
                                    "min_absolute_ess": float(early_stop_min_absolute_ess),
                                    "min_decorrelated_samples": int(early_stop_min_decorrelated_samples),
                                    "max_delta_g_drift_kJ_mol": float(early_stop_max_delta_g_drift_kJ_mol),
                                    "max_uncertainty_kJ_mol": float(early_stop_max_uncertainty_kJ_mol),
                                }
                                if not _early_stop_configs_match(
                                    cached_conv.get("early_stop_config"), _current_early_stop_config
                                ):
                                    early_stop_ok = False
                                    early_stop_reject_reason = (
                                        f"缓存的 early_stop_config（{cached_conv.get('early_stop_config')}）"
                                        f"与当前调用（{_current_early_stop_config}）不一致"
                                    )
                        # 🔑 [lj_tail_lrc_protocol_version] LRC 公式版本必须匹配——
                        # v1 只补 cutoff 外的 r^-6、忽略 switching 区间，v2 是真正
                        # switching+softcore-aware 的积分；旧协议下算出的
                        # target_energies 里叠加的 LRC 数值跟当前公式完全不是同一
                        # 回事，即使 λ/WCA/IBS 偏置协议都对得上，也不能当成同一份
                        # 数据复用。缺失该字段（比 v1 更旧、这个字段还不存在的
                        # 缓存）同样保守地判不匹配。
                        lrc_version_match = cached_conv.get("lj_tail_lrc_protocol_version") == TRADITIONAL_LJ_LRC_PROTOCOL_VERSION
                        if (
                            cached_e.ndim == 2
                            and cached_e.shape[0] == len(lc_win)
                            and cached_e.shape[1] > 0
                            and lambdas_match
                            and version_match
                            and bias_protocol_match
                            and lrc_version_match
                            and early_stop_ok
                        ):
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
                                f"{cached_conv.get('ibs_bias_protocol_version')!r}（期望 {IBS_BIAS_PROTOCOL_VERSION}），"
                                "IBS 偏置预热/冻结协议已变更，视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0 and not lrc_version_match:
                            print(
                                f"  ⚠️ 窗口 {window_idx} 缓存的 lj_tail_lrc_protocol_version="
                                f"{cached_conv.get('lj_tail_lrc_protocol_version')!r}（期望 "
                                f"{TRADITIONAL_LJ_LRC_PROTOCOL_VERSION}），LJ 长程尾项修正公式已变更"
                                "（switching-aware），视为无效缓存，将重新采样该窗口。"
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
            effective_frozen_validation_budget = int(
                (frozen_validation_step_overrides or {}).get(window_idx, mbar_calibration_reserved_steps)
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
            if not is_resumed_ibs:
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
                        sampler.collect_energies()
                        # 只在偏置力足够大时才更新 f_k，避免弱偏置下噪声主导
                        if target_scale >= 0.5 and len(sampler.energy_buffer) >= 20:
                            sampler.update_weights()

                    state = sim.context.getState(getEnergy=True)
                    if not np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
                        raise RuntimeError(f"偏置预热阶段在 scale={target_scale} 时能量 NaN")
                    print(" 完成")

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
            #      在真正固定的 Hamiltonian 下连续采若干个新 batch → 只有这些新
            #      证据也连续通过同样门槛，才真正宣布 converged。判据必须用该方法
            #      返回的原始 batch 概率，不能用 self.ema_mean_p——gamma=0.9 的 EMA
            #      对单个塌缩 batch 反应迟钝（例如两态均匀 batch 后紧跟一个完全
            #      塌缩到 [1,0,0] 的 batch，EMA=0.9*[1/3,1/3,1/3]+0.1*[1,0,0]=
            #      [0.40,0.30,0.30] 仍能轻松通过 K=3 的门槛），拿 EMA 当"独立 batch
            #      连续通过"的证据，等于把假收敛的 bug 原样搬进了验证阶段本身。
            #   3) 冻结验证失败：候选收敛是假的，恢复 [learning]（重新允许
            #      update_weights() 调整 f_k），在同一个步数预算内继续尝试。
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
            check_chunk = 500
            f_stability_threshold_kJ_mol = 0.05
            frozen_burn_in_steps = 5000
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
            sgd_step_budget = max(
                int(max_bias_warmup_steps) - int(mbar_calibration_reserved_steps),
                int(frozen_burn_in_steps) + check_chunk,
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
            steps_at_full_bias = 0
            bias_update_count = 0
            bias_converged = False
            consecutive_pass_count = 0
            validation_pass_count = 0
            learning_to_validation_cycles = 0
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12 修复，未升版本号——见下方修复处注释]
            # 仅用于诊断的计数器：resumed_calibration_pending 续验分支里每次单批
            # 验证失败都会 +1，但绝不触发退回 learning。跟 learning_to_validation_cycles
            # 分开计数，因为后者的语义是"退回 learning 的次数"，续验分支永远不退回
            # learning，必须让它保持 0，否则会污染下面失败打印/诊断里"是否发生过
            # 假收敛重学习"的含义。
            frozen_validation_retry_count = 0
            last_f_delta = float("nan")
            freeze_burn_in_done = 0
            validation_batch_history = []
            last_validation_batch_p = None
            early_probe_triggered = False
            early_probe_trigger_reason = None
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
            else:
                mode = "learning"
                frozen_f_k_snapshot = None

            while steps_at_full_bias < sgd_step_budget:
                if mode == "learning" and bias_update_count >= int(max_bias_updates):
                    break
                # 🔑 [IBS_EARLY_PROBE_TRIGGER_ENABLED] learning_to_validation_cycles
                # 是单调递增计数器（只在冻结验证失败时 +1），一旦到 2 就永远 >=2；
                # 必须只在 mode=="learning" 时检查，否则会在 freeze_burn_in/
                # validating 即将真正通过的最后几百步把它打断，反而触发更贵的
                # fixed-H overlap 探针+bias 校准兜底。跟下面的轨迹库重构是两件
                # 独立的事，用同名开关分别验证（见常量定义处）。
                if (
                    IBS_EARLY_PROBE_TRIGGER_ENABLED
                    and mode == "learning"
                    and learning_to_validation_cycles >= 2
                    and steps_at_full_bias >= 100_000
                    and K <= 4
                    and stage_type == "vdw"
                ):
                    # 🔑 记下触发原因，不能让下游只看到一个笼统的
                    # "hit_update_or_safety_cap_unconverged"——那个状态字符串本来
                    # 描述的是"真正烧完了步数预算"，跟"主动提前放弃 SGD、抢跑
                    # fixed-H 探针"是两种不同的情况，必须能从落盘诊断里区分开。
                    early_probe_triggered = True
                    early_probe_trigger_reason = (
                        f"learning_to_validation_cycles={learning_to_validation_cycles}>=2 "
                        f"and steps_at_full_bias={steps_at_full_bias}>=100000 "
                        f"and K={K}<=4 and stage_type=='vdw'"
                    )
                    break
                sim.step(check_chunk)
                steps_at_full_bias += check_chunk
                sampler.collect_energies()

                if mode == "learning":
                    # check_chunk（每 500 步检查一次）比 update_weights() 真正
                    # 触发的频率（energy_buffer 攒够 10 帧才执行）密得多；没有
                    # 新更新的检查直接跳过，避免同一批结果被反复计成"连续通过"。
                    f_updated = None
                    if len(sampler.energy_buffer) >= 20:
                        f_updated = sampler.update_weights()
                    if f_updated is None:
                        continue
                    bias_update_count += 1
                    if len(sampler.f_history) < 2:
                        continue
                    if bias_update_count < int(min_bias_updates):
                        continue

                    p = np.asarray(sampler.ema_mean_p, dtype=np.float64)
                    min_p_ok = bool(np.all(p > min_probability_threshold))
                    coverage_ess = float(1.0 / np.sum(np.square(p)))
                    coverage_ok = bool(coverage_ess > coverage_ess_threshold)
                    last_f_delta = float(
                        np.max(np.abs(np.asarray(sampler.f_history[-1]) - np.asarray(sampler.f_history[-2])))
                    )
                    if min_p_ok and coverage_ok:
                        consecutive_pass_count += 1
                    else:
                        consecutive_pass_count = 0
                    if consecutive_pass_count >= int(required_consecutive_bias_updates):
                        # 候选收敛：冻结 f_k，切到 burn-in，绝不再更新权重。
                        frozen_f_k_snapshot = [
                            float(sim.context.getParameter(f"{self.prefix}_f_{k}"))
                            for k in range(K)
                        ]
                        mode = "freeze_burn_in"
                        freeze_burn_in_done = 0
                        sampler.energy_buffer = []
                        sampler.ema_mean_p = None
                    continue

                if mode == "freeze_burn_in":
                    # 冻结后的样本需要重新平衡到这个固定 Hamiltonian 下的分布，
                    # 之前在漂移偏置下攒的帧不能算数——整批丢弃，不进入验证统计。
                    freeze_burn_in_done += check_chunk
                    sampler.energy_buffer = []
                    if freeze_burn_in_done >= frozen_burn_in_steps:
                        mode = "validating"
                        validation_pass_count = 0
                    continue

                # mode == "validating"：只读评估，绝不调用 update_weights()。判据必须
                # 用这次调用的原始返回值，不能读 self.ema_mean_p——见函数上方注释。
                if len(sampler.energy_buffer) < 20:
                    continue
                batch_p = sampler.evaluate_frozen_batch_probability()
                if batch_p is None:
                    continue
                p = np.asarray(batch_p, dtype=np.float64)
                last_validation_batch_p = p.tolist()
                validation_batch_history.append(last_validation_batch_p)
                if resumed_calibration_pending:
                    # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 每评估完一批
                    # （无论 pass/fail）就存一次主窗口 checkpoint，覆盖式落盘
                    # （不按代际累积，续验永远只需要"最后一次看到的状态"）。
                    # 这是应对 HPC 作业被抢占/撞墙时限杀掉的直接对策：不这样做
                    # 的话，一次 300,000 步续验预算里的全部进展会在作业被杀时
                    # 完全丢失，下次只能从头再续验一遍。
                    _atomic_save_openmm_checkpoint(sim, main_ckpt_path)
                    _atomic_write_json(main_manifest_path, expected_main_manifest)
                    # 🔑 [checkpoint/累计步数同步修复] 上面的 .chk 每批都覆盖式落盘，
                    # 但 frozen_validation_cumulative_steps 在这个修复之前只在整次
                    # attempt 正常结束时才写回 JSON（下面 bias_converged 分支或
                    # 失败分支）——一旦作业在这两点之间被杀，.chk 里的坐标已经
                    # 反映了新跑的步数，JSON 里的累计步数却还是这次 attempt开始前
                    # 的旧值，下次续算会用这个偏低的旧值重新计算剩余预算，导致
                    # 已经真正跑过的步数被重复计入下一轮预算。这里让两者同频率
                    # 落盘，跟 .chk 一样每批覆盖式更新一次，任何一批之后被杀都能
                    # 精确续算，不会重跑或漏算已完成的步数。
                    sampler.frozen_validation_cumulative_steps = (
                        prior_cumulative_steps + steps_at_full_bias
                    )
                    sampler.save_ibs_state(ibs_state_file, lc_win, lv_win)
                min_p_ok = bool(np.all(p > min_probability_threshold))
                coverage_ess = float(1.0 / np.sum(np.square(p)))
                coverage_ok = bool(coverage_ess > coverage_ess_threshold)
                if min_p_ok and coverage_ok:
                    validation_pass_count += 1
                else:
                    validation_pass_count = 0
                if validation_pass_count >= int(required_consecutive_bias_updates):
                    bias_converged = True
                    break
                if validation_pass_count == 0:
                    if resumed_calibration_pending:
                        # 🔑 [修复：resume 续验冻结校准 f_k 时验证失败绝不能退回
                        # learning] 这份 f_k 已经用 fixed-H overlap 探针 + bias
                        # 校准探针（真实物理量，不是猜测）证明过是对的；从
                        # resumed_calibration_pending 分支进来的整段循环里，
                        # Hamiltonian（即这份冻结 f_k）从未被 update_weights()
                        # 触碰、也没有理由被触碰。单批验证没通过只说明还需要
                        # 更多独立样本，不说明这份 f_k 本身不稳定。不重新
                        # burn-in：burn-in 存在的唯一目的是让样本"忘记"切换
                        # Hamiltonian 前的漂移历史，这里 Hamiltonian 全程不变，
                        # 没有需要忘记的东西，直接继续在 mode="validating" 下采
                        # 下一批全新样本即可（validation_pass_count 已在上面
                        # 重置为 0）。用独立计数器
                        # frozen_validation_retry_count 记录重试次数，不能用
                        # learning_to_validation_cycles——那个计数器的语义是
                        # "退回 learning 的次数"，这条路径永远不退回 learning，
                        # 必须保持 0。
                        frozen_validation_retry_count += 1
                        print(
                            f"    🧊 冻结验证第 {frozen_validation_retry_count} 次批次未通过，"
                            "保持同一组已校准 f_k 不变，继续只读验证"
                            "（续验模式下绝不回退 learning）。"
                        )
                    else:
                        # 冻结验证失败：候选收敛是假的，恢复权重学习，不能拿同一个
                        # 已经证明不稳的 f_k 反复重测；EMA 也一并清空，避免把冻结
                        # 验证阶段（不同 Hamiltonian）的统计量混进下一轮 learning。
                        # 学习率也在这里减半（见 apply_learning_rate_penalty），避免
                        # 重新学习时用几乎相同的大步长继续来回振荡。
                        learning_to_validation_cycles += 1
                        new_eta_penalty = sampler.apply_learning_rate_penalty()
                        print(
                            f"    ⚠️ 冻结验证第 {learning_to_validation_cycles} 次失败，恢复 learning，"
                            f"学习率乘子降至 {new_eta_penalty:.4f}"
                        )
                        mode = "learning"
                        consecutive_pass_count = 0
                        sampler.ema_mean_p = None
                        sampler.energy_buffer = []

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
            bias_warmup_diag = {
                "status": (
                    "frozen_validation_converged"
                    if bias_converged
                    else (
                        "early_probe_triggered_unconverged"
                        if early_probe_triggered
                        else "hit_update_or_safety_cap_unconverged"
                    )
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
                "last_f_k_delta_kJ_mol": last_f_delta,
                "f_stability_threshold_kJ_mol": float(f_stability_threshold_kJ_mol),
                "min_bias_updates": int(min_bias_updates),
                "max_bias_updates": int(max_bias_updates),
                "bias_update_count": int(bias_update_count),
                "required_consecutive_passes": int(required_consecutive_bias_updates),
                "consecutive_pass_count": int(consecutive_pass_count),
                "validation_pass_count": int(validation_pass_count),
                "validation_batch_probabilities": validation_batch_history,
                "resumed_from_cache": bool(resumed_frozen_f_k is not None),
                "learning_to_validation_cycles": int(learning_to_validation_cycles),
                "resumed_calibration_pending": bool(resumed_calibration_pending),
                "frozen_validation_retry_count": int(frozen_validation_retry_count),
                "frozen_burn_in_steps": int(frozen_burn_in_steps),
                "frozen_f_k_at_last_freeze": frozen_f_k_snapshot,
                "final_mode": mode,
                "steps_at_full_bias": int(steps_at_full_bias),
                "max_bias_warmup_steps_safety_cap": int(max_bias_warmup_steps),
                "sgd_step_budget": int(sgd_step_budget),
                "mbar_calibration_reserved_steps": int(mbar_calibration_reserved_steps),
                "was_resumed": bool(is_resumed_ibs),
                "f_delta_is_diagnostic_only": True,
                "feedback_action": (
                    # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=7] split_window_from_warmup_failure
                    # 要求两个孩子各自 >=3 态、共享 1 态，父窗口因此至少要 5 态
                    # 才能这样拆（3+3-1）；K=4 现在和 K<=3 一样直接走 fixed-H
                    # overlap 探针，不再被拆成 K=2+K=3 这种统计脆弱的窗口。
                    "split_window_without_lambda_insertion"
                    if K >= 5
                    else "probe_fixed_hamiltonian_bidirectional_overlap"
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
            if not bias_converged and not resumed_calibration_pending and K <= 4 and stage_type == "vdw":
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
            if not bias_converged and not resumed_calibration_pending and overlap_probe.get("all_passed"):
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
                        batch_p = sampler.evaluate_frozen_batch_probability()
                        if batch_p is None:
                            continue
                        p = np.asarray(batch_p, dtype=np.float64)
                        last_validation_batch_p = p.tolist()
                        validation_batch_history.append(last_validation_batch_p)
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
                        calib_min_p_ok = bool(np.all(p > min_probability_threshold))
                        calib_coverage_ess = float(1.0 / np.sum(np.square(p)))
                        calib_coverage_ok = bool(calib_coverage_ess > coverage_ess_threshold)
                        if calib_min_p_ok and calib_coverage_ok:
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
                        bias_warmup_diag["status"] = "frozen_validation_converged_after_mbar_calibration"
                        bias_warmup_diag["coverage_ess"] = float(coverage_ess)
                        bias_warmup_diag["ema_mean_p"] = ema_mean_p_values
                        bias_warmup_diag["max_ema_mean_p"] = max_ema_p
                        bias_warmup_diag["min_ema_mean_p"] = min_ema_p
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
                            bias_warmup_diag["coverage_ess"] = float(coverage_ess)
                            bias_warmup_diag["ema_mean_p"] = ema_mean_p_values
                            bias_warmup_diag["max_ema_mean_p"] = max_ema_p
                            bias_warmup_diag["min_ema_mean_p"] = min_ema_p
                        print(
                            "    ⚠️ 用 BAR/MBAR（ΔF_bias，含 WCA、不含 LRC）校准的 f_k 冻结验证仍未"
                            "通过——λ 网格和求解器都已排除，可能是构象弛豫过慢或偏置表达式本身有"
                            "问题，需要人工检查，不再自动重试。"
                        )

            if bias_converged:
                print(
                    f" 完成（{bias_update_count} 次权重更新、{learning_to_validation_cycles} 次冻结验证失败"
                    f"重学习后，最终在冻结 f_k 下连续 {required_consecutive_bias_updates} 次独立 batch "
                    f"（原始概率，非 EMA）通过覆盖门：min(p_k)={min_ema_p:.4f}>{min_probability_threshold:.4f}，"
                    f"coverage_ESS={coverage_ess:.3f}>{coverage_ess_threshold:.3f}）"
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
                    if resumed_calibration_pending:
                        steps_spent_this_attempt = int(steps_at_full_bias)
                        new_cumulative_steps = (
                            int(getattr(sampler, "frozen_validation_cumulative_steps", 0))
                            + steps_spent_this_attempt
                        )
                    else:
                        steps_spent_this_attempt = int(
                            bias_warmup_diag["mbar_calibration"]["steps_used"]
                        )
                        new_cumulative_steps = steps_spent_this_attempt
                    sampler.frozen_validation_cumulative_steps = new_cumulative_steps
                    is_final_rung = bool((frozen_validation_is_final_rung or {}).get(window_idx, False))
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
                    + f"{steps_at_full_bias}/{max_bias_warmup_steps} 步安全上限，最终阶段={mode}）"
                )
                print(f"    ema_mean_p[k] = {ema_mean_p_values}")
                print(
                    "    相邻 Δu 诊断 = "
                    f"{sampler.last_update_diagnostics.get('adjacent_delta_u', [])}"
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
                raise IBSWarmupConvergenceError(
                    f"窗口 {window_idx} 的 IBS 偏置预热在 {bias_update_count} 次权重更新、"
                    f"{learning_to_validation_cycles} 次冻结验证失败重学习后仍未能在冻结 f_k 下通过"
                    f"独立验证；完整诊断已写入 {failure_path}。"
                    "该失败先反馈为拆窗；只有最小窗口的 fixed-H overlap 失败才允许插点。",
                    diagnostics=bias_warmup_diag,
                )

            # 恢复步长
            sim.integrator.setStepSize(original_dt)

            # 打印预热后的 f_k 值
            f_vals = [sim.context.getParameter(f"{self.prefix}_f_{k}") for k in range(len(lc_win))]
            print(f"  ✅ 偏置预热完成（冻结 f_k 已通过独立验证）: {[f'{v:.1f}' for v in f_vals]}")
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

            # ---- 进入生产采样：不再改变 bias_scale/最小化/重新爬坡 ----
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=7] 旧版本这里有"生产前卸压"：bias_scale
            # 清零→最小化→5000 步无偏置运行→重新爬回 1.0，紧接着直接进 production，
            # 从未对这个被重新扰动过的构型重新验证过覆盖度——哪怕前面的收敛判据是
            # 真的，这一步也会在没有复检的情况下把刚验证过的分布毁掉。现在冻结
            # 验证已经在完全固定的 production Hamiltonian（bias_scale=1.0、f_k 不再
            # 变化）下完成，不需要也不允许再碰 bias_scale/最小化/爬坡：只做一次
            # 非破坏性的安全力检查（不达标就直接报错，不再用更多扰动去"修复"一个
            # 刚验证通过的构型），然后原样进入生产采样。
            safety_state = sim.context.getState(getForces=True, getPositions=True)
            safety_forces = safety_state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
            fmax_safety = float(np.max(np.linalg.norm(safety_forces, axis=1)))
            preprod_force_threshold = 7000.0
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
            sampler.energy_history = []
            sampler.bias_history = []
            sampler.base_energy_history = []
            production_pos_backup = safety_state.getPositions(asNumpy=True)

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
            print(f"\n  [阶段5] 生产采样 ({effective_n_steps_per_window} 步)...")
            n_updates = effective_n_steps_per_window // steps_per_update

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

                if energy_bad or (fmax is not None and ((not np.isfinite(fmax)) or fmax > 7000.0)):
                    sim.context.setPositions(pos_backup)
                    sim.context.setVelocitiesToTemperature(self.temperature)
                    print("    🔧 触发回退，执行局部最小化释放应力...")
                    sim.minimizeEnergy(maxIterations=2000, tolerance=1.0)
                    current_dt_ps = sim.integrator.getStepSize().value_in_unit(unit.picoseconds)
                    new_dt_ps = max(0.0001, current_dt_ps * 0.5)
                    sim.integrator.setStepSize(new_dt_ps * unit.picoseconds)
                    fmax_report = fmax if fmax is not None else float("nan")
                    print(
                        f"    ⚠️ 灾难检测触发: update={up}/{n_updates}, "
                        f"E_total={e_total_n:.1f}, max|F|={fmax_report:.1f}. "
                        f"已回退坐标并将步长降至 {new_dt_ps*1000.0:.1f} fs"
                    )
                    if debug_mode:
                        diagnose_force_groups_detailed(sim.context, win_sys, prefix=f"窗口{window_idx}_回退_update{up}")
                        diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_回退_update{up}")
                    continue

                if do_force_check:
                    # 复用同一次 getState 里已经取回的坐标，不再单独发一次 getPositions 请求
                    production_pos_backup = state_n.getPositions(asNumpy=True)

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
                    local_result = _solve_single_window_local_mbar(
                        u_kj_raw, bias_kj, base_kj, list(range(start, end)), early_stop_kt,
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

                        ess_ratio_ok = min_ess_ratio is not None and min_ess_ratio >= early_stop_min_ess_ratio
                        absolute_ess_ok = absolute_ess is not None and absolute_ess >= early_stop_min_absolute_ess
                        decorrelated_ok = n_frames_used >= early_stop_min_decorrelated_samples
                        uncertainty_ok = np.isfinite(local_uncertainty) and local_uncertainty <= early_stop_max_uncertainty_kJ_mol
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
                # 🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION] LJ 长程尾项修正公式版本
                # （尽管名字里写着 traditional，v2 起这个常量同时覆盖 ACE/dual_lambda
                # 路径和传统 Beutler REMD 路径共用的 switching+softcore-aware LRC
                # 积分——见该常量定义处）。旧版本（v1，只补 cutoff 外的 r^-6、忽略
                # switching 区间）下的能量文件里，target_energies 里叠加的 LRC 数值
                # 跟当前公式不是同一回事，resume / 窗口产物复用逻辑必须校验这个
                # 字段，不能只看 λ/WCA/IBS 偏置协议是否匹配。
                "lj_tail_lrc_protocol_version": TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
                "n_steps_per_window_default": int(n_steps_per_window),
                "n_steps_per_window_effective": int(effective_n_steps_per_window),
                "n_updates": int(len(sampler.f_history)),
                "free_energy_history_kT": [
                    np.asarray(f_k, dtype=float).tolist()
                    for f_k in sampler.f_history
                ],
                "n_energy_samples": int(e_save.shape[1]),
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
            with open(
                os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_convergence.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(convergence, f, indent=2)
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


def _solve_single_window_local_mbar(
    u_kj_raw: np.ndarray,
    bias_kj: np.ndarray,
    base_kj: np.ndarray,
    win_lams: List[int],
    kt: float,
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
    n_frames = min(n_frames, len(bias_kj), len(base_kj))
    if n_frames < min_frames:
        return {"error": "insufficient_frames", "n_frames": int(n_frames)}
    u_kj_raw = u_kj_raw[:, :n_frames]
    bias_kj = bias_kj[:n_frames]
    base_kj = base_kj[:n_frames]

    raw_sampled_series = base_kj + bias_kj
    sub_idx, g_val = subsample_series_by_autocorrelation(raw_sampled_series)
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

    sampled_row = int(sampled_distribution_row)
    n_k_local = np.zeros(len(win_lams) + 1, dtype=np.int32)
    if not (0 <= sampled_row < len(n_k_local)):
        sampled_row = 0
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

    min_ess_ratio = None
    ess_ratio_per_lambda = None
    try:
        neff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)
        denom = max(int(n_k_local[sampled_row]), 1)
        target_idx = [i for i in range(len(neff)) if i != sampled_row]
        if target_idx:
            ess_ratio = neff[target_idx] / denom
            min_ess_ratio = float(np.min(ess_ratio))
            ess_ratio_per_lambda = {int(lam): float(r) for lam, r in zip(win_lams, ess_ratio)}
    except Exception:
        pass

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
    ) -> Dict:
        """局部 TMBAR + 自洽拼接

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
            # 🔑 修复（审查报告 #1）: 自相关子采样
            # ------------------------------------------------------------------
            # 该窗口只有一个真实被采样的分布 (base_kj + bias_kj)；用它自身的能量
            # 时间序列估计统计非效率 g，再对本窗口所有相关数组做同步去相关子采
            # 样，避免把强相关的逐帧 MD 输出当独立样本喂给 MBAR（会让误差棒系统
            # 性偏小）。
            raw_sampled_series = base_kj + bias_kj
            sub_idx, g_val = subsample_series_by_autocorrelation(raw_sampled_series)
            if sub_idx.size < n_frames:
                u_kj_raw = u_kj_raw[:, sub_idx]
                bias_kj = bias_kj[sub_idx]
                base_kj = base_kj[sub_idx]
                n_frames = int(sub_idx.size)
            window_g_values.append(float(g_val))
            if n_frames < 10:
                print(f"  ⚠️ 窗口 {w_idx} 去相关子采样后有效帧数 ({n_frames}) < 10，跳过")
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

                # ------------------------------------------------------------------
                # 🔑 修复（审查报告 #2）：真实的重加权质量诊断
                # ------------------------------------------------------------------
                # 该窗口的 MBAR 只有一个真实被采样的态 (row=sampled_row)，其余物理态
                # 样本数均为 0；标准 MBAR 重叠矩阵 compute_overlap() 在这种"单一采样
                # 分布 + 多个零样本目标态"场景下会退化（未采样列恒为 0，无法反映真实
                # 重叠）。这里改用 compute_effective_sample_number()：它衡量"把 row 0
                # 的样本重加权到每个目标 λ 态后，还剩多少个等效独立样本"，是这类单向
                # 重加权场景下物理上正确、数值上已验证有效的重叠代理（详见审查报告）。
                min_ess_ratio = None
                ess_ratio_per_lambda = None
                min_absolute_ess_this_window = None
                try:
                    neff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)
                    denom = max(int(n_k_local[sampled_row]), 1)
                    target_idx = [i for i in range(len(neff)) if i != sampled_row]
                    if target_idx:
                        ess_ratio = neff[target_idx] / denom
                        min_ess_ratio = float(np.min(ess_ratio))
                        # 🔑 绝对有效样本数（不是比例）：min_ess_ratio 只衡量"相对
                        # 于本窗口采样帧数"够不够，样本总数本身很少时，比例仍可能
                        # 轻易超过阈值——最终收敛门需要同时核对绝对数量。
                        min_absolute_ess_this_window = float(np.min(neff[target_idx]))
                        # target_idx is strictly [1..K] here (sampled_row is always 0
                        # from get_stage_data_for_analysis), i.e. rows 1..K of the
                        # augmented MBAR matrix -- which are win_lams in the same
                        # order. Keep the per-state breakdown, not just the min, so
                        # a caller can find *which* lambda inside this window is the
                        # actual overlap bottleneck instead of only knowing the
                        # window as a whole failed (needed by
                        # refine_stage_lambda_path_by_overlap in abfe_preoptimizer.py).
                        ess_ratio_per_lambda = {
                            int(lam): float(r) for lam, r in zip(win_lams, ess_ratio)
                        }
                except Exception as ess_exc:
                    print(f"  ⚠️ 窗口 {w_idx} 有效样本数(ESS)重叠诊断计算失败: {ess_exc}")
                window_overlap_records.append({
                    "window_index": w_idx,
                    "lambdas": win_lams,
                    "min_ess_ratio": min_ess_ratio,
                    "ess_ratio_per_lambda": ess_ratio_per_lambda,
                    "absolute_ess": min_absolute_ess_this_window,
                    # 🔑 去相关、剔除 NaN/Inf 列之后真正喂进 MBAR 的样本数（不是
                    # 去相关前的原始帧数）——跟下面 n_k_local[sampled_row]<10 的
                    # 门槛用的是同一个量。
                    "n_frames_decorrelated": int(n_k_local[sampled_row]),
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
        
        # 总自由能变化 (从第一个 lambda 到最后一个)
        total_dg = float(f_curve[-1] - f_curve[0])
        total_err = float(np.sqrt(global_var[sorted_lams[0]] + global_var[sorted_lams[-1]]))
        endpoint_error_after_offset = float(np.sqrt(err_curve[0]**2 + err_curve[-1]**2))
        
        # ------------------------------------------------------------------
        # 🔑 修复（审查报告 #2）：converged/min_overlap 曾经分别是"每个窗口的
        # MBAR 是否解出来了"和"自由能曲线相邻点间距的单调函数"——前者与统计收敛
        # 无关，后者根本不是重叠度。现在改用每个窗口 compute_effective_sample_
        # number() 算出的真实重加权有效样本比例（见上面窗口内循环），
        # min_overlap = 所有窗口中最差的那个比例；converged 要求全部窗口都成功
        # 解出 *且* 最差重叠比例不低于阈值。旧的 Δf 间距量保留在
        # lambda_spacing_max_step_kJ_mol 里供参考，但不再冒充"重叠度"。
        # ------------------------------------------------------------------
        ess_ratios = [w["min_ess_ratio"] for w in window_overlap_records if w["min_ess_ratio"] is not None]
        min_overlap = float(np.min(ess_ratios)) if ess_ratios else None
        min_overlap_threshold = float(final_min_ess_ratio)  # 保守阈值：单分布重加权到目标 λ 态的有效样本占比 <5% 视为不可信

        # 🔑 [P1 修复] 只检查 ratio 不够——样本总数很少时绝对有效样本数可能只有
        # 个位数，比例却轻易超过阈值。补上绝对 ESS、去相关样本数、端点不确定度
        # 三项硬门槛，跟在线 early-stop 用的判据集合一致（不含漂移，一次性拼接
        # 没有"上一次检查"的基线）。
        abs_ess_vals = [w["absolute_ess"] for w in window_overlap_records if w.get("absolute_ess") is not None]
        min_absolute_ess = float(np.min(abs_ess_vals)) if abs_ess_vals else None

        n_frames_vals = [
            w["n_frames_decorrelated"] for w in window_overlap_records
            if w.get("n_frames_decorrelated") is not None
        ]
        min_decorrelated_samples = int(np.min(n_frames_vals)) if n_frames_vals else 0

        unc_vals = [
            w["endpoint_diff_uncertainty_kJ_mol"] for w in window_overlap_records
            if w.get("endpoint_diff_uncertainty_kJ_mol") is not None
        ]
        max_endpoint_uncertainty_kJ_mol = float(np.max(unc_vals)) if unc_vals else None

        converged = bool(
            len(local_results) == len(valid_windows)
            and min_overlap is not None
            and min_overlap >= min_overlap_threshold
            and min_absolute_ess is not None
            and min_absolute_ess >= float(final_min_absolute_ess)
            and min_decorrelated_samples >= int(final_min_decorrelated_samples)
            and max_endpoint_uncertainty_kJ_mol is not None
            and np.isfinite(max_endpoint_uncertainty_kJ_mol)
            and max_endpoint_uncertainty_kJ_mol <= float(final_max_uncertainty_kJ_mol)
        )
        lambda_spacing_max_step = float(np.max(np.abs(np.diff(f_curve)))) if f_curve.size > 1 else 0.0

        return {
            "total_delta_G": total_dg,
            "total_error": total_err,
            "endpoint_error_after_offset": endpoint_error_after_offset,
            "offset_error_contribution": float(np.sqrt(np.sum(offset_var_terms))) if offset_var_terms else 0.0,
            "converged": converged,
            "min_overlap": min_overlap,
            "min_overlap_threshold": min_overlap_threshold,
            "min_overlap_method": "per_window_effective_sample_number_ratio",
            "min_absolute_ess": min_absolute_ess,
            "min_absolute_ess_threshold": float(final_min_absolute_ess),
            "min_decorrelated_samples": min_decorrelated_samples,
            "min_decorrelated_samples_threshold": int(final_min_decorrelated_samples),
            "max_endpoint_uncertainty_kJ_mol": max_endpoint_uncertainty_kJ_mol,
            "max_endpoint_uncertainty_kJ_mol_threshold": float(final_max_uncertainty_kJ_mol),
            "lambda_spacing_max_step_kJ_mol": lambda_spacing_max_step,
            "window_overlap_diagnostics": window_overlap_records,
            "statistical_inefficiency_per_window": window_g_values,
            "method": "Local-TMBAR-Stitched (offset-error-propagated, ESS-overlap-checked)",
            "uncertainty_note": (
                "局部 MBAR 拼接误差已传播窗口 offset 方差（逆方差加权，见 window_overlap_"
                "diagnostics）；converged 现在同时要求 ESS ratio、绝对 ESS、去相关样本数、"
                "端点不确定度四项都达标，不再只看 ratio。仍未包含跨窗口的完整全局 MBAR "
                "协方差矩阵。"
            ),
            "f_k": f_curve.tolist(),
            "df_k": err_curve.tolist(),
            "lambdas": sorted_lams
        }

    def _fallback(self, msg: str) -> Dict:
        return {"error": msg, "converged": False, "total_delta_G": 0.0, "total_error": 999.9}

# 模块级便捷入口
def solve_stage_integrated(
    window_outputs: List[Dict],
    kt: float,
    stage_name: str = "",
    final_min_ess_ratio: float = 0.05,
    final_min_absolute_ess: float = 50.0,
    final_min_decorrelated_samples: int = 20,
    final_max_uncertainty_kJ_mol: float = 1.0,
) -> Dict:
    if not window_outputs:
        return {"total_delta_G": 0.0, "total_error": 999.9, "converged": False}
    analyzer = GlobalMBARAnalyzer(kt=kt)
    res = analyzer.solve_stage_integrated(
        window_outputs,
        final_min_ess_ratio=final_min_ess_ratio,
        final_min_absolute_ess=final_min_absolute_ess,
        final_min_decorrelated_samples=final_min_decorrelated_samples,
        final_max_uncertainty_kJ_mol=final_max_uncertainty_kJ_mol,
    )
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
    rec = [int(i) for i in boresch_params["receptor_indices"]]
    lig = [int(i) for i in boresch_params["ligand_indices"]]

    H0, H1, H2 = pos[rec[0]], pos[rec[1]], pos[rec[2]]
    L0, L1, L2 = pos[lig[0]], pos[lig[1]], pos[lig[2]]
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
        lj_tail_prefactor = None
        lj_tail_lambda_power = 1.0
        lj_lrc_metadata = {
            "protocol_version": TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
            "applicable": bool(needs_traditional_lrc),
            "applied": False,
            "model": "analytic_mean_field_r6_tail",
            "cutoff_nm": 1.2,
            "lambda_vdw_power": lj_tail_lambda_power,
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
            tail_s = _lj_tail_correction_S_kj_nm6(
                all_params, list(ligand_indices), env_indices
            )
            cutoff_nm = 1.2
            lj_tail_prefactor = -(16.0 * math.pi / 3.0) * tail_s / cutoff_nm ** 3
            lj_lrc_metadata.update({
                "applied": True,
                "status": "implemented_analytic_mean_field",
                "tail_S_kj_nm6": float(tail_s),
                "prefactor_kj_nm3_mol": float(lj_tail_prefactor),
                "volume_mean_nm3": mean_volume,
                "volume_relative_span": relative_span,
            })
            print(
                "  ✅ 传统 Beutler LRC: 离线解析 r^-6 尾项已启用 "
                f"(V={mean_volume:.6f} nm^3, relative_span={relative_span:.3e})"
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
                    "lj_tail_prefactor_kj_nm3_mol": lj_tail_prefactor,
                    "lj_tail_lambda_vdw_power": lj_tail_lambda_power,
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

    def solve(self, u_kn: np.ndarray, decorrelate: bool = True) -> Dict:
        u_kn = np.asarray(u_kn, dtype=np.float64)
        K, N = u_kn.shape
        if not hasattr(self, "_last_n_k"):
            raise RuntimeError(
                "TraditionalMBARAnalyzer.solve() 缺少每态样本数 n_k。"
                "请先调用 compute_u_kn()，或在 solve() 前显式设置 analyzer._last_n_k。"
            )
        n_k = np.asarray(self._last_n_k, dtype=int)
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
                return {
                    "delta_G": dg,
                    "error": err,
                    "method": method_name,
                    "n_frames": N,
                    "n_states": K,
                    "converged": converged,
                    "min_overlap": min_overlap,
                    "diagnostics": diagnostics,
                }
            except Exception as exc:
                last_exc = exc
                print(f"  ⚠️ MBAR {protocol} 求解失败: {exc}")

        if no_uncertainty_fallback is not None:
            print(
                "  🚨 所有 solver protocol 均无法给出不确定度估计；"
                f"返回 {no_uncertainty_fallback['method']} 的 ΔG，error 标记为 nan——"
                "这条腿/阶段的最终误差棒不可信，请检查窗口重叠率与采样长度。"
            )
            return no_uncertainty_fallback

        raise RuntimeError(f"MBAR 求解失败，最后错误: {last_exc}")
