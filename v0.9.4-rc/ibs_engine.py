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
        "  ⚠️ [VDW softcore] CustomNonbondedForce LRC 已禁用；"
        "本 VDW 腿未自动补偿 LJ tail/dispersion correction，"
        "需在 thermodynamic cycle 中显式处理；APBS 外部项仅用于静电/连续介质类长程修正。"
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
        except Exception as e:
            # ⚠️ 不能静默吞掉：e_base 会被直接写入 base_energy_history 并喂给
            # GlobalMBARAnalyzer (u_phys_kj = base_kj + u_kj_raw)，一次静默失败
            # 会在自由能重构中注入一个虚假的 0.0 物理能量且无迹可查。
            print(f"  🚨 Base 能量 (Group 0,2,3,4,5) 获取失败，本帧回退为 0.0 并标记：{e}")
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
        debug_mode: bool = False,
    ):
        """
        执行全部窗口采样，能量保存至 {output_dir}/dual_window_{window_idx}_{stage_type}_energies.npy

        诊断功能 (debug_mode=True，生产环境默认关闭)：
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
                        if cached_e.ndim == 2 and cached_e.shape[0] == len(lc_win) and cached_e.shape[1] > 0:
                            print(
                                f"  ⏭️  窗口 {window_idx} 已有有效缓存能量 {cached_e.shape}，"
                                f"resume 模式下跳过重新采样。"
                            )
                            continue
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
            sampler = IBSSampler(sim.context, len(lc_win), self.temperature, self.prefix, ibs_wrapper=ibs_wrap)
            ibs_state_file = os.path.join(
                self.checkpoint_dir,
                f"ibs_state_{stage_type}_window_{window_idx}.json",
            )
            
            # 🔑 核心修复：断点续传状态检测
            is_resumed_ibs = False
            if resume and os.path.exists(ibs_state_file):
                is_resumed_ibs = sampler.load_ibs_state(ibs_state_file)

            # bias 预热收敛诊断（在 is_resumed_ibs 分支之外先占位，确保后面落盘时
            # 一定有值可写，不依赖走了哪条分支）。
            bias_warmup_diag = {"status": "not_run"}

            # ---------- 渐进预热 (仅在非续传时执行) ----------
            if not is_resumed_ibs:
                if enable_gradual_warmup:
                    # 只做时间步长爬坡；偏置力的唯一平滑引入通道是下面的
                    # "[偏置预热]" 块，两者不再重复覆盖同一个 bias_scale 目标。
                    self._gradual_warmup_debug(sim, ibs_wrap, sampler, warmup_steps, win_sys, window_idx, debug_mode)
                # 不管上面是否做了时间步长爬坡，bias_scale 的唯一一次 0→1.0
                # 平滑引入都交给紧接着的"[偏置预热]"块（它会显式把 bias_scale
                # 重置到 0.0 再爬升，不依赖这里是什么值）。
            else:
                # 🔑 续传时：直接开启偏置，跳过 Warmup（含下面的偏置预热块），防止二次叠加
                sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
                print(f"  🚀 检测到 IBS 历史状态，跳过 Warmup，直接进入生产采样")
                bias_warmup_diag = {"status": "skipped_resume"}

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
                        if target_scale >= 0.5 and len(sampler.energy_buffer) >= 10:
                            sampler.update_weights()

                    state = sim.context.getState(getEnergy=True)
                    if not np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
                        raise RuntimeError(f"偏置预热阶段在 scale={target_scale} 时能量 NaN")
                    print(" 完成")

                # ---- 最后一档 (bias_scale=1.0)：用真实收敛判据代替固定步数 ----
                # ✅ 收敛性修复：eta_sgd = 1/(1+t/100) 随调用次数衰减，"f_k 打印值
                # 不再变化"既可能是真收敛，也可能是学习率已经耗尽却没追上——这两种
                # 情况必须用 ema_mean_p（批次平均概率）是否已经拉平到 ~1/K 来区分，
                # 不能只看步数或 f_k 是否还在动。窗口之间各自独立收敛（不做跨窗口
                # f_k 拷贝），但每个窗口自己的收敛与否必须是可检验、可落盘的事实。
                sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
                print("    → Bias Scale 设为 1.0，按收敛判据运行...", end="", flush=True)
                K = len(lc_win)
                target_p = 1.0 / K
                convergence_threshold = 2.0 * target_p
                min_steps_at_full_bias = 5000
                max_steps_at_full_bias = 40000
                check_chunk = 500
                steps_at_full_bias = 0
                bias_converged = False
                while steps_at_full_bias < max_steps_at_full_bias:
                    sim.step(check_chunk)
                    steps_at_full_bias += check_chunk
                    sampler.collect_energies()
                    if len(sampler.energy_buffer) >= 10:
                        sampler.update_weights()
                    if (
                        steps_at_full_bias >= min_steps_at_full_bias
                        and sampler.ema_mean_p is not None
                        and float(np.max(sampler.ema_mean_p)) < convergence_threshold
                    ):
                        bias_converged = True
                        break

                state = sim.context.getState(getEnergy=True)
                if not np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
                    raise RuntimeError("偏置预热阶段在 scale=1.0 时能量 NaN")

                max_ema_p = float(np.max(sampler.ema_mean_p)) if sampler.ema_mean_p is not None else float("nan")
                bias_warmup_diag = {
                    "status": "converged" if bias_converged else "hit_step_cap_unconverged",
                    "n_states": int(K),
                    "target_p": float(target_p),
                    "convergence_threshold": float(convergence_threshold),
                    "max_ema_mean_p": max_ema_p,
                    "steps_at_full_bias": int(steps_at_full_bias),
                    "max_steps_at_full_bias": int(max_steps_at_full_bias),
                }
                if bias_converged:
                    print(f" 完成（{steps_at_full_bias} 步收敛，max(ema_mean_p)={max_ema_p:.3f} < {convergence_threshold:.3f}）")
                else:
                    print(
                        f" 未收敛（达到步数上限 {max_steps_at_full_bias}，"
                        f"max(ema_mean_p)={max_ema_p:.3f} ≥ {convergence_threshold:.3f}）"
                    )
                    print(
                        f"  ⚠️ 窗口 {window_idx} 的 IBS 权重预热未在步数上限内收敛，"
                        "该窗口结果可能仍带坍缩偏置；已写入 diagnostics，建议检查该阶段 "
                        "λ 状态数/窗口宽度。"
                    )

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
            else:
                print("  ⏭️  续传：跳过偏置预热块（f_k 已从历史状态恢复）")

            print("\n  🧯 [生产前卸压] 开始...")
            sim.context.setParameter(f"{self.prefix}_bias_scale", 0.0)
            sim.minimizeEnergy(maxIterations=2000)
            for _ in range(20):
                sim.step(250)
            # ✅ 性能修复：偏置力在上面已经完整、平滑地爬升过一次（f_k 也已收敛
            # 到位），这里只是最小化之后的一次短暂再引入，不需要重新走一遍完整的
            # 15 档/15000 步爬坡——那实质上是第三次清零重爬，参见上面两处修复。
            for scale in np.linspace(0.25, 1.0, 4):
                sim.context.setParameter(f"{self.prefix}_bias_scale", float(scale))
                sim.step(500)

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
            convergence = {
                "window_idx": int(window_idx),
                "stage_type": stage_type,
                "n_updates": int(len(sampler.f_history)),
                "free_energy_history_kT": [
                    np.asarray(f_k, dtype=float).tolist()
                    for f_k in sampler.f_history
                ],
                "n_energy_samples": int(e_save.shape[1]),
                "bias_warmup": bias_warmup_diag,
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
                try:
                    neff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)
                    denom = max(int(n_k_local[sampled_row]), 1)
                    target_idx = [i for i in range(len(neff)) if i != sampled_row]
                    if target_idx:
                        ess_ratio = neff[target_idx] / denom
                        min_ess_ratio = float(np.min(ess_ratio))
                except Exception as ess_exc:
                    print(f"  ⚠️ 窗口 {w_idx} 有效样本数(ESS)重叠诊断计算失败: {ess_exc}")
                window_overlap_records.append({
                    "window_index": w_idx,
                    "lambdas": win_lams,
                    "min_ess_ratio": min_ess_ratio,
                })

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
        min_overlap_threshold = 0.05  # 保守阈值：单分布重加权到目标 λ 态的有效样本占比 <5% 视为不可信
        converged = bool(
            len(local_results) == len(valid_windows)
            and min_overlap is not None
            and min_overlap >= min_overlap_threshold
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
            "lambda_spacing_max_step_kJ_mol": lambda_spacing_max_step,
            "window_overlap_diagnostics": window_overlap_records,
            "statistical_inefficiency_per_window": window_g_values,
            "method": "Local-TMBAR-Stitched (offset-error-propagated, ESS-overlap-checked)",
            "uncertainty_note": (
                "局部 MBAR 拼接误差已传播窗口 offset 方差（逆方差加权，见 window_overlap_"
                "diagnostics）；converged/min_overlap 现在基于每窗口 MBAR "
                "compute_effective_sample_number() 算出的真实重加权有效样本比例。"
                "仍未包含跨窗口的完整全局 MBAR 协方差矩阵。"
            ),
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
        system, topology, perturbed_indices,
        restraint_params=restraint_params, box_vectors=box_vectors,
    )
    lambdas_bridge_s = sorted(float(s) for s in lambdas_bridge_s)
    if lambdas_bridge_s[0] != 0.0 or lambdas_bridge_s[-1] != 1.0:
        raise ValueError("lambdas_bridge_s 必须以 0.0 (PME full charge) 开始、以 1.0 (Shadow full charge) 结束。")

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
        bridge_sys, topology, traj_files, lambdas_bridge_s,
        s_param_name=s_name, box_vectors=box_vectors, platform_name="CPU",
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
        "diagnostics": res.get("diagnostics", {}),
    }


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
