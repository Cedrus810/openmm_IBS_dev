#!/usr/bin/env python
"""离线比较 charging 的 llfreeze 与全静电 annihilation 口径。

本脚本不运行 MD，也不覆盖任何主结果。它读取既有 decharging replica DCD 和
``decharging_pme_u_kn.npy``，把原来的 K 个 llfreeze 采样态作为 ``N_k > 0`` 的
MBAR 态；随后在同一批帧上计算 K-1 个全静电 annihilation 目标态，并以
``N_k = 0`` 加入扩展 MBAR。

这里的 annihilation 定义为：

* ligand-environment 普通 Coulomb 与 exception 按 ``lambda`` 缩放；
* ligand-ligand 普通 Coulomb、PME reciprocal/self 和 exception 按
  ``lambda**2`` 缩放；
* LJ、键合项和 Boresch restraint 不变。

粒子电荷使用 ``q_i(lambda) = lambda*q_i``，所以 L-L 普通 Coulomb 与 PME
self/reciprocal 自然按 lambda**2 变化；L-L exception 另用一个全局参数显式
设置为 lambda**2。

重要限制
--------
这是零样本目标态重加权。只有当 annihilation 端点的目标 ESS/权重集中度可接受
时，得到的自由能位移才可作定量判断；覆盖不足时，输出只用于量级和符号诊断。
"""

from __future__ import annotations

# 默认运行目录：统一由 tools/_run_dir.py 解析（ABFE_OUTPUT_DIR -> abfe_config.json
# 的 "output" -> ./output）。2026-08-31 前这里硬编码 output_lrc_fix，那是
# Atenolol-rank11 的验收基线目录，不在本工程区分支里。显式传参永远优先。
import sys as _abfe_rd_sys
from pathlib import Path as _AbfeRdPath

_ABFE_TOOLS_ROOT = _AbfeRdPath(__file__).resolve().parents[1]
if str(_ABFE_TOOLS_ROOT) not in _abfe_rd_sys.path:
    _abfe_rd_sys.path.insert(0, str(_ABFE_TOOLS_ROOT))
from _run_dir import DEFAULT_RUN_DIR  # noqa: E402


# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))


import argparse
import hashlib
import json
import math
import os
import platform as py_platform
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


R_KJ_MOL_K = 8.31446261815324e-3
KCAL_TO_KJ = 4.184
PROTOCOL_VERSION = 1
LAMBDA_ENV_NAME = "diag_lambda_coul"
LAMBDA_LL_NAME = "diag_lambda_coul_sq"
EVAL_FORCE_GROUP = 31


@dataclass(frozen=True)
class LegPaths:
    label: str
    meta: str
    system_xml: str
    topology_cif: str
    trajectory_dir: str


def _json_load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _cache_fingerprint(
    paths: LegPaths,
    sampled_u_path: str,
    sampled_n_k_path: str,
    lambdas: np.ndarray,
    trajectory_files: Sequence[str],
) -> str:
    """Fingerprint all inputs that determine the cached target energies."""
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "target": "full_electrostatic_annihilation_q=lambda*q_ll=lambda^2",
        "system_xml_sha256": _sha256_file(paths.system_xml),
        "sampled_u_kn_sha256": _sha256_file(sampled_u_path),
        "sampled_n_k_sha256": _sha256_file(sampled_n_k_path),
        "lambdas": [float(value) for value in lambdas],
        # Full DCD hashing would reread ~0.5 GB.  For this local acceleration
        # cache, resolved path + size + mtime is a deliberate invalidation key.
        "trajectories": [
            {
                "path": os.path.abspath(path),
                "size": int(os.stat(path).st_size),
                "mtime_ns": int(os.stat(path).st_mtime_ns),
            }
            for path in trajectory_files
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _npy_paths_from_meta(meta_path: str) -> Tuple[str, str]:
    if not meta_path.endswith(".meta.json"):
        raise ValueError(f"meta 文件名必须以 .meta.json 结尾: {meta_path}")
    base = meta_path[: -len(".meta.json")]
    return base + ".npy", base + ".npy.n_k.npy"


def _load_sampled_leg(
    paths: LegPaths,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, List[str]]:
    meta = _json_load(paths.meta)
    u_path, n_k_path = _npy_paths_from_meta(paths.meta)
    sampled_u_kn = np.asarray(np.load(u_path, allow_pickle=False), dtype=np.float64)
    n_k = np.asarray(np.load(n_k_path, allow_pickle=False), dtype=int)
    lambdas = np.asarray(meta["lambdas_coul"], dtype=np.float64)

    if sampled_u_kn.ndim != 2:
        raise ValueError(f"{paths.label}: sampled u_kn 必须是二维，收到 {sampled_u_kn.shape}")
    K, N = sampled_u_kn.shape
    if lambdas.shape != (K,):
        raise ValueError(
            f"{paths.label}: lambda 数量 {lambdas.size} 与 u_kn 状态数 {K} 不一致"
        )
    if n_k.shape != (K,) or int(np.sum(n_k)) != N:
        raise ValueError(
            f"{paths.label}: n_k={n_k.tolist()} 与 u_kn.shape={sampled_u_kn.shape} 不一致"
        )
    if not (
        np.all(np.diff(lambdas) < 0.0)
        and abs(float(lambdas[0]) - 1.0) < 1.0e-8
        and abs(float(lambdas[-1])) < 1.0e-8
    ):
        raise ValueError(
            f"{paths.label}: 本脚本要求 lambda 从1严格降到0，收到 {lambdas.tolist()}"
        )
    if not np.all(np.isfinite(sampled_u_kn)):
        raise ValueError(f"{paths.label}: sampled u_kn 含 NaN/Inf")

    traj_files = [
        os.path.join(paths.trajectory_dir, f"decharging_rep{k}.dcd") for k in range(K)
    ]
    missing = [path for path in traj_files if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"{paths.label}: 缺 replica DCD: {missing}")
    for required in (paths.system_xml, paths.topology_cif):
        if not os.path.isfile(required):
            raise FileNotFoundError(f"{paths.label}: 缺输入文件: {required}")
    return meta, sampled_u_kn, n_k, lambdas, traj_files


def _openmm_imports():
    try:
        import mdtraj as md
        import openmm
        from openmm import app, unit
    except ImportError as exc:
        raise RuntimeError(
            "离线能量重算需要运行生产环境中的 openmm 和 mdtraj。"
            "请在生成这些 DCD 的同一 conda 环境运行脚本。"
        ) from exc
    return md, openmm, app, unit


def _find_single_nonbonded_force(system, openmm):
    matches = [
        force
        for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"期望恰好1个 NonbondedForce，实际 {len(matches)}")
    return matches[0]


def _global_parameter_names(nonbonded) -> List[str]:
    return [
        str(nonbonded.getGlobalParameterName(i))
        for i in range(nonbonded.getNumGlobalParameters())
    ]


def build_full_electrostatic_annihilation_system(
    native_system,
    ligand_indices: Sequence[int],
):
    """构建仅供离线求能的全静电 annihilation System。

    lambda=1 与 native System 精确相同；lambda=0 时所有 ligand 电荷及
    ligand 参与的 exception chargeProd 都为零，LJ 参数保持不变。
    """
    _, openmm, _, unit = _openmm_imports()
    system = openmm.XmlSerializer.deserialize(
        openmm.XmlSerializer.serialize(native_system)
    )
    system.thisown = 1
    nb = _find_single_nonbonded_force(system, openmm)
    existing = set(_global_parameter_names(nb))
    collisions = existing.intersection({LAMBDA_ENV_NAME, LAMBDA_LL_NAME})
    if collisions:
        raise RuntimeError(f"诊断全局参数名与原 System 冲突: {sorted(collisions)}")
    nb.addGlobalParameter(LAMBDA_ENV_NAME, 1.0)
    nb.addGlobalParameter(LAMBDA_LL_NAME, 1.0)

    ligand_set = {int(i) for i in ligand_indices}
    if not ligand_set:
        raise ValueError("ligand_indices 为空")
    if min(ligand_set) < 0 or max(ligand_set) >= system.getNumParticles():
        raise IndexError(
            f"ligand index 越界: range={min(ligand_set)}..{max(ligand_set)}, "
            f"n_particles={system.getNumParticles()}"
        )

    net_charge_e = 0.0
    for particle in sorted(ligand_set):
        charge, sigma, epsilon = nb.getParticleParameters(particle)
        net_charge_e += charge.value_in_unit(unit.elementary_charge)
        nb.setParticleParameters(
            particle,
            0.0 * unit.elementary_charge,
            sigma,
            epsilon,
        )
        nb.addParticleParameterOffset(
            LAMBDA_ENV_NAME,
            particle,
            charge,
            0.0 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )

    n_ll_exceptions = 0
    n_lenv_exceptions = 0
    for exc_idx in range(nb.getNumExceptions()):
        p1, p2, charge_prod, sigma, epsilon = nb.getExceptionParameters(exc_idx)
        p1, p2 = int(p1), int(p2)
        in1, in2 = p1 in ligand_set, p2 in ligand_set
        if not (in1 or in2):
            continue
        nb.setExceptionParameters(
            exc_idx,
            p1,
            p2,
            0.0 * unit.elementary_charge**2,
            sigma,
            epsilon,
        )
        parameter = LAMBDA_LL_NAME if in1 and in2 else LAMBDA_ENV_NAME
        nb.addExceptionParameterOffset(
            parameter,
            exc_idx,
            charge_prod,
            0.0 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )
        if in1 and in2:
            n_ll_exceptions += 1
        else:
            n_lenv_exceptions += 1

    # 离线只读取这个 NonbondedForce；把其他力移出目标组，避免已有 group=31 碰撞。
    for force in system.getForces():
        force.setForceGroup(0)
    nb.setForceGroup(EVAL_FORCE_GROUP)

    metadata = {
        "ligand_net_charge_e": float(net_charge_e),
        "n_ligand_particles": int(len(ligand_set)),
        "n_ll_exceptions_scaled_lambda_squared": int(n_ll_exceptions),
        "n_ligand_environment_exceptions_scaled_lambda": int(n_lenv_exceptions),
        "particle_charge_rule": "q_i(lambda)=lambda*q_i",
        "ll_rule": "normal PME/self/reciprocal and exceptions scale as lambda^2",
    }
    if abs(net_charge_e) > 1.0e-3:
        raise RuntimeError(
            f"ligand 净电荷为 {net_charge_e:+.6f} e；当前离线 target 会改变盒子总电荷。"
            "必须先复现生产计算的 co-alchemical counterion 与有限尺寸修正口径，"
            "本脚本拒绝给出误导结果。"
        )
    return system, metadata


def _platform_and_properties(openmm, platform_spec: str):
    spec = str(platform_spec or "CPU").strip()
    if ":" in spec:
        name, device = spec.split(":", 1)
        name, device = name.strip(), device.strip()
    else:
        name, device = spec, ""
    platform_obj = openmm.Platform.getPlatformByName(name)
    props: Dict[str, str] = {}
    if name.upper() in {"CUDA", "OPENCL"}:
        props["Precision"] = "mixed"
        if device:
            props["DeviceIndex"] = device
    return platform_obj, props


def _set_lambda(context, value: float) -> None:
    context.setParameter(LAMBDA_ENV_NAME, float(value))
    context.setParameter(LAMBDA_LL_NAME, float(value) ** 2)


def _trajectory_frame_iterator(
    traj_files: Sequence[str],
    topology,
    expected_n_k: np.ndarray,
) -> Iterable[Tuple[int, int, Any]]:
    md, _, _, _ = _openmm_imports()
    md_topology = md.Topology.from_openmm(topology)
    for state_index, (path, expected) in enumerate(zip(traj_files, expected_n_k)):
        trajectory = md.load_dcd(path, top=md_topology)
        if int(trajectory.n_frames) != int(expected):
            raise RuntimeError(
                f"replica {state_index}: DCD 帧数 {trajectory.n_frames} != n_k {expected}"
            )
        for local_index in range(trajectory.n_frames):
            yield state_index, local_index, trajectory[local_index]


def compute_annihilation_target_u_kn(
    paths: LegPaths,
    sampled_u_kn: np.ndarray,
    n_k: np.ndarray,
    lambdas: np.ndarray,
    ligand_indices: Sequence[int],
    temperature_k: float,
    platform_spec: str,
    progress_every: int = 25,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """在原帧上计算 annihilation 目标态，并逐帧对齐到 sampled λ=1 行。"""
    _, openmm, app, unit = _openmm_imports()
    with open(paths.system_xml, encoding="utf-8") as handle:
        native_system = openmm.XmlSerializer.deserialize(handle.read())
    native_system.thisown = 1
    target_system, system_meta = build_full_electrostatic_annihilation_system(
        native_system, ligand_indices
    )
    topology = app.PDBxFile(paths.topology_cif).topology
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    platform_obj, props = _platform_and_properties(openmm, platform_spec)
    context = openmm.Context(target_system, integrator, platform_obj, props)

    K, N = sampled_u_kn.shape
    kt = R_KJ_MOL_K * float(temperature_k)
    target_u_kn = np.empty((K, N), dtype=np.float64)
    cursor = 0
    try:
        for _, _, frame in _trajectory_frame_iterator(
            [
                os.path.join(paths.trajectory_dir, f"decharging_rep{k}.dcd")
                for k in range(K)
            ],
            topology,
            n_k,
        ):
            context.setPositions(frame.xyz[0] * unit.nanometer)
            if frame.unitcell_vectors is not None:
                context.setPeriodicBoxVectors(
                    *[
                        openmm.Vec3(*np.asarray(vector, dtype=float)) * unit.nanometer
                        for vector in frame.unitcell_vectors[0]
                    ]
                )

            # 以同一帧的物理 λ=1 端点为零点。stored sampled_u_kn[0] 已包含
            # Boresch/键合等公共项；这里只加 NonbondedForce 的目标态相对变化，
            # 因而所有行保留完全一致的逐帧能量零点。
            _set_lambda(context, 1.0)
            ref_energy = context.getState(
                getEnergy=True, groups={EVAL_FORCE_GROUP}
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            for target_index, lam in enumerate(lambdas):
                if target_index == 0:
                    target_u_kn[target_index, cursor] = sampled_u_kn[0, cursor]
                    continue
                _set_lambda(context, float(lam))
                energy = context.getState(
                    getEnergy=True, groups={EVAL_FORCE_GROUP}
                ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                target_u_kn[target_index, cursor] = (
                    sampled_u_kn[0, cursor] + (float(energy) - float(ref_energy)) / kt
                )

            cursor += 1
            if progress_every > 0 and (cursor % progress_every == 0 or cursor == N):
                print(f"    {paths.label}: annihilation 能量 {cursor}/{N} 帧")
    finally:
        del context, integrator, target_system, native_system

    if cursor != N:
        raise RuntimeError(f"{paths.label}: 实际处理 {cursor} 帧，预期 {N}")
    if not np.all(np.isfinite(target_u_kn)):
        raise RuntimeError(f"{paths.label}: annihilation target u_kn 含 NaN/Inf")

    endpoint_identity_max_abs = float(
        np.max(np.abs(target_u_kn[0] - sampled_u_kn[0]))
    )
    if endpoint_identity_max_abs > 1.0e-12:
        raise RuntimeError(
            f"{paths.label}: λ=1逐帧对齐失败，max|Δu|={endpoint_identity_max_abs}"
        )
    return target_u_kn, {
        **system_meta,
        "platform": str(platform_spec),
        "n_frames": int(N),
        "endpoint_lambda1_identity_max_abs_reduced": endpoint_identity_max_abs,
    }


def _build_mbar(u_kn: np.ndarray, n_k: np.ndarray):
    try:
        import pymbar
    except ImportError as exc:
        raise RuntimeError(
            "扩展 MBAR 需要 pymbar；请在生产 conda 环境运行脚本。"
        ) from exc

    stable = np.asarray(u_kn, dtype=np.float64)
    stable = stable - np.min(stable, axis=0, keepdims=True)
    attempts = (
        {"solver_protocol": "robust", "relative_tolerance": 1.0e-10, "verbose": False},
        {"solver_protocol": "default", "relative_tolerance": 1.0e-10, "verbose": False},
        {"verbose": False},
        {},
    )
    last_error = None
    for kwargs in attempts:
        try:
            return pymbar.MBAR(stable, n_k, **kwargs)
        except (TypeError, ValueError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"PyMBAR 无法建立扩展模型: {last_error}") from last_error


def _free_energy_arrays(mbar) -> Tuple[np.ndarray, np.ndarray]:
    try:
        result = mbar.compute_free_energy_differences(compute_uncertainty=True)
    except TypeError:
        result = mbar.compute_free_energy_differences()
    delta_f = np.asarray(result["Delta_f"], dtype=float)
    delta_df = np.asarray(result["dDelta_f"], dtype=float)
    return delta_f, delta_df


def _target_weight_diagnostics(mbar, target_index: int) -> Dict[str, Any]:
    weights = getattr(mbar, "W_nk", None)
    if weights is None:
        return {"available": False, "reason": "pymbar.W_nk unavailable"}
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or target_index >= weights.shape[1]:
        return {
            "available": False,
            "reason": f"unexpected W_nk shape {weights.shape}",
        }
    w = np.asarray(weights[:, target_index], dtype=np.float64)
    total = float(np.sum(w))
    if not np.isfinite(total) or total <= 0.0:
        return {"available": False, "reason": "nonpositive target weight sum"}
    w = w / total
    ordered = np.sort(w)[::-1]
    ess = float(1.0 / np.sum(np.square(w)))
    return {
        "available": True,
        "effective_sample_size": ess,
        "max_normalized_weight": float(ordered[0]),
        "top_10_weight_fraction": float(np.sum(ordered[: min(10, ordered.size)])),
        "top_1_percent_weight_fraction": float(
            np.sum(ordered[: max(1, int(math.ceil(0.01 * ordered.size)))])
        ),
        "n_frames": int(w.size),
    }


def analyse_leg(
    paths: LegPaths,
    platform_spec: str,
    cache_path: Optional[str],
    progress_every: int,
    min_target_ess: float,
    max_target_weight: float,
) -> Tuple[Dict[str, Any], np.ndarray]:
    meta, sampled_u_kn, n_k, lambdas, traj_files = _load_sampled_leg(paths)
    temperature_k = float(meta["temperature_k"])
    ligand_indices = [int(i) for i in meta["ligand_indices"]]

    sampled_u_path, sampled_n_k_path = _npy_paths_from_meta(paths.meta)
    input_fingerprint = _cache_fingerprint(
        paths, sampled_u_path, sampled_n_k_path, lambdas, traj_files
    )
    cache_used = False
    target_meta: Dict[str, Any]
    if cache_path and os.path.isfile(cache_path):
        with np.load(cache_path, allow_pickle=False) as cached:
            target_u_kn = np.asarray(cached["target_u_kn"], dtype=np.float64)
            cached_lambdas = np.asarray(cached["lambdas"], dtype=np.float64)
            cached_fingerprint = (
                str(cached["input_fingerprint"].item())
                if "input_fingerprint" in cached.files
                else ""
            )
            cache_valid = bool(
                target_u_kn.shape == sampled_u_kn.shape
                and np.allclose(cached_lambdas, lambdas, rtol=0.0, atol=1.0e-12)
                and cached_fingerprint == input_fingerprint
            )
            if cache_valid:
                target_meta = json.loads(str(cached["target_meta_json"].item()))
                cache_used = True
        if cache_used:
            print(f"  {paths.label}: 复用已校验 target u_kn cache: {cache_path}")
        else:
            print(f"  {paths.label}: cache 输入已变化或版本过旧，重新计算")

    if not cache_used:
        print(f"  {paths.label}: 开始在原 DCD 上计算 annihilation 目标态")
        target_u_kn, target_meta = compute_annihilation_target_u_kn(
            paths=paths,
            sampled_u_kn=sampled_u_kn,
            n_k=n_k,
            lambdas=lambdas,
            ligand_indices=ligand_indices,
            temperature_k=temperature_k,
            platform_spec=platform_spec,
            progress_every=progress_every,
        )
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            np.savez_compressed(
                cache_path,
                target_u_kn=target_u_kn,
                lambdas=lambdas,
                n_k=n_k,
                input_fingerprint=input_fingerprint,
                target_meta_json=json.dumps(target_meta, sort_keys=True),
            )
            print(f"  {paths.label}: target u_kn cache 已写入 {cache_path}")

    K, N = sampled_u_kn.shape
    # target λ=1 与 sampled λ=1 是同一物理态，不重复加入。其余 K-1 个目标态 N_k=0。
    extended_u_kn = np.vstack([sampled_u_kn, target_u_kn[1:]])
    extended_n_k = np.concatenate([n_k, np.zeros(K - 1, dtype=int)])
    mbar = _build_mbar(extended_u_kn, extended_n_k)
    delta_f, delta_df = _free_energy_arrays(mbar)
    kt = R_KJ_MOL_K * temperature_k

    sampled_start, sampled_end = 0, K - 1
    target_start, target_end = sampled_start, extended_u_kn.shape[0] - 1
    llfreeze_dg = float(delta_f[sampled_start, sampled_end] * kt)
    llfreeze_err = float(delta_df[sampled_start, sampled_end] * kt)
    annihilation_dg = float(delta_f[target_start, target_end] * kt)
    annihilation_err = float(delta_df[target_start, target_end] * kt)
    shift = annihilation_dg - llfreeze_dg

    weight_diag = _target_weight_diagnostics(mbar, target_end)
    target_ess = float(weight_diag.get("effective_sample_size", 0.0) or 0.0)
    target_max_weight = float(weight_diag.get("max_normalized_weight", 1.0) or 1.0)
    reliable = bool(
        weight_diag.get("available")
        and target_ess >= float(min_target_ess)
        and target_max_weight <= float(max_target_weight)
        and np.isfinite(annihilation_err)
    )

    per_target = []
    for j in range(1, K):
        ext_index = K + j - 1
        diag_j = _target_weight_diagnostics(mbar, ext_index)
        per_target.append(
            {
                "lambda": float(lambdas[j]),
                "extended_state_index": int(ext_index),
                "delta_G_from_lambda1_kJ_mol": float(delta_f[0, ext_index] * kt),
                "error_kJ_mol": float(delta_df[0, ext_index] * kt),
                "weight_diagnostics": diag_j,
            }
        )

    result = {
        "label": paths.label,
        "temperature_K": temperature_k,
        "lambdas": lambdas.tolist(),
        "n_k_sampled_llfreeze": n_k.tolist(),
        "n_frames_total": int(N),
        "llfreeze": {
            "delta_G_1_to_0_kJ_mol": llfreeze_dg,
            "error_kJ_mol": llfreeze_err,
            "source": "sampled decharging_pme_u_kn rows in extended MBAR",
        },
        "annihilation_reweighted": {
            "delta_G_1_to_0_kJ_mol": annihilation_dg,
            "error_kJ_mol": annihilation_err,
            "shift_vs_llfreeze_kJ_mol": shift,
            "target_endpoint_weight_diagnostics": weight_diag,
            "reweighting_reliable": reliable,
            "reliability_thresholds": {
                "min_target_ess": float(min_target_ess),
                "max_target_normalized_weight": float(max_target_weight),
            },
            "per_target_state": per_target,
        },
        "target_system": {
            **target_meta,
            "cache_used": cache_used,
            "definition": {
                "ligand_environment": "lambda",
                "ligand_ligand_normal_and_pme": "lambda^2",
                "ligand_ligand_exceptions": "lambda^2",
                "lennard_jones": "unchanged",
            },
        },
        "inputs": {
            "meta": os.path.abspath(paths.meta),
            "system_xml": os.path.abspath(paths.system_xml),
            "topology_cif": os.path.abspath(paths.topology_cif),
            "trajectory_files": [os.path.abspath(path) for path in traj_files],
            "sampled_u_kn_sha256": _sha256_file(_npy_paths_from_meta(paths.meta)[0]),
            "sampled_n_k_sha256": _sha256_file(_npy_paths_from_meta(paths.meta)[1]),
            "system_xml_sha256": _sha256_file(paths.system_xml),
            "cache_input_fingerprint": input_fingerprint,
        },
    }
    return result, target_u_kn


def _default_paths(root: str) -> Tuple[LegPaths, LegPaths]:
    root = os.path.abspath(root)
    complex_paths = LegPaths(
        label="complex",
        meta=os.path.join(root, "decharging", "decharging_pme_u_kn.meta.json"),
        system_xml=os.path.join(root, "system_native.xml"),
        topology_cif=os.path.join(root, "topology.cif"),
        trajectory_dir=os.path.join(root, "decharging"),
    )
    solvent_paths = LegPaths(
        label="solvent",
        meta=os.path.join(
            root, "solvent_leg", "decharging", "decharging_pme_u_kn.meta.json"
        ),
        system_xml=os.path.join(root, "system_solvent.xml"),
        topology_cif=os.path.join(root, "topology_solvent.cif"),
        trajectory_dir=os.path.join(root, "solvent_leg", "decharging"),
    )
    return complex_paths, solvent_paths


def _override_leg_paths(default: LegPaths, args, prefix: str) -> LegPaths:
    return LegPaths(
        label=default.label,
        meta=getattr(args, f"{prefix}_meta") or default.meta,
        system_xml=getattr(args, f"{prefix}_system") or default.system_xml,
        topology_cif=getattr(args, f"{prefix}_topology") or default.topology_cif,
        trajectory_dir=getattr(args, f"{prefix}_traj_dir") or default.trajectory_dir,
    )


def _print_leg_result(result: Dict[str, Any]) -> None:
    ll = result["llfreeze"]
    ann = result["annihilation_reweighted"]
    wd = ann["target_endpoint_weight_diagnostics"]
    print(f"\n=== {result['label']} ===")
    print(
        f"  llfreeze     ΔG = {ll['delta_G_1_to_0_kJ_mol']:.4f} "
        f"± {ll['error_kJ_mol']:.4f} kJ/mol"
    )
    print(
        f"  annihilation ΔG = {ann['delta_G_1_to_0_kJ_mol']:.4f} "
        f"± {ann['error_kJ_mol']:.4f} kJ/mol"
    )
    print(f"  shift ann−llfreeze = {ann['shift_vs_llfreeze_kJ_mol']:+.4f} kJ/mol")
    if wd.get("available"):
        print(
            f"  target λ=0 ESS={wd['effective_sample_size']:.1f}, "
            f"max_weight={wd['max_normalized_weight']:.4f}, "
            f"top10={wd['top_10_weight_fraction']:.3f}"
        )
    print(
        "  reliability: "
        + ("✓ 可作定量诊断" if ann["reweighting_reliable"] else "⚠️ 覆盖不足，只看符号/量级")
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-root", default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", default="./diagnostics/charging_annihilation_reweight.json")
    parser.add_argument("--cache-dir", default="./diagnostics/charging_annihilation_cache")
    parser.add_argument(
        "--platform",
        default="CUDA",
        help="OpenMM 求能平台，例如 CUDA、CUDA:0、OpenCL、CPU",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--min-target-ess", type=float, default=50.0)
    parser.add_argument("--max-target-weight", type=float, default=0.10)
    parser.add_argument("--reference-complex", type=float, default=75.10)
    parser.add_argument("--reference-solvent", type=float, default=68.10)

    for prefix in ("complex", "solvent"):
        parser.add_argument(f"--{prefix}-meta", default=None)
        parser.add_argument(f"--{prefix}-system", default=None)
        parser.add_argument(f"--{prefix}-topology", default=None)
        parser.add_argument(f"--{prefix}-traj-dir", default=None)
    args = parser.parse_args(argv)

    default_complex, default_solvent = _default_paths(args.output_root)
    complex_paths = _override_leg_paths(default_complex, args, "complex")
    solvent_paths = _override_leg_paths(default_solvent, args, "solvent")
    cache_dir = os.path.abspath(args.cache_dir) if args.cache_dir else None
    complex_cache = (
        os.path.join(cache_dir, "complex_annihilation_target_u_kn.npz")
        if cache_dir
        else None
    )
    solvent_cache = (
        os.path.join(cache_dir, "solvent_annihilation_target_u_kn.npz")
        if cache_dir
        else None
    )

    print("离线 charging Hamiltonian 口径诊断")
    print("  sampled: llfreeze (N_k>0)")
    print("  target : full electrostatic annihilation (N_k=0)")
    print(f"  platform: {args.platform}")

    complex_result, _ = analyse_leg(
        complex_paths,
        platform_spec=args.platform,
        cache_path=complex_cache,
        progress_every=args.progress_every,
        min_target_ess=args.min_target_ess,
        max_target_weight=args.max_target_weight,
    )
    solvent_result, _ = analyse_leg(
        solvent_paths,
        platform_spec=args.platform,
        cache_path=solvent_cache,
        progress_every=args.progress_every,
        min_target_ess=args.min_target_ess,
        max_target_weight=args.max_target_weight,
    )

    _print_leg_result(complex_result)
    _print_leg_result(solvent_result)

    complex_shift = float(
        complex_result["annihilation_reweighted"]["shift_vs_llfreeze_kJ_mol"]
    )
    solvent_shift = float(
        solvent_result["annihilation_reweighted"]["shift_vs_llfreeze_kJ_mol"]
    )
    binding_shift = solvent_shift - complex_shift
    current_binding = (
        float(solvent_result["llfreeze"]["delta_G_1_to_0_kJ_mol"])
        - float(complex_result["llfreeze"]["delta_G_1_to_0_kJ_mol"])
    )
    annihilation_binding = (
        float(
            solvent_result["annihilation_reweighted"]["delta_G_1_to_0_kJ_mol"]
        )
        - float(
            complex_result["annihilation_reweighted"]["delta_G_1_to_0_kJ_mol"]
        )
    )
    both_reliable = bool(
        complex_result["annihilation_reweighted"]["reweighting_reliable"]
        and solvent_result["annihilation_reweighted"]["reweighting_reliable"]
    )

    summary = {
        "complex_shift_annihilation_minus_llfreeze_kJ_mol": complex_shift,
        "solvent_shift_annihilation_minus_llfreeze_kJ_mol": solvent_shift,
        "common_shift_min_abs_same_sign_kJ_mol": (
            float(math.copysign(min(abs(complex_shift), abs(solvent_shift)), complex_shift))
            if complex_shift * solvent_shift > 0.0
            else 0.0
        ),
        "binding_charging_llfreeze_kJ_mol": current_binding,
        "binding_charging_annihilation_kJ_mol": annihilation_binding,
        "binding_shift_annihilation_minus_llfreeze_kJ_mol": binding_shift,
        "binding_shift_annihilation_minus_llfreeze_kcal_mol": binding_shift
        / KCAL_TO_KJ,
        "reference": {
            "complex_decharging_kJ_mol": float(args.reference_complex),
            "solvent_decharging_kJ_mol": float(args.reference_solvent),
            "binding_charging_kJ_mol": float(args.reference_solvent)
            - float(args.reference_complex),
        },
        "both_legs_reweighting_reliable": both_reliable,
        "interpretation": (
            "quantitative"
            if both_reliable
            else "qualitative_only_due_to_target_overlap_or_weight_concentration"
        ),
    }

    print("\n" + "=" * 76)
    print("binding charging 口径差")
    print("=" * 76)
    print(f"  complex shift ann−llfreeze = {complex_shift:+.4f} kJ/mol")
    print(f"  solvent shift ann−llfreeze = {solvent_shift:+.4f} kJ/mol")
    print(
        f"  对 ΔG_bind 的净移动       = {binding_shift:+.4f} kJ/mol "
        f"= {binding_shift / KCAL_TO_KJ:+.4f} kcal/mol"
    )
    print(
        f"  charging contribution: llfreeze {current_binding:+.4f} → "
        f"annihilation {annihilation_binding:+.4f} kJ/mol"
    )
    if not both_reliable:
        print("  ⚠️ 至少一条腿的零样本 target 覆盖不足：不得把该数字写入主结果。")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "method": "expanded_MBAR_sampled_llfreeze_plus_zero_sample_annihilation_targets",
        "complex": complex_result,
        "solvent": solvent_result,
        "summary": summary,
        "runtime": {
            "python": sys.version,
            "platform": py_platform.platform(),
            "numpy": np.__version__,
        },
    }
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"\n诊断结果已写入: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
