"""Closed-form refit of the outer-λ local residual from **finished stage-2 runs** (2026-10-17).

Why this exists (docs/design/PLAN_outer_center_selection_2026-10-17.md):
the in-run refit (`refit.closed_form_refit`, frames = `pre_equilibration.dcd`) samples λ_coul = 1,
λ_vdw = 1 and reweights to λ_coul = 0 targets; the weights collapse to one frame (Kish ESS = 1.0 in
all five SERT runs), so the fitted B is noise. Here the training frames are the stage-2 production
frames themselves, so every target state has overlap by construction. Workflow:
  1. run without outer (the run that showed the problem); stage-2 sidecars keep nearby water
     (`ibs_engine.STAGE2_COORDS_WATER_NM`)
  2. this module (CLI: tools/refit_outer_residual_from_stage2.py) fits B on one or more such runs
     of the same ligand and writes payload + weights + manifest
  3. rerun with `--outer-lambda-local-residual-ibs --outer-lambda-resource-manifest <manifest>`

Data per run and stage-2 window w (complex leg, `<run>/vanishing*` = segments):
  frames    sidecar frames (`dual_window_<w>_vdw_coords.npz`: all non-water atoms + per-frame water
            shell); a run whose sidecars have no water is refused — the plugin sees water edges at
            run time, so features without them are a different function
  weights   the production multi-segment solve of the window (`stage2_rounds_evidence.solve_window`
            on sampling_states / energies / each segment's production f_k), at the sidecar frames
  target    adjacent gaps of the window's target-state energies / kT
  A_k       `refit.a_k_schedule` of the window's λ_vdw (λ_coul = 0 in stage 2)
Features: centres = ligand atoms, environment = every other atom present in the frame, same envelope,
radial basis and (centre element, environment element) accumulation — the function itself is
`refit.frame_basis_features` (about 0.09 s per SERT frame). Loss and solver: `refit._quadratic_form` summed over windows, `refit._solve` with the
production smoothness prior; ridge by leave-one-run-out with ≥ 2 runs, otherwise by 3 contiguous time
blocks of every window.
Capacities: `max_edges = max_environment_atoms = n_ligand × max_neighbors_per_ligand (80)`. That is the
exact upper bound once the per-atom neighbour cap holds, so no margin is guessed; the CUDA kernel uses
these only as overflow ceilings (CudaLocalManyBodyResidualKernels.cpp:120/130), buffers are sized by the
candidate-list capacity, which must stay above max_edges (ligands up to 102 atoms).
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import refit as rf
from .refit import RefitError

KT_PER_KELVIN = 0.008314462618
MAX_NEIGHBORS_PER_LIGAND = 80
STAGE2_REFIT_CHAIN_VERSION = 1


def _log_to(log):
    return log if log is not None else (lambda _m: None)


def _segment_index(path: str) -> int:
    m = re.search(r"_(\d+)$", os.path.basename(path.rstrip(os.sep)))
    return int(m.group(1)) if m else 1


def _segment_checkpoint_dir(run_dir: str, n: int) -> str:
    # abfe_pipeline._stage2_segment_dirs: segment 1 = base dirs, segment n = <dir>_n / checkpoints/segment_n
    return os.path.join(run_dir, "checkpoints") if n == 1 else os.path.join(run_dir, "checkpoints", f"segment_{n}")


def stage2_capacities(n_ligand_atoms: int, candidate_list_capacity: int) -> dict[str, int]:
    bound = int(n_ligand_atoms) * MAX_NEIGHBORS_PER_LIGAND
    if bound >= int(candidate_list_capacity):
        raise RefitError(
            f"配体 {n_ligand_atoms} 个原子 × 每原子邻居上限 {MAX_NEIGHBORS_PER_LIGAND} = {bound} 条边，"
            f"不小于插件候选表容量 {candidate_list_capacity}（CUDA initialize 要求严格大于）"
        )
    return {"max_environment_atoms": bound, "max_edges": bound,
            "max_neighbors_per_ligand": MAX_NEIGHBORS_PER_LIGAND}


def frame_features(positions_angstrom, box_angstrom, ligand_local, types, n_types, config):
    """One frame: φ (n_types² × n_radial) and realized usage. Ligand = centres, all else = environment.

    Delegates to `refit.frame_basis_features` (pairwise fractional-coordinate minimum image, the same
    rule as the CUDA plugin). A faster unwrap-once + KD-tree variant was tried and dropped: on a skewed
    box it picks a different image for a few pairs near a face (one edge in 38 on the unit test).
    """

    import numpy as np

    ligand_local = np.asarray(sorted(int(i) for i in ligand_local), dtype=np.int64)
    environment = np.setdiff1d(np.arange(len(types)), ligand_local)
    outer = float(config.outer_cutoff_angstrom)
    n_radial = int(config.n_radial_basis)
    return rf.frame_basis_features(
        positions_angstrom=np.asarray(positions_angstrom, dtype=np.float64),
        box_angstrom=np.asarray(box_angstrom, dtype=np.float64),
        ligand_indices=ligand_local, environment_indices=environment,
        ligand_type_index=np.asarray(types)[ligand_local], environment_type_index=np.asarray(types),
        n_types=int(n_types), radial_centers=np.linspace(0.0, outer, n_radial),
        radial_width=outer / max(n_radial - 1, 1),
        inner_cutoff_angstrom=float(config.inner_cutoff_angstrom), outer_cutoff_angstrom=outer,
    )


def _run_identity(run_dir: str):
    """(topology, system, ligand ids, ligand atomic numbers, all atomic numbers) of one finished run."""

    import openmm
    from openmm import app

    from .openmm_plugin import topology_atomic_numbers

    topology = app.PDBxFile(os.path.join(run_dir, "topology.cif")).topology
    with open(os.path.join(run_dir, "system_native.xml")) as fh:
        system = openmm.XmlSerializer.deserialize(fh.read())
    with open(os.path.join(run_dir, "ligand_indices.json")) as fh:
        ligand = sorted(int(v) for v in json.load(fh)["ligand_indices"])
    z = [int(v) for v in topology_atomic_numbers(topology, system=system)]
    return topology, system, ligand, [z[i] for i in ligand], z


def load_run_windows(run_dir, *, ligand, atomic_numbers, vocabulary, config, temperature_kelvin,
                     frame_stride=1, log=None):
    """All stage-2 windows of one complex-leg run → list of per-window training blocks."""

    import numpy as np

    import stage2_rounds_evidence as sev

    _log = _log_to(log)
    kt = KT_PER_KELVIN * float(temperature_kelvin)
    z_to_type = {int(z): i for i, z in enumerate(vocabulary)}
    n_types = len(vocabulary)
    lig_set = set(ligand)
    dirs = sorted(glob.glob(os.path.join(run_dir, "vanishing*")), key=_segment_index)
    if not dirs:
        raise RefitError(f"{run_dir} 没有 stage2 产物（vanishing*）")
    windows = sorted({int(m.group(1)) for d in dirs for f in os.listdir(d)
                      for m in [re.match(r"dual_window_(\d+)_vdw_energies\.npy$", f)] if m})
    blocks = []
    realized = {"edges": 0, "environment_atoms": 0, "neighbors": 0}
    for w in windows:
        segs = [d for d in dirs if os.path.exists(os.path.join(d, f"dual_window_{w}_vdw_energies.npy"))]
        sampling, target, f_k = [], [], []
        for d in segs:
            sampling.append(np.load(os.path.join(d, f"dual_window_{w}_vdw_sampling_states.npy")).T)
            target.append(np.load(os.path.join(d, f"dual_window_{w}_vdw_energies.npy")))
            with open(os.path.join(_segment_checkpoint_dir(run_dir, _segment_index(d)),
                                   f"ibs_state_vdw_window_{w}.json")) as fh:
                state = json.load(fh)
            f_k.append(state.get("production_entry_f_k") or state.get("f_k"))
        solved = sev.solve_window(sampling, target, f_k, kt)
        weights = np.asarray(solved["weights"])
        u_reduced = np.hstack(target) / kt
        with open(os.path.join(segs[0], f"dual_window_{w}_vdw_convergence.json")) as fh:
            lam = [float(v) for v in json.load(fh)["lambdas_vdw"]]
        delta_a, _ = rf.a_k_schedule(lam)
        phi, gaps, log_imp, seg_of = [], [], [], []
        offset = 0
        for s, (d, t) in enumerate(zip(segs, target)):
            path = os.path.join(d, f"dual_window_{w}_vdw_coords.npz")
            if not os.path.exists(path):
                raise RefitError(f"{path} 不存在：这一段没有坐标旁车，帧和能量对不上")
            with np.load(path) as z:
                if "water_offsets" not in z.files:
                    raise RefitError(
                        f"{path} 没存水：插件运行时会算水边，用没水的帧拟出的是另一个函数。"
                        "请用带水旁车（ibs_engine.STAGE2_COORDS_WATER_NM > 0）的 run")
                fixed = z["atom_indices"]
                lig_pos = {int(g): i for i, g in enumerate(fixed) if int(g) in lig_set}
                if len(lig_pos) != len(ligand):
                    raise RefitError(f"{path} 的固定原子里配体不全")
                for j in range(0, len(z["idx"]), int(frame_stride)):
                    w_idx = z["water_idx"][z["water_offsets"][j]:z["water_offsets"][j + 1]]
                    w_xyz = z["water_xyz"][z["water_offsets"][j]:z["water_offsets"][j + 1]]
                    atoms = np.concatenate([fixed, w_idx])
                    xyz = np.concatenate([z["xyz"][j], w_xyz]) * 10.0
                    types = np.asarray([z_to_type[int(atomic_numbers[a])] for a in atoms], dtype=np.int64)
                    ligand_local = np.asarray([lig_pos[i] for i in ligand], dtype=np.int64)
                    row, used = frame_features(xyz, z["box"][j] * 10.0, ligand_local, types, n_types, config)
                    for k in realized:
                        realized[k] = max(realized[k], used[k])
                    col = offset + int(z["idx"][j])
                    phi.append(row)
                    gaps.append(np.diff(u_reduced[:, col]))
                    log_imp.append(np.log(np.maximum(weights[:, col], 1e-300)))
                    seg_of.append(s)
            offset += t.shape[1]
        li = np.asarray(log_imp)
        wn = np.exp(li - li.max(0))
        wn /= wn.sum(0)
        ess = 1.0 / (wn ** 2).sum(0)
        blocks.append({"window": w, "lambdas_vdw": lam, "features": np.asarray(phi),
                       "gaps": np.asarray(gaps), "log_importance": li, "delta_a": np.asarray(delta_a),
                       "segment": np.asarray(seg_of), "ess": ess})
        _log(f"  [stage2-refit] {os.path.basename(run_dir.rstrip(os.sep))} w{w}: {len(phi)} 帧，"
             f"目标态 Kish ESS {np.round(ess, 1).tolist()}")
    return blocks, realized


def _quad(blocks, select):
    """Σ_window _quadratic_form over the frames `select(block)` picks."""

    import numpy as np

    M = b = None
    c = 0.0
    for blk in blocks:
        subset = select(blk)
        if subset.size < 2:
            continue
        m_, b_, c_ = rf._quadratic_form(blk["features"], blk["gaps"], blk["log_importance"], blk["delta_a"], subset)
        if M is None:
            M, b = np.zeros_like(m_), np.zeros_like(b_)
        M += m_
        b += b_
        c += c_
    return M, b, c


def stage2_closed_form_refit(run_dirs: Sequence[str], *, temperature_kelvin: float = 300.0,
                             frame_stride: int = 1, ridge_grid=None, log=None) -> dict[str, Any]:
    """Fit on one or more finished complex-leg runs of the same ligand. Returns a result dict that
    `refit.write_refit_payload` accepts, plus `ligand_ids` / `reference_run` for the manifest."""

    import numpy as np

    from .openmm_plugin import FROZEN_CANDIDATE_LIST_CAPACITY
    from .softlift import derived_r1_config

    _log = _log_to(log)
    runs = [str(r) for r in run_dirs]
    if not runs:
        raise RefitError("至少给一个 run")
    idents = [_run_identity(r) for r in runs]
    lig_z = idents[0][3]
    for r, ident in zip(runs, idents):
        if ident[3] != lig_z:
            raise RefitError(f"{r} 的配体原子序数序列和 {runs[0]} 不同：权重绑配体，不能混训")
    vocabulary = tuple(sorted({z for ident in idents for z in ident[4]}))
    n_ligand = len(idents[0][2])
    capacities = stage2_capacities(n_ligand, FROZEN_CANDIDATE_LIST_CAPACITY)
    protocol_sha256 = hashlib.sha256(json.dumps({
        "chain": "local_residual.stage2_refit", "version": STAGE2_REFIT_CHAIN_VERSION,
        "frames": "stage2_sidecars_with_water_shell", "weights": "stage2_rounds_evidence.solve_window",
        "lambdas_coul": "0 (stage2)", "type_vocabulary": list(vocabulary), "n_ligand_atoms": n_ligand,
        "capacities": capacities, "A_definition": "sin_squared_pi_lambda_vdw",
        "regularizer": "second_difference_smoothness_prior_in_r",
        "ridge_selection": "leave_one_run_out" if len(runs) > 1 else "three_time_blocks_per_window",
    }, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = derived_r1_config(protocol_sha256, type_vocabulary=vocabulary, n_ligand_atoms=n_ligand, **capacities)

    per_run, realized = [], {"edges": 0, "environment_atoms": 0, "neighbors": 0}
    for r, ident in zip(runs, idents):
        blocks, used = load_run_windows(r, ligand=ident[2], atomic_numbers=ident[4], vocabulary=vocabulary,
                                        config=config, temperature_kelvin=temperature_kelvin,
                                        frame_stride=frame_stride, log=_log)
        per_run.append(blocks)
        for k in realized:
            realized[k] = max(realized[k], used[k])
    if realized["neighbors"] > MAX_NEIGHBORS_PER_LIGAND:
        raise RefitError(f"训练帧里单个配体原子的邻居数到了 {realized['neighbors']}，超过插件上限 "
                         f"{MAX_NEIGHBORS_PER_LIGAND}：运行时会 NEIGHBOR_OVERFLOW")
    _log(f"  [stage2-refit] 实测用量 {realized}；容量 {capacities}")

    if ridge_grid is None:
        ridge_grid = [10.0 ** p for p in range(-8, 5)]
    if len(runs) > 1:
        folds = [([blk for i, bl in enumerate(per_run) if i != k for blk in bl], per_run[k]) for k in range(len(runs))]
        all_frames = lambda blk: np.arange(len(blk["features"]))  # noqa: E731
        fold_pairs = [((tr, all_frames), (te, all_frames)) for tr, te in folds]
    else:
        blocks = per_run[0]

        def time_block(k):
            def pick(blk, k=k, keep=True):
                n = len(blk["features"])
                lo, hi = round(k * n / 3), round((k + 1) * n / 3)
                idx = np.arange(n)
                return idx[(idx >= lo) & (idx < hi)] if keep else idx[(idx < lo) | (idx >= hi)]
            return pick

        fold_pairs = [((blocks, lambda blk, p=time_block(k): p(blk, keep=False)),
                       (blocks, time_block(k))) for k in range(3)]
    omega = None
    scores = []
    for ridge in ridge_grid:
        held = []
        for (train_blocks, train_sel), (test_blocks, test_sel) in fold_pairs:
            M, b, _ = _quad(train_blocks, train_sel)
            if omega is None:
                omega = rf._smoothness_prior(len(vocabulary), int(config.n_radial_basis))
            theta = rf._solve(M, b, omega, ridge)
            Mt, bt, ct = _quad(test_blocks, test_sel)
            held.append(float(theta @ Mt @ theta + 2.0 * bt @ theta + ct) / ct)
        scores.append({"ridge": float(ridge), "mean_held_out_loss_ratio": float(np.mean(held)),
                       "per_fold_held_out_loss_ratio": held})
    best = min(scores, key=lambda s: s["mean_held_out_loss_ratio"])
    everything = [blk for bl in per_run for blk in bl]
    M, b, c = _quad(everything, lambda blk: np.arange(len(blk["features"])))
    theta = rf._solve(M, b, omega, best["ridge"])
    fitted = float(theta @ M @ theta + 2.0 * b @ theta + c)
    features = np.vstack([blk["features"] for blk in everything])
    raw = features @ theta
    offset = float(np.mean(raw))
    centered = raw - offset
    saturated = float(np.mean(np.abs(centered) > float(config.b_max_reduced)))
    _log(f"  [stage2-refit] 岭 {best['ridge']:.0e}；留出 loss 比 {best['mean_held_out_loss_ratio']:.3f}"
         f"（1 = 不比 B=0 好）；全量相对改善 {(c - fitted) / c:.3f}；B 量程 [{centered.min():.2f}, "
         f"{centered.max():.2f}] kT，b_max {config.b_max_reduced:.1f}，tanh 咬合 {saturated:.1%}")
    return {
        "theta": theta,
        "features": features,
        "raw_offset_reduced": offset,
        "linear_basis_reduced": centered,
        "tanh_saturated_fraction": saturated,
        "config": config,
        "vocabulary": vocabulary,
        "capacities": capacities,
        "realized_capacity_usage": realized,
        "fit_report": {
            "ridge_selected": best["ridge"], "ridge_grid_scores": scores,
            "loss_without_basis": float(c), "loss_with_basis": fitted,
            "relative_improvement": (c - fitted) / c if c > 0 else 0.0,
            "held_out_loss_ratio": best["mean_held_out_loss_ratio"],
            "n_frames": int(features.shape[0]), "n_parameters": int(theta.size),
        },
        "n_frames": int(features.shape[0]),
        "trajectory_path": "stage2 sidecars of: " + ", ".join(runs),
        "protocol_sha256": protocol_sha256,
        "ligand_ids": idents[0][2],
        "reference_run": runs[0],
        "target_state_ess": {r: {int(blk["window"]): [float(v) for v in blk["ess"]] for blk in bl}
                             for r, bl in zip(runs, per_run)},
    }
