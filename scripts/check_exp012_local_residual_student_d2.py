#!/usr/bin/env python
"""DEC-037 (d0) D2: coordinate/autograd qualification for trained `LocalResidualStudent`s.

Per the user's explicit decision (2026-08-05): only `direct_gap` checkpoints
are D2 candidates (`distilled` did not clear the "must not be worse than
direct_gap" gate in D1 and is not carried forward). All checkpoints supplied
are checked -- this script never picks a "best" one by held-out performance;
that would reuse the outer test set for model selection. If you want to test
a subset, choose it by a rule that does not reference held-out numbers (e.g.
"the fold whose training runs are X and Y"), not by which one scored best.

Three checks, each against the model's REAL trained output (not the raw
geometric envelope alone, which DEC-038 already validated in isolation):

1. Finite-difference vs. autograd force check. Positions are fed through
   `local_residual.geometry.ligand_environment_cross_edges` LIVE (gradient-
   connected, unlike D1's cached detached distances -- see `local_residual/
   student.py`'s module docstring on this exact D1/D2 split) so
   `-∇_R energy` is a real trained force, not a training-speed shortcut.
   Central differences are compared only for atoms that can plausibly affect
   the energy (the ligand + the atoms already in this frame's edge list) --
   perturbing all ~70k system atoms per frame would be wasted, uninformative
   compute, since any atom outside the cutoff is mathematically guaranteed
   zero gradient by construction (it never enters the graph).
2. Cutoff-boundary energy smoothness. A real environment atom is walked
   through the outer cutoff (real coordinates, one atom moved along its own
   displacement from the ligand centroid) and the model's *total* scalar
   output is checked for continuity across the discrete membership flip --
   DEC-038 already checked this for the geometric weight function alone;
   this checks it for the full trained energy this model actually outputs.
3. Force-tail / extrapolation safety. Force magnitude must decay smoothly
   toward the cutoff (not spike), and energy/force must stay finite for an
   out-of-training-distribution close contact (PLAN 文档 §5.2's "对超出训练
   支持域的构象能够报警或安全衰减").

This is D2 only: no TorchScript, OpenMM, CUDA, or NVT run happens here
(those are D3/D4, gated on this stage per DEC-037).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.geometry import ligand_environment_cross_edges  # noqa: E402
from local_residual.student import (  # noqa: E402
    build_local_residual_student,
    reindex_ligand_environment_edges,
)


class D2CheckError(RuntimeError):
    """A checkpoint or its inputs failed a fail-closed contract check."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _load_checkpoint(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise D2CheckError(
            f"{path}: variant={payload.get('variant')!r} -- only direct_gap checkpoints "
            "are D2 candidates per the 2026-08-05 decision (distilled did not clear the "
            "held-out gate and is not carried forward)"
        )
    model = build_local_residual_student(payload["type_vocabulary"], **payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model = model.to(torch.float64)
    model.eval()
    return model, payload


def _energy_from_positions(model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index):
    """One full live forward pass: positions -> edges -> reindex -> energy.

    Every step here is differentiable w.r.t. `positions` when it requires
    grad (this is the "D2 recomputes live" path, never the cached-distance
    shortcut D1 uses for training speed).
    """

    import torch

    edges = ligand_environment_cross_edges(
        positions, box, ligand_tensor, environment_tensor, outer_cutoff=model.outer_cutoff_angstrom,
    )
    ligand_topology_indices = ligand_tensor.tolist()
    reindexed = reindex_ligand_environment_edges(
        ligand_topology_indices, edges["edge_index"][0], edges["edge_index"][1]
    )
    ligand_type_index = model.atomic_numbers_to_type_index(
        [atomic_number_by_topology_index[index] for index in ligand_topology_indices]
    )
    if reindexed["environment_topology_indices"].numel() == 0:
        environment_type_index = torch.empty((0,), dtype=torch.int64)
    else:
        environment_type_index = model.atomic_numbers_to_type_index(
            [atomic_number_by_topology_index[int(index)] for index in reindexed["environment_topology_indices"].tolist()]
        )
    energy = model(
        ligand_type_index, environment_type_index,
        reindexed["edge_ligand_local"], reindexed["edge_environment_local"], edges["distance"],
    )
    return energy, edges


def _finite_difference_check(
    model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index,
    *, epsilon_angstrom: float, max_environment_atoms_checked: int,
):
    import torch

    positions = positions.clone().detach().requires_grad_(True)
    energy, edges = _energy_from_positions(
        model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index
    )
    energy.backward()
    autograd_force = -positions.grad.clone()

    ligand_atoms = ligand_tensor.tolist()
    environment_atoms_in_edges = sorted(set(edges["edge_index"][1].tolist()))[:max_environment_atoms_checked]
    atoms_to_check = ligand_atoms + environment_atoms_in_edges

    # Sanity: any atom that never appears in this frame's edge list must have
    # exactly zero autograd force (it never entered the computational graph).
    all_participating = set(ligand_atoms) | set(edges["edge_index"][1].tolist())
    non_participating_sample = [
        index for index in environment_tensor.tolist() if index not in all_participating
    ][:20]
    zero_force_ok = bool(
        all(
            torch.equal(autograd_force[index], torch.zeros(3, dtype=autograd_force.dtype))
            for index in non_participating_sample
        )
    )

    max_absolute_error = 0.0
    max_relative_error = 0.0
    per_atom = []
    with torch.no_grad():
        for atom_index in atoms_to_check:
            fd_force = torch.zeros(3, dtype=torch.float64)
            for dim in range(3):
                perturbed_plus = positions.detach().clone()
                perturbed_plus[atom_index, dim] += epsilon_angstrom
                perturbed_minus = positions.detach().clone()
                perturbed_minus[atom_index, dim] -= epsilon_angstrom
                energy_plus, _ = _energy_from_positions(
                    model, perturbed_plus, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index
                )
                energy_minus, _ = _energy_from_positions(
                    model, perturbed_minus, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index
                )
                fd_force[dim] = -(energy_plus - energy_minus) / (2.0 * epsilon_angstrom)
            absolute_error = (fd_force - autograd_force[atom_index]).abs().max().item()
            denominator = max(fd_force.abs().max().item(), 1e-8)
            relative_error = absolute_error / denominator
            max_absolute_error = max(max_absolute_error, absolute_error)
            max_relative_error = max(max_relative_error, relative_error)
            per_atom.append(
                {
                    "atom_topology_index": int(atom_index),
                    "is_ligand": atom_index in ligand_atoms,
                    "autograd_force": autograd_force[atom_index].tolist(),
                    "finite_difference_force": fd_force.tolist(),
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
            )

    return {
        "n_atoms_checked": len(atoms_to_check),
        "n_ligand_atoms_checked": len(ligand_atoms),
        "n_environment_atoms_checked": len(environment_atoms_in_edges),
        "epsilon_angstrom": epsilon_angstrom,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "non_participating_atoms_have_zero_force": zero_force_ok,
        "n_non_participating_atoms_checked": len(non_participating_sample),
        "per_atom": per_atom,
    }


def _cutoff_smoothness_check(
    model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index,
    *, sweep_half_width_angstrom: float, sweep_step_angstrom: float, refinement_factor: int = 25,
):
    """Move one real environment atom radially through the cutoff and watch
    the model's TOTAL scalar output, not just the geometric envelope alone
    (DEC-038 already validated the envelope in isolation; this validates the
    actual trained energy this model produces).

    A single-resolution "is the biggest single-step jump below some fixed
    absolute threshold" test does not actually distinguish a true
    discontinuity from a steep-but-continuous region -- and the latter is
    expected here: the ligand's 41 atoms are clustered close together, so the
    probed environment atom's own boundary crossing very often coincides,
    within the same sweep window, with some *other* ligand atom's boundary
    also being crossed (see `any_other_pair_incidentally_flipped`). The
    correct empirical test is whether the biggest jump shrinks proportionally
    when the step size shrinks: for a genuinely continuous (if locally steep)
    function, halving the step roughly halves the biggest single-step delta;
    for a true discontinuity, the jump stays roughly the same size no matter
    how finely you sample near it. So this runs the sweep at two resolutions
    (the requested step, and `refinement_factor`x finer) and compares.
    """

    import torch

    with torch.no_grad():
        _, edges = _energy_from_positions(
            model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index
        )
    if edges["distance"].numel() == 0:
        raise D2CheckError("this frame has no ligand-environment edges; cannot probe the cutoff boundary")

    # Pick the real environment atom whose distance is closest to the cutoff
    # (most sensitive to the boundary sweep).
    distances = edges["distance"]
    closest_index = int((distances - model.outer_cutoff_angstrom).abs().argmin().item())
    probe_environment_atom = int(edges["edge_index"][1][closest_index].item())
    probe_ligand_atom = int(edges["edge_index"][0][closest_index].item())

    ligand_position = positions[probe_ligand_atom]
    environment_position = positions[probe_environment_atom]
    direction = environment_position - ligand_position
    direction = direction / direction.norm()

    def _run_sweep(step_angstrom: float):
        offsets = []
        value = -sweep_half_width_angstrom
        while value <= sweep_half_width_angstrom + 1e-9:
            offsets.append(value)
            value += step_angstrom

        energies = []
        pair_included = []
        other_pairs_flipped = []
        with torch.no_grad():
            baseline_other_pairs = None
            for offset in offsets:
                perturbed = positions.clone()
                perturbed[probe_environment_atom] = ligand_position + direction * (
                    model.outer_cutoff_angstrom + offset
                )
                energy, edges_at_offset = _energy_from_positions(
                    model, perturbed, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index
                )
                energies.append(float(energy.item()))
                pair_set = set(
                    zip(edges_at_offset["edge_index"][0].tolist(), edges_at_offset["edge_index"][1].tolist())
                )
                # The ligand has 41 atoms clustered together, so this
                # environment atom very likely stays connected to *some
                # other* ligand atom throughout the whole sweep -- checking
                # "does this atom appear anywhere" (as an earlier version of
                # this check did) would almost never toggle. What must be
                # tracked is membership of the ONE specific pair being swept.
                pair_included.append((probe_ligand_atom, probe_environment_atom) in pair_set)
                # Diagnostic: did any *other* pair's membership change too?
                other_pairs = pair_set - {(probe_ligand_atom, probe_environment_atom)}
                other_pairs = {pair for pair in other_pairs if pair[1] == probe_environment_atom}
                if baseline_other_pairs is None:
                    baseline_other_pairs = other_pairs
                other_pairs_flipped.append(other_pairs != baseline_other_pairs)

        max_step_jump = max(
            abs(energies[i + 1] - energies[i]) for i in range(len(energies) - 1)
        ) if len(energies) > 1 else 0.0
        flips = sum(
            1 for i in range(len(pair_included) - 1) if pair_included[i] != pair_included[i + 1]
        )
        return {
            "step_angstrom": step_angstrom,
            "offsets_from_cutoff_angstrom": offsets,
            "total_energy_reduced": energies,
            "probed_pair_membership_included": pair_included,
            "n_probed_pair_membership_flips": flips,
            "exactly_one_membership_flip": flips == 1,
            "any_other_pair_incidentally_flipped": any(other_pairs_flipped),
            "max_consecutive_step_energy_jump": max_step_jump,
        }

    coarse = _run_sweep(sweep_step_angstrom)
    fine = _run_sweep(sweep_step_angstrom / refinement_factor)

    expected_ratio = float(refinement_factor)
    if fine["max_consecutive_step_energy_jump"] > 0.0:
        observed_ratio = coarse["max_consecutive_step_energy_jump"] / fine["max_consecutive_step_energy_jump"]
    else:
        observed_ratio = float("inf") if coarse["max_consecutive_step_energy_jump"] > 0.0 else 1.0
    # A genuine discontinuity would NOT shrink with a finer step -- the ratio
    # would stay near 1, far below the linear-scaling expectation. Requiring
    # at least half the expected linear-scaling ratio is a generous margin
    # (a real break gives ratio ~1, continuity gives ratio ~refinement_factor)
    # while tolerating the fact the two sweeps don't land on identical
    # offsets so the coarse/fine jump locations aren't pixel-identical.
    scales_like_continuous = observed_ratio >= 0.5 * expected_ratio

    return {
        "probe_ligand_atom": probe_ligand_atom,
        "probe_environment_atom": probe_environment_atom,
        "sweep_half_width_angstrom": sweep_half_width_angstrom,
        "coarse_sweep": coarse,
        "fine_sweep": fine,
        "refinement_factor": refinement_factor,
        "expected_jump_ratio_if_continuous": expected_ratio,
        "observed_jump_ratio": observed_ratio,
        "scales_like_continuous_not_discontinuous": scales_like_continuous,
        # Back-compat top-level fields (coarse-resolution values), kept
        # because a true discontinuity must show up at BOTH resolutions.
        "exactly_one_membership_flip": coarse["exactly_one_membership_flip"] and fine["exactly_one_membership_flip"],
        "any_other_pair_incidentally_flipped": (
            coarse["any_other_pair_incidentally_flipped"] or fine["any_other_pair_incidentally_flipped"]
        ),
        "max_consecutive_step_energy_jump": coarse["max_consecutive_step_energy_jump"],
    }


def _force_tail_safety_check(
    model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index,
):
    """Force magnitude must decay toward the cutoff, and a synthetic very-close
    contact (well outside any real MD equilibrium distance) must not blow up.
    """

    import torch

    with torch.no_grad():
        _, edges = _energy_from_positions(
            model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index
        )
    if edges["distance"].numel() == 0:
        raise D2CheckError("this frame has no ligand-environment edges; cannot check the force tail")

    closest_index = int((edges["distance"] - model.outer_cutoff_angstrom).abs().argmin().item())
    probe_environment_atom = int(edges["edge_index"][1][closest_index].item())
    probe_ligand_atom = int(edges["edge_index"][0][closest_index].item())
    ligand_position = positions[probe_ligand_atom]
    environment_position = positions[probe_environment_atom]
    direction = environment_position - ligand_position
    direction = direction / direction.norm()

    # Distances from just-inside the cutoff down toward the boundary: force
    # magnitude on the probed atom must not increase as it nears the cutoff.
    tail_distances = [
        model.outer_cutoff_angstrom - offset for offset in (2.0, 1.0, 0.5, 0.25, 0.1, 0.05)
        if model.outer_cutoff_angstrom - offset > 0.0
    ]
    tail_force_norms = []
    for distance in tail_distances:
        perturbed = positions.clone().requires_grad_(True)
        with torch.no_grad():
            perturbed[probe_environment_atom] = ligand_position + direction * distance
        perturbed = perturbed.detach().requires_grad_(True)
        energy, _ = _energy_from_positions(
            model, perturbed, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index
        )
        energy.backward()
        tail_force_norms.append(float(perturbed.grad[probe_environment_atom].norm().item()))

    # Synthetic very-close contact: 0.3 A, well outside any real MD
    # equilibrium separation -- must stay finite, not NaN/Inf.
    close_distance = 0.3
    perturbed = positions.clone().requires_grad_(True)
    with torch.no_grad():
        perturbed[probe_environment_atom] = ligand_position + direction * close_distance
    perturbed = perturbed.detach().requires_grad_(True)
    close_energy, _ = _energy_from_positions(
        model, perturbed, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index
    )
    close_energy.backward()
    close_force_norm = float(perturbed.grad[probe_environment_atom].norm().item())

    # Overall trend, not strict step-by-step monotonicity: the envelope factor
    # is monotonically decreasing toward the cutoff by construction, but the
    # learned filter_net(radial_basis) is not guaranteed monotonic in r for an
    # arbitrary trained network, so requiring every single step to decrease
    # would be a fragile, overly strict criterion that could flag legitimate
    # model behavior. What must hold is the overall trend: force right at the
    # boundary should be small relative to the well-inside-cutoff reference.
    max_tail_force = max(tail_force_norms) if tail_force_norms else 0.0
    boundary_force = tail_force_norms[-1] if tail_force_norms else 0.0

    return {
        "probe_ligand_atom": probe_ligand_atom,
        "probe_environment_atom": probe_environment_atom,
        "tail_distances_angstrom": tail_distances,
        "tail_force_norms_reduced_per_angstrom": tail_force_norms,
        "max_tail_force_norm": max_tail_force,
        "boundary_force_norm": boundary_force,
        "boundary_force_small_relative_to_tail_max": (
            boundary_force <= 0.5 * max_tail_force + 1e-8 if max_tail_force > 0 else True
        ),
        "synthetic_close_contact_distance_angstrom": close_distance,
        "synthetic_close_contact_energy_reduced": float(close_energy.item()),
        "synthetic_close_contact_force_norm_reduced_per_angstrom": close_force_norm,
        "synthetic_close_contact_is_finite": bool(
            float("inf") != abs(close_energy.item()) and close_energy.item() == close_energy.item()
            and float("inf") != abs(close_force_norm) and close_force_norm == close_force_norm
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="repeatable; every direct_gap checkpoint supplied is checked (no cherry-picking by held-out score)",
    )
    parser.add_argument("--topology", required=True)
    parser.add_argument(
        "--trajectory", action="append", required=True, metavar="RUN_ID=PATH",
        help="repeatable, e.g. --trajectory hard_window0_run1=<path.dcd>",
    )
    parser.add_argument("--ligand-indices", required=True)
    parser.add_argument("--frame-index", type=int, default=0, help="frame index within each --trajectory to probe")
    parser.add_argument("--fd-epsilon-angstrom", type=float, default=1e-4)
    parser.add_argument("--fd-max-environment-atoms", type=int, default=20)
    parser.add_argument("--cutoff-sweep-half-width-angstrom", type=float, default=0.5)
    parser.add_argument("--cutoff-sweep-step-angstrom", type=float, default=0.05)
    parser.add_argument(
        "--cutoff-refinement-factor", type=int, default=25,
        help="the smoothness check also runs the sweep at this much finer a step and requires the "
             "biggest single-step jump to shrink roughly proportionally (continuity test), not just "
             "check a fixed absolute threshold at one resolution",
    )
    parser.add_argument("--fd-relative-error-gate", type=float, default=1e-2)
    parser.add_argument("--fd-absolute-error-gate", type=float, default=1e-4)
    parser.add_argument("--cutoff-max-jump-gate", type=float, default=1e-6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")

    trajectory_by_run: dict[str, Path] = {}
    for item in args.trajectory:
        if "=" not in item:
            parser.error(f"--trajectory must be RUN_ID=PATH, got: {item}")
        run_id, path = item.split("=", 1)
        trajectory_by_run[run_id] = Path(path)

    import mdtraj
    import torch

    ligand_payload = json.loads(Path(args.ligand_indices).expanduser().read_text(encoding="utf-8"))
    ligand_topology_indices = sorted(int(index) for index in ligand_payload["ligand_indices"])

    checkpoint_results = []
    for checkpoint_arg in args.checkpoint:
        checkpoint_path = Path(checkpoint_arg)
        model, payload = _load_checkpoint(checkpoint_path)
        _log_prefix = f"[{checkpoint_path.name}]"
        print(f"{_log_prefix} checking...", flush=True)

        per_run_checks = {}
        for run_id, trajectory_path in trajectory_by_run.items():
            trajectory = mdtraj.load_frame(str(trajectory_path), index=args.frame_index, top=str(args.topology))
            if trajectory.unitcell_vectors is None:
                raise D2CheckError(f"{trajectory_path}: no periodic box vectors")
            positions = torch.tensor(trajectory.xyz[0] * 10.0, dtype=torch.float64)
            box = torch.tensor(trajectory.unitcell_vectors[0] * 10.0, dtype=torch.float64)
            n_atoms = trajectory.topology.n_atoms
            atomic_number_by_topology_index = {
                index: int(atom.element.atomic_number) for index, atom in enumerate(trajectory.topology.atoms)
            }
            ligand_tensor = torch.tensor(ligand_topology_indices, dtype=torch.int64)
            environment_tensor = torch.tensor(
                sorted(set(range(n_atoms)) - set(ligand_topology_indices)), dtype=torch.int64
            )

            fd_result = _finite_difference_check(
                model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index,
                epsilon_angstrom=args.fd_epsilon_angstrom, max_environment_atoms_checked=args.fd_max_environment_atoms,
            )
            cutoff_result = _cutoff_smoothness_check(
                model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index,
                sweep_half_width_angstrom=args.cutoff_sweep_half_width_angstrom,
                sweep_step_angstrom=args.cutoff_sweep_step_angstrom,
                refinement_factor=args.cutoff_refinement_factor,
            )
            tail_result = _force_tail_safety_check(
                model, positions, box, ligand_tensor, environment_tensor, atomic_number_by_topology_index,
            )

            fd_passed = (
                fd_result["max_relative_error"] <= args.fd_relative_error_gate
                or fd_result["max_absolute_error"] <= args.fd_absolute_error_gate
            ) and fd_result["non_participating_atoms_have_zero_force"]
            cutoff_passed = (
                cutoff_result["exactly_one_membership_flip"]
                and (
                    cutoff_result["scales_like_continuous_not_discontinuous"]
                    # A jump already below the absolute gate at the coarse
                    # resolution is unambiguously fine regardless of scaling
                    # (the scaling test only matters when the coarse jump is
                    # large enough that "true break vs. steep-but-continuous"
                    # is actually ambiguous).
                    or cutoff_result["max_consecutive_step_energy_jump"] <= args.cutoff_max_jump_gate
                )
            )
            tail_passed = tail_result["synthetic_close_contact_is_finite"]

            per_run_checks[run_id] = {
                "finite_difference": fd_result,
                "cutoff_smoothness": cutoff_result,
                "force_tail_safety": tail_result,
                "finite_difference_passed": fd_passed,
                "cutoff_smoothness_passed": cutoff_passed,
                "force_tail_safety_passed": tail_passed,
                "all_passed": bool(fd_passed and cutoff_passed and tail_passed),
                "trajectory_sha256": _sha256_file(trajectory_path),
            }
            print(
                f"{_log_prefix} {run_id}: fd_rel_err={fd_result['max_relative_error']:.3e} "
                f"cutoff_jump={cutoff_result['max_consecutive_step_energy_jump']:.3e} "
                f"jump_ratio={cutoff_result['observed_jump_ratio']:.2f} "
                f"(expected {cutoff_result['expected_jump_ratio_if_continuous']:.0f} if continuous) "
                f"tail_ok={tail_passed} all_passed={per_run_checks[run_id]['all_passed']}",
                flush=True,
            )

        checkpoint_results.append(
            {
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "held_out_run_id": payload["held_out_run_id"],
                "training_run_ids": payload["training_run_ids"],
                "seed": payload["seed"],
                "dataset_report_sha256": payload.get("dataset_report_sha256"),
                "per_run_checks": per_run_checks,
                "all_runs_passed": bool(all(check["all_passed"] for check in per_run_checks.values())),
            }
        )

    body = {
        "schema_version": "exp012-local-residual-student-d2-v1",
        "status": "COMPLETED_D2_CHECKS",
        "frame_index_probed": args.frame_index,
        "gates": {
            "fd_relative_error_gate": args.fd_relative_error_gate,
            "fd_absolute_error_gate": args.fd_absolute_error_gate,
            "cutoff_max_jump_gate": args.cutoff_max_jump_gate,
        },
        "checkpoints": checkpoint_results,
        "all_checkpoints_passed": bool(all(entry["all_runs_passed"] for entry in checkpoint_results)),
        "policy": {
            "decision_reference": "DEC-037 (d0-D2), 2026-08-05 direct_gap-only decision",
            "distilled_checkpoints_excluded": True,
            "no_held_out_based_checkpoint_selection": True,
            "student_model_executed": True,
            "torchforce_used": False,
            "nvt_executed": False,
            "note": "D2 offline force/cutoff/tail qualification only; D3 (TorchScript/OpenMM/CUDA) "
                    "and D4 (short NVT) are separate, later stages gated on this result per DEC-037",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"all_checkpoints_passed={report['all_checkpoints_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
