#!/usr/bin/env python
"""DEC-030(d0) item 1 real-data smoke: student online-environment funnel.

DEC-037 froze the `LocalResidualStudent` design contract's items 2-5 but left
item 1 -- "online dynamic environment representation" -- as the one question
that must be answered before any network/code choice. The proposed answer:
recompute a fresh ligand-environment cutoff funnel every step (no persistent
water identity, no fixed manifest), using two primitives that already exist
but have never been run on real data or cross-checked against the teacher:

- ``local_residual.geometry.ligand_environment_cross_edges`` -- the candidate
  funnel itself (an independently written minimum-image cutoff pass, distinct
  from the teacher's ``local_residual.mace_graph``/``teacher_graph`` code path).
- ``local_residual.geometry.quintic_c2_cutoff`` -- the smooth per-pair contact
  weight that must replace a hard {0,1} membership gate on the energy.

This script is the "2" in the user's "2 -> 1" directive: it must actually run
on real Atenolol frames and produce a real go/no-go verdict before DEC-038
gets written. It does not train anything, does not compute the gap-variance
loss, does not wrap a TorchForce, and does not run any OpenMM integrator --
only ``local_residual.geometry``/``mace_graph``/``teacher_graph`` are used.

Two independent things are checked against real coordinates:

1. Does the funnel's ligand<->environment edge set (nodes, pairs, periodic
   unit shifts) exactly match the teacher's already-audited two-hop closure's
   ligand<->hop-1 subset (``local_residual.teacher_graph.compute_canonical_graph_membership``)?
   This is the same class of question that produced a real CPU/CUDA
   disagreement before (DEC-025/026) and a real fixed-manifold-size blowup
   before (DEC-031/032) -- both times because a plausible-sounding
   code-reading conclusion did not survive a real run.
2. Does discrete candidate membership near the cutoff boundary decouple
   cleanly from a smooth (non-binary) per-pair energy weight, so that a real
   environment atom drifting across the cutoff never produces an energy
   discontinuity?

This report's boolean fields are a real gate for DEC-038, unlike the earlier
``COMPARISON_ONLY_NOT_A_GATE`` smokes -- a failing check here is new
information about the online-environment design, not something to be patched
around silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.geometry import (  # noqa: E402
    ligand_environment_cross_edges,
    minimum_image_displacement,
    quintic_c2_cutoff,
)
from local_residual.mace_graph import topology_n_hop_closure  # noqa: E402
from local_residual.teacher_graph import compute_canonical_graph_membership  # noqa: E402


def _sha(path: str | Path) -> str:
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


def _unit_shift(source_positions, target_positions, cell):
    """Independent re-derivation of the integer periodic-image shift.

    Both ``local_residual.mace_graph`` and ``local_residual.geometry`` wrap a
    displacement the same way (``fractional -= round(fractional)``); this is
    that same formula, used only to convert ``geometry``'s wrapped
    displacement into the same discrete integer-shift representation the
    teacher's edges already carry, so the two can be compared pair-by-pair.
    """

    import torch

    raw = target_positions - source_positions
    fractional = torch.linalg.solve(cell.T, raw.reshape(-1, 3).T).T
    return (-torch.round(fractional)).reshape(raw.shape)


def _min_distance_to_ligand(positions, cell, ligand_indices, environment_indices, *, chunk: int = 2000):
    """Per-environment-atom minimum distance to any ligand atom, chunked."""

    import torch

    ligand_positions = positions[ligand_indices]
    pieces = []
    for start in range(0, int(environment_indices.numel()), chunk):
        block = environment_indices[start : start + chunk]
        block_positions = positions[block]
        source = ligand_positions[:, None, :].expand(-1, block_positions.shape[0], -1).reshape(-1, 3)
        target = block_positions[None, :, :].expand(ligand_positions.shape[0], -1, -1).reshape(-1, 3)
        displacement = minimum_image_displacement(source, target, cell).reshape(
            ligand_positions.shape[0], block_positions.shape[0], 3
        )
        pieces.append(torch.linalg.vector_norm(displacement, dim=-1).min(dim=0).values)
    return torch.cat(pieces) if pieces else positions.new_empty((0,))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="label only, e.g. hard_window0_run1")
    parser.add_argument("--topology", required=True, help="mdtraj-compatible topology")
    parser.add_argument("--trajectory", required=True, help="mdtraj-compatible frame source")
    parser.add_argument("--ligand-indices", required=True, help="JSON file with a ligand_indices array")
    parser.add_argument("--frame-index", type=int, required=True)
    parser.add_argument("--edge-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--interaction-layers", type=int, required=True)
    parser.add_argument(
        "--boundary-inner-cutoff-angstrom", type=float,
        help="smoke-only quintic_c2_cutoff inner radius; default is outer_cutoff - 1.0",
    )
    parser.add_argument(
        "--boundary-probe-count", type=int, default=10,
        help="how many real environment atoms closest to the cutoff boundary to check",
    )
    parser.add_argument(
        "--expected-hop1-count", type=int,
        help="optional external cross-check against a pre-registered teacher hop==1 atom count",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.edge_cutoff_angstrom != 5.0:
        parser.error(
            "this smoke is scoped to the derived_5a candidate only "
            f"(--edge-cutoff-angstrom 5.0); got {args.edge_cutoff_angstrom}"
        )
    if args.interaction_layers != 2:
        parser.error("this smoke is scoped to interaction_layers=2 only")
    inner_cutoff = (
        args.boundary_inner_cutoff_angstrom
        if args.boundary_inner_cutoff_angstrom is not None
        else args.edge_cutoff_angstrom - 1.0
    )
    if not (0.0 < inner_cutoff < args.edge_cutoff_angstrom):
        parser.error("--boundary-inner-cutoff-angstrom must satisfy 0 < inner < outer_cutoff")
    if args.boundary_probe_count < 1:
        parser.error("--boundary-probe-count must be positive")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")

    import mdtraj
    import torch

    started = time.perf_counter()
    cutoff = float(args.edge_cutoff_angstrom)
    layers = int(args.interaction_layers)

    ligand_payload = json.loads(Path(args.ligand_indices).expanduser().read_text(encoding="utf-8"))
    ligand_indices_raw = ligand_payload.get("ligand_indices")
    if not isinstance(ligand_indices_raw, list) or not ligand_indices_raw:
        raise RuntimeError("--ligand-indices JSON must contain a non-empty ligand_indices array")
    ligand_indices = sorted(int(index) for index in ligand_indices_raw)

    trajectory = mdtraj.load_frame(args.trajectory, index=args.frame_index, top=args.topology)
    if trajectory.n_frames != 1 or trajectory.unitcell_vectors is None:
        raise RuntimeError("frame source must provide exactly one frame with triclinic cell vectors")
    positions = torch.tensor(trajectory.xyz[0] * 10.0, dtype=torch.float64)
    cell = torch.tensor(trajectory.unitcell_vectors[0] * 10.0, dtype=torch.float64)
    atom_count = int(positions.shape[0])
    if ligand_indices[-1] >= atom_count or ligand_indices[0] < 0:
        raise RuntimeError("ligand_indices is out of range for this frame's topology")

    ligand_tensor = torch.tensor(ligand_indices, dtype=torch.int64)
    environment_tensor = torch.tensor(
        sorted(set(range(atom_count)) - set(ligand_indices)), dtype=torch.int64
    )

    # -- Teacher reference (already-audited canonical membership) --------
    membership = compute_canonical_graph_membership(
        positions, cell,
        ligand_indices=ligand_indices, edge_cutoff_angstrom=cutoff, interaction_layers=layers,
    )
    closure, minimum_hop = topology_n_hop_closure(
        positions, cell, ligand_indices, edge_cutoff_angstrom=cutoff, interaction_layers=layers,
    )
    reconstructed_closure = torch.zeros(atom_count, dtype=torch.bool)
    reconstructed_closure[membership["topology_index"]] = True
    teacher_internal_consistency_ok = bool(torch.equal(closure, reconstructed_closure)) and (
        int(membership["node_count"]) == int(closure.sum().item())
    )

    topology_index = membership["topology_index"]
    hop_at_selected = minimum_hop[topology_index]
    reconstructed_hop_counts = [
        int((hop_at_selected == layer).sum().item()) for layer in range(layers + 1)
    ]
    teacher_internal_consistency_ok = teacher_internal_consistency_ok and (
        reconstructed_hop_counts == list(membership["hop_counts_by_layer"])
    )

    teacher_hop1_topology = sorted(
        int(value) for value in topology_index[hop_at_selected == 1].tolist()
    )
    teacher_ligand_set = sorted(
        int(value) for value in topology_index[membership["ligand_mask"]].tolist()
    )
    ligand_set_matches = teacher_ligand_set == ligand_indices

    edge_index_local = membership["edge_index"]
    ligand_mask_local = membership["ligand_mask"]
    sender_is_ligand = ligand_mask_local[edge_index_local[0]]
    receiver_is_environment = ~ligand_mask_local[edge_index_local[1]]
    keep = sender_is_ligand & receiver_is_environment
    teacher_ligand_topo = topology_index[edge_index_local[0][keep]]
    teacher_env_topo = topology_index[edge_index_local[1][keep]]
    teacher_shift = membership["unit_shifts"][keep].round().to(torch.int64)
    teacher_pairs = {
        (int(teacher_ligand_topo[i]), int(teacher_env_topo[i])): tuple(teacher_shift[i].tolist())
        for i in range(int(teacher_ligand_topo.numel()))
    }

    # -- Student funnel (independent implementation under test) ----------
    funnel = ligand_environment_cross_edges(
        positions, cell, ligand_tensor, environment_tensor, outer_cutoff=cutoff
    )
    funnel_ligand_topo = funnel["edge_index"][0]
    funnel_env_topo = funnel["edge_index"][1]
    funnel_shift = _unit_shift(
        positions[funnel_ligand_topo], positions[funnel_env_topo], cell
    ).round().to(torch.int64)
    funnel_pairs = {
        (int(funnel_ligand_topo[i]), int(funnel_env_topo[i])): tuple(funnel_shift[i].tolist())
        for i in range(int(funnel_ligand_topo.numel()))
    }

    funnel_hop1_topology = sorted(set(int(value) for value in funnel_env_topo.tolist()))
    topology_indices_consistent = (
        set(funnel_ligand_topo.tolist()) <= set(ligand_indices)
        and set(teacher_ligand_topo.tolist()) <= set(ligand_indices)
    )
    edge_pairs_match = set(teacher_pairs.keys()) == set(funnel_pairs.keys())
    s1_atom_set_matches = funnel_hop1_topology == teacher_hop1_topology
    common_keys = set(teacher_pairs) & set(funnel_pairs)
    shift_mismatches = sorted(
        key for key in common_keys if teacher_pairs[key] != funnel_pairs[key]
    )
    unit_shifts_consistent = edge_pairs_match and not shift_mismatches

    expected_hop1_ok = (
        args.expected_hop1_count is None
        or len(teacher_hop1_topology) == args.expected_hop1_count
    )

    # -- Boundary membership agreement on real atoms ----------------------
    min_distance = _min_distance_to_ligand(positions, cell, ligand_tensor, environment_tensor)
    gap = (min_distance - cutoff).abs()
    probe_count = min(args.boundary_probe_count, int(gap.numel()))
    closest_local = torch.argsort(gap)[:probe_count]
    boundary_probes = []
    teacher_hop1_set = set(teacher_hop1_topology)
    funnel_hop1_set = set(funnel_hop1_topology)
    for local_index in closest_local.tolist():
        topo_index = int(environment_tensor[local_index])
        teacher_included = topo_index in teacher_hop1_set
        funnel_included = topo_index in funnel_hop1_set
        boundary_probes.append(
            {
                "topology_index": topo_index,
                "min_distance_to_ligand_angstrom": float(min_distance[local_index].item()),
                "gap_to_cutoff_angstrom": float(gap[local_index].item()),
                "teacher_included": teacher_included,
                "funnel_included": funnel_included,
                "agrees": teacher_included == funnel_included,
            }
        )
    boundary_membership_agrees = all(probe["agrees"] for probe in boundary_probes)

    # -- Synthetic boundary-smoothness sweep on the single closest atom ---
    probe = boundary_probes[0]
    probe_topology_index = probe["topology_index"]
    probe_ligand_distances = torch.linalg.vector_norm(
        minimum_image_displacement(
            positions[ligand_tensor], positions[probe_topology_index].expand_as(positions[ligand_tensor]), cell
        ),
        dim=-1,
    )
    nearest_ligand_local = int(torch.argmin(probe_ligand_distances))
    nearest_ligand_topology_index = int(ligand_tensor[nearest_ligand_local])
    displacement = minimum_image_displacement(
        positions[nearest_ligand_topology_index], positions[probe_topology_index], cell
    )
    direction = displacement / torch.linalg.vector_norm(displacement)
    ligand_position = positions[nearest_ligand_topology_index]

    sweep = []
    step = 0.05
    distance = cutoff - 0.5
    while distance <= cutoff + 0.5 + 1e-9:
        synthetic_position = ligand_position + direction * distance
        actual_distance = float(
            torch.linalg.vector_norm(synthetic_position - ligand_position).item()
        )
        weight = float(
            quintic_c2_cutoff(
                torch.tensor([actual_distance], dtype=torch.float64),
                inner_cutoff=inner_cutoff, outer_cutoff=cutoff,
            ).item()
        )
        sweep.append(
            {
                "target_distance_angstrom": actual_distance,
                "discrete_included": actual_distance < cutoff,
                "contact_weight": weight,
            }
        )
        distance += step

    weights = [point["contact_weight"] for point in sweep]
    max_consecutive_abs_diff = max(
        (abs(weights[i + 1] - weights[i]) for i in range(len(weights) - 1)), default=0.0
    )
    contact_weight_continuous = max_consecutive_abs_diff < 0.5
    contact_weight_is_not_hard_binary = any(1e-9 < weight < 1.0 - 1e-9 for weight in weights)
    contact_weight_zero_beyond_cutoff = all(
        point["contact_weight"] == 0.0 for point in sweep if point["target_distance_angstrom"] >= cutoff
    )
    included_flags = [point["discrete_included"] for point in sweep]
    discrete_membership_flips_at_cutoff = included_flags == sorted(included_flags, reverse=True)

    checks = {
        "teacher_internal_consistency_ok": teacher_internal_consistency_ok,
        "ligand_set_matches": ligand_set_matches,
        "topology_indices_consistent": topology_indices_consistent,
        "edge_pairs_match": edge_pairs_match,
        "s1_atom_set_matches": s1_atom_set_matches,
        "unit_shifts_consistent": unit_shifts_consistent,
        "expected_hop1_count_matches": expected_hop1_ok,
        "boundary_membership_agrees": boundary_membership_agrees,
        "discrete_membership_flips_at_cutoff": discrete_membership_flips_at_cutoff,
        "contact_weight_continuous": contact_weight_continuous,
        "contact_weight_is_not_hard_binary": contact_weight_is_not_hard_binary,
        "contact_weight_zero_beyond_cutoff": contact_weight_zero_beyond_cutoff,
    }
    all_checks_passed = all(checks.values())

    inputs = {
        name: {"path": str(Path(path).resolve()), "sha256": _sha(path)}
        for name, path in (
            ("topology", args.topology), ("trajectory", args.trajectory),
            ("ligand_indices", args.ligand_indices),
        )
    }
    body = {
        "schema_version": "exp012-student-environment-funnel-smoke-v1",
        "run_id": args.run_id,
        "frame_index": args.frame_index,
        "edge_cutoff_angstrom": cutoff,
        "interaction_layers": layers,
        "checks": checks,
        "all_checks_passed": all_checks_passed,
        "teacher": {
            "node_count": int(membership["node_count"]),
            "edge_count": int(membership["edge_count"]),
            "hop_counts_by_layer": [int(value) for value in membership["hop_counts_by_layer"]],
            "hop1_atom_count": len(teacher_hop1_topology),
            "graph_membership_sha256": membership["graph_membership_sha256"],
        },
        "funnel": {
            "hop1_atom_count": len(funnel_hop1_topology),
            "edge_count": int(funnel_ligand_topo.numel()),
        },
        "mismatch_detail": {
            "hop1_teacher_only": sorted(set(teacher_hop1_topology) - set(funnel_hop1_topology)),
            "hop1_funnel_only": sorted(set(funnel_hop1_topology) - set(teacher_hop1_topology)),
            "shift_mismatches": [list(key) for key in shift_mismatches],
        },
        "boundary_probes": boundary_probes,
        "boundary_smoothness_sweep": {
            "probe_topology_index": probe_topology_index,
            "nearest_ligand_topology_index": nearest_ligand_topology_index,
            "inner_cutoff_angstrom": inner_cutoff,
            "outer_cutoff_angstrom": cutoff,
            "max_consecutive_abs_diff": max_consecutive_abs_diff,
            "points": sweep,
        },
        "inputs": inputs,
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "training_executed": False,
            "torchforce_used": False,
            "nvt_executed": False,
            "gap_loss_computed": False,
            "decision_reference": "DEC-037_item_1_pending_DEC-038",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"all_checks_passed={all_checks_passed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
