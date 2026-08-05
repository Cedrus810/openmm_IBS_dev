#!/usr/bin/env python
"""DEC-039 d0-5: pure-geometry, no-MACE, no-GPU sweep of the student funnel.

DEC-038 verified, on 2 real frames, that
``local_residual.geometry.ligand_environment_cross_edges`` (the student's online
funnel: one directed ``environment sender -> ligand receiver`` cross edge per
atom pair within the 5.0 Angstrom cutoff -- no reverse edge, no ligand-ligand
edge, no environment-environment edge, no self-edge) exactly reproduces the
teacher's hop==1 neighborhood. This script does not repeat that comparison --
it sweeps the funnel's *own* geometry across every frame of all three
registered ``hard_window0_run1/2/3`` trajectories (1500 frames total), to
measure the real distribution the d0-5/DEC-039 graph-scale budget must be
frozen against, instead of extrapolating from 2 sampled frames.

Per frame this records: the unique S1 (environment-side) atom count, the total
directed environment->ligand edge count, and the per-ligand-atom neighbor
count (one value per ligand atom). Membership is decided on CPU in float64
throughout -- the same discipline DEC-032/033's Option C already established
for the teacher's two-hop closure -- because this is an offline measurement
tool, not an online deployment path; no CUDA, no MACE model, and no student
model are involved anywhere in this script.

The final report also evaluates the exact budget-freezing rules agreed for
DEC-039: if the measured maximum directed-edge count is <=1536, target/ceiling
freeze at 1536/2048; if it falls in (1536, 2048], the ceiling stays at 2048 but
the target rises to the next multiple of 128 at or above the measured maximum;
if it exceeds 2048, this script does NOT raise any ceiling itself -- it flags
that DEC-039 must stop and the graph design must be revisited before any
number is frozen. The per-ligand-atom neighbor-count target/ceiling follow a
separate, always-computed rule: target = the measured maximum rounded up to
the next multiple of 16, ceiling = that target scaled by 1.25 and rounded up
to the next multiple of 16.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp012_xed.schema import load_preregistration  # noqa: E402
from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.geometry import ligand_environment_cross_edges  # noqa: E402


class GeometryAuditError(ValueError):
    """The audit's own inputs, or a structural funnel invariant, are violated."""


_STUDENT_FUNNEL_CUTOFF_ANGSTROM = 5.0


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_num_workers() -> int:
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count() or 1


def _select_run(registration: Any, run_id: str) -> dict[str, Any]:
    runs = registration.payload["inputs"]["runs"]
    selected = [run for run in runs if run["run_id"] == run_id]
    if len(selected) != 1:
        raise GeometryAuditError(f"unknown or duplicate run_id: {run_id}")
    return selected[0]


def _round_up_to_multiple(value: float, multiple: int) -> int:
    return int(math.ceil(value / multiple) * multiple)


# Populated once per worker process by _init_worker's initargs: ligand/environment
# membership and the cutoff are constant across all 1500 frames (same topology,
# same ligand definition), so this is computed once, not re-derived per frame.
_WORKER_STATE: dict[str, Any] = {}


def _init_worker(ligand_indices: list[int], atom_count: int, edge_cutoff_angstrom: float) -> None:
    import torch

    torch.set_num_threads(1)
    ligand_tensor = torch.tensor(sorted(ligand_indices), dtype=torch.int64)
    mask = torch.ones(atom_count, dtype=torch.bool)
    mask[ligand_tensor] = False
    environment_tensor = torch.nonzero(mask, as_tuple=True)[0]
    ligand_local_index = torch.full((atom_count,), -1, dtype=torch.long)
    ligand_local_index[ligand_tensor] = torch.arange(ligand_tensor.numel(), dtype=torch.long)
    _WORKER_STATE["ligand_tensor"] = ligand_tensor
    _WORKER_STATE["environment_tensor"] = environment_tensor
    _WORKER_STATE["ligand_local_index"] = ligand_local_index
    _WORKER_STATE["ligand_topology_indices"] = ligand_tensor.tolist()
    _WORKER_STATE["edge_cutoff_angstrom"] = edge_cutoff_angstrom


def _frame_funnel_worker(task: tuple) -> dict[str, Any]:
    frame_index, positions_angstrom, cell_angstrom = task
    ligand_tensor = _WORKER_STATE["ligand_tensor"]
    environment_tensor = _WORKER_STATE["environment_tensor"]
    ligand_local_index = _WORKER_STATE["ligand_local_index"]
    cutoff = _WORKER_STATE["edge_cutoff_angstrom"]
    import numpy as np
    import torch

    positions = torch.tensor(np.asarray(positions_angstrom), dtype=torch.float64)
    cell = torch.tensor(np.asarray(cell_angstrom), dtype=torch.float64)
    funnel = ligand_environment_cross_edges(
        positions, cell, ligand_tensor, environment_tensor, outer_cutoff=cutoff
    )
    edge_index = funnel["edge_index"]
    sender, receiver = edge_index[0], edge_index[1]

    # Structural invariant, verified every frame rather than assumed once:
    # every edge must be ligand-sender/environment-receiver in the array's own
    # layout (semantically environment->ligand, sender=environment/receiver=ligand
    # in the message-passing sense the student aggregates over), never the
    # reverse, never ligand-ligand, never environment-environment, never self.
    if sender.numel():
        if not bool(torch.isin(sender, ligand_tensor).all().item()):
            raise GeometryAuditError(f"frame {frame_index}: a cross edge's ligand-side endpoint is not a ligand atom")
        if bool(torch.isin(receiver, ligand_tensor).any().item()):
            raise GeometryAuditError(f"frame {frame_index}: a cross edge's environment-side endpoint is a ligand atom")

    s1_atom_count = int(torch.unique(receiver).numel()) if receiver.numel() else 0
    local_senders = ligand_local_index[sender] if sender.numel() else sender
    neighbor_count_by_ligand_atom = (
        torch.bincount(local_senders, minlength=ligand_tensor.numel()).tolist()
        if sender.numel()
        else [0] * ligand_tensor.numel()
    )
    return {
        "frame_index": frame_index,
        "s1_atom_count": s1_atom_count,
        "edge_count": int(edge_index.shape[1]),
        "neighbor_count_by_ligand_atom": neighbor_count_by_ligand_atom,
    }


def _percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summarize(values: list[int]) -> dict[str, float]:
    return {
        "max": max(values) if values else 0,
        "mean": sum(values) / len(values) if values else 0.0,
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default="protocols/EXP-012_preregistration.json")
    parser.add_argument("--ligand-indices", required=True, help="JSON file with a ligand_indices array")
    parser.add_argument("--edge-cutoff-angstrom", type=float, required=True)
    parser.add_argument(
        "--run-id", action="append", default=None,
        help="repeatable; default is every run_id registered in the preregistration",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.edge_cutoff_angstrom != _STUDENT_FUNNEL_CUTOFF_ANGSTROM:
        parser.error(
            "--edge-cutoff-angstrom must be the DEC-038-frozen student funnel cutoff "
            f"({_STUDENT_FUNNEL_CUTOFF_ANGSTROM})"
        )
    if args.frame_stride < 1:
        parser.error("--frame-stride must be a positive integer")
    if args.num_workers < 1:
        parser.error("--num-workers must be a positive integer")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")

    import mdtraj as md

    registration = load_preregistration(
        Path(args.preregistration) if Path(args.preregistration).is_absolute()
        else ROOT / args.preregistration,
        workspace_root=ROOT, verify_files=True,
    )
    ligand_payload = json.loads(Path(args.ligand_indices).expanduser().read_text(encoding="utf-8"))
    ligand_indices = ligand_payload.get("ligand_indices")
    if not isinstance(ligand_indices, list) or not ligand_indices:
        raise GeometryAuditError("--ligand-indices JSON must contain a non-empty ligand_indices array")
    ligand_indices = sorted(int(index) for index in ligand_indices)

    topology_relative = registration.payload["inputs"]["artifacts"]["topology"]["path"]
    topology_path = ROOT / topology_relative
    run_ids = args.run_id or [run["run_id"] for run in registration.payload["inputs"]["runs"]]

    atom_count = md.load(str(topology_path)).topology.n_atoms

    started = time.perf_counter()
    run_reports = []
    overall_max_edge = {"value": -1, "run_id": None, "frame_index": None}
    overall_max_s1 = {"value": -1, "run_id": None, "frame_index": None}
    overall_max_neighbor = {"value": -1, "run_id": None, "frame_index": None, "ligand_topology_index": None}
    all_neighbor_counts: list[int] = []

    with ProcessPoolExecutor(
        max_workers=args.num_workers, initializer=_init_worker,
        initargs=(ligand_indices, atom_count, args.edge_cutoff_angstrom),
    ) as executor:
        for run_id in run_ids:
            run = _select_run(registration, run_id)
            trajectory_relative = run["trajectory"]["path"]
            trajectory_path = ROOT / trajectory_relative
            observed_sha = _sha256_file(trajectory_path)
            if observed_sha != run["trajectory"]["sha256"]:
                raise GeometryAuditError(f"trajectory SHA-256 mismatch for {run_id}: {trajectory_path}")
            expected_frames = int(run["frame_count"])
            trajectory = md.load(str(trajectory_path), top=str(topology_path))
            if trajectory.n_frames != expected_frames:
                raise GeometryAuditError(f"{run_id}: trajectory frame count differs from preregistration")
            if trajectory.unitcell_vectors is None:
                raise GeometryAuditError(f"{run_id}: trajectory lacks periodic box vectors")
            if trajectory.n_atoms != atom_count:
                raise GeometryAuditError(f"{run_id}: trajectory atom count differs from the topology")

            frame_indices = list(range(0, expected_frames, args.frame_stride))
            tasks = (
                (
                    frame_index,
                    trajectory.xyz[frame_index] * 10.0,
                    trajectory.unitcell_vectors[frame_index] * 10.0,
                )
                for frame_index in frame_indices
            )
            chunksize = max(1, len(frame_indices) // (4 * args.num_workers))
            run_started = time.perf_counter()
            frames = []
            completed = 0
            for result in executor.map(_frame_funnel_worker, tasks, chunksize=chunksize):
                completed += 1
                frames.append(result)
                if result["edge_count"] > overall_max_edge["value"]:
                    overall_max_edge = {
                        "value": result["edge_count"], "run_id": run_id,
                        "frame_index": result["frame_index"],
                    }
                if result["s1_atom_count"] > overall_max_s1["value"]:
                    overall_max_s1 = {
                        "value": result["s1_atom_count"], "run_id": run_id,
                        "frame_index": result["frame_index"],
                    }
                frame_neighbor_counts = result["neighbor_count_by_ligand_atom"]
                all_neighbor_counts.extend(frame_neighbor_counts)
                frame_local_max = max(frame_neighbor_counts) if frame_neighbor_counts else -1
                if frame_local_max > overall_max_neighbor["value"]:
                    local_index = frame_neighbor_counts.index(frame_local_max)
                    overall_max_neighbor = {
                        "value": frame_local_max, "run_id": run_id,
                        "frame_index": result["frame_index"],
                        "ligand_topology_index": ligand_indices[local_index],
                    }
                if completed % 50 == 0 or completed == len(frame_indices):
                    print(f"{run_id}: {completed}/{len(frame_indices)} frames swept", flush=True)

            s1_counts = [frame["s1_atom_count"] for frame in frames]
            edge_counts = [frame["edge_count"] for frame in frames]
            run_reports.append(
                {
                    "run_id": run_id,
                    "trajectory": {"path": trajectory_relative, "sha256": observed_sha},
                    "audited_frame_count": len(frames),
                    "frame_stride": args.frame_stride,
                    "s1_atom_count": _summarize(s1_counts),
                    "edge_count": _summarize(edge_counts),
                    "frames": frames,
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
            )

    max_edge_value = overall_max_edge["value"]
    if max_edge_value <= 1536:
        edge_budget = {
            "rule_applied": "measured_maximum_at_or_below_1536",
            "target": 1536, "hard_ceiling": 2048, "stop_dec_039_graph_redesign_required": False,
        }
    elif max_edge_value <= 2048:
        edge_budget = {
            "rule_applied": "measured_maximum_between_1537_and_2048_raise_target_only",
            "target": _round_up_to_multiple(max_edge_value, 128), "hard_ceiling": 2048,
            "stop_dec_039_graph_redesign_required": False,
        }
    else:
        edge_budget = {
            "rule_applied": "measured_maximum_exceeds_2048_ceiling_not_raised",
            "target": None, "hard_ceiling": None, "stop_dec_039_graph_redesign_required": True,
        }

    neighbor_target = _round_up_to_multiple(overall_max_neighbor["value"], 16) if all_neighbor_counts else None
    neighbor_ceiling = _round_up_to_multiple(neighbor_target * 1.25, 16) if neighbor_target is not None else None

    body = {
        "schema_version": "exp012-student-environment-funnel-geometry-v1",
        "status": "COMPLETED_GEOMETRY_ONLY_NO_MACE_NO_GPU_NO_STUDENT",
        "edge_direction_semantics": "environment_sender_to_ligand_receiver_bipartite_only",
        "excludes": [
            "reverse_edges", "ligand_ligand_edges", "environment_environment_edges", "self_edges",
        ],
        "preregistration_sha256": registration.payload_sha256,
        "edge_cutoff_angstrom": args.edge_cutoff_angstrom,
        "ligand_atom_count": len(ligand_indices),
        "system_atom_count": atom_count,
        "num_workers": args.num_workers,
        "graph_membership_device": "cpu",
        "graph_membership_dtype": "float64",
        "runs": run_reports,
        "overall_max_edge_count_frame": overall_max_edge,
        "overall_max_s1_atom_count_frame": overall_max_s1,
        "overall_max_neighbor_count_single_ligand_atom": overall_max_neighbor,
        "neighbor_count_distribution": _summarize(all_neighbor_counts),
        "graph_scale_budget_evaluation": {
            "directed_environment_to_ligand_edges": edge_budget,
            "max_neighbors_per_ligand_atom": {
                "rule_applied": "target_is_measured_maximum_rounded_up_to_next_multiple_of_16;"
                                 "hard_ceiling_is_target_times_1_25_rounded_up_to_next_multiple_of_16",
                "target": neighbor_target, "hard_ceiling": neighbor_ceiling,
            },
        },
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "provisional_not_sealed": True,
            "scientific_qualification": False,
            "decision_reference": "DEC-039_pending",
            "mace_forward_executed": False,
            "gpu_used": False,
            "student_model_executed": False,
            "training_executed": False,
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(
        f"overall_max_edge={overall_max_edge['value']} "
        f"overall_max_s1={overall_max_s1['value']} "
        f"overall_max_neighbor={overall_max_neighbor['value']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
