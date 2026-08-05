#!/usr/bin/env python
"""DEC-032: pure-geometry, no-MACE sweep of the per-frame exact-closure graph.

For every frame of the registered EXP-012 target-ledger trajectories
(``hard_window0_run1/2/3``), this decides that frame's exact two-hop
cutoff-graph closure -- no fixed manifest, no residue completion -- via
``local_residual.teacher_graph.compute_canonical_graph_membership`` and
reports its size, a deterministic membership hash, and its composition. No
MACE model runs here; this is graph construction only, so a frame's actual
worst-case cost can be identified *before* spending any CPU/CUDA C1 smoke
time on it, instead of assuming frame0 is representative.

Membership is always decided on CPU in float64 -- not configurable -- because
that is the exact same canonical computation
``scripts/build_exp012_teacher_latent_cache.py`` uses to decide membership
before handing the discrete result to CUDA for the actual MACE forward. An
earlier version of this script let ``--dtype`` vary and ran everything on
CPU regardless, and a bulk run on CUDA disagreed with it by a couple of
edges on one frame -- not a bug in either side, just two independent
floating-point implementations of the same boundary-sensitive discrete
decision. Sharing one canonical CPU float64 computation removes that
disagreement by construction; see DEC-032 Option C.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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

# DEC-027: exactly two preregistered encoder variants are allowed.
ENCODER_VARIANTS = {
    6.0: "original_6a",
    5.0: "derived_5a",
}
_WATER_RESIDUE_NAMES = {"HOH", "WAT", "SOL", "TP3", "TIP3"}
_ION_RESIDUE_NAMES = {"NA", "CL", "K", "MG", "CA", "ZN", "LI", "RB", "CS", "F", "BR", "I"}

from exp012_xed.schema import load_preregistration  # noqa: E402
from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.teacher_graph import compute_canonical_graph_membership  # noqa: E402


class GeometryAuditError(ValueError):
    """The audit's own inputs are malformed."""


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


# Populated once per worker process by _init_worker's initargs, not re-sent
# with every one of the 1500 per-frame tasks: the residue-name lookup alone is
# a 73536-entry list, and repickling it per task would move gigabytes of
# identical data through the IPC pipe for no reason.
_WORKER_STATE: dict[str, Any] = {}


def _init_worker(
    ligand_indices: list[int],
    residue_name_by_topology_index: list[str],
    edge_cutoff_angstrom: float,
    interaction_layers: int,
) -> None:
    import torch

    torch.set_num_threads(1)
    _WORKER_STATE["ligand_indices"] = ligand_indices
    _WORKER_STATE["residue_name_by_topology_index"] = residue_name_by_topology_index
    _WORKER_STATE["edge_cutoff_angstrom"] = edge_cutoff_angstrom
    _WORKER_STATE["interaction_layers"] = interaction_layers


def _frame_geometry_worker(task: tuple) -> dict[str, Any]:
    frame_index, positions_angstrom, cell_angstrom = task
    ligand_indices = _WORKER_STATE["ligand_indices"]
    residue_name_by_topology_index = _WORKER_STATE["residue_name_by_topology_index"]
    edge_cutoff_angstrom = _WORKER_STATE["edge_cutoff_angstrom"]
    interaction_layers = _WORKER_STATE["interaction_layers"]
    import numpy as np
    import torch

    # DEC-032 Option C: graph membership is decided exactly once, on CPU
    # float64, by the same function the bulk CUDA cache uses for its
    # membership decision -- so this report's counts and membership hash are
    # what the bulk run will also compute, not an independent CPU estimate of
    # it that can disagree at a cutoff-boundary pair.
    positions = torch.tensor(np.asarray(positions_angstrom), dtype=torch.float64)
    cell = torch.tensor(np.asarray(cell_angstrom), dtype=torch.float64)
    membership = compute_canonical_graph_membership(
        positions, cell, ligand_indices=ligand_indices,
        edge_cutoff_angstrom=edge_cutoff_angstrom, interaction_layers=interaction_layers,
    )
    topology_order = membership["topology_index"].tolist()

    ligand_set = {int(index) for index in ligand_indices}
    water = ion = other_environment = 0
    for topology_atom_index in topology_order:
        if topology_atom_index in ligand_set:
            continue
        name = residue_name_by_topology_index[topology_atom_index]
        if name in _WATER_RESIDUE_NAMES:
            water += 1
        elif name in _ION_RESIDUE_NAMES:
            ion += 1
        else:
            other_environment += 1

    return {
        "frame_index": frame_index,
        "node_count": membership["node_count"],
        "edge_count": membership["edge_count"],
        "hop_counts_by_layer": membership["hop_counts_by_layer"],
        "graph_membership_sha256": membership["graph_membership_sha256"],
        "environment_water_atom_count": water,
        "environment_ion_atom_count": ion,
        "environment_other_atom_count": other_environment,
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
    parser.add_argument("--interaction-layers", type=int, required=True)
    parser.add_argument(
        "--run-id", action="append", default=None,
        help="repeatable; default is every run_id registered in the preregistration",
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=_default_num_workers())
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.edge_cutoff_angstrom not in ENCODER_VARIANTS:
        parser.error(
            "--edge-cutoff-angstrom must be exactly one of the preregistered "
            f"EXP-012 encoder variants: {sorted(ENCODER_VARIANTS)}"
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

    # Computed once from the topology file alone (no DCD needed) and handed to
    # every worker exactly once via initargs -- see _WORKER_STATE.
    residue_name_by_topology_index = [
        str(atom.residue.name).upper() for atom in md.load(str(topology_path)).topology.atoms
    ]

    started = time.perf_counter()
    run_reports = []
    overall_max_edge = {"value": -1, "run_id": None, "frame_index": None}
    overall_max_node = {"value": -1, "run_id": None, "frame_index": None}

    with ProcessPoolExecutor(
        max_workers=args.num_workers, initializer=_init_worker,
        initargs=(
            ligand_indices, residue_name_by_topology_index,
            args.edge_cutoff_angstrom, args.interaction_layers,
        ),
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
            if trajectory.n_atoms != len(residue_name_by_topology_index):
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
            for result in executor.map(_frame_geometry_worker, tasks, chunksize=chunksize):
                completed += 1
                frames.append(result)
                if result["edge_count"] > overall_max_edge["value"]:
                    overall_max_edge = {
                        "value": result["edge_count"], "run_id": run_id,
                        "frame_index": result["frame_index"],
                    }
                if result["node_count"] > overall_max_node["value"]:
                    overall_max_node = {
                        "value": result["node_count"], "run_id": run_id,
                        "frame_index": result["frame_index"],
                    }
                if completed % 50 == 0 or completed == len(frame_indices):
                    print(f"{run_id}: {completed}/{len(frame_indices)} frames swept", flush=True)

            node_counts = [frame["node_count"] for frame in frames]
            edge_counts = [frame["edge_count"] for frame in frames]
            run_reports.append(
                {
                    "run_id": run_id,
                    "trajectory": {"path": trajectory_relative, "sha256": observed_sha},
                    "audited_frame_count": len(frames),
                    "frame_stride": args.frame_stride,
                    "node_count": _summarize(node_counts),
                    "edge_count": _summarize(edge_counts),
                    "environment_water_atom_count": _summarize(
                        [frame["environment_water_atom_count"] for frame in frames]
                    ),
                    "environment_ion_atom_count": _summarize(
                        [frame["environment_ion_atom_count"] for frame in frames]
                    ),
                    "environment_other_atom_count": _summarize(
                        [frame["environment_other_atom_count"] for frame in frames]
                    ),
                    "frames": frames,
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
            )

    body = {
        "schema_version": "exp012-per-frame-teacher-graph-geometry-v2",
        "status": "COMPLETED_GEOMETRY_ONLY_NO_MACE",
        "encoder_variant": ENCODER_VARIANTS[args.edge_cutoff_angstrom],
        "preregistration_sha256": registration.payload_sha256,
        "edge_cutoff_angstrom": args.edge_cutoff_angstrom,
        "interaction_layers": args.interaction_layers,
        "ligand_atom_count": len(ligand_indices),
        "num_workers": args.num_workers,
        "graph_membership_device": "cpu",
        "graph_membership_dtype": "float64",
        "runs": run_reports,
        "overall_max_edge_count_frame": overall_max_edge,
        "overall_max_node_count_frame": overall_max_node,
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "provisional_not_sealed": True,
            "scientific_qualification": False,
            "decision_reference": "DEC-032",
            "mace_forward_executed": False,
            "training_executed": False,
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
