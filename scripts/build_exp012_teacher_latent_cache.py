#!/usr/bin/env python
"""DEC-032 step (b): build the offline per-frame teacher latent cache.

Graph *membership* (which atoms, which edges, which periodic image -- DEC-032
Option C) is decided once per frame on CPU in float64, via
``local_residual.teacher_graph.compute_canonical_graph_membership``, the exact
same canonical computation ``scripts/audit_exp012_per_frame_teacher_graph_geometry.py``
uses. Only the resulting discrete tensors are moved to the execution device
(CUDA float32) via ``build_teacher_graph_from_membership``, which selects
positions and builds shifts from the *target* device's live tensors but never
re-derives membership. An earlier version of this script decided membership
independently on whatever device/dtype it happened to execute on, and CUDA
float32 disagreed with the CPU-based geometry audit by a couple of edges on
one frame -- not a bug, just two floating-point implementations separately
answering a boundary-sensitive cutoff question. Sharing one canonical
decision removes that disagreement by construction.

The frozen MACE teacher then runs under ``torch.no_grad()`` on the target
device -- no coordinate gradient is needed for the held-out gap-variance
readout this cache feeds (DEC-030 step c), so none is computed or retained,
and positions do not need ``requires_grad=True`` for this (only
``require_coordinate_grad=True`` calls do).

This is representation only. No ledger fields (``adjacent_gap_reduced``,
importance weights, train/held-out labels) are written here -- that join is a
separate step and must fail-closed align on run_id/frame_index/trajectory
SHA/frame_count/preregistration SHA. Keeping representation and thermodynamic
target separate means the cache doesn't depend on any particular loss
definition and can't leak a fold/target choice into the features themselves.

Every frame is checked against
``scripts/audit_exp012_per_frame_teacher_graph_geometry.py``'s already-measured
worst case: node count, edge count, AND the graph membership hash must match
that report's recorded values for the same frame exactly, and node/edge count
must not exceed its registered maximum. A mismatch is never silently
accepted -- it means the input, the graph-construction code, or the
trajectory identity changed since the geometry audit ran.

Bulk CUDA generation across 1500 frames is O(hours), so each frame's result is
checkpointed atomically to a per-run work directory as it completes; a
re-invocation resumes from the last completed frame. The final per-run
``.npz`` + report are only written once every frame in that run is present,
and refuse to overwrite an existing final artifact.
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

# DEC-027: exactly two preregistered encoder variants are allowed.
ENCODER_VARIANTS = {
    6.0: "original_6a",
    5.0: "derived_5a",
}

from exp012_xed.schema import load_preregistration  # noqa: E402
from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.mace_latent import MaceLatentBasisAdapter, load_c0_report  # noqa: E402
from local_residual.teacher_graph import (  # noqa: E402
    build_teacher_graph_from_membership,
    compute_canonical_graph_membership,
)


class LatentCacheError(RuntimeError):
    """A frame failed one of the fail-closed identity/geometry gates."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_run(registration: Any, run_id: str) -> dict[str, Any]:
    runs = registration.payload["inputs"]["runs"]
    selected = [run for run in runs if run["run_id"] == run_id]
    if len(selected) != 1:
        raise LatentCacheError(f"unknown or duplicate run_id: {run_id}")
    return selected[0]


def _atomic_write_npz(path: Path, arrays: dict) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
        ) as handle:
            temporary = handle.name
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


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


def _compute_frame_latent(
    adapter: MaceLatentBasisAdapter,
    positions_angstrom_np: Any,
    cell_angstrom_np: Any,
    *,
    ligand_indices: list[int],
    atomic_numbers_by_topology_index: list[int],
    model_atomic_numbers: Any,
    edge_cutoff_angstrom: float,
    interaction_layers: int,
    device: Any,
    torch_dtype: Any,
) -> tuple[Any, dict, Any, str]:
    """One frame: CPU float64 membership, target-device execution, one no-grad forward.

    Graph membership (which atoms, which edges, which periodic image) is
    decided on CPU float64 by ``compute_canonical_graph_membership`` -- the
    same canonical computation the geometry audit uses -- then handed to
    ``build_teacher_graph_from_membership`` to assemble the graph on the
    target device/dtype. No autograd graph is built or kept, and positions
    never need ``requires_grad=True`` here since the forward call uses
    ``require_coordinate_grad=False``.
    """
    import numpy as np
    import torch

    membership_positions = torch.tensor(np.asarray(positions_angstrom_np), dtype=torch.float64)
    membership_cell = torch.tensor(np.asarray(cell_angstrom_np), dtype=torch.float64)
    membership = compute_canonical_graph_membership(
        membership_positions, membership_cell,
        ligand_indices=ligand_indices,
        edge_cutoff_angstrom=edge_cutoff_angstrom, interaction_layers=interaction_layers,
    )

    target_positions = torch.tensor(positions_angstrom_np, dtype=torch_dtype, device=device)
    target_cell = torch.tensor(cell_angstrom_np, dtype=torch_dtype, device=device)
    graph = build_teacher_graph_from_membership(
        membership, target_positions, target_cell,
        atomic_numbers_by_topology_index=atomic_numbers_by_topology_index,
        model_atomic_numbers=model_atomic_numbers,
    )
    with torch.no_grad():
        result = adapter.forward(graph, require_coordinate_grad=False)
    ligand_topology_index = (
        graph["topology_indices_by_mace_node_index"][graph["ligand_mask"]].cpu().numpy()
    )
    latent = result["ligand_latent"].detach().cpu().numpy()
    return (
        latent, dict(graph["diagnostics"]), ligand_topology_index,
        membership["graph_membership_sha256"],
    )


def _build_run_cache(
    *,
    run_id: str,
    registration: Any,
    geometry_lookup: dict,
    node_ceiling: int,
    edge_ceiling: int,
    adapter: MaceLatentBasisAdapter,
    ligand_indices: list[int],
    expected_ligand_topology_index: Any,
    atomic_numbers_by_topology_index: list[int],
    model_atomic_numbers: Any,
    edge_cutoff_angstrom: float,
    interaction_layers: int,
    device: Any,
    torch_dtype: Any,
    topology_path: str,
    work_dir_override: str | None,
    output_dir: Path,
    provenance: dict,
) -> None:
    import mdtraj
    import numpy as np

    run = _select_run(registration, run_id)
    trajectory_relative = run["trajectory"]["path"]
    trajectory_path = ROOT / trajectory_relative
    observed_sha = _sha256_file(trajectory_path)
    if observed_sha != run["trajectory"]["sha256"]:
        raise LatentCacheError(f"trajectory SHA-256 mismatch for {run_id}: {trajectory_path}")
    expected_frames = int(run["frame_count"])

    final_npz_path = output_dir / f"latent_cache_{run_id}.npz"
    final_report_path = output_dir / f"latent_cache_{run_id}_report.json"
    if final_npz_path.exists() or final_report_path.exists():
        raise LatentCacheError(f"refusing to overwrite an existing final cache for {run_id}")

    # A shared --work-dir is a parent, not the literal per-run directory: two
    # runs processed in one invocation must never write frame_NNNN.npz into
    # the same directory, or their checkpoints would collide and corrupt
    # each other silently.
    work_dir = (Path(work_dir_override) / run_id) if work_dir_override else output_dir / f".work_{run_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    trajectory = mdtraj.load(str(trajectory_path), top=topology_path)
    if trajectory.n_frames != expected_frames:
        raise LatentCacheError(f"{run_id}: trajectory frame count differs from preregistration")
    if trajectory.unitcell_vectors is None:
        raise LatentCacheError(f"{run_id}: trajectory lacks periodic box vectors")

    started = time.perf_counter()
    for frame_index in range(expected_frames):
        checkpoint_path = work_dir / f"frame_{frame_index:04d}.npz"
        if checkpoint_path.exists():
            continue
        key = (run_id, frame_index)
        if key not in geometry_lookup:
            raise LatentCacheError(f"{run_id} frame {frame_index}: not covered by the geometry report")
        expected_node_count, expected_edge_count, expected_membership_sha256 = geometry_lookup[key]

        positions_np = trajectory.xyz[frame_index] * 10.0
        cell_np = trajectory.unitcell_vectors[frame_index] * 10.0
        latent, diagnostics, ligand_topology_index, membership_sha256 = _compute_frame_latent(
            adapter, positions_np, cell_np,
            ligand_indices=ligand_indices,
            atomic_numbers_by_topology_index=atomic_numbers_by_topology_index,
            model_atomic_numbers=model_atomic_numbers,
            edge_cutoff_angstrom=edge_cutoff_angstrom, interaction_layers=interaction_layers,
            device=device, torch_dtype=torch_dtype,
        )
        node_count = int(diagnostics["node_count"])
        edge_count = int(diagnostics["edge_count"])

        if latent.shape != (len(ligand_indices), adapter.latent_dimension):
            raise LatentCacheError(f"{run_id} frame {frame_index}: unexpected latent shape {latent.shape}")
        if not np.isfinite(latent).all():
            raise LatentCacheError(f"{run_id} frame {frame_index}: non-finite latent")
        if not np.array_equal(ligand_topology_index, expected_ligand_topology_index):
            raise LatentCacheError(f"{run_id} frame {frame_index}: ligand topology order drifted")
        if diagnostics.get("support_definition") != "exact_cutoff_graph_n_hop_closure_no_fixed_manifest":
            raise LatentCacheError(f"{run_id} frame {frame_index}: unexpected graph support_definition")
        if diagnostics.get("complete_residue_expansion") is not False:
            raise LatentCacheError(f"{run_id} frame {frame_index}: residue expansion leaked back in")
        if diagnostics.get("fixed_environment_manifest") is not False:
            raise LatentCacheError(f"{run_id} frame {frame_index}: fixed environment manifest leaked back in")
        if membership_sha256 != expected_membership_sha256:
            raise LatentCacheError(
                f"{run_id} frame {frame_index}: graph membership hash {membership_sha256} "
                f"differs from the geometry report {expected_membership_sha256} -- discrete "
                "graph membership no longer matches even though both are computed on CPU "
                "float64; something about the input or the membership code changed"
            )
        if (node_count, edge_count) != (expected_node_count, expected_edge_count):
            raise LatentCacheError(
                f"{run_id} frame {frame_index}: node/edge count ({node_count},{edge_count}) "
                f"differs from the geometry report ({expected_node_count},{expected_edge_count})"
            )
        if node_count > node_ceiling or edge_count > edge_ceiling:
            raise LatentCacheError(
                f"{run_id} frame {frame_index}: node/edge count ({node_count},{edge_count}) "
                f"exceeds the registered maximum ({node_ceiling},{edge_ceiling})"
            )

        _atomic_write_npz(
            checkpoint_path,
            {
                "ligand_latent": latent.astype(np.float32),
                "node_count": np.int32(node_count),
                "edge_count": np.int32(edge_count),
                "hop_counts": np.asarray(diagnostics["hop_counts_by_layer"], dtype=np.int32),
                "ligand_topology_index": ligand_topology_index.astype(np.int64),
                "graph_membership_sha256": np.array(membership_sha256, dtype="<U64"),
            },
        )
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == expected_frames:
            print(f"{run_id}: {frame_index + 1}/{expected_frames} frames cached", flush=True)

    # Consolidate -- every frame's checkpoint must already exist at this point.
    latent_dim = adapter.latent_dimension
    ligand_count = len(ligand_indices)
    all_latent = np.empty((expected_frames, ligand_count, latent_dim), dtype=np.float32)
    all_node_count = np.empty(expected_frames, dtype=np.int32)
    all_edge_count = np.empty(expected_frames, dtype=np.int32)
    all_hop_counts = np.empty((expected_frames, interaction_layers + 1), dtype=np.int32)
    all_membership_sha256 = np.empty(expected_frames, dtype="<U64")
    ligand_topology_index_final = None
    for frame_index in range(expected_frames):
        checkpoint_path = work_dir / f"frame_{frame_index:04d}.npz"
        with np.load(checkpoint_path) as data:
            all_latent[frame_index] = data["ligand_latent"]
            all_node_count[frame_index] = data["node_count"]
            all_edge_count[frame_index] = data["edge_count"]
            all_hop_counts[frame_index] = data["hop_counts"]
            all_membership_sha256[frame_index] = data["graph_membership_sha256"].item()
            if ligand_topology_index_final is None:
                ligand_topology_index_final = data["ligand_topology_index"]
            elif not np.array_equal(ligand_topology_index_final, data["ligand_topology_index"]):
                raise LatentCacheError(
                    f"{run_id} frame {frame_index}: ligand topology order drifted across checkpoints"
                )
        # A checkpoint from a resumed run may have been validated against a
        # *different* --geometry-report than this invocation's. Skipping
        # recompute must not also skip re-validation -- otherwise switching
        # the geometry report between runs would silently leave stale,
        # unchecked frames in the final cache.
        key = (run_id, frame_index)
        if key not in geometry_lookup:
            raise LatentCacheError(f"{run_id} frame {frame_index}: not covered by the geometry report")
        expected_node_count, expected_edge_count, expected_membership_sha256 = geometry_lookup[key]
        stored_node_count = int(all_node_count[frame_index])
        stored_edge_count = int(all_edge_count[frame_index])
        stored_membership_sha256 = str(all_membership_sha256[frame_index])
        if stored_membership_sha256 != expected_membership_sha256:
            raise LatentCacheError(
                f"{run_id} frame {frame_index}: checkpointed graph membership hash "
                f"{stored_membership_sha256} differs from the current geometry report "
                f"{expected_membership_sha256} -- this checkpoint was validated against a "
                "different geometry report; delete it and recompute"
            )
        if (stored_node_count, stored_edge_count) != (expected_node_count, expected_edge_count):
            raise LatentCacheError(
                f"{run_id} frame {frame_index}: checkpointed node/edge count "
                f"({stored_node_count},{stored_edge_count}) differs from the current geometry "
                f"report ({expected_node_count},{expected_edge_count}) -- this checkpoint was "
                "validated against a different geometry report; delete it and recompute"
            )
        if stored_node_count > node_ceiling or stored_edge_count > edge_ceiling:
            raise LatentCacheError(
                f"{run_id} frame {frame_index}: checkpointed node/edge count "
                f"({stored_node_count},{stored_edge_count}) exceeds the registered maximum "
                f"({node_ceiling},{edge_ceiling})"
            )

    pooled_latent = all_latent.mean(axis=1)
    _atomic_write_npz(
        final_npz_path,
        {
            "ligand_latent": all_latent,
            "frame_index": np.arange(expected_frames, dtype=np.int64),
            "ligand_topology_index": ligand_topology_index_final,
            "node_count": all_node_count,
            "edge_count": all_edge_count,
            "hop_counts": all_hop_counts,
            "graph_membership_sha256": all_membership_sha256,
            "pooled_latent": pooled_latent,
        },
    )
    npz_sha = _sha256_file(final_npz_path)

    body = {
        "schema_version": "exp012-teacher-latent-cache-v2",
        "status": "COMPLETED_LATENT_ONLY_NOT_JOINED_WITH_LEDGER",
        "run_id": run_id,
        "trajectory": {"path": trajectory_relative, "sha256": observed_sha},
        "preregistration_sha256": registration.payload_sha256,
        "frame_count": expected_frames,
        "latent_shape": [expected_frames, ligand_count, latent_dim],
        "node_count_range": [int(all_node_count.min()), int(all_node_count.max())],
        "edge_count_range": [int(all_edge_count.min()), int(all_edge_count.max())],
        "geometry_node_ceiling": node_ceiling,
        "geometry_edge_ceiling": edge_ceiling,
        "graph_membership_device": "cpu",
        "graph_membership_dtype": "float64",
        "model_execution_device": str(device),
        "model_execution_dtype": str(torch_dtype),
        "npz_path": str(final_npz_path.resolve()),
        "npz_sha256": npz_sha,
        "inputs": provenance,
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "provisional_not_sealed": True,
            "scientific_qualification": False,
            "decision_reference": "DEC-032",
            "graph_policy": "per_frame_exact_two_hop_closure_no_fixed_manifest",
            "complete_residue_expansion": False,
            "fixed_environment_manifest": False,
            "coordinate_gradient_computed": False,
            "no_grad_used": True,
            "training_executed": False,
            "energy_fields_used": False,
            "fragment_subtraction_used": False,
            "ledger_joined": False,
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(final_report_path, report)
    print(f"{run_id}: {report['report_sha256']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default="protocols/EXP-012_preregistration.json")
    parser.add_argument(
        "--geometry-report", required=True,
        help="output of scripts/audit_exp012_per_frame_teacher_graph_geometry.py",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--c0-report", required=True)
    parser.add_argument("--ligand-indices", required=True, help="JSON file with a ligand_indices array")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--edge-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--product-layer-index", type=int, default=1)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument(
        "--run-id", action="append", default=None,
        help="repeatable; default is every run_id registered in the preregistration",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--work-dir",
        help=(
            "parent directory for per-frame checkpoints; a per-run_id subdirectory is "
            "always used underneath it, so multiple --run-id values never collide. "
            "Default is <output-dir>/.work_<run_id> per run"
        ),
    )
    args = parser.parse_args(argv)

    if args.edge_cutoff_angstrom not in ENCODER_VARIANTS:
        parser.error(
            "--edge-cutoff-angstrom must be exactly one of the preregistered "
            f"EXP-012 encoder variants: {sorted(ENCODER_VARIANTS)}"
        )

    import mdtraj
    import numpy as np
    import torch

    registration = load_preregistration(
        Path(args.preregistration) if Path(args.preregistration).is_absolute()
        else ROOT / args.preregistration,
        workspace_root=ROOT, verify_files=True,
    )
    geometry_report = json.loads(Path(args.geometry_report).read_text(encoding="utf-8"))
    if geometry_report.get("status") != "COMPLETED_GEOMETRY_ONLY_NO_MACE":
        raise LatentCacheError("--geometry-report is not a completed geometry-only audit")
    if geometry_report.get("schema_version") != "exp012-per-frame-teacher-graph-geometry-v2":
        raise LatentCacheError(
            "--geometry-report predates the per-frame graph_membership_sha256 field "
            "(DEC-032 Option C) -- regenerate it with the current "
            "scripts/audit_exp012_per_frame_teacher_graph_geometry.py"
        )
    if (
        geometry_report.get("graph_membership_device") != "cpu"
        or geometry_report.get("graph_membership_dtype") != "float64"
    ):
        raise LatentCacheError(
            "--geometry-report's graph membership was not computed on CPU float64"
        )
    if geometry_report.get("preregistration_sha256") != registration.payload_sha256:
        raise LatentCacheError(
            "--geometry-report was built against a different preregistration than the current one"
        )
    node_ceiling = int(geometry_report["overall_max_node_count_frame"]["value"])
    edge_ceiling = int(geometry_report["overall_max_edge_count_frame"]["value"])
    geometry_lookup: dict[tuple[str, int], tuple[int, int, str]] = {}
    for run_entry in geometry_report["runs"]:
        for frame_entry in run_entry["frames"]:
            geometry_lookup[(run_entry["run_id"], int(frame_entry["frame_index"]))] = (
                int(frame_entry["node_count"]), int(frame_entry["edge_count"]),
                str(frame_entry["graph_membership_sha256"]),
            )

    ligand_payload = json.loads(Path(args.ligand_indices).expanduser().read_text(encoding="utf-8"))
    ligand_indices = ligand_payload.get("ligand_indices")
    if not isinstance(ligand_indices, list) or not ligand_indices:
        raise LatentCacheError("--ligand-indices JSON must contain a non-empty ligand_indices array")
    ligand_indices = sorted(int(index) for index in ligand_indices)
    expected_ligand_topology_index = np.asarray(ligand_indices, dtype=np.int64)

    c0 = load_c0_report(args.c0_report)
    interaction_layers = args.product_layer_index + 1
    torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    topology_object = mdtraj.load(args.topology).topology
    atomic_numbers_by_topology_index = [
        int(atom.element.atomic_number) for atom in topology_object.atoms
    ]
    model_atomic_numbers = c0["expected"]["atomic_numbers"]

    adapter = MaceLatentBasisAdapter(
        c0_report=c0, model_path=args.model, device=args.device, dtype=args.dtype,
        product_layer_index=args.product_layer_index,
    )
    # The model is lazily loaded on first forward(); force it now so a load
    # failure surfaces immediately instead of after already checkpointing
    # some frames, and so the frozen-parameter check below has a model to
    # check. _load() itself already raises MaceLatentError if any parameter
    # is unfrozen, so this is a defensive, self-documenting recheck.
    adapter._load()
    if any(parameter.requires_grad for parameter in adapter._model.parameters()):
        raise LatentCacheError("MACE parameters are not completely frozen")

    run_ids = args.run_id or [run["run_id"] for run in registration.payload["inputs"]["runs"]]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = {
        name: {"path": str(Path(path).resolve()), "sha256": _sha256_file(path)}
        for name, path in (
            ("model", args.model), ("c0_report", args.c0_report),
            ("ligand_indices", args.ligand_indices), ("topology", args.topology),
            ("geometry_report", args.geometry_report),
        )
    }

    for run_id in run_ids:
        _build_run_cache(
            run_id=run_id, registration=registration, geometry_lookup=geometry_lookup,
            node_ceiling=node_ceiling, edge_ceiling=edge_ceiling,
            adapter=adapter, ligand_indices=ligand_indices,
            expected_ligand_topology_index=expected_ligand_topology_index,
            atomic_numbers_by_topology_index=atomic_numbers_by_topology_index,
            model_atomic_numbers=model_atomic_numbers,
            edge_cutoff_angstrom=args.edge_cutoff_angstrom, interaction_layers=interaction_layers,
            device=device, torch_dtype=torch_dtype, topology_path=args.topology,
            work_dir_override=args.work_dir, output_dir=output_dir, provenance=provenance,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
