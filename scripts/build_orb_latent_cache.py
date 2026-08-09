#!/usr/bin/env python
"""Build the pre-registered ORB-v3 layer-2 representation cache.

The cache is representation-only and remains separate from the MM ledger.
Each frame first derives the exact L2 closure from the complete parent graph
on CPU float64.  A frame fails closed if any outgoing neighbor count reaches
the official 120-neighbor cap.  The selected local closure then inherits the
frozen parent-system conditioning contract ``Q=0, M=1``.

The final filenames intentionally match
``scripts/join_exp012_teacher_latent_cache_with_ledger.py``:
``latent_cache_<run_id>.npz`` and ``latent_cache_<run_id>_report.json``.
L5 is not accepted by this command; this command is layer-2 primary only.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp012_xed.schema import load_preregistration  # noqa: E402
from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.orb_graph import audit_lhop_graphs  # noqa: E402
from local_residual.orb_latent import (  # noqa: E402
    OrbLatentAdapter,
    OrbLatentError,
    OrbModelSpec,
    OrbParentConditioningContract,
)


class OrbCacheError(RuntimeError):
    """A frame failed an ORB cache identity, graph, or latent gate."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _atomic_write_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n"
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


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must use RUN_ID=TRAJECTORY_PATH")
    run_id, trajectory = value.split("=", 1)
    if not run_id or not trajectory:
        raise argparse.ArgumentTypeError("--run must use RUN_ID=TRAJECTORY_PATH")
    return run_id, Path(trajectory).expanduser().resolve()


def _load_checkpoint(path: Path, *, expected_frame: int, expected_topology_indices, expected_layer: dict):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        if int(data["frame_index"]) != expected_frame:
            raise OrbCacheError(f"checkpoint frame mismatch: {path}")
        if not np.array_equal(data["topology_indices"], expected_topology_indices):
            raise OrbCacheError(f"checkpoint graph membership changed: {path}")
        for key in ("node_count", "edge_count", "max_outgoing_neighbors", "cap_hit_node_count"):
            if int(data[key]) != int(expected_layer[key]):
                raise OrbCacheError(f"checkpoint {key} mismatch at frame {expected_frame}: {path}")
        if bool(data["cap_hit"]):
            raise OrbCacheError(f"checkpoint records a neighbor-cap hit at frame {expected_frame}: {path}")


def _run_cache(
    *,
    run_id: str,
    trajectory_path: Path,
    topology_path: Path,
    ligand_indices: list[int],
    adapter: OrbLatentAdapter,
    contract: OrbParentConditioningContract,
    output_dir: Path,
    work_parent: Path,
    registration_sha256: str,
    model_name: str,
) -> None:
    import mdtraj as md
    import numpy as np

    final_npz = output_dir / f"latent_cache_{run_id}.npz"
    final_report = output_dir / f"latent_cache_{run_id}_report.json"
    if final_npz.exists() or final_report.exists():
        raise OrbCacheError(f"refusing to overwrite existing final cache for {run_id}")

    topology = md.load(str(topology_path)).topology
    atomic_numbers_by_index = [int(atom.element.atomic_number) for atom in topology.atoms]
    if max(ligand_indices) >= len(atomic_numbers_by_index):
        raise OrbCacheError(f"{run_id}: ligand index is outside topology")
    work_dir = work_parent / run_id
    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    frame_count = 0
    checkpoint_count = 0
    node_counts = []
    edge_counts = []
    max_neighbors = []
    topology_hashes = []

    for chunk in md.iterload(str(trajectory_path), top=str(topology_path), chunk=1):
        if chunk.unitcell_vectors is None:
            raise OrbCacheError(f"{run_id}: trajectory has no periodic cell")
        positions = np.asarray(chunk.xyz[0], dtype=np.float64) * 10.0
        cell = np.asarray(chunk.unitcell_vectors[0], dtype=np.float64) * 10.0
        graph = audit_lhop_graphs(
            positions,
            cell,
            ligand_indices=ligand_indices,
            cutoff_angstrom=6.0,
            max_num_neighbors=120,
            max_layer=2,
        )
        layer2 = graph["layers"][1]
        if layer2["cap_hit"]:
            raise OrbCacheError(
                f"{run_id} frame {frame_count}: official 120-neighbor cap hit "
                f"({layer2['cap_hit_node_count']} nodes); cache is fail-closed"
            )
        topology_indices = np.asarray(layer2["topology_indices"], dtype=np.int64)
        topology_hash = hashlib.sha256(np.ascontiguousarray(topology_indices).tobytes()).hexdigest()
        checkpoint = work_dir / f"frame_{frame_count:04d}.npz"
        if checkpoint.exists():
            _load_checkpoint(
                checkpoint,
                expected_frame=frame_count,
                expected_topology_indices=topology_indices,
                expected_layer=layer2,
            )
        else:
            local_index_by_topology = {int(value): index for index, value in enumerate(topology_indices)}
            try:
                local_ligand_indices = [local_index_by_topology[index] for index in ligand_indices]
            except KeyError as exc:
                raise OrbCacheError(f"{run_id} frame {frame_count}: ligand omitted from L2 closure") from exc
            local_positions = positions[topology_indices]
            local_atomic_numbers = [int(atomic_numbers_by_index[index]) for index in topology_indices]
            try:
                result = adapter.extract_frame(
                    local_positions,
                    cell,
                    atomic_numbers=local_atomic_numbers,
                    ligand_indices=local_ligand_indices,
                    topology_indices=topology_indices,
                    conditioning_contract=contract,
                    layer=2,
                )
            except OrbLatentError as exc:
                raise OrbCacheError(f"{run_id} frame {frame_count}: ORB extraction failed: {exc}") from exc
            latent = result.ligand_latent.detach().cpu().numpy().astype(np.float32)
            if latent.shape != (len(ligand_indices), 256) or not np.isfinite(latent).all():
                raise OrbCacheError(f"{run_id} frame {frame_count}: invalid latent shape or finite check")
            _atomic_write_npz(
                checkpoint,
                {
                    "frame_index": np.int64(frame_count),
                    "ligand_latent": latent,
                    "topology_indices": topology_indices,
                    "node_count": np.int32(layer2["node_count"]),
                    "edge_count": np.int32(layer2["edge_count"]),
                    "hop_counts": np.asarray([layer2["hop_counts"][str(hop)] for hop in range(3)], dtype=np.int32),
                    "max_outgoing_neighbors": np.int32(layer2["max_outgoing_neighbors"]),
                    "cap_hit_node_count": np.int32(layer2["cap_hit_node_count"]),
                    "cap_hit": np.bool_(layer2["cap_hit"]),
                    "topology_indices_sha256": np.asarray(topology_hash),
                    "edge_set_sha256": np.asarray(result.diagnostics["edge_set_sha256"]),
                    "closure_edge_set_sha256": np.asarray(result.diagnostics["closure_edge_set_sha256"]),
                },
            )
            checkpoint_count += 1
        node_counts.append(int(layer2["node_count"]))
        edge_counts.append(int(layer2["edge_count"]))
        max_neighbors.append(int(layer2["max_outgoing_neighbors"]))
        topology_hashes.append(topology_hash)
        frame_count += 1
        if frame_count % 25 == 0:
            print(f"{run_id}: {frame_count} frames audited/cached", flush=True)

    if frame_count == 0:
        raise OrbCacheError(f"{run_id}: trajectory yielded no frames")
    all_latent = np.empty((frame_count, len(ligand_indices), 256), dtype=np.float32)
    all_hop_counts = np.empty((frame_count, 3), dtype=np.int32)
    for frame_index in range(frame_count):
        checkpoint = work_dir / f"frame_{frame_index:04d}.npz"
        if not checkpoint.is_file():
            raise OrbCacheError(f"{run_id}: missing checkpoint for frame {frame_index}")
        with np.load(checkpoint, allow_pickle=False) as data:
            all_latent[frame_index] = data["ligand_latent"]
            all_hop_counts[frame_index] = data["hop_counts"]
    pooled_latent = all_latent.mean(axis=1)
    _atomic_write_npz(
        final_npz,
        {
            "ligand_latent": all_latent,
            "pooled_latent": pooled_latent,
            "frame_index": np.arange(frame_count, dtype=np.int64),
            "ligand_topology_index": np.asarray(ligand_indices, dtype=np.int64),
            "node_count": np.asarray(node_counts, dtype=np.int32),
            "edge_count": np.asarray(edge_counts, dtype=np.int32),
            "hop_counts": all_hop_counts,
            "max_outgoing_neighbors": np.asarray(max_neighbors, dtype=np.int32),
            "topology_indices_sha256": np.asarray(topology_hashes, dtype="<U64"),
        },
    )
    npz_sha = _sha256_file(final_npz)
    body = {
        "schema_version": "exp012-orb-latent-cache-v1",
        "status": "COMPLETED_LATENT_ONLY_NOT_JOINED_WITH_LEDGER",
        "run_id": run_id,
        "trajectory": {"path": str(trajectory_path), "sha256": _sha256_file(trajectory_path)},
        "topology": {"path": str(topology_path), "sha256": _sha256_file(topology_path)},
        "preregistration_sha256": registration_sha256,
        "frame_count": frame_count,
        "latent_shape": [frame_count, len(ligand_indices), 256],
        "node_count_range": [min(node_counts), max(node_counts)],
        "edge_count_range": [min(edge_counts), max(edge_counts)],
        "max_outgoing_neighbors_range": [min(max_neighbors), max(max_neighbors)],
        "cap_hit_frame_count": 0,
        "graph_cutoff_angstrom": 6.0,
        "max_num_neighbors": 120,
        "graph_membership_device": "cpu",
        "graph_membership_dtype": "float64",
        "closure_graph_device": "cpu",
        "closure_graph_dtype": "float64",
        "closure_graph_edge_method": "project._build_cutoff_edges_chunked_mic",
        "orb_input_graph_device": str(adapter.spec.device),
        "orb_input_graph_dtype": adapter.spec.graph_construction_dtype,
        "orb_input_graph_edge_method": adapter.spec.edge_method,
        "orb_input_graph_output_dtype": adapter.spec.output_dtype,
        "edge_set_equivalence_gate": "per_frame_fail_fast",
        "edge_set_equivalent_all_cached_frames": True,
        "half_supercell": adapter.spec.half_supercell,
        "wrap": adapter.spec.wrap,
        "model": {
            "model_name": model_name,
            "model_path": str(adapter.model_path),
            "model_sha256": adapter.model_sha256,
            "checkpoint_size_bytes": adapter.model_size_bytes,
            "orb_models_version": adapter.provenance["orb_models_version"],
            "orb_models_source_commit": adapter.provenance["orb_models_source_commit"],
            "torch_version": adapter.provenance["torch_version"],
            "device": adapter.spec.device,
            "precision": adapter.spec.precision,
            "compile": False,
            "layer": 2,
            "latent_dimension": 256,
        },
        "initialization_timing": adapter.load_timing,
        "conditioning_contract": contract.to_dict(),
        "npz_path": str(final_npz.resolve()),
        "npz_sha256": npz_sha,
        "ligand_indices": ligand_indices,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_count_created_this_run": checkpoint_count,
        "policy": {
            "primary_layer_frozen": True,
            "l5_full_cache_not_pursued": True,
            "representation_only": True,
            "coordinate_gradient_computed": False,
            "orb_total_energy_used_as_target": False,
            "fragment_energy_subtraction_used": False,
            "frame_graph_cap_fail_closed": True,
            "ledger_joined": False,
            "training_executed": False,
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_write_json(final_report, report)
    print(f"{run_id}: {report['report_sha256']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default="protocols/EXP-012_preregistration.json")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--run", action="append", required=True, type=_parse_run, metavar="RUN_ID=TRAJECTORY")
    parser.add_argument("--ligand-indices", required=True, help="comma-separated topology indices")
    parser.add_argument("--model-name", default="orb-v3-conservative-omol")
    parser.add_argument("--model-path", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--work-dir")
    args = parser.parse_args(argv)

    import numpy as np

    if not args.run:
        parser.error("at least one --run is required")
    run_ids = [run_id for run_id, _ in args.run]
    if len(set(run_ids)) != len(run_ids):
        parser.error("--run IDs must be unique")
    ligand_indices = [int(value) for value in args.ligand_indices.split(",") if value.strip()]
    if not ligand_indices or len(set(ligand_indices)) != len(ligand_indices):
        parser.error("--ligand-indices must be non-empty and unique")

    output_dir = Path(args.output_dir)
    work_parent = Path(args.work_dir) if args.work_dir else output_dir / ".work"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_parent.mkdir(parents=True, exist_ok=True)

    registration_path = Path(args.preregistration)
    if not registration_path.is_absolute():
        registration_path = ROOT / registration_path
    print("loading EXP-012 registration", flush=True)
    registration = load_preregistration(registration_path, workspace_root=ROOT, verify_files=True)
    contract = OrbParentConditioningContract(role="primary")
    contract.validate()
    print("loading frozen ORB model", flush=True)
    adapter = OrbLatentAdapter(
        OrbModelSpec(model_name=args.model_name, model_path=args.model_path, primary_layer=2)
    )
    print("frozen ORB model loaded", flush=True)
    topology_path = Path(args.topology).resolve()
    if not topology_path.is_file():
        parser.error(f"topology does not exist: {topology_path}")
    for run_id, trajectory_path in args.run:
        if not trajectory_path.is_file():
            parser.error(f"trajectory does not exist for {run_id}: {trajectory_path}")
        _run_cache(
            run_id=run_id,
            trajectory_path=trajectory_path,
            topology_path=topology_path,
            ligand_indices=ligand_indices,
            adapter=adapter,
            contract=contract,
            output_dir=output_dir,
            work_parent=work_parent,
            registration_sha256=registration.payload_sha256,
            model_name=args.model_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
