#!/usr/bin/env python
"""Measure ORB-001b cold initialization, warm extraction, and scalar backward.

This benchmark never writes a latent cache.  It reuses the same edge
equivalence fail-fast as ORB-001a and records loader phases, graph construction,
prefix forward, scalar coordinate backward, wall time, and process RSS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.orb_graph import audit_lhop_graphs  # noqa: E402
from local_residual.orb_latent import (  # noqa: E402
    OrbLatentAdapter,
    OrbModelSpec,
    OrbParentConditioningContract,
)


def _rss_kb() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_inputs(trajectory, ligand_indices):
    import numpy as np

    positions = np.asarray(trajectory.xyz[0], dtype=np.float64) * 10.0
    cell = np.asarray(trajectory.unitcell_vectors[0], dtype=np.float64) * 10.0
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
        raise RuntimeError("CPU float64 closure reaches the official 120-neighbor cap")
    topology_indices = np.asarray(layer2["topology_indices"], dtype=np.int64)
    local_map = {int(value): index for index, value in enumerate(topology_indices)}
    local_ligand = [local_map[index] for index in ligand_indices]
    atomic_numbers = [
        int(trajectory.topology.atom(int(index)).element.atomic_number)
        for index in topology_indices
    ]
    return positions[topology_indices], cell, atomic_numbers, local_ligand, topology_indices, layer2


def _extract(adapter, trajectory, ligand_indices, contract, *, require_grad: bool):
    positions, cell, atomic_numbers, local_ligand, topology_indices, layer2 = _frame_inputs(
        trajectory, ligand_indices
    )
    started = time.perf_counter()
    result = adapter.extract_frame(
        positions,
        cell,
        atomic_numbers=atomic_numbers,
        ligand_indices=local_ligand,
        topology_indices=topology_indices,
        conditioning_contract=contract,
        layer=2,
        require_coordinate_grad=require_grad,
        verify_edge_equivalence=True,
    )
    extraction_seconds = time.perf_counter() - started
    return result, layer2, extraction_seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--ligand-indices", required=True)
    parser.add_argument("--model-name", default="orb-v3-conservative-omol")
    parser.add_argument("--model-path", default="auto")
    parser.add_argument("--warm-frame-count", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    import mdtraj as md
    import torch

    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    if args.warm_frame_count < 1:
        parser.error("--warm-frame-count must be positive")
    ligand_indices = [int(value) for value in args.ligand_indices.split(",") if value.strip()]
    topology_path = Path(args.topology).resolve()
    trajectory_path = Path(args.trajectory).resolve()
    if not topology_path.is_file() or not trajectory_path.is_file():
        parser.error("topology and trajectory must exist")
    load_started = time.perf_counter()
    adapter = OrbLatentAdapter(
        OrbModelSpec(model_name=args.model_name, model_path=args.model_path, primary_layer=2)
    )
    load_seconds = time.perf_counter() - load_started
    contract = OrbParentConditioningContract(role="primary")
    contract.validate()

    def get_frame(index: int):
        frame = md.load_frame(str(trajectory_path), index, top=str(topology_path))
        if frame.unitcell_vectors is None:
            raise RuntimeError(f"frame {index} has no periodic cell")
        return frame

    cold_frame = get_frame(0)
    cold_started = time.perf_counter()
    cold_result, cold_layer, cold_extract_seconds = _extract(
        adapter, cold_frame, ligand_indices, contract, require_grad=False
    )
    cold_wall_seconds = time.perf_counter() - cold_started

    warm = []
    warm_started = time.perf_counter()
    for frame_index in range(1, args.warm_frame_count + 1):
        frame = get_frame(frame_index)
        result, layer2, extract_seconds = _extract(
            adapter, frame, ligand_indices, contract, require_grad=False
        )
        warm.append(
            {
                "frame_index": frame_index,
                "node_count": layer2["node_count"],
                "edge_count": layer2["edge_count"],
                "max_outgoing_neighbors": result.diagnostics["orb_input_graph"]["max_outgoing_neighbors"],
                "extract_seconds": extract_seconds,
                "timing": result.diagnostics["timing"],
                "edge_set_sha256": result.diagnostics["edge_set_sha256"],
            }
        )
    warm_wall_seconds = time.perf_counter() - warm_started

    backward_frame = get_frame(0)
    backward_started = time.perf_counter()
    backward_result, backward_layer, backward_extract_seconds = _extract(
        adapter, backward_frame, ligand_indices, contract, require_grad=True
    )
    scalar = backward_result.ligand_latent.mean()
    gradient = torch.autograd.grad(scalar, backward_result.batch.node_features["positions"])[0]
    backward_seconds = time.perf_counter() - backward_started
    if not bool(torch.isfinite(gradient).all().item()):
        raise RuntimeError("scalar coordinate backward produced non-finite gradients")

    body = {
        "schema_version": "orb-001b-initialization-benchmark-v1",
        "status": "COMPLETED_ORB_001B_BENCHMARK",
        "gate": "ORB-001b",
        "inputs": {
            "topology": {"path": str(topology_path), "sha256": _sha256(topology_path)},
            "trajectory": {"path": str(trajectory_path), "sha256": _sha256(trajectory_path)},
            "ligand_indices": ligand_indices,
            "cold_frame_index": 0,
            "warm_frame_indices": list(range(1, args.warm_frame_count + 1)),
        },
        "model": adapter.provenance,
        "model_spec": {
            "edge_method": adapter.spec.edge_method,
            "graph_construction_dtype": adapter.spec.graph_construction_dtype,
            "output_dtype": adapter.spec.output_dtype,
            "device": adapter.spec.device,
            "compile": False,
            "half_supercell": adapter.spec.half_supercell,
            "wrap": adapter.spec.wrap,
        },
        "conditioning_contract": contract.to_dict(),
        "initialization": {
            "load_wall_seconds": load_seconds,
            "load_rss_peak_kb": _rss_kb(),
            "phases": adapter.load_timing,
        },
        "cold_1_frame": {
            "frame_index": 0,
            "node_count": cold_layer["node_count"],
            "edge_count": cold_layer["edge_count"],
            "extract_wall_seconds": cold_wall_seconds,
            "extract_seconds": cold_extract_seconds,
            "timing": cold_result.diagnostics["timing"],
            "edge_set_equivalent": cold_result.diagnostics["edge_set_equivalent"],
            "rss_peak_kb": _rss_kb(),
        },
        "warm_10_frames": {
            "frame_count": len(warm),
            "wall_seconds": warm_wall_seconds,
            "mean_extract_seconds": sum(item["extract_seconds"] for item in warm) / len(warm),
            "min_extract_seconds": min(item["extract_seconds"] for item in warm),
            "max_extract_seconds": max(item["extract_seconds"] for item in warm),
            "rss_peak_kb": _rss_kb(),
            "frames": warm,
        },
        "scalar_backward": {
            "frame_index": 0,
            "extract_with_grad_seconds": backward_extract_seconds,
            "end_to_end_seconds": backward_seconds,
            "gradient_shape": [int(value) for value in gradient.shape],
            "gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
            "gradient_finite": True,
            "timing": backward_result.diagnostics["timing"],
            "rss_peak_kb": _rss_kb(),
        },
        "policy": {
            "formal_1500_frame_cache_started": False,
            "edge_equivalence_fail_fast": True,
            "l5_full_cache_not_pursued": True,
        },
    }
    body["report_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": body["status"], "warm_frames": len(warm)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
