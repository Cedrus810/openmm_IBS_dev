#!/usr/bin/env python
"""Run the ORB-001a actual-edge equivalence gate on selected frames.

The gate constructs the exact L2 closure on CPU float64, then sends that
closure through the official ORB ASE adapter with an explicit graph backend.
It compares canonical topology-index edge identities, per-node neighbor
counts, and the 120-neighbor cap state before any cache is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.orb_graph import audit_lhop_graphs  # noqa: E402
from local_residual.orb_latent import (  # noqa: E402
    OrbLatentAdapter,
    OrbLatentError,
    OrbModelSpec,
    OrbParentConditioningContract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--run", action="append", required=True, metavar="RUN_ID=TRAJECTORY")
    parser.add_argument("--frame-index", action="append", required=True, type=int)
    parser.add_argument("--ligand-indices", required=True)
    parser.add_argument("--model-name", default="orb-v3-conservative-omol")
    parser.add_argument("--model-path", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    import mdtraj as md
    import numpy as np

    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    if len(args.run) != len(args.frame_index):
        parser.error("--run and --frame-index must have equal lengths")
    run_pairs = []
    for value, frame_index in zip(args.run, args.frame_index):
        if "=" not in value:
            parser.error("--run must use RUN_ID=TRAJECTORY")
        run_id, trajectory = value.split("=", 1)
        run_pairs.append((run_id, Path(trajectory).expanduser().resolve(), int(frame_index)))
    ligand_indices = [int(value) for value in args.ligand_indices.split(",") if value.strip()]
    if not ligand_indices:
        parser.error("--ligand-indices must not be empty")

    print("loading frozen ORB model for ORB-001a", flush=True)
    adapter = OrbLatentAdapter(
        OrbModelSpec(model_name=args.model_name, model_path=args.model_path, primary_layer=2)
    )
    contract = OrbParentConditioningContract(role="primary")
    contract.validate()
    frames = []
    all_passed = True
    started = time.perf_counter()
    for run_id, trajectory_path, frame_index in run_pairs:
        record = {
            "run_id": run_id,
            "frame_index": frame_index,
            "trajectory": str(trajectory_path),
            "passed": False,
        }
        try:
            trajectory = md.load_frame(str(trajectory_path), frame_index, top=str(Path(args.topology)))
            if trajectory.unitcell_vectors is None:
                raise RuntimeError("trajectory has no periodic cell")
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
                raise RuntimeError("CPU float64 closure already reaches the 120-neighbor cap")
            topology_indices = np.asarray(layer2["topology_indices"], dtype=np.int64)
            local_index_by_topology = {int(value): index for index, value in enumerate(topology_indices)}
            local_ligand_indices = [local_index_by_topology[index] for index in ligand_indices]
            atomic_numbers = [
                int(trajectory.topology.atom(int(index)).element.atomic_number)
                for index in topology_indices
            ]
            result = adapter.extract_frame(
                positions[topology_indices],
                cell,
                atomic_numbers=atomic_numbers,
                ligand_indices=local_ligand_indices,
                topology_indices=topology_indices,
                conditioning_contract=contract,
                layer=2,
                verify_edge_equivalence=True,
            )
            record.update(
                {
                    "passed": bool(result.diagnostics["edge_set_equivalent"]),
                    "node_count": result.diagnostics["node_count"],
                    "edge_count": result.diagnostics["edge_count"],
                    "max_outgoing_neighbors": result.diagnostics["orb_input_graph"]["max_outgoing_neighbors"],
                    "cap_hit": result.diagnostics["orb_input_graph"]["cap_hit"],
                    "edge_set_sha256": result.diagnostics["edge_set_sha256"],
                    "closure_edge_set_sha256": result.diagnostics["closure_edge_set_sha256"],
                    "edge_set_comparison": result.diagnostics["edge_set_comparison"],
                    "orb_diagnostics": result.diagnostics,
                }
            )
        except (OrbLatentError, RuntimeError, KeyError, ValueError) as exc:
            record["error"] = str(exc)
            all_passed = False
        all_passed = all_passed and bool(record["passed"])
        frames.append(record)
        print(f"{run_id} frame {frame_index}: {'PASS' if record['passed'] else 'FAIL'}", flush=True)

    body = {
        "schema_version": "orb-001a-edge-equivalence-v1",
        "status": "PASSED_ORB_001A" if all_passed else "FAILED_ORB_001A",
        "gate": "ORB-001a",
        "all_frames_passed": all_passed,
        "frame_count": len(frames),
        "frames": frames,
        "inputs": {
            "topology": {"path": str(Path(args.topology).resolve()), "sha256": _sha256(Path(args.topology))},
            "ligand_indices": ligand_indices,
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
        "initialization_timing": adapter.load_timing,
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "formal_1500_frame_cache_started": False,
            "layer": 2,
            "l5_full_cache_not_pursued": True,
            "fail_fast_on_any_edge_or_cap_mismatch": True,
        },
    }
    body["report_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": body["status"], "all_frames_passed": all_passed}, sort_keys=True))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
