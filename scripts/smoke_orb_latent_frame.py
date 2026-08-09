#!/usr/bin/env python
"""Run one reproducible ORB shallow-latent frame smoke.

This command is an ORB-000/001 preflight only.  It first computes the exact
CPU-float64 6-A local L-hop closure, then feeds that selected local graph to
the official ORB ASE adapter and explicitly executes the requested GNS prefix.
It writes representation-only output: no ORB energy head and no MM ledger
target are used.

The charge/spin contract is explicit and recorded in the report.  Unless the
caller marks it ``frozen``, the output is deliberately non-primary and cannot
be used as an EXP-012 join artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.orb_graph import audit_lhop_graphs  # noqa: E402
from local_residual.orb_latent import (  # noqa: E402
    OrbLatentAdapter,
    OrbModelSpec,
    OrbParentConditioningContract,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _latent_summary(latent) -> dict:
    import numpy as np

    array = np.asarray(latent, dtype=np.float32)
    digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
    return {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "l2_norm": float(np.linalg.norm(array)),
        "sha256_raw_float32": digest,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--ligand-indices", required=True, help="comma-separated topology indices")
    parser.add_argument("--model-name", default="orb-v3-conservative-omol")
    parser.add_argument("--model-path", default="auto")
    parser.add_argument("--layer", type=int, default=2)
    parser.add_argument("--cutoff-angstrom", type=float, default=6.0)
    parser.add_argument("--max-num-neighbors", type=int, default=120)
    parser.add_argument("--total-charge", type=float, required=True)
    parser.add_argument("--spin-multiplicity", type=float, required=True)
    parser.add_argument(
        "--charge-spin-contract-status",
        choices=("unfrozen-exploratory", "frozen-parent-system"),
        required=True,
        help="mark whether the parent-system conditioning contract is frozen",
    )
    parser.add_argument("--output", required=True, help="JSON smoke report; refuses overwrite")
    parser.add_argument("--latent-output", help="optional representation-only NPZ; refuses overwrite")
    args = parser.parse_args(argv)

    import mdtraj as md
    import numpy as np

    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    latent_output = Path(args.latent_output) if args.latent_output else None
    if latent_output is not None and latent_output.exists():
        parser.error(f"refusing to overwrite existing latent NPZ: {latent_output}")
    ligand_indices = [int(value) for value in args.ligand_indices.split(",") if value.strip()]
    if not ligand_indices:
        parser.error("--ligand-indices must not be empty")
    if args.layer != 2:
        parser.error("this pre-registered primary smoke is fixed to layer 2")
    if args.cutoff_angstrom != 6.0 or args.max_num_neighbors != 120:
        parser.error("the official ORB-v3 inf graph audit is fixed to 6.0 A and 120 neighbors")

    topology_path = Path(args.topology).resolve()
    trajectory_path = Path(args.trajectory).resolve()
    trajectory = md.load_frame(str(trajectory_path), args.frame_index, top=str(topology_path))
    if trajectory.unitcell_vectors is None:
        raise SystemExit("trajectory has no periodic cell vectors")
    positions = np.asarray(trajectory.xyz[0], dtype=np.float64) * 10.0
    cell = np.asarray(trajectory.unitcell_vectors[0], dtype=np.float64) * 10.0

    graph = audit_lhop_graphs(
        positions,
        cell,
        ligand_indices=ligand_indices,
        cutoff_angstrom=args.cutoff_angstrom,
        max_num_neighbors=args.max_num_neighbors,
        max_layer=2,
    )
    layer_report = graph["layers"][1]
    topology_indices = np.asarray(layer_report["topology_indices"], dtype=np.int64)
    local_index_by_topology = {int(value): index for index, value in enumerate(topology_indices)}
    try:
        local_ligand_indices = [local_index_by_topology[index] for index in ligand_indices]
    except KeyError as exc:
        raise SystemExit(f"layer-2 closure omitted ligand topology index {exc.args[0]}") from exc

    topology = trajectory.topology
    atomic_numbers = [int(topology.atom(int(index)).element.atomic_number) for index in topology_indices]
    local_positions = positions[topology_indices]

    adapter = OrbLatentAdapter(
        OrbModelSpec(
            model_name=args.model_name,
            model_path=args.model_path,
            primary_layer=args.layer,
            cutoff_angstrom=args.cutoff_angstrom,
            max_num_neighbors=args.max_num_neighbors,
        )
    )
    contract = OrbParentConditioningContract(
        total_charge=args.total_charge,
        spin_multiplicity=args.spin_multiplicity,
        role="primary",
    )
    result = adapter.extract_frame(
        local_positions,
        cell,
        atomic_numbers=atomic_numbers,
        ligand_indices=local_ligand_indices,
        topology_indices=topology_indices,
        conditioning_contract=contract,
        layer=args.layer,
        require_coordinate_grad=False,
    )
    latent = result.ligand_latent.detach().cpu().numpy()
    if latent.shape != (len(ligand_indices), 256):
        raise SystemExit(f"unexpected ORB ligand latent shape: {latent.shape}")

    if latent_output is not None:
        latent_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            latent_output,
            ligand_latent=np.asarray(latent, dtype=np.float32),
            frame_index=np.asarray([args.frame_index], dtype=np.int64),
            ligand_topology_index=np.asarray(ligand_indices, dtype=np.int64),
            topology_indices=topology_indices,
        )

    body = {
        "schema_version": "orb-latent-frame-smoke-v1",
        "status": "COMPLETED_EXPLORATORY_SMOKE" if args.charge_spin_contract_status != "frozen-parent-system" else "COMPLETED_PRIMARY_CONTRACT_SMOKE",
        "primary_qualification": args.charge_spin_contract_status == "frozen-parent-system",
        "command": " ".join(sys.argv),
        "model_name": args.model_name,
        "layer": args.layer,
        "charge_spin_contract": {
            **contract.to_dict(),
            "status": args.charge_spin_contract_status,
        },
        "inputs": {
            "topology": {"path": str(topology_path), "sha256": _sha256(topology_path)},
            "trajectory": {"path": str(trajectory_path), "sha256": _sha256(trajectory_path)},
            "frame_index": args.frame_index,
            "ligand_indices": ligand_indices,
        },
        "graph": {
            "cutoff_angstrom": graph["cutoff_angstrom"],
            "max_num_neighbors": graph["max_num_neighbors"],
            "layer": layer_report["layer"],
            "node_count": layer_report["node_count"],
            "edge_count": layer_report["edge_count"],
            "hop_counts": layer_report["hop_counts"],
            "max_outgoing_neighbors": layer_report["max_outgoing_neighbors"],
            "cap_hit_node_count": layer_report["cap_hit_node_count"],
            "cap_hit_edge_count": layer_report["cap_hit_edge_count"],
            "cap_hit": layer_report["cap_hit"],
            "local_ligand_indices": local_ligand_indices,
        },
        "orb": result.diagnostics,
        "latent": _latent_summary(latent),
        "latent_output": (
            {"path": str(latent_output.resolve()), "sha256": _sha256(latent_output)}
            if latent_output is not None else None
        ),
        "policy": {
            "representation_only": True,
            "orb_total_energy_used_as_target": False,
            "full_1500_frame_probe_executed": False,
            "primary_layer_frozen": True,
            "charge_spin_required_before_primary_probe": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "latent_shape": body["latent"]["shape"], "primary_qualification": body["primary_qualification"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
