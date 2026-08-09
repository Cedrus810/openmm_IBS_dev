#!/usr/bin/env python
"""Audit exact 6-A L-hop ORB graph scale and the official 120-neighbor cap."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--ligand-indices", required=True, help="comma-separated topology indices")
    parser.add_argument("--cutoff-angstrom", type=float, default=6.0)
    parser.add_argument("--max-num-neighbors", type=int, default=120)
    parser.add_argument("--max-layer", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    output = Path(args.output)
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    ligand_indices = [int(value) for value in args.ligand_indices.split(",") if value.strip()]
    if not ligand_indices:
        parser.error("--ligand-indices must not be empty")

    try:
        import mdtraj as md
        import numpy as np

        trajectory = md.load_frame(args.trajectory, args.frame_index, top=args.topology)
        if trajectory.unitcell_vectors is None:
            raise RuntimeError("trajectory has no periodic cell")
        positions = np.asarray(trajectory.xyz[0], dtype=np.float64) * 10.0
        cell = np.asarray(trajectory.unitcell_vectors[0], dtype=np.float64) * 10.0
        graph_report = audit_lhop_graphs(
            positions,
            cell,
            ligand_indices=ligand_indices,
            cutoff_angstrom=args.cutoff_angstrom,
            max_num_neighbors=args.max_num_neighbors,
            max_layer=args.max_layer,
        )
    except Exception as exc:
        raise SystemExit(f"ORB graph audit failed: {exc}") from exc

    body = {
        "schema_version": "orb-graph-audit-report-v1",
        "status": "COMPLETED_GRAPH_AUDIT",
        "command": " ".join(sys.argv),
        "inputs": {
            "topology": {"path": str(Path(args.topology).resolve()), "sha256": _sha256(Path(args.topology))},
            "trajectory": {"path": str(Path(args.trajectory).resolve()), "sha256": _sha256(Path(args.trajectory))},
            "frame_index": args.frame_index,
            "ligand_indices": ligand_indices,
        },
        "graph": graph_report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "layers": len(graph_report["layers"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
