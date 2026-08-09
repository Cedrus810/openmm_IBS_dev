#!/usr/bin/env python
"""Run the frozen EXP-012-compatible ORB layer-2 ridge/LOO probe.

The input NPZ is intentionally representation-only plus the already audited
MM-ledger arrays.  It must contain ``pooled_latent`` with shape ``(frames,
256)``, ``adjacent_gap_reduced``, ``log_importance_unnormalized``, ``delta_A``
and ``partition_index``.  This command never loads ORB weights and never
recomputes the MM ledger.
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

from local_residual.orb_probe import evaluate_loo_probe  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ridge-grid", nargs="+", type=float, default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0])
    parser.add_argument("--model-name", default="orb-v3-conservative-omol")
    parser.add_argument("--layer", type=int, default=2)
    args = parser.parse_args(argv)

    input_path = Path(args.features_npz)
    output = Path(args.output)
    if not input_path.is_file():
        parser.error(f"features NPZ does not exist: {input_path}")
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    if args.layer != 2:
        parser.error("primary ORB probe is frozen to layer 2; exploratory layers need a separate report")

    import numpy as np
        
    required = {
        "pooled_latent",
        "adjacent_gap_reduced",
        "log_importance_unnormalized",
        "delta_A",
        "partition_index",
    }
    with np.load(input_path, allow_pickle=False) as arrays:
        missing = sorted(required - set(arrays.files))
        if missing:
            parser.error(f"features NPZ is missing required arrays: {missing}")
        import torch

        values = {
            name: torch.tensor(arrays[name], dtype=torch.float64)
            for name in required
        }
    report = evaluate_loo_probe(
        values["pooled_latent"],
        values["adjacent_gap_reduced"],
        values["log_importance_unnormalized"],
        values["delta_A"],
        values["partition_index"].to(dtype=torch.int64),
        ridge_grid=tuple(args.ridge_grid),
    )
    body = {
        **report,
        "model_name": args.model_name,
        "layer": args.layer,
        "features_npz": {"path": str(input_path.resolve()), "sha256": _sha256(input_path)},
        "policy": {
            "primary_layer_frozen": True,
            "model_selection_from_outer_held_out_forbidden": True,
            "orb_total_energy_used_as_target": False,
            "a_k_frozen": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "promotion_passed": body["promotion_gate"]["passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
