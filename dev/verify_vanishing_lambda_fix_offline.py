#!/usr/bin/env python
"""Read-only preview of the Stage-2 v18 Fisher+human lambda schedule."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from abfe_preoptimizer import (
    VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
    VANISHING_PROBE_BASE_STATE_COUNT,
    redistribute_vanishing_lambda_subdomains,
    validate_single_shared_boundary_ranges,
)


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    output_dir = args[0] if args else "./output"
    path = os.path.join(output_dir, "checkpoints", "preopt_dual_vanishing.json")
    with open(path, "r", encoding="utf-8") as handle:
        cached = json.load(handle)
    diag = cached.get("path_diagnostics") or {}
    pilot_lambdas = np.asarray(diag.get("pilot_lambdas") or [], dtype=float)
    metric_g = np.asarray(diag.get("metric_g") or [], dtype=float)
    if pilot_lambdas.size < 2 or pilot_lambdas.size != metric_g.size:
        print(f"ERROR: {path} 没有完整 pilot_lambdas/metric_g", file=sys.stderr)
        return 2

    lambdas, _cumulative, _edges, ranges, allocation = (
        redistribute_vanishing_lambda_subdomains(
            pilot_lambdas,
            metric_g,
            VANISHING_PROBE_BASE_STATE_COUNT,
            first_ensemble_target_intervals=(
                VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS
            ),
        )
    )
    validate_single_shared_boundary_ranges(ranges, len(lambdas))
    print(f"pilot cache: {path}")
    print(f"probe base (17): {allocation['probe_base_lambdas']}")
    print(f"final lambda_0..lambda_20: {lambdas.tolist()}")
    print(f"Python half-open windows: {ranges}")
    print("Human closed windows: [(0,5),(5,9),(9,13),(13,17),(17,20)]")
    print(f"window state slots: {allocation['total_window_state_slots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
