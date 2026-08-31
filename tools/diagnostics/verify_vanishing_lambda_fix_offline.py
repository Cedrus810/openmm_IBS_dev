#!/usr/bin/env python
"""Read-only preview of the Stage-2 v21 metric-blended lambda schedule.

Usage: pass either the run output directory or the cached json itself --
    verify_vanishing_lambda_fix_offline.py output
    verify_vanishing_lambda_fix_offline.py .../checkpoints/preopt_dual_vanishing.json
"""

from __future__ import annotations

# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))


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
    target = args[0] if args else "./output"
    # 收输出根目录或直接收缓存文件本身；否则很容易把已经拼好的路径再拼一次。
    if os.path.isdir(target):
        path = os.path.join(target, "checkpoints", "preopt_dual_vanishing.json")
    else:
        path = target
    if not os.path.isfile(path):
        print(
            f"ERROR: 找不到 {path}（参数可以是运行输出目录，也可以是 "
            "preopt_dual_vanishing.json 本身）",
            file=sys.stderr,
        )
        return 2
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
    print(f"placement: {allocation['base_lambda_placement']}")
    print(
        f"geometric floor beta={allocation['geometric_floor_weight']}, "
        f"max|Δλ|={allocation['realized_max_lambda_gap']:.4f} "
        f"(bound {allocation['max_lambda_gap_bound']:.4f})"
    )
    print(
        f"edge thermodynamic length: max="
        f"{allocation['realized_max_edge_thermodynamic_length']:.4f}, min="
        f"{allocation['realized_min_edge_thermodynamic_length']:.4f}"
    )
    print(f"final lambda_0..lambda_22: {lambdas.tolist()}")
    print(f"Python half-open windows: {ranges}")
    print(f"window state slots: {allocation['total_window_state_slots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
