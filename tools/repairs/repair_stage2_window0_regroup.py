#!/usr/bin/env python
"""One-off, LOCAL-ONLY regroup of the cached Stage 2 (vanishing) window
partition, to skip re-running the ~20-minute pilot probe when only the
GROUPING changed (THERMODYNAMIC_PATH_PROTOCOL_VERSION 12/13 -> 14: window 0
now uses VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS instead of the uniform
target), not the underlying lambda placement.

Why this is safe to do without rerunning the pilot
---------------------------------------------------
`redistribute_vanishing_lambda_subdomains` has two independent steps:
  1. `redistribute_lambda_by_thermodynamic_length` computes `lambdas_var` from
     `pilot_lambdas`/`metric_g` -- unaffected by `first_ensemble_target_intervals`.
  2. `vanishing_subdomain_ranges_from_lambdas` groups the ALREADY-computed
     `lambdas_var` into windows -- this is the ONLY thing the new parameter
     changes.
Since step 1's inputs (pilot_lambdas/metric_g, i.e. the actual GPU-measured
data) haven't changed, the cached `lambdas_var` is still exactly correct;
only `window_ranges` needs to be recomputed, using the exact same generator
function the real pipeline uses (not hand-picked numbers).

`protocol_key`/`provenance`/`path_diagnostics` are left untouched: the narrow
preopt fingerprint (`_stage2_preopt_key` in abfe_pipeline.py) does not include
a code hash, only physical inputs (potential_type/Boresch/temperature/etc.),
none of which changed -- only the window-grouping helper's internal logic
did. `path_protocol_version` is bumped to the current
THERMODYNAMIC_PATH_PROTOCOL_VERSION so abfe_pipeline.py's
`path_protocol_match` check (used by the auto-repair-source allowlist) is
also consistent, even though this run doesn't need that path (n_states is
unchanged at 18, so the plain `len(cached_lambdas) == stage2_states` branch
already accepts it).

Usage (no GPU needed -- pure Python/numpy, same env as the rest of the repo):

    python tools/repairs/repair_stage2_window0_regroup.py ./output

Then resume normally:

    python runabfe.py --config abfe_config.json --ligand MOL --resume
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
import shutil
import sys


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    output_dir = args[0] if args else "./output"
    preopt2_file = os.path.join(output_dir, "checkpoints", "preopt_dual_vanishing.json")

    if not os.path.isfile(preopt2_file):
        print(f"ERROR: {preopt2_file} not found.", file=sys.stderr)
        return 2

    with open(preopt2_file, "r") as f:
        cached = json.load(f)

    lambdas_var = cached.get("lambdas_var")
    old_window_ranges = cached.get("window_ranges")
    if not lambdas_var or not old_window_ranges:
        print(
            f"ERROR: {preopt2_file} is missing lambdas_var/window_ranges -- "
            "nothing to regroup.",
            file=sys.stderr,
        )
        return 2

    from abfe_preoptimizer import (
        vanishing_subdomain_ranges_from_lambdas,
        VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
        THERMODYNAMIC_PATH_PROTOCOL_VERSION,
    )

    new_window_ranges = vanishing_subdomain_ranges_from_lambdas(
        lambdas_var,
        first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
    )

    print(f"Old window_ranges ({len(old_window_ranges)} windows): {old_window_ranges}")
    print(f"New window_ranges ({len(new_window_ranges)} windows): {new_window_ranges}")
    print(f"New window state counts: {[e - s for s, e in new_window_ranges]}")
    print(
        f"Window 0: {lambdas_var[new_window_ranges[0][0]]:.4f} -> "
        f"{lambdas_var[new_window_ranges[0][1] - 1]:.4f} "
        f"({new_window_ranges[0][1] - new_window_ranges[0][0]} states)"
    )

    if [tuple(r) for r in old_window_ranges] == new_window_ranges:
        print("\nNo change -- old and new window_ranges are identical. Not writing anything.")
        return 0

    backup_path = preopt2_file + ".before_window0_regroup.bak"
    shutil.copy(preopt2_file, backup_path)
    print(f"\nBacked up current preopt file to {backup_path}")

    cached["window_ranges"] = [list(r) for r in new_window_ranges]
    cached["path_protocol_version"] = THERMODYNAMIC_PATH_PROTOCOL_VERSION
    # lambdas_var / protocol_key / provenance / path_diagnostics / n_states left untouched.

    with open(preopt2_file, "w") as f:
        json.dump(cached, f, indent=2)
    print(f"Wrote regrouped schedule to {preopt2_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
