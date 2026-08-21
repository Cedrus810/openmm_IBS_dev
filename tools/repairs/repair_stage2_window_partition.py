#!/usr/bin/env python
"""One-off repair for a Stage 2 (vanishing) lambda-refinement result that was
corrupted by a bug in partition_windows_by_delta_f_budget (abfe_preoptimizer.py).

Background
----------
partition_windows_by_delta_f_budget used to test
    abs(f_k[end+1] - f_k[start]) <= max_window_span_kJ
i.e. the NET displacement between a window's start and a candidate end point,
instead of the cumulative total variation (sum of |consecutive differences|)
along the path. When the measured f(lambda) curve is not strictly monotonic
(very plausible for a "medium probe" refinement pass -- much less sampling
than production, and the softcore/WCA-shield potential does not guarantee a
monotonic dG/dlambda everywhere), a window can wander away from its starting
value and back close to it again, keeping the net displacement small while
the path it actually traced was large. That produced exactly what was
observed: an 18-state, 12-window partition with one window swallowing 7
states, `(2, 9)`, sitting next to normal 3-state windows.

The bug has been fixed (now uses a cumulative |Delta f| prefix sum). But by
the time it was caught, the pipeline had already run the medium-probe
refinement once with the buggy code, and overwrote
checkpoints/preopt_dual_vanishing.json with the bad 12-window result
(provenance.source == "refine_stage_lambda_path_from_data", so a plain
--resume will NOT re-run the refinement -- it thinks it's already done).

This script does NOT re-run any MD. The medium-probe scratch sampling in
vanishing_refine_probe/ is still valid (only the post-processing partition
step was buggy, not the sampling itself). It:

  1. Reads provenance.prior_window_ranges from the current (corrupted) preopt
     file -- this is the *original* 4-window structure the scratch data was
     actually sampled under, which the pipeline itself recorded before
     overwriting the file.
  2. Stages a preopt file with that prior window structure and the prior
     lambda schedule (hardcoded below from the run's own log output --
     verify PRIOR_LAMBDAS_VAR against your log before trusting this without
     checking).
  3. Re-calls refine_stage_lambda_path_from_data (now with the fixed
     partitioning code) against the existing scratch data in
     vanishing_refine_probe/, producing a corrected window partition, and
     writes it back to checkpoints/preopt_dual_vanishing.json (auto-backed up
     to the same path + ".bak" by that function, same as a normal run).

Usage (no GPU/PBS needed -- just needs the same Python env, for `import
openmm` inside ibs_engine.py/abfe_preoptimizer.py; solve_stage_integrated
itself does not touch a Platform/Context):

    python tools/repairs/repair_stage2_window_partition.py ./output_lrc_fix

After this, resume normally:

    python runabfe.py --config abfe_config.json --ligand MOL \
        --output ./output_lrc_fix --resume
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

# The lambda schedule that was actually used to produce the data currently
# sitting in vanishing_refine_probe/ -- copied verbatim from the PBS job's own
# printed "Stage 2 路径优化完成" line for this run. If you have re-run anything
# since and the log printed a *different* array, replace this list with that
# one before running -- this script trusts this list, it does not re-derive it.
PRIOR_LAMBDAS_VAR = [
    1.0, 0.97767033, 0.95534067, 0.933011, 0.91068133, 0.88835166,
    0.85991232, 0.82922876, 0.79665252, 0.76349016, 0.72568166, 0.68936523,
    0.65468917, 0.60699052, 0.52124044, 0.42479788, 0.30417162, 0.0,
]


def main(argv=None):
    output_dir = (argv or sys.argv[1:])[0] if (argv or sys.argv[1:]) else "./output_lrc_fix"
    scratch_dir = os.path.join(output_dir, "vanishing_refine_probe")
    preopt2_file = os.path.join(output_dir, "checkpoints", "preopt_dual_vanishing.json")

    if not os.path.isdir(scratch_dir):
        print(f"ERROR: scratch dir not found: {scratch_dir}", file=sys.stderr)
        return 2
    if not os.path.isfile(preopt2_file):
        print(f"ERROR: preopt file not found: {preopt2_file}", file=sys.stderr)
        return 2

    with open(preopt2_file, "r") as f:
        current = json.load(f)

    provenance = current.get("provenance") or {}
    if provenance.get("source") != "refine_stage_lambda_path_from_data":
        print(
            "WARNING: current preopt file does not look like it was already refined "
            "(no matching provenance.source) -- this script is meant to REPAIR an "
            "already-refined-but-buggy result. Double check you're pointing at the "
            "right --output directory before proceeding.",
            file=sys.stderr,
        )
    prior_window_ranges = provenance.get("prior_window_ranges")
    if not prior_window_ranges:
        print("ERROR: no provenance.prior_window_ranges found in the preopt file; "
              "cannot reconstruct what the scratch data was sampled under.", file=sys.stderr)
        return 2

    n_expected = len(PRIOR_LAMBDAS_VAR)
    covered = sorted({i for s, e in prior_window_ranges for i in range(s, e)})
    if covered != list(range(n_expected)):
        print(
            f"ERROR: prior_window_ranges {prior_window_ranges} do not cover "
            f"range(0, {n_expected}) implied by PRIOR_LAMBDAS_VAR -- update "
            "PRIOR_LAMBDAS_VAR to match this run's actual log output before rerunning.",
            file=sys.stderr,
        )
        return 2

    staged = {
        "lambdas_var": PRIOR_LAMBDAS_VAR,
        "window_ranges": [list(r) for r in prior_window_ranges],
        "n_states": n_expected,
    }
    with open(preopt2_file, "w") as f:
        json.dump(staged, f, indent=2)
    print(f"Staged pre-refinement preopt state (window_ranges={staged['window_ranges']}) at {preopt2_file}")

    from abfe_preoptimizer import refine_stage_lambda_path_from_data

    result = refine_stage_lambda_path_from_data(
        stage_dir=scratch_dir,
        preopt_path=preopt2_file,
        temperature_k=300.0,
        n_states=n_expected,
        max_window_span_kJ=35.0,
        overlap=2,
        stage_type="vdw",
    )
    print("Repartitioned with the fixed partition_windows_by_delta_f_budget:")
    print(f"  n_states: {result['n_states']}")
    print(f"  window_ranges: {result['window_ranges']}")
    print(f"  window sizes: {[e - s for s, e in result['window_ranges']]}")
    print(f"(backup of the pre-repair file written to {preopt2_file}.bak)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
