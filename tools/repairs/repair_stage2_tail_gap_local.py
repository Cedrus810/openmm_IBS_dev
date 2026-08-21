#!/usr/bin/env python
"""Data-driven, LOCAL-ONLY repair of the Stage 2 (vanishing) lambda schedule's
tail gap. No hand-picked lambda values, no need to redo the whole stage.

Why this exists
----------------
partition_windows_by_delta_f_budget (abfe_preoptimizer.py) had a bug (now
fixed) that let one non-monotonic stretch of the medium-probe f(lambda) curve
produce a wildly oversized window. By the time it was caught, the pipeline had
already overwritten checkpoints/preopt_dual_vanishing.json with that bad
12-window result. A plain --resume will not fix this: it sees
provenance.source == "refine_stage_lambda_path_from_data" and thinks the
refinement is already done.

Re-running the full enable_lambda_refine pipeline path would redistribute ALL
18 lambda values from scratch and therefore force a resample of every window,
including the three (covering states 0-13) that never had a problem. This
script instead:

  1. Recovers the EXACT original (pre-refinement) 18-state schedule and
     4-window structure byte-for-byte from
     checkpoints/preopt_dual_vanishing.json.bak (the backup
     refine_stage_lambda_path_from_data wrote before its first, buggy run) --
     no manual retyping, no rounding.
  2. Solves the real, globally-stitched f(lambda) MBAR curve from the medium-probe
     data already sampled in vanishing_refine_probe/ under that exact original
     structure (that sampling itself was fine; only the downstream partition
     function was buggy).
  3. Flags per-step |Delta f| outliers automatically (default: > 3x the median
     of the other steps) -- does not assume "the last step is the bad one",
     discovers it from the real curve.
  4. For each flagged step, inserts ceil(step / median_other_step) - 1 new
     lambda points, evenly spaced across that step's own measured |Delta f|
     span (the only defensible default when there is no intermediate sampled
     data inside that specific gap).
  5. Every lambda value OUTSIDE a flagged step is kept 100% identical to the
     recovered original schedule.
  6. Re-partitions windows with the already-fixed partition_windows_by_delta_f_budget,
     reusing the same max_window_span_kJ/overlap constants used everywhere else
     in this pipeline (no new magic numbers).
  7. Backs up the current (buggy 12-window) preopt file and writes the repaired
     schedule to checkpoints/preopt_dual_vanishing.json.

This does NOT touch any dual_window_*_vdw_* production files. Windows covering
states 0-13 ((0,6),(4,10),(8,14)) are unchanged from the original design, but
their actual production .npy files were deleted on an earlier turn (an overly
broad `rm dual_window_*` -- not this script's doing) and will need exactly one
fresh resample; this script guarantees they will not need a second one because
of any further lambda/window churn.

Usage (needs the project's Python env for numpy/openmm imports; no GPU/PBS
required -- solve_stage_integrated only does MBAR math on cached arrays):

    python tools/repairs/repair_stage2_tail_gap_local.py ./output_lrc_fix

Then resume normally. Also set "enable_lambda_refine": false in
abfe_config.json first, so the next --resume does not immediately re-run the
global refiner and overwrite this local patch.
"""

from __future__ import annotations

# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))


import glob
import json
import os
import shutil
import sys

OUTLIER_FACTOR = 3.0       # a step is flagged if it's this many times the median of the OTHER steps
MAX_WINDOW_SPAN_KJ = 35.0  # same constant already used elsewhere in this pipeline (refine_max_window_span_kJ default)
OVERLAP = 2                # same constant already used elsewhere in this pipeline (refine_overlap default)


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    output_dir = args[0] if args else "./output_lrc_fix"
    scratch_dir = os.path.join(output_dir, "vanishing_refine_probe")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    preopt2_file = os.path.join(checkpoint_dir, "preopt_dual_vanishing.json")
    bak_file = preopt2_file + ".bak"

    if not os.path.isfile(bak_file):
        print(
            f"ERROR: {bak_file} not found -- this script expects the pre-refinement "
            "backup that refine_stage_lambda_path_from_data wrote automatically the "
            "first time it ran. Without it there's no byte-exact record of the "
            "original schedule to recover from.",
            file=sys.stderr,
        )
        return 2

    with open(bak_file, "r") as f:
        original = json.load(f)
    lambdas_var = list(original["lambdas_var"])
    window_ranges = [tuple(r) for r in original["window_ranges"]]
    n_states = original["n_states"]
    print(f"Recovered original schedule from {bak_file}: {n_states} states, windows {window_ranges}")

    import numpy as np
    from ibs_engine import solve_stage_integrated
    from abfe_preoptimizer import partition_windows_by_delta_f_budget

    e_files = sorted(glob.glob(os.path.join(scratch_dir, "dual_window_*_vdw_energies.npy")))
    if len(e_files) != len(window_ranges):
        print(
            f"ERROR: found {len(e_files)} window energy files in {scratch_dir}, expected "
            f"{len(window_ranges)} to match the recovered original window_ranges. Refusing "
            "to proceed against a mismatched dataset -- check the scratch dir wasn't "
            "partially overwritten by a later run.",
            file=sys.stderr,
        )
        return 2

    window_data = []
    for w_idx, e_file in enumerate(e_files):
        u_kn = np.load(e_file)
        bias = np.load(e_file.replace("_energies.npy", "_bias.npy"))
        base = np.load(e_file.replace("_energies.npy", "_base.npy"))
        start, end = window_ranges[w_idx]
        window_data.append({
            "u_kn": u_kn,
            "bias_energies": bias,
            "base_energies": base,
            "lambda_indices": list(range(start, end)),
        })

    kt = 0.008314462618 * 300.0
    res = solve_stage_integrated(window_data, kt, stage_name="vdw")
    if res.get("error"):
        print(f"ERROR: MBAR solve on the real medium-probe data failed: {res['error']}", file=sys.stderr)
        return 2

    lambdas_sorted = res["lambdas"]
    f_k = np.asarray(res["f_k"], dtype=float)
    lam_in_order = np.asarray([lambdas_var[i] for i in lambdas_sorted], dtype=float)
    # Force descending 1.0 -> 0.0 order regardless of how solve_stage_integrated returned it.
    order = np.argsort(-lam_in_order)
    lam_in_order = lam_in_order[order]
    f_k = f_k[order]

    print("\nReal measured f(lambda) curve (medium probe, globally stitched via MBAR):")
    for lam, f in zip(lam_in_order, f_k):
        print(f"  lambda={lam:.6f}  f={f:.3f} kJ/mol")

    steps = np.abs(np.diff(f_k))
    print("\nPer-step |Delta f| (kJ/mol):", steps.tolist())

    new_lambdas = [float(lam_in_order[0])]
    flagged_any = False
    for i, step in enumerate(steps):
        other_steps = np.concatenate([steps[:i], steps[i + 1:]])
        target = float(np.median(other_steps)) if len(other_steps) else float(step)
        if target > 0 and step > OUTLIER_FACTOR * target:
            flagged_any = True
            n_sub = int(np.ceil(step / target))
            print(
                f"  step {i}: lambda {lam_in_order[i]:.6f} -> {lam_in_order[i + 1]:.6f}, "
                f"|Delta f|={step:.2f} kJ/mol is {step / target:.1f}x the median of the "
                f"rest ({target:.2f} kJ/mol) -> splitting into {n_sub} sub-steps"
            )
            lam_start, lam_end = float(lam_in_order[i]), float(lam_in_order[i + 1])
            for j in range(1, n_sub):
                frac = j / n_sub
                new_lambdas.append(lam_start + frac * (lam_end - lam_start))
        new_lambdas.append(float(lam_in_order[i + 1]))

    if not flagged_any:
        print(
            f"\nNo outlier steps found (nothing > {OUTLIER_FACTOR}x the median of the "
            "rest) -- the original schedule does not need a tail fix by this criterion. "
            "Not writing anything."
        )
        return 0

    new_lambdas = np.array(new_lambdas, dtype=float)
    n_new = len(new_lambdas)

    lam_asc_order = np.argsort(lam_in_order)
    f_at_new = np.interp(new_lambdas, lam_in_order[lam_asc_order], f_k[lam_asc_order])
    new_window_ranges = partition_windows_by_delta_f_budget(f_at_new, MAX_WINDOW_SPAN_KJ, overlap=OVERLAP)

    print(f"\nFinal schedule: {n_new} states (was {n_states})")
    print(f"New lambda values: {new_lambdas.tolist()}")
    print(f"New window_ranges: {new_window_ranges}")
    print(f"New window sizes: {[e - s for s, e in new_window_ranges]}")

    backup_path = preopt2_file + ".buggy_12window.bak"
    shutil.copy(preopt2_file, backup_path)
    print(f"\nBacked up the current (buggy 12-window) preopt file to {backup_path}")

    result = {
        "lambdas_var": new_lambdas.tolist(),
        "window_ranges": [list(r) for r in new_window_ranges],
        "n_states": n_new,
        "provenance": {
            "source": "repair_tail_gap_local_only",
            "based_on_measured_f_curve": True,
            "max_window_span_kJ_mol": MAX_WINDOW_SPAN_KJ,
            "outlier_factor": OUTLIER_FACTOR,
            "original_n_states": n_states,
            "original_window_ranges": [list(r) for r in window_ranges],
        },
    }
    with open(preopt2_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote repaired schedule to {preopt2_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
