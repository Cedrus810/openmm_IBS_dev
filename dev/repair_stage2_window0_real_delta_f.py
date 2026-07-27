#!/usr/bin/env python
"""One-off, LOCAL-ONLY repair: replace vanishing window 0's 3 lambda points
with 7, placed using REAL measured Delta-f (not the pilot probe's crude
variance proxy), without re-running the ~20-minute pilot probe.

Background
----------
Vanishing window 0 (the fully-coupled vdW endpoint) has failed
IBSWarmupConvergenceError four times: three different state-count groupings
(6/4/3 states, THERMODYNAMIC_PATH_PROTOCOL_VERSION 13/14) and one adaptive
pilot-grid refinement (v15) -- see VANISHING_WINDOW0_HANDOFF.md for the full
history. v15's refinement DID trigger for real (26 pilot points, 2 rounds) and
revealed a genuine, sharp non-monotonic metric_g peak around lambda~0.96-0.97
-- but that peak sits OUTSIDE window 0's own span (window 0 only covers
lambda=1.0 -> ~0.9848) and window 0 still failed on its own terms.

The failed run's own IBSSampler.save_ibs_state left behind ~1000 REAL sampled
frames (window 0's `tmbar_history`). Re-solving those with
GlobalMBARAnalyzer.solve_stage_integrated (see analyze_window0_real_tmbar_data.py)
gives REAL measured f(lambda) at window 0's 3 states:

    lambda:  1.0       0.990616   0.984823
    f_k:     96.354    71.030     55.358   (kJ/mol, mean-centered)
    real Delta_f edge0->1: -25.324 kJ/mol (~10.2 kT)
    real Delta_f edge1->2: -15.672 kJ/mol (~6.3 kT)

Both edges' implied dF/dlambda (~2694 and ~2702 kJ/mol per unit lambda) are
nearly identical -- this specific span is close to LINEAR, not pathological.
This is simply "too few states for a steep but well-behaved slope",
straightforwardly fixable by adding real-data-placed intermediate states, with
much higher confidence than probe-based guessing (which cannot resolve a
sharp, non-monotonic landscape, but has no trouble at all with a linear one).

This script follows the exact pattern already validated this session in
repair_stage2_window0_regroup.py (LOCAL-ONLY regroup, no pilot rerun): it
touches ONLY `lambdas_var`, `window_ranges`, `n_states`, and
`path_protocol_version` in the cached preopt file; `protocol_key`/
`provenance`/`path_diagnostics` are left untouched, exactly as that script
does, for the same reason (the narrow preopt fingerprint has no code hash, and
nothing about the physical inputs changed -- only window 0's own lambda
placement, driven by real data this script supplies). Resuming afterward still
needs `ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT=1` (same as after
repair_stage2_window0_regroup.py) since `protocol_key` itself isn't
recomputed here.

Concretely:
  1. Loads the real tmbar_history from the failed run's ibs_state file and
     re-solves it via solve_stage_integrated to get window 0's 3 real
     (lambda, f_k) points (same computation as analyze_window0_real_tmbar_data.py).
  2. Uses the existing redistribute_lambda_by_delta_f to place N new lambda
     points (default 7) evenly spaced in REAL cumulative |Delta_f| between
     lambda=1.0 and window 0's existing right edge.
  3. Splices these into the cached preopt_dual_vanishing.json's lambdas_var
     in place of window 0's original 3 states; every state from the old
     window 0's right edge onward is untouched (same lambda values, just
     shifted to higher indices).
  4. Recomputes window_ranges via vanishing_subdomain_ranges_from_lambdas
     using VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS (must already be set to
     N-1 in abfe_preoptimizer.py -- this script does NOT hardcode that number
     twice) so the very next cache-validation check in abfe_pipeline.py
     matches what it independently recomputes.

Does NOT touch any other window's lambda values, and does NOT run any new MD.

Usage (needs numpy/pymbar, i.e. the openmm_dev env; no GPU touched):

    python repair_stage2_window0_real_delta_f.py ./output_lrc_fix --n-states 7

Then resume with the fingerprint bypass (same as after
repair_stage2_window0_regroup.py):

    ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT=1 python runabfe.py \
        --config abfe_config.json --ligand MOL --resume
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", nargs="?", default="./output_lrc_fix")
    parser.add_argument(
        "--n-states", type=int, default=7,
        help="How many states window 0 should become (default 7 -> 6 real-Delta_f-"
             "equalized steps covering the same lambda=1.0 -> existing-window0-edge "
             "span). Must match VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS+1 in "
             "abfe_preoptimizer.py, or the next resume's cache-validation check "
             "will reject this repair and silently re-run the full pilot instead.",
    )
    parser.add_argument(
        "--ibs-state-file", default=None,
        help="Path to the failed window 0's ibs_state json (default: "
             "<output_dir>/checkpoints/ibs_state_vdw_window_0.json).",
    )
    args = parser.parse_args(argv)

    ibs_state_path = args.ibs_state_file or os.path.join(
        args.output_dir, "checkpoints", "ibs_state_vdw_window_0.json"
    )
    preopt_path = os.path.join(args.output_dir, "checkpoints", "preopt_dual_vanishing.json")

    with open(ibs_state_path, "r") as f:
        ibs_state = json.load(f)
    tmbar_history_raw = ibs_state.get("tmbar_history") or []
    if not tmbar_history_raw:
        print(f"ERROR: no tmbar_history in {ibs_state_path}", file=sys.stderr)
        return 2
    window0_lambdas_vdw = ibs_state["lambdas_vdw"]

    window_data = []
    for entry in tmbar_history_raw:
        window_data.append({
            "u_kn": np.asarray(entry["u_kn"], dtype=np.float64),
            "bias_energies": np.asarray(entry["bias_energies"], dtype=np.float64),
            "base_energies": np.asarray(entry["base_energies"], dtype=np.float64),
            "lambda_indices": [int(x) for x in entry["lambda_indices"]],
            "sampled_distribution_row": int(entry.get("sampled_distribution_row", 0)),
        })

    from ibs_engine import solve_stage_integrated
    kt = 0.008314462618 * 300.0
    res = solve_stage_integrated(window_data, kt, stage_name="vdw_window0_repair")
    if res.get("error"):
        print(f"ERROR: solve_stage_integrated failed: {res['error']}", file=sys.stderr)
        return 2

    lambdas_sorted = res["lambdas"]
    f_k = np.asarray(res["f_k"], dtype=float)
    lam_in_order = np.asarray([window0_lambdas_vdw[i] for i in lambdas_sorted], dtype=float)
    print(f"Real measured window 0 f(lambda): lambda={lam_in_order.tolist()}, f_k={f_k.tolist()}")
    print(f"Real Delta_f per edge (kJ/mol): {np.diff(f_k).tolist()}")

    from abfe_preoptimizer import (
        redistribute_lambda_by_delta_f,
        vanishing_subdomain_ranges_from_lambdas,
        VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
        THERMODYNAMIC_PATH_PROTOCOL_VERSION,
    )

    n_new_window0_states = int(args.n_states)
    if n_new_window0_states - 1 != VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS:
        print(
            f"ERROR: --n-states={n_new_window0_states} implies "
            f"{n_new_window0_states - 1} intervals, but abfe_preoptimizer.py's "
            f"VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS is "
            f"{VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS}. Update the constant (and "
            "bump THERMODYNAMIC_PATH_PROTOCOL_VERSION) to match before running this "
            "repair, or the window_ranges this script computes won't match what "
            "abfe_pipeline.py independently recomputes on the next resume.",
            file=sys.stderr,
        )
        return 2

    new_window0_lambdas = redistribute_lambda_by_delta_f(
        lam_in_order, f_k, n_new_window0_states
    )
    print(f"New window 0 lambdas ({n_new_window0_states} states): {new_window0_lambdas.tolist()}")

    with open(preopt_path, "r") as f:
        cached = json.load(f)
    old_lambdas_var = cached["lambdas_var"]

    if abs(old_lambdas_var[0] - lam_in_order[0]) > 1e-6:
        print(
            f"ERROR: preopt_dual_vanishing.json's first lambda ({old_lambdas_var[0]}) "
            f"does not match the failed run's window 0 first lambda ({lam_in_order[0]}) "
            "-- this preopt file was not the one window 0 actually ran under; refusing "
            "to splice mismatched data.",
            file=sys.stderr,
        )
        return 2
    right_edge_lambda = lam_in_order[-1]
    tail_start_idx = next(
        (i for i, lam in enumerate(old_lambdas_var) if abs(lam - right_edge_lambda) < 1e-6),
        None,
    )
    if tail_start_idx is None:
        print(
            f"ERROR: could not find window 0's right edge lambda ({right_edge_lambda}) "
            "in preopt_dual_vanishing.json's lambdas_var -- refusing to splice.",
            file=sys.stderr,
        )
        return 2

    new_lambdas_var = list(new_window0_lambdas) + list(old_lambdas_var[tail_start_idx + 1:])
    n_new_total = len(new_lambdas_var)
    print(f"Spliced lambdas_var: {n_new_total} states total (was {len(old_lambdas_var)})")

    new_window_ranges = vanishing_subdomain_ranges_from_lambdas(
        new_lambdas_var,
        first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
    )
    print(f"New window_ranges ({len(new_window_ranges)} windows): {new_window_ranges}")
    print(f"Window 0: {new_lambdas_var[0]:.4f} -> {new_lambdas_var[new_window_ranges[0][1]-1]:.4f} "
          f"({new_window_ranges[0][1] - new_window_ranges[0][0]} states)")

    backup_path = preopt_path + ".before_window0_real_delta_f.bak"
    shutil.copy(preopt_path, backup_path)
    print(f"\nBacked up current preopt file to {backup_path}")

    # Same minimal-touch convention as repair_stage2_window0_regroup.py:
    # protocol_key/provenance/path_diagnostics left untouched (no code hash in
    # the narrow fingerprint; nothing about the physical inputs changed).
    cached["lambdas_var"] = [float(x) for x in new_lambdas_var]
    cached["window_ranges"] = [list(r) for r in new_window_ranges]
    cached["n_states"] = n_new_total
    cached["path_protocol_version"] = THERMODYNAMIC_PATH_PROTOCOL_VERSION

    with open(preopt_path, "w") as f:
        json.dump(cached, f, indent=2)
    print(f"Wrote repaired schedule to {preopt_path}")
    print(
        "\nResume with: ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT=1 python runabfe.py "
        "--config abfe_config.json --ligand MOL --resume"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
