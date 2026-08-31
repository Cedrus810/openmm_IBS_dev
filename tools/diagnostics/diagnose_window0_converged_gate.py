#!/usr/bin/env python3
"""CPU-only (no GPU) direct re-check of window 0's learning candidate gate.

Loads the real, already-saved tmbar_history from ibs_state_vdw_window_0.json
and re-runs the CURRENT solve_stage_integrated on it, printing each
sub-condition of the `converged` boolean individually -- instead of trusting
the terse summary line, this shows exactly which criterion (if any) is still
failing and by how much, against the CURRENT code on disk.

Usage: python tools/diagnostics/diagnose_window0_converged_gate.py [output_dir]
"""

# 默认运行目录：统一由 tools/_run_dir.py 解析（ABFE_OUTPUT_DIR -> abfe_config.json
# 的 "output" -> ./output）。2026-08-31 前这里硬编码 output_lrc_fix，那是
# Atenolol-rank11 的验收基线目录，不在本工程区分支里。显式传参永远优先。
import sys as _abfe_rd_sys
from pathlib import Path as _AbfeRdPath

_ABFE_TOOLS_ROOT = _AbfeRdPath(__file__).resolve().parents[1]
if str(_ABFE_TOOLS_ROOT) not in _abfe_rd_sys.path:
    _abfe_rd_sys.path.insert(0, str(_ABFE_TOOLS_ROOT))
from _run_dir import DEFAULT_RUN_DIR  # noqa: E402


# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))

import datetime
import json
import os
import sys

import numpy as np

import ibs_engine
from ibs_engine import solve_stage_integrated, _meets_minimum_with_roundoff, _meets_maximum_with_roundoff


def main(argv=None):
    print(f"[diagnostic run started at {datetime.datetime.now().isoformat()}]")
    print(f"[ibs_engine.py loaded from: {ibs_engine.__file__}]")
    print(f"[ibs_engine.py mtime:       {datetime.datetime.fromtimestamp(os.path.getmtime(ibs_engine.__file__)).isoformat()}]")
    print(f"[has min_frames_per_window fix: {'< min_frames_per_window' in open(ibs_engine.__file__).read()}]")
    output_dir = (argv or sys.argv[1:] or [DEFAULT_RUN_DIR])[0]
    ibs_state_path = f"{output_dir}/checkpoints/ibs_state_vdw_window_0.json"
    with open(ibs_state_path) as f:
        ibs_state = json.load(f)
    tmbar_history_raw = ibs_state.get("tmbar_history") or []
    if not tmbar_history_raw:
        print(f"ERROR: no tmbar_history in {ibs_state_path}", file=sys.stderr)
        return 2
    print(f"n_tmbar_entries = {len(tmbar_history_raw)}")

    window_data = []
    for entry in tmbar_history_raw:
        window_data.append({
            "u_kn": np.asarray(entry["u_kn"], dtype=np.float64),
            "bias_energies": np.asarray(entry["bias_energies"], dtype=np.float64),
            "base_energies": np.asarray(entry["base_energies"], dtype=np.float64),
            "lambda_indices": [int(x) for x in entry["lambda_indices"]],
            "sampled_distribution_row": int(entry.get("sampled_distribution_row", 0)),
        })

    kt = 0.008314462618 * 300.0
    # Same candidate thresholds used by run_all_windows/update_weights defaults.
    candidate_min_ess_ratio = 0.05
    candidate_min_absolute_ess = 1.0
    candidate_min_decorrelated_samples = 3
    candidate_max_uncertainty_kJ_mol = 5.0

    res = solve_stage_integrated(
        window_data,
        kt,
        stage_name="window0_gate_diagnosis",
        final_min_ess_ratio=candidate_min_ess_ratio,
        final_min_absolute_ess=candidate_min_absolute_ess,
        final_min_decorrelated_samples=candidate_min_decorrelated_samples,
        final_max_uncertainty_kJ_mol=candidate_max_uncertainty_kJ_mol,
        min_frames_per_window=3,  # matches _solve_tmbar_and_recenter's actual candidate-gate call
    )
    if "error" in res:
        print(f"ERROR: solve_stage_integrated failed: {res['error']}", file=sys.stderr)
        return 2

    n_valid_windows = sum(1 for w in window_data if w.get("u_kn") is not None and w["u_kn"].size > 0)
    n_local_results = len(res.get("window_overlap_diagnostics") or [])
    print(f"\nlen(valid_windows)  = {n_valid_windows}")
    print(f"len(local_results)  = {n_local_results}  (via len(window_overlap_diagnostics))")
    print(f"len(local_results) == len(valid_windows) -> {n_local_results == n_valid_windows}")
    if n_local_results != n_valid_windows:
        print(f"  ^ {n_valid_windows - n_local_results} minibatch(es) were dropped inside the per-window loop "
              "-- look for '⚠️' warning lines printed above this point for which index and why.")

    print(f"\nraw res['converged']       = {res.get('converged')}")
    print(f"min_overlap                = {res.get('min_overlap')!r}  (threshold {candidate_min_ess_ratio})")
    print(f"min_absolute_ess           = {res.get('min_absolute_ess')!r}  (threshold {candidate_min_absolute_ess})")
    print(f"min_decorrelated_samples   = {res.get('min_decorrelated_samples')!r}  (threshold {candidate_min_decorrelated_samples})")
    print(f"max_endpoint_uncertainty   = {res.get('max_endpoint_uncertainty_kJ_mol')!r}  (threshold {candidate_max_uncertainty_kJ_mol})")

    print("\n--- re-evaluating each sub-condition of `converged` by hand, against the CURRENT code ---")
    min_overlap = res.get("min_overlap")
    min_absolute_ess = res.get("min_absolute_ess")
    min_decorrelated_samples = res.get("min_decorrelated_samples")
    max_endpoint_uncertainty_kJ_mol = res.get("max_endpoint_uncertainty_kJ_mol")

    c1 = min_overlap is not None
    c2 = c1 and _meets_minimum_with_roundoff(min_overlap, candidate_min_ess_ratio)
    c3 = min_absolute_ess is not None
    c4 = c3 and _meets_minimum_with_roundoff(min_absolute_ess, float(candidate_min_absolute_ess))
    c5 = min_decorrelated_samples >= int(candidate_min_decorrelated_samples)
    c6 = max_endpoint_uncertainty_kJ_mol is not None
    c7 = c6 and np.isfinite(max_endpoint_uncertainty_kJ_mol)
    c8 = c7 and _meets_maximum_with_roundoff(max_endpoint_uncertainty_kJ_mol, float(candidate_max_uncertainty_kJ_mol))

    print(f"min_overlap is not None                                    -> {c1}")
    print(f"_meets_minimum_with_roundoff(min_overlap, {candidate_min_ess_ratio})              -> {c2}")
    print(f"min_absolute_ess is not None                                -> {c3}")
    print(f"_meets_minimum_with_roundoff(min_absolute_ess, {candidate_min_absolute_ess})       -> {c4}")
    print(f"min_decorrelated_samples >= {candidate_min_decorrelated_samples}                            -> {c5}")
    print(f"max_endpoint_uncertainty_kJ_mol is not None                -> {c6}")
    print(f"np.isfinite(max_endpoint_uncertainty_kJ_mol)                -> {c7}")
    print(f"_meets_maximum_with_roundoff(max_endpoint_uncertainty, {candidate_max_uncertainty_kJ_mol}) -> {c8}")

    all_ok = bool(c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8)
    print(f"\nhand-recomputed converged (missing only len(local_results)==len(valid_windows) check) = {all_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
