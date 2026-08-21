#!/usr/bin/env python
"""Offline (no GPU) analysis of window 0's REAL sampled data from its failed
warmup attempt -- not a probe, the actual physically-sampled frames.

Background: vanishing window 0 has failed IBSWarmupConvergenceError three
times under different state-count groupings (6/4/3 states), and pilot-grid
refinement (THERMODYNAMIC_PATH_PROTOCOL_VERSION=15) did not fix it either --
it revealed the true difficulty is a sharp, non-monotonic peak around
lambda~0.96-0.97, not a smooth endpoint effect at lambda=1.0. All of this used
short PROBE data (crude by design). This script instead reads the REAL,
physically-sampled data left behind by the failed warmup itself:
IBSSampler.save_ibs_state persists `tmbar_history` even when
IBSWarmupConvergenceError is raised ("保留全部已采数据") -- ~50 real batches x
20 frames = up to ~1000 real sampled frames across window 0's states, in
exactly the format GlobalMBARAnalyzer.solve_stage_integrated expects (it's the
same TMBAR machinery used online during learning, just re-solved once more
here, offline, for inspection).

This does NOT require a new GPU run -- the data already exists on disk from
the last failed attempt. It complements (does not replace) the pilot-based
schedule redesign: it tells you where the REAL measured free-energy landscape
is steep *within window 0's existing states*, using far more real samples
than any single pilot probe point.

Usage (needs numpy/pymbar, i.e. the openmm_dev env; no OpenMM Platform/Context
touched, so no GPU needed):

    conda run -n openmm_dev python tools/diagnostics/analyze_window0_real_tmbar_data.py \
        ./output_lrc_fix/checkpoints/ibs_state_vdw_window_0.json
"""

from __future__ import annotations

# Allow direct execution from tools/* while keeping live modules at repo root.
import sys as _abfe_sys
from pathlib import Path as _AbfePath

_ABFE_REPO_ROOT = _AbfePath(__file__).resolve().parents[2]
if str(_ABFE_REPO_ROOT) not in _abfe_sys.path:
    _abfe_sys.path.insert(0, str(_ABFE_REPO_ROOT))


import json
import sys

import numpy as np


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    state_path = args[0] if args else "./output_lrc_fix/checkpoints/ibs_state_vdw_window_0.json"

    with open(state_path, "r") as f:
        state = json.load(f)

    n_states = int(state["n_states"])
    lambdas_vdw = state.get("lambdas_vdw")
    lambdas_coul = state.get("lambdas_coul")
    tmbar_history_raw = state.get("tmbar_history") or []
    if not tmbar_history_raw:
        print("ERROR: tmbar_history is empty in this state file -- nothing to analyze.", file=sys.stderr)
        return 2

    print(f"n_states={n_states}  lambdas_vdw={lambdas_vdw}  lambdas_coul={lambdas_coul}")
    print(f"tmbar_history entries: {len(tmbar_history_raw)}")

    # Reconstruct the same window_data shape IBSSampler._append_tmbar_batch_from_buffer
    # produces (u_kn as (K, N) arrays), for solve_stage_integrated.
    window_data = []
    total_frames = 0
    for entry in tmbar_history_raw:
        u_kn = np.asarray(entry["u_kn"], dtype=np.float64)
        bias = np.asarray(entry["bias_energies"], dtype=np.float64)
        base = np.asarray(entry["base_energies"], dtype=np.float64)
        window_data.append({
            "u_kn": u_kn,
            "bias_energies": bias,
            "base_energies": base,
            "lambda_indices": [int(x) for x in entry["lambda_indices"]],
            "sampled_distribution_row": int(entry.get("sampled_distribution_row", 0)),
        })
        total_frames += u_kn.shape[1]
    print(f"total raw frames across all batches: {total_frames}")

    # --- Part 1: raw adjacent-state delta-U statistics, pooled across ALL
    # real batches (not a single short probe) -- tells you directly, from
    # real sampled configurations, how big and how variable each adjacent
    # edge's energy gap actually is.
    print("\n=== Pooled raw adjacent-state delta-U (kJ/mol), all real frames ===")
    all_u = np.concatenate([w["u_kn"] for w in window_data], axis=1)  # (K, total_frames)
    for k in range(n_states - 1):
        du = all_u[k + 1] - all_u[k]
        print(
            f"  edge {k}->{k+1} (lambda {lambdas_vdw[k]:.4f}->{lambdas_vdw[k+1]:.4f}): "
            f"mean={np.mean(du):8.2f}  std={np.std(du):8.2f}  "
            f"p05={np.percentile(du,5):8.2f}  p50={np.percentile(du,50):8.2f}  p95={np.percentile(du,95):8.2f}"
        )

    # --- Part 2: real MBAR-based f(lambda), pooling ALL real batches via the
    # same TMBAR stitching used online (solve_stage_integrated) -- this is
    # the "real Delta-f" measurement, not a probe proxy.
    try:
        from ibs_engine import solve_stage_integrated
    except Exception as e:
        print(f"\nERROR: could not import solve_stage_integrated ({e}); "
              "run this inside the openmm_dev env.", file=sys.stderr)
        return 2

    kt = 0.008314462618 * 300.0
    res = solve_stage_integrated(window_data, kt, stage_name="vdw_window0_offline_analysis")
    print("\n=== Real MBAR-based f(lambda) from all real sampled frames ===")
    if res.get("error"):
        print(f"  solve_stage_integrated error: {res['error']}")
    else:
        f_k = res["f_k"]
        lambdas_sorted = res["lambdas"]
        print(f"  lambdas (state index order): {lambdas_sorted}")
        print(f"  f_k (kJ/mol, mean-centered by solve_stage_integrated's own convention): {[round(x,3) for x in f_k]}")
        print(f"  converged={res.get('converged')}  min_overlap={res.get('min_overlap')}  "
              f"min_absolute_ess={res.get('min_absolute_ess')}  "
              f"max_endpoint_uncertainty_kJ_mol={res.get('max_endpoint_uncertainty_kJ_mol')}")
        for edge_idx in range(len(f_k) - 1):
            print(f"  real Delta_f edge {edge_idx}->{edge_idx+1}: {f_k[edge_idx+1]-f_k[edge_idx]:.3f} kJ/mol")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
