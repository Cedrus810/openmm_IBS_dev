
# 默认运行目录：统一由 tools/_run_dir.py 解析（ABFE_OUTPUT_DIR -> abfe_config.json
# 的 "output" -> ./output）。2026-08-31 前这里硬编码 output_lrc_fix，那是
# Atenolol-rank11 的验收基线目录，不在本工程区分支里。显式传参永远优先。
import sys as _abfe_rd_sys
from pathlib import Path as _AbfeRdPath

_ABFE_TOOLS_ROOT = _AbfeRdPath(__file__).resolve().parents[1]
if str(_ABFE_TOOLS_ROOT) not in _abfe_rd_sys.path:
    _abfe_rd_sys.path.insert(0, str(_ABFE_TOOLS_ROOT))
from _run_dir import DEFAULT_RUN_DIR  # noqa: E402

import argparse
import json
import os

import numpy as np


parser = argparse.ArgumentParser(description="Inspect per-window ESS diagnostics.")
parser.add_argument("output_dir", nargs="?", default=DEFAULT_RUN_DIR)
args = parser.parse_args()
root = os.path.abspath(args.output_dir)
kt = 8.31446261815324e-3 * 300.0

for window_index in range(6):
    energy_path = os.path.join(
        root, "vanishing", f"dual_window_{window_index}_vdw_energies.npy"
    )
    bias_path = os.path.join(
        root, "vanishing", f"dual_window_{window_index}_vdw_bias.npy"
    )
    checkpoint_path = os.path.join(
        root, "checkpoints", f"ibs_state_vdw_window_{window_index}.json"
    )
    if not os.path.exists(energy_path):
        continue

    energies = np.load(energy_path).astype(float)
    bias = np.load(bias_path).astype(float)
    with open(checkpoint_path, encoding="utf-8") as handle:
        f_k = np.asarray(json.load(handle)["f_k"], dtype=float)

    log_importance = -(energies - bias[None, :]) / kt
    ess_ratio = []
    top_one_percent_weight = []
    for row in log_importance:
        weights = np.exp(row - np.max(row))
        weights /= weights.sum()
        ess_ratio.append(1.0 / (weights @ weights) / len(weights))
        n_top = max(1, len(weights) // 100)
        top_one_percent_weight.append(np.sort(weights)[-n_top:].sum())

    logits = -(energies - f_k[:, None]) / kt
    max_logits = np.max(logits, axis=0)
    expected_ibs_bias = -kt * (
        max_logits
        + np.log(np.exp(logits - max_logits[None, :]).sum(axis=0))
    )
    extra_sampling_bias = bias - expected_ibs_bias
    probabilities = np.exp(logits - max_logits[None, :])
    probabilities /= probabilities.sum(axis=0)
    occupancy = probabilities.mean(axis=1)

    print(
        window_index,
        "shape=", energies.shape,
        "raw_ess_ratio=", np.round(ess_ratio, 5).tolist(),
        "top1pct_weight=", np.round(top_one_percent_weight, 3).tolist(),
        "occupancy=", np.round(occupancy, 3).tolist(),
        "extra_bias_std=", round(float(extra_sampling_bias.std()), 3),
        "extra_bias_p01_p50_p99=",
        np.round(np.quantile(extra_sampling_bias, [0.01, 0.5, 0.99]), 3).tolist(),
    )
