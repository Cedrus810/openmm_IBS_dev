#!/usr/bin/env bash
set -euo pipefail

# WP-0B：不修改 production，重建困难 window 0 的独立 scratch ensemble，
# 生成坐标后自动排序 ligand rotatable torsions、pocket sidechain chi1，
# 并独立报告 ligand 第一水合壳层 coordination（不同量纲，不与 torsion 混排）。

PROJECT_DIR="${PROJECT_DIR:-/home/ruigengji/ABFE_IBS/Atenolol-rank11}"
PYTHON_BIN="${PYTHON_BIN:-/home/ruigengji/mambaforge/envs/openmm_dev/bin/python}"
BASELINE_ROOT="${BASELINE_ROOT:-${PROJECT_DIR}/output_lrc_fix}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/output/outer_lambda_slow_variable_screen/hard_window0_run1}"
PLATFORM="${PLATFORM:-CUDA}"
# 正式首轮：50 ps scratch burn-in + 500 ps screening，1 ps/帧。
BURNIN_STEPS="${BURNIN_STEPS:-25000}"
SAMPLING_STEPS="${SAMPLING_STEPS:-250000}"
REPORT_INTERVAL_STEPS="${REPORT_INTERVAL_STEPS:-500}"
SEED="${SEED:-20260731}"
MODULE="${PROJECT_DIR}/outer_lambda_neural_basis.py"
SAMPLE_DIR="${RUN_DIR}/scratch_sample"

if [[ -e "${RUN_DIR}/sample_report.json" || -e "${RUN_DIR}/candidate_screen.json" ]]; then
  echo "拒绝覆盖已有慢变量筛选运行: ${RUN_DIR}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" "${MODULE}" sample-hard-window-scratch \
  --baseline-root "${BASELINE_ROOT}" \
  --output-dir "${SAMPLE_DIR}" \
  --window-index 0 \
  --initial-trajectory "${BASELINE_ROOT}/rebalance_traj.dcd" \
  --burnin-steps "${BURNIN_STEPS}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --report-interval-steps "${REPORT_INTERVAL_STEPS}" \
  --platform "${PLATFORM}" \
  --seed "${SEED}" \
  --output "${RUN_DIR}/sample_report.json"

"${PYTHON_BIN}" "${MODULE}" screen-slow-variables \
  --trajectory "${SAMPLE_DIR}/hard_window_screening.dcd" \
  --topology "${BASELINE_ROOT}/topology.cif" \
  --system-xml "${BASELINE_ROOT}/system_native.xml" \
  --ligand-indices "${BASELINE_ROOT}/ligand_indices.json" \
  --frames all \
  --pocket-cutoff-nm 0.6 \
  --output "${RUN_DIR}/candidate_screen.json"

echo "困难窗口慢变量候选筛选完成: ${RUN_DIR}/candidate_screen.json"
