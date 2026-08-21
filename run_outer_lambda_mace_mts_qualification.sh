#!/usr/bin/env bash
set -euo pipefail

# EXP-009：冻结 coefficient=0.09 的 BAOAB-rRESPA N=1/2/4 矩阵。
# OpenMM 外层步长=N*0.5 fs；base group 0 每外步 N 次，MACE group 31 每外步一次。

PROJECT_DIR="${PROJECT_DIR:-/home/ruigengji/ABFE_IBS/Atenolol-rank11}"
PYTHON_BIN="${PYTHON_BIN:-/home/ruigengji/mambaforge/envs/openmm_dev/bin/python}"
MODEL_NAME="${MODEL_NAME:-mace-off24-medium}"
MODEL_PATH="${MODEL_PATH:-auto}"
DEVICE="${DEVICE:-cuda}"
PLATFORM="${PLATFORM:-CUDA}"
# 正式矩阵默认每臂 50 ps；可显式覆盖为短开发运行。
# 500 inner steps = 0.25 ps，三个 ratio 均能整除，并提供 200 个诊断样本/臂。
N_INNER_STEPS="${N_INNER_STEPS:-100000}"
REPORT_INTERVAL_INNER_STEPS="${REPORT_INTERVAL_INNER_STEPS:-500}"
INNER_TIMESTEP_FS="${INNER_TIMESTEP_FS:-0.5}"
LAMBDA_VALUE="${LAMBDA_VALUE:-0.5}"
FRAME_SPEC="${FRAME_SPEC:-last}"
MINIMUM_N4_NS_PER_DAY="${MINIMUM_N4_NS_PER_DAY:-1.0}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/output/outer_lambda_mace_mts/exp009}"

SELECTION_META="${SELECTION_META:-${PROJECT_DIR}/output/dexp_experiment/fit_label_cache_meta.json}"
# WP-0 从现有 IBS 基线选择困难窗口；MTS 必须复用 EXP-007 已资格化的坐标协议。
WP0_TRAJECTORY="${WP0_TRAJECTORY:-${PROJECT_DIR}/output_lrc_fix/pre_equilibration.dcd}"
WP0_REBALANCE_TRAJECTORY="${WP0_REBALANCE_TRAJECTORY:-${PROJECT_DIR}/output_lrc_fix/rebalance_traj.dcd}"
WP0_TOPOLOGY="${WP0_TOPOLOGY:-${PROJECT_DIR}/output_lrc_fix/topology.cif}"
TRAJECTORY="${TRAJECTORY:-${PROJECT_DIR}/output/pre_equilibration.dcd}"
TOPOLOGY="${TOPOLOGY:-${PROJECT_DIR}/output/topology.cif}"
SYSTEM_XML="${SYSTEM_XML:-${PROJECT_DIR}/output/system_native.xml}"
FINAL_RESULTS="${FINAL_RESULTS:-${PROJECT_DIR}/output_lrc_fix/final_results.json}"
MODULE="${PROJECT_DIR}/outer_lambda_neural_basis.py"
CONFIG="${RUN_DIR}/outer_lambda_existing_model.json"

if [[ -e "${RUN_DIR}/prepare_report.json" || -e "${RUN_DIR}/mts_qualification.json" ]]; then
  echo "拒绝覆盖已有 EXP-009 运行目录: ${RUN_DIR}" >&2
  echo "请设置新的 RUN_DIR；已有运行证据必须保留。" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" "${MODULE}" wp0-select \
  --final-results "${FINAL_RESULTS}" \
  --topology "${WP0_TOPOLOGY}" \
  --trajectory "${WP0_TRAJECTORY}" \
  --trajectory "${WP0_REBALANCE_TRAJECTORY}" \
  --torsion-indices 4586,4584,4591,4592 \
  --slow-variable-name atenolol_C4_N2_C9_C10 \
  --output "${RUN_DIR}/wp0_selection.json"

"${PYTHON_BIN}" "${MODULE}" prepare-existing \
  --selection-meta "${SELECTION_META}" \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${RUN_DIR}" \
  --coefficient 0.09 \
  --energy-offset-kj-mol -343.0 \
  --max-abs-basis-energy-kj-mol 1000 \
  --max-abs-path-energy-kj-mol 20 \
  --max-force-norm-kj-mol-nm 3000 \
  --min-pair-distance-nm 0.07 \
  --max-pair-distance-nm 2.5 \
  --max-radius-of-gyration-nm 0.85 \
  --output "${RUN_DIR}/prepare_report.json"

"${PYTHON_BIN}" "${MODULE}" mace-mts-qualification \
  --config "${CONFIG}" \
  --system-xml "${SYSTEM_XML}" \
  --trajectory "${TRAJECTORY}" \
  --topology "${TOPOLOGY}" \
  --selection-meta "${SELECTION_META}" \
  --frame "${FRAME_SPEC}" \
  --torsion-indices 4586,4584,4591,4592 \
  --lambda "${LAMBDA_VALUE}" \
  --inner-steps "${N_INNER_STEPS}" \
  --report-interval-inner-steps "${REPORT_INTERVAL_INNER_STEPS}" \
  --inner-timestep-fs "${INNER_TIMESTEP_FS}" \
  --device "${DEVICE}" \
  --platform "${PLATFORM}" \
  --minimum-n4-ns-per-day "${MINIMUM_N4_NS_PER_DAY}" \
  --output "${RUN_DIR}/mts_qualification.json"

echo "EXP-009 MTS qualification 通过: ${RUN_DIR}/mts_qualification.json"
