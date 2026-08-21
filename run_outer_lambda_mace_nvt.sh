#!/usr/bin/env bash
set -euo pipefail

# WP-4 每步 MACE Force 短 NVT。只读取 production System XML，并在模块内复制；
# 不调用或修改生产入口。

PROJECT_DIR="${PROJECT_DIR:-/home/ruigengji/ABFE_IBS/Atenolol-rank11}"
PYTHON_BIN="${PYTHON_BIN:-/home/ruigengji/mambaforge/envs/openmm_dev/bin/python}"
MODEL_NAME="${MODEL_NAME:-mace-off24-medium}"
MODEL_PATH="${MODEL_PATH:-auto}"
DEVICE="${DEVICE:-cuda}"
PLATFORM="${PLATFORM:-CUDA}"
N_STEPS="${N_STEPS:-1}"
REPORT_INTERVAL="${REPORT_INTERVAL:-1}"
TIMESTEP_FS="${TIMESTEP_FS:-0.5}"
LAMBDA_VALUE="${LAMBDA_VALUE:-0.5}"
FRAME_SPEC="${FRAME_SPEC:-last}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/output/outer_lambda_mace_nvt}"

SELECTION_META="${SELECTION_META:-${PROJECT_DIR}/output/dexp_experiment/fit_label_cache_meta.json}"
TRAJECTORY="${TRAJECTORY:-${PROJECT_DIR}/output/pre_equilibration.dcd}"
TOPOLOGY="${TOPOLOGY:-${PROJECT_DIR}/output/topology.cif}"
SYSTEM_XML="${SYSTEM_XML:-${PROJECT_DIR}/output/system_native.xml}"
MODULE="${PROJECT_DIR}/outer_lambda_neural_basis.py"
CONFIG="${RUN_DIR}/outer_lambda_existing_model.json"

mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" "${MODULE}" prepare-existing \
  --selection-meta "${SELECTION_META}" \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${RUN_DIR}" \
  --coefficient 0.1 \
  --energy-offset-kj-mol 0.0 \
  --max-abs-basis-energy-kj-mol 5000 \
  --max-abs-path-energy-kj-mol 1000 \
  --max-force-norm-kj-mol-nm 5000 \
  --output "${RUN_DIR}/prepare_report.json"

"${PYTHON_BIN}" "${MODULE}" mace-nvt-smoke \
  --config "${CONFIG}" \
  --system-xml "${SYSTEM_XML}" \
  --trajectory "${TRAJECTORY}" \
  --topology "${TOPOLOGY}" \
  --selection-meta "${SELECTION_META}" \
  --frame "${FRAME_SPEC}" \
  --lambda "${LAMBDA_VALUE}" \
  --steps "${N_STEPS}" \
  --report-interval "${REPORT_INTERVAL}" \
  --timestep-fs "${TIMESTEP_FS}" \
  --device "${DEVICE}" \
  --platform "${PLATFORM}" \
  --output "${RUN_DIR}/nvt_smoke.json"

echo "完成: ${RUN_DIR}/nvt_smoke.json"
