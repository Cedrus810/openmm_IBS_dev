#!/usr/bin/env bash
set -euo pipefail

# WP-4 正式 MACE NVT qualification。阈值由此前 100-step calibration 冻结；
# 只读取 production artifacts，在独立模块复制的 System 中运行。

PROJECT_DIR="${PROJECT_DIR:-/home/ruigengji/ABFE_IBS/Atenolol-rank11}"
PYTHON_BIN="${PYTHON_BIN:-/home/ruigengji/mambaforge/envs/openmm_dev/bin/python}"
MODEL_NAME="${MODEL_NAME:-mace-off24-medium}"
MODEL_PATH="${MODEL_PATH:-auto}"
DEVICE="${DEVICE:-cuda}"
PLATFORM="${PLATFORM:-CUDA}"
N_STEPS="${N_STEPS:-1000}"
REPORT_INTERVAL="${REPORT_INTERVAL:-25}"
TIMESTEP_FS="${TIMESTEP_FS:-0.5}"
LAMBDA_VALUE="${LAMBDA_VALUE:-0.5}"
COEFFICIENT="${COEFFICIENT:-0.1}"
ENERGY_OFFSET_KJ_MOL="${ENERGY_OFFSET_KJ_MOL:--343.0}"
FRAME_SPEC="${FRAME_SPEC:-last}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/output/outer_lambda_mace_qualification}"

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
  --coefficient "${COEFFICIENT}" \
  --energy-offset-kj-mol "${ENERGY_OFFSET_KJ_MOL}" \
  --max-abs-basis-energy-kj-mol 1000 \
  --max-abs-path-energy-kj-mol 20 \
  --max-force-norm-kj-mol-nm 3000 \
  --min-pair-distance-nm 0.07 \
  --max-pair-distance-nm 2.5 \
  --max-radius-of-gyration-nm 0.85 \
  --output "${RUN_DIR}/prepare_report.json"

"${PYTHON_BIN}" "${MODULE}" mace-nvt-qualification \
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
  --minimum-steps 1000 \
  --max-path-force-kj-mol-nm 250 \
  --max-energy-closure-error-kj-mol 0.1 \
  --max-integration-seconds-per-step 0.2 \
  --output "${RUN_DIR}/nvt_qualification.json"

echo "WP-4 qualification 通过: ${RUN_DIR}/nvt_qualification.json"
