#!/usr/bin/env bash
set -euo pipefail

# 独立节点入口：不调用或修改 runabfe.py / abfe_core.py / ibs_engine.py。
# 默认 MACE smoke；也可通过 MODEL_NAME/MODEL_PATH 切换 conservative ORB。

PROJECT_DIR="${PROJECT_DIR:-/home/ruigengji/ABFE_IBS/Atenolol-rank11}"
PYTHON_BIN="${PYTHON_BIN:-/home/ruigengji/mambaforge/envs/openmm_dev/bin/python}"
MODEL_NAME="${MODEL_NAME:-mace-off24-medium}"
MODEL_PATH="${MODEL_PATH:-auto}"
DEVICE="${DEVICE:-cuda}"
FRAME_SPEC="${FRAME_SPEC:-last}"
LAMBDA_SCHEDULE="${LAMBDA_SCHEDULE:-0,0.25,0.5,0.75,1}"
COEFFICIENT="${COEFFICIENT:-0.1}"
ENERGY_OFFSET_KJ_MOL="${ENERGY_OFFSET_KJ_MOL:-0.0}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/output/outer_lambda_existing_model/${MODEL_NAME}}"

SELECTION_META="${SELECTION_META:-${PROJECT_DIR}/output/dexp_experiment/fit_label_cache_meta.json}"
TRAJECTORY="${TRAJECTORY:-${PROJECT_DIR}/output/pre_equilibration.dcd}"
TOPOLOGY="${TOPOLOGY:-${PROJECT_DIR}/output/topology.cif}"
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
  --max-abs-basis-energy-kj-mol 5000 \
  --max-abs-path-energy-kj-mol 1000 \
  --max-force-norm-kj-mol-nm 5000 \
  --output "${RUN_DIR}/prepare_report.json"

"${PYTHON_BIN}" "${MODULE}" label-trajectory \
  --config "${CONFIG}" \
  --trajectory "${TRAJECTORY}" \
  --topology "${TOPOLOGY}" \
  --selection-meta "${SELECTION_META}" \
  --frames "${FRAME_SPEC}" \
  --model-name "${MODEL_NAME}" \
  --device "${DEVICE}" \
  --lambdas "${LAMBDA_SCHEDULE}" \
  --output "${RUN_DIR}/labels.json"

echo "完成: ${RUN_DIR}/labels.json"
