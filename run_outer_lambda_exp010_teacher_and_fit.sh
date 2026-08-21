#!/usr/bin/env bash
set -euo pipefail

# EXP-010：完整 MACE 只做离线教师；训练/验证按三条独立 run 分组。
# 默认每 5 ps 取一帧：每 run 100 帧，共 300 个教师标签。

PROJECT_DIR="${PROJECT_DIR:-/home/ruigengji/ABFE_IBS/Atenolol-rank11}"
PYTHON_BIN="${PYTHON_BIN:-/home/ruigengji/mambaforge/envs/openmm_dev/bin/python}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/output/outer_lambda_exp010/run1}"
DEVICE="${DEVICE:-cuda}"
FRAME_SPEC="${FRAME_SPEC:-::5}"
MODULE="${PROJECT_DIR}/outer_lambda_neural_basis.py"
MODEL_NAME="${MODEL_NAME:-mace-off24-medium}"
MODEL_PATH="${MODEL_PATH:-auto}"
CONFIG="${CONFIG:-${RUN_DIR}/outer_lambda_existing_model.json}"
MANIFEST="${MANIFEST:-${PROJECT_DIR}/output/outer_lambda_slow_variable_screen/slow_variable_manifest.json}"
TOPOLOGY="${TOPOLOGY:-${PROJECT_DIR}/output_lrc_fix/topology.cif}"
SOURCE_SELECTION_META="${SOURCE_SELECTION_META:-${PROJECT_DIR}/output/dexp_experiment/fit_label_cache_meta.json}"
SELECTION_META="${SELECTION_META:-${RUN_DIR}/protein_only_selection_meta.json}"
SCREEN_ROOT="${SCREEN_ROOT:-${PROJECT_DIR}/output/outer_lambda_slow_variable_screen}"
DATASET="${RUN_DIR}/teacher_dataset.json"

if [[ -e "${DATASET}" ]]; then
  echo "拒绝覆盖已有 EXP-010 教师数据集: ${DATASET}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" "${MODULE}" exp010-prepare-selection \
  --selection-meta "${SOURCE_SELECTION_META}" \
  --topology "${TOPOLOGY}" \
  --output-selection-meta "${SELECTION_META}" \
  --output "${RUN_DIR}/selection_prepare_report.json"

"${PYTHON_BIN}" "${MODULE}" prepare-existing \
  --selection-meta "${SELECTION_META}" \
  --model-path "${MODEL_PATH}" \
  --model-name "${MODEL_NAME}" \
  --output-dir "${RUN_DIR}" \
  --coefficient 0.09 \
  --energy-offset-kj-mol 0 \
  --max-abs-basis-energy-kj-mol 1000 \
  --max-abs-path-energy-kj-mol 20 \
  --max-force-norm-kj-mol-nm 3000 \
  --min-pair-distance-nm 0.07 \
  --max-pair-distance-nm 2.5 \
  --max-radius-of-gyration-nm 0.85 \
  --output "${RUN_DIR}/teacher_prepare_report.json"

"${PYTHON_BIN}" "${MODULE}" exp010-label \
  --config "${CONFIG}" \
  --manifest "${MANIFEST}" \
  --trajectory "${SCREEN_ROOT}/hard_window0_run1/scratch_sample/hard_window_screening.dcd" \
  --trajectory "${SCREEN_ROOT}/hard_window0_run2/scratch_sample/hard_window_screening.dcd" \
  --trajectory "${SCREEN_ROOT}/hard_window0_run3/scratch_sample/hard_window_screening.dcd" \
  --topology "${TOPOLOGY}" \
  --selection-meta "${SELECTION_META}" \
  --frames "${FRAME_SPEC}" \
  --device "${DEVICE}" \
  --energy-offset-mode dataset_mean \
  --support-violation-policy exclude \
  --max-support-exclusion-fraction 0.05 \
  --output "${DATASET}"

for dimensions_order in 1:2 1:4 1:6 2:2 2:3 2:4; do
  dimensions="${dimensions_order%%:*}"
  order="${dimensions_order##*:}"
  "${PYTHON_BIN}" "${MODULE}" exp010-fit \
    --dataset "${DATASET}" \
    --dimensions "${dimensions}" \
    --order "${order}" \
    --ridge 1e-6 \
    --conditional-bins 24 \
    --output "${RUN_DIR}/fit_${dimensions}d_order${order}.json"
done

echo "EXP-010 教师标注与候选拟合完成: ${RUN_DIR}"
