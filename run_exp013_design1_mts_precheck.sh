#!/usr/bin/env bash
set -euo pipefail

# DEC-056：方案① whole fused Group-1 slow 的低成本 N=1/2/4/8 预检。
# 预检脚本自身拒绝覆盖已有 report/TorchScript，并且不构造 N=16/32。

PROJECT_DIR="${PROJECT_DIR:-/home/ruigengji/ABFE_IBS/Atenolol-rank11}"
MAMBA_PROFILE="${MAMBA_PROFILE:-/home/ruigengji/mambaforge/etc/profile.d/mamba.sh}"
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-openmm_dev}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/output_lrc_fix}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/output/outer_lambda_exp013_design1_precheck}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/output/outer_lambda_exp012/student_checkpoints/hard_window0_run1__direct_gap__seed0.pt}"
SCRIPT="${PROJECT_DIR}/scripts/check_exp013_design1_mts_precheck.py"

if [[ ! -f "${MAMBA_PROFILE}" ]]; then
  echo "missing mamba profile: ${MAMBA_PROFILE}" >&2
  exit 2
fi
source "${MAMBA_PROFILE}"
mamba activate "${MAMBA_ENV_NAME}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the CUDA-bound production checkpoint" >&2
  exit 2
fi
nvidia-smi
"${PYTHON_BIN}" -c 'import torch; import openmm; assert torch.cuda.is_available(), "torch.cuda.is_available() is false"; print("python=", __import__("sys").executable); print("torch=", torch.__version__); print("openmm_platforms=", [openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())])'

mkdir -p "${RUN_DIR}"

"${PYTHON_BIN}" "${SCRIPT}" \
  --output-root "${OUTPUT_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --torchscript-output "${RUN_DIR}/student_torchscript.pt" \
  --output "${RUN_DIR}/report.json" \
  "$@"

echo "EXP-013 方案①低成本预检完成: ${RUN_DIR}/report.json"
