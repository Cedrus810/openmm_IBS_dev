#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAMBA_PROFILE="${MAMBA_PROFILE:-/home/ruigengji/mambaforge/etc/profile.d/mamba.sh}"
MAMBA_ENV_NAME="${MAMBA_ENV_NAME:-openmm_dev}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/output_lrc_fix}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/output/outer_lambda_exp012/student_checkpoints/hard_window0_run1__direct_gap__seed0.pt}"
TORCHSCRIPT="${TORCHSCRIPT:-${PROJECT_DIR}/output/outer_lambda_exp013_design1_qualification/student_torchscript.pt}"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/output/outer_lambda_exp013_design2_n1_ess}"
REPORT="${REPORT:-${RUN_DIR}/report.json}"

if [[ ! -f "${MAMBA_PROFILE}" ]]; then
    echo "missing mamba profile: ${MAMBA_PROFILE}" >&2
    exit 2
fi
source "${MAMBA_PROFILE}"
mamba activate "${MAMBA_ENV_NAME}"
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "missing environment python: ${PYTHON_BIN}" >&2
    exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required: scheme-2 N=1 must run on CUDA" >&2
    exit 2
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"${PYTHON_BIN}" - <<'PY'
import openmm
import torch
print(f"python_cuda_available={torch.cuda.is_available()}")
print(f"torch_version={torch.__version__}")
print(f"openmm_version={openmm.version.version}")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false; refusing non-CUDA fallback")
PY

SCRIPT="${PROJECT_DIR}/scripts/check_exp013_design2_n1_ess.py"
if [[ -e "${REPORT}" ]]; then
    echo "refusing to overwrite existing report: ${REPORT}" >&2
    exit 2
fi
mkdir -p "${RUN_DIR}"
"${PYTHON_BIN}" "${SCRIPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --platform CUDA \
    --checkpoint "${CHECKPOINT}" \
    --torchscript "${TORCHSCRIPT}" \
    --output "${REPORT}" \
    "$@"

echo "EXP-013 方案② N=1 ESS 信号检查完成: ${REPORT}"
