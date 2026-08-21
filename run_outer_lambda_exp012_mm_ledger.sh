#!/usr/bin/env bash
# Formal CPU relabeling of the three frozen EXP-012 scratch trajectories.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${EXP012_PYTHON:-/home/ruigengji/mambaforge/envs/openmm_dev/bin/python}"
OUTPUT_ROOT="${EXP012_LEDGER_ROOT:-${ROOT}/output/outer_lambda_exp012/mm_ledger}"
PLATFORM="${EXP012_PLATFORM:-CPU}"
DEVICE_INDEX="${EXP012_DEVICE_INDEX:-}"
RUN_IDS_TEXT="${EXP012_RUN_IDS:-hard_window0_run1 hard_window0_run2 hard_window0_run3}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "EXP-012 Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

for RUN_ID in ${RUN_IDS_TEXT}; do
    RUN_OUTPUT="${OUTPUT_ROOT}/${RUN_ID}"
    REPORT="${RUN_OUTPUT}/ledger_report.json"
    if [[ -f "${REPORT}" ]]; then
        "${PYTHON_BIN}" -c '
import hashlib, json, pathlib, sys
report_path = pathlib.Path(sys.argv[1])
expected_run = sys.argv[2]
report = json.loads(report_path.read_text())
array_path = pathlib.Path(report["arrays_path"])
digest = hashlib.sha256(array_path.read_bytes()).hexdigest()
if report.get("status") != "COMPLETED" or report.get("run_id") != expected_run:
    raise SystemExit(f"invalid completed report: {report_path}")
if not array_path.is_file() or digest != report.get("arrays_sha256"):
    raise SystemExit(f"completed ledger array identity mismatch: {array_path}")
print(f"Skipping verified completed ledger: {expected_run}")
' "${REPORT}" "${RUN_ID}"
        continue
    fi
    if [[ -e "${RUN_OUTPUT}" ]]; then
        echo "Refusing to overwrite existing ledger output: ${RUN_OUTPUT}" >&2
        exit 1
    fi
    DEVICE_ARGS=()
    if [[ -n "${DEVICE_INDEX}" ]]; then
        DEVICE_ARGS=(--device-index "${DEVICE_INDEX}")
    fi
    "${PYTHON_BIN}" "${ROOT}/scripts/run_exp012_mm_ledger.py" \
        --preregistration "${ROOT}/protocols/EXP-012_preregistration.json" \
        --run-id "${RUN_ID}" \
        --output-dir "${RUN_OUTPUT}" \
        --platform "${PLATFORM}" \
        "${DEVICE_ARGS[@]}"
done
