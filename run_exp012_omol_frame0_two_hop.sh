#!/usr/bin/env bash
set -euo pipefail

# One-frame C1 smoke only.  It does not train or scan the full trajectory.
# Old provisional artifacts are intentionally left untouched.

MODEL_PATH="${MODEL_PATH:-/home/ruigengji/.cache/mace/MACE-omol-0-extra-large-1024.model}"
OUT_DIR="${OUT_DIR:-output/outer_lambda_exp012/two_hop_frame0}"
DTYPE="${DTYPE:-float32}"
MAX_NODE_COUNT="${MAX_NODE_COUNT:-2500}"
MEMORY_LIMIT_GB="${MEMORY_LIMIT_GB:-64}"
TRAJECTORY="output/outer_lambda_slow_variable_screen/hard_window0_run1/scratch_sample/hard_window_screening.dcd"

mkdir -p "${OUT_DIR}"

python -m pytest -q \
  tests/test_exp012_mace_graph.py \
  tests/test_exp012_environment_discovery.py \
  tests/test_exp012_atom_mapping_cli.py \
  tests/test_exp012_mace_latent.py

python scripts/discover_exp012_environment_config.py \
  --topology output_lrc_fix/topology.cif \
  --trajectory "${TRAJECTORY}" \
  --frame-index 0 \
  --ligand-indices output/ligand_indices.json \
  --edge-cutoff-angstrom 6.0 \
  --interaction-layers 2 \
  --base-system-xml output_lrc_fix/system_native.xml \
  --box-vectors output_lrc_fix/box_vectors.npy \
  --report-output "${OUT_DIR}/environment_discovery_report.json" \
  --config-output "${OUT_DIR}/mace_environment_config.json"

python scripts/build_exp012_environment_manifest.py \
  --config "${OUT_DIR}/mace_environment_config.json" \
  --output "${OUT_DIR}/mace_environment_manifest.json"

python scripts/build_exp012_atom_mapping.py \
  --environment-manifest "${OUT_DIR}/mace_environment_manifest.json" \
  --config output/outer_lambda_exp012/provisional_mace_atom_mapping_config.json \
  --rebind-source-environment-manifest-sha \
  --resolved-config-output "${OUT_DIR}/mace_atom_mapping_config.json" \
  --output "${OUT_DIR}/mace_atom_mapping.json"

python scripts/smoke_exp012_mace_latent.py \
  --model "${MODEL_PATH}" \
  --c0-report output/outer_lambda_exp012/mace_omol_contract.json \
  --environment-manifest "${OUT_DIR}/mace_environment_manifest.json" \
  --atom-mapping "${OUT_DIR}/mace_atom_mapping.json" \
  --topology output_lrc_fix/topology.cif \
  --trajectory "${TRAJECTORY}" \
  --frame-indices 0 \
  --edge-cutoff-angstrom 6.0 \
  --geometric-upper-bound-angstrom 12.0 \
  --product-layer-index 1 \
  --device cpu \
  --dtype "${DTYPE}" \
  --max-node-count "${MAX_NODE_COUNT}" \
  --memory-limit-gb "${MEMORY_LIMIT_GB}" \
  --output "${OUT_DIR}/cpu_omol_latent_smoke.json"
