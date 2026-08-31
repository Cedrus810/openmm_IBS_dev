#!/usr/bin/env bash
# EXP-028: regression suite to run after the addArg->setArg argument-binding
# fix, before any performance claims. Compiles+runs each existing native C++
# test harness against the freshly rebuilt build/ (-> build_exp026_a2/) .so
# files. Stops at the first failure.
set -uo pipefail

CONDA_PREFIX="${CONDA_PREFIX:-/home/ruigengji/mambaforge/envs/openmm_dev}"
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"
mkdir -p "$BUILD/tests"

CXX=g++
CXXSTD=-std=c++17
COMMON_FLAGS=(-fPIC -O2 "$CXXSTD" -Wno-deprecated-declarations)
INC=(-I"$CONDA_PREFIX/include" -I"$CONDA_PREFIX/include/openmm/cuda" -I"$CONDA_PREFIX/include/openmm/reference"
     -I"$CUDA_HOME/include" -I"$HERE/openmmapi/include" -I"$HERE")
DEFS=(-DOPENMM_CONDA_PLUGIN_DIR="\"$CONDA_PREFIX/lib/plugins\"" -DPLUGIN_DIR="\"$BUILD\"")
LIBDIR=(-L"$CONDA_PREFIX/lib" -L"$BUILD")
RPATH=(-Wl,-rpath,"$CONDA_PREFIX/lib" -Wl,-rpath,"$BUILD")
LIBS=(-lOpenMM -lOpenMMLocalManyBodyResidual -lcrypto)

TESTS=(
  g0_smoke_test.cpp
  g0_serialization_roundtrip_test.cpp
  exp026_control_plane_layout_test.cpp
  exp026_a1_1_dbdq_finiteness_test.cpp
  exp026_a2_first_error_wins_test.cpp
)

# yyjson is used only by offline JSON-fixture readers, never by the plugin
# shared libraries or the EXP-029 runtime.  Keep these tests opt-in until the
# development header/library are installed on the compute node.
YYJSON_TESTS=(
  g1g_openmm_reference_parity_test.cpp
  g2_cuda_reference_parity_test.cpp
  g3_local_csr_test.cpp
  exp026_control_plane_correctness_test.cpp
)
if [[ "${ENABLE_YYJSON_TESTS:-0}" == "1" ]]; then
  if [[ ! -f "$CONDA_PREFIX/include/yyjson.h" ]]; then
    echo "!! ENABLE_YYJSON_TESTS=1 but $CONDA_PREFIX/include/yyjson.h is missing"
    exit 2
  fi
  LIBS+=(-lyyjson)
  TESTS+=("${YYJSON_TESTS[@]}")
else
  echo "== yyjson-dependent offline fixture tests skipped (set ENABLE_YYJSON_TESTS=1 to enable) =="
fi

overall=0
for t in "${TESTS[@]}"; do
  name="${t%.cpp}"
  echo "======================================================================"
  echo "== $name =="
  echo "======================================================================"
  if ! "$CXX" "${COMMON_FLAGS[@]}" "${INC[@]}" "${DEFS[@]}" "$HERE/$t" \
        "${LIBDIR[@]}" "${LIBS[@]}" "${RPATH[@]}" -o "$BUILD/tests/$name" 2>&1; then
    echo "!! COMPILE FAILED: $name"
    overall=1
    continue
  fi
  if ! "$BUILD/tests/$name"; then
    echo "!! RUNTIME FAILED: $name"
    overall=1
  else
    echo "-- PASS: $name --"
  fi
done

echo "======================================================================"
if [ "$overall" -eq 0 ]; then
  echo "ALL TESTS PASS"
else
  echo "AT LEAST ONE TEST FAILED"
fi
exit "$overall"
