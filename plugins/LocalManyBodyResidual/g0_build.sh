#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# ABFE-IBS -- LocalManyBodyResidual plugin
#
# Copyright (c) 2026 Ruigeng Ji
#
# This plugin's directory layout, build scaffolding and API skeleton are
# derived from the OpenMM example plugin, which is MIT-licensed.
# Portions copyright (c) Stanford University and the Authors.
#
# Distributed under the MIT License; see LICENSE at the repository root for
# the full text.  This plugin is compiled against OpenMM headers and linked
# at run time against a separately installed OpenMM, whose CUDA, HIP and
# OpenCL platforms are covered by the LGPL -- see NOTICE.
# ----------------------------------------------------------------------------

# EXP-025 G0 build script.
#
# Deliberately NOT CMake yet (see docs/experiments/PLAN_EXP-025_local_manybody_cuda.md section
# 4.1 for the eventual CMakeLists.txt layout). This is a manual g++ build
# against the already-installed OpenMM 8.5.2 (git_revision 36a30cb) conda
# package's public API headers, private Reference-platform headers, and
# private CUDA/Common-platform headers -- no OpenMM source rebuild involved.
# Produces three plugin .so files loadable individually via
# openmm.Platform.loadPluginLibrary(), without touching the existing OpenMM
# install.
set -euo pipefail

CONDA_PREFIX="${CONDA_PREFIX:-/home/ruigengji/mambaforge/envs/openmm_dev}"
CUDA_HOME="${CUDA_HOME:-/opt/cuda}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"
mkdir -p "$BUILD"

CXX=g++
CXXSTD=-std=c++17
COMMON_FLAGS=(-fPIC -O2 "$CXXSTD" -Wno-deprecated-declarations)

INC_OPENMM=(-I"$CONDA_PREFIX/include")
INC_CUDA_PLATFORM=(-I"$CONDA_PREFIX/include/openmm/cuda")
INC_REFERENCE_PLATFORM=(-I"$CONDA_PREFIX/include/openmm/reference")
INC_CUDA_TOOLKIT=(-I"$CUDA_HOME/include")

LIBDIR=(-L"$CONDA_PREFIX/lib")
RPATH=(-Wl,-rpath,"$CONDA_PREFIX/lib" -Wl,-rpath,"$BUILD")

echo "== [1/3] public API lib: libOpenMMLocalManyBodyResidual.so =="
INC_SERIALIZATION=(-I"$HERE/serialization/include")
$CXX "${COMMON_FLAGS[@]}" "${INC_OPENMM[@]}" -I"$HERE/openmmapi/include" -c \
    "$HERE/openmmapi/src/LocalManyBodyResidualForce.cpp" \
    -o "$BUILD/LocalManyBodyResidualForce.o"
$CXX "${COMMON_FLAGS[@]}" "${INC_OPENMM[@]}" -I"$HERE/openmmapi/include" -c \
    "$HERE/openmmapi/src/LocalManyBodyResidualForceImpl.cpp" \
    -o "$BUILD/LocalManyBodyResidualForceImpl.o"
$CXX "${COMMON_FLAGS[@]}" "${INC_OPENMM[@]}" -I"$HERE/openmmapi/include" "${INC_SERIALIZATION[@]}" -c \
    "$HERE/serialization/src/LocalManyBodyResidualForceProxy.cpp" \
    -o "$BUILD/LocalManyBodyResidualForceProxy.o"
$CXX "${COMMON_FLAGS[@]}" "${INC_OPENMM[@]}" -I"$HERE/openmmapi/include" "${INC_SERIALIZATION[@]}" -c \
    "$HERE/serialization/src/LocalManyBodyResidualSerializationProxyRegistration.cpp" \
    -o "$BUILD/LocalManyBodyResidualSerializationProxyRegistration.o"
$CXX -shared "${COMMON_FLAGS[@]}" \
    "$BUILD/LocalManyBodyResidualForce.o" "$BUILD/LocalManyBodyResidualForceImpl.o" \
    "$BUILD/LocalManyBodyResidualForceProxy.o" "$BUILD/LocalManyBodyResidualSerializationProxyRegistration.o" \
    "${LIBDIR[@]}" -lOpenMM "${RPATH[@]}" \
    -o "$BUILD/libOpenMMLocalManyBodyResidual.so"

echo "== [2/3] Reference platform lib: libOpenMMLocalManyBodyResidualReference.so =="
$CXX "${COMMON_FLAGS[@]}" "${INC_OPENMM[@]}" "${INC_REFERENCE_PLATFORM[@]}" \
    -I"$HERE/platforms/reference/include" -I"$HERE/openmmapi/include" -c \
    "$HERE/platforms/reference/src/ReferenceLocalManyBodyResidualKernelFactory.cpp" \
    -o "$BUILD/ReferenceLocalManyBodyResidualKernelFactory.o"
$CXX "${COMMON_FLAGS[@]}" "${INC_OPENMM[@]}" "${INC_REFERENCE_PLATFORM[@]}" \
    -I"$HERE/platforms/reference/include" -I"$HERE/openmmapi/include" -c \
    "$HERE/platforms/reference/src/ReferenceLocalManyBodyResidualKernels.cpp" \
    -o "$BUILD/ReferenceLocalManyBodyResidualKernels.o"
$CXX -shared "${COMMON_FLAGS[@]}" \
    "$BUILD/ReferenceLocalManyBodyResidualKernelFactory.o" "$BUILD/ReferenceLocalManyBodyResidualKernels.o" \
    "${LIBDIR[@]}" -L"$BUILD" -lOpenMM -lOpenMMLocalManyBodyResidual "${RPATH[@]}" \
    -o "$BUILD/libOpenMMLocalManyBodyResidualReference.so"

echo "== [3/3] CUDA platform lib: libOpenMMLocalManyBodyResidualCUDA.so =="
$CXX "${COMMON_FLAGS[@]}" "${INC_OPENMM[@]}" "${INC_CUDA_PLATFORM[@]}" "${INC_CUDA_TOOLKIT[@]}" \
    -I"$HERE/platforms/cuda/include" -I"$HERE/openmmapi/include" -c \
    "$HERE/platforms/cuda/src/CudaLocalManyBodyResidualKernelFactory.cpp" \
    -o "$BUILD/CudaLocalManyBodyResidualKernelFactory.o"
$CXX "${COMMON_FLAGS[@]}" "${INC_OPENMM[@]}" "${INC_CUDA_PLATFORM[@]}" "${INC_CUDA_TOOLKIT[@]}" \
    -I"$HERE/platforms/cuda/include" -I"$HERE/openmmapi/include" -c \
    "$HERE/platforms/cuda/src/CudaLocalManyBodyResidualKernels.cpp" \
    -o "$BUILD/CudaLocalManyBodyResidualKernels.o"
$CXX -shared "${COMMON_FLAGS[@]}" \
    "$BUILD/CudaLocalManyBodyResidualKernelFactory.o" "$BUILD/CudaLocalManyBodyResidualKernels.o" \
    "${LIBDIR[@]}" -L"$BUILD" -lOpenMM -lOpenMMCUDA -lOpenMMLocalManyBodyResidual "${RPATH[@]}" \
    -o "$BUILD/libOpenMMLocalManyBodyResidualCUDA.so"

echo "== done =="
ls -la "$BUILD"/*.so
