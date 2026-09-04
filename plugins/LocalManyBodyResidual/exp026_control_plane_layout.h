/* ---------------------------------------------------------------------------- *
 * ABFE-IBS -- LocalManyBodyResidual plugin                                     *
 *                                                                              *
 * Copyright (c) 2026 Ruigeng Ji                                                *
 *                                                                              *
 * This plugin's directory layout, build scaffolding and API skeleton are       *
 * derived from the OpenMM example plugin, which is MIT-licensed.               *
 * Portions copyright (c) Stanford University and the Authors.                  *
 *                                                                              *
 * Distributed under the MIT License; see LICENSE at the repository root for    *
 * the full text.  This plugin is compiled against OpenMM headers and linked    *
 * at run time against a separately installed OpenMM, whose CUDA, HIP and       *
 * OpenCL platforms are covered by the LGPL -- see NOTICE.                      *
 * ---------------------------------------------------------------------------- */

#ifndef EXP026_CONTROL_PLANE_LAYOUT_H_
#define EXP026_CONTROL_PLANE_LAYOUT_H_

/*
 * EXP-026 private CUDA control-plane layout.
 *
 * This is NOT a new Force, KernelImpl, public API, serialization schema, or
 * model layout.  It is included only by the existing EXP-025
 * CudaCalcLocalManyBodyResidualForceKernel implementation and its tests.
 *
 * The device buffer is ComputeArray<int>[8].  Array offsets are used instead
 * of a C/CUDA struct so NVRTC and the host cannot disagree about padding.
 */

#include "r1_model_layout.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

/* deviceStatusDevice: int[EXP026_STATUS_WORDS] */
#define EXP026_STATUS_ERROR_CODE 0
#define EXP026_STATUS_ERROR_STAGE 1
#define EXP026_STATUS_ACTIVE_EDGES 2
#define EXP026_STATUS_MAX_NEIGHBORS 3
#define EXP026_STATUS_UNIQUE_ENVIRONMENTS 4
#define EXP026_STATUS_CANDIDATES 5
#define EXP026_STATUS_EPOCH 6
#define EXP026_STATUS_RESERVED 7
#define EXP026_STATUS_WORDS 8

/* First-error stage.  Error code values remain EXP025_DEVICE_ERROR_* exactly. */
#define EXP026_STAGE_NONE 0
#define EXP026_STAGE_REBUILD_PREFIX 1
#define EXP026_STAGE_REBUILD_FILL 2
#define EXP026_STAGE_COMPUTE_Q 3
#define EXP026_STAGE_READOUT 4
#define EXP026_STAGE_FORCE_SCATTER 5
#define EXP026_STAGE_HOST_VALIDATE 6

namespace OpenMM {

struct Exp026ControlPlaneStatus {
    int errorCode;
    int errorStage;
    int activeEdges;
    int maxNeighbors;
    int uniqueEnvironments;
    int candidates;
    int epoch;
    int reserved;
};

inline Exp026ControlPlaneStatus decodeExp026ControlPlaneStatus(const int* words, int count) {
    if (words == nullptr || count != EXP026_STATUS_WORDS)
        throw std::invalid_argument("EXP-026 device status must contain exactly 8 int32 words");
    Exp026ControlPlaneStatus s;
    s.errorCode = words[EXP026_STATUS_ERROR_CODE];
    s.errorStage = words[EXP026_STATUS_ERROR_STAGE];
    s.activeEdges = words[EXP026_STATUS_ACTIVE_EDGES];
    s.maxNeighbors = words[EXP026_STATUS_MAX_NEIGHBORS];
    s.uniqueEnvironments = words[EXP026_STATUS_UNIQUE_ENVIRONMENTS];
    s.candidates = words[EXP026_STATUS_CANDIDATES];
    s.epoch = words[EXP026_STATUS_EPOCH];
    s.reserved = words[EXP026_STATUS_RESERVED];
    return s;
}

/*
 * Epoch zero means "never touched".  The implementation uses positive signed
 * int epochs because ComputeArray<int> is already the plugin's frozen device
 * scalar ABI.  Reaching INT_MAX is not an error: after the previous execute()
 * has completed, upload one all-zero tag array once and restart from epoch 1.
 */
inline int nextExp026UniqueEnvironmentEpoch(int current, bool& requiresTagReset) {
    if (current < 0)
        throw std::invalid_argument("EXP-026 epoch must be nonnegative");
    if (current == std::numeric_limits<int>::max()) {
        requiresTagReset = true;
        return 1;
    }
    requiresTagReset = false;
    return current+1;
}

/* Compile-time proof that the frozen EXP-025 public error ABI did not move. */
static_assert(EXP025_DEVICE_ERROR_OK == 0, "EXP-025 error ABI changed");
static_assert(EXP025_DEVICE_ERROR_HALF_BOX_TIE == 1, "EXP-025 error ABI changed");
static_assert(EXP025_DEVICE_ERROR_MIN_DISTANCE == 2, "EXP-025 error ABI changed");
static_assert(EXP025_DEVICE_ERROR_EDGE_OVERFLOW == 3, "EXP-025 error ABI changed");
static_assert(EXP025_DEVICE_ERROR_NEIGHBOR_OVERFLOW == 4, "EXP-025 error ABI changed");
static_assert(EXP025_DEVICE_ERROR_UNIQUE_ENV_OVERFLOW == 5, "EXP-025 error ABI changed");
static_assert(EXP025_DEVICE_ERROR_NONFINITE == 6, "EXP-025 error ABI changed");
static_assert(EXP025_DEVICE_ERROR_CANDIDATE_OVERFLOW == 7, "EXP-025 error ABI changed");
static_assert(EXP025_DEVICE_ERROR_UNSUPPORTED_BOX == 8, "EXP-025 error ABI changed");

} // namespace OpenMM

#endif