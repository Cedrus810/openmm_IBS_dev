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

#include "exp026_control_plane_layout.h"

#include <array>
#include <cassert>
#include <iostream>
#include <limits>

using namespace OpenMM;

int main() {
    std::array<int, EXP026_STATUS_WORDS> raw = {
        EXP025_DEVICE_ERROR_NEIGHBOR_OVERFLOW,
        EXP026_STAGE_COMPUTE_Q,
        1206,
        81,
        300,
        2575,
        42,
        0
    };
    Exp026ControlPlaneStatus s =
        decodeExp026ControlPlaneStatus(raw.data(), (int) raw.size());
    assert(s.errorCode == EXP025_DEVICE_ERROR_NEIGHBOR_OVERFLOW);
    assert(s.errorStage == EXP026_STAGE_COMPUTE_Q);
    assert(s.activeEdges == 1206);
    assert(s.maxNeighbors == 81);
    assert(s.uniqueEnvironments == 300);
    assert(s.candidates == 2575);
    assert(s.epoch == 42);

    bool reset = false;
    assert(nextExp026UniqueEnvironmentEpoch(0, reset) == 1 && !reset);
    assert(nextExp026UniqueEnvironmentEpoch(41, reset) == 42 && !reset);
    assert(nextExp026UniqueEnvironmentEpoch(std::numeric_limits<int>::max(), reset) == 1 && reset);

    bool threw = false;
    try {
        (void) nextExp026UniqueEnvironmentEpoch(-1, reset);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);

    std::cout << "EXP-026 CONTROL-PLANE PRIVATE LAYOUT: PASS" << std::endl;
    return 0;
}