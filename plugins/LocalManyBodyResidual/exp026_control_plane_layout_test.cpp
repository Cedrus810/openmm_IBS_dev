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