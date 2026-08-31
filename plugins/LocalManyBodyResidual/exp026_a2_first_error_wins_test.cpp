// EXP-026 A2: first-error-wins across kernel stages, fault injection test.
//
// Scope: A2's mechanical change (consolidating 6 per-kernel errorFlagDevice/
// atomicExch checks into one deviceStatusDevice block, with a first-error-
// wins atomicCAS helper replacing last-write-wins atomicExch) has never had
// a test that actually exercises TWO DIFFERENT kernels each independently
// capable of raising a DIFFERENT error code within the SAME evaluation. All
// existing fail-closed tests (G2/G3's own, exp026_control_plane_correctness_
// test.cpp's EXP026-A/B/C, exp026_a1_1_dbdq_finiteness_test.cpp) each
// engineer exactly ONE error condition per fixture -- none of them prove
// that when kernel K1 (computeQ, checks EDGE_OVERFLOW/NEIGHBOR_OVERFLOW/
// UNIQUE_ENV_OVERFLOW/MIN_DISTANCE/HALF_BOX_TIE) and kernel K2 (readout,
// checks NONFINITE on dBdq) would BOTH independently fail on the same
// frame, the reported error is the chronologically FIRST one (K1, which
// runs before K2 in every evaluation), not whichever kernel happened to
// finish last on the GPU, and not silently corrupted by K2 partially
// executing on data it should never have been allowed to touch.
//
// This is exactly the risk the A2 conversion's own status doc flagged as a
// genuine (not just mechanical) finding: device-side self-gating had to be
// added at the top of kernels downstream of a possible earlier failure so
// they check-and-return before doing any work, rather than relying on the
// host to stop the launch sequence (which it does not do -- all kernels for
// one execute() call are enqueued into the same CUDA stream regardless of
// whether an earlier one flagged an error; only the FINAL consolidated host-
// side checkDeviceStatusOnce() after all kernels have run decides whether to
// throw). This test constructs a SINGLE fixture that is simultaneously:
//   (a) certain to overflow max_neighbors_per_ligand at K1 (computeQ), AND
//   (b) certain to also produce a NaN dBdq at K2 (readout) via a poisoned
//       typed MLP -- IF K1's overflow did not stop things first.
// If A2's first-error-wins design is correct, the reported exception must
// be EDGE/NEIGHBOR_OVERFLOW (from K1) every time, never NONFINITE (from
// K2) -- because K1 always executes before K2 within one execute() call,
// and once K1's status write lands, K2's own self-check must see the
// already-set error and refuse to overwrite it or do further work.
#include "openmm/LocalManyBodyResidualForce.h"
#include "OpenMM.h"

#include <iostream>
#include <string>
#include <vector>

using namespace OpenMM;
using namespace std;

namespace {

int g_failures = 0;
const double TEMPERATURE_KELVIN = 300.0;

void expectTrue(const string& label, bool ok, const string& detail = "") {
    cout << "  " << (ok ? "PASS" : "FAIL") << " " << label;
    if (!detail.empty()) cout << " (" << detail << ")";
    cout << "\n";
    if (!ok) g_failures++;
}

// Same poisoned-MLP construction as exp026_a1_1_dbdq_finiteness_test.cpp's
// buildPoisonedForce(): w0=1e38 (just under FLT_MAX~3.4e38) so w0*q
// overflows to +inf once q is O(10) or larger, while w0*0 stays exactly 0
// -- this is what makes rhoQGrad come out NaN (0*inf) while rhoQ/bReduced/
// sech2Shared all stay finite, exactly the fault A1.1's isfinite(dBdq)
// check exists to catch. Reused verbatim (not re-derived) since this
// specific poisoning is already validated to reliably produce a K2
// NONFINITE failure when nothing else stops it first.
//
// maxNeighborsPerLigand is set to 2 here (vs. 100 in the original A1.1
// test) -- with the SAME 4-environment-atom geometry, the true neighbor
// count (4) now exceeds this ceiling, guaranteeing a K1 NEIGHBOR_OVERFLOW
// on the exact same fixture that would otherwise also NaN out at K2.
LocalManyBodyResidualForce* buildDoubleFaultForce(int maxNeighborsPerLigand) {
    LocalManyBodyResidualForce* force = new LocalManyBodyResidualForce();
    force->setTemperatureKelvin(TEMPERATURE_KELVIN);
    force->setLigandTopologyIds({0});
    force->setTypeVocabulary({1});
    force->setAtomTypeIndex({0, 0, 0, 0, 0});
    force->setInnerCutoffAngstrom(4.0);
    force->setOuterCutoffAngstrom(5.0);
    force->setRadialCenters({0.0});
    force->setRadialWidthAngstrom(100.0);
    force->setPairWeight({5.0});
    force->setBMaxReduced(10.0);
    // maxEdges/maxEnvironmentAtoms left generously large -- this fixture's
    // single anchor makes active_edges==unique_environments==neighbor_count,
    // so only maxNeighborsPerLigand is deliberately made the binding
    // ceiling; a distinct sibling check below repeats this with maxEdges
    // instead, to confirm the result doesn't depend on which specific K1
    // ceiling fires first.
    force->setCapacityCeilings(/*maxEdges=*/100, maxNeighborsPerLigand, /*maxEnvironmentAtoms=*/100);

    LocalManyBodyTypedMLP mlp;
    mlp.w0.assign(16, 1.0e38);
    mlp.b0.assign(16, 0.0);
    mlp.w2.assign(256, 1.0);
    mlp.b2.assign(16, 0.0);
    mlp.w4.assign(16, 1.0);
    mlp.b4 = 0.0;
    force->setTypedMLP(0, mlp);
    return force;
}

LocalManyBodyResidualForce* buildDoubleFaultForceEdgeCeiling(int maxEdges) {
    LocalManyBodyResidualForce* force = new LocalManyBodyResidualForce();
    force->setTemperatureKelvin(TEMPERATURE_KELVIN);
    force->setLigandTopologyIds({0});
    force->setTypeVocabulary({1});
    force->setAtomTypeIndex({0, 0, 0, 0, 0});
    force->setInnerCutoffAngstrom(4.0);
    force->setOuterCutoffAngstrom(5.0);
    force->setRadialCenters({0.0});
    force->setRadialWidthAngstrom(100.0);
    force->setPairWeight({5.0});
    force->setBMaxReduced(10.0);
    force->setCapacityCeilings(maxEdges, /*maxNeighborsPerLigand=*/100, /*maxEnvironmentAtoms=*/100);

    LocalManyBodyTypedMLP mlp;
    mlp.w0.assign(16, 1.0e38);
    mlp.b0.assign(16, 0.0);
    mlp.w2.assign(256, 1.0);
    mlp.b2.assign(16, 0.0);
    mlp.w4.assign(16, 1.0);
    mlp.b4 = 0.0;
    force->setTypedMLP(0, mlp);
    return force;
}

Context* buildContext(LocalManyBodyResidualForce* force, System& system) {
    const int nAtoms = 5;
    for (int i = 0; i < nAtoms; i++) system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(20, 0, 0), Vec3(0, 20, 0), Vec3(0, 0, 20));
    system.addForce(force);
    VerletIntegrator* integrator = new VerletIntegrator(0.001);
    Platform& platform = Platform::getPlatformByName("CUDA");
    Context* context = new Context(system, *integrator, platform);
    vector<Vec3> positions = {
        Vec3(0.0, 0.0, 0.0),
        Vec3(0.10, 0.0, 0.0), Vec3(0.15, 0.0, 0.0), Vec3(0.20, 0.0, 0.0), Vec3(0.25, 0.0, 0.0),
    };
    context->setPositions(positions);
    return context;
}

}  // namespace

int main(int argc, char** argv) {
    (void) argc; (void) argv;
    try {
#ifdef OPENMM_CONDA_PLUGIN_DIR
        Platform::loadPluginsFromDirectory(OPENMM_CONDA_PLUGIN_DIR);
#endif
#ifdef PLUGIN_DIR
        Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidual.so");
        Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidualReference.so");
        Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidualCUDA.so");
#endif

        cout << "\n=== EXP026-A2-1: NEIGHBOR_OVERFLOW (K1) wins over a simultaneous NONFINITE (K2) ===\n";
        {
            // True neighbor count = 4, ceiling = 2 -> guaranteed K1
            // NEIGHBOR_OVERFLOW. Same poisoned MLP as A1.1's own test ->
            // guaranteed K2 NONFINITE if K1 did not stop things first.
            System system;
            LocalManyBodyResidualForce* force = buildDoubleFaultForce(/*maxNeighborsPerLigand=*/2);
            Context* context = nullptr;
            bool threw = false;
            string what;
            try {
                context = buildContext(force, system);
                context->getState(State::Energy);
            } catch (const exception& e) {
                threw = true;
                what = e.what();
            }
            expectTrue("double-fault fixture (neighbor overflow AND poisoned MLP) fails closed", threw, what);
            if (threw) {
                expectTrue("reported error is NEIGHBOR_OVERFLOW (K1, the chronologically first kernel)",
                           what.find("max_neighbors_per_ligand") != string::npos, what);
                expectTrue("reported error is NOT nonfinite (K2 must never win, or even be allowed to report, once K1 already failed)",
                           what.find("nonfinite") == string::npos, what);
                expectTrue("reported error does NOT name readout/K2 as the failing stage",
                           what.find("K2") == string::npos, what);
            }
            delete context;
        }

        cout << "\n=== EXP026-A2-2: EDGE_OVERFLOW (K1) wins over a simultaneous NONFINITE (K2) ===\n";
        {
            // Same idea, but binding on the total-edge ceiling instead of
            // the per-ligand-neighbor ceiling -- confirms the result isn't
            // an accident of which specific K1 check happens to be the one
            // that fires; both K1-stage error codes must equally take
            // precedence over the later K2 NONFINITE.
            System system;
            LocalManyBodyResidualForce* force = buildDoubleFaultForceEdgeCeiling(/*maxEdges=*/2);
            Context* context = nullptr;
            bool threw = false;
            string what;
            try {
                context = buildContext(force, system);
                context->getState(State::Energy);
            } catch (const exception& e) {
                threw = true;
                what = e.what();
            }
            expectTrue("double-fault fixture (edge overflow AND poisoned MLP) fails closed", threw, what);
            if (threw) {
                expectTrue("reported error is EDGE_OVERFLOW (K1, the chronologically first kernel)",
                           what.find("max_edges") != string::npos, what);
                expectTrue("reported error is NOT nonfinite (K2 must never win once K1 already failed)",
                           what.find("nonfinite") == string::npos, what);
            }
            delete context;
        }

        cout << "\n=== EXP026-A2-3: repeated fresh-Context evaluation of the double-fault fixture is deterministic (no first-error-wins race) ===\n";
        {
            // The A2 conversion replaced last-write-wins atomicExch with a
            // first-error-wins atomicCAS specifically because GPU kernel
            // completion order across threads/blocks is not otherwise
            // guaranteed -- a flaky "sometimes reports NEIGHBOR_OVERFLOW,
            // sometimes NONFINITE" result here would indicate the CAS
            // helper (or the self-gating that's supposed to stop K2 from
            // running at all once K1 has failed) has a real race, not just
            // an ordinary bug.
            const int REPEATS = 15;
            int neighborOverflowCount = 0, otherCount = 0;
            for (int i = 0; i < REPEATS; i++) {
                System system;
                LocalManyBodyResidualForce* force = buildDoubleFaultForce(/*maxNeighborsPerLigand=*/2);
                Context* context = nullptr;
                string what;
                try {
                    context = buildContext(force, system);
                    context->getState(State::Energy);
                } catch (const exception& e) {
                    what = e.what();
                }
                if (what.find("max_neighbors_per_ligand") != string::npos) neighborOverflowCount++;
                else otherCount++;
                delete context;
            }
            expectTrue("all " + to_string(REPEATS) + "/" + to_string(REPEATS) + " fresh attempts report NEIGHBOR_OVERFLOW (not flaky)",
                       neighborOverflowCount == REPEATS,
                       to_string(neighborOverflowCount) + "/" + to_string(REPEATS) + " were NEIGHBOR_OVERFLOW, "
                       + to_string(otherCount) + " were something else");
        }

        cout << "\n=== EXP-026 A2 FIRST-ERROR-WINS TEST: " << (g_failures == 0 ? "PASS" : "FAIL")
             << " (" << g_failures << " failing checks) ===\n";
        return g_failures == 0 ? 0 : 1;
    } catch (const exception& e) {
        cerr << "EXP-026 A2 FIRST-ERROR-WINS TEST FAIL-CLOSED: " << e.what() << "\n";
        return 1;
    }
}
