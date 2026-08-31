// EXP-026 Patch A1.1: dBdq finiteness fault-injection test.
//
// Proves the gap Patch A1.1 closed in exp025Readout (K2): the pre-existing
// check only ever validated the SHARED bReduced/sech2Shared scalars, never
// each anchor's own dBdq[anchor] = sech2Shared * rhoQGrad product. This test
// constructs a fully synthetic single-anchor Force (no real EXP-020 payload
// needed) whose typed MLP is deliberately poisoned so that:
//
//   - q (the raw C2*RBF sum, computed in K1/K6a) stays a perfectly ordinary
//     finite number (~20) -- the EXISTING K1/K6a NONFINITE check on the raw
//     q-sum does NOT fire, confirming the fault is isolated to K2's own MLP
//     evaluation, not upstream geometry/accumulation;
//   - at that q, every hidden-layer pre-activation h0=w0*q+b0 genuinely
//     overflows float32 (w0=1e38, q~20 -> w0*q~2e39 > FLT_MAX), so the
//     forward value a0=SiLU(h0) becomes a CLEAN +inf (SiLU(+inf)=+inf, not
//     NaN) while the *gradient* term silu'(h0) evaluates the IEEE754
//     ill-defined 1 - 1*(1 - 1) x inf = 1 + inf*0 = NaN internally -- a
//     textbook 0*inf=NaN trap that only the *derivative* path hits, not the
//     forward path;
//   - at x=0 (the separate rhoZero evaluation the SAME MLP is run at),
//     w0*0=0 exactly (finite*0=0, not NaN -- only inf*0 is NaN), so rhoZero
//     stays perfectly ordinary/finite;
//   - therefore rhoQ (=value) diverges to a CLEAN +inf while rhoQGrad
//     (=grad) becomes NaN -- and because tanh(+inf) is EXACTLY 1.0 in
//     IEEE754, bReduced=bMaxReduced*1.0 and sech2Shared=1-1^2=0 BOTH stay
//     finite, so the pre-A1.1 check (isfinite(bReduced)||isfinite
//     (sech2Shared)) would have missed this entirely. Only the A1.1 check on
//     dBdq[anchor]=sech2Shared*rhoQGrad=0*NaN=NaN catches it.
//
// This test does NOT try to prove the counterfactual "the old binary passes"
// (that binary no longer exists to run) -- it proves the NEW check fires,
// with a message naming the correct stage, on a scenario engineered so the
// OLD check's own logic (spelled out above) provably would not have fired.
#include "openmm/LocalManyBodyResidualForce.h"
#include "OpenMM.h"

#include <cmath>
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

// Single anchor, four environment atoms all well inside the inner cutoff
// (c2~1 for each), each contributing pairWeight=5.0 * RBF~1.0 -- q~20.
// numRadialBasis=1 (broad width=100 Angstrom keeps the single Gaussian
// essentially flat/~1 across the whole 0-5 Angstrom active range) so the
// per-edge contribution is dominated entirely by the pairWeight constant,
// making q trivially predictable by hand.
LocalManyBodyResidualForce* buildPoisonedForce() {
    LocalManyBodyResidualForce* force = new LocalManyBodyResidualForce();
    force->setTemperatureKelvin(TEMPERATURE_KELVIN);
    force->setLigandTopologyIds({0});
    force->setTypeVocabulary({1});          // single type, arbitrary atomic number
    force->setAtomTypeIndex({0, 0, 0, 0, 0}); // anchor + 4 environment atoms, all type 0
    force->setInnerCutoffAngstrom(4.0);
    force->setOuterCutoffAngstrom(5.0);
    force->setRadialCenters({0.0});          // numRadialBasis == 1
    force->setRadialWidthAngstrom(100.0);    // broad -> Gaussian ~= 1 everywhere in [0,5] A
    force->setPairWeight({5.0});             // typeCount(1)^2 * numRadialBasis(1) == 1 entry
    force->setBMaxReduced(10.0);
    force->setCapacityCeilings(/*maxEdges=*/100, /*maxNeighborsPerLigand=*/100, /*maxEnvironmentAtoms=*/100);

    // Poisoned typed MLP: w0 finite (1e38, just under FLT_MAX~3.4e38) so
    // w0*q (q~20) overflows float32 to +inf at runtime, while w0*0 (the
    // separate rhoZero evaluation) stays exactly 0. All connections
    // positive and uniform so infinities never cancel (no inf-inf anywhere).
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

Context* buildContext(LocalManyBodyResidualForce* force, const string& platformName, System& system) {
    // anchor at origin; 4 environment atoms at 1.0/1.5/2.0/2.5 Angstrom
    // (all < 4 Angstrom inner cutoff, so C2~=1 for each -- q ~= 4*5.0 = 20,
    // comfortably clearing the ~3.4 threshold needed to overflow w0=1e38).
    const int nAtoms = 5;
    for (int i = 0; i < nAtoms; i++) system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(Vec3(20, 0, 0), Vec3(0, 20, 0), Vec3(0, 0, 20));
    system.addForce(force);
    VerletIntegrator* integrator = new VerletIntegrator(0.001);
    Platform& platform = Platform::getPlatformByName(platformName);
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

        cout << "\n=== EXP026-A1.1: dBdq=NaN fails closed (single precision) ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildPoisonedForce();
            Context* context = nullptr;
            bool threw = false;
            string what;
            try {
                context = buildContext(force, "CUDA", system);
                context->getState(State::Energy);
            } catch (const exception& e) {
                threw = true;
                what = e.what();
            }
            expectTrue("poisoned MLP (rhoQ=+inf clean, rhoQGrad=NaN, tanh-saturated bReduced/sech2Shared BOTH finite) fails closed", threw, what);
            if (threw) {
                expectTrue("failure names K2 (readout), where the A1.1 dBdq check lives",
                           what.find("readout") != string::npos, what);
                expectTrue("failure is NONFINITE (not some unrelated overflow/geometry error)",
                           what.find("nonfinite") != string::npos, what);
            }
            delete context;
        }

        cout << "\n=== EXP026-A1.1: same poisoned MLP under mixed precision ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildPoisonedForce();
            map<string, string> props = {{"Precision", "mixed"}};
            for (int i = 0; i < 5; i++) system.addParticle(1.0);
            system.setDefaultPeriodicBoxVectors(Vec3(20, 0, 0), Vec3(0, 20, 0), Vec3(0, 0, 20));
            system.addForce(force);
            VerletIntegrator* integrator = new VerletIntegrator(0.001);
            Platform& platform = Platform::getPlatformByName("CUDA");
            Context* context = new Context(system, *integrator, platform, props);
            vector<Vec3> positions = {
                Vec3(0.0, 0.0, 0.0),
                Vec3(0.10, 0.0, 0.0), Vec3(0.15, 0.0, 0.0), Vec3(0.20, 0.0, 0.0), Vec3(0.25, 0.0, 0.0),
            };
            context->setPositions(positions);
            bool threw = false;
            string what;
            try { context->getState(State::Energy); }
            catch (const exception& e) { threw = true; what = e.what(); }
            expectTrue("poisoned MLP fails closed under mixed precision too (production always runs mixed)", threw, what);
            if (threw)
                expectTrue("mixed-precision failure also names K2 (readout)", what.find("readout") != string::npos, what);
            delete context;
        }

        cout << "\n=== EXP026-A1.1: healthy (unpoisoned) MLP still computes finite energy normally ===\n";
        {
            // Sanity control: an all-zero MLP (same geometry, no poisoning)
            // must NOT trip the new check -- proves A1.1 doesn't false-positive
            // on the ordinary case.
            System system;
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
            force->setCapacityCeilings(100, 100, 100);
            LocalManyBodyTypedMLP mlp;
            mlp.w0.assign(16, 0.0); mlp.b0.assign(16, 0.0);
            mlp.w2.assign(256, 0.0); mlp.b2.assign(16, 0.0);
            mlp.w4.assign(16, 0.0); mlp.b4 = 0.0;
            force->setTypedMLP(0, mlp);
            Context* context = buildContext(force, "CUDA", system);
            bool threw = false;
            double energy = 0.0;
            try { energy = context->getState(State::Energy).getPotentialEnergy(); }
            catch (const exception&) { threw = true; }
            expectTrue("all-zero (unpoisoned) MLP does not fail closed", !threw);
            expectTrue("all-zero MLP energy is exactly 0", !threw && energy == 0.0, "energy=" + to_string(energy));
            delete context;
        }

        cout << "\n=== EXP-026 A1.1 dBdq FINITENESS FAULT-INJECTION TEST: "
             << (g_failures == 0 ? "PASS" : "FAIL") << " (" << g_failures << " failing checks) ===\n";
        return g_failures == 0 ? 0 : 1;
    } catch (const exception& e) {
        cerr << "EXP-026 A1.1 TEST FAIL-CLOSED (unexpected): " << e.what() << "\n";
        return 1;
    }
}
