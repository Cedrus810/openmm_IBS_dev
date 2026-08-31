// EXP-026 Patch A1: device-resident control-plane correctness test.
//
// Scope: this test targets exactly what Patch A changed (unique-environment
// epoch tags + the DeviceStatusV1 active_edges/max_neighbors/unique_
// environments counters) and NOTHING else. The inherited G0-G3 suite
// already re-validates energy/force correctness end-to-end on the real
// canonical fixture and on every pre-existing scenario (mixed precision,
// triclinic, reorder, rebuild/no-rebuild, etc.) -- see
// project_exp026_patch_a1_regression_pass memory / session log. What that
// suite does NOT do is exercise the *counting* semantics Patch A actually
// introduced:
//
//   - PLAN section 9.2 item 1: unique-env count vs an independent host
//     reference, at an EXACT boundary (not "+5 over", which the existing
//     G2/G3 fail-closed tests use and which only ever happens to exercise
//     NEIGHBOR_OVERFLOW because those tests use a single anchor).
//   - PLAN section 9.2 item 4/5: edges<=2048, neighbors<=80, unique-env
//     <=320 boundary PASS, and each of the three +1-over ceilings FAILS
//     CLOSED with its own distinct error code -- EDGE_OVERFLOW and
//     UNIQUE_ENV_OVERFLOW are never actually triggered by any existing G2/
///    G3 test (both use one ligand atom, so a per-ligand neighbor overflow
//     and a total-edge overflow are numerically indistinguishable there).
//   - PLAN section 9.2 item 2: repeated evaluation on the same live Context
//     does not contaminate/leak counts across the epoch-tag reset.
//   - PLAN section 9.2 item 20 (reduced from the full 10,000-step
//     production-frame run for wall-clock budget, but same intent): long
//     repeated evaluation does not drift, hang, or go non-finite.
//   - PLAN section 9.2 item 18: a fail-closed exception on one evaluation
//     must not corrupt a later, valid evaluation on the SAME Context (no
//     partial-force / stale-status leakage).
//
// Every check here still goes through the PUBLIC OpenMM API only (energy/
// forces/exceptions) -- there is no accessor for uniqueEnvironmentCount,
// activeEdges, or maxNeighbors, and this test does not add one. Exact
// counts are verified INDIRECTLY: a system built so the true count is
// exactly K is checked to pass when the ceiling is K and fail closed when
// the ceiling is K-1; if the device-side count were off by even one in
// either direction, one of those two assertions would flip.
#include "openmm/LocalManyBodyResidualForce.h"
#include "OpenMM.h"
#include "g1_math_core.h"
#include "g1_payload_io.h"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace OpenMM;
using namespace exp025_g1;
using namespace std;

namespace {

int g_failures = 0;
const double TEMPERATURE_KELVIN = 300.0;
const double ENERGY_ABS_TOL = 1e-4;   // kJ/mol -- same G2/G3 tolerance
const double FORCE_ABS_TOL = 1e-3;    // kJ/mol/nm

void expectTrue(const string& label, bool ok, const string& detail = "") {
    cout << "  " << (ok ? "PASS" : "FAIL") << " " << label;
    if (!detail.empty()) cout << " (" << detail << ")";
    cout << "\n";
    if (!ok) g_failures++;
}

void expectClose(const string& label, double actual, double expected, double tol) {
    double diff = fabs(actual - expected);
    bool ok = diff <= tol;
    cout << "  " << (ok ? "PASS" : "FAIL") << " " << label << ": actual=" << actual
         << " expected=" << expected << " |diff|=" << diff << " tol=" << tol << "\n";
    if (!ok) g_failures++;
}

// Same construction as every other EXP-025/026 test file (buildForceFromPayload/
// buildContext are deliberately re-authored per test file rather than shared --
// see G1/G2/G3's own comments on why: no shared internal-state shortcuts).
// `maxEdges`/`maxNeighborsPerLigand`/`maxEnvironmentAtoms` are caller-controlled
// here (not taken from the payload's real model ceilings) because this test's
// entire point is putting those three ceilings at deliberately small, exact
// boundary values -- the real model's ceilings (2048/80/320) are far larger
// than the tiny synthetic geometries below.
LocalManyBodyResidualForce* buildForceFromPayload(const LoadedPayload& loaded, const AtomSystemView& fx,
                                                   int maxEdges, int maxNeighborsPerLigand, int maxEnvironmentAtoms) {
    const ModelParams& model = loaded.model;
    LocalManyBodyResidualForce* force = new LocalManyBodyResidualForce();
    force->setTemperatureKelvin(TEMPERATURE_KELVIN);
    force->setLigandTopologyIds(vector<int>(fx.ligandTopologyIds.begin(), fx.ligandTopologyIds.end()));
    force->setTypeVocabulary(vector<int>(loaded.typeVocabulary.begin(), loaded.typeVocabulary.end()));
    force->setAtomTypeIndex(vector<int>(fx.atomTypeIndex.begin(), fx.atomTypeIndex.end()));
    force->setInnerCutoffAngstrom(model.innerCutoffAngstrom);
    force->setOuterCutoffAngstrom(model.outerCutoffAngstrom);
    force->setRadialCenters(model.radialCenters);
    force->setRadialWidthAngstrom(model.radialWidth);
    force->setPairWeight(model.pairWeight);
    force->setBMaxReduced(model.bMaxReduced);
    force->setCapacityCeilings(maxEdges, maxNeighborsPerLigand, maxEnvironmentAtoms);  // G2 brute-force path: candidateListCapacity left at 0
    for (int t = 0; t < model.typeCount; t++) {
        const TypedMLP& src = model.rho[t];
        LocalManyBodyTypedMLP mlp;
        mlp.w0.assign(src.W0.begin(), src.W0.end());
        mlp.b0.assign(src.b0.begin(), src.b0.end());
        mlp.w2.resize(256);
        for (int o = 0; o < 16; o++)
            for (int k = 0; k < 16; k++) mlp.w2[(size_t) o * 16 + k] = src.W2[o][k];
        mlp.b2.assign(src.b2.begin(), src.b2.end());
        mlp.w4.assign(src.W4.begin(), src.W4.end());
        mlp.b4 = src.b4;
        force->setTypedMLP(t, mlp);
    }
    return force;
}

Context* buildContext(const AtomSystemView& fx, LocalManyBodyResidualForce* force, const string& platformName, System& system) {
    for (int i = 0; i < fx.nAtoms; i++) system.addParticle(1.0);
    system.setDefaultPeriodicBoxVectors(
        Vec3(fx.boxNm[0][0], fx.boxNm[0][1], fx.boxNm[0][2]),
        Vec3(fx.boxNm[1][0], fx.boxNm[1][1], fx.boxNm[1][2]),
        Vec3(fx.boxNm[2][0], fx.boxNm[2][1], fx.boxNm[2][2]));
    system.addForce(force);
    VerletIntegrator* integrator = new VerletIntegrator(0.001);
    Platform& platform = Platform::getPlatformByName(platformName);
    Context* context = new Context(system, *integrator, platform);
    vector<Vec3> positions(fx.nAtoms);
    for (int i = 0; i < fx.nAtoms; i++) positions[i] = Vec3(fx.positionsNm[i][0], fx.positionsNm[i][1], fx.positionsNm[i][2]);
    context->setPositions(positions);
    return context;
}

double maxForceDiff(const vector<Vec3>& a, const vector<Vec3>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        for (int c = 0; c < 3; c++) m = max(m, fabs(a[i][c] - b[i][c]));
    return m;
}

// Two ligand anchors 8 Angstrom apart (each atom-index 0 and 1), five
// environment atoms placed along the same axis so the anchor<->environment
// membership is exactly controlled by construction (outer_cutoff = 5
// Angstrom from the real payload -- see the distance table in the block
// comment at the call sites below):
//
//   atom 0 = anchor0                     (0.0 Angstrom)
//   atom 1 = anchor1                     (8.0 Angstrom)
//   atom 2 = e1        private to anchor0 (1.0 Angstrom: 1.0 from anchor0, 7.0 from anchor1)
//   atom 3 = e2        private to anchor0 (2.0 Angstrom: 2.0 from anchor0, 6.0 from anchor1)
//   atom 4 = eShared   seen by BOTH       (4.0 Angstrom: 4.0 from anchor0, 4.0 from anchor1)
//   atom 5 = e3        private to anchor1 (10.0 Angstrom: 10.0 from anchor0, 2.0 from anchor1)
//   atom 6 = e4        private to anchor1 (11.0 Angstrom: 11.0 from anchor0, 3.0 from anchor1)
//
// => anchor0 active neighbors = {e1,e2,eShared} = 3; anchor1 active
//    neighbors = {eShared,e3,e4} = 3; TOTAL ACTIVE EDGES = 6 (edges are
//    anchor-scoped, eShared counts once per anchor that sees it); UNIQUE
//    ENVIRONMENT ATOMS = {e1,e2,eShared,e3,e4} = 5 (eShared counts once,
//    period, regardless of how many anchors touch it -- this is exactly
//    the distinction Patch A's epoch-tag dedup exists to get right).
// Box is 20 nm cubic: outer_cutoff=0.5 nm, so 2*outer_cutoff=1.0 nm <<
// any face height, comfortably satisfying the half-box-tie safety
// precondition with huge margin.
AtomSystemView twoAnchorFiveEnvFixture() {
    AtomSystemView fx;
    fx.nAtoms = 7; fx.nLigand = 2;
    fx.positionsNm = {
        {0.0, 0.0, 0.0},   // anchor0
        {0.8, 0.0, 0.0},   // anchor1
        {0.1, 0.0, 0.0},   // e1
        {0.2, 0.0, 0.0},   // e2
        {0.4, 0.0, 0.0},   // eShared
        {1.0, 0.0, 0.0},   // e3
        {1.1, 0.0, 0.0},   // e4
    };
    fx.boxNm = {{{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
    fx.atomTypeIndex.assign(fx.nAtoms, 0);
    fx.ligandTopologyIds = {0, 1};
    return fx;
}

// Single anchor with exactly 3 active neighbors -- for isolating the
// per-ligand neighbor ceiling from the edge/unique-env ceilings (with only
// one anchor, active_edges == neighbor_count and unique_environments ==
// neighbor_count too, so this fixture is deliberately NOT used for the
// edge/unique-env boundary tests above; it exists only to pin down
// max_neighbors_per_ligand in isolation).
AtomSystemView oneAnchorThreeEnvFixture() {
    AtomSystemView fx;
    fx.nAtoms = 4; fx.nLigand = 1;
    fx.positionsNm = {
        {0.0, 0.0, 0.0},   // anchor0
        {0.1, 0.0, 0.0},   // e1 (1 Angstrom)
        {0.2, 0.0, 0.0},   // e2 (2 Angstrom)
        {0.4, 0.0, 0.0},   // e3 (4 Angstrom)
    };
    fx.boxNm = {{{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
    fx.atomTypeIndex.assign(fx.nAtoms, 0);
    fx.ligandTopologyIds = {0};
    return fx;
}

// Same anchor+3-neighbor geometry as oneAnchorThreeEnvFixture, PLUS a 5th
// atom (e4) parked at 19 Angstrom -- outside the 5 Angstrom outer_cutoff, so
// it starts INACTIVE. With ceiling maxNeighborsPerLigand==3, this fixture's
// initial configuration is the exact same safe boundary as
// oneAnchorThreeEnvFixture (e4 does not count). Moving e4 alone via
// Context::setPositions() (fixed particle count, no rebuild-from-System
// needed) can then genuinely drive the true neighbor count to 4 (over the
// ceiling) and back to 3 (safe) on ONE live Context -- this is what makes a
// real "trigger overflow, then recover" test possible without adding
// particles mid-Context.
AtomSystemView oneAnchorFourEnvFixture() {
    AtomSystemView fx;
    fx.nAtoms = 5; fx.nLigand = 1;
    fx.positionsNm = {
        {0.0, 0.0, 0.0},   // anchor0
        {0.1, 0.0, 0.0},   // e1 (1 Angstrom)
        {0.2, 0.0, 0.0},   // e2 (2 Angstrom)
        {0.4, 0.0, 0.0},   // e3 (4 Angstrom)
        {1.9, 0.0, 0.0},   // e4, PARKED outside cutoff (19 Angstrom) -- inactive initially
    };
    fx.boxNm = {{{20.0, 0.0, 0.0}, {0.0, 20.0, 0.0}, {0.0, 0.0, 20.0}}};
    fx.atomTypeIndex.assign(fx.nAtoms, 0);
    fx.ligandTopologyIds = {0};
    return fx;
}

// Runs one full construct-and-evaluate cycle, returning whether it threw
// and (if it didn't) the resulting energy/forces. Each call builds a BRAND
// NEW System/Force/Context -- exactly the same pattern every existing G2/G3
// fail-closed test uses (a Context is not exercised further after an
// exception in this test suite's philosophy; recovery-after-failure on a
// single live Context is tested separately and explicitly below, with its
// own reasoning for why that specific case is safe to probe).
struct EvalResult { bool threw = false; string what; double energy = 0.0; vector<Vec3> forces; };

EvalResult evaluateOnce(const LoadedPayload& loaded, const AtomSystemView& fx,
                         int maxEdges, int maxNeighborsPerLigand, int maxEnvironmentAtoms) {
    EvalResult r;
    System system;
    LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx, maxEdges, maxNeighborsPerLigand, maxEnvironmentAtoms);
    Context* context = nullptr;
    try {
        context = buildContext(fx, force, "CUDA", system);
        State state = context->getState(State::Energy | State::Forces);
        r.energy = state.getPotentialEnergy();
        r.forces = state.getForces();
    } catch (const exception& e) {
        r.threw = true;
        r.what = e.what();
    }
    delete context;
    return r;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "usage: exp026_control_plane_correctness_test <r1_model_payload_v1.json> <r1_model_weights_f64.bin> <g1_reference_dir>\n";
        return 2;
    }
    string payloadJsonPath = argv[1];
    string weightsBinPath = argv[2];
    // g1_reference_dir (argv[3]) is accepted for CLI-shape consistency with
    // the other EXP-025/026 test binaries but unused here -- this test never
    // needs the 73536-atom canonical fixture, only the model payload/weights.

    try {
#ifdef OPENMM_CONDA_PLUGIN_DIR
        Platform::loadPluginsFromDirectory(OPENMM_CONDA_PLUGIN_DIR);
#endif
#ifdef PLUGIN_DIR
        Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidual.so");
        Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidualReference.so");
        Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidualCUDA.so");
#endif
        LoadedPayload loaded = loadModelPayload(payloadJsonPath, weightsBinPath);
        AtomSystemView twoAnchor = twoAnchorFiveEnvFixture();
        AtomSystemView oneAnchor = oneAnchorThreeEnvFixture();
        AtomSystemView oneAnchorFourEnv = oneAnchorFourEnvFixture();

        cout << "\n=== EXP026-A: unique-environment ceiling, exact boundary (PLAN 9.2 items 1/4/5) ===\n";
        {
            // true union = 5; edges/neighbors ceilings set generously large
            // so only the unique-env ceiling can possibly fire.
            EvalResult atK = evaluateOnce(loaded, twoAnchor, /*maxEdges=*/20, /*maxNeighbors=*/10, /*maxEnv=*/5);
            expectTrue("unique-env==5 with ceiling=5 PASSES (exact boundary)", !atK.threw,
                       atK.threw ? atK.what : "energy=" + to_string(atK.energy));
            EvalResult overK = evaluateOnce(loaded, twoAnchor, /*maxEdges=*/20, /*maxNeighbors=*/10, /*maxEnv=*/4);
            expectTrue("unique-env==5 with ceiling=4 FAILS CLOSED (+1 over)", overK.threw, overK.what);
            if (overK.threw)
                expectTrue("failure message names the unique-env ceiling specifically",
                           overK.what.find("unique environment") != string::npos, overK.what);
        }

        cout << "\n=== EXP026-B: active-edge ceiling, exact boundary (PLAN 9.2 items 1/4/5) ===\n";
        {
            // true total edges = 6; neighbor/unique-env ceilings set
            // generously large so only the edge ceiling can possibly fire.
            EvalResult atK = evaluateOnce(loaded, twoAnchor, /*maxEdges=*/6, /*maxNeighbors=*/10, /*maxEnv=*/10);
            expectTrue("active-edges==6 with ceiling=6 PASSES (exact boundary)", !atK.threw,
                       atK.threw ? atK.what : "energy=" + to_string(atK.energy));
            EvalResult overK = evaluateOnce(loaded, twoAnchor, /*maxEdges=*/5, /*maxNeighbors=*/10, /*maxEnv=*/10);
            expectTrue("active-edges==6 with ceiling=5 FAILS CLOSED (+1 over)", overK.threw, overK.what);
            if (overK.threw)
                expectTrue("failure message names the edge ceiling specifically",
                           overK.what.find("max_edges") != string::npos, overK.what);
        }

        cout << "\n=== EXP026-C: per-ligand neighbor ceiling, exact boundary, isolated from edge/unique-env (PLAN 9.2 items 1/4/5) ===\n";
        {
            // single anchor, true neighbor count = 3; edge/unique-env
            // ceilings set generously large.
            EvalResult atK = evaluateOnce(loaded, oneAnchor, /*maxEdges=*/20, /*maxNeighbors=*/3, /*maxEnv=*/20);
            expectTrue("neighbors==3 with ceiling=3 PASSES (exact boundary)", !atK.threw,
                       atK.threw ? atK.what : "energy=" + to_string(atK.energy));
            EvalResult overK = evaluateOnce(loaded, oneAnchor, /*maxEdges=*/20, /*maxNeighbors=*/2, /*maxEnv=*/20);
            expectTrue("neighbors==3 with ceiling=2 FAILS CLOSED (+1 over)", overK.threw, overK.what);
            if (overK.threw)
                expectTrue("failure message names the neighbor ceiling specifically",
                           overK.what.find("max_neighbors_per_ligand") != string::npos, overK.what);
        }

        cout << "\n=== EXP026-D: repeated evaluation on the SAME live Context does not leak counts across the epoch reset (PLAN 9.2 item 2) ===\n";
        {
            // Boundary-exact-pass config (unique-env==5, ceiling==5): if the
            // epoch tag/reset were broken (e.g. the epoch failed to advance,
            // or advanced but the reset kernel didn't actually re-arm
            // per-evaluation state), a stale count from a PRIOR evaluation
            // would either falsely overflow this passing config on some
            // later call, or (silently) undercount forever. Same positions
            // every call, so energy must also be bit-stable, not just
            // non-throwing.
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, twoAnchor, 20, 10, 5);
            Context* context = buildContext(twoAnchor, force, "CUDA", system);
            const int REPEATS = 25;
            bool allOk = true;
            double firstEnergy = 0.0;
            for (int i = 0; i < REPEATS; i++) {
                try {
                    State state = context->getState(State::Energy | State::Forces);
                    double e = state.getPotentialEnergy();
                    if (i == 0) firstEnergy = e;
                    else if (fabs(e - firstEnergy) > 1e-9) { allOk = false; break; }
                } catch (const exception&) { allOk = false; break; }
            }
            expectTrue("25 repeated evaluations on one Context all pass with identical energy", allOk,
                       "first energy=" + to_string(firstEnergy));
            delete context;
        }

        cout << "\n=== EXP026-E: repeated fresh-Context evaluation of the SAME overflow config fails EVERY time (PLAN 9.2 item 2, GPU-nondeterminism guard) ===\n";
        {
            // Rebuilt from scratch each iteration (this suite's normal
            // post-exception discipline -- see the block comment on
            // evaluateOnce()). A flaky "sometimes passes" result here would
            // indicate a race in the first-error-wins CAS or the reset
            // kernel, not just an ordinary overflow bug.
            const int REPEATS = 10;
            int failures = 0;
            for (int i = 0; i < REPEATS; i++) {
                EvalResult r = evaluateOnce(loaded, twoAnchor, 20, 10, 4);  // unique-env 5 > ceiling 4, every time
                if (!r.threw) failures++;
            }
            expectTrue("unique-env overflow fails closed on all " + to_string(REPEATS) + "/" + to_string(REPEATS) + " fresh attempts",
                       failures == 0, to_string(REPEATS - failures) + "/" + to_string(REPEATS) + " threw");
        }

        cout << "\n=== EXP026-F: long repeated evaluation stays finite and does not hang/leak (PLAN 9.2 item 20, reduced from 10,000-step production frame for wall-clock budget) ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, twoAnchor, 20, 10, 5);
            Context* context = buildContext(twoAnchor, force, "CUDA", system);
            const int REPEATS = 10000;
            bool allFinite = true;
            double firstEnergy = 0.0;
            for (int i = 0; i < REPEATS; i++) {
                State state = context->getState(State::Energy);
                double e = state.getPotentialEnergy();
                if (i == 0) firstEnergy = e;
                if (!isfinite(e) || fabs(e - firstEnergy) > 1e-9) { allFinite = false; break; }
            }
            expectTrue(to_string(REPEATS) + " consecutive evaluations stay finite and stable, no hang", allFinite,
                       "first energy=" + to_string(firstEnergy));
            delete context;
        }

        cout << "\n=== EXP026-G: a fail-closed exception on one evaluation does not corrupt a LATER valid evaluation on the SAME live Context (PLAN 9.2 item 18) ===\n";
        {
            // Independent Reference-platform energy/forces for the fixture's
            // initial (safe, e4 parked outside cutoff) configuration -- the
            // ground truth the post-overflow CUDA evaluation must still
            // match once e4 is moved back out of range.
            double refEnergy;
            vector<Vec3> refForces;
            {
                System refSystem;
                LocalManyBodyResidualForce* refForce = buildForceFromPayload(loaded, oneAnchorFourEnv, 20, 3, 20);
                Context* refContext = buildContext(oneAnchorFourEnv, refForce, "Reference", refSystem);
                State refState = refContext->getState(State::Energy | State::Forces);
                refEnergy = refState.getPotentialEnergy();
                refForces = refState.getForces();
                delete refContext;
            }

            System system;
            // ceiling == 3 == the true initial active-neighbor count (e1,e2,e3; e4 parked outside).
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, oneAnchorFourEnv, 20, 3, 20);
            Context* context = buildContext(oneAnchorFourEnv, force, "CUDA", system);

            // Step 1: safe (3 active neighbors == ceiling, e4 inactive) -- must pass.
            bool step1Ok = true;
            try { context->getState(State::Energy); } catch (const exception&) { step1Ok = false; }
            expectTrue("step 1 (3 active == ceiling, e4 parked outside) passes", step1Ok);

            // Step 2: move e4 (atom index 4) from 19 Angstrom to 3.5
            // Angstrom -- e1/e2/e3 stay at their ORIGINAL positions (0.1/0.2/
            // 0.4 nm = 1/2/4 Angstrom; these are nm, not Angstrom -- writing
            // 0.01/0.02/0.04 here would silently redo all three at 1/10th
            // the intended distance and instead trip MIN_DISTANCE, not the
            // neighbor ceiling this step exists to test, which is exactly a
            // mistake an earlier draft of this test made). Now genuinely a
            // 4th active neighbor (e1,e2,e3,e4), 4 > ceiling 3 -- MUST fail
            // closed with NEIGHBOR_OVERFLOW specifically.
            vector<Vec3> withE4Active = {
                Vec3(0.0, 0.0, 0.0), Vec3(0.1, 0.0, 0.0), Vec3(0.2, 0.0, 0.0),
                Vec3(0.4, 0.0, 0.0), Vec3(0.35, 0.0, 0.0)};
            context->setPositions(withE4Active);
            bool step2Threw = false;
            string step2What;
            try { context->getState(State::Energy); }
            catch (const exception& e) { step2Threw = true; step2What = e.what(); }
            expectTrue("step 2 (e4 moved into range, 4 > ceiling 3) fails closed", step2Threw, step2What);
            expectTrue("step 2 failure is specifically NEIGHBOR_OVERFLOW (not some other fail-closed path)",
                       step2Threw && step2What.find("max_neighbors_per_ligand") != string::npos, step2What);

            // Step 3: move e4 back OUTSIDE the cutoff -- back to exactly the
            // fixture's original (safe, Reference-matched) configuration.
            // If step 2's exception left any stale/partial device state
            // (leftover status/epoch/force-buffer contamination), this
            // evaluation would either throw again (it must not) or return
            // forces/energy that no longer match the independent Reference
            // computed above (they must match, within the same G2/G3 tolerance).
            vector<Vec3> restored(5);
            for (int i = 0; i < 5; i++)
                restored[i] = Vec3(oneAnchorFourEnv.positionsNm[i][0], oneAnchorFourEnv.positionsNm[i][1], oneAnchorFourEnv.positionsNm[i][2]);
            context->setPositions(restored);
            bool step3Ok = true;
            State finalState;
            try { finalState = context->getState(State::Energy | State::Forces); }
            catch (const exception&) { step3Ok = false; }
            expectTrue("step 3 (e4 moved back out of range) does not throw after step 2's failure", step3Ok);
            if (step3Ok) {
                expectClose("post-overflow-recovery CUDA energy matches independent Reference", finalState.getPotentialEnergy(), refEnergy, ENERGY_ABS_TOL);
                expectClose("post-overflow-recovery CUDA forces match independent Reference", maxForceDiff(finalState.getForces(), refForces), 0.0, FORCE_ABS_TOL);
            }
            delete context;
        }

        cout << "\n=== EXP-026 CONTROL-PLANE CORRECTNESS TEST: " << (g_failures == 0 ? "PASS" : "FAIL")
             << " (" << g_failures << " failing checks) ===\n";
        return g_failures == 0 ? 0 : 1;
    } catch (const exception& e) {
        cerr << "EXP-026 CONTROL-PLANE TEST FAIL-CLOSED: " << e.what() << "\n";
        return 1;
    }
}
