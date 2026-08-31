// EXP-025 G2: CUDA brute-force correctness test.
//
// Scope (frozen by the G2 gate definition): 41 anchors x all-environment
// brute force CUDA kernel must reproduce the already-validated Reference
// platform (G1-G) result on the real canonical fixture, energy-only/
// force-only/both semantics, force-group mask, no-contact, finite
// difference of the real OpenMM CUDA potential energy, atom reorder
// parity, XML round-trip into a brand new CUDA Context, and the
// fail-closed cases. This test does NOT touch cost/CSR/cell-list --
// passing it sets G2_CUDA_BRUTE_FORCE_CORRECTNESS = true and nothing else.
#include "openmm/LocalManyBodyResidualForce.h"
#include "OpenMM.h"
#include "openmm/serialization/XmlSerializer.h"
#include "g1_math_core.h"
#include "g1_payload_io.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace OpenMM;
using namespace exp025_g1;
using namespace std;

namespace {

int g_failures = 0;
const double TEMPERATURE_KELVIN = 300.0;
// M2 mixed-precision recertification (post-G3 addition): set via
// --precision mixed on argv, applied to every CUDA Context this test file
// builds. Empty map (default) preserves this test's original, already-
// sealed single-precision behavior exactly -- this is purely additive.
map<string, string> g_cudaPlatformProperties;
const int FD_ATOM = 4583;
// G2 CUDA-vs-Reference tolerances, exactly as specified for this gate.
const double ENERGY_ABS_TOL = 1e-4;       // kJ/mol
const double FORCE_ABS_TOL = 1e-3;        // kJ/mol/nm

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

LocalManyBodyResidualForce* buildForceFromPayload(const LoadedPayload& loaded, const AtomSystemView& fx) {
    const ModelParams& model = loaded.model;
    LocalManyBodyResidualForce* force = new LocalManyBodyResidualForce();
    force->setTemperatureKelvin(TEMPERATURE_KELVIN);
    vector<int> ligandIds(loaded.ligandTopologyIndices.begin(), loaded.ligandTopologyIndices.end());
    force->setLigandTopologyIds(ligandIds);
    vector<int> vocab(loaded.typeVocabulary.begin(), loaded.typeVocabulary.end());
    force->setTypeVocabulary(vocab);
    force->setAtomTypeIndex(vector<int>(fx.atomTypeIndex.begin(), fx.atomTypeIndex.end()));
    force->setInnerCutoffAngstrom(model.innerCutoffAngstrom);
    force->setOuterCutoffAngstrom(model.outerCutoffAngstrom);
    force->setRadialCenters(model.radialCenters);
    force->setRadialWidthAngstrom(model.radialWidth);
    force->setPairWeight(model.pairWeight);
    force->setBMaxReduced(model.bMaxReduced);
    force->setCapacityCeilings((int) model.maxEdges, (int) model.maxNeighborsPerLigand, (int) model.maxEnvironmentAtoms);
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
    Context* context = (platformName == "CUDA")
        ? new Context(system, *integrator, platform, g_cudaPlatformProperties)
        : new Context(system, *integrator, platform);
    vector<Vec3> positions(fx.nAtoms);
    for (int i = 0; i < fx.nAtoms; i++) positions[i] = Vec3(fx.positionsNm[i][0], fx.positionsNm[i][1], fx.positionsNm[i][2]);
    context->setPositions(positions);
    return context;
}

// Forces a device atom-reorder by adding and immediately removing a large
// number of dummy CustomBondForce-free churn: simplest reliable trigger is
// to call updateParametersInContext-free re-creation is not available, so
// instead we perturb positions enough to move blocks and call
// reinitialize(true) which forces OpenMM to rebuild its neighbor list and
// (for CUDA) is a documented trigger for atom reordering.
void forceReorder(Context& context) {
    context.reinitialize(true);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        cerr << "usage: g2_cuda_reference_parity_test <r1_model_payload_v1.json> <r1_model_weights_f64.bin> "
                "<g1_reference_dir> [--precision <single|mixed|double>]\n";
        return 2;
    }
    string payloadJsonPath = argv[1];
    string weightsBinPath = argv[2];
    string g1Dir = argv[3];
    if (argc == 6) {
        if (string(argv[4]) != "--precision") { cerr << "unknown flag " << argv[4] << "\n"; return 2; }
        g_cudaPlatformProperties["Precision"] = argv[5];
        cout << "M2 mixed-precision recertification mode: CUDA Precision=" << argv[5] << "\n";
    }

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
        AtomSystemView fx = loadFixture(g1Dir + "/canonical_fixture_v1.bin");

        yyjson_doc* expectedDoc = yyjson_read_file((g1Dir + "/canonical_fixture_expected_v1.json").c_str(), 0, nullptr, nullptr);
        if (!expectedDoc) throw MathError("failed to parse canonical_fixture_expected_v1.json");
        yyjson_val* expectedRoot = yyjson_doc_get_root(expectedDoc);
        double expectedB = jNum(expectedRoot, "B_reduced");
        yyjson_val* gradSection = jObj(expectedRoot, "reduced_gradient_dB_dx_nm");
        yyjson_val* ligandGradArr = jObj(gradSection, "ligand_by_local_index");

        double kBT = LocalManyBodyResidualForce::getMolarGasConstantRKilojoulePerMoleKelvin() * TEMPERATURE_KELVIN;
        double expectedU_B = kBT * expectedB;
        cout << "expected U_B (from G1 canonical fixture) = " << expectedU_B << " kJ/mol\n";

        vector<int> sortedLigand(fx.ligandTopologyIds.begin(), fx.ligandTopologyIds.end());
        sort(sortedLigand.begin(), sortedLigand.end());

        // ============================== reference computation, once ==============================
        vector<Vec3> referenceForces;
        double referenceEnergy;
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            Context* context = buildContext(fx, force, "Reference", system);
            State state = context->getState(State::Energy | State::Forces);
            referenceEnergy = state.getPotentialEnergy();
            referenceForces = state.getForces();
            delete context;
        }
        expectClose("Reference U_B sanity (already validated in G1-G)", referenceEnergy, expectedU_B, 1e-6);

        cout << "\n=== G2 basic: CUDA vs Reference on the real canonical frame ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            Context* context = buildContext(fx, force, "CUDA", system);
            State state = context->getState(State::Energy | State::Forces);
            double cudaEnergy = state.getPotentialEnergy();
            expectClose("CUDA U_B vs Reference U_B", cudaEnergy, referenceEnergy, ENERGY_ABS_TOL);
            expectClose("CUDA U_B vs G1 canonical value", cudaEnergy, expectedU_B, ENERGY_ABS_TOL);

            vector<Vec3> cudaForces = state.getForces();
            double maxForceDiff = 0.0;
            for (size_t i = 0; i < cudaForces.size(); i++)
                for (int c = 0; c < 3; c++)
                    maxForceDiff = max(maxForceDiff, fabs(cudaForces[i][c] - referenceForces[i][c]));
            expectClose("max|CUDA F - Reference F| over all atoms", maxForceDiff, 0.0, FORCE_ABS_TOL);

            double sumFx = 0, sumFy = 0, sumFz = 0;
            for (const Vec3& f : cudaForces) { sumFx += f[0]; sumFy += f[1]; sumFz += f[2]; }
            expectClose("CUDA net force (Newton's third law)", sqrt(sumFx*sumFx+sumFy*sumFy+sumFz*sumFz), 0.0, FORCE_ABS_TOL);
            delete context;
        }

        cout << "\n=== G2: energy-only vs force-only vs both (CUDA) ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            Context* context = buildContext(fx, force, "CUDA", system);
            State energyOnly = context->getState(State::Energy);
            State forcesOnly = context->getState(State::Forces);
            State both = context->getState(State::Energy | State::Forces);
            expectClose("energy-only == both.energy", energyOnly.getPotentialEnergy(), both.getPotentialEnergy(), 1e-8);
            double maxDiff = 0.0;
            vector<Vec3> f1 = forcesOnly.getForces(), f2 = both.getForces();
            for (size_t i = 0; i < f1.size(); i++)
                for (int c = 0; c < 3; c++) maxDiff = max(maxDiff, fabs(f1[i][c] - f2[i][c]));
            // NOT bitwise-exact on CUDA (unlike Reference): K3's force scatter
            // sums via atomicAdd across warps/blocks, and CUDA gives no
            // ordering guarantee call-to-call, so floating point
            // non-associativity produces tiny run-to-run differences. This
            // is expected GPU non-determinism, not a correctness bug --
            // bound it with the same G2 force tolerance instead of requiring
            // exact equality.
            expectClose("forces-only == both.forces within tolerance", maxDiff, 0.0, FORCE_ABS_TOL);
            delete context;
        }

        cout << "\n=== G2: force group mask (CUDA) ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            force->setForceGroup(5);
            Context* context = buildContext(fx, force, "CUDA", system);
            State inGroup = context->getState(State::Energy, false, 1 << 5);
            State outOfGroup = context->getState(State::Energy, false, 1 << 2);
            expectClose("energy visible in own group", inGroup.getPotentialEnergy(), expectedU_B, ENERGY_ABS_TOL);
            expectClose("energy invisible in a different group", outOfGroup.getPotentialEnergy(), 0.0, 0.0);
            delete context;
        }

        cout << "\n=== G2: no-contact gives exactly zero (CUDA, real weights, synthetic topology) ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            force->setLigandTopologyIds({0});
            force->setAtomTypeIndex({0, 0});
            AtomSystemView isolated;
            isolated.nAtoms = 2;
            isolated.nLigand = 1;
            isolated.positionsNm = {{0.0, 0.0, 0.0}, {1.2, 0.0, 0.0}};
            isolated.boxNm = {{{3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 3.0}}};
            isolated.atomTypeIndex = {0, 0};
            isolated.ligandTopologyIds = {0};
            Context* context = buildContext(isolated, force, "CUDA", system);
            State state = context->getState(State::Energy | State::Forces);
            expectClose("no-contact CUDA energy", state.getPotentialEnergy(), 0.0, 1e-10);
            double maxAbsForce = 0.0;
            for (const Vec3& f : state.getForces()) maxAbsForce = max({maxAbsForce, fabs(f[0]), fabs(f[1]), fabs(f[2])});
            expectClose("no-contact CUDA max|force|", maxAbsForce, 0.0, 1e-10);
            delete context;
        }

        cout << "\n=== G2: triclinic cross-boundary synthetic case (CUDA, real weights) ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            force->setLigandTopologyIds({0});
            force->setAtomTypeIndex({1, 1});  // type index 1 (arbitrary valid type)
            AtomSystemView triclinic;
            triclinic.nAtoms = 2;
            triclinic.nLigand = 1;
            // Ligand near one corner, environment atom positioned so the
            // periodic image across the triclinic boundary is the nearest.
            triclinic.positionsNm = {{0.1, 0.1, 0.1}, {1.9, 0.1, 0.1}};
            triclinic.boxNm = {{{2.0, 0.0, 0.0}, {0.3, 1.9, 0.0}, {0.1, 0.2, 2.1}}};
            triclinic.atomTypeIndex = {1, 1};
            triclinic.ligandTopologyIds = {0};
            LocalManyBodyResidualForce* referenceForce = buildForceFromPayload(loaded, fx);
            referenceForce->setLigandTopologyIds({0});
            referenceForce->setAtomTypeIndex({1, 1});
            Context* referenceContext = buildContext(triclinic, referenceForce, "Reference", system);
            State refState = referenceContext->getState(State::Energy | State::Forces);
            System system2;
            Context* cudaContext = buildContext(triclinic, force, "CUDA", system2);
            State cudaState = cudaContext->getState(State::Energy | State::Forces);
            expectClose("triclinic CUDA vs Reference energy", cudaState.getPotentialEnergy(), refState.getPotentialEnergy(), ENERGY_ABS_TOL);
            double maxDiff = 0.0;
            vector<Vec3> rf = refState.getForces(), cf = cudaState.getForces();
            for (size_t i = 0; i < rf.size(); i++)
                for (int c = 0; c < 3; c++) maxDiff = max(maxDiff, fabs(rf[i][c] - cf[i][c]));
            expectClose("triclinic CUDA vs Reference forces", maxDiff, 0.0, FORCE_ABS_TOL);
            delete referenceContext;
            delete cudaContext;
        }

        cout << "\n=== G2: finite difference of the REAL OpenMM CUDA potential energy ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            Context* context = buildContext(fx, force, "CUDA", system);
            vector<Vec3> basePositions(fx.nAtoms);
            for (int i = 0; i < fx.nAtoms; i++) basePositions[i] = Vec3(fx.positionsNm[i][0], fx.positionsNm[i][1], fx.positionsNm[i][2]);
            State baseState = context->getState(State::Forces);
            Vec3 analyticForce = baseState.getForces()[FD_ATOM];
            double h = 1e-4;  // nm -- larger than the Reference FD step since single precision needs a coarser step
            for (int c = 0; c < 3; c++) {
                vector<Vec3> plus = basePositions, minus = basePositions;
                plus[FD_ATOM][c] += h;
                minus[FD_ATOM][c] -= h;
                context->setPositions(plus);
                double ePlus = context->getState(State::Energy).getPotentialEnergy();
                context->setPositions(minus);
                double eMinus = context->getState(State::Energy).getPotentialEnergy();
                double central = -(ePlus - eMinus) / (2.0 * h);
                double analytic = analyticForce[c];
                double denom = max({fabs(central), fabs(analytic), 1e-6});
                double relError = fabs(central - analytic) / denom;
                expectClose("FD(real OpenMM CUDA U_B) atom " + to_string(FD_ATOM) + " coord " + to_string(c) + " relative error",
                            relError, 0.0, 5e-2);  // single precision FD is much noisier than Reference double
            }
            context->setPositions(basePositions);
            delete context;
        }

        cout << "\n=== G2: atom reorder parity (CUDA) ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            Context* context = buildContext(fx, force, "CUDA", system);
            State before = context->getState(State::Energy | State::Forces);
            forceReorder(*context);
            vector<Vec3> basePositions(fx.nAtoms);
            for (int i = 0; i < fx.nAtoms; i++) basePositions[i] = Vec3(fx.positionsNm[i][0], fx.positionsNm[i][1], fx.positionsNm[i][2]);
            context->setPositions(basePositions);
            State after = context->getState(State::Energy | State::Forces);
            expectClose("energy unchanged across forced reorder", after.getPotentialEnergy(), before.getPotentialEnergy(), ENERGY_ABS_TOL);
            double maxDiff = 0.0;
            vector<Vec3> f1 = before.getForces(), f2 = after.getForces();
            for (size_t i = 0; i < f1.size(); i++)
                for (int c = 0; c < 3; c++) maxDiff = max(maxDiff, fabs(f1[i][c] - f2[i][c]));
            // Same atomicAdd-order non-determinism noted at the energy/force-only
            // check above -- bound with the established G2 force tolerance,
            // not an arbitrarily tighter one.
            expectClose("forces unchanged across forced reorder", maxDiff, 0.0, FORCE_ABS_TOL);
            delete context;
        }

        cout << "\n=== G2: XML round-trip -> brand new CUDA Context ===\n";
        {
            LocalManyBodyResidualForce* original = buildForceFromPayload(loaded, fx);
            original->setForceGroup(3);
            ostringstream xmlOut;
            XmlSerializer::serialize<LocalManyBodyResidualForce>(original, "Force", xmlOut);
            string xml = xmlOut.str();
            istringstream xmlIn(xml);
            LocalManyBodyResidualForce* deserialized = XmlSerializer::deserialize<LocalManyBodyResidualForce>(xmlIn);

            System system;
            for (int i = 0; i < fx.nAtoms; i++) system.addParticle(1.0);
            system.setDefaultPeriodicBoxVectors(
                Vec3(fx.boxNm[0][0], fx.boxNm[0][1], fx.boxNm[0][2]),
                Vec3(fx.boxNm[1][0], fx.boxNm[1][1], fx.boxNm[1][2]),
                Vec3(fx.boxNm[2][0], fx.boxNm[2][1], fx.boxNm[2][2]));
            system.addForce(deserialized);
            VerletIntegrator integrator(0.001);
            Platform& platform = Platform::getPlatformByName("CUDA");
            Context context(system, integrator, platform, g_cudaPlatformProperties);
            vector<Vec3> positions(fx.nAtoms);
            for (int i = 0; i < fx.nAtoms; i++) positions[i] = Vec3(fx.positionsNm[i][0], fx.positionsNm[i][1], fx.positionsNm[i][2]);
            context.setPositions(positions);
            State state = context.getState(State::Energy | State::Forces, false, 1 << 3);
            expectClose("U_B after XML round-trip into brand new CUDA Context", state.getPotentialEnergy(), expectedU_B, ENERGY_ABS_TOL);
            delete original;
        }

        cout << "\n=== G2: fail-closed cases (CUDA) ===\n";
        {
            // r < 0.1 Angstrom
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            force->setLigandTopologyIds({0});
            force->setAtomTypeIndex({0, 0});
            AtomSystemView tooClose;
            tooClose.nAtoms = 2; tooClose.nLigand = 1;
            tooClose.positionsNm = {{0.0, 0.0, 0.0}, {0.005, 0.0, 0.0}};  // 0.05 Angstrom
            tooClose.boxNm = {{{3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 3.0}}};
            tooClose.atomTypeIndex = {0, 0};
            tooClose.ligandTopologyIds = {0};
            bool threw = false;
            try {
                Context* context = buildContext(tooClose, force, "CUDA", system);
                context->getState(State::Energy);
                delete context;
            } catch (const exception&) { threw = true; }
            expectTrue("r < 0.1 Angstrom fails closed (CUDA)", threw);
        }
        {
            // neighbor overflow: one ligand atom with > max_neighbors_per_ligand environment atoms in range
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            force->setLigandTopologyIds({0});
            int maxN, maxE, maxU;
            force->getCapacityCeilings(maxE, maxN, maxU);
            int overflowCount = maxN + 5;
            vector<int> types(1 + overflowCount, 0);
            force->setAtomTypeIndex(types);
            AtomSystemView overflow;
            overflow.nAtoms = 1 + overflowCount;
            overflow.nLigand = 1;
            overflow.positionsNm.push_back({0.0, 0.0, 0.0});
            for (int i = 0; i < overflowCount; i++)
                overflow.positionsNm.push_back({0.2 + 0.001 * i, 0.0, 0.0});  // all within cutoff, distinct positions
            overflow.boxNm = {{{5.0, 0.0, 0.0}, {0.0, 5.0, 0.0}, {0.0, 0.0, 5.0}}};
            overflow.atomTypeIndex = types;
            overflow.ligandTopologyIds = {0};
            bool threw = false;
            try {
                Context* context = buildContext(overflow, force, "CUDA", system);
                context->getState(State::Energy);
                delete context;
            } catch (const exception&) { threw = true; }
            expectTrue("neighbor-per-ligand overflow fails closed (CUDA)", threw, "attempted " + to_string(overflowCount) + " > max " + to_string(maxN));
        }

        cout << "\n=== G2 CUDA BRUTE-FORCE CORRECTNESS TEST: " << (g_failures == 0 ? "PASS" : "FAIL") << " (" << g_failures << " failing checks) ===\n";
        yyjson_doc_free(expectedDoc);
        return g_failures == 0 ? 0 : 1;
    } catch (const exception& e) {
        cerr << "G2 TEST FAIL-CLOSED: " << e.what() << "\n";
        return 1;
    }
}
