// EXP-025 G1-G: real LocalManyBodyResidualForce running in an actual OpenMM
// Reference Context must reproduce the same energy/force as the
// already-validated standalone oracle (g1_reference_oracle.cpp) -- AND must
// honor the frozen kBT/exactly-once contract: this Force outputs the full
// physical U_B = kBT*B (kJ/mol) and F = -kBT*gradient_nm itself; the outer
// wrapper is not involved at all in this test.
//
// kBT = MOLAR_GAS_CONSTANT_R * 300 K, matching openmm.unit exactly (see
// LocalManyBodyResidualForce::getMolarGasConstantRKilojoulePerMoleKelvin()).
// The canonical reference values below were cross-checked against an
// independent external computation the user supplied: kBT * B_reduced from
// canonical_fixture_expected_v1.json reproduces -16.190576210439794 kJ/mol
// to 15+ significant figures (2026-08-12) -- i.e. B_reduced in that JSON
// (-6.490929101094429) is the correct value, not a nearby-but-different
// number that was also floated during development.
#include "openmm/LocalManyBodyResidualForce.h"
#include "OpenMM.h"
#include "openmm/serialization/XmlSerializer.h"
#include "g1_math_core.h"
#include "g1_payload_io.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace OpenMM;
using namespace exp025_g1;
using namespace std;

namespace {

int g_failures = 0;

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

const double TEMPERATURE_KELVIN = 300.0;
const int FD_ATOM = 4583;

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
    force->setCapacityCeilings((int)model.maxEdges, (int)model.maxNeighborsPerLigand, (int)model.maxEnvironmentAtoms);
    for (int t = 0; t < model.typeCount; t++) {
        const TypedMLP& src = model.rho[t];
        LocalManyBodyTypedMLP mlp;
        mlp.w0.assign(src.W0.begin(), src.W0.end());
        mlp.b0.assign(src.b0.begin(), src.b0.end());
        mlp.w2.resize(256);
        for (int o = 0; o < 16; o++)
            for (int k = 0; k < 16; k++) mlp.w2[(size_t)o * 16 + k] = src.W2[o][k];
        mlp.b2.assign(src.b2.begin(), src.b2.end());
        mlp.w4.assign(src.W4.begin(), src.W4.end());
        mlp.b4 = src.b4;
        force->setTypedMLP(t, mlp);
    }
    return force;
}

struct BuiltSystem {
    System system;
    LocalManyBodyResidualForce* force;  // owned by system after addForce
};

// atomTypeIndexOverride lets a caller substitute the fixture's own type
// array with something else (unused here, kept for symmetry with the
// production API's expectation that atomTypeIndex is System-particle-count
// sized).
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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "usage: g1g_openmm_reference_parity_test <r1_model_payload_v1.json> <r1_model_weights_f64.bin> <g1_reference_dir>\n";
        return 2;
    }
    string payloadJsonPath = argv[1];
    string weightsBinPath = argv[2];
    string g1Dir = argv[3];

    try {
#ifdef OPENMM_CONDA_PLUGIN_DIR
        Platform::loadPluginsFromDirectory(OPENMM_CONDA_PLUGIN_DIR);
#endif
#ifdef PLUGIN_DIR
        Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidual.so");
        Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidualReference.so");
#endif
        LoadedPayload loaded = loadModelPayload(payloadJsonPath, weightsBinPath);
        AtomSystemView fx = loadFixture(g1Dir + "/canonical_fixture_v1.bin");

        yyjson_doc* expectedDoc = yyjson_read_file((g1Dir + "/canonical_fixture_expected_v1.json").c_str(), 0, nullptr, nullptr);
        if (!expectedDoc) throw MathError("failed to parse canonical_fixture_expected_v1.json");
        yyjson_val* expectedRoot = yyjson_doc_get_root(expectedDoc);
        double expectedB = jNum(expectedRoot, "B_reduced");
        yyjson_val* gradSection = jObj(expectedRoot, "reduced_gradient_dB_dx_nm");
        yyjson_val* ligandGradArr = jObj(gradSection, "ligand_by_local_index");
        yyjson_val* envGradObj = jObj(gradSection, "environment_by_topology_id");

        double kBT = LocalManyBodyResidualForce::getMolarGasConstantRKilojoulePerMoleKelvin() * TEMPERATURE_KELVIN;
        double expectedU_B = kBT * expectedB;
        cout << "kBT (300K) = " << kBT << " kJ/mol\n";
        cout << "expected U_B = kBT * B_reduced = " << expectedU_B << " kJ/mol\n";
        expectClose("kBT matches independently-verified openmm.unit.MOLAR_GAS_CONSTANT_R*300", kBT, 2.494338785445972, 1e-12);
        expectClose("expected U_B matches externally-supplied reference value", expectedU_B, -16.190576210439794, 1e-9);

        vector<int> sortedLigand(fx.ligandTopologyIds.begin(), fx.ligandTopologyIds.end());
        sort(sortedLigand.begin(), sortedLigand.end());

        cout << "\n=== G1-G basic: energy + forces in a real Reference Context ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            Context* context = buildContext(fx, force, "Reference", system);
            State state = context->getState(State::Energy | State::Forces);
            double energy = state.getPotentialEnergy();
            expectClose("U_B (potential energy)", energy, expectedU_B, 1e-6);

            vector<Vec3> forces = state.getForces();
            double maxLigDiff = 0.0;
            for (int i = 0; i < (int)sortedLigand.size(); i++) {
                yyjson_val* triple = yyjson_arr_get(ligandGradArr, i);
                int atom = sortedLigand[i];
                for (int c = 0; c < 3; c++) {
                    double gradNm = yyjson_get_num(yyjson_arr_get(triple, c));
                    double expectedForce = -kBT * gradNm;
                    double actualForce = forces[atom][c];
                    maxLigDiff = max(maxLigDiff, fabs(actualForce - expectedForce));
                }
            }
            expectClose("max|ligand F - (-kBT*grad)| ", maxLigDiff, 0.0, 1e-6);

            yyjson_val* key;
            yyjson_obj_iter iter = yyjson_obj_iter_with(envGradObj);
            double maxEnvDiff = 0.0;
            while ((key = yyjson_obj_iter_next(&iter))) {
                int atom = atoi(yyjson_get_str(key));
                yyjson_val* triple = yyjson_obj_iter_get_val(key);
                for (int c = 0; c < 3; c++) {
                    double gradNm = yyjson_get_num(yyjson_arr_get(triple, c));
                    double expectedForce = -kBT * gradNm;
                    double actualForce = forces[atom][c];
                    maxEnvDiff = max(maxEnvDiff, fabs(actualForce - expectedForce));
                }
            }
            expectClose("max|environment F - (-kBT*grad)|", maxEnvDiff, 0.0, 1e-6);

            double sumFx = 0, sumFy = 0, sumFz = 0;
            for (const Vec3& f : forces) { sumFx += f[0]; sumFy += f[1]; sumFz += f[2]; }
            double netForceNorm = sqrt(sumFx * sumFx + sumFy * sumFy + sumFz * sumFz);
            expectClose("net force over all atoms (Newton's third law)", netForceNorm, 0.0, 1e-6);

            delete context;
        }

        cout << "\n=== G1-G: energy-only vs force-only vs both ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            Context* context = buildContext(fx, force, "Reference", system);
            State energyOnly = context->getState(State::Energy);
            State forcesOnly = context->getState(State::Forces);
            State both = context->getState(State::Energy | State::Forces);
            expectClose("energy-only == both.energy", energyOnly.getPotentialEnergy(), both.getPotentialEnergy(), 1e-12);
            double maxDiff = 0.0;
            vector<Vec3> f1 = forcesOnly.getForces(), f2 = both.getForces();
            for (size_t i = 0; i < f1.size(); i++)
                for (int c = 0; c < 3; c++) maxDiff = max(maxDiff, fabs(f1[i][c] - f2[i][c]));
            expectTrue("forces-only == both.forces exactly", maxDiff == 0.0, "maxDiff=" + to_string(maxDiff));
            delete context;
        }

        cout << "\n=== G1-G: force group mask ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            force->setForceGroup(5);
            Context* context = buildContext(fx, force, "Reference", system);
            State inGroup = context->getState(State::Energy, false, 1 << 5);
            State outOfGroup = context->getState(State::Energy, false, 1 << 2);
            expectClose("energy visible when querying its own group", inGroup.getPotentialEnergy(), expectedU_B, 1e-6);
            expectClose("energy invisible when querying a different group", outOfGroup.getPotentialEnergy(), 0.0, 0.0);
            delete context;
        }

        cout << "\n=== G1-G: no-contact gives exactly zero (real Context, real R1 weights) ===\n";
        {
            // The dense 73536-atom fixture is periodic and fully packed
            // (protein+membrane+water) -- there is no "empty pocket" to move
            // the ligand into within it without colliding with some other
            // part of the system (a first attempt at this hit max_edges
            // fail-closed for exactly that reason). Use a small synthetic
            // 2-atom system instead, carrying over the REAL R1 weights
            // (cutoffs/RBF/pair_weight/typed MLPs/bMax/ceilings) from the
            // loaded payload -- only the topology (1 ligand, 1 far-away
            // environment atom) is synthetic, matching D2's own convention
            // of using synthetic systems for edge-case checks.
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            force->setLigandTopologyIds({0});
            force->setAtomTypeIndex({0, 0});
            AtomSystemView isolated;
            isolated.nAtoms = 2;
            isolated.nLigand = 1;
            isolated.positionsNm = {{0.0, 0.0, 0.0}, {1.2, 0.0, 0.0}};  // 12 Angstrom apart, no half-box tie
            isolated.boxNm = {{{3.0, 0.0, 0.0}, {0.0, 3.0, 0.0}, {0.0, 0.0, 3.0}}};
            isolated.atomTypeIndex = {0, 0};
            isolated.ligandTopologyIds = {0};
            Context* context = buildContext(isolated, force, "Reference", system);
            State state = context->getState(State::Energy | State::Forces);
            expectClose("no-contact energy", state.getPotentialEnergy(), 0.0, 0.0);
            double maxAbsForce = 0.0;
            for (const Vec3& f : state.getForces()) maxAbsForce = max({maxAbsForce, fabs(f[0]), fabs(f[1]), fabs(f[2])});
            expectClose("no-contact max|force|", maxAbsForce, 0.0, 0.0);
            delete context;
        }

        cout << "\n=== G1-G: finite difference of the REAL OpenMM potential energy ===\n";
        {
            System system;
            LocalManyBodyResidualForce* force = buildForceFromPayload(loaded, fx);
            Context* context = buildContext(fx, force, "Reference", system);
            vector<Vec3> basePositions(fx.nAtoms);
            for (int i = 0; i < fx.nAtoms; i++) basePositions[i] = Vec3(fx.positionsNm[i][0], fx.positionsNm[i][1], fx.positionsNm[i][2]);
            State baseState = context->getState(State::Forces);
            Vec3 analyticForce = baseState.getForces()[FD_ATOM];
            double h = 1e-6;  // nm
            for (int c = 0; c < 3; c++) {
                vector<Vec3> plus = basePositions, minus = basePositions;
                plus[FD_ATOM][c] += h;
                minus[FD_ATOM][c] -= h;
                context->setPositions(plus);
                double ePlus = context->getState(State::Energy).getPotentialEnergy();
                context->setPositions(minus);
                double eMinus = context->getState(State::Energy).getPotentialEnergy();
                double central = -(ePlus - eMinus) / (2.0 * h);  // F = -dU/dx
                double analytic = analyticForce[c];
                double denom = max({fabs(central), fabs(analytic), 1e-8});
                double relError = fabs(central - analytic) / denom;
                expectClose("FD(real OpenMM U_B) atom " + to_string(FD_ATOM) + " coord " + to_string(c) + " relative error",
                            relError, 0.0, 1e-3);
            }
            context->setPositions(basePositions);
            delete context;
        }

        cout << "\n=== G1-G: XML round-trip with the REAL 3,048-double R1 payload ===\n";
        {
            LocalManyBodyResidualForce* original = buildForceFromPayload(loaded, fx);
            original->setForceGroup(2);
            original->setName("R1_canonical_run1_seed0");
            ostringstream xmlOut;
            XmlSerializer::serialize<LocalManyBodyResidualForce>(original, "Force", xmlOut);
            string xml = xmlOut.str();
            cout << "  serialized XML size: " << xml.size() << " bytes\n";

            istringstream xmlIn(xml);
            LocalManyBodyResidualForce* deserialized = XmlSerializer::deserialize<LocalManyBodyResidualForce>(xmlIn);
            expectTrue("forceGroup preserved", deserialized->getForceGroup() == 2);
            expectTrue("name preserved", deserialized->getName() == "R1_canonical_run1_seed0");
            expectTrue("temperatureKelvin preserved exactly", deserialized->getTemperatureKelvin() == original->getTemperatureKelvin());
            expectTrue("ligandTopologyIds preserved exactly", deserialized->getLigandTopologyIds() == original->getLigandTopologyIds());
            expectTrue("pairWeight preserved exactly (784 doubles)", deserialized->getPairWeight() == original->getPairWeight());

            // The deserialized object must be self-contained: it must not
            // depend on the original .bin path at all. Build a completely
            // fresh System/Context from it and confirm the same U_B/forces.
            System system;
            for (int i = 0; i < fx.nAtoms; i++) system.addParticle(1.0);
            system.setDefaultPeriodicBoxVectors(
                Vec3(fx.boxNm[0][0], fx.boxNm[0][1], fx.boxNm[0][2]),
                Vec3(fx.boxNm[1][0], fx.boxNm[1][1], fx.boxNm[1][2]),
                Vec3(fx.boxNm[2][0], fx.boxNm[2][1], fx.boxNm[2][2]));
            system.addForce(deserialized);
            VerletIntegrator integrator(0.001);
            Platform& platform = Platform::getPlatformByName("Reference");
            Context context(system, integrator, platform);
            vector<Vec3> positions(fx.nAtoms);
            for (int i = 0; i < fx.nAtoms; i++) positions[i] = Vec3(fx.positionsNm[i][0], fx.positionsNm[i][1], fx.positionsNm[i][2]);
            context.setPositions(positions);
            State state = context.getState(State::Energy | State::Forces, false, 1 << 2);
            expectClose("U_B after XML round-trip, brand new Context", state.getPotentialEnergy(), expectedU_B, 1e-6);
            double maxLigDiff2 = 0.0;
            vector<Vec3> forces2 = state.getForces();
            for (int i = 0; i < (int)sortedLigand.size(); i++) {
                yyjson_val* triple = yyjson_arr_get(ligandGradArr, i);
                int atom = sortedLigand[i];
                for (int c = 0; c < 3; c++) {
                    double expectedForce = -kBT * yyjson_get_num(yyjson_arr_get(triple, c));
                    maxLigDiff2 = max(maxLigDiff2, fabs(forces2[atom][c] - expectedForce));
                }
            }
            expectClose("max|ligand F diff| after XML round-trip", maxLigDiff2, 0.0, 1e-6);
            delete original;
        }

        cout << "\n=== G1-G OPENMM REFERENCE PARITY TEST: " << (g_failures == 0 ? "PASS" : "FAIL") << " (" << g_failures << " failing checks) ===\n";
        yyjson_doc_free(expectedDoc);
        return g_failures == 0 ? 0 : 1;
    } catch (const exception& e) {
        cerr << "G1-G TEST FAIL-CLOSED: " << e.what() << "\n";
        return 1;
    }
}
