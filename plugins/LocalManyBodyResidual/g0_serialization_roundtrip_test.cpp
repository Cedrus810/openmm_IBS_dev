// EXP-025 G0 serialization round-trip test -- cheap STRUCTURAL check only.
//
// schema_version bumped 1 -> 2 by G3 (LocalManyBodyResidualForceProxy.cpp):
// skinAngstrom/candidateListCapacity are new required-together fields that
// change the math/cost contract, not a cosmetic add. This test's own
// assertion is updated to match (schema_version="2") -- the underlying G0
// conclusion (the serialization MECHANISM works: Force -> XML -> brand new
// Force/Context round-trips cleanly on both platforms) is unchanged and
// still verified below; only the literal version string moved.
//
// Force -> XML string -> deserialize -> brand new Force* -> brand new
// System/Context (Reference, then CUDA). Confirms:
//   - schema_version=2 round-trips through SerializationProxy;
//   - forceGroup and name survive the round trip;
//   - the DESERIALIZED force (not the original) still gives energy=0,
//     max|force|=0 on both platforms;
//   - the CUDA kernel re-initializes/re-JITs cleanly in a Context that
//     never saw the original Force or its device state -- no reliance on
//     stale device pointers.
//
// This uses trivial (all-zero) MLP weights and a no-contact geometry, so it
// only proves the serialization MECHANISM works, not that the real
// 3,048-double R1 payload round-trips correctly -- that is a separate,
// stronger check in g1g_openmm_reference_parity_test.cpp using the actual
// frozen checkpoint weights.
#include "openmm/LocalManyBodyResidualForce.h"
#include "OpenMM.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace OpenMM;
using namespace std;

static void fillMinimalNoContactParameters(LocalManyBodyResidualForce& force) {
    force.setTemperatureKelvin(300.0);
    force.setLigandTopologyIds({0});
    force.setAtomTypeIndex({0, 0});
    force.setTypeVocabulary({6});
    force.setInnerCutoffAngstrom(4.0);
    force.setOuterCutoffAngstrom(5.0);
    force.setRadialCenters(vector<double>(16, 0.0));
    force.setRadialWidthAngstrom(1.0);
    force.setPairWeight(vector<double>(1 * 1 * 16, 0.0));
    force.setBMaxReduced(10.0);
    force.setCapacityCeilings(10, 10, 10);
    LocalManyBodyTypedMLP mlp;
    mlp.w0.assign(16, 0.0);
    mlp.b0.assign(16, 0.0);
    mlp.w2.assign(256, 0.0);
    mlp.b2.assign(16, 0.0);
    mlp.w4.assign(16, 0.0);
    mlp.b4 = 0.0;
    force.setTypedMLP(0, mlp);
}

static bool runDeserializedOnPlatform(const string& platformName, LocalManyBodyResidualForce* force) {
    System system;
    system.setDefaultPeriodicBoxVectors(Vec3(3, 0, 0), Vec3(0, 3, 0), Vec3(0, 0, 3));
    system.addParticle(1.0);
    system.addParticle(1.0);
    system.addForce(force); // System takes ownership

    vector<Vec3> positions = {
        Vec3(0.0, 0.0, 0.0), Vec3(1.2, 0.0, 0.0),  // 12 Angstrom apart, no contact,
                                                     // not a half-box MIC tie
    };

    VerletIntegrator integrator(0.001);
    Platform& platform = Platform::getPlatformByName(platformName);
    Context context(system, integrator, platform);
    context.setPositions(positions);
    State state = context.getState(State::Energy | State::Forces);

    double energy = state.getPotentialEnergy();
    double maxAbsForce = 0.0;
    for (const Vec3& f : state.getForces())
        maxAbsForce = max({maxAbsForce, fabs(f[0]), fabs(f[1]), fabs(f[2])});

    cout << "[" << platformName << "] (deserialized force) potential energy = " << energy
         << " kJ/mol, max|force| = " << maxAbsForce << endl;
    return energy == 0.0 && maxAbsForce == 0.0;
}

int main() {
    cout << "loading builtin platform plugins from: " << OPENMM_CONDA_PLUGIN_DIR << endl;
    Platform::loadPluginsFromDirectory(OPENMM_CONDA_PLUGIN_DIR);
    Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidual.so");
    Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidualReference.so");
    Platform::loadPluginLibrary(PLUGIN_DIR "/libOpenMMLocalManyBodyResidualCUDA.so");
    cout << "all plugin libraries loaded OK" << endl;

    bool ok = true;

    // Build the original force with non-default fields to prove round-trip fidelity.
    LocalManyBodyResidualForce original;
    original.setForceGroup(5);
    original.setName("MyLMBR_G0_test");
    fillMinimalNoContactParameters(original);
    original.setSkinAngstrom(1.0);
    original.setCandidateListCapacity(4096);

    ostringstream xmlOut;
    XmlSerializer::serialize<LocalManyBodyResidualForce>(&original, "Force", xmlOut);
    string xml = xmlOut.str();
    cout << "\n--- serialized XML ---\n" << xml << "--- end XML ---\n" << endl;

    bool hasSchemaVersion = xml.find("schema_version=\"2\"") != string::npos;
    cout << "schema_version=\"2\" present in XML: " << (hasSchemaVersion ? "yes" : "NO") << endl;
    ok = ok && hasSchemaVersion;

    istringstream xmlIn(xml);
    LocalManyBodyResidualForce* deserialized = XmlSerializer::deserialize<LocalManyBodyResidualForce>(xmlIn);

    bool groupOk = deserialized->getForceGroup() == 5;
    bool nameOk = deserialized->getName() == "MyLMBR_G0_test";
    cout << "forceGroup preserved (expected 5, got " << deserialized->getForceGroup() << "): " << (groupOk ? "yes" : "NO") << endl;
    cout << "name preserved (expected 'MyLMBR_G0_test', got '" << deserialized->getName() << "'): " << (nameOk ? "yes" : "NO") << endl;
    ok = ok && groupOk && nameOk;

    // G3 schema-v2 fields (skinAngstrom/candidateListCapacity) round-trip exactly.
    bool skinOk = deserialized->getSkinAngstrom() == 1.0;
    bool capacityOk = deserialized->getCandidateListCapacity() == 4096;
    cout << "skinAngstrom preserved (expected 1, got " << deserialized->getSkinAngstrom() << "): " << (skinOk ? "yes" : "NO") << endl;
    cout << "candidateListCapacity preserved (expected 4096, got " << deserialized->getCandidateListCapacity() << "): " << (capacityOk ? "yes" : "NO") << endl;
    ok = ok && skinOk && capacityOk;

    // schema-v2 refresh (user-required): a v1 XML must be explicitly REJECTED,
    // never silently reinterpreted with frozen defaults for the missing
    // skinAngstrom/candidateListCapacity fields. Hand-craft a v1-shaped XML
    // by taking the real v2 XML and rewriting only schema_version, then
    // dropping the two new attributes (simulating what an actual old-format
    // v1 writer would have produced) -- deserialize() must throw.
    {
        string v1Xml = xml;
        size_t pos = v1Xml.find("schema_version=\"2\"");
        v1Xml.replace(pos, string("schema_version=\"2\"").size(), "schema_version=\"1\"");
        auto stripAttr = [](string& s, const string& name) {
            size_t p = s.find(name + "=\"");
            if (p == string::npos) return;
            size_t q = s.find('"', p + name.size() + 2);
            s.erase(p, q + 1 - p + 1);  // +1 to also eat the trailing space
        };
        stripAttr(v1Xml, "skinAngstrom");
        stripAttr(v1Xml, "candidateListCapacity");
        istringstream v1In(v1Xml);
        bool threw = false;
        try {
            LocalManyBodyResidualForce* shouldNotExist = XmlSerializer::deserialize<LocalManyBodyResidualForce>(v1In);
            delete shouldNotExist;
        } catch (const exception& e) {
            threw = true;
            cout << "v1 XML correctly rejected: " << e.what() << endl;
        }
        cout << "v1 (schema_version=1) XML fails closed rather than silently defaulting: " << (threw ? "yes" : "NO") << endl;
        ok = ok && threw;
    }

    // Deliberately test on a Context that never touched `original` or any
    // prior device state -- `deserialized` came purely from the XML string.
    cout << "\n--- Reference platform (deserialized force) ---" << endl;
    try {
        ok = runDeserializedOnPlatform("Reference", deserialized) && ok;
    } catch (const exception& e) {
        cout << "Reference round-trip test threw: " << e.what() << endl;
        ok = false;
    }

    // Second independent deserialize, so the CUDA leg also never touches
    // the Reference leg's Force/System/Context objects.
    istringstream xmlIn2(xml);
    LocalManyBodyResidualForce* deserialized2 = XmlSerializer::deserialize<LocalManyBodyResidualForce>(xmlIn2);
    cout << "\n--- CUDA platform (independently deserialized force) ---" << endl;
    try {
        ok = runDeserializedOnPlatform("CUDA", deserialized2) && ok;
    } catch (const exception& e) {
        cout << "CUDA round-trip test threw: " << e.what() << endl;
        ok = false;
    }

    cout << "\nG0 SERIALIZATION ROUND-TRIP TEST: " << (ok ? "PASS" : "FAIL") << endl;
    return ok ? 0 : 1;
}
