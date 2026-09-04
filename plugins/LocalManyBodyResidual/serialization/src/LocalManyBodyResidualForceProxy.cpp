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

#include "openmm/serialization/LocalManyBodyResidualForceProxy.h"
#include "openmm/serialization/SerializationNode.h"
#include "openmm/Force.h"
#include "openmm/LocalManyBodyResidualForce.h"
#include "openmm/OpenMMException.h"

#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

using namespace OpenMM;
using namespace std;

// Bumped 1 -> 2 for EXP-025 G3: skinAngstrom/candidateListCapacity are new
// REQUIRED-together fields that change the math/cost contract (selecting the
// local-CSR code path), not a cosmetic add -- an old schema_version=1 XML
// must not be silently reinterpreted as "skin=0", it must fail closed.
static const int SCHEMA_VERSION = 2;

// Large per-atom/per-parameter arrays (atomTypeIndex alone can have tens of
// thousands of entries) are encoded as a single whitespace-separated string
// property rather than one XML child node per entry -- avoids O(n) XML
// element overhead for n in the tens of thousands. Doubles use
// max_digits10 precision so the XML round-trip is exact, not lossy.
static string encodeIntArray(const vector<int>& values) {
    ostringstream oss;
    for (size_t i = 0; i < values.size(); i++) {
        if (i) oss << ' ';
        oss << values[i];
    }
    return oss.str();
}
static vector<int> decodeIntArray(const string& s) {
    vector<int> out;
    istringstream iss(s);
    int v;
    while (iss >> v) out.push_back(v);
    return out;
}
static string encodeDoubleArray(const vector<double>& values) {
    ostringstream oss;
    oss << setprecision(numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < values.size(); i++) {
        if (i) oss << ' ';
        oss << values[i];
    }
    return oss.str();
}
static vector<double> decodeDoubleArray(const string& s) {
    vector<double> out;
    istringstream iss(s);
    double v;
    while (iss >> v) out.push_back(v);
    return out;
}
static string encodeDouble(double v) {
    ostringstream oss;
    oss << setprecision(numeric_limits<double>::max_digits10) << v;
    return oss.str();
}

LocalManyBodyResidualForceProxy::LocalManyBodyResidualForceProxy() : SerializationProxy("LocalManyBodyResidualForce") {
}

void LocalManyBodyResidualForceProxy::serialize(const void* object, SerializationNode& node) const {
    node.setIntProperty("schema_version", SCHEMA_VERSION);
    const LocalManyBodyResidualForce& force = *reinterpret_cast<const LocalManyBodyResidualForce*>(object);
    node.setIntProperty("forceGroup", force.getForceGroup());
    node.setStringProperty("name", force.getName());

    node.setStringProperty("temperatureKelvin", encodeDouble(force.getTemperatureKelvin()));
    node.setStringProperty("innerCutoffAngstrom", encodeDouble(force.getInnerCutoffAngstrom()));
    node.setStringProperty("outerCutoffAngstrom", encodeDouble(force.getOuterCutoffAngstrom()));
    node.setStringProperty("radialWidthAngstrom", encodeDouble(force.getRadialWidthAngstrom()));
    node.setStringProperty("bMaxReduced", encodeDouble(force.getBMaxReduced()));
    int maxEdges, maxNeighborsPerLigand, maxEnvironmentAtoms;
    force.getCapacityCeilings(maxEdges, maxNeighborsPerLigand, maxEnvironmentAtoms);
    node.setIntProperty("maxEdges", maxEdges);
    node.setIntProperty("maxNeighborsPerLigand", maxNeighborsPerLigand);
    node.setIntProperty("maxEnvironmentAtoms", maxEnvironmentAtoms);
    node.setStringProperty("skinAngstrom", encodeDouble(force.getSkinAngstrom()));
    node.setIntProperty("candidateListCapacity", force.getCandidateListCapacity());
    node.setStringProperty("sourceCheckpointSha256", force.getSourceCheckpointSha256());

    node.setStringProperty("ligandTopologyIds", encodeIntArray(force.getLigandTopologyIds()));
    node.setStringProperty("atomTypeIndex", encodeIntArray(force.getAtomTypeIndex()));
    node.setStringProperty("typeVocabulary", encodeIntArray(force.getTypeVocabulary()));
    node.setStringProperty("radialCenters", encodeDoubleArray(force.getRadialCenters()));
    node.setStringProperty("pairWeight", encodeDoubleArray(force.getPairWeight()));

    int numTypes = force.getNumTypes();
    node.setIntProperty("numTypes", numTypes);
    SerializationNode& rhoNode = node.createChildNode("TypedMLPs");
    for (int t = 0; t < numTypes; t++) {
        const LocalManyBodyTypedMLP& mlp = force.getTypedMLP(t);
        SerializationNode& mlpNode = rhoNode.createChildNode("MLP");
        mlpNode.setStringProperty("w0", encodeDoubleArray(mlp.w0));
        mlpNode.setStringProperty("b0", encodeDoubleArray(mlp.b0));
        mlpNode.setStringProperty("w2", encodeDoubleArray(mlp.w2));
        mlpNode.setStringProperty("b2", encodeDoubleArray(mlp.b2));
        mlpNode.setStringProperty("w4", encodeDoubleArray(mlp.w4));
        mlpNode.setStringProperty("b4", encodeDouble(mlp.b4));
    }
}

void* LocalManyBodyResidualForceProxy::deserialize(const SerializationNode& node) const {
    int schemaVersion = node.getIntProperty("schema_version");
    if (schemaVersion != SCHEMA_VERSION)
        throw OpenMMException("LocalManyBodyResidualForce: unsupported schema_version " + std::to_string(schemaVersion) +
            " (this plugin build only supports " + std::to_string(SCHEMA_VERSION) + ")");
    LocalManyBodyResidualForce* force = new LocalManyBodyResidualForce();
    try {
        force->setForceGroup(node.getIntProperty("forceGroup", 0));
        force->setName(node.getStringProperty("name", force->getName()));

        force->setTemperatureKelvin(stod(node.getStringProperty("temperatureKelvin", "1")));
        force->setInnerCutoffAngstrom(stod(node.getStringProperty("innerCutoffAngstrom", "0")));
        force->setOuterCutoffAngstrom(stod(node.getStringProperty("outerCutoffAngstrom", "0")));
        force->setRadialWidthAngstrom(stod(node.getStringProperty("radialWidthAngstrom", "0")));
        force->setBMaxReduced(stod(node.getStringProperty("bMaxReduced", "0")));
        int maxEdges = node.getIntProperty("maxEdges", 0);
        int maxNeighborsPerLigand = node.getIntProperty("maxNeighborsPerLigand", 0);
        int maxEnvironmentAtoms = node.getIntProperty("maxEnvironmentAtoms", 0);
        force->setCapacityCeilings(maxEdges, maxNeighborsPerLigand, maxEnvironmentAtoms);
        force->setSkinAngstrom(stod(node.getStringProperty("skinAngstrom", "0")));
        force->setCandidateListCapacity(node.getIntProperty("candidateListCapacity", 0));
        force->setSourceCheckpointSha256(node.getStringProperty("sourceCheckpointSha256", ""));

        force->setLigandTopologyIds(decodeIntArray(node.getStringProperty("ligandTopologyIds", "")));
        force->setAtomTypeIndex(decodeIntArray(node.getStringProperty("atomTypeIndex", "")));
        force->setTypeVocabulary(decodeIntArray(node.getStringProperty("typeVocabulary", "")));
        force->setRadialCenters(decodeDoubleArray(node.getStringProperty("radialCenters", "")));
        force->setPairWeight(decodeDoubleArray(node.getStringProperty("pairWeight", "")));

        // numTypes==0 (a default-constructed Force never wrote a TypedMLPs
        // child at all) must skip getChildNode(), which would otherwise throw.
        int numTypes = node.getIntProperty("numTypes", 0);
        if (numTypes > 0) {
            const SerializationNode& rhoNode = node.getChildNode("TypedMLPs");
            const auto& children = rhoNode.getChildren();
            if ((int)children.size() != numTypes)
                throw OpenMMException("LocalManyBodyResidualForce: TypedMLPs child count disagrees with numTypes");
            for (int t = 0; t < numTypes; t++) {
                const SerializationNode& mlpNode = children[t];
                LocalManyBodyTypedMLP mlp;
                mlp.w0 = decodeDoubleArray(mlpNode.getStringProperty("w0"));
                mlp.b0 = decodeDoubleArray(mlpNode.getStringProperty("b0"));
                mlp.w2 = decodeDoubleArray(mlpNode.getStringProperty("w2"));
                mlp.b2 = decodeDoubleArray(mlpNode.getStringProperty("b2"));
                mlp.w4 = decodeDoubleArray(mlpNode.getStringProperty("w4"));
                mlp.b4 = stod(mlpNode.getStringProperty("b4"));
                force->setTypedMLP(t, mlp);
            }
        }
    }
    catch (...) {
        delete force;
        throw;
    }
    return force;
}
