#include "openmm/LocalManyBodyResidualForce.h"
#include "openmm/internal/LocalManyBodyResidualForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;
using namespace std;

LocalManyBodyResidualForce::LocalManyBodyResidualForce() {
}

ForceImpl* LocalManyBodyResidualForce::createImpl() const {
    return new LocalManyBodyResidualForceImpl(*this);
}

void LocalManyBodyResidualForce::setTemperatureKelvin(double t) {
    if (t <= 0.0) throw OpenMMException("LocalManyBodyResidualForce: temperatureKelvin must be positive");
    temperatureKelvin = t;
}
double LocalManyBodyResidualForce::getTemperatureKelvin() const { return temperatureKelvin; }

double LocalManyBodyResidualForce::getMolarGasConstantRKilojoulePerMoleKelvin() {
    // CODATA 2018, matching openmm/unit/constants.py exactly:
    //   AVOGADRO_CONSTANT_NA = 6.02214076e23 / mol
    //   BOLTZMANN_CONSTANT_kB = 1.380649e-23 J/K
    //   MOLAR_GAS_CONSTANT_R = NA * kB = 8.31446261815324 J/(K mol)
    // OpenMM's C++ API has no named symbol for this constant; the literal
    // below was verified bit-for-bit against openmm.unit.MOLAR_GAS_CONSTANT_R
    // in this exact conda install (2026-08-12).
    return 8.31446261815324e-3;  // kJ/(mol K)
}

double LocalManyBodyResidualForce::getKBTKilojoulePerMole() const {
    return getMolarGasConstantRKilojoulePerMoleKelvin() * temperatureKelvin;
}

void LocalManyBodyResidualForce::setLigandTopologyIds(const vector<int>& ids) { ligandTopologyIds = ids; }
const vector<int>& LocalManyBodyResidualForce::getLigandTopologyIds() const { return ligandTopologyIds; }

void LocalManyBodyResidualForce::setAtomTypeIndex(const vector<int>& types) { atomTypeIndex = types; }
const vector<int>& LocalManyBodyResidualForce::getAtomTypeIndex() const { return atomTypeIndex; }

void LocalManyBodyResidualForce::setTypeVocabulary(const vector<int>& atomicNumbers) { typeVocabulary = atomicNumbers; }
const vector<int>& LocalManyBodyResidualForce::getTypeVocabulary() const { return typeVocabulary; }

void LocalManyBodyResidualForce::setInnerCutoffAngstrom(double v) { innerCutoffAngstrom = v; }
double LocalManyBodyResidualForce::getInnerCutoffAngstrom() const { return innerCutoffAngstrom; }
void LocalManyBodyResidualForce::setOuterCutoffAngstrom(double v) { outerCutoffAngstrom = v; }
double LocalManyBodyResidualForce::getOuterCutoffAngstrom() const { return outerCutoffAngstrom; }

void LocalManyBodyResidualForce::setRadialCenters(const vector<double>& centers) { radialCenters = centers; }
const vector<double>& LocalManyBodyResidualForce::getRadialCenters() const { return radialCenters; }
void LocalManyBodyResidualForce::setRadialWidthAngstrom(double v) { radialWidthAngstrom = v; }
double LocalManyBodyResidualForce::getRadialWidthAngstrom() const { return radialWidthAngstrom; }

void LocalManyBodyResidualForce::setPairWeight(const vector<double>& weights) { pairWeight = weights; }
const vector<double>& LocalManyBodyResidualForce::getPairWeight() const { return pairWeight; }

void LocalManyBodyResidualForce::setTypedMLP(int typeIndex, const LocalManyBodyTypedMLP& mlp) {
    if (typeIndex < 0) throw OpenMMException("LocalManyBodyResidualForce: typeIndex must be non-negative");
    if ((int)rho.size() <= typeIndex) rho.resize(typeIndex + 1);
    if (mlp.w0.size() != 16 || mlp.b0.size() != 16 || mlp.w2.size() != 256 || mlp.b2.size() != 16 || mlp.w4.size() != 16)
        throw OpenMMException("LocalManyBodyResidualForce: typed MLP tensor shapes must be (16,16,256,16,16,1)");
    rho[typeIndex] = mlp;
}
const LocalManyBodyTypedMLP& LocalManyBodyResidualForce::getTypedMLP(int typeIndex) const {
    return rho.at(typeIndex);
}
int LocalManyBodyResidualForce::getNumTypes() const { return (int)rho.size(); }
int LocalManyBodyResidualForce::getNumRadialBasis() const { return (int)radialCenters.size(); }

void LocalManyBodyResidualForce::setBMaxReduced(double v) { bMaxReduced = v; }
double LocalManyBodyResidualForce::getBMaxReduced() const { return bMaxReduced; }

void LocalManyBodyResidualForce::setCapacityCeilings(int edges, int neighbors, int environmentAtoms) {
    maxEdges = edges;
    maxNeighborsPerLigand = neighbors;
    maxEnvironmentAtoms = environmentAtoms;
}
void LocalManyBodyResidualForce::getCapacityCeilings(int& edges, int& neighbors, int& environmentAtoms) const {
    edges = maxEdges;
    neighbors = maxNeighborsPerLigand;
    environmentAtoms = maxEnvironmentAtoms;
}

void LocalManyBodyResidualForce::setSkinAngstrom(double v) { skinAngstrom = v; }
double LocalManyBodyResidualForce::getSkinAngstrom() const { return skinAngstrom; }
void LocalManyBodyResidualForce::setCandidateListCapacity(int v) { candidateListCapacity = v; }
int LocalManyBodyResidualForce::getCandidateListCapacity() const { return candidateListCapacity; }

void LocalManyBodyResidualForce::setSourceCheckpointSha256(const string& sha256) { sourceCheckpointSha256 = sha256; }
const string& LocalManyBodyResidualForce::getSourceCheckpointSha256() const { return sourceCheckpointSha256; }
