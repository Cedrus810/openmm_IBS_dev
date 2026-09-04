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

#include "ReferenceLocalManyBodyResidualKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include <set>
#include <sstream>

using namespace OpenMM;
using namespace std;
using namespace exp025_g1;

// Local re-declarations matching OpenMM's own Reference platform convention
// (these are `static` in the core ReferenceKernels.cpp and not exported via
// any header -- every Reference-platform plugin re-declares them the same
// way; see plugins/drude, plugins/amoeba in the OpenMM source tree).
static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->positions;
}
static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}
static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return data->periodicBoxVectors;
}

void ReferenceCalcLocalManyBodyResidualForceKernel::initialize(const System& system, const LocalManyBodyResidualForce& force) {
    int numParticles = system.getNumParticles();

    const vector<int>& ligandIds = force.getLigandTopologyIds();
    const vector<int>& types = force.getAtomTypeIndex();
    const vector<int>& vocab = force.getTypeVocabulary();

    if ((int)types.size() != numParticles)
        throw OpenMMException("LocalManyBodyResidualForce: atomTypeIndex size (" + to_string(types.size()) +
                               ") must equal the System particle count (" + to_string(numParticles) + ")");
    int typeCount = (int)vocab.size();
    if (typeCount == 0) throw OpenMMException("LocalManyBodyResidualForce: type vocabulary is empty");
    if (force.getNumTypes() != typeCount)
        throw OpenMMException("LocalManyBodyResidualForce: number of typed MLPs (" + to_string(force.getNumTypes()) +
                               ") must equal type vocabulary size (" + to_string(typeCount) + ")");
    for (int t : types)
        if (t < 0 || t >= typeCount)
            throw OpenMMException("LocalManyBodyResidualForce: atomTypeIndex entry " + to_string(t) + " is outside the type vocabulary");

    set<int> ligandSet;
    for (int id : ligandIds) {
        if (id < 0 || id >= numParticles)
            throw OpenMMException("LocalManyBodyResidualForce: ligand topology id " + to_string(id) + " is outside the System");
        if (!ligandSet.insert(id).second)
            throw OpenMMException("LocalManyBodyResidualForce: duplicate ligand topology id " + to_string(id));
    }
    if (ligandIds.empty())
        throw OpenMMException("LocalManyBodyResidualForce: ligandTopologyIds must be non-empty");

    int nRadialBasis = force.getNumRadialBasis();
    if (nRadialBasis <= 0) throw OpenMMException("LocalManyBodyResidualForce: n_radial_basis must be positive");
    if ((int)force.getRadialCenters().size() != nRadialBasis)
        throw OpenMMException("LocalManyBodyResidualForce: radialCenters size disagrees with n_radial_basis");
    if ((int)force.getPairWeight().size() != typeCount * typeCount * nRadialBasis)
        throw OpenMMException("LocalManyBodyResidualForce: pairWeight size disagrees with type_count^2 * n_radial_basis");
    if (!(force.getInnerCutoffAngstrom() > 0.0 && force.getInnerCutoffAngstrom() < force.getOuterCutoffAngstrom()))
        throw OpenMMException("LocalManyBodyResidualForce: cutoffs must satisfy 0 < inner < outer");
    if (!(force.getBMaxReduced() > 0.0)) throw OpenMMException("LocalManyBodyResidualForce: b_max_reduced must be positive");
    if (!(force.getTemperatureKelvin() > 0.0)) throw OpenMMException("LocalManyBodyResidualForce: temperatureKelvin must be positive");

    int maxEdges, maxNeighborsPerLigand, maxEnvironmentAtoms;
    force.getCapacityCeilings(maxEdges, maxNeighborsPerLigand, maxEnvironmentAtoms);
    if (maxEdges <= 0 || maxNeighborsPerLigand <= 0 || maxEnvironmentAtoms <= 0)
        throw OpenMMException("LocalManyBodyResidualForce: capacity ceilings must all be positive");

    model.nLigandAtoms = (int)ligandIds.size();
    model.nRadialBasis = nRadialBasis;
    model.typeCount = typeCount;
    model.innerCutoffAngstrom = force.getInnerCutoffAngstrom();
    model.outerCutoffAngstrom = force.getOuterCutoffAngstrom();
    model.bMaxReduced = force.getBMaxReduced();
    model.maxEdges = maxEdges;
    model.maxNeighborsPerLigand = maxNeighborsPerLigand;
    model.maxEnvironmentAtoms = maxEnvironmentAtoms;
    model.radialCenters = force.getRadialCenters();
    model.radialWidth = force.getRadialWidthAngstrom();
    model.pairWeight = force.getPairWeight();
    model.rho.resize(typeCount);
    for (int t = 0; t < typeCount; t++) {
        const LocalManyBodyTypedMLP& src = force.getTypedMLP(t);
        TypedMLP& dst = model.rho[t];
        for (int k = 0; k < 16; k++) { dst.W0[k] = src.w0[k]; dst.b0[k] = src.b0[k]; }
        for (int o = 0; o < 16; o++)
            for (int k = 0; k < 16; k++) dst.W2[o][k] = src.w2[(size_t)o * 16 + k];
        for (int k = 0; k < 16; k++) dst.b2[k] = src.b2[k];
        for (int k = 0; k < 16; k++) dst.W4[k] = src.w4[k];
        dst.b4 = src.b4;
    }

    ligandTopologyIds.assign(ligandIds.begin(), ligandIds.end());
    atomTypeIndex.assign(types.begin(), types.end());
    kBTKilojoulePerMole = force.getKBTKilojoulePerMole();
}

double ReferenceCalcLocalManyBodyResidualForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<Vec3>& positions = extractPositions(context);
    Vec3* box = extractBoxVectors(context);

    AtomSystemView fx;
    fx.nAtoms = (int)positions.size();
    fx.nLigand = (int)ligandTopologyIds.size();
    fx.positionsNm.resize(fx.nAtoms);
    for (int i = 0; i < fx.nAtoms; i++) fx.positionsNm[i] = {positions[i][0], positions[i][1], positions[i][2]};
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) fx.boxNm[r][c] = box[r][c];
    fx.atomTypeIndex = atomTypeIndex;
    fx.ligandTopologyIds = ligandTopologyIds;

    EnumerationResult enumeration = enumerateEdges(fx, model);
    ForwardResult result = evaluate(fx, model, enumeration);

    if (includeForces) {
        vector<Vec3>& forces = extractForces(context);
        for (const auto& [atomIndex, gradAngstrom] : result.gradientAngstromByAtom) {
            // Angstrom -> nm chain rule (x10), then gradient -> physical
            // force (-kBT), applied EXACTLY ONCE, here, and nowhere else.
            double fx_ = -kBTKilojoulePerMole * 10.0 * gradAngstrom[0];
            double fy_ = -kBTKilojoulePerMole * 10.0 * gradAngstrom[1];
            double fz_ = -kBTKilojoulePerMole * 10.0 * gradAngstrom[2];
            forces[atomIndex][0] += fx_;
            forces[atomIndex][1] += fy_;
            forces[atomIndex][2] += fz_;
        }
    }

    if (!includeEnergy) return 0.0;
    return kBTKilojoulePerMole * result.bReduced;  // U_B = kBT * B, applied exactly once, here.
}
