#ifndef REFERENCE_LOCALMANYBODYRESIDUAL_KERNELS_H_
#define REFERENCE_LOCALMANYBODYRESIDUAL_KERNELS_H_

// EXP-025 G1-G: real R1 math on the Reference platform.
//
// Deliberately uses ../../../g1_math_core.h -- the SAME functions the
// already-validated standalone oracle (g1_reference_oracle.cpp) uses. This
// file must NOT depend on yyjson/OpenSSL/file I/O; parameter values arrive
// exclusively via LocalManyBodyResidualForce's public getters.

#include "openmm/reference/ReferencePlatform.h"
#include "openmm/LocalManyBodyResidualKernels.h"
#include "../../../g1_math_core.h"

namespace OpenMM {

class ReferenceCalcLocalManyBodyResidualForceKernel : public CalcLocalManyBodyResidualForceKernel {
public:
    ReferenceCalcLocalManyBodyResidualForceKernel(const std::string& name, const Platform& platform) : CalcLocalManyBodyResidualForceKernel(name, platform) {
    }
    void initialize(const System& system, const LocalManyBodyResidualForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);

private:
    exp025_g1::ModelParams model;
    std::vector<int32_t> ligandTopologyIds;
    std::vector<int32_t> atomTypeIndex;  // one entry per atom in the System
    double kBTKilojoulePerMole = 0.0;
};

} // namespace OpenMM

#endif /*REFERENCE_LOCALMANYBODYRESIDUAL_KERNELS_H_*/
