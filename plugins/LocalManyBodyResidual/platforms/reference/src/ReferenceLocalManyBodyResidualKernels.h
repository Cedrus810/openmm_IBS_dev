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
