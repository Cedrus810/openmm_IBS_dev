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

#ifndef LOCALMANYBODYRESIDUAL_KERNELS_H_
#define LOCALMANYBODYRESIDUAL_KERNELS_H_

// EXP-025 G0 scaffold. Abstract kernel interface; Reference/CUDA implement it.

#include "openmm/LocalManyBodyResidualForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/System.h"
#include <string>

namespace OpenMM {

/**
 * G0: this kernel is required to contribute exactly zero energy and zero
 * force on every platform. It exists only to prove the plugin ABI/load/
 * Context lifecycle. See PLAN_EXP-025_local_manybody_cuda.md section 3 for
 * the real R1 math contract this will carry once G0 passes.
 */
class CalcLocalManyBodyResidualForceKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcLocalManyBodyResidualForce";
    }
    CalcLocalManyBodyResidualForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    virtual void initialize(const System& system, const LocalManyBodyResidualForce& force) = 0;
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;
};

} // namespace OpenMM

#endif /*LOCALMANYBODYRESIDUAL_KERNELS_H_*/
