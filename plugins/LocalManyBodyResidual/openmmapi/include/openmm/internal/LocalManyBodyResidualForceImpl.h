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

#ifndef OPENMM_LOCALMANYBODYRESIDUALFORCEIMPL_H_
#define OPENMM_LOCALMANYBODYRESIDUALFORCEIMPL_H_

// EXP-025 G0 scaffold. See LocalManyBodyResidualForce.h for scope.

#include "openmm/LocalManyBodyResidualForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>
#include <vector>

namespace OpenMM {

class System;

class OPENMM_EXPORT_LMBR LocalManyBodyResidualForceImpl : public ForceImpl {
public:
    LocalManyBodyResidualForceImpl(const LocalManyBodyResidualForce& owner);
    ~LocalManyBodyResidualForceImpl();
    void initialize(ContextImpl& context);
    const LocalManyBodyResidualForce& getOwner() const {
        return owner;
    }
    void updateContextState(ContextImpl& context, bool& forcesInvalid) {
        // G0: no context-state-dependent behavior.
    }
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return {};
    }
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(ContextImpl& context);
private:
    const LocalManyBodyResidualForce& owner;
    Kernel kernel;
};

} // namespace OpenMM

#endif /*OPENMM_LOCALMANYBODYRESIDUALFORCEIMPL_H_*/
