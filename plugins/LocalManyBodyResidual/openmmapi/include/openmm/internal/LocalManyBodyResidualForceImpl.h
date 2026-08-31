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
