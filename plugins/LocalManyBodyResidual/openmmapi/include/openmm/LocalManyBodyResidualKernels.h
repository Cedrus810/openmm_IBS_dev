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
