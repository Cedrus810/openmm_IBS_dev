#ifndef OPENMM_CUDALOCALMANYBODYRESIDUALKERNELFACTORY_H_
#define OPENMM_CUDALOCALMANYBODYRESIDUALKERNELFACTORY_H_

// EXP-025 G0 scaffold.

#include "openmm/KernelFactory.h"

namespace OpenMM {

class CudaLocalManyBodyResidualKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_CUDALOCALMANYBODYRESIDUALKERNELFACTORY_H_*/
