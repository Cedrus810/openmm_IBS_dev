#include "ReferenceLocalManyBodyResidualKernelFactory.h"
#include "ReferenceLocalManyBodyResidualKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceLocalManyBodyResidualKernelFactory* factory = new ReferenceLocalManyBodyResidualKernelFactory();
            platform.registerKernelFactory(CalcLocalManyBodyResidualForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerLocalManyBodyResidualReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceLocalManyBodyResidualKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    if (name == CalcLocalManyBodyResidualForceKernel::Name())
        return new ReferenceCalcLocalManyBodyResidualForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
