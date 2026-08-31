#include <exception>

#include "CudaLocalManyBodyResidualKernelFactory.h"
#include "CudaLocalManyBodyResidualKernels.h"
#include "CudaContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaLocalManyBodyResidualKernelFactory* factory = new CudaLocalManyBodyResidualKernelFactory();
        platform.registerKernelFactory(CalcLocalManyBodyResidualForceKernel::Name(), factory);
    }
    catch (std::exception& ex) {
        // No CUDA platform registered yet / available. Ignore, matching the
        // convention used by OpenMM's own bundled platform plugins (see
        // plugins/drude/platforms/cuda/src/CudaDrudeKernelFactory.cpp).
    }
}

extern "C" OPENMM_EXPORT void registerLocalManyBodyResidualCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    }
    catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}

KernelImpl* CudaLocalManyBodyResidualKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcLocalManyBodyResidualForceKernel::Name())
        return new CudaCalcLocalManyBodyResidualForceKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
