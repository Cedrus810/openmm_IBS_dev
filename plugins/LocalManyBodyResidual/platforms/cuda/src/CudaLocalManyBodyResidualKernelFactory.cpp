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
