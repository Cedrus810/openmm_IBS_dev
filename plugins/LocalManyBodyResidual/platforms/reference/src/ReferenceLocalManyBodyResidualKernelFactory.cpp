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
