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

#ifdef WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <dirent.h>
#include <cstdlib>
#endif

#include "openmm/OpenMMException.h"
#include "openmm/LocalManyBodyResidualForce.h"
#include "openmm/serialization/SerializationProxy.h"
#include "openmm/serialization/LocalManyBodyResidualForceProxy.h"

#if defined(WIN32)
    #include <windows.h>
    extern "C" OPENMM_EXPORT_LMBR void registerLocalManyBodyResidualSerializationProxies();
    BOOL WINAPI DllMain(HANDLE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
        if (ul_reason_for_call == DLL_PROCESS_ATTACH)
            registerLocalManyBodyResidualSerializationProxies();
        return TRUE;
    }
#else
    extern "C" void __attribute__((constructor)) registerLocalManyBodyResidualSerializationProxies();
#endif

using namespace OpenMM;

extern "C" OPENMM_EXPORT_LMBR void registerLocalManyBodyResidualSerializationProxies() {
    SerializationProxy::registerProxy(typeid(LocalManyBodyResidualForce), new LocalManyBodyResidualForceProxy());
}
