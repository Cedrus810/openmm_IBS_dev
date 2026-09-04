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

#ifndef OPENMM_WINDOWSEXPORTLOCALMANYBODYRESIDUAL_H_
#define OPENMM_WINDOWSEXPORTLOCALMANYBODYRESIDUAL_H_

/*
 * EXP-025 G0 scaffold. See PLAN_EXP-025_local_manybody_cuda.md section 4.
 * Linux-only export macro (this project only targets Linux/CUDA); the
 * Windows dllexport/dllimport branching is kept only for source-compatibility
 * with the OpenMM plugin template this file was derived from.
 */

#ifdef _MSC_VER
    #pragma warning(disable:4996)
    #pragma warning(disable:4251)
    #if defined(OPENMM_LMBR_BUILDING_SHARED_LIBRARY)
        #define OPENMM_EXPORT_LMBR __declspec(dllexport)
    #elif defined(OPENMM_LMBR_BUILDING_STATIC_LIBRARY) || defined(OPENMM_LMBR_USE_STATIC_LIBRARIES)
        #define OPENMM_EXPORT_LMBR
    #else
        #define OPENMM_EXPORT_LMBR __declspec(dllimport)
    #endif
#else
    #define OPENMM_EXPORT_LMBR
#endif

#endif // OPENMM_WINDOWSEXPORTLOCALMANYBODYRESIDUAL_H_
