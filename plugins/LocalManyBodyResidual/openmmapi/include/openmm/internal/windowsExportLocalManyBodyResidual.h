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
