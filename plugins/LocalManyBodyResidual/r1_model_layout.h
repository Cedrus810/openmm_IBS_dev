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

#ifndef EXP025_R1_MODEL_LAYOUT_H_
#define EXP025_R1_MODEL_LAYOUT_H_

/*
 * EXP-025 shared numeric layout contract -- CONSTANTS AND OFFSETS ONLY.
 *
 * Per the user's explicit G2 instruction: the CPU Reference math
 * (g1_math_core.h) and the CUDA device math (embedded kernel source string
 * built in CudaLocalManyBodyResidualKernels.cpp) must implement the actual
 * computation INDEPENDENTLY -- sharing one implementation between platforms
 * would let the same bug hide behind "CUDA matches Reference". What IS safe
 * and useful to share is the pure numeric layout: how the frozen typed-MLP
 * weights are packed into one flat buffer per type, so the host-side upload
 * code and the device-side kernel agree on where each tensor lives without
 * duplicating (and risking disagreement on) the arithmetic by hand.
 *
 * These are plain C preprocessor macros (not C++ constexpr) so the exact
 * same numeric literals can be re-stringified into the NVRTC-compiled CUDA
 * source text (see CudaLocalManyBodyResidualKernels.cpp) -- there is
 * exactly one place these numbers are typed.
 *
 * Per-type flat layout (EXP025_MLP_STRIDE doubles/reals per ligand type):
 *   [0 .. 16)    W0   Linear(1,16).weight,  16x1 flattened
 *   [16 .. 32)   b0   Linear(1,16).bias,    16
 *   [32 .. 288)  W2   Linear(16,16).weight, 16x16 row-major [out][in]
 *   [288 .. 304) b2   Linear(16,16).bias,   16
 *   [304 .. 320) W4   Linear(16,1).weight,  1x16 flattened
 *   [320 .. 321) b4   Linear(16,1).bias,    1
 */

#define EXP025_HIDDEN_RHO 16

#define EXP025_MLP_W0_SIZE EXP025_HIDDEN_RHO
#define EXP025_MLP_B0_SIZE EXP025_HIDDEN_RHO
#define EXP025_MLP_W2_SIZE (EXP025_HIDDEN_RHO * EXP025_HIDDEN_RHO)
#define EXP025_MLP_B2_SIZE EXP025_HIDDEN_RHO
#define EXP025_MLP_W4_SIZE EXP025_HIDDEN_RHO
#define EXP025_MLP_B4_SIZE 1

#define EXP025_MLP_OFFSET_W0 0
#define EXP025_MLP_OFFSET_B0 (EXP025_MLP_OFFSET_W0 + EXP025_MLP_W0_SIZE)
#define EXP025_MLP_OFFSET_W2 (EXP025_MLP_OFFSET_B0 + EXP025_MLP_B0_SIZE)
#define EXP025_MLP_OFFSET_B2 (EXP025_MLP_OFFSET_W2 + EXP025_MLP_W2_SIZE)
#define EXP025_MLP_OFFSET_W4 (EXP025_MLP_OFFSET_B2 + EXP025_MLP_B2_SIZE)
#define EXP025_MLP_OFFSET_B4 (EXP025_MLP_OFFSET_W4 + EXP025_MLP_W4_SIZE)

#define EXP025_MLP_STRIDE \
    (EXP025_MLP_W0_SIZE + EXP025_MLP_B0_SIZE + EXP025_MLP_W2_SIZE + \
     EXP025_MLP_B2_SIZE + EXP025_MLP_W4_SIZE + EXP025_MLP_B4_SIZE)

/* pairWeight flat layout: index(ligandType, envType, p) for a model with
 * `typeCount` types and `nRadialBasis` (== 16 in the frozen R1 contract,
 * but not hardcoded here) radial basis functions per pair. */
#define EXP025_PAIR_WEIGHT_INDEX(ligandType, envType, p, typeCount, nRadialBasis) \
    ((size_t)((ligandType) * (typeCount) + (envType)) * (size_t)(nRadialBasis) + (size_t)(p))

/* GPU fail-closed error codes (see PLAN discussion: G2 allows a synchronous
 * post-kernel host readback + throw; removing that sync is a G3/G4 cost
 * concern, not a G2 correctness concern). */
#define EXP025_DEVICE_ERROR_OK 0
#define EXP025_DEVICE_ERROR_HALF_BOX_TIE 1
#define EXP025_DEVICE_ERROR_MIN_DISTANCE 2
#define EXP025_DEVICE_ERROR_EDGE_OVERFLOW 3
#define EXP025_DEVICE_ERROR_NEIGHBOR_OVERFLOW 4
#define EXP025_DEVICE_ERROR_UNIQUE_ENV_OVERFLOW 5
#define EXP025_DEVICE_ERROR_NONFINITE 6
/* G3 (local CSR/Verlet) additions -- see CudaLocalManyBodyResidualKernels.cpp.
 * CANDIDATE_OVERFLOW: total <0.6nm candidates across all 41 anchors exceeded
 * the frozen candidateListCapacity (separate ceiling from the four G2 active-
 * support ceilings above, which stay scoped to <0.5nm active edges). */
#define EXP025_DEVICE_ERROR_CANDIDATE_OVERFLOW 7
#define EXP025_DEVICE_ERROR_UNSUPPORTED_BOX 8

#define EXP025_MIN_DISTANCE_ANGSTROM 0.1

/* Half-box MIC tie epsilon. Deliberately LOOSER than the CPU Reference's
 * 1e-9 (g1_math_core.h HALF_BOX_TIE_EPSILON): the default CUDA platform
 * precision on this install is single ("real" = float, ~7 decimal digits),
 * so a fractional coordinate near an exact 0.5 tie can carry ~1e-7-scale
 * rounding noise from single-precision position storage alone, well above
 * 1e-9. Using the CPU's tighter epsilon here would essentially never fire
 * and give a false sense of tie coverage on the CUDA path. */
#define EXP025_HALF_BOX_TIE_EPSILON 1e-6

#endif /* EXP025_R1_MODEL_LAYOUT_H_ */
