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

/* Legacy fail-closed floor. KEPT: EXP025_DEVICE_ERROR_MIN_DISTANCE (=2) is
 * pinned by a static_assert in exp026_control_plane_layout.h, so the error
 * code stays part of the ABI. As of 2026-09-16 (LR-06 plan A) no kernel
 * raises it any more -- the four sites clamp instead of erroring. This macro
 * survives only because the device source still #defines it and tests read it.
 */
#define EXP025_MIN_DISTANCE_ANGSTROM 0.1

/* [LR-06 plan A, 2026-09-16] Pair-distance floor used for CLAMPING.
 *
 * Why clamping replaced the fail-closed error
 * -------------------------------------------
 * `bias_scale` multiplies the WHOLE Group-1 expression, which contains
 * cv_k_int + cv_k_rest (the ligand<->environment softcore interaction). So
 * `bias_scale = 0` is not "bias off", it makes the ligand a full ghost:
 * water passes straight through it and r -> 0 is inevitable. Three code
 * paths enter that state (EM, the dt ramp, the bias ramp) and two of them
 * are deliberate. Meanwhile CustomCVForce evaluates EVERY collective
 * variable regardless of its coefficient, so the plugin still ran on ghost
 * geometry and hit the old hard gate. Blocking individual evaluation sites
 * was tried and failed: three separate ones were found in one day
 * (integrator force groups, groupless getState, getCollectiveVariableValues).
 * Clamping removes the failure mode instead of arranging for it to go
 * unobserved.
 *
 * Why 1.5 and not EXP025_MIN_DISTANCE_ANGSTROM (0.1)
 * --------------------------------------------------
 * 0.1 A only ever meant "keep 1/r from blowing up"; it is not a statement
 * about the model. This value is the TRAINING SUPPORT LOWER BOUND. Measured
 * on the shipped R1 model's training frames (pre_equilibration.dcd, 200
 * frames, lambda=1, per-frame box): closest ligand<->environment approach is
 * 1.517 A (water hydrogen) / 1.524 A (LJ-bearing), median 1.8 A.
 *
 * This matters because the radial basis does NOT decay at small r -- its 16
 * centers are spread uniformly over [0, 5] A with width 0.333, so the basis
 * is ~1.0 at 0.1 A. And the pair weights sitting on the five centers below
 * the support bound (0.333 / 0.667 / 1.0 / 1.333 A) have the SAME magnitude
 * as the well-trained ones (max|w| 0.30-0.37 vs 0.32-0.39): fully active,
 * never constrained by data. Clamping at 0.1 A would keep evaluating them.
 *
 * ponytail: this is a compile-time constant matching the SHIPPED R1 model.
 * A retrained model with a different support domain would silently reuse it.
 * Upgrade path when that happens: carry `r_floor_angstrom` in the payload
 * config + manifest (the offline trainer already measures it) and emit it
 * through `defines` in buildAndLoadKernels(), the way NUM_RADIAL_BASIS is.
 */
#define EXP025_R_FLOOR_ANGSTROM 1.5

/* Half-box MIC tie epsilon. Deliberately LOOSER than the CPU Reference's
 * 1e-9 (g1_math_core.h HALF_BOX_TIE_EPSILON): the default CUDA platform
 * precision on this install is single ("real" = float, ~7 decimal digits),
 * so a fractional coordinate near an exact 0.5 tie can carry ~1e-7-scale
 * rounding noise from single-precision position storage alone, well above
 * 1e-9. Using the CPU's tighter epsilon here would essentially never fire
 * and give a false sense of tie coverage on the CUDA path. */
#define EXP025_HALF_BOX_TIE_EPSILON 1e-6

#endif /* EXP025_R1_MODEL_LAYOUT_H_ */
