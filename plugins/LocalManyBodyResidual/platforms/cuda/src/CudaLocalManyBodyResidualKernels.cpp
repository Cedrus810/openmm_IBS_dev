#include "CudaLocalManyBodyResidualKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "../../../r1_model_layout.h"
#include "../../../exp026_control_plane_layout.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>

using namespace OpenMM;
using namespace std;

#define EXP025_STRINGIFY_HELPER(x) #x
#define EXP025_STRINGIFY(x) EXP025_STRINGIFY_HELPER(x)

namespace {

// One block per ligand anchor for K1/K3. 41 anchors x this many threads
// each brute-force-scanning all atoms -- this IS the G2 brute-force cost
// profile, deliberately not optimized (that is G3/G4).
const int CUDA_BLOCK_SIZE = 128;

// Independent CUDA device math -- NOT a port of g1_math_core.h. Written
// from the frozen formulas (PLAN_EXP-025 section 3) directly, so that "CUDA
// matches Reference" is a real correctness signal, not two copies of one
// bug. Shares ONLY the pure layout/offset constants from r1_model_layout.h
// (re-stringified below into #defines) and the physical/geometric
// constants (cutoffs, kBT, box) that come from the Force's own parameters,
// not from any Reference-platform code.
//
// ============================== Mixed-precision position handling ==============================
// `real` and `mixed` are precision typedefs OpenMM's own ComputeContext::
// compileProgram()/createModule() auto-prepend to every compiled source
// (confirmed by reading that code directly during G2, not assumed):
//   - single   precision: real=float,  mixed=float  (mixed degenerates to real)
//   - mixed    precision: real=float,  mixed=double (genuine extra precision)
//   - double   precision: real=double, mixed=double (mixed degenerates to real)
// `USE_MIXED_PRECISION` below is OUR OWN preprocessor define (passed via
// compileProgram()'s `defines` map, set from cu.getUseMixedPrecision()) --
// it controls ONLY whether posqCorrection is read at all (that array does
// not exist unless the Context is genuinely in mixed-precision mode, per
// CudaContext::getPosqCorrection()'s own doc comment: "This only exists if
// getUseMixedPrecision() returns true"). It is NOT what makes `mixed` exist
// as a type -- that is unconditional in all three modes.
//
// Every position read anywhere in this file (G2's K1/K3 brute-force AND
// G3's K0-K6 CSR pipeline) goes through exp025LoadPos()+exp025WrapDeltaM()+
// exp025FoldToCellM(), all operating on `mixed` scalars (never a `mixed3`/
// `mixed4` vector type -- those are not confirmed to exist, so this plugin
// defines its own plain-scalar math instead of guessing at unconfirmed
// vector-type names). In single/double mode `mixed`==`real`, so this is
// byte-for-byte identical to the pre-mixed-precision G2/G3 code (nothing
// about the already-qualified single/double behavior changes); only mixed
// mode gains real extra precision. Downstream model math (RBF/C2/typed MLP)
// stays `real`-typed throughout, unaffected -- only the "how far apart are
// these two atoms" geometry chain benefits from the wider type.
const char* kDeviceSource = R"CUDA(
// EXP-026 Patch A2 DRAFT (PLAN section 20.1) -- NOT YET BUILT/TESTED as of
// this edit. Consolidates the legacy per-kernel errorFlag path into the
// single DeviceStatusV1 status block, first-error-wins. Every previous
// `atomicExch(errorFlag, CODE)` call site becomes
// `exp026SetFirstError(status, CODE, STAGE)` so a later kernel in the same
// evaluation can never silently overwrite an earlier real error -- the OLD
// atomicExch-based semantics allowed exactly that (last write wins), which
// is the bug this consolidation fixes, not just a transport optimization.
extern "C" __global__ void exp026ResetSupportStatus(
        int* __restrict__ status, int epoch) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    status[EXP026_STATUS_ERROR_CODE] = EXP025_DEVICE_ERROR_OK;
    status[EXP026_STATUS_ERROR_STAGE] = EXP026_STAGE_NONE;
    status[EXP026_STATUS_ACTIVE_EDGES] = 0;
    status[EXP026_STATUS_MAX_NEIGHBORS] = 0;
    status[EXP026_STATUS_UNIQUE_ENVIRONMENTS] = 0;
    status[EXP026_STATUS_CANDIDATES] = 0;
    status[EXP026_STATUS_EPOCH] = epoch;
    status[EXP026_STATUS_RESERVED] = 0;
}

// First-error-wins: only the FIRST thread to successfully CAS the error
// code away from OK gets to also write the stage -- every later caller
// (same kernel, later kernel, doesn't matter) sees old != OK and does
// nothing, so the stage can never be overwritten by a second, unrelated
// fault. atomicCAS already implies the ordering guarantee for status[ERROR_
// CODE] itself; __threadfence() additionally makes the ERROR_STAGE write
// visible to any thread that later reads ERROR_CODE via a plain load (the
// only such reader is the host, after the device stream has been
// synchronized by the one consolidated D2H, so this is a belt-and-suspenders
// correctness note more than a load-bearing requirement, matching PLAN
// section 20.1.2's own snippet).
__device__ inline void exp026SetFirstError(int* __restrict__ status, int code, int stage) {
    if (atomicCAS(&status[EXP026_STATUS_ERROR_CODE], EXP025_DEVICE_ERROR_OK, code) == EXP025_DEVICE_ERROR_OK) {
        status[EXP026_STATUS_ERROR_STAGE] = stage;
        __threadfence();
    }
}

__device__ inline void exp026MarkUniqueEnvironment(
        int envId, int* __restrict__ uniqueEnvEpoch, int currentEpoch,
        int* __restrict__ status) {
    int previous = atomicExch(&uniqueEnvEpoch[envId], currentEpoch);
    if (previous != currentEpoch) {
        int count = atomicAdd(&status[EXP026_STATUS_UNIQUE_ENVIRONMENTS], 1) + 1;
        if (count > MAX_ENVIRONMENT_ATOMS)
            exp026SetFirstError(status, EXP025_DEVICE_ERROR_UNIQUE_ENV_OVERFLOW, EXP026_STAGE_COMPUTE_Q);
    }
}

__device__ inline void exp026CommitAnchorCounts(int activeNeighbors, int* __restrict__ status) {
    int total = atomicAdd(&status[EXP026_STATUS_ACTIVE_EDGES], activeNeighbors) + activeNeighbors;
    atomicMax(&status[EXP026_STATUS_MAX_NEIGHBORS], activeNeighbors);
    if (activeNeighbors > MAX_NEIGHBORS_PER_LIGAND)
        exp026SetFirstError(status, EXP025_DEVICE_ERROR_NEIGHBOR_OVERFLOW, EXP026_STAGE_COMPUTE_Q);
    if (total > MAX_ACTIVE_EDGES)
        exp026SetFirstError(status, EXP025_DEVICE_ERROR_EDGE_OVERFLOW, EXP026_STAGE_COMPUTE_Q);
}

__device__ inline real exp025QuinticC2(real r, real inner, real outer) {
    if (r <= inner) return (real) 1;
    if (r >= outer) return (real) 0;
    real x = (r - inner) / (outer - inner);
    real x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
    return (real) 1 - (real) 10 * x3 + (real) 15 * x4 - (real) 6 * x5;
}

__device__ inline real exp025QuinticC2Grad(real r, real inner, real outer) {
    if (r <= inner || r >= outer) return (real) 0;
    real x = (r - inner) / (outer - inner);
    real x2 = x * x, x3 = x2 * x, x4 = x3 * x;
    real dTransition = -(real) 30 * x2 + (real) 60 * x3 - (real) 30 * x4;
    return dTransition / (outer - inner);
}

// Reconstructs atom `atomIdx`'s position at the widest precision this
// Context actually offers. See file header for the single/mixed/double
// behavior table. `posqCorrection` is always passed as a real argument
// (uniform kernel signatures, no ifdef'd parameter lists at call sites);
// it is simply never dereferenced when !USE_MIXED_PRECISION, and the host
// side passes a harmless already-valid pointer (posq itself) as a
// placeholder in that case -- see CudaLocalManyBodyResidualKernels.cpp's
// addPosArgs() helper.
__device__ inline void exp025LoadPos(int atomIdx, const real4* __restrict__ posq,
        const real4* __restrict__ posqCorrection, mixed* outX, mixed* outY, mixed* outZ) {
    real4 p = posq[atomIdx];
#if USE_MIXED_PRECISION
    real4 c = posqCorrection[atomIdx];
    *outX = (mixed) p.x + (mixed) c.x;
    *outY = (mixed) p.y + (mixed) c.y;
    *outZ = (mixed) p.z + (mixed) c.z;
#else
    *outX = (mixed) p.x;
    *outY = (mixed) p.y;
    *outZ = (mixed) p.z;
#endif
}

// Round-to-nearest with explicit half-box tie detection, at `mixed`
// precision. Deliberately does NOT write straight to the global error
// flag; the caller only escalates the tie for pairs actually inside
// outer_cutoff. This is NOT "there are too many checks so skip most of
// them" -- it is geometrically exact given the box safety precondition
// this kernel enforces before ever launching (see the minimum-face-height
// check in execute()): once the box's minimum periodic face height exceeds
// 2*r_cut, a genuine half-box MIC tie (a pair sitting near a periodic
// boundary along some axis) and an active, within-cutoff pair are mutually
// exclusive by construction -- a pair close enough to be a tie candidate
// along an axis is, by the face-height bound, necessarily farther than
// r_cut away in true minimum-image distance. So a tie on a far-away
// candidate is not a "probably harmless, ignore it" judgment call; it
// CANNOT be an active edge, full stop, and cannot affect q, energy, or
// force regardless of which periodic image was chosen for it. Only a tie
// that somehow coexists with r < outer_cutoff would indicate the safety
// precondition itself was violated, which is exactly what gets escalated
// below. The tie epsilon (EXP025_HALF_BOX_TIE_EPSILON) is kept the SAME
// across all three precisions rather than tightened for mixed/double: a
// looser epsilon only flags ties MORE often (still always geometrically
// safe per the argument above), never fewer, so it cannot hide a real
// ambiguity -- tightening it is a pure diagnostics refinement, not a
// correctness requirement, and out of scope here.
__device__ inline mixed exp025RoundWithTieCheckM(mixed x, int* localTieFlag) {
    mixed fl = floor(x);
    mixed frac = x - fl;
    if (fabs(frac - (mixed) 0.5) < (mixed) EXP025_HALF_BOX_TIE_EPSILON)
        *localTieFlag = 1;
    return floor(x + (mixed) 0.5);
}

// Sequential z->y->x reduced-form minimum-image wrap, matching OpenMM's own
// triclinic convention (same box vectors / same physical meaning as the
// platform's own APPLY_PERIODIC_TO_DELTA macro), reimplemented by hand (not
// via that macro) so the intermediate fractional values are inspectable for
// tie detection. Operates on plain `mixed` scalars, not a vector-type
// struct (see file header).
__device__ inline void exp025WrapDeltaM(mixed dx, mixed dy, mixed dz,
        real4 boxVecX, real4 boxVecY, real4 boxVecZ, real4 invBoxSize, int* localTieFlag,
        mixed* outDx, mixed* outDy, mixed* outDz) {
    mixed scale3 = exp025RoundWithTieCheckM(dz * (mixed) invBoxSize.z, localTieFlag);
    dx -= scale3 * (mixed) boxVecZ.x;
    dy -= scale3 * (mixed) boxVecZ.y;
    dz -= scale3 * (mixed) boxVecZ.z;
    mixed scale2 = exp025RoundWithTieCheckM(dy * (mixed) invBoxSize.y, localTieFlag);
    dx -= scale2 * (mixed) boxVecY.x;
    dy -= scale2 * (mixed) boxVecY.y;
    mixed scale1 = exp025RoundWithTieCheckM(dx * (mixed) invBoxSize.x, localTieFlag);
    dx -= scale1 * (mixed) boxVecX.x;
    *outDx = dx; *outDy = dy; *outDz = dz;
}

__device__ inline real exp025Silu(real x) {
    return x / ((real) 1 + EXP(-x));
}
__device__ inline real exp025SiluGrad(real x) {
    real s = (real) 1 / ((real) 1 + EXP(-x));
    return s * ((real) 1 + x * ((real) 1 - s));
}

// mlpBase points at the start of this ligand type's EXP025_MLP_STRIDE-length
// flat parameter block (see r1_model_layout.h for the offset contract).
__device__ inline void exp025EvalMlpWithGrad(const real* __restrict__ mlpBase, real x, real* outValue, real* outGrad) {
    real a0[EXP025_HIDDEN_RHO], da0[EXP025_HIDDEN_RHO];
    for (int k = 0; k < EXP025_HIDDEN_RHO; k++) {
        real w0 = mlpBase[EXP025_MLP_OFFSET_W0 + k];
        real b0 = mlpBase[EXP025_MLP_OFFSET_B0 + k];
        real h0 = w0 * x + b0;
        a0[k] = exp025Silu(h0);
        da0[k] = exp025SiluGrad(h0) * w0;
    }
    real a2[EXP025_HIDDEN_RHO], da2[EXP025_HIDDEN_RHO];
    for (int o = 0; o < EXP025_HIDDEN_RHO; o++) {
        real h = mlpBase[EXP025_MLP_OFFSET_B2 + o];
        real dh = (real) 0;
        for (int k = 0; k < EXP025_HIDDEN_RHO; k++) {
            real w2 = mlpBase[EXP025_MLP_OFFSET_W2 + o * EXP025_HIDDEN_RHO + k];
            h += w2 * a0[k];
            dh += w2 * da0[k];
        }
        a2[o] = exp025Silu(h);
        da2[o] = exp025SiluGrad(h) * dh;
    }
    real value = mlpBase[EXP025_MLP_OFFSET_B4];
    real grad = (real) 0;
    for (int k = 0; k < EXP025_HIDDEN_RHO; k++) {
        real w4 = mlpBase[EXP025_MLP_OFFSET_W4 + k];
        value += w4 * a2[k];
        grad += w4 * da2[k];
    }
    *outValue = value;
    *outGrad = grad;
}

// ============================== K1: compute q[NUM_LIGANDS] ==============================
extern "C" __global__ void exp025ComputeQ(
        const real4* __restrict__ posq, const real4* __restrict__ posqCorrection,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 invPeriodicBoxSize,
        const int* __restrict__ anchorDeviceIds,
        const int* __restrict__ typeByDevice,
        const int* __restrict__ isLigandByDevice,
        const real* __restrict__ radialCenters,
        real radialWidth,
        const real* __restrict__ pairWeight,
        real innerCutoffAngstrom,
        real outerCutoffAngstrom,
        real* __restrict__ q,
        int* __restrict__ neighborCount,
        int* __restrict__ uniqueEnvEpoch,
        int currentEpoch,
        int* __restrict__ deviceStatus) {
    int anchor = blockIdx.x;
    int ligandDeviceId = anchorDeviceIds[anchor];
    mixed ligX, ligY, ligZ;
    exp025LoadPos(ligandDeviceId, posq, posqCorrection, &ligX, &ligY, &ligZ);
    int ligandType = typeByDevice[ligandDeviceId];

    real localSum = (real) 0;
    int localNeighborCount = 0;

    for (int envId = threadIdx.x; envId < NUM_ATOMS; envId += blockDim.x) {
        if (isLigandByDevice[envId]) continue;
        mixed envX, envY, envZ;
        exp025LoadPos(envId, posq, posqCorrection, &envX, &envY, &envZ);
        int localTie = 0;
        mixed dx, dy, dz;
        exp025WrapDeltaM(envX - ligX, envY - ligY, envZ - ligZ, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ,
                          invPeriodicBoxSize, &localTie, &dx, &dy, &dz);
        mixed rNmM = sqrt(dx * dx + dy * dy + dz * dz);
        real rAngstrom = (real) ((mixed) 10 * rNmM);
        if (rAngstrom < (real) EXP025_MIN_DISTANCE_ANGSTROM) {
            exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_MIN_DISTANCE, EXP026_STAGE_COMPUTE_Q);
            continue;
        }
        if (rAngstrom < outerCutoffAngstrom) {
            if (localTie) exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_HALF_BOX_TIE, EXP026_STAGE_COMPUTE_Q);
            localNeighborCount++;
            exp026MarkUniqueEnvironment(envId, uniqueEnvEpoch, currentEpoch, deviceStatus);
            int envType = typeByDevice[envId];
            real c2 = exp025QuinticC2(rAngstrom, innerCutoffAngstrom, outerCutoffAngstrom);
            real g = (real) 0;
            const real* w = pairWeight + (size_t) (ligandType * NUM_TYPES + envType) * NUM_RADIAL_BASIS;
            for (int p = 0; p < NUM_RADIAL_BASIS; p++) {
                real diff = rAngstrom - radialCenters[p];
                real z = diff / radialWidth;
                real gp = EXP(-(real) 0.5 * z * z);
                g += w[p] * gp;
            }
            localSum += c2 * g;
        }
    }

    __shared__ real sumBuffer[CUDA_BLOCK_SIZE];
    __shared__ int countBuffer[CUDA_BLOCK_SIZE];
    sumBuffer[threadIdx.x] = localSum;
    countBuffer[threadIdx.x] = localNeighborCount;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sumBuffer[threadIdx.x] += sumBuffer[threadIdx.x + offset];
            countBuffer[threadIdx.x] += countBuffer[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (!isfinite(sumBuffer[0]))
            exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_NONFINITE, EXP026_STAGE_COMPUTE_Q);
        q[anchor] = sumBuffer[0];
        neighborCount[anchor] = countBuffer[0];
        exp026CommitAnchorCounts(countBuffer[0], deviceStatus);
    }
}

// ============================== K2: 41-anchor readout, entirely on GPU ==============================
extern "C" __global__ void exp025Readout(
        const int* __restrict__ anchorDeviceIds,
        const int* __restrict__ typeByDevice,
        const real* __restrict__ q,
        const real* __restrict__ typedMlpFlat,
        real bMaxReduced,
        real* __restrict__ dBdq,
        real* __restrict__ energyScratch,
        int* __restrict__ deviceStatus) {
    int anchor = threadIdx.x;  // single block, NUM_LIGANDS threads
    int ligandDeviceId = anchorDeviceIds[anchor];
    int ligandType = typeByDevice[ligandDeviceId];
    const real* mlpBase = typedMlpFlat + (size_t) ligandType * EXP025_MLP_STRIDE;

    real qValue = q[anchor];
    real rhoQ, rhoQGrad, rhoZero, rhoZeroGradUnused;
    exp025EvalMlpWithGrad(mlpBase, qValue, &rhoQ, &rhoQGrad);
    exp025EvalMlpWithGrad(mlpBase, (real) 0, &rhoZero, &rhoZeroGradUnused);
    real perLigand = rhoQ - rhoZero;

    __shared__ real totalS;
    __shared__ real sech2Shared;
    if (threadIdx.x == 0) totalS = (real) 0;
    __syncthreads();
    atomicAdd(&totalS, perLigand);
    __syncthreads();

    if (threadIdx.x == 0) {
        real tanhVal = tanh(totalS / bMaxReduced);
        real bReduced = bMaxReduced * tanhVal;
        sech2Shared = (real) 1 - tanhVal * tanhVal;
        if (!isfinite(bReduced) || !isfinite(sech2Shared))
            exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_NONFINITE, EXP026_STAGE_READOUT);
        energyScratch[0] = bReduced;
    }
    __syncthreads();

    // A2 fix (found while consolidating error reporting, PLAN 20.1.3): the
    // check above only covers the shared bReduced/sech2Shared scalars, NOT
    // this per-anchor product -- rhoQGrad itself (from the MLP evaluated at
    // this anchor's own q) could be non-finite even when bReduced/sech2Shared
    // are both finite, and that would previously have gone undetected here,
    // only surfacing later via scatterForce[FromCSR]'s own `coeff` check.
    // Checking it HERE is required before K2's error status can safely be
    // treated as "covers everything scatter checks" (the precondition for
    // moving the consolidated status harvest to before scatter runs).
    real dBdqValue = sech2Shared * rhoQGrad;
    if (!isfinite(dBdqValue))
        exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_NONFINITE, EXP026_STAGE_READOUT);
    dBdq[anchor] = dBdqValue;
}

// ============================== K3: brute-force conservative force scatter ==============================
extern "C" __global__ void exp025ScatterForce(
        const real4* __restrict__ posq, const real4* __restrict__ posqCorrection,
        real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ, real4 invPeriodicBoxSize,
        const int* __restrict__ anchorDeviceIds,
        const int* __restrict__ typeByDevice,
        const int* __restrict__ isLigandByDevice,
        const real* __restrict__ radialCenters,
        real radialWidth,
        const real* __restrict__ pairWeight,
        real innerCutoffAngstrom,
        real outerCutoffAngstrom,
        const real* __restrict__ dBdq,
        real kBT,
        int paddedNumAtoms,
        unsigned long long* __restrict__ forceBuffer,
        int* __restrict__ deviceStatus) {
    // A2 partial-force safety (PLAN 20.1.3): self-gated on ANY fatal status
    // already set by an earlier kernel THIS evaluation (K1/K2/K4/K5 all run
    // strictly before this kernel on the same CUDA stream, so their status
    // writes are already visible here without any host round-trip). Without
    // this, removing the host-side check-before-scatter (this draft's A2
    // consolidation) would let scatter unconditionally commit atomicAdd's
    // into the REAL OpenMM force buffer even after an upstream stage already
    // detected a fatal error -- exactly the "partial residual force" bug
    // PLAN 20.1.3 requires closing, not something safe to skip.
    if (deviceStatus[EXP026_STATUS_ERROR_CODE] != EXP025_DEVICE_ERROR_OK) return;

    int anchor = blockIdx.x;
    int ligandDeviceId = anchorDeviceIds[anchor];
    mixed ligX, ligY, ligZ;
    exp025LoadPos(ligandDeviceId, posq, posqCorrection, &ligX, &ligY, &ligZ);
    int ligandType = typeByDevice[ligandDeviceId];
    real dBdqAnchor = dBdq[anchor];

    real ligandForceX = (real) 0, ligandForceY = (real) 0, ligandForceZ = (real) 0;

    for (int envId = threadIdx.x; envId < NUM_ATOMS; envId += blockDim.x) {
        if (isLigandByDevice[envId]) continue;
        mixed envX, envY, envZ;
        exp025LoadPos(envId, posq, posqCorrection, &envX, &envY, &envZ);
        int localTie = 0;
        mixed dx, dy, dz;
        exp025WrapDeltaM(envX - ligX, envY - ligY, envZ - ligZ, periodicBoxVecX, periodicBoxVecY, periodicBoxVecZ,
                          invPeriodicBoxSize, &localTie, &dx, &dy, &dz);
        mixed rNmM = sqrt(dx * dx + dy * dy + dz * dz);
        real rAngstrom = (real) ((mixed) 10 * rNmM);
        if (rAngstrom < (real) EXP025_MIN_DISTANCE_ANGSTROM) {
            exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_MIN_DISTANCE, EXP026_STAGE_FORCE_SCATTER);
            continue;
        }
        if (rAngstrom < outerCutoffAngstrom) {
            if (localTie) exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_HALF_BOX_TIE, EXP026_STAGE_FORCE_SCATTER);
            int envType = typeByDevice[envId];
            real c2 = exp025QuinticC2(rAngstrom, innerCutoffAngstrom, outerCutoffAngstrom);
            real dc2 = exp025QuinticC2Grad(rAngstrom, innerCutoffAngstrom, outerCutoffAngstrom);
            real g = (real) 0, dg = (real) 0;
            const real* w = pairWeight + (size_t) (ligandType * NUM_TYPES + envType) * NUM_RADIAL_BASIS;
            for (int p = 0; p < NUM_RADIAL_BASIS; p++) {
                real diff = rAngstrom - radialCenters[p];
                real z = diff / radialWidth;
                real gp = EXP(-(real) 0.5 * z * z);
                g += w[p] * gp;
                dg += w[p] * gp * (-diff / (radialWidth * radialWidth));
            }
            real dEdgeQ_dr = dc2 * g + c2 * dg;
            real dB_dr = dBdqAnchor * dEdgeQ_dr;
            // d_hat = (env - lig)/r ; gradient(lig)=-10*dB_dr*d_hat, gradient(env)=+10*dB_dr*d_hat
            // physical force(lig)=+10*kBT*dB_dr*d_hat, physical force(env)=-10*kBT*dB_dr*d_hat
            mixed invRM = (mixed) 1 / rNmM;
            real ux = (real) (dx * invRM), uy = (real) (dy * invRM), uz = (real) (dz * invRM);
            real coeff = (real) 10 * kBT * dB_dr;
            if (!isfinite(coeff)) exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_NONFINITE, EXP026_STAGE_FORCE_SCATTER);
            ligandForceX += coeff * ux;
            ligandForceY += coeff * uy;
            ligandForceZ += coeff * uz;
            atomicAdd(&forceBuffer[envId], (unsigned long long) realToFixedPoint(-coeff * ux));
            atomicAdd(&forceBuffer[envId + paddedNumAtoms], (unsigned long long) realToFixedPoint(-coeff * uy));
            atomicAdd(&forceBuffer[envId + 2 * paddedNumAtoms], (unsigned long long) realToFixedPoint(-coeff * uz));
        }
    }

    __shared__ real sumFX, sumFY, sumFZ;
    if (threadIdx.x == 0) { sumFX = (real) 0; sumFY = (real) 0; sumFZ = (real) 0; }
    __syncthreads();
    atomicAdd(&sumFX, ligandForceX);
    atomicAdd(&sumFY, ligandForceY);
    atomicAdd(&sumFZ, ligandForceZ);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&forceBuffer[ligandDeviceId], (unsigned long long) realToFixedPoint(sumFX));
        atomicAdd(&forceBuffer[ligandDeviceId + paddedNumAtoms], (unsigned long long) realToFixedPoint(sumFY));
        atomicAdd(&forceBuffer[ligandDeviceId + 2 * paddedNumAtoms], (unsigned long long) realToFixedPoint(sumFZ));
    }
}
)CUDA";

// ============================== G3: local CSR/Verlet (linked-cell) pipeline ==============================
//
// K0 checkDisplacement -> K1 clearCellHeads -> K2 binEnvironmentAtoms ->
// K3 countCandidates -> K4 prefixSumOffsets -> K5 fillCandidates -> K6
// computeQFromCSR/scatterForceFromCSR. K1-K5 (and the copy/clear kernels
// below) all self-gate on *rebuildFlag at kernel entry -- they are launched
// EVERY step regardless, but no-op (immediate return) unless a rebuild is
// actually needed, so no host round-trip decides whether to launch them.
// rebuildFlag itself is a device-owned int that is NEVER downloaded to host;
// only the existing errorFlag (already downloaded every step, see G2) is
// used to fail closed on overflow.
//
// Shares exp025QuinticC2[Grad]/exp025LoadPos/exp025WrapDeltaM/exp025Silu[Grad]
// with the G2 source above (same compileProgram() call, same translation
// unit) -- no redefinition. Adds one new shared __device__ helper
// (exp025EdgeQAndGrad, the per-edge C2*RBF value+radial-derivative) used by
// BOTH new CSR kernels below; this is ordinary same-platform code reuse
// (see the .h file header comment), not a cross-check duplication concern.
//
// lastPositions (K0/copyLastPositions) is `mixed`-typed on the device side
// (see file header table) -- the host allocates it as double when this
// Context is mixed OR double precision, float when single, exactly
// matching what `mixed` resolves to in each mode (see addPosArgs()/
// initialize() in the .cpp for the host-side allocation).
const char* kDeviceSourceCSR = R"CUDA(
__device__ inline void exp025EdgeQAndGrad(real r, int ligandType, int envType,
        const real* __restrict__ radialCenters, real radialWidth, const real* __restrict__ pairWeight,
        real inner, real outer, real* outEdgeQ, real* outDEdgeQDr) {
    real c2 = exp025QuinticC2(r, inner, outer);
    real dc2 = exp025QuinticC2Grad(r, inner, outer);
    real g = (real) 0, dg = (real) 0;
    const real* w = pairWeight + (size_t) (ligandType * NUM_TYPES + envType) * NUM_RADIAL_BASIS;
    for (int p = 0; p < NUM_RADIAL_BASIS; p++) {
        real diff = r - radialCenters[p];
        real z = diff / radialWidth;
        real gp = EXP(-(real) 0.5 * z * z);
        g += w[p] * gp;
        dg += w[p] * gp * (-diff / (radialWidth * radialWidth));
    }
    *outEdgeQ = c2 * g;
    *outDEdgeQDr = dc2 * g + c2 * dg;
}

// Folds an absolute (mixed-precision) position into the primary unit cell
// via the same sequential z->y->x reduction as exp025WrapDeltaM above
// (restricted triclinic: A=(Ax,0,0), B=(Bx,By,0), C=(Cx,Cy,Cz)), using
// floor() to fold into [0, boxVec) rather than round() to wrap a delta into
// [-half,half). Returns a cell index per axis in [0,nCellsAxis).
__device__ inline void exp025FoldToCellM(mixed px, mixed py, mixed pz,
        real4 boxVecX, real4 boxVecY, real4 boxVecZ,
        int nCellsX, int nCellsY, int nCellsZ, int* outCx, int* outCy, int* outCz) {
    mixed nz = floor(pz / (mixed) boxVecZ.z);
    px -= nz * (mixed) boxVecZ.x;
    py -= nz * (mixed) boxVecZ.y;
    pz -= nz * (mixed) boxVecZ.z;

    mixed ny = floor(py / (mixed) boxVecY.y);
    px -= ny * (mixed) boxVecY.x;
    py -= ny * (mixed) boxVecY.y;

    mixed nx = floor(px / (mixed) boxVecX.x);
    px -= nx * (mixed) boxVecX.x;

    mixed fx = px / (mixed) boxVecX.x, fy = py / (mixed) boxVecY.y, fz = pz / (mixed) boxVecZ.z;
    int cx = (int) (fx * (mixed) nCellsX);
    int cy = (int) (fy * (mixed) nCellsY);
    int cz = (int) (fz * (mixed) nCellsZ);
    if (cx >= nCellsX) cx = nCellsX - 1; if (cx < 0) cx = 0;
    if (cy >= nCellsY) cy = nCellsY - 1; if (cy < 0) cy = 0;
    if (cz >= nCellsZ) cz = nCellsZ - 1; if (cz < 0) cz = 0;
    *outCx = cx; *outCy = cy; *outCz = cz;
}

// ---- K0: per-atom displacement since last rebuild -> device-owned rebuildFlag (no host download) ----
extern "C" __global__ void exp025CheckDisplacement(
        const real4* __restrict__ posq, const real4* __restrict__ posqCorrection,
        const mixed* __restrict__ lastPositions,
        mixed halfSkinNmSquared, int* __restrict__ rebuildFlag) {
    // Deliberately NOT minimum-image: a raw (unwrapped) coordinate delta can
    // only ever be >= the true physical displacement (a periodic-wrap event
    // between samples inflates it, never deflates it below the true value),
    // so this is a conservative, safe trigger -- may rebuild a little more
    // often near a periodic boundary, never misses a real rebuild.
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS; i += blockDim.x * gridDim.x) {
        mixed nowX, nowY, nowZ;
        exp025LoadPos(i, posq, posqCorrection, &nowX, &nowY, &nowZ);
        mixed lastX = lastPositions[3 * i], lastY = lastPositions[3 * i + 1], lastZ = lastPositions[3 * i + 2];
        mixed dx = nowX - lastX, dy = nowY - lastY, dz = nowZ - lastZ;
        if (dx * dx + dy * dy + dz * dz > halfSkinNmSquared)
            atomicOr(rebuildFlag, 1);
    }
}

// ---- K1: clear cell heads (self-gated on rebuildFlag) ----
extern "C" __global__ void exp025ClearCellHeads(int* __restrict__ cellHead, int nCells, const int* __restrict__ rebuildFlag) {
    if (*rebuildFlag == 0) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nCells; i += blockDim.x * gridDim.x)
        cellHead[i] = -1;
}

// ---- K2: bin environment atoms into a per-cell linked list via atomicExch (no sort needed) ----
extern "C" __global__ void exp025BinEnvironmentAtoms(
        const real4* __restrict__ posq, const real4* __restrict__ posqCorrection,
        real4 boxVecX, real4 boxVecY, real4 boxVecZ,
        const int* __restrict__ isLigandByDevice, int nCellsX, int nCellsY, int nCellsZ,
        int* __restrict__ cellHead, int* __restrict__ nextAtom, const int* __restrict__ rebuildFlag) {
    if (*rebuildFlag == 0) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS; i += blockDim.x * gridDim.x) {
        if (isLigandByDevice[i]) continue;  // only ligand-environment cross edges are in scope (PLAN section 3.1)
        mixed px, py, pz;
        exp025LoadPos(i, posq, posqCorrection, &px, &py, &pz);
        int cx, cy, cz;
        exp025FoldToCellM(px, py, pz, boxVecX, boxVecY, boxVecZ, nCellsX, nCellsY, nCellsZ, &cx, &cy, &cz);
        int cellIdx = (cz * nCellsY + cy) * nCellsX + cx;
        int old = atomicExch(&cellHead[cellIdx], i);
        nextAtom[i] = old;
    }
}

// ---- K3: count <r_list candidates per anchor via a 27-cell stencil (one block/anchor, one thread/neighbor-cell) ----
extern "C" __global__ void exp025CountCandidates(
        const real4* __restrict__ posq, const real4* __restrict__ posqCorrection,
        real4 boxVecX, real4 boxVecY, real4 boxVecZ, real4 invBoxSize,
        const int* __restrict__ anchorDeviceIds, int nCellsX, int nCellsY, int nCellsZ,
        const int* __restrict__ cellHead, const int* __restrict__ nextAtom,
        real rListAngstrom, const int* __restrict__ rebuildFlag,
        int* __restrict__ anchorCandidateCount) {
    if (*rebuildFlag == 0) return;
    int anchor = blockIdx.x;
    mixed ligX, ligY, ligZ;
    exp025LoadPos(anchorDeviceIds[anchor], posq, posqCorrection, &ligX, &ligY, &ligZ);
    __shared__ int localCount;
    if (threadIdx.x == 0) localCount = 0;
    __syncthreads();
    if (threadIdx.x < 27) {
        int dx = threadIdx.x / 9 - 1, dy = (threadIdx.x / 3) % 3 - 1, dz = threadIdx.x % 3 - 1;
        int cx0, cy0, cz0;
        exp025FoldToCellM(ligX, ligY, ligZ, boxVecX, boxVecY, boxVecZ, nCellsX, nCellsY, nCellsZ, &cx0, &cy0, &cz0);
        int cx = ((cx0 + dx) % nCellsX + nCellsX) % nCellsX;
        int cy = ((cy0 + dy) % nCellsY + nCellsY) % nCellsY;
        int cz = ((cz0 + dz) % nCellsZ + nCellsZ) % nCellsZ;
        int atomId = cellHead[(cz * nCellsY + cy) * nCellsX + cx];
        int myCount = 0;
        while (atomId != -1) {
            mixed envX, envY, envZ;
            exp025LoadPos(atomId, posq, posqCorrection, &envX, &envY, &envZ);
            int localTie = 0;
            mixed wdx, wdy, wdz;
            exp025WrapDeltaM(envX - ligX, envY - ligY, envZ - ligZ, boxVecX, boxVecY, boxVecZ, invBoxSize, &localTie, &wdx, &wdy, &wdz);
            mixed rNmM = sqrt(wdx * wdx + wdy * wdy + wdz * wdz);
            real rAngstrom = (real) ((mixed) 10 * rNmM);
            if (rAngstrom < rListAngstrom) myCount++;
            atomId = nextAtom[atomId];
        }
        atomicAdd(&localCount, myCount);
    }
    __syncthreads();
    if (threadIdx.x == 0) anchorCandidateCount[anchor] = localCount;
}

// ---- K4: serial prefix-sum of 41 counts -> 42 CSR offsets (trivially cheap at this size) ----
extern "C" __global__ void exp025PrefixSumOffsets(
        const int* __restrict__ anchorCandidateCount, int* __restrict__ anchorOffsets,
        int candidateListCapacity, const int* __restrict__ rebuildFlag, int* __restrict__ deviceStatus) {
    if (*rebuildFlag == 0) return;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int running = 0;
    anchorOffsets[0] = 0;
    bool overflow = false;
    for (int i = 0; i < NUM_LIGANDS; i++) {
        running += anchorCandidateCount[i];
        if (running > candidateListCapacity) overflow = true;
        anchorOffsets[i + 1] = running;
    }
    if (overflow) exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_CANDIDATE_OVERFLOW, EXP026_STAGE_REBUILD_PREFIX);
}

// ---- K5: fill the compact CSR (re-walks the SAME immutable cellHead/nextAtom as K3, so set+count agree) ----
extern "C" __global__ void exp025FillCandidates(
        const real4* __restrict__ posq, const real4* __restrict__ posqCorrection,
        real4 boxVecX, real4 boxVecY, real4 boxVecZ, real4 invBoxSize,
        const int* __restrict__ anchorDeviceIds, int nCellsX, int nCellsY, int nCellsZ,
        const int* __restrict__ cellHead, const int* __restrict__ nextAtom,
        real rListAngstrom, const int* __restrict__ rebuildFlag,
        const int* __restrict__ anchorOffsets, int* __restrict__ edgeAtoms,
        int* __restrict__ deviceStatus) {
    if (*rebuildFlag == 0) return;
    if (deviceStatus[EXP026_STATUS_ERROR_CODE] != EXP025_DEVICE_ERROR_OK) return;  // K4 already flagged total capacity overflow -- do not write
    int anchor = blockIdx.x;
    mixed ligX, ligY, ligZ;
    exp025LoadPos(anchorDeviceIds[anchor], posq, posqCorrection, &ligX, &ligY, &ligZ);
    int base = anchorOffsets[anchor];
    int capacity = anchorOffsets[anchor + 1] - base;
    __shared__ int writeIndex;
    if (threadIdx.x == 0) writeIndex = 0;
    __syncthreads();
    if (threadIdx.x < 27) {
        int dx = threadIdx.x / 9 - 1, dy = (threadIdx.x / 3) % 3 - 1, dz = threadIdx.x % 3 - 1;
        int cx0, cy0, cz0;
        exp025FoldToCellM(ligX, ligY, ligZ, boxVecX, boxVecY, boxVecZ, nCellsX, nCellsY, nCellsZ, &cx0, &cy0, &cz0);
        int cx = ((cx0 + dx) % nCellsX + nCellsX) % nCellsX;
        int cy = ((cy0 + dy) % nCellsY + nCellsY) % nCellsY;
        int cz = ((cz0 + dz) % nCellsZ + nCellsZ) % nCellsZ;
        int atomId = cellHead[(cz * nCellsY + cy) * nCellsX + cx];
        while (atomId != -1) {
            mixed envX, envY, envZ;
            exp025LoadPos(atomId, posq, posqCorrection, &envX, &envY, &envZ);
            int localTie = 0;
            mixed wdx, wdy, wdz;
            exp025WrapDeltaM(envX - ligX, envY - ligY, envZ - ligZ, boxVecX, boxVecY, boxVecZ, invBoxSize, &localTie, &wdx, &wdy, &wdz);
            mixed rNmM = sqrt(wdx * wdx + wdy * wdy + wdz * wdz);
            real rAngstrom = (real) ((mixed) 10 * rNmM);
            if (rAngstrom < rListAngstrom) {
                int slot = atomicAdd(&writeIndex, 1);
                // Defensive only: K3 and K5 replay the identical immutable
                // cellHead/nextAtom structure with the identical distance
                // test, so this should be unreachable by construction; kept
                // as fail-closed insurance rather than an unchecked write.
                if (slot < capacity) edgeAtoms[base + slot] = atomId;
                else exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_CANDIDATE_OVERFLOW, EXP026_STAGE_REBUILD_FILL);
            }
            atomId = nextAtom[atomId];
        }
    }
}

// ---- rebuild bookkeeping: snapshot positions, then clear the flag (strictly after everything above reads it) ----
extern "C" __global__ void exp025CopyLastPositions(
        const real4* __restrict__ posq, const real4* __restrict__ posqCorrection,
        mixed* __restrict__ lastPositions, const int* __restrict__ rebuildFlag) {
    if (*rebuildFlag == 0) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < NUM_ATOMS; i += blockDim.x * gridDim.x) {
        mixed px, py, pz;
        exp025LoadPos(i, posq, posqCorrection, &px, &py, &pz);
        lastPositions[3 * i] = px; lastPositions[3 * i + 1] = py; lastPositions[3 * i + 2] = pz;
    }
}
extern "C" __global__ void exp025ClearRebuildFlag(int* __restrict__ rebuildFlag) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *rebuildFlag = 0;
}

// ---- K6: q accumulation / conservative force scatter, reading the CSR instead of an all-N brute-force scan ----
extern "C" __global__ void exp025ComputeQFromCSR(
        const real4* __restrict__ posq, const real4* __restrict__ posqCorrection,
        real4 boxVecX, real4 boxVecY, real4 boxVecZ, real4 invBoxSize,
        const int* __restrict__ anchorDeviceIds, const int* __restrict__ typeByDevice,
        const int* __restrict__ anchorOffsets, const int* __restrict__ edgeAtoms,
        const real* __restrict__ radialCenters, real radialWidth, const real* __restrict__ pairWeight,
        real innerCutoffAngstrom, real outerCutoffAngstrom,
        real* __restrict__ q, int* __restrict__ neighborCount, int* __restrict__ uniqueEnvEpoch,
        int currentEpoch, int* __restrict__ deviceStatus) {
    int anchor = blockIdx.x;
    int ligandDeviceId = anchorDeviceIds[anchor];
    mixed ligX, ligY, ligZ;
    exp025LoadPos(ligandDeviceId, posq, posqCorrection, &ligX, &ligY, &ligZ);
    int ligandType = typeByDevice[ligandDeviceId];
    int base = anchorOffsets[anchor], count = anchorOffsets[anchor + 1] - base;

    real localSum = (real) 0;
    int localNeighborCount = 0;
    for (int k = threadIdx.x; k < count; k += blockDim.x) {
        int envId = edgeAtoms[base + k];
        mixed envX, envY, envZ;
        exp025LoadPos(envId, posq, posqCorrection, &envX, &envY, &envZ);
        int localTie = 0;
        mixed dx, dy, dz;
        exp025WrapDeltaM(envX - ligX, envY - ligY, envZ - ligZ, boxVecX, boxVecY, boxVecZ, invBoxSize, &localTie, &dx, &dy, &dz);
        mixed rNmM = sqrt(dx * dx + dy * dy + dz * dz);
        real rAngstrom = (real) ((mixed) 10 * rNmM);
        if (rAngstrom < (real) EXP025_MIN_DISTANCE_ANGSTROM) {
            exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_MIN_DISTANCE, EXP026_STAGE_COMPUTE_Q);
            continue;
        }
        if (rAngstrom < outerCutoffAngstrom) {
            if (localTie) exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_HALF_BOX_TIE, EXP026_STAGE_COMPUTE_Q);
            localNeighborCount++;
            exp026MarkUniqueEnvironment(envId, uniqueEnvEpoch, currentEpoch, deviceStatus);
            int envType = typeByDevice[envId];
            real edgeQ, dEdgeQDr;
            exp025EdgeQAndGrad(rAngstrom, ligandType, envType, radialCenters, radialWidth, pairWeight,
                                innerCutoffAngstrom, outerCutoffAngstrom, &edgeQ, &dEdgeQDr);
            localSum += edgeQ;
        }
    }
    __shared__ real sumBuffer[CUDA_BLOCK_SIZE];
    __shared__ int countBuffer[CUDA_BLOCK_SIZE];
    sumBuffer[threadIdx.x] = localSum;
    countBuffer[threadIdx.x] = localNeighborCount;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sumBuffer[threadIdx.x] += sumBuffer[threadIdx.x + offset];
            countBuffer[threadIdx.x] += countBuffer[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (!isfinite(sumBuffer[0])) exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_NONFINITE, EXP026_STAGE_COMPUTE_Q);
        q[anchor] = sumBuffer[0];
        neighborCount[anchor] = countBuffer[0];
        exp026CommitAnchorCounts(countBuffer[0], deviceStatus);
    }
}

extern "C" __global__ void exp025ScatterForceFromCSR(
        const real4* __restrict__ posq, const real4* __restrict__ posqCorrection,
        real4 boxVecX, real4 boxVecY, real4 boxVecZ, real4 invBoxSize,
        const int* __restrict__ anchorDeviceIds, const int* __restrict__ typeByDevice,
        const int* __restrict__ anchorOffsets, const int* __restrict__ edgeAtoms,
        const real* __restrict__ radialCenters, real radialWidth, const real* __restrict__ pairWeight,
        real innerCutoffAngstrom, real outerCutoffAngstrom,
        const real* __restrict__ dBdq, real kBT, int paddedNumAtoms,
        unsigned long long* __restrict__ forceBuffer, int* __restrict__ deviceStatus) {
    // A2 partial-force safety gate -- see exp025ScatterForce's identical
    // comment above; same reasoning applies verbatim to the CSR path.
    if (deviceStatus[EXP026_STATUS_ERROR_CODE] != EXP025_DEVICE_ERROR_OK) return;

    int anchor = blockIdx.x;
    int ligandDeviceId = anchorDeviceIds[anchor];
    mixed ligX, ligY, ligZ;
    exp025LoadPos(ligandDeviceId, posq, posqCorrection, &ligX, &ligY, &ligZ);
    int ligandType = typeByDevice[ligandDeviceId];
    real dBdqAnchor = dBdq[anchor];
    int base = anchorOffsets[anchor], count = anchorOffsets[anchor + 1] - base;

    real ligandForceX = (real) 0, ligandForceY = (real) 0, ligandForceZ = (real) 0;
    for (int k = threadIdx.x; k < count; k += blockDim.x) {
        int envId = edgeAtoms[base + k];
        mixed envX, envY, envZ;
        exp025LoadPos(envId, posq, posqCorrection, &envX, &envY, &envZ);
        int localTie = 0;
        mixed dx, dy, dz;
        exp025WrapDeltaM(envX - ligX, envY - ligY, envZ - ligZ, boxVecX, boxVecY, boxVecZ, invBoxSize, &localTie, &dx, &dy, &dz);
        mixed rNmM = sqrt(dx * dx + dy * dy + dz * dz);
        real rAngstrom = (real) ((mixed) 10 * rNmM);
        if (rAngstrom < (real) EXP025_MIN_DISTANCE_ANGSTROM) {
            exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_MIN_DISTANCE, EXP026_STAGE_FORCE_SCATTER);
            continue;
        }
        if (rAngstrom < outerCutoffAngstrom) {
            if (localTie) exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_HALF_BOX_TIE, EXP026_STAGE_FORCE_SCATTER);
            int envType = typeByDevice[envId];
            real edgeQ, dEdgeQDr;
            exp025EdgeQAndGrad(rAngstrom, ligandType, envType, radialCenters, radialWidth, pairWeight,
                                innerCutoffAngstrom, outerCutoffAngstrom, &edgeQ, &dEdgeQDr);
            real dB_dr = dBdqAnchor * dEdgeQDr;
            mixed invRM = (mixed) 1 / rNmM;
            real ux = (real) (dx * invRM), uy = (real) (dy * invRM), uz = (real) (dz * invRM);
            real coeff = (real) 10 * kBT * dB_dr;
            if (!isfinite(coeff)) exp026SetFirstError(deviceStatus, EXP025_DEVICE_ERROR_NONFINITE, EXP026_STAGE_FORCE_SCATTER);
            ligandForceX += coeff * ux; ligandForceY += coeff * uy; ligandForceZ += coeff * uz;
            atomicAdd(&forceBuffer[envId], (unsigned long long) realToFixedPoint(-coeff * ux));
            atomicAdd(&forceBuffer[envId + paddedNumAtoms], (unsigned long long) realToFixedPoint(-coeff * uy));
            atomicAdd(&forceBuffer[envId + 2 * paddedNumAtoms], (unsigned long long) realToFixedPoint(-coeff * uz));
        }
    }
    __shared__ real sumFX, sumFY, sumFZ;
    if (threadIdx.x == 0) { sumFX = (real) 0; sumFY = (real) 0; sumFZ = (real) 0; }
    __syncthreads();
    atomicAdd(&sumFX, ligandForceX);
    atomicAdd(&sumFY, ligandForceY);
    atomicAdd(&sumFZ, ligandForceZ);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(&forceBuffer[ligandDeviceId], (unsigned long long) realToFixedPoint(sumFX));
        atomicAdd(&forceBuffer[ligandDeviceId + paddedNumAtoms], (unsigned long long) realToFixedPoint(sumFY));
        atomicAdd(&forceBuffer[ligandDeviceId + 2 * paddedNumAtoms], (unsigned long long) realToFixedPoint(sumFZ));
    }
}
)CUDA";

string buildDefinesPrefix() {
    ostringstream defs;
    defs << "#define EXP025_HIDDEN_RHO " << EXP025_STRINGIFY(EXP025_HIDDEN_RHO) << "\n";
    defs << "#define EXP025_MLP_STRIDE " << EXP025_STRINGIFY(EXP025_MLP_STRIDE) << "\n";
    defs << "#define EXP025_MLP_OFFSET_W0 " << EXP025_STRINGIFY(EXP025_MLP_OFFSET_W0) << "\n";
    defs << "#define EXP025_MLP_OFFSET_B0 " << EXP025_STRINGIFY(EXP025_MLP_OFFSET_B0) << "\n";
    defs << "#define EXP025_MLP_OFFSET_W2 " << EXP025_STRINGIFY(EXP025_MLP_OFFSET_W2) << "\n";
    defs << "#define EXP025_MLP_OFFSET_B2 " << EXP025_STRINGIFY(EXP025_MLP_OFFSET_B2) << "\n";
    defs << "#define EXP025_MLP_OFFSET_W4 " << EXP025_STRINGIFY(EXP025_MLP_OFFSET_W4) << "\n";
    defs << "#define EXP025_MLP_OFFSET_B4 " << EXP025_STRINGIFY(EXP025_MLP_OFFSET_B4) << "\n";
    defs << "#define EXP025_MIN_DISTANCE_ANGSTROM " << EXP025_STRINGIFY(EXP025_MIN_DISTANCE_ANGSTROM) << "\n";
    defs << "#define EXP025_HALF_BOX_TIE_EPSILON " << EXP025_STRINGIFY(EXP025_HALF_BOX_TIE_EPSILON) << "\n";
    defs << "#define EXP025_DEVICE_ERROR_OK " << EXP025_STRINGIFY(EXP025_DEVICE_ERROR_OK) << "\n";
    defs << "#define EXP025_DEVICE_ERROR_HALF_BOX_TIE " << EXP025_STRINGIFY(EXP025_DEVICE_ERROR_HALF_BOX_TIE) << "\n";
    defs << "#define EXP025_DEVICE_ERROR_MIN_DISTANCE " << EXP025_STRINGIFY(EXP025_DEVICE_ERROR_MIN_DISTANCE) << "\n";
    defs << "#define EXP025_DEVICE_ERROR_EDGE_OVERFLOW " << EXP025_STRINGIFY(EXP025_DEVICE_ERROR_EDGE_OVERFLOW) << "\n";
    defs << "#define EXP025_DEVICE_ERROR_NEIGHBOR_OVERFLOW " << EXP025_STRINGIFY(EXP025_DEVICE_ERROR_NEIGHBOR_OVERFLOW) << "\n";
    defs << "#define EXP025_DEVICE_ERROR_UNIQUE_ENV_OVERFLOW " << EXP025_STRINGIFY(EXP025_DEVICE_ERROR_UNIQUE_ENV_OVERFLOW) << "\n";
    defs << "#define EXP025_DEVICE_ERROR_NONFINITE " << EXP025_STRINGIFY(EXP025_DEVICE_ERROR_NONFINITE) << "\n";
    defs << "#define EXP025_DEVICE_ERROR_CANDIDATE_OVERFLOW " << EXP025_STRINGIFY(EXP025_DEVICE_ERROR_CANDIDATE_OVERFLOW) << "\n";
    defs << "#define EXP025_DEVICE_ERROR_UNSUPPORTED_BOX " << EXP025_STRINGIFY(EXP025_DEVICE_ERROR_UNSUPPORTED_BOX) << "\n";
    defs << "#define EXP026_STATUS_ERROR_CODE " << EXP025_STRINGIFY(EXP026_STATUS_ERROR_CODE) << "\n";
    defs << "#define EXP026_STATUS_ERROR_STAGE " << EXP025_STRINGIFY(EXP026_STATUS_ERROR_STAGE) << "\n";
    defs << "#define EXP026_STATUS_ACTIVE_EDGES " << EXP025_STRINGIFY(EXP026_STATUS_ACTIVE_EDGES) << "\n";
    defs << "#define EXP026_STATUS_MAX_NEIGHBORS " << EXP025_STRINGIFY(EXP026_STATUS_MAX_NEIGHBORS) << "\n";
    defs << "#define EXP026_STATUS_UNIQUE_ENVIRONMENTS " << EXP025_STRINGIFY(EXP026_STATUS_UNIQUE_ENVIRONMENTS) << "\n";
    defs << "#define EXP026_STATUS_CANDIDATES " << EXP025_STRINGIFY(EXP026_STATUS_CANDIDATES) << "\n";
    defs << "#define EXP026_STATUS_EPOCH " << EXP025_STRINGIFY(EXP026_STATUS_EPOCH) << "\n";
    defs << "#define EXP026_STATUS_RESERVED " << EXP025_STRINGIFY(EXP026_STATUS_RESERVED) << "\n";
    defs << "#define EXP026_STAGE_NONE " << EXP025_STRINGIFY(EXP026_STAGE_NONE) << "\n";
    // A2: the other stage constants -- A1 only ever needed EXP026_STAGE_NONE
    // (for the reset kernel); every kernel's own exp026SetFirstError() call
    // needs its stage macro forwarded into the NVRTC device compilation too
    // (a #include of the host header does not do this -- these macros must
    // be re-stringified into the actual device source text, same as every
    // other constant above).
    defs << "#define EXP026_STAGE_REBUILD_PREFIX " << EXP025_STRINGIFY(EXP026_STAGE_REBUILD_PREFIX) << "\n";
    defs << "#define EXP026_STAGE_REBUILD_FILL " << EXP025_STRINGIFY(EXP026_STAGE_REBUILD_FILL) << "\n";
    defs << "#define EXP026_STAGE_COMPUTE_Q " << EXP025_STRINGIFY(EXP026_STAGE_COMPUTE_Q) << "\n";
    defs << "#define EXP026_STAGE_READOUT " << EXP025_STRINGIFY(EXP026_STAGE_READOUT) << "\n";
    defs << "#define EXP026_STAGE_FORCE_SCATTER " << EXP025_STRINGIFY(EXP026_STAGE_FORCE_SCATTER) << "\n";
    defs << "#define EXP026_STAGE_HOST_VALIDATE " << EXP025_STRINGIFY(EXP026_STAGE_HOST_VALIDATE) << "\n";
    return defs.str();
}

// Small precision-dispatch helpers -- CUDA kernel argument marshaling is
// exact about byte size/layout, so every scalar/vector argument must be
// added as the SAME width (float vs double) the compiled module expects.
struct Real4F { float x, y, z, w; };
struct Real4D { double x, y, z, w; };

// ---- EXP-028: bound/idx argument marshaling ----
// See CudaLocalManyBodyResidualKernels.h's addPosArgs() doc comment for the
// full story: addArg() PERMANENTLY APPENDS a slot rather than updating one,
// so every helper below now takes (bound, idx) -- bound==false (this
// kernel's first-ever invocation) still calls addArg() to lay the slot down
// for the first and only time; bound==true (every later invocation) calls
// setArg(idx, ...) against that same already-bound slot instead. Every call
// site is responsible for passing the SAME idx for the SAME logical
// argument on every invocation (guaranteed here because each kernel's
// argument-adding code always runs the identical sequence of these helper
// calls in the identical order every time it runs at all).
template <typename T>
void addScalarArg(ComputeKernel& kernel, bool bound, int idx, const T& value) {
    if (bound) kernel->setArg<T>(idx, value);
    else kernel->addArg<T>(value);
}

void addArrArg(ComputeKernel& kernel, bool bound, int idx, ArrayInterface& value) {
    if (bound) kernel->setArg(idx, value);
    else kernel->addArg(value);
}

void addRealArg(ComputeKernel& kernel, bool bound, int idx, bool useDouble, double value) {
    if (useDouble) addScalarArg<double>(kernel, bound, idx, value);
    else addScalarArg<float>(kernel, bound, idx, (float) value);
}

// `mixed` in the DEVICE-typed sense is double whenever the Context is mixed
// OR double precision, float only in genuine single precision -- this host
// helper adds a scalar argument matching that same rule (used for
// halfSkinNmSquared, which the K0 kernel receives as a `mixed` parameter).
void addMixedArg(ComputeKernel& kernel, bool bound, int idx, bool useWidePositionStorage, double value) {
    if (useWidePositionStorage) addScalarArg<double>(kernel, bound, idx, value);
    else addScalarArg<float>(kernel, bound, idx, (float) value);
}

void addReal4Arg(ComputeKernel& kernel, bool bound, int idx, bool useDouble, double x, double y, double z, double w) {
    if (useDouble) { Real4D v{x, y, z, w}; addScalarArg<Real4D>(kernel, bound, idx, v); }
    else { Real4F v{(float) x, (float) y, (float) z, (float) w}; addScalarArg<Real4F>(kernel, bound, idx, v); }
}

double downloadRealScalar(ComputeArray& arr, bool useDouble) {
    if (useDouble) {
        vector<double> data;
        arr.download(data);
        return data.at(0);
    }
    vector<float> data;
    arr.download(data);
    return (double) data.at(0);
}

void uploadRealArray(ComputeArray& arr, CudaContext& cu, bool useDouble, const vector<double>& hostData, const string& name) {
    if (useDouble) {
        arr.initialize<double>(cu, hostData.size(), name);
        arr.upload(hostData);
    } else {
        vector<float> converted(hostData.begin(), hostData.end());
        arr.initialize<float>(cu, hostData.size(), name);
        arr.upload(converted);
    }
}

}  // namespace

class CudaCalcLocalManyBodyResidualForceKernel::ReorderListenerImpl : public CudaContext::ReorderListener {
public:
    ReorderListenerImpl(CudaCalcLocalManyBodyResidualForceKernel& owner) : owner(owner) {}
    void execute() override { owner.onAtomsReordered(); }
private:
    CudaCalcLocalManyBodyResidualForceKernel& owner;
};

CudaCalcLocalManyBodyResidualForceKernel::~CudaCalcLocalManyBodyResidualForceKernel() {
    // Do NOT delete reorderListener here: ComputeContext::addReorderListener()
    // takes ownership and deletes it itself when the Context is destroyed
    // (see openmm/common/ComputeContext.h). Deleting it here too is a
    // double-free -- this crashed inside ~CudaContext() the first time.
}

void CudaCalcLocalManyBodyResidualForceKernel::onAtomsReordered() {
    deviceStateValid = false;
    // G3: device slots got reshuffled, so lastRebuildPositionsDevice (indexed
    // by OLD device slot) and the CSR's anchorDeviceIds/edgeAtoms (also OLD
    // device slots) are now meaningless -- force a full rebuild next step
    // rather than letting K0 compare current positions against stale slots.
    // This invalidation is precision-mode-agnostic: whatever `mixed`-typed
    // (float or double) values happen to be sitting in
    // lastRebuildPositionsDevice from before the reorder are simply never
    // read, because K0 is skipped entirely on the forced-rebuild step (see
    // execute()) -- posqCorrection itself is also reordered by OpenMM
    // consistently with posq on any atom reorder, so once a rebuild does
    // run, exp025LoadPos() reads self-consistent, current-ordering data.
    candidateListValid = false;
}

void CudaCalcLocalManyBodyResidualForceKernel::initialize(const System& system, const LocalManyBodyResidualForce& force) {
    int numParticles = system.getNumParticles();
    const vector<int>& ligandIds = force.getLigandTopologyIds();
    const vector<int>& types = force.getAtomTypeIndex();
    const vector<int>& vocab = force.getTypeVocabulary();

    if ((int) types.size() != numParticles)
        throw OpenMMException("LocalManyBodyResidualForce (CUDA): atomTypeIndex size must equal the System particle count");
    typeCount = (int) vocab.size();
    if (typeCount == 0) throw OpenMMException("LocalManyBodyResidualForce (CUDA): type vocabulary is empty");
    if (force.getNumTypes() != typeCount)
        throw OpenMMException("LocalManyBodyResidualForce (CUDA): number of typed MLPs must equal type vocabulary size");
    for (int t : types)
        if (t < 0 || t >= typeCount)
            throw OpenMMException("LocalManyBodyResidualForce (CUDA): atomTypeIndex entry outside the type vocabulary");

    set<int> ligandSet;
    for (int id : ligandIds) {
        if (id < 0 || id >= numParticles)
            throw OpenMMException("LocalManyBodyResidualForce (CUDA): ligand topology id outside the System");
        if (!ligandSet.insert(id).second)
            throw OpenMMException("LocalManyBodyResidualForce (CUDA): duplicate ligand topology id");
    }
    if (ligandIds.empty())
        throw OpenMMException("LocalManyBodyResidualForce (CUDA): ligandTopologyIds must be non-empty");

    numRadialBasis = force.getNumRadialBasis();
    if (numRadialBasis <= 0) throw OpenMMException("LocalManyBodyResidualForce (CUDA): n_radial_basis must be positive");
    if ((int) force.getRadialCenters().size() != numRadialBasis)
        throw OpenMMException("LocalManyBodyResidualForce (CUDA): radialCenters size disagrees with n_radial_basis");
    if ((int) force.getPairWeight().size() != typeCount * typeCount * numRadialBasis)
        throw OpenMMException("LocalManyBodyResidualForce (CUDA): pairWeight size disagrees with type_count^2 * n_radial_basis");
    if (!(force.getInnerCutoffAngstrom() > 0.0 && force.getInnerCutoffAngstrom() < force.getOuterCutoffAngstrom()))
        throw OpenMMException("LocalManyBodyResidualForce (CUDA): cutoffs must satisfy 0 < inner < outer");
    if (!(force.getBMaxReduced() > 0.0)) throw OpenMMException("LocalManyBodyResidualForce (CUDA): b_max_reduced must be positive");
    if (!(force.getTemperatureKelvin() > 0.0)) throw OpenMMException("LocalManyBodyResidualForce (CUDA): temperatureKelvin must be positive");

    int maxEdgesInt, maxNeighborsInt, maxEnvInt;
    force.getCapacityCeilings(maxEdgesInt, maxNeighborsInt, maxEnvInt);
    if (maxEdgesInt <= 0 || maxNeighborsInt <= 0 || maxEnvInt <= 0)
        throw OpenMMException("LocalManyBodyResidualForce (CUDA): capacity ceilings must all be positive");

    // ---- G3: skinAngstrom==0 (Force default) => G2 legacy brute-force path, byte-for-byte unchanged ----
    skinAngstrom = force.getSkinAngstrom();
    int candidateCapacityInt = force.getCandidateListCapacity();
    g3Enabled = (skinAngstrom > 0.0);
    if (g3Enabled) {
        if (candidateCapacityInt <= 0)
            throw OpenMMException("LocalManyBodyResidualForce (CUDA): skinAngstrom > 0 requires candidateListCapacity > 0 (G3 local-CSR path)");
        if (candidateCapacityInt <= maxEdgesInt)
            throw OpenMMException("LocalManyBodyResidualForce (CUDA): candidateListCapacity must exceed max_edges "
                                   "(candidates at r<outer_cutoff+skin are always a superset of active edges at r<outer_cutoff)");
    } else if (candidateCapacityInt != 0) {
        throw OpenMMException("LocalManyBodyResidualForce (CUDA): candidateListCapacity set without skinAngstrom "
                               "(both or neither -- see setSkinAngstrom doc)");
    }
    candidateListCapacity = candidateCapacityInt;

    numLigandAtoms = (int) ligandIds.size();
    ligandSystemIds.assign(ligandIds.begin(), ligandIds.end());
    sort(ligandSystemIds.begin(), ligandSystemIds.end());
    typeBySystemId.assign(types.begin(), types.end());
    innerCutoffAngstrom = force.getInnerCutoffAngstrom();
    outerCutoffAngstrom = force.getOuterCutoffAngstrom();
    bMaxReduced = force.getBMaxReduced();
    radialWidthAngstrom = force.getRadialWidthAngstrom();
    maxEdges = maxEdgesInt;
    maxNeighborsPerLigand = maxNeighborsInt;
    maxEnvironmentAtoms = maxEnvInt;
    kBTKilojoulePerMole = force.getKBTKilojoulePerMole();

    cu.setAsCurrent();
    bool useDouble = cu.getUseDoublePrecision();
    // "mixed" (the device typedef, not the OpenMM precision MODE name) is
    // double whenever this Context is mixed-precision OR genuinely
    // double-precision -- see the kDeviceSource file header table. This
    // controls ONLY host-side buffer element sizing for lastRebuildPositionsDevice
    // and the K0 scalar threshold argument; it is unrelated to `useDouble`
    // (which governs every OTHER real-typed buffer/argument and is false in
    // mixed mode, since `real`=float there).
    bool useWidePositionStorage = useDouble || cu.getUseMixedPrecision();

    uploadRealArray(radialCentersDevice, cu, useDouble, force.getRadialCenters(), "lmbrRadialCenters");
    uploadRealArray(pairWeightDevice, cu, useDouble, force.getPairWeight(), "lmbrPairWeight");

    vector<double> mlpFlat((size_t) typeCount * EXP025_MLP_STRIDE, 0.0);
    for (int t = 0; t < typeCount; t++) {
        const LocalManyBodyTypedMLP& mlp = force.getTypedMLP(t);
        double* base = mlpFlat.data() + (size_t) t * EXP025_MLP_STRIDE;
        for (int k = 0; k < 16; k++) { base[EXP025_MLP_OFFSET_W0 + k] = mlp.w0[k]; base[EXP025_MLP_OFFSET_B0 + k] = mlp.b0[k]; }
        for (int i = 0; i < 256; i++) base[EXP025_MLP_OFFSET_W2 + i] = mlp.w2[i];
        for (int k = 0; k < 16; k++) { base[EXP025_MLP_OFFSET_B2 + k] = mlp.b2[k]; base[EXP025_MLP_OFFSET_W4 + k] = mlp.w4[k]; }
        base[EXP025_MLP_OFFSET_B4] = mlp.b4;
    }
    uploadRealArray(typedMlpFlatDevice, cu, useDouble, mlpFlat, "lmbrTypedMlpFlat");

    anchorDeviceIdsDevice.initialize<int>(cu, numLigandAtoms, "lmbrAnchorDeviceIds");
    typeByDeviceDevice.initialize<int>(cu, numParticles, "lmbrTypeByDevice");
    isLigandByDeviceDevice.initialize<int>(cu, numParticles, "lmbrIsLigandByDevice");

    uploadRealArray(qDevice, cu, useDouble, vector<double>(numLigandAtoms, 0.0), "lmbrQ");
    uploadRealArray(dBdqDevice, cu, useDouble, vector<double>(numLigandAtoms, 0.0), "lmbrDBdq");
    uploadRealArray(energyScratchDevice, cu, useDouble, vector<double>(1, 0.0), "lmbrEnergyScratch");
    neighborCountDevice.initialize<int>(cu, numLigandAtoms, "lmbrNeighborCount");
    uniqueEnvEpochDevice.initialize<int>(cu, numParticles, "lmbrUniqueEnvEpoch");
    uniqueEnvEpochDevice.upload(vector<int>(numParticles, 0));  // initialize once; no per-evaluation full-array reset
    deviceStatusDevice.initialize<int>(cu, EXP026_STATUS_WORDS, "lmbrExp026SupportStatus");
    deviceStatusDevice.upload(vector<int>(EXP026_STATUS_WORDS, 0));
    uniqueEnvironmentEpoch = 0;

    if (g3Enabled) {
        nextAtomDevice.initialize<int>(cu, numParticles, "lmbrNextAtom");
        anchorCandidateCountDevice.initialize<int>(cu, numLigandAtoms, "lmbrAnchorCandidateCount");
        anchorCandidateCountDevice.upload(vector<int>(numLigandAtoms, 0));
        anchorOffsetsDevice.initialize<int>(cu, numLigandAtoms + 1, "lmbrAnchorOffsets");
        anchorOffsetsDevice.upload(vector<int>(numLigandAtoms + 1, 0));
        edgeAtomsDevice.initialize<int>(cu, candidateListCapacity, "lmbrEdgeAtoms");
        edgeAtomsDevice.upload(vector<int>(candidateListCapacity, -1));
        // Flat [x0,y0,z0,x1,y1,z1,...] layout, `mixed`-typed on the device
        // side -- sized as double when useWidePositionStorage, matching
        // what `mixed` actually resolves to in that mode (see above).
        uploadRealArray(lastRebuildPositionsDevice, cu, useWidePositionStorage,
                         vector<double>((size_t) numParticles * 3, 0.0), "lmbrLastRebuildPositions");
        rebuildFlagDevice.initialize<int>(cu, 1, "lmbrRebuildFlag");
        rebuildFlagDevice.upload(vector<int>(1, 0));
        // cellHeadDevice is deliberately NOT allocated here -- its size
        // depends on the live runtime box (via face heights), which is not
        // reliably known at initialize() time (only System::
        // getDefaultPeriodicBoxVectors() is, and a caller could still call
        // Context::setPeriodicBoxVectors() before the first execute()).
        // allocateCellGrid() allocates it lazily on the first execute().
        cellGridAllocated = false;
        candidateListValid = false;  // forces a full rebuild on the first execute()
    }

    buildAndLoadKernels();

    reorderListener = new ReorderListenerImpl(*this);
    cu.addReorderListener(reorderListener);
    deviceStateValid = false;
    ensureDeviceStateCurrent();
}

void CudaCalcLocalManyBodyResidualForceKernel::buildAndLoadKernels() {
    map<string, string> defines;
    defines["NUM_LIGANDS"] = to_string(numLigandAtoms);
    defines["NUM_ATOMS"] = to_string(cu.getNumAtoms());
    defines["NUM_TYPES"] = to_string(typeCount);
    defines["NUM_RADIAL_BASIS"] = to_string(numRadialBasis);
    defines["MAX_NEIGHBORS_PER_LIGAND"] = to_string(maxNeighborsPerLigand);
    defines["MAX_ACTIVE_EDGES"] = to_string(maxEdges);
    defines["MAX_ENVIRONMENT_ATOMS"] = to_string(maxEnvironmentAtoms);
    defines["CUDA_BLOCK_SIZE"] = to_string(CUDA_BLOCK_SIZE);
    // Our own compile-time flag (NOT an OpenMM-provided macro): controls
    // only whether exp025LoadPos() reads+adds posqCorrection. See the
    // kDeviceSource file header for the full single/mixed/double table.
    defines["USE_MIXED_PRECISION"] = cu.getUseMixedPrecision() ? "1" : "0";

    string source = buildDefinesPrefix() + string(kDeviceSource);
    if (g3Enabled) source += string(kDeviceSourceCSR);
    program = cu.compileProgram(source, defines);
    resetStatusKernel = program->createKernel("exp026ResetSupportStatus");
    computeQKernel = program->createKernel("exp025ComputeQ");
    readoutKernel = program->createKernel("exp025Readout");
    scatterForceKernel = program->createKernel("exp025ScatterForce");
    if (g3Enabled) {
        checkDisplacementKernel = program->createKernel("exp025CheckDisplacement");
        clearCellHeadsKernel = program->createKernel("exp025ClearCellHeads");
        binEnvironmentAtomsKernel = program->createKernel("exp025BinEnvironmentAtoms");
        countCandidatesKernel = program->createKernel("exp025CountCandidates");
        prefixSumOffsetsKernel = program->createKernel("exp025PrefixSumOffsets");
        fillCandidatesKernel = program->createKernel("exp025FillCandidates");
        copyLastPositionsKernel = program->createKernel("exp025CopyLastPositions");
        clearRebuildFlagKernel = program->createKernel("exp025ClearRebuildFlag");
        computeQFromCsrKernel = program->createKernel("exp025ComputeQFromCSR");
        scatterForceFromCsrKernel = program->createKernel("exp025ScatterForceFromCSR");
    }
    hasInitializedKernels = true;
}

void CudaCalcLocalManyBodyResidualForceKernel::allocateCellGrid(int nCellsX, int nCellsY, int nCellsZ) {
    if (!cellGridAllocated) {
        cellHeadDevice.initialize<int>(cu, (size_t) nCellsX * nCellsY * nCellsZ, "lmbrCellHead");
        allocatedNCellsX = nCellsX; allocatedNCellsY = nCellsY; allocatedNCellsZ = nCellsZ;
        cellGridAllocated = true;
        return;
    }
    if (nCellsX != allocatedNCellsX || nCellsY != allocatedNCellsY || nCellsZ != allocatedNCellsZ)
        throw OpenMMException("LocalManyBodyResidualForce (CUDA) fail-closed: box change altered the G3 cell grid "
                               "(" + to_string(allocatedNCellsX) + "," + to_string(allocatedNCellsY) + "," + to_string(allocatedNCellsZ) +
                               ") -> (" + to_string(nCellsX) + "," + to_string(nCellsY) + "," + to_string(nCellsZ) +
                               ") -- G3 is fixed-box NVT only (UNSUPPORTED_BOX), not dynamically reallocated");
}

void CudaCalcLocalManyBodyResidualForceKernel::ensureDeviceStateCurrent() {
    if (deviceStateValid) return;
    const vector<int>& atomIndex = cu.getAtomIndex();  // device slot -> system id
    int numParticles = (int) typeBySystemId.size();
    if ((int) atomIndex.size() < numParticles)
        throw OpenMMException("LocalManyBodyResidualForce (CUDA): atom index array smaller than particle count");

    vector<int> deviceIdOfSystemId(numParticles, -1);
    for (int deviceSlot = 0; deviceSlot < numParticles; deviceSlot++)
        deviceIdOfSystemId[atomIndex[deviceSlot]] = deviceSlot;

    vector<int> anchorDeviceIds(numLigandAtoms);
    for (int i = 0; i < numLigandAtoms; i++) anchorDeviceIds[i] = deviceIdOfSystemId[ligandSystemIds[i]];

    vector<int> typeByDevice(numParticles), isLigandByDevice(numParticles, 0);
    for (int deviceSlot = 0; deviceSlot < numParticles; deviceSlot++)
        typeByDevice[deviceSlot] = typeBySystemId[atomIndex[deviceSlot]];
    for (int id : anchorDeviceIds) isLigandByDevice[id] = 1;

    anchorDeviceIdsDevice.upload(anchorDeviceIds);
    typeByDeviceDevice.upload(typeByDevice);
    isLigandByDeviceDevice.upload(isLigandByDevice);
    deviceStateValid = true;
}

// A2 (PLAN section 20.1.4): ONE consolidated download+check of the whole
// DeviceStatusV1 block per force evaluation, replacing A1's ~6 separate
// checkDeviceErrorFlag() calls (each its own blocking D2H). The exception
// text names the fail-closed stage, the frozen error code's reason, AND the
// key device-resident counts (active edges / max neighbors / unique
// environments / candidates) at the moment of first failure -- per PLAN's
// own requirement that the report include "stage、code 和关键计数", not
// just a bare error string.
//
// NOT yet done in this draft (see A2_DRAFT_STATUS.md): this is called from
// a single site in execute(), positioned AFTER scatter -- the same relative
// position as A1/EXP-025's LAST checkDeviceErrorFlag() call, not before
// scatter. Moving it before scatter (PLAN 20.1.3's actual partial-force
// fix) requires proving q/K1/K6a + readout's own preflight checks are a
// strict superset of everything scatter separately checks; that proof
// (fault injection per error stage) has not been written yet, so this draft
// deliberately keeps the conservative (identical-to-A1) check position and
// only consolidates the DOWNLOAD COUNT (~6 -> 1), not yet the check TIMING.
void CudaCalcLocalManyBodyResidualForceKernel::checkDeviceStatusOnce(const char* completedStage) {
    vector<int> words;
    deviceStatusDevice.download(words);
    Exp026ControlPlaneStatus status = decodeExp026ControlPlaneStatus(words.data(), (int) words.size());
    if (status.errorCode == EXP025_DEVICE_ERROR_OK) return;

    string reason;
    switch (status.errorCode) {
        case EXP025_DEVICE_ERROR_HALF_BOX_TIE: reason = "half-box MIC tie"; break;
        case EXP025_DEVICE_ERROR_MIN_DISTANCE: reason = "pair distance below 0.1 Angstrom"; break;
        case EXP025_DEVICE_ERROR_EDGE_OVERFLOW: reason = "active edge count exceeded max_edges"; break;
        case EXP025_DEVICE_ERROR_NEIGHBOR_OVERFLOW: reason = "a ligand atom exceeded max_neighbors_per_ligand"; break;
        case EXP025_DEVICE_ERROR_UNIQUE_ENV_OVERFLOW: reason = "unique environment atom count exceeded max_environment_atoms"; break;
        case EXP025_DEVICE_ERROR_NONFINITE: reason = "nonfinite value produced"; break;
        case EXP025_DEVICE_ERROR_CANDIDATE_OVERFLOW: reason = "G3 candidate count exceeded candidateListCapacity"; break;
        case EXP025_DEVICE_ERROR_UNSUPPORTED_BOX: reason = "G3 box/cell-grid safety precondition violated"; break;
        default: reason = "unknown device error code " + to_string(status.errorCode);
    }
    string stageName;
    switch (status.errorStage) {
        case EXP026_STAGE_REBUILD_PREFIX: stageName = "K4 (prefixSumOffsets)"; break;
        case EXP026_STAGE_REBUILD_FILL: stageName = "K5 (fillCandidates)"; break;
        case EXP026_STAGE_COMPUTE_Q: stageName = "K1/K6a (computeQ)"; break;
        case EXP026_STAGE_READOUT: stageName = "K2 (readout)"; break;
        case EXP026_STAGE_FORCE_SCATTER: stageName = "K3/K6b (scatterForce)"; break;
        case EXP026_STAGE_HOST_VALIDATE: stageName = "host validation"; break;
        default: stageName = "unknown stage " + to_string(status.errorStage);
    }
    throw OpenMMException(
        string("LocalManyBodyResidualForce (CUDA) fail-closed after ") + completedStage +
        " in " + stageName + " (code " + to_string(status.errorCode) + "): " + reason +
        " [active_edges=" + to_string(status.activeEdges) +
        " max_neighbors=" + to_string(status.maxNeighbors) +
        " unique_environments=" + to_string(status.uniqueEnvironments) +
        " candidates=" + to_string(status.candidates) +
        " epoch=" + to_string(status.epoch) + "]");
}

// Adds the (posq, posqCorrection) argument pair any position-touching kernel
// expects. posqCorrection only genuinely exists once
// cu.getUseMixedPrecision() is true (see CudaContext::getPosqCorrection()'s
// own doc comment); in single/double precision mode this passes cu.getPosq()
// again as a harmless placeholder -- the compiled kernel (USE_MIXED_PRECISION=0
// in that case) never dereferences the second argument, so its value is
// irrelevant as long as it is a valid device pointer, which cu.getPosq()
// always is.
void CudaCalcLocalManyBodyResidualForceKernel::addPosArgs(ComputeKernel& kernel, bool bound, int idx) {
    addArrArg(kernel, bound, idx, cu.getPosq());
    if (cu.getUseMixedPrecision()) addArrArg(kernel, bound, idx + 1, cu.getPosqCorrection());
    else addArrArg(kernel, bound, idx + 1, cu.getPosq());
}

double CudaCalcLocalManyBodyResidualForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    cu.setAsCurrent();
    ensureDeviceStateCurrent();

    // EXP-026 Patch A1: advance an evaluation epoch instead of clearing and
    // round-tripping a numParticles-sized unique-environment flag array.
    bool resetEpochTags = false;
    uniqueEnvironmentEpoch = nextExp026UniqueEnvironmentEpoch(uniqueEnvironmentEpoch, resetEpochTags);
    if (resetEpochTags)
        uniqueEnvEpochDevice.upload(vector<int>(typeBySystemId.size(), 0));  // only on signed-int epoch wrap

    // A2: resetStatusKernel now clears the ENTIRE status block (error_code/
    // stage included) every evaluation -- this is the only per-evaluation
    // upload of any kind on the error/status path; there is no longer a
    // separate errorFlagDevice to reset.
    addArrArg(resetStatusKernel, argsBoundReset, 0, deviceStatusDevice);
    addScalarArg<int>(resetStatusKernel, argsBoundReset, 1, uniqueEnvironmentEpoch);
    resetStatusKernel->execute(1, 1);
    argsBoundReset = true;

    bool useDouble = cu.getUseDoublePrecision();
    bool useWidePositionStorage = useDouble || cu.getUseMixedPrecision();
    // Box vector pointers are precision-typed (float4 or double4) by
    // useDouble, same as before mixed-precision support -- box vectors
    // themselves are NOT subject to the posqCorrection mechanism (only
    // per-atom positions are), so this is unaffected by mixed precision.
    double bx, by, bz, ax, ay, az, cx, cy, cz, ix, iy, iz;
    if (useDouble) {
        double4* vx = (double4*) cu.getPeriodicBoxVecXPointer();
        double4* vy = (double4*) cu.getPeriodicBoxVecYPointer();
        double4* vz = (double4*) cu.getPeriodicBoxVecZPointer();
        double4* inv = (double4*) cu.getInvPeriodicBoxSizePointer();
        ax = vx->x; ay = vx->y; az = vx->z;
        bx = vy->x; by = vy->y; bz = vy->z;
        cx = vz->x; cy = vz->y; cz = vz->z;
        ix = inv->x; iy = inv->y; iz = inv->z;
    } else {
        float4* vx = (float4*) cu.getPeriodicBoxVecXPointer();
        float4* vy = (float4*) cu.getPeriodicBoxVecYPointer();
        float4* vz = (float4*) cu.getPeriodicBoxVecZPointer();
        float4* inv = (float4*) cu.getInvPeriodicBoxSizePointer();
        ax = vx->x; ay = vx->y; az = vx->z;
        bx = vy->x; by = vy->y; bz = vy->z;
        cx = vz->x; cy = vz->y; cz = vz->z;
        ix = inv->x; iy = inv->y; iz = inv->z;
    }

    if (!g3Enabled) {
        // ==================== G2 legacy brute-force path ====================
        // (Now supports mixed precision too -- see kDeviceSource header. The
        // single/double behavior below is byte-for-byte unchanged from the
        // pre-mixed-precision G2 code; only addPosArgs()'s SECOND argument
        // and the device-side exp025LoadPos() reconstruction are new.)
        double volume = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
        double bcx = by * cz - bz * cy, bcy = bz * cx - bx * cz, bcz = bx * cy - by * cx;  // b x c
        double cax = cy * az - cz * ay, cay = cz * ax - cx * az, caz = cx * ay - cy * ax;  // c x a
        double abx = ay * bz - az * by, aby = az * bx - ax * bz, abz = ax * by - ay * bx;  // a x b
        double hA = fabs(volume) / sqrt(bcx * bcx + bcy * bcy + bcz * bcz);
        double hB = fabs(volume) / sqrt(cax * cax + cay * cay + caz * caz);
        double hC = fabs(volume) / sqrt(abx * abx + aby * aby + abz * abz);
        double minFaceHeightNm = min({hA, hB, hC});
        double rCutNm = outerCutoffAngstrom / 10.0;
        if (!(minFaceHeightNm > 2.0 * rCutNm))
            throw OpenMMException("LocalManyBodyResidualForce (CUDA) fail-closed: minimum periodic face height (" +
                                   to_string(minFaceHeightNm) + " nm) does not exceed 2*outer_cutoff (" +
                                   to_string(2.0 * rCutNm) + " nm) -- half-box tie safety precondition violated (UNSUPPORTED_BOX)");

        // ---- K1: compute q[NUM_LIGANDS] ----
        addPosArgs(computeQKernel, argsBoundComputeQ, 0);
        addReal4Arg(computeQKernel, argsBoundComputeQ, 2, useDouble, ax, ay, az, 0.0);
        addReal4Arg(computeQKernel, argsBoundComputeQ, 3, useDouble, bx, by, bz, 0.0);
        addReal4Arg(computeQKernel, argsBoundComputeQ, 4, useDouble, cx, cy, cz, 0.0);
        addReal4Arg(computeQKernel, argsBoundComputeQ, 5, useDouble, ix, iy, iz, 0.0);
        addArrArg(computeQKernel, argsBoundComputeQ, 6, anchorDeviceIdsDevice);
        addArrArg(computeQKernel, argsBoundComputeQ, 7, typeByDeviceDevice);
        addArrArg(computeQKernel, argsBoundComputeQ, 8, isLigandByDeviceDevice);
        addArrArg(computeQKernel, argsBoundComputeQ, 9, radialCentersDevice);
        addRealArg(computeQKernel, argsBoundComputeQ, 10, useDouble, radialWidthAngstrom);
        addArrArg(computeQKernel, argsBoundComputeQ, 11, pairWeightDevice);
        addRealArg(computeQKernel, argsBoundComputeQ, 12, useDouble, innerCutoffAngstrom);
        addRealArg(computeQKernel, argsBoundComputeQ, 13, useDouble, outerCutoffAngstrom);
        addArrArg(computeQKernel, argsBoundComputeQ, 14, qDevice);
        addArrArg(computeQKernel, argsBoundComputeQ, 15, neighborCountDevice);
        addArrArg(computeQKernel, argsBoundComputeQ, 16, uniqueEnvEpochDevice);
        addScalarArg<int>(computeQKernel, argsBoundComputeQ, 17, uniqueEnvironmentEpoch);
        addArrArg(computeQKernel, argsBoundComputeQ, 18, deviceStatusDevice);
        computeQKernel->execute(numLigandAtoms * CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
        argsBoundComputeQ = true;
        // A2: no intermediate status check here (G2 brute-force path -- see
        // the G3 branch below for why G3 keeps ONE intermediate check and
        // G2 does not: G2 has no analogous "later kernel reads an unsafely-
        // sized buffer if an earlier kernel silently overflowed" hazard).
    } else {
        // ==================== G3 local-CSR path ====================
        double volume = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx);
        double bcx = by * cz - bz * cy, bcy = bz * cx - bx * cz, bcz = bx * cy - by * cx;
        double cax = cy * az - cz * ay, cay = cz * ax - cx * az, caz = cx * ay - cy * ax;
        double abx = ay * bz - az * by, aby = az * bx - ax * bz, abz = ax * by - ay * bx;
        double hA = fabs(volume) / sqrt(bcx * bcx + bcy * bcy + bcz * bcz);
        double hB = fabs(volume) / sqrt(cax * cax + cay * cay + caz * caz);
        double hC = fabs(volume) / sqrt(abx * abx + aby * aby + abz * abz);
        double rListNm = (outerCutoffAngstrom + skinAngstrom) / 10.0;
        int nCellsX = (int) floor(hA / rListNm), nCellsY = (int) floor(hB / rListNm), nCellsZ = (int) floor(hC / rListNm);
        if (nCellsX < 3 || nCellsY < 3 || nCellsZ < 3)
            throw OpenMMException("LocalManyBodyResidualForce (CUDA) fail-closed: G3 cell grid (" + to_string(nCellsX) + "," +
                                   to_string(nCellsY) + "," + to_string(nCellsZ) + ") has an axis with <3 cells at r_list=" +
                                   to_string(rListNm) + " nm (face heights " + to_string(hA) + "," + to_string(hB) + "," +
                                   to_string(hC) + " nm) -- UNSUPPORTED_BOX");
        allocateCellGrid(nCellsX, nCellsY, nCellsZ);  // first call allocates; later calls require an exact dimension match

        bool boxChanged = !(ax == lastRebuildBox[0][0] && ay == lastRebuildBox[0][1] && az == lastRebuildBox[0][2] &&
                             bx == lastRebuildBox[1][0] && by == lastRebuildBox[1][1] && bz == lastRebuildBox[1][2] &&
                             cx == lastRebuildBox[2][0] && cy == lastRebuildBox[2][1] && cz == lastRebuildBox[2][2]);
        bool mustForceRebuild = !candidateListValid || boxChanged;
        if (mustForceRebuild) {
            rebuildFlagDevice.upload(vector<int>(1, 1));
        } else {
            // K0: cheap O(N) displacement check, device-owned flag, no host
            // download -- only runs when the CSR is otherwise known-valid
            // (lastRebuildPositionsDevice holds real data from a real
            // previous rebuild, not initialize()-time zeros).
            double skinNm = skinAngstrom / 10.0;
            double halfSkinNmSquared = (skinNm / 2.0) * (skinNm / 2.0);
            addPosArgs(checkDisplacementKernel, argsBoundCheckDisplacement, 0);
            addArrArg(checkDisplacementKernel, argsBoundCheckDisplacement, 2, lastRebuildPositionsDevice);
            addMixedArg(checkDisplacementKernel, argsBoundCheckDisplacement, 3, useWidePositionStorage, halfSkinNmSquared);
            addArrArg(checkDisplacementKernel, argsBoundCheckDisplacement, 4, rebuildFlagDevice);
            checkDisplacementKernel->execute(cu.getNumAtoms());
            argsBoundCheckDisplacement = true;
        }
        candidateListValid = true;
        lastRebuildBox[0][0] = ax; lastRebuildBox[0][1] = ay; lastRebuildBox[0][2] = az;
        lastRebuildBox[1][0] = bx; lastRebuildBox[1][1] = by; lastRebuildBox[1][2] = bz;
        lastRebuildBox[2][0] = cx; lastRebuildBox[2][1] = cy; lastRebuildBox[2][2] = cz;

        int nCells = nCellsX * nCellsY * nCellsZ;
        double rListAngstrom = outerCutoffAngstrom + skinAngstrom;

        // ---- K1: clear cell heads (self-gated on rebuildFlag) ----
        addArrArg(clearCellHeadsKernel, argsBoundClearCellHeads, 0, cellHeadDevice);
        addScalarArg<int>(clearCellHeadsKernel, argsBoundClearCellHeads, 1, nCells);
        addArrArg(clearCellHeadsKernel, argsBoundClearCellHeads, 2, rebuildFlagDevice);
        clearCellHeadsKernel->execute(nCells);
        argsBoundClearCellHeads = true;

        // ---- K2: bin environment atoms ----
        addPosArgs(binEnvironmentAtomsKernel, argsBoundBinEnv, 0);
        addReal4Arg(binEnvironmentAtomsKernel, argsBoundBinEnv, 2, useDouble, ax, ay, az, 0.0);
        addReal4Arg(binEnvironmentAtomsKernel, argsBoundBinEnv, 3, useDouble, bx, by, bz, 0.0);
        addReal4Arg(binEnvironmentAtomsKernel, argsBoundBinEnv, 4, useDouble, cx, cy, cz, 0.0);
        addArrArg(binEnvironmentAtomsKernel, argsBoundBinEnv, 5, isLigandByDeviceDevice);
        addScalarArg<int>(binEnvironmentAtomsKernel, argsBoundBinEnv, 6, nCellsX);
        addScalarArg<int>(binEnvironmentAtomsKernel, argsBoundBinEnv, 7, nCellsY);
        addScalarArg<int>(binEnvironmentAtomsKernel, argsBoundBinEnv, 8, nCellsZ);
        addArrArg(binEnvironmentAtomsKernel, argsBoundBinEnv, 9, cellHeadDevice);
        addArrArg(binEnvironmentAtomsKernel, argsBoundBinEnv, 10, nextAtomDevice);
        addArrArg(binEnvironmentAtomsKernel, argsBoundBinEnv, 11, rebuildFlagDevice);
        binEnvironmentAtomsKernel->execute(cu.getNumAtoms());
        argsBoundBinEnv = true;

        // ---- K3: count <r_list candidates per anchor (one block/anchor, 32 threads: 27 used, 1 per neighbor cell) ----
        addPosArgs(countCandidatesKernel, argsBoundCountCandidates, 0);
        addReal4Arg(countCandidatesKernel, argsBoundCountCandidates, 2, useDouble, ax, ay, az, 0.0);
        addReal4Arg(countCandidatesKernel, argsBoundCountCandidates, 3, useDouble, bx, by, bz, 0.0);
        addReal4Arg(countCandidatesKernel, argsBoundCountCandidates, 4, useDouble, cx, cy, cz, 0.0);
        addReal4Arg(countCandidatesKernel, argsBoundCountCandidates, 5, useDouble, ix, iy, iz, 0.0);
        addArrArg(countCandidatesKernel, argsBoundCountCandidates, 6, anchorDeviceIdsDevice);
        addScalarArg<int>(countCandidatesKernel, argsBoundCountCandidates, 7, nCellsX);
        addScalarArg<int>(countCandidatesKernel, argsBoundCountCandidates, 8, nCellsY);
        addScalarArg<int>(countCandidatesKernel, argsBoundCountCandidates, 9, nCellsZ);
        addArrArg(countCandidatesKernel, argsBoundCountCandidates, 10, cellHeadDevice);
        addArrArg(countCandidatesKernel, argsBoundCountCandidates, 11, nextAtomDevice);
        addRealArg(countCandidatesKernel, argsBoundCountCandidates, 12, useDouble, rListAngstrom);
        addArrArg(countCandidatesKernel, argsBoundCountCandidates, 13, rebuildFlagDevice);
        addArrArg(countCandidatesKernel, argsBoundCountCandidates, 14, anchorCandidateCountDevice);
        countCandidatesKernel->execute(numLigandAtoms * 32, 32);
        argsBoundCountCandidates = true;

        // ---- K4: serial prefix sum -> anchorOffsets (also flags total-capacity overflow) ----
        addArrArg(prefixSumOffsetsKernel, argsBoundPrefixSum, 0, anchorCandidateCountDevice);
        addArrArg(prefixSumOffsetsKernel, argsBoundPrefixSum, 1, anchorOffsetsDevice);
        addScalarArg<int>(prefixSumOffsetsKernel, argsBoundPrefixSum, 2, (int) candidateListCapacity);
        addArrArg(prefixSumOffsetsKernel, argsBoundPrefixSum, 3, rebuildFlagDevice);
        addArrArg(prefixSumOffsetsKernel, argsBoundPrefixSum, 4, deviceStatusDevice);
        prefixSumOffsetsKernel->execute(1, 1);
        argsBoundPrefixSum = true;

        // ---- K5: fill the compact CSR (no-ops if K4 already flagged overflow) ----
        addPosArgs(fillCandidatesKernel, argsBoundFillCandidates, 0);
        addReal4Arg(fillCandidatesKernel, argsBoundFillCandidates, 2, useDouble, ax, ay, az, 0.0);
        addReal4Arg(fillCandidatesKernel, argsBoundFillCandidates, 3, useDouble, bx, by, bz, 0.0);
        addReal4Arg(fillCandidatesKernel, argsBoundFillCandidates, 4, useDouble, cx, cy, cz, 0.0);
        addReal4Arg(fillCandidatesKernel, argsBoundFillCandidates, 5, useDouble, ix, iy, iz, 0.0);
        addArrArg(fillCandidatesKernel, argsBoundFillCandidates, 6, anchorDeviceIdsDevice);
        addScalarArg<int>(fillCandidatesKernel, argsBoundFillCandidates, 7, nCellsX);
        addScalarArg<int>(fillCandidatesKernel, argsBoundFillCandidates, 8, nCellsY);
        addScalarArg<int>(fillCandidatesKernel, argsBoundFillCandidates, 9, nCellsZ);
        addArrArg(fillCandidatesKernel, argsBoundFillCandidates, 10, cellHeadDevice);
        addArrArg(fillCandidatesKernel, argsBoundFillCandidates, 11, nextAtomDevice);
        addRealArg(fillCandidatesKernel, argsBoundFillCandidates, 12, useDouble, rListAngstrom);
        addArrArg(fillCandidatesKernel, argsBoundFillCandidates, 13, rebuildFlagDevice);
        addArrArg(fillCandidatesKernel, argsBoundFillCandidates, 14, anchorOffsetsDevice);
        addArrArg(fillCandidatesKernel, argsBoundFillCandidates, 15, edgeAtomsDevice);
        addArrArg(fillCandidatesKernel, argsBoundFillCandidates, 16, deviceStatusDevice);
        fillCandidatesKernel->execute(numLigandAtoms * 32, 32);
        argsBoundFillCandidates = true;

        // ---- snapshot positions, THEN clear the flag (strictly after everything above has read it) ----
        addPosArgs(copyLastPositionsKernel, argsBoundCopyLastPositions, 0);
        addArrArg(copyLastPositionsKernel, argsBoundCopyLastPositions, 2, lastRebuildPositionsDevice);
        addArrArg(copyLastPositionsKernel, argsBoundCopyLastPositions, 3, rebuildFlagDevice);
        copyLastPositionsKernel->execute(cu.getNumAtoms());
        argsBoundCopyLastPositions = true;
        addArrArg(clearRebuildFlagKernel, argsBoundClearRebuildFlag, 0, rebuildFlagDevice);
        clearRebuildFlagKernel->execute(1, 1);
        argsBoundClearRebuildFlag = true;

        // A2: this ONE intermediate check is NOT removed by the consolidation
        // -- it is a genuine safety GATE, not just an early error report.
        // K4's prefix sum writes anchorOffsets from the UNCLAMPED running
        // total (only setting the overflow bit, never clamping the offsets
        // themselves), so on CANDIDATE_OVERFLOW, anchorOffsets can describe
        // a range that extends past edgeAtomsDevice's actual
        // candidateListCapacity-sized allocation. Removing this check would
        // let K6a below launch and read out-of-bounds device memory through
        // that stale anchorOffsets range -- a real memory-safety bug, not a
        // delayed-error nicety. G2's brute-force path above has no
        // equivalent hazard (no CSR, no capacity-bounded buffer another
        // kernel indexes through), which is why it has zero intermediate
        // checks instead of one.
        checkDeviceStatusOnce("G3 rebuild (K1-K5)");

        // ---- K6a: compute q[NUM_LIGANDS] from the CSR instead of an all-N brute-force scan ----
        addPosArgs(computeQFromCsrKernel, argsBoundComputeQFromCsr, 0);
        addReal4Arg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 2, useDouble, ax, ay, az, 0.0);
        addReal4Arg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 3, useDouble, bx, by, bz, 0.0);
        addReal4Arg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 4, useDouble, cx, cy, cz, 0.0);
        addReal4Arg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 5, useDouble, ix, iy, iz, 0.0);
        addArrArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 6, anchorDeviceIdsDevice);
        addArrArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 7, typeByDeviceDevice);
        addArrArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 8, anchorOffsetsDevice);
        addArrArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 9, edgeAtomsDevice);
        addArrArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 10, radialCentersDevice);
        addRealArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 11, useDouble, radialWidthAngstrom);
        addArrArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 12, pairWeightDevice);
        addRealArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 13, useDouble, innerCutoffAngstrom);
        addRealArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 14, useDouble, outerCutoffAngstrom);
        addArrArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 15, qDevice);
        addArrArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 16, neighborCountDevice);
        addArrArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 17, uniqueEnvEpochDevice);
        addScalarArg<int>(computeQFromCsrKernel, argsBoundComputeQFromCsr, 18, uniqueEnvironmentEpoch);
        addArrArg(computeQFromCsrKernel, argsBoundComputeQFromCsr, 19, deviceStatusDevice);
        computeQFromCsrKernel->execute(numLigandAtoms * CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
        argsBoundComputeQFromCsr = true;
        // A2: no intermediate check here either -- K6a's own possible
        // errors (MIN_DISTANCE/HALF_BOX_TIE/NONFINITE/ceiling overflows) are
        // pure "detect and report", not a memory-safety gate for a LATER
        // kernel the way K4/K5's CANDIDATE_OVERFLOW is; they are safely
        // covered by the single consolidated check at the end of execute().
    }

    // ---- K2: readout (always run -- needed for dB/dq even in force-only calls) ----
    addArrArg(readoutKernel, argsBoundReadout, 0, anchorDeviceIdsDevice);
    addArrArg(readoutKernel, argsBoundReadout, 1, typeByDeviceDevice);
    addArrArg(readoutKernel, argsBoundReadout, 2, qDevice);
    addArrArg(readoutKernel, argsBoundReadout, 3, typedMlpFlatDevice);
    addRealArg(readoutKernel, argsBoundReadout, 4, useDouble, bMaxReduced);
    addArrArg(readoutKernel, argsBoundReadout, 5, dBdqDevice);
    addArrArg(readoutKernel, argsBoundReadout, 6, energyScratchDevice);
    addArrArg(readoutKernel, argsBoundReadout, 7, deviceStatusDevice);
    readoutKernel->execute(numLigandAtoms, numLigandAtoms);
    argsBoundReadout = true;

    double energy = 0.0;
    if (includeEnergy) {
        double bReduced = downloadRealScalar(energyScratchDevice, useDouble);
        energy = kBTKilojoulePerMole * bReduced;
    }

    // ---- K3/K6b: force scatter ----
    if (includeForces) {
        if (!g3Enabled) {
            addPosArgs(scatterForceKernel, argsBoundScatterForce, 0);
            addReal4Arg(scatterForceKernel, argsBoundScatterForce, 2, useDouble, ax, ay, az, 0.0);
            addReal4Arg(scatterForceKernel, argsBoundScatterForce, 3, useDouble, bx, by, bz, 0.0);
            addReal4Arg(scatterForceKernel, argsBoundScatterForce, 4, useDouble, cx, cy, cz, 0.0);
            addReal4Arg(scatterForceKernel, argsBoundScatterForce, 5, useDouble, ix, iy, iz, 0.0);
            addArrArg(scatterForceKernel, argsBoundScatterForce, 6, anchorDeviceIdsDevice);
            addArrArg(scatterForceKernel, argsBoundScatterForce, 7, typeByDeviceDevice);
            addArrArg(scatterForceKernel, argsBoundScatterForce, 8, isLigandByDeviceDevice);
            addArrArg(scatterForceKernel, argsBoundScatterForce, 9, radialCentersDevice);
            addRealArg(scatterForceKernel, argsBoundScatterForce, 10, useDouble, radialWidthAngstrom);
            addArrArg(scatterForceKernel, argsBoundScatterForce, 11, pairWeightDevice);
            addRealArg(scatterForceKernel, argsBoundScatterForce, 12, useDouble, innerCutoffAngstrom);
            addRealArg(scatterForceKernel, argsBoundScatterForce, 13, useDouble, outerCutoffAngstrom);
            addArrArg(scatterForceKernel, argsBoundScatterForce, 14, dBdqDevice);
            addRealArg(scatterForceKernel, argsBoundScatterForce, 15, useDouble, kBTKilojoulePerMole);
            addScalarArg<int>(scatterForceKernel, argsBoundScatterForce, 16, cu.getPaddedNumAtoms());
            addArrArg(scatterForceKernel, argsBoundScatterForce, 17, cu.getForce());
            addArrArg(scatterForceKernel, argsBoundScatterForce, 18, deviceStatusDevice);
            scatterForceKernel->execute(numLigandAtoms * CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
            argsBoundScatterForce = true;
        } else {
            addPosArgs(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 0);
            addReal4Arg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 2, useDouble, ax, ay, az, 0.0);
            addReal4Arg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 3, useDouble, bx, by, bz, 0.0);
            addReal4Arg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 4, useDouble, cx, cy, cz, 0.0);
            addReal4Arg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 5, useDouble, ix, iy, iz, 0.0);
            addArrArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 6, anchorDeviceIdsDevice);
            addArrArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 7, typeByDeviceDevice);
            addArrArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 8, anchorOffsetsDevice);
            addArrArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 9, edgeAtomsDevice);
            addArrArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 10, radialCentersDevice);
            addRealArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 11, useDouble, radialWidthAngstrom);
            addArrArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 12, pairWeightDevice);
            addRealArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 13, useDouble, innerCutoffAngstrom);
            addRealArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 14, useDouble, outerCutoffAngstrom);
            addArrArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 15, dBdqDevice);
            addRealArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 16, useDouble, kBTKilojoulePerMole);
            addScalarArg<int>(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 17, cu.getPaddedNumAtoms());
            addArrArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 18, cu.getForce());
            addArrArg(scatterForceFromCsrKernel, argsBoundScatterForceFromCsr, 19, deviceStatusDevice);
            scatterForceFromCsrKernel->execute(numLigandAtoms * CUDA_BLOCK_SIZE, CUDA_BLOCK_SIZE);
            argsBoundScatterForceFromCsr = true;
        }
    }

    // A2: the ONE consolidated status harvest for everything that isn't the
    // G3 rebuild memory-safety gate above -- covers K1/K6a (computeQ),
    // K2 (readout), and K3/K6b (scatter) in a single D2H, positioned after
    // scatter (same relative position as A1/EXP-025's LAST check; see
    // checkDeviceStatusOnce()'s own doc comment for why this draft has not
    // yet attempted moving it earlier).
    checkDeviceStatusOnce("force evaluation");

    return energy;
}
