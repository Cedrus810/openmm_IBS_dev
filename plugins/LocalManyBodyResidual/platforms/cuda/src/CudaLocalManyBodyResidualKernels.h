#ifndef CUDA_LOCALMANYBODYRESIDUAL_KERNELS_H_
#define CUDA_LOCALMANYBODYRESIDUAL_KERNELS_H_

/*
 * EXP-025 G2/G3: CUDA correctness kernels.
 *
 * G2 scope (frozen by the G2 gate definition, UNCHANGED by G3): 41 anchors x
 * all-environment brute force. Passing G2's own test sets
 * G2_CUDA_BRUTE_FORCE_CORRECTNESS = true and nothing else.
 *
 * G3 scope (this addition): a local CSR/Verlet candidate list built from a
 * GPU linked-cell grid (K0 displacement/box/reorder check -> K1 clear cell
 * heads -> K2 bin environment atoms -> K3 count candidates per anchor via a
 * 27-cell stencil -> K4 prefix-sum 41 counts -> K5 fill the compact CSR),
 * with the existing q/readout/force math re-pointed at the CSR instead of a
 * brute-force all-N scan. NO cost qualification yet -- that is G4.
 *
 * G2/G3 code-path selection is a Force-level switch, not a second plugin:
 * skinAngstrom == 0 (the LocalManyBodyResidualForce default) selects the
 * sealed G2 brute-force path byte-for-byte unchanged (so the frozen G2
 * report/test can never silently start exercising different code).
 * skinAngstrom > 0 (together with candidateListCapacity > 0, validated at
 * initialize()) selects the new G3 local-CSR path. This is a deliberate
 * design choice recorded here and in the G3 report, not implied by the PLAN
 * doc's own prose.
 *
 * Deliberately does NOT #include g1_math_core.h or any Reference-platform
 * math: the actual arithmetic (RBF/C2/typed-MLP/gradient) is reimplemented
 * independently in CUDA device code (see the kernel source strings in the
 * .cpp file) so that "CUDA matches Reference" is a real correctness check,
 * not two copies of the same possible bug. r1_model_layout.h is the only
 * shared file, and it contains layout/offset constants only, no math. The
 * G3 CSR kernels DO share the per-edge (C2/RBF/gradient) __device__ helper
 * with G2's brute-force kernels -- both are CUDA-platform code written by
 * the same author, so that sharing is ordinary code reuse, not the
 * cross-platform "two independent implementations" concern above.
 *
 * Mixed precision (posqCorrection) IS supported (added post-G3, see
 * kDeviceSource's file header in the .cpp for the single/mixed/double
 * position-reconstruction table): exp025LoadPos() combines posq+
 * posqCorrection at `mixed` (device typedef) precision everywhere positions
 * are read, in both the G2 brute-force and G3 CSR kernels. This is a
 * separate, additive re-qualification (G2_CUDA_MIXED_CORRECTNESS /
 * G3_CSR_MIXED_CORRECTNESS) on top of the already-sealed single/double
 * results (G2_CUDA_SINGLE_CORRECTNESS / G3_CSR_SINGLE_CORRECTNESS) -- see
 * g4_preflight_report.json. Explicit G2/G3 scope limits (documented, not
 * silent gaps):
 *  - multi-GPU is NOT supported (matches PLAN section 13).
 *  - G3 is fixed-box NVT only: the cell grid's (nCellsX,nCellsY,nCellsZ) is
 *    fixed at the first rebuild and any later box change that would alter
 *    those dimensions fails closed (UNSUPPORTED_BOX) rather than silently
 *    reallocating -- NPT/barostat is explicitly out of scope (PLAN section
 *    6.2/13), not merely "not yet tested".
 */

#include "CudaContext.h"
#include "openmm/LocalManyBodyResidualKernels.h"
#include "openmm/common/ComputeArray.h"

namespace OpenMM {

class CudaCalcLocalManyBodyResidualForceKernel : public CalcLocalManyBodyResidualForceKernel {
public:
    CudaCalcLocalManyBodyResidualForceKernel(const std::string& name, const Platform& platform, CudaContext& cu) :
        CalcLocalManyBodyResidualForceKernel(name, platform), cu(cu), hasInitializedKernels(false), deviceStateValid(false) {
    }
    ~CudaCalcLocalManyBodyResidualForceKernel();
    void initialize(const System& system, const LocalManyBodyResidualForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);

    // Called by CudaReorderListener (defined in the .cpp) whenever OpenMM
    // changes its device atom ordering. Rebuilds anchorDeviceIds/
    // typeByDevice/isLigandByDevice from the CURRENT cu.getAtomIndexArray()
    // -- this is P0 for G2, not deferred, per the frozen atom-identity
    // contract (Force stores topology/system ids; device order can change
    // any time OpenMM likes).
    void onAtomsReordered();

private:
    void ensureDeviceStateCurrent();
    void buildAndLoadKernels();
    // A2: consolidated single status harvest per force evaluation, replacing
    // the legacy A1/EXP-025 checkDeviceErrorFlag() pattern that was called
    // up to ~6 times/evaluation. Downloads deviceStatusDevice[8] EXACTLY
    // once per execute() call (currently positioned after scatter, matching
    // where the LAST legacy check used to run -- see execute()'s own comment
    // for why this draft has NOT yet attempted PLAN 20.1.3's "move the check
    // before scatter" partial-force optimization: that requires a fault-
    // injection proof this draft has not yet written).
    void checkDeviceStatusOnce(const char* completedStage);
    void allocateCellGrid(int nCellsX, int nCellsY, int nCellsZ);
    // Adds (bound==false) or updates (bound==true) the (posq, posqCorrection)
    // argument pair at slots [idx, idx+1] of any position-touching kernel;
    // see the .cpp definition for the mixed-precision placeholder rule when
    // posqCorrection does not genuinely exist.
    //
    // EXP-028: bound/idx exist because of a real perf bug (see
    // PLAN_EXP-028 / the .cpp's execute() doc comment): every kernel
    // argument used to be re-added via addArg() on EVERY execute() call.
    // addArg() permanently APPENDS a new argument slot rather than updating
    // one in place -- the CUDA ComputeKernel::execute() implementation walks
    // its whole (ever-growing) argument vector on every launch, so per-step
    // CPU-side launch prep cost grew without bound over a long trajectory
    // (confirmed: 2.74->6.22 ms/step over 30,000 steps in a real online
    // run). The fix: bind every argument slot exactly ONCE via addArg() the
    // first time a given kernel is invoked (bound==false), then use
    // setArg(idx, ...) on every later invocation (bound==true) -- same
    // total argument count and order every call, just not re-appended.
    void addPosArgs(ComputeKernel& kernel, bool bound, int idx);

    CudaContext& cu;
    bool hasInitializedKernels;
    bool deviceStateValid;  // false right after construction or a reorder; rebuilt lazily before the next execute()

    // ---- host-side bookkeeping (system/topology space, stable across reorders) ----
    int numLigandAtoms = 0, numRadialBasis = 0, typeCount = 0;
    std::vector<int> ligandSystemIds;   // [numLigandAtoms], SORTED ascending (matches Reference convention)
    std::vector<int> typeBySystemId;    // [numParticles]
    double innerCutoffAngstrom = 0, outerCutoffAngstrom = 0, bMaxReduced = 0, radialWidthAngstrom = 0;
    int64_t maxEdges = 0, maxNeighborsPerLigand = 0, maxEnvironmentAtoms = 0;
    double kBTKilojoulePerMole = 0.0;

    // ---- G3 (local CSR): skinAngstrom==0 => G2 legacy brute-force path (see header comment) ----
    bool g3Enabled = false;
    double skinAngstrom = 0.0;
    int64_t candidateListCapacity = 0;
    // Cell grid is fixed at the first rebuild (fixed-box NVT assumption); a
    // later box change that would require a DIFFERENT (nCellsX,nCellsY,nCellsZ)
    // fails closed (UNSUPPORTED_BOX) instead of reallocating -- see header.
    bool cellGridAllocated = false;
    int allocatedNCellsX = 0, allocatedNCellsY = 0, allocatedNCellsZ = 0;
    // Host-side rebuild-trigger bookkeeping. candidateListValid=false forces
    // a host-issued rebuild this step (skips running the device K0
    // displacement kernel against what would otherwise be stale/undefined
    // lastRebuildPositions -- see .cpp for why this is safe). lastRebuildBox
    // is compared against the CURRENT box every step (cheap: 9 doubles we
    // already marshal for the face-height safety check) to detect box
    // changes without any extra device round-trip.
    bool candidateListValid = false;
    double lastRebuildBox[3][3] = {{0,0,0},{0,0,0},{0,0,0}};

    // ---- persistent device buffers (model parameters; uploaded once in initialize(), never reorder-dependent) ----
    ComputeArray radialCentersDevice;   // real[numRadialBasis]
    ComputeArray pairWeightDevice;      // real[typeCount*typeCount*numRadialBasis]
    ComputeArray typedMlpFlatDevice;    // real[typeCount*EXP025_MLP_STRIDE]

    // ---- device buffers that MUST be rebuilt whenever atoms reorder ----
    ComputeArray anchorDeviceIdsDevice;  // int[numLigandAtoms]
    ComputeArray typeByDeviceDevice;     // int[numParticles], indexed by CURRENT device slot
    ComputeArray isLigandByDeviceDevice; // int[numParticles], indexed by CURRENT device slot

    // ---- per-step scratch (reused every execute() call) ----
    ComputeArray qDevice;               // real[numLigandAtoms]
    ComputeArray dBdqDevice;            // real[numLigandAtoms]
    ComputeArray energyScratchDevice;   // real[1] (mixed accumulation not needed: single scalar)
    ComputeArray neighborCountDevice;   // int[numLigandAtoms] (diagnostics + neighbor-ceiling check)
    ComputeArray uniqueEnvEpochDevice;  // int[numParticles], persistent epoch tags; zero-filled only at initialize/wrap
    // A2: the SOLE error/status carrier. errorFlagDevice (A1's legacy
    // int[1]) is gone -- every kernel that used to write it now writes
    // deviceStatusDevice[EXP026_STATUS_ERROR_CODE/STAGE] via the device-side
    // exp026SetFirstError() first-error-wins helper, and the host reads this
    // ONE array exactly once per evaluation (checkDeviceStatusOnce()).
    ComputeArray deviceStatusDevice;    // int[EXP026_STATUS_WORDS]
    int uniqueEnvironmentEpoch = 0;     // positive evaluation epoch; zero is reserved for untouched tags

    // ---- G3 cell-list + CSR device buffers (only allocated/used when g3Enabled) ----
    ComputeArray cellHeadDevice;         // int[nCellsX*nCellsY*nCellsZ], -1 sentinel; (re)sized once, see allocateCellGrid
    ComputeArray nextAtomDevice;         // int[numParticles], linked-list "next" pointer, indexed by CURRENT device slot
    ComputeArray anchorCandidateCountDevice; // int[numLigandAtoms], K3 output
    ComputeArray anchorOffsetsDevice;    // int[numLigandAtoms+1], K4 output (true prefix-summed CSR offsets)
    ComputeArray edgeAtomsDevice;        // int[candidateListCapacity], K5 output (candidate device ids, <r_list)
    ComputeArray lastRebuildPositionsDevice; // mixed flat [3*numParticles] (x0,y0,z0,x1,...), nm, snapshot as of the
                                              // last successful rebuild; double-element when useDouble||mixedPrecision
    ComputeArray rebuildFlagDevice;      // int[1]; device-owned, never downloaded (see .cpp), persists across steps

    ComputeProgram program;
    ComputeKernel resetStatusKernel, computeQKernel, readoutKernel, scatterForceKernel;
    // G3 pipeline kernels (only created/used when g3Enabled)
    ComputeKernel checkDisplacementKernel, clearCellHeadsKernel, binEnvironmentAtomsKernel,
        countCandidatesKernel, prefixSumOffsetsKernel, fillCandidatesKernel,
        copyLastPositionsKernel, clearRebuildFlagKernel,
        computeQFromCsrKernel, scatterForceFromCsrKernel;

    // EXP-028: one "have this kernel's argument slots been bound yet" flag
    // per kernel invocation SITE in execute() (each kernel has exactly one
    // call site that lays out its full argument list, even the ones only
    // invoked conditionally -- e.g. checkDisplacementKernel only runs when
    // a rebuild is not forced). false until that kernel's FIRST real
    // invocation (which still uses addArg(), same as before); every
    // invocation after that uses setArg() against the same slots instead.
    // See addPosArgs()'s doc comment (above) for why this exists.
    bool argsBoundReset = false, argsBoundComputeQ = false, argsBoundReadout = false, argsBoundScatterForce = false;
    bool argsBoundCheckDisplacement = false, argsBoundClearCellHeads = false, argsBoundBinEnv = false,
        argsBoundCountCandidates = false, argsBoundPrefixSum = false, argsBoundFillCandidates = false,
        argsBoundCopyLastPositions = false, argsBoundClearRebuildFlag = false, argsBoundComputeQFromCsr = false,
        argsBoundScatterForceFromCsr = false;

    class ReorderListenerImpl;
    ReorderListenerImpl* reorderListener = nullptr;
};

} // namespace OpenMM

#endif /*CUDA_LOCALMANYBODYRESIDUAL_KERNELS_H_*/
