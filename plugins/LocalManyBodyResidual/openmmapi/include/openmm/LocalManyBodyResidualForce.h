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

#ifndef OPENMM_LOCALMANYBODYRESIDUALFORCE_H_
#define OPENMM_LOCALMANYBODYRESIDUALFORCE_H_

/*
 * EXP-025 G1-G: real R1 parameter contract.
 *
 * This Force holds the frozen EXP-020 R1 checkpoint's numeric content
 * (see local_residual/softlift.py R1Model and
 * scripts/export_exp025_g1_reference_payload.py) plus the physical constant
 * needed to turn the model's reduced output into an energy.
 *
 * ================= exactly-once kBT contract (frozen, not a G1 choice) =================
 * Per PLAN_EXP-025_local_manybody_cuda.md section 3.5: this Force/Kernel
 * outputs the RAW BASIS energy U_B = kBT * B, in kJ/mol, with kBT applied
 * exactly once, HERE, not deferred to any outer wrapper. The outer
 * OuterLambda/IBS wrapper only ever applies A_k and an energy offset to this
 * already-physical U_B; it must never apply kBT itself, and this Force must
 * never apply A_k or an offset. temperatureKelvin is held and serialized so
 * this contract is self-contained and auditable from the XML alone.
 *
 * kBT = MOLAR_GAS_CONSTANT_R * temperatureKelvin, using the same CODATA 2018
 * constant OpenMM's Python unit system uses
 * (openmm/unit/constants.py: AVOGADRO_CONSTANT_NA * BOLTZMANN_CONSTANT_kB =
 * 8.31446261815324 J/(K mol)). OpenMM's C++ API does not expose this
 * constant directly, so it is a literal here -- see getMolarGasConstantR().
 *
 * ================= units =================
 * Distances/cutoffs in this Force's stored parameters are in Angstrom
 * (matching local_residual/softlift.py), NOT nm. The Kernel is responsible
 * for converting OpenMM's nm-space positions internally; see
 * g1_math_core.h. Gradients are converted Angstrom->nm (factor of 10) and
 * then to a physical force via -kBT exactly once, inside the Kernel -- never
 * twice, never by the caller.
 *
 * ================= no in-place parameter updates yet =================
 * updateParametersInContext() is not yet implemented for same-shape
 * coefficient updates (see PLAN section 4.2); changing any parameter after
 * a Context has been created requires creating a new Context. This is a
 * deliberate G1 scope limit, not a silent gap -- ForceImpl::
 * updateParametersInContext() will throw rather than silently no-op.
 */

#include "openmm/Context.h"
#include "openmm/Force.h"
#include "internal/windowsExportLocalManyBodyResidual.h"
#include <string>
#include <vector>

namespace OpenMM {

/**
 * One ligand-type-specific readout network:
 * Linear(1,16) -> SiLU -> Linear(16,16) -> SiLU -> Linear(16,1).
 * Weight layout matches PyTorch nn.Linear: flattened row-major [out][in].
 */
struct OPENMM_EXPORT_LMBR LocalManyBodyTypedMLP {
    std::vector<double> w0;  // 16 (16x1 flattened)
    std::vector<double> b0;  // 16
    std::vector<double> w2;  // 256 (16x16 flattened, row-major [out][in])
    std::vector<double> b2;  // 16
    std::vector<double> w4;  // 16 (1x16 flattened)
    double b4 = 0.0;
};

class OPENMM_EXPORT_LMBR LocalManyBodyResidualForce : public Force {
public:
    LocalManyBodyResidualForce();

    // ---- physical constant (exactly-once kBT contract, see file header) ----
    void setTemperatureKelvin(double temperatureKelvin);
    double getTemperatureKelvin() const;
    /** kBT in kJ/mol at the currently held temperature. */
    double getKBTKilojoulePerMole() const;
    /** The CODATA 2018 molar gas constant R, in kJ/(mol K), matching
     *  openmm/unit/constants.py exactly (OpenMM's C++ API has no named
     *  symbol for this). */
    static double getMolarGasConstantRKilojoulePerMoleKelvin();

    // ---- topology identity ----
    void setLigandTopologyIds(const std::vector<int>& ligandTopologyIds);
    const std::vector<int>& getLigandTopologyIds() const;
    /** One entry per atom in the System; value is an index into
     *  getTypeVocabulary(), NOT a raw atomic number. */
    void setAtomTypeIndex(const std::vector<int>& atomTypeIndex);
    const std::vector<int>& getAtomTypeIndex() const;
    /** Provenance only (documents what the type indices mean); not consumed
     *  by the math, which only ever sees already-mapped indices. */
    void setTypeVocabulary(const std::vector<int>& atomicNumbers);
    const std::vector<int>& getTypeVocabulary() const;

    // ---- radial density basis ----
    void setInnerCutoffAngstrom(double innerCutoffAngstrom);
    double getInnerCutoffAngstrom() const;
    void setOuterCutoffAngstrom(double outerCutoffAngstrom);
    double getOuterCutoffAngstrom() const;
    void setRadialCenters(const std::vector<double>& radialCentersAngstrom);
    const std::vector<double>& getRadialCenters() const;
    void setRadialWidthAngstrom(double radialWidthAngstrom);
    double getRadialWidthAngstrom() const;
    /** Flattened [type_count][type_count][n_radial_basis], row-major. */
    void setPairWeight(const std::vector<double>& pairWeight);
    const std::vector<double>& getPairWeight() const;

    // ---- typed nonlinear readout ----
    void setTypedMLP(int typeIndex, const LocalManyBodyTypedMLP& mlp);
    const LocalManyBodyTypedMLP& getTypedMLP(int typeIndex) const;
    int getNumTypes() const;
    int getNumRadialBasis() const;

    void setBMaxReduced(double bMaxReduced);
    double getBMaxReduced() const;

    // ---- hard capacity ceilings (fail-closed, not truncation) ----
    // Scoped to ACTIVE edges only (r < outer_cutoff, i.e. the physical support
    // domain of the model). G3's candidate list (r < outer_cutoff + skin) has
    // its own, separate, larger ceiling -- see setCandidateListCapacity below.
    void setCapacityCeilings(int maxEdges, int maxNeighborsPerLigand, int maxEnvironmentAtoms);
    void getCapacityCeilings(int& maxEdges, int& maxNeighborsPerLigand, int& maxEnvironmentAtoms) const;

    // ---- G3: local CSR/Verlet list (PLAN section 6) ----
    // skinAngstrom == 0 (the default) selects the G2 legacy brute-force code
    // path (kept byte-for-byte, so the sealed G2 gate/report never silently
    // changes behavior). skinAngstrom > 0 selects the G3 local-CSR code path
    // and REQUIRES candidateListCapacity > 0 too (validated together at
    // kernel initialize() time, not here -- this Force is a pure data holder).
    void setSkinAngstrom(double skinAngstrom);
    double getSkinAngstrom() const;
    // Hard ceiling on the TOTAL number of <r_list candidates across all 41
    // anchors (r_list = outer_cutoff + skin). Deliberately separate from
    // maxEdges above: candidates include the skin shell so this must be
    // larger, and conflating the two would either reject a normal skin list
    // or silently widen the active-support ceiling.
    void setCandidateListCapacity(int candidateListCapacity);
    int getCandidateListCapacity() const;

    // ---- provenance (opaque tags, for reports/manifests only) ----
    void setSourceCheckpointSha256(const std::string& sha256);
    const std::string& getSourceCheckpointSha256() const;

protected:
    ForceImpl* createImpl() const;

private:
    double temperatureKelvin = 0.0;
    std::vector<int> ligandTopologyIds;
    std::vector<int> atomTypeIndex;
    std::vector<int> typeVocabulary;
    double innerCutoffAngstrom = 0.0, outerCutoffAngstrom = 0.0;
    std::vector<double> radialCenters;
    double radialWidthAngstrom = 0.0;
    std::vector<double> pairWeight;
    std::vector<LocalManyBodyTypedMLP> rho;
    double bMaxReduced = 0.0;
    int maxEdges = 0, maxNeighborsPerLigand = 0, maxEnvironmentAtoms = 0;
    double skinAngstrom = 0.0;
    int candidateListCapacity = 0;
    std::string sourceCheckpointSha256;
};

} // namespace OpenMM

#endif /*OPENMM_LOCALMANYBODYRESIDUALFORCE_H_*/
