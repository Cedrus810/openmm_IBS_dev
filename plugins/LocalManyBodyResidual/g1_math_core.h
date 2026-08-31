#ifndef EXP025_G1_MATH_CORE_H_
#define EXP025_G1_MATH_CORE_H_

// EXP-025 G1 shared math core -- NO file I/O, NO JSON, NO OpenMM dependency.
//
// This header is included by BOTH the production
// ReferenceCalcLocalManyBodyResidualForceKernel (plugins/LocalManyBodyResidual/
// platforms/reference/src/ReferenceLocalManyBodyResidualKernels.cpp) and the
// standalone offline oracle/test tooling (g1_reference_oracle.cpp,
// g1g_openmm_reference_parity_test.cpp). Using the exact same functions in
// both places is deliberate: it is the strongest available guarantee that
// "the real OpenMM kernel" and "the already-validated standalone oracle"
// cannot silently diverge by having someone edit one copy and not the other.
//
// Units: this file operates entirely in Angstrom for distances/cutoffs
// (matching local_residual/softlift.py) and expects the caller to have
// already converted from OpenMM's native nm. AtomSystemView below stores
// positions/box in nm (as OpenMM does) and the x10 conversion happens
// explicitly inside enumerateEdges()/evaluate(), never silently.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace exp025_g1 {

struct MathError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// ---------------- typed MLP: Linear(1,16) -> SiLU -> Linear(16,16) -> SiLU -> Linear(16,1) ----------------
// Weight layout matches PyTorch nn.Linear: W[out][in], y = W @ x + b.
struct TypedMLP {
    std::array<double, 16> W0{};                  // (16,1) flattened
    std::array<double, 16> b0{};
    std::array<std::array<double, 16>, 16> W2{};  // W2[out][in]
    std::array<double, 16> b2{};
    std::array<double, 16> W4{};                  // (1,16) flattened
    double b4 = 0.0;

    static double silu(double x) { return x / (1.0 + std::exp(-x)); }
    static double siluGrad(double x) {
        double s = 1.0 / (1.0 + std::exp(-x));
        return s * (1.0 + x * (1.0 - s));
    }

    // Returns (value, d(value)/dx).
    std::pair<double, double> evalWithGrad(double x) const {
        std::array<double, 16> h0{}, a0{}, da0{}, a2{}, da2{};
        for (int k = 0; k < 16; k++) {
            h0[k] = W0[k] * x + b0[k];
            a0[k] = silu(h0[k]);
            da0[k] = siluGrad(h0[k]) * W0[k];
        }
        for (int o = 0; o < 16; o++) {
            double h = b2[o], dh = 0.0;
            for (int k = 0; k < 16; k++) {
                h += W2[o][k] * a0[k];
                dh += W2[o][k] * da0[k];
            }
            a2[o] = silu(h);
            da2[o] = siluGrad(h) * dh;
        }
        double value = b4, grad = 0.0;
        for (int k = 0; k < 16; k++) {
            value += W4[k] * a2[k];
            grad += W4[k] * da2[k];
        }
        return {value, grad};
    }

    double eval(double x) const { return evalWithGrad(x).first; }
};

// ---------------- frozen model parameters (numeric content only, no I/O) ----------------
struct ModelParams {
    int nLigandAtoms = 0, nRadialBasis = 0, typeCount = 0;
    double innerCutoffAngstrom = 0, outerCutoffAngstrom = 0, bMaxReduced = 0;
    int64_t maxEdges = 0, maxNeighborsPerLigand = 0, maxEnvironmentAtoms = 0;
    std::vector<double> radialCenters;  // [nRadialBasis]
    double radialWidth = 0;
    std::vector<double> pairWeight;  // flattened [typeCount][typeCount][nRadialBasis]
    std::vector<TypedMLP> rho;       // [typeCount]

    double pairWeightAt(int ligandType, int envType, int p) const {
        return pairWeight[(size_t)(ligandType * typeCount + envType) * nRadialBasis + p];
    }
};

// ---------------- atom system view: positions/box/types (NOT edges) ----------------
struct AtomSystemView {
    int nAtoms = 0, nLigand = 0;
    std::vector<std::array<double, 3>> positionsNm;
    std::array<std::array<double, 3>, 3> boxNm{};
    std::vector<int32_t> atomTypeIndex;      // [nAtoms], index into type_vocabulary
    std::vector<int32_t> ligandTopologyIds;  // [nLigand]
};

// ---------------- geometry: minimum-image displacement with explicit tie fail-closed ----------------
struct Box3x3Inverse {
    std::array<std::array<double, 3>, 3> inv{};
};

inline Box3x3Inverse invertBox(const std::array<std::array<double, 3>, 3>& box) {
    const auto& m = box;
    double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                 m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                 m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if (!std::isfinite(det) || std::fabs(det) < 1e-12) throw MathError("box is singular or near-singular");
    double invDet = 1.0 / det;
    Box3x3Inverse result;
    result.inv[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * invDet;
    result.inv[1][0] = -(m[1][0] * m[2][2] - m[1][2] * m[2][0]) * invDet;
    result.inv[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * invDet;
    result.inv[0][1] = -(m[0][1] * m[2][2] - m[0][2] * m[2][1]) * invDet;
    result.inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invDet;
    result.inv[2][1] = -(m[0][0] * m[2][1] - m[0][1] * m[2][0]) * invDet;
    result.inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invDet;
    result.inv[1][2] = -(m[0][0] * m[1][2] - m[0][2] * m[1][0]) * invDet;
    result.inv[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * invDet;
    return result;
}

constexpr double HALF_BOX_TIE_EPSILON = 1e-9;

// Round-half-away-from-zero for the ordinary case; throws on an exact (to
// within epsilon) half-integer fractional coordinate rather than silently
// picking a convention that may disagree with PyTorch's round-half-to-even
// -- see PLAN_EXP-025_local_manybody_cuda.md section 6.3 half-box tie policy.
inline double roundWithTieGuard(double x) {
    double frac = x - std::floor(x);  // in [0, 1)
    if (std::fabs(frac - 0.5) < HALF_BOX_TIE_EPSILON)
        throw MathError("half-box tie encountered in minimum-image wrapping (fail-closed by design)");
    return std::round(x);
}

// displacement = target - source, minimum-image wrapped, matching
// local_residual/geometry.py minimum_image_displacement() (independent
// re-implementation -- not shared code with the Python side).
inline std::array<double, 3> minimumImageDisplacement(const std::array<double, 3>& source,
                                                        const std::array<double, 3>& target,
                                                        const std::array<std::array<double, 3>, 3>& box,
                                                        const Box3x3Inverse& invBox) {
    std::array<double, 3> cartesian = {target[0] - source[0], target[1] - source[1], target[2] - source[2]};
    std::array<double, 3> fractional{};
    for (int c = 0; c < 3; c++) {
        double acc = 0.0;
        for (int r = 0; r < 3; r++) acc += cartesian[r] * invBox.inv[r][c];
        fractional[c] = acc;
    }
    std::array<double, 3> wrappedFractional{};
    for (int c = 0; c < 3; c++) wrappedFractional[c] = fractional[c] - roundWithTieGuard(fractional[c]);
    std::array<double, 3> wrapped{};
    for (int c = 0; c < 3; c++) {
        double acc = 0.0;
        for (int r = 0; r < 3; r++) acc += wrappedFractional[r] * box[r][c];
        wrapped[c] = acc;
    }
    return wrapped;
}

// ---------------- quintic C2 envelope ----------------

inline double quinticC2(double r, double inner, double outer) {
    if (r <= inner) return 1.0;
    if (r >= outer) return 0.0;
    double x = (r - inner) / (outer - inner);
    double x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
    return 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5;
}

inline double quinticC2Grad(double r, double inner, double outer) {
    if (r <= inner || r >= outer) return 0.0;
    double x = (r - inner) / (outer - inner);
    double x2 = x * x, x3 = x2 * x, x4 = x3 * x;
    double dTransition_dx = -30.0 * x2 + 60.0 * x3 - 30.0 * x4;
    return dTransition_dx / (outer - inner);
}

// ---------------- edge enumeration ----------------

struct EdgeGeometry {
    int ligandLocal;      // 0..nLigand-1, index into the ASCENDING-sorted ligand topology id list
    int environmentAtom;  // global atom index
    double rAngstrom;
    std::array<double, 3> dispAngstrom;  // environment - ligand, wrapped
};

struct EnumerationResult {
    std::vector<int> sortedLigandTopologyIds;
    std::vector<EdgeGeometry> edges;          // ligand-major / environment-ascending order
    std::vector<int> perLigandNeighborCount;  // [nLigand]
    std::set<int> uniqueEnvironmentAtoms;
};

// Rebuilds edges purely from positions/box/types -- NEVER consumes an
// externally-provided edge list. Fails closed (throws) on: r < 0.1 Angstrom,
// a half-box MIC tie, or any of the three EXP-020 hard capacity ceilings
// (max_edges / max_neighbors_per_ligand / max_environment_atoms) being
// exceeded.
inline EnumerationResult enumerateEdges(const AtomSystemView& fx, const ModelParams& model) {
    Box3x3Inverse invBox = invertBox(fx.boxNm);
    std::set<int> ligandSet(fx.ligandTopologyIds.begin(), fx.ligandTopologyIds.end());

    EnumerationResult result;
    result.sortedLigandTopologyIds.assign(fx.ligandTopologyIds.begin(), fx.ligandTopologyIds.end());
    std::sort(result.sortedLigandTopologyIds.begin(), result.sortedLigandTopologyIds.end());
    result.perLigandNeighborCount.assign(result.sortedLigandTopologyIds.size(), 0);

    for (int localIdx = 0; localIdx < (int)result.sortedLigandTopologyIds.size(); localIdx++) {
        int ligandAtom = result.sortedLigandTopologyIds[localIdx];
        int neighborCount = 0;
        for (int envAtom = 0; envAtom < fx.nAtoms; envAtom++) {
            if (ligandSet.count(envAtom)) continue;  // ascending order over all non-ligand atoms
            std::array<double, 3> dispNm = minimumImageDisplacement(fx.positionsNm[ligandAtom], fx.positionsNm[envAtom], fx.boxNm, invBox);
            std::array<double, 3> dispAngstrom = {dispNm[0] * 10.0, dispNm[1] * 10.0, dispNm[2] * 10.0};
            double r = std::sqrt(dispAngstrom[0] * dispAngstrom[0] + dispAngstrom[1] * dispAngstrom[1] + dispAngstrom[2] * dispAngstrom[2]);
            if (r < 0.1)
                throw MathError("near-singular pair distance < 0.1 Angstrom (ligand atom " + std::to_string(ligandAtom) +
                                 ", environment atom " + std::to_string(envAtom) + "): fail-closed");
            if (r < model.outerCutoffAngstrom) {
                result.edges.push_back({localIdx, envAtom, r, dispAngstrom});
                neighborCount++;
                result.uniqueEnvironmentAtoms.insert(envAtom);
                if ((int64_t)result.edges.size() > model.maxEdges)
                    throw MathError("active edge count exceeds max_edges=" + std::to_string(model.maxEdges) + ": fail-closed");
            }
        }
        result.perLigandNeighborCount[localIdx] = neighborCount;
        if ((int64_t)neighborCount > model.maxNeighborsPerLigand)
            throw MathError("ligand atom " + std::to_string(ligandAtom) + " exceeds max_neighbors_per_ligand=" +
                             std::to_string(model.maxNeighborsPerLigand) + ": fail-closed");
    }
    if ((int64_t)result.uniqueEnvironmentAtoms.size() > model.maxEnvironmentAtoms)
        throw MathError("unique environment atom count exceeds max_environment_atoms=" +
                         std::to_string(model.maxEnvironmentAtoms) + ": fail-closed");
    return result;
}

// ---------------- forward + analytic backward ----------------

struct ForwardResult {
    std::vector<double> q;          // [nLigand]
    std::vector<double> rhoQ;       // [nLigand]
    std::vector<double> rhoZero;    // [nLigand]
    std::vector<double> perLigand;  // [nLigand]
    double rawS = 0.0;
    double bReduced = 0.0;
    // Reduced gradient of B (NOT physical force, NOT yet Angstrom->nm converted)
    // keyed by global atom index; only ligand atoms and edge-contacted
    // environment atoms ever appear here.
    std::map<int, std::array<double, 3>> gradientAngstromByAtom;
};

// Convention (independently re-derived, matches PLAN_EXP-025 D-hat/gradient
// discussion): let d_hat = (r_env - r_lig) / r. Then
//   grad_lig(B) = -alpha * d_hat      (ligand gets the NEGATIVE sign)
//   grad_env(B) = +alpha * d_hat      (environment gets the POSITIVE sign)
// where alpha = dB/dr for this edge. This is the GRADIENT of the reduced
// scalar B, not a physical force -- physical force = -kBT * gradient
// (applied by the caller; this function never multiplies by kBT).
inline ForwardResult evaluate(const AtomSystemView& fx, const ModelParams& model, const EnumerationResult& enumeration) {
    ForwardResult out;
    int nLigand = (int)enumeration.sortedLigandTopologyIds.size();
    out.q.assign(nLigand, 0.0);

    auto edgeQAndGrad = [&](double r, int ligandType, int envType, double& edgeQ, double& dEdgeQ_dr) {
        double c2 = quinticC2(r, model.innerCutoffAngstrom, model.outerCutoffAngstrom);
        double dc2 = quinticC2Grad(r, model.innerCutoffAngstrom, model.outerCutoffAngstrom);
        double g = 0.0, dg = 0.0;
        for (int p = 0; p < model.nRadialBasis; p++) {
            double diff = r - model.radialCenters[p];
            double z = diff / model.radialWidth;
            double gp = std::exp(-0.5 * z * z);
            double w = model.pairWeightAt(ligandType, envType, p);
            g += w * gp;
            dg += w * gp * (-diff / (model.radialWidth * model.radialWidth));
        }
        edgeQ = c2 * g;
        dEdgeQ_dr = dc2 * g + c2 * dg;
    };

    for (const EdgeGeometry& e : enumeration.edges) {
        int ligandAtom = enumeration.sortedLigandTopologyIds[e.ligandLocal];
        int ligandType = fx.atomTypeIndex[ligandAtom];
        int envType = fx.atomTypeIndex[e.environmentAtom];
        double edgeQ, dEdgeQ_dr;
        edgeQAndGrad(e.rAngstrom, ligandType, envType, edgeQ, dEdgeQ_dr);
        out.q[e.ligandLocal] += edgeQ;
    }

    out.rhoQ.assign(nLigand, 0.0);
    out.rhoZero.assign(nLigand, 0.0);
    out.perLigand.assign(nLigand, 0.0);
    std::vector<double> dRhoQ_dq(nLigand, 0.0);
    for (int i = 0; i < nLigand; i++) {
        int ligandAtom = enumeration.sortedLigandTopologyIds[i];
        int t = fx.atomTypeIndex[ligandAtom];
        auto [rhoQVal, rhoQGrad] = model.rho[t].evalWithGrad(out.q[i]);
        double rhoZeroVal = model.rho[t].eval(0.0);
        out.rhoQ[i] = rhoQVal;
        out.rhoZero[i] = rhoZeroVal;
        out.perLigand[i] = rhoQVal - rhoZeroVal;
        dRhoQ_dq[i] = rhoQGrad;
        out.rawS += out.perLigand[i];
    }
    double bMax = model.bMaxReduced;
    out.bReduced = bMax * std::tanh(out.rawS / bMax);
    double tanhVal = std::tanh(out.rawS / bMax);
    double sech2 = 1.0 - tanhVal * tanhVal;
    std::vector<double> dB_dq(nLigand, 0.0);
    for (int i = 0; i < nLigand; i++) dB_dq[i] = sech2 * dRhoQ_dq[i];

    for (const EdgeGeometry& e : enumeration.edges) {
        int ligandAtom = enumeration.sortedLigandTopologyIds[e.ligandLocal];
        int ligandType = fx.atomTypeIndex[ligandAtom];
        int envType = fx.atomTypeIndex[e.environmentAtom];
        double edgeQ, dEdgeQ_dr;
        edgeQAndGrad(e.rAngstrom, ligandType, envType, edgeQ, dEdgeQ_dr);
        double dB_dr = dB_dq[e.ligandLocal] * dEdgeQ_dr;
        std::array<double, 3> unitVec = {e.dispAngstrom[0] / e.rAngstrom, e.dispAngstrom[1] / e.rAngstrom, e.dispAngstrom[2] / e.rAngstrom};
        auto& gLig = out.gradientAngstromByAtom[ligandAtom];
        auto& gEnv = out.gradientAngstromByAtom[e.environmentAtom];
        for (int c = 0; c < 3; c++) {
            gLig[c] += dB_dr * (-unitVec[c]);  // ligand: NEGATIVE
            gEnv[c] += dB_dr * (unitVec[c]);   // environment: POSITIVE
        }
    }
    return out;
}

}  // namespace exp025_g1

#endif  // EXP025_G1_MATH_CORE_H_
