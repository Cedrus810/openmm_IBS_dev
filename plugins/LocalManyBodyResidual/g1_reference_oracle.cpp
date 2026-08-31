// EXP-025 G1 standalone Reference oracle.
//
// CPU double precision, no PyTorch, no OpenMM. Uses g1_math_core.h (the same
// math the real OpenMM ReferenceCalcLocalManyBodyResidualForceKernel uses)
// plus g1_payload_io.h (yyjson-based test-only loading) to read:
//   - r1_model_payload_v1.json + r1_model_weights_f64.bin  (frozen EXP-020 R1 weights)
//   - canonical_fixture_v1.bin                              (positions_nm/box_nm/types, NOT edges)
// and reconstructs edges/q/rho/B/gradient itself. Compares against
// canonical_fixture_expected_v1.json (comparison-only -- its edge list is
// NEVER used as computation input here).
//
// Coverage order (G1-B..G1-F): edge/MIC parity -> q[41] parity -> typed
// rho(q)/rho(0) parity -> raw S / B parity -> analytic gradient + finite
// difference. G1-G (real OpenMM ReferenceCalcLocalManyBodyResidualForceKernel
// parity, including the kBT/energy/force contract) lives in
// g1g_openmm_reference_parity_test.cpp.
#include "g1_math_core.h"
#include "g1_payload_io.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace exp025_g1;

namespace {

int g_failures = 0;

void expectClose(const std::string& label, double actual, double expected, double tol) {
    double diff = std::fabs(actual - expected);
    bool ok = diff <= tol;
    std::cout << "  " << (ok ? "PASS" : "FAIL") << " " << label << ": actual=" << actual
              << " expected=" << expected << " |diff|=" << diff << " tol=" << tol << "\n";
    if (!ok) g_failures++;
}

void expectEqualStr(const std::string& label, const std::string& actual, const std::string& expected) {
    bool ok = actual == expected;
    std::cout << "  " << (ok ? "PASS" : "FAIL") << " " << label << ": actual=" << actual << " expected=" << expected << "\n";
    if (!ok) g_failures++;
}

double forwardBOnly(AtomSystemView fx, const ModelParams& model, int atom, int coord, double deltaNm) {
    fx.positionsNm[atom][coord] += deltaNm;
    EnumerationResult enumeration = enumerateEdges(fx, model);
    ForwardResult result = evaluate(fx, model, enumeration);
    return result.bReduced;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: g1_reference_oracle <r1_model_payload_v1.json> <r1_model_weights_f64.bin> <g1_reference_dir>\n";
        return 2;
    }
    std::string payloadJsonPath = argv[1];
    std::string weightsBinPath = argv[2];
    std::string g1Dir = argv[3];

    try {
        LoadedPayload loaded = loadModelPayload(payloadJsonPath, weightsBinPath);
        const ModelParams& model = loaded.model;
        AtomSystemView fx = loadFixture(g1Dir + "/canonical_fixture_v1.bin");
        std::cout << "loaded model (type_count=" << model.typeCount << ", n_ligand_atoms=" << model.nLigandAtoms
                  << ") and fixture (n_atoms=" << fx.nAtoms << ", n_ligand=" << fx.nLigand << ")\n";

        yyjson_doc* expectedDoc = yyjson_read_file((g1Dir + "/canonical_fixture_expected_v1.json").c_str(), 0, nullptr, nullptr);
        if (!expectedDoc) throw MathError("failed to parse canonical_fixture_expected_v1.json");
        yyjson_val* expectedRoot = yyjson_doc_get_root(expectedDoc);
        yyjson_val* expectedEdges = jObj(expectedRoot, "edges");

        std::cout << "\n=== G1-B: edge / MIC parity ===\n";
        EnumerationResult enumeration = enumerateEdges(fx, model);
        expectClose("edge count", (double)enumeration.edges.size(), jNum(expectedEdges, "count"), 0.0);
        expectClose("observed_unique_environment_atoms", (double)enumeration.uniqueEnvironmentAtoms.size(),
                    jNum(expectedEdges, "observed_unique_environment_atoms"), 0.0);
        int maxNeighbors = 0;
        for (int c : enumeration.perLigandNeighborCount) maxNeighbors = std::max(maxNeighbors, c);
        expectClose("observed_max_neighbors_per_ligand", (double)maxNeighbors,
                    jNum(expectedEdges, "observed_max_neighbors_per_ligand"), 0.0);

        std::vector<std::pair<int, int>> pairs;
        for (const EdgeGeometry& e : enumeration.edges)
            pairs.emplace_back(enumeration.sortedLigandTopologyIds[e.ligandLocal], e.environmentAtom);
        std::sort(pairs.begin(), pairs.end());
        std::vector<unsigned char> pairBytes;
        pairBytes.reserve(pairs.size() * 16);
        for (auto& [lig, env] : pairs) {
            int64_t l = lig, e = env;
            unsigned char buf[16];
            std::memcpy(buf, &l, 8);
            std::memcpy(buf + 8, &e, 8);
            pairBytes.insert(pairBytes.end(), buf, buf + 16);
        }
        std::string pairsSha = sha256Hex(pairBytes.data(), pairBytes.size());
        expectEqualStr("sorted_ligand_environment_topology_pairs_sha256", pairsSha,
                        jStr(expectedEdges, "sorted_ligand_environment_topology_pairs_sha256"));

        std::cout << "\n=== G1-C: q[41] parity ===\n";
        ForwardResult result = evaluate(fx, model, enumeration);
        std::vector<double> expectedQ = jDoubleArray(jObj(expectedRoot, "q"));
        for (int i = 0; i < model.nLigandAtoms; i++)
            expectClose("q[" + std::to_string(i) + "]", result.q[i], expectedQ.at(i), 1e-10);

        std::cout << "\n=== G1-D: typed rho(q) / rho(0) / per_ligand parity ===\n";
        std::vector<double> expectedRhoQ = jDoubleArray(jObj(expectedRoot, "rho_q"));
        std::vector<double> expectedRhoZero = jDoubleArray(jObj(expectedRoot, "rho_zero"));
        std::vector<double> expectedPerLigand = jDoubleArray(jObj(expectedRoot, "per_ligand"));
        for (int i = 0; i < model.nLigandAtoms; i++) {
            expectClose("rho_q[" + std::to_string(i) + "]", result.rhoQ[i], expectedRhoQ.at(i), 1e-10);
            expectClose("rho_zero[" + std::to_string(i) + "]", result.rhoZero[i], expectedRhoZero.at(i), 1e-10);
            expectClose("per_ligand[" + std::to_string(i) + "]", result.perLigand[i], expectedPerLigand.at(i), 1e-10);
        }

        std::cout << "\n=== G1-E: raw S / B parity ===\n";
        expectClose("raw_S", result.rawS, jNum(expectedRoot, "raw_S"), 1e-9);
        expectClose("B_reduced", result.bReduced, jNum(expectedRoot, "B_reduced"), 1e-9);

        std::cout << "\n=== G1-F: analytic gradient + finite difference ===\n";
        yyjson_val* gradSection = jObj(expectedRoot, "reduced_gradient_dB_dx_nm");
        yyjson_val* ligandGradArr = jObj(gradSection, "ligand_by_local_index");
        double maxLigandGradDiff = 0.0;
        for (int i = 0; i < model.nLigandAtoms; i++) {
            int ligandAtom = enumeration.sortedLigandTopologyIds[i];
            std::array<double, 3> gradAngstrom = {0, 0, 0};
            auto it = result.gradientAngstromByAtom.find(ligandAtom);
            if (it != result.gradientAngstromByAtom.end()) gradAngstrom = it->second;
            yyjson_val* triple = yyjson_arr_get(ligandGradArr, i);
            for (int c = 0; c < 3; c++) {
                double gradNm = 10.0 * gradAngstrom[c];
                double expectedVal = yyjson_get_num(yyjson_arr_get(triple, c));
                maxLigandGradDiff = std::max(maxLigandGradDiff, std::fabs(gradNm - expectedVal));
            }
        }
        expectClose("max|ligand dB/dx_nm diff| over all 41x3 components", maxLigandGradDiff, 0.0, 1e-8);

        yyjson_val* envGradObj = jObj(gradSection, "environment_by_topology_id");
        double maxEnvGradDiff = 0.0;
        for (int envAtom : enumeration.uniqueEnvironmentAtoms) {
            std::array<double, 3> gradAngstrom = {0, 0, 0};
            auto it = result.gradientAngstromByAtom.find(envAtom);
            if (it != result.gradientAngstromByAtom.end()) gradAngstrom = it->second;
            yyjson_val* triple = yyjson_obj_get(envGradObj, std::to_string(envAtom).c_str());
            if (!triple) throw MathError("expected fixture has no gradient entry for environment atom " + std::to_string(envAtom));
            for (int c = 0; c < 3; c++) {
                double gradNm = 10.0 * gradAngstrom[c];
                double expectedVal = yyjson_get_num(yyjson_arr_get(triple, c));
                maxEnvGradDiff = std::max(maxEnvGradDiff, std::fabs(gradNm - expectedVal));
            }
        }
        expectClose("max|environment dB/dx_nm diff|", maxEnvGradDiff, 0.0, 1e-8);

        std::cout << "  PASS non-participating atoms: " << (fx.nAtoms - (int)enumeration.uniqueEnvironmentAtoms.size() - model.nLigandAtoms)
                  << " atoms never accumulated into (exact zero by construction, not computed)\n";

        int fdAtom = 4583;
        double h = 1e-6;  // nm
        for (int coord = 0; coord < 3; coord++) {
            double bPlus = forwardBOnly(fx, model, fdAtom, coord, +h);
            double bMinus = forwardBOnly(fx, model, fdAtom, coord, -h);
            double central = (bPlus - bMinus) / (2.0 * h);
            std::array<double, 3> gradAngstrom = result.gradientAngstromByAtom.at(fdAtom);
            double analytic = 10.0 * gradAngstrom[coord];
            double denom = std::max({std::fabs(central), std::fabs(analytic), 1e-12});
            double relError = std::fabs(central - analytic) / denom;
            expectClose("finite-difference atom " + std::to_string(fdAtom) + " coord " + std::to_string(coord) +
                            " relative error", relError, 0.0, 1e-3);
        }

        std::cout << "\n=== G1 STANDALONE ORACLE: " << (g_failures == 0 ? "PASS" : "FAIL") << " (" << g_failures << " failing checks) ===\n";
        std::cout << "NOTE: this is the standalone C++ oracle only. G1-G (real OpenMM Reference "
                     "Kernel/Context parity) is a separate step, see g1g_openmm_reference_parity_test.cpp.\n";
        yyjson_doc_free(expectedDoc);
        return g_failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "G1 ORACLE FAIL-CLOSED: " << e.what() << "\n";
        return 1;
    }
}
