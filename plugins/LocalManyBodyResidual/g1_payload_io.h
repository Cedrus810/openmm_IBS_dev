#ifndef EXP025_G1_PAYLOAD_IO_H_
#define EXP025_G1_PAYLOAD_IO_H_

// EXP-025 G1 offline test tooling ONLY. Depends on yyjson + OpenSSL.
//
// The production plugin (ReferenceLocalManyBodyResidualKernels.cpp /
// CudaLocalManyBodyResidualKernels.cpp) must NEVER include this header --
// it has no business parsing JSON or reading files. This is purely how test
// drivers (g1_reference_oracle.cpp, g1g_openmm_reference_parity_test.cpp)
// load the frozen EXP-020 checkpoint export and the canonical fixture to
// populate a real LocalManyBodyResidualForce / exp025_g1::ModelParams.
#include "g1_math_core.h"

#include <yyjson.h>
#include <openssl/evp.h>

#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace exp025_g1 {

inline std::string sha256Hex(const unsigned char* data, size_t len) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digestLen = 0;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx || EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, data, len) != 1 ||
        EVP_DigestFinal_ex(ctx, digest, &digestLen) != 1) {
        if (ctx) EVP_MD_CTX_free(ctx);
        throw MathError("sha256 computation failed");
    }
    EVP_MD_CTX_free(ctx);
    static const char* hex = "0123456789abcdef";
    std::string out(digestLen * 2, '0');
    for (unsigned int i = 0; i < digestLen; i++) {
        out[2 * i] = hex[(digest[i] >> 4) & 0xF];
        out[2 * i + 1] = hex[digest[i] & 0xF];
    }
    return out;
}

inline std::vector<unsigned char> readFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw MathError("cannot open " + path);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

inline double jNum(yyjson_val* v, const char* key) {
    yyjson_val* f = yyjson_obj_get(v, key);
    if (!f) throw MathError(std::string("missing JSON field: ") + key);
    return yyjson_get_num(f);
}
inline int64_t jInt(yyjson_val* v, const char* key) {
    yyjson_val* f = yyjson_obj_get(v, key);
    if (!f) throw MathError(std::string("missing JSON field: ") + key);
    return yyjson_get_sint(f);
}
inline std::string jStr(yyjson_val* v, const char* key) {
    yyjson_val* f = yyjson_obj_get(v, key);
    if (!f) throw MathError(std::string("missing JSON field: ") + key);
    const char* s = yyjson_get_str(f);
    if (!s) throw MathError(std::string("JSON field not a string: ") + key);
    return std::string(s);
}
inline yyjson_val* jObj(yyjson_val* v, const char* key) {
    yyjson_val* f = yyjson_obj_get(v, key);
    if (!f) throw MathError(std::string("missing JSON field: ") + key);
    return f;
}
inline std::vector<double> jDoubleArray(yyjson_val* arr) {
    std::vector<double> out;
    size_t idx, max;
    yyjson_val* v;
    yyjson_arr_foreach(arr, idx, max, v) { out.push_back(yyjson_get_num(v)); }
    return out;
}
inline std::vector<int64_t> jIntArray(yyjson_val* arr) {
    std::vector<int64_t> out;
    size_t idx, max;
    yyjson_val* v;
    yyjson_arr_foreach(arr, idx, max, v) { out.push_back(yyjson_get_sint(v)); }
    return out;
}

struct LoadedPayload {
    ModelParams model;
    std::vector<int64_t> typeVocabulary;
    std::vector<int64_t> ligandTopologyIndices;
};

// Loads + fail-closed-verifies r1_model_payload_v1.json + r1_model_weights_f64.bin
// (see scripts/export_exp025_g1_reference_payload.py). Every tensor's sha256
// is re-checked against the manifest while reading it back from the blob.
inline LoadedPayload loadModelPayload(const std::string& payloadJsonPath, const std::string& weightsBinPath) {
    yyjson_doc* doc = yyjson_read_file(payloadJsonPath.c_str(), 0, nullptr, nullptr);
    if (!doc) throw MathError("failed to parse " + payloadJsonPath);
    yyjson_val* root = yyjson_doc_get_root(doc);

    std::string schemaVersion = jStr(root, "schema_version");
    if (schemaVersion != "exp025-g1-r1-reference-payload-v1")
        throw MathError("unexpected payload schema_version: " + schemaVersion);

    LoadedPayload loaded;
    yyjson_val* config = jObj(root, "config");
    ModelParams& model = loaded.model;
    model.nLigandAtoms = (int)jInt(config, "n_ligand_atoms");
    model.nRadialBasis = (int)jInt(config, "n_radial_basis");
    model.innerCutoffAngstrom = jNum(config, "inner_cutoff_angstrom");
    model.outerCutoffAngstrom = jNum(config, "outer_cutoff_angstrom");
    model.bMaxReduced = jNum(config, "b_max_reduced");
    model.maxEdges = jInt(config, "max_edges");
    model.maxNeighborsPerLigand = jInt(config, "max_neighbors_per_ligand");
    model.maxEnvironmentAtoms = jInt(config, "max_environment_atoms");
    loaded.typeVocabulary = jIntArray(jObj(config, "type_vocabulary"));
    model.typeCount = (int)loaded.typeVocabulary.size();
    loaded.ligandTopologyIndices = jIntArray(jObj(root, "ligand_topology_indices"));
    if ((int)loaded.ligandTopologyIndices.size() != model.nLigandAtoms)
        throw MathError("ligand_topology_indices count disagrees with n_ligand_atoms");

    yyjson_val* archFacts = jObj(root, "architecture_facts_not_in_config");
    int hiddenRho = (int)jInt(archFacts, "hidden_rho");
    if (hiddenRho != 16) throw MathError("this loader hardcodes hidden_rho=16; payload says " + std::to_string(hiddenRho));

    std::vector<unsigned char> blob = readFile(weightsBinPath);
    yyjson_val* weightsFile = jObj(root, "weights_file");
    std::string expectedBlobSha = jStr(weightsFile, "sha256");
    std::string actualBlobSha = sha256Hex(blob.data(), blob.size());
    if (actualBlobSha != expectedBlobSha)
        throw MathError("weights blob sha256 mismatch: expected " + expectedBlobSha + " got " + actualBlobSha);

    std::map<std::string, std::vector<double>> tensors;
    yyjson_val* manifest = jObj(root, "tensor_manifest");
    size_t idx, max;
    yyjson_val* entry;
    yyjson_arr_foreach(manifest, idx, max, entry) {
        std::string name = jStr(entry, "name");
        int64_t offset = jInt(entry, "byte_offset");
        int64_t byteCount = jInt(entry, "byte_count");
        std::string expectedSha = jStr(entry, "sha256");
        if (offset < 0 || byteCount < 0 || (size_t)(offset + byteCount) > blob.size())
            throw MathError("tensor " + name + ": offset/byte_count out of range");
        std::string actualSha = sha256Hex(blob.data() + offset, (size_t)byteCount);
        if (actualSha != expectedSha) throw MathError("tensor " + name + ": sha256 mismatch reading back from blob");
        size_t count = (size_t)byteCount / sizeof(double);
        std::vector<double> values(count);
        std::memcpy(values.data(), blob.data() + offset, (size_t)byteCount);
        tensors[name] = std::move(values);
    }

    model.radialCenters = tensors.at("radial_centers");
    if ((int)model.radialCenters.size() != model.nRadialBasis) throw MathError("radial_centers size mismatch");
    model.radialWidth = tensors.at("radial_width").at(0);
    model.pairWeight = tensors.at("pair_weight");
    if ((int)model.pairWeight.size() != model.typeCount * model.typeCount * model.nRadialBasis)
        throw MathError("pair_weight size mismatch");

    model.rho.resize(model.typeCount);
    for (int t = 0; t < model.typeCount; t++) {
        TypedMLP& mlp = model.rho[t];
        const std::string prefix = "rho." + std::to_string(t) + ".";
        const std::vector<double>& w0 = tensors.at(prefix + "0.weight");
        const std::vector<double>& b0 = tensors.at(prefix + "0.bias");
        const std::vector<double>& w2 = tensors.at(prefix + "2.weight");
        const std::vector<double>& b2 = tensors.at(prefix + "2.bias");
        const std::vector<double>& w4 = tensors.at(prefix + "4.weight");
        const std::vector<double>& b4 = tensors.at(prefix + "4.bias");
        if (w0.size() != 16 || b0.size() != 16 || w2.size() != 256 || b2.size() != 16 || w4.size() != 16 || b4.size() != 1)
            throw MathError("rho." + std::to_string(t) + ": unexpected tensor shapes");
        for (int k = 0; k < 16; k++) { mlp.W0[k] = w0[k]; mlp.b0[k] = b0[k]; }
        for (int o = 0; o < 16; o++)
            for (int k = 0; k < 16; k++) mlp.W2[o][k] = w2[(size_t)o * 16 + k];
        for (int k = 0; k < 16; k++) mlp.b2[k] = b2[k];
        for (int k = 0; k < 16; k++) mlp.W4[k] = w4[k];
        mlp.b4 = b4[0];
    }

    yyjson_doc_free(doc);
    return loaded;
}

// Loads canonical_fixture_v1.bin (see scripts/export_exp025_g1_canonical_fixture.py).
inline AtomSystemView loadFixture(const std::string& path) {
    std::vector<unsigned char> data = readFile(path);
    if (data.size() < 24) throw MathError("fixture file too small");
    char magic[9] = {0};
    std::memcpy(magic, data.data(), 8);
    if (std::string(magic) != "EXP025F1") throw MathError("fixture magic mismatch");
    uint32_t formatVersion, endianCanary, nAtomsU, nLigandU;
    std::memcpy(&formatVersion, data.data() + 8, 4);
    std::memcpy(&endianCanary, data.data() + 12, 4);
    std::memcpy(&nAtomsU, data.data() + 16, 4);
    std::memcpy(&nLigandU, data.data() + 20, 4);
    if (formatVersion != 1) throw MathError("unsupported fixture format_version");
    if (endianCanary != 0x01020304u) throw MathError("fixture endian canary mismatch -- byte order incompatible");

    AtomSystemView fx;
    fx.nAtoms = (int)nAtomsU;
    fx.nLigand = (int)nLigandU;
    size_t offset = 24;
    size_t posBytes = (size_t)fx.nAtoms * 3 * sizeof(double);
    size_t boxBytes = 9 * sizeof(double);
    size_t typeBytes = (size_t)fx.nAtoms * sizeof(int32_t);
    size_t ligandBytes = (size_t)fx.nLigand * sizeof(int32_t);
    if (data.size() != offset + posBytes + boxBytes + typeBytes + ligandBytes)
        throw MathError("fixture file size does not match header-declared layout");

    fx.positionsNm.resize(fx.nAtoms);
    std::vector<double> flatPos((size_t)fx.nAtoms * 3);
    std::memcpy(flatPos.data(), data.data() + offset, posBytes);
    offset += posBytes;
    for (int i = 0; i < fx.nAtoms; i++) fx.positionsNm[i] = {flatPos[3 * i], flatPos[3 * i + 1], flatPos[3 * i + 2]};

    std::array<double, 9> flatBox{};
    std::memcpy(flatBox.data(), data.data() + offset, boxBytes);
    offset += boxBytes;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) fx.boxNm[r][c] = flatBox[r * 3 + c];

    fx.atomTypeIndex.resize(fx.nAtoms);
    std::memcpy(fx.atomTypeIndex.data(), data.data() + offset, typeBytes);
    offset += typeBytes;

    fx.ligandTopologyIds.resize(fx.nLigand);
    std::memcpy(fx.ligandTopologyIds.data(), data.data() + offset, ligandBytes);
    offset += ligandBytes;

    return fx;
}

}  // namespace exp025_g1

#endif  // EXP025_G1_PAYLOAD_IO_H_
