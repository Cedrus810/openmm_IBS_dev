import copy
import hashlib
import json

import pytest

from local_residual.atom_mapping import (
    AtomMappingError,
    AtomMappingIntegrityError,
    atom_mapping_sha256,
    build_atom_mapping,
    load_atom_mapping,
    validate_atom_mapping,
)
from local_residual.environment import build_environment_manifest


pytestmark = pytest.mark.cpu_only


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment(tmp_path):
    sources = {}
    for name in ("topology", "base_system", "box"):
        path = tmp_path / name
        path.write_bytes(name.encode())
        sources[name] = {"path": name, "sha256": _sha(path)}
    config = {
        "schema_version": "exp012-environment-config-v1",
        "payload": {
            "sources": sources,
            "atom_count": 10,
            "ligand_indices": [5, 2],
            "environment_candidate_indices": [9, 1],
            "metadata_fields": ["atomic_number", "element"],
            "atoms": [
                {"index": 9, "stable_id": "W:H", "atomic_number": 1, "element": "H"},
                {"index": 2, "stable_id": "L:C", "atomic_number": 6, "element": "C"},
                {"index": 1, "stable_id": "W:O", "atomic_number": 8, "element": "O"},
                {"index": 5, "stable_id": "L:N", "atomic_number": 7, "element": "N"},
            ],
        },
    }
    return build_environment_manifest(config, workspace_root=tmp_path)


def _config(environment, *, mode="manifest_canonical"):
    ordering = {"mode": mode}
    if mode == "explicit":
        ordering.update(
            {
                "local_topology_indices": [5, 1, 9, 2],
                "mace_topology_indices": [9, 5, 2, 1],
            }
        )
    return {
        "schema_version": "exp012-atom-mapping-config-v1",
        "source_environment_manifest_sha256": environment["canonical_sha256"],
        "supported_atomic_numbers": [8, 1, 7, 6],
        "ordering": ordering,
    }


def test_manifest_canonical_mapping_is_stable_and_has_three_index_bijection(tmp_path):
    environment = _environment(tmp_path)
    first = build_atom_mapping(environment, _config(environment))
    permuted = _config(environment)
    permuted["supported_atomic_numbers"].reverse()
    second = build_atom_mapping(environment, permuted)

    assert first == second
    assert first["canonical_sha256"] == atom_mapping_sha256(first["payload"])
    payload = first["payload"]
    assert payload["topology_indices_by_local_graph_index"] == [1, 2, 5, 9]
    assert payload["topology_indices_by_mace_node_index"] == [1, 2, 5, 9]
    assert [node["local_graph_index"] for node in payload["nodes_by_topology_index"]] == [0, 1, 2, 3]
    assert [node["ligand_mask"] for node in payload["nodes_by_topology_index"]] == [False, True, True, False]


def test_explicit_permutations_are_preserved_and_inverted(tmp_path):
    environment = _environment(tmp_path)
    document = build_atom_mapping(environment, _config(environment, mode="explicit"))
    payload = document["payload"]
    assert payload["topology_indices_by_local_graph_index"] == [5, 1, 9, 2]
    assert payload["topology_indices_by_mace_node_index"] == [9, 5, 2, 1]
    by_topology = {node["topology_index"]: node for node in payload["nodes_by_topology_index"]}
    assert (by_topology[5]["local_graph_index"], by_topology[5]["mace_node_index"]) == (0, 1)
    assert (by_topology[1]["local_graph_index"], by_topology[1]["mace_node_index"]) == (1, 3)
    assert validate_atom_mapping(document, environment_manifest=environment) == document


def test_json_roundtrip_with_source_validation(tmp_path):
    environment = _environment(tmp_path)
    document = build_atom_mapping(environment, _config(environment, mode="explicit"))
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    assert load_atom_mapping(str(path), environment_manifest=environment) == document


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("local_topology_indices", [5, 1, 9, 5], "duplicate"),
        ("local_topology_indices", [5, 1, 9], "missing"),
        ("mace_topology_indices", [9, 5, 2, 8], "extra"),
    ],
)
def test_explicit_order_rejects_duplicates_missing_and_extra_nodes(
    tmp_path, field, value, message
):
    environment = _environment(tmp_path)
    config = _config(environment, mode="explicit")
    config["ordering"][field] = value
    with pytest.raises(AtomMappingError, match=message):
        build_atom_mapping(environment, config)


def test_unsupported_atomic_number_fails_closed(tmp_path):
    environment = _environment(tmp_path)
    config = _config(environment)
    config["supported_atomic_numbers"].remove(8)
    with pytest.raises(AtomMappingError, match="unsupported atomic number 8"):
        build_atom_mapping(environment, config)


def test_source_sha_mismatch_fails_closed(tmp_path):
    environment = _environment(tmp_path)
    config = _config(environment)
    config["source_environment_manifest_sha256"] = "0" * 64
    with pytest.raises(AtomMappingIntegrityError, match="source environment manifest SHA mismatch"):
        build_atom_mapping(environment, config)

    document = build_atom_mapping(environment, _config(environment))
    changed_environment = copy.deepcopy(environment)
    changed_environment["payload"]["atoms"][0]["stable_id"] = "changed"
    changed_environment["canonical_sha256"] = hashlib.sha256(
        json.dumps(
            changed_environment["payload"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    with pytest.raises(AtomMappingIntegrityError, match="source environment manifest SHA mismatch"):
        validate_atom_mapping(document, environment_manifest=changed_environment)


def test_mapping_hash_tamper_and_rehashed_order_drift_fail_closed(tmp_path):
    environment = _environment(tmp_path)
    document = build_atom_mapping(environment, _config(environment))
    tampered = copy.deepcopy(document)
    tampered["payload"]["nodes_by_topology_index"][0]["stable_id"] = "tampered"
    with pytest.raises(AtomMappingIntegrityError, match="canonical SHA mismatch"):
        validate_atom_mapping(tampered)

    drifted = copy.deepcopy(document)
    drifted["payload"]["topology_indices_by_mace_node_index"] = [2, 1, 5, 9]
    drifted["canonical_sha256"] = atom_mapping_sha256(drifted["payload"])
    with pytest.raises(AtomMappingError, match="manifest_canonical ordering has drifted"):
        validate_atom_mapping(drifted)


def test_rehashed_node_identity_drift_is_rejected_against_source(tmp_path):
    environment = _environment(tmp_path)
    document = build_atom_mapping(environment, _config(environment))
    drifted = copy.deepcopy(document)
    drifted["payload"]["nodes_by_topology_index"][0]["stable_id"] = "W:OTHER"
    drifted["canonical_sha256"] = atom_mapping_sha256(drifted["payload"])
    with pytest.raises(AtomMappingError, match="identity differs"):
        validate_atom_mapping(drifted, environment_manifest=environment)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ligand_mask", [], "ligand_mask must be boolean"),
        ("atomic_number", True, "atomic_number must be an integer"),
    ],
)
def test_malformed_rehashed_node_fields_fail_closed(tmp_path, field, value, message):
    environment = _environment(tmp_path)
    document = build_atom_mapping(environment, _config(environment))
    document["payload"]["nodes_by_topology_index"][0][field] = value
    document["canonical_sha256"] = atom_mapping_sha256(document["payload"])
    with pytest.raises(AtomMappingError, match=message):
        validate_atom_mapping(document)


def test_environment_without_atomic_numbers_is_rejected(tmp_path):
    environment = _environment(tmp_path)
    payload = environment["payload"]
    payload["metadata_fields"].remove("atomic_number")
    for atom in payload["atoms"]:
        del atom["atomic_number"]
    environment["canonical_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(AtomMappingError, match="declare atomic_number"):
        build_atom_mapping(environment, _config(environment))
