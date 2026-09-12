"""Fail-closed atom-index mappings for EXP-012 local residual models.

The mapping binds OpenMM topology indices, local graph indices, and MACE node
indices.  It consumes an already validated environment manifest and never
selects atoms or loads a MACE model.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from local_residual.environment import (
    EnvironmentIntegrityError,
    EnvironmentManifestError,
    canonical_json_bytes,
    validate_environment_manifest,
)


CONFIG_SCHEMA = "exp012-atom-mapping-config-v1"
MAPPING_SCHEMA = "exp012-atom-mapping-v1"


class AtomMappingError(ValueError):
    """An atom mapping is incomplete, ambiguous, or internally inconsistent."""


class AtomMappingIntegrityError(AtomMappingError):
    """A mapping or its source environment identity has changed."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AtomMappingError(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise AtomMappingError(f"{field} is missing fields: {missing}")
    if unknown:
        raise AtomMappingError(f"{field} has unknown fields: {unknown}")


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AtomMappingError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _positive_ints(value: Any, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise AtomMappingError(f"{field} must be a non-empty array")
    result: list[int] = []
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise AtomMappingError(f"{field}[{position}] must be a positive integer")
        result.append(item)
    if len(set(result)) != len(result):
        raise AtomMappingError(f"{field} contains duplicates")
    return result


def _topology_order(value: Any, field: str, selected: set[int]) -> list[int]:
    if not isinstance(value, list):
        raise AtomMappingError(f"{field} must be an array")
    result: list[int] = []
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise AtomMappingError(f"{field}[{position}] must be a non-negative integer")
        result.append(item)
    if len(set(result)) != len(result):
        raise AtomMappingError(f"{field} contains duplicate topology indices")
    observed = set(result)
    missing = sorted(selected - observed)
    extra = sorted(observed - selected)
    if missing or extra:
        raise AtomMappingError(
            f"{field} must contain exactly selected topology indices; "
            f"missing={missing}, extra={extra}"
        )
    return result


def _stored_topology_order(value: Any, field: str, node_count: int) -> list[int]:
    if not isinstance(value, list) or len(value) != node_count:
        raise AtomMappingError(f"{field} length differs from node_count")
    result: list[int] = []
    for position, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise AtomMappingError(f"{field}[{position}] must be a non-negative integer")
        result.append(item)
    if len(set(result)) != node_count:
        raise AtomMappingError(f"{field} contains duplicate topology indices")
    return result


def atom_mapping_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _source_atoms(environment: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], set[int]]:
    payload = _mapping(environment["payload"], "environment.payload")
    fields = payload.get("metadata_fields")
    if not isinstance(fields, list) or "atomic_number" not in fields:
        raise AtomMappingError(
            "environment manifest must explicitly declare atomic_number metadata"
        )
    atoms_value = payload.get("atoms")
    if not isinstance(atoms_value, list) or not atoms_value:
        raise AtomMappingError("environment manifest has no selected atoms")
    atoms = [_mapping(atom, f"environment.payload.atoms[{i}]") for i, atom in enumerate(atoms_value)]
    ligand_value = payload.get("ligand_indices")
    if not isinstance(ligand_value, list):
        raise AtomMappingError("environment manifest ligand_indices is invalid")
    return atoms, set(ligand_value)


def _orders_from_config(
    config: Mapping[str, Any], canonical_order: list[int], selected: set[int]
) -> tuple[str, list[int], list[int]]:
    ordering = _mapping(config.get("ordering"), "config.ordering")
    mode = ordering.get("mode")
    if mode == "manifest_canonical":
        _exact_keys(ordering, {"mode"}, "config.ordering")
        return mode, list(canonical_order), list(canonical_order)
    if mode == "explicit":
        _exact_keys(
            ordering,
            {"mode", "local_topology_indices", "mace_topology_indices"},
            "config.ordering",
        )
        local = _topology_order(
            ordering["local_topology_indices"],
            "config.ordering.local_topology_indices",
            selected,
        )
        mace = _topology_order(
            ordering["mace_topology_indices"],
            "config.ordering.mace_topology_indices",
            selected,
        )
        return mode, local, mace
    raise AtomMappingError("config.ordering.mode must be manifest_canonical or explicit")


def build_atom_mapping(
    environment_manifest: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a canonical three-index bijection from explicit ordering policy."""
    try:
        environment = validate_environment_manifest(environment_manifest)
    except EnvironmentIntegrityError as error:
        raise AtomMappingIntegrityError("source environment manifest SHA mismatch") from error
    except EnvironmentManifestError as error:
        raise AtomMappingError("source environment manifest is invalid") from error

    config = _mapping(config, "config")
    _exact_keys(
        config,
        {
            "schema_version",
            "source_environment_manifest_sha256",
            "supported_atomic_numbers",
            "ordering",
        },
        "config",
    )
    if config["schema_version"] != CONFIG_SCHEMA:
        raise AtomMappingError(f"config.schema_version must be {CONFIG_SCHEMA}")
    source_sha = _sha(
        config["source_environment_manifest_sha256"],
        "config.source_environment_manifest_sha256",
    )
    if source_sha != environment["canonical_sha256"]:
        raise AtomMappingIntegrityError("source environment manifest SHA mismatch")

    supported = sorted(
        _positive_ints(config["supported_atomic_numbers"], "config.supported_atomic_numbers")
    )
    atoms, ligand_indices = _source_atoms(environment)
    atom_by_topology: dict[int, Mapping[str, Any]] = {}
    for atom in atoms:
        topology_index = atom.get("index")
        atomic_number = atom.get("atomic_number")
        if isinstance(topology_index, bool) or not isinstance(topology_index, int):
            raise AtomMappingError("environment atom topology index is invalid")
        if isinstance(atomic_number, bool) or not isinstance(atomic_number, int):
            raise AtomMappingError("environment atom atomic_number is invalid")
        if atomic_number not in supported:
            raise AtomMappingError(
                f"unsupported atomic number {atomic_number} at topology index {topology_index}"
            )
        atom_by_topology[topology_index] = atom

    canonical_order = sorted(atom_by_topology)
    selected = set(canonical_order)
    mode, local_order, mace_order = _orders_from_config(config, canonical_order, selected)
    local_index = {topology: index for index, topology in enumerate(local_order)}
    mace_index = {topology: index for index, topology in enumerate(mace_order)}
    nodes = []
    for topology_index in canonical_order:
        atom = atom_by_topology[topology_index]
        nodes.append(
            {
                "topology_index": topology_index,
                "local_graph_index": local_index[topology_index],
                "mace_node_index": mace_index[topology_index],
                "ligand_mask": topology_index in ligand_indices,
                "atomic_number": atom["atomic_number"],
                "stable_id": atom["stable_id"],
            }
        )
    payload = {
        "source_environment_manifest_sha256": source_sha,
        "ordering_mode": mode,
        "supported_atomic_numbers": supported,
        "node_count": len(nodes),
        "topology_indices_by_local_graph_index": local_order,
        "topology_indices_by_mace_node_index": mace_order,
        "nodes_by_topology_index": nodes,
    }
    return {
        "schema_version": MAPPING_SCHEMA,
        "payload": payload,
        "canonical_sha256": atom_mapping_sha256(payload),
    }


def validate_atom_mapping(
    document: Mapping[str, Any],
    *,
    environment_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the mapping hash, bijections, identities, and optional source."""
    document = _mapping(document, "root")
    _exact_keys(document, {"schema_version", "payload", "canonical_sha256"}, "root")
    if document["schema_version"] != MAPPING_SCHEMA:
        raise AtomMappingError(f"schema_version must be {MAPPING_SCHEMA}")
    expected_sha = _sha(document["canonical_sha256"], "canonical_sha256")
    payload = _mapping(document["payload"], "payload")
    expected_payload_fields = {
        "source_environment_manifest_sha256",
        "ordering_mode",
        "supported_atomic_numbers",
        "node_count",
        "topology_indices_by_local_graph_index",
        "topology_indices_by_mace_node_index",
        "nodes_by_topology_index",
    }
    _exact_keys(payload, expected_payload_fields, "payload")
    if atom_mapping_sha256(payload) != expected_sha:
        raise AtomMappingIntegrityError("atom mapping canonical SHA mismatch")

    source_sha = _sha(
        payload["source_environment_manifest_sha256"],
        "payload.source_environment_manifest_sha256",
    )
    supported = _positive_ints(
        payload["supported_atomic_numbers"], "payload.supported_atomic_numbers"
    )
    if supported != sorted(supported):
        raise AtomMappingError("supported_atomic_numbers is not canonically ordered")
    mode = payload["ordering_mode"]
    if mode not in {"manifest_canonical", "explicit"}:
        raise AtomMappingError("payload.ordering_mode is invalid")
    node_count = payload["node_count"]
    if isinstance(node_count, bool) or not isinstance(node_count, int) or node_count < 1:
        raise AtomMappingError("payload.node_count must be positive")
    local_order = _stored_topology_order(
        payload["topology_indices_by_local_graph_index"],
        "payload.topology_indices_by_local_graph_index",
        node_count,
    )
    mace_order = _stored_topology_order(
        payload["topology_indices_by_mace_node_index"],
        "payload.topology_indices_by_mace_node_index",
        node_count,
    )
    if set(local_order) != set(mace_order):
        raise AtomMappingError("mapping orders are not bijections over the same topology atoms")
    if mode == "manifest_canonical" and (
        local_order != sorted(local_order) or mace_order != local_order
    ):
        raise AtomMappingError("manifest_canonical ordering has drifted")

    nodes_value = payload["nodes_by_topology_index"]
    if not isinstance(nodes_value, list) or len(nodes_value) != node_count:
        raise AtomMappingError("nodes_by_topology_index length differs from node_count")
    expected_node_keys = {
        "topology_index",
        "local_graph_index",
        "mace_node_index",
        "ligand_mask",
        "atomic_number",
        "stable_id",
    }
    topology_seen: list[int] = []
    local_seen: list[int] = []
    mace_seen: list[int] = []
    stable_seen: list[str] = []
    for position, item in enumerate(nodes_value):
        node = _mapping(item, f"payload.nodes_by_topology_index[{position}]")
        _exact_keys(node, expected_node_keys, f"payload.nodes_by_topology_index[{position}]")
        topology = node["topology_index"]
        local = node["local_graph_index"]
        mace = node["mace_node_index"]
        atomic_number = node["atomic_number"]
        stable_id = node["stable_id"]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (topology, local, mace)):
            raise AtomMappingError("node indices must be integers")
        if not isinstance(node["ligand_mask"], bool):
            raise AtomMappingError("node ligand_mask must be boolean")
        if isinstance(atomic_number, bool) or not isinstance(atomic_number, int):
            raise AtomMappingError("node atomic_number must be an integer")
        if atomic_number not in supported:
            raise AtomMappingError(f"unsupported atomic number {atomic_number}")
        if not isinstance(stable_id, str) or not stable_id:
            raise AtomMappingError("node stable_id must be non-empty")
        if local < 0 or local >= node_count or mace < 0 or mace >= node_count:
            raise AtomMappingError("node local/MACE index is out of bounds")
        if local_order[local] != topology or mace_order[mace] != topology:
            raise AtomMappingError("node mapping order has drifted")
        topology_seen.append(topology)
        local_seen.append(local)
        mace_seen.append(mace)
        stable_seen.append(stable_id)
    if topology_seen != sorted(topology_seen):
        raise AtomMappingError("nodes_by_topology_index is not canonically ordered")
    if len(set(topology_seen)) != node_count or set(topology_seen) != set(local_order):
        raise AtomMappingError("nodes contain missing or extra topology atoms")
    if set(local_seen) != set(range(node_count)) or set(mace_seen) != set(range(node_count)):
        raise AtomMappingError("local/MACE indices are not bijections")
    if len(set(stable_seen)) != node_count:
        raise AtomMappingError("node stable IDs are not unique")

    if environment_manifest is not None:
        try:
            environment = validate_environment_manifest(environment_manifest)
        except EnvironmentManifestError as error:
            raise AtomMappingIntegrityError("source environment manifest is invalid") from error
        if environment["canonical_sha256"] != source_sha:
            raise AtomMappingIntegrityError("source environment manifest SHA mismatch")
        source_atoms, source_ligand = _source_atoms(environment)
        source_by_topology = {atom["index"]: atom for atom in source_atoms}
        if set(source_by_topology) != set(topology_seen):
            raise AtomMappingError("mapping has missing or extra source environment nodes")
        for node in nodes_value:
            source = source_by_topology[node["topology_index"]]
            if (
                node["stable_id"] != source["stable_id"]
                or node["atomic_number"] != source["atomic_number"]
                or node["ligand_mask"] != (node["topology_index"] in source_ligand)
            ):
                raise AtomMappingError("mapping node identity differs from source environment")

    return json.loads(canonical_json_bytes(document))


def load_atom_mapping(
    path: str, *, environment_manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise AtomMappingError(f"cannot read atom mapping: {path}") from error
    return validate_atom_mapping(document, environment_manifest=environment_manifest)


__all__ = [
    "CONFIG_SCHEMA",
    "MAPPING_SCHEMA",
    "AtomMappingError",
    "AtomMappingIntegrityError",
    "atom_mapping_sha256",
    "build_atom_mapping",
    "load_atom_mapping",
    "validate_atom_mapping",
]
