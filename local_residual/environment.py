"""Canonical, configuration-only atom environment manifests for EXP-012.

This module deliberately does not inspect molecular files to infer atom selections,
types, cutoffs, or other scientific choices.  It only validates caller-declared
identity data and binds it to hashed source artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping


CONFIG_SCHEMA = "exp012-environment-config-v1"
MANIFEST_SCHEMA = "exp012-environment-manifest-v1"
_SOURCE_NAMES = ("base_system", "box", "topology")
_METADATA_FIELDS = ("atomic_number", "element", "atom_type", "charge_e")
_ELEMENTS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()
_ATOMIC_NUMBER = {symbol: index for index, symbol in enumerate(_ELEMENTS, start=1)}


class EnvironmentManifestError(ValueError):
    """The explicit environment declaration is invalid or incomplete."""


class EnvironmentIntegrityError(EnvironmentManifestError):
    """A source artifact or manifest does not match its declared SHA-256."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON representation used for manifest hashing."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EnvironmentManifestError("payload is not canonical JSON data") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EnvironmentManifestError(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise EnvironmentManifestError(f"{field} is missing fields: {missing}")
    if unknown:
        raise EnvironmentManifestError(f"{field} has unknown fields: {unknown}")


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EnvironmentManifestError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise EnvironmentManifestError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {".", ""}:
        raise EnvironmentManifestError(f"{field} must stay inside the workspace")
    return path.as_posix()


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EnvironmentManifestError(f"{field} must be an integer >= {minimum}")
    return value


def _indices(value: Any, field: str, atom_count: int) -> list[int]:
    if not isinstance(value, list) or not value:
        raise EnvironmentManifestError(f"{field} must be a non-empty array")
    result = [_integer(item, f"{field}[{position}]") for position, item in enumerate(value)]
    if len(set(result)) != len(result):
        raise EnvironmentManifestError(f"{field} contains duplicate indices")
    if any(index >= atom_count for index in result):
        raise EnvironmentManifestError(f"{field} contains an out-of-bounds index")
    return sorted(result)


def _normalize_sources(
    value: Any, *, workspace_root: Path | None, verify_sources: bool
) -> dict[str, dict[str, str]]:
    sources = _mapping(value, "sources")
    _exact_keys(sources, set(_SOURCE_NAMES), "sources")
    normalized: dict[str, dict[str, str]] = {}
    for name in _SOURCE_NAMES:
        record = _mapping(sources[name], f"sources.{name}")
        _exact_keys(record, {"path", "sha256"}, f"sources.{name}")
        relative = _relative_path(record["path"], f"sources.{name}.path")
        expected = _sha(record["sha256"], f"sources.{name}.sha256")
        if verify_sources:
            if workspace_root is None:
                raise EnvironmentManifestError("workspace_root is required to verify sources")
            root = workspace_root.resolve()
            source_path = (root / relative).resolve()
            if root != source_path and root not in source_path.parents:
                raise EnvironmentManifestError(f"sources.{name}.path escapes workspace")
            if not source_path.is_file():
                raise EnvironmentIntegrityError(f"source artifact is missing: {relative}")
            if _sha256_file(source_path) != expected:
                raise EnvironmentIntegrityError(f"source SHA-256 mismatch: {relative}")
        normalized[name] = {"path": relative, "sha256": expected}
    return normalized


def _normalize_metadata_fields(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise EnvironmentManifestError("metadata_fields must be an array")
    if any(not isinstance(item, str) for item in value):
        raise EnvironmentManifestError("metadata_fields entries must be strings")
    if len(set(value)) != len(value):
        raise EnvironmentManifestError("metadata_fields entries must be unique")
    unknown = sorted(set(value) - set(_METADATA_FIELDS))
    if unknown:
        raise EnvironmentManifestError(f"metadata_fields has unknown fields: {unknown}")
    return [name for name in _METADATA_FIELDS if name in value]


def _normalize_type_vocabulary(value: Any, *, required: bool) -> list[str] | None:
    if not required:
        if value is not None:
            raise EnvironmentManifestError(
                "atom_type_vocabulary must be omitted unless atom_type is declared"
            )
        return None
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise EnvironmentManifestError(
            "atom_type_vocabulary must explicitly list non-empty type names"
        )
    if len(set(value)) != len(value):
        raise EnvironmentManifestError("atom_type_vocabulary entries must be unique")
    return sorted(value)


def _normalize_atom(
    value: Any,
    *,
    position: int,
    fields: list[str],
    type_vocabulary: list[str] | None,
    atom_count: int,
) -> dict[str, Any]:
    field = f"atoms[{position}]"
    atom = _mapping(value, field)
    expected_fields = {"index", "stable_id", *fields}
    _exact_keys(atom, expected_fields, field)
    index = _integer(atom["index"], f"{field}.index")
    if index >= atom_count:
        raise EnvironmentManifestError(f"{field}.index is out of bounds")
    stable_id = atom["stable_id"]
    if not isinstance(stable_id, str) or not stable_id:
        raise EnvironmentManifestError(f"{field}.stable_id must be a non-empty string")
    result: dict[str, Any] = {"index": index, "stable_id": stable_id}

    if "atomic_number" in fields:
        result["atomic_number"] = _integer(
            atom["atomic_number"], f"{field}.atomic_number", minimum=1
        )
        if result["atomic_number"] > len(_ELEMENTS):
            raise EnvironmentManifestError(f"{field}.atomic_number is unknown")
    if "element" in fields:
        element = atom["element"]
        if not isinstance(element, str) or element not in _ATOMIC_NUMBER:
            raise EnvironmentManifestError(f"{field}.element is unknown")
        result["element"] = element
    if "atomic_number" in fields and "element" in fields:
        if _ATOMIC_NUMBER[result["element"]] != result["atomic_number"]:
            raise EnvironmentManifestError(f"{field} element and atomic_number disagree")
    if "atom_type" in fields:
        atom_type = atom["atom_type"]
        if not isinstance(atom_type, str) or atom_type not in (type_vocabulary or []):
            raise EnvironmentManifestError(f"{field}.atom_type is undeclared")
        result["atom_type"] = atom_type
    if "charge_e" in fields:
        charge = atom["charge_e"]
        if isinstance(charge, bool) or not isinstance(charge, (int, float)):
            raise EnvironmentManifestError(f"{field}.charge_e must be numeric")
        charge = float(charge)
        if not math.isfinite(charge):
            raise EnvironmentManifestError(f"{field}.charge_e must be finite")
        result["charge_e"] = charge
    return result


def _normalize_payload(
    payload: Mapping[str, Any], *, workspace_root: Path | None, verify_sources: bool
) -> dict[str, Any]:
    optional = {"atom_type_vocabulary"}
    required = {
        "sources",
        "atom_count",
        "ligand_indices",
        "environment_candidate_indices",
        "metadata_fields",
        "atoms",
    }
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required - optional)
    if missing:
        raise EnvironmentManifestError(f"payload is missing fields: {missing}")
    if unknown:
        raise EnvironmentManifestError(f"payload has unknown fields: {unknown}")

    atom_count = _integer(payload["atom_count"], "atom_count", minimum=1)
    ligand = _indices(payload["ligand_indices"], "ligand_indices", atom_count)
    environment = _indices(
        payload["environment_candidate_indices"],
        "environment_candidate_indices",
        atom_count,
    )
    if set(ligand) & set(environment):
        raise EnvironmentManifestError("ligand and environment indices overlap")
    fields = _normalize_metadata_fields(payload["metadata_fields"])
    vocabulary = _normalize_type_vocabulary(
        payload.get("atom_type_vocabulary"), required="atom_type" in fields
    )
    atoms_value = payload["atoms"]
    if not isinstance(atoms_value, list):
        raise EnvironmentManifestError("atoms must be an array")
    atoms = [
        _normalize_atom(
            item,
            position=position,
            fields=fields,
            type_vocabulary=vocabulary,
            atom_count=atom_count,
        )
        for position, item in enumerate(atoms_value)
    ]
    indices = [atom["index"] for atom in atoms]
    stable_ids = [atom["stable_id"] for atom in atoms]
    if len(set(indices)) != len(indices):
        raise EnvironmentManifestError("atoms contains duplicate indices")
    if len(set(stable_ids)) != len(stable_ids):
        raise EnvironmentManifestError("atoms contains duplicate stable IDs")
    selected = set(ligand) | set(environment)
    if set(indices) != selected:
        raise EnvironmentManifestError(
            "atoms must describe exactly the ligand and environment candidate indices"
        )

    result: dict[str, Any] = {
        "sources": _normalize_sources(
            payload["sources"], workspace_root=workspace_root, verify_sources=verify_sources
        ),
        "atom_count": atom_count,
        "ligand_indices": ligand,
        "environment_candidate_indices": environment,
        "metadata_fields": fields,
        "atoms": sorted(atoms, key=lambda atom: atom["index"]),
    }
    if vocabulary is not None:
        result["atom_type_vocabulary"] = vocabulary
    return result


def environment_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a normalized manifest payload (the digest is not self-referential)."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def build_environment_manifest(
    config: Mapping[str, Any], *, workspace_root: str | Path
) -> dict[str, Any]:
    """Validate an explicit config, verify its sources, and build a manifest."""
    config = _mapping(config, "root")
    _exact_keys(config, {"schema_version", "payload"}, "root")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise EnvironmentManifestError(f"schema_version must be {CONFIG_SCHEMA}")
    payload = _normalize_payload(
        _mapping(config["payload"], "payload"),
        workspace_root=Path(workspace_root),
        verify_sources=True,
    )
    return {
        "schema_version": MANIFEST_SCHEMA,
        "payload": payload,
        "canonical_sha256": environment_manifest_sha256(payload),
    }


def validate_environment_manifest(
    document: Mapping[str, Any],
    *,
    workspace_root: str | Path | None = None,
    verify_sources: bool = False,
) -> dict[str, Any]:
    """Fail closed on malformed, non-canonical, tampered, or stale manifests."""
    document = _mapping(document, "root")
    _exact_keys(document, {"schema_version", "payload", "canonical_sha256"}, "root")
    if document["schema_version"] != MANIFEST_SCHEMA:
        raise EnvironmentManifestError(f"schema_version must be {MANIFEST_SCHEMA}")
    expected = _sha(document["canonical_sha256"], "canonical_sha256")
    normalized = _normalize_payload(
        _mapping(document["payload"], "payload"),
        workspace_root=Path(workspace_root) if workspace_root is not None else None,
        verify_sources=verify_sources,
    )
    if canonical_json_bytes(normalized) != canonical_json_bytes(document["payload"]):
        raise EnvironmentManifestError("manifest payload is not canonically ordered")
    if environment_manifest_sha256(normalized) != expected:
        raise EnvironmentIntegrityError("manifest canonical SHA-256 mismatch")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "payload": normalized,
        "canonical_sha256": expected,
    }


def load_environment_manifest(
    path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    verify_sources: bool = False,
) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EnvironmentManifestError(f"cannot read environment manifest: {path}") from error
    return validate_environment_manifest(
        document, workspace_root=workspace_root, verify_sources=verify_sources
    )


def write_environment_manifest(path: str | Path, document: Mapping[str, Any]) -> None:
    """Validate and atomically write an already-built canonical manifest."""
    normalized = validate_environment_manifest(document)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        normalized, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except OSError as error:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except OSError:
                pass
        raise EnvironmentManifestError(f"cannot write environment manifest: {output}") from error


__all__ = [
    "CONFIG_SCHEMA",
    "MANIFEST_SCHEMA",
    "EnvironmentIntegrityError",
    "EnvironmentManifestError",
    "build_environment_manifest",
    "canonical_json_bytes",
    "environment_manifest_sha256",
    "load_environment_manifest",
    "validate_environment_manifest",
    "write_environment_manifest",
]
