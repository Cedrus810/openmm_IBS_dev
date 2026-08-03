import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from local_residual.environment import (
    EnvironmentIntegrityError,
    EnvironmentManifestError,
    build_environment_manifest,
    environment_manifest_sha256,
    load_environment_manifest,
    validate_environment_manifest,
    write_environment_manifest,
)


pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(tmp_path: Path) -> dict:
    sources = {}
    for name, contents in (
        ("topology", b"topology"),
        ("base_system", b"<System/>"),
        ("box", b"1 0 0\n0 1 0\n0 0 1\n"),
    ):
        path = tmp_path / f"{name}.dat"
        path.write_bytes(contents)
        sources[name] = {"path": path.name, "sha256": _sha(path)}
    return {
        "schema_version": "exp012-environment-config-v1",
        "payload": {
            "sources": sources,
            "atom_count": 8,
            "ligand_indices": [3, 1],
            "environment_candidate_indices": [7, 4],
            "metadata_fields": [
                "charge_e",
                "atom_type",
                "element",
                "atomic_number",
            ],
            "atom_type_vocabulary": ["solvent-O", "ligand-C", "solvent-H"],
            "atoms": [
                {
                    "index": 7,
                    "stable_id": "W1:H1",
                    "atomic_number": 1,
                    "element": "H",
                    "atom_type": "solvent-H",
                    "charge_e": 0.417,
                },
                {
                    "index": 1,
                    "stable_id": "LIG:C1",
                    "atomic_number": 6,
                    "element": "C",
                    "atom_type": "ligand-C",
                    "charge_e": -0.12,
                },
                {
                    "index": 4,
                    "stable_id": "W1:O",
                    "atomic_number": 8,
                    "element": "O",
                    "atom_type": "solvent-O",
                    "charge_e": -0.834,
                },
                {
                    "index": 3,
                    "stable_id": "LIG:C2",
                    "atomic_number": 6,
                    "element": "C",
                    "atom_type": "ligand-C",
                    "charge_e": 0.12,
                },
            ],
        },
    }


def test_manifest_is_deterministic_under_permutation(tmp_path):
    first_config = _config(tmp_path)
    second_config = copy.deepcopy(first_config)
    second_config["payload"]["ligand_indices"].reverse()
    second_config["payload"]["environment_candidate_indices"].reverse()
    second_config["payload"]["metadata_fields"].reverse()
    second_config["payload"]["atom_type_vocabulary"].reverse()
    second_config["payload"]["atoms"].reverse()

    first = build_environment_manifest(first_config, workspace_root=tmp_path)
    second = build_environment_manifest(second_config, workspace_root=tmp_path)

    assert first == second
    assert first["canonical_sha256"] == environment_manifest_sha256(first["payload"])
    assert first["payload"]["ligand_indices"] == [1, 3]
    assert [atom["index"] for atom in first["payload"]["atoms"]] == [1, 3, 4, 7]


def test_atomic_write_and_verified_roundtrip(tmp_path):
    document = build_environment_manifest(_config(tmp_path), workspace_root=tmp_path)
    output = tmp_path / "nested" / "environment.json"
    write_environment_manifest(output, document)
    loaded = load_environment_manifest(
        output, workspace_root=tmp_path, verify_sources=True
    )
    assert loaded == document
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p["ligand_indices"].append(1), "duplicate"),
        (lambda p: p["environment_candidate_indices"].append(3), "overlap"),
        (lambda p: p["environment_candidate_indices"].append(8), "out-of-bounds"),
        (lambda p: p["atoms"].__setitem__(1, copy.deepcopy(p["atoms"][0])), "duplicate indices"),
        (lambda p: p["atoms"][1].__setitem__("stable_id", "W1:H1"), "duplicate stable"),
        (lambda p: p["atoms"].pop(), "exactly"),
    ],
)
def test_invalid_selections_and_identities_fail_closed(tmp_path, mutate, message):
    config = _config(tmp_path)
    mutate(config["payload"])
    with pytest.raises(EnvironmentManifestError, match=message):
        build_environment_manifest(config, workspace_root=tmp_path)


def test_source_sha_mismatch_and_path_escape_fail_closed(tmp_path):
    config = _config(tmp_path)
    config["payload"]["sources"]["topology"]["sha256"] = "0" * 64
    with pytest.raises(EnvironmentIntegrityError, match="SHA-256 mismatch"):
        build_environment_manifest(config, workspace_root=tmp_path)

    config = _config(tmp_path)
    config["payload"]["sources"]["box"]["path"] = "../box.dat"
    with pytest.raises(EnvironmentManifestError, match="inside the workspace"):
        build_environment_manifest(config, workspace_root=tmp_path)


def test_unknown_or_undeclared_metadata_fails_closed(tmp_path):
    config = _config(tmp_path)
    config["payload"]["metadata_fields"].append("mass")
    with pytest.raises(EnvironmentManifestError, match="unknown fields"):
        build_environment_manifest(config, workspace_root=tmp_path)

    config = _config(tmp_path)
    config["payload"]["atoms"][0]["atom_type"] = "unknown-H"
    with pytest.raises(EnvironmentManifestError, match="undeclared"):
        build_environment_manifest(config, workspace_root=tmp_path)

    config = _config(tmp_path)
    del config["payload"]["atoms"][0]["atom_type"]
    with pytest.raises(EnvironmentManifestError, match="missing fields"):
        build_environment_manifest(config, workspace_root=tmp_path)


@pytest.mark.parametrize("charge", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_charge_fails_closed(tmp_path, charge):
    config = _config(tmp_path)
    config["payload"]["atoms"][0]["charge_e"] = charge
    with pytest.raises(EnvironmentManifestError, match="finite"):
        build_environment_manifest(config, workspace_root=tmp_path)


def test_element_number_mismatch_and_unknown_element_fail_closed(tmp_path):
    config = _config(tmp_path)
    config["payload"]["atoms"][0]["atomic_number"] = 8
    with pytest.raises(EnvironmentManifestError, match="disagree"):
        build_environment_manifest(config, workspace_root=tmp_path)

    config = _config(tmp_path)
    config["payload"]["atoms"][0]["element"] = "Xx"
    with pytest.raises(EnvironmentManifestError, match="unknown"):
        build_environment_manifest(config, workspace_root=tmp_path)


def test_tampering_and_noncanonical_payload_fail_closed(tmp_path):
    document = build_environment_manifest(_config(tmp_path), workspace_root=tmp_path)
    tampered = copy.deepcopy(document)
    tampered["payload"]["atoms"][0]["charge_e"] += 0.01
    with pytest.raises(EnvironmentIntegrityError, match="canonical SHA-256 mismatch"):
        validate_environment_manifest(tampered)

    noncanonical = copy.deepcopy(document)
    noncanonical["payload"]["atoms"].reverse()
    with pytest.raises(EnvironmentManifestError, match="canonically ordered"):
        validate_environment_manifest(noncanonical)


def test_cli_requires_explicit_paths_and_writes_requested_output(tmp_path):
    config = _config(tmp_path)
    # CLI resolves declared source paths under the repository root.  These
    # small tracked files stand in for each opaque artifact role; the builder
    # intentionally checks identity, not molecular-file contents.
    fixture_paths = {
        "topology": ROOT / "PLAN_outer_lambda_neural_basis.md",
        "base_system": ROOT / "IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md",
        "box": ROOT / "EXPERIMENT_LOG_outer_lambda_neural_basis.md",
    }
    for name, source_path in fixture_paths.items():
        config["payload"]["sources"][name] = {
            "path": source_path.relative_to(ROOT).as_posix(),
            "sha256": _sha(source_path),
        }
    config_path = tmp_path / "config.json"
    output = tmp_path / "manifest.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_exp012_environment_manifest.py"),
            "--config",
            str(config_path),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert result.stdout.strip() == load_environment_manifest(output)["canonical_sha256"]

    missing = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_exp012_environment_manifest.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing.returncode != 0
    assert "--config" in missing.stderr and "--output" in missing.stderr
