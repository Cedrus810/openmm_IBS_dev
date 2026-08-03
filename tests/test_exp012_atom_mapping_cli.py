import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from local_residual.atom_mapping import load_atom_mapping
from local_residual.environment import build_environment_manifest, write_environment_manifest


pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment_config(tmp_path: Path) -> dict:
    fixture_paths = {
        "topology": ROOT / "PLAN_outer_lambda_neural_basis.md",
        "base_system": ROOT / "IMPLEMENTATION_PLAN_outer_lambda_neural_basis.md",
        "box": ROOT / "EXPERIMENT_LOG_outer_lambda_neural_basis.md",
    }
    sources = {
        name: {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)}
        for name, path in fixture_paths.items()
    }
    return {
        "schema_version": "exp012-environment-config-v1",
        "payload": {
            "sources": sources,
            "atom_count": 4,
            "ligand_indices": [0, 1],
            "environment_candidate_indices": [2, 3],
            "metadata_fields": ["atomic_number", "element"],
            "atoms": [
                {"index": 0, "stable_id": "LIG:C1", "atomic_number": 6, "element": "C"},
                {"index": 1, "stable_id": "LIG:C2", "atomic_number": 6, "element": "C"},
                {"index": 2, "stable_id": "ALA:CA", "atomic_number": 6, "element": "C"},
                {"index": 3, "stable_id": "ALA:CB", "atomic_number": 6, "element": "C"},
            ],
        },
    }


def _sealed_manifest_path(tmp_path: Path) -> Path:
    document = build_environment_manifest(_environment_config(tmp_path), workspace_root=ROOT)
    manifest_path = tmp_path / "environment_manifest.json"
    write_environment_manifest(manifest_path, document)
    return manifest_path


def test_cli_builds_canonical_mapping_from_sealed_manifest(tmp_path):
    manifest_path = _sealed_manifest_path(tmp_path)
    manifest_sha = json.loads(manifest_path.read_text(encoding="utf-8"))["canonical_sha256"]

    mapping_config = {
        "schema_version": "exp012-atom-mapping-config-v1",
        "source_environment_manifest_sha256": manifest_sha,
        "supported_atomic_numbers": [6],
        "ordering": {"mode": "manifest_canonical"},
    }
    config_path = tmp_path / "mapping_config.json"
    config_path.write_text(json.dumps(mapping_config), encoding="utf-8")
    output_path = tmp_path / "mapping.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_exp012_atom_mapping.py"),
            "--environment-manifest",
            str(manifest_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert output_path.is_file()
    assert result.stdout.strip() == json.loads(output_path.read_text(encoding="utf-8"))["canonical_sha256"]

    loaded = load_atom_mapping(
        str(output_path),
        environment_manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
    )
    assert loaded["payload"]["node_count"] == 4
    assert sorted(loaded["payload"]["topology_indices_by_local_graph_index"]) == [0, 1, 2, 3]


def test_cli_fails_closed_on_manifest_sha_mismatch(tmp_path):
    manifest_path = _sealed_manifest_path(tmp_path)

    mapping_config = {
        "schema_version": "exp012-atom-mapping-config-v1",
        "source_environment_manifest_sha256": "0" * 64,
        "supported_atomic_numbers": [6],
        "ordering": {"mode": "manifest_canonical"},
    }
    config_path = tmp_path / "mapping_config.json"
    config_path.write_text(json.dumps(mapping_config), encoding="utf-8")
    output_path = tmp_path / "mapping.json"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_exp012_atom_mapping.py"),
            "--environment-manifest",
            str(manifest_path),
            "--config",
            str(config_path),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not output_path.exists()


def test_cli_requires_explicit_paths():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_exp012_atom_mapping.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--environment-manifest" in result.stderr
    assert "--config" in result.stderr
    assert "--output" in result.stderr
