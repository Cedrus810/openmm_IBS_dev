import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from exp012_xed.ledger_audit import (
    ARRAY_SHAPES,
    EXPECTED_RUN_IDS,
    audit_exp012_ledgers,
    compare_reference_arrays,
)


pytestmark = pytest.mark.cpu_only


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_root(tmp_path, platform="CUDA"):
    root = tmp_path / platform.lower()
    mapping = {
        "lambdas_coul": [0.0, 0.25, 0.5, 0.75, 1.0],
        "lambdas_vdw": [0.0, 0.25, 0.5, 0.75, 1.0],
        "f_k_kj_mol": [0.0, 1.0, 2.0, 3.0, 4.0],
    }
    for run_id in EXPECTED_RUN_IDS:
        directory = root / run_id
        directory.mkdir(parents=True)
        arrays = {name: np.zeros(shape) for name, shape in ARRAY_SHAPES.items()}
        arrays["frame_index"] = np.arange(500, dtype=np.int64)
        np.savez_compressed(directory / "ledger_arrays.npz", **arrays)
        report = {
            "report_type": "exp012_complete_mm_target_ledger",
            "report_version": 1,
            "ledger_schema_version": "exp012-mm-ledger-v1",
            "status": "COMPLETED",
            "run_id": run_id,
            "frame_count": 500,
            "frame_indices": list(range(500)),
            "state_count": 5,
            "platform": platform,
            "production_data_mutated": False,
            "scratch_system_sha256_expected": "a" * 64,
            "scratch_system_sha256_rebuilt": "a" * 64,
            "preregistration_payload_sha256": "b" * 64,
            "trajectory_sha256": "c" * 64,
            "arrays_sha256": _sha(directory / "ledger_arrays.npz"),
            **mapping,
        }
        (directory / "ledger_report.json").write_text(json.dumps(report))
    return root


def test_valid_three_run_audit_passes(tmp_path):
    result = audit_exp012_ledgers(_make_root(tmp_path))
    assert result["status"] == "PASSED"
    assert all(run["passed"] for run in result["runs"])


def test_tampered_array_hash_fails(tmp_path):
    root = _make_root(tmp_path)
    path = root / EXPECTED_RUN_IDS[1] / "ledger_arrays.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    result = audit_exp012_ledgers(root)
    assert not result["passed"]
    assert "SHA-256 differs" in " ".join(result["runs"][1]["errors"])


def test_report_identity_tamper_fails(tmp_path):
    root = _make_root(tmp_path)
    path = root / EXPECTED_RUN_IDS[2] / "ledger_report.json"
    report = json.loads(path.read_text())
    report["preregistration_payload_sha256"] = "d" * 64
    path.write_text(json.dumps(report))
    result = audit_exp012_ledgers(root)
    assert not result["passed"]
    assert "do not share" in " ".join(result["errors"])


def test_nonfinite_array_fails_even_with_updated_hash(tmp_path):
    root = _make_root(tmp_path)
    directory = root / EXPECTED_RUN_IDS[0]
    with np.load(directory / "ledger_arrays.npz") as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["softcore_cv_kj_mol"][0, 0] = np.nan
    np.savez_compressed(directory / "ledger_arrays.npz", **arrays)
    report_path = directory / "ledger_report.json"
    report = json.loads(report_path.read_text())
    report["arrays_sha256"] = _sha(directory / "ledger_arrays.npz")
    report_path.write_text(json.dumps(report))
    result = audit_exp012_ledgers(root)
    assert not result["passed"]
    assert "non-finite" in " ".join(result["runs"][0]["errors"])


def test_cpu_cuda_comparison_tolerance_pass_and_fail():
    reference = {name: np.ones(shape) for name, shape in ARRAY_SHAPES.items()}
    reference["frame_index"] = np.arange(500)
    candidate = {name: value.copy() for name, value in reference.items()}
    candidate["softcore_cv_kj_mol"][0, 0] += 9.0e-4
    candidate["base_energy_kj_mol"][0] += 1.9e-6
    assert compare_reference_arrays(reference, candidate)["passed"]
    candidate["adjacent_gap_reduced"][0, 0] += 1.1e-4
    comparison = compare_reference_arrays(reference, candidate)
    assert not comparison["passed"]
    assert not comparison["arrays"]["adjacent_gap_reduced"]["passed"]
