"""Fail-closed machine audit for EXP-012 complete-MM ledgers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .mm_ledger import LEDGER_SCHEMA_VERSION


EXPECTED_RUN_IDS = (
    "hard_window0_run1",
    "hard_window0_run2",
    "hard_window0_run3",
)
EXPECTED_FRAMES = 500
EXPECTED_STATES = 5

ARRAY_SHAPES = {
    "frame_index": (EXPECTED_FRAMES,),
    "base_energy_kj_mol": (EXPECTED_FRAMES,),
    "softcore_cv_kj_mol": (EXPECTED_FRAMES, EXPECTED_STATES),
    "lrc_kj_mol": (EXPECTED_FRAMES, EXPECTED_STATES),
    "ibs_bias_kj_mol": (EXPECTED_FRAMES,),
    "wca_bias_kj_mol": (EXPECTED_FRAMES,),
    "total_context_kj_mol": (EXPECTED_FRAMES,),
    "potential_closure_error_kj_mol": (EXPECTED_FRAMES,),
    "ibs_bias_closure_error_kj_mol": (EXPECTED_FRAMES,),
    "reported_potential_delta_kj_mol": (EXPECTED_FRAMES,),
    "target_interaction_kj_mol": (EXPECTED_FRAMES, EXPECTED_STATES),
    "target_total_kj_mol": (EXPECTED_FRAMES, EXPECTED_STATES),
    "sampling_bias_kj_mol": (EXPECTED_FRAMES,),
    "sampling_total_kj_mol": (EXPECTED_FRAMES,),
    "target_reduced_potential": (EXPECTED_FRAMES, EXPECTED_STATES),
    "sampling_reduced_potential": (EXPECTED_FRAMES,),
    "log_importance_unnormalized": (EXPECTED_FRAMES, EXPECTED_STATES),
    "adjacent_gap_reduced": (EXPECTED_FRAMES, EXPECTED_STATES - 1),
}

COMPARISON_TOLERANCES = {
    "frame_index": {"kind": "exact", "limit": 0.0},
    "lrc_kj_mol": {"kind": "max_abs", "limit": 1.0e-12},
    "softcore_cv_kj_mol": {"kind": "max_abs", "limit": 1.0e-3},
    "target_interaction_kj_mol": {"kind": "max_abs", "limit": 1.0e-3},
    "ibs_bias_kj_mol": {"kind": "max_abs", "limit": 1.0e-3},
    "wca_bias_kj_mol": {"kind": "max_abs", "limit": 1.0e-3},
    "base_energy_kj_mol": {"kind": "max_relative", "limit": 2.0e-6},
    "target_total_kj_mol": {"kind": "max_relative", "limit": 2.0e-6},
    "sampling_total_kj_mol": {"kind": "max_relative", "limit": 2.0e-6},
    "adjacent_gap_reduced": {"kind": "max_abs", "limit": 1.0e-4},
    "log_importance_unnormalized": {"kind": "max_abs", "limit": 1.0e-4},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_report(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("report JSON is not an object")
    return value


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(ARRAY_SHAPES):
            missing = sorted(set(ARRAY_SHAPES) - set(archive.files))
            extra = sorted(set(archive.files) - set(ARRAY_SHAPES))
            raise ValueError(f"array schema mismatch: missing={missing}, extra={extra}")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    for name, expected_shape in ARRAY_SHAPES.items():
        value = arrays[name]
        if value.shape != expected_shape:
            raise ValueError(f"{name} shape {value.shape}, expected {expected_shape}")
        if not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{name} is not numeric")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
    return arrays


def _identity(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scratch_system_sha256": report.get("scratch_system_sha256_rebuilt"),
        "preregistration_payload_sha256": report.get("preregistration_payload_sha256"),
        "platform": report.get("platform"),
        "state_mapping": {
            "lambdas_coul": report.get("lambdas_coul"),
            "lambdas_vdw": report.get("lambdas_vdw"),
            "f_k_kj_mol": report.get("f_k_kj_mol"),
        },
    }


def _validate_run(directory: Path, run_id: str, platform: str) -> tuple[dict[str, Any], dict[str, np.ndarray], Mapping[str, Any]]:
    errors: list[str] = []
    report_path = directory / run_id / "ledger_report.json"
    arrays_path = directory / run_id / "ledger_arrays.npz"
    try:
        report = _read_report(report_path)
    except Exception as error:
        raise ValueError(f"cannot read {report_path}: {error}") from error
    required = {
        "report_type": "exp012_complete_mm_target_ledger",
        "report_version": 1,
        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
        "status": "COMPLETED",
        "run_id": run_id,
        "frame_count": EXPECTED_FRAMES,
        "state_count": EXPECTED_STATES,
        "platform": platform,
        "production_data_mutated": False,
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            errors.append(f"{field}={report.get(field)!r}, expected {expected!r}")
    expected_indices = list(range(EXPECTED_FRAMES))
    if report.get("frame_indices") != expected_indices:
        errors.append("report frame_indices are not exactly 0..499")
    expected_sha = report.get("scratch_system_sha256_expected")
    rebuilt_sha = report.get("scratch_system_sha256_rebuilt")
    if not isinstance(expected_sha, str) or rebuilt_sha != expected_sha:
        errors.append("expected and rebuilt scratch System SHA-256 differ")
    for field in ("preregistration_payload_sha256", "trajectory_sha256", "arrays_sha256"):
        value = report.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"{field} is not a SHA-256 digest")
    if not arrays_path.is_file():
        errors.append(f"missing arrays file: {arrays_path}")
        arrays = {}
    else:
        observed_sha = _sha256(arrays_path)
        if observed_sha != report.get("arrays_sha256"):
            errors.append("ledger_arrays.npz SHA-256 differs from report")
        try:
            arrays = _load_arrays(arrays_path)
        except Exception as error:
            errors.append(str(error))
            arrays = {}
    if arrays and not np.array_equal(arrays["frame_index"], np.arange(EXPECTED_FRAMES)):
        errors.append("array frame_index is not exactly 0..499")
    result = {
        "run_id": run_id,
        "passed": not errors,
        "report_path": str(report_path.resolve()),
        "arrays_path": str(arrays_path.resolve()),
        "arrays_sha256": report.get("arrays_sha256"),
        "errors": errors,
    }
    return result, arrays, report


def compare_reference_arrays(reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Compare a CPU reference against CUDA using frozen engineering tolerances."""

    results: dict[str, Any] = {}
    passed = True
    for name, spec in COMPARISON_TOLERANCES.items():
        left = np.asarray(reference[name])
        right = np.asarray(candidate[name])
        if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            metric = float("inf")
        elif spec["kind"] == "exact":
            metric = 0.0 if np.array_equal(left, right) else float("inf")
        elif spec["kind"] == "max_abs":
            metric = float(np.max(np.abs(left - right)))
        else:
            # Symmetric scale avoids unstable relative errors at zero while
            # retaining a strict dimensionless gate for large total energies.
            scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0)
            metric = float(np.max(np.abs(left - right) / scale))
        item_passed = bool(metric <= spec["limit"])
        passed = passed and item_passed
        results[name] = {
            "passed": item_passed,
            "metric": spec["kind"],
            "observed": metric,
            "limit": spec["limit"],
        }
    return {"passed": passed, "arrays": results}


def audit_exp012_ledgers(
    cuda_root: str | Path,
    *,
    cpu_reference_root: str | Path | None = None,
    run_ids: Sequence[str] = EXPECTED_RUN_IDS,
) -> dict[str, Any]:
    """Audit three CUDA ledgers and optionally compare run1 to CPU reference."""

    cuda_directory = Path(cuda_root)
    errors: list[str] = []
    if tuple(run_ids) != EXPECTED_RUN_IDS:
        errors.append(f"run IDs must be exactly {list(EXPECTED_RUN_IDS)}")
    run_results: list[dict[str, Any]] = []
    arrays_by_run: dict[str, dict[str, np.ndarray]] = {}
    reports: list[Mapping[str, Any]] = []
    for run_id in run_ids:
        try:
            result, arrays, report = _validate_run(cuda_directory, run_id, "CUDA")
            run_results.append(result)
            if arrays:
                arrays_by_run[run_id] = arrays
            reports.append(report)
        except Exception as error:
            run_results.append({"run_id": run_id, "passed": False, "errors": [str(error)]})

    identities = [_identity(report) for report in reports]
    common_identity = identities[0] if identities else None
    if identities and any(identity != common_identity for identity in identities[1:]):
        errors.append("CUDA ledgers do not share System SHA, preregistration digest, platform, and state mapping")
    if common_identity:
        mapping = common_identity["state_mapping"]
        if any(not isinstance(mapping[name], list) or len(mapping[name]) != EXPECTED_STATES for name in mapping):
            errors.append("common state mapping does not contain five Coulomb/vdW/f_k entries")

    comparison = None
    if cpu_reference_root is not None:
        try:
            cpu_result, cpu_arrays, cpu_report = _validate_run(
                Path(cpu_reference_root), EXPECTED_RUN_IDS[0], "CPU"
            )
            if not cpu_result["passed"]:
                comparison = {"passed": False, "errors": cpu_result["errors"]}
            elif EXPECTED_RUN_IDS[0] not in arrays_by_run:
                comparison = {"passed": False, "errors": ["CUDA run1 arrays unavailable"]}
            else:
                comparison = compare_reference_arrays(cpu_arrays, arrays_by_run[EXPECTED_RUN_IDS[0]])
                comparison["cpu_report_identity"] = _identity(cpu_report)
                comparison["cuda_report_identity"] = _identity(reports[0]) if reports else None
                if _identity(cpu_report)["scratch_system_sha256"] != common_identity["scratch_system_sha256"]:
                    comparison["passed"] = False
                    comparison.setdefault("errors", []).append("CPU/CUDA scratch System SHA differs")
                if _identity(cpu_report)["preregistration_payload_sha256"] != common_identity["preregistration_payload_sha256"]:
                    comparison["passed"] = False
                    comparison.setdefault("errors", []).append("CPU/CUDA preregistration digest differs")
        except Exception as error:
            comparison = {"passed": False, "errors": [str(error)]}

    passed = (
        not errors
        and len(run_results) == len(EXPECTED_RUN_IDS)
        and all(result["passed"] for result in run_results)
        and (comparison is None or comparison["passed"])
    )
    return {
        "audit_type": "exp012_mm_ledger_audit",
        "audit_version": 1,
        "status": "PASSED" if passed else "FAILED",
        "passed": passed,
        "cuda_root": str(cuda_directory.resolve()),
        "expected_run_ids": list(EXPECTED_RUN_IDS),
        "runs": run_results,
        "common_identity": common_identity,
        "cross_platform_comparison": comparison,
        "tolerances": COMPARISON_TOLERANCES,
        "errors": errors,
    }


__all__ = [
    "ARRAY_SHAPES",
    "COMPARISON_TOLERANCES",
    "EXPECTED_RUN_IDS",
    "audit_exp012_ledgers",
    "compare_reference_arrays",
]
