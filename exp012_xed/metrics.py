"""Pre-feature data-support metrics for EXP-012 whole-run holdouts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import Exp012IntegrityError, Exp012ProtocolError


DATA_SUPPORT_SCHEMA_VERSION = "exp012-data-support-v1"


def normalized_importance_weights(log_weights: Any):
    import numpy as np

    values = np.asarray(log_weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise Exp012ProtocolError("log weights must be a non-empty frame-by-state matrix")
    if not np.all(np.isfinite(values)):
        raise Exp012ProtocolError("log weights contain NaN or infinity")
    shifted = values - np.max(values, axis=0, keepdims=True)
    weights = np.exp(shifted)
    normalizers = np.sum(weights, axis=0, keepdims=True)
    if not np.all(np.isfinite(normalizers)) or np.any(normalizers <= 0.0):
        raise Exp012ProtocolError("importance weights cannot be normalized")
    return weights / normalizers


def importance_effective_sample_size(normalized_weights: Any):
    import numpy as np

    weights = np.asarray(normalized_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] == 0:
        raise Exp012ProtocolError("normalized weights must be a frame-by-state matrix")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise Exp012ProtocolError("normalized weights must be finite and non-negative")
    if not np.allclose(np.sum(weights, axis=0), 1.0, rtol=0.0, atol=1.0e-12):
        raise Exp012ProtocolError("weights must be normalized independently for each state")
    return 1.0 / np.sum(weights * weights, axis=0)


def weighted_population_mean_variance(values: Any, normalized_weights: Any) -> tuple[float, float]:
    import numpy as np

    series = np.asarray(values, dtype=np.float64)
    weights = np.asarray(normalized_weights, dtype=np.float64)
    if series.ndim != 1 or weights.shape != series.shape or series.size == 0:
        raise Exp012ProtocolError("weighted variance inputs must be equal non-empty vectors")
    if not np.all(np.isfinite(series)) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise Exp012ProtocolError("weighted variance inputs must be finite and non-negative")
    if not math.isclose(float(np.sum(weights)), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise Exp012ProtocolError("weighted variance weights must sum to one")
    mean = float(np.sum(weights * series))
    variance = float(np.sum(weights * (series - mean) ** 2))
    return mean, variance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assess_whole_run_data_support(
    ledger_root: str | Path,
    *,
    run_ids: Sequence[str] = (
        "hard_window0_run1",
        "hard_window0_run2",
        "hard_window0_run3",
    ),
    minimum_raw_importance_ess_per_target: float,
) -> Mapping[str, Any]:
    import numpy as np

    root = Path(ledger_root).resolve()
    threshold = float(minimum_raw_importance_ess_per_target)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise Exp012ProtocolError("minimum ESS must be finite and positive")
    reports = []
    failed = []
    common_identity = None
    for run_id in run_ids:
        report_path = root / run_id / "ledger_report.json"
        if not report_path.is_file():
            raise Exp012IntegrityError(f"missing ledger report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        arrays_path = Path(report["arrays_path"])
        if not arrays_path.is_file() or _sha256(arrays_path) != report["arrays_sha256"]:
            raise Exp012IntegrityError(f"ledger arrays identity mismatch for {run_id}")
        if report.get("run_id") != run_id or report.get("status") != "COMPLETED":
            raise Exp012IntegrityError(f"ledger report identity/status mismatch for {run_id}")
        identity = {
            "scratch_system_sha256": report["scratch_system_sha256_rebuilt"],
            "platform": report["platform"],
            "lambdas_coul": report["lambdas_coul"],
            "lambdas_vdw": report["lambdas_vdw"],
            "f_k_kj_mol": report["f_k_kj_mol"],
        }
        if common_identity is None:
            common_identity = identity
        elif identity != common_identity:
            raise Exp012IntegrityError("whole-run ledgers do not share one sampling Hamiltonian")
        with np.load(arrays_path) as arrays:
            log_weights = np.asarray(arrays["log_importance_unnormalized"], dtype=np.float64)
            gaps = np.asarray(arrays["adjacent_gap_reduced"], dtype=np.float64)
            frame_index = np.asarray(arrays["frame_index"])
        if gaps.shape != (len(frame_index), log_weights.shape[1] - 1):
            raise Exp012IntegrityError(f"gap/weight shape mismatch for {run_id}")
        weights = normalized_importance_weights(log_weights)
        ess = importance_effective_sample_size(weights)
        state_results = []
        for state_index, value in enumerate(ess):
            passed = bool(float(value) >= threshold)
            state_results.append(
                {
                    "state_index": state_index,
                    "raw_importance_ess": float(value),
                    "ess_ratio": float(value / len(frame_index)),
                    "threshold": threshold,
                    "passed": passed,
                }
            )
            if not passed:
                failed.append({"run_id": run_id, "state_index": state_index, "ess": float(value)})
        edge_results = []
        for edge in range(gaps.shape[1]):
            sides = []
            for state_index in (edge, edge + 1):
                mean, variance = weighted_population_mean_variance(
                    gaps[:, edge], weights[:, state_index]
                )
                sides.append(
                    {
                        "target_state_index": state_index,
                        "weighted_mean_reduced": mean,
                        "weighted_population_variance_reduced2": variance,
                    }
                )
            edge_results.append({"edge": [edge, edge + 1], "sides": sides})
        reports.append(
            {
                "run_id": run_id,
                "frame_count": int(len(frame_index)),
                "arrays_sha256": report["arrays_sha256"],
                "states": state_results,
                "adjacent_edges": edge_results,
                "passed": all(item["passed"] for item in state_results),
            }
        )
    passed = not failed
    return {
        "report_type": "exp012_whole_run_data_support",
        "report_version": 1,
        "schema_version": DATA_SUPPORT_SCHEMA_VERSION,
        "status": "PASSED" if passed else "FAILED_INSUFFICIENT_TARGET_ESS",
        "passed": passed,
        "ledger_root": str(root),
        "weighting": {
            "source": "actual_sampling_to_target_importance_v1",
            "normalization": "per_target_state_within_each_heldout_run",
            "clipping": "forbidden",
        },
        "gate": {
            "metric": "raw_importance_ess_before_time_decorrelation",
            "minimum_per_target_per_heldout_run": threshold,
            "interpretation": "necessary_not_sufficient; time correlation can only reduce support",
            "provenance": "caller-supplied draft gate; must be sealed before A/B/C/D feature results",
        },
        "common_identity": common_identity,
        "runs": reports,
        "failed_targets": failed,
        "decision": (
            "eligible_to_freeze_feature_diagnostic_protocol"
            if passed
            else "do_not_fit_or_compare_A_B_C_on_current_three_run_holdouts"
        ),
        "scientific_qualification": False,
    }


__all__ = [
    "DATA_SUPPORT_SCHEMA_VERSION",
    "assess_whole_run_data_support",
    "importance_effective_sample_size",
    "normalized_importance_weights",
    "weighted_population_mean_variance",
]
