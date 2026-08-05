"""DEC-030(c): linear/ridge readout end-to-end, on a synthetic solvable case.

Rather than mock the low-level fit/evaluate helpers in isolation, this builds
a tiny synthetic join (matching the real schema from
scripts/join_exp012_teacher_latent_cache_with_ledger.py) where the gap is, by
construction, an exact linear function of the cached feature -- shared
identically across all three synthetic runs. If the whole pipeline (masking,
leave-one-run-out, inner ridge CV, standardization, LBFGS) is wired correctly,
every held-out fold must show a large, unambiguous improvement over the B=0
baseline. This is the real thing the script needs to get right, not a proxy.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

_MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "fit_exp012_local_residual_linear_readout.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "fit_exp012_local_residual_linear_readout", _MODULE_PATH
)
_fit_module = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _fit_module
_SPEC.loader.exec_module(_fit_module)

pytestmark = pytest.mark.cpu_only


def _write_synthetic_join(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    frames_per_run = 40
    feature_dim = 8
    run_ids = ["synthetic_run1", "synthetic_run2", "synthetic_run3"]
    delta_a = np.array([0.3, 0.5], dtype=np.float64)  # 2 edges -> 3 states
    true_weight = rng.normal(size=feature_dim)

    pooled_latent_parts = []
    gap_parts = []
    log_weight_parts = []
    partition_parts = []
    for label, _run_id in enumerate(run_ids):
        features = rng.normal(size=(frames_per_run, feature_dim))
        # The true correction exactly cancels a synthetic "base gap" via the
        # same linear map and the same delta_A in every run -- so a readout
        # that recovers true_weight (up to the ridge/standardization affine
        # transform) must drive gap variance to ~0 on every held-out run too.
        basis = features @ true_weight
        base_gap = rng.normal(scale=0.05, size=(frames_per_run, 2))  # small residual noise only
        gaps = base_gap - delta_a[None, :] * basis[:, None]
        log_weights = np.zeros((frames_per_run, 3), dtype=np.float64)  # uniform importance
        pooled_latent_parts.append(features)
        gap_parts.append(gaps)
        log_weight_parts.append(log_weights)
        partition_parts.append(np.full(frames_per_run, label, dtype=np.int64))

    joined_path = tmp_path / "joined.npz"
    np.savez(
        joined_path,
        pooled_latent=np.concatenate(pooled_latent_parts, axis=0).astype(np.float32),
        adjacent_gap_reduced=np.concatenate(gap_parts, axis=0),
        log_importance_unnormalized=np.concatenate(log_weight_parts, axis=0),
        partition_index=np.concatenate(partition_parts, axis=0),
        delta_A=delta_a,
        A_k_window=np.array([0.0, 0.3, 0.8], dtype=np.float64),
    )
    report = {
        "schema_version": "exp012-teacher-latent-ledger-join-v1",
        "status": "COMPLETED_JOIN_ONLY_NOT_FIT",
        "run_id_by_partition_index": run_ids,
        "report_sha256": "0" * 64,
    }
    joined_path.with_name("joined_report.json").write_text(json.dumps(report), encoding="utf-8")
    return joined_path


def test_every_held_out_fold_improves_on_an_exactly_learnable_signal(tmp_path):
    joined_path = _write_synthetic_join(tmp_path)
    output_path = tmp_path / "readout_report.json"

    exit_code = _fit_module.main(
        [
            "--joined", str(joined_path),
            "--ridge-grid", "1e-6", "1e-3", "1.0",
            "--output", str(output_path),
        ]
    )
    assert exit_code == 0

    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["status"] == "COMPLETED_HELD_OUT_EVALUATION"
    assert len(report["folds"]) == 3
    assert report["all_folds_improved_over_baseline"] is True
    for fold in report["folds"]:
        assert fold["held_out_fitted_gap_variance_loss"] < fold["held_out_baseline_gap_variance_loss"]
        # With near-zero label noise and an exactly linear, shared-across-runs
        # signal, the fitted loss should collapse close to zero, not just
        # nudge downward.
        assert fold["held_out_relative_improvement"] > 0.9
    assert report["policy"]["a_k_learned"] is False
    assert report["policy"]["mace_encoder_trained"] is False
    assert report["policy"]["local_residual_student_trained"] is False


def test_mask_for_labels_selects_exactly_the_requested_partitions():
    partition = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64)
    mask = _fit_module._mask_for_labels(partition, [0, 2])
    assert mask.tolist() == [True, False, True, True, False, True]
