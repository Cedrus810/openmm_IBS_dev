import json
from pathlib import Path

import numpy as np
import pytest

from exp012_xed.metrics import (
    assess_whole_run_data_support,
    importance_effective_sample_size,
    normalized_importance_weights,
    weighted_population_mean_variance,
)
from exp012_xed.schema import Exp012ProtocolError


pytestmark = pytest.mark.cpu_only


def test_importance_weight_normalization_and_ess():
    weights = normalized_importance_weights([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    np.testing.assert_allclose(weights.sum(axis=0), 1.0)
    np.testing.assert_allclose(importance_effective_sample_size(weights), [3.0, 3.0])


def test_weighted_population_variance():
    mean, variance = weighted_population_mean_variance([1.0, 3.0], [0.25, 0.75])
    assert mean == pytest.approx(2.5)
    assert variance == pytest.approx(0.75)


def test_nonfinite_weights_fail_closed():
    with pytest.raises(Exp012ProtocolError):
        normalized_importance_weights([[0.0, np.nan], [1.0, 2.0]])


def _write_run(root: Path, run_id: str, log_weights):
    run = root / run_id
    run.mkdir(parents=True)
    arrays = run / "ledger_arrays.npz"
    np.savez_compressed(
        arrays,
        frame_index=np.arange(len(log_weights)),
        log_importance_unnormalized=np.asarray(log_weights, dtype=float),
        adjacent_gap_reduced=np.ones((len(log_weights), 1)),
    )
    import hashlib

    digest = hashlib.sha256(arrays.read_bytes()).hexdigest()
    report = {
        "run_id": run_id,
        "status": "COMPLETED",
        "arrays_path": str(arrays.resolve()),
        "arrays_sha256": digest,
        "scratch_system_sha256_rebuilt": "a" * 64,
        "platform": "CUDA",
        "lambdas_coul": [0.0, 0.0],
        "lambdas_vdw": [1.0, 0.5],
        "f_k_kj_mol": [0.0, 1.0],
    }
    (run / "ledger_report.json").write_text(json.dumps(report), encoding="utf-8")


def test_whole_run_data_support_gate_passes_and_fails(tmp_path):
    run_ids = ("r1", "r2", "r3")
    for run_id in run_ids:
        _write_run(tmp_path, run_id, np.zeros((30, 2)))
    passed = assess_whole_run_data_support(tmp_path, run_ids=run_ids, minimum_raw_importance_ess_per_target=25)
    assert passed["passed"]

    # Concentrating one target on one frame makes raw ESS approach one.
    concentrated = np.zeros((30, 2))
    concentrated[0, 0] = 20.0
    run = tmp_path / "r2"
    for path in run.iterdir():
        path.unlink()
    run.rmdir()
    _write_run(tmp_path, "r2", concentrated)
    failed = assess_whole_run_data_support(tmp_path, run_ids=run_ids, minimum_raw_importance_ess_per_target=25)
    assert not failed["passed"]
    assert failed["decision"] == "do_not_fit_or_compare_A_B_C_on_current_three_run_holdouts"
