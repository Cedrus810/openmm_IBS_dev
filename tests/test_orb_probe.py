from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from local_residual.orb_probe import (  # noqa: E402
    OrbProbeError,
    assess_exp012_promotion_gate,
    evaluate_loo_probe,
)


def test_revised_promotion_gate_requires_two_folds_and_mean_gain():
    passing = [
        {"held_out_improved": True, "held_out_relative_improvement": 0.20},
        {"held_out_improved": True, "held_out_relative_improvement": 0.05},
        {"held_out_improved": False, "held_out_relative_improvement": -0.05},
    ]
    gate = assess_exp012_promotion_gate(passing)
    assert gate["hard_floor_2_of_3"] is True
    assert gate["complete_gate_3_of_3"] is False
    assert gate["passed"] is True

    failing = [
        {"held_out_improved": True, "held_out_relative_improvement": 0.20},
        {"held_out_improved": True, "held_out_relative_improvement": 0.05},
        {"held_out_improved": False, "held_out_relative_improvement": -0.11},
    ]
    assert assess_exp012_promotion_gate(failing)["passed"] is False


def test_probe_requires_256_dim_features():
    with pytest.raises(OrbProbeError, match="256"):
        evaluate_loo_probe(
            torch.zeros((6, 8), dtype=torch.float64),
            torch.zeros((6, 1), dtype=torch.float64),
            torch.zeros((6, 2), dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.tensor([0, 0, 1, 1, 2, 2]),
            ridge_grid=(1.0,),
        )


def test_probe_runs_three_fold_protocol_and_reports_each_fold():
    features = torch.zeros((9, 256), dtype=torch.float64)
    features[:, 0] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    gaps = torch.zeros((9, 1), dtype=torch.float64)
    gaps[:, 0] = torch.tensor([2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    log_weights = torch.zeros((9, 2), dtype=torch.float64)
    delta_a = torch.ones(1, dtype=torch.float64)
    partition = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int64)
    report = evaluate_loo_probe(
        features,
        gaps,
        log_weights,
        delta_a,
        partition,
        ridge_grid=(1.0,),
    )
    assert report["feature_shape"] == [9, 256]
    assert report["readout_type"] == "linear_no_intercept"
    assert len(report["folds"]) == 3
    assert set(report["folds"][0]) >= {
        "held_out_partition",
        "selected_ridge_coefficient",
        "held_out_relative_improvement",
    }
