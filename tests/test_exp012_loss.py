"""Unit tests for the EXP-012 differentiable reduced-unit loss core."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from local_residual.loss import (  # noqa: E402
    bidirectional_gap_variance_loss,
    normalize_log_importance,
    weighted_population_variance,
)
from local_residual.schema import Exp012ProtocolError  # noqa: E402


def _loss(gaps, basis, delta_a, log_weights, **kwargs):
    return bidirectional_gap_variance_loss(
        gaps,
        basis,
        delta_a,
        log_weights,
        energy_regularization_coefficient=kwargs.pop("energy_coefficient", 0.0),
        force_regularization_coefficient=kwargs.pop("force_coefficient", 0.0),
        **kwargs,
    )


def test_constant_corrected_gap_has_zero_population_variance():
    gaps = torch.full((5, 2), 3.0, dtype=torch.float64)
    result = _loss(
        gaps,
        torch.zeros(5, dtype=torch.float64),
        torch.tensor([0.2, -0.4], dtype=torch.float64),
        torch.zeros((5, 3), dtype=torch.float64),
    )
    assert result["gap_variance_loss"].item() == pytest.approx(0.0)
    assert result["loss"].item() == pytest.approx(0.0)


def test_bidirectional_state_weight_exchange_is_symmetric():
    gaps = torch.tensor([[0.0], [1.0], [4.0]], dtype=torch.float64)
    log_weights = torch.log(
        torch.tensor([[0.7, 0.1], [0.2, 0.3], [0.1, 0.6]], dtype=torch.float64)
    )
    first = _loss(
        gaps,
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        log_weights,
    )
    second = _loss(
        gaps,
        torch.zeros(3, dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        log_weights.flip(dims=(1,)),
    )
    assert first["gap_variance_loss"].item() == pytest.approx(
        second["gap_variance_loss"].item()
    )


def test_edge_losses_are_arithmetic_mean_not_sum():
    gaps = torch.tensor([[0.0, 0.0], [2.0, 4.0]], dtype=torch.float64)
    result = _loss(
        gaps,
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.zeros((2, 3), dtype=torch.float64),
    )
    # Uniform population variances are 1 and 4, hence their edge mean is 2.5.
    assert result["gap_variance_loss"].item() == pytest.approx(2.5)


def test_logsumexp_normalization_is_stable_and_partition_local():
    log_weights = torch.tensor(
        [[10_000.0, -10_000.0], [9_999.0, -10_001.0], [3.0, 8.0], [2.0, 7.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    partitions = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    weights = normalize_log_importance(log_weights, partition_index=partitions)
    assert torch.isfinite(weights).all()
    for partition in (0, 1):
        assert torch.allclose(
            weights[partitions == partition].sum(dim=0),
            torch.ones(2, dtype=torch.float64),
        )
    weights[0, 0].backward()
    assert log_weights.grad is not None
    assert torch.isfinite(log_weights.grad).all()


def test_gap_and_regularizers_preserve_autograd():
    coordinates = torch.tensor(
        [[0.2, -0.1], [0.6, 0.4], [-0.3, 0.8]],
        dtype=torch.float64,
        requires_grad=True,
    )
    basis = coordinates.square().sum(dim=1)
    force_gradient = torch.autograd.grad(basis.sum(), coordinates, create_graph=True)[0]
    result = _loss(
        torch.tensor([[0.1], [1.2], [-0.4]], dtype=torch.float64),
        basis,
        torch.tensor([0.3], dtype=torch.float64),
        torch.zeros((3, 2), dtype=torch.float64),
        energy_coefficient=0.2,
        force_coefficient=0.1,
        force_gradient_reduced=force_gradient,
    )
    result["loss"].backward()
    assert coordinates.grad is not None
    assert torch.isfinite(coordinates.grad).all()
    assert torch.count_nonzero(coordinates.grad).item() > 0


def test_api_uses_reduced_units_and_cannot_accept_beta():
    parameters = inspect.signature(bidirectional_gap_variance_loss).parameters
    assert "beta" not in parameters
    with pytest.raises(TypeError, match="beta"):
        _loss(
            torch.zeros((2, 1), dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
            beta=0.4,
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: normalize_log_importance(torch.tensor([[0.0, float("nan")]])),
        lambda: weighted_population_variance(torch.tensor([1.0]), torch.tensor([0.0])),
        lambda: _loss(
            torch.zeros((2, 1)),
            torch.zeros(3),
            torch.zeros(1),
            torch.zeros((2, 2)),
        ),
        lambda: _loss(
            torch.zeros((2, 1)),
            torch.zeros(2),
            torch.zeros(1),
            torch.zeros((2, 2)),
            energy_coefficient=-1.0,
        ),
        lambda: _loss(
            torch.zeros((2, 1)),
            torch.zeros(2),
            torch.zeros(1),
            torch.zeros((2, 2)),
            force_coefficient=1.0,
        ),
    ],
)
def test_invalid_inputs_fail_closed(call):
    with pytest.raises(Exp012ProtocolError):
        call()
