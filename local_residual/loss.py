"""Differentiable reduced-unit loss primitives for EXP-012.

All energy-like inputs in this module are already reduced (dimensionless).
In particular, ``adjacent_gap_reduced`` comes directly from the MM ledger and
``basis_reduced`` is the reduced local-residual basis.  There is deliberately
no ``beta`` argument: applying beta to either input again is a unit error.
"""

from __future__ import annotations

from typing import Any

from .schema import Exp012ProtocolError


def _require_floating_tensor(value: Any, name: str):
    import torch

    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise Exp012ProtocolError(f"{name} must be a floating-point Torch tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise Exp012ProtocolError(f"{name} must contain only finite values")
    return value


def _partition_labels(partition_index: Any, frame_count: int, device):
    import torch

    if partition_index is None:
        return torch.zeros(frame_count, dtype=torch.int64, device=device)
    if not isinstance(partition_index, torch.Tensor):
        raise Exp012ProtocolError("partition_index must be an integer Torch tensor")
    if partition_index.ndim != 1 or partition_index.shape[0] != frame_count:
        raise Exp012ProtocolError("partition_index must have one entry per frame")
    if partition_index.device != device:
        raise Exp012ProtocolError("partition_index and log weights must share a device")
    if partition_index.dtype == torch.bool or partition_index.is_floating_point():
        raise Exp012ProtocolError("partition_index must use an integer dtype")
    return partition_index


def normalize_log_importance(
    log_importance_unnormalized: Any,
    *,
    partition_index: Any | None = None,
):
    """Normalize log importance per target state inside each partition.

    The input shape is ``(frames, target_states)``.  The returned weights have
    the same shape and each target-state column sums to one separately within
    every partition.  No clipping is performed.
    """

    import torch

    log_weights = _require_floating_tensor(
        log_importance_unnormalized, "log_importance_unnormalized"
    )
    if log_weights.ndim != 2 or log_weights.shape[0] == 0 or log_weights.shape[1] < 2:
        raise Exp012ProtocolError(
            "log_importance_unnormalized must be a non-empty frame-by-state matrix"
        )
    labels = _partition_labels(partition_index, log_weights.shape[0], log_weights.device)
    normalized = torch.empty_like(log_weights)
    for label in torch.unique(labels):
        mask = labels == label
        selected = log_weights[mask]
        if selected.shape[0] == 0:
            raise Exp012ProtocolError("every declared partition must contain frames")
        log_normalizer = torch.logsumexp(selected, dim=0, keepdim=True)
        if not bool(torch.isfinite(log_normalizer).all().item()):
            raise Exp012ProtocolError("importance weights have zero or non-finite mass")
        weights = torch.exp(selected - log_normalizer)
        mass = weights.sum(dim=0)
        if not bool(torch.isfinite(mass).all().item()) or bool((mass <= 0).any().item()):
            raise Exp012ProtocolError("importance weights have zero or non-finite mass")
        normalized[mask] = weights
    return normalized


def weighted_population_variance(values: Any, normalized_weights: Any):
    """Return the differentiable weighted population variance of one vector."""

    import torch

    series = _require_floating_tensor(values, "values")
    weights = _require_floating_tensor(normalized_weights, "normalized_weights")
    if series.ndim != 1 or series.shape[0] == 0 or weights.shape != series.shape:
        raise Exp012ProtocolError(
            "values and normalized_weights must be equal non-empty vectors"
        )
    if series.device != weights.device:
        raise Exp012ProtocolError("values and normalized_weights must share a device")
    if bool((weights < 0).any().item()):
        raise Exp012ProtocolError("normalized_weights must be non-negative")
    mass = weights.sum()
    if not bool(torch.isfinite(mass).item()) or bool((mass <= 0).item()):
        raise Exp012ProtocolError("normalized_weights have zero or non-finite mass")
    tolerance = 32.0 * torch.finfo(weights.dtype).eps
    if not bool(torch.isclose(mass, mass.new_tensor(1.0), rtol=0.0, atol=tolerance).item()):
        raise Exp012ProtocolError("normalized_weights must sum to one")
    mean = torch.sum(weights * series)
    return torch.sum(weights * (series - mean).square())


def bidirectional_gap_variance_loss(
    adjacent_gap_reduced: Any,
    basis_reduced: Any,
    delta_A: Any,
    log_importance_unnormalized: Any,
    *,
    partition_index: Any | None = None,
    energy_regularization_coefficient: float,
    force_regularization_coefficient: float,
    force_gradient_reduced: Any | None = None,
):
    """Compute the EXP-012 gap loss plus explicit safety regularizers.

    For every partition and adjacent edge, the corrected gap is
    ``adjacent_gap_reduced + delta_A * basis_reduced``.  The edge loss is half
    the variance under each of its two target states.  The gap term is the
    explicit arithmetic mean over all partition-edge values.

    ``force_gradient_reduced`` is ``grad(basis_reduced, coordinates)`` with
    shape ``(frames, ...)``.  It is required only when its explicit coefficient
    is positive.  Both coefficients are caller choices; this module supplies
    no scientific defaults.
    """

    import math
    import torch

    gaps = _require_floating_tensor(adjacent_gap_reduced, "adjacent_gap_reduced")
    basis = _require_floating_tensor(basis_reduced, "basis_reduced")
    envelope_delta = _require_floating_tensor(delta_A, "delta_A")
    log_weights = _require_floating_tensor(
        log_importance_unnormalized, "log_importance_unnormalized"
    )
    if gaps.ndim != 2 or gaps.shape[0] == 0 or gaps.shape[1] == 0:
        raise Exp012ProtocolError("adjacent_gap_reduced must be a non-empty frame-by-edge matrix")
    frame_count, edge_count = gaps.shape
    if basis.shape != (frame_count,):
        raise Exp012ProtocolError("basis_reduced must have one scalar per frame")
    if envelope_delta.shape != (edge_count,):
        raise Exp012ProtocolError("delta_A must have one value per adjacent edge")
    if log_weights.shape != (frame_count, edge_count + 1):
        raise Exp012ProtocolError("log weights must contain one column per target state")
    tensors = (basis, envelope_delta, log_weights)
    if any(value.device != gaps.device for value in tensors):
        raise Exp012ProtocolError("all loss tensors must share a device")
    if any(value.dtype != gaps.dtype for value in tensors):
        raise Exp012ProtocolError("all loss tensors must share a floating dtype")

    coefficients = {
        "energy_regularization_coefficient": energy_regularization_coefficient,
        "force_regularization_coefficient": force_regularization_coefficient,
    }
    checked_coefficients = {}
    for name, value in coefficients.items():
        if isinstance(value, bool):
            raise Exp012ProtocolError(f"{name} must be finite and non-negative")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise Exp012ProtocolError(f"{name} must be finite and non-negative") from exc
        if not math.isfinite(numeric) or numeric < 0.0:
            raise Exp012ProtocolError(f"{name} must be finite and non-negative")
        checked_coefficients[name] = numeric

    labels = _partition_labels(partition_index, frame_count, gaps.device)
    weights = normalize_log_importance(log_weights, partition_index=labels)
    corrected_gaps = gaps + basis[:, None] * envelope_delta[None, :]
    partition_edge_losses = []
    for label in torch.unique(labels):
        mask = labels == label
        for edge in range(edge_count):
            left_variance = weighted_population_variance(
                corrected_gaps[mask, edge], weights[mask, edge]
            )
            right_variance = weighted_population_variance(
                corrected_gaps[mask, edge], weights[mask, edge + 1]
            )
            partition_edge_losses.append(0.5 * (left_variance + right_variance))
    if not partition_edge_losses:
        raise Exp012ProtocolError("gap loss has no partition-edge observations")
    gap_loss = torch.stack(partition_edge_losses).mean()

    energy_penalty = basis.square().mean()
    force_coefficient = checked_coefficients["force_regularization_coefficient"]
    if force_gradient_reduced is None:
        if force_coefficient > 0.0:
            raise Exp012ProtocolError(
                "force_gradient_reduced is required when force regularization is positive"
            )
        force_penalty = gap_loss.new_zeros(())
    else:
        force_gradient = _require_floating_tensor(
            force_gradient_reduced, "force_gradient_reduced"
        )
        if force_gradient.ndim < 2 or force_gradient.shape[0] != frame_count:
            raise Exp012ProtocolError(
                "force_gradient_reduced must have shape (frames, coordinate_dimensions...)"
            )
        if force_gradient.device != gaps.device or force_gradient.dtype != gaps.dtype:
            raise Exp012ProtocolError("force_gradient_reduced must match loss device and dtype")
        force_penalty = force_gradient.flatten(start_dim=1).square().sum(dim=1).mean()

    total = (
        gap_loss
        + checked_coefficients["energy_regularization_coefficient"] * energy_penalty
        + force_coefficient * force_penalty
    )
    return {
        "loss": total,
        "gap_variance_loss": gap_loss,
        "energy_penalty": energy_penalty,
        "force_penalty": force_penalty,
        "corrected_gap_reduced": corrected_gaps,
        "normalized_importance_weights": weights,
    }


__all__ = [
    "bidirectional_gap_variance_loss",
    "normalize_log_importance",
    "weighted_population_variance",
]
