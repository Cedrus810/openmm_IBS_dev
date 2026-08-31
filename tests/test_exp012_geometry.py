"""Tests for differentiable EXP-012 periodic geometry primitives."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from local_residual.geometry import (  # noqa: E402
    GeometryError,
    ligand_environment_cross_edges,
    minimum_image_displacement,
    quintic_c2_cutoff,
)


DTYPE = torch.float64


def test_minimum_image_is_invariant_to_whole_box_translation_and_wrap():
    box = torch.diag(torch.tensor([2.0, 3.0, 4.0], dtype=DTYPE))
    source = torch.tensor([[0.2, 0.3, 0.4]], dtype=DTYPE)
    target = torch.tensor([[1.9, 2.8, 3.7]], dtype=DTYPE)
    reference = minimum_image_displacement(source, target, box)
    assert torch.allclose(reference, torch.tensor([[-0.3, -0.5, -0.7]], dtype=DTYPE))
    translation = torch.tensor([[4.2, -5.1, 8.0]], dtype=DTYPE)
    assert torch.allclose(
        minimum_image_displacement(source + translation, target + translation, box), reference
    )
    assert torch.allclose(minimum_image_displacement(source, target + box[0] - box[2], box), reference)


def test_triclinic_row_vector_convention_matches_hand_calculation():
    box = torch.tensor([[2.0, 0.0, 0.0], [0.5, 1.5, 0.0], [0.2, 0.3, 1.2]], dtype=DTYPE)
    source = torch.zeros((1, 3), dtype=DTYPE)
    fractional_delta = torch.tensor([[0.6, -0.6, 0.2]], dtype=DTYPE)
    target = fractional_delta @ box
    expected = torch.tensor([[-0.4, 0.4, 0.2]], dtype=DTYPE) @ box
    assert torch.allclose(minimum_image_displacement(source, target, box), expected)


def test_cross_edges_are_lexicographic_without_top_k_and_empty_is_legal():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.2, 0.0, 0.0], [0.8, 0.0, 0.0]],
        dtype=DTYPE,
    )
    box = torch.eye(3, dtype=DTYPE) * 10.0
    result = ligand_environment_cross_edges(
        positions,
        box,
        torch.tensor([1, 0]),
        torch.tensor([3, 2]),
        outer_cutoff=0.65,
    )
    assert result["edge_index"].tolist() == [[0, 1, 1], [2, 2, 3]]
    assert torch.allclose(result["distance"], torch.tensor([0.2, 0.3, 0.3], dtype=DTYPE))
    empty = ligand_environment_cross_edges(
        positions,
        box,
        torch.tensor([0]),
        torch.tensor([3]),
        outer_cutoff=0.1,
    )
    assert empty["edge_index"].shape == (2, 0)
    assert empty["displacement"].shape == (0, 3)
    assert empty["distance"].shape == (0,)


def test_geometry_rotates_covariantly_and_preserves_coordinate_gradients():
    angle = 0.37
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
    )
    box = torch.tensor([[3.0, 0.0, 0.0], [0.4, 2.5, 0.0], [0.2, 0.1, 2.0]], dtype=DTYPE)
    source = torch.tensor([[0.2, 0.4, 0.1]], dtype=DTYPE, requires_grad=True)
    target = torch.tensor([[1.0, 0.8, 0.5]], dtype=DTYPE, requires_grad=True)
    displacement = minimum_image_displacement(source, target, box)
    rotated = minimum_image_displacement(source @ rotation.T, target @ rotation.T, box @ rotation.T)
    assert torch.allclose(rotated, displacement @ rotation.T, atol=1.0e-12)
    displacement.square().sum().backward()
    assert source.grad is not None and target.grad is not None
    assert torch.isfinite(source.grad).all() and torch.isfinite(target.grad).all()
    assert torch.allclose(source.grad, -target.grad)


def _first_and_second_derivative(radius: float):
    value = torch.tensor(radius, dtype=DTYPE, requires_grad=True)
    envelope = quintic_c2_cutoff(value, inner_cutoff=1.0, outer_cutoff=2.0)
    first = torch.autograd.grad(envelope, value, create_graph=True)[0]
    second = torch.autograd.grad(first, value)[0]
    return envelope.item(), first.item(), second.item()


def test_quintic_cutoff_values_and_c2_boundary_continuity():
    radii = torch.tensor([0.2, 1.0, 1.5, 2.0, 3.0], dtype=DTYPE)
    values = quintic_c2_cutoff(radii, inner_cutoff=1.0, outer_cutoff=2.0)
    assert torch.allclose(values, torch.tensor([1.0, 1.0, 0.5, 0.0, 0.0], dtype=DTYPE))
    for boundary, expected in ((1.0, 1.0), (2.0, 0.0)):
        exact = _first_and_second_derivative(boundary)
        assert exact == pytest.approx((expected, 0.0, 0.0), abs=1.0e-12)
        left = _first_and_second_derivative(boundary - 1.0e-6)
        right = _first_and_second_derivative(boundary + 1.0e-6)
        assert left[0] == pytest.approx(right[0], abs=2.0e-6)
        assert left[1] == pytest.approx(right[1], abs=1.0e-8)
        assert left[2] == pytest.approx(right[2], abs=7.0e-5)


@pytest.mark.parametrize(
    "call",
    [
        lambda: minimum_image_displacement(
            torch.zeros(3, dtype=DTYPE), torch.ones(3, dtype=DTYPE), torch.zeros((3, 3), dtype=DTYPE)
        ),
        lambda: minimum_image_displacement(
            torch.zeros(3, dtype=DTYPE), torch.ones(2, dtype=DTYPE), torch.eye(3, dtype=DTYPE)
        ),
        lambda: ligand_environment_cross_edges(
            torch.zeros((2, 3), dtype=DTYPE),
            torch.eye(3, dtype=DTYPE),
            torch.tensor([0, 0]),
            torch.tensor([1]),
            outer_cutoff=1.0,
        ),
        lambda: ligand_environment_cross_edges(
            torch.zeros((2, 3), dtype=DTYPE),
            torch.eye(3, dtype=DTYPE),
            torch.tensor([0]),
            torch.tensor([2]),
            outer_cutoff=1.0,
        ),
        lambda: ligand_environment_cross_edges(
            torch.zeros((2, 3), dtype=DTYPE),
            torch.eye(3, dtype=DTYPE),
            torch.tensor([0]),
            torch.tensor([1]),
            outer_cutoff=0.0,
        ),
        lambda: quintic_c2_cutoff(
            torch.tensor([1.0], dtype=DTYPE), inner_cutoff=2.0, outer_cutoff=1.0
        ),
        lambda: quintic_c2_cutoff(
            torch.tensor([-0.1], dtype=DTYPE), inner_cutoff=1.0, outer_cutoff=2.0
        ),
    ],
)
def test_invalid_box_cutoff_and_indices_fail_closed(call):
    with pytest.raises(GeometryError):
        call()

