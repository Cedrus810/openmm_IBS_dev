"""Differentiable periodic geometry primitives for EXP-012.

Box vectors use the row-vector convention::

    cartesian = fractional @ box

Consequently a Cartesian displacement is converted with ``d @ inv(box)``.
Minimum-image wrapping uses the centered fractional cell via ``round``.  It is
differentiable away from periodic-image boundaries; neighbor membership itself
is discrete at the outer cutoff and is not claimed to be differentiable there.
"""

from __future__ import annotations

import math
from typing import Any


class GeometryError(ValueError):
    """Raised when an EXP-012 geometry input violates its explicit contract."""


def _floating_tensor(value: Any, name: str):
    import torch

    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise GeometryError(f"{name} must be a floating-point Torch tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise GeometryError(f"{name} must contain only finite values")
    return value


def _validated_box(box: Any, reference):
    import torch

    cell = _floating_tensor(box, "box")
    if cell.shape != (3, 3):
        raise GeometryError("box must have shape (3, 3) with lattice vectors as rows")
    if cell.device != reference.device or cell.dtype != reference.dtype:
        raise GeometryError("box must share dtype and device with Cartesian coordinates")
    sign, log_absolute_determinant = torch.linalg.slogdet(cell)
    if bool((sign == 0).item()) or not bool(torch.isfinite(log_absolute_determinant).item()):
        raise GeometryError("box must be finite and non-singular")
    return cell


def minimum_image_displacement(source: Any, target: Any, box: Any):
    """Return the centered-cell displacement ``target - source`` under PBC.

    ``source`` and ``target`` may have any identical leading shape, ending in
    Cartesian xyz.  The rounding image choice is piecewise constant, preserving
    coordinate gradients except on periodic-image boundaries.
    """

    import torch

    source_coordinates = _floating_tensor(source, "source")
    target_coordinates = _floating_tensor(target, "target")
    if source_coordinates.shape != target_coordinates.shape or source_coordinates.shape[-1:] != (3,):
        raise GeometryError("source and target must have identical shapes ending in xyz")
    if source_coordinates.device != target_coordinates.device or source_coordinates.dtype != target_coordinates.dtype:
        raise GeometryError("source and target must share dtype and device")
    cell = _validated_box(box, source_coordinates)
    cartesian = target_coordinates - source_coordinates
    fractional = torch.linalg.solve(cell.T, cartesian.reshape(-1, 3).T).T
    fractional = fractional - torch.round(fractional)
    return (fractional @ cell).reshape(cartesian.shape)


def _validated_indices(indices: Any, name: str, atom_count: int, device):
    import torch

    if not isinstance(indices, torch.Tensor):
        raise GeometryError(f"{name} must be an integer Torch tensor")
    if indices.ndim != 1 or indices.device != device:
        raise GeometryError(f"{name} must be a one-dimensional tensor on the coordinate device")
    if indices.dtype == torch.bool or indices.is_floating_point() or indices.is_complex():
        raise GeometryError(f"{name} must use an integer dtype")
    canonical = torch.sort(indices.to(dtype=torch.int64)).values
    if canonical.numel() and (
        bool((canonical < 0).any().item()) or bool((canonical >= atom_count).any().item())
    ):
        raise GeometryError(f"{name} contains an out-of-range atom index")
    if canonical.numel() > 1 and bool((canonical[1:] == canonical[:-1]).any().item()):
        raise GeometryError(f"{name} must not contain duplicate atom indices")
    return canonical


def _positive_scalar(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise GeometryError(f"{name} must be an explicit finite positive scalar")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{name} must be an explicit finite positive scalar") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise GeometryError(f"{name} must be an explicit finite positive scalar")
    return result


def ligand_environment_cross_edges(
    positions: Any,
    box: Any,
    ligand_indices: Any,
    environment_indices: Any,
    *,
    outer_cutoff: float,
):
    """Build deterministic ligand-to-environment edges from fixed candidates.

    Candidate indices are canonicalized to increasing order, and retained
    edges are ligand-major lexicographic pairs.  Every candidate pair strictly
    inside ``outer_cutoff`` is retained; hard top-k selection is never used.
    An empty neighborhood is valid and returns correctly shaped empty tensors.
    """

    import torch

    coordinates = _floating_tensor(positions, "positions")
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise GeometryError("positions must have shape (atoms, 3)")
    cell = _validated_box(box, coordinates)
    ligand = _validated_indices(
        ligand_indices, "ligand_indices", coordinates.shape[0], coordinates.device
    )
    environment = _validated_indices(
        environment_indices, "environment_indices", coordinates.shape[0], coordinates.device
    )
    if ligand.numel() and environment.numel():
        if bool(torch.isin(ligand, environment).any().item()):
            raise GeometryError("ligand and environment candidate sets must be disjoint")
    cutoff = _positive_scalar(outer_cutoff, "outer_cutoff")

    if ligand.numel() == 0 or environment.numel() == 0:
        return {
            "edge_index": torch.empty((2, 0), dtype=torch.int64, device=coordinates.device),
            "displacement": coordinates.new_empty((0, 3)),
            "distance": coordinates.new_empty((0,)),
        }

    ligand_grid = ligand[:, None].expand(-1, environment.numel()).reshape(-1)
    environment_grid = environment[None, :].expand(ligand.numel(), -1).reshape(-1)
    displacement = minimum_image_displacement(
        coordinates[ligand_grid], coordinates[environment_grid], cell
    )
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    retained = distance < cutoff
    return {
        "edge_index": torch.stack((ligand_grid[retained], environment_grid[retained]), dim=0),
        "displacement": displacement[retained],
        "distance": distance[retained],
    }


def quintic_c2_cutoff(distance: Any, *, inner_cutoff: float, outer_cutoff: float):
    """Return a quintic C2 envelope: one inside inner and zero outside outer."""

    import torch

    radii = _floating_tensor(distance, "distance")
    if bool((radii < 0).any().item()):
        raise GeometryError("distance must be non-negative")
    inner = _positive_scalar(inner_cutoff, "inner_cutoff")
    outer = _positive_scalar(outer_cutoff, "outer_cutoff")
    if not inner < outer:
        raise GeometryError("cutoffs must satisfy 0 < inner_cutoff < outer_cutoff")
    scaled = (radii - inner) / (outer - inner)
    transition = 1.0 - 10.0 * scaled.pow(3) + 15.0 * scaled.pow(4) - 6.0 * scaled.pow(5)
    return torch.where(
        radii <= inner,
        torch.ones_like(radii),
        torch.where(radii >= outer, torch.zeros_like(radii), transition),
    )


__all__ = [
    "GeometryError",
    "ligand_environment_cross_edges",
    "minimum_image_displacement",
    "quintic_c2_cutoff",
]
