"""Torch-native, fail-closed graph construction for EXP-012 MACE smoke tests.

Coordinates and cells use Angstrom and the repository row-vector convention.
Edge membership and periodic image integers are piecewise fixed; ``shifts`` are
recomputed from the live cell so edge displacements remain in the autograd graph.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from local_residual.atom_mapping import AtomMappingError, validate_atom_mapping
from local_residual.environment import EnvironmentManifestError, validate_environment_manifest


class MaceGraphError(ValueError):
    """The declared graph or support domain is invalid or incomplete."""


@dataclass(frozen=True)
class MaceGraphConfig:
    edge_cutoff_angstrom: float
    interaction_layers: int
    geometric_upper_bound_angstrom: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.edge_cutoff_angstrom, bool)
            or not math.isfinite(float(self.edge_cutoff_angstrom))
            or float(self.edge_cutoff_angstrom) <= 0.0
        ):
            raise MaceGraphError("edge_cutoff_angstrom must be finite and positive")
        if (
            isinstance(self.interaction_layers, bool)
            or not isinstance(self.interaction_layers, int)
            or self.interaction_layers < 1
        ):
            raise MaceGraphError("interaction_layers must be a positive integer")
        required = float(self.edge_cutoff_angstrom) * self.interaction_layers
        if (
            isinstance(self.geometric_upper_bound_angstrom, bool)
            or not math.isfinite(float(self.geometric_upper_bound_angstrom))
            or float(self.geometric_upper_bound_angstrom) < required
        ):
            raise MaceGraphError(
                "candidate support geometric upper bound is insufficient: smaller than "
                "edge_cutoff_angstrom * interaction_layers"
            )


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MaceGraphError("Torch is required only when constructing a MACE graph") from exc
    return torch


def _floating_tensor(value: Any, name: str, *, shape: tuple[int, ...] | None = None):
    torch = _torch()
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise MaceGraphError(f"{name} must be a floating-point Torch tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise MaceGraphError(f"{name} must have shape {shape}")
    if not bool(torch.isfinite(value).all().item()):
        raise MaceGraphError(f"{name} contains non-finite values")
    return value


def _validate_identity(
    environment_manifest: Mapping[str, Any], atom_mapping: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        environment = validate_environment_manifest(environment_manifest)
        mapping = validate_atom_mapping(atom_mapping, environment_manifest=environment)
    except (EnvironmentManifestError, AtomMappingError) as exc:
        raise MaceGraphError(f"environment/mapping identity validation failed: {exc}") from exc
    return environment, mapping


def _face_heights(cell):
    """Perpendicular cell heights for row lattice vectors."""
    torch = _torch()
    volume = torch.abs(torch.linalg.det(cell))
    return torch.stack(
        (
            volume / torch.linalg.vector_norm(torch.linalg.cross(cell[1], cell[2])),
            volume / torch.linalg.vector_norm(torch.linalg.cross(cell[2], cell[0])),
            volume / torch.linalg.vector_norm(torch.linalg.cross(cell[0], cell[1])),
        )
    )


def _minimum_image(source, target, cell):
    delta = target - source
    fractional = _torch().linalg.solve(cell.T, delta.reshape(-1, 3).T).T
    unit_shift = -_torch().round(fractional)
    shifts = unit_shift @ cell
    return (delta.reshape(-1, 3) + shifts), unit_shift, shifts


def topology_n_hop_closure(
    positions_angstrom: Any,
    cell_angstrom: Any,
    seed_indices: list[int] | tuple[int, ...],
    *,
    edge_cutoff_angstrom: float,
    interaction_layers: int,
    maximum_pair_batch: int = 100_000,
) -> tuple[Any, Any]:
    """Return the exact cutoff-graph closure and minimum hop for every atom.

    This is a graph traversal, not a radial selection.  Starting from the
    seeds, each layer expands only from the newly reached frontier using
    minimum-image distances strictly smaller than ``edge_cutoff_angstrom``.
    Consequently an atom inside the geometric ``cutoff * layers`` sphere is
    not selected unless a real cutoff-edge path connects it to a seed.

    The returned boolean mask has shape ``[N]``.  The hop tensor has value 0
    for seeds, 1..``interaction_layers`` for reached atoms, and -1 otherwise.
    Pair evaluation is double-chunked to avoid materialising a frontier-by-N
    displacement tensor for large solvated systems.
    """

    torch = _torch()
    positions = _floating_tensor(positions_angstrom, "positions_angstrom")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise MaceGraphError("positions_angstrom must have shape (N, 3)")
    cell = _floating_tensor(cell_angstrom, "cell_angstrom", shape=(3, 3))
    if positions.device != cell.device or positions.dtype != cell.dtype:
        raise MaceGraphError("positions and cell must share dtype and device")
    if (
        isinstance(edge_cutoff_angstrom, bool)
        or not math.isfinite(float(edge_cutoff_angstrom))
        or float(edge_cutoff_angstrom) <= 0.0
    ):
        raise MaceGraphError("edge_cutoff_angstrom must be finite and positive")
    if (
        isinstance(interaction_layers, bool)
        or not isinstance(interaction_layers, int)
        or interaction_layers < 1
    ):
        raise MaceGraphError("interaction_layers must be a positive integer")
    if (
        isinstance(maximum_pair_batch, bool)
        or not isinstance(maximum_pair_batch, int)
        or maximum_pair_batch < 1
    ):
        raise MaceGraphError("maximum_pair_batch must be a positive integer")

    atom_count = int(positions.shape[0])
    seeds = sorted({int(index) for index in seed_indices})
    if not seeds or seeds[0] < 0 or seeds[-1] >= atom_count:
        raise MaceGraphError("seed_indices must be non-empty and within positions")

    # Edge membership is discrete and intentionally detached from autograd.
    membership_positions = positions.detach()
    membership_cell = cell.detach()
    fractional_positions = torch.linalg.solve(
        membership_cell.T, membership_positions.T
    ).T
    closure = torch.zeros(atom_count, dtype=torch.bool, device=positions.device)
    hop = torch.full((atom_count,), -1, dtype=torch.long, device=positions.device)
    seed_tensor = torch.tensor(seeds, dtype=torch.long, device=positions.device)
    closure[seed_tensor] = True
    hop[seed_tensor] = 0
    frontier = seed_tensor
    cutoff = float(edge_cutoff_angstrom)

    # Cap each source block as well as the Cartesian product.  This keeps the
    # temporary [source, target, 3] tensors bounded even for a dense water shell.
    maximum_source_batch = min(64, maximum_pair_batch)
    for layer in range(1, interaction_layers + 1):
        if frontier.numel() == 0:
            break
        reached = torch.zeros(atom_count, dtype=torch.bool, device=positions.device)
        for source_start in range(0, int(frontier.numel()), maximum_source_batch):
            source_indices = frontier[source_start : source_start + maximum_source_batch]
            source_count = int(source_indices.numel())
            target_batch = max(1, maximum_pair_batch // source_count)
            for target_start in range(0, atom_count, target_batch):
                target_stop = min(atom_count, target_start + target_batch)
                target_indices = torch.arange(
                    target_start, target_stop, dtype=torch.long, device=positions.device
                )
                source_fractional = fractional_positions[source_indices][:, None, :].expand(
                    -1, target_indices.numel(), -1
                )
                target_fractional = fractional_positions[target_indices][None, :, :].expand(
                    source_count, -1, -1
                )
                fractional_displacement = target_fractional - source_fractional
                fractional_displacement = (
                    fractional_displacement - torch.round(fractional_displacement)
                )
                displacement = fractional_displacement @ membership_cell
                adjacent = (
                    torch.linalg.vector_norm(
                        displacement, dim=-1
                    )
                    < cutoff
                ).any(dim=0)
                reached[target_indices] |= adjacent
        new_frontier_mask = reached & ~closure
        hop[new_frontier_mask] = layer
        closure |= reached
        frontier = torch.nonzero(new_frontier_mask, as_tuple=False).flatten()

    return closure, hop


def _build_cutoff_edges_chunked(
    selected_positions: Any,
    cell: Any,
    cutoff: float,
    *,
    maximum_pair_batch: int = 100_000,
) -> tuple[Any, Any]:
    """Build directed MIC cutoff edges without allocating an N-by-N tensor."""

    torch = _torch()
    count = int(selected_positions.shape[0])
    membership_positions = selected_positions.detach()
    membership_cell = cell.detach()
    fractional_positions = torch.linalg.solve(
        membership_cell.T, membership_positions.T
    ).T
    sender_parts = []
    receiver_parts = []
    shift_parts = []
    # Choose the source block so all targets normally fit in one block.  This
    # preserves the historical sender-major edge ordering without a global sort.
    maximum_source_batch = max(
        1, min(64, maximum_pair_batch // max(1, count))
    )

    for source_start in range(0, count, maximum_source_batch):
        source_stop = min(count, source_start + maximum_source_batch)
        source_indices = torch.arange(
            source_start, source_stop, dtype=torch.long, device=selected_positions.device
        )
        source_count = int(source_indices.numel())
        target_batch = max(1, maximum_pair_batch // source_count)
        for target_start in range(0, count, target_batch):
            target_stop = min(count, target_start + target_batch)
            target_indices = torch.arange(
                target_start, target_stop, dtype=torch.long, device=selected_positions.device
            )
            source_fractional = fractional_positions[source_indices][:, None, :]
            target_fractional = fractional_positions[target_indices][None, :, :]
            fractional_delta = target_fractional - source_fractional
            unit_shift = -torch.round(fractional_delta)
            displacement = (fractional_delta + unit_shift) @ membership_cell
            keep = torch.linalg.vector_norm(displacement, dim=-1) < cutoff
            keep &= source_indices[:, None] != target_indices[None, :]
            if not bool(keep.any().item()):
                continue
            local_sender, local_receiver = torch.nonzero(keep, as_tuple=True)
            sender_parts.append(source_indices[local_sender])
            receiver_parts.append(target_indices[local_receiver])
            shift_parts.append(unit_shift[local_sender, local_receiver])

    if sender_parts:
        sender = torch.cat(sender_parts)
        receiver = torch.cat(receiver_parts)
        unit_shifts = torch.cat(shift_parts)
        edge_index = torch.stack((sender, receiver), dim=0)
    else:
        edge_index = torch.empty(
            (2, 0), dtype=torch.long, device=selected_positions.device
        )
        unit_shifts = torch.empty(
            (0, 3), dtype=selected_positions.dtype, device=selected_positions.device
        )
    return edge_index, unit_shifts


def build_mace_graph(
    full_positions_angstrom: Any,
    cell_angstrom: Any,
    *,
    environment_manifest: Mapping[str, Any],
    atom_mapping: Mapping[str, Any],
    model_atomic_numbers: tuple[int, ...] | list[int],
    config: MaceGraphConfig,
) -> dict[str, Any]:
    """Build one real MACE 0.3.16 input dictionary from verified identities.

    The full topology coordinate tensor is required so omitted atoms in the
    exact cutoff-graph message-passing closure can be detected instead of
    silently ignored.
    """

    torch = _torch()
    environment, mapping = _validate_identity(environment_manifest, atom_mapping)
    atom_count = environment["payload"]["atom_count"]
    positions = _floating_tensor(
        full_positions_angstrom, "full_positions_angstrom", shape=(atom_count, 3)
    )
    cell = _floating_tensor(cell_angstrom, "cell_angstrom", shape=(3, 3))
    if positions.device != cell.device or positions.dtype != cell.dtype:
        raise MaceGraphError("positions and cell must share dtype and device")
    sign, logdet = torch.linalg.slogdet(cell)
    if bool((sign == 0).item()) or not bool(torch.isfinite(logdet).item()):
        raise MaceGraphError("cell must be non-singular")
    cutoff = float(config.edge_cutoff_angstrom)
    # This builder emits one MIC image per ordered atom pair. Multiple images
    # would be required in a smaller cell, so fail rather than omit valid edges.
    if bool((_face_heights(cell) <= 2.0 * cutoff).any().item()):
        raise MaceGraphError("cell is too small for unique minimum-image edges at this cutoff")

    payload = mapping["payload"]
    topology_order = payload["topology_indices_by_mace_node_index"]
    if len(topology_order) != payload["node_count"]:
        raise MaceGraphError("mapping is incomplete")
    topology_index = torch.tensor(topology_order, dtype=torch.long, device=positions.device)
    selected_positions = positions[topology_index]
    selected_set = set(topology_order)
    ligand_topology = environment["payload"]["ligand_indices"]

    # Exact message-passing support is the N-hop closure of the ligand on the
    # configured edge_cutoff_angstrom graph.  ``cutoff * layers`` is only a
    # geometric upper bound; using it as a ligand-centred sphere incorrectly
    # pulls in disconnected water/protein atoms and many irrelevant
    # environment--environment edges.
    closure, minimum_hop = topology_n_hop_closure(
        positions,
        cell,
        ligand_topology,
        edge_cutoff_angstrom=cutoff,
        interaction_layers=config.interaction_layers,
    )
    omitted = [
        index
        for index in range(atom_count)
        if bool(closure[index].item()) and index not in selected_set
    ]
    if omitted:
        raise MaceGraphError(
            "candidate support omits topology atoms in the exact cutoff-graph "
            "message-passing closure: "
            + repr(omitted[:20])
        )

    nodes = payload["nodes_by_topology_index"]
    node_by_topology = {node["topology_index"]: node for node in nodes}
    atomic_numbers = [node_by_topology[index]["atomic_number"] for index in topology_order]
    model_numbers = tuple(int(value) for value in model_atomic_numbers)
    if not model_numbers or len(set(model_numbers)) != len(model_numbers):
        raise MaceGraphError("model_atomic_numbers must be a unique non-empty sequence")
    unsupported = sorted(set(atomic_numbers) - set(model_numbers))
    if unsupported:
        raise MaceGraphError(f"model does not support atomic numbers: {unsupported}")
    number_to_channel = {number: channel for channel, number in enumerate(model_numbers)}
    channels = torch.tensor(
        [number_to_channel[number] for number in atomic_numbers],
        dtype=torch.long,
        device=positions.device,
    )
    node_attrs = torch.nn.functional.one_hot(channels, num_classes=len(model_numbers)).to(
        dtype=positions.dtype
    )

    count = selected_positions.shape[0]
    edge_index, unit_shifts = _build_cutoff_edges_chunked(
        selected_positions, cell, cutoff
    )
    # Recompute from the live cell; do not store detached Cartesian images.
    shifts = unit_shifts @ cell
    if edge_index.numel() and (
        bool((edge_index < 0).any().item()) or bool((edge_index >= count).any().item())
    ):
        raise MaceGraphError("edge index is out of bounds")
    if count > 1 and edge_index.shape[1] == 0:
        raise MaceGraphError(f"graph has no edges at the declared {cutoff} Angstrom cutoff")

    ligand_mask = torch.tensor(
        [bool(node_by_topology[index]["ligand_mask"]) for index in topology_order],
        dtype=torch.bool,
        device=positions.device,
    )
    if not bool(ligand_mask.any().item()) or bool(ligand_mask.all().item()):
        raise MaceGraphError("graph must contain both ligand and environment nodes")
    graph = {
        "positions": selected_positions,
        "node_attrs": node_attrs,
        "edge_index": edge_index,
        "shifts": shifts,
        "unit_shifts": unit_shifts.to(dtype=positions.dtype),
        "cell": cell.unsqueeze(0),
        "batch": torch.zeros(count, dtype=torch.long, device=positions.device),
        "ptr": torch.tensor([0, count], dtype=torch.long, device=positions.device),
        "head": torch.zeros(1, dtype=torch.long, device=positions.device),
        "pbc": torch.ones((1, 3), dtype=torch.bool, device=positions.device),
        "total_charge": torch.zeros(1, dtype=positions.dtype, device=positions.device),
        "total_spin": torch.ones(1, dtype=positions.dtype, device=positions.device),
    }
    return {
        "data": graph,
        "ligand_mask": ligand_mask,
        "environment_mask": ~ligand_mask,
        "topology_indices_by_mace_node_index": topology_index,
        "diagnostics": {
            "node_count": count,
            "edge_count": edge_index.shape[1],
            "ligand_count": int(ligand_mask.sum().item()),
            "environment_count": int((~ligand_mask).sum().item()),
            "edge_cutoff_angstrom": cutoff,
            "interaction_layers": config.interaction_layers,
            "geometric_upper_bound_angstrom": float(config.geometric_upper_bound_angstrom),
            "support_definition": "exact_cutoff_graph_n_hop_closure",
            "full_topology_closure_count": int(closure.sum().item()),
            "full_topology_closure_counts_by_hop": [
                int((minimum_hop == layer).sum().item())
                for layer in range(config.interaction_layers + 1)
            ],
            "edge_membership_piecewise_fixed": True,
            "edge_enumeration": "chunked_no_n_by_n_allocation",
            "maximum_pair_batch": 100_000,
            "cartesian_shifts_recomputed_from_live_cell": True,
        },
    }


__all__ = [
    "MaceGraphConfig",
    "MaceGraphError",
    "build_mace_graph",
    "topology_n_hop_closure",
]
