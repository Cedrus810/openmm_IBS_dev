"""Per-frame exact-closure MACE graph for the offline EXP-012 teacher.

``local_residual.mace_graph.build_mace_graph`` requires a fixed, sealed
environment manifest/atom mapping -- a fixed node identity contract that
makes sense for an online model that must produce a stable differentiable
graph inside an OpenMM step. The offline teacher (`original_6a`/`derived_5a`,
DEC-030) never enters OpenMM or MD, so nothing requires its node set to stay
fixed across frames.

DEC-031 tried building one manifest as the union of all 1500 target-ledger
frames' closures, to get exact per-frame coverage from a single fixed graph.
That grew the candidate pool from 1444 to 4915 nodes -- larger than the
2135-node graph that already failed to fit in 24 GiB of VRAM at 6 Angstrom.
DEC-032 rejects that policy: for an offline tool, there is no reason to pay
for a fixed graph at all. This module builds ``S_a``, the exact two-hop
cutoff-graph closure of the ligand at frame ``a``'s live positions, fresh for
every frame. That is smaller than any fixed union, has zero support-domain
violations by construction (it *is* the true closure, not an approximation of
it), and never has to exclude a frame to keep a fixed node set small -- which
would have meant preferentially dropping exactly the highest-motion frames a
latent cache most needs to see.

This also drops complete-residue expansion. That expansion existed for
EXP-010's fragment-energy decomposition, which required whole residues to
subtract a valid fragment energy; the ligand latent read out here never
computes a fragment energy, so a residue only partially inside the closure is
not a completeness violation the way it was in EXP-010. Whether dropping it
changes the ligand latent is an empirical question, not an assumption --
see ``scripts/smoke_exp012_teacher_graph_equivalence.py``.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from local_residual.environment import canonical_json_bytes
from local_residual.mace_graph import (
    MaceGraphConfig,
    MaceGraphError,
    _build_cutoff_edges_chunked,
    _face_heights,
    _floating_tensor,
    _torch,
    topology_n_hop_closure,
)


def build_teacher_graph_for_frame(
    full_positions_angstrom: Any,
    cell_angstrom: Any,
    *,
    ligand_indices: Sequence[int],
    atomic_numbers_by_topology_index: Sequence[int],
    model_atomic_numbers: Sequence[int],
    config: MaceGraphConfig,
) -> dict[str, Any]:
    """Build one frame's exact two-hop-closure MACE graph.

    ``S_a = S0 (ligand) | S1 (5 A neighbors of S0) | S2 (5 A neighbors of S1)``
    is computed directly from this frame's live positions -- there is no
    environment manifest or atom mapping to validate against, and no
    complete-residue expansion. ``atomic_numbers_by_topology_index`` must
    cover every atom in the topology (not just a candidate subset), since the
    closure can reach any atom.
    """

    torch = _torch()
    positions = _floating_tensor(full_positions_angstrom, "full_positions_angstrom")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise MaceGraphError("full_positions_angstrom must have shape (N, 3)")
    cell = _floating_tensor(cell_angstrom, "cell_angstrom", shape=(3, 3))
    if positions.device != cell.device or positions.dtype != cell.dtype:
        raise MaceGraphError("positions and cell must share dtype and device")
    sign, logdet = torch.linalg.slogdet(cell)
    if bool((sign == 0).item()) or not bool(torch.isfinite(logdet).item()):
        raise MaceGraphError("cell must be non-singular")
    cutoff = float(config.edge_cutoff_angstrom)
    if bool((_face_heights(cell) <= 2.0 * cutoff).any().item()):
        raise MaceGraphError("cell is too small for unique minimum-image edges at this cutoff")

    atom_count = int(positions.shape[0])
    if len(atomic_numbers_by_topology_index) != atom_count:
        raise MaceGraphError("atomic_numbers_by_topology_index must cover every topology atom")

    ligand_topology = sorted({int(index) for index in ligand_indices})
    if not ligand_topology or ligand_topology[0] < 0 or ligand_topology[-1] >= atom_count:
        raise MaceGraphError("ligand_indices is out of range for this topology")
    ligand_set = set(ligand_topology)

    closure, minimum_hop = topology_n_hop_closure(
        positions, cell, ligand_topology,
        edge_cutoff_angstrom=cutoff, interaction_layers=config.interaction_layers,
    )
    topology_order = sorted(
        int(index) for index in torch.nonzero(closure, as_tuple=False).flatten().tolist()
    )
    topology_index = torch.tensor(topology_order, dtype=torch.long, device=positions.device)
    selected_positions = positions[topology_index]
    count = int(selected_positions.shape[0])

    atomic_numbers = [int(atomic_numbers_by_topology_index[index]) for index in topology_order]
    model_numbers = tuple(int(value) for value in model_atomic_numbers)
    if not model_numbers or len(set(model_numbers)) != len(model_numbers):
        raise MaceGraphError("model_atomic_numbers must be a unique non-empty sequence")
    unsupported = sorted(set(atomic_numbers) - set(model_numbers))
    if unsupported:
        raise MaceGraphError(f"model does not support atomic numbers: {unsupported}")
    number_to_channel = {number: channel for channel, number in enumerate(model_numbers)}
    channels = torch.tensor(
        [number_to_channel[number] for number in atomic_numbers],
        dtype=torch.long, device=positions.device,
    )
    node_attrs = torch.nn.functional.one_hot(channels, num_classes=len(model_numbers)).to(
        dtype=positions.dtype
    )

    edge_index, unit_shifts = _build_cutoff_edges_chunked(selected_positions, cell, cutoff)
    shifts = unit_shifts @ cell
    if edge_index.numel() and (
        bool((edge_index < 0).any().item()) or bool((edge_index >= count).any().item())
    ):
        raise MaceGraphError("edge index is out of bounds")
    if count > 1 and edge_index.shape[1] == 0:
        raise MaceGraphError(f"graph has no edges at the declared {cutoff} Angstrom cutoff")

    ligand_mask = torch.tensor(
        [index in ligand_set for index in topology_order],
        dtype=torch.bool, device=positions.device,
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
    hop_at_selected = minimum_hop[topology_index]
    return {
        "data": graph,
        "ligand_mask": ligand_mask,
        "environment_mask": ~ligand_mask,
        "topology_indices_by_mace_node_index": topology_index,
        "diagnostics": {
            "node_count": count,
            "edge_count": int(edge_index.shape[1]),
            "ligand_count": int(ligand_mask.sum().item()),
            "environment_count": int((~ligand_mask).sum().item()),
            "edge_cutoff_angstrom": cutoff,
            "interaction_layers": config.interaction_layers,
            "support_definition": "exact_cutoff_graph_n_hop_closure_no_fixed_manifest",
            "hop_counts_by_layer": [
                int((hop_at_selected == layer).sum().item())
                for layer in range(config.interaction_layers + 1)
            ],
            "complete_residue_expansion": False,
            "fixed_environment_manifest": False,
        },
    }


def _graph_membership_sha256(topology_index: Any, edge_index: Any, unit_shifts: Any) -> str:
    """Deterministic hash over the discrete graph decision.

    Covers which atoms (by topology index) and which directed edges -- each
    with its periodic image -- were selected. Independent of execution
    device/dtype: two membership computations that agree on this hash used
    the identical discrete graph, not just the same counts.
    """
    torch = _torch()
    topo_list = [int(value) for value in topology_index.tolist()]
    edges = edge_index.t().tolist()
    shift_ints = unit_shifts.round().to(dtype=torch.int64).tolist()
    edge_records = sorted(
        (int(sender), int(receiver), tuple(int(component) for component in shift))
        for (sender, receiver), shift in zip(edges, shift_ints)
    )
    payload = {
        "topology_index": topo_list,
        "edges": [[sender, receiver, list(shift)] for sender, receiver, shift in edge_records],
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_canonical_graph_membership(
    positions_angstrom_cpu_float64: Any,
    cell_angstrom_cpu_float64: Any,
    *,
    ligand_indices: Sequence[int],
    edge_cutoff_angstrom: float,
    interaction_layers: int,
) -> dict[str, Any]:
    """Decide graph membership exactly once, on CPU float64.

    A CPU-only geometry audit and a CUDA float32 bulk run independently
    deciding cutoff-graph membership can disagree on a handful of atom pairs
    sitting within floating-point rounding distance of the exact cutoff --
    not a bug in either computation, just two different floating-point
    implementations separately answering a boundary-sensitive discrete
    question. Deciding membership in exactly one place, on CPU float64, and
    handing the *result* (not a re-derivation) to whatever device actually
    runs MACE removes that disagreement by construction instead of tolerating
    it with a tolerance band.
    """

    torch = _torch()
    positions = _floating_tensor(positions_angstrom_cpu_float64, "positions_angstrom_cpu_float64")
    if positions.device.type != "cpu" or positions.dtype != torch.float64:
        raise MaceGraphError("graph membership must be computed on CPU in float64")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise MaceGraphError("positions_angstrom_cpu_float64 must have shape (N, 3)")
    cell = _floating_tensor(cell_angstrom_cpu_float64, "cell_angstrom_cpu_float64", shape=(3, 3))
    if cell.device.type != "cpu" or cell.dtype != torch.float64:
        raise MaceGraphError("graph membership must be computed on CPU in float64")
    sign, logdet = torch.linalg.slogdet(cell)
    if bool((sign == 0).item()) or not bool(torch.isfinite(logdet).item()):
        raise MaceGraphError("cell must be non-singular")
    cutoff = float(edge_cutoff_angstrom)
    if bool((_face_heights(cell) <= 2.0 * cutoff).any().item()):
        raise MaceGraphError("cell is too small for unique minimum-image edges at this cutoff")

    atom_count = int(positions.shape[0])
    ligand_topology = sorted({int(index) for index in ligand_indices})
    if not ligand_topology or ligand_topology[0] < 0 or ligand_topology[-1] >= atom_count:
        raise MaceGraphError("ligand_indices is out of range for this topology")
    ligand_set = set(ligand_topology)

    closure, minimum_hop = topology_n_hop_closure(
        positions, cell, ligand_topology,
        edge_cutoff_angstrom=cutoff, interaction_layers=interaction_layers,
    )
    topology_order = sorted(
        int(index) for index in torch.nonzero(closure, as_tuple=False).flatten().tolist()
    )
    topology_index = torch.tensor(topology_order, dtype=torch.long)
    selected_positions = positions[topology_index]
    count = int(selected_positions.shape[0])

    edge_index, unit_shifts = _build_cutoff_edges_chunked(selected_positions, cell, cutoff)
    if count > 1 and edge_index.shape[1] == 0:
        raise MaceGraphError(f"graph has no edges at the declared {cutoff} Angstrom cutoff")

    ligand_mask = torch.tensor([index in ligand_set for index in topology_order], dtype=torch.bool)
    if not bool(ligand_mask.any().item()) or bool(ligand_mask.all().item()):
        raise MaceGraphError("graph must contain both ligand and environment nodes")

    hop_at_selected = minimum_hop[topology_index]
    hop_counts_by_layer = [
        int((hop_at_selected == layer).sum().item()) for layer in range(interaction_layers + 1)
    ]
    return {
        "topology_index": topology_index,
        "edge_index": edge_index,
        "unit_shifts": unit_shifts,
        "ligand_mask": ligand_mask,
        "node_count": count,
        "edge_count": int(edge_index.shape[1]),
        "hop_counts_by_layer": hop_counts_by_layer,
        "edge_cutoff_angstrom": cutoff,
        "interaction_layers": interaction_layers,
        "graph_membership_sha256": _graph_membership_sha256(topology_index, edge_index, unit_shifts),
        "graph_membership_device": "cpu",
        "graph_membership_dtype": "float64",
    }


def build_teacher_graph_from_membership(
    membership: Mapping[str, Any],
    target_positions: Any,
    target_cell: Any,
    *,
    atomic_numbers_by_topology_index: Sequence[int],
    model_atomic_numbers: Sequence[int],
) -> dict[str, Any]:
    """Assemble the MACE-ready graph on ``target_positions``'s device/dtype.

    Discrete membership (which atoms, which edges, which periodic image) is
    never recomputed here -- it was already decided by
    ``compute_canonical_graph_membership``. Only atom selection and the
    differentiable shift/position tensors are built on the target device, so
    a CUDA float32 execution never re-derives (and can never disagree with)
    the CPU float64 membership decision.
    """

    torch = _torch()
    target_positions = _floating_tensor(target_positions, "target_positions")
    if target_positions.ndim != 2 or target_positions.shape[1] != 3:
        raise MaceGraphError("target_positions must have shape (N, 3)")
    target_cell = _floating_tensor(target_cell, "target_cell", shape=(3, 3))
    if target_positions.device != target_cell.device or target_positions.dtype != target_cell.dtype:
        raise MaceGraphError("target_positions and target_cell must share dtype and device")

    device = target_positions.device
    dtype = target_positions.dtype
    source_topology_index = membership["topology_index"]
    if int(source_topology_index.max().item()) >= int(target_positions.shape[0]):
        raise MaceGraphError(
            "target_positions has fewer atoms than the membership topology requires"
        )
    topology_index = source_topology_index.to(device=device)
    edge_index = membership["edge_index"].to(device=device)
    unit_shifts = membership["unit_shifts"].to(device=device, dtype=dtype)
    ligand_mask = membership["ligand_mask"].to(device=device)

    selected_positions = target_positions[topology_index]
    count = int(selected_positions.shape[0])
    shifts = unit_shifts @ target_cell
    if edge_index.numel() and (
        bool((edge_index < 0).any().item()) or bool((edge_index >= count).any().item())
    ):
        raise MaceGraphError("edge index is out of bounds")

    topology_order = [int(value) for value in source_topology_index.tolist()]
    if len(atomic_numbers_by_topology_index) <= max(topology_order):
        raise MaceGraphError("atomic_numbers_by_topology_index must cover every topology atom")
    atomic_numbers = [int(atomic_numbers_by_topology_index[index]) for index in topology_order]
    model_numbers = tuple(int(value) for value in model_atomic_numbers)
    if not model_numbers or len(set(model_numbers)) != len(model_numbers):
        raise MaceGraphError("model_atomic_numbers must be a unique non-empty sequence")
    unsupported = sorted(set(atomic_numbers) - set(model_numbers))
    if unsupported:
        raise MaceGraphError(f"model does not support atomic numbers: {unsupported}")
    number_to_channel = {number: channel for channel, number in enumerate(model_numbers)}
    channels = torch.tensor(
        [number_to_channel[number] for number in atomic_numbers],
        dtype=torch.long, device=device,
    )
    node_attrs = torch.nn.functional.one_hot(channels, num_classes=len(model_numbers)).to(dtype=dtype)

    graph = {
        "positions": selected_positions,
        "node_attrs": node_attrs,
        "edge_index": edge_index,
        "shifts": shifts,
        "unit_shifts": unit_shifts,
        "cell": target_cell.unsqueeze(0),
        "batch": torch.zeros(count, dtype=torch.long, device=device),
        "ptr": torch.tensor([0, count], dtype=torch.long, device=device),
        "head": torch.zeros(1, dtype=torch.long, device=device),
        "pbc": torch.ones((1, 3), dtype=torch.bool, device=device),
        "total_charge": torch.zeros(1, dtype=dtype, device=device),
        "total_spin": torch.ones(1, dtype=dtype, device=device),
    }
    return {
        "data": graph,
        "ligand_mask": ligand_mask,
        "environment_mask": ~ligand_mask,
        "topology_indices_by_mace_node_index": topology_index,
        "diagnostics": {
            "node_count": count,
            "edge_count": int(edge_index.shape[1]),
            "ligand_count": int(ligand_mask.sum().item()),
            "environment_count": int((~ligand_mask).sum().item()),
            "edge_cutoff_angstrom": membership["edge_cutoff_angstrom"],
            "interaction_layers": membership["interaction_layers"],
            "support_definition": "exact_cutoff_graph_n_hop_closure_no_fixed_manifest",
            "hop_counts_by_layer": membership["hop_counts_by_layer"],
            "complete_residue_expansion": False,
            "fixed_environment_manifest": False,
            "graph_membership_sha256": membership["graph_membership_sha256"],
            "graph_membership_device": membership["graph_membership_device"],
            "graph_membership_dtype": membership["graph_membership_dtype"],
            "model_execution_device": str(device),
            "model_execution_dtype": str(dtype),
        },
    }


__all__ = [
    "build_teacher_graph_for_frame",
    "build_teacher_graph_from_membership",
    "compute_canonical_graph_membership",
]
