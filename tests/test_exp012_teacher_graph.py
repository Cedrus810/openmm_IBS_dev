"""DEC-032: per-frame exact-closure teacher graph, no fixed environment manifest.

Unlike ``build_mace_graph`` (tested in test_exp012_mace_graph.py), this
function takes no environment manifest or atom mapping -- ``S_a`` is computed
fresh from live positions every call. The behavior worth pinning down is
exactly the DEC-032 rationale: the node set is the true closure (nothing more
-- no complete-residue expansion), ligand nodes keep a deterministic relative
order regardless of how many environment atoms are interspersed, and an
atom outside the closure never leaks in.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from local_residual.mace_graph import MaceGraphConfig, MaceGraphError
from local_residual.teacher_graph import (
    build_teacher_graph_for_frame,
    build_teacher_graph_from_membership,
    compute_canonical_graph_membership,
)


def _config(edge_cutoff=6.0, layers=2, upper_bound=12.0):
    return MaceGraphConfig(
        edge_cutoff_angstrom=edge_cutoff, interaction_layers=layers,
        geometric_upper_bound_angstrom=upper_bound,
    )


def test_closure_only_no_residue_expansion(tmp_path):
    # Atom 4 sits inside the 12 A geometric upper bound but is not reachable
    # within two real cutoff-graph hops -- it must not appear, exactly as in
    # build_mace_graph, but here there is no manifest declaring that up front:
    # the closure computation itself is the only source of truth.
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],   # S0 ligand
            [0.1, 0.0, 0.0],   # S0 ligand
            [5.0, 0.0, 0.0],   # S1
            [10.0, 0.0, 0.0],  # S2 through atom 2
            [-8.0, 0.0, 0.0],  # inside 12 A, but disconnected within two hops
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = [6, 6, 8, 8, 8]
    graph = build_teacher_graph_for_frame(
        positions, torch.eye(3, dtype=torch.float64) * 40.0,
        ligand_indices=[0, 1],
        atomic_numbers_by_topology_index=atomic_numbers,
        model_atomic_numbers=[6, 8],
        config=_config(),
    )
    assert graph["diagnostics"]["node_count"] == 4
    assert graph["diagnostics"]["hop_counts_by_layer"] == [2, 1, 1]
    assert graph["diagnostics"]["complete_residue_expansion"] is False
    assert graph["diagnostics"]["fixed_environment_manifest"] is False
    assert 4 not in graph["topology_indices_by_mace_node_index"].tolist()


def test_ligand_relative_order_is_fixed_regardless_of_environment_size():
    # Ligand atoms are topology indices 3 and 4; environment atoms sit both
    # below and above them in topology-index order. Sorting by topology index
    # keeps 3 before 4 among ligand nodes no matter how many environment
    # atoms are interspersed -- this is what lets a cached ligand latent line
    # up across frames whose environment atom count differs.
    positions = torch.tensor(
        [
            [-5.0, 0.0, 0.0],  # env, topology index 0, below ligand
            [-3.0, 0.0, 0.0],  # env, topology index 1, below ligand
            [-1.0, 0.0, 0.0],  # env, topology index 2, below ligand
            [0.0, 0.0, 0.0],   # ligand, topology index 3
            [0.1, 0.0, 0.0],   # ligand, topology index 4
            [2.0, 0.0, 0.0],   # env, topology index 5, above ligand
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    atomic_numbers = [8, 8, 8, 6, 6, 8]
    graph = build_teacher_graph_for_frame(
        positions, torch.eye(3, dtype=torch.float64) * 40.0,
        ligand_indices=[3, 4],
        atomic_numbers_by_topology_index=atomic_numbers,
        model_atomic_numbers=[6, 8],
        config=_config(edge_cutoff=6.0, layers=2, upper_bound=12.0),
    )
    order = graph["topology_indices_by_mace_node_index"].tolist()
    assert order == sorted(order)
    ligand_positions_in_order = [
        index for index, topo in enumerate(order) if bool(graph["ligand_mask"][index].item())
    ]
    ligand_topo_indices = [order[index] for index in ligand_positions_in_order]
    assert ligand_topo_indices == [3, 4]


def test_unsupported_element_fails_closed():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    with pytest.raises(MaceGraphError, match="does not support"):
        build_teacher_graph_for_frame(
            positions, torch.eye(3, dtype=torch.float64) * 40.0,
            ligand_indices=[0],
            atomic_numbers_by_topology_index=[6, 8],
            model_atomic_numbers=[6],
            config=_config(),
        )


def test_differentiable_through_selected_positions():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64, requires_grad=True,
    )
    graph = build_teacher_graph_for_frame(
        positions, torch.eye(3, dtype=torch.float64) * 40.0,
        ligand_indices=[0],
        atomic_numbers_by_topology_index=[6, 8, 8],
        model_atomic_numbers=[6, 8],
        config=_config(),
    )
    graph["data"]["positions"].square().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()


def _membership_case():
    # Same layout as test_closure_only_no_residue_expansion: atom 4 sits
    # inside the geometric upper bound but outside the true two-hop closure.
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [5.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [-8.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 40.0
    return positions, cell


def test_membership_requires_cpu_float64():
    positions, cell = _membership_case()
    with pytest.raises(MaceGraphError, match="CPU in float64"):
        compute_canonical_graph_membership(
            positions.to(torch.float32), cell.to(torch.float32),
            ligand_indices=[0, 1], edge_cutoff_angstrom=6.0, interaction_layers=2,
        )


def test_split_membership_and_assembly_matches_combined_function():
    # DEC-032 Option C: deciding membership separately from execution must not
    # change the resulting graph when both happen on the same device/dtype --
    # the split is a refactor of *where* the decision is made, not a change
    # to what it decides.
    positions, cell = _membership_case()
    atomic_numbers = [6, 6, 8, 8, 8]
    model_atomic_numbers = [6, 8]

    combined = build_teacher_graph_for_frame(
        positions.clone().requires_grad_(True), cell,
        ligand_indices=[0, 1],
        atomic_numbers_by_topology_index=atomic_numbers,
        model_atomic_numbers=model_atomic_numbers,
        config=_config(),
    )

    membership = compute_canonical_graph_membership(
        positions, cell, ligand_indices=[0, 1],
        edge_cutoff_angstrom=6.0, interaction_layers=2,
    )
    split = build_teacher_graph_from_membership(
        membership, positions.clone().requires_grad_(True), cell,
        atomic_numbers_by_topology_index=atomic_numbers,
        model_atomic_numbers=model_atomic_numbers,
    )

    assert split["diagnostics"]["node_count"] == combined["diagnostics"]["node_count"]
    assert split["diagnostics"]["edge_count"] == combined["diagnostics"]["edge_count"]
    assert split["diagnostics"]["hop_counts_by_layer"] == combined["diagnostics"]["hop_counts_by_layer"]
    assert torch.equal(
        split["topology_indices_by_mace_node_index"],
        combined["topology_indices_by_mace_node_index"],
    )
    assert torch.equal(split["ligand_mask"], combined["ligand_mask"])
    assert torch.allclose(split["data"]["positions"], combined["data"]["positions"])
    assert torch.equal(split["data"]["edge_index"], combined["data"]["edge_index"])
    assert split["diagnostics"]["graph_membership_device"] == "cpu"
    assert split["diagnostics"]["graph_membership_dtype"] == "float64"
    assert split["diagnostics"]["model_execution_device"] == "cpu"
    assert split["diagnostics"]["model_execution_dtype"] == "torch.float64"


def test_assembly_at_different_precision_keeps_membership_hash_and_reports_execution_precision():
    # Simulates the real bulk-cache pathway (CPU float64 membership, CUDA
    # float32 execution) using CPU float32 as the stand-in execution
    # precision, since this test suite doesn't require a GPU. The membership
    # hash must be carried through unchanged -- it describes a decision that
    # was never re-derived at the new precision.
    positions, cell = _membership_case()
    atomic_numbers = [6, 6, 8, 8, 8]
    membership = compute_canonical_graph_membership(
        positions, cell, ligand_indices=[0, 1],
        edge_cutoff_angstrom=6.0, interaction_layers=2,
    )
    execution_positions = positions.to(torch.float32).requires_grad_(True)
    execution_cell = cell.to(torch.float32)
    graph = build_teacher_graph_from_membership(
        membership, execution_positions, execution_cell,
        atomic_numbers_by_topology_index=atomic_numbers,
        model_atomic_numbers=[6, 8],
    )
    assert graph["diagnostics"]["node_count"] == membership["node_count"]
    assert graph["diagnostics"]["edge_count"] == membership["edge_count"]
    assert graph["diagnostics"]["graph_membership_sha256"] == membership["graph_membership_sha256"]
    assert graph["diagnostics"]["graph_membership_device"] == "cpu"
    assert graph["diagnostics"]["graph_membership_dtype"] == "float64"
    assert graph["diagnostics"]["model_execution_device"] == "cpu"
    assert graph["diagnostics"]["model_execution_dtype"] == "torch.float32"
    assert graph["data"]["positions"].dtype == torch.float32
    assert graph["data"]["unit_shifts"].dtype == torch.float32


def test_membership_topology_shorter_than_target_positions_fails_closed():
    positions, cell = _membership_case()
    membership = compute_canonical_graph_membership(
        positions, cell, ligand_indices=[0, 1],
        edge_cutoff_angstrom=6.0, interaction_layers=2,
    )
    truncated_positions = positions[:2].clone().requires_grad_(True)
    with pytest.raises(MaceGraphError, match="fewer atoms"):
        build_teacher_graph_from_membership(
            membership, truncated_positions, cell,
            atomic_numbers_by_topology_index=[6, 6],
            model_atomic_numbers=[6, 8],
        )
