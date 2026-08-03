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
from local_residual.teacher_graph import build_teacher_graph_for_frame


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
