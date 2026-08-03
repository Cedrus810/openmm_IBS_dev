from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from local_residual.atom_mapping import build_atom_mapping
from local_residual.environment import build_environment_manifest
from local_residual.mace_graph import MaceGraphConfig, MaceGraphError, build_mace_graph


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(tmp_path: Path, *, atom_count: int = 5, selected=(0, 1, 2, 3, 4)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name in ("topology", "base_system", "box"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        sources[name] = {"path": name, "sha256": _sha(path)}
    ligand = [0, 1]
    environment = [index for index in selected if index not in ligand]
    atoms = [
        {"index": index, "stable_id": f"A:{index}", "atomic_number": 6 if index < 2 else 8}
        for index in selected
    ]
    manifest = build_environment_manifest(
        {
            "schema_version": "exp012-environment-config-v1",
            "payload": {
                "sources": sources,
                "atom_count": atom_count,
                "ligand_indices": ligand,
                "environment_candidate_indices": environment,
                "metadata_fields": ["atomic_number"],
                "atoms": atoms,
            },
        },
        workspace_root=tmp_path,
    )
    mapping = build_atom_mapping(
        manifest,
        {
            "schema_version": "exp012-atom-mapping-config-v1",
            "source_environment_manifest_sha256": manifest["canonical_sha256"],
            "supported_atomic_numbers": [1, 6, 8],
            "ordering": {"mode": "manifest_canonical"},
        },
    )
    return manifest, mapping


def _config():
    return MaceGraphConfig(
        edge_cutoff_angstrom=6.0,
        interaction_layers=2,
        geometric_upper_bound_angstrom=12.0,
    )


def test_directed_edges_are_sender_major_and_use_triclinic_pbc(tmp_path):
    manifest, mapping = _identity(tmp_path)
    positions = torch.tensor(
        [[0.2, 0.1, 0.1], [29.0, 0.1, 0.1], [2.0, 0.1, 0.1], [15.0, 15.0, 15.0], [16.0, 15.0, 15.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cell = torch.tensor([[30.0, 0.0, 0.0], [2.0, 30.0, 0.0], [1.0, 0.5, 30.0]], dtype=torch.float64)
    graph = build_mace_graph(
        positions,
        cell,
        environment_manifest=manifest,
        atom_mapping=mapping,
        model_atomic_numbers=[1, 6, 8],
        config=_config(),
    )
    pairs = list(map(tuple, graph["data"]["edge_index"].T.tolist()))
    assert pairs == sorted(pairs)
    assert (0, 1) in pairs and (1, 0) in pairs
    edge = pairs.index((0, 1))
    assert graph["data"]["unit_shifts"][edge].tolist() == [-1.0, 0.0, 0.0]
    vector = (
        graph["data"]["positions"][1]
        - graph["data"]["positions"][0]
        + graph["data"]["shifts"][edge]
    )
    assert torch.allclose(vector, torch.tensor([-1.2, 0.0, 0.0], dtype=torch.float64))
    vector.square().sum().backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert positions.grad[0].abs().sum() > 0 and positions.grad[1].abs().sum() > 0


def test_exact_six_angstrom_cutoff_and_directed_pairs(tmp_path):
    manifest, mapping = _identity(tmp_path)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [5.999, 0.0, 0.0], [6.0, 0.0, 0.0], [14.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    graph = build_mace_graph(
        positions, torch.eye(3, dtype=torch.float64) * 40.0,
        environment_manifest=manifest, atom_mapping=mapping,
        model_atomic_numbers=[1, 6, 8], config=_config(),
    )
    pairs = set(map(tuple, graph["data"]["edge_index"].T.tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs
    assert (0, 2) not in pairs and (2, 0) not in pairs


def test_unsupported_element_incomplete_support_and_small_buffer_fail(tmp_path):
    manifest, mapping = _identity(tmp_path)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [15.0, 0.0, 0.0], [16.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    with pytest.raises(MaceGraphError, match="does not support"):
        build_mace_graph(
            positions, torch.eye(3, dtype=torch.float64) * 40.0,
            environment_manifest=manifest, atom_mapping=mapping,
            model_atomic_numbers=[1, 6], config=_config(),
        )
    with pytest.raises(MaceGraphError, match="insufficient"):
        MaceGraphConfig(6.0, 2, 11.999)

    incomplete_manifest, incomplete_mapping = _identity(
        tmp_path / "incomplete", atom_count=6, selected=(0, 1, 2, 3, 4)
    )
    incomplete_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [15.0, 0.0, 0.0], [16.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    with pytest.raises(MaceGraphError, match="omits topology atoms"):
        build_mace_graph(
            incomplete_positions, torch.eye(3, dtype=torch.float64) * 40.0,
            environment_manifest=incomplete_manifest, atom_mapping=incomplete_mapping,
            model_atomic_numbers=[1, 6, 8], config=_config(),
        )


def test_completeness_uses_two_hop_graph_closure_not_twelve_angstrom_sphere(tmp_path):
    manifest, mapping = _identity(
        tmp_path / "closure", atom_count=5, selected=(0, 1, 2, 3)
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [5.0, 0.0, 0.0],   # S1
            [10.0, 0.0, 0.0],  # S2 through atom 2
            [-8.0, 0.0, 0.0],  # inside 12 A, but disconnected within two hops
        ],
        dtype=torch.float64,
    )
    graph = build_mace_graph(
        positions,
        torch.eye(3, dtype=torch.float64) * 40.0,
        environment_manifest=manifest,
        atom_mapping=mapping,
        model_atomic_numbers=[1, 6, 8],
        config=_config(),
    )
    assert graph["diagnostics"]["support_definition"] == "exact_cutoff_graph_n_hop_closure"
    assert graph["diagnostics"]["full_topology_closure_counts_by_hop"] == [2, 1, 1]
    assert 4 not in graph["topology_indices_by_mace_node_index"].tolist()


def test_tampered_or_nonbijective_mapping_fails_closed(tmp_path):
    manifest, mapping = _identity(tmp_path)
    mapping["payload"]["topology_indices_by_mace_node_index"].pop()
    with pytest.raises(MaceGraphError, match="identity validation failed"):
        build_mace_graph(
            torch.zeros((5, 3), dtype=torch.float64), torch.eye(3, dtype=torch.float64) * 40,
            environment_manifest=manifest, atom_mapping=mapping,
            model_atomic_numbers=[1, 6, 8], config=_config(),
        )
