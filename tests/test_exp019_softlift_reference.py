"""EXP-019 R1/R2/R3 packed reference contracts."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from local_residual.softlift import (  # noqa: E402
    PackedSoftLiftBatch,
    SoftLiftConfig,
    build_softlift_model,
    count_trainable_parameters,
)

pytestmark = pytest.mark.cpu_only


def _config(rung: str, n_ligand: int = 2) -> SoftLiftConfig:
    return SoftLiftConfig(
        schema_version="exp019-softlift-v1", rung=rung,
        type_vocabulary=(1, 6, 8), n_ligand_atoms=n_ligand,
        n_radial_basis=8, n_channels=1 if rung == "R1" else 4,
        pair_dim=0 if rung == "R1" else 8,
        context_dim=0 if rung == "R1" else 8,
        inner_cutoff_angstrom=4.0, outer_cutoff_angstrom=5.0,
        b_max_reduced=10.0, max_environment_atoms=320,
        max_edges=2048, max_neighbors_per_ligand=80,
        no_contact_output="exact_zero", protocol_sha256="0" * 64,
    )


def _batch(rung: str = "R1", distances=None, displacement=False):
    distances = torch.tensor([2.0, 3.0, 2.5], dtype=torch.float64) if distances is None else distances
    disp = None
    if displacement:
        disp = torch.stack((distances, torch.zeros_like(distances), torch.zeros_like(distances)), dim=1)
    return PackedSoftLiftBatch(
        ligand_type_index=torch.tensor([[0, 2]], dtype=torch.int64),
        edge_frame=torch.zeros(3, dtype=torch.int64),
        edge_ligand_local=torch.tensor([0, 0, 1], dtype=torch.int64),
        edge_ligand_type=torch.tensor([0, 0, 2], dtype=torch.int64),
        edge_environment_type=torch.tensor([1, 2, 1], dtype=torch.int64),
        edge_distance_angstrom=distances,
        edge_displacement_angstrom=disp,
    )


def test_r1_empty_neighborhood_is_exact_zero_and_is_bounded():
    config = _config("R1")
    model = build_softlift_model(config).double().eval()
    empty = PackedSoftLiftBatch(
        ligand_type_index=torch.tensor([[0, 2]], dtype=torch.int64),
        edge_frame=torch.empty(0, dtype=torch.int64),
        edge_ligand_local=torch.empty(0, dtype=torch.int64),
        edge_ligand_type=torch.empty(0, dtype=torch.int64),
        edge_environment_type=torch.empty(0, dtype=torch.int64),
        edge_distance_angstrom=torch.empty(0, dtype=torch.float64),
    )
    output = model(empty)
    assert output.shape == (1,)
    assert output.item() == 0.0
    assert count_trainable_parameters(model) < 10_000


def test_r1_edge_permutation_and_packed_vs_single_frame_are_identical():
    # Deliberately make the per-frame ceiling smaller than the packed total;
    # this guards against confusing a ragged batch's total E with its frame E.
    config = replace(_config("R1"), max_edges=3)
    model = build_softlift_model(config).double().eval()
    batch = _batch()
    output = model(batch)
    perm = torch.tensor([2, 0, 1], dtype=torch.int64)
    permuted = replace(
        batch,
        edge_frame=batch.edge_frame[perm],
        edge_ligand_local=batch.edge_ligand_local[perm],
        edge_ligand_type=batch.edge_ligand_type[perm],
        edge_environment_type=batch.edge_environment_type[perm],
        edge_distance_angstrom=batch.edge_distance_angstrom[perm],
    )
    assert torch.allclose(output, model(permuted), rtol=0.0, atol=1.0e-14)
    single = model(batch)
    two = PackedSoftLiftBatch(
        ligand_type_index=torch.cat((batch.ligand_type_index, batch.ligand_type_index), dim=0),
        edge_frame=torch.cat((batch.edge_frame, batch.edge_frame + 1)),
        edge_ligand_local=torch.cat((batch.edge_ligand_local, batch.edge_ligand_local)),
        edge_ligand_type=torch.cat((batch.edge_ligand_type, batch.edge_ligand_type)),
        edge_environment_type=torch.cat((batch.edge_environment_type, batch.edge_environment_type)),
        edge_distance_angstrom=torch.cat((batch.edge_distance_angstrom, batch.edge_distance_angstrom)),
    )
    assert torch.allclose(torch.stack((single[0], single[0])), model(two), rtol=0.0, atol=1.0e-14)


def test_r1_two_neighbors_have_nonzero_mixed_second_derivative():
    config = _config("R1", n_ligand=1)
    model = build_softlift_model(config).double().eval()
    with torch.no_grad():
        model.pair_weight.fill_(1.0)
    distances = torch.tensor([2.0, 2.7], dtype=torch.float64, requires_grad=True)
    batch = PackedSoftLiftBatch(
        ligand_type_index=torch.tensor([[0]], dtype=torch.int64),
        edge_frame=torch.zeros(2, dtype=torch.int64),
        edge_ligand_local=torch.zeros(2, dtype=torch.int64),
        edge_ligand_type=torch.zeros(2, dtype=torch.int64),
        edge_environment_type=torch.ones(2, dtype=torch.int64),
        edge_distance_angstrom=distances,
    )
    value = model(batch)[0]
    first = torch.autograd.grad(value, distances, create_graph=True)[0][0]
    mixed = torch.autograd.grad(first, distances)[0][1]
    assert abs(float(mixed.item())) > 1.0e-12


@pytest.mark.parametrize("rung", ["R2", "R3"])
def test_context_rungs_are_finite_and_r3_is_rotation_invariant(rung):
    model = build_softlift_model(_config(rung)).double().eval()
    batch = _batch(rung, displacement=rung == "R3")
    value = model(batch)
    assert value.shape == (1,)
    assert bool(torch.isfinite(value).all())
    if rung == "R3":
        rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        rotated = replace(batch, edge_displacement_angstrom=batch.edge_displacement_angstrom @ rotation.T)
        assert torch.allclose(value, model(rotated), rtol=0.0, atol=1.0e-12)
