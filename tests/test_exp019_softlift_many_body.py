"""The R1 nonlinear anchor readout is not pair-additive."""

import pytest

pytestmark = pytest.mark.needs_gpu  # torch 为 GPU 版构建，CPU 上无意义，归 GPU 节点套件

torch = pytest.importorskip("torch")

from local_residual.softlift import PackedSoftLiftBatch, SoftLiftConfig, build_softlift_model  # noqa: E402


def test_r1_has_anchor_level_nonlinearity_and_explicit_zero_centering():
    config = SoftLiftConfig(
        schema_version="exp019-softlift-v1", rung="R1", type_vocabulary=(1, 6), n_ligand_atoms=1,
        n_radial_basis=16, n_channels=1, pair_dim=0, context_dim=0,
        inner_cutoff_angstrom=4.0, outer_cutoff_angstrom=5.0, b_max_reduced=10.0,
        max_environment_atoms=320, max_edges=2048, max_neighbors_per_ligand=80,
        no_contact_output="exact_zero", protocol_sha256="0" * 64,
    )
    model = build_softlift_model(config).double().eval()
    batch_one = PackedSoftLiftBatch(
        ligand_type_index=torch.tensor([[0]], dtype=torch.int64),
        edge_frame=torch.zeros(1, dtype=torch.int64), edge_ligand_local=torch.zeros(1, dtype=torch.int64),
        edge_ligand_type=torch.zeros(1, dtype=torch.int64), edge_environment_type=torch.ones(1, dtype=torch.int64),
        edge_distance_angstrom=torch.tensor([2.0], dtype=torch.float64),
    )
    batch_two = PackedSoftLiftBatch(
        ligand_type_index=torch.tensor([[0]], dtype=torch.int64),
        edge_frame=torch.zeros(2, dtype=torch.int64), edge_ligand_local=torch.zeros(2, dtype=torch.int64),
        edge_ligand_type=torch.zeros(2, dtype=torch.int64), edge_environment_type=torch.ones(2, dtype=torch.int64),
        edge_distance_angstrom=torch.tensor([2.0, 2.0], dtype=torch.float64),
    )
    with torch.no_grad():
        model.pair_weight.fill_(1.0)
    one = model(batch_one)
    two = model(batch_two)
    assert float(two.abs().item()) > 0.0
    # A nonlinear rho makes the same two contacts differ from two separate
    # anchor readouts; this is the capability EXP-014 lacked.
    assert not torch.allclose(two, 2.0 * one, rtol=0.0, atol=1.0e-12)
