"""DEC-037 (d0-2)/(d0-5): the minimal `LocalResidualStudent` architecture.

Covers the pure `reindex_ligand_environment_edges` plumbing (no torch
required), the model's forward/autograd contract on synthetic geometry built
with the already-audited `local_residual.geometry` primitives, the (d0-5)
trainable-parameter hard ceiling for the frozen default hyperparameters, and a
synthetic exactly-learnable case (mirroring
`test_exp012_local_residual_linear_readout.py`'s style) confirming the model
can actually drive the DEC-034/035 gap-variance loss down via real gradient
training -- not just produce a finite forward pass.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from local_residual.geometry import ligand_environment_cross_edges  # noqa: E402
from local_residual.loss import bidirectional_gap_variance_loss  # noqa: E402
from local_residual.student import (  # noqa: E402
    LocalResidualStudentError,
    build_local_residual_student,
    count_trainable_parameters,
    reindex_ligand_environment_edges,
)

pytestmark = pytest.mark.cpu_only

DTYPE = torch.float64
TYPE_VOCABULARY = (1, 6, 7, 8, 16)  # H, C, N, O, S -- matches the project's existing element set


# ---------------------------------------------------------------------------
# reindex_ligand_environment_edges (pure, no torch-module construction)
# ---------------------------------------------------------------------------


def test_reindex_maps_topology_indices_into_compact_local_space():
    ligand_topology_indices = [10, 20, 30]
    edge_ligand_topology = torch.tensor([20, 10, 20, 30], dtype=torch.int64)
    edge_environment_topology = torch.tensor([500, 500, 400, 500], dtype=torch.int64)

    result = reindex_ligand_environment_edges(
        ligand_topology_indices, edge_ligand_topology, edge_environment_topology
    )

    assert result["environment_topology_indices"].tolist() == [400, 500]
    assert result["edge_ligand_local"].tolist() == [1, 0, 1, 2]
    # env 500 -> local 1, env 400 -> local 0 (sorted ascending)
    assert result["edge_environment_local"].tolist() == [1, 1, 0, 1]


def test_reindex_handles_empty_edges():
    result = reindex_ligand_environment_edges([1, 2], torch.empty((0,), dtype=torch.int64), torch.empty((0,), dtype=torch.int64))
    assert result["environment_topology_indices"].shape == (0,)
    assert result["edge_ligand_local"].shape == (0,)
    assert result["edge_environment_local"].shape == (0,)


def test_reindex_rejects_duplicate_ligand_ordering_and_unknown_topology_index():
    with pytest.raises(LocalResidualStudentError):
        reindex_ligand_environment_edges(
            [1, 1], torch.tensor([1], dtype=torch.int64), torch.tensor([9], dtype=torch.int64)
        )
    with pytest.raises(LocalResidualStudentError):
        reindex_ligand_environment_edges(
            [1, 2], torch.tensor([3], dtype=torch.int64), torch.tensor([9], dtype=torch.int64)
        )


# ---------------------------------------------------------------------------
# Model construction and input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_dim": 0},
        {"n_interaction_blocks": 0},
        {"n_radial_basis": 0},
        {"inner_cutoff_angstrom": 5.0, "outer_cutoff_angstrom": 5.0},
        {"inner_cutoff_angstrom": 6.0, "outer_cutoff_angstrom": 5.0},
        {"b_max_reduced": 0.0},
        {"b_max_reduced": -1.0},
    ],
)
def test_invalid_hyperparameters_fail_closed(kwargs):
    with pytest.raises(LocalResidualStudentError):
        build_local_residual_student(TYPE_VOCABULARY, **kwargs)


def test_duplicate_type_vocabulary_fails_closed():
    with pytest.raises(LocalResidualStudentError):
        build_local_residual_student((1, 1, 6))


def test_unknown_atomic_number_fails_closed():
    model = build_local_residual_student(TYPE_VOCABULARY)
    with pytest.raises(LocalResidualStudentError):
        model.atomic_numbers_to_type_index([1, 6, 99])


def test_repeated_construction_reuses_the_same_class_object():
    first = build_local_residual_student(TYPE_VOCABULARY)
    second = build_local_residual_student(TYPE_VOCABULARY)
    assert type(first) is type(second)
    assert isinstance(second, type(first))


# ---------------------------------------------------------------------------
# (d0-5) trainable-parameter hard ceiling for the frozen default hyperparameters
# ---------------------------------------------------------------------------


def test_default_hyperparameters_stay_within_the_d0_5_hard_ceiling():
    model = build_local_residual_student(TYPE_VOCABULARY)
    n_params = count_trainable_parameters(model)
    assert n_params > 0
    # (d0-5), DEC-039: target <=50,000, hard ceiling <=100,000.
    assert n_params <= 50_000, (
        f"default LocalResidualStudent has {n_params} trainable parameters, "
        "above the (d0-5) target of 50,000 -- shrink hidden_dim/blocks/radial "
        "basis or explicitly re-review the (d0-5) budget, do not silently proceed"
    )


# ---------------------------------------------------------------------------
# Forward pass on synthetic geometry (built with the audited geometry module)
# ---------------------------------------------------------------------------


def _synthetic_frame(rng: torch.Generator, n_ligand: int = 5, n_environment: int = 30, box_length: float = 5.5):
    # box_length is deliberately small relative to the model's default 5.0 A
    # outer cutoff: the maximum possible minimum-image distance in a cubic box
    # of this size is (box_length/2)*sqrt(3) = 2.75*sqrt(3) =~ 4.76 A < 5.0 A,
    # so every ligand-environment pair is guaranteed (not just probable) to be
    # a real edge -- these tests must not be flaky depending on the RNG seed.
    box = torch.eye(3, dtype=DTYPE) * box_length
    n_atoms = n_ligand + n_environment
    positions = torch.rand((n_atoms, 3), generator=rng, dtype=DTYPE) * box_length
    ligand_topology_indices = list(range(n_ligand))
    environment_topology_indices = torch.arange(n_ligand, n_atoms, dtype=torch.int64)
    atomic_numbers = [6] * n_ligand + [8] * n_environment  # arbitrary but in-vocabulary
    return positions, box, ligand_topology_indices, environment_topology_indices, atomic_numbers


def test_forward_is_finite_and_bounded_by_b_max():
    rng = torch.Generator().manual_seed(0)
    b_max = 3.0
    # .to(DTYPE): parameters default to float32; the geometry primitives below
    # are exercised in float64 (matching test_exp012_geometry.py's own
    # convention), and torch's Linear/Embedding require matching dtypes.
    model = build_local_residual_student(TYPE_VOCABULARY, b_max_reduced=b_max).to(DTYPE)
    positions, box, ligand_topology_indices, environment_topology_indices, atomic_numbers = _synthetic_frame(rng)

    edges = ligand_environment_cross_edges(
        positions, box,
        torch.tensor(ligand_topology_indices, dtype=torch.int64), environment_topology_indices,
        outer_cutoff=model.outer_cutoff_angstrom,
    )
    reindexed = reindex_ligand_environment_edges(
        ligand_topology_indices, edges["edge_index"][0], edges["edge_index"][1]
    )
    assert edges["edge_index"].shape[1] > 0, "synthetic box/cutoff should produce at least one real edge"

    ligand_type_index = model.atomic_numbers_to_type_index(atomic_numbers[: len(ligand_topology_indices)])
    environment_atomic_numbers = [
        atomic_numbers[len(ligand_topology_indices):][int(index) - len(ligand_topology_indices)]
        for index in reindexed["environment_topology_indices"].tolist()
    ]
    environment_type_index = model.atomic_numbers_to_type_index(environment_atomic_numbers)

    output = model(
        ligand_type_index, environment_type_index,
        reindexed["edge_ligand_local"], reindexed["edge_environment_local"],
        edges["distance"].to(DTYPE),
    )
    assert output.shape == ()
    assert torch.isfinite(output).item()
    assert abs(output.item()) < b_max  # strict: tanh(x) is in the open interval (-1, 1) for finite x


def test_forward_with_zero_environment_neighbors_still_returns_finite_bounded_scalar():
    model = build_local_residual_student(TYPE_VOCABULARY)
    ligand_type_index = model.atomic_numbers_to_type_index([6, 6, 8])
    empty = torch.empty((0,), dtype=torch.int64)
    output = model(ligand_type_index, empty, empty, empty, torch.empty((0,), dtype=DTYPE))
    assert torch.isfinite(output).item()
    assert abs(output.item()) < model.b_max_reduced


def test_gradient_flows_from_output_back_to_cartesian_coordinates():
    rng = torch.Generator().manual_seed(1)
    model = build_local_residual_student(TYPE_VOCABULARY).to(DTYPE)
    positions, box, ligand_topology_indices, environment_topology_indices, atomic_numbers = _synthetic_frame(rng)
    positions = positions.clone().requires_grad_(True)

    edges = ligand_environment_cross_edges(
        positions, box,
        torch.tensor(ligand_topology_indices, dtype=torch.int64), environment_topology_indices,
        outer_cutoff=model.outer_cutoff_angstrom,
    )
    reindexed = reindex_ligand_environment_edges(
        ligand_topology_indices, edges["edge_index"][0], edges["edge_index"][1]
    )
    ligand_type_index = model.atomic_numbers_to_type_index(atomic_numbers[: len(ligand_topology_indices)])
    environment_atomic_numbers = [
        atomic_numbers[len(ligand_topology_indices):][int(index) - len(ligand_topology_indices)]
        for index in reindexed["environment_topology_indices"].tolist()
    ]
    environment_type_index = model.atomic_numbers_to_type_index(environment_atomic_numbers)

    output = model(
        ligand_type_index, environment_type_index,
        reindexed["edge_ligand_local"], reindexed["edge_environment_local"],
        edges["distance"],
    )
    output.backward()

    assert positions.grad is not None
    assert torch.isfinite(positions.grad).all()
    # Only ligand atoms and the environment atoms actually in an edge can have
    # received a nonzero gradient; assert at least the ligand block moved.
    ligand_grad_norm = positions.grad[: len(ligand_topology_indices)].norm().item()
    assert ligand_grad_norm > 0.0


# ---------------------------------------------------------------------------
# Real gradient-descent training on a synthetic, exactly-learnable gap signal
# ---------------------------------------------------------------------------


def test_training_reduces_gap_variance_on_a_learnable_synthetic_signal():
    """Mirrors test_exp012_local_residual_linear_readout.py's exact-signal style,
    but for the nonlinear student instead of the closed-form linear readout:
    build several synthetic frames per state where the true adjacent-gap
    residual is, by construction, proportional to a real geometric feature
    (mean ligand-environment contact distance) the model can actually learn
    from coordinates -- not from a precomputed latent. A handful of gradient
    steps must reduce `gap_variance_loss` below the B=0 baseline.

    Note: `_synthetic_frame`'s box is deliberately small enough that every
    ligand-environment pair is always in range (see its docstring), so contact
    *count* is constant across frames and would be a useless target here --
    the varying signal across frames is each frame's mean contact *distance*
    (closer configurations vs. farther ones), which the radial-basis/envelope
    features are exactly designed to be sensitive to.
    """

    torch.manual_seed(0)
    rng = torch.Generator().manual_seed(2)
    model = build_local_residual_student(
        TYPE_VOCABULARY, hidden_dim=16, n_interaction_blocks=1, n_radial_basis=8
    ).to(DTYPE)

    n_frames = 24
    delta_a = torch.tensor([0.4, 0.6], dtype=DTYPE)  # 2 edges -> 3 target states
    basis_values = []
    for _ in range(n_frames):
        positions, box, ligand_topology_indices, environment_topology_indices, atomic_numbers = _synthetic_frame(
            rng, n_ligand=4, n_environment=20
        )
        edges = ligand_environment_cross_edges(
            positions, box,
            torch.tensor(ligand_topology_indices, dtype=torch.int64), environment_topology_indices,
            outer_cutoff=model.outer_cutoff_angstrom,
        )
        assert edges["distance"].numel() > 0, "synthetic box is sized so all pairs must be edges"
        reindexed = reindex_ligand_environment_edges(
            ligand_topology_indices, edges["edge_index"][0], edges["edge_index"][1]
        )
        ligand_type_index = model.atomic_numbers_to_type_index(atomic_numbers[: len(ligand_topology_indices)])
        environment_atomic_numbers = [
            atomic_numbers[len(ligand_topology_indices):][int(index) - len(ligand_topology_indices)]
            for index in reindexed["environment_topology_indices"].tolist()
        ]
        environment_type_index = model.atomic_numbers_to_type_index(environment_atomic_numbers)
        basis_values.append(
            {
                "ligand_type_index": ligand_type_index,
                "environment_type_index": environment_type_index,
                "edge_ligand_local": reindexed["edge_ligand_local"],
                "edge_environment_local": reindexed["edge_environment_local"],
                "distance": edges["distance"],
                "mean_proximity": float((model.outer_cutoff_angstrom - edges["distance"]).mean().item()),
            }
        )

    mean_proximity = torch.tensor([frame["mean_proximity"] for frame in basis_values], dtype=DTYPE)
    # Ground truth the model must approximately learn: gap := -delta_A * mean_proximity
    # (plus small noise), so correcting with a basis proportional to
    # mean_proximity drives the corrected gap toward the noise floor only.
    noise = 0.02 * torch.randn(n_frames, 2, generator=torch.Generator().manual_seed(3), dtype=DTYPE)
    true_gaps = noise - delta_a[None, :] * mean_proximity[:, None]
    log_importance = torch.zeros((n_frames, 3), dtype=DTYPE)

    def _forward_all():
        return torch.stack(
            [
                model(
                    frame["ligand_type_index"], frame["environment_type_index"],
                    frame["edge_ligand_local"], frame["edge_environment_local"], frame["distance"],
                )
                for frame in basis_values
            ]
        )

    with torch.no_grad():
        baseline = bidirectional_gap_variance_loss(
            true_gaps, torch.zeros(n_frames, dtype=DTYPE), delta_a, log_importance,
            energy_regularization_coefficient=0.0, force_regularization_coefficient=0.0,
        )["gap_variance_loss"].item()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(400):
        optimizer.zero_grad()
        basis_reduced = _forward_all()
        result = bidirectional_gap_variance_loss(
            true_gaps, basis_reduced, delta_a, log_importance,
            energy_regularization_coefficient=1.0e-4, force_regularization_coefficient=0.0,
        )
        result["loss"].backward()
        optimizer.step()

    with torch.no_grad():
        trained = bidirectional_gap_variance_loss(
            true_gaps, _forward_all(), delta_a, log_importance,
            energy_regularization_coefficient=0.0, force_regularization_coefficient=0.0,
        )["gap_variance_loss"].item()

    assert trained < baseline, (
        f"trained gap_variance_loss ({trained}) did not improve over the B=0 "
        f"baseline ({baseline}) on an exactly-learnable synthetic signal"
    )
