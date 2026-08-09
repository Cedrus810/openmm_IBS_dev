"""DEC-037 D3 sub-item 4 (post-profiling): the deployable student's cell-list
neighbor search must find the exact same candidate edges as the original
O(n_ligand * n_system) all-pairs path it replaces.

Profiling (`scripts/profile_exp012_student_torchforce_overhead_d3.py`)
confirmed all-pairs graph construction dominated the measured 258% ms/step
overhead (~7.7ms vs ~0.3ms network math per call, ~27x). Per the explicit
instruction this fix responds to ("replace only the neighbor-discovery
implementation while preserving the exact cutoff, edge set, periodic shifts,
model weights, and energy function"), this file is a fast, pure-CPU,
synthetic-geometry check that the new `_cell_list_candidates` path and the
unchanged `_brute_force_candidates` fallback agree exactly -- run before
(and independently of) the expensive real-trajectory/GPU D3 rerun, which
re-checks the same property against real production geometry via
`reference_eager_vs_deployable_eager` in
`scripts/check_exp012_student_deployment_d3.py`.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from local_residual.student import build_local_residual_student  # noqa: E402
from local_residual.student_deploy import build_deployable_student_module  # noqa: E402

pytestmark = pytest.mark.cpu_only

DTYPE = torch.float64
TYPE_VOCABULARY = (1, 6, 7, 8, 16)


def _synthetic_deployable(*, outer_cutoff_angstrom=5.0, inner_cutoff_angstrom=4.0):
    model = build_local_residual_student(
        TYPE_VOCABULARY,
        hidden_dim=8,
        n_interaction_blocks=2,
        n_radial_basis=6,
        inner_cutoff_angstrom=inner_cutoff_angstrom,
        outer_cutoff_angstrom=outer_cutoff_angstrom,
        b_max_reduced=10.0,
    ).to(DTYPE)
    model.eval()

    n_ligand = 5
    n_environment = 300
    n_atoms = n_ligand + n_environment
    ligand_topology_indices = list(range(n_ligand))
    generator = torch.Generator().manual_seed(0)
    atomic_numbers = [int(TYPE_VOCABULARY[i % len(TYPE_VOCABULARY)]) for i in range(n_atoms)]

    deployable = build_deployable_student_module(
        model,
        ligand_topology_indices=ligand_topology_indices,
        all_topology_atomic_numbers=atomic_numbers,
        temperature_kelvin=300.0,
        a_k=1.0,
    ).to(DTYPE)
    deployable.eval()
    return deployable, n_atoms, generator


def _edge_pair_set(ligand_topology, environment_topology):
    return set(
        zip(ligand_topology.tolist(), environment_topology.tolist())
    )


@pytest.mark.parametrize("box_length_angstrom", [15.0, 20.0, 25.3])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_cell_list_matches_brute_force_edge_set_and_distances(box_length_angstrom, seed):
    deployable, n_atoms, _ = _synthetic_deployable()
    generator = torch.Generator().manual_seed(seed)
    positions = torch.rand((n_atoms, 3), generator=generator, dtype=DTYPE) * box_length_angstrom
    box = torch.eye(3, dtype=DTYPE) * box_length_angstrom

    # This test's box (length >= 15 = 3 * outer_cutoff_angstrom(5.0)) always
    # satisfies the cell list's own >=3-bins-per-axis correctness condition,
    # so both code paths are valid to call directly and compare.
    box_diagonal = torch.stack((box[0, 0], box[1, 1], box[2, 2]))
    n_bins = torch.floor(box_diagonal / deployable.outer_cutoff_angstrom).to(torch.int64)
    assert int(n_bins.min().item()) >= 3, "test box too small for the property being tested"

    brute_ligand, brute_environment, brute_distance = deployable._brute_force_candidates(positions, box)
    cell_ligand, cell_environment, cell_distance = deployable._cell_list_candidates(positions, box, n_bins)

    assert brute_ligand.numel() > 0, "synthetic density should produce at least one edge"
    assert _edge_pair_set(brute_ligand, brute_environment) == _edge_pair_set(cell_ligand, cell_environment), (
        "cell list must find the exact same (ligand, environment) candidate pairs as brute force"
    )

    # Same pairs, but not necessarily the same order -- sort both by
    # (ligand, environment) before comparing per-edge distances.
    def _sorted_by_pair(ligand_topology, environment_topology, distance):
        order = sorted(
            range(ligand_topology.numel()),
            key=lambda i: (int(ligand_topology[i]), int(environment_topology[i])),
        )
        index = torch.tensor(order, dtype=torch.int64)
        return distance.index_select(0, index)

    brute_distance_sorted = _sorted_by_pair(brute_ligand, brute_environment, brute_distance)
    cell_distance_sorted = _sorted_by_pair(cell_ligand, cell_environment, cell_distance)
    max_distance_error = float((brute_distance_sorted - cell_distance_sorted).abs().max().item())
    assert max_distance_error < 1e-10, f"per-edge distance mismatch: {max_distance_error:.3e}"


def test_cell_list_matches_brute_force_full_forward_energy_and_force():
    # Compare the two candidate-finding paths end-to-end (through the shared
    # embedding/message-passing/readout pipeline, `_energy_from_edges`) on
    # the *exact same* (positions, box) input -- not forward()'s automatic
    # dispatch forced apart by perturbing the box. A perturbed box is a
    # different physical configuration (its own minimum-image geometry in
    # `_minimum_image_displacement` shifts by O(the perturbation), regardless
    # of any cell-list bug), so comparing forward() on two different boxes
    # conflates "different code path" with "different physics" and cannot
    # isolate the property this test is meant to check.
    deployable, n_atoms, generator = _synthetic_deployable()
    box_length_angstrom = 20.0
    positions_angstrom = torch.rand((n_atoms, 3), generator=generator, dtype=DTYPE) * box_length_angstrom
    box_angstrom = torch.eye(3, dtype=DTYPE) * box_length_angstrom

    n_bins = torch.floor(
        torch.stack((box_angstrom[0, 0], box_angstrom[1, 1], box_angstrom[2, 2]))
        / deployable.outer_cutoff_angstrom
    ).to(torch.int64)
    assert int(n_bins.min().item()) >= 3, "test box too small for the property being tested"

    positions_cell_list = positions_angstrom.clone().detach().requires_grad_(True)
    cell_edges = deployable._cell_list_candidates(positions_cell_list, box_angstrom, n_bins)
    energy_cell_list = deployable._energy_from_edges(positions_cell_list, *cell_edges)
    energy_cell_list.backward()
    force_via_cell_list = positions_cell_list.grad.clone()

    positions_brute_force = positions_angstrom.clone().detach().requires_grad_(True)
    brute_edges = deployable._brute_force_candidates(positions_brute_force, box_angstrom)
    energy_brute_force = deployable._energy_from_edges(positions_brute_force, *brute_edges)
    energy_brute_force.backward()
    force_via_brute_force = positions_brute_force.grad.clone()

    energy_error = abs(float(energy_cell_list.item()) - float(energy_brute_force.item()))
    force_error = float((force_via_cell_list - force_via_brute_force).abs().max().item())
    assert energy_error < 1e-8, f"energy mismatch between cell-list and brute-force forward paths: {energy_error:.3e}"
    assert force_error < 1e-8, f"force mismatch between cell-list and brute-force forward paths: {force_error:.3e}"


def test_small_box_falls_back_to_brute_force_without_raising():
    # A box smaller than 3 * outer_cutoff_angstrom must not attempt the cell
    # list (its correctness condition would not hold); forward() must still
    # produce a finite, valid result via the brute-force fallback.
    deployable, n_atoms, generator = _synthetic_deployable(outer_cutoff_angstrom=5.0)
    small_box_length_angstrom = 8.0  # < 3 * 5.0, forces min_bins < 3
    positions_angstrom = torch.rand((n_atoms, 3), generator=generator, dtype=DTYPE) * small_box_length_angstrom
    positions_nm = positions_angstrom / 10.0
    box_nm = torch.eye(3, dtype=DTYPE) * small_box_length_angstrom / 10.0

    energy = deployable(positions_nm, box_nm)
    assert bool(torch.isfinite(energy).item())


def test_torchscript_export_still_succeeds_with_cell_list(tmp_path):
    deployable, n_atoms, generator = _synthetic_deployable()
    scripted = torch.jit.script(deployable)
    box_length_angstrom = 20.0
    positions_nm = (torch.rand((n_atoms, 3), generator=generator, dtype=DTYPE) * box_length_angstrom) / 10.0
    box_nm = torch.eye(3, dtype=DTYPE) * box_length_angstrom / 10.0

    eager_energy = deployable(positions_nm, box_nm)
    scripted_energy = scripted(positions_nm, box_nm)
    assert abs(float(eager_energy.item()) - float(scripted_energy.item())) < 1e-10
