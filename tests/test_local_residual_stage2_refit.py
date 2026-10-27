"""local_residual.stage2_refit (2026-10-17): features are refit.frame_basis_features (ligand centres,
environment = everything else, local ligand order handled); capacities are
the exact neighbour-cap bound and refuse what the CUDA plugin would refuse; runs without water refused."""

import numpy as np
import pytest

from local_residual import refit as rf
from local_residual import stage2_refit as s2
from local_residual.softlift import derived_r1_config


def _config(n_types, n_ligand):
    caps = s2.stage2_capacities(n_ligand, 8192)
    return derived_r1_config("0" * 64, type_vocabulary=tuple(range(1, n_types + 1)), n_ligand_atoms=n_ligand, **caps)


def test_features_match_reference_triclinic():
    rng = np.random.default_rng(0)
    # rhombic-dodecahedron-like reduced box (Å)
    a = 40.0
    box = np.array([[a, 0, 0], [0, a, 0], [a / 2, a / 2, a * np.sqrt(2) / 2]])
    n, n_types = 400, 3
    x = rng.uniform(0, 1, (n, 3)) @ box
    ligand = np.arange(8)
    x[ligand] = x[0] + rng.normal(0, 1.5, (8, 3))
    x[0] = box[0] * 0.98  # ligand straddles a face: edges must use the minimum image
    types = rng.integers(0, n_types, n)
    cfg = _config(n_types, len(ligand))
    fast, used = s2.frame_features(x, box, ligand, types, n_types, cfg)
    env = np.setdiff1d(np.arange(n), ligand)
    ref, ref_used = rf.frame_basis_features(
        positions_angstrom=x, box_angstrom=box, ligand_indices=ligand, environment_indices=env,
        ligand_type_index=types[ligand], environment_type_index=types, n_types=n_types,
        radial_centers=np.linspace(0, cfg.outer_cutoff_angstrom, cfg.n_radial_basis),
        radial_width=cfg.outer_cutoff_angstrom / (cfg.n_radial_basis - 1),
        inner_cutoff_angstrom=cfg.inner_cutoff_angstrom, outer_cutoff_angstrom=cfg.outer_cutoff_angstrom)
    assert ref_used["edges"] > 0
    assert used == ref_used
    np.testing.assert_allclose(fast, ref, rtol=1e-10, atol=1e-10)


def test_capacities_bound_and_plugin_limit():
    caps = s2.stage2_capacities(46, 8192)
    assert caps == {"max_environment_atoms": 3680, "max_edges": 3680, "max_neighbors_per_ligand": 80}
    with pytest.raises(rf.RefitError):
        s2.stage2_capacities(103, 8192)  # 103 × 80 = 8240 ≥ 8192


def test_run_without_water_is_refused(tmp_path):
    seg = tmp_path / "vanishing"
    seg.mkdir()
    (tmp_path / "checkpoints").mkdir()
    K, N = 3, 10
    np.save(seg / "dual_window_0_vdw_energies.npy", np.zeros((K, N)))
    np.save(seg / "dual_window_0_vdw_sampling_states.npy", np.zeros((N, K)))
    (tmp_path / "checkpoints" / "ibs_state_vdw_window_0.json").write_text('{"f_k": [0, 0, 0]}')
    (seg / "dual_window_0_vdw_convergence.json").write_text('{"lambdas_vdw": [0.6, 0.5, 0.4]}')
    np.savez(seg / "dual_window_0_vdw_coords.npz", idx=np.arange(0, N, 5), xyz=np.zeros((2, 2, 3)),
             box=np.tile(np.eye(3) * 4, (2, 1, 1)), atom_indices=np.array([0, 1]))
    with pytest.raises(rf.RefitError, match="没存水"):
        s2.load_run_windows(str(tmp_path), ligand=[0], atomic_numbers=[6, 8], vocabulary=(6, 8),
                            config=_config(2, 1), temperature_kelvin=300.0)
