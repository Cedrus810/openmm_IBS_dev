"""EXP-019 canonical edge and dataset-v1 audit tests."""

import pytest

np = pytest.importorskip("numpy")

from local_residual.softlift_dataset import (  # noqa: E402
    SoftLiftDatasetError,
    canonicalize_frame_edges,
    validate_softlift_dataset_arrays,
)


def test_canonical_key_sorts_and_rejects_duplicate_keys():
    common = dict(
        ligand_topology_indices=[10, 11], outer_cutoff_angstrom=5.0,
        min_distance_support_angstrom=0.1, max_environment_atoms=320,
        max_edges=2048, max_neighbors_per_ligand=80,
    )
    result = canonicalize_frame_edges(
        ligand_topology=np.array([11, 10]), environment_topology=np.array([4, 3]),
        displacement_angstrom=np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]]),
        distance_angstrom=np.array([2.0, 3.0]), pair_type=np.array([[6, 8], [6, 8]]),
        unit_shift=np.zeros((2, 3), dtype=np.int64), **common,
    )
    assert result["edge_ligand_local"].tolist() == [0, 1]
    with pytest.raises(SoftLiftDatasetError):
        canonicalize_frame_edges(
            ligand_topology=np.array([10, 10]), environment_topology=np.array([4, 4]),
            displacement_angstrom=np.array([[0.0, 2.0, 0.0], [0.0, 2.0, 0.0]]),
            distance_angstrom=np.array([2.0, 2.0]), pair_type=np.array([[6, 8], [6, 8]]),
            unit_shift=np.zeros((2, 3), dtype=np.int64), **common,
        )


def test_dataset_audit_checks_pair_type_and_box_safety():
    arrays = {
        "partition_index": np.array([0]), "frame_index_within_run": np.array([0]),
        "box_vectors_angstrom": np.eye(3)[None, :, :] * 20.0,
        "edge_offsets": np.array([0, 1]), "edge_ligand_topology": np.array([10]),
        "edge_environment_topology": np.array([4]), "edge_ligand_local": np.array([0]),
        "edge_distance_angstrom": np.array([2.0]), "edge_displacement_angstrom": np.array([[0.0, 2.0, 0.0]]),
        "edge_unit_shift": np.zeros((1, 3), dtype=np.int64), "edge_pair_type": np.array([[6, 8]]),
        "adjacent_gap_reduced": np.zeros((1, 1)), "log_importance_unnormalized": np.zeros((1, 2)),
        "ligand_topology_indices": np.array([10]), "ligand_atomic_numbers": np.array([6]),
        "all_topology_atomic_numbers": np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 6]),
        "type_vocabulary": np.array([6, 8]), "delta_A": np.array([0.1]), "A_k_window": np.array([0.0, 0.1]),
    }
    audit = validate_softlift_dataset_arrays(arrays)
    assert audit["edge_count"] == 1
    arrays["edge_pair_type"] = np.array([[1, 8]])
    with pytest.raises(SoftLiftDatasetError):
        validate_softlift_dataset_arrays(arrays)
