"""`_numpy_cross_edges` non-diagonal branch: a half-box tie only fails closed on active pairs.

2026-10-08, SERT (rhombic-dodecahedron box): the closed-form refit died with
"half-box MIC tie detected" on a pair 89.8 A apart (frame 11 of a float32 DCD),
far outside the 5 A cutoff. The CUDA plugin (exp025RoundWithTieCheckM) and the
Torch deploy path (softlift_deploy.py) already escalate ties only for pairs
inside outer_cutoff; the numpy fallback must agree with them.
"""

import numpy as np
import pytest

from local_residual.softlift_dataset import SoftLiftDatasetError, _numpy_cross_edges


def _dodecahedron(length):
    return np.array(
        [
            [length, 0.0, 0.0],
            [0.0, length, 0.0],
            [0.5 * length, 0.5 * length, 0.5 * np.sqrt(2.0) * length],
        ]
    )


def test_distant_half_box_tie_does_not_fail_closed():
    box = _dodecahedron(100.0)
    ligand = np.zeros(3)
    near = np.array([3.0, 0.0, 0.0])
    # exactly half a box along the third lattice vector: fractional (0, 0, 0.5), ~50 A away
    tied = 0.5 * box[2]
    positions = np.stack([ligand, near, tied])
    edge_ligand, edge_env, _, distance = _numpy_cross_edges(positions, box, [0], [1, 2], 5.0)
    assert edge_ligand.tolist() == [0]
    assert edge_env.tolist() == [1]
    assert distance.tolist() == pytest.approx([3.0])


def test_active_half_box_tie_still_fails_closed():
    # box so small that a half-box tie lies inside the cutoff: must still refuse
    box = _dodecahedron(8.0)
    positions = np.stack([np.zeros(3), 0.5 * box[0]])  # 4 A away, fractional (0.5, 0, 0)
    with pytest.raises(SoftLiftDatasetError, match="half-box MIC tie"):
        _numpy_cross_edges(positions, box, [0], [1], 5.0)
