"""Boresch anchors: θA/θB must stay inside a K-independent [30°, 150°] window.

openfe's "angle not flat" test (½Kθ² ≥ 10 kT) with K = 334.72 admits θ down to
~22°; brd4_ligand2 got θA≈156–159° / θB≈25–27°, near-collinear anchors that only
pass because K is stiff (collinearity cost 1–2 kT once K is softened to 41.84).
"""

import numpy as np
import pytest

import abfe_core as ac

pytestmark = pytest.mark.cpu_only


def test_window_bounds():
    deg = np.radians([10.0, 29.9, 30.0, 90.0, 150.0, 150.1, 170.0])
    assert ac.anchor_theta_in_window(deg).tolist() == [False, False, True, True, True, False, False]


def test_near_collinear_passes_old_flat_test_but_fails_window():
    th = np.radians(np.full(50, 157.0))
    assert ac.openfe_angle_not_flat(th, ac.OPENFE_FIXED_KANGLE_KJ_PER_MOL_RAD2, 300.0).all()
    assert not ac.anchor_theta_in_window(th).any()


class _FakeTraj:
    def __init__(self, xyz):
        self.xyz = np.asarray(xyz, dtype=float)


def _six_atoms(theta_b_deg):
    """H2 H1 H0 G0 G1 G2 with θA = 90° and a chosen θB; small per-frame jitter."""
    frames = []
    rng = np.random.default_rng(0)
    tb = np.radians(theta_b_deg)
    for _ in range(20):
        H0 = np.array([0.0, 0.0, 0.0])
        H1 = np.array([-0.15, 0.0, 0.0])
        H2 = np.array([-0.20, 0.10, 0.10])          # out of plane: φA well away from ±π
        G0 = np.array([0.0, 0.8, 0.0])              # θA = H1-H0-G0 = 90°
        # G0->G1 makes angle tb with G0->H0 (= -y), bent towards +z: φB ≈ 90°
        G1 = G0 + 0.14 * (np.cos(tb) * np.array([0.0, -1.0, 0.0]) + np.sin(tb) * np.array([0.0, 0.0, 1.0]))
        G2 = G1 + np.array([0.10, 0.0, 0.05])
        X = np.stack([H0, H1, H2, G0, G1, G2]) + rng.normal(scale=1e-4, size=(6, 3))
        frames.append(X)
    return _FakeTraj(frames)


@pytest.mark.parametrize("theta_b, ok", [(90.0, True), (25.0, False), (155.0, False)])
def test_anchor_metrics_rejects_outside_window(theta_b, ok):
    # openfe_anchor_metrics lives on OrbBoreschEstimator, whose __init__ needs torch;
    # the method itself only reads config/thresholds, so skip the constructor.
    est = object.__new__(ac.OrbBoreschEstimator)
    est.config = {}
    est.T = 300.0
    est.gas_constant_kj_per_mol_k = 8.314e-3
    traj = _six_atoms(theta_b)
    accepted, reason, _, _ = est.openfe_anchor_metrics((0, 1, 2), (3, 4, 5), traj,
                                                       k_angle=ac.OPENFE_FIXED_KANGLE_KJ_PER_MOL_RAD2,
                                                       temperature_k=300.0)
    assert accepted is ok, reason
    if not ok:
        assert "30°" in reason
