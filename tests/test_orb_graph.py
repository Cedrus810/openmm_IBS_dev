from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from local_residual.orb_graph import OrbGraphAuditError, audit_lhop_graphs


def test_lhop_audit_reports_scale_and_neighbor_cap():
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    report = audit_lhop_graphs(
        positions,
        torch.eye(3, dtype=torch.float64) * 40.0,
        ligand_indices=[0],
        cutoff_angstrom=6.0,
        max_num_neighbors=1,
        max_layer=2,
    )
    assert report["membership_dtype"] == "float64"
    assert report["layers"][0]["node_count"] == 2
    assert report["layers"][1]["node_count"] == 3
    assert report["layers"][1]["hop_counts"] == {"0": 1, "1": 1, "2": 1}
    assert report["layers"][1]["max_outgoing_neighbors"] == 2
    assert report["layers"][1]["cap_hit"] is True


def test_lhop_audit_requires_cpu_float64():
    with pytest.raises(OrbGraphAuditError, match="CPU float64"):
        audit_lhop_graphs(
            torch.zeros((2, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32) * 40.0,
            ligand_indices=[0],
        )


def test_lhop_audit_accepts_numpy_float64_inputs():
    import numpy as np

    report = audit_lhop_graphs(
        np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float64),
        np.eye(3, dtype=np.float64) * 40.0,
        ligand_indices=[0],
        max_layer=1,
    )
    assert report["layers"][0]["node_count"] == 1
