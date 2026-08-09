"""回归：Boresch JSON 格式化必须能处理 NumPy 锚点索引。"""

import json

import numpy as np
import pytest

pytest.importorskip("openmm")

from abfe_core import UnitFormatter  # noqa: E402


def test_format_boresch_json_casts_numpy_anchor_indices():
    params = {
        "boresch_anchors": {
            "receptor_indices": [np.int64(1), np.int64(2), np.int64(3)],
            "ligand_indices": [np.int64(4), np.int64(5), np.int64(6)],
            "equilibrium_values": {
                "r0": 0.4,
                "thetaA0": 2.0,
                "thetaB0": 1.8,
                "phiA0": 0.1,
                "phiB0": -0.2,
                "phiC0": 0.3,
            },
            "force_constants": {
                "kr": 2000.0,
                "kthetaA": 200.0,
                "kthetaB": 200.0,
                "kphiA": 100.0,
                "kphiB": 100.0,
                "kphiC": 100.0,
            },
        },
        "diagnostics": {"source": "regression"},
    }

    formatted = UnitFormatter.format_boresch_json(params)

    # This is intentionally the plain stdlib encoder: formatter output should
    # be directly serializable without relying on a NumPy-aware JSON encoder.
    json.dumps(formatted)
    assert formatted["boresch_anchors"]["receptor_indices"] == [1, 2, 3]
    assert formatted["boresch_anchors"]["ligand_indices"] == [4, 5, 6]
