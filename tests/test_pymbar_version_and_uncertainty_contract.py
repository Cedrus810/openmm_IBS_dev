"""Runtime dependency contract for GitHub issue #76."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

EXPECTED_PYMBAR_VERSION = "4.2.0"


def test_pymbar_420_default_uncertainty_matches_explicit_svd_ew():
    pymbar = pytest.importorskip("pymbar")
    actual_version = str(pymbar.__version__)
    assert actual_version == EXPECTED_PYMBAR_VERSION, (
        f"当前环境加载 pymbar {actual_version}，协议要求 {EXPECTED_PYMBAR_VERSION}。"
        "environment.yml 只定义目标环境，不会自动升级已激活环境；请运行 "
        "`mamba install -n openmm_dev -c conda-forge pymbar-core=4.2.0`，"
        "重新激活环境后再测试。"
    )

    # Two sampled states, four configurations from each state.  The values are
    # deterministic and have overlap, so both covariance calls are well posed.
    u_kn = np.array(
        [
            [0.00, 0.10, -0.10, 0.05, 0.80, 1.00, 0.90, 1.10],
            [0.90, 1.10, 1.00, 0.80, 0.00, 0.10, -0.10, 0.05],
        ],
        dtype=np.float64,
    )
    n_k = np.array([4, 4], dtype=np.int64)
    estimator = pymbar.MBAR(u_kn, n_k, verbose=False)

    default = estimator.compute_free_energy_differences(uncertainty_method=None)
    explicit = estimator.compute_free_energy_differences(uncertainty_method="svd-ew")

    np.testing.assert_allclose(default["Delta_f"], explicit["Delta_f"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(default["dDelta_f"], explicit["dDelta_f"], atol=0.0, rtol=0.0)
