"""2026-10-21: frozen outer-residual deployment correction (manifest `deployment` section).

SERT_ABFE/experiments/2026-10-18_ato1Na_outer_ab/PLAN.md §12: per-window basis offset, per-window
re-referenced A_k and scale c. Without the section the factory must behave exactly as before.
"""

import inspect
import math

import pytest

import ibs_engine
from local_residual import openmm_plugin as op


class _Controller:
    def coefficient_matrix(self, lambdas):
        return tuple((math.sin(math.pi * float(lam)) ** 2,) for lam in lambdas)


def _runtime(deployment):
    fields = {f: None for f in op.OuterLambdaLocalResidualRuntime.__dataclass_fields__}
    fields.update(controller=_Controller(), deployment=deployment)
    return op.OuterLambdaLocalResidualRuntime(**fields)


W3 = [0.4, 0.36, 0.32, 0.28, 0.24, 0.2]


def test_no_deployment_section_keeps_plain_coefficients():
    assert op._parse_deployment({}) is None
    got = _runtime(None).state_coefficients_factory([0.0] * 6, W3)
    assert isinstance(got, list) and not isinstance(got, op.WindowCoefficients)
    assert got == pytest.approx([math.sin(math.pi * x) ** 2 for x in W3])


def test_rereference_scale_and_window_offset():
    dep = op._parse_deployment({"deployment": {
        "scale": 0.3, "rereference_window_coefficients": True,
        "window_basis_offsets_kj_mol": [{"lambdas_vdw": [0.4, 0.2], "offset_kj_mol": 7.5}]}})
    got = _runtime(dep).state_coefficients_factory([0.0] * 6, W3)
    a = [math.sin(math.pi * x) ** 2 for x in W3]
    mean = sum(a) / len(a)
    assert isinstance(got, op.WindowCoefficients)
    assert list(got) == pytest.approx([0.3 * (v - mean) for v in a])
    assert sum(got) == pytest.approx(0.0, abs=1e-12)
    assert got.energy_offset_kj_mol == 7.5
    # adjacent differences (the only thing the training objective sees) are c * dA_k
    assert [got[k + 1] - got[k] for k in range(5)] == pytest.approx([0.3 * (a[k + 1] - a[k]) for k in range(5)])


def test_window_not_in_offset_table_is_an_error():
    dep = op._parse_deployment({"deployment": {
        "window_basis_offsets_kj_mol": [{"lambdas_vdw": [1.0, 0.8], "offset_kj_mol": 22.0}]}})
    with pytest.raises(RuntimeError, match="没有覆盖窗口"):
        _runtime(dep).state_coefficients_factory([0.0] * 6, W3)


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
def test_bad_scale_rejected(scale):
    with pytest.raises(RuntimeError, match="scale"):
        op._parse_deployment({"deployment": {"scale": scale}})


def test_ibs_engine_uses_window_offset_when_present():
    src = inspect.getsource(ibs_engine)
    assert 'window_offset = getattr(window_coefficients, "energy_offset_kj_mol", None)' in src
    assert "self.residual_energy_offset_kj_mol if window_offset is None else float(window_offset)" in src
