"""Offline R1 D2 geometry and locality checks."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_gpu  # torch 为 GPU 版构建，CPU 上无意义，归 GPU 节点套件

torch = pytest.importorskip("torch")

from local_residual.softlift import build_softlift_model, primary_r1_config  # noqa: E402


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_exp020_softlift_d2.py"
_SPEC = importlib.util.spec_from_file_location("exp020_softlift_d2", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_c2_and_triclinic_pbc_checks_pass():
    assert _MODULE._check_c2_cutoff()["passed"] is True
    assert _MODULE._check_triclinic_pbc()["passed"] is True


def test_synthetic_r1_invariance_no_contact_and_nonparticipant_locality_pass():
    torch.manual_seed(0)
    config = primary_r1_config("0" * 64)
    model = build_softlift_model(config).double().eval()
    result = _MODULE._check_synthetic_invariance_and_locality(model, config)
    assert result["passed"] is True
    assert result["no_contact_energy_reduced"] == 0.0
    assert result["no_contact_force_max_abs_reduced_per_angstrom"] == 0.0
    assert result["nonparticipant_force_max_abs_reduced_per_angstrom"] == 0.0
