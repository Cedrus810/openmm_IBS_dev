"""CPU contracts derived from open issues #32, #75 and #76."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_environment_pins_or_bounds_pymbar_core():
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    dependency_lines = [
        line.strip()[1:].strip()
        for line in environment.splitlines()
        if re.match(r"^\s*-\s*pymbar-core(?:\s|[<>=!~])", line)
    ]
    assert len(dependency_lines) == 1
    requirement = dependency_lines[0]
    assert re.search(r"(?:==|=|>=|<=|~=|>|<)\s*\d", requirement), (
        "pymbar-core must be pinned or bounded so uncertainty semantics cannot "
        "change during an environment solve"
    )
    assert requirement == "pymbar-core=4.2.0"


def _lrc_helper():
    pytest.importorskip("openmm")
    pytest.importorskip("pymbar")
    from ibs_engine import (  # imported lazily to keep dependency test lightweight
        TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
        _lj_tail_lrc_coefficients_kj_mol,
    )

    assert TRADITIONAL_LJ_LRC_PROTOCOL_VERSION == 3
    return _lj_tail_lrc_coefficients_kj_mol


def test_lrc_v3_coefficients_are_finite_per_state_and_zero_at_zero_lambda():
    calculate = _lrc_helper()
    lambdas = np.array([0.0, 0.2, 0.7, 1.0])
    coeff = calculate(
        lambdas_vdw=lambdas,
        sigma_nm=np.array([0.28, 0.35]),
        s6_per_sigma_kj_nm6=np.array([0.12, 0.08]),
        s12_per_sigma_kj_nm12=np.array([0.004, 0.002]),
        alpha_lj=0.5,
        m_lj=2.0,
        n_lj=2.0,
    )

    assert coeff.shape == lambdas.shape
    assert np.all(np.isfinite(coeff))
    assert coeff[0] == 0.0


def test_lrc_v3_rejects_misaligned_sigma_resolved_moments():
    calculate = _lrc_helper()
    with pytest.raises(ValueError, match="长度必须一致"):
        calculate(
            lambdas_vdw=[0.0, 1.0],
            sigma_nm=[0.28, 0.35],
            s6_per_sigma_kj_nm6=[0.12],
            s12_per_sigma_kj_nm12=[0.004, 0.002],
            alpha_lj=0.5,
            m_lj=2.0,
            n_lj=2.0,
        )


def test_issue_75_has_behavioral_segment_regressions_not_only_source_checks():
    tests = (ROOT / "tests" / "test_warmup_overlap_protocol.py").read_text(encoding="utf-8")
    required_behavioral_tests = {
        "test_corrupted_checkpoint_keeps_frames_but_opens_new_segment",
        "test_decorrelate_per_segment_calls_autocorrelation_once_per_segment",
        "test_decorrelate_per_segment_excludes_segments_below_floor_without_calling_autocorrelation",
    }
    missing = sorted(name for name in required_behavioral_tests if f"def {name}(" not in tests)
    assert not missing, f"#75 behavioral coverage disappeared: {missing}"
