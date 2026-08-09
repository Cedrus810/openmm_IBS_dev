"""Policy and static-contract tests for the EXP-013 scheme-1 precheck."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/check_exp013_design1_mts_precheck.py"
SPEC = importlib.util.spec_from_file_location("exp013_design1_precheck", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

_N_VALUES = MODULE._N_VALUES
block_mean_and_sem = MODULE.block_mean_and_sem
evaluate_precheck_gate = MODULE.evaluate_precheck_gate
resolve_phase_lengths = MODULE.resolve_phase_lengths


def _healthy_gate_inputs(*, health_failed_n=None, shifted_n=None):
    health_failed_n = set() if health_failed_n is None else set(health_failed_n)
    shifted_n = set() if shifted_n is None else set(shifted_n)
    return (
        {n_value: True for n_value in _N_VALUES},
        {n_value: n_value not in health_failed_n for n_value in _N_VALUES},
        {n_value: n_value in shifted_n for n_value in _N_VALUES[1:]},
    )


def test_any_absolute_health_failure_blocks_n16():
    all_finite, health, shifts = _healthy_gate_inputs(health_failed_n={4})
    result = evaluate_precheck_gate("qualification", all_finite, health, shifts)

    assert result["all_absolute_health_passed"] is False
    assert result["qualification_passed"] is False
    assert result["eligible_for_n16_followup"] is False


def test_any_systematic_shift_blocks_n16():
    all_finite, health, shifts = _healthy_gate_inputs(shifted_n={8})
    result = evaluate_precheck_gate("qualification", all_finite, health, shifts)

    assert result["no_systematic_shift"] is False
    assert result["qualification_passed"] is False
    assert result["eligible_for_n16_followup"] is False


def test_only_qualification_with_all_healthy_and_no_shift_allows_n16():
    all_finite, health, shifts = _healthy_gate_inputs()
    smoke = evaluate_precheck_gate("smoke", all_finite, health, shifts)
    qualification = evaluate_precheck_gate("qualification", all_finite, health, shifts)

    assert smoke["smoke_passed"] is True
    assert smoke["eligible_for_n16_followup"] is False
    assert qualification["qualification_passed"] is True
    assert qualification["eligible_for_n16_followup"] is True


def test_phase_lengths_and_block_sem_are_frozen():
    assert resolve_phase_lengths("smoke") == (16, 32)
    assert resolve_phase_lengths("qualification") == (400, 2000)
    assert resolve_phase_lengths("qualification", 400, 2000) == (400, 2000)
    try:
        resolve_phase_lengths("qualification", 16, 32)
    except ValueError:
        pass
    else:
        raise AssertionError("qualification lengths must not be overridden")

    mean, sem, n_blocks = block_mean_and_sem([1.0, 1.0, 3.0, 3.0], 2)
    assert mean == 2.0
    assert sem == 1.0
    assert n_blocks == 2


def test_script_never_constructs_n16_n32_and_never_loads_checkpoint_into_mts():
    script_path = Path(__file__).resolve().parents[1] / "scripts/check_exp013_design1_mts_precheck.py"
    source = script_path.read_text(encoding="utf-8")
    assert "_N_VALUES = (1, 2, 4, 8)" in source
    assert '"n16_n32_run": False' in source

    tree = ast.parse(source)
    checkpoint_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "loadCheckpoint"
    ]
    assert len(checkpoint_calls) == 1

    mts_functions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_make_mts_simulation"
    ]
    assert len(mts_functions) == 1
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "loadCheckpoint"
        for node in ast.walk(mts_functions[0])
    )
