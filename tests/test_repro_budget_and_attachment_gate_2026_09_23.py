"""复现模式两处"放开等于没放开"（2026-09-23）。"""
import inspect

import pytest

import abfe_core
import ibs_engine


def test_attachment_validator_honors_payload_gate_flag():
    payload = {"converged": False, "convergence_gates_enforced": False,
               "attachment_delta_G_kJ_mol": 1.0, "attachment_error_kJ_mol": 0.1}
    assert abfe_core.validate_boresch_attachment_result(payload) == (1.0, 0.1)
    with pytest.raises(abfe_core.FinalResultValidationError):
        abfe_core.validate_boresch_attachment_result({**payload, "convergence_gates_enforced": True})
    with pytest.raises(abfe_core.FinalResultValidationError):
        abfe_core.validate_boresch_attachment_result(
            {k: v for k, v in payload.items() if k != "convergence_gates_enforced"})


def test_repro_mode_does_not_inherit_warmup_ledger_across_segments():
    src = inspect.getsource(ibs_engine)
    assert ("_ledger = None if debug_provisional_on_warmup_failure else (\n"
            "                    inherit_warmup_ledger_across_segments(") in src
