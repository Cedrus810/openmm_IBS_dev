"""`complex_stage2_n_steps_per_window`: complex-leg stage-2 segment length, set
independently of `n_steps_per_window` (which keeps driving the solvent leg).
Unset ⟹ old path unchanged; set ⟹ enters the complex vanishing fingerprint only,
so lengthening complex stage 2 reuses complex stage 1 and the solvent leg.
"""

import ast
import inspect
import textwrap

import pytest

import abfe_pipeline as ap
from runabfe import _complex_leg_only_kwargs, _path_evolution_kwargs

pytestmark = pytest.mark.cpu_only


class _C(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


def test_helper_falls_back_to_n_steps_per_window():
    assert ap._complex_stage2_n_steps_per_window({}, 250000) == 250000
    assert ap._complex_stage2_n_steps_per_window({"complex_stage2_n_steps_per_window": None}, 250000) == 250000
    assert ap._complex_stage2_n_steps_per_window({"complex_stage2_n_steps_per_window": 0}, 250000) == 250000
    assert ap._complex_stage2_n_steps_per_window({"complex_stage2_n_steps_per_window": 1000000}, 250000) == 1000000


def test_config_key_reaches_complex_kwargs_only_when_set():
    assert _complex_leg_only_kwargs(_C()) == {}
    out = _complex_leg_only_kwargs(_C({"complex_stage2_n_steps_per_window": "1000000"}))
    assert out == {"complex_stage2_n_steps_per_window": 1000000}
    # the shared helper (used by both legs) must not carry it to the solvent leg
    assert "complex_stage2_n_steps_per_window" not in _path_evolution_kwargs(
        _C({"complex_stage2_n_steps_per_window": 1000000}))


def test_solvent_call_does_not_get_complex_only_kwargs():
    import runabfe
    src = inspect.getsource(runabfe)
    tree = ast.parse(src)
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        kws = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}
        if kws.get("system_type") == "'solvent'":
            starred = [ast.unparse(k.value) for k in call.keywords if k.arg is None]
            assert not any("_complex_leg_only_kwargs" in s for s in starred)


def test_key_scoped_to_vanishing_fingerprint():
    rc = {"kwargs": {"complex_stage2_n_steps_per_window": 1000000, "stage1_n_steps_per_window": 750000,
                     "stage2_lcr_rounds": True, "warmup_steps": 1}}
    dech = ap._scope_stage2_lcr_switch(rc, "decharging")["kwargs"]
    van = ap._scope_stage2_lcr_switch(rc, "vanishing")["kwargs"]
    assert dech == {"stage1_n_steps_per_window": 750000, "warmup_steps": 1}
    assert van == {"complex_stage2_n_steps_per_window": 1000000, "stage2_lcr_rounds": True, "warmup_steps": 1}
    # unset ⟹ the run_config object passes through untouched (old fingerprints bit-identical)
    rc0 = {"kwargs": {"warmup_steps": 1}}
    assert ap._scope_stage2_only_kwargs(rc0, "decharging") is rc0


def test_stage2_block_reads_segment_length_from_helper():
    src = textwrap.dedent(inspect.getsource(ap.ABFEPipeline.run_full_pipeline))
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_stage2_once")
    call = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("_run_dual_lambda_stage"))
    kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert kw["n_steps_per_window"] == "_s2_steps_per_window"
    rounds = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("_run_stage2_rounds"))
    kw = {k.arg: ast.unparse(k.value) for k in rounds.keywords}
    assert kw["n_steps_per_window"] == "int(_s2_steps_per_window)"
