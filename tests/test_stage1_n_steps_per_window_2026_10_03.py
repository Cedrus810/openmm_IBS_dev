"""`stage1_n_steps_per_window`: decharging REMD production length per state, set
independently of `n_steps_per_window` (which is also the stage-2 segment length).
Unset ⟹ old path unchanged; set ⟹ enters the decharging fingerprint only.
"""

import ast
import inspect
import textwrap

import pytest

import abfe_pipeline as ap
from runabfe import _path_evolution_kwargs

pytestmark = pytest.mark.cpu_only


class _C(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


def test_helper_falls_back_to_n_steps_per_window():
    assert ap._stage1_n_steps_per_window({}, 250000) == 250000
    assert ap._stage1_n_steps_per_window({"stage1_n_steps_per_window": None}, 250000) == 250000
    assert ap._stage1_n_steps_per_window({"stage1_n_steps_per_window": 0}, 250000) == 250000
    assert ap._stage1_n_steps_per_window({"stage1_n_steps_per_window": 1250000}, 250000) == 1250000


def test_config_key_reaches_kwargs_only_when_set():
    assert "stage1_n_steps_per_window" not in _path_evolution_kwargs(_C())
    out = _path_evolution_kwargs(_C({"stage1_n_steps_per_window": "1250000"}))
    assert out["stage1_n_steps_per_window"] == 1250000


def test_key_scoped_to_decharging_fingerprint():
    rc = {"kwargs": {"stage1_n_steps_per_window": 1250000, "warmup_steps": 1}}
    assert ap._scope_stage1_only_kwargs(rc, "decharging")["kwargs"] == rc["kwargs"]
    assert ap._scope_stage1_only_kwargs(rc, "vanishing")["kwargs"] == {"warmup_steps": 1}
    # unset ⟹ the run_config object passes through untouched (old fingerprints bit-identical)
    rc0 = {"kwargs": {"warmup_steps": 1}}
    assert ap._scope_stage1_only_kwargs(rc0, "vanishing") is rc0


def test_stage1_call_site_uses_helper():
    src = textwrap.dedent(inspect.getsource(ap.ABFEPipeline.run_full_pipeline))
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "_run_stage1_once")
    call = next(n for n in ast.walk(fn)
                if isinstance(n, ast.Call) and ast.unparse(n.func).endswith("_run_dual_lambda_stage"))
    kw = {k.arg: ast.unparse(k.value) for k in call.keywords}
    assert kw["n_steps_per_window"] == "_stage1_n_steps_per_window(kwargs, n_steps_per_window)"
