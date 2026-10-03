"""stage2 LCR relay (`stage2_lcr_relay`): each segment of a window starts from the
production end state of the window's previous sampled segment.

Reuses the fake-engine harness of test_stage2_rounds_controller_2026_09_25.py
(loaded by path: tests/ is not a package).
"""

import importlib.util
import json
import os

import numpy as np
import pytest
from openmm import unit

import abfe_pipeline as ap
import runabfe

pytestmark = pytest.mark.cpu_only

_spec = importlib.util.spec_from_file_location(
    "lcr_harness", os.path.join(os.path.dirname(__file__), "test_stage2_rounds_controller_2026_09_25.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)


class _RelayEngine(H._FakeEngine):
    """Accepts the relay kwargs, writes a final_state whose coordinates encode (segment, window)."""

    def __init__(self, tmp, **kw):
        super().__init__(tmp, **kw)
        self.starts = {}
        self.saved_flags = {}

    def __call__(self, n_states, lam, ranges, *, _window_start_positions=None,
                 _save_window_final_state=False, **kw):
        (w,) = kw["_only_window_indices"]
        seg = int(kw["_segment_seed_tag"].replace("seg", ""))
        start = None
        if _window_start_positions is not None:
            q = _window_start_positions[w]
            start = float(np.asarray(q.value_in_unit(unit.nanometer))[0, 0])
        self.starts[(seg, w)] = start
        self.saved_flags[(seg, w)] = _save_window_final_state
        res = super().__call__(n_states, lam, ranges, **kw)
        out = kw["_output_dir_override"] or os.path.join(self.tmp, "vanishing")
        if _save_window_final_state:
            np.savez(os.path.join(out, f"dual_window_{w}_vdw_final_state.npz"),
                     positions_nm=np.full((4, 3), 100.0 * seg + w), box_nm=np.eye(3) * 5.0)
        return res


def test_relay_chains_segments_within_each_window(tmp_path):
    stub = H._Stub(tmp_path)
    eng = _RelayEngine(tmp_path)
    H._run(stub, eng, stage2_lcr_relay=True)
    assert all(eng.saved_flags.values())
    for w in range(len(H.RANGES)):
        order = [seg for seg, ww in eng.calls if ww == w]
        assert eng.starts[(order[0], w)] is None            # first segment: pre-equilibration
        for prev, cur in zip(order, order[1:]):
            assert eng.starts[(cur, w)] == pytest.approx(100.0 * prev + w)   # previous segment's end
    with open(os.path.join(stub.checkpoint_dir, "stage2_rounds_ledger.json")) as fh:
        led = json.load(fh)
    starts = {(s["segment"], s["window"]): s.get("start_from") for r in led["rounds"] for s in r["specs"]}
    assert starts[(1, 0)] == "pre_equilibration"
    assert starts[(2, 0)] == "segment_1" and starts[(3, 0)] == "segment_2"


def test_relay_off_passes_no_relay_kwargs(tmp_path):
    stub = H._Stub(tmp_path)
    eng = H._FakeEngine(tmp_path)          # its signature would reject the relay kwargs
    H._run(stub, eng)
    assert eng.calls


def test_relay_switch_is_vanishing_identity_only_when_true():
    rc = {"kwargs": {"stage2_lcr_relay": True, "x": 1}}
    assert "stage2_lcr_relay" in ap._scope_stage2_lcr_switch(rc, "vanishing")["kwargs"]
    assert "stage2_lcr_relay" not in ap._scope_stage2_lcr_switch(rc, "decharging")["kwargs"]
    off = {"kwargs": {"stage2_lcr_relay": False, "x": 1}}
    assert "stage2_lcr_relay" not in ap._scope_stage2_lcr_switch(off, "vanishing")["kwargs"]


def test_runabfe_forwards_relay_only_when_true():
    class C(dict):
        def get(self, k, d=None):
            return dict.get(self, k, d)
    assert runabfe._path_evolution_kwargs(C(stage2_lcr_relay=True)).get("stage2_lcr_relay") is True
    assert "stage2_lcr_relay" not in runabfe._path_evolution_kwargs(C())
