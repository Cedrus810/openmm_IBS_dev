"""
No-GPU control-flow test for the IBS non_mutating_v1 policy (round-1 P0-4).

Verifies, WITHOUT any OpenMM/GPU/MD, that under the non-mutating policy the
outer stage driver `_run_stage_with_overlap_autorepair`:

  1. runs the stage exactly ONCE (calls `run_once` a single time),
  2. never reaches ANY ensemble-mutating helper or fixed-H/calibration probe
     (each is patched to raise AssertionError if touched),
  3. returns the run_once result when the hard gates pass, and
  4. propagates the hard-gate exception unchanged when a gate fails
     (so the caller never marks the stage completed).

Run (on a node with the project env):
    conda run -n openmm_dev python -m pytest test_non_mutating_policy.py -v
"""
import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pytest

import abfe_pipeline as ap
import ibs_engine as ie


def _good_result():
    # A canned result whose values pass every hard gate in
    # _assert_stage_result_sane (min_overlap here = importance-ESS ratio).
    return {
        "total_delta_G": -12.34,
        "total_error": 0.42,
        "converged": True,
        "min_overlap": 0.20,
        "min_overlap_threshold": 0.05,
        "min_absolute_ess": 250.0,
        "min_absolute_ess_threshold": 50.0,
        "min_decorrelated_samples": 40,
        "min_decorrelated_samples_threshold": 20,
        "max_endpoint_uncertainty_kJ_mol": 0.4,
        "max_endpoint_uncertainty_kJ_mol_threshold": 1.0,
    }


def _make_fake_pipeline():
    """A minimal stand-in exposing only what the non-mutating early-return path
    touches: _log and _assert_stage_result_sane (the REAL hard-gate logic).
    Every mutating method is wired to explode if reached."""
    fake = types.SimpleNamespace()
    fake._log = lambda *a, **k: None
    fake._assert_stage_result_sane = types.MethodType(
        ap.ABFEPipeline._assert_stage_result_sane, fake
    )

    def _boom(name):
        def _f(*a, **k):
            raise AssertionError(f"mutator {name} reached under non_mutating_v1")
        return _f

    # method-level mutators that must never be reached
    for meth in (
        "_invalidate_stage_window_files",
        "_invalidate_single_window_production",
        "_diagnose_and_repair_all_pass_low_ess_window",
        "_apply_already_good_repairs",
        "_probe_vdw_window_fixed_overlap",
        "_remap_window_by_lambda_content",
    ):
        setattr(fake, meth, _boom(meth))
    return fake


def _patch_all_mutators(monkeypatch):
    def _boom(name):
        def _f(*a, **k):
            raise AssertionError(f"{name} called under non_mutating_v1")
        return _f

    # module-level outer mutators (imported into abfe_pipeline namespace)
    for fn in ("split_window_from_warmup_failure", "insert_lambda_from_overlap_failure"):
        monkeypatch.setattr(ap, fn, _boom(fn), raising=True)
    # inner fixed-H / asymmetric / calibration probes (ibs_engine namespace)
    for fn in (
        "probe_adjacent_path_overlap_bank",
        "detect_passed_but_asymmetric_overlap_bottleneck",
        "probe_adjacent_bias_calibration_bank",
    ):
        monkeypatch.setattr(ie, fn, _boom(fn), raising=False)


def _call(fake, run_once):
    return ap.ABFEPipeline._run_stage_with_overlap_autorepair(
        fake,
        "TestStage (vanishing)",
        "vanishing",
        "/nonexistent/preopt.json",
        4,
        [1.0, 0.9, 0.8, 0.0],
        [(0, 2), (2, 4)],
        run_once,
        protocol_key={},
    )


def test_non_mutating_runs_once_and_returns(monkeypatch):
    _patch_all_mutators(monkeypatch)
    calls = {"run_once": 0}
    good = _good_result()

    def run_once(n_states, lambdas, ranges, *a, **k):
        calls["run_once"] += 1
        return good

    fake = _make_fake_pipeline()
    result, n_states, lambdas, ranges = _call(fake, run_once)

    assert calls["run_once"] == 1, "stage must run exactly once (no repair loop)"
    assert result is good, "must return the single run_once result unchanged"
    assert n_states == 4


def test_non_mutating_hard_gate_failure_propagates(monkeypatch):
    _patch_all_mutators(monkeypatch)
    calls = {"run_once": 0}
    bad = _good_result()
    bad["min_overlap"] = 0.001  # below threshold -> importance-ESS hard gate fails

    def run_once(n_states, lambdas, ranges, *a, **k):
        calls["run_once"] += 1
        return bad

    fake = _make_fake_pipeline()
    with pytest.raises(RuntimeError):
        _call(fake, run_once)

    # ran once, surfaced the failure, and reached NO mutator (else the raise
    # would have been AssertionError, not the RuntimeError from the hard gate).
    assert calls["run_once"] == 1


class _RecordingContext:
    def __init__(self):
        self.set_calls = []

    def setParameter(self, name, value):
        self.set_calls.append((name, value))


def _write_ibs_state(path: Path, policy):
    state = {
        "n_states": 2,
        "prefix": "abfe",
        "f_k": [1.25, -1.25],
        "t": 3,
        "eta_penalty": 1.0,
        "e_offset": 0.0,
        "tmbar_history": [],
        "tmbar_history_dropped_entries": 0,
        "bias_converged": True,
        "bias_status": "converged",
        "frozen_f_k_pending": None,
        "frozen_validation_cumulative_steps": 0,
        "ibs_bias_protocol_version": ie.IBS_BIAS_PROTOCOL_VERSION,
        "lambdas_coul": [0.0, 0.0],
        "lambdas_vdw": [1.0, 0.5],
    }
    if policy is not None:
        state["sampling_repair_policy"] = policy
    path.write_text(json.dumps(state), encoding="utf-8")


def _bare_sampler(context):
    """Construct only the state used by load_ibs_state; no OpenMM System/GPU."""
    sampler = object.__new__(ie.IBSSampler)
    sampler.context = context
    sampler.n_states = 2
    sampler.prefix = "abfe"
    sampler.sampling_repair_policy = "non_mutating_v1"
    sampler.e_offset = 0.0
    sampler.bias_converged = False
    sampler.bias_status = "unconverged"
    sampler.frozen_f_k_pending = None
    sampler.frozen_validation_cumulative_steps = 0
    sampler.f_history = []
    sampler.tmbar_history = []
    sampler.tmbar_history_dropped_entries = 0
    sampler.eta_penalty = 1.0
    return sampler


@pytest.mark.parametrize("cached_policy", [None, "legacy_mutating"])
def test_load_ibs_state_policy_mismatch_raises_before_fk_injection(
    tmp_path, cached_policy
):
    state_path = tmp_path / "ibs_state.json"
    _write_ibs_state(state_path, cached_policy)
    before_bytes = state_path.read_bytes()
    before_mtime = state_path.stat().st_mtime_ns
    context = _RecordingContext()
    sampler = _bare_sampler(context)

    with pytest.raises(ie.ExistingEnsembleRequiresRescueAudit) as exc_info:
        sampler.load_ibs_state(
            str(state_path), lambdas_coul=[0.0, 0.0], lambdas_vdw=[1.0, 0.5]
        )

    assert context.set_calls == [], "policy rejection must precede f_k injection"
    assert state_path.read_bytes() == before_bytes
    assert state_path.stat().st_mtime_ns == before_mtime
    assert exc_info.value.diagnostics["cached_sampling_repair_policy"] == cached_policy
    assert exc_info.value.diagnostics["current_sampling_repair_policy"] == "non_mutating_v1"


def test_load_ibs_state_matching_policy_restores_normally(tmp_path):
    state_path = tmp_path / "ibs_state.json"
    _write_ibs_state(state_path, "non_mutating_v1")
    context = _RecordingContext()
    sampler = _bare_sampler(context)

    loaded = sampler.load_ibs_state(
        str(state_path), lambdas_coul=[0.0, 0.0], lambdas_vdw=[1.0, 0.5]
    )

    assert loaded is True
    assert context.set_calls == [("abfe_f_0", 1.25), ("abfe_f_1", -1.25)]
    assert sampler.bias_converged is True
    assert sampler.bias_status == "converged"


def test_load_pre_policy_protocol_state_is_rejected_before_policy_gate(tmp_path):
    """A v13 state cannot contain the v14 policy identity reliably.

    It must follow the ordinary protocol-version invalidation path instead of
    being misclassified as a current ensemble requiring a rescue audit. This
    is the rollout regression that previously stopped real resume runs after
    the expensive Boresch ramp.
    """
    state_path = tmp_path / "ibs_state_v13_without_policy.json"
    _write_ibs_state(state_path, None)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["ibs_bias_protocol_version"] = 13
    state_path.write_text(json.dumps(state), encoding="utf-8")
    before_bytes = state_path.read_bytes()
    before_mtime = state_path.stat().st_mtime_ns
    context = _RecordingContext()
    sampler = _bare_sampler(context)

    loaded = sampler.load_ibs_state(
        str(state_path), lambdas_coul=[0.0, 0.0], lambdas_vdw=[1.0, 0.5]
    )

    assert loaded is False
    assert context.set_calls == []
    assert state_path.read_bytes() == before_bytes
    assert state_path.stat().st_mtime_ns == before_mtime


def _snapshot_files(paths):
    return {
        str(path): (
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_mtime_ns,
        )
        for path in paths
    }


def test_old_energy_cache_policy_exception_propagates_without_build_or_write(
    tmp_path, monkeypatch
):
    """NM-01 regression: the structured rescue stop must escape the broad
    cache-load handler before any System/Context construction or re-sampling."""
    energies_path = tmp_path / "dual_window_0_vdw_energies.npy"
    convergence_path = tmp_path / "dual_window_0_vdw_convergence.json"
    np.save(energies_path, np.zeros((2, 3), dtype=np.float64))
    convergence_path.write_text(
        json.dumps(
            {
                "window_idx": 0,
                "stage_type": "vdw",
                "lambdas_coul": [0.0, 0.0],
                "lambdas_vdw": [1.0, 0.5],
                "wca_accounting_version": ie.WCA_ACCOUNTING_VERSION,
                "ibs_bias_protocol_version": ie.IBS_BIAS_PROTOCOL_VERSION,
                "lj_tail_lrc_protocol_version": ie.TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
                "sampling_repair_policy": "legacy_mutating",
                "n_steps_per_window_effective": 100,
                "early_stop_triggered": False,
            }
        ),
        encoding="utf-8",
    )
    before = _snapshot_files([energies_path, convergence_path])

    manager = object.__new__(ie.IBSWindowManagerDualLambda)
    manager.platform_name = "CPU"
    manager.topology = object()
    manager.system_template = object()
    manager.ranges = [(0, 2)]
    manager.lambdas_coul = np.asarray([0.0, 0.0])
    manager.lambdas_vdw = np.asarray([1.0, 0.5])
    manager.output_dir = str(tmp_path)
    manager.checkpoint_dir = str(tmp_path / "checkpoints")

    build_calls = []

    def _forbidden_build(*args, **kwargs):
        build_calls.append((args, kwargs))
        raise AssertionError("System/Context construction reached after rescue stop")

    manager._build_window_system = _forbidden_build
    monkeypatch.setattr(ie, "_build_platform_properties", lambda name: ("CPU", {}))
    monkeypatch.setattr(
        ie.openmm,
        "Platform",
        types.SimpleNamespace(getPlatformByName=lambda name: object()),
    )
    monkeypatch.setattr(
        ie, "_resolve_periodic_box_vectors", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(ie, "_system_requires_periodic_box", lambda system: False)

    with pytest.raises(ie.ExistingEnsembleRequiresRescueAudit) as exc_info:
        ie.IBSWindowManagerDualLambda.run_all_windows(
            manager,
            positions=None,
            box_vectors=None,
            n_steps_per_window=100,
            steps_per_update=10,
            stage_type="vdw",
            resume=True,
            repair_policy="non_mutating_v1",
        )

    assert build_calls == []
    assert exc_info.value.diagnostics["window_index"] == 0
    assert _snapshot_files([energies_path, convergence_path]) == before


def test_should_run_legacy_repair_enum_fail_closed():
    # The single decision point for the inner fixed-H/recalibration branch.
    assert ie.should_run_legacy_repair("non_mutating_v1") is False
    assert ie.should_run_legacy_repair("legacy_mutating") is True
    # ANY other value (typo, casing, empty, None) must fail closed — never
    # silently fall back to the mutating path.
    for bad in ("non_mutating", "NON_MUTATING_V1", "mutating", "",
                "non_mutating_v2", "legacy", None, 0):
        with pytest.raises(ValueError):
            ie.should_run_legacy_repair(bad)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
