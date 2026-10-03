"""Small CPU reproductions for the 2026-10-03 accuracy review.

Uses the production evidence/decision functions and the existing synthetic
engine harness. No benchmark inputs, production trajectories or caches are
modified. Run in the project's openmm_dev environment from any directory.
These examples establish software behavior, not real-system error sizes.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import stage2_rounds as sr
import stage2_rounds_evidence as sre


def evidence_object(nums, *, merged=None):
    return sr.WindowEvidence(
        window=0, lambdas_vdw=[1.0, 0.5],
        state_ess=nums["state_ess"], top1pct_weight=nums["top1pct_weight"],
        sigma_mbar_kJ=nums["sigma_mbar_all_frames_kJ"],
        loo_delta_f_kJ=nums["loo_delta_f_kJ"], n_independent_seeds=3,
        merged_f_k_kJ=nums["merged_f_k_kJ"] if merged is None else merged,
        delta_f_finite=nums["delta_f_finite"],
        loo_min_state_ess=nums["loo_min_state_ess"],
    )


def sampling_gauge_reproduction():
    kt = 2.494
    rng = np.random.default_rng(20261003)
    sampling = []
    for _ in range(3):
        component = rng.integers(0, 2, size=400)
        x = rng.normal(component, 1.0)
        sampling.append(0.5 * kt * (x[None, :] - np.array([0., 1.])[:, None]) ** 2)
    fixed = [np.zeros(2)] * 3
    tail = np.array([0.0, 3.0])
    sampling_nums = sre.window_evidence_numbers(sampling, sampling, fixed, kt)
    target_nums = sre.window_evidence_numbers(
        sampling, [s + tail[:, None] for s in sampling], fixed, kt)
    ev = evidence_object(target_nums)
    center, source = sr._next_center(ev, trust_min_ess=sr.TRUST_MIN_ESS)
    difference = np.array(center) - np.array(sampling_nums["merged_f_k_kJ"])
    assert np.allclose(difference, tail, atol=1e-9, rtol=0.)
    return {
        "tail_constants_kJ": tail.tolist(),
        "next_center_source": source,
        "next_center_minus_sampling_free_energy_kJ": difference.tolist(),
        "relative_member_population_multiplier": math.exp(3. / kt),
        "interpretation": "Sampling allocation changes; this alone does not invalidate MBAR DeltaG.",
    }


def hidden_basin_reproduction():
    # Exact two-state, two-basin model: both states have U(A)=0;
    # U_0(B)=0 and U_1(B)=-8 kT. All observed trajectories remain in A.
    # Independent random velocities cannot establish visits to the unseen B.
    kt = 2.494
    sampling = [np.zeros((2, 400)) for _ in range(3)]
    nums = sre.window_evidence_numbers(sampling, sampling, [np.zeros(2)] * 3, kt)
    label, failures = sr.classify_window(evidence_object(nums))
    exact = -kt * (float(np.logaddexp(0., 8.)) - math.log(2.))
    assert label == sr.LABEL_CONVERGED
    assert abs(nums["delta_f_kJ"]) < 1e-9
    return {
        "label": label, "failed_conditions": failures,
        "reported_delta_f_kJ": nums["delta_f_kJ"],
        "exact_two_basin_delta_f_kJ": exact,
        "state_ess": nums["state_ess"],
        "jackknife_sigma_kJ": sr.jackknife_sigma(nums["loo_delta_f_kJ"]),
        "interpretation": "A coverage limitation of the convergence gate, not an MBAR algebra bug or benchmark error estimate.",
    }


def stale_ledger_reproduction():
    harness_path = ROOT / "tests/test_stage2_rounds_controller_2026_09_25.py"
    spec = importlib.util.spec_from_file_location("accuracy_review_lcr_harness", harness_path)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    with tempfile.TemporaryDirectory(prefix="abfe-accuracy-review-") as tmp:
        stub = harness._Stub(tmp)
        initial_engine = harness._FakeEngine(tmp)
        first = harness._run(stub, initial_engine, stage2_lcr_max_rounds=1)
        old_evidence = first["ledger"]["rounds"][0]["evidence"]
        changed_engine = harness._FakeEngine(tmp)
        second = stub._run_stage2_rounds(
            changed_engine, lambdas_var=harness.LAM, window_ranges=harness.RANGES,
            kt=harness.KT * 1.2,
            pilot_lambdas=harness.PILOT_L, pilot_mean_dU_dlambda=harness.PILOT_M,
            pilot_std_dU_dlambda=None, pilot_n_dU_dlambda_samples=None,
            n_steps_per_window=250000, run_kwargs={"stage2_lcr_max_rounds": 1})
        retained = old_evidence == second["ledger"]["rounds"][0]["evidence"]
        assert changed_engine.calls == [] and retained
        return {
            "old_kt_kJ": harness.KT, "new_kt_kJ": harness.KT * 1.2,
            "initial_sampling_calls": len(initial_engine.calls),
            "sampling_calls_after_temperature_change": len(changed_engine.calls),
            "old_evidence_reused": retained,
            "interpretation": "Execution/evidence layer reproduced with a fake sampler; no full MD or final-result cache experiment.",
        }


def missing_box_expectation_reproduction():
    fixture_path = ROOT / "tests/test_resume_reuse_contracts.py"
    spec = importlib.util.spec_from_file_location("accuracy_review_resume_fixtures", fixture_path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    import ibs_engine as ie
    stale = fixture._matching_conv(lj_tail_lrc_box_volume_nm3=323.16780367462405)
    common = (
        stale, fixture.GOOD_SHAPE, fixture.LC_WIN, fixture.LV_WIN,
        fixture.REPAIR_POLICY, fixture.LSE_TOL, False,
        dict(fixture.EARLY_STOP_CONFIG), fixture.TARGET_STEPS)
    missing = ie._resume_cached_window_gate_status(*common)
    supplied = ie._resume_cached_window_gate_status(*common, current_box_volume_nm3=308.)
    assert missing["usable"] and not supplied["usable"]
    return {
        "cached_volume_nm3": 323.16780367462405, "expected_volume_nm3": 308.,
        "usable_without_expected_volume": missing["usable"],
        "usable_with_expected_volume": supplied["usable"],
        "interpretation": "Lower-level gate reproduced; callers must supply the committed production volume.",
    }


if __name__ == "__main__":
    print(json.dumps({
        "sampling_gauge": sampling_gauge_reproduction(),
        "hidden_basin": hidden_basin_reproduction(),
        "stale_ledger": stale_ledger_reproduction(),
        "missing_box_expectation": missing_box_expectation_reproduction(),
    }, ensure_ascii=False, indent=2))
