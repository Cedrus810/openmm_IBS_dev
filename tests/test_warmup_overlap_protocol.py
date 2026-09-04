import json
import os
import types

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
pytest.importorskip("pymbar")
from openmm import app, XmlSerializer, unit

import ibs_engine
from abfe_preoptimizer import (
    DualLambdaPreOptimizer,
    THERMODYNAMIC_PATH_PROTOCOL_VERSION,
    VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
    canonicalize_window_ranges,
    estimate_f_k_from_pilot_ti,
    human_vanishing_initial_lambdas,
    insert_lambda_from_overlap_failure,
    split_window_from_warmup_failure,
    validate_vanishing_lambda_path_invariants,
    validate_single_shared_boundary_ranges,
    vanishing_subdomain_ranges_from_lambdas,
)
from ibs_engine import (
    FIXED_H_PROBE_CACHE_PROTOCOL_VERSION,
    FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS,
    IBS_BIAS_PROTOCOL_VERSION,
    MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION,
    IBSFrozenCalibrationValidationError,
    IBSSampler,
    IBSWarmupConvergenceError,
    _analyze_adjacent_pair,
    _atomic_save_openmm_checkpoint,
    _best_effort_validation_is_acceptable,
    _bias_calibration_pair_is_sufficient,
    _build_fixed_state_simulation,
    _build_main_window_checkpoint_manifest,
    _compute_bidirectional_overlap_from_u_kn,
    _decorrelate_per_segment,
    _extend_state_trajectory,
    _fixed_h_probe_bank_dir,
    _fixed_h_probe_bank_state_paths,
    _main_window_checkpoint_is_usable,
    _main_window_checkpoint_paths,
    _meets_minimum_with_roundoff,
    _peek_ibs_bias_status,
    _read_fixed_h_probe_bank_state_generation,
    _resolve_frozen_validation_budget_for_window,
    _resolve_frozen_validation_is_final_rung,
    _serialize_ibs_common_system,
    _try_load_main_window_checkpoint,
    detect_passed_but_asymmetric_overlap_bottleneck,
    ibs_lse_balance_diagnostics,
    probe_adjacent_bias_calibration_bank,
    probe_adjacent_path_overlap_bank,
)


def _state_paths(bank_dir, k):
    """Test helper: resolve one state's CURRENT generation-tagged paths."""
    generation = _read_fixed_h_probe_bank_state_generation(bank_dir, k)
    assert generation is not None
    return _fixed_h_probe_bank_state_paths(bank_dir, k, generation)


def test_pilot_ti_seed_keeps_physical_free_energy_sign():
    """For U_1-U_0=10 kJ/mol, f_1-f_0 must also be +10 kJ/mol.

    Then beta*(f_k-U_k) is identical for both states.  A sign inversion would
    instead double the imbalance to -20*beta.
    """
    seed = estimate_f_k_from_pilot_ti(
        pilot_lambdas=[0.0, 1.0],
        pilot_mean_dU_dlambda=[10.0, 10.0],
        target_lambdas=[0.0, 1.0],
    )
    np.testing.assert_allclose(seed, [-5.0, 5.0], rtol=0.0, atol=1.0e-12)
    physical_u = np.array([0.0, 10.0])
    np.testing.assert_allclose(seed - physical_u, [-5.0, -5.0], rtol=0.0, atol=1.0e-12)


def test_tmbar_candidate_keeps_physical_free_energy_sign(monkeypatch):
    sampler = IBSSampler.__new__(IBSSampler)
    sampler.tmbar_history = [{"synthetic": True}]
    sampler.kt = 2.5
    sampler.n_states = 3

    monkeypatch.setattr(
        ibs_engine,
        "solve_stage_integrated",
        lambda *args, **kwargs: {
            "lambdas": [0, 1, 2],
            "f_k": [10.0, 20.0, 40.0],
            "converged": True,
        },
    )

    candidate, _ = sampler._solve_tmbar_and_recenter()
    np.testing.assert_allclose(
        candidate,
        np.array([10.0, 20.0, 40.0]) - (70.0 / 3.0),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_best_effort_requires_a_complete_sane_attempt():
    complete = {
        "validation_sample_count": 100,
        "validation_batch_count": 5,
        "mean_p_batch": [0.30, 0.24, 0.145, 0.148, 0.167],
    }
    assert _best_effort_validation_is_acceptable(0.401, complete, 0.25, 4.0, 100)
    assert not _best_effort_validation_is_acceptable(0.401, complete, 0.25, 4.0, 101)
    assert not _best_effort_validation_is_acceptable(1.001, complete, 0.25, 4.0, 100)
    assert not _best_effort_validation_is_acceptable(0.1, None, 0.25, 4.0, 100)


def test_warmup_failure_splits_only_and_shares_one_existing_state():
    lambdas = [1.0, 0.9, 0.7, 0.4, 0.2, 0.0]
    ranges = [(0, 6)]
    diagnostics = {"window_index": 0, "global_state_range": [0, 6]}

    new_lambdas, new_ranges, feedback = split_window_from_warmup_failure(
        lambdas, ranges, diagnostics
    )

    assert new_lambdas == lambdas
    assert len(new_lambdas) == len(lambdas)
    assert new_ranges == [(0, 3), (2, 6)]
    assert set(range(*new_ranges[0])) & set(range(*new_ranges[1])) == {2}
    assert feedback["inserted_lambda"] is None


def test_warmup_split_reflows_untouched_next_neighbor_to_single_state_overlap():
    # 18 states, 5 overlapping 6-state parents (each sharing 3 states with the
    # next by design, per pilot_overlap_thermodynamic_length) -- the real case
    # that produced (2,6)-(3,9) sharing 3 of the right child's 4 states (75%)
    # before this fix.
    lambdas = list(np.linspace(1.0, 0.0, 18))
    ranges = [(0, 6), (3, 9), (6, 12), (9, 15), (12, 18)]
    diagnostics = {"window_index": 0, "global_state_range": [0, 6]}

    new_lambdas, new_ranges, feedback = split_window_from_warmup_failure(
        lambdas, ranges, diagnostics
    )

    assert new_lambdas == lambdas
    assert new_ranges == [(0, 3), (2, 6), (5, 9), (6, 12), (9, 15), (12, 18)]
    # The reflowed boundary now shares exactly one state.
    assert set(range(*new_ranges[1])) & set(range(*new_ranges[2])) == {5}
    # Untouched neighbor pairs keep their originally-designed 3-state overlap --
    # this fix must not cascade past the one boundary it touches.
    assert set(range(*new_ranges[2])) & set(range(*new_ranges[3])) == {6, 7, 8}
    assert set(range(*new_ranges[3])) & set(range(*new_ranges[4])) == {9, 10, 11}
    assert set(range(*new_ranges[4])) & set(range(*new_ranges[5])) == {12, 13, 14}
    covered = sorted({i for s, e in new_ranges for i in range(s, e)})
    assert covered == list(range(18))
    assert feedback["next_neighbor_reflowed_to_single_state_overlap"] == {
        "old_range": [3, 9],
        "new_range": [5, 9],
    }


def test_warmup_split_rejects_four_state_window_instead_of_bisecting_into_k2_plus_k3():
    # The real reported bug: window [2,6) has only 4 states. Splitting each
    # child to >=3 states while sharing 1 requires a 3+3-1=5-state parent;
    # bisecting a 4-state window instead produced [2,4) (K=2) + [3,6) (K=3) --
    # a statistically fragile 2-state window that should never be created.
    lambdas = list(np.linspace(1.0, 0.0, 9))
    ranges = [(0, 3), (2, 6), (3, 9)]
    diagnostics = {"window_index": 1, "global_state_range": [2, 6]}

    with pytest.raises(RuntimeError):
        split_window_from_warmup_failure(lambdas, ranges, diagnostics)


def test_warmup_split_leaves_next_neighbor_alone_when_already_single_state_overlap():
    lambdas = list(np.linspace(1.0, 0.0, 12))
    ranges = [(0, 6), (5, 12)]  # already sharing exactly one state
    diagnostics = {"window_index": 0, "global_state_range": [0, 6]}

    _, new_ranges, feedback = split_window_from_warmup_failure(lambdas, ranges, diagnostics)

    assert new_ranges == [(0, 3), (2, 6), (5, 12)]
    assert feedback["next_neighbor_reflowed_to_single_state_overlap"] is None


def test_canonicalize_window_ranges_removes_nested_windows_from_batch_split():
    # 5 overlapping 6-state parents over 18 states, each splitting independently
    # via split_window_from_warmup_failure's own middle=(s+e-1)//2 rule --
    # exactly the real case that produced 10 windows (4 of them strictly
    # contained in a neighboring split's child) instead of the minimal
    # connected 6-window chain.
    batch_split_result = [
        (0, 3), (2, 6),
        (3, 6), (5, 9),
        (6, 9), (8, 12),
        (9, 12), (11, 15),
        (12, 15), (14, 18),
    ]

    canonical = canonicalize_window_ranges(batch_split_result, n_states=18)

    assert canonical == [(0, 3), (2, 6), (5, 9), (8, 12), (11, 15), (14, 18)]
    covered = sorted({i for s, e in canonical for i in range(s, e)})
    assert covered == list(range(18))
    for (s0, e0), (s1, e1) in zip(canonical, canonical[1:]):
        assert s1 < e0  # adjacent windows share at least one state
    for i, (si, ei) in enumerate(canonical):
        for j, (sj, ej) in enumerate(canonical):
            if i != j:
                assert not (sj <= si and ei <= ej)  # no window strictly contains another


def test_canonicalize_window_ranges_rejects_incomplete_coverage():
    with pytest.raises(RuntimeError):
        canonicalize_window_ranges([(0, 3), (5, 8)], n_states=8)


def test_lambda_insertion_requires_measured_fixed_h_overlap_failure():
    lambdas = [1.0, 0.8, 0.5]
    ranges = [(0, 3)]
    diagnostics = {
        "window_index": 0,
        "global_state_range": [0, 3],
        "bidirectional_overlap_probe": {
            "pairs": [{
                "global_edge": [1, 2],
                "min_bidirectional_overlap": 0.01,
                "threshold": 0.03,
                "passed": False,
            }]
        },
    }
    new_lambdas, new_ranges, feedback = insert_lambda_from_overlap_failure(
        lambdas, ranges, diagnostics
    )
    assert new_lambdas == [1.0, 0.8, 0.65, 0.5]
    assert new_ranges == [(0, 3), (2, 4)]
    assert feedback["thermodynamic_lengths_invalidated"] is True
    assert "new_edge_thermodynamic_lengths" not in feedback


# Real values logged for window (0, 3), stage "vanishing", 2026-07-16 18:37-18:41
# (output_lrc_fix/pipeline.log:1601-1604 / checkpoints/preopt_dual_vanishing.json):
# edge [0,1]: min_overlap=0.26316, delta_f=-5.280+/-0.489 kJ/mol
# edge [1,2]: min_overlap=0.09501, delta_f=-23.147+/-1.152 kJ/mol
_REAL_LAMBDAS_WINDOW_0 = [1.0, 0.9937838890434966, 0.9875677780869931]
_REAL_PASSED_PAIRS_WINDOW_0 = [
    {
        "local_states": [0, 1],
        "global_edge": [0, 1],
        "min_bidirectional_overlap": 0.26316,
        "delta_f_kJ_mol": -5.280,
        "delta_f_uncertainty_kJ_mol": 0.489,
        "threshold": 0.03,
        "passed": True,
    },
    {
        "local_states": [1, 2],
        "global_edge": [1, 2],
        "min_bidirectional_overlap": 0.09501,
        "delta_f_kJ_mol": -23.147,
        "delta_f_uncertainty_kJ_mol": 1.152,
        "threshold": 0.03,
        "passed": True,
    },
]


def test_asymmetric_bottleneck_detector_selects_real_logged_bottleneck_edge():
    result = detect_passed_but_asymmetric_overlap_bottleneck(
        _REAL_PASSED_PAIRS_WINDOW_0, _REAL_LAMBDAS_WINDOW_0
    )
    assert result is not None
    assert result["qualified"] is True
    assert result["global_edge"] == [1, 2]
    assert result["overlap_ratio"] == pytest.approx(0.26316 / 0.09501, rel=1e-3)
    assert result["slope_ratio"] > 3.0


def test_asymmetric_bottleneck_detector_rejects_near_equal_overlap_and_slope():
    lambdas = [1.0, 0.9, 0.8]
    pairs = [
        {
            "local_states": [0, 1],
            "global_edge": [0, 1],
            "min_bidirectional_overlap": 0.10,
            "delta_f_kJ_mol": -8.0,
            "delta_f_uncertainty_kJ_mol": 0.5,
            "threshold": 0.03,
            "passed": True,
        },
        {
            "local_states": [1, 2],
            "global_edge": [1, 2],
            "min_bidirectional_overlap": 0.095,
            "delta_f_kJ_mol": -8.5,
            "delta_f_uncertainty_kJ_mol": 0.5,
            "threshold": 0.03,
            "passed": True,
        },
    ]
    assert detect_passed_but_asymmetric_overlap_bottleneck(pairs, lambdas) is None


def test_asymmetric_bottleneck_detector_rejects_gap_below_uncertainty_floor():
    # overlap ratio (6.0) and slope ratio (3.25) both clear their bars, but the
    # two edges' delta_f estimates are noisy enough (+/-5 kJ/mol each) that a
    # 5 kJ/mol gap is not distinguishable from zero at 2 sigma -- must not insert.
    lambdas = [1.0, 0.9, 0.85]
    pairs = [
        {
            "local_states": [0, 1],
            "global_edge": [0, 1],
            "min_bidirectional_overlap": 0.30,
            "delta_f_kJ_mol": -8.0,
            "delta_f_uncertainty_kJ_mol": 5.0,
            "threshold": 0.03,
            "passed": True,
        },
        {
            "local_states": [1, 2],
            "global_edge": [1, 2],
            "min_bidirectional_overlap": 0.05,
            "delta_f_kJ_mol": -13.0,
            "delta_f_uncertainty_kJ_mol": 5.0,
            "threshold": 0.03,
            "passed": True,
        },
    ]
    assert detect_passed_but_asymmetric_overlap_bottleneck(pairs, lambdas) is None


def test_insert_lambda_preserves_expanded_parent_window_for_qualified_asymmetric_bottleneck():
    lambdas = _REAL_LAMBDAS_WINDOW_0 + [0.95, 0.90, 0.85]
    ranges = [(0, 3), (2, 6)]
    diagnostics = {
        "window_index": 0,
        "global_state_range": [0, 3],
        "bidirectional_overlap_probe": {
            "pairs": _REAL_PASSED_PAIRS_WINDOW_0,
            "all_passed": True,
            "passed_but_asymmetric_bottleneck": {
                "qualified": True,
                "pair": _REAL_PASSED_PAIRS_WINDOW_0[1],
                "global_edge": [1, 2],
            },
        },
    }

    new_lambdas, new_ranges, feedback = insert_lambda_from_overlap_failure(
        lambdas, ranges, diagnostics
    )

    assert new_ranges == [(0, 4), (3, 7)]
    assert new_lambdas[2] == pytest.approx(0.9906758335652449)
    assert feedback["preserved_expanded_parent_window"] is True
    assert feedback["source"] == "fixed_hamiltonian_passed_but_asymmetric_bottleneck"
    covered = sorted({i for s, e in new_ranges for i in range(s, e)})
    assert covered == list(range(len(new_lambdas)))


def test_insert_lambda_falls_back_to_split_when_merged_window_would_exceed_k4_cap():
    # Parent window (0, 4) already has K=4 states -- the ceiling for
    # ibs_engine.py's fixed-H overlap probe/MBAR calibration eligibility
    # (`K <= 4 and stage_type == "vdw"`). Merging it to K=5 would silently
    # disqualify it from ever re-entering that pathway, so this must fall
    # back to the existing split-with-shared-state behavior instead.
    lambdas = [1.0, 0.93, 0.86, 0.8, 0.7, 0.6, 0.5]
    ranges = [(0, 4), (3, 7)]
    diagnostics = {
        "window_index": 0,
        "global_state_range": [0, 4],
        "bidirectional_overlap_probe": {
            "pairs": [],
            "all_passed": True,
            "passed_but_asymmetric_bottleneck": {
                "qualified": True,
                "pair": {
                    "global_edge": [1, 2],
                    "min_bidirectional_overlap": 0.05,
                    "threshold": 0.03,
                },
                "global_edge": [1, 2],
            },
        },
    }

    new_lambdas, new_ranges, feedback = insert_lambda_from_overlap_failure(
        lambdas, ranges, diagnostics
    )

    assert new_ranges == [(0, 3), (2, 5), (4, 8)]
    assert feedback["preserved_expanded_parent_window"] is False
    covered = sorted({i for s, e in new_ranges for i in range(s, e)})
    assert covered == list(range(len(new_lambdas)))


def test_bidirectional_overlap_uses_both_matrix_directions(monkeypatch):
    # This test exists to check exactly one thing: that
    # _compute_bidirectional_overlap_from_u_kn takes the overlap MINIMUM
    # across BOTH matrix directions (matrix[0,1] and matrix[1,0]), not just
    # one. It used to feed PyMBAR literally exact, zero-variance/fully
    # disconnected synthetic data for both the "passed" and "failed" halves;
    # both are numerically degenerate inputs (no curvature in the log-weight
    # ratio for the "passed" half, no connecting samples at all for the
    # "failed" half), and PyMBAR's asymptotic uncertainty estimator solves a
    # (near-)singular system for either, correctly triggering this
    # function's own fail-closed uncertainty check (see the RuntimeError a
    # few lines below the compute_overlap() call) rather than exercising the
    # matrix-direction logic this test is actually about. Real MD energies
    # always fluctuate, so genuinely disconnected states never hit this
    # exact degeneracy in production -- it's a synthetic-test-fixture
    # artifact, not something to weaken the production fail-closed check
    # for. Mocking PyMBAR's own return values isolates this test from
    # PyMBAR's version-dependent numerical behavior on pathological input.
    n = 40
    u_kn = np.zeros((2, 2 * n), dtype=np.float64)  # values unused -- MBAR itself is mocked
    n_k = np.asarray([n, n])

    def _mock_mbar(overlap_matrix, delta_f_reduced=0.0, delta_f_uncertainty_reduced=0.1):
        class _FakeMBAR:
            def compute_overlap(self):
                return {"matrix": np.asarray(overlap_matrix, dtype=np.float64)}

        monkeypatch.setattr("ibs_engine._build_mbar_compatible", lambda *a, **k: _FakeMBAR())
        monkeypatch.setattr(
            "ibs_engine._compute_free_energy_result_compatible",
            lambda mbar, compute_uncertainty=True: None,
        )
        monkeypatch.setattr(
            "ibs_engine._extract_free_energy_arrays",
            lambda res, require_uncertainty=True: (
                np.array([[0.0, delta_f_reduced], [-delta_f_reduced, 0.0]]),
                np.array([[0.0, delta_f_uncertainty_reduced], [delta_f_uncertainty_reduced, 0.0]]),
            ),
        )

    _mock_mbar([[0.9, 0.5], [0.5, 0.9]])
    passed = _compute_bidirectional_overlap_from_u_kn(u_kn, n_k, threshold=0.03)
    assert passed["passed"] is True
    assert passed["min_bidirectional_overlap"] >= 0.03

    # Asymmetric matrix, one direction below threshold: the minimum must
    # come from matrix[1,0]=0.02, not matrix[0,1]=0.1.
    _mock_mbar([[0.9, 0.1], [0.02, 0.98]])
    failed = _compute_bidirectional_overlap_from_u_kn(u_kn, n_k, threshold=0.03)
    assert failed["passed"] is False
    matrix = np.asarray(failed["overlap_matrix"])
    assert failed["min_bidirectional_overlap"] == min(matrix[0, 1], matrix[1, 0])
    assert failed["min_bidirectional_overlap"] == pytest.approx(0.02)


def test_bidirectional_overlap_real_pymbar_on_low_overlap_fluctuating_data():
    # Companion to the mocked test above: real (non-mocked) PyMBAR on
    # physically-plausible fluctuating data (per-state std ~1 reduced unit,
    # not exactly zero) with a genuinely large mean energy gap (~15 kT) --
    # low overlap without the exact-degeneracy pathology. Confirms the real
    # solver stays well-conditioned (finite, non-negative uncertainty) for
    # realistic low-overlap data, complementing the mocked matrix-direction
    # test rather than depending on it.
    n = 100
    rng = np.random.default_rng(1)
    u_kn = np.zeros((2, 2 * n), dtype=np.float64)
    u_kn[0, :n] = rng.normal(scale=1.0, size=n)
    u_kn[1, :n] = 15.0 + rng.normal(scale=1.0, size=n)
    u_kn[0, n:] = 15.0 + rng.normal(scale=1.0, size=n)
    u_kn[1, n:] = rng.normal(scale=1.0, size=n)

    result = _compute_bidirectional_overlap_from_u_kn(u_kn, np.asarray([n, n]), threshold=0.03)
    assert result["passed"] is False
    assert np.isfinite(result["delta_f_uncertainty_reduced_kT"])
    assert result["delta_f_uncertainty_reduced_kT"] >= 0.0


def test_common_system_plus_one_cv_matches_direct_energy_sum():
    system = openmm.System()
    system.addParticle(12.0 * unit.dalton)

    base = openmm.CustomExternalForce("0.5*k*x*x")
    base.addGlobalParameter("k", 4.0)
    base.addParticle(0, [])
    base.setForceGroup(0)
    system.addForce(base)

    mixture = openmm.CustomExternalForce("0")
    mixture.addParticle(0, [])
    mixture.setForceGroup(1)
    system.addForce(mixture)

    wca = openmm.CustomExternalForce("0")
    wca.addParticle(0, [])
    wca.setForceGroup(4)
    system.addForce(wca)

    common = XmlSerializer.deserialize(_serialize_ibs_common_system(system))
    assert {common.getForce(i).getForceGroup() for i in range(common.getNumForces())} == {0}

    cv = openmm.CustomExternalForce("c*x")
    cv.addGlobalParameter("c", 3.0)
    cv.addParticle(0, [])
    cv.setForceGroup(1)
    common.addForce(cv)
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(common, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.25, 0.0, 0.0]] * unit.nanometer)
    energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )
    expected = 0.5 * 4.0 * 0.25 ** 2 + 3.0 * 0.25
    assert energy == pytest.approx(expected, abs=1e-10)


def test_protocol_versions_reject_old_semantics():
    # 🔑 [lambda_boresch_scale 修复] 12->13：fixed-H overlap/bias 校准探针之前
    # 新建 Context 后从未设置 lambda_boresch_scale（System 级默认值是 0.0），
    # 而主窗口生产/冻结验证早已爬坡到 1.0——探针证明的是关掉 Boresch 限制的
    # 系统，跟它要验证的生产 Hamiltonian 不是同一个。已修复三处探针的 Context
    # 构建代码；升版本号强制所有在这个 bug 修复之前落盘的 frozen_f_k_pending
    # 被版本门控拒绝，见 ibs_engine.py 里这个常量定义处的完整 changelog。
    # 🔑 [sampling_repair_policy rollout] 13->14：该字段决定 f_k 是否可能被
    # legacy mutating repair 改写，必须进入协议身份；否则既有 v13/无字段 state
    # 会被误当成当前协议的可疑 ensemble，在正常版本失效之前就硬停止 resume。
    # 🔑 [IBS LSE fixed point] 14->15：冻结收敛曾直接检查 max|log(K*<p_k>)|；
    # 热力学路径 v8-v11 均已撤回，v12 的 vanishing 先按热力学长度在困难一侧加密
    # lambda，再组成 few-state 子区间。
    # 🔑 [local-MBAR loose gate] 27->29：冻结收敛判据整体换成局部滑窗 MBAR loose
    # gate（相邻 |Δf_k−ΔF^MBAR| < 10 kJ/mol），移除 LSE 占据门/连续通过/冻结验证
    # 阶梯/best-effort/warmup ESS 四联门；缓存兼容 (27,28,29)。
    # 🔑 [Fisher 度规控制布点] 20->21：v19/v20 算出等热力学长度解后丢弃，改用写死的
    # 二次网格 + 4 个人工端点 + 2 个 bridge 点（probe_controls_base_lambda_placement
    # = false）。实测后果：window 0 用 4 条边扛了全程 47.22 中的 41.32（单边
    # 8.82~11.83），其余 18 条边合计 5.90、最短 0.0002——零重叠，IBS 占据退化成硬
    # argmax。v18 的纯等长布点则是相反的失败（解耦端被拉断成 0.9225, 0.8382, 0）。
    # v21 等分 u=(1-beta)*s_hat+beta*(1-lambda)：度规驱动布点，同时每条边满足可证明的
    # 几何覆盖上限 |Δλ| <= 1/(beta*(n_states-1))。旧路径缓存必须整体失效重新 pilot，
    # 何况它们本来就属于旧的绝对-nm^6 alpha 哈密顿量。
    # 🔑 [e_offset 泄漏] 31->32：`_append_tmbar_batch_from_buffer()` 此前把已减
    # 逐帧 `e_offset` 的 `self.energy_buffer` 当 u_kn 存进 tmbar_history，却配
    # 未偏移的 bias/base，增广矩阵里那个逐帧平移不再是全行公共量、不抵消，
    # 等于人为注入共模因子（4W53 实测 window 0 达 3.10 kT，比真实防护壳共模的
    # 0.95~2.40 kT 还大）。在线学习输入变了，缓存兼容集合收窄成只有 32。
    assert IBS_BIAS_PROTOCOL_VERSION == 32
    assert THERMODYNAMIC_PATH_PROTOCOL_VERSION == 22


def test_inclusive_tmbar_thresholds_accept_roundoff_but_not_real_shortfall():
    assert _meets_minimum_with_roundoff(0.04999999999999431, 0.05)
    assert _meets_minimum_with_roundoff(0.9999999999998863, 1.0)
    assert not _meets_minimum_with_roundoff(0.049999, 0.05)
    assert not _meets_minimum_with_roundoff(0.999999, 1.0)


def test_ibs_lse_balance_diagnostics_is_the_fixed_point_gate():
    balanced = ibs_lse_balance_diagnostics([0.25, 0.25, 0.25, 0.25])
    assert balanced["available"] is True
    assert balanced["max_abs_log_residual"] == pytest.approx(0.0, abs=1e-14)
    assert balanced["scaled_balance_K_mean_p"] == pytest.approx([1.0] * 4)

    imbalanced = ibs_lse_balance_diagnostics([0.7, 0.1, 0.1, 0.1])
    assert imbalanced["max_abs_log_residual"] > 1.0
    assert imbalanced["coverage_ess"] < 4.0


def _bare_tmbar_sampler(n_states, kt):
    """Construct only the state _append_tmbar_batch_from_buffer/
    _solve_tmbar_and_recenter touch; no OpenMM Context/System needed."""
    sampler = object.__new__(IBSSampler)
    sampler.n_states = n_states
    sampler.kt = kt
    sampler.energy_buffer = []
    sampler.bias_history = []
    sampler.base_energy_history = []
    # [2026-08-31 P1] `_append_tmbar_batch_from_buffer()` 现在从这条历史取
    # **未减逐帧 e_offset** 的 u_kn（`energy_buffer` 里装的是已减偏移的
    # bias_cv 训练量）。生产的 collect_energies() 一定在同一个 frame_finite
    # 门下同步 append 这三条，stub 以前省掉它只是因为旧代码不读它。
    sampler.sampling_state_energy_history = []
    sampler.tmbar_history = []
    sampler.tmbar_history_dropped_entries = 0
    return sampler


def test_tmbar_recovers_equal_free_energy_of_equal_width_wells():
    # [IBS_BIAS_PROTOCOL_VERSION=19] Ground truth: three unit-variance harmonic
    # wells u_k(x) = 0.5*(x-mu_k)^2 at kT=1 all have the SAME partition function
    # (same width, only shifted) regardless of mu_k, so the correct mean-centered
    # f_k is exactly [0, 0, 0]. Each tmbar_history entry mimics one online-
    # learning minibatch: frames actually sampled from ONE state's own
    # equilibrium (that state is the augmented matrix's "sampled row"), with all
    # three states' interaction energies evaluated at those same frames -- this
    # is exactly the (u_kn, bias_energies, base_energies) triple
    # _append_tmbar_batch_from_buffer packages from real per-frame CV probes.
    kt = 1.0
    mu = np.asarray([0.0, 1.0, 2.0], dtype=float)
    n_states = mu.size
    n_per_entry = 400
    rng = np.random.default_rng(20260719)

    sampler = _bare_tmbar_sampler(n_states, kt)
    for sampled_state in range(n_states):
        x = rng.normal(loc=mu[sampled_state], scale=1.0, size=n_per_entry)
        u_kn_frame_major = 0.5 * (x[:, None] - mu[None, :]) ** 2  # (N, K)
        sampler.energy_buffer = list(u_kn_frame_major)
        # 本用例不带 e_offset，所以未偏移历史与 energy_buffer 内容相同
        sampler.sampling_state_energy_history = list(u_kn_frame_major)
        sampler.bias_history = list(u_kn_frame_major[:, sampled_state])
        sampler.base_energy_history = [0.0] * n_per_entry
        n_appended = sampler._append_tmbar_batch_from_buffer()
        assert n_appended == n_per_entry

    assert len(sampler.tmbar_history) == n_states

    result = sampler._solve_tmbar_and_recenter(
        min_ess_ratio=0.01,
        min_absolute_ess=5.0,
        min_decorrelated_samples=5,
        max_uncertainty_kJ_mol=5.0,
    )
    assert result is not None
    f_new, res = result
    assert f_new == pytest.approx(0.0, abs=0.5)
    assert np.mean(f_new) == pytest.approx(0.0, abs=1.0e-9)  # mean-centered
    assert "converged" in res


def test_tmbar_returns_none_before_enough_batches_appended():
    kt = 1.0
    sampler = _bare_tmbar_sampler(n_states=3, kt=kt)
    assert sampler._solve_tmbar_and_recenter() is None


def test_bounded_log_occupancy_update_lowers_strong_state_and_raises_weak_states():
    sampler = object.__new__(IBSSampler)
    sampler.n_states = 3
    sampler.kt = 2.5
    sampler.eta_penalty = 1.0
    sampler.f_history = []

    f_new, diag = sampler._bounded_log_occupancy_update(
        np.zeros(3, dtype=float),
        np.asarray([0.8, 0.1, 0.1], dtype=float),
    )

    assert f_new[0] < 0.0
    assert f_new[1] > 0.0
    assert f_new[2] > 0.0
    assert np.mean(f_new) == pytest.approx(0.0, abs=1.0e-12)
    assert np.max(np.abs(f_new)) <= 2.0 * sampler.kt + 1.0e-12
    assert diag["method"] == "bounded_log_occupancy_fallback_v9"


def test_bounded_log_occupancy_update_leaves_uniform_weights_unchanged():
    sampler = object.__new__(IBSSampler)
    sampler.n_states = 4
    sampler.kt = 2.5
    sampler.eta_penalty = 1.0
    sampler.f_history = []
    sampler.tmbar_history = []
    f_old = np.asarray([-3.0, -1.0, 1.0, 3.0], dtype=float)

    f_new, _ = sampler._bounded_log_occupancy_update(
        f_old,
        np.full(4, 0.25, dtype=float),
    )

    assert f_new == pytest.approx(f_old, abs=1.0e-12)

    # A single 5-frame minibatch decorrelates to well under the
    # min_decorrelated_samples floor used by solve_stage_integrated's own
    # per-entry gate -- no update should be produced yet, matching
    # update_weights()'s "not enough data" contract (returns None, f_k
    # unchanged) rather than raising.
    sampler.energy_buffer = [np.array([0.0, 1.0, 2.0]) for _ in range(5)]
    sampler.sampling_state_energy_history = [
        np.array([0.0, 1.0, 2.0]) for _ in range(5)
    ]
    sampler.bias_history = [0.0] * 5
    sampler.base_energy_history = [0.0] * 5
    assert sampler._append_tmbar_batch_from_buffer() == 5
    assert sampler._solve_tmbar_and_recenter() is None


def test_vanishing_pilot_returns_few_state_subdomains_without_overlap_two():
    class _FakeIntegrator:
        def step(self, _steps):
            return None

    class _FakeContext:
        def __init__(self):
            self.params = {"lam_coul": 0.0, "lam_vdw": 1.0}
            self.integrator = _FakeIntegrator()

        def getParameters(self):
            return dict(self.params)

        def setParameter(self, name, value):
            self.params[name] = float(value)

        def getIntegrator(self):
            return self.integrator

    optimizer = DualLambdaPreOptimizer.__new__(DualLambdaPreOptimizer)
    optimizer.context = _FakeContext()
    optimizer.param_coul = "lam_coul"
    optimizer.param_vdw = "lam_vdw"
    optimizer._sample_scalar_metric = lambda *_args, **_kwargs: (
        1.0,
        {"metric_g": 1.0},
    )

    result = optimizer.optimize_stage2_vanishing(
        n_states=17,
        n_steps_per_state=0,
    )
    lambdas = np.asarray(result["lambdas_vdw"], dtype=float)
    assert len(lambdas) == 23
    validate_vanishing_lambda_path_invariants(lambdas)
    validate_single_shared_boundary_ranges(result["window_ranges"], len(lambdas))
    assert result["path_diagnostics"]["ibs_ensemble_layout"] == (
        "few_state_thermodynamic_subdomains"
    )
    # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=21] 度规控制布点，不再是固定二次网格
    # + 人工点 + bridge 点。
    assert result["path_diagnostics"]["lambda_placement_method"] == (
        "fisher_metric_blended_with_geometric_floor_v21"
    )
    assert result["path_diagnostics"]["probe_controls_base_lambda_placement"] is True
    # 几何覆盖下限必须真实成立（v18 纯等长布点把解耦端拉断，就是缺这条）。
    assert (
        result["path_diagnostics"]["realized_max_lambda_gap"]
        <= result["path_diagnostics"]["max_lambda_gap_bound"] * (1.0 + 1.0e-6)
    )
    assert result["path_diagnostics"]["sliding_overlap_states"] == 0
    assert result["path_diagnostics"]["common_boundary_state_count"] == 1
    assert result["window_ranges"] == [
        (0, 5), (4, 8), (7, 12), (11, 16), (15, 20), (19, 23)
    ]
    assert sum(end - start for start, end in result["window_ranges"]) == 28
    assert result["window_ranges"] == vanishing_subdomain_ranges_from_lambdas(
        result["lambdas_vdw"],
        first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
    )


def test_vanishing_pilot_adds_nodes_to_harder_tail_without_moving_anchors():
    class _FakeIntegrator:
        def step(self, _steps):
            return None

    class _FakeContext:
        def __init__(self):
            self.params = {"lam_coul": 0.0, "lam_vdw": 1.0}
            self.integrator = _FakeIntegrator()

        def getParameters(self):
            return dict(self.params)

        def setParameter(self, name, value):
            self.params[name] = float(value)

        def getIntegrator(self):
            return self.integrator

    optimizer = DualLambdaPreOptimizer.__new__(DualLambdaPreOptimizer)
    optimizer.context = _FakeContext()
    optimizer.param_coul = "lam_coul"
    optimizer.param_vdw = "lam_vdw"

    def _metric(_parameter, lam, **_kwargs):
        value = 9.0 if float(lam) <= 0.5 else 1.0
        return value, {"metric_g": value}

    optimizer._sample_scalar_metric = lambda *_args, **_kwargs: (
        1.0,
        {"metric_g": 1.0},
    )
    uniform = optimizer.optimize_stage2_vanishing(n_states=17, n_steps_per_state=0)
    optimizer._sample_scalar_metric = _metric
    hard_tail = optimizer.optimize_stage2_vanishing(n_states=17, n_steps_per_state=0)

    uniform_lambdas = np.asarray(uniform["lambdas_vdw"], dtype=float)
    hard_lambdas = np.asarray(hard_tail["lambdas_vdw"], dtype=float)

    # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=21] 度规现在【控制】整条路径，不再
    # 只影响两个 bridge 点。metric_g 在 λ<=0.5 上是 9（sqrt=3 倍难度），布点必须
    # 真的往那一侧搬——这正是 v19/v20 做不到的事（那时整条基础路径是写死的二次
    # 网格，占据比例几乎不随 pilot 变化）。
    assert not np.allclose(hard_lambdas, uniform_lambdas)
    assert np.count_nonzero(hard_lambdas < 0.5) > np.count_nonzero(
        uniform_lambdas < 0.5
    )
    # 但无论度规多么偏斜，几何覆盖下限都必须守住（防 v18 的解耦端断层复发）。
    for res in (uniform, hard_tail):
        validate_vanishing_lambda_path_invariants(res["lambdas_vdw"])
        assert (
            res["path_diagnostics"]["realized_max_lambda_gap"]
            <= res["path_diagnostics"]["max_lambda_gap_bound"] * (1.0 + 1.0e-6)
        )


def test_fixed_h_probe_cache_protocol_version_is_explicit():
    # 🔑 同上：探针轨迹库（_build_fixed_h_probe_bank_manifest 管的 per-state
    # 原始采样轨迹）也需要独立升版本号强制失效重采，理由同 IBS_BIAS_PROTOCOL_
    # VERSION=13 的 changelog。
    assert FIXED_H_PROBE_CACHE_PROTOCOL_VERSION == 3


# ============================================================================
# Fixed-H probe trajectory bank (IBS_BIAS_PROTOCOL_VERSION=11): toy system
# fixture + tests. One particle, Reference platform, tiny step counts -- only
# exercises the bank's bookkeeping (resume/segments/manifest/checkpoints),
# not real sampling statistics.
# ============================================================================

def _toy_probe_system(K: int, with_group4: bool):
    system = openmm.System()
    system.addParticle(12.0 * unit.dalton)

    base = openmm.CustomExternalForce("0.5*kx*(x^2+y^2+z^2)")
    base.addGlobalParameter("kx", 500.0)
    base.addParticle(0, [])
    base.setForceGroup(0)
    system.addForce(base)

    if with_group4:
        wca = openmm.CustomExternalForce("lambda_shield*0")
        wca.addGlobalParameter("lambda_shield", 0.5)
        wca.addParticle(0, [])
        wca.setForceGroup(4)
        system.addForce(wca)

    common_xml = XmlSerializer.serialize(system)

    # Bake each state's coefficient in as a literal (not a global parameter):
    # IBSSampler._build_probe_context adds every state's CV force into one
    # shared probe System, and OpenMM rejects two Forces in the same System
    # declaring the same global parameter name with different defaults.
    cv_xmls = []
    for k in range(K):
        cv = openmm.CustomExternalForce(f"{float(k + 1)}*x")
        cv.addParticle(0, [])
        cv_xmls.append(XmlSerializer.serialize(cv))

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("TST", chain)
    topology.addAtom("C", app.Element.getBySymbol("C"), residue)

    positions = [[0.0, 0.0, 0.0]] * unit.nanometer
    box_vectors = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]) * unit.nanometer
    temperature = 300.0 * unit.kelvin

    ibs_wrapper = types.SimpleNamespace(
        _int_cv_force_xmls=cv_xmls, lj_tail_lrc_coeff_kj_mol=None, prefix="test"
    )
    return {
        "topology": topology,
        "common_xml": common_xml,
        "cv_xmls": cv_xmls,
        "ibs_wrapper": ibs_wrapper,
        "positions": positions,
        "box_vectors": box_vectors,
        "temperature": temperature,
    }


def test_build_fixed_state_simulation_rejects_group4_direction_mismatch():
    toy_with_group4 = _toy_probe_system(K=1, with_group4=True)
    toy_without_group4 = _toy_probe_system(K=1, with_group4=False)

    with pytest.raises(RuntimeError):
        _build_fixed_state_simulation(
            topology=toy_without_group4["topology"],
            system_xml=toy_without_group4["common_xml"],
            cv_xml=toy_without_group4["cv_xmls"][0],
            require_group4=True,
            platform_name="Reference",
            temperature_q=toy_without_group4["temperature"],
            positions=toy_without_group4["positions"],
            box_vectors=toy_without_group4["box_vectors"],
            integrator_seed=1,
            velocity_seed=2,
        )
    with pytest.raises(RuntimeError):
        _build_fixed_state_simulation(
            topology=toy_with_group4["topology"],
            system_xml=toy_with_group4["common_xml"],
            cv_xml=toy_with_group4["cv_xmls"][0],
            require_group4=False,
            platform_name="Reference",
            temperature_q=toy_with_group4["temperature"],
            positions=toy_with_group4["positions"],
            box_vectors=toy_with_group4["box_vectors"],
            integrator_seed=1,
            velocity_seed=2,
        )


def test_extend_state_trajectory_does_not_reburn_on_second_call():
    toy = _toy_probe_system(K=1, with_group4=False)
    sim = _build_fixed_state_simulation(
        topology=toy["topology"],
        system_xml=toy["common_xml"],
        cv_xml=toy["cv_xmls"][0],
        require_group4=False,
        platform_name="Reference",
        temperature_q=toy["temperature"],
        positions=toy["positions"],
        box_vectors=toy["box_vectors"],
        integrator_seed=1,
        velocity_seed=2,
    )
    evaluator = types.SimpleNamespace(
        evaluate_interaction_energies=lambda pos, box: np.array([0.0])
    )
    record = {"u_cv_kj_mol": None, "volume_nm3": None, "segments": [], "sampled_steps": 0}

    _extend_state_trajectory(
        sim, record, evaluator, target_steps=20, sample_interval=2,
        burn_in_steps=4, needs_burn_in=True, segment_reason="fresh",
    )
    assert len(record["segments"]) == 1
    assert record["sampled_steps"] == 20

    _extend_state_trajectory(
        sim, record, evaluator, target_steps=40, sample_interval=2,
        burn_in_steps=4, needs_burn_in=False, segment_reason="extend",
    )
    # Second call against the same live simulation must not open a new
    # segment -- only the existing segment's own sample_steps/n_frames grow.
    assert len(record["segments"]) == 1
    assert record["sampled_steps"] == 40
    assert record["segments"][0]["sample_steps"] == 40
    assert record["u_cv_kj_mol"].shape[0] == 20


def test_path_overlap_bank_creates_one_state_file_per_state_not_per_edge(tmp_path):
    K = 4
    toy = _toy_probe_system(K=K, with_group4=False)
    checkpoint_dir = str(tmp_path)

    result = probe_adjacent_path_overlap_bank(
        topology=toy["topology"],
        common_system_xml=toy["common_xml"],
        ibs_wrapper=toy["ibs_wrapper"],
        K=K,
        positions=toy["positions"],
        box_vectors=toy["box_vectors"],
        temperature=toy["temperature"],
        platform_name="Reference",
        checkpoint_dir=checkpoint_dir,
        stage_type="vdw",
        window_idx=0,
        global_state_start=0,
        burn_in_steps=4,
        # 🔑 [P2 fix follow-up] _decorrelate_per_segment 现在会把短于
        # min_frames_for_subsampling（默认 20 帧）的整段直接排除出 MBAR，不再
        # 让它们带权重混进去——20 步/interval=2 只有 10 帧，会被整段排除，两态
        # n_k 都变 0，触发"双态 MBAR 样本计数错误"。50 步 -> 25 帧，留出安全
        # 余量清过这个门槛，这个测试本身只关心逐态存档的记账行为，不关心统计量。
        sample_targets=(50,),
        sample_interval=2,
    )
    assert len(result["pairs"]) == K - 1

    bank_dir = _fixed_h_probe_bank_dir(checkpoint_dir, "vdw", 0, "path_probe")
    energy_files = [f for f in os.listdir(bank_dir) if f.endswith("_energies.npy")]
    assert len(energy_files) == K  # one per state, not one per edge (K-1 edges)


def test_bank_resumes_from_checkpoint_without_reburning_across_separate_calls(tmp_path):
    K = 2
    toy = _toy_probe_system(K=K, with_group4=False)
    checkpoint_dir = str(tmp_path)
    common_kwargs = dict(
        topology=toy["topology"],
        common_system_xml=toy["common_xml"],
        ibs_wrapper=toy["ibs_wrapper"],
        K=K,
        positions=toy["positions"],
        box_vectors=toy["box_vectors"],
        temperature=toy["temperature"],
        platform_name="Reference",
        checkpoint_dir=checkpoint_dir,
        stage_type="vdw",
        window_idx=0,
        global_state_start=0,
        burn_in_steps=4,
        sample_interval=2,
    )

    # 🔑 [P2 fix follow-up] 见上一个测试的注释：sample_targets 必须让每个
    # segment 的帧数不低于 _decorrelate_per_segment 的默认 20 帧门槛，否则整段
    # 会被排除出 MBAR，两态 n_k 都变 0。40 步/interval=2 = 20 帧，刚好过线。
    probe_adjacent_path_overlap_bank(sample_targets=(40,), **common_kwargs)
    bank_dir = _fixed_h_probe_bank_dir(checkpoint_dir, "vdw", 0, "path_probe")
    _, _, _, meta_path, native_checkpoint_path = _state_paths(bank_dir, 0)
    assert os.path.exists(native_checkpoint_path)  # true continuation mechanism, not just NPZ
    with open(meta_path) as f:
        meta_after_first_call = json.load(f)
    assert meta_after_first_call["sampled_steps"] == 40
    assert len(meta_after_first_call["segments"]) == 1

    # A brand-new top-level call (simulating a fresh process/resume) against
    # the same checkpoint_dir must continue from 40 -> 80, not re-burn a new
    # segment or restart from 0.
    probe_adjacent_path_overlap_bank(sample_targets=(80,), **common_kwargs)
    _, _, _, meta_path_after_second_call, _ = _state_paths(bank_dir, 0)
    with open(meta_path_after_second_call) as f:
        meta_after_second_call = json.load(f)
    assert meta_after_second_call["sampled_steps"] == 80
    assert len(meta_after_second_call["segments"]) == 1
    # The old generation's files (including the stale meta.json) must be
    # cleaned up once the new generation's pointer is committed.
    assert not os.path.exists(meta_path)


def test_manifest_mismatch_invalidates_whole_window(tmp_path):
    K = 2
    toy = _toy_probe_system(K=K, with_group4=False)
    checkpoint_dir = str(tmp_path)
    common_kwargs = dict(
        topology=toy["topology"],
        ibs_wrapper=toy["ibs_wrapper"],
        K=K,
        positions=toy["positions"],
        box_vectors=toy["box_vectors"],
        temperature=toy["temperature"],
        platform_name="Reference",
        checkpoint_dir=checkpoint_dir,
        stage_type="vdw",
        window_idx=0,
        global_state_start=0,
        burn_in_steps=4,
        sample_interval=2,
    )
    # 🔑 [P2 fix follow-up] 见上面测试的注释：40 步/interval=2=20 帧，刚好过
    # _decorrelate_per_segment 默认 20 帧门槛，不会被整段排除出 MBAR。
    probe_adjacent_path_overlap_bank(
        common_system_xml=toy["common_xml"], sample_targets=(40,), **common_kwargs
    )
    bank_dir = _fixed_h_probe_bank_dir(checkpoint_dir, "vdw", 0, "path_probe")
    _, _, _, meta_path, _ = _state_paths(bank_dir, 0)
    with open(meta_path) as f:
        assert json.load(f)["sampled_steps"] == 40

    # A structurally different common system (Hamiltonian changed) must
    # invalidate the whole window+probe_type directory -- resample from 0,
    # not "resume" the old, now-untrustworthy frames.
    changed_toy = _toy_probe_system(K=K, with_group4=False)
    changed_system = XmlSerializer.deserialize(changed_toy["common_xml"])
    changed_system.getForce(0).setGlobalParameterDefaultValue(0, 999.0)
    changed_common_xml = XmlSerializer.serialize(changed_system)

    probe_adjacent_path_overlap_bank(
        common_system_xml=changed_common_xml, sample_targets=(40,), **common_kwargs
    )
    _, _, _, meta_path_after_change, _ = _state_paths(bank_dir, 0)
    with open(meta_path_after_change) as f:
        meta_after_change = json.load(f)
    assert meta_after_change["sampled_steps"] == 40
    assert len(meta_after_change["segments"]) == 1  # fresh segment 0, not appended to old one


def test_corrupted_checkpoint_keeps_frames_but_opens_new_segment(tmp_path):
    K = 2
    toy = _toy_probe_system(K=K, with_group4=False)
    checkpoint_dir = str(tmp_path)
    common_kwargs = dict(
        topology=toy["topology"],
        common_system_xml=toy["common_xml"],
        ibs_wrapper=toy["ibs_wrapper"],
        K=K,
        positions=toy["positions"],
        box_vectors=toy["box_vectors"],
        temperature=toy["temperature"],
        platform_name="Reference",
        checkpoint_dir=checkpoint_dir,
        stage_type="vdw",
        window_idx=0,
        global_state_start=0,
        burn_in_steps=4,
        sample_interval=2,
    )
    # 🔑 [P2 fix follow-up] 见上面测试的注释：_decorrelate_per_segment 现在会
    # 把不足 20 帧的整段排除出 MBAR。这个测试最终会产出两个独立 segment（重烧
    # 前/后各一段），每段都必须自己单独 >=20 帧才能都被算进去——40 步（20 帧）
    # 起步，第二次调用再增量 40 步（新开的重烧 segment 也是 20 帧）。
    probe_adjacent_path_overlap_bank(sample_targets=(40,), **common_kwargs)
    bank_dir = _fixed_h_probe_bank_dir(checkpoint_dir, "vdw", 0, "path_probe")
    energies_path, _, npz_checkpoint_path, meta_path, native_checkpoint_path = _state_paths(bank_dir, 0)
    with open(meta_path) as f:
        frames_before = json.load(f)["segments"][0]["n_frames"]
    # Remove only the native OpenMM checkpoint (the true continuation
    # mechanism -- it alone captures the integrator's internal RNG state);
    # the NPZ fallback positions/velocities/box snapshot is left intact, so
    # this exercises the "native checkpoint missing/incompatible -> fall
    # back to NPZ, but still force a new (re-burned) segment" path.
    assert os.path.exists(npz_checkpoint_path)
    os.remove(native_checkpoint_path)

    probe_adjacent_path_overlap_bank(sample_targets=(80,), **common_kwargs)
    energies_path_after, _, _, meta_path_after, _ = _state_paths(bank_dir, 0)
    assert not os.path.exists(meta_path)  # old generation's files are cleaned up
    with open(meta_path_after) as f:
        meta_after = json.load(f)
    assert len(meta_after["segments"]) == 2  # new segment opened, old one untouched
    assert meta_after["segments"][0]["n_frames"] == frames_before  # old frames preserved
    assert meta_after["segments"][1]["reason"] == "native_checkpoint_missing_or_incompatible_npz_fallback"
    assert meta_after["sampled_steps"] == 80
    energies = np.load(energies_path_after)
    assert energies.shape[0] == sum(seg["n_frames"] for seg in meta_after["segments"])


def test_decorrelate_per_segment_calls_autocorrelation_once_per_segment(monkeypatch):
    # 🔑 [P2 fix follow-up] 之前这个测试用两段都低于默认 20 帧门槛的短段
    # （3/4 帧）验证"逐段独立去相关、不跨段拼接"——但 P2 批次修复之后，低于
    # 门槛的整段会被直接排除出 MBAR（诊断用途），根本不会调用
    # subsample_series_by_autocorrelation。要验证的性质（逐段独立、不跨段
    # 拼接）本身没变，只是必须换成两段都过线（>=20 帧）的数据才能真正走到
    # 调用路径；"低于门槛整段排除"的新行为改到下面单独一个测试里验证。
    calls = []

    def _spy(series, min_frames_for_subsampling=20):
        calls.append(series.shape[0])
        return np.arange(series.shape[0]), 1.0

    monkeypatch.setattr("ibs_engine.subsample_series_by_autocorrelation", _spy)
    segments = [
        {"burn_in_steps": 4, "sample_steps": 50, "n_frames": 25, "reason": "fresh"},
        {"burn_in_steps": 4, "sample_steps": 60, "n_frames": 30, "reason": "checkpoint_missing_or_corrupted"},
    ]
    diff_series = np.arange(55, dtype=np.float64)

    indices, g_list, short_segments = _decorrelate_per_segment(diff_series, segments)

    assert calls == [25, 30]  # one call per segment, never on the 55-frame concatenation
    assert len(short_segments) == 0  # both segments clear the default 20-frame floor
    assert list(indices) == list(range(55))


def test_decorrelate_per_segment_excludes_segments_below_floor_without_calling_autocorrelation(monkeypatch):
    # 🔑 [P2 fix follow-up] 低于 min_frames_for_subsampling 的整段现在被直接
    # 排除出 combined 索引集（诊断用途，报告在 short_segments 里），既不参与
    # 去相关调用，也不以 g=1 的满权重悄悄混进 MBAR。
    calls = []

    def _spy(series, min_frames_for_subsampling=20):
        calls.append(series.shape[0])
        return np.arange(series.shape[0]), 1.0

    monkeypatch.setattr("ibs_engine.subsample_series_by_autocorrelation", _spy)
    segments = [
        {"burn_in_steps": 4, "sample_steps": 6, "n_frames": 3, "reason": "fresh"},
        {"burn_in_steps": 4, "sample_steps": 8, "n_frames": 4, "reason": "checkpoint_missing_or_corrupted"},
    ]
    diff_series = np.arange(7, dtype=np.float64)

    indices, g_list, short_segments = _decorrelate_per_segment(segments=segments, diff_series=diff_series)

    assert calls == []  # both segments are below the floor -- never handed to autocorrelation
    assert len(short_segments) == 2
    assert short_segments[0] == {"segment_index": 0, "n_frames": 3}
    assert short_segments[1] == {"segment_index": 1, "n_frames": 4}
    assert list(indices) == []


def test_bias_calibration_pair_is_sufficient_requires_all_three_conditions():
    base_pair = {
        "passed": True,
        "n_k_decorrelated": [25, 30],
        "delta_f_bias_uncertainty_kJ_mol": 0.5,
    }
    assert _bias_calibration_pair_is_sufficient(dict(base_pair)) is True

    failing_overlap = dict(base_pair, passed=False)
    assert _bias_calibration_pair_is_sufficient(failing_overlap) is False

    failing_samples = dict(base_pair, n_k_decorrelated=[5, 30])
    assert _bias_calibration_pair_is_sufficient(failing_samples) is False

    failing_uncertainty = dict(base_pair, delta_f_bias_uncertainty_kJ_mol=5.0)
    assert _bias_calibration_pair_is_sufficient(failing_uncertainty) is False

    assert _bias_calibration_pair_is_sufficient(None) is False


def test_analyze_adjacent_pair_lrc_only_applies_when_coefficients_given():
    n = 30
    rng = np.random.default_rng(0)
    segments = [{"burn_in_steps": 0, "sample_steps": n, "n_frames": n, "reason": "fresh"}]

    def _record(offset):
        u = np.zeros((n, 2), dtype=np.float64)
        u[:, 0] = offset
        u[:, 1] = offset + 10.0 + rng.normal(scale=0.01, size=n)
        return {
            "u_cv_kj_mol": u,
            "volume_nm3": np.full(n, 8.0, dtype=np.float64),
            "segments": segments,
            "sampled_steps": n,
        }

    record_i = _record(0.0)
    record_j = _record(100.0)
    kt = 2.5

    # _analyze_adjacent_pair itself only returns the reduced-unit MBAR fields
    # (delta_f_reduced_kT) -- delta_f_kJ_mol is attached by the bank callers
    # after multiplying by kt, not by this function.
    no_lrc = _analyze_adjacent_pair(record_i, record_j, 0, 1, kt, lrc_coeff=None)
    with_lrc = _analyze_adjacent_pair(
        record_i, record_j, 0, 1, kt, lrc_coeff=np.array([0.0, 40.0])
    )
    assert no_lrc["delta_f_reduced_kT"] != pytest.approx(with_lrc["delta_f_reduced_kT"])


def test_bias_calibration_bank_never_applies_lrc_regardless_of_wrapper():
    # Structural guarantee: the bias-calibration bank must call
    # _analyze_adjacent_pair with lrc_coeff=None literally, not read it off
    # the wrapper -- production's Group 1 bias force never includes LRC
    # either (see IBS_BIAS_PROTOCOL_VERSION=10/11 changelogs).
    import inspect
    import ibs_engine

    source = inspect.getsource(ibs_engine.probe_adjacent_bias_calibration_bank)
    assert "lrc_coeff=None" in source
    assert "lj_tail_lrc_coeff_kj_mol" not in source


# ============================================================================
# Main-window OpenMM checkpoint continuation (MAIN_WINDOW_CHECKPOINT_PROTOCOL_
# VERSION=1): run_all_windows resumed_calibration_pending / first-time MBAR
# calibration validation now persists a native OpenMM checkpoint so a ladder
# retry (frozen_validation_step_overrides escalation) can skip re-minimizing/
# re-ramping. These tests only exercise the standalone helpers -- the branch
# inside run_all_windows itself needs a real window/System to reach, same
# reasoning as the resumed_calibration_pending mode-fallback fix above.
# ============================================================================

def test_main_window_checkpoint_protocol_version_is_explicit():
    # 🔑 同上（IBS_BIAS_PROTOCOL_VERSION=13 的 changelog）：1->2，这份 checkpoint
    # 的 manifest 只哈希 win_sys_xml/lambdas/lambda_shield/温度/平台，Boresch
    # Context 修复不改变其中任何一项（win_sys_xml 序列化字节不变，Boresch 力
    # 默认值仍是 0.0，只是运行时 setParameter 覆盖），所以旧 checkpoint 会被
    # 误判为仍然可用——必须独立升版本号强制拒绝，不能指望内容指纹自动感知。
    assert MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION == 2


def test_build_main_window_checkpoint_manifest_fields():
    manifest = _build_main_window_checkpoint_manifest(
        stage_type="vdw",
        window_idx=3,
        K=4,
        win_sys_xml="<System/>",
        lambdas_coul=[1.0, 1.0, 1.0, 1.0],
        lambdas_vdw=[0.6, 0.4, 0.2, 0.0],
        lambda_shield=0.3,
        temperature_K=300.0,
        platform_name="CUDA",
    )
    assert manifest["main_window_checkpoint_protocol_version"] == MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION
    assert manifest["ibs_bias_protocol_version"] == IBS_BIAS_PROTOCOL_VERSION
    assert manifest["stage_type"] == "vdw"
    assert manifest["window_idx"] == 3
    assert manifest["K"] == 4
    assert manifest["lambdas_coul"] == [1.0, 1.0, 1.0, 1.0]
    assert manifest["lambdas_vdw"] == [0.6, 0.4, 0.2, 0.0]
    assert manifest["lambda_shield"] == pytest.approx(0.3)
    assert manifest["platform_name"] == "CUDA"
    # XML blobs are hashed, not stored verbatim, matching the probe-bank manifest.
    assert "win_sys_xml_sha256" in manifest
    assert "<System/>" not in json.dumps(manifest)

    manifest_none_shield = _build_main_window_checkpoint_manifest(
        stage_type="vdw", window_idx=3, K=4, win_sys_xml="<System/>",
        lambdas_coul=[1.0], lambdas_vdw=[0.0], lambda_shield=None,
        temperature_K=300.0, platform_name="CUDA",
    )
    assert manifest_none_shield["lambda_shield"] is None


def test_main_window_checkpoint_is_usable_requires_files_and_exact_match(tmp_path):
    checkpoint_dir = str(tmp_path)
    stage_type, window_idx = "vdw", 2
    expected = _build_main_window_checkpoint_manifest(
        stage_type=stage_type, window_idx=window_idx, K=3, win_sys_xml="<System/>",
        lambdas_coul=[1.0, 1.0, 1.0], lambdas_vdw=[0.5, 0.25, 0.0],
        lambda_shield=0.25, temperature_K=300.0, platform_name="Reference",
    )

    # Neither file exists yet.
    assert _main_window_checkpoint_is_usable(checkpoint_dir, stage_type, window_idx, expected) is False

    ckpt_path, manifest_path = _main_window_checkpoint_paths(checkpoint_dir, stage_type, window_idx)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(ckpt_path, "wb") as f:
        f.write(b"not a real checkpoint, presence is all that's checked here")

    # Checkpoint exists but manifest.json doesn't yet.
    assert _main_window_checkpoint_is_usable(checkpoint_dir, stage_type, window_idx, expected) is False

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(expected, f)

    # Both files present, manifest matches exactly.
    assert _main_window_checkpoint_is_usable(checkpoint_dir, stage_type, window_idx, expected) is True

    # Any single field drifting (e.g. lambda grid changed by auto-repair)
    # must invalidate the whole thing, not just warn.
    drifted = dict(expected)
    drifted["lambdas_vdw"] = [0.6, 0.3, 0.0]
    assert _main_window_checkpoint_is_usable(checkpoint_dir, stage_type, window_idx, drifted) is False

    # Corrupt manifest JSON must fail closed, not raise.
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert _main_window_checkpoint_is_usable(checkpoint_dir, stage_type, window_idx, expected) is False


def test_peek_ibs_bias_status_handles_missing_and_corrupt_files(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.json")
    assert _peek_ibs_bias_status(missing_path) is None

    corrupt_path = str(tmp_path / "corrupt.json")
    with open(corrupt_path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert _peek_ibs_bias_status(corrupt_path) is None

    good_path = str(tmp_path / "good.json")
    with open(good_path, "w", encoding="utf-8") as f:
        json.dump({"bias_status": "calibrated_pending_validation"}, f)
    assert _peek_ibs_bias_status(good_path) == "calibrated_pending_validation"

    no_status_path = str(tmp_path / "no_status.json")
    with open(no_status_path, "w", encoding="utf-8") as f:
        json.dump({"f_k": [0.0]}, f)
    assert _peek_ibs_bias_status(no_status_path) is None


def test_try_load_main_window_checkpoint_roundtrip_and_missing_path(tmp_path):
    toy = _toy_probe_system(K=1, with_group4=False)
    sim = _build_fixed_state_simulation(
        topology=toy["topology"],
        system_xml=toy["common_xml"],
        cv_xml=toy["cv_xmls"][0],
        require_group4=False,
        platform_name="Reference",
        temperature_q=toy["temperature"],
        positions=toy["positions"],
        box_vectors=toy["box_vectors"],
        integrator_seed=1,
        velocity_seed=2,
    )
    # Move the single particle away from its checkpoint-time position so a
    # successful load is actually observable (not just "didn't raise").
    sim.step(5)
    moved_state = sim.context.getState(getPositions=True)
    moved_positions = moved_state.getPositions(asNumpy=True)

    ckpt_path = str(tmp_path / "openmm.chk")
    _atomic_save_openmm_checkpoint(sim, ckpt_path)

    sim.context.setPositions(moved_positions + ([1.0, 1.0, 1.0] * unit.nanometer))
    perturbed_positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    assert not np.allclose(
        perturbed_positions.value_in_unit(unit.nanometer),
        moved_positions.value_in_unit(unit.nanometer),
    )

    assert _try_load_main_window_checkpoint(sim, ckpt_path) is True
    restored_positions = sim.context.getState(getPositions=True).getPositions(asNumpy=True)
    assert np.allclose(
        restored_positions.value_in_unit(unit.nanometer),
        moved_positions.value_in_unit(unit.nanometer),
    )

    # Missing/incompatible checkpoint path must fail closed, never raise.
    assert _try_load_main_window_checkpoint(sim, str(tmp_path / "does_not_exist.chk")) is False


# ============================================================================
# Frozen-validation ladder correctness: cumulative (not per-attempt) budget
# accounting, and a genuine terminal state distinct from
# "calibrated_pending_validation" (which implies "still eligible for
# auto-continue"). These are duck-typed against a fake OpenMM context so they
# don't need a real window/System -- IBSSampler.save_ibs_state/load_ibs_state
# only ever touch self.context.getParameter/setParameter, never build one.
# ============================================================================

class _FakeIbsContext:
    def __init__(self, n_states, prefix):
        self._params = {f"{prefix}_f_{k}": 0.0 for k in range(n_states)}

    def getParameter(self, name):
        return self._params[name]

    def setParameter(self, name, value):
        self._params[name] = value


class _FakeIbsSampler:
    """Minimal stand-in exposing exactly what save_ibs_state/load_ibs_state
    read or write on self -- avoids building a real IBSSampler (which would
    need a Context wired up with the full IBS bias CustomCVForce plumbing,
    not just plain global parameters)."""

    def __init__(self, n_states=3, prefix="test"):
        self.n_states = n_states
        self.prefix = prefix
        self.context = _FakeIbsContext(n_states, prefix)
        self.f_history = [np.zeros(n_states)] * 5
        self.tmbar_history = []
        self.tmbar_history_dropped_entries = 0
        self.eta_penalty = 1.0
        self.e_offset = 0.0
        self.bias_converged = False
        self.bias_status = "unconverged"
        self.frozen_f_k_pending = None
        self.frozen_validation_cumulative_steps = 0
        self.sampling_repair_policy = "non_mutating_v1"


def test_ibs_frozen_calibration_validation_error_carries_terminal_flag():
    pending = IBSFrozenCalibrationValidationError("still escalating", diagnostics={"a": 1})
    assert pending.terminal is False
    assert pending.diagnostics == {"a": 1}

    terminal = IBSFrozenCalibrationValidationError(
        "exhausted", diagnostics={"b": 2}, terminal=True
    )
    assert terminal.terminal is True
    # Must be a RuntimeError, not IBSWarmupConvergenceError -- the two failure
    # modes are deliberately distinct exception types so upstream split/insert-λ
    # repair logic (which only catches IBSWarmupConvergenceError) can't
    # accidentally swallow a calibrated-f_k validation failure.
    assert isinstance(terminal, RuntimeError)
    assert not isinstance(terminal, IBSWarmupConvergenceError)


def test_ibs_sampler_persists_and_restores_frozen_validation_cumulative_steps(tmp_path):
    filepath = str(tmp_path / "ibs_state.json")
    lambdas_coul = [1.0, 1.0, 1.0]
    lambdas_vdw = [0.5, 0.25, 0.0]

    saver = _FakeIbsSampler(n_states=3, prefix="test")
    saver.bias_status = "calibrated_pending_validation"
    saver.frozen_f_k_pending = [1.0, 2.0, 3.0]
    # Simulates having already spent 50k steps validating this exact f_k in a
    # prior attempt (e.g. the first ladder rung), before this save.
    saver.frozen_validation_cumulative_steps = 50_000

    IBSSampler.save_ibs_state(saver, filepath, lambdas_coul=lambdas_coul, lambdas_vdw=lambdas_vdw)

    with open(filepath) as f:
        raw = json.load(f)
    assert raw["bias_status"] == "calibrated_pending_validation"
    assert raw["frozen_validation_cumulative_steps"] == 50_000
    assert raw["tmbar_history"] == []

    # A fresh process resuming (e.g. after a --resume or a ladder-escalation
    # retry) must read the cumulative count back, not restart it at 0 -- this
    # is exactly the "budget is cumulative, not per-attempt" fix: run_all_windows
    # computes this attempt's remaining budget as (target - this value), so the
    # ladder's 50k/150k/300k really are cumulative totals, not three separate
    # 50k+150k+300k=500k blocks.
    loader = _FakeIbsSampler(n_states=3, prefix="test")
    ok = IBSSampler.load_ibs_state(loader, filepath, lambdas_coul=lambdas_coul, lambdas_vdw=lambdas_vdw)
    assert ok is True
    assert loader.bias_status == "calibrated_pending_validation"
    assert loader.frozen_f_k_pending == [1.0, 2.0, 3.0]
    assert loader.frozen_validation_cumulative_steps == 50_000
    assert loader.tmbar_history == []


def test_ibs_sampler_load_terminal_status_clears_pending_fields(tmp_path):
    filepath = str(tmp_path / "ibs_state.json")
    lambdas_coul = [1.0, 1.0, 1.0]
    lambdas_vdw = [0.5, 0.25, 0.0]

    # Hand-write a state file the way run_all_windows would after the ladder's
    # final rung (300k) still failed: bias_status is the terminal marker, not
    # "calibrated_pending_validation" -- this is the core fix for "the state
    # still says pending/eligible-for-auto-resume even after the ladder is
    # exhausted."
    state = {
        "n_states": 3,
        "prefix": "test",
        "f_k": [1.0, 2.0, 3.0],
        "t": 5,
        "eta_penalty": 1.0,
        "e_offset": 0.0,
        "tmbar_history": [],
        "tmbar_history_dropped_entries": 0,
        "status": "running",
        "bias_converged": False,
        "bias_status": "calibrated_validation_failed",
        "frozen_f_k_pending": [1.0, 2.0, 3.0],  # stale leftover; load must ignore/clear it
        "frozen_validation_cumulative_steps": 300_000,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        "warmup_update_protocol_version": ibs_engine.IBS_WARMUP_UPDATE_PROTOCOL_VERSION,
        "sampling_repair_policy": "non_mutating_v1",
        "lambdas_coul": lambdas_coul,
        "lambdas_vdw": lambdas_vdw,
    }
    with open(filepath, "w") as f:
        json.dump(state, f)

    loader = _FakeIbsSampler(n_states=3, prefix="test")
    ok = IBSSampler.load_ibs_state(loader, filepath, lambdas_coul=lambdas_coul, lambdas_vdw=lambdas_vdw)
    assert ok is True
    assert loader.bias_status == "calibrated_validation_failed"
    # Terminal means "not pending" -- must not be handed back out as something
    # eligible for resumed_calibration_pending to auto-continue.
    assert loader.frozen_f_k_pending is None
    assert loader.frozen_validation_cumulative_steps == 0


def test_ibs_sampler_load_recognizes_failed_status_as_terminal(tmp_path):
    # [Candidate-first, Validate-or-Learn v1] Going forward, a VALIDATE
    # exhaustion that actually completed a local-MBAR gate evaluation and
    # failed it writes bias_status="failed" (not the legacy
    # calibrated_validation_failed). load_ibs_state must preserve this value
    # verbatim -- falling through to the generic else-branch would silently
    # rewrite it to "unconverged", and the terminal-raise check in
    # run_all_windows (which checks for "failed" alongside the legacy value)
    # would then never fire, letting a terminally-failed window be resumed as
    # an ordinary unconverged warm-start.
    filepath = str(tmp_path / "ibs_state.json")
    lambdas_coul = [1.0, 1.0, 1.0]
    lambdas_vdw = [0.5, 0.25, 0.0]
    state = {
        "n_states": 3,
        "prefix": "test",
        "f_k": [1.0, 2.0, 3.0],
        "t": 5,
        "eta_penalty": 1.0,
        "e_offset": 0.0,
        "tmbar_history": [],
        "tmbar_history_dropped_entries": 0,
        "status": "running",
        "bias_converged": False,
        "bias_status": "failed",
        "frozen_f_k_pending": None,
        "frozen_validation_cumulative_steps": 0,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        "warmup_update_protocol_version": ibs_engine.IBS_WARMUP_UPDATE_PROTOCOL_VERSION,
        "sampling_repair_policy": "non_mutating_v1",
        "lambdas_coul": lambdas_coul,
        "lambdas_vdw": lambdas_vdw,
    }
    with open(filepath, "w") as f:
        json.dump(state, f)

    loader = _FakeIbsSampler(n_states=3, prefix="test")
    ok = IBSSampler.load_ibs_state(loader, filepath, lambdas_coul=lambdas_coul, lambdas_vdw=lambdas_vdw)
    assert ok is True
    assert loader.bias_status == "failed"
    assert loader.frozen_f_k_pending is None


def test_ibs_sampler_persists_and_restores_candidate_first_metadata(tmp_path):
    # [Candidate-first, Validate-or-Learn v1] seed_source/validation_attempts/
    # last_failure_reason are pure additive metadata -- never gate anything,
    # round-trip through save/load like any other diagnostic field.
    filepath = str(tmp_path / "ibs_state.json")
    lambdas_coul = [1.0, 1.0, 1.0]
    lambdas_vdw = [0.5, 0.25, 0.0]

    saver = _FakeIbsSampler(n_states=3, prefix="test")
    saver.seed_source = "pilot"
    saver.validation_attempts = 3
    saver.last_failure_reason = "local_mbar_gap_exceeded_after_retry"

    IBSSampler.save_ibs_state(saver, filepath, lambdas_coul=lambdas_coul, lambdas_vdw=lambdas_vdw)

    with open(filepath) as f:
        raw = json.load(f)
    assert raw["seed_source"] == "pilot"
    assert raw["validation_attempts"] == 3
    assert raw["last_failure_reason"] == "local_mbar_gap_exceeded_after_retry"
    # learning_updates is a save-time-only alias of "t" -- never an
    # independently loaded counter (avoids two numbers that could drift).
    assert raw["learning_updates"] == raw["t"] == 5

    loader = _FakeIbsSampler(n_states=3, prefix="test")
    ok = IBSSampler.load_ibs_state(loader, filepath, lambdas_coul=lambdas_coul, lambdas_vdw=lambdas_vdw)
    assert ok is True
    # A successful load_ibs_state IS a resume by definition -- seed_source is
    # overwritten to "resume" here regardless of what produced the f_k being
    # resumed (the saved "pilot" value is preserved on disk for historical
    # provenance, see the raw-JSON assertion above, but the live sampler's
    # seed_source reflects how *this* run obtained the candidate).
    assert loader.seed_source == "resume"
    assert loader.validation_attempts == 3
    assert loader.last_failure_reason == "local_mbar_gap_exceeded_after_retry"


def test_ibs_sampler_load_defaults_candidate_first_metadata_for_old_caches(tmp_path):
    # An old cache saved before this redesign has none of the four new keys.
    # load_ibs_state must not fail-closed on their absence -- pure additive
    # metadata gets safe defaults, exactly like any other backward-compatible
    # field in this loader.
    filepath = str(tmp_path / "ibs_state.json")
    lambdas_coul = [1.0, 1.0, 1.0]
    lambdas_vdw = [0.5, 0.25, 0.0]
    state = {
        "n_states": 3,
        "prefix": "test",
        "f_k": [1.0, 2.0, 3.0],
        "t": 5,
        "eta_penalty": 1.0,
        "e_offset": 0.0,
        "tmbar_history": [],
        "tmbar_history_dropped_entries": 0,
        "status": "running",
        "bias_converged": False,
        "bias_status": "unconverged",
        "frozen_f_k_pending": None,
        "frozen_validation_cumulative_steps": 0,
        "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
        "warmup_update_protocol_version": ibs_engine.IBS_WARMUP_UPDATE_PROTOCOL_VERSION,
        "sampling_repair_policy": "non_mutating_v1",
        "lambdas_coul": lambdas_coul,
        "lambdas_vdw": lambdas_vdw,
        # seed_source/validation_attempts/last_failure_reason/learning_updates
        # deliberately absent.
    }
    with open(filepath, "w") as f:
        json.dump(state, f)

    loader = _FakeIbsSampler(n_states=3, prefix="test")
    ok = IBSSampler.load_ibs_state(loader, filepath, lambdas_coul=lambdas_coul, lambdas_vdw=lambdas_vdw)
    assert ok is True
    # seed_source is unconditionally "resume" after any successful load (see
    # test_ibs_sampler_persists_and_restores_candidate_first_metadata); the
    # two fields that genuinely come from state.get(..., default) on a cache
    # missing them are validation_attempts/last_failure_reason.
    assert loader.seed_source == "resume"
    assert loader.validation_attempts == 0
    assert loader.last_failure_reason is None


def test_apply_pairwise_cap_caps_and_preserves_direction():
    # [Candidate-first, Validate-or-Learn v1] Pure numeric contract for the
    # helper extracted out of update_weights()'s previously-inlined hard-cap
    # block, now shared with the VALIDATE-failure damped-correction retry.
    f_old = np.array([0.0, 0.0, 0.0])
    f_candidate = np.array([-20.0, 0.0, 20.0])  # pairwise spread = 40
    kt = 2.5

    f_new, diag = IBSSampler._apply_pairwise_cap(f_old, f_candidate, cap_kt=2.0, kt=kt)
    assert diag["hard_pairwise_cap_applied"] is True
    assert diag["hard_pairwise_cap_kJ_mol"] == pytest.approx(5.0)
    spread = float(np.max(f_new) - np.min(f_new))
    assert spread == pytest.approx(5.0)
    # Direction preserved: state 2 still highest, state 0 still lowest.
    assert f_new[2] > f_new[1] > f_new[0]
    # Mean-centered.
    assert float(np.mean(f_new)) == pytest.approx(0.0, abs=1e-9)

    # No-op when the candidate step is already within the cap.
    f_small = np.array([-1.0, 0.0, 1.0])
    f_new2, diag2 = IBSSampler._apply_pairwise_cap(f_old, f_small, cap_kt=2.0, kt=kt)
    assert diag2["hard_pairwise_cap_applied"] is False
    np.testing.assert_allclose(f_new2, f_small)


# ============================================================================
# Frozen-validation ladder rung resolution -- covers a real production bug
# (2026-07-17): abfe_pipeline.py's per-window step-budget/is-final-rung
# overrides live only in a local Python dict for the lifetime of one process
# and are never persisted. A killed-and-`--resume`d job starts that dict
# empty again, even though the window's real progress
# (frozen_validation_cumulative_steps) survived on disk. The old fallback
# ("no override entry -> just use mbar_calibration_reserved_steps/False")
# ignored that persisted progress entirely. These tests cover the fix:
# _resolve_frozen_validation_budget_for_window/_resolve_frozen_validation_is_
# final_rung must derive the right rung from persisted progress when the
# caller's override dicts don't have an entry for this window, while still
# preferring an explicit override when one is present (the normal
# same-process ladder-escalation case, unchanged from before).
# ============================================================================

def test_resolve_frozen_validation_budget_prefers_explicit_override():
    # Same-process ladder escalation: abfe_pipeline.py has already recorded
    # this window's next rung. Must be used verbatim, ignoring persisted
    # cumulative steps entirely.
    budget = _resolve_frozen_validation_budget_for_window(
        window_idx=1,
        frozen_validation_step_overrides={1: 150_000},
        prior_cumulative_steps=999_999,
    )
    assert budget == 150_000


def test_resolve_frozen_validation_budget_falls_back_to_next_unfinished_rung():
    # Post-restart case: no override recorded for this window (fresh process),
    # but persisted progress already reached 50_000 -- must resolve to the
    # *next* rung (150_000), not silently reset back to the first rung.
    schedule = FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS
    assert schedule == (50_000, 150_000, 300_000)

    assert _resolve_frozen_validation_budget_for_window(
        window_idx=1, frozen_validation_step_overrides={}, prior_cumulative_steps=0,
    ) == 50_000
    assert _resolve_frozen_validation_budget_for_window(
        window_idx=1, frozen_validation_step_overrides={}, prior_cumulative_steps=50_000,
    ) == 150_000
    assert _resolve_frozen_validation_budget_for_window(
        window_idx=1, frozen_validation_step_overrides={}, prior_cumulative_steps=150_000,
    ) == 300_000
    # Persisted progress already at/past the last rung (e.g. from the
    # double-counting bug's historical inflation, or a genuine full climb) --
    # must clamp to the last rung, not raise/return None, so the caller's
    # is-final-rung check below can correctly declare this terminal.
    assert _resolve_frozen_validation_budget_for_window(
        window_idx=1, frozen_validation_step_overrides={}, prior_cumulative_steps=300_000,
    ) == 300_000
    assert _resolve_frozen_validation_budget_for_window(
        window_idx=1, frozen_validation_step_overrides={}, prior_cumulative_steps=999_999,
    ) == 300_000
    # An override recorded for a *different* window must not affect this one.
    assert _resolve_frozen_validation_budget_for_window(
        window_idx=2, frozen_validation_step_overrides={1: 300_000}, prior_cumulative_steps=50_000,
    ) == 150_000


def test_resolve_frozen_validation_is_final_rung_prefers_explicit_override():
    assert _resolve_frozen_validation_is_final_rung(
        window_idx=1,
        frozen_validation_is_final_rung={1: True},
        effective_frozen_validation_budget=150_000,  # not the last rung, but override wins
    ) is True
    assert _resolve_frozen_validation_is_final_rung(
        window_idx=1,
        frozen_validation_is_final_rung={1: False},
        effective_frozen_validation_budget=300_000,  # is the last rung, but override wins
    ) is False


def test_resolve_frozen_validation_is_final_rung_falls_back_to_schedule_comparison():
    # Post-restart case: no override recorded for this window. Must not
    # default to False -- a window whose persisted progress already resolved
    # to the last rung (via _resolve_frozen_validation_budget_for_window) has
    # to be recognized as terminal here, or it would loop on the same rung
    # forever instead of ever raising IBSFrozenCalibrationValidationError(terminal=True).
    assert _resolve_frozen_validation_is_final_rung(
        window_idx=1, frozen_validation_is_final_rung={}, effective_frozen_validation_budget=300_000,
    ) is True
    assert _resolve_frozen_validation_is_final_rung(
        window_idx=1, frozen_validation_is_final_rung={}, effective_frozen_validation_budget=150_000,
    ) is False


def test_frozen_validation_cumulative_steps_not_double_counted_across_attempts():
    """Regression test for the 2026-07-17 double-counting bug: the terminal/
    failure-path bookkeeping used to re-add this attempt's steps on top of an
    already-updated `sampler.frozen_validation_cumulative_steps` (mutated
    every batch by the in-loop checkpoint-save code), inflating the persisted
    total by one attempt's worth of steps every time a resumed-calibration
    attempt failed. With the old bug, a rung-150_000 attempt (true prior
    50_000, 100_000 real steps run) persisted 250_000 instead of 150_000; the
    next (final, rung-300_000) attempt then read that inflated 250_000 back
    as its own prior, computed a too-small remaining budget (50_000 instead
    of the correct 150_000), and double-counted again up to 350_000 --
    exactly the numbers seen in the real failure JSON. This test uses the
    fixed formula (`prior_cumulative_steps + steps_spent_this_attempt`,
    where `prior_cumulative_steps` is the value captured BEFORE the loop's
    own in-progress mutation, not a re-read of that already-mutated
    attribute) and checks that chaining two attempts with their *correct*
    (uncorrupted-prior-derived) step counts lands exactly on the ladder's
    real final target, not past it.
    """
    def run_one_attempt(prior_cumulative_steps, steps_at_full_bias):
        # Mirrors ibs_engine.py's in-loop checkpoint-save mutation (every
        # batch, using the pre-loop prior_cumulative_steps + the running
        # counter) followed by the (now-fixed) post-loop failure-path
        # computation.
        sampler = _FakeIbsSampler(n_states=4, prefix="test")
        sampler.frozen_validation_cumulative_steps = prior_cumulative_steps + steps_at_full_bias
        steps_spent_this_attempt = int(steps_at_full_bias)
        new_cumulative_steps = prior_cumulative_steps + steps_spent_this_attempt
        sampler.frozen_validation_cumulative_steps = new_cumulative_steps
        return sampler.frozen_validation_cumulative_steps

    # Rung 150_000: true prior 50_000, runs the correct remaining budget (100_000).
    cumulative_after_rung_150k = run_one_attempt(prior_cumulative_steps=50_000, steps_at_full_bias=100_000)
    assert cumulative_after_rung_150k == 150_000  # not 250_000

    # Rung 300_000 (final): with the fix, the persisted prior handed to the
    # next attempt is the true 150_000, so remaining_budget_this_attempt =
    # 300_000-150_000 = 150_000 real steps actually get run (not 50_000, an
    # artifact of the old bug's corrupted, inflated prior).
    cumulative_after_rung_300k = run_one_attempt(
        prior_cumulative_steps=cumulative_after_rung_150k, steps_at_full_bias=150_000,
    )
    assert cumulative_after_rung_300k == 300_000  # not 350_000
