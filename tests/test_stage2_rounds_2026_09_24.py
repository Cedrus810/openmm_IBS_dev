"""stage 2 轮次控制器纯函数层（`stage2_rounds.py`）。设计见
docs/design/PROPOSAL_stage2_lcr_rounds_2026-09-24.md。

占据的判据全部用 IBS 混合的平衡占据 p_k ∝ exp[(f_k − F_k)/kT] 来写，
不是只比数组 —— 符号约定错了，这些测试会直接暴露出来。
"""

import ast
import pathlib

import numpy as np
import pytest

import stage2_rounds as sr

KT = 2.494
ROOT = pathlib.Path(__file__).resolve().parent.parent


def _occ(f, F):
    """归一化占据 K·p_k。"""
    x = (np.asarray(f) - np.asarray(F)) / KT
    p = np.exp(x - x.max())
    p /= p.sum()
    return len(p) * p


def _ev(**kw):
    base = dict(
        window=0, lambdas_vdw=[0.8, 0.76, 0.72, 0.68, 0.64, 0.6],
        state_ess=[100.0] * 6, top1pct_weight=0.1, sigma_mbar_kJ=0.2,
        loo_delta_f_kJ=[10.0, 10.1, 9.9], n_independent_seeds=3,
        merged_f_k_kJ=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        anchor_f_k_kJ=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0], trust_radius_kJ=3.0 * KT,
    )
    base.update(kw)
    return sr.WindowEvidence(**base)


# ---------------------------------------------------------------------------
# 模块边界与常量对账
# ---------------------------------------------------------------------------

def test_module_is_pure_no_openmm_no_disk():
    tree = ast.parse((ROOT / "stage2_rounds.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "dataclasses", "typing", "numpy"}, imported
    src = (ROOT / "stage2_rounds.py").read_text(encoding="utf-8")
    assert "open(" not in src


def test_constants_mirror_the_engine():
    ie = pytest.importorskip("ibs_engine")
    assert sr.TILT_KT == ie.IBS_REPRO_TILT_MARGIN_KT
    assert sr.ESS_MIN == ie.TARGET_SUPPORT_MIN_ABSOLUTE_ESS
    assert sr.TOP1PCT_MAX == ie.TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT


def test_decided_defaults():
    # 09-24 设计 §10 拍板的数
    assert sr.TILT_KT == 2.0
    assert sr.TRUST_FLOOR_KT == 3.0
    assert sr.TRUST_MIN_ESS == 50.0
    assert sr.SIGMA_TOL_KJ == 1.5
    assert sr.MAX_ROUNDS == 3
    assert sr.ZERO_WINDOW_END_SHIFT_KT_COMPLEX == 2.0


# ---------------------------------------------------------------------------
# 倾斜 / 末态上抬
# ---------------------------------------------------------------------------

def test_tilt_shape_and_sign_semantics():
    F = np.array([0.0, 3.0, 5.0, 6.0, 6.5])
    left = sr.tilt_f_k(F, KT, +1)
    right = sr.tilt_f_k(F, KT, -1)
    # 首末总差 = Δ·kT，线性
    assert (left[0] - F[0]) - (left[-1] - F[-1]) == pytest.approx(2.0 * KT)
    assert np.allclose(np.diff(left - F), np.diff(left - F)[0])
    # +1 抬首态：首态占据变多；−1 抬末态：末态占据变多
    assert _occ(left, F)[0] > 1.0 > _occ(left, F)[-1]
    assert _occ(right, F)[-1] > 1.0 > _occ(right, F)[0]
    # 首末权重比 = e^Δ ≈ 7（IBS 里倾斜必须小）
    o = _occ(left, F)
    assert o[0] / o[-1] == pytest.approx(np.exp(2.0), rel=1e-9)
    assert np.array_equal(sr.tilt_f_k(F, KT, 0), F)


def test_shift_last_state_raises_last_relative_to_first_and_keeps_mean():
    a = np.array([1.0, -2.0, 0.5, 3.0])
    g = sr.shift_last_state(a, KT, 2.0)
    assert (g[-1] - g[0]) - (a[-1] - a[0]) == pytest.approx(2.0 * KT)
    assert g.mean() == pytest.approx(a.mean())


def test_window_touches_zero():
    assert sr.window_touches_zero([0.2, 0.1, 0.0])
    assert sr.window_touches_zero([0.2, 1e-12])
    assert not sr.window_touches_zero([0.4, 0.2])


# ---------------------------------------------------------------------------
# 信赖域
# ---------------------------------------------------------------------------

def test_projection_is_noop_inside_the_region():
    a = np.array([0.0, 2.0, 5.0, 9.0])
    f = a + np.array([0.0, 1.0, -1.0, 2.0]) + 7.0      # 偏移都 < R，外加整体常数
    g, clamped = sr.project_to_trust_region(f, a, 3.0 * KT)
    assert not clamped
    assert np.allclose(g, f)


def test_projection_clips_runaway_and_is_gauge_invariant():
    a = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    runaway = a * 3.0                                   # 09-23 那种谱宽被追到几倍
    R = 3.0 * KT
    g, clamped = sr.project_to_trust_region(runaway, a, R)
    assert clamped
    dev = (g - g[0]) - (a - a[0])
    assert np.all(np.abs(dev) <= R + 1e-9)
    assert g.mean() == pytest.approx(runaway.mean())
    g2, _ = sr.project_to_trust_region(runaway + 123.0, a - 45.0, R)
    assert np.allclose((g2 - g2[0]), (g - g[0]))


def test_trust_radius_floor_and_sigma_terms():
    lam = [1.0, 0.75, 0.5, 0.25, 0.0]
    # 精度数据缺失 ⟹ 下限
    assert sr.pilot_trust_radius_kJ(lam, None, None, [0.75, 0.5], KT) == pytest.approx(3.0 * KT)
    # sem 很小 ⟹ 下限
    small = sr.pilot_trust_radius_kJ(lam, [1.0] * 5, [600] * 5, [0.75, 0.5], KT)
    assert small == pytest.approx(3.0 * KT)
    # sem 很大 ⟹ n_sigma·σ；σ = sqrt((0.5·0.25)²·2·sem²)
    big = sr.pilot_trust_radius_kJ(lam, [400.0] * 5, [1] * 5, [0.75, 0.5], KT)
    assert big == pytest.approx(3.0 * np.sqrt((0.5 * 0.25) ** 2 * 2 * 400.0 ** 2))


# ---------------------------------------------------------------------------
# jackknife / 标签 / σ
# ---------------------------------------------------------------------------

def test_jackknife_sigma_value_and_single_segment_dependence():
    x = [10.0, 11.0, 12.0]
    assert sr.jackknife_sigma(x) == pytest.approx(np.sqrt(2 / 3 * 2.0))
    assert sr.jackknife_sigma([10.0, float("nan"), 12.0]) is None
    assert sr.jackknife_sigma([10.0]) is None


def test_classify_window_labels():
    assert sr.classify_window(_ev())[0] == sr.LABEL_CONVERGED
    lab, why = sr.classify_window(_ev(n_independent_seeds=1))
    assert lab == sr.LABEL_PROVISIONAL and "fewer_than_two_independent_seeds" in why
    lab, why = sr.classify_window(_ev(loo_delta_f_kJ=[10.0, float("nan"), 9.0]))
    assert lab == sr.LABEL_PROVISIONAL and "sigma_jk_unavailable" in why
    lab, why = sr.classify_window(_ev(loo_delta_f_kJ=[5.0, 10.0, 15.0]))
    assert "sigma_jk_above_tol" in why
    lab, _ = sr.classify_window(_ev(state_ess=[100, 100, 0.0, 100, 100, 100]))
    assert lab == sr.LABEL_INDETERMINATE


def test_sigma_window_is_the_max_of_available_terms():
    ev = _ev(sigma_mbar_kJ=0.3, loo_delta_f_kJ=[8.0, 10.0, 12.0])
    assert sr.sigma_window_kJ(ev) == pytest.approx(ev.sigma_jk_kJ)
    assert sr.sigma_window_kJ(ev, split_half_sigma_kJ=9.0) == pytest.approx(9.0)
    ev2 = _ev(sigma_mbar_kJ=0.3, loo_delta_f_kJ=[1.0, float("nan"), 2.0])
    assert sr.sigma_window_kJ(ev2) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 第一轮
# ---------------------------------------------------------------------------

def test_first_round_three_segments_per_window_with_distinct_seeds():
    wins = [
        (0, [1.0, 0.9, 0.8], [0.0, 1.0, 2.0]),
        (4, [0.2, 0.1, 0.0], [0.0, 3.0, 5.0]),
    ]
    specs = sr.first_round_specs(wins, KT, is_complex_leg=True, zero_window_end_shift_kT=0.0)
    assert len(specs) == 6
    for w in (0, 4):
        mine = [s for s in specs if s.window == w]
        assert sorted(s.tilt_sign for s in mine) == [-1, 0, 1]
        assert sorted(s.seed_id for s in mine) == [0, 1, 2]
        assert all(s.center_source == sr.CENTER_ANCHOR and s.round_index == 1 for s in mine)
        assert [s.tilt_kT for s in mine if s.tilt_sign == 0] == [0.0]
    # 种子 0 = 中段（段 1 = 基准目录）
    assert all(s.tilt_sign == 0 for s in specs if s.seed_id == 0)


def test_solvent_leg_first_round_is_one_untilted_segment_per_window():
    """09-25 定：溶剂腿第一轮每窗只跑中段（设计 §3.1.2 / §11.5）。"""
    wins = [(0, [1.0, 0.9, 0.8], [0.0, 1.0, 2.0]), (4, [0.2, 0.1, 0.0], [0.0, 3.0, 5.0])]
    specs = sr.first_round_specs(wins, KT, is_complex_leg=False)
    assert [(s.window, s.tilt_sign, s.seed_id) for s in specs] == [(0, 0, 0), (4, 0, 0)]
    # 溶剂腿 λ=0 窗口不上抬
    assert np.allclose(specs[1].center_f_k_kJ, [0.0, 3.0, 5.0])
    assert sr.FIRST_ROUND_SEGMENTS_SOLVENT == 1 and sr.MIN_INDEPENDENT_SEEDS_SOLVENT == 1
    with pytest.raises(ValueError):
        sr.first_round_specs(wins, KT, is_complex_leg=False, n_segments=2)


def test_min_independent_seeds_is_leg_specific():
    one_seed = _ev(n_independent_seeds=1)
    assert sr.classify_window(one_seed, min_independent_seeds=1)[0] == sr.LABEL_CONVERGED
    assert sr.classify_window(one_seed)[0] == sr.LABEL_PROVISIONAL        # 复合物腿默认要 2
    assert sr.plan_next_round([one_seed], completed_round=1, used_seed_ids=[0],
                              min_independent_seeds=1) == []
    specs = sr.plan_next_round([one_seed], completed_round=1, used_seed_ids=[0])
    assert len(specs) == 1 and specs[0].reason == "too_few_independent_seeds"


def test_first_round_zero_window_shift_only_on_complex_leg():
    wins = [(3, [0.4, 0.3, 0.2], [0.0, 1.0, 2.0]), (4, [0.2, 0.1, 0.0], [0.0, 3.0, 5.0])]
    specs = sr.first_round_specs(wins, KT, is_complex_leg=True)
    c3 = next(s for s in specs if s.window == 3 and s.tilt_sign == 0).center_f_k_kJ
    c4 = np.array(next(s for s in specs if s.window == 4 and s.tilt_sign == 0).center_f_k_kJ)
    assert np.allclose(c3, [0.0, 1.0, 2.0])          # 非零窗口不动
    a = np.array([0.0, 3.0, 5.0])
    assert (c4[-1] - c4[0]) - (a[-1] - a[0]) == pytest.approx(2.0 * KT)


# ---------------------------------------------------------------------------
# plan_next_round
# ---------------------------------------------------------------------------

def test_converged_window_gets_nothing():
    assert sr.plan_next_round([_ev()], completed_round=1, used_seed_ids=[0, 1, 2]) == []


@pytest.mark.parametrize("ess", [
    [5, 80, 90, 90, 90, 90],     # 缺首态
    [90, 90, 90, 90, 80, 5],     # 缺末态
    [90, 90, 4, 90, 90, 90],     # 缺中间态
    [6, 90, 90, 90, 90, 7],      # 两端都缺
])
def test_starved_window_gets_an_untilted_recentred_segment(ess):
    """09-25 定：第二轮起一律不倾斜、只重定中心（设计 §3.3 / §11.3）。"""
    specs = sr.plan_next_round([_ev(state_ess=ess)], completed_round=1, used_seed_ids=[0, 1, 2])
    assert len(specs) == 1
    s0 = specs[0]
    assert s0.tilt_sign == 0 and s0.tilt_kT == 0.0
    assert s0.round_index == 2 and s0.seed_id == 3
    assert s0.reason.startswith("starved_state_")
    # 以合并 F 为真值，重定中心后占据是平的
    F = np.array(_ev().merged_f_k_kJ)
    assert np.allclose(_occ(s0.sampling_f_k(KT), F), 1.0)


def test_disagreeing_segments_get_an_untilted_segment_from_a_new_start():
    evs = [
        _ev(window=0, loo_delta_f_kJ=[5.0, 10.0, 15.0]),                 # σ_jk 大
        _ev(window=1, loo_delta_f_kJ=[10.0, float("nan"), 10.0]),        # 单段依赖
        _ev(window=2, top1pct_weight=0.8),
    ]
    specs = sr.plan_next_round(evs, completed_round=1, used_seed_ids=[0, 1, 2])
    assert [s.window for s in specs] == [0, 1, 2]
    assert all(s.tilt_sign == 0 and s.tilt_kT == 0.0 for s in specs)
    assert [s.seed_id for s in specs] == [3, 4, 5]
    assert specs[1].reason == "single_segment_dependence"


def test_single_segment_dependence_is_defined_by_loo_ess_collapse():
    ok = _ev(loo_min_state_ess=[40.0, 35.0, 50.0])
    dep = _ev(loo_min_state_ess=[40.0, 0.3, 50.0])
    assert not ok.single_segment_dependent and dep.single_segment_dependent
    assert sr.plan_next_round([ok], completed_round=1, used_seed_ids=[0, 1, 2]) == []
    specs = sr.plan_next_round([dep], completed_round=1, used_seed_ids=[0, 1, 2])
    assert len(specs) == 1 and specs[0].reason == "single_segment_dependence"
    lab, why = sr.classify_window(dep)
    assert lab == sr.LABEL_PROVISIONAL and "single_segment_dependent" in why


def test_starved_pattern_one_sided_vs_all_low_only_changes_the_label():
    starved = [90, 90, 90, 90, 90, 8]
    one_sided = _ev(state_ess=starved, first_round_segment_state_ess=[
        [30] * 5 + [1.0], [30] * 5 + [1.5], [30] * 5 + [6.0]])
    all_low = _ev(state_ess=starved, first_round_segment_state_ess=[
        [30] * 5 + [2.0], [30] * 5 + [3.0], [30] * 5 + [3.0]])
    assert sr.starved_pattern(one_sided) == sr.PATTERN_ONE_SIDED
    assert sr.starved_pattern(all_low) == sr.PATTERN_ALL_LOW
    assert sr.starved_pattern(_ev()) is None
    a = sr.plan_next_round([one_sided], completed_round=1, used_seed_ids=[0, 1, 2])[0]
    b = sr.plan_next_round([all_low], completed_round=1, used_seed_ids=[0, 1, 2])[0]
    # 动作相同，只有理由不同
    assert (a.tilt_sign, a.center_f_k_kJ, a.center_source) == (b.tilt_sign, b.center_f_k_kJ, b.center_source)
    assert a.reason.endswith("one_sided") and b.reason.endswith("all_low")
    assert "sampling_limited_cause_unresolved" in sr.classify_window(all_low)[1]
    assert "sampling_limited_cause_unresolved" not in sr.classify_window(one_sided)[1]


def test_later_rounds_never_tilt():
    evs = [_ev(window=w, state_ess=e) for w, e in enumerate(
        [[1, 90, 90, 90, 90, 90], [90, 90, 90, 90, 90, 1], [90, 90, 1, 90, 90, 90]])]
    for r in (1, 2):
        assert all(s.tilt_sign == 0 for s in
                   sr.plan_next_round(evs, completed_round=r, used_seed_ids=[0, 1, 2]))


def test_small_sigma_jk_below_tol_does_not_trigger_even_if_above_2_sigma_mbar():
    # MBAR σ 很小（困盆窗口典型 0.2），σ_jk = 0.8 > 2·0.2，但 < σ_tol = 1.5 ⟹ 不追
    ev = _ev(sigma_mbar_kJ=0.2, loo_delta_f_kJ=[9.0, 10.0, 11.0])
    assert 0.4 < ev.sigma_jk_kJ < 1.5
    assert sr.plan_next_round([ev], completed_round=1, used_seed_ids=[0, 1, 2]) == []


def test_sigma_jk_above_tol_but_within_2_sigma_mbar_still_triggers():
    """2026-09-29 REVIEW-0929 M2：σ_jk ∈ (σ_tol, 2σ_MBAR] 原先标 PROVISIONAL 却不补段。
    补段门现在与标签门一致（σ_jk > σ_tol）；理由区分「段间不一致」与「精度没到」。"""
    ev = _ev(sigma_mbar_kJ=2.0, loo_delta_f_kJ=[8.0, 10.0, 12.0])
    assert 1.5 < ev.sigma_jk_kJ <= 2.0 * 2.0
    label, why = sr.classify_window(ev)
    assert label == sr.LABEL_PROVISIONAL and "sigma_jk_above_tol" in why
    specs = sr.plan_next_round([ev], completed_round=1, used_seed_ids=[0, 1, 2])
    assert len(specs) == 1 and specs[0].reason.startswith("sigma_jk_above_tol_")
    # 超过 2σ_MBAR 的仍标「段间不一致」
    ev2 = _ev(sigma_mbar_kJ=0.2, loo_delta_f_kJ=[5.0, 10.0, 15.0])
    specs2 = sr.plan_next_round([ev2], completed_round=1, used_seed_ids=[0, 1, 2])
    assert specs2[0].reason.startswith("segments_disagree_sigma_jk_")


def test_round_cap():
    ev = _ev(state_ess=[1, 90, 90, 90, 90, 90])
    assert sr.plan_next_round([ev], completed_round=3, used_seed_ids=[0, 1, 2]) == []
    assert len(sr.plan_next_round([ev], completed_round=2, used_seed_ids=[0, 1, 2])) == 1


def test_center_projection_only_when_evidence_is_weak_and_not_zero_window():
    a = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    far = [0.0, 4.0, 8.0, 12.0, 16.0, 30.0]                 # 末态偏锚点 20 kJ ≈ 8 kT
    weak = _ev(state_ess=[10, 60, 60, 60, 60, 60], merged_f_k_kJ=far, anchor_f_k_kJ=a)
    strong_but_disagree = _ev(state_ess=[500] * 6, merged_f_k_kJ=far, anchor_f_k_kJ=a,
                              loo_delta_f_kJ=[5.0, 10.0, 15.0])
    zero = _ev(state_ess=[10, 60, 60, 60, 60, 60], merged_f_k_kJ=far, anchor_f_k_kJ=a,
               lambdas_vdw=[0.2, 0.16, 0.12, 0.08, 0.04, 0.0])
    s_weak = sr.plan_next_round([weak], completed_round=1, used_seed_ids=[0, 1, 2])[0]
    s_strong = sr.plan_next_round([strong_but_disagree], completed_round=1, used_seed_ids=[0, 1, 2])[0]
    s_zero = sr.plan_next_round([zero], completed_round=1, used_seed_ids=[0, 1, 2])[0]
    assert s_weak.center_source == sr.CENTER_MERGED_PROJECTED
    c = np.array(s_weak.center_f_k_kJ)
    dev = (c - c[0]) - (np.array(a) - a[0])
    assert np.all(np.abs(dev) <= 3.0 * KT + 1e-9)
    # 证据强：直接信 MBAR（09-23 批 cyclod outer rep1 溶剂 w1 那种情况）
    assert s_strong.center_source == sr.CENTER_MERGED
    assert np.allclose(s_strong.center_f_k_kJ, far)
    # λ=0 窗口不投影
    assert s_zero.center_source == sr.CENTER_MERGED


def test_indeterminate_window_is_treated_as_starved():
    ev = _ev(state_ess=[90, 90, 90, 90, 90, 0.0])
    specs = sr.plan_next_round([ev], completed_round=1, used_seed_ids=[0, 1, 2])
    assert len(specs) == 1 and specs[0].tilt_sign == 0
    assert specs[0].reason.startswith("starved_state_5")
