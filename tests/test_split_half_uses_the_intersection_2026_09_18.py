"""半程窗口集合少于全量时，split-half 必须**只丢缺的那些窗口**，不是整段关掉。

原行为：`set(seg_first) != set(seg_full)` ⟹ 整个 stage 的漂移诊断
`available=False`。实测后果（09-18 benchmark，24 run）：

  · complex 腿 20 条里 **12 条**的漂移诊断完全是暗的；溶剂腿 0/22；
  · 失败的恰好是最弱窗口已经贴/破样本下限的那些 ——
    `min_decorrelated_samples` 中位 **18.0** vs 可用组 26.5，
    12 条里 6 条在**全量**数据上就已经低于门 20。

机制：半程帧数砍半 ⟹ 那个边缘窗口被求解器 `continue` 掉 ⟹ 集合不一致
⟹ 整段关闭。**与「最需要漂移检验」反相关，而且是构造出来的 fail-open。**

`sigma_inflated_from_split_half` 早就逐窗处理「这个窗口没有漂移证据」
（`sigma_floor_unavailable`），所以交集结果可以直接喂给它。

⚠️ 总量项**不能**一起救：窗口集合不同 ⟹ 两个半程积的不是同一段 λ，
相减没有物理意义。集合不全时必须是 `None`，**不是 0** ——
0 会被读成「实测总漂移为零」。

见 docs/archive/STAGE2_OFFLINE_FORENSICS_2026-09-18.md §9。
"""
import numpy as np
import pytest

import ibs_engine as ie

pytestmark = pytest.mark.cpu_only


def _window(n_states=3, n_frames=40):
    """最小窗口：只要能过 `_slice_window_frames`，数值不参与本测试的断言。"""
    return {
        "u_kn": np.zeros((n_states, n_frames)),
        "bias_energies": np.zeros(n_frames),
        "base_energies": np.zeros(n_frames),
    }


def _result(segments, total_error=0.7):
    return {
        "covariance_chain_segments": [
            {"window_index": w, "delta_G_kJ_mol": dg, "uncertainty_kJ_mol": 0.5}
            for w, dg in segments
        ],
        "total_error": total_error,
        "total_delta_G": sum(dg for _, dg in segments),
    }


def _patch_halves(monkeypatch, first, second):
    """半程求解替换成固定返回值：本测试只验集合逻辑，不验 MBAR。"""
    calls = iter([first, second])
    monkeypatch.setattr(ie, "solve_stage_integrated", lambda *a, **k: next(calls))


def test_a_window_missing_from_the_halves_no_longer_blinds_the_whole_stage(monkeypatch):
    full = _result([(0, 10.0), (1, 20.0)])
    _patch_halves(monkeypatch, _result([(0, 9.0)]), _result([(0, 11.0)]))

    d = ie.split_half_drift_diagnostics([_window(), _window()], 2.4943, full, {})

    assert d["available"] is True, "有共同窗口就必须出结果，不能整段关掉"
    assert d["windows_missing_from_halves"] == [1]
    assert d["n_windows_compared"] == 1
    assert d["coverage_complete"] is False
    assert [w["window_index"] for w in d["per_window"]] == [0]
    assert d["per_window"][0]["drift_kJ_mol"] == pytest.approx(2.0)


def test_totals_are_none_not_zero_when_the_window_sets_differ(monkeypatch):
    """集合不同 ⟹ 两半积的 λ 段不同 ⟹ 总量相减无意义。0 会被读成「无漂移」。"""
    full = _result([(0, 10.0), (1, 20.0)])
    _patch_halves(monkeypatch, _result([(0, 9.0)]), _result([(0, 11.0)]))

    d = ie.split_half_drift_diagnostics([_window(), _window()], 2.4943, full, {})

    assert d["total_drift_kJ_mol"] is None
    assert d["total_drift_over_2sigma"] is None
    assert d["total_delta_G_first_half_kJ_mol"] is None
    assert d["total_delta_G_second_half_kJ_mol"] is None


def test_complete_coverage_still_reports_the_totals(monkeypatch):
    """集合一致时行为**逐字不变**：总量照报（这是 09-15 修的那条，别回退）。"""
    full = _result([(0, 10.0), (1, 20.0)])
    _patch_halves(
        monkeypatch,
        _result([(0, 9.0), (1, 19.0)]),
        _result([(0, 11.0), (1, 21.0)]),
    )

    d = ie.split_half_drift_diagnostics([_window(), _window()], 2.4943, full, {})

    assert d["coverage_complete"] is True
    assert d["windows_missing_from_halves"] == []
    assert d["n_windows_compared"] == 2
    assert d["total_drift_kJ_mol"] == pytest.approx(4.0)   # 32 - 28
    assert d["total_delta_G_first_half_kJ_mol"] == pytest.approx(28.0)


def test_no_common_window_at_all_is_still_unavailable_with_its_own_reason(monkeypatch):
    """完全没有共同窗口是**另一回事**：半程数据下整条路径都解不出来。
    它仍然 unavailable，但理由必须能和「缺了几个窗口」区分开 ——
    实测有 6 条腿属于这一类，它们不是代码 bug，是预算不够。"""
    full = _result([(0, 10.0), (1, 20.0)])
    _patch_halves(monkeypatch, _result([]), _result([]))

    d = ie.split_half_drift_diagnostics([_window(), _window()], 2.4943, full, {})

    assert d["available"] is False
    assert "没有任何共同窗口" in d["reason"]
    assert d["windows_missing_from_halves"] == [0, 1]


def test_the_sigma_floor_consumes_the_intersection_and_names_what_it_lacks(monkeypatch):
    """交集结果喂给 σ 下界：有证据的窗口膨胀，缺的窗口如实记名、不膨胀。"""
    full = _result([(0, 10.0), (1, 20.0)])
    _patch_halves(monkeypatch, _result([(0, 9.0)]), _result([(0, 11.0)]))
    d = ie.split_half_drift_diagnostics([_window(), _window()], 2.4943, full, {})

    infl = ie.sigma_inflated_from_split_half(full, d)

    assert infl["available"] is True
    assert infl["windows_with_sigma_floor_unavailable"] == [1]
    assert infl["sigma_floor_coverage_complete"] is False
    rows = {r["window_index"]: r for r in infl["per_window"]}
    # w0 有 2.0 kJ/mol 漂移 ⟹ 下界 1.0 > σ_MBAR 0.5 ⟹ 膨胀
    assert rows[0]["sigma_effective_kJ_mol"] == pytest.approx(1.0)
    assert rows[0]["inflated"] is True
    # w1 无证据 ⟹ 维持 MBAR 值，且标记出来（不是「实测漂移为 0」）
    assert rows[1]["sigma_effective_kJ_mol"] == pytest.approx(0.5)
    assert rows[1]["sigma_floor_unavailable"] is True
