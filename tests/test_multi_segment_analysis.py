"""多采样段分析适配层的离线契约测试（无 GPU）。"""

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only
pytest.importorskip("pymbar")

import multi_segment_analysis as msa


KT = 2.494
F_TRUE = np.array([0.0, 2.0, 5.0, 9.0, 14.0])
CENTERS = np.arange(F_TRUE.size) * 1.0
WIDTH = 0.7


def _draw(n, f_k, seed):
    """从以 f_k 为偏置的混合系综采 n 帧，返回采样态能量 (K,n)。"""
    rng = np.random.default_rng(seed)
    w = np.exp(-(F_TRUE - np.asarray(f_k)) / KT)
    w /= w.sum()
    x = rng.normal(CENTERS[rng.choice(F_TRUE.size, size=n, p=w)], WIDTH)
    return np.array([
        0.5 * ((x - CENTERS[k]) / WIDTH) ** 2 * KT + F_TRUE[k]
        for k in range(F_TRUE.size)
    ])


def _two_state_reference(cross, src, u, kt):
    """独立参照：直接建 S+K MBAR，取目标态间 ΔF。"""
    from pymbar import MBAR
    counts = np.bincount(src)
    n_k = np.concatenate([counts, np.zeros(u.shape[0], dtype=int)])
    m = MBAR(np.vstack([cross, u]) / kt, n_k, verbose=False)
    S = counts.size
    return m.compute_free_energy_differences()["Delta_f"][S:, S:] * kt


def test_single_segment_collapse_is_exact():
    """S=1 必须逐比特退化成原来的 bias —— 这是所有现存单段窗口的零回归保证。"""
    S1 = _draw(800, F_TRUE, 1)
    cross, src = msa.build_cross_bias([S1], [F_TRUE], KT)
    assert cross.shape == (1, 800) and src.max() == 0

    out = msa.collapse_segments(cross, src, S1, KT)
    original_bias = msa.mixture_energy(S1, F_TRUE, KT)
    # 混合偏置与原 bias 只能差一个共模常数（规范），逐帧离散度必须是 0。
    resid = out["bias_mix_kJ_mol"] - original_bias
    assert float(np.std(resid)) < 1e-12, f"S=1 未精确退化，sd={np.std(resid):.3e}"


def test_collapse_matches_the_multistate_solve_on_two_segments():
    """塌缩路径与 S+K MBAR 的**全部**目标态间 ΔF 必须对到 1e-6。"""
    S1 = _draw(1500, np.zeros(F_TRUE.size), 2)      # 冷启动 f_k
    S2 = _draw(1500, F_TRUE, 3)                     # 重标定后的 f_k
    cross, src = msa.build_cross_bias([S1, S2], [np.zeros(F_TRUE.size), F_TRUE], KT)
    u = np.hstack([S1, S2])

    out = msa.collapse_segments(cross, src, u, KT)
    ref = _two_state_reference(cross, src, u, KT)
    worst = msa.assert_collapse_matches_multistate(out["target_delta_f_kJ_mol"], ref)
    assert worst <= msa.CROSS_CHECK_MAX_DELTA_F_KJ_MOL

    # σ 必须来自多采样态解，且是有限的成对标准误。
    sig = out["target_dDelta_f_kJ_mol"]
    assert sig.shape == (F_TRUE.size, F_TRUE.size)
    assert np.all(np.isfinite(sig)) and sig[0, -1] > 0.0


def test_mismatched_cross_check_is_refused_not_silently_accepted():
    a = np.zeros((3, 3)); b = a.copy(); b[0, 2] = 1e-3
    with pytest.raises(msa.MultiSegmentSolveError, match="拒绝把塌缩结果当等价物"):
        msa.assert_collapse_matches_multistate(a, b)


def test_zero_count_segment_is_excluded_and_recorded():
    """抽帧后某段可能一帧不剩；它必须从采样子问题剔除，且规范锚在第一个**有效**段。"""
    S1 = _draw(600, np.zeros(F_TRUE.size), 4)
    S2 = _draw(600, F_TRUE, 5)
    cross, src = msa.build_cross_bias([S1, S2], [np.zeros(F_TRUE.size), F_TRUE], KT)
    keep = np.flatnonzero(src == 1)                  # 只留第 2 段
    out = msa.collapse_segments(cross[:, keep], src[keep], np.hstack([S1, S2])[:, keep], KT)
    assert list(out["active_segments"]) == [1]
    assert out["reference_sampling_segment"] == 1
    assert out["excluded_segments"] == [
        {"segment": 0, "n_frames": 0, "reason": "zero_count_after_frame_selection"}
    ]


def test_input_errors_never_degrade_to_the_numerical_fallback():
    """能量错位/身份不一致/形状不对必须 fail-closed，不能走数值降级那条路。

    这两类如果共用一个 except，等于把静默错误重新埋回去。
    """
    S1 = _draw(300, F_TRUE, 6)
    with pytest.raises(msa.MultiSegmentInputError):          # f_k 长度不对
        msa.build_cross_bias([S1], [F_TRUE[:-1]], KT)
    with pytest.raises(msa.MultiSegmentInputError):          # 段数不一致
        msa.build_cross_bias([S1, S1], [F_TRUE], KT)
    with pytest.raises(msa.MultiSegmentInputError):          # 各段态数不一致
        msa.build_cross_bias([S1, S1[:-1]], [F_TRUE, F_TRUE[:-1]], KT)
    bad = S1.copy(); bad[0, 0] = np.nan
    with pytest.raises(msa.MultiSegmentInputError):          # 非有限值
        msa.build_cross_bias([bad], [F_TRUE], KT)

    cross, src = msa.build_cross_bias([S1], [F_TRUE], KT)
    with pytest.raises(msa.MultiSegmentInputError):          # 帧数对不上
        msa.collapse_segments(cross, src[:-1], S1, KT)
    # 输入错误绝不能是 SolveError（那一类才允许降级）
    for exc in (msa.MultiSegmentInputError,):
        assert not issubclass(exc, msa.MultiSegmentSolveError)


def test_merged_coverage_uses_the_mixture_not_the_worst_segment():
    S1 = _draw(1000, np.zeros(F_TRUE.size), 7)
    S2 = _draw(1000, F_TRUE, 8)
    fs = [np.zeros(F_TRUE.size), F_TRUE]
    cross, src = msa.build_cross_bias([S1, S2], fs, KT)
    u = np.hstack([S1, S2])
    out = msa.collapse_segments(cross, src, u, KT)
    cov = msa.merged_coverage_diagnostics(out["responsibility"], u, fs, KT)

    assert len(cov["coverage_ess"]) == F_TRUE.size
    assert cov["n_frames"] == 2000
    assert all(0.0 <= r <= 1.0 for r in cov["coverage_ess_ratio"])
    # 归一化占据的均值应当是 1（K * mean(p) 对 k 求平均 = 1）。
    assert abs(float(np.mean(cov["occupancy_normalized"])) - 1.0) < 1e-9
    # 后验概率逐帧归一。
    assert np.allclose(np.sum(out["responsibility"], axis=0), 1.0, atol=1e-10)


def test_cross_bias_must_be_rebuilt_from_the_sampling_gauge():
    """用含逐态常数（LJ 尾项）的能量重建交叉能量会破坏 logsumexp 的形状。

    真实产物实测：用 sampling_states 对账 0.0000，用 energies 得 0.735/0.373/
    0.182/0.094（随解耦单调变小，那正是尾项本身）。
    """
    S1 = _draw(500, F_TRUE, 9)
    tail = np.array([0.0, 0.4, 0.9, 1.6, 2.5])[:, None]      # 逐态常数
    right = msa.mixture_energy(S1, F_TRUE, KT)
    wrong = msa.mixture_energy(S1 + tail, F_TRUE, KT)
    d = wrong - right
    assert float(np.std(d - d.mean())) > 0.01, "逐态常数必须改变形状，不只是平移"


def test_single_segment_coverage_matches_the_existing_diagnostic():
    """S=1 时适配层的覆盖度必须与现有 `_ibs_reweighting_quality_diagnostics`
    的 `mixture_ess` 同口径 —— 否则合并会悄悄换掉门的语义。

    真实产物实测（cyclod rep1 四窗口）：192.92/289.76/603.19/261.94，两者一致。
    顺带记录：这个体系里 raw_ess 与 mixture_ess 几乎重合（目标态与采样态只差
    逐 λ 态常数，在 ESS 比值里约掉），是 IBS 的固有性质，不是实现巧合。
    """
    pytest.importorskip("openmm")
    import ibs_engine as ie

    S1 = _draw(1200, F_TRUE, 21)
    bias = msa.mixture_energy(S1, F_TRUE, KT)
    base = np.zeros(S1.shape[1])
    q = ie._ibs_reweighting_quality_diagnostics(
        S1, bias, F_TRUE, KT, sampling_kj=S1, sampling_gauge_required=True
    )
    cross, src = msa.build_cross_bias([S1], [F_TRUE], KT)
    out = msa.collapse_segments(cross, src, S1, KT)
    cov = msa.merged_coverage_diagnostics(out["responsibility"], S1, [F_TRUE], KT)

    existing = np.asarray(q["mixture_ess"], dtype=float)
    mine = np.asarray(cov["coverage_ess"], dtype=float)
    assert np.allclose(existing, mine, rtol=1e-6, atol=1e-6), (
        f"S=1 覆盖度口径不一致：现有 {existing} vs 适配层 {mine}"
    )
