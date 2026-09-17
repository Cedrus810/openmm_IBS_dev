"""瓶颈态 (g, η, ratio) 观测量的数据契约（2026-09-17，用户拍板 #3）。

射程判据 `R_now × H` 假定 η = N_eff/N 与 g 恒定，两个假设实测都朝不利方向走
（g 实测 12.25→17.63 = 1.44×）。要把实测增长纳进可达性判断，前提是**先逐块记下
g 与 η** —— 而历史台账原来只有步数、去相关帧数、ratio，连 g 都没有。

⚠️ 本轮**只记录、不改任何门**（先记 → 可回填 → shadow 对比 → 数据够了再启用），
所以这份文件里一个阈值都没有，测的全是**口径**。
"""
import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

from abfe_preoptimizer import _bottleneck_observation


def _chk(*, k, g, n_eff, n_in, ratio, raw):
    return {
        "worst_state_by_n_eff_over_g": k,
        "statistical_inefficiency_per_lambda": g,
        "n_eff_per_state": n_eff,
        "n_eff_input_n_frames_per_state": n_in,
        "n_eff_over_g_per_state": ratio,
        "n_eff_frame_set": "segments_kept_by_decorrelation",
        "n_eff_frame_set_n_frames_raw": raw,
    }


def test_the_two_forced_identities_hold():
    """强制校验：`eta == n_eff / n_eff_input_n_frames` 且 `ratio == n_eff / g`。"""
    o = _bottleneck_observation(_chk(
        k=1, g=[1.0, 17.63], n_eff=[9.0, 123.4], n_in=[1000, 998],
        ratio=[9.0, 123.4 / 17.63], raw=1000))
    assert o["bottleneck_state"] == 1
    assert o["bottleneck_eta"] == pytest.approx(123.4 / 998)
    assert o["bottleneck_ratio"] == pytest.approx(o["bottleneck_n_eff"] / o["bottleneck_g"])


def test_eta_is_immune_to_both_decorrelated_counts():
    """**去相关帧数不得进 η 的分母** —— 自检侧与求解器侧实测差 2–8 倍。

    它们是 solver eligibility 的量，与「权重剖面的效率」无关。分母混错一次，
    η 就系统性差几倍，而且这个错会一直躺在历史里没人发现。
    """
    base = _chk(k=0, g=[10.0], n_eff=[50.0], n_in=[500],
                ratio=[5.0], raw=500)
    eta0 = _bottleneck_observation(base)["bottleneck_eta"]
    for extra in ({"n_frames_decorrelated": 81},
                  {"solver_n_frames_decorrelated": 37},
                  {"n_frames_decorrelated": 3, "solver_n_frames_decorrelated": 999}):
        o = _bottleneck_observation({**base, **extra})
        assert o["bottleneck_eta"] == eta0, extra
    assert eta0 == pytest.approx(50.0 / 500)


def test_eta_uses_the_finite_masked_count_not_the_raw_frame_set_size():
    """注入非有限帧 ⟹ η 的分母是**瓶颈态 finite 之后**的帧数，不是 `_bias_fs.size`。

    这是最隐蔽的那一层：不同态的非有限帧可以不同，而 `n_eff_frame_set_n_frames_raw`
    是 finite mask **之前**的数。拿它当分母只在"所有帧对 k* 都有限"时才碰巧对，
    否则 η 被系统性压低 —— 而且永远偏向「看起来更差」。
    """
    o = _bottleneck_observation(_chk(
        k=0, g=[10.0], n_eff=[50.0],
        n_in=[400],          # 1000 帧里只有 400 帧对 k* 有限
        ratio=[5.0], raw=1000))
    assert o["bottleneck_eta"] == pytest.approx(50.0 / 400)
    assert o["bottleneck_eta"] != pytest.approx(50.0 / 1000)
    # 原始帧集大小照样落盘（供审计），只是**不是**分母
    assert o["n_eff_frame_set_n_frames_raw"] == 1000
    assert o["bottleneck_n_eff_input_n_frames"] == 400


def test_missing_keys_are_unknown_never_zero():
    """老产物缺键 ⟹ 全 `None`（UNKNOWN）。**不填 0** —— 0 会被读成「效率为零」。"""
    for bad in ({}, None, {"worst_state_by_n_eff_over_g": None},
                {"worst_state_by_n_eff_over_g": 5, "n_eff_per_state": [1.0]}):
        o = _bottleneck_observation(bad)
        assert o["bottleneck_eta"] is None, bad
        assert o["bottleneck_g"] is None, bad


def test_the_bottleneck_is_the_worst_ratio_not_the_worst_numerator():
    """`k*` 取 `worst_state_by_n_eff_over_g`，**不是** `worst_state_by_n_eff`。

    分子最小 ≠ 比值最小（g 逐态不同）。取错会让整条 g/η 曲线描述另一个态。
    """
    chk = _chk(k=1, g=[100.0, 2.0], n_eff=[80.0, 10.0], n_in=[1000, 1000],
               ratio=[0.8, 5.0], raw=1000)
    chk["worst_state_by_n_eff"] = 1          # 分子最小的是 1
    chk["worst_state_by_n_eff_over_g"] = 0   # 比值最小的是 0
    o = _bottleneck_observation(chk)
    assert o["bottleneck_state"] == 0
    assert o["bottleneck_g"] == 100.0


def test_the_writer_emits_the_per_state_finite_count():
    """写侧（`ibs_engine`）必须落 `n_eff_input_n_frames_per_state`，否则读侧恒 UNKNOWN。"""
    import inspect
    import ibs_engine as ie
    src = inspect.getsource(ie)
    assert '"n_eff_input_n_frames_per_state"' in src
    assert "n_eff_input_n_frames_per_state.append(int(_sr.size))" in src, (
        "分母必须来自 finite mask 之后的 `_sr`，不是 `_bias_fs`")
