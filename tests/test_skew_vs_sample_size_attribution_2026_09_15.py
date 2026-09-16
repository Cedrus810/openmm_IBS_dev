"""`min_n_eff_over_g` 是**偏斜**，不是样本量不足。

真机 brd4_ligand1/rep2 的反例：

    win1  n_decorr = 888（地板 10 的 88 倍）、min N_eff/g = 8.68
          ⟹ 帧一点都不缺，缺的是权重压不到目标态上

先前 `_SAMPLE_SIZE_VERDICT_SOURCES` 把 `min_n_eff_over_g` 算成"样本量问题、加帧对症"
⟹ 控制器连发 4 块 `RUN_PRODUCTION` 给 win1，配额烧光 → NO_FEASIBLE_ACTION，
而真正缺窗的 win4 全程排在 `blocked_by_upstream` 里、一次都没被看过。

本仓 §5.1 实测：同分布 250k→1M 让 top1% 从 0.545 涨到 0.762、ESS 比值**反而更差**；
而重标定一次 rawESS 27.5→503。⟹ 同分布加帧治不了偏斜。

⚠️ `solver_eligibility` **必须留在样本量那一档**：长 τ 的解耦端窗口是真的帧不够，
插 λ 不缩短构象慢模态的 τ_int。这次只动 `min_n_eff_over_g` 一个。
"""
from abfe_preoptimizer import (
    _SAMPLE_SIZE_VERDICT_SOURCES,
    support_failure_is_skew as is_skew,
)


def test_a_low_ratio_with_plenty_of_frames_is_skew():
    """win1 的形状：帧多得离谱、比值低 ⟹ 偏斜。"""
    assert is_skew("INSUFFICIENT_DATA", "min_n_eff_over_g",
                   n_decorrelated=888, min_frames=10) is True


def test_frame_starvation_is_still_sample_size():
    """win4 的形状：n_decorr=7 < 10 ⟹ 真缺帧，加帧**正是**对症动作。

    插 λ 不缩短构象慢模态的 τ_int —— 这一档判错会把解耦端窗口送去改布局。
    """
    assert is_skew("HARD_INSUFFICIENT", "solver_eligibility",
                   n_decorrelated=7, min_frames=10) is False
    assert "solver_eligibility" in _SAMPLE_SIZE_VERDICT_SOURCES


def test_min_n_eff_over_g_is_no_longer_called_a_sample_size_source():
    assert "min_n_eff_over_g" not in _SAMPLE_SIZE_VERDICT_SOURCES


def test_top1pct_veto_stays_skew():
    assert is_skew("INSUFFICIENT_DATA", "top1pct_veto",
                   n_decorrelated=500, min_frames=10) is True


def test_a_passing_window_is_never_skew():
    """通过的窗口同样带 `verdict_source` —— 只看来源会把健康窗口标成失败。"""
    assert is_skew("ANALYSIS_ELIGIBLE", "min_n_eff_over_g",
                   n_decorrelated=900, min_frames=10) is False


def test_the_helper_verifies_frames_itself_instead_of_trusting_the_writer():
    """写侧的覆盖顺序保证 `min_n_eff_over_g` ⟹ 帧数已够；但判据自己再验一次。

    不把正确性押在另一个文件的不变量上 —— 那条链断了就是静默误路由。
    """
    # 假设写侧的不变量破了：来源是比值、帧数却没到地板 ⟹ 仍按样本量处理
    assert is_skew("INSUFFICIENT_DATA", "min_n_eff_over_g",
                   n_decorrelated=3, min_frames=10) is False
    # 读不到帧数（老产物）⟹ 退回只看来源，判偏斜
    assert is_skew("INSUFFICIENT_DATA", "min_n_eff_over_g") is True


def test_both_call_sites_pass_the_frame_counts():
    import inspect

    from abfe_preoptimizer import Stage2RepairController
    for src in (inspect.getsource(Stage2RepairController.decide),
                inspect.getsource(Stage2RepairController._read_single_stage)
                if hasattr(Stage2RepairController, "_read_single_stage")
                else inspect.getsource(Stage2RepairController)):
        if "support_failure_is_skew(" in src:
            assert "n_decorrelated=" in src, "调用点没把帧数传进去"
