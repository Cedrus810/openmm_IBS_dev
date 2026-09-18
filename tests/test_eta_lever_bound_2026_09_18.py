"""η ≤ 1 的硬上界，以及它在归因/路由两层的贯穿。

**推导（不是阈值，是定义的推论）**：验收量
``R = min_k N_eff,k / g_k``（`ibs_engine.window_self_support_check`，门 10.0）。
记 N = 真正进 N_eff 的帧数、η = N_eff/N，则 ``R = η·N/g``；而 Kish ESS
``N_eff = (Σw)²/Σw²`` 经 Cauchy–Schwarz 有 ``(Σw)² ≤ N·Σw²`` ⟹ ``N_eff ≤ N``
⟹ **η ≤ 1** ⟹ **R ≤ N/g**。

于是 `N/g < 门` 时，**只改 η 的动作全部被证明够不到门** —— 插 λ、重标定/重学
f_k、有界重窗都在此列。此前它们被归到 `STRUCTURAL`，而那一档**授权的正是缩跨度**。
"""
import os
import sys
import tempfile

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import (  # noqa: E402
    SUPPORT_FAILURE_N_OVER_G_BOUND,
    SUPPORT_FAILURE_SAMPLE_SIZE,
    SUPPORT_FAILURE_STRUCTURAL,
    eta_lever_can_reach,
    support_failure_attribution,
)
from test_stage2_subwindow_terminal_retirement_2026_09_18 import (  # noqa: E402
    _ctl, _view, _w,
)


def _bad_window(**kw):
    w = dict(self_verdict="INSUFFICIENT_DATA", self_sufficient=False,
             self_n_frames_decorrelated=888, self_min_frames=10,
             min_n_eff_over_g=4.0, self_n_eff_over_g_eligible=10.0)
    w.update(kw)
    return _w(0, **w)


# ------------------------------------------------------------------ 上界本身


def test_bound_is_n_over_g():
    assert eta_lever_can_reach(50, 10.0, 10.0) is False     # 5 < 10
    assert eta_lever_can_reach(500, 10.0, 10.0) is True     # 50 >= 10
    assert eta_lever_can_reach(100, 10.0, 10.0) is True     # 恰好相等 ⟹ 不排除


@pytest.mark.parametrize("n,g,t", [
    (None, 10.0, 10.0), (50, None, 10.0), (50, 10.0, None),
    (50, 0.0, 10.0),                      # g<=0 判不了
    (0, 10.0, 10.0),                      # 一帧都没有 ⟹ 判不了，不是"够不着"
    (float("nan"), 10.0, 10.0),
])
def test_missing_or_degenerate_inputs_give_unknown_not_false(n, g, t):
    """判不了必须是 `None`。判成 `False` 会把数据缺口升级成"证明够不着"。"""
    assert eta_lever_can_reach(n, g, t) is None


# ------------------------------------------------------- 归因层：第四态不被压掉


def test_weight_collapse_below_the_bound_is_not_structural():
    """top1% 塌缩 + `N/g` 够不着 ⟹ `N_OVER_G_BOUND`，**不是** `STRUCTURAL`。

    判成 STRUCTURAL 就等于授权缩跨度，而缩跨度在这里已经被 η≤1 证伪。
    """
    got = support_failure_attribution(
        "INSUFFICIENT_DATA", "top1pct_veto",
        n_eff_input_n_frames=50, bottleneck_g=10.0, n_eff_over_g_target=10.0)
    assert got == SUPPORT_FAILURE_N_OVER_G_BOUND


def test_weight_collapse_above_the_bound_stays_structural():
    """`N/g` 够得着 ⟹ 缺口确实在 η ⟹ 维持 `STRUCTURAL`（既有行为不变）。"""
    got = support_failure_attribution(
        "INSUFFICIENT_DATA", "top1pct_veto",
        n_eff_input_n_frames=500, bottleneck_g=10.0, n_eff_over_g_target=10.0)
    assert got == SUPPORT_FAILURE_STRUCTURAL


def test_no_readings_reproduces_the_old_attribution():
    """两个读数缺失 ⟹ 逐位回到旧行为（老产物不受影响）。"""
    assert support_failure_attribution(
        "INSUFFICIENT_DATA", "top1pct_veto") == SUPPORT_FAILURE_STRUCTURAL


def test_hard_sample_size_evidence_still_wins():
    """帧数没到求解器下限是**硬证据**，排在这个上界之前。"""
    got = support_failure_attribution(
        "HARD_INSUFFICIENT", "solver_eligibility",
        n_eff_input_n_frames=50, bottleneck_g=10.0, n_eff_over_g_target=10.0)
    assert got == SUPPORT_FAILURE_SAMPLE_SIZE


# ------------------------------------------------------------ 路由层：贯穿到动作


def test_below_the_bound_routes_to_frames_not_to_span_shrinking():
    """够不着的窗口**不得**收到缩跨度动作（插 λ / 拆窗 / 有界重窗）。"""
    ctl = _ctl(); ctl.checkpoint_dir = tempfile.mkdtemp()
    p = ctl._decide_once(_view(
        [_bad_window(self_verdict_source="top1pct_veto",
                     bottleneck_n_eff_input_n_frames=50, bottleneck_g=10.0),
         _w(1)]))
    assert p["action"] == "RUN_PRODUCTION", (
        f"够不着门时发了 {p['action']}，而它只动 η —— 已被 η≤1 证伪"
    )
    assert "N/g" in p["reason"]


def test_weight_collapse_prefers_relearning_f_k_over_shrinking_span():
    """η 确实是瓶颈时，先用**没有副作用**的那个 η 杠杆。

    插 λ 在 model B 下必然把溢出推给末窗（后置断言
    `new_ranges = ranges[:-1] + [(tail_lo, tail_hi+n)]`）⟹ 它是拿末窗的 η
    去补这个窗口的 η。换 f_k 没有这个代价。
    """
    ctl = _ctl(); ctl.checkpoint_dir = tempfile.mkdtemp()
    p = ctl._decide_once(_view(
        [_bad_window(self_verdict_source="top1pct_veto",
                     bottleneck_n_eff_input_n_frames=500, bottleneck_g=10.0,
                     warmup_steps_left=5_000_000),
         _w(1)]))
    assert p["action"] == "RELEARN_FK_EPOCH"
    assert "top1pct_veto" in p["reason"]


def test_relearn_is_one_shot_then_falls_back_to_span_shrinking():
    """一个窗口只给一次全新 Epoch；用过就落回缩跨度族。"""
    import abfe_preoptimizer as pre
    ctl = _ctl(); ctl.checkpoint_dir = tempfile.mkdtemp()
    view = _view([_bad_window(self_verdict_source="top1pct_veto",
                              bottleneck_n_eff_input_n_frames=500,
                              bottleneck_g=10.0,
                              warmup_steps_left=5_000_000),
                  _w(1)])
    assert ctl._decide_once(view)["action"] == "RELEARN_FK_EPOCH"
    pre.mark_relearn_epoch_consumed(ctl.checkpoint_dir, 1, 0, detail={})
    assert ctl._decide_once(view)["action"] != "RELEARN_FK_EPOCH"
