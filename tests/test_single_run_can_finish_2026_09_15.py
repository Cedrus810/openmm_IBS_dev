"""单次计算必须能跑完；未标定的代理量**只报告、不拦**。

2026-09-15 用户拍板。三条事实:

1. `DONE` 的充要条件是 `precision_status == MEETS_CROSS_REPEAT_TARGET`，而它由
   `abfe_core.cross_repeat_precision` 算，需要 **≥3 个独立重复**；单次计算只有
   1 个 ⟹ 恒 `UNMEASURED` ⟹ 出口恒 `ANALYSIS_COMPLETE_PRECISION_UNMEASURED`。
   先前 `_assert_stage_result_sane` 要求 `exit ∈ (DONE, DONE_UNTRUSTED)`
   ⟹ **单次计算在构造上永远过不去**，哪怕六个窗全 `min N_eff/g=50`。
2. `min N_eff/g >= 10` 是个裸字面量（`ibs_engine.py` 的 `_ratio < 10.0`），
   与另外两个**完全不同**的量（去相关帧数下限）共用同一个数字 10，
   注释自己写着「五个窗口不足以拟合科学最优阈值」。它不是硬不变量。
3. 实测 σ：这几个 run 的 vanishing 腿总 σ = 0.63~1.19 kJ/mol（split-half 抬高后
   1.49~2.11），而 FEP 的目标是 1 kcal/mol = 4.18 kJ/mol ⟹ error bar 本来就在
   目标里面 2~6 倍，却被上面那两条判死。

⚠️ 仍然必须拦的是**身份类**判据（不是阈值）：缺窗的和是另一个量，不是 ΔG。
"""
import inspect

import numpy as np

import abfe_core
import abfe_pipeline as ap
from abfe_pipeline import ABFEPipeline
import pytest

pytestmark = pytest.mark.cpu_only


def _pipeline_stub(exit_):
    p = ABFEPipeline.__new__(ABFEPipeline)
    p._log = lambda *a, **k: None
    p._stage2_autonomous_outcome = {
        "status": "TERMINAL", "exit": exit_, "exit_emitted_by": "controller",
        "reason": "窗口 0 块数上限", "iterations_used": 6,
    }
    return p


def _complete_result(**over):
    r = {
        "total_delta_G": -20.0, "total_error": 0.63,
        "analysis_status": "ANALYSIS_COMPLETE", "precision_status": "UNMEASURED",
        "target_support_gate": {"passed": False},   # 未标定阈值，只报告
        "coverage_diagnostics": {"complete": True},
        "primary_estimator": "mbar",
    }
    r.update(over)
    return r


def test_a_single_run_is_not_blocked_by_a_cross_repeat_criterion():
    """单次计算产不出跨重复离散度 ⟹ 不得拿它当放行条件。"""
    for exit_ in ("ANALYSIS_COMPLETE_PRECISION_UNMEASURED", "NO_FEASIBLE_ACTION",
                  "HALT_FRAMES_ADMISSION_CAP"):
        _pipeline_stub(exit_)._assert_stage_result_sane(
            "Stage 2 (vanishing)", _complete_result())


def test_an_uncalibrated_support_gate_never_blocks():
    """`target_support_gate.passed=False` ⟹ 标记不可信，**不抛**。"""
    _pipeline_stub("NO_FEASIBLE_ACTION")._assert_stage_result_sane(
        "Stage 2 (vanishing)", _complete_result(
            target_support_gate={"passed": False,
                                 "raw_min_absolute_ess": 2.8}))


def test_an_incomplete_path_is_still_refused():
    """**身份类判据不放宽**：缺窗的和是另一个量，不是 ΔG。"""
    import pytest
    with pytest.raises(RuntimeError, match="ANALYSIS_INCOMPLETE"):
        _pipeline_stub("NO_FEASIBLE_ACTION")._assert_stage_result_sane(
            "Stage 2 (vanishing)",
            _complete_result(analysis_status="ANALYSIS_INCOMPLETE",
                             analysis_incomplete_reasons=["只跑了窗口 [0]"]))


def test_a_non_finite_result_is_still_refused():
    import pytest
    for bad in ({"total_delta_G": float("nan")}, {"total_error": float("inf")}):
        with pytest.raises(RuntimeError):
            _pipeline_stub("DONE")._assert_stage_result_sane(
                "Stage 2 (vanishing)", _complete_result(**bad))


def test_a_window_subset_partial_sum_is_still_refused():
    import pytest
    with pytest.raises(RuntimeError, match="部分和"):
        _pipeline_stub("DONE")._assert_stage_result_sane(
            "Stage 2 (vanishing)",
            _complete_result(subset_partial_sum_not_delta_G=True,
                             window_subset_indices=[0]))


def test_cross_repeat_precision_still_needs_three_repeats():
    """降级的是**门**，不是这个量本身 —— 它照旧诚实返回 UNMEASURED。"""
    one = abfe_core.cross_repeat_precision([-5.0])
    assert one["status"] == abfe_core.PRECISION_UNMEASURED
    assert one["n_repeats"] == 1
    three = abfe_core.cross_repeat_precision([-5.0, -5.2, -4.9])
    assert three["status"] == abfe_core.PRECISION_MEETS
    assert three["threshold_kcal_per_mol"] == 1.0


def test_the_loop_always_analyses_before_exiting():
    """判据量只存在于 stage 求解结果里 ⟹ 退出前必须跑一次全路径 ANALYZE。

    不跑它：`solver_n_frames_decorrelated` 恒 None ⟹
    `marginal_gain_stalled` 判 `NOT_ENOUGH_POINTS` ⟹ 边际刹车从没判过一次，
    只剩硬计数；而且性能分析要的逐窗支撑/σ/overlap 一条都拿不到。
    """
    src = inspect.getsource(ABFEPipeline._run_stage2_autonomous)
    tail = src.split('outcome.get("exit") not in (')[-2]
    assert "_only_window_indices=None" in tail, "退出前没有全路径求解"
    assert "_persist_inprogress_stage_result" in tail


def test_marginal_brake_is_inert_without_the_criterion():
    """钉住根因本身：判据全 None 时边际刹车判不了 —— 所以 ANALYZE 不能省。

    ⚠️ [S2-A，2026-09-17] **verdict 细分了，不变量一个字没变。**
    先前「只有 1-2 块」和「**一个输入都没有**」都报 `NOT_ENOUGH_POINTS`，
    读起来像"再跑跑看"，而真相是这道刹车**从头到尾没带过电**——两者处置相反：
    前者继续跑就会有数据，后者再跑多少轮也不会有。
    真机三个 run 的补帧刹车全程如此（判据量逐块 `[None, None, None]`），
    那几个窗口的块是被"块数硬上限"停的，不是被"没增益"停的。
    现在「全 None」报 `NO_CRITERION_INPUT_AT_ALL`，并带 `inoperative_reason`；
    **行为不变**（仍然返回"没停滞"）。
    """
    from abfe_preoptimizer import marginal_gain_stalled
    stalled, diag = marginal_gain_stalled([None, None, None, None])
    assert stalled is False, "行为必须不变：判不了 ≠ 判它停滞"
    assert diag["verdict"] == "NO_CRITERION_INPUT_AT_ALL", (
        "「一个输入都没有」必须与「点数不够」分开报 —— 两者处置相反")
    assert diag["n_points"] == 0 and diag["n_rows_seen"] == 4
    assert "没有带电" in diag["inoperative_reason"]

    # 反面：真的只是点数不够（有真实读数、但不到 3 个）⟹ 仍然是 NOT_ENOUGH_POINTS
    _s2, d2 = marginal_gain_stalled([3.0, 5.0])
    assert _s2 is False and d2["verdict"] == "NOT_ENOUGH_POINTS"
    assert "inoperative_reason" not in d2, "有真实读数时不该说没带电"
