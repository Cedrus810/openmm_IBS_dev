"""**部分和永远不许冒充 ΔG。**

真机事故 `cyclod_ligand2/rep1`（2026-09-12 跑完，09-14 查出）：

    stage2_vanishing.json   ΔG = -49.04   converged = True
    coverage_diagnostics:
        input_window_indices  = [5]
        solved_window_indices = [5]
        covered_lambda_indices = [17, 18, …, 24]      ← 只覆盖 λ 17 之后

那是**只解了末窗一个窗口**的部分和，却带着 `converged=True` 写进 stage 缓存完成
标记、被 `final_results.json` 直接采信。与同体系另一次跑（完整 21 态路径）差
**90 kJ/mol、连符号都不同**；而同一个 run 自己的全路径中间结果是 +43.00，
与对照 run 的 +44.09 只差 1.1 —— 所以那 90 不是重复间离散，是**部分和冒充了 ΔG**。

成因：09-12 那次只给"子集求解**失败**"开了 carve-out，子集求解**成功**时结果
照样一路流下去。

两道互相独立的门：
  ① 上游：窗口子集跑一律标 `window_subset_no_stage_verdict` + `converged=False`
  ② 下游：`_assert_stage_result_sane` 只看"覆盖了几个 λ 态"，与上游标记无关
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_pipeline import ABFEPipeline  # noqa: E402


class _Stub:
    """只借 `_assert_stage_result_sane` —— 它只读 result 和 `_last_run_config`。"""
    _last_run_config = {}
    _log = staticmethod(lambda *a, **k: None)
    _assert_stage_result_sane = ABFEPipeline._assert_stage_result_sane
    _stage_quality_failure_details = ABFEPipeline._stage_quality_failure_details
    _format_stage_quality_failure_details = (
        ABFEPipeline._format_stage_quality_failure_details)


def _rep1_shaped(n_states=24, covered=None):
    """复刻真机那份结果的形状。"""
    return {
        "stage": "vanishing", "converged": True,
        "total_delta_G": -49.040107224763666, "total_error": 0.9866,
        "n_states": n_states,
        "coverage_diagnostics": {
            "input_window_indices": [5], "valid_window_indices": [5],
            "solved_window_indices": [5],
            "covered_lambda_indices": (
                covered if covered is not None else list(range(17, n_states))),
        },
    }


def test_the_real_incident_shape_is_now_refused():
    with pytest.raises(RuntimeError, match="不是 ΔG"):
        _Stub()._assert_stage_result_sane("Stage 2 (vanishing)", _rep1_shaped())


def test_a_subset_marked_result_is_refused_even_without_coverage_info():
    """上游标记这一道单独就够 —— 不依赖 coverage 诊断存在。"""
    r = {"stage": "vanishing", "converged": False, "total_delta_G": 1.0,
         "total_error": 0.5,
         "stage_scope": "window_subset_no_stage_verdict",
         "window_subset_indices": [5]}
    with pytest.raises(RuntimeError, match="部分和"):
        _Stub()._assert_stage_result_sane("Stage 2 (vanishing)", r)


def test_incomplete_coverage_is_refused_even_if_nobody_marked_it():
    """下游这一道与上游标记**独立** —— 上游漏标照样拦得住。"""
    r = _rep1_shaped(covered=[i for i in range(24) if i != 7])
    with pytest.raises(RuntimeError, match="缺窗口的和不是 ΔG"):
        _Stub()._assert_stage_result_sane("Stage 2 (vanishing)", r)


def test_a_fully_covered_result_still_passes_this_gate():
    """别把门修成谁都过不去：完整覆盖的结果照常放行到后面的判据。"""
    r = _rep1_shaped(covered=list(range(24)))
    r["target_support_gate"] = {"passed": True, "failed_checks": []}
    r["min_overlap"] = 0.2
    _Stub()._assert_stage_result_sane("Stage 2 (vanishing)", r)


def test_the_subset_run_is_pinned_to_not_converged_upstream():
    """上游：窗口子集跑一律钉成 `converged=False` —— 成功也不例外。"""
    import inspect
    src = inspect.getsource(ABFEPipeline._run_dual_lambda_stage)
    # 判据必须真的挂在 `only_window_indices is not None` 上 —— 把条件改成 False
    # 之类的"绕过"必须被抓到，所以这里钉的是**整段连续文本**。
    assert (
        'if only_window_indices is not None:\n'
        '            stage_result["stage_scope"] = "window_subset_no_stage_verdict"'
    ) in src
    assert 'stage_result["converged"] = False' in src
    assert 'subset_partial_sum_not_delta_G' in src
