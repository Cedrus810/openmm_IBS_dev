"""固定预算模式下，一个窗口"无结论"不得吞掉整个 run。

真机 cmet_ligand2 B 臂 seed=20260916：`stage2_autonomous_controller=false`
（固定预算对照实验）⟹ `route_to_controller` 为假 ⟹
`IBSValidationBudgetIndeterminateError` 直接 `raise` 炸穿 `run_full_pipeline`，
窗口 6 之后一个窗口都没跑，整个 run 报废。

而这个信号的语义是「**没测出来**，对这份 f_k 无结论」——既不是收敛也不是不收敛。

语义（用户 2026-09-16 定死，四条都钉在下面）：
  1. 记成 `indeterminate`，完整诊断保留；
  2. 继续跑其余窗口；
  3. **禁止**当成失败值/零值，**禁止**从配对分析里静默删除；
  4. 关键对齐窗口无结论 ⟹ 该 run 的配对指标"不可判定"，不许硬算差值
     （本层的体现：`analysis_status=ANALYSIS_INCOMPLETE` 且**不产出 ΔG**）。

「异常路由/状态收集开启，自适应追加预算关闭」—— 所以这里**不得**出现延长预算、
改 f_k、动布局的动作。
"""
import os
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import abfe_pipeline  # noqa: E402
import ibs_engine  # noqa: E402


def test_the_exception_carries_a_machine_readable_window_index():
    """先前只有中文消息里写着窗口号，调用方只能去正则它。"""
    e = ibs_engine.IBSValidationBudgetIndeterminateError(
        "窗口 6 的冻结验证…", {"status": "x"}, window_idx=6)
    assert e.window_idx == 6
    assert e.diagnostics == {"status": "x"}
    # 不传时是 None（旧调用点不变），调用方据此 fail-closed 而不是猜 0。
    assert ibs_engine.IBSValidationBudgetIndeterminateError("m", {}).window_idx is None


def _drive(pipeline, run_once, n_windows, *, route_to_controller):
    """跑 `_run_stage2_with_path_evolution` 的非演化早退路径。"""
    return abfe_pipeline.ABFEPipeline._run_stage2_with_path_evolution(
        pipeline, run_once,
        lambdas_var=[1.0 - i * (1.0 / (n_windows * 3)) for i in range(n_windows * 3)],
        window_ranges=[(3 * i, 3 * i + 3) for i in range(n_windows)],
        checkpoint_dir="/nonexistent", preopt_file="/nonexistent",
        repair_policy="non_mutating_v1",          # ⟹ 走 `_guarded_once` 早退
        route_to_controller=route_to_controller,
    )


class _Stub:
    """只提供 `_run_stage2_with_path_evolution` 这条路径用得到的东西。"""
    def __init__(self):
        self.logs = []

    def _log(self, msg):
        self.logs.append(str(msg))


def test_a_later_window_still_runs_and_the_hole_is_reported():
    calls = []

    def run_once(n, lam, ranges, **kw):
        only = kw.get("_only_window_indices")
        calls.append(only)
        if only is None:                       # 第一次：跑到窗口 1 无结论
            raise ibs_engine.IBSValidationBudgetIndeterminateError(
                "窗口 1 …", {"status": "validation_budget_exhausted_indeterminate"},
                window_idx=1)
        return {"total_delta_G": 42.0, "total_error": 1.0,
                "analysis_status": "ANALYSIS_COMPLETE"}

    res, _lam, _rng = _drive(_Stub(), run_once, 3, route_to_controller=False)

    # (2) 其余窗口真的跑了 —— 第二次调用带着 [2]
    assert calls == [None, [2]], calls
    # (1)(3) 洞被显式记下来，不是删掉也不是填 0
    assert [w["window_idx"] for w in res["indeterminate_windows"]] == [1]
    assert res["indeterminate_windows"][0]["diagnostics"]["status"] == \
        "validation_budget_exhausted_indeterminate"
    # (4) 有洞就不产出 ΔG，状态是"没跑完"而不是"跑完了但很差"
    assert res["analysis_status"] == abfe_pipeline.ANALYSIS_INCOMPLETE
    assert "total_delta_G" not in res and "total_error" not in res


def test_the_last_window_being_indeterminate_still_returns_a_marked_result():
    """真机形状：无结论的正是最后一个窗口 ⟹ 没有"其余窗口"可跑。"""
    def run_once(n, lam, ranges, **kw):
        raise ibs_engine.IBSValidationBudgetIndeterminateError(
            "窗口 2 …", {"status": "s"}, window_idx=2)

    res, _lam, _rng = _drive(_Stub(), run_once, 3, route_to_controller=False)
    assert [w["window_idx"] for w in res["indeterminate_windows"]] == [2]
    assert res["analysis_status"] == abfe_pipeline.ANALYSIS_INCOMPLETE
    assert "total_delta_G" not in res


def test_a_window_without_an_index_is_not_guessed():
    """定位不到窗口就 fail closed —— 不许猜 0、不许吞。"""
    def run_once(n, lam, ranges, **kw):
        raise ibs_engine.IBSValidationBudgetIndeterminateError("窗口 ? …", {})

    with pytest.raises(ibs_engine.IBSValidationBudgetIndeterminateError):
        _drive(_Stub(), run_once, 2, route_to_controller=False)


def test_the_same_window_twice_stops_instead_of_looping():
    """子集机制没推动它 ⟹ 如实抛，不转圈烧 GPU。"""
    def run_once(n, lam, ranges, **kw):
        raise ibs_engine.IBSValidationBudgetIndeterminateError(
            "窗口 0 …", {}, window_idx=0)

    with pytest.raises(ibs_engine.IBSValidationBudgetIndeterminateError):
        _drive(_Stub(), run_once, 3, route_to_controller=False)


def test_with_a_controller_downstream_the_signal_is_still_routed_not_collected():
    """自治控制器在下游时行为**逐字不变**：交回去重判，不在这里收集。"""
    def run_once(n, lam, ranges, **kw):
        raise ibs_engine.IBSValidationBudgetIndeterminateError(
            "窗口 1 …", {}, window_idx=1)

    res, _lam, _rng = _drive(_Stub(), run_once, 3, route_to_controller=True)
    assert res.get("routing_signal") == "LOCAL_VALIDATION_CAP", res
    assert "indeterminate_windows" not in res


def test_no_budget_extension_is_ever_attempted():
    """固定预算：收集路径里不得出现延长预算/改 f_k/动布局的措辞或动作。"""
    stub = _Stub()

    def run_once(n, lam, ranges, **kw):
        if kw.get("_only_window_indices") is None:
            raise ibs_engine.IBSValidationBudgetIndeterminateError(
                "窗口 0 …", {}, window_idx=0)
        return {"total_delta_G": 1.0, "analysis_status": "ANALYSIS_COMPLETE"}

    _drive(stub, run_once, 2, route_to_controller=False)
    joined = "\n".join(stub.logs)
    assert "不延长预算" in joined and "不改 f_k" in joined, joined
