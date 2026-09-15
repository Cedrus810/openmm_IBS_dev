"""CTL-03：**被 rewindow 取代的父窗不许抢在子窗前决策。**

父窗已经退出求解覆盖（子系综接管了它那段 λ），它的旧 warmup / f_k 状态只是历史
记录。先前它照样能当 `earliest`，于是分支 1e（父窗预热预算耗尽 ⟹ 发
`INSERT_LAMBDA` / `NO_ACTION`）排在未完成子窗的路由**之前** —— 用一个已经不在
覆盖里的窗口的旧状态，挡住子窗自己的补帧。
"""
import json
import os
import pathlib
import sys

import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abfe_preoptimizer import Stage2RepairController  # noqa: E402

from test_rewindow_sampling_units import _with_rewindow  # noqa: E402


def _exhaust_parent_warmup(run, parent=1):
    """把父窗做成「预热预算耗尽 + f_k 未判定」—— 1e 分支的触发条件。"""
    conv = (pathlib.Path(run) / "vanishing"
            / f"dual_window_{parent}_vdw_convergence.json")
    d = json.loads(conv.read_text())
    d["bias_warmup"]["warmup_budget_ledger"] = {
        "learning_steps": 555000, "cumulative_cap_steps": 555000}
    conv.write_text(json.dumps(d))
    st = (pathlib.Path(run) / "checkpoints"
          / f"ibs_state_vdw_window_{parent}.json")
    s = json.loads(st.read_text())
    s["f_k_evidence_status"] = "indeterminate"
    s["bias_status"] = "frozen_validation_indeterminate"
    st.write_text(json.dumps(s))
    # 还要让它在三态分类里判 PROBLEM —— `self_verdict` 在 `phase` **之前**被检查，
    # 只改 bias_status 的话它仍然是 ANALYSIS_ELIGIBLE（= ELIGIBLE），1e 根本不触发。
    ss = (pathlib.Path(run) / "vanishing"
          / f"dual_window_{parent}_vdw_self_support.json")
    d2 = json.loads(ss.read_text())
    d2.update({"verdict": "INSUFFICIENT_DATA", "sufficient": False})
    ss.write_text(json.dumps(d2))


def test_an_incomplete_child_wins_over_its_exhausted_parent(tmp_path):
    run = _with_rewindow(
        tmp_path,
        child_states=[("ANALYSIS_ELIGIBLE", 250000), ("ANALYSIS_ELIGIBLE", 250000)],
        solver_decorr=[200, 11],          # 子窗 1 只有 11/20，确实缺帧
    )
    _exhaust_parent_warmup(run, parent=1)

    plan = Stage2RepairController(run, "vanishing").decide()
    assert plan["action"] == "RUN_PRODUCTION", plan["reason"]
    assert plan["unit_id"] == "rw:abc123:1", (
        f"父窗的旧预热状态又把子窗挡住了：{plan['reason'][:160]}"
    )


def test_a_replaced_parent_is_never_the_earliest_unresolved_window(tmp_path):
    run = _with_rewindow(
        tmp_path,
        child_states=[("ANALYSIS_ELIGIBLE", 250000), ("ANALYSIS_ELIGIBLE", 250000)],
        solver_decorr=[200, 200],
    )
    _exhaust_parent_warmup(run, parent=1)
    view = Stage2RepairController(run, "vanishing").read()
    plan = Stage2RepairController(run, "vanishing").decide(view)

    assert 1 in (view["immutable_rewindow"]["parents_done"] or [])
    assert plan.get("blocked_by") != 1
    # 父窗既已退出覆盖，它不该成为路由目标
    assert plan.get("windows") != [1] or plan.get("unit_id"), plan["reason"]
