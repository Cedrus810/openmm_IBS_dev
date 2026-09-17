"""路由上下文（`earliest` / `blocked`）必须到达**每一个**出口。

`decide()` 的退役换窗靠 `plan["earliest_unresolved_window"]` 找退役目标，
`None` 就直接放弃 ⟹ 少传一次 = 那条路径上退役永不触发（末窗全程没被看过）。
先前这两个量要在 44 个调用点各写一遍，实测漏了 6 个。
"""
import ast
import inspect

import abfe_preoptimizer as pre

CTL = pre.Stage2RepairController


def _ctl():
    c = CTL.__new__(CTL)
    c.allow_untrusted = False
    c.lo, c.hi, c.max_path_insertions = 4, 8, 3
    c.stage_type = "vanishing"
    c.stage_dir = c.checkpoint_dir = "/nonexistent"
    c.feasible = lambda view=None, n_insert=1: {
        "insert_lambda": "不可行", "split_tail_window": "不可行"}
    c._read_stage_result = lambda: None
    return c


def _view():
    def w(i, **kw):
        base = dict(
            window_idx=i, phase="PRODUCTION", verdict="ANALYSIS_ELIGIBLE",
            bias_status="converged", f_k_evidence_status="verified",
            warmup_steps_left=500000, production_steps=250000,
            production_steps_target=250000, self_verdict="ANALYSIS_ELIGIBLE",
        )
        base.update(kw)
        return base
    return {
        # 窗口 0 的 f_k 被有统计功效地驳回 ⟹ 它是 earliest，走分支 2。
        "windows": [w(0, verdict="STATISTICALLY_REJECTED",
                      f_k_evidence_status="refuted"), w(1)],
        "path": {"window_ranges": [[0, 4], [4, 8]], "n_states": 9},
        "skipped_windows": [], "sampling_units": [],
        "stage_analysis_status": "ANALYSIS_INCOMPLETE",
        "has_stage_result": False,
    }


def test_branch_2_refuted_carries_the_routing_context():
    """分支 2（f_k 被驳回）以前不传 `earliest` / `blocked`。"""
    plan = _ctl()._decide_once(_view())
    assert plan["exit"] == "HALT_FK_REFUTED", plan["reason"]
    assert plan["earliest_unresolved_window"] == 0, (
        "退役换窗靠这个键找目标，None 等于这条路径上退役永不触发")
    assert plan["blocked_by_upstream"] == [1]


def test_no_plan_call_can_silently_lose_the_context():
    """静态兜底：`plan()` 必须从 `_ctx` 取默认值，不能退回"每个调用点自己传"。"""
    src = inspect.getsource(CTL._decide_once)
    tree = ast.parse(src.lstrip())
    fn = tree.body[0]
    plan_def = next(n for n in ast.walk(fn)
                    if isinstance(n, ast.FunctionDef) and n.name == "plan")
    body = ast.unparse(plan_def)
    assert "_ctx.get('earliest')" in body and "_ctx.get('blocked')" in body, (
        "plan() 不再从 _ctx 取默认值 ⟹ 44 个调用点里漏一个就是静默失效")
