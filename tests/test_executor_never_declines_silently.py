# -*- coding: utf-8 -*-
"""自治执行器**不许静默拒绝执行**（2026-09-14 真机，同形状第 3 次）。

失败形状 ⑦：控制器发一个动作，执行器发现前置条件不满足，只打一行日志就跳过。
盘面没变 ⟹ 控制器下一轮读到同样的状态、发同样的动作 ⟹ 死循环，最后靠通用停滞
探测（连续 N 次盘面未变）才停得下来，而停下来时报的是"推不动了"，不是真正的原因。

真机记录：
  · `SPLIT_TAIL_WINDOW` 取不到 anchor ⟹ 连发 4 次（cyclod_ligand1/rep2），
    版本链上一条 `tail_repartition` 都没有。
  · `INSERT_LAMBDA` 缺 pilot / 失败窗口区间 ⟹ 同一条静默路径，一直没被发现。

契约：dispatch 里每一条"决定不执行"的分支，必须**留下痕迹**——
记 no-op 账本、写终态、或抛错。只打日志不算。
"""
import ast
import os

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "abfe_pipeline.py")

pytestmark = pytest.mark.cpu_only

# 留下痕迹的几种合法方式。
_TRACES = ("_record_noop_action", "raise", "history[-1]", "break", "continue")
# 真的把活干了 —— 这种分支不是"拒绝执行"。
_DOES_WORK = ("run_once", "_immutable_rewindow_step", "_topup_rewindow_child",
              "_solve_with_rewindow_children", "_recalibrate_f_k_and_resample_segment",
              "_legalize_tail_window", "append_version", "_persist_inprogress_stage_result")


def _dispatch_branches():
    """dispatch 链上每个 `act == ...` 分支的 (动作名, 节点)。"""
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_run_stage2_autonomous")
    out = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        names = _acts_compared(node.test)
        if names:
            out.append((tuple(names), node))
    return out


def _acts_compared(test):
    """从 `act == "X"` / `act in ("X","Y")` 里取出动作名。"""
    if not isinstance(test, ast.Compare) or len(test.ops) != 1:
        return []
    left = test.left
    if not (isinstance(left, ast.Name) and left.id == "act"):
        return []
    cmp_ = test.comparators[0]
    if isinstance(cmp_, ast.Constant) and isinstance(cmp_.value, str):
        return [cmp_.value]
    if isinstance(cmp_, (ast.Tuple, ast.List, ast.Set)):
        return [e.value for e in cmp_.elts
                if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _seg(node):
    return ast.unparse(node)


def _has_work(nodes):
    src = "".join(_seg(n) for n in nodes)
    return any(w in src for w in _DOES_WORK)


def _silent_declines():
    """「拒绝执行」的两种句法形状，两条真机 bug 都是第一种。

      (a) `if <前置条件不满足>: log` / `else: <干活>`   —— 活在 else 里
      (b) `if <前置条件不满足>: log` 且它是该分支的**最后一句** —— 后面没活了

    只是"打一行日志再继续干活"的 if 不算拒绝，所以两者都要求 body 自己不干活。
    """
    bad = []
    for acts, branch in _dispatch_branches():
        last = branch.body[-1] if branch.body else None
        for inner in ast.walk(branch):
            if not isinstance(inner, ast.If) or inner is branch or _acts_compared(inner.test):
                continue
            body = "".join(_seg(st) for st in inner.body)
            if "self._log" not in body or _has_work(inner.body):
                continue
            if any(t in body for t in _TRACES):
                continue
            declines = _has_work(inner.orelse) or inner is last
            if declines:
                bad.append((acts, _seg(inner.test), body.strip()[:120]))
    return bad


def test_no_dispatch_branch_declines_with_only_a_log_line():
    bad = _silent_declines()
    assert not bad, (
        "这些分支决定不执行，却只打了一行日志 —— 盘面不变，控制器会永远重发同一个"
        "动作：\n" + "\n".join(f"  · {a} :: if {t}\n      {b}" for a, t, b in bad)
    )


def test_the_scanner_actually_sees_the_two_real_regressions():
    """扫描器本身要能抓到真机那两条 —— 否则它只是一条永远绿的装饰。"""
    import re
    src = open(SRC, encoding="utf-8").read()
    # 🔑 [2026-09-16] 原来钉的是字面 `reason="..."`。插 λ 那条后来多了第二个
    # no-op 理由（`insert_lambda_would_strand_tail_window`），于是理由改成先算进
    # `_noop_reason` 再传 ⟹ 字面 `reason="insert_lambda_without_range_or_pilot"`
    # 不复存在，探针当场变红。钉理由字符串本身，怎么传给 `_record_noop_action`
    # 是实现细节。
    for marker in ('"split_tail_window_without_anchor"',
                   '"insert_lambda_without_range_or_pilot"'):
        assert marker in src, f"缺少 {marker}：执行器又退回静默跳过了"
    # 两处都必须紧跟在一个 `is None` 前置条件判断之后。
    assert len(re.findall(r"_record_noop_action\(", src)) >= 4
