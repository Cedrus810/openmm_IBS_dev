"""停滞降级必须过 `plan()` 已有的两道预算闸 —— 否则它能开一个跑不完的 Epoch。

`_run_stage2_autonomous` 的停滞保护直接改写 `act = "PROBE_REANCHOR_EPOCH"` 然后执行，
而这个动作同时落在 `plan()` 的 `_EPOCH_ACTIONS`（换 Epoch 的最低验证额度）与
`_PRODUCTION_CHARGED`（按一块收生产预算）两张表里 —— 正常路径两道闸都要过，
这条旁路一道都不过。两道闸各自的存在理由就是它的失败形状：
  · 「不能启动动作之后才发现新 f_k 没预算验」—— win2 连死三次；
  · 剩 100k 也照发一个 250k 块 = 允许超支一整块。
两条都让整跑**跑不完**（烧光预算再半路死），不是跑出个坏答案。
"""
import ast
import inspect
import os

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _autonomous_loop_ast():
    with open(os.path.join(ROOT, "abfe_pipeline.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef)
                and n.name == "_run_stage2_autonomous")


def test_escalation_consults_the_same_epoch_budget_gate_as_plan():
    """降级路径必须调**同一个** `_epoch_validation_unaffordable`，不许另写判据。"""
    fn = _autonomous_loop_ast()
    src = ast.unparse(fn)
    assert "_epoch_validation_unaffordable" in src, (
        "停滞降级没有查换 Epoch 的最低验证额度 ⟹ 它能启动一个付不起验证的 Epoch，"
        "把预算烧光再半路死掉（win2 连死三次的形状）。"
        "判据必须复用 `plan()` 那一份，不许在执行器里另写。")
    assert "production_block_steps" in src and "stage_remaining_steps" in src, (
        "停滞降级没有查生产预算余量 ⟹ 剩 100k 也会照发一个 250k 块。")


def test_the_escalation_assignment_is_guarded(  # noqa: D103
):
    """`act = "PROBE_REANCHOR_EPOCH"` 这一步必须在预算判定**之后**、且受它保护。

    只检查"函数里出现过那个调用"不够——调用可能在赋值之后。这里要求：
    赋值语句所在的 `if` 分支，其上文确实先算过 `_esc_broke` 且条件里引用了它。
    """
    fn = _autonomous_loop_ast()
    assigns = [n for n in ast.walk(fn)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "act" for t in n.targets)
               and isinstance(n.value, ast.Constant)
               and n.value.value == "PROBE_REANCHOR_EPOCH"]
    assert assigns, "找不到降级赋值点——本测试失效了，先修测试"
    guards = [n for n in ast.walk(fn)
              if isinstance(n, ast.If)
              and any(a in ast.walk(n) for a in assigns)
              and "_esc_broke" in ast.unparse(n.test)]
    assert guards, (
        "降级赋值没有被预算判定的结果 `_esc_broke` 保护 ⟹ 付不起也照样降级。")


def test_budget_gate_is_not_reimplemented_in_the_executor():
    """执行器不得自己算"最低验证额度"——那是 `plan()` 已有判据的第二份实现。"""
    import abfe_preoptimizer as pre
    fn = _autonomous_loop_ast()
    src = ast.unparse(fn)
    # 阶梯常量只允许出现在控制器一侧
    assert "FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS" not in src, (
        "执行器自己读了冻结验证阶梯常量 ⟹ 同一个不变量两份实现，"
        "两边一漂就是「控制器批准、执行器拒绝并终止整跑」。")
    assert hasattr(pre.Stage2RepairController, "_epoch_validation_unaffordable")
