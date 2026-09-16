"""控制器的**词汇表**必须与它真正发出的词一致（2026-09-13）。

`ACTIONS` / `EXITS` / `TERMINAL_EXITS` 是三份**声明式**清单：没有任何代码校验
它们，所以它们会悄悄漂。实测漂出来的三种形状，每一种都有真实后果：

  1. **声明了却从没发过的死词**。`HALT_NO_FEASIBLE_ACTION` 就是：真正在用的是
     `NO_FEASIBLE_ACTION`（无 `HALT_` 前缀），而它**不在 `TERMINAL_EXITS` 里** ——
     谁照着清单发它，`terminal` 不成立、主循环不 break，得到一个看起来像终止、
     实际不终止的出口。
  2. **发得出来却没声明**。`GLOBAL_BUDGET_EXHAUSTED` / `NO_FEASIBLE_ACTION` 都是
     真终态却漏在 `EXITS` 之外 —— 「每一个结局都在这里」这句话本身不成立。
  3. **同一个词列两遍**（`ACTIONS` 里的 `INSERT_LAMBDA`）。

所以这里按**源码 AST** 对账，不是按运行时行为：把 `decide()` 里每个
`plan(..., exit_=...)` 的字符串字面量抽出来，与清单互相比。
（同类源码断言的先例见 `docs/STAGE2_CONTROLLER_DESIGN_2026-09-12.md` §9.5。）
"""

import ast
import inspect

import pytest

pytestmark = pytest.mark.cpu_only

import abfe_preoptimizer as pre

CTL = pre.Stage2RepairController


def _decide_tree():
# 🔑 [2026-09] `decide()` 现在只是 23 行的外壳（"退役一个窗口再判一次"），判断体是 `_decide_once`（1831 行）。
# 源码探针指着 `decide` 会一无所获 —— 断言"存在"的当场红，断言"不存在"的**静默变成假绿**。
    return ast.parse(inspect.getsource(CTL.decide).lstrip()
                     + "\n" + inspect.getsource(CTL._decide_once).lstrip())


def _str_choices(node):
    """这个表达式**可能取到的字符串字面量**。

    ⚠️ 不能用 `ast.walk` 无差别收 —— 它会把子表达式里**不相干**的字符串也算进来，
    比如 `plan("INSERT_LAMBDA" if feas.get("insert_lambda") is None else ...)` 里
    那个小写的字典键 `"insert_lambda"`，于是断言炸在一个根本不是动作的词上。
    只沿「条件表达式的两支」和「or/and 的各支」下钻，遇到函数调用就停。
    """
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    if isinstance(node, ast.IfExp):
        return _str_choices(node.body) | _str_choices(node.orelse)
    if isinstance(node, ast.BoolOp):
        out = set()
        for v in node.values:
            out |= _str_choices(v)
        return out
    return set()


def _plan_calls():
    for node in ast.walk(_decide_tree()):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "plan"):
            yield node


def _literals_of(kwarg: str):
    """`decide()` 里所有 `plan(..., <kwarg>=...)` 可能取到的字符串字面量。

    条件表达式（`"A" if cond else "B"`）的两支都会被收进来 —— 正是这种写法
    最容易让一个分支的出口悄悄失配。
    """
    found = set()
    for node in _plan_calls():
        for kw in node.keywords:
            if kw.arg == kwarg:
                found |= _str_choices(kw.value)
    # [2026-09-14] `plan()` 内部还有一道**统一闸**会改写 action/exit_
    # （生产预算已知且确实耗尽 ⟹ NO_ACTION + GLOBAL_BUDGET_EXHAUSTED）。
    # 那是真正的发出点，只是写法是赋值而不是关键字参数 —— 不收它，守卫会把一个
    # 活着的词误判成死词。
    for node in ast.walk(_decide_tree()):
        if not isinstance(node, ast.Assign):
            continue
        targets = node.targets[0]
        names = (targets.elts if isinstance(targets, ast.Tuple) else [targets])
        values = (node.value.elts if isinstance(node.value, ast.Tuple)
                  else [node.value])
        for nm, val in zip(names, values):
            if isinstance(nm, ast.Name) and nm.id == kwarg:
                found |= _str_choices(val)
    return found


def _first_positional_literals():
    """`plan()` 的第一个位置参数就是 action。"""
    found = set()
    for node in _plan_calls():
        if node.args:
            found |= _str_choices(node.args[0])
    return found


def test_no_duplicates_in_any_vocabulary():
    for name in ("ACTIONS", "EXITS", "TERMINAL_EXITS"):
        words = getattr(CTL, name)
        assert len(words) == len(set(words)), f"{name} 里有重复的词：{words}"


def test_terminal_exits_are_a_subset_of_exits():
    """终态也是结局。`EXITS` 自称是完整清单，就必须装得下 `TERMINAL_EXITS`。"""
    missing = [x for x in CTL.TERMINAL_EXITS if x not in CTL.EXITS]
    assert not missing, f"这些终态没有登记进 EXITS：{missing}"


def test_every_emitted_exit_is_declared():
    emitted = _literals_of("exit_")
    undeclared = sorted(x for x in emitted if x not in CTL.EXITS)
    assert not undeclared, f"decide() 发得出来、但 EXITS 里没有：{undeclared}"


def test_every_emitted_action_is_declared():
    emitted = _first_positional_literals()
    undeclared = sorted(x for x in emitted if x not in CTL.ACTIONS)
    assert not undeclared, f"decide() 发得出来、但 ACTIONS 里没有：{undeclared}"


def test_no_dead_exits_left_in_the_declaration():
    """声明了却从没发过的词 = 死词，必须删掉。

    豁免的只有 `DONE`/`DONE_UNTRUSTED` 之外那些**由主循环**（而不是 `decide()`）
    写进 history 的出口 —— 目前没有这类，所以清单为空。豁免要加就写在这里，
    并注明谁发它，别默默放行。
    """
    emitted = _literals_of("exit_")
    # [2026-09-14 裁决 2] `GLOBAL_BUDGET_EXHAUSTED` 重新由 `decide()` 发出，
    # 但**换了判据**：只有 stage 的**生产预算已知且确实耗尽**时才发（`plan()` 里
    # 那道统一闸）。它先前被挂在"所有窗口预热余额为 0"上 —— 拿 A 账本终止一个
    # 只花 B 账本的动作。停滞保护不再推断预算（推不动 ≠ 没钱）。
    emitted_by_outer_loop = set()
    dead = sorted(
        x for x in CTL.EXITS
        if x not in emitted and x not in emitted_by_outer_loop
    )
    assert not dead, (
        f"这些出口在 EXITS 里声明了，但 decide() 从没发过：{dead}。"
        "死词会被下一个人照着用 —— 尤其危险的是它可能不在 TERMINAL_EXITS 里，"
        "发出去不终止。"
    )


def test_no_action_is_only_ever_paired_with_a_terminal_exit():
    """`NO_ACTION` 不是动作，是「没有动作」。它必须终止，否则执行器会收到一个不认识的词。

    终态在主循环里**先于**执行器分发被 break（`abfe_pipeline._run_stage2_autonomous`），
    这正是执行器不需要认识 `NO_ACTION` 的前提。

    ⚠️ **这个检查有已知盲区，不要把它读成「全覆盖」**（2026-09-14 实际踩到过）：
    它只看 `plan(...)` **调用点上字面写出来的** `action` / `exit_`。`plan()` 内部
    把 `action` 改写成 `NO_ACTION` 的那几道闸（补帧准入、no-op 台账、预算闸）
    在这里**一条都看不见** —— 它们的 `exit_` 由条件表达式在函数体里算出来。
    那几处曾经配过非终态出口而本测试仍然是绿的。
    不变量本身没有放宽，只是检查手段看不见 —— **在 `plan()` 里改写 `action` 时，
    要自己保证配的是终态出口。**
    """
    for node in _plan_calls():
        if not node.args:
            continue
        if "NO_ACTION" not in _str_choices(node.args[0]):
            continue
        exits = set()
        for kw in node.keywords:
            if kw.arg == "exit_":
                exits |= _str_choices(kw.value)
        assert exits, "NO_ACTION 必须带出口"
        assert exits <= set(CTL.TERMINAL_EXITS), (
            f"NO_ACTION 配了非终态出口 {sorted(exits - set(CTL.TERMINAL_EXITS))} ⟹ "
            "主循环不会 break，执行器会收到一个它不认识的动作"
        )
