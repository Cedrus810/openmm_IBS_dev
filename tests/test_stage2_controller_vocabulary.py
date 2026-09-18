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
（同类源码断言的先例见 `docs/archive/STAGE2_CONTROLLER_DESIGN_2026-09-12.md` §9.5。）
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


def test_no_dead_actions_left_in_the_declaration():
    """`ACTIONS` 里声明的动作，`decide()` 必须真的发得出。

    这是 `test_no_dead_exits_left_in_the_declaration` 的**动作侧孪生**。
    只守出口不守动作，就会出现「成本表、执行器、测试都齐了，唯独判断侧没有发出点」
    这种缺口 —— `IMMUTABLE_REWINDOW` 2026-09-16 实测就是这样。
    """
    emitted = set()
    for node in _plan_calls():
        if node.args:
            emitted |= _str_choices(node.args[0])
    for node in ast.walk(_decide_tree()):
        if not isinstance(node, ast.Assign):
            continue
        tg = node.targets[0]
        names = [x.id for x in (tg.elts if isinstance(tg, ast.Tuple) else [tg])
                 if isinstance(x, ast.Name)]
        vals = (node.value.elts if isinstance(node.value, ast.Tuple)
                else [node.value])
        for nm, v in zip(names, vals):
            if nm in ("act", "action"):
                emitted |= _str_choices(v)
    dead = sorted(x for x in CTL.ACTIONS if x not in emitted)
    assert not dead, (
        f"这些动作在 ACTIONS 里声明了，但 decide() 从没发过：{dead}。"
        "成本表/执行器/测试可能都齐了，唯独判断侧没有发出点 —— "
        "那条能力对生产等于不存在。"
    )


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


# ---------------------------------------------------------------------------
# [2026-09-17，用户拍板 A] 归因三态：「还没被证伪」不得当成「帧不够」
# ---------------------------------------------------------------------------

def test_reachability_only_yields_unknown_never_sample_size():
    """射程判据 `ratio × headroom ≥ target` 只能给 `UNKNOWN`。

    真实关系是 `R_future = R_now × H × (η_f/η_n) × (g_n/g_f)`（`η = N_eff/N`），
    而 `n_eff_over_g_reachable_by_frames` 只留 `R_now × H`、**假定 η 与 g 恒定** ——
    两个假设实测都朝不利方向走（g 12.25→17.63；η 见 §5.1）。所以它的 `True`
    只够说「还没被证伪」。先前它被升级成「这是样本量问题」，于是 O1 拿一个
    **乐观上界**覆盖掉分支刚做出的「f_k 不对，换 Epoch」诊断。
    """
    # 射程够得着（5.0 × 2.5 = 12.5 ≥ 10）且帧数在地板之上 ⟹ UNKNOWN，不是 SAMPLE_SIZE
    assert pre.support_failure_attribution(
        "INSUFFICIENT_DATA", "min_n_eff_over_g",
        n_decorrelated=400, min_frames=10,
        min_n_eff_over_g=5.0, n_eff_over_g_target=10.0,
        frames_headroom=2.5) == pre.SUPPORT_FAILURE_UNKNOWN
    # 射程够不着 ⟹ STRUCTURAL
    assert pre.support_failure_attribution(
        "INSUFFICIENT_DATA", "min_n_eff_over_g",
        n_decorrelated=400, min_frames=10,
        min_n_eff_over_g=5.0, n_eff_over_g_target=10.0,
        frames_headroom=1.2) == pre.SUPPORT_FAILURE_STRUCTURAL
    # 帧数在地板之下 ⟹ **硬证据**的 SAMPLE_SIZE（这才是唯一能授权降级去补帧的）
    assert pre.support_failure_attribution(
        "HARD_INSUFFICIENT", "min_n_eff_over_g",
        n_decorrelated=7, min_frames=10,
        min_n_eff_over_g=5.0, n_eff_over_g_target=10.0,
        frames_headroom=2.5) == pre.SUPPORT_FAILURE_SAMPLE_SIZE
    # 通过的窗口 ⟹ 不是失败，None
    assert pre.support_failure_attribution(
        "ANALYSIS_ELIGIBLE", "min_n_eff_over_g") is None


def test_is_skew_bool_is_unchanged_by_the_three_state_split():
    """`support_failure_is_skew()` 的布尔值**逐位不变** —— 它问的是「是不是 STRUCTURAL」。

    拆三态是为了**只**收紧 O1（它以前拿 `not is_skew` 当"样本量类"，把 UNKNOWN
    也算了进去）。其余调用点一个都不该受影响，这条就是那个保证。
    """
    cases = [
        ("ANALYSIS_ELIGIBLE", "min_n_eff_over_g", {}),
        ("INSUFFICIENT_DATA", "solver_eligibility", {}),
        ("HARD_INSUFFICIENT", "top1pct_veto", {}),
        ("HARD_INSUFFICIENT", None, {}),
        ("INSUFFICIENT_DATA", None, {}),
        ("INSUFFICIENT_DATA", "min_n_eff_over_g",
         dict(n_decorrelated=400, min_frames=10, min_n_eff_over_g=5.0,
              n_eff_over_g_target=10.0, frames_headroom=2.5)),
        ("INSUFFICIENT_DATA", "min_n_eff_over_g",
         dict(n_decorrelated=400, min_frames=10, min_n_eff_over_g=5.0,
              n_eff_over_g_target=10.0, frames_headroom=1.2)),
        ("HARD_INSUFFICIENT", "min_n_eff_over_g",
         dict(n_decorrelated=7, min_frames=10)),
    ]
    for verdict, src, kw in cases:
        assert pre.support_failure_is_skew(verdict, src, **kw) == (
            pre.support_failure_attribution(verdict, src, **kw)
            == pre.SUPPORT_FAILURE_STRUCTURAL), (verdict, src, kw)
