"""出口语义的唯一注册表契约（2026-09-17，用户拍板）。

旧设计把**三个正交概念**混进字符串名单，分散在 `EXITS` / `TERMINAL_EXITS` /
`_RETIRABLE_EXITS` 三张手写表里：
    ① 为什么停（exit reason） ② 停的是谁（窗口/stage/路径） ③ 主循环终不终止
必然漂移，实证四次 —— 其中两次是**漏项导致停掉一个仍有活干的 stage**
（`SUPPORT_ATTRIBUTION_UNKNOWN`、`D3_REWINDOW_DEPTH_EXHAUSTED`）。

新契约：`plan()` 出 `halt_scope` / `blocking_window` / `blocking_unit`，
`_retirable_window()` **只看 scope、不认 exit 字符串**；语义在 `EXIT_SPECS` 声明一次。
"""
import ast
import inspect
import os

import pytest

pytestmark = pytest.mark.cpu_only

import abfe_preoptimizer as pre
from abfe_preoptimizer import EXIT_SPECS, HALT_SCOPES
from abfe_preoptimizer import Stage2RepairController as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ───────────────────────── 静态：注册表是唯一真相 ─────────────────────────

def test_both_exit_tables_derive_from_the_registry():
    assert set(C.EXITS) == set(EXIT_SPECS)
    assert set(C.TERMINAL_EXITS) == {k for k, v in EXIT_SPECS.items() if v.terminal}
    assert not hasattr(C, "_RETIRABLE_EXITS"), (
        "`_RETIRABLE_EXITS` 已被 `halt_scope` 取代，不得复活 —— "
        "它正是「靠人记得往名单里加一笔」那个形状。")


def _exit_values(node):
    """`exit_=` 实参**可能取到**的字符串值。

    ⚠️ 不能 `ast.walk` 整棵子树：条件表达式的**条件部分**里也有字符串字面量
    （`"预算已用尽" in str(...)`），收进来会把它们当成出口名。只取真正的取值位置。
    """
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    if isinstance(node, ast.IfExp):
        return _exit_values(node.body) | _exit_values(node.orelse)
    return set()


def test_every_emitted_exit_exists_in_the_registry():
    """源码里发出的每一个 exit 字面量都必须在注册表里声明过。"""
    emitted = set()
    for name in ("abfe_preoptimizer.py", "abfe_pipeline.py"):
        with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "plan"):
                continue
            for kw in node.keywords:
                if kw.arg != "exit_":
                    continue
                emitted |= _exit_values(kw.value)
    unknown = sorted(emitted - set(EXIT_SPECS))
    assert not unknown, (
        f"这些出口被发出、却不在唯一注册表 `EXIT_SPECS` 里：{unknown}。"
        "它们的 terminal / scope 语义没有任何地方声明过 ⟹ 静默按默认处理。")


def test_every_terminal_emit_site_can_resolve_a_scope():
    """每个发终态的调用点，要么显式传 `halt_scope`，要么它的出口在注册表里有默认。"""
    src = inspect.getsource(C._decide_once)
    fn = ast.parse(src.lstrip()).body[0]
    bad = []
    for node in ast.walk(fn):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "plan"):
            continue
        kws = {k.arg for k in node.keywords}
        if "exit_" not in kws or "halt_scope" in kws:
            continue
        ex = next(k.value for k in node.keywords if k.arg == "exit_")
        names = {c.value for c in ast.walk(ex)
                 if isinstance(c, ast.Constant) and isinstance(c.value, str)}
        for nm in names:
            spec = EXIT_SPECS.get(nm)
            if spec and spec.terminal and spec.default_scope is None:
                bad.append((node.lineno, nm))
    assert not bad, (
        f"这些发出点发的是**没有默认 scope** 的终态、却没显式传 `halt_scope=`：{bad}。"
        "`NO_FEASIBLE_ACTION` 同时承载局部与全局两种语义，猜默认必然错一半。")


def test_no_feasible_action_has_no_default_scope():
    """它必须保持"无默认"——给了默认就等于替一半的发出点猜错。"""
    assert EXIT_SPECS["NO_FEASIBLE_ACTION"].default_scope is None
    assert all(v.default_scope in HALT_SCOPES
               for k, v in EXIT_SPECS.items()
               if v.terminal and k != "NO_FEASIBLE_ACTION"), (
        "除 NO_FEASIBLE_ACTION 外，每个终态出口都必须有一个合法的默认 scope")


def test_the_four_hot_fixed_exits_have_the_ruled_scope():
    """用户 2026-09-17 逐条拍板的四条。"""
    assert EXIT_SPECS["D3_REWINDOW_DEPTH_EXHAUSTED"].default_scope == "TARGET_LOCAL"
    assert EXIT_SPECS["SUPPORT_ATTRIBUTION_UNKNOWN"].default_scope == "TARGET_LOCAL"
    assert EXIT_SPECS["HALT_FRAMES_ADMISSION_CAP"].default_scope == "STAGE_GLOBAL"
    assert EXIT_SPECS["HALT_INVALID_INPUT"].default_scope == "PATH_INVALID"


# ───────────────────────── 行为：退役只认 scope ─────────────────────────

def _plan(scope, win=1, **kw):
    p = {"halt_scope": scope, "blocking_window": win, "exit": "IRRELEVANT"}
    p.update(kw)
    return p


def _view(*, budget_elsewhere=True):
    """w1 卡住、w2 未解决。`budget_elsewhere` 决定 w2 花不花得出钱。"""
    def w(i, **kw):
        base = dict(window_idx=i, phase="WARMUP_LEARN", verdict="INSUFFICIENT_DATA",
                    self_verdict=None, f_k_evidence_status="indeterminate",
                    warmup_steps_left=500000)
        base.update(kw)
        return base
    # ⚠️ 「别处没预算」必须**两本账都空**：预热余量为 0 之外，生产块配额也要用满。
    # 只清预热账时 `frames_growth_headroom` 仍然 > 1 ⟹ 窗口照样可路由 ——
    # 这正是 `_is_routable_candidate` 的双账语义（两本账互不代替）。
    _blocks = {} if budget_elsewhere else {2: [{}] * 4}
    return {"windows": [w(1), w(2, warmup_steps_left=500000 if budget_elsewhere else 0)],
            "skipped_windows": [], "sampling_units": [],
            "immutable_rewindow": {}, "max_production_blocks_per_window": 4,
            "production_blocks_total_by_window": _blocks,
            "production_blocks_by_window": _blocks}


@pytest.mark.parametrize("exit_name", ["D3_REWINDOW_DEPTH_EXHAUSTED",
                                       "SUPPORT_ATTRIBUTION_UNKNOWN"])
def test_local_halt_retires_and_keeps_the_stage_going(exit_name):
    """D3 / UNKNOWN + 别处有预算 ⟹ 换窗，**不**终止整条 stage。"""
    assert EXIT_SPECS[exit_name].default_scope == "TARGET_LOCAL"
    ctl = C.__new__(C)
    got = ctl._retirable_window(_view(budget_elsewhere=True),
                                _plan("TARGET_LOCAL", win=1), [])
    assert got == 1, "局部卡死且别处有活干，必须退役换窗"


def test_local_halt_without_budget_elsewhere_keeps_the_terminal():
    """别处确实没预算 ⟹ 原样交出终态，不假装还有路走。"""
    ctl = C.__new__(C)
    got = ctl._retirable_window(_view(budget_elsewhere=False),
                                _plan("TARGET_LOCAL", win=1), [])
    assert got is None


@pytest.mark.parametrize("scope", ["STAGE_GLOBAL", "PATH_INVALID"])
def test_global_and_path_invalid_never_retire(scope):
    """即使别处有预算也不得退役：换谁都没用 / 绕过去等于拿不成立的路径烧 GPU。"""
    ctl = C.__new__(C)
    got = ctl._retirable_window(_view(budget_elsewhere=True), _plan(scope, win=1), [])
    assert got is None, f"{scope} 不该触发退役换窗"


def test_retirement_loop_is_bounded_by_the_window_count():
    """退役过的不再参选 ⟹ `decide()` 的循环次数不超过物理窗口数。"""
    src = inspect.getsource(C.decide)
    assert "retired.append" in src and "retired=tuple(retired)" in src
    # 同一个窗口不得被退役两次（`_retirable_window` 第一道守卫）
    ctl = C.__new__(C)
    assert ctl._retirable_window(_view(), _plan("TARGET_LOCAL", win=1), [1]) is None
