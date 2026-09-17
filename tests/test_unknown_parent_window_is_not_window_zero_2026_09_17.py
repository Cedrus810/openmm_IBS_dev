"""「父窗未知」不得被记成 window 0（2026-09-17）。

全仓原有五处 `int(u.get("parent_window") or 0)`。「未知不是零」在这里有两个后果：

  · **块账**：把一个父窗未知的子窗算进 window 0 的补帧配额 —— 而 window 0 是本
    方法的必然坏窗口（解耦端点），它的配额最金贵；同时真正的父窗一块都不记、
    补帧准入对它**恒放行**。
  · **排序**：`or 0` 让它插到调度队首，被当成 window 0 的子窗路由，可能越过
    真正未解决的上游窗口。

目前执行器两处都写 `int(window_idx)`，所以实测不会触发 —— 但这是**静默**错，
出事时没有任何迹象。本文件钉的是判据，不是"现在有没有发生"。
"""
import inspect

import pytest

pytestmark = pytest.mark.cpu_only

import abfe_preoptimizer as pre


def test_no_or_zero_default_on_parent_window_anywhere():
    """源码探针：`parent_window` 后面不许再跟 `or 0`。

    这是**唯一**能防住它复发的东西 —— 行为级测试造不出"父窗未知"的真实盘面
    （执行器总是写得出 window_idx），所以这条只能靠静态判据。
    """
    # 只看**可执行代码**：注释与 docstring 里引用这个反例是允许的（它们解释
    # 为什么不许这么写）。用 AST 剥掉字符串常量与注释，剩下的才算真代码。
    import ast
    tree = ast.parse(inspect.getsource(pre))
    lines = inspect.getsource(pre).split("\n")
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                doc_lines.add(ln)
    bad = [l.strip() for i, l in enumerate(lines, start=1)
           if 'parent_window' in l and 'or 0' in l
           and not l.strip().startswith("#") and i not in doc_lines]
    assert not bad, (
        "「未知不是零」：父窗读不出来时不得默认 window 0。\n  " + "\n  ".join(bad))


def test_the_sort_key_puts_an_unknown_parent_last_not_first():
    """排序兜底必须是**最后**（大数），不是 0。"""
    src = inspect.getsource(pre.Stage2RepairController._decide_once)
    assert "_pw_or_last" in src, "父窗排序没有走共享兜底"
    assert "1 << 30" in src, "未知父窗的排序兜底不是「排到最后」"


def test_a_subwindow_with_an_unknown_parent_is_not_routed():
    """父窗未知 ⟹ 不知道它在因果顺序里排哪儿 ⟹ **不路由**（fail-closed）。

    路由它可能越过未解决的上游窗口，而上游重锚会作废下游 lineage。
    """
    src = inspect.getsource(pre.Stage2RepairController._decide_once)
    assert "if _pw is None:" in src, "缺帧子窗那条没有对「父窗未知」fail-closed"
    assert "_spw is not None and" in src, "偏斜子窗那条没有对「父窗未知」fail-closed"
