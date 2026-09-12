"""仓库里不许再出现「个人绝对路径当 argparse 默认值」。

这是第二次了（第一次是 `gmx_path`，2026-08-24 删掉两条本机路径）。危害不是"不好看"：
默认值会让忘了传参的调用**静默去读别人机器上的目录**，而不是报错。注释/docstring 里
的示例路径不在此列——那批是刻意留的、读的人一眼知道要替换。

守的是**根因**：加一处新的就会红，不用等下一次审计翻出来。
"""
import ast
import pathlib

import pytest

pytestmark = pytest.mark.cpu_only

REPO = pathlib.Path(__file__).resolve().parent.parent
PERSONAL_PREFIXES = ("/home/", "/Users/", "C:\\Users", "/root/")
SKIP_DIRS = {".git", "__pycache__", ".codex-test-tmp", "attic", "archive", "plugins"}


def _string_constants(node):
    """这个 AST 节点底下所有字符串字面量（含 Path("...")、f-string 的常量段）。"""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child.value


def _argparse_default_offenders(tree, path):
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in {"add_argument", "add_option"}:
            continue
        for keyword in node.keywords:
            if keyword.arg != "default":
                continue
            for text in _string_constants(keyword.value):
                if text.startswith(PERSONAL_PREFIXES):
                    offenders.append(f"{path}:{node.lineno}  default={text!r}")
    return offenders


def test_no_argparse_default_points_at_a_personal_absolute_path():
    offenders = []
    for path in REPO.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        offenders.extend(
            _argparse_default_offenders(tree, path.relative_to(REPO))
        )
    assert not offenders, (
        "这些 argparse 默认值指向某台机器上的个人路径——忘了传参时它们会静默读错目录，"
        "而不是报错。把 default 删掉改成 required=True，或从别的参数派生：\n  "
        + "\n  ".join(offenders)
    )
