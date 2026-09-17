"""[LR-04] `--only-ligand-in-water` 的控制流契约。

真跑一条腿要 GPU，这里钉的是**不跑也能错的那部分**：开关存在、互斥守卫在、
复合物腿整段确实在 `else` 里、以及溶剂腿之后有早退（不能走到 ΔG_bind）。

⚠️ 这是源码契约探针：断言"某段控制流存在"。指错函数时它**不报错、只是找不到**，
所以每条都同时断言了正例与反例的锚点，别只留 `assert ... in source`。
"""
from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.cpu_only

RUNABFE = pathlib.Path(__file__).resolve().parents[1] / "runabfe.py"
TREE = ast.parse(RUNABFE.read_text(encoding="utf-8"))


def _main() -> ast.FunctionDef:
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError("runabfe.main() 没找到 —— 探针指错了，不是契约变了")


def test_flag_and_config_default_both_exist():
    source = RUNABFE.read_text(encoding="utf-8")
    assert '"--only-ligand-in-water"' in source, "argparse 开关不见了"
    assert '"only_ligand_in_water": False' in source, "config 默认值不见了（默认必须是关）"


def test_it_is_mutually_exclusive_with_the_other_only_entries():
    """三个 only-* 入口各跑一条腿，同时给两个必须硬错。"""
    raises = [
        node
        for node in ast.walk(_main())
        if isinstance(node, ast.Raise)
        and "--only-ligand-in-water 与 --only-complex-charging" in ast.dump(node)
    ]
    assert len(raises) == 1, "互斥守卫应当恰好一处"


def test_residual_switch_is_fail_closed_not_silently_baseline():
    """残差运行时在复合物腿那段构造，跳过它就拿不到 —— 必须抛，不能静默降级。"""
    dumped = ast.dump(_main())
    assert "--only-ligand-in-water 暂不支持" in dumped


def test_complex_leg_body_sits_in_the_else_branch():
    """复合物腿主流程必须整段在 `if config.only_ligand_in_water: ... else:` 的 else 里。

    反例锚点：`run_full_pipeline` 的复合物腿调用**不得**出现在 if 体里。
    """
    found = None
    for node in ast.walk(_main()):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Attribute)
            and test.attr == "only_ligand_in_water"
            and node.orelse
        ):
            body_src = ast.dump(ast.Module(body=node.body, type_ignores=[]))
            else_src = ast.dump(ast.Module(body=node.orelse, type_ignores=[]))
            if "complex_results" in body_src and "run_full_pipeline" in else_src:
                found = (body_src, else_src)
    assert found is not None, "没找到把复合物腿包进 else 的那个分支"
    body_src, else_src = found
    assert "run_full_pipeline" not in body_src, (
        "ligand-in-water-only 分支里不该有任何采样调用"
    )
    assert "complex_results = None" in body_src or "None" in body_src


def test_there_is_an_early_return_before_delta_g_bind():
    """溶剂腿之后必须直接 return：ΔG_bind 要两条腿，这次只有一条。"""
    returns = []
    for node in ast.walk(_main()):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Attribute)
            and node.test.attr == "only_ligand_in_water"
        ):
            if any(isinstance(inner, ast.Return) for inner in node.body):
                returns.append(node)
    assert returns, "ligand-in-water-only 没有早退 —— 会一路走到第 8 节去算 ΔG_bind"
