"""E-03 / ATT-27 回归：已被否决的变异逻辑不得搬回生产代码。

2026-07-27 移除了三块**不可达或从未被调用**的代码，全部逐字归档在 `docs/archive/`：

| 移除对象 | 行数 | 归档 |
|---|---|---|
| `_run_stage_with_overlap_autorepair` 里 `return` 之后的 ensemble 变异循环 | 872 | `removed_overlap_autorepair_mutation_loop.md` |
| `_refine_lambda_path_with_medium_probe`（`enable_lambda_refine` 的实现半边） | 83 | `removed_refine_lambda_path_with_medium_probe.md` |
| `_retired_overlapping_vdw_schedule_design`（docstring 自认 retired） | 188 | `removed_retired_overlapping_vdw_schedule_design.md` |

为什么值得写测试守着：这三块都属于"把相邻态 fixed-H overlap 当成 IBS 收敛仲裁标准"
那条已被证伪的路线。IBS 通过把**一个**冻结的积分混合分布重加权到各目标态取 ΔG，
adjacent overlap 根本不是它的正确性判据。按那个判据做的自动拆窗 / 插 λ /
`recalibrate_f_k` 循环曾**烧掉约一周 GPU 而没产出任何 ΔG**。

风险不对称：复活的代价远大于保留的收益。所以这里钉的不是"代码行数"，而是
**那些函数名不能再出现，且短路 return 不能被挪走**。

同批扫描发现 `OnlineConvergenceMonitor`(230 行) / `ChunkedMBARAnalyzer`(98 行)
也完全无人调用，但**刻意保留**——它们是"写完未接入流水线"，接错了只是不工作，
不会产出错误的自由能；与"已被证伪的错误路线"性质不同。本文件不对它们做断言。
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = REPO_ROOT / "abfe_pipeline.py"

REMOVED_FUNCTIONS = [
    "_refine_lambda_path_with_medium_probe",
    "_retired_overlapping_vdw_schedule_design",
]


def _pipeline_tree():
    return ast.parse(PIPELINE.read_text(encoding="utf-8"), filename="abfe_pipeline.py")


@pytest.mark.parametrize("name", REMOVED_FUNCTIONS)
def test_removed_function_is_not_back(name):
    tree = _pipeline_tree()
    defined = {
        n.name for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert name not in defined, (
        f"{name} 又回到 abfe_pipeline.py 了。它属于已被证伪的 adjacent-overlap "
        f"仲裁路线，归档在 docs/archive/ 里；要重做请基于当前 abfe_preoptimizer 重写。"
    )


def test_overlap_autorepair_stays_a_short_circuit():
    """`_run_stage_with_overlap_autorepair` 必须仍以无条件 return 结束。

    这是整块删除成立的前提：只要有人在这个 return 之前插一层分支、或把它挪走，
    就等于给"变异逻辑"重新开了门（哪怕实现已经不在这个文件里）。
    """
    tree = _pipeline_tree()
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_run_stage_with_overlap_autorepair"
    )
    # 函数体最后一条可执行语句必须是 Return
    assert isinstance(fn.body[-1], ast.Return), (
        "函数体最后一条语句不再是 return —— 短路被破坏了，"
        f"实际是 {type(fn.body[-1]).__name__}"
    )
    # 且函数体顶层不得再出现 While（变异循环的形状）
    top_level_kinds = {type(s).__name__ for s in fn.body}
    assert "While" not in top_level_kinds, (
        "函数体顶层又出现了 While —— 变异循环回来了"
    )


def test_function_stays_small():
    """删完之后这个函数应当很短。回到几百行就说明有东西被搬回来了。"""
    tree = _pipeline_tree()
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_run_stage_with_overlap_autorepair"
    )
    n_lines = fn.end_lineno - fn.lineno + 1
    assert n_lines < 200, (
        f"_run_stage_with_overlap_autorepair 现在 {n_lines} 行（删除后约 110 行）。"
        "涨到 200 行以上请检查是不是把归档里的变异逻辑搬回来了。"
    )


def test_enable_lambda_refine_is_rejected_at_the_entry_point():
    """`enable_lambda_refine` 的拒绝必须在 `run_full_pipeline` 靠前的位置。

    原来它埋在函数体第 ~795 行——预平衡、Boresch 锚定、Stage 1 全跑完之后才炸，
    配置手滑要烧掉几小时才知道。实现已移除，守卫必须留，且必须早。
    """
    tree = _pipeline_tree()
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "run_full_pipeline"
    )
    hits = [
        node.lineno for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "enable_lambda_refine"
    ]
    assert hits, "run_full_pipeline 里找不到 enable_lambda_refine 的守卫——不能静默放行"
    offset = min(hits) - fn.lineno
    assert offset < 60, (
        f"enable_lambda_refine 的守卫距函数起点 {offset} 行，太深了。"
        "它必须在任何采样/预平衡之前拒绝，否则用户要烧掉几小时才拿到这个报错。"
    )


@pytest.mark.parametrize(
    "doc_name",
    [
        "removed_overlap_autorepair_mutation_loop.md",
        "removed_refine_lambda_path_with_medium_probe.md",
        "removed_retired_overlapping_vdw_schedule_design.md",
    ],
)
def test_archive_exists_and_is_non_executable(doc_name):
    """E-03 要求「保留历史可读性时放入归档文档，不得保留可被误激活的可执行代码」。

    归档必须真的存在（否则删除就是净丢失），且是 .md 而非 .py（不会被 import、
    不会被 pytest 收集、不会被误激活）。
    """
    path = REPO_ROOT / "docs" / "archive" / doc_name
    assert path.is_file(), f"归档缺失：{path}——删掉的代码就真的没了"
    assert path.suffix == ".md"
    text = path.read_text(encoding="utf-8")
    assert "```python" in text, "归档里没有代码块，等于没归档"
    assert len(text.splitlines()) > 30, "归档内容太短，可能是占位文件"
