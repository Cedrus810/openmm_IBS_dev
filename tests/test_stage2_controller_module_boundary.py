"""S2-D：Stage-2 控制器代码的归属边界。

收拢的判据不是"都塞进一个文件"，是**决策与执行分居**：

  · 布局/证据侧的纯判断（段名解析、λ 版本读回、末窗合法化、新 Epoch 预算）
    跟 `insert_lambda_in_failed_ibs_window` / `repartition_tail_from_anchor` /
    `feasible_repair_actions` 同住 `abfe_preoptimizer` —— 它们是同一套判据，
    分居两个文件就是"同一个不变量的 N 份实现"（设计文档 §6.1）。
  · **执行器留在 `abfe_pipeline`**：`_run_stage2_autonomous` 写盘、跑采样。
    控制器只读这条边界（`test_controller_never_writes_anything`）不因收拢而模糊。

这条测试在"有人又把 helper 挪回 pipeline"或"把执行器塞进 preopt"时红。
"""
import ast
import pathlib

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = pathlib.Path(__file__).resolve().parents[1]

# 搬家名单 = 设计文档 §9.4 点名的那几个（本次已落地的部分）。
# ⚠️ 名单由 §9.4 定，**不要自行增删** —— `_legalize_tail_window` /
# `_latest_segment_dirs` / `_solve_merged_segments_if_any` 的去留见那一节。
MOVED = [
    "relearn_epoch_required_steps",
    "segment_dirs_for_evidence",
    "lambdas_from_version_record",
    "existing_segment_names",
]


def _module(name):
    return ast.parse((ROOT / name).read_text(encoding="utf-8"))


def test_decision_helpers_live_in_the_preoptimizer():
    top = {n.name for n in _module("abfe_preoptimizer.py").body
           if isinstance(n, ast.FunctionDef)}
    missing = [n for n in MOVED if n not in top]
    assert not missing, f"这些决策侧纯函数不在 abfe_preoptimizer 顶层: {missing}"


def test_they_are_gone_from_the_pipeline_class():
    cls = next(n for n in _module("abfe_pipeline.py").body
               if isinstance(n, ast.ClassDef) and n.name == "ABFEPipeline")
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    dupes = [n for n in MOVED if f"_{n}" in methods or n in methods]
    assert not dupes, f"ABFEPipeline 上还留着已迁走的同名方法（两份实现）: {dupes}"


def test_the_executor_stays_in_the_pipeline():
    """执行器不许搬进 preopt —— 那会把只读控制器和写盘执行器混进同一个模块。"""
    cls = next(n for n in _module("abfe_pipeline.py").body
               if isinstance(n, ast.ClassDef) and n.name == "ABFEPipeline")
    methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert "_run_stage2_autonomous" in methods

    pre_top = {n.name for n in _module("abfe_preoptimizer.py").body
               if isinstance(n, ast.FunctionDef)}
    assert "_run_stage2_autonomous" not in pre_top
    assert "run_stage2_autonomous" not in pre_top


def test_relearn_budget_helper_is_actually_callable_from_the_loop():
    """迁移前这里是 bug：类级 `@staticmethod` 被当**裸名字**调
    ⟹ `RELEARN_FK_EPOCH` 分支一走到就 `NameError`。"""
    src = (ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert "_pre.relearn_epoch_required_steps(" in src
    assert "_need = relearn_epoch_required_steps(" not in src
    assert "_need = _relearn_epoch_required_steps(" not in src


def test_moved_helpers_take_no_self():
    """搬成模块级 ⟹ 不许残留 `self`（日志走 `log=` 回调）。"""
    import abfe_preoptimizer as pre
    import inspect

    for name in MOVED:
        params = list(inspect.signature(getattr(pre, name)).parameters)
        assert "self" not in params, f"{name} 还带着 self"


def test_segment_dirs_for_evidence_fails_closed_on_split_segments():
    import abfe_preoptimizer as pre

    with pytest.raises(ValueError, match="多个采样段"):
        pre.segment_dirs_for_evidence(
            {"vanishing_2", "vanishing_3"}, "/r/vanishing", "/r/checkpoints")
