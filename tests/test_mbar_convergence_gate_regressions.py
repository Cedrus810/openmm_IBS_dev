"""P1-11：single/2D 路径必须保留并硬检查 MBAR 收敛状态。

## 缺陷是什么（已修，本文件现在是契约测试）

`TraditionalMBARAnalyzer.solve` 在 overlap 不足时**不抛异常**，只返回
`converged=False` + `min_overlap`（以及有限的 ΔG）。而 `_run_2d_lambda_stage`
组装 `stage_result` 时只复制 `delta_G`/`error`/`diagnostics`，把收敛状态丢掉。
后果：不收敛的采样结果照样被写进 `sampling_single_lambda.json` /
`sampling_2d_diagonal.json` / `sampling_2d_geodesic.json`、被标
`completed`、拼出最终 ΔG_bind，并在 resume 时被原样复用。

## 修法（2026-08-30）

- `_run_2d_lambda_stage` 透传 `converged`/`min_overlap`/`min_overlap_threshold`，
  并在返回前硬检查（`_assert_sampling_result_converged`）；
- 三条路径的缓存写入前过同一 sanity gate；缓存命中也走同一 gate——缓存里
  `converged` 不是 True（含旧缓存缺字段）一律拒绝复用、重新采样。

## 不要这样让本文件变绿

让 gate 在 converged 缺席时放行、或把拒绝改成打 warning 后继续写缓存。
"""

from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

import abfe_pipeline as pl
from abfe_pipeline import (
    _assert_sampling_result_converged,
    _sampling_result_convergence_rejection_reason,
)

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "abfe_pipeline.py"


CONVERGED_RESULT = {
    "total_delta_G": -3.2,
    "total_error": 0.4,
    "converged": True,
    "min_overlap": 0.2,
    "min_overlap_threshold": 0.03,
}


# ---------------------------------------------------------------------------
# 1. gate 本体
# ---------------------------------------------------------------------------


def test_converged_result_passes_the_gate():
    assert _sampling_result_convergence_rejection_reason(dict(CONVERGED_RESULT)) is None
    _assert_sampling_result_converged(dict(CONVERGED_RESULT), context="t")


def test_finite_delta_G_with_converged_false_is_rejected():
    """验收口径：solver 返回有限 ΔG 但 converged=false ⟹ 拒绝。"""
    bad = dict(CONVERGED_RESULT)
    bad["converged"] = False
    reason = _sampling_result_convergence_rejection_reason(bad)
    assert reason is not None and "converged" in reason
    with pytest.raises(RuntimeError, match="P1-11"):
        _assert_sampling_result_converged(bad, context="t")


def test_missing_converged_field_is_rejected():
    """旧缓存缺 converged 字段 ⟹ 拒绝复用（fail closed，不因缺字段放行）。"""
    bad = {"total_delta_G": -3.2, "total_error": 0.4}
    reason = _sampling_result_convergence_rejection_reason(bad)
    assert reason is not None and "converged" in reason


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0])
def test_nonfinite_or_negative_error_is_rejected(bad_value):
    bad = dict(CONVERGED_RESULT)
    bad["total_error"] = bad_value
    assert _sampling_result_convergence_rejection_reason(bad) is not None


# ---------------------------------------------------------------------------
# 2. 接线：三条路径的写缓存与缓存命中都过同一 gate（静态守护）
# ---------------------------------------------------------------------------


def _run_full_pipeline_source() -> str:
    import ast

    src = PIPELINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    pipeline_class = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ABFEPipeline"
    )
    method = next(
        n for n in pipeline_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_full_pipeline"
    )
    return ast.get_source_segment(src, method)


def test_all_three_schemes_gate_the_write_and_the_cache_hit():
    src = _run_full_pipeline_source()
    for scheme in ("single_lambda", "2d_diagonal", "2d_geodesic"):
        sampling_file = f'sampling_{scheme}.json'
        assert sampling_file in src, f"{scheme} 的采样缓存文件名变了，请同步本测试"
    # 写侧：三条路径都必须在写缓存前 assert。
    assert src.count("_assert_sampling_result_converged(") >= 3, (
        "single_lambda/2d_diagonal/2d_geodesic 的写缓存路径都必须过收敛 gate"
    )
    # 读侧：三条路径的命中条件都必须调用同一拒绝原因函数。
    assert src.count(
        "_sampling_result_convergence_rejection_reason(cached_sample) is None"
    ) >= 3, (
        "single_lambda/2d_diagonal/2d_geodesic 的缓存命中都必须过同一收敛 gate"
    )


def test_stage_result_passes_convergence_state_through():
    """`_run_2d_lambda_stage` 的 stage_result 必须透传 converged/min_overlap。"""
    import ast

    src = PIPELINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    pipeline_class = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "ABFEPipeline"
    )
    method = next(
        n for n in pipeline_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "_run_2d_lambda_stage"
    )
    method_src = ast.get_source_segment(src, method)
    assert '"converged"' in method_src, (
        "_run_2d_lambda_stage 仍丢弃 MBAR 收敛状态（P1-11）"
    )
    assert '"min_overlap"' in method_src
    assert "_assert_sampling_result_converged(" in method_src, (
        "stage 返回前必须硬检查收敛状态"
    )
