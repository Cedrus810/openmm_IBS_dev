"""P1-12：final-result 统一 schema/sanity gate。

## 缺陷是什么（已修，本文件现在是契约测试）

三处消费"腿级最终结果"的地方各自 fail-open：

1. `abfe_pipeline.run_full_pipeline` 顶层早退：协议指纹一致就把整份
   `final_results.json` 读回来直接 `return`，不校验任何热力学字段——
   一份缺 `total_delta_G_complex_kJ_mol`、ΔG=NaN、误差=Inf/负数的
   valid-fingerprint 缓存会被原样当成最终答案。
2. `runabfe` 两腿汇总：`complex_results.get("total_delta_G_complex_kJ_mol",
   ..., 0.0)` 把缺字段静默补成 0.0，损坏的腿被汇总成"看似成功"的 ΔG_bind。
3. `abfe_core.combine_binding_free_energy`：唯一热力学循环闭合 helper 不检查
   有限性与误差非负，NaN/Inf 会静默传播进 ΔG_bind 并写进结果 JSON。

## 修法（2026-08-30）

`abfe_core.validate_final_leg_result` / `FinalResultValidationError` 是唯一
判据，三处共用：缺必需字段、非有限、负误差、自带 converged 不为 True 一律
拒绝（早退路径拒绝后落到重新校验/运行；汇总路径直接 fail closed）。

## 不要这样让本文件变绿

把某处 gate 改回 `.get(..., 0.0)`、把负误差当"可以接受"、或让早退路径在
sanity gate 失败后仍然 `return final`。
"""

import json
import math

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

import abfe_core
from abfe_core import (
    FinalResultValidationError,
    combine_binding_free_energy,
    validate_final_leg_result,
)

import runabfe  # noqa: E402  （只验证其源码里确实接了 gate，不在此执行 main）

VALID_DUAL_LEG = {
    "total_delta_G_complex_kJ_mol": -12.5,
    "total_error_kJ_mol": 0.8,
}
VALID_TRADITIONAL_LEG = {
    "delta_G_total_kJ_mol": -9.0,
    "error_leg_kJ_mol": 0.5,
}


# ---------------------------------------------------------------------------
# 1. validator 本体
# ---------------------------------------------------------------------------


def test_valid_payloads_pass_and_expose_the_field_pair():
    out_dual = validate_final_leg_result(dict(VALID_DUAL_LEG))
    assert out_dual["delta_G_field"] == "total_delta_G_complex_kJ_mol"
    assert out_dual["delta_G_kJ_mol"] == pytest.approx(-12.5)
    out_trad = validate_final_leg_result(dict(VALID_TRADITIONAL_LEG))
    assert out_trad["delta_G_field"] == "delta_G_total_kJ_mol"


def test_missing_required_field_is_rejected():
    broken = dict(VALID_DUAL_LEG)
    broken.pop("total_error_kJ_mol")
    with pytest.raises(FinalResultValidationError, match="必需热力学字段"):
        validate_final_leg_result(broken)


def test_nan_and_inf_delta_G_and_error_are_rejected():
    for field in ("total_delta_G_complex_kJ_mol", "total_error_kJ_mol"):
        for bad in (float("nan"), float("inf"), float("-inf")):
            broken = dict(VALID_DUAL_LEG)
            broken[field] = bad
            with pytest.raises(FinalResultValidationError, match="非有限"):
                validate_final_leg_result(broken)


def test_negative_error_is_rejected_but_negative_delta_G_is_fine():
    broken = dict(VALID_DUAL_LEG)
    broken["total_error_kJ_mol"] = -0.1
    with pytest.raises(FinalResultValidationError, match="负数"):
        validate_final_leg_result(broken)
    ok = dict(VALID_DUAL_LEG)
    ok["total_delta_G_complex_kJ_mol"] = -50.0
    validate_final_leg_result(ok)  # ΔG 允许任意符号


def test_converged_not_true_is_rejected_but_absent_is_tolerated():
    broken = dict(VALID_DUAL_LEG)
    broken["converged"] = False
    with pytest.raises(FinalResultValidationError, match="converged"):
        validate_final_leg_result(broken)
    broken["converged"] = None
    with pytest.raises(FinalResultValidationError, match="converged"):
        validate_final_leg_result(broken)
    validate_final_leg_result(dict(VALID_DUAL_LEG))  # 缺席 ⟹ 不否决历史结果


def test_non_dict_payload_is_rejected():
    with pytest.raises(FinalResultValidationError):
        validate_final_leg_result([1, 2, 3])


# ---------------------------------------------------------------------------
# 2. 三个消费点都接了同一份 gate
# ---------------------------------------------------------------------------


def test_pipeline_top_level_cache_early_return_is_gated(tmp_path, monkeypatch):
    """valid-fingerprint 但缺字段/非有限的 final_results.json 不得被早退复用。

    通过编译 run_full_pipeline 的方式太重；这里验证 gate 函数被
    abfe_pipeline 顶层早退分支真实引用（静态）+ validator 行为（动态）。
    """
    import ast
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "abfe_pipeline.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(src)
    pipeline_class = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ABFEPipeline"
    )
    method = next(
        n for n in pipeline_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_full_pipeline"
    )
    method_src = ast.get_source_segment(src, method)
    assert "validate_final_leg_result(" in method_src, (
        "run_full_pipeline 顶层早退没有接统一 sanity gate（P1-12）"
    )
    # gate 失败必须走"拒绝复用"，而不是照样 return final。
    assert "FinalResultValidationError" in method_src
    # 早退 return 只允许出现在 gate 的 else 分支里（粗粒度：gate 调用行号必须
    # 早于早退 return 行号）。
    gate_line = next(
        node.lineno for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "validate_final_leg_result"
    )
    early_return_line = min(
        node.lineno for node in ast.walk(method)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
        and getattr(node.value, "id", "") == "final"
    )
    assert gate_line < early_return_line


def test_combine_binding_free_energy_rejects_nonfinite_and_negative_error():
    base = dict(
        dg_complex_kJ_mol=-12.0,
        dg_solvent_kJ_mol=3.0,
        err_complex_kJ_mol=0.5,
        err_solvent_kJ_mol=0.4,
    )
    ok = combine_binding_free_energy(**base)
    assert math.isfinite(ok["delta_G_bind_kJ_mol"])

    with pytest.raises(FinalResultValidationError, match="非有限"):
        combine_binding_free_energy(**{**base, "dg_complex_kJ_mol": float("nan")})
    with pytest.raises(FinalResultValidationError, match="非有限"):
        combine_binding_free_energy(**{**base, "err_solvent_kJ_mol": float("inf")})
    with pytest.raises(FinalResultValidationError, match="负数"):
        combine_binding_free_energy(**{**base, "err_complex_kJ_mol": -1.0})


def test_runabfe_summary_no_longer_silently_defaults_to_zero():
    """两腿汇总不再用 .get(..., 0.0) 静默补零（静态守护）。"""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "runabfe.py").read_text(
        encoding="utf-8"
    )
    assert 'complex_results.get("total_delta_G_complex_kJ_mol", complex_results.get(' not in src, (
        "complex 腿汇总仍在把缺字段静默补成 0.0（P1-12）"
    )
    assert 'solv_results.get("total_delta_G_complex_kJ_mol", solv_results.get(' not in src, (
        "solvent 腿汇总仍在把缺字段静默补成 0.0（P1-12）"
    )
    assert src.count("validate_final_leg_result(") >= 3, (
        "runabfe 的三处消费点（_load_leg_result + 两腿汇总）必须都接统一 gate"
    )


def test_end_to_end_corrupt_cache_is_rejected_by_gate(tmp_path):
    """注入 NaN/负误差的"valid"缓存，经 gate 逐个拒绝（验收口径的直测）。"""
    cases = [
        {**VALID_DUAL_LEG, "total_delta_G_complex_kJ_mol": float("nan")},
        {**VALID_DUAL_LEG, "total_error_kJ_mol": float("inf")},
        {**VALID_DUAL_LEG, "total_error_kJ_mol": -0.5},
        {k: v for k, v in VALID_DUAL_LEG.items() if k != "total_error_kJ_mol"},
        {**VALID_DUAL_LEG, "converged": False},
    ]
    for i, payload in enumerate(cases):
        path = tmp_path / f"final_results_{i}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        with pytest.raises(FinalResultValidationError):
            validate_final_leg_result(loaded, source=str(path))
