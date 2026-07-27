"""P1-15 回归：stage 结果的收敛/覆盖证据必须真的落盘。

2026-07-27 的 vdW 腿报出 `ΔG_vdw = 145.908 ± 1.384 kJ/mol`，**却没有任何审计痕迹**：

  * `final_results.json` 的 `diagnostics = {}`、`stage_diagnostics.stage2 = {}`
    （`stage1` 是满的，对比明显）
  * `checkpoints/stage2_vanishing.json` 只有 stage/total_delta_G/total_error/
    n_states/protocol_key
  * `immutable_bridge_rescue` 全盘搜索零命中——它只存在于内存
  * `pipeline.log` 在 11:48:00–12:12:21 之间一行都没有

根因：vanishing rescue 合并那条路径直接调 `solve_stage_integrated`，
**绕过 `_run_ibs_stage`**，于是那段 `stage_result["diagnostics"].update({...})`
从来没执行过；而 `_build_stage_cache_payload` 存的是 `result.get("diagnostics", {})`，
落盘一个空字典。

第二个坑：`_atomic_write_json` 用的是不带 `cls=NumpyEncoder` 的 `json.dump`。
`solve_stage_integrated` 返回的 `window_overlap_diagnostics` / `f_k` /
`coverage_diagnostics` 都含 numpy，把它们纳入落盘范围而不做转换，
会让整个 checkpoint 写入 `TypeError` 失败——比丢诊断更糟。
"""

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# _json_safe
# ---------------------------------------------------------------------------


def test_json_safe_handles_everything_solve_stage_integrated_returns():
    from abfe_pipeline import _json_safe

    payload = {
        "f_k": np.array([1.0, 2.0, 3.0]),
        "n": np.int64(7),
        "x": np.float64(1.5),
        "flag": np.bool_(True),
        "nested": [{"m": np.array([[1.0, 2.0]])}, (np.float32(0.5),)],
        "plain": "ok",
    }
    safe = _json_safe(payload)
    # 关键断言：能被不带 cls= 的 json.dump 直接写出去。
    text = json.dumps(safe)
    back = json.loads(text)
    assert back["f_k"] == [1.0, 2.0, 3.0]
    assert back["n"] == 7 and isinstance(back["n"], int)
    assert back["x"] == 1.5
    assert back["flag"] is True
    assert back["nested"][0]["m"] == [[1.0, 2.0]]
    assert back["plain"] == "ok"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_json_safe_maps_nonfinite_to_null(bad):
    """NaN/Inf 会被 json.dump 写成非标准 JSON，别的工具读不了。"""
    from abfe_pipeline import _json_safe

    safe = _json_safe({"v": bad, "arr": np.array([1.0, bad])})
    text = json.dumps(safe)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["v"] is None
    assert json.loads(text)["arr"][1] is None


def test_json_safe_is_applied_by_build_stage_cache_payload():
    """真正的回归：把含 numpy 的诊断塞进去，payload 必须仍可直接序列化。"""
    from abfe_pipeline import ABFEPipeline

    result = {
        "total_delta_G": np.float64(145.9),
        "total_error": np.float64(1.38),
        "method": "Local-TMBAR covariance-chain (ESS-overlap-checked)",
        "converged": np.bool_(True),
        "coverage_diagnostics": {"covered_lambda_indices": np.arange(23)},
        "diagnostics": {
            "window_overlap_diagnostics": [
                {"min_ess_ratio": np.float64(0.5), "f_k": np.zeros(3)}
            ],
            "max_endpoint_uncertainty_kJ_mol": np.float64(float("nan")),
        },
        "lambda_endpoint_diagnostics": {},
    }
    payload = ABFEPipeline._build_stage_cache_payload(
        "vanishing", result, 23, {"schema_version": 1},
        [1.0, 0.5, 0.0], [(0, 2), (1, 3)],
    )
    text = json.dumps(payload, indent=2)  # 刻意不传 cls=，复刻 _atomic_write_json
    assert "NaN" not in text
    back = json.loads(text)
    assert back["converged"] is True
    assert back["coverage_diagnostics"]["covered_lambda_indices"][-1] == 22
    assert back["diagnostics"]["max_endpoint_uncertainty_kJ_mol"] is None


def test_build_stage_cache_payload_persists_convergence_evidence():
    """converged / coverage_diagnostics 必须真的进 payload，不能只留在内存。"""
    from abfe_pipeline import ABFEPipeline

    result = {
        "total_delta_G": 1.0, "total_error": 0.1,
        "converged": False,
        "coverage_diagnostics": {"dropped_window_indices": [3]},
        "diagnostics": {"min_overlap": 0.02},
        "lambda_endpoint_diagnostics": {},
    }
    payload = ABFEPipeline._build_stage_cache_payload(
        "vanishing", result, 23, {}, [1.0, 0.0], [(0, 2)]
    )
    assert payload["converged"] is False
    assert payload["coverage_diagnostics"]["dropped_window_indices"] == [3]


def _pipeline_without_init():
    from abfe_pipeline import ABFEPipeline

    return ABFEPipeline.__new__(ABFEPipeline)


def test_reusable_stage_cache_rehydrates_and_rechecks_every_gate():
    pipeline = _pipeline_without_init()
    cached = {
        "stage": "vanishing",
        "total_delta_G": 12.0,
        "total_error": 0.5,
        "converged": True,
        "coverage_diagnostics": {"covered_lambda_indices": [0, 1, 2]},
        "diagnostics": {
            "min_overlap": 0.20,
            "min_overlap_threshold": 0.05,
            "min_decorrelated_samples": 50,
            "min_decorrelated_samples_threshold": 20,
            "max_endpoint_uncertainty_kJ_mol": 0.5,
            "max_endpoint_uncertainty_kJ_mol_threshold": 1.0,
            "window_overlap_diagnostics": [],
        },
    }
    pipeline._assert_reusable_stage_cache_sane("Stage 2", cached)
    assert cached["min_overlap"] == 0.20
    assert cached["min_decorrelated_samples"] == 50


@pytest.mark.parametrize(
    "cached, message",
    [
        (
            {
                "stage": "vanishing",
                "total_delta_G": 1.0,
                "total_error": 0.1,
                "converged": False,
                "coverage_diagnostics": {},
                "diagnostics": {},
            },
            "converged=True",
        ),
        (
            {
                "stage": "vanishing",
                "total_delta_G": 1.0,
                "total_error": 0.1,
                "converged": True,
                "coverage_diagnostics": None,
                "diagnostics": {},
            },
            "coverage_diagnostics",
        ),
        (
            {
                "stage": "vanishing",
                "total_delta_G": 1.0,
                "total_error": 0.1,
                "converged": True,
                "coverage_diagnostics": {},
                "diagnostics": {
                    "min_overlap": 0.01,
                    "min_overlap_threshold": 0.05,
                },
            },
            "低于阈值",
        ),
    ],
)
def test_reusable_stage_cache_fails_closed(cached, message):
    pipeline = _pipeline_without_init()
    with pytest.raises(RuntimeError, match=message):
        pipeline._assert_reusable_stage_cache_sane("Stage 2", cached)


def test_both_stage_resume_branches_recheck_cached_scientific_gates():
    source = (REPO_ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    assert source.count("self._assert_reusable_stage_cache_sane(") == 2


# ---------------------------------------------------------------------------
# _populate_stage_diagnostics
# ---------------------------------------------------------------------------


REQUIRED_DIAGNOSTIC_KEYS = [
    "converged",
    "coverage_diagnostics",
    "covariance_chain_segments",
    "min_overlap",
    "min_overlap_threshold",
    "min_occupancy_normalized",
    "min_decorrelated_samples",
    "max_endpoint_uncertainty_kJ_mol",
    "max_endpoint_uncertainty_kJ_mol_threshold",
    "min_absolute_ess",
    "window_overlap_diagnostics",
    "immutable_bridge_rescue",
    "production_rescue_targets",
]


def test_populate_stage_diagnostics_carries_every_gate():
    from abfe_pipeline import ABFEPipeline

    result = {
        "total_delta_G": 145.9, "total_error": 1.38, "converged": True,
        "min_overlap": 0.5, "min_overlap_threshold": 0.05,
        "min_occupancy_normalized": 0.9,
        "min_decorrelated_samples": 142,
        "max_endpoint_uncertainty_kJ_mol": 0.7,
        "max_endpoint_uncertainty_kJ_mol_threshold": 1.0,
        "min_absolute_ess": 73.6, "min_absolute_ess_threshold": None,
        "covariance_chain_segments": [{"delta_G_kJ_mol": 1.0}],
        "coverage_diagnostics": {"covered_lambda_indices": [0, 1]},
        "window_overlap_diagnostics": [{"min_ess_ratio": 0.5}],
        "immutable_bridge_rescue": {"plan_id": "c3594e2d792a"},
        "production_rescue_targets": {2: 1000000},
    }
    ABFEPipeline._populate_stage_diagnostics(result)
    diag = result["diagnostics"]
    for key in REQUIRED_DIAGNOSTIC_KEYS:
        assert key in diag, f"diagnostics 缺 {key}——归档结果将无法复核为何放行"
    assert diag["converged"] is True
    assert diag["immutable_bridge_rescue"]["plan_id"] == "c3594e2d792a"


def test_populate_stage_diagnostics_does_not_clobber_existing_entries():
    from abfe_pipeline import ABFEPipeline

    result = {"diagnostics": {"pme_self_correction": {"applied": False}}}
    ABFEPipeline._populate_stage_diagnostics(result)
    assert result["diagnostics"]["pme_self_correction"] == {"applied": False}


# ---------------------------------------------------------------------------
# 调用点契约：rescue 分支必须也调
# ---------------------------------------------------------------------------


# `_run_shadow_ibs_decharging_leg` 也直调 solve_stage_integrated，但**不适用**这条契约：
# 它返回的是 bridge 腿 + shadow-IBS 腿的**组合**结果，顶层 total_delta_G 是两段相加，
# 顶层并不存在 min_decorrelated_samples / covariance_chain_segments 这些量——照搬会
# 写进一堆 None。它自己把完整子结果嵌在 `diagnostics.shadow_ibs_leg` 里，证据没有丢。
# 这是有意的豁免，不是漏网。
_EXEMPT_FROM_POPULATE_CONTRACT = {"_run_shadow_ibs_decharging_leg"}


def _solve_and_populate_by_function():
    """按**所在函数**统计两种调用的行号，而不是全文件行号。

    早先版本只断言"文件里 solve 之后某处有 populate"——那会因为行号先后顺序
    平凡通过，等于没测。必须限定在同一个函数体内。
    """
    source = (REPO_ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="abfe_pipeline.py")
    out = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        solves, populates = [], []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Name) and f.id == "solve_stage_integrated":
                solves.append(node.lineno)
            elif (isinstance(f, ast.Attribute) and f.attr == "_populate_stage_diagnostics") or (
                isinstance(f, ast.Name) and f.id == "_populate_stage_diagnostics"
            ):
                populates.append(node.lineno)
        if solves:
            out[fn.name] = (sorted(solves), sorted(populates))
    return out


def test_every_direct_solver_call_site_populates_diagnostics():
    """每个"直接从 solve_stage_integrated 造 stage 结果"的函数都必须填 diagnostics。

    这正是 P1-15 的根因：`run_full_pipeline` 里的 rescue 合并分支绕过了
    `_run_ibs_stage`，于是它的结果落盘成 `diagnostics={}`。
    """
    by_fn = _solve_and_populate_by_function()
    assert by_fn, "找不到任何 solve_stage_integrated 调用"

    offenders = []
    for name, (solves, populates) in by_fn.items():
        if name in _EXEMPT_FROM_POPULATE_CONTRACT:
            continue
        if not any(p > min(solves) for p in populates):
            offenders.append((name, solves, populates))
    assert not offenders, (
        "以下函数直接调了 solve_stage_integrated 却没有在**同一函数体内**调用 "
        f"_populate_stage_diagnostics：{offenders}。它们的结果会落盘成 "
        "diagnostics={}，事后无法复核为何放行（P1-15）。"
    )


def test_the_two_known_call_sites_are_both_covered():
    """钉住具体的两处，防止有人把 rescue 分支整个删了让上面那条空过。"""
    by_fn = _solve_and_populate_by_function()
    for name in ("_run_dual_lambda_stage", "run_full_pipeline"):
        assert name in by_fn, f"{name} 里不再有 solve_stage_integrated 调用？"
        solves, populates = by_fn[name]
        assert populates, f"{name} 缺 _populate_stage_diagnostics 调用"


def test_shadow_path_still_preserves_its_evidence_some_other_way():
    """豁免不等于可以什么都不留——shadow 路径必须仍把子结果嵌进 diagnostics。"""
    source = (REPO_ROOT / "abfe_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="abfe_pipeline.py")
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_run_shadow_ibs_decharging_leg"
    )
    keys = {
        k.value for n in ast.walk(fn) if isinstance(n, ast.Dict)
        for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }
    assert "shadow_ibs_leg" in keys and "bridge_leg" in keys, (
        "shadow 路径被豁免于 _populate_stage_diagnostics 契约，前提是它自己把两段"
        "子结果嵌在 diagnostics 里；现在这个前提不成立了"
    )
