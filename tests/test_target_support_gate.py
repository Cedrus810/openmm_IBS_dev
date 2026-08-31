"""TARGET_SUPPORT_GATE_PROTOCOL_VERSION=1 回归。

背景：STAGE2_ROOT_CAUSE_2026-08-28.md。4W53（T4L99A + 甲苯）的 stage2 溶剂腿
报出 `+35.61 kJ/mol`，独立参考算例（不 import 仓库任何模块、同一份
`system_solvent.xml`、同一条 λ 路径）给的是 `-6.29 ± 0.34`，差 **+41.9 kJ/mol**。
而当时**所有五道收敛门全部通过**：

    converged                = True
    min_overlap              = 0.4684   (阈值 0.05)   ← mixture 覆盖度
    raw_min_overlap          = 0.0196                 ← 设计上"只报告不设门"
    min_decorrelated_samples = 332      (阈值 20)
    split_half_max_window_z  = 2.004
    statistical_inefficiency = [1.424, 1.158, 1.506]

原因是受门的 `min_ess_ratio` 是**共模因子被除掉之后**的混合覆盖度：它衡量
"已采到的帧在窗口内各 λ 态之间分不分得开"，而不是"这批被 Group-4 λ-WCA 防护壳
偏置过、整窗只有一条轨迹的采样，能不能重加权到没有防护壳的真实物理系综"。
本文件锁住的就是后面那份证据现在必须是硬门。

⚠️ 这道门是止血不是根治。该文档 §3.2 的 window 2 相邻 ⟨ΔU⟩ 只有 0.4~0.6 kT，
任何基于能量的重叠判据都判优，却错得最多（+19.49 kJ/mol）——失效模式是"该采的
构型一次都没采到"。根治方案见该文档 §8.2。
"""

import numpy as np
import pytest

from ibs_engine import (
    TARGET_SUPPORT_GATE_PROTOCOL_VERSION,
    TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT,
    TARGET_SUPPORT_MIN_ABSOLUTE_ESS,
    solve_stage_integrated,
)

KT = 0.008314462618 * 298.15


def _logsumexp_rows(a, axis=0):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def _make_window(n_states, n_frames, common_mode_sigma_kT, seed):
    """一个混合覆盖度健康、但 raw 目标支撑度由 `common_mode_sigma_kT` 控制的窗口。

    构造方式就是真实机制本身：记录下来的采样偏置是
        bias_n = IBS 混合偏置(u, f_k) + r_n
    其中 `r_n` 是任何 (U'_k, f_k) 组合都复现不出来的共模残差（防护壳 + LRC 口径
    差）。`r_n` 只整体平移每一帧的 raw 权重，**完全不改变**逐帧的态间相对权重
    `p_k`——所以 mixture 覆盖度不受影响，而 raw 单参考 ESS 按 exp(σ_r²) 被压掉。
    这正是 4W53 落盘数据的形状：mixture 0.4684 / raw 0.0196。
    """
    rng = np.random.default_rng(seed)
    # 各态之间只差几分之一 kT —— 混合覆盖度会很健康。
    offsets = np.linspace(0.0, 1.5 * KT, n_states)
    u_kn = offsets[:, None] + rng.normal(0.0, 0.5 * KT, size=(n_states, n_frames))
    f_k = np.zeros(n_states, dtype=float)

    logits = -(u_kn - f_k[:, None]) / KT
    ibs_bias = -KT * _logsumexp_rows(logits, axis=0)
    residual = rng.normal(0.0, float(common_mode_sigma_kT) * KT, size=n_frames)
    bias = ibs_bias + residual
    return {
        "u_kn": u_kn,
        "bias_energies": bias,
        "base_energies": rng.normal(0.0, 5.0, size=n_frames),
        "lambda_indices": list(range(n_states)),
        "f_k": f_k,
        "window_index": 0,
        "window_range": [0, n_states],
    }


def test_healthy_window_passes_the_target_support_gate():
    res = solve_stage_integrated(
        [_make_window(4, 600, common_mode_sigma_kT=0.0, seed=11)],
        KT,
        stage_name="vanishing",
    )
    gate = res["target_support_gate"]
    assert gate["passed"] is True, gate
    assert gate["failure_reason"] is None
    assert res["converged"] is True


def test_high_mixture_coverage_with_low_raw_target_support_fails():
    """4W53 的失效形状：mixture 覆盖度很高，raw 目标支撑度塌掉 → 必须判失败。"""
    res = solve_stage_integrated(
        [_make_window(4, 600, common_mode_sigma_kT=2.5, seed=11)],
        KT,
        stage_name="vanishing",
    )
    # 旧的受门指标依然"健康"——这正是当初放行错值的原因。
    assert res["min_overlap"] > res["min_overlap_threshold"]
    gate = res["target_support_gate"]
    assert gate["passed"] is False, gate
    assert gate["failure_reason"] == "insufficient_target_support"
    assert gate["failed_checks"], gate
    assert res["converged"] is False
    # raw 与 mixture 必须真的分道扬镳，否则这个用例没测到该测的东西。
    assert res["raw_min_overlap"] < 0.1 * res["min_overlap"]


def test_frame_count_mismatch_is_rejected_outright():
    """帧数对不上时求解器直接拒绝，不会产出一份"证据缺失但 converged=True"的结果。

    这是 raw 支撑度证据可能缺失的最主要入口；它在更早的地方就 fail 掉了，
    所以本文件只需锁住"绝不静默放行"这一点。
    """
    win = _make_window(4, 600, common_mode_sigma_kT=0.0, seed=11)
    win["bias_energies"] = win["bias_energies"][:-1]
    with pytest.raises(ValueError, match="帧数不一致"):
        solve_stage_integrated([win], KT, stage_name="vanishing")


def test_incomplete_evidence_is_not_a_pass():
    """证据不全（某个窗口算不出 raw 支撑度）必须当失败处理，不是"没测到就放行"。"""
    from abfe_pipeline import ABFEPipeline

    class _Dummy:
        _stage_quality_failure_details = staticmethod(
            ABFEPipeline._stage_quality_failure_details
        )
        _format_stage_quality_failure_details = staticmethod(
            ABFEPipeline._format_stage_quality_failure_details
        )

    result = {
        "stage": "vanishing",
        "converged": False,
        "total_delta_G": 1.0,
        "total_error": 0.1,
        "min_overlap": 0.5,
        "min_overlap_threshold": 0.05,
        "window_overlap_diagnostics": [],
        "target_support_gate": {
            "passed": False,
            "failure_reason": "insufficient_target_support",
            "failed_checks": ["incomplete_evidence"],
            "raw_min_absolute_ess": None,
            "raw_min_absolute_ess_threshold": TARGET_SUPPORT_MIN_ABSOLUTE_ESS,
            "max_top1pct_raw_weight": None,
            "max_top1pct_raw_weight_threshold": TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT,
        },
    }
    with pytest.raises(RuntimeError, match="insufficient_target_support"):
        ABFEPipeline._assert_stage_result_sane(_Dummy(), "Stage 2 (vanishing)", result)


def test_thresholds_are_reported_alongside_the_verdict():
    """阈值必须跟判定一起落盘，否则事后无法回答"当时用的是哪套门槛"。"""
    res = solve_stage_integrated(
        [_make_window(4, 600, common_mode_sigma_kT=0.0, seed=11)],
        KT,
        stage_name="vanishing",
    )
    assert res["target_support_gate_protocol_version"] == TARGET_SUPPORT_GATE_PROTOCOL_VERSION
    assert res["raw_min_absolute_ess_threshold"] == pytest.approx(
        TARGET_SUPPORT_MIN_ABSOLUTE_ESS
    )
    assert res["max_top1pct_raw_weight_threshold"] == pytest.approx(
        TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT
    )


def test_thresholds_are_configurable_and_actually_take_effect():
    win = _make_window(4, 600, common_mode_sigma_kT=0.0, seed=11)
    strict = solve_stage_integrated(
        [win], KT, stage_name="vanishing",
        final_min_target_absolute_ess=1e9,
    )
    assert strict["target_support_gate"]["passed"] is False
    assert "raw_absolute_ess_below_threshold" in strict["target_support_gate"]["failed_checks"]
    assert strict["converged"] is False


def test_gate_thresholds_are_part_of_the_stage_protocol_fingerprint():
    """阈值必须进 stage 协议指纹，否则升级前写下的 "completed" 缓存会被静默复用。"""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "abfe_pipeline.py").read_text(
        encoding="utf-8"
    )
    block = re.search(r"_final_gate_thresholds = \{(.*?)\n            \}", source, re.S)
    assert block is not None, "找不到 _final_gate_thresholds 的构造"
    body = block.group(1)
    assert "final_min_target_absolute_ess" in body
    assert "final_max_top1pct_raw_weight" in body
    assert "target_support_gate_protocol_version" in body


def test_pipeline_refuses_a_vanishing_result_without_the_gate():
    """升级前的求解路径产出的 stage2 结果，其 raw 支撑度从未被判定过 → 拒绝。"""
    from abfe_pipeline import ABFEPipeline

    class _Dummy:
        _stage_quality_failure_details = staticmethod(
            ABFEPipeline._stage_quality_failure_details
        )
        _format_stage_quality_failure_details = staticmethod(
            ABFEPipeline._format_stage_quality_failure_details
        )

    legacy = {
        "stage": "vanishing",
        "converged": True,
        "total_delta_G": 35.61,
        "total_error": 0.84,
        "min_overlap": 0.4684,
        "min_overlap_threshold": 0.05,
        "min_decorrelated_samples": 332,
        "min_decorrelated_samples_threshold": 20,
        "max_endpoint_uncertainty_kJ_mol": 0.75,
        "max_endpoint_uncertainty_kJ_mol_threshold": 1.0,
        "window_overlap_diagnostics": [],
    }
    with pytest.raises(RuntimeError, match="target_support_gate"):
        ABFEPipeline._assert_stage_result_sane(_Dummy(), "Stage 2 (vanishing)", legacy)


def test_pipeline_attributes_the_failure_to_the_right_windows():
    """逐窗口归因必须点名 raw 支撑度这两项，而不是笼统地说"重叠不足"。

    数字取自 4W53 溶剂腿落盘的 window_overlap_diagnostics：
    win0 raw_ESS=85.9 / top1%=0.118（过），win1 8.48 / 0.552（挂），
    win2 9.57 / 0.469（挂）。
    """
    from abfe_pipeline import ABFEPipeline

    records = [
        {"window_index": 0, "min_ess_ratio": 0.468, "absolute_ess": 164.0,
         "n_frames_decorrelated": 351, "endpoint_diff_uncertainty_kJ_mol": 0.2,
         "raw_min_absolute_ess": 85.93, "top1pct_raw_weight": [0.080, 0.069, 0.118],
         "lambdas": [0, 1, 2]},
        {"window_index": 1, "min_ess_ratio": 0.479, "absolute_ess": 207.0,
         "n_frames_decorrelated": 432, "endpoint_diff_uncertainty_kJ_mol": 0.4,
         "raw_min_absolute_ess": 8.48,
         "top1pct_raw_weight": [0.162, 0.311, 0.408, 0.499, 0.552],
         "lambdas": [2, 3, 4, 5, 6]},
        {"window_index": 2, "min_ess_ratio": 0.894, "absolute_ess": 297.0,
         "n_frames_decorrelated": 332, "endpoint_diff_uncertainty_kJ_mol": 0.75,
         "raw_min_absolute_ess": 9.57,
         "top1pct_raw_weight": [0.330, 0.361, 0.397, 0.431, 0.454, 0.469],
         "lambdas": [6, 7, 8, 9, 10, 11]},
    ]
    details = ABFEPipeline._stage_quality_failure_details({
        "window_overlap_diagnostics": records,
        "raw_min_absolute_ess_threshold": TARGET_SUPPORT_MIN_ABSOLUTE_ESS,
        "max_top1pct_raw_weight_threshold": TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT,
    })
    failed = {d["window_index"]: d["failed_gates"] for d in details}
    assert 0 not in failed, "window 0 的 raw 支撑度是健康的，不该被点名"
    assert set(failed) == {1, 2}
    for idx in (1, 2):
        assert "target_support_raw_absolute_ess" in failed[idx]
        assert "target_support_top1pct_raw_weight" in failed[idx]
