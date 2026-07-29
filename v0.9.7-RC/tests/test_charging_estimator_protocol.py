#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""charging（stage1 / decharging 腿）估计量协议的防回归测试。

**背景（改这块之前必读）**

2026-07-28 把 charging 的主值从**去相关 MBAR** 换成**相邻 BAR**，一致性门用
**重加权 FD-TI**。依据是同一批 u_kn 上实测：

    相邻 BAR 65.0762±0.6148 / 重加权 FD-TI 65.1262 / 全帧 MBAR 65.0032  ← 三者一致
    去相关 MBAR 64.4113                                                  ← 偏低 0.6649

溶剂腿同样偏低 0.5252。根因是**自相关子采样导致有限样本点估计不稳定**（丢帧之后
选中的那个有限子集不稳），**不是「MBAR 本身有偏」**。

⛔ **这些测试刻意只覆盖 charging。vdW/stage2 只能用 TMBAR，本文件不对它做任何
断言、也不引入任何非 TMBAR 口径。** 第一条测试反过来保证了 vdW 不可能走 BAR：
`adjacent_bar_chain` 对零样本态 fail closed，而 IBS 每窗只有 row 0 有样本。

全部纯 numpy/pymbar，无 GPU、无 MD，毫秒~秒级。
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from ibs_engine import (
    ESTIMATOR_ANALYSIS_PROTOCOL_VERSION,
    TraditionalMBARAnalyzer,
    adjacent_bar_chain,
    estimator_policy_fingerprint,
    reweighted_fd_ti,
    stage1_estimator_crosschecks,
    stage1_ti_consistency_gate,
)

R_KJ = 0.008314462618
T_K = 300.0
KT = R_KJ * T_K


# ---------------------------------------------------------------------------
# 合成数据：势对 λ 线性 ⟹ 解析可解，用来钉死单位与符号
# ---------------------------------------------------------------------------

def _linear_reduced_u_kn(lambdas, n_per_state=400, slope_kJ=12.0, seed=20260728):
    """构造 u_kn（**约化**势）：U_k(x) = λ_k · slope · x，x ~ N(0,1) 的势能井。

    取 U_k = (λ_k·slope·x + x²/2·kT) / kT —— 二次项与 λ 无关，所以
    ΔG(λ_a→λ_b) 有解析解：对高斯积分，
        G(λ) = −kT·ln∫exp(−(λ·slope·x/kT + x²/2))dx = −(λ·slope)²/(2·kT) + const
    于是 ΔG = G(λ_last) − G(λ_first)，单位 kJ/mol。
    """
    rng = np.random.default_rng(seed)
    lam = np.asarray(lambdas, dtype=float)
    K = lam.size
    # 每个态从自己的偏移高斯里采样：x ~ N(−λ·slope/kT, 1)
    xs = [rng.normal(loc=-l * slope_kJ / KT, scale=1.0, size=n_per_state) for l in lam]
    x_all = np.concatenate(xs)
    n_k = np.full(K, n_per_state, dtype=int)
    u_kn = np.empty((K, x_all.size), dtype=float)
    for k in range(K):
        u_kn[k] = (lam[k] * slope_kJ * x_all) / KT + 0.5 * x_all ** 2
    return u_kn, n_k, lam, slope_kJ


def _analytic_delta_g_kJ(lam, slope_kJ):
    lam = np.asarray(lam, dtype=float)
    g = -((lam * slope_kJ) ** 2) / (2.0 * KT)
    return float(g[-1] - g[0])


# ---------------------------------------------------------------------------
# 1. vdW/stage2 永远不可能走 BAR —— 零样本态必须 fail closed
# ---------------------------------------------------------------------------

def test_adjacent_bar_rejects_ibs_style_n_k_with_only_row_zero_sampled():
    """IBS stage2 的 n_k 形状（只有 row 0 有样本）必须让 BAR 抛错。

    这是"vdW 只能用 TMBAR"的机械保证：IBS 每个窗口只有偏置混合分布真正有样本，
    物理 λ 行 n_k=0（见 `GlobalMBARAnalyzer.solve_stage_integrated` 里
    `n_k_local[sampled_row] = n_frames`）。BAR 要求两个端点系综各自有样本，
    把偏置混合帧当端点帧会给出**看似正常实则无效**的数——所以必须当场抛错，
    而不是返回一个能被误引用的值。
    """
    n_frames = 200
    K = 6                                  # 1 个采样行 + 5 个物理 λ 行
    u_kn = np.random.default_rng(0).normal(size=(K, n_frames))
    n_k = np.zeros(K, dtype=int)
    n_k[0] = n_frames                      # ← IBS 的形状

    with pytest.raises(ValueError) as exc:
        adjacent_bar_chain(u_kn, n_k, KT)
    msg = str(exc.value)
    assert "n_k=0" in msg, f"错误信息应点明零样本态: {msg}"
    # 错误信息必须解释清楚为什么 stage2 不能用，否则下一个人会以为是数据坏了
    assert "stage2" in msg and "TI" in msg, f"错误信息应说明 stage2 的 BAR/TI 都不适用: {msg}"


def test_adjacent_bar_rejects_shape_mismatch():
    """n_k 与 u_kn 帧数不符必须抛错，而不是静默算错一部分。"""
    u_kn = np.zeros((3, 10))
    with pytest.raises(ValueError):
        adjacent_bar_chain(u_kn, np.array([5, 5, 5]), KT)   # sum=15 != 10


# ---------------------------------------------------------------------------
# 2-3. FD-TI 的单位与 λ 网格
# ---------------------------------------------------------------------------

def test_fd_ti_multiplies_kt_because_stage1_u_kn_is_reduced():
    """stage1 的 u_kn 是**约化**势 ⟹ FD-TI 必须乘 kT 才是 kJ/mol。

    漏乘 kT 会让结果差 kT≈2.494 倍。这里同时断言"命中解析值"与"不等于漏乘版"，
    后者才是真正卡住这个 bug 的那半——只断言前者，容差稍松就放行了。
    """
    lam = [0.0, 0.25, 0.5, 0.75, 1.0]
    u_kn, n_k, lam_arr, slope = _linear_reduced_u_kn(lam)
    dg, dudl = reweighted_fd_ti(u_kn, n_k, lam_arr, KT)

    expected = _analytic_delta_g_kJ(lam_arr, slope)
    assert abs(dg - expected) < 0.05 * abs(expected), (
        f"FD-TI {dg:.4f} 应接近解析值 {expected:.4f} kJ/mol"
    )
    # 漏乘 kT 的那个值必须被明确排除
    missing_kt = dg / KT
    assert abs(missing_kt - expected) > 0.5 * abs(expected), (
        "漏乘 kT 的值不应也落在容差内，否则这条测试卡不住单位 bug"
    )
    assert len(dudl) == len(lam)


def test_fd_ti_uses_real_lambda_grid_not_state_index():
    """非均匀 λ 表：⟨∂U/∂λ⟩ 必须是对**真实 λ** 的导数。

    ⚠️ 这里刻意**不**断言"总 ΔG 用序号会算错"——梯形积分下总量对横坐标的重参数化
    近乎不变（分母的 Δλ 与积分步长的 Δλ 相互抵消），实测两者只差 0.2%。也就是说
    「用序号代替 λ」这个 bug **抓不到**在总量上，必须查逐态导数。
    解析解：G(λ) = −(λ·slope)²/(2kT) ⟹ dG/dλ = −λ·slope²/kT。
    """
    lam = [0.0, 0.05, 0.2, 0.6, 1.0]          # 刻意强非均匀
    u_kn, n_k, lam_arr, slope = _linear_reduced_u_kn(lam, n_per_state=800)

    dg_real, dudl_real = reweighted_fd_ti(u_kn, n_k, lam_arr, KT)
    _, dudl_index = reweighted_fd_ti(u_kn, n_k, np.arange(len(lam), dtype=float), KT)

    expected_dudl = -lam_arr * slope ** 2 / KT
    dudl_real = np.asarray(dudl_real)
    dudl_index = np.asarray(dudl_index)
    scale = float(np.max(np.abs(expected_dudl)))

    # 真实 λ 网格下逐态导数必须命中解析值
    assert np.max(np.abs(dudl_real - expected_dudl)) < 0.15 * scale, (
        f"真实 λ 下 dU/dλ={dudl_real} 应接近解析 {expected_dudl}"
    )
    # 用序号当 λ ⟹ 逐态导数被 Δλ/Δi 因子歪掉，必须明显偏离
    assert np.max(np.abs(dudl_index - expected_dudl)) > 0.5 * scale, (
        f"用态序号当 λ 竟然也给出正确的 dU/dλ（{dudl_index}），这条测试没有区分力"
    )
    # 总量仍应正确（这是它该有的性质，不是判据）
    assert abs(dg_real - _analytic_delta_g_kJ(lam_arr, slope)) < 0.05 * abs(
        _analytic_delta_g_kJ(lam_arr, slope)
    )


def test_fd_ti_descending_lambda_path_gives_positive_delta_g():
    """降序 λ（本仓库解耦腿的约定）符号必须为正，不得再翻一次。

    `np.trapz(y, x)` 沿给定顺序算 ∫_{x[0]}^{x[-1]}，所以降序 λ 直接就是
    ΔG(λ=1 → λ=0)。曾在这里多加过一次取负，会把 +65 变成 −65。
    """
    lam_desc = [1.0, 0.75, 0.5, 0.25, 0.0]
    u_kn, n_k, lam_arr, slope = _linear_reduced_u_kn(lam_desc)
    dg, _ = reweighted_fd_ti(u_kn, n_k, lam_arr, KT)
    assert dg > 0.0, f"降序 λ 的 ΔG(1→0) 应为正，得到 {dg:.4f}"
    expected = _analytic_delta_g_kJ(lam_arr, slope)
    assert abs(dg - expected) < 0.05 * abs(expected)


# ---------------------------------------------------------------------------
# 4-6. solve() 的主值切换、fail closed 与 TI 门
# ---------------------------------------------------------------------------

def test_solve_with_adjacent_bar_primary_takes_bar_and_records_all_four():
    """primary_estimator='adjacent_bar' ⟹ delta_G 来自 BAR，四个口径全部落盘。"""
    lam = [1.0, 0.75, 0.5, 0.25, 0.0]
    u_kn, n_k, lam_arr, _ = _linear_reduced_u_kn(lam)

    an = TraditionalMBARAnalyzer(temperature=T_K)
    an._last_n_k = n_k
    res = an.solve(
        u_kn, primary_estimator="adjacent_bar", lambdas=lam_arr,
        ti_gate_tolerance_kJ_mol=0.5,
    )

    bar_dg, bar_err, _ = adjacent_bar_chain(u_kn, n_k, KT)
    assert res["primary_estimator"] == "adjacent_bar"
    assert res["method"] == "adjacent-BAR-chain"
    assert res["delta_G"] == pytest.approx(bar_dg, abs=1e-12)
    assert res["error"] == pytest.approx(bar_err, abs=1e-12)
    # 旧主值必须保留可比痕迹，而不是被覆盖掉
    assert "delta_G_mbar_decorrelated_kJ_mol" in res
    cc = res["crosschecks"]
    for key in ("decorrelated_mbar", "full_frame_mbar", "adjacent_bar", "reweighted_fd_ti"):
        assert key in cc, f"crosschecks 缺 {key}"
        assert cc[key].get("delta_G_kJ_mol") is not None, f"{key} 没算出来"
    assert res["estimator_analysis_protocol_version"] == ESTIMATOR_ANALYSIS_PROTOCOL_VERSION
    assert isinstance(res["estimator_policy_fingerprint"], str)


def test_solve_fails_closed_when_bar_primary_requested_but_bar_unavailable():
    """要 BAR 主值却算不出 BAR ⟹ 必须抛错，**不得静默退回 MBAR 值**。

    静默退回是最危险的失败模式：结果看着正常，但 `primary_estimator` 字段说
    的是 BAR、数字却来自另一个估计量。
    """
    lam = [1.0, 0.5, 0.0]
    u_kn, n_k, lam_arr, _ = _linear_reduced_u_kn(lam, n_per_state=120)
    # 把中间态的样本数抹成 0（BAR 前提被破坏），但保持 sum(n_k) 与帧数一致：
    # 让它的帧归给第一个态。MBAR 仍能解（两个态有样本），BAR 必须拒绝。
    n_k_broken = n_k.copy()
    n_k_broken[0] += n_k_broken[1]
    n_k_broken[1] = 0

    an = TraditionalMBARAnalyzer(temperature=T_K)
    an._last_n_k = n_k_broken
    with pytest.raises(RuntimeError) as exc:
        an.solve(u_kn, primary_estimator="adjacent_bar", lambdas=lam_arr,
                 ti_gate_tolerance_kJ_mol=0.5)
    assert "adjacent_bar" in str(exc.value)


def test_ti_gate_fails_closed_on_real_disagreement_and_abstains_without_tolerance():
    """TI 门：分歧超容差 → converged=False；没给容差 → passed=None（不冒充通过）。"""
    # (a) 人为把 TI 值拉远，门必须判失败
    gate = stage1_ti_consistency_gate(dg_bar=65.0, err_bar=0.6, dg_ti=75.0,
                                      tolerance_kJ_mol=0.5)
    assert gate["passed"] is False
    assert gate["tolerance_kJ_mol"] == pytest.approx(max(0.5, 3 * 0.6))

    # (b) 实测量级（|BAR−TI| = 0.050 / 0.226）必须通过
    assert stage1_ti_consistency_gate(65.0762, 0.6148, 65.1262, 0.5)["passed"] is True
    assert stage1_ti_consistency_gate(63.4117, 0.6604, 63.6378, 0.5)["passed"] is True

    # (c) 没有 TI 值 ⟹ 不判通过
    assert stage1_ti_consistency_gate(65.0, 0.6, None, 0.5)["passed"] is None

    # (d) solve() 层面：不给容差时不得冒充通过
    lam = [1.0, 0.5, 0.0]
    u_kn, n_k, lam_arr, _ = _linear_reduced_u_kn(lam, n_per_state=200)
    an = TraditionalMBARAnalyzer(temperature=T_K)
    an._last_n_k = n_k
    res = an.solve(u_kn, primary_estimator="adjacent_bar", lambdas=lam_arr)
    assert res["ti_gate"]["passed"] is None
    assert "显式" in res["ti_gate"]["reason"]


def test_ti_gate_tolerance_is_not_inherited_from_attachment_constant():
    """charging 的容差必须与 attachment 的 1.0 kJ/mol 解耦（两条腿量级不同）。"""
    import abfe_pipeline
    from ibs_engine import ATTACHMENT_BAR_TI_ABS_TOL_KJ

    assert abfe_pipeline.CHARGING_TI_GATE_TOL_KJ_MOL != ATTACHMENT_BAR_TI_ABS_TOL_KJ
    kwargs = abfe_pipeline._CHARGING_ESTIMATOR_KWARGS([1.0, 0.5, 0.0])
    assert kwargs["primary_estimator"] == "adjacent_bar"
    assert kwargs["ti_gate_tolerance_kJ_mol"] == abfe_pipeline.CHARGING_TI_GATE_TOL_KJ_MOL
    assert kwargs["crosschecks"] is True
    # λ 必须是真实的 λ 值列表，不是态序号
    assert kwargs["lambdas"] == [1.0, 0.5, 0.0]


# ---------------------------------------------------------------------------
# 7. 默认参数必须逐位保持旧行为（回归锁）
# ---------------------------------------------------------------------------

def test_default_solve_behavior_is_bit_identical_to_legacy_path():
    """不传新参数时 delta_G/error/method 必须与历史行为逐位一致。

    `solve()` 加了 4 个新参数，默认值必须完全不改变既有 6 处调用点的结果。
    这里用"默认调用"与"显式声明历史口径"的两次结果对比，任何默认值漂移都会被抓到。
    """
    lam = [1.0, 0.6, 0.2, 0.0]
    u_kn, n_k, _, _ = _linear_reduced_u_kn(lam, n_per_state=300)

    an1 = TraditionalMBARAnalyzer(temperature=T_K)
    an1._last_n_k = n_k
    default = an1.solve(u_kn)

    an2 = TraditionalMBARAnalyzer(temperature=T_K)
    an2._last_n_k = n_k
    explicit = an2.solve(
        u_kn, decorrelate=True, primary_estimator="mbar_decorrelated",
        lambdas=None, ti_gate_tolerance_kJ_mol=None, crosschecks=False,
    )

    assert default["delta_G"] == explicit["delta_G"]
    assert default["error"] == explicit["error"]
    assert default["method"] == explicit["method"]
    assert default["method"].startswith("MBAR"), (
        f"默认口径必须仍是 MBAR，得到 {default['method']}"
    )
    # 默认路径不做任何额外计算：不该出现 crosschecks / ti_gate
    assert "crosschecks" not in default
    assert "ti_gate" not in default
    # 但provenance 三件套必须有（纯记录，不改数值）
    assert default["primary_estimator"] == "mbar_decorrelated"
    assert default["estimator_analysis_protocol_version"] == ESTIMATOR_ANALYSIS_PROTOCOL_VERSION


def test_solve_rejects_unknown_primary_estimator():
    """未知 primary_estimator 必须抛错，不得静默当成默认。"""
    u_kn, n_k, _, _ = _linear_reduced_u_kn([1.0, 0.0], n_per_state=50)
    an = TraditionalMBARAnalyzer(temperature=T_K)
    an._last_n_k = n_k
    with pytest.raises(ValueError):
        an.solve(u_kn, primary_estimator="whatever_bar")


def test_estimator_policy_fingerprint_separates_bar_from_mbar_primary():
    """指纹必须能区分主值口径，否则生产切换后旧缓存会被静默复用。"""
    base = {
        "estimator_analysis_protocol_version": ESTIMATOR_ANALYSIS_PROTOCOL_VERSION,
        "frame_selection": "decorrelated",
        "sigma_policy": "asymptotic",
        "sigma_inflation_applied": False,
        "ti_gate_tolerance_kJ_mol": 0.5,
    }
    fp_mbar = estimator_policy_fingerprint({**base, "primary_estimator": "mbar_decorrelated"})
    fp_bar = estimator_policy_fingerprint({**base, "primary_estimator": "adjacent_bar"})
    assert fp_mbar != fp_bar
    # 无关字段不得影响指纹（否则多传一个键就让全仓缓存失效）
    assert fp_bar == estimator_policy_fingerprint(
        {**base, "primary_estimator": "adjacent_bar", "irrelevant": 123}
    )


# ---------------------------------------------------------------------------
# 8. 旧 debug 旁路不得绕过 Stage 2 指纹门（P1-20）
# ---------------------------------------------------------------------------

def test_stage2_fingerprint_debug_bypass_env_var_has_no_fail_open_path():
    """`ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT` 不得再有任何 fail-open 通路。

    这条门在 Stage 2 预优化缓存恢复处（`run_full_pipeline` 内），端到端触发它需要
    真实 pipeline 实例 + 磁盘上的 preopt2 缓存，所以这里用 AST 断言**代码结构**：
      1. 每一处引用该环境变量的 `if` 分支体内必须有 `raise`；
      2. 该环境变量的分支里不得出现对 `protocol_match` 的赋值（旧旁路的原形是
         在那里硬置 `protocol_match = True`）。

    ⚠️ 刻意**不**断言"全文件不存在 `protocol_match = True`"：`abfe_pipeline.py:5846`
    那处是合法的 schema 迁移自愈，前提是
    `_preopt_cache_matches_ignoring_code_hash()` 已逐字段核对过物理输入，与本旁路
    无关。把它一起禁掉会变成一条误报的测试。
    """
    import ast

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "abfe_pipeline.py")
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    env_name = "ABFE_DEBUG_SKIP_STAGE2_FINGERPRINT"
    guards = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.If) and env_name in ast.dump(node.test)
    ]
    assert guards, f"找不到任何针对 {env_name} 的守卫；旁路可能被整段删掉而没留 fail closed"
    for node in guards:
        assert any(isinstance(n, ast.Raise) for n in ast.walk(node)), (
            f"{env_name} 的守卫分支里没有 raise —— 那就是 fail-open"
        )
        for inner in ast.walk(node):
            if isinstance(inner, ast.Assign):
                for tgt in inner.targets:
                    assert not (isinstance(tgt, ast.Name) and tgt.id == "protocol_match"), (
                        f"abfe_pipeline.py:{inner.lineno}：{env_name} 的分支里给 "
                        "protocol_match 赋值，这正是 P1-20 删掉的那条指纹旁路"
                    )


# ---------------------------------------------------------------------------
# 9. 规范表本身
# ---------------------------------------------------------------------------

def test_crosschecks_table_has_all_four_estimators_and_consistent_bar():
    """四口径规范表：每格都算得出，且 BAR 与独立调用逐位一致。"""
    lam = [1.0, 0.7, 0.4, 0.1, 0.0]
    u_kn, n_k, lam_arr, _ = _linear_reduced_u_kn(lam, n_per_state=250)
    cc = stage1_estimator_crosschecks(u_kn, n_k, KT, lambdas=lam_arr)

    bar_dg, bar_err, _ = adjacent_bar_chain(u_kn, n_k, KT)
    assert cc["adjacent_bar"]["delta_G_kJ_mol"] == pytest.approx(bar_dg, abs=1e-12)
    assert cc["adjacent_bar"]["error_kJ_mol"] == pytest.approx(bar_err, abs=1e-12)
    for key in ("decorrelated_mbar", "full_frame_mbar", "reweighted_fd_ti"):
        assert cc[key].get("delta_G_kJ_mol") is not None

    # 没给 λ 时 FD-TI 必须写明不适用，而不是拿态序号硬算
    cc_no_lam = stage1_estimator_crosschecks(u_kn, n_k, KT, lambdas=None)
    assert cc_no_lam["reweighted_fd_ti"]["delta_G_kJ_mol"] is None
    assert "lambdas" in cc_no_lam["reweighted_fd_ti"]["not_applicable_reason"]
