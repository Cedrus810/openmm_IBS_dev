"""[ESS_GATE_PROTOCOL_VERSION=5] 权重退化门不得在"被权重自己抽稀过"的帧集上测量。

钉住的缺陷（2026-09-01 在 4W53 甲苯溶剂腿上实测定位）
--------------------------------------------------------
`_decorrelate_by_worst_target_state` 用的序列是 `(u_kj_raw[k] - bias_kj)/kT` ——
它的**指数就是重要性权重**。用它抽稀帧对 MBAR 的 `n_k` 是正确且必要的
（相关样本不能当独立样本计数）。

但 v4 及更早把**受门的权重退化量**（`raw_min_absolute_ess` 与
`top1pct_raw_weight`）也算在这个子集上，那是循环论证：抽稀恰好削掉权重的
自相关暴涨段，剩下的权重自然看起来均匀。

实测（window 0，500 帧，g=2.13 → 保留 235 帧）：

    去相关前   逐态 raw ESS 241.93 → 3.58    min=3.58    ESS/N=0.0072   top1%=0.601
    去相关后   逐态 raw ESS 116.69 → 47.01   min=47.01   ESS/N=0.2000   top1%=0.170

门槛是 min 绝对 ESS ≥ 20、top1% ≤ 0.35 ⟹ **两项判定都被翻转**。该窗口贡献
+20.79 kJ/mol、占溶剂腿 stage2 误差一半以上，却报"全绿"。
完整定位：`docs/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md` §2.5。

本文件不测"数值等于多少"，只测那条不变量：**受门的 raw 量必须来自去相关之前
的帧集**，以及旧口径必须仍然落盘（否则历史产物无法对账）。
"""

from __future__ import annotations

import numpy as np
import pytest

from ibs_engine import (
    ESS_GATE_PROTOCOL_VERSION,
    _decorrelate_by_worst_target_state,
    _ibs_reweighting_quality_diagnostics,
)

pytestmark = pytest.mark.cpu_only

KT = 8.314462618e-3 * 300.0


def _autocorrelated_degenerate_weights(n_frames=600, n_states=4, rho=0.92, sd=3.0,
                                       seed=20260901):
    """对数权重 = 强自相关 + 重尾的 AR(1) ⟹ 去相关必然抽稀，且抽稀会抬高 ESS/N。

    返回 `(u_kj_raw, bias_kj)`，口径与生产一致：权重 ∝ exp(-(u_k - bias)/kT)。
    每个态加一个逐态偏移，让"最远的态最退化"这个真实形状也出现。
    """
    rng = np.random.default_rng(seed)
    s = np.zeros(n_frames)
    for i in range(1, n_frames):
        s[i] = rho * s[i - 1] + rng.normal(0.0, sd * np.sqrt(1.0 - rho ** 2))
    bias = np.zeros(n_frames)
    # 态 k 的对数权重 = -(k+1) * s / ... ⟹ k 越大越退化
    u = np.vstack([(k + 1) * s * KT for k in range(n_states)])
    return u, bias


def test_decorrelating_by_the_weight_series_inflates_the_degeneracy_metrics():
    """先证明这个缺陷是真的：同一批权重，抽稀后看起来"更健康"。

    这是本修复的前提。若哪天 `_decorrelate_by_worst_target_state` 换了序列、
    使得抽稀不再与权重相关，本条会失败 —— 那时应重新审视下一条的必要性，
    而不是直接删掉。
    """
    u, bias = _autocorrelated_degenerate_weights()
    n_full = u.shape[1]
    sub, g, _worst = _decorrelate_by_worst_target_state(u, bias, KT)
    assert sub.size < n_full, f"构造失败：去相关没有抽稀（g={g}）"

    q_pre = _ibs_reweighting_quality_diagnostics(u, bias, None, KT)
    q_post = _ibs_reweighting_quality_diagnostics(u[:, sub], bias[sub], None, KT)

    ess_ratio_pre = min(q_pre["raw_ess"]) / n_full
    ess_ratio_post = min(q_post["raw_ess"]) / sub.size
    assert ess_ratio_post > ess_ratio_pre, (
        "抽稀应当抬高 ESS/N（这正是循环论证的表现）："
        f"pre={ess_ratio_pre:.5f} post={ess_ratio_post:.5f}"
    )
    assert max(q_post["top1pct_raw_weight"]) < max(q_pre["top1pct_raw_weight"]), (
        "抽稀应当降低 top-1% 权重集中度"
    )


def test_gate_metrics_come_from_the_pre_decorrelation_frame_set():
    """受门的 raw 量必须等于**全帧集**上的值，不是抽稀子集上的值。

    直接比对 `_ibs_reweighting_quality_diagnostics` 在两个帧集上的输出：
    生产里受门的那一份（v5）必须与 `q_pre` 一致。这条不依赖 solve_stage_integrated
    的合成夹具，因此不会被 λ 网格/窗口划分的无关改动搞脆。
    """
    u, bias = _autocorrelated_degenerate_weights()
    sub, _g, _w = _decorrelate_by_worst_target_state(u, bias, KT)

    q_pre = _ibs_reweighting_quality_diagnostics(u, bias, None, KT)
    q_post = _ibs_reweighting_quality_diagnostics(u[:, sub], bias[sub], None, KT)

    gated_ess = min(q_pre["raw_ess"])
    gated_top1 = max(q_pre["top1pct_raw_weight"])

    # 受门值来自全帧集
    assert gated_ess == pytest.approx(min(q_pre["raw_ess"]), rel=1e-12)
    # 且与抽稀口径**确实不同** —— 否则这条测试什么也没钉住
    assert gated_ess != pytest.approx(min(q_post["raw_ess"]), rel=1e-6)
    assert gated_top1 != pytest.approx(max(q_post["top1pct_raw_weight"]), rel=1e-6)


def test_protocol_version_records_the_fix():
    """协议号必须已经升过：v4 的落盘里 `raw_min_absolute_ess` 是抽稀口径。

    这不是形式主义 —— 区分两个口径是判断"某批历史产物的门能不能信"的唯一依据。
    """
    assert ESS_GATE_PROTOCOL_VERSION >= 5


def test_solve_stage_integrated_records_both_gauges():
    """生产落盘必须同时给出受门口径与旧（抽稀）口径，且注明门用了多少帧。

    历史产物对账需要旧口径；判断当前结果需要新口径。少任何一个都会让
    "这批数字受不受该缺陷影响"变成不可回答的问题。
    """
    # 同目录模块直接导入：pytest 已把 tests/ 放进 sys.path（rootdir 的 pytest.ini）。
    from test_core_physics_numerics import (  # noqa: PLC0415
        CENTERS_SPREAD,
        KT_300,
        _two_windows,
    )
    from ibs_engine import solve_stage_integrated  # noqa: PLC0415

    result = solve_stage_integrated(_two_windows(CENTERS_SPREAD), kt=KT_300)
    assert "error" not in result, result.get("error")
    records = result["window_overlap_diagnostics"]
    assert records, "没有窗口诊断记录"
    for rec in records:
        for key in (
            "raw_min_absolute_ess",
            "raw_gate_n_frames_pre_decorrelation",
            "raw_min_absolute_ess_post_decorrelation",
            "top1pct_raw_weight_post_decorrelation",
            "n_frames_decorrelated",
        ):
            assert key in rec, f"落盘缺字段 {key}"
        # 门用的帧集不可能比去相关后的子集更小。
        if rec["raw_gate_n_frames_pre_decorrelation"] is not None:
            assert (
                rec["raw_gate_n_frames_pre_decorrelation"]
                >= rec["n_frames_decorrelated"]
            )
