"""[0831issue] ESS_GATE_PROTOCOL_VERSION=4 + TraditionalMBARAnalyzer overlap 容差。

两条独立的 0831 第九轮 P1：

1. **混合覆盖度门的口径**。受门的 `min_ess_ratio` / `min_occupancy_normalized` /
   `common_mode_log_sigma_kT` 都由 `p_k ∝ exp(-(S_k - f_k)/kT)` 推出，而 `f_k` 是在
   **sampling-state** 能量 `S_k = U_k^softcore + s_residual·A_k·(B_φ − offset)` 上学
   出来的（`IBS_BIAS_PROTOCOL_VERSION=31`）。旧实现把物理目标 `u_kn`（softcore+LRC，
   **不含**残差）配上 residual 口径的 `f_k`，在 residual 臂上算出的既不是采样混合分布
   也不是物理混合分布，还把逐态 rank-1 项 `A_k·B_φ(x)` 漏进了本应只含 WCA 防护壳 +
   LRC 失配的共模 σ。

2. **`TraditionalMBARAnalyzer.solve` 的 overlap 比较**。同文件其它判据统一走
   `_meets_minimum_with_roundoff`（真实案例 0.04999999999999431 vs 0.05），只有这里
   是裸 `>=`，合法值落在阈值下方几个 ulp 时会被判不收敛并触发无意义重算。
"""

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")
pytest.importorskip("pymbar")

import ibs_engine  # noqa: E402
from ibs_engine import (  # noqa: E402
    ESS_GATE_PROTOCOL_VERSION,
    _ibs_reweighting_quality_diagnostics,
    _meets_minimum_with_roundoff,
)


KT = 2.494339
N_FRAMES = 400
N_STATES = 3


def _synthetic_arm(residual_coefficients, seed=20260901):
    """造一窗数据：物理 u_kn 与 sampling-state S_k 只差逐态 rank-1 残差项。

    `A_k`（residual_coefficients）逐态不同，`B_φ(x_n)` 逐帧涨落 —— 正是不会在
    逐帧 softmax 里抵消的那一项。`f_k` 取 S_k 的态均值，即"IBS 已经把采样混合
    分布拉平"那份权重；这样 sampling 口径下占据应当接近均匀，而物理口径下不会。
    """
    rng = np.random.default_rng(seed)
    u_phys = rng.normal(scale=2.0, size=(N_STATES, N_FRAMES)) + np.array(
        [[0.0], [1.5], [3.0]]
    )
    b_phi = rng.normal(loc=0.0, scale=6.0, size=N_FRAMES)
    a_k = np.asarray(residual_coefficients, dtype=float)[:, None]
    sampling = u_phys + a_k * b_phi[None, :]
    f_k = sampling.mean(axis=1)
    # 采样偏置就是这份 (S_k, f_k) 蕴含的 IBS 偏置 —— 让共模残差恒为 0，任何
    # 非零 σ 都只能来自口径错配。
    bias = -KT * np.log(np.exp(-(sampling - f_k[:, None]) / KT).sum(axis=0))
    return u_phys, sampling, f_k, bias


def test_baseline_arm_is_bit_identical_without_sampling_gauge():
    """residual 关闭（A_k 全零）→ S_k ≡ U_k → 传与不传 sampling_kj 逐位相同。"""
    u_phys, sampling, f_k, bias = _synthetic_arm([0.0, 0.0, 0.0])
    assert np.array_equal(sampling, u_phys)

    without = _ibs_reweighting_quality_diagnostics(u_phys, bias, f_k, KT)
    with_gauge = _ibs_reweighting_quality_diagnostics(
        u_phys, bias, f_k, KT, sampling_kj=sampling
    )

    assert without["mixture_gauge"] == "physical_targets"
    assert with_gauge["mixture_gauge"] == "sampling_states"
    for key in (
        "mixture_ess",
        "mixture_ess_ratio",
        "mixture_occupancy_normalized",
        "common_mode_log_sigma_kT",
        "raw_ess",
        "top1pct_raw_weight",
    ):
        assert np.allclose(
            np.asarray(without[key], dtype=float),
            np.asarray(with_gauge[key], dtype=float),
            rtol=0.0,
            atol=0.0,
        ), f"{key} 在 A_k=0 时必须逐位相同"


def test_residual_arm_physical_gauge_fabricates_common_mode_sigma():
    """residual 开启时，旧（物理）口径会凭空造出共模 σ；新口径把它归零。

    构造上采样偏置 = (S_k, f_k) 蕴含的 IBS 偏置，所以 sampling 口径下
    `residual_kj = bias + kT·log_norm` 恒为 0 → σ_r = 0。物理口径下差的正是
    `A_k·B_φ` 那一项，它逐态不同、逐帧涨落，于是 σ_r 被人为撑大。
    """
    u_phys, sampling, f_k, bias = _synthetic_arm([0.0, 0.6, 1.2])

    phys = _ibs_reweighting_quality_diagnostics(u_phys, bias, f_k, KT)
    samp = _ibs_reweighting_quality_diagnostics(
        u_phys, bias, f_k, KT, sampling_kj=sampling
    )

    assert samp["common_mode_log_sigma_kT"] == pytest.approx(0.0, abs=1e-10)
    # 本仓库 ESS_GATE 注释记录的**真实**防护壳共模是 0.95~2.40 kT；这里被凭空
    # 造出来的量必须显著非零，否则这条回归证明不了什么。
    assert phys["common_mode_log_sigma_kT"] > 0.5

    # 受门的两项也必须实质性改变——否则这次修复无从体现。方向刻意**不**断言：
    # 旧口径既可能偏乐观也可能偏悲观（本组数据里它偏乐观，min 占据 0.92 vs 真值
    # 0.72，即旧门会放过一个其实更饿的态），这正是"算的既不是采样混合分布也不是
    # 物理混合分布"的表现，不是一个有固定符号的偏差。
    samp_occ = np.asarray(samp["mixture_occupancy_normalized"], dtype=float)
    phys_occ = np.asarray(phys["mixture_occupancy_normalized"], dtype=float)
    assert abs(samp_occ.min() - phys_occ.min()) > 0.05

    samp_ess = np.asarray(samp["mixture_ess"], dtype=float)
    phys_ess = np.asarray(phys["mixture_ess"], dtype=float)
    assert not np.allclose(samp_ess, phys_ess, rtol=0.05)


def test_raw_block_always_stays_in_the_physical_gauge():
    """`raw_*` / `top1pct_raw_weight` 量的是"重加权到真实物理系综"，必须不受影响。"""
    u_phys, sampling, f_k, bias = _synthetic_arm([0.0, 0.6, 1.2])
    phys = _ibs_reweighting_quality_diagnostics(u_phys, bias, f_k, KT)
    samp = _ibs_reweighting_quality_diagnostics(
        u_phys, bias, f_k, KT, sampling_kj=sampling
    )
    for key in ("raw_ess", "raw_ess_ratio", "top1pct_raw_weight"):
        assert np.allclose(
            np.asarray(phys[key], dtype=float),
            np.asarray(samp[key], dtype=float),
            rtol=0.0,
            atol=0.0,
        ), f"{key} 不应随混合口径改变"


def test_missing_sampling_gauge_when_required_fails_closed():
    """声明了 residual 却拿不到 S_k → 混合项全 None + 显式 error，不得静默降级。"""
    u_phys, _, f_k, bias = _synthetic_arm([0.0, 0.6, 1.2])
    out = _ibs_reweighting_quality_diagnostics(
        u_phys, bias, f_k, KT, sampling_kj=None, sampling_gauge_required=True
    )
    assert out["error"] == "missing_sampling_state_energies"
    assert out["mixture_ess"] is None
    assert out["mixture_occupancy_normalized"] is None
    assert out["mixture_gauge"] is None
    # raw 块在 f_k 校验之前就算好了，仍应保留（它不依赖口径）。
    assert out["raw_ess"] is not None


def test_malformed_sampling_gauge_fails_closed():
    u_phys, sampling, f_k, bias = _synthetic_arm([0.0, 0.6, 1.2])

    bad_shape = _ibs_reweighting_quality_diagnostics(
        u_phys, bias, f_k, KT, sampling_kj=sampling[:, :-1]
    )
    assert bad_shape["error"] == (
        "sampling_state_energy_shape_or_finiteness_mismatch"
    )
    assert bad_shape["mixture_ess"] is None

    nonfinite = sampling.copy()
    nonfinite[1, 7] = np.nan
    bad_finite = _ibs_reweighting_quality_diagnostics(
        u_phys, bias, f_k, KT, sampling_kj=nonfinite
    )
    assert bad_finite["error"] == (
        "sampling_state_energy_shape_or_finiteness_mismatch"
    )
    assert bad_finite["mixture_ess"] is None


def test_local_mbar_forwards_and_aligns_the_sampling_gauge():
    """`_solve_single_window_local_mbar` 必须把 S_k 一起过去相关子采样与 valid_mask。"""
    u_phys, sampling, f_k, bias = _synthetic_arm([0.0, 0.6, 1.2])
    base = np.zeros(N_FRAMES)

    res = ibs_engine._solve_single_window_local_mbar(
        u_phys, bias, base, [0, 1, 2], KT,
        f_k=f_k, sampled_distribution_row=0, w_idx=0,
        sampling_kj=sampling,
    )
    assert "error" not in res, res.get("error")
    assert res["ess_gate_mixture_gauge"] == "sampling_states"
    assert res["ess_gate_protocol_version"] == ESS_GATE_PROTOCOL_VERSION
    assert res["min_ess_ratio"] is not None

    # 形状不一致必须在进 MBAR 之前 fail closed，而不是悄悄错位。
    bad = ibs_engine._solve_single_window_local_mbar(
        u_phys, bias, base, [0, 1, 2], KT,
        f_k=f_k, sampled_distribution_row=0, w_idx=0,
        sampling_kj=sampling[:, :-1],
    )
    assert "error" in bad and "sampling_state_energies" in bad["error"]

    # required 但缺 S_k → 门算不出来（converged 因此 fail closed）。
    missing = ibs_engine._solve_single_window_local_mbar(
        u_phys, bias, base, [0, 1, 2], KT,
        f_k=f_k, sampled_distribution_row=0, w_idx=0,
        sampling_gauge_required=True,
    )
    assert missing.get("min_ess_ratio") is None
    assert missing.get("reweighting_quality_error") == (
        "missing_sampling_state_energies"
    )


def test_ess_gate_protocol_version_is_five():
    # v4 → v5（2026-09-01）：受门的 raw 权重退化量改在**去相关之前**的帧集上算
    # （原来算在 `_decorrelate_by_worst_target_state` 的输出上，而那个抽稀用的
    # 序列就是权重的指数 —— 循环论证）。见 ibs_engine.py 该常量处的注释。
    assert ESS_GATE_PROTOCOL_VERSION == 5


# ---------------------------------------------------------------------------
# TraditionalMBARAnalyzer.solve 的 overlap roundoff 容差
# ---------------------------------------------------------------------------


def test_traditional_solve_uses_roundoff_tolerant_overlap_comparison():
    """min_overlap 落在阈值下方 1 ulp 量级时必须仍判 converged。

    直接把 `_build_mbar_compatible` / `compute_overlap` 打桩，只测 `solve()` 里
    `converged` 那一步的比较语义 —— 这正是回归点所在，不需要真实 MBAR 求解。
    """
    threshold = 0.03
    just_below = np.nextafter(threshold, 0.0)
    assert just_below < threshold                      # 真的在阈值下方
    assert _meets_minimum_with_roundoff(just_below, threshold)

    class _FakeMBAR:
        def compute_overlap(self):
            # 相邻元素两两都取"恰好差 1 ulp"的合法值。
            m = np.full((2, 2), 0.5)
            m[0, 1] = just_below
            m[1, 0] = just_below
            return {"matrix": m}

        def compute_effective_sample_number(self):
            return np.array([120.0, 120.0])

    def _fake_build(*_args, **_kwargs):
        return _FakeMBAR()

    def _fake_result(*_args, **_kwargs):
        return object()

    def _fake_extract(_res, require_uncertainty=False):
        df = np.array([[0.0, 1.0], [-1.0, 0.0]])
        ddf = np.array([[0.0, 0.1], [0.1, 0.0]])
        return df, ddf

    analyzer = ibs_engine.TraditionalMBARAnalyzer(temperature=300.0)
    analyzer._last_n_k = np.array([50, 50], dtype=int)
    u_kn = np.zeros((2, 100), dtype=float)

    monkeys = {
        "_build_mbar_compatible": _fake_build,
        "_compute_free_energy_result_compatible": _fake_result,
        "_extract_free_energy_arrays": _fake_extract,
    }
    saved = {k: getattr(ibs_engine, k) for k in monkeys}
    try:
        for k, v in monkeys.items():
            setattr(ibs_engine, k, v)
        result = analyzer.solve(u_kn, decorrelate=False)
    finally:
        for k, v in saved.items():
            setattr(ibs_engine, k, v)

    assert result["min_overlap"] == pytest.approx(just_below, rel=0, abs=0)
    assert result["converged"] is True, (
        "overlap 落在阈值下方 1 ulp 时被判不收敛 —— 裸 `>=` 回归了"
    )


def test_traditional_solve_still_rejects_materially_low_overlap():
    """容差只接受舍入尺度的相等，不得放松物理阈值。"""
    assert not _meets_minimum_with_roundoff(0.0299, 0.03)
    assert not _meets_minimum_with_roundoff(None, 0.03)
    assert not _meets_minimum_with_roundoff(float("nan"), 0.03)
