"""核心物理/数值单元测试（ATT-19 / #59）。

与 `test_audit_protocol_regressions.py` / `test_warmup_overlap_protocol.py` 的分工：
那两个文件是**协议契约**测试——读源码文本断言某个门存在、某个 protocol version
进了指纹、某份旧缓存会 fail-closed。它们能防"改代码忘了递增版本号"，但证明不了
**数学算对了**。本文件补的正是后者：把真正决定 ΔG 数值的四处计算跟独立手算的
期望值逐个比对。

四项被测对象：
  1. `abfe_core.calculate_boresch_analytical_correction` —— 标准态释放修正，确定性，<1e-6
  2. `ibs_engine.solve_stage_integrated` —— 合成已知 ΔG 的 2 窗口数据，验证拼接与误差棒
  3. `ibs_engine.IBSBiasForce` —— 在真实 OpenMM Context 里求值，对比手算 log-sum-exp
  4. `abfe_preoptimizer.estimate_f_k_from_pilot_ti` —— f_k = F_k - mean(F)，不反号

全部 CPU 可跑（IBSBiasForce 是 CustomCVForce，用 Reference platform 求值），
不需要 GPU、不需要真实体系拓扑。
"""

import math

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
pytest.importorskip("pymbar")
from openmm import unit
from scipy import constants
from scipy.special import logsumexp

from abfe_core import calculate_boresch_analytical_correction
from abfe_preoptimizer import estimate_f_k_from_pilot_ti
import ibs_engine as ie
from ibs_engine import IBSBiasForce, solve_stage_integrated


# ============================================================================
# 共享常量
# ============================================================================

R_KJ = constants.R / 1000.0          # kJ/mol/K —— 与 abfe_core 用的同一个来源
T_REF = 300.0
RT_REF = R_KJ * T_REF
V0_NM3 = 1.6605                      # 标准摩尔体积，nm³


# ============================================================================
# 1. Boresch 解析修正 —— 确定性，误差 < 1e-6
# ============================================================================

def _boresch_reference_via_log_sum(eq, fc, T=T_REF):
    """独立参考实现：把 -RT·ln[A·B] 展开成**对数之和**再求值。

    刻意不复刻被测函数"先算乘积/商再取一次 log"的写法——展开成
        -RT·[ ln(8π²V₀) - 2·ln r₀ - ln sinθA - ln sinθB
              + 0.5·Σ ln K_i - 3·ln(2πRT) ]
    是一条数值路径完全不同的计算（不构造 √Kdet≈3e6 与 (2πRT)³≈3.8e3 这两个大数，
    也不做它们的商），因此能抓住指数写错（1.5 次方 vs 3 次方）、漏项、符号反了
    这类代数错误，而不只是"把同一份代码抄了两遍"。
    """
    RT = (constants.R / 1000.0) * T
    log_arg = (
        math.log(8.0 * math.pi**2 * V0_NM3)
        - 2.0 * math.log(eq["r0"])
        - math.log(math.sin(eq["thetaA0"]))
        - math.log(math.sin(eq["thetaB0"]))
        + 0.5 * sum(
            math.log(fc[key])
            for key in ("kr", "kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC")
        )
        - 3.0 * math.log(2.0 * math.pi * RT)
    )
    return -RT * log_arg


def _valid_boresch_params(**overrides):
    """一组全部落在被测函数合法区间内的 Boresch 参数（强制标准单位：nm / rad /
    kJ/mol/nm² / kJ/mol/rad²）。kr 必须在 [50, 5000]，角度力常数在 [10, 1000]。"""
    eq = {
        "r0": 0.5,          # nm
        "thetaA0": 1.2,     # rad
        "thetaB0": 1.2,     # rad
        "phiA0": 0.3,
        "phiB0": -0.4,
        "phiC0": 0.5,
    }
    fc = {
        "kr": 1000.0,       # kJ/mol/nm²
        "kthetaA": 100.0,   # kJ/mol/rad²
        "kthetaB": 100.0,
        "kphiA": 100.0,
        "kphiB": 100.0,
        "kphiC": 100.0,
    }
    for key, value in overrides.items():
        if key in eq:
            eq[key] = value
        elif key in fc:
            fc[key] = value
        else:
            raise KeyError(f"未知的 Boresch 参数名: {key}")
    return eq, fc


def test_boresch_correction_matches_hand_computed_value():
    eq, fc = _valid_boresch_params()
    got = calculate_boresch_analytical_correction(eq, fc, T=T_REF)
    expected = _boresch_reference_via_log_sum(eq, fc, T=T_REF)

    assert abs(got - expected) < 1e-6, (
        f"Boresch 解析修正与独立手算不一致: got={got!r}, expected={expected!r}"
    )
    # 量级哨兵：标准态释放修正在这组参数下约 -32.7 kJ/mol。这条不是精度断言，
    # 是防"两边同时按同一个错误公式算、差值为 0 但物理量级荒谬"的兜底。
    assert -50.0 < got < -20.0, f"Boresch 修正量级不合理: {got}"


def test_boresch_correction_scales_with_sqrt_kdet():
    """kr ×16 → √Kdet ×4 → 返回值精确减少 RT·ln 4。

    单独这一条还不能钉死 (2πRT) 的指数（√Kdet 与分母无关），需要配合下面
    test_boresch_correction_responds_to_RT_as_cubed 一起看。
    """
    eq, fc_low = _valid_boresch_params(kr=250.0)
    _, fc_high = _valid_boresch_params(kr=4000.0)   # ×16，仍在 [50, 5000] 内

    low = calculate_boresch_analytical_correction(eq, fc_low, T=T_REF)
    high = calculate_boresch_analytical_correction(eq, fc_high, T=T_REF)

    assert abs((low - high) - RT_REF * math.log(4.0)) < 1e-9, (
        f"√Kdet ×4 应让修正精确减少 RT·ln4={RT_REF * math.log(4.0)!r}，"
        f"实测 {low - high!r}"
    )


def test_boresch_correction_responds_to_RT_as_cubed():
    """温度变化时 (2πRT)^n 的指数 n 可被反解出来，断言 n == 3。

    6 个谐振 Boresch 自由度的高斯积分给出 (2πRT)³，写成 1.5 次方是 docstring
    明确点出过的错误类型。固定几何与力常数、只改 T：
        -ΔG/RT = C - n·ln(2πRT)
    两个温度联立消去 C 即可解出 n，无需信任被测函数的任何中间量。
    """
    eq, fc = _valid_boresch_params()
    t1, t2 = 250.0, 350.0
    g1 = calculate_boresch_analytical_correction(eq, fc, T=t1)
    g2 = calculate_boresch_analytical_correction(eq, fc, T=t2)

    rt1, rt2 = R_KJ * t1, R_KJ * t2
    lhs = (-g1 / rt1) - (-g2 / rt2)
    rhs_unit = -(math.log(2.0 * math.pi * rt1) - math.log(2.0 * math.pi * rt2))
    n_recovered = lhs / rhs_unit

    assert abs(n_recovered - 3.0) < 1e-9, (
        f"(2πRT) 的指数应为 3（6 个谐振自由度），反解得到 {n_recovered!r}"
    )


def test_boresch_correction_is_sensitive_to_r0_unit_mistake():
    """r0 必须是 nm。误传 Å 数值（×10）会让结果整体偏移 RT·ln(100)≈11.5 kJ/mol。

    这个函数的 docstring 用 ⚠️ 特别警告过 r0 单位，说明是实际踩过的坑。这条
    保证单位量级错误会**在数值上显形**，而不是静默通过。
    """
    eq_nm, fc = _valid_boresch_params(r0=0.5)
    eq_angstrom_mistake, _ = _valid_boresch_params(r0=5.0)

    g_nm = calculate_boresch_analytical_correction(eq_nm, fc, T=T_REF)
    g_wrong = calculate_boresch_analytical_correction(eq_angstrom_mistake, fc, T=T_REF)

    # arg ∝ 1/r0²；r0 ×10 → arg /100 → -RT·ln(arg) 增加 RT·ln(100)
    assert abs((g_wrong - g_nm) - RT_REF * math.log(100.0)) < 1e-9
    assert abs(g_wrong - g_nm) > 10.0, "单位错误必须造成 >10 kJ/mol 的可见偏差"


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"kr": 10.0}, "kr 低于合理下限 50"),
        ({"kr": 1.0e4}, "kr 高于合理上限 5000"),
        ({"kthetaA": 5.0}, "kthetaA 低于 [10, 1000]"),
        ({"kthetaB": 5.0e3}, "kthetaB 高于 [10, 1000]"),
        ({"thetaA0": 1.0e-6}, "sinθA≈0，锚点几何奇点"),
        ({"thetaB0": math.pi - 1.0e-6}, "sinθB≈0，锚点几何奇点"),
        ({"kphiA": 0.0}, "力常数为零，Kdet 非正"),
        ({"kphiC": -100.0}, "力常数为负，Kdet 非正"),
    ],
)
def test_boresch_correction_fails_closed_on_bad_input(overrides, reason):
    """非法输入必须 raise ValueError，不许返回 NaN/inf 或静默给出一个数。"""
    eq, fc = _valid_boresch_params(**overrides)
    with pytest.raises(ValueError):
        calculate_boresch_analytical_correction(eq, fc, T=T_REF)


def test_boresch_correction_accepts_openmm_quantity_temperature():
    """T 允许传 OpenMM Quantity（代码里有 value_in_unit 分支），结果须与裸 float 一致。"""
    eq, fc = _valid_boresch_params()
    as_float = calculate_boresch_analytical_correction(eq, fc, T=T_REF)
    as_quantity = calculate_boresch_analytical_correction(
        eq, fc, T=T_REF * unit.kelvin
    )
    assert abs(as_float - as_quantity) < 1e-12


# ============================================================================
# 2. solve_stage_integrated —— 合成已知 ΔG 的多窗口数据
# ============================================================================
#
# 输入契约（读 ibs_engine.py GlobalMBARAnalyzer.solve_stage_integrated 确认）：
#   u_kn           : (K_local, N)，存的是 U_k_int；代码内部重建 U_k = base + u_kn
#   base_energies  : (N,)  ┐ base + bias 是**真实被采样**的势能，
#   bias_energies  : (N,)  ┘ 用于自相关子采样和增广矩阵第 0 行
#   lambda_indices : 全局 state 索引（拼接靠它找 overlap）
#   window_index / sampled_distribution_row(默认 0)
#
# 合成模型：每个态是**同一弹簧常数**的谐振井
#     U_k(x) = 0.5·k_s·(x - c_k)² + Δ_k
# 同一 k_s 时高斯配分函数的宽度项完全抵消，于是
#     F_k - F_j = Δ_k - Δ_j          （精确成立，与 c_k 无关）
# 所以 total_delta_G 的解析期望值就是 Δ_last - Δ_first。

K_SPRING = 1000.0                              # kJ/mol/nm²
KT_300 = RT_REF                                # kJ/mol
SIGMA_X = math.sqrt(KT_300 / K_SPRING)         # 采样分布标准差 ≈ 0.0499 nm
DELTAS = np.array([0.0, 3.0, -2.0, 5.0, 1.5])  # 各态的能量偏移，kJ/mol
N_FRAMES = 1500                                # converged 需绝对 ESS≥50、去相关样本≥20
CONST_BIAS = 7.0                               # 常数偏置：不改变采样分布，但不得泄漏进 ΔG


def _harmonic_window(
    lambda_indices,
    centers,
    window_index,
    seed,
    n_frames=N_FRAMES,
    bias_offset=CONST_BIAS,
    with_f_k=True,
):
    """造一个窗口的数据：从"该窗口中间态的井"里抽 iid 样本。

    iid 抽样是刻意的——`subsample_series_by_autocorrelation` 会给出 g≈1，几乎
    不丢帧，从而稳定通过 min_decorrelated_samples 门；同时 MBAR 的独立样本假设
    严格成立，估计量渐近无偏，容差才有意义。
    """
    rng = np.random.default_rng(seed)
    sampled_center = float(centers[len(centers) // 2])
    x = rng.normal(loc=sampled_center, scale=SIGMA_X, size=n_frames)

    base = 0.5 * K_SPRING * (x - sampled_center) ** 2
    bias = np.full(n_frames, float(bias_offset))

    u_kn = np.empty((len(lambda_indices), n_frames), dtype=float)
    for row, (lam_idx, center) in enumerate(zip(lambda_indices, centers)):
        u_k = 0.5 * K_SPRING * (x - center) ** 2 + float(DELTAS[lam_idx])
        u_kn[row] = u_k - base          # 存 U_k_int = U_k - base

    window = {
        "u_kn": u_kn,
        "bias_energies": bias,
        "base_energies": base,
        "lambda_indices": list(lambda_indices),
        "window_index": int(window_index),
        "sampled_distribution_row": 0,
        "window_label": f"win{window_index}",
        "window_range": [int(lambda_indices[0]), int(lambda_indices[-1]) + 1],
    }
    if with_f_k:
        # [ESS_GATE_PROTOCOL_VERSION=2] 混合覆盖度 ESS 门需要该窗口冻结的 f_k。
        # 这些井全部共用同一个 K_SPRING，谐振子配分函数与井心无关，所以解析上
        # F_k = DELTAS[k] + const；而 IBS 的平坦权重条件正是 f_k = F_k + const，
        # 因此 f_k = DELTAS[lam_idx] 就是这组合成数据"已收敛"的那份 f_k，不是
        # 为了让门通过而挑的数。
        window["f_k"] = np.asarray(
            [float(DELTAS[i]) for i in lambda_indices], dtype=float
        )
    return window


def _two_windows(centers_by_state, seed_a=20260726, seed_b=20260727, **kwargs):
    """窗口 A=[0,1,2]、B=[2,3,4]，共享全局态 2（拼接与协方差链都需要 overlap）。"""
    win_a = _harmonic_window(
        [0, 1, 2], [centers_by_state[i] for i in (0, 1, 2)], 0, seed_a, **kwargs
    )
    win_b = _harmonic_window(
        [2, 3, 4], [centers_by_state[i] for i in (2, 3, 4)], 1, seed_b, **kwargs
    )
    return [win_a, win_b]


# 全部态共用同一井心 -> U_k - U_j 恒为常数 -> 重加权权重全相等 -> MBAR 精确可解。
# 这一组是**确定性**断言的基础：把 β 换算、拼接、协方差链的算术钉死到 1e-6，
# 不受采样噪声影响。
CENTERS_DEGENERATE = [0.0] * 5
# 井心逐态平移 0.4σ -> 重加权权重非平凡，检验真实的 MBAR 重加权路径。
CENTERS_SPREAD = [0.0, 0.4 * SIGMA_X, 0.8 * SIGMA_X, 1.2 * SIGMA_X, 1.6 * SIGMA_X]


def test_solve_stage_integrated_recovers_known_delta_g_exactly_when_weights_are_uniform():
    """同井心 → 各态只差常数 → MBAR 权重全相等 → ΔG 应精确等于 Δ_4 - Δ_0。

    这条同时是 `u_mbar = u_mbar * self.beta` 那处修复的回归测试：漏掉 β 换算时
    MBAR 会在一个错误的有效温度下求解，末尾再乘 kt，结果被系统性放大 kt≈2.494 倍。
    ΔG 期望值 = 5.0 - 0.0 = 5.0 kJ/mol，放大后会是 12.47，1e-6 的容差绝对抓得到。
    """
    windows = _two_windows(CENTERS_DEGENERATE)
    result = solve_stage_integrated(windows, kt=KT_300, stage_name="synthetic")

    expected = float(DELTAS[4] - DELTAS[0])
    assert "error" not in result, f"求解失败: {result.get('error')}"
    assert abs(result["total_delta_G"] - expected) < 1e-6, (
        f"total_delta_G={result['total_delta_G']!r} 应为 {expected!r}"
        "（漏乘 beta 会放大约 kt=2.494 倍）"
    )
    assert result["stage"] == "synthetic"


def test_solve_stage_integrated_recovers_known_delta_g_with_nontrivial_weights():
    """井心错开 → 重加权权重非平凡 → 走真实 MBAR 路径，仍应还原 Δ_4 - Δ_0。

    容差刻意**不是** 1e-6：MBAR 是统计估计量，固定 seed 只保证结果可复现，不保证
    机器精度。用 max(0.5 kJ/mol, 3×报告误差) —— 既检验准确性，也顺带检验报告的
    误差棒本身自洽（真值落在 3σ 内）。
    """
    windows = _two_windows(CENTERS_SPREAD)
    result = solve_stage_integrated(windows, kt=KT_300)

    expected = float(DELTAS[4] - DELTAS[0])
    assert "error" not in result, f"求解失败: {result.get('error')}"
    total_err = float(result["total_error"])
    assert np.isfinite(total_err) and total_err < 1.0, f"误差棒异常: {total_err}"

    tolerance = max(0.5, 3.0 * total_err)
    assert abs(result["total_delta_G"] - expected) < tolerance, (
        f"total_delta_G={result['total_delta_G']!r} 偏离解析值 {expected!r} "
        f"超过 {tolerance!r}（报告误差 {total_err!r}）"
    )
    assert result["converged"] is True, (
        f"1500 帧 iid、重叠良好的合成数据应判 converged；诊断: "
        f"min_overlap={result['min_overlap']}, "
        f"min_absolute_ess={result['min_absolute_ess']}, "
        f"min_decorrelated_samples={result['min_decorrelated_samples']}, "
        f"max_endpoint_uncertainty={result['max_endpoint_uncertainty_kJ_mol']}"
    )


def test_ess_gate_is_mixture_coverage_and_raw_ess_is_diagnostic_only():
    """[ESS_GATE_PROTOCOL_VERSION=2] 受门量 = 扣掉共模因子的混合覆盖度 ESS；
    raw 单参考 ESS 只报告不设门；absolute_ess 的阈值已退役。

    这条钉住的是真实 GPU 上观察到的失效模式：生产采样势含 Group-4 λ-WCA 防护壳
    (+LRC)，它对窗口内所有 k 是同一个逐帧共模因子，会把 raw 单参考 ESS 按
    exp(σ_r²) 整体压掉（实测 σ_r 0.95→2.40 kT → ESS/N 上限 0.40→0.003），而同一批
    数据的占据是平坦的 0.249-0.251、端点不确定度可以低到 0.07 kJ/mol。用 raw 量当
    收敛门等于要求一个数学上不可满足、且与被估计量无关的目标。
    """
    result = solve_stage_integrated(_two_windows(CENTERS_SPREAD), kt=KT_300)

    assert result["min_overlap_method"] == (
        "per_window_mixture_coverage_ess_ratio_common_mode_removed"
    )
    assert result["ess_gate_protocol_version"] == 2
    # absolute_ess 仍然算、仍然落盘，但阈值必须是 None——它在构造上等于
    # min_ess_ratio × n_frames_decorrelated，不是第二份独立证据。阈值置 None 同时
    # 让 abfe_pipeline 侧两处"字段存在才检查"的镜像判据自动失活。
    assert result["min_absolute_ess"] is not None
    assert result["min_absolute_ess_threshold"] is None
    assert "min_absolute_ess_gate_retired_reason" in result
    # raw 量必须仍然被报告出来（诊断防护壳收了多少重加权税），且与受门量分开存放。
    for record in result["window_overlap_diagnostics"]:
        assert record["raw_min_ess_ratio"] is not None
        assert record["raw_min_absolute_ess"] is not None
        assert record["top1pct_raw_weight"] is not None
        assert record["common_mode_log_sigma_kT"] is not None
        assert record["ess_gate_metric"] == "mixture_coverage_ess_common_mode_removed"


def test_mixture_ess_alone_cannot_see_a_uniformly_starved_state():
    """ESS 逐态尺度不变 → 单靠它抓不到"均匀被饿死"的态，必须配一阶矩 K*<p_k>。

    构造：三个态，state 2 的 U' 高出 ~80 kT 且 f_k **没有**补偿它。它的 p_2 逐帧都
    是 ~e^-80，但逐帧**相对**起伏很小，所以 ESS(p_2)/N 照样报健康值。这条钉住
    "ESS 报健康 + 占据报饿死"这个组合，防止有人以为 ESS 一项就够、把占据门删掉。
    """
    rng = np.random.default_rng(1)
    n = 400
    u_kn = np.vstack([
        rng.normal(0.0, 2.0, n),
        rng.normal(3.0, 2.0, n),
        rng.normal(200.0, 2.0, n),
    ])
    starved = ie._ibs_reweighting_quality_diagnostics(
        u_kn, np.zeros(n), np.array([0.0, 3.0, 0.0]), KT_300
    )
    # ESS 完全看不出问题——这正是盲点。
    assert min(starved["mixture_ess_ratio"]) > 0.3
    # 占据必须看出来，并低于 warmup 那边同一套口径的下限。
    assert min(starved["mixture_occupancy_normalized"]) < 1e-10
    assert (
        min(starved["mixture_occupancy_normalized"])
        < ie.IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION
    )

    # f_k 补偿后两项都健康（说明占据门不是无条件收紧，只针对真饿死）。
    healthy = ie._ibs_reweighting_quality_diagnostics(
        u_kn, np.zeros(n), np.array([0.0, 3.0, 200.0]), KT_300
    )
    assert min(healthy["mixture_ess_ratio"]) > 0.3
    assert min(healthy["mixture_occupancy_normalized"]) > (
        ie.IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION
    )


def test_ess_gate_fails_closed_when_frozen_f_k_is_unavailable():
    """没有 f_k 就没法把共模因子除干净 → min_overlap=None、converged=False。

    绝不能静默退回 raw 量当受门指标（那正是被退役的语义），也绝不能当作通过。
    物理量（ΔG/误差棒）不受影响，只是这个门无法评估。
    """
    windows = _two_windows(CENTERS_SPREAD, with_f_k=False)
    assert all("f_k" not in w for w in windows)
    result = solve_stage_integrated(windows, kt=KT_300)

    assert "error" not in result
    assert result["min_overlap"] is None
    assert result["converged"] is False
    # raw 诊断仍可用，且明确记录了为什么算不出 mixture 量。
    assert result["raw_min_overlap"] is not None
    for record in result["window_overlap_diagnostics"]:
        assert record["min_ess_ratio"] is None
        assert record["reweighting_quality_error"] == "missing_f_k"
    # 物理结果本身不受门的影响。
    expected = float(DELTAS[4] - DELTAS[0])
    assert abs(result["total_delta_G"] - expected) < max(
        0.5, 3.0 * float(result["total_error"])
    )


def test_decorrelation_uses_reweighting_series_not_total_potential():
    """[ESS_GATE_PROTOCOL_VERSION=2] g 必须来自"权重本身的指数" Δu_k=(U'_k−V_bias)/kT，
    而不是总势能 base+bias。

    总势能被溶剂涨落主导、与权重的慢相关无关，实测在真实 Atenolol vdw 数据上把 g
    低估 3-10 倍（窗口 0/2/5：1.76/2.58/2.44 vs 19.6/26.5/7.7），而低估 g 会让喂进
    MBAR 的 n_k 虚高、误差棒系统性偏小——正是子采样本身要防的那件事。
    """
    idx, g, worst = ie._decorrelate_by_worst_target_state(
        np.zeros((3, 200)), np.zeros(200), KT_300
    )
    # 常数序列没有涨落 -> g=1、不子采样（沿用 subsample_series_by_autocorrelation 的约定）
    assert g == 1.0 and idx.size == 200 and worst == -1

    # 给某一个目标态注入强自相关（随机游走），它必须成为决定子采样的那个态。
    rng = np.random.default_rng(20260726)
    n = 600
    u = np.zeros((3, n))
    u[2] = np.cumsum(rng.normal(size=n))      # 慢：随机游走
    u[1] = rng.normal(size=n)                 # 快：iid
    idx, g, worst = ie._decorrelate_by_worst_target_state(u, np.zeros(n), KT_300)
    assert worst == 2, f"应由自相关最强的 state 2 决定子采样，实际 {worst}"
    assert g > 1.0 and idx.size < n


def test_solve_stage_integrated_error_is_covariance_chain_not_quadrature():
    """总误差必须是各窗口"连接态→端点"直接 dDelta_f 的独立方差相加。

    源码注释明令禁止把两个相对采样态的边际不确定度做 sqrt(a²+b²)（同一次 MBAR
    拟合、同一批样本，彼此有协方差，平方相加会系统性算错）。这里逐项核对
    covariance_chain_segments 与 total_error 的关系，把这个约定钉死。
    """
    windows = _two_windows(CENTERS_SPREAD)
    result = solve_stage_integrated(windows, kt=KT_300)

    assert result["total_error_method"] == (
        "independent_window_segment_variances_using_direct_dDelta_f"
    )
    segments = result["covariance_chain_segments"]
    assert len(segments) == 2, f"两个窗口应给出两段协方差链: {segments}"

    # 段的 join/end 必须沿路径推进且不重复积分：窗口 0 从全局态 0 到 2，
    # 窗口 1 从共享态 2 到 4。
    assert (segments[0]["join_lambda_index"], segments[0]["end_lambda_index"]) == (0, 2)
    assert (segments[1]["join_lambda_index"], segments[1]["end_lambda_index"]) == (2, 4)

    recomputed_dg = sum(seg["delta_G_kJ_mol"] for seg in segments)
    recomputed_err = math.sqrt(
        sum(seg["uncertainty_kJ_mol"] ** 2 for seg in segments)
    )
    assert abs(result["total_delta_G"] - recomputed_dg) < 1e-12
    assert abs(result["total_error"] - recomputed_err) < 1e-12
    assert abs(result["endpoint_error_after_offset"] - recomputed_err) < 1e-12


def test_solve_stage_integrated_is_invariant_to_constant_bias_shift():
    """给 bias_energies 整体加常数不改变采样分布，也绝不能泄漏进 ΔG。"""
    baseline = solve_stage_integrated(_two_windows(CENTERS_SPREAD), kt=KT_300)
    shifted = solve_stage_integrated(
        _two_windows(CENTERS_SPREAD, bias_offset=CONST_BIAS + 250.0), kt=KT_300
    )
    assert abs(baseline["total_delta_G"] - shifted["total_delta_G"]) < 1e-6, (
        "bias 常数偏移不是物理量，不得改变 total_delta_G"
    )


def test_solve_stage_integrated_is_invariant_to_window_input_order():
    """窗口以乱序传入时代码会按 λ 索引重排；ΔG 必须与顺序无关。"""
    windows = _two_windows(CENTERS_DEGENERATE)
    forward = solve_stage_integrated(windows, kt=KT_300)
    reversed_in = solve_stage_integrated(list(reversed(windows)), kt=KT_300)
    assert abs(forward["total_delta_G"] - reversed_in["total_delta_G"]) < 1e-9


@pytest.mark.parametrize("missing_key", ["bias_energies", "base_energies"])
def test_solve_stage_integrated_refuses_to_substitute_zero_for_missing_energies(
    missing_key,
):
    """缺 bias/base 必须 raise —— 源码明写"IBS-TMBAR 禁止以零替代"。"""
    windows = _two_windows(CENTERS_DEGENERATE)
    windows[0][missing_key] = None
    with pytest.raises(ValueError):
        solve_stage_integrated(windows, kt=KT_300)


def test_solve_stage_integrated_rejects_frame_count_mismatch():
    windows = _two_windows(CENTERS_DEGENERATE)
    windows[0]["bias_energies"] = windows[0]["bias_energies"][:-5]
    with pytest.raises(ValueError):
        solve_stage_integrated(windows, kt=KT_300)


@pytest.mark.parametrize("target", ["u_kn", "bias_energies", "base_energies"])
def test_solve_stage_integrated_rejects_nonfinite_energies(target):
    """NaN/Inf 不许被静默剔除后当有效数据用——必须 raise。"""
    windows = _two_windows(CENTERS_DEGENERATE)
    arr = np.array(windows[0][target], dtype=float)
    if arr.ndim == 2:
        arr[0, 3] = np.nan
    else:
        arr[3] = np.inf
    windows[0][target] = arr
    with pytest.raises(ValueError):
        solve_stage_integrated(windows, kt=KT_300)


def test_solve_stage_integrated_fails_closed_when_windows_do_not_overlap():
    """无共享 λ 时返回 error 且 converged=False（不是抛异常，也不是拼出个数来）。"""
    centers = CENTERS_DEGENERATE
    win_a = _harmonic_window([0, 1], [centers[0], centers[1]], 0, 11)
    win_b = _harmonic_window([3, 4], [centers[3], centers[4]], 1, 12)
    result = solve_stage_integrated([win_a, win_b], kt=KT_300)
    assert result["converged"] is False
    assert result.get("error") in {
        "window_overlap_broken",
        "window_overlap_broken_for_covariance_chain",
    }, f"应报 overlap 断裂，实际: {result.get('error')}"


def test_solve_stage_integrated_empty_input_is_not_converged():
    result = solve_stage_integrated([], kt=KT_300)
    assert result["converged"] is False
    assert result["total_delta_G"] == 0.0


def test_solve_stage_integrated_skips_windows_below_min_frames():
    """帧数低于 min_frames_per_window 的窗口会被跳过；跳过后不足以覆盖全部
    有效窗口时，converged 必须为 False（不能只剩一个窗口就宣称收敛）。"""
    windows = _two_windows(CENTERS_DEGENERATE)
    windows[1] = _harmonic_window(
        [2, 3, 4], [CENTERS_DEGENERATE[i] for i in (2, 3, 4)], 1, 99, n_frames=5
    )
    result = solve_stage_integrated(windows, kt=KT_300, min_frames_per_window=10)
    assert result["converged"] is False


# ============================================================================
# 3. IBSBiasForce —— 在真实 OpenMM Context 里求值，对比手算 log-sum-exp
# ============================================================================
#
# 表达式（ibs_engine.py IBSBiasForce.__init__）：
#   V = bias_scale·[ (cv_0_int+cv_0_rest-f_0)
#                    - kt·( M + log(max(1e-300, Σ_i exp(logit_i - M))) ) ]
#   logit_k = -β·(X_k - X_0),  X_k = cv_k_int + cv_k_rest - f_k,  logit_0 = 0
#   M = max(0, logit_1, ..., logit_{K-1})
# 解析上恒等于  V = -kT·log Σ_k exp(-β·X_k)。

X0_NM = 0.7          # 粒子 0 的 x 坐标，nm
BIAS_TEMP = 300.0


def _kt_beta(temperature_k=BIAS_TEMP):
    kt = (
        unit.MOLAR_GAS_CONSTANT_R * (temperature_k * unit.kelvin)
    ).value_in_unit(unit.kilojoule_per_mole)
    return kt, 1.0 / kt


def _build_bias_context(cv_coefficients, f_values, temperature_k=BIAS_TEMP):
    """把 IBSBiasForce 装进一个 3 粒子裸系统里，返回 (bias, context)。

    每个 CV 用 `CustomExternalForce("c*x")` 只作用在粒子 0 上 → CV 值 = c·x₀，
    完全可控且与坐标线性相关（力非零，顺带保证表达式可微）。
    """
    n_states = len(cv_coefficients)
    system = openmm.System()
    for _ in range(3):
        system.addParticle(1.0 * unit.amu)

    bias = IBSBiasForce(n_states=n_states, temperature=temperature_k * unit.kelvin)
    for k, (c_int, c_rest) in enumerate(cv_coefficients):
        for suffix, coeff in (("int", c_int), ("rest", c_rest)):
            cv_force = openmm.CustomExternalForce(f"{float(coeff)!r}*x")
            cv_force.addParticle(0, [])
            bias.addCollectiveVariable(f"cv_{k}_{suffix}", cv_force)
    system.addForce(bias.get_force())

    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(
        [
            openmm.Vec3(X0_NM, 0.0, 0.0),
            openmm.Vec3(1.0, 0.0, 0.0),
            openmm.Vec3(2.0, 0.0, 0.0),
        ]
        * unit.nanometer
    )
    bias.update_parameters(context, np.asarray(f_values, dtype=float))
    return bias, context


def _bias_energy(context):
    state = context.getState(getEnergy=True, groups={1})
    return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)


def _analytic_bias_energy(cv_coefficients, f_values, temperature_k=BIAS_TEMP):
    """手算 -kT·log Σ_k exp(-β·X_k)。

    用 scipy 的 logsumexp（内部做 max-shift）而不是朴素 Σexp —— 下面那条大能量差
    的测试里朴素写法本身就会溢出成 inf，参考实现必须先站得住。
    """
    kt, beta = _kt_beta(temperature_k)
    x_values = np.array(
        [
            (c_int + c_rest) * X0_NM - float(f_k)
            for (c_int, c_rest), f_k in zip(cv_coefficients, f_values)
        ],
        dtype=float,
    )
    return -kt * logsumexp(-beta * x_values)


def test_ibs_bias_energy_equals_log_sum_exp():
    coefficients = [(3.0, 1.0), (-2.0, 4.0), (8.0, -3.0), (0.5, 0.5)]
    f_values = [0.0, 2.5, -4.0, 1.25]

    _, context = _build_bias_context(coefficients, f_values)
    got = _bias_energy(context)
    expected = _analytic_bias_energy(coefficients, f_values)

    assert abs(got - expected) < 1e-6 * max(1.0, abs(expected)), (
        f"IBS 偏置能量 {got!r} 与手算 log-sum-exp {expected!r} 不一致"
    )


def test_ibs_bias_max_pivot_survives_large_negative_energy_gap():
    """某个 k>0 的 X_k - X_0 = -2500 kJ/mol → logit_k ≈ +1000。

    旧实现用固定的 M = logit_0 = 0 作平移基准，这里 exp(1000) 会溢出成 inf；
    IBS_BIAS_PROTOCOL_VERSION=12 换成真正的 max-pivot（并去掉 80·tanh 饱和）
    后必须既有限、又精确等于解析值。这条直接锁住那次修复。
    """
    coefficients = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    # 所有 CV 系数相同 → X_k 的差异完全来自 f_k：X_1 - X_0 = -2500
    f_values = [0.0, 2500.0, -10.0]

    _, context = _build_bias_context(coefficients, f_values)
    got = _bias_energy(context)
    expected = _analytic_bias_energy(coefficients, f_values)

    assert math.isfinite(got), f"大能量差下偏置能量溢出为 {got!r}"
    assert abs(got - expected) < 1e-6 * max(1.0, abs(expected)), (
        f"max-pivot 结果 {got!r} 应等于解析值 {expected!r}"
    )
    # 主导项就是 X_1，饱和实现会把它系统性压低，这里给一个独立的量级核对。
    dominant_x = 1.0 * X0_NM - 2500.0
    assert abs(got - dominant_x) < 1.0, (
        f"主导态能量 {dominant_x!r} 应支配结果，实测 {got!r}"
    )


def test_ibs_bias_is_correct_when_state_zero_dominates():
    """把巨大 gap 换到 k=0 上（X_0 最低），结果同样必须有限且正确。

    这检验 max-pivot 不是"恰好对 k>0 有效"：k=0 主导时 pivot 落在 0.0 上，
    Σ 里其余项是极小的 exp(负大数)，log(max(1e-300, ...)) 的地板不能把结果拽歪。
    """
    coefficients = [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    f_values = [2500.0, 0.0, -10.0]

    _, context = _build_bias_context(coefficients, f_values)
    got = _bias_energy(context)
    expected = _analytic_bias_energy(coefficients, f_values)

    assert math.isfinite(got)
    assert abs(got - expected) < 1e-6 * max(1.0, abs(expected))


def test_ibs_bias_f_k_uniform_shift_is_absorbed_exactly():
    """f_k 全体加常数 C → 偏置能量精确减少 C。

    -kT log Σ exp(-β(X_k - C)) = -C - kT log Σ exp(-β X_k)
    这条验证 `update_parameters` 写进去的 f_k 语义确实是 exp[-β(U_k - f_k)] 里的
    那个 f_k（而不是差了个符号或因子）。
    """
    coefficients = [(3.0, 1.0), (-2.0, 4.0), (8.0, -3.0)]
    f_values = np.array([0.0, 2.5, -4.0])
    shift = 17.0

    _, ctx_base = _build_bias_context(coefficients, f_values)
    _, ctx_shifted = _build_bias_context(coefficients, f_values + shift)

    assert abs((_bias_energy(ctx_base) - _bias_energy(ctx_shifted)) - shift) < 1e-6


def test_ibs_bias_update_parameters_is_live_on_an_existing_context():
    """同一个 Context 上改 f_k 必须立即改变能量（生产里在线学习就靠这条）。"""
    coefficients = [(3.0, 1.0), (-2.0, 4.0), (8.0, -3.0)]
    f_values = np.array([0.0, 2.5, -4.0])

    bias, context = _build_bias_context(coefficients, f_values)
    before = _bias_energy(context)

    new_f = f_values + np.array([0.0, 5.0, 0.0])
    bias.update_parameters(context, new_f)
    after = _bias_energy(context)

    assert abs(after - _analytic_bias_energy(coefficients, new_f)) < 1e-6 * max(
        1.0, abs(after)
    )
    assert abs(after - before) > 1e-9, "改了 f_k 能量却没变，说明参数没写进 Context"


def test_ibs_bias_scale_zero_disables_the_force():
    coefficients = [(3.0, 1.0), (-2.0, 4.0)]
    f_values = [0.0, 2.5]

    bias, context = _build_bias_context(coefficients, f_values)
    assert abs(_bias_energy(context)) > 1e-9, "基准能量不应恰好为 0，否则本测试无意义"

    bias.set_bias_enabled(context, False)
    assert abs(_bias_energy(context)) < 1e-9

    bias.set_bias_enabled(context, True)
    assert (
        abs(_bias_energy(context) - _analytic_bias_energy(coefficients, f_values)) < 1e-6
    )


def test_ibs_bias_force_group_is_one():
    """e_base/e_bias 的力组切分（WCA_ACCOUNTING_VERSION）假定偏置在 Group 1。"""
    coefficients = [(1.0, 0.0), (1.0, 0.0)]
    bias, _ = _build_bias_context(coefficients, [0.0, 0.0])
    assert bias.get_force().getForceGroup() == 1


# ============================================================================
# 4. estimate_f_k_from_pilot_ti —— f_k = F_k - mean(F)，不反号
# ============================================================================

PILOT_LAMBDAS = [0.0, 0.25, 0.5, 0.75]
PILOT_GRAD = [2.0, 4.0, 6.0, 8.0]          # <dU/dλ>，恒正 → F(λ) 单调递增


def _trapezoid_mean_centered_reference(lambdas, grad, targets):
    """独立参考实现：梯形累积 F(λ)（gauge: F(λ₀)=0）后插值到 targets 再减均值。"""
    lam = np.asarray(lambdas, dtype=float)
    g = np.asarray(grad, dtype=float)
    segments = 0.5 * (g[:-1] + g[1:]) * np.diff(lam)
    f_curve = np.concatenate(([0.0], np.cumsum(segments)))
    at_targets = np.interp(
        np.asarray(targets, dtype=float), lam, f_curve,
        left=f_curve[0], right=f_curve[-1],
    )
    return at_targets - at_targets.mean()


def test_pilot_ti_seed_matches_trapezoidal_integration():
    targets = [0.0, 0.25, 0.5, 0.75]
    got = estimate_f_k_from_pilot_ti(PILOT_LAMBDAS, PILOT_GRAD, targets)
    expected = _trapezoid_mean_centered_reference(PILOT_LAMBDAS, PILOT_GRAD, targets)

    assert got is not None
    # 手算：seg=[0.75, 1.25, 1.75] → F=[0, 0.75, 2.0, 3.75]，mean=1.625
    #        → f=[-1.625, -0.875, 0.375, 2.125]
    np.testing.assert_allclose(
        got, [-1.625, -0.875, 0.375, 2.125], atol=1e-9, rtol=0.0
    )
    np.testing.assert_allclose(got, expected, atol=1e-9, rtol=0.0)


def test_pilot_ti_seed_keeps_physical_sign_and_is_not_inverted():
    """<dU/dλ> 恒正 ⇒ F(λ) 递增 ⇒ f_k 递增、f_k[0] < 0 < f_k[-1]。

    **必须显式断言符号本身**：反号实现（f_k = -(F_k - mean F)）会给出一条同样
    "单调"的曲线，只是整体翻转，任何只检查单调性或只检查均值为零的断言都会
    照样通过。2026-07-20 那次真实的符号反转 bug 正是这种形态。
    """
    got = estimate_f_k_from_pilot_ti(PILOT_LAMBDAS, PILOT_GRAD, PILOT_LAMBDAS)
    assert got is not None
    assert np.all(np.diff(got) > 0), f"梯度恒正时 f_k 应单调递增: {got}"
    assert got[0] < 0.0 < got[-1], f"mean-centering 后首负尾正: {got}"
    # 同时确认曲线本身不是全零（否则符号无从检验，测试会变成空断言）。
    assert np.max(np.abs(got)) > 1e-6


def test_pilot_ti_seed_flips_with_the_gradient_sign():
    """<dU/dλ> 取反 → f_k 精确取反。把符号约定从两个方向都钉住。"""
    positive = estimate_f_k_from_pilot_ti(PILOT_LAMBDAS, PILOT_GRAD, PILOT_LAMBDAS)
    negative = estimate_f_k_from_pilot_ti(
        PILOT_LAMBDAS, [-g for g in PILOT_GRAD], PILOT_LAMBDAS
    )
    assert positive is not None and negative is not None
    np.testing.assert_allclose(negative, -np.asarray(positive), atol=1e-12, rtol=0.0)


def test_pilot_ti_seed_is_exactly_mean_centered():
    got = estimate_f_k_from_pilot_ti(PILOT_LAMBDAS, PILOT_GRAD, [0.0, 0.3, 0.6, 0.75])
    assert got is not None
    assert abs(float(np.mean(got))) < 1e-12


def test_pilot_ti_seed_is_invariant_to_pilot_input_order():
    """pilot 点乱序传入时代码内部会排序；结果必须一致（gauge 由 mean-centering 定）。"""
    shuffled_lambdas = [0.5, 0.0, 0.75, 0.25]
    shuffled_grad = [6.0, 2.0, 8.0, 4.0]
    ordered = estimate_f_k_from_pilot_ti(PILOT_LAMBDAS, PILOT_GRAD, PILOT_LAMBDAS)
    shuffled = estimate_f_k_from_pilot_ti(shuffled_lambdas, shuffled_grad, PILOT_LAMBDAS)
    np.testing.assert_allclose(ordered, shuffled, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(
    "pilot_lambdas, pilot_grad, targets, reason",
    [
        (None, PILOT_GRAD, PILOT_LAMBDAS, "pilot_lambdas 缺失"),
        (PILOT_LAMBDAS, None, PILOT_LAMBDAS, "mean_dU_dlambda 缺失"),
        ([], [], PILOT_LAMBDAS, "空 pilot（旧缓存没有 pilot_points）"),
        ([0.3], [5.0], PILOT_LAMBDAS, "只有 1 个点，梯形积分至少需要 2 个"),
        ([0.0, 0.5], [1.0, 2.0, 3.0], PILOT_LAMBDAS, "长度不匹配"),
        ([0.0, float("nan")], [1.0, 2.0], PILOT_LAMBDAS, "λ 含 NaN"),
        ([0.0, 0.5], [1.0, float("inf")], PILOT_LAMBDAS, "梯度含 Inf"),
        (PILOT_LAMBDAS, PILOT_GRAD, [], "target 为空"),
        (PILOT_LAMBDAS, PILOT_GRAD, [float("nan")], "target 含 NaN"),
        (PILOT_LAMBDAS, [1.0, None, 3.0, 4.0], PILOT_LAMBDAS, "梯度含 None（旧/部分记录）"),
    ],
)
def test_pilot_ti_seed_returns_none_on_untrustworthy_input(
    pilot_lambdas, pilot_grad, targets, reason
):
    """不可信输入必须返回 None（调用方回退 f_k=0.0），既不抛异常也不编造数值。"""
    assert estimate_f_k_from_pilot_ti(pilot_lambdas, pilot_grad, targets) is None, reason


def test_pilot_ti_seed_clamps_out_of_range_targets_without_extrapolating():
    """超出 pilot 实测范围的 target 钳位到边界值，不做外推。

    pilot 只测到 λ∈[0, 0.75]；λ=1.0 必须取 F(0.75) 的值而不是沿斜率外推。
    外推实现会给出明显更大的数，两者可区分。
    """
    targets = [0.0, 0.75, 1.0, -0.5]
    got = estimate_f_k_from_pilot_ti(PILOT_LAMBDAS, PILOT_GRAD, targets)
    assert got is not None
    # 索引 2 (λ=1.0) 钳到右边界 λ=0.75；索引 3 (λ=-0.5) 钳到左边界 λ=0.0
    assert abs(got[2] - got[1]) < 1e-12, "越界 target 未钳位到最近边界（疑似外推）"
    assert abs(got[3] - got[0]) < 1e-12
