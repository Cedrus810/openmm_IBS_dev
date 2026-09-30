"""stage 2 轮次控制器的纯函数层：第一轮每窗左/中/右三段，之后缺啥补啥。

设计全文见 `docs/design/PROPOSAL_stage2_lcr_rounds_2026-09-24.md`。本模块只放**不碰
OpenMM、不碰磁盘**的决策与数值：倾斜、信赖域投影、段间 jackknife、逐窗标签、
`plan_next_round`。执行（开段、采样、合并求解）在 pipeline 里，不在这里。

约定（与 `ibs_engine` 一致）：
  · f_k 单位 kJ/mol，IBS 混合里态 k 的平衡占据 ∝ exp[(f_k − F_k)/kT] ⟹ f_k = F_k + 常数
    时占据平坦。f 只差一个整体常数（规范自由），所以投影、倾斜都只看**相对**量。
  · 倾斜符号：sign > 0 抬首态（窗口里 λ 最大的那个态），sign < 0 抬末态。
    与 `ibs_engine.repro_fk_tilt` 同一形状：首态 +Δ/2、末态 −Δ/2，中间线性。

⚠️ 倾斜必须小。IBS 里 f_k 差 1 kT = 态权重差 e 倍；Δ=2 kT ⟹ 首末权重比 e²≈7，
5 kT 就是 ~150 倍，混合系综塌成单态。锚点误差大的窗口**不靠加大倾斜去罩**，
靠下一轮用合并 MBAR 重定中心。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

STAGE2_ROUNDS_PROTOCOL_VERSION = 1

#: 第一轮左/右倾斜的首末总差（kT）。= `ibs_engine.IBS_REPRO_TILT_MARGIN_KT`（测试钉住两者相等）。
TILT_KT = 2.0
#: 信赖域半径下限（kT）。只挡「追飞」（09-23 批那些窗口偏离锚点 8–24 kT）；
#: w0–2 锚点误差最大 3.1 kT，刚好压线。见设计 §3.5。
TRUST_FLOOR_KT = 3.0
#: 信赖域半径里 pilot 传播误差的倍数。
TRUST_N_SIGMA = 3.0
#: 合并证据的最小逐态 ESS 低于它才投影回信赖域；证据强时 MBAR 比 pilot 准。见设计 §3.5 / §11。
TRUST_MIN_ESS = 50.0
#: 段间 jackknife σ 的绝对容忍度（kJ/mol）。09-24 用 19:16 批标定，见设计 §11.2。
SIGMA_TOL_KJ = 1.5
#: 总轮数上限（含第一轮）。09-24 暂定。
MAX_ROUNDS = 3
#: 逐态绝对 ESS 下限、top-1% 权重上限。= `ibs_engine.TARGET_SUPPORT_*`（测试钉住两者相等）。
ESS_MIN = 20.0
TOP1PCT_MAX = 0.35
#: 复合物腿含 λ_vdw=0 的窗口，第一轮中心在末态上抬多少 kT（09-24 暂定，见设计 §3.1.1 / §10 第 6 条）。
#: pilot 锚点在复合物腿 λ=0 端系统性偏低（19:16 批中位 +2.77 kT，21/23 为正）；溶剂腿没有这个偏差。
ZERO_WINDOW_END_SHIFT_KT_COMPLEX = 2.0
#: 留一段后某态 ESS 低于它 ⟹ 单段依赖（该态只剩那一段在支撑）。
SINGLE_SEGMENT_ESS_FLOOR = 1.0
#: 病因标签：第一轮三段对缺帧态的 ESS 贡献 max/min ≥ 它算「一边高一边低」（f 放偏），否则「三段都低」。
#: 只影响标签，不影响动作；阈值未标定（19:16 批没有新协议的第一轮数据）。
ONE_SIDED_RATIO = 3.0

PATTERN_ONE_SIDED = "one_sided"
PATTERN_ALL_LOW = "all_low"
#: 第一轮每窗段数：复合物腿 3（中 / 左 / 右），溶剂腿 1（只中段）。09-25 定，依据设计 §11.5：
#: 溶剂腿 w0–2、w4 rep 间 SD ≤ 0.9 kJ、三段 σ_jk 无一 > 1.5；唯一出问题的 w3（灌水双稳态）
#: 单段切块 jackknife 也能抓到。复合物腿的慢模态（口袋侧链态）在一段里不翻转，切块抓不到。
FIRST_ROUND_SEGMENTS_COMPLEX = 3
FIRST_ROUND_SEGMENTS_SOLVENT = 1
#: 收敛标签要求的最少独立种子段数：复合物腿 2，溶剂腿 1（同上依据）。
MIN_INDEPENDENT_SEEDS_COMPLEX = 2
MIN_INDEPENDENT_SEEDS_SOLVENT = 1

LABEL_CONVERGED = "CONVERGED"
LABEL_PROVISIONAL = "PROVISIONAL"
LABEL_INDETERMINATE = "INDETERMINATE"

CENTER_ANCHOR = "anchor"
CENTER_MERGED = "merged_mbar"
CENTER_MERGED_PROJECTED = "merged_mbar_projected"


def window_touches_zero(lambdas_vdw: Sequence[float]) -> bool:
    """窗口是否含 λ_vdw = 0。这类窗口 pilot 锚点最不准（`ibs_engine.repro_fk_tilt` 同一判据）。"""
    return bool(np.isclose(float(np.min(np.asarray(lambdas_vdw, dtype=float))), 0.0))


def _linear_ramp(k: int) -> np.ndarray:
    """0 → 1 的线性坐标，首态 0、末态 1。单态窗口返回 [0]。"""
    return np.linspace(0.0, 1.0, k) if k > 1 else np.zeros(1)


def tilt_f_k(center: Sequence[float], kt: float, sign: int, delta_kT: float = TILT_KT) -> np.ndarray:
    """左/右倾斜：首态 +sign·Δ/2、末态 −sign·Δ/2，中间线性；sign = 0 原样返回。"""
    f = np.asarray(center, dtype=np.float64).ravel().copy()
    if int(sign) == 0:
        return f
    s = 1.0 if int(sign) > 0 else -1.0
    return f + s * float(delta_kT) * float(kt) * (0.5 - _linear_ramp(f.size))


def shift_last_state(center: Sequence[float], kt: float, shift_kT: float) -> np.ndarray:
    """末态相对首态上抬 shift_kT（线性），保持均值不变（f 的规范约定是均值居中）。"""
    f = np.asarray(center, dtype=np.float64).ravel()
    g = f + float(shift_kT) * float(kt) * _linear_ramp(f.size)
    return g - g.mean() + f.mean()


def pilot_trust_radius_kJ(
    pilot_lambdas: Optional[Sequence[float]],
    pilot_std_dU_dlambda: Optional[Sequence[float]],
    pilot_n_samples: Optional[Sequence[int]],
    target_lambdas: Sequence[float],
    kt: float,
    floor_kT: float = TRUST_FLOOR_KT,
    n_sigma: float = TRUST_N_SIGMA,
) -> float:
    """信赖域半径 R = max(floor·kT, n_sigma·σ)。

    σ = pilot 在本窗 λ 跨度上的梯形传播误差：
    sqrt(Σ (½Δλ)²·(sem_i² + sem_{i+1}²))，sem = std/√n（与
    `abfe_preoptimizer.pilot_ti_seed_trust_diagnostics` 同一口径）。
    ⚠️ pilot 样本有自相关，sem 偏小 ⟹ 实际上几乎总是下限起作用；精度数据缺失时直接用下限。
    """
    floor = float(floor_kT) * float(kt)
    try:
        lam = np.asarray(pilot_lambdas, dtype=float).ravel()
        std = np.asarray(pilot_std_dU_dlambda, dtype=float).ravel()
        n = np.asarray(pilot_n_samples, dtype=float).ravel()
    except (TypeError, ValueError):
        return floor
    if lam.size < 2 or std.size != lam.size or n.size != lam.size:
        return floor
    if not (np.all(np.isfinite(lam)) and np.all(np.isfinite(std)) and np.all(n > 0)):
        return floor
    order = np.argsort(lam)
    lam, sem = lam[order], (std / np.sqrt(n))[order]
    tgt = np.asarray(target_lambdas, dtype=float).ravel()
    lo, hi = float(np.min(tgt)), float(np.max(tgt))
    var = 0.0
    for i in range(lam.size - 1):
        a, b = lam[i], lam[i + 1]
        if b <= lo or a >= hi:
            continue
        d = min(b, hi) - max(a, lo)
        var += (0.5 * d) ** 2 * (sem[i] ** 2 + sem[i + 1] ** 2)
    return max(floor, float(n_sigma) * float(np.sqrt(var)))


def project_to_trust_region(
    f_k: Sequence[float], anchor: Sequence[float], radius_kJ: float
) -> Tuple[np.ndarray, bool]:
    """把逐态累计偏移 (f_k − f_0) 投影回 (a_k − a_0) ± R，保持 f 的均值不变。

    约束的是**每个态**相对首态的偏移，不只是首末总差：总差对了、中间某个态偏到天上，
    也会让窗口内部一侧的占据塌掉。返回 (投影后的 f, 是否真的裁了)。
    """
    f = np.asarray(f_k, dtype=np.float64).ravel()
    a = np.asarray(anchor, dtype=np.float64).ravel()
    if f.size != a.size:
        raise ValueError(f"f_k 长度 {f.size} 与锚点长度 {a.size} 不符")
    dev = (f - f[0]) - (a - a[0])
    r = float(radius_kJ)
    clamped = bool(np.any(np.abs(dev) > r))
    g = f[0] + (a - a[0]) + np.clip(dev, -r, r)
    return g - g.mean() + f.mean(), clamped


def jackknife_sigma(loo_values: Sequence[float]) -> Optional[float]:
    """留一段 jackknife：σ² = (n−1)/n · Σ(x_i − x̄)²。

    n < 2 或有非有限值就返回 None（证据不足，**不能**当成 σ = 0）。
    ⚠️ 有限能量输入下留一求解实际上不会失败（19:16 批 234 窗 × 3 次留一 0 失败）；
    「单段依赖」看的是留一之后某态 ESS 是否塌掉，见 `WindowEvidence.single_segment_dependent`。
    """
    x = np.asarray(list(loo_values), dtype=float).ravel()
    if x.size < 2 or not np.all(np.isfinite(x)):
        return None
    n = x.size
    return float(np.sqrt((n - 1) / n * np.sum((x - x.mean()) ** 2)))


@dataclass
class WindowEvidence:
    """一轮合并求解之后，单个窗口的证据。全部 kJ/mol。"""

    window: int
    lambdas_vdw: List[float]
    state_ess: List[float]                  # 合并 MBAR 的逐态绝对 ESS
    top1pct_weight: float
    sigma_mbar_kJ: float
    loo_delta_f_kJ: List[float]             # 留一段的窗口 ΔF
    n_independent_seeds: int                # 贡献过帧的独立速度种子段数（起点统一是预平衡末帧）
    merged_f_k_kJ: List[float]              # 合并 MBAR 的逐态 F_k（任意规范）
    anchor_f_k_kJ: Optional[List[float]] = None
    trust_radius_kJ: Optional[float] = None
    delta_f_finite: bool = True
    #: 每次留一段之后的最小逐态 ESS（与 loo_delta_f_kJ 同序）。缺省 = 不判单段依赖。
    loo_min_state_ess: Optional[List[float]] = None
    #: 第一轮三段各自对每个态贡献的有效样本（段 × 态）。只用于病因标签，不参与计划。
    first_round_segment_state_ess: Optional[List[List[float]]] = None

    @property
    def sigma_jk_kJ(self) -> Optional[float]:
        return jackknife_sigma(self.loo_delta_f_kJ)

    @property
    def single_segment_dependent(self) -> bool:
        """去掉某一段后某态 ESS 塌到 ~0（只有那一段支撑它），或留一 ΔF 非有限。"""
        if any(not np.isfinite(x) for x in self.loo_delta_f_kJ):
            return True
        if self.loo_min_state_ess is None:
            return False
        return any(float(x) < SINGLE_SEGMENT_ESS_FLOOR for x in self.loo_min_state_ess)


def starved_pattern(ev: WindowEvidence, ess_min: float = ESS_MIN) -> Optional[str]:
    """缺帧态在第一轮三段里的 ESS 模式；没缺帧或没有三段数据返回 None。"""
    ess = np.asarray(ev.state_ess, dtype=float)
    if ev.first_round_segment_state_ess is None or ess.size == 0:
        return None
    ess = np.where(np.isfinite(ess), ess, 0.0)
    if not np.any(ess < ess_min):
        return None
    k = int(np.argmin(ess))
    per_seg = np.array([float(row[k]) for row in ev.first_round_segment_state_ess])
    lo, hi = float(per_seg.min()), float(per_seg.max())
    if hi > 0 and (lo <= 0 or hi / lo >= ONE_SIDED_RATIO):
        return PATTERN_ONE_SIDED
    return PATTERN_ALL_LOW


@dataclass
class SegmentSpec:
    """要采的一段。执行层照它开段：中心 f_k 加倾斜，从预平衡末帧、用 `seed_id` 派生的速度种子起跑。"""

    window: int
    center_f_k_kJ: List[float]
    center_source: str
    tilt_sign: int
    tilt_kT: float
    seed_id: int
    round_index: int
    reason: str = ""

    def sampling_f_k(self, kt: float) -> np.ndarray:
        return tilt_f_k(self.center_f_k_kJ, kt, self.tilt_sign, self.tilt_kT)


def first_round_specs(
    windows: Sequence[Tuple[int, Sequence[float], Sequence[float]]],
    kt: float,
    *,
    is_complex_leg: bool,
    tilt_kT: float = TILT_KT,
    zero_window_end_shift_kT: float = ZERO_WINDOW_END_SHIFT_KT_COMPLEX,
    n_segments: Optional[int] = None,
) -> List[SegmentSpec]:
    """第一轮：复合物腿每窗三段（中 0 / 左 +1 / 右 −1），溶剂腿每窗只中段（设计 §3.1.2）。

    中心 = pilot 锚点，种子 id 从 0 起、各段不同。`windows` 每项是 (窗口号, 该窗 λ_vdw, 该窗锚点 f_k)。
    复合物腿含 λ=0 的窗口，中心先在末态上抬 `zero_window_end_shift_kT`（设计 §3.1.1）；溶剂腿不抬。
    倾斜**只在这一轮**用（设计 §3.3）。`n_segments` 缺省按腿取 FIRST_ROUND_SEGMENTS_*。
    """
    if n_segments is None:
        n_segments = FIRST_ROUND_SEGMENTS_COMPLEX if is_complex_leg else FIRST_ROUND_SEGMENTS_SOLVENT
    if int(n_segments) not in (1, 3):
        raise ValueError(f"第一轮每窗段数只能是 1 或 3，得到 {n_segments!r}")
    signs = (0, 1, -1)[: int(n_segments)]
    specs: List[SegmentSpec] = []
    for w, lv, anchor in windows:
        center = np.asarray(anchor, dtype=np.float64).ravel()
        if is_complex_leg and window_touches_zero(lv) and zero_window_end_shift_kT:
            center = shift_last_state(center, kt, zero_window_end_shift_kT)
        # 顺序 中 / 左 / 右 ⟹ 种子 0 = 中 = 段 1（基准目录），与既有段约定一致。
        for seed_id, sign in enumerate(signs):
            specs.append(SegmentSpec(
                window=int(w), center_f_k_kJ=[float(x) for x in center],
                center_source=CENTER_ANCHOR, tilt_sign=sign,
                tilt_kT=float(tilt_kT) if sign else 0.0,
                seed_id=seed_id, round_index=1, reason="first_round",
            ))
    return specs


def classify_window(
    ev: WindowEvidence,
    *,
    sigma_tol_kJ: float = SIGMA_TOL_KJ,
    ess_min: float = ESS_MIN,
    top1pct_max: float = TOP1PCT_MAX,
    min_independent_seeds: int = MIN_INDEPENDENT_SEEDS_COMPLEX,
) -> Tuple[str, List[str]]:
    """逐窗标签（设计 §3.4 / §12.10）。返回 (标签, 不满足的条件列表)。

    单段窗口的 σ_jk 来自段内切块 jackknife（`stage2_rounds_evidence`），溶剂腿用它就够（§11.5）。
    """
    ess = np.asarray(ev.state_ess, dtype=float)
    if not ev.delta_f_finite or ess.size == 0 or np.any(~np.isfinite(ess)) or np.any(ess <= 0.0):
        return LABEL_INDETERMINATE, ["state_without_usable_frames"]
    failed: List[str] = []
    if np.any(ess < ess_min):
        failed.append("state_ess_below_min")
        if starved_pattern(ev, ess_min) == PATTERN_ALL_LOW:
            failed.append("sampling_limited_cause_unresolved")
    if not (ev.top1pct_weight <= top1pct_max):
        failed.append("top1pct_weight_above_max")
    sjk = ev.sigma_jk_kJ
    if sjk is None:
        failed.append("sigma_jk_unavailable")
    elif sjk > sigma_tol_kJ:
        failed.append("sigma_jk_above_tol")
    if ev.single_segment_dependent:
        failed.append("single_segment_dependent")
    if ev.n_independent_seeds < int(min_independent_seeds):
        failed.append("fewer_than_two_independent_seeds")
    return (LABEL_CONVERGED if not failed else LABEL_PROVISIONAL), failed


def sigma_window_kJ(ev: WindowEvidence, split_half_sigma_kJ: Optional[float] = None) -> float:
    """σ_window = max(σ_MBAR, σ_jk, split-half 膨胀值)（设计 §3.6）；缺的项不参与。"""
    vals = [float(ev.sigma_mbar_kJ)]
    if ev.sigma_jk_kJ is not None:
        vals.append(ev.sigma_jk_kJ)
    if split_half_sigma_kJ is not None and np.isfinite(split_half_sigma_kJ):
        vals.append(float(split_half_sigma_kJ))
    return max(vals)


def _next_center(ev: WindowEvidence, trust_min_ess: float) -> Tuple[List[float], str]:
    """补段的中心：合并 MBAR 的 F_k；证据弱（最小逐态 ESS < trust_min_ess）且不含 λ=0 时投影回信赖域。"""
    f = np.asarray(ev.merged_f_k_kJ, dtype=np.float64).ravel()
    weak = float(np.min(ev.state_ess)) < float(trust_min_ess)
    if (weak and ev.anchor_f_k_kJ is not None and ev.trust_radius_kJ is not None
            and not window_touches_zero(ev.lambdas_vdw)):
        g, clamped = project_to_trust_region(f, ev.anchor_f_k_kJ, ev.trust_radius_kJ)
        if clamped:
            return [float(x) for x in g], CENTER_MERGED_PROJECTED
    return [float(x) for x in f], CENTER_MERGED


def plan_next_round(
    evidences: Sequence[WindowEvidence],
    *,
    completed_round: int,
    used_seed_ids: Sequence[int],
    max_rounds: int = MAX_ROUNDS,
    sigma_tol_kJ: float = SIGMA_TOL_KJ,
    ess_min: float = ESS_MIN,
    top1pct_max: float = TOP1PCT_MAX,
    trust_min_ess: float = TRUST_MIN_ESS,
    min_independent_seeds: int = MIN_INDEPENDENT_SEEDS_COMPLEX,
) -> List[SegmentSpec]:
    """控制器唯一的决策：第 `completed_round` 轮之后，每窗最多补一段（设计 §3.3）。

    每窗按序看，第一条命中的规则只决定**理由**：
      1. 缺帧：某态 ESS < ess_min（含某态一帧都没有）。
      2. 段间不一致：单段依赖，或 σ_jk > σ_tol，或 top-1% > 上限。
         （09-29 起补段门与 `classify_window` 的 PROVISIONAL 门同为 σ_jk > σ_tol；原先是
         max(2σ_MBAR, σ_tol) ⟹ σ_jk ∈ (σ_tol, 2σ_MBAR] 的窗口标 PROVISIONAL 却永远不补。
         理由只用来区分两种情况：超过 2σ_MBAR = 段间不一致，否则 = 精度没到。）
      3. 都不命中 ⟹ 该窗完成。
    命中任一条，**动作都一样**（09-25 定，数据见设计 §11.3）：不倾斜、中心 = 合并 MBAR 的 F_k
    （证据弱且不含 λ=0 时投影回信赖域）、换一个没用过的种子 id。倾斜只在第一轮用。
    达到轮数上限返回空表。
    """
    if int(completed_round) >= int(max_rounds):
        return []
    next_round = int(completed_round) + 1
    next_seed = (max(used_seed_ids) + 1) if len(used_seed_ids) else 0
    specs: List[SegmentSpec] = []
    for ev in sorted(evidences, key=lambda e: e.window):
        ess = np.asarray(ev.state_ess, dtype=float)
        ess = np.where(np.isfinite(ess), ess, 0.0)
        reason = ""
        if ess.size and np.any(ess < ess_min):
            k = int(np.argmin(ess))
            reason = f"starved_state_{k}_ess_{float(ess[k]):.1f}"
            pattern = starved_pattern(ev, ess_min)
            if pattern is not None:
                reason += f"_{pattern}"
        elif ev.single_segment_dependent:
            reason = "single_segment_dependence"
        elif ev.n_independent_seeds < int(min_independent_seeds):
            reason = "too_few_independent_seeds"
        else:
            sjk = ev.sigma_jk_kJ
            if sjk is None:
                reason = "sigma_jk_unavailable"
            elif sjk > float(sigma_tol_kJ):
                if sjk > 2.0 * float(ev.sigma_mbar_kJ):
                    reason = f"segments_disagree_sigma_jk_{sjk:.2f}"
                else:
                    reason = f"sigma_jk_above_tol_{sjk:.2f}"
            elif not (ev.top1pct_weight <= top1pct_max):
                reason = f"top1pct_weight_{ev.top1pct_weight:.2f}"
        if not reason:
            continue
        center, source = _next_center(ev, trust_min_ess)
        specs.append(SegmentSpec(
            window=int(ev.window), center_f_k_kJ=center, center_source=source,
            tilt_sign=0, tilt_kT=0.0, seed_id=next_seed, round_index=next_round, reason=reason,
        ))
        next_seed += 1
    return specs
