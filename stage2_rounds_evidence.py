"""stage 2 轮次控制器的逐窗证据（数值层）。

输入是**一个窗口**在各段的产物（采样规范能量、目标能量、该段生产用的 f_k），输出
`stage2_rounds.WindowEvidence` 需要的数：逐态 ESS、top-1% 权重、合并 F_k、窗口 ΔF、
留一段的 ΔF 与最小逐态 ESS、第一轮各段对每个态的 ESS 贡献、相邻态重叠（只记录）。

⚠️ 不另写 MBAR：多段求解全部走 `multi_segment_analysis`（`build_cross_bias` +
`collapse_segments`，与生产合并求解同一份实现）；top-1% 权重走
`ibs_engine._top1pct_weight_by_empirical_cdf`（唯一实现）。
⚠️ 用**全部帧**（不去相关）：σ 的去相关版本来自生产合并求解，这里只出 ESS / 权重 /
留一段，口径与设计 §11 的重放标定一致。

设计见 docs/design/PROPOSAL_stage2_lcr_rounds_2026-09-24.md §3.2 / §12.3。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import multi_segment_analysis as msa

STAGE2_ROUNDS_EVIDENCE_PROTOCOL_VERSION = 1
#: 单段窗口的 jackknife 把这一段按时间切成几块（09-25，溶剂腿第一轮只跑一段，设计 §11.5）。
WITHIN_SEGMENT_BLOCKS = 3


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(m, axis=axis) + np.log(np.sum(np.exp(x - m), axis=axis))


def _top1pct(w: np.ndarray) -> float:
    from ibs_engine import _top1pct_weight_by_empirical_cdf
    return float(_top1pct_weight_by_empirical_cdf(w))


def solve_window(
    sampling_by_segment: Sequence[np.ndarray],
    target_by_segment: Sequence[np.ndarray],
    f_by_segment: Sequence[Sequence[float]],
    kt: float,
) -> Dict[str, Any]:
    """一次多段合并求解。能量形状一律 (K, N_s)，单位 kJ/mol。

    返回：窗口 ΔF（首态 → 末态）及其 MBAR σ、逐态 F_k（首态为 0）、逐态 ESS、
    top-1% 权重（逐态取最大）、归一化目标权重 W (K, N)、帧所属段 source_id。
    """
    cross, src = msa.build_cross_bias(sampling_by_segment, f_by_segment, kt)
    u = np.hstack([np.asarray(t, dtype=np.float64) for t in target_by_segment])
    res = msa.collapse_segments(cross, src, u, kt, n_segments=len(sampling_by_segment))
    k = u.shape[0]
    logw = -u / float(kt) - np.asarray(res["log_D"], dtype=np.float64)[None, :]
    log_norm = _logsumexp(logw, axis=1)
    w = np.exp(logw - log_norm[:, None])
    f_k = -float(kt) * log_norm
    f_k = f_k - f_k[0]
    ess = 1.0 / np.sum(w ** 2, axis=1)
    top1 = max(_top1pct(w[i]) for i in range(k))
    d_f = np.asarray(res["target_delta_f_kJ_mol"], dtype=np.float64)
    dd_f = np.asarray(res["target_dDelta_f_kJ_mol"], dtype=np.float64)
    return {
        "delta_f_kJ": float(d_f[0, k - 1]),
        "sigma_mbar_kJ": float(dd_f[0, k - 1]),
        "f_k_kJ": [float(x) for x in f_k],
        "state_ess": [float(x) for x in ess],
        "top1pct_weight": float(top1),
        "weights": w,
        "source_id": np.asarray(src, dtype=np.int64),
    }


def per_segment_state_ess(weights: np.ndarray, source_id: np.ndarray, segments: Sequence[int]) -> List[List[float]]:
    """每段对每个态贡献的有效样本：(Σ_{n∈s} w_nk)² / Σ_{n∈s} w_nk²。"""
    out = []
    for s in segments:
        ws = weights[:, source_id == int(s)]
        num = np.sum(ws, axis=1) ** 2
        den = np.sum(ws ** 2, axis=1)
        out.append([float(a / b) if b > 0 else 0.0 for a, b in zip(num, den)])
    return out


def adjacent_overlap(weights: np.ndarray) -> List[float]:
    """相邻态重叠 Σ_n min(w_{k,n}, w_{k+1,n})（0..1）。**只记录，不参与决策**（设计 §3.3）。"""
    return [float(np.minimum(weights[i], weights[i + 1]).sum()) for i in range(weights.shape[0] - 1)]


def window_evidence_numbers(
    sampling_by_segment: Sequence[np.ndarray],
    target_by_segment: Sequence[np.ndarray],
    f_by_segment: Sequence[Sequence[float]],
    kt: float,
    *,
    first_round_segments: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """一个窗口的全部证据数（不含 σ_MBAR 的去相关版本、不含锚点 —— 那两样由调用方补）。

    `first_round_segments`：第一轮三段在输入列表里的下标（用于病因标签）；None = 不算。
    留一段：段数 ≥ 2 时按段做；**只有一段时按时间切成 `WITHIN_SEGMENT_BLOCKS` 块做**（09-25：
    溶剂腿第一轮只跑一段；切块在溶剂腿能抓到 w3 的灌水双稳态，复合物腿的慢模态抓不到，
    所以复合物腿仍靠多段）。某次留一求解失败记 nan（设计 §12.10 A11：有限输入下不应发生）。
    """
    n_seg = len(sampling_by_segment)
    full = solve_window(sampling_by_segment, target_by_segment, f_by_segment, kt)
    loo_df: List[float] = []
    loo_ess: List[float] = []
    if n_seg >= 2:
        jk_s, jk_t, jk_f, jk_kind = sampling_by_segment, target_by_segment, f_by_segment, "segments"
    else:
        n_frames = np.asarray(sampling_by_segment[0]).shape[1]
        cuts = np.array_split(np.arange(n_frames), WITHIN_SEGMENT_BLOCKS)
        jk_s = [np.asarray(sampling_by_segment[0])[:, c] for c in cuts]
        jk_t = [np.asarray(target_by_segment[0])[:, c] for c in cuts]
        jk_f = [f_by_segment[0]] * len(cuts)
        jk_kind = "within_segment_blocks"
    n_jk = len(jk_s)
    if n_jk >= 2:
        for i in range(n_jk):
            keep = [j for j in range(n_jk) if j != i]
            try:
                r = solve_window(
                    [jk_s[j] for j in keep],
                    [jk_t[j] for j in keep],
                    [jk_f[j] for j in keep],
                    kt,
                )
                loo_df.append(r["delta_f_kJ"])
                loo_ess.append(float(min(r["state_ess"])))
            except (msa.MultiSegmentSolveError, FloatingPointError, ValueError):
                loo_df.append(float("nan"))
                loo_ess.append(0.0)
    seg_state_ess = (
        per_segment_state_ess(full["weights"], full["source_id"], first_round_segments)
        if first_round_segments is not None else None
    )
    return {
        "protocol_version": STAGE2_ROUNDS_EVIDENCE_PROTOCOL_VERSION,
        "n_segments": int(n_seg),
        "delta_f_kJ": full["delta_f_kJ"],
        "sigma_mbar_all_frames_kJ": full["sigma_mbar_kJ"],
        "merged_f_k_kJ": full["f_k_kJ"],
        "state_ess": full["state_ess"],
        "top1pct_weight": full["top1pct_weight"],
        "loo_delta_f_kJ": loo_df,
        "loo_min_state_ess": loo_ess,
        "jackknife_kind": jk_kind,
        "first_round_segment_state_ess": seg_state_ess,
        "adjacent_overlap": adjacent_overlap(full["weights"]),
        "delta_f_finite": bool(np.isfinite(full["delta_f_kJ"])),
    }
