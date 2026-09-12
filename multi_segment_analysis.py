"""多采样段分析适配层：把同一个 IBS 窗口的多段采样**相加**而不是替换。

背景。窗口的 f_k 被重标定之后再采一段，两段就是**两个不同的采样分布**，不能
当成一条轨迹的延长。直接把两段记录的 `bias_energies` 摞起来当单一采样态是错的
（合成对照 12 seed：偏差 +0.12 kJ/mol、RMSE 2.6 倍）。而把第二段当成
`window_outputs` 里的另一条 entry 也没用 —— `solve_stage_integrated` 是「每窗口
局部解 + 用共享态偏移拼接」，同一组 `lambda_indices` 的第二条只贡献一个偏移量，
实测结果逐比特不变，第二段静默失效。

本层的做法：保留段身份，在**最终帧集**上做一次自洽塌缩，产出一条等价的混合
偏置交给现有单采样态路径；σ 另由真正的 S+K MBAR 给出。

⚠️ 三条硬约束，都是实测踩出来的：
  · 交叉能量必须用 `sampling_states` 重建，**不能**用 `energies`。后者比前者多一个
    逐 λ 态常数（LJ 长程尾项：在分析侧目标能量里，不在采样哈密顿量里）。逐态常数
    不是共模、不会在 logsumexp 里抵消。真实产物实测（cyclod rep1 四窗口）：用
    sampling_states 对账 sd=0.0000，用 energies 得 0.735/0.373/0.182/0.094。
  · 塌缩必须发生在帧集**定下来之后**（去相关抽帧、split-half 切半都会改变各段
    计数，混合权重跟着变）。用全帧解出的权重切片后不再精确。
  · 数值不收敛可以降级成「各段独立分析」，**输入错误不行**。能量错位、采样身份
    不一致、账本损坏必须继续 fail-closed，否则等于把静默错误重新埋回去。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

MULTI_SEGMENT_ANALYSIS_PROTOCOL_VERSION = 1

#: 塌缩路径与 S+K MBAR 路径的目标态间 ΔF 必须对到这个精度，否则报错。
CROSS_CHECK_MAX_DELTA_F_KJ_MOL = 1.0e-6


class MultiSegmentInputError(ValueError):
    """输入本身不自洽（形状/规范/身份/账本）。**绝不允许降级**。"""


class MultiSegmentSolveError(RuntimeError):
    """数值求解失败（不收敛、奇异、pymbar 抛错）。允许降级成各段独立分析。"""


def _logsumexp(a: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return (m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))).squeeze(axis)


def mixture_energy(sampling_kj: np.ndarray, f_k: np.ndarray, kt: float) -> np.ndarray:
    """某组冻结 f_k 定义的 IBS 混合系综能量，可对**任意**帧求值。

    这就是落盘 `bias_energies` 的定义式；对自己那段的帧重算应逐帧吻合到
    float32 精度（~1e-5 kJ/mol），这一点被 `recalibrate_f_k_from_production`
    的规范对账用作 fail-closed 判据。
    """
    s = np.asarray(sampling_kj, dtype=np.float64)
    f = np.asarray(f_k, dtype=np.float64).ravel()
    if s.ndim != 2 or s.shape[0] != f.size:
        raise MultiSegmentInputError(
            f"sampling_states 形状 {s.shape} 与 f_k 长度 {f.size} 不匹配"
        )
    return -float(kt) * _logsumexp(-(s - f[:, None]) / float(kt), axis=0)


def build_cross_bias(
    sampling_by_segment: Sequence[np.ndarray],
    frozen_f_by_segment: Sequence[Sequence[float]],
    kt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """拼帧 + 补交叉能量。返回 ``(cross_bias (S,N), source_id (N,))``。

    交叉能量 = 每个采样分布在**全部**帧上的 bias。MBAR 要求对每个采样态 s 能算出
    u_s(x_n) 对所有 n，而不只是从 s 抽出来的那些帧 —— 这正是「直接摞 bias」错在
    哪里。
    """
    if len(sampling_by_segment) != len(frozen_f_by_segment):
        raise MultiSegmentInputError(
            f"段数不一致：sampling {len(sampling_by_segment)} 段、"
            f"f_k {len(frozen_f_by_segment)} 段"
        )
    if not sampling_by_segment:
        raise MultiSegmentInputError("至少需要一段采样")

    mats = [np.asarray(x, dtype=np.float64) for x in sampling_by_segment]
    k_states = {m.shape[0] for m in mats}
    if len(k_states) != 1:
        raise MultiSegmentInputError(f"各段态数不一致：{sorted(k_states)}")
    for i, (m, f) in enumerate(zip(mats, frozen_f_by_segment)):
        if m.ndim != 2:
            raise MultiSegmentInputError(f"第 {i} 段 sampling_states 不是二维：{m.shape}")
        if len(np.asarray(f).ravel()) != m.shape[0]:
            raise MultiSegmentInputError(
                f"第 {i} 段 f_k 长度 {len(np.asarray(f).ravel())} != 态数 {m.shape[0]}"
            )
        if not np.all(np.isfinite(m)):
            raise MultiSegmentInputError(f"第 {i} 段 sampling_states 含非有限值")

    all_sampling = np.hstack(mats)
    source_id = np.concatenate([
        np.full(m.shape[1], i, dtype=np.int64) for i, m in enumerate(mats)
    ])
    cross = np.vstack([
        mixture_energy(all_sampling, np.asarray(f, dtype=np.float64), kt)
        for f in frozen_f_by_segment
    ])
    return cross, source_id


def collapse_segments(
    cross_bias: np.ndarray,
    source_id: np.ndarray,
    u_targets: np.ndarray,
    kt: float,
    *,
    n_segments: Optional[int] = None,
) -> Dict[str, Any]:
    """在**给定的这批帧**上自洽塌缩，并给出真正的多采样态 σ。

    一次 ``S+K`` MBAR 同时拿到：采样态自由能 ``f_s``（用来构造混合偏置）和物理
    目标态之间的成对标准误（塌缩后的单态解会把估计出来的混合权重当已知常数，
    σ 会偏小，所以必须由这次解回填）。

    零计数段从采样子问题中剔除（它对混合没有贡献，且 log(0) 会炸），但保留原始
    段号到有效行号的映射与剔除原因。规范锚在**第一个有效**采样态上。
    """
    cross = np.asarray(cross_bias, dtype=np.float64)
    src = np.asarray(source_id, dtype=np.int64).ravel()
    u = np.asarray(u_targets, dtype=np.float64)
    if cross.ndim != 2 or u.ndim != 2:
        raise MultiSegmentInputError(f"维度不对：cross={cross.shape} u_targets={u.shape}")
    if cross.shape[1] != u.shape[1] or src.size != cross.shape[1]:
        raise MultiSegmentInputError(
            f"帧数不一致：cross={cross.shape[1]} u_targets={u.shape[1]} source_id={src.size}"
        )
    if not (np.all(np.isfinite(cross)) and np.all(np.isfinite(u))):
        raise MultiSegmentInputError("cross_bias / u_targets 含非有限值")

    n_seg = int(n_segments if n_segments is not None else cross.shape[0])
    if cross.shape[0] != n_seg:
        raise MultiSegmentInputError(f"cross_bias 行数 {cross.shape[0]} != 段数 {n_seg}")
    counts = np.bincount(src, minlength=n_seg).astype(np.int64)
    if counts.sum() != src.size:
        raise MultiSegmentInputError(f"source_id 含越界段号：max={src.max()} 段数={n_seg}")

    active = np.flatnonzero(counts > 0)
    if active.size == 0:
        raise MultiSegmentInputError("所有段的帧数都是 0")
    excluded = [
        {"segment": int(s), "n_frames": 0, "reason": "zero_count_after_frame_selection"}
        for s in range(n_seg) if counts[s] == 0
    ]

    n_frames = int(src.size)
    n_targets = int(u.shape[0])
    u_sampled_active = cross[active] / float(kt)
    u_all = np.vstack([u_sampled_active, u / float(kt)])
    n_k = np.concatenate([counts[active], np.zeros(n_targets, dtype=np.int64)])

    try:
        from pymbar import MBAR
    except Exception as exc:  # pragma: no cover - 环境问题不是数值失败
        raise MultiSegmentInputError(f"pymbar 不可用：{exc!r}") from exc

    try:
        mbar = MBAR(u_all, n_k, verbose=False)
        solved = mbar.compute_free_energy_differences()
    except Exception as exc:
        raise MultiSegmentSolveError(f"S+K MBAR 求解失败：{exc!r}") from exc

    f_all = np.asarray(mbar.f_k, dtype=np.float64).ravel()
    if f_all.size != active.size + n_targets or not np.all(np.isfinite(f_all)):
        raise MultiSegmentSolveError("MBAR 解含非有限自由能")

    f_s = f_all[: active.size]
    f_s = f_s - f_s[0]                       # 规范锚在第一个**有效**采样态
    log_counts = np.log(counts[active].astype(np.float64))
    log_D = _logsumexp(log_counts[:, None] + f_s[:, None] - u_sampled_active, axis=0)
    if not np.all(np.isfinite(log_D)):
        raise MultiSegmentSolveError("混合分母 log_D 含非有限值")
    bias_mix = -float(kt) * (log_D - np.log(float(counts.sum())))

    dF = np.asarray(solved["Delta_f"], dtype=np.float64)
    ddF = np.asarray(solved["dDelta_f"], dtype=np.float64)
    s0 = active.size
    return {
        "protocol_version": MULTI_SEGMENT_ANALYSIS_PROTOCOL_VERSION,
        "bias_mix_kJ_mol": bias_mix,
        "log_D": log_D,
        "f_sampling_reduced": f_s,
        "counts": counts,
        "active_segments": active,
        "reference_sampling_segment": int(active[0]),
        "excluded_segments": excluded,
        "n_frames": n_frames,
        # 真正的多采样态口径：物理目标态之间的 ΔF 与成对标准误。
        "target_delta_f_kJ_mol": dF[s0:, s0:] * float(kt),
        "target_dDelta_f_kJ_mol": ddF[s0:, s0:] * float(kt),
        "responsibility": np.exp(
            log_counts[:, None] + f_s[:, None] - u_sampled_active - log_D[None, :]
        ),
    }


def assert_collapse_matches_multistate(
    collapsed_target_delta_f_kJ_mol: np.ndarray,
    reference_target_delta_f_kJ_mol: np.ndarray,
    *,
    tol_kJ_mol: float = CROSS_CHECK_MAX_DELTA_F_KJ_MOL,
) -> float:
    """两条路径的**全部**目标态间 ΔF 必须一致。差异超限即报错，不静默采用。"""
    a = np.asarray(collapsed_target_delta_f_kJ_mol, dtype=np.float64)
    b = np.asarray(reference_target_delta_f_kJ_mol, dtype=np.float64)
    if a.shape != b.shape:
        raise MultiSegmentInputError(f"对账形状不一致：{a.shape} vs {b.shape}")
    worst = float(np.max(np.abs(a - b))) if a.size else 0.0
    if not np.isfinite(worst) or worst > float(tol_kJ_mol):
        raise MultiSegmentSolveError(
            f"塌缩路径与 S+K MBAR 的目标态间 ΔF 最大差 {worst:.3e} kJ/mol "
            f"> {tol_kJ_mol:.1e}；拒绝把塌缩结果当等价物使用"
        )
    return worst


def merged_coverage_diagnostics(
    responsibility: np.ndarray,
    sampling_kj: np.ndarray,
    frozen_f_by_active_segment: Sequence[Sequence[float]],
    kt: float,
) -> Dict[str, Any]:
    """合并采样分布的覆盖度/占据。

    每帧先算它属于各采样分布的后验概率 ``r_s(n)``（就是塌缩里那个分母的分项），
    再按 ``p_k^mix(n) = Σ_s r_s(n) · p_{k|s}(n)`` 混合各段的 λ 成员概率。
    ``p_{k|s}`` 用该段**冻结的 f_k** 和**采样态能量**算 —— 与
    `_ibs_reweighting_quality_diagnostics` 的 ``p_k ∝ exp(-(S_k-f_k)/kT)`` 同口径。

    ⚠️ 逐段最差只作诊断落盘，不当合并门：合并的 ESS 既不保证严格相加、也不保证
    单调增加，拿"最差段的绝对 ESS"配"两段总帧数"当比值更是分子分母不匹配。
    """
    r = np.asarray(responsibility, dtype=np.float64)
    s = np.asarray(sampling_kj, dtype=np.float64)
    if r.shape[1] != s.shape[1]:
        raise MultiSegmentInputError(f"帧数不一致：r={r.shape} sampling={s.shape}")
    if r.shape[0] != len(frozen_f_by_active_segment):
        raise MultiSegmentInputError(
            f"有效段数不一致：r={r.shape[0]} f_k={len(frozen_f_by_active_segment)}"
        )
    n_states, n_frames = s.shape
    p_mix = np.zeros((n_states, n_frames), dtype=np.float64)
    for i, f in enumerate(frozen_f_by_active_segment):
        f = np.asarray(f, dtype=np.float64).ravel()
        log_p = -(s - f[:, None]) / float(kt)
        log_p = log_p - _logsumexp(log_p, axis=0)[None, :]
        p_mix += r[i][None, :] * np.exp(log_p)

    sums = np.sum(p_mix, axis=1)
    sq = np.sum(p_mix ** 2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ess = np.where(sq > 0.0, sums ** 2 / sq, 0.0)
    return {
        "coverage_ess": [float(x) for x in ess],
        "coverage_ess_ratio": [float(x) / float(n_frames) for x in ess],
        "occupancy_normalized": [
            float(n_states * m) for m in np.mean(p_mix, axis=1)
        ],
        "n_frames": int(n_frames),
        "source": "merged_mixture",
    }
