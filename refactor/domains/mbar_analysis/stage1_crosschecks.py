def stage1_estimator_crosschecks(
    u_kn: np.ndarray,
    n_k: np.ndarray,
    kt: float,
    lambdas: Optional[Sequence[float]] = None,
) -> Dict:
    """[v2] 每态都有样本的腿（stage1 decharging / attachment）的四口径规范表。

    `u_kn` 必须是**约化**势（无量纲）。返回 kJ/mol。

    四个口径：去相关 MBAR / 全帧 MBAR / 相邻 BAR / 重加权 FD-TI。
    2026-07-28 stage1 实测：BAR 65.076 ± 0.615、FD-TI 65.126、全帧 MBAR 65.003
    三者一致，**去相关 MBAR 64.411 两条腿共同偏低 0.5–0.6 kJ/mol**——那是
    自相关子采样在有限样本下选帧不稳定造成的，不是 MBAR 有偏。
    所以主值取相邻 BAR、TI 作一致性门、两个 MBAR 降级为诊断。

    ⚠️ 两腿的偏移基本抵消：换估计量后 charging 对 ΔG_bind 的净移动只有约
    **−0.140 kJ/mol = −0.033 kcal/mol**。它是个真问题，但**不是** `result.txt`
    那 1.3 kcal charging 缺口的解释，别把两件事混起来。
    """
    u_kn = np.asarray(u_kn, dtype=np.float64)
    n_k = np.asarray(n_k, dtype=int)
    out: Dict[str, Any] = {
        "estimator_analysis_protocol_version": int(ESTIMATOR_ANALYSIS_PROTOCOL_VERSION),
        "u_kn_units": "reduced (dimensionless) — 转 kJ/mol 需乘 kT",
        "n_states": int(len(n_k)),
        "n_frames_total": int(np.sum(n_k)),
    }

    def _mbar_dg(sub_u, sub_n):
        stable = sub_u - np.min(sub_u, axis=0, keepdims=True)
        m = _build_mbar_compatible(
            stable, sub_n, solver_protocol="default", initialize="BAR",
            relative_tolerance=1e-6, verbose=False,
        )
        res = _compute_free_energy_result_compatible(m, compute_uncertainty=True)
        df, ddf = _extract_free_energy_arrays(res, require_uncertainty=True)
        return float((df[0, -1] - df[0, 0]) * kt), float(ddf[0, -1] * kt), m

    # --- 全帧 MBAR（诊断） ---
    try:
        dg_full, err_full, mbar_full = _mbar_dg(u_kn, n_k)
        out["full_frame_mbar"] = {"delta_G_kJ_mol": dg_full, "error_kJ_mol": err_full}
    except Exception as exc:
        mbar_full = None
        out["full_frame_mbar"] = {"delta_G_kJ_mol": None, "error": str(exc)}

    # --- 去相关 MBAR（诊断） ---
    try:
        offsets = np.concatenate(([0], np.cumsum(n_k)))
        keep, n_k_sub, g_per_state = [], np.zeros_like(n_k), []
        for k in range(len(n_k)):
            s, e = int(offsets[k]), int(offsets[k + 1])
            if e <= s:
                g_per_state.append(1.0)
                continue
            idx_k, g_k = subsample_series_by_autocorrelation(u_kn[k, s:e])
            g_per_state.append(float(g_k))
            keep.append(s + idx_k)
            n_k_sub[k] = idx_k.size
        keep_arr = np.concatenate(keep)
        dg_dec, err_dec, _ = _mbar_dg(u_kn[:, keep_arr], n_k_sub)
        out["decorrelated_mbar"] = {
            "delta_G_kJ_mol": dg_dec,
            "error_kJ_mol": err_dec,
            "statistical_inefficiency_per_state": g_per_state,
            "n_frames_after": int(keep_arr.size),
        }
    except Exception as exc:
        out["decorrelated_mbar"] = {"delta_G_kJ_mol": None, "error": str(exc)}

    # --- 相邻 BAR（主值候选） ---
    try:
        dg_bar, err_bar, edges = adjacent_bar_chain(u_kn, n_k, kt)
        out["adjacent_bar"] = {
            "delta_G_kJ_mol": dg_bar, "error_kJ_mol": err_bar, "edges": edges,
        }
    except Exception as exc:
        out["adjacent_bar"] = {"delta_G_kJ_mol": None, "not_applicable_reason": str(exc)}

    # --- 重加权 FD-TI（一致性门） ---
    if lambdas is None:
        out["reweighted_fd_ti"] = {
            "delta_G_kJ_mol": None,
            "not_applicable_reason": "未提供 lambdas；FD-TI 必须用真实非均匀 λ 网格，不得用态序号代替",
        }
    else:
        try:
            weights = getattr(mbar_full, "W_nk", None) if mbar_full is not None else None
            dg_ti, dudl = reweighted_fd_ti(u_kn, n_k, lambdas, kt, weights=weights)
            out["reweighted_fd_ti"] = {
                "delta_G_kJ_mol": dg_ti,
                "mean_dU_dlambda_kJ_mol": dudl,
                "lambdas": [float(x) for x in lambdas],
                "weights": "mbar_W_nk" if weights is not None else "per_state_block_mean",
            }
        except Exception as exc:
            out["reweighted_fd_ti"] = {"delta_G_kJ_mol": None, "error": str(exc)}

    return out


def stage1_ti_consistency_gate(
    dg_bar: float,
    err_bar: float,
    dg_ti: Optional[float],
    tolerance_kJ_mol: float,
) -> Dict:
    """BAR vs 重加权 FD-TI 的一致性门。

    ⚠️ 容差**必须显式给**，不得沿用 attachment 那个
    `ATTACHMENT_BAR_TI_ABS_TOL_KJ = 1.0`：两条腿的量级和 TI 变体都不同。
    stage1 实测 |BAR − FD-TI| = 0.050（complex）/ 0.226（solvent）kJ/mol。
    """
    if dg_ti is None:
        return {"passed": None, "reason": "没有 TI 值，无法判定（不当作通过）"}
    tol = max(float(tolerance_kJ_mol), 3.0 * max(float(err_bar), 0.0))
    diff = abs(float(dg_bar) - float(dg_ti))
    return {
        "passed": bool(diff <= tol),
        "abs_diff_kJ_mol": diff,
        "tolerance_kJ_mol": tol,
        "tolerance_source": "max(显式容差, 3*sigma_BAR)",
        "delta_G_bar_kJ_mol": float(dg_bar),
        "delta_G_ti_kJ_mol": float(dg_ti),
    }


