def _pilot_segment_lengths(pilot_lambdas, metric_g) -> np.ndarray:
    """相邻 pilot 点之间的热力学长度 ``0.5*(sqrt(g_i)+sqrt(g_{i+1}))*|dλ|``。

    从 `_refine_pilot_grid_in_steep_segments` 里抽出来的共享实现（原来那里
    是内联重复代码），数值行为不变；`classify_pilot_point_risk_zone` 也用它
    判断"当前最长热力学区间"。
    """
    sqrt_g = np.sqrt(np.clip(np.asarray(metric_g, dtype=float), 1.0e-12, None))
    lam = np.asarray(pilot_lambdas, dtype=float)
    if lam.size < 2:
        return np.zeros(0, dtype=float)
    return 0.5 * (sqrt_g[:-1] + sqrt_g[1:]) * np.abs(np.diff(lam))


def pilot_block_running_diagnostics(
    values: np.ndarray, temperature_K: float
) -> Dict[str, Any]:
    """给定某个 pilot 点截至目前采到的 dU/dlambda 样本，算一组"假想现在停
    下"的诊断量。纯数值，不抛异常——样本太少时相应字段退化成 NaN，由调用方
    按 ``n_samples`` 自己决定要不要信。

    Returns
    -------
    dict：``n_samples``、``mean_dU_dlambda_kJ_mol``、``std_dU_dlambda_kJ_mol``、
    ``sem_kJ_mol``、``metric_g``（beta^2*Var）、``excess_kurtosis``（超额峰
    度，>0 说明比正态分布更厚尾，可能是还没等到的稀有事件的早期信号）、
    ``max_abs_robust_zscore``（基于 MAD 的稳健 z 分数最大绝对值，抓单个突发
    异常值，不像普通 z 分数那样会被该值自己拉高的标准差稀释）。
    """
    values = np.asarray(values, dtype=float).ravel()
    n = int(values.size)
    out: Dict[str, Any] = {"n_samples": n}
    if n < 2:
        out.update(
            mean_dU_dlambda_kJ_mol=float("nan"),
            std_dU_dlambda_kJ_mol=float("nan"),
            sem_kJ_mol=float("nan"),
            metric_g=float("nan"),
            excess_kurtosis=float("nan"),
            max_abs_robust_zscore=float("nan"),
        )
        return out

    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    beta = 1.0 / (0.008314462618 * float(temperature_K))
    out["mean_dU_dlambda_kJ_mol"] = mean
    out["std_dU_dlambda_kJ_mol"] = std
    out["sem_kJ_mol"] = float(std / np.sqrt(n))
    out["metric_g"] = float(beta * beta * std * std)

    if n >= 4 and std > 0.0:
        out["excess_kurtosis"] = float(np.mean((values - mean) ** 4) / std**4 - 3.0)
    else:
        out["excess_kurtosis"] = float("nan")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0.0:
        robust_z = 0.6745 * (values - median) / mad
        out["max_abs_robust_zscore"] = float(np.max(np.abs(robust_z)))
    else:
        out["max_abs_robust_zscore"] = 0.0
    return out


def classify_pilot_point_risk_zone(
    pilot_lambdas,
    metric_g,
    is_refinement_point,
    lambda_near_one_floor: float = 0.875,
) -> List[str]:
    """对每个 pilot 点标 "risk" / "easy"，纯事后打标签，不影响任何真实采样。

    风险判据（跟用户敲定的设计一一对应）：
      - 加密点（``is_refinement_point[i]`` 为 True，来自
        `_refine_pilot_grid_in_steep_segments`）——插入的理由本来就是父区间
        空间信息不足，继承父区间风险，不因为是"额外点"缩短预算。
      - λ ≥ ``lambda_near_one_floor``（默认 0.875，覆盖
        `human_vanishing_initial_lambdas` 17 点网格里 λ=1.0 起最前两段）——
        07-19 那次真实回归的端点区域。
      - 当前最长热力学区间（`_pilot_segment_lengths` 最大值，允许并列）的两
        个端点。

    其余点标 "easy"。数组长度不一致时整体退化成全 "risk"（宁可保守，不猜）。
    """
    lam = np.asarray(pilot_lambdas, dtype=float).ravel()
    g = np.asarray(metric_g, dtype=float).ravel()
    refine_flags = list(is_refinement_point)
    n = int(lam.size)
    if not (n == g.size == len(refine_flags)) or n < 2:
        return ["risk"] * max(n, 0)

    tags = ["easy"] * n
    for i in range(n):
        if bool(refine_flags[i]):
            tags[i] = "risk"
        elif lam[i] >= float(lambda_near_one_floor):
            tags[i] = "risk"

    seg_lengths = _pilot_segment_lengths(lam, g)
    if seg_lengths.size:
        worst = float(np.max(seg_lengths))
        for i, length in enumerate(seg_lengths):
            if length >= worst - 1.0e-12 * max(worst, 1.0):
                tags[i] = "risk"
                tags[i + 1] = "risk"
    return tags


def pilot_early_stop_pressure_test(
    pilot_lambdas,
    final_metric_g,
    point_index: int,
    checkpoint_metric_g: float,
    worst_case_inflation_ratio: float = 3.0,
    max_allowed_lambda_shift: float = 0.01,
    # 🔑 [0831issue P2] 默认值改成 None、在函数体内再读模块常量。
    # 默认参数在**函数定义时**求值一次，所以写成
    # `= VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS` 会把常量当时的值永久焊进签名；
    # 该常量历史上经过 2→6→4 的演进，而校验方 `redistribute_vanishing_lambda_subdomains`
    # 读的是**当前**全局值——两者会静默失配，压力测试基线与生产布点契约就对不上了。
    first_ensemble_target_intervals: Optional[int] = None,
) -> Dict[str, Any]:
    """压力测试：如果 ``point_index`` 这个点在某个 checkpoint 就已经拿到了
    ``checkpoint_metric_g``（而不是跑满 30000 步后的真实
    ``final_metric_g[point_index]``），production λ 布点会挪动多少；再把这
    个 checkpoint 估计按 ``worst_case_inflation_ratio`` 向上膨胀重算一次，两
    次位移都要低于 ``max_allowed_lambda_shift`` 才算通过压力测试。

    🔑 ``worst_case_inflation_ratio`` 默认值 3.0 是占位符，不是已验证的数
    字——本函数落地时仓库里还没有真实的 shadow 数据；Phase B 拿到真机 30000
    步的 checkpoint 序列、反推出真实的"部分估计 vs 最终估计"比值分布之后，
    必须回填一个有实测依据的值，调用方不应该信任这个默认值本身代表任何安全
    边际。

    永不抛异常：`redistribute_vanishing_lambda_subdomains` 失败（输入不满足
    不变量等）时返回 ``{"valid": False, "reason": ...}``——这是离线诊断函
    数，不能让分析脚本因为一次异常输入就整体崩溃。
    """
    # [0831issue P2] None → 此刻读模块常量的当前值，跟校验方
    # redistribute_vanishing_lambda_subdomains 用同一个来源，不会被定义期快照冻住。
    if first_ensemble_target_intervals is None:
        first_ensemble_target_intervals = VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS
    try:
        lam = np.asarray(pilot_lambdas, dtype=float)
        g_final = np.asarray(final_metric_g, dtype=float)
        if not (0 <= int(point_index) < lam.size) or lam.size != g_final.size:
            return {"valid": False, "reason": "bad_point_index_or_shape_mismatch"}

        baseline_lambdas, *_ = redistribute_vanishing_lambda_subdomains(
            lam, g_final, VANISHING_PROBE_BASE_STATE_COUNT,
            first_ensemble_target_intervals=first_ensemble_target_intervals,
        )

        def _shift_for(substitute_metric_g: float) -> float:
            g_mod = g_final.copy()
            g_mod[int(point_index)] = float(substitute_metric_g)
            candidate_lambdas, *_ = redistribute_vanishing_lambda_subdomains(
                lam, g_mod, VANISHING_PROBE_BASE_STATE_COUNT,
                first_ensemble_target_intervals=first_ensemble_target_intervals,
            )
            return float(np.max(np.abs(candidate_lambdas - baseline_lambdas)))

        raw_shift = _shift_for(checkpoint_metric_g)
        inflated_shift = _shift_for(
            float(checkpoint_metric_g) * float(worst_case_inflation_ratio)
        )
        passes = (
            raw_shift <= max_allowed_lambda_shift
            and inflated_shift <= max_allowed_lambda_shift
        )
        return {
            "valid": True,
            "raw_lambda_shift": raw_shift,
            "inflated_lambda_shift": inflated_shift,
            "max_allowed_lambda_shift": float(max_allowed_lambda_shift),
            "worst_case_inflation_ratio": float(worst_case_inflation_ratio),
            "would_pass_pressure_test": bool(passes),
        }
    except Exception as e:  # noqa: BLE001 -- 离线诊断，fail-closed 不能崩调用方
        return {"valid": False, "reason": f"redistribute_failed: {e}"}


