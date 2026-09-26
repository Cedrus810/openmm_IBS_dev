# ---------------------------------------------------------------------------
# split-half 时序一致性诊断
# ---------------------------------------------------------------------------
# 现有五道门（overlap / occupancy / decorrelated-samples / endpoint-σ / ESS）
# 全部是对整批样本的统计量：把帧的时间顺序打乱，结果一模一样。所以它们对
# "系综还在漂" 这件事完全没有分辨力。2026-07-28 的溶剂盒扫描实测：18 个窗口里
# 有 5 个的前后半程 ΔG 之差超过自身报出 σ 的 2 倍（最大 4.34），而
# `min_decorrelated_samples` 的阈值是 20、实测 137~1868，永远不可能响。
#
# 判据：把每个窗口的帧按时间切成前后两半，各自解一遍。两个半程各自的标准误
# 约 √2·σ，它们之差的标准误约 2σ，所以 z = |后半 − 前半| / (2σ_win)。
#
# 默认只诊断不阻断（`split_half_max_z=None`）：把它直接设成阻断会让现有大量
# 运行当场失败，那是采样预算问题，不该由一次代码改动替使用者决定。要变成硬门，
# 显式传 `split_half_max_z=2.0`。
SPLIT_HALF_GATE_PROTOCOL_VERSION = 1
SPLIT_HALF_DEFAULT_MAX_Z = 2.0


def _slice_window_frames(window: Dict, lo_frac: float, hi_frac: float) -> Optional[Dict]:
    """按时间顺序取窗口的 [lo_frac, hi_frac) 段帧；帧数不足返回 None。"""
    u_kn = window.get("u_kn")
    if u_kn is None:
        return None
    u_kn = np.asarray(u_kn, dtype=np.float64)
    if u_kn.ndim != 2:
        return None
    n_frames = u_kn.shape[1]

    # 🔑 [多采样段] 多段窗口的帧是 段1|段2 拼接的，直接取连续前半会几乎全是段 1、
    # 一帧段 2 都没有 —— 那不是"前半程"，是"段 1"。必须**在每段内部**各取前/后半，
    # 再重新拼接并同步重建来源标签。每半程之后会各自去相关、各自重新塌缩，
    # 绝不继承全程的混合权重（各半的段计数不同，权重也不同）。
    _sid = window.get("sampling_source_id")
    if _sid is not None:
        _sid = np.asarray(_sid, dtype=np.int64).ravel()
        if _sid.size != n_frames:
            return None
        _take = []
        for _s in np.unique(_sid):
            _pos = np.flatnonzero(_sid == _s)
            _n = _pos.size
            _a = max(0, min(int(lo_frac * _n), _n))
            _b = max(_a, min(int(hi_frac * _n), _n))
            if _b > _a:
                _take.append(_pos[_a:_b])
        if not _take:
            return None
        _idx = np.sort(np.concatenate(_take))
        if _idx.size < 2:
            return None
        sliced = dict(window)
        sliced["u_kn"] = u_kn[:, _idx]
        for _key in ("bias_energies", "base_energies"):
            _v = window.get(_key)
            if _v is None:
                return None
            sliced[_key] = np.asarray(_v, dtype=np.float64)[_idx]
        _ss = window.get("sampling_state_energies")
        if _ss is None:
            return None          # 多段必须有采样规范能量，否则塌缩无从做起
        _ss = np.asarray(_ss, dtype=np.float64)
        if _ss.ndim != 2 or _ss.shape[1] != n_frames:
            return None
        sliced["sampling_state_energies"] = _ss[:, _idx]
        sliced["sampling_source_id"] = _sid[_idx]
        # 逐窗口循环会按新的来源标签重建去相关分段；旧的 production_segments
        # 帧号已经失效，留着只会让校验器误判。
        sliced.pop("production_segments", None)
        return sliced

    lo = max(0, min(int(lo_frac * n_frames), n_frames))
    hi = max(lo, min(int(hi_frac * n_frames), n_frames))
    if hi - lo < 2:
        return None
    bias = window.get("bias_energies")
    base = window.get("base_energies")
    if bias is None or base is None:
        return None
    sliced = dict(window)
    sliced["u_kn"] = u_kn[:, lo:hi]
    sliced["bias_energies"] = np.asarray(bias, dtype=np.float64)[lo:hi]
    sliced["base_energies"] = np.asarray(base, dtype=np.float64)[lo:hi]
    # 🔑 [2026-09-09] residual 臂上 loader 会带一个 (K, N) 的
    # `sampling_state_energies`。原来 `dict(window)` 把它整份抄过来却不切，于是半窗
    # 携带全长 (K, N) 去对 (K, N/2) 的 u_kn，`solve_stage_integrated` 抛形状错，
    # 再被 `split_half_drift_diagnostics` 外层的裸 `except Exception` 吞成
    # `available=False` —— **整条 residual 臂的 split-half 漂移诊断永久是暗的**，
    # 而那是本文件里唯一对帧时序敏感的门。fail-open，不是 fail-closed。
    sampling_states = window.get("sampling_state_energies")
    if sampling_states is not None:
        sampling_states = np.asarray(sampling_states, dtype=np.float64)
        if sampling_states.ndim == 2 and sampling_states.shape[1] == n_frames:
            sliced["sampling_state_energies"] = sampling_states[:, lo:hi]
        else:
            # 形状本来就对不上 u_kn —— 这不是切片能修的，交给上游去发现，
            # 不要在这里悄悄塞一个半对的数组。
            return None
    if "production_segments" in window:
        _validate_production_segments(window["production_segments"], n_frames)
        sliced["production_segments"] = [
            dict(segment, start_frame=max(segment["start_frame"], lo)-lo,
                 end_frame=min(segment["end_frame"], hi)-lo,
                 n_frames=min(segment["end_frame"], hi)-max(segment["start_frame"], lo))
            for segment in window["production_segments"]
            if max(segment["start_frame"], lo) < min(segment["end_frame"], hi)
        ]
    return sliced


def _segment_delta_g_by_window(result: Dict) -> Dict[int, float]:
    segs = result.get("covariance_chain_segments") or []
    return {int(s["window_index"]): float(s["delta_G_kJ_mol"]) for s in segs}


def split_half_drift_diagnostics(
    window_outputs: List[Dict],
    kt: float,
    full_result: Dict,
    solver_kwargs: Optional[Dict] = None,
) -> Dict:
    """前后半程一致性诊断。只读，不改 `full_result`。"""
    solver_kwargs = dict(solver_kwargs or {})
    halves: Dict[str, Optional[Dict]] = {}
    for label, (lo, hi) in (("first", (0.0, 0.5)), ("second", (0.5, 1.0))):
        sliced = [_slice_window_frames(w, lo, hi) for w in window_outputs]
        if any(s is None for s in sliced):
            return {
                "split_half_gate_protocol_version": int(SPLIT_HALF_GATE_PROTOCOL_VERSION),
                "available": False,
                "reason": f"{label} half: 有窗口帧数不足以切半",
            }
        halves[label] = solve_stage_integrated(
            sliced, kt, split_half_max_z=None, _split_half_recursion_guard=True,
            **solver_kwargs,
        )

    seg_full = _segment_delta_g_by_window(full_result)
    seg_first = _segment_delta_g_by_window(halves["first"])
    seg_second = _segment_delta_g_by_window(halves["second"])
    sigma_win = {
        int(s["window_index"]): float(s["uncertainty_kJ_mol"])
        for s in (full_result.get("covariance_chain_segments") or [])
    }
    # 🔑 [2026-09-18] 半程窗口集合与全量不一致时，原来**整段**返回 available=False。
    # 实测后果（09-18 benchmark 24 run）：complex 腿 20 条里 12 条的漂移诊断完全是
    # 暗的，溶剂腿 0/22 —— 而且失败的恰好是最弱窗口已经贴/破样本下限的那些
    # （`min_decorrelated_samples` 中位 18.0 vs 可用组 26.5，12 条里 6 条在**全量**
    # 数据上就已经低于门 20）。半程帧数砍半 ⟹ 那个边缘窗口被 solver `continue`
    # 掉 ⟹ 集合不一致 ⟹ 整段关闭。**与"最需要漂移检验"反相关，而且是构造出来的
    # fail-open。** 见 docs/archive/STAGE2_OFFLINE_FORENSICS_2026-09-18.md §9。
    #
    # 改法：逐窗比较取**交集**，缺的窗口如实记名。
    # `sigma_inflated_from_split_half` 早就逐窗处理"这个窗口没有漂移证据"
    # （`sigma_floor_unavailable`），所以交集结果可以直接喂给它，缺的窗口照旧
    # 不膨胀、并在名单里可见 —— 数值语义不变，只是不再把**有证据的窗口**
    # 一起丢掉。
    #
    # ⚠️ 总量那几项**不能**这么救：窗口集合不同 ⟹ 两个半程的 `total_delta_G`
    # 积的不是同一段 λ，相减没有物理意义。集合不全时一律置 None（**不是 0**）。
    common = sorted(set(seg_full) & set(seg_first) & set(seg_second))
    missing_from_halves = sorted(set(seg_full) - set(common))
    if not common:
        return {
            "split_half_gate_protocol_version": int(SPLIT_HALF_GATE_PROTOCOL_VERSION),
            "available": False,
            "reason": "两个半程与全量没有任何共同窗口，无法逐窗比较",
            "windows_missing_from_halves": [int(w) for w in sorted(set(seg_full))],
        }

    per_window = []
    max_z = 0.0
    for w in common:
        drift = seg_second[w] - seg_first[w]
        sigma = sigma_win.get(w, 0.0)
        # σ_win 为 0 时 z 无意义（不是"完美"），记 None 而不是 inf。
        z = abs(drift) / (2.0 * sigma) if sigma > 0.0 else None
        if z is not None:
            max_z = max(max_z, z)
        per_window.append({
            "window_index": int(w),
            "delta_G_full_kJ_mol": seg_full[w],
            "delta_G_first_half_kJ_mol": seg_first[w],
            "delta_G_second_half_kJ_mol": seg_second[w],
            "drift_kJ_mol": drift,
            "uncertainty_kJ_mol": sigma,
            "drift_over_2sigma": z,
        })

    total_sigma = float(full_result.get("total_error", 0.0) or 0.0)
    coverage_complete = not missing_from_halves
    # 集合不全 ⟹ 两个半程积的不是同一段 λ，总量相减无物理意义，置 None。
    total_first = (
        float(halves["first"].get("total_delta_G", 0.0)) if coverage_complete else None
    )
    total_second = (
        float(halves["second"].get("total_delta_G", 0.0)) if coverage_complete else None
    )
    total_drift = (
        total_second - total_first
        if (total_first is not None and total_second is not None) else None
    )
    return {
        "split_half_gate_protocol_version": int(SPLIT_HALF_GATE_PROTOCOL_VERSION),
        "available": True,
        "metric": "abs(second_half - first_half) / (2 * sigma)",
        "note": (
            "两个半程各自 SE≈√2·σ，其差的 SE≈2σ，所以判据分母是 2σ 而不是 σ。"
            "该诊断对帧的时间顺序敏感，是现有五道门都不具备的维度。"
        ),
        # [2026-09-18] 逐窗结果取的是「全量 ∩ 前半 ∩ 后半」。缺的窗口在
        # `windows_missing_from_halves` 里如实记名 —— 它们**没有**漂移证据，
        # 不是"实测漂移为 0"；下游 `sigma_inflated_from_split_half` 会把它们标成
        # `sigma_floor_unavailable` 并计入 `sigma_floor_coverage_complete`。
        "windows_missing_from_halves": [int(w) for w in missing_from_halves],
        "n_windows_compared": int(len(common)),
        "coverage_complete": bool(coverage_complete),
        "coverage_note": (
            "" if coverage_complete else
            "半程解出的窗口集合少于全量（帧数砍半后边缘窗口被求解器跳过）。"
            "逐窗漂移只覆盖交集；总量项因两半积的 λ 段不同而置 None，**不是 0**。"
        ),
        "total_delta_G_first_half_kJ_mol": total_first,
        "total_delta_G_second_half_kJ_mol": total_second,
        "total_drift_kJ_mol": total_drift,
        "total_drift_over_2sigma": (
            abs(total_drift) / (2.0 * total_sigma)
            if (total_drift is not None and total_sigma > 0.0) else None
        ),
        "max_window_drift_over_2sigma": max_z,
        "per_window": per_window,
    }


def sigma_inflated_from_split_half(full_result: Dict, drift: Dict) -> Dict:
    """[P1-19] 用 split-half 实测漂移给 per-window σ 定一个下界。

    `segment_error` 取的是 pymbar 的**渐近协方差**（`dDelta_f[join, end]`），
    它假定样本独立同分布且系综已收敛。本仓库已三次实测到它系统性低估：

      1. 溶剂盒扫描 18 个窗口里 5 个的 split-half z > 2（σ 正确时期望 0.8 个）；
         window 4 在所有现有指标上都是优等生（ESS 348、n_decorr 357、g 1.40、
         ess_ratio ≥ 0.976）却 z=4.34 —— 问题不在采样质量的代理量，在 σ 本身。
      2. attachment 腿两轮独立测量差 0.4454 kJ/mol = 报出 σ 的 4.4 倍。
      3. charging 重跑与原值差 10.63 kJ/mol 而报出 σ 只有 0.88。

    修法：两个半程之差的标准差是 2σ，所以观测到 |漂移| 就意味着
    `σ ≳ |漂移|/2`。取 `σ_eff = max(σ_MBAR, |漂移|/2)` 作为下界。
    这只用已经算出来的量，零额外计算。

    **只返回结果，不改 `full_result`。** 调用方决定要不要采用——因为
    `final_max_uncertainty_kJ_mol` 那道门吃的就是 max(σ_win)，直接套用会让
    一批本来通过的运行突然失败，那是使用者的决定而不是这个函数的。
    """
    segs = full_result.get("covariance_chain_segments") or []
    if not segs or not drift.get("available"):
        return {"available": False, "reason": "缺 covariance_chain_segments 或 split-half 不可用"}

    by_window = {int(w["window_index"]): w for w in drift.get("per_window", [])}
    rows, var_orig, var_infl = [], 0.0, 0.0
    windows_without_drift = []
    for seg in segs:
        w = int(seg["window_index"])
        s_mbar = float(seg["uncertainty_kJ_mol"])
        d = by_window.get(w, {}).get("drift_kJ_mol")
        # 🔑 [0831issue P2] `d is None` 表示这个窗口**没有 split-half 证据**，不表示
        # "实测漂移为 0"。以前它静默退化成 floor=0.0 → 该窗口 σ 维持 MBAR 渐近值、
        # 总 σ 被低估，而且落盘里完全看不出哪些窗口其实无证据。现在逐窗口标记，
        # 并在汇总里给出名单与计数；数值行为不变（仍然不膨胀），只是不再静默。
        drift_available = d is not None
        floor = abs(float(d)) / 2.0 if drift_available else 0.0
        if not drift_available:
            windows_without_drift.append(w)
        s_eff = max(s_mbar, floor)
        var_orig += s_mbar ** 2
        var_infl += s_eff ** 2
        rows.append({
            "window_index": w,
            "sigma_mbar_kJ_mol": s_mbar,
            "sigma_floor_from_drift_kJ_mol": floor,
            "sigma_floor_unavailable": bool(not drift_available),
            "sigma_effective_kJ_mol": s_eff,
            "inflated": bool(s_eff > s_mbar + 1e-12),
            "inflation_factor": (s_eff / s_mbar) if s_mbar > 0 else None,
        })
    return {
        "available": True,
        "rule": "sigma_eff = max(sigma_mbar, |split_half_drift| / 2)",
        "rationale": "两个半程之差的 SD 是 2σ，故观测到 |漂移| 蕴含 σ ≳ |漂移|/2",
        "per_window": rows,
        # [0831issue P2] 缺 split-half 证据的窗口：它们的 σ 没有被膨胀，**不是因为
        # 实测漂移小，而是因为没有实测**。总 σ 因此是一个下界，不是已校正值。
        "windows_with_sigma_floor_unavailable": [int(w) for w in windows_without_drift],
        "n_windows_with_sigma_floor_unavailable": int(len(windows_without_drift)),
        "sigma_floor_coverage_complete": bool(not windows_without_drift),
        "total_error_mbar_kJ_mol": float(np.sqrt(var_orig)),
        "total_error_inflated_kJ_mol": float(np.sqrt(var_infl)),
        "total_inflation_factor": (
            float(np.sqrt(var_infl) / np.sqrt(var_orig)) if var_orig > 0 else None
        ),
        "n_windows_inflated": int(sum(1 for r in rows if r["inflated"])),
        "max_endpoint_sigma_mbar_kJ_mol": max((r["sigma_mbar_kJ_mol"] for r in rows), default=None),
        "max_endpoint_sigma_inflated_kJ_mol": max((r["sigma_effective_kJ_mol"] for r in rows), default=None),
    }


def solve_stage_integrated(
    window_outputs: List[Dict],
    kt: float,
    stage_name: str = "",
    final_min_ess_ratio: float = 0.05,
    final_min_absolute_ess: float = 50.0,
    final_min_decorrelated_samples: int = 20,
    final_max_uncertainty_kJ_mol: float = 1.0,
    final_min_target_absolute_ess: float = TARGET_SUPPORT_MIN_ABSOLUTE_ESS,
    final_max_top1pct_raw_weight: float = TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT,
    min_frames_per_window: int = 10,
    split_half_max_z: Optional[float] = None,
    inflate_sigma_from_split_half: bool = False,
    _split_half_recursion_guard: bool = False,
    skip_split_half_diagnostics: bool = False,
) -> Dict:
    """stage 级 IBS-TMBAR 求解入口（stage2 = vdW）。

    ⛔ **只能用 TMBAR。** 不要在这里加 BAR / TI / 全帧主值 / √g σ 缩放 /
    bootstrap σ / σ evidence 汇总——理由与被撤回的先例见
    `ESTIMATOR_ANALYSIS_PROTOCOL_VERSION` 的注释。

    `skip_split_half_diagnostics`：[P1-19] split-half 前后半程诊断把
    `window_outputs` 的列表位置当成物理 λ 窗口编号（"window 0" = 列表第 0
    个元素）。整段生产/最终分析里两者重合，诊断有意义；但在线 TMBAR
    学习（`IBSSampler._solve_tmbar_and_recenter`）传入的是一个容量固定的
    滑动 minibatch 历史（`self.tmbar_history`），列表位置只是"当前还留在
    窗口里第几旧的 minibatch"，与任何物理 λ 窗口无关。这种调用方应该传
    `skip_split_half_diagnostics=True`，避免刷出"window 0 漂移 X"这种在
    该场景下没有实际信息量的告警（同一条 minibatch 在滑窗塞满前会被反复
    诊断出完全相同的数字），也省下这次多余的求解开销。见
    `docs/experiments/P1-19_ONLINE_SLIDING_WINDOW_SPLITHALF_MISMATCH.md`。
    """
    if not window_outputs:
        return {
            "total_delta_G": 0.0,
            "total_error": 999.9,
            "analysis_status": "ANALYSIS_INCOMPLETE",
            "analysis_incomplete_reasons": ["window_outputs 为空，没有任何窗口可积分。"],
            "precision_status": "UNMEASURED",
            "precision_evidence": cross_repeat_precision([]),
        }
    analyzer = GlobalMBARAnalyzer(kt=kt)
    solver_kwargs = dict(
        final_min_ess_ratio=final_min_ess_ratio,
        final_min_absolute_ess=final_min_absolute_ess,
        final_min_decorrelated_samples=final_min_decorrelated_samples,
        final_max_uncertainty_kJ_mol=final_max_uncertainty_kJ_mol,
        final_min_target_absolute_ess=final_min_target_absolute_ess,
        final_max_top1pct_raw_weight=final_max_top1pct_raw_weight,
        min_frames_per_window=min_frames_per_window,
    )
    res = analyzer.solve_stage_integrated(window_outputs, **solver_kwargs)
    res.setdefault("stage", stage_name)

    if not _split_half_recursion_guard and not skip_split_half_diagnostics:
        try:
            drift = split_half_drift_diagnostics(window_outputs, kt, res, solver_kwargs)
        except Exception as exc:  # 诊断绝不能把主求解带崩
            drift = {
                "split_half_gate_protocol_version": int(SPLIT_HALF_GATE_PROTOCOL_VERSION),
                "available": False,
                "reason": f"split-half 诊断异常: {exc}",
            }
        res["split_half_diagnostics"] = drift
        max_z = drift.get("max_window_drift_over_2sigma") if drift.get("available") else None
        res["split_half_max_window_z"] = max_z
        res["split_half_max_z_threshold"] = split_half_max_z
        # 🔑🔑 [2026-09-15] **总量必须和逐窗一起报出来。**
        #
        # `split_half_drift_diagnostics` 一直同时算 `total_drift_kJ_mol`，但先前
        # 只有逐窗那条进日志 ⟹ 唯一能区分下面两件事的数字**算了却没人看得见**：
        #   · 真漂移         —— 前后半程在算不同的数，总量也跟着动；
        #   · 归属重排       —— 总量稳，只是逐窗 segment ΔG 的拆分变了。
        # 后者在本仓是**预期行为**：整条路径是一个拼接的全局 MBAR 解，窗口不是
        # 独立可分的单元；帧砍一半、全局解重排，per-window 的归属自然会动。
        # 拿逐窗漂移当"σ 低估"的证据，在这种情形下会系统性高估
        # （`sigma_inflated_from_split_half` 吃的正是逐窗那个量）。
        # 真机 cyclod_ligand2/rep1：逐窗报 +9.339 kJ/mol，而总量那条从没打印过。
        _tot_z = drift.get("total_drift_over_2sigma") if drift.get("available") else None
        res["split_half_total_z"] = _tot_z
        res["split_half_total_drift_kJ_mol"] = (
            drift.get("total_drift_kJ_mol") if drift.get("available") else None
        )
        if max_z is not None and max_z > SPLIT_HALF_DEFAULT_MAX_Z:
            worst = max(
                (w for w in drift["per_window"] if w["drift_over_2sigma"] is not None),
                key=lambda w: w["drift_over_2sigma"],
            )
            print(
                f"  [WARN] [split-half] {stage_name or 'stage'} 前后半程不一致："
                f"window {worst['window_index']} 漂移 {worst['drift_kJ_mol']:+.3f} kJ/mol "
                f"= {worst['drift_over_2sigma']:.2f}×2σ（σ_win={worst['uncertainty_kJ_mol']:.3f}）。"
                f"报出的不确定度低估了实际抽样波动。"
            )
            # ⚠️ 触发判据**没变**，仍是逐窗 max_z（这一行只是把总量摆出来一起看）。
            print(
                # [2026-09-18] 窗口集合不全时总量项是 None（两半积的 λ 段不同，
                # 相减无意义）。不能再无条件 `:+.3f`，否则诊断自己把主求解打断。
                (
                    f"    ↳ [split-half 总量] 半程一 "
                    f"{drift.get('total_delta_G_first_half_kJ_mol'):+.3f} → "
                    f"半程二 {drift.get('total_delta_G_second_half_kJ_mol'):+.3f}，"
                    f"总漂移 {drift.get('total_drift_kJ_mol'):+.3f} kJ/mol"
                    + (f" = {_tot_z:.2f}×2σ" if _tot_z is not None
                       else "（σ_total=0，z 无意义）")
                    if drift.get("total_drift_kJ_mol") is not None else
                    f"    ↳ [split-half 总量] **不可比** —— 半程解出的窗口集合"
                    f"少于全量（缺 {drift.get('windows_missing_from_halves')}），"
                    f"两半积的 λ 段不同；逐窗漂移只覆盖交集"
                    f"（{drift.get('n_windows_compared')} 个窗口）"
                )
                + "。**总量稳而逐窗大 ⟹ 多半是归属重排,不是漂移**："
                "整条路径是一个拼接的全局 MBAR 解,窗口不是独立可分的单元,"
                "帧砍一半会让 per-window 的 segment ΔG 重新分配。"
                "两者都大才是真的前后半程不一致。"
            )
        if split_half_max_z is not None and max_z is not None and max_z > float(split_half_max_z):
            # 🔑 [2026-09-15] 只报告，不再改判。`converged` 已删除，而 split-half
            # 漂移是阈值判据（split_half_max_z），按定义不进 analysis_status
            # ——analysis_status 只收硬不变量。绝不要在这里 `res["converged"]=False`
            # 把那个键复活：那会造出一个「只在失败时出现」的幽灵键。
            res["split_half_gate_failed"] = True

        # [P1-19] σ 下界。**始终计算并落盘，默认不采用**——
        # `final_max_uncertainty_kJ_mol` 那道门吃的就是 max(σ_win)，
        # 直接套用会让一批本来通过的运行突然失败，那是使用者的决定。
        infl = sigma_inflated_from_split_half(res, drift)
        res["sigma_inflation_from_split_half"] = infl
        res["sigma_inflation_applied"] = False
        if infl.get("available") and infl.get("n_windows_inflated"):
            print(
                f"  [P1-19] {stage_name or 'stage'} 若按 σ≥|漂移|/2 定下界："
                f"{infl['n_windows_inflated']}/{len(infl['per_window'])} 个窗口的 σ 被抬高，"
                f"总 σ {infl['total_error_mbar_kJ_mol']:.4f} → "
                f"{infl['total_error_inflated_kJ_mol']:.4f} kJ/mol "
                f"(×{infl['total_inflation_factor']:.2f})。默认未采用。"
                # ⚠️ [2026-09-15] **这个下界吃的是逐窗漂移,不是总漂移。**
                # 整条路径是一个拼接的全局 MBAR 解 ⟹ 帧砍一半会让 per-window 的
                # segment ΔG 重新分配,即使总量纹丝不动。那种情形下这个 ×N 是
                # **归属重排的幅度**,不是 σ 被低估的证据,会系统性高估。
                # 判它到底是哪一种,看上面 `[split-half 总量]` 那行的总漂移。
                " ⚠️ 本下界取自**逐窗**漂移；总量见上一行 —— "
                "总量稳而逐窗大 ⟹ 是归属重排,这个 ×N 不构成 σ 被低估的证据。"
            )
        if inflate_sigma_from_split_half and infl.get("available"):
            res["total_error_mbar_only_kJ_mol"] = float(res.get("total_error", 0.0))
            res["total_error"] = float(infl["total_error_inflated_kJ_mol"])
            for seg in res.get("covariance_chain_segments") or []:
                row = next(
                    (r for r in infl["per_window"]
                     if r["window_index"] == int(seg["window_index"])),
                    None,
                )
                if row:
                    seg["uncertainty_mbar_kJ_mol"] = seg["uncertainty_kJ_mol"]
                    seg["uncertainty_kJ_mol"] = row["sigma_effective_kJ_mol"]
            # ---- [P1-23] 抬高了 σ，端点 σ 门与 converged 必须一起重算 ----
            #
            # 原先只替换 `total_error` 与逐段 `uncertainty_kJ_mol`，却**不动**
            # `max_endpoint_uncertainty_kJ_mol`、也不重判 `converged` ——
            # 等于"σ 抬上去了，而门还在读抬高前的小 σ"。σ 抬高的**全部意义**就是
            # 让门看见真实的不确定度；门读旧值就是 fail-open：一个本该被端点 σ 门
            # 拦下的 stage 会带着 `converged=True` 通过。
            #
            # 这里只重算「端点 σ」这一项并据此重判，不碰 overlap / 独立样本数
            # 那两项（它们与 σ 口径正交，值没变）。原始值保留成
            # `*_mbar_only_*`，两套口径都在报告里可查。
            _seg_unc = [
                float(seg["uncertainty_kJ_mol"])
                for seg in (res.get("covariance_chain_segments") or [])
                if seg.get("uncertainty_kJ_mol") is not None
            ]
            if _seg_unc:
                _old_max = res.get("max_endpoint_uncertainty_kJ_mol")
                _new_max = float(np.max(_seg_unc))
                _thr = res.get("max_endpoint_uncertainty_kJ_mol_threshold")
                res["max_endpoint_uncertainty_kJ_mol_mbar_only"] = _old_max
                res["max_endpoint_uncertainty_kJ_mol"] = _new_max
                if _thr is not None and np.isfinite(_new_max):
                    _endpoint_ok = _meets_maximum_with_roundoff(_new_max, float(_thr))
                    # 🔑 [2026-09-15] 只报告，不再改判。这道门吃的就是被降级的
                    # `max_endpoint_uncertainty_kJ_mol`（1.0 kJ/mol 那个无依据的
                    # 魔法数），它不再是接受判据，也不得复活 `converged` 键。
                    res["max_endpoint_uncertainty_gate_passed"] = bool(_endpoint_ok)
                    if not _endpoint_ok:
                        print(
                            f"  [WARN] [P1-23] σ 抬高后端点 σ = {_new_max:.4f} kJ/mol "
                            f"超过**报告用**参考线 {float(_thr):.4f}（诊断，不改判）"
                        )
                if _old_max is not None:
                    print(
                        f"  [P1-23] 端点 σ 随之更新：{float(_old_max):.4f} → "
                        f"{_new_max:.4f} kJ/mol"
                    )
            # 🔑 [0831issue P2] 逐段 `uncertainty_kJ_mol`、`total_error`、
            # `max_endpoint_uncertainty_kJ_mol` 三项已按抬高后的 σ 重写，但
            # `df_k`（逐态误差曲线）与 `endpoint_error_after_offset` 仍是 MBAR-only 值。
            # **刻意不改它们的数值**：把逐窗口的 σ 下界映射到逐态 df_k 不是简单的
            # 对角缩放（MBAR 协方差有非对角结构），随手乘个因子等于编一个误差棒。
            # 但也不能让两套口径并存而只有一处标注 —— 下游读 df_k 会以为它是抬高后的值。
            # 这里显式标注这两个字段的口径，并把 MBAR-only 副本命名清楚。
            res["sigma_inflation_scope"] = {
                "inflated_fields": [
                    "total_error",
                    "covariance_chain_segments[].uncertainty_kJ_mol",
                    "max_endpoint_uncertainty_kJ_mol",
                ],
                "not_inflated_fields": ["df_k", "endpoint_error_after_offset"],
                "reason": (
                    "逐窗口 σ 下界无法无歧义地映射到逐态 df_k（MBAR 协方差非对角），"
                    "按下界重写会编造误差棒；这两个字段保持 MBAR 渐近口径。"
                ),
                "consumer_note": (
                    "需要抬高后的口径请读 total_error / covariance_chain_segments；"
                    "df_k 与 endpoint_error_after_offset 是 MBAR-only，不可与前者混用。"
                ),
            }
            if "df_k" in res:
                res["df_k_gauge"] = "mbar_asymptotic_only"
            if "endpoint_error_after_offset" in res:
                res["endpoint_error_after_offset_gauge"] = "mbar_asymptotic_only"
            res["sigma_inflation_applied"] = True
            print(
                f"  [OK] [P1-19] 已采用 σ 下界：总 σ = "
                f"{res['total_error']:.4f} kJ/mol"
                "（df_k / endpoint_error_after_offset 仍为 MBAR-only 口径，见 "
                "sigma_inflation_scope）"
            )

    return res

