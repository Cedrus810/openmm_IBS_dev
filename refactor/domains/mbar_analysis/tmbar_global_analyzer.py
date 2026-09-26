# ============================================================================
# 4. 全局 TMBAR 分析器 (集成 bar.py 修复版逻辑)
# ============================================================================
class GlobalMBARAnalyzer:
    """
    严格 IBS-TMBAR 实现 (JCTC 2026 Sec 2.3)。
    集成 bar.py 修复版逻辑：
    1. 全局能量偏移 (Global Offset) 防止 exp 下溢/上溢。
    2. 增广矩阵构建 (Augmented Matrix) 正确处理偏置采样分布。
    3. 列平移稳定性 (Column-wise Shift) 确保 pymbar 数值收敛。
    
    输入: window_data = [{u_kn: (K_local, N), bias_energies: (N,), base_energies: (N,), lambda_indices: [...]}]
    输出: 全局自由能曲线与误差。
    """
    def __init__(self, kt: float):
        self.kt = kt
        self.beta = 1.0 / kt

    def solve_stage_integrated(
        self,
        window_data: List[Dict],
        final_min_ess_ratio: float = 0.05,
        final_min_absolute_ess: float = 50.0,
        final_min_decorrelated_samples: int = 20,
        final_max_uncertainty_kJ_mol: float = 1.0,
        final_min_target_absolute_ess: float = TARGET_SUPPORT_MIN_ABSOLUTE_ESS,
        final_max_top1pct_raw_weight: float = TARGET_SUPPORT_MAX_TOP1PCT_WEIGHT,
        min_frames_per_window: int = 10,
    ) -> Dict:
        """局部 TMBAR + 自洽拼接

        ⛔ **这条腿（stage2 = vdW）只能用 TMBAR。** 不要在这里加 BAR、TI、全帧主值、
        √g σ 缩放或 bootstrap σ ——理由与被撤回的先例都写在
        `ESTIMATOR_ANALYSIS_PROTOCOL_VERSION` 的注释里。

        🔑🔑 [2026-09-15 用户拍板] **`converged` 这个键已被删除**，换成两个正交状态：

            analysis_status ∈ {ANALYSIS_COMPLETE, ANALYSIS_INCOMPLETE}
                "求解器有没有在一条完整的路径上解出有限的数" —— **只由不依赖任何
                拟合阈值的硬不变量决定**（见下）。
            precision_status ∈ {UNMEASURED, MEETS_*, EXCEEDS_*}
                "这个数准不准" —— 由 `abfe_core.cross_repeat_precision()` 跨**独立
                重复**的样本 SD 判定。单次求解结构上只有一个重复，所以这里恒为
                UNMEASURED；真正的评估在循环级（收集多个 run 之后）。

        **故意删键而不是同名改义。** 同名改义是静默失效：不抛异常、消费者在别的
        文件里照旧读到一个布尔。删掉之后所有 `result.get("converged")` 当场拿到
        None，是 fail-loud。

        `analysis_status == ANALYSIS_COMPLETE` 的充要条件（三条，全部是硬不变量）：
          1. 路径完整：`len(local_results) == len(valid_windows)` 且 `skipped_windows`
             为空（无缺窗、无跳窗）；
          2. 数值有限：`total_delta_G` / `total_error` / 每段 `delta_G_kJ_mol` 与
             `uncertainty_kJ_mol` 全部 `np.isfinite`；
          3. 结构自洽：覆盖到的 λ 索引连续无洞，且相邻窗口恰好共享一个边界态。
        不满足的项逐条写进 `analysis_incomplete_reasons`。

        ---- 为什么 `max_endpoint_uncertainty_kJ_mol ≤ 1.0` 被赶出接受判据 ----
        （provenance，别再把它加回任何合取）
          · **1.0 的来历是一个默认形参**，这段 docstring 自己曾写明它是从
            `run_all_windows` 的 early-stop 启发式（`early_stop_max_uncertainty_kJ_mol`）
            **抄**来的；而那条路径 `enable_early_stop=False` 默认就关着。全仓没有
            任何标定、任何依据支持 1.0 这个数。
          · **它是逐段量，科学目标是两腿合成量。** 它取的是
            `max over chain_segments[*].uncertainty_kJ_mol`，而我们要的是 σ_bind。
            实测：各段 `[0.3×5, 2.0]` ⟹ σ_bind = 2.98 已达标，门却判未收敛并命令
            补帧；把那一段从 2.0 压到 1.0 需要 **4×** 采样量，换来的 ΔG 精度提升
            为零。
          · **MBAR 的 σ 是渐近估计**，前提「去相关后近独立」在本仓不成立
            （`N_eff/g` 常年个位数）。
          · **它对「该采的构型一次都没采到」结构性失明。** 本仓两次大错都是这个
            形状：4W53 stage2 偏 **+32 kJ/mol**、decharging 偏 **−88 kJ/mol**，
            两次 σ 都很小、都没报警。

        final_min_ess_ratio / final_min_absolute_ess / final_min_decorrelated_samples /
        final_max_uncertainty_kJ_mol / final_min_target_absolute_ess /
        final_max_top1pct_raw_weight：**形参保留（不改签名，避免打断调用方），但它们
        现在只影响报告字段、不影响 `analysis_status`。** 对应的量与阈值全部原样留在
        返回 dict 里供审查（用户明确要求），只是不再参与任何合取、也不再有下游据此
        自动补帧。它们是"求解器能不能工作 / 重加权有没有明显退化"的最低条件，不是
        精度判据。**不要给它们设新阈值** —— 设了就是又一个魔法数字。
        """
        valid_windows = [w for w in window_data if w.get("u_kn") is not None and w["u_kn"].size > 0]
        if not valid_windows:
            return self._fallback("no_valid_windows")

        # 按 λ 索引排序，确保拼接顺序正确
        valid_windows.sort(key=lambda d: min(d.get("lambda_indices", [10**9])))

        local_results = []
        skipped_windows: List[Dict[str, Any]] = []
        window_g_values = []          # 每个窗口自相关子采样得到的统计非效率 g（诊断用）
        window_overlap_records = []   # 每个窗口的重加权有效样本比例（真实 overlap 诊断，见 #2）
        # 🔑 [2026-09-11] **生产侧的端到端累计 f_k 偏差，与 warmup gate 分开存。**
        # warmup gate 那份只有 200~600 帧、口径也不同；两边现在对不上，
        # 合并会互相盖掉。这里是 scope="production" 的独立记录。
        cumulative_fk_records: List[Dict[str, Any]] = []

        # 🔑🔑 [2026-09-16] **循环下标不是窗口号，别让它叫 `w_idx`。**
        #
        # `valid_windows` 只含**被载入**的窗口：部分段分析（`[部分窗口段] 本段只采了
        # 窗口 [3]`）里它只有一个元素，于是 `enumerate` 的 0 会被十几条 `窗口 {w_idx}`
        # 的日志当成物理窗口号打出来 —— 真机 brd4_ligand1/rep1 把物理 win3 报成
        # "窗口 0 去相关子采样后有效帧数 (5) < 10，跳过"，排查被直接引到错误的窗口。
        # 结构化字段一直是对的（都走 `source_window_idx`），撒谎的只有人读的那几行。
        # 这个循环里 `w_idx` **只**出现在消息文本中，所以在源头改名一次就全对了：
        # 列表位置叫 `_list_pos`，`w_idx` 从此就是物理窗口号。
        for _list_pos, w in enumerate(valid_windows):
            w_idx = source_window_idx = int(w.get("window_index", _list_pos))
            u_kj_raw = np.asarray(w["u_kn"], dtype=np.float64) # (K_local, N)
            if w.get("bias_energies") is None or w.get("base_energies") is None:
                raise ValueError(
                    f"窗口 {w_idx} 缺少 bias/base 能量；IBS-TMBAR 禁止以零替代"
                )
            bias_kj = np.asarray(w["bias_energies"], dtype=np.float64)
            base_kj = np.asarray(w["base_energies"], dtype=np.float64)
            win_lams = list(w.get("lambda_indices", []))
            # 🔑 [0831issue / ESS_GATE_PROTOCOL_VERSION=4] residual 臂的混合覆盖度
            # 门必须用 sampling-state 能量（f_k 就是在它上面学的），见
            # _ibs_reweighting_quality_diagnostics 的 docstring。这个数组必须跟着
            # 下面的去相关子采样一起走，否则分子分母来自不同帧子集。baseline 窗口和
            # tmbar_history entry 没有这个键 → None → 沿用物理口径，逐位不变。
            # residual 臂标记由生产者显式写入（落盘侧见
            # `_load_validated_joint_score_ledgers`，在线侧见 run_all_windows）。
            # 它是下面 `sampling_gauge_required` 的唯一判据。
            _residual_arm = bool(w.get("residual_sampling_arm"))
            sampling_kj = w.get("sampling_state_energies")
            if sampling_kj is not None:
                sampling_kj = np.asarray(sampling_kj, dtype=np.float64)
                if sampling_kj.shape != u_kj_raw.shape:
                    raise ValueError(
                        f"窗口 {w_idx} sampling_state_energies 形状 "
                        f"{sampling_kj.shape} 与物理 u_kn {u_kj_raw.shape} 不一致；"
                        "拒绝用错位数据算混合覆盖度门"
                    )

            if u_kj_raw.ndim != 2 or len(win_lams) != u_kj_raw.shape[0]:
                print(f"  [WARN] 窗口 {w_idx} 数据维度不匹配，跳过")
                continue
            
            # 确定采样态索引 (通常 IBS 采样的是当前窗口的某个特定态，或者是加权平均)
            # 在 IBS 引擎中，通常假设样本来自当前窗口的“有效采样分布”。
            # 如果未指定 sampled_lambda_index，默认取窗口中间态作为参考，或者取第一个态。
            # 这里为了兼容旧逻辑，若未指定则取中间态，但需注意 bias_energies 必须对应正确的采样态。
            n_frames = u_kj_raw.shape[1]
            if n_frames < min_frames_per_window:
                print(f"  [WARN] 窗口 {w_idx} 原始帧数 ({n_frames}) < min_frames_per_window ({min_frames_per_window})，跳过")
                continue
            if len(bias_kj) != n_frames or len(base_kj) != n_frames:
                raise ValueError(
                    f"窗口 {w_idx} energies/bias/base 帧数不一致"
                )
            if (
                not np.all(np.isfinite(u_kj_raw))
                or not np.all(np.isfinite(bias_kj))
                or not np.all(np.isfinite(base_kj))
            ):
                raise ValueError(f"窗口 {w_idx} energies/bias/base 含 NaN/Inf")
            bias_kj = bias_kj[:n_frames]
            base_kj = base_kj[:n_frames]

            # ------------------------------------------------------------------
            # 🔑 修复（审查报告 #1）: 自相关子采样
            # ------------------------------------------------------------------
            # 该窗口只有一个真实被采样的分布；用能量时间序列估计统计非效率 g，再对
            # 本窗口所有相关数组做同步去相关子采样，避免把强相关的逐帧 MD 输出当独
            # 立样本喂给 MBAR（会让误差棒系统性偏小）。
            # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 序列从 base_kj+bias_kj 换成每个目标态
            # 自己的 Δu_k=(U'_k−V_bias)/kT、取 g 最大的那个态——总势能被溶剂涨落主导，
            # 实测低估 g 达 3-10 倍（见 _decorrelate_by_worst_target_state）。
            # ==================================================================
            # 🔑 [ESS_GATE_PROTOCOL_VERSION=5 / 2026-09-01] 循环论证修复
            # ------------------------------------------------------------------
            # 下面的去相关**用权重序列本身**抽稀帧（`_decorrelate_by_worst_target_state`
            # 里 `series = (u_kj_raw[k] - bias_kj)/kT`，它的指数**就是**权重）。
            # 那对 MBAR 的 n_k 是正确且必要的：相关样本不能当独立样本计数。
            #
            # 但**权重退化门**（raw 绝对 ESS + top-1% 权重集中度）绝不能在这个
            # 子集上算 —— 那是拿权重挑样本、再用挑出来的样本量同一批权重的退化
            # 程度，循环论证。抽稀恰好削掉权重的自相关暴涨段，剩下的权重自然
            # 看起来均匀。
            #
            # 4W53 甲苯溶剂腿 window 0 实测（500 帧，g=2.13，保留 235 帧）：
            #     去相关前  逐态 ESS 241.93 → 3.58   min=3.58   ESS/N=0.0072
            #     去相关后  逐态 ESS 116.69 → 47.01  min=47.01  ESS/N=0.2000
            # 扔掉 53% 的帧让 ESS/N 涨 28 倍 —— 对合法 ESS 估计数学上不可能，
            # 只可能来自"剔除与权重相关"。而绝对门槛是 20：
            # **3.58 该拦、47.01 放行，循环论证直接翻转了判定。**
            # 更糟的是形状：逐态单调崩塌从 68 倍（241.93→3.58）被抹平到 2.5 倍
            # （116.69→47.01）—— 门要探测的信号本身被消掉了。
            # 该窗口贡献 +20.79 kJ/mol、占溶剂腿 stage2 误差的一半以上，却全绿。
            # 完整定位见 docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md §2.5。
            #
            # 修法：去相关照旧（MBAR 需要），但**门改在去相关之前的帧集上算**。
            # 两份都落盘：`raw_*`（受门，去相关前）与
            # `raw_*_post_decorrelation`（仅诊断），这样历史产物里那个被抬高的
            # 数字仍然可解释、也能一眼看出差多少。
            # ==================================================================
            _u_kj_pre_decorr = u_kj_raw
            _bias_kj_pre_decorr = bias_kj
            _sampling_kj_pre_decorr = sampling_kj

            # 🔑 [多采样段] 逐段去相关。多段窗口的两段是**两条独立轨迹**，不能当一条
            # 连续序列估 g；用来源标签切出段边界交给已有的 segments 机制。
            _seg_source_id = w.get("sampling_source_id")
            _seg_f_ks = w.get("sampling_segment_f_k")
            _is_multi_segment = bool(
                _seg_source_id is not None and _seg_f_ks is not None and len(_seg_f_ks) > 1
            )
            _decorr_segments = w.get("production_segments")
            if _is_multi_segment:
                _sid = np.asarray(_seg_source_id, dtype=np.int64).ravel()
                _bounds, _cur = [], 0
                for _s in range(len(_seg_f_ks)):
                    _n = int(np.count_nonzero(_sid == _s))
                    if _n:
                        _bounds.append({
                            "start_frame": _cur, "end_frame": _cur + _n,
                            "n_frames": _n, "session_id": f"sampling_segment_{_s}",
                            "reason": "multi_sampling_segment",
                        })
                        _cur += _n
                _decorr_segments = _bounds
            segment_diagnostics = []
            _g_per_state: List[float] = []
            sub_idx, g_val, g_worst_state = _decorrelate_by_worst_target_state(
                u_kj_raw, bias_kj, self.kt, segments=_decorr_segments,
                segment_diagnostics=segment_diagnostics,
                per_state_g_out=_g_per_state,
            )
            # [DECORR-01 裁决 4] 求解器这一侧的输入身份 + 输出。
            # ⚠️ 与自检那一份**分别记录**：两侧的 g 与去相关帧数实测差 2–8 倍，
            # 差异来源尚未证明，**不许改任一侧的数去追平另一侧**。
            _decorr_prov = decorrelation_provenance(
                caller="solve_stage_integrated",
                energy_array_source=(
                    "window_outputs[...]['u_kn'] (merged across sampling segments)"
                    if w.get("sampling_source_id") is not None
                    else "window_outputs[...]['u_kn']"
                ),
                observable="delta_u_k_over_kT_worst_target_state",
                n_frames_in=int(n_frames), segments=_decorr_segments,
                kt=float(self.kt), g=float(g_val), worst_state=g_worst_state,
                subsample_indices=sub_idx,
                window_idx=int(source_window_idx),
                # [CTL-09] 这里的 u_kj_raw/bias_kj 还是**抽稀前**的原始输入
                # （下面 `sub_idx.size < n_frames` 那一段才切片）；指纹必须取这一份，
                # 取切片后的等于拿自己的输出去比对方的输入，永远对不上。
                energy_array=u_kj_raw, bias_array=bias_kj,
            )
            # 🔑 [CTL-09] **同输入对照在生产里自动跑一次并落盘**，不靠事后手工比。
            # 两侧的去相关帧数实测差 2–8 倍（rep2 win3：自检 56 帧判"够"，求解器
            # 9/7 帧直接跳窗）；只把两份 provenance 各存一边、没人去比，等于这条
            # 分歧永远停在"已记录、未查明"。放在这里（抽稀前）是因为重放必须拿
            # **完全相同的输入**：切片之后重放的是求解器自己的输出，对照就失去意义。
            # ⚠️ 全程 try/except：这是诊断，**不得中断分析**；出错只记 error。
            _decorr_xcheck: Optional[Dict[str, Any]] = None
            try:
                _self_prov = _load_self_support_decorrelation_provenance(w)
                if _self_prov:
                    _decorr_xcheck = compare_decorrelation_provenance(
                        _self_prov, _decorr_prov
                    )
                    if _decorr_xcheck.get("inputs_identical"):
                        # 只有判定输入一致才重放 —— 输入不同还重放，等于又造了
                        # 第三个互相不可比的数（见 replay_decorrelation 的 docstring）。
                        _decorr_xcheck["replay"] = replay_decorrelation(
                            u_kj_raw, bias_kj, self.kt, segments=_decorr_segments
                        )
            except Exception as _xc_err:  # noqa: BLE001 —— 诊断不得中断分析
                _decorr_xcheck = {"error": repr(_xc_err)}
            for segment in segment_diagnostics:
                start = segment["start_frame"]
                segment["base_energy_jump_kJ_mol"] = float(base_kj[start] - base_kj[start-1]) if start else None
            if sub_idx.size < n_frames:
                u_kj_raw = u_kj_raw[:, sub_idx]
                bias_kj = bias_kj[sub_idx]
                base_kj = base_kj[sub_idx]
                if sampling_kj is not None:
                    sampling_kj = sampling_kj[:, sub_idx]
                n_frames = int(sub_idx.size)
            # 🔑🔑 [S2-A 数据链，2026-09-14] **逐段残差必须拿「同一个 sub_idx 切片后
            # 的、collapse 之前的」那一份。** 下面多段窗口会把 `bias_kj` 整个换成
            # 自洽塌缩后的 `bias_mix_kJ_mol`（那是**混合**偏置，不是任何一段生产时
            # 真正用的偏置），`sampling_source_id` 也还是未裁剪的全长版本。
            # 直接拿那两个去算逐段残差 ⟹ 要么长度不符、要么算的根本不是那一段的
            # 偏置。这里在两件事发生**之前**各留一份，切片口径与 `bias_kj` 逐帧对齐。
            _fk_bias_production = np.array(bias_kj, dtype=np.float64, copy=True)
            _fk_source_ids = (
                None if _seg_source_id is None else
                np.asarray(_seg_source_id, dtype=np.int64).ravel()[sub_idx]
            )
            window_g_values.append(float(g_val))
            if n_frames < min_frames_per_window:
                # 🔑 跳过的窗口必须被**显式记录**：这个总和缺了一段路径，却照样会被
                # 当成 stage 的 total_delta_G 打印和比较。实测踩过一次：段 1 的 w3
                # 因去相关后只剩 5 帧被跳过，86.42 是个部分和，跟段 2 的 52.75 并排
                # 摆出来时看着像"两次采样差 33.7 kJ/mol"，其实是缺了一个窗口。
                print(f"  [WARN] 窗口 {w_idx} 去相关子采样后有效帧数 ({n_frames}) < {min_frames_per_window}，跳过")
                skipped_windows.append({
                    "window_index": int(source_window_idx),
                    "lambda_indices": [int(x) for x in win_lams],
                    "n_frames_after_decorrelation": int(n_frames),
                    "min_frames_per_window": int(min_frames_per_window),
                    "statistical_inefficiency": float(g_val),
                    # 🔑 逐态 g 剖面。被跳过的窗口恰恰是"整窗一个 g"最误导的那一格：
                    # 远端单调衰减（跨度太大）与整体偏低（采样不够）在单个 g 上
                    # 长得一模一样，而两者的对症动作完全不同。
                    "statistical_inefficiency_per_lambda": [
                        float(x) for x in _g_per_state
                    ],
                    "reason": "insufficient_frames_after_decorrelation",
                    # 跳窗记录同样带上输入身份 —— 它正是"自检说够、求解器说不够"
                    # 那条分歧最需要对照的一份。
                    "decorrelation_provenance": _decorr_prov,
                    # [CTL-09] 被跳过的窗口恰恰是分歧最尖锐的一格；对照结论只放
                    # overlap 记录里的话，这里就永远看不到（跳窗不进 overlap）。
                    "decorrelation_cross_check": _decorr_xcheck,
                })
                # 🔑🔑 [CTL-08] 跳窗也必须留下一条**显式 UNMEASURED 的逐段 f_k 残差
                # 记录**。这个 `continue` 在残差计算**之前**，先前被跳过的窗口因此
                # 连一条记录都没有 —— 而下游控制器的语义是「**缺证据 ≠ 没这回事**」：
                # `cumulative_fk_residual_production` 里干脆没有这个窗口，读起来跟
                # "这个窗口查过、没问题"完全一样，于是路径缺一截却照样往下走。
                # 显式记 UNMEASURED 才能让"没测到"和"测了且通过"分开。
                # 字段与成功路径同形状（scope/window_index/lambdas/frameset 身份），
                # 否则下游按键取值会在跳窗记录上 KeyError 或静默取 None。
                _skip_segs = w.get("production_segments") or []
                cumulative_fk_records.append({
                    "scope": "production",
                    "window_index": int(source_window_idx),
                    "lambdas": list(win_lams),
                    "verdict": "UNMEASURED",
                    "reason": "window_skipped_before_residual_computation",
                    "n_frames_after_decorrelation": int(n_frames),
                    "min_frames_per_window": int(min_frames_per_window),
                    "n_frames_in_frameset": int(u_kj_raw.shape[1]),
                    "frameset_id": _frameset_identity(
                        u_kj_raw.shape[1], _skip_segs
                    ),
                    "segment_ids": [x.get("session_id") for x in _skip_segs] or None,
                    "energy_gauge": "sampling_states",
                })
                continue

            # 🔑 [多采样段] 帧集**已经定下来**了，现在才做自洽塌缩：抽帧改变了各段
            # 计数，用全帧解出的混合权重切片后不再精确（实测差 0.178 kJ/mol）。
            _multi_seg = None
            if _is_multi_segment:
                if sampling_kj is None:
                    raise ValueError(
                        f"窗口 {w_idx} 是多采样段，但缺 sampling_state_energies。"
                        "交叉能量必须从采样规范重建，不能用含逐 λ 态尾项的 energies 代替。"
                    )
                import multi_segment_analysis as _msa
                _sid_sub = np.asarray(_seg_source_id, dtype=np.int64).ravel()[sub_idx]
                _cross_sub = np.vstack([
                    _msa.mixture_energy(sampling_kj, np.asarray(_f, dtype=np.float64), self.kt)
                    for _f in _seg_f_ks
                ])
                _multi_seg = _msa.collapse_segments(
                    _cross_sub, _sid_sub, u_kj_raw, self.kt, n_segments=len(_seg_f_ks)
                )
                bias_kj = _multi_seg["bias_mix_kJ_mol"]

            # ------------------------------------------------------------------
            # 🔑 修复 1: 全局能量偏移 (Global Offset)
            # ------------------------------------------------------------------
            # 这里 u_kj_raw 存的是 U_k_int；分析时重建 U_k = E_base + U_k_int。
            u_phys_kj = base_kj[None, :] + u_kj_raw
            mean_energies = np.mean(u_phys_kj, axis=1)
            global_offset = np.min(mean_energies)
            
            # 对所有物理态能量进行平移
            u_kj_shifted = u_phys_kj - global_offset
            
            # ------------------------------------------------------------------
            # 🔑 修复 2: 构建增广矩阵 (Augmented Matrix)
            # ------------------------------------------------------------------
            # Row 0: 真实采样势能 E_base + V_bias
            # Rows 1..K: 各物理态 E_base + U_k_int
            u_sampled_eff = base_kj + bias_kj - global_offset
            
            # 构建 (K+1, N) 矩阵
            u_mbar = np.vstack([u_sampled_eff, u_kj_shifted])

            # 🔑 关键修复：pymbar.MBAR 要求输入是无量纲的约化势 (β·U)，这里在此之前
            # 一直是 kJ/mol 原始能量，从未乘过 self.beta（self.beta/self.kt 在这个
            # 方法里定义了却从没被用上）。把 kJ/mol 尺度的数直接喂给期望"已经是
            # kT 量级"的 MBAR，等价于让它在一个错误的（偏冷很多的）有效温度下求解，
            # 算出来的自由能差会被系统性放大。末尾 `f_phys_kj = f_phys_kt * self.kt`
            # 那一步本身没错，但前提是喂给 MBAR 的输入必须已经是约化势——现在补上这个
            # 转换，让输入输出的单位换算配对。
            u_mbar = u_mbar * self.beta

            # 样本计数：只有第 0 行（采样分布，而非物理 lambda 态）有 N 个样本。
            # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 上面 u_mbar 的 vstack 把 row 0 硬编码
            # 成采样分布、row 1..K 硬编码成物理态，所以 sampled_row 只能是 0。旧代码
            # 只对"越界"值回退到 0，却放过"合法但非零"的值——那会让 MBAR 以为样本来自
            # 某个*物理*态（物理错误），并让下面 `i != sampled_row` + zip(win_lams, ...)
            # 的 ESS→λ 映射整体错位。改为 fail closed。
            sampled_row = int(w.get("sampled_distribution_row", 0))
            n_k_local = np.zeros(len(win_lams) + 1, dtype=np.int32)
            if sampled_row != 0:
                raise ValueError(
                    f"窗口 {w_idx} sampled_distribution_row={sampled_row} != 0；"
                    "增广矩阵 row 0 固定是 IBS 采样分布、row 1..K 是物理态，非零采样行"
                    "会同时破坏 MBAR 的采样态声明与 ESS→λ 映射，拒绝继续。"
                )
            n_k_local[sampled_row] = n_frames
            
            # ------------------------------------------------------------------
            # 🔑 修复 3: 列平移稳定性 (Column-wise Shift)
            # ------------------------------------------------------------------
            # 进一步对每一列减去最小值，防止 exp 溢出
            u_min_col = np.min(u_mbar, axis=0, keepdims=True)
            u_mbar_stable = u_mbar - u_min_col
            
            # 剔除含 NaN 或 Inf 的列
            valid_mask = np.isfinite(u_mbar_stable).all(axis=0)
            u_mbar_final = u_mbar_stable[:, valid_mask]
            n_k_local[sampled_row] = np.sum(valid_mask) # 更新有效样本数

            # 🔑 [v23 numerical follow-up #2] 这里之前硬编码了字面量 10，而不是
            # 用调用方传入的 min_frames_per_window（该参数默认值恰好也是 10，
            # 这个硬编码大概率是把默认值抄成字面量留下的）。_solve_tmbar_and_
            # recenter 在线学习路径故意把 min_frames_per_window 放宽到 3
            # （docstring 原话："去相关后统计非效率 g≈5-15 时常剩 2-7 帧，用默认
            # 的 10 帧门槛会让绝大多数 minibatch 被 solve_stage_integrated 自身
            # 的逐 entry 门槛跳过、白白丢弃已经采到的真实数据"）——但这条硬编码
            # 完全无视那个放宽，任何 3-9 帧的 minibatch 依然在这里被无声跳过。
            # 真实后果：len(local_results) < len(valid_windows)，导致
            # converged 恒为 False，且不打印任何警告，表现上跟"roundoff 卡在
            # 阈值"一模一样，实际根因完全不同（这条本身不是舍入问题，是这里从
            # 未真正遵守 min_frames_per_window 参数）。
            if n_k_local[sampled_row] < min_frames_per_window:
                print(
                    f"  [WARN] 窗口 {w_idx} 剔除 NaN/Inf 列后有效帧数 "
                    f"({int(n_k_local[sampled_row])}) < min_frames_per_window "
                    f"({min_frames_per_window})，跳过"
                )
                continue

            if not has_pymbar():
                return self._fallback("pymbar_not_installed")
            
            try:
                # 使用混合求解器，提高收敛性
                mbar = _build_mbar_compatible(
                    u_mbar_final,
                    n_k_local,
                    relative_tolerance=1e-7,
                    initialize="BAR",
                    solver_protocol="default",
                )
                
                res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
                df_matrix, ddf_matrix = _extract_free_energy_arrays(res, require_uncertainty=True)

                # ------------------------------------------------------------------
                # 🔑 修复（审查报告 #2）：真实的重加权质量诊断
                # ------------------------------------------------------------------
                # 该窗口的 MBAR 只有一个真实被采样的态 (row=sampled_row)，其余物理态
                # 样本数均为 0；标准 MBAR 重叠矩阵 compute_overlap() 在这种"单一采样
                # 分布 + 多个零样本目标态"场景下会退化（未采样列恒为 0，无法反映真实
                # 重叠）。这里改用 compute_effective_sample_number()：它衡量"把 row 0
                # 的样本重加权到每个目标 λ 态后，还剩多少个等效独立样本"，是这类单向
                # 重加权场景下物理上正确、数值上已验证有效的重叠代理（详见审查报告）。
                # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 受门的 min_ess_ratio 换成"扣掉共模
                # 因子后的混合覆盖度 ESS(p_k)"；raw 单参考 ESS 降级为只报告不设门。
                # 理由见 _ibs_reweighting_quality_diagnostics 顶部长注释：raw 量被
                # Group-4 λ-WCA 防护壳 + LRC 的逐帧共模因子按 exp(σ_r²) 整体压掉
                # （实测 σ_r 0.95→2.40 kT，对应 ESS/N 上限 0.40→0.003），而那个因子在
                # 本方法真正报出去的物理态↔物理态自由能差里大部分抵消。
                min_ess_ratio = None
                ess_ratio_per_lambda = None
                min_absolute_ess_this_window = None
                raw_min_ess_ratio = None
                raw_min_absolute_ess = None
                raw_ess_ratio_per_lambda = None
                min_occ_norm_this_window = None
                # [ESS_GATE_PROTOCOL_VERSION=5] 去相关后的 raw 量：仅诊断、不受门。
                raw_min_ess_ratio_post_decorrelation = None
                raw_min_absolute_ess_post_decorrelation = None
                raw_ess_ratio_per_lambda_post_decorrelation = None
                raw_gate_n_frames_pre_decorrelation = None
                quality = {}
                quality_pre = {}
                try:
                    neff = np.asarray(mbar.compute_effective_sample_number(), dtype=float)
                    denom = max(int(n_k_local[sampled_row]), 1)
                    # sampled_row 已在上面 fail-closed 保证为 0，所以 target_idx 严格是
                    # [1..K]，与 win_lams 同序，zip 映射成立。
                    target_idx = [i for i in range(len(neff)) if i != sampled_row]
                    if target_idx:
                        # ⚠️ [ESS_GATE_PROTOCOL_VERSION=5] neff 来自**去相关后**的
                        # 增广矩阵，因此被上面那段注释里的循环论证抬高，**不再受门**，
                        # 只作为诊断落盘（历史产物对账用）。受门的量在下面重算。
                        raw_ratio_post = neff[target_idx] / denom
                        raw_min_ess_ratio_post_decorrelation = float(np.min(raw_ratio_post))
                        raw_min_absolute_ess_post_decorrelation = float(
                            np.min(neff[target_idx])
                        )
                        raw_ess_ratio_per_lambda_post_decorrelation = {
                            int(lam): float(r) for lam, r in zip(win_lams, raw_ratio_post)
                        }

                        # ---- 受门的 raw 权重退化：在**去相关之前**的帧集上算 ----
                        # 复用同一个诊断函数，不另写数学：raw_ess / top1pct 的定义
                        # 只依赖 (u, bias, kt)，与去相关无关。
                        _pre_finite = (
                            np.isfinite(_u_kj_pre_decorr).all(axis=0)
                            & np.isfinite(_bias_kj_pre_decorr)
                        )
                        quality_pre = _ibs_reweighting_quality_diagnostics(
                            _u_kj_pre_decorr[:, _pre_finite],
                            _bias_kj_pre_decorr[_pre_finite],
                            w.get("f_k"),
                            self.kt,
                            sampling_kj=(
                                None if _sampling_kj_pre_decorr is None
                                else _sampling_kj_pre_decorr[:, _pre_finite]
                            ),
                            # 🔑 [2026-09-09] 判据取自窗口自带的 residual 臂标记，
                            # 不再用 `_sampling_kj_pre_decorr is not None` —— 那是拿被
                            # 守护的数组自己推导要不要守护它，恒真、永不触发。
                            sampling_gauge_required=_residual_arm,
                        )
                        _pre_denom = max(int(np.count_nonzero(_pre_finite)), 1)
                        if quality_pre.get("raw_ess") is not None:
                            _raw_pre = np.asarray(quality_pre["raw_ess"], dtype=float)
                            raw_min_absolute_ess = float(np.min(_raw_pre))
                            raw_min_ess_ratio = float(np.min(_raw_pre / _pre_denom))
                            # 保留逐态明细而不只是 min：调用方（abfe_preoptimizer.py 的
                            # refine_stage_lambda_path_by_overlap）需要知道是窗口内*哪个*
                            # λ 才是瓶颈，而不只是"这个窗口整体没过"。
                            raw_ess_ratio_per_lambda = {
                                int(lam): float(r)
                                for lam, r in zip(win_lams, _raw_pre / _pre_denom)
                            }
                        raw_gate_n_frames_pre_decorrelation = int(_pre_denom)

                    # 必须用 valid_mask 之后的那批帧：denom 是
                    # n_k_local[sampled_row]=sum(valid_mask)，若 quality 用未过滤
                    # 的 n_frames 帧，比例的分子分母就来自不同样本集。
                    # `sampling_kj` 已在上面跟 u_kj_raw 一起走过去相关子采样，这里
                    # 只需再套同一个 valid_mask。见其赋值处的说明。
                    quality = _ibs_reweighting_quality_diagnostics(
                        u_kj_raw[:, valid_mask],
                        bias_kj[valid_mask],
                        w.get("f_k"),
                        self.kt,
                        sampling_kj=(
                            None if sampling_kj is None
                            else sampling_kj[:, valid_mask]
                        ),
                        # 同上：不用 `sampling_kj is not None` 自证。
                        sampling_gauge_required=_residual_arm,
                    )
                    if _multi_seg is not None:
                        # 🔑 [多采样段] 覆盖度/占据必须按**合并的混合分布**算：
                        # 每帧先算它属于各采样分布的后验 r_s(n)，再混合各段的 λ
                        # 成员概率 p_{k|s}（用该段冻结的 f_k 与采样态能量，与
                        # `_ibs_reweighting_quality_diagnostics` 同口径）。
                        # 逐段最差只落诊断、不当门：合并 ESS 既不保证严格相加也不
                        # 保证单调增，拿最差段的绝对 ESS 配两段总帧数更是分子分母
                        # 不匹配。原来那份单组 f_k 的结果在多段下没有意义。
                        import multi_segment_analysis as _msa
                        _merged_cov = _msa.merged_coverage_diagnostics(
                            _multi_seg["responsibility"][:, valid_mask],
                            sampling_kj[:, valid_mask],
                            [
                                _seg_f_ks[int(i)]
                                for i in _multi_seg["active_segments"]
                            ],
                            self.kt,
                        )
                        quality["mixture_ess"] = _merged_cov["coverage_ess"]
                        quality["mixture_ess_ratio"] = _merged_cov["coverage_ess_ratio"]
                        quality["mixture_occupancy_normalized"] = (
                            _merged_cov["occupancy_normalized"]
                        )
                        quality["multi_segment_merged_coverage"] = True
                    if quality.get("mixture_ess") is not None:
                        mix_ess = np.asarray(quality["mixture_ess"], dtype=float)
                        mix_ratio = mix_ess / denom
                        min_ess_ratio = float(np.min(mix_ratio))
                        min_absolute_ess_this_window = float(np.min(mix_ess))
                        ess_ratio_per_lambda = {
                            int(lam): float(r) for lam, r in zip(win_lams, mix_ratio)
                        }
                        occ_norm = quality.get("mixture_occupancy_normalized")
                        min_occ_norm_this_window = (
                            float(np.min(occ_norm)) if occ_norm else None
                        )
                    elif "f_k" in w:
                        # 只在调用方**本来打算**给 f_k 的时候才报警。完全没有 'f_k'
                        # 键的调用方（在线 warmup 的 tmbar_history、λ 重划分探针、
                        # 合成数据）是刻意不用这个门的，它们每次求解会遍历十几个
                        # entry，逐 entry 打印会把日志刷爆；对它们 min_ess_ratio=None
                        # 本身就是信号（converged 因此 fail closed）。
                        print(
                            f"  [WARN] 窗口 {w_idx} 提供了 f_k 但无法计算混合覆盖度 ESS "
                            f"({quality.get('error')})；该窗口 min_ess_ratio=None，"
                            "收敛门会因此判失败（fail closed，不当作通过）"
                        )
                except Exception as ess_exc:
                    print(f"  [WARN] 窗口 {w_idx} 有效样本数(ESS)重叠诊断计算失败: {ess_exc}")
                # 🔑 [0831issue P2] 这条 overlap 记录**先构造、后 append**：以前它在这里
                # 就直接进了 window_overlap_records，而 local_results 要到下面才进。
                # 两者之间还有 ddf_matrix 的端点索引（n_lams>1 时 ddf_matrix[1, n_lams]）
                # 等可能抛异常的步骤，一旦抛出就被本 try 的 except 吞掉 → 该窗口进了
                # "落盘统计的全部窗口"，却没进协方差链/积分区间，两份账目分叉（converged
                # 仍被 fail-closed 挡住，但诊断会误导）。现在两个 append 放同一原子块。
                _overlap_record = {
                    "window_index": source_window_idx,
                    "decorrelation_provenance": _decorr_prov,
                    "decorrelation_cross_check": _decorr_xcheck,
                    "window_label": w.get("window_label"),
                    "window_range": list(w.get("window_range", [])),
                    "lambdas": win_lams,
                    "lambdas_coul": list(w.get("lambdas_coul", [])),
                    "lambdas_vdw": list(w.get("lambdas_vdw", [])),
                    "min_ess_ratio": min_ess_ratio,
                    "ess_ratio_per_lambda": ess_ratio_per_lambda,
                    "absolute_ess": min_absolute_ess_this_window,
                    # 一阶矩：K*<p_k> 的最小值（理想=1）。与 min_ess_ratio 一起构成
                    # "权重质量"这一份证据的两半——ESS 是尺度不变的，抓不到"均匀被
                    # 饿死"的态，必须靠这一项。
                    "min_occupancy_normalized": min_occ_norm_this_window,
                    "mixture_occupancy_normalized": quality.get(
                        "mixture_occupancy_normalized"
                    ),
                    # 🔑 去相关、剔除 NaN/Inf 列之后真正喂进 MBAR 的样本数（不是
                    # 去相关前的原始帧数）——跟下面 n_k_local[sampled_row]<10 的
                    # 门槛用的是同一个量。
                    "n_frames_decorrelated": int(n_k_local[sampled_row]),
                    # 🔑 逐态 g 剖面（诊断用，不设门）。与已有的
                    # raw_ess_ratio_per_lambda 配成一对：一个答"这个态的重要性
                    # 权重有多少支撑"，一个答"这个态的序列去相关有多慢"。
                    "statistical_inefficiency_per_lambda": [
                        float(x) for x in _g_per_state
                    ],
                    # ---- 只报告、不设门 ----
                    "ess_gate_protocol_version": int(ESS_GATE_PROTOCOL_VERSION),
                    "ess_gate_metric": "mixture_coverage_ess_common_mode_removed",
                    # [0831issue / ESS_GATE_PROTOCOL_VERSION=4] 混合覆盖度用的是哪个
                    # 口径："sampling_states"（residual 臂，f_k 的学习口径）还是
                    # "physical_targets"（baseline，两者恒等）。None = 门未算出。
                    "ess_gate_mixture_gauge": quality.get("mixture_gauge"),
                    # 🔑 [ESS_GATE_PROTOCOL_VERSION=5] 这四项 raw_* 现在算在**去相关
                    # 之前**的帧集上（循环论证修复，见上方长注释）。受门。
                    "raw_min_ess_ratio": raw_min_ess_ratio,
                    "raw_min_absolute_ess": raw_min_absolute_ess,
                    "raw_ess_ratio_per_lambda": raw_ess_ratio_per_lambda,
                    "top1pct_raw_weight": quality_pre.get("top1pct_raw_weight"),
                    "raw_gate_n_frames_pre_decorrelation": (
                        raw_gate_n_frames_pre_decorrelation
                    ),
                    # ---- 去相关后的同名量：仅诊断，用于跟历史产物对账 ----
                    # 两者之比就是循环论证抬高了多少倍。v4 及更早的落盘里，
                    # `raw_min_absolute_ess` 记的是**下面这个**值。
                    "raw_min_ess_ratio_post_decorrelation": (
                        raw_min_ess_ratio_post_decorrelation
                    ),
                    "raw_min_absolute_ess_post_decorrelation": (
                        raw_min_absolute_ess_post_decorrelation
                    ),
                    "raw_ess_ratio_per_lambda_post_decorrelation": (
                        raw_ess_ratio_per_lambda_post_decorrelation
                    ),
                    "top1pct_raw_weight_post_decorrelation": quality.get(
                        "top1pct_raw_weight"
                    ),
                    # 🔑 [2026-09-02] 偏置涨落/物理信号 之比（仅诊断，不设门）。
                    # 目前唯一实测能预测 ΔF 误差的量（Spearman +0.886），而同批
                    # 数据 ESS 与误差是 −0.600 反相关。取 quality_pre：与其它 raw
                    # 量同一帧集（去相关前）。详见其计算处的长注释。
                    "bias_to_signal_ratio": quality_pre.get("bias_to_signal_ratio"),
                    "bias_fluctuation_kJ_mol": quality_pre.get(
                        "bias_fluctuation_kJ_mol"
                    ),
                    "target_energy_fluctuation_kJ_mol": quality_pre.get(
                        "target_energy_fluctuation_kJ_mol"
                    ),
                    "common_mode_log_sigma_kT": quality.get("common_mode_log_sigma_kT"),
                    "common_mode_log_mean_kJ_mol": quality.get(
                        "common_mode_log_mean_kJ_mol"
                    ),
                    "reweighting_quality_error": quality.get("error"),
                    "statistical_inefficiency": float(g_val),
                    "statistical_inefficiency_worst_state": int(g_worst_state),
                    "production_segments": segment_diagnostics,
                }

                # Delta_f[0, k] 是第 k 个物理态相对于采样态 (Row 0) 的自由能差 (单位: kT)
                # 注意：res['Delta_f'] 的形状是 (K+1, K+1)
                # 我们想要的是物理态 (indices 1..K) 的结果
                f_phys_kt = df_matrix[0, 1:] 
                df_phys_kt = ddf_matrix[0, 1:]
                
                # 转换为 kJ/mol
                f_phys_kj = f_phys_kt * self.kt
                df_phys_kj = df_phys_kt * self.kt

                # 🔑 端点（第一个 vs 最后一个物理态）自由能差的不确定度，直接读
                # ddf_matrix 里这两个物理态之间的元素，不用 sqrt(a^2+b^2) 合并
                # 各自相对采样态的边际不确定度——同一次 MBAR 拟合、同一批样本，
                # 彼此有协方差，平方相加会系统性算错。约定跟
                # _solve_single_window_local_mbar（在线 early-stop 用）完全一致：
                # 物理态在增广矩阵里占据第 1..len(win_lams) 行，第 0 行是采样分布。
                n_lams = len(win_lams)
                endpoint_diff_uncertainty_kj = (
                    float(ddf_matrix[1, n_lams] * self.kt) if n_lams > 1 else 0.0
                )
                _overlap_record["endpoint_diff_uncertainty_kJ_mol"] = endpoint_diff_uncertainty_kj

                # 注意：这里的 f_phys_kj 是相对于“采样态有效能量”的绝对自由能估计。
                # 为了拼接，我们需要相对于窗口内某个特定 lambda 的相对自由能。
                # 通常我们关心的是窗口内各 lambda 之间的相对差值。
                # 由于所有态都减去了同一个 global_offset，且参考态一致，相对差值是准确的。
                
                if _multi_seg is not None:
                    import multi_segment_analysis as _msa
                    _msa.assert_collapse_matches_multistate(
                        np.asarray(f_phys_kj, dtype=np.float64)[None, :]
                        - np.asarray(f_phys_kj, dtype=np.float64)[:, None],
                        _multi_seg["target_delta_f_kJ_mol"],
                    )
                local_results.append({
                    # 🔑 [P0-8] 真实来源窗口号；下面 chain_segments 里那个
                    # "window_index" 其实是 local_results 的位置下标，两者在有窗口被
                    # 跳过时会分叉，覆盖诊断必须用这个。
                    "source_window_index": source_window_idx,
                    "lambdas": win_lams,
                    "f": f_phys_kj.astype(float), # 相对于采样参考点的绝对 F
                    "df": df_phys_kj.astype(float),
                    "dDelta_f": (
                        # 🔑 [多采样段] σ 必须来自真正的 S+K 解。塌缩后的单态解把
                        # **估计出来的**混合权重当已知常数用，σ 会偏小。
                        _multi_seg["target_dDelta_f_kJ_mol"]
                        if _multi_seg is not None
                        else np.asarray(ddf_matrix[1:, 1:], dtype=np.float64) * self.kt
                    ),
                    "weight": int(n_k_local[0]),
                    "global_offset": global_offset # 记录偏移量，用于调试
                })
                # [0831issue P2] 同一原子块：只有这个窗口真的进了协方差链，它的
                # overlap 诊断才进落盘统计。
                window_overlap_records.append(_overlap_record)

                # ---- 生产侧累计 f_k 残差（scope=production）----
                # ⚠️ **必须用 sampling gauge 另解一次 local MBAR**：这里的
                # `u_kj_raw` 是 `energies`（物理目标态，ΔG 用它），而 f_k 残差用它会被
                # 逐 λ 态常数（LRC）污染 —— 实测把某窗 span 抬高 2.2 倍。
                # 代价是每窗多一次 CPU MBAR，无 GPU。
                try:
                    _fk_eff = w.get("f_k")
                    if _fk_eff is None and sampling_kj is not None:
                        # 🔑🔑 [S2-A 逐段门，2026-09-14 裁决 3]
                        # 多段窗口的帧采自**两份不同偏置**，`_load_ibs_window_outputs_merged`
                        # 因此显式 `base.pop("f_k")`。先前这里直接记
                        # `no_effective_f_k_for_these_frames` ⟹ 这项证据对多段窗口
                        # **永久缺失**（循环一旦用换 Epoch 修好一个窗口，它就再也拿不到）。
                        #
                        # 正解**不是**兜一份冻结值冒充全体帧的偏置，而是**按段分别算**：
                        # `sampling_source_id` 给出逐帧的有序段映射，
                        # `sampling_segment_f_k` 给出每段自己的冻结 f_k。每段在**自己的
                        # 帧切片**上独立解一次，**绝不**拿合并后的 ΔF 作逐段望远镜相消。
                        _cum = _per_segment_cumulative_fk_residual(
                            sampling_kj, _fk_bias_production, base_kj,
                            win_lams, self.kt,
                            source_ids=_fk_source_ids,
                            segment_f_ks=w.get("sampling_segment_f_k"),
                            provenance=w.get("sampling_segment_provenance"),
                            w_idx=int(source_window_idx),
                        )
                    elif _fk_eff is None:
                        _cum = {"error": "no_effective_f_k_for_these_frames"}
                    elif sampling_kj is None:
                        _cum = {"error": "no_sampling_state_energies"}
                    else:
                        _cum = cumulative_fk_residual(
                            sampling_kj, bias_kj, base_kj, win_lams, self.kt,
                            np.asarray(_fk_eff, dtype=np.float64),
                            w_idx=int(source_window_idx),
                        )
                    _segs = w.get("production_segments") or []
                    _fk_arr = (
                        None if _fk_eff is None
                        else np.asarray(_fk_eff, dtype=np.float64).ravel()
                    )
                    _cum.update({
                        "scope": "production",
                        "window_index": int(source_window_idx),
                        "lambdas": list(win_lams),
                        "n_frames_in_frameset": int(u_kj_raw.shape[1]),
                        # frameset 身份：帧数 + 分段边界。够用来判"这份残差是从哪批帧
                        # 算出来的"，而且不把自产数组的 sha256 当身份
                        # （那是本仓库已复发 4 次的坑）。
                        "frameset_id": _frameset_identity(
                            u_kj_raw.shape[1], _segs
                        ),
                        "segment_ids": [
                            x.get("session_id") for x in _segs
                        ] or None,
                        # 与 frozen_candidate_fingerprint 同口径（**已减均值**），
                        # 否则同一份 f_k 会在两处算出两个指纹。
                        "f_k_fingerprint": (
                            None if _fk_arr is None
                            else frozen_candidate_fingerprint(_fk_arr)
                        ),
                        "energy_gauge": "sampling_states",
                    })
                    cumulative_fk_records.append(_cum)
                except Exception as _cum_err:  # noqa: BLE001 —— 诊断不得中断分析
                    cumulative_fk_records.append({
                        "scope": "production",
                        "window_index": int(source_window_idx),
                        "error": repr(_cum_err),
                    })
                
            except Exception as e:
                print(f"  [WARN] 局部 TMBAR 窗口 {w_idx} (Lams {win_lams}) 失败: {e}")
                import traceback
                traceback.print_exc()

        if not local_results:
            return self._incomplete_path_fallback(
                "no_local_tmbAR_results", skipped_windows)

        # ------------------------------------------------------------------
        # 拼接全局自由能曲线
        # ------------------------------------------------------------------
        global_f = {}
        global_var = {}
        offset_var_terms = []
        
        # 初始化：使用第一个窗口的结果
        first = local_results[0]
        for i, lam in enumerate(first["lambdas"]):
            global_f[lam] = float(first["f"][i])
            global_var[lam] = float(max(first["df"][i], 1e-6)**2)
            
        # 逐个窗口拼接
        for local in local_results[1:]:
            # 寻找当前窗口与前一个已拼接全局结果的_overlap_ lambda
            overlap_lams = [lam for lam in local["lambdas"] if lam in global_f]
            
            if not overlap_lams:
                return self._incomplete_path_fallback(
                    "window_overlap_broken", skipped_windows)
            
            # 计算偏移量 (Offset)
            # 对于重叠的 lambda，计算 (Global_F - Local_F) 的加权平均值
            offsets = []
            offset_vars = []
            for lam in overlap_lams:
                idx_in_local = local["lambdas"].index(lam)
                # 局部自由能
                f_loc = float(local["f"][idx_in_local])
                var_loc_overlap = float(max(local["df"][idx_in_local], 1e-6)**2)
                # 全局自由能
                f_glob = global_f[lam]
                var_glob_overlap = global_var[lam]
                offsets.append(f_glob - f_loc)
                offset_vars.append(var_glob_overlap + var_loc_overlap)

            # 🔑 修复（审查报告 #4）：offset 之前用的是重叠点的非加权均值，而下面
            # 逐 λ 合并却是逆方差加权——当重叠点之间不确定度差异较大时，二者不一致
            # 会让 offset 本身产生偏置，拖动整条拼接曲线。这里统一改为逆方差加权，
            # 与下方合并公式使用同一套权重逻辑。
            offsets_arr = np.asarray(offsets, dtype=float)
            offset_vars_arr = np.asarray(offset_vars, dtype=float)
            inv_var = 1.0 / np.maximum(offset_vars_arr, 1e-12)
            offset = float(np.sum(offsets_arr * inv_var) / np.sum(inv_var))
            offset_var = float(1.0 / np.sum(inv_var))
            offset_var_terms.append(offset_var)
            
            # 应用偏移量并合并
            for i, lam in enumerate(local["lambdas"]):
                f_loc = float(local["f"][i]) + offset
                var_loc = float(max(local["df"][i], 1e-6)**2) + offset_var
                
                if lam in global_f:
                    # 逆方差加权平均
                    w_old = 1.0 / max(global_var[lam], 1e-12)
                    w_new = 1.0 / max(var_loc, 1e-12)
                    
                    global_f[lam] = (global_f[lam] * w_old + f_loc * w_new) / (w_old + w_new)
                    global_var[lam] = 1.0 / (w_old + w_new)
                else:
                    global_f[lam] = f_loc
                    global_var[lam] = var_loc

        # 整理结果
        sorted_lams = sorted(global_f.keys())
        f_curve = np.array([global_f[lam] for lam in sorted_lams])
        err_curve = np.sqrt([global_var[lam] for lam in sorted_lams])
        
        # 总自由能与误差必须来自每个独立窗口中“连接点→新增端点”的直接
        # MBAR 差值。各窗口样本独立，因此窗口段方差可相加；同一窗口内的
        # 端点协方差已经包含在 dDelta_f[i,j] 中。禁止再把两个相对采样态的
        # 边际方差做 sqrt(a²+b²)。
        chain_segments = []
        covered_lambdas = set()
        total_dg = 0.0
        total_variance = 0.0
        for local_idx, local in enumerate(local_results):
            local_lams = [int(x) for x in local["lambdas"]]
            if not local_lams:
                continue
            if local_idx == 0:
                join_lam = local_lams[0]
            else:
                shared = [lam for lam in local_lams if lam in covered_lambdas]
                if not shared:
                    return self._incomplete_path_fallback(
                        "window_overlap_broken_for_covariance_chain", skipped_windows)
                # 选择路径方向上最靠后的共享态，避免多重 overlap 被重复积分。
                join_lam = max(shared)
            end_lam = local_lams[-1]
            join_idx = local_lams.index(join_lam)
            end_idx = len(local_lams) - 1
            segment_dg = float(local["f"][end_idx] - local["f"][join_idx])
            segment_error = float(local["dDelta_f"][join_idx, end_idx])
            if not np.isfinite(segment_dg) or not np.isfinite(segment_error):
                return self._fallback("nonfinite_covariance_chain_segment")
            total_dg += segment_dg
            total_variance += segment_error ** 2
            chain_segments.append({
                # 注意：这个键是 local_results 的位置下标，不是来源窗口号——有窗口被
                # 跳过时两者会分叉。保持原键以免破坏既有读取方，真实来源见下一行。
                "window_index": int(local_idx),
                "source_window_index": int(
                    local.get("source_window_index", local_idx)
                ),
                "join_lambda_index": int(join_lam),
                "end_lambda_index": int(end_lam),
                "delta_G_kJ_mol": segment_dg,
                "uncertainty_kJ_mol": segment_error,
            })
            covered_lambdas.update(local_lams)
        total_err = float(np.sqrt(total_variance))
        endpoint_error_after_offset = total_err
        
        # ------------------------------------------------------------------
        # 🔑 修复（审查报告 #2）：converged/min_overlap 曾经分别是"每个窗口的
        # MBAR 是否解出来了"和"自由能曲线相邻点间距的单调函数"——前者与统计收敛
        # 无关，后者根本不是重叠度。现在改用每个窗口 compute_effective_sample_
        # number() 算出的真实重加权有效样本比例（见上面窗口内循环），
        # min_overlap = 所有窗口中最差的那个比例；converged 要求全部窗口都成功
        # 解出 *且* 最差重叠比例不低于阈值。旧的 Δf 间距量保留在
        # lambda_spacing_max_step_kJ_mol 里供参考，但不再冒充"重叠度"。
        # ------------------------------------------------------------------
        # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] min_ess_ratio 现在是"扣掉共模因子后的混合
        # 覆盖度"，不再是 raw 单参考 ESS——阈值语义因此才成立（raw 量的上限被防护壳
        # 按 exp(σ_r²) 压到 0.003，任何 5% 量级的阈值对它都是数学上不可满足的）。
        # 任一窗口算不出 min_ess_ratio（f_k 缺失/态数不符）时 ess_ratios 会少一项，
        # 下面 min_overlap 就不是"全部窗口都达标"的证据了——所以额外要求样本齐全。
        ess_ratios = [w["min_ess_ratio"] for w in window_overlap_records if w["min_ess_ratio"] is not None]
        min_overlap = (
            float(np.min(ess_ratios))
            if ess_ratios and len(ess_ratios) == len(window_overlap_records)
            else None
        )
        min_overlap_threshold = float(final_min_ess_ratio)

        # 🔑 [P1 修复] 只检查 ratio 不够——样本总数很少时绝对有效样本数可能只有
        # 个位数，比例却轻易超过阈值。补上去相关样本数、端点不确定度硬门槛，
        # 跟在线 early-stop 用的判据集合一致（不含漂移，一次性拼接没有"上一次
        # 检查"的基线）。
        # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] absolute_ess 从"独立硬门"降级为纯诊断。
        # 它在构造上就是 min_ess_ratio × N_decorrelated（denom 是同一个标量，所以
        # min(neff)/denom == min(neff/denom) 是恒等式），不是第二份独立证据——给
        # 同一个量配两个阈值，效果是让 ratio 那个阈值失去意义：final_min_absolute_ess
        # =50 在 N_decorrelated=114 时等价于要求 ratio ≥ 0.44，而日志里报的门是 0.05，
        # 真实放行条件其实是 ratio ≥ 50/N_decorrelated，且样本越少门越严——跟"延长
        # 采样"的修复方向相反。"比例达标但绝对样本数不够"这件事本来就该由
        # min_decorrelated_samples 直接看（它才是与 ratio 正交的那份证据），端点精度
        # 由 max_endpoint_uncertainty_kJ_mol 直接看。
        # min_absolute_ess 继续算、继续落盘；把 min_absolute_ess_threshold 置 None，
        # 这样 abfe_pipeline.py 里 _assert_stage_result_sane /
        # _stage_quality_failure_details 那两处"只在字段存在时才检查"的镜像判据会
        # 自动失活，无需改动 pipeline 侧逻辑。
        abs_ess_vals = [w["absolute_ess"] for w in window_overlap_records if w.get("absolute_ess") is not None]
        min_absolute_ess = float(np.min(abs_ess_vals)) if abs_ess_vals else None
        raw_ess_vals = [
            w["raw_min_ess_ratio"] for w in window_overlap_records
            if w.get("raw_min_ess_ratio") is not None
        ]
        raw_min_overlap = float(np.min(raw_ess_vals)) if raw_ess_vals else None
        raw_abs_vals = [
            w["raw_min_absolute_ess"] for w in window_overlap_records
            if w.get("raw_min_absolute_ess") is not None
        ]
        raw_min_absolute_ess = float(np.min(raw_abs_vals)) if raw_abs_vals else None
        # [TARGET_SUPPORT_GATE_PROTOCOL_VERSION=1] 权重集中度：每个窗口取它自己
        # 最差（最集中）的那个目标态，再对窗口取最大。与 raw 绝对 ESS 正交——
        # ESS 是二阶矩，抓不住"权重塌缩到个别帧"塌得还不够彻底的中间状态。
        top1pct_per_window = []
        for w in window_overlap_records:
            vals = w.get("top1pct_raw_weight")
            if not vals:
                continue
            finite = [
                float(x) for x in vals
                if x is not None and np.isfinite(float(x))
            ]
            if finite:
                top1pct_per_window.append(float(np.max(finite)))
        max_top1pct_raw_weight = (
            float(np.max(top1pct_per_window)) if top1pct_per_window else None
        )
        # 证据必须**每个窗口都有**才算数：少一个窗口，min/max 就不再是"全部窗口
        # 都达标"的证据（与 min_overlap 那里同一套 fail-closed 逻辑）。
        target_support_evidence_complete = bool(
            len(raw_abs_vals) == len(window_overlap_records)
            and len(top1pct_per_window) == len(window_overlap_records)
            and window_overlap_records
        )
        # 一阶矩伴随诊断：ESS 尺度不变，抓不到"均匀被饿死"的态（实测一个 f_k 未补偿、
        # 高出 80 kT 的态 ESS/N 仍报 0.34）。但 warmup 已明确把 occupancy 定义为只诊断、
        # 不参与 production entry；最终 stage 不能在全部 GPU 采样完成后再用同一指标反向否决。
        occ_vals = [
            w["min_occupancy_normalized"] for w in window_overlap_records
            if w.get("min_occupancy_normalized") is not None
        ]
        min_occupancy_normalized = (
            float(np.min(occ_vals))
            if occ_vals and len(occ_vals) == len(window_overlap_records)
            else None
        )
        min_occupancy_normalized_threshold = float(
            IBS_LOCAL_MBAR_GATE_OCC_MIN_FRACTION
        )
        sigma_vals = [
            w["common_mode_log_sigma_kT"] for w in window_overlap_records
            if w.get("common_mode_log_sigma_kT") is not None
        ]
        max_common_mode_log_sigma_kT = float(np.max(sigma_vals)) if sigma_vals else None

        n_frames_vals = [
            w["n_frames_decorrelated"] for w in window_overlap_records
            if w.get("n_frames_decorrelated") is not None
        ]
        min_decorrelated_samples = int(np.min(n_frames_vals)) if n_frames_vals else 0

        unc_vals = [
            segment["uncertainty_kJ_mol"] for segment in chain_segments
        ]
        max_endpoint_uncertainty_kJ_mol = float(np.max(unc_vals)) if unc_vals else None

        # [v23 numerical follow-up, no protocol bump] These are inclusive
        # physical thresholds. Normalized ESS arithmetic can undershoot an
        # exact boundary by ~1e-13; rejecting 0.9999999999998863 against 1.0
        # (and 0.04999999999999431 against 0.05) is a floating-point bug, not a
        # stricter scientific gate. Only roundoff-scale equality is accepted.
        # ------------------------------------------------------------------
        # [TARGET_SUPPORT_GATE_PROTOCOL_VERSION=1] 无防护壳物理目标的支撑度硬门。
        # 见文件上方 TARGET_SUPPORT_GATE_PROTOCOL_VERSION 的长注释：mixture 覆盖度
        # 把共模因子除掉了，它证明不了"这批轨迹能重加权到真实物理系综"。
        # 证据不全 == 不通过（fail closed），绝不当作通过。
        # ------------------------------------------------------------------
        target_support_min_abs_ess_threshold = float(final_min_target_absolute_ess)
        target_support_max_top1pct_threshold = float(final_max_top1pct_raw_weight)
        target_support_failed: List[str] = []
        if not target_support_evidence_complete:
            target_support_failed.append("incomplete_evidence")
        if raw_min_absolute_ess is None or not np.isfinite(raw_min_absolute_ess):
            target_support_failed.append("raw_absolute_ess_unavailable")
        elif not _meets_minimum_with_roundoff(
            raw_min_absolute_ess, target_support_min_abs_ess_threshold
        ):
            target_support_failed.append("raw_absolute_ess_below_threshold")
        if max_top1pct_raw_weight is None or not np.isfinite(max_top1pct_raw_weight):
            target_support_failed.append("top1pct_raw_weight_unavailable")
        elif not _meets_maximum_with_roundoff(
            max_top1pct_raw_weight, target_support_max_top1pct_threshold
        ):
            target_support_failed.append("top1pct_raw_weight_above_threshold")
        target_support_gate = {
            "protocol_version": int(TARGET_SUPPORT_GATE_PROTOCOL_VERSION),
            "passed": not target_support_failed,
            "failure_reason": (
                "insufficient_target_support" if target_support_failed else None
            ),
            "failed_checks": list(target_support_failed),
            "raw_min_absolute_ess": raw_min_absolute_ess,
            "raw_min_absolute_ess_threshold": target_support_min_abs_ess_threshold,
            "max_top1pct_raw_weight": max_top1pct_raw_weight,
            "max_top1pct_raw_weight_threshold": target_support_max_top1pct_threshold,
            "raw_min_ess_ratio": raw_min_overlap,
            "max_common_mode_log_sigma_kT": max_common_mode_log_sigma_kT,
            "evidence_complete": bool(target_support_evidence_complete),
            "n_windows_with_evidence": len(top1pct_per_window),
            "n_windows": len(window_overlap_records),
            "per_window_raw_min_absolute_ess": [
                w.get("raw_min_absolute_ess") for w in window_overlap_records
            ],
            "per_window_max_top1pct_raw_weight": list(top1pct_per_window),
            "gate_measures": (
                "重加权到**没有 Group-4 防护壳**的物理目标态之后剩下的等效独立样本数"
                "与权重集中度；与受门的 mixture 覆盖度正交（后者把共模因子除掉了）。"
            ),
            "not_sufficient_note": (
                "通过这道门只说明目标态有起码的重要性采样支撑，**不**说明结果正确："
                "docs/archive/STAGE2_ROOT_CAUSE_2026-08-28.md §3.2 的 window 2 相邻 <dU> 仅 0.4~0.6 kT、"
                "任何能量重叠判据都判优，却错得最多，因为失效模式是结构性的"
                "（'该采的构型一次都没采到'），不是'采到的构型之间散布不够'。"
                "[2026-09-02] 该窗口原先标注的 +19.49 kJ/mol 是壳还在时测的幅度，"
                "其中 98% 已定罪为 lambda-WCA 防护壳并随壳退役消失；教训本身不变。"
                "根治见 docs/archive/BUG_LOCATION_stage2_ibs_window0_shell_2026-09-01.md。"
            ),
        }

        # ==================================================================
        # analysis_status —— **只用硬不变量**，一个拟合阈值都不许进来。
        # 判据与 provenance 见本函数 docstring。
        # ==================================================================
        analysis_incomplete_reasons: List[str] = []

        # (1) 路径完整
        if len(local_results) != len(valid_windows):
            analysis_incomplete_reasons.append(
                f"路径缺窗：只解出 {len(local_results)} 个窗口，输入的有效窗口有 "
                f"{len(valid_windows)} 个（差额在求解器内被 continue 掉了）。"
            )
        if skipped_windows:
            analysis_incomplete_reasons.append(
                "存在被跳过的窗口 "
                f"{[w.get('window_index') for w in skipped_windows]}；"
                "缺窗口的总和是部分和，不是完整 ΔG。"
            )

        # (2) 数值有限
        if not np.isfinite(total_dg) or not np.isfinite(total_err):
            analysis_incomplete_reasons.append(
                f"总量非有限：total_delta_G={total_dg!r}, total_error={total_err!r}。"
            )
        for _seg in chain_segments:
            if not (
                np.isfinite(_seg["delta_G_kJ_mol"])
                and np.isfinite(_seg["uncertainty_kJ_mol"])
            ):
                analysis_incomplete_reasons.append(
                    f"协方差链段 window_index={_seg['window_index']} 的 ΔG/σ 非有限："
                    f"{_seg['delta_G_kJ_mol']!r} / {_seg['uncertainty_kJ_mol']!r}。"
                )

        # (3) 结构自洽：λ 覆盖连续无洞 + 相邻窗恰好共享一个边界态
        _cov = sorted(int(x) for x in covered_lambdas)
        if _cov and _cov != list(range(_cov[0], _cov[-1] + 1)):
            _holes = sorted(set(range(_cov[0], _cov[-1] + 1)) - set(_cov))
            analysis_incomplete_reasons.append(
                f"λ 覆盖不连续：区间 [{_cov[0]}, {_cov[-1]}] 内缺 λ 索引 {_holes}。"
            )
        for _i in range(1, len(local_results)):
            _prev = set(int(x) for x in local_results[_i - 1]["lambdas"])
            _cur = set(int(x) for x in local_results[_i]["lambdas"])
            _shared = sorted(_prev & _cur)
            if len(_shared) != 1:
                analysis_incomplete_reasons.append(
                    f"相邻窗口 {_i - 1}/{_i} 共享 {len(_shared)} 个 λ 态 {_shared}，"
                    "要求恰好 1 个边界态（0 个=断链，>1 个=区间重叠会被重复积分）。"
                )

        analysis_status = (
            "ANALYSIS_INCOMPLETE" if analysis_incomplete_reasons else "ANALYSIS_COMPLETE"
        )

        # precision —— 单次求解结构上只有**一个**重复，传空列表拿 UNMEASURED。
        # 如实这么写：不伪造、也不跳过这个字段。真正的评估在循环级。
        precision_evidence = cross_repeat_precision([])
        precision_evidence["scope_note"] = (
            "单次 solve_stage_integrated 只看得见一个 run，结构上无法测跨重复离散度，"
            "所以这里恒为 UNMEASURED。真正的精度评估在循环级（收集 ≥"
            f"{precision_evidence.get('min_independent_repeats')} 个独立重复的 ΔG 之后）。"
        )
        precision_status = precision_evidence["status"]
        lambda_spacing_max_step = float(np.max(np.abs(np.diff(f_curve)))) if f_curve.size > 1 else 0.0

        # 🔑 [P0-8] 把"这条曲线到底覆盖了哪些窗口/哪段 λ"落盘。loader 侧
        # (_assert_expected_windows_all_loaded) 已经挡住了"预期窗口没产出数据"这个
        # 入口，但求解器自己还会因维度不符/帧数不足 `continue` 掉窗口——那部分由
        # `len(local_results) == len(valid_windows)` 抓。这里记录的是审计证据：
        # 光看 total_delta_G 无法判断它积分的是不是完整的 λ 区间。
        input_window_indices = sorted(
            int(w.get("window_index", i)) for i, w in enumerate(window_data)
        )
        solved_window_indices = sorted(
            int(r.get("source_window_index", i)) for i, r in enumerate(local_results)
        )
        covered_lambda_indices = sorted(int(x) for x in covered_lambdas)
        coverage_diagnostics = {
            "input_window_indices": input_window_indices,
            "valid_window_indices": sorted(
                int(w.get("window_index", i)) for i, w in enumerate(valid_windows)
            ),
            "solved_window_indices": solved_window_indices,
            "dropped_window_indices": sorted(
                set(input_window_indices) - set(solved_window_indices)
            ),
            "covered_lambda_indices": covered_lambda_indices,
            "covered_lambda_index_first": (
                covered_lambda_indices[0] if covered_lambda_indices else None
            ),
            "covered_lambda_index_last": (
                covered_lambda_indices[-1] if covered_lambda_indices else None
            ),
            "n_covered_lambda_indices": len(covered_lambda_indices),
            "note": (
                "求解器只能证明它积分了这些 λ 索引，无法证明它们就是该 stage 的完整"
                "路径——'预期窗口是否全部产出数据'必须在 loader 出口用 "
                "_assert_expected_windows_all_loaded 判定。"
            ),
        }

        return {
            "coverage_diagnostics": coverage_diagnostics,
            "total_delta_G": total_dg,
            "total_error": total_err,
            "endpoint_error_after_offset": endpoint_error_after_offset,
            "covariance_chain_segments": chain_segments,
            "total_error_method": (
                "independent_window_segment_variances_using_direct_dDelta_f"
            ),
            "offset_error_contribution": float(np.sqrt(np.sum(offset_var_terms))) if offset_var_terms else 0.0,
            # 🔑 [2026-09-15] `converged` 已删除（**不是改义**）。读它的地方会拿到
            # None —— 那是故意的 fail-loud。见本函数 docstring。
            "analysis_status": analysis_status,
            "analysis_incomplete_reasons": list(analysis_incomplete_reasons),
            # 🔑 [2026-09-15] 哪些窗口的帧是在**未经验证**的冻结 f_k 下采的
            # （临时生产）。非空 ⟹ 这条路径完整、数值有限，但**不是可信 PASS**。
            # 刻意**不**进 `analysis_status` 的合取：路径完整性与 f_k 证据强度
            # 是两维，混起来会让「分析跑完整了」这个硬不变量说别的事。
            "provisional_production_windows": sorted(
                int(w.get("window_index", i))
                for i, w in enumerate(valid_windows)
                if w.get("provisional_production")
            ),
            "precision_status": precision_status,
            "precision_evidence": precision_evidence,
            "diagnostics_are_not_acceptance": (
                "min_overlap / target_support_gate / min_decorrelated_samples / "
                "max_endpoint_uncertainty_kJ_mol 及各自的阈值、chain_segments 逐段 σ —— "
                "这些量是『求解器能不能工作 / 重加权有没有明显退化』的最低条件，"
                "**不是精度判据**，不参与 analysis_status 的任何合取。"
                "1.0 kJ/mol 那条的来历（抄自默认关闭的 early-stop 启发式、全仓无依据、"
                "逐段量而目标是两腿合成量）与为什么不能当门（MBAR σ 是渐近估计，"
                "且对『该采的构型一次都没采到』失明：4W53 +32、decharging −88 两次都没报警），"
                "见 solve_stage_integrated 的 docstring。"
            ),
            "min_overlap": min_overlap,
            "min_overlap_threshold": min_overlap_threshold,
            "min_overlap_method": (
                "per_window_mixture_coverage_ess_ratio_common_mode_removed"
            ),
            "ess_gate_protocol_version": int(ESS_GATE_PROTOCOL_VERSION),
            "min_occupancy_normalized": min_occupancy_normalized,
            # occupancy 与 warmup 一致，只报告、不设门；保留 0.5 作为诊断参考线。
            "min_occupancy_normalized_threshold": None,
            "min_occupancy_normalized_diagnostic_reference": min_occupancy_normalized_threshold,
            "min_occupancy_is_gate": False,
            "min_occupancy_gate_retired_reason": (
                "warmup 明确将 occupancy 定义为 diagnostics-only；最终 stage 不得在生产"
                "全部完成后用同一指标反向否决已放行的 ensemble。"
            ),
            "min_absolute_ess": min_absolute_ess,
            # None = 该项已降级为纯诊断，见上面 ESS_GATE_PROTOCOL_VERSION=2 的说明。
            # 置 None 同时让 abfe_pipeline 侧两处镜像检查自动失活。
            "min_absolute_ess_threshold": None,
            "min_absolute_ess_gate_retired_reason": (
                "absolute_ess == min_ess_ratio * n_frames_decorrelated 恒等，不是独立"
                "证据；给同一个量配两个阈值会让 ratio 阈值失效（final_min_absolute_ess"
                f"={float(final_min_absolute_ess):.4g} 在 N_decorrelated=114 时等价于"
                "要求 ratio>=0.44）。改由 min_decorrelated_samples 与 "
                "max_endpoint_uncertainty_kJ_mol 分别承担样本量与精度这两份正交证据。"
            ),
            # ---- 只报告、不设门：防护壳/LRC 共模因子收了多少重加权税 ----
            "raw_min_overlap": raw_min_overlap,
            "raw_min_absolute_ess": raw_min_absolute_ess,
            "max_common_mode_log_sigma_kT": max_common_mode_log_sigma_kT,
            # ---- [TARGET_SUPPORT_GATE_PROTOCOL_VERSION=1] 受门：物理目标支撑度 ----
            # raw_* 从"只报告不设门"升级为硬门，理由与实测数据见文件上方
            # TARGET_SUPPORT_GATE_PROTOCOL_VERSION 的注释。
            "target_support_gate_protocol_version": int(
                TARGET_SUPPORT_GATE_PROTOCOL_VERSION
            ),
            "target_support_gate": target_support_gate,
            "raw_min_absolute_ess_threshold": target_support_min_abs_ess_threshold,
            "max_top1pct_raw_weight": max_top1pct_raw_weight,
            "max_top1pct_raw_weight_threshold": target_support_max_top1pct_threshold,
            "min_decorrelated_samples": min_decorrelated_samples,
            "min_decorrelated_samples_threshold": int(final_min_decorrelated_samples),
            "max_endpoint_uncertainty_kJ_mol": max_endpoint_uncertainty_kJ_mol,
            "max_endpoint_uncertainty_kJ_mol_threshold": float(final_max_uncertainty_kJ_mol),
            "lambda_spacing_max_step_kJ_mol": lambda_spacing_max_step,
            "window_overlap_diagnostics": window_overlap_records,
            # **与 warmup gate 的那份分开**：scope=production，帧集/分段/f_k 指纹
            # /口径/路径版本全部自带，不许互相覆盖。
            "cumulative_fk_residual_production": cumulative_fk_records,
            # 🔑 被跳过的窗口必须显式出现在结果里。此前只有一行 WARN，而
            # total_delta_G 照样按"完整路径"返回：实测踩过一次，段 1 的 w3 因去
            # 相关后只剩 5 帧被跳过，86.42 是个部分和，却被跟段 2 的 52.75 并排
            # 打印成"两次采样差 33.7 kJ/mol"。缺窗口的总和不是完整 ΔG。
            "skipped_windows": list(skipped_windows),
            "path_is_complete": not bool(skipped_windows),
            "statistical_inefficiency_per_window": window_g_values,
            "method": "Local-TMBAR covariance-chain (ESS-overlap-checked)",
            "uncertainty_note": (
                "总误差对每个独立窗口直接读取连接态到新增端点的 dDelta_f，并合并独立"
                "窗口段方差；同一窗口内的端点协方差因此不会重复丢失。不同窗口来自独立"
                "采样，未假设跨窗口协方差。⚠️ 这个 σ **不是**接受判据：它是 MBAR 的"
                "渐近估计，对『该采的构型一次都没采到』失明（见 "
                "diagnostics_are_not_acceptance）。"
            ),
            "f_k": f_curve.tolist(),
            "df_k": err_curve.tolist(),
            "lambdas": sorted_lams
        }

    def _fallback(self, msg: str) -> Dict:
        # 🔑 [2026-09-15] 与成功路径同口径：`converged` 已删除，改报 analysis_status。
        # 这里必然是 ANALYSIS_INCOMPLETE —— 求解器根本没解出一条完整路径。
        return {
            "error": msg,
            "analysis_status": "ANALYSIS_INCOMPLETE",
            "analysis_incomplete_reasons": [f"求解器提前退出：{msg}。"],
            "precision_status": "UNMEASURED",
            "precision_evidence": cross_repeat_precision([]),
            "total_delta_G": 0.0,
            "total_error": 999.9,
        }

    def _incomplete_path_fallback(self, msg: str, skipped_windows) -> Dict:
        """拼接失败的出口：**必须把「哪些窗口被跳过」带出去。**

        这几个 reason（`window_overlap_broken` / `no_local_tmbAR_results` / …）
        的成因只有一个 —— 有窗口去相关后帧数不足、在生成任何诊断之前就被
        `continue` 掉，于是拼接链缺共享 λ 节点。而 `_fallback()` 只返回四个键，
        `skipped_windows` 全丢 ⟹ 下游（自治控制器）看到的是「路径不完整，但不知道
        缺谁」，它那条「被踢出协方差链 ⟹ 补采」的分支因此**不可达**，只能对着
        同一个窗口反复 ANALYZE 直到停滞保护退出（真机 2026-09-14）。

        成功路径在 22019 附近本来就带这两个键，这里只是让失败路径口径一致。
        """
        out = self._fallback(msg)
        out["skipped_windows"] = list(skipped_windows or [])
        out["path_is_complete"] = not bool(skipped_windows)
        return out

# 模块级便捷入口
