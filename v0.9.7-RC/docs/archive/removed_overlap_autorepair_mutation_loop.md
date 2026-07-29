# 已移除：`_run_stage_with_overlap_autorepair` 的 ensemble 变异自动修复循环

移除日期：2026-07-27　　对应待办：**E-03** / **ATT-27**

## 为什么移除

这段代码位于 `abfe_pipeline.py::_run_stage_with_overlap_autorepair` 里
**无条件 `return` 之后**，已经不可达。它自己的横幅注释（保留在源文件中那条
`[deprecated_non_mutating_policy]`）写明了原因：

> IBS 通过把**一个**冻结的积分混合分布重加权到各目标态来取 ΔG，
> **相邻态 fixed-H overlap 根本不是它的正确性判据**。

把 adjacent overlap 当成 IBS 收敛的仲裁标准是一个设计错误，它触发的自动拆窗 /
插 λ / `recalibrate_f_k` 循环曾**烧掉约一周 GPU 而没产出任何 ΔG**。
`non_mutating_v1` 策略取代了它：stage 只跑一次，f_k 收敛在
`ibs_engine.run_all_windows` 内部验证，失败就 `IBSWarmupConvergenceError`
抛出交人工/rescue 审计，绝不变异 ensemble。

原注释里承诺的是 “kept verbatim for review and will be excised in a separate
change” —— 本文件就是那次 excise 的归档。

## 为什么不留在源文件里

E-03 的要求是「保留历史可读性时放入归档文档，**不得保留可被误激活的可执行代码**」。
881 行不可达逻辑留在一个 7900+ 行的主模块里有两个实际代价：

1. 每次阅读 `_run_stage_with_overlap_autorepair` 都要先确认它不可达；
2. 只要有人把上面那个 `return` 挪走或加一层分支，整套已被否决的变异逻辑就会
   悄悄复活 —— 而它烧过一周 GPU。

现存的守护：`test_non_mutating_policy.py`、`test_audit_protocol_regressions.py`
断言 `non_mutating_v1` 策略与调用点行为；`test_att27_dead_code_removed.py`
断言这段代码不会被搬回来。

## 移除的内容（881 行，`abfe_pipeline.py` 原 5434-6314 行）

以下为**非可执行**归档，仅供查阅。不要复制回生产代码。

```python
        # 🔑 熔断器：加密只应该在"确实能改善重叠"的前提下继续。之前发现的真实
        # 案例是 min_overlap 一路 0.01553→0.01948→0.01328→0.01266→0.007631→
        # 0.007973→0.003479，不是收敛趋势，是噪声里夹杂着系统性变差——真正的
        # 瓶颈（IBS 偏置未收敛）不会被插点修好，continue 只会一轮轮重复同样的
        # 失败还烧更多 GPU 时间。一旦某一轮加密后 min_overlap 没有改善（含打平
        # 或变差），立即停止，不再等到 max_repair_rounds 耗尽。
        previous_min_overlap = None
        # 🔑 window_idx -> 生产步数覆盖，供 reseed_resample 真正"延长"采样（而不是
        # 只是删旧样本用同样步数重采）。只在纯 sampling-repair 轮（λ 路径不变）
        # 里被写入/消费，见 _diagnose_and_repair_all_pass_low_ess_window 调用点；
        # 键是这一轮 effective_old_ranges 里的 window_idx，只有在写入它的那一轮和
        # 紧接着消费它的下一轮之间 λ 路径确实没变时才有效——这正是
        # path_will_change 门控保证的前提。
        pending_step_overrides: Dict[int, int] = {}
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] window_idx -> 这个窗口冻结验证的
        # 累计目标预算（50k/150k/300k 阶梯）。只在 IBSFrozenCalibrationValidationError
        # 诊断报告 calibration_pending_validation=True 时写入/消费——不同于
        # pending_step_overrides（服务生产 ESS 的 reseed_resample），这个只服务
        # "MBAR 校准好但冻结验证没在累计预算内通过"这一种失败模式。
        frozen_validation_step_overrides: Dict[int, int] = {}
        # 🔑 [四处修复之一] window_idx -> 下一次调用时这个窗口的目标预算是否已经
        # 是阶梯（50k/150k/300k）的最后一档——true 时若仍未通过，
        # ibs_engine.py::run_all_windows 会把该窗口判定为终态失败
        # （calibrated_validation_failed），不会再落盘成"pending"。跟
        # frozen_validation_step_overrides 一样按 window_idx 累积，同一个窗口
        # 一旦被标记过就不会撤销（阶梯只会前进不会后退）。
        frozen_validation_is_final_rung: Dict[int, bool] = {}
        # 🔑 reseed_resample 每次触发时，把该窗口下一轮的步数在当前覆盖倍数上
        # 乘以这个增长因子，封顶下面这个最大倍数——避免一个持续不收敛的窗口
        # 无界烧 GPU；如果封顶后仍不收敛，交给下一轮的 max_repair_rounds 熔断
        # 或人工检查，而不是继续无限加码。
        resample_step_growth_factor = 2.0
        max_resample_step_multiplier = 4.0

        # 🔑 [ladder 独立预算修复] 之前用 for attempt in range(max_repair_rounds+1)，
        # 冻结验证阶梯的 continue（下面 except IBSFrozenCalibrationValidationError
        # 分支）跟拆窗/插λ/production ESS 修复的 continue 共用同一个 attempt 计数器
        # ——阶梯自己的耗尽判据（schedule_idx+1>=len(schedule)）虽然不再依赖
        # attempt>=max_repair_rounds，但如果本轮之前的拆窗/插λ/ESS 修复已经消耗
        # 了大部分共享迭代次数，阶梯可能还没跑到 300k 这一档，for 循环就已经耗尽，
        # 落到下面"自动修复循环异常退出"的兜底错误——阶梯并未真正获得独立预算。
        # 现在改成 while True + 独立的 repair_round 计数器：只有拆窗/插λ/production
        # ESS 修复才会递增 repair_round（语义与之前的 attempt 完全一致，
        # >=max_repair_rounds 时终止），冻结验证阶梯的 continue 完全不触碰它，
        # 阶梯的进退真正只由它自己的 3 档 schedule 长度决定。
        repair_round = 0
        while True:
            try:
                result = run_once(
                    current_n_states, current_lambdas, current_ranges,
                    dict(pending_step_overrides) if pending_step_overrides else None,
                    dict(frozen_validation_step_overrides) if frozen_validation_step_overrides else None,
                    dict(frozen_validation_is_final_rung) if frozen_validation_is_final_rung else None,
                )
            except IBSFrozenCalibrationValidationError as calib_exc:
                # 🔑 [四处修复] 这个 except 分支必须独立于下面
                # `except IBSWarmupConvergenceError` 用的 attempt>=max_repair_rounds
                # 门槛，且必须先于它检查（两者是同级 RuntimeError 子类，不是父子
                # 类型，顺序由代码本身决定，不是异常类型系统自动决定的）。之前的
                # bug：calibration_pending_validation 分支写在
                # `except IBSWarmupConvergenceError` 内部、`attempt>=max_repair_rounds`
                # 判断之后——如果这一轮之前的拆窗/插λ/production ESS 修复已经把
                # 共享的 attempt 计数器耗尽，冻结验证阶梯（本身只有 3 档、有自己
                # 独立的耗尽判据）从未获得执行机会，会直接把原始异常原样抛出，
                # 根本到不了"延长预算续验"或"封顶转人工检查"的分支。现在冻结
                # 验证阶梯的进退完全由它自己的 schedule 长度决定，不受拆窗/插λ/
                # production ESS 修复轮数影响。
                diagnostics = calib_exc.diagnostics
                if calib_exc.terminal:
                    # ibs_engine.py 已经把这个窗口判定为终态失败
                    # （bias_status="calibrated_validation_failed"）并带着
                    # terminal=True 抛出——不再自动重试，直接向上传播这个
                    # 语义清晰的异常（不是被误当成"偏置预热失败"的
                    # IBSWarmupConvergenceError，也不再被拆窗/插λ的逻辑捕获）。
                    raise
                window_idx = diagnostics.get("window_index")
                # 🔑 [跨文件单一数据源修复] 之前这里自己硬编码一份 (50_000, 150_000,
                # 300_000)，跟 ibs_engine.py 内部用于"调用方未提供覆盖字典时"的
                # 阶梯回退逻辑各自维护一份 tuple——两处必须永远保持一致，否则两侧
                # 对"第几档""是否最后一档"的理解会不一致。现在改成从 ibs_engine.py
                # 导入同一个常量，只有一处定义。
                schedule = FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS
                prev_budget = frozen_validation_step_overrides.get(window_idx, schedule[0])
                try:
                    schedule_idx = schedule.index(prev_budget)
                except ValueError:
                    schedule_idx = 0
                if schedule_idx + 1 >= len(schedule):
                    # 阶梯理应已经在上一轮把这个窗口标记为最后一档
                    # （frozen_validation_is_final_rung[window_idx]=True），
                    # ibs_engine.py 那一侧应该已经带着 terminal=True 抛出、被
                    # 上面的分支处理掉，不应该走到这里。真的走到这里说明两侧
                    # 阶梯状态不一致，直接兜底报错，不静默重试、也不假装还能
                    # 继续延长。
                    raise RuntimeError(
                        f"窗口 {window_idx} 的冻结验证阶梯已经在 {schedule[-1]} 步耗尽，"
                        "但收到的异常未标记为 terminal——ibs_engine.py 与 "
                        "abfe_pipeline.py 的阶梯状态不一致，需要人工检查（不应该发生，"
                        "属于代码 bug 而不是正常的采样失败）。"
                    ) from calib_exc
                next_budget = schedule[schedule_idx + 1]
                frozen_validation_step_overrides[window_idx] = next_budget
                is_next_final = (schedule_idx + 1 == len(schedule) - 1)
                frozen_validation_is_final_rung[window_idx] = is_next_final
                self._log(
                    f"  ⏳ {stage_label}: 窗口 {window_idx} 的 MBAR 校准 f_k 冻结验证在累计 "
                    f"{prev_budget} 步预算内未通过，延长累计预算至 {next_budget} 步续验"
                    + ("（这是最后一档，若仍不通过将判定为终态失败 calibrated_validation_failed，"
                       "不再自动重试）" if is_next_final else "")
                    + "（不回 SGD、不重跑 fixed-H overlap/校准探针，只续验累计预算里还没跑完的"
                    "差值，不是重新烧一遍）。"
                )
                continue
            except IBSWarmupConvergenceError as warmup_exc:
                if stage_name != "vanishing" or repair_round >= max_repair_rounds:
                    raise
                cached_preopt = {}
                if os.path.exists(preopt_file):
                    with open(preopt_file, "r") as f:
                        cached_preopt = json.load(f)
                effective_old_ranges = current_ranges or generate_overlapping_windows(
                    current_n_states, pts_per_window=6, overlap=2
                )
                diagnostics = warmup_exc.diagnostics
                # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=7] 边界条件曾经写错：
                # split_window_from_warmup_failure 要求两个孩子各自至少 3 个态、
                # 共享 1 个态，父窗口因此至少需要 3+3-1=5 个态才能这样拆；旧代码
                # 用 >=4 判断"能不能拆"，导致 K=4 的窗口（如 [2,6)）被拆成
                # K=2+K=3（如 [2,4)+[3,6)），产出一个明知统计脆弱的两态窗口。
                # K=4 现在和 K<=3 一样，直接走下面的 fixed-H 双向 overlap 探针，
                # 不再盲拆。
                if int(diagnostics.get("n_states", 0)) >= 5:
                    new_lambdas, new_ranges, feedback = split_window_from_warmup_failure(
                        current_lambdas, effective_old_ranges, diagnostics
                    )
                    self._log(
                        f"  ✂️ {stage_label}: warmup 在窗口 "
                        f"{diagnostics.get('window_index')} coverage 失败；不插 λ，"
                        f"只拆为 {feedback['child_ranges']}，共享旧态 "
                        f"{feedback['shared_global_state']}"
                    )
                else:
                    fixed_probe = diagnostics.get("bidirectional_overlap_probe", {})
                    if not fixed_probe.get("pairs"):
                        raise RuntimeError(
                            "最小 IBS 窗口 coverage 失败，但缺少 fixed-H 双向 overlap 诊断；"
                            "拒绝回退到 Delta-u/算术二分插点。"
                        ) from warmup_exc
                    asymmetric = fixed_probe.get("passed_but_asymmetric_bottleneck")
                    if bool(fixed_probe.get("all_passed", False)) and not (
                        asymmetric and asymmetric.get("qualified")
                    ):
                        raise RuntimeError(
                            "最小 IBS 窗口 coverage 失败，但所有相邻 fixed-H 双向 overlap 均已通过，"
                            "且没有检测到显著局部热力学瓶颈；拒绝自动插点。"
                        ) from warmup_exc
                    new_lambdas, new_ranges, feedback = insert_lambda_from_overlap_failure(
                        current_lambdas, effective_old_ranges, diagnostics
                    )
                    self._log(
                        f"  🧪 {stage_label}: 最小窗口 fixed-H 双向 overlap="
                        f"{feedback['measured_min_bidirectional_overlap']:.5f} < "
                        f"{feedback['overlap_threshold']:.5f}；仅在实测失败边 "
                        f"{feedback['failed_global_edge']} 插入待重测 λ="
                        f"{feedback['inserted_lambda']:.8f}"
                    )
                # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=7] 这条 warmup 失败修复
                # 路径此前直接落盘 new_ranges，从未 canonicalize 过——split 只替换
                # 失败的父窗口，未拆的旧邻窗原样保留，产出的孩子完全可能被邻窗
                # 严格包含（真实案例：[2,4) 完全落在旧邻窗 [3,9) 里，一次采样都是
                # 白跑）。落盘前统一归约一次，跟 production ESS 分支保持一致。
                new_ranges = canonicalize_window_ranges(new_ranges, len(new_lambdas))
                self._invalidate_stage_window_files(
                    stage_name,
                    stage_type,
                    old_lambdas=current_lambdas,
                    old_ranges=effective_old_ranges,
                    new_lambdas=new_lambdas,
                    new_ranges=new_ranges,
                )
                # 🔑 同上（见下面 production ESS 拆窗/插 λ 分支）：warmup coverage
                # 失败触发的拆窗/插 λ 同样重排 window_idx，旧的生产步数覆盖必须
                # 清空，不能带着旧编号进入下一轮。
                if pending_step_overrides:
                    self._log(
                        f"  🧹 {stage_label}: warmup coverage 修复改变了窗口编号，"
                        f"清空 {len(pending_step_overrides)} 条旧的生产步数覆盖 "
                        f"{sorted(pending_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                    )
                    pending_step_overrides.clear()
                # 🔑 [window_idx 陈旧覆盖修复] frozen_validation_step_overrides/
                # frozen_validation_is_final_rung 跟 pending_step_overrides 一样按
                # window_idx 写入，同样会被这里的拆窗/插 λ 重排废掉——不清空的话，
                # 重排后编号恰好相同的新窗口会直接继承旧窗口已经烧到的阶梯预算
                # （甚至"已是最后一档"标记），从未做过冻结验证就被判定终态失败。
                if frozen_validation_step_overrides or frozen_validation_is_final_rung:
                    self._log(
                        f"  🧹 {stage_label}: warmup coverage 修复改变了窗口编号，"
                        f"清空 {len(frozen_validation_step_overrides)} 条旧的冻结验证阶梯覆盖 "
                        f"{sorted(frozen_validation_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                    )
                    frozen_validation_step_overrides.clear()
                    frozen_validation_is_final_rung.clear()
                path_diagnostics = dict(cached_preopt.get("path_diagnostics", {}))
                if feedback.get("thermodynamic_lengths_invalidated", False):
                    # Do not repeat the old 0.5L + 0.5L fiction.  The inserted
                    # coordinate is a hypothesis and its two new edges have no
                    # thermodynamic length until they are sampled.
                    path_diagnostics.pop("optimized_edge_thermodynamic_lengths", None)
                    path_diagnostics["requires_pilot_remeasurement"] = True
                path_diagnostics.setdefault("warmup_feedback_history", []).append(feedback)
                os.makedirs(os.path.dirname(preopt_file), exist_ok=True)
                with open(preopt_file, "w") as f:
                    json.dump({
                        "lambdas_var": new_lambdas,
                        "window_ranges": [list(r) for r in new_ranges],
                        "n_states": len(new_lambdas),
                        "protocol_key": preopt_protocol_key,
                        "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
                        "path_diagnostics": path_diagnostics,
                        "provenance": feedback,
                    }, f, indent=2)
                current_lambdas = new_lambdas
                current_ranges = new_ranges
                current_n_states = len(new_lambdas)
                repair_round += 1
                continue

            if not self._is_overlap_failure(result):
                # 要么通过，要么失败原因不是重叠（NaN/求解失败等）——两种情况
                # 都不该走自动修复，直接交给 _assert_stage_result_sane 决定
                # （通过就直接返回，失败就按原有语义硬性报错）。
                self._assert_stage_result_sane(stage_label, result)
                return result, current_n_states, current_lambdas, current_ranges

            min_overlap = result.get("min_overlap")
            threshold = result.get("min_overlap_threshold")
            diagnostics = result.get("window_overlap_diagnostics")

            # This stage-wide circuit breaker only makes sense for the legacy
            # arithmetic-midpoint path (probe_window_overlap_fn is None): there,
            # every repair round bisects the single worst lambda edge across the
            # whole path, so a non-improving global min_overlap really does mean
            # continued bisection is unlikely to help. The new split-first /
            # fixed-H-probe path (below) can legitimately NOT improve the
            # stage-wide worst ESS on a round that only split a large window
            # (splitting doesn't insert any lambda, so it isn't expected to
            # move the global minimum yet) or on a round that fixed one
            # genuine gap while a different, not-yet-processed window still
            # holds the global worst value. That path already has its own
            # fail-closed gates (an all-passed fixed-H probe or a missing
            # probe result raises immediately), so it doesn't need this
            # cross-round, cross-window comparison to stay safe.
            if probe_window_overlap_fn is None and (
                previous_min_overlap is not None
                and min_overlap is not None
                and min_overlap <= previous_min_overlap
            ):
                raise RuntimeError(
                    f"{stage_label} 阶段自动加密 λ 路径未能改善重叠度：上一轮 "
                    f"min_overlap={previous_min_overlap:.4g}，本轮加密后是 {min_overlap:.4g}"
                    f"（阈值 {threshold:.4g}，未改善或变差）。继续插点不太可能修好它——"
                    "这通常说明真正的瓶颈不是 λ 密度不够，而是 IBS 偏置未收敛/构象弛豫"
                    "过慢/restraint 有问题，请检查 window_overlap_diagnostics 与各窗口"
                    "convergence.json 里的 bias_warmup 状态，而不是继续自动加密 λ 硬跑。"
                )
            previous_min_overlap = min_overlap

            if repair_round >= max_repair_rounds:
                raise RuntimeError(
                    f"{stage_label} 阶段 min_overlap={min_overlap:.4g} 低于阈值 {threshold:.4g}，"
                    f"自动加密 λ 路径并重新采样已连续尝试 {max_repair_rounds} 轮仍未通过。"
                    "这通常说明问题不是 λ 密度不够，而是采样本身有结构性问题（IBS 偏置未收敛、"
                    "构象陷阱、restraint 不一致等），请人工检查 window_overlap_diagnostics 与"
                    "bias_warmup 状态，而不是继续加密 λ 硬跑。"
                )

            effective_old_ranges = current_ranges or generate_overlapping_windows(
                current_n_states, pts_per_window=6, overlap=2
            )

            if probe_window_overlap_fn is not None:
                # Unified with the warmup-failure branch above: a whole window's
                # low ESS is not evidence a specific lambda edge is too wide (a
                # saturated bias/slow relaxation depresses every state equally),
                # so failing windows are only ever split here; a real lambda
                # insertion still requires a measured fixed-H overlap gap.
                to_split, to_probe = plan_vdw_overlap_repair_targets(
                    effective_old_ranges, diagnostics, threshold, min_states_before_split=5,
                )
                if not to_split and not to_probe:
                    raise RuntimeError(
                        f"{stage_label} 阶段 min_overlap={min_overlap:.4g} 低于阈值 {threshold:.4g}，"
                        "但自动修复逻辑未能从 window_overlap_diagnostics 里定位到需要处理的窗口"
                        "（lambdas/min_ess_ratio 明细缺失，或窗口范围对不上当前方案），拒绝盲目重试。"
                    )

                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=7] 之前只按"失败窗口占比 > 50%"硬停止,
                # 太粗糙：(a) 从不实际读取每个失败窗口自己的 warmup 是否真的通过了
                # frozen validation,只是假设"进了 production 就等于 converged 属实";
                # (b) 一个局部坏边可能同时污染两个重叠窗口——总共只有 3 个窗口时,
                # 2/3 已经超过 50%,但根因仍可能只是同一处局部 λ gap,不该被误判成
                # 全局问题。改成两步更精确的判据：
                #   1) 先核实每个失败窗口自己的 convergence.json 是否真的记录了
                #      bias_warmup.status == "frozen_validation_converged"——不确认
                #      直接硬停止,不能把"warmup JSON 写着 converged"当充分证据。
                #   2) 把失败窗口按全局 λ 区间重叠关系分组：IBS 相邻窗口按设计一定
                #      重叠,所以同一段连续失败的窗口自然会被分进同一个 connected
                #      component；只有当失败窗口分散在多个互不相邻的区域时,才说明
                #      这不是某一处局部 gap,而更可能是全局采样协议问题,才硬停止。
                #      单一 connected component（哪怕包含全部窗口）仍按局部问题处理，
                #      继续走下面的拆窗/probe 逻辑。
                failing_windows = to_split + to_probe
                # 🔑 warmup 冻结验证成功有两种落盘状态：SGD 学习本身收敛的
                # "frozen_validation_converged"，以及 fixed-H overlap 全通过后用
                # BAR/MBAR 校准 f_k 再验证通过的"frozen_validation_converged_
                # after_mbar_calibration"（见 ibs_engine.py run_all_windows）。
                # 之前只认第一种字面量，任何被 MBAR 校准修好的窗口都会在这里被
                # 误判成"未确认收敛"而硬停止——必须两者都算作已确认收敛。
                valid_bias_warmup_statuses = {
                    "frozen_validation_converged",
                    "frozen_validation_converged_after_mbar_calibration",
                }
                unvalidated = [
                    (se, self._load_window_bias_warmup_status(stage_name, stage_type, effective_old_ranges, se))
                    for se in failing_windows
                ]
                unvalidated = [(se, status) for se, status in unvalidated if status not in valid_bias_warmup_statuses]
                if unvalidated:
                    raise RuntimeError(
                        f"{stage_label}: 窗口 {unvalidated} 的 convergence.json 里 bias_warmup.status "
                        f"不属于 {sorted(valid_bias_warmup_statuses)}（或读取失败/缺失）——无法确认这些"
                        "窗口自身的偏置真的通过了冻结验证，也就无法判断 production ESS 低究竟是局部 "
                        "λ 密度问题还是 warmup/IBS 偏置协议本身有系统性缺陷，拒绝继续自动拆窗/插点。"
                    )
                # 🔑 之前"失败窗口分散在多个互不相邻区域"直接硬停止在这里执行，
                # 早于下面逐窗口的 fixed-H 分类（哪些窗口真正 fixed-H 失败、哪些
                # 全通过只是 production ESS/f_k 问题）。这个全局判断和"逐窗口独立
                # 分类、互不一票否决"的设计冲突：区域分散本身不能证明是系统性
                # 协议问题，也可能只是恰好有多处独立的局部 λ gap，或多处独立的
                # f_k/采样问题，都能被下面的逐窗口分类正确处理。降级为警告，不再
                # 在探针之前全局拦截；仍然打印出来供人工关注。
                failure_components = self._merge_overlapping_ranges_into_components(failing_windows)
                if len(failure_components) > 1:
                    self._log(
                        f"  ⚠️ {stage_label}: production ESS 低于阈值 {threshold:.4g} 的窗口分散在 "
                        f"{len(failure_components)} 个互不相邻的区域：{failure_components}。各失败窗口"
                        "自身的 warmup 都已确认通过冻结验证；不再因为区域分散就整体硬停止——继续走下面"
                        "逐窗口的 fixed-H 分类，每个窗口按自己的探针结果独立判断是插 λ、拆窗，还是"
                        "fixed-H 通过但 production ESS 低需要 f_k/采样诊断。"
                    )

                # Probe every to_probe window, reusing a cached result whenever
                # its content fingerprint (protocol_key + this window's actual
                # lambda_vdw values + probe threshold) already has a complete
                # entry on disk -- otherwise a fresh process (resume) would
                # burn the same expensive burn-in + sampling per edge again
                # for windows that were already fully probed in an earlier
                # round or an earlier run. Freshly computed edges are
                # persisted one at a time via the on_edge_done callback (not
                # after the whole window finishes), so a crash partway
                # through a multi-edge window only loses the one in-flight
                # edge, not everything computed so far. All results are used
                # (not just the first all-passed one) because each to_probe
                # window is classified and acted on independently below --
                # unlike the old global veto, nothing here is thrown away, so
                # there is no early-stop shortcut to take.
                probe_results = {}
                probe_file = self._fixed_h_probe_file(stage_name)
                for se in to_probe:
                    fingerprint = self._fixed_h_probe_fingerprint(protocol_key, se, current_lambdas)
                    cached_entry = self._load_fixed_h_probe_cache(stage_name).get(fingerprint)
                    if cached_entry is not None and cached_entry.get("complete"):
                        probe_results[se] = cached_entry["pairs"]
                        self._log(
                            f"  ♻️ {stage_label}: 窗口 {se} 复用已缓存的 fixed-H 探针结果"
                            f"（λ 内容/协议指纹匹配，跳过重新采样；见 {probe_file}）"
                        )
                        self._persist_fixed_h_probe_edge(
                            stage_name, stage_type, repair_round, fingerprint, se, cached_entry["pairs"], complete=True,
                        )
                        continue

                    # 🔑 之前只在 complete=True 时才复用缓存；complete=False 的
                    # 部分结果（比如上次跑到第 5/9 条边时进程崩溃）会被整体忽略，
                    # 从第 0 条边重新算——每条边都是独立的固定 Hamiltonian burn-in
                    # + 采样（用同一份 relaxed_positions/relaxed_box 起跑，边与边
                    # 之间不互相依赖），恢复已缓存的前几条边、只补算剩余边，跟
                    # 一次性算完全部边等价，不是近似。
                    resume_pairs = (
                        list(cached_entry["pairs"])
                        if cached_entry is not None and cached_entry.get("pairs")
                        else []
                    )
                    if resume_pairs:
                        self._log(
                            f"  ♻️ {stage_label}: 窗口 {se} 从已缓存的 {len(resume_pairs)} 条边续算 fixed-H "
                            f"探针（λ 内容/协议指纹匹配，只补算剩余边；见 {probe_file}）"
                        )
                    collected_pairs = list(resume_pairs)

                    def _on_edge_done(pair, _se=se, _fp=fingerprint, _pairs=collected_pairs):
                        # 🔑 [两阶段探针] _probe_vdw_window_fixed_overlap 现在可能对
                        # 同一条边调用两次 on_edge_done——阶段一给出 path-only 结果
                        # （bias_calibration_sufficient=None 占位），窗口全部 path
                        # edge 都通过后，阶段二再对同一条边补上真正的 calibration
                        # 结果。按 global_edge 原地覆盖而不是盲目 append，否则同一条
                        # 边会在 _pairs 里出现两次、破坏顺序和长度。
                        edge_key = pair.get("global_edge")
                        replaced = False
                        for _idx, _existing in enumerate(_pairs):
                            if _existing.get("global_edge") == edge_key:
                                _pairs[_idx] = pair
                                replaced = True
                                break
                        if not replaced:
                            _pairs.append(pair)
                        self._log(
                            f"    · 窗口 {_se} fixed-H edge {pair.get('global_edge')}: "
                            f"min_overlap={pair.get('min_bidirectional_overlap', float('nan')):.5f} "
                            f"(阈值 {pair.get('threshold', float('nan')):.5f}, "
                            f"passed={pair.get('passed')}), ΔF="
                            f"{pair.get('delta_f_kJ_mol', float('nan')):.3f}±"
                            f"{pair.get('delta_f_uncertainty_kJ_mol', float('nan')):.3f} kJ/mol, "
                            f"N_decorrelated={pair.get('n_k_decorrelated')}"
                            + (
                                f", bias_calibration_sufficient={pair.get('bias_calibration_sufficient')}"
                                if pair.get("bias_calibration_sufficient") is not None
                                else ""
                            )
                        )
                        self._persist_fixed_h_probe_edge(
                            stage_name, stage_type, repair_round, _fp, _se, list(_pairs), complete=False,
                        )

                    pairs = probe_window_overlap_fn(
                        se, current_lambdas, on_edge_done=_on_edge_done, resume_pairs=resume_pairs,
                    )
                    probe_file = self._persist_fixed_h_probe_edge(
                        stage_name, stage_type, repair_round, fingerprint, se, pairs, complete=True,
                    )
                    probe_results[se] = pairs

                missing = [se for se in to_probe if not probe_results[se]]
                if missing:
                    raise RuntimeError(
                        f"{stage_label}: 窗口 {missing} production ESS 低，但 fixed-H overlap 探针"
                        "未能返回结果；拒绝回退到旧的按 ESS-per-lambda 算术二分插点。"
                    )

                # 🔑 之前一旦 to_probe 里任何一个窗口的 fixed-H 双向 overlap 全部
                # 通过，就把整个 to_probe（包括真正 fixed-H 失败、理应插 λ 的窗口）
                # 一起硬停止——真实案例：production ESS 低的窗口 [2,6)/[5,9)/[14,18)
                # 里，[2,6)/[14,18) 的 fixed-H 探针全通过，[5,9) 至少有一条边未通过，
                # 旧代码却整体 raise，[5,9) 的真实缺口从未被处理，也从未插过 λ。
                # 现在逐窗口分类：fixed-H 确有失败边的窗口照常留在 to_probe 里，
                # 按原逻辑插入待重测 λ；fixed-H 全通过的窗口不插 λ、不拆窗——但也
                # 不能就此把 production ESS 低当作"跟 lambda 无关，忽略"就结束：
                # 两个门槛本来就不同（production ESS 0.05 vs fixed-H overlap
                # 0.03），"fixed-H 全通过"只表示 λ 边已达到最低连通标准，自动插点
                # 缺少证据支持；最终自由能仍然来自这批低 ESS 的 production 数据，
                # 所以改为真正诊断+修复：用相邻边的 BAR/MBAR ΔF 累计出一份独立
                # f_k，跟生产冻结的 f_k 比较——差异明显就用校准 f_k 重新冻结验证，
                # 差异不明显就只重采这个窗口（见
                # _diagnose_and_repair_all_pass_low_ess_window），只针对这一个
                # 窗口生效，不影响 to_split/still_failing 的处理。
                already_good = [
                    se for se in to_probe
                    if all(p.get("passed") for p in probe_results[se])
                ]
                still_failing = [se for se in to_probe if se not in already_good]

                # 🔑 [starvation 修复] 之前只要 to_split/still_failing 非空
                # （path_will_change=True），already_good 窗口的校准/重采样修复
                # 就整轮推迟到"下一轮路径稳定之后"——但一条 18-20 态的路径几乎
                # 每轮都会在别处新冒出一条失败边，导致 already_good 窗口被反复
                # 推迟、永远轮不到，同时白白消耗共享的 repair_round 预算（真实
                # 案例：窗口 (0,3)/(2,6) 被推迟两轮以上，直到 5 轮预算耗尽直接
                # 硬停止，从未真正被修过）。现在分成两条路：路径本轮不变
                # （path_will_change=False）时按原逻辑立即处理；路径本轮会变时
                # 不再"整轮跳过"，而是在下面完成拆窗/插 λ/_invalidate_stage_
                # window_files 之后，按 λ 内容把每个 already_good 窗口重新定位到
                # 新的 (start,end)，同一轮内就把它们的校准/重采样也做掉——
                # _invalidate_stage_window_files 的重命名/清理必须先跑完
                # （它依赖每个窗口的 convergence.json 还在），这份修复才安全。
                path_will_change = bool(to_split or still_failing)
                if not path_will_change:
                    repair_actions = self._apply_already_good_repairs(
                        stage_name, stage_type, stage_label, threshold, repair_round,
                        [(se, se, effective_old_ranges) for se in already_good],
                        probe_results, pending_step_overrides, n_steps_per_window,
                        resample_step_growth_factor, max_resample_step_multiplier,
                    )
                else:
                    # 本轮路径会变，already_good 的修复推迟到下面拆窗/插 λ/
                    # _invalidate_stage_window_files 完成之后，在同一轮内按新
                    # (start,end) 应用——不再是"推迟到下一轮"。
                    repair_actions = []
                    if already_good:
                        self._log(
                            f"  ⏳ {stage_label}: 窗口 {already_good} fixed-H 全通过但 production ESS 低于"
                            f"阈值 {threshold:.4g}；本轮还有需要拆分/插 λ 的窗口，λ 路径即将变化——先完成"
                            "拆窗/插λ/窗口重映射，再在同一轮内按重映射后的新窗口范围对它们做 f_k 校准/"
                            "重采样修复，不再推迟到下一轮。"
                        )

                acted_on = [
                    tuple(a["window_range"]) for a in repair_actions
                    if a["decision"] in ("recalibrate_f_k", "reseed_resample")
                ]
                if not to_split and not still_failing and not acted_on:
                    raise RuntimeError(
                        f"{stage_label}: 本轮除 {already_good} 外没有其它可自动处理的窗口，且它们的"
                        "fixed-H 通过但 production ESS 低都无法自动诊断/修复（既无需要拆分的大窗口，"
                        "也无实测确认存在失败边的 fixed-H 探针，也没有可信的生产冻结 f_k 可供比较）；"
                        f"重跑只会得到相同结果，拒绝盲目重试。请参考上面的诊断信息、{probe_file} 里的"
                        "实测 overlap/ΔF 数值人工判断下一步。"
                    )
                to_probe = still_failing

                if not to_split and not to_probe and acted_on:
                    # Pure sampling-repair round: the λ path/window ranges are
                    # completely unchanged (nothing to split, nothing to
                    # insert). Do NOT fall through to
                    # _invalidate_stage_window_files()/preopt rewrite below --
                    # there is no path change for it to reconcile, and running
                    # it anyway would re-derive its reuse map from each
                    # window's convergence.json, which the sampling repair
                    # above just deleted for every acted-on window; that would
                    # flag them as "unmatched" and purge the ibs_state
                    # overwrite/keep this step just made.
                    # _invalidate_single_window_production() already did the
                    # only invalidation this round needs, scoped to exactly
                    # the touched windows -- just retry next round.
                    continue

                new_ranges = list(effective_old_ranges)
                split_feedback_list = []
                # Splitting never changes lambda count/global indices, so every
                # failing large window can be split in the same round with no
                # index-shift hazard. Must process highest-start-first though:
                # split_window_from_warmup_failure now also reflows the failed
                # window's immediate NEXT neighbor down to single-state overlap
                # (see its docstring) -- if a lower-start window were processed
                # before a higher-start one that's also in to_split, the later
                # call's own recorded (s, e) could already have been shifted by
                # the earlier call's neighbor-reflow and no longer match what's
                # in new_ranges. Processing right-to-left means any window that
                # could touch a given window's neighbor slot is handled after
                # that window itself, so each call's own (s, e) is always still
                # exactly what's on file at the time it's processed.
                for (s, e) in sorted(to_split, key=lambda se: -se[0]):
                    _, new_ranges, split_feedback = split_window_from_warmup_failure(
                        current_lambdas, new_ranges, {"window_index": -1, "global_state_range": [s, e]},
                    )
                    split_feedback_list.append(split_feedback)
                    self._log(
                        f"  ✂️ {stage_label}: production ESS 整窗低（窗口 [{s},{e})，态数={e - s}）；"
                        f"不插 λ，只拆为 {split_feedback['child_ranges']}，"
                        f"共享旧态 {split_feedback['shared_global_state']}"
                    )

                if to_split:
                    # Multiple overlapping parent windows split independently
                    # can produce a child that lands entirely inside a
                    # NEIGHBORING parent's span (IBS windows overlap by
                    # design) -- e.g. parents (0,6)/(3,9) each splitting on
                    # their own midpoint leaves (3,6) strictly contained in
                    # (2,6). Coverage is unaffected (the contained window adds
                    # no lambda index its superset doesn't already have), but
                    # canonicalizing here removes the redundant extra
                    # sampling before it gets persisted/resampled.
                    pre_canonical_count = len(new_ranges)
                    new_ranges = canonicalize_window_ranges(new_ranges, len(current_lambdas))
                    if len(new_ranges) != pre_canonical_count:
                        self._log(
                            f"  🧹 {stage_label}: 批量拆窗产生 {pre_canonical_count} 个窗口，"
                            f"归约掉 {pre_canonical_count - len(new_ranges)} 个被相邻窗口严格包含的"
                            f"冗余窗口，剩 {len(new_ranges)} 个"
                        )

                new_lambdas = current_lambdas
                insert_feedback_list = []
                if to_probe:
                    # 🔑 [批量插边修复] 之前每轮只处理 to_probe 里"最差窗口"的一条
                    # 失败边，哪怕同一轮里其它窗口也有失败边——导致一条边一条边
                    # 排队修，占满 repair_round 预算的同时，让 already_good 窗口
                    # （真正需要 f_k 重新校准/重采样的窗口）因为 path_will_change
                    # 被反复推迟，长期得不到修复（真实案例：窗口 (0,3)/(2,6) 早在
                    # 好几轮之前就被诊断出该修，却因为总有别的窗口这一轮还有失败边
                    # 一直没轮到，直到 5 轮预算耗尽直接硬停止）。现在收集这一轮所有
                    # still_failing 窗口各自的最差失败边，按全局边索引去重、从大到
                    # 小依次插入——insert_lambda_from_overlap_failure 每次插入都会
                    # 平移它所拿到的整份 ranges 列表，处理顺序从大到小保证已经处理
                    # 过的边不会再被后面的插入影响，只需要手动同步更新"尚未处理"的
                    # 窗口自己的 (start,end)。
                    #
                    # 拆窗（to_split）阶段可能已经移动了某个 to_probe 窗口的
                    # (start,end)（拆窗会把失败窗口的紧邻下一个窗口的 start 前移到
                    # 单态重叠）——先按 λ 内容把每个 to_probe 窗口重新定位到
                    # new_ranges 里对应的新范围，不能继续假设 effective_old_ranges
                    # 里的旧 (start,end) 仍然有效，否则会从 insert_lambda_from_
                    # overlap_failure 内部直接 raise（"failed_range not in ranges"）。
                    pending = []
                    unmatched_to_probe = []
                    for se in to_probe:
                        cur_range = self._remap_window_by_lambda_content(
                            se, current_lambdas, new_lambdas, new_ranges,
                        )
                        if cur_range is None:
                            unmatched_to_probe.append(se)
                            continue
                        failed_pairs = [p for p in probe_results[se] if not p.get("passed")]
                        worst_pair = min(
                            failed_pairs,
                            key=lambda p: float(p.get("min_bidirectional_overlap", np.inf)),
                        )
                        pending.append({
                            "orig_se": se,
                            "cur_range": cur_range,
                            "pair": worst_pair,
                            "global_edge": int(worst_pair["global_edge"][0]),
                        })
                    if unmatched_to_probe:
                        self._log(
                            f"  ⚠️ {stage_label}: 窗口 {unmatched_to_probe} 在本轮拆窗/归约后找不到"
                            "λ 内容匹配的新窗口范围（可能被 canonicalize_window_ranges 归约掉，或被"
                            "相邻拆窗的邻窗重排吞并）；本轮跳过它们的插 λ 处理，下一轮重新分类/探测。"
                        )

                    by_edge: Dict[int, List[Dict]] = {}
                    for item in pending:
                        by_edge.setdefault(item["global_edge"], []).append(item)
                    ordered_edges = sorted(by_edge.keys(), reverse=True)

                    for edge_pos, edge in enumerate(ordered_edges):
                        group = by_edge[edge]
                        # 同一条全局边可能同时是多个重叠窗口各自的最差失败边——只
                        # 需要真正插一次；挑测得更差的那个窗口作为
                        # insert_lambda_from_overlap_failure 的"失败窗口"（决定
                        # 拆成两个孩子的是哪个窗口），其余窗口会被它内部通用的
                        # 平移分支自动一并修好，不重复插点。
                        primary = min(
                            group,
                            key=lambda it: (
                                float(it["pair"]["min_bidirectional_overlap"]),
                                it["cur_range"][0],
                            ),
                        )
                        others = [it for it in group if it is not primary]
                        diag = {
                            "window_index": -1,
                            "global_state_range": list(primary["cur_range"]),
                            "bidirectional_overlap_probe": {"pairs": [primary["pair"]]},
                        }
                        new_lambdas, new_ranges, insert_feedback = insert_lambda_from_overlap_failure(
                            new_lambdas, new_ranges, diag,
                        )
                        insert_feedback_list.append(insert_feedback)
                        insert_at = int(insert_feedback["failed_global_edge"][1])
                        failed_range = tuple(primary["cur_range"])
                        windows_fixed_for_free = [it["orig_se"] for it in others]
                        self._log(
                            f"  🧪 {stage_label}: 窗口 {primary['orig_se']} fixed-H 双向 overlap="
                            f"{insert_feedback['measured_min_bidirectional_overlap']:.5f} < "
                            f"{insert_feedback['overlap_threshold']:.5f}；在实测失败边 "
                            f"{insert_feedback['failed_global_edge']} 插入待重测 λ="
                            f"{insert_feedback['inserted_lambda']:.8f}"
                            + (f"；同一条边同时覆盖窗口 {windows_fixed_for_free}，一并解决，不重复插点"
                               if windows_fixed_for_free else "")
                        )
                        # 同步更新尚未处理窗口的 (start,end)——跟
                        # insert_lambda_from_overlap_failure 内部完全一致的 4 分支
                        # 位移规则（原地匹配失败窗口/整体在插入点左侧/整体在插入点
                        # 右侧/跨插入点），保证下一次迭代里这些窗口自己的
                        # cur_range 仍然精确对应 new_ranges 里的实际内容。
                        for other_edge in ordered_edges[edge_pos + 1:]:
                            for it in by_edge[other_edge]:
                                s, e = it["cur_range"]
                                if (s, e) == failed_range:
                                    it["cur_range"] = (s, insert_at + 1)
                                elif e <= insert_at:
                                    pass
                                elif s >= insert_at:
                                    it["cur_range"] = (s + 1, e + 1)
                                else:
                                    it["cur_range"] = (s, e + 1)

                # Final canonicalization pass right before anything is
                # persisted, independent of whether the split loop above
                # already ran one -- cheap and idempotent on an already-
                # canonical list, and it's the one check that must hold no
                # matter which combination of split/insert produced new_ranges.
                new_ranges = canonicalize_window_ranges(new_ranges, len(new_lambdas))

                self._invalidate_stage_window_files(
                    stage_name,
                    stage_type,
                    old_lambdas=current_lambdas,
                    old_ranges=effective_old_ranges,
                    new_lambdas=new_lambdas,
                    new_ranges=new_ranges,
                )
                # 🔑 split/insert 之后 window_idx 的编号会整体重排（拆窗新增窗口、
                # 插 λ 改变全局态编号），pending_step_overrides 是按上一轮的
                # window_idx 位置写入的，路径变了就不再对应同一个物理窗口——
                # 留着会把延长步数的覆盖值发给这一轮里编号恰好相同、但其实是
                # 另一个窗口的目标，白烧 GPU 且让真正欠采样的窗口拿不到延长。
                # 直接清空，下一轮任何窗口需要 reseed_resample 时都从默认步数
                # 重新开始按倍数增长，不去猜哪个旧 key 还对得上。
                if pending_step_overrides:
                    self._log(
                        f"  🧹 {stage_label}: λ 路径本轮拆窗/插 λ 改变了窗口编号，"
                        f"清空 {len(pending_step_overrides)} 条旧的生产步数覆盖 "
                        f"{sorted(pending_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                    )
                    pending_step_overrides.clear()
                # 🔑 [window_idx 陈旧覆盖修复] 同上：frozen_validation_step_overrides/
                # frozen_validation_is_final_rung 一样按 window_idx 写入，必须跟
                # pending_step_overrides 一起清空，理由同下面 legacy 分支的注释。
                if frozen_validation_step_overrides or frozen_validation_is_final_rung:
                    self._log(
                        f"  🧹 {stage_label}: λ 路径本轮拆窗/插 λ 改变了窗口编号，"
                        f"清空 {len(frozen_validation_step_overrides)} 条旧的冻结验证阶梯覆盖 "
                        f"{sorted(frozen_validation_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                    )
                    frozen_validation_step_overrides.clear()
                    frozen_validation_is_final_rung.clear()
                # 🔑 [starvation 修复，slow lane] 上面已经完成拆窗/插 λ/窗口
                # 重映射、_invalidate_stage_window_files，以及本轮路径变化触发
                # 的 pending_step_overrides/frozen_validation_* 陈旧覆盖清空——
                # 这里才第一次安全地处理本轮被 path_will_change 挡住的
                # already_good 窗口：按 λ 内容把每个窗口重新定位到 new_ranges 里
                # 对应的新 (start,end)，同一轮内立即做校准/重采样修复，不再拖到
                # 下一轮。必须放在陈旧覆盖清空之后运行——若提前到清空之前，
                # reseed_resample 分支这一轮刚写入 pending_step_overrides 的全新
                # （按新 window_idx 编号的）延长步数覆盖会被紧接着的"清空陈旧
                # 覆盖"逻辑一并冲掉，下一轮读不到。同理也必须在
                # _invalidate_stage_window_files 完成之后运行（它依赖每个窗口的
                # convergence.json 还在原位置；_diagnose_and_repair_all_pass_
                # low_ess_window 的修复分支会删除这个文件，顺序反了会被上面的
                # 重用判断误当成"未匹配、该清理"）。probe_results 仍按探测时的
                # 原始 se 为键（探针结果只跟 λ 内容有关，不跟位置有关），只有
                # 传给诊断函数的窗口范围/全量范围列表要用重映射后的新值。
                if already_good:
                    already_good_entries = []
                    unmatched_already_good = []
                    for se in already_good:
                        new_se = self._remap_window_by_lambda_content(
                            se, current_lambdas, new_lambdas, new_ranges,
                        )
                        if new_se is None:
                            unmatched_already_good.append(se)
                            continue
                        already_good_entries.append((se, new_se, new_ranges))
                    if unmatched_already_good:
                        self._log(
                            f"  ⚠️ {stage_label}: 窗口 {unmatched_already_good} fixed-H 全通过，"
                            "但本轮拆窗/插 λ 后找不到 λ 内容匹配的新窗口范围（可能被 "
                            "canonicalize_window_ranges 归约掉，或被相邻拆窗的邻窗重排吞并）；"
                            "暂缓其 f_k 校准/重采样修复，下一轮重新分类/按需重新探测。"
                        )
                    self._apply_already_good_repairs(
                        stage_name, stage_type, stage_label, threshold, repair_round,
                        already_good_entries,
                        probe_results, pending_step_overrides, n_steps_per_window,
                        resample_step_growth_factor, max_resample_step_multiplier,
                    )
                cached_preopt = {}
                if os.path.exists(preopt_file):
                    with open(preopt_file, "r") as f:
                        cached_preopt = json.load(f)
                path_diagnostics = dict(cached_preopt.get("path_diagnostics", {}))
                if insert_feedback_list:
                    # Do not repeat the old 0.5L + 0.5L fiction -- the inserted
                    # coordinate is a hypothesis and its two new edges have no
                    # thermodynamic length until they are sampled.
                    path_diagnostics.pop("optimized_edge_thermodynamic_lengths", None)
                    path_diagnostics["requires_pilot_remeasurement"] = True
                path_diagnostics.setdefault("production_repair_history", []).extend(
                    split_feedback_list + insert_feedback_list
                )
                os.makedirs(os.path.dirname(preopt_file), exist_ok=True)
                with open(preopt_file, "w") as f:
                    json.dump({
                        "lambdas_var": new_lambdas,
                        "window_ranges": [list(r) for r in new_ranges],
                        "n_states": len(new_lambdas),
                        "protocol_key": preopt_protocol_key,
                        "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
                        "path_diagnostics": path_diagnostics,
                        "provenance": {
                            "source": "production_overlap_repair_split_then_probe",
                            "round": repair_round + 1,
                            "prior_min_overlap": min_overlap,
                            "prior_min_overlap_threshold": threshold,
                        },
                    }, f, indent=2)
                current_lambdas, current_ranges = new_lambdas, new_ranges
                current_n_states = len(new_lambdas)
                repair_round += 1
                continue

            new_lambdas, new_ranges = refine_stage_lambda_path_by_overlap(
                current_lambdas,
                effective_old_ranges,
                diagnostics,
                threshold,
            )
            if new_lambdas is None:
                raise RuntimeError(
                    f"{stage_label} 阶段 min_overlap={min_overlap:.4g} 低于阈值 {threshold:.4g}，"
                    "但自动修复逻辑未能从 window_overlap_diagnostics 里定位到需要加密的 λ 区间"
                    "（缺少 ess_ratio_per_lambda 明细，或诊断结构异常），拒绝盲目重试。"
                )

            self._log(
                f"  🔧 {stage_label}: min_overlap={min_overlap:.4g} < {threshold:.4g}，"
                f"第 {repair_round + 1}/{max_repair_rounds} 轮自动加密 λ 路径 "
                f"({len(current_lambdas)} -> {len(new_lambdas)} 个态)，按 λ 内容比对复用旧窗口产物"
            )
            self._invalidate_stage_window_files(
                stage_name,
                stage_type,
                old_lambdas=current_lambdas,
                old_ranges=effective_old_ranges,
                new_lambdas=new_lambdas,
                new_ranges=new_ranges,
            )
            # 🔑 同上（见拆窗/插 λ 分支）：加密 λ 路径同样会重排 window_idx，
            # pending_step_overrides 里按旧编号写入的延长步数覆盖不再对应
            # 同一个物理窗口，必须清空，不能带着旧编号进入下一轮。
            if pending_step_overrides:
                self._log(
                    f"  🧹 {stage_label}: λ 路径本轮加密改变了窗口编号，"
                    f"清空 {len(pending_step_overrides)} 条旧的生产步数覆盖 "
                    f"{sorted(pending_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                )
                pending_step_overrides.clear()
            # 🔑 [window_idx 陈旧覆盖修复] frozen_validation_step_overrides/
            # frozen_validation_is_final_rung 同样按 window_idx 写入，加密 λ 路径
            # 重排编号后必须一起清空，理由同上（见拆窗/插 λ 分支的对应注释）：
            # 否则重排后编号恰好相同的新窗口会继承旧窗口的阶梯预算/终态标记。
            if frozen_validation_step_overrides or frozen_validation_is_final_rung:
                self._log(
                    f"  🧹 {stage_label}: λ 路径本轮加密改变了窗口编号，"
                    f"清空 {len(frozen_validation_step_overrides)} 条旧的冻结验证阶梯覆盖 "
                    f"{sorted(frozen_validation_step_overrides.keys())}，避免误发给重排后的其它窗口。"
                )
                frozen_validation_step_overrides.clear()
                frozen_validation_is_final_rung.clear()
            os.makedirs(os.path.dirname(preopt_file), exist_ok=True)
            with open(preopt_file, "w") as f:
                json.dump(
                    {
                        "lambdas_var": new_lambdas,
                        "window_ranges": [list(r) for r in new_ranges],
                        "n_states": len(new_lambdas),
                        # 🔑 落这个协议指纹，是为了让 resume 时能安全地信任一份
                        # n_states 已经不等于最初请求值、但确实是本协议下已验证过的
                        # 自动加密结果的缓存（见 run_full_pipeline 里的 preopt resume
                        # 读取逻辑），而不是盲目要求 n_states 精确等于最初的猜测值。
                        "protocol_key": preopt_protocol_key,
                        "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
                        "provenance": {
                            "source": "auto_repair_by_overlap",
                            "round": repair_round + 1,
                            "prior_n_states": current_n_states,
                            "prior_min_overlap": min_overlap,
                            "prior_min_overlap_threshold": threshold,
                            "prior_window_overlap_diagnostics": diagnostics,
                        },
                    },
                    f,
                    indent=2,
                )
            current_lambdas, current_ranges = new_lambdas, new_ranges
            current_n_states = len(new_lambdas)
            repair_round += 1

        raise RuntimeError(f"{stage_label}: 自动修复循环异常退出（不应到达这里）")
```
