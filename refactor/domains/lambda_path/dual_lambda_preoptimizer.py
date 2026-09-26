# =============================================================================
# 添加双λ路径优化类
# =============================================================================
# 修复 DualLambdaPreOptimizer 类
# =============================================================================
# =============================================================================
# 修复 DualLambdaPreOptimizer 类 (完整修复版)
# =============================================================================
# ================= abfe_preoptimizer.py =================
# 完整替换 DualLambdaPreOptimizer 类
class DualLambdaPreOptimizer:
    """双λ预采样优化器 (全链路 Debug 版)"""
    def __init__(self, system, context, temperature=300.0):
        print(f"\n[DEBUG-OPT] DualLambdaPreOptimizer 初始化...")
        self.system = system
        self.context = context
        self.temperature = temperature
        self.param_coul = self._normalize_param_name(
            self._detect_param("coul", ["lam_coul", "lambda_coul"])
        )
        self.param_vdw = self._normalize_param_name(
            self._detect_param("vdw", ["lam_vdw", "lambda_vdw"])
        )
        print(f"[DEBUG-OPT] 探测结果 -> Coul: '{self.param_coul}', VdW: '{self.param_vdw}'")

    @staticmethod
    def _normalize_param_name(param) -> Optional[str]:
        if param is None:
            return None
        if isinstance(param, np.ndarray):
            flat = np.asarray(param).ravel()
            if flat.size == 0:
                return None
            param = flat[0]
        return str(param)

    def _detect_param(self, keyword: str, fallbacks: list) -> Optional[str]:
        print(f"  [SCAN] 搜索关键词: '{keyword}', 候选: {fallbacks}")
        # 1. 扫 Force
        if self.system is not None:
            for f in self.system.getForces():
                if isinstance(f, openmm.CustomNonbondedForce):
                    names = [f.getGlobalParameterName(i) for i in range(f.getNumGlobalParameters())]
                    print(f"  [SCAN] Force 包含参数: {names}")
                    for n in names:
                        if keyword in n.lower(): 
                            print(f"  [SCAN] [OK] Force 匹配到: {n}")
                            return n
        # 2. 扫 Context
        try:
            ctx_p = list(self.context.getParameters().keys())
            print(f"  [SCAN] Context 包含参数: {ctx_p}")
            for k in ctx_p:
                if keyword in k.lower(): 
                    print(f"  [SCAN] [OK] Context 匹配到: {k}")
                    return k
        except Exception as e: print(f"  [SCAN] Context 读取失败: {e}")
        return None

    def _metric_force_groups(self, parameter_name: str) -> set:
        """差分 `parameter_name` 时要计入哪些 force group。

        规则只有一条：**把该参数的依赖项全部算进来，且只算这些**。

        * 软核 ACES 力（group 1）同时带 lam_coul 与 lam_vdw ⟹ 永远计入。
        * 原生 NonbondedForce（`PREOPT_NATIVE_NONBONDED_FORCE_GROUP`，只在带电腿
          存在）只带 lam_coul（B3 的 PME ParameterOffset）⟹ **只在差分 lam_coul
          时**计入。差分 lam_vdw 时它是个 ~10^6 kJ/mol 的常数，算进来就是在巨大
          公共项上做差再除以 delta≈0.02，纯灾难性相消。
        * 中性腿本来就没有力在那个 group 里，集合多写一个空 group 无副作用。
        """
        if self.param_coul is not None and parameter_name == self.param_coul:
            return {1, PREOPT_NATIVE_NONBONDED_FORCE_GROUP}
        return {1}

    def _metric_energy_at(self, parameter_name: str, lam: float) -> float:
        self.context.setParameter(parameter_name, float(lam))
        state = self.context.getState(
            getEnergy=True, groups=self._metric_force_groups(parameter_name)
        )
        return state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def _finite_difference_derivative_1d(
        self,
        parameter_name: str,
        lam: float,
        delta: float,
    ) -> float:
        """Evaluate dU/dlambda on one frozen configuration and restore lambda."""
        lam = float(lam)
        lo = max(0.0, lam - float(delta))
        hi = min(1.0, lam + float(delta))
        try:
            if hi > lam and lo < lam:
                e_hi = self._metric_energy_at(parameter_name, hi)
                e_lo = self._metric_energy_at(parameter_name, lo)
                return (e_hi - e_lo) / (hi - lo)
            e_0 = self._metric_energy_at(parameter_name, lam)
            if hi > lam:
                return (self._metric_energy_at(parameter_name, hi) - e_0) / (hi - lam)
            if lo < lam:
                return (e_0 - self._metric_energy_at(parameter_name, lo)) / (lam - lo)
            raise RuntimeError(f"lambda={lam} 没有可用的有限差分邻点")
        finally:
            self.context.setParameter(parameter_name, lam)

    def _sample_scalar_metric(
        self,
        parameter_name: str,
        lam: float,
        n_steps: int,
        delta: float,
        sample_interval: int = 50,
        shadow_checkpoint_steps: Optional[List[int]] = None,
    ) -> Tuple[float, Dict]:
        """Short-pilot estimate of beta**2 Var[dU/dlambda]."""
        # 🔑 [性能计时] 只加计时，不改任何积分/有限差分逻辑或默认参数——
        # sample_interval/n_steps 直接影响 λ 路径优化结果，这次不动，见
        # optimize_stage2_vanishing 调用处的说明。目的是把"这个 λ 点到底
        # 花在积分 vs 有限差分能量读取上多少时间"变成可测量的数字。
        point_timers: Dict[str, float] = {}
        derivative_samples = []
        # 🔑 [shadow early-stop 插桩，Phase A，2026-08-26] shadow_checkpoint_
        # steps 为 None（默认）时下面这段完全不产生任何额外计算/字段——循环
        # 仍然无条件跑满传入的 n_steps，真实采样长度、返回值形状逐字节不变。
        # 传入时也不改变真实采样长度：batches 仍然全部跑完；只是在累积步数
        # 跨过每个请求的 checkpoint 时，用当时已经采到的 derivative_samples
        # 多算一次"假想现在停下会怎样"的诊断，写进 shadow_trace，不参与任何
        # 真实判断分支（是否继续采样、metric_g 怎么算，都跟今天完全一样）。
        pending_checkpoints = (
            sorted({int(s) for s in shadow_checkpoint_steps})
            if shadow_checkpoint_steps
            else []
        )
        shadow_trace: List[Dict[str, Any]] = []
        cumulative_steps = 0
        full_batches, remainder = divmod(int(n_steps), int(sample_interval))
        batches = [int(sample_interval)] * full_batches
        if remainder:
            batches.append(remainder)
        for batch_steps in batches:
            with _timed(point_timers, "integration_s"):
                guarded_step(self.context.getIntegrator(), batch_steps, "pilot 标量度量采样")
            with _timed(point_timers, "finite_difference_s"):
                derivative = self._finite_difference_derivative_1d(
                    parameter_name, lam, delta
                )
            if np.isfinite(derivative):
                derivative_samples.append(float(derivative))
            cumulative_steps += int(batch_steps)
            while pending_checkpoints and cumulative_steps >= pending_checkpoints[0]:
                checkpoint_step = pending_checkpoints.pop(0)
                snapshot = pilot_block_running_diagnostics(
                    np.asarray(derivative_samples, dtype=float),
                    float(self.temperature),
                )
                snapshot["cumulative_steps"] = int(cumulative_steps)
                snapshot["requested_checkpoint_steps"] = int(checkpoint_step)
                shadow_trace.append(snapshot)

        if len(derivative_samples) < 10:
            raise RuntimeError(
                f"lambda={lam:.6f} 只有 {len(derivative_samples)} 个有效 dU/dlambda 样本；"
                "至少需要 10 个，拒绝用欠采样度量生成路径"
            )
        values = np.asarray(derivative_samples, dtype=float)
        beta = 1.0 / (0.008314462618 * float(self.temperature))
        metric_g = float(beta * beta * np.var(values, ddof=1))
        diag = {
            "lambda": float(lam),
            "n_derivative_samples": int(values.size),
            "mean_dU_dlambda_kJ_mol": float(np.mean(values)),
            "std_dU_dlambda_kJ_mol": float(np.std(values, ddof=1)),
            "metric_g": metric_g,
            "timing_s": dict(point_timers),
        }
        if shadow_checkpoint_steps:
            diag["shadow_trace"] = shadow_trace
        return metric_g, diag

    def optimize_stage1_decharging(self, n_states=12, n_steps_per_state=2000):
        # This legacy entry point samples ``Var(U_group1)`` from a cutoff
        # probe.  That is not the Fisher metric of the production PME
        # Hamiltonian (which requires beta² Var[dU/dlambda]), and it can also
        # include lambda-independent environment noise.  The production
        # pipeline already uses a validated linear Stage-1 path; keep this
        # public API fail-closed until a PME derivative sampler is implemented.
        raise RuntimeError(
            "Stage 1 自适应去电荷预优化已禁用：旧实现使用 Var(U) 而非生产 PME 的 "
            "beta² Var[dU/dlambda]，其路径不能用于热力学采样。请使用 pipeline 的线性路径。"
        )

        print(f"\n[STAGE1] 开始去电荷路径优化 (n_states={n_states})...")
        print(f"[STAGE1] 当前 param_coul='{self.param_coul}', param_vdw='{self.param_vdw}'")
        
        if self.param_coul is None:
            print(f"[STAGE1] [WARN] 探针系统未注册 Coulomb λ 参数，直接生成线性回退路径")
            return {"stage": "decharging", "lambdas_coul": np.linspace(1.0, 0.0, n_states).tolist(), "lambdas_vdw": [1.0]*n_states, "n_states": n_states}
            
        # 安全设置
        # ✅ 修复：统一增加 None 保护
        current_params = dict(self.context.getParameters())
        if self.param_vdw is not None:
            if self.param_vdw in current_params:
                self.context.setParameter(self.param_vdw, 1.0)
                print(f"[STAGE1] 固定 λ_vdw = 1.0")
                
        self.context.setParameter(self.param_coul, 1.0)
        print(f"[STAGE1] 设置初始 λ_coul = 1.0")
        guarded_step(self.context.getIntegrator(), 5000, "Stage1 pilot 起始预平衡")

        variance_data = []
        lambdas = np.linspace(1.0, 0.0, n_states)
        print(f"[STAGE1] 线性采样点: {lambdas}")

        for i, lam in enumerate(lambdas):
            self.context.setParameter(self.param_coul, float(lam))
            if self.param_vdw is not None:  # ✅ 修复：增加 None 守卫，确保 Stage1 安全
                self.context.setParameter(self.param_vdw, 1.0)
            guarded_step(self.context.getIntegrator(), 500, f"Stage1 pilot λ_coul={lam:.4f} 预平衡")
            energies = [
                e for e in _sample_group1_energies(self.context, n_steps_per_state, sample_interval=50)
                if not (np.isnan(e) or np.isinf(e))
            ]
            variance_data.append(np.var(energies) if len(energies) >1 else 1.0)

        # 路径重分布 (保持原逻辑)
        std_dev = np.sqrt(np.array(variance_data) + 1e-10)
        density_weight = np.asarray(_normalize_variance_weights(std_dev, max_ratio=0.15), dtype=float).ravel()
        # [0831issue P2] 同 optimize_lambda_path_adaptive：用前 N-1 个权重当区间宽度、
        # 按它们自己的和归一化，末端天然为 1.0，不再事后覆盖 c_{N-2}（那会把
        # λ[N-2] 的权重双重计入）。
        interval_weights = np.asarray(density_weight, dtype=float).ravel()[:-1]
        interval_total = max(1e-10, float(np.sum(interval_weights)))
        xp = np.concatenate(
            ([0.0], np.cumsum(interval_weights) / interval_total)
        ).astype(float).ravel()
        fp = np.asarray(lambdas, dtype=float).ravel()
        target_cumulative = np.linspace(0, 1.0, n_states)
        optimized_lambdas = np.asarray(np.interp(target_cumulative, xp, fp), dtype=float).ravel()
        # 🔑 之前这里只 clip/排序/钉端点，没有最小间距、没有去重、状态数不足时
        # 也没有 fail-closed/回退——跟单 λ optimize_lambda_path_adaptive 的同一步
        # 逻辑不一致，一次看似正常的 CDF 插值完全可能悄悄给出两个数值相同的 λ，
        # 破坏 MBAR 对相邻态"确实是不同态"的假设，且不会有任何报错/日志。改用
        # 共享的 finalize_descending_lambda_path，跟单 λ 路径统一同一套不变量
        # （密度权重公式本身仍保持 Stage1 自己的实现，未改动）。
        optimized_lambdas, _min_spacing, fell_back = finalize_descending_lambda_path(
            optimized_lambdas, n_states
        )
        if fell_back:
            print(f"[STAGE1] [WARN] 去重后状态数少于目标 ({n_states})，使用线性路径")

        print(f"[STAGE1] [OK] 优化完成，返回前5个λ: {optimized_lambdas[:5]}")
        return {"stage": "decharging", "lambdas_coul": optimized_lambdas.tolist(), "lambdas_vdw": [1.0]*len(optimized_lambdas), "n_states": len(optimized_lambdas)}

    def _refine_pilot_grid_in_steep_segments(
        self,
        pilot_lambdas,
        metric_g,
        pilot_points,
        n_steps_per_state,
        finite_difference_delta,
        max_segment_length_fraction: float = 0.2,
        extra_points_per_segment: int = 4,
        max_rounds: int = 2,
        shadow_checkpoint_steps: Optional[List[int]] = None,
        pilot_states=None,
    ):
        """[THERMODYNAMIC_PATH_PROTOCOL_VERSION=15] Probe additional points
        strictly inside whichever single coarse pilot segment dominates the
        total thermodynamic length, instead of trusting a straight-line
        interpolation across it.

        Three independent real GPU runs of vanishing window 0 (6/4/3-state
        groupings) all failed with occupation stuck ~97-99% at state 0 and
        min_absolute_ess~1.0 -- regrouping never changes the actual lambda
        values, only which states share one IBS bias, so it could not (and
        did not) fix a real overlap problem at the state0/state1 edge. The
        real cached pilot data showed why: the coarse grid's very first
        segment (lambda=1.0 -> ~0.94) alone contributed ~47% of the entire
        path's thermodynamic length, but is defined by only 2 raw pilot
        points -- so redistribute_lambda_by_thermodynamic_length's arc-length
        interpolation could only ever place new states *linearly* inside it,
        with no real data on how the true difficulty is actually distributed
        there. This method is the fix: reuse the exact same
        _sample_scalar_metric measurement, just at more points, specifically
        inside whatever segment is currently blind, and merge the results
        back into the pilot arrays before redistribution ever runs. Returns
        (pilot_lambdas, metric_g, pilot_points) with the same shapes/meaning
        as the inputs, just more entries -- every downstream consumer already
        works generically on however many pilot points it receives.
        """
        # 🔑 [2026-09-10] 加密点必须从**相邻的高 λ 端点**续接采样，不能从上一个
        # 主 pilot 点（λ=0，配体已完全解耦）直接跳回 λ≈0.99。
        #
        # 主 pilot 是 λ=1 → 0 顺序走的，跑完才进这个函数。原来这里直接
        # `setParameter(λ_vdw, 0.99)` 然后只给 500 步预平衡就开测 —— 那是在
        # "配体突然重新长回一个已经塌陷的空腔"的**非平衡**构型上测 dU/dλ，
        # mean 和 Var 都被系统性抬高。而这批点恰恰是 λ 布点与 f_k 种子的输入，
        # 同时也是崩溃源（重新耦合方向的 500 步很容易在 guarded_step 里抛 NaN）。
        #
        # 现在要求调用方把每个主 pilot λ 的 (坐标, 速度, 盒子) 快照一并传进来：
        # 细化某段之前先恢复该段**高 λ 端点**的状态，再向低 λ 顺序采；新插入点
        # 自己的状态也存回列表，供下一轮从它续接。这样加密点的采样历史与主网格
        # 同类，500 步预平衡才站得住。
        #
        # ⚠️ 这改变了 pilot 的采样语义 ⟹ 旧 preopt 缓存不能冒充等价，必须由
        # 上游的 layer-1 采样语义字段挡住（不是靠版本号）。
        if pilot_states is None:
            raise ValueError(
                "pilot 网格加密需要每个主 pilot λ 的 endpoint states（坐标/速度/盒子快照）："
                "加密点必须从相邻高 λ 端点续接采样，不能从 λ=0 的完全解耦构型跳回去。"
                "调用方未提供 pilot_states —— 拒绝在没有续接状态的情况下加密（fail closed）。"
            )
        pilot_states = list(pilot_states)
        if len(pilot_states) != len(list(pilot_lambdas)):
            raise ValueError(
                "endpoint states 数量与 pilot λ 数量不一致："
                f"{len(pilot_states)} != {len(list(pilot_lambdas))}。"
                "两者必须逐位对应，否则会从错误的构型续接。"
            )

        pilot_lambdas = [float(x) for x in pilot_lambdas]
        metric_g = list(metric_g)
        pilot_points = list(pilot_points)
        current_params = dict(self.context.getParameters())

        for _round in range(int(max_rounds)):
            # 🔑 [重用] 跟 classify_pilot_point_risk_zone 用同一份共享实现
            # （原来这里是内联重复代码），数值行为不变。
            seg_lengths = _pilot_segment_lengths(pilot_lambdas, metric_g).tolist()
            total_length = float(sum(seg_lengths))
            if total_length <= 0.0 or not seg_lengths:
                break
            worst_idx = max(range(len(seg_lengths)), key=lambda i: seg_lengths[i])
            worst_fraction = seg_lengths[worst_idx] / total_length
            if worst_fraction <= float(max_segment_length_fraction):
                break

            lam_hi = pilot_lambdas[worst_idx]
            lam_lo = pilot_lambdas[worst_idx + 1]
            new_lams = np.linspace(lam_hi, lam_lo, int(extra_points_per_segment) + 2)[1:-1]
            print(
                f"  [pilot 加密] 段 [{lam_lo:.4f}, {lam_hi:.4f}] 占当前总热力学长度 "
                f"{worst_fraction * 100:.1f}%（阈值 {float(max_segment_length_fraction) * 100:.0f}%），"
                f"插入 {len(new_lams)} 个额外探针点重测（第 {_round + 1} 轮）"
            )

            insert_at = worst_idx + 1
            # 恢复该段**高 λ 端点**的状态，然后向低 λ 顺序采（new_lams 已是 hi→lo）。
            # 只在进入这一段时恢复一次；段内各点自然地一个接一个续下去，
            # 与主 pilot 的遍历方向一致。
            self.context.setState(pilot_states[worst_idx])
            for lam in new_lams:
                self.context.setParameter(self.param_vdw, float(lam))
                if self.param_coul is not None and self.param_coul in current_params:
                    self.context.setParameter(self.param_coul, 0.0)
                guarded_step(self.context.getIntegrator(), 500, f"pilot 网格加密 λ_vdw={lam:.4f} 预平衡")
                g_lam, point_diag = self._sample_scalar_metric(
                    self.param_vdw,
                    float(lam),
                    n_steps=int(n_steps_per_state),
                    delta=float(finite_difference_delta),
                    shadow_checkpoint_steps=shadow_checkpoint_steps,
                )
                # 🔑 加密点永远标记为风险点（classify_pilot_point_risk_zone
                # 消费这个字段）——插入的理由本来就是父区间空间信息不足，不
                # 因为是"额外点"缩短预算判断。
                point_diag["is_refinement_point"] = True
                _timing = point_diag.get("timing_s", {})
                print(
                    f"    [preopt 加密 λ={float(lam):.4f}] "
                    + ", ".join(f"{k}={v:.1f}s" for k, v in _timing.items())
                )
                pilot_lambdas.insert(insert_at, float(lam))
                metric_g.insert(insert_at, g_lam)
                pilot_points.insert(insert_at, point_diag)
                # 新点自己的状态存回同一位置：下一轮若选中与它相邻的段，
                # 就从这里续接，而不是回退到更高的 λ。
                pilot_states.insert(
                    insert_at,
                    self.context.getState(
                        getPositions=True, getVelocities=True, getParameters=True
                    ),
                )
                insert_at += 1

        return np.asarray(pilot_lambdas, dtype=float), metric_g, pilot_points

    def optimize_stage2_vanishing(
        self,
        n_states=VANISHING_PROBE_BASE_STATE_COUNT,
        n_steps_per_state=2000,
        finite_difference_delta=0.01,
        shadow_checkpoint_interval: Optional[int] = None,
        final_state_count: int = VANISHING_FINAL_STATE_COUNT,
        # 🔑 [2026-08-27] 之前硬编码在 _refine_pilot_grid_in_steep_segments 的
        # 默认参数里（这里没暴露），现在做成真正能传的参数，默认值不变。
        refine_extra_points_per_segment: int = 4,
        # 🔑 [2026-08-27] 只在 final_state_count != VANISHING_FINAL_STATE_COUNT
        # 时生效，见 redistribute_vanishing_lambda_subdomains。
        min_states_per_window: int = 4,
        max_states_per_window: int = 6,
        # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=22] 自由能定向加密点数，默认 0
        # （布点与 v21 逐字节相同）。总态数仍是 final_state_count，成本不变。
        free_energy_densify_points: int = VANISHING_FREE_ENERGY_DENSIFY_POINTS,
    ):
        print(
            f"\n→ Stage 2: 去 VDW 路径优化 "
            f"({n_states} 点 Fisher 探针网格 → {final_state_count} 态"
            f"度规布点，几何覆盖下限 beta={VANISHING_GEOMETRIC_FLOOR_WEIGHT})..."
        )
        current_params = dict(self.context.getParameters())
        if self.param_vdw is None or self.param_vdw not in current_params:
            raise RuntimeError(f"探针系统未注册 VdW λ 参数，无法执行自适应优化")

        # 🔑 [shadow early-stop 插桩，Phase A，2026-08-26] shadow_checkpoint_
        # interval 默认 None——下面这行给出 None，_sample_scalar_metric 里
        # pending_checkpoints 恒为空列表，真实采样长度/返回值形状逐字节不
        # 变。显式传入正整数时才会在每跑够这么多步就多记一次"假想提前停"的
        # 诊断，不改变任何一次真实采样的步数或判断分支。
        shadow_checkpoint_steps = (
            list(
                range(
                    int(shadow_checkpoint_interval),
                    int(n_steps_per_state) + 1,
                    int(shadow_checkpoint_interval),
                )
            )
            if shadow_checkpoint_interval
            else None
        )

        if self.param_coul is not None and self.param_coul in current_params:
            self.context.setParameter(self.param_coul, 0.0)
        self.context.setParameter(self.param_vdw, 1.0)
        guarded_step(self.context.getIntegrator(), 5000, "Stage2 pilot 起始预平衡")

        # Probe a conventional grid for diagnostics.  Production lambda
        # placement keeps the v19 quadratic base so the lambda~0 tail cannot
        # collapse again; v20 additionally lets this metric insert two bridge
        # states into the longest remaining production edges.
        pilot_lambdas = human_vanishing_initial_lambdas(int(n_states))
        metric_g = []
        pilot_points = []
        # 🔑 [2026-09-10] 逐 λ 存一份 (坐标, 速度, 盒子) 快照。
        #
        # 加密阶段（`_refine_pilot_grid_in_steep_segments`）要从**相邻高 λ 端点**
        # 续接采样，而不是从这个循环结束时的 λ=0（配体完全解耦）构型跳回 λ≈0.99。
        # 没有这些快照它会 fail closed，理由见那个函数的说明。
        pilot_states = []
        for lam in pilot_lambdas:
            self.context.setParameter(self.param_vdw, float(lam))
            if self.param_coul is not None and self.param_coul in current_params:
                self.context.setParameter(self.param_coul, 0.0)
            guarded_step(self.context.getIntegrator(), 500, f"Stage2 pilot λ_vdw={lam:.4f} 预平衡")
            g_lam, point_diag = self._sample_scalar_metric(
                self.param_vdw,
                float(lam),
                n_steps=int(n_steps_per_state),
                delta=float(finite_difference_delta),
                shadow_checkpoint_steps=shadow_checkpoint_steps,
            )
            point_diag["is_refinement_point"] = False
            metric_g.append(g_lam)
            pilot_points.append(point_diag)
            pilot_states.append(
                self.context.getState(
                    getPositions=True, getVelocities=True, getParameters=True
                )
            )
            _timing = point_diag.get("timing_s", {})
            print(
                f"    [preopt λ={float(lam):.4f}] "
                + ", ".join(f"{k}={v:.1f}s" for k, v in _timing.items())
            )

        pilot_lambdas, metric_g, pilot_points = self._refine_pilot_grid_in_steep_segments(
            pilot_lambdas,
            metric_g,
            pilot_points,
            pilot_states=pilot_states,
            n_steps_per_state=n_steps_per_state,
            finite_difference_delta=finite_difference_delta,
            shadow_checkpoint_steps=shadow_checkpoint_steps,
            extra_points_per_segment=int(refine_extra_points_per_segment),
        )

        (
            optimized_lambdas,
            cumulative_length,
            optimized_edge_lengths,
            window_ranges,
            subdomain_allocation,
        ) = redistribute_vanishing_lambda_subdomains(
                pilot_lambdas,
                np.asarray(metric_g, dtype=float),
                int(n_states),
                first_ensemble_target_intervals=VANISHING_FIRST_ENSEMBLE_TARGET_INTERVALS,
                final_state_count=int(final_state_count),
                min_states_per_window=int(min_states_per_window),
                max_states_per_window=int(max_states_per_window),
                free_energy_densify_points=int(free_energy_densify_points),
                pilot_mean_dU_dlambda=_pilot_mean_gradients_or_none(pilot_points),
        )
        optimized_lambdas = np.asarray(optimized_lambdas, dtype=float).ravel()
        optimized_lambdas = np.clip(optimized_lambdas, 0.0, 1.0)
        optimized_lambdas[0], optimized_lambdas[-1] = 1.0, 0.0
        # Thermodynamic length determines lambda density; few-state grouping is
        # performed afterwards along that coordinate.  No fixed lambda=0.5 cut
        # and no legacy overlap=2 construction that duplicates an interval are
        # used; one boundary node is still shared as the ensemble reference.
        
        # 🔑 [shadow early-stop 插桩，Phase A] 纯事后打标签，只读 pilot_points
        # 里已经落盘的 is_refinement_point，不影响上面任何一次真实采样/布点
        # 决定。shadow_checkpoint_steps 为 None 时 risk_zone_tags 也是 None，
        # 诊断字典形状对未启用 shadow 模式的调用完全不变。
        risk_zone_tags = None
        if shadow_checkpoint_steps:
            risk_zone_tags = classify_pilot_point_risk_zone(
                pilot_lambdas,
                metric_g,
                [bool(p.get("is_refinement_point", False)) for p in pilot_points],
            )

        diagnostics = {
            "estimator": "beta^2_var_dU_dlambda_finite_difference",
            "lambda_placement_method": (
                "fisher_metric_blended_with_geometric_floor_v21"
                if not int(free_energy_densify_points)
                else "fisher_metric_blended_with_geometric_floor_v21"
                     "+free_energy_densified_v22"
            ),
            "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
            "probe_controls_base_lambda_placement": True,
            "geometric_floor_weight": subdomain_allocation["geometric_floor_weight"],
            "max_lambda_gap_bound": subdomain_allocation["max_lambda_gap_bound"],
            "realized_max_lambda_gap": subdomain_allocation["realized_max_lambda_gap"],
            "realized_max_edge_thermodynamic_length": subdomain_allocation[
                "realized_max_edge_thermodynamic_length"
            ],
            "realized_min_edge_thermodynamic_length": subdomain_allocation[
                "realized_min_edge_thermodynamic_length"
            ],
            "requested_probe_base_state_count": int(n_states),
            "actual_state_count": int(len(optimized_lambdas)),
            "pilot_lambdas": [float(x) for x in pilot_lambdas],
            "metric_g": [float(x) for x in metric_g],
            "pilot_cumulative_thermodynamic_length": [float(x) for x in cumulative_length],
            "total_thermodynamic_length": float(cumulative_length[-1]),
            "optimized_edge_thermodynamic_lengths": [float(x) for x in optimized_edge_lengths],
            "finite_difference_delta": float(finite_difference_delta),
            "ibs_ensemble_layout": "few_state_thermodynamic_subdomains",
            "subdomain_allocation": subdomain_allocation,
            "sliding_overlap_states": 0,
            "common_boundary_state_count": 1,
            "pilot_points": pilot_points,
            "shadow_mode_enabled": bool(shadow_checkpoint_steps is not None),
        }
        if risk_zone_tags is not None:
            diagnostics["risk_zone_tags"] = risk_zone_tags
        print(
            f"  [OK] Stage 2 热力学长度路径完成：L={cumulative_length[-1]:.3f}, "
            f"{len(optimized_lambdas)} 态, {len(window_ranges)} 个 IBS 子区间"
        )
        print(f"    λ_vdw: {optimized_lambdas}")
        print(f"    windows: {window_ranges}")
        return {
            "stage": "vanishing",
            "lambdas_coul": [0.0] * len(optimized_lambdas),
            "lambdas_vdw": optimized_lambdas.tolist(),
            "n_states": len(optimized_lambdas),
            "window_ranges": window_ranges,
            "path_protocol_version": THERMODYNAMIC_PATH_PROTOCOL_VERSION,
            "path_diagnostics": diagnostics,
        }


# 修复 9: warmup safety check
def apply_safety_checks_on_disable_warmup(simulation, enable_warmup, warmup_steps):
    from openmm import unit
    import numpy as np
    import warnings
    if not enable_warmup:
        if simulation.context is None:
            print("  [WARN] simulation.context 未初始化，跳过安全检查")
            return
        try:
            state = simulation.context.getState(getEnergy=True, getForces=True)
            forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            force_norms = np.linalg.norm(forces, axis=1)
            rms_force = np.sqrt(np.mean(force_norms**2))
            max_force = np.max(force_norms)
            
            # ✅ RMS 阈值 5000 + 极值兜底 20000
            if np.isnan(rms_force) or np.isinf(rms_force) or rms_force > 5000 or max_force > 20000:
                warnings.warn("[WARN] 检测到不合理 RMS 力或极值，强制能量最小化...", UserWarning)
                simulation.minimizeEnergy(maxIterations=10000)
                state = simulation.context.getState(getEnergy=True, getForces=True)
                forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
                force_norms = np.linalg.norm(forces, axis=1)
                rms_force = np.sqrt(np.mean(force_norms**2))
                print(f"  [OK] 最小化后: RMS|F|={rms_force:.2e}, max|F|={np.max(force_norms):.2e}")
            else:
                print(f"  [OK] 安全检查通过: RMS|F|={rms_force:.2e}")
        except Exception as e:
            print(f"  [ERR] 安全检查失败: {e}")
            raise

