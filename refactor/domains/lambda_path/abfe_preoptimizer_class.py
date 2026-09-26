class ABFEPreOptimizer:
    """ABFE 预采样优化器 - ACES 路径优化版
    【修复 3】绑定 context 生命周期，不再单独保存 system
    """

    def __init__(
        self,
        system: openmm.System,
        context: openmm.Context,
        lambdas: List[float],
        temperature: float = 300.0,
        target_phase: str = "auto",
    ):
        # ✅ 修复 3: 只通过 context 获取 system，确保生命周期一致
        self.context = context
        self.lambdas = np.array(lambdas)
        self.n_states = len(lambdas)
        self.temperature = temperature

        # ✅ 修复 5: 保存 lambdas 为实例变量
        self.original_lambdas = lambdas.copy()  # 修复变量名错误

        # 结果存储
        self.optimized_params = {}
        self.initial_weights = None
        self.energy_history = []
        self.lambda_density = None

        # 【新增】最优衰减指数
        self.optimal_charge_exponent = 2.0
        self.optimal_vdw_exponent = 1.0
        self.boresch_params = None
        self.optimized_params["lambda_path"] = {}
        
        # #[P1 FIX] 动态探测 lambda 参数名，避免硬编码导致的更新失效问题
        # 🔑 [0831issue P2] `target_phase` 现在可显式指定。默认 "auto" 的优先级表把
        # `lam_coul` 排在 `lam_vdw` 前面（见 _detect_active_parameter），所以对一个
        # **同时注册了两个轴**的探针系统，vdW/vanishing 阶段会沿着 λ_coul 那根轴去
        # 测方差、路径密度权重与目标阶段完全不对应。构造时就知道自己是哪个阶段的
        # 调用方应显式传 "vdw"/"vanishing" 或 "coul"/"decharging"；不传则保持
        # 原来的 auto 行为，逐位不变。
        self.target_phase = str(target_phase or "auto")
        self._active_lambda_param = self._detect_active_parameter(self.target_phase)

    def _detect_active_parameter(self, target_phase: str = "auto") -> str:
        """探测系统中实际使用的 Lambda 参数名（支持双λ动态优先级）"""
        params = []
        try:
            system = self.context.getSystem()
            # 1. 获取 System 级别的全局参数
            for i in range(system.getNumGlobalParameters()):
                params.append(system.getGlobalParameterName(i))
            # 2. 获取所有 Force 级别的全局参数 (CustomForce 通常将 lambda 挂载在此)
            for i in range(system.getNumForces()):
                force = system.getForce(i)
                if hasattr(force, 'getNumGlobalParameters'):
                    for j in range(force.getNumGlobalParameters()):
                        params.append(force.getGlobalParameterName(j))
        except Exception as e:
            print(f"  [WARN] 获取系统参数失败: {e}，探针系统可能未正确注入软核力。")
            return "lam_coul"

        # ✅ 核心修复：根据目标阶段动态调整匹配优先级
        if target_phase.lower() in ("vdw", "vanishing"):
            priority_order = ["lam_vdw", "lambda_vdw", "lam_coul", "lambda_coul"]
        elif target_phase.lower() in ("coul", "decharging"):
            priority_order = ["lam_coul", "lambda_coul", "lam_vdw", "lambda_vdw"]
        else:
            # auto 模式：优先返回已存在的任意 λ 参数
            priority_order = ["lam_coul", "lam_vdw", "lambda_coul", "lambda_vdw"]

        # 1. 精确匹配
        for name in priority_order:
            if name in params:
                print(f"  探测到有效 Lambda 参数: '{name}' (phase={target_phase})")
                return name

        # 2. 模糊匹配（含关键字的候选）
        coul_candidates = [p for p in params if "coul" in p.lower() and "lam" in p.lower()]
        vdw_candidates  = [p for p in params if ("vdw" in p.lower() or "lj" in p.lower()) and "lam" in p.lower()]

        if target_phase.lower() in ("vdw", "vanishing") and vdw_candidates:
            return vdw_candidates[0]
        if coul_candidates:
            return coul_candidates[0]

        # 3. 通用/历史别名回退
        for name in ["lam", "lambda", "lambda1", "LIG_lambda", "lig_lambda"]:
            if name in params:
                print(f"  探测到通用 Lambda 参数: '{name}'")
                return name

        # 4. 终极兜底
        lam_params = [p for p in params if "lam" in p.lower()]
        if lam_params:
            print(f"  模糊匹配到 Lambda 参数: '{lam_params[0]}'")
            return lam_params[0]

        raise RuntimeError("系统中找不到有效的 Lambda 参数名，请检查探针系统构建逻辑")

    def analyze_gradient_and_optimize_path(self, n_steps_per_state: int = 5000) -> Dict:
        """
        [步骤 1.5] 轻量级能量景观分析 (Pathfinding)
        【修复 4】安全参数设置

        🚨 [0831issue P2] **已禁用，与 PHY-08 对 `optimize_stage1_decharging` 的处置
        完全同源同理由**：本方法用 `Var(U_group1)` 当度量（见下方 `np.var(energies)`），
        而生产 PME Hamiltonian 的 Fisher 度量是 `beta² Var[dU/dλ]`。
        `Var(U)` 里混着大量 λ 无关的环境涨落（总势能被溶剂主导），据它排出来的 λ 路径
        不能用于热力学采样。PHY-08 当时只禁掉了同模式的 Stage-1 入口，漏了这个
        同样公开的姐妹入口。

        到达性已核实：唯一调用者是 `abfe_pipeline.ABFEPipeline.run_preoptimization`，
        而 `run_preoptimization` **全仓库没有任何调用者**（连测试都没有）；另一个是
        本类的兼容包装 `run_probing_sampling`。所以这道 fail-closed 不影响任何生产路径，
        作用是让将来复活这条路的人先把度量换成 `beta² Var[dU/dλ]`
        （本类 `_sample_scalar_metric` 已有正确实现，冻结构型有限差分 + force group 隔离）。
        """
        raise RuntimeError(
            "轻量能量景观分析已禁用（0831issue P2 / 同 PHY-08）：旧实现使用 Var(U_group1) "
            "而非生产 PME 的 beta² Var[dU/dlambda]，其路径不能用于热力学采样。"
            "正确度量见 ABFEPreOptimizer._sample_scalar_metric；"
            "生产流程请使用已验证的线性/测地线路径。"
        )

        print(f"\n→ 正在执行能量景观分析 ({n_steps_per_state} 步/状态)... ")

        # === 【修复 4】安全检查参数是否存在 ===
        # abfe_preoptimizer.py -> analyze_gradient_and_optimize_path 方法 (约第 120 行)

        # === 【修复】安全参数设置：先检查后设置 ===
        initial_lam = float(self.lambdas[0])
        active_p = self._active_lambda_param
        param_exists = False
        
        try:
            params_dict = self.context.getParameters()
            if active_p in params_dict:
                self.context.setParameter(active_p, initial_lam)
                param_exists = True
            else:
                # 🔑 修复：兼容别名回退并同步局部变量
                for param_name in ["lambda", "lambda1", "lambda_vdw", "lambda_coul"]:
                    if param_name in params_dict:
                        self.context.setParameter(param_name, initial_lam)
                        self._active_lambda_param = param_name
                        active_p = param_name  # ✅ 关键：同步更新循环使用的变量名
                        param_exists = True
                        print(f"  已回退至 Lambda 别名: '{param_name}'")
                        break
        except (openmm.OpenMMException, AttributeError) as e:
            print(f"  [WARN] 参数 '{active_p}' 设置失败: {e}")
            # ✅ 修复：不 pass，记录失败并尝试强制注入常见名称
            for p_name in list(self.context.getParameters().keys()):
                if "lam" in p_name.lower():
                    try:
                        self.context.setParameter(p_name, initial_lam)
                        active_p = p_name
                        param_exists = True
                        print(f"  [OK] 强制注入成功: {p_name}")
                        break
                    except (openmm.OpenMMException, AttributeError, TypeError, ValueError):
                        continue

        if not param_exists:
            raise RuntimeError(f"[ERR] 无法在 Context 中找到或设置任何 Lambda 参数，优化终止。")
            
        if param_exists:
            print(f"  设置初始 {active_p}={initial_lam:.2f} 进行预平衡...")
            guarded_step(self.context.getIntegrator(), 25000, "单 λ 路径优化：初始预平衡")

        variance_data = []
        mean_energy = []

        # 主采样循环
        for i, lam in enumerate(self.lambdas):
            try:
                self.context.setParameter(active_p, float(lam))  # ✅ 此时 active_p 已是有效名称
            except openmm.OpenMMException as e:
                print(f"  [ERR] 无法设置 Lambda={lam:.3f}: {e}。采样中断。")
                raise

            # 先平衡 500 步再采样
            guarded_step(self.context.getIntegrator(), 500, "单 λ 路径优化：采样前平衡")

            energies = []
            nan_count = 0
            n_sampled = 0

            # 🔑 [0831issue P2 / PHY-08 同类] NaN/Inf 样本必须**丢弃**，不能替换成
            # "前一帧的值"或 0.0 再计入方差。旧写法把坏帧换成前值后照样 append，于是
            # (a) 方差被人为压低（重复值零离差），(b) 首帧就坏时注入一个纯虚构的 0.0，
            # 两者都直接歪曲这个度量，而它是 λ 路径密度的唯一依据。
            # nan_count/n_sampled 的比例判据保持原语义（分母仍是总采样帧数）。
            for e in _sample_group1_energies(self.context, n_steps_per_state, sample_interval=50):
                n_sampled += 1
                if np.isnan(e) or np.isinf(e):
                    nan_count += 1
                    continue
                energies.append(e)

            if nan_count > n_sampled * 0.5:
                print(
                    f"  [WARN] lam={lam:.2f} 能量异常过多 ({nan_count}/{n_sampled})，使用默认值 "
                )
                variance_data.append(1.0)
                mean_energy.append(0.0)
            elif len(energies) > 1:
                variance = np.var(energies)
                if np.isnan(variance):
                    variance = 1.0
                variance_data.append(variance)
                mean_energy.append(np.mean(energies))
            else:
                variance_data.append(1.0)
                mean_energy.append(energies[0] if energies else 0.0)

        variance_data = np.array(variance_data)
        std_dev = np.sqrt(variance_data + 1e-10)

        # === 【修复 4】方差截断 (防止异常值主导) ===
        threshold = np.percentile(std_dev, 90) * 2.0  # ✅ 从 3.0 改为 2.0 更保守
        if np.isnan(threshold):
            threshold = 10.0

        std_dev_clipped = np.clip(std_dev, None, threshold)
        norm_variance = std_dev_clipped / (np.max(std_dev_clipped) + 1e-6)

        print(
            f"  [OK] 能量景观分析完成。最大标准差位置：lam={self.lambdas[np.argmax(std_dev)]:.2f} "
        )
        print(f"  [OK] 方差截断阈值：{threshold:.2f} (原始最大：{np.max(std_dev):.2f}) ")

        return {
            "variance": variance_data,
            "std_dev": std_dev,
            "std_dev_clipped": std_dev_clipped,
            "norm_variance": norm_variance,
            "mean_energy": mean_energy,
        }

    def optimize_softcore_parameters(
        self, ligand_indices: List[int]
    ) -> ACESoftcorePotential:
        """[步骤 1] 优化软核参数"""
        n_ligand_atoms = len(ligand_indices)
        params = ACESoftcorePotential.optimize_alpha(n_ligand_atoms)

        softcore_obj = ACESoftcorePotential(
            alpha_lj=params["alpha_lj"],
            alpha_coul=params["alpha_coul"],
            power_lj=params["power_lj"],
            power_coul=params["power_coul"],
        )

        self.optimized_params["softcore"] = softcore_obj
        print(
            f"→ 软核参数已优化：α_LJ={softcore_obj.alpha_lj}, α_Coul={softcore_obj.alpha_coul}"
        )

        return softcore_obj

    def generate_lambda_path(
        self, phase: str = "vdw", n_windows: int = 4, states_per_window: int = 10
    ) -> List[float]:
        """
        [步骤 2 备选] Lambda 路径预设（如果不用 Pathfinding）
        【修复】添加此方法以兼容 openmm_abfe_pipeline.py
        """
        total_states = n_windows * states_per_window

        if phase == "vdw":
            lambdas = (np.linspace(1.0, 0.0, total_states) ** 2).tolist()
            print(f"→ VdW 阶段：生成 {total_states} 状态非线性 Lambda 路径 (λ²)")
        elif phase == "charge":
            lambdas = np.linspace(1.0, 0.0, total_states).tolist()
            print(f"→ 电荷阶段：生成 {total_states} 状态线性 Lambda 路径")
        else:
            lambdas = np.linspace(1.0, 0.0, total_states).tolist()

        self.optimized_params["lambda_path"] = {
            "phase": phase,
            "n_windows": n_windows,
            "states_per_window": states_per_window,
            "distribution": "nonlinear" if phase == "vdw" else "linear",
        }

        self.lambdas = np.array(lambdas)
        self.n_states = len(lambdas)

        return lambdas

    def optimize_window_ranges(
        self, n_ib_windows: int = 4, overlap: int = 3
    ) -> List[Tuple[int, int]]:
        """[步骤 3] IBS 窗口划分"""
        total = self.n_states

        if total <= 6 or n_ib_windows == 1:
            ranges = [(0, total)]
            print(f"→ 状态数较少 ({total})，使用单窗口：{ranges}")
            return ranges

        min_window_size = overlap + 1
        if total < n_ib_windows * min_window_size:
            n_ib_windows = max(1, total // min_window_size)
            print(f"  [WARN] 状态数不足，调整窗口数为 {n_ib_windows}")

        if n_ib_windows > 1:
            step = (total - overlap) // n_ib_windows
            step = max(1, step)
        else:
            step = total

        ranges = []
        for i in range(n_ib_windows):
            start = i * step
            if start >= total:
                break

            if i < n_ib_windows - 1:
                end = start + step + overlap
            else:
                end = total

            end = min(end, total)
            if end > start:
                ranges.append((start, end))

            if end == total:
                break

        if not ranges or ranges[-1][1] < total:
            ranges = []
            simple_step = max(1, (total - overlap) // n_ib_windows)
            for i in range(n_ib_windows):
                s = i * simple_step
                if i < n_ib_windows - 1:
                    e = s + simple_step + overlap
                else:
                    e = total
                e = min(e, total)
                if s < e:
                    ranges.append((s, e))
            if ranges and ranges[-1][1] < total:
                ranges[-1] = (ranges[-1][0], total)

        print(f"→ ACES 建议窗口划分 ({len(ranges)} 个): {ranges}")
        return ranges

    def optimize_window_ranges_for_ibes(
        self, n_ib_windows: int = 3, overlap: int = 4
    ) -> List[Tuple[int, int]]:
        """[步骤 3 备选] IBS 窗口划分（别名）"""
        return self.optimize_window_ranges(n_ib_windows=n_ib_windows, overlap=overlap)

    def get_optimization_report(self) -> Dict:
        """获取优化报告"""
        softcore_dict = {}
        if "softcore" in self.optimized_params:
            sc = self.optimized_params["softcore"]
            if hasattr(sc, "alpha_lj"):
                softcore_dict = sc.get_parameters_dict()

        return {
            "temperature": float(self.temperature),
            "n_states": int(self.n_states),
            "softcore_params": softcore_dict,
            "lambda_path": self.optimized_params.get("lambda_path", {}),
            "initial_weights": self.initial_weights.tolist()
            if self.initial_weights is not None
            else None,
            "boresch_correction": self.boresch_params.get("analytical_correction")
            if self.boresch_params
            else None,  # ✅ 现在可以安全访问
        }

    # =============================================================================
    # 替换 optimize_lambda_path_adaptive 方法 (完整修复版)
    # =============================================================================
    def optimize_lambda_path_adaptive(
        self,
        landscape_data,
        target_n_states: int = None,
        charge_exponent: float = 2.0,
        vdw_exponent: float = 1.0,
    ) -> List[float]:
        """
        [步骤 1.6] 根据能量方差自适应调整 Lambda 分布

        【关键修复】
        1. 确保插值前 Lambda 序列转为升序 (np.interp 要求 xp 递增)
        2. 使用对数平滑方差，防止单个点主导
        3. 强制边界为 1.0 和 0.0
        4. 添加最小间距检查，防止负数
        """
        if "lambda_path" not in self.optimized_params:
            self.optimized_params["lambda_path"] = {}

        # === 目标状态数处理 ===
        if target_n_states is None:
            target_n_states = self.n_states

        # 【修复】确保 target_n_states 至少为 12
        if target_n_states < 12:
            print(f"  [WARN] 目标状态数 ({target_n_states}) 太少，调整为 12 ")
            target_n_states = 12

        # === 检查 landscape_data 有效性 ===
        if landscape_data is None or landscape_data.get("std_dev_clipped") is None:
            print("  [WARN] landscape_data 无效，使用线性 Lambda 路径 ")
            return np.linspace(1.0, 0.0, target_n_states).tolist()

        # === 【步骤 1】获取方差数据 ===
        std_dev_clipped = landscape_data["std_dev_clipped"].copy()

        # === 【步骤 2】长度检查与对齐 ===
        if len(std_dev_clipped) != len(self.lambdas):
            print(
                f"[WARN] 警告：std_dev_clipped 长度 ({len(std_dev_clipped)}) 与 self.lambdas 长度 ({len(self.lambdas)}) 不匹配 "
            )
            min_len = min(len(std_dev_clipped), len(self.lambdas))
            std_dev_clipped = std_dev_clipped[:min_len]
            self.lambdas = self.lambdas[:min_len]
            self.n_states = min_len

        # === 【步骤 3】方差平滑 (对数化防止极值主导) ===
        # 【关键修复】使用 log1p 平滑，缓解λ=1.0 处 Clash 带来的极值影响
        try:
            from scipy.ndimage import gaussian_filter1d

            std_dev_smooth = gaussian_filter1d(std_dev_clipped, sigma=1)
            std_dev_smooth[0] = std_dev_clipped[0]
            std_dev_smooth[-1] = std_dev_clipped[-1]
            std_dev_clipped = std_dev_smooth
        except ImportError:
            pass

        # === 【步骤 4】计算密度权重 (使用对数缩放 + 软归一化) ===
        # ✅ 修复10：使用对数缩放 + 软归一化，避免硬截断破坏概率密度
        MAX_RATIO = 0.10  # ✅ 显式声明，避免后续引用报错
        log_std = np.log1p(std_dev_clipped + 1e-6)
        # 归一化为概率密度，保证 ∫ρ(x)dx = 1
        density_weight = log_std / (np.sum(log_std) + 1e-10)
        
        # 保留高λ区加密逻辑
        for i, lam in enumerate(self.lambdas):
            if lam > 0.8:
                density_weight[i] *= 1.5
        # 重新归一化
        density_weight /= np.sum(density_weight)

        # === 【步骤 7】累积分布与插值 ===
        # 🔑 [0831issue P2] 构造与 lambda 节点一一对应的单调 CDF：首节点 0、末节点 1。
        # 旧写法是 `xp = [0] + cumsum(w)[:-1]/sum(w)` 然后把末元素**覆盖**成 1.0 —— 那个赋值
        # **覆盖**掉了倒数第二个累积坐标 c_{N-2}/T，于是最后一个区间的宽度从
        # w[N-2] 变成 w[N-2]+w[N-1]，λ[N-2] 的权重被双重计入，λ→0 尾段的加密方向失真。
        # 正解：N 个节点之间只有 N-1 个区间，就用前 N-1 个权重当区间宽度、并按
        # **它们自己的和**归一化——末端于是天然等于 1.0，不需要事后覆盖。
        interval_weights = np.asarray(density_weight, dtype=float)[:-1]
        interval_total = max(1e-10, float(np.sum(interval_weights)))
        xp = np.concatenate(([0.0], np.cumsum(interval_weights) / interval_total))

        # 原始 lambdas 是降序 [1.0, ..., 0.0]，长度必须与 xp 严格一致。
        original_lambdas = np.asarray(self.lambdas.copy(), dtype=float)
        lambda_xp = original_lambdas

        if HAS_SCIPY and len(xp) >= 3:
            # ✅ 使用 PCHIP 保持单调性，无需手动翻转
            # 注意：xp 递增，original_lambdas 递减 → 插值函数自动处理反向映射
            interp_func = PchipInterpolator(xp, lambda_xp, extrapolate=False)
            target_cumulative = np.linspace(0, 1.0, target_n_states)
            optimized_lambdas = interp_func(target_cumulative)
        else:
            # 回退到原逻辑 (带翻转)
            target_cumulative = np.linspace(0, 1.0, target_n_states)
            unique_xp, idx_map = np.unique(xp, return_index=True)
            fp_filtered = lambda_xp[idx_map]
            xp = unique_xp
            if len(unique_xp) < 2:
                return np.linspace(1.0, 0.0, target_n_states).tolist()
            optimized_lambdas = np.interp(target_cumulative, xp, fp_filtered)

        optimized_lambdas = np.asarray(optimized_lambdas, dtype=float).ravel()
        if not np.all(np.isfinite(optimized_lambdas)):
            print("  [WARN] 自适应插值产生非有限 λ，使用线性路径")
            optimized_lambdas = np.linspace(1.0, 0.0, target_n_states)

        # === 【步骤 8】边界强制、最小间距与去重 (共享纯函数，见
        # finalize_descending_lambda_path；DualLambdaPreOptimizer.
        # optimize_stage1_decharging 复用同一份逻辑) ===
        optimized_lambdas, min_spacing, fell_back = finalize_descending_lambda_path(
            optimized_lambdas, target_n_states
        )
        if fell_back:
            print(f"  [WARN] 去重后状态数少于目标 ({target_n_states})，使用线性路径 ")

        if not (np.isclose(optimized_lambdas[0], 1.0) and np.isclose(optimized_lambdas[-1], 0.0)):
            raise RuntimeError(
                f"优化后的 lambda 路径端点异常: first={optimized_lambdas[0]}, last={optimized_lambdas[-1]}"
            )

        # === 【步骤 9】更新状态 ===
        self.optimized_params["lambda_path"].update(
            {
                "method": "adaptive_variance_v6",
                "n_states": len(optimized_lambdas),
                "target_n_states": target_n_states,
                "log_scaling": True,
                "max_ratio": MAX_RATIO,
                "min_spacing": min_spacing,
            }
        )

        self.lambdas = np.array(optimized_lambdas)
        self.n_states = len(optimized_lambdas)

        # === 输出诊断信息 ===
        print(f"→ Lambda 路径已优化。高方差区已加密。 ")
        print(
            f"  Lambda 范围：[{np.min(optimized_lambdas):.3f}, {np.max(optimized_lambdas):.3f}] "
        )
        print(f"  总状态数：{len(optimized_lambdas)} (目标：{target_n_states}) ")
        print(f"  Lambda 间距：{np.diff(optimized_lambdas)} ")

        # 【关键验证】检查是否有负数
        if np.any(optimized_lambdas < 0.0):
            print(f"  [WARN] 警告：检测到负 Lambda 值，已修正 ")
        if np.any(optimized_lambdas > 1.0):
            print(f"  [WARN] 警告：检测到 Lambda>1.0，已修正 ")

        # === 确保返回 list ===
        return optimized_lambdas.tolist()

    # 在 ABFEPreOptimizer 类中添加以下兼容方法

    def run_probing_sampling(self, n_steps: int = 5000) -> Dict:
        """【兼容方法】调用 analyze_gradient_and_optimize_path"""
        return self.analyze_gradient_and_optimize_path(n_steps_per_state=n_steps)

    def optimize_path(self, landscape_data: Dict) -> List[float]:
        """【兼容方法】调用 optimize_lambda_path_adaptive"""
        return self.optimize_lambda_path_adaptive(landscape_data=landscape_data)

    # =============================================================================
    # 在 ABFEPreOptimizer 类中添加窗口划分方法
    # =============================================================================
    # 替换原 partition_ibs_windows_fixed 方法体为：
    def partition_ibs_windows_fixed(self, n_states: int = None, n_ib_windows: int = 4, pts_per_window: int = 6, overlap: int = 2) -> List[Tuple[int, int]]:
        if n_states is None: n_states = self.n_states
        windows = generate_overlapping_windows(
            n_states,
            pts_per_window=pts_per_window,
            overlap=overlap,
            n_windows=n_ib_windows,
        )
        print(f"→ IBS 窗口划分 ({len(windows)} 个): {windows} (覆盖 {n_states} 个状态)")
        return windows


# =============================================================================
# λ 路径 pilot 探针 shadow early-stop 诊断（Phase A，2026-08-26）
# =============================================================================
# 下面这组是纯 Python/numpy 函数，不依赖 OpenMM Context，可离线单测。目的是
# 回答"Stage2 vanishing pilot 的 n_steps_per_state=30000 是不是处处都要跑
# 满"——但本阶段（Phase A）只做诊断/记录，不改变任何真实采样长度：
# `_sample_scalar_metric`/`optimize_stage2_vanishing`/
# `_refine_pilot_grid_in_steep_segments` 在 shadow_checkpoint_steps /
# shadow_checkpoint_interval 为 None（默认值）时逐字节保持原行为不变。
#
# 背景（详见 abfe_pipeline.py 里 "vanishing" 分支调用 _run_dual_lambda_
# optimization 处 2026-07-19 的原地注释）：那次真实 GPU 回归发现 10000 步的
# 短 pilot 会系统性低估 λ≈1 端点由稀有/发作性事件主导的
# beta^2*Var[dU/dlambda]，才把预算拉长到当前生产用的 30000。任何缩短 pilot
# 预算的方案都必须先证明不会重新踩这个坑——这组函数只是用来在真机上收集
# "如果提前停会怎样"的影子数据供之后离线验证，本身不做任何提前停的决定。


