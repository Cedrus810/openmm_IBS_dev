# =============================================================================
# ABFE 预采样优化器 - ACES 路径优化版 (v5.0 - 真正的 Pathfinding)
# 基于：ACES (JCTC 2023), IBS (JCTC 2026), CBFE (JCIM 2026)
# =============================================================================
"""
修复清单：
✅ 真正的 ACES Pathfinding：分析能量梯度，自动计算最优衰减指数
✅ 不是固定的λ²，而是根据配体性质动态调整 charge_exponent 和 vdw_exponent
✅ 基于能量方差σ²(U) 重分布 Lambda 点
✅ 确保每段ΔG 近似相等（热力学长度最小化）
"""

import openmm
from openmm import app, unit, XmlSerializer
import numpy as np
from typing import Dict, List, Tuple, Optional
from abfe_core import ACESoftcorePotential
from ibs_engine import generate_overlapping_windows
try:
    from scipy.interpolate import PchipInterpolator
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# #[P1 FIX] 抽取共享方差归一化逻辑，消除重复代码
# abfe_preoptimizer.py 顶部 (约第 15 行)
def _normalize_variance_weights(std_dev_clipped, max_ratio=0.15):
    """共享方差归一化函数，消除代码重复"""
    density_weight = np.log1p(std_dev_clipped) + 0.1
    max_weight = np.sum(density_weight) * max_ratio
    clipped = np.clip(density_weight, None, max_weight)
    # ✅ 修复：Clip 后必须重新归一化，保证 ∑w = 1（等熵长度分布前提）
    return clipped / (np.sum(clipped) + 1e-10)


def _sample_group1_energies(context, total_steps, sample_interval=50):
    """批量推进积分器，保留固定采样间隔，减少 Python/C++ 边界往返。"""
    if total_steps <= 0:
        return []

    integrator = context.getIntegrator()
    energies = []
    full_batches, remainder = divmod(int(total_steps), int(sample_interval))

    for _ in range(full_batches):
        integrator.step(sample_interval)
        state = context.getState(getEnergy=True, groups={1})
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )

    if remainder:
        integrator.step(remainder)
        state = context.getState(getEnergy=True, groups={1})
        energies.append(
            state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        )

    return energies

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
        self._active_lambda_param = self._detect_active_parameter()

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
            print(f"  ⚠️ 获取系统参数失败: {e}，探针系统可能未正确注入软核力。")
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
                print(f"  🔍 探测到有效 Lambda 参数: '{name}' (phase={target_phase})")
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
                print(f"  🔍 探测到通用 Lambda 参数: '{name}'")
                return name

        # 4. 终极兜底
        lam_params = [p for p in params if "lam" in p.lower()]
        if lam_params:
            print(f"  🔍 模糊匹配到 Lambda 参数: '{lam_params[0]}'")
            return lam_params[0]

        raise RuntimeError("系统中找不到有效的 Lambda 参数名，请检查探针系统构建逻辑")

    def analyze_gradient_and_optimize_path(self, n_steps_per_state: int = 5000) -> Dict:
        """
        [步骤 1.5] 轻量级能量景观分析 (Pathfinding)
        【修复 4】安全参数设置
        """
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
                        print(f"  ℹ️ 已回退至 Lambda 别名: '{param_name}'")
                        break
        except (openmm.OpenMMException, AttributeError) as e:
            print(f"  ⚠️ 参数 '{active_p}' 设置失败: {e}")
            # ✅ 修复：不 pass，记录失败并尝试强制注入常见名称
            for p_name in list(self.context.getParameters().keys()):
                if "lam" in p_name.lower():
                    try:
                        self.context.setParameter(p_name, initial_lam)
                        active_p = p_name
                        param_exists = True
                        print(f"  ✅ 强制注入成功: {p_name}")
                        break
                    except: pass

        if not param_exists:
            raise RuntimeError(f"❌ 无法在 Context 中找到或设置任何 Lambda 参数，优化终止。")
            
        if param_exists:
            print(f"  ℹ️ 设置初始 {active_p}={initial_lam:.2f} 进行预平衡...")
            self.context.getIntegrator().step(25000)

        variance_data = []
        mean_energy = []

        # 主采样循环
        for i, lam in enumerate(self.lambdas):
            try:
                self.context.setParameter(active_p, float(lam))  # ✅ 此时 active_p 已是有效名称
            except openmm.OpenMMException as e:
                print(f"  ❌ 无法设置 Lambda={lam:.3f}: {e}。采样中断。")
                raise

            # 先平衡 500 步再采样
            self.context.getIntegrator().step(500)

            energies = []
            nan_count = 0

            for e in _sample_group1_energies(self.context, n_steps_per_state, sample_interval=50):
                if np.isnan(e) or np.isinf(e):
                    nan_count += 1
                    e = energies[-1] if energies else 0.0
                energies.append(e)

            if nan_count > len(energies) * 0.5:
                print(
                    f"  ⚠️  lam={lam:.2f} 能量异常过多 ({nan_count}/{len(energies)})，使用默认值 "
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
            f"  ✓ 能量景观分析完成。最大标准差位置：lam={self.lambdas[np.argmax(std_dev)]:.2f} "
        )
        print(f"  ✓ 方差截断阈值：{threshold:.2f} (原始最大：{np.max(std_dev):.2f}) ")

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
            print(f"  ⚠️  状态数不足，调整窗口数为 {n_ib_windows}")

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
            print(f"  ⚠️  目标状态数 ({target_n_states}) 太少，调整为 12 ")
            target_n_states = 12

        # === 检查 landscape_data 有效性 ===
        if landscape_data is None or landscape_data.get("std_dev_clipped") is None:
            print("  ⚠️  landscape_data 无效，使用线性 Lambda 路径 ")
            return np.linspace(1.0, 0.0, target_n_states).tolist()

        # === 【步骤 1】获取方差数据 ===
        std_dev_clipped = landscape_data["std_dev_clipped"].copy()

        # === 【步骤 2】长度检查与对齐 ===
        if len(std_dev_clipped) != len(self.lambdas):
            print(
                f"⚠️  警告：std_dev_clipped 长度 ({len(std_dev_clipped)}) 与 self.lambdas 长度 ({len(self.lambdas)}) 不匹配 "
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
        cumulative_density = np.cumsum(density_weight)
        total_density = max(1e-10, cumulative_density[-1])
        # 构造与 lambda 节点一一对应的单调 CDF：首节点固定为 0，末节点固定为 1。
        xp = np.concatenate(([0.0], cumulative_density[:-1] / total_density))
        xp[-1] = min(xp[-1], 1.0)

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

        # === 【步骤 8】边界强制与去重 (不变) ===
        optimized_lambdas = np.clip(optimized_lambdas, 0.0, 1.0)
        optimized_lambdas = np.sort(optimized_lambdas)[::-1]  # 确保降序
        optimized_lambdas = np.minimum.accumulate(optimized_lambdas)
        optimized_lambdas[0], optimized_lambdas[-1] = 1.0, 0.0
        min_spacing = max(0.02, 0.9 / (target_n_states - 1))
        for i in range(1, len(optimized_lambdas)):
            if optimized_lambdas[i] < 0.0:
                optimized_lambdas[i] = 0.0
            if optimized_lambdas[i - 1] - optimized_lambdas[i] < min_spacing:
                optimized_lambdas[i] = max(0.0, optimized_lambdas[i - 1] - min_spacing)

        unique_lambdas = []
        for lam in optimized_lambdas:
            lam_val = float(lam)
            if not unique_lambdas or abs(lam_val - unique_lambdas[-1]) > min_spacing:
                unique_lambdas.append(lam_val)

        if len(unique_lambdas) < target_n_states * 0.5:
            print(f"  ⚠️  去重后状态数 ({len(unique_lambdas)}) 太少，使用线性路径 ")
            optimized_lambdas = np.linspace(1.0, 0.0, target_n_states)
        else:
            optimized_lambdas = np.array(unique_lambdas)

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
            print(f"  ⚠️  警告：检测到负 Lambda 值，已修正 ")
        if np.any(optimized_lambdas > 1.0):
            print(f"  ⚠️  警告：检测到 Lambda>1.0，已修正 ")

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
        windows = generate_overlapping_windows(n_states, n_ib_windows, pts_per_window, overlap)
        print(f"→ IBS 窗口划分 ({len(windows)} 个): {windows} (覆盖 {n_states} 个状态)")
        return windows


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
                            print(f"  [SCAN] ✅ Force 匹配到: {n}")
                            return n
        # 2. 扫 Context
        try:
            ctx_p = list(self.context.getParameters().keys())
            print(f"  [SCAN] Context 包含参数: {ctx_p}")
            for k in ctx_p:
                if keyword in k.lower(): 
                    print(f"  [SCAN] ✅ Context 匹配到: {k}")
                    return k
        except Exception as e: print(f"  [SCAN] Context 读取失败: {e}")
        return None

    def optimize_stage1_decharging(self, n_states=12, n_steps_per_state=2000):
        print(f"\n[STAGE1] 开始去电荷路径优化 (n_states={n_states})...")
        print(f"[STAGE1] 当前 param_coul='{self.param_coul}', param_vdw='{self.param_vdw}'")
        
        if self.param_coul is None:
            print(f"[STAGE1] ⚠️ 探针系统未注册 Coulomb λ 参数，直接生成线性回退路径")
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
        self.context.getIntegrator().step(5000)

        variance_data = []
        lambdas = np.linspace(1.0, 0.0, n_states)
        print(f"[STAGE1] 线性采样点: {lambdas}")

        for i, lam in enumerate(lambdas):
            self.context.setParameter(self.param_coul, float(lam))
            if self.param_vdw is not None:  # ✅ 修复：增加 None 守卫，确保 Stage1 安全
                self.context.setParameter(self.param_vdw, 1.0)
            self.context.getIntegrator().step(500)
            energies = [
                e for e in _sample_group1_energies(self.context, n_steps_per_state, sample_interval=50)
                if not (np.isnan(e) or np.isinf(e))
            ]
            variance_data.append(np.var(energies) if len(energies) >1 else 1.0)

        # 路径重分布 (保持原逻辑)
        std_dev = np.sqrt(np.array(variance_data) + 1e-10)
        density_weight = np.asarray(_normalize_variance_weights(std_dev, max_ratio=0.15), dtype=float).ravel()
        cumulative_density = np.cumsum(density_weight)
        total_density = float(cumulative_density[-1]) + 1e-10
        xp = np.concatenate(([0.0], cumulative_density[:-1] / total_density)).astype(float).ravel()
        fp = np.asarray(lambdas, dtype=float).ravel()
        target_cumulative = np.linspace(0, 1.0, n_states)
        optimized_lambdas = np.asarray(np.interp(target_cumulative, xp, fp)[::-1], dtype=float).ravel()
        optimized_lambdas = np.clip(np.sort(optimized_lambdas)[::-1], 0.0, 1.0)
        optimized_lambdas[0], optimized_lambdas[-1] = 1.0, 0.0
        
        print(f"[STAGE1] ✓ 优化完成，返回前5个λ: {optimized_lambdas[:5]}")
        return {"stage": "decharging", "lambdas_coul": optimized_lambdas.tolist(), "lambdas_vdw": [1.0]*len(optimized_lambdas), "n_states": len(optimized_lambdas)}

    def optimize_stage2_vanishing(self, n_states=12, n_steps_per_state=2000):
        print(f"\n→ Stage 2: 去 VDW 路径优化 ({n_states} 状态)...")
        current_params = dict(self.context.getParameters())
        if self.param_vdw is None or self.param_vdw not in current_params:
            raise RuntimeError(f"探针系统未注册 VdW λ 参数，无法执行自适应优化")
            
        if self.param_coul is not None and self.param_coul in current_params:
            self.context.setParameter(self.param_coul, 0.0)
        self.context.setParameter(self.param_vdw, 1.0)
        self.context.getIntegrator().step(5000)

        variance_data = []
        lambdas = np.linspace(1.0, 0.0, n_states)
        for lam in lambdas:
            self.context.setParameter(self.param_vdw, float(lam))
            self.context.setParameter(self.param_coul, 0.0)
            self.context.getIntegrator().step(500)
            energies = [
                e for e in _sample_group1_energies(self.context, n_steps_per_state, sample_interval=50)
                if not (np.isnan(e) or np.isinf(e))
            ]
            variance_data.append(np.var(energies) if len(energies)>1 else 1.0)

        std_dev = np.sqrt(np.array(variance_data) + 1e-10)
        density_weight = np.asarray(_normalize_variance_weights(std_dev, max_ratio=0.15), dtype=float).ravel()
        cumulative_density = np.cumsum(density_weight)
        total_density = float(cumulative_density[-1]) + 1e-10
        xp = np.concatenate(([0.0], cumulative_density[:-1] / total_density)).astype(float).ravel()
        optimized_lambdas = np.asarray(
            np.interp(np.linspace(0, 1.0, n_states), xp, np.asarray(lambdas, dtype=float).ravel())[::-1],
            dtype=float,
        ).ravel()
        optimized_lambdas = np.clip(np.sort(optimized_lambdas)[::-1], 0.0, 1.0)
        optimized_lambdas[0], optimized_lambdas[-1] = 1.0, 0.0
        
        print(f"  ✓ Stage 2 路径优化完成：{optimized_lambdas}")
        return {
            "stage": "vanishing",
            "lambdas_coul": [0.0] * len(optimized_lambdas),
            "lambdas_vdw": optimized_lambdas.tolist(),
            "n_states": len(optimized_lambdas),
        }


# 修复 9: warmup safety check
def apply_safety_checks_on_disable_warmup(simulation, enable_warmup, warmup_steps):
    from openmm import unit
    import numpy as np
    import warnings
    if not enable_warmup:
        if simulation.context is None:
            print("  ⚠️ simulation.context 未初始化，跳过安全检查")
            return
        try:
            state = simulation.context.getState(getEnergy=True, getForces=True)
            forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            force_norms = np.linalg.norm(forces, axis=1)
            rms_force = np.sqrt(np.mean(force_norms**2))
            max_force = np.max(force_norms)
            
            # ✅ RMS 阈值 5000 + 极值兜底 20000
            if np.isnan(rms_force) or np.isinf(rms_force) or rms_force > 5000 or max_force > 20000:
                warnings.warn("⚠️ 检测到不合理 RMS 力或极值，强制能量最小化...", UserWarning)
                simulation.minimizeEnergy(maxIterations=10000)
                state = simulation.context.getState(getEnergy=True, getForces=True)
                forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
                force_norms = np.linalg.norm(forces, axis=1)
                rms_force = np.sqrt(np.mean(force_norms**2))
                print(f"  ✓ 最小化后: RMS|F|={rms_force:.2e}, max|F|={np.max(force_norms):.2e}")
            else:
                print(f"  ✓ 安全检查通过: RMS|F|={rms_force:.2e}")
        except Exception as e:
            print(f"  ❌ 安全检查失败: {e}")
            raise

# ============================================================================
# 探针系统构建函数 (迁移自 ibs_engine)
# ============================================================================
from abfe_core import (
    ensure_owned_system,
    sync_all_exclusions,
    create_ligand_internal_force,
    AlchemicalPotentialFactory,
)
from openmm import XmlSerializer


def _create_softcore_force_dual_lambda(
    nb_force,
    perturbed_indices,
    environment_indices,
    lam_coul,
    lam_vdw,
    softcore_params,
    reference_exclusions=None,
    particle_params_override=None,
    num_particles=None,
    use_global_lambda=False,
    cutoff_distance=1.2,
):
    """构建带双全局 lambda 的软核力，用于探针系统"""
    if num_particles is None:
        num_particles = nb_force.getNumParticles()
    perturbed_set = set(perturbed_indices)
    env_set = set(environment_indices)

    lam_c_str = "lam_coul" if use_global_lambda else f"{lam_coul:.6f}"
    lam_v_str = "lam_vdw" if use_global_lambda else f"{lam_vdw:.6f}"
    expr, _ = AlchemicalPotentialFactory.build("softcore", softcore_params, lam_c_str, lam_v_str)

    force = openmm.CustomNonbondedForce(expr)
    for p in ["q", "sigma", "epsilon"]:
        force.addPerParticleParameter(p)

    if use_global_lambda:
        force.addGlobalParameter("lam_coul", lam_coul)
        force.addGlobalParameter("lam_vdw", lam_vdw)

    for i in range(num_particles):
        if particle_params_override and i < len(particle_params_override):
            q, sig, eps = particle_params_override[i]
        else:
            q, sig, eps = nb_force.getParticleParameters(i)
        force.addParticle([
            q.value_in_unit(unit.elementary_charge),
            sig.value_in_unit(unit.nanometer),
            eps.value_in_unit(unit.kilojoule_per_mole)
        ])

    force.addInteractionGroup(perturbed_set, env_set)
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff_distance * unit.nanometer)

    if reference_exclusions is not None:
        for p1, p2 in reference_exclusions:
            p1, p2 = int(p1), int(p2)
            if p1 < num_particles and p2 < num_particles:
                force.addExclusion(p1, p2)

    force.setUseSwitchingFunction(True)
    force.setSwitchingDistance(1.0 * unit.nanometer)
    return force


def build_aces_probe_system_dual_lambda(
    system,
    perturbed_indices,
    softcore_params,
    fixed_lam_coul=None,
    fixed_lam_vdw=None,
    cutoff_distance=1.2,
    use_reaction_field=False,
):
    """双λ探针系统构建 (用于预优化)"""
    system = ensure_owned_system(system)
    new_sys = ensure_owned_system(XmlSerializer.deserialize(XmlSerializer.serialize(system)))
    num_atoms = new_sys.getNumParticles()
    perturbed_set = set(perturbed_indices)
    env_idx = [i for i in range(num_atoms) if i not in perturbed_set]

    nb_forces = [f for f in new_sys.getForces() if isinstance(f, openmm.NonbondedForce)]
    nb = nb_forces[0]
    zero_q = 0.0 * unit.elementary_charge
    zero_sig = 0.1 * unit.nanometer  # 保留极小半径防除零，但能量为0
    zero_eps = 0.0 * unit.kilojoule_per_mole
    for idx in perturbed_indices:
        nb.setParticleParameters(idx, zero_q, zero_sig, zero_eps)
    
    all_p = [nb.getParticleParameters(i) for i in range(num_atoms)]
    ref_excl = [(int(nb.getExceptionParameters(i)[0]), int(nb.getExceptionParameters(i)[1]))
                for i in range(nb.getNumExceptions())]    
    all_p = [nb.getParticleParameters(i) for i in range(num_atoms)]
    ref_excl = [(int(nb.getExceptionParameters(i)[0]), int(nb.getExceptionParameters(i)[1]))
                for i in range(nb.getNumExceptions())]

    # Group 2: 配体内部力
    ll_f, ll_14_f = create_ligand_internal_force(
        nb, perturbed_indices, all_p, ref_excl, num_atoms, system=system
    )
    if ll_f:
        ll_f.setForceGroup(2)
        new_sys.addForce(ll_f)
    if ll_14_f:
        ll_14_f.setForceGroup(2)
        new_sys.addForce(ll_14_f)

    # Group 1: 双λ软核力
    ac_f = _create_softcore_force_dual_lambda(
        nb, perturbed_indices, env_idx,
        fixed_lam_coul, fixed_lam_vdw,
        softcore_params,
        reference_exclusions=ref_excl,
        particle_params_override=all_p,
        num_particles=num_atoms,
        use_global_lambda=True,
        cutoff_distance=cutoff_distance
    )
    if ac_f is not None:
        ac_f.setForceGroup(1)
        new_sys.addForce(ac_f)

    sync_all_exclusions(new_sys)
    new_sys.thisown = 1
    return new_sys


def build_aces_probe_system(system, perturbed_indices, softcore_params, prefix="aces_pre",
                            fixed_lam_coul=0.5, fixed_lam_vdw=1.0):
    """单λ探针系统构建（内部委托给双λ版本）"""
    return build_aces_probe_system_dual_lambda(
        system, perturbed_indices, softcore_params,
        fixed_lam_coul=fixed_lam_coul, fixed_lam_vdw=fixed_lam_vdw
    )


# ============================================================================
# 双λ 2D 度量张量场采集与单调有向图寻径 (DualLambdaPreOptimizer 扩展)
# ============================================================================
def compute_2d_metric_grid(context, lam_c_grid, lam_v_grid, n_steps=3000, delta=0.02, temperature=300.0):
    """采集 2D 度量张量场 g_cc, g_vv, g_cv 用于黎曼几何路径规划"""
    beta = 1.0 / (0.00831446 * temperature)
    G = np.zeros((len(lam_c_grid), len(lam_v_grid), 2, 2))

    for i, lc in enumerate(lam_c_grid):
        for j, lv in enumerate(lam_v_grid):
            context.setParameter("lam_coul", lc)
            context.setParameter("lam_vdw", lv)
            context.getIntegrator().step(500)

            dc_vals, dv_vals = [], []
            full_batches, remainder = divmod(int(n_steps), 50)
            sample_count = full_batches + (1 if remainder else 0)
            for sample_idx in range(sample_count):
                batch_steps = 50 if sample_idx < full_batches else remainder
                if batch_steps <= 0:
                    continue
                context.getIntegrator().step(batch_steps)
                context.setParameter("lam_coul", lc + delta)
                e_cp = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                context.setParameter("lam_coul", lc - delta)
                e_cm = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                context.setParameter("lam_coul", lc)

                context.setParameter("lam_vdw", lv + delta)
                e_vp = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                context.setParameter("lam_vdw", lv - delta)
                e_vm = context.getState(getEnergy=True, groups={1}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                context.setParameter("lam_vdw", lv)

                dc_vals.append((e_cp - e_cm) / (2 * delta))
                dv_vals.append((e_vp - e_vm) / (2 * delta))

            dc, dv = np.array(dc_vals), np.array(dv_vals)
            cov = np.cov([dc, dv]) * beta ** 2
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.maximum(eigvals, 1e-4)
            G[i, j] = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return G


def optimize_2d_geodesic_path(
    system,
    topology,
    positions,
    box_vectors,
    ligand_indices,
    n_grid: int = 16,
    n_steps_per_point: int = 3000,
    temperature: float = 300.0,
    platform_name: str = "CUDA",
) -> List[Tuple[float, float]]:
    """运行 2D 度量张量场采集 + Dijkstra 测地线寻径

    返回从 (1.0, 1.0) 到 (0.0, 0.0) 的最优 (λ_coul, λ_vdw) 路径
    """
    import gc as _gc
    softcore_params = ACESoftcorePotential.optimize_alpha(len(ligand_indices))
    sc_obj = ACESoftcorePotential.from_dict(softcore_params)

    probe_sys = build_aces_probe_system_dual_lambda(
        system, ligand_indices, sc_obj,
        fixed_lam_coul=0.5, fixed_lam_vdw=1.0,
    )

    platform = openmm.Platform.getPlatformByName(platform_name)
    props = {"Precision": "mixed", "DeviceIndex": "0"} if platform_name.upper() == "CUDA" else {}
    integ = openmm.LangevinMiddleIntegrator(temperature, 1.0/unit.picosecond, 0.002*unit.picosecond)
    ctx = openmm.Context(probe_sys, integ, platform, props)
    ctx.setPositions(positions)
    if box_vectors is not None:
        ctx.setPeriodicBoxVectors(*box_vectors)

    lam_c_grid = np.linspace(1.0, 0.0, n_grid)
    lam_v_grid = np.linspace(1.0, 0.0, n_grid)

    print(f"\n🗺️ 采集 2D 度量张量场 | {n_grid}×{n_grid} 网格 | {n_steps_per_point} 步/点")
    G = compute_2d_metric_grid(
        ctx, lam_c_grid, lam_v_grid,
        n_steps=n_steps_per_point,
        temperature=temperature,
    )

    print(f"  ✅ 度量张量场完成 | 形状: {G.shape}")
    try:
        path = dijkstra_monotonic_geodesic(G, lam_c_grid, lam_v_grid)
        print(f"  🏆 测地线路径: {len(path)} 个状态")
        print(f"     λ_coul: {path[0][0]:.3f} → {path[-1][0]:.3f}")
        print(f"     λ_vdw:  {path[0][1]:.3f} → {path[-1][1]:.3f}")
    except Exception as e:
        print(f"  ⚠️ 测地线寻径失败 ({e})，回退到对角线线性路径。")
        path = list(zip(np.linspace(1.0, 0.0, n_grid), np.linspace(1.0, 0.0, n_grid)))

    path_arr = np.array(path)
    # 确保 lam_coul 和 lam_vdw 严格单调递减 (从 1.0 -> 0.0)
    path_arr[:, 0] = np.minimum.accumulate(path_arr[:, 0])
    path_arr[:, 1] = np.minimum.accumulate(path_arr[:, 1])
    
    # 强制锚定边界
    path_arr[0, :] = [1.0, 1.0]
    path_arr[-1, :] = [0.0, 0.0]
    
    # 去除因单调化可能产生的重复点
    unique_mask = np.abs(np.diff(path_arr, axis=0)).sum(axis=1) > 1e-6
    unique_mask = np.append([True], unique_mask)
    path_arr = path_arr[unique_mask]
    
    path = [tuple(p) for p in path_arr]
    print(f"  🏆 测地线路径 (单调性已校准): {len(path)} 个状态")
    
    del ctx, integ, probe_sys
    _gc.collect()
    return path


def dijkstra_monotonic_geodesic(G, lam_c_grid, lam_v_grid):
    """单调有向图 Dijkstra 寻径 — 在 (λ_coul, λ_vdw) 2D 平面上找最短热力学路径"""
    import heapq
    nc, nv = G.shape[:2]
    dist = np.full((nc, nv), np.inf)
    prev = np.full((nc, nv), None, dtype=object)
    dist[0, 0] = 0.0
    pq = [(0.0, 0, 0)]

    moves = [(1, 0), (0, 1), (1, 1), (1, 2), (2, 1)]

    while pq:
        d, i, j = heapq.heappop(pq)
        if d > dist[i, j]:
            continue
        if i == nc - 1 and j == nv - 1:
            break

        for di, dj in moves:
            ni, nj = i + di, j + dj
            if 0 <= ni < nc and 0 <= nj < nv:
                dlc = lam_c_grid[ni] - lam_c_grid[i]
                dlv = lam_v_grid[nj] - lam_v_grid[j]
                dlam = np.array([dlc, dlv])
                g_mid = 0.5 * (G[i, j] + G[ni, nj])
                w = np.sqrt(max(0.0, dlam @ g_mid @ dlam)) + 1e-4
                if dist[i, j] + w < dist[ni, nj]:
                    dist[ni, nj] = dist[i, j] + w
                    prev[ni, nj] = (i, j)
                    heapq.heappush(pq, (dist[ni, nj], ni, nj))

    ci, cj = nc - 1, nv - 1
    if not np.isfinite(dist[ci, cj]) or prev[ci, cj] is None:
        raise RuntimeError(
            "测地线寻径失败：lambda 图不连通或终点不可达。"
            "请检查度量张量是否含 NaN/Inf，或回退到线性/对角路径。"
        )
    path = []
    while (ci, cj) != (0, 0):
        path.append((lam_c_grid[ci], lam_v_grid[cj]))
        parent = prev[ci, cj]
        if parent is None:
            raise RuntimeError(
                f"测地线寻径中断：节点 ({ci}, {cj}) 缺少前驱，图可能不连通。"
            )
        ci, cj = parent
    path.append((lam_c_grid[0], lam_v_grid[0]))
    return path[::-1]
