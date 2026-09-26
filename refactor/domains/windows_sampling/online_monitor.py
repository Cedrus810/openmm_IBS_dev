

# ============================================================================
# 7. 在线收敛监控器 (生产级动态诊断)
# ============================================================================
class OnlineConvergenceMonitor:
    """
    ABFE 在线收敛监控器 (核心物理组件)
    职责：实时分析自由能轨迹的平稳性、统计误差和相空间重叠度。
    设计原则：
    - 增量热启动 (initial_f_k) 实现 O(1) 级别 MBAR 重算
    - 五维正交判据杜绝"假收敛"
    - 单位原生支持 (OpenMM unit -> kJ/mol)

    ⚠️ 严格契约：输入 u_kn_chunk 必须为 Total Reduced Potential
       即 u_k(x) = β[U_phys(x) + U_restraint(x)]
       若仅传入纯物理势能，Overlap 与 N_eff 指标将失效，可能导致假收敛。
    """

    def __init__(
        self,
        temperature: unit.Quantity,
        check_interval: int = 10,
        ma_window: int = 5,
        precision_thresholds: Optional[Dict] = None,
    ):

        self.kt = (unit.MOLAR_GAS_CONSTANT_R * temperature).value_in_unit(
            unit.kilojoules_per_mole
        )
        self.interval = check_interval
        self.ma_window = ma_window

        self.thr = {
            "drift": 0.5,
            "error": 0.8,
            "neff_ratio": 0.20,
            "overlap": 0.85,
            "min_neighbor_overlap": 0.03,
            "ma_std": 0.30,
        }
        if precision_thresholds:
            self.thr.update(precision_thresholds)

        self.f_k_prev = None
        self.history = []
        self.dg_deque = deque(maxlen=ma_window + 2)

    def add_diagnostic(self, u_kn_chunk: np.ndarray, step: int) -> Dict:
        """
        核心诊断逻辑：输入势能矩阵，输出多维收敛报告
        参数：
            u_kn_chunk: (K_states, N_frames) 约化势能矩阵 (已除以 kT)
            step: 当前模拟步数 (用于日志)
        返回：
            dict: {converged: bool, dg: float, error: float, ...}
        """
        K, N = u_kn_chunk.shape
        if N < 20:
            return {"converged": False, "msg": "waiting_for_data", "step": step}

        energy_mean = np.mean(u_kn_chunk) * self.kt
        energy_var = np.var(u_kn_chunk, axis=1)

        if energy_mean < -2500.0 and np.max(energy_var) < 5.0:
            print(
                f"  [WARN] [Monitor] 能量矩阵疑似遗漏限制力 (μ={energy_mean:.1f}, max(σ²)={np.max(energy_var):.1f})"
            )
            print(
                f"     → 请确保 ibs_engine 中 getState(groups={{group_id}}) 包含 Boresch 力"
            )
            return {
                "converged": False,
                "error": "suspected_missing_restraint_energy",
                "step": step,
            }

        try:
            K, N = u_kn_chunk.shape
            n_k_array = np.full(K, N, dtype=int)
            mbar = _build_mbar_compatible(
                u_kn_chunk,
                n_k_array,
                initial_f_k=self.f_k_prev,
                solver_protocol="hybr",
                solver_tolerance=1e-5,
                verbose=False,
            )

            res = _compute_free_energy_result_compatible(mbar, compute_uncertainty=True)
            df, ddf = _extract_free_energy_arrays(res, require_uncertainty=True)

            self.f_k_prev = df[0, :].copy()

            dg = (df[0, -1] - df[0, 0]) * self.kt
            err = ddf[0, -1] * self.kt
            # 🔑 之前这里直接调用 mbar.compute_effective_sample_number()/
            # mbar.compute_overlap()["matrix"]，既没有走项目里其它调用点
            # （ibs_engine.py）已经在用的 dict-or-ndarray 兼容判断，也没有对
            # 返回值做有限性校验。改走下面两个项目内的兼容层函数，跟其它
            # MBAR 结果消费点保持一致。
            neff = _compute_effective_sample_number_compatible(mbar)
            neff_ratio = float(np.min(neff) / N) if N > 0 else 0.0

            overlap_mat = _compute_overlap_matrix_compatible(mbar)
            overlap = float(np.max(np.diag(overlap_mat)))
            
            # ✅ MBAR 重叠度自动诊断与降级
            min_offdiag = np.min([
                overlap_mat[i, j] 
                for i in range(K) for j in range(K) 
                if abs(i-j) == 1
            ])
            if min_offdiag < 0.03:
                print(f"  [ERR] [Monitor] 相邻窗口最小重叠 {min_offdiag:.3f} < 0.03，MBAR 误差可能低估！")
                print(f"     → 建议：延长采样 20% 或在重叠最差区域附近插值窗口")

            self.dg_deque.append(dg)
            ma_std = (
                float(np.std(list(self.dg_deque)[-self.ma_window :]))
                if len(self.dg_deque) >= self.ma_window
                else 99.0
            )
            drift = (
                float(abs(self.dg_deque[-1] - self.dg_deque[0]))
                if len(self.dg_deque) > 1
                else 99.0
            )

            is_stable = drift < self.thr["drift"] and ma_std < self.thr["ma_std"]
            is_precise = err < self.thr["error"] and neff_ratio > self.thr["neff_ratio"]
            is_connected = min_offdiag >= self.thr.get("min_neighbor_overlap", 0.03)

            converged = is_stable and is_precise and is_connected

            report = {
                "step": step,
                "converged": converged,
                "dg": dg,
                "error": err,
                "neff_ratio": neff_ratio,
                "overlap_max_diag": overlap,
                "overlap_min_offdiag": float(min_offdiag),
                "drift": drift,
                "ma_std": ma_std,
                "details": {
                    "stable": is_stable,
                    "precise": is_precise,
                    "connected": is_connected,
                },
            }
            if min_offdiag < 0.03:
                report["warning"] = "low_overlap"
            self.history.append(report)
            return report

        except ImportError:
            return {"converged": False, "error": "pymbar_not_installed", "step": step}
        except Exception as e:
            return {
                "converged": False,
                "error": f"mbar_failed: {str(e)[:60]}",
                "step": step,
            }

    def export_convergence_data(self, path: str):
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, cls=NumpyEncoder)

    def plot_convergence(self, output_path: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.history:
            print(f"  [WARN] 无收敛历史数据，跳过绘图: {output_path}")
            return

        valid_data = [h for h in self.history if "dg" in h and "error" in h]
        if not valid_data:
            return

        steps = [h["step"] for h in valid_data]
        dgs = [h["dg"] for h in valid_data]
        errs = [h["error"] for h in valid_data]

        plt.figure(figsize=(10, 6), dpi=300)

        plt.errorbar(
            steps,
            dgs,
            yerr=errs,
            fmt="o-",
            color="#2E86AB",
            ecolor="#A23B72",
            capsize=3,
            label="ΔG ± error",
            linewidth=1.5,
        )

        if dgs:
            plt.axhline(
                y=dgs[-1],
                color="gray",
                linestyle="--",
                alpha=0.5,
                label=f"Final ΔG: {dgs[-1]:.2f} kJ/mol",
            )

        plt.xlabel("Simulation Step", fontsize=11)
        plt.ylabel("ΔG (kJ/mol)", fontsize=11)
        plt.title("ABFE Convergence Monitor", fontsize=13, pad=15)
        plt.legend(frameon=True, fancybox=True, shadow=False)
        plt.grid(True, alpha=0.3, linestyle=":")

        plt.tight_layout()

        try:
            plt.savefig(output_path, bbox_inches="tight")
            print(f"  [OK] 收敛曲线已保存: {output_path}")
        except Exception as e:
            print(f"  [WARN] 保存图片失败: {e}")
        finally:
            plt.close()
