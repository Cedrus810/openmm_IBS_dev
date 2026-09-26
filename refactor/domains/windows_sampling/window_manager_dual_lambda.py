from dataclasses import dataclass, field
from typing import Any


@dataclass
class WarmupState:
    """单窗预热累加器（CUTTING_PLAN.md §5 序1、§0.2）。

    51 条散落在 run_all_windows 相位2 的常量型初始化收拢于此：
    字段顺序 = 原 init 行序，默认值 = 原初值（verify_rename ③逐条对账）。
    可变对象：本类承载跨语句累加，不做 frozen；每窗重建一次
    （run_all_windows 窗口循环体内的 `warm = WarmupState()`）。
    """

    is_resumed_ibs: bool = False
    bias_warmup_diag: dict = field(default_factory=lambda: {'status': 'not_run'})
    resumed_frozen_f_k: Any = None
    f_k_warm_started: bool = False
    f_stability_threshold_kJ_mol: float = 0.05
    frozen_burn_in_steps: int = 5000
    steps_at_full_bias: int = 0
    bias_update_count: int = 0
    bias_converged: bool = False
    restored_from_production_checkpoint: bool = False
    best_effort_acceptance: bool = False
    best_effort_acceptance_reason: Any = None
    best_effort_reburnin_required: bool = False
    best_effort_reburnin_steps: int = 0
    truncated_validation_frames_ignored: int = 0
    safety_cap_best_effort_tmbar_converged: bool = False
    budget_fallback_used: bool = False
    consecutive_pass_count: int = 0
    validation_pass_count: int = 0
    last_lse_balance: Any = None
    lse_residual_history: list = field(default_factory=list)
    learning_to_validation_cycles: int = 0
    validation_feedback_update_count: int = 0
    validate_direct_retry_pending: bool = False
    ever_completed_a_validate_attempt: bool = False
    last_failure_reason: Any = None
    frozen_validation_retry_count: int = 0
    last_f_delta: float = float('nan')
    freeze_burn_in_done: int = 0
    best_effort_residual: float = float('inf')
    best_effort_f_k: Any = None
    best_effort_validation_stats: Any = None
    validation_batch_history: list = field(default_factory=list)
    validation_cumulative_history: list = field(default_factory=list)
    last_validation_batch_p: Any = None
    validation_sample_count: int = 0
    validation_batch_count: int = 0
    validation_steps_this_freeze: int = 0
    early_probe_triggered: bool = False
    early_probe_trigger_reason: Any = None
    frozen_mbar_batches: list = field(default_factory=list)
    validation_solve_at_batches: int = IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
    validation_insufficient_attempts: int = 0
    validation_g_history: List[float] = field(default_factory=list)
    updates_since_freeze: int = 0
    have_frozen_once: bool = False
    local_mbar_gate_history: list = field(default_factory=list)
    last_local_mbar_gate: Any = None
    validation_indeterminate_diag: Any = None
    provisional_production: bool = False
    warmup_timers: Dict[str, float] = field(default_factory=dict)
    # —— 刀1 B段第1步收编（2026-09-20）：3 个跨界标量，解锁相位3/4边界 ——
    frozen_f_k_snapshot: Any = None
    mode: str = "learning"
    ema_mean_p_values: Any = None
@dataclass(frozen=True, slots=True)
class EarlyStopConfig:
    """early-stop 参数簇（D 线归组第一刀，CUTTING_PLAN.md §2.3；2026-09-20）。

    run_all_windows 入口自形参构造，向 _produce_window/_finish_production
    穿线。只读性实测：三函数体内对 9 名零 Store/Del（执行器核验）。
    字段无默认 = 必填配置（构造点全 kwargs 提供）。
    """

    early_stop_check_interval_steps: Any
    early_stop_max_delta_g_drift_kJ_mol: Any
    early_stop_max_uncertainty_kJ_mol: Any
    early_stop_min_absolute_ess: Any
    early_stop_min_decorrelated_samples: Any
    early_stop_min_ess_ratio: Any
    early_stop_min_steps: Any
    early_stop_required_consecutive_passes: Any
    enable_early_stop: Any
@dataclass(slots=True)
class WindowContext:
    """窗口级对象簇（D 线归组第二刀，2026-09-20）。

    sampler/sim/win_sys/lc_win/lv_win/win_sys_xml：跨相位存活最久的六个
    窗口对象（A 段相位测量）。每窗在循环体顶 `wc = WindowContext()` 重建，
    字段在 phase-1 渐进绑定（六名首写均先于任何读，默认 None 不可观测）；
    清理协议（del sampler / sim = None / …）随改写转为字段操作，
    gc.collect() 时序不变。
    """

    lc_win: Any = None
    lv_win: Any = None
    sampler: Any = None
    sim: Any = None
    win_sys: Any = None
    win_sys_xml: Any = None

@dataclass
class ProductionState:
    """单窗生产段状态（刀1 B段第2步收编，2026-09-20）。

    26 个跨相位 6/7/8 边界的名字（路径/预算/early-stop 累加器），原地收编：
    初始化语句保留在 _prepare_production 段内（prod.<field> = ...），每窗经
    `prod = ProductionState()` 重建。字段默认 None 不可观测（26/26 首写均为
    体顶层无条件赋值，先写后读；执行器头注释有实测口径）。
    """

    actual_production_updates: Any = None
    early_stop_check_history: Any = None
    early_stop_check_interval_updates: Any = None
    early_stop_kt: Any = None
    early_stop_min_updates: Any = None
    early_stop_pass_count: Any = None
    early_stop_previous_local_dg: Any = None
    early_stop_stop_reason: Any = None
    early_stop_triggered: Any = None
    effective_n_steps_per_window: Any = None
    expected_production_manifest: Any = None
    force_check_interval: Any = None
    loop_timers: Any = None
    n_updates: Any = None
    prior_cumulative_production_steps: Any = None
    production_base_path: Any = None
    production_bias_path: Any = None
    production_ckpt_path: Any = None
    production_conv_path: Any = None
    production_energies_path: Any = None
    production_f_k_lock: Any = None
    production_history_backup_len: Any = None
    production_manifest_path: Any = None
    production_pos_backup: Any = None
    production_residual_basis_path: Any = None
    production_sampling_states_path: Any = None


class IBSWindowManagerDualLambda:
    """双λ IBS 采样管理器 (生产级)"""
    def __init__(
        self,
        system_template,
        topology,
        perturbed_atom_indices,
        lambdas_coul: List[float],
        lambdas_vdw: List[float],
        temperature,
        window_ranges: List[Tuple[int, int]],
        alchemical_params,
        potential_type: str = "softcore",
        dispersion_protocol: Optional[str] = None,
        # [B6-FIX] 这条腿的环境类型。溶剂腿构造时天然是 soluble（runabfe 刻意不给
        # 溶剂腿传 environment_type/membrane，B1 的接线契约测试钉着），所以纯水盒
        # 里那条**合法**的均匀体相尾项修正会被正确保留。
        environment_type: Optional[str] = None,
        restraint_params: Optional[Dict] = None,
        prefix: str = "abfe_dual",
        platform_name: str = "CUDA",
        output_dir: str = "./output",
        checkpoint_dir: str = "./checkpoints",
        pilot_lambdas: Optional[List[float]] = None,
        pilot_mean_dU_dlambda: Optional[List[float]] = None,
        # 🔑 [窗口预热状态机重构 Stage 0] pilot 网格早就在 _sample_scalar_metric
        # 里算出了 std_dU_dlambda_kJ_mol/n_derivative_samples，只是之前没被
        # 提取、没被传到这里——没有它们就没法判断某个 pilot TI 种子对某个
        # 窗口够不够精确（见 pilot_ti_seed_trust_diagnostics）。默认 None
        # 向后兼容：旧调用点/旧 preopt cache 不传这两个字段时，行为不变。
        pilot_std_dU_dlambda: Optional[List[float]] = None,
        pilot_n_dU_dlambda_samples: Optional[List[int]] = None,
        coion_identity: Optional[Dict[str, Any]] = None,
        seed_ledger: Optional[Exp019SeedLedger] = None,
        leg_name: Optional[str] = None,
        # EXP-030: return a fresh Force for every System/window. OpenMM Force
        # objects cannot be owned by multiple Systems. The coefficient factory
        # receives this window's local lambda vectors and returns one A_k/state.
        residual_basis_force_factory: Optional[Any] = None,
        residual_state_coefficients_factory: Optional[Any] = None,
        residual_energy_offset_kj_mol: float = 0.0,
        # Canonical JointStateScoreSpec hash, persisted as resume identity.
        sampling_score_sha256: Optional[str] = None,
        residual_plugin_identity: Optional[Dict[str, Any]] = None,
        residual_em_policy: str = "no_residual_twin",
        residual_feature_name: str = "Outer-Lambda Local Residual for IBS",
    ):
        self.system_template = system_template
        self.topology = topology
        self.ligand_indices = perturbed_atom_indices
        self.lambdas_coul = lambdas_coul
        self.lambdas_vdw = lambdas_vdw
        self.temperature = temperature
        self.ranges = window_ranges
        self.alchemical_params = alchemical_params
        self.potential_type = potential_type
        # [B6] 膜体系用非 legacy 色散路线时，炼金 ligand–environment 的均匀密度
        # LRC 必须关闭（memtodolist §1.3）。None → 与改动前逐位一致。
        self.dispersion_protocol = dispersion_protocol
        self.environment_type = environment_type
        self.boresch = restraint_params
        self.prefix = prefix
        self.platform_name = platform_name
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir
        # B5: the caller resolves/freeze-checks this once; manifests only
        # record the payload and never read the spec or select an ion.
        self.coion_identity = coion_identity
        self.seed_ledger = seed_ledger
        self.leg_name = str(leg_name) if leg_name is not None else None
        if self.seed_ledger is not None and self.leg_name != self.seed_ledger.leg:
            raise ValueError("IBS manager leg_name 与 EXP-019 seed ledger 不一致")
        residual_enabled = residual_basis_force_factory is not None
        if residual_enabled != (residual_state_coefficients_factory is not None):
            raise ValueError(
                "residual_basis_force_factory 与 residual_state_coefficients_factory "
                "必须同时提供或同时省略"
            )
        if residual_enabled and not callable(residual_basis_force_factory):
            raise TypeError("residual_basis_force_factory 必须可调用")
        if residual_enabled and not callable(residual_state_coefficients_factory):
            raise TypeError("residual_state_coefficients_factory 必须可调用")
        if not math.isfinite(float(residual_energy_offset_kj_mol)):
            raise ValueError("residual_energy_offset_kj_mol 必须有限")
        if not residual_enabled and float(residual_energy_offset_kj_mol) != 0.0:
            raise ValueError("未启用 residual 时 energy offset 必须为 0")
        if residual_enabled and sampling_score_sha256 is None:
            raise ValueError("residual-enabled manager 必须绑定 sampling_score_sha256")
        if sampling_score_sha256 is not None:
            score_hash = str(sampling_score_sha256).lower()
            if len(score_hash) != 64 or any(ch not in "0123456789abcdef" for ch in score_hash):
                raise ValueError("sampling_score_sha256 必须是 64 位十六进制 SHA-256")
            self.sampling_score_sha256 = score_hash
        else:
            self.sampling_score_sha256 = None
        self.residual_basis_force_factory = residual_basis_force_factory
        self.residual_state_coefficients_factory = residual_state_coefficients_factory
        self.residual_energy_offset_kj_mol = float(residual_energy_offset_kj_mol)
        self.residual_plugin_identity = (
            dict(residual_plugin_identity) if residual_plugin_identity is not None else None
        )
        self.residual_em_policy = str(residual_em_policy)
        self.residual_feature_name = str(residual_feature_name)
        if residual_enabled and self.residual_em_policy != "no_residual_twin":
            raise ValueError(
                "residual-enabled manager 必须使用 no_residual_twin EM 策略"
            )
        # [IBS_BIAS_PROTOCOL_VERSION warm-start] Stage 2's pilot probe already
        # measures the mean gradient (mean_dU_dlambda_kJ_mol), not just the
        # variance proxy used for lambda spacing -- kept here so
        # run_all_windows can TI-integrate it into a real f_k seed for a
        # window's first learning attempt instead of cold-starting at 0.0.
        # None for any manager that isn't given pilot data (e.g. decharging,
        # Shadow-Coulomb) -- estimate_f_k_from_pilot_ti() treats that as
        # "no seed available" and callers fall back to today's behavior.
        self.pilot_lambdas = pilot_lambdas
        self.pilot_mean_dU_dlambda = pilot_mean_dU_dlambda
        # 🔑 [窗口预热状态机重构 Stage 0] 见上面参数说明；同样 None 表示"这个
        # manager 没有精度数据"，消费方（未来的 pilot-first 信任度判断）必须
        # 按 pilot_ti_seed_trust_diagnostics() 的约定把它当"不可信"处理，
        # 不能当异常。
        self.pilot_std_dU_dlambda = pilot_std_dU_dlambda
        self.pilot_n_dU_dlambda_samples = pilot_n_dU_dlambda_samples
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        _temp_q = temperature if hasattr(temperature, 'value_in_unit') else temperature * unit.kelvin
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * _temp_q).value_in_unit(unit.kilojoule_per_mole)

    def _seed_for(
        self,
        phase: str,
        stage: str,
        window: Any,
        stream: str,
        attempt: int = 0,
    ) -> Optional[int]:
        if self.seed_ledger is None:
            return None
        return self.seed_ledger.derive(phase, stage, window, stream, attempt)

    def _build_window_system(self, lc_win, lv_win, resolved_box, positions):
        """构建单个窗口的 (System, IBSBiasForce)。

        子类覆盖这个方法即可接入不同的 CV 构造（例如 Shadow-Coulomb 去电荷），
        同时复用本类其余的窗口调度/最小化/Boresch 爬坡/渐进预热/生产采样/
        断点续传/TMBAR 落盘逻辑——这些逻辑只依赖 (win_sys, ibs_wrap) 这两个
        返回值和 lc_win 的长度，不关心 CV 具体代表哪种物理量。
        """
        # 🔑 [2026-09-10] serialize→deserialize 往返 → clone()。
        # 这是 2026-09-09 那批迁移漏掉的一半。clone() 走 SerializationNode，
        # 不生成中间字符串；原写法每个窗口都要在堆上开一份多兆字节的 str
        # （大体系上是几十 MB），紧接着就是 build_ibs_dual_system 里
        # 30k 原子 × getParticleParameters 的分配风暴。
        residual_kwargs = {}
        if self.residual_basis_force_factory is not None:
            coefficients = [
                float(value)
                for value in self.residual_state_coefficients_factory(wc.lc_win, wc.lv_win)
            ]
            if len(coefficients) != len(wc.lc_win):
                raise ValueError(
                    "residual coefficient factory 返回长度与当前窗口状态数不一致："
                    f"{len(coefficients)} != {len(wc.lc_win)}"
                )
            if not all(math.isfinite(value) for value in coefficients):
                raise ValueError("residual coefficient factory 返回了非有限 A_k")
            residual_force = self.residual_basis_force_factory()
            if residual_force is None:
                raise ValueError("residual_basis_force_factory 返回了 None")
            residual_kwargs = {
                "residual_basis_force": residual_force,
                "residual_state_coefficients": coefficients,
                "residual_energy_offset_kj_mol": self.residual_energy_offset_kj_mol,
            }
        return build_ibs_dual_system(
            ensure_owned_system(XmlSerializer.clone(self.system_template)),
            self.topology,
            self.ligand_indices,
            wc.lc_win,
            wc.lv_win,
            self.alchemical_params,
            self.potential_type,
            self.boresch,
            self.temperature,
            self.prefix,
            box_vectors=resolved_box,
            reference_positions=positions,
            dispersion_protocol=self.dispersion_protocol,
            environment_type=self.environment_type,
            **residual_kwargs,
        )

    def _enqueue_window_snapshot(self, window_idx: int, stage_type: str, sampler) -> None:
        """同步原子刷盘能量快照。"""
        e_arr = np.array(wc.sampler.energy_history, dtype=np.float64, copy=True) if wc.sampler.energy_history else np.zeros((0, 0), dtype=np.float64)
        e_save = e_arr.T if e_arr.size > 0 else np.zeros((0, 0), dtype=np.float64)
        bias = np.array(wc.sampler.bias_history, dtype=np.float64, copy=True) if wc.sampler.bias_history else np.zeros((0,), dtype=np.float64)
        base = np.array(wc.sampler.base_energy_history, dtype=np.float64, copy=True) if wc.sampler.base_energy_history else np.zeros((0,), dtype=np.float64)
        sampling_states = np.asarray(wc.sampler.sampling_state_energy_history, dtype=np.float64)
        # Durable residual ledger contract: frames x states.  Analysis returns
        # states x frames below in _load_validated_joint_score_ledgers, but the
        # on-disk producer/consumer contract stays row-wise and unambiguous.
        sampling_states_save = sampling_states if sampling_states.size else np.zeros((0, 0), dtype=np.float64)
        residual_basis = np.asarray(wc.sampler.residual_basis_history, dtype=np.float64)

        _atomic_save_npy(
            os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_energies.npy"),
            e_save,
        )
        if bias.size > 0:
            _atomic_save_npy(
                os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_bias.npy"),
                bias,
            )
        if base.size > 0:
            _atomic_save_npy(
                os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_base.npy"),
                base,
            )
        if sampling_states_save.size > 0:
            _atomic_save_npy(
                os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_sampling_states.npy"),
                sampling_states_save,
            )
        if residual_basis.size > 0:
            _atomic_save_npy(
                os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_residual_basis.npy"),
                residual_basis,
            )

    def _production_disaster_rollback(
        self,
        sim,
        pos_backup,
        sampler,
        production_history_backup_len: int,
        stage_type: str,
        window_idx: int,
        attempt: int,
        e_total_n: float,
        fmax,
        win_sys,
        debug_mode: bool,
        label_prefix: str = "",
        progress_note: str = "",
        diagnose_prefix: str = "",
    ) -> int:
        """灾难回滚共享实现（生产主循环 + 余数补齐块此前各自内联一份几乎
        相同的代码，见 run_all_windows）。回退坐标 → 用同一套种子规则重设
        速度 → 局部最小化释放应力 → 步长减半 → 同步截断生产 history 的顺序
        和参数在两处必须逐字节一致，抽成一个方法只把两处真正不同的部分
        （attempt 编号、print 文案前缀、诊断文件名前缀）作为参数传入，不改
        choreography 本身。返回同步丢弃的 history 帧数（供调用方打日志用）。
        两个生产异常入口都汇聚到同一个 ``_truncate_production_history(`` 截断点。
        """
        wc.sim.context.setPositions(pos_backup)
        recovery_velocity_seed = self._seed_for(
            "recovery", stage_type, window_idx, "velocity", attempt=attempt
        )
        if recovery_velocity_seed is None:
            wc.sim.context.setVelocitiesToTemperature(self.temperature)
        else:
            wc.sim.context.setVelocitiesToTemperature(
                self.temperature, recovery_velocity_seed
            )
        print(f"    {label_prefix}触发回退，执行局部最小化释放应力...")
        wc.sim.minimizeEnergy(maxIterations=2000, tolerance=1.0)
        current_dt_ps = wc.sim.integrator.getStepSize().value_in_unit(unit.picoseconds)
        new_dt_ps = max(0.0001, current_dt_ps * 0.5)
        wc.sim.integrator.setStepSize(new_dt_ps * unit.picoseconds)
        # 🔑 [P1-13] 坐标退回备份点了，备份之后写入的帧属于被放弃的分支，
        # 必须同步截断——否则它们会和重启后长出的新分支拼成一条"连续"轨迹
        # 交给 _decorrelate_by_worst_target_state 估自相关。
        dropped = _truncate_production_history(wc.sampler, production_history_backup_len)
        _start_production_segment(wc.sampler, "catastrophe_rollback_rebuild")
        fmax_report = fmax if fmax is not None else float("nan")
        print(
            f"    [WARN] {label_prefix}灾难检测触发: {progress_note}"
            f"E_total={e_total_n:.1f}, max|F|={fmax_report:.1f}. "
            f"已回退坐标并将步长降至 {new_dt_ps*1000.0:.1f} fs"
            + (f"；同步丢弃被放弃分支的 {dropped} 帧生产 history" if dropped else "")
        )
        if debug_mode:
            diagnose_force_groups_detailed(wc.sim.context, wc.win_sys, prefix=diagnose_prefix)
            diagnose_force_breakdown(wc.sim.context, wc.win_sys, prefix=diagnose_prefix)
        return dropped

    def run_all_windows(
        self,
        positions,
        box_vectors,
        n_steps_per_window: int,
        steps_per_update: int,
        stage_type: str = "coul",
        initial_velocities=None,
        resume: bool = False,
        enable_gradual_warmup: bool = True,
        warmup_steps: int = 500000,
        min_bias_updates: int = 12,
        max_bias_updates: int = 50,
        required_consecutive_bias_updates: int = 3,
        # 🔑 [2026-09-10] 逐窗口 f_k 热启动种子（{window_idx: f_k}）。优先于 pilot-TI
        # 种子。用途：拿**上一段生产帧**重解出的 f_k 作为下一段的起点
        # （见 `recalibrate_f_k_from_production`）—— warmup 的 loose gate 只用 200 帧
        # 判、只要求 <10 kJ/mol，留下的偏置可能让混合塌向少数态；生产帧多几个数量级。
        # 这里只是**种子**，不是"已冻结"：预热照常学习并过同一个 loose gate 才冻结，
        # 所以不绕过任何现有门，只是起点好得多。
        initial_f_k_by_window: Optional[Dict[int, Sequence[float]]] = None,
        # 🔑 [2026-09-11，实验开关，默认 False = 行为逐字不变] 用生产帧重标定出来的
        # f_k，进生产前不再**被** loose gate 拦住。
        # 理由（老板拍板，PLAN 9k）：证据强度是倒挂的 —— 新 f_k 由该窗口几千帧生产
        # 数据解出，却要用 5 批×40=200 帧 warmup 样本去"验证"它；实测 win2 就死在
        # 这一步（g=67 凑不出去相关样本，烧光 180k 步一次都没解出来，然后中止整个 run）。
        # 而坏 f_k 的段只会**少贡献信息、不会把答案带偏**（MBAR 会给它该有的权重），
        # 所以事前拦一道对正确性没有增益。
        # ⚠️ 门**照跑照记**，只是不再阻断/抛异常 —— "过一下看看"保留，"拦着"去掉。
        # ⚠️ 仍然要求真的做过 burn-in（steps_at_full_bias>0），绝不零平衡进生产。
        accept_recalibrated_f_k_without_gate: bool = False,
        # 🔑 [2026-09-10] 只跑这些窗口（None = 全跑，逐字保持原行为）。
        # 用途：f_k 重标定判定完就已经知道**哪几个窗口**的偏置真的偏了，没有理由
        # 把没被重播种的窗口也从零重采一遍 —— 那既浪费 GPU，又因为新段序列更短而
        # 让它在 g/ESS 诊断上显得更好（实测：段 1 的 w3 全长 1000 帧 g=202.9，
        # 截到 500 帧变 13.9/31.0）。未跑的窗口在分析时沿用上一段的产物。
        only_window_indices: Optional[Sequence[int]] = None,
        # 🔑 [窗口预热状态机重构 Stage 1a] 改名 max_bias_warmup_steps →
        # max_bias_learning_steps：这个参数从来控制的都是"learning 阶段的步数
        # 预算"（sgd_step_budget/full_bias_step_budget 都从它派生），跟真正的
        # "爬坡"（dt/bias ramp）完全是两回事，旧名字容易让人以为它管的是
        # 预热爬坡步数。语义/默认值不变，纯改名。外部配置面（abfe_config.json
        # 的 "max_bias_warmup_steps" 键、exp029_protocol.py 的协议 schema、
        # abfe_pipeline.py 里 kwargs.get("max_bias_warmup_steps", ...) 的查找
        # 键名）不跟着改——那些是已经落盘/已经在用的外部契约，只有这个函数
        # 自己的 Python 形参改名。
        max_bias_learning_steps: int = 500000,
        mbar_calibration_reserved_steps: int = 50000,
        frozen_validation_step_overrides: Optional[Dict[int, int]] = None,
        frozen_validation_is_final_rung: Optional[Dict[int, bool]] = None,
        provisional_production_windows: Optional[Sequence[int]] = None,
        production_step_overrides: Optional[Dict[int, int]] = None,
        enable_early_stop: bool = False,
        early_stop_min_steps: int = 100000,
        early_stop_check_interval_steps: int = 20000,
        early_stop_required_consecutive_passes: int = 3,
        early_stop_min_ess_ratio: float = 0.05,
        early_stop_min_absolute_ess: float = 50.0,
        early_stop_min_decorrelated_samples: int = 20,
        early_stop_max_delta_g_drift_kJ_mol: float = 0.5,
        early_stop_max_uncertainty_kJ_mol: float = 1.0,
        debug_mode: bool = False,
        repair_policy: str = "non_mutating_v1",
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] 0.25 -> 0.5：见 version 25
        # changelog (a) 段——这个容差只影响重加权效率/方差，不影响
        # solve_stage_integrated（单参考重要性 MBAR）的无偏性；真正独立的
        # 正确性门槛是生产后的 _assert_stage_result_sane（min_overlap/
        # 绝对 ESS/去相关样本数/端点不确定度），完全不读这个值。放宽后仍然
        # 会被那道独立门槛兜底，不会让真正采样不足的 f_k 悄悄蒙混过关。
        lse_log_residual_tolerance: float = 0.5,
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] "见好就收"兜底：完整冻结验证
        # attempt 连续失败达到这个次数后，不再无限重试——接受目前观察到的
        # 最小 max_abs_log_residual 那次 attempt 的 f_k，标记为 best-effort
        # 而非严格达标，直接放行进生产。同一条理由：真正的正确性门槛在生产
        # 后独立检查，这里只影响"要不要再多花一次 attempt 的预算去碰运气"。
        # 预热只负责给生产确定一组固定 f_k，不负责用预热样本替代生产统计。
        # 完成一次足额 fixed-f 验证后，无论占据平坦残差是否达到理想阈值，均
        # 锁定该 attempt 的 f_k 并进入独立生产；残差只作为效率诊断。真正的
        # 可用性由生产后的 overlap/ESS/去相关样本/不确定度硬门判断。
        max_frozen_validation_cycles_before_accept_best: int = 1,
        # 仅用于把预热残差标为 within/outside sane bound，便于日志和诊断解释
        # f_k 的预计效率；生产运行不再拿它作准入硬门。warmup_only 设计审计
        # 没有后续生产统计兜底，因此仍要求落入该诊断范围才可返回成功。
        best_effort_max_residual_multiple: float = 4.0,
        # [IBS_BIAS_PROTOCOL_VERSION=19] learning 候选门读取的 TMBAR
        # solve_stage_integrated 阈值。故意比 final_*（stage 级最终验收，
        # min_absolute_ess=50/min_decorrelated_samples=20/max_uncertainty=1.0
        # kJ/mol）宽松得多：候选门只是"值不值得冻结去走 frozen validation"，
        # 真正严格的接受/拒绝判据仍是下面独立、未变的冻结验证阶段——候选门
        # 放宽不会让坏 f_k 蒙混过关，只会让它更快地被送去验证，验证失败后
        # 样本并入 tmbar_history 继续学习，不会像直接复用 final_* 那样让
        # consecutive_pass_count 在 max_bias_updates 预算内几乎不可能达到。
        candidate_min_ess_ratio: float = 0.05,
        candidate_min_absolute_ess: float = 1.0,
        # [v23 numerical follow-up #3] Was 5 -- mathematically unreachable
        # given `_solve_tmbar_and_recenter`'s own `min_frames_per_window=3`
        # (documented there as the actual floor for running the augmented
        # MBAR matrix at all, precisely because decorrelation commonly
        # leaves only 2-7 frames per ~20-frame minibatch). Any minibatch with
        # 3 or 4 decorrelated frames is deliberately let into local_results
        # by that floor, which then caps the aggregate min_decorrelated_
        # samples at 3 or 4 for as long as such a minibatch exists in
        # tmbar_history -- requiring >=5 here made this candidate criterion
        # permanently unsatisfiable in exactly the regime it's meant to
        # evaluate, same failure shape as the min_absolute_ess 2.5->1.0 fix
        # above. Lowered to 3 to match the actual floor; final_*/frozen-
        # validation gates are untouched.
        candidate_min_decorrelated_samples: int = 3,
        candidate_max_uncertainty_kJ_mol: float = 5.0,
        warmup_only: bool = False,
        # [Candidate-first, Validate-or-Learn v1] Best-effort acceptance has
        # no place in a production protocol: an unvalidated f_k may not enter
        # production merely because the calibration budget was exhausted.
        # Default flipped True->False: EXP-030 already passed False
        # explicitly; this flip makes that the default for every production
        # caller too. Only smoke/debug entrypoints should opt back in
        # explicitly. The fallback code path itself is left in place.
        allow_best_effort_warmup: bool = False,
    ):
        """
        执行全部窗口采样，能量保存至 {output_dir}/dual_window_{window_idx}_{stage_type}_energies.npy

        repair_policy: 采样修复策略。默认 "non_mutating_v1" —— IBS 非变异策略：
            某个 vdw 窗口 SGD 预热结束仍未获得任何足额 fixed-f attempt 时，绝不启动 fixed-H
            overlap 探针 / asymmetric-ΔF-slope 判据 / bias 校准探针，也绝不用累计
            ΔF 覆盖 context 里的 f_k；直接落盘 warmup_failure 并抛
            IBSWarmupConvergenceError（f_not_converged），交人工/救援审计处理。
            任何其它值恢复旧的 fixed-H 探针 + 就地重校准 f_k 行为（已弃用，仅为
            对照/回滚保留）。fixed-H adjacent overlap 因此被降为纯诊断，绝不再自动
            触发插点/重校准。见 plan a-relative-binding-free-energy-framewor-immutable-starlight.md。

        warmup_only: 设计期 IBS 窗口验证模式。严格执行与生产相同的 Log-Sum-Exp
            权重学习、冻结 burn-in 和只读自洽验证；通过后立即返回该窗口诊断，
            不进入 production、不写能量数组。失败仍按 non_mutating_v1 原样抛出，
            由调用方在独立 design scratch ensemble 中决定拆窗或插入热力学中点，
            绝不修改正式生产 ensemble。

        诊断功能 (debug_mode=True，生产环境默认关闭)：
            - 构建后打印力组架构
            - 关键阶段调用 diagnose_force_breakdown (非侵入式力分解)
            - 若检测到 NaN 或力爆炸，立即抛出异常并终止

        provisional_production_windows: 可选的窗口下标集合。**调用方显式授权**：
            这些窗口的冻结验证若撞到 IBS_LOCAL_MBAR_GATE_MAX_BATCHES 批上限、
            Δf−ΔF 从未被求出（`validation_budget_exhausted_indeterminate`，
            对这份 f_k **无结论**、既非通过也非被驳回），就用这份冻结 f_k 进
            **一块**诊断生产，而不是抛 IBSValidationBudgetIndeterminateError。
            ⚠️ 这**不是**把门放宽：
              · `f_k_evidence_status` 保持 `indeterminate`，**绝不**升级成
                `verified`；`bias_status` 写 `provisional_production`，与
                `converged` 在词汇上分开，下游不会把它读成可信 PASS；
              · `*_warmup_failure.json` 作为证据**保留**（收敛路径会删掉它）；
              · 真正的验收仍然是生产后的 min N_eff/g 支撑门 + 最终 MBAR +
                `_assert_stage_result_sane` —— 被放宽的只是**生产前**那道
                f_k 预筛，而且是一次性的、记录在案的。
            为什么需要它：15 批是**工程预算**，撞上限只说明"这一轮没测出来"。
            在没有这条路径时，这样的窗口既进不了生产（引擎抛路由信号）、
            又拿不到新证据（补帧动作在构造上推不动它），整条流水线只能停在
            NO_FEASIBLE_ACTION —— 真机 cyclod_ligand2/rep2 win4 就是这样卡死的。
        frozen_validation_is_final_rung: 可选的 {window_idx: 这次是否已经是调用方
            冻结验证阶梯的最后一档} 表。只在 calibration_pending 场景下参与判断：
            若 True 且这次仍未通过独立验证，直接判定为终态失败
            （bias_status="calibrated_validation_failed"，raise
            IBSFrozenCalibrationValidationError(terminal=True)），不再落盘为
            "calibrated_pending_validation"、不再暗示"resume 会继续自动延长预算"。
            调用方（abfe_pipeline.py::_run_stage_with_overlap_autorepair）应在把
            某个窗口的 frozen_validation_step_overrides 升级到阶梯最后一档时，
            同步把这个窗口标记为 True。

        production_step_overrides: 可选的 {window_idx: 该窗口实际生产步数} 覆盖表，
            未列出的窗口仍用 n_steps_per_window。供 abfe_pipeline.py 的
            production ESS sampling-repair 分支使用：fixed-H 全通过但 production
            ESS 低、且判定偏置本身没问题（只是采样太短/构象弛豫慢）时，不能只是
            删掉旧样本再用同样步数重采一次（那只是换一批同样长度的独立样本，
            不会真正增加信息量）；调用方应在这里传入更大的步数，真正延长该窗口
            的采样，而不是原地打转。window_idx 必须对应本次调用 self.ranges 里
            该窗口在这次调用中的实际位置——调用方需要自己保证窗口顺序在设置
            override 和真正消费它的这次调用之间没有变化（例如夹在两次
            split/insert 之间的“纯 sampling-repair 轮”）。

        mbar_calibration_reserved_steps: 从 max_bias_learning_steps 总预算里为
            fixed-H overlap 全通过后的 MBAR 校准 f_k 单独预留的步数（burn-in +
            冻结验证）。SGD learning/freeze_burn_in/validating 阶段只能用
            max_bias_learning_steps - mbar_calibration_reserved_steps；这样即使
            SGD 把自己的份额烧光，MBAR 校准仍有独立预算跑完整验证，而不会因为
            共享的 steps_at_full_bias 已耗尽而必然判失败。

        frozen_validation_step_overrides: 可选的 {window_idx: 该窗口这次冻结
            验证的目标预算步数} 覆盖表 [IBS_BIAS_PROTOCOL_VERSION=12]。fixed-H
            overlap 全通过 + MBAR 校准出的 f_k 冻结后，若冻结验证在
            mbar_calibration_reserved_steps（或上一轮的覆盖值）内仍未通过
            覆盖门，本方法不会像旧协议那样直接判失败并把 f_k 退回当作普通
            未收敛热启动——而是把 bias_status 存为
            "calibrated_pending_validation"、frozen_f_k_pending 存下这份已经
            校准好的 f_k，并在诊断里标记 calibration_pending_validation=True。
            调用方（见 abfe_pipeline.py::_run_stage_with_overlap_autorepair）
            据此按 50k→150k→300k 的阶梯延长这里的预算重试，同一个窗口 resume
            时会跳过 [learning] 和 fixed-H 探针/MBAR 校准，直接带着冻结的
            f_k 从 freeze_burn_in 开始，只消耗新增的那部分步数（不是重新烧
            完整个新预算）。未列出的窗口沿用 mbar_calibration_reserved_steps。

        enable_early_stop 及 early_stop_*：默认关闭（enable_early_stop=False），
            此前这个开关在调用链上是纯空转——abfe_pipeline.py 从未把它传进这个
            方法，这里也从未定义对应参数，打开它不会有任何效果。现在是真正接入
            的在线判据，但阈值默认值尚未用已有完整轨迹做离线回放校准过，见
            EARLY_STOP_PROTOCOL_VERSION 定义处；在离线回放验证通过、确定合适的
            默认阈值之前，不要在生产配置里打开这个开关。

            达到 early_stop_min_steps 步之后才开始检查；此后每满
            early_stop_check_interval_steps 步，用该窗口至今累积的样本
            （sampler.energy_history/bias_history/base_energy_history）跑一次
            local MBAR（_solve_single_window_local_mbar），同时核对五项：
            (1) 绝对有效样本数 >= early_stop_min_absolute_ess；
            (2) 最差 ESS 比例 >= early_stop_min_ess_ratio；
            (3) 去相关后样本数 >= early_stop_min_decorrelated_samples；
            (4) 本窗口局部 ΔG（f[-1]-f[0]）相对上一次检查的漂移
                <= early_stop_max_delta_g_drift_kJ_mol（第一次检查没有"上一次"
                可比，直接判不通过，因此至少要连续通过
                required_consecutive_passes+1 次检查才可能真正停止）；
            (5) 局部 ΔG 端点合并不确定度 <= early_stop_max_uncertainty_kJ_mol。
            五项全部通过才计入一次"连续通过"，任一项失败清零连续计数；连续
            达到 early_stop_required_consecutive_passes 次才真正停止该窗口的
            生产采样。n_steps_per_window（含 production_step_overrides 覆盖值）
            始终是硬上限，early stop 只会提前结束，不会超过它。

            触发后 convergence.json 会记录 actual_production_steps（真正跑了
            多少步）、early_stop_triggered、stop_reason、
            early_stop_check_history（每次检查的 ESS/ΔG/不确定度和是否通过）、
            early_stop_protocol_version 与本次调用的阈值配置——resume 时如果
            缓存显示 early_stop_triggered=True，只有当前调用同样启用 early
            stop、协议版本一致、且当前目标步数没有被调高，才会复用这份提前
            停止的短样本；任何一项不满足都视为缓存不可信，强制重新采样，避免
            "以后关闭 early stop 或调高预算"时被一次短样本悄悄糊弄过去。
        """
        es = EarlyStopConfig(
            early_stop_check_interval_steps=early_stop_check_interval_steps,
            early_stop_max_delta_g_drift_kJ_mol=early_stop_max_delta_g_drift_kJ_mol,
            early_stop_max_uncertainty_kJ_mol=early_stop_max_uncertainty_kJ_mol,
            early_stop_min_absolute_ess=early_stop_min_absolute_ess,
            early_stop_min_decorrelated_samples=early_stop_min_decorrelated_samples,
            early_stop_min_ess_ratio=early_stop_min_ess_ratio,
            early_stop_min_steps=early_stop_min_steps,
            early_stop_required_consecutive_passes=early_stop_required_consecutive_passes,
            enable_early_stop=enable_early_stop,
        )
        # 🔑 [non_mutating_v1] Validate the sampling repair policy ONCE at entry,
        # fail-closed on any unrecognized value (a typo must never silently
        # re-open the deprecated mutating fixed-H/recalibration path).
        legacy_repair = should_run_legacy_repair(repair_policy)
        if not np.isfinite(lse_log_residual_tolerance) or float(lse_log_residual_tolerance) <= 0.0:
            raise ValueError("lse_log_residual_tolerance 必须是有限正数")
        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)
        resolved_box = _resolve_periodic_box_vectors(box_vectors, topology=self.topology, system=self.system_template)
        if _system_requires_periodic_box(self.system_template) and resolved_box is None:
            raise ValueError(
                "当前系统使用了周期性非键方法，但没有可用的周期性盒子。"
                "请检查输入坐标/拓扑是否带 box，或显式传入 box_vectors。"
            )

        warmup_results = []
        _only = (
            None if only_window_indices is None
            else {int(x) for x in only_window_indices}
        )
        if _only is not None:
            _unknown = _only - set(range(len(self.ranges)))
            if _unknown:
                raise ValueError(
                    f"only_window_indices 含不存在的窗口 {sorted(_unknown)}；"
                    f"本阶段只有 {len(self.ranges)} 个窗口。拒绝静默忽略。"
                )
            print(
                f"  [窗口子集] 本次只跑窗口 {sorted(_only)}（共 {len(self.ranges)} 个）；"
                "其余窗口不重采，分析时沿用上一段产物。"
            )
        for window_idx, (start, end) in enumerate(self.ranges):
            wc = WindowContext()
            if _only is not None and window_idx not in _only:
                continue
            wc.lc_win = self.lambdas_coul[start:end]
            wc.lv_win = self.lambdas_vdw[start:end]
            print(f"\n{'='*80}")
            print(f"[窗口 {window_idx}] 索引 [{start}:{end}] | 状态数: {len(wc.lc_win)}")
            print(f"{'='*80}")

            # [2026-09-14 / 阶梯空转] 窗口级"只差步数"标记：身份全对、仅补帧目标
            # 涨了 ⟹ 可以走增量续采，不必把整条前置链（EM + dt 测试 + Boresch 爬坡
            # + 冻结 burn-in + 只读复验）重建一遍，反正紧接着的 production
            # checkpoint 恢复会把 Context 覆盖回上次生产结束那一刻。
            # 无缓存 / 缓存不可读 ⟹ 保持 False ⟹ 走今天的完整路径。
            topup_only = False

            # ---------- 断点续传：按窗口跳过已完成的采样 ----------
            # 此前 resume 只热启动了 IBS bias 权重 (f_k)，中途崩溃后重跑会把
            # 已经跑完的窗口从头再来一遍，白白浪费 GPU 时间且违背"Checkpoint"的
            # 设计承诺。这里在真正重建系统/最小化/生产采样之前，先检查该窗口
            # 是否已有形状匹配的有效能量文件，若有则直接跳过整窗。
            if resume:
                energies_path = os.path.join(
                    self.output_dir, f"dual_window_{window_idx}_{stage_type}_energies.npy"
                )
                bias_path = os.path.join(
                    self.output_dir, f"dual_window_{window_idx}_{stage_type}_bias.npy"
                )
                base_path = os.path.join(
                    self.output_dir, f"dual_window_{window_idx}_{stage_type}_base.npy"
                )
                convergence_path = os.path.join(
                    self.output_dir, f"dual_window_{window_idx}_{stage_type}_convergence.json"
                )
                if os.path.exists(energies_path) and os.path.exists(convergence_path):
                    try:
                        with open(convergence_path, "r", encoding="utf-8") as f:
                            cached_conv = json.load(f)
                        cached_policy_early = cached_conv.get(
                            "sampling_repair_policy"
                        )
                        # 🔑🔑 [2026-09-15 真机 cyclod_ligand1/rep1] **「没有这个键」
                        # 不是「另一套策略的数据」。**
                        #
                        # `ExistingEnsembleRequiresRescueAudit` 的语义是「盘上躺着
                        # **旧变异策略**采的 ensemble，不许就地覆盖，留给审计」。
                        # 而键**整个缺失**只说明这份 convergence.json 不是一次完整的
                        # 窗口收尾写出来的：
                        #   · 生产中途的周期快照（见 `_periodic_conv`：文件不存在时
                        #     从 `{}` 起写，只有 8 个键、一个身份键都没有）——真机就是
                        #     作业被杀在 win0 生产 200k/250k 步，重启即崩，且**永远**
                        #     进不去这个窗口；
                        #   · 或者没有这个字段的老格式产物。
                        # 两者都该走「无效缓存 ⟹ 重采」那条路（跟其余 9 门一样打 WARN），
                        # 不该把整条流水线打死。
                        # ⚠️ 键**存在但不同**仍然一律 fail-closed —— 那才是真的旧策略数据。
                        if (
                            not legacy_repair
                            and cached_policy_early is not None
                            and cached_policy_early != repair_policy
                        ):
                            raise ExistingEnsembleRequiresRescueAudit(
                                f"窗口 {window_idx}: 磁盘能量缓存的 sampling_repair_policy="
                                f"{cached_policy_early!r}（当前 {repair_policy!r}）；"
                                "拒绝在原 ensemble 目录覆盖旧策略数据。",
                                diagnostics={
                                    "window_index": int(window_idx),
                                    "stage_type": stage_type,
                                    "cached_sampling_repair_policy": cached_policy_early,
                                    "current_sampling_repair_policy": repair_policy,
                                    "requires_rescue_audit": True,
                                },
                            )
                        cached_e, _cached_bias, _cached_base = (
                            _load_validated_window_data_triplet(
                                energies_path,
                                bias_path,
                                base_path,
                                cached_conv,
                            )
                        )
                        # 🔑 只按 window_idx + shape 判断"这份缓存还能不能用"曾经是不安全的：
                        # λ 路径被自动加密/重新划分窗口后，同一个 window_idx 完全可能对应
                        # 一段全新的 λ 区间，只要凑巧态数相同（shape 相同）就会被静默当成
                        # 有效缓存复用——采样对了，但对的是错的 λ。这里额外校验 convergence.json
                        # 里存的实际 λ 值（见下方保存逻辑）跟本次真正要跑的 lc_win/lv_win 是否
                        # 完全一致；旧格式缓存没有这个字段的，保守地当作不匹配，强制重采，而不是
                        # 假设它"大概率没变"。
                        _gate = _resume_cached_window_gate_status(
                            cached_conv,
                            tuple(cached_e.shape),
                            wc.lc_win,
                            wc.lv_win,
                            repair_policy,
                            lse_log_residual_tolerance,
                            es.enable_early_stop,
                            {
                                "min_steps": int(es.early_stop_min_steps),
                                "check_interval_steps": int(es.early_stop_check_interval_steps),
                                "required_consecutive_passes": int(es.early_stop_required_consecutive_passes),
                                "min_ess_ratio": float(es.early_stop_min_ess_ratio),
                                "min_absolute_ess": float(es.early_stop_min_absolute_ess),
                                "min_decorrelated_samples": int(es.early_stop_min_decorrelated_samples),
                                "max_delta_g_drift_kJ_mol": float(es.early_stop_max_delta_g_drift_kJ_mol),
                                "max_uncertainty_kJ_mol": float(es.early_stop_max_uncertainty_kJ_mol),
                            },
                            (
                                int(production_step_overrides[window_idx])
                                if production_step_overrides and window_idx in production_step_overrides
                                else int(n_steps_per_window)
                            ),
                            current_coion_identity=self.coion_identity,
                            stage_type=stage_type,
                            current_sampling_score_sha256=self.sampling_score_sha256,
                            current_stage_protocol_key=getattr(self, "stage_protocol_key", None),
                        )
                        # 🔑 这 10 个门原先内联在这里（~110 行），现已抽成模块级纯函数
                        # _resume_cached_window_gate_status，逐门语义与阈值一字未改，
                        # 只是变得可以用 mock 的 convergence.json 单独测试（见
                        # test_resume_reuse_contracts.py）。下面那串逐门诊断打印仍然
                        # 读同名局部变量，因此完全保持原样。
                        topup_only = bool(_gate["incremental_topup_only"])
                        lambdas_match = _gate["lambdas_match"]
                        version_match = _gate["version_match"]
                        bias_protocol_match = _gate["bias_protocol_match"]
                        com_restraint_version_match = _gate["com_restraint_version_match"]
                        cached_lse_tolerance = _gate["cached_lse_tolerance"]
                        lse_tolerance_match = _gate["lse_tolerance_match"]
                        lrc_version_match = _gate["lrc_version_match"]
                        vdw_nb_version_match = _gate["vdw_nb_version_match"]
                        repair_policy_match = _gate["repair_policy_match"]
                        coion_identity_match = _gate["coion_identity_match"]
                        early_stop_ok = _gate["early_stop_ok"]
                        early_stop_reject_reason = _gate["early_stop_reject_reason"]
                        if _gate["usable"]:
                            print(
                                f"  窗口 {window_idx} 已有有效缓存能量 {cached_e.shape} 且 λ 值匹配，"
                                f"resume 模式下跳过重新采样。"
                            )
                            continue
                        if cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0 and not version_match:
                            print(
                                f"  [WARN] 窗口 {window_idx} 缓存的 wca_accounting_version="
                                f"{cached_conv.get('wca_accounting_version')!r}（期望 {WCA_ACCOUNTING_VERSION}），"
                                "base/bias 力组口径已变更，视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0 and not bias_protocol_match:
                            print(
                                f"  [WARN] 窗口 {window_idx} 缓存的 ibs_bias_protocol_version="
                                f"{cached_conv.get('ibs_bias_protocol_version')!r}（兼容版本 "
                                f"{sorted(IBS_BIAS_CACHE_COMPATIBLE_PROTOCOL_VERSIONS)}），"
                                "IBS 偏置预热/冻结协议已变更，视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0 and not com_restraint_version_match:
                            print(
                                f"  [WARN] 窗口 {window_idx} 缓存的 ligand_com_restraint_protocol_version="
                                f"{cached_conv.get('ligand_com_restraint_protocol_version')!r}"
                                f"（期望 {LIGAND_COM_RESTRAINT_PROTOCOL_VERSION}）：v1 的 Group 5 "
                                "COM 限制力在 CUDA 上产生永久激活、跨边界跳变的定向拖拽，该窗口"
                                "轨迹不满足 MBAR 的平衡采样前提，视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0 and not lse_tolerance_match:
                            print(
                                f"  [WARN] 窗口 {window_idx} 缓存的 LSE log 残差容差="
                                f"{cached_lse_tolerance!r}（当前 {lse_log_residual_tolerance}），"
                                "自洽收敛门已改变，视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0 and not lrc_version_match:
                            print(
                                f"  [WARN] 窗口 {window_idx} 缓存的 lj_tail_lrc_protocol_version="
                                f"{cached_conv.get('lj_tail_lrc_protocol_version')!r}（期望 "
                                f"{TRADITIONAL_LJ_LRC_PROTOCOL_VERSION}），LJ 长程尾项修正公式已变更"
                                "（switching-aware），视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0 and not vdw_nb_version_match:
                            print(
                                f"  [WARN] 窗口 {window_idx}（stage_type={stage_type!r}）缓存的 "
                                f"vdw_nonbonded_protocol_version="
                                f"{cached_conv.get('vdw_nonbonded_protocol_version')!r}（期望 "
                                f"{VDW_NONBONDED_PROTOCOL_VERSION}），vdW softcore cutoff/switching "
                                "协议已变更（MEM-00h：1.2nm+switch → 1.0nm 无switch），"
                                "视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0 and not repair_policy_match:
                            # 🔑 [non_mutating_v1] 磁盘上已有旧策略的能量数据。绝不能
                            # 就地重采覆盖它——那会毁掉 rescue 审计所需的原始 ensemble。
                            # 保留原文件，抛 ExistingEnsembleRequiresRescueAudit，交审计
                            # 判定是否可救 / 是否需重跑（可强制写入新 ensemble-ID 目录）。
                            # 🔑🔑 [2026-09-15] 与上面那道早门同一条修正：**键缺失 ≠
                            # 旧策略数据**。缺失只说明这份记录不完整（生产中途的周期
                            # 快照 / 老格式），走"无效缓存 ⟹ 重采"，不走 rescue audit。
                            # 这两处是同一判据的两份实现，改一处必须改另一处。
                            if (
                                not legacy_repair
                                and cached_conv.get("sampling_repair_policy") is not None
                            ):
                                raise ExistingEnsembleRequiresRescueAudit(
                                    f"窗口 {window_idx}: 磁盘能量缓存的 sampling_repair_policy="
                                    f"{cached_conv.get('sampling_repair_policy')!r}（当前 {repair_policy!r}）——"
                                    "产自旧的变异修复策略（f_k 可能被就地重校准过，属于不同参考系）。"
                                    "non_mutating_v1 拒绝复用，也拒绝就地重采覆盖（会毁掉 rescue 审计所需"
                                    "的原始数据）。已保留原文件，交 rescue 审计判定是否可救或需重跑。",
                                    diagnostics={
                                        "window_index": int(window_idx),
                                        "stage_type": stage_type,
                                        "cached_sampling_repair_policy": cached_conv.get("sampling_repair_policy"),
                                        "requires_rescue_audit": True,
                                    },
                                )
                            print(
                                f"  [WARN] 窗口 {window_idx} 缓存的 sampling_repair_policy="
                                f"{cached_conv.get('sampling_repair_policy')!r}（期望 {repair_policy!r}）——"
                                "该缓存产自旧的变异修复策略（legacy_mutating 下按无效缓存重采）。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0 and not coion_identity_match:
                            print(
                                f"  [WARN] 窗口 {window_idx} convergence.json 的 co-ion runtime identity="
                                f"{cached_conv.get('coion_identity')!r} 与当前"
                                f" {self.coion_identity!r} 不一致或缺失，拒绝复用能量缓存，"
                                "将重新采样该窗口。"
                            )
                        elif _gate["shape_ok"] and (not _gate["segment_metadata_match"] or not _gate["stage_protocol_match"]):
                            print(f"  [WARN] 窗口 {window_idx} 缓存不能复用: {_gate['reason']}；将重新采样该窗口。")
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0 and not _gate["step_budget_ok"]:
                            # [2026-09-14] 这一条原来和 early-stop 那条共用文案，把**纯预算
                            # 短缺**说成 "early stop 提前停止产出的短样本" —— enable_early_stop
                            # 默认 False，绝大多数现场根本没触发过 early stop，这句话把两个
                            # 会话的归因都带偏过。两种情况必须分开说。
                            print(
                                f"  [INFO] 窗口 {window_idx} 缓存身份"
                                + ("全部匹配" if _gate["incremental_topup_only"] else "有其它不匹配项")
                                + f"，但{_gate['step_budget_reject_reason']}"
                                + ("；走增量续采（沿用已有帧，只补差额步数）。"
                                   if _gate["incremental_topup_only"]
                                   else "，将重新采样该窗口。")
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0 and not early_stop_ok:
                            print(
                                f"  [WARN] 窗口 {window_idx} 缓存是 early stop 提前停止产出的短样本，"
                                f"但{early_stop_reject_reason}，视为无效缓存，将重新采样该窗口。"
                            )
                        elif cached_e.ndim == 2 and cached_e.shape[0] == len(wc.lc_win) and cached_e.shape[1] > 0:
                            print(
                                f"  [WARN] 窗口 {window_idx} 缓存能量形状匹配但 λ 值不匹配"
                                f"（缺少 λ 元数据或 λ 路径已变更），视为无效缓存，将重新采样该窗口。"
                            )
                        else:
                            print(
                                f"  [WARN] 窗口 {window_idx} 缓存能量形状 {cached_e.shape} 与期望 "
                                f"({len(wc.lc_win)}, N) 不符，将重新采样该窗口。"
                            )
                    except ExistingEnsembleRequiresRescueAudit:
                        raise
                    except Exception as e:
                        print(f"  [WARN] 窗口 {window_idx} 缓存能量加载失败 ({e})，将重新采样该窗口。")

            # ---------- 构建系统 ----------
            wc.win_sys, ibs_wrap = self._build_window_system(wc.lc_win, wc.lv_win, resolved_box, positions)
            if debug_mode:
                print_force_group_details(wc.win_sys, prefix=f"窗口{window_idx}_系统构建后")

            # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] ibs_state_file 提前到这里
            # 拼接（原来在下面"初始化采样器"处才拼），纯字符串操作、无副作用——
            # 需要在还没建 Simulation/走最小化/爬坡之前就先廉价 peek 一下这个窗口
            # 是不是"续验冻结校准 f_k"的场景，才能决定要不要跳过重建。
            ibs_state_file = os.path.join(
                self.checkpoint_dir,
                f"ibs_state_{stage_type}_window_{window_idx}.json",
            )
            main_ckpt_path, main_manifest_path = _main_window_checkpoint_paths(
                self.checkpoint_dir, stage_type, window_idx
            )
            # 🔑 [2026-09-10] 整个窗口只序列化 win_sys **一次**。
            #
            # 下面生产 checkpoint manifest 那处（原来是
            # `win_sys_xml_for_prod_ckpt = openmm.XmlSerializer.serialize(win_sys)`）
            # 序列化的是同一个对象，而两处之间 `win_sys` 只被只读方法碰过
            # （getNumConstraints / getForce / getNumParticles），没有任何
            # addForce/removeForce/set*，所以两次的字节逐字相同。
            #
            # 重复那一次不是"多花点时间"：被序列化的是装配好的窗口 System
            # （Group-1 的 CustomCVForce 里装着 K 个 softcore CustomNonbondedForce，
            # 每个都带完整 per-particle 参数和整张排除表），大体系上是几十 MB 的
            # 瞬时峰值，而且它发生在整段 CUDA 预热之后。2026-09-10 的 SEGV 就落在
            # 第二次这一调用里（ibs_engine.py:16540），第一次同样的调用是过的。
            wc.win_sys_xml = openmm.XmlSerializer.serialize(wc.win_sys)
            expected_main_manifest = _build_main_window_checkpoint_manifest(
                stage_type,
                window_idx,
                len(wc.lc_win),
                wc.win_sys_xml,
                wc.lc_win,
                wc.lv_win,
                (float(np.mean(wc.lv_win)) if _system_has_global_parameter(wc.win_sys, "lambda_shield") else None),
                # 🔑 [temperature_K 类型修复] self.temperature 是带 kelvin 单位的
                # openmm Quantity（LangevinMiddleIntegrator 等物理调用都要求这个
                # 类型），但 _build_main_window_checkpoint_manifest 的 temperature_K
                # 形参约定是纯 float——直接传 Quantity 进去，里面的 float(temperature_K)
                # 会报 TypeError: Quantity.__float__ returned non-float。跟文件里其它
                # manifest 构造调用（如 _build_fixed_h_probe_bank_manifest 用
                # temperature_q.value_in_unit(unit.kelvin)）保持同样的显式转换。
                self.temperature.value_in_unit(unit.kelvin),
                self.platform_name,
                repair_policy=repair_policy,
                coion_identity=self.coion_identity,
            )
            attempt_checkpoint_restore = bool(
                resume
                and _peek_ibs_bias_status(ibs_state_file) in (
                    "calibrated_pending_validation",
                    # 🔑 [2026-09-11] 无法判定态也要恢复 Context：否则会先走一次
                    # freeze_burn_in，把上次累计的验证批次冲掉、等于重新领额度。
                    "frozen_validation_indeterminate",
                )
                and _main_window_checkpoint_is_usable(
                    self.checkpoint_dir, stage_type, window_idx, expected_main_manifest
                )
            )

            integrator = LangevinMiddleIntegrator(self.temperature, 2.0 / unit.picosecond, 0.002 * unit.picosecond)
            window_integrator_seed = self._seed_for(
                "sampling", stage_type, window_idx, "integrator"
            )
            if window_integrator_seed is not None:
                integrator.setRandomNumberSeed(window_integrator_seed)
            integrator.setConstraintTolerance(1e-3)
            if hasattr(integrator, 'setRemoveCMMotion'):
                integrator.setRemoveCMMotion(True)

            wc.sim = app.Simulation(self.topology, wc.win_sys, integrator, platform, props)
            if resolved_box is not None:
                wc.sim.context.setPeriodicBoxVectors(*resolved_box)
            wc.sim.context.setPositions(positions)
            # lambda_shield 只存在于 VDW softcore 系统变体（build_ibs_dual_system）；
            # Shadow-Coulomb 变体（build_shadow_coul_ibs_system）里 VdW 全程满强度、
            # 不经过 λ-WCA 防护壳，不会注册这个 global parameter，需先判存在再设。
            if _system_has_global_parameter(wc.win_sys, "lambda_shield"):
                lam_vdw_center = float(np.mean(wc.lv_win))
                wc.sim.context.setParameter("lambda_shield", lam_vdw_center)
                print(f"  λ-WCA 防护壳已同步: lambda_shield={lam_vdw_center:.4f}")

            # 🔑 [2026-08-05 修，排序 bug] Boresch 必须在最小化**之前**就已经在
            # scale=1.0（生产强度），不能先无限制力自由最小化、再事后爬坡拽回来
            # ——旧顺序正是下面"阶段3"（已注释掉）存在的唯一理由，也是它仍会在
            # 某些窗口爬坡中途发散（Particle coordinate is NaN）的根因。
            if _has_valid_boresch_restraint(self.boresch):
                wc.sim.context.setParameter("lambda_boresch_scale", 1.0)

            # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 续验冻结校准 f_k 时，
            # 若存在指纹匹配的主窗口 checkpoint，直接续算这个 Context 的坐标/
            # 速度/盒子/积分器 RNG 状态，跳过下面的最小化/dt 测试步进/Boresch
            # 爬坡——这份 checkpoint 就是上次冻结验证中断时的真实动力学快照，
            # 对它做最小化是人为的势能下降移动、不是动力学步，会把样本挪出
            # 续采轨迹本该在的分布，不能因为"反正很快"就无条件加上。
            # loadCheckpoint 失败（缺失/损坏/跟当前 System/Platform 不兼容）必须
            # 安全回退到今天的完整重建流程，不能让一个坏 checkpoint 搞崩整窗口。
            restored_from_window_checkpoint = False
            if attempt_checkpoint_restore:
                restored_from_window_checkpoint = _try_load_main_window_checkpoint(wc.sim, main_ckpt_path)
                if restored_from_window_checkpoint:
                    print(
                        f"  窗口 {window_idx} 从主窗口 OpenMM checkpoint 续算"
                        "（跳过最小化/dt测试步进/Boresch爬坡，构象/速度/积分器RNG状态与上次中断时完全一致）。"
                    )
                    # 防御性重设：checkpoint 是否连带保存了 setParameter 的全局
                    # 参数值未经 100% 确认，代价极低，直接显式重设两个安全关键
                    # 参数，不依赖假设——呼应之前"探针最小化忘了同步 lambda_shield
                    # 导致动力学第一步 NaN"那次真实事故，这里不重蹈。
                    if _has_valid_boresch_restraint(self.boresch):
                        wc.sim.context.setParameter("lambda_boresch_scale", 1.0)
                    if _system_has_global_parameter(wc.win_sys, "lambda_shield"):
                        wc.sim.context.setParameter("lambda_shield", float(np.mean(wc.lv_win)))
                    # 🔑 [IBS_BIAS_PROTOCOL_VERSION=30] 续算的 checkpoint 已经跑过
                    # 最小化，残差力理应早就是全强度——同样按防御性重设原则显式
                    # 写回 1.0，不依赖 checkpoint 是否连带保存了这个新参数。
                    if ibs_wrap.residual_enabled:
                        wc.sim.context.setParameter(f"{ibs_wrap.prefix}_s_residual", 1.0)
                else:
                    print(f"  [WARN] 窗口 {window_idx} 主窗口 checkpoint 加载失败，回退到完整重建流程。")

            # ---------- 最小化 ----------
            if not restored_from_window_checkpoint:
                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=30] 2026-08-25 EXP-030 window_0
                # 诊断：EM 时 bias_scale 仍是构造默认值 1.0，candidate 在冷启动
                # f_k=0 下就已经用全强度残差力做无温控 L-BFGS 最小化；配对
                # baseline/candidate 密度探针显示同一起点逐点一致，candidate 之后
                # 局部环境原子数雪崩、baseline 全程稳定（不能只归咎于弱化的
                # softcore vdW）。原则同 2026-08-05 Boresch 排序修复：该在最小化
                # 前定好的状态不能等最小化后再纠正。这里只关残差项，不像
                # bias_scale=0 那样连 baseline 也在正常使用的物理 softcore-state
                # 混合力一起关掉。
                if ibs_wrap.residual_enabled:
                    wc.sim.context.setParameter(f"{ibs_wrap.prefix}_s_residual", 0.0)
                print(f"\n  [阶段1] 开始能量最小化...")
                wc.sim.minimizeEnergy(maxIterations=20000)
                print(f"  [OK] 最小化完成")
                initial_velocity_seed = self._seed_for(
                    "sampling", stage_type, window_idx, "velocity"
                )
                if initial_velocities is not None:
                    # EXP-030 paired arms consume the exact velocity array
                    # from their shared, hash-frozen source checkpoint.
                    wc.sim.context.setVelocities(initial_velocities)
                elif initial_velocity_seed is not None:
                    wc.sim.context.setVelocitiesToTemperature(
                        self.temperature, initial_velocity_seed
                    )

            # 几何检查
            if _has_valid_boresch_restraint(self.boresch):
                ok, r0_chk, thA_chk, thB_chk = _check_boresch_geometry_safe(wc.sim.context, self.boresch)
                if not ok:
                    raise RuntimeError(
                        f"窗口 {window_idx} Boresch 几何不合格 "
                        f"(r0={r0_chk*10:.1f}Å, θA={thA_chk:.1f}°, θB={thB_chk:.1f}°)"
                    )
                print(f"  [OK] Boresch 几何检查通过：r0={r0_chk*10:.2f}Å，θA={thA_chk:.1f}°，θB={thB_chk:.1f}°")

            # 分段回退（_step_with_chunk_rollback）耗尽步长下限时统一打这套诊断。
            # 热化和偏置预热共用一份，不各写一遍。
            def _nan_diagnostics(label: str) -> None:
                prefix = f"窗口{window_idx}_NaN_{label}"
                diagnose_force_groups_detailed(wc.sim.context, wc.win_sys, prefix=prefix)
                diagnose_force_breakdown(wc.sim.context, wc.win_sys, prefix=prefix)
                try:
                    diagnose_top_force_atoms(
                        wc.sim.context, wc.win_sys,
                        topology=self.topology,
                        ligand_indices=self.ligand_indices,
                        prefix=prefix,
                    )
                except Exception as diag_exc:
                    print(f"    [WARN] 原子级定位诊断失败：{diag_exc}")

            pre_test_breakdown = None
            if debug_mode:
                diagnose_force_groups_detailed(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_最小化后")
                pre_test_breakdown = diagnose_force_breakdown(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_最小化后")

            if not restored_from_window_checkpoint:
                # ---------- 测试步进 ----------
                print(f"\n[阶段2] 测试性步进（Boresch 全程 scale=1.0）...")
                if _has_valid_boresch_restraint(self.boresch):
                    # 只做 minimum-image 诊断，绝不通过改坐标“修复”锚点距离。
                    # 旧逻辑把整个配体平移 H0-L0，必然令 L0'=H0，制造零距离。
                    state_chk = wc.sim.context.getState(getPositions=True)
                    pos_chk = state_chk.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                    rec_idx = self.boresch["receptor_indices"]
                    lig_idx = self.boresch["ligand_indices"]
                    H0, L0 = pos_chk[rec_idx[0]], pos_chk[lig_idx[0]]
                    minimum_image = _minimum_image_displacement_nm(
                        H0 - L0,
                        state_chk.getPeriodicBoxVectors(),
                    )
                    image_dist = float(np.linalg.norm(minimum_image))
                    if not np.isfinite(image_dist) or image_dist <= 1.0e-6:
                        raise RuntimeError(
                            f"窗口 {window_idx} Boresch minimum-image 锚点距离无效: "
                            f"{image_dist} nm"
                        )
                    print(
                        f"  [OK] Boresch minimum-image 锚点距离: {image_dist*10:.2f}Å "
                        "（未修改坐标）"
                    )
                    # 🔑 [2026-08-05 注释掉，排序 bug 修复的一部分] 不再把 scale
                    # 缩到 1%——它现在已经在 setPositions 之后、最小化之前被设成
                    # 1.0（生产强度），全程保持，不需要在这里往下调。
                    # sim.context.setParameter("lambda_boresch_scale", 0.01)  # 1%
                    # print(f"  🔧 Boresch 力常数缩放至 1%")

                original_dt = wc.sim.integrator.getStepSize()
                # 🔑 [2026-09-03] 这里原来是一段"测试步进 schedule"：把 dt 从
                # 0.01 fs 一路爬到 0.5 fs、每 50 步下载一次整帧
                # getState(positions+forces+energy)（单窗口 218 次 GPU→CPU 同步），
                # 一炸就直接 raise、没有任何挽救。
                #
                # 那是脚手架，不是物理：EM 之后 max|F| 已经收敛，配体↔环境全部走
                # softcore（能量有限、力有界，没有 r^-13 奇点），常规情况下根本不
                # 需要从 0.01 fs 起步。**现在没有任何预备 schedule** —— 直接用生产
                # 步长跑一小段热化，只有真的炸了才处理，而且只回退炸掉的那一小段：
                # 回滚到该段开始前的构型/速度 → 步长减半 → 重做这一段 → 恢复步长
                # 继续。连续减半到 0.01 fs 仍然炸才判这个窗口失败。
                _THERMALIZE_STEPS = 10000          # 生产步长下的热化总步数
                _CHUNK = 500                       # 一"帧"= 检查粒度，也是回退粒度
                _MIN_DT_PS = 1.0e-5                # 减半下限 0.01 fs，到这儿还炸就是真坏了

                _rescued = step_with_chunk_rollback(
                    wc.sim,
                    _THERMALIZE_STEPS,
                    f"窗口 {window_idx} 热化",
                    chunk=_CHUNK,
                    min_dt_ps=_MIN_DT_PS,
                    on_exhausted=_nan_diagnostics,
                )
                if _rescued:
                    print(f"  [OK] 热化完成（回退重做 {_rescued} 次）")

                wc.sim.integrator.setStepSize(original_dt)
                print(f"  [OK] 测试步进通过，恢复步长 {original_dt.value_in_unit(unit.picoseconds):.3f} ps")
                post_test_breakdown = None
                if debug_mode:
                    post_test_breakdown = diagnose_force_breakdown(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_测试步进后")
    
                deadlock_msg = _detect_constraint_deadlock(
                    pre_test_breakdown,
                    post_test_breakdown,
                    wc.win_sys.getNumConstraints(),
                )
                if deadlock_msg is not None:
                    print(f"  [ERR] [约束死锁预警] {deadlock_msg}")
                    print("  自动切换到 1.0 fs 保守步长，并执行额外最小化/松弛以避免生产阶段首步崩溃...")
                    original_dt = 0.001 * unit.picoseconds
                    wc.sim.integrator.setStepSize(original_dt)
                    wc.sim.minimizeEnergy(maxIterations=5000, tolerance=10.0)
                    step_with_chunk_rollback(
                        wc.sim, 2000, f"窗口 {window_idx} 死锁缓解松弛",
                        chunk=200, on_exhausted=_nan_diagnostics,
                    )
                    if debug_mode:
                        diagnose_force_breakdown(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_死锁缓解后")
    
                # 🔑 [2026-08-05 整段注释掉，排序 bug 修复] 原"阶段3 Boresch 安全
                # 爬坡"存在的唯一理由是补偿上面已经修掉的排序 bug：旧代码先在
                # scale=0 下自由最小化（配体漂离 committed r0 达 ~0.2nm），再靠
                # 这段 16 级自定义阶梯把它硬拽回限制力平衡点——拽的过程本身在
                # 某些窗口（如这次的窗口 5）会在 sim.minimizeEnergy() 内部就已经
                # 发散，抛出 Particle coordinate is NaN，而这段爬坡自己的能量/力
                # 检查全部是"minimizeEnergy 返回之后"才做，根本挡不住这一种失败
                # 模式。现在 Boresch 从最小化第一步起就已经在 scale=1.0，最小化
                # 收敛点本身就满足限制力平衡，不再有"漂开再拽回"这回事，这整段
                # 机制随之失去存在理由。整段注释掉而不是删除，留作历史参考；
                # 不要因为将来某个窗口在满强度最小化下不稳定，就把这段爬坡重新
                # 启用——那只是把同一个问题从"爬坡中途"挪回"最小化中途"，该查
                # 的是那个窗口本身的起始构象/λ 组合，不是重新引入分级 ramp。
                # ---------- Boresch 安全爬坡 ----------
                # ================================================================
                # Boresch 安全爬坡：自定义阶梯，逐个恢复力强度
                # ================================================================
                # if _has_valid_boresch_restraint(self.boresch):
                #     print(f"\n[阶段3] Boresch 安全爬坡（自定义阶梯）...")
                #
                #     # 自定义阶梯序列：低强度区采用更细分辨率，避免在高内应力底座上一步踩空。
                #     custom_scales = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
                #     dt_ramp = 0.001                  # 爬坡期间极小步长 (ps)
                #
                #     # 保存原始步长，并设置为爬坡步长
                #     original_dt = sim.integrator.getStepSize()
                #     sim.integrator.setStepSize(dt_ramp * unit.picoseconds)
                #     print(f"  → 爬坡使用步长 {dt_ramp} ps，低强度区采用更细台阶")
                #
                #     # 确保从当前 scale 开始（例如之前测试步进时设置的 0.01）
                #     try:
                #         current_scale = sim.context.getParameter("lambda_boresch_scale")
                #     except Exception:
                #         current_scale = 0.01
                #     print(f"  → 起始 Boresch scale = {current_scale:.3f}")
                #     prev_energy = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                #
                #     ramp_success = True
                #     for target_scale in custom_scales:
                #         # 只在目标大于当前值时才爬升（避免回退）
                #         if target_scale <= current_scale:
                #             continue
                #
                #         sim.context.setParameter("lambda_boresch_scale", float(target_scale))
                #         n_steps_per_level = 1500 if target_scale <= 0.10 else (1000 if target_scale <= 0.30 else 500)
                #         print(f"  🔹 设置 Boresch scale = {target_scale:.2f}，松弛 {n_steps_per_level} 步...", end="", flush=True)
                #         sim.minimizeEnergy(maxIterations=200, tolerance=20.0)
                #         for _ in range(max(1, n_steps_per_level // 100)):
                #             sim.step(100)
                #
                #         # 检查能量与受力
                #         state = sim.context.getState(getEnergy=True, getForces=True)
                #         e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                #         forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
                #         max_f = np.max(np.linalg.norm(forces, axis=1))
                #         delta_e = e - prev_energy
                #
                #         if (not np.isfinite(e)) or max_f > 50000 or (abs(delta_e) > 5e5 and max_f > 15000):
                #             print(f"\n  🚨 Boresch 爬坡在 scale={target_scale:.2f} 处失败！")
                #             print(f"    当前势能 = {e:.1f} kJ/mol，ΔE = {delta_e:.1f} kJ/mol，最大力 = {max_f:.1f} kJ/(mol·nm)")
                #             if debug_mode:
                #                 diagnose_force_breakdown(sim.context, win_sys, prefix=f"窗口{window_idx}_Boresch爬坡失败_scale{target_scale:.2f}")
                #             ramp_success = False
                #             break
                #         else:
                #             print(f" 势能 = {e:.2f} kJ/mol，ΔE = {delta_e:.2f} kJ/mol，最大力 = {max_f:.2f} kJ/(mol·nm)")
                #             # 记录新的当前 scale
                #             current_scale = target_scale
                #             prev_energy = e
                #
                #     # 恢复原始步长
                #     sim.integrator.setStepSize(original_dt)
                #
                #     if not ramp_success:
                #         raise RuntimeError(
                #             f"窗口 {window_idx} Boresch 爬坡失败（scale={target_scale:.2f}），系统可能已崩溃。"
                #             "请检查锚点几何、力常数或初始结构。"
                #         )
                #
                #     print(f"  ✅ Boresch 爬坡成功完成 (scale 已恢复至 1.0)")
                # else:
                #     print(f"\n[阶段3] 无 Boresch 限制力，跳过爬坡。")

            # ================================================================
            # 热化完成（Boresch 全程 scale=1.0，阶段3 爬坡已注释掉不再需要）：
            # 立刻进行一次非侵入式力分解，供对比分析
            # ================================================================
            if debug_mode and self.boresch:
                diagnose_force_breakdown(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_热化完成")

            # ---------- 初始化采样器 ----------
            # ibs_state_file 已经在窗口开头（checkpoint restore 判断之前）提前
            # 拼接过，这里不再重复。
            wc.sampler = IBSSampler(wc.sim.context, len(wc.lc_win), self.temperature, self.prefix, ibs_wrapper=ibs_wrap)
            wc.sampler.sampling_score_sha256 = self.sampling_score_sha256
            wc.sampler.residual_plugin_identity = self.residual_plugin_identity
            wc.sampler.residual_em_policy = self.residual_em_policy
            wc.sampler.residual_feature_name = self.residual_feature_name
            # 🔑 [non_mutating_v1] stamp the current run's policy so save_ibs_state
            # records it (old-policy state files carry None → detected on load).
            wc.sampler.sampling_repair_policy = repair_policy
            # 🔑🔑 [2026-09-14 真机根因] **采样身份必须盖到 sampler 上。**
            #
            # `save_ibs_state` 是 `IBSSampler` 的方法，写的是
            # `getattr(self, "stage_protocol_key", None)` —— 而 `stage_protocol_key`
            # 只存在于**管理器**（`IBSWindowManagerDualLambda`）。全仓从来没有任何一处
            # 把它盖到 sampler 上 ⟹ **每一份 `ibs_state_*.json` 里这个字段永远是 None**。
            #
            # 后果不是少一条元数据，是一条**永久关闭的闸**：
            #     save → 永远 None
            #     load → `loaded_stage_protocol_key` 永远 None
            #     判定 → `_identity_matches` 永远 False
            #     ⟹ `skip_warmup_entirely` 永远 False
            #     ⟹ **每个 resume 的窗口都重走一遍完整预热收敛判定**，白烧预热预算。
            # 而预热预算耗尽正是窗口"再也进不去"的原因（补帧动作被入口门弹回、
            # 自治循环空转到迭代上限）。2026-09-11 加的那道「旧 PASS 只在同一采样身份
            # 下有效」的闸，从落地那天起就没真正生效过 —— 它每次报的都是
            # "身份**缺失**"，不是"身份不符"。
            #
            # 五个 benchmark run 实测：`ibs_state_vdw_window_*.json` 里
            # `stage_protocol_key` / `coion_identity` 全是 None，而同窗口的
            # `convergence.json` 里两者都有。
            #
            # ⚠️ 这**不改变**任何判据，只是让写侧真的把它写进去。既有 state 里仍是
            # None ⟹ 第一次 resume 照旧重走一遍预热（那次是真的没身份），之后才干净。
            wc.sampler.stage_protocol_key = getattr(self, "stage_protocol_key", None)
            wc.sampler.coion_identity = getattr(self, "coion_identity", None)

            # 🔑 核心修复：断点续传状态检测
            warm = WarmupState()
            if resume and os.path.exists(ibs_state_file):
                warm.is_resumed_ibs = wc.sampler.load_ibs_state(
                    ibs_state_file, wc.lc_win, wc.lv_win, stage_type=stage_type
                )

            # 🔑 [终态硬停止] load_ibs_state 已经完成了 n_states/prefix/协议版本/
            # λ 内容的严格匹配（不是廉价 peek），确认这份状态真的属于当前窗口后，
            # 若上一轮已经把它判定为 calibrated_validation_failed（冻结验证累计
            # 预算用到最后一档仍未通过），必须立刻硬停止——不做任何 SGD/续验
            # 尝试，也不落入下面"is_resumed_ibs and not sampler.bias_converged"
            # 分支被当成普通未收敛热启动重新学习。调用方（abfe_pipeline.py）看到
            # 这个 terminal=True 的异常应该直接向上传播，不再自动重试。
            # [Candidate-first, Validate-or-Learn v1] 唯一在这段区间内做的改动：
            # 新协议下 VALIDATE 耗尽预算后的终态失败写 "failed"（不再写
            # calibrated_validation_failed），必须在这里同样被识别为终态，
            # 否则会被下面 `elif is_resumed_ibs and not sampler.bias_converged`
            # 分支误当成普通未收敛热启动重新学习。除这一处 in (...) 之外，
            # 10586-10689 区间的其余分支/语义本次刻意不做任何改动。
            if warm.is_resumed_ibs and wc.sampler.bias_status in (
                "calibrated_validation_failed", "failed",
            ):
                raise IBSFrozenCalibrationValidationError(
                    f"窗口 {window_idx} 的冻结校准验证此前已判定为终态失败"
                    f"（bias_status={wc.sampler.bias_status!r}）：VALIDATE 累计预算已耗尽仍未能"
                    "通过独立验证——不再自动续验、不延长预算、不回退 learning。需要人工检查"
                    "（构象弛豫确实很慢，或偏置表达式仍有问题），人工确认后需手动清空该窗口的 "
                    "IBS 状态文件才能重新开始。",
                    diagnostics={
                        "window_index": int(window_idx),
                        "stage_type": stage_type,
                        "bias_status": wc.sampler.bias_status,
                        "calibration_validation_terminally_failed": True,
                        # 🔑 [2026-09-14 / S1] **这个 dict 是封存记录的唯一身份来源。**
                        # 原来这里只有上面 4 个键 ⟹ 调用方封存出来的每条记录都是
                        # `f_k_kJ_mol: []` + 指纹 null ⟹ `sealed_candidate_matches`
                        # 永远匹配不上，"永不续验"形同虚设。终态失败时
                        # `frozen_f_k_pending` 已被清空，但被驳回的那份 f_k 就在
                        # Context 参数里（load_ibs_state 刚注入过），从那里取。
                        **refuted_candidate_identity(
                            [float(wc.sim.context.getParameter(f"{self.prefix}_f_{_k}"))
                             for _k in range(len(wc.lc_win))],
                            window_idx,
                        ),
                    },
                    terminal=True,
                )

            # 🔑 [non_mutating_v1] Fail closed on any resumed state that ONLY the
            # deprecated mutating calibration path could have produced. Under
            # non_mutating_v1 f_k is never MBAR-recalibrated in place, so a
            # loaded "calibrated_pending_validation" status / non-null
            # frozen_f_k_pending can only come from an old mutating run.
            # Continuing to validate it would build production on an f_k from a
            # different reference frame — refuse, preserve the state files, and
            # hand the ensemble to the rescue audit.
            if (
                warm.is_resumed_ibs
                and not legacy_repair
                and getattr(wc.sampler, "bias_status", "unconverged")
                != "frozen_validation_indeterminate"
                and (
                    getattr(wc.sampler, "bias_status", "unconverged") == "calibrated_pending_validation"
                    or wc.sampler.frozen_f_k_pending is not None
                )
            ):
                raise ExistingEnsembleRequiresRescueAudit(
                    f"窗口 {window_idx}: 载入的 IBS 状态带有 calibrated_pending_validation / "
                    f"frozen_f_k_pending（bias_status={getattr(wc.sampler, 'bias_status', None)!r}），"
                    "这只可能由已弃用的变异校准路径产生。non_mutating_v1 从不就地重校准 f_k，"
                    "拒绝续验（否则会把生产建立在旧参考系的 f_k 上），保留该窗口全部状态文件，"
                    "交 rescue 审计判定是否可救、是否需要重跑。",
                    diagnostics={
                        "window_index": int(window_idx),
                        "stage_type": stage_type,
                        "bias_status": getattr(wc.sampler, "bias_status", None),
                        "requires_rescue_audit": True,
                    },
                )

            # bias 预热收敛诊断（在分支之外先占位，确保后面落盘时一定有值可写）。

            # ---------- 渐进预热 ----------
            # 🔑 之前"续传就跳过 Warmup"是不安全的：load_ibs_state 恢复的 f_k
            # 可能是上次中断时还没收敛的中间值（旧协议下甚至可能是"生产阶段仍在
            # 被 update_weights() 调整"的漂移值），直接当成已收敛放行进生产，
            # 会让整段生产采样建立在一个从未真正验证过的偏置上。现在只有当
            # load_ibs_state 明确报告 bias_converged=True（该状态本身是在严格
            # 收敛判据下才会被置位并落盘的）时才真正跳过；否则一律走下面的收敛
            # 判定循环——已恢复的 f_k 只是被当成一个更好的热启动起点，不是免检的
            # 通行证。
            # 🔑 [2026-09-11] 旧 PASS 只在**同一个采样身份**下有效。
            # 用 _stage_window_sampling_identity 比对（它刻意剥掉预算/门限这些
            # 可以变的项，只留"采的是不是同一个 Hamiltonian"）。
            # 身份不符、或旧状态根本没记身份 ⟹ **不给 skip_warmup_entirely**。
            # 注意这不是把状态整份丢掉：f_k 仍然作为热启动种子走
            # `is_resumed_ibs and not bias_converged` 那条路，只是不再能声称
            # "已经收敛、只需只读复验"。
            _cached_stage_identity = getattr(wc.sampler, "loaded_stage_protocol_key", None)
            _current_stage_identity = getattr(self, "stage_protocol_key", None)
            _identity_matches = (
                _current_stage_identity is not None
                and _cached_stage_identity is not None
                and _stage_window_sampling_identity(_cached_stage_identity)
                == _stage_window_sampling_identity(_current_stage_identity)
            )
            skip_warmup_entirely = bool(
                warm.is_resumed_ibs and wc.sampler.bias_converged and _identity_matches
            )
            if warm.is_resumed_ibs and wc.sampler.bias_converged and not _identity_matches:
                print(
                    f"  [WARN] 窗口 {window_idx} 载入的 IBS 状态标记为已收敛，但采样身份"
                    f"{'缺失' if _cached_stage_identity is None else '与当前不符'}"
                    "（λ / Hamiltonian / box / 口径版本任一变化即算不同系综）。"
                    "旧 PASS 不再粘住：f_k 仍作热启动，但要重新走完整收敛判定。"
                )
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=7] 缓存的 bias_converged=True 只能证明这个
            # f_k 曾经在旧构型下有效，不能证明本次新建 Context 的当前构型已经在这个
            # 固定偏置下平衡过——旧代码在这里直接跳过整个预热块进生产，等于零平衡就
            # 采信一个从未在当前 Context 里验证过的分布。现在不再整段跳过：仍然完全
            # 跳过 learning（不重新调整已恢复的 f_k），但下面的状态机会从
            # freeze_burn_in 开始，对这份恢复的 f_k 重新做一次冻结 burn-in + 只读验证，
            # 通过了才真正放行进生产；验证失败则和全新窗口一样自动回退到 learning。
            if skip_warmup_entirely:
                wc.sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
                print(
                    "  检测到已收敛的 IBS 历史状态：跳过 learning，"
                    "直接对已恢复的 f_k 重新做一次冻结 burn-in + 只读验证"
                    "（不重新学习权重，但也不会零平衡就直接进生产）。"
                )
                warm.resumed_frozen_f_k = [
                    float(wc.sim.context.getParameter(f"{self.prefix}_f_{k}"))
                    for k in range(len(wc.lc_win))
                ]
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] MBAR 校准过的 f_k 冻结验证未在
            # 预算内通过时不能落进下面这个"普通未收敛"分支——那个分支会把
            # bias_scale 设为 1.0 后让状态机从 mode="learning" 开始，SGD 会
            # 立刻开始修改这份其实已经用 fixed-H overlap + bias 校准探针证明
            # 过是对的 f_k。必须单独识别 bias_status=="calibrated_pending_validation"，
            # 直接用 frozen_f_k_pending 当 resumed_frozen_f_k、跳过 learning，
            # 从 freeze_burn_in 开始只延长冻结验证。
            elif (
                warm.is_resumed_ibs
                and getattr(wc.sampler, "bias_status", "unconverged") == "calibrated_pending_validation"
                and wc.sampler.frozen_f_k_pending is not None
            ):
                wc.sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
                for k, f_val in enumerate(wc.sampler.frozen_f_k_pending):
                    wc.sim.context.setParameter(f"{self.prefix}_f_{k}", float(f_val))
                print(
                    "  检测到 MBAR 校准已冻结但验证尚未通过的历史状态："
                    "跳过 learning 与 fixed-H overlap/bias 校准探针，直接延长这份"
                    "冻结 f_k 的 burn-in + 只读验证（不重新学习权重，也不重新校准）。"
                )
                warm.resumed_frozen_f_k = [float(x) for x in wc.sampler.frozen_f_k_pending]
            elif (
                warm.is_resumed_ibs
                and getattr(wc.sampler, "bias_status", "unconverged")
                == "frozen_validation_indeterminate"
                and wc.sampler.frozen_f_k_pending is not None
            ):
                # 🔑 [2026-09-11] 上次是"验证预算耗尽、无法判定"（数据不足，Δf−ΔF
                # 从未求出）。这对 f_k 不是反面证据：不重新 SGD、不重新校准，恢复
                # 同一份冻结 f_k 接着验；已耗批数从 frozen_validation_batches_done
                # 恢复，不重新领一份批次额度。
                wc.sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
                for k, f_val in enumerate(wc.sampler.frozen_f_k_pending):
                    wc.sim.context.setParameter(f"{self.prefix}_f_{k}", float(f_val))
                print(
                    "  检测到上次'验证预算耗尽、无法判定'的历史状态：恢复同一份冻结"
                    f" f_k 接着验（已累计 "
                    f"{getattr(wc.sampler, 'frozen_validation_batches_done', 0)} 批），"
                    "不重新学习、不重新领批次额度。"
                )
                warm.resumed_frozen_f_k = [float(x) for x in wc.sampler.frozen_f_k_pending]
            elif warm.is_resumed_ibs and not wc.sampler.bias_converged:
                print(
                    "  检测到 IBS 历史状态，但此前预热未判定收敛——"
                    "以该状态为热启动起点继续走严格收敛判定，不清零重来，也不假设已经收敛。"
                )
                wc.sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
            else:
                if enable_gradual_warmup:
                    # 只做时间步长爬坡；偏置力的唯一平滑引入通道是下面的
                    # "[偏置预热]" 块，两者不再重复覆盖同一个 bias_scale 目标。
                    self._gradual_warmup_debug(wc.sim, ibs_wrap, wc.sampler, warmup_steps, wc.win_sys, window_idx, debug_mode)
                # 不管上面是否做了时间步长爬坡，bias_scale 的唯一一次 0→1.0
                # 平滑引入都交给紧接着的"[偏置预热]"块（它会显式把 bias_scale
                # 重置到 0.0 再爬升，不依赖这里是什么值）。

            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=30] 到这里为止（这一整段仍在
            # `not restored_from_window_checkpoint` 内，即本窗口这次确实跑过
            # 最小化）四个分支都可能已经把 bias_scale 直接设成 1.0 并跳过下面
            # 的"[偏置预热]"爬坡块（skip_warmup_entirely / calibrated_pending_
            # validation / is_resumed_ibs-but-not-converged 三种），或者完全没碰
            # bias_scale、交给下面的爬坡块处理——无论走了哪一条，最小化阶段临时
            # 关闭的残差项都必须在这里统一恢复到 1.0，不能只在爬坡块里恢复
            # （否则前三个分支会让残差力从这个窗口开始永久关闭）。
            if ibs_wrap.residual_enabled:
                wc.sim.context.setParameter(f"{ibs_wrap.prefix}_s_residual", 1.0)

            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 这次 resume 是否直接续验已经
            # 校准好的冻结 f_k——是的话，下面必须跳过 fixed-H overlap 探针和
            # MBAR 校准（f_k 已经是校准好的，不需要也不应该重新学习/重新校准），
            # 且这次冻结验证要用 frozen_validation_step_overrides 里给这个窗口
            # 累计延长过的预算，而不是默认的 mbar_calibration_reserved_steps。
            resumed_calibration_pending = bool(
                warm.resumed_frozen_f_k is not None and warm.is_resumed_ibs
                and getattr(wc.sampler, "bias_status", "unconverged") == "calibrated_pending_validation"
            )
            # 🔑 [2026-09-11] 概念拆分。`resumed_calibration_pending` 以前同时表达
            # 三件事：要不要恢复冻结验证、这份 f_k 有没有校准证据、用哪份预算。
            # 无法判定态需要**第一件**（把对当前候选的测量做完），但绝不能继承
            # 第二件（"f_k 已正确"）——所以只有动作那一层共用。
            resume_frozen_validation = bool(
                warm.resumed_frozen_f_k is not None and warm.is_resumed_ibs
                and getattr(wc.sampler, "bias_status", "unconverged") in (
                    "calibrated_pending_validation",
                    "frozen_validation_indeterminate",
                )
            )
            # 只有调用方**显式**给了这个窗口一个新的累计上限才算升档。以前没有持久
            # 账本，所以 resolver 在 override 缺失时按已耗步数往上找一档兜底——那等于
            # 每次 --resume 自动升一档、白送额度。现在上限随账本持久化，兜底取消。
            _explicit_frozen_validation_rung = (
                int(frozen_validation_step_overrides[window_idx])
                if window_idx in (frozen_validation_step_overrides or {})
                else None
            )
            # 🔑 [跨进程阶梯状态修复] frozen_validation_step_overrides/
            # frozen_validation_is_final_rung 只存在调用方 abfe_pipeline.py 一次
            # 进程运行期间的本地 dict，从不落盘——一旦作业被杀、用 --resume 起
            # 一个全新进程，这两个 dict 对这个窗口都是空的，之前这里会直接退回
            # 一个跟已持久化进度无关的固定默认值 mbar_calibration_reserved_steps
            # (50000)，如果这个窗口的 frozen_validation_cumulative_steps 早就
            # 超过了 50000（几乎总是这样，只要它已经续验过至少一轮），
            # remaining_budget_this_attempt 会被下面的 max(..., check_chunk) 封底
            # 成几乎为零的一小段（500 步），这一整个 attempt 基本被浪费掉，阶梯
            # 还得从第一档重新往上爬——真实预算被这个重启悄悄削减。现在的回退
            # 逻辑不再假设"没有覆盖字典就等于第一档"，而是直接从已持久化的
            # frozen_validation_cumulative_steps 反推"下一档没跑完的目标"：
            # 取 FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS 中第一个大于当前累计
            # 步数的档位（找不到就说明已经跑满最后一档，直接落在最后一档，让
            # 下面 is_final_rung 的回退逻辑正确判定终态）。调用方显式传入的
            # 覆盖值（同进程内正常阶梯升档的情况）优先于这个回退，行为不变。
            effective_frozen_validation_budget = _resolve_frozen_validation_budget_for_window(
                window_idx,
                frozen_validation_step_overrides,
                int(getattr(wc.sampler, "frozen_validation_cumulative_steps", 0)),
            )

            if debug_mode:
                print(f"\n[诊断] 预热后状态检查：")
                diagnose_force_groups_detailed(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_预热后")
                diagnose_force_breakdown(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_预热后")
                diagnose_softcore_cv_values(wc.sim.context, ibs_wrap, wc.lc_win, wc.lv_win, prefix=f"窗口{window_idx}_预热后", sampler=wc.sampler)
            raw_probe = wc.sampler.get_raw_interaction_energies()
            if len(raw_probe) >= 2:
                raw_span = float(np.max(raw_probe) - np.min(raw_probe))
                if raw_span < 1e-6 and np.max(np.abs(raw_probe)) < 1e-6:
                    raise RuntimeError(
                        f"窗口 {window_idx} 的 IBS 软核 CV 全部为 0。"
                        "这不是窗口重叠问题，而是 CV 构造/读取链路异常；已中止以避免产出全零溶剂腿或失真复合物腿。"
                    )

            # 清空 sampler 的能量缓存
            wc.sampler.energy_buffer = []
            wc.sampler.energy_history = []
            wc.sampler.bias_history = []
            wc.sampler.base_energy_history = []

            # ⚠️ original_dt 在此无条件捕获（不能挪进下面的 if 分支里）：续传
            # 分支跳过偏置预热块，但"生产前卸压"结束后仍会无条件用它恢复步长。
            original_dt = wc.sim.integrator.getStepSize()

            # ---------- 偏置力预热 (Bias Ramp) ----------
            # ✅ 性能/正确性修复：这是 bias_scale 唯一一次真正的 0→1.0 平滑爬坡
            # （_gradual_warmup_debug 只做时间步长爬坡，不再重复覆盖 bias_scale）。
            # 续传时 (is_resumed_ibs=True) 上面已经把 bias_scale 直接设为 1.0 并
            # 打印"跳过 Warmup"——这里必须真的跳过，否则会打脸自己的日志，且对
            # 已收敛的 f_k 做一次无意义的清零重爬。
            # [IBS_BIAS_PROTOCOL_VERSION=29] 是否已经用 pilot 种子播过 f_k；否则冷启动
            # 时下面（scale=1.0 后）用自举 TI 种子兜底，避免从 f_k=0 慢爬。
            if not warm.is_resumed_ibs:
                # [IBS_BIAS_PROTOCOL_VERSION=27] The pilot-TI estimator returns
                # the physical F(lambda), mean-centered, which is the
                # ready-to-use convention for exp[-beta*(U_k-f_k)].  Do not
                # infer this sign from instantaneous occupancy: occupancy-high
                # states are lowered by the separate online negative feedback.
                # 显式给的逐窗口种子优先于 pilot-TI：它来自上一段的**生产帧**，
                # 比 pilot 探针准得多。长度不符一律忽略（下面统一走长度校验）。
                _explicit_seed = None
                if initial_f_k_by_window:
                    _explicit_seed = initial_f_k_by_window.get(int(window_idx))
                if _explicit_seed is not None:
                    warm_start_seed = np.asarray(_explicit_seed, dtype=np.float64).ravel()
                    _seed_source = "production_recalibration"
                else:
                    from abfe_preoptimizer import estimate_f_k_from_pilot_ti
                    warm_start_seed = estimate_f_k_from_pilot_ti(
                        getattr(self, "pilot_lambdas", None),
                        getattr(self, "pilot_mean_dU_dlambda", None),
                        target_lambdas=wc.lv_win,
                    )
                    _seed_source = "pilot"
                if warm_start_seed is not None and len(warm_start_seed) == len(wc.lv_win):
                    for k in range(len(wc.lv_win)):
                        wc.sim.context.setParameter(f"{self.prefix}_f_{k}", float(warm_start_seed[k]))
                    warm.f_k_warm_started = True
                    wc.sampler.seed_source = _seed_source
                    print(
                        f"  [{_seed_source} 热启动] 窗口 {window_idx} f_k 初始值（非冷启动 0.0）: "
                        f"{[round(float(x), 3) for x in warm_start_seed]} kJ/mol"
                    )
                elif warm_start_seed is not None:
                    print(
                        f"  [WARN] [{_seed_source} 热启动] 窗口 {window_idx} 估计出的种子长度 "
                        f"({len(warm_start_seed)}) 与本窗口态数 ({len(wc.lv_win)}) 不符，"
                        "拒绝注入，回退 f_k=0.0 冷启动"
                    )
                else:
                    print(f"  [pilot TI 热启动] 窗口 {window_idx} 无可用 pilot 数据，f_k 仍从 0.0 冷启动")

                print(f"\n  [偏置预热] 开始... (从零缓慢加载偏置力)")
                wc.sim.integrator.setStepSize(0.001 * unit.picoseconds)

                wc.sim.context.setParameter(f"{self.prefix}_bias_scale", 0.0)

                # 🔑 [2026-09-09] 补上 1.0 这一档。原来爬坡停在 0.7，然后在下面
                # （`bias_scale` 设 1.0 处）直接跳到满偏置：**最后也是最大的一次
                # 增量既没有弛豫档、也没有分段回退保护**，跳完之后第一段动力学
                # 是裸的 `guarded_step`（偏置学习循环）——而 pilot 种子存在时
                # `f_k_warm_started=True`，中间那段自举 TI 采样整块跳过，所以那
                # 一步就是跳到 1.0 之后的第一次积分。
                #
                # 配上 pilot TI 种子的量级（实测窗口 0 是 ±36 kJ/mol ≈ 29 kT），
                # Group-1 的 log-sum-exp 会塌到单态、施加的偏置力就是那个态的完整
                # softcore 力。2026-09-09 的 OpenFF benchmark 里 10 个 run 全部死在
                # 这一跳，最后一行日志正是上面那句 pilot 种子打印。
                #
                # 爬坡那半边当天已经接上分段回退，1.0 这半边没有。补齐后整条爬坡
                # 每一档都受同一套保护，且 0.7→1.0 这个最大增量也有弛豫。
                ramp_stages = [
                    (0.2, 2000),  # 可以适当增加每一步的步数，让系统充分弛豫
                    (0.3, 2000),
                    (0.5, 2000),
                    (0.7, 2000),
                    (1.0, 2000),
                ]
                wc.sampler.energy_buffer = []
                wc.sampler.energy_history = []
                wc.sampler.base_energy_history = []
                wc.sampler.bias_history = []
                # 🔑 [2026-09-09] 这两条必须跟上面四条一起清。它们与
                # `energy_history` 受 `collect_energies()` 里**同一个** frame_finite
                # 门同步 append（8121-8129），而 `_append_tmbar_batch_from_buffer`
                # (8176) 取的是 `sampling_state_energy_history[-m0:]` 并只校验尾部
                # 长度 —— 那是个尾部检查，测不出前面已经错位。少清这两条，
                # 下一批 tmbar entry 的 u_kn 就会对错 bias/base energies。
                wc.sampler.sampling_state_energy_history = []
                wc.sampler.residual_basis_history = []
                wc.sampler.ema_mean_p = None
                for target_scale, steps in ramp_stages:
                    wc.sim.context.setParameter(f"{self.prefix}_bias_scale", target_scale)
                    print(f"    → Bias Scale 设为 {target_scale}, 运行 {steps} 步...", end="", flush=True)

                    # 🔑 [2026-09-09] 这里原来是 for + sim.step(200)，注释写着
                    # "每 200 步检查一次"但循环体里根本没有检查，唯一的有限性
                    # 检查在整段跑完之后——而 NaN 是从 step() 内部抛出来的，那个
                    # 检查永远走不到，窗口直接炸穿。现在复用热化那段同一个分段
                    # 回退：炸了就回滚这一小段、步长减半重做。
                    _ramp_rescued = step_with_chunk_rollback(
                        wc.sim,
                        steps,
                        f"窗口 {window_idx} 偏置预热 scale={target_scale}",
                        chunk=200,
                        on_exhausted=_nan_diagnostics,
                    )
                    print(" 完成" + (f"（回退重做 {_ramp_rescued} 次）" if _ramp_rescued else ""))
            # ---- bias_scale=1.0：两阶段严格收敛判据（learning -> freeze -> validate）----
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=7] v6 判据有个更深的问题：一轮"连续
            # N 次通过"里，每一次通过用的 mean_p_batch/ema_mean_p 都是从
            # update_weights() 内部的 f_old 算出来的，而 update_weights() 在同一
            # 次调用里立刻把 f_old 换成 f_new 并写回 context——真正冻结进生产的
            # 那个 f_k，从未被任何一次"通过"检验过；"连续 3 次通过"检验的是 3 个
            # 不同、都已经被丢弃的旧 f_k，不是同一个冻结 Hamiltonian 下的 3 次独立
            # 验证，不能证明分布已经稳定。正确做法改成两阶段：
            #   1) [learning] 沿用旧的 update_weights() + 连续通过判据，达到后只是
            #      "候选收敛"，不直接宣布 converged。已恢复的收敛缓存
            #      （resumed_frozen_f_k 非 None）跳过这一阶段，直接从 freeze_burn_in
            #      开始，用恢复的 f_k 当候选，而不是重新学习。
            #   2) 冻结 f_k（此后到验证结束前绝不再调用 update_weights()）→ 丢弃
            #      一段 burn-in（之前的样本来自还在漂移的偏置，冻结后需要重新
            #      平衡）→ 用 evaluate_frozen_batch_probability()（只读，不写 f_k）
            #      在真正固定的 Hamiltonian 下累计新 batch；至少达到指定 batch 数
            #      且累计 <p_k> 通过同一 LSE 门才宣布 converged。不能用 EMA，也不能
            #      像 v17 那样让一个仅 20-frame 的 batch 一失败就立刻切回 learning。
            #   3) 一次完整冻结验证 attempt 仍失败：先把这批 held-out 样本作为新的
            #      Sec. 2.3 TA iteration 并入 Q_hat，再恢复 [learning]；下一候选仍需
            #      重新 burn-in，并用全新的固定-f 样本独立验证。
            # 更新次数上限 max_bias_updates 只应该限制 learning：如果恰好在第
            # max_bias_updates 次更新时形成候选收敛，mode 已切到 freeze_burn_in，但
            # 循环条件若仍把 bias_update_count 当整体退出条件，下一轮会因为
            # "count < max_bias_updates" 为假直接退出，burn-in/validation 根本来
            # 不及执行——所以这里改成只在 mode=="learning" 时检查这个上限，
            # freeze_burn_in/validating 不消耗、也不受这个额度限制，只受
            # steps_at_full_bias 这个总步数安全帽约束。
            wc.sim.context.setParameter(f"{self.prefix}_bias_scale", 1.0)
            print("    → Bias Scale 设为 1.0，按两阶段收敛判据运行...", end="", flush=True)
            K = len(wc.lc_win)
            target_p = 1.0 / K
            min_probability_threshold = 0.5 * target_p
            coverage_ess_threshold = 0.8 * K
            check_chunk = IBS_WARMUP_FRAME_STRIDE_STEPS
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 自举 TI 种子（pilot 缺失时的兜底）：
            # 冷启动（无 pilot、非续算、非恢复冻结 f_k）时，在 scale=1.0 下采一小批
            # ~IBS_TMBAR_LEARNING_MINIBATCH_FRAMES 帧，用每态平均 softcore 能量把
            # f_k 一步播到 f_k=⟨u_k⟩（去均值）——这是从采样系综测得的零阶 EXP/TI
            # 种子（与 pilot TI 同一约定 f_k≈F_k），把偏置一步放到大致尺寸，避免从
            # f_k=0 靠 2 kT/步慢爬。这段样本只用于播种、不进 tmbar_history，也不计入
            # steps_at_full_bias 预算。重叠好的窗口据此近乎立即变平；低重叠窗口种子
            # 可能过/欠，但随后的 bounded 自适应更新会快速纠偏。
            if (
                not warm.is_resumed_ibs
                and not warm.f_k_warm_started
                and warm.resumed_frozen_f_k is None
            ):
                _boot_target = int(IBS_TMBAR_LEARNING_MINIBATCH_FRAMES)
                wc.sampler.energy_buffer = []
                _boot_guard = 0
                while (
                    len(wc.sampler.energy_buffer) < _boot_target
                    and _boot_guard < _boot_target * 4
                ):
                    guarded_step(wc.sim, check_chunk, f"窗口 {window_idx} 自举 TI 种子采样", on_error=_nan_diagnostics)
                    wc.sampler.collect_energies()
                    _boot_guard += 1
                _boot_u = (
                    np.asarray(wc.sampler.energy_buffer, dtype=np.float64)
                    if wc.sampler.energy_buffer else np.empty((0, K))
                )
                _boot_valid = (
                    ~np.isnan(_boot_u).any(axis=1)
                    if _boot_u.size else np.zeros(0, dtype=bool)
                )
                if _boot_u.size and int(np.sum(_boot_valid)) >= max(5, K + 1):
                    _boot_mean_u = np.mean(_boot_u[_boot_valid], axis=0)
                    _boot_seed = _boot_mean_u - float(np.mean(_boot_mean_u))
                    if np.all(np.isfinite(_boot_seed)) and _boot_seed.shape[0] == K:
                        for k in range(K):
                            wc.sim.context.setParameter(
                                f"{self.prefix}_f_{k}", float(_boot_seed[k])
                            )
                        warm.f_k_warm_started = True
                        wc.sampler.seed_source = "bootstrap"
                        print(
                            f"  [自举 TI 种子] 窗口 {window_idx} 冷启动无 pilot，用首批 "
                            f"{int(np.sum(_boot_valid))} 帧每态平均 softcore 能量播 "
                            f"f_k=⟨u_k⟩（去均值）: "
                            f"{[round(float(x), 2) for x in _boot_seed]} kJ/mol"
                            "（跳过 f_k=0 慢爬）"
                        )
                    else:
                        print(
                            f"  [WARN] [自举 TI 种子] 窗口 {window_idx} 种子非有限/维度不符，"
                            "回退 f_k=0 冷启动"
                        )
                else:
                    print(
                        f"  [WARN] [自举 TI 种子] 窗口 {window_idx} 有效帧不足，回退 f_k=0 冷启动"
                    )
                wc.sampler.energy_buffer = []
                wc.sampler.ema_mean_p = None
            # [IBS_BIAS_PROTOCOL_VERSION=29] 冻结 f_k 后、采 local-MBAR 门数据前的
            # 重新平衡 burn-in。这不是"等 f_k 稳定"（loose-gate 不等 f_k 收敛），
            # 而是让构型忘掉上一段漂移偏置轨迹、在这个刚冻结的固定 Hamiltonian 下
            # 重新平衡；之前 v27 用 20k 步冻结验证阶梯，loose-gate 只需一小段短
            # burn-in（5k 步≈10 ps）即可，随后累计固定-f_k 帧一次性喂 local MBAR。
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=9] steps_at_full_bias 是 SGD 三阶段
            # （learning/freeze_burn_in/validating）和后面 fixed-H overlap 全通过
            # 后的 MBAR 校准验证共用的同一个计数器。如果 SGD 把 max_bias_learning_steps
            # 全部烧光才退出循环，MBAR 校准验证的 while 条件一次都不会执行，
            # calibration_converged 必然是 False——不是校准本身失败，是预算已经
            # 没有了。因此这里把总预算拆成两块：SGD 只能用
            # sgd_step_budget = max_bias_learning_steps - mbar_calibration_reserved_steps，
            # 剩下的 mbar_calibration_reserved_steps 留给 MBAR 校准（下面校准循环
            # 用独立的 calibration_steps_used 计数，不再检查 steps_at_full_bias 是否
            # 撞到 max_bias_learning_steps），保证 fixed-H overlap 全通过时校准验证
            # 一定有机会真正跑完 burn-in + 连续通过判据。
            # non_mutating_v1 永远不会进入下面 legacy fixed-H/MBAR 校准分支，
            # 因而不能白白预留 50k：那正是日志出现 45/50 updates、450k/500k
            # 却提前退出的原因。只有 legacy_mutating 才扣校准预留预算。
            effective_mbar_calibration_reserved_steps = (
                int(mbar_calibration_reserved_steps) if legacy_repair else 0
            )
            frozen_validation_reserved_steps = (
                0 if legacy_repair else int(mbar_calibration_reserved_steps)
            )
            sgd_step_budget = max(
                int(max_bias_learning_steps)
                - int(effective_mbar_calibration_reserved_steps),
                int(warm.frozen_burn_in_steps) + check_chunk,
            )
            # Frozen burn-in/validation is a distinct read-only phase.  Give it
            # its own reserve instead of stealing that reserve from learning;
            # otherwise update 50 can never be frozen and validated.
            _default_cap_steps = (
                int(max_bias_learning_steps)
                + int(warm.frozen_burn_in_steps)
                + int(frozen_validation_reserved_steps)
            )
            # 🔑 [2026-09-11] 预算统一成**累计上限**：可用 = max(0, B − S)。
            # learning / freeze burn-in / frozen validation 分开记账，共受同一个 B。
            _ledger = getattr(wc.sampler, "warmup_budget_ledger", None)
            if not _ledger:
                # 🔑 新采样段（checkpoints/segment_N/）的 ibs_state 从零开始，但
                # warmup 消耗**不得清零** —— 否则每开一个段白送一整份额度
                # （实测 window 4 跨段共烧 540k，而段 2 的账本报"剩 490k"）。
                _ledger = inherit_warmup_ledger_across_segments(
                    self.checkpoint_dir, stage_type, window_idx, _default_cap_steps
                )
                if _ledger is None:
                    _ledger = new_warmup_budget_ledger(_default_cap_steps)
                else:
                    print(
                        f"  [预算] 窗口 {window_idx} 继承上一采样段的 warmup 消耗："
                        f"{warmup_ledger_total_steps(_ledger)} 步已耗、"
                        f"剩 {warmup_ledger_remaining_steps(_ledger)} 步"
                        f"（上限 {_ledger.get('cumulative_cap_steps')}）。"
                        "新段是新的 f_k Epoch，但账本里的总消耗不清零。"
                    )
                wc.sampler.warmup_budget_ledger = _ledger
            _ledger_consumed = warmup_ledger_total_steps(_ledger)
            _stored_cap = int(_ledger.get("cumulative_cap_steps", 0) or 0)
            if _stored_cap <= 0:
                _stored_cap = _default_cap_steps
            elif _default_cap_steps > _stored_cap:
                # 🔑 [2026-09-11 二修] 用户**改 config 抬高 max_bias_warmup_steps**
                # 本身就是"显式升档"，必须生效。原来只认调用方的 per-window rung
                # override，于是已经有账本的窗口永远停在旧上限 —— 实测：config 从
                # 500000 改到 900000（上限 555k→955k）之后，日志仍然打
                # "累计上限 555000"，改动完全没生效。
                # ⚠️ 只升不降：config 调小时用 max() 保住已记录的上限，不追溯缩水
                # （那会把已经花掉的步数变成"超支"）。"不自动升档"针对的是 resolver
                # 按已耗步数自己往上找一档的兜底，不是用户的显式配置。
                _ledger["budget_rung"] = int(_default_cap_steps)
                print(
                    f"  [预算升档] 窗口 {window_idx} 累计上限 {_stored_cap} → "
                    f"{_default_cap_steps}（来自 config 的 max_bias_warmup_steps 等项）。"
                )
                _stored_cap = int(_default_cap_steps)
            if (
                _explicit_frozen_validation_rung is not None
                and int(_explicit_frozen_validation_rung) > _stored_cap
            ):
                # 显式升档：新增可用 = max(0, B₂ − S)。普通 resume 走不到这里。
                _ledger["budget_rung"] = int(_explicit_frozen_validation_rung)
                _stored_cap = int(_explicit_frozen_validation_rung)
            _ledger["cumulative_cap_steps"] = int(_stored_cap)
            full_bias_step_budget = max(0, _stored_cap - _ledger_consumed)
            # 本次 attempt 的分项消耗，循环里按 mode 计，退出时并进账本。
            _attempt_ledger = {k: 0 for k in WARMUP_LEDGER_BUCKETS}
            if not _ledger.get("complete", True):
                print(
                    f"  [WARN] 窗口 {window_idx} 的 warmup 预算账本**不完整**"
                    f"（migrated_from={_ledger.get('migrated_from')}）：旧 checkpoint 只能"
                    " 恢复冻结验证累计步数，learning/burn-in 的历史消耗无持久记录，"
                    "实际已耗可能被低估。不据此升档。"
                )
            if _ledger_consumed:
                print(
                    f"  [预算账本] 累计已耗 {_ledger_consumed} 步"
                    f"（learning={_ledger.get('learning_steps', 0)}, "
                    f"burn_in={_ledger.get('freeze_burn_in_steps', 0)}, "
                    f"validation={_ledger.get('frozen_validation_steps', 0)}），"
                    f"累计上限 {_stored_cap}，本次可用 {full_bias_step_budget} 步。"
                )
            # 🔑 [2026-09-11] 上提到 if 之外：这份冻结 f_k 进入本次 attempt 之前
            # 已累计的验证步数。原来只在 resumed_calibration_pending 分支里定义，
            # 下面"验证预算耗尽、无法判定"的落盘分支对全新窗口也要用它。
            prior_cumulative_steps = int(
                getattr(wc.sampler, "frozen_validation_cumulative_steps", 0)
            )
            if resume_frozen_validation:
                # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 续验已经冻结的 f_k 时，
                # 从一开始就 mode="freeze_burn_in"，永远不会进入 learning，
                # 不需要也不应该套用"给 learning 大部分预算、只留
                # mbar_calibration_reserved_steps 给校准验证"的拆分——这里的
                # 预算是 frozen_validation_step_overrides 给这个窗口的累计
                # 目标步数（50k→150k→300k 阶梯，见调用方
                # abfe_pipeline.py::_run_stage_with_overlap_autorepair）。
                # 🔑 [修复：累计预算记的是"目标总量"，不是"这一轮要新跑多少"]
                # effective_frozen_validation_budget 是"这份校准 f_k 从第一次
                # 冻结验证开始、累计应该验证到的总步数"，不是"这次 attempt 单独
                # 要跑的步数"——之前这里直接把它当成 sgd_step_budget，导致
                # 50k/150k/300k 三档实际总步数变成 50k+150k+300k=500k，而不是
                # 阶梯设计意图的"累计延长到 300k"。现在扣掉
                # sampler.frozen_validation_cumulative_steps（之前所有 attempt
                # 已经真正花在这份冻结 f_k 验证上的步数，跨 resume 持久化），
                # 这一轮只跑差值——如果从主窗口 checkpoint 续算（见
                # MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION），差值步数接的就是
                # 上次中断的真实动力学轨迹，不是重新烧一遍。
                # 🔑 [2026-09-11] 剩余预算现在由账本统一给出（上面已算好
                # full_bias_step_budget = max(0, B − S)），这里不再自己减一次——
                # 原来那份只看 frozen_validation_cumulative_steps，learning/burn-in
                # 的消耗完全不在账内。sgd_step_budget 在续验路径上不会被用到
                # （永远不进 learning），保持一个不小于 burn-in 的值即可。
                sgd_step_budget = max(
                    int(full_bias_step_budget),
                    int(warm.frozen_burn_in_steps) + check_chunk,
                )
                frozen_validation_reserved_steps = 0
            validation_attempt_budget_steps = (
                int(full_bias_step_budget)
                if resume_frozen_validation
                else max(
                    int(frozen_validation_reserved_steps),
                    int(required_consecutive_bias_updates) * 20 * int(check_chunk),
                )
            )
            # 🔑🔑 [S2-B，2026-09-13] **完整性要求与预算解耦。**
            #
            # 原式是 `max(统计目标, validation_attempt_budget_steps / check_chunk)`，
            # 而续验路径上 `validation_attempt_budget_steps = full_bias_step_budget`
            # ⟹ **给的预算越多、要求的帧数越高**（实测 max(200, 515000//250) = 2061，
            # 是统计目标的 10 倍）。那不是一个"要求"，那是把预算改名叫要求。
            #
            # 它现在**只进报告、不当门**（本仓自第一个 commit 起就是如此），所以修它
            # 不改变任何判定；修的理由是这个数**会骗读它的人**：可达性预检的 T 一度
            # 就被错取成它，把 gcrit 算小 20 倍、把只差 26% 帧数的 win4 判成
            # "差 7.5 倍不可达"（见 docs/archive/STAGE2_AUTONOMOUS_LOOP_STATUS_2026-09-11.md
            # §9）。一个随预算浮动的数还会让两次 run 的报告没法横向比。
            #
            # ⚠️ 与可达性预检的 T（`IBS_LOCAL_MBAR_GATE_MIN_FRAMES`=10，**去相关**
            # 帧数下限）仍然是**两个量**，别再合并：这里是"这次 attempt 攒够**原始**
            # 帧没有"的完整性目标，那里是去相关之后还剩几帧。
            minimum_complete_validation_frames = (
                int(required_consecutive_bias_updates) * 20
            )

            # ========================================================================
            # 🔑🔑 [2026-09-14 / 阶梯空转] **增量续采时跳过那次注定被丢弃的冻结复验。**
            #
            # 现场：补帧目标每涨 250k，窗口级 resume 门就判"视为无效缓存"，于是
            # EM(20000 步) + 阶段 2 dt 测试 + Boresch 爬坡 + 冻结 burn-in + 只读复验
            # 整条前置链重跑一遍，只换来 250k 帧；而紧接着
            # `_try_load_main_window_checkpoint(sim, production_ckpt_path)` 又把
            # Context 的坐标/速度/盒子/积分器 RNG **全部覆盖**回上次生产结束那一刻
            # ⟹ 这条链产出的动力学**逐字被丢弃**。帧没丢，丢的是 GPU 时间。
            #
            # **被拆掉的保护只有一条**（IBS_BIAS_PROTOCOL_VERSION=7 立的）：
            # "缓存的 `bias_converged=True` 不能证明本次新建 Context 的**当前构型**
            # 已经在这个固定偏置下平衡过"。那条规矩针对的是 **EM 出来的全新构型**；
            # 而 top-up 恢复的是上一段生产**结束那一刻**的坐标/速度/盒子/RNG ——
            # 本身就是在同一份冻结 f_k 下采出来的、且已经通过过一次冻结验证，
            # 根本没有"新构型待平衡"这回事。
            #
            # 守卫三条，缺一不可；checkpoint 因任何原因加载不上 ⟹
            # `restored_from_production_checkpoint` 为 False ⟹ 退回今天的完整
            # burn-in + 复验，一步不少：
            #   (1) topup_only                         —— resume 门判"身份全对、只差步数"
            #   (2) skip_warmup_entirely               —— f_k 已收敛、不重新学习
            #   (3) restored_from_production_checkpoint —— 生产 checkpoint 真的装进
            #                                             了 Context（不是"应该能装"）
            # 下面 18xxx 处那段原样保留：再调一次 `_production_window_checkpoint_is_usable`
            # 只是纯 JSON 比较，再调一次 `_try_load_main_window_checkpoint` 是把同一份
            # 状态重新写进同一个 Context，幂等。
            if topup_only and skip_warmup_entirely and warm.resumed_frozen_f_k is not None:
                try:
                    _topup_target_steps = (
                        int(production_step_overrides[window_idx])
                        if production_step_overrides and window_idx in production_step_overrides
                        else int(n_steps_per_window)
                    )
                    _topup_manifest = _build_production_window_checkpoint_manifest(
                        stage_type, window_idx, len(wc.lc_win), wc.win_sys_xml, wc.lc_win, wc.lv_win,
                        (float(np.mean(wc.lv_win))
                         if _system_has_global_parameter(wc.win_sys, "lambda_shield") else None),
                        (1.0 if _has_valid_boresch_restraint(self.boresch) else None),
                        [float(x) for x in warm.resumed_frozen_f_k],
                        self.temperature.value_in_unit(unit.kelvin),
                        self.platform_name,
                        repair_policy=repair_policy,
                        coion_identity=self.coion_identity,
                    )
                    _topup_manifest["stage_protocol_key"] = getattr(
                        self, "stage_protocol_key", None
                    )
                    _topup_ckpt_path, _ = _production_window_checkpoint_paths(
                        self.checkpoint_dir, stage_type, window_idx
                    )
                    _topup_conv_path = os.path.join(
                        self.output_dir,
                        f"dual_window_{window_idx}_{stage_type}_convergence.json",
                    )
                    _topup_prior_steps = 0
                    if _production_window_checkpoint_is_usable(
                        self.checkpoint_dir, stage_type, window_idx, _topup_manifest
                    ):
                        with open(_topup_conv_path, "r", encoding="utf-8") as _f:
                            _topup_prior_steps = int(
                                json.load(_f).get("cumulative_production_steps", 0)
                            )
                        if 0 < _topup_prior_steps < _topup_target_steps:
                            warm.restored_from_production_checkpoint = (
                                _try_load_main_window_checkpoint(wc.sim, _topup_ckpt_path)
                            )
                except Exception as _topup_err:  # noqa: BLE001 —— 任何差错都退回完整路径
                    print(
                        f"  [WARN] 窗口 {window_idx} 增量续采预检失败（{_topup_err!r}），"
                        "退回完整 burn-in + 冻结复验。"
                    )
                    warm.restored_from_production_checkpoint = False
            skip_frozen_revalidation = bool(
                topup_only and skip_warmup_entirely and warm.restored_from_production_checkpoint
            )
            if skip_frozen_revalidation:
                warm.bias_converged = True
                warm.frozen_f_k_snapshot = warm.resumed_frozen_f_k
                print(
                    f"  窗口 {window_idx} 增量续采（已采 {_topup_prior_steps} 步 / 目标 "
                    f"{_topup_target_steps} 步）：Context 已恢复到上次生产结束那一刻，"
                    "跳过这次注定被覆盖掉的冻结 burn-in + 只读复验。"
                )
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] True 当且仅当这个窗口是靠"见好
            # 就收"兜底（达到重试上限、采用最优 attempt）而不是严格
            # lse_log_residual_tolerance 判据放行进生产的——见 version 25
            # changelog (a)，写入 bias_warmup_diag/convergence.json 供事后追溯。
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] 实现 PROPOSAL_frozen_validation_
            # fallback.md：learning 阶段候选连续通过次数从未凑满、纯粹因为
            # max_bias_updates 耗尽时，给当前 f_k 一次真正的冻结验证机会
            # （而不是直接 f_not_converged），guard 确保每个窗口 attempt 只
            # 触发一次——第二次再撞到同样的耗尽条件说明这次真的没戏，仍然
            # 老实 break。
            # [Candidate-first, Validate-or-Learn v1] validate_direct_retry_pending
            # limits the damped+pairwise-capped VALIDATE-failure retry (see the
            # gate-failure router below) to exactly one direct re-validation per
            # freeze cycle -- a second consecutive gap failure falls back to
            # bounded-occupancy LEARN instead of chaining retries indefinitely.
            # ever_completed_a_validate_attempt/last_failure_reason distinguish a
            # real, completed local-MBAR gate failure (evidence against this f_k,
            # -> terminal "failed" on exhaustion) from "never got that far"
            # (still eligible for another resume, -> "unconverged").
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12 修复，未升版本号——见下方修复处注释]
            # 仅用于诊断的计数器：resumed_calibration_pending 续验分支里每次单批
            # 验证失败都会 +1，但绝不触发退回 learning。跟 learning_to_validation_cycles
            # 分开计数，因为后者的语义是"退回 learning 的次数"，续验分支永远不退回
            # learning，必须让它保持 0，否则会污染下面失败打印/诊断里"是否发生过
            # 假收敛重学习"的含义。
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] "见好就收"兜底状态：记录目前
            # 观察到的、max_abs_log_residual 最小的那次完整冻结验证 attempt
            # 的 f_k/统计信息。达到 max_frozen_validation_cycles_before_
            # accept_best 次失败后，直接采用这份最优结果而非继续无限重试——
            # 见 version 25 changelog (a)。
            validation_probability_sum = np.zeros(K, dtype=np.float64)
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 局部滑窗 MBAR loose-gate 状态：
            # frozen_mbar_batches 累计当前冻结 f_k 下最近若干批固定-f_k minibatch
            # （每条是 tmbar_history 里的 {u_kn,bias_energies,base_energies}）；攒满
            # IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES 批就拼一次 local MBAR 判一次门。
            # updates_since_freeze/have_frozen_once 控制"首次冻结前累计 min_bias_
            # updates 次更新、此后每次门失败只需再更新一轮就重新冻结复检"。
            # [2026-09-12] 同一候选、连续数据上每次「样本不足」时的 g。
            # **只记录不判定** —— 提前外推判死在 2026-09-11 已被否决（见下方
            # `不做"没救了"的提前外推` 那段）。可达性判定在控制器做，
            # 而且只在**周期用尽之后**决定要不要开下一个周期。
            # 非 None 表示本窗口以"验证预算耗尽、无法判定"退出（见下方分级求解）。
            # [2026-09-15] 调用方是否授权这个窗口在撞批次上限时进**临时生产**
            # （见 `provisional_production_windows` 的形参说明）。
            provisional_authorized = int(window_idx) in {
                int(x) for x in (provisional_production_windows or ())
            }
            # 本窗口这次**实际**是以临时生产进的生产（授权 + 真的撞了上限）。
            if warm.resumed_frozen_f_k is not None:
                # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 从主窗口 checkpoint
                # 续算时不需要 freeze_burn_in：checkpoint 保存的那一刻本身就已经
                # 在 validating（或刚做完一次 freeze_burn_in）里，冻结 f_k 全程
                # 没变，没有需要"忘记"的漂移历史，直接继续采下一批全新样本即可。
                # 没有可用 checkpoint 时（缺失/不兼容/首次校准刚失败一次）保持
                # 原有行为：先走一次 freeze_burn_in 再进入 validating。
                warm.mode = "validating" if restored_from_window_checkpoint else "freeze_burn_in"
                # 🔑 [2026-09-11] 上次是"验证预算耗尽、无法判定"且 Context 完整恢复
                # 时，接着上次的批次继续累计（批次数据本身在 tmbar_history 里，f_k
                # 全程没变、Hamiltonian 也没变，这些帧仍是同一分布的合法样本）。
                # 不恢复的话 resume 等于重新领一份批次额度。
                if (
                    restored_from_window_checkpoint
                    and getattr(wc.sampler, "bias_status", "unconverged")
                    == "frozen_validation_indeterminate"
                    and int(getattr(wc.sampler, "frozen_validation_batches_done", 0)) > 0
                    and getattr(wc.sampler, "frozen_candidate_fingerprint", None)
                    == frozen_candidate_fingerprint(wc.sampler.frozen_f_k_pending)
                ):
                    _n_done = min(
                        int(wc.sampler.frozen_validation_batches_done),
                        len(wc.sampler.tmbar_history),
                        IBS_LOCAL_MBAR_GATE_MAX_BATCHES,
                    )
                    warm.frozen_mbar_batches = list(wc.sampler.tmbar_history[-_n_done:])
                    warm.validation_batch_count = _n_done
                    warm.validation_insufficient_attempts = max(
                        1, _n_done // IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
                    )
                    warm.validation_solve_at_batches = min(
                        IBS_LOCAL_MBAR_GATE_MAX_BATCHES,
                        _n_done + IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES,
                    )
                    print(
                        f"  恢复上次已累计的 {_n_done} 批固定-f_k 验证数据，"
                        f"下一次求解在第 {warm.validation_solve_at_batches} 批"
                        f"（单周期上限 {IBS_LOCAL_MBAR_GATE_MAX_BATCHES} 批）。"
                    )
                # 🔑 [2026-09-14 / S1] 候选进入冻结验证的**第 1 个入口**（resume：
                # 接着验上次那份冻结 f_k）。三个入口都要过这道门，否则"永不续验"
                # 只剩 relearn_epoch_used 的"每窗再试一次"计数。
                assert_candidate_not_sealed(
                    warm.resumed_frozen_f_k, checkpoint_dir=self.checkpoint_dir,
                    window_idx=window_idx, lambdas_vdw=wc.lv_win, stage_type=stage_type,
                )
                warm.frozen_f_k_snapshot = warm.resumed_frozen_f_k
                wc.sampler.energy_buffer = []
                wc.sampler.ema_mean_p = None
                # 恢复的是一份已经冻结过的 f_k，直接进 freeze/validate。若这次 loose
                # gate 未过退回 learning，则和普通窗口一样由 readiness 门决定何时重新
                # 冻结（占据重新平坦、步长≤1 kT、连续满足），不会"回来一轮就立刻重冻"。
                warm.have_frozen_once = True
            else:
                warm.mode = "learning"
                warm.frozen_f_k_snapshot = None

            # 🔑 [性能计时] warmup/learning 控制面没有 guard、也没有周期性
            # checkpoint（见下方循环体），桶比生产循环少；同样默认常开，循环
            # 结束时打印一次汇总，作为"这一段到底花在积分/CV-probe/权重更新
            # 上多少时间"的实测证据。
            # [2026-09-14] `skip_frozen_revalidation` 为真时这整段不跑：见上面
            # `bias_converged = True` 处的长注释（守卫三条 + 退回路径）。
            self._run_warmup_loop(
                K=K,
                _attempt_ledger=_attempt_ledger,
                _nan_diagnostics=_nan_diagnostics,
                check_chunk=check_chunk,
                expected_main_manifest=expected_main_manifest,
                full_bias_step_budget=full_bias_step_budget,
                ibs_state_file=ibs_state_file,
                k=k,
            wc=wc,
                main_ckpt_path=main_ckpt_path,
                main_manifest_path=main_manifest_path,
                max_bias_updates=max_bias_updates,
                prior_cumulative_steps=prior_cumulative_steps,
                provisional_authorized=provisional_authorized,
                resumed_calibration_pending=resumed_calibration_pending,
                skip_frozen_revalidation=skip_frozen_revalidation,
                skip_warmup_entirely=skip_warmup_entirely,
                stage_type=stage_type,
                start=start,
                warm=warm,
                window_idx=window_idx,
            )

            # 🔑 [性能计时] warmup/learning 循环结束，打印一次耗时拆分汇总——
            # 只是诊断输出，不落盘、不参与任何 resume/协议比较。
            self._summarize_warmup(
                K=K,
                _attempt_ledger=_attempt_ledger,
                _ledger=_ledger,
                _nan_diagnostics=_nan_diagnostics,
                accept_recalibrated_f_k_without_gate=accept_recalibrated_f_k_without_gate,
                allow_best_effort_warmup=allow_best_effort_warmup,
                check_chunk=check_chunk,
                coverage_ess_threshold=coverage_ess_threshold,
                effective_frozen_validation_budget=effective_frozen_validation_budget,
                effective_mbar_calibration_reserved_steps=effective_mbar_calibration_reserved_steps,
                end=end,
                expected_main_manifest=expected_main_manifest,
                frozen_validation_reserved_steps=frozen_validation_reserved_steps,
                full_bias_step_budget=full_bias_step_budget,
                ibs_state_file=ibs_state_file,
                ibs_wrap=ibs_wrap,
                k=k,
            wc=wc,
                lse_log_residual_tolerance=lse_log_residual_tolerance,
                main_ckpt_path=main_ckpt_path,
                main_manifest_path=main_manifest_path,
                max_bias_updates=max_bias_updates,
                min_bias_updates=min_bias_updates,
                min_probability_threshold=min_probability_threshold,
                minimum_complete_validation_frames=minimum_complete_validation_frames,
                required_consecutive_bias_updates=required_consecutive_bias_updates,
                resume_frozen_validation=resume_frozen_validation,
                resumed_calibration_pending=resumed_calibration_pending,
                sgd_step_budget=sgd_step_budget,
                stage_type=stage_type,
                start=start,
                target_p=target_p,
                validation_attempt_budget_steps=validation_attempt_budget_steps,
                warm=warm,
                warmup_only=warmup_only,
                window_idx=window_idx,
            )

            self._commit_frozen_f_k(
                effective_frozen_validation_budget=effective_frozen_validation_budget,
                ema_mean_p_values=warm.ema_mean_p_values,
                expected_main_manifest=expected_main_manifest,
                frozen_validation_is_final_rung=frozen_validation_is_final_rung,
                frozen_validation_reserved_steps=frozen_validation_reserved_steps,
                full_bias_step_budget=full_bias_step_budget,
                ibs_state_file=ibs_state_file,
            wc=wc,
                main_ckpt_path=main_ckpt_path,
                main_manifest_path=main_manifest_path,
                max_bias_updates=max_bias_updates,
                mode=warm.mode,
                prior_cumulative_steps=prior_cumulative_steps,
                resumed_calibration_pending=resumed_calibration_pending,
                sgd_step_budget=sgd_step_budget,
                skip_frozen_revalidation=skip_frozen_revalidation,
                stage_type=stage_type,
                warm=warm,
                warmup_only=warmup_only,
                window_idx=window_idx,
            )

            # 恢复步长
            wc.sim.integrator.setStepSize(original_dt)

            # 打印预热后的 f_k 值
            f_vals = [wc.sim.context.getParameter(f"{self.prefix}_f_{k}") for k in range(len(wc.lc_win))]
            print(
                "  [OK] 偏置预热结束（f_k 已锁定；严格通过/预算终止类型见诊断）: "
                f"{[f'{v:.1f}' for v in f_vals]}"
            )
            if max(f_vals) - min(f_vals) < 1.0:
                print(f"  [WARN] 警告: f_k 仍未明显分化，可能需延长预热或检查窗口重叠度。")
            if debug_mode:
                diagnose_softcore_cv_values(wc.sim.context, ibs_wrap, wc.lc_win, wc.lv_win, prefix=f"窗口{window_idx}_偏置预热后", sampler=wc.sampler)
            raw_probe = wc.sampler.get_raw_interaction_energies()
            if len(raw_probe) >= 2:
                raw_span = float(np.max(raw_probe) - np.min(raw_probe))
                if raw_span < 1e-6 and np.max(np.abs(raw_probe)) < 1e-6:
                    raise RuntimeError(
                        f"窗口 {window_idx} 的 IBS 软核 CV 在偏置预热后仍全部为 0。"
                        "已阻止进入生产采样，请继续检查 VDW CV 构造或状态注入。"
                    )

            if warmup_only:
                warm.bias_warmup_diag["warmup_only"] = True
                warm.bias_warmup_diag["frozen_f_k_kJ_mol"] = [float(v) for v in f_vals]
                warmup_results.append(dict(warm.bias_warmup_diag))
                if getattr(wc.sampler, "_probe_context", None) is not None:
                    del wc.sampler._probe_context
                    wc.sampler._probe_context = None
                if getattr(wc.sampler, "_probe_integrator", None) is not None:
                    del wc.sampler._probe_integrator
                    wc.sampler._probe_integrator = None
                del wc.sampler
                del wc.sim.context
                # 这里不用 `del sim`：嵌套的 `_chunk_is_finite()` 闭包引用 sim，而 ruff 的
                # 延迟作用域模型看到的是外层结束时 sim 已被 del ⟹ 报 F821。那是误报
                # （两处 del 都在紧跟 continue / 循环结束的清理路径上，闭包早已用完），
                # 但 CI 的 ruff 门会因此变红。`= None` 的引用计数效果与 del 相同。
                wc.sim = None
                # 🔑 [2026-09-10] 与正常收尾路径同一处修复，理由见那里的长注释：
                # 释放 System 之前先断掉 `ibs_wrap.force` 与 `_nan_diagnostics`
                # 闭包这两条指向它内部 Force 的引用，否则代理会悬垂到下一个窗口，
                # 而且 System 根本不在这里析构。
                ibs_wrap = None
                _nan_diagnostics = None
                wc.win_sys = None
                gc.collect()
                continue

            # ---- 进入生产采样：不再改变 bias_scale/最小化/重新爬坡 ----
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=7] 旧版本这里有"生产前卸压"：bias_scale
            # 清零→最小化→5000 步无偏置运行→重新爬回 1.0，紧接着直接进 production，
            # 从未对这个被重新扰动过的构型重新验证过覆盖度——哪怕前面的收敛判据是
            # 真的，这一步也会在没有复检的情况下把刚验证过的分布毁掉。现在冻结
            # 验证已经在完全固定的 production Hamiltonian（bias_scale=1.0、f_k 不再
            # 变化）下完成，不需要也不允许再碰 bias_scale/最小化/爬坡：只做一次
            # 非破坏性的安全力检查（不达标就直接报错，不再用更多扰动去"修复"一个
            # 刚验证通过的构型），然后原样进入生产采样。

            # 🔑 [2026-08-27，见 docs/experiments/EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md 补验收]
            # 生产入口标记（production_entry_f_k）不再在这里赋值——这里是
            # checkpoint 续采判断/恢复**之前**，早于真正确定这次生产是否续算
            # 同一份冻结 f_k。真正落盘的赋值挪到下面 production_f_k_lock 断言
            # 通过之后（[阶段5] 打印之前），那才是"checkpoint 恢复与入口调整都
            # 已完成、第一步生产采样之前"的正确时刻；同时也在那里补上写入
            # ibs_state_file 的 save_ibs_state() 调用，否则这份标记只存在于
            # 内存里，续跑和下游冻结检查都读不到它。

            self._produce_window(
                K=K,
                debug_mode=debug_mode,
            es=es,
                end=end,
                ibs_state_file=ibs_state_file,
                ibs_wrap=ibs_wrap,
            wc=wc,
                lse_log_residual_tolerance=lse_log_residual_tolerance,
                n_steps_per_window=n_steps_per_window,
                production_step_overrides=production_step_overrides,
                repair_policy=repair_policy,
                skip_frozen_revalidation=skip_frozen_revalidation,
                stage_type=stage_type,
                start=start,
                steps_per_update=steps_per_update,
                warm=warm,
                window_idx=window_idx,
            )

            # 清理
            if getattr(wc.sampler, "_probe_context", None) is not None:
                del wc.sampler._probe_context
                wc.sampler._probe_context = None
            if getattr(wc.sampler, "_probe_integrator", None) is not None:
                del wc.sampler._probe_integrator
                wc.sampler._probe_integrator = None
            del wc.sampler
            del wc.sim.context
            wc.sim = None  # 同 warmup_only 分支：不用 del，理由见那里的注释
            # 🔑 [2026-09-10] 释放 System **之前**必须先断掉所有指向它内部 Force 的
            # Python 代理，否则那些代理会悬垂到下一个窗口。
            #
            # 原来这里只有 `win_sys = None`，漏了两样：
            #
            #   1. `ibs_wrap` 持有 `self.force` —— 那个 `CustomCVForce` 是
            #      `new_sys.addForce()` 进去的，所有权归 C++ System。System 一析构，
            #      它的 Force 全被删掉，而 `ibs_wrap.force` 这个 SWIG 代理还指着
            #      已释放的内存。`ibs_wrap` 要到下一轮循环开头
            #      `win_sys, ibs_wrap = self._build_window_system(...)` 才被重新绑定，
            #      也就是说它**带着悬垂代理活过了整个窗口切换**。
            #   2. `_nan_diagnostics` 闭包同样引用 `win_sys`（上一行注释自己写了
            #      这句，却只是因此改用 `= None` 而没有解开引用）。所以
            #      `win_sys = None` 在这里**什么都没释放**：System 一直活到下一个
            #      窗口重新定义那个闭包时才真正析构 —— 正好落在窗口 1 建系/序列化
            #      的当口，而彼时上面那个悬垂代理还在。
            #
            # 2026-09-10 的 SEGV 时序与此逐段吻合：窗口 0 完整跑完 → 收尾（并未真的
            # 释放）→ 窗口 1 一开始建系就死在 `XmlSerializer.serialize`。
            #
            # 顺序不能换：先断代理（ibs_wrap / 闭包），最后才是 System 本身。
            ibs_wrap = None
            _nan_diagnostics = None
            wc.win_sys = None
            gc.collect()

        print(f"\n{'='*80}")
        print(f"[OK] 所有窗口采样完成")
        print(f"{'='*80}")
        return warmup_results if warmup_only else None


    def _gradual_warmup_debug(self, sim, ibs_wrap, sampler, warmup_steps, win_sys, window_idx, debug_mode=True):
        """步长爬坡（不含偏置力）。

        ✅ 性能/收敛修复：这里以前还带一段独立的"纯偏置预热"（bias_scale 0→1，
        预算 warmup_steps，默认高达 50 万步），而调用方紧接着又会做一次完整的
        清零重爬（见 IBSWindowManagerDualLambda 里的"[偏置预热]"段）。两段都在
        往同一个 f_k 上叠加更新，前一段辛苦爬升的 bias_scale 会被后一段立刻推回
        0 重来——除了浪费 GPU，还会把 IBSSampler.update_weights() 里随调用次数
        衰减的学习率 (eta_sgd) 提前消耗在一个马上被丢弃的目标上。现在只保留物理
        上真正必要的部分（时间步长爬坡），偏置力的唯一平滑引入通道移到调用方
        那一段，避免同一件事做两遍。warmup_steps 参数保留仅为向后兼容签名，
        当前不再被本函数使用。
        """
        print("  渐进预热：时间步长爬坡...")

        # 偏置力在时间步长爬坡期间保持关闭，避免爬 dt 的同时还叠加一个尚未
        # 校准的偏置力，两个不稳定源混在一起更难诊断。
        wc.sim.context.setParameter(f"{self.prefix}_bias_scale", 0.0)

        original_dt = wc.sim.integrator.getStepSize()
        for dt_ps in [0.0005, 0.001, 0.002]:
            wc.sim.integrator.setStepSize(dt_ps * unit.picoseconds)
            guarded_step(wc.sim, 5000, f"dt 扫描 {dt_ps} ps")

            if debug_mode:
                state = wc.sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/openmm.unit.nanometer)
                max_f = np.max(np.linalg.norm(forces, axis=1))
                print(f"    [爬坡 dt={dt_ps}ps] E={e:.1f}, max|F|={max_f:.1f}")

        wc.sim.integrator.setStepSize(original_dt)
        print("  [OK] 时间步长爬坡完成（偏置力将在下一步统一引入）")


    def _safe_boresch_ramp(self, sim, final_scale=1.0, n_steps=200):
        """
        渐进式提升 Boresch 力强度，每步后用极短时间步长松弛并检测能量。
        返回 True 表示爬坡成功，False 表示在中途检测到异常。

        🚨 [0831issue P2] **当前是死代码（全仓库无调用点），且原来的灾难判据是错的。**
        原判据 `abs(e) > 1e5` 拿的是**全体系总势能**：7 万原子的溶剂/膜盒本身就在
        −5e5 ~ −1e6 kJ/mol 量级，所以它对任何正常体系都会在第一步就"检测到几何奇点"
        并返回 False。一旦有人把这个函数重新接线，它会把全部正常运行判成失败。
        已改为判**爬坡过程中的能量增量** ΔE（相对爬坡起点），并保留 NaN/Inf 检查：
        Boresch 力增强导致的真实几何奇点表现为势能急剧**上升**，与盒子绝对能标无关。
        """
        try:
            current_scale = wc.sim.context.getParameter("lambda_boresch_scale")
        except Exception:
            logger.warning(
                "读取 lambda_boresch_scale 失败，回退到默认初始值 0.01",
                exc_info=True,
            )
            current_scale = 0.01
        n_ramp = 10
        scales = np.linspace(current_scale, final_scale, n_ramp + 1)[1:]

        original_dt = wc.sim.integrator.getStepSize()
        wc.sim.integrator.setStepSize(0.001 * unit.picoseconds)

        # [0831issue P2] 判据的参考点：爬坡**开始前**的总势能。灾难看 ΔE 不看 |E|。
        e_ref = (
            wc.sim.context.getState(getEnergy=True)
            .getPotentialEnergy()
            .value_in_unit(unit.kilojoule_per_mole)
        )
        if not np.isfinite(e_ref):
            print(f"  [ERR] Boresch 爬坡起点势能非有限（{e_ref}），拒绝爬坡")
            wc.sim.integrator.setStepSize(original_dt)
            return False
        # Boresch 力全强度下三个二面角全反转的量级是 ~1200 kJ/mol（见
        # BoreschAttachmentREMDManager 的 docstring），所以真实几何奇点的 ΔE 远超
        # 这个阈值；正常爬坡的 ΔE 只是限制力本身的势能，量级小得多。
        max_delta_e = 1.0e5

        for s in scales:
            wc.sim.context.setParameter("lambda_boresch_scale", float(s))
            guarded_step(wc.sim, n_steps, f"Boresch scale 扫描 s={s}")

            state = wc.sim.context.getState(getEnergy=True)
            e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            delta_e = e - e_ref
            if not np.isfinite(e) or delta_e > max_delta_e:
                print(
                    f"  [ERR] Boresch 爬坡中断：scale={s:.3f}，势能={e:.1f} kJ/mol"
                    f"（ΔE={delta_e:+.1f} > {max_delta_e:.0f}），可能几何奇点"
                )
                wc.sim.integrator.setStepSize(original_dt)
                return False
            print(
                f"  ↪ Boresch scale → {s:.3f}，当前势能={e:.1f} kJ/mol"
                f"（ΔE={delta_e:+.1f}）"
            )

        wc.sim.integrator.setStepSize(original_dt)
        print("  [OK] Boresch 限制力已完全启用")
        return True

    def get_stage_data_for_analysis(
        self, stage_type: str = "coul", *, excluded_local_windows: Optional[set] = None,
    ) -> List[Dict]:
        return load_ibs_window_outputs_from_dir(
            self.output_dir, self.ranges, self.lambdas_coul, self.lambdas_vdw,
            checkpoint_dir=self.checkpoint_dir, stage_type=stage_type,
            excluded_local_windows=excluded_local_windows,
            current_sampling_score_sha256=self.sampling_score_sha256,
        )



    def _produce_window(
        self,
        *,
        K,
        debug_mode,
        es,
        end,
        ibs_state_file,
        ibs_wrap,
        wc,
        lse_log_residual_tolerance,
        n_steps_per_window,
        production_step_overrides,
        repair_policy,
        skip_frozen_revalidation,
        stage_type,
        start,
        steps_per_update,
        warm,
        window_idx,
    ):
        """相位 6-8（生产准备/生产循环/收口）—— 刀1 A 段自 run_all_windows 原样搬出。

        纯搬运（verify_extract --block for --tail 82 逐节点等价）；跨界变量按原名
        作 keyword-only 形参。窗口级控制流与引用释放协议（if warmup_only 块、
        del sampler / sim = None / …）留在 run_all_windows 调用点
        （CUTTING_PLAN.md §2.2/§2.3，实测修正见本文件头注释）。
        """
        # 🔑 提前到这里计算（原来在安全检查/生产采样断言之后才算）：下面的
        # production checkpoint 续采检测需要提前知道本次目标步数，才能判断
        # "已有的累计步数是否还没到目标"。
        prod = ProductionState()
        prod.effective_n_steps_per_window = (
            int(production_step_overrides[window_idx])
            if production_step_overrides and window_idx in production_step_overrides
            else int(n_steps_per_window)
        )
        if prod.effective_n_steps_per_window != int(n_steps_per_window):
            print(
                f"  窗口 {window_idx} 使用步数覆盖：{prod.effective_n_steps_per_window} 步"
                f"（默认 {n_steps_per_window} 步），来自 production_step_overrides"
            )

        # 🔑 [production checkpoint 续采] 之前 production ESS 低触发
        # reseed_resample 时，是整窗删除 energies/bias/base/convergence.json
        # 后从这里重新最小化+dt测试+Boresch爬坡+冻结重验证+从零步 production
        # 重采一遍（"250k 延长到 500k"实际是"扔掉 250k、重跑独立的 500k"，
        # 不是真续算）。这里检查是否存在 λ/窗口/系统/冻结 f_k 完全一致的
        # production checkpoint，且 energies/bias/base/convergence.json 仍在
        # （没有被 _invalidate_single_window_production 清掉——f_k 真的变了
        # 时它会被清掉，这里自然检测不到，回退到今天的完整重采），若有且
        # 累计步数小于本次目标，就在下面的安全检查之前先恢复到上次结束的
        # 坐标/速度/积分器 RNG 状态，让安全检查/production_pos_backup/生产
        # 采样本身都在这份续算状态上进行；已有的能量/偏置/基准能量历史在
        # 下面 sampler.energy_history 等重置之后再追加读回。
        production_frozen_f_k = [
            float(wc.sim.context.getParameter(f"{self.prefix}_f_{k}")) for k in range(len(wc.lc_win))
        ]
        # 复用窗口开头算好的那份（见那里的说明）：同一个 win_sys，中间未被修改，
        # 字节逐字相同。原来这里再序列化一次，SEGV 就落在这一调用上。
        win_sys_xml_for_prod_ckpt = wc.win_sys_xml
        prod.expected_production_manifest = _build_production_window_checkpoint_manifest(
            stage_type, window_idx, len(wc.lc_win), win_sys_xml_for_prod_ckpt, wc.lc_win, wc.lv_win,
            (float(np.mean(wc.lv_win)) if _system_has_global_parameter(wc.win_sys, "lambda_shield") else None),
            (1.0 if _has_valid_boresch_restraint(self.boresch) else None),
            production_frozen_f_k,
            self.temperature.value_in_unit(unit.kelvin),
            self.platform_name,
            repair_policy=repair_policy,
            coion_identity=self.coion_identity,
        )
        prod.expected_production_manifest["stage_protocol_key"] = getattr(self, "stage_protocol_key", None)
        prod.production_energies_path = os.path.join(
            self.output_dir, f"dual_window_{window_idx}_{stage_type}_energies.npy"
        )
        prod.production_bias_path = os.path.join(
            self.output_dir, f"dual_window_{window_idx}_{stage_type}_bias.npy"
        )
        prod.production_base_path = os.path.join(
            self.output_dir, f"dual_window_{window_idx}_{stage_type}_base.npy"
        )
        prod.production_conv_path = os.path.join(
            self.output_dir, f"dual_window_{window_idx}_{stage_type}_convergence.json"
        )
        prod.production_ckpt_path, prod.production_manifest_path = _production_window_checkpoint_paths(
            self.checkpoint_dir, stage_type, window_idx
        )
        prod.production_sampling_states_path = os.path.join(
            self.output_dir, f"dual_window_{window_idx}_{stage_type}_sampling_states.npy"
        )
        prod.production_residual_basis_path = os.path.join(
            self.output_dir, f"dual_window_{window_idx}_{stage_type}_residual_basis.npy"
        )
        prod.prior_cumulative_production_steps = 0
        resumed_production_checkpoint = False
        prior_energy_history = None
        prior_bias_history = None
        prior_sampling_state_history = None
        prior_residual_basis_history = None
        prior_base_energy_history = None
        if (
            os.path.exists(prod.production_energies_path)
            and os.path.exists(prod.production_conv_path)
            and _production_window_checkpoint_is_usable(
                self.checkpoint_dir, stage_type, window_idx, prod.expected_production_manifest
            )
        ):
            try:
                with open(prod.production_conv_path, "r", encoding="utf-8") as f:
                    prior_production_convergence = json.load(f)
                prod.prior_cumulative_production_steps = int(
                    prior_production_convergence.get("cumulative_production_steps", 0)
                )
            except Exception:
                prod.prior_cumulative_production_steps = 0
            if 0 < prod.prior_cumulative_production_steps < prod.effective_n_steps_per_window:
                if _try_load_main_window_checkpoint(wc.sim, prod.production_ckpt_path):
                    try:
                        _prior_e, _prior_b, _prior_base = (
                            _load_validated_window_data_triplet(
                                prod.production_energies_path,
                                prod.production_bias_path,
                                prod.production_base_path,
                                prior_production_convergence,
                            )
                        )
                        prior_segments = _validate_production_segments(
                            prior_production_convergence.get("production_segments"), _prior_e.shape[1]
                        )
                        if _prior_e.shape[1] == _prior_b.shape[0] == _prior_base.shape[0]:
                            prior_energy_history = [_prior_e[:, i] for i in range(_prior_e.shape[1])]
                            prior_bias_history = [float(x) for x in _prior_b]
                            prior_base_energy_history = [float(x) for x in _prior_base]
                            resumed_production_checkpoint = True
                        else:
                            print(
                                f"  [WARN] 窗口 {window_idx} 生产 checkpoint 匹配，但已有能量/偏置/"
                                "基准数组帧数不一致，回退完整重采。"
                            )
                    except Exception as e:
                        print(f"  [WARN] 窗口 {window_idx} 生产历史能量数组加载失败（{e}），回退完整重采。")
                else:
                    print(f"  [WARN] 窗口 {window_idx} 生产 checkpoint 加载失败，回退完整重采。")
        if resumed_production_checkpoint:
            try:
                if self.sampling_score_sha256 is not None and (
                    prior_production_convergence.get(
                        "residual_sampling_protocol_version"
                    ) != IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
                ):
                    raise ValueError(
                        "joint-score production ledger protocol is missing or old; "
                        "refusing to guess its axis orientation"
                    )
                _prior_sampling_states = np.load(
                    prod.production_sampling_states_path, allow_pickle=False
                )
                _prior_residual_basis = np.load(
                    prod.production_residual_basis_path, allow_pickle=False
                )
                expected_frames = len(prior_energy_history)
                if (
                    _prior_sampling_states.shape != (expected_frames, len(wc.lc_win))
                    or _prior_residual_basis.shape != (expected_frames,)
                    or not np.all(np.isfinite(_prior_sampling_states))
                    or not np.all(np.isfinite(_prior_residual_basis))
                ):
                    raise ValueError("joint-score production ledgers are invalid")
                prior_sampling_state_history = [
                    _prior_sampling_states[i, :].copy()
                    for i in range(expected_frames)
                ]
                prior_residual_basis_history = [
                    float(value) for value in _prior_residual_basis
                ]
            except Exception as exc:
                print(f"  [WARN] joint-score production ledgers unavailable ({exc}); full resample required")
                resumed_production_checkpoint = False
        # 🔑 [2026-09-14 / 阶梯空转] fail-closed：上面已经按"能续采"把这次冻结
        # 复验跳掉了，这里却没能真的续上（能量三件套读不回来 / 帧数对不上），
        # 那就等于"既没验证、又要从 0 步重采并覆盖已有产物"。这条路必须炸，
        # 不许静默降级 —— 预检与真正恢复之间唯一可能变的只有磁盘上的数组文件。
        if skip_frozen_revalidation and not resumed_production_checkpoint:
            raise RuntimeError(
                f"窗口 {window_idx}: 增量续采预检通过（已跳过冻结 burn-in + 只读复验），"
                "但生产历史未能恢复 ⟹ 既无验证又将从 0 步覆盖已有产物。fail-closed。"
            )
        if not resumed_production_checkpoint:
            prod.prior_cumulative_production_steps = 0
            prior_energy_history = None
            prior_sampling_state_history = None
            prior_residual_basis_history = None
            prior_bias_history = None
            prior_base_energy_history = None
            print(
                f"  窗口 {window_idx} 预热/冻结验证与生产严格隔离："
                "验证样本不计入生产，生产从 0 步开始"
            )

        safety_state = wc.sim.context.getState(getForces=True, getPositions=True)
        safety_forces = safety_state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        fmax_safety = float(np.max(np.linalg.norm(safety_forces, axis=1)))
        preprod_force_threshold = 10000.0
        print(f"  生产前安全检查 max|F| = {fmax_safety:.1f} kJ/(mol·nm)（未做任何扰动）")
        if fmax_safety > preprod_force_threshold:
            if debug_mode:
                diagnose_force_groups_detailed(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_生产前拦截")
                diagnose_force_breakdown(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_生产前拦截")
                diagnose_top_force_atoms(
                    wc.sim.context,
                    wc.win_sys,
                    topology=self.topology,
                    ligand_indices=self.ligand_indices,
                    prefix=f"窗口{window_idx}_生产前拦截",
                )
            raise RuntimeError(
                f"窗口 {window_idx} 在已验证收敛的冻结 Hamiltonian 下 max|F|={fmax_safety:.1f} "
                f"kJ/(mol·nm) 仍超过安全阈值 {preprod_force_threshold:.0f}——这与刚通过的统计验证"
                "矛盾，可能是构象陷阱或数值问题，已阻止进入生产采样而不是用更多扰动掩盖它。"
            )

        wc.sampler.energy_buffer = []
        # 生产历史只有一个合法来源：同一冻结 f_k、同一 manifest 的生产
        # checkpoint。预热、freeze burn-in 和冻结验证的 history 全部清空，
        # 绝不混入生产统计。
        if resumed_production_checkpoint and prior_energy_history is not None:
            wc.sampler.energy_history = list(prior_energy_history)
            wc.sampler.bias_history = list(prior_bias_history)
            wc.sampler.base_energy_history = list(prior_base_energy_history)
            wc.sampler.sampling_state_energy_history = list(prior_sampling_state_history)
            wc.sampler.residual_basis_history = list(prior_residual_basis_history)
        else:
            wc.sampler.energy_history = []
            wc.sampler.bias_history = []
            wc.sampler.base_energy_history = []
            wc.sampler.sampling_state_energy_history = []
            wc.sampler.residual_basis_history = []
        wc.sampler.production_segment_starts = (
            [{key: seg[key] for key in ("start_frame", "reason", "session_id")}
             for seg in prior_segments] if resumed_production_checkpoint else []
        )
        _start_production_segment(
            wc.sampler, "cross_process_resume" if resumed_production_checkpoint else "fresh_or_rebuilt"
        )
        prod.production_pos_backup = safety_state.getPositions(asNumpy=True)
        # 🔑 [P1-13] 与坐标备份**同时**记下三份 history 的长度。灾难回退只把
        # 坐标退回这个备份点，却把备份之后已经写入的 energy/bias/base history
        # 留在原地，然后从旧坐标 + 新随机速度长出另一条分支——被放弃的分支与
        # 重启分支共享祖先，却仍被当作一条连续时间序列做自相关子采样
        # （_decorrelate_by_worst_target_state），g 与 N_decorr 的口径就不再可靠。
        # 有了这个长度，回退时才能把三份 history 同步截断回同一个分叉点。
        prod.production_history_backup_len = _production_history_lengths(wc.sampler)

        # ---------- 生产采样 ----------
        # 🔑 防御性断言：走到这里必须已经满足严格收敛判据（sampler.bias_converged
        # 为 True），否则说明上面的收敛判据/raise 逻辑被绕过或本函数被以未预期的
        # 方式调用——不能让这种情况悄悄进入生产采样，MBAR 需要一个 f_k 已冻结的
        # 单一固定采样分布。
        if not wc.sampler.bias_converged:
            raise RuntimeError(
                f"窗口 {window_idx} 内部状态异常：即将进入生产采样但 "
                "sampler.bias_converged 不为 True。这不应该发生——说明收敛判据/"
                "raise 逻辑被绕过，拒绝继续，请检查上面的预热分支逻辑。"
            )
        # 生产阶段的硬边界：从这一行起 f_k 只读。即便未来有人误把
        # update/recalibration/restore 逻辑插进生产循环，下面逐 update 的
        # 运行时断言也会立即终止，而不是悄悄生成混合偏置数据。
        prod.production_f_k_lock = np.asarray(
            [
                wc.sim.context.getParameter(f"{self.prefix}_f_{k}")
                for k in range(K)
            ],
            dtype=np.float64,
        )
        if not np.allclose(
            prod.production_f_k_lock,
            np.asarray(production_frozen_f_k, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                f"窗口 {window_idx} 生产入口的 f_k 与冻结 manifest 不一致；"
                "拒绝进入生产。生产阶段禁止恢复、重算或修改 f_k。"
            )

        # 🔑 [2026-08-27，见 docs/experiments/EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md
        # 补验收] 真正的生产入口标记：checkpoint 恢复/续采判断、安全检查、
        # f_k 与冻结 manifest 一致性断言全部通过之后，第一步生产采样之前。
        # load_ibs_state 可能已经从上一次 attempt 的 IBS JSON 里恢复出这份
        # 冻结 f_k 的入口标记——如果数值上就是这一次 production_f_k_lock，
        # 说明这是同一份冻结 f_k 的续采，必须原样保留旧标记，不能用"现在"
        # 重新读到的 Context 值覆盖"最初"的值；否则（None，或数值对不上、
        # 说明标记来自另一份已作废的校准）才在这里第一次写入，并立即调用
        # save_ibs_state() 落盘——不这样做的话这份标记只活在内存里，进程
        # 一退出、续跑重新构造 sampler 时就彻底丢了，下游冻结检查永远读到
        # None，静默退化成旧的单段比较。
        #
        # 🔑 [续跑边界补丁] marker 为 None 时不能一律当成"第一次进入生产"
        # 直接写现在的值：如果 resumed_production_checkpoint 为 True（已有
        # 之前 attempt 留下的、真正被用来驱动过采样的生产历史——energies/
        # bias/base.npy + 匹配的 checkpoint），那份历史最初进入生产时的
        # f_k 就是真的不知道（旧版本从没记过），现在写一个"现在"的值会把
        # 一段其实入口未知的旧历史悄悄升级成"生产入口标记验证通过"，这正是
        # 这个字段本该防止的事情——必须继续保持 None，让下游 reconcile
        # 走 degraded 路径。只有 resumed_production_checkpoint 为 False
        # （这次生产是从 0 步真正重新开始，没有沿用任何旧生产样本）时，
        # marker=None 才代表"这次的生产历史确实还没有入口标记"，可以在这
        # 里第一次写入。
        #
        # 判定逻辑本身在 _resolve_production_entry_marker()（纯函数，
        # 独立于 sampler/磁盘，见其 docstring）——这里只按返回值执行写/
        # 存/拒绝三种动作，保证真正跑生产用的分支判断和离线单元测试用的
        # 是同一份代码，而不是各自维护一份容易漂移的复制。
        _entry_marker_action, _entry_marker_value = _resolve_production_entry_marker(
            wc.sampler.production_entry_f_k, prod.production_f_k_lock, resumed_production_checkpoint
        )
        if _entry_marker_action == "refuse":
            raise RuntimeError(
                f"窗口 {window_idx} 已恢复的生产入口标记与本次冻结 f_k 驱动的"
                "生产 checkpoint 不一致——这份 checkpoint 的 manifest 已确认"
                "生产历史属于当前冻结 f_k，但保存的入口标记却对不上，是真实"
                "数据矛盾，不能用当前值静默覆盖。已阻止继续生产，请人工核查"
                "这份 IBS 状态文件与生产 checkpoint 的一致性。"
            )
        elif _entry_marker_action == "set":
            wc.sampler.production_entry_f_k = _entry_marker_value
            wc.sampler.save_ibs_state(
                ibs_state_file, wc.lc_win, wc.lv_win, stage_type=stage_type
            )
        # else "keep": sampler.production_entry_f_k 原样不动，不写盘。

        print(f"\n  [阶段5] 生产采样 ({prod.effective_n_steps_per_window} 步)...")
        # 🔑 [production checkpoint 续采] 只跑目标步数与已完成累计步数的差值，
        # 不是把 effective_n_steps_per_window 当成"这次要新跑这么多步"（那样
        # 会变成每次 reseed_resample 都把已经真实跑过的步数重复计入，
        # 250k->500k 实际跑出 750k）。
        remaining_production_steps = max(
            int(prod.effective_n_steps_per_window) - int(prod.prior_cumulative_production_steps), 0
        )
        if resumed_production_checkpoint:
            print(
                f"  窗口 {window_idx} 从生产 checkpoint 续算：累计已完成 "
                f"{prod.prior_cumulative_production_steps} 步，本次只需再跑 "
                f"{remaining_production_steps} 步（目标 {prod.effective_n_steps_per_window} 步），"
                f"保留 {len(wc.sampler.energy_history)} 帧已有样本，坐标/速度/积分器 RNG "
                "状态与上次中断时完全一致（注意：这里跳过的只是重跑 production 本身——"
                "最小化/dt测试/Boresch爬坡/冻结重验证仍会照常先跑一遍，代价远小于"
                "重跑整个 production 步数预算）。"
            )
        prod.n_updates = remaining_production_steps // steps_per_update

        # 🔑 [EARLY_STOP_PROTOCOL_VERSION] 在线 early-stop 监测状态。默认关闭
        # （enable_early_stop=False，见 run_all_windows 参数说明），下面这段
        # 只在显式打开时才会真正跑 local MBAR/提前退出；关闭时行为与之前完全
        # 一致（跑满 n_updates，不做任何额外检查）。
        prod.early_stop_kt = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(unit.kilojoule_per_mole)
        prod.early_stop_min_updates = max(1, int(es.early_stop_min_steps) // max(1, int(steps_per_update)))
        prod.early_stop_check_interval_updates = max(1, int(es.early_stop_check_interval_steps) // max(1, int(steps_per_update)))
        prod.early_stop_pass_count = 0
        prod.early_stop_check_history = []
        prod.early_stop_triggered = False
        prod.early_stop_stop_reason = None
        prod.early_stop_previous_local_dg = None
        prod.actual_production_updates = prod.n_updates  # overwritten below only if early-stopped

        # 以下 CV/力组诊断仅用于排障，默认关闭（debug_mode=False），
        # 避免生产环境每个窗口都无条件刷屏并产生额外的 getState 开销。
        if debug_mode:
            for gid in [0, 2, 3]:
                e_g = wc.sim.context.getState(getEnergy=True, groups={gid}).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
                print(f"Group {gid} energy: {e_g:.1f} kJ/mol")

            cv_vals = ibs_wrap.get_force().getCollectiveVariableValues(wc.sim.context)
            print(f"First CV value (should be LE interaction): {cv_vals[0]:.1f} kJ/mol")
            # 获取 CV 力对象（第一个 CV）
            cv_force_0 = ibs_wrap.get_force().getCollectiveVariable(0)
            # 检查它的交互组
            num_groups = cv_force_0.getNumInteractionGroups()
            print(f"  [诊断] CV 交互组数量 = {num_groups}")
            for gidx in range(num_groups):
                set1, set2 = cv_force_0.getInteractionGroupParameters(gidx)
                print(f"  [诊断] 交互组 {gidx}: size={len(set1)} × {len(set2)}")

            print("\n软核 CV 详细诊断：")

            # 1. 获取第一个 CV 力对象
            cv_force = ibs_wrap.get_force().getCollectiveVariable(0)

            # 2. 打印其能量表达式
            print(f"  CV 表达式：\n{cv_force.getEnergyFunction()}\n")

            # 3. 打印 CV 力内部的所有全局参数及其当前值
            print("  CV 全局参数：")
            for i in range(cv_force.getNumGlobalParameters()):
                name = cv_force.getGlobalParameterName(i)
                try:
                    val = wc.sim.context.getParameter(name)
                except Exception:
                    val = cv_force.getGlobalParameterDefaultValue(i)
                print(f"    {name} = {val}")

            # 4. 打印配体和环境原子各自的前 5 个参数
            lig_indices = self.ligand_indices
            num_atoms = wc.win_sys.getNumParticles()
            cv_param_names = [
                cv_force.getPerParticleParameterName(i)
                for i in range(cv_force.getNumPerParticleParameters())
            ]

            def _format_cv_particle(idx: int) -> str:
                params = cv_force.getParticleParameters(idx)
                values = []
                for name, val in zip(cv_param_names, params):
                    if hasattr(val, "value_in_unit"):
                        if "charge" in name.lower() or name.lower().startswith("q"):
                            val = val.value_in_unit(unit.elementary_charge)
                        elif "sigma" in name.lower() or "r0" in name.lower():
                            val = val.value_in_unit(unit.nanometer)
                        else:
                            val = val.value_in_unit(unit.kilojoule_per_mole)
                    values.append(f"{name}={float(val):.4f}")
                return ", ".join(values)

            print(f"\n  配体原子前 5 个参数（{', '.join(cv_param_names)}）：")
            for idx in lig_indices[:5]:
                print(f"    atom {idx}: {_format_cv_particle(idx)}")

            # 环境原子前 5 个参数
            env_list = [i for i in range(num_atoms) if i not in set(lig_indices)]
            print("\n  环境原子前 5 个参数：")
            for idx in env_list[:5]:
                print(f"    atom {idx}: {_format_cv_particle(idx)}")

        # ✅ 性能修复：getForces=True/getPositions=True 每次都要把全原子数组从
        # GPU 传回 CPU，代价远高于纯 getEnergy=True。原来每个 update（默认每
        # 500 步）都做一次完整的力+坐标查询，纯粹为了灾难检测和回退备份——
        # 但灾难本身是小概率事件。改为默认只查能量（便宜很多），每
        # IBS_FORCE_CHECK_INTERVAL 个 update 才做一次完整的力/坐标查询；一旦
        # 能量本身已经 NaN/Inf，无论轮到没轮到都立刻补查一次力，不能让"跳过"
        # 的 update 错过真正的灾难信号。
        # ⚠️ 权衡：两次完整检查之间，如果力在没有把能量推成 NaN/Inf 的前提下
        # 短暂冲高又回落，这个窗口是看不到的——比原来"每 update 都查"的覆盖率
        # 低。position 备份只在做了完整检查的 update 上刷新，回退时可能回退到
        # 更早的坐标（多丢几个 update 的进度），但回退本身就是保守操作，不影响
        # 正确性。
        prod.force_check_interval = max(1, int(os.environ.get("IBS_FORCE_CHECK_INTERVAL", "10")))

        # 开始生产采样循环
        # 🔑 [性能计时] 按"积分/guard/CV-probe/ledger-IO"四档累加墙钟耗时，
        # 默认常开（perf_counter 单次开销可忽略）——用来把后续任何"是否
        # 值得继续优化"的判断建立在实测数字上，而不是猜测。跟余数补齐块
        # 共用同一份累加器，因为它是同一窗口生产采样的尾巴。
        prod.loop_timers: Dict[str, float] = {}
        self._finish_production(
            K=K,
            debug_mode=debug_mode,
            es=es,
            end=end,
            ibs_state_file=ibs_state_file,
            ibs_wrap=ibs_wrap,
            wc=wc,
            lse_log_residual_tolerance=lse_log_residual_tolerance,
            n_steps_per_window=n_steps_per_window,
            prod=prod,
            repair_policy=repair_policy,
            stage_type=stage_type,
            start=start,
            steps_per_update=steps_per_update,
            warm=warm,
            window_idx=window_idx,
        )

    def _commit_frozen_f_k(
        self,
        *,
        effective_frozen_validation_budget,
        ema_mean_p_values,
        expected_main_manifest,
        frozen_validation_is_final_rung,
        frozen_validation_reserved_steps,
        full_bias_step_budget,
        ibs_state_file,
        wc,
        main_ckpt_path,
        main_manifest_path,
        max_bias_updates,
        mode,
        prior_cumulative_steps,
        resumed_calibration_pending,
        sgd_step_budget,
        skip_frozen_revalidation,
        stage_type,
        warm,
        warmup_only,
        window_idx,
    ):
        """相位 5：f_k 冻结/提交—— 刀1 第二刀·刀a 自 run_all_windows 原样搬出。

        纯搬运（verify_extract --block for --slice 120:121 逐节点等价）；
        跨界变量按原名作 keyword-only 形参（CUTTING_PLAN.md §2.2.1）。
        """
        if warm.bias_converged:
            # [IBS_BIAS_PROTOCOL_VERSION=29] 收敛报告改用 local-MBAR loose gate
            # 的相邻态残差（max|Δf_k−ΔF^MBAR|），不再打印已移除的占据 LSE 残差。
            _gate_gap = (
                warm.last_local_mbar_gate.get("max_adjacent_delta_kJ_mol")
                if warm.last_local_mbar_gate is not None else None
            )
            _gate_gap_repr = (
                f"{float(_gate_gap):.3f} kJ/mol"
                if _gate_gap is not None else "n/a"
            )
            if skip_frozen_revalidation:
                # [2026-09-14] 这次根本没跑复验，别打出 "通过 loose gate
                # max|Δf_k−ΔF^MBAR|=n/a" 这种假声明。
                print(
                    " 完成（增量续采：沿用上一次已通过冻结验证的 f_k，本次未重跑"
                    "冻结 burn-in / 只读复验；Context 由生产 checkpoint 恢复）"
                )
            elif warm.best_effort_acceptance:
                print(
                    f" 完成（{warm.bias_update_count} 次权重更新、"
                    f"{warm.learning_to_validation_cycles} 次 loose-gate 未过后，采用 "
                    f"{warm.best_effort_acceptance_reason} best-effort：warmup 预算耗尽即接受"
                    f"当前 f_k；最近一次 max|Δf_k−ΔF^MBAR|={_gate_gap_repr}，阈值 "
                    f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f} kJ/mol；"
                    "未声称严格达标，生产后独立质量门 + 最终 MBAR 仍生效）"
                )
            else:
                print(
                    f" 完成（{warm.bias_update_count} 次权重更新、"
                    f"{warm.learning_to_validation_cycles} 次 loose-gate 未过后，最终在冻结 "
                    f"f_k 下通过 local-MBAR loose gate："
                    f"max|Δf_k−ΔF^MBAR|={_gate_gap_repr} < "
                    f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f} kJ/mol"
                    f"（{warm.validation_sample_count} frames / "
                    f"{warm.last_local_mbar_gate.get('n_batches') if warm.last_local_mbar_gate else 0} "
                    "batches）；真正的自由能/ESS/overlap/误差交生产后 MBAR）"
                )
            wc.sampler.bias_converged = True
            # 🔑🔑 [2026-09-15] **临时生产与收敛在词汇上必须分开。**
            # 走到这里的临时生产窗口，它的 Δf−ΔF **从未被求出** —— 把它写成
            # `converged` / `verified` 等于凭空发明一份不存在的证据，而下游
            # （控制器的 `_VERDICT`、任何读 bias_status 的审计）会据此认为
            # 这个窗口已经验收过。`bias_converged=True` 只表达一件事：
            # **这个窗口不再需要重新进预热**（Epoch 已经花掉、生产产物在盘上），
            # 证据强度由 `f_k_evidence_status` 单独表达。
            _prov = bool(warm.provisional_production)
            wc.sampler.bias_status = (
                "provisional_production" if _prov else "converged"
            )
            wc.sampler.f_k_evidence_status = (
                F_K_EVIDENCE_INDETERMINATE if _prov else F_K_EVIDENCE_VERIFIED
            )
            wc.sampler.frozen_f_k_pending = None
            wc.sampler.last_failure_reason = None
            wc.sampler.save_ibs_state(
                ibs_state_file, wc.lc_win, wc.lv_win, stage_type=stage_type
            )
            # 🔑 一次成功的收敛必须让这个窗口的失败记录彻底过期——否则一次
            # 先失败、重跑后成功的窗口会留下一份跟当前 convergence.json 矛盾
            # 的旧 warmup_failure.json，误导任何检查"这个窗口是否曾失败过"的
            # 下游逻辑（人工排查、自动化审计脚本等）。
            # ⚠️ 临时生产**例外**：那份 warmup_failure.json 正是"这个窗口的
            # f_k 从未被验证过"的**唯一**落盘证据，删掉它等于把临时生产洗成
            # 一次干净的收敛。改为刷新它（带上临时生产标记），不删。
            stale_failure_path = os.path.join(
                self.output_dir,
                f"dual_window_{window_idx}_{stage_type}_warmup_failure.json",
            )
            if _prov:
                warm.bias_warmup_diag["status"] = "provisional_production"
                warm.bias_warmup_diag["provisional_production"] = True
                warm.bias_warmup_diag["validation_indeterminate"] = (
                    warm.validation_indeterminate_diag
                )
                # 冻结的那份 f_k 就在 `validation_indeterminate_diag` 里
                # （撞上限那一刻从 Context 读的），别再手抄一份来源。
                warm.bias_warmup_diag["frozen_f_k_kJ_mol"] = (
                    (warm.validation_indeterminate_diag or {}).get("frozen_f_k_kJ_mol")
                )
                _atomic_write_json(stale_failure_path, warm.bias_warmup_diag)
            elif os.path.exists(stale_failure_path):
                try:
                    os.remove(stale_failure_path)
                except OSError:
                    pass
            # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 同理清理主窗口
            # checkpoint+manifest——窗口已经真正收敛，不再需要这份"续验用"
            # 的二进制快照，留着只会占磁盘、并在以后误导人工排查。
            for _stale_path in (main_ckpt_path, main_manifest_path):
                if os.path.exists(_stale_path):
                    try:
                        os.remove(_stale_path)
                    except OSError:
                        pass
        else:
            failure_path = os.path.join(
                self.output_dir,
                f"dual_window_{window_idx}_{stage_type}_warmup_failure.json",
            )
            if warm.validation_indeterminate_diag is not None:
                # 🔑 [2026-09-11] 第四类结局：**无法判定**。冻结验证一直样本不足，
                # Δf−ΔF 从未被求出来。这既不是"f_k 收敛"也不是"f_k 不对"——
                # 不得据此插 λ / 拆窗 / 退回 SGD（那只是把误判推迟 10 万步），
                # 所以走一个独立异常，路径演化循环故意不捕获它。
                # 冻结 f_k、已累计批数、已耗步数全部落盘，resume 接着验。
                warm.bias_warmup_diag["status"] = "validation_budget_exhausted_indeterminate"
                warm.bias_warmup_diag["validation_indeterminate"] = (
                    warm.validation_indeterminate_diag
                )
                wc.sampler.bias_status = "frozen_validation_indeterminate"
                # 无法判定 ≠ 已校准。恢复冻结验证是为了把测量做完，不是认可候选。
                wc.sampler.f_k_evidence_status = F_K_EVIDENCE_INDETERMINATE
                # HALT_BUDGET 那一支可能连一个冻结候选都没有（循环没进过），
                # 此时不要伪造一份 f_k_pending —— 没有候选就是没有。
                _pending_fk = warm.validation_indeterminate_diag.get("frozen_f_k_kJ_mol")
                wc.sampler.frozen_f_k_pending = (
                    [float(x) for x in _pending_fk] if _pending_fk else None
                )
                wc.sampler.frozen_validation_batches_done = int(
                    warm.validation_indeterminate_diag["n_batches"]
                )
                wc.sampler.frozen_candidate_fingerprint = frozen_candidate_fingerprint(
                    wc.sampler.frozen_f_k_pending
                )
                wc.sampler.frozen_validation_cumulative_steps = int(
                    prior_cumulative_steps + warm.steps_at_full_bias
                )
                wc.sampler.last_failure_reason = (
                    "halt_budget_no_steps_available"
                    if warm.validation_indeterminate_diag.get(
                        "halt_budget_no_steps_available")
                    else "validation_budget_exhausted_indeterminate"
                )
                wc.sampler.bias_converged = False
                _atomic_write_json(failure_path, warm.bias_warmup_diag)
                wc.sampler.save_ibs_state(
                    ibs_state_file, wc.lc_win, wc.lv_win, stage_type=stage_type
                )
                _atomic_save_openmm_checkpoint(wc.sim, main_ckpt_path)
                _atomic_write_json(main_manifest_path, expected_main_manifest)
                raise IBSValidationBudgetIndeterminateError(
                    f"窗口 {window_idx} 的冻结验证在单周期 "
                    f"{warm.validation_indeterminate_diag['n_batches']}/"
                    f"{IBS_LOCAL_MBAR_GATE_MAX_BATCHES} 批（"
                    f"{warm.validation_indeterminate_diag['validation_sample_count']} frames、"
                    f"{warm.validation_indeterminate_diag['steps_at_full_bias']} 步）内始终 "
                    f"{warm.validation_indeterminate_diag['last_gate_error']}："
                    "Δf−ΔF 从未被求出，对这份 f_k **无结论**（既非收敛也非不收敛）。"
                    "不改 f_k、不退回 SGD、不插 λ。冻结状态/批次/已耗预算已落盘，"
                    f"resume 会接着验；完整诊断在 {failure_path}。"
                    "要继续请延长该窗口的冻结验证预算，或先处理相关时间过长的根因。",
                    diagnostics=warm.bias_warmup_diag,
                    window_idx=window_idx,
                )
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 区分两种不同的"未收敛"：
            # (a) 这次（或之前 resume 续验的那次）已经用 MBAR 校准探针给出过
            #     一份 fixed-H overlap 证实过 λ 网格没问题、且校准探针自身
            #     样本/不确定度达标的 f_k，只是冻结验证还没在预算内通过——
            #     这份 f_k 不该被当成"求解器没找到答案"，不能退回 SGD；
            # (b) 真正从未获得过这样一份校准 f_k（SGD 没收敛、fixed-H overlap
            #     没通过、或校准探针本身样本不足）——这才是需要 SGD 继续
            #     搜索、或交给拆窗/插 λ 的普通未收敛。
            calibration_pending = bool(
                resumed_calibration_pending or ("mbar_calibration" in warm.bias_warmup_diag)
            )
            is_final_rung = False
            new_cumulative_steps = 0
            steps_spent_this_attempt = 0
            if calibration_pending:
                pending_f_k = (
                    warm.resumed_frozen_f_k if resumed_calibration_pending
                    else warm.bias_warmup_diag["mbar_calibration"]["calibrated_f_k_kJ_mol"]
                )
                # 🔑 [累计步数记账] resumed_calibration_pending 分支从不进入
                # learning，这次 attempt 的 steps_at_full_bias 就是纯验证步数；
                # 全新窗口首次 MBAR 校准的验证步数记在独立的 calibration_steps_used
                # （bias_warmup_diag["mbar_calibration"]["steps_used"]）。累加到
                # 跨 resume 持久化的 frozen_validation_cumulative_steps 上——
                # 全新校准是这份 f_k 的第一次验证，不累加任何旧值。
                # 🔑 [重复计数修复] 循环内每个 batch 的 checkpoint 落盘代码（见
                # 上面 "每评估完一批（无论 pass/fail）就存一次主窗口 checkpoint"
                # 处）已经把 sampler.frozen_validation_cumulative_steps 更新成
                # prior_cumulative_steps + steps_at_full_bias——循环退出时这个
                # 属性已经是这次 attempt 的正确累计值。这里如果再读一次这个
                # 属性、并在上面加一次 steps_spent_this_attempt，就等于把这次
                # attempt 的步数计了两遍（真实案例：prior=250000、这次真实跑了
                # 50000 步，循环内已经把属性更新成 300000，这里再加一次 50000
                # 变成 350000——足足多算了 50000 步，且下一次 resume 会拿着这个
                # 虚高的 350000 当 prior_cumulative_steps，把后续阶梯档位的真实
                # 预算越挤越少）。必须用进入这次 attempt 之前捕获的
                # prior_cumulative_steps（第 5997 行），不能读循环已经写脏的
                # sampler 属性。
                if resumed_calibration_pending:
                    steps_spent_this_attempt = int(warm.steps_at_full_bias)
                    new_cumulative_steps = prior_cumulative_steps + steps_spent_this_attempt
                else:
                    steps_spent_this_attempt = int(
                        warm.bias_warmup_diag["mbar_calibration"]["steps_used"]
                    )
                    new_cumulative_steps = steps_spent_this_attempt
                wc.sampler.frozen_validation_cumulative_steps = new_cumulative_steps
                # 🔑 [跨进程阶梯状态修复] 同上——调用方没有显式告诉我们这是不是
                # 最后一档时（本地 dict 在进程重启后为空），不能默认 False：
                # 这次 effective_frozen_validation_budget 如果本来就是回退逻辑
                # 解出来的 FROZEN_VALIDATION_LADDER_SCHEDULE_STEPS[-1]（说明
                # 持久化的累计步数已经追上或超过最后一档），就应该判定为终态，
                # 而不是静默地当成"还有下一档可以延长"，那样只会在下一次调用
                # 时重复同一档、原地打转。
                is_final_rung = _resolve_frozen_validation_is_final_rung(
                    window_idx,
                    frozen_validation_is_final_rung,
                    effective_frozen_validation_budget,
                )
                if is_final_rung:
                    # 🔑 [终态] 调用方已经把这次标记为冻结验证阶梯的最后一档，
                    # 仍未通过独立验证——这不再是"等待延长"，是真正的终态失败。
                    # 绝不能存成 calibrated_pending_validation（那会让下次
                    # resume 看起来还能继续自动续验），frozen_f_k_pending 清空。
                    wc.sampler.bias_status = "calibrated_validation_failed"
                    wc.sampler.f_k_evidence_status = F_K_EVIDENCE_REFUTED
                    wc.sampler.frozen_f_k_pending = None
                    warm.bias_warmup_diag["calibration_pending_validation"] = False
                    warm.bias_warmup_diag["calibration_validation_terminally_failed"] = True
                else:
                    wc.sampler.bias_status = "calibrated_pending_validation"
                    wc.sampler.f_k_evidence_status = F_K_EVIDENCE_CALIBRATED
                    wc.sampler.frozen_f_k_pending = [float(x) for x in pending_f_k]
                    warm.bias_warmup_diag["calibration_pending_validation"] = True
                    warm.bias_warmup_diag["frozen_f_k_kJ_mol"] = list(wc.sampler.frozen_f_k_pending)
                warm.bias_warmup_diag["frozen_validation_budget_this_attempt"] = int(
                    effective_frozen_validation_budget
                )
                warm.bias_warmup_diag["frozen_validation_cumulative_steps"] = int(new_cumulative_steps)
                # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 共享的最终兜底
                # 保存点——不管这次失败是来自 resumed_calibration_pending 续验
                # 还是全新窗口的首次 MBAR 校准后验证，都在这里再存一次最新
                # checkpoint（跟循环内"每批一次"的保存是同一份文件，覆盖式
                # 落盘，这里只是防御性地确保退出前一定有一份最新的）。终态
                # 失败时也存一份：万一以后人工决定基于这份构型继续排查，不用
                # 从头重建。
                _atomic_save_openmm_checkpoint(wc.sim, main_ckpt_path)
                _atomic_write_json(main_manifest_path, expected_main_manifest)
            else:
                # [Candidate-first, Validate-or-Learn v1] Only writes
                # "failed"/"unconverged" going forward (never the legacy
                # calibrated_* values) -- see the calibration_pending
                # branch above, which is legacy-only and left untouched.
                # "failed" is reserved for a real, completed local-MBAR
                # gate rejection (gate ran cleanly, gap stayed too big
                # even after the one damped+capped retry): genuine
                # evidence against this candidate. Everything else
                # (never reached a completed gate check, occupancy
                # collapsing, or local-MBAR numerically unsolvable) is
                # "unconverged" -- worth another resume with more budget,
                # not a terminal verdict.
                if warm.last_failure_reason in (
                    "local_mbar_gap_exceeded_retry_applied",
                    "local_mbar_gap_exceeded_after_retry",
                ):
                    wc.sampler.bias_status = "failed"
                    wc.sampler.f_k_evidence_status = F_K_EVIDENCE_REFUTED
                else:
                    wc.sampler.bias_status = "unconverged"
                    wc.sampler.f_k_evidence_status = F_K_EVIDENCE_NONE
                wc.sampler.frozen_f_k_pending = None
                wc.sampler.frozen_validation_cumulative_steps = 0
            wc.sampler.last_failure_reason = warm.last_failure_reason
            _atomic_write_json(failure_path, warm.bias_warmup_diag)
            wc.sampler.bias_converged = False
            wc.sampler.save_ibs_state(
                ibs_state_file, wc.lc_win, wc.lv_win, stage_type=stage_type
            )
            print(
                f" 未收敛（{warm.bias_update_count}/{max_bias_updates} 次权重更新，"
                f"{warm.learning_to_validation_cycles} 次冻结验证失败重学习，"
                + (
                    f"{warm.frozen_validation_retry_count} 次续验批次未通过（续验模式下未退回 learning），"
                    if resumed_calibration_pending else ""
                )
                + (
                    f"{warm.steps_at_full_bias}/{full_bias_step_budget} 步总上限"
                    f"（learning={sgd_step_budget}, freeze burn-in={warm.frozen_burn_in_steps}, "
                    f"冻结累计验证预留={frozen_validation_reserved_steps}），最终阶段={mode}）"
                )
            )
            print(
                "    reported mean_p[k]（若进入验证则为固定 f_k 累计均值；"
                f"否则为 raw EMA 诊断）= {ema_mean_p_values}"
            )
            # 🔑 [2026-09-10] 这里原来打的是
            #   `TMBAR: n_entries=…, converged=…, min_overlap=…, min_absolute_ess=…`
            # 后三个字段读的是 `last_update_diagnostics["tmbar_update"]`，
            # 而它的**唯一**写入方是 `IBSSampler.update_weights()` —— 该方法自
            # Candidate-first / Validate-or-Learn v1 起已**零调用点**。于是这三项
            # 在生产日志里永远打印 `None`（`min_overlap`/`min_absolute_ess` 更是
            # 连写入方都没写过这两个键，双重死）。
            #
            # 这不是无害的噪声：2026-09-09 那批失败 run 的日志里满屏
            # `converged=None`，读的人（包括排查的人）会据此认为"收敛判据坏了、
            # 门永远判不过"，而真实情况是占据塌陷、门根本没轮到上场。
            # 只打真正活着的两项：TMBAR 历史条数，和**真正决定放行的**
            # local-MBAR loose gate。
            print(
                f"    TMBAR 历史条数 n_entries={len(wc.sampler.tmbar_history)}"
                "（仅计数，不参与任何判据）"
            )
            if warm.last_local_mbar_gate is not None:
                print(
                    "    放行判据 local-MBAR loose gate: "
                    f"max_adjacent_delta="
                    f"{warm.last_local_mbar_gate.get('max_adjacent_delta_kJ_mol')} kJ/mol, "
                    f"situation={warm.last_local_mbar_gate.get('situation')}"
                )
            else:
                print(
                    "    放行判据 local-MBAR loose gate: 从未被评估过 —— "
                    "说明失败发生在更前面（占据从未平坦到可冻结，或刚冻结即被"
                    "早退探针判定塌陷退回），不是这道门判否的。"
                )
            _weight_diag = wc.sampler.last_update_diagnostics.get("weight_update", {})
            if _weight_diag:
                print(
                    "    bounded weight update: "
                    f"eta={_weight_diag.get('eta')}, "
                    f"max|delta_f|={_weight_diag.get('max_abs_delta_f_kJ_mol')} kJ/mol, "
                    f"pairwise_spread={_weight_diag.get('pairwise_delta_f_spread_kJ_mol')} kJ/mol, "
                    f"pairwise_limit={_weight_diag.get('pairwise_step_limit_kJ_mol')} kJ/mol, "
                    f"regime={_weight_diag.get('gain_regime')}"
                )
            if "bidirectional_overlap_probe" in warm.bias_warmup_diag:
                print(
                    "    fixed-H 双向 overlap = "
                    f"{warm.bias_warmup_diag['bidirectional_overlap_probe']['pairs']}"
                )
            if calibration_pending:
                if is_final_rung:
                    print(
                        f"    [ERR] 校准后的冻结 f_k 冻结验证累计 {new_cumulative_steps} 步后仍未通过，"
                        "已判定为终态失败（calibrated_validation_failed）——"
                        "不再自动续验/延长预算/回退 learning，需要人工检查。"
                    )
                    raise_msg = (
                        f"窗口 {window_idx} 的冻结校准验证累计 {new_cumulative_steps} 步后仍未能通过"
                        "独立验证，已判定为终态失败（calibrated_validation_failed）——这份 f_k 已经用 "
                        "fixed-H overlap 探针 + bias 校准探针证明过物理正确，终态不再自动续验/延长"
                        f"预算/回退 learning，需要人工检查。完整诊断已写入 {failure_path}。"
                    )
                else:
                    print(
                        f"    校准后的冻结 f_k 已存为 calibrated_pending_validation"
                        f"（本次冻结验证累计 {new_cumulative_steps}/{effective_frozen_validation_budget} "
                        "步仍未通过），resume 会跳过 SGD/重新校准，直接续验"
                        f"（不重新烧已经验证过的这 {steps_spent_this_attempt} 步）。"
                    )
                    raise_msg = (
                        f"窗口 {window_idx} 的冻结校准验证在累计 {effective_frozen_validation_budget} "
                        f"步目标内仍未通过独立验证（本次 attempt 实际验证了 {steps_spent_this_attempt} "
                        f"步，累计已验证 {new_cumulative_steps} 步）。这份 f_k 已经用 fixed-H overlap "
                        "探针 + bias 校准探针证明过物理正确，不应拆窗/插 λ/退回 SGD，应延长冻结验证"
                        f"累计预算重试。完整诊断已写入 {failure_path}。"
                    )
                # 🔑 [2026-09-14 / S1] `bias_warmup_diag` 在这里就是封存记录的
                # **唯一身份来源**，所以进 raise 之前必须把身份三键补全：
                # 权威名 `frozen_f_k_kJ_mol` 此前只在 non-final 分支写过（终态
                # 分支反而把 `frozen_f_k_pending` 清空了），指纹从来没写进来过。
                # 用同一个生成器，别在这里手写键名。
                warm.bias_warmup_diag.update(
                    refuted_candidate_identity(pending_f_k, window_idx)
                )
                raise IBSFrozenCalibrationValidationError(
                    raise_msg,
                    diagnostics=warm.bias_warmup_diag,
                    terminal=is_final_rung,
                )
            if mode == "learning":
                _ordinary_failure_reason = (
                    "time-averaged Q̂ 的 learning 候选平衡在预算内未收敛，"
                    "尚未进入冻结独立验证"
                )
            else:
                _ordinary_failure_reason = "候选 f_k 在冻结后的独立验证中未收敛"
            if not legacy_repair:
                # non_mutating_v1 has priority over warmup_only: design-time
                # execution is not permission to mutate this sampled grid.
                _ordinary_failure_action = (
                    "保留全部已采数据；不按相邻 overlap 拆窗/插点/重校准，交人工 / "
                    "rescue-coverage 审计决定是否需要新的 immutable ensemble。"
                )
            elif warmup_only:
                _ordinary_failure_action = (
                    "legacy 设计期将先用已有 lambda 拆分 IBS ensemble；只有不可再拆的"
                    "两态窗口才允许插入热力学长度中点，并必须重新通过同一 LSE 自洽验证。"
                )
            else:
                _ordinary_failure_action = (
                    "legacy repair 将反馈为拆窗；只有最小窗口的 fixed-H overlap 失败才允许插点。"
                )
            raise IBSWarmupConvergenceError(
                f"窗口 {window_idx} 的 IBS 偏置预热在 {warm.bias_update_count} 次权重更新、"
                f"{warm.learning_to_validation_cycles} 次冻结验证失败重学习后，"
                f"{_ordinary_failure_reason}（f_not_converged）；完整诊断已写入 {failure_path}。"
                f"{_ordinary_failure_action}",
                diagnostics=warm.bias_warmup_diag,
            )

    def _run_warmup_loop(
        self,
        *,
        K,
        _attempt_ledger,
        _nan_diagnostics,
        check_chunk,
        expected_main_manifest,
        full_bias_step_budget,
        ibs_state_file,
        k,
        wc,
        main_ckpt_path,
        main_manifest_path,
        max_bias_updates,
        prior_cumulative_steps,
        provisional_authorized,
        resumed_calibration_pending,
        skip_frozen_revalidation,
        skip_warmup_entirely,
        stage_type,
        start,
        warm,
        window_idx,
    ):
        """相位 3：预热循环—— 刀1 第二刀·刀b 自 run_all_windows 原样搬出。

        纯搬运（verify_extract --block for --slice 逐节点等价）；跨界变量按
        原名作 keyword-only 形参（CUTTING_PLAN.md §2.2.1）。
        """
        while (not skip_frozen_revalidation) and warm.steps_at_full_bias < full_bias_step_budget:
            # learning 阶段的更新次数上限只约束 learning；freeze_burn_in/
            # validating 只受总步数安全帽 full_bias_step_budget 约束。反复未
            # 通过 loose-gate 把更新次数烧到上限时退出循环，由下方"预算耗尽即
            # 接受当前 f_k"分支放行进生产（见 IBS_BIAS_PROTOCOL_VERSION=29）。
            if warm.mode == "learning" and warm.bias_update_count >= int(max_bias_updates):
                break
            with _timed(warm.warmup_timers, "integration_s"):
                guarded_step(wc.sim, check_chunk, f"窗口 {window_idx} 偏置学习", on_error=_nan_diagnostics)
            warm.steps_at_full_bias += check_chunk
            _attempt_ledger[warmup_ledger_bucket_for_mode(warm.mode)] += check_chunk
            with _timed(warm.warmup_timers, "cv_probe_s"):
                wc.sampler.collect_energies()

            if warm.mode == "learning":
                # [Candidate-first, Validate-or-Learn v1] LEARN controls
                # occupancy only -- no full-history TMBAR solve, no
                # trusted/untrusted/self_consistent/legacy_quality branch
                # selection (update_weights()/_solve_tmbar_and_recenter()
                # are left fully defined but no longer called from this
                # loop), no fixed min_bias_updates batch count. Sole
                # update rule: bounded log-occupancy feedback
                # (Δf_k=-η·kT·log(K·p_k)). Freeze the instant the raw
                # residual drops to/below IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW
                # -- VALIDATE (the unchanged real-Hamiltonian local-MBAR
                # loose gate below) is the sole production entry proof,
                # not a batch counter or a consecutive-pass streak.
                f_old = np.array(
                    [wc.sim.context.getParameter(f"{self.prefix}_f_{k}") for k in range(K)]
                )
                with _timed(warm.warmup_timers, "weight_update_s"):
                    mean_p_batch = wc.sampler.evaluate_frozen_batch_probability(
                        min_buffer_size=IBS_TMBAR_LEARNING_MINIBATCH_FRAMES
                    )
                if mean_p_batch is None:
                    continue
                f_new, learn_diag = wc.sampler._bounded_log_occupancy_update(
                    f_old,
                    mean_p_batch,
                    severe_max_pairwise_step_kT=IBS_TMBAR_FALLBACK_SGD_PAIRWISE_STEP_KT,
                )
                for k in range(K):
                    wc.sim.context.setParameter(f"{self.prefix}_f_{k}", float(f_new[k]))
                wc.sampler.f_history.append(f_new.copy())
                wc.sampler.last_update_diagnostics = {
                    "source": "learn_bounded_log_occupancy",
                    "weight_update": learn_diag,
                    "adjacent_delta_u_is_convergence_gate": False,
                }
                if wc.sampler.seed_source != "learned":
                    wc.sampler.seed_source = "learned"
                warm.bias_update_count += 1
                warm.updates_since_freeze += 1
                if len(wc.sampler.f_history) >= 2:
                    warm.last_f_delta = float(
                        np.max(
                            np.abs(
                                np.asarray(wc.sampler.f_history[-1])
                                - np.asarray(wc.sampler.f_history[-2])
                            )
                        )
                    )
                residual_severity = float(
                    learn_diag.get("residual_severity", float("inf"))
                )
                if residual_severity <= IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW:
                    warm.frozen_f_k_snapshot = [
                        float(wc.sim.context.getParameter(f"{self.prefix}_f_{k}"))
                        for k in range(K)
                    ]
                    # 🔑 [2026-09-14 / S1] **第 2 个入口**：fresh LEARN 收敛出来
                    # 的新候选。这正是要挡的那条 —— learning 再收敛到同一个点
                    # 时不许拿去重验（"反复试到偶然通过"）。
                    assert_candidate_not_sealed(
                        warm.frozen_f_k_snapshot, checkpoint_dir=self.checkpoint_dir,
                        window_idx=window_idx, lambdas_vdw=wc.lv_win,
                        stage_type=stage_type,
                    )
                    print(
                        f"    占据 raw_residual={residual_severity:.3f}≤"
                        f"{float(IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW):.2f} → 冻结跑 "
                        "local-MBAR Δf−ΔF 门（VALIDATE 是唯一生产入口证明，"
                        "不再要求固定 min_bias_updates 批数）"
                    )
                    warm.mode = "freeze_burn_in"
                    warm.have_frozen_once = True
                    warm.updates_since_freeze = 0
                    warm.freeze_burn_in_done = 0
                    warm.frozen_mbar_batches = []
                    warm.validation_solve_at_batches = IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
                    warm.validation_insufficient_attempts = 0
                    warm.validation_g_history = []
                    wc.sampler._last_dominant_k = None
                    wc.sampler.energy_buffer = []
                    wc.sampler.ema_mean_p = None
                continue

            if warm.mode == "freeze_burn_in":
                # 冻结 f_k 后构型要在这个固定 Hamiltonian 下重新平衡；之前在漂移
                # 偏置下攒的帧一律丢弃，不进 local MBAR 门统计。
                warm.freeze_burn_in_done += check_chunk
                wc.sampler.energy_buffer = []
                if warm.freeze_burn_in_done >= warm.frozen_burn_in_steps:
                    warm.mode = "validating"
                    warm.frozen_mbar_batches = []
                    warm.validation_solve_at_batches = IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
                    warm.validation_insufficient_attempts = 0
                    warm.validation_g_history = []
                    warm.validation_batch_count = 0
                    warm.validation_sample_count = 0
                    warm.validation_steps_this_freeze = 0
                continue

            # mode == "validating"：保持同一冻结 f_k，累计最近
            # IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES 批固定-f_k minibatch，拼成一次
            # 单参考 local MBAR，只设一个 loose gate：相邻态 ΔF^MBAR 与当前相邻
            # Δf_k 的最大偏差 < 阈值即冻结进生产（相邻差 gauge 无关，不比较带任意
            # 公共常数的绝对 f_k）。
            warm.validation_steps_this_freeze += check_chunk
            if len(wc.sampler.energy_buffer) < IBS_TMBAR_LEARNING_MINIBATCH_FRAMES:
                continue
            # 把这批固定-f_k 帧并入持久 tmbar_history（loose-gate 失败退回
            # learning 时，下一轮 update_weights 会复用它们继续学习），并取回
            # 同一份 (u_kn,bias_energies,base_energies) 数组喂 local MBAR——这三
            # 者跟 _solve_single_window_local_mbar 的入参一一对应（同 collect_
            # energies 记录的纯 softcore/base/bias，与 f_k 训练所用口径一致）。
            n_appended = wc.sampler._append_tmbar_batch_from_buffer()
            wc.sampler.energy_buffer = []
            if n_appended <= 0:
                continue
            warm.frozen_mbar_batches.append(wc.sampler.tmbar_history[-1])
            # 🔑 [2026-09-11] 只在**还没遇到数据不足**时保持"最近 5 批"的滑窗。
            # 一旦解出 insufficient_frames*，继续采样却仍然只留最近 5 批 = 新批
            # 进来旧批被挤掉，用于这次求解的总帧数根本不增长，相关时间长的窗口
            # 会永远不足。进入累计模式后不再裁剪，直到解出来或撞 MAX_BATCHES。
            if (
                warm.validation_insufficient_attempts == 0
                and len(warm.frozen_mbar_batches) > IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
            ):
                warm.frozen_mbar_batches = warm.frozen_mbar_batches[
                    -IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES:
                ]
            warm.validation_batch_count += 1
            warm.validation_sample_count += int(n_appended)
            if resumed_calibration_pending:
                # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 每批覆盖式落盘主
                # 窗口 checkpoint + 累计步数，抢占/撞墙被杀也能精确续算，不重跑
                # 也不漏算（沿用 v27 的每批落盘频率，只是判据换成 loose-gate）。
                _atomic_save_openmm_checkpoint(wc.sim, main_ckpt_path)
                _atomic_write_json(main_manifest_path, expected_main_manifest)
                wc.sampler.frozen_validation_cumulative_steps = (
                    prior_cumulative_steps + warm.steps_at_full_bias
                )
                wc.sampler.save_ibs_state(
                    ibs_state_file, wc.lc_win, wc.lv_win, stage_type=stage_type
                )
            # [Candidate-first, Validate-or-Learn v1] 攒满 IBS_LOCAL_MBAR_GATE_
            # SLIDING_BATCHES 批之前的读诊断早退：占据用跟 _diagnose_local_mbar_
            # situation 完全相同的公式/阈值判断是否已现塌陷迹象，若是则不等烧满
            # 全部 ~200 帧验证预算即可判定、退回 bounded-occupancy learning。
            # 纯路由决策——不设置 gate_ok/bias_converged，占据本身对生产入口门
            # 依旧只是诊断，不参与放行。
            # 🔑 但"不参与放行"必须是真的：早退每次都在完整 gate 求解（下面那次
            # _solve_single_window_local_mbar + validation_attempts += 1）之前
            # continue 回 learning，所以只要它每个 freeze cycle 都命中，本窗口就
            # 一次生产入口判据都评不到，占据事实上变成了唯一的放行门——正是 v29
            # 注释里"否则会把这个宽松 10 kJ/mol 门偷偷变成严格收敛门"要禁止的。
            # 实测 4/4 撞 IBSWarmupConvergenceError 的 vanishing 窗口 validation_
            # attempts≈0（其中一个 local_mbar_gate_history 为空），失败全部由这里
            # 产生，且失败后本该接手的 3-way router / damped 直接重验分支从未执行。
            # 因此本窗口拿到至少一次完整 gate 评估之前不允许早退：只多烧
            # (SLIDING_BATCHES-1) 批 minibatch 一次，换回"每个窗口至少被真正的
            # 判据评一次"。之后的 freeze cycle 仍照常享受这个早退优化。
            # 纯 warmup 路由/停止动力学，不动 production Hamiltonian、f_k 约定、
            # 采样方式或训练输入口径，故与 v28/v29 同类：协议版本保持 33、缓存兼容。
            if (
                warm.ever_completed_a_validate_attempt
                # 🔑 [2026-09-11] 正在为"数据不足"扩充验证数据时不得早退：
                # 早退会退回 learning 并清空批次，等于把刚攒的数据丢掉重来。
                and warm.validation_insufficient_attempts == 0
                and len(warm.frozen_mbar_batches) < IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
            ):
                _f_frozen_probe = np.asarray(
                    [
                        wc.sim.context.getParameter(f"{self.prefix}_f_{k}")
                        for k in range(K)
                    ],
                    dtype=np.float64,
                )
                _u_probe = np.concatenate(
                    [
                        np.asarray(b["u_kn"], dtype=np.float64)
                        for b in warm.frozen_mbar_batches
                    ],
                    axis=1,
                )
                _occ_probe = _softmax_occupancy_per_state(
                    _u_probe, _f_frozen_probe, wc.sampler.kt
                )
                _target_p_probe = 1.0 / float(K) if K > 0 else 0.0
                _sum_sq_probe = (
                    float(np.sum(np.square(_occ_probe))) if K > 0 else 0.0
                )
                _coverage_ess_probe = (
                    float(1.0 / _sum_sq_probe) if _sum_sq_probe > 0.0 else 0.0
                )
                _min_occ_probe = float(np.min(_occ_probe)) if K > 0 else 0.0
                _early_exit_collapsed = bool(
                    K > 0
                    and (
                        _coverage_ess_probe
                        < IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_COVERAGE_ESS_FRACTION
                        * float(K)
                        or _min_occ_probe
                        < IBS_LOCAL_MBAR_GATE_OCC_COLLAPSE_MIN_FRACTION
                        * _target_p_probe
                    )
                )
                if _early_exit_collapsed:
                    print(
                        "    ⟳ VALIDATE 早退：第 "
                        f"{len(warm.frozen_mbar_batches)}/{IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES} "
                        f"批已现占据塌陷迹象（coverage_ESS={_coverage_ess_probe:.2f}, "
                        f"min_occ={_min_occ_probe:.3f}），不等攒满全部验证预算即退回 "
                        "bounded-occupancy learning（纯路由决策，不参与 gate_ok）"
                    )
                    wc.sampler.last_update_diagnostics = {
                        "source": "validate_early_exit_occupancy_collapse",
                        "coverage_ess": _coverage_ess_probe,
                        "min_occ": _min_occ_probe,
                        "adjacent_delta_u_is_convergence_gate": False,
                    }
                    warm.last_failure_reason = "occupancy_collapse_early_exit"
                    warm.mode = "learning"
                    warm.validate_direct_retry_pending = False
                    warm.updates_since_freeze = 0
                    warm.frozen_mbar_batches = []
                    warm.validation_solve_at_batches = IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
                    warm.validation_insufficient_attempts = 0
                    warm.validation_g_history = []
                    wc.sampler._last_dominant_k = None
                    wc.sampler.ema_mean_p = None
                    wc.sampler.energy_buffer = []
                    validation_probability_sum = np.zeros(K, dtype=np.float64)
                    warm.validation_sample_count = 0
                    warm.validation_batch_count = 0
                    warm.validation_steps_this_freeze = 0
                    continue
            # 攒齐 validation_solve_at_batches 批再解一次（分级：5 → 10 → 15）。
            # 数据不足时不是每来一批就重复求解，而是跳到下一档。
            if len(warm.frozen_mbar_batches) < warm.validation_solve_at_batches:
                continue
            u_kn_gate = np.concatenate(
                [
                    np.asarray(b["u_kn"], dtype=np.float64)
                    for b in warm.frozen_mbar_batches
                ],
                axis=1,
            )
            bias_gate = np.concatenate(
                [
                    np.asarray(b["bias_energies"], dtype=np.float64)
                    for b in warm.frozen_mbar_batches
                ]
            )
            base_gate = np.concatenate(
                [
                    np.asarray(b["base_energies"], dtype=np.float64)
                    for b in warm.frozen_mbar_batches
                ]
            )
            f_frozen_now = np.asarray(
                [
                    wc.sim.context.getParameter(f"{self.prefix}_f_{k}")
                    for k in range(K)
                ],
                dtype=np.float64,
            )
            # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] f_frozen_now 上移到这里：混合
            # 覆盖度 ESS 需要"这批 minibatch 采样时真正生效的 f_k"才能把共模
            # 因子除干净（gauge-free 的逐帧 softmax 捷径在谱宽大的窗口会给出
            # 差数十倍的结果，见 _ibs_reweighting_quality_diagnostics）。这批
            # batch 是在冻结 f_k 下采的，所以此刻读到的就是采样时的值。
            # 🔑 [0831issue / ESS_GATE_PROTOCOL_VERSION=4] 这里**不**需要传
            # sampling_kj：u_kn_gate 来自 tmbar_history entry，而
            # IBS_BIAS_PROTOCOL_VERSION=32 之后那些 entry 的 u_kn 存的已经是
            # 未偏移的 sampling_state_energies（见
            # _append_tmbar_batch_from_buffer），所以 u_kn 与 f_frozen_now 本来
            # 就同口径。生产期那两个调用点用的是物理 energy_history，才必须显式
            # 传 sampling_kj。
            gate_mbar = _solve_single_window_local_mbar(
                u_kn_gate,
                bias_gate,
                base_gate,
                list(range(K)),
                wc.sampler.kt,
                f_k=f_frozen_now,
                sampled_distribution_row=0,
                w_idx=window_idx,
            )
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 生产入口门 = 纯 Δf−ΔF loose gate：
            # 只要 local MBAR 解出有限 f_mbar，就算相邻态 |Δf_k−ΔF^MBAR|（gauge
            # 无关），max < IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL 即冻结进
            # 生产。abs_ess / 占据平坦度 / coverage_ESS / raw_residual / tmbar_self_
            # consistent 全部只作诊断，不参与放行——否则会把这个宽松 10 kJ/mol 门
            # 偷偷变成严格收敛门（挡住 Δf 其实已在 10 kJ/mol 内、只是短 warmup ESS
            # 薄的窗口）。前面那种 908 kJ/mol 的零重叠外推自然因 908>10 被拒，不需要
            # abs_ess 门才能拒。最终绝对 ESS/误差/真实自由能交生产后独立
            # _assert_stage_result_sane + 最终 MBAR 兜底。
            gate_solver_error = gate_mbar.get("error")
            # 以下三个只作诊断（不参与放行）。
            _gate_min_ess_ratio = gate_mbar.get("min_ess_ratio")
            _gate_n_used = int(gate_mbar.get("n_frames_used", 0) or 0)
            _gate_abs_ess = (
                float(_gate_min_ess_ratio) * _gate_n_used
                if (_gate_min_ess_ratio is not None
                    and np.isfinite(_gate_min_ess_ratio))
                else 0.0
            )
            try:
                gate_situation = _diagnose_local_mbar_situation(
                    u_kn_gate, f_frozen_now, wc.sampler.kt, gate_mbar, int(start)
                )
            except Exception as _situ_err:
                gate_situation = {"error": repr(_situ_err)}
            max_adjacent_gap_kJ_mol = float("inf")
            adjacent_gaps = None
            adjacent_gaps_signed = None
            cumulative_residual = None
            cumulative_span = None
            cumulative_span_sigma = None
            cumulative_span_verdict = None
            gate_error = gate_solver_error
            if gate_solver_error is None:
                f_mbar = np.asarray(gate_mbar.get("f"), dtype=np.float64)
                if f_mbar.size == K and np.all(np.isfinite(f_mbar)):
                    if K > 1:
                        df_current = np.diff(f_frozen_now)
                        dF_mbar = np.diff(f_mbar)
                        # 🔑 [2026-09-11] **带符号残差必须一并落盘。**
                        # 绝对值只够判"局部最大误差"这道门；判不了老板要的
                        # "**端到端累计偏差**" —— 那要求知道逐边残差是否**同号**。
                        # 实测 win3 逐边 |Δ| = [2.897, 3.634, 4.548]，每条都远低于
                        # 门槛 10，但窗内和 11.08 **已超门槛本身**；而 11.08 只是
                        # "若同号"的上界，没有符号就没法确认。
                        adjacent_gaps_signed = df_current - dF_mbar
                        adjacent_gaps = np.abs(adjacent_gaps_signed)
                        max_adjacent_gap_kJ_mol = float(np.max(adjacent_gaps))
                        # 🔑🔑 [2026-09-11 老板规格] **端到端累计偏差**。
                        #
                        # C_j = Σ_{i<j} r_i 而 r_i = (f_{i+1}−f_i)−(F̂_{i+1}−F̂_i)
                        # ⟹ 望远镜求和后 **C_j = (f_j − f_0) − (F̂_j − F̂_0)**。
                        # f 是**冻结**的（无误差），所以
                        #     σ(C_j) = σ(F̂_j − F̂_0) = ddf_phys[0, j]
                        # **不需要任何误差传播**：直接读同一次 MBAR 拟合的矩阵元，
                        # 相邻差累加起来误差**不是**独立相加（有协方差）。
                        #
                        # 门看 **span = max(C) − min(C)**，因为它 **gauge-free**、
                        # 不依赖选哪个端点作锚；**只看终点会漏掉"中间先偏后抵消"**。
                        #
                        # 实测（生产帧，四窗**全部同号、零抵消**，Σ|r| = |Σr|）：
                        #   win0 span 7.19  win1 6.52  win2 3.30  win3 **11.92**
                        # win3 = 4.81 kT，e^4.81≈123，而它末/首支撑比 0.019≈1/53
                        # —— 同一量级。而每条边 |r_i| 最大才 4.15，
                        # **现有 max adjacent < 10 的门完全看不见它**。
                        _c_cum = np.concatenate(
                            ([0.0], np.cumsum(adjacent_gaps_signed))
                        )
                        cumulative_residual = _c_cum
                        # [2026-09-14 审计 #51] 这里原来有一份**独立拷贝**的
                        # span 判据（旧刻度：span ± 2σ_预选对 vs 10 kJ/mol），
                        # 与 `cumulative_fk_residual` 那份会对同一批数给出不同
                        # 结论、又都落进报告 ⟹ 读的人只能猜哪个对。现在两处共用
                        # `cumulative_fk_span_verdict`，口径逐字相同。
                        cumulative_span_verdict = cumulative_fk_span_verdict(
                            _c_cum,
                            gate_mbar.get("ddf_physical_matrix_kJ_mol"),
                            float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL),
                        )
                        cumulative_span = cumulative_span_verdict["span_kJ_mol"]
                        # 预选对 σ：只作报告，已不参与判据。
                        cumulative_span_sigma = cumulative_span_verdict[
                            "sigma_preselected_pair_kJ_mol"
                        ]
                    else:
                        max_adjacent_gap_kJ_mol = 0.0
                else:
                    gate_error = "nan_or_shape_mismatch_in_local_mbar_f"
            gate_ok = bool(
                gate_error is None
                and np.isfinite(max_adjacent_gap_kJ_mol)
                and max_adjacent_gap_kJ_mol
                < float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL)
            )
            gate_diag = {
                "phase": "frozen_local_mbar_loose_gate",
                "n_batches": int(len(warm.frozen_mbar_batches)),
                "max_adjacent_delta_kJ_mol": (
                    max_adjacent_gap_kJ_mol
                    if np.isfinite(max_adjacent_gap_kJ_mol) else None
                ),
                "adjacent_delta_kJ_mol": (
                    [float(x) for x in adjacent_gaps]
                    if adjacent_gaps is not None else None
                ),
                # **带符号**的 Δf_k − ΔF^MBAR，用于端到端累计偏差判读：
                # 同号累积 ⟹ 门只约束局部最大误差、缺累计偏差门；
                # 符号交替但转移单向 ⟹ 动力学/初始化方向性。两者处置不同。
                "adjacent_delta_signed_kJ_mol": (
                    [float(x) for x in adjacent_gaps_signed]
                    if adjacent_gaps_signed is not None else None
                ),
                "adjacent_delta_signed_sum_kJ_mol": (
                    float(np.sum(adjacent_gaps_signed))
                    if adjacent_gaps_signed is not None else None
                ),
                "adjacent_delta_abs_sum_kJ_mol": (
                    float(np.sum(adjacent_gaps))
                    if adjacent_gaps is not None else None
                ),
                # ---- 端到端累计偏差（老板规格 v1）----
                "signed_adjacent_residual_kJ_mol": (
                    [float(x) for x in adjacent_gaps_signed]
                    if adjacent_gaps_signed is not None else None
                ),
                "signed_cumulative_residual_from_first_kJ_mol": (
                    [float(x) for x in cumulative_residual]
                    if cumulative_residual is not None else None
                ),
                "cumulative_residual_span_kJ_mol": cumulative_span,
                "cumulative_residual_span_sigma_kJ_mol": cumulative_span_sigma,
                "endpoint_signed_residual_kJ_mol": (
                    float(cumulative_residual[-1])
                    if cumulative_residual is not None else None
                ),
                # 三档：阈值仍是同一个数，但 [2026-09-14 审计 #51] 起卡的是
                # **标准化 span**（span / 零假设期望），因为零假设下 E[span]
                # 随边数增长而门槛刻意 K-无关。判据本体在
                # `cumulative_fk_span_verdict`，与离线那份逐字同一实现。
                "cumulative_residual_verdict": (
                    None if cumulative_span_verdict is None
                    else cumulative_span_verdict["verdict"]
                ),
                "cumulative_residual_threshold_kJ_mol": float(
                    IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL
                ),
                "cumulative_residual_threshold_applies_to": (
                    "cumulative_residual_span_standardized"
                ),
                "cumulative_residual_span_standardized": (
                    None if cumulative_span_verdict is None
                    else cumulative_span_verdict["span_standardized"]
                ),
                "cumulative_residual_sigma_adj_kJ_mol": (
                    None if cumulative_span_verdict is None
                    else cumulative_span_verdict["sigma_adj_kJ_mol"]
                ),
                "cumulative_residual_n_edges": (
                    None if cumulative_span_verdict is None
                    else cumulative_span_verdict["n_edges"]
                ),
                "cumulative_residual_span_null_expectation_kJ_mol": (
                    None if cumulative_span_verdict is None
                    else cumulative_span_verdict["null_expectation_kJ_mol"]
                ),
                "cumulative_residual_span_band_standardized": (
                    None if cumulative_span_verdict is None
                    else cumulative_span_verdict["band_standardized"]
                ),
                "cumulative_residual_unmeasured_reason": (
                    None if cumulative_span_verdict is None
                    else cumulative_span_verdict["unmeasured_reason"]
                ),
                # 🔑🔑 [2026-09-11 老板要求] **累计 f_k 残差的口径必须明写。**
                #
                # 这里的 u_kn_gate 来自 tmbar_history entry，存的是**未偏移的
                # sampling_state_energies**（见上方调用点注释），与
                # f_frozen_now 同口径 ⟹ `sampling_states`。**这是正确的那个。**
                #
                # ⚠️ **绝不能用 `energies`**：它比 sampling_states 多一个**逐 λ 态
                # 常数**（LRC 长程尾项，实测 std 精确为 0）。逐态常数本身无害，
                # 但它的**逐边差是正的、单调递减** ⟹ 会**凭空伪造出**"全负号、
                # 单调递增"的残差形状。实测污染幅度：win0 的 span 被抬高 2.2 倍
                # （3.27 → 7.19），win3 几乎没变（11.47 → 11.92，因为那段 LRC
                # 边差本来就小）—— 于是还会伪造出"span 越大末/首比越小"的
                # 假单调关系。
                #
                # ⚠️ 注意与 `N_eff/g` 的区别：**支撑量不受影响**（逐态常数在
                # 归一化里被除掉，两口径逐位相同）。**只有 f_k 残差这一类会被污染。**
                "energy_gauge": "sampling_states",
                "energy_gauge_required": "sampling_states",
                "energy_gauge_note": (
                    "累计 f_k 残差**只能**用 sampling_states。energies 多一个逐 λ 态"
                    "常数（LRC），其逐边差会伪造出单调同号的残差形状。"
                    "支撑量（N_eff/g）不受影响，两口径逐位相同。"
                ),
                "adjacent_delta_all_same_sign": (
                    bool(np.all(adjacent_gaps_signed > 0)
                         or np.all(adjacent_gaps_signed < 0))
                    if (adjacent_gaps_signed is not None
                        and np.size(adjacent_gaps_signed)) else None
                ),
                "gate_threshold_kJ_mol": float(
                    IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL
                ),
                "error": gate_error,
                "steps_at_full_bias": int(warm.steps_at_full_bias),
                "validation_sample_count": int(warm.validation_sample_count),
                # 以下仅诊断，不参与放行：
                "n_frames_used": int(_gate_n_used),
                "min_ess_ratio": _gate_min_ess_ratio,
                "min_absolute_ess": float(_gate_abs_ess),
                "statistical_inefficiency": gate_mbar.get("statistical_inefficiency"),
                "situation": gate_situation,
                "diagnostics_only_note": (
                    "abs_ess/coverage_ess/occupancy/raw_residual do NOT gate "
                    "production entry; only max_adjacent_delta < threshold does"
                ),
            }
            warm.local_mbar_gate_history.append(gate_diag)
            warm.last_local_mbar_gate = gate_diag
            warm.lse_residual_history.append(gate_diag)
            # loose-gate 不用占据概率判据；last_validation_batch_p 仅用于失败/
            # 收敛报告兜底展示，保持 None（下游报告退回 EMA/nan）。
            warm.last_validation_batch_p = None
            # [Candidate-first, Validate-or-Learn v1] This is a real,
            # completed local-MBAR gate evaluation (pass or fail) --
            # counted once here regardless of outcome, distinct from the
            # cheap early-exit probe above which never reaches this line.
            # 🔑 [2026-09-11] 只有**真的解出** Δf−ΔF 才算"完整评估过一次"。
            # 原来无论 gate_error 是什么都置 True，于是第一次
            # insufficient_frames_after_decorrelation（根本没求出数）之后，
            # 占据早退在后续每个 freeze cycle 又重新可用，本窗口可以一路早退、
            # 永远拿不到一次真正的判据评估 —— 正是这道守卫本来要堵的事。
            # validation_attempts 保持原语义（含求解失败的尝试），只改这个标志。
            if np.isfinite(max_adjacent_gap_kJ_mol):
                warm.ever_completed_a_validate_attempt = True
            wc.sampler.validation_attempts = (
                int(getattr(wc.sampler, "validation_attempts", 0)) + 1
            )
            if gate_ok:
                warm.validation_pass_count = int(warm.validation_batch_count)
                warm.bias_converged = True
                warm.last_failure_reason = None
                print(
                    "    [OK] local-MBAR loose gate 通过："
                    f"max|Δf_k−ΔF^MBAR|={max_adjacent_gap_kJ_mol:.3f} kJ/mol < "
                    f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f} kJ/mol"
                    f"（{warm.validation_sample_count} frames；abs_ess={_gate_abs_ess:.1f}、"
                    f"min_ess_ratio={_gate_min_ess_ratio} 仅诊断，不参与放行）；"
                    "冻结 f_k 进生产，最终绝对 ESS/误差/自由能交生产后 MBAR"
                )
                break

            # 🔑 [2026-09-11] 四路分诊。以前这里只有两路（解出没过 / "暂不可解"），
            # 后者把"结构性输入错误"和"根本没测出来"一起塞进 learning，于是
            #   · 输入/能量身份不一致被当成暂时现象，一直重试到预算耗尽；
            #   · 数据不足被当成 f_k 的问题，清空批次退回 learning，下个周期
            #     重新攒同样大小的数据集，大概率再次不足；最终以
            #     f_not_converged 收场，还可能触发插 λ —— 把误判推迟而已。
            _gate_error_class = classify_local_mbar_gate_error(gate_error)
            if _gate_error_class == "input_identity":
                raise RuntimeError(
                    f"窗口 {window_idx} 的 local-MBAR 验证门收到**结构性**错误："
                    f"{gate_error}。这是输入/能量身份不一致，不是数据不足，"
                    "继续采样不会让它消失；拒绝当成'暂不可解'退回 learning 烧完预算。"
                )
            if _gate_error_class == "insufficient_data":
                # 数据不足只说明**目前无法判断 f_k**：既没有理由修改它，也没有
                # 依据插 λ。保持同一份冻结 f_k、保留已有批次继续累计，到下一档
                # 再解一次；不清空、不退回 learning、不计入
                # learning_to_validation_cycles（那是"退回 learning 的次数"）。
                # 相关性/有效样本量由求解器在**拼接后的整份**固定-f_k 数据集上
                # 重新估计（_decorrelate_by_worst_target_state 作用在
                # np.concatenate 之后的数组上），不是把每批 ESS 相加。
                warm.validation_insufficient_attempts += 1
                _g_now = gate_mbar.get("statistical_inefficiency")
                if _g_now is not None:
                    warm.validation_g_history.append(float(_g_now))
                _n_now = len(warm.frozen_mbar_batches)
                # 🔑 [2026-09-11] **不做"没救了"的提前外推。** ESS 本来就跟采样
                # 量走：n_eff = N/g，而 g 只有在 N ≫ τ 之后才稳定。实测
                # window 4 的 g 还在随 N 涨（200 帧 38.5 → 400 帧 66.9），说明
                # 估计器远没饱和 —— 在这个区间拿 n_eff 线性外推去判"再攒也不够"
                # 是无效的，只会把本来能测出来的窗口提前判死。
                # 该做的就是按预算老老实实攒到上限（5 → 10 → 15 批）。
                if _n_now >= IBS_LOCAL_MBAR_GATE_MAX_BATCHES:
                    # 🔑 [2026-09-11] 这份 f_k 上一次 attempt 已经**真正通过过**
                    # loose gate（bias_status=converged，走的是 skip_warmup_
                    # entirely 复验分支）。本次复验的目的只是"别零平衡就进生产"，
                    # burn-in 已经跑过了。测不出来**不是反面证据** —— 不得据此把
                    # 一个已验证的窗口降级，更不得中止整个 run 把已经采到的帧作废。
                    if skip_warmup_entirely:
                        warm.bias_converged = True
                        warm.last_failure_reason = None
                        print(
                            f"    [复验未能测出] 窗口 {window_idx} 是 resume 前就已经"
                            f"通过验证的窗口（bias_status=converged）。本次复验在 "
                            f"{_n_now} 批 / {warm.validation_sample_count} frames 下仍然 "
                            f"{gate_error}（g="
                            f"{gate_mbar.get('statistical_inefficiency')}）：**没测出来，"
                            "不是测出来不合格**。保持原有 f_k 与已验证结论进生产，"
                            "burn-in 已完成、不是零平衡。真实 ESS/误差/自由能仍交生产后 "
                            "MBAR 与 _assert_stage_result_sane 把关。"
                        )
                        break
                    warm.validation_indeterminate_diag = {
                        "reason": "validation_budget_exhausted_indeterminate",
                        "n_batches": int(_n_now),
                        "max_batches": int(IBS_LOCAL_MBAR_GATE_MAX_BATCHES),
                        "validation_sample_count": int(warm.validation_sample_count),
                        "insufficient_attempts": int(warm.validation_insufficient_attempts),
                        "last_gate_error": str(gate_error),
                        "statistical_inefficiency": gate_mbar.get(
                            "statistical_inefficiency"
                        ),
                        # 可达性预检要的是**多个检查点**，不是单点：单点在小样本
                        # 下会误杀。嵌套样本（400 帧含 200 帧）不满足 bootstrap
                        # 的独立性前提，所以下界取最小值而不是 bootstrap。
                        "validation_g_history": [
                            float(x) for x in warm.validation_g_history
                        ],
                        # 🔑 T = 触发 `insufficient_frames_after_decorrelation`
                        # 的那个下限，也就是 `_solve_single_window_local_mbar`
                        # 的 `min_frames`（默认 10）。
                        # ⚠️ **不是** `minimum_complete_validation_frames`（200）——
                        # 那管的是"这次 attempt 攒够**原始**帧没有"，是完整性
                        # 要求，跟去相关下限是两回事。混掉会把 gcrit 算小 20 倍，
                        # 把一个只差 26% 帧数的窗口判成"差 7.5 倍不可达"。
                        "decorrelated_frames_required": int(
                            IBS_LOCAL_MBAR_GATE_MIN_FRAMES
                        ),
                        "n_frames_used": int(_gate_n_used),
                        "steps_at_full_bias": int(warm.steps_at_full_bias),
                        "frozen_f_k_kJ_mol": [float(x) for x in f_frozen_now],
                    }
                    print(
                        f"    [无法判定] 单周期验证预算用尽（{_n_now}/"
                        f"{IBS_LOCAL_MBAR_GATE_MAX_BATCHES} 批、"
                        f"{warm.validation_sample_count} frames）仍然 {gate_error}："
                        "Δf−ΔF 从未被求出，对这份 f_k **无结论**。不改 f_k、"
                        "不退回 SGD、不插 λ；冻结状态与数据保留，交上层决定是否"
                        "延长预算。"
                    )
                    # 🔑🔑 [2026-09-15] **调用方授权过 ⟹ 走临时生产，不抛路由信号。**
                    # 「没测出来」不是「不合格」：这份 f_k 没有任何反面证据。
                    # 在没有这条路径时，这样的窗口进不了生产（引擎抛
                    # IBSValidationBudgetIndeterminateError）、又拿不到新证据
                    # （补帧动作在构造上推不动它）⟹ 整条流水线停在
                    # NO_FEASIBLE_ACTION（真机 cyclod_ligand2/rep2 win4）。
                    # ⚠️ 这里只改**走向**，一个判据都没放宽：
                    #   · `validation_indeterminate_diag` 照样留着当证据；
                    #   · `f_k_evidence_status` 在下面保持 indeterminate；
                    #   · 真正的验收仍是生产后的支撑门 + 最终 MBAR。
                    # ⚠️ 块大小由调用方的 `production_step_overrides` 决定
                    #    （策略是**一块** +250k），引擎不在这里发明预算。
                    if provisional_authorized:
                        warm.provisional_production = True
                        warm.bias_converged = True
                        warm.last_failure_reason = None
                        print(
                            f"    [临时生产] 窗口 {window_idx} 已被调用方显式授权："
                            "用这份**未经验证**的冻结 f_k 进一块诊断生产。"
                            "**这不是可信 PASS** —— f_k 证据保持 `indeterminate`，"
                            "warmup_failure.json 作为证据保留，"
                            "能否采用交生产后的 min N_eff/g 支撑门与最终 MBAR 裁决。"
                        )
                    break
                warm.validation_solve_at_batches = min(
                    IBS_LOCAL_MBAR_GATE_MAX_BATCHES,
                    _n_now + IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES,
                )
                print(
                    f"    ⏸ 验证数据不足（{gate_error}，{_n_now} 批 / "
                    f"{warm.validation_sample_count} frames，g="
                    f"{gate_mbar.get('statistical_inefficiency')}）："
                    "**没测出来 ≠ f_k 错了**。保持冻结 f_k 与已有批次继续累计到 "
                    f"{warm.validation_solve_at_batches} 批再解一次（单周期上限 "
                    f"{IBS_LOCAL_MBAR_GATE_MAX_BATCHES} 批，另受 warmup 总步数封顶）。"
                )
                continue

            # 未过（gap≥阈值，或 MBAR 解出 NaN）。abs_ess/占据只在现场里打印，
            # 不参与判定——gate_ok 的定义完全不变。
            warm.learning_to_validation_cycles += 1
            if gate_error is None:
                print(
                    "    ⟳ local-MBAR loose gate 未过："
                    f"max|Δf_k−ΔF^MBAR|={max_adjacent_gap_kJ_mol:.3f} kJ/mol ≥ "
                    f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f}；"
                    "现场："
                    f"{_format_local_mbar_situation(gate_situation)}"
                )
            else:
                print(
                    f"    ⟳ local MBAR 暂不可解（{gate_error}）；不作为 f_k 的反面证据，"
                    "继续 bounded-occupancy learning。现场："
                    f"{_format_local_mbar_situation(gate_situation)}"
                )
            wc.sampler.last_update_diagnostics = {
                "source": "failed_local_mbar_loose_gate",
                "local_mbar_gate": gate_diag,
                "adjacent_delta_u_is_convergence_gate": False,
            }
            # [Candidate-first, Validate-or-Learn v1] 3-way VALIDATE-failure
            # router. occupancy_collapsed/gate_error remain diagnostics-only
            # for gate_ok itself (unchanged above) -- they only choose the
            # recovery path here:
            #   (i)   occupancy_collapsed -> bounded-occupancy LEARN.
            #   (ii)  occupancy OK, gate_error is None, gap failed, no
            #         retry spent yet this freeze cycle -> ONE damped +
            #         pairwise-capped correction using this attempt's
            #         local-MBAR candidate, then re-VALIDATE directly
            #         (skip LEARN for this one retry).
            #   (iii) a second consecutive gap failure after that one
            #         retry, or gate_error is not None (local-MBAR
            #         unsolvable, not evidence against f_k) -> LEARN.
            _occ_collapsed_now = bool(gate_situation.get("occupancy_collapsed"))
            if _occ_collapsed_now:
                warm.last_failure_reason = "occupancy_collapse"
                warm.mode = "learning"
                warm.validate_direct_retry_pending = False
            elif gate_error is None and not warm.validate_direct_retry_pending:
                f_mbar_retry = np.asarray(gate_mbar.get("f"), dtype=np.float64)
                f_damped, _damp_diag = wc.sampler._damped_tmbar_absolute_update(
                    f_frozen_now, f_mbar_retry, damping=IBS_TMBAR_UPDATE_DAMPING
                )
                f_capped, _cap_diag = wc.sampler._apply_pairwise_cap(
                    f_frozen_now,
                    f_damped,
                    IBS_MAX_APPLIED_PAIRWISE_STEP_KT,
                    wc.sampler.kt,
                )
                for k in range(K):
                    wc.sim.context.setParameter(f"{self.prefix}_f_{k}", float(f_capped[k]))
                warm.frozen_f_k_snapshot = f_capped.astype(float).tolist()
                # 🔑 [2026-09-14 / S1] **第 3 个入口**：阻尼+pairwise-capped 修正
                # 后直接重验的那份候选。修正量可能小到落回已封存的那一份里，
                # 这道门同样要过。
                assert_candidate_not_sealed(
                    warm.frozen_f_k_snapshot, checkpoint_dir=self.checkpoint_dir,
                    window_idx=window_idx, lambdas_vdw=wc.lv_win,
                    stage_type=stage_type,
                )
                print(
                    "    occupancy 尚可但 local-MBAR gap 未过：对冻结 f_k 应用一次"
                    "阻尼+pairwise-capped 修正（用本次 local-MBAR 候选），直接重新"
                    "验证（跳过 learning，本轮仅此一次）"
                )
                warm.last_failure_reason = "local_mbar_gap_exceeded_retry_applied"
                warm.validate_direct_retry_pending = True
                warm.mode = "freeze_burn_in"
            elif gate_error is None:
                warm.last_failure_reason = "local_mbar_gap_exceeded_after_retry"
                warm.mode = "learning"
                warm.validate_direct_retry_pending = False
            else:
                warm.last_failure_reason = "local_mbar_unsolvable"
                warm.mode = "learning"
                warm.validate_direct_retry_pending = False
            warm.updates_since_freeze = 0
            warm.freeze_burn_in_done = 0
            warm.frozen_mbar_batches = []
            warm.validation_solve_at_batches = IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
            warm.validation_insufficient_attempts = 0
            warm.validation_g_history = []
            wc.sampler._last_dominant_k = None
            wc.sampler.ema_mean_p = None
            wc.sampler.energy_buffer = []
            validation_probability_sum = np.zeros(K, dtype=np.float64)
            warm.validation_sample_count = 0
            warm.validation_batch_count = 0
            warm.validation_steps_this_freeze = 0
            continue

    def _summarize_warmup(
        self,
        *,
        K,
        _attempt_ledger,
        _ledger,
        _nan_diagnostics,
        accept_recalibrated_f_k_without_gate,
        allow_best_effort_warmup,
        check_chunk,
        coverage_ess_threshold,
        effective_frozen_validation_budget,
        effective_mbar_calibration_reserved_steps,
        end,
        expected_main_manifest,
        frozen_validation_reserved_steps,
        full_bias_step_budget,
        ibs_state_file,
        ibs_wrap,
        k,
        wc,
        lse_log_residual_tolerance,
        main_ckpt_path,
        main_manifest_path,
        max_bias_updates,
        min_bias_updates,
        min_probability_threshold,
        minimum_complete_validation_frames,
        required_consecutive_bias_updates,
        resume_frozen_validation,
        resumed_calibration_pending,
        sgd_step_budget,
        stage_type,
        start,
        target_p,
        validation_attempt_budget_steps,
        warm,
        warmup_only,
        window_idx,
    ):
        """相位 4：预热收口/诊断—— 刀1 第二刀·刀c 自 run_all_windows 原样搬出。

        纯搬运（verify_extract --block for --slice 逐节点等价）；跨界变量按
        原名作 keyword-only 形参（CUTTING_PLAN.md §2.2.1）。
        """
        print(
            "    [warmup 计时] "
            + ", ".join(f"{k}={v:.1f}s" for k, v in warm.warmup_timers.items())
        )

        # [IBS_BIAS_PROTOCOL_VERSION=29] 跑满整个 warmup 步数预算仍没有任何一次
        # local-MBAR loose gate < 阈值时，直接接受当前 f_k 进生产（best-effort）。
        # loose gate 只是"别让某个局部边完全饿死"的粗门（10 kJ/mol≈4 kT）；真正的
        # 自由能/ESS/overlap/误差全部交给生产后独立的 _assert_stage_result_sane +
        # 最终 MBAR。resumed_calibration_pending 续验同样走这条（不再有阶梯升级/
        # 无限续验：判据已换成 loose gate，预算耗尽即接受）。接受的就是循环退出
        # 时 Context 里的这份 f_k，不需要恢复更早的 attempt，也不需要额外 re-burn-in。
        # 条件用 `not bias_converged` 覆盖两种退出：撞总步数帽，或反复未过 gate
        # 把 learning 更新次数烧到 max_bias_updates 而 break——两者都算预算耗尽。
        # 但 warmup_only 是设计期验证模式，本就不进生产、其职责是如实报告窗口是
        # 否收敛，故不 best-effort 接受——让它照常落到下面 else 分支报未收敛。
        if not warm.bias_converged and not warmup_only and allow_best_effort_warmup:
            accepted_f_k = np.asarray(
                [
                    wc.sim.context.getParameter(f"{self.prefix}_f_{k}")
                    for k in range(K)
                ],
                dtype=np.float64,
            )
            warm.frozen_f_k_snapshot = accepted_f_k.astype(float).tolist()
            warm.validation_pass_count = int(warm.validation_batch_count)
            warm.bias_converged = True
            warm.best_effort_acceptance = True
            warm.best_effort_acceptance_reason = "warmup_budget_exhausted_loose_gate"
            warm.best_effort_residual = (
                float(warm.last_local_mbar_gate["max_adjacent_delta_kJ_mol"])
                if (
                    warm.last_local_mbar_gate is not None
                    and warm.last_local_mbar_gate.get("max_adjacent_delta_kJ_mol")
                    is not None
                )
                else float("inf")
            )
            warm.mode = "best_effort_budget_exhausted_accepted"
            _last_gap_repr = (
                f"{warm.best_effort_residual:.3f} kJ/mol"
                if np.isfinite(warm.best_effort_residual)
                else "n/a（local MBAR 从未在预算内解出）"
            )
            print(
                "    warmup 步数预算耗尽仍未通过 local-MBAR loose gate；接受当前 "
                f"f_k 进生产（最近一次 max|Δf_k−ΔF^MBAR|={_last_gap_repr}，阈值 "
                f"{float(IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL):.1f} kJ/mol）；"
                "生产后独立质量门 + 最终 MBAR 继续兜底"
            )
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 现场判决必须区分两种"预算耗尽"：
            # (a) 占据真的塌了（某态低于 0.5/K）→ 硬热力学瓶颈，权重救不了，建议
            #     插 λ/拆窗；(b) 占据其实平坦、只是短 warmup 的 local MBAR 绝对
            #     ESS/端点 reweight ESS 偏低 → 不是瓶颈，已接受进生产，最终 ESS/
            #     误差交生产后 MBAR。之前只凭"MBAR 没解出可信 ΔF"就一律报瓶颈，会
            #     把 (b)（如占据 [0.41,0.27,0.20,0.125] 这种健康窗口）误报成需要插 λ。
            _acc_situation = (
                warm.last_local_mbar_gate.get("situation")
                if warm.last_local_mbar_gate is not None else None
            )
            _never_trustworthy = (
                warm.last_local_mbar_gate is None
                or warm.last_local_mbar_gate.get("max_adjacent_delta_kJ_mol") is None
            )
            if _acc_situation:
                print(
                    f"    现场：{_format_local_mbar_situation(_acc_situation)}"
                )
            _situ_ok = bool(_acc_situation and not _acc_situation.get("error"))
            if _situ_ok and _acc_situation.get("occupancy_is_flat"):
                # (b) 占据平坦：良性，不是插 λ 的场景。
                print(
                    "    [OK] 判决：冻结占据已平坦（coverage_ESS="
                    f"{_acc_situation.get('coverage_ess', float('nan')):.2f}≥"
                    f"{IBS_LOCAL_MBAR_GATE_OCC_MIN_COVERAGE_ESS_FRACTION * float(K):.2f}），"
                    "只是短 warmup 的 local MBAR 绝对 ESS/端点 reweight ESS 偏低、未能走 "
                    "Δf−ΔF 主门；已接受进生产，最终绝对 ESS/误差/自由能交生产后 "
                    "_assert_stage_result_sane + 最终 MBAR。这不是需要插 λ 的瓶颈；"
                    "若最终 MBAR 显示某端点采样不足，再考虑延长生产采样。"
                )
            elif _never_trustworthy and _situ_ok and _acc_situation.get(
                "occupancy_collapsed"
            ):
                # (a) 占据真的塌陷：硬瓶颈。
                _sg = _acc_situation.get("starved_global_state")
                _edges = _acc_situation.get("starved_edges_global") or []
                print(
                    "    [WARN] 判决：占据塌陷 + local MBAR 全程重叠/ESS 不足 → 这是"
                    f"{'端点' if _acc_situation.get('is_endpoint') else '内部'}"
                    f"热力学瓶颈（global s{_sg} 占据="
                    f"{_acc_situation.get('starved_occupancy'):.2e} 低于健康下限 "
                    f"{_acc_situation.get('occupancy_floor'):.3f}，与相邻态零重叠），"
                    f"权重无法弥补物理缺口。建议在相邻 global 态之间插入 λ 或拆窗："
                    f"边 {_edges}。（生产后 _assert_stage_result_sane 亦会独立判不可用。）"
                )
            elif _never_trustworthy and _situ_ok:
                # 中间态：占据没塌到下限、但也没达到平坦兜底门（如 coverage 略低）。
                print(
                    "    判决：占据尚可但未达平坦兜底门、local MBAR 绝对 ESS 也不足；"
                    "已接受进生产（best-effort）。优先延长生产采样；若最终 MBAR 仍显示"
                    "某相邻边重叠不足，再考虑插 λ。现场见上。"
                )

        state = wc.sim.context.getState(getEnergy=True)
        if not np.isfinite(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)):
            raise RuntimeError("偏置预热阶段在 scale=1.0 时能量 NaN")

        # 报告用的 min/max/coverage 优先取最后一次真正驱动收敛判定的原始
        # validation batch；只有从未进入过 validating（例如始终卡在
        # learning）时才退回 EMA，仅供参考，不代表任何"已验证"的结论。
        if warm.last_validation_batch_p is not None:
            _report_p = np.asarray(warm.last_validation_batch_p, dtype=np.float64)
            min_ema_p = float(np.min(_report_p))
            max_ema_p = float(np.max(_report_p))
            warm.ema_mean_p_values = warm.last_validation_batch_p
            coverage_ess = float(1.0 / np.sum(np.square(_report_p)))
        else:
            max_ema_p = float(np.max(wc.sampler.ema_mean_p)) if wc.sampler.ema_mean_p is not None else float("nan")
            min_ema_p = float(np.min(wc.sampler.ema_mean_p)) if wc.sampler.ema_mean_p is not None else float("nan")
            warm.ema_mean_p_values = (
                np.asarray(wc.sampler.ema_mean_p, dtype=float).tolist()
                if wc.sampler.ema_mean_p is not None
                else []
            )
            coverage_ess = (
                float(1.0 / np.sum(np.square(np.asarray(wc.sampler.ema_mean_p, dtype=np.float64))))
                if wc.sampler.ema_mean_p is not None
                else float("nan")
            )
        report_lse_balance = ibs_lse_balance_diagnostics(warm.ema_mean_p_values)
        warm.bias_warmup_diag = {
            "status": (
                (
                    "frozen_validation_best_effort_accepted"
                    if warm.best_effort_acceptance
                    else "frozen_validation_converged"
                )
                if warm.bias_converged
                else (
                    "early_probe_triggered_unconverged"
                    if warm.early_probe_triggered
                    else "hit_update_or_safety_cap_unconverged"
                )
            ),
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] 见 version 25 changelog (a)：
            # True 表示这个窗口是靠"见好就收"重试上限兜底放行进生产的，
            # 而不是严格达到 lse_log_residual_tolerance——生产后独立的
            # _assert_stage_result_sane 仍是真正的正确性把关。
            "best_effort_acceptance": bool(warm.best_effort_acceptance),
            "best_effort_acceptance_reason": warm.best_effort_acceptance_reason,
            "best_effort_residual": (
                float(warm.best_effort_residual)
                if np.isfinite(warm.best_effort_residual)
                else None
            ),
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=25] PROPOSAL_frozen_validation_
            # fallback.md：True 表示这个窗口至少有一次是 learning 候选
            # streak 从未凑满、纯粹靠 max_bias_updates 耗尽 fallback 才
            # 进入 freeze_burn_in 的，不是"reached via genuine candidate
            # streak"。
            "reached_freeze_via_budget_exhaustion_fallback": bool(
                warm.budget_fallback_used
            ),
            "early_probe_triggered": bool(warm.early_probe_triggered),
            "early_probe_trigger_reason": warm.early_probe_trigger_reason,
            "window_index": int(window_idx),
            "global_state_range": [int(start), int(end)],
            "global_lambda_indices": list(range(int(start), int(end))),
            "lambdas_coul": [float(x) for x in wc.lc_win],
            "lambdas_vdw": [float(x) for x in wc.lv_win],
            "n_states": int(K),
            "target_p": float(target_p),
            "min_probability_threshold": float(min_probability_threshold),
            "coverage_ess": float(coverage_ess),
            "coverage_ess_threshold": float(coverage_ess_threshold),
            "ema_mean_p": warm.ema_mean_p_values,
            "max_ema_mean_p": max_ema_p,
            "min_ema_mean_p": min_ema_p,
            "last_update_diagnostics": wc.sampler.last_update_diagnostics,
            "tmbar_history_length": len(wc.sampler.tmbar_history),
            "last_tmbar_update": wc.sampler.last_update_diagnostics.get("tmbar_update", {}),
            "last_f_k_delta_kJ_mol": warm.last_f_delta,
            "f_stability_threshold_kJ_mol": float(warm.f_stability_threshold_kJ_mol),
            # 🔑 [2026-09-02] `min_bias_updates` 与 `required_consecutive_passes`
            # 是**失效配置**：全仓 grep 只在函数签名与本落盘 dict 出现，
            # **从不进入任何条件判断**。这是 Candidate-first / Validate-or-Learn v1
            # 的刻意设计（见上方 mode=="learning" 分支注释："no fixed
            # min_bias_updates batch count"、"不再要求固定 min_bias_updates 批数"），
            # 冻结判据换成了 `residual_severity <= IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW`。
            #
            # 但字段照原样落盘会**说谎**：4W53 甲苯溶剂腿 window 0 的记录是
            # `status=frozen_validation_converged` + `bias_update_count=1`
            # + `min_bias_updates=12` + `consecutive_pass_count=0`
            # + `required_consecutive_passes=3`，读起来像"施加了 12 批/3 连过的
            # 判据并通过了"，实际一条都没判。所以显式标注它们已失效，
            # 并把真正生效的判据一并写出来。
            # 要么真正强制、要么这样标注；不允许留一个看起来像判据的死字段。
            "min_bias_updates": int(min_bias_updates),
            "max_bias_updates": int(max_bias_updates),
            "bias_update_count": int(warm.bias_update_count),
            "required_consecutive_passes": int(required_consecutive_bias_updates),
            "consecutive_pass_count": int(warm.consecutive_pass_count),
            "min_bias_updates_is_enforced": False,
            "required_consecutive_passes_is_enforced": False,
            "inert_criteria_note": (
                "min_bias_updates / required_consecutive_passes 自 "
                "Candidate-first Validate-or-Learn v1 起**不再被判定**，"
                "仅为兼容保留在本记录里。实际生效的冻结判据是 "
                "residual_severity <= IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW"
                f"（={float(IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW)}，由单个 "
                f"{int(IBS_TMBAR_LEARNING_MINIBATCH_FRAMES)} 帧 minibatch 的占据算出）；"
                "实际生产放行判据是 local-MBAR loose gate 的 "
                "max_adjacent_delta_kJ_mol < gate_threshold_kJ_mol。"
                "不要把 consecutive_pass_count 当成'判过了几次'。"
            ),
            "freeze_criterion_actually_used": (
                "residual_severity<=IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW"
            ),
            "freeze_criterion_residual_low_threshold": float(
                IBS_UPDATE_ADAPTIVE_RESIDUAL_LOW
            ),
            "freeze_criterion_minibatch_frames": int(
                IBS_TMBAR_LEARNING_MINIBATCH_FRAMES
            ),
            "validation_pass_count": int(warm.validation_pass_count),
            "validation_batch_probabilities": warm.validation_batch_history,
            "validation_cumulative_probabilities": warm.validation_cumulative_history,
            "validation_attempt_budget_steps": int(validation_attempt_budget_steps),
            "minimum_complete_validation_frames": int(
                minimum_complete_validation_frames
            ),
            "truncated_validation_frames_ignored": int(
                warm.truncated_validation_frames_ignored
            ),
            "best_effort_reburnin_steps": int(warm.best_effort_reburnin_steps),
            "safety_cap_best_effort_tmbar_converged": bool(
                warm.safety_cap_best_effort_tmbar_converged
            ),
            "lse_log_residual_tolerance": float(lse_log_residual_tolerance),
            "lse_balance": report_lse_balance,
            "lse_residual_history": warm.lse_residual_history,
            # 🔑 [IBS_BIAS_PROTOCOL_VERSION=29] 局部滑窗 MBAR loose-gate 诊断：
            # 每次冻结复检解出的相邻态 |Δf_k−ΔF^MBAR| 及阈值/滑窗深度，供事后
            # 追溯这个窗口是靠 loose-gate 通过还是预算耗尽 best-effort 放行。
            "local_mbar_loose_gate": {
                "gate_threshold_kJ_mol": float(
                    IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL
                ),
                "sliding_window_batches": int(
                    IBS_LOCAL_MBAR_GATE_SLIDING_BATCHES
                ),
                "last": warm.last_local_mbar_gate,
                "history": warm.local_mbar_gate_history,
            },
            "convergence_criterion": {
                "learning_candidate": "existing_update_weights_tmbar_or_bounded_sgd",
                "frozen_validation": (
                    "max_adjacent_abs_delta_f_minus_local_mbar_delta_F_below_"
                    "IBS_LOCAL_MBAR_GATE_MAX_ADJACENT_DELTA_KJ_MOL"
                ),
            },
            "resumed_from_cache": bool(warm.resumed_frozen_f_k is not None),
            "learning_to_validation_cycles": int(warm.learning_to_validation_cycles),
            "validation_feedback_update_count": int(
                warm.validation_feedback_update_count
            ),
            "resumed_calibration_pending": bool(resumed_calibration_pending),
            "frozen_validation_retry_count": int(warm.frozen_validation_retry_count),
            "frozen_burn_in_steps": int(warm.frozen_burn_in_steps),
            "frozen_f_k_at_last_freeze": warm.frozen_f_k_snapshot,
            "final_mode": warm.mode,
            "steps_at_full_bias": int(warm.steps_at_full_bias),
            "max_bias_learning_steps_safety_cap": int(full_bias_step_budget),
            "sgd_step_budget": int(sgd_step_budget),
            "frozen_validation_reserved_steps": int(
                frozen_validation_reserved_steps
            ),
            "mbar_calibration_reserved_steps": int(
                effective_mbar_calibration_reserved_steps
            ),
            "was_resumed": bool(warm.is_resumed_ibs),
            "f_delta_is_diagnostic_only": True,
            "feedback_action": (
                (
                    "design_split_window_using_existing_lambdas"
                    if K >= 3
                    else "design_insert_thermodynamic_midpoint_then_revalidate_lse"
                )
                if warmup_only
                else
                # 🔑 [non_mutating_v1] 非变异策略下不做任何拆窗/插点/fixed-H 探针
                # 建议——未收敛就是 f_not_converged，保留数据交人工/rescue 审计。
                # 只有已弃用的 legacy_mutating 策略才给旧的拆窗/探针语义（见下）。
                (
                    # 🔑 [THERMODYNAMIC_PATH_PROTOCOL_VERSION=7]
                    # split_window_from_warmup_failure 要求两个孩子各自 >=3 态、
                    # 共享 1 态，父窗口因此至少要 5 态才能这样拆（3+3-1）；K=4
                    # 走 fixed-H overlap 探针，不再被拆成 K=2+K=3。
                    ("split_window_without_lambda_insertion" if K >= 5
                     else "probe_fixed_hamiltonian_bidirectional_overlap")
                    if legacy_repair
                    else "none_non_mutating_f_not_converged"
                )
            ),
        }

        # Only a minimal-or-unsplittable window is allowed to request a new
        # lambda, and only after independent fixed-H trajectories prove
        # that a real adjacent edge lacks overlap. K<=4 here must match
        # split_window_from_warmup_failure's min_states_before_split=5
        # floor in abfe_preoptimizer.py/abfe_pipeline.py -- K=4 cannot be
        # split into two >=3-state children sharing one state, so it must
        # come here instead of being blindly bisected into K=2+K=3.
        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=12] 续验已冻结的校准 f_k 时，λ 网格
        # 覆盖度和 f_k 校准都已经在上一轮证明过了——重新跑 fixed-H overlap
        # 探针和 MBAR 校准既浪费 GPU，也会覆盖掉这份已经验证正确的
        # frozen_f_k_pending。只有从未校准过（或本次是全新窗口）才需要
        # 走这条探针+校准分支。
        # 🔑 [non_mutating_v1] 只有旧的（已弃用）变异策略才进入 fixed-H overlap
        # 探针 + asymmetric 判据 + bias 校准 + 就地覆盖 f_k 这条分支。非变异策略
        # 下 not bias_converged 直接落到下面的 `else` 分支，抛 f_not_converged，
        # 绝不烧 fixed-H GPU、绝不重写 f_k。overlap_probe 保持空 → 下面 6512 的
        # asymmetric 块与 6528 的校准块都因 all_passed 为假而自然跳过。
        if (
            legacy_repair
            and not warm.bias_converged
            and not resume_frozen_validation
            and K <= 4
            and stage_type == "vdw"
        ):
            try:
                current_state = wc.sim.context.getState(getPositions=True)
                bank_result = probe_adjacent_path_overlap_bank(
                    topology=self.topology,
                    common_system_xml=ibs_wrap._common_system_xml,
                    ibs_wrapper=ibs_wrap,
                    K=K,
                    positions=current_state.getPositions(),
                    box_vectors=current_state.getPeriodicBoxVectors(),
                    temperature=self.temperature,
                    platform_name=self.platform_name,
                    checkpoint_dir=self.checkpoint_dir,
                    stage_type=stage_type,
                    window_idx=window_idx,
                    global_state_start=int(start),
                    coion_identity=self.coion_identity,
                )
                warm.bias_warmup_diag["bidirectional_overlap_probe"] = {
                    "pairs": bank_result["pairs"],
                    "all_passed": bool(bank_result["all_passed"]),
                }
            except Exception as probe_error:
                warm.bias_warmup_diag["bidirectional_overlap_probe"] = {
                    "pairs": [],
                    "all_passed": False,
                    "error": repr(probe_error),
                }

        # 🔑 [IBS_BIAS_PROTOCOL_VERSION=10] fixed-H overlap 全部通过说明 λ 网格
        # 本身没问题：固定态之间确实能互相重加权，问题出在 f_k 求解器（SGD
        # 学习率过大/EMA 跨 Hamiltonian）没能稳定收敛，而不是密度不够。但用来
        # 下这个判断的 probe_pairs（上面的路径 overlap 探针）绝不能直接拿来
        # 校准 f_k——它的动力学是 U_common + CV_k（不含 Group 4 WCA），采的是
        # 跟生产不同的构象系综；它的 delta_f_kJ_mol 还叠加了 LRC，而生产 Group1
        # 偏置力本身从不含 LRC（LRC 只在 collect_energies() 里事后加进
        # target_energies 喂 MBAR，不进偏置力）。这曾经是一个真实的、已确认的
        # bug：用错了 ensemble、又把不属于偏置力的能量项校准了进去。这里改为
        # 单独跑 probe_bidirectional_overlap_for_bias_calibration——同一个窗口、
        # 同一个 lambda_shield、保留 Group 4 WCA、绝不加 LRC——只用它返回的
        # delta_f_bias_kJ_mol 校准 f_k；且要求每条边的去相关样本数/不确定度都
        # 达标（不够就延长采样重试，不能直接拿一次勉强通过 overlap 阈值但样本
        # 严重不足的估计去覆盖 f_k）。校准后只给一次"冻结 burn-in + 独立验证"
        # 的机会，不再回退 learning；仍不通过就是真实的构象弛豫慢/偏置表达式
        # 问题，交给人工检查，而不是继续无意义地烧 GPU 重试同一个不稳定求解器。
        overlap_probe = warm.bias_warmup_diag.get("bidirectional_overlap_probe", {})

        asymmetric_bottleneck = None
        if overlap_probe.get("all_passed"):
            asymmetric_bottleneck = detect_passed_but_asymmetric_overlap_bottleneck(
                overlap_probe.get("pairs", []), wc.lv_win,
            )
            if asymmetric_bottleneck is not None:
                overlap_probe["passed_but_asymmetric_bottleneck"] = asymmetric_bottleneck
                warm.bias_warmup_diag["status"] = "passed_but_asymmetric_overlap_bottleneck"
                warm.bias_warmup_diag["feedback_action"] = "insert_lambda_at_passed_bottleneck"
                print(
                    "    [WARN] fixed-H 各边虽均超过绝对阈值，但检测到局部热力学瓶颈："
                    f"edge={asymmetric_bottleneck['global_edge']}，"
                    f"overlap ratio={asymmetric_bottleneck['overlap_ratio']:.2f}，"
                    f"slope ratio={asymmetric_bottleneck['slope_ratio']:.2f}；"
                    "跳过 f_k 校准阶梯，反馈路径插点。"
                )

        if (
            not warm.bias_converged
            and not resume_frozen_validation
            and overlap_probe.get("all_passed")
            and asymmetric_bottleneck is None
        ):
            lam_vdw_center_for_calibration = float(np.mean(wc.lv_win))
            calib_state = wc.sim.context.getState(getPositions=True)
            calib_positions = calib_state.getPositions()
            calib_box = calib_state.getPeriodicBoxVectors()
            calib_bank_result = probe_adjacent_bias_calibration_bank(
                topology=self.topology,
                common_plus_wca_system_xml=ibs_wrap._common_plus_wca_system_xml,
                ibs_wrapper=ibs_wrap,
                K=K,
                positions=calib_positions,
                box_vectors=calib_box,
                temperature=self.temperature,
                platform_name=self.platform_name,
                checkpoint_dir=self.checkpoint_dir,
                stage_type=stage_type,
                window_idx=window_idx,
                global_state_start=int(start),
                lambda_shield=lam_vdw_center_for_calibration,
                coion_identity=self.coion_identity,
            )
            calibration_pairs = calib_bank_result["pairs"]
            calibration_sufficient = bool(calib_bank_result["all_sufficient"])

            warm.bias_warmup_diag["bias_calibration_probe"] = {
                "pairs": calibration_pairs,
                "all_sufficient": calibration_sufficient,
                "lambda_shield": lam_vdw_center_for_calibration,
            }

            if not calibration_sufficient:
                print(
                    "    [WARN] bias 校准探针在多次延长采样后，仍有边的去相关样本数/ΔF 不确定度"
                    "未达标，放弃本次 MBAR 校准（不拿不够精确的估计覆盖 f_k），"
                    "继续按未收敛处理，交给人工检查。"
                )
            else:
                delta_f_bias_edges = [float(p["delta_f_bias_kJ_mol"]) for p in calibration_pairs]
                f_calibrated = np.concatenate(([0.0], np.cumsum(delta_f_bias_edges)))
                f_calibrated = f_calibrated - np.mean(f_calibrated)
                for k in range(K):
                    wc.sim.context.setParameter(f"{self.prefix}_f_{k}", float(f_calibrated[k]))
                print(
                    f"    fixed-H 双向 overlap 全部通过（λ 网格没问题），bias 校准探针"
                    f"（含 WCA、不含 LRC）也已达到精度门槛："
                    f"用 ΔF_bias 校准 f_k={[f'{v:.2f}' for v in f_calibrated]}，"
                    "不再继续 SGD 搜索，重新做一次冻结 burn-in + 只读验证。"
                )
                wc.sampler.energy_buffer = []
                wc.sampler.ema_mean_p = None
                calibration_burn_in_done = 0
                calibration_validation_pass_count = 0
                calibration_converged = False
                calibration_steps_used = 0
                # 用独立预留的预算，而不是继续检查
                # steps_at_full_bias < max_bias_learning_steps——SGD 阶段已经把它的
                # 预算（sgd_step_budget）用尽退出，如果这里还沿用同一个已耗尽的
                # 计数器，这个循环会一次都不执行，calibration_converged 必然为
                # False，跟校准本身是否有效无关。默认是
                # mbar_calibration_reserved_steps，但 [IBS_BIAS_PROTOCOL_VERSION=12]
                # 起如果调用方通过 frozen_validation_step_overrides 给这个窗口
                # 累计延长过预算（上一轮校准好但验证未通过），这里用那个更大的值。
                while calibration_steps_used < int(effective_frozen_validation_budget):
                    guarded_step(wc.sim, check_chunk, f"窗口 {window_idx} 冻结验证", on_error=_nan_diagnostics)
                    warm.steps_at_full_bias += check_chunk
                    _attempt_ledger["frozen_validation_steps"] += check_chunk
                    calibration_steps_used += check_chunk
                    wc.sampler.collect_energies()
                    if calibration_burn_in_done < warm.frozen_burn_in_steps:
                        calibration_burn_in_done += check_chunk
                        wc.sampler.energy_buffer = []
                        continue
                    if len(wc.sampler.energy_buffer) < 20:
                        continue
                    # 🔑 有意不调用 sampler._append_tmbar_batch_from_buffer()：
                    # 这是已弃用的 legacy_mutating 专属 fixed-H/MBAR 校准回退路径
                    # （repair_policy != "non_mutating_v1" 才会走到这里），验证失败
                    # 时是终态（不像上面 non_mutating_v1 的 validating 分支那样退回
                    # "learning"），tmbar_history 只在 learning 模式被解出使用，
                    # 追加进去也永远不会被读取，白白增大落盘状态。
                    batch_p = wc.sampler.evaluate_frozen_batch_probability()
                    if batch_p is None:
                        continue
                    p = np.asarray(batch_p, dtype=np.float64)
                    warm.last_validation_batch_p = p.tolist()
                    warm.validation_batch_history.append(warm.last_validation_batch_p)
                    warm.last_lse_balance = ibs_lse_balance_diagnostics(p)
                    warm.lse_residual_history.append(dict(warm.last_lse_balance))
                    # 🔑 [MAIN_WINDOW_CHECKPOINT_PROTOCOL_VERSION] 这个循环
                    # 本来就只服务于刚校准出来的 f_k 的验证，不需要额外条件
                    # ——跟上面 resumed_calibration_pending 分支一样，每评估
                    # 完一批就存一次，覆盖式落盘，应对 HPC 作业被抢占/撞墙
                    # 时限杀掉的情况。
                    _atomic_save_openmm_checkpoint(wc.sim, main_ckpt_path)
                    _atomic_write_json(main_manifest_path, expected_main_manifest)
                    # 🔑 [checkpoint/累计步数同步修复] 同上面 resumed_calibration_
                    # pending 分支：这是这份 f_k 第一次被冻结验证（不是续验），
                    # 累计步数就是这次 attempt 里已经跑过的 calibration_steps_used，
                    # 必须跟 .chk 同频率落盘，理由同上。
                    wc.sampler.frozen_validation_cumulative_steps = int(calibration_steps_used)
                    wc.sampler.save_ibs_state(
                        ibs_state_file, wc.lc_win, wc.lv_win, stage_type=stage_type
                    )
                    calib_lse_ok = bool(
                        warm.last_lse_balance.get("available", False)
                        and warm.last_lse_balance["max_abs_log_residual"]
                        <= float(lse_log_residual_tolerance)
                    )
                    if calib_lse_ok:
                        calibration_validation_pass_count += 1
                    else:
                        calibration_validation_pass_count = 0
                    if calibration_validation_pass_count >= int(required_consecutive_bias_updates):
                        calibration_converged = True
                        break

                warm.bias_warmup_diag["mbar_calibration"] = {
                    "attempted": True,
                    "delta_f_bias_edges_kJ_mol": delta_f_bias_edges,
                    "calibrated_f_k_kJ_mol": f_calibrated.tolist(),
                    "converged": calibration_converged,
                    "steps_used": int(calibration_steps_used),
                    "steps_reserved": int(effective_frozen_validation_budget),
                    "validation_pass_count": int(calibration_validation_pass_count),
                }
                if calibration_converged:
                    warm.bias_converged = True
                    warm.frozen_f_k_snapshot = f_calibrated.tolist()
                    warm.mode = "validating_after_mbar_calibration"
                    _report_p = np.asarray(warm.last_validation_batch_p, dtype=np.float64)
                    min_ema_p = float(np.min(_report_p))
                    max_ema_p = float(np.max(_report_p))
                    warm.ema_mean_p_values = warm.last_validation_batch_p
                    coverage_ess = float(1.0 / np.sum(np.square(_report_p)))
                    report_lse_balance = ibs_lse_balance_diagnostics(_report_p)
                    warm.bias_warmup_diag["status"] = "frozen_validation_converged_after_mbar_calibration"
                    warm.bias_warmup_diag["coverage_ess"] = float(coverage_ess)
                    warm.bias_warmup_diag["ema_mean_p"] = warm.ema_mean_p_values
                    warm.bias_warmup_diag["max_ema_mean_p"] = max_ema_p
                    warm.bias_warmup_diag["min_ema_mean_p"] = min_ema_p
                    warm.bias_warmup_diag["lse_balance"] = report_lse_balance
                    warm.bias_warmup_diag["lse_residual_history"] = warm.lse_residual_history
                    # 🔑 final_mode/frozen_f_k_at_last_freeze 是在 SGD 循环退出时就写进
                    # bias_warmup_diag 的，用的是当时的 mode/frozen_f_k_snapshot；这两个
                    # 局部变量在 MBAR 校准成功后已经被重新赋值（above），但字典里的旧
                    # entry 从未被刷新——不同步的话，诊断会显示校准前的旧 mode/f_k，跟
                    # 实际让这个窗口收敛的其实是校准后的验证这一事实矛盾。
                    warm.bias_warmup_diag["final_mode"] = warm.mode
                    warm.bias_warmup_diag["frozen_f_k_at_last_freeze"] = warm.frozen_f_k_snapshot
                else:
                    # 🔑 之前这个分支不更新 ema_mean_p_values/min_ema_p/max_ema_p/
                    # coverage_ess，导致下面失败报告打印的是校准前 SGD 阶段的旧值，
                    # 不是这次 MBAR 校准冻结验证真正观察到的最后一批概率——现在用
                    # last_validation_batch_p（本轮 while 循环里每次真实评估都会
                    # 更新）覆盖这几个报告变量，保证失败诊断反映真实情况。
                    if warm.last_validation_batch_p is not None:
                        _report_p = np.asarray(warm.last_validation_batch_p, dtype=np.float64)
                        min_ema_p = float(np.min(_report_p))
                        max_ema_p = float(np.max(_report_p))
                        warm.ema_mean_p_values = warm.last_validation_batch_p
                        coverage_ess = float(1.0 / np.sum(np.square(_report_p)))
                        report_lse_balance = ibs_lse_balance_diagnostics(_report_p)
                        warm.bias_warmup_diag["coverage_ess"] = float(coverage_ess)
                        warm.bias_warmup_diag["ema_mean_p"] = warm.ema_mean_p_values
                        warm.bias_warmup_diag["max_ema_mean_p"] = max_ema_p
                        warm.bias_warmup_diag["min_ema_mean_p"] = min_ema_p
                        warm.bias_warmup_diag["lse_balance"] = report_lse_balance
                        warm.bias_warmup_diag["lse_residual_history"] = warm.lse_residual_history
                    print(
                        "    [WARN] 用 BAR/MBAR（ΔF_bias，含 WCA、不含 LRC）校准的 f_k 冻结验证仍未"
                        "通过——λ 网格和求解器都已排除，可能是构象弛豫过慢或偏置表达式本身有"
                        "问题，需要人工检查，不再自动重试。"
                    )

        # 🔑 [2026-09-11] 累计上限已经用完、这次一步都没跑，而候选的证据状态仍是
        # "无法判定"：结局必须还是无法判定。否则会掉进下面的普通未收敛分支，被
        # 报成 f_not_converged —— 那是"测出来了 f_k 不对"的语义，还会触发插 λ。
        # 🔑 [2026-09-11 二修] 原来这道守卫只认两种情形：(a) 攒满
        # MAX_BATCHES 仍不足（在循环里判），(b) resume 时剩余预算已为 0 且
        # 证据状态已是 indeterminate。漏掉了最常见的第三种：**正在分档累计的
        # 途中，步数预算先耗尽**。实测 window 2 段2：gate 跑了 2 次、两次都是
        # insufficient_frames_after_decorrelation（从没测出过数），攒到 10 批时
        # 180000/180000 步用光，于是掉进普通未收敛分支、被写成 f_not_converged
        # 抛 IBSWarmupConvergenceError —— 正是"没测出来"被改写成"f_k 错了"。
        # 判据：本周期进过累计模式(validation_insufficient_attempts>0) 且
        # 从未解出过一次 Δf−ΔF(ever_completed_a_validate_attempt False)。
        _never_measured = bool(
            warm.validation_insufficient_attempts > 0
            and not warm.ever_completed_a_validate_attempt
        )
        # 🔑 [2026-09-11 三修] 第三个变体：**可用预算为 0 ⟹ 循环一次都没进**。
        # 实测 win2 连死三次，最后一次 steps_at_full_bias=0、gate history 空，
        # 却被标成 f_not_converged。这既不是"f_k 不收敛"也不是"测了没测出来"——
        # 是**根本没开始**。老板定案：这类走 HALT_BUDGET + INSUFFICIENT_DATA。
        # 跟上面两个变体的区别：这里 frozen_f_k_snapshot 可能是 None（还没冻结过
        # 任何候选），所以不能挂在"必须有候选"那个前提下。
        _no_budget_at_all = bool(
            warm.steps_at_full_bias == 0
            and full_bias_step_budget <= 0
            and not warm.ever_completed_a_validate_attempt
        )
        if (
            not warm.bias_converged
            and warm.validation_indeterminate_diag is None
            and (
                _never_measured
                or _no_budget_at_all
                or (
                    full_bias_step_budget <= 0
                    and getattr(wc.sampler, "f_k_evidence_status", F_K_EVIDENCE_NONE)
                    == F_K_EVIDENCE_INDETERMINATE
                )
            )
            and (warm.frozen_f_k_snapshot is not None or _no_budget_at_all)
        ):
            warm.validation_indeterminate_diag = {
                "reason": "validation_budget_exhausted_indeterminate",
                "n_batches": int(
                    getattr(wc.sampler, "frozen_validation_batches_done", 0)
                ),
                "max_batches": int(IBS_LOCAL_MBAR_GATE_MAX_BATCHES),
                "validation_sample_count": 0,
                "insufficient_attempts": 0,
                "last_gate_error": "no_budget_remaining_on_resume",
                "statistical_inefficiency": None,
                "n_frames_used": 0,
                "steps_at_full_bias": 0,
                "frozen_f_k_kJ_mol": (
                    [float(x) for x in warm.frozen_f_k_snapshot]
                    if warm.frozen_f_k_snapshot is not None else None
                ),
                "never_measured_mid_accumulation": bool(_never_measured),
                # HALT_BUDGET：可用预算为 0，连一步动力学都没跑
                "halt_budget_no_steps_available": bool(_no_budget_at_all),
                "cumulative_cap_steps": int(
                    _ledger.get("cumulative_cap_steps", 0) or 0
                ),
                "cumulative_consumed_steps": warmup_ledger_total_steps(_ledger),
                "n_batches_when_budget_ran_out": int(len(warm.frozen_mbar_batches)),
            }
            print(
                f"  [无法判定] 窗口 {window_idx} 累计预算已用尽"
                f"（上限 {_ledger.get('cumulative_cap_steps')}、已耗 "
                f"{warmup_ledger_total_steps(_ledger)}），本次可用 0 步。"
                "候选仍未被测量，保持无法判定；要继续必须显式升档。"
            )

        # 🔑 [2026-09-11，实验开关] 重标定 f_k 不再被 loose gate 拦住。
        # 见 accept_recalibrated_f_k_without_gate 的参数注释。门已经在上面跑过
        # 并记进 local_mbar_gate_history / validation_indeterminate_diag，这里
        # 只是不让它阻断。
        if (
            not warm.bias_converged
            and accept_recalibrated_f_k_without_gate
            and getattr(wc.sampler, "seed_source", None) == "production_recalibration"
            and warm.steps_at_full_bias > 0      # 绝不零平衡进生产
        ):
            warm.bias_converged = True
            warm.best_effort_acceptance = True
            warm.best_effort_acceptance_reason = (
                "recalibrated_f_k_accepted_without_gate"
            )
            warm.validation_indeterminate_diag = None
            warm.last_failure_reason = None
            _gate_note = (
                warm.last_local_mbar_gate.get("error")
                or warm.last_local_mbar_gate.get("max_adjacent_delta_kJ_mol")
                if warm.last_local_mbar_gate else "门从未评估"
            )
            print(
                f"  [重标定 f_k 直放] 窗口 {window_idx}：这份 f_k 由生产帧重标定"
                f"而来，按实验开关不再被 loose gate 拦住（门的结果仍已记录："
                f"{_gate_note}）。已完成 {warm.steps_at_full_bias} 步冻结动力学，"
                "不是零平衡进生产。真实 ESS/误差/自由能交生产后 MBAR 与 "
                "_assert_stage_result_sane 把关。"
            )

        # 🔑 [2026-09-11] 所有耗步的循环都结束了，把本次分项消耗并进持久账本。
        # 必须在任何 save_ibs_state 之前，且只并一次（下面三条退出路径共用）。
        for _bucket in WARMUP_LEDGER_BUCKETS:
            _ledger[_bucket] = int(_ledger.get(_bucket, 0) or 0) + int(
                _attempt_ledger[_bucket]
            )
        wc.sampler.warmup_budget_ledger = _ledger
        warm.bias_warmup_diag["warmup_budget_ledger"] = dict(_ledger)
        warm.bias_warmup_diag["warmup_budget_remaining_steps"] = (
            warmup_ledger_remaining_steps(_ledger)
        )

    def _finish_production(
        self,
        *,
        K,
        debug_mode,
        es,
        end,
        ibs_state_file,
        ibs_wrap,
        wc,
        lse_log_residual_tolerance,
        n_steps_per_window,
        prod,
        repair_policy,
        stage_type,
        start,
        steps_per_update,
        warm,
        window_idx,
    ):
        """相位 7+8：生产采样循环 + 收口/落盘—— 刀1 B段第3步自
        _produce_window 原样搬出（联合段；7/8 细分等 n_updates==0 路径
        语义拍板，见执行器头注释）。纯搬运（verify_extract --slice 逐节点
        等价）；跨界变量按原名作 keyword-only 形参。
        """
        for up in range(prod.n_updates):
            pos_backup = prod.production_pos_backup
            with _timed(prod.loop_timers, "integration_s"):
                try:
                    wc.sim.step(steps_per_update)
                except Exception as e:
                    print(f"\n  [ERR] 采样崩溃于 update={up}/{prod.n_updates}")
                    if debug_mode:
                        diagnose_force_groups_detailed(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_崩溃_update{up}")
                        diagnose_force_breakdown(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_崩溃_update{up}")
                    raise

            current_production_f_k = np.asarray(
                [
                    wc.sim.context.getParameter(f"{self.prefix}_f_{k}")
                    for k in range(K)
                ],
                dtype=np.float64,
            )
            if not np.allclose(
                current_production_f_k,
                prod.production_f_k_lock,
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    f"窗口 {window_idx} 生产阶段 update={up} 检测到 f_k 被修改；"
                    "已立即停止，拒绝继续生成混合偏置生产数据。"
                )

            with _timed(prod.loop_timers, "guard_s"):
                do_force_check = (up % prod.force_check_interval == 0) or (up == prod.n_updates - 1)
                # 🔑 [性能修复] guard 在 do_force_check=True 时已经把 positions/box
                # 一并取回；下面 collect_energies() 会复用这份数据，不再独立
                # 重新查一次同一个 context、同一时刻的状态。
                guard_positions = None
                guard_box_vectors = None
                # [性能修复：合并 guard 查询，PLAN 2026-08-26] 弱分支（大多数
                # update）不需要 forces——fmax 只在强分支才非 None——只需要
                # "总能量是否有限"这一个数。生产窗口系统（build_ibs_dual_system/
                # build_shadow_coul_ibs_system，已逐一核实两者的力只落在
                # Group 0-5，且 _build_window_system() 之后没有第三方再往
                # win_sys 加力）里，e_base(groups={0,2,3,5}) + e_bias(groups=
                # {1,4}) 恒等于不过滤的 getState(getEnergy=True) 总能量——不是
                # 近似，是这套系统构造保证的精确恒等式。直接查这两组既够判
                # disaster，又能把这两个数喂给下面 collect_energies()，省掉它
                # 原本还要对同一瞬时状态重新查一遍的两次 getState()。
                guard_e_base = None
                guard_e_bias = None
                if do_force_check:
                    state_n = wc.sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                    forces_n = state_n.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                    fmax = np.max(np.linalg.norm(forces_n, axis=1))
                    guard_positions = state_n.getPositions(asNumpy=True)
                    guard_box_vectors = state_n.getPeriodicBoxVectors()
                    e_total_n = state_n.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                else:
                    # 各自包一层 try/except，语义对齐 collect_energies() 内部
                    # 原来对这两组的处理——查询失败不让整个 update 崩，标 NaN
                    # 交给下面统一的 disaster 判据处理。
                    try:
                        guard_e_base = (
                            wc.sim.context.getState(getEnergy=True, groups=set(IBS_E_BASE_FORCE_GROUPS))
                            .getPotentialEnergy()
                            .value_in_unit(unit.kilojoule_per_mole)
                        )
                    except Exception:
                        guard_e_base = float("nan")
                    try:
                        guard_e_bias = (
                            wc.sim.context.getState(getEnergy=True, groups=set(IBS_E_BIAS_FORCE_GROUPS))
                            .getPotentialEnergy()
                            .value_in_unit(unit.kilojoule_per_mole)
                        )
                    except Exception:
                        guard_e_bias = float("nan")
                    e_total_n = guard_e_base + guard_e_bias
                    fmax = None

                energy_bad = not np.isfinite(e_total_n)
                if energy_bad and not do_force_check:
                    # 能量已经异常，不能再等下一次排期检查——立刻补查力和坐标。
                    # 升级到强分支后立刻 continue（见下面的 disaster 判据），
                    # guard_e_base/guard_e_bias 不会被用到，不需要重新算。
                    state_n = wc.sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                    forces_n = state_n.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                    fmax = np.max(np.linalg.norm(forces_n, axis=1))
                    guard_positions = state_n.getPositions(asNumpy=True)
                    guard_box_vectors = state_n.getPeriodicBoxVectors()
                    do_force_check = True

                if energy_bad or (fmax is not None and ((not np.isfinite(fmax)) or fmax > 10000.0)):
                    self._production_disaster_rollback(
                        wc.sim, pos_backup, wc.sampler, prod.production_history_backup_len,
                        stage_type, window_idx, attempt=up + 1,
                        e_total_n=e_total_n, fmax=fmax, win_sys=wc.win_sys, debug_mode=debug_mode,
                        label_prefix="", progress_note=f"update={up}/{prod.n_updates}, ",
                        diagnose_prefix=f"窗口{window_idx}_回退_update{up}",
                    )
                    continue

                if do_force_check:
                    # 复用同一次 getState 里已经取回的坐标（guard_positions），
                    # 不再单独发一次 getPositions 请求
                    prod.production_pos_backup = guard_positions
                    # 坐标备份与 history 长度必须成对更新，否则回退会截到错的分叉点。
                    prod.production_history_backup_len = _production_history_lengths(wc.sampler)

            # 🔑 生产阶段 f_k 已冻结（sampler.bias_converged=True，见上面严格收敛
            # 判据），不再调用 update_weights()——否则同一窗口的样本会来自随时间
            # 漂移的多个偏置，而不是 MBAR 假设的单一固定采样分布。只采样、记录
            # 能量；定期清空 energy_buffer 避免其无谓增长（update_weights() 之前
            # 顺带做的这件事，现在没人做了）。
            with _timed(prod.loop_timers, "cv_probe_s"):
                e = wc.sampler.collect_energies(
                    reuse_positions=guard_positions,
                    reuse_box_vectors=guard_box_vectors,
                    reuse_e_base=guard_e_base,
                    reuse_e_bias=guard_e_bias,
                )
            if len(wc.sampler.energy_buffer) >= 10:
                wc.sampler.energy_buffer = []
            if (up + 1) % 100 == 0:
                with _timed(prod.loop_timers, "ledger_io_s"):
                    self._enqueue_window_snapshot(window_idx, stage_type, wc.sampler)
                    # 🔑 [production checkpoint 续采] 每 100 个 update 覆盖式落盘一次
                    # 生产 checkpoint（坐标/速度/积分器 RNG）+ 累计步数——应对 HPC
                    # 作业被抢占/撞墙时限杀掉的情况：不这样做的话，一次很长的
                    # production 预算在被杀时会丢失全部进展，下次只能整窗从零重采
                    # （这正是这次要修的问题本身）。
                    # 🔑 [性能修复：重复落盘] 能量/偏置/基准能量/采样态/residual
                    # 五份数组已经在上面 _enqueue_window_snapshot() 里原子写过一次
                    # 了——production_energies_path 等变量本来就是跟
                    # _enqueue_window_snapshot 写的 dual_window_*.npy 完全相同的
                    # 路径，此前这里又把同一份 sampler.*_history 重新序列化一遍
                    # 写到同一批文件，是逐字节相同、纯浪费的第二次落盘，故删除。
                    # convergence.json 只合并更新 cumulative_production_steps 字段，
                    # 不覆盖其它字段——这份文件此刻还不是"最终结果"，真正完整的
                    # 版本在窗口正常结束时于下面"保存能量"区块整份重写。
                    _atomic_save_openmm_checkpoint(wc.sim, prod.production_ckpt_path)
                    _atomic_write_json(prod.production_manifest_path, prod.expected_production_manifest)
                    _periodic_conv = {}
                    if os.path.exists(prod.production_conv_path):
                        try:
                            with open(prod.production_conv_path, "r", encoding="utf-8") as pf:
                                _periodic_conv = json.load(pf)
                        except Exception:
                            _periodic_conv = {}
                    _periodic_conv["stage_protocol_key"] = getattr(self, "stage_protocol_key", None)
                    _periodic_conv["production_segment_protocol_version"] = PRODUCTION_SEGMENT_PROTOCOL_VERSION
                    _periodic_conv["production_segments"] = _production_segments_snapshot(wc.sampler)
                    _periodic_conv["cumulative_production_steps"] = (
                        int(prod.prior_cumulative_production_steps) + (up + 1) * int(steps_per_update)
                    )
                    _periodic_conv["window_data_protocol_version"] = (
                        IBS_WINDOW_DATA_PROTOCOL_VERSION
                    )
                    if stage_type == "vdw":
                        _periodic_conv["vdw_nonbonded_protocol_version"] = (
                            VDW_NONBONDED_PROTOCOL_VERSION
                        )
                    _periodic_conv["window_data"] = _window_data_metadata(
                        prod.production_energies_path,
                        prod.production_bias_path,
                        prod.production_base_path,
                    )
                    if self.sampling_score_sha256 is not None:
                        _periodic_conv["residual_sampling_protocol_version"] = (
                            IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
                        )
                        _periodic_conv["joint_score_window_data"] = {
                            "protocol_version": IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION,
                            "sampling_states": {
                                "shape": [
                                    len(wc.sampler.sampling_state_energy_history),
                                    len(wc.lc_win),
                                ],
                                "dtype": "float64",
                                "sha256": _sha256_file(
                                    prod.production_sampling_states_path
                                ),
                            },
                            "residual_basis": {
                                "shape": [len(wc.sampler.residual_basis_history)],
                                "dtype": "float64",
                                "sha256": _sha256_file(
                                    prod.production_residual_basis_path
                                ),
                            },
                            "sampling_state_definition": (
                                "softcore_U_k_plus_A_k_times_residual_basis_minus_offset"
                            ),
                            "physical_target_excludes_residual": True,
                        }
                    _periodic_conv["loop_timing_s"] = dict(prod.loop_timers)
                    _atomic_write_json(prod.production_conv_path, _periodic_conv)

            if (
                es.enable_early_stop
                and (up + 1) >= prod.early_stop_min_updates
                and (up + 1) % prod.early_stop_check_interval_updates == 0
            ):
                u_kj_raw = (
                    np.asarray(wc.sampler.energy_history, dtype=np.float64).T
                    if wc.sampler.energy_history else np.zeros((len(wc.lc_win), 0))
                )
                bias_kj = (
                    np.asarray(wc.sampler.bias_history, dtype=np.float64)
                    if wc.sampler.bias_history else np.zeros((0,))
                )
                base_kj = (
                    np.asarray(wc.sampler.base_energy_history, dtype=np.float64)
                    if wc.sampler.base_energy_history else np.zeros((0,))
                )
                # 🔑 [ESS_GATE_PROTOCOL_VERSION=2] 生产期 f_k 已冻结（进生产后
                # 不再调用 update_weights），所以此刻从 context 读到的就是这段
                # 历史采样时生效的 f_k，可以直接用来除掉共模因子。
                f_k_frozen = np.asarray(
                    [
                        wc.sim.context.getParameter(f"{self.prefix}_f_{k}")
                        for k in range(len(wc.lc_win))
                    ],
                    dtype=np.float64,
                )
                # 🔑 [0831issue / ESS_GATE_PROTOCOL_VERSION=4] f_k_frozen 是在
                # sampling-state 能量 S_k = U_k^sc + s_residual·A_k·(B_φ−offset)
                # 上学出来的（IBS_BIAS_PROTOCOL_VERSION=31），而 u_kj_raw 是
                # 物理目标（softcore+LRC，不含残差）。residual 臂上两者相差一个
                # 逐态 rank-1 项，在逐帧 softmax 里不抵消，所以混合覆盖度 ESS /
                # 占据 / 共模 σ 必须用 S_k 而不是 u_kj_raw。生产期 s_residual
                # 已恢复为 1.0（见本函数上方两处 setParameter），full-strength 的
                # sampling_state_energy_history 正是这段历史真正的采样态能量。
                # baseline 臂 residual_enabled=False → S_k ≡ U_k^sc，这里传 None，
                # 数值逐位不变。residual 臂拿不到 ledger 时 required=True 让门
                # fail closed，而不是静默退回错口径。
                _residual_on = bool(
                    getattr(
                        getattr(wc.sampler, "ibs_wrapper", None),
                        "residual_enabled",
                        False,
                    )
                )
                sampling_kj_gate = None
                if _residual_on and wc.sampler.sampling_state_energy_history:
                    sampling_kj_gate = np.asarray(
                        wc.sampler.sampling_state_energy_history, dtype=np.float64
                    ).T
                local_result = _solve_single_window_local_mbar(
                    u_kj_raw, bias_kj, base_kj, list(range(start, end)), prod.early_stop_kt,
                    f_k=f_k_frozen,
                    sampled_distribution_row=0, w_idx=window_idx,
                    production_segments=_production_segments_snapshot(wc.sampler),
                    sampling_kj=sampling_kj_gate,
                    sampling_gauge_required=_residual_on,
                )
                # 🔑 [0831issue P2] resume 续采时 `up` 从 0 重新计数，所以
                # `(up+1)*steps_per_update` 只是**本次 attempt** 内跑的步数，
                # 不是这个窗口累计跑到第几步。落盘的 early_stop_check_history
                # 里那个 "step" 标签因此在续跑后系统性偏小、跟
                # cumulative_production_steps 对不上。加回 resume 前已完成的步数。
                # 纯诊断标签，不参与任何判据（判据用的是 n_frames_used 等量）。
                step_at_check = (
                    int(prod.prior_cumulative_production_steps)
                    + (up + 1) * steps_per_update
                )
                if "error" in local_result:
                    prod.early_stop_pass_count = 0
                    prod.early_stop_check_history.append({
                        "step": int(step_at_check), "passed": False, "error": local_result["error"],
                    })
                    print(
                        f"    [early-stop] 窗口 {window_idx} 第 {step_at_check} 步检查：local MBAR "
                        f"未解出 ({local_result['error']})，连续通过计数清零"
                    )
                else:
                    f_arr = np.asarray(local_result["f"], dtype=np.float64)
                    df_arr = np.asarray(local_result["df"], dtype=np.float64)
                    local_dg = float(f_arr[-1] - f_arr[0]) if f_arr.size > 1 else 0.0
                    # 🔑 端点差的不确定度必须直接读 MBAR 算出的两个物理态之间
                    # 的成对不确定度（_solve_single_window_local_mbar 已经从
                    # ddf_matrix 里取好了），不能用 sqrt(df[0]^2+df[-1]^2) 把
                    # 两个边际（相对采样态）不确定度当独立量合并——它们来自
                    # 同一次 MBAR 拟合、同一批样本，彼此有协方差，平方相加
                    # 会系统性算错，可能让判据过早通过，也可能让它永远通不过。
                    local_uncertainty = (
                        float(local_result.get("endpoint_diff_uncertainty_kJ_mol", float("nan")))
                        if f_arr.size > 1
                        else (float(df_arr[0]) if f_arr.size else float("nan"))
                    )
                    min_ess_ratio = local_result.get("min_ess_ratio")
                    n_frames_used = int(local_result.get("n_frames_used", 0))
                    absolute_ess = (min_ess_ratio * n_frames_used) if min_ess_ratio is not None else None

                    # 🔑 同一类 ESS/overlap/不确定度浮点舍入问题（见
                    # _meets_minimum_with_roundoff/_meets_maximum_with_roundoff
                    # 的 docstring 与 v23 numerical follow-up）也会出现在这里——
                    # 这四项判据在数学上和 solve_stage_integrated 里的 converged
                    # 判据同源，必须用同一套只接受舍入尺度相等的容差比较，不能
                    # 只修一处。
                    ess_ratio_ok = min_ess_ratio is not None and _meets_minimum_with_roundoff(min_ess_ratio, es.early_stop_min_ess_ratio)
                    absolute_ess_ok = absolute_ess is not None and _meets_minimum_with_roundoff(absolute_ess, es.early_stop_min_absolute_ess)
                    decorrelated_ok = n_frames_used >= es.early_stop_min_decorrelated_samples
                    uncertainty_ok = np.isfinite(local_uncertainty) and _meets_maximum_with_roundoff(local_uncertainty, es.early_stop_max_uncertainty_kJ_mol)
                    # 第一次检查没有"上一次"可比，直接判不通过——至少要连续
                    # required_consecutive_passes+1 次检查才可能真正停止，
                    # 保证漂移判据不会在只有一个数据点时被跳过。
                    if prod.early_stop_previous_local_dg is None:
                        drift_ok = False
                        drift = None
                    else:
                        drift = float(abs(local_dg - prod.early_stop_previous_local_dg))
                        drift_ok = drift <= es.early_stop_max_delta_g_drift_kJ_mol
                    all_ok = bool(ess_ratio_ok and absolute_ess_ok and decorrelated_ok and drift_ok and uncertainty_ok)

                    prod.early_stop_check_history.append({
                        "step": int(step_at_check),
                        "min_ess_ratio": min_ess_ratio,
                        "absolute_ess": absolute_ess,
                        "n_frames_decorrelated": n_frames_used,
                        "local_delta_g_kJ_mol": local_dg,
                        "local_uncertainty_kJ_mol": local_uncertainty,
                        "drift_from_previous_kJ_mol": drift,
                        "checks": {
                            "ess_ratio_ok": ess_ratio_ok,
                            "absolute_ess_ok": absolute_ess_ok,
                            "decorrelated_ok": decorrelated_ok,
                            "drift_ok": drift_ok,
                            "uncertainty_ok": uncertainty_ok,
                        },
                        "passed": all_ok,
                    })
                    prod.early_stop_previous_local_dg = local_dg
                    prod.early_stop_pass_count = prod.early_stop_pass_count + 1 if all_ok else 0
                    print(
                        f"    [early-stop] 窗口 {window_idx} 第 {step_at_check} 步检查："
                        f"{'通过' if all_ok else '未通过'}（连续 {prod.early_stop_pass_count}/"
                        f"{es.early_stop_required_consecutive_passes}），ESS比例={min_ess_ratio}, "
                        f"绝对ESS={absolute_ess}, 去相关样本={n_frames_used}, "
                        f"局部ΔG={local_dg:.3f} kJ/mol, 不确定度={local_uncertainty:.3f} kJ/mol, "
                        f"漂移={drift}"
                    )
                    if prod.early_stop_pass_count >= es.early_stop_required_consecutive_passes:
                        prod.early_stop_triggered = True
                        prod.early_stop_stop_reason = "consecutive_passes_reached"
                        prod.actual_production_updates = up + 1
                        print(
                            f"    [OK] [early-stop] 窗口 {window_idx} 连续 {prod.early_stop_pass_count} 次独立 "
                            f"block 通过，提前于第 {step_at_check} 步停止生产采样"
                        )
                        break

            if debug_mode and up % 10 == 0:
                e_total = e_total_n
                has_nan_energy = not np.isfinite(e_total)
                has_nan_e = np.any(np.isnan(e))
                print(f"    [采样] update={up}/{prod.n_updates}: E_total={e_total:.1f}, NaN(E_total)={has_nan_energy}, NaN(E_k)={has_nan_e}")
                if has_nan_energy or has_nan_e:
                    print(f"    [ERR] 检测到 NaN 能量！主系统各组能量与最大受力：")
                    num_forces = wc.win_sys.getNumForces()
                    for i in range(num_forces):
                        force = wc.win_sys.getForce(i)
                        gid = force.getForceGroup()
                        try:
                            # 获取该 ForceGroup 的能量
                            state_g = wc.sim.context.getState(getEnergy=True, groups={gid})
                            e_g = state_g.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                            # 获取该 ForceGroup 的力
                            state_f = wc.sim.context.getState(getForces=True, groups={gid})
                            forces = state_f.getForces(asNumpy=True).value_in_unit(
                                unit.kilojoule_per_mole / unit.nanometer
                            )
                            max_f = np.max(np.linalg.norm(forces, axis=1)) if forces.size else 0.0
                            print(f"      Group {gid:2d} ({type(force).__name__:30s}) E={e_g:14.3f} kJ/mol, max|F|={max_f:12.2f}")
                        except Exception as e:
                            print(f"      Group {gid:2d} ({type(force).__name__:30s}) 获取失败: {e}")

        # early stop 提前退出时，跳过余数补齐——那是为了让"跑满预算"时的总
        # 步数严格达标而设计的，跟"提前停止"的意图矛盾。
        remaining_steps = 0 if prod.early_stop_triggered else prod.effective_n_steps_per_window % steps_per_update
        if remaining_steps > 0:
            pos_backup = prod.production_pos_backup
            with _timed(prod.loop_timers, "integration_s"):
                try:
                    wc.sim.step(remaining_steps)
                except Exception as e:
                    print(f"\n  [ERR] 余数补齐阶段崩溃 ({remaining_steps} 步)")
                    if debug_mode:
                        diagnose_force_groups_detailed(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_余数补齐崩溃")
                        diagnose_force_breakdown(wc.sim.context, wc.win_sys, prefix=f"窗口{window_idx}_余数补齐崩溃")
                    raise

            with _timed(prod.loop_timers, "guard_s"):
                # 🔑 [性能修复] 补上 getPositions=True，跟下面 else 分支复用同一次
                # State 里的坐标/box，不再像之前那样再单独发一次 getState(getPositions=True)。
                state_n = wc.sim.context.getState(getEnergy=True, getForces=True, getPositions=True)
                e_total_n = state_n.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                forces_n = state_n.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                fmax = np.max(np.linalg.norm(forces_n, axis=1))
                guard_positions = state_n.getPositions(asNumpy=True)
                guard_box_vectors = state_n.getPeriodicBoxVectors()
                if (not np.isfinite(e_total_n)) or (not np.isfinite(fmax)) or fmax > 10000.0:
                    self._production_disaster_rollback(
                        wc.sim, pos_backup, wc.sampler, prod.production_history_backup_len,
                        stage_type, window_idx, attempt=prod.n_updates + 1,
                        e_total_n=e_total_n, fmax=fmax, win_sys=wc.win_sys, debug_mode=debug_mode,
                        label_prefix="余数补齐", progress_note="",
                        diagnose_prefix=f"窗口{window_idx}_余数补齐回退",
                    )
                    remainder_disaster = True
                else:
                    prod.production_pos_backup = guard_positions
                    prod.production_history_backup_len = _production_history_lengths(wc.sampler)
                    remainder_disaster = False

            if not remainder_disaster:
                # 生产阶段 f_k 已冻结，不再调用 update_weights()，见上方主循环同类注释。
                with _timed(prod.loop_timers, "cv_probe_s"):
                    e = wc.sampler.collect_energies(
                        reuse_positions=guard_positions,
                        reuse_box_vectors=guard_box_vectors,
                    )
                if len(wc.sampler.energy_buffer) >= 10:
                    wc.sampler.energy_buffer = []
                # 🔑 [性能修复：重复落盘] 不在这里再调 _enqueue_window_snapshot()——
                # 紧接着下面无条件执行的"保存能量"区块几行之后会把同一批
                # sampler.*_history 原样重写到完全相同的 dual_window_*.npy 路径，
                # 这里再写一次纯属逐字节重复。
                print(f"    [采样] 补齐余数 {remaining_steps} 步，总步数严格达标 {prod.effective_n_steps_per_window}")
                if debug_mode:
                    has_nan_energy = not np.isfinite(e_total_n)
                    has_nan_e = np.any(np.isnan(e))
                    print(
                        f"    [采样-余数] E_total={e_total_n:.1f}, "
                        f"NaN(E_total)={has_nan_energy}, NaN(E_k)={has_nan_e}"
                    )

        # ---------- 保存能量 ----------
        # 窗口结束时即使样本数不足 100，也必须执行最终失败率门；否则短窗口
        # 中的间歇性丢帧会逃过运行期检查。
        wc.sampler.assert_energy_query_quality(final=True)
        e_arr = np.array(wc.sampler.energy_history) if wc.sampler.energy_history else np.zeros((0, len(wc.lc_win)))
        e_save = e_arr.T if e_arr.size > 0 else np.zeros((len(wc.lc_win), 0))
        _atomic_save_npy(prod.production_energies_path, e_save)
        _atomic_save_npy(
            prod.production_bias_path,
            np.asarray(wc.sampler.bias_history, dtype=np.float64),
        )
        _atomic_save_npy(
            prod.production_base_path,
            np.asarray(wc.sampler.base_energy_history, dtype=np.float64),
        )
        sampling_states_arr = np.asarray(
            wc.sampler.sampling_state_energy_history, dtype=np.float64
        )
        _atomic_save_npy(
            prod.production_sampling_states_path,
            sampling_states_arr,
        )
        _atomic_save_npy(
            prod.production_residual_basis_path,
            np.asarray(wc.sampler.residual_basis_history, dtype=np.float64),
        )
        window_data_metadata = _window_data_metadata(
            prod.production_energies_path,
            prod.production_bias_path,
            prod.production_base_path,
        )
        # 🔑 [2026-09-15] LRC 系数 + 盒体积，供事后离线对账（见下面那个键的注释）。
        # 取不到就写 None —— 不适用（dexp / 膜复合物腿）时它本来就是 None。
        lrc_coeff_for_conv = getattr(
            ibs_wrap, "lj_tail_lrc_coeff_kj_mol", None)
        _lrc_box_volume_nm3 = None
        try:
            _bv = wc.sim.context.getState().getPeriodicBoxVectors(asNumpy=True)
            _lrc_box_volume_nm3 = float(abs(np.linalg.det(
                np.asarray(_bv.value_in_unit(unit.nanometer), dtype=np.float64)
            )))
        except Exception:  # noqa: BLE001 —— 纯诊断，取不到不影响采样
            _lrc_box_volume_nm3 = None
        convergence = {
            "production_segment_protocol_version": PRODUCTION_SEGMENT_PROTOCOL_VERSION,
            "production_segments": _production_segments_snapshot(wc.sampler),
            "stage_protocol_key": getattr(self, "stage_protocol_key", None),
            "window_idx": int(window_idx),
            "stage_type": stage_type,
            # 🔑 [2026-09-15] 这个窗口的帧是不是在一份**从未通过冻结验证**的
            # f_k 下采的（临时生产）。True 时它仍是合法的 MBAR 数据（单一固定
            # 采样分布），但**不是可信 PASS** —— 采不采用由生产后的支撑门判。
            # 落在 convergence.json 里，是为了让"这份 ΔG 里含临时生产窗口"
            # 这件事跟着数据走，而不是只活在日志里。
            "provisional_production": bool(warm.provisional_production),
            # 该窗口这次实际采样的 λ 值——不是"这个位置理论上该有什么"，是这份
            # 能量文件里的每一行真实对应哪个 λ。resume 断点续传 / abfe_pipeline.py
            # 的窗口产物复用逻辑都靠这个字段做内容校验，而不是只信任 window_idx。
            "lambdas_coul": [float(x) for x in wc.lc_win],
            "lambdas_vdw": [float(x) for x in wc.lv_win],
            # base/bias 的力组切分口径版本，见 WCA_ACCOUNTING_VERSION 定义处；
            # resume / 窗口产物复用逻辑必须校验这个字段，不能只比 λ 值。
            "wca_accounting_version": WCA_ACCOUNTING_VERSION,
            # IBS 偏置预热/冻结协议版本，见 IBS_BIAS_PROTOCOL_VERSION 定义处；
            # 旧协议下这份能量文件可能是在 f_k 全程漂移的情况下采出来的，
            # 不满足 MBAR 的单一固定采样分布假设，同样必须校验，不能只看 λ 值。
            "ibs_bias_protocol_version": IBS_BIAS_PROTOCOL_VERSION,
            # 🔑 [ligand_com_restraint_protocol_version] Group 5 配体 COM 限制力
            # 的协议版本。v1 的非周期绝对锚点实现在 CUDA 上产生定向拖拽，那种
            # 轨迹不满足 MBAR 的平衡采样前提；stage 级指纹拦不住**逐窗口**的
            # 能量数组复用（它发生在建 Context 之前），所以必须逐窗口落盘并在
            # _resume_cached_window_gate_status 里校验。
            "ligand_com_restraint_protocol_version": (
                LIGAND_COM_RESTRAINT_PROTOCOL_VERSION
            ),
            "lse_log_residual_tolerance": float(lse_log_residual_tolerance),
            # 🔑 [TRADITIONAL_LJ_LRC_PROTOCOL_VERSION] LJ 长程尾项修正公式版本
            # （尽管名字里写着 traditional，v2 起这个常量同时覆盖 ACE/dual_lambda
            # 路径和传统 Beutler REMD 路径共用的 switching+softcore-aware LRC
            # 积分——见该常量定义处）。旧版本（v1，只补 cutoff 外的 r^-6、忽略
            # switching 区间）下的能量文件里，target_energies 里叠加的 LRC 数值
            # 跟当前公式不是同一回事，resume / 窗口产物复用逻辑必须校验这个
            # 字段，不能只看 λ/WCA/IBS 偏置协议是否匹配。
            "lj_tail_lrc_protocol_version": TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
            # 🔑🔑 [2026-09-15] **把系数本身也落盘，不只落协议版本号。**
            #
            # `coeff[k]` 先前只活在 `ibs_wrapper.lj_tail_lrc_coeff_kj_mol`
            # （内存）里，盘上只有一个版本号 ⟹ 事后想核对「LRC 对这条腿贡献
            # 多少」必须从 system XML 把 σ/ε 全部重建、重跑
            # `_lj_tail_correction_sigma_resolved_moments` +
            # `_lj_tail_lrc_coefficients_kj_mol` 复算一遍。真机上为了排除
            # 「两腿 LRC 差是不是那个 −1.85 kJ/重原子系统项的来源」就这么干过一次
            # （结论：不是，两腿只差 −0.59 kJ/mol ≈ −0.0105/原子，因为
            #  `coeff ∝ N_env` ⟹ `coeff/V ∝ 数密度`，两个盒子的 ρ_env
            #  只差 0.2%，6.7 倍体积差被抵消）。
            #
            # 落盘之后这类对账是**读文件**，不是重算。逐 λ 一个数，成本可忽略。
            # ⚠️ 纯诊断字段：`_resume_cached_window_gate_status` 的十道门一个都
            # 不读它，加它不影响任何缓存身份。
            "lj_tail_lrc_coeff_kj_nm3_per_mol": (
                [float(x) for x in lrc_coeff_for_conv]
                if lrc_coeff_for_conv is not None else None
            ),
            # 每帧修正 = coeff/V，所以没有 V 那串系数读出来也没法换算成能量。
            "lj_tail_lrc_box_volume_nm3": _lrc_box_volume_nm3,
            # 🔑 [vdw_nonbonded_protocol_version，MEM-00h] softcore cutoff/
            # switching 协议版本（1.2nm+switch → 1.0nm 无switch）。只在
            # stage_type=="vdw" 时写真实版本号——Stage 1 charging 不构造软核
            # CV，这个字段对它没有意义，写 None 也不会被 _resume_cached_
            # window_gate_status 拿来判它无效（该门对 stage_type!="vdw" 恒为
            # True，见该函数定义处）。
            "vdw_nonbonded_protocol_version": (
                VDW_NONBONDED_PROTOCOL_VERSION if stage_type == "vdw" else None
            ),
            # 🔑 [non_mutating_v1] 采样修复策略。旧的变异策略可能在采样中途用
            # 累计 ΔF 就地覆盖过 f_k（不同参考系）；resume / 窗口产物复用必须校
            # 验这个字段，绝不能把旧策略缓存当成非变异策略的有效数据复用。
            "sampling_repair_policy": repair_policy,
            "sampling_score_sha256": self.sampling_score_sha256,
            "residual_sampling_protocol_version": (
                IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION
                if self.sampling_score_sha256 is not None
                else None
            ),
            # [B5] Completed-window energy reuse happens before checkpoint
            # manifests are consulted, so bind the energy triplet itself to
            # the frozen runtime co-ion identity.
            "coion_identity": self.coion_identity,
            "n_steps_per_window_default": int(n_steps_per_window),
            "n_steps_per_window_effective": int(prod.effective_n_steps_per_window),
            "n_updates": int(len(wc.sampler.f_history)),
            # [P1-19] 命名修正：sampler.f_history 存的是 f_new（物理 F_k，
            # 单位 kJ/mol，见 _solve_tmbar_and_recenter 的说明），不是无量纲
            # 的 kT 归一化值——Context 里 `{prefix}_f_{k}` 全局参数与
            # cv_k_int/cv_k_rest 同单位相减（见 IBSBiasWrapper 的 CV 表达式
            # 构造），本来就是 kJ/mol。旧字段名 `free_energy_history_kT`
            # 只是记错了单位，从未被任何 resume/复用逻辑读取，改名安全。
            "free_energy_history_kJ_mol": [
                np.asarray(f_k, dtype=float).tolist()
                for f_k in wc.sampler.f_history
            ],
            "n_energy_samples": int(e_save.shape[1]),
            "energy_query_diagnostics": wc.sampler.energy_query_diagnostics(),
            # 🔑 [性能计时，纯诊断字段] 整窗累计的"积分/guard/CV-probe/
            # ledger-IO"墙钟耗时（秒）。不参与任何 resume/协议指纹比较——
            # resume 逻辑只按字段名 .get(...) 读取自己关心的键
            # （cumulative_production_steps 等），新增诊断字段不会破坏它。
            "loop_timing_s": dict(prod.loop_timers),
            "window_data_protocol_version": IBS_WINDOW_DATA_PROTOCOL_VERSION,
            "window_data": window_data_metadata,
            "bias_warmup": warm.bias_warmup_diag,
            # 🔑 [EARLY_STOP_PROTOCOL_VERSION] 见 run_all_windows 参数说明和该
            # 常量定义处。actual_production_steps 是这个窗口真正跑了多少步
            # （早停时 < n_steps_per_window_effective，未早停/未启用时等于它）；
            # resume 时只有 early_stop_triggered=True 且协议版本/enable_early_stop/
            # 目标步数都与当前调用一致才允许复用这份缓存，见上面 resume 分支
            "joint_score_window_data": {
                "sampling_states": {
                    "shape": list(sampling_states_arr.shape),
                    "dtype": str(sampling_states_arr.dtype),
                    "sha256": _sha256_file(prod.production_sampling_states_path),
                },
                "protocol_version": IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION,
                "residual_basis": {
                    "shape": [len(wc.sampler.residual_basis_history)],
                    "dtype": "float64",
                    "sha256": _sha256_file(prod.production_residual_basis_path),
                },
                "sampling_state_definition": "softcore_U_k_plus_A_k_times_residual_basis_minus_offset",
                "physical_target_excludes_residual": True,
            },
            # 的 early_stop_ok 校验。
            # 未提前停止时，余数补齐（remaining_steps）会让总步数严格达到
            # n_steps_per_window_effective；只有提前停止时才用真正跑到的
            # update 数折算实际步数。
            "actual_production_steps": (
                int(prod.actual_production_updates * steps_per_update)
                if prod.early_stop_triggered
                else int(prod.effective_n_steps_per_window)
            ),
            # 🔑 [production checkpoint 续采] 这是跨多次 reseed_resample 续算
            # 累加的总步数（不是这次 attempt 单独跑了多少步），供下一次
            # run_all_windows 调用时判断"已有的累计步数是否还没到目标"，
            # 从而只跑差值、真正续算，而不是把 effective_n_steps_per_window
            # 当成"这次要新跑这么多步"（那样会变成每次延长都重复计入已经跑
            # 过的步数）。窗口真正收敛/被判定终态失败时，这个字段的语义就
            # 不再重要——下一次若 λ/窗口/f_k 变了，_production_window_
            # checkpoint_is_usable 的指纹比对会自然拒绝复用这份累计值。
            "cumulative_production_steps": (
                int(prod.prior_cumulative_production_steps) + int(prod.actual_production_updates * steps_per_update)
                if prod.early_stop_triggered
                # 未早停时余数补齐让这次 attempt 严格跑完 remaining_production_steps
                # （= effective_n_steps_per_window - prior_cumulative_production_steps），
                # 累计总数因此就等于目标本身，不需要再分别相加。
                else int(prod.effective_n_steps_per_window)
            ),
            "early_stop_enabled": bool(es.enable_early_stop),
            "early_stop_triggered": bool(prod.early_stop_triggered),
            "early_stop_stop_reason": prod.early_stop_stop_reason,
            "early_stop_protocol_version": EARLY_STOP_PROTOCOL_VERSION if es.enable_early_stop else None,
            "early_stop_check_history": prod.early_stop_check_history,
            "early_stop_config": {
                "min_steps": int(es.early_stop_min_steps),
                "check_interval_steps": int(es.early_stop_check_interval_steps),
                "required_consecutive_passes": int(es.early_stop_required_consecutive_passes),
                "min_ess_ratio": float(es.early_stop_min_ess_ratio),
                "min_absolute_ess": float(es.early_stop_min_absolute_ess),
                "min_decorrelated_samples": int(es.early_stop_min_decorrelated_samples),
                "max_delta_g_drift_kJ_mol": float(es.early_stop_max_delta_g_drift_kJ_mol),
                "max_uncertainty_kJ_mol": float(es.early_stop_max_uncertainty_kJ_mol),
            } if es.enable_early_stop else None,
        }
        if self.sampling_score_sha256 is not None:
            convergence["residual_sampling"] = {
                "feature": getattr(
                    self,
                    "residual_feature_name",
                    "Outer-Lambda Local Residual for IBS",
                ),
                "em_policy": getattr(
                    self, "residual_em_policy", "no_residual_twin"
                ),
                "plugin_identity": getattr(
                    self, "residual_plugin_identity", None
                ),
            }
        # 🔑 [原子写修复] 这里之前是普通 open(...)+json.dump，不是这个文件
        # 别处到处使用的 _atomic_write_json（先写临时文件再 os.replace）——
        # 进程在这次写入中途被杀，会留下一份截断/损坏的 convergence.json，
        # 而 energies/bias/base.npy 这三个数组已经通过 _atomic_save_npy
        # 安全落盘，二者的原子性保证不一致。改为同样的 temp+replace 模式。
        _atomic_write_json(
            os.path.join(self.output_dir, f"dual_window_{window_idx}_{stage_type}_convergence.json"),
            convergence,
        )
        # 🔑 [production checkpoint 续采] 窗口生产采样正常结束（无论是否早停）
        # 时再存一次最新的生产 checkpoint——每 100 个 update 的周期性落盘之外，
        # 补上循环结束、余数补齐这段尾巴，保证 checkpoint 反映的是这次 attempt
        # 真正结束时的坐标/速度/积分器 RNG 状态，跟上面刚写的
        # cumulative_production_steps 一致。不在这里删除这份 checkpoint——
        # 它可能还会被下一轮 reseed_resample 用来继续续算；只有 production
        # 数据真的因为 f_k/λ 改变而被作废时才应该删（见
        # _invalidate_production_window_checkpoint 的调用点）。
        _atomic_save_openmm_checkpoint(wc.sim, prod.production_ckpt_path)
        _atomic_write_json(prod.production_manifest_path, prod.expected_production_manifest)

        # 🔑 [2026-09-11 / PLAN P2-9c] **每个窗口跑完就判一次它自己够不够。**
        # 老板原话："每个窗口跑完就判一次，这不是更早判定吗"。
        # 实测浪费：5×250k=125 万步全部烧完才做第一次预算判断，然后五个一起
        # 翻倍重来 —— 而判据（该窗口去相关后剩几帧）在它刚跑完这一刻就算得出来。
        # **只判不动手**：落盘 + 打印，不改 production_step_overrides、
        # 不碰 converged / rescue 的任何既有判据。
        try:
            _self = window_self_support_check(
                self.output_dir, stage_type, window_idx, self.kt,
            )
            if _self is None:
                print(
                    f"  [自检] 窗口 {window_idx} 的支撑自检不可用"
                    "（产物缺失或形状不符），跳过；不影响采样。"
                )
            else:
                _atomic_write_json(
                    window_self_support_path(
                        self.output_dir, stage_type, window_idx
                    ),
                    _self,
                )
                # ⚠️ **两个判据不许拼进同一句。** 原来 `_mark`（来自
                # `sufficient = verdict == ANALYSIS_ELIGIBLE`，是**支撑度**判据）
                # 被接在"去相关帧数 vs 门槛"后面，于是真机印出
                # 「去相关后 21/1000 帧（门槛 10）⟹ **不够**」—— 21 ≥ 10，
                # 去相关这一关明明过了。把病因指错的后果是实打实的：
                # 读的人去加帧，而真正不达标的是 min N_eff/g，加帧只会更糟
                # （实测 win4 500→1000 帧，min N_eff/g 3.44→2.29、g 15.1→49.2）。
                _dec_ok = (
                    int(_self["n_frames_decorrelated"])
                    >= int(_self["min_frames_per_window"])
                )
                print(
                    f"  [自检] 窗口 {window_idx} 求解器资格："
                    f"去相关后 {_self['n_frames_decorrelated']}"
                    f"/{_self['n_frames_total']} 帧 vs 下限 "
                    f"{_self['min_frames_per_window']} ⟹ "
                    f"{'够' if _dec_ok else '**不够**'}；"
                    f"g={_self['statistical_inefficiency']}"
                )
                print(
                    f"  [自检] 窗口 {window_idx} **验收量** min N_eff/g="
                    f"{_self.get('min_n_eff_over_g')}（门 10）"
                    f" 最差态 local={_self['worst_target_state_local']}"
                    f" λ={_self['worst_target_lambda_vdw']}"
                    f" | verdict={_self['verdict']}"
                )
                if not _self["sufficient"]:
                    _short = _self.get("frames_short_by")
                    _why = (
                        f"去相关帧数还差 {_short} 帧"
                        if _short else
                        "**不是帧数不够**（去相关下限已达标）——"
                        "差的是目标态的有效样本 min N_eff/g。"
                        "同分布加帧治不了偏斜：对症动作是换 f_k（重加权）"
                        "或缩窗跨度（插 λ），不是更多帧"
                    )
                    print(
                        f"  [自检] ⚠️ 窗口 {window_idx}：{_why}。"
                        "INSUFFICIENT_DATA ≠ FAIL：这是「还没测够」，不是「不合格」。"
                        "本检查只判不动手。"
                    )
        except Exception as _self_err:  # noqa: BLE001 —— 诊断不得中断采样
            print(f"  [自检] 计算失败（不影响采样）：{_self_err!r}")

        # 🔑 [2026-09-11 / PLAN P2-9a] **join λ 两侧支撑：自动做掉。**
        # 老板的原话是"我要的是 win0、win1、**win0+win1 验证** —— 自动的"。
        # 前两个本来就自动（各自的 loose gate），第三个此前完全不存在 ——
        # 那些 join 数字一直是人手算的。
        #
        # 触发点定在这里：本窗口的产物（energies/bias/sampling_states/
        # convergence + manifest）**全部持久化之后**，与**前一个**窗口比一次。
        # 只比两边共享的那一个 λ，滑动相邻对、不做累积前缀（§3bis.3）。
        #
        # 只报告、不设门、失败不中断采样 —— 它是证据不是裁决。
        if window_idx > 0:
            try:
                _join = join_lambda_two_sided_support(
                    self.output_dir, stage_type,
                    window_idx - 1, window_idx, self.kt,
                )
                if _join is None:
                    print(
                        f"  [join λ] 窗口 {window_idx-1}↔{window_idx} 暂不可比"
                        "（上游产物缺失或两窗不共享 λ），跳过；不影响采样。"
                    )
                else:
                    _atomic_write_json(
                        join_support_path(
                            self.output_dir, stage_type, window_idx - 1, window_idx
                        ),
                        _join,
                    )
                    _u, _d = _join["upstream"], _join["downstream"]
                    print(
                        f"  [join λ={_join['join_lambda_vdw']:.4f}] "
                        f"上游 win{window_idx-1}: rawESS={_u['raw_ess']}"
                        # [审计 #54] 标签是 g 就读 g 那个键，别再读 tau_int。
                        f"/{_u['n_finite']} g={_u['statistical_inefficiency_g']}"
                        f" | 下游 win{window_idx}: rawESS={_d['raw_ess']}"
                        f"/{_d['n_finite']} g={_d['statistical_inefficiency_g']}"
                        f" | 下游/上游={_join['downstream_over_upstream_raw_ess']}"
                        f"（{_join['asymmetry_direction']}，口径 {_join['gauge']}）"
                        " —— 只报告，不参与放行。"
                    )
            except Exception as _join_err:  # noqa: BLE001 —— 诊断不得中断采样
                print(f"  [join λ] 计算失败（不影响采样）：{_join_err!r}")
        # 🔑 [2026-08-27，见 docs/experiments/EXP-030_FROZEN_SNAPSHOT_TIMING_BUG_2026-08-26.md
        # 补验收] 生产结束时必须再存一次 IBS JSON：这是下游冻结检查唯一会
        # 读取的文件（见 exp030_window_state_machine.py 的 state_path），
        # 之前只存 OpenMM checkpoint + manifest，IBS JSON 从生产开始前
        # （12461 附近那次 save）起就再没刷新过，导致 final_state["f_k"]
        # 其实是生产开始前的值，不是生产结束时的值。save_ibs_state()
        # 内部从当前（仍未 del 的）sim.context 重新读 f_k 写入 "f_k" 字段，
        # 同时把上面已经落盘、生产期间从未再赋值过的 production_entry_f_k
        # 原样带出——JSON 里因此同时有"生产开始时"和"生产结束时"两份值，
        # 供下游做真正的两段比较，而不是只有构造出来测试用的假数据。
        wc.sampler.save_ibs_state(
            ibs_state_file, wc.lc_win, wc.lv_win, stage_type=stage_type
        )
        print(f"  窗口 {window_idx} 完成，能量已保存 ({e_save.shape})")
