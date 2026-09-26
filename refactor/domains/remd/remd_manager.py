# ============================================================================
# 6. 传统λ-REMD 采样引擎 (从 traditional_abfe_remd.py 迁移)
# ============================================================================
class REMDManager:
    """传统 λ-REMD 引擎：相邻窗口 Metropolis 交换 + 轨迹落盘"""
    class _ReporterIntegratorView:
        def __init__(self, step_size):
            self._step_size = step_size

        def getStepSize(self):
            return self._step_size

    class _ReporterSimulationView:
        def __init__(self, topology, current_step: int, step_size):
            self.topology = topology
            self.currentStep = int(current_step)
            self.integrator = REMDManager._ReporterIntegratorView(step_size)

    def __init__(
        self,
        system_template: openmm.System,
        topology: app.Topology,
        positions,
        box_vectors,
        ligand_indices: List[int],
        lambdas_coul: List[float],
        lambdas_vdw: List[float],
        temperature: float = 300.0,
        platform_name: str = "CUDA",
        output_dir: str = "./remd_output",
        boresch_params: Optional[Dict] = None,
        random_seed: Optional[int] = None,
        max_resident_contexts: Optional[int] = None,
        co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
        seed_ledger: Optional[Exp019SeedLedger] = None,
        seed_stage: Optional[str] = None,
        seed_leg: Optional[str] = None,
        seed_phase: Optional[str] = None,
    ):
        self.topology = topology
        self.positions = positions
        self.box_vectors = box_vectors
        self.ligand_indices = ligand_indices
        # [MEM-00c] 冻结的 co-ion 身份。带电配体时必须由调用方传入
        # （`ibs_engine.select_co_alchemical_ion_once()` 选一次的产物）；
        # 中性配体为 None。所有 replica 共用它，不各自重选。
        self.co_alchemical_ion_spec = co_alchemical_ion_spec
        self.seed_ledger = seed_ledger
        self.seed_stage = str(seed_stage) if seed_stage is not None else "remd"
        self.seed_leg = str(seed_leg) if seed_leg is not None else None
        # 🔑 [0831issue P2] 种子域的 phase 以前在 `_build_replicas` 里**硬编码**成
        # "charging"，而这个类同时服务 mixed / vanishing / shadow_bridge 等腿：
        # ledger 快照会把非 charging 腿记在 charging 键下（审计失真），且同一条腿的
        # 两个 stage 共用 seed_stage 时会撞种子（当前潜伏）。
        # 现在 phase 可由构造方注入。**默认仍是 "charging"**：改这个字符串会改变
        # 派生出来的种子、也就是改变随机流，那属于协议变更，不能顺手做（见
        # RELEASE_READINESS "不要为旧 resume 偷补新 seed 或更改随机流"）。
        # 调用方在准备好承担冷启动代价时显式传 seed_phase 即可。
        self.seed_phase = str(seed_phase) if seed_phase is not None else "charging"
        if self.seed_ledger is not None and self.seed_leg != self.seed_ledger.leg:
            raise ValueError("REMD seed_leg 与 EXP-019 seed ledger 不一致")
        self.lambdas_coul = np.array(lambdas_coul)
        self.lambdas_vdw = np.array(lambdas_vdw)
        if self.lambdas_coul.ndim != 1 or self.lambdas_vdw.ndim != 1:
            raise ValueError("REMD lambda arrays must be one-dimensional")
        if len(self.lambdas_coul) < 2:
            # This engine performs adjacent-state exchanges.  A one-state
            # object is not REMD and used to reach the exchange-rate reduction
            # with a 0/0 denominator; reject it at construction instead of
            # silently reporting a non-exchange simulation as REMD.
            raise ValueError("REMD 至少需要两个 replica/lambda 状态")
        if len(self.lambdas_coul) != len(self.lambdas_vdw):
            raise ValueError("lambdas_coul 与 lambdas_vdw 长度必须一致")
        if not np.all(np.isfinite(self.lambdas_coul)) or not np.all(np.isfinite(self.lambdas_vdw)):
            raise ValueError("REMD lambda 必须全部为有限数")
        if np.any((self.lambdas_coul < 0.0) | (self.lambdas_coul > 1.0)):
            raise ValueError("lambdas_coul 必须位于 [0, 1]")
        if np.any((self.lambdas_vdw < 0.0) | (self.lambdas_vdw > 1.0)):
            raise ValueError("lambdas_vdw 必须位于 [0, 1]")
        self.n_replicas = len(lambdas_coul)
        self.has_coulomb_scaling = not np.allclose(
            self.lambdas_coul,
            self.lambdas_coul[0],
        )
        self.has_vdw_scaling = not np.allclose(
            self.lambdas_vdw,
            self.lambdas_vdw[0],
        )
        self.is_pme_coulomb_leg = (
            np.allclose(self.lambdas_vdw, 1.0)
            and self.has_coulomb_scaling
        )
        self.is_mixed_pme_alchemical = self.has_coulomb_scaling and self.has_vdw_scaling
        temp_k = (
            float(temperature.value_in_unit(unit.kelvin))
            if hasattr(temperature, "value_in_unit")
            else float(temperature)
        )
        if not np.isfinite(temp_k) or temp_k <= 0.0:
            raise ValueError(f"REMD temperature 必须为正有限数，收到 {temperature!r}")
        self.temperature = temp_k * unit.kelvin
        self.kt = (unit.MOLAR_GAS_CONSTANT_R * self.temperature).value_in_unit(unit.kilojoule_per_mole)
        self.beta = 1.0 / self.kt
        self.platform_name = platform_name
        requested_platform, _ = _split_platform_spec(platform_name)
        if max_resident_contexts is None:
            # 🔑 [2026-07-27] 这里原来对 CUDA/OPENCL 默认 **1**。但当前交换实现
            # 天生要求每个 replica 同时持有独立 Context（`context_residency_mode
            # = "all_resident"`），于是「n_replicas > 1」恒成立 → **任何** REMD
            # 都会在建 Context 之前静默回退 CPU，GPU 路径事实上不可达。
            #
            # 代价极不对称：
            #   · CPU 回退让整个阶段慢约两个数量级（实测 73536 原子 × 12 副本 ×
            #     PME：GPU 上 ~24 分钟跑完 500 轮交换；回退 CPU 后 23 分钟连第一个
            #     DCD 帧都没写出来）；而且这个决定原先只 print 到终端、
            #     `pipeline.log` 里完全看不见，表现成"卡住了"。
            #   · 真正的 GPU OOM **本来就有优雅处理**：`_build_replicas` 的
            #     `except` 分支会 `_clear_replica_contexts()` 再回退 CPU 重建
            #     （`_is_gpu_context_failure`）。OOM 是响亮且立即的，慢 100 倍是静默的。
            #
            # 实测 12 个 73536 原子 PME Context 在 11 GB RTX 2080 Ti 上装得下
            # （`--only-complex-charging` 那条路径一直这么跑，manifest 记录
            # `platform_name: CUDA`、`platform_fallback_reason: None`）。
            # 所以默认改为「不预防性回退」，把判断交给真实的构建期 OOM 处理；
            # 显存小的机器仍可显式传小值来强制 CPU 回退。
            max_resident_contexts = self.n_replicas
        self.max_resident_contexts = int(max_resident_contexts)
        if self.max_resident_contexts < 1:
            raise ValueError("max_resident_contexts 必须至少为 1")
        self.context_residency_mode = "all_resident"
        # 当前交换实现要求每个 replica 同时持有独立 Context。与其在单 GPU 上
        # 先创建 N 个 Context、OOM 后才补救，不如在构建前 fail-safe 地切到 CPU。
        # 这保持算法和 RNG/交换语义不变，同时保证 GPU 常驻 Context 不超过上限。
        if (
            requested_platform.upper() in {"CUDA", "OPENCL"}
            and self.n_replicas > self.max_resident_contexts
        ):
            # 只有调用方**显式**传了小于 n_replicas 的上限才会走到这里。
            # 这个决定会让整个阶段慢约两个数量级，必须响亮且可在归档日志里看见——
            # 用 logger.warning 而不是裸 print，避免只出现在终端 stdout。
            _msg = (
                f"[WARN] REMD replica 数({self.n_replicas})超过显式指定的 GPU 常驻 Context "
                f"上限({self.max_resident_contexts})；在创建任何 GPU Context 前回退 CPU。"
                "\n     注意：CPU 回退会让本阶段慢约两个数量级（实测 73536 原子 × "
                "12 副本 × PME：GPU ~24 分钟跑完 500 轮交换，CPU 上 23 分钟连第一帧 "
                "DCD 都写不出来，表现得像卡死）。"
                "\n     若显存足够，请提高 max_resident_contexts（完整流程用 "
                "--charging-max-resident-contexts）；真实 OOM 由构建期回退兜底。"
            )
            logger.warning(_msg)
            print(f"  {_msg}")
            self.platform_name = "CPU"
            self.context_residency_mode = "cpu_fallback_bounded_gpu_contexts"
            self.platform_fallback_reason = (
                f"explicit_max_resident_contexts={self.max_resident_contexts}"
                f"_below_n_replicas={self.n_replicas}"
            )
        self.output_dir = output_dir
        self.boresch_params = boresch_params
        if random_seed is None:
            seed_env = os.environ.get("IBS_RANDOM_SEED") or os.environ.get("ABFE_RANDOM_SEED")
            random_seed = int(seed_env) if seed_env not in (None, "") else None
        self.random_seed = random_seed
        if self.seed_ledger is not None and self.random_seed is None:
            # [0831issue P2] 与 _build_replicas 用同一个 phase（默认仍是 "charging"，
            # 见 self.seed_phase 的说明——改它会改随机流）。
            self.random_seed = self.seed_ledger.derive(
                self.seed_phase, self.seed_stage, "exchange", "numpy"
            )
        self.rng = np.random.default_rng(self.random_seed)
        os.makedirs(output_dir, exist_ok=True)

        self.contexts = []
        self.integrators = []
        self.replica_systems = []
        self._atoms = list(topology.atoms()) if topology is not None else []
        self._ligand_set = set(int(i) for i in ligand_indices)
        self._state_to_context = list(range(self.n_replicas))
        self._context_to_state = list(range(self.n_replicas))
        self._steps_completed = 0
        self._is_warmed_up = False
        # 不要无条件置 None：上面「显式上限低于 n_replicas → 预防性回退 CPU」那个
        # 分支已经写过这个字段，无条件覆盖会把回退原因抹掉，让 exchange_diagnostics
        # 里显示 platform_fallback_reason=None 而实际却在 CPU 上跑。
        if getattr(self, "platform_fallback_reason", None) is None:
            self.platform_fallback_reason = None
        self._build_replicas(system_template)

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

    @staticmethod
    def _try_set_context_parameter(context, name: str, value: float) -> None:
        try:
            context.setParameter(name, float(value))
        except Exception as exc:
            msg = str(exc)
            if "invalid parameter name" not in msg:
                raise

    def _clear_replica_contexts(self) -> None:
        """Release already-created OpenMM Contexts after a partial REMD build failure."""
        self.contexts = []
        self.integrators = []
        self.replica_systems = []
        self._state_to_context = list(range(self.n_replicas))
        self._context_to_state = list(range(self.n_replicas))
        gc.collect()

    @staticmethod
    def _is_gpu_context_failure(exc: Exception) -> bool:
        msg = str(exc).lower()
        needles = (
            "no compatible cuda device",
            "cuda_error_out_of_memory",
            "out of memory",
            "failed to create cuda context",
            "cucontextcreate",
            "cuda device",
        )
        return any(token in msg for token in needles)

    def _build_replicas(self, system_template, allow_platform_fallback: bool = True):
        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)

        # [P0-REMD-CUDA] 建 Context 前后记显存。
        #
        # 为什么必须记：`No compatible CUDA device is available` 是 OpenMM 在**所有**
        # 设备都初始化失败时给的**通用文案**，真 OOM 也长这样。没有"第几个 replica
        # 断的 / 断时剩多少显存"这两个数，"显存不够"和"其它初始化失败"就区分不了，
        # 只能反复猜。离线探针（`memtest/probe_remd_context_capacity.py`）已证明
        # 12 × 45354 原子的 Context 单进程能建满、每个 ≈ 338 MiB，
        # 连"先跑一遍 λ 预优化"都复现不了失败 —— 所以差异只可能在生产路径上，
        # 必须就地打点。
        _vram_baseline = _gpu_memory_mib()
        if _vram_baseline and str(resolved_platform).upper() == "CUDA":
            logger.info(
                "[REMD] 建 %d 个 %s Context 之前显存: used=%d free=%d total=%d MiB "
                "(props=%s)",
                self.n_replicas, resolved_platform,
                _vram_baseline[0], _vram_baseline[1], _vram_baseline[2], props,
            )
        # 宿主内存同样要打：这一段里第一个大分配是 `_prepare_pme_coulomb_leg_system`
        # 的 XML 往返，宿主侧耗尽会表现成 `std::bad_alloc` 而不是任何显存信息。
        _host_baseline = _host_memory_mib()
        if _host_baseline:
            logger.info(
                "[REMD] 建 replica 之前宿主内存: RSS=%d MiB (进程峰值 %d MiB)",
                _host_baseline[0], _host_baseline[1],
            )

        try:
            if (
                str(resolved_platform).upper() in {"CUDA", "OPENCL"}
                and self.n_replicas > self.max_resident_contexts
            ):
                raise RuntimeError(
                    "REMD GPU Context 需求超过 max_resident_contexts；"
                    "拒绝无界创建 replica Context"
                )
            for i in range(self.n_replicas):
                if self.is_pme_coulomb_leg:
                    replica_sys = _prepare_pme_coulomb_leg_system(
                        system_template,
                        self.ligand_indices,
                        lambda_name="lambda_coul",
                        allow_charged_ligand=True,
                        topology=self.topology,
                        positions=self.positions,
                        box_vectors=self.box_vectors,
                        # [MEM-00c] 所有 replica 共用同一份冻结身份，不各自重选。
                        co_alchemical_ion_spec=self.co_alchemical_ion_spec,
                    )
                    _add_physical_boresch_restraint(replica_sys, self.boresch_params, force_group=3)
                elif self.is_mixed_pme_alchemical:
                    replica_sys = _prepare_pme_mixed_alchemical_system(
                        system_template,
                        self.ligand_indices,
                        topology=self.topology,
                        positions=self.positions,
                        box_vectors=self.box_vectors,
                        lambda_coul_name="lambda_coul",
                        lambda_vdw_name="lambda_vdw",
                        restraint_params=self.boresch_params,
                        co_alchemical_ion_spec=self.co_alchemical_ion_spec,
                    )
                else:
                    # [2026-09-10] 同上：往返 → clone()。这一处在 per-replica 循环体内，
                    # 每个 replica 都要开一份完整 System 的字符串。
                    replica_sys = openmm.XmlSerializer.clone(system_template)
                    replica_sys.thisown = 1
                    nb = [f for f in replica_sys.getForces() if isinstance(f, openmm.NonbondedForce)][0]
                    _restore_ligand_internal_nonbonded(
                        replica_sys,
                        nb,
                        self.ligand_indices,
                        zero_original_exceptions=True,
                    )
                    env_idx = [j for j in range(replica_sys.getNumParticles()) if j not in self.ligand_indices]
                    sc_force = BeutlerSoftcoreBuilder.build(nb, self.ligand_indices, env_idx)
                    sc_force.setForceGroup(1)
                    replica_sys.addForce(sc_force)
                    _add_physical_boresch_restraint(replica_sys, self.boresch_params, force_group=3)

                    for idx in self.ligand_indices:
                        nb.setParticleParameters(idx, 0.0, 0.1*unit.nanometer, 0.0)

                integ = openmm.LangevinMiddleIntegrator(self.temperature, 1.0/unit.picosecond, 0.002*unit.picosecond)
                integrator_seed = self._seed_for(
                    self.seed_phase, self.seed_stage, i, "integrator"
                )
                if integrator_seed is not None:
                    integ.setRandomNumberSeed(integrator_seed)
                ctx = openmm.Context(replica_sys, integ, platform, props)
                ctx.setPositions(self.positions)
                if self.box_vectors is not None:
                    ctx.setPeriodicBoxVectors(*self.box_vectors)
                self._try_set_context_parameter(ctx, "lambda_coul", self.lambdas_coul[i])
                self._try_set_context_parameter(ctx, "lambda_vdw", self.lambdas_vdw[i])
                velocity_seed = self._seed_for(
                    self.seed_phase, self.seed_stage, i, "velocity"
                )
                if velocity_seed is None:
                    ctx.setVelocitiesToTemperature(self.temperature)
                else:
                    ctx.setVelocitiesToTemperature(self.temperature, velocity_seed)

                self.contexts.append(ctx)
                self.integrators.append(integ)
                self.replica_systems.append(replica_sys)
        except Exception as exc:
            # ⚠️ 显存必须在 `_clear_replica_contexts()` **之前**读——释放之后再读就
            # 只剩一个"失败后已回收"的数，判不了当时到底是不是不够用。
            _vram_at_failure = _gpu_memory_mib()
            # 同理，宿主内存也必须在释放之前读。`std::bad_alloc` 只说"分配失败"，
            # 不说是谁占着——没有这个数就分不清"体系太大"和"前面某段泄漏"。
            _host_at_failure = _host_memory_mib()
            if _host_at_failure:
                logger.error(
                    "[REMD] 建 replica 失败瞬间宿主内存: RSS=%d MiB (进程峰值 %d MiB)"
                    "%s",
                    _host_at_failure[0], _host_at_failure[1],
                    (
                        f"，建 replica 之前是 RSS={_host_baseline[0]} MiB"
                        if _host_baseline else ""
                    ),
                )
            _n_built = len(self.contexts)
            self._clear_replica_contexts()
            # 🔑 [2026-09-09] `_clear_replica_contexts()` 只清那三个 list —— 失败那一轮
            # 的循环局部量 `ctx` / `integ` / `replica_sys` 仍然绑在**本函数的栈帧**上。
            # 而下面的 CPU 回退重建就跑在同一个 except 帧里（`_build_replicas(...,
            # allow_platform_fallback=False)`），于是"最后一个 GPU Context + 一份完整
            # System"会与 N 个新建的 CPU Context 同时存在 —— 恰好是内存已经不够的时刻，
            # 也恰好是日志刚打完"已释放已创建的 replica contexts"的时刻。
            ctx = None
            integ = None
            replica_sys = None
            gc.collect()
            if (
                allow_platform_fallback
                and str(resolved_platform).upper() in {"CUDA", "OPENCL"}
                and self._is_gpu_context_failure(exc)
            ):
                self.platform_fallback_reason = str(exc)
                # 判 OOM / 非 OOM：断点还剩得下一个 Context 的量 ⟹ **不是**显存不够，
                # 别按 OOM 修（把 λ 数减小只是掩盖）。阈值取实测的每 Context 量级。
                _diagnosis = "显存读数不可用（无 nvidia-smi？）"
                if _vram_at_failure:
                    _per_ctx = (
                        (_vram_at_failure[0] - _vram_baseline[0]) / max(1, _n_built)
                        if _vram_baseline and _n_built else float("nan")
                    )
                    _looks_like_oom = _vram_at_failure[1] < max(
                        _REMD_CONTEXT_VRAM_FLOOR_MIB, _per_ctx if _per_ctx == _per_ctx else 0.0
                    )
                    _diagnosis = (
                        f"失败瞬间 used={_vram_at_failure[0]} free={_vram_at_failure[1]} "
                        f"total={_vram_at_failure[2]} MiB，已建成 {_n_built} 个"
                        + (f"（平均每 Context ≈ {_per_ctx:.0f} MiB）" if _per_ctx == _per_ctx else "")
                        + "。判定: "
                    )
                    if not _looks_like_oom:
                        _diagnosis += (
                            "**不像 OOM** —— 剩余显存仍足够再建一个 Context，"
                            "所以这是别的初始化失败；减 λ 数只会掩盖它，别按 OOM 修。"
                        )
                    else:
                        # 🔑 显存不够有**两种**完全不同的原因，别混成一句"减 λ 就行"：
                        #   (a) REMD 自己的 N 个 Context 就把卡吃满了 ⟹ 减 λ 是对的；
                        #   (b) **开跑前**卡上已经被占掉一大块 ⟹ 真问题是那个占用者
                        #       （同卡上的别的作业 / 上一次没杀干净的进程 / 本进程前面
                        #       某个 CUDA 阶段没释放）。这时减 λ 只是把症状盖住，
                        #       下次窗口一多又会撞上。
                        # 实测教训（2026-08-04）：`vram_before=[12197, 3646, 16303]`，
                        # 12 个 Context 只需 3804 MiB 却只剩 3646 MiB —— 差 158 MiB。
                        # 本条腿之前的 CUDA 阶段（预平衡 1 + attachment 4 + λ 预优化 1）
                        # 约 2 GB，剩下约 10 GB 来源不明。若当时只报"减 λ"，
                        # 那 10 GB 就永远不会被查。
                        _need_all = _per_ctx * self.n_replicas if _per_ctx == _per_ctx else float("nan")
                        _diagnosis += "**像 OOM**（剩余显存已不足一个 Context）。"
                        if _vram_baseline:
                            _baseline_used, _baseline_free, _baseline_total = _vram_baseline
                            _diagnosis += (
                                f" 但注意开跑**之前**就已 used={_baseline_used} "
                                f"free={_baseline_free} MiB"
                            )
                            if _need_all == _need_all:
                                _diagnosis += (
                                    f"；{self.n_replicas} 个 Context 共需 ≈ {_need_all:.0f} MiB"
                                )
                            if _baseline_used > 0.25 * _baseline_total:
                                _diagnosis += (
                                    f"。[WARN] 开跑前的占用已达全卡 "
                                    f"{100.0 * _baseline_used / _baseline_total:.0f}% —— "
                                    "**先查是谁占的**（`nvidia-smi` 看 PID：同卡上的别的作业？"
                                    "上一次没杀干净的进程？本进程前面某个 CUDA 阶段没释放？）。"
                                    "只减 λ 数能让这次跑起来，但会把这块占用永久掩盖，"
                                    "窗口一多又会撞上。"
                                )
                            else:
                                _diagnosis += (
                                    "。开跑前占用不大 ⟹ 确实是 REMD 自己的 Context 数"
                                    "超过了这张卡，减少 λ 状态数"
                                    "（= replica 数 = 常驻 Context 数）是对的修法。"
                                )
                        else:
                            _diagnosis += (
                                " 拿不到开跑前的显存基线，无法区分"
                                "'REMD 自己吃满'与'开跑前已被占'——先补这个数再决定要不要减 λ。"
                            )
                _message = (
                    "[WARN] REMD GPU Context 构建失败，已释放已创建的 replica contexts；"
                    f"回退 CPU 重建。\n"
                    f"     原始错误 [{type(exc).__name__}]: {exc}\n"
                    f"     {_diagnosis}\n"
                    "     CPU 回退会让本阶段慢约两个数量级（实测 45354 原子 × 12 副本："
                    "第 0 轮交换用了 29 分钟，500 轮约 10 天），表现得像卡死。"
                )
                logger.warning(_message)

                # 🔑 当场落盘，不等阶段跑完。
                #
                # 这条告警此前只到终端：`logger` 没有 FileHandler，而 `pipeline.log`
                # 是 `ABFEPipeline._log` 自己写的另一条通路，所以归档日志里**一行都没有**
                # （2026-08-03/04 因此绕了很多轮：只能靠 `nvidia-smi` 看到卡是空的，
                # 才推断出它退了 CPU）。而 `platform_fallback_reason` 只在阶段**跑完**时
                # 进 `*_exchange_diagnostics.json` —— 一旦像这次那样在 CPU 上磨到被杀，
                # 那份文件永远不会出现，证据就彻底没了。
                try:
                    os.makedirs(self.output_dir, exist_ok=True)
                    _fallback_path = os.path.join(
                        self.output_dir, "remd_platform_fallback.json"
                    )
                    with open(_fallback_path, "w", encoding="utf-8") as _handle:
                        json.dump(
                            {
                                "requested_platform": str(resolved_platform),
                                "platform_properties": dict(props),
                                "n_replicas": int(self.n_replicas),
                                "max_resident_contexts": int(self.max_resident_contexts),
                                "n_contexts_built_before_failure": int(_n_built),
                                "exception_type": type(exc).__name__,
                                "exception": str(exc),
                                "vram_before_mib": (
                                    list(_vram_baseline) if _vram_baseline else None
                                ),
                                "vram_at_failure_mib": (
                                    list(_vram_at_failure) if _vram_at_failure else None
                                ),
                                "diagnosis": _diagnosis,
                            },
                            _handle,
                            indent=2,
                            ensure_ascii=False,
                        )
                    print(f"  平台回退证据已落盘: {_fallback_path}")
                except Exception as _write_exc:  # noqa: BLE001
                    logger.warning("平台回退证据落盘失败: %s", _write_exc)

                print(f"  {_message}")
                self.platform_name = "CPU"
                self._build_replicas(system_template, allow_platform_fallback=False)
                return
            raise

    def _set_context_state(self, context_idx: int, state_idx: int) -> None:
        ctx = self.contexts[context_idx]
        self._try_set_context_parameter(ctx, "lambda_coul", self.lambdas_coul[state_idx])
        self._try_set_context_parameter(ctx, "lambda_vdw", self.lambdas_vdw[state_idx])

    @staticmethod
    def _crossed_save_boundary(prev_step: int, next_step: int, save_interval: int) -> bool:
        if save_interval <= 0:
            return False
        return (prev_step // save_interval) != (next_step // save_interval)

    def _atom_label(self, idx: int) -> str:
        idx = int(idx)
        role = "ligand" if idx in self._ligand_set else "env"
        if 0 <= idx < len(self._atoms):
            atom = self._atoms[idx]
            residue = atom.residue
            resname = residue.name.upper()
            if role != "ligand":
                if resname in {"HOH", "WAT", "SOL", "TIP3", "TIP3P"}:
                    role = "water"
                elif resname in {"NA", "SOD", "CL", "CLA", "K", "POT", "MG", "CA"}:
                    role = "ion"
            return f"{idx}:{atom.name}/{residue.name}{residue.index}:{role}"
        return f"{idx}:{role}"

    def _context_lambda_label(self, context_idx: int) -> Tuple[int, float, float]:
        state_idx = int(self._context_to_state[int(context_idx)])
        return state_idx, float(self.lambdas_coul[state_idx]), float(self.lambdas_vdw[state_idx])

    def _diagnose_context_failure(
        self,
        context_idx: int,
        phase: str,
        exc: Optional[Exception] = None,
    ) -> Dict[str, Any]:
        context_idx = int(context_idx)
        ctx = self.contexts[context_idx]
        system = self.replica_systems[context_idx] if context_idx < len(self.replica_systems) else None
        state_idx, lam_coul, lam_vdw = self._context_lambda_label(context_idx)
        diag: Dict[str, Any] = {
            "phase": str(phase),
            "context_idx": context_idx,
            "state_idx": state_idx,
            "lambda_coul": lam_coul,
            "lambda_vdw": lam_vdw,
            "exception": str(exc) if exc is not None else None,
        }

        print("\n[ERR] [REMD-NaN诊断] 捕获到不稳定 context")
        print(
            f"  phase={phase} | context={context_idx} | state={state_idx} | "
            f"lambda_coul={lam_coul:.6f} | lambda_vdw={lam_vdw:.6f}"
        )
        if exc is not None:
            print(f"  OpenMM 异常: {exc}")

        state = None
        try:
            state = ctx.getState(
                getPositions=True,
                getEnergy=True,
                getForces=True,
                enforcePeriodicBox=False,
            )
        except Exception as state_exc:
            diag["state_read_error"] = str(state_exc)
            print(f"  [WARN] 无法同时读取能量/坐标/受力: {state_exc}")
            try:
                state = ctx.getState(getPositions=True, enforcePeriodicBox=False)
            except Exception as pos_exc:
                diag["position_read_error"] = str(pos_exc)
                print(f"  [WARN] 坐标也无法读取: {pos_exc}")

        if state is not None:
            try:
                positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                bad_pos = np.where(~np.isfinite(positions_nm).all(axis=1))[0]
                diag["n_nonfinite_position_atoms"] = int(len(bad_pos))
                diag["nonfinite_position_atoms"] = [self._atom_label(i) for i in bad_pos[:20]]
                if len(bad_pos) > 0:
                    print(f"  [ERR] 非有限坐标原子数: {len(bad_pos)}")
                    print("     " + ", ".join(diag["nonfinite_position_atoms"]))
            except Exception as pos_exc:
                diag["position_parse_error"] = str(pos_exc)

            try:
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                diag["potential_energy_kj_mol"] = float(energy)
                print(f"  E_potential={energy:.3f} kJ/mol")
            except Exception as energy_exc:
                diag["energy_error"] = str(energy_exc)

            try:
                forces = state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                )
                norms = np.linalg.norm(forces, axis=1)
                bad_force = np.where(~np.isfinite(forces).all(axis=1))[0]
                finite_norms = np.where(np.isfinite(norms), norms, -np.inf)
                top = np.argsort(finite_norms)[::-1][:10]
                diag["n_nonfinite_force_atoms"] = int(len(bad_force))
                diag["top_force_atoms"] = [
                    {"atom": self._atom_label(i), "force_kj_mol_nm": float(norms[int(i)])}
                    for i in top
                    if np.isfinite(norms[int(i)])
                ]
                if np.any(np.isfinite(norms)):
                    diag["max_force_kj_mol_nm"] = float(np.nanmax(norms))
                    print(f"  max|F|={diag['max_force_kj_mol_nm']:.1f} kJ/mol/nm")
                if len(bad_force) > 0:
                    diag["nonfinite_force_atoms"] = [self._atom_label(i) for i in bad_force[:20]]
                    print(f"  [ERR] 非有限受力原子数: {len(bad_force)}")
                    print("     " + ", ".join(diag["nonfinite_force_atoms"]))
                if diag["top_force_atoms"]:
                    print("  Top受力原子:")
                    for row in diag["top_force_atoms"][:5]:
                        print(f"    {row['atom']:<28} {row['force_kj_mol_nm']:12.1f}")
            except Exception as force_exc:
                diag["force_error"] = str(force_exc)

        if _has_valid_boresch_restraint(self.boresch_params):
            try:
                ok, r0_chk, thA_chk, thB_chk = _check_boresch_geometry_safe(ctx, self.boresch_params)
                diag["boresch_geometry"] = {
                    "safe": bool(ok),
                    "r_nm": float(r0_chk),
                    "thetaA_deg": float(thA_chk),
                    "thetaB_deg": float(thB_chk),
                }
                print(
                    f"  Boresch当前几何: r={r0_chk:.3f} nm, "
                    f"thetaA={thA_chk:.1f}°, thetaB={thB_chk:.1f}°, safe={ok}"
                )
            except Exception as boresch_exc:
                diag["boresch_geometry_error"] = str(boresch_exc)
                print(f"  [WARN] Boresch 几何诊断失败: {boresch_exc}")

        group_stats = []
        if system is not None:
            for gid in sorted({force.getForceGroup() for force in system.getForces()}):
                row: Dict[str, Any] = {"group": int(gid)}
                try:
                    g_state = ctx.getState(getEnergy=True, getForces=True, groups={gid})
                    g_energy = g_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    g_forces = g_state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer
                    )
                    g_norms = np.linalg.norm(g_forces, axis=1)
                    row["energy_kj_mol"] = float(g_energy)
                    row["max_force_kj_mol_nm"] = float(np.nanmax(g_norms))
                except Exception as group_exc:
                    row["error"] = str(group_exc)
                group_stats.append(row)
            diag["force_groups"] = group_stats
            print("  ForceGroup 分解:")
            for row in group_stats:
                if "error" in row:
                    print(f"    G{row['group']}: ERROR {row['error']}")
                else:
                    print(
                        f"    G{row['group']}: E={row['energy_kj_mol']:.3f} kJ/mol | "
                        f"max|F|={row['max_force_kj_mol_nm']:.1f}"
                    )

        try:
            out_path = os.path.join(self.output_dir, "remd_last_failure.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2, ensure_ascii=False)
            print(f"  REMD 失败诊断已写入: {out_path}")
        except Exception as write_exc:
            print(f"  [WARN] REMD 诊断写入失败: {write_exc}")

        return diag

    def _preflight_context(self, context_idx: int) -> None:
        ctx = self.contexts[int(context_idx)]
        state_idx, lam_coul, lam_vdw = self._context_lambda_label(context_idx)
        max_force_limit = float(os.environ.get("IBS_REMD_PREFLIGHT_MAX_FORCE", "100000"))
        try:
            state = ctx.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            norms = np.linalg.norm(forces, axis=1)
            max_force = float(np.nanmax(norms))
            if (not np.isfinite(energy)) or (not np.isfinite(max_force)):
                self._diagnose_context_failure(context_idx, f"preflight_nonfinite_state{state_idx}")
                raise RuntimeError(
                    f"REMD preflight 非有限能量/受力: state={state_idx}, "
                    f"lambda_coul={lam_coul}, lambda_vdw={lam_vdw}"
                )
            if max_force > max_force_limit:
                print(
                    f"  [WARN] [REMD] context {context_idx} state {state_idx} "
                    f"预检查 max|F|={max_force:.1f} > {max_force_limit:.1f}，执行短最小化"
                )
                openmm.LocalEnergyMinimizer.minimize(ctx, tolerance=20.0, maxIterations=1000)
                velocity_seed = self._seed_for(
                    "recovery", self.seed_stage, state_idx, "velocity", attempt=1
                )
                if velocity_seed is None:
                    ctx.setVelocitiesToTemperature(self.temperature)
                else:
                    ctx.setVelocitiesToTemperature(self.temperature, velocity_seed)
        except Exception as exc:
            self._diagnose_context_failure(context_idx, f"preflight_state{state_idx}", exc)
            raise

    def _relax_context_before_preheat(self, context_idx: int) -> None:
        """Minimize and gently settle each REMD replica before the first long MD step."""
        ctx = self.contexts[int(context_idx)]
        state_idx, lam_coul, lam_vdw = self._context_lambda_label(context_idx)
        max_force_limit = float(os.environ.get("IBS_REMD_PREFLIGHT_MAX_FORCE", "100000"))
        em_iters = int(os.environ.get("IBS_REMD_PREHEAT_EM_ITERS", "5000"))
        em_tol = float(os.environ.get("IBS_REMD_PREHEAT_EM_TOL", "10.0"))
        try:
            openmm.LocalEnergyMinimizer.minimize(
                ctx,
                tolerance=em_tol * unit.kilojoule_per_mole / unit.nanometer,
                maxIterations=em_iters,
            )
            velocity_seed = self._seed_for(
                "recovery", self.seed_stage, state_idx, "velocity", attempt=2
            )
            if velocity_seed is None:
                ctx.setVelocitiesToTemperature(self.temperature)
            else:
                ctx.setVelocitiesToTemperature(self.temperature, velocity_seed)

            state = ctx.getState(getEnergy=True, getForces=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            max_force = float(np.nanmax(np.linalg.norm(forces, axis=1)))
            if (not np.isfinite(energy)) or (not np.isfinite(max_force)) or max_force > max_force_limit:
                self._diagnose_context_failure(context_idx, f"post_em_preheat_state{state_idx}")
                raise RuntimeError(
                    f"REMD 预热前 EM 后仍不稳定: state={state_idx}, "
                    f"lambda_coul={lam_coul:.6f}, lambda_vdw={lam_vdw:.6f}, "
                    f"E={energy}, max|F|={max_force:.1f}"
                )
        except Exception as exc:
            self._diagnose_context_failure(context_idx, f"preheat_em_state{state_idx}", exc)
            raise

    def _preheat_context_gently(self, context_idx: int, total_steps: int, phase: str) -> None:
        """Run preheat with a short timestep ramp instead of one long 2 fs block."""
        ctx = self.contexts[int(context_idx)]
        integ = ctx.getIntegrator()
        original_dt = integ.getStepSize()
        total_steps = max(0, int(total_steps))
        if total_steps <= 0:
            return

        schedule = [
            (0.0001, min(500, total_steps), "0.1fs"),
            (0.00025, min(1000, total_steps), "0.25fs"),
            (0.0005, min(1500, total_steps), "0.5fs"),
        ]
        used = 0
        try:
            for dt_ps, n_step, label in schedule:
                if used >= total_steps:
                    break
                n_step = min(int(n_step), total_steps - used)
                if n_step <= 0:
                    continue
                integ.setStepSize(dt_ps * unit.picoseconds)
                self._step_context_with_diagnostics(
                    context_idx,
                    n_step,
                    f"{phase}:ramp_{label}",
                )
                used += n_step

            remaining = total_steps - used
            if remaining > 0:
                integ.setStepSize(original_dt)
                self._step_context_with_diagnostics(
                    context_idx,
                    remaining,
                    f"{phase}:ramp_full_dt",
                )
        finally:
            integ.setStepSize(original_dt)

    def _step_context_with_diagnostics(self, context_idx: int, n_steps: int, phase: str) -> None:
        ctx = self.contexts[int(context_idx)]
        n_steps = int(n_steps)
        if n_steps <= 0:
            return
        chunk = int(os.environ.get("IBS_REMD_STEP_CHUNK", str(n_steps)))
        chunk = max(1, min(chunk, n_steps))
        steps_done = 0
        try:
            while steps_done < n_steps:
                this_chunk = min(chunk, n_steps - steps_done)
                ctx.getIntegrator().step(this_chunk)
                steps_done += this_chunk
        except Exception as exc:
            self._diagnose_context_failure(
                context_idx,
                f"{phase}; local_step={steps_done}/{n_steps}",
                exc,
            )
            state_idx, lam_coul, lam_vdw = self._context_lambda_label(context_idx)
            raise RuntimeError(
                f"REMD context {context_idx} 在 {phase} 爆炸: "
                f"state={state_idx}, lambda_coul={lam_coul:.6f}, lambda_vdw={lam_vdw:.6f}"
            ) from exc

    def begin_production_stage(self) -> None:
        """同一实例换 `stage_name` 再次 `run()` 之前必须调用。

        `run()` 用 `_steps_completed > 0` 决定 DCDReporter 的 append 模式，而它
        表达的是"本实例已经跑过多少步"，不是"目标 DCD 里已经有多少步"。平衡段跑
        完后它是正数，生产段换了 `stage_name`、对应 DCD 还不存在 —— append 打开
        不存在的文件会崩（`'DCDReporter' object has no attribute '_out'`，且真实
        异常被 `__del__` 的二次异常盖住）。归零让新 stage 的 DCD 以写模式建立。
        `_is_warmed_up` 不动，预热不会重做。

        [0831issue] 原来只有 `BoreschAttachmentREMDManager` 有这个方法；基类缺失
        意味着任何走同一模式的子类/调用方都会踩同一个坑。上提到基类，子类只在
        此基础上追加它自己的清理（`state_history`）。
        """
        self._steps_completed = 0

    def run(
        self,
        n_steps: int = 500000,
        exchange_interval: int = 1000,
        save_interval: int = 5000,
        stage_name: str = "complex",
    ):
        n_steps = int(n_steps)
        exchange_interval = int(exchange_interval)
        save_interval = int(save_interval)
        if n_steps < 0:
            raise ValueError("REMD n_steps 不能为负")
        if exchange_interval <= 0:
            raise ValueError("REMD exchange_interval 必须为正")
        if save_interval <= 0:
            raise ValueError("REMD save_interval 必须为正")
        n_exchanges = n_steps // exchange_interval
        remaining_steps = n_steps - n_exchanges * exchange_interval
        traj_files = [os.path.join(self.output_dir, f"{stage_name}_rep{i}.dcd") for i in range(self.n_replicas)]
        # `_steps_completed` 只在 __init__ 归零、只在本方法内累加，**从不从盘上
        # 恢复**——所以它表达的严格是"同一个实例本次生命周期内已经跑过多少步"，
        # 不是"这个 output_dir 里已经有多少步的轨迹"。两个推论，都在下面显式处理：
        #
        #  (a) 新进程 + 盘上已有同名 DCD → append=False，旧文件被截断重写。这是
        #      **正确**的：调用方（abfe_pipeline 三处 / run_shadow_bridge_leg）只有
        #      在判定缓存 DCD 不完整或 sampling fingerprint 不匹配时才会走到这里，
        #      并且随后会从第 0 步跑满 n_steps，产出一份完整的新轨迹。旧帧属于被
        #      主动作废的采样分布，保留反而会把两个 Hamiltonian 的帧混进一个文件。
        #      唯一的问题是它此前完全静默 —— 下面补一条日志。
        #  (b) 同一个实例换 stage_name 二次 run()（平衡段→生产段）→ append=True
        #      但新 stage 的 DCD 还不存在，DCDReporter 会以
        #      `'DCDReporter' object has no attribute '_out'` 崩掉，且真实异常被
        #      `__del__` 的二次异常盖住。子类 BoreschAttachmentREMDManager 靠
        #      begin_production_stage() 归零来规避；该方法现已上提到本基类，同时
        #      在这里对"要求 append 却没有文件"直接 fail closed，给出可读报错。
        append_mode = self._steps_completed > 0
        for f in traj_files:
            exists = os.path.exists(f)
            if append_mode and not exists:
                raise RuntimeError(
                    f"REMD 轨迹 append 模式要求 {f} 已存在，但该文件不存在。"
                    f"_steps_completed={self._steps_completed} 说明本实例已经跑过步数，"
                    f"而 stage_name={stage_name!r} 对应的 DCD 还没建立——同一实例换 "
                    "stage 前必须先调用 begin_production_stage()（把 _steps_completed "
                    "归零，让新 stage 的 DCD 以写模式建立）。"
                )
            if not append_mode and exists:
                print(
                    f"  [WARN] [REMD] {os.path.basename(f)} 已存在且本次以写模式重建"
                    "（旧帧属于被作废的采样分布，本次将从第 0 步跑满 n_steps）。"
                )
        reporters = [
            app.DCDReporter(f, save_interval, append=append_mode, enforcePeriodicBox=False)
            for f in traj_files
        ]
        # 🔑 [2026-09-09] 整段包进 try/finally。原来 `reporters.clear()` 只在
        # 成功路径执行，而交换循环的失败按设计一律重抛
        # （`_step_context_with_diagnostics`）—— 于是每次中止都把 N 个 DCDReporter
        # 连同它们持有的打开文件留给 GC 去收，落盘时机不确定，留下的是半截 DCD；
        # 而 resume 侧只看文件在不在，`append_mode` 又完全由进程内的
        # `_steps_completed` 决定，那些截断文件于是被再截断一次而不是修复。
        # 放进 finally 至少让收尾变成确定性的、且异常路径与成功路径一致。
        try:
            print(f"\n启动传统 REMD (单卡懒加载极速版) | {self.n_replicas} 副本 | 交换间隔={exchange_interval}")
        
            # 预热
            if not self._is_warmed_up:
                print("  [REMD] 预热前执行能量/受力 preflight...")
                for ctx_idx in range(self.n_replicas):
                    self._preflight_context(ctx_idx)
                    self._relax_context_before_preheat(ctx_idx)
                for ctx_idx in range(self.n_replicas):
                    self._preheat_context_gently(
                        ctx_idx,
                        5000,
                        f"{stage_name}:preheat",
                    )
                self._is_warmed_up = True
            
            exchange_log = []
            # 🔑 [2026-09-03] 逐**边**接受率。`exchange_log`/`mean_acceptance` 是逐
            # **轮**的（一轮里所有相邻对一起平均），`min_acceptance` 是最差那一轮而
            # 不是最差那条边——于是"哪条 λ 边太稀"这个信息一直拿不到，而定态数的
            # 约束恰恰是最差那条边（δ_max），不是平均。
            # 只统计真正执行过 Metropolis 判定的尝试：下面能量非有限的分支 `continue`
            # 掉，那不是一次"被拒绝"，不能计入分母。
            edge_accepts = [0] * max(0, self.n_replicas - 1)
            edge_attempts = [0] * max(0, self.n_replicas - 1)

            for step in range(n_exchanges):
                # 1. 批量提交步进任务 (GPU 会在底层自动流水线并发，无需 Python 干预)
                prev_steps = self._steps_completed
                for ctx_idx in range(self.n_replicas):
                    state_idx, _, _ = self._context_lambda_label(ctx_idx)
                    self._step_context_with_diagnostics(
                        ctx_idx,
                        exchange_interval,
                        f"{stage_name}:exchange_round={step}:state={state_idx}",
                    )
                self._steps_completed += exchange_interval
                
                # 2. 轨迹落盘
                if self._crossed_save_boundary(prev_steps, self._steps_completed, save_interval):
                    for state_idx, ctx_idx in enumerate(self._state_to_context):
                        ctx = self.contexts[ctx_idx]
                        state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
                        reporters[state_idx].report(
                            self._ReporterSimulationView(
                                self.topology, self._steps_completed, ctx.getIntegrator().getStepSize()
                            ), state,
                        )
                
                accepted = 0
            
                # 3. 交换状态映射而非整份坐标/速度，避免接受交换时的大对象搬运
                for state_i in range(self.n_replicas - 1):
                    state_j = state_i + 1
                    ctx_idx_i = self._state_to_context[state_i]
                    ctx_idx_j = self._state_to_context[state_j]
                    ctx_i = self.contexts[ctx_idx_i]
                    ctx_j = self.contexts[ctx_idx_j]
                
                    # --- 阶段 A: 仅获取能量 (极快，PCIe 传输量仅几字节，不阻塞 GPU) ---
                    U_i_i = ctx_i.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    U_j_j = ctx_j.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                
                    # --- 阶段 B: 计算交叉能量 (通过修改参数，GPU 原地重算) ---
                    self._set_context_state(ctx_idx_i, state_j)
                    U_i_j = ctx_i.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                
                    self._set_context_state(ctx_idx_j, state_i)
                    U_j_i = ctx_j.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                
                    # 恢复原始参数
                    self._set_context_state(ctx_idx_i, state_i)
                    self._set_context_state(ctx_idx_j, state_j)

                    if not np.all(np.isfinite([U_i_i, U_j_j, U_i_j, U_j_i])):
                        print(
                            f"  [WARN] [REMD] 交换能量非有限: pair=({state_i},{state_j}) "
                            f"Uii={U_i_i}, Ujj={U_j_j}, Uij={U_i_j}, Uji={U_j_i}"
                        )
                        self._diagnose_context_failure(
                            ctx_idx_i,
                            f"{stage_name}:exchange_energy_state{state_i}_{state_j}",
                        )
                        self._diagnose_context_failure(
                            ctx_idx_j,
                            f"{stage_name}:exchange_energy_state{state_i}_{state_j}",
                        )
                        # 刻意不 edge_attempts[state_i] += 1：这一对根本没做判定，
                        # 计入分母会把"探针坏了"混成"这条边重叠差"。
                        continue
                
                    # --- 阶段 C: Metropolis 判定 ---
                    delta = self.beta * (U_i_j + U_j_i - U_i_i - U_j_j)
                    if not np.isfinite(delta):
                        accept = bool(np.isneginf(delta))
                    else:
                        accept = delta < 0 or self.rng.random() < np.exp(-delta)
                
                    # --- 阶段 D: 仅交换状态分配，Context 本身保持连续推进 ---
                    if accept:
                        self._state_to_context[state_i], self._state_to_context[state_j] = (
                            ctx_idx_j,
                            ctx_idx_i,
                        )
                        self._context_to_state[ctx_idx_i], self._context_to_state[ctx_idx_j] = (
                            state_j,
                            state_i,
                        )
                        self._set_context_state(ctx_idx_i, state_j)
                        self._set_context_state(ctx_idx_j, state_i)
                        accepted += 1
                    edge_attempts[state_i] += 1
                    if accept:
                        edge_accepts[state_i] += 1
                    
                # A one-replica REMD run is valid (it is simply a single-state
                # simulation), but has no exchange pairs. Avoid 0/0 and record a
                # well-defined zero acceptance rate.
                exchange_log.append(accepted / max(1, self.n_replicas - 1))
                if step % 50 == 0:
                    print(f"  [REMD] 交换轮次 {step}/{n_exchanges} | 接受率: {exchange_log[-1]:.2f}")

            if remaining_steps > 0:
                prev_steps = self._steps_completed
                for ctx_idx in range(self.n_replicas):
                    state_idx, _, _ = self._context_lambda_label(ctx_idx)
                    self._step_context_with_diagnostics(
                        ctx_idx,
                        remaining_steps,
                        f"{stage_name}:remaining:state={state_idx}",
                    )
                self._steps_completed += remaining_steps
                if self._crossed_save_boundary(prev_steps, self._steps_completed, save_interval):
                    for state_idx, ctx_idx in enumerate(self._state_to_context):
                        ctx = self.contexts[ctx_idx]
                        state = ctx.getState(getPositions=True, enforcePeriodicBox=True)
                        reporters[state_idx].report(
                            self._ReporterSimulationView(
                                self.topology, self._steps_completed, ctx.getIntegrator().getStepSize()
                            ),
                            state,
                        )
                
        finally:
            # OpenMM 的 DCDReporter 没有公共 close()；对象释放时会完成底层文件收尾。
            reporters.clear()
            gc.collect()
        exchange_summary = {
            "stage_name": stage_name,
            "n_replicas": int(self.n_replicas),
            "platform_name": str(self.platform_name),
            "platform_fallback_reason": self.platform_fallback_reason,
            "n_exchange_attempts": int(n_exchanges),
            # ⚠️ 下面四个是逐**轮**口径（一轮里所有相邻对一起平均）。
            # `min_acceptance` 是最差那一**轮**，不是最差那条**边**——别拿它当
            # "最稀的那条 λ 边"用。逐边口径见 edge_* 字段。
            "acceptance_by_round": [float(v) for v in exchange_log],
            "mean_acceptance": float(np.mean(exchange_log)) if exchange_log else 0.0,
            "min_acceptance": float(np.min(exchange_log)) if exchange_log else 0.0,
            "max_acceptance": float(np.max(exchange_log)) if exchange_log else 0.0,
            # 🔑 [2026-09-03] 逐**边**口径。`edge_acceptance[i]` 是状态 i↔i+1 这条
            # 边在整段运行里的接受率，分母只算真正做过 Metropolis 判定的尝试
            # （能量非有限而跳过的对不计入，见交换循环里的说明）。
            #
            # 用途：相邻态接受率与该边的热力学长度是一一对应的
            # （高斯近似下 ⟨P_acc⟩ = erfc(δ/2)，故 δ = 2·erfc⁻¹(P_acc)），所以逐边
            # 接受率就是逐边 δ 的直接测量——这是判断 λ 态数够不够的判据，而定态数
            # 的约束是**最差那条边**。刻意只落原始测量值、不在这里落反解出来的 δ：
            # 那个反解依赖高斯近似，属于解释而不是测量，放在离线审计工具里做并在
            # 那里说明假设。
            #
            # 长度恒为 n_replicas-1；`edge_attempts[i] == 0` 表示那条边一次判定都
            # 没成功执行过（探针失败），此时 `edge_acceptance[i]` 记为 None 而不是
            # 0.0——0.0 会被误读成"完全不接受"。
            "edge_attempts": [int(v) for v in edge_attempts],
            "edge_accepts": [int(v) for v in edge_accepts],
            "edge_acceptance": [
                (float(a) / float(n)) if n > 0 else None
                for a, n in zip(edge_accepts, edge_attempts)
            ],
        }
        _edge_ok = [
            float(a) / float(n)
            for a, n in zip(edge_accepts, edge_attempts)
            if n > 0
        ]
        exchange_summary["min_edge_acceptance"] = min(_edge_ok) if _edge_ok else None
        exchange_summary["max_edge_acceptance"] = max(_edge_ok) if _edge_ok else None
        exchange_summary["edges_without_attempts"] = [
            int(i) for i, n in enumerate(edge_attempts) if n == 0
        ]
        with open(os.path.join(self.output_dir, f"{stage_name}_exchange_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(exchange_summary, f, indent=2)
        print(f"[OK] REMD 完成 | 平均交换接受率: {exchange_summary['mean_acceptance']:.3f}")
        if _edge_ok:
            _worst = int(min(range(len(_edge_ok)), key=lambda k: _edge_ok[k]))
            print(
                "   逐边接受率: "
                + " ".join(
                    ("--" if p is None else f"{p:.2f}")
                    for p in exchange_summary["edge_acceptance"]
                )
                + f"  | 最差边 {_worst}↔{_worst + 1}: {min(_edge_ok):.3f}"
                f" | 最好边: {max(_edge_ok):.3f}"
            )
        if exchange_summary["edges_without_attempts"]:
            print(
                "   [WARN] 以下 λ 边一次交换判定都没成功执行过（探针能量非有限）: "
                f"{exchange_summary['edges_without_attempts']}"
            )
        return traj_files


