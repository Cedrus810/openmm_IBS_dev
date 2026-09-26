# ============================================================================
# 6.5 Shadow-PME Bridge 专用 REMD：复用 REMDManager 通用交换机制，
#     只替换"每个 replica 的系统/参数怎么建"这一层。
# ============================================================================
class ShadowBridgeREMDManager(REMDManager):
    """
    REMDManager 的最小子类：交换循环 (Metropolis 判据、DCD 落盘、诊断) 完全继承
    自 REMDManager.run()，不做任何改动——这部分只通过 _set_context_state /
    _context_lambda_label 间接接触 lambda_coul/lambda_vdw，本身跟具体是哪个
    参数无关。这里只重写三个方法，把驱动变量换成 build_shadow_bridge_system()
    产出的单一全局参数 "lambda_bridge_s"：

      - _build_replicas: 不需要 REMDManager 原来那套按 leg 类型分支建系统的重
        逻辑（PME-coulomb-leg / mixed-alchemical / Beutler-softcore），因为
        Bridge 只有一个已经建好的系统，每个 replica 只是它的一份深拷贝 + 不同
        的 lambda_bridge_s 取值。
      - _set_context_state / _context_lambda_label: 把 lambda_coul/lambda_vdw
        换成 lambda_bridge_s。

    基类 __init__ 仍然会用占位的 lambdas_coul=[0]*n / lambdas_vdw=[1]*n 调用
    （只是为了让 has_coulomb_scaling / is_pme_coulomb_leg 等旧字段存在且不报错，
    这两个占位数组本身不会被下面任何重写方法用到）。
    """

    def __init__(
        self,
        bridge_system: openmm.System,
        topology: app.Topology,
        positions,
        box_vectors,
        ligand_indices: List[int],
        lambdas_bridge_s: List[float],
        s_param_name: str = "lambda_bridge_s",
        temperature: float = 300.0,
        platform_name: str = "CUDA",
        output_dir: str = "./shadow_bridge_remd",
        random_seed: Optional[int] = None,
        seed_ledger: Optional[Exp019SeedLedger] = None,
        seed_stage: Optional[str] = None,
        seed_leg: Optional[str] = None,
    ):
        self.lambdas_bridge_s = np.asarray(lambdas_bridge_s, dtype=float)
        self.s_param_name = s_param_name
        self._bridge_system_template = bridge_system
        n = len(self.lambdas_bridge_s)
        super().__init__(
            system_template=bridge_system,
            topology=topology,
            positions=positions,
            box_vectors=box_vectors,
            ligand_indices=ligand_indices,
            lambdas_coul=[0.0] * n,
            lambdas_vdw=[1.0] * n,
            temperature=temperature,
            platform_name=platform_name,
            output_dir=output_dir,
            boresch_params=None,
            random_seed=random_seed,
            seed_ledger=seed_ledger,
            seed_stage=seed_stage or "shadow_bridge",
            seed_leg=seed_leg,
        )

    def _build_replicas(self, system_template, allow_platform_fallback: bool = True):
        resolved_platform, props = _build_platform_properties(self.platform_name)
        platform = openmm.Platform.getPlatformByName(resolved_platform)
        try:
            for i in range(self.n_replicas):
                # [2026-09-10] 同上：往返 → clone()。这一处在 per-replica 循环体内，
                # 每个 replica 都要开一份完整 System 的字符串。
                replica_sys = openmm.XmlSerializer.clone(system_template)
                replica_sys.thisown = 1
                integ = openmm.LangevinMiddleIntegrator(self.temperature, 1.0 / unit.picosecond, 0.002 * unit.picosecond)
                integrator_seed = self._seed_for(
                    "bridge", self.seed_stage, i, "integrator"
                )
                if integrator_seed is not None:
                    integ.setRandomNumberSeed(integrator_seed)
                ctx = openmm.Context(replica_sys, integ, platform, props)
                ctx.setPositions(self.positions)
                if self.box_vectors is not None:
                    ctx.setPeriodicBoxVectors(*self.box_vectors)
                self._try_set_context_parameter(ctx, self.s_param_name, self.lambdas_bridge_s[i])
                velocity_seed = self._seed_for(
                    "bridge", self.seed_stage, i, "velocity"
                )
                if velocity_seed is None:
                    ctx.setVelocitiesToTemperature(self.temperature)
                else:
                    ctx.setVelocitiesToTemperature(self.temperature, velocity_seed)
                self.contexts.append(ctx)
                self.integrators.append(integ)
                self.replica_systems.append(replica_sys)
        except Exception as exc:
            self._clear_replica_contexts()
            if (
                allow_platform_fallback
                and str(resolved_platform).upper() in {"CUDA", "OPENCL"}
                and self._is_gpu_context_failure(exc)
            ):
                self.platform_fallback_reason = str(exc)
                print(
                    "  [WARN] Shadow-Bridge REMD GPU Context 构建失败，已释放已创建的 replica contexts；"
                    f"回退 CPU 重建。原始错误: {exc}"
                )
                self.platform_name = "CPU"
                self._build_replicas(system_template, allow_platform_fallback=False)
                return
            raise

    def _set_context_state(self, context_idx: int, state_idx: int) -> None:
        ctx = self.contexts[context_idx]
        self._try_set_context_parameter(ctx, self.s_param_name, self.lambdas_bridge_s[state_idx])

    def _context_lambda_label(self, context_idx: int) -> Tuple[int, float, float]:
        state_idx = int(self._context_to_state[int(context_idx)])
        s_val = float(self.lambdas_bridge_s[state_idx])
        return state_idx, s_val, s_val


def compute_shadow_bridge_u_kn(
    bridge_system: openmm.System,
    topology: app.Topology,
    traj_files: List[str],
    lambdas_bridge_s: List[float],
    s_param_name: str = "lambda_bridge_s",
    box_vectors=None,
    platform_name: str = "CPU",
    energy_force_group: Optional[int] = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    离线重算 Shadow-Bridge 的 u_kn (未除 kT 的势能矩阵，单位 kJ/mol；调用方自己
    乘 beta)：对每条轨迹里的每一帧，在全部 len(lambdas_bridge_s) 个 s 状态下
    重新评估势能。

    只取 energy_force_group=1 (bridge_force 所在的 force group) 而不是总能量：
    U_common (bonded/环境 PME/配体内部) 对同一帧在所有状态下都完全相同，是纯粹
    的按帧可加常数，在 MBAR 的自洽方程里精确抵消，不影响任何 f_k 结果——只算
    force group 1 更省算力，效果完全等价。跳过整数分组不存在的兼容性问题时可传
    energy_force_group=None 改回算总能量。
    """
    import mdtraj as md

    n_states = len(lambdas_bridge_s)
    if len(traj_files) != n_states:
        raise ValueError(f"traj_files 数量 ({len(traj_files)}) 与 lambdas_bridge_s 数量 ({n_states}) 不一致")

    # [2026-09-10] 同上：往返 → clone()。
    eval_sys = openmm.XmlSerializer.clone(bridge_system)
    eval_sys.thisown = 1
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    resolved_platform, props = _build_platform_properties(platform_name)
    platform = openmm.Platform.getPlatformByName(resolved_platform)
    ctx = openmm.Context(eval_sys, integrator, platform, props)
    if box_vectors is not None:
        ctx.setPeriodicBoxVectors(*box_vectors)

    n_k = []
    all_xyz = []
    all_box = []
    # 🔑 [2026-09-09] 循环里原来是 `top=topology if not isinstance(topology, str)
    # else topology` —— 两个分支完全一样的死条件，实际总是把 OpenMM 的
    # `app.Topology` 原样递给 mdtraj，而 mdtraj 的 `_parse_topology` 对它只会落到
    # catch-all 抛 `TypeError: A topology is required`。两个同胞函数
    # （`compute_u_kn`、attachment 腿那份）都正确地先 `from_openmm`。后果是
    # shadow-bridge 腿在付完整个 REMD GPU 代价**之后**才死在分析步。
    # 顺带把转换提到循环外：每个 traj 重做一次纯属白费。
    md_top = (
        topology if isinstance(topology, md.Topology) else md.Topology.from_openmm(topology)
    )
    for traj_path in traj_files:
        traj = md.load_dcd(traj_path, top=md_top)
        n_k.append(traj.n_frames)
        all_xyz.append(traj.xyz)
        if traj.unitcell_vectors is not None:
            all_box.append(traj.unitcell_vectors)
        else:
            all_box.append([None] * traj.n_frames)
    n_k = np.asarray(n_k, dtype=int)
    n_total = int(np.sum(n_k))
    u_kn = np.zeros((n_states, n_total), dtype=np.float64)

    groups = {energy_force_group} if energy_force_group is not None else None
    try:
        frame_cursor = 0
        for k in range(n_states):
            xyz_k = all_xyz[k]
            box_k = all_box[k]
            for local_f in range(xyz_k.shape[0]):
                ctx.setPositions(xyz_k[local_f] * unit.nanometer)
                if box_k[local_f] is not None:
                    ctx.setPeriodicBoxVectors(*(row * unit.nanometer for row in box_k[local_f]))
                n = frame_cursor + local_f
                for j in range(n_states):
                    ctx.setParameter(s_param_name, float(lambdas_bridge_s[j]))
                    e = ctx.getState(getEnergy=True, groups=groups).getPotentialEnergy()
                    u_kn[j, n] = e.value_in_unit(unit.kilojoule_per_mole)
            frame_cursor += xyz_k.shape[0]
    finally:
        del ctx, integrator, eval_sys

    if not np.all(np.isfinite(u_kn)):
        raise RuntimeError("Shadow-Bridge u_kn 含 NaN/Inf，拒绝继续求解 MBAR。")
    return u_kn, n_k


def run_shadow_bridge_leg(
    system: openmm.System,
    topology: app.Topology,
    positions,
    box_vectors,
    perturbed_indices: List[int],
    lambdas_bridge_s: List[float],
    temperature_k: float = 300.0,
    platform_name: str = "CUDA",
    output_dir: str = "./shadow_bridge",
    n_steps_per_state: int = 200000,
    exchange_interval: int = 1000,
    save_interval: int = 5000,
    restraint_params: Optional[Dict] = None,
    random_seed: Optional[int] = None,
    seed_ledger: Optional[Exp019SeedLedger] = None,
    seed_stage: Optional[str] = None,
    seed_leg: Optional[str] = None,
) -> Dict:
    """
    Shadow-PME Bridge 的完整入口：build_shadow_bridge_system -> ShadowBridgeREMDManager
    跑 REMD -> 离线重算 u_kn -> TraditionalMBARAnalyzer 求解 ΔG_bridge。

    lambdas_bridge_s 建议只放 2~4 个点（s=0 是 PME full charge 端点，s=1 是
    Shadow full charge 端点），因为这一段要跨越的只是 reciprocal+self+PME 修正
    的残差（对本项目 Atenolol 实测量级 ~11 kJ/mol），不需要像正式去电荷那样几
    十个窗口。
    """
    bridge_sys, s_name, build_diag = build_shadow_bridge_system(
        system,
        topology,
        perturbed_indices,
        restraint_params=restraint_params,
        box_vectors=box_vectors,
    )
    lambdas_bridge_s = sorted(float(s) for s in lambdas_bridge_s)
    if len(lambdas_bridge_s) < 2:
        raise ValueError("Shadow-Bridge 至少需要 2 个 s 状态")
    if not np.all(np.isfinite(lambdas_bridge_s)):
        raise ValueError("Shadow-Bridge s 必须全部为有限数")
    if any(s < 0.0 or s > 1.0 for s in lambdas_bridge_s):
        raise ValueError("Shadow-Bridge s 必须位于 [0, 1]")
    if lambdas_bridge_s[0] != 0.0 or lambdas_bridge_s[-1] != 1.0:
        raise ValueError(
            "lambdas_bridge_s 必须以 0.0 (PME full charge) 开始、以 1.0 "
            "(Shadow full charge) 结束。"
        )

    remd = ShadowBridgeREMDManager(
        bridge_system=bridge_sys,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=perturbed_indices,
        lambdas_bridge_s=lambdas_bridge_s,
        s_param_name=s_name,
        temperature=temperature_k,
        platform_name=platform_name,
        output_dir=output_dir,
        random_seed=random_seed,
        seed_ledger=seed_ledger,
        seed_stage=seed_stage or "shadow_bridge",
        seed_leg=seed_leg,
    )
    traj_files = remd.run(
        n_steps=n_steps_per_state,
        exchange_interval=exchange_interval,
        save_interval=save_interval,
        stage_name="shadow_bridge",
    )

    u_kn, n_k = compute_shadow_bridge_u_kn(
        bridge_sys,
        topology,
        traj_files,
        lambdas_bridge_s,
        s_param_name=s_name,
        box_vectors=box_vectors,
        platform_name="CPU",
    )
    analyzer = TraditionalMBARAnalyzer(temperature=temperature_k)
    analyzer._last_n_k = n_k
    res = analyzer.solve(analyzer.beta * u_kn)
    return {
        "stage": "shadow_bridge",
        "total_delta_G": float(res.get("delta_G", 0.0)),
        "total_error": float(res.get("error", 0.0)),
        "method": "Shadow-PME-Bridge-REMD-MBAR",
        "n_states": len(lambdas_bridge_s),
        "lambdas_bridge_s": lambdas_bridge_s,
        "build_diagnostics": build_diag,
        "converged": res.get("converged"),
        "min_overlap": res.get("min_overlap"),
        "min_overlap_threshold": res.get("diagnostics", {}).get("min_overlap_threshold"),
        "diagnostics": res.get("diagnostics", {}),
    }

