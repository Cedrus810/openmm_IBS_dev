# ============================================================================
# 6.4 Boresch attachment 专用 HREMD：同温度，只交换 lambda_boresch_scale
# ============================================================================


class BoreschAttachmentREMDManager(REMDManager):
    """Hamiltonian REMD over `lambda_boresch_scale`（同温，不是温度 REMD）。

    ⛔ **不是生产路径，默认不可达。** 2026-07-28 首次启用产出
    `38.6006 ± 109.9858 kJ/mol`、零 round trip，而同一体系的顺序窗口给
    `5.3784`（BAR）/ `5.3867`（TI），两者一致。

    根因不是实现错，是物理：Boresch 的 `k(1−cosΔ)` 项在反转（Δ=π）时取 2k——
    单个二面角 359–469 kJ/mol（144–188 kT），三个全反转 1189 kJ/mol = 477 kT。
    副本交换一旦把某个副本送进翻转态，那一帧的天文 U_B 就支配了指数平均
    （σ=110 就是它的指纹）。顺序窗口从不离开原盆地，反而良态。

    而且**单盆地是必要的**：配套的解析释放项
    （`calculate_boresch_analytical_correction`）本身假定单一简谐盆地，
    配一个会采到翻转的 attachment 腿反而不自洽。

    要重新启用，先解决翻转态的处理（例如对 U_B 设物理上界、或改用不含
    `1−cos` 反转分支的限制形式），并且必须过 `max(round_trips) > 0` 这道门。

    与 `ShadowBridgeREMDManager` 同一套路：交换循环（Metropolis 判据、DCD 落盘、
    诊断）完全继承自 `REMDManager.run()`——那部分只通过 `_set_context_state` /
    `_context_lambda_label` 间接接触 lambda_coul/lambda_vdw，本身与具体参数无关。
    这里只重写三个方法，把驱动变量换成 `lambda_boresch_scale`。

    额外记录 `state_history`：每次落帧时快照 context→state 映射，用来数
    λ=0 ↔ 1 的 round trip。REMDManager 基类只维护当前映射、不留历史。
    """

    def __init__(
        self,
        attachment_system: openmm.System,
        topology: app.Topology,
        positions,
        box_vectors,
        ligand_indices: List[int],
        lambdas_boresch: List[float],
        lam_name: str = BORESCH_ATTACHMENT_LAMBDA_NAME,
        temperature: float = 300.0,
        platform_name: str = "CUDA",
        output_dir: str = "./boresch_attachment_remd",
        random_seed: Optional[int] = None,
        seed_ledger: Optional[Exp019SeedLedger] = None,
        seed_stage: Optional[str] = None,
        seed_leg: Optional[str] = None,
    ):
        self.lambdas_boresch = np.asarray(lambdas_boresch, dtype=float)
        self.lam_name = lam_name
        self.state_history: List[List[int]] = []
        n = len(self.lambdas_boresch)
        super().__init__(
            system_template=attachment_system,
            topology=topology,
            positions=positions,
            box_vectors=box_vectors,
            ligand_indices=ligand_indices,
            lambdas_coul=[0.0] * n,   # 占位，只为让基类旧字段存在
            lambdas_vdw=[1.0] * n,
            temperature=temperature,
            platform_name=platform_name,
            output_dir=output_dir,
            boresch_params=None,      # 限制力已经在 attachment_system 里了
            random_seed=random_seed,
            seed_ledger=seed_ledger,
            seed_stage=seed_stage or "boresch_attachment",
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
                integ = openmm.LangevinMiddleIntegrator(
                    self.temperature, 1.0 / unit.picosecond, 0.002 * unit.picosecond
                )
                integrator_seed = self._seed_for(
                    "attachment", self.seed_stage, i, "integrator"
                )
                if integrator_seed is not None:
                    integ.setRandomNumberSeed(integrator_seed)
                ctx = openmm.Context(replica_sys, integ, platform, props)
                ctx.setPositions(self.positions)
                if self.box_vectors is not None:
                    ctx.setPeriodicBoxVectors(*self.box_vectors)
                self._try_set_context_parameter(ctx, self.lam_name, self.lambdas_boresch[i])
                velocity_seed = self._seed_for(
                    "attachment", self.seed_stage, i, "velocity"
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
                    "  [WARN] attachment HREMD GPU Context 构建失败，已释放已创建的 contexts；"
                    f"回退 CPU 重建。原始错误: {exc}"
                )
                self.platform_name = "CPU"
                self._build_replicas(system_template, allow_platform_fallback=False)
                return
            raise

    def _set_context_state(self, context_idx: int, state_idx: int) -> None:
        ctx = self.contexts[context_idx]
        self._try_set_context_parameter(ctx, self.lam_name, self.lambdas_boresch[state_idx])

    def _context_lambda_label(self, context_idx: int) -> Tuple[int, float, float]:
        state_idx = int(self._context_to_state[int(context_idx)])
        lam_val = float(self.lambdas_boresch[state_idx])
        if int(context_idx) == 0:
            # 每一轮落帧从 context 0 开始，借这里快照整张映射表。
            self.state_history.append(list(self._context_to_state))
        return state_idx, lam_val, lam_val

    def begin_production_stage(self) -> None:
        """平衡段跑完、生产段开始前调用。

        两件事：
        1. `super()`：把 `_steps_completed` 归零（DCD append 模式，见基类
           `REMDManager.begin_production_stage` 的说明）；
        2. 清空 `state_history`，别把平衡期的交换记录混进生产统计。
        `_is_warmed_up` 不动，预热不会重做。
        """
        super().begin_production_stage()
        self.state_history = []

    def cleanup(self) -> None:
        self._clear_replica_contexts()

    def round_trip_counts(self) -> List[int]:
        """每个 replica 完成了几次 λ=0 ↔ λ=1 的往返。

        往返 = 摸到一端后再摸到另一端算半程，两个半程算一次。REMD 的采样效率
        直接由它决定：没有往返，副本交换等于没做。
        """
        if not self.state_history:
            return [0] * self.n_replicas
        top = self.n_replicas - 1
        trips = []
        for ctx in range(self.n_replicas):
            last_end, halves = None, 0
            for snapshot in self.state_history:
                st = snapshot[ctx]
                if st == 0 or st == top:
                    end = "lo" if st == 0 else "hi"
                    if last_end is not None and end != last_end:
                        halves += 1
                    last_end = end
            trips.append(halves // 2)
        return trips


# ============================================================================
# ⚠️ [0831issue P2] 以下两个函数当前**没有任何调用方**（生产代码与 tests/ 均为
# 零引用，已 grep 确认）。它们实现的是 #79 的 attachment 收敛诊断
# （round-trip 硬门、⟨U_B⟩ 单调性），也就是说**那道诊断当前不在线上执行**。
#
# 这里刻意**不删**、也不擅自接线：
#   * 接线会改变 attachment 腿的放行判据（新增硬门），属协议变更，需要先决定
#     阈值与 fail-closed 语义，不能顺手做；
#   * 删除有先例（`tests/test_att27_dead_code_removed.py` 归档删过一批），但那
#     是在确认该诊断不再需要之后。#79 并没有被撤销。
# 需要做的决定：接线（并定阈值）还是按 att27 先例归档删除。在做出决定前，别把
# "代码存在" 当成 "这道诊断在跑"。
# ============================================================================
def compute_boresch_attachment_u_kn(
    attachment_system: openmm.System,
    topology: app.Topology,
    traj_files: List[str],
    box_vectors=None,
    platform_name: str = "CPU",
) -> Tuple[np.ndarray, np.ndarray]:
    """离线重算每帧的 **U_Boresch(x)**（kJ/mol，λ=1 下的未缩放值）。

    只取 Boresch 力组。U_common（键合 / 环境 PME / 配体内部）对同一帧在所有 λ
    下完全相同，是逐帧可加常数，在 MBAR 的自洽方程里精确抵消——与
    `compute_shadow_bridge_u_kn` 同一论证。因为势对 λ 严格线性
    （表达式里 λ 是整体前因子），拿到 U_B 之后 u_kn[j] = λ_j·U_B 即可，
    不需要逐 λ 重新评估。

    返回 ``(u_boresch_kj_per_frame, n_k)``，帧按 traj_files 顺序拼接。
    """
    import mdtraj as md

    # [2026-09-10] 同上：往返 → clone()。
    eval_sys = openmm.XmlSerializer.clone(attachment_system)
    eval_sys.thisown = 1
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    resolved_platform, props = _build_platform_properties(platform_name)
    ctx = openmm.Context(
        eval_sys, integrator, openmm.Platform.getPlatformByName(resolved_platform), props
    )
    if box_vectors is not None:
        ctx.setPeriodicBoxVectors(*box_vectors)
    # 只要 U_B，所以固定 λ=1 量一次即可。
    #
    # 直接调用、不吞异常——这是本文件唯一一处会给 setParameter 包 try/except 的
    # 地方（其它调用点，如 1778/1993/2023/2027/2033/2043 行，全都直接调用让异常
    # 传出去）。这个函数的唯一任务就是在 λ=1 下重算整条 attachment leg 每一帧的
    # U_Boresch，喂给 MBAR 的 u_kn；如果 eval_sys 上真的没有这个全局参数（比如
    # 未来改名/序列化路径变了），静默吞掉意味着后面每一帧都在"某个默认 λ"下算，
    # 而不是 λ=1，整条 leg 的自由能贡献会悄悄错掉且零信号——必须让它在这里就炸。
    ctx.setParameter(BORESCH_ATTACHMENT_LAMBDA_NAME, 1.0)

    values, n_k = [], []
    mdtop = md.Topology.from_openmm(topology)
    for path in traj_files:
        traj = md.load(path, top=mdtop)
        n_k.append(int(traj.n_frames))
        for f in range(traj.n_frames):
            ctx.setPositions(traj.xyz[f] * unit.nanometer)
            if traj.unitcell_vectors is not None:
                ctx.setPeriodicBoxVectors(
                    *[openmm.Vec3(*v) * unit.nanometer for v in traj.unitcell_vectors[f]]
                )
            e = ctx.getState(
                getEnergy=True, groups={BORESCH_ATTACHMENT_FORCE_GROUP}
            ).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            values.append(float(e))
    del ctx, integrator
    return np.asarray(values, dtype=np.float64), np.asarray(n_k, dtype=int)


def attachment_convergence_diagnostics(
    u_kn_sub, n_k_sub, lam, u_boresch_kj, n_k_raw, kt, remd, dg, err, log=print
) -> Dict:
    """按验收清单出诊断：交换接受率 / round trip / ⟨U_B⟩ 单调性 / 前后半 / 三口径一致。"""
    K = int(lam.size)
    offsets = np.concatenate(([0], np.cumsum(n_k_raw))).astype(int)
    mean_ub = [float(np.mean(u_boresch_kj[offsets[k]:offsets[k + 1]])) for k in range(K)]

    # ⟨U_B⟩ 必须随 λ 减小而单调上升：限制越松、偏离参考几何越远。倒挂 = 没采够。
    violations = [
        {"lambda_hi": float(lam[k + 1]), "lambda_lo": float(lam[k]),
         "mean_u_hi": mean_ub[k + 1], "mean_u_lo": mean_ub[k]}
        for k in range(K - 1)
        if mean_ub[k] < mean_ub[k + 1] - 1e-9
    ]

    trips = remd.round_trip_counts()
    exch_path = os.path.join(remd.output_dir, "attachment_exchange_diagnostics.json")
    exch = {}
    if os.path.isfile(exch_path):
        with open(exch_path, encoding="utf-8") as fh:
            exch = json.load(fh)

    # 前后半程
    half = {}
    try:
        sub_off = np.concatenate(([0], np.cumsum(n_k_sub))).astype(int)
        for label, (lo, hi) in (("first", (0.0, 0.5)), ("second", (0.5, 1.0))):
            cols, nk2 = [], []
            for k in range(K):
                a, b = sub_off[k], sub_off[k + 1]
                n = b - a
                s, e = a + int(lo * n), a + int(hi * n)
                if e - s < 2:
                    raise ValueError("帧数不足以切半")
                cols.append(np.arange(s, e)); nk2.append(e - s)
            an = TraditionalMBARAnalyzer(temperature=kt / 0.008314462618)
            an._last_n_k = np.asarray(nk2, dtype=int)
            half[label] = float(an.solve(u_kn_sub[:, np.concatenate(cols)], decorrelate=False)["delta_G"])
        half["drift"] = half["second"] - half["first"]
        half["drift_over_2sigma"] = abs(half["drift"]) / (2.0 * err) if err > 0 else None
    except Exception as exc:
        half = {"error": str(exc)}

    mean_acc = float(exch.get("mean_acceptance", float("nan")))
    log(f"  交换接受率 平均 {mean_acc:.3f} / 最低 {exch.get('min_acceptance', float('nan'))}")
    log(f"  round trip 每副本: {trips}")
    log(f"  ⟨U_B⟩ 逐 λ: {[round(v, 3) for v in mean_ub]}"
        + ("  [ERR] 有倒挂" if violations else "  [OK] 单调"))
    if isinstance(half.get("drift"), float):
        log(f"  前后半程 {half['first']:.4f} / {half['second']:.4f}，"
            f"漂移 {half['drift']:+.4f}（|漂移|/2σ = {half['drift_over_2sigma']:.2f}）")

    if np.isfinite(mean_acc) and mean_acc < 0.20:
        log(f"  [WARN] 平均交换接受率 {mean_acc:.3f} < 0.20，λ 太稀，加中间态而不是加时间")
    if max(trips) == 0:
        log("  [WARN] 没有任何副本完成 λ=0↔1 往返——副本交换等于没做，结果不比顺序窗口强")
    if violations:
        log(f"  [WARN] ⟨U_B⟩ 有 {len(violations)} 处倒挂，仍未收敛")

    # 零 round trip = 副本交换等于没做，结果不比顺序窗口强，禁止进候选。
    if trips and max(trips) == 0:
        raise RuntimeError(
            "attachment HREMD 零 round trip：没有任何副本走通 λ=0↔1，"
            "副本交换实际未生效，拒绝把该结果作为候选。"
        )

    return {
        "mean_u_boresch_per_state_kJ_mol": mean_ub,
        "monotonicity_violations": violations,
        "round_trips_per_replica": trips,
        "exchange": exch,
        "split_half": half,
        "acceptance_ok": bool(np.isfinite(mean_acc) and mean_acc >= 0.20),
        "round_trip_ok": bool(max(trips) > 0) if trips else False,
        "monotonic_ok": not violations,
    }


