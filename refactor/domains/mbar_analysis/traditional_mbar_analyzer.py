class TraditionalMBARAnalyzer:
    """标准离线 MBAR：读取轨迹 → 重算 u_kn → 求解 ΔG"""
    def __init__(self, temperature: float = 300.0):
        self.kt = 0.008314462618 * temperature
        self.beta = 1.0 / self.kt
        self._last_lj_lrc_metadata = {
            "protocol_version": TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
            "applicable": False,
            "applied": False,
        }

    def compute_u_kn(
        self,
        traj_files: List[str],
        system_template: openmm.System,
        ligand_indices: List[int],
        lambdas_coul: List[float],
        lambdas_vdw: List[float],
        platform_name: str = "CPU",
        topology: app.Topology = None,
        reference_positions=None,
        reference_box_vectors=None,
        boresch_params: Optional[Dict] = None,
        co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        import mdtraj as md
        if topology is None:
            raise ValueError("compute_u_kn 需要 OpenMM topology，不能用 System 构建 mdtraj Topology")
        md_top = md.Topology.from_openmm(topology)

        traj_list = [md.load(path, top=md_top) for path in traj_files]
        n_k = np.array([t.n_frames for t in traj_list], dtype=int)
        traj = md.join(traj_list, check_topology=False)
        self._last_n_k = n_k
        # ✅ 在其下方紧跟着插入 PBC 解包逻辑：
        # 🔑 核心修复：强制 PBC 分子完整性解包，消除跨盒“假撕裂”导致的能量 Spike
        # 🔑 [PBC-01] 连通性取自 **System**，不取 topology 的键。这里原来是裸的
        # `traj.image_molecules(inplace=True)`：既没补刚性水的约束（O–H 只以约束
        # 存在 ⟹ 跨边界的水被逐原子回卷撕开，MEM-15），也挡不住 mmCIF 往返在链数
        # > 26 时造出来的假键（假边把水拽出一个盒长，2026-09-09 brd4 benchmark）。
        # 而且外面那层 `except → warning` 是 fail-open：撕开的坐标照样进 u_kn，
        # PME 排除对跨盒会给出完全错的能量。回卷失败现在直接抛。
        image_molecules_by_system(traj, system_template, log=print)
        print("  [OK] 轨迹 PBC 分子完整性已修复（按 System 归组），消除撕裂隐患")
        n_frames = traj.n_frames
        n_states = len(lambdas_coul)
        u_kn = np.zeros((n_states, n_frames))
        lambdas_coul_arr = np.asarray(lambdas_coul, dtype=float)
        lambdas_vdw_arr = np.asarray(lambdas_vdw, dtype=float)
        is_pme_coulomb_leg = (
            np.allclose(lambdas_vdw_arr, 1.0)
            and not np.allclose(lambdas_coul_arr, lambdas_coul_arr[0])
        )
        is_mixed_pme_alchemical = (
            not np.allclose(lambdas_coul_arr, lambdas_coul_arr[0])
            and not np.allclose(lambdas_vdw_arr, lambdas_vdw_arr[0])
        )
        use_total_energy = is_pme_coulomb_leg or is_mixed_pme_alchemical
        ligand_charge_square_sum = 0.0
        apply_pme_self_correction = False
        pme_self_metadata = {
            "applicable": bool(use_total_energy),
            "applied": False,
            "lambda_name": "lambda_coul",
            "charge_square_sum_e2": 0.0,
            "source": "not_applicable",
        }
        if use_total_energy:
            nb_force_ref = next((f for f in system_template.getForces() if isinstance(f, openmm.NonbondedForce)), None)
            if nb_force_ref is None:
                raise RuntimeError("PME 去电荷路径未找到参考 NonbondedForce，无法估算自能修正。")
            for idx in ligand_indices:
                q, _, _ = nb_force_ref.getParticleParameters(int(idx))
                q_val = q.value_in_unit(unit.elementary_charge)
                ligand_charge_square_sum += q_val * q_val
            pme_self_metadata.update({
                "charge_square_sum_e2": float(ligand_charge_square_sum),
                "source": "ligand_particle_charges_before_offset_preparation",
            })
        # 🔑 [MBAR-HOST-MEM 2026-09-09] 坐标全程留在 float32。mdtraj 的
        # `Trajectory.xyz` 恒为 float32（setter 里 `ensure_type(value, np.float32,
        # ...)`），DCD 在磁盘上也是 4 字节/坐标——原来这个 `dtype=np.float64` 一分
        # 精度都换不到，只是在宿主内存里把整条轨迹又复制一份并放大一倍（而且是
        # 在上面 `del traj_list, traj` 把别的份都还掉**之后**唯一留下的那份）。
        # 现在 `np.asarray` 是零拷贝。worker 侧仍按 chunk 升到 float64
        # （`_compute_u_kn_chunk` 第一行），那份只有 chunk_size 帧，是有界的。
        xyz_all = np.asarray(traj.xyz, dtype=np.float32)
        box_all = None
        if traj.unitcell_vectors is not None and len(traj.unitcell_vectors) > 0:
            box_all = np.asarray(traj.unitcell_vectors, dtype=np.float64)
        # 🔑 [2026-09-09] 到这里为止 `traj_list`（每个文件一份 float32）和 `traj`
        # （join 出来的第二份 float32）已经被 `xyz_all`/`box_all` 完整取代，
        # 但它们仍然被局部名字压着、活到函数结束——而函数后半段正是要 spawn
        # N 个 worker 的地方。n_k 早在 21954 就取走了，没有别的消费者。
        # 三份全量坐标同时驻留是白占，这里显式断掉。
        del traj_list, traj
        gc.collect()

        # BeutlerSoftcoreBuilder uses a CutoffPeriodic L-E force with no OpenMM
        # long-range correction.  For every leg containing that force, add the
        # missing attractive r^-6 tail to every target-state u_k(x).  This is
        # exact for the traditional REMD trajectories only when V is constant:
        # under NPT, the omitted 1/V term would also have changed the sampled
        # volume distribution, which cannot be repaired by post-processing.
        needs_traditional_lrc = not is_pme_coulomb_leg
        lj_tail_lrc_coeff = None
        lj_lrc_metadata = {
            "protocol_version": TRADITIONAL_LJ_LRC_PROTOCOL_VERSION,
            "applicable": bool(needs_traditional_lrc),
            "applied": False,
            "model": "switching_softcore_aware_r6_r12_tail",
            "switching_nm": LJ_TAIL_LRC_R_SWITCH_NM,
            "cutoff_nm": LJ_TAIL_LRC_R_CUTOFF_NM,
        }
        if needs_traditional_lrc:
            if box_all is None or len(box_all) != n_frames:
                raise RuntimeError(
                    "传统 Beutler LRC 需要每一帧的周期盒向量；当前轨迹缺少完整 unitcell_vectors。"
                )
            volumes = np.asarray(
                [_periodic_box_volume_nm3(box) for box in box_all], dtype=np.float64
            )
            mean_volume = float(np.mean(volumes))
            relative_span = float((np.max(volumes) - np.min(volumes)) / mean_volume)
            if relative_span > 1.0e-3:
                raise RuntimeError(
                    "传统 REMD 的 Beutler 软核力缺少在线 LRC，而当前轨迹盒体积有明显波动"
                    f"（relative_span={relative_span:.3e} > 1e-3）。离线追加 1/V 尾项不能"
                    "修复未按该哈密尔顿量采样的 NPT 体积分布；请改用固定盒 NVT 生产，或先"
                    "把同一 LRC 真正加入 REMD 采样系统后重跑。"
                )
            nb_force_ref = next(
                (f for f in system_template.getForces() if isinstance(f, openmm.NonbondedForce)),
                None,
            )
            if nb_force_ref is None:
                raise RuntimeError("传统 Beutler LRC 找不到参考 NonbondedForce。")
            all_params = [
                nb_force_ref.getParticleParameters(i)
                for i in range(nb_force_ref.getNumParticles())
            ]
            ligand_set = set(int(i) for i in ligand_indices)
            env_indices = [
                i for i in range(nb_force_ref.getNumParticles()) if i not in ligand_set
            ]
            tail_sigma, tail_s6_per_sigma, tail_s12_per_sigma = (
                _lj_tail_correction_sigma_resolved_moments(
                    all_params, list(ligand_indices), env_indices
                )
            )
            tail_s6 = float(np.sum(tail_s6_per_sigma))
            tail_s12 = float(np.sum(tail_s12_per_sigma))
            # BeutlerSoftcoreBuilder.build() defaults: the LJ term is scaled by
            # lambda_vdw**1 and, under
            # SOFTCORE_ALPHA_CONVENTION=dimensionless_sigma_scaled_v2,
            # D_ij(r)=0.5*sigma_ij^6*(1-lambda_vdw)**1 + r^6 (apart from its
            # negligible 1e-4*sigma_ij^6 safety floor near r=0).  Use the same
            # v3 sigma-resolved integrator as the dual-lambda/IBS path so
            # producer and worker share both the key and the physical definition.
            beutler_alpha_lj = 0.5
            beutler_m_lj = 1.0
            beutler_n_lj = 1.0
            lj_tail_lrc_coeff = _lj_tail_lrc_coefficients_kj_mol(
                lambdas_vdw_arr,
                tail_sigma,
                tail_s6_per_sigma,
                tail_s12_per_sigma,
                beutler_alpha_lj,
                beutler_m_lj,
                beutler_n_lj,
                LJ_TAIL_LRC_R_SWITCH_NM,
                LJ_TAIL_LRC_R_CUTOFF_NM,
            )
            lj_lrc_metadata.update({
                "applied": True,
                "status": "implemented_switching_softcore_aware",
                "tail_S6_kj_nm6": float(tail_s6),
                "tail_S12_kj_nm12": float(tail_s12),
                "n_sigma_groups": int(tail_sigma.size),
                "coefficients_kj_nm3_mol": lj_tail_lrc_coeff.tolist(),
                "alpha_lj": beutler_alpha_lj,
                "alpha_convention": ACESoftcorePotential.ALPHA_CONVENTION,
                "m_lj": beutler_m_lj,
                "n_lj": beutler_n_lj,
                "volume_mean_nm3": mean_volume,
                "volume_relative_span": relative_span,
            })
            print(
                f"  [OK] 传统 Beutler LRC: v{TRADITIONAL_LJ_LRC_PROTOCOL_VERSION} "
                "switching+softcore-aware r^-6/r^-12 离线尾项已启用 "
                f"({tail_sigma.size} 个 sigma 分组, V={mean_volume:.6f} nm^3, "
                f"relative_span={relative_span:.3e})"
            )

        cpu_count = max(1, os.cpu_count() or 1)
        chunk_size = max(25, min(250, int(math.ceil(n_frames / max(1, cpu_count * 2)))))
        n_chunks = max(1, int(math.ceil(n_frames / chunk_size)))
        n_workers = min(cpu_count, n_chunks)
        # 🔑 [2026-09-09] worker 数原来**只按 CPU 数**定，完全不看内存。
        #
        # 这条 Pool 用的是 `spawn`：每个子进程都要重新 import 整条 ibs_engine
        # （连带 openmm / mdtraj / numpy / torch / pymbar-JAX）、反序列化整份
        # System、再建一个全体系的 OpenMM Context。也就是说**每个 worker 的常驻
        # 成本大致就是父进程当前的 RSS**，而不是"一个 chunk 的坐标"。
        # 实测 2026-09-09 cyclod_ligand1/rep1：`400 帧 × 8 态 | workers=16` 之后
        # 子进程抛 `terminate called after throwing an instance of 'std::bad_alloc'`
        # ——不是体系大，是 16 份全量运行时同时驻留。
        #
        # 按"可用内存 / 单 worker 估算成本"再压一次，估算成本直接取父进程 RSS
        # （spawn 子进程要把同样的东西再付一遍），留一半余量给 chunk 传输和
        # 各 Context 自己的临时分配。拿不到内存读数时不改变原有行为。
        _host_now = _host_memory_mib()
        _avail = _available_host_mib()
        if _host_now and _avail:
            _per_worker_mib = max(256.0, float(_host_now[0]))
            _budget = max(1, int((0.5 * _avail) // _per_worker_mib))
            if _budget < n_workers:
                logger.warning(
                    "[u_kn] worker 数按内存下调 %d → %d（可用 %d MiB，"
                    "单 worker 估算 %.0f MiB —— spawn 子进程要重新 import 全套依赖 "
                    "+ 反序列化整份 System + 建 Context，成本≈父进程 RSS）",
                    n_workers, _budget, int(_avail), _per_worker_mib,
                )
                n_workers = _budget
        # 🔑 [性能修复：进程×线程过度并行] 每个 worker 进程的 OpenMM CPU
        # Context 之前不设线程上限，默认想用满所有物理核——n_workers 个进程
        # 同时这么干就是经典的过度并行。这里按"总物理核数 / worker 进程数"
        # 给每个 Context 分配线程预算，保证 n_workers × cpu_threads_per_worker
        # ≤ cpu_count。n_workers==1 时预算等于 cpu_count，跟原来隐式"用满
        # 所有核"的行为一致，不改变单进程路径的性能特征。
        cpu_threads_per_worker = max(1, cpu_count // max(1, n_workers))
        if is_pme_coulomb_leg:
            # ✅ 硬性保护：is_pme_coulomb_leg 仅靠 np.allclose 推断，不能作为最终防线。
            # 一旦上游误传了非纯 decharging 的 λ 表（例如 vdw 未锁定在 1.0 却被误判为
            # decharging 腿），下面的 PME-only 能量路径会静默给出物理上错误的结果。
            # 这里显式断言 lambda_vdw 必须严格等于 1.0，配置错误时硬报错而不是沉默产出错误 ΔG。
            if not np.allclose(lambdas_vdw_arr, 1.0, rtol=0.0, atol=1e-6):
                raise RuntimeError(
                    "PME 去电荷 (decharging) 分支要求所有态的 lambda_vdw 严格等于 1.0，"
                    f"但实际检测到 lambdas_vdw={lambdas_vdw_arr.tolist()}。这通常意味着 λ 路径"
                    "配置错误（例如把 vanishing/mixed 腿的 λ 表误传入了 decharging 分析），"
                    "请检查上游 lambda 生成逻辑后重试。"
                )
            prepared_system = _prepare_pme_coulomb_leg_system(
                system_template,
                ligand_indices,
                lambda_name="lambda_coul",
                allow_charged_ligand=True,
                topology=topology,
                positions=reference_positions,
                box_vectors=reference_box_vectors,
                # [MEM-00c] 同上：只读消费冻结身份，不按 reference_positions 重选。
                co_alchemical_ion_spec=co_alchemical_ion_spec,
            )
            nb_prepared = next((f for f in prepared_system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
            if nb_prepared is None:
                raise RuntimeError("PME 去电荷路径 prepared system 缺少 NonbondedForce。")
            offset_qsq, offset_rows = pme_offset_charge_square_sum(nb_prepared, "lambda_coul")
            # 🔑 关键修复：此前这里认为 OpenMM 报告的总 PME 能量里含有一个"坐标无关的
            # 自能伪项 -C·λ²"，需要在离线 u_kn 里手动加回 +C·λ² 抵消掉。但
            # NonbondedForce.addParticleParameterOffset 是 OpenMM 官方支持的线性电荷
            # 缩放机制：每次 getState(getEnergy=True) 都会用当前 λ 对应的真实（已缩放）
            # 电荷重新计算包括 Ewald self-energy (-ke·alpha/√π·Σq_i²) 在内的完整 PME
            # 能量，这个 self-energy 项是该 λ 态总哈密顿量里真实存在、且必须存在的一部分
            # （它精确抵消了倒易空间求和里对粒子自身的重复计数，不是数值伪影），MBAR/TMBAR
            # 只要求 U_k(x) 是状态 k 的正确势能，self-energy 项本就应该被完整保留在 U_k
            # 里参与自由能估计。手动加回 +C·λ² 相当于凭空注入了一个真实存在的物理量的
            # 反向修正，量级是 ke·alpha·Σq²/√π（对本配体 Σq²≈5 e² 时可达数百 kJ/mol），
            # 实测正是它把 decharging 腿的 ΔG 从物理上合理的量级拉到了 -954.81 kJ/mol
            # 这种明显偏大的结果。这里保留 Σq_offset² 的诊断计算（对排查/存档有用），但不再
            # 把它转换成能量修正项叠加进 u_kn。
            if offset_qsq > 0.0:
                pme_self_metadata.update({
                    "applied": False,
                    "charge_square_sum_e2": float(offset_qsq),
                    "source": "particle_parameter_offsets_including_coalchemical_counterion",
                    "offset_particles": offset_rows,
                    "note": (
                        "diagnostic only: OpenMM's NonbondedForce particle-parameter-offset "
                        "charge scaling already reports the physically correct, λ-dependent "
                        "PME self-energy at every state; it is a real Hamiltonian term, not an "
                        "artifact, so it is intentionally NOT added back as a correction."
                    ),
                })
                print(
                    "  PME 自能诊断（未应用修正）: "
                    f"Σq_offset²={offset_qsq:.6f} e²；OpenMM 已在每个 λ 态正确报告该项，"
                    "不再额外叠加 +Cλ² 修正。"
                )
            _add_physical_boresch_restraint(prepared_system, boresch_params, force_group=3)
            system_xml = openmm.XmlSerializer.serialize(prepared_system)
            del prepared_system
        elif is_mixed_pme_alchemical:
            prepared_system = _prepare_pme_mixed_alchemical_system(
                system_template,
                ligand_indices,
                topology=topology,
                positions=reference_positions,
                box_vectors=reference_box_vectors,
                lambda_coul_name="lambda_coul",
                lambda_vdw_name="lambda_vdw",
                restraint_params=boresch_params,
                # [MEM-00c] 重算必须用**动力学当时冻结的那个粒子**。
                # `reference_positions` 在 resume 进程里与首跑不同，
                # 以前这里会据此重新选一个离子 —— 那正是静默不一致的来源。
                co_alchemical_ion_spec=co_alchemical_ion_spec,
            )
            nb_prepared = next((f for f in prepared_system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
            if nb_prepared is None:
                raise RuntimeError("PME mixed alchemical system 缺少 NonbondedForce。")
            offset_qsq, offset_rows = pme_offset_charge_square_sum(nb_prepared, "lambda_coul")
            # 同上（is_pme_coulomb_leg 分支）：不再把 Σq_offset² 转换为 +Cλ² 能量修正，
            # 只保留诊断记录。理由见上方注释。
            if offset_qsq > 0.0:
                pme_self_metadata.update({
                    "applied": False,
                    "charge_square_sum_e2": float(offset_qsq),
                    "source": "particle_parameter_offsets_including_coalchemical_counterion",
                    "offset_particles": offset_rows,
                    "note": (
                        "diagnostic only: OpenMM's NonbondedForce particle-parameter-offset "
                        "charge scaling already reports the physically correct, λ-dependent "
                        "PME self-energy at every state; it is a real Hamiltonian term, not an "
                        "artifact, so it is intentionally NOT added back as a correction."
                    ),
                })
                print(
                    "  PME 自能诊断（未应用修正）: "
                    f"Σq_offset²={offset_qsq:.6f} e²；OpenMM 已在每个 λ 态正确报告该项，"
                    "不再额外叠加 +Cλ² 修正。"
                )
            system_xml = openmm.XmlSerializer.serialize(prepared_system)
            del prepared_system
        else:
            eval_sys = openmm.XmlSerializer.clone(system_template)
            eval_sys.thisown = 1
            nb = [f for f in eval_sys.getForces() if isinstance(f, openmm.NonbondedForce)][0]
            _restore_ligand_internal_nonbonded(
                eval_sys,
                nb,
                ligand_indices,
                zero_original_exceptions=True,
            )
            env_idx = [i for i in range(eval_sys.getNumParticles()) if i not in ligand_indices]
            sc = BeutlerSoftcoreBuilder.build(nb, ligand_indices, env_idx)
            sc.setForceGroup(1)
            eval_sys.addForce(sc)
            _add_physical_boresch_restraint(eval_sys, boresch_params, force_group=3)
            for idx in ligand_indices:
                nb.setParticleParameters(idx, 0.0, 0.1 * unit.nanometer, 0.0)
            system_xml = openmm.XmlSerializer.serialize(eval_sys)
            del eval_sys

        print(
            f"\n开始离线能量重算 | {n_frames} 帧 × {n_states} 态 | workers={n_workers} | "
            f"chunk_size={chunk_size} | cpu_threads_per_worker={cpu_threads_per_worker}"
        )
        # 🔑 [2026-09-09] 原来这里把**所有** chunk task 提前物化进 `tasks`，每个
        # task 带一份 `xyz_all[a:b].copy()` —— 合起来就是全部坐标的又一整份
        # float64 副本，在 Pool 起来之前就已经驻留，然后 `pool_tasks` 再把这批
        # dict 浅拷一遍。改成按需生成：父进程任何时刻只压着一个 chunk。
        # 串行回退路径同样走这个生成器，行为不变（只是不再预先建表）。
        _frame_offsets = list(range(0, n_frames, chunk_size))

        def _make_task(frame_offset: int, *, with_context_fields: bool) -> Dict[str, Any]:
            frame_end = min(frame_offset + chunk_size, n_frames)
            task = {
                "frame_offset": frame_offset,
                "xyz": xyz_all[frame_offset:frame_end].copy(),
                "box_vectors": None if box_all is None else box_all[frame_offset:frame_end].copy(),
                "ligand_indices": list(ligand_indices),
                "lambdas_coul": lambdas_coul_arr,
                "lambdas_vdw": lambdas_vdw_arr,
                "kt": self.kt,
                "use_total_energy": use_total_energy,
                "apply_pme_self_correction": apply_pme_self_correction,
                "ligand_charge_square_sum": ligand_charge_square_sum,
                "pme_self_correction_metadata": pme_self_metadata,
                "lj_tail_lrc_coeff_kj_mol": (
                    None if lj_tail_lrc_coeff is None else lj_tail_lrc_coeff.copy()
                ),
                "traditional_lj_lrc_protocol_version": (
                    TRADITIONAL_LJ_LRC_PROTOCOL_VERSION
                ),
            }
            if with_context_fields:
                # 串行路径（n_workers==1 / 多进程失败回退）自己建 Context，
                # 需要这三个字段；多进程路径由 `_mbar_worker_init` 在 worker
                # 启动时建好并复用，不必逐 chunk 再传一遍。
                task["system_xml"] = system_xml
                task["platform_name"] = platform_name
                task["cpu_threads"] = cpu_threads_per_worker
            return task

        def _run_serial(reason: Optional[str] = None) -> None:
            if reason:
                print(f"  [WARN] 多进程重算失败，回退单进程: {reason}")
            for offset in _frame_offsets:
                frame_offset, u_chunk = _compute_u_kn_chunk(
                    _make_task(offset, with_context_fields=True)
                )
                frame_end = frame_offset + u_chunk.shape[1]
                u_kn[:, frame_offset:frame_end] = u_chunk
                print(f"  → 帧 {frame_end}/{n_frames} 完成")

        if n_workers == 1:
            _run_serial()
        else:
            try:
                # 🔑 [性能修复：worker Context 复用 + 不重复传输 System XML]
                # 用 initializer 让每个 worker 进程启动时只反序列化一次
                # System、建一次 Context（_mbar_worker_init），同一个 worker
                # 后续通过 imap_unordered 拉到的所有 chunk 都复用它——不再
                # 像之前那样每个 chunk 都重新反序列化+重建。既然 Context 已
                # 经在 worker 启动时用 system_xml/platform_name/cpu_threads
                # 建好了，chunk task 就不用再各自带一份（这三个字段在一次
                # compute_u_kn 调用里对所有 chunk 都相同）——省掉一份不必要的
                # 重复 IPC 序列化开销。串行回退路径用带这三个字段的版本，
                # 行为不变。
                #
                # 🔑 [2026-09-09] 喂给 imap_unordered 的是**生成器**，不是预先
                # 建好的列表：父进程任何时刻只压着一个 chunk 的坐标副本，而不是
                # 全部 chunk 的副本（那等于全轨迹又一整份 float64）。
                # 🔑 [2026-09-09] 必须带超时地取结果，否则这一步会**永远挂住**。
                #
                # `multiprocessing.Pool` 检测不到 worker 死亡：
                #   * 子进程被 OOM killer SIGKILL ⇒ `imap_unordered` 永不返回；
                #   * `initializer`（这里是 `_mbar_worker_init`，要反序列化整份
                #     System 再建 Context）在子进程里抛异常 ⇒ Pool **无限重生**
                #     失败的 worker。
                # 两种都不会给父进程任何输出，表现就是打完"开始离线能量重算"
                # 之后一行都没有（2026-09-09 cyclod_ligand1/rep1 实况）。
                # `concurrent.futures.ProcessPoolExecutor` 会抛 `BrokenProcessPool`，
                # `Pool` 不会 —— 所以这里自己加时限：超时抛出去，由下面的
                # `except` 转成单进程回退，宁可慢也不要静默挂死。
                #
                # 时限按"一个 chunk 该花多久"给足余量：worker 冷启动要重新 import
                # 整套依赖再建 Context，第一个结果本来就慢，所以首个结果单独给一份
                # 更长的预算。
                _first_chunk_timeout_s = float(
                    os.environ.get("ABFE_UKN_FIRST_CHUNK_TIMEOUT_S", 900.0)
                )
                _chunk_timeout_s = float(
                    os.environ.get("ABFE_UKN_CHUNK_TIMEOUT_S", 600.0)
                )
                ctx = mp.get_context("spawn")
                with ctx.Pool(
                    processes=n_workers,
                    initializer=_mbar_worker_init,
                    initargs=(system_xml, platform_name, cpu_threads_per_worker),
                ) as pool:
                    _pool_tasks = (
                        _make_task(offset, with_context_fields=False)
                        for offset in _frame_offsets
                    )
                    _results = pool.imap_unordered(_compute_u_kn_chunk, _pool_tasks)
                    print(
                        f"  {n_workers} 个 worker 已提交，等待第一个 chunk"
                        f"（worker 冷启动含 import + 反序列化 System + 建 Context；"
                        f"首个结果超时 {_first_chunk_timeout_s:.0f}s，"
                        f"其后每个 {_chunk_timeout_s:.0f}s）...",
                        flush=True,
                    )
                    _done_chunks = 0
                    while True:
                        try:
                            frame_offset, u_chunk = _results.next(
                                timeout=(
                                    _first_chunk_timeout_s
                                    if _done_chunks == 0
                                    else _chunk_timeout_s
                                )
                            )
                        except StopIteration:
                            break
                        except mp.TimeoutError as _timeout_exc:
                            raise RuntimeError(
                                f"离线能量重算卡住：已完成 {_done_chunks}/{len(_frame_offsets)} "
                                f"个 chunk 后超过时限仍无新结果。"
                                "multiprocessing.Pool 检测不到 worker 死亡——"
                                "worker 被 OOM killer 杀掉、或 _mbar_worker_init 在子进程里"
                                "抛异常（Pool 会无限重生它），都会让这里永远等下去。"
                                f"当前宿主内存：{_host_memory_mib()}（RSS, 峰值 MiB）、"
                                f"可用 {_available_host_mib()} MiB。"
                                "下面回退单进程；要调时限用 ABFE_UKN_CHUNK_TIMEOUT_S。"
                            ) from _timeout_exc
                        frame_end = frame_offset + u_chunk.shape[1]
                        u_kn[:, frame_offset:frame_end] = u_chunk
                        _done_chunks += 1
                        print(f"  → 帧 {frame_end}/{n_frames} 完成")
            except Exception as exc:
                _run_serial(reason=str(exc))

        self._last_pme_self_correction_metadata = pme_self_metadata
        self._last_lj_lrc_metadata = lj_lrc_metadata
        return u_kn

    def solve(
        self,
        u_kn: np.ndarray,
        decorrelate: bool = True,
        primary_estimator: str = "mbar_decorrelated",
        lambdas: Optional[Sequence[float]] = None,
        ti_gate_tolerance_kJ_mol: Optional[float] = None,
        crosschecks: bool = False,
    ) -> Dict:
        """标准 MBAR 求解。

        **默认参数一字不动地保持历史行为**（`primary_estimator="mbar_decorrelated"`、
        `crosschecks=False`），所以现有 6 处生产调用点不受影响。

        primary_estimator ["mbar_decorrelated" | "adjacent_bar"]：
          `"adjacent_bar"` 时 `delta_G`/`error` 取相邻 BAR 链，并（若给了
          `lambdas` 与 `ti_gate_tolerance_kJ_mol`）跑 BAR-vs-FD-TI 一致性门。
          依据见 `stage1_estimator_crosschecks`：BAR/TI/全帧 MBAR 三者一致，
          去相关 MBAR 偏低 0.5–0.6 kJ/mol。
        crosschecks [False]：True 时把四个口径全部算出来放进
          `result["crosschecks"]`（`primary_estimator="adjacent_bar"` 时自动开启，
          因为主值本身就来自那张表）。
        """
        if primary_estimator not in ("mbar_decorrelated", "adjacent_bar"):
            raise ValueError(
                f"primary_estimator 只能是 'mbar_decorrelated' 或 'adjacent_bar'，"
                f"收到 {primary_estimator!r}"
            )
        u_kn = np.asarray(u_kn, dtype=np.float64)
        # 交叉检查必须用**原始**数组：下面 decorrelate 分支会就地换掉 u_kn/n_k。
        u_kn_original = u_kn
        K, N = u_kn.shape
        if not hasattr(self, "_last_n_k"):
            raise RuntimeError(
                "TraditionalMBARAnalyzer.solve() 缺少每态样本数 n_k。"
                "请先调用 compute_u_kn()，或在 solve() 前显式设置 analyzer._last_n_k。"
            )
        n_k = np.asarray(self._last_n_k, dtype=int)
        n_k_original = n_k
        if len(n_k) != K or int(np.sum(n_k)) != N:
            raise ValueError(f"MBAR 样本数不匹配: len(n_k)={len(n_k)}, sum(n_k)={np.sum(n_k)}, N={N}")
        if not np.all(np.isfinite(u_kn)):
            raise ValueError("u_kn 含 NaN/Inf，无法执行 MBAR")
        if not has_pymbar():
            raise ImportError("需要 pymbar 包，请安装: pip install pymbar")

        # ------------------------------------------------------------------
        # 🔑 修复（审查报告 #1）：自相关子采样
        # ------------------------------------------------------------------
        # u_kn 的列按 n_k 分块，每一块都来自同一个态自己的（demux 后）REMD 轨迹。
        # MBAR/BAR 假设逐帧样本互相独立，但相邻输出帧强相关；这里对每个态自己
        # 的能量时间序列估计统计非效率 g，再据此做去相关子采样，避免 n_k 虚高、
        # 误差棒系统性偏小（常见 2-10 倍）。
        decorrelation_diag = {"applied": False}
        if decorrelate:
            offsets = np.concatenate(([0], np.cumsum(n_k)))
            keep_cols = []
            g_per_state = []
            n_k_sub = np.zeros_like(n_k)
            for k in range(K):
                start, end = int(offsets[k]), int(offsets[k + 1])
                if end <= start:
                    g_per_state.append(1.0)
                    continue
                local_idx, g_k = subsample_series_by_autocorrelation(u_kn[k, start:end])
                g_per_state.append(g_k)
                keep_cols.append(start + local_idx)
                n_k_sub[k] = local_idx.size
            if keep_cols:
                keep_cols_arr = np.concatenate(keep_cols)
                n_frames_before = N
                u_kn = u_kn[:, keep_cols_arr]
                n_k = n_k_sub
                N = u_kn.shape[1]
                decorrelation_diag = {
                    "applied": True,
                    "statistical_inefficiency_per_state": g_per_state,
                    "n_frames_before": int(n_frames_before),
                    "n_frames_after": int(N),
                }

        # 逐列平移不会改变自由能差，但能显著改善大体系绝对能量下的数值条件。
        u_kn_stable = u_kn - np.min(u_kn, axis=0, keepdims=True)
        delta_u_diag = delta_u_distribution_diagnostics(u_kn_stable)

        last_exc = None
        last_mbar = None
        # ✅ 修复：以前 "default" protocol 一旦协方差求解失败就立刻 return
        # error=nan，"robust" protocol（专门为数值困难场景准备的备选）根本没有
        # 机会尝试——所以 err_complex=nan 会一路传到最终 ΔG_bind，而日志里那句
        # "⚠️ MBAR 协方差求解失败...回退为仅计算 ΔG 不估计误差" 淹没在一堆输出里，
        # 不容易注意到这其实意味着最终结果完全没有误差棒。现在协方差失败时先
        # 保留这个 fallback（不确定度=nan），继续尝试下一个 protocol；只有全部
        # protocol 都拿不到真正的不确定度时，才退而求其次返回第一个成功拿到 ΔG
        # 的 fallback 结果。
        no_uncertainty_fallback = None
        protocol_plan = (
            ("default", "MBAR-default"),
            ("robust", "MBAR-robust"),
        )
        for protocol, method_name in protocol_plan:
            try:
                last_mbar = _build_mbar_compatible(
                    u_kn_stable,
                    n_k,
                    solver_protocol=protocol,
                    initialize="BAR",
                    relative_tolerance=1e-6,
                    verbose=False,
                )
                try:
                    res = _compute_free_energy_result_compatible(
                        last_mbar,
                        compute_uncertainty=True,
                    )
                    df_matrix, ddf_matrix = _extract_free_energy_arrays(
                        res,
                        require_uncertainty=True,
                    )
                    err = float(ddf_matrix[0, -1] * self.kt)
                except Exception as cov_exc:
                    print(
                        f"  [WARN] MBAR 协方差求解失败 ({protocol}): {cov_exc}，"
                        "尝试下一个 solver protocol 以求拿到真正的不确定度..."
                    )
                    if no_uncertainty_fallback is None:
                        res_fb = _compute_free_energy_result_compatible(
                            last_mbar,
                            compute_uncertainty=False,
                        )
                        df_fb, _ = _extract_free_energy_arrays(
                            res_fb,
                            require_uncertainty=False,
                        )
                        dg_fb = float((df_fb[0, -1] - df_fb[0, 0]) * self.kt)
                        no_uncertainty_fallback = {
                            "delta_G": dg_fb,
                            "error": float("nan"),
                            "method": f"{method_name}-no-uncertainty",
                            "n_frames": N,
                            "n_states": K,
                            "converged": False,
                            "min_overlap": None,
                            "diagnostics": {
                                "delta_u_distribution": delta_u_diag,
                                "pme_self_correction": getattr(self, "_last_pme_self_correction_metadata", {}),
                                "uncertainty_solve_error": str(cov_exc),
                                "decorrelation": decorrelation_diag,
                            },
                        }
                    continue

                dg = float((df_matrix[0, -1] - df_matrix[0, 0]) * self.kt)
                diagnostics = {
                    "delta_u_distribution": delta_u_diag,
                    "pme_self_correction": getattr(self, "_last_pme_self_correction_metadata", {}),
                    "overlap_matrix": None,
                    "effective_sample_number": None,
                    "decorrelation": decorrelation_diag,
                }
                # ------------------------------------------------------------------
                # 🔑 修复（审查报告 #2）：这里所有 K 个态都真正有样本（REMD 全采样），
                # 标准 MBAR 重叠矩阵在这种场景下是有效的，不像 GlobalMBARAnalyzer 那种
                # 单一采样分布 + 多个零样本目标态的场景。用相邻态（|i-j|=1）重叠矩阵
                # 元素的最小值作为 min_overlap，阈值 0.03 与 abfe_core.py 在线监控里
                # 相邻窗口重叠的既有约定保持一致。
                # ------------------------------------------------------------------
                min_overlap = None
                min_overlap_threshold = 0.03

                def _min_adjacent_overlap(mbar_obj):
                    res_o = mbar_obj.compute_overlap()
                    mat = res_o["matrix"] if isinstance(res_o, dict) else res_o
                    mat = np.asarray(mat, dtype=float)
                    if K < 2:
                        return mat, None
                    vals = [float(mat[i, i + 1]) for i in range(K - 1)]
                    vals += [float(mat[i + 1, i]) for i in range(K - 1)]
                    return mat, (float(np.min(vals)) if vals else None)

                # 🔑 [2026-09-01] **重叠必须用全帧算，不能用去相关子集。**
                #
                # 重叠是**分布**的性质，不是样本量的性质。去相关只是把"我有多少独立
                # 样本"这件事表达出来，而那件事已经由 n_k / ESS / statistical_inefficiency
                # 单独把关了；再拿去相关子集去估重叠，等于把同一件事惩罚两次，
                # 而且第二次用的是一个在小样本上会崩的估计量。
                #
                # 实测事故（4W53 output_v3，甲苯 12 态去电荷，λ_coul 间距 0.0909）：
                #     全帧 600      → min_overlap = 0.0834   ΔG = −28.070 ± 0.071
                #     去相关 398    → min_overlap = 0.0205   ΔG = −28.083 ± 0.085
                # ΔG 与误差棒几乎一致（分布确实重叠得好），但去相关后态 0/3 的 g≈6.7、
                # 只剩 8 帧，重叠估计量在 8 个样本上直接崩到 0.02，把整条生产卡死。
                # 阈值 0.03 是按"分布重叠"定的，拿它去卡一个小样本估计量是错配。
                #
                # 所以：ΔG/误差仍然用去相关结果（那才是独立样本的正确口径），
                # 只有**重叠门**改用全帧。两个数都落盘，便于审计。
                try:
                    overlap_matrix, min_overlap = _min_adjacent_overlap(last_mbar)
                    diagnostics["overlap_matrix"] = overlap_matrix.tolist()
                    diagnostics["min_overlap_decorrelated"] = min_overlap
                    diagnostics["min_overlap_gauge"] = "decorrelated"
                except Exception as overlap_exc:
                    diagnostics["overlap_error"] = str(overlap_exc)

                if decorrelation_diag.get("applied"):
                    try:
                        _full_stable = u_kn_original - np.min(
                            u_kn_original, axis=0, keepdims=True
                        )
                        _full_mbar = _build_mbar_compatible(
                            _full_stable,
                            n_k_original,
                            relative_tolerance=1e-7,
                            initialize="BAR",
                            solver_protocol=protocol,
                        )
                        _full_matrix, _full_min = _min_adjacent_overlap(_full_mbar)
                        diagnostics["overlap_matrix_full_frames"] = _full_matrix.tolist()
                        diagnostics["min_overlap_full_frames"] = _full_min
                        if _full_min is not None:
                            min_overlap = _full_min
                            diagnostics["min_overlap_gauge"] = "full_frames"
                    except Exception as full_exc:
                        # fail-closed：全帧口径算不出来就沿用去相关口径，并留痕，
                        # 绝不静默当作"重叠没问题"。
                        diagnostics["overlap_full_frames_error"] = str(full_exc)
                try:
                    neff = last_mbar.compute_effective_sample_number()
                    diagnostics["effective_sample_number"] = np.asarray(neff, dtype=float).tolist()
                except Exception as neff_exc:
                    diagnostics["effective_sample_number_error"] = str(neff_exc)
                diagnostics["min_overlap_threshold"] = min_overlap_threshold
                # 🔑 [0831issue] 这里原来是裸 `min_overlap >= min_overlap_threshold`，
                # 与本文件 17xxx 行 solve_stage_integrated / early-stop 判据统一使用的
                # `_meets_minimum_with_roundoff` 不一致。overlap 来自归一化浮点求和，
                # 合法值可以落在数学边界下方几个 ulp（v23 真实案例
                # 0.04999999999999431 vs 0.05），裸 `>=` 会把它判成不收敛并触发
                # 无意义的重算。该 helper 只把舍入尺度的相等当相等，不放松物理阈值，
                # 且对 None/NaN 一律返回 False（保持原来的 fail-closed 语义）。
                converged = _meets_minimum_with_roundoff(
                    min_overlap, min_overlap_threshold
                )
                return self._finalize_solve_result(
                    {
                        "delta_G": dg,
                        "error": err,
                        "method": method_name,
                        "n_frames": N,
                        "n_states": K,
                        "converged": converged,
                        "min_overlap": min_overlap,
                        "diagnostics": diagnostics,
                    },
                    u_kn_original,
                    n_k_original,
                    primary_estimator=primary_estimator,
                    lambdas=lambdas,
                    ti_gate_tolerance_kJ_mol=ti_gate_tolerance_kJ_mol,
                    crosschecks=crosschecks,
                )
            except Exception as exc:
                last_exc = exc
                print(f"  [WARN] MBAR {protocol} 求解失败: {exc}")

        if no_uncertainty_fallback is not None:
            print(
                "  [ERR] 所有 solver protocol 均无法给出不确定度估计；"
                f"返回 {no_uncertainty_fallback['method']} 的 ΔG，error 标记为 nan——"
                "这条腿/阶段的最终误差棒不可信，请检查窗口重叠率与采样长度。"
            )
            return self._finalize_solve_result(
                no_uncertainty_fallback,
                u_kn_original,
                n_k_original,
                primary_estimator=primary_estimator,
                lambdas=lambdas,
                ti_gate_tolerance_kJ_mol=ti_gate_tolerance_kJ_mol,
                crosschecks=crosschecks,
            )

        raise RuntimeError(f"MBAR 求解失败，最后错误: {last_exc}")

    def _finalize_solve_result(
        self,
        result: Dict,
        u_kn_original: np.ndarray,
        n_k_original: np.ndarray,
        primary_estimator: str,
        lambdas: Optional[Sequence[float]],
        ti_gate_tolerance_kJ_mol: Optional[float],
        crosschecks: bool,
    ) -> Dict:
        """挂上分析协议号/策略指纹，按需算交叉检查并切换主值。

        默认路径（`primary_estimator="mbar_decorrelated"`、`crosschecks=False`）
        只加三个纯记录字段，**不做任何额外计算、不改数值**。
        """
        want_crosschecks = bool(crosschecks) or primary_estimator == "adjacent_bar"
        cc = None
        if want_crosschecks:
            try:
                cc = stage1_estimator_crosschecks(
                    u_kn_original, n_k_original, self.kt, lambdas=lambdas,
                )
            except Exception as exc:
                cc = {"error": f"交叉检查失败: {exc}"}
            result["crosschecks"] = cc

        sigma_policy = "mbar_asymptotic_on_decorrelated_frames"
        charging_frame_selection = (
            "decorrelated_per_state"
            if bool((result.get("diagnostics", {}).get("decorrelation") or {}).get("applied"))
            else "all_frames"
        )
        if primary_estimator == "adjacent_bar":
            bar = (cc or {}).get("adjacent_bar") or {}
            dg_bar = bar.get("delta_G_kJ_mol")
            if dg_bar is None:
                # fail closed：要 BAR 主值却算不出 BAR，绝不静默退回 MBAR 值。
                raise RuntimeError(
                    "primary_estimator='adjacent_bar' 但相邻 BAR 不可用："
                    f"{bar.get('not_applicable_reason') or bar.get('error')}"
                )
            result["delta_G_mbar_decorrelated_kJ_mol"] = result.get("delta_G")
            result["error_mbar_decorrelated_kJ_mol"] = result.get("error")
            result["delta_G"] = float(dg_bar)
            result["error"] = float(bar.get("error_kJ_mol", float("nan")))
            result["method"] = "adjacent-BAR-chain"
            sigma_policy = "bar_asymptotic_edge_variance_sum"
            charging_frame_selection = "all_frames_per_state"
            # BAR 的点估计刻意使用每态全部帧；当前边方差和仍是把相关帧近似为
            # 独立样本的渐近误差。P1-19 尚未给 charging 建立可采纳的块 bootstrap，
            # 所以必须把这个限制写成机器可读标记，不能只藏在 TODO 的文字里。
            result["sigma_suspect_underestimated"] = True
            result["sigma_suspect_underestimated_reason"] = (
                "adjacent BAR uses all per-state frames, while the reported "
                "asymptotic edge-variance sum is not corrected for temporal "
                "autocorrelation"
            )
            # TI 一致性门
            ti = ((cc or {}).get("reweighted_fd_ti") or {}).get("delta_G_kJ_mol")
            if ti_gate_tolerance_kJ_mol is None:
                result["ti_gate"] = {
                    "passed": None,
                    "reason": (
                        "未提供 ti_gate_tolerance_kJ_mol。容差必须显式给——不得沿用 "
                        "attachment 的 1.0 kJ/mol，两条腿量级与 TI 变体都不同。"
                    ),
                }
            else:
                gate = stage1_ti_consistency_gate(
                    float(dg_bar), float(result["error"]), ti,
                    float(ti_gate_tolerance_kJ_mol),
                )
                result["ti_gate"] = gate
            # fail closed：BAR 主值的独立一致性证据必须明确通过。TI 缺失、
            # 计算异常或调用方忘记给容差都属于“没有通过”，不能靠原 MBAR overlap
            # 的 converged=True 混过去。
            if result["ti_gate"].get("passed") is not True:
                result["converged"] = False
                result["ti_gate_failed"] = True
                if result["ti_gate"].get("passed") is False:
                    print(
                        f"  [ERR] [TI 门] BAR 与重加权 FD-TI 分歧 "
                        f"{result['ti_gate']['abs_diff_kJ_mol']:.4f} > 容差 "
                        f"{result['ti_gate']['tolerance_kJ_mol']:.4f} kJ/mol "
                        "⟹ converged=False"
                    )
                else:
                    print(
                        "  [ERR] [TI 门] BAR 主值缺少可判定且明确通过的 FD-TI 证据 "
                        f"({result['ti_gate'].get('reason')}) ⟹ converged=False"
                    )

        result["primary_estimator"] = str(primary_estimator)
        result["estimator_analysis_protocol_version"] = int(
            ESTIMATOR_ANALYSIS_PROTOCOL_VERSION
        )
        if primary_estimator == "adjacent_bar":
            result["charging_frame_selection"] = charging_frame_selection
            result["sigma_policy"] = sigma_policy
        result["estimator_policy_fingerprint"] = estimator_policy_fingerprint({
            "estimator_analysis_protocol_version": int(
                ESTIMATOR_ANALYSIS_PROTOCOL_VERSION
            ),
            "primary_estimator": str(primary_estimator),
            "charging_frame_selection": charging_frame_selection,
            "sigma_policy": sigma_policy,
            "sigma_inflation_applied": False,
            "ti_gate_tolerance_kJ_mol": (
                float(ti_gate_tolerance_kJ_mol)
                if ti_gate_tolerance_kJ_mol is not None else None
            ),
        })
        return result


# [CLI-01, 2026-09-02] `HAS_PYMBAR` 改成惰性探测后模块里不再有这个全局名；
# 外部若仍读 `ibs_engine.HAS_PYMBAR`，这里用 PEP 562 补上布尔值。
# 模块内部一律调 `has_pymbar()`（全局名查找不走 `__getattr__`）。
