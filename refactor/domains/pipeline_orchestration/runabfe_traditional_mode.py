# ---------------------------------------------------------------------------
# 传统 ABFE-REMD 模式
# ---------------------------------------------------------------------------
def _assert_traditional_protocol_supported(config: RunConfig) -> None:
    """[P1-14] traditional 模式的协议声明检查（建任何 OpenMM Context 之前调用）。

    膜 barostat/色散协议/力场族这些配置此前只会进入临时 Boresch ABFEPipeline，
    真正的 TraditionalABFEPipeline 从不消费它们——用户声明的协议被静默忽略、
    按默认路线跑。这里把"traditional 不支持的声明"明确拒绝，只放行默认
    （不声明 = soluble / legacy_uniform_density_lrc / auto 力场族）路线。
    """
    unsupported = []
    system_type = config.get("system_type")
    membrane = config.get("membrane")
    dispersion = config.get("dispersion_protocol")
    forcefield = config.get("forcefield_family")
    if system_type is not None and str(system_type) != "soluble":
        unsupported.append(f"system_type={system_type!r}")
    if membrane:
        unsupported.append(f"membrane={membrane!r}")
    if dispersion is not None and str(dispersion) != "legacy_uniform_density_lrc":
        unsupported.append(f"dispersion_protocol={dispersion!r}")
    if forcefield is not None and str(forcefield).strip().lower() not in ("", "auto"):
        unsupported.append(f"forcefield_family={forcefield!r}")
    if unsupported:
        raise RuntimeError(
            "traditional 模式不支持以下协议声明（它们会被静默忽略并按默认路线"
            "运行，这比报错更糟）：\n    - " + "\n    - ".join(unsupported)
            + "\n传统 PME-REMD 路径只实现默认（soluble / legacy_uniform_density_lrc /"
            " auto 力场族）协议。需要膜/非默认色散/显式力场族请使用 IBS dual_lambda，"
            "或从配置/命令行里去掉这些声明。"
        )


def run_traditional_mode(config: RunConfig):
    """传统双阶段 λ-REMD：分别计算复合物腿与溶剂腿并汇总结合自由能。"""
    log.info("🔧 启动传统 ABFE-REMD 模式")
    output_dir = config.output
    os.makedirs(output_dir, exist_ok=True)

    if not config.gro or not config.top:
        raise ValueError("traditional 模式必须提供 gro/top 输入，或在配置文件中定义 gro/top。")

    system, topology, positions, box_vectors, ligand_indices = build_system_from_gromacs(
        config.gro,
        config.top,
        config.ligand,
        find_gmx_include_dir(config.gmx_path),
    )
    _traditional_raw_charge = float(_compute_ligand_net_charge(system, ligand_indices))
    if not np.isfinite(_traditional_raw_charge):
        raise ValueError(
            f"traditional 配体净电荷不是有限数：{_traditional_raw_charge!r}"
        )
    _traditional_integer_charge = int(round(_traditional_raw_charge))
    if (
        abs(_traditional_raw_charge - _traditional_integer_charge)
        > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E
    ):
        raise ValueError(
            f"traditional 配体净电荷 {_traditional_raw_charge:+.8f} e 不接近整数"
        )
    if _traditional_integer_charge != 0:
        raise RuntimeError(
            "traditional PME-REMD 当前只支持中性配体；检测到配体净电荷 "
            f"{_traditional_integer_charge:+d} e。该路径尚未实现 co-alchemical ion，"
            "已在创建任何 OpenMM Context 前拒绝运行。请使用 IBS dual_lambda。"
        )

    # 🔑 [P1-14] traditional 接入协议声明检查：见 _assert_traditional_protocol_supported。
    # 必须在建任何 OpenMM Context（预平衡/REMD）之前 fail closed。
    _assert_traditional_protocol_supported(config)
    diagnose_14_scaling(system)

    if hasattr(positions, "value_in_unit"):
        pos_nm = positions.value_in_unit(unit.nanometer)
    else:
        pos_nm = positions
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    if pos_nm.ndim == 1:
        pos_nm = pos_nm.reshape(-1, 3)
    positions = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_nm] * unit.nanometer
    positions, box_vectors = center_system_rigidly(positions, box_vectors, ligand_indices)

    ligand_resname = _get_residue_name_by_atom_index(topology, ligand_indices[0])
    solvent_identity = _ligand_parameter_identity(
        system,
        topology,
        ligand_indices,
        ligand_resname,
        config.top,
        getattr(config, "ligand_xml", None),
        find_gmx_include_dir(config.gmx_path),
        positions=positions,
    )
    if config.reset or not solvent_cache_exists(
        output_dir,
        config.solvent_ionic_strength_molar,
        solvent_identity,
        charge_treatment=config.get("charge_treatment"),
    ):
        log.info("traditional 模式准备溶剂腿缓存...")
        if not build_and_cache_solvent_leg(
            output_dir,
            topology,
            positions,
            ligand_indices,
            ligand_resname,
            ligand_ffxml=getattr(config, "ligand_xml", None),
            top_file=config.top,
            gmx_include_dir=find_gmx_include_dir(config.gmx_path),
            ionic_strength_molar=config.solvent_ionic_strength_molar,
            cache_identity=solvent_identity,
            charge_treatment=config.get("charge_treatment"),
        ):
            raise RuntimeError("traditional 模式自动构建溶剂腿缓存失败。")

    # 🔑 [P1-13] 基线预平衡独立于 Boresch。此前预平衡只发生在 `if config.boresch:`
    # 里（借道 resolve_boresch_restraint 的"无条件先跑一次基线预平衡"），显式
    # 关闭 Boresch 后两条腿直接 setPositions/setVelocities 进 REMD——原始或仅
    # 居中的坐标直接进入生产采样。现在无条件先建统一 ABFEPipeline、按统一
    # fingerprint/checkpoint/完成门做（或复用）基线预平衡；Boresch 阶段
    # （resolve_boresch_restraint）会通过 equilibrium_is_done 自然复用它，
    # 不会跑两遍。
    traditional_baseline = ABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=config.temperature,
        output_dir=output_dir,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        platform_name=config.platform,
        # 复合物腿环境 → 走与主流程同一份膜协议声明（P1-14 已把非默认声明
        # 在这里之前拒绝，所以能走到这里的只有默认 soluble）。
        environment_type=config.get("system_type"),
        membrane=config.get("membrane"),
        confirm_soluble_with_lipids=bool(
            config.get("confirm_soluble_with_lipids", False)
        ),
        dispersion_protocol=config.get("dispersion_protocol"),
        forcefield_family=config.get("forcefield_family"),
        charge_treatment=config.get("charge_treatment"),
    )
    _trad_n_equil_steps = config.get("n_equil_steps", 5_000_000)
    # 🔑 [2026-09-01] 必须走 pipeline 的统一入口：它用**构造期冻结**的坐标并量化，
    # 否则这里（PBC 修复之前）与 pre_equilibrate 内部（修复之后）算出的指纹永远不同。
    _trad_equil_fingerprint = (
        traditional_baseline.pre_equilibration_identity_fingerprint(_trad_n_equil_steps)
    )
    if not config.reset and equilibrium_is_done(
        output_dir,
        expected_fingerprint=_trad_equil_fingerprint,
        # [P1-07] checkpoint 以目标 Simulation 的真实 loadCheckpoint 成功为准。
        simulation=_checkpoint_probe_simulation(traditional_baseline),
    ):
        log.info(
            "traditional 基线预平衡已完成，跳过重跑（坐标仍为初始坐标，"
            "未读回平衡末帧）"
        )
    else:
        log.info("▶️ 执行 traditional 基线预平衡")
        traditional_baseline.pre_equilibrate(
            n_steps=_trad_n_equil_steps,
            save_traj=True,
            resume=config.resume and not config.reset,
            # [2026-09-09] 同 resolve_boresch_restraint 那处：不传就走默认 True，
            # 配置里的 enable_equilibration_convergence_stop 形同虚设。
            enable_convergence_stop=bool(
                config.get("enable_equilibration_convergence_stop", True)
            ),
        )
    positions = traditional_baseline.positions
    box_vectors = traditional_baseline.box_vectors

    dg_boresch = 0.0
    err_boresch = 0.0
    boresch_restraint = None
    boresch_attachment_result = None
    if config.boresch:
        boresch_restraint = resolve_boresch_restraint(config, traditional_baseline)
        if boresch_restraint:
            boresch_correction_result = traditional_baseline.apply_boresch_correction(
                boresch_restraint,
                autoload_from_disk=False,
            )
            dg_boresch = float(boresch_correction_result.get("delta_g_rest", 0.0))
            err_boresch = float(boresch_correction_result.get("error", 0.0))
            positions = traditional_baseline.positions
            box_vectors = traditional_baseline.box_vectors
            from ibs_engine import run_boresch_attachment_leg
            boresch_attachment_result = run_boresch_attachment_leg(
                system,
                topology,
                positions,
                box_vectors,
                boresch_restraint,
                temperature_k=float(config.temperature),
                lambdas=config.get("attachment_lambdas"),
                n_steps_per_state=int(config.get("attachment_n_steps_per_state", 250_000)),
                equil_steps_per_state=int(config.get("attachment_equil_steps_per_state", 50_000)),
                steps_per_sample=int(config.get("attachment_steps_per_sample", 1_000)),
                platform_name=config.platform,
                seed=int(config.get("attachment_seed", 20260728)),
                n_seeds=int(config.get("attachment_n_seeds", 1)),
                output_dir=os.path.join(output_dir, "traditional_attachment"),
                log=log.info,
            )

    complex_pipeline = TraditionalABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=config.temperature,
        platform_name=config.platform,
        output_dir=os.path.join(output_dir, "traditional_complex"),
        repeat_seed=config.get("repeat_seed"),
        leg_name="complex",
    )

    log.info("开始传统复合物腿 REMD + MBAR (n_lambda=%d, n_steps=%d)",
             config.n_lambda, config.n_steps_per_window or 500000)
    complex_results = complex_pipeline.run_full(
        allow_untrusted_stage_results=bool(
            config.get("allow_untrusted_stage_results", False)
        ),
        n_lambda=config.n_lambda,
        n_steps_per_leg=config.n_steps_per_window or 500000,
        boresch_correction=0.0,
        boresch_params=boresch_restraint,
        attachment_result=boresch_attachment_result,
        potential_type=config.potential,
        resume=config.resume and not config.reset,
    )

    sys_solv, top_solv, pos_solv, box_solv, lig_idx_solv = load_native_system(
        output_dir,
        phase="solvent",
        prefer_equilibrated=not config.reset,
    )
    pos_solv, box_solv = center_system_rigidly(pos_solv, box_solv, lig_idx_solv)

    # 🔑 [P1-13] 溶剂腿同样要有基线预平衡（"两条腿预平衡"）：此前溶剂腿直接从
    # 缓存/居中坐标进 REMD。runtime_dir 与主流程一致（output_dir/solvent_leg），
    # 这样 load_native_system 的 prefer_equilibrated 读的就是同一份严格校验过
    # 的预平衡轨迹，两种模式共享同一布局与完成门。
    _solv_runtime_dir = os.path.join(output_dir, "solvent_leg")
    solvent_baseline = ABFEPipeline(
        system=sys_solv,
        topology=top_solv,
        positions=pos_solv,
        box_vectors=box_solv,
        ligand_indices=lig_idx_solv,
        temperature=config.temperature,
        output_dir=_solv_runtime_dir,
        checkpoint_dir=os.path.join(_solv_runtime_dir, "checkpoints"),
        platform_name=config.platform,
        # 与 complex 侧 baseline 完全同一组协议声明（P1-14 已把膜/非默认色散
        # 声明在更早处拒绝，能走到这里的只有默认 soluble 路线）；审计测试要求
        # 每个 pipeline 构造都显式做出膜/色散决定，不许留默认隐式值。
        environment_type=config.get("system_type"),
        membrane=config.get("membrane"),
        confirm_soluble_with_lipids=bool(
            config.get("confirm_soluble_with_lipids", False)
        ),
        dispersion_protocol=config.get("dispersion_protocol"),
        forcefield_family=config.get("forcefield_family"),
        charge_treatment=config.get("charge_treatment"),
    )
    # 🔑 [2026-09-01] 必须走 pipeline 的统一入口：它用**构造期冻结**的坐标并量化，
    # 否则这里（PBC 修复之前）与 pre_equilibrate 内部（修复之后）算出的指纹永远不同。
    _solv_equil_fingerprint = (
        solvent_baseline.pre_equilibration_identity_fingerprint(_trad_n_equil_steps)
    )
    if not config.reset and equilibrium_is_done(
        _solv_runtime_dir,
        expected_fingerprint=_solv_equil_fingerprint,
        simulation=_checkpoint_probe_simulation(solvent_baseline),
    ):
        log.info(
            "traditional 溶剂腿基线预平衡已完成，跳过重跑（坐标仍为初始坐标，"
            "未读回平衡末帧）"
        )
    else:
        log.info("▶️ 执行 traditional 溶剂腿基线预平衡")
        solvent_baseline.pre_equilibrate(
            n_steps=_trad_n_equil_steps,
            save_traj=True,
            resume=config.resume and not config.reset,
            # [2026-09-09] 同上。
            enable_convergence_stop=bool(
                config.get("enable_equilibration_convergence_stop", True)
            ),
        )
    pos_solv = solvent_baseline.positions
    box_solv = solvent_baseline.box_vectors

    solvent_pipeline = TraditionalABFEPipeline(
        system=sys_solv,
        topology=top_solv,
        positions=pos_solv,
        box_vectors=box_solv,
        ligand_indices=lig_idx_solv,
        temperature=config.temperature,
        platform_name=config.platform,
        output_dir=os.path.join(output_dir, "traditional_solvent"),
        repeat_seed=config.get("repeat_seed"),
        leg_name="solvent",
    )
    log.info("开始传统溶剂腿 REMD + MBAR (n_lambda=%d, n_steps=%d)",
             config.n_lambda, config.n_steps_per_window or 500000)
    solvent_results = solvent_pipeline.run_full(
        allow_untrusted_stage_results=bool(
            config.get("allow_untrusted_stage_results", False)
        ),
        n_lambda=config.n_lambda,
        n_steps_per_leg=config.n_steps_per_window or 500000,
        boresch_correction=0.0,
        boresch_params=None,
        potential_type=config.potential,
        resume=config.resume and not config.reset,
    )

    dg_complex = float(complex_results["delta_G_total_kJ_mol"])
    dg_solvent = float(solvent_results["delta_G_total_kJ_mol"])
    err_complex = float(complex_results["error_leg_kJ_mol"])
    err_solvent = float(solvent_results["error_leg_kJ_mol"])
    _constraint_identity = _assert_matching_ligand_constraint_identity(
        system,
        ligand_indices,
        sys_solv,
        lig_idx_solv,
        context="traditional ABFE 热力学循环",
    )
    # 🔑 [ATT-09] 循环闭合统一走 abfe_core.combine_binding_free_energy。
    # 这条路径 `TraditionalABFEPipeline.run_full` 是以 boresch_correction=0.0 调的，
    # 所以 complex 腿**不含**释放项，already_included=False，由 helper 减一次。
    #
    # 🔑 [ATT-09] 同时补上此前完全缺失的 APBS 项：main() 和 run_post_analysis()
    # 一直在读 config["apbs_correction_kJ_mol"]，只有 traditional 这条路径从来没读，
    # 等于对带电配体静默漏掉整项有限尺寸静电修正。这个标量由离线的
    # apbs_correction.py（prepare → run → collect）算出，collect 的输出里直接给了
    # 要传的 --apbs-correction-kj-mol；这里只是消费同一个值，不在流程内调 APBS。
    apbs_correction = float(config.get("apbs_correction_kJ_mol", 0.0) or 0.0)
    cycle = combine_binding_free_energy(
        dg_complex_kJ_mol=dg_complex,
        dg_solvent_kJ_mol=dg_solvent,
        err_complex_kJ_mol=err_complex,
        err_solvent_kJ_mol=err_solvent,
        dg_boresch_kJ_mol=dg_boresch,
        boresch_already_included_in_complex=False,
        apbs_correction_kJ_mol=apbs_correction,
    )
    delta_g_bind = cycle["delta_G_bind_kJ_mol"]
    # 🔑 [0831issue P2] `err_boresch` 不再并入 total_error。
    # `abfe_core.combine_binding_free_energy` 的 docstring 是唯一约定：
    # "Boresch 解析释放项与 APBS 都是确定性解析量，没有独立采样方差，不并入。"
    # 这里原来单独加了一次平方和，于是同一批工件经 `run_traditional_mode` 与
    # `run_post_analysis --analyze-only --mode traditional` 会得到**两个不同的 ±**，
    # 跨 seed 的 SEM 与文献比较口径随之漂移。现在两条路径都用 cycle 的口径；
    # `err_boresch` 仍作为独立字段落盘（`boresch_correction_error_kJ_mol`），
    # 需要它的下游自己按解析模型误差处理，不混进统计误差棒。
    total_err_bind = float(cycle["total_error_kJ_mol"])

    final = {
        "complex_leg": complex_results,
        "solvent_leg": solvent_results,
        "complex_delta_G_kJ_mol": dg_complex,
        "solvent_delta_G_kJ_mol": dg_solvent,
        "boresch_correction_kJ_mol": float(dg_boresch),
        "boresch_correction_error_kJ_mol": float(err_boresch),
        "constraint_identity": _constraint_identity,
        # ✅ TraditionalABFEPipeline.run_full 是以 boresch_correction=0.0 调用的，
        # 因此 complex_delta_G_kJ_mol 不含 Boresch 释放修正；该修正只作为独立项
        # 已经减到 delta_G_bind_kJ_mol 里（见上方 delta_g_bind 公式），下游不要
        # 再对 complex_delta_G_kJ_mol 或 delta_G_bind_kJ_mol 二次扣减。
        "boresch_correction_already_included_in_complex_delta_G": False,
        "delta_G_bind_uncorrected_kJ_mol": float(
            cycle["delta_G_bind_uncorrected_kJ_mol"]
        ),
        "apbs_correction_kJ_mol": float(apbs_correction),
        "delta_G_bind_kJ_mol": float(delta_g_bind),
        "delta_G_bind_kcal_mol": float(delta_g_bind / 4.184),
        "total_error_kJ_mol": total_err_bind,
        "thermodynamic_cycle_terms": cycle,
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(output_dir, "final_binding_results_traditional.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, cls=NumpyEncoder)
    log.info("[OK] 传统 ABFE 完成: ΔG_bind = %.2f ± %.2f kJ/mol", delta_g_bind, total_err_bind)
    log.info("传统模式最终结果已保存: %s", out_path)
    return final

