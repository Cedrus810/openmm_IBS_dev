# ---------------------------------------------------------------------------
# Boresch 参数统一管理
# ---------------------------------------------------------------------------
def _sanitize_boresch_params(params: Dict) -> Dict:
    """清洗 Boresch 参数字典，去除单位后缀，解包嵌套"""
    if params is None:
        return None

    # 尝试解包嵌套结构
    anchors = params.get("boresch_anchors") or params
    eq = anchors.get("equilibrium_values", params.get("equilibrium_values", {}))
    fc = anchors.get("force_constants", params.get("force_constants", {}))

    eq_mapping = {
        "r0_nm": "r0", "thetaA0_rad": "thetaA0", "thetaB0_rad": "thetaB0",
        "phiA0_rad": "phiA0", "phiB0_rad": "phiB0", "phiC0_rad": "phiC0"
    }
    fc_mapping = {
        "kr_kJ_mol_nm2": "kr", "kthetaA_kJ_mol_rad2": "kthetaA",
        "kthetaB_kJ_mol_rad2": "kthetaB", "kphiA_kJ_mol_rad2": "kphiA",
        "kphiB_kJ_mol_rad2": "kphiB", "kphiC_kJ_mol_rad2": "kphiC"
    }

    try:
        clean_eq = {eq_mapping.get(k, k): float(v) for k, v in eq.items()}
        clean_fc = {fc_mapping.get(k, k): float(v) for k, v in fc.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Boresch 参数包含非数值字段: {exc}") from exc

    cleaned = {
        "receptor_indices": anchors.get("receptor_indices", params.get("receptor_indices", [])),
        "ligand_indices": anchors.get("ligand_indices", params.get("ligand_indices", [])),
        "equilibrium_values": clean_eq,
        "force_constants": clean_fc,
        "is_fallback": params.get("is_fallback", False),
    }
    for key in (
        "method",
        "diagnostics",
        "force_constants_raw",
        "force_constant_clip_ranges",
        "force_constant_clipped",
        "equilibrium_update_error",
        "real_bond_protocol_version",
    ):
        if isinstance(params, dict) and key in params:
            cleaned[key] = params[key]
        elif isinstance(anchors, dict) and key in anchors:
            cleaned[key] = anchors[key]
    return cleaned


def _has_valid_boresch_anchors(params: Optional[Dict]) -> bool:
    """判定 Boresch 参数是否包含完整 3+3 锚点，避免键存在但值为空时静默失效。"""
    if not isinstance(params, dict):
        return False
    rec_idx = params.get("receptor_indices") or []
    lig_idx = params.get("ligand_indices") or []
    return len(rec_idx) == 3 and len(lig_idx) == 3


def _sanitize_boresch_params_strict(params: Dict) -> Dict:
    """严格清洗并校验 Boresch 参数，发现锚点缺失时立即报错。"""
    cleaned = _sanitize_boresch_params(params)
    if cleaned is None:
        return None
    if not _has_valid_boresch_anchors(cleaned):
        anchors = (params.get("boresch_anchors") or params) if isinstance(params, dict) else {}
        raise ValueError(
            "Boresch 参数缺失有效锚点：需要 3 个 receptor_indices 和 3 个 ligand_indices。"
            f" 当前解析结果 receptor_indices={anchors.get('receptor_indices', params.get('receptor_indices', [])) if isinstance(params, dict) else []},"
            f" ligand_indices={anchors.get('ligand_indices', params.get('ligand_indices', [])) if isinstance(params, dict) else []}"
        )
    required_eq = ("r0", "thetaA0", "thetaB0", "phiA0", "phiB0", "phiC0")
    required_fc = ("kr", "kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC")
    eq = cleaned.get("equilibrium_values") or {}
    fc = cleaned.get("force_constants") or {}
    missing_eq = [k for k in required_eq if k not in eq]
    missing_fc = [k for k in required_fc if k not in fc]
    if missing_eq or missing_fc:
        raise ValueError(
            "Boresch 参数缺失必要字段："
            f"equilibrium missing={missing_eq}, force_constants missing={missing_fc}"
        )
    values = [float(eq[k]) for k in required_eq] + [float(fc[k]) for k in required_fc]
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Boresch 参数包含 NaN/Inf: {values}")
    if float(eq["r0"]) <= 0.0 or any(float(fc[k]) <= 0.0 for k in required_fc):
        raise ValueError("Boresch r0 和所有力常数必须为正值")
    return cleaned


BORESCH_GEOMETRY_CONVENTION_VERSION = 2
BORESCH_REAL_BOND_PROTOCOL_VERSION = 1


def resolve_boresch_restraint(config: RunConfig, pipeline: ABFEPipeline) -> Optional[Dict]:
    """统一获取 Boresch 参数。来源：
    - traditional/orb_ml: 加载外部参数文件
    - simple/fluctuation: 纯几何涨落估算 (GeometricRestraintEstimator)，不依赖任何力场/ML 势
    - orb_simple: 基于 ORB/MACE 口袋力投影的单候选估算 (原先的 "simple"，需要 MACE-OFF 许可证/模型)
    - auto: 基于 ORB/MACE 的多候选枚举估算
    """
    if not config.boresch:
        return None

    source = config.boresch_source
    output_dir = config.output

    # 🔑 基线预平衡必须无条件先跑一次，不能只在需要用轨迹估算 Boresch 的来源
    # （auto/simple/fluctuation）才触发。之前 traditional/orb_ml 直接读外部
    # 参数文件返回，从来不会走到这里，导致后续 `_rebalance_with_boresch()`
    # 直接从未平衡的原始坐标开始加 Boresch 限制力——变成"原始坐标 → Boresch
    # rebalance → IBS"，而不是正确的"基线预平衡一次 → Boresch 参数生成/加载 →
    # 带 Boresch rebalance"。这里的判断本身沿用原有的 resume/指纹缓存逻辑，
    # 只是把触发时机提到了 source 分支之前，让所有来源都必然先有这一步。
    _n_equil_steps = config.get("n_equil_steps", 5_000_000)
    # 🔑 [2026-09-01] 必须走 pipeline 的统一入口：它用**构造期冻结**的坐标并量化，
    # 否则这里（PBC 修复之前）与 pre_equilibrate 内部（修复之后）算出的指纹永远不同。
    _equil_fingerprint = pipeline.pre_equilibration_identity_fingerprint(
        _n_equil_steps
    )
    if not config.reset and equilibrium_is_done(
        output_dir,
        expected_fingerprint=_equil_fingerprint,
        # [P1-07] 用 pipeline 自己的 platform 建一次性目标 Simulation，
        # checkpoint 以真实 loadCheckpoint 成功为准。
        simulation=_checkpoint_probe_simulation(pipeline),
    ):
        # ⚠️ 措辞如实：这里**没有**把平衡末帧读回 pipeline.positions（坐标经
        # positions_sha256 进 _pre_equilibration_fingerprint，读回会让指纹自相矛盾、
        # 每次 resume 都重跑预平衡；见 load_native_system 里那段说明）。跳过的是
        # 计算，用的仍是身份绑定的初始坐标 —— 真要复用平衡末帧属协议变更。
        log.info(
            "基线预平衡已完成，跳过重跑（注意：坐标仍为身份绑定的初始坐标，"
            "未读回平衡末帧）"
        )
    else:
        log.info("▶️ 执行基线预平衡")
        pipeline.pre_equilibrate(
            n_steps=_n_equil_steps,
            save_traj=True,
            resume=config.resume and not config.reset,
            # 🔑 [2026-09-09] 原来不传这个参数，于是走 `pre_equilibrate` 的默认
            # `True`，`enable_equilibration_convergence_stop: false` 这个配置项
            # **完全无效**。`run_full_pipeline` 那条路径是转发的
            # （abfe_pipeline.py:10149），但真正跑满 5M 步基线预平衡的是**这一条**
            # —— 到 run_full_pipeline 时 equilibrium_is_done 已经是 True、整段跳过。
            # 结果：用户明确关掉早停，运行仍然在 4.25M/5M 步收敛早停，
            # 而 `equilibrium_is_done` 又把它当"合法早停"接受，生产系综实际起自
            # 一段比要求更短的平衡。
            enable_convergence_stop=getattr(
                pipeline, "enable_equilibration_convergence_stop", True
            ),
        )

    # 传统文件来源
    if source in ("traditional", "orb_ml"):
        path = config.boresch_orb if source == "orb_ml" else config.boresch_anchors
        if not path or not os.path.exists(path):
            raise ValueError(f"Boresch 参数文件不存在: {path}")
        with open(path, encoding="utf-8") as f:
            params = json.load(f)
        log.info("[OK] 加载外部 Boresch 参数: %s", path)
        return _sanitize_boresch_params_strict(params)

    # 自动估算来源 (auto / simple / fluctuation)
    boresch_file = os.path.join(output_dir, f"boresch_{source}.json")
    if config.resume and not config.reset and os.path.exists(boresch_file):
        log.info("从缓存加载 Boresch 参数: %s", boresch_file)
        with open(boresch_file, encoding="utf-8") as f:
            params = json.load(f)
        if source == "auto" and isinstance(params, dict) and isinstance(params.get("candidates"), list):
            candidates = params["candidates"]
            select_idx = int(getattr(config, "boresch_select", 1) or 1) - 1
            if select_idx < 0 or select_idx >= len(candidates):
                raise ValueError(
                    f"boresch_select={select_idx + 1} 超出缓存候选范围：共 {len(candidates)} 个候选"
                )
            params = candidates[select_idx]
        cache_convention = int(params.get("geometry_convention_version", 0) or 0)
        if (
            source in ("simple", "fluctuation")
            and cache_convention != BORESCH_GEOMETRY_CONVENTION_VERSION
        ):
            log.warning(
                "[WARN] Boresch 缓存几何约定版本 %d != %d；旧缓存可能含末帧二面角反号，"
                "将从无约束预平衡轨迹重新生成。",
                cache_convention,
                BORESCH_GEOMETRY_CONVENTION_VERSION,
            )
        elif source in ("simple", "fluctuation") and params.get("real_bond_protocol_version") != BORESCH_REAL_BOND_PROTOCOL_VERSION:
            log.warning("Boresch 缓存没有真实配体键来源，将使用力场键图重新估计锚点。")
        else:
            return _sanitize_boresch_params_strict(params)

    traj_file = os.path.join(output_dir, "pre_equilibration.dcd")
    if not os.path.exists(traj_file):
        raise RuntimeError("预平衡轨迹不存在，无法估算 Boresch 参数")
    traj_top = _boresch_mdtraj_topology(
        pipeline, runtime_gromacs_top(output_dir, config.top),
        getattr(config, "gmx_include_dir", None) or find_gmx_include_dir(getattr(config, "gmx_path", None))
    )

    # 根据来源调用不同估算器
    if source == "auto":
        from abfe_core import OrbBoreschEstimator
        estimator = OrbBoreschEstimator(temperature=config.temperature)
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        # 自动估算（默认使用最后 5ns）
        n_candidates = max(int(getattr(config, "boresch_batch", 0) or 1), 1)
        candidates = estimator.estimate_multiple_anchors_from_trajectory(
            traj, config.ligand, n_candidates=n_candidates, output_path=boresch_file, use_last_ns=5.0
        )
        if not candidates:
            raise RuntimeError("自动 Boresch 估算失败，未找到合格候选")
        select_idx = int(getattr(config, "boresch_select", 1) or 1) - 1
        if select_idx < 0 or select_idx >= len(candidates):
            raise ValueError(
                f"boresch_select={select_idx + 1} 超出候选范围：共生成 {len(candidates)} 个候选"
            )
        boresch = candidates[select_idx]

    elif source == "orb_simple":
        # 🔁 原来的 "simple"：基于 ORB/MACE-OFF 口袋力投影的单候选估算，
        # 需要加载 MACE-OFF 模型 (受限许可证)。改名以避免跟下面纯几何的
        # "simple" 混淆——如果不想用 MACE，请用 "simple" 或 "fluctuation"。
        from abfe_core import OrbBoreschEstimator
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        estimator = OrbBoreschEstimator(temperature=config.temperature)
        boresch = estimator.estimate_from_trajectory(traj, config.ligand, output_path=boresch_file)

    elif source in ("simple", "fluctuation"):
        # ✅ 传统方法：纯几何涨落估算 (mdtraj 距离/角度/二面角方差 + 等配分定理)，
        # 不加载任何力场/ML 势，不需要 MACE-OFF 许可证。"simple" 是 "fluctuation" 的别名。
        estimator = GeometricRestraintEstimator(
            temperature=config.temperature, allow_geometric_bond_fallback=False
        )
        import mdtraj as md
        traj = md.load(traj_file, top=traj_top)
        boresch = estimator.estimate_from_trajectory(traj, config.ligand, output_path=boresch_file)
        boresch["geometry_convention_version"] = BORESCH_GEOMETRY_CONVENTION_VERSION
        boresch["real_bond_protocol_version"] = BORESCH_REAL_BOND_PROTOCOL_VERSION
    else:
        raise ValueError(f"未识别的 Boresch 来源: {source}")

    # 用最后一帧更新平衡几何量
    #
    # 🔑 [2026-09-09] 这里要的是一份**未被估算器改过**的干净轨迹，所以确实要重
    # 加载；但原来是直接 `traj = md.load(...)` 覆盖，CPython 会先把新 Trajectory
    # 建出来再重绑名字 —— 峰值是**两份**全轨迹同时驻留
    # （n_frames × n_atoms × 3 × 4 B 各一份，膜体系上是几十 GB 量级）。
    # 先显式断掉旧引用（估算器可能也还攥着它）再重载，峰值降回一份。
    import mdtraj as md
    del traj, estimator
    traj = md.load(traj_file, top=traj_top)
    # [PBC-01] 连通性取自 System，不取 topology 的键（后者可能缺刚性水的 O–H、
    # 也可能多出 mmCIF 往返造的假键）。
    #
    # ⚠️ 顺序不能动，且不能包进下面那个 try：
    #   * 回卷必须在 superpose **之前**。`superpose` 只转坐标、不转
    #     `unitcell_vectors`，转完之后坐标系和盒子已经错开，再拿这套盒矢量去做
    #     minimum-image 平移就是在错的参照系里搬分子，r/θ/φ 全变。
    #   * 它的回卷后复查（`abfe_core.image_molecules_by_system` 末尾）是
    #     fail-closed 的，和 abfe_pipeline / ibs_engine 两处调用一致。放进下面
    #     那个 `except Exception → is_fallback` 里会被降级成一条 warning，
    #     撕开的构型照样喂给 assess_boresch_harmonicity。
    image_molecules_by_system(traj, pipeline.system, log=log.warning)
    protein_sel = traj.topology.select("protein and backbone")
    if len(protein_sel) > 0:
        traj.superpose(traj, 0, atom_indices=protein_sel)
    traj.center_coordinates()

    try:
        last_frame_pos = traj.xyz[-1] * unit.nanometer
        new_eq = calc_boresch_from_last_frame(
            last_frame_pos, boresch["receptor_indices"], boresch["ligand_indices"],
            # BOR-01：这里上面刚 image + center 过，解缠应当是逐位恒等；
            # 带上 box 是为了"不依赖调用点的口头保证"。
            box_vectors=(
                traj.unitcell_vectors[-1]
                if getattr(traj, "unitcell_vectors", None) is not None
                else None
            ),
        )
        if source in ("simple", "fluctuation"):
            log.info(
                "[OK] Boresch 平衡值保留无约束预平衡轨迹系综均值"
                "（geometry convention v%d；不做末帧重锚）",
                BORESCH_GEOMETRY_CONVENTION_VERSION,
            )
        else:
            boresch["equilibrium_values"] = new_eq
            log.info(
                "[OK] Boresch 平衡值已用最后一帧更新: r0=%.3f nm", new_eq.get("r0", 0)
            )
    except Exception as e:
        # ⚠️ 不能只打个警告就悄悄继续：平衡值仍是估算器给出的、未经最后一帧精修的旧值。
        # 显式标记 is_fallback，使其经 _sanitize_boresch_params_strict 透传进最终
        # provenance/结果 JSON，保证一次隐性失败可以被事后追溯，而不是产出一个
        # 查不出来的错误 ΔG_bind。
        log.warning("[WARN] 更新 Boresch 平衡值失败: %s，将使用估算器原始平衡值（未经最后一帧精修）", e)
        boresch["is_fallback"] = True
        boresch["equilibrium_update_error"] = str(e)

    # ✅ 谐振/高斯假设的模型无关校验：不管来源是 auto/orb_simple/simple/fluctuation
    # 哪一种估算器，都直接用锁定的 6 个锚点在轨迹上重算涨落分布并做非谐性判定，
    # 结果写入 diagnostics 并在不满足假设时显式报警（原先只有 fluctuation 来源
    # 才有涨落诊断，且从不影响是否报警；scan_boresch_1d_pes 需要 ML 势且只实现
    # r 坐标，从未被任何流程调用，这里用不依赖 ML 势的轨迹统计校验替代其职责）。
    try:
        harmonicity = assess_boresch_harmonicity(
            traj, boresch["receptor_indices"], boresch["ligand_indices"]
        )
        boresch.setdefault("diagnostics", {})
        boresch["diagnostics"]["boresch_harmonicity_check"] = harmonicity
        if harmonicity.get("ok") and not harmonicity.get("harmonic_assumption_ok", True):
            log.warning("[WARN] %s", harmonicity.get("warning", ""))
            boresch["diagnostics"].setdefault("warnings", [])
            if harmonicity["warning"] not in boresch["diagnostics"]["warnings"]:
                boresch["diagnostics"]["warnings"].append(harmonicity["warning"])
        elif not harmonicity.get("ok"):
            log.info("Boresch 谐振性校验未执行: %s", harmonicity.get("reason"))
    except Exception as e:
        log.warning("[WARN] Boresch 谐振性校验失败（不影响主流程，但意味着本次没有该项诊断）: %s", e)
        boresch.setdefault("diagnostics", {})
        boresch["diagnostics"]["boresch_harmonicity_check"] = {"ok": False, "reason": f"exception:{e}"}

    # 裁剪力常数到安全范围
    if "force_constants" in boresch:
        fc = boresch["force_constants"]
        # 🔑 这套范围必须跟 abfe_core.py 里另外两处对力常数的处理保持完全一致，
        # 否则同一个物理量在不同阶段被裁到不同的上限，等于悄悄改写了估计器
        # 给出的合法结果：
        #   - GeometricRestraintEstimator.estimate_from_trajectory 自己的
        #     force_constant_ranges 已经把 kthetaA/B、kphiA/B/C 裁到 [10, 1000]
        #     （simple/fluctuation 来源在到达这里之前就已经clip过一次）；
        #   - calculate_boresch_analytical_correction 对传入的 kthetaA/kthetaB
        #     做的硬性 ValueError 校验也是 [10, 1000]（其注释明确写着要跟前者
        #     一致，见该函数）。
        # 之前这里角度类力常数的上限是 200，比上面两处都窄——一个已经通过
        # estimator 自身 [10,1000] clip 的合法值（比如 500）会在这里被第二次
        # 悄悄裁到 200，且默认来源正是 simple/fluctuation，等于默认路径下的
        # Boresch restraint 被静默软化。统一成同一套 [10, 1000]，不再有第二次
        # 更紧的裁剪。kr 范围本来就与 estimator 自身一致（[100, 2000]），维持不变。
        post_clip_ranges = {
            "kr": [100.0, 2000.0],
            "kthetaA": [10.0, 1000.0],
            "kthetaB": [10.0, 1000.0],
            "kphiA": [10.0, 1000.0],
            "kphiB": [10.0, 1000.0],
            "kphiC": [10.0, 1000.0],
        }
        post_clip_raw = dict(boresch.get("force_constants_raw", {}))
        post_clip_flags = dict(boresch.get("force_constant_clipped", {}))
        for key, (lower, upper) in post_clip_ranges.items():
            if key not in fc:
                continue
            raw_value = float(fc[key])
            post_clip_raw.setdefault(key, raw_value)
            clipped_value = float(np.clip(raw_value, lower, upper))
            fc[key] = clipped_value
            post_clip_flags[key] = bool(post_clip_flags.get(key, False) or abs(clipped_value - raw_value) > 1e-8)
        boresch["force_constants_raw"] = post_clip_raw
        boresch["force_constant_clip_ranges"] = post_clip_ranges
        boresch["force_constant_clipped"] = post_clip_flags
        boresch.setdefault("diagnostics", {})
        if any(post_clip_flags.values()):
            boresch["diagnostics"].setdefault("warnings", [])
            warning = "Boresch force constants were clipped during final sanitation."
            if warning not in boresch["diagnostics"]["warnings"]:
                boresch["diagnostics"]["warnings"].append(warning)
    # 保存
    with open(boresch_file, "w", encoding="utf-8") as f:
        json.dump(boresch, f, indent=2, cls=NumpyEncoder)
    log.info("Boresch 参数已保存: %s", boresch_file)
    return _sanitize_boresch_params_strict(boresch)


