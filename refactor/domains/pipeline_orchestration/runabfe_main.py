# ---------------------------------------------------------------------------
def _scope_pipeline_with_optional_outer_lambda_em(
    pipeline: ABFEPipeline,
    em_enabled: bool,
):
    """Scope the candidate-only twin EM policy to the next leg call.

    The context is deliberately scoped to the complete leg, because Stage 2
    also builds fixed-endpoint/overlap probe Systems.  Baseline runs never
    install the monkeypatch and therefore retain native OpenMM EM exactly.
    """
    if not em_enabled:
        return
    original_run = pipeline.run_full_pipeline

    def scoped_run(*args, **kwargs):
        install_outer_lambda_em_policy()
        try:
            return original_run(*args, **kwargs)
        finally:
            uninstall_outer_lambda_em_policy()
            pipeline.run_full_pipeline = original_run

    # An instance-level wrapper keeps the public source-level call and its
    # explicit keyword contract intact while guaranteeing cleanup on errors.
    pipeline.run_full_pipeline = scoped_run


# [CLI-01, 2026-09-02] 只读诊断子命令的名字。它们各自带一个 parser（在
# `abfe_diagnostics` 里），所以必须在主 parser 之前分流：`--json` 这类选项
# 主 parser 不认识，交给它解析会直接报错退出。
DIAGNOSTIC_COMMANDS = ("doctor", "validate-config", "config-template")


def dispatch_diagnostic_command(argv: List[str]) -> Optional[int]:
    """argv[0] 是诊断子命令就执行并返回退出码，否则返回 None。

    独立成函数是为了能在不碰 `sys.argv` 的前提下测试分流本身。
    """
    if not argv or argv[0] not in DIAGNOSTIC_COMMANDS:
        return None
    import abfe_diagnostics

    handlers = {
        "doctor": abfe_diagnostics.doctor_main,
        "validate-config": abfe_diagnostics.validate_config_main,
        "config-template": abfe_diagnostics.config_template_main,
    }
    return handlers[argv[0]](argv[1:])



def _path_evolution_kwargs(config) -> dict:
    """路径演化的两个开关，**只在配置里显式设置了才透传**。

    不能无条件 `config.get(...)` 就往下传：那会往 run_config 里塞进一个键（哪怕值是
    None），而 run_config 是逐字段进 stage 协议指纹的 —— 等于让所有现有缓存平白失配、
    重跑 GPU（`ABFEPipeline._stage_protocol_key` 的注释记过这个代价：Stage 1 约 28
    分钟 + 6 个 vanishing 窗口全部重采样）。没设置就一个键都不加，行为逐字不变。
    """
    out = {}
    policy = config.get("sampling_repair_policy")
    if policy is not None:
        out["sampling_repair_policy"] = str(policy)
    budget = config.get("max_path_insertions")
    if budget is not None:
        out["max_path_insertions"] = int(budget)
    recal = config.get("stage2_recalibrate_f_k_on_rescue")
    if recal is not None:
        out["stage2_recalibrate_f_k_on_rescue"] = bool(recal)
    shift = config.get("stage2_f_k_recalibration_min_shift")
    if shift is not None:
        out["stage2_f_k_recalibration_min_shift"] = float(shift)
    # 🔑 [2026-09-11 实验开关，默认 False] 重标定出来的 f_k 进生产前不再被
    # loose gate **拦住**（门照跑照记，只是不阻断）。见 ibs_engine 里
    # accept_recalibrated_f_k_without_gate 的参数注释。
    _accept_recal = config.get("stage2_accept_recalibrated_f_k_without_gate")
    if _accept_recal is not None:
        out["stage2_accept_recalibrated_f_k_without_gate"] = bool(_accept_recal)
    ranges = config.get("stage2_window_ranges")
    if ranges:
        out["stage2_window_ranges"] = [[int(a), int(b)] for a, b in ranges]
    criterion = config.get("stage2_window_partition")
    if criterion is not None:
        out["stage2_window_partition"] = str(criterion)
    n_windows = config.get("stage2_n_windows")
    if n_windows is not None:
        out["stage2_n_windows"] = int(n_windows)
    # 🔑 [2026-09-15] 耦合端（win0）专用的态数上限。理由与代价见 abfe_config.json
    # 里同名键的 `_comment`；它进 `_PREOPT_DERIVED_PATH_KEYS`，改动作废全部窗口缓存。
    first_cap = config.get("stage2_first_window_max_states")
    if first_cap is not None:
        out["stage2_first_window_max_states"] = int(first_cap)
    # 🔑 [2026-09-18] 末窗上界（溢出槽）。与 `stage2_window_max_states`（非末窗）
    # 是**两个数**：末窗是 model B 插 λ 的溢出落点，天然更大；共用一个数的话，
    # 要给末窗留空间就得把每个非末窗一起抬上去。
    last_cap = config.get("stage2_last_window_max_states")
    if last_cap is not None:
        out["stage2_last_window_max_states"] = int(last_cap)
    # 🔑 [2026-09-15] 固定节奏重锚 f_k 的节奏（步）。缺省 500k = 1 ns @2fs。
    # **不进指纹**（它只改控制器什么时候重锚，不改布局/协议）。
    cadence = config.get("stage2_f_k_reanchor_cadence_steps")
    if cadence is not None:
        out["stage2_f_k_reanchor_cadence_steps"] = int(cadence)
    return out


def _frozen_residual_manifest_covers_ligand(manifest_path, topology, ligand_indices, system) -> bool:
    """冻结 manifest 声明的配体是不是当前这个。

    判据用 loader 自己那把尺子（`ligand_chemical_identity`：配体局部原子序数序列 +
    内部键图的 canonical-JSON SHA-256，跟蛋白无关），所以这里判 True 就等于
    `build_outer_lambda_local_residual_runtime` 的身份闸门会放行。
    判 False 不是错误 —— 是"这个配体要先重训"的信号。
    """

    from local_residual.openmm_plugin import (
        _load_resource_manifest,
        ligand_chemical_identity,
    )

    try:
        manifest, _payload, _weights = _load_resource_manifest(manifest_path)
    except Exception:  # noqa: BLE001 - 读不动/不存在都按"没覆盖"处理，交给重训
        return False
    expected = manifest.get("supported_ligand") or {}
    ligand_ids = tuple(int(v) for v in ligand_indices)
    if len(ligand_ids) != int(expected.get("atom_count", -1)):
        return False
    identity = ligand_chemical_identity(topology, ligand_ids, system=system)
    return (
        identity["atomic_numbers"] == expected.get("atomic_numbers")
        and identity["internal_bonds"] == expected.get("internal_bonds")
    )


def _refit_outer_lambda_residual_for_this_ligand(
    *, pipeline, config, output_dir, topology, system, ligand_indices, log,
):
    """换配体时的一次闭式重训（EXP-033 §5 P1），产出一份 run 内 manifest。

    帧源写死 `<output_dir>/pre_equilibration.dcd` —— 名字是固定的，有多少帧用多少。
    调用点必须在**基线预平衡之后**：那条轨迹到那时才存在。

    为什么可以用 run 内的帧：`B_φ` 在 outer，**不进哈密顿量**（physical target 永远
    不含残差），所以换哪个系综的帧都不改变 ΔG 的正确性，只改变这个采样增强项好不好用。
    EXP-033 P1 item 1 原文写"用那次 run 自己的 stage2 产物"，是因为当时有 rerun1/2/3
    三条平行实验、LORO 能按独立 run 分折；而新配体在跑 stage 2 之前根本没有那些产物。

    ⚠️ 做 A/B 时不要走这条路：两臂各自重训会让 `sampling_score_sha256` 变成
    run-dependent，两臂不再共用同一把尺子。A/B 要的是先离线冻一份、两臂共用
    （`tools/retrain_local_residual_offline.py`），并按 EXP-027 U3 口径各自标定 f_k。
    """

    import numpy as np

    from local_residual.refit import closed_form_refit, write_refit_payload

    trajectory = os.path.join(output_dir, "pre_equilibration.dcd")
    if not os.path.exists(trajectory):
        raise RuntimeError(
            f"闭式重训要用预平衡轨迹，但找不到 {trajectory}。"
            "这条重训必须挂在基线预平衡之后 —— 挂错时序就会撞到这里。"
        )
    work_dir = os.path.join(output_dir, "outer_lambda_refit")
    lam_max = float(config.get("outer_lambda_refit_lambda_max", 1.0))
    lam_min = float(config.get("outer_lambda_refit_lambda_min", 0.5))
    n_states = int(config.get("outer_lambda_refit_n_states", 8))
    log.info(
        "%s：当前配体不在冻结 manifest 的覆盖范围内 ⟹ 走 EXP-033 §5 P1 闭式重训"
        "（帧源 pre_equilibration.dcd，λ_vdw %.3g→%.3g，%d 态）",
        OUTER_LAMBDA_RESIDUAL_FEATURE_NAME, lam_max, lam_min, n_states,
    )
    result = closed_form_refit(
        system=system,
        topology=topology,
        positions=pipeline.positions,
        box_vectors=pipeline.box_vectors,
        ligand_indices=ligand_indices,
        trajectory_path=trajectory,
        lambdas_vdw=np.linspace(lam_max, lam_min, n_states).tolist(),
        temperature_kelvin=float(config.temperature),
        platform_name=config.platform,
        output_dir=work_dir,
        ligand_name=str(config.ligand),
        topology_cif=os.path.join(output_dir, "topology.cif"),
        ligand_indices_path=os.path.join(output_dir, "ligand_indices.json"),
        system_xml=os.path.join(output_dir, "system_native.xml"),
        log=lambda message: log.info("%s", message),
    )
    resources = os.path.join(work_dir, "resources")
    write_refit_payload(
        result,
        output_dir=resources,
        ligand_topology_indices=sorted(int(v) for v in ligand_indices),
        protocol_sha256=result["protocol_sha256"],
        log=lambda message: log.info("%s", message),
    )
    manifest_path = os.path.join(resources, "manifest.json")
    # 顶层名是 `abfe_scripts` 不是 `scripts`：`mace_torch` 装了个同名顶层正规包，
    # 正规包永远赢过 PEP 420 namespace 目录 ⟹ 旧写法在任何装了 MACE 的环境里必然
    # ModuleNotFoundError（2026-09-12 实测）。改名后这里就是一句普通 import，
    # 也让 tests/test_fresh_clone_imports.py 的 AST 门重新看得见这个依赖。
    from abfe_scripts.write_local_residual_resource_manifest import main as write_manifest

    write_manifest([
        "--payload", os.path.join(resources, "r1_model_payload_v1.json"),
        "--weights", os.path.join(resources, "r1_model_weights_f64.bin"),
        "--topology", os.path.join(output_dir, "topology.cif"),
        "--ligand-indices", os.path.join(output_dir, "ligand_indices.json"),
        "--system", os.path.join(output_dir, "system_native.xml"),
        "--ligand-name", str(config.ligand),
        "--experiment-id", "EXP-033",
        "--output", manifest_path,
    ])
    log.info(
        "%s：闭式重训完成 | 相对改善 %.4f | tanh 改形 %.2f%% | manifest=%s"
        " ⟹ 这是「值得上机的信号」，不是验收（验收口径见 EXP-033 §4）",
        OUTER_LAMBDA_RESIDUAL_FEATURE_NAME,
        result["fit_report"]["relative_improvement"],
        100.0 * result["tanh_saturated_fraction"],
        manifest_path,
    )
    return manifest_path


_LEG_STATUS_KEYS = (
    "results_untrusted",
    "results_untrusted_stages",
    "precision_status",
    "stage_quality_failures",
    "publishable_as_accepted_result",
    "publishable_rejection_reason",
    # [2026-09-18] 未闭合的 charge-transfer 禁报标记（中性配体这三个键根本不存在
    # ⟹ `if k in leg` 过滤掉，既有产物逐字节不变）。
    "must_not_report_delta_g_bind",
    "must_not_report_delta_g_bind_reason",
    "production_qualified",
    "closes_thermodynamic_cycle",
)

# 汇总文件顶层要复刻的禁报键。少一个都会让"只读 final_binding_results.json"的人
# 看不出这份 ΔG_bind 不得引用 —— 这正是 2026-09-16 那次修的同一类漏。
_MUST_NOT_REPORT_KEYS = (
    "must_not_report_delta_g_bind",
    "must_not_report_delta_g_bind_reason",
    "production_qualified",
    "closes_thermodynamic_cycle",
)


def _leg_sampling_drift(leg_results) -> Dict[str, Any]:
    """这条腿 stage-2 的半程漂移与**未采用**的 σ 下界，压成一小块摘要。

    两个量都只是把 `stage_diagnostics.stage2` 里已经算好的数搬出来，不重算、
    不改判：`sigma_inflation_applied` 原样带出，默认仍是 `False`（P1-19 的决定 ——
    要不要采用这个下界是使用者的事）。搬出来的唯一理由是**汇总文件里看得见**。
    """
    s2 = ((leg_results or {}).get("stage_diagnostics") or {}).get("stage2") or {}
    drift = s2.get("split_half_diagnostics") or {}
    infl = s2.get("sigma_inflation_from_split_half") or {}
    if not drift.get("available") and not infl.get("available"):
        return {"available": False}
    return {
        "available": True,
        "total_drift_kJ_mol": drift.get("total_drift_kJ_mol"),
        "total_drift_over_2sigma": drift.get("total_drift_over_2sigma"),
        "max_window_drift_over_2sigma": drift.get("max_window_drift_over_2sigma"),
        "total_error_mbar_kJ_mol": infl.get("total_error_mbar_kJ_mol"),
        "total_error_inflated_kJ_mol": infl.get("total_error_inflated_kJ_mol"),
        "sigma_inflation_applied": bool(s2.get("sigma_inflation_applied")),
        "note": (
            "报出的 ± 是 MBAR 内部标准误差。总漂移与逐窗漂移都超过 2σ 时它低估了"
            "实际抽样波动；`total_error_inflated_kJ_mol` 是按 σ≥|逐窗漂移|/2 定下界"
            "的结果，**默认未采用**（`sigma_inflation_applied`），要采用见 "
            "`inflate_sigma_from_split_half`。⚠️ 逐窗漂移大而总量稳 ⟹ 多半是窗口间"
            "归属重排，那个下界会高估。"
        ),
    }


def _binding_result_status(complex_results, solv_results) -> Dict[str, Any]:
    """把两条腿的**质量状态**并成一块，跟着 ΔG_bind 一起落盘。

    🔑🔑 [2026-09-16] 先前这四个键只写在**逐腿**的 `final_results.json` 里：
    `abfe_pipeline` 判门失败会写 `results_untrusted=True` + `stage_quality_failures`
    并**刻意不 raise**（为了拿到溶剂腿），但 `final_binding_results.json` 一个都没带，
    收尾又无条件打印「计算完成」⟹ 只读最终 ΔG / ± / 时间戳的汇总看不见任何异常。
    三个真机 run（brd4_ligand2 rep1/rep3、cyclod_ligand2 rep2）全部
    `results_untrusted=true`，而汇总里读不出来。

    这里**不改判**任何一条腿的结论，只是如实合并：
      · `results_untrusted`  —— 任一腿不可信 ⟹ 合成结果不可信（或运算）
      · `publishable_as_accepted_result` —— 两腿都可发布才可发布（与运算）
      · `precision_status`   —— 有任何一腿 `UNMEASURED` ⟹ 合成也是"没测"
    """
    legs = {"complex": complex_results or {}, "solvent": solv_results or {}}
    per_leg = {
        name: {k: leg.get(k) for k in _LEG_STATUS_KEYS if k in leg}
        for name, leg in legs.items()
    }
    for name, leg in legs.items():
        per_leg[name]["sampling_drift"] = _leg_sampling_drift(leg)
    untrusted = [n for n, leg in legs.items() if bool(leg.get("results_untrusted"))]
    # 缺键 = 这条腿没表态。**不当成通过**：`publishable` 只在两腿都显式 True 时为 True。
    publishable = all(
        legs[n].get("publishable_as_accepted_result") is True for n in legs
    )
    _prec = [str(legs[n].get("precision_status") or "UNMEASURED") for n in legs]
    precision = "UNMEASURED" if "UNMEASURED" in _prec else (
        _prec[0] if len(set(_prec)) == 1 else "MIXED:" + "/".join(_prec)
    )
    failures = [
        dict(f, leg=n) for n in legs
        for f in (legs[n].get("stage_quality_failures") or [])
        if isinstance(f, dict)
    ]
    return {
        "results_untrusted": bool(untrusted),
        "results_untrusted_legs": untrusted,
        "precision_status": precision,
        "publishable_as_accepted_result": bool(publishable),
        "publishable_rejection_reason": (
            None if publishable else "；".join(
                filter(None, [
                    (f"{n} 腿：{legs[n].get('publishable_rejection_reason')}"
                     if legs[n].get("publishable_rejection_reason") else None)
                    for n in legs
                ])
            ) or "至少一条腿没有显式声明 publishable_as_accepted_result=True"
        ),
        "stage_quality_failures": failures,
        "per_leg": per_leg,
        # [2026-09-18] 任一腿声明禁报 ⟹ 合成结果禁报（两条腿共用同一份电荷协议，
        # 所以这里是"原样上浮"而不是重新判一次）。中性配体两腿都没有这些键 ⟹
        # 这块是空 dict，既有 final_binding_results.json 逐字节不变。
        **_merged_must_not_report(legs),
        "note": (
            "本块是两条腿自报状态的合并，不改判任何一条腿的结论。"
            "`results_untrusted=true` 的 ΔG_bind **不得**作为可发布结果引用。"
        ),
    }


def _merged_must_not_report(legs) -> Dict[str, Any]:
    """两条腿的禁报标记上浮到汇总层。没有腿声明 ⟹ 返回 `{}`。"""
    declaring = [leg for leg in legs.values()
                 if leg.get("must_not_report_delta_g_bind") is True]
    if not declaring:
        return {}
    return {k: declaring[0][k] for k in _MUST_NOT_REPORT_KEYS if k in declaring[0]}


def main():
    _diagnostic_exit = dispatch_diagnostic_command(sys.argv[1:])
    if _diagnostic_exit is not None:
        sys.exit(_diagnostic_exit)

    args = parse_arguments()

    if args.command == "self-test":
        sys.exit(run_self_tests())

    if args.command == "prepare":
        run_prepare_command(args)
        return

    if args.command == "refine-lambda-path":
        from abfe_preoptimizer import refine_stage_lambda_path_from_data
        result = refine_stage_lambda_path_from_data(
            stage_dir=args.stage_dir,
            preopt_path=args.preopt_file,
            temperature_k=args.temperature,
            n_states=args.n_states,
            max_window_span_kJ=args.max_window_span_kj,
            overlap=args.overlap,
            stage_type=args.stage_type,
        )
        log.info(
            "[OK] 已按累积|Δf|重新设计 λ 路径：%d 个状态，%d 个窗口（旧文件备份为 %s.bak）",
            result["n_states"], len(result["window_ranges"]), args.preopt_file,
        )
        log.info("下次 --resume 时，形状不匹配的窗口会被自动判定为需要重新采样。")
        return

    # 创建配置对象
    config = RunConfig(args)

    # 🔑 [P1-04] analyze-only 必须最先分流：此前它在加载 DEXP 参数、要求
    # --ligand 等模拟输入**之后**才分流，纯分析现有结果会被与模拟无关的
    # 前置校验提前阻断（比如 --potential=dexp 但没给 dexp 参数文件的运行
    # 根本进不了 analyze-only）。
    if args.analyze_only:
        run_post_analysis(config)
        return

    # 🔑 [RELEASE_READINESS R1] 唯一 resolved seed。以前这里只读环境变量，而
    # traditional 路径读 `config.get("repeat_seed")`、RunConfig 又会保留配置文件里的
    # `repeat_seed` —— 用户在 JSON 里写了 seed，配置快照记着它，IBS 管线却根本没收到。
    # 现在集中解析一次并回写 config.data，两种模式与配置快照从同一个值出发。
    _repeat_seed, _repeat_seed_source = _resolve_repeat_seed(args, config)
    if _repeat_seed is not None:
        # Include the resolved repeat seed in the immutable runtime snapshot;
        # the backend's derived map is added after each leg completes.
        config.data["repeat_seed"] = _repeat_seed
        config.data["repeat_seed_source"] = _repeat_seed_source
        log.info(
            "resolved repeat_seed=%d（来源：%s）；两腿/预平衡/attachment/采样统一使用该值",
            _repeat_seed,
            _repeat_seed_source,
        )
    else:
        # 未显式设置 —— 保留历史行为（各处沿用自己的默认常量），不偷补新 seed。
        log.info(
            "未显式设置 repeat_seed（--seed / ABFE_RANDOM_SEED / config 均未给）；"
            "沿用历史默认随机流，不注入新 seed"
        )
    dexp_params = _load_dexp_params_fail_closed(
        config.potential,
        config.dexp_params,
    )
    # 🔑 [PLAN_openmm_8_6_remd_backend.md P0] 后端只在这里解析一次（§3 实施要求 6），
    # 位置刻意选在建立任何 Context / 写任何轨迹之前：显式 --remd-backend openmm 而
    # 环境或适配器不合格时必须在烧 GPU 之前 SystemExit，不能跑到一半才发现。
    # 采样开始后不得再重新解析或换引擎。
    try:
        _remd_resolution = free_energy_engine.resolve_remd_backend(
            config.get("remd_backend")
        )
    except free_energy_engine.UnsupportedRemdBackendError as exc:
        raise SystemExit(str(exc)) from exc
    except ValueError as exc:
        raise SystemExit(f"--remd-backend 取值非法：{exc}") from exc
    log.info(_remd_resolution.format_log())
    # 存进 config.data 供 provenance 落盘；下划线前缀表示它是解析产物而非用户配置。
    config.data["_remd_backend_resolution"] = _remd_resolution.to_provenance()
    if not config.ligand:
        log.error("未提供配体残基名称。请通过 --ligand 或配置文件中的 ligand 指定。")
        sys.exit(2)
    if config.only_complex_charging and config.reset:
        raise RuntimeError(
            "--only-complex-charging 不能与 --reset 同用；拒绝在读取冻结结果前"
            "重建或覆盖主输出缓存"
        )
    if config.only_complex_charging and not config.resume:
        raise RuntimeError(
            "--only-complex-charging 必须与 --resume 同用，以只读加载现有 committed "
            "Boresch、Stage 2 和 solvent 结果"
        )
    if config.only_boresch_attachment and config.only_complex_charging:
        raise RuntimeError(
            "--only-boresch-attachment 与 --only-complex-charging 互斥"
        )
    if config.only_boresch_attachment and config.reset:
        raise RuntimeError(
            "--only-boresch-attachment 不能与 --reset 同用；拒绝在读取冻结结果前"
            "重建或覆盖主输出缓存"
        )
    if config.only_boresch_attachment and not config.resume:
        raise RuntimeError(
            "--only-boresch-attachment 必须与 --resume 同用，以只读加载现有 committed "
            "Boresch、stage1/stage2 和 solvent 结果"
        )
    if config.only_ligand_in_water and (
        config.only_complex_charging or config.only_boresch_attachment
    ):
        raise RuntimeError(
            "--only-ligand-in-water 与 --only-complex-charging / --only-boresch-attachment 互斥："
            "三者各自只跑一条腿，同时给两个说明调用方没想清楚要哪条"
        )
    if config.only_ligand_in_water and config.outer_lambda_local_residual_ibs:
        raise RuntimeError(
            f"--only-ligand-in-water 暂不支持 {OUTER_LAMBDA_RESIDUAL_FEATURE_NAME}："
            "残差运行时是在复合物腿那一段构造出来、再按溶剂腿拓扑重建的，"
            "跳过复合物腿就拿不到它。fail closed，不静默跑成 baseline 溶剂腿"
        )
    if config.outer_lambda_local_residual_ibs and (
        config.mode != "ibs" or config.decoupling != "dual_lambda"
    ):
        raise RuntimeError(
            f"{OUTER_LAMBDA_RESIDUAL_FEATURE_NAME} 只支持 IBS + dual_lambda 的 Stage-2 VDW；"
            "当前组合已 fail closed"
        )
    if config.outer_lambda_local_residual_ibs and (
        config.only_complex_charging or config.only_boresch_attachment
    ):
        raise RuntimeError(
            f"{OUTER_LAMBDA_RESIDUAL_FEATURE_NAME} 不能用于只跑 charging/attachment 的入口；"
            "该开关要求完整 dual_lambda 两腿流程"
        )
    # 准备输出目录
    output_dir = config.output
    os.makedirs(output_dir, exist_ok=True)

    # 🔑 [ATT-23 / issue #142] 作业级防护：**SIGTERM/SIGINT 有序退出**。
    #   默认行为是直接终止，`finally` 一个都不跑：Context 不销毁、reporter 不收尾、
    #   日志里连"我被信号杀了"这句都没有。2026-09-09 那次作业就是这样消失的
    #   （两个 log 停在同一行、无 traceback）。
    #
    # 这里**不再**取输出目录独占锁（见 abfe_pipeline.guard_run_directory 的
    # docstring：锁文件在共享盘上判不出 stale，反而挡住 --resume）。代价是
    # "两个 pipeline 写同一个 --output"不再有任何东西拦得住 —— 这条要靠调度侧
    # 保证，别指望代码。
    # ⚠️ SIGKILL（含 OOM killer）捕不到，那条路要靠内存预算本身，不是这里。
    #
    # 🔑 [2026-09-10] 装在 traditional 分流**之前**。原来它排在
    # `if config.mode == "traditional": ...; return` 后面，于是 traditional
    # 整条腿一个信号处理器都没装 —— 正是这条改动要修的那个"作业悄无声息消失"。
    guard_run_directory(output_dir, log=log.info)

    # 分析模式已在 main 开头分流（[P1-04]：不被模拟输入校验阻断）
    # 传统模式单独处理
    if config.mode == "traditional":
        run_traditional_mode(config)
        return

    # ----- 1. 系统加载：优先从缓存，否则 GROMACS 构建并立即落盘 -----
    #
    # 环境类型必须在**加载 System 之前**就知道：膜体系要求 `load_native_system`
    # 返回**带键**的拓扑（mmCIF 不保存脂质/配体的键，而 PBC 分子完整性修复靠键）。
    # 这里只用 config 判环境类型——完整的膜协议解析（要 topology 做脂质交叉检查）
    # 仍在下面 `resolve_membrane_protocol` 那一处，两者取值一致由下方断言保证。
    _environment_type = resolve_environment_type(config.get("system_type"))
    _require_bonded_topology = _environment_type == "membrane"

    # [CACHE-01, 2026-09-02] `find_gmx_include_dir` 只在真的要解析 GROMACS
    # 依赖树时才调用。以前它在这里无条件执行，于是 `--openmm-cache-only`
    # 这一路（根本不读 include 树）也照样打「找不到 GROMACS 力场 include
    # 目录」警告。cache-only 下 `include_dir` 一次都不会被解引用——三条使用
    # 路径逐个核对过，见 docs/TROUBLESHOOTING.md 同名小节——所以这里留 None。
    include_dir: Optional[str] = None
    _cache_only_audits = {}
    if args.openmm_cache_only:
        if _require_bonded_topology:
            raise RuntimeError(
                "--openmm-cache-only does not support membrane systems because topology.cif "
                "does not preserve the bonded topology required for PBC repair"
            )
        _cache_only_audits["complex"] = validate_openmm_cache_only(
            output_dir, phase="complex"
        )
        main_cache_identity = {
            "identity": _cache_only_audits["complex"],
            "identity_sha256": _cache_only_audits["complex"]["audit_sha256"],
        }
        config.data["openmm_cache_only"] = True
        config.data["openmm_cache_only_complex_audit_sha256"] = (
            _cache_only_audits["complex"]["audit_sha256"]
        )
        log.warning(
            "[WARN] OpenMM-cache-only：已验证迁移缓存自身哈希/形状；"
            "本机未重验 GROMACS dependency tree"
        )
    else:
        include_dir = find_gmx_include_dir(config.gmx_path)
        # 🔑 [ATT-24 / issue #64] 物理输入预检。放在这里是因为此刻 gro/top/ligand/
        # include_dir 都已确定、而 System 还没建 —— 下面这五类问题原本都要等到
        # 建完 System 甚至跑起来之后才以别的面目暴露：
        #   未解析的 #include → "某个 moleculetype 找不到"
        #   --ligand 打错     → "ligand_indices 为空"
        #   gro/top 原子数不符 → 所有按序号取的原子选择静默错位
        #   盒子 < 2×cutoff   → PME **不报错**，只是给出错误的静电
        #   配体/受体太小     → Boresch 六个锚点定义不出来
        # 一次列全所有问题再抛，不是遇到第一个就停。
        # cutoff 用 1.0 nm：与本仓生产口径一致（MEM-00h 把 evaluation clone
        # 归一化到 cutoff=1.0nm，softcore 探针 builder 的默认也是 1.0）。
        preflight_physical_inputs(
            config.gro,
            config.top,
            config.ligand,
            gmx_include_dir=include_dir,
            cutoff_nm=1.0,
            require_boresch=bool(config.get("boresch")),
            log=log.info,
        )
        main_cache_identity = _main_cache_identity(
            config.gro, config.top, config.ligand, include_dir
        )
    if args.openmm_cache_only or (not config.reset and system_cache_exists(
        output_dir,
        config.gro,
        config.top,
        config.ligand,
        include_dir,
        charge_treatment=config.get("charge_treatment"),
    )):
        system, topology, positions, box_vectors, ligand_indices = load_native_system(
            output_dir, 
            gro_file=config.gro, 
            top_file=config.top, 
            gmx_include_dir=include_dir,
            # 膜体系必须拿**带键**的拓扑：mmCIF 不保存脂质/配体的键，而
            # PBC 分子完整性修复靠键判断分子边界，键丢了会把跨边界脂质撕开。
            require_bonded_topology=_require_bonded_topology,
            reconstruct_bonds_from_system=args.openmm_cache_only,
        )
        log.info("从缓存加载 System 完成")
        # 缓存命中时不需要再转换（缓存里存的是转换后构建出来的 System）。
        # 转换版本已进主缓存指纹，所以指纹一致就代表转换口径一致。
        _pairs_conversion = None
    else:
        if not config.gro or not config.top:
            log.error("未提供 --gro/--top 且无缓存，无法构建系统")
            sys.exit(1)
        # OpenMM 的 GromacsTopFile 只接受 `[ pairs ]` funct 1；CHARMM-GUI 的
        # AMBER FF-Converter 会对部分对写 funct 2（多出 fudgeQQ/q1/q2 三列）。
        # 那三列经**逐对校验**若与全局 fudgeQQ 和 `[ atoms ]` 电荷一致，就是冗余重述，
        # 可等价转成 funct 1；不一致则 fail closed（硬转会静默改变哈密顿量）。
        # 转换写到 output_dir 下的独立目录，**原始输入一个字节都不动**。
        # 主路径显式指定转换目录，让产物落在 output_dir 里便于审计；
        # 下游各加载点即使自己再调一次，也会命中同一份（幂等 + 内容寻址）。
        _top_for_openmm, _pairs_conversion = openmm_compatible_gromacs_top(
            config.top,
            include_dir,
            compat_dir=os.path.join(output_dir, "gromacs_openmm_compat"),
        )
        if _pairs_conversion:
            log.info(
                "[ pairs ] funct 2（OpenMM 不支持）已等价转换 → %s"
                "（%s 条；原始输入未修改）",
                _top_for_openmm,
                _pairs_conversion.get("n_pairs_converted", "复用已有"),
            )

        # 从 GROMACS 构建（build_system_from_gromacs 内部也走唯一入口，这里传
        # 转换后的路径只是为了让 provenance 记录的就是实际用的那份）。
        system, topology, positions, box_vectors, ligand_indices = build_system_from_gromacs(
            config.gro, _top_for_openmm, config.ligand,
            include_dir
        )
        diagnose_14_scaling(system)
        _raw_ligand_net_charge = _compute_ligand_net_charge(system, ligand_indices)
        _rounded_ligand_charge = int(round(float(_raw_ligand_net_charge)))
        if abs(float(_raw_ligand_net_charge) - _rounded_ligand_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
            raise RuntimeError(
                f"配体净电荷 {_raw_ligand_net_charge:+.6f} e 不接近整数；"
                "拒绝在协议解析前用 round() 猜测 charge_treatment"
            )
        _resolved_builder_treatment = config.get("charge_treatment")
        if not _resolved_builder_treatment:
            _resolved_builder_treatment = (
                "neutral" if _rounded_ligand_charge == 0
                else CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
            )
        # 🔑🔑 [2026-09-16] **先把 reserved co-ion 建出来，再进数量契约校验。**
        # 契约（`co_alchemical_ion_builder_identity_payload`）要求盒子里已经有
        # |q_L| 个中性 ion-shaped dummy；溶剂腿有插入步骤，复合物腿先前没有，
        # 于是带电配体在这里当场炸。两侧现在走同一套选点规则与同样的复核。
        _coion_derivation = None
        if (
            _resolved_builder_treatment
            == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
            and _rounded_ligand_charge != 0
        ):
            (
                system, topology, positions, box_vectors, ligand_indices,
                _coion_derivation,
            ) = _ensure_complex_reserved_coions(
                system=system,
                topology=topology,
                positions=positions,
                box_vectors=box_vectors,
                ligand_indices=ligand_indices,
                gro_file=config.gro,
                top_file=_top_for_openmm,
                ligand_resname=config.ligand,
                gmx_include_dir=include_dir,
                output_dir=output_dir,
                ligand_net_charge_e=_rounded_ligand_charge,
            )
        _builder_identity = co_alchemical_ion_builder_identity_payload(
            system=system,
            topology=topology,
            charge_treatment=_resolved_builder_treatment,
            ligand_net_charge_e=_rounded_ligand_charge,
        )
        if _coion_derivation is not None:
            _builder_identity = dict(
                _builder_identity, complex_coion_derivation=_coion_derivation
            )
        # 立即保存为原生缓存
        save_native_system(
            output_dir,
            system,
            topology,
            ligand_indices,
            positions,
            box_vectors,
            cache_identity=main_cache_identity,
            reserved_coion_builder_identity=_builder_identity,
            charge_treatment=_resolved_builder_treatment,
        )
        log.info("[OK] 原生缓存已生成")
        # 重新从缓存加载，确保后续所有对象都来自落盘文件
        # 替换原有的 load_native_system 调用为：
        # ⚠️ 注意这一步：即使是**全新构建**，这里也会立刻从落盘文件重新读回，
        # 所以生产的每一次运行（首跑或缓存命中）都会拿到 `load_native_system` 的
        # 拓扑。膜体系必须要求带键拓扑，否则首跑同样会被 mmCIF 丢键坑到——
        # 这正是"直接用 build_system_from_gromacs 复现不出 NaN"的原因。
        system, topology, positions, box_vectors, ligand_indices = load_native_system(
            output_dir, 
            gro_file=config.gro, 
            top_file=config.top, 
            gmx_include_dir=include_dir,
            prefer_equilibrated=False,
            require_bonded_topology=_require_bonded_topology,
        )
        log.info("已从缓存重新加载 System (使用落盘后对象)")

    # 元素覆盖是**输入拓扑的性质**，不需要跑任何 MD 就能判，所以放在拿到拓扑的
    # 第一时间。默认不做（配置键为 None 时整段跳过，不 import torch）。
    _coverage_model = getattr(config, "element_coverage_model", None)
    if _coverage_model:
        from local_residual.element_coverage import check_topology_element_coverage

        _elements, _table = check_topology_element_coverage(
            topology, _coverage_model, system=system
        )
        log.info(
            "元素覆盖检查通过 | 模型=%s（%d 种元素）| 体系 %d 种 %s",
            _coverage_model, len(_table), len(_elements), list(_elements),
        )

    # 坐标安全处理：转换为纯 numpy 再转为 Vec3 列表（防止类型问题）
    if hasattr(positions, "value_in_unit"):
        pos_nm = positions.value_in_unit(unit.nanometer)
    else:
        pos_nm = positions
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    if pos_nm.ndim == 1:
        pos_nm = pos_nm.reshape(-1, 3)
    positions = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_nm] * unit.nanometer
    # ========== PBC 确定性居中 ==========
    positions, box_vectors = center_system_rigidly(
        positions, box_vectors, ligand_indices
    )
    # 🔑 [P1-14] 这里**只是**整体质心平移。原来这行写的是"分子完整性修复完毕"，
    # 是假的：center_system_rigidly 的 docstring 自己就写着"彻底禁用原子级 PBC
    # Wrap"，它不碰拓扑、修不了任何跨盒断裂的分子。真正的整分子周期平移是
    # ABFEPipeline.repair_pbc_molecule_integrity()，现已提前到第一次建 Context /
    # 最小化 / 预平衡之前执行（pre_equilibrate 开头），并在失败时 fail closed。
    log.info("  [OK] 配体已居中（仅整体质心平移；整分子 PBC 修复在 pre_equilibrate 之前执行）")
    # 膜协议在这里先解析一次：provenance 写在 Pipeline 构建之前，而 §6.1 要求
    # 协议组合的 fail-closed 检查在创建任何 Context 之前完成。下面 ABFEPipeline
    # 会用同一个纯函数、同一份输入再解析一次，结果必然一致（单一实现，无第二套判据）。
    _barostat_protocol = resolve_membrane_protocol(
        config.get("system_type"),
        membrane_config=config.get("membrane"),
        topology=topology,
        confirm_soluble_with_lipids=bool(
            config.get("confirm_soluble_with_lipids", False)
        ),
    )
    # 两处环境类型解析必须一致（前者只看 config，后者还做脂质交叉检查）。
    # 不一致意味着有人给其中一条加了额外来源，那会让"要不要带键拓扑"与
    # "用哪种 barostat"基于不同的判断，属静默不一致。
    if _barostat_protocol["system_type"] != _environment_type:
        raise SystemExit(
            f"环境类型解析不一致：加载 System 前判为 {_environment_type!r}，"
            f"完整协议解析判为 {_barostat_protocol['system_type']!r}。"
            "两者必须同源（都来自 config 的 system_type）。"
        )

    if _barostat_protocol["system_type"] != "soluble":
        log.info(
            "环境类型=%s，复合物腿将使用 %s",
            _barostat_protocol["system_type"],
            _barostat_protocol["barostat_class"],
        )

    # 净电荷处理协议（memtodolist §1.2，B2）。同样在建任何 Context 之前判死，
    # 且**自动计算配体净电荷并与配置交叉核对**（§6.1）。
    # 净电荷用 ibs_engine 里既有的唯一实现算，不另造一套判据。
    _ligand_net_charge_e = _compute_ligand_net_charge(system, ligand_indices)
    _reservoir_correction_input = config.get(
        "charge_transfer_reservoir_correction"
    )
    if isinstance(_reservoir_correction_input, (str, os.PathLike)):
        _reservoir_correction_input = _load_json_object_file(
            str(_reservoir_correction_input),
            "charge-transfer reservoir correction",
        )
    _charge_protocol = resolve_charge_treatment(
        config.get("charge_treatment"),
        ligand_net_charge_e=_ligand_net_charge_e,
        apbs_correction_kJ_mol=config.get("apbs_correction_kJ_mol", 0.0),
        environment_type=config.get("system_type"),
        apbs_evidence=(
            _load_json_object_file(config.get("apbs_evidence"), "APBS 证据")
            if config.get("apbs_evidence")
            else None
        ),
        charge_transfer_reservoir_correction=_reservoir_correction_input,
    )
    if config.get("co_alchemical_ion"):
        log.warning(
            "[WARN] --co-alchemical-ion / co_alchemical_ion 是旧版全局输入，"
            "仅作 deprecated audit record；B5 不会把它用于任一条腿。"
            "请让 complex/solvent pipeline 各自冻结 coalchemical_ion_spec.json。"
        )
    if _charge_protocol["charge_treatment"] != "neutral":
        log.info(
            "净电荷协议=%s（配体净电荷 %+d e，APBS %s）",
            _charge_protocol["charge_treatment"],
            _charge_protocol["ligand_net_charge_e"],
            "适用" if _charge_protocol["apbs_applicable"] else "不适用",
        )
        if _charge_protocol["experimental_not_for_production"]:
            log.warning(
                "[WARN] charge_treatment=co_annihilation_experimental 是**实验对照专用**，"
                "其数值不得进入任何 ΔG_bind 汇总（memtodolist MEM-00a-2）。"
            )
        # The charging builder exists, but the tethered carrier's
        # reservoir/standard-state correction has not been validated.
        # Refuse before creating any Context; a finite two-leg number would
        # otherwise be mistaken for a closed ABFE result.
        _ct_err = charge_transfer_admission_error(
            _charge_protocol,
            diagnostics_mode=bool(
                config.get("run_unclosed_charge_transfer_diagnostics", False)
            ),
        )
        if _ct_err:
            raise RuntimeError(_ct_err)
        if (
            _charge_protocol["charge_treatment"]
            == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
            and _charge_protocol.get("closes_thermodynamic_cycle") is not True
        ):
            log.warning(
                "[WARN] --run-unclosed-charge-transfer-diagnostics 已放行一条**未验收**的 "
                "charge-transfer 运行：carrier reservoir/C4-C5 closure 未验证 ⟹ "
                "本跑的 ΔG_bind **不得发布**（provenance 已带 "
                "closes_thermodynamic_cycle=False / must_not_report_delta_g_bind=True）。"
            )

    # 力场族与 LJ/色散路线（memtodolist §1.1 / §1.3，B6）。同样在建 Context 之前判死。
    # 力场族识别只在**膜体系或用户显式声明了色散路线/力场族**时才强制——可溶体系
    # 走 legacy_uniform_density_lrc，与识别结果无关，不能因为识别不出就挡住现有运行。
    _forcefield_family_info = None
    _needs_forcefield_family = (
        _barostat_protocol["system_type"] != "soluble"
        or config.get("dispersion_protocol")
        or config.get("forcefield_family")
    )
    if _needs_forcefield_family:
        _forcefield_family_info = resolve_forcefield_family(
            top_path=config.get("top"),
            explicit_family=config.get("forcefield_family"),
            gmx_include_dir=include_dir,
        )
        log.info(
            "力场族=%s（来源 %s）",
            _forcefield_family_info["family"],
            _forcefield_family_info["source"],
        )
    _dispersion_protocol = resolve_dispersion_protocol(
        config.get("dispersion_protocol"),
        environment_type=config.get("system_type"),
        forcefield_family=(
            _forcefield_family_info["family"] if _forcefield_family_info else None
        ),
        force_switch_deviation_evidence=(
            _load_json_object_file(
                config.get("force_switch_deviation_evidence"),
                "force-switch 偏差论证",
            )
            if config.get("force_switch_deviation_evidence")
            else None
        ),
    )
    if not _dispersion_protocol["was_defaulted"]:
        log.info(
            "色散路线=%s（均匀密度 LRC %s）",
            _dispersion_protocol["dispersion_protocol"],
            "生效" if _dispersion_protocol["uniform_density_lrc_active"] else "关闭",
        )

    # ---- §3.3 膜输入核对 + §9/§15 预平衡时长门（只在膜体系生效）----
    _membrane_input_report = None
    if _barostat_protocol["system_type"] == "membrane":
        _declaration_path = config.get("membrane_input_declaration")
        if not _declaration_path:
            raise SystemExit(
                "system_type=membrane 需要 --membrane-input-declaration：§3.3 要求输入是"
                "已完成膜构建和主要平衡的体系，且构建工具/参数/最终平衡作业/来源结构"
                "可追溯。缺这份声明就无法核对，拒绝继续。"
            )
        _membrane_declared = _load_json_object_file(_declaration_path, "膜输入声明")

        # 身份一律以 `.top` 的 [ molecules ] + [ moleculetype ] 为准，不靠残基名猜。
        # 实测理由：CHARMM-GUI 的 AMBER 体系里一个 POPC 是三个残基（PA+PC+OL）、
        # 水叫 TP3、离子叫 Na+/Cl-，靠残基名会把脂质数数成 3 倍、水和离子数成 0。
        _system_composition = classify_system_composition(
            parse_gromacs_topology(
                config.get("top"), find_gmx_include_dir(config.get("gmx_path"))
            ),
            ligand_molecule_name=_membrane_declared.get("ligand_molecule_name"),
            declared_roles=_membrane_declared.get("molecule_roles"),
        )
        log.info(
            "体系组成（来自 .top）：%s",
            {
                name: f"{role}×{_system_composition['molecule_counts'][name]}"
                for name, role in _system_composition["roles"].items()
            },
        )
        if _system_composition["n_atoms_total"] != topology.getNumAtoms():
            raise SystemExit(
                f".top 展开得到 {_system_composition['n_atoms_total']} 个原子，"
                f"但拓扑有 {topology.getNumAtoms()} 个。两者必须一致，否则原子索引"
                "对不上，后续所有按索引的选择都会静默错位。"
            )

        _membrane_input_report = validate_membrane_input(
            topology,
            positions,
            box_vectors,
            declared=_membrane_declared,
            normal_axis=(_barostat_protocol["membrane"] or {}).get("normal_axis", "z"),
            composition=_system_composition,
        )
        log.info(
            "膜输入核对通过：%d 原子，上叶 %d / 下叶 %d 脂质（不平衡 %.1f%%），"
            "水 %d，离子 %s",
            _membrane_input_report["n_atoms"],
            _membrane_input_report["leaflets"]["n_upper"],
            _membrane_input_report["leaflets"]["n_lower"],
            100.0 * _membrane_input_report["leaflets"]["imbalance_fraction"],
            _membrane_input_report["n_water"],
            _membrane_input_report["ion_counts"],
        )

        # §9/§15：把"拿可溶体系的 10 ns 默认值去跑膜"挡在配置阶段，而不是等质量门
        # 事后否掉——后者要先烧掉整轮采样才知道。
        _equil_steps = int(config.get("n_equil_steps", 5_000_000))
        # 🔑 [2026-08-31 P1] `timestep_ps` 只被**读**、从不接到积分器：预平衡的真实
        # 步长是常量 `abfe_core.PRE_EQUILIBRATION_TIMESTEP_PS`（见
        # abfe_pipeline.py 里建 LangevinMiddleIntegrator 的那处）。而这里算出的
        # `_equil_ns` 是 §9/§15 膜预平衡硬门（下面那个 SystemExit）的输入。
        # 于是配 `timestep_ps: 0.004` 会让标称平衡时长翻倍、"≥100 ns" 硬门在真实
        # 只跑了一半的情况下静默通过 —— 硬门被一个不生效的配置键绕过。
        # `abfe_core.py` 里那个同名形参只用于帧间隔换算，不是积分步长，反而强化了
        # "这个键可配置"的错觉。这里显式 fail-closed：声明值与真实常量不一致就拒跑。
        _declared_timestep_ps = config.get("timestep_ps")
        if _declared_timestep_ps is not None and abs(
            float(_declared_timestep_ps) - float(PRE_EQUILIBRATION_TIMESTEP_PS)
        ) > 1e-12:
            raise SystemExit(
                f"配置声明 timestep_ps={float(_declared_timestep_ps)} ps，但预平衡真实"
                f"步长是常量 PRE_EQUILIBRATION_TIMESTEP_PS={float(PRE_EQUILIBRATION_TIMESTEP_PS)} ps"
                "（该键从不接到积分器）。若按声明值折算，§9/§15 的膜预平衡时长硬门会被"
                "一个不生效的键绕过。请删掉该键，或改 n_equil_steps 来真正延长平衡。"
            )
        _timestep_ps = float(PRE_EQUILIBRATION_TIMESTEP_PS)
        _equil_ns = _equil_steps * _timestep_ps / 1000.0
        # 上游平衡的两条合法表述已由 validate_membrane_input 裁决过，这里只用结论。
        _upstream_ns = _membrane_input_report["upstream_equilibration_ns"]
        _nominal_precheck = _membrane_input_report[
            "nominal_equilibration_precheck_applicable"
        ]
        _shortfall_note = _membrane_declared.get("equilibration_shortfall_justification")
        if not _nominal_precheck:
            # 上游生产已完成但时长不可考 → 标称时长预检不适用。
            # §9 的实测质量门（末段漂移 + 脂质横向弛豫时间尺度）仍会独立判定。
            _total_equil_ns = None
            log.info(
                "上游平衡：%s（证据 %s）→ 标称时长预检不适用；"
                "本流程再平衡 %.1f ns。§9 实测质量门仍为硬门。",
                _membrane_input_report["upstream_equilibration_status"],
                _membrane_declared.get("final_equilibration_job"),
                _equil_ns,
            )
        else:
            _total_equil_ns = float(_upstream_ns) + _equil_ns
            log.info(
                "平衡时长：上游 %.1f ns + 本流程 %.1f ns = %.1f ns（下限 %.0f ns）",
                _upstream_ns, _equil_ns, _total_equil_ns, MEMBRANE_MIN_EQUILIBRATION_NS,
            )
        if _total_equil_ns is not None and _total_equil_ns < MEMBRANE_MIN_EQUILIBRATION_NS:
            if not _shortfall_note:
                raise SystemExit(
                    f"膜体系总平衡时长 {_total_equil_ns:.1f} ns"
                    f"（上游声明 {_upstream_ns:.1f} + 本流程 {_equil_ns:.1f}）"
                    f"低于下限 {MEMBRANE_MIN_EQUILIBRATION_NS:.0f} ns。"
                    "§9：通用 10 ns 不是膜平衡充分性的证明；§15：膜体系预平衡 ≥ 100 ns。"
                    "正解是调大 n_equil_steps 或延长上游平衡。"
                    "确有理由放行时，在膜输入声明里填 "
                    "equilibration_shortfall_justification 写明依据（会进 provenance）——"
                    "但注意这**只**放开这道标称时长预检，"
                    "§9 基于实测漂移与脂质横向弛豫时间的质量门仍然是硬门，不受影响。"
                )
            log.warning(
                "[WARN] 膜平衡标称时长不足（%.1f < %.0f ns），已按声明的理由放行：%s。"
                "§9 的实测质量门仍会独立判定。",
                _total_equil_ns, MEMBRANE_MIN_EQUILIBRATION_NS, _shortfall_note,
            )

    outer_lambda_runtime = None
    # 🔑 [EXP-033 §5 P1] 冻结 manifest 覆盖不到当前配体时，**不是**在这里失败，
    # 而是推迟到基线预平衡之后做一次闭式重训——训练帧源 `pre_equilibration.dcd`
    # 到那时才存在。覆盖得到就走原路，一个字节都不变。
    _residual_refit_pending = False
    # 🔑 本次运行**实际**用的那份 manifest。两条腿都必须读这一个变量：重训产出的
    # manifest 只写在这里，溶剂腿若仍去读 `config.outer_lambda_resource_manifest`
    # 就会拿冻结那份去建，新配体在那里撞身份闸门 —— 而且是在复合物腿跑完之后才炸。
    # 权重**绑配体不绑体系**（docs/RETRAIN_LOCAL_RESIDUAL.md 的表：复合物腿↔溶剂腿
    # 不用重训），所以一次重训的产物两条腿共用，不重训第二次。
    _residual_resource_manifest = getattr(config, "outer_lambda_resource_manifest", None)
    if config.outer_lambda_local_residual_ibs:
        _frozen_manifest = _residual_resource_manifest or (
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "resources/outer_lambda_local_residual/manifest.json",
            )
        )
        _residual_refit_pending = not _frozen_residual_manifest_covers_ligand(
            _frozen_manifest, topology, ligand_indices, system
        )
    if config.outer_lambda_local_residual_ibs and not _residual_refit_pending:
        # This is the sole mainline plugin entry point.  It validates the
        # frozen source/model artifacts and loads all plugin libraries before
        # any residual XML Force can be deserialized by a window builder.
        outer_lambda_runtime = build_outer_lambda_local_residual_runtime(
            topology=topology,
            ligand_indices=ligand_indices,
            system=system,
            temperature_kelvin=float(config.temperature),
            potential_type=str(config.potential),
            output_dir=output_dir,
            ligand_indices_path=os.path.join(output_dir, "ligand_indices.json"),
            leg_name="complex",
            platform_name=config.platform,
            resource_manifest=_residual_resource_manifest,
        )
        config.data["outer_lambda_local_residual_ibs_identity"] = (
            outer_lambda_runtime.provenance_payload()
        )
        log.info(
            "%s 已加载 | score=%s | em_policy=%s",
            OUTER_LAMBDA_RESIDUAL_FEATURE_NAME,
            outer_lambda_runtime.sampling_score_sha256,
            outer_lambda_runtime.em_policy,
        )

    run_provenance = None
    coion_identities = {}
    if not config.only_complex_charging:
        run_provenance = _write_run_provenance(
            output_dir, config, system, topology, positions,
            barostat_protocol=_barostat_protocol,
            charge_protocol=_charge_protocol,
            dispersion_protocol=_dispersion_protocol,
            forcefield_family=_forcefield_family_info,
            membrane_input=_membrane_input_report,
            pairs_conversion=_pairs_conversion,
            membrane_quality_gate_mode=(
                resolve_membrane_quality_gate_mode(config.get("membrane_quality_gate"))
                if _barostat_protocol["system_type"] == "membrane"
                else None
            ),
        )
        log.info(
            "运行 provenance 已保存: %s",
            os.path.join(output_dir, "run_provenance.json"),
        )

    # ----- 1.5 自动构建溶剂腿缓存 -----
    if config.only_complex_charging:
        log.info("charging-only：跳过溶剂腿缓存构建与校验")
    else:
        if args.openmm_cache_only:
            _cache_only_audits["solvent"] = validate_openmm_cache_only(
                output_dir, phase="solvent"
            )
            config.data["openmm_cache_only_solvent_audit_sha256"] = (
                _cache_only_audits["solvent"]["audit_sha256"]
            )
            audit_path = os.path.join(output_dir, "openmm_cache_only_audit.json")
            temporary = f"{audit_path}.{os.getpid()}.tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(_cache_only_audits, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, audit_path)
            log.warning(
                "[WARN] OpenMM-cache-only：复合物与溶剂缓存通过自洽校验；审计写入 %s",
                audit_path,
            )
        else:
            ligand_resname = _get_residue_name_by_atom_index(
                topology, ligand_indices[0]
            )
            solvent_identity = _ligand_parameter_identity(
                system,
                topology,
                ligand_indices,
                ligand_resname,
                config.top,
                getattr(config, "ligand_xml", None),
                include_dir,
                positions=positions,
            )
            if config.reset or not solvent_cache_exists(
                output_dir,
                config.solvent_ionic_strength_molar,
                solvent_identity,
                charge_treatment=_charge_protocol["charge_treatment"],
            ):
                log.info(
                    "溶剂腿缓存缺失或盐协议不匹配，开始构建 %.3f M NaCl 配体体系...",
                    config.solvent_ionic_strength_molar,
                )
                if not build_and_cache_solvent_leg(
                    output_dir,
                    topology,
                    positions,
                    ligand_indices,
                    ligand_resname,
                    ligand_ffxml=getattr(config, "ligand_xml", None),
                    top_file=config.top,
                    gmx_include_dir=include_dir,
                    ionic_strength_molar=config.solvent_ionic_strength_molar,
                    cache_identity=solvent_identity,
                    charge_treatment=_charge_protocol["charge_treatment"],
                ):
                    log.error("[ERR] 自动构建溶剂腿失败，无法继续一键 ABFE")
                    sys.exit(1)
            else:
                log.info("检测到已有溶剂腿缓存，跳过重建")

    # ----- 2. 初始化 Pipeline -----
    pipeline = ABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=config.temperature,
        output_dir=output_dir,
        checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        platform_name=config.platform,
        # 膜体系只影响复合物腿（memtodolist §3.2 / docs/TODO.md 的膜受体前置条）。
        # 协议在这里解析，即在创建任何 Context 之前完成 fail-closed 检查（§6.1）。
        environment_type=config.get("system_type"),
        membrane=config.get("membrane"),
        confirm_soluble_with_lipids=bool(
            config.get("confirm_soluble_with_lipids", False)
        ),
        dispersion_protocol=_dispersion_protocol["dispersion_protocol"],
        forcefield_family=_dispersion_protocol["forcefield_family"],
        charge_treatment=_charge_protocol["charge_treatment"],
        charge_transfer_reservoir_correction=_charge_protocol.get(
            "charge_transfer_reservoir_correction"
        ),
        # 只读：让这条腿的 final_results.json 带上禁报标记（中性配体是 no-op）。
        charge_protocol=_charge_protocol,
        unclosed_charge_transfer_diagnostics=bool(
            config.get("run_unclosed_charge_transfer_diagnostics", False)
        ),
        # §9 质量门需要的显式输入，从膜输入声明里取（可溶体系为空 dict，不进该分支）。
        membrane_quality_inputs=(
            {
                key: (_membrane_input_report["declared"] or {}).get(key)
                for key in (
                    "pocket_atom_indices",
                    "coion_atom_index",
                    "literature_apl_nm2",
                    # 诊断专用参考值：只落盘偏差，不判门（MEM-12）。与
                    # `literature_apl_nm2` 刻意分成两个字段，见
                    # `abfe_core.evaluate_membrane_quality_gate` 里的说明。
                    "pure_lipid_reference_apl_nm2",
                    "equilibration_length_ns",
                    "ligand_resname",
                )
            }
            | {
                "composition": _system_composition,
                # [MEM-17 已移除 2026-08-03] 这里曾传 `expected_pre_equilibration_frames`
                # 给 §9 质量门做帧数对账。resume 追加的重复帧是真的（实测那条 100 ns
                # 多 1 帧），但对账拦的是主线而不是根因 —— 根因在
                # `abfe_pipeline.pre_equilibrate` 的 `DCDReporter(append=True)`
                # 没有先把 DCD 截断到 checkpoint 帧边界。按用户决定删除，别再加回来。
                # §9 的判据吃的是**总**平衡时长（上游 + 本流程），不是只看本流程。
                # 上游时长不可考时传 None，提取器会退回用轨迹自身跨度——那是我们
                # 唯一能实测到的时长，比塞一个编出来的数诚实。
                "equilibration_length_ns": _total_equil_ns,
            }
            if _membrane_input_report is not None
            else None
        ),
        membrane_quality_gate_mode=config.get("membrane_quality_gate"),
        repeat_seed=_repeat_seed,
        leg_name="complex",
        residual_sampling_enabled=outer_lambda_runtime is not None,
        residual_basis_force_factory=(
            outer_lambda_runtime.force_factory if outer_lambda_runtime is not None else None
        ),
        residual_state_coefficients_factory=(
            outer_lambda_runtime.state_coefficients_factory
            if outer_lambda_runtime is not None
            else None
        ),
        residual_energy_offset_kj_mol=(
            outer_lambda_runtime.energy_offset_kj_mol
            if outer_lambda_runtime is not None
            else 0.0
        ),
        sampling_score_sha256=(
            outer_lambda_runtime.sampling_score_sha256
            if outer_lambda_runtime is not None
            else None
        ),
        residual_plugin_identity=(
            outer_lambda_runtime.provenance_payload()
            if outer_lambda_runtime is not None
            else None
        ),
        residual_em_policy=(
            outer_lambda_runtime.em_policy
            if outer_lambda_runtime is not None
            else "no_residual_twin"
        ),
        residual_feature_name=OUTER_LAMBDA_RESIDUAL_FEATURE_NAME,
    )
    # 预平衡 block 收敛早停开关（PLAN 2026-08-26）。不设配置键时默认开；
    # 膜体系/无配体重原子时 `pre_equilibrate()` 内部仍会强制不生效（fail-closed）。
    pipeline.enable_equilibration_convergence_stop = bool(
        config.get("enable_equilibration_convergence_stop", True)
    )

    # ----- 3. 加载可选参数 -----
    # 二面角修正
    torsion_params = (
        _load_json_object_file(config.torsion_params, "torsion 参数")
        if config.torsion_params
        else None
    )

    # DEXP 势能
    # ----- 4. Boresch 参数获取（可能触发预平衡估算） -----
    if config.boresch:
        log.info("复合物腿 Boresch 已启用 | source=%s", config.boresch_source)
    else:
        log.info("复合物腿 Boresch 已显式关闭")
    # 🔑 [2026-07-28] 只跑单条腿的增量模式**必须在 rebalance 之前**就换成冻结腿
    # 采样时用的那组 Boresch，否则 rebalance 会在另一个限制势下平衡坐标，
    # 而后续计算用的是冻结那组——起始系综与被求值的 Hamiltonian 对不上。
    #
    # attachment 腿此前漏了这一步（只在 `_run_boresch_attachment_only` 内部换，
    # 那已经是 rebalance 之后），实测留下证据：
    #     rebalance : r0=0.429 nm, θA=83.5°, θB=69.5°   ← boresch_simple.json
    #     attachment: r0=0.431 nm, θA=83.7°, θB=74.68°  ← committed
    # λ=1 那 50k 步平衡不能替代它：本体系的瓶颈已经确认是**慢构象弛豫**，
    # 短平衡消不掉起始系综的记忆。
    #
    # 参考实现没有这个断点：`pipeline.zsh` 里 charging-complex(749) 与
    # restraint(763) 都从同一个 `prerun.run` 出发，而 restr_pull.mdp 与
    # restr_pull_decouple.mdp 由**同一份** restrinfo 生成(678)。
    if config.only_complex_charging or config.only_boresch_attachment:
        _mode = "charging-only" if config.only_complex_charging else "attachment-only"
        # [0831issue P2] 这两个 only-* 入口会把冻结 Stage 2 与本次新采样求和，
        # 所以在这里就把温度一致性挡住（详见 _load_frozen_stage_result）。
        boresch_restraint = _load_frozen_stage2_boresch(
            output_dir, expected_temperature_K=float(config.temperature)
        )
        log.info(
            "%s：rebalance 之前就采用冻结 Stage 2 的 Boresch Hamiltonian "
            "(r0=%.6f nm)，不读取/重估 boresch_simple.json",
            _mode,
            float(boresch_restraint["equilibrium_values"]["r0"]),
        )
    elif config.only_ligand_in_water:
        # [LR-04] `resolve_boresch_restraint` 会**无条件**跑完基线预平衡，
        # 那是复合物腿的开销；溶剂腿一个字节都用不上它。
        boresch_restraint = None
        log.info("ligand-in-water-only：跳过 Boresch 估算与带限制力 rebalance")
    else:
        boresch_restraint = resolve_boresch_restraint(config, pipeline)

    # ----- 4b. 换配体：闭式重训 outer-λ 局部残差（EXP-033 §5 P1） -----
    # 位置是**刻意**选在这里：`resolve_boresch_restraint` 内部会无条件跑完基线预平衡，
    # 所以 `pre_equilibration.dcd` 到这一行才保证存在；而 vanishing 采样还没开始，
    # 残差要影响的正是它。绑定走 pipeline 的统一入口，与构造期共用同一段 fail-closed
    # 校验（no_residual_twin / plugin identity 两道都不绕过）。
    if _residual_refit_pending:
        _residual_resource_manifest = _refit_outer_lambda_residual_for_this_ligand(
            pipeline=pipeline, config=config, output_dir=output_dir,
            topology=topology, system=system, ligand_indices=ligand_indices, log=log,
        )
        outer_lambda_runtime = build_outer_lambda_local_residual_runtime(
            topology=topology,
            ligand_indices=ligand_indices,
            system=system,
            temperature_kelvin=float(config.temperature),
            potential_type=str(config.potential),
            output_dir=output_dir,
            ligand_indices_path=os.path.join(output_dir, "ligand_indices.json"),
            leg_name="complex",
            platform_name=config.platform,
            resource_manifest=_residual_resource_manifest,
        )
        config.data["outer_lambda_local_residual_ibs_identity"] = (
            outer_lambda_runtime.provenance_payload()
        )
        pipeline.attach_residual_sampling_runtime(
            enabled=True,
            basis_force_factory=outer_lambda_runtime.force_factory,
            state_coefficients_factory=outer_lambda_runtime.state_coefficients_factory,
            energy_offset_kj_mol=outer_lambda_runtime.energy_offset_kj_mol,
            sampling_score_sha256=outer_lambda_runtime.sampling_score_sha256,
            plugin_identity=outer_lambda_runtime.provenance_payload(),
            em_policy=outer_lambda_runtime.em_policy,
            feature_name=OUTER_LAMBDA_RESIDUAL_FEATURE_NAME,
        )
        log.info(
            "%s 已延迟绑定 | score=%s | em_policy=%s",
            OUTER_LAMBDA_RESIDUAL_FEATURE_NAME,
            outer_lambda_runtime.sampling_score_sha256,
            outer_lambda_runtime.em_policy,
        )

    # ----- 5. 带限制力再平衡（如果启用） -----
    if boresch_restraint and not config.skip_rebalance:
        log.info("执行带 Boresch 限制力的再平衡...")
        rebal_data = pipeline._rebalance_with_boresch(
            boresch_params=boresch_restraint,
            n_steps=config.rebalance_steps,
            resume=config.resume and not config.reset,
        )
        pipeline.positions = rebal_data["positions"]
        pipeline.box_vectors = rebal_data["box_vectors"]
        log.info("[OK] 再平衡完成，坐标已更新")

    if config.only_boresch_attachment:
        candidate_path = _run_boresch_attachment_only(
            config,
            pipeline,
            boresch_restraint,
        )
        log.info(
            "attachment-only 已结束；stage1/stage2/solvent 未运行，主结果未覆盖。候选文件: %s",
            candidate_path,
        )
        return

    if config.only_complex_charging:
        candidate_path = _run_complex_charging_only(
            config,
            pipeline,
            boresch_restraint,
        )
        log.info(
            "charging-only 已结束；Stage 2 与 solvent 未运行，主结果未覆盖。候选文件: %s",
            candidate_path,
        )
        return

    # [B5] Freeze the complex-leg co-ion identity before any stage, preopt, or
    # top-level protocol key is built.  The spec is selected once (or read and
    # verified on resume) and the minimal identity is what downstream caches
    # receive; no later stage is allowed to select from coordinates again.
    _complex_coion_spec_relpath = os.path.relpath(
        pipeline._co_alchemical_ion_spec_path(), output_dir
    )
    pipeline._coion_spec_relative_path = _complex_coion_spec_relpath
    complex_coion_identity = pipeline.co_alchemical_ion_runtime_identity(
        "complex", spec_relative_path=_complex_coion_spec_relpath
    )
    if complex_coion_identity is not None:
        coion_identities["complex"] = complex_coion_identity
        run_provenance = _update_run_provenance_co_alchemical_ions(
            output_dir,
            {"complex": complex_coion_identity, "solvent": None},
            current=run_provenance,
        )

    # ----- 6. 运行复合物腿主流程 -----
    if config.only_ligand_in_water:
        # [LR-04] ligand-in-water-only 调试入口：复合物腿整段不跑 ⟹ 没有 ΔG_complex，
        # 也就不可能有 ΔG_bind。溶剂腿跑完在下面早退，绝不走到第 8 节。
        log.info("ligand-in-water-only：跳过复合物腿主流程（stage1 / stage2 / attachment 均未运行）")
        complex_results = None
        dg_complex = err_complex = dg_boresch = None
        attachment_result = {}
    else:
        log.info("启动复合物腿主采样流程 (%s)", config.decoupling)
        _scope_pipeline_with_optional_outer_lambda_em(
            pipeline, outer_lambda_runtime is not None
        )
        # 🔑 开关开着 ⟹ 要么真挂上了，要么**报错**。绝不静默跑 baseline。
        #
        # 缺冻结 manifest 是**正常初始状态**，不是错误状态 —— 输入的时候谁都没有这份
        # 蒸馏出来的权重。所以 fail-closed 的位置从"资源缺失"挪到了这里："训完了仍然
        # 没有"。中间那一步（EXP-033 P1 闭式重训）自己失败会当场抛，不会走到这。
        #
        # 这道断言防的是**未来重构**：`_residual_refit_pending` 的赋值、重训调用点、
        # 腿的分支只要有一处挪位，开着开关的 run 就会拿 `residual_sampling_enabled=False`
        # 造出来的 pipeline 一路跑完，报告里一切正常，而残差一次都没生效。
        if config.outer_lambda_local_residual_ibs and outer_lambda_runtime is None:
            raise RuntimeError(
                f"{OUTER_LAMBDA_RESIDUAL_FEATURE_NAME} 开关是开的，但到开始采样时运行时"
                "仍然没有绑上。这不该发生：冻结 manifest 覆盖不到当前配体时应该走自动"
                "闭式重训（docs/archive/EXP-033_P1_LANDED_2026-09-12.md）。"
                "拒绝静默按 baseline 跑完 —— 那会产出一份看起来一切正常、而残差从未生效的结果。"
            )

        complex_results = pipeline.run_full_pipeline(
            # 🔑 [2026-09-01] 质量门放行开关（见 --allow-untrusted-stage-results）。
            # 不传的话 _last_run_config 里就是 False，行为与之前完全一致（fail-closed）。
            allow_untrusted_stage_results=bool(
                config.get("allow_untrusted_stage_results", False)
            ),
            # 🔑 [2026-09-18] split-half σ 下界是否采用（默认 True）。
            # ⚠️ 必须在这里显式透传：本函数是**逐键枚举**往 run_full_pipeline 传参，
            # 没枚举的键永远进不了 `kwargs` —— `stage2_autonomous_controller` 就栽过
            # 同一个坑（见本文件下方注释：「配置里写了也没用」）。漏了这一行的后果
            # 不是默认值失效，而是**关不掉**：做 σ 口径 A/B 时没有对照臂。
            inflate_sigma_from_split_half=bool(
                config.get("inflate_sigma_from_split_half", True)
            ),
            decoupling_scheme=config.decoupling,
            potential_type=config.potential,
            dexp_params=dexp_params,
            boresch_params=boresch_restraint,
            torsion_params=torsion_params,
            n_equil_steps=config.get("n_equil_steps", 5_000_000),
            resume=config.resume and not config.reset,
            run_equilibration=not equilibrium_is_done(
                output_dir,
                # 🔑 [2026-09-01] 同上：走统一入口，口径与 pre_equilibrate 内部一致。
                expected_fingerprint=pipeline.pre_equilibration_identity_fingerprint(
                    config.get("n_equil_steps", 5_000_000)
                ),
                # [P1-07] checkpoint 以目标 Simulation 的真实 loadCheckpoint 成功为准。
                simulation=_checkpoint_probe_simulation(pipeline),
            ) or config.reset,
            # 注意：这个 system_type 是**腿身份**（complex/solvent），与膜协议的
            # 环境类型（soluble/membrane）是两个不同的轴，后者走
            # ABFEPipeline(environment_type=...)。
            system_type="complex",
            n_steps_per_window=config.n_steps_per_window,
            steps_per_update=config.steps_per_update,
            n_states_per_stage=config.get("stage1_n_states", 16),
            stage1_n_states=config.get("stage1_n_states", 16),
            stage2_n_states=config.get("stage2_n_states", config.get("stage1_n_states", 16)),
            # None 时不改变行为（optimize_stage2_vanishing 自己的默认 23 生效）。
            stage2_final_n_states=config.get("stage2_final_n_states"),
            stage2_refine_extra_points_per_segment=config.get(
                "stage2_refine_extra_points_per_segment"
            ),
            stage2_window_min_states=config.get("stage2_window_min_states"),
            # 🔑 [2026-09-16] `stage2_autonomous_controller` 在 abfe_pipeline 里一直是
            # `kwargs.get(..., True)`，但**从来没有人往下传** ⟹ 配置里写了也没用，
            # 自治循环无法关闭。做「固定预算、不按中间结果提前停」的对照实验时，
            # 自适应补帧本身就是要消除的选择偏差，必须能关。默认仍是 True，
            # 不写这个键时行为逐字不变。
            stage2_autonomous_controller=config.get(
                "stage2_autonomous_controller", True),
            stage2_window_max_states=config.get("stage2_window_max_states"),
            # 🔑🔑 [2026-09-15] Stage-2 控制器的两道生产预算闸。读侧（pipeline 组装
            # `effective_config` → `Stage2RepairController`）早就有了，**上游一直没接**
            # ⟹ 两条腿都退回读侧默认（预算"未知"=不拦、每块 250k），配置/预设/CLI
            # 里设的值一个都没生效。两个键都是执行策略、不进缓存身份
            # （见 abfe_pipeline._NON_IDENTITY_KWARGS），所以可以无条件透传：
            # 没配就是 None，pipeline 侧按"缺键=未知"过滤掉，行为逐字不变。
            stage2_production_budget_steps=config.get("stage2_production_budget_steps"),
            stage2_max_production_blocks_per_window=config.get(
                "stage2_max_production_blocks_per_window"
            ),
            **_path_evolution_kwargs(config),
            stage2_free_energy_densify_points=config.get(
                "stage2_free_energy_densify_points"
            ),
            enable_early_stop=config.enable_early_stop,
            enable_gradual_warmup=config.enable_gradual_warmup,
            warmup_steps=config.warmup_steps,
            n_workers=config.n_workers,
            parallel_stages=config.parallel_stages,
            decharge_method=config.get("decharge_method", "pme"),
            allow_disk_boresch_autoload=True,
            enable_lambda_refine=config.get("enable_lambda_refine", False),
            refine_n_steps_per_window=config.get("refine_n_steps_per_window", 30000),
            refine_steps_per_update=config.get("refine_steps_per_update", config.steps_per_update),
            refine_max_window_span_kJ=config.get("refine_max_window_span_kJ", 35.0),
            pilot_finite_difference_delta=config.get("pilot_finite_difference_delta", 0.01),
            pilot_n_steps_per_state=config.get("pilot_n_steps_per_state", 10000),
            # 🔑 [0831issue P2] abfe_config.json 把这个键文档化成可用开关，但 main() 从来
            # 没透传过 —— 用户按注释设成正整数后一个 shadow 诊断都不会写，却毫无提示。
            # None（默认）时 abfe_pipeline 侧行为不变，不产生任何 shadow 诊断。
            pilot_shadow_checkpoint_interval=config.get("pilot_shadow_checkpoint_interval"),
            ibs_lse_log_residual_tolerance=config.get(
                "ibs_lse_log_residual_tolerance", 0.5
            ),
            min_bias_updates=config.get("min_bias_updates", 12),
            max_bias_updates=config.get("max_bias_updates", 50),
            required_consecutive_bias_updates=config.get(
                "required_consecutive_bias_updates", 3
            ),
            max_bias_warmup_steps=config.get("max_bias_warmup_steps", 500000),
            # 🔑 [2026-07-27] 此前这个参数只在 --only-complex-charging 那条路径接通，
            # 完整 dual_lambda 链路根本没传，REMDManager 于是用默认上限、在建任何 GPU
            # Context 前预防性回退 CPU——整个 decharging 阶段慢约两个数量级，而且那条
            # 告警只 print 到终端、pipeline.log 里看不见，表现得像卡死。
            charging_max_resident_contexts=config.get("charging_max_resident_contexts"),
            attachment_lambdas=config.get("attachment_lambdas"),
            attachment_n_steps_per_state=config.get("attachment_n_steps_per_state", 250_000),
            attachment_equil_steps_per_state=config.get("attachment_equil_steps_per_state", 50_000),
            attachment_steps_per_sample=config.get("attachment_steps_per_sample", 1_000),
            attachment_seed=config.get("attachment_seed", 20260728),
            attachment_n_seeds=config.get("attachment_n_seeds", 1),
        )
    
        # 🔑 [P1-12] 之前这里 .get(..., 0.0) 会把缺字段静默补成 0：一条损坏/部分
        # 失败的腿会被汇总成"看似成功"的 ΔG_bind。统一 sanity gate 缺字段/非有限/
        # 负误差/converged 不为 True 一律 fail closed。
        _complex_leg_gate = validate_final_leg_result(
            complex_results, context="complex leg (in-memory)", source="run_full_pipeline"
        )
        dg_complex = _complex_leg_gate["delta_G_kJ_mol"]
        err_complex = _complex_leg_gate["error_kJ_mol"]
        if _repeat_seed is not None:
            run_provenance = _update_run_provenance_seed_contract(
                output_dir,
                {"complex": pipeline.seed_contract_snapshot()},
                current=run_provenance,
            )
        dg_boresch = complex_results.get("boresch_correction_kJ_mol", 0.0)
        attachment_result = complex_results.get("boresch_attachment", {})

    # ----- 7. 自动加载溶剂相缓存并运行溶剂腿 (ABFE 必选项) -----
    log.info("\n" + "="*70)
    log.info("启动溶剂相 (Ligand-in-Water) 配体腿计算 (自动加载缓存)...")
    log.info("="*70)
    
    try:
        sys_solv, top_solv, pos_solv, box_solv, lig_idx_solv = load_native_system(
            output_dir,
            phase="solvent",
            prefer_equilibrated=not config.reset,
            reconstruct_bonds_from_system=args.openmm_cache_only,
        )
    except FileNotFoundError:
        log.error("[ERR] 未找到溶剂相缓存，自动构建步骤未成功完成")
        sys.exit(1)
        
    pos_solv, box_solv = center_system_rigidly(pos_solv, box_solv, lig_idx_solv)

    outer_lambda_runtime_solv = None
    if outer_lambda_runtime is not None:
        # The loader/factory contract is shared, but the plugin Force is bound
        # to each leg's concrete topology.  Do not reuse a Force or silently
        # reuse complex atom-type indices in the solvent System.
        outer_lambda_runtime_solv = build_outer_lambda_local_residual_runtime(
            topology=top_solv,
            ligand_indices=lig_idx_solv,
            system=sys_solv,
            temperature_kelvin=float(config.temperature),
            potential_type=str(config.potential),
            output_dir=output_dir,
            ligand_indices_path=os.path.join(output_dir, "ligand_indices_solvent.json"),
            leg_name="solvent",
            platform_name=config.platform,
            # 见上面 `_residual_resource_manifest` 的说明：两段 vdW 共用同一份权重。
            resource_manifest=_residual_resource_manifest,
        )

    solvent_out_dir = os.path.join(output_dir, "solvent_leg")
    # 🔑 溶剂腿有两条**方向相反**的规矩，别搞混：
    #
    #   - **恒压器**：刻意不接 environment_type / membrane。memtodolist §3.2
    #     "溶剂腿继续使用普通 MonteCarloBarostat"——溶剂腿是配体在体相水里，与膜无关，
    #     即使复合物腿跑 membrane，这里也必须保持各向同性恒压器。不要"为了一致"
    #     把 config 的 system_type 传进来，那会给纯水盒装上膜 barostat。
    #   - **色散路线**：必须与复合物腿**完全相同**。§1.3 路线 A 与 §7.5 要求
    #     "复合物腿和溶剂腿使用同一套 ligand–environment 非键定义"。若复合物腿
    #     关掉了炼金均匀密度 LRC 而溶剂腿还留着，两腿的 ligand–environment 口径
    #     就不一致，ΔG_bind = ΔG_solv − ΔG_cplx 的差值里会混进一个协议差。
    pipeline_solv = ABFEPipeline(
        system=sys_solv, topology=top_solv, positions=pos_solv, box_vectors=box_solv,
        ligand_indices=lig_idx_solv, temperature=config.temperature,
        output_dir=solvent_out_dir, checkpoint_dir=os.path.join(solvent_out_dir, "checkpoints"),
        platform_name=config.platform,
        dispersion_protocol=_dispersion_protocol["dispersion_protocol"],
        forcefield_family=_dispersion_protocol["forcefield_family"],
        # ⚠️ 与恒压器方向相反：色散路线与净电荷路线**必须**两腿相同（§1.3 路线 A / §6.1），
        # 只有 environment_type/membrane 是溶剂腿刻意不接的。
        charge_treatment=_charge_protocol["charge_treatment"],
        charge_transfer_reservoir_correction=_charge_protocol.get(
            "charge_transfer_reservoir_correction"
        ),
        # 只读：让这条腿的 final_results.json 带上禁报标记（中性配体是 no-op）。
        charge_protocol=_charge_protocol,
        unclosed_charge_transfer_diagnostics=bool(
            config.get("run_unclosed_charge_transfer_diagnostics", False)
        ),
        repeat_seed=_repeat_seed,
        leg_name="solvent",
        residual_sampling_enabled=outer_lambda_runtime_solv is not None,
        residual_basis_force_factory=(
            outer_lambda_runtime_solv.force_factory
            if outer_lambda_runtime_solv is not None
            else None
        ),
        residual_state_coefficients_factory=(
            outer_lambda_runtime_solv.state_coefficients_factory
            if outer_lambda_runtime_solv is not None
            else None
        ),
        residual_energy_offset_kj_mol=(
            outer_lambda_runtime_solv.energy_offset_kj_mol
            if outer_lambda_runtime_solv is not None
            else 0.0
        ),
        sampling_score_sha256=(
            outer_lambda_runtime_solv.sampling_score_sha256
            if outer_lambda_runtime_solv is not None
            else None
        ),
        residual_plugin_identity=(
            outer_lambda_runtime_solv.provenance_payload()
            if outer_lambda_runtime_solv is not None
            else None
        ),
        residual_em_policy=(
            outer_lambda_runtime_solv.em_policy
            if outer_lambda_runtime_solv is not None
            else "no_residual_twin"
        ),
        residual_feature_name=OUTER_LAMBDA_RESIDUAL_FEATURE_NAME,
    )
    # 与复合物腿同一个开关（PLAN 2026-08-26）。
    pipeline_solv.enable_equilibration_convergence_stop = bool(
        config.get("enable_equilibration_convergence_stop", True)
    )

    # [B5] Use a path relative to the run root so complex and solvent records
    # are unambiguous when audited together.  This call is before the solvent
    # stage protocol key is built inside run_full_pipeline().
    _solvent_coion_spec_relpath = os.path.relpath(
        pipeline_solv._co_alchemical_ion_spec_path(), output_dir
    )
    pipeline_solv._coion_spec_relative_path = _solvent_coion_spec_relpath
    solvent_coion_identity = pipeline_solv.co_alchemical_ion_runtime_identity(
        "solvent", spec_relative_path=_solvent_coion_spec_relpath
    )
    if solvent_coion_identity is not None:
        coion_identities["solvent"] = solvent_coion_identity
        run_provenance = _update_run_provenance_co_alchemical_ions(
            output_dir,
            {
                "complex": coion_identities.get("complex"),
                "solvent": solvent_coion_identity,
            },
            current=run_provenance,
        )

    # [B5] Do not start the solvent leg (and never reach final aggregation)
    # with a partially populated charge-transfer identity block.  The two legs
    # may select different atom indices/fingerprints, but their route and ion /
    # restraint protocol must be the same.
    _assert_coion_legs_complete_and_compatible(
        coion_identities,
        pipeline.resolve_co_alchemical_ion_spec(),
        pipeline_solv.resolve_co_alchemical_ion_spec(),
        _charge_protocol["charge_treatment"],
    )
    
    # 运行溶剂腿 (🔑 强制关闭 Boresch)
    _scope_pipeline_with_optional_outer_lambda_em(
        pipeline_solv, outer_lambda_runtime_solv is not None
    )
    if config.outer_lambda_local_residual_ibs and outer_lambda_runtime_solv is None:
        raise RuntimeError(
            f"{OUTER_LAMBDA_RESIDUAL_FEATURE_NAME} 开关是开的，但溶剂腿的运行时没绑上。"
            "两段 vdW 必须用同一份权重；只有一条腿挂上残差，两腿的 ligand–environment "
            "口径就不一致，ΔG_bind = ΔG_solv − ΔG_cplx 的差值里会混进一个协议差。"
        )

    solv_results = pipeline_solv.run_full_pipeline(
        # 🔑 [2026-09-18] 与复合物腿同一个开关，两腿必须同口径 ——
        # 一腿放大一腿不放大，ΔG_bind 的误差合成就是混口径的。
        inflate_sigma_from_split_half=bool(
            config.get("inflate_sigma_from_split_half", True)
        ),
        decoupling_scheme=config.decoupling,
        potential_type=config.potential,
        dexp_params=dexp_params,
        boresch_params=None,  # 绝对不加 Boresch
        torsion_params=torsion_params,
        n_equil_steps=config.get("n_equil_steps", 5_000_000),
        resume=config.resume and not config.reset,
        run_equilibration=not equilibrium_is_done(
            solvent_out_dir,
            # 🔑 [2026-09-01] 同上：走统一入口。溶剂腿永远是 soluble
            # （§3.2：继续用普通 MonteCarloBarostat），barostat payload 恒为 None。
            expected_fingerprint=pipeline_solv.pre_equilibration_identity_fingerprint(
                config.get("n_equil_steps", 5_000_000)
            ),
            # [P1-07] checkpoint 以目标 Simulation 的真实 loadCheckpoint 成功为准。
            simulation=_checkpoint_probe_simulation(pipeline_solv),
        ) or config.reset,
        system_type="solvent",
        n_steps_per_window=config.n_steps_per_window,
        steps_per_update=config.steps_per_update,
        n_states_per_stage=config.get("stage1_n_states", 16),
        stage1_n_states=config.get("stage1_n_states", 16),
        stage2_n_states=config.get("stage2_n_states", config.get("stage1_n_states", 16)),
        # None 时不改变行为（optimize_stage2_vanishing 自己的默认 23 生效）。
        stage2_final_n_states=config.get("stage2_final_n_states"),
        stage2_refine_extra_points_per_segment=config.get(
            "stage2_refine_extra_points_per_segment"
        ),
        stage2_window_min_states=config.get("stage2_window_min_states"),
        # 🔑 [2026-09-16] `stage2_autonomous_controller` 在 abfe_pipeline 里一直是
        # `kwargs.get(..., True)`，但**从来没有人往下传** ⟹ 配置里写了也没用，
        # 自治循环无法关闭。做「固定预算、不按中间结果提前停」的对照实验时，
        # 自适应补帧本身就是要消除的选择偏差，必须能关。默认仍是 True，
        # 不写这个键时行为逐字不变。
        stage2_autonomous_controller=config.get(
            "stage2_autonomous_controller", True),
        stage2_window_max_states=config.get("stage2_window_max_states"),
        # 🔑🔑 [2026-09-15] Stage-2 控制器的两道生产预算闸。读侧（pipeline 组装
        # `effective_config` → `Stage2RepairController`）早就有了，**上游一直没接**
        # ⟹ 两条腿都退回读侧默认（预算"未知"=不拦、每块 250k），配置/预设/CLI
        # 里设的值一个都没生效。两个键都是执行策略、不进缓存身份
        # （见 abfe_pipeline._NON_IDENTITY_KWARGS），所以可以无条件透传：
        # 没配就是 None，pipeline 侧按"缺键=未知"过滤掉，行为逐字不变。
        stage2_production_budget_steps=config.get("stage2_production_budget_steps"),
        stage2_max_production_blocks_per_window=config.get(
            "stage2_max_production_blocks_per_window"
        ),
        **_path_evolution_kwargs(config),
        stage2_free_energy_densify_points=config.get(
            "stage2_free_energy_densify_points"
        ),
        enable_early_stop=config.enable_early_stop,
        enable_gradual_warmup=config.enable_gradual_warmup,
        warmup_steps=config.warmup_steps,
        n_workers=config.n_workers,
        parallel_stages=config.parallel_stages,
        decharge_method=config.get("decharge_method", "pme"),
        allow_disk_boresch_autoload=False,
        enable_lambda_refine=config.get("enable_lambda_refine", False),
        refine_n_steps_per_window=config.get("refine_n_steps_per_window", 30000),
        refine_steps_per_update=config.get("refine_steps_per_update", config.steps_per_update),
        refine_max_window_span_kJ=config.get("refine_max_window_span_kJ", 35.0),
        pilot_finite_difference_delta=config.get("pilot_finite_difference_delta", 0.01),
        pilot_n_steps_per_state=config.get("pilot_n_steps_per_state", 10000),
        # 🔑 [0831issue P2] abfe_config.json 把这个键文档化成可用开关，但 main() 从来
        # 没透传过 —— 用户按注释设成正整数后一个 shadow 诊断都不会写，却毫无提示。
        # None（默认）时 abfe_pipeline 侧行为不变，不产生任何 shadow 诊断。
        pilot_shadow_checkpoint_interval=config.get("pilot_shadow_checkpoint_interval"),
        ibs_lse_log_residual_tolerance=config.get(
            "ibs_lse_log_residual_tolerance", 0.5
        ),
        min_bias_updates=config.get("min_bias_updates", 12),
        max_bias_updates=config.get("max_bias_updates", 50),
        required_consecutive_bias_updates=config.get(
            "required_consecutive_bias_updates", 3
        ),
        max_bias_warmup_steps=config.get("max_bias_warmup_steps", 500000),
        charging_max_resident_contexts=config.get("charging_max_resident_contexts"),
    )
    
    # 🔑 [P1-12] 同 complex 腿：缺字段不再静默补 0，统一 sanity gate fail closed。
    _solvent_leg_gate = validate_final_leg_result(
        solv_results, context="solvent leg (in-memory)", source="run_full_pipeline"
    )
    dg_solvent = _solvent_leg_gate["delta_G_kJ_mol"]
    err_solvent = _solvent_leg_gate["error_kJ_mol"]

    if config.only_ligand_in_water:
        # [LR-04] 在这里退，而不是让它走到第 8 节：ΔG_bind 需要两条腿，
        # 复合物腿这次根本没跑。早退比"补个 None 往下传"安全 ——
        # 后者每加一个下游分支就多一次把半份结果写成完整结果的机会。
        log.info("=" * 70)
        log.info(
            "ligand-in-water-only 已结束：ΔG_solvent = %.4f ± %.4f kJ/mol",
            dg_solvent, err_solvent,
        )
        log.info(
            "这是**配体在水中的去耦自由能，不是结合能**：复合物腿本次没有运行，"
            "ΔG_bind 需要两条腿，%s 未被覆盖。",
            os.path.join(output_dir, "final_results.json"),
        )
        log.info("溶剂腿结果: %s", os.path.join(solvent_out_dir, "final_results.json"))
        log.info("=" * 70)
        return
    if _repeat_seed is not None:
        run_provenance = _update_run_provenance_seed_contract(
            output_dir,
            {
                "complex": pipeline.seed_contract_snapshot(),
                "solvent": pipeline_solv.seed_contract_snapshot(),
            },
            current=run_provenance,
        )

    # Repeat the final-summary gate at the last boundary so a future change to
    # either pipeline cannot turn a valid preflight into a null leg in output.
    _assert_coion_legs_complete_and_compatible(
        coion_identities,
        pipeline.resolve_co_alchemical_ion_spec(),
        pipeline_solv.resolve_co_alchemical_ion_spec(),
        _charge_protocol["charge_treatment"],
    )

    if coion_identities:
        # Both leg result files must remain self-auditing when copied away from
        # the run directory.  The complex file was written before solvent was
        # constructed, so merge the now-complete two-leg block atomically.
        _combined_coion_identities = {
            "complex": coion_identities.get("complex"),
            "solvent": coion_identities.get("solvent"),
        }
        _augment_final_result_with_co_alchemical_ions(
            os.path.join(output_dir, "final_results.json"),
            _combined_coion_identities,
        )
        _augment_final_result_with_co_alchemical_ions(
            os.path.join(solvent_out_dir, "final_results.json"),
            _combined_coion_identities,
        )
    
    # ----- 8. 计算最终结合自由能 ΔG_bind -----
    # 🔑 关键修复：标准双解耦循环推导（态 A=复合物真实结合；态 D=去耦+释放限制力后
    # 的 apo 蛋白+标准态配体；G(D)=G(A)+ΔG_complex）给出的是
    # ΔG_bind = G(A) - G(apo) - G(配体,溶液,coupled) = ΔG_solvent - ΔG_complex，
    # 不是 ΔG_complex - ΔG_solvent。原因很直观：对一个真实结合的配体，口袋里去耦
    # 花的自由能应该比溶液里去耦花的更多（口袋相互作用更强，这正是它愿意结合的
    # 原因），所以 ΔG_complex > ΔG_solvent；正确公式 ΔG_solvent-ΔG_complex 应为负
    # （有利结合），之前 ΔG_complex-ΔG_solvent 会给出正值（看起来"不利结合"），
    # 这与本次用参考方法交叉验证时符号持续相反的现象完全吻合。
    # 🔑 [ATT-09] 循环闭合统一走 abfe_core.combine_binding_free_energy，
    # 不再在这里手写公式。这里 dg_complex 取自 total_delta_G_complex_kJ_mol，
    # Boresch 释放项已经烘焙在里面（abfe_pipeline: total_dg = dg_phys +
    # cons_correction + dg_boresch），所以 already_included=True，不再减第二次。
    _constraint_identity = _assert_matching_ligand_constraint_identity(
        pipeline.system,
        pipeline.ligand_indices,
        pipeline_solv.system,
        pipeline_solv.ligand_indices,
        context="dual_lambda ABFE 热力学循环",
    )
    apbs_correction = float(config.get("apbs_correction_kJ_mol", 0.0) or 0.0)
    _reservoir_payload = _charge_protocol.get(
        "charge_transfer_reservoir_correction"
    )
    # 🔑🔑 [2026-09-18] 未闭合 charge-transfer 的**第三道**闸（前两道：runabfe 建
    # Context 之前、abfe_pipeline.run_full_pipeline 入口）。三道必须一起解，否则
    # 诊断模式只是把死点挪到两条腿都烧完之后。
    # ⚠️ 这一道解除的代价跟前两道**不同**：`_reservoir_delta` 会取 0.0 ⟹ carrier
    # reservoir / 标准态那一项不是"未验证"，是**整项被省掉**。所以除了三个禁报
    # 标记之外，还要显式落 `charge_transfer_reservoir_correction_omitted`，
    # 免得下游看到一个有限数就以为循环是齐的。
    _reservoir_omitted = False
    if _charge_protocol["charge_treatment"] == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        if not isinstance(_reservoir_payload, dict):
            if not bool(config.get(
                    "run_unclosed_charge_transfer_diagnostics", False)):
                raise RuntimeError(
                    "charge-transfer 已通过预检但缺少已验证的双腿 reservoir correction；"
                    "拒绝汇总"
                    "（只要诊断数值 ⟹ `--run-unclosed-charge-transfer-diagnostics`；"
                    "届时 reservoir 项按 0 计、产物标记不得发布。）"
                )
            _reservoir_omitted = True
            log.warning(
                "[WARN] [诊断模式] 缺少 carrier reservoir correction，汇总时该项**按 0 计**"
                "⟹ ΔG_bind 少了一整项：这不是「未验证」，是「不完整」。"
                "产物带 charge_transfer_reservoir_correction_omitted=true + "
                "must_not_report_delta_g_bind=true，**不得**发布或进任何汇总表。"
            )
    if (_charge_protocol["charge_treatment"]
            == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
            and not _reservoir_omitted):
        _reservoir_canonical = json.dumps(
            _reservoir_payload, sort_keys=True, separators=(",", ":")
        )
        for _leg_name, _leg_result in (
            ("complex", complex_results),
            ("solvent", solv_results),
        ):
            _leg_correction = (
                _leg_result.get("charge_transfer_reservoir_correction")
                if isinstance(_leg_result, dict)
                else None
            )
            if not isinstance(_leg_correction, dict):
                raise RuntimeError(
                    f"charge-transfer {_leg_name} leg 缺少 reservoir correction；"
                    "拒绝混合未闭合或旧缓存"
                )
            _leg_correction = validate_charge_transfer_reservoir_correction(
                _leg_correction,
                temperature_k=float(config.temperature),
            )
            if json.dumps(_leg_correction, sort_keys=True, separators=(",", ":")) != _reservoir_canonical:
                raise RuntimeError(
                    f"charge-transfer {_leg_name} leg 的 reservoir correction 与主协议不一致；"
                    "拒绝混合不同温度/不同 C4-C5 证据的 artifact"
                )
        _reservoir_delta = float(
            _reservoir_payload["delta_G_bind_correction_kJ_mol"]
        )
        _reservoir_error = float(_reservoir_payload["error_kJ_mol"])
    else:
        _reservoir_delta = 0.0
        _reservoir_error = 0.0
    cycle = combine_binding_free_energy(
        dg_complex_kJ_mol=dg_complex,
        dg_solvent_kJ_mol=dg_solvent,
        err_complex_kJ_mol=err_complex,
        err_solvent_kJ_mol=err_solvent,
        dg_boresch_kJ_mol=dg_boresch,
        boresch_already_included_in_complex=True,
        apbs_correction_kJ_mol=apbs_correction,
        charge_transfer_reservoir_correction_kJ_mol=_reservoir_delta,
        charge_transfer_reservoir_error_kJ_mol=_reservoir_error,
        # [P0-12a / §3.0] 两条腿的配体构象系综必须重叠，否则这个减法没有意义。
        # 门在 combine_binding_free_energy 里（唯一一处），这里只负责把两侧的
        # 自报 summary 交上去；缺哪一侧就记 not_evaluated，不会伪装成通过。
        complex_conformer_summary=complex_results.get("ligand_conformer_diagnostics"),
        solvent_conformer_summary=solv_results.get("ligand_conformer_diagnostics"),
        strict_cross_leg_conformer=bool(
            config.get("strict_cross_leg_conformer", False)
        ),
    )
    if cycle.get("cross_leg_conformer_gate") == "WARN":
        log.warning(
            "[跨腿构象门 WARN] 两条腿的配体构象系综不重叠：%s",
            (cycle.get("cross_leg_conformer_report") or {}).get("reason", ""),
        )
        log.warning(
            "[跨腿构象门 WARN] **不阻断汇总**（构象差在这类装置上是必然产物："
            "溶剂腿里 vdW 一关配体就塌，复合物腿里主体腔卡住它）。"
            "但这**不等于没事** —— 下面这个 ΔG_bind 里有一项构象自由能差没被计入，"
            "结果中永久记为 cross_leg_conformer_gate=WARN。"
        )
    delta_g_bind_uncorrected = cycle["delta_G_bind_uncorrected_kJ_mol"]
    delta_g_bind = cycle["delta_G_bind_kJ_mol"]
    total_err_bind = cycle["total_error_kJ_mol"]
    
    log.info("\n" + "="*70)
    log.info("ABFE 最终结合自由能计算结果:")
    log.info("   复合物腿 (膜/蛋白+水) ΔG_cplx  = %.2f ± %.2f kJ/mol", dg_complex, err_complex)
    log.info("   溶剂腿   (纯水)       ΔG_solv  = %.2f ± %.2f kJ/mol", dg_solvent, err_solvent)
    if attachment_result.get("present"):
        log.info(
            "   Boresch attachment     ΔG_attach = %.2f ± %.2f kJ/mol",
            float(attachment_result.get("delta_G_kJ_mol", 0.0)),
            float(attachment_result.get("error_kJ_mol", 0.0)),
        )
    log.info("   Boresch 解析修正      ΔG_rest  = %.2f kJ/mol", dg_boresch)
    log.info("   APBS 外部长程修正     ΔG_APBS  = %.2f kJ/mol", apbs_correction)
    log.info("   --------------------------------------------------------")
    log.info("   结合自由能 ΔG_bind           = %.2f ± %.2f kJ/mol", delta_g_bind, total_err_bind)
    log.info("                              = %.2f ± %.2f kcal/mol", delta_g_bind/4.184, total_err_bind/4.184)
    _result_status = _binding_result_status(complex_results, solv_results)
    if _result_status["results_untrusted"]:
        log.error(
            "   [结果不可信] %s 腿的质量门未通过（results_untrusted=true）⟹ 上面这个 "
            "ΔG_bind **不得**作为可发布结果引用。逐条证据见 final_binding_results.json "
            "的 result_status.stage_quality_failures。",
            "/".join(_result_status["results_untrusted_legs"]),
        )
    for _leg_name, _leg_status in _result_status["per_leg"].items():
        _d = _leg_status.get("sampling_drift") or {}
        _z = _d.get("total_drift_over_2sigma")
        if _d.get("available") and _z is not None and float(_z) > 1.0:
            log.warning(
                "   [半程漂移] %s 腿 stage-2 总漂移 %.2f kJ/mol = %.2f×2σ，"
                "逐窗最大 %.2f×2σ；报出的 ± 是 MBAR 内部误差（σ=%.2f kJ/mol），"
                "按逐窗漂移定下界会是 %.2f kJ/mol（默认未采用）。",
                _leg_name, float(_d.get("total_drift_kJ_mol") or 0.0), float(_z),
                float(_d.get("max_window_drift_over_2sigma") or 0.0),
                float(_d.get("total_error_mbar_kJ_mol") or 0.0),
                float(_d.get("total_error_inflated_kJ_mol") or 0.0),
            )
    log.info("="*70)

    # 保存最终结合结果
    final_bind_result = {
        # 🔑 [2026-09-16] 质量状态与 ΔG 同级落盘：先前这四个键只在逐腿
        # `final_results.json` 里，汇总文件一个都没带（见 `_binding_result_status`）。
        "result_status": _result_status,
        "results_untrusted": _result_status["results_untrusted"],
        "precision_status": _result_status["precision_status"],
        "publishable_as_accepted_result": _result_status[
            "publishable_as_accepted_result"],
        "stage_quality_failures": _result_status["stage_quality_failures"],
        # [2026-09-18] 禁报标记与 ΔG_bind 同级：只读顶层的人也必须看得见。
        **{k: _result_status[k] for k in _MUST_NOT_REPORT_KEYS
           if k in _result_status},
        "complex_delta_G_kJ_mol": float(dg_complex),
        "solvent_delta_G_kJ_mol": float(dg_solvent),
        "boresch_correction_kJ_mol": float(dg_boresch),
        "boresch_attachment": attachment_result,
        "constraint_identity": _constraint_identity,
        "charge_transfer_reservoir_correction": _reservoir_payload,
        # [2026-09-18] 诊断模式下 reservoir 项按 0 计 ⟹ 这个 ΔG_bind **少了一整项**。
        # 只在真的省掉时才写这个键（正常 run 一个字节都不变）。
        **({"charge_transfer_reservoir_correction_omitted": True,
            "charge_transfer_reservoir_correction_omitted_reason": (
                "run_unclosed_charge_transfer_diagnostics: carrier reservoir/"
                "standard-state correction 缺失，汇总时按 0 计；ΔG_bind 不完整，"
                "不得发布或进任何汇总表"
            )} if _reservoir_omitted else {}),
        # ✅ complex_delta_G_kJ_mol 来自 total_delta_G_complex_kJ_mol，已经在
        # abfe_pipeline.py 里把 Boresch 释放修正烘焙进去 (total_dg = dg_phys +
        # cons_correction + dg_boresch)；下面 delta_g_bind_uncorrected 没有再单独
        # 减一次 dg_boresch。这里显式标记，避免下游看到 boresch_correction_kJ_mol
        # 和 complex_delta_G_kJ_mol 并列就误以为还需要再手动扣一次。
        "boresch_correction_already_included_in_complex_delta_G": True,
        "delta_G_bind_uncorrected_kJ_mol": float(delta_g_bind_uncorrected),
        "apbs_correction_kJ_mol": float(apbs_correction),
        "delta_G_bind_kJ_mol": float(delta_g_bind),
        "delta_G_bind_kcal_mol": float(delta_g_bind / 4.184),
        "total_error_kJ_mol": float(total_err_bind),
        # [ATT-09] 循环闭合的完整记账，由唯一实现给出。
        "thermodynamic_cycle_terms": cycle,
        "timestamp": datetime.now().isoformat(),
        "provenance": run_provenance,
        "thermodynamic_cycle": THERMODYNAMIC_CYCLE_DOC,
        "external_corrections": {
            "apbs": {
                "delta_G_kJ_mol": float(apbs_correction),
                "applied_to": "final_binding_free_energy",
                "note": config.get("apbs_correction_note", ""),
                "status": "applied" if abs(apbs_correction) > 0.0 else "not_applied",
                "helper": "apbs_correction.py",
                "protocol_warning": (
                    "APBS correction is treated as an explicit external final term. "
                    "Use only when the generated APBS cycle matches the intended thermodynamic correction."
                ),
            },
            "charge_transfer_reservoir": (
                {
                    "delta_G_kJ_mol": float(_reservoir_delta),
                    "error_kJ_mol": float(_reservoir_error),
                    "status": "applied" if _reservoir_payload is not None else "not_applicable",
                    "protocol_version": (
                        _reservoir_payload.get("protocol_version")
                        if isinstance(_reservoir_payload, dict)
                        else None
                    ),
                }
                if _reservoir_payload is not None
                else {
                    "delta_G_kJ_mol": 0.0,
                    "error_kJ_mol": 0.0,
                    "status": "not_applicable",
                    "protocol_version": None,
                }
            ),
        },
        "diagnostics": {
            "complex": complex_results.get("diagnostics", {}),
            "complex_stage_diagnostics": complex_results.get("stage_diagnostics", {}),
            "solvent": solv_results.get("diagnostics", {}),
            "solvent_stage_diagnostics": solv_results.get("stage_diagnostics", {}),
            "boresch": {
                "method": boresch_restraint.get("method") if isinstance(boresch_restraint, dict) else None,
                "diagnostics": boresch_restraint.get("diagnostics", {}) if isinstance(boresch_restraint, dict) else {},
                "force_constants_raw": boresch_restraint.get("force_constants_raw", {}) if isinstance(boresch_restraint, dict) else {},
                "force_constant_clipped": boresch_restraint.get("force_constant_clipped", {}) if isinstance(boresch_restraint, dict) else {},
                "uses_analytical_release_formula": bool(config.boresch and boresch_restraint),
                "analytical_release_assumption": (
                    "Boresch release correction assumes locally harmonic, approximately Gaussian restraint-coordinate fluctuations."
                ),
                "analytical_release_assumption_checked": (
                    isinstance(boresch_restraint, dict)
                    and bool(boresch_restraint.get("diagnostics", {}).get("boresch_harmonicity_check", {}).get("ok"))
                ),
                "analytical_release_reliable": (
                    boresch_restraint.get("diagnostics", {}).get("boresch_harmonicity_check", {}).get("harmonic_assumption_ok")
                    if isinstance(boresch_restraint, dict)
                    else None
                ),
            },
            "independent_repeats": {
                "performed": False,
                "note": "Independent repeat runs are not launched automatically; run the same config with distinct random seeds and compare final_binding_results.json.",
            },
        },
    }
    _qualification = charge_treatment_qualification_payload(
        _charge_protocol.get("charge_treatment")
    )
    if _qualification:
        final_bind_result.update(_qualification)
        final_bind_result["production_qualification"] = dict(_qualification)

    if coion_identities:
        # [B5] The combined result records both leg identities, while neutral
        # runs retain the legacy shape (no synthetic empty co-ion record).
        final_bind_result["co_alchemical_ions"] = {
            "complex": coion_identities.get("complex"),
            "solvent": coion_identities.get("solvent"),
        }
    bind_out_path = os.path.join(output_dir, "final_binding_results.json")
    _atomic_write_json_with_encoder(bind_out_path, final_bind_result)
    log.info("最终结合自由能结果已保存: %s", bind_out_path)

    # ----- 7. 输出最终结果 -----
    # 🔑 [2026-09-16] 收尾这行**不再无条件报 OK**：质量门失败时流程是刻意不中止的
    # （为了拿到溶剂腿），于是"跑完了"和"结果能用"是两件事，收尾必须分开说。
    if _result_status["results_untrusted"]:
        log.error(
            "[不可信] ABFE 流程已跑完，但结果被标记 results_untrusted=true"
            "（%s 腿质量门未通过）⟹ 不得作为可发布结果引用。",
            "/".join(_result_status["results_untrusted_legs"]),
        )
    elif not _result_status["publishable_as_accepted_result"]:
        log.warning(
            "[完成但未验收] ABFE 流程已跑完；precision_status=%s ⟹ %s",
            _result_status["precision_status"],
            _result_status["publishable_rejection_reason"],
        )
    else:
        log.info("[OK] ABFE 计算完成")
    log.info("结果已生成，见 %s", bind_out_path)


if __name__ == "__main__":
    main()
