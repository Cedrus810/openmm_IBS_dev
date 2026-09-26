# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 复合物 charging-only 隔离重跑
# ---------------------------------------------------------------------------
def _load_frozen_stage_result(
    path: str,
    expected_stage: str,
    expected_temperature_K: Optional[float] = None,
) -> Dict:
    """Load a completed stage as a read-only input for an isolated rerun.

    🔑 [0831issue P2] `expected_temperature_K` 关掉"混温候选"这个缺口。
    `--only-complex-charging` / `--only-boresch-attachment` 会把这份冻结 stage 与
    **本次**新采样的结果求和，而新采样侧的指纹两边都用本次温度、恒过；冻结侧的
    `protocol_key.payload.temperature_K` 一直在手上却从不比对。于是
    "stage2/预平衡跑完后改 `--temperature` 300→310 再 `--resume --only-*`" 会产出一份
    跨温度非法求和的 `final_binding_results_candidate.json`（kBT 差约 3%），且无任何告警。
    对照组 `_analyze_dual_leg_artifacts` 早就有同样的硬校验。

    不一致 → 直接拒绝。字段缺失 → 大声告警但不阻断：那只说明这份工件来自记录该字段
    之前的版本，硬拒会连带堵掉合法的旧工件重跑。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少冻结的 {expected_stage} 结果: {path}")
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
    if result.get("stage") != expected_stage:
        raise RuntimeError(
            f"冻结结果 stage={result.get('stage')!r}，期望 {expected_stage!r}: {path}"
        )
    for key in ("total_delta_G", "total_error"):
        value = result.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise RuntimeError(f"冻结结果 {path} 的 {key}={value!r} 非法")
    if expected_temperature_K is not None:
        frozen_temp = (
            ((result.get("protocol_key") or {}).get("payload") or {})
            .get("temperature_K")
        )
        if frozen_temp is None:
            log.warning(
                "[WARN] 冻结 %s 结果没有记录 temperature_K（%s）：无法核对它与本次 "
                "--temperature=%.4f K 是否一致。若这份工件是在别的温度下采的，"
                "把它与本次新采样求和是跨温度非法组装。",
                expected_stage, path, float(expected_temperature_K),
            )
        elif not np.isclose(
            float(frozen_temp), float(expected_temperature_K), rtol=0.0, atol=1e-6
        ):
            raise RuntimeError(
                f"冻结 {expected_stage} 的采样温度 {float(frozen_temp)} K 与本次 "
                f"--temperature={float(expected_temperature_K)} K 不一致（{path}）。"
                "两者会被求和进同一个 ΔG_bind，kBT 口径不同即非法组装；"
                "请用原温度重跑，或换一个新的 --output 目录重新采样。"
            )
    return result


def _load_frozen_stage2_boresch(
    output_dir: str, expected_temperature_K: Optional[float] = None
) -> Dict:
    """Use the exact restraint Hamiltonian that produced frozen Stage 2."""
    stage2_path = os.path.join(
        os.path.abspath(output_dir),
        "checkpoints",
        "stage2_vanishing.json",
    )
    # [0831issue P2] 顺带核对冻结 Stage 2 的采样温度，见 _load_frozen_stage_result。
    stage2 = _load_frozen_stage_result(
        stage2_path, "vanishing", expected_temperature_K=expected_temperature_K
    )
    params = (
        ((stage2.get("protocol_key") or {}).get("payload") or {}).get(
            "boresch_params"
        )
    )
    if not isinstance(params, dict):
        raise RuntimeError(
            f"冻结 Stage 2 未记录 Boresch Hamiltonian，无法安全重跑 charging: "
            f"{stage2_path}"
        )
    return _sanitize_boresch_params_strict(params)


def _boresch_core_signature(params: Optional[Dict]) -> Optional[Dict]:
    if not isinstance(params, dict):
        return None
    params = _sanitize_boresch_params(params)
    return {
        "receptor_indices": [int(v) for v in params.get("receptor_indices", [])],
        "ligand_indices": [int(v) for v in params.get("ligand_indices", [])],
        "equilibrium_values": {
            str(k): round(float(v), 8)
            for k, v in (params.get("equilibrium_values") or {}).items()
        },
        "force_constants": {
            str(k): round(float(v), 8)
            for k, v in (params.get("force_constants") or {}).items()
        },
    }


def _prepare_charging_rerun_dir(
    output_dir: str, requested: Optional[str], label: str = "charging"
) -> str:
    """为「只跑某一条腿」的隔离重跑准备一个空目录。

    label 只影响默认子目录名与报错措辞；attachment 腿复用同一套隔离/防覆盖规则。
    """
    if requested:
        rerun_dir = os.path.abspath(requested)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rerun_dir = os.path.abspath(
            os.path.join(output_dir, f"{label}_rerun", stamp)
        )
    source_dir = os.path.abspath(output_dir)
    if rerun_dir == source_dir:
        raise RuntimeError(f"{label}-only 输出目录不能等于主 output 目录")
    if os.path.isdir(rerun_dir) and os.listdir(rerun_dir):
        raise FileExistsError(
            f"{label}-only 输出目录已存在且非空，拒绝覆盖: {rerun_dir}"
        )
    os.makedirs(rerun_dir, exist_ok=True)
    return rerun_dir


def _charging_mbar_crosschecks(
    u_kn_path: str,
    n_k_path: str,
    temperature_k: float,
    production_result: Dict,
) -> Dict:
    """Re-solve one u_kn with raw MBAR and adjacent BAR as diagnostics."""
    u_kn = np.load(u_kn_path, allow_pickle=False)
    n_k = np.load(n_k_path, allow_pickle=False).astype(int)
    analyzer = TraditionalMBARAnalyzer(temperature=temperature_k)
    analyzer._last_n_k = n_k
    raw = analyzer.solve(u_kn, decorrelate=False)

    from pymbar import other_estimators

    kt = 0.008314462618 * float(temperature_k)
    offsets = np.concatenate(([0], np.cumsum(n_k)))
    bar_delta_f = []
    bar_variance = []
    for k in range(len(n_k) - 1):
        start_k, end_k = int(offsets[k]), int(offsets[k + 1])
        start_j, end_j = int(offsets[k + 1]), int(offsets[k + 2])
        w_forward = u_kn[k + 1, start_k:end_k] - u_kn[k, start_k:end_k]
        w_reverse = u_kn[k, start_j:end_j] - u_kn[k + 1, start_j:end_j]
        edge = other_estimators.bar(
            w_forward,
            w_reverse,
            compute_uncertainty=True,
        )
        bar_delta_f.append(float(edge["Delta_f"]))
        bar_variance.append(float(edge["dDelta_f"]) ** 2)
    bar_dg = float(np.sum(bar_delta_f) * kt)
    bar_error = float(np.sqrt(np.sum(bar_variance)) * kt)

    prod_dg = float(production_result["total_delta_G"])
    prod_error = float(production_result["total_error"])
    tolerance = max(
        2.0,
        3.0 * max(prod_error, float(raw["error"]), bar_error),
    )
    return {
        "production_decorrelated_mbar": {
            "delta_G_kJ_mol": prod_dg,
            "error_kJ_mol": prod_error,
        },
        "raw_all_frame_mbar": {
            "delta_G_kJ_mol": float(raw["delta_G"]),
            "error_kJ_mol": float(raw["error"]),
        },
        "adjacent_bar_sum": {
            "delta_G_kJ_mol": bar_dg,
            "error_kJ_mol": bar_error,
            "uncertainty_note": (
                "相邻 BAR 方差按独立边近似相加，仅作为交叉检查。"
            ),
        },
        "acceptance_tolerance_kJ_mol": tolerance,
        "consistent": bool(
            abs(float(raw["delta_G"]) - prod_dg) <= tolerance
            and abs(bar_dg - prod_dg) <= tolerance
        ),
    }


def _run_boresch_attachment_only(
    config: RunConfig,
    source_pipeline: ABFEPipeline,
    boresch_restraint: Dict,
) -> str:
    """[P1-17] 只跑 Boresch attachment 腿 A′→A，复用冻结的 stage1/stage2/solvent。

    stage1/stage2 是在**受约束**系综里测的，补上 attachment 项不会改变它们，
    所以这条腿可以作为增量跑（~2–3 h），不必重跑整条复合物腿（~7 h）。
    """
    if config.reset:
        raise RuntimeError(
            "--only-boresch-attachment 不能与 --reset 同用；该模式必须只读复用"
            "现有 stage1/stage2 与 solvent 结果"
        )
    if config.decoupling != "dual_lambda":
        raise RuntimeError("--only-boresch-attachment 只支持 --decoupling dual_lambda")
    # [§9 / MEM-14] 增量重跑同样是在**消费**那次预平衡，所以门必须先过。
    # 否则"门失败 → 用 --only-* 重跑"就成了另一条绕过路径。可溶体系直接短路。
    source_pipeline.ensure_membrane_quality_gate_passed()
    # 运行时解析出的这组只是个参考——真正要用的是冻结腿 payload 里那组（见下）。
    # 所以这里不拦，解析不出来也能继续。
    if not _has_valid_boresch_anchors(boresch_restraint):
        log.warning(
            "本次运行没有解析出有效的 Boresch 锚点；将直接采用冻结 stage1/stage2 采样时用的那组"
        )

    from ibs_engine import run_boresch_attachment_leg

    source_dir = os.path.abspath(config.output)
    # [0831issue P2] 冻结两阶段必须与本次 --temperature 同温，否则求和跨温度非法。
    _expected_temp_K = float(config.temperature)
    stage1 = _load_frozen_stage_result(
        os.path.join(source_dir, "checkpoints", "stage1_decharging.json"), "decharging",
        expected_temperature_K=_expected_temp_K,
    )
    stage2 = _load_frozen_stage_result(
        os.path.join(source_dir, "checkpoints", "stage2_vanishing.json"), "vanishing",
        expected_temperature_K=_expected_temp_K,
    )
    solvent_path = os.path.join(source_dir, "solvent_leg", "final_results.json")
    if not os.path.isfile(solvent_path):
        raise FileNotFoundError(f"缺少冻结的 solvent 最终结果: {solvent_path}")
    with open(solvent_path, encoding="utf-8") as handle:
        solvent = json.load(handle)
    for key in ("total_delta_G_complex_kJ_mol", "total_error_kJ_mol"):
        value = solvent.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise RuntimeError(f"冻结 solvent 结果的 {key}={value!r} 非法")

    # attachment 腿必须挂**冻结腿实际采样时用的那组**限制，而不是本次运行时解析
    # 出来的那组。两者经常不同且属正常：`--boresch-source simple` 解析的是
    # `boresch_simple.json`，而生产采样走 P0-10 的 commit 机制，用的是
    # `checkpoints/boresch_equilibrium_committed.json` 里重新锚定过的平衡值
    # （本例中 r0 0.42916 vs 0.43081、thetaA0 83.52° vs 83.69°）。
    # 挂错那组 = 给循环补的是另一个限制势的 attachment 项，循环照样不闭合。
    # 所以这里**不报错**，直接采用冻结值并把差异打出来。
    frozen_boresch = {}
    frozen_sigs = {}
    for name, frozen in (("stage1", stage1), ("stage2", stage2)):
        payload_bp = (
            ((frozen.get("protocol_key") or {}).get("payload") or {}).get("boresch_params")
        )
        if payload_bp is None:
            raise RuntimeError(f"冻结 {name} 的 protocol_key 里没有 boresch_params，无法确定它采样时用的限制势")
        frozen_boresch[name] = _sanitize_boresch_params(payload_bp)
        frozen_sigs[name] = _boresch_core_signature(payload_bp)

    # 这一条才是真正的硬矛盾：两条解耦腿如果用了不同的限制势，现有结果本身就废了。
    if frozen_sigs["stage1"] != frozen_sigs["stage2"]:
        raise RuntimeError(
            "冻结 stage1 与 stage2 用的 Boresch 限制势不同，现有复合物腿本身不自洽，"
            "拒绝在其上补 attachment 腿"
        )

    sampled_restraint = frozen_boresch["stage2"]

    # 🔑 三处必须是同一份 restraint signature，否则整条腿不是「同端点闭合实验」：
    #   (1) 传进来的 boresch_restraint —— rebalance 用它平衡了起始坐标
    #   (2) 冻结 stage1/stage2 采样时用的那组
    #   (3) 下面 apply_boresch_correction 算解析释放时用的那组
    # (3) 直接复用 (2)，所以这里只需把 (1) 和 (2) 对上。
    # 主流程已改成在 rebalance **之前**就加载冻结值，所以正常路径下这里必然一致；
    # 一旦不一致，说明有人改动了加载顺序 —— 那是硬错误，不能只打警告，
    # 因为起始系综会来自另一个限制势下的平衡态（λ=1 的短平衡消不掉它）。
    if _boresch_core_signature(boresch_restraint) != frozen_sigs["stage2"]:
        cur_eq = (_sanitize_boresch_params(boresch_restraint) or {}).get("equilibrium_values", {})
        frz_eq = sampled_restraint.get("equilibrium_values", {})
        diffs = []
        for key in sorted(set(cur_eq) | set(frz_eq)):
            a, b = cur_eq.get(key), frz_eq.get(key)
            if a is None or b is None or abs(float(a) - float(b)) > 1e-9:
                diffs.append(f"{key}: rebalance 用 {a} vs 冻结腿用 {b}")
        raise RuntimeError(
            "rebalance 用的 Boresch 参数与冻结 stage1/stage2 采样时用的那组不一致，"
            "起始坐标来自另一个限制势下的平衡态，attachment 腿不是同端点闭合实验。"
            "主流程应在 rebalance 之前就加载冻结值（见 runabfe.py 第 4 节）。\n  "
            + "\n  ".join(diffs)
        )
    log.info("rebalance / attachment / 解析释放 三处 Boresch signature 一致")
    boresch_restraint = sampled_restraint

    rerun_dir = _prepare_charging_rerun_dir(
        source_dir, config.get("attachment_rerun_dir"), label="attachment"
    )
    rerun_pipeline = ABFEPipeline(
        system=source_pipeline.system,
        topology=source_pipeline.topology,
        positions=source_pipeline.positions,
        box_vectors=source_pipeline.box_vectors,
        ligand_indices=source_pipeline.ligand_indices,
        temperature=config.temperature,
        output_dir=rerun_dir,
        checkpoint_dir=os.path.join(rerun_dir, "checkpoints"),
        platform_name=config.platform,
        # 🔑 rerun **继承 source pipeline 已解析的协议**，不重读 config——
        # 增量重跑必须与被复用的那次用同一个恒压器集合，而 config 可能已经改过。
        environment_type=source_pipeline.environment_type,
        membrane=source_pipeline.barostat_protocol.get("membrane"),
        confirm_soluble_with_lipids=bool(
            source_pipeline.barostat_protocol.get("soluble_with_lipids_confirmed", False)
        ),
        # 色散路线同样继承：换了它 u_kn 口径就变，增量段与被复用段不可比。
        dispersion_protocol=source_pipeline.dispersion_protocol,
        forcefield_family=source_pipeline.dispersion_protocol_info.get(
            "forcefield_family"
        ),
        # 净电荷路线同样继承：换了它 co-ion 身份与 charging 哈密顿量就变。
        charge_treatment=getattr(source_pipeline, "charge_treatment", None),
    )
    run_provenance = _write_run_provenance(
        rerun_dir,
        config,
        rerun_pipeline.system,
        rerun_pipeline.topology,
        rerun_pipeline.positions,
        barostat_protocol=rerun_pipeline.barostat_protocol,
    )

    stage0 = run_boresch_attachment_leg(
        source_pipeline.system,
        source_pipeline.topology,
        source_pipeline.positions,
        source_pipeline.box_vectors,
        boresch_restraint,
        temperature_k=float(config.temperature),
        lambdas=config.get("attachment_lambdas"),
        n_steps_per_state=int(config.get("attachment_n_steps_per_state", 250_000)),
        equil_steps_per_state=int(config.get("attachment_equil_steps_per_state", 50_000)),
        steps_per_sample=int(config.get("attachment_steps_per_sample", 1_000)),
        platform_name=config.platform,
        seed=int(config.get("attachment_seed", 20260728)),
        n_seeds=int(config.get("attachment_n_seeds", 1)),
        output_dir=os.path.join(rerun_dir, "attachment"),
        log=log.info,
    )
    os.makedirs(rerun_pipeline.checkpoint_dir, exist_ok=True)
    with open(
        os.path.join(rerun_pipeline.checkpoint_dir, "stage0_attachment.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(stage0, handle, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    correction = rerun_pipeline.apply_boresch_correction(
        boresch_restraint, autoload_from_disk=False
    )
    rerun_pipeline._last_run_config = {
        "potential_type": config.potential,
        "mode": "only_boresch_attachment",
    }
    complex_candidate = rerun_pipeline.compute_final_results(
        {"stage0": stage0, "stage1": stage1, "stage2": stage2},
        correction,
        system=rerun_pipeline.system,
        decoupling_scheme="dual_lambda",
    )
    cycle = combine_binding_free_energy(
        dg_complex_kJ_mol=float(complex_candidate["total_delta_G_complex_kJ_mol"]),
        dg_solvent_kJ_mol=float(solvent["total_delta_G_complex_kJ_mol"]),
        err_complex_kJ_mol=float(complex_candidate["total_error_kJ_mol"]),
        err_solvent_kJ_mol=float(solvent["total_error_kJ_mol"]),
        dg_boresch_kJ_mol=float(complex_candidate["boresch_correction_kJ_mol"]),
        boresch_already_included_in_complex=True,
        apbs_correction_kJ_mol=float(config.get("apbs_correction_kJ_mol", 0.0) or 0.0),
    )

    candidate = {
        "mode": "only_boresch_attachment_candidate",
        "promoted_to_primary_results": False,
        "attachment_delta_G_kJ_mol": float(stage0["attachment_delta_G_kJ_mol"]),
        "attachment_error_kJ_mol": float(stage0["attachment_error_kJ_mol"]),
        "attachment_delta_G_kcal_mol": float(stage0["attachment_delta_G_kJ_mol"]) / 4.184,
        "complex_delta_G_kJ_mol": float(complex_candidate["total_delta_G_complex_kJ_mol"]),
        "solvent_delta_G_kJ_mol": float(solvent["total_delta_G_complex_kJ_mol"]),
        "delta_G_bind_kJ_mol": float(cycle["delta_G_bind_kJ_mol"]),
        "delta_G_bind_kcal_mol": float(cycle["delta_G_bind_kJ_mol"]) / 4.184,
        "total_error_kJ_mol": float(cycle["total_error_kJ_mol"]),
        "thermodynamic_cycle_terms": cycle,
        "attachment_stage": stage0,
        "frozen_inputs": {
            "stage1_total_delta_G": float(stage1["total_delta_G"]),
            "stage2_total_delta_G": float(stage2["total_delta_G"]),
            "solvent_final_results": solvent_path,
        },
        "provenance": run_provenance,
        "timestamp": datetime.now().isoformat(),
    }
    candidate_path = os.path.join(rerun_dir, "final_binding_results_candidate.json")
    with open(candidate_path, "w", encoding="utf-8") as handle:
        json.dump(candidate, handle, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    log.info(
        "ΔG_attach = %.4f ± %.4f kJ/mol (%.4f kcal/mol)；参考 result.txt 的对应项是 +0.442 kcal/mol",
        candidate["attachment_delta_G_kJ_mol"],
        candidate["attachment_error_kJ_mol"],
        candidate["attachment_delta_G_kcal_mol"],
    )
    log.info(
        "ΔG_bind 候选 = %.4f kJ/mol = %.4f kcal/mol（原 −2.121）",
        candidate["delta_G_bind_kJ_mol"],
        candidate["delta_G_bind_kcal_mol"],
    )
    return candidate_path


def _run_complex_charging_only(
    config: RunConfig,
    source_pipeline: ABFEPipeline,
    boresch_restraint: Dict,
) -> str:
    if config.reset:
        raise RuntimeError(
            "--only-complex-charging 不能与 --reset 同用；该模式必须只读复用"
            "现有 Stage 2 与 solvent 结果"
        )
    if config.decoupling != "dual_lambda":
        raise RuntimeError("--only-complex-charging 只支持 --decoupling dual_lambda")
    if config.get("decharge_method", "pme") != "pme":
        raise RuntimeError("--only-complex-charging 只支持生产级 PME charging 路径")
    if not isinstance(boresch_restraint, dict):
        raise RuntimeError("charging-only 需要现有 complex Boresch 参数")
    # [§9 / MEM-14] 同上：增量重跑也在消费那次预平衡，门必须先过。
    source_pipeline.ensure_membrane_quality_gate_passed()

    source_dir = os.path.abspath(config.output)
    stage2_path = os.path.join(source_dir, "checkpoints", "stage2_vanishing.json")
    solvent_path = os.path.join(source_dir, "solvent_leg", "final_results.json")
    # [0831issue P2] 见 _load_frozen_stage_result：拒绝跨温度组装。
    stage2 = _load_frozen_stage_result(
        stage2_path, "vanishing", expected_temperature_K=float(config.temperature)
    )
    if not os.path.isfile(solvent_path):
        raise FileNotFoundError(f"缺少冻结的 solvent 最终结果: {solvent_path}")
    with open(solvent_path, encoding="utf-8") as handle:
        solvent = json.load(handle)
    for key in ("total_delta_G_complex_kJ_mol", "total_error_kJ_mol"):
        value = solvent.get(key)
        if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
            raise RuntimeError(f"冻结 solvent 结果的 {key}={value!r} 非法")

    stage2_boresch = (
        ((stage2.get("protocol_key") or {}).get("payload") or {}).get(
            "boresch_params"
        )
    )
    if _boresch_core_signature(stage2_boresch) != _boresch_core_signature(
        boresch_restraint
    ):
        raise RuntimeError(
            "当前 Boresch 参数与冻结 Stage 2 的协议不一致；拒绝拼接不同限制势的腿"
        )

    rerun_dir = _prepare_charging_rerun_dir(
        source_dir, config.get("charging_rerun_dir")
    )
    rerun_pipeline = ABFEPipeline(
        system=source_pipeline.system,
        topology=source_pipeline.topology,
        positions=source_pipeline.positions,
        box_vectors=source_pipeline.box_vectors,
        ligand_indices=source_pipeline.ligand_indices,
        temperature=config.temperature,
        output_dir=rerun_dir,
        checkpoint_dir=os.path.join(rerun_dir, "checkpoints"),
        platform_name=config.platform,
        # 🔑 rerun **继承 source pipeline 已解析的协议**，不重读 config——
        # 增量重跑必须与被复用的那次用同一个恒压器集合，而 config 可能已经改过。
        environment_type=source_pipeline.environment_type,
        membrane=source_pipeline.barostat_protocol.get("membrane"),
        confirm_soluble_with_lipids=bool(
            source_pipeline.barostat_protocol.get("soluble_with_lipids_confirmed", False)
        ),
        # 色散路线同样继承：换了它 u_kn 口径就变，增量段与被复用段不可比。
        dispersion_protocol=source_pipeline.dispersion_protocol,
        forcefield_family=source_pipeline.dispersion_protocol_info.get(
            "forcefield_family"
        ),
        # 净电荷路线同样继承：换了它 co-ion 身份与 charging 哈密顿量就变。
        charge_treatment=getattr(source_pipeline, "charge_treatment", None),
    )
    run_provenance = _write_run_provenance(
        rerun_dir,
        config,
        rerun_pipeline.system,
        rerun_pipeline.topology,
        rerun_pipeline.positions,
        barostat_protocol=rerun_pipeline.barostat_protocol,
    )

    n_states = int(config.get("stage1_n_states", 12))
    max_contexts = config.get("charging_max_resident_contexts")
    if max_contexts is not None and int(max_contexts) < 1:
        raise RuntimeError("--charging-max-resident-contexts 必须至少为 1")
    lambdas_coul = np.linspace(1.0, 0.0, n_states).tolist()
    lambdas_vdw = [1.0] * n_states
    expected_meta = _pme_u_kn_meta_payload(
        n_states=n_states,
        lambdas_coul=lambdas_coul,
        lambdas_vdw=lambdas_vdw,
        temperature_k=float(config.temperature),
        system=rerun_pipeline.system,
        topology=rerun_pipeline.topology,
        ligand_indices=rerun_pipeline.ligand_indices,
        boresch_params=boresch_restraint,
    )
    manifest_path = os.path.join(rerun_dir, "charging_sampling_manifest.json")
    manifest = {
        "mode": "only_complex_charging",
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "pid": os.getpid(),
        "fresh_sampling_required": True,
        "resume": False,
        "sampling_and_analysis_same_process": True,
        "remd_execution": {
            "requested_platform": str(config.platform),
            "max_resident_contexts": (
                None if max_contexts is None else int(max_contexts)
            ),
        },
        "expected_pme_u_kn_metadata": expected_meta,
        "frozen_inputs": {
            "stage2_vanishing": {
                "path": stage2_path,
                "sha256": _sha256_file(stage2_path),
                "delta_G_kJ_mol": float(stage2["total_delta_G"]),
                "error_kJ_mol": float(stage2["total_error"]),
            },
            "solvent_final": {
                "path": solvent_path,
                "sha256": _sha256_file(solvent_path),
                "delta_G_kJ_mol": float(
                    solvent["total_delta_G_complex_kJ_mol"]
                ),
                "error_kJ_mol": float(solvent["total_error_kJ_mol"]),
            },
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, cls=NumpyEncoder)

    try:
        stage1 = rerun_pipeline._run_dual_lambda_stage(
            "decharging",
            fixed_lam_coul=1.0,
            fixed_lam_vdw=1.0,
            n_states=n_states,
            n_steps_per_window=int(config.n_steps_per_window),
            steps_per_update=int(config.steps_per_update),
            system_type="complex",
            resume=False,
            potential_type=config.potential,
            dexp_params=_load_dexp_params_fail_closed(
                config.potential,
                config.dexp_params,
            ),
            optimized_lambdas=lambdas_coul,
            window_ranges=None,
            enable_early_stop=False,
            boresch_params=boresch_restraint,
            remd_max_resident_contexts=max_contexts,
        )
        rerun_pipeline._assert_stage_result_sane(
            "isolated complex decharging", stage1
        )

        decharging_dir = os.path.join(rerun_dir, "decharging")
        meta_path = os.path.join(
            decharging_dir, "decharging_pme_u_kn.meta.json"
        )
        u_kn_path = os.path.join(decharging_dir, "decharging_pme_u_kn.npy")
        n_k_path = u_kn_path + ".n_k.npy"
        with open(meta_path, encoding="utf-8") as handle:
            actual_meta = json.load(handle)
        if actual_meta != expected_meta:
            raise RuntimeError(
                "charging 采样前 manifest 与离线 u_kn Hamiltonian 指纹不一致"
            )

        crosschecks = _charging_mbar_crosschecks(
            u_kn_path,
            n_k_path,
            float(config.temperature),
            stage1,
        )
        if not crosschecks["consistent"]:
            raise RuntimeError(
                "charging 的去相关 MBAR、全帧 MBAR 与相邻 BAR 未通过一致性检查"
            )

        correction = rerun_pipeline.apply_boresch_correction(
            boresch_restraint, autoload_from_disk=False
        )
        rerun_pipeline._last_run_config = {
            "potential_type": config.potential,
            "mode": "only_complex_charging",
        }
        complex_candidate = rerun_pipeline.compute_final_results(
            {"stage1": stage1, "stage2": stage2},
            correction,
            system=rerun_pipeline.system,
            decoupling_scheme="dual_lambda",
        )
        dg_complex = float(
            complex_candidate["total_delta_G_complex_kJ_mol"]
        )
        err_complex = float(complex_candidate["total_error_kJ_mol"])
        dg_solvent = float(solvent["total_delta_G_complex_kJ_mol"])
        err_solvent = float(solvent["total_error_kJ_mol"])
        dg_boresch = float(complex_candidate["boresch_correction_kJ_mol"])
        apbs = float(config.get("apbs_correction_kJ_mol", 0.0) or 0.0)
        cycle = combine_binding_free_energy(
            dg_complex_kJ_mol=dg_complex,
            dg_solvent_kJ_mol=dg_solvent,
            err_complex_kJ_mol=err_complex,
            err_solvent_kJ_mol=err_solvent,
            dg_boresch_kJ_mol=dg_boresch,
            boresch_already_included_in_complex=True,
            apbs_correction_kJ_mol=apbs,
        )
        binding_candidate = {
            "mode": "only_complex_charging_candidate",
            "promoted_to_primary_results": False,
            "complex_delta_G_kJ_mol": dg_complex,
            "solvent_delta_G_kJ_mol": dg_solvent,
            "boresch_correction_kJ_mol": dg_boresch,
            "boresch_correction_already_included_in_complex_delta_G": True,
            "delta_G_bind_kJ_mol": float(cycle["delta_G_bind_kJ_mol"]),
            "delta_G_bind_kcal_mol": float(
                cycle["delta_G_bind_kJ_mol"] / 4.184
            ),
            "total_error_kJ_mol": float(cycle["total_error_kJ_mol"]),
            "thermodynamic_cycle_terms": cycle,
            "charging_crosschecks": crosschecks,
            "frozen_inputs": manifest["frozen_inputs"],
            "provenance": run_provenance,
            "timestamp": datetime.now().isoformat(),
        }
        binding_path = os.path.join(
            rerun_dir, "final_binding_results_candidate.json"
        )
        with open(binding_path, "w", encoding="utf-8") as handle:
            json.dump(binding_candidate, handle, indent=2, cls=NumpyEncoder)

        dcd_inventory = []
        for path in sorted(glob.glob(os.path.join(decharging_dir, "*.dcd"))):
            stat_result = os.stat(path)
            dcd_inventory.append(
                {
                    "path": path,
                    "size_bytes": int(stat_result.st_size),
                    "mtime_ns": int(stat_result.st_mtime_ns),
                }
            )
        exchange_path = os.path.join(
            decharging_dir, "decharging_exchange_diagnostics.json"
        )
        exchange_diagnostics = None
        if os.path.isfile(exchange_path):
            with open(exchange_path, encoding="utf-8") as handle:
                exchange_diagnostics = json.load(handle)
        manifest.update(
            {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "actual_pme_u_kn_metadata": actual_meta,
                "metadata_match": True,
                "dcd_inventory": dcd_inventory,
                "exchange_diagnostics": exchange_diagnostics,
                "charging_result": stage1,
                "crosschecks": crosschecks,
                "candidate_binding_result": binding_path,
            }
        )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, cls=NumpyEncoder)
    except BaseException as exc:
        manifest.update(
            {
                "status": "failed",
                "failed_at": datetime.now().isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, cls=NumpyEncoder)
        raise

    log.info("[OK] complex charging-only 候选结果完成: %s", binding_path)
    return binding_path


# 主入口
