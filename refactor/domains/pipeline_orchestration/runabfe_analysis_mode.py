# ---------------------------------------------------------------------------
# 分析模式（简化版，调用原有后处理）
# ---------------------------------------------------------------------------
def _resolve_production_qualification_from_sources(*sources: Dict) -> Dict:
    """Resolve one fail-closed production-qualification payload from result sources.

    A post-analysis run may read two leg results plus run_provenance.json. A
    charge-transfer result must not lose its experimental/not-qualified boundary
    merely because this alternate output path was used. Conversely, disagreeing
    leg/provenance identities are evidence of mixed artifacts and must not be
    silently reconciled.
    """
    candidates = []
    for source in sources:
        if source is None:
            continue
        if not isinstance(source, dict):
            raise RuntimeError(
                "production qualification source must be a JSON object, got "
                f"{type(source).__name__}"
            )

        explicit = source.get("production_qualification")
        if explicit is not None:
            if not isinstance(explicit, dict):
                raise RuntimeError(
                    "production_qualification must be a JSON object; refusing "
                    "to post-process a malformed qualification record"
                )
            required_keys = {
                "production_qualification_protocol_version",
                "feature_status",
                "production_qualified",
                "production_qualification_reason",
            }
            missing_keys = sorted(required_keys.difference(explicit))
            if missing_keys:
                raise RuntimeError(
                    "production_qualification is incomplete; missing "
                    f"{missing_keys}; refusing to guess"
                )
            if not isinstance(explicit.get("production_qualified"), bool):
                raise RuntimeError(
                    "production_qualification is missing boolean "
                    "production_qualified; refusing to guess"
                )
            candidates.append(dict(explicit))

        treatments = []
        charge_protocol = source.get("charge_protocol")
        config = source.get("config")
        for value in (
            source.get("charge_treatment"),
            charge_protocol.get("charge_treatment")
            if isinstance(charge_protocol, dict)
            else None,
            config.get("charge_treatment") if isinstance(config, dict) else None,
        ):
            if value is not None:
                treatments.append(str(value))
        unique_treatments = sorted(set(treatments))
        if len(unique_treatments) > 1:
            raise RuntimeError(
                "conflicting charge_treatment identities within one result "
                f"source: {unique_treatments}"
            )
        if unique_treatments:
            derived = charge_treatment_qualification_payload(unique_treatments[0])
            if derived:
                candidates.append(dict(derived))

    if not candidates:
        return {}
    expected = candidates[0]
    for candidate in candidates[1:]:
        if candidate != expected:
            raise RuntimeError(
                "complex/solvent/provenance production qualification mismatch; "
                "refusing to combine artifacts from different protocol states"
            )
    return dict(expected)


def _analyze_dual_leg_artifacts(base_dir, temperature, *, expected_context=None):
    """Read-only recovery through the same stage/window contracts as production."""
    import inspect
    import abfe_pipeline as pipeline_module
    import ibs_engine as engine

    temperature_K = float(temperature.value_in_unit(unit.kelvin) if hasattr(temperature, "value_in_unit") else temperature)
    kt = float((unit.MOLAR_GAS_CONSTANT_R * temperature_K * unit.kelvin).value_in_unit(unit.kilojoule_per_mole))
    context_path = os.path.join(base_dir, "checkpoints", "analysis_context.json")
    if expected_context is None:
        with open(context_path, encoding="utf-8") as handle:
            context_doc = json.load(handle)
        if context_doc != pipeline_module._protocol_fingerprint(context_doc.get("payload")):
            raise ValueError("analysis_context fingerprint 不匹配")
        expected_context = context_doc["payload"]
    stage_protocols = expected_context["stage_protocol_keys"]
    validator = ABFEPipeline.__new__(ABFEPipeline)
    total_dg, total_var = 0.0, 0.0
    stages = {}
    for number, (name, kind) in enumerate((("decharging", "coul"), ("vanishing", "vdw")), 1):
        expected_key = stage_protocols[name]
        if expected_key != pipeline_module._protocol_fingerprint(expected_key.get("payload")):
            raise ValueError(f"{name} 预期 protocol_key fingerprint 不匹配")
        protocol = expected_key["payload"]
        for key, value in pipeline_module._stage_analysis_protocol_versions(name).items():
            if protocol.get(key) != value:
                raise ValueError(f"{name} 当前协议 {key} 不匹配，拒绝旧 Hamiltonian/分析结果")
        if float(protocol["temperature_K"]) != temperature_K:
            raise ValueError(f"{name} 分析温度与采样协议不匹配")
        for field in ("constraint_identity", "co_alchemical_ion_runtime_identity", "charge_transfer_reservoir_correction"):
            if protocol.get(field) != expected_context.get(field):
                raise ValueError(f"{name} {field} 身份与本腿不匹配")
        if protocol.get("boresch_params") != stage_protocols["decharging"]["payload"].get("boresch_params"):
            raise ValueError("两阶段 Boresch 采样身份不一致")
        checkpoint_dir = os.path.join(base_dir, "checkpoints")
        with open(os.path.join(checkpoint_dir, f"preopt_dual_{name}.json"), encoding="utf-8") as handle:
            path = json.load(handle)
        lambdas = path["lambdas_var"]
        ranges = path["window_ranges"]
        n_states = len(lambdas)
        if (n_states < 2 or not ranges or lambdas[0] != 1.0 or lambdas[-1] != 0.0
                or not np.all(np.isfinite(lambdas)) or np.any(np.diff(lambdas) >= 0)):
            raise ValueError(f"{name} 预期 lambda 路径/窗口覆盖非法")
        covered = set()
        for start, end in ranges:
            if not 0 <= start < end <= n_states:
                raise ValueError(f"{name} window_ranges 越界")
            covered.update(range(start, end))
        if covered != set(range(n_states)):
            raise ValueError(f"{name} window_ranges 未覆盖全部 lambda 态")
        stage_path = os.path.join(checkpoint_dir, f"stage{number}_{name}.json")
        cached = None
        if os.path.isfile(stage_path):
            with open(stage_path, encoding="utf-8") as handle:
                cached = validator._validate_stage_checkpoint(json.load(handle), name, expected_key, lambdas, ranges)
        stage_dir = os.path.join(base_dir, name)
        is_ibs = kind == "vdw" or protocol.get("decharge_method") != "pme"
        window_kind = "shadow_coul" if protocol.get("decharge_method") == "shadow_ibs" else kind
        if is_ibs:
            config = protocol["run_config"]
            kwargs = {} if window_kind == "shadow_coul" else (config.get("kwargs") or {})
            defaults = inspect.signature(engine.IBSWindowManagerDualLambda.run_all_windows).parameters
            def setting(key):
                return kwargs.get(key, defaults[key].default)
            early_keys = ("min_steps", "check_interval_steps", "required_consecutive_passes", "min_ess_ratio",
                          "min_absolute_ess", "min_decorrelated_samples", "max_delta_g_drift_kJ_mol", "max_uncertainty_kJ_mol")
            expected_window_protocol = dict(
                repair_policy=protocol["sampling_repair_policy"],
                lse_log_residual_tolerance=kwargs.get("ibs_lse_log_residual_tolerance", defaults["lse_log_residual_tolerance"].default),
                enable_early_stop=window_kind != "shadow_coul" and bool(config.get("enable_early_stop", False)),
                current_early_stop_config={key: setting("early_stop_" + key) for key in early_keys},
                effective_target_steps=int(config["n_steps_per_window"]),
                current_coion_identity=expected_context.get("co_alchemical_ion_runtime_identity"),
                current_stage_protocol_key=expected_key,
            )
            # The production route replaces the final vdW window with a
            # separately sampled endpoint segment. Never invent its frames.
            excluded = {len(ranges)-1} if kind == "vdw" else set()
            windows = engine.load_ibs_window_outputs_from_dir(
                stage_dir, ranges, lambdas if kind == "coul" else [0.0]*n_states,
                [1.0]*n_states if kind == "coul" else lambdas,
                checkpoint_dir=checkpoint_dir, stage_type=window_kind,
                excluded_local_windows=excluded,
                expected_window_protocol=expected_window_protocol,
                current_sampling_score_sha256=((protocol.get("run_config") or {}).get("residual_sampling") or {}).get("sampling_score_sha256") if kind == "vdw" else None,
            )
            endpoint = None
            if kind == "vdw":
                try:
                    endpoint = pipeline_module._endpoint_analysis_artifact(
                        checkpoint_dir, stage_protocol_key=expected_key,
                        lambda_path_fingerprint=validator._lambda_path_fingerprint(lambdas, ranges),
                    )
                except FileNotFoundError as exc:
                    raise RuntimeError("回退分析拒绝用纯 IBS 窗口产出 vanishing 主值；缺少已验证的独立端点拼接结果") from exc
            if cached is None:
                if protocol.get("decharge_method") == "shadow_ibs":
                    raise RuntimeError("Shadow-IBS 回退缺少已验证的 bridge + IBS 阶段结果，不能只累计 IBS 段")
                gates = dict(protocol.get("final_gate_thresholds") or {})
                gates.pop("target_support_gate_protocol_version", None)
                cached = engine.solve_stage_integrated(windows, kt, **gates)
                if endpoint is not None:
                    cached = engine.combine_ibs_and_independent_endpoint(
                        cached, endpoint["solve_all"], endpoint["wet_dry_gate"], n_states=n_states,
                        target_support_min_absolute_ess=gates.get("final_min_target_absolute_ess"),
                    )
                    cached["independent_endpoint_diagnostics"] = endpoint["diagnostics"]
                cached["stage"] = name
                cached["lambda_endpoint_diagnostics"] = pipeline_module._stage_lambda_endpoint_diagnostics(
                    name, lambdas if kind == "coul" else [0.0]*n_states,
                    [1.0]*n_states if kind == "coul" else lambdas,
                )
                validator._populate_stage_diagnostics(cached)
                validator._assert_stage_result_sane(name, cached)
                pipeline_module._assert_sampling_result_converged(cached, context=name)
        elif cached is None:
            raise RuntimeError("PME decharging 缺少协议验证后的 stage checkpoint；不能用 IBS 回退")
        stages[name] = cached
        total_dg += float(cached["total_delta_G"])
        total_var += float(cached["total_error"])**2
    if stage_protocols["decharging"]["payload"].get("boresch_params"):
        with open(os.path.join(base_dir, "checkpoints", "stage0_attachment.json"), encoding="utf-8") as handle:
            attachment_doc = json.load(handle)
        restraint = pipeline_module._preopt_boresch_protocol_payload(stage_protocols["decharging"]["payload"]["boresch_params"])
        attachment_key = attachment_doc.get("protocol_key", {})
        if pipeline_module._protocol_fingerprint_ignoring_code_hash(attachment_key) != pipeline_module._protocol_fingerprint_ignoring_code_hash(expected_context.get("attachment_protocol_key")):
            raise ValueError("Boresch attachment 与当前 attachment 协议不匹配")
        if attachment_key != pipeline_module._protocol_fingerprint(attachment_key.get("payload")) or attachment_key["payload"].get("boresch_params") != restraint:
            raise ValueError("Boresch attachment protocol_key 不匹配")
        for field in ("temperature_K", "system_xml_sha256", "topology_sha256", "ligand_indices", "constraint_identity"):
            if attachment_key["payload"].get(field) != stage_protocols["decharging"]["payload"].get(field):
                raise ValueError(f"Boresch attachment {field} 与采样协议不匹配")
        dg, error = pipeline_module.validate_boresch_attachment_result(attachment_doc["result"])
        total_dg += dg
        total_var += error**2
    return dict(
        decoupling_delta_G_kJ_mol=total_dg, total_error_kJ_mol=float(np.sqrt(total_var)),
        constraint_identity=expected_context["constraint_identity"],
        charge_treatment=expected_context.get("charge_treatment"),
        charge_transfer_reservoir_correction=expected_context.get("charge_transfer_reservoir_correction"),
        co_alchemical_ion_runtime_identity=expected_context.get("co_alchemical_ion_runtime_identity"),
        ligand_conformer_diagnostics=expected_context.get("ligand_conformer_diagnostics"),
        stage_results=stages, analysis_context=expected_context,
    )


def run_post_analysis(args):
    output_dir = args.output
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"输出目录不存在: {output_dir}")

    log.info("进入后处理分析模式...")
    _run_provenance_path = os.path.join(output_dir, "run_provenance.json")
    _run_provenance = {}
    if os.path.exists(_run_provenance_path):
        with open(_run_provenance_path, "r", encoding="utf-8") as f:
            _run_provenance = json.load(f)
        if not isinstance(_run_provenance, dict):
            raise RuntimeError(
                f"run provenance {_run_provenance_path} is not a JSON object"
            )
    else:
        log.warning(
            "[WARN] analyze-only：找不到 %s，无法恢复原运行的协议身份"
            "（temperature/mode/decoupling 取自本次命令/预设默认值）。",
            _run_provenance_path,
        )

    # 🔑 [P1-04] 协议身份恢复：原运行是 310 K / 非 default 路线时，analyze-only
    # 若直接用本次命令的默认值，会算出错误的 kT/Boresch 修正或走错分析分支。
    # 规则：默认从 run_provenance.json 恢复原运行值；只有本次命令行**显式**
    # 传了对应 flag 才允许覆盖，且覆盖行为必须留审计记录。
    _provenance_cfg = _run_provenance.get("config") or {}
    _explicit_cli_flags = {
        token[2:].split("=", 1)[0]
        for token in sys.argv[1:]
        if token.startswith("--")
    }
    for _key in ("temperature", "mode", "decoupling"):
        _prov_value = _provenance_cfg.get(_key)
        _current_value = getattr(args, _key, None)
        if _key in _explicit_cli_flags:
            if _prov_value is not None and str(_prov_value) != str(_current_value):
                log.warning(
                    "[analyze-only 审计] 显式覆盖原运行的 %s：%s → %s"
                    "（原值来自 run_provenance.json）",
                    _key, _prov_value, _current_value,
                )
            continue
        if _prov_value is None or str(_prov_value) == str(_current_value):
            continue
        log.info(
            "[analyze-only] 从 run_provenance.json 恢复原运行的 %s=%s"
            "（本次命令默认 %s，未显式覆盖；如需覆盖请显式传 --%s）",
            _key, _prov_value, _current_value, _key,
        )
        setattr(args, _key, _prov_value)

    temp = args.temperature * unit.kelvin
    kt = (unit.MOLAR_GAS_CONSTANT_R * temp).value_in_unit(unit.kilojoule_per_mole)
    def _load_boresch_params(base_dir: str):
        for json_name in [
            "boresch_params.json",
            "boresch_auto.json",
            "boresch_simple.json",
            "boresch_fluctuation.json",
        ]:
            path = os.path.join(base_dir, json_name)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    return _sanitize_boresch_params(json.load(f))
        return None

    def _analyze_dual_leg(base_dir: str) -> Dict:
        return _analyze_dual_leg_artifacts(base_dir, temp)

    def _load_leg_result(base_dir: str) -> Dict:
        result_path = os.path.join(base_dir, "final_results.json")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"缺少腿结果文件: {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        # 🔑 [P1-12] 缺必需热力学字段/非有限/负误差/converged 不为 True 的腿结果
        # 一律拒绝汇总——统一 gate 与 abfe_pipeline cache 早退、
        # combine_binding_free_energy 共用，见 abfe_core.validate_final_leg_result。
        validate_final_leg_result(
            payload, context=f"leg result {base_dir}", source=result_path
        )
        return payload

    complex_boresch = _load_boresch_params(output_dir)
    dg_boresch = 0.0
    if complex_boresch and complex_boresch.get("force_constants") and complex_boresch.get("equilibrium_values"):
        # 🔑 之前这里计算失败只打 warning，dg_boresch 保持初始化的 0.0 继续往下走——
        # 在没有正式 final_results.json 的原始窗口回退场景（见下面
        # boresch_included_in_complex_dg=False 的分支），这个 0.0 会被直接减进
        # delta_g_bind_uncorrected 的公式，等于悄悄漏掉整项
        # restraint release 修正。已经找到 Boresch 参数（force_constants/
        # equilibrium_values 都在）说明这个体系确实需要这项修正，计算失败就必须
        # fail closed；只有从一开始就找不到 Boresch 参数（上面 if 判断为 False，
        # 即体系本身没有做 Boresch restraint）才允许 dg_boresch 维持 0.0。
        dg_boresch = calculate_boresch_analytical_correction(
            eq=complex_boresch["equilibrium_values"],
            fc=complex_boresch["force_constants"],
            T=temp,
        )

    # 追踪 dg_boresch 这一物理量本身是否已经被烘焙进 dg_complex
    # （total_delta_G_complex_kJ_mol）。这既是下游消费者判断"能不能对
    # complex_delta_G_kJ_mol 再扣一次 Boresch"的依据，也是
    # combine_binding_free_energy 决定减不减第二次的唯一开关。
    #
    # 🔑 [ATT-09] 这里原来还有一个并行的 `dg_boresch_term`（在同样两个分支里被置 0），
    # 与这个布尔量是同一件事的两份记账。公式统一进 helper 之后它已无人读取，
    # 直接删掉——留着一个"曾经控制公式、现在没人读"的变量正是日后被错误接回去的引信。
    boresch_included_in_complex_dg = False
    if args.mode == "traditional":
        complex_leg = _load_leg_result(os.path.join(output_dir, "traditional_complex"))
        solvent_leg = _load_leg_result(os.path.join(output_dir, "traditional_solvent"))
        dg_complex = float(complex_leg["delta_G_total_kJ_mol"])
        dg_solvent = float(solvent_leg["delta_G_total_kJ_mol"])
        err_complex = float(complex_leg["error_leg_kJ_mol"])
        err_solvent = float(solvent_leg["error_leg_kJ_mol"])
    elif args.decoupling == "dual_lambda":
        _complex_final_path = os.path.join(output_dir, "final_results.json")
        _solvent_final_path = os.path.join(output_dir, "solvent_leg", "final_results.json")
        if os.path.exists(_complex_final_path) and os.path.exists(_solvent_final_path):
            # ✅ 正式流程已经把 PME 自能修正、Boresch 释放等修正项烘焙进
            # total_delta_G_complex_kJ_mol 里落盘为 final_results.json；这里优先复用
            # 该权威口径，而不是用 _analyze_dual_leg 从原始窗口能量重新拼出一份可能
            # 遗漏这些修正的"影子"总量，避免 --analyze-only 与正式流程结果对不上。
            complex_leg = _load_leg_result(output_dir)
            solvent_leg = _load_leg_result(os.path.join(output_dir, "solvent_leg"))
            dg_complex = float(complex_leg["total_delta_G_complex_kJ_mol"])
            dg_solvent = float(solvent_leg["total_delta_G_complex_kJ_mol"])
            err_complex = float(complex_leg["total_error_kJ_mol"])
            err_solvent = float(solvent_leg["total_error_kJ_mol"])
            # ✅ 同下方 "else" 分支的既有约定：total_delta_G_complex_kJ_mol 已经把
            # Boresch 释放修正烘焙进去了（见 abfe_pipeline.py total_dg = dg_phys +
            # cons_correction + dg_boresch），所以循环闭合时绝不能再减一次，
            # 否则同一个 Boresch 修正会被计入两遍。
            boresch_included_in_complex_dg = True
        else:
            log.warning(
                "未找到正式 final_results.json，回退为从原始窗口能量文件重新估算 "
                "decoupling_delta_G_kJ_mol（该值不含正式流程里烘焙的 PME 自能/长程修正等项，"
                "仅供粗略核查，不等价于正式 total/APBS 组装口径）。"
            )
            complex_leg = _analyze_dual_leg(output_dir)
            solvent_leg = _analyze_dual_leg(os.path.join(output_dir, "solvent_leg"))
            recovered_boresch = complex_leg["analysis_context"]["stage_protocol_keys"]["decharging"]["payload"].get("boresch_params")
            from abfe_pipeline import _preopt_boresch_protocol_payload
            if _preopt_boresch_protocol_payload(complex_boresch) != _preopt_boresch_protocol_payload(recovered_boresch):
                raise ValueError("analyze-only Boresch release 参数与采样协议不匹配")
            dg_complex = float(complex_leg["decoupling_delta_G_kJ_mol"])
            dg_solvent = float(solvent_leg["decoupling_delta_G_kJ_mol"])
            err_complex = float(complex_leg["total_error_kJ_mol"])
            err_solvent = float(solvent_leg["total_error_kJ_mol"])
    else:
        complex_leg = _load_leg_result(output_dir)
        solvent_leg = _load_leg_result(os.path.join(output_dir, "solvent_leg"))
        dg_complex = float(complex_leg["total_delta_G_complex_kJ_mol"])
        dg_solvent = float(solvent_leg["total_delta_G_complex_kJ_mol"])
        err_complex = float(complex_leg["total_error_kJ_mol"])
        err_solvent = float(solvent_leg["total_error_kJ_mol"])
        boresch_included_in_complex_dg = True

    # 🔑 [ATT-09] 循环闭合统一走 abfe_core.combine_binding_free_energy。
    # 这条路径上 `boresch_included_in_complex_dg` 就是 helper 要的那个开关：
    # 上面三个分支已经分别判定过 complex 腿有没有烘焙释放项，helper 据此决定
    # 减不减第二次。原来这里手写 `- dg_boresch_term`，与那个布尔量是两份记账，
    # 现在合成一份。
    # 🔑 之前这里读 getattr(args, "apbs_correction_kj_mol", None)——小写 j，是
    # argparse 从 --apbs-correction-kj-mol 派生的 CLI dest 名，只在这次
    # --analyze-only 调用本身显式带了这个参数时才非 None；正式跑
    # (RunConfig.data 存的是大写 J 的 "apbs_correction_kJ_mol"，main() 里
    # config.get("apbs_correction_kJ_mol", ...) 读的是这个) 用过的值从未被
    # 读取——不带 CLI 参数重跑 --analyze-only 会静默把 APBS 修正算成 0，
    # 且不报错。正式跑一定会在早期写一份 run_provenance.json
    # (_write_run_provenance，含 config.as_dict() 的完整快照)，这里改为优先
    # 读它；只有这次 --analyze-only 调用显式重新传了 --apbs-correction-kj-mol
    # 才用它覆盖（跟 RunConfig.__init__ 里 _flag_present 的判断方式一致，不
    # 依赖 CLI 参数默认值是否恰好为 None 这种脆弱判断）。
    _argv = sys.argv[1:]
    def _flag_present(*flags):
        for flag in flags:
            if flag in _argv:
                return True
            prefix = f"{flag}="
            if any(token.startswith(prefix) for token in _argv):
                return True
        return False
    if _flag_present("--apbs-correction-kj-mol"):
        apbs_correction = float(getattr(args, "apbs_correction_kj_mol", None) or 0.0)
    else:
        apbs_correction = 0.0
        if _run_provenance:
            apbs_correction = float(
                _run_provenance.get("config", {}).get("apbs_correction_kJ_mol", 0.0) or 0.0
            )
    # 🔑 [0831issue P2] note 必须跟着值一起从 provenance 恢复。以前值被恢复、note 却在
    # 落盘处写成 `getattr(args, ...) or ""`，于是 `--analyze-only` 重跑会产出一个
    # **非零 APBS 修正却没有任何来源说明**的结果文件，违背 provenance 可追溯口径。
    # 与值同样的判据：只有本次显式传了 --apbs-correction-note 才用命令行的。
    if _flag_present("--apbs-correction-note"):
        apbs_correction_note = getattr(args, "apbs_correction_note", None) or ""
    else:
        apbs_correction_note = ""
        if _run_provenance:
            apbs_correction_note = str(
                _run_provenance.get("config", {}).get("apbs_correction_note", "") or ""
            )

    _production_qualification = _resolve_production_qualification_from_sources(
        complex_leg, solvent_leg, _run_provenance
    )
    # Charge-transfer is only analyzable when both legs carry the same,
    # evidence-backed reservoir-release correction.  Do not reconstruct it
    # from a scalar in config or infer cancellation from matching restraints.
    _reservoir_candidates = []
    _configured_reservoir = args.get("charge_transfer_reservoir_correction")
    if isinstance(_configured_reservoir, (str, os.PathLike)):
        _configured_reservoir = _load_json_object_file(
            str(_configured_reservoir), "charge-transfer reservoir correction"
        )
    if _configured_reservoir is not None:
        _reservoir_candidates.append(
            validate_charge_transfer_reservoir_correction(
                _configured_reservoir,
                temperature_k=float(args.temperature),
            )
        )
    for _source in (complex_leg, solvent_leg, _run_provenance):
        if not isinstance(_source, dict):
            continue
        _candidate = _source.get("charge_transfer_reservoir_correction")
        if _candidate is None and isinstance(_source.get("charge_protocol"), dict):
            _candidate = _source["charge_protocol"].get(
                "charge_transfer_reservoir_correction"
            )
        if _candidate is None and isinstance(_source.get("provenance"), dict):
            _candidate = _source["provenance"].get(
                "charge_transfer_reservoir_correction"
            )
        if _candidate is not None:
            _reservoir_candidates.append(
                validate_charge_transfer_reservoir_correction(
                    _candidate,
                    temperature_k=float(args.temperature),
                )
            )
    _reservoir_payload = None
    if _reservoir_candidates:
        _reservoir_payload = _reservoir_candidates[0]
        _canonical_reservoir = json.dumps(
            _reservoir_payload, sort_keys=True, separators=(",", ":")
        )
        if any(
            json.dumps(item, sort_keys=True, separators=(",", ":"))
            != _canonical_reservoir
            for item in _reservoir_candidates[1:]
        ):
            raise RuntimeError(
                "analyze-only 的 charge-transfer reservoir correction 在腿/"
                "provenance 之间不一致；拒绝混合 artifact"
            )
    _treatment_values = []
    for _source in (complex_leg, solvent_leg, _run_provenance):
        if not isinstance(_source, dict):
            continue
        _treatment_values.append(_source.get("charge_treatment"))
        if isinstance(_source.get("charge_protocol"), dict):
            _treatment_values.append(_source["charge_protocol"].get("charge_treatment"))
        if isinstance(_source.get("provenance"), dict):
            _treatment_values.append(_source["provenance"].get("charge_treatment"))
    _is_charge_transfer = any(
        str(value or "").strip().lower()
        == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
        for value in _treatment_values
    )
    # 🔑🔑 [2026-09-18] 未闭合 charge-transfer 的**第四道**闸：`--analyze-only`
    # 重新分析既有产物。诊断模式下同样要放行，否则一个诊断 run 跑完之后连
    # 重新分析都做不了。代价与汇总那道一致 —— reservoir 项按 0 计。
    _reservoir_omitted = False
    if _is_charge_transfer and _reservoir_payload is None:
        if not bool(args.get("run_unclosed_charge_transfer_diagnostics", False)):
            raise RuntimeError(
                "analyze-only 的 charge-transfer 结果缺少已验证的双腿 reservoir correction；"
                "拒绝把 tethered-carrier endpoint 当作闭合 ABFE"
                "（只要诊断数值 ⟹ `--run-unclosed-charge-transfer-diagnostics`；"
                "届时 reservoir 项按 0 计、产物标记不得发布。）"
            )
        _reservoir_omitted = True
        log.warning(
            "[WARN] [诊断模式] analyze-only 缺少 carrier reservoir correction，"
            "该项**按 0 计** ⟹ ΔG_bind 少了一整项，不得发布。"
        )
    _reservoir_delta = (
        float(_reservoir_payload["delta_G_bind_correction_kJ_mol"])
        if _reservoir_payload is not None
        else 0.0
    )
    _reservoir_error = (
        float(_reservoir_payload["error_kJ_mol"])
        if _reservoir_payload is not None
        else 0.0
    )
    _assert_matching_result_constraint_identity(
        complex_leg,
        solvent_leg,
        context="analyze-only ABFE 热力学循环",
    )
    cycle = combine_binding_free_energy(
        dg_complex_kJ_mol=dg_complex,
        dg_solvent_kJ_mol=dg_solvent,
        err_complex_kJ_mol=err_complex,
        err_solvent_kJ_mol=err_solvent,
        dg_boresch_kJ_mol=dg_boresch,
        boresch_already_included_in_complex=bool(boresch_included_in_complex_dg),
        apbs_correction_kJ_mol=apbs_correction,
        charge_transfer_reservoir_correction_kJ_mol=_reservoir_delta,
        charge_transfer_reservoir_error_kJ_mol=_reservoir_error,
        complex_conformer_summary=complex_leg.get("ligand_conformer_diagnostics"),
        solvent_conformer_summary=solvent_leg.get("ligand_conformer_diagnostics"),
    )
    delta_g_bind_uncorrected = cycle["delta_G_bind_uncorrected_kJ_mol"]
    final_dg = cycle["delta_G_bind_kJ_mol"]
    final_err = cycle["total_error_kJ_mol"]
    log.info("ABFE 后处理完成: ΔG_bind = %.2f ± %.2f kJ/mol", final_dg, final_err)

    result = {
        "complex_leg_delta_G_kJ_mol": float(dg_complex),
        "solvent_leg_delta_G_kJ_mol": float(dg_solvent),
        # ✅ 报告 Boresch 修正的真实物理量级 (dg_boresch)，而不是公式里实际被减掉
        # 的那一项（已内含时为 0，见 cycle["boresch_term_subtracted_kJ_mol"]）。
        # 同时显式标记它是否已经烘焙进 complex_leg_delta_G_kJ_mol，避免下游
        # 消费者误以为这里恒为独立可加项而对 complex_delta_G 二次扣减。
        "boresch_correction_kJ_mol": float(dg_boresch),
        # 🔑 [0831issue P2] 与 run_traditional_mode 的字段集对齐，让同一批工件经两条
        # 路径产出的 JSON 可以逐字段比对。这里的 Boresch 修正由
        # `calculate_boresch_analytical_correction` 直接给出标量，没有伴随的模型误差
        # 估计（run_traditional_mode 那边来自 `apply_boresch_correction` 的 "error"），
        # 所以显式写 None + 原因，而不是用 0.0 假装"误差为零"。
        "boresch_correction_error_kJ_mol": None,
        "boresch_correction_error_unavailable_reason": (
            "analyze-only 从解析公式重算 Boresch 释放项，不经过 apply_boresch_correction，"
            "拿不到其模型误差估计；该项按约定本就不并入 total_error_kJ_mol"
        ),
        "constraint_identity": complex_leg.get("constraint_identity"),
        "charge_transfer_reservoir_correction": _reservoir_payload,
        # 诊断模式下 reservoir 项按 0 计 ⟹ 这个 ΔG_bind 少了一整项（同汇总处）。
        **({"charge_transfer_reservoir_correction_omitted": True,
            "must_not_report_delta_g_bind": True} if _reservoir_omitted else {}),
        "boresch_correction_already_included_in_complex_delta_G": bool(boresch_included_in_complex_dg),
        "boresch_correction_note": (
            "boresch_correction_kJ_mol 已经烘焙进 complex_leg_delta_G_kJ_mol，"
            "且已完整体现在 delta_G_bind_kJ_mol 中；下游不要再对二者中任何一个二次扣减 Boresch 修正。"
            if boresch_included_in_complex_dg else
            "boresch_correction_kJ_mol 未包含在 complex_leg_delta_G_kJ_mol 中，"
            "而是作为独立项已经减到 delta_G_bind_kJ_mol 里；下游不要再对 delta_G_bind_kJ_mol 二次扣减。"
        ),
        "delta_G_bind_uncorrected_kJ_mol": float(delta_g_bind_uncorrected),
        "apbs_correction_kJ_mol": float(apbs_correction),
        # [0831issue P2] 与 apbs_correction 同源解析（见上方 _flag_present 分支），
        # 不再无条件回落到本次 CLI 参数。
        "apbs_correction_note": apbs_correction_note,
        "delta_G_bind_kJ_mol": float(final_dg),
        "delta_G_bind_kcal_mol": float(final_dg / 4.184),
        "total_error_kJ_mol": final_err,
        "thermodynamic_cycle_terms": cycle,
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "decoupling": args.decoupling,
    }
    if _production_qualification:
        result.update(_production_qualification)
        result["production_qualification"] = dict(_production_qualification)
    out_path = os.path.join(output_dir, "final_results_postprocess.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    log.info("结果已保存: %s", out_path)

def run_prepare_command(args):
    """简化版预处理命令：生成系统缓存 + 可选 Boresch / DEXP 估算"""
    log.info("🔧 执行 prepare 子命令")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. 构建系统（与 main 中首次构建一致）
    include_dir = find_gmx_include_dir(args.gmx_path)
    cache_identity = _main_cache_identity(
        args.gro, args.top, args.ligand, include_dir
    )
    system, topology, positions, box_vectors, ligand_indices = build_system_from_gromacs(
        args.gro, args.top, args.ligand, include_dir
    )
    diagnose_14_scaling(system)
    save_native_system(
        output_dir,
        system,
        topology,
        ligand_indices,
        positions,
        box_vectors,
        cache_identity=cache_identity,
    )

    # 2. 从缓存重载（保证一致性）
    system, topology, positions, box_vectors, ligand_indices = load_native_system(
        output_dir, gro_file=args.gro, top_file=args.top, gmx_include_dir=include_dir
    )

    # 3. PBC 居中（与主流程完全一致）
    if hasattr(positions, "value_in_unit"):
        pos_nm = positions.value_in_unit(unit.nanometer)
    else:
        pos_nm = positions
    pos_nm = np.asarray(pos_nm, dtype=np.float64)
    if pos_nm.ndim == 1:
        pos_nm = pos_nm.reshape(-1, 3)
    positions = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in pos_nm] * unit.nanometer
    positions, box_vectors = center_system_rigidly(positions, box_vectors, ligand_indices)

    # 4. 初始化 Pipeline（只做预平衡估算用）
    # 这里也走膜协议：prepare 会调 pre_equilibrate()，膜体系必须用膜 barostat，
    # 否则 prepare 产出的预平衡轨迹是各向同性缩放下的，APL 已经跑掉了。
    pipeline = ABFEPipeline(
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        temperature=args.temperature,
        output_dir=output_dir,
        platform_name=args.platform,
        environment_type=getattr(args, "system_type", None),
        membrane=(_membrane_overrides_from_args(args) or None),
        confirm_soluble_with_lipids=bool(
            getattr(args, "confirm_soluble_with_lipids", False)
        ),
        dispersion_protocol=getattr(args, "dispersion_protocol", None),
        forcefield_family=getattr(args, "forcefield_family", None),
        # [B3] 净电荷路线：两条腿必须传同一个（§6.1）。它只决定 co-ion 身份怎么冻结，
        # 中性配体（当前生产体系）根本不进 co-ion 分支，行为逐位不变。
        charge_treatment=getattr(args, "charge_treatment", None),
    )

    # 5. 预平衡生成轨迹
    equil_data = pipeline.pre_equilibrate(
        n_steps=args.n_steps,
        save_traj=True,
        platform_name=args.platform,
    )
    traj_file = equil_data["trajectory_file"]

    # 6. Boresch 估算（默认 fluctuation 模式）
    if args.save_boresch:
        import mdtraj as md
        traj = md.load(traj_file, top=_boresch_mdtraj_topology(
            pipeline, runtime_gromacs_top(output_dir, args.top), include_dir))
        estimator = GeometricRestraintEstimator(
            temperature=args.temperature, allow_geometric_bond_fallback=False
        )
        boresch = estimator.estimate_from_trajectory(
            traj, args.ligand,
            output_path=os.path.join(output_dir, args.save_boresch)
        )
        log.info("Boresch 参数已保存至 %s", args.save_boresch)

    log.info("[OK] prepare 完成，文件已输出至 %s", output_dir)

