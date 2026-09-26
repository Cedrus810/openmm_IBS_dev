# ---------------------------------------------------------------------------
# 配置管理
# ---------------------------------------------------------------------------
# membrane.* 的 CLI 子键 → argparse 属性名。RunConfig 与 run_prepare_command 共用，
# 不各写一份（`--membrane-*` 的 argparse 默认值全是 None，所以 "is not None"
# 等价于"用户显式给了"，不需要再去翻 argv）。
_MEMBRANE_CLI_ARG_NAMES = {
    "normal_axis": "membrane_normal_axis",
    "surface_tension_bar_nm": "membrane_surface_tension_bar_nm",
    "xy_mode": "membrane_xy_mode",
    "z_mode": "membrane_z_mode",
    "barostat_frequency": "membrane_barostat_frequency",
}


def _membrane_overrides_from_args(args) -> Dict:
    """从 argparse 命名空间取出显式给出的 membrane.* 子键。"""
    overrides = {}
    for key, attr in _MEMBRANE_CLI_ARG_NAMES.items():
        value = getattr(args, attr, None)
        if value is not None:
            overrides[key] = value
    return overrides


def _load_config(config_path: str) -> dict:
    """加载配置文件，支持 .json / .yaml / .yml 格式"""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"显式配置文件不存在或不是普通文件: {config_path}")
    ext = os.path.splitext(config_path)[1].lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except ImportError:
            raise ImportError(
                "读取 .yaml 配置文件需要 pyyaml，请运行: pip install pyyaml\n"
                "或使用 .json 格式的配置文件。"
            )
    else:
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(
            f"配置文件顶层必须是 JSON/YAML 对象，实际为 {type(config).__name__}: "
            f"{config_path}"
        )
    return config


def _load_json_object_file(path: str, label: str) -> Dict:
    """Load an explicitly requested JSON object without silent fallback."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"显式 {label} 文件不存在或不是普通文件: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取或解析 {label} 文件 {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(
            f"{label} 文件顶层必须是 JSON 对象，实际为 {type(value).__name__}: {path}"
        )
    return value


def _load_dexp_params_fail_closed(
    potential_type: str,
    params_path: Optional[str],
) -> Optional[Dict]:
    """Load the narrow production DEXP contract, or return None for other potentials."""
    if str(potential_type).strip().lower() != "dexp":
        return None
    if not params_path:
        raise ValueError(
            "potential='dexp' 必须显式提供 --dexp-params JSON；"
            "拒绝静默使用默认参数。"
        )
    dexp_dict = _load_json_object_file(params_path, "DEXP 参数")
    return DEXPSurrogatePotential.from_dict(dexp_dict).get_parameters_dict()


def _coerce_positive_seed(raw: Any, origin: str) -> int:
    """把一个 seed 输入解析成正整数，失败一律 SystemExit（启动期硬门）。

    非整数值（如 `1.5`）必须**拒绝**而不是 `int()` 静默截断：seed 决定整个独立重复
    的随机流身份，"我写了 1.5、实际用了 1" 这种静默改写会让 provenance 里的 seed
    与真正消费的 seed 不是同一个数。
    """
    if isinstance(raw, bool):
        raise SystemExit(f"{origin} 必须是正整数，收到 {raw!r}")
    try:
        seed = int(raw)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{origin} 必须是正整数，收到 {raw!r}") from exc
    # 截断检测：float/带小数的字符串都会在这里被挡住。
    try:
        if float(raw) != float(seed):
            raise SystemExit(
                f"{origin} 必须是整数，收到 {raw!r}（int() 会静默截断成 {seed}，"
                "而 seed 决定随机流身份，不允许静默改写）"
            )
    except (TypeError, ValueError):
        raise SystemExit(f"{origin} 必须是正整数，收到 {raw!r}") from None
    if seed <= 0:
        raise SystemExit(f"{origin} 必须是正整数，收到 {raw!r}")
    return seed


def _resolve_repeat_seed(
    args: argparse.Namespace, config: "RunConfig"
) -> Tuple[Optional[int], Optional[str]]:
    """[RELEASE_READINESS R1] 解析唯一的 repeat seed，返回 `(seed, source)`。

    优先级 **CLI `--seed` > 环境变量 `ABFE_RANDOM_SEED` > 配置文件 `repeat_seed`**，
    与 `RunConfig` 自己"命令行 > 配置文件 > 预设"的约定同向。三者都没给则返回
    `(None, None)`：**刻意保留未显式设置 seed 时的历史行为**，不为旧 resume 偷补新
    seed、不改动任何既有随机流。

    ⚠️ [2026-09-01] 第三档**不静默生效**：只有配置文件写了 `repeat_seed`、而 CLI 与
    env 都没给时，本函数 `SystemExit` 要求显式确认，而不是直接采用。理由见该分支注释
    —— 那等于在用户没表态的情况下改掉一个既有 output 目录的随机流。
    所以实际语义是「CLI/env 直接生效；config 需确认；都没有则保留历史行为」。

    这个函数是唯一入口：解析结果由 `main()` 回写 `config.data["repeat_seed"]`，
    所以 traditional 路径（读 `config.get("repeat_seed")`）与 IBS 路径（读局部
    `_repeat_seed`）从此拿到的是同一个值，配置快照也不可能再声明一个管线没收到的 seed。
    """
    cli_seed = getattr(args, "seed", None)
    if cli_seed is not None:
        return _coerce_positive_seed(cli_seed, "--seed"), "cli:--seed"

    env_raw = os.environ.get("ABFE_RANDOM_SEED")
    if env_raw not in (None, ""):
        return (
            _coerce_positive_seed(env_raw, "ABFE_RANDOM_SEED"),
            "env:ABFE_RANDOM_SEED",
        )

    cfg_raw = config.get("repeat_seed")
    if cfg_raw not in (None, ""):
        # 🔑 [2026-09-01] 配置文件里的 repeat_seed **不再静默采用**，要求显式确认。
        #
        # R1 要求"建立唯一 resolved seed、明确优先级"，但同一段还有一句同等重要的：
        # 「保留未显式设置 seed 的历史行为；**不要为旧 resume 偷补新 seed 或更改随机流**。」
        # 把 config 里的 repeat_seed 接进管线本身是对的（以前 IBS 管线根本收不到它，
        # 配置快照会声明一个管线没用过的 seed）；但对一个**过去实际靠 env 或什么都不靠
        # 跑出来的 output 目录**，现在突然开始用 config 里那个值，等于在用户没表态的
        # 情况下改掉了它的随机流 —— 而随机流决定独立重复的身份。
        #
        # 所以这条路径改成 fail-closed：报清楚检测到什么、影响是什么、两条出路分别是
        # 什么，由人来选。CLI `--seed` 与 `ABFE_RANDOM_SEED` 都是**显式意图**，不受影响。
        _cfg_seed = _coerce_positive_seed(cfg_raw, "config repeat_seed")
        raise SystemExit(
            f"配置文件里写了 repeat_seed={_cfg_seed}，但命令行 --seed 和环境变量 "
            "ABFE_RANDOM_SEED 都没给。\n"
            "拒绝静默采用配置文件里的值：这个 output 目录过去可能是靠 env 或不带 seed "
            "跑出来的，现在改用 config 的 seed 会改变随机流，而随机流决定"
            "「独立重复」的身份（验收门要求 3 个独立重复且跨重复 σ ≤ 1 kcal/mol）。\n"
            "两条出路，按你的真实意图选一条：\n"
            f"  · 确实要用这个 seed → 显式加 --seed {_cfg_seed}（或设 "
            f"ABFE_RANDOM_SEED={_cfg_seed}）\n"
            "  · 只想保持原样重跑 → 从配置文件里删掉 repeat_seed 键\n"
            "（CLI/env 给了 seed 时本检查不触发；三者都不给时保留历史随机流，也不触发。）"
        )

    return None, None


class RunConfig:
    """统一运行时配置，优先级：命令行 > 配置文件 > 预设"""

    def __init__(self, args: argparse.Namespace):
        argv = sys.argv[1:]

        def _flag_present(*flags: str) -> bool:
            for flag in flags:
                if flag in argv:
                    return True
                prefix = f"{flag}="
                if any(token.startswith(prefix) for token in argv):
                    return True
            return False

        # 1. 载入预设
        preset = PRESET_CONFIGS.get(args.preset, PRESET_CONFIGS["production"]).copy()

        # 2. 合并配置文件（支持 .json / .yaml / .yml）
        #
        # 2026-08-24：这里曾经加过一个 unknown-key 校验（不认识的字段直接
        # raise ValueError），本意是抓 typo；已撤掉——它是一个新增的、会在启动期
        # 直接炸掉 resume 的硬门，用户明确要求关掉，不要让任何新加的校验有阻断
        # resume 的风险。恢复成最简单的直接合并，不做字段名校验。
        if hasattr(args, "config") and args.config:
            file_conf = _load_config(args.config)
            for k, v in file_conf.items():
                if k.startswith("_"):
                    continue
                preset[k] = v
            log.info("已合并配置文件: %s", args.config)

        # 🔑 [2026-09-18] **诊断模式只认命令行，且无条件覆盖配置文件。**
        # 这里是唯一一处**不带 `if _flag_present`** 的赋值，故意的：它要把配置
        # 文件里可能写着的同名键**擦掉**。理由是这个开关放行的是一条物理上
        # 未闭合的路线（`--run-unclosed-charge-transfer-diagnostics`），
        # 一旦它能沉进 config 就会被 resume/复制配置带着到处跑，而产物只是
        # "带标记"不是"被拦住"。要求每次运行显式指定，代价只有多打一个参数。
        preset["run_unclosed_charge_transfer_diagnostics"] = _flag_present(
            "--run-unclosed-charge-transfer-diagnostics"
        )

        # 3. 仅当命令行显式提供参数时才覆盖配置文件，避免 parser 默认值反向污染配置。
        if _flag_present("--resume"):
            preset["resume"] = bool(args.resume)
        if _flag_present("--reset"):
            preset["reset"] = bool(args.reset)
        if _flag_present("--n-steps-per-window"):
            preset["n_steps_per_window"] = args.n_steps_per_window
        if _flag_present("--steps-per-update"):
            preset["steps_per_update"] = args.steps_per_update
        if _flag_present("--n-equil-steps"):
            preset["n_equil_steps"] = args.n_equil_steps
        if _flag_present("--n-states-per-stage"):
            preset["stage1_n_states"] = args.n_states_per_stage
            preset["stage2_n_states"] = args.n_states_per_stage
        # 独立覆盖，优先级高于 --n-states-per-stage（去电荷/去VDW通常需要不同的状态数）
        if _flag_present("--stage1-n-states"):
            preset["stage1_n_states"] = args.stage1_n_states
        if _flag_present("--stage2-n-states"):
            preset["stage2_n_states"] = args.stage2_n_states
        # 独立于探针网格密度——最终生产窗口数，见 abfe_preoptimizer 里
        # VANISHING_FINAL_STATE_COUNT 的说明；不传就是 None，pipeline 内部用
        # optimize_stage2_vanishing 自己的默认值（23），行为不变。
        if _flag_present("--stage2-final-n-states"):
            preset["stage2_final_n_states"] = args.stage2_final_n_states
        if _flag_present("--stage2-refine-extra-points"):
            preset["stage2_refine_extra_points_per_segment"] = args.stage2_refine_extra_points
        if _flag_present("--stage2-window-min-states"):
            preset["stage2_window_min_states"] = args.stage2_window_min_states
        if _flag_present("--stage2-window-max-states"):
            preset["stage2_window_max_states"] = args.stage2_window_max_states
        if _flag_present("--stage2-free-energy-densify-points"):
            preset["stage2_free_energy_densify_points"] = (
                args.stage2_free_energy_densify_points
            )
        if _flag_present("--stage2-production-budget-steps"):
            preset["stage2_production_budget_steps"] = args.stage2_production_budget_steps
        if _flag_present("--stage2-max-production-blocks-per-window"):
            preset["stage2_max_production_blocks_per_window"] = (
                args.stage2_max_production_blocks_per_window
            )
        if _flag_present("--max-path-insertions"):
            preset["max_path_insertions"] = args.max_path_insertions
        if _flag_present("--temperature"):
            preset["temperature"] = args.temperature
        if _flag_present("--solvent-ionic-strength-molar"):
            preset["solvent_ionic_strength_molar"] = (
                args.solvent_ionic_strength_molar
            )
        if _flag_present("--platform"):
            preset["platform"] = args.platform
        if _flag_present("--output"):
            preset["output"] = args.output
        if _flag_present("--mode"):
            preset["mode"] = args.mode
        if _flag_present("--decoupling"):
            preset["decoupling"] = args.decoupling
        if _flag_present("--potential"):
            preset["potential"] = args.potential
        if _flag_present("--dexp-params"):
            preset["dexp_params"] = args.dexp_params
        if _flag_present("--remd-backend"):
            preset["remd_backend"] = args.remd_backend
        if _flag_present("--decharge-method"):
            preset["decharge_method"] = args.decharge_method
        if _flag_present("--gro"):
            preset["gro"] = args.gro
        if _flag_present("--top"):
            preset["top"] = args.top
        if _flag_present("--ligand"):
            preset["ligand"] = args.ligand
        if _flag_present("--ligand-xml"):
            preset["ligand_xml"] = args.ligand_xml
        if _flag_present("--gmx-path"):
            preset["gmx_path"] = args.gmx_path
        if _flag_present("--torsion-params"):
            preset["torsion_params"] = args.torsion_params
        if _flag_present("--boresch", "--no-boresch"):
            preset["boresch"] = args.boresch
        if _flag_present("--boresch-source"):
            preset["boresch_source"] = args.boresch_source
        if _flag_present("--boresch-anchors"):
            preset["boresch_anchors"] = args.boresch_anchors
        if _flag_present("--boresch-orb"):
            preset["boresch_orb"] = args.boresch_orb
        if _flag_present("--boresch-batch"):
            preset["boresch_batch"] = args.boresch_batch
        if _flag_present("--boresch-select"):
            preset["boresch_select"] = args.boresch_select
        if _flag_present("--enable-early-stop"):
            preset["enable_early_stop"] = bool(args.enable_early_stop)
        if _flag_present("--disable-warmup"):
            preset["enable_gradual_warmup"] = False
        elif _flag_present("--enable-gradual-warmup"):
            preset["enable_gradual_warmup"] = True
        if _flag_present("--warmup-steps"):
            preset["warmup_steps"] = args.warmup_steps
        if _flag_present("--rebalance-steps"):
            preset["rebalance_steps"] = args.rebalance_steps
        if _flag_present("--skip-rebalance"):
            preset["skip_rebalance"] = bool(args.skip_rebalance)
        if _flag_present("--n-workers"):
            preset["n_workers"] = args.n_workers
        if _flag_present("--parallel-stages"):
            preset["parallel_stages"] = bool(args.parallel_stages)
        if _flag_present("--n-lambda"):
            preset["n_lambda"] = args.n_lambda
        if _flag_present("--apbs-correction-kj-mol"):
            preset["apbs_correction_kJ_mol"] = args.apbs_correction_kj_mol
        if _flag_present("--apbs-correction-note"):
            preset["apbs_correction_note"] = args.apbs_correction_note
        if _flag_present("--charging-rerun-dir"):
            preset["charging_rerun_dir"] = args.charging_rerun_dir
        if _flag_present("--attachment-rerun-dir"):
            preset["attachment_rerun_dir"] = args.attachment_rerun_dir
        if _flag_present("--attachment-n-steps-per-state"):
            preset["attachment_n_steps_per_state"] = int(
                args.attachment_n_steps_per_state
            )
        if _flag_present("--attachment-n-seeds"):
            preset["attachment_n_seeds"] = int(args.attachment_n_seeds)
        if _flag_present("--attachment-lambdas"):
            preset["attachment_lambdas"] = [
                float(x) for x in str(args.attachment_lambdas).split(",") if x.strip()
            ]
        if _flag_present("--charging-max-resident-contexts"):
            preset["charging_max_resident_contexts"] = (
                args.charging_max_resident_contexts
            )
        if _flag_present("--only-complex-charging"):
            preset["only_complex_charging"] = bool(args.only_complex_charging)
        if _flag_present("--only-boresch-attachment"):
            preset["only_boresch_attachment"] = bool(args.only_boresch_attachment)
        if _flag_present("--only-ligand-in-water"):
            preset["only_ligand_in_water"] = bool(args.only_ligand_in_water)

        # The formal product has one user-facing boolean.  Keep the temporary
        # EXP-030 key as a read-only compatibility alias for old config files,
        # but never expose it as a second runtime switch.  If both spellings
        # are present, an actual disagreement is a hard configuration error.
        if _flag_present(
            "--outer-lambda-local-residual-ibs",
            "--no-outer-lambda-local-residual-ibs",
        ):
            preset["outer_lambda_local_residual_ibs"] = bool(
                args.outer_lambda_local_residual_ibs
            )
        if getattr(args, "element_coverage_model", None) is not None:
            preset["element_coverage_model"] = str(args.element_coverage_model)
        if getattr(args, "outer_lambda_resource_manifest", None) is not None:
            preset["outer_lambda_resource_manifest"] = str(
                args.outer_lambda_resource_manifest
            )

        # ---- 膜体系（B1）。配置键按 memtodolist §3.1：顶层 system_type + 嵌套 membrane.* ----
        if _flag_present("--system-type"):
            preset["system_type"] = args.system_type
        if _flag_present("--charge-treatment"):
            preset["charge_treatment"] = args.charge_treatment
        if _flag_present("--charge-transfer-reservoir-correction"):
            preset["charge_transfer_reservoir_correction"] = (
                args.charge_transfer_reservoir_correction
            )
        if _flag_present("--co-alchemical-ion"):
            preset["co_alchemical_ion"] = args.co_alchemical_ion
        if _flag_present("--apbs-evidence"):
            preset["apbs_evidence"] = args.apbs_evidence
        if _flag_present("--dispersion-protocol"):
            preset["dispersion_protocol"] = args.dispersion_protocol
        if _flag_present("--forcefield-family"):
            preset["forcefield_family"] = args.forcefield_family
        if _flag_present("--membrane-input-declaration"):
            preset["membrane_input_declaration"] = args.membrane_input_declaration
        if _flag_present("--force-switch-deviation-evidence"):
            preset["force_switch_deviation_evidence"] = (
                args.force_switch_deviation_evidence
            )
        if _flag_present("--membrane-quality-gate"):
            preset["membrane_quality_gate"] = args.membrane_quality_gate
        if _flag_present("--confirm-soluble-with-lipids"):
            preset["confirm_soluble_with_lipids"] = bool(
                args.confirm_soluble_with_lipids
            )
        _membrane_overrides = _membrane_overrides_from_args(args)
        if _membrane_overrides:
            # 命令行只覆盖显式给出的子键，不清空配置文件里的其余 membrane.* 设置。
            merged_membrane = dict(preset.get("membrane") or {})
            merged_membrane.update(_membrane_overrides)
            preset["membrane"] = merged_membrane

        _product_key_explicit = "outer_lambda_local_residual_ibs" in preset
        defaults = {
            "resume": False,
            "reset": False,
            "temperature": 300.0,
            "platform": "CUDA",
            "output": "./output",
            "mode": "ibs",
            "decoupling": "dual_lambda",
            "potential": "softcore",
            "decharge_method": "pme",
            "boresch_batch": 0,
            "boresch_select": 1,
            "enable_early_stop": False,
            "enable_gradual_warmup": True,
            "warmup_steps": 500000,
            "rebalance_steps": 50000,
            "skip_rebalance": False,
            "parallel_stages": False,
            "n_lambda": 12,
            "apbs_correction_kJ_mol": 0.0,
            "apbs_correction_note": "",
            "solvent_ionic_strength_molar": DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR,
            "only_complex_charging": False,
            "charging_rerun_dir": None,
            "charging_max_resident_contexts": None,
            # PLAN_openmm_8_6_remd_backend.md §3 的目标默认值。
            "remd_backend": free_energy_engine.REMD_BACKEND_DEFAULT,
            # [P1-17] Boresch attachment 腿 A′→A
            "only_boresch_attachment": False,
            # [LR-04] 只跑溶剂腿的调试入口，不产出 ΔG_bind
            "only_ligand_in_water": False,
            "attachment_rerun_dir": None,
            "attachment_lambdas": None,
            "attachment_n_steps_per_state": 250000,
            "attachment_equil_steps_per_state": 50000,
            "attachment_steps_per_sample": 1000,
            "attachment_seed": 20260728,
            "attachment_n_seeds": 1,
            # 膜功能默认关闭（§7.7）：不声明时与改动前逐位一致。
            "system_type": None,
            "membrane": None,
            "confirm_soluble_with_lipids": False,
            # §9 质量门默认 enforce（严格）。
            "membrane_quality_gate": None,
            # 净电荷协议默认由配体净电荷推导（§1.2 生产默认值）。
            "charge_treatment": None,
            "charge_transfer_reservoir_correction": None,
            "co_alchemical_ion": None,
            "apbs_evidence": None,
            # LJ/色散：soluble 不声明 = legacy_uniform_density_lrc（行为不变）。
            "dispersion_protocol": None,
            "forcefield_family": None,
            "force_switch_deviation_evidence": None,
            "membrane_input_declaration": None,
            # Product switch: the legacy path remains byte-for-byte disabled.
            "outer_lambda_local_residual_ibs": False,
            "element_coverage_model": None,
            "outer_lambda_resource_manifest": None,
        }
        for key, value in defaults.items():
            preset.setdefault(key, value)

        residual_key = "residual_sampling_enabled"
        product_key = "outer_lambda_local_residual_ibs"
        legacy_present = residual_key in preset
        product_present = _product_key_explicit
        if legacy_present:
            legacy_value = preset[residual_key]
            if not isinstance(legacy_value, bool):
                raise TypeError(f"{residual_key} 必须是 bool")
        if product_present:
            product_value = preset[product_key]
            if not isinstance(product_value, bool):
                raise TypeError(f"{product_key} 必须是 bool")
        if legacy_present and product_present and bool(preset[residual_key]) != bool(
            preset[product_key]
        ):
            raise ValueError(
                "outer_lambda_local_residual_ibs 与旧版 residual_sampling_enabled "
                "同时存在但取值冲突；拒绝继续"
            )
        if legacy_present and not product_present:
            preset[product_key] = bool(preset[residual_key])
        # Do not let the temporary name become a second user-visible config
        # field or be consumed by downstream code.
        preset.pop(residual_key, None)

        # 复合物腿默认启用 Boresch；若用户未显式指定来源，则默认走自动估算。
        if preset.get("boresch") is None:
            preset["boresch"] = True
        if preset.get("boresch_source") in (None, "", "traditional") and preset.get("boresch", False):
            if getattr(args, "boresch_source", None) is None and not preset.get("boresch_anchors"):
                preset["boresch_source"] = "auto"

        self.data = preset
        self.args = args  # 保留原始 args 供其它函数使用

    def __getattr__(self, item):
        # 优先从配置数据获取
        if item in self.data:
            return self.data[item]
        # 其次从命令行参数获取
        return getattr(self.args, item, None)

    def get(self, key, default=None):
        """兼容字典式获取。

        🔑 [2026-09-09] argparse 的 dest **总是存在**，没给参数时值是 `None`，
        所以原来的 `getattr(self.args, key, default)` 会返回 `None` 而不是调用方
        写的 `default` —— 调用方的默认值形同虚设。

        实例：`config.get("n_equil_steps", 5_000_000)`（runabfe 里 7 处）在既没有
        `--n-equil-steps` 也没有配置文件 `n_equil_steps` 时返回 `None`，
        随后 `pre_equilibrate(n_steps=None)` 在步数运算上 TypeError。
        仓库自带的 abfe_config.json 写了这个键，所以只有纯 CLI / 手写配置会踩到。

        注意区分两种 None：**配置文件里显式写 null**（`key in self.data` 为真）
        仍然如实返回 None —— 那是用户明确表达的"不设"，不能被 default 顶掉。
        只有"argparse 侧压根没给"才回落到 default。
        """
        if key in self.data:
            return self.data[key]
        value = getattr(self.args, key, None)
        return default if value is None else value

    def as_dict(self) -> Dict:
        """Return a JSON-serializable snapshot of the resolved runtime config."""
        return dict(self.data)


def _write_run_provenance(
    output_dir: str,
    config: RunConfig,
    system: Optional[openmm.System] = None,
    topology: Optional[app.Topology] = None,
    positions=None,
    barostat_protocol: Optional[Dict] = None,
    charge_protocol: Optional[Dict] = None,
    dispersion_protocol: Optional[Dict] = None,
    forcefield_family: Optional[Dict] = None,
    membrane_input: Optional[Dict] = None,
    pairs_conversion: Optional[Dict] = None,
    membrane_quality_gate_mode: Optional[str] = None,
    co_alchemical_ions: Optional[Dict] = None,
) -> Dict:
    provenance = _collect_pipeline_provenance(
        config=config.as_dict(),
        system=system,
        topology=topology,
        positions=positions,
        command_line=sys.argv,
    )
    # memtodolist §10：system_type / barostat_protocol 必须落进 provenance，
    # 且 §3.2 要求 XY/Z 模式与膜法向可审计。这里写的是**已解析**的协议，
    # 不是原始配置字段——原始字段已在 provenance["config"] 里，两者可对照。
    if barostat_protocol is not None:
        provenance["barostat_protocol"] = barostat_protocol
        provenance["system_type"] = barostat_protocol.get("system_type")
    # memtodolist §10：charge_treatment / ligand_net_charge / coion_identity /
    # apbs_applicable-applied 必须落盘；§1.2 要求"最终说明"能直接回答电荷怎么处理。
    if dispersion_protocol is not None:
        provenance["dispersion_protocol"] = dispersion_protocol
    if forcefield_family is not None:
        provenance["forcefield_family"] = forcefield_family
    if pairs_conversion is not None:
        provenance["gromacs_pairs_funct2_conversion"] = pairs_conversion
    # §3.3 / §10：膜输入核对结果（含实测叶片数/水/离子计数与声明来源）落盘。
    if membrane_quality_gate_mode is not None:
        # advisory 放行过的运行必须在 provenance 里认得出来——否则事后无从判断
        # 那个 ΔG_bind 有没有生产资格。
        provenance["membrane_quality_gate_mode"] = membrane_quality_gate_mode
    if membrane_input is not None:
        provenance["membrane_input"] = membrane_input
        provenance["membrane_composition"] = (membrane_input.get("declared") or {}).get(
            "membrane_composition"
        )
    # §13：阈值必须进 provenance，否则"当时用的哪套阈值"无从追溯。
    provenance["acceptance_thresholds"] = acceptance_thresholds_payload()

    # 🔑 [2026-09-02] 软核族 / alpha 约定必须进**顶层** provenance。
    #
    # 缺这一条已经真的害过人：2026-09-02 拿独立参考真值给 stage2 做逐窗口误差
    # 归因时，必须知道生产的 softcore 指数才能判断"中间态是否同一个哈密顿量"
    # （端点与指数无关，中间态不是）。`run_provenance.json` 里查不到，只能翻某个
    # 窗口的 `convergence.json.stage_protocol_key.payload.aces_softcore_params`
    # 才挖出 `power_lj=[2,2]`。而 abfe_core 里那条注释（ALPHA_CONVENTION 处）
    # 明写"写进 get_parameters_dict() → 进入协议指纹"，读起来像顶层可查。
    #
    # 记的是**类级约定 + 默认构造值**：alpha 数值可能被 optimize_alpha(n) 按态数
    # 调整，所以逐 stage 的**有效**值仍以各 stage 的
    # `stage_protocol_key.payload.aces_softcore_params` 为准 —— 这里显式给出指针，
    # 不在顶层复制一份可能过期的数字。
    try:
        _sc_defaults = ACESoftcorePotential().get_parameters_dict()
    except Exception as _sc_exc:  # 绝不让 provenance 记录本身打断运行
        _sc_defaults = {"error": f"unavailable: {_sc_exc}"}
    provenance["softcore_family"] = {
        "class": "ACESoftcorePotential",
        "alpha_convention": getattr(
            ACESoftcorePotential, "ALPHA_CONVENTION", None
        ),
        "constructor_defaults": _sc_defaults,
        "effective_values_are_per_stage": True,
        "effective_values_location": (
            "<stage>_convergence.json / final_results.json -> "
            "stage_protocol_key.payload.aces_softcore_params"
        ),
        "why_it_matters": (
            "λ 端点（λ=1 精确 LJ、λ=0 为 0）与 power_lj/power_coul 无关，"
            "但**中间态**是不同的哈密顿量。拿外部参考做逐窗口对比前必须先对齐指数，"
            "否则只有总量可比。见 docs/reference_data/README.md。"
        ),
    }
    if charge_protocol is not None:
        provenance["charge_protocol"] = charge_protocol
        provenance["charge_treatment"] = charge_protocol.get("charge_treatment")
        provenance["ligand_net_charge"] = charge_protocol.get("ligand_net_charge_e")
        # The old global singular input is not a runtime identity: B5 freezes
        # complex/solvent specs independently after each pipeline is built.
        # Keep it only as an explicitly deprecated audit record so old readers
        # can see what was supplied, but never let resume/provenance consumers
        # treat it as the spec actually used by either leg.
        provenance["deprecated_global_co_alchemical_ion_input"] = config.get(
            "co_alchemical_ion",
            charge_protocol.get("co_alchemical_ion"),
        )
        provenance["coion_identity"] = None
        provenance["coion_identity_deprecated"] = True
        provenance["apbs_applicable"] = charge_protocol.get("apbs_applicable")
        provenance["apbs_applied"] = charge_protocol.get("apbs_applied")
        _qualification = charge_treatment_qualification_payload(
            charge_protocol.get("charge_treatment")
        )
        if _qualification:
            provenance.update(_qualification)
            provenance["production_qualification"] = dict(_qualification)
        if charge_protocol.get("experimental_not_for_production"):
            provenance["experimental_not_for_production"] = True
        # B3/B4 实现状态如实落盘：一条声明 charge-transfer 的运行到底闭不闭合循环，
        # 事后必须能从 provenance 直接读出来，而不是靠回忆当时代码到哪一步了。
        if charge_protocol.get("closes_thermodynamic_cycle") is False:
            provenance["closes_thermodynamic_cycle"] = False
            provenance["must_not_report_delta_g_bind"] = True
            provenance["incomplete_cycle_reason"] = (
                "charge_transfer_tethered_carrier_reservoir_correction_not_validated"
            )
    if co_alchemical_ions is not None:
        provenance["co_alchemical_ions"] = co_alchemical_ions
    # PLAN_openmm_8_6_remd_backend.md §4.3：requested/resolved backend、原始与解析后
    # OpenMM 版本、适配器协议版本、选择原因、交换算法都必须可审计地落盘。
    # 🔑 用 config.get 而不是属性访问：旧的 provenance 消费者和测试里的 config stub
    # 不一定有这个键，缺失时安静跳过，不能让新增字段炸掉既有 resume。
    _remd_backend_provenance = config.get("_remd_backend_resolution")
    if _remd_backend_provenance:
        provenance["remd_backend"] = dict(_remd_backend_provenance)
    provenance["input_files"] = {
        "gro": config.get("gro"),
        "top": config.get("top"),
        "ligand_xml": config.get("ligand_xml"),
        "config": config.get("config"),
        "torsion_params": config.get("torsion_params"),
        "dexp_params": config.get("dexp_params"),
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_provenance.json")
    _atomic_write_json_with_encoder(path, provenance)
    return provenance


def _atomic_write_json_with_encoder(path: str, payload: Dict) -> None:
    """Write a JSON object atomically while retaining the project's encoder.

    Provenance is read by resume and by external audit tooling.  A process being
    interrupted between ``open(..., 'w')`` and ``json.dump`` must not leave a
    truncated document that looks like a valid-but-empty provenance record.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, cls=NumpyEncoder)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        if os.name == "posix":
            parent = os.path.dirname(path) or "."
            dir_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _update_run_provenance_co_alchemical_ions(
    output_dir: str,
    co_alchemical_ions: Dict,
    current: Optional[Dict] = None,
) -> Dict:
    """Merge the two-leg co-ion identities into run provenance atomically."""
    path = os.path.join(output_dir, "run_provenance.json")
    if current is not None:
        provenance = dict(current)
    elif os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            provenance = loaded if isinstance(loaded, dict) else {}
        except (OSError, ValueError):
            provenance = {}
    else:
        provenance = {}
    provenance["co_alchemical_ions"] = co_alchemical_ions
    _atomic_write_json_with_encoder(path, provenance)
    return provenance


def _update_run_provenance_seed_contract(
    output_dir: str,
    contracts: Dict[str, Dict],
    current: Optional[Dict] = None,
) -> Dict:
    """Persist the seed values actually requested by both backend legs."""
    path = os.path.join(output_dir, "run_provenance.json")
    if current is not None:
        provenance = dict(current)
    elif os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        provenance = loaded if isinstance(loaded, dict) else {}
    else:
        provenance = {}
    versions = [
        int(payload["protocol_version"])
        for payload in contracts.values()
        if isinstance(payload, dict) and payload.get("protocol_version") is not None
    ]
    provenance["random_seed_contract"] = {
        "status": "APPLIED_BACKEND_SEED_MAP",
        "protocol_version": max(versions) if versions else None,
        "legs": contracts,
    }
    _atomic_write_json_with_encoder(path, provenance)
    return provenance


def _augment_final_result_with_co_alchemical_ions(
    path: str, co_alchemical_ions: Dict
) -> None:
    """Add the combined two-leg identity block to an already-written leg result."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"应写入 co-ion 双腿身份的 final_results 不存在：{path}"
        )
    try:
        with open(path, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        if not isinstance(result, dict):
            return
        result["co_alchemical_ions"] = co_alchemical_ions
        _atomic_write_json_with_encoder(path, result)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"无法原子更新 co-ion 双腿身份到 final_results：{path}: {exc}"
        ) from exc


def _coion_leg_protocol_signature(spec: Dict) -> Dict:
    """Return the cross-leg co-ion *protocol*, excluding leg-local identity.

    Atom/residue indices, anchors, and frozen displacements are intentionally
    omitted: complex and solvent have different topologies/coordinates and
    therefore are allowed to have different fingerprints.  The route, charge,
    endpoint convention, ion model, and restraint implementation must still be
    identical, otherwise the two legs do not form the same thermodynamic cycle.
    """
    if not isinstance(spec, dict):
        raise ValueError(f"co-ion spec 必须是 dict，收到 {type(spec).__name__}")
    ions = spec.get("ions")
    if not isinstance(ions, list) or not ions:
        raise ValueError("co-ion spec 缺少非空 ions 列表")
    ion_protocol = []
    for ion in ions:
        if not isinstance(ion, dict):
            raise ValueError("co-ion spec 的 ions 项必须是 dict")
        restraint = ion.get("restraint")
        if not isinstance(restraint, dict):
            raise ValueError("co-ion spec 缺少 restraint 协议")
        ion_protocol.append(
            {
                "residue_name": str(ion.get("residue_name")),
                "element": str(ion.get("element")),
                "charge_at_lambda1_e": float(ion.get("charge_at_lambda1_e")),
                "charge_at_lambda0_e": float(ion.get("charge_at_lambda0_e")),
                "sigma_nm": float(ion.get("sigma_nm")),
                "epsilon_kj_mol": float(ion.get("epsilon_kj_mol")),
                "mass_amu": float(ion.get("mass_amu")),
                "restraint": {
                    key: restraint.get(key)
                    for key in (
                        "form",
                        "expression",
                        "reference_frame",
                        "box_model",
                        "k_kj_per_mol_nm2",
                        "flat_bottom_radius_nm",
                        "force_group",
                        "anchor_selection_rule",
                    )
                },
            }
        )
    return {
        "charge_treatment": str(spec.get("charge_treatment")),
        "ligand_net_charge_e": int(spec.get("ligand_net_charge_e")),
        "lambda_direction": str(spec.get("lambda_direction")),
        "ions": ion_protocol,
    }


def _assert_coion_legs_complete_and_compatible(
    coion_identities: Dict[str, Optional[Dict]],
    complex_spec: Optional[Dict],
    solvent_spec: Optional[Dict],
    charge_treatment: Optional[str],
) -> None:
    """Fail closed before a charge-transfer two-leg cycle can be reported."""
    if str(charge_treatment) != CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        return
    if set(coion_identities) != {"complex", "solvent"}:
        raise RuntimeError(
            "charge-transfer 最终汇总要求 complex 与 solvent 两条腿都提供 co-ion 身份；"
            f"当前字段={sorted(coion_identities)}"
        )
    complex_identity = coion_identities.get("complex")
    solvent_identity = coion_identities.get("solvent")
    if not isinstance(complex_identity, dict) or not isinstance(solvent_identity, dict):
        raise RuntimeError(
            "charge-transfer 两腿 co-ion 身份必须都是有效 dict，不能为 None。"
        )
    if not isinstance(complex_spec, dict) or not isinstance(solvent_spec, dict):
        raise RuntimeError(
            "charge-transfer 两腿必须各自有已冻结并验证的 co-ion spec。"
        )
    complex_protocol = _coion_leg_protocol_signature(complex_spec)
    solvent_protocol = _coion_leg_protocol_signature(solvent_spec)
    if complex_protocol != solvent_protocol:
        raise RuntimeError(
            "charge-transfer complex/solvent co-ion 协议不一致："
            "charge_treatment、配体净电荷、lambda 方向、ion 模型或 restraint 协议不同；"
            "两腿 fingerprint 可以不同，但物理协议必须一致。"
        )


def run_self_tests() -> int:
    """Lightweight built-in tests for physics conventions and cache signatures."""
    failures = []
    skipped = []

    def check(name: str, condition: bool, detail: str = ""):
        if condition:
            log.info("self-test PASS | %s", name)
        else:
            failures.append(f"{name}: {detail}")
            log.error("self-test FAIL | %s | %s", name, detail)

    eq = {"r0": 1.0, "thetaA0": 1.5708, "thetaB0": 1.5708}
    fc = {"kr": 418.4, "kthetaA": 41.84, "kthetaB": 41.84, "kphiA": 41.84, "kphiB": 41.84, "kphiC": 41.84}
    dg = calculate_boresch_analytical_correction(eq, fc, 300.0)
    check("Boresch analytical reference", abs(dg - (-22.382123)) < 1e-3, f"got {dg:.6f} kJ/mol")

    # ⚠️ 这只是纯数学工具函数 pme_self_correction_energy_kj 自身的符号单元测试，
    # 不代表生产路径会应用它：apply_pme_self_correction 在生产代码里永远是 False
    # （见 ibs_engine.py 附近注释与 PHYSICS_DEFECTS.md #撤销 +C*lambda^2 修正），
    # OpenMM 的 NonbondedForce ParameterOffset 已经在每个 λ 态里正确算出完整
    # PME 自能，不需要也不应该再手动叠加这个修正。这两个函数只作为历史遗留
    # 工具保留，供未来若要做“已验证的带电荷体系协同炼金中和循环”时复用。
    pref = pme_self_correction_prefactor_kj(alpha_ewald_inv_nm=3.0, ligand_charge_square_sum=2.0)
    corr_lam1 = pme_self_correction_energy_kj(1.0, pref)
    corr_lam0 = pme_self_correction_energy_kj(0.0, pref)
    check(
        "PME self correction helper sign (legacy math util, not applied in production)",
        corr_lam1 > 0.0 and corr_lam0 == 0.0,
        f"pref={pref}, lam1={corr_lam1}",
    )

    endpoints = lambda_endpoint_diagnostics([1.0, 0.5, 0.0], [1.0, 0.5, 0.0])
    bad_endpoints = lambda_endpoint_diagnostics([1.0, 0.5], [1.0, 0.5])
    check("lambda endpoint diagnostics", endpoints["ok"] and not bad_endpoints["ok"], f"{endpoints} / {bad_endpoints}")

    sys_a = openmm.System()
    sys_a.addParticle(12.0)
    sys_b = openmm.System()
    sys_b.addParticle(13.0)
    meta_a = _pme_u_kn_meta_payload(2, [1.0, 0.0], [1.0, 1.0], 300.0, sys_a, None, [0], None)
    meta_b = _pme_u_kn_meta_payload(2, [1.0, 0.0], [1.0, 1.0], 300.0, sys_b, None, [0], None)
    # [2026-09-09] `_system_xml_hash` 已退役、恒返回 None —— 它哈希的是**我们自己
    # 生成的中间产物**而不是用户输入，每次进缓存身份都导致"改一行代码、或做一个按
    # 设计改坐标的修复（PBC-01）⟹ 用户已烧掉的 GPU 时间全作废"（复发两次，理由写在
    # abfe_pipeline._system_xml_hash 的函数体上方）。
    # 所以这里断的是**退役契约**，不是"不同质量给出不同哈希"。
    # ⚠️ 这条如果红了，说明有人把 sha256 加回了缓存身份 —— 先去读上面那段注释，
    # 不要通过"让哈希重新生效"来把它弄绿。换了 HMR/质量这类真实差异，走的是 config
    # 进 stage_protocol_key 那条路，与本字段无关。
    # 这里用两个真 System（质量 12 vs 13，XML 确实不同）比 test_todo_verified_fixes.py
    # 的 `fn(None)` 更强：它证明真实输入下也确实不产出哈希。
    check(
        "resume/cache invalidation system hash retired (always None)",
        meta_a["system_xml_sha256"] is None and meta_b["system_xml_sha256"] is None,
        f"system_xml_sha256 复活了: a={meta_a['system_xml_sha256']}, b={meta_b['system_xml_sha256']}",
    )

    try:
        u_kn, n_k = synthetic_mbar_u_kn(delta_f_kT=1.25, n_per_state=200)
        analyzer = TraditionalMBARAnalyzer(temperature=300.0)
        analyzer._last_n_k = n_k
        res = analyzer.solve(u_kn)
        expected = 1.25 * analyzer.kt
        check("MBAR synthetic data", abs(res["delta_G"] - expected) < 1.0, f"got {res['delta_G']:.3f}, expected {expected:.3f}")
    except ImportError as exc:
        skipped.append(f"MBAR synthetic data: {exc}")
        log.warning("self-test SKIP | MBAR synthetic data | %s", exc)

    check("thermodynamic cycle doc present", "PME/self correction" in THERMODYNAMIC_CYCLE_DOC, "missing PME section")

    if skipped:
        log.warning("self-test skipped: %s", skipped)
    if failures:
        log.error("self-test failures: %s", failures)
        return 1
    log.info("self-test completed successfully")
    return 0


