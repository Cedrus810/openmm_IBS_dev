

# ============================================================================
# 净电荷处理协议：charge_treatment（docs/status/memtodolist.md §1.2 / §2 / §7.1；B2）
#
# 这一节是**纯校验层**：把"配体净电荷 / co-ion / APBS / 环境类型"这四个量的合法
# 组合一次性判死，在创建任何 Context 之前（§6.1）。它不构建任何 Hamiltonian——
# charge-transfer 的 charging System 构建是 B3，溶剂腿 co-ion 身份是 B4。
#
# 关键设计约束逐条对应清单：
#   §1.2 必须是**显式配置**；禁止根据"有没有 APBS 数值"猜协议。
#   §1.2 co-ion 路线与 Rocklin/APBS 是二选一，禁止重复修正（双计数）。
#   §0.5.1 MEM-00a-1 `CHARGE_TRANSFER_PROTOCOL_VERSION` 必须**独立**，
#          不复用 SOLVENT_CACHE_PROTOCOL_VERSION / IBS_BIAS_PROTOCOL_VERSION。
#   §0.5.1 MEM-00a-2 co-annihilation 降级为实验对照：membrane 一律 fail closed，
#          输出必须带 `experimental_not_for_production: true`，
#          其数值不得进入任何 ΔG_bind 汇总。
#   §7.7 中性配体路径行为不变——这是当前生产体系（Atenolol 中性）走的那条。
# ============================================================================

# MEM-00a-1：与其它协议版本并列的独立版本号。charge-transfer 协议本身变化时递增，
# 不要因为溶剂缓存或 IBS bias 协议变了就动它，反之亦然。
CHARGE_TRANSFER_PROTOCOL_VERSION = 1

CHARGE_TREATMENT_NEUTRAL = "neutral"
CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER = "co_alchemical_charge_transfer"
CHARGE_TREATMENT_ROCKLIN_APBS = "rocklin_apbs_neutralizing_plasma"
CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL = "co_annihilation_experimental"

CHARGE_TREATMENTS = (
    CHARGE_TREATMENT_NEUTRAL,
    CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    CHARGE_TREATMENT_ROCKLIN_APBS,
    CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
)

# ✅ B3 已落地（2026-08-04）：charge-transfer 的 charging Hamiltonian
# （ligand q→0 与 co-ion 0→q 由同一个 `lam_coul` 反向驱动、co-ion 电荷走 PME
# particle offset）实现在 `ibs_engine.configure_charge_transfer_decharging`，
# restraint 换成 MEM-00d 的 flat-bottom 锚点相对形式。
# 证据：`tests/test_charge_transfer_hamiltonian.py`（§7.2 逐 λ 电荷守恒 / §7.3 co-ion 物理）。
CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED = True

# ✅ B4 已落地（2026-08-05）：`runabfe.build_and_cache_solvent_leg` 现在会在
# `charge_treatment=co_alchemical_charge_transfer` 且配体带净电荷时，额外插入
# `|q_L|` 个建系时预留的中性 ion-shaped dummy（§4.1），让
# `ibs_engine.select_co_alchemical_ion_once` 能在溶剂腿里认出它——复合物腿和
# 溶剂腿从此走同一条身份识别路径，热力学循环闭得上了。
#
# ⚠️ 这个 True 只代表"builder 会产出满足判据的粒子"，不代表已经在真实带电配体
# 体系上跑通验证过（本仓库当前的生产体系 Atenolol 净电荷为 0，这条路径测不出来）；
# §4.2/§4.4 的盒子尺寸敏感性、平衡稳定性仍待真正带电配体上机验证。
CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED = True

# C4/C5 是带电膜 charge-transfer 路线获得生产资格前的强制验收。底层
# Hamiltonian/双腿 builder 已实现并不等于这两项科学验收已经完成。
CHARGE_TRANSFER_FEATURE_STATUS = "experimental"
CHARGE_TRANSFER_PRODUCTION_QUALIFICATION_PROTOCOL_VERSION = 1
CHARGE_TRANSFER_C4_PASSED = False
CHARGE_TRANSFER_C5_PASSED = False

# A restrained charge carrier is not a free reservoir particle.  The release
# free energy must therefore be measured in each leg and carried explicitly
# into the final cycle.  This schema is intentionally separate from the
# Hamiltonian identity: changing a correction artifact must invalidate the
# result even when the underlying PME system is unchanged.
CHARGE_TRANSFER_RESERVOIR_CORRECTION_PROTOCOL_VERSION = 1
CHARGE_TRANSFER_RESERVOIR_CLOSURE_TOLERANCE_KJ_MOL = 1.0
CHARGE_TRANSFER_RESERVOIR_REQUIRED_LEG_FIELDS = (
    "delta_G_release_kJ_mol",
    "error_kJ_mol",
    "converged",
    "restraint_protocol_version",
    "box_model",
)


def validate_charge_transfer_reservoir_correction(
    correction: Any,
    *,
    temperature_k: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate an externally measured co-ion restraint-release correction.

    Charge transfer with a tethered carrier is only a closed cycle after the
    restrained decoupled endpoint is released in *both* legs.  This helper is
    deliberately evidence-driven: it accepts no bare scalar and does not
    infer cancellation from matching force constants.  ``complex`` and
    ``solvent`` contain the measured ``G_free - G_restrained`` release terms;
    the net contribution to ``ΔG_bind`` is solvent minus complex.

    The top-level ``temperature_K`` binds the correction to the sampled kT;
    the pipeline rejects a mismatch.  The ``validation`` block is the
    machine-readable C4/C5 evidence contract.
    It must include at least one independent residual/box/anchor/force scan,
    explicit sample counts, and a non-empty evidence reference.  Numerical
    residuals are checked against the declared tolerance, so a payload cannot
    claim closure merely by setting a boolean flag.
    """
    if not isinstance(correction, dict):
        raise ValueError(
            "charge-transfer reservoir correction 必须是 JSON object，不能是一个裸数"
        )
    def _strict_int(value: Any, field: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"reservoir correction {field} 必须是整数，不能是 bool")
        try:
            integer = int(value)
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"reservoir correction {field} 必须是整数") from exc
        if not math.isfinite(numeric) or numeric != integer:
            raise ValueError(f"reservoir correction {field} 必须是整数")
        return integer

    protocol_version = correction.get("protocol_version")
    if _strict_int(protocol_version, "protocol_version") != CHARGE_TRANSFER_RESERVOIR_CORRECTION_PROTOCOL_VERSION:
        raise ValueError(
            "charge-transfer reservoir correction protocol_version 不匹配："
            f"期望 {CHARGE_TRANSFER_RESERVOIR_CORRECTION_PROTOCOL_VERSION}，"
            f"收到 {protocol_version!r}"
        )
    if str(correction.get("charge_treatment", "")).strip().lower() != (
        CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
    ):
        raise ValueError(
            "reservoir correction 只能用于 co_alchemical_charge_transfer，"
            f"收到 {correction.get('charge_treatment')!r}"
        )
    if correction.get("box_model") != CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL:
        raise ValueError(
            "reservoir correction 的 box_model 必须与当前 co-ion restraint 一致："
            f"{CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL!r}"
        )
    # Reservoir-release free energies are temperature-dependent.  A correction
    # without a declared temperature cannot be compared with the pipeline's
    # kT and is therefore never admissible.  Keep the JSON spelling explicit
    # (``temperature_K``) so a lower-case typo cannot silently bypass this gate.
    if "temperature_K" not in correction:
        raise ValueError("reservoir correction 缺少必需的 temperature_K")
    try:
        declared_temp = float(correction["temperature_K"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reservoir correction temperature_K 必须是正有限数") from exc
    if not math.isfinite(declared_temp) or declared_temp <= 0.0:
        raise ValueError("reservoir correction temperature_K 必须是正有限数")
    if temperature_k is not None:
        try:
            temp = float(temperature_k)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("temperature_k 必须是正有限数") from exc
        if not math.isfinite(temp) or temp <= 0.0:
            raise ValueError("temperature_k 必须是正有限数")
        if not math.isclose(declared_temp, temp, rel_tol=1.0e-6, abs_tol=1.0e-6):
            raise ValueError(
                "reservoir correction temperature_K 与当前 pipeline 温度不一致："
                f"correction={declared_temp:g} K, pipeline={temp:g} K"
            )

    normalized: Dict[str, Any] = {
        "protocol_version": CHARGE_TRANSFER_RESERVOIR_CORRECTION_PROTOCOL_VERSION,
        "charge_treatment": CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        "box_model": CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL,
        "temperature_K": declared_temp,
    }
    for leg_name in ("complex", "solvent"):
        leg = correction.get(leg_name)
        if not isinstance(leg, dict):
            raise ValueError(
                f"reservoir correction 缺少 {leg_name} leg 的完整测量对象"
            )
        missing = [field for field in CHARGE_TRANSFER_RESERVOIR_REQUIRED_LEG_FIELDS if field not in leg]
        if missing:
            raise ValueError(f"reservoir correction {leg_name} 缺少字段 {missing}")
        try:
            dg = float(leg["delta_G_release_kJ_mol"])
            err = float(leg["error_kJ_mol"])
            restraint_version = _strict_int(
                leg["restraint_protocol_version"],
                f"{leg_name}.restraint_protocol_version",
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"reservoir correction {leg_name} 数值字段非法") from exc
        if not math.isfinite(dg) or not math.isfinite(err) or err < 0.0:
            raise ValueError(
                f"reservoir correction {leg_name} 的 ΔG/error 必须有限且 error≥0"
            )
        if leg.get("converged") is not True:
            raise ValueError(
                f"reservoir correction {leg_name} 必须带 converged=true；"
                "未收敛的释放腿不能闭合循环"
            )
        if restraint_version != CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION:
            raise ValueError(
                f"reservoir correction {leg_name} restraint_protocol_version={restraint_version}"
                " 与当前 co-ion identity protocol 不一致"
            )
        if leg.get("box_model") != CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL:
            raise ValueError(f"reservoir correction {leg_name} box_model 不一致")
        normalized[leg_name] = {
            "delta_G_release_kJ_mol": dg,
            "error_kJ_mol": err,
            "converged": True,
            "restraint_protocol_version": restraint_version,
            "box_model": leg["box_model"],
            "sample_count": _strict_int(
                leg.get("sample_count", 0), f"{leg_name}.sample_count"
            ),
        }
        if normalized[leg_name]["sample_count"] < 2:
            raise ValueError(f"reservoir correction {leg_name} sample_count 必须至少为 2")

    validation = correction.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("reservoir correction 缺少 validation（C4/C5）证据块")
    if validation.get("c4_passed") is not True or validation.get("c5_passed") is not True:
        raise ValueError("reservoir correction 必须同时通过 c4_passed=true 与 c5_passed=true")
    evidence = validation.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(item, str) and item.strip() for item in evidence
    ):
        raise ValueError(
            "reservoir correction validation.evidence 必须是非空字符串列表"
        )
    try:
        tolerance = float(validation.get("tolerance_kJ_mol", CHARGE_TRANSFER_RESERVOIR_CLOSURE_TOLERANCE_KJ_MOL))
        residual = float(validation["max_abs_closure_residual_kJ_mol"])
        box_sensitivity = float(validation["max_abs_box_size_sensitivity_kJ_mol"])
        anchor_sensitivity = float(validation["max_abs_anchor_sensitivity_kJ_mol"])
        restraint_sensitivity = float(validation["max_abs_restraint_sensitivity_kJ_mol"])
        n_box = _strict_int(validation["n_box_sizes"], "validation.n_box_sizes")
        n_anchor = _strict_int(
            validation["n_anchor_choices"], "validation.n_anchor_choices"
        )
        n_restraint = _strict_int(
            validation["n_restraint_settings"],
            "validation.n_restraint_settings",
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("reservoir correction validation 缺少 C4/C5 数值扫描字段") from exc
    scan_values = (tolerance, residual, box_sensitivity, anchor_sensitivity, restraint_sensitivity)
    if not all(math.isfinite(value) for value in scan_values) or tolerance <= 0.0:
        raise ValueError("reservoir correction validation 数值必须有限且 tolerance>0")
    if tolerance > CHARGE_TRANSFER_RESERVOIR_CLOSURE_TOLERANCE_KJ_MOL:
        raise ValueError(
            "reservoir correction validation tolerance 不能放宽项目闭合门："
            f"最大 {CHARGE_TRANSFER_RESERVOIR_CLOSURE_TOLERANCE_KJ_MOL:g} kJ/mol"
        )
    if any(value < 0.0 or value > tolerance for value in scan_values[1:]):
        raise ValueError(
            "reservoir correction C4/C5 扫描残差超过声明 tolerance，拒绝闭合循环"
        )
    if min(n_box, n_anchor, n_restraint) < 3:
        raise ValueError("C4/C5 至少需要 3 个盒尺寸、锚点选择和 restraint 设置")
    normalized["validation"] = {
        "c4_passed": True,
        "c5_passed": True,
        "evidence": [item.strip() for item in evidence],
        "tolerance_kJ_mol": tolerance,
        "max_abs_closure_residual_kJ_mol": residual,
        "max_abs_box_size_sensitivity_kJ_mol": box_sensitivity,
        "max_abs_anchor_sensitivity_kJ_mol": anchor_sensitivity,
        "max_abs_restraint_sensitivity_kJ_mol": restraint_sensitivity,
        "n_box_sizes": n_box,
        "n_anchor_choices": n_anchor,
        "n_restraint_settings": n_restraint,
    }
    normalized["delta_G_bind_correction_kJ_mol"] = (
        normalized["solvent"]["delta_G_release_kJ_mol"]
        - normalized["complex"]["delta_G_release_kJ_mol"]
    )
    normalized["error_kJ_mol"] = float(
        math.sqrt(
            normalized["complex"]["error_kJ_mol"] ** 2
            + normalized["solvent"]["error_kJ_mol"] ** 2
        )
    )
    return normalized

# [Stage2 handoff，2026-08-11] charge-transfer 配体的 charging→vanishing
# 交接：vanishing 阶段的输入 System 现在会先用
# `bake_global_parameter_into_fixed_nonbonded_force` 把一份独立的 charging
# 配置固化到 λ_coul=0 端点（配体 0 电荷、co-ion 满电、配体内部 exception
# 保留物理 chargeProd），再喂给 `build_ibs_dual_system`——不是直接把原始
# `self.system` 传过去（那样带净电配体会被 `build_ibs_dual_system` 自己的
# 电中性防御拒绝）。见 `docs/experiments/STAGE2_CHARGE_TRANSFER_HANDOFF_PROPOSAL.md`。
#
# 这个协议版本号只在 `ABFEPipeline._charge_transfer_vanishing_handoff_active()`
# 返回 True（即 charge_treatment 是 charge-transfer 且配体净电荷非零）时才
# 写入 Stage 2 的缓存指纹——中性配体（当前唯一生产路径，Atenolol 净电荷为 0）
# 完全不受影响，指纹不变，旧缓存不会被误判失效。
CHARGE_TRANSFER_VANISHING_HANDOFF_PROTOCOL_VERSION = 1

# §13.2 数值自洽的两个容差。放在这里是因为本层就要用；§13 的完整阈值表另立。
LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E = 1.0e-3
TOTAL_CHARGE_CONSERVATION_TOLERANCE_E = 1.0e-6

# §3.4 要求写入 manifest 的 co-ion 身份字段。B2 只校验"齐不齐、算不算得平"，
# 真正的选择与落盘是 B4。
CO_ALCHEMICAL_ION_REQUIRED_FIELDS = (
    "atom_index",
    "residue_index",
    "residue_name",
    "element",
    "charge_at_lambda1_e",
    "charge_at_lambda0_e",
    "sigma_nm",
    "epsilon_kj_mol",
    "mass_amu",
    "restraint",
)

# Candidate residue names used by both the solvent builder and the frozen
# identity layer.  A residue name alone never proves that a particle is a
# reserved dummy; callers must also verify its zero lambda=1 charge.
CO_ALCHEMICAL_ION_RESIDUE_NAMES = frozenset(
    {"CL", "CLA", "CL-", "NA", "NA+", "SOD", "K", "K+", "POT", "MG", "CA"}
)

CO_ALCHEMICAL_ION_BUILDER_IDENTITY_SCHEMA_VERSION = 1

# §5 / §1.2：选 Rocklin 路线时必须真的有 APBS 证据，不能只填一个数。
APBS_REQUIRED_EVIDENCE_FIELDS = (
    "manifest_path",
    "result_path",
    "dielectric_map_paths",
    "lipid_charge_map_path",
    "net_charge_e",
)


def charge_treatment_qualification_payload(charge_treatment: Optional[str]) -> Dict[str, Any]:
    """Return the single-source production-qualification payload.

    Neutral and Rocklin paths keep their legacy result shape. Charged
    charge-transfer is merged as an experimental capability: C4/C5 are not
    complete, so even an externally supplied reservoir correction cannot
    silently promote the numerical result to production-qualified.
    """
    resolved = str(charge_treatment or "").strip().lower()
    if resolved == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        c4_passed = bool(CHARGE_TRANSFER_C4_PASSED)
        c5_passed = bool(CHARGE_TRANSFER_C5_PASSED)
        return {
            "production_qualification_protocol_version": (
                CHARGE_TRANSFER_PRODUCTION_QUALIFICATION_PROTOCOL_VERSION
            ),
            "feature_status": CHARGE_TRANSFER_FEATURE_STATUS,
            "production_qualified": bool(c4_passed and c5_passed),
            "c4_passed": c4_passed,
            "c5_passed": c5_passed,
            "production_qualification_reason": (
                "charged_membrane_charge_transfer_requires_c4_and_c5"
            ),
        }
    if resolved == CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL:
        return {
            "production_qualification_protocol_version": (
                CHARGE_TRANSFER_PRODUCTION_QUALIFICATION_PROTOCOL_VERSION
            ),
            "feature_status": "experimental_method_comparison_only",
            "production_qualified": False,
            "c4_passed": False,
            "c5_passed": False,
            "production_qualification_reason": "co_annihilation_not_for_production",
        }
    return {}


def charge_transfer_result_markers(
    charge_protocol: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """未闭合/未验收的 charge-transfer ⟹ 跟着**每份产物**走的禁报标记。

    🔑 [2026-09-18] `--run-unclosed-charge-transfer-diagnostics` 让这类运行能起，
    代价是它的数**一步都不能进 ΔG_bind 汇总**。先前这条信息只保证写进
    `run_provenance.json`，逐腿 `final_results.json` 与 `final_binding_results.json`
    只有 `production_qualified=false` —— 只读结果文件的人看不出"禁报"。
    这里把三个键做成一份、由**已解析的协议 payload** 派生（不另判一次循环闭合，
    那是 `resolve_charge_treatment` 的事），两边产物用同一个来源。

    中性 / Rocklin ⟹ 返回 `{}`（`charge_treatment_qualification_payload` 对这两条
    路线本来就返回空）⟹ 既有产物逐字节不变。
    """
    if not isinstance(charge_protocol, dict):
        return {}
    qualification = charge_treatment_qualification_payload(
        charge_protocol.get("charge_treatment")
    )
    if not qualification:
        return {}
    closes = charge_protocol.get("closes_thermodynamic_cycle") is True
    qualified = bool(qualification.get("production_qualified"))
    if qualified and closes:
        return {}
    return {
        "production_qualified": qualified,
        "closes_thermodynamic_cycle": closes,
        "must_not_report_delta_g_bind": True,
        "must_not_report_delta_g_bind_reason": (
            charge_protocol.get("incomplete_cycle_reason")
            or qualification.get("production_qualification_reason")
        ),
    }


def resolve_charge_treatment(
    charge_treatment: Optional[str],
    ligand_net_charge_e: float,
    apbs_correction_kJ_mol: float = 0.0,
    co_alchemical_ion: Optional[Any] = None,
    require_co_alchemical_ion: bool = False,
    environment_type: Optional[str] = None,
    apbs_evidence: Optional[Dict[str, Any]] = None,
    charge_transfer_reservoir_correction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """校验净电荷处理协议，返回可直接进 fingerprint/provenance 的解析结果。

    `charge_treatment=None` 时按 §1.2 的生产默认推导：中性配体 → `neutral`，
    带电配体 → `co_alchemical_charge_transfer`。**这不是"猜协议"**——它只看配体
    净电荷这一个客观量，与清单禁止的"根据有没有 APBS 数值猜"是两回事。

    Charge-transfer additionally requires a validated two-leg reservoir-release
    correction.  Without ``charge_transfer_reservoir_correction`` the protocol
    is intentionally marked as an open cycle; callers must fail closed before
    constructing an OpenMM context.  Supplying the evidence-backed object
    does not by itself promote the route to production qualification (the
    project-wide C4/C5 flags remain an explicit release decision).

    `co_alchemical_ion` 现在只接受一个**可选的兼容性 override**：真正的 B3–B5
    运行身份由 complex/solvent 两条 pipeline 各自选择并冻结，不能靠一个全局
    atom-index spec 作为前置放行条件。若调用方确实要在这个纯校验层验证一份
    外部 spec，可显式传入它；旧的“必须在前置阶段提供全局 spec”行为只有在
    `require_co_alchemical_ion=True` 时才启用。

    `ligand_net_charge_e` 由调用方用现有实现算出并传入（复合物腿走
    `ibs_engine._compute_ligand_net_charge`），本函数不自己再数一遍电荷，
    避免出现第二套净电荷判据。
    """
    raw_q = float(ligand_net_charge_e)
    if not math.isfinite(raw_q):
        raise ValueError(f"配体净电荷不是有限数：{ligand_net_charge_e!r}")
    q_int = int(round(raw_q))
    if abs(raw_q - q_int) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
        raise ValueError(
            f"配体净电荷 {raw_q:+.6f} e 不接近整数"
            f"（容差 {LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:g} e）。"
            "§2.2：非整数净电荷必须先作为输入错误调查，"
            "不要静默塞给一个分数价 co-ion。"
        )

    try:
        apbs_value = float(apbs_correction_kJ_mol or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"apbs_correction_kJ_mol={apbs_correction_kJ_mol!r} 不是数值"
        ) from exc
    if not math.isfinite(apbs_value):
        raise ValueError("apbs_correction_kJ_mol 必须有限")

    # ---- 解析协议名 ----
    if charge_treatment is None or str(charge_treatment).strip() == "":
        resolved = (
            CHARGE_TREATMENT_NEUTRAL
            if q_int == 0
            else CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
        )
        was_defaulted = True
    else:
        resolved = str(charge_treatment).strip().lower()
        was_defaulted = False
    if resolved not in CHARGE_TREATMENTS:
        raise ValueError(
            f"charge_treatment={charge_treatment!r} 不是合法值；"
            f"允许 {list(CHARGE_TREATMENTS)}。"
        )

    if (
        charge_transfer_reservoir_correction is not None
        and resolved != CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
    ):
        raise ValueError(
            "charge_transfer_reservoir_correction 只能与"
            " charge_treatment=co_alchemical_charge_transfer 一起使用"
        )

    resolved_environment = resolve_environment_type(environment_type)
    is_experimental = resolved == CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL

    # ---- 逐协议规则（§1.2 的 fail-closed 清单）----
    if resolved == CHARGE_TREATMENT_NEUTRAL:
        # fail-closed #2：neutral 但检测到配体净电荷不为 0。
        if q_int != 0:
            raise ValueError(
                f"charge_treatment=neutral 但配体净电荷为 {q_int:+d} e。"
                "带电配体必须显式选择净电荷处理路线："
                f"生产用 {CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER}，"
                f"方法对照用 {CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL}，"
                f"neutralizing plasma 用 {CHARGE_TREATMENT_ROCKLIN_APBS}。"
            )
        if co_alchemical_ion:
            raise ValueError("charge_treatment=neutral 不得创建 co-alchemical ion。")
        if apbs_value != 0.0:
            raise ValueError(
                f"charge_treatment=neutral 但 apbs_correction_kJ_mol={apbs_value:+.6f}。"
                "中性配体既不需要 co-ion，也不需要 Rocklin 净电荷修正（§0 第 3 条）。"
            )

    elif resolved in (
        CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
    ):
        if q_int == 0:
            raise ValueError(
                f"charge_treatment={resolved} 但配体净电荷为 0。"
                "中性配体不需要共炼金离子，请用 neutral。"
            )
        # fail-closed #1：co-ion 路线与 APBS 是二选一，禁止双计数。
        if apbs_value != 0.0:
            raise ValueError(
                f"charge_treatment={resolved} 且 "
                f"apbs_correction_kJ_mol={apbs_value:+.6f} —— 这是**重复修正**。"
                "共炼金路线全程保持体系总电荷不变，Rocklin/APBS 净电荷修正必须为 0，"
                "并记录 not_applicable_co_alchemical_charge_transfer（§0 第 3 条）。"
            )

        if is_experimental:
            # MEM-00a-2：膜生产一律 fail closed。
            if resolved_environment == ENVIRONMENT_TYPE_MEMBRANE:
                raise ValueError(
                    "charge_treatment=co_annihilation_experimental 不允许出现在膜体系中。"
                    "co-annihilation 同时执行 ligand: q→0 与 counterion: −q→0，"
                    "两个异号离子在膜/水非均匀环境中的消失自由能不能可靠抵消"
                    "（一个在结合位点、一个在体相水，介电环境完全不同）。"
                    "膜生产请用 co_alchemical_charge_transfer；"
                    "本路线只允许用于水盒 / lipid slab 的方法对照（§8.1/§8.2 末条）。"
                )
        else:
            # B5：前置解析不再依赖一个全局 atom-index spec；真正的
            # co-ion 身份/参数/restraint 会在每条 pipeline 冻结后由 runtime
            # verifier fail closed。显式 override 仍可在这里做兼容性校验。
            if require_co_alchemical_ion and co_alchemical_ion is None:
                raise ValueError(
                    "charge_treatment=co_alchemical_charge_transfer 但没有提供 "
                    "co_alchemical_ion 身份、参数与 restraint（§1.2 fail-closed 第 3 条、§3.4）。"
                )
            if co_alchemical_ion is not None:
                _validate_co_alchemical_ion_spec(co_alchemical_ion, q_int)
            if not CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED:
                raise NotImplementedError(
                    "charge_treatment=co_alchemical_charge_transfer 的 charging "
                    "Hamiltonian 尚未实现（memtodolist Phase B3）。"
                    "`ibs_engine.py` 里现存的共炼金实现是 co-annihilation，"
                    "不是 ligand q→0 / co-ion 0→q 的 charge-transfer（MEM-00a）。"
                    "在 B3 落地前拒绝放行：否则结果会声明 charge-transfer 而实际跑的是别的哈密顿量。"
                    "方法对照请显式选 co_annihilation_experimental。"
                )

    elif resolved == CHARGE_TREATMENT_ROCKLIN_APBS:
        if co_alchemical_ion:
            raise ValueError(
                "charge_treatment=rocklin_apbs_neutralizing_plasma 禁止创建 "
                "co-alchemical ion —— 两条路线是二选一，同时用就是重复修正。"
            )
        # fail-closed #4：缺 APBS 来源说明/结果文件。
        missing = [
            field
            for field in APBS_REQUIRED_EVIDENCE_FIELDS
            if not (apbs_evidence or {}).get(field)
        ]
        if missing:
            raise ValueError(
                f"charge_treatment=rocklin_apbs_neutralizing_plasma 缺少 APBS 证据字段 "
                f"{missing}（§1.2 fail-closed 第 4 条、§5）。"
                "必须提供真实 APBS manifest/result、介电图与脂质电荷图；"
                "只填一个 apbs_correction_kJ_mol 数值不算。"
            )

    payload: Dict[str, Any] = {
        "protocol_version": CHARGE_TRANSFER_PROTOCOL_VERSION,
        "charge_treatment": resolved,
        "was_defaulted_from_net_charge": bool(was_defaulted),
        "ligand_net_charge_e": q_int,
        "ligand_net_charge_raw_e": raw_q,
        "environment_type": resolved_environment,
        "apbs_correction_kJ_mol": apbs_value,
        "apbs_applicable": resolved == CHARGE_TREATMENT_ROCKLIN_APBS,
        "apbs_applied": resolved == CHARGE_TREATMENT_ROCKLIN_APBS and apbs_value != 0.0,
        "co_alchemical_ion": co_alchemical_ion or None,
        "experimental_not_for_production": bool(is_experimental),
    }

    if resolved in (
        CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
    ):
        payload["apbs_not_applicable_reason"] = (
            "not_applicable_co_alchemical_charge_transfer"
            if not is_experimental
            else "not_applicable_co_annihilation_conserves_total_charge"
        )
        payload["total_charge_conserved_at_every_lambda"] = True
        payload["charging_hamiltonian_implemented"] = bool(
            CHARGE_TRANSFER_HAMILTONIAN_IMPLEMENTED
        )
        # B4 的 builder 身份已经落地，但这不等于 tethered carrier 的
        # reservoir/配分函数项在 complex 与 solvent 两腿严格相消。该项还
        # 需要完整的双腿 closure 与盒/锚点/力常数扫描（C4/C5）；在此之前
        # 必须明确标成未闭合，禁止下游把数值写成生产 ΔG_bind。
        payload["solvent_leg_builder_implemented"] = bool(
            CHARGE_TRANSFER_SOLVENT_LEG_IMPLEMENTED
        )
        payload["closes_thermodynamic_cycle"] = False
        payload["thermodynamic_cycle_closure_validated"] = False
        payload["incomplete_cycle_reason"] = (
            "tethered_charge_carrier_reservoir_correction_not_validated"
        )
        if charge_transfer_reservoir_correction is not None:
            validated_correction = validate_charge_transfer_reservoir_correction(
                charge_transfer_reservoir_correction,
                temperature_k=None,
            )
            payload["charge_transfer_reservoir_correction"] = validated_correction
            payload["closes_thermodynamic_cycle"] = True
            payload["thermodynamic_cycle_closure_validated"] = True
            payload.pop("incomplete_cycle_reason", None)
        payload.update(charge_treatment_qualification_payload(resolved))
    elif resolved == CHARGE_TREATMENT_ROCKLIN_APBS:
        payload["total_charge_conserved_at_every_lambda"] = False
        payload["apbs_evidence"] = dict(apbs_evidence or {})
    else:
        payload["total_charge_conserved_at_every_lambda"] = True
        payload["apbs_not_applicable_reason"] = "not_applicable_neutral_ligand"

    return payload


def _validate_co_alchemical_ion_spec(spec: Any, ligand_net_charge: int) -> None:
    """校验 co-ion 规格：字段齐全、每粒子不超过一个单位电荷、总变化抵消（§2.2/§3.4）。

    接受单个 dict 或 dict 列表——`|q_L| > 1` 时按 §2.2 必须用多个单价 co-ion 分担，
    不允许把多个单位电荷集中到一个非物理多价粒子上。
    """
    ions = spec if isinstance(spec, (list, tuple)) else [spec]
    if not ions:
        raise ValueError("co_alchemical_ion 为空")

    seen_indices = set()
    total_transferred = 0.0
    for position, ion in enumerate(ions):
        if not isinstance(ion, dict):
            raise ValueError(
                f"co_alchemical_ion[{position}] 必须是 dict，收到 {type(ion).__name__}"
            )
        missing = [f for f in CO_ALCHEMICAL_ION_REQUIRED_FIELDS if f not in ion]
        if missing:
            raise ValueError(
                f"co_alchemical_ion[{position}] 缺少 §3.4 要求的字段 {missing}"
            )
        index = int(ion["atom_index"])
        if index in seen_indices:
            raise ValueError(
                f"co_alchemical_ion 出现重复 atom_index={index}；"
                "同一个粒子不能被当成两个 co-ion。"
            )
        seen_indices.add(index)

        q1 = float(ion["charge_at_lambda1_e"])
        q0 = float(ion["charge_at_lambda0_e"])
        # §2.2：λ=1 时它是"中性但保留 LJ 的 ion-shaped dummy"。
        if abs(q1) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            raise ValueError(
                f"co_alchemical_ion[{position}] 在 λ=1 的电荷应为 0（中性 dummy），"
                f"实际 {q1:+.6f} e。"
            )
        transferred = q0 - q1
        if abs(transferred) - 1.0 > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            raise ValueError(
                f"co_alchemical_ion[{position}] 转移了 {transferred:+.6f} e。"
                "§2.2：每个 co-ion 最多转移一个单位电荷；"
                f"|q_L| > 1 请用 {abs(ligand_net_charge)} 个单价 co-ion 分担，"
                "不要构造非物理多价粒子。"
            )
        if ligand_net_charge > 0 and transferred <= 0:
            raise ValueError(
                f"配体净电荷 {ligand_net_charge:+d} e 为正，co-ion 必须是同号（阳离子型，"
                f"0 → +1），实际转移 {transferred:+.6f} e。"
                "注意 charge-transfer 不是 co-annihilation：不是去掉一个异号反离子。"
            )
        if ligand_net_charge < 0 and transferred >= 0:
            raise ValueError(
                f"配体净电荷 {ligand_net_charge:+d} e 为负，co-ion 必须是同号（阴离子型，"
                f"0 → −1），实际转移 {transferred:+.6f} e。"
            )
        if not ion.get("restraint"):
            raise ValueError(
                f"co_alchemical_ion[{position}] 缺少 restraint（§2.3 要求可审计的 "
                "flat-bottom restraint，参考位置随盒缩放）。"
            )
        total_transferred += transferred

    # fail-closed #5：配体电荷变化与 co-ion 电荷变化之和不为 0。
    ligand_change = 0.0 - float(ligand_net_charge)
    residual = ligand_change + total_transferred
    if abs(residual) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
        raise ValueError(
            f"配体电荷变化 ({ligand_change:+.6f} e) 与 co-ion 电荷变化之和 "
            f"({total_transferred:+.6f} e) 不为 0，残差 {residual:+.6e} e "
            f"> {TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:g} e"
            "（§1.2 fail-closed 第 5 条）。总电荷必须在每个 λ 严格守恒。"
        )


# ============================================================================
# MEM-00c：co-ion 身份冻结（B3 的前置条件，不是 B5 的缓存细节）
#
# 为什么必须冻结，而不是"每个入口自己选一次就行"：
# `ibs_engine._select_bulk_water_counterion` 按**传入坐标**当场排序挑离子
# （主键 = 到最近溶质的 minimum-image 距离，一个连续量），没有持久化身份、
# 也没有只读模式。而喂给它的坐标在**跨进程 resume** 时会变：
#   · 首跑：`pre_equilibrate()` 的输出**再叠 2000 步快速最小化**；
#   · resume（`skip_equil`）：直接读 `pre_equilibration.dcd` **末帧**，不做最小化。
# `tests/test_coalchemical_ion_identity.py` 实测 **0.05 nm 位移就足以翻转选择结果**，
# 而 2000 步最小化的原子位移正是 0.01–0.1 nm 量级。于是首跑用粒子 A 跑动力学、
# resume 进程用粒子 B 重算 u_kn —— u_kn 与动力学 Hamiltonian 静默不一致，
# ΔG 会错而没有任何异常现象。
#
# 落地形态（顺序不可颠倒）：
#     选一次 → 落成 spec（含指纹） → dynamics / replicas / u_kn / resume 全部只读消费
#     → 任何下游重选路径都不可达
#
# 字段直接复用 §3.4 的 `CO_ALCHEMICAL_ION_REQUIRED_FIELDS`，不另造 schema。
# ============================================================================

# v1 → v2（2026-08-04，MEM-00d + B3）：restraint 形式从"绝对笛卡尔参考点的纯谐振子"
# 换成"锚点相对的 flat-bottom"（见下方 §2.3/MEM-00d 一节），restraint 字典的键因此
# 整体改变。v1 的 spec 一律拒绝复用——它记的 `reference_frame` 是
# `absolute_cartesian_at_selection_time`，那个参考点在膜半各向异性 NPT 下会把离子拖向膜。
CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION = 2

# 指纹覆盖的字段 = "任一项变化都必须拒绝旧缓存"的那张表。少一项就等于允许
# 一个静默不一致的 resume：例如只钉 atom_index 而不钉 sigma/epsilon，
# 换了离子类型（Cl⁻→Br⁻）却仍复用旧 u_kn。
CO_ALCHEMICAL_ION_IDENTITY_FINGERPRINT_FIELDS = (
    "atom_index",       # 粒子 index
    "residue_index",    # 残基 index（index 相同但残基变了 = 拓扑换了）
    "residue_name",     # 离子类型
    "element",          # 离子类型
    "mass_amu",
    "sigma_nm",
    "epsilon_kj_mol",
    "charge_at_lambda1_e",  # 端点电荷
    "charge_at_lambda0_e",
    "restraint",        # restraint 形式 + 参考位置
)


# ============================================================================
# MEM-00d：co-ion restraint 的形式（§2.3 / §13.1）
#
# 旧形式（v1，已退役）：`0.5*k*periodicdistance(x,y,z,x0,y0,z0)^2`，k = 25，
# 参考点是**选中那一刻的绝对笛卡尔坐标**。它有两个毛病：
#   1. 没有平坦区 —— §2.3 要求"平坦区足够大的 flat-bottom，避免把 co-ion 锁死在
#      一个异常水构象"；
#   2. 参考点在膜半各向异性 NPT 下不随盒缩放 —— Z 方向盒长变而参考点不动，
#      离子会被系统性地拖向膜。
#
# 新形式：**锚点相对**的 flat-bottom。参考点 = 锚点原子当前位置 + 冻结的位移向量 d0。
# 这只保证平移/刚体运动不改变井心；d0 本身仍是笛卡尔 per-bond 参数，不能随
# barostat 的盒矢量缩放。charge-transfer 因而明确限定为固定盒 NVT，NPT/半各向异性
# 体系在注入 restraint 前 fail closed（见 `_inject_co_alchemical_ion_restraints`）。
#
# 为什么用 `CustomCompoundBondForce` + `pointdistance` 而不是别的：
#   * `periodicdistance` **只存在于 CustomExternalForce**——实测
#     `CustomCentroidBondForce` 与 `CustomCompoundBondForce` 都报
#     `unknown function: periodicdistance`（2026-08-04 在 OpenMM 8.5.1 上逐个试过）。
#     而 CustomExternalForce 只能吃**绝对**参考点，正是要退役的那个形式。
#   * `CustomCompoundBondForce` 打开 PBC 后会把同一个 bond 里的粒子平移到与第一个
#     粒子相同的周期镜像再求值，所以 `pointdistance` 在其中**就是** minimum-image
#     距离。实测：离子 z=0.2、锚点 z=9.4、盒 z=12 → 得 0.2 nm（不是 9.2 nm）。
#   * 三斜/各向异性盒都走同一条 minimum-image 逻辑，无需自己写盒矩阵运算。
#
# 锚点取**配体重原子中离配体质心最近的那一个**，两条腿用同一条规则：
#   * 它随锚点平移，消掉绝对坐标井心的系统性拖拽；
#   * 它让"co-ion ↔ 配体 minimum-image 距离"在结构上被 restraint 本身钉住
#     （§13.1 的全程下限从"事后诊断"变成"构造时可证"）；
#   * 两条腿同一个锚点规则、同一个 k/r₀ 仍不足以证明 carrier reservoir
#     配分函数相消；该项在 C4/C5 完成前不计作闭合循环；
#   * 取"离质心最近"而不是随便一个重原子：配体转动时该原子位移最小，井心抖动最小。
# ============================================================================

CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM = "flat_bottom_anchor_relative"
# ⚠️ 这个表达式与 `ibs_engine._create_co_alchemical_ion_restraint` 里实际注入的
# 必须逐字符相同 —— 它进身份指纹，所以"记录的形式"与"跑的形式"分叉就等于
# 让一份 spec 描述了另一个哈密顿量。有契约测试钉住两者相等。
CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION = (
    "0.5*k_ion*max(0, pointdistance(x1,y1,z1, x2+dx0, y2+dy0, z2+dz0) - r0_ion)^2"
)
CO_ALCHEMICAL_ION_RESTRAINT_REFERENCE_FRAME = "anchor_atom_relative_displacement"
# The frozen Cartesian displacement is valid only in a fixed-box (NVT) leg.
# Keep this marker in every identity spec so a future dynamic fractional-box
# implementation cannot silently reuse the old Hamiltonian/cache.
CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL = "fixed_cartesian_displacement_nvt_only"
# restraint 与 Boresch（force group 3）分开，且**逐 λ 完全相同**（§2.3、§6.4）。
CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP = 6
# 判"是重原子"的质量下限：H 是 1.008，D 是 2.014，最轻的重原子 C 是 12。
CO_ALCHEMICAL_ION_ANCHOR_MIN_MASS_AMU = 2.5

# 走出平坦区之后墙很软（k=100 时 0.316 nm 才 2 kT），所以"全程 ≥ 1.2 nm"这条
# 不能只按平坦区半径算，必须再留一段热涨落余量。取 2 kT 对应的位移：
#     0.5 * k * d² = 2 kT(300 K) = 4.988 kJ/mol  ⟹  d = 0.316 nm
CO_ALCHEMICAL_ION_RESTRAINT_THERMAL_MARGIN_KT = 2.0
CO_ALCHEMICAL_ION_RESTRAINT_REFERENCE_TEMPERATURE_K = 300.0


def co_alchemical_ion_restraint_wall_margin_nm(
    k_kj_per_mol_nm2: Optional[float] = None,
    margin_kt: float = CO_ALCHEMICAL_ION_RESTRAINT_THERMAL_MARGIN_KT,
    temperature_k: float = CO_ALCHEMICAL_ION_RESTRAINT_REFERENCE_TEMPERATURE_K,
) -> float:
    """平坦区之外还能走多远才付得起 `margin_kt` 个 kT（nm）。

    用途是把 §13.1 的"co-ion ↔ 配体全程 ≥ 1.2 nm"变成**构造时可证的几何条件**，
    而不是只靠事后逐帧诊断去发现它已经被违反了。

    `k_kj_per_mol_nm2=None` 时取 §13.1 的 `COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2`
    ——不写成默认参数值是因为那个常量在本文件里定义得更晚（§13 一节），
    默认参数在 def 执行时求值会直接 NameError。
    """
    k = (
        COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2
        if k_kj_per_mol_nm2 is None
        else float(k_kj_per_mol_nm2)
    )
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"restraint 力常数必须为正有限数，收到 {k_kj_per_mol_nm2!r}")
    temp = float(temperature_k)
    if not math.isfinite(temp) or temp <= 0.0:
        raise ValueError(f"温度必须为正有限数，收到 {temperature_k!r}")
    margin = float(margin_kt)
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError(f"margin_kt 必须为正有限数，收到 {margin_kt!r}")
    kt = 0.008314462618 * temp
    return math.sqrt(2.0 * margin * kt / k)


def minimum_image_displacement_nm(displacement, box_vectors) -> np.ndarray:
    """三斜盒的 minimum-image 位移（行向量盒矩阵，nm）。

    这是本仓库 minimum-image 数学的**唯一**实现；`ibs_engine._minimum_image_displacement_nm`
    是它的薄包装（那一层负责把 OpenMM Quantity / list-of-Vec3 归一成 (3,3) 数组）。
    放在 abfe_core 是因为 co-ion 的几何判据（§13.1）在这一层就要用，而 abfe_core
    在 ibs_engine 下层、不能反向 import。
    """
    delta = np.asarray(displacement, dtype=np.float64)
    box = np.asarray(box_vectors, dtype=np.float64)
    if box.shape != (3, 3) or not np.all(np.isfinite(box)):
        raise ValueError("minimum-image 计算需要有限的 (3,3) 周期盒向量")
    det = float(np.linalg.det(box))
    if not np.isfinite(det) or abs(det) <= 1.0e-12:
        raise ValueError("周期盒向量奇异，无法计算 minimum-image 位移")
    if delta.shape[-1:] != (3,):
        raise ValueError("minimum-image 位移的最后一维必须是 3")
    if not np.all(np.isfinite(delta)):
        raise ValueError("minimum-image 位移必须全部为有限数")
    # Empty selections are valid for some diagnostic paths.  Avoid taking a
    # reduction over an empty array below, while preserving the requested
    # shape and dtype.
    if delta.size == 0:
        return np.array(delta, dtype=np.float64, copy=True)
    # Component-wise rounding is not exact for a skew triclinic cell. Search
    # integer lattice points around the Babai/rounded point and expand the
    # cube until a singular-value lower bound proves that every point outside
    # it is farther away.  Unlike a fixed 27-neighbour stencil this remains
    # correct for strongly sheared (but nonsingular) cells.
    fractional = delta @ np.linalg.inv(box)
    base = np.floor(fractional + 0.5).astype(np.int64)
    sigma_min = float(np.linalg.svd(box, compute_uv=False)[-1])
    if not np.isfinite(sigma_min) or sigma_min <= 0.0:
        raise ValueError("周期盒向量最小奇异值非法，无法计算 minimum-image 位移")
    best_wrapped = None
    best_norm2 = np.full(delta.shape[:-1], np.inf, dtype=np.float64)
    radius = 0
    while True:
        offsets = np.asarray(
            [
                (i, j, k)
                for i in range(-radius, radius + 1)
                for j in range(-radius, radius + 1)
                for k in range(-radius, radius + 1)
            ],
            dtype=np.int64,
        )
        candidates = (base[..., None, :] + offsets).astype(np.float64)
        wrapped = delta[..., None, :] - np.einsum("...ni,ij->...nj", candidates, box)
        norms2 = np.einsum("...ni,...ni->...n", wrapped, wrapped)
        local_best = np.argmin(norms2, axis=-1)
        local_norm2 = np.take_along_axis(norms2, local_best[..., None], axis=-1)[..., 0]
        replace = local_norm2 < best_norm2
        if best_wrapped is None:
            best_wrapped = np.take_along_axis(
                wrapped, local_best[..., None, None], axis=-2
            )[..., 0, :]
        else:
            local_wrapped = np.take_along_axis(
                wrapped, local_best[..., None, None], axis=-2
            )[..., 0, :]
            best_wrapped = np.where(replace[..., None], local_wrapped, best_wrapped)
        best_norm2 = np.minimum(best_norm2, local_norm2)

        # For an integer point outside [-radius, radius]^3, ||n-f||₂ is at
        # least radius+1-sqrt(3)/2 because each component of f-base is ≤1/2.
        outside_lower = sigma_min * max(0.0, radius + 1.0 - math.sqrt(3.0) / 2.0)
        if outside_lower * outside_lower >= float(np.max(best_norm2)):
            return best_wrapped
        radius += 1
        if radius > 64:
            raise RuntimeError(
                "minimum-image closest-lattice 搜索未在 64 层内收敛；"
                "周期盒过度病态，拒绝猜测周期像。"
            )


def box_vectors_to_nm_array(box_vectors) -> Optional[np.ndarray]:
    """把 OpenMM Quantity / list-of-Vec3 / 裸数组的周期盒归一成 (3,3) nm 数组。

    唯一实现放在这里（`ibs_engine._box_vectors_to_nm_array` 是它的别名），因为
    `unwrap_boresch_anchors_nm` 与 co-ion 的几何判据在本层就要用。
    盒信息缺失或形状不对返回 `None`，由调用方决定是 fail-closed 还是跳过。
    """
    if box_vectors is None:
        return None
    if hasattr(box_vectors, "value_in_unit"):
        return np.asarray(box_vectors.value_in_unit(unit.nanometer), dtype=np.float64)
    if isinstance(box_vectors, np.ndarray):
        return box_vectors.astype(np.float64, copy=False)
    if isinstance(box_vectors, (list, tuple)):
        if len(box_vectors) != 3:
            return None
        first = box_vectors[0]
        if hasattr(first, "x"):
            return np.asarray([[v.x, v.y, v.z] for v in box_vectors], dtype=np.float64)
        return np.asarray(box_vectors, dtype=np.float64)
    return None


def unwrap_boresch_anchors_nm(rec_coords, lig_coords, box_vectors) -> np.ndarray:
    """把 6 个 Boresch 锚点解缠到同一个周期像，返回 (6,3) 的 [H0,H1,H2,L0,L1,L2]。

    **BOR-01**：这是本仓库 Boresch 六锚点解缠的唯一实现。原先只有校验那份
    （`ibs_engine._check_boresch_geometry_safe`）解缠，而真正把 r0/θ/φ 写进限制势的
    `calc_boresch_from_last_frame` 用的是裸 `norm(a-b)`。锚点对一旦跨周期边界，
    提交的 r0 就静默差一个盒矢量（~4–5 nm），校验函数因为自己解了缠**反而看不出分歧**。
    跟"同一个不变量的 N 份实现"是同一个形状（λ 身份那次是四份）。

    解缠链与校验那份一致：H0→H1→H2 与 H0→L0→L1→L2（逐跳，不是全部对 H0）。

    平移量取**整数格矢**（`raw - ((raw-ref) - minimum_image(raw-ref))`）而不是
    `ref + minimum_image(raw-ref)`：没跨边界时格矢恰为 0.0，返回坐标与输入**逐位相同**，
    既有 Boresch 平衡值不会被一次浮点重排整体作废。

    `box_vectors` 为 `None`（或无法解析成 (3,3)）时原样返回 —— 等于调用方声明
    这组坐标已经连续（例如 `runabfe` 里刚跑过 `image_molecules_by_system`）。
    """
    anchors = np.asarray(
        [np.asarray(p, dtype=np.float64) for p in (*rec_coords, *lig_coords)],
        dtype=np.float64,
    )
    if anchors.shape != (6, 3):
        raise ValueError(f"Boresch 锚点解缠需要 3+3 个点，实得 {anchors.shape}")
    box = box_vectors_to_nm_array(box_vectors)
    if box is None:
        return anchors

    def _to(ref, raw):
        delta = raw - ref
        return raw - (delta - minimum_image_displacement_nm(delta, box))

    H0 = anchors[0]
    H1 = _to(H0, anchors[1])
    H2 = _to(H1, anchors[2])
    L0 = _to(H0, anchors[3])
    L1 = _to(L0, anchors[4])
    L2 = _to(L1, anchors[5])
    return np.asarray([H0, H1, H2, L0, L1, L2], dtype=np.float64)

def co_alchemical_ion_anchor_atom_index(
    *,
    system: Any,
    ligand_indices: Sequence[int],
    positions_nm: Sequence[Sequence[float]],
    box_vectors,
) -> Dict[str, Any]:
    """选 restraint 锚点原子：**离配体质心最近的配体重原子**（MEM-00d）。

    两条腿用同一条规则，规则本身与坐标无关（只有"哪一个最近"这一步看坐标），
    结果连同 minimum-image 位移一起冻结进 spec，此后只读核对。

    返回 dict：`anchor_atom_index` / `anchor_position_nm` / `ligand_extent_nm`
    （配体任一原子到锚点的最大 minimum-image 距离，供 §13.1 的几何余量判据用）。
    """
    indices = [int(i) for i in ligand_indices]
    if not indices:
        raise ValueError("配体原子列表为空，无法确定 co-ion restraint 锚点。")
    pos = np.asarray(positions_nm, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions_nm 形状非法：{pos.shape}")

    heavy = [
        i
        for i in indices
        if float(system.getParticleMass(i).value_in_unit(unit.dalton))
        >= CO_ALCHEMICAL_ION_ANCHOR_MIN_MASS_AMU
    ]
    if not heavy:
        raise ValueError(
            "配体没有质量 ≥ "
            f"{CO_ALCHEMICAL_ION_ANCHOR_MIN_MASS_AMU} amu 的重原子，无法选 restraint 锚点。"
        )

    # 质心用 minimum-image 相对第一个重原子累加，避免配体跨周期边界时质心跑到盒中间。
    origin = pos[heavy[0]]
    offsets = minimum_image_displacement_nm(pos[heavy] - origin, box_vectors)
    masses = np.asarray(
        [float(system.getParticleMass(i).value_in_unit(unit.dalton)) for i in heavy],
        dtype=np.float64,
    )
    centroid = origin + (offsets * masses[:, None]).sum(axis=0) / masses.sum()
    distances = np.linalg.norm(
        minimum_image_displacement_nm(pos[heavy] - centroid, box_vectors), axis=1
    )
    anchor_index = int(heavy[int(np.argmin(distances))])
    extent = float(
        np.max(
            np.linalg.norm(
                minimum_image_displacement_nm(
                    pos[indices] - pos[anchor_index], box_vectors
                ),
                axis=1,
            )
        )
    )
    return {
        "anchor_atom_index": anchor_index,
        "anchor_position_nm": [round(float(v), 6) for v in pos[anchor_index]],
        "ligand_centroid_nm": [round(float(v), 6) for v in centroid],
        "ligand_extent_from_anchor_nm": extent,
        "anchor_selection_rule": "ligand_heavy_atom_nearest_ligand_center_of_mass",
        "ligand_heavy_atom_count": int(len(heavy)),
    }


def co_alchemical_ion_restraint_spec(
    *,
    anchor: Dict[str, Any],
    reference_displacement_nm: Sequence[float],
    flat_bottom_radius_nm: Optional[float] = None,
    k_kj_per_mol_nm2: Optional[float] = None,
) -> Dict[str, Any]:
    """MEM-00d 的 restraint 描述（进身份指纹 ⟹ 改形式即作废旧 spec/缓存）。

    `reference_displacement_nm` 是"锚点 → co-ion"的 minimum-image 位移，在选择那一刻
    冻结。它是笛卡尔 per-bond 参数，仅适用于固定盒 NVT；NPT/半各向异性 barostat
    会改变盒矢量而不改变该参数，注入层会明确拒绝。
    """
    radius = (
        COION_FLAT_BOTTOM_RADIUS_NM
        if flat_bottom_radius_nm is None
        else float(flat_bottom_radius_nm)
    )
    k = (
        COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2
        if k_kj_per_mol_nm2 is None
        else float(k_kj_per_mol_nm2)
    )
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError(f"flat-bottom 半径必须为正有限数，收到 {radius!r}")
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"restraint 力常数必须为正有限数，收到 {k!r}")
    raw_displacement = list(reference_displacement_nm)
    if len(raw_displacement) != 3:
        raise ValueError(
            f"reference_displacement_nm 需要 3 个分量，收到 {reference_displacement_nm!r}"
        )
    displacement = [round(float(v), 6) for v in raw_displacement]
    if not all(math.isfinite(value) for value in displacement):
        raise ValueError("reference_displacement_nm 必须全部为有限数")
    return {
        "form": CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM,
        "expression": CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION,
        "reference_frame": CO_ALCHEMICAL_ION_RESTRAINT_REFERENCE_FRAME,
        "box_model": CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL,
        "k_kj_per_mol_nm2": k,
        "flat_bottom_radius_nm": radius,
        "force_group": CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP,
        "anchor_atom_index": int(anchor["anchor_atom_index"]),
        "anchor_selection_rule": str(anchor["anchor_selection_rule"]),
        "reference_displacement_nm": displacement,
    }


def validate_co_alchemical_ion_placement(
    *,
    restraint: Dict[str, Any],
    ligand_extent_from_anchor_nm: float,
    strict: bool,
) -> Dict[str, Any]:
    """§13.1：flat-bottom 井是否**构造性地**保证"co-ion ↔ 配体全程 ≥ 1.2 nm"。

    判据（全部量都是 minimum-image）：

        |d0| − r₀ − wall_margin − ligand_extent  ≥  COION_LIGAND_MIN_IMAGE_RUNTIME_NM

    `|d0|` 是锚点到 co-ion 的冻结位移长度，`ligand_extent` 是配体任一原子到锚点的
    最大距离（锚点不是配体最外缘，所以必须减掉它），`wall_margin` 是走出平坦区
    2 kT 对应的位移（墙很软，只算平坦区会高估约束力）。

    `strict=False` 时只返回报告不 raise —— 那是 co-annihilation 实验对照专用路径
    （MEM-00a-2：只能用于水盒/lipid slab 方法对照，其数值不得进入任何 ΔG_bind 汇总），
    它的反离子是从既有盐里挑的物理离子、不是我们摆的，几何余量由诊断如实记录。
    ⚠️ charge-transfer 生产路线一律 `strict=True`。
    """
    d0 = np.asarray(restraint["reference_displacement_nm"], dtype=np.float64)
    d0_norm = float(np.linalg.norm(d0))
    radius = float(restraint["flat_bottom_radius_nm"])
    margin = co_alchemical_ion_restraint_wall_margin_nm(
        restraint.get("k_kj_per_mol_nm2")
    )
    extent = float(ligand_extent_from_anchor_nm)
    guaranteed = d0_norm - radius - margin - extent
    report = {
        "anchor_to_coion_distance_nm": d0_norm,
        "flat_bottom_radius_nm": radius,
        "wall_margin_nm": margin,
        "wall_margin_kt": CO_ALCHEMICAL_ION_RESTRAINT_THERMAL_MARGIN_KT,
        "ligand_extent_from_anchor_nm": extent,
        "guaranteed_min_ligand_distance_nm": guaranteed,
        "required_min_ligand_distance_nm": COION_LIGAND_MIN_IMAGE_RUNTIME_NM,
        "initial_threshold_nm": COION_LIGAND_MIN_IMAGE_INITIAL_NM,
        "satisfies_runtime_threshold": bool(
            guaranteed >= COION_LIGAND_MIN_IMAGE_RUNTIME_NM
        ),
        "satisfies_initial_threshold": bool(
            d0_norm - extent >= COION_LIGAND_MIN_IMAGE_INITIAL_NM
        ),
        "enforced": bool(strict),
    }
    if strict and not (
        report["satisfies_runtime_threshold"] and report["satisfies_initial_threshold"]
    ):
        raise ValueError(
            "co-ion 摆放不满足 §13.1 的几何余量：\n"
            f"    锚点↔co-ion 冻结距离 |d0| = {d0_norm:.3f} nm\n"
            f"    配体最外缘到锚点     = {extent:.3f} nm\n"
            f"    平坦区半径 r₀        = {radius:.3f} nm\n"
            f"    2 kT 软墙余量        = {margin:.3f} nm\n"
            f"    ⟹ 可保证的最小配体距离 = {guaranteed:.3f} nm"
            f"（要求 ≥ {COION_LIGAND_MIN_IMAGE_RUNTIME_NM} nm 全程、"
            f"初始 ≥ {COION_LIGAND_MIN_IMAGE_INITIAL_NM} nm）\n"
            "    正解是把 reserved co-ion 摆得更远（并保证 minimum-image 安全，§4.2），"
            "不是放宽本判据或缩小 flat-bottom 半径 —— 后者只是把违反藏进事后诊断。"
        )
    return report


# ============================================================================
# B3：charging λ 的逐粒子电荷映射（§2.1 / §2.4 / §7.2）
#
# OpenMM 的 `NonbondedForce.addParticleParameterOffset` 给出的是
#     q(λ) = q_base + λ · q_scale
# 而 `ibs_engine` 的 λ 约定是 `lam_coul` 从 1 → 0。于是两条共炼金路线只差
# "base/scale 怎么填"：
#
#   co-annihilation（实验对照，MEM-00a-2）：配体与**异号**反离子同步消电
#       ligand i : base 0,        scale q_i          ⟹ q(λ) = λ q_i
#       counterion: base 0,       scale q_phys       ⟹ q(λ) = λ q_phys
#
#   charge-transfer（生产，§2.1）：电荷从配体**搬到**体相水里的同号 co-ion
#       ligand i : base 0,        scale q_i          ⟹ q(λ) = λ q_i
#       co-ion j : base share_j,  scale −share_j     ⟹ q(λ) = (1−λ) share_j
#
# 两者的总电荷守恒条件是同一句话：**Σ scale = 0**。因为
#     Σq(λ) = Σq_base + λ · Σq_scale
# 所以 Σscale = 0 ⟺ 总电荷与 λ 无关 —— 这不是在几个 λ 点上抽查，而是对**所有** λ
# （含中间态，§7.2 要求）的一次代数证明。co-annihilation 满足它的方式是
# Σscale = q_L + (−q_L) = 0（异号反离子）；charge-transfer 是 q_L + (−q_L) = 0
# （配体升、co-ion 降）。
# ============================================================================


def co_alchemical_charge_offset_plan(
    *,
    charge_treatment: str,
    ligand_net_charge_e: int,
    ligand_charges_e: Dict[int, float],
    co_ion_physical_charges_e: Dict[int, float],
) -> Dict[str, Any]:
    """把"哪个粒子在 λ 上怎么变"算成 `{index: (base, scale)}`，并证明总电荷守恒。

    纯数学层：不碰 OpenMM 对象，所以可以在没有 GPU、没有真实体系的情况下逐 λ 验。
    `ibs_engine` 负责把结果写进 `NonbondedForce`，并在写完之后用同一份 plan 核对
    ——生产者与校验者共用同一份真相，不各写一套。
    """
    q_l = int(ligand_net_charge_e)
    if charge_treatment not in (
        CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
    ):
        raise ValueError(
            f"charge_treatment={charge_treatment!r} 不涉及共炼金离子，"
            "不该为它构造 charging 电荷计划。"
        )
    if q_l == 0:
        raise ValueError("配体净电荷为 0：中性路径不需要共炼金电荷计划。")

    plan: Dict[int, Tuple[float, float]] = {}
    for idx, q in ligand_charges_e.items():
        plan[int(idx)] = (0.0, float(q))

    ion_indices = sorted(int(i) for i in co_ion_physical_charges_e)
    if charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        if len(ion_indices) != abs(q_l):
            raise ValueError(
                f"charge-transfer 需要 {abs(q_l)} 个单价 co-ion（配体净电荷 {q_l:+d} e，"
                f"§2.2 禁止把多个单位电荷集中到一个非物理多价粒子上），"
                f"实际拿到 {len(ion_indices)} 个：{ion_indices}"
            )
        share = 1.0 if q_l > 0 else -1.0
        for idx in ion_indices:
            physical = float(co_ion_physical_charges_e[idx])
            if abs(physical) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
                raise ValueError(
                    f"reserved co-ion（粒子 {idx}）在输入体系里的电荷是 "
                    f"{physical:+.6f} e，不是 0。\n"
                    "    charge-transfer 要求 λ=1 端它是**中性但保留 LJ 的 "
                    "ion-shaped dummy**（§2.2），配体的净电荷则由**普通离子**在建系时"
                    "配平（§4.3）。\n"
                    "    换句话说：不能把一个已经带电的物理离子拿来当 co-ion —— 那样"
                    "λ=1 端的总电荷就不再是物理体系的总电荷了。\n"
                    "    正解是在建系时额外加入 "
                    f"{abs(q_l)} 个电荷为 0 的 ion-shaped 粒子（B4 的溶剂腿 builder "
                    "会这么做；复合物腿的输入拓扑必须自带）。"
                )
            plan[idx] = (share, -share)
    else:
        for idx in ion_indices:
            plan[idx] = (0.0, float(co_ion_physical_charges_e[idx]))

    base_sum = float(sum(base for base, _ in plan.values()))
    scale_sum = float(sum(scale for _, scale in plan.values()))
    if abs(scale_sum) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
        raise ValueError(
            f"charging 电荷计划的 Σscale = {scale_sum:+.6e} e ≠ 0，"
            f"容差 {TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:g} e。\n"
            "    Σq(λ) = Σq_base + λ·Σq_scale，所以 Σscale ≠ 0 就意味着总电荷随 λ 变 —— "
            "PME 会用一个逐 λ 变化的中和背景电荷把它掩盖掉，ΔG 静默出错（§7.2）。"
        )
    return {
        "charge_treatment": str(charge_treatment),
        "ligand_net_charge_e": q_l,
        "offsets": plan,
        "base_sum_e": base_sum,
        "scale_sum_e": scale_sum,
        "co_ion_indices": ion_indices,
        "total_charge_is_lambda_independent": True,
    }


def charge_at_lambda(base_e: float, scale_e: float, lam: float) -> float:
    """OpenMM particle-parameter-offset 的电荷模型：`q(λ) = base + λ·scale`。

    单独抽出来是为了让"期望值"只有一处定义 —— 测试、诊断与断言都调它，
    不各自手写一遍 `base + lam*scale`（写歪了就会自己对上自己）。
    """
    return float(base_e) + float(lam) * float(scale_e)


def _canonical_fingerprint(payload: Any) -> str:
    """稳定的 canonical-JSON sha256。dict 顺序、浮点格式都不能影响结果。"""
    import hashlib

    blob = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _co_alchemical_ion_endpoint_charges(
    charge_treatment: str,
    physical_charge_e: float,
    ligand_net_charge_e: int,
    share_e: Optional[float] = None,
) -> Tuple[float, float]:
    """按所选电荷路线给出 co-ion 在 λ=1 / λ=0 的电荷。

    λ 约定与 `ibs_engine` 一致：`lam_coul` 从 **1 → 0**，粒子电荷走
    `addParticleParameterOffset`，即 q(λ) = λ · q_offset。

    * `co_annihilation_experimental`（`ibs_engine` 现存实现）：反离子是**异号**的，
      与配体**同步**消电 ⟹ λ=1 时是它的物理电荷、λ=0 时为 0。
    * `co_alchemical_charge_transfer`（B3 目标）：co-ion 是**同号**的中性 dummy，
      电荷由配体**转移**过来 ⟹ λ=1 时为 0、λ=0 时为 +q_L。

    两条路线的端点电荷不同，所以它进指纹：一份 co-annihilation 的 spec
    绝不能被一次声明 charge-transfer 的运行拿去复用。
    """
    if charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        # 中性 dummy → 接过配体电荷中属于**这一个粒子**的那一份。
        # `share_e` 由调用方按 §2.2 拆成 |q_L| 个单价份额；不传时退回单 co-ion 情形
        # （此时份额就是整份净电荷）。⚠️ 不要在这里默默把 |q_L|>1 塞给一个粒子——
        # 那正是 §2.2 禁止的非物理多价 co-ion。
        if share_e is None:
            return 0.0, float(ligand_net_charge_e)
        return 0.0, float(share_e)
    if charge_treatment == CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL:
        return float(physical_charge_e), 0.0
    raise ValueError(
        f"charge_treatment={charge_treatment!r} 不涉及 co-alchemical ion，"
        "不应该为它构造 co-ion 身份 spec。"
    )


def build_co_alchemical_ion_identity(
    *,
    system: Any,
    topology: Any,
    ion_atom_indices: Sequence[int],
    ligand_indices: Sequence[int],
    positions_nm: Sequence[Sequence[float]],
    box_vectors: Any,
    ligand_net_charge_e: int,
    charge_treatment: str,
    lambda_direction: str = "lam_coul_1_to_0",
    selection_provenance: Optional[Dict[str, Any]] = None,
    flat_bottom_radius_nm: Optional[float] = None,
    k_kj_per_mol_nm2: Optional[float] = None,
    enforce_placement_thresholds: Optional[bool] = None,
) -> Dict[str, Any]:
    """把"这一次选出来的 co-ion"固化成可落盘、可跨进程核对的身份 spec。

    只在**选择发生的那一次**调用。此后所有消费者（动力学 System 构建、REMD 副本、
    `compute_u_kn`、resume）都只读这份 spec，不再调用选择器。

    restraint 的**形式不接受调用方指定**：它由 `co_alchemical_ion_restraint_spec()`
    唯一生成（MEM-00d 的 flat-bottom + 锚点相对），调用方只能覆盖 k 与平坦区半径。
    这样就不可能出现"spec 里记着一种形式、实际注入另一种"的分叉。参考量（锚点
    index + 冻结位移）属于身份的一部分，所以整份 restraint 进指纹。

    `enforce_placement_thresholds` 默认按路线取：charge-transfer（生产）强制 §13.1
    几何余量，co-annihilation（实验对照）只记录诊断。见
    `validate_co_alchemical_ion_placement` 的 docstring。
    """
    import openmm

    nb_force = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None
    )
    if nb_force is None:
        raise RuntimeError("构造 co-ion 身份需要 NonbondedForce，但 System 里没有。")

    indices = [int(i) for i in ion_atom_indices]
    if not indices:
        raise ValueError(
            "ion_atom_indices 为空。中性配体不需要 co-ion —— 那种情况不该走到这里"
            "（调用方应当在 ligand_net_charge_e == 0 时完全跳过身份构造）。"
        )
    if len(indices) != len(set(indices)):
        raise ValueError(f"ion_atom_indices 有重复：{indices}")

    q_l = int(ligand_net_charge_e)
    if charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER and len(
        indices
    ) != abs(q_l):
        raise ValueError(
            f"charge-transfer 需要 {abs(q_l)} 个单价 co-ion（配体净电荷 {q_l:+d} e，"
            f"§2.2），实际给了 {len(indices)} 个：{indices}"
        )

    pos_nm = np.asarray(positions_nm, dtype=np.float64)
    anchor = co_alchemical_ion_anchor_atom_index(
        system=system,
        ligand_indices=ligand_indices,
        positions_nm=pos_nm,
        box_vectors=box_vectors,
    )
    strict = (
        charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
        if enforce_placement_thresholds is None
        else bool(enforce_placement_thresholds)
    )
    share = (1.0 if q_l > 0 else -1.0) if q_l else 0.0

    atoms = list(topology.atoms())
    ions: List[Dict[str, Any]] = []
    for position, index in enumerate(indices):
        if not 0 <= index < len(atoms):
            raise ValueError(
                f"co-ion atom_index={index} 越界（拓扑只有 {len(atoms)} 个原子）。"
            )
        atom = atoms[index]
        q, sigma, epsilon = nb_force.getParticleParameters(index)
        physical_charge = q.value_in_unit(unit.elementary_charge)
        q1, q0 = _co_alchemical_ion_endpoint_charges(
            charge_treatment, physical_charge, q_l, share_e=share
        )
        displacement = minimum_image_displacement_nm(
            pos_nm[index] - np.asarray(anchor["anchor_position_nm"], dtype=np.float64),
            box_vectors,
        )
        ion_restraint = co_alchemical_ion_restraint_spec(
            anchor=anchor,
            reference_displacement_nm=displacement,
            flat_bottom_radius_nm=flat_bottom_radius_nm,
            k_kj_per_mol_nm2=k_kj_per_mol_nm2,
        )
        # 选择那一刻的绝对坐标：**只作审计记录**，不再被任何 restraint 消费。
        # 刻意不叫 `reference_position_nm`（v1 的键名）—— 那样一个还在读旧键的消费者
        # 会静默拿到已退役的绝对参考点；改了名字它会 KeyError，这是有意的。
        ion_restraint["selection_time_absolute_position_nm"] = [
            round(float(v), 6) for v in pos_nm[index]
        ]
        placement = validate_co_alchemical_ion_placement(
            restraint=ion_restraint,
            ligand_extent_from_anchor_nm=anchor["ligand_extent_from_anchor_nm"],
            strict=strict,
        )
        ions.append(
            {
                "atom_index": int(index),
                "placement_diagnostics": placement,
                "residue_index": int(atom.residue.index),
                "residue_name": str(atom.residue.name),
                "element": str(getattr(atom.element, "symbol", "") or ""),
                "charge_at_lambda1_e": float(q1),
                "charge_at_lambda0_e": float(q0),
                "sigma_nm": float(sigma.value_in_unit(unit.nanometer)),
                "epsilon_kj_mol": float(
                    epsilon.value_in_unit(unit.kilojoule_per_mole)
                ),
                "mass_amu": float(
                    system.getParticleMass(index).value_in_unit(unit.dalton)
                ),
                "restraint": ion_restraint,
                # 诊断字段（不进指纹）：物理电荷用于事后核对端点推导是否合理。
                "physical_charge_e": float(physical_charge),
                "spec_position": int(position),
            }
        )

    spec: Dict[str, Any] = {
        "protocol_version": CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION,
        "charge_treatment": str(charge_treatment),
        "lambda_direction": str(lambda_direction),
        "ligand_net_charge_e": int(ligand_net_charge_e),
        "ions": ions,
        "selection_provenance": dict(selection_provenance or {}),
    }
    spec["fingerprint"] = co_alchemical_ion_identity_fingerprint(spec)
    return spec


def co_alchemical_ion_identity_fingerprint(spec: Dict[str, Any]) -> str:
    """身份指纹：只吃"变了就必须作废旧缓存"的那些字段。

    刻意**不吃** `selection_provenance`（当时的排序距离、水配位数等诊断量）——
    那些数每次读坐标都会变一点，进指纹会让每次 resume 都误判成身份漂移。
    """
    payload = {
        "protocol_version": int(spec["protocol_version"]),
        "charge_treatment": str(spec["charge_treatment"]),
        "lambda_direction": str(spec["lambda_direction"]),
        "ligand_net_charge_e": int(spec["ligand_net_charge_e"]),
        "ions": [
            {field: ion[field] for field in CO_ALCHEMICAL_ION_IDENTITY_FINGERPRINT_FIELDS}
            for ion in spec["ions"]
        ],
    }
    return _canonical_fingerprint(payload)


def verify_co_alchemical_ion_identity(
    spec: Dict[str, Any],
    *,
    system: Any,
    topology: Any,
    charge_treatment: Optional[str] = None,
    ligand_net_charge_e: Optional[int] = None,
    context: str = "",
) -> List[int]:
    """按 spec 核对当前 System/拓扑，通过则返回被钉住的 atom_index 列表。

    **只读**：任何不符都 raise，绝不"重新选一个能对上的"。这就是 MEM-00c 的修法——
    把"每个入口自己选"换成"选一次 + 处处核对"。

    返回的 index 列表可直接交给 `ibs_engine` 的 offset 注入路径使用。
    """
    import openmm

    where = f"（{context}）" if context else ""
    if not isinstance(spec, dict) or "ions" not in spec:
        raise ValueError(f"co-ion 身份 spec 结构非法{where}：{type(spec).__name__}")

    version = int(spec.get("protocol_version", -1))
    if version != CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION:
        raise ValueError(
            f"co-ion 身份 spec 的 protocol_version={version}{where}，"
            f"当前实现是 {CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION}。"
            "协议版本变了就意味着字段含义可能变了，拒绝复用旧 spec；请重新选择并落盘。"
        )

    recomputed = co_alchemical_ion_identity_fingerprint(spec)
    if recomputed != spec.get("fingerprint"):
        raise ValueError(
            f"co-ion 身份 spec 自身指纹不符{where}："
            f"记录 {spec.get('fingerprint')}，重算 {recomputed}。"
            "spec 被手工改过或写坏了 —— 不接受，请重新选择并落盘。"
            "（不要手改 spec 让它对上；那等于把不一致藏起来。）"
        )

    if charge_treatment is not None and str(charge_treatment) != spec["charge_treatment"]:
        raise ValueError(
            f"co-ion 身份 spec 记录的电荷路线是 {spec['charge_treatment']!r}，"
            f"本次运行声明的是 {charge_treatment!r}{where}。"
            "两条路线的端点电荷不同（co-annihilation 是 q_phys→0，"
            "charge-transfer 是 0→q_L），spec 不可跨路线复用。"
        )
    if (
        ligand_net_charge_e is not None
        and int(ligand_net_charge_e) != int(spec["ligand_net_charge_e"])
    ):
        raise ValueError(
            f"co-ion 身份 spec 是为配体净电荷 {spec['ligand_net_charge_e']:+d} e 选的，"
            f"当前体系是 {int(ligand_net_charge_e):+d} e{where}。"
        )

    nb_force = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None
    )
    if nb_force is None:
        raise RuntimeError(f"核对 co-ion 身份需要 NonbondedForce，但 System 里没有{where}。")

    atoms = list(topology.atoms())
    pinned: List[int] = []
    for ion in spec["ions"]:
        index = int(ion["atom_index"])
        if not 0 <= index < len(atoms):
            raise ValueError(
                f"co-ion atom_index={index} 越界{where}（拓扑只有 {len(atoms)} 个原子）——"
                "拓扑变了，旧身份不可用。"
            )
        atom = atoms[index]
        q, sigma, epsilon = nb_force.getParticleParameters(index)

        def _check(field: str, actual: Any, tolerance: Optional[float] = None) -> None:
            expected = ion[field]
            if tolerance is None:
                same = str(expected) == str(actual)
            else:
                same = abs(float(expected) - float(actual)) <= tolerance
            if not same:
                raise ValueError(
                    f"co-ion 身份漂移{where}：atom_index={index} 的 {field} "
                    f"记录为 {expected!r}，当前 System/拓扑给的是 {actual!r}。\n"
                    "    这正是 MEM-00c 要拦的那类静默不一致（u_kn 与动力学"
                    "Hamiltonian 用了不同粒子/参数）。\n"
                    "    正解是**重跑**这条腿并重新选择 co-ion；"
                    "**不要**放宽本核对或改 spec 让它对上。"
                )

        _check("residue_index", int(atom.residue.index))
        _check("residue_name", str(atom.residue.name))
        _check("element", str(getattr(atom.element, "symbol", "") or ""))
        _check("sigma_nm", sigma.value_in_unit(unit.nanometer), 1.0e-9)
        _check(
            "epsilon_kj_mol", epsilon.value_in_unit(unit.kilojoule_per_mole), 1.0e-9
        )
        _check(
            "mass_amu",
            system.getParticleMass(index).value_in_unit(unit.dalton),
            1.0e-6,
        )
        # 端点电荷：核对的是"当前 System 里这个粒子的电荷仍然支持记录的端点"。
        # 注意此处的 System 可能已经被 offset 改写过——那时粒子的**基**电荷是：
        #   * co-annihilation：0（真实电荷整份挪进 ParameterOffset）；
        #   * charge-transfer：share（= λ=0 端电荷，offset 里放 −share）。
        # 所以合法读数是一个小集合，逐项列出来而不是"够小就放过"。
        current_q = q.value_in_unit(unit.elementary_charge)
        expected_physical = float(ion.get("physical_charge_e", current_q))
        acceptable = [expected_physical, 0.0]
        if spec["charge_treatment"] == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
            acceptable.append(float(ion["charge_at_lambda0_e"]))
            # charge-transfer 的 co-ion 必须是**中性 dummy**（§2.2）。这一项不在指纹里，
            # 所以只有在这儿核对才能拦住手改过的 spec。
            if abs(expected_physical) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
                raise ValueError(
                    f"co-ion 身份非法{where}：atom_index={index} 记录的物理电荷是 "
                    f"{expected_physical:+.6f} e，而 charge-transfer 要求 λ=1 端它是"
                    "中性但保留 LJ 的 ion-shaped dummy（§2.2）。"
                    "带电的物理离子不能当 co-ion —— 那样 λ=1 端总电荷就不是物理体系的了。"
                )
        if all(abs(current_q - value) > 1.0e-6 for value in acceptable):
            raise ValueError(
                f"co-ion 身份漂移{where}：atom_index={index} 的物理电荷记录为 "
                f"{expected_physical:+.6f} e，当前 System 给的是 {current_q:+.6f} e"
                f"（本路线可接受的基电荷读数：{acceptable}）。"
                "离子类型变了 —— 旧 u_kn 与旧动力学都不可复用。"
            )

        # restraint：spec 记录的形式必须就是当前实现会注入的那一个（MEM-00d）。
        # 形式对不上意味着这份 spec 描述的是另一个哈密顿量。
        restraint = ion.get("restraint") or {}
        if restraint.get("form") != CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM:
            raise ValueError(
                f"co-ion restraint 形式不符{where}：spec 记的是 {restraint.get('form')!r}，"
                f"当前实现是 {CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM!r}（MEM-00d："
                "旧的『绝对笛卡尔参考点纯谐振子』已退役，它在膜半各向异性 NPT 下会把"
                "离子系统性拖向膜）。请重新选择 co-ion 并落盘新 spec。"
            )
        if restraint.get("box_model") != CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL:
            raise ValueError(
                f"co-ion restraint box_model 不符{where}：spec 记的是 "
                f"{restraint.get('box_model')!r}，当前实现仅支持 "
                f"{CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL!r}。"
            )
        if restraint.get("expression") != CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION:
            raise ValueError(
                f"co-ion restraint 表达式不符{where}：spec 记的是 "
                f"{restraint.get('expression')!r}。它必须与实际注入的逐字符相同。"
            )
        anchor_index = int(restraint.get("anchor_atom_index", -1))
        if not 0 <= anchor_index < len(atoms):
            raise ValueError(
                f"co-ion restraint 锚点 atom_index={anchor_index} 越界{where}"
                f"（拓扑只有 {len(atoms)} 个原子）——拓扑变了，旧身份不可用。"
            )
        pinned.append(index)

    if len(pinned) != len(set(pinned)):
        raise ValueError(f"co-ion 身份 spec 出现重复 atom_index{where}：{pinned}")
    return pinned


def co_alchemical_ion_cache_identity_payload(
    spec: Optional[Dict[str, Any]],
    *,
    system: Any,
    topology: Any,
    leg: str,
    spec_relative_path: str,
    charge_treatment: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return the minimal frozen co-ion identity consumed by cache protocols.

    This is deliberately a thin read-only adapter around the one canonical
    spec verifier/fingerprint.  It never selects an ion, never reads
    coordinates, and never includes ``selection_provenance`` or absolute
    positions.  Neutral legs return ``None`` so their legacy cache semantics
    remain unchanged.
    """
    requested_treatment = (
        None if charge_treatment is None else str(charge_treatment)
    )
    if spec is None:
        if requested_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
            raise ValueError(
                "co-ion runtime identity 缺少 charge-transfer spec："
                f"leg={leg!r}, path={spec_relative_path!r}"
            )
        return None
    if not isinstance(spec, dict):
        raise ValueError(
            f"co-ion runtime identity for leg={leg!r} has invalid spec type "
            f"{type(spec).__name__}; spec={spec_relative_path}"
        )
    resolved_leg = str(leg)
    if resolved_leg not in {"complex", "solvent"}:
        raise ValueError(f"co-ion runtime identity has unknown leg={resolved_leg!r}")
    try:
        q_l = int(spec["ligand_net_charge_e"])
        treatment = str(spec["charge_treatment"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"co-ion runtime identity spec is incomplete for leg={resolved_leg!r}: "
            f"{spec_relative_path}"
        ) from exc
    if requested_treatment is not None and requested_treatment != treatment:
        raise ValueError(
            "co-ion runtime identity charge_treatment 不匹配："
            f"当前运行={requested_treatment!r}，spec={treatment!r}，"
            f"leg={resolved_leg!r}, path={spec_relative_path!r}"
        )
    verify_co_alchemical_ion_identity(
        spec,
        system=system,
        topology=topology,
        # Always pass the route requested by the current caller.  When a
        # legacy direct caller omits it, None deliberately means "no route
        # assertion"; never substitute the spec's self-declaration here.
        charge_treatment=requested_treatment,
        ligand_net_charge_e=q_l,
        context=f"B5 runtime identity leg={resolved_leg} path={spec_relative_path}",
    )
    return {
        "schema_version": 1,
        "leg": resolved_leg,
        "charge_treatment": treatment,
        "identity_protocol_version": int(spec["protocol_version"]),
        "fingerprint": str(spec["fingerprint"]),
        "ligand_net_charge_e": q_l,
        "lambda_direction": str(spec["lambda_direction"]),
        "ion_atom_indices": [int(ion["atom_index"]) for ion in spec["ions"]],
        "spec_relative_path": os.path.normpath(str(spec_relative_path)),
    }


def co_alchemical_ion_builder_identity_payload(
    *,
    system: Any,
    topology: Any,
    charge_treatment: Optional[str],
    ligand_net_charge_e: int,
) -> Optional[Dict[str, Any]]:
    """Recompute the coordinate-free identity of reserved co-ion dummies.

    The identity is derived from the current base System/Topology, never
    copied back from a manifest.  It is intentionally separate from the
    runtime spec fingerprint: the builder identity answers *which dummy
    particles were built*, while the runtime identity answers *which frozen
    spec was used by a leg*.
    """
    treatment = None if charge_treatment is None else str(charge_treatment)
    if treatment != CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
        return None
    q_l = int(ligand_net_charge_e)
    if q_l == 0:
        return None
    try:
        import openmm
        nb_force = next(
            force for force in system.getForces()
            if isinstance(force, openmm.NonbondedForce)
        )
    except (StopIteration, AttributeError) as exc:
        raise RuntimeError(
            "无法重算 reserved co-ion builder identity：System 缺少 NonbondedForce"
        ) from exc

    atoms = list(topology.atoms())
    candidates = []
    for atom in atoms:
        if str(atom.residue.name).upper() not in CO_ALCHEMICAL_ION_RESIDUE_NAMES:
            continue
        charge, sigma, epsilon = nb_force.getParticleParameters(int(atom.index))
        charge_e = float(charge.value_in_unit(unit.elementary_charge))
        if abs(charge_e) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            continue
        candidates.append(
            {
                "atom_index": int(atom.index),
                "residue_index": int(atom.residue.index),
                "residue_name": str(atom.residue.name),
                "element": str(getattr(atom.element, "symbol", "") or ""),
                "charge_at_lambda1_e": charge_e,
                "sigma_nm": float(sigma.value_in_unit(unit.nanometer)),
                "epsilon_kj_mol": float(
                    epsilon.value_in_unit(unit.kilojoule_per_mole)
                ),
                "mass_amu": float(
                    system.getParticleMass(int(atom.index)).value_in_unit(unit.dalton)
                ),
            }
        )
    candidates.sort(key=lambda item: item["atom_index"])
    expected_count = abs(q_l)
    if len(candidates) != expected_count:
        raise ValueError(
            "reserved co-ion builder identity 不满足 charge-transfer 数量契约："
            f"leg system 中找到 {len(candidates)} 个中性 ion-shaped dummy，"
            f"但配体净电荷 {q_l:+d} e 需要 {expected_count} 个。"
        )
    return {
        "schema_version": CO_ALCHEMICAL_ION_BUILDER_IDENTITY_SCHEMA_VERSION,
        "charge_treatment": treatment,
        "ligand_net_charge_e": q_l,
        "reserved_coion_count": expected_count,
        "ions": candidates,
        "restraint_protocol": {
            "protocol_version": int(CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION),
            "form": CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM,
            "reference_frame": CO_ALCHEMICAL_ION_RESTRAINT_REFERENCE_FRAME,
            "expression": CO_ALCHEMICAL_ION_RESTRAINT_EXPRESSION,
            "default_k_kj_per_mol_nm2": float(COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2),
            "default_r0_nm": float(COION_FLAT_BOTTOM_RADIUS_NM),
            "force_group": int(CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP),
        },
    }


# Descriptive alias used by cache/provenance callers and tests.
reserved_coion_builder_identity_payload = co_alchemical_ion_builder_identity_payload


# ============================================================================
# §13 验收阈值：全部落成命名常量并进 provenance
#
# 清单原文通篇写"预设阈值"但没给数——没有数就写不了 fail-closed 检查，也没法判
# 验收。§13 要求"必须在 Phase A 结束前落成常量并进 provenance，不许运行时凭感觉判"。
# 这里就是那份常量表；数值取 §13 的提案值，可改，但改必须改这里、且会进 provenance。
# ============================================================================

ACCEPTANCE_THRESHOLDS_VERSION = 2

# ---- §13.1 co-ion 几何 ----
COION_LIGAND_MIN_IMAGE_INITIAL_NM = 1.6
# [MEM-00h，2026-08-06 复核后拍板] 这个值**不再**要求等于
# `ibs_engine.SOFTCORE_CUTOFF_NM`——早先的注释/交叉检查测试把它当成 softcore
# cutoff 的派生量，但用户明确决定两者解耦：这是一条独立的、保守的几何安全
# 门（防止 co-ion 在动力学里游到离配体太近的地方），不是"必须等于 softcore
# cutoff 才有意义"的量。softcore cutoff 这次统一收敛到 1.0 nm（见
# ibs_engine.SOFTCORE_CUTOFF_NM），但这条门槛保持 1.2 nm 不变——没必要为了
# 那次 cutoff 收敛顺带把这里也降到 1.0，1.2 本来就更保守。
# 见 tests/test_dispersion_and_forcefield_protocol.py 里的对应测试
# （已改成断言"独立常量、不要求相等"，不再断言相等）。
COION_LIGAND_MIN_IMAGE_RUNTIME_NM = 1.2
# §2.2 多个单价 co-ion 分摊（|q_L| ≥ 2）：这些 dummy 彼此之间也必须留够安全边距。
# 2026-08-06 用 Ca²⁺(+2) 在真实水盒里第一次测这条路径就抓到：
# `runabfe._insert_reserved_coalchemical_ion_dummies` 原来只按"离配体质心最远"
# 独立给每个 dummy 打分，方盒里"离中心最远的 N 个点"天然会挤在同一个远角——
# 实测两个 dummy 相距 0.18~0.43 nm，λ→0 时两者同号各带 +1e，几乎贴脸的静电
# 排斥直接把 charging MBAR 拖到不收敛（ΔG≈660 kJ/mol，min_overlap<0.02）。
# 阈值取与 COION_LIGAND_MIN_IMAGE_INITIAL_NM 相同量级，不是拍脑袋另设一档。
COION_COION_MIN_IMAGE_INITIAL_NM = 1.6
COION_PROTEIN_HEAVY_ATOM_MIN_NM = 1.2
COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM = 3.0
COION_NEAREST_PHOSPHORUS_MIN_NM = 1.0
COION_FIRST_SHELL_WATER_CUTOFF_NM = 0.32
# 按离子类型给首层水配位数下限（§13.1）。键为残基名/元素的大写形式。
COION_FIRST_SHELL_MIN_WATER_COUNT = {
    "NA": 5, "SOD": 5, "K": 5, "POT": 5,
    "CL": 6, "CLA": 6,
}
COION_FLAT_BOTTOM_RADIUS_NM = 0.5
COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2 = 100.0

# ---- §13.2 数值自洽 ----
# 总电荷守恒容差 TOTAL_CHARGE_CONSERVATION_TOLERANCE_E 已在 B2 一节定义，此处不重复。
LIGAND_CHARGE_LAMBDA_TOLERANCE_E = 1.0e-6
ENDPOINT_ENERGY_RELATIVE_TOLERANCE = 1.0e-5
ENDPOINT_FORCE_MAX_ABS_TOLERANCE_KJ_PER_MOL_NM = 1.0e-3
# λ=0 端 ligand–environment 静电与 LJ 必须是**严格零**，不是"很小"。
DECOUPLED_ENDPOINT_ENERGY_ABS_TOLERANCE_KJ_PER_MOL = 1.0e-6
GROMACS_OPENMM_COMPONENT_RELATIVE_TOLERANCE = 1.0e-4
GROMACS_OPENMM_TOTAL_ABS_TOLERANCE_KJ_PER_MOL = 0.1

# ---- 预平衡轨迹的时间轴（生产者与消费者共用同一组数）----
#
# 为什么必须共用：mdtraj 读 DCD 时**不传播真实步长**，`traj.time` 是整数帧号，
# 所以 §9 的时间轴只能由"reporter 保存间隔 × integrator 步长"重建。这两个数
# 一个写在 `pre_equilibrate()` 的 DCDReporter 里、一个写在它的 Integrator 里，
# 各自散着就必然错开（实测 memtest 那条 10 ns 轨迹被当成 0.499 ns，差 20 倍）。
PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS = 10000
PRE_EQUILIBRATION_TIMESTEP_PS = 0.002


def pre_equilibration_frame_interval_ps(
    timestep_ps: Optional[float] = None,
    interval_steps: Optional[int] = None,
) -> float:
    """预平衡 DCD 每帧对应多少 ps。默认值即生产设置（10000 × 2 fs = 20 ps）。"""
    dt = PRE_EQUILIBRATION_TIMESTEP_PS if timestep_ps is None else float(timestep_ps)
    steps = (
        PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS
        if interval_steps is None
        else int(interval_steps)
    )
    if not np.isfinite(dt) or dt <= 0.0 or steps <= 0:
        raise ValueError(
            f"预平衡帧间隔参数非法：timestep_ps={dt!r}, interval_steps={steps!r}"
        )
    return dt * steps


# ---- 物理预平衡 block 收敛早停（PLAN 2026-08-26：5M 步地板+block 检查）----
#
# 只用于非膜体系（见 `abfe_pipeline.pre_equilibrate` 里的分支）。判的是**相邻
# block 之间的变化量**是否已经走平，不是绝对精度门——选紧只会少触发早停（退回
# 跑满 n_steps，与改动前等价，零风险），选松才有"提前停在没真正平衡"的风险，
# 所以初版数值偏保守，等真机 block 监控 CSV 攒出来再按数据松绑。
PRE_EQUILIBRATION_CONVERGENCE_MIN_STEPS = 1_000_000  # 2 ns 地板，硬性下限内不判据
PRE_EQUILIBRATION_CONVERGENCE_BLOCK_STEPS = 250_000  # 每 block 500 ps
PRE_EQUILIBRATION_CONVERGENCE_STABLE_BLOCKS_REQUIRED = 3  # 连续 3 个 block 都稳定才停
PRE_EQUILIBRATION_CONVERGENCE_TEMP_DRIFT_TOL_K = 1.5
PRE_EQUILIBRATION_CONVERGENCE_VOLUME_DRIFT_TOL_PERCENT = 0.5
PRE_EQUILIBRATION_CONVERGENCE_DENSITY_DRIFT_TOL_PERCENT = 0.5
PRE_EQUILIBRATION_CONVERGENCE_PE_DRIFT_TOL_KJ_PER_MOL_PER_ATOM = 0.02
PRE_EQUILIBRATION_CONVERGENCE_RMSD_DRIFT_TOL_NM = 0.02  # block-to-block Δ，非绝对值门
# 🔑 [2026-09-03] 配体**位姿漂移**（相对受体的刚体位移）单独一个容差，不再复用上面
# 那个 0.02。0.02 是给 backbone self-refit RMSD 定的——骨架内部柔性在 250k 步 block
# 之间确实只变动 ~0.005 nm。但配体在口袋里的位姿本身有 ~0.05 nm 的**热涨落幅度**：
# 4W53 实测（output_v3_seed20260908/pre_equilibration_convergence_monitor.csv，去掉
# 镜像污染后）16 个 block 里 13 个的 Δpose > 0.02，最大 0.0756。拿 0.02 去要求"逐
# block 变化小于它自身的自然涨落"，是原理上无法满足的判据——它测的不是收敛而是噪声，
# 结果预平衡永远跑满 n_equil_steps（实测 5,000,000 步，而温度/体积/密度/势能四条判据
# 从 1.25M 步之后就一直以 2.3–4.8 倍余量通过）。0.10 nm 取自实测涨落幅度的上包络。
PRE_EQUILIBRATION_CONVERGENCE_POSE_DRIFT_TOL_NM = 0.10
# 🔑 [2026-09-03] 配体位姿漂移的**绝对**上限（不是 block 间 Δ）。这是新增的 fail-closed
# 守卫，不是放松：超过它说明配体已经在预平衡期间离开了结合位姿，此时"收敛"没有意义。
# 口径对齐 fepsuite 的 EQ_RMSD_CUTOFF=0.4（abfe/rundir_template/para_conf.zsh：
# "If ligand RMSD during equilibration exceeds this value, calculation stops"）。
# 与 fepsuite 不同的是这里**不中止运行**，只是拒绝宣布稳定（于是退化为跑满
# n_equil_steps）并打一条显著警告——新增一条硬 abort 路径的风险比它挡掉的问题更大。
PRE_EQUILIBRATION_LIGAND_POSE_ESCAPE_TOL_NM = 0.4


# ---- §13.3 膜质量门（判据统一为"末段窗口内线性漂移小于阈值"）----
MEMBRANE_QUALITY_GATE_TAIL_WINDOW_NS = 20.0
APL_MAX_DRIFT_PERCENT_PER_NS = 0.2
APL_MAX_DEVIATION_FROM_LITERATURE_PERCENT = 3.0

# ---- 含蛋白膜的 APL：必须先扣掉蛋白横截面才能与纯脂文献值比 ----
#
# 实测 memtest（PROA 1 + POPC 90，08-02 那条 10 ns 轨迹）：
# raw APL = 横向面积 / 每叶脂质数 = 0.807 nm²，而 POPC 纯脂文献值 ≈ 0.645。
# 差值来自跨膜蛋白占掉的横向面积，**不是**体系有问题。所以 §13.3 那条
# 「与文献值差 ≤ 3%」若拿 raw APL 去比，会把物理上正常的含蛋白膜判失败；
# 先前的应对是干脆不设 literature_apl_nm2，代价是这道门整条缺席。
#
# ## 为什么是"最近原子归属"而不是"蛋白外扩一个探针半径"
#
# 第一版用的是"每个蛋白重原子外扩 0.17 nm 求并集"。实测它**系统性高估蛋白面积**：
# 外扩会沿蛋白周长加一圈宽 0.17 nm 的边（周长约 10 nm → 约 1.7 nm²），
# 于是校正后 APL = 0.564，比文献值低 12.6% —— 门会因为**方法偏差**而不过，
# 而不是因为体系有问题。那种门迟早会被"调参调绿"，正是本仓库反复吃亏的模式。
#
# 现在改为**无探针半径**的定义（Voronoi 式，APL@Voro 的同一思路）：把该叶片
# slab 内的脂质重原子与蛋白重原子放在一起，横向栅格的每个格子归给**最近的**那个
# 原子，蛋白面积 = 归给蛋白原子的格子面积之和。边界自动落在两者中间，没有可调的
# 半径；唯一的方法参数是栅格边长，而它只带来无偏的离散化误差。
PROTEIN_CROSS_SECTION_GRID_NM = 0.05
# 栅格边长的敏感性检查（用 2× 粗栅格复算若干帧，确认离散化误差可忽略）。
PROTEIN_CROSS_SECTION_GRID_SENSITIVITY_FACTOR = 2.0
# 敏感性只在若干等间隔帧上算（主序列仍逐帧），避免为一个诊断量把成本翻倍。
PROTEIN_CROSS_SECTION_SENSITIVITY_MAX_FRAMES = 25
# 校正后 APL 的观测量名。**不进** REQUIRED_MEMBRANE_QUALITY_OBSERVABLES：
# 判定层要能吃老报告和手工构造的观测量；缺它时绝对值门会退回未校正值，
# 但会把 criterion 写成 `..._uncorrected`，事后分得清当时比的是什么。
APL_PROTEIN_CORRECTED_OBSERVABLE = "apl_protein_corrected_nm2"

BILAYER_THICKNESS_MAX_DRIFT_NM_PER_TAIL_WINDOW = 0.05
PROTEIN_BACKBONE_MAX_RMSD_NM = 0.30
TRANSMEMBRANE_TILT_MAX_DRIFT_DEG = 5.0
POCKET_MAX_RMSD_NM = 0.20
LIGAND_HEAVY_ATOM_MAX_RMSD_NM = 0.25

# ---- §13.4 结果验收 ----
CROSS_REPEAT_MAX_STDDEV_KCAL_PER_MOL = 1.0
MIN_INDEPENDENT_REPEATS = 3
BENCHMARK_MIN_LIGANDS = 5
BENCHMARK_MAX_MAE_KCAL_PER_MOL = 1.5
BENCHMARK_MAX_ABS_OUTLIER_KCAL_PER_MOL = 3.0
# §3.0 空腔填充迟滞：正反向 / 双起点 stage 2 的 ΔF 差 ≤ 2σ。
STAGE2_HYSTERESIS_MAX_SIGMA = 2.0

# 精度状态词汇表。**三个值，不是布尔** —— 「没测」与「测了不达标」必须分得开。
PRECISION_UNMEASURED = "UNMEASURED"
PRECISION_MEETS = "MEETS_CROSS_REPEAT_TARGET"
PRECISION_EXCEEDS = "EXCEEDS_CROSS_REPEAT_TARGET"


def cross_repeat_precision(
    repeat_delta_g_kcal_per_mol,
    *,
    min_repeats: int = MIN_INDEPENDENT_REPEATS,
    max_stddev_kcal_per_mol: float = CROSS_REPEAT_MAX_STDDEV_KCAL_PER_MOL,
) -> Dict[str, Any]:
    """跨**独立重复**的样本标准差 vs 1 kcal/mol。数据不够就诚实返回 `UNMEASURED`。

    🔑🔑 [2026-09-15 用户拍板] **这是目前唯一不依赖 MBAR 渐近性的统计精度判据，
    而它此前只是两个声明在 `acceptance_thresholds_payload()` 里的常量 —— 全仓没有
    任何代码计算它。** 与此同时，一个**不可信**的量（MBAR 的
    `max_endpoint_uncertainty_kJ_mol ≤ 1.0`）在卡 `converged` 并驱动自动补帧。

    为什么 MBAR 的 σ 不能当精度门：
      · 它是**渐近**估计，前提是「去相关后样本近独立」。本仓实测 `N_eff/g` 常年
        个位数、`g` 能从 15 涨到 168 —— 前提不成立时它系统性**偏小**。
      · 它对「**该采的构型一次都没采到**」结构性失明。本仓两次大错都是这个形状：
        4W53 stage2 偏 **+32 kJ/mol**（单轨迹重加权抓不到空腔重组熵项）、
        decharging 偏 **−88 kJ/mol**（两腿分子内库仑）。两次 σ 都很小、都没报警。

    ⚠️⚠️ **本判据测的是「重复性」，不是「真实误差」。**
    三个重复可以共享**同一个系统偏差**（同一份力场、同一个建系脚本、同一条 λ 路径、
    同一个估计器）—— 它们会一致地错，而 s 很小。所以：
      · 达标（`MEETS_*`）**只说明可重复**，**不构成**「真实误差 < 1 kcal/mol」的证明；
      · 不达标（`EXCEEDS_*`）是**确定**的坏消息（连自己都重复不出来）。
    真实误差还需要独立手段：已知答案体系的基准、循环闭合、系统误差审查。

    ⚠️ **比的是样本标准差 s（ddof=1），不是标准误 s/√n。**
    用 s/√n 会把「重复之间离散得厉害」除以 √n 粉饰成「均值很准」—— 那是对
    *均值*的不确定度，不是我们要问的*离散度*。n=3 时两者差 1.73 倍。

    返回 dict（永远带 `status`，永远不抛）：
        status / n_repeats / sample_stddev_kcal_per_mol / threshold_kcal_per_mol
        / mean_kcal_per_mol / values_kcal_per_mol / unmeasured_reason
        / measures_repeatability_not_true_error
    """
    out: Dict[str, Any] = {
        "status": PRECISION_UNMEASURED,
        "n_repeats": 0,
        "min_independent_repeats": int(min_repeats),
        "sample_stddev_kcal_per_mol": None,
        "threshold_kcal_per_mol": float(max_stddev_kcal_per_mol),
        "mean_kcal_per_mol": None,
        "values_kcal_per_mol": [],
        "unmeasured_reason": None,
        "stddev_convention": "sample_stddev_ddof_1_NOT_standard_error",
        "measures_repeatability_not_true_error": (
            "跨重复 SD 只测重复性。三个重复可能共享同一个系统偏差（力场/建系/λ 路径/"
            "估计器都一样），一致地错而 s 很小。达标不构成真实误差 < 1 kcal/mol 的证明。"
        ),
    }
    vals = []
    for x in (repeat_delta_g_kcal_per_mol or []):
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            vals.append(v)
    out["values_kcal_per_mol"] = vals
    out["n_repeats"] = len(vals)
    if len(vals) < int(min_repeats):
        out["unmeasured_reason"] = (
            f"只有 {len(vals)} 个有限的独立重复，少于 {int(min_repeats)} 个 ⟹ "
            "跨重复离散度**没有测**。这不是「达标」也不是「不达标」。"
        )
        return out
    s = float(np.std(np.asarray(vals, dtype=float), ddof=1))
    out["sample_stddev_kcal_per_mol"] = s
    out["mean_kcal_per_mol"] = float(np.mean(vals))
    if not np.isfinite(s):
        out["unmeasured_reason"] = "样本标准差不是有限数"
        return out
    out["status"] = (PRECISION_MEETS if s <= float(max_stddev_kcal_per_mol)
                     else PRECISION_EXCEEDS)
    return out


def acceptance_thresholds_payload() -> Dict[str, Any]:
    """§13 全部阈值的可序列化快照，供 provenance 落盘。

    目的是让每一份结果都能回答"当时用的是哪套阈值"——阈值改了而结果没重跑，
    对照 provenance 就能看出来。
    """
    return {
        "version": ACCEPTANCE_THRESHOLDS_VERSION,
        "coion_geometry": {
            "ligand_min_image_initial_nm": COION_LIGAND_MIN_IMAGE_INITIAL_NM,
            "ligand_min_image_runtime_nm": COION_LIGAND_MIN_IMAGE_RUNTIME_NM,
            "coion_coion_min_image_initial_nm": COION_COION_MIN_IMAGE_INITIAL_NM,
            "protein_heavy_atom_min_nm": COION_PROTEIN_HEAVY_ATOM_MIN_NM,
            "membrane_midplane_min_abs_z_nm": COION_MEMBRANE_MIDPLANE_MIN_ABS_Z_NM,
            "nearest_phosphorus_min_nm": COION_NEAREST_PHOSPHORUS_MIN_NM,
            "first_shell_water_cutoff_nm": COION_FIRST_SHELL_WATER_CUTOFF_NM,
            "first_shell_min_water_count": dict(COION_FIRST_SHELL_MIN_WATER_COUNT),
            "flat_bottom_radius_nm": COION_FLAT_BOTTOM_RADIUS_NM,
            "flat_bottom_k_kj_per_mol_nm2": COION_FLAT_BOTTOM_K_KJ_PER_MOL_NM2,
        },
        "numerical_selfconsistency": {
            "total_charge_conservation_e": TOTAL_CHARGE_CONSERVATION_TOLERANCE_E,
            "ligand_charge_lambda_e": LIGAND_CHARGE_LAMBDA_TOLERANCE_E,
            "endpoint_energy_relative": ENDPOINT_ENERGY_RELATIVE_TOLERANCE,
            "endpoint_force_max_abs_kj_per_mol_nm": (
                ENDPOINT_FORCE_MAX_ABS_TOLERANCE_KJ_PER_MOL_NM
            ),
            "decoupled_endpoint_energy_abs_kj_per_mol": (
                DECOUPLED_ENDPOINT_ENERGY_ABS_TOLERANCE_KJ_PER_MOL
            ),
            "gromacs_openmm_component_relative": (
                GROMACS_OPENMM_COMPONENT_RELATIVE_TOLERANCE
            ),
            "gromacs_openmm_total_abs_kj_per_mol": (
                GROMACS_OPENMM_TOTAL_ABS_TOLERANCE_KJ_PER_MOL
            ),
        },
        "membrane_quality_gate": {
            "tail_window_ns": MEMBRANE_QUALITY_GATE_TAIL_WINDOW_NS,
            "apl_max_drift_percent_per_ns": APL_MAX_DRIFT_PERCENT_PER_NS,
            "apl_max_deviation_from_literature_percent": (
                APL_MAX_DEVIATION_FROM_LITERATURE_PERCENT
            ),
            # 蛋白横截面校正的方法参数：绝对值门比的是校正后的 APL，
            # 而校正结果依赖这三个数，所以它们必须与阈值一起进 provenance。
            "protein_cross_section_grid_nm": PROTEIN_CROSS_SECTION_GRID_NM,
            "protein_cross_section_grid_sensitivity_factor": (
                PROTEIN_CROSS_SECTION_GRID_SENSITIVITY_FACTOR
            ),
            "bilayer_thickness_max_drift_nm_per_tail_window": (
                BILAYER_THICKNESS_MAX_DRIFT_NM_PER_TAIL_WINDOW
            ),
            "protein_backbone_max_rmsd_nm": PROTEIN_BACKBONE_MAX_RMSD_NM,
            "transmembrane_tilt_max_drift_deg": TRANSMEMBRANE_TILT_MAX_DRIFT_DEG,
            "pocket_max_rmsd_nm": POCKET_MAX_RMSD_NM,
            "ligand_heavy_atom_max_rmsd_nm": LIGAND_HEAVY_ATOM_MAX_RMSD_NM,
        },
        "result_acceptance": {
            "cross_repeat_max_stddev_kcal_per_mol": (
                CROSS_REPEAT_MAX_STDDEV_KCAL_PER_MOL
            ),
            "min_independent_repeats": MIN_INDEPENDENT_REPEATS,
            "benchmark_min_ligands": BENCHMARK_MIN_LIGANDS,
            "benchmark_max_mae_kcal_per_mol": BENCHMARK_MAX_MAE_KCAL_PER_MOL,
            "benchmark_max_abs_outlier_kcal_per_mol": (
                BENCHMARK_MAX_ABS_OUTLIER_KCAL_PER_MOL
            ),
            "stage2_hysteresis_max_sigma": STAGE2_HYSTERESIS_MAX_SIGMA,
        },
    }
