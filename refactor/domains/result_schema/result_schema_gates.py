

# ============================================================================
# [P1-12] final-result 统一 schema / sanity gate（2026-08-30）
# ============================================================================
# 一份"腿级最终结果"缺必需热力学字段、ΔG/误差非有限、误差为负、或自带
# converged 诊断却不为 True 时，任何消费者都不得把它当成有效结果复用或汇总。
# 判据只此一份：cache 早退（abfe_pipeline）、两腿汇总与回退分析
# （runabfe）、热力学循环闭合（combine_binding_free_energy）都走这里，
# 不许再各自发明"缺了就补 0.0"之类的宽松口径。
FINAL_LEG_RESULT_REQUIRED_FIELD_SETS = (
    # 一份合法的腿级最终结果必须**同时**带有以下任一组 (ΔG, 误差) 字段：
    ("total_delta_G_complex_kJ_mol", "total_error_kJ_mol"),  # dual_lambda / single / 2D 腿
    ("delta_G_total_kJ_mol", "error_leg_kJ_mol"),            # traditional 两腿汇总
)


class FinalResultValidationError(RuntimeError):
    """腿级最终结果未通过统一 schema/sanity gate（P1-12）。"""


def validate_boresch_attachment_result(payload: Any) -> Tuple[float, float]:
    """Require a converged, finite A′→A contribution before cache/cycle use."""
    if not isinstance(payload, dict) or payload.get("converged") is not True:
        raise FinalResultValidationError("Boresch attachment stage0 必须带 converged=true")
    values = []
    for field in ("attachment_delta_G_kJ_mol", "attachment_error_kJ_mol"):
        try:
            value = float(payload[field])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FinalResultValidationError(f"Boresch attachment stage0 缺少或非法 {field}") from exc
        if not math.isfinite(value) or value < 0.0:
            raise FinalResultValidationError(f"Boresch attachment stage0 {field} 非有限或 < 0")
        values.append(value)
    return values[0], values[1]


def validate_final_leg_result(
    payload: Any,
    *,
    context: str = "final-result",
    source: str = "memory",
) -> Dict[str, Any]:
    """[P1-12] 腿级最终结果的统一 schema/sanity gate。

    拒绝（raise `FinalResultValidationError`）：
      * payload 不是 dict；
      * `FINAL_LEG_RESULT_REQUIRED_FIELD_SETS` 里没有任何一组的两个字段**都在**；
      * ΔG 或误差不是有限数值（NaN/Inf/字符串等）；
      * 误差为负；
      * nested traditional stages or Boresch attachment lack explicit convergence;
      * analytical Boresch release is present without attachment;
      * payload 自带 ``converged`` 字段且值不是 ``True``（缺席不算 —— 旧版
        本的生产 writer 本就不写这个字段，不能让所有历史结果失效；但一旦
        写了，就必须是 True）。

    返回抽出的 ``{"delta_G_field", "delta_G_kJ_mol", "error_field",
    "error_kJ_mol"}``，方便调用方直接用同一组字段，不必再各取一遍。

    `context`/`source` 只用于报错信息定位（哪条腿、内存还是哪个文件）。
    """
    if not isinstance(payload, dict):
        raise FinalResultValidationError(
            f"[{context}] ({source}) 最终结果不是 dict，而是 {type(payload).__name__}；拒绝消费"
        )
    chosen = None
    for dg_field, err_field in FINAL_LEG_RESULT_REQUIRED_FIELD_SETS:
        if dg_field in payload and err_field in payload:
            chosen = (dg_field, err_field)
            break
    if chosen is None:
        missing = [
            field
            for dg_field, err_field in FINAL_LEG_RESULT_REQUIRED_FIELD_SETS
            for field in (dg_field, err_field)
            if field not in payload
        ]
        raise FinalResultValidationError(
            f"[{context}] ({source}) 缺少必需热力学字段 {sorted(set(missing))}；"
            "拒绝把残缺结果当成有效最终结果消费（P1-12）"
        )
    dg_field, err_field = chosen
    try:
        dg_value = float(payload[dg_field])
        err_value = float(payload[err_field])
    except (TypeError, ValueError) as exc:
        raise FinalResultValidationError(
            f"[{context}] ({source}) {dg_field}/{err_field} 不是数值"
            f"（{payload[dg_field]!r}/{payload[err_field]!r}）：{exc}"
        ) from exc
    if not (math.isfinite(dg_value) and math.isfinite(err_value)):
        raise FinalResultValidationError(
            f"[{context}] ({source}) {dg_field}/{err_field} 含非有限值"
            f"（{dg_value!r}/{err_value!r}）；拒绝写进或复用最终结果（P1-12）"
        )
    if err_value < 0.0:
        raise FinalResultValidationError(
            f"[{context}] ({source}) {err_field} 为负数（{err_value!r}）；拒绝消费（P1-12）"
        )
    if "converged" in payload and payload.get("converged") is not True:
        raise FinalResultValidationError(
            f"[{context}] ({source}) 自带 converged 诊断但值不是 True"
            f"（converged={payload.get('converged')!r}）；拒绝消费（P1-12）"
        )
    for stage_name in ("stage_decharging", "stage_vanishing"):
        if stage_name in payload:
            stage = payload[stage_name]
            # 🔑🔑 [2026-09-15 用户拍板] **stage 的 `converged` 已被删除。**
            # 它把「算出了数」误报成「精度已验收」：那个合取里四项全是未经标定的
            # 拟合阈值，其中 `max_endpoint_uncertainty ≤ 1.0` 抄自一个默认关闭的
            # early-stop 启发式、全仓无依据，而且是**逐段**量、目标却是两腿合成的
            # σ_bind。改成两个正交状态后，这道门只管**结构**那一半：
            #   · `analysis_status` —— 只含不依赖拟合阈值的硬不变量（路径完整、
            #     数值有限、结构自洽）。这里 fail-closed。
            #   · `precision_status` —— 跨重复离散度，单次 run 结构上恒为
            #     `UNMEASURED`。**它不在这里判**：本函数是「这份结果能不能被消费/
            #     复用」的 schema 门，而「能不能作为**已验收**结果发布」是发布路径
            #     的事。在这里要求精度达标会让流水线连产出都做不到。
            if not isinstance(stage, dict):
                raise FinalResultValidationError(
                    f"[{context}] ({source}) {stage_name} 不是 dict；拒绝消费"
                )
            # ⚠️ **按生产者分流，不要按键名一刀切。** 这个位置上会出现两种 stage：
            #   · IBS / dual-lambda 腿 ⟹ `solve_stage_integrated`，它带
            #     `covariance_chain_segments`，且已改用 `analysis_status`；
            #   · traditional 腿 ⟹ `run_leg`，**另一个生产者**，它的 `converged`
            #     没有被删、语义也不同（不是那四项阈值的合取）。
            # 一刀切要求 `analysis_status` 会把 traditional 腿整条打死。
            _is_ibs_producer = ("analysis_status" in stage
                                or "covariance_chain_segments" in stage)
            if _is_ibs_producer:
                _st = stage.get("analysis_status")
                if _st != "ANALYSIS_COMPLETE":
                    _why = stage.get("analysis_incomplete_reasons")
                    raise FinalResultValidationError(
                        f"[{context}] ({source}) {stage_name} 的 analysis_status="
                        f"{_st!r}（需要 'ANALYSIS_COMPLETE'）；拒绝消费。"
                        + (f" 缺什么：{_why}" if _why else
                           " 带 covariance_chain_segments 却没有 analysis_status ⟹ "
                           "删除 `converged` 之前的老产物，fail-closed：老产物的 "
                           "`converged=true` **不等于**结构完整，它还混着四项未标定"
                           "的阈值（min_overlap / target_support / "
                           "min_decorrelated_samples / max_endpoint_uncertainty≤1.0），"
                           "不得据此放行。")
                    )
            elif stage.get("converged") is not True:
                raise FinalResultValidationError(
                    f"[{context}] ({source}) {stage_name}（traditional 生产者）"
                    f"缺少 converged=true；拒绝消费"
                )
    attachment = payload.get("boresch_attachment_result")
    if attachment is not None:
        validate_boresch_attachment_result(attachment)
    attachment_summary = payload.get("boresch_attachment") or {}
    if not isinstance(attachment_summary, dict):
        raise FinalResultValidationError(f"[{context}] boresch_attachment 不是结果字典")
    correction_diagnostics = payload.get("boresch_correction_diagnostics") or {}
    if not isinstance(correction_diagnostics, dict):
        raise FinalResultValidationError(f"[{context}] boresch_correction_diagnostics 不是字典")
    requires_attachment = bool(
        correction_diagnostics.get("uses_analytical_release_formula")
    )
    if requires_attachment and attachment_summary.get("present") is not True:
        raise FinalResultValidationError(f"[{context}] Boresch release 缺少 attachment stage0")
    if attachment_summary.get("present"):
        validate_boresch_attachment_result({
            "converged": attachment_summary.get("converged"),
            "attachment_delta_G_kJ_mol": attachment_summary.get("delta_G_kJ_mol"),
            "attachment_error_kJ_mol": attachment_summary.get("error_kJ_mol"),
        })
    return {
        "delta_G_field": dg_field,
        "delta_G_kJ_mol": dg_value,
        "error_field": err_field,
        "error_kJ_mol": err_value,
    }


def _validate_cycle_input_scalar(value: float, name: str) -> float:
    """combine_binding_free_energy 输入的有限性/误差非负检查（P1-12）。"""
    try:
        value_f = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FinalResultValidationError(
            f"combine_binding_free_energy: {name} 不是数值（{value!r}）；"
            "拒绝在热力学循环里汇总损坏输入（P1-12）"
        ) from exc
    if not math.isfinite(value_f):
        raise FinalResultValidationError(
            f"combine_binding_free_energy: {name} 非有限（{value_f!r}）；"
            "拒绝在热力学循环里汇总损坏输入（P1-12）"
        )
    return value_f


def combine_binding_free_energy(
    *,
    dg_complex_kJ_mol: float,
    dg_solvent_kJ_mol: float,
    err_complex_kJ_mol: float = 0.0,
    err_solvent_kJ_mol: float = 0.0,
    dg_boresch_kJ_mol: float = 0.0,
    boresch_already_included_in_complex: bool = True,
    apbs_correction_kJ_mol: float = 0.0,
    charge_transfer_reservoir_correction_kJ_mol: float = 0.0,
    charge_transfer_reservoir_error_kJ_mol: float = 0.0,
    complex_conformer_summary: Optional[Dict[str, Any]] = None,
    solvent_conformer_summary: Optional[Dict[str, Any]] = None,
    strict_cross_leg_conformer: bool = False,
) -> Dict[str, Any]:
    """🔑 [ATT-09] 热力学循环闭合的**唯一**实现。

    上面 `THERMODYNAMIC_CYCLE_DOC` 记的公式：

        ΔG_bind = ΔG_solvent - ΔG_complex + ΔG_APBS + ΔG_reservoir

    其中 ΔG_complex **按约定已经包含** Boresch 标准态释放项。若调用方拿到的
    complex 腿还没烘焙这一项（例如 `TraditionalABFEPipeline.run_full` 是用
    `boresch_correction=0.0` 调的），把 `boresch_already_included_in_complex=False`
    传进来，这里会替它减一次——**只减一次**。

    为什么要有这个函数：改之前同一个公式在四处独立维护，而且它们并不等价——

      * `runabfe.main()`：Boresch 已内含，**加了** APBS；
      * `runabfe.run_traditional_mode()`：Boresch 显式减，**完全没有** APBS；
      * `runabfe.run_post_analysis()`：Boresch 条件置零，加了 APBS（从
        `run_provenance.json` 重推）；
      * `ABFEPipeline.run_full_abfe_loop()`：Boresch 已内含，**完全没有** APBS。

    也就是说后两条路径对带电配体会静默漏掉整项有限尺寸静电修正。这不是代码
    整洁问题，是数值错误。统一到这里之后，那两条路径的输出会变——那是修复。

    误差：两腿采样独立，且 reservoir-release correction 的 complex/solvent 测量
    误差独立，按平方和开根合并。Boresch 解析释放项与 APBS 都是确定性解析量，
    没有独立采样方差，不并入。

    返回一个自带记账字段的 dict，调用方直接摊进自己的结果 JSON，
    不要再在外面重算任何一项。
    """
    # 🔑 [P1-12] 唯一循环闭合 helper 自己也要把住输入门：ΔG 非有限、误差非有限
    # 或为负，一律 raise —— 不能让 NaN/Inf 在公式里静默传播成"看似成功的
    # ΔG_bind"，更不能把负误差平方加进方差。ΔG 本身允许任意符号。
    dg_complex = _validate_cycle_input_scalar(dg_complex_kJ_mol, "dg_complex_kJ_mol")
    dg_solvent = _validate_cycle_input_scalar(dg_solvent_kJ_mol, "dg_solvent_kJ_mol")
    err_complex = _validate_cycle_input_scalar(err_complex_kJ_mol or 0.0, "err_complex_kJ_mol")
    err_solvent = _validate_cycle_input_scalar(err_solvent_kJ_mol or 0.0, "err_solvent_kJ_mol")
    dg_boresch = _validate_cycle_input_scalar(dg_boresch_kJ_mol or 0.0, "dg_boresch_kJ_mol")
    apbs = _validate_cycle_input_scalar(apbs_correction_kJ_mol or 0.0, "apbs_correction_kJ_mol")
    reservoir = _validate_cycle_input_scalar(
        charge_transfer_reservoir_correction_kJ_mol or 0.0,
        "charge_transfer_reservoir_correction_kJ_mol",
    )
    reservoir_err = _validate_cycle_input_scalar(
        charge_transfer_reservoir_error_kJ_mol or 0.0,
        "charge_transfer_reservoir_error_kJ_mol",
    )
    for _err_name, _err_value in (
        ("err_complex_kJ_mol", err_complex),
        ("err_solvent_kJ_mol", err_solvent),
        ("charge_transfer_reservoir_error_kJ_mol", reservoir_err),
    ):
        if _err_value < 0.0:
            raise FinalResultValidationError(
                f"combine_binding_free_energy: {_err_name} 为负数（{_err_value!r}）；"
                "拒绝在热力学循环里汇总（P1-12）"
            )

    # 只有 complex 腿还没烘焙释放项时，公式里才再减一次。
    boresch_term = 0.0 if boresch_already_included_in_complex else dg_boresch

    delta_g_bind_uncorrected = dg_solvent - dg_complex - boresch_term
    delta_g_bind = delta_g_bind_uncorrected + apbs + reservoir
    total_err = float(np.sqrt(err_complex ** 2 + err_solvent ** 2 + reservoir_err ** 2))

    # [P0-12a / §3.0] 汇总 ΔG_bind 之前先判跨腿构象一致性：不重叠就不许汇总。
    # 门放在这里（热力学循环闭合的唯一实现）而不是各调用点，理由与 ATT-09 相同 ——
    # 公式只有一份，门也只能有一份。
    conformer_report = evaluate_cross_leg_conformer_consistency(
        complex_conformer_summary, solvent_conformer_summary
    )
    # 🔑 [2026-09-12 老板裁决] **这道门最多是 WARN 级，不阻断汇总。**
    #
    # 两条腿的配体构象差在这类装置上是**必然产物**而不是采样不足：溶剂腿里
    # 配体没有任何东西撑着、vdW 一关就塌；复合物腿里主体腔把它卡住、塌不了。
    # 拿它去硬拦 ΔG_bind 等于用一个装置固有的性质否决整条热力学循环。
    #
    # ⚠️ 但它**也不能反过来当作"完全跳出"的判据** —— 不是"警告了就等于没事"。
    # 所以：判据一字不改、report 完整带出去、不重叠时 `cross_leg_conformer_gate`
    # 永远写成 `WARN`，读数的人一眼看得见这条路径上有未计入的构象自由能差。
    # 想要硬阻断的调用方显式传 `strict_cross_leg_conformer=True`。
    # 口径跟 `assert_cross_leg_conformer_consistency` 一字不差：
    # `evaluated=False`（判不了）不算不合格，只有"评估了且没过"才 WARN。
    _r = conformer_report or {}
    _conformer_ok = not (_r.get("evaluated") and not _r.get("passed"))
    if strict_cross_leg_conformer:
        assert_cross_leg_conformer_consistency(conformer_report)

    return {
        # PASSED / WARN —— **永远**写，永远带着完整 report。
        # WARN 不是"已处理"，是"这个 ΔG_bind 里有一项构象自由能差没被计入"。
        "cross_leg_conformer_gate": (
            "NOT_EVALUATED" if not _r.get("evaluated")
            else ("PASSED" if _conformer_ok else "WARN")
        ),
        "cross_leg_conformer_report": conformer_report,
        "complex_delta_G_kJ_mol": dg_complex,
        "solvent_delta_G_kJ_mol": dg_solvent,
        "boresch_correction_kJ_mol": dg_boresch,
        "boresch_correction_already_included_in_complex_delta_G": bool(
            boresch_already_included_in_complex
        ),
        # 公式里真正被减掉的那一项（已内含时为 0）；与上面那个"物理量本身"区分开，
        # 下游据此判断能不能再对 complex_delta_G 二次扣减。
        "boresch_term_subtracted_kJ_mol": boresch_term,
        "apbs_correction_kJ_mol": apbs,
        "charge_transfer_reservoir_correction_kJ_mol": reservoir,
        "charge_transfer_reservoir_error_kJ_mol": reservoir_err,
        "delta_G_bind_uncorrected_kJ_mol": delta_g_bind_uncorrected,
        "delta_G_bind_kJ_mol": delta_g_bind,
        "delta_G_bind_kcal_mol": delta_g_bind / 4.184,
        "total_error_kJ_mol": total_err,
        "total_error_kcal_mol": total_err / 4.184,
        "cycle_formula": (
            "delta_G_bind = delta_G_solvent - delta_G_complex"
            " - boresch_term_subtracted + delta_G_APBS"
            " + delta_G_charge_transfer_reservoir"
        ),
        # [P0-12a] 循环闭合的前提是两条腿采的是同一个构象族。判不了门时
        # `evaluated=False` 会如实记录，不会伪装成"通过"。
        "ligand_conformer_cross_leg": conformer_report,
    }


def boresch_dihedral_rad(a, b, c, d):
    """四点二面角，rad，范围 [-π, π]，**标准（IUPAC）符号约定**。

    🚨 2026-07-29 事故的根源就在这个符号上，务必不要"顺手简化"回去。

    Boresch 限制势的参考值 phiA0/phiB0/phiC0 是喂给 OpenMM
    `CustomCompoundBondForce` 表达式里的 `dihedral(p1,p2,p3,p4)` 的
    （见 `LambdaDependentBoreschForce`）。OpenMM 的 `dihedral()`、
    `mdtraj.compute_dihedrals` 用的都是标准约定：

        n1 = b1×b2, n2 = b2×b3,  φ = atan2( (n1×n2)·b2̂ , n1·n2 )

    此前 abfe_core 里有**四份**手写副本都写成了

        m1 = n1 × b2̂ ;  φ = atan2(m1·n2, n1·n2)

    而 (n1×b2̂)·n2 = −(n1×n2)·b2̂，所以那四份返回的是 **−φ**。距离和键角不受
    影响（arccos 无符号），只有三个二面角整体反号——也就是给出限制势参考几何的
    **镜像**。

    实测后果（`output_lrc_fix`，2026-07-29 02:02）：带 Boresch 的 rebalance 用
    `boresch_simple.json` 的（mdtraj 算的、正确的）参考值把配体稳稳按在自己的
    pose 上；紧接着 `update_boresch_from_last_frame` 用错号的副本重算并**覆盖**了
    参考值，提交下去的 phiA0/phiB0/phiC0 全部反号：

        phiA0  +1.6696 → −1.7168      phiB0  −1.8045 → +1.8136
        phiC0  −0.6839 → +0.6163      (r0/θA/θB 只差 <0.03，因为它们无符号)

    于是 attachment 腿的 λ=1 参考态变成了当前 pose 的镜像，每个二面角都坐在
    k(1−cosΔ) 的势壁顶上（Δ≈π ⟹ ≈2k）：λ=0 实测 ⟨U_B⟩=777 kJ/mol、
    max=1115 ≈ Σ2k_φ=1140，ΔG(A′→A) 从应有的 ~5.5 kJ/mol 涨到 98.8 kJ/mol，
    BAR/TI 一致性门失败（8.27 > 8.13 kJ/mol）。

    符号自检（可手算复核）：a=(0,1,0) b=(0,0,0) c=(1,0,0) d=(1,0,1) ⟹ φ=+π/2。
    见 `test_boresch_dihedral_convention.py`，它把本函数直接钉在 OpenMM
    `dihedral()` 上。

    退化情形（|b2|≈0 或两个法向量之一退化）返回 0.0，与被替换的四份副本中最
    保守的那份（`calc_boresch_from_last_frame`）保持一致。
    """
    b1 = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    b2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    b3 = np.asarray(d, dtype=np.float64) - np.asarray(c, dtype=np.float64)
    norm_b2 = float(np.linalg.norm(b2))
    if norm_b2 < 1e-6:
        return 0.0
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    denom = float(np.linalg.norm(n1)) * float(np.linalg.norm(n2))
    if denom < 1e-10:
        return 0.0
    return float(np.arctan2(np.dot(np.cross(n1, n2), b2 / norm_b2), np.dot(n1, n2)))


def calculate_boresch_analytical_correction(eq, fc, T=300.0):
    """
    计算 Boresch 解析修正。

    返回值是“解耦采样中保留 Boresch restraint”时需要加到 leg 上的
    标准态释放修正:

        ΔG_release = -RT ln[
            (8π² V° / (r0² sinθA sinθB))
            * sqrt(Kr KθA KθB KφA KφB KφC) / (2πRT)^3
        ]

    6 个谐振 Boresch 自由度的高斯积分给出 (2πRT)^3，而不是 1.5 次方。
    【强制标准单位】kJ/mol/nm², nm, rad
    ⚠️ 注意：eq["r0"] 必须是 nm 单位（不是 Å）
    ⚠️ 注意：fc["kr"] 必须为 kJ/mol/nm²，fc["kthetaA"] 等为 kJ/mol/rad²
    """
    T = T.value_in_unit(unit.kelvin) if hasattr(T, "value_in_unit") else float(T)
    R = constants.R / 1000.0
    RT = R * T
    V0 = 1.6605  # nm³ (标准摩尔体积)

    # ✅ 修复1：增加单位量级与物理合理性断言
    kr, ktA, ktB = fc.get("kr", 0), fc.get("kthetaA", 0), fc.get("kthetaB", 0)
    if not (50 <= kr <= 5000):
        raise ValueError(f"kr 超出合理范围 [50, 5000] kJ/mol/nm²: {kr}")
    # ✅ 与 GeometricRestraintEstimator 的 clip 范围 [10, 1000] 保持一致
    # （见 force_constant_ranges["kthetaA"/"kthetaB"]），否则该估计器给出的合法
    # 力常数（500~1000 区间）会在这里被误判为"超出合理范围"而硬报错。
    if not (10 <= ktA <= 1000 and 10 <= ktB <= 1000):
        raise ValueError("角度力常数 ktA/ktB 建议范围 [10, 1000] kJ/mol/rad²")

    # ✅ 强制标准单位，不再进行任何转换
    r0 = eq["r0"]  # nm
    thA = eq["thetaA0"]  # rad
    thB = eq["thetaB0"]  # rad

    kr = fc["kr"]  # kJ/mol/nm²
    ktA = fc["kthetaA"]  # kJ/mol/rad²
    ktB = fc["kthetaB"]
    kpA = fc["kphiA"]
    kpB = fc["kphiB"]
    kpC = fc["kphiC"]

    Kdet = kr * ktA * ktB * kpA * kpB * kpC
    if Kdet <= 0:
        raise ValueError("Boresch 力常数存在零值或负值，无法计算解析修正")
    
    sin_t = math.sin(thA) * math.sin(thB)

    if sin_t < 1e-4:
        raise ValueError("Boresch 锚点几何奇点 (sinθ≈0)")

    standard_state_factor = (8.0 * math.pi**2 * V0) / (r0**2 * sin_t)
    restraint_integral_factor = math.sqrt(Kdet) / ((2.0 * math.pi * RT) ** 3.0)
    argument = standard_state_factor * restraint_integral_factor
    if argument <= 0 or not math.isfinite(argument):
        raise ValueError(f"Boresch 解析修正对数参数异常: {argument}")

    return -RT * math.log(argument)


def _boresch_cosine_release_reference(eq, fc, T=300.0) -> float:
    """Numerical release free energy for the *actual* cosine restraint.

    ``calculate_boresch_analytical_correction`` intentionally retains the
    historical Gaussian expression (and its public numeric contract).  The
    OpenMM force below, however, uses ``k*(1-cos(delta))`` for angles and
    dihedrals.  This quadrature evaluates the corresponding six-coordinate
    configurational integral, including ``r²`` and ``sin(theta)`` Jacobians,
    so callers can report the harmonic-vs-cosine model discrepancy instead of
    claiming a zero uncertainty.  It is a deterministic diagnostic, not a
    replacement for a sampled attachment leg.
    """
    T = T.value_in_unit(unit.kelvin) if hasattr(T, "value_in_unit") else float(T)
    if not math.isfinite(T) or T <= 0.0:
        raise ValueError("Boresch 数值参考需要正有限温度")
    R = constants.R / 1000.0
    RT = R * T
    r0 = float(eq["r0"])
    theta_a = float(eq["thetaA0"])
    theta_b = float(eq["thetaB0"])
    if not math.isfinite(r0) or r0 <= 0.0:
        raise ValueError("Boresch r0 必须为正有限 nm")
    kr = float(fc["kr"])
    kta = float(fc["kthetaA"])
    ktb = float(fc["kthetaB"])
    kphis = [float(fc[k]) for k in ("kphiA", "kphiB", "kphiC")]
    if any((not math.isfinite(k) or k <= 0.0) for k in [kr, kta, ktb, *kphis]):
        raise ValueError("Boresch 数值参考需要正有限力常数")

    # Gauss-Legendre quadrature is smooth for all allowed force constants.  A
    # finite radial upper bound leaves <1e-20 of the harmonic tail, including
    # the softest accepted kr=50 kJ mol^-1 nm^-2.
    x, w = np.polynomial.legendre.leggauss(128)
    radial_sigma = math.sqrt(RT / kr)
    r_hi = max(r0 + 12.0 * radial_sigma, 12.0 * radial_sigma)
    r_nodes = 0.5 * r_hi * (x + 1.0)
    r_weights = 0.5 * r_hi * w
    radial = np.sum(
        r_weights * r_nodes**2
        * np.exp(-0.5 * kr * (r_nodes - r0) ** 2 / RT)
    )

    def _theta_integral(k, theta0):
        theta = 0.5 * math.pi * (x + 1.0)
        weights = 0.5 * math.pi * w
        return float(
            np.sum(
                weights
                * np.sin(theta)
                * np.exp(-k * (1.0 - np.cos(theta - theta0)) / RT)
            )
        )

    angular = _theta_integral(kta, theta_a) * _theta_integral(ktb, theta_b)
    # The wrapped dihedral integral is independent of phi0 and has a closed
    # form; numpy.i0 is stable over the accepted k range.
    for k in kphis:
        xk = k / RT
        angular *= 2.0 * math.pi * math.exp(-xk) * float(np.i0(xk))
    z_rest = radial * angular
    if not math.isfinite(z_rest) or z_rest <= 0.0:
        raise RuntimeError("Boresch cosine 数值积分得到非法配分函数")
    return -RT * math.log((8.0 * math.pi**2 * 1.6605) / z_rest)



class LambdaDependentBoreschForce(openmm.CustomCompoundBondForce):
    def __init__(
        self,
        rec_idx,
        lig_idx,
        eq,
        fc,
        lam_name="lambda_rest",
        fixed_lam=None,
        sign=1.0,
        use_pbc=False,
        # ✅ 移除 unit_sys 参数，强制标准单位
    ):
        if len(rec_idx) != 3 or len(lig_idx) != 3:
            raise ValueError("需 exactly 3 受体 + 3 配体原子")

        # ✅ 强制标准单位，不再进行任何转换
        r0 = eq["r0"]  # nm
        thA = eq["thetaA0"]  # rad
        thB = eq["thetaB0"]  # rad
        phA = eq["phiA0"]  # rad
        phB = eq["phiB0"]  # rad
        phC = eq["phiC0"]  # rad

        kr = fc["kr"]  # kJ/mol/nm²
        ktA = fc["kthetaA"]  # kJ/mol/rad²
        ktB = fc["kthetaB"]
        kpA = fc["kphiA"]
        kpB = fc["kphiB"]
        kpC = fc["kphiC"]

        ls = f"{fixed_lam:.6f}" if fixed_lam is not None else lam_name

        # ✅ 修复2：标准谐波势 (distance-r0)^2，导数连续且数值稳定
        # 🚨 关键修复：atom-index 顺序与 thetaA0/thetaB0/phiA0/phiB0/phiC0 的
        # 计算约定必须严格一致。addBond(rec_idx+lig_idx) 的顺序是
        # [R0(离配体最近), R1, R2(离配体最远), L0(离受体最近), L1, L2]——
        # 这也是 calc_boresch_from_last_frame / _check_boresch_geometry_safe /
        # _validate_boresch_geometry_strict 全部使用的约定：
        #   r0      = distance(R0, L0)
        #   thetaA0 = angle(R1, R0, L0)         顶点=R0
        #   thetaB0 = angle(R0, L0, L1)         顶点=L0
        #   phiA0   = dihedral(R2, R1, R0, L0)
        #   phiB0   = dihedral(R1, R0, L0, L1)
        #   phiC0   = dihedral(R0, L0, L1, L2)
        # 旧表达式误用 angle(p2,p3,p4)/angle(p3,p4,p5) 和
        # dihedral(p1,p2,p3,p4) 等，把顶点/参考原子错当成了 R2(最远的受体
        # 锚点，选择时只保证"刚性"而不保证与 R1/L0 不共线)，导致实际被约束
        # 的角度和平衡值计算出的角度根本不是同一个几何量：一来平衡值形同虚设、
        # 限制力没有真正锁住原有构象；二来一旦 R2 恰好与 R1、L0 接近共线，
        # angle()/dihedral() 的解析梯度出现 1/sinθ 型奇点，能量看起来正常但
        # 力却能炸到 10^7~10^8 kJ/mol/nm 量级——这正是本次 REMD 预热崩溃的根源。
        expr = (
            f"({sign})*{ls}*("
            "0.5*kr*(distance(p1,p4)-r0)^2+"
            "ktA*(1-cos(angle(p2,p1,p4)-thetaA0))+"
            "ktB*(1-cos(angle(p1,p4,p5)-thetaB0))+"
            "kpA*(1-cos(dihedral(p3,p2,p1,p4)-phiA0))+"
            "kpB*(1-cos(dihedral(p2,p1,p4,p5)-phiB0))+"
            "kpC*(1-cos(dihedral(p1,p4,p5,p6)-phiC0))"
            ")"
        )
        super().__init__(6, expr)  # ✅ N=6

        if fixed_lam is None:
            self.addGlobalParameter(lam_name, 0.0)

        for n, v in [
            ("r0", r0),
            ("thetaA0", thA),
            ("thetaB0", thB),
            ("phiA0", phA),
            ("phiB0", phB),
            ("phiC0", phC),
            ("kr", kr),
            ("ktA", ktA),
            ("ktB", ktB),
            ("kpA", kpA),
            ("kpB", kpB),
            ("kpC", kpC),
        ]:
            self.addGlobalParameter(n, v)

        if hasattr(self, "setUsesPeriodicBoundaryConditions"):
            self.setUsesPeriodicBoundaryConditions(bool(use_pbc))

        self.addBond(list(rec_idx) + list(lig_idx))
