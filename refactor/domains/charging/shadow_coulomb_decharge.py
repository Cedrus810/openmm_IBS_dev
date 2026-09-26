# ============================================================================
# 1.5 Shadow-Coulomb 去电荷：PME-Bridge (容斥探针) + Reciprocal-free IBS 高速公路
# ============================================================================
# 物理背景：
#   ΔG_decharge_PME = ΔG_bridge(PME full -> Shadow full) + ΔG_shadow_ibs(Shadow full -> Shadow decharged)
# 记 A = 配体原子 (perturbed_indices)，E = 其余全部原子：
#   U_shadow_cross(A,E) = A-E 间的短程"影子"库仑势，核函数 erfc(alpha*r)/r，alpha 取
#       自该系统真实 PME 的 Ewald alpha —— 与真实 PME real-space 项逐 pair 严格相等。
#   E_PME_cross(A,E) = E_full - E_A_only - E_E_only （容斥原理，三个纯静电 PME 探针），
#       隔离出 PME 下 A-E 的总静电交叉贡献 (real + reciprocal + self + PME 修正)。
#   Bridge:     U_bridge(s) = U_common + s*U_shadow_cross + (1-s)*E_PME_cross
#   Shadow IBS: U_shadow_ibs(k) = U_common + lambda_shadow_coul_k * U_shadow_cross(A,E)
# 只支持电中性配体 (Q_ligand ≈ 0，已对本项目 Atenolol 拓扑实测验证)；带净电配体需要
# 额外的共炼金反离子逻辑 (思路同 configure_coalchemical_neutral_decharging，驱动变量
# 换成 lambda_shadow_coul)，本实现尚未接入，遇到带净电配体会显式报错而不是静默算错。
# ============================================================================

_SHADOW_ONE_4PI_EPS0 = 138.935456  # kJ/mol*nm/e^2，与 OpenMM NonbondedForce 内部常量一致


def get_pme_alpha_for_system(
    system: openmm.System,
    topology=None,
    box_vectors=None,
) -> float:
    """
    用 OpenMM 自己派生 Ewald alpha 的闭式公式 (1/nm)，直接从静态的
    cutoff/ewaldErrorTolerance 算出来，不建 Context：

        alpha = sqrt(-log(2*ewaldErrorTolerance)) / cutoff

    这与 OpenMM 内部（NonbondedForceImpl）在没有显式 setPMEParameters() 时自动
    派生 alpha 用的公式完全一致，已经用本项目真实的 system_native.xml 实测校验
    （getPMEParametersInContext 给出的 alpha 与该公式逐位吻合）。

    🔑 为什么不像别处那样用 getPMEParametersInContext(ctx)：这需要现建一个一次性
    Context 才能拿到真实值，但实测发现——只要进程里已经创建过至少一个 CPU 平台的
    Context 并释放（生产流程里几乎必然如此，比如更早的 preopt 探针/REMD 阶段），
    再建一个新的一次性 CPU Context 去查 PME 参数会稳定触发 OpenMM/SWIG 的
    `std::bad_cast`（很可能是 CPU 平台 kernel 工厂在同进程内跨 Context 生命周期
    复用时的底层问题）。闭式公式完全不需要 Context，从根上绕开这个坑。
    topology/box_vectors 参数保留只是为了跟其他 builder 签名一致，这里不需要用到。
    """
    nb_forces = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)]
    if not nb_forces:
        raise ValueError("系统中未找到 NonbondedForce，无法提取 PME alpha。")
    nb = nb_forces[0]
    if nb.getNonbondedMethod() not in (openmm.NonbondedForce.PME, openmm.NonbondedForce.LJPME):
        raise RuntimeError(
            "Shadow-Coulomb 桥接依赖真实 PME 的 Ewald alpha，但该系统 NonbondedForce 的 "
            f"NonbondedMethod={nb.getNonbondedMethod()} 不是 PME，拒绝继续。"
        )
    # OpenMM returns the explicitly committed PME parameters here when the
    # caller used setPMEParameters().  Prefer those values: deriving alpha
    # from the tolerance is only equivalent for OpenMM's automatic-parameter
    # path and can silently disagree with a manually frozen grid/alpha.
    try:
        explicit_alpha, nx, ny, nz = nb.getPMEParameters()
        alpha_explicit = (
            float(explicit_alpha.value_in_unit(1 / unit.nanometer))
            if hasattr(explicit_alpha, "value_in_unit")
            else float(explicit_alpha)
        )
        grid = (int(nx), int(ny), int(nz))
    except Exception:
        alpha_explicit = 0.0
        grid = (0, 0, 0)
    if alpha_explicit > 0.0 or any(v != 0 for v in grid):
        if (
            not math.isfinite(alpha_explicit)
            or alpha_explicit <= 0.0
            or any(v <= 0 for v in grid)
        ):
            raise RuntimeError(
                "NonbondedForce 的显式 PME alpha/grid 不完整或非法；"
                "拒绝用 cutoff/tolerance 猜测另一个 Hamiltonian。"
            )
        return alpha_explicit

    cutoff_nm = nb.getCutoffDistance().value_in_unit(unit.nanometer)
    tolerance = float(nb.getEwaldErrorTolerance())
    if cutoff_nm <= 0.0 or not (0.0 < tolerance < 0.5):
        raise RuntimeError(
            f"NonbondedForce 的 cutoff={cutoff_nm} nm / ewaldErrorTolerance={tolerance} "
            "不在合理范围内，无法派生 PME alpha。"
        )
    alpha = math.sqrt(-math.log(2.0 * tolerance)) / cutoff_nm
    if alpha <= 0.0:
        raise RuntimeError("派生出的 Ewald alpha <= 0，拒绝继续构建 Shadow 力。")
    return alpha


def _pme_static_parameter_signature(nb: openmm.NonbondedForce) -> Dict[str, Any]:
    """Return the committed PME alpha/grid for provenance and cache checks.

    OpenMM reports zeros when alpha/grid are automatic.  In that case we still
    record the fact explicitly; callers that need an actual alpha use
    :func:`get_pme_alpha_for_system`, which applies the same closed-form
    derivation as OpenMM.  Keeping this helper separate avoids silently
    dropping an explicitly frozen grid from the Shadow bridge diagnostics.
    """
    try:
        alpha, nx, ny, nz = nb.getPMEParameters()
        alpha_nm = _value_in_inverse_nanometer(alpha)
        grid = [int(nx), int(ny), int(nz)]
    except Exception:
        alpha_nm = 0.0
        grid = [0, 0, 0]
    explicit = bool(alpha_nm > 0.0 or any(v != 0 for v in grid))
    return {
        "alpha_per_nm": float(alpha_nm),
        "grid": grid,
        "explicit": explicit,
    }


def _assert_neutral_ligand_for_shadow_coul(nb: openmm.NonbondedForce, ligand_indices: List[int]) -> float:
    """
    Shadow-Coulomb 支路目前只支持电中性配体：带净电配体单独去电荷会让炼金路径中段
    整体净电荷偏离 0，需要跟旧 configure_coalchemical_neutral_decharging 一样的共
    炼金反离子逻辑，本实现尚未接入，遇到带净电配体显式报错而不是悄悄算错。
    返回实测的净电荷（供调用方记录 provenance）。
    """
    net_q = 0.0
    for idx in ligand_indices:
        q, _, _ = nb.getParticleParameters(idx)
        net_q += q.value_in_unit(unit.elementary_charge)
    # Do not round before testing: +0.4e is not a neutral ligand and must not
    # pass merely because round(+0.4) == 0. Use the central integer-charge
    # tolerance as the single source of truth for this experimental path.
    if not np.isfinite(net_q):
        raise RuntimeError(
            f"配体净电荷为 NaN/Inf ({net_q!r})，Shadow-Coulomb 只支持有限中性配体"
        )
    nearest_integer = int(round(net_q))
    if (
        abs(net_q - nearest_integer) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E
        or nearest_integer != 0
    ):
        raise RuntimeError(
            f"检测到非中性配体 (Net Charge: {net_q:+.6f} e)。Shadow-Coulomb 去电荷支路"
            "目前只支持电中性配体：单独关掉带净电配体会让炼金路径中段净电荷偏离 0。"
            "请先接入共炼金反离子（同 configure_coalchemical_neutral_decharging 的思路，"
            "驱动变量换成 lambda_shadow_coul），本实现暂不支持，拒绝静默继续。"
        )
    return net_q


def _collect_shadow_cross_exclusions(
    nb: openmm.NonbondedForce,
    ligand_indices: List[int],
    environment_indices: List[int],
    ref_excl: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[int, int]]:
    """
    跨组 (配体-环境) exclusion 对，供 Shadow 力使用，逻辑同 _collect_softcore_exclusions。

    配体与环境原子之间通常没有真实的键连 1-2/1-3/1-4 exception（配体不与蛋白/水共价
    相连），所以这里天然算出来的跨组对经常是空集——这在物理上完全正确（Shadow 力的
    addInteractionGroup(A,E) 本来就该覆盖所有 A-E pair，没有谁需要被排除）。但如果
    传了 ref_excl（nb 的完整排除表），会把它一并并入返回值：这纯粹是为了让这个力的
    exclusion **数量** 跟同一个 CustomCVForce 里另外几个 NonbondedForce 探针（它们
    照抄了 nb 的完整排除表）保持一致——OpenMM 要求塞进同一个 CustomCVForce 的所有
    CV 互相之间排除表数量必须相同，否则会在建 Context 时报 'All Forces must have
    identical exclusions'。多出来的这些 ref_excl 对本来就落在 interaction group 之外
    （比如环境蛋白/水分子自身的 1-2/1-3/1-4 对），从不会被这个力实际计算，纯粹是
    账本对齐，不影响物理结果。
    """
    ligand_set = set(int(i) for i in ligand_indices)
    env_set = set(int(i) for i in environment_indices)
    pairs = set()
    scaled_cross_pairs = []
    for i in range(nb.getNumExceptions()):
        p1, p2, cross_qq, _, _ = nb.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if (p1 in ligand_set and p2 in env_set) or (p2 in ligand_set and p1 in env_set):
            pairs.add((min(p1, p2), max(p1, p2)))
            # 🔑 [0831issue P2] 跨组 exception 里 **chargeProd != 0** 的那些，是真实的
            # 1-4 缩放静电（只有配体与环境**共价相连**时才会出现，例如共价抑制剂）。
            # Shadow / mixed-alchemical 三条路都一致地把它们丢掉：
            #   (a) 背景力把跨组 exception 的 chargeProd 归零；
            #   (b) shadow 力把同一批对加进 exclusion；
            #   (c) PME 探针抄的是已被改写的例外表。
            # 于是这一项在两处都不算，静默少一项静电。Atenolol 非共价 → 跨组
            # exception 的 chargeProd 恒为 0（纯排除对），当前影响为 0；换共价体系
            # 就会静默算错，所以这里 fail closed 而不是留个注释。
            qq = float(
                cross_qq.value_in_unit(unit.elementary_charge**2)
                if hasattr(cross_qq, "value_in_unit")
                else cross_qq
            )
            if abs(qq) > 1.0e-12:
                scaled_cross_pairs.append(
                    (min(p1, p2), max(p1, p2), qq)
                )
    if scaled_cross_pairs:
        preview = ", ".join(
            f"({a},{b}) chargeProd={q:.6g}" for a, b, q in scaled_cross_pairs[:5]
        )
        raise RuntimeError(
            "Shadow/Bridge 路径遇到带非零 chargeProd 的跨组（配体-环境）exception "
            f"{len(scaled_cross_pairs)} 对：{preview}"
            f"{' ...' if len(scaled_cross_pairs) > 5 else ''}。"
            "这类对只在配体与环境共价相连时出现（真实的 1-4 缩放静电）；"
            "当前实现会在背景力里把它们归零、又在 shadow 力里把它们排除，"
            "两边都不算 → 静默少一项静电。共价体系请先实现这一项再启用 Shadow 路径。"
        )
    if ref_excl:
        pairs |= {(min(int(p1), int(p2)), max(int(p1), int(p2))) for p1, p2 in ref_excl if int(p1) != int(p2)}
    return sorted(pairs)


def _zero_ligand_environment_charge_in_background(
    nb: openmm.NonbondedForce,
    ligand_indices: List[int],
) -> List[Any]:
    """
    只把主 NonbondedForce 里配体的电荷 (不含 VdW) 归零，并把跨组 (配体-环境) 的
    exception 电荷乘积也归零，避免 Shadow/Bridge 的显式电荷探针与主背景力双计
    ligand-environment 静电。配体内部电荷、跨组/配体内部 VdW 一律保持原样不动——
    VdW 在 Bridge/Shadow-IBS 这两个 leg 里从头到尾都是满强度、未被炼金，属于
    U_common，不需要像 build_ibs_dual_system 那样连 VdW 一起归零。
    返回每个粒子归零前的原始 (q, sigma, eps) 快照，供构建 Shadow 力时取真实电荷。
    """
    ligand_set = set(int(i) for i in ligand_indices)
    num_atoms = nb.getNumParticles()
    original_params = [nb.getParticleParameters(i) for i in range(num_atoms)]
    for idx in ligand_set:
        _, sig, eps = original_params[idx]
        nb.setParticleParameters(idx, 0.0 * unit.elementary_charge, sig, eps)
    for i in range(nb.getNumExceptions()):
        p1, p2, cp, sig, eps = nb.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if (p1 in ligand_set) != (p2 in ligand_set):
            nb.setExceptionParameters(i, p1, p2, 0.0 * unit.elementary_charge ** 2, sig, eps)
    return original_params


def _build_shadow_coul_cross_force(
    alpha_ewald: float,
    original_params: List[Any],
    ligand_indices: List[int],
    environment_indices: List[int],
    lambda_value: float,
    exclusions: List[Tuple[int, int]],
    cutoff_nm: float,
) -> openmm.CustomNonbondedForce:
    """
    A(配体)-E(环境) 短程"影子"库仑势：lambda * ONE_4PI_EPS0 * q1*q2 * erfc(alpha*r)/r，
    只在 addInteractionGroup(A, E) 之间计算（不含 A-A/E-E），避免引入 lambda^2 项，
    与旧 co-alchemical 去电荷的线性 offset 约定保持一致。lambda 以字面常量刻入表达式
    （不是 runtime 可调的 global parameter），供 IBS 多态窗口 CV（每态一个实例，
    lambda_shadow_coul_k 不同）或 Bridge 的"满强度探针"（lambda=1.0）场景复用同一
    个 builder。
    """
    cutoff_nm = float(cutoff_nm)
    if not math.isfinite(cutoff_nm) or cutoff_nm <= 0.0:
        raise ValueError(f"Shadow-Coulomb cutoff 必须为正有限数，收到 {cutoff_nm!r}")
    expr = (
        f"{float(lambda_value)} * {_SHADOW_ONE_4PI_EPS0} * charge1*charge2 * "
        f"erfc({float(alpha_ewald)}*r) / r"
    )
    force = openmm.CustomNonbondedForce(expr)
    force.addPerParticleParameter("charge")
    for q, _, _ in original_params:
        force.addParticle([q.value_in_unit(unit.elementary_charge)])
    force.addInteractionGroup(
        list(set(int(i) for i in ligand_indices)),
        list(set(int(i) for i in environment_indices)),
    )
    force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(cutoff_nm * unit.nanometer)
    force.setUseSwitchingFunction(False)
    force.setUseLongRangeCorrection(False)
    for p1, p2 in exclusions:
        force.addExclusion(int(p1), int(p2))
    return force


def _build_electrostatics_only_pme_probe(
    original_params: List[Any],
    zero_indices: Optional[List[int]],
    reference_nb: openmm.NonbondedForce,
) -> openmm.NonbondedForce:
    """
    构建一个"只算静电、VdW 全部关闭"的 PME 探针 NonbondedForce 副本：epsilon 全部
    置零 (LJ 贡献消失)，PME method/cutoff/switching/ewald tolerance 与 reference_nb
    完全一致（容斥相减要求三个探针的 PME 配置逐一相同，否则减不干净）。可选把
    zero_indices 里的粒子电荷置零，用来构造 A-only / E-only 探针；zero_indices=None
    时构造"全电荷都在"的 full 探针。
    """
    probe = openmm.NonbondedForce()
    zero_set = set(int(i) for i in zero_indices) if zero_indices else set()
    for i, (q, sig, _eps) in enumerate(original_params):
        q_val = 0.0 * unit.elementary_charge if i in zero_set else q
        probe.addParticle(q_val, sig, 0.0 * unit.kilojoule_per_mole)
    for i in range(reference_nb.getNumExceptions()):
        p1, p2, cp, sig, _eps = reference_nb.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        cp_val = 0.0 * unit.elementary_charge ** 2 if (p1 in zero_set or p2 in zero_set) else cp
        probe.addException(p1, p2, cp_val, sig, 0.0 * unit.kilojoule_per_mole)
    probe.setNonbondedMethod(reference_nb.getNonbondedMethod())
    probe.setCutoffDistance(reference_nb.getCutoffDistance())
    probe.setEwaldErrorTolerance(reference_nb.getEwaldErrorTolerance())
    # Explicit PME alpha/grid settings are part of the Hamiltonian. Merely
    # copying method/cutoff/tolerance can make OpenMM choose a different grid,
    # breaking the inclusion-exclusion identity.
    if reference_nb.getNonbondedMethod() in (
        openmm.NonbondedForce.PME,
        openmm.NonbondedForce.LJPME,
    ):
        try:
            alpha, nx, ny, nz = reference_nb.getPMEParameters()
            probe.setPMEParameters(alpha, nx, ny, nz)
        except Exception:
            # Older OpenMM versions may report automatic (zero) parameters.
            pass
    if reference_nb.getUseSwitchingFunction():
        probe.setUseSwitchingFunction(True)
        probe.setSwitchingDistance(reference_nb.getSwitchingDistance())
    probe.setUseDispersionCorrection(False)
    return probe


def build_shadow_coul_ibs_system(
    system: openmm.System,
    topology,
    perturbed_indices: List[int],
    lambdas_shadow_coul: List[float],
    restraint_params: Optional[Dict] = None,
    prefix: str = "abfe_shadow",
    box_vectors=None,
    temperature=None,
) -> Tuple[openmm.System, "IBSBiasForce"]:
    """
    构建 Shadow-Coulomb IBS 去电荷系统：Shadow full charge (λ=1) -> Shadow decharged
    (λ=0)，只处理 A(配体)-E(环境) 的短程 erfc(alpha*r)/r "影子"电荷交叉项，VdW 全程
    满强度不炼金 (属于 U_common)。复用 IBSBiasForce 的多态 log-sum-exp 偏置机制
    （与 build_ibs_dual_system 的 VDW CV 完全同一套框架，只是把 CV 换成静电）。
    """
    if not lambdas_shadow_coul:
        raise ValueError("Shadow-Coulomb 至少需要一个 lambda 状态")
    if len(lambdas_shadow_coul) < 2:
        raise ValueError("Shadow-Coulomb 至少需要两个 lambda 状态以定义去电荷腿")
    # [EXP-031 S1] 同 build_ibs_dual_system：按当前形态算 CV 预算上限。
    _omit_zero_rest_cvs = IBS_BIAS_OMIT_ZERO_REST_CVS
    _max_lambda_states = ibs_dual_max_lambda_states(_omit_zero_rest_cvs)
    if len(lambdas_shadow_coul) > _max_lambda_states:
        raise ValueError(
            f"Shadow-Coulomb lambda 状态数 {len(lambdas_shadow_coul)} 超过 CustomCVForce "
            f"{OPENMM_CUSTOM_CV_MAX_VARIABLES} 变量上限对应的 {_max_lambda_states} 态上限"
        )
    lambda_values = np.asarray(lambdas_shadow_coul, dtype=float)
    if not np.all(np.isfinite(lambda_values)):
        raise ValueError("Shadow-Coulomb lambda 必须全部为有限数")
    if np.any((lambda_values < 0.0) | (lambda_values > 1.0)):
        raise ValueError("Shadow-Coulomb lambda 必须位于 [0, 1]")
    if not np.isclose(lambda_values[0], 1.0) or not np.isclose(lambda_values[-1], 0.0):
        raise ValueError(
            "Shadow-Coulomb lambda 必须以 1.0（满电荷）开始、以 0.0（去电荷）结束"
        )
    # The production Shadow-Coulomb leg is defined as full charge (1) ->
    # decharged (0).  Keep that orientation explicit so endpoint labels and
    # bridge bookkeeping cannot be silently reversed.
    lambda_diffs = np.diff(lambda_values)
    if np.any(lambda_diffs > 0.0) and np.any(lambda_diffs < 0.0):
        raise ValueError("Shadow-Coulomb lambda 必须单调（允许 1 -> 0 或 0 -> 1）")
    system = ensure_owned_system(system)
    new_sys = ensure_owned_system(XmlSerializer.clone(system))
    resolved_box = _resolve_periodic_box_vectors(box_vectors, topology=topology, system=new_sys)
    if _system_requires_periodic_box(new_sys):
        if resolved_box is None:
            raise ValueError("系统包含周期性非键力，但未提供有效的周期性盒子。")
        new_sys.setDefaultPeriodicBoxVectors(*resolved_box)

    num_atoms = new_sys.getNumParticles()
    perturbed_set = set(int(i) for i in perturbed_indices)
    env_indices = [i for i in range(num_atoms) if i not in perturbed_set]

    nb_forces = [f for f in new_sys.getForces() if isinstance(f, openmm.NonbondedForce)]
    if not nb_forces:
        raise ValueError("未找到 NonbondedForce")
    nb = nb_forces[0]

    _assert_neutral_ligand_for_shadow_coul(nb, perturbed_indices)
    alpha_ewald = get_pme_alpha_for_system(system, topology=topology, box_vectors=box_vectors)

    ref_excl = [
        (int(p1), int(p2))
        for p1, p2, _, _, _ in (nb.getExceptionParameters(i) for i in range(nb.getNumExceptions()))
    ]
    shadow_excl = _collect_shadow_cross_exclusions(nb, perturbed_indices, env_indices, ref_excl=ref_excl)

    # ---------- 1. 主 NonbondedForce：只关掉配体的电荷 (VdW 保持满强度) ----------
    original_params = _zero_ligand_environment_charge_in_background(nb, perturbed_indices)

    # ---------- 2. 配体内部力 (Group 2)：**只补普通库仑** ----------
    #
    # 🔑 [2026-09-10] 这里原来照抄 build_ibs_dual_system（连 σ/ε 带 1-4 全额重建），
    # 但两个 builder 的背景处理**根本不同**，照抄就是双计：
    #
    #   dual  ：把配体 σ/ε 压成 (0.1nm, 0)、清掉所有沾配体的 exception
    #           ⇒ 主 NB 不出任何 L–L，Group 2 全额重建，×1 ✓
    #   shadow：`_zero_ligand_environment_charge_in_background()` 只清**粒子电荷**
    #           和**跨组** exception 的 chargeProd；配体 σ/ε 与 L–L exception
    #           **原样保留**（有意的：VdW 在 shadow 腿全程满强度、属于 U_common）
    #
    # 于是照抄之后：
    #   L–L 普通库仑  主 NB 0（电荷已清） + Group 2 补回        = ×1 ✓
    #   L–L 普通 LJ   主 NB 仍在          + Group 2 又一份      = ×2 ✗
    #   L–L 1-4       主 NB 仍在          + ll_14_force 又一份  = ×2 ✗（LJ 与库仑都翻倍）
    #
    # "与 λ 无关所以在 ΔF 里相消"是错的：它在同一构型的能量差里确实消失，
    # 但它改变构型的 Boltzmann 权重 ⟹ 改变采样系综；翻倍的 r⁻¹² 芯还是不稳定源。
    #
    # 正确划分（与 shadow 自己的设计一致）：
    #   * 背景 NB 负责 L–L 普通 LJ + 全部 L–L exception（1-4 的 LJ 与库仑）；
    #   * Group 2 只补 L–L 普通库仑 —— 传进去的参数把配体 ε 压成 0（σ 留 0.1nm
    #     防除零），并且**不再挂 ll_14_force**。
    #   * `create_ligand_internal_force` 本来就排除 1-2/1-3/1-4，正好与背景的
    #     exception 互补，不重不漏。
    #
    # 由 `tests/test_shadow_coul_no_double_counted_ligand_internals.py` 钉住：
    # 全体零电荷时 group 0+2 必须逐位等于原体系总能量与力（修复前实测偏大一倍）。
    internal_ref_excl = [(p1, p2) for p1, p2 in ref_excl if p1 in perturbed_set and p2 in perturbed_set]
    coulomb_only_params = list(original_params)
    for idx in perturbed_indices:
        q, _sig, _eps = coulomb_only_params[int(idx)]
        coulomb_only_params[int(idx)] = (
            q,
            0.1 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )
    ll_f, _ll_14_f_unused = create_ligand_internal_force(
        nb, perturbed_indices, coulomb_only_params, internal_ref_excl, num_atoms, system=new_sys
    )
    ll_f.setForceGroup(2)
    new_sys.addForce(ll_f)
    # ⚠️ 刻意**不加** `_ll_14_f_unused`：1-4 已经完整留在背景 NB 的 exception 里。
    # `create_ligand_internal_force` 仍然返回它（该函数为 dual builder 而写，
    # 那边的背景已经被清空、确实需要它），这里只是不消费。

    # ---------- 3. 物理 Boresch 限制力 (Group 3)，与 build_ibs_dual_system 一致 ----------
    if _has_valid_boresch_restraint(restraint_params):
        rest_f_phys = LambdaDependentBoreschForce(
            rec_idx=restraint_params["receptor_indices"], lig_idx=restraint_params["ligand_indices"],
            eq=restraint_params["equilibrium_values"], fc=restraint_params["force_constants"],
            lam_name="lambda_boresch_scale", fixed_lam=None, sign=1.0, use_pbc=True
        )
        rest_f_phys.setForceGroup(3)
        new_sys.addForce(rest_f_phys)

    # ---------- 4. IBS 偏置力与 Shadow 电荷软核 CV (Group 1) ----------
    if temperature is None:
        raise ValueError("Shadow-Coulomb IBS 必须显式提供采样温度")
    temp_quantity = (
        temperature if hasattr(temperature, "value_in_unit") else float(temperature) * unit.kelvin
    )
    ibs_wrapper = IBSBiasForce(len(lambdas_shadow_coul), temp_quantity, prefix=prefix)
    cutoff_nm = nb.getCutoffDistance().value_in_unit(unit.nanometer)
    cv_template = _build_shadow_coul_cross_force(
        alpha_ewald, original_params, perturbed_indices, env_indices,
        lambda_value=0.0, exclusions=shadow_excl, cutoff_nm=cutoff_nm,
    )
    template_excl = {
        tuple(sorted(map(int, cv_template.getExclusionParticles(i))))
        for i in range(cv_template.getNumExclusions())
    }
    for k, lam in enumerate(lambdas_shadow_coul):
        int_f_cv = _build_shadow_coul_cross_force(
            alpha_ewald, original_params, perturbed_indices, env_indices,
            lambda_value=float(lam), exclusions=shadow_excl, cutoff_nm=cutoff_nm,
        )
        current_excl = {
            tuple(sorted(map(int, int_f_cv.getExclusionParticles(i))))
            for i in range(int_f_cv.getNumExclusions())
        }
        if current_excl != template_excl:
            raise RuntimeError(f"Shadow CV {k} 排除表不一致，将破坏 IBS 邻居表复用条件。")
        ibs_wrapper.addCollectiveVariable(f"cv_{k}_int", int_f_cv)
        # [EXP-031 S1] 同 build_ibs_dual_system：omit 形态下不建恒零占位槽。
        if not ibs_wrapper.omit_zero_rest_cvs:
            rest_f_cv = openmm.CustomExternalForce("0")
            ibs_wrapper.addCollectiveVariable(f"cv_{k}_rest", rest_f_cv)

    # 🔑 [2026-09-09] 与 `build_ibs_dual_system` 对齐：在 Force 进 System 之前
    # fail closed。这个 builder 原来**没有**这道校验，而它和 dual builder 一样是
    # 靠调用方在上面那个循环里逐个 `addCollectiveVariable` 来满足表达式里引用的
    # `cv_k_int`/`cv_k_rest` —— 漏一个的后果不是报错，是 `CustomCVForce` 引用未注册
    # 符号后**静默给出一个有限但错误的能量**（EXP-025 G4 Layer-1 实测差 2.27 kJ/mol，
    # 那次就是靠"每个输入独立验证"才发现的）。纯防御性检查，不改变有效哈密顿量。
    ibs_wrapper.validate_wiring()
    new_sys.addForce(ibs_wrapper.get_force())

    sync_all_exclusions(new_sys)
    print(
        f"[Shadow-Coulomb IBS System] 构建完成，alpha_ewald={alpha_ewald:.6f}/nm，"
        f"共 {new_sys.getNumForces()} 个力对象，{len(lambdas_shadow_coul)} 个 λ_shadow_coul 状态。"
    )
    return new_sys, ibs_wrapper


def build_shadow_bridge_system(
    system: openmm.System,
    topology,
    perturbed_indices: List[int],
    restraint_params: Optional[Dict] = None,
    box_vectors=None,
) -> Tuple[openmm.System, str, Dict[str, Any]]:
    """
    构建 Shadow-PME Bridge 系统：单个 System 里同时放 4 个探针 CV
    (cv_full / cv_A_only / cv_E_only / cv_shadow)，用一个 CustomCVForce 按

        U_bridge(s) = U_common + s*cv_shadow + (1-s)*(cv_full - cv_A_only - cv_E_only)

    组合，s 是名为 "lambda_bridge_s" 的 global parameter，运行期可以直接
    context.setParameter("lambda_bridge_s", s) 在 0 (PME full charge) 到 1
    (Shadow full charge) 之间切换 —— 不需要为每个 bridge 窗口重新建系统。
    Bridge 只有 1~3 个窗口，直接用传统 REMD/BAR 采样，不进 IBS。

    返回 (new_sys, "lambda_bridge_s", diagnostics)。
    """
    system = ensure_owned_system(system)
    new_sys = ensure_owned_system(XmlSerializer.clone(system))
    resolved_box = _resolve_periodic_box_vectors(box_vectors, topology=topology, system=new_sys)
    if _system_requires_periodic_box(new_sys):
        if resolved_box is None:
            raise ValueError("系统包含周期性非键力，但未提供有效的周期性盒子。")
        new_sys.setDefaultPeriodicBoxVectors(*resolved_box)

    num_atoms = new_sys.getNumParticles()
    perturbed_set = set(int(i) for i in perturbed_indices)
    env_indices = [i for i in range(num_atoms) if i not in perturbed_set]

    nb_forces = [f for f in new_sys.getForces() if isinstance(f, openmm.NonbondedForce)]
    if not nb_forces:
        raise ValueError("未找到 NonbondedForce")
    nb = nb_forces[0]

    _assert_neutral_ligand_for_shadow_coul(nb, perturbed_indices)
    alpha_ewald = get_pme_alpha_for_system(system, topology=topology, box_vectors=box_vectors)
    pme_static_signature = _pme_static_parameter_signature(nb)

    ref_excl = [
        (int(p1), int(p2))
        for p1, p2, _, _, _ in (nb.getExceptionParameters(i) for i in range(nb.getNumExceptions()))
    ]
    shadow_excl = _collect_shadow_cross_exclusions(nb, perturbed_indices, env_indices, ref_excl=ref_excl)
    original_params_before_zeroing = [nb.getParticleParameters(i) for i in range(num_atoms)]

    # ---------- 1. 主 NonbondedForce：只关掉配体的电荷 (VdW 保持满强度) ----------
    original_params = _zero_ligand_environment_charge_in_background(nb, perturbed_indices)

    # ---------- 2. 配体内部力 (Group 2)：**只补普通库仑**，与 Shadow-IBS 一致 ----------
    #
    # 🔑 [2026-09-10] 同 `build_shadow_coul_ibs_system` 的那处双计（完整推导见那里）。
    # 这条 bridge 腿的背景处理与 Shadow-IBS 逐字相同 —— 都是
    # `_zero_ligand_environment_charge_in_background()`，只清粒子电荷和跨组
    # exception，配体 σ/ε 与 L–L exception 原样保留 —— 所以照抄 dual builder 的
    # 全额重建同样会把 L–L 普通 LJ 和 1-4（LJ 与库仑）各算两遍。
    internal_ref_excl = [(p1, p2) for p1, p2 in ref_excl if p1 in perturbed_set and p2 in perturbed_set]
    coulomb_only_params = list(original_params)
    for idx in perturbed_indices:
        q, _sig, _eps = coulomb_only_params[int(idx)]
        coulomb_only_params[int(idx)] = (
            q,
            0.1 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )
    ll_f, _ll_14_f_unused = create_ligand_internal_force(
        nb, perturbed_indices, coulomb_only_params, internal_ref_excl, num_atoms, system=new_sys
    )
    ll_f.setForceGroup(2)
    new_sys.addForce(ll_f)
    # ⚠️ 刻意不加 `_ll_14_f_unused`：1-4 完整留在背景 NB 的 exception 里。

    # ---------- 3. 物理 Boresch 限制力 (Group 3) ----------
    if _has_valid_boresch_restraint(restraint_params):
        rest_f_phys = LambdaDependentBoreschForce(
            rec_idx=restraint_params["receptor_indices"], lig_idx=restraint_params["ligand_indices"],
            eq=restraint_params["equilibrium_values"], fc=restraint_params["force_constants"],
            lam_name="lambda_boresch_scale", fixed_lam=None, sign=1.0, use_pbc=True
        )
        rest_f_phys.setForceGroup(3)
        new_sys.addForce(rest_f_phys)

    # ---------- 4. 三个纯静电 PME 探针 (容斥) + Shadow 探针，一起塞进一个 CustomCVForce ----------
    probe_full = _build_electrostatics_only_pme_probe(original_params_before_zeroing, None, nb)
    probe_a = _build_electrostatics_only_pme_probe(original_params_before_zeroing, env_indices, nb)
    probe_e = _build_electrostatics_only_pme_probe(original_params_before_zeroing, list(perturbed_set), nb)
    shadow_full = _build_shadow_coul_cross_force(
        alpha_ewald, original_params_before_zeroing, perturbed_indices, env_indices,
        lambda_value=1.0, exclusions=shadow_excl,
        cutoff_nm=nb.getCutoffDistance().value_in_unit(unit.nanometer),
    )

    bridge_expr = (
        "lambda_bridge_s*cv_shadow + (1.0-lambda_bridge_s)*(cv_full - cv_A_only - cv_E_only)"
    )
    bridge_force = openmm.CustomCVForce(bridge_expr)
    bridge_force.addGlobalParameter("lambda_bridge_s", 0.0)
    bridge_force.addCollectiveVariable("cv_full", probe_full)
    bridge_force.addCollectiveVariable("cv_A_only", probe_a)
    bridge_force.addCollectiveVariable("cv_E_only", probe_e)
    bridge_force.addCollectiveVariable("cv_shadow", shadow_full)
    bridge_force.setForceGroup(1)
    new_sys.addForce(bridge_force)

    sync_all_exclusions(new_sys)
    diagnostics = {
        "alpha_ewald_per_nm": alpha_ewald,
        "reference_pme_parameters": pme_static_signature,
        "reference_pme_cutoff_nm": float(
            nb.getCutoffDistance().value_in_unit(unit.nanometer)
        ),
        "shadow_coulomb_cutoff_nm": float(
            nb.getCutoffDistance().value_in_unit(unit.nanometer)
        ),
        "n_ligand_atoms": len(perturbed_set),
        "n_environment_atoms": len(env_indices),
    }
    print(
        f"[Shadow-PME Bridge System] 构建完成，alpha_ewald={alpha_ewald:.6f}/nm，"
        f"共 {new_sys.getNumForces()} 个力对象。lambda_bridge_s=0 -> PME full charge，"
        "lambda_bridge_s=1 -> Shadow full charge。"
    )
    return new_sys, "lambda_bridge_s", diagnostics


