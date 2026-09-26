# ============================================================================
# 0. 通用工具 (从 abfe_core 导入)
# ============================================================================
# ensure_owned_system, sync_all_exclusions, create_ligand_internal_force 已从 abfe_core 导入
# ================= ibs_engine.py / abfe_core.py =================
import openmm
import numpy as np

def charging_charge_conservation_report(
    nb_force: openmm.NonbondedForce,
    lambda_name: str,
    *,
    ligand_indices: Optional[Sequence[int]] = None,
    co_ion_indices: Optional[Sequence[int]] = None,
    ligand_net_charge_e: Optional[float] = None,
    lambdas: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> Dict[str, Any]:
    """[§7.2] 逐 λ 电荷账目：总电荷是否恒定、配体/co-ion 是否走在该走的路上。

    直接读**已经配置好的** `NonbondedForce`（粒子基电荷 + 该 λ 名下的 ParameterOffset），
    所以它核对的是"实际会跑的那个哈密顿量"，不是我们打算写进去的那个。

    总电荷守恒是一次代数证明而不是抽查：`Σq(λ) = Σq_base + λ·Σq_scale`，
    所以只要 `Σq_scale = 0`，**所有** λ（含中间态）的总电荷都相同。逐 λ 的数值仍然
    一并算出来落盘，因为它是最容易被人读懂的证据。
    """
    n = nb_force.getNumParticles()
    base = np.zeros(n, dtype=np.float64)
    for i in range(n):
        q, _, _ = nb_force.getParticleParameters(i)
        base[i] = q.value_in_unit(unit.elementary_charge)
    scale = np.zeros(n, dtype=np.float64)
    n_offsets = 0
    for offset_idx in range(nb_force.getNumParticleParameterOffsets()):
        param, particle_idx, charge_scale, _s, _e = nb_force.getParticleParameterOffset(
            offset_idx
        )
        if str(param) != str(lambda_name):
            continue
        scale[int(particle_idx)] += _value_in_elementary_charge(charge_scale)
        n_offsets += 1

    base_sum = float(base.sum())
    scale_sum = float(scale.sum())
    totals = {
        f"{float(lam):.6g}": float(charge_at_lambda(base_sum, scale_sum, float(lam)))
        for lam in lambdas
    }
    report: Dict[str, Any] = {
        "lambda_name": str(lambda_name),
        "n_particle_offsets": int(n_offsets),
        "base_sum_e": base_sum,
        "scale_sum_e": scale_sum,
        "total_charge_by_lambda_e": totals,
        "total_charge_is_lambda_independent": bool(
            abs(scale_sum) <= TOTAL_CHARGE_CONSERVATION_TOLERANCE_E
        ),
        "tolerance_e": float(TOTAL_CHARGE_CONSERVATION_TOLERANCE_E),
    }
    if ligand_indices is not None:
        lig = np.asarray(sorted({int(i) for i in ligand_indices}), dtype=int)
        report["ligand_charge_by_lambda_e"] = {
            f"{float(lam):.6g}": float(
                charge_at_lambda(base[lig].sum(), scale[lig].sum(), float(lam))
            )
            for lam in lambdas
        }
        if ligand_net_charge_e is not None:
            report["ligand_charge_matches_lambda_times_qL"] = all(
                abs(
                    charge_at_lambda(base[lig].sum(), scale[lig].sum(), float(lam))
                    - float(lam) * float(ligand_net_charge_e)
                )
                <= LIGAND_CHARGE_LAMBDA_TOLERANCE_E
                for lam in lambdas
            )
    if co_ion_indices is not None and len(list(co_ion_indices)):
        ions = np.asarray(sorted({int(i) for i in co_ion_indices}), dtype=int)
        report["co_ion_charge_by_lambda_e"] = {
            f"{float(lam):.6g}": float(
                charge_at_lambda(base[ions].sum(), scale[ions].sum(), float(lam))
            )
            for lam in lambdas
        }
    if not report["total_charge_is_lambda_independent"]:
        raise RuntimeError(
            "charging 腿的总电荷随 λ 变化："
            f"Σq_scale = {scale_sum:+.6e} e（容差 {TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:g} e）。\n"
            f"    逐 λ 总电荷：{totals}\n"
            "    PME 会用一个逐 λ 变化的中和背景电荷把这件事掩盖掉，ΔG 静默出错（§7.2）。"
        )
    return report


def _inject_co_alchemical_ion_restraints(
    system: openmm.System,
    co_alchemical_ion_spec: Dict[str, Any],
    log_prefix: str = "  ",
) -> List[Dict[str, Any]]:
    """[MEM-00d] 按冻结 spec 注入每个 co-ion 的 flat-bottom 锚点相对位置限制。

    两条路线共用这一处（§4.4 要求复合物腿与溶剂腿"同一函数形式和力常数"，
    co-annihilation 与 charge-transfer 同理没有理由各写一份）。
    所有参数都来自 spec，**不看当前坐标** —— 参考量属于身份的一部分。
    """
    # The restraint stores a Cartesian anchor→ion displacement as a per-bond
    # parameter.  OpenMM barostats scale the box/coordinates but cannot scale
    # that parameter, so using this Hamiltonian in NPT would move the well
    # centre relative to the physical fractional position.  Refuse every
    # barostat variant instead of silently producing a box-size-dependent
    # carrier reservoir.  A future fractional-coordinate implementation must
    # use a new restraint box_model/protocol version.
    barostats = [
        force.__class__.__name__
        for force in system.getForces()
        if "Barostat" in force.__class__.__name__
    ]
    if barostats:
        raise RuntimeError(
            "co-ion Cartesian anchor-relative restraint 仅支持固定盒 NVT；"
            f"检测到 barostat={barostats}。请在注入前移除 barostat，或实现并验证分数坐标 tether。"
        )
    injected: List[Dict[str, Any]] = []
    for ion in co_alchemical_ion_spec["ions"]:
        restraint_spec = ion["restraint"]
        if restraint_spec.get("form") != CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM:
            raise RuntimeError(
                f"co-ion restraint 形式 {restraint_spec.get('form')!r} 不是当前实现的 "
                f"{CO_ALCHEMICAL_ION_RESTRAINT_FORM_FLAT_BOTTOM!r}（MEM-00d）。"
                "旧 spec 不可复用，请重新选择 co-ion 并落盘。"
            )
        if restraint_spec.get("box_model") != CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL:
            raise RuntimeError(
                "co-ion restraint spec 未声明固定盒 box_model；拒绝将旧笛卡尔 tether 注入当前 Hamiltonian。"
            )
        force = _create_co_alchemical_ion_restraint(
            ion_index=int(ion["atom_index"]),
            anchor_atom_index=int(restraint_spec["anchor_atom_index"]),
            reference_displacement_nm=restraint_spec["reference_displacement_nm"],
            force_constant_kj_per_mol_nm2=float(restraint_spec["k_kj_per_mol_nm2"]),
            flat_bottom_radius_nm=float(restraint_spec["flat_bottom_radius_nm"]),
        )
        # §6.4 末条：restraint 逐 λ 相同，且必须待在自己的 force group 里，
        # 不许混进任何 λ 相关的能量分解或 u_kn 差值。
        force.setForceGroup(int(restraint_spec.get(
            "force_group", CO_ALCHEMICAL_ION_RESTRAINT_FORCE_GROUP
        )))
        system.addForce(force)
        d0 = [float(v) for v in restraint_spec["reference_displacement_nm"]]
        injected.append(
            {
                "ion_index": int(ion["atom_index"]),
                "anchor_atom_index": int(restraint_spec["anchor_atom_index"]),
                "reference_displacement_nm": d0,
                "k_kj_per_mol_nm2": float(restraint_spec["k_kj_per_mol_nm2"]),
                "flat_bottom_radius_nm": float(restraint_spec["flat_bottom_radius_nm"]),
                "force_group": int(force.getForceGroup()),
            }
        )
        print(
            f"{log_prefix} co-ion flat-bottom 位置限制已注入: 粒子 {ion['atom_index']} "
            f"↔ 锚点 {restraint_spec['anchor_atom_index']}, "
            f"k={float(restraint_spec['k_kj_per_mol_nm2']):.1f} kJ/mol/nm², "
            f"r₀={float(restraint_spec['flat_bottom_radius_nm']):.2f} nm, "
            f"d0=({d0[0]:.3f}, {d0[1]:.3f}, {d0[2]:.3f}) nm"
        )
    return injected


def configure_charge_transfer_decharging(
    system: openmm.System,
    ligand_indices: List[int],
    topology,
    lambda_name: str = "lam_coul",
    co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, List[int]]:
    """[B3] charge-transfer 的 charging Hamiltonian：ligand q→0 与 co-ion 0→q。

    §2.1 的 λ 定义（`lam_coul` 1 → 0，即 `lambda_q`）：

        q_lig_i(λ) = λ · q_i                    base 0,       scale  q_i
        q_coion(λ) = (1 − λ) · share            base share,   scale −share

    于是 `Σq_lig(λ) + Σq_coion(λ) = q_L` 对**所有** λ 恒成立，全盒总电荷与 λ=1
    的物理体系逐 λ 严格相同（§2.1 末句）。这一点由
    `charging_charge_conservation_report` 在配置完成后**读回真实 Force** 证明，
    而不是靠这段注释。

    与 co-annihilation（`configure_coalchemical_neutral_decharging`）的区别只在
    co-ion 那两行 base/scale，但物理含义相反：那条是"同时湮灭一对异号电荷"，
    这条是"把电荷从结合位点搬到体相水"。两者不可混用，spec 里记着
    `charge_treatment` 并在此处强制核对。

    电荷映射本身来自 `abfe_core.co_alchemical_charge_offset_plan()`——这一层只负责
    把它写进 OpenMM，不重新推导一遍（写歪了会自己对上自己）。
    """
    nb_force = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None
    )
    if nb_force is None:
        raise RuntimeError("系统中未找到 NonbondedForce")

    ligand_set = {int(i) for i in ligand_indices}
    raw_lig_net_charge = 0.0
    ligand_params: Dict[int, Any] = {}
    for idx in sorted(ligand_set):
        q, sig, eps = nb_force.getParticleParameters(idx)
        ligand_params[idx] = (q, sig, eps)
        raw_lig_net_charge += q.value_in_unit(unit.elementary_charge)
    if not np.isfinite(raw_lig_net_charge):
        raise RuntimeError("配体净电荷为 NaN/Inf，拒绝配置 charge-transfer Hamiltonian")
    lig_net_charge = int(round(raw_lig_net_charge))
    if abs(raw_lig_net_charge - lig_net_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
        raise RuntimeError(
            f"配体净电荷 {raw_lig_net_charge:+.6f} e 不接近整数（容差 1e-3 e）"
        )
    if lig_net_charge == 0:
        raise RuntimeError(
            "charge-transfer 被调用，但配体净电荷为 0。中性配体应当走 "
            "`charge_treatment=neutral` 的普通 ligand-only offset 路径 —— "
            "给中性配体造 co-ion 只会凭空加一个不必要的炼金粒子。"
        )
    if co_alchemical_ion_spec is None:
        raise RuntimeError(
            f"检测到带电配体 (Net Charge: {lig_net_charge:+d}) 且路线为 charge-transfer，"
            "但没有传入 `co_alchemical_ion_spec`。\n"
            "    [MEM-00c] co-ion 身份必须由 `select_co_alchemical_ion_once()` 选一次并"
            "落盘，动力学 / REMD 副本 / compute_u_kn / resume 全部只读消费。\n"
            "    **不要**在这里恢复自动选择。"
        )

    pinned = verify_co_alchemical_ion_identity(
        co_alchemical_ion_spec,
        system=system,
        topology=topology,
        charge_treatment=CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=lig_net_charge,
        context="configure_charge_transfer_decharging",
    )
    ion_indices = [int(i) for i in pinned]
    if ligand_set & set(ion_indices):
        raise RuntimeError(
            f"co-ion 与配体原子重叠：{sorted(ligand_set & set(ion_indices))}。"
            "co-ion 必须是体相水里另一个粒子。"
        )

    plan = co_alchemical_charge_offset_plan(
        charge_treatment=CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=lig_net_charge,
        ligand_charges_e={
            idx: ligand_params[idx][0].value_in_unit(unit.elementary_charge)
            for idx in sorted(ligand_set)
        },
        co_ion_physical_charges_e={
            idx: nb_force.getParticleParameters(idx)[0].value_in_unit(
                unit.elementary_charge
            )
            for idx in ion_indices
        },
    )

    existing_params = [
        nb_force.getGlobalParameterName(i)
        for i in range(nb_force.getNumGlobalParameters())
    ]
    if lambda_name not in existing_params:
        nb_force.addGlobalParameter(lambda_name, 1.0)

    original_charges: Dict[int, Any] = {}
    alchemical_set = set(ligand_set) | set(ion_indices)
    for idx, (base_e, scale_e) in sorted(plan["offsets"].items()):
        q, sig, eps = nb_force.getParticleParameters(idx)
        original_charges[idx] = q
        nb_force.setParticleParameters(idx, base_e * unit.elementary_charge, sig, eps)
        nb_force.addParticleParameterOffset(
            lambda_name,
            idx,
            scale_e * unit.elementary_charge,
            0.0 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )

    # co-ion 的 exception：单原子离子在 GROMACS/OpenMM 拓扑里**不该有**任何
    # 1-2/1-3/1-4 exception。真有的话说明它不是单原子离子，而"base 非 0 的粒子
    # 走 exception offset"需要 (base·q_other, −base·q_other) 那一对系数，
    # 与下面配体那段的 (0, chargeProd) 不是同一个式子 —— 所以这里 fail closed
    # 而不是套用配体的写法。
    for exc_idx in range(nb_force.getNumExceptions()):
        p1, p2, _cp, _s, _e = nb_force.getExceptionParameters(exc_idx)
        if int(p1) in ion_indices or int(p2) in ion_indices:
            raise RuntimeError(
                f"co-ion（粒子 {int(p1)} / {int(p2)} 之一）带有 NonbondedForce exception。"
                "charge-transfer 的第一版只支持**单原子** co-ion（§2.2：只改 charge，"
                "mass/sigma/epsilon 逐 λ 不变）；多原子 co-ion 的 exception offset 系数"
                "与配体那套不同，必须单独实现并验证，不许套用。"
            )

    # 配体内部静电的 λ 口径（P0-01 之后的 v3 annihilation）：既有 L–L exception
    # （chargeProd 是独立参数、不读粒子电荷）逐 λ 恒定；普通 L–L 对随粒子 offset
    # 线性湮灭。不再把普通对补成显式 exception——那会改写真实 PME 的 λ=1 端点。
    # [0831issue P2] 记下 (exc_idx, chargeProd) 而不只是 pair：函数末尾的
    # _assert_frozen_ligand_ligand_exceptions 需要它们来复核"确实冻结了"。
    frozen_ll_pairs: Dict[Tuple[int, int], Tuple[int, float]] = {}
    for exc_idx in range(nb_force.getNumExceptions()):
        p1, p2, charge_prod, sig, eps = nb_force.getExceptionParameters(exc_idx)
        p1, p2 = int(p1), int(p2)
        if p1 in ligand_set and p2 in ligand_set:
            frozen_ll_pairs[(min(p1, p2), max(p1, p2))] = (
                int(exc_idx),
                float(
                    charge_prod.value_in_unit(unit.elementary_charge**2)
                    if hasattr(charge_prod, "value_in_unit")
                    else charge_prod
                ),
            )
            continue
        if (p1 in alchemical_set) ^ (p2 in alchemical_set):
            nb_force.setExceptionParameters(
                exc_idx, p1, p2, 0.0 * unit.elementary_charge**2, sig, eps
            )
            nb_force.addExceptionParameterOffset(
                lambda_name,
                exc_idx,
                charge_prod,
                0.0 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
            )

    # [P0-01, 2026-08-30] 同 configure_pme_ligand_charge_offsets：不再把普通
    # L–L 对补成 exception（会改写真实 PME 的 λ=1 物理端点）。既有 L–L
    # exception 冻结；普通 L–L 对在主 NB 力里按 λ² 缩放，由下面的
    # (1-λ²) 补偿项补回（v5，couple-intramol=no），口径见上一处注释。

    # [0831issue P2] P0-01 前提的断言（此前只收集不校验），见该函数 docstring。
    # [couple-intramol=no] 配体内部 ≥1-5 库仑冻结为 λ 无关（见函数 docstring）。
    _freeze_ligand_internal_coulomb(
        system, nb_force, ligand_set, lambda_name, original_charges, "charge-transfer decharging"
    )
    _assert_frozen_ligand_ligand_exceptions(
        nb_force, lambda_name, frozen_ll_pairs, "charge-transfer decharging"
    )

    _inject_co_alchemical_ion_restraints(system, co_alchemical_ion_spec)

    report = charging_charge_conservation_report(
        nb_force,
        lambda_name,
        ligand_indices=sorted(ligand_set),
        co_ion_indices=ion_indices,
        ligand_net_charge_e=lig_net_charge,
    )
    print(
        f"  [B3] charge-transfer charging 已配置: 配体 {lig_net_charge:+d} e → 0，"
        f"co-ion {ion_indices} 0 → {lig_net_charge:+d} e（每粒子 ≤ 1 单位电荷）；"
        f"逐 λ 总电荷恒为 {report['base_sum_e']:+.6f} e "
        f"(Σscale={report['scale_sum_e']:+.2e} e)"
    )
    return original_charges, ion_indices


def _assert_frozen_ligand_ligand_exceptions(
    nb_force: openmm.NonbondedForce,
    lambda_name: str,
    frozen_ll_pairs: Dict[Tuple[int, int], Tuple[int, float]],
    context: str,
) -> None:
    """[0831issue P2] 守住 P0-01 的关键前提：既有 L–L exception 逐 λ 恒定。

    三个 decharging builder 都会收集 `frozen_ll_pairs`，但**收集完从来没被读过**
    —— 也就是说 "既有 L–L exception 已冻结" 这个 P0-01 赖以成立的前提，此前完全
    没有断言守护。它一旦不成立，配体内部静电就会随 λ 变化，两条腿的 λ=1 物理端点
    被改写，而这不会有任何报错。

    这里逐对复核两件事：
      1. 这些 exception 的 `chargeProd` 与改写前逐位一致（没有被后续代码动过）；
      2. 没有任何 `lambda_name` 的 exception parameter offset 指向它们
         （有 offset 就意味着它们其实随 λ 变，"冻结"是假的）。
    """
    for exc_idx, expected_qq in frozen_ll_pairs.values():
        p1, p2, charge_prod, _sig, _eps = nb_force.getExceptionParameters(int(exc_idx))
        actual_qq = float(
            charge_prod.value_in_unit(unit.elementary_charge**2)
            if hasattr(charge_prod, "value_in_unit")
            else charge_prod
        )
        if abs(actual_qq - float(expected_qq)) > 1.0e-12:
            raise RuntimeError(
                f"{context}: 配体内部 exception ({int(p1)}, {int(p2)}) 的 chargeProd "
                f"被改写（{expected_qq!r} → {actual_qq!r}）。P0-01 要求既有 L–L "
                "exception 逐 λ 恒定，否则配体内部静电随 λ 变化、λ=1 物理端点被改写。"
            )

    frozen_indices = {int(exc_idx) for exc_idx, _q in frozen_ll_pairs.values()}
    for off_idx in range(nb_force.getNumExceptionParameterOffsets()):
        param, exc_idx, _qq, _sig, _eps = nb_force.getExceptionParameterOffset(off_idx)
        if str(param) != str(lambda_name):
            continue
        if int(exc_idx) in frozen_indices:
            raise RuntimeError(
                f"{context}: 配体内部 exception #{int(exc_idx)} 上挂了 "
                f"{lambda_name} 的 parameter offset —— 它并没有被冻结。"
                "P0-01 的 v3 annihilation 口径要求既有 L–L exception 逐 λ 恒定。"
            )


# ================= ibs_engine.py 顶部新增 =================
def configure_coalchemical_neutral_decharging(
    system: openmm.System,
    ligand_indices: List[int],
    topology,
    positions,
    box_vectors=None,
    lambda_name: str = "lam_coul",
    co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict, List[int]]:
    """共炼金反离子策略：反离子与配体同步消电，保持全局严格电中性。

    ⚠️ 这是 **co-annihilation**（异号反离子跟着一起消电），**不是** §0 选定的
    charge-transfer（ligand q→0 / co-ion 0→q）。生产路线见 MEM-00a / B3
    （`configure_charge_transfer_decharging`）。本函数按 MEM-00a-2 降级为
    **实验对照专用**，只允许用于水盒 / lipid slab 的方法对照。

    ⚠️ **[MEM-00c] 配体带净电荷时必须传 `co_alchemical_ion_spec`。**
    本函数**不再自己选离子**：身份由 `select_co_alchemical_ion_once()` 选一次、
    落盘，这里只做 `verify_co_alchemical_ion_identity()` 只读核对。
    没有 spec 就 fail closed —— 因为"每个入口自己选一次"在跨进程 resume 下
    会静默选中不同粒子，让 u_kn 与动力学用上不同的 Hamiltonian。
    """
    nb_force = next((f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
    if nb_force is None:
        raise RuntimeError("系统中未找到 NonbondedForce")

    ligand_set = set(int(i) for i in ligand_indices)
    raw_lig_net_charge = 0.0
    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(idx)
        raw_lig_net_charge += q.value_in_unit(unit.elementary_charge)
    if not np.isfinite(raw_lig_net_charge):
        raise RuntimeError("配体净电荷为 NaN/Inf，拒绝配置 co-annihilation Hamiltonian")
    lig_net_charge = int(round(raw_lig_net_charge))
    if abs(raw_lig_net_charge - lig_net_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
        raise RuntimeError(
            f"配体净电荷 {raw_lig_net_charge:+.6f} e 不接近整数（容差 1e-3 e）"
        )

    if lig_net_charge == 0:
        print("  配体为电中性，无需共消电反离子。使用标准 PME Offset。")
        best_ion_indices: List[int] = []
        ion_meta: Dict[str, Any] = {}
    else:
        # [MEM-00c] 只核对，不重选。没有 spec 就停 —— 见本函数 docstring。
        if co_alchemical_ion_spec is None:
            raise RuntimeError(
                f"检测到带电配体 (Net Charge: {lig_net_charge:+d})，但没有传入 "
                "`co_alchemical_ion_spec`。\n"
                "    [MEM-00c] co-ion 身份必须由 `select_co_alchemical_ion_once()` "
                "**选一次并落盘**，动力学 / REMD 副本 / compute_u_kn / resume 全部只读消费。\n"
                "    本函数以前会在这里自己调一次选择器；因为选择结果是坐标的连续函数，"
                "而首跑（预平衡输出 + 2000 步最小化）与 resume（直接读 DCD 末帧）"
                "喂进来的坐标不同，于是同一条腿的动力学与 u_kn 可能选中**不同粒子** —— "
                "ΔG 会错且没有任何异常现象。\n"
                "    修法是在上游选一次、把 spec 传下来；**不要**在这里恢复自动选择。"
            )
        pinned = verify_co_alchemical_ion_identity(
            co_alchemical_ion_spec,
            system=system,
            topology=topology,
            ligand_net_charge_e=lig_net_charge,
            context="configure_coalchemical_neutral_decharging",
        )
        if (
            co_alchemical_ion_spec.get("charge_treatment")
            == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
        ):
            raise RuntimeError(
                "传进来的是 charge-transfer 的 co-ion 身份 spec，但调用的是 "
                "co-annihilation 的 charging builder。两者的哈密顿量相反"
                "（q_phys→0 vs 0→q_L），请改调 `configure_charge_transfer_decharging`。"
            )
        best_ion_indices = list(pinned)
        # restraint 参数全部取自 spec，不取当前坐标（MEM-00c/00d：参考量属于身份的
        # 一部分）。注入由 `_inject_co_alchemical_ion_restraints` 统一负责，
        # 与 charge-transfer 共用同一份形式（§4.4）。
        ion_meta = dict(co_alchemical_ion_spec.get("selection_provenance") or {})
        print(
            f"  [MEM-00c] 复用已冻结的共炼金反离子身份: Indices {best_ion_indices} "
            f"(fingerprint {str(co_alchemical_ion_spec.get('fingerprint'))[:12]}…)"
        )

    # 3. 注入全局 Lambda 与 ParameterOffset
    existing_params = [nb_force.getGlobalParameterName(i) for i in range(nb_force.getNumGlobalParameters())]
    if lambda_name not in existing_params:
        nb_force.addGlobalParameter(lambda_name, 1.0)

    original_charges = {}
    ligand_params = {}
    alchemical_set = set(ligand_set)
    
    # 3.1 配体 Offset
    for idx in ligand_set:
        q, sig, eps = nb_force.getParticleParameters(idx)
        original_charges[idx] = q
        ligand_params[idx] = (q, sig, eps)
        nb_force.setParticleParameters(idx, 0.0*unit.elementary_charge, sig, eps)
        nb_force.addParticleParameterOffset(lambda_name, idx, q, 0.0*unit.nanometer, 0.0*unit.kilojoule_per_mole)

    # 3.2 反离子 Offset
    for best_ion_idx in best_ion_indices:
        ion_q, ion_sig, ion_eps = nb_force.getParticleParameters(best_ion_idx)
        original_charges[best_ion_idx] = ion_q
        alchemical_set.add(int(best_ion_idx))
        nb_force.setParticleParameters(best_ion_idx, 0.0*unit.elementary_charge, ion_sig, ion_eps)
        nb_force.addParticleParameterOffset(lambda_name, best_ion_idx, ion_q, 0.0*unit.nanometer, 0.0*unit.kilojoule_per_mole)

    # 3.3 保持配体内部静电常量；仅对“单端炼金”的 exception 施加线性 offset。
    # [0831issue P2] 见 _assert_frozen_ligand_ligand_exceptions。
    frozen_ll_pairs: Dict[Tuple[int, int], Tuple[int, float]] = {}
    for i in range(nb_force.getNumExceptions()):
        p1, p2, charge_prod, sig, eps = nb_force.getExceptionParameters(i)
        p1, p2 = int(p1), int(p2)
        if p1 in ligand_set and p2 in ligand_set:
            frozen_ll_pairs[(min(p1, p2), max(p1, p2))] = (
                int(i),
                float(
                    charge_prod.value_in_unit(unit.elementary_charge**2)
                    if hasattr(charge_prod, "value_in_unit")
                    else charge_prod
                ),
            )
            continue
        if (p1 in alchemical_set) ^ (p2 in alchemical_set):
            nb_force.setExceptionParameters(i, p1, p2, 0.0*unit.elementary_charge**2, sig, eps)
            nb_force.addExceptionParameterOffset(lambda_name, i, charge_prod, 0.0*unit.nanometer, 0.0*unit.kilojoule_per_mole)

    # [P0-01, 2026-08-30] 配体内部普通 L–L 对（≥1-5、无 exception）**不再**补成
    # 显式 exception。原因：OpenMM 的 exception 从不使用普通非键的 cutoff/PME
    # 处理（API 文档 "cutoffs are never applied to exceptions"；OpenMM #2310 也
    # 记录了把参数相同的普通对改成 exception 时 PME 能量/力会变），补对会让
    # λ=1 的物理端点不再是原始 System——tests/test_pme_decharge_endpoint_
    # equivalence.py 在 26 粒子最小 PME 体系上实测能量偏差 ~1e-2 kJ/mol。
    # 现口径（PME_DECHARGE_MODEL_VERSION → v5，couple-intramol=no）：
    #   - 既有 L–L exception（1-2/1-3 排除、1-4 缩放）的 chargeProd 是独立参数、
    #     不读粒子电荷 → 逐 λ 恒定（frozen_ll_pairs 的冻结语义不变）；
    #   - 普通 L–L 对在主 NB 力里随粒子 offset 按 λ² 缩放，由
    #     `_freeze_ligand_internal_coulomb` 挂的 (1-λ²) 补偿项补回，合计逐 λ 恒定。
    # ⚠️ v4 曾把普通 L–L 对整个湮灭掉，理由是"两腿的配体内部 Hamiltonian 完全相同，
    # 湮灭项在 ΔG_bind 里严格相消"。**那个理由是错的**——哈密顿量相同不等于自由能
    # 相同，结合态与自由态的配体构象系综不同，这一项的两腿之差是真实的构象重组功，
    # 必须靠采样得到。实测在柔性极性配体上它是 −88.7 kJ/mol，恰好是该体系
    # ΔG_bind 的全部误差；在刚性非极性配体（4W53/toluene）上只有 1.56 kJ/mol，
    # 所以一直没暴露。详见 `_freeze_ligand_internal_coulomb` 的 docstring。

    # [0831issue P2] P0-01 前提的断言（此前只收集不校验）。
    # [couple-intramol=no] 配体内部 ≥1-5 库仑冻结为 λ 无关（见函数 docstring）。
    _freeze_ligand_internal_coulomb(
        system, nb_force, ligand_set, lambda_name, original_charges, "co-alchemical neutral decharging"
    )
    _assert_frozen_ligand_ligand_exceptions(
        nb_force, lambda_name, frozen_ll_pairs, "co-alchemical neutral decharging"
    )

    if best_ion_indices:
        _inject_co_alchemical_ion_restraints(system, co_alchemical_ion_spec)
        # §7.2：逐 λ 电荷账目由同一个函数核对（与 charge-transfer 共用），
        # 不靠"异号反离子应该会抵消"这句话。
        charging_charge_conservation_report(
            nb_force,
            lambda_name,
            ligand_indices=sorted(ligand_set),
            co_ion_indices=best_ion_indices,
            ligand_net_charge_e=lig_net_charge,
        )

    print("  [OK] 共炼金反离子防御阵列部署完毕。PME 倒空间计算全程严格电中性！")
    return original_charges, list(best_ion_indices)



def _freeze_ligand_internal_coulomb(
    system: openmm.System,
    nb_force: openmm.NonbondedForce,
    ligand_indices,
    lambda_name: str,
    original_charges,
    context_label: str,
) -> int:
    """给去电荷腿挂上 (1-λ²) 的配体内部库仑补偿项，使分子内静电逐 λ 恒定。

    去电荷腿把配体的 base 电荷清零、整份电荷挂成 `lambda_name` 的 particle offset，
    于是**所有**含配体的对都按 λ 走 —— 配体–环境对是 λ¹（对侧不带 offset），
    配体–配体对是 λ²（两侧都带）。后者意味着配体的分子内库仑随 λ 一起湮灭。

    以前的口径（`pme_decharge_v4_..._normal_pairs_annihilated`）认为这一项会在
    ΔG_bind 里「严格相消」，理由是两腿的配体内部 Hamiltonian 完全相同。

    把定性说准，因为这里最容易被读歪：

    * v4 的循环**形式上是闭合的** —— 两腿终态是同一个参考态（内部库仑被湮灭的
      配体）。完美采样极限下 v4 与 v5 给同一个 ΔG_bind。**v4 不是「公式写错了」。**
    * 但「**严格相消**」这句仍然是假的：它不是恒等抵消。哈密顿量相同不等于自由能
      相同 —— 自由能是系综平均，而结合态（Boresch 约束在口袋里）与自由态（体相水）
      的配体构象系综不同，这一项的两腿之差是**真实的构象重组功**，必须靠采样得到。
      写成「严格相消」等于宣称它不必被采样 —— 那正是 v4 被放行的根据。
    * 真正的失效是**条件数**：把一个 ~90 kJ/mol 的物理量做成两个 ~500 kJ/mol 项之差，
      而结合态配体在 500 ps × 8 个 λ 态里完不成构象弛豫 ⟹ 滞后。实测误差与
      ⟨U_intra⟩ 的两腿平均能差 1:1 吻合（零补偿、纯线性响应），正是「根本没弛豫」
      的特征，不是收敛后的真值。
    * **v5 不是删掉物理**：该项变成 λ 无关 ⟹ 对两腿 ΔG 各贡献 0，重组功改走
      配体–环境那条条件良好的路。

    实测：4W53/toluene 这种刚性非极性配体上两腿只差 +1.56 kJ/mol（所以一直没暴露），
    柔性极性配体上 ⟨U_intra⟩ 两腿差到 −88.7 kJ/mol —— 恰好是该体系 ΔG_bind 的全部误差。

    改成 `couple-intramol=no` 的口径后，stage 1 只承载配体–环境那一项，与 vanishing
    腿（softcore 力本来就只 `addInteractionGroup(配体, 环境)`）口径一致；参考态也从
    「内部静电被删掉的虚构分子」变回真实分子在真空中。

    返回挂上的对数（0 表示配体只有一个原子之类的退化情形）。
    """
    ligand_set = sorted(set(int(i) for i in ligand_indices))
    charges = {}
    offsets_by_particle = {}
    for off_idx in range(nb_force.getNumParticleParameterOffsets()):
        param, particle_idx, charge_scale, _s, _e = nb_force.getParticleParameterOffset(off_idx)
        if str(param) != str(lambda_name):
            continue
        offsets_by_particle.setdefault(int(particle_idx), []).append(
            _value_in_elementary_charge(charge_scale)
        )

    for idx in ligand_set:
        q_base, _sig, _eps = nb_force.getParticleParameters(idx)
        q_base_e = _value_in_elementary_charge(q_base)
        scales = offsets_by_particle.get(idx, [])
        if abs(q_base_e) > 1.0e-9 or len(scales) != 1:
            raise RuntimeError(
                f"[{context_label}] 配体内部库仑冻结要求每个配体原子 base 电荷为 0 且只带"
                f"一个 {lambda_name} offset，但粒子 {idx} 是 base={q_base_e:.12g} e、"
                f"offset 数={len(scales)}。此时 q_i(λ)·q_j(λ) 不是 λ²·q_i·q_j，"
                "(1-λ²) 的补偿项就不再精确 —— 与其静默算错，这里 fail closed。"
            )
        q_orig = original_charges.get(idx) if original_charges else None
        q_orig_e = _value_in_elementary_charge(q_orig) if q_orig is not None else scales[0]
        if abs(q_orig_e - scales[0]) > 1.0e-9:
            raise RuntimeError(
                f"[{context_label}] 粒子 {idx} 的 {lambda_name} offset ({scales[0]:.12g} e) "
                f"与记录的原始电荷 ({q_orig_e:.12g} e) 不一致。"
            )
        charges[idx] = scales[0]

    ref_excl = [
        (int(nb_force.getExceptionParameters(i)[0]), int(nb_force.getExceptionParameters(i)[1]))
        for i in range(nb_force.getNumExceptions())
    ]
    force = create_ligand_internal_coulomb_force(
        nb_force,
        ligand_set,
        charges,
        system=system,
        reference_exclusions=ref_excl,
        scale_expr=f"1 - {lambda_name}^2",
        global_parameters=[(lambda_name, 1.0)],
    )
    if force is None:
        return 0
    system.addForce(force)
    print(
        f"  [couple-intramol=no] 配体内部 ≥1-5 库仑已冻结为 λ 无关："
        f"{force.getNumBonds()} 对，前缀 (1 - {lambda_name}^2)（λ=1 时恒为 0，端点不变）。"
    )
    return force.getNumBonds()


def configure_pme_ligand_charge_offsets(
    system: openmm.System,
    ligand_indices: List[int],
    lambda_name: str = "lambda_coul",
    allow_charged_ligand: bool = False,
    topology=None,
    positions=None,
    box_vectors=None,
    co_alchemical_ion_spec: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Configure a PME-preserving Coulomb leg on the native NonbondedForce.

    This avoids putting ligand-environment electrostatics into
    CustomNonbondedForce, which would truncate Coulomb at the cutoff.  Charged
    ligands are rejected by default because co-alchemical ion schemes change the
    thermodynamic cycle unless they are implemented and validated explicitly.
    """
    nb_force = next((f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)), None)
    if nb_force is None:
        raise RuntimeError("系统中未找到 NonbondedForce，无法配置 PME 去电荷阶段。")

    ligand_set = set(int(i) for i in ligand_indices)
    net_charge = 0.0
    for idx in ligand_set:
        q, _, _ = nb_force.getParticleParameters(idx)
        net_charge += q.value_in_unit(unit.elementary_charge)
    if not np.isfinite(net_charge):
        raise RuntimeError("配体净电荷为 NaN/Inf，拒绝配置 PME 去电荷阶段")
    rounded_charge = int(round(net_charge))
    if abs(net_charge - rounded_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
        raise RuntimeError(
            f"配体净电荷 {net_charge:+.6f} e 不接近整数（容差 1e-3 e）；"
            "拒绝把异常部分电荷判为中性"
        )
    if rounded_charge != 0 and not allow_charged_ligand:
        raise RuntimeError(
            f"检测到带净电配体 ({rounded_charge:+d} e)。当前 PME 去电荷路径不再自动共炼金反离子，"
            "请先使用净电中性配体、显式实现可验证的离子方案，或改用支持 PME 多态炼金的底层实现。"
        )

    if rounded_charge != 0:
        if topology is None:
            raise RuntimeError(
                "带电配体的 PME 去电荷路径需要 topology 才能核对冻结的 co-ion 身份。"
            )
        # 走哪条共炼金路线由**冻结 spec 里记录的 charge_treatment** 决定，不是由这里
        # 再猜一次。理由：spec 的端点电荷已经按某一条路线算好了（q_phys→0 还是 0→q_L），
        # 用另一条路线的 builder 消费它就是"声明一种哈密顿量、实际跑另一种"。
        # spec 缺失时保持既有 co-annihilation 分支的 fail-closed 报错口径（MEM-00c）。
        treatment = (co_alchemical_ion_spec or {}).get("charge_treatment")
        if treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
            original_charges, ion_indices = configure_charge_transfer_decharging(
                system,
                ligand_indices,
                topology,
                lambda_name=lambda_name,
                co_alchemical_ion_spec=co_alchemical_ion_spec,
            )
            mode = "co_alchemical_charge_transfer"
        else:
            if positions is None:
                raise RuntimeError(
                    "co-annihilation 的 PME 去电荷路径需要初始 positions。"
                )
            original_charges, ion_indices = configure_coalchemical_neutral_decharging(
                system,
                ligand_indices,
                topology,
                positions,
                box_vectors=box_vectors,
                lambda_name=lambda_name,
                # [MEM-00c] 身份只读透传；为 None 时下游 fail closed，不再自动选。
                co_alchemical_ion_spec=co_alchemical_ion_spec,
            )
            mode = "coalchemical_counterion"
        return {
            "mode": mode,
            "charge_treatment": treatment,
            "ion_indices": [int(i) for i in ion_indices],
            "n_offsets": len(original_charges),
            "co_alchemical_ion_fingerprint": (
                (co_alchemical_ion_spec or {}).get("fingerprint")
            ),
        }

    existing_params = [nb_force.getGlobalParameterName(i) for i in range(nb_force.getNumGlobalParameters())]
    if lambda_name not in existing_params:
        nb_force.addGlobalParameter(lambda_name, 1.0)

    ligand_params = {}
    for idx in ligand_set:
        q, sig, eps = nb_force.getParticleParameters(idx)
        ligand_params[idx] = (q, sig, eps)
        nb_force.setParticleParameters(idx, 0.0 * unit.elementary_charge, sig, eps)
        nb_force.addParticleParameterOffset(
            lambda_name,
            idx,
            q,
            0.0 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )

    # 冻结所有 L-L 非键对，使去电荷腿只作用于配体-环境静电，而不污染配体内部 self-energy。
    # [0831issue P2] 记下 (exc_idx, chargeProd) 而不只是 pair：函数末尾的
    # _assert_frozen_ligand_ligand_exceptions 需要它们来复核"确实冻结了"。
    frozen_ll_pairs: Dict[Tuple[int, int], Tuple[int, float]] = {}
    for exc_idx in range(nb_force.getNumExceptions()):
        p1, p2, charge_prod, sig, eps = nb_force.getExceptionParameters(exc_idx)
        p1, p2 = int(p1), int(p2)
        if p1 in ligand_set and p2 in ligand_set:
            frozen_ll_pairs[(min(p1, p2), max(p1, p2))] = (
                int(exc_idx),
                float(
                    charge_prod.value_in_unit(unit.elementary_charge**2)
                    if hasattr(charge_prod, "value_in_unit")
                    else charge_prod
                ),
            )
            continue
        if (p1 in ligand_set) ^ (p2 in ligand_set):
            nb_force.setExceptionParameters(
                exc_idx,
                p1,
                p2,
                0.0 * unit.elementary_charge**2,
                sig,
                eps,
            )
            nb_force.addExceptionParameterOffset(
                lambda_name,
                exc_idx,
                charge_prod,
                0.0 * unit.nanometer,
                0.0 * unit.kilojoule_per_mole,
            )

    # [P0-01, 2026-08-30] 配体内部普通 L–L 对（≥1-5、无 exception）**不再**补成
    # 显式 exception。原因：OpenMM 的 exception 从不使用普通非键的 cutoff/PME
    # 处理（API 文档 "cutoffs are never applied to exceptions"；OpenMM #2310 也
    # 记录了把参数相同的普通对改成 exception 时 PME 能量/力会变），补对会让
    # λ=1 的物理端点不再是原始 System——tests/test_pme_decharge_endpoint_
    # equivalence.py 在 26 粒子最小 PME 体系上实测能量偏差 ~1e-2 kJ/mol。
    # 现口径（PME_DECHARGE_MODEL_VERSION → v5，couple-intramol=no）：
    #   - 既有 L–L exception（1-2/1-3 排除、1-4 缩放）的 chargeProd 是独立参数、
    #     不读粒子电荷 → 逐 λ 恒定（frozen_ll_pairs 的冻结语义不变）；
    #   - 普通 L–L 对在主 NB 力里随粒子 offset 按 λ² 缩放，由
    #     `_freeze_ligand_internal_coulomb` 挂的 (1-λ²) 补偿项补回，合计逐 λ 恒定。
    # ⚠️ v4 曾把普通 L–L 对整个湮灭掉，理由是"两腿的配体内部 Hamiltonian 完全相同，
    # 湮灭项在 ΔG_bind 里严格相消"。**那个理由是错的**——哈密顿量相同不等于自由能
    # 相同，结合态与自由态的配体构象系综不同，这一项的两腿之差是真实的构象重组功，
    # 必须靠采样得到。实测在柔性极性配体上它是 −88.7 kJ/mol，恰好是该体系
    # ΔG_bind 的全部误差；在刚性非极性配体（4W53/toluene）上只有 1.56 kJ/mol，
    # 所以一直没暴露。详见 `_freeze_ligand_internal_coulomb` 的 docstring。
    # [0831issue P2] P0-01 前提的断言（此前只收集不校验）。
    # [couple-intramol=no] 配体内部 ≥1-5 库仑冻结为 λ 无关（见函数 docstring）。
    _freeze_ligand_internal_coulomb(
        system,
        nb_force,
        ligand_set,
        lambda_name,
        # 中性分支用的是 `ligand_params`（带电分支才有 `original_charges`）。
        {idx: params[0] for idx, params in ligand_params.items()},
        "PME ligand-only charge offsets",
    )
    _assert_frozen_ligand_ligand_exceptions(
        nb_force, lambda_name, frozen_ll_pairs, "PME ligand-only charge offsets"
    )
    return {"mode": "ligand_only_offset", "ion_index": None, "n_offsets": len(ligand_set)}

def generate_overlapping_windows(
    n_states: int,
    pts_per_window: int = 6,
    overlap: int = 2,
    n_windows: int = None
) -> List[Tuple[int, int]]:
    """
    生成重叠窗口划分

    示例：n_states=13, pts_per_window=6, overlap=2 → [(0,6), (4,10), (7,13)]

    🔑 [0831issue P2] 这个示例原来写的是 `(8,13)`，实测是 `(7,13)`。
    起点由 `np.linspace(0, max_start, n_windows)` 后逐个 `int(round(...))` 得到，
    而 numpy/Python 的 `round` 是**银行家舍入**（round-half-to-even），
    linspace 给出的 3.5 被舍成 4 而不是 5，最后一窗起点因此是 7。
    末尾有覆盖性 `RuntimeError` 兜底（保证并集覆盖全部态），所以这只是文档与实现
    不符，没有正确性风险 —— 但按错的示例去推窗口边界会误判。
    """
    n_states = int(n_states)
    pts_per_window = int(pts_per_window)
    overlap = max(0, int(overlap))
    if n_windows is not None:
        n_windows = max(1, int(n_windows))

    if n_states <= pts_per_window:
        return [(0, n_states)]

    if n_windows is None:
        stride = max(1, pts_per_window - overlap)
        n_windows = max(2, math.ceil((n_states - overlap) / stride))
        effective_pts = pts_per_window
    else:
        effective_pts = max(
            pts_per_window,
            math.ceil((n_states + (n_windows - 1) * overlap) / n_windows),
        )

    if n_windows > 1:
        max_start = max(0, n_states - effective_pts)
        starts = np.linspace(0, max_start, n_windows)
        starts = [int(round(x)) for x in starts]
    else:
        starts = [0]

    windows = []
    for start in starts:
        if start >= n_states:
            break
        end = min(start + effective_pts, n_states)
        windows.append((start, end))

    if windows and windows[-1][1] < n_states:
        windows[-1] = (max(0, n_states - effective_pts), n_states)

    # 🔑 [2026-09-09] 去掉重复窗口。
    #
    # 调用方给了显式 `n_windows` 且 `max_start < n_windows - 1` 时，
    # `linspace` 取整必然产生重复起点。实例（config 可达，
    # `abfe_pipeline` 把 `kwargs.get("n_windows")` 原样透传）：
    #   n_states=8, n_windows=6, pts_per_window=6, overlap=2
    #   → effective_pts=6, max_start=2, linspace(0,2,6) → [0,0,1,1,2,2]
    #   → [(0,6),(0,6),(1,7),(1,7),(2,8),(2,8)]
    # 下面的覆盖性检查只看并集，重复窗口照样放行；而每个重复窗口随后都会成为
    # 自己的 `local_idx`、有自己的能量文件（`load_ibs_window_outputs_from_dir`
    # 纯按位置索引），并作为**独立窗口**进入协方差链 —— 同一段 λ 的 ΔF 被走两遍。
    deduped = []
    for window in windows:
        if window not in deduped:
            deduped.append(window)
    if len(deduped) != len(windows):
        logger.warning(
            "[窗口划分] n_states=%d, n_windows=%s, pts_per_window=%d, overlap=%d "
            "产生了 %d 个重复窗口（linspace 起点取整重合），已去重：%d → %d 个窗口。"
            "如果需要更多窗口，减小 pts_per_window 或增大 n_states。",
            n_states, n_windows, pts_per_window, overlap,
            len(windows) - len(deduped), len(windows), len(deduped),
        )
    windows = deduped

    # 验证覆盖性
    covered = set()
    for s, e in windows:
        covered.update(range(s, e))
    if len(covered) != n_states:
        raise RuntimeError(f"窗口划分未完全覆盖 {n_states} 个状态")
    return windows

