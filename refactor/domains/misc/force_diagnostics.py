# ============================================================================
# 力组诊断工具 (开发/调试用，保留)
# ============================================================================

def _check_boresch_geometry_safe(context, boresch_params, min_sin_theta=0.1):
    """
    检查当前锚点几何是否远离奇点（θ 偏离 0° 和 180°）。
    返回 (is_safe, r0_nm, thA_deg, thB_deg)
    """
    state = context.getState(getPositions=True)
    pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    box_vectors = state.getPeriodicBoxVectors()
    rec = [int(i) for i in boresch_params["receptor_indices"]]
    lig = [int(i) for i in boresch_params["ligand_indices"]]

    # BOR-01：解缠与 `calc_boresch_from_last_frame`（真正写进限制势的那份）共用
    # 同一份实现，否则"校验解了缠、提交没解"会让跨边界的 r0 差一个盒矢量还看不出来。
    try:
        H0, H1, H2, L0, L1, L2 = unwrap_boresch_anchors_nm(
            [pos[i] for i in rec], [pos[i] for i in lig], box_vectors
        )
    except ValueError as exc:
        print(f"  [ERR] Boresch minimum-image 几何无法计算：{exc}")
        return False, np.nan, np.nan, np.nan
    anchor_coords = np.asarray([H0, H1, H2, L0, L1, L2], dtype=float)
    if not np.all(np.isfinite(anchor_coords)):
        print("  [ERR] Boresch 锚点坐标包含 NaN/Inf，几何检查判定为不安全")
        return False, np.nan, np.nan, np.nan

    r0 = np.linalg.norm(H0 - L0)
    if not np.isfinite(r0):
        print("  [ERR] Boresch r0 为 NaN/Inf，几何检查判定为不安全")
        return False, np.nan, np.nan, np.nan

    def calc_angle(a, b, c):
        ba, bc = a - b, c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom <= 1e-12 or not np.isfinite(denom):
            return np.nan
        cos_val = np.clip(
            np.dot(ba, bc) / denom,
            -1.0, 1.0
        )
        return np.arccos(cos_val)

    thA = calc_angle(H1, H0, L0)
    thB = calc_angle(H0, L0, L1)
    if not np.all(np.isfinite([thA, thB])):
        print("  [ERR] Boresch θA/θB 为 NaN/Inf，几何检查判定为不安全")
        return False, float(r0), np.nan, np.nan

    sinA, sinB = np.sin(thA), np.sin(thB)
    if (
        not np.all(np.isfinite([sinA, sinB]))
        or sinA < min_sin_theta
        or sinB < min_sin_theta
    ):
        print(f"  [ERR] Boresch 角度接近奇点：θA={np.degrees(thA):.1f}° (sin={sinA:.3f})，"
              f"θB={np.degrees(thB):.1f}° (sin={sinB:.3f})")
        return False, r0, np.degrees(thA), np.degrees(thB)

    return True, r0, np.degrees(thA), np.degrees(thB)


def _has_valid_boresch_restraint(params) -> bool:
    """仅当 Boresch 参数包含完整 3+3 锚点时才认为可注入/可启用。"""
    if not isinstance(params, dict):
        return False
    rec_idx = params.get("receptor_indices") or []
    lig_idx = params.get("ligand_indices") or []
    return len(rec_idx) == 3 and len(lig_idx) == 3


def current_max_force_from_context(context) -> float:
    """返回当前上下文的全体系最大合力范数，用于生产前安全拦截。"""
    state = context.getState(getForces=True)
    forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    if forces.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(forces, axis=1)))


def diagnose_top_force_atoms(
    context,
    system,
    topology=None,
    ligand_indices=None,
    prefix: str = "窗口诊断",
    top_n: int = 20,
):
    """打印最大受力原子及各 force group 贡献，用于定位高力来源。"""
    from openmm import unit
    ligand_set = set(int(i) for i in (ligand_indices or []))
    atoms = list(topology.atoms()) if topology is not None else []
    water_names = {"HOH", "WAT", "SOL", "TIP3", "TIP3P"}
    ion_names = {"NA", "SOD", "CL", "CLA", "K", "POT", "MG", "CA"}

    state = context.getState(getForces=True)
    total_forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    if total_forces.size == 0:
        return

    group_forces = {}
    for gid in sorted({force.getForceGroup() for force in system.getForces()}):
        try:
            g_state = context.getState(getForces=True, groups={gid})
            group_forces[gid] = g_state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
        except Exception:
            continue

    norms = np.linalg.norm(total_forces, axis=1)
    order = np.argsort(norms)[::-1][:top_n]

    print(f"\n[{prefix}] 最大受力原子定位:")
    print(
        f"{'rank':<4} | {'atom':<7} | {'residue':<12} | {'role':<7} | "
        f"{'total':>10} | {'G0':>10} | {'G1':>10} | {'G2':>10} | {'G3':>10} | {'G4':>10}"
    )
    print("-" * 105)
    for rank, idx in enumerate(order, 1):
        atom_label = str(idx)
        residue_label = "N/A"
        role = "env"
        if idx < len(atoms):
            atom = atoms[int(idx)]
            residue = atom.residue
            atom_label = f"{idx}:{atom.name}"
            residue_label = f"{residue.name}{residue.index}"
            res_name = residue.name.upper()
            if idx in ligand_set:
                role = "ligand"
            elif res_name in water_names:
                role = "water"
            elif res_name in ion_names:
                role = "ion"
        elif idx in ligand_set:
            role = "ligand"

        def gnorm(gid):
            arr = group_forces.get(gid)
            if arr is None or idx >= len(arr):
                return 0.0
            return float(np.linalg.norm(arr[int(idx)]))

        print(
            f"{rank:<4} | {atom_label:<7} | {residue_label:<12} | {role:<7} | "
            f"{float(norms[int(idx)]):10.1f} | {gnorm(0):10.1f} | {gnorm(1):10.1f} | "
            f"{gnorm(2):10.1f} | {gnorm(3):10.1f} | {gnorm(4):10.1f}"
        )
    print("-" * 105)


def _detect_constraint_deadlock(pre_breakdown, post_breakdown, n_constraints: int) -> Optional[str]:
    """根据测试步进前后 Bond/Angle 核心力异常放大，识别约束-非键死锁。"""
    if n_constraints < 1000 or not pre_breakdown or not post_breakdown:
        return None
    pre_bond = float(pre_breakdown.get("Bond", {}).get("max", 0.0))
    pre_angle = float(pre_breakdown.get("Angle", {}).get("max", 0.0))
    post_bond = float(post_breakdown.get("Bond", {}).get("max", 0.0))
    post_angle = float(post_breakdown.get("Angle", {}).get("max", 0.0))

    bond_exploded = post_bond > max(3000.0, 1.4 * max(pre_bond, 1.0))
    angle_exploded = post_angle > max(2500.0, 1.4 * max(pre_angle, 1.0))
    if bond_exploded and angle_exploded:
        return (
            f"疑似约束死锁：Bond Max {pre_bond:.1f} -> {post_bond:.1f}, "
            f"Angle Max {pre_angle:.1f} -> {post_angle:.1f}, constraints={n_constraints}"
        )
    return None

def diagnose_force_groups_detailed(context, system, prefix="窗口诊断"):
    """按唯一 ForceGroup 聚合的受力拆解，避免同组总力被重复打印到每个力对象上。"""
    from openmm import unit
    print(f"\n🔍 [{prefix}] 逐力组受力拆解报告:")
    print(f"{'Group':<8} | {'成员力类型':<52} | {'Max|F| (kJ/mol/nm)':<20} | {'RMS|F|':<12} | {'状态'}")
    print("-" * 125)

    group_members = {}
    for force in system.getForces():
        gid = force.getForceGroup()
        group_members.setdefault(gid, []).append(type(force).__name__)

    for gid in sorted(group_members):
        member_types = group_members[gid]
        member_summary = ", ".join(member_types[:3])
        if len(member_types) > 3:
            member_summary += f", ... x{len(member_types)}"
        try:
            state = context.getState(getForces=True, groups={gid})
            forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/openmm.unit.nanometer)
            if forces.size == 0:
                continue
            norms = np.linalg.norm(forces, axis=1)
            max_f = np.max(norms)
            rms_f = np.sqrt(np.mean(norms**2))
            status = "[ERR] 爆炸源" if max_f > 10000 else ("[WARN] 偏高" if max_f > 2000 else "[OK] 正常")
            print(f"Group {gid:<2} | {member_summary:<52} | {max_f:<20.2f} | {rms_f:<12.2f} | {status}")
        except Exception as e:
            print(f"Group {gid:<2} | {member_summary:<52} | {'(CV/元力跳过)':<20} | {'N/A':<12} | {str(e)[:25]}")
    print("-" * 125)
    n_cons = system.getNumConstraints()
    if n_cons > 0:
        print(f"  系统含 {n_cons} 个刚性约束 (SHAKE/LINCS)。若上述力组均正常但合力爆炸，根因必为约束死锁或初始键长畸变。")


def print_force_group_details(system: openmm.System, prefix: str = "系统"):
    """生产级力组详细诊断：打印每个力的类型、组号、粒子/键数、关键全局参数"""
    print(f"\n🔍 [{prefix}] 力组架构详细报告:")
    print(f"{'ID':<4} | {'Force Group':<12} | {'Force Type':<28} | {'Particles/Bonds':<16} | {'Key Global Params'}")
    print("-" * 90)
    for i, f in enumerate(system.getForces()):
        gid = f.getForceGroup()
        ftype = type(f).__name__
        if hasattr(f, 'getNumParticles'):
            count_str = f"{f.getNumParticles()} particles"
        elif hasattr(f, 'getNumBonds'):
            count_str = f"{f.getNumBonds()} bonds"
        elif hasattr(f, 'getNumAngles'):
            count_str = f"{f.getNumAngles()} angles"
        elif hasattr(f, 'getNumTorsions'):
            count_str = f"{f.getNumTorsions()} torsions"
        else:
            count_str = "N/A"
        params = []
        if hasattr(f, 'getNumGlobalParameters'):
            for j in range(f.getNumGlobalParameters()):
                name = f.getGlobalParameterName(j)
                val = f.getGlobalParameterDefaultValue(j)
                params.append(f"{name}={val:.3g}")
        param_str = ", ".join(params[:4]) + ("..." if len(params) > 4 else "")
        print(f"{i:<4} | Group {gid:<6} | {ftype:<28} | {count_str:<16} | {param_str}")
    print("-" * 90)

def diagnose_force_breakdown(main_context, main_system, prefix=""):
    """
    非侵入式力分解：利用临时系统 + 当前坐标，单独计算 Bond/Angle/Torsion/NB 的受力。
    完全不修改主系统的 ForceGroup 或力常数。

    ⚠️ [0831issue P2] **这是一份"近似重建"，不是主 Hamiltonian 的忠实分解。**
    下面重建的 NonbondedForce 复制了 particle/exception 参数，但**没有**复制：

      * switching function（`setUseSwitchingFunction` / `setSwitchingDistance`）；
      * λ 电荷 offset（`addParticleParameterOffset`，即去电荷那一路的全部 λ 依赖）；
      * 配体内部相互作用的清零（生产 builder 会把配体-配体项挪进软核 CV）。

    因此这里报出的 Group-12（NB）`max|F|` 可能被上面这些**幻影项**主导，把"爆炸源"
    指到错的地方。它只适合当"某一类键合项是不是异常大"的粗筛；要定位真实的
    NB 受力，请直接读主 context 的对应 force group。
    `_detect_constraint_deadlock` 不依赖这份重建，不受影响。
    """
    import openmm
    from openmm import unit
    import numpy as np

    # 从主系统收集需要的力类型
    bond_forces = [f for f in main_system.getForces() if isinstance(f, openmm.HarmonicBondForce)]
    angle_forces = [f for f in main_system.getForces() if isinstance(f, openmm.HarmonicAngleForce)]
    torsion_forces = [f for f in main_system.getForces() if isinstance(f, openmm.PeriodicTorsionForce)]
    nb_forces = [f for f in main_system.getForces() if isinstance(f, openmm.NonbondedForce)]

    # 构建一个极简系统，只包含上述力，并使用与主系统相同的粒子数
    num_particles = main_system.getNumParticles()
    diag_sys = openmm.System()
    for _ in range(num_particles):
        diag_sys.addParticle(1.0)  # 质量随意

    # 复制 Bond 力并设 Group 10
    for bf in bond_forces:
        new_bf = openmm.HarmonicBondForce()
        for i in range(bf.getNumBonds()):
            p1, p2, length, k = bf.getBondParameters(i)
            new_bf.addBond(p1, p2, length, k)
        new_bf.setForceGroup(10)
        diag_sys.addForce(new_bf)

    # 复制 Angle 力 → Group 11
    for af in angle_forces:
        new_af = openmm.HarmonicAngleForce()
        for i in range(af.getNumAngles()):
            p1, p2, p3, angle, k = af.getAngleParameters(i)
            new_af.addAngle(p1, p2, p3, angle, k)
        new_af.setForceGroup(11)
        diag_sys.addForce(new_af)

    # 复制 Torsion 力 → Group 13
    for tf in torsion_forces:
        new_tf = openmm.PeriodicTorsionForce()
        for i in range(tf.getNumTorsions()):
            p1, p2, p3, p4, per, phase, k = tf.getTorsionParameters(i)
            new_tf.addTorsion(p1, p2, p3, p4, per, phase, k)
        new_tf.setForceGroup(13)
        diag_sys.addForce(new_tf)

    # 复制 NonbondedForce → Group 12 （保持原始参数，不归零）
    for nb in nb_forces:
        new_nb = openmm.NonbondedForce()
        for i in range(num_particles):
            q, sig, eps = nb.getParticleParameters(i)
            new_nb.addParticle(q, sig, eps)
        # 复制排除表
        for i in range(nb.getNumExceptions()):
            p1, p2, cp, sig, eps = nb.getExceptionParameters(i)
            new_nb.addException(p1, p2, cp, sig, eps)
        new_nb.setNonbondedMethod(nb.getNonbondedMethod())
        new_nb.setCutoffDistance(nb.getCutoffDistance())
        new_nb.setForceGroup(12)
        diag_sys.addForce(new_nb)

    # 获取主 context 的当前坐标
    state = main_context.getState(getPositions=True)
    pos = state.getPositions()
    box = state.getPeriodicBoxVectors()

    # 创建临时 integrator/context（仅用于力计算）
    integ = openmm.VerletIntegrator(0.001)
    plat = openmm.Platform.getPlatformByName("Reference")  # 避免 GPU 上下文冲突
    diag_context = openmm.Context(diag_sys, integ, plat)
    diag_context.setPositions(pos)
    if box is not None:
        diag_context.setPeriodicBoxVectors(*box)

    # 读取各组力
    group_map = {10: "Bond", 11: "Angle", 12: "Nonbonded", 13: "Torsion"}
    print(f"\n[{prefix}] 非侵入式力分解报告：")
    stats = {}
    for gid, name in group_map.items():
        try:
            fstate = diag_context.getState(getForces=True, groups={gid})
            forces = fstate.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
            norms = np.linalg.norm(forces, axis=1)
            max_f, avg_f, rms_f = np.max(norms), np.mean(norms), np.sqrt(np.mean(norms**2))
            stats[name] = {"max": float(max_f), "avg": float(avg_f), "rms": float(rms_f)}
            print(f"  {name:<12} | Max:{max_f:12.2f} | Avg:{avg_f:12.2f} | RMS:{rms_f:12.2f} kJ/(mol·nm)")
        except Exception as e:
            stats[name] = {"error": str(e)}
            print(f"  {name:<12} | 无法获取: {e}")

    # 清理
    del diag_context
    del diag_sys
    gc.collect()
    return stats


