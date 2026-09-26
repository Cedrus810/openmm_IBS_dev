

#=============================================================================
# 修复 2: 约束 Jacobian 解析修正 (Constraint Correction)
#=============================================================================
def calculate_constraint_jacobian_correction(system, ligand_indices, temperature=300.0):
    """
    Compatibility shim for the retired constraint-Jacobian correction.

    The present alchemical Hamiltonian keeps constraints, masses and bond
    lengths identical at both endpoints, so the constrained configurational
    factor cancels.  The former ``0.5 RT log(mu/1 Da)`` expression was neither
    a Fixman determinant nor dimensionally tied to a flexible reference and is
    intentionally not evaluated.
    """
    # Deprecated compatibility shim. OpenMM constraints are unchanged across
    # the endpoint Hamiltonians used by this project, so their configurational
    # Jacobians cancel exactly. The former mass/reference-mass expression was
    # dimensionally arbitrary and must never contribute to a production result.
    return 0.0


def constraint_identity_fingerprint(system, ligand_indices) -> Dict[str, Any]:
    """Describe the ligand constraint/mass identity used by an ABFE leg.

    The current protocol targets the *same constrained Hamiltonian* at both
    alchemical endpoints, so no standalone Fixman/Jacobian term is added.
    This helper makes that assumption auditable: a changed ligand HMR mass,
    constrained bond pair, or constrained distance changes the fingerprint and
    therefore invalidates cached legs.  Only constraints touching the ligand
    are included; environment water/protein constraints are intentionally not
    compared between complex and solvent legs.
    """
    # 与本文件其它五处（_*_hash 系列）一致的函数内局部 import：abfe_core 顶层
    # 没有 import hashlib。缺这一行会让 run_full_pipeline -> _stage_protocol_key
    # -> 这里在**流程刚启动时**就 NameError，连采样都进不去。
    import hashlib

    indices = sorted({int(i) for i in ligand_indices})
    n_particles = int(system.getNumParticles())
    if any(i < 0 or i >= n_particles for i in indices):
        raise ValueError("ligand_indices 包含超出 System 粒子范围的约束身份索引")
    ligand_set = set(indices)
    masses = []
    for i in indices:
        mass = system.getParticleMass(i)
        value = float(
            mass.value_in_unit(unit.dalton)
            if hasattr(mass, "value_in_unit")
            else mass
        )
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"配体原子 {i} 的质量非法: {value!r}")
        masses.append([i, round(value, 12)])

    constraints = []
    normalized_constraints = []
    local_index = {global_index: local for local, global_index in enumerate(indices)}
    for cidx in range(int(system.getNumConstraints())):
        a, b, distance = system.getConstraintParameters(cidx)
        a, b = int(a), int(b)
        if a not in ligand_set and b not in ligand_set:
            continue
        distance_nm = float(
            distance.value_in_unit(unit.nanometer)
            if hasattr(distance, "value_in_unit")
            else distance
        )
        if not math.isfinite(distance_nm) or distance_nm <= 0.0:
            raise ValueError(
                f"配体约束 {a}-{b} 的距离非法: {distance_nm!r} nm"
            )
        constraints.append([min(a, b), max(a, b), round(distance_nm, 12)])
        # A second, leg-comparable representation uses ligand-local ordinals;
        # complex and solvent systems need not assign the ligand the same
        # global atom indices.
        normalized_constraints.append([
            local_index.get(a, "external"),
            local_index.get(b, "external"),
            round(distance_nm, 12),
        ])
    constraints.sort()
    normalized_constraints.sort(key=lambda row: (str(row[0]), str(row[1]), row[2]))
    payload = {
        "version": 1,
        "ligand_indices": indices,
        "ligand_masses_amu": masses,
        "ligand_constraints": constraints,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    comparable_payload = {
        "version": 1,
        "ligand_masses_amu": [mass for _, mass in masses],
        "ligand_constraints": normalized_constraints,
    }
    comparable_canonical = json.dumps(
        comparable_payload, sort_keys=True, separators=(",", ":")
    )
    return {
        **payload,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "comparison_sha256": hashlib.sha256(
            comparable_canonical.encode("utf-8")
        ).hexdigest(),
    }


#=============================================================================
# 修复 6: Boresch 平衡值从预平衡最后一帧直接计算
#=============================================================================
# 完整替换 abfe_core.py 中的 calc_boresch_from_last_frame 函数
def calc_boresch_from_last_frame(positions, rec_idx, lig_idx, box_vectors=None):
    """✅ 修复 2.3：兼容 Quantity 包裹、Numpy 数组、OpenMM Vec3 列表

    **BOR-01**：传 `box_vectors` 则六个锚点先走 `unwrap_boresch_anchors_nm` 解缠，
    与校验那份（`ibs_engine._check_boresch_geometry_safe`）共用同一份解缠实现。
    不传等于调用方声明这组坐标已经连续（`runabfe` 里刚 image 过的轨迹就是这种）；
    生产调用点都传。没跨边界时解缠是**逐位恒等**，不会作废既有平衡值。
    """
    # 1. 尝试剥离单位
    if hasattr(positions, "value_in_unit"):
        pos = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    elif isinstance(positions, np.ndarray):
        pos = positions.astype(np.float64, copy=False)
    else:
        # 兼容 [openmm.Vec3, ...] 列表
        pos = np.array([[getattr(p, 'x', p[0]), getattr(p, 'y', p[1]), getattr(p, 'z', p[2])] 
                        for p in positions], dtype=np.float64)
                        
    # 确保形状为 (N, 3)。
    #
    # 之前这里对 (3, 3) 形状做过"猜转置"：假设它一定是误传的 box_vectors
    # （getPeriodicBoxVectors() 那种需要转置的形状），自动转置了事。但 (3, 3) 本身
    # 是真正歧义的——它也可能是恰好只有 3 个原子的合法 positions，此时转置反而是
    # 把 x/y/z 分量在原子间错位互换，产出错误几何却不报错（本文件开头 5.5→98.8
    # kJ/mol 的那次符号/几何混淆就是这个形状的问题）。生产调用点传的都是全体系
    # positions（原子数远大于 3），从来不会撞上 (3, 3) 这条分支；真撞上了，
    # 交给调用方保证方向正确、原样拒绝，比在这里猜一次更安全。
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions 形状异常: {pos.shape}，期望 (N, 3)")

    rec_idx = [int(i) for i in rec_idx]
    lig_idx = [int(i) for i in lig_idx]
    if len(rec_idx) != 3 or len(lig_idx) != 3:
        raise ValueError("Boresch 平衡值计算需要 3 个受体锚点和 3 个配体锚点")
    if not np.all(np.isfinite(pos)):
        raise ValueError("positions 包含 NaN/Inf，拒绝更新 Boresch 平衡几何")

    r_coords = pos[rec_idx]
    l_coords = pos[lig_idx]
    if not np.all(np.isfinite(r_coords)) or not np.all(np.isfinite(l_coords)):
        raise ValueError("Boresch 锚点坐标包含 NaN/Inf，拒绝更新平衡几何")

    # 受体锚点顺序必须在估算阶段确定后保持锁定，绝不能按瞬时几何动态重排。
    H0, H1, H2, L0, L1, L2 = unwrap_boresch_anchors_nm(
        r_coords, l_coords, box_vectors
    )

    def dist(a, b): return np.linalg.norm(a - b)
    def angle(a, b, c):
        ba, bc = a - b, c - b
        norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6: return np.pi / 2
        cos_val = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc + 1e-10), -1.0, 1.0)
        return np.arccos(cos_val)
    # 🚨 符号约定必须与 OpenMM `dihedral()` 一致。这里曾是一份返回 **−φ** 的手写
    # 副本，它把 2026-07-29 那次 attachment 腿的参考几何整体镜像掉了
    # （ΔG(A′→A) 5.5 → 98.8 kJ/mol）。详见 `boresch_dihedral_rad` 的 docstring。
    dihedral = boresch_dihedral_rad

    r0 = dist(H0, L0)
    if not np.isfinite(r0):
        raise ValueError("Boresch r0 为 NaN/Inf，拒绝更新平衡几何")
    if r0 < 0.3 or r0 > 2.0:
        raise RuntimeError(
            f"Boresch r0={r0*10:.2f}Å 超出合理范围 [3, 20]Å；"
            "拒绝使用默认几何继续生产 ABFE。"
        )

    thetaA0 = angle(H1, H0, L0)
    thetaB0 = angle(H0, L0, L1)
    phiA0 = dihedral(H2, H1, H0, L0)
    phiB0 = dihedral(H1, H0, L0, L1)
    phiC0 = dihedral(H0, L0, L1, L2)
    geom = np.array([r0, thetaA0, thetaB0, phiA0, phiB0, phiC0], dtype=float)
    if not np.all(np.isfinite(geom)):
        raise ValueError(f"Boresch 平衡几何包含 NaN/Inf: {geom.tolist()}")

    return {
        "r0": float(r0),
        "thetaA0": float(thetaA0),  # H1-H0-L0
        "thetaB0": float(thetaB0),  # H0-L0-L1
        "phiA0": float(phiA0),      # H2-H1-H0-L0
        "phiB0": float(phiB0),      # H1-H0-L0-L1
        "phiC0": float(phiC0),      # H0-L0-L1-L2
    }


def assess_boresch_harmonicity(traj, receptor_indices, ligand_indices) -> Dict:
    """Model-free check of the harmonic/Gaussian assumption behind
    `calculate_boresch_analytical_correction`, computed directly from the
    trajectory that locked the anchor choice.

    Runs unconditionally for every Boresch source (auto/orb_simple/simple/
    fluctuation). The retired ML scanner in
    `dexp_退役.py::OrbScanner.scan_boresch_1d_pes` only implements the
    r-coordinate and was never called from a production pipeline. This uses
    the same distance/angle/dihedral convention as
    `calc_boresch_from_last_frame` (receptor_indices[0] nearest ligand) and
    reuses `GeometricRestraintEstimator._fluctuation_diagnostics` so the same
    skew/kurtosis/under-sampling criteria apply regardless of which estimator
    produced the anchors.
    """
    if not HAS_MDTRAJ:
        return {"ok": False, "reason": "mdtraj_unavailable"}

    rec_idx = [int(i) for i in receptor_indices]
    lig_idx = [int(i) for i in ligand_indices]
    if len(rec_idx) != 3 or len(lig_idx) != 3:
        return {"ok": False, "reason": "invalid_anchor_index_count"}
    if len(traj) < 4:
        return {"ok": False, "reason": "too_few_trajectory_frames"}

    dist_idx = [[rec_idx[0], lig_idx[0]]]
    angleA_idx = [[rec_idx[1], rec_idx[0], lig_idx[0]]]
    angleB_idx = [[rec_idx[0], lig_idx[0], lig_idx[1]]]
    dihA_idx = [[rec_idx[2], rec_idx[1], rec_idx[0], lig_idx[0]]]
    dihB_idx = [[rec_idx[1], rec_idx[0], lig_idx[0], lig_idx[1]]]
    dihC_idx = [[rec_idx[0], lig_idx[0], lig_idx[1], lig_idx[2]]]

    r = mdtraj.compute_distances(traj, dist_idx)[:, 0]
    thetaA = mdtraj.compute_angles(traj, angleA_idx)[:, 0]
    thetaB = mdtraj.compute_angles(traj, angleB_idx)[:, 0]
    phiA = mdtraj.compute_dihedrals(traj, dihA_idx)[:, 0]
    phiB = mdtraj.compute_dihedrals(traj, dihB_idx)[:, 0]
    phiC = mdtraj.compute_dihedrals(traj, dihC_idx)[:, 0]

    def _unwrap(vals):
        vals = np.asarray(vals, dtype=float).copy()
        for t in range(1, len(vals)):
            diff = vals[t] - vals[t - 1]
            vals[t] -= 2 * np.pi * np.round(diff / (2 * np.pi))
        mean_val = float(np.mean(vals)) if len(vals) else 0.0
        vals -= 2 * np.pi * np.round(mean_val / (2 * np.pi))
        return vals

    coords = {
        "r": r,
        "thetaA": thetaA,
        "thetaB": thetaB,
        "phiA": _unwrap(phiA),
        "phiB": _unwrap(phiB),
        "phiC": _unwrap(phiC),
    }
    fluctuation_diagnostics = [
        GeometricRestraintEstimator._fluctuation_diagnostics(vals, name)
        for name, vals in coords.items()
    ]
    n_bad = sum(1 for item in fluctuation_diagnostics if not item.get("ok", False))
    harmonic_ok = n_bad == 0

    result = {
        "ok": True,
        "method": "trajectory_fluctuation_v1",
        "n_frames_used": int(len(r)),
        "receptor_indices": rec_idx,
        "ligand_indices": lig_idx,
        "fluctuation_distribution": fluctuation_diagnostics,
        "n_non_gaussian_or_under_sampled_terms": int(n_bad),
        "harmonic_assumption_ok": bool(harmonic_ok),
        "warning": "",
    }
    if not harmonic_ok:
        result["warning"] = (
            f"{n_bad}/6 Boresch restraint coordinates show non-Gaussian or under-sampled "
            "fluctuations over the trajectory used to lock this restraint. "
            "calculate_boresch_analytical_correction assumes independent, approximately "
            "Gaussian coordinates; its result may be biased for this anchor choice. "
            "Consider a different --boresch-select candidate, longer pre-equilibration, "
            "or a numerical (non-analytical) release free-energy estimate."
        )
    return result


#=============================================================================
# 修复 3: 基于 RMSF 的自动化锚点选择器 (Automatic Anchor Selection)
#=============================================================================
def auto_select_boresch_anchors_rmsf(
    traj_path: str, top_path: str, ligand_resname: str,
    temperature: float = 300.0, rmsf_threshold_nm: float = 0.12,
    r0_range_angstrom: Tuple[float, float] = (5.0, 10.0),
    output_path: Optional[str] = None
) -> Dict:
    import mdtraj as md

    traj = md.load(traj_path, top=top_path)
    top = traj.topology

    align_atoms = top.select("protein and backbone")
    if len(align_atoms) >= 3:
        traj.superpose(traj, 0, atom_indices=align_atoms)

    ca_atoms = top.select("protein and name CA")
    if len(ca_atoms) < 3:
        raise RuntimeError("受体 CA 原子不足3个，无法构建 Boresch 限制")

    ca_rmsf = md.rmsf(traj, traj, 0, atom_indices=ca_atoms)
    rmsf_by_atom = {int(atom): float(value) for atom, value in zip(ca_atoms, ca_rmsf)}
    rigid_cas = [int(atom) for atom, value in zip(ca_atoms, ca_rmsf) if value <= rmsf_threshold_nm]
    if len(rigid_cas) < 3:
        order = np.argsort(ca_rmsf)
        rigid_cas = [int(ca_atoms[i]) for i in order[: min(12, len(order))]]
    else:
        rigid_cas = rigid_cas[: min(12, len(rigid_cas))]
    
    # 🔑 修复 1：枚举配体重原子三元组，而非死板取前 3 个
    lig_heavy = top.select(f"resname {ligand_resname} and not element H")
    if len(lig_heavy) < 3:
        raise RuntimeError("配体重原子不足3个，无法构建 Boresch 限制")
    
    # 按原子质量排序，优先选择重原子作为锚点候选
    lig_masses = np.array([top.atom(i).element.mass for i in lig_heavy])
    sorted_lig_idx = np.argsort(lig_masses)[::-1]
    top_lig_candidates = [int(lig_heavy[i]) for i in sorted_lig_idx[:min(10, len(sorted_lig_idx))]]
    
    best_score, best_config = -np.inf, None
    for rec_combo in combinations(rigid_cas, 3):
        for lig_combo in combinations(top_lig_candidates, 3):
            r_coords = traj.xyz[0, list(rec_combo)]
            l_coords = traj.xyz[0, list(lig_combo)]
            
            r0 = np.linalg.norm(r_coords[0] - l_coords[0]) * 10.0
            if not (r0_range_angstrom[0] <= r0 <= r0_range_angstrom[1]):
                continue
                
            def angle(a, b, c):
                v1, v2 = a - b, c - b
                return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10), -1, 1)))
            
            thA = angle(r_coords[1], r_coords[0], l_coords[0])
            thB = angle(r_coords[0], l_coords[0], l_coords[1])  # 🔑 修复 2：增加 thB 计算
            
            # 🔑 修复 3：严格同时拦截 thA 和 thB 的奇点
            if not (45 <= thA <= 135) or not (45 <= thB <= 135):
                continue
                
            # 锚点内部距离检查 (防止共线)
            if np.linalg.norm(r_coords[0]-r_coords[1]) < 0.3 or np.linalg.norm(l_coords[0]-l_coords[1]) < 0.2:
                continue
                
            rec_rmsf_mean = float(np.mean([rmsf_by_atom.get(int(i), rmsf_threshold_nm) for i in rec_combo]))
            score = 100 - abs(r0 - 7.5) * 5 - rec_rmsf_mean * 500
            if score > best_score:
                best_score = score
                best_config = {
                    "receptor_indices": list(rec_combo),
                    "ligand_indices": [int(i) for i in lig_combo],
                    "equilibrium_r0": float(r0 * 0.1),
                    "rmsf_mean": rec_rmsf_mean
                }
                
    if best_config is None:
        raise RuntimeError("未找到符合几何 (thA/thB) 与稳定性条件的锚点组合")
        
    print(f"[OK] 自动锚点选择完成: r0={best_config['equilibrium_r0']:.2f}nm, RMSF={best_config['rmsf_mean']:.3f}nm")
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2, cls=NumpyEncoder)
    return best_config


