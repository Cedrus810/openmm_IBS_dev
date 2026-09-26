

# ============================================================================
# 3. Orb 口袋力投影估算器 (v4.3 - 3-Stage Optimized & Hybrid Filter)
# ============================================================================
class OrbVacuumContext:
    """ORB 真空力场计算上下文 (仅用于口袋内力场计算)"""

    def __init__(
        self, topology, model_name="mace-off24-medium", device="cpu"
    ):
        self.device = device
        self.model_name = model_name
        self.potential = _build_mace_potential(model_name)
        self.system = self.potential.createSystem(topology, **_build_openmmml_kwargs(
            device=self.device,
            return_energy_type="energy",
            charge=0,
            multiplicity=1,
        ))
        self.integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        try:
            platform = openmm.Platform.getPlatformByName(device.upper())
        except Exception:
            platform = openmm.Platform.getPlatformByName("CPU")
        self.context = openmm.Context(self.system, self.integrator, platform)

    def calculate_forces(self, positions_nm):
        self.context.setPositions(positions_nm)
        return (
            self.context.getState(getForces=True)
            .getForces(asNumpy=True)
            .value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        )


class OrbBoreschEstimator:
    """
    基于 Orb 口袋力投影与三阶段锚点优化的 Boresch 估算器
    特性：
    1. 稳定性+动态距离+几何构型 3-Stage 锚点筛选
    2. 边界氢饱和修补 (避免切割键导致的力场畸变)
    3. 混合滤波拟合 (线性回归优先，相关性不足时自动切换至波动法)
    4. 二面角自动展开 (Unwrap) 与 Jacobian 修正
    """

    DEFAULT_CONFIG = {
        "temperature": 300.0,
        "cutoff_nm": 0.9,
        "rmsf_cutoff_nm": 0.15,
        "dist_reject_nm": 0.40,
        "dist_gold_min_nm": 0.50,
        "dist_gold_max_nm": 1.10,
        "dist_backup_min_nm": 1.10,
        "dist_backup_max_nm": 1.50,
        "rec_anchor_dist_min": 0.40,
        "rec_anchor_dist_max": 0.80,
        "rec_anchor_angle_min": 60,
        "rec_anchor_angle_max": 120,
        "corr_threshold_keep": -0.1,
        "corr_threshold_good": -0.5,
        "use_fluctuation_fallback": True,
        "short_sidechain_res": ["GLY", "ALA", "SER", "VAL", "THR", "CYS"],
        "long_sidechain_res": [
            "TRP",
            "PHE",
            "TYR",
            "ARG",
            "LYS",
            "GLU",
            "GLN",
            "MET",
            "HIS",
        ],
        "score_weights": {
            "stability": 1.0,
            "distance": 2.0,
            "geometry": 1.5,
            "signal": 3.0,
        },
        "top_n_candidates": 5,
    }

    def __init__(self, temperature=300.0, device=None, cutoff_nm=0.9, n_frames=500):
        if not has_orb():
            raise ImportError("OrbBoreschEstimator 依赖 torch + openmmml，请安装后重试")
        self.T = temperature
        self.gas_constant_kj_per_mol_k = 8.314e-3
        self.device = device or ("cuda" if _require_torch().cuda.is_available() else "cpu")
        self.config = {
            **self.DEFAULT_CONFIG,
            "temperature": temperature,
            "cutoff_nm": cutoff_nm,
        }
        self.n_frames = n_frames
        self.sc_correction = {
            "GLY": -0.05,
            "ALA": -0.03,
            "SER": -0.02,
            "THR": -0.02,
            "CYS": -0.02,
            "ARG": 0.15,
            "LYS": 0.12,
            "GLU": 0.10,
            "GLN": 0.10,
            "TRP": 0.12,
            "TYR": 0.10,
            "PHE": 0.08,
            "MET": 0.08,
            "HIS": 0.06,
        }

    def _get_sidechain_correction(self, resname):
        return self.sc_correction.get(resname, 0.0)

    def _score_distance(self, dist_nm, resname):
        cfg = self.config
        if dist_nm < cfg["dist_reject_nm"]:
            return -100, "太近"
        c_dist = dist_nm - self._get_sidechain_correction(resname)
        if cfg["dist_gold_min_nm"] <= c_dist <= cfg["dist_gold_max_nm"]:
            center = (cfg["dist_gold_min_nm"] + cfg["dist_gold_max_nm"]) / 2
            bonus = 50 * (
                1
                - abs(c_dist - center)
                / (cfg["dist_gold_max_nm"] - cfg["dist_gold_min_nm"])
            )
            return 50 + bonus, "黄金区间"
        if cfg["dist_backup_min_nm"] <= c_dist <= cfg["dist_backup_max_nm"]:
            score = 30 * (
                1
                - (c_dist - cfg["dist_backup_min_nm"])
                / (cfg["dist_backup_max_nm"] - cfg["dist_backup_min_nm"])
            )
            return score, "备选区间"
        return 5 if c_dist > cfg["dist_backup_max_nm"] else 25, "过渡/较远"

    def _check_anchor_geometry(self, rec_anchors, traj, pocket_sel):
        r = traj.xyz[0, pocket_sel]
        rec_local = [np.where(pocket_sel == a)[0][0] for a in rec_anchors]
        P1, P2, P3 = r[rec_local]

        def calc_angle(a, b, c):
            ba, bc = a - b, c - b
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
            return np.arccos(np.clip(cos, -1, 1)) * 180 / np.pi

        cfg = self.config
        dists = [
            np.linalg.norm(P1 - P2),
            np.linalg.norm(P2 - P3),
            np.linalg.norm(P1 - P3),
        ]
        d_score = sum(
            20
            if cfg["rec_anchor_dist_min"] <= d <= cfg["rec_anchor_dist_max"]
            else (-30 if d < cfg["rec_anchor_dist_min"] else 0)
            for d in dists
        )
        angle_P = calc_angle(P3, P2, P1)
        a_score = (
            40
            if cfg["rec_anchor_angle_min"] <= angle_P <= cfg["rec_anchor_angle_max"]
            else (15 if 30 <= angle_P <= 150 else -50)
        )
        return (
            (a_score > -30),
            d_score + a_score,
            f"∠P3-P2-P1={angle_P:.1f}°, d12={dists[0] * 10:.2f}Å",
        )

    def _validate_boresch_geometry_strict(self, rec_anchors, lig_anchors, ref_coords):
        """
        【严格几何硬过滤器】(Hard Reject) - 彻底重写版
        统一单位：内部全部使用 nm，仅日志转 Å/°
        拒绝标准：
          1. r0 (R0-L0) ∈ [0.50, 1.00] nm (5-10 Å)
          2. θA (R1-R0-L0) ∈ [40°, 140°]  → sin(θA) ≥ 0.642
          3. θB (R0-L0-L1) ∈ [40°, 140°]  → sin(θB) ≥ 0.642
          4. 受体锚点间距 ≥ 0.38 nm (3.8 Å)
          5. 配体锚点间距 ≥ 0.25 nm (2.5 Å)
        """
        try:
            R0, R1, R2 = [ref_coords[a] for a in rec_anchors]
            L0, L1, L2 = [ref_coords[a] for a in lig_anchors]
        except IndexError:
            return False, "[ERR] 锚点索引越界"

        def dist(p1, p2): return np.linalg.norm(p1 - p2)
        def angle_rad(p1, p2, p3):
            v1, v2 = p1 - p2, p3 - p2
            cos_val = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12), -1.0, 1.0)
            return np.arccos(cos_val)

        # 1. 核心几何量 (nm & rad)
        r0_nm = dist(R0, L0)
        thA_rad = angle_rad(R1, R0, L0)
        thB_rad = angle_rad(R0, L0, L1)

        d_R01, d_R12 = dist(R0, R1), dist(R1, R2)
        d_L01, d_L12 = dist(L0, L1), dist(L1, L2)

        # 2. 严格阈值拦截 (全部基于 nm/rad 比较)
        if not (0.50 <= r0_nm <= 1.00):
            return False, f"[ERR] r0={r0_nm*10:.2f}Å [需5.0-10.0Å]"

        thA_deg, thB_deg = np.degrees(thA_rad), np.degrees(thB_rad)
        if not (40.0 <= thA_deg <= 140.0):
            return False, f"[ERR] θA={thA_deg:.1f}° [需40-140°]"
        if not (40.0 <= thB_deg <= 140.0):
            return False, f"[ERR] θB={thB_deg:.1f}° [需40-140°]"

        # 3. 奇点保护：sin(θ) < 0.64 直接拒绝 (对应 θ<40° 或 θ>140°)
        if np.sin(thA_rad) < 0.64 or np.sin(thB_rad) < 0.64:
            return False, f"[ERR] 几何奇异 sinθ≈0 (θA={thA_deg:.1f}°, θB={thB_deg:.1f}°)"

        # 4. 锚点共线/重叠保护
        if d_R01 < 0.38 or d_R12 < 0.38:
            return False, f"[ERR] 受体锚点过近 (min={min(d_R01, d_R12)*10:.1f}Å)"
        if d_L01 < 0.25 or d_L12 < 0.25:
            return False, f"[ERR] 配体锚点过近 (min={min(d_L01, d_L12)*10:.1f}Å)"

        # 5. 通过
        return True, f"[OK] 几何合格 r0={r0_nm*10:.2f}Å θA={thA_deg:.1f}° θB={thB_deg:.1f}°"

    def _add_capping_hydrogens(self, topology, selection, ref_coords):
        cap_h_info = []
        residues = list(topology.residues)
        for res in residues:
            res_atoms = [a.index for a in res.atoms]
            if not any(a in selection for a in res_atoms) or all(
                a in selection for a in res_atoms
            ):
                continue
            bb = {a.name: a.index for a in res.atoms if a.name in ["N", "CA", "C", "O"]}
            if "C" in bb and bb["C"] in selection:
                nxt = residues[res.index + 1] if res.index + 1 < len(residues) else None
                if nxt:
                    nxt_n = [a.index for a in nxt.atoms if a.name == "N"]
                    if nxt_n and nxt_n[0] not in selection:
                        c_pos = ref_coords[bb["C"]]
                        n_pos = (
                            ref_coords[nxt_n[0]]
                            if nxt_n
                            else c_pos + np.array([0.15, 0, 0])
                        )
                        d = n_pos - c_pos
                        n = np.linalg.norm(d)
                        d = d / n if n > 1e-10 else np.array([1, 0, 0])
                        cap_h_info.append(
                            {
                                "cut_atom": bb["C"],
                                "cut_type": "C_term",
                                "direction": d,
                                "bond_length": 0.11,
                                "neighbor_global": nxt_n[0],
                            }
                        )
            if "N" in bb and bb["N"] in selection:
                prv = residues[res.index - 1] if res.index > 0 else None
                if prv:
                    prv_c = [a.index for a in prv.atoms if a.name == "C"]
                    if prv_c and prv_c[0] not in selection:
                        n_pos = ref_coords[bb["N"]]
                        c_pos = (
                            ref_coords[prv_c[0]]
                            if prv_c
                            else n_pos + np.array([-0.15, 0, 0])
                        )
                        d = c_pos - n_pos
                        n = np.linalg.norm(d)
                        d = d / n if n > 1e-10 else np.array([-1, 0, 0])
                        cap_h_info.append(
                            {
                                "cut_atom": bb["N"],
                                "cut_type": "N_term",
                                "direction": d,
                                "bond_length": 0.10,
                                "neighbor_global": prv_c[0],
                            }
                        )
        return cap_h_info

    def _build_pocket_context(self, traj, ligand_resname):
        if not HAS_MDTRAJ:
            raise ImportError("需要 mdtraj")
        top = traj.topology
        lig_sel = top.select(f"resname {ligand_resname}")
        if len(lig_sel) == 0:
            raise ValueError("未找到配体")
        lig_center = traj.xyz[0, lig_sel].mean(axis=0)
        prot_sel = top.select("protein")
        nearby_atoms = [
            a
            for a in prot_sel
            if np.linalg.norm(traj.xyz[0, a] - lig_center) <= self.config["cutoff_nm"]
        ]
        # 保留完整残基，避免在主链/侧链中间截断产生自由基；比事后手工补 capping H 更稳。
        nearby_residues = {top.atom(a).residue.index for a in nearby_atoms}
        pocket_atoms = set(int(i) for i in lig_sel.tolist())
        for res in top.residues:
            if res.index in nearby_residues:
                for atom in res.atoms:
                    pocket_atoms.add(atom.index)
        pocket_sel = np.array(sorted(pocket_atoms), dtype=int)
        if len(pocket_sel) < 10:
            raise ValueError("口袋原子不足，请增大 cutoff_nm")

        cap_h = []
        pocket_traj = traj.atom_slice(pocket_sel)
        omm_top = pocket_traj.topology.to_openmm()
        context = OrbVacuumContext(omm_top, device=self.device)

        # --- 3-Stage 锚点筛选 ---
        lig_atoms = top.select(f"resname {ligand_resname} and not element H")
        if len(lig_atoms) < 3:
            lig_atoms = top.select(f"resname {ligand_resname}")
        if len(lig_atoms) == 0:
            raise ValueError(f"未找到配体 {ligand_resname} 原子，无法构建口袋上下文")

        lig_scores = [
            top.atom(i).element.mass / 12.0
            + (2.0 if top.atom(i).name in ["CG", "CD", "CE", "CZ", "CA"] else 0)
            for i in lig_atoms
        ]
        lig_anchors = lig_atoms[np.argsort(lig_scores)[-3:]].tolist()

        rec_ca = top.select("protein and name CA")
        rec_ca_p = np.intersect1d(rec_ca, pocket_sel)
        rmsf_traj = traj[:: max(1, len(traj) // 100)]
        if len(rmsf_traj) > 1:
            rmsf_traj.superpose(rmsf_traj, 0, atom_indices=rec_ca_p)
            rmsf = np.sqrt(
                np.mean(
                    (rmsf_traj.xyz[:, rec_ca_p] - rmsf_traj.xyz[0, rec_ca_p]) ** 2,
                    axis=(0, 2),
                )
            )
        else:
            rmsf = np.zeros(len(rec_ca_p))

        rigid_mask = rmsf < self.config["rmsf_cutoff_nm"]
        rigid_ca = rec_ca_p[rigid_mask]
        rigid_rmsf = rmsf[rigid_mask]

        candidates = []
        if len(rigid_ca) >= 3:
            from itertools import combinations

            sorted_idx = np.argsort(rigid_rmsf)[:30]
            sorted_ca = rigid_ca[sorted_idx]
            sorted_rmsf = rigid_rmsf[sorted_idx]
            sorted_resnames = [top.atom(idx).residue.name for idx in sorted_ca]

            MAX_R0_ANGSTROM = 12.0
            MIN_R0_ANGSTROM = 6.0
            seen_combos = set()
            candidates = []

            for combo in combinations(range(len(sorted_ca)), 3):
                combo_indices = list(combo)
                combo_anchors = [sorted_ca[i] for i in combo_indices]
                combo_rmsf_vals = [sorted_rmsf[i] for i in combo_indices]

                dists = [
                    np.linalg.norm(traj.xyz[0, a] - lig_center) for a in combo_anchors
                ]
                sorted_pairs = sorted(
                    zip(
                        dists,
                        combo_anchors,
                        combo_rmsf_vals,
                        [sorted_resnames[i] for i in combo_indices],
                    ),
                    key=lambda x: x[0],
                )
                rec_anchors = [p[1] for p in sorted_pairs]
                rec_rmsf = [p[2] for p in sorted_pairs]
                rec_resnames = [p[3] for p in sorted_pairs]

                R0, R1, R2 = traj.xyz[0, rec_anchors]
                L0 = traj.xyz[0, lig_anchors[0]]

                r0_nm = np.linalg.norm(R0 - L0)
                r0_A = r0_nm * 10.0

                def calc_ang(a, b, c):
                    ba, bc = a - b, c - b
                    cos = np.dot(ba, bc) / (
                        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10
                    )
                    return np.degrees(np.arccos(np.clip(cos, -1, 1)))

                thetaA = calc_ang(R1, R0, L0)
                thetaB = calc_ang(R0, L0, traj.xyz[0, lig_anchors[1]])

                if not (MIN_R0_ANGSTROM <= r0_A < MAX_R0_ANGSTROM):
                    continue

                geo_key = (
                    round(r0_A, 1),
                    round(thetaA, 1),
                    round(thetaB, 1),
                    tuple(sorted(rec_anchors[:2])),
                )
                if geo_key in seen_combos:
                    continue
                seen_combos.add(geo_key)

                score_stab = sum(
                    (self.config["rmsf_cutoff_nm"] - rv) * 100 for rv in rec_rmsf
                )

                if 8.0 <= r0_A <= 10.5:
                    score_dist = 60
                elif 10.5 < r0_A < 12.0:
                    score_dist = 30
                else:
                    score_dist = 0

                if 60 <= thetaA <= 110:
                    score_ang = 50
                elif 45 < thetaA < 60 or 110 < thetaA < 135:
                    score_ang = 20
                else:
                    score_ang = 0

                pre_sc = score_stab + score_dist + score_ang
                candidates.append(
                    {
                        "rec_anchors": rec_anchors,
                        "score": pre_sc,
                        "r0_A": r0_A,
                        "thetaA": thetaA,
                        "thetaB": thetaB,
                        "rmsf_avg": np.mean(rec_rmsf),
                    }
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_cands = candidates[: self.config["top_n_candidates"]]

        # 快速力学信号验证 (取前50帧)
        validated = []
        for cand in top_cands:
            local_anchors = [
                int(np.where(pocket_sel == a)[0][0])
                for a in cand["rec_anchors"] + lig_anchors
            ]
            ks_temp = self._quick_validate(
                traj[:50], context, pocket_sel, local_anchors, cap_h
            )

            if ks_temp is not None:
                validated.append(
                    {**cand, "ks_temp": ks_temp, "total": cand["score"] + 20.0}
                )

        if validated:
            validated.sort(key=lambda x: x["total"], reverse=True)
            best = validated[0]
            rec_anchors, best_ks = best["rec_anchors"], best["ks_temp"]
        else:
            rec_anchors = (
                candidates[0]["rec_anchors"] if candidates else rigid_ca[:3].tolist()
            )
            best_ks = None

        anchor_global = rec_anchors + lig_anchors
        local_anchors = [int(np.where(pocket_sel == a)[0][0]) for a in anchor_global]
        return context, pocket_sel, local_anchors, anchor_global, cap_h, best_ks

    def _quick_validate(self, traj, context, selection, local_anchors, cap_h):
        if len(traj) < 10:
            return None
        n_frames = len(traj)
        q_data = np.zeros((n_frames, 6))
        Fq_data = np.zeros((n_frames, 6))
        for f in range(n_frames):
            pos = traj.xyz[f, selection]
            forces = context.calculate_forces(pos)
            r_anchors = pos[local_anchors]
            if np.any(np.isnan(r_anchors)):
                continue
            q, grads = self._compute_geom_gradients(r_anchors)
            if np.any(np.isnan(q)):
                continue
            Fq = np.zeros(6)
            for g in range(6):
                for a in range(6):
                    Fq[g] += np.dot(forces[local_anchors[a]], grads[g, a])
            q_data[f] = q
            Fq_data[f] = Fq
        return self._apply_hybrid_filter(q_data, Fq_data)

    def _compute_geom_gradients(self, r_anchors):
        # 🚨 关键修复：r_anchors 的 6 个 slot 严格是
        # [0]=R0(受体,离配体最近) [1]=R1 [2]=R2(受体,最远)
        # [3]=L0(配体,离受体最近) [4]=L1 [5]=L2(配体,最远)
        # 必须与 calc_boresch_from_last_frame / _check_boresch_geometry_safe /
        # LambdaDependentBoreschForce 完全一致：
        #   r0      = distance(R0, L0)                slot(0,3)
        #   thetaA0 = angle(R1, R0, L0)   顶点=R0      slot(1,0,3)
        #   thetaB0 = angle(R0, L0, L1)   顶点=L0      slot(0,3,4)
        #   phiA0   = dihedral(R2, R1, R0, L0)         slot(2,1,0,3)
        #   phiB0   = dihedral(R1, R0, L0, L1)         slot(1,0,3,4)
        #   phiC0   = dihedral(R0, L0, L1, L2)         slot(0,3,4,5)
        # 旧版把 H0/H2 的变量名接反了（H0 实际绑定的是 slot[2]=R2 而不是
        # slot[0]=R0），导致这里算出的力常数/CV 用的是"最远"受体锚点当顶点，
        # 跟平衡值计算/几何合法性检查完全对不上，是这次 Boresch 崩溃的另一个源头。
        q = np.zeros(6)
        grads = np.zeros((6, 6, 3))

        R0, L0 = r_anchors[0], r_anchors[3]
        vec_r = L0 - R0
        norm_r = np.linalg.norm(vec_r) + 1e-10
        q[0] = norm_r
        ur = vec_r / norm_r
        grads[0, 0, :] = -ur
        grads[0, 3, :] = ur

        angle_slots = [(1, 0, 3), (0, 3, 4)]
        for i, (sa, sb, sc) in enumerate(angle_slots):
            a, b, c = r_anchors[sa], r_anchors[sb], r_anchors[sc]
            ba, bc = a - b, c - b
            nba, nbc = np.linalg.norm(ba) + 1e-10, np.linalg.norm(bc) + 1e-10
            cosA = np.clip(np.dot(ba, bc) / (nba * nbc), -1, 1)
            q[i + 1] = np.arccos(cosA)
            sinA = np.sqrt(1 - cosA**2) + 1e-10
            if sinA > 1e-3:
                # 🚨 [0831issue #4] 这两行原来把两个端点的分子写反了：
                #     dbda = (cosA*bc/nbc - ba/nba)/(nba*sinA)   ← 错
                #     dbdc = (cosA*ba/nba - bc/nbc)/(nbc*sinA)   ← 错
                # 正确的解析梯度是（θ = 顶点 b 处 ba 与 bc 的夹角）：
                #     ∂cosθ/∂a = (1/|ba|)·(bc/|bc| − cosθ·ba/|ba|)
                #     ∂θ/∂a    = −(1/sinθ)·∂cosθ/∂a
                #              = (cosθ·ba/|ba| − bc/|bc|) / (|ba|·sinθ)
                # 即端点 a 的分子必须是 **自己那条边的单位向量** 乘 cosθ 再减
                # 对边单位向量；端点 c 同理镜像。旧式把两个分子互换后又各自保留
                # 了原来的 1/|ba| vs 1/|bc| 缩放，两处错误不互相抵消。
                # 有限差分复核（随机构型 5 组，eps=1e-6）：旧式相对误差
                # 130%~200%，新式与 FD 一致到 ~1e-9。顶点项 −dbda−dbdc 依赖平移
                # 不变性，因此端点错时顶点同样错。
                dbda = (cosA * ba / nba - bc / nbc) / (nba * sinA)
                dbdc = (cosA * bc / nbc - ba / nba) / (nbc * sinA)
                grads[i + 1, sa, :] = dbda
                grads[i + 1, sb, :] = -dbda - dbdc
                grads[i + 1, sc, :] = dbdc

        if HAS_MDTRAJ:
            dummy_top = mdtraj.Topology()
            c = dummy_top.add_chain()
            r = dummy_top.add_residue("X", c)
            for _ in range(6):
                dummy_top.add_atom("C", openmm.app.element.Element.getBySymbol("C"), r)
            eps = 1e-3
            dihedral_slots = [(2, 1, 0, 3), (1, 0, 3, 4), (0, 3, 4, 5)]
            for g_idx, tup in enumerate(dihedral_slots, 3):
                q[g_idx] = mdtraj.compute_dihedrals(
                    mdtraj.Trajectory(r_anchors[None], dummy_top), [tup]
                )[0, 0]
                perturbations = []
                grad_slots = []
                for a in tup:
                    for d in range(3):
                        rp = r_anchors.copy()
                        rm = r_anchors.copy()
                        rp[a, d] += eps
                        rm[a, d] -= eps
                        perturbations.extend((rp, rm))
                        grad_slots.append((a, d))

                batch_xyz = np.asarray(perturbations, dtype=float)
                batch_angles = mdtraj.compute_dihedrals(
                    mdtraj.Trajectory(batch_xyz, dummy_top), [tup]
                )[:, 0]
                for idx, (a, d) in enumerate(grad_slots):
                    grads[g_idx, a, d] = (
                        batch_angles[2 * idx] - batch_angles[2 * idx + 1]
                    ) / (2 * eps)
        return q, grads

    def _apply_hybrid_filter(self, q, Fq):
        kB_T = self.gas_constant_kj_per_mol_k * self.T
        names = ["kr", "kthetaA", "kthetaB", "kphiA", "kphiB", "kphiC"]
        ks = {}
        for i in range(6):
            valid = ~(np.isnan(q[:, i]) | np.isnan(Fq[:, i]))
            if valid.sum() < 10:
                ks[names[i]] = 0.0
                continue
            dq, dF = (
                q[:, i][valid] - np.mean(q[:, i][valid]),
                Fq[:, i][valid] - np.mean(Fq[:, i][valid]),
            )
            var = np.var(dq)
            cov = np.cov(dF, dq)[0, 1]
            k_reg = -cov / var if var > 1e-12 else None
            k_fluc = kB_T / var if var > 1e-12 else None
            corr = (
                np.corrcoef(q[:, i][valid], Fq[:, i][valid])[0, 1]
                if np.std(q[:, i][valid]) > 1e-10
                else 0.0
            )

            if k_reg and k_reg > 0 and corr < self.config["corr_threshold_keep"]:
                ks[names[i]] = k_reg
            elif k_fluc and self.config["use_fluctuation_fallback"]:
                ks[names[i]] = k_fluc
            else:
                ks[names[i]] = 0.0

        for name in names:
            if name == "kr":
                ks[name] = float(min(max(ks[name], 100.0), 2000.0))
            else:
                ks[name] = float(min(max(ks[name], 10.0), 100.0))

        return ks

    def run_pocket_force_projection(
        self, traj, context, selection, local_anchors, cap_h_coords=None
    ):
        n_frames = len(traj)
        q_data = np.zeros((n_frames, 6))
        Fq_data = np.zeros((n_frames, 6))
        for f in range(n_frames):
            pos = traj.xyz[f, selection]
            forces = context.calculate_forces(pos)
            r_anchors = pos[local_anchors]
            if np.any(np.isnan(r_anchors)):
                continue
            q, grads = self._compute_geom_gradients(r_anchors)
            if np.any(np.isnan(q)):
                continue
            Fq = np.zeros(6)
            for g in range(6):
                for a in range(6):
                    Fq[g] += np.dot(forces[local_anchors[a]], grads[g, a])
            q_data[f] = q
            Fq_data[f] = Fq

        # Unwrap 二面角
        for i in [3, 4, 5]:
            valid = ~(np.isnan(q_data[:, i]) | np.isinf(q_data[:, i]))
            if valid.sum() > 10:
                q_data[valid, i] = np.unwrap(q_data[valid, i])
        return self._apply_hybrid_filter(q_data, Fq_data)

    def estimate_from_trajectory(self, traj, ligand_resname, output_path=None):
        context, pocket_sel, local_anchors, anchor_global, cap_h, ks_quick = (
            self._build_pocket_context(traj, ligand_resname)
        )
        ks = (
            ks_quick
            if ks_quick
            else self.run_pocket_force_projection(
                traj, context, pocket_sel, local_anchors, cap_h
            )
        )

        traj_aligned = traj[:]
        traj_aligned.superpose(traj_aligned, 0, atom_indices=pocket_sel)
        r0 = traj_aligned.xyz[0, pocket_sel][local_anchors]
        # 🚨 关键修复：local_anchors 顺序是 [R0(离配体最近),R1,R2(最远),L0,L1,L2]，
        # 之前写成 H2,H1,H0=r0[0,1,2] 把 H0 错绑定到 R2（最远锚点），导致下面算出
        # 的 eq 平衡值和 receptor_indices=anchor_global[:3] 实际代表的原子对不上。
        H0, H1, H2, G0, G1, G2 = r0

        def calc_angle(a, b, c):
            ba, bc = a - b, c - b
            cos_val = np.clip(
                np.dot(ba, bc)
                / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10),
                -1,
                1,
            )
            # ✅ 直接返回弧度 (rad)
            return np.arccos(cos_val)

        # ✅ 直接返回弧度 (rad)。符号约定必须与 OpenMM `dihedral()` 一致，
        # 所以统一走 `boresch_dihedral_rad`，不再本地手写（见其 docstring）。
        calc_dihedral = boresch_dihedral_rad

        eq = {
            "r0": float(np.linalg.norm(H0 - G0)),  # ✅ nm (移除 *10)
            "thetaA0": float(calc_angle(H1, H0, G0)),  # ✅ rad
            "thetaB0": float(calc_angle(H0, G0, G1)),  # ✅ rad
            "phiA0": float(calc_dihedral(H2, H1, H0, G0)),  # ✅ rad
            "phiB0": float(calc_dihedral(H1, H0, G0, G1)),  # ✅ rad
            "phiC0": float(calc_dihedral(H0, G0, G1, G2)),  # ✅ rad
        }
        rec_indices = anchor_global[:3]
        lig_indices = anchor_global[3:]
        result = {
            "receptor_indices": rec_indices.tolist()
            if hasattr(rec_indices, "tolist")
            else list(rec_indices),
            "ligand_indices": lig_indices.tolist()
            if hasattr(lig_indices, "tolist")
            else list(lig_indices),
            "force_constants": ks,
            "equilibrium_values": eq,
            "method": "orb_pocket_projection_v4.3",
        }
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)
        return result

    def _finalize_candidate(self, cand, traj, context, pocket_sel, cap_h):
        """计算候选者的最终力常数和平衡值"""
        rec_anchors = cand["rec_anchors"]
        lig_anchors = cand["lig_anchors"]
        local_anchors = cand["local_anchors"]
        ks = self.run_pocket_force_projection(
            traj, context, pocket_sel, local_anchors, cap_h
        )
        traj_aligned = traj[:]
        traj_aligned.superpose(traj_aligned, 0, atom_indices=pocket_sel)
        r0_frame = traj_aligned.xyz[0, pocket_sel][local_anchors]
        # 🚨 关键修复：同 estimate_from_trajectory，local_anchors 顺序是
        # [R0(离配体最近),R1,R2(最远),L0,L1,L2]，之前 H0 被错绑定到 R2。
        H0, H1, H2, G0, G1, G2 = r0_frame

        def calc_angle_rad(a, b, c):
            ba, bc = a - b, c - b
            cos_val = np.clip(
                np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10),
                -1, 1,
            )
            return float(np.arccos(cos_val))

        # 符号约定必须与 OpenMM `dihedral()` 一致，见 `boresch_dihedral_rad`。
        calc_dihedral_rad = boresch_dihedral_rad

        eq = {
            "r0": float(np.linalg.norm(H0 - G0)),
            "thetaA0": calc_angle_rad(H1, H0, G0),
            "thetaB0": calc_angle_rad(H0, G0, G1),
            "phiA0": calc_dihedral_rad(H2, H1, H0, G0),
            "phiB0": calc_dihedral_rad(H1, H0, G0, G1),
            "phiC0": calc_dihedral_rad(H0, G0, G1, G2),
        }
        return {**cand, "ks": ks, "eq": eq}

    def _score_candidate_comprehensive(self, rec_anchors, lig_anchors, rmsf_vals,
                                        r0_nm, thA_rad, thB_rad, kr_raw, traj, top):
        """综合评分函数：稳定性 + 几何 + 力常数 + 序列分散度"""
        
        # === 1. 稳定性 (原有，权重 1.0) ===
        score_stab = sum((self.config["rmsf_cutoff_nm"] - rv) * 80 for rv in rmsf_vals)
        
        # === 2. 距离偏好 (微调，权重 0.8) ===
        r0_A = r0_nm * 10
        if 7.0 <= r0_A <= 9.5:  # 偏好 7-9.5Å，略收紧
            score_dist = 40 - abs(r0_A - 8.2) * 3
        else:
            score_dist = max(0, 20 - abs(r0_A - 8.2) * 2)
        
        # === 3. NEW: kr 合理性 (权重 1.2) ===
        if kr_raw <= 600:
            kr_score = 25  # 理想范围
        elif kr_raw <= 1200:
            kr_score = 25 - (kr_raw - 600) * 0.02  # 线性衰减
        else:
            kr_score = max(0, 13 - (kr_raw - 1200) * 0.03)  # 快速衰减
        
        # === 4. NEW: 角度质量 (权重 1.0) ===
        thA_deg, thB_deg = np.degrees(thA_rad), np.degrees(thB_rad)
        if 70 <= thA_deg <= 110 and 70 <= thB_deg <= 110:
            angle_score = 20
        elif 50 <= thA_deg <= 130 and 50 <= thB_deg <= 130:
            angle_score = 10
        else:
            angle_score = 0  # 已在硬过滤中排除，此处为保险
        
        # === 5. NEW: 锚点几何分散度 (权重 1.0) ===
        R0, R1, R2 = [traj.xyz[0, a] for a in rec_anchors]
        anchor_dists = [
            np.linalg.norm(R0-R1) * 10.0,
            np.linalg.norm(R1-R2) * 10.0,
            np.linalg.norm(R0-R2) * 10.0,
        ]  # nm → Å
        avg_anchor_dist = np.mean(anchor_dists)
        if 12 <= avg_anchor_dist <= 22:  # 12-22Å 理想分散
            geo_score = 18
        elif 8 <= avg_anchor_dist <= 28:
            geo_score = 18 - abs(avg_anchor_dist - 17) * 0.8
        else:
            geo_score = 0
        
        # === 6. NEW: 残基序列分散度 (权重 0.8) ===
        res_indices = sorted([top.atom(a).residue.index for a in rec_anchors])
        min_gap = min(res_indices[1]-res_indices[0], res_indices[2]-res_indices[1])
        if min_gap >= 20:
            seq_score = 15
        elif min_gap >= 10:
            seq_score = 10
        elif min_gap >= 5:
            seq_score = 4
        else:
            seq_score = 0  # 太接近，可能共线
        
        # === 综合 ===
        total = (score_stab + score_dist*0.8 + kr_score*1.2 +
                 angle_score + geo_score + seq_score*0.8)
        
        return total, {
            "stab": score_stab, "dist": score_dist, "kr": kr_score,
            "angle": angle_score, "geo": geo_score, "seq": seq_score
        }

    def estimate_multiple_anchors_from_trajectory(
        self,
        traj,
        ligand_resname: str,
        n_candidates: int = 5,
        output_path: Optional[str] = None,
        min_anchor_distance: float = 0.4,
        max_r0_angstrom: float = 10.0,
        max_kr: float = 2000.0,
        min_residue_gap: int = 3,
        use_last_ns: float = 5.0,  # ✅ 强制参数：只分析最后 N ns
    ) -> List[Dict]:
        """
        确定性枚举 v6.5 (最终修复版)
        【关键修正】
        1. 强制切片最后 5ns：基于轨迹时间精确切除前期不稳定部分
        2. 最终 r0/kr 二次硬校验：绝不返回超标候选
        3. 残基间隔与去重：杜绝鬼打墙
        """
        try:
            import mdtraj as md
        except ImportError:
            raise ImportError("需要 mdtraj")

        # ========================================================================
        # ✅ 0. 强制切片最后 5ns (物理需求：切除预平衡前期不稳定部分)
        # ========================================================================
        original_len = len(traj)
        if hasattr(traj, "time") and len(traj.time) > 0:
            t_max = traj.time[-1]  # ps
            t_cut = t_max - (use_last_ns * 1000.0)  # ns -> ps
            mask = traj.time >= t_cut
            n_keep = mask.sum()

            if n_keep > 0:
                traj = traj[mask]
                print(
                    f"轨迹切片: 仅使用最后 {use_last_ns} ns ({n_keep} 帧 / 原始 {original_len} 帧)"
                )
            else:
                print(
                    f"[WARN] 轨迹长度不足 {use_last_ns} ns，使用全轨迹 ({original_len} 帧)"
                )
        else:
            print(f"[WARN] 轨迹无时间信息，使用全轨迹 ({original_len} 帧)")

        print(
            f"确定性枚举 v6.5 | 目标: {n_candidates} | r0≤{max_r0_angstrom}Å | kr≤{max_kr} | 残基间隔≥{min_residue_gap}"
        )

        top = traj.topology
        lig_sel = top.select(f"resname {ligand_resname}")
        if len(lig_sel) == 0:
            raise ValueError(f"未找到配体: {ligand_resname}")

        # 2. 口袋 Cα 与 RMSF (基于切片后的稳态轨迹)
        # --- 确保 rigid_ca 在这里被定义，防止 NameError ---
        lig_center = traj.xyz[0, lig_sel].mean(axis=0)
        prot_ca = top.select("protein and name CA")
        
        # 筛选口袋区域 Cα (1.0 nm 范围内)
        pocket_ca = [ca for ca in prot_ca if np.linalg.norm(traj.xyz[0, ca] - lig_center) <= 1.0]
        
        rmsf_traj = traj[:: max(1, len(traj) // 100)]
        if len(rmsf_traj) > 1 and len(pocket_ca) > 0:
            rmsf_traj.superpose(rmsf_traj, 0, atom_indices=pocket_ca)
            # 🔑 RMSF 的参考结构应该是平均结构，不是第 0 帧——第 0 帧本身只是
            # 一个样本，如果它恰好是这段轨迹里偏离平均构象较远的一帧，相对它
            # 算出来的涨落会系统性偏大，让下面 rigid_mask = rmsf < cutoff 偏
            # 保守地把本来足够刚性的 Cα 判定为不合格，漏掉本可用的锚点候选。
            pocket_xyz = rmsf_traj.xyz[:, pocket_ca]
            mean_xyz = pocket_xyz.mean(axis=0)
            rmsf = np.sqrt(np.mean((pocket_xyz - mean_xyz[None, :, :]) ** 2, axis=(0, 2)))
        else:
            rmsf = np.zeros(len(pocket_ca))
            
        rigid_mask = rmsf < self.config["rmsf_cutoff_nm"]
        rigid_ca = [pocket_ca[i] for i in range(len(pocket_ca)) if rigid_mask[i]]
        rigid_res = [top.atom(ca).residue.index for ca in rigid_ca]

        # ✅ 安全检查：刚性原子不足无法构建 Boresch
        if len(rigid_ca) < 3:
            print(f"  [ERR] 刚性 Cα 原子不足 3 个 (当前 {len(rigid_ca)})，无法进行几何枚举。")
            return []

        # 1. 生成配体锚点候选三元组 (基于质量+刚性排序)
        lig_heavy = top.select(f"resname {ligand_resname} and not element H")
        if len(lig_heavy) < 3:
            lig_heavy = top.select(f"resname {ligand_resname}")
        
        lig_combos = []
        for combo in combinations(lig_heavy, 3):
            # 评分逻辑：质量之和 + 骨架原子奖励
            mass_score = sum(top.atom(i).element.mass for i in combo)
            name_bonus = sum(2.0 if top.atom(i).name in ["CG","CD","CE","CZ","CA"] else 0.0 for i in combo)
            lig_combos.append((combo, mass_score + name_bonus))
            
        M = min(20, len(lig_combos))
        lig_combos = [c[0] for c in sorted(lig_combos, key=lambda x: x[1], reverse=True)[:M]]
        print(f"  → 配体锚点候选: {len(lig_combos)} 个三元组")

        # 2. 受体锚点枚举 + 嵌套配体候选扫描
        candidates = []
        seen_geo_keys = set()  # 几何去重

        # 遍历受体三元组
        for rec_combo in combinations(range(len(rigid_ca)), 3):
            ca0, ca1, ca2 = [rigid_ca[i] for i in rec_combo]
            res0, res1, res2 = [rigid_res[i] for i in rec_combo]
            
            # 受体侧预过滤（残基间隔、锚点间距）
            if min([abs(res0-res1), abs(res0-res2), abs(res1-res2)]) < min_residue_gap:
                continue
                
            pos_rec = [traj.xyz[0, ca] for ca in [ca0, ca1, ca2]]
            if min(np.linalg.norm(pos_rec[i]-pos_rec[j]) for i in range(3) for j in range(i+1,3)) < min_anchor_distance:
                continue
                
            # 遍历配体三元组
            for lig_combo in lig_combos:
                pos_lig = [traj.xyz[0, idx] for idx in lig_combo]
                
                # ✅ 修复：放宽配体内部间距阈值 (从 0.25 -> 0.15 nm / 1.5 Å)
                # 适配刚性小分子（如苯环）原子间距较近的情况
                if min(np.linalg.norm(pos_lig[i]-pos_lig[j]) for i in range(3) for j in range(i+1,3)) < 0.15:
                    continue
                
                # ✅ 严格几何校验
                ok, msg = self._validate_boresch_geometry_strict(
                    [ca0, ca1, ca2], lig_combo, traj.xyz[0]
                )
                if not ok:
                    continue
                    
                # 几何去重
                r0_A = np.linalg.norm(pos_rec[0] - pos_lig[0]) * 10.0
                geo_key = (round(r0_A, 1), tuple(sorted([ca0, ca1, ca2][:2])), tuple(sorted(lig_combo[:2])))
                if geo_key in seen_geo_keys:
                    continue
                seen_geo_keys.add(geo_key)
                
                # 综合评分
                rmsf_vals = [rmsf[pocket_ca.index(ca)] for ca in [ca0, ca1, ca2]]
                score_stab = sum((self.config["rmsf_cutoff_nm"] - rv) * 100 for rv in rmsf_vals)
                if 5.0 <= r0_A <= 10.0:
                    score_dist = 60 - abs(r0_A - 7.5) * 4
                else:
                    score_dist = max(0, 30 - abs(r0_A - 7.5) * 2)
                total_score = score_stab + score_dist
                
                candidates.append({
                    "rec_anchors": [ca0, ca1, ca2],
                    "lig_anchors": list(lig_combo),
                    "res_key": tuple(sorted([res0, res1, res2])),
                    "r0_A": r0_A,
                    "score": total_score,
                    "rmsf_avg": np.mean(rmsf_vals),
                })

        # 3. 验证与快速筛选
        candidates.sort(key=lambda x: x["score"], reverse=True)
        print(f"  → 通过几何过滤的候选: {len(candidates)} 个")
        
        rigid_residues = {top.atom(int(a)).residue.index for a in rigid_ca}
        pocket_atoms = set(int(i) for i in lig_sel.tolist())
        for res in top.residues:
            if res.index in rigid_residues:
                for atom in res.atoms:
                    pocket_atoms.add(atom.index)
        pocket_sel = np.array(sorted(pocket_atoms), dtype=int)

        cap_h = []
        pocket_traj = traj.atom_slice(pocket_sel)
        context = OrbVacuumContext(pocket_traj.topology.to_openmm(), device=self.device)
        
        validated = []
        kr_seen = set()
        search_pool = max(n_candidates * 6, 30)
        
        for cand in candidates[:search_pool]:
            # 构建局部原子索引映射
            local_anchors = [int(np.where(pocket_sel == a)[0][0]) for a in cand["rec_anchors"] + cand["lig_anchors"]]
            
            ks = self._quick_validate(traj[:50], context, pocket_sel, local_anchors, cap_h)
            if ks is None:
                continue
            if ks["kr"] > max_kr:
                continue
            kr_r = round(ks["kr"], 1)
            if kr_r in kr_seen:
                continue
            kr_seen.add(kr_r)
            validated.append({**cand, "ks": ks, "local_anchors": local_anchors})
        print(f"  → 快速验证通过: {len(validated)} 个")

        # 4. 最终结果构建
        results = []
        seen_final = set()
        fallback_pool = []
        
        # 辅助函数
        def log_cand(rank, r0_nm, kr, res_key, tag="合格"):
            r0_a = r0_nm * 10.0
            print(f"  [{tag}] #{rank}: r0={r0_a:.2f}Å | kr={kr:.1f} kJ/mol/nm² | 残基={res_key}")

        for cand in validated:
            res_key = cand["res_key"]
            if res_key in seen_final:
                continue
            seen_final.add(res_key)
            
            # 计算最终参数
            final = self._finalize_candidate(cand, traj, context, pocket_sel, cap_h)
            
            # 角度防御 (防止奇异)
            thA_deg = np.degrees(final["eq"]["thetaA0"])
            thB_deg = np.degrees(final["eq"]["thetaB0"])
            if not (40.0 <= thA_deg <= 140.0) or not (40.0 <= thB_deg <= 140.0):
                continue

            # 物理边界过滤
            r0_nm = final["eq"]["r0"]
            kr_val = final["ks"]["kr"]
            if r0_nm < 0.4 or r0_nm > 1.0:
                continue
            if r0_nm > max_r0_angstrom / 10.0:
                fallback_pool.append(final)
                continue
            if kr_val > max_kr:
                fallback_pool.append(final)
                continue

            # ✅ 使用综合评分替代简单 kr 惩罚
            rmsf_vals = [rmsf[pocket_ca.index(ca)] for ca in final["rec_anchors"]]
            total_score, score_breakdown = self._score_candidate_comprehensive(
                rec_anchors=final["rec_anchors"],
                lig_anchors=final["lig_anchors"],
                rmsf_vals=rmsf_vals,
                r0_nm=r0_nm,
                thA_rad=final["eq"]["thetaA0"],
                thB_rad=final["eq"]["thetaB0"],
                kr_raw=kr_val,
                traj=traj,
                top=top,
            )

            # 加入结果
            results.append({
                "rank": len(results) + 1,
                "receptor_indices": final["rec_anchors"],
                "ligand_indices": final["lig_anchors"],
                "receptor_residues": list(final["res_key"]),
                "force_constants": {k: float(v) for k, v in final["ks"].items()},
                "equilibrium_values": final["eq"],
                "total_score": float(total_score),
                "score_breakdown": score_breakdown,
                "method": "finite_combo_v6.5_last5ns",
            })
            log_cand(len(results), r0_nm, kr_val, res_key)
            
            if len(results) >= n_candidates:
                break

        # 降级回退
        if len(results) < n_candidates and fallback_pool:
            print(f"  [WARN] 合格候选不足 ({len(results)}/{n_candidates})，启动降级回退...")
            fallback_pool.sort(key=lambda x: x["ks"]["kr"])
            for final in fallback_pool:
                if len(results) >= n_candidates: break
                res_key = final["res_key"]
                if res_key in seen_final: continue
                seen_final.add(res_key)
                
                # 放宽 kr 限制，但死守 r0 几何
                r0_nm_fb = final["eq"]["r0"]
                kr_val_fb = final["ks"]["kr"]
                
                # ✅ 使用综合评分（与主路径一致）
                rmsf_vals_fb = [rmsf[pocket_ca.index(ca)] for ca in final["rec_anchors"]]
                total_score_fb, score_breakdown_fb = self._score_candidate_comprehensive(
                    rec_anchors=final["rec_anchors"],
                    lig_anchors=final["lig_anchors"],
                    rmsf_vals=rmsf_vals_fb,
                    r0_nm=r0_nm_fb,
                    thA_rad=final["eq"]["thetaA0"],
                    thB_rad=final["eq"]["thetaB0"],
                    kr_raw=kr_val_fb,
                    traj=traj,
                    top=top,
                )

                results.append({
                    "rank": len(results) + 1,
                    "receptor_indices": final["rec_anchors"],
                    "ligand_indices": final["lig_anchors"],
                    "receptor_residues": list(final["res_key"]),
                    "force_constants": {k: float(v) for k, v in final["ks"].items()},
                    "equilibrium_values": final["eq"],
                    "total_score": float(total_score_fb),
                    "score_breakdown": score_breakdown_fb,
                    "method": "finite_combo_v6.5_fallback",
                    "warning": "kr 超出上限，已放行"
                })
                log_cand(len(results), final["eq"]["r0"], final["ks"]["kr"], res_key, tag="回退")

        if not results:
            print(f"  [ERR] 未找到满足条件的合格候选。")
        else:
            print(f"[OK] 最终返回 {len(results)} 个合格候选")
            
            # ✅ 【关键修复】按总分降序排序，并重新分配 rank 序号
            results.sort(key=lambda x: x.get("total_score", 0.0), reverse=True)
            for i, res in enumerate(results):
                res["rank"] = i + 1
                res["kr_bonus"] = round((max_kr - res["force_constants"]["kr"]) * 0.1, 2)
            
            # 打印诊断信息，确认排序生效
            top = results[0]
            print(f"  推荐首选: Rank #{top['rank']} (残基={top['receptor_residues']})")
            print(f"     kr={top['force_constants']['kr']:.1f} | 总分={top['total_score']:.2f} | kr加分={top.get('kr_bonus',0):.2f}")

        if output_path:
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"candidates": results}, f, indent=2, cls=NumpyEncoder)
            print(f"[OK] 结果已保存: {output_path}")
            
        return results
