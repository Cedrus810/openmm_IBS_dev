

# ============================================================================
# 8. 纯轨迹几何波动 Boresch 估算器 (基于化学连通性 + 方差最小化)
# ============================================================================
class GeometricRestraintEstimator:
    """
    基于轨迹几何波动和化学连通性的 Boresch 参数估算器。
    不依赖任何力场，仅需 mdtraj 轨迹。
    核心流程：
      1. 受体候选原子：指定原子名 (默认 CA,CB,C,N,O)
      2. 0.5 nm 接触搜索找到 (锚点,配体) 最近原子对
      3. 0.22 nm 成键延伸构建受体三元组和配体三元组 (保证化学连通)
      4. 计算全轨迹距离/角度/二面角，周期性展开
      5. 硬截断 θ ∈ [45°,135°]
      6. 方差加权评分 (物理力常数尺度) 选择最优组合
      7. 力常数 = kBT / 方差 (并裁剪到安全范围)
    """

    def __init__(self, temperature=300.0,
                 search_dist=0.5,         # nm
                 bond_dist=0.22,          # nm，仅在拓扑不含键时作为显式回退
                 anchor_atom_names=None,
                 allow_geometric_bond_fallback=True,
                 # 某一侧要用拓扑真实键，该侧至少这个比例的原子能起出 2 深链。
                 # 真正描述了成键的拓扑接近 100%（实测受体 1404/1404）；只零星
                 # 知道几根残基间连接的接近 10%（实测配体 2/19）。0.5 把两者
                 # 分得很开，不是贴着数据挑的边界。
                 bond_topology_min_coverage=0.5):
        self.temperature = temperature
        self.gas_constant_kj_per_mol_k = 8.314e-3
        self.search_dist = search_dist
        self.bond_dist = bond_dist
        # 🔑 [ATT-11] 拓扑里有可用键时用真实键；这个开关只决定"某一侧拓扑没有可用键"
        # 时是回退到几何阈值还是直接 fail closed。
        self.allow_geometric_bond_fallback = bool(allow_geometric_bond_fallback)
        self.bond_topology_min_coverage = float(bond_topology_min_coverage)
        # 🔑 键来源**逐侧**记录。受体与配体的情况可以完全不同：实测本体系
        # （topology.cif，`_chem_comp_bond = 0`）受体锚点 1404/1404 都有真实键，
        # 而配体作为非标准残基只有 2/19 个重原子蹭到键——两侧必须分别决策。
        self.bond_source_receptor = None   # "topology" | "geometric_fallback"
        self.bond_source_ligand = None     # 同上
        self.bond_coverage = {}            # 逐侧 2 深链起点计数，供诊断落盘
        if anchor_atom_names is None:
            anchor_atom_names = ["CA", "CB", "C", "N", "O"]
        self.anchor_atom_names = anchor_atom_names

    @property
    def bond_source(self):
        """两侧的汇总：只有两侧都用真实键才算 "topology"。

        保留这个属性是为了兼容既有的诊断落盘读取方；真正的信息在
        `bond_source_receptor` / `bond_source_ligand` 里。
        """
        sides = (self.bond_source_receptor, self.bond_source_ligand)
        if None in sides:
            return None
        return "topology" if set(sides) == {"topology"} else "mixed_or_geometric_fallback"

    # ----------------------------------------------------------------
    # 工具：化学键邻居
    # ----------------------------------------------------------------
    def _build_bond_adjacency(self, topology) -> Optional[Dict[int, set]]:
        """🔑 [ATT-11] 从拓扑的真实成键关系建邻接表。

        原实现用 `距离 <= 0.22 nm` 冒充共价键，有三个问题：

        1. **区分不了成键与非键近接**。蛋白侧 haystack 是预筛过的锚点名子集
           （默认 CA/CB/C/N/O），其中 CA-CB≈0.153 nm、CA-C≈0.152 nm 是真键，
           但**非键**的 i/i+1 残基间 C-N≈0.133 nm、CA…N≈0.146 nm 同样落在
           0.22 nm 以内，会被当成键——于是"化学连通"的受体三元组可能跨残基
           拼出一条根本不存在的链。
        2. **漏掉长键**。S-S（≈0.205 nm）贴着阈值，金属配位键普遍超过 0.22 nm。
        3. **只看第 0 帧**（见 `_generate_anchor_combos` 的 `ref_xyz = traj.xyz[0]`），
           一次热涨落就能翻转键拓扑。

        Boresch 六原子锚点是由这张邻接表枚举出来的，锚点选错会直接改变解析释放
        修正——2026-07-27 的 P0-10 已经演示过锚点/平衡值出错的代价。

        返回 None 表示拓扑里没有键信息，由调用方决定回退还是 fail closed。
        """
        if topology is None:
            return None
        bonds = getattr(topology, "bonds", None)
        # mdtraj 的 Topology.bonds 是 property（生成器），OpenMM 的是方法。
        if callable(bonds):
            try:
                bonds = bonds()
            except TypeError:
                return None
        if bonds is None:
            return None
        adjacency: Dict[int, set] = {}
        n_bonds = 0
        for bond in bonds:
            try:
                a1, a2 = bond[0], bond[1]
            except (TypeError, KeyError, IndexError):
                a1, a2 = getattr(bond, "atom1", None), getattr(bond, "atom2", None)
            if a1 is None or a2 is None:
                continue
            i = int(getattr(a1, "index", a1))
            j = int(getattr(a2, "index", a2))
            adjacency.setdefault(i, set()).add(j)
            adjacency.setdefault(j, set()).add(i)
            n_bonds += 1
        return adjacency if n_bonds > 0 else None

    @staticmethod
    def _count_two_deep_chain_starts(adjacency, indices) -> int:
        """数一下该原子子集里有多少个原子能起出 `a→b→c` 且 b、c 都还在子集内。

        🔑 [ATT-11 回归修复] 这才是 `_generate_anchor_combos` 真正消费的性质。

        原先的覆盖度判据是 `any(adjacency.get(i) for i in indices)`——「该侧只要有
        任意一个原子有键就放行」。实测那道判据太弱到直接造成生产崩溃：
        `topology.cif` 的 `_chem_comp_bond = 0`，配体 `MOL` 作为非标准残基只有
        **2/19** 个重原子从 `_struct_conn` 蹭到键，`any()` 被这 2 个原子放行，
        于是走了拓扑路径；而那 2 个原子恰好不在接触对里，配体侧最内层枚举
        从不执行 → `化学连通候选组合数: 0` → `RuntimeError: 没有符合条件的6原子组合`。

        注意「链上原子必须都在子集内」不是多余的限制：受体 haystack 是预筛的
        CA/CB/C/N/O 子集，配体 haystack 排除了氢；链走出子集就不能用来构造
        Boresch 三元组。
        """
        if not adjacency:
            return 0
        subset = {int(x) for x in indices}
        n = 0
        for a in subset:
            for b in adjacency.get(a, ()):  # noqa: B007
                if int(b) not in subset:
                    continue
                if (adjacency.get(int(b), set()) & subset) - {a}:
                    n += 1
                    break
        return n

    def _resolve_side_adjacency(self, side_label, adjacency, indices):
        """为某一侧（受体 / 配体）决定用拓扑真实键还是几何回退。

        返回 `(adjacency_or_None, source, n_two_deep, n_atoms)`。

        **判据是覆盖度比例，不是"有没有"。** 一个真正描述了该侧成键的拓扑，
        几乎每个原子都能起出 2 深链（实测受体锚点 1404/1404 = 100%）；
        而只零星知道几根残基间连接的拓扑给出的是 10% 这个量级
        （实测配体 2/19 = 10.5%）。用 `>= 1` 当门槛挡不住后者——那 2 个原子
        恰好不在接触对里，配体侧枚举照样全灭，正是 2026-07-27 生产崩溃的原因。

        **逐侧决策的理由是一个真实的不对称**：

        - **配体侧**：haystack 是全部重原子，`0.22 nm` 的最近邻**确实就是化学键**
          （小分子键长 0.13–0.16 nm，次近邻 ≥ 0.24 nm）。所以几何回退在这一侧
          是可靠的。
        - **受体侧**：haystack 被按原子名预筛成 CA/CB/C/N/O，于是**非键**的残基间
          C–N（≈0.133 nm）、CA…N（≈0.146 nm）也落在阈值内，会拼出跨残基的假
          「化学连通」三元组。所以这一侧必须用真实键。

        换句话说：ATT-11 的收益全在受体侧，而配体侧本来就不太需要它。
        """
        n_atoms = len(indices)
        n_two_deep = self._count_two_deep_chain_starts(adjacency, indices)
        frac = (n_two_deep / n_atoms) if n_atoms else 0.0
        min_frac = float(self.bond_topology_min_coverage)

        if n_atoms and frac >= min_frac:
            print(
                f"  [OK] [Boresch 锚点] {side_label}使用拓扑真实键"
                f"（2 深链起点 {n_two_deep}/{n_atoms} = {frac:.0%}）"
            )
            return adjacency, "topology", n_two_deep, n_atoms

        if not self.allow_geometric_bond_fallback:
            raise RuntimeError(
                f"{side_label}的拓扑成键覆盖度不足："
                f"2 深链起点仅 {n_two_deep}/{n_atoms} = {frac:.0%}，"
                f"低于要求的 {min_frac:.0%}，不足以可靠枚举 Boresch 三元组；"
                f"且已禁用几何回退，拒绝用 `距离 <= {self.bond_dist} nm` 冒充共价键。"
                "\n  请提供带该侧完整键的拓扑（例如带 CONECT 的 PDB / prmtop / "
                "OpenMM Topology），或显式设 allow_geometric_bond_fallback=True。"
                "\n  提示：OpenMM 写出的 mmCIF 不含 `_chem_comp_bond`，"
                "非标准残基（配体）在其中没有键。"
            )
        print(
            f"  [WARN] [Boresch 锚点] {side_label}拓扑成键覆盖度不足（2 深链起点 "
            f"{n_two_deep}/{n_atoms} = {frac:.0%} < {min_frac:.0%}），"
            f"该侧退回几何阈值 {self.bond_dist} nm。"
        )
        return None, "geometric_fallback", n_two_deep, n_atoms

    def _find_bonded_neighbors(self, atom_idx, haystack, ref_xyz, adjacency=None):
        """返回 haystack 里与 atom_idx 成键的原子。

        `adjacency` 非空时用真实成键关系；为空时（拓扑无键）才回退到
        `距离 <= bond_dist` 的几何近似——见 `_build_bond_adjacency` 的说明。
        """
        haystack = np.asarray(haystack)
        if adjacency is not None:
            neighbors = adjacency.get(int(atom_idx), ())
            return [int(b) for b in haystack if int(b) in neighbors and int(b) != int(atom_idx)]
        vec = ref_xyz[haystack] - ref_xyz[atom_idx]
        dist = np.linalg.norm(vec, axis=1)
        bonded = haystack[dist <= self.bond_dist]
        return [int(b) for b in bonded if int(b) != int(atom_idx)]

    @staticmethod
    def _clip_force_constant(value, lower, upper):
        raw = float(value)
        clipped = float(np.clip(raw, lower, upper))
        return clipped, bool(abs(clipped - raw) > 1e-8)

    @staticmethod
    def _fluctuation_diagnostics(values, name):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 4:
            return {
                "name": name,
                "n": int(vals.size),
                "ok": False,
                "reason": "too_few_finite_samples",
            }

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std <= 1e-12:
            return {
                "name": name,
                "n": int(vals.size),
                "ok": False,
                "mean": mean,
                "std": std,
                "reason": "near_zero_variance",
            }

        centered = (vals - mean) / std
        skew = float(np.mean(centered ** 3))
        excess_kurtosis = float(np.mean(centered ** 4) - 3.0)
        p01, p50, p99 = np.percentile(vals, [1, 50, 99])
        ok = bool(abs(skew) <= 2.0 and abs(excess_kurtosis) <= 7.0)
        reason = "ok" if ok else "non_gaussian_tail_or_asymmetry"
        return {
            "name": name,
            "n": int(vals.size),
            "ok": ok,
            "mean": mean,
            "std": std,
            "skew": skew,
            "excess_kurtosis": excess_kurtosis,
            "p01": float(p01),
            "p50": float(p50),
            "p99": float(p99),
            "reason": reason,
        }

    # ----------------------------------------------------------------
    # 生成所有化学连通的 6-原子组合
    # ----------------------------------------------------------------
    def _generate_anchor_combos(self, traj, prot_indices, lig_heavy_indices):
        ref_xyz = traj.xyz[0]

        # 1. 接触对 (anch, lig) 距离 ≤ search_dist
        lig_pos = ref_xyz[lig_heavy_indices]
        prot_pos = ref_xyz[prot_indices]
        dist_mat = np.linalg.norm(prot_pos[:, None, :] - lig_pos[None, :, :], axis=2)
        contact_pairs = [(prot_indices[i], lig_heavy_indices[j])
                         for i, j in zip(*np.where(dist_mat <= self.search_dist))]
        if not contact_pairs:
            raise RuntimeError("未找到锚点-配体接触对，请增大 search_dist")

        # 2. 预计算键合邻居字典
        # 🔑 [ATT-11 + 回归修复] 受体与配体**各自独立**决定键来源。
        # 实测本体系（topology.cif，`_chem_comp_bond = 0`）：受体锚点 1404/1404
        # 都有真实键，而配体作为非标准残基只有 2/19 个重原子蹭到键。全局决策
        # （任一侧不合格就整体退几何）会白丢受体侧的收益；反之若因为「配体有 2 个
        # 原子有键」就整体用拓扑，配体侧枚举直接全灭——那正是 2026-07-27 生产崩溃
        # （`化学连通候选组合数: 0`）的原因。
        adjacency = self._build_bond_adjacency(getattr(traj, "topology", None))
        prot_adj, self.bond_source_receptor, prot_deep, prot_n = (
            self._resolve_side_adjacency("受体锚点侧", adjacency, prot_indices)
        )
        lig_adj, self.bond_source_ligand, lig_deep, lig_n = (
            self._resolve_side_adjacency("配体侧", adjacency, lig_heavy_indices)
        )
        self.bond_coverage = {
            "receptor_two_deep_chain_starts": int(prot_deep),
            "receptor_n_atoms": int(prot_n),
            "ligand_two_deep_chain_starts": int(lig_deep),
            "ligand_n_atoms": int(lig_n),
        }

        prot_nei = {
            idx: self._find_bonded_neighbors(idx, prot_indices, ref_xyz, prot_adj)
            for idx in prot_indices
        }
        lig_nei = {
            idx: self._find_bonded_neighbors(idx, lig_heavy_indices, ref_xyz, lig_adj)
            for idx in lig_heavy_indices
        }

        anclig_combos = []
        for anc, lig in contact_pairs:
            # 受体侧：以 anc 为 a，找与其键合的 b，再找与 b 键合的 c -> (c, b, a)
            for b in prot_nei.get(anc, []):
                for c in prot_nei.get(b, []):
                    if c == anc: continue
                    rec_tri = (c, b, anc)
                    # 配体侧：以 lig 为 a，找键合的 b，再找与 b 键合的 c -> (a, b, c)
                    for b_lig in lig_nei.get(lig, []):
                        for c_lig in lig_nei.get(b_lig, []):
                            if c_lig == lig: continue
                            lig_tri = (lig, b_lig, c_lig)
                            anclig_combos.append((rec_tri, lig_tri))

        # 去重
        unique = []
        seen = set()
        for rec, lig in anclig_combos:
            key = rec + lig
            if key not in seen:
                seen.add(key)
                unique.append((rec, lig))
        return unique

    # ----------------------------------------------------------------
    # 主估算函数
    # ----------------------------------------------------------------
    def estimate_from_trajectory(self, traj, ligand_resname, output_path=None):
        top = traj.topology

        # 1. 受体锚点候选原子 (基于原子名)
        anchor_query = "protein and name " + ' '.join(self.anchor_atom_names)
        prot_indices = top.select(anchor_query)
        if len(prot_indices) == 0:
            raise RuntimeError(f"没有找到锚点原子：{self.anchor_atom_names}")

        # 2. 配体重原子
        lig_heavy = top.select(f"resname {ligand_resname} and not element H")
        if len(lig_heavy) == 0:
            raise RuntimeError(f"未找到配体 {ligand_resname} 的重原子")

        # 3. 化学连通组合枚举
        combos = self._generate_anchor_combos(traj, prot_indices, lig_heavy)
        print(f"  化学连通候选组合数: {len(combos)}")
        if len(combos) == 0:
            raise RuntimeError("没有符合条件的6原子组合")

        # 4. 构建 mdtraj 原子索引列表 (用于批量计算几何)
        n_combos = len(combos)
        dist_indices   = [[c[0][2], c[1][0]] for c in combos]   # anc_a - lig_a
        angleA_indices = [[c[0][1], c[0][2], c[1][0]] for c in combos]  # anc_b, anc_a, lig_a
        angleB_indices = [[c[0][2], c[1][0], c[1][1]] for c in combos]
        dihA_indices   = [[c[0][0], c[0][1], c[0][2], c[1][0]] for c in combos]
        dihB_indices   = [[c[0][1], c[0][2], c[1][0], c[1][1]] for c in combos]
        dihC_indices   = [[c[0][2], c[1][0], c[1][1], c[1][2]] for c in combos]

        # 5. 逐帧计算几何量 (分块避免 OOM)
        n_frames = len(traj)
        dists    = np.zeros((n_frames, n_combos))
        angles_a = np.zeros((n_frames, n_combos))
        angles_b = np.zeros((n_frames, n_combos))
        diheds_a = np.zeros((n_frames, n_combos))
        diheds_b = np.zeros((n_frames, n_combos))
        diheds_c = np.zeros((n_frames, n_combos))

        chunk_size = 100  # 可调整
        for i in range(0, n_frames, chunk_size):
            chunk = traj[i:i+chunk_size]
            dists[i:i+len(chunk)]    = mdtraj.compute_distances(chunk, dist_indices)
            angles_a[i:i+len(chunk)] = mdtraj.compute_angles(chunk, angleA_indices)
            angles_b[i:i+len(chunk)] = mdtraj.compute_angles(chunk, angleB_indices)
            diheds_a[i:i+len(chunk)] = mdtraj.compute_dihedrals(chunk, dihA_indices)
            diheds_b[i:i+len(chunk)] = mdtraj.compute_dihedrals(chunk, dihB_indices)
            diheds_c[i:i+len(chunk)] = mdtraj.compute_dihedrals(chunk, dihC_indices)

        # 6. 周期性二面角展开 (按列展开，保持连续性)
        def periodic_unwrap(dh_array):
            for col in range(dh_array.shape[1]):
                vals = dh_array[:, col]
                for t in range(1, len(vals)):
                    diff = vals[t] - vals[t-1]
                    vals[t] -= 2*np.pi * np.round(diff / (2*np.pi))
                mean_val = np.mean(vals)
                vals -= 2*np.pi * np.round(mean_val / (2*np.pi))
                dh_array[:, col] = vals

        periodic_unwrap(diheds_a)
        periodic_unwrap(diheds_b)
        periodic_unwrap(diheds_c)

        # 7. 方差加权评分 (物理力常数尺度)
        dist_weight = 4184.0       # kJ/mol/nm²
        angle_weight = 41.84       # kJ/mol/rad²
        dihedral_weight = 41.84

        var_dist = np.var(dists, axis=0)
        var_angA = np.var(angles_a, axis=0)
        var_angB = np.var(angles_b, axis=0)
        var_dihA = np.var(diheds_a, axis=0)
        var_dihB = np.var(diheds_b, axis=0)
        var_dihC = np.var(diheds_c, axis=0)

        total_var = (dist_weight * var_dist +
                     angle_weight * (var_angA + var_angB) +
                     dihedral_weight * (var_dihA + var_dihB + var_dihC))

        # 8. 硬截断：排除平均角度不在 [45°,135°] 的候选 (避免 1/sinθ 奇点)
        avg_angA = np.mean(angles_a, axis=0)
        avg_angB = np.mean(angles_b, axis=0)
        banned = (avg_angA < np.deg2rad(45)) | (avg_angA > np.deg2rad(135)) | \
                 (avg_angB < np.deg2rad(45)) | (avg_angB > np.deg2rad(135))
        total_var[banned] = np.inf

        # 🔑 核心修复：拦截全 inf 灾难
        if np.all(np.isinf(total_var)):
            raise RuntimeError(
                "[ERR] 所有候选锚点组合的几何角度 (θA/θB) 均超出安全范围 [45°, 135°]！\n"
                "   体系可能存在严重畸变或配体脱离口袋。请检查预平衡轨迹，或使用 --boresch-source auto 切换至 Orb 估算。"
            )

        # 9. 选择最优组合
        best_idx = np.argmin(total_var)
        best_combo = combos[best_idx]

        # 10. 提取平衡值 (平均值) 和力常数
        eq = {
            "r0":       float(np.mean(dists[:, best_idx])),
            "thetaA0":  float(np.mean(angles_a[:, best_idx])),
            "thetaB0":  float(np.mean(angles_b[:, best_idx])),
            "phiA0":    float(np.mean(diheds_a[:, best_idx])),
            "phiB0":    float(np.mean(diheds_b[:, best_idx])),
            "phiC0":    float(np.mean(diheds_c[:, best_idx])),
        }

        kBT = self.gas_constant_kj_per_mol_k * self.temperature
        raw_fc = {
            "kr":       kBT / (var_dist[best_idx] + 1e-10),
            "kthetaA":  kBT / (var_angA[best_idx] + 1e-10),
            "kthetaB":  kBT / (var_angB[best_idx] + 1e-10),
            "kphiA":    kBT / (var_dihA[best_idx] + 1e-10),
            "kphiB":    kBT / (var_dihB[best_idx] + 1e-10),
            "kphiC":    kBT / (var_dihC[best_idx] + 1e-10),
        }
        force_constant_ranges = {
            "kr": [100.0, 2000.0],
            "kthetaA": [10.0, 1000.0],
            "kthetaB": [10.0, 1000.0],
            "kphiA": [10.0, 1000.0],
            "kphiB": [10.0, 1000.0],
            "kphiC": [10.0, 1000.0],
        }
        fc = {}
        clipped_flags = {}
        for key, raw_value in raw_fc.items():
            lower, upper = force_constant_ranges[key]
            fc[key], clipped_flags[key] = self._clip_force_constant(raw_value, lower, upper)

        fluctuation_diagnostics = [
            self._fluctuation_diagnostics(dists[:, best_idx], "r"),
            self._fluctuation_diagnostics(angles_a[:, best_idx], "thetaA"),
            self._fluctuation_diagnostics(angles_b[:, best_idx], "thetaB"),
            self._fluctuation_diagnostics(diheds_a[:, best_idx], "phiA"),
            self._fluctuation_diagnostics(diheds_b[:, best_idx], "phiB"),
            self._fluctuation_diagnostics(diheds_c[:, best_idx], "phiC"),
        ]
        n_bad_diag = sum(1 for item in fluctuation_diagnostics if not item.get("ok", False))
        n_clipped = sum(1 for clipped in clipped_flags.values() if clipped)

        # 🚨 关键修复：best_combo[0] (rec_tri) 内部是按 (c,b,anc)=(最远,中间,最近)
        # 的顺序构建的——上面 dist/angle/dihedral 的 index 列表都正确利用了这个
        # 顺序算出了符合 R0(最近)-顶点约定的 eq/fc；但如果直接原样存成
        # receptor_indices，会跟 _check_boresch_geometry_safe /
        # calc_boresch_from_last_frame / LambdaDependentBoreschForce 全部假设的
        # "receptor_indices[0]=离配体最近的锚点" 顺序相反，导致下游重新读取这份
        # 结果时把最远锚点当成了 R0。这里显式反转，使其对外统一为最近在前。
        result = {
            "receptor_indices": list(reversed(best_combo[0])),
            "ligand_indices": list(best_combo[1]),
            "equilibrium_values": eq,
            "force_constants": fc,
            "force_constants_raw": {k: float(v) for k, v in raw_fc.items()},
            "force_constant_clip_ranges": force_constant_ranges,
            "force_constant_clipped": clipped_flags,
            "diagnostics": {
                "n_frames": int(n_frames),
                "n_candidates": int(n_combos),
                # 🔑 [ATT-11] 锚点三元组是靠成键关系枚举出来的；这里逐侧记录用的是
                # 拓扑真实键还是 0.22 nm 几何近似，后者会把残基间非键近接
                # （C-N≈0.133 nm）误判成键。锚点选错直接改变解析释放修正。
                # 两侧情况可以不同：mmCIF 常有全部蛋白键但配体（非标准残基）无键。
                "bond_source": self.bond_source,           # 两侧汇总，兼容旧读取方
                "bond_source_receptor": self.bond_source_receptor,
                "bond_source_ligand": self.bond_source_ligand,
                "bond_coverage": dict(self.bond_coverage),
                "bond_dist_nm_if_geometric": (
                    float(self.bond_dist)
                    if "geometric_fallback" in (
                        self.bond_source_receptor, self.bond_source_ligand
                    ) else None
                ),
                "n_angle_banned_candidates": int(np.sum(banned)),
                "best_total_variance_score": float(total_var[best_idx]),
                "fluctuation_distribution": fluctuation_diagnostics,
                "n_non_gaussian_or_under_sampled_terms": int(n_bad_diag),
                "n_clipped_force_constants": int(n_clipped),
                "warnings": [
                    "Some fluctuation-derived force constants were clipped to conservative bounds."
                    if n_clipped else "",
                    "One or more restraint coordinates show non-Gaussian or under-sampled fluctuations."
                    if n_bad_diag else "",
                ],
            },
            "method": "geometric_fluctuation_v2_clipped",
        }
        result["diagnostics"]["warnings"] = [
            warning for warning in result["diagnostics"]["warnings"] if warning
        ]

        if output_path:
            with open(output_path, 'w', encoding="utf-8") as f:
                json.dump(result, f, indent=2, cls=NumpyEncoder)

        print(f"  最优锚点: 受体 {result['receptor_indices']} | 配体 {result['ligand_indices']}")
        print(f"     r0={eq['r0']*10:.2f} Å, θA={np.degrees(eq['thetaA0']):.1f}°, θB={np.degrees(eq['thetaB0']):.1f}°")
        print(f"     kr={fc['kr']:.1f} kJ/mol/nm², kθA={fc['kthetaA']:.1f} kJ/mol/rad²")
        if n_clipped:
            print(f"  [WARN] fluctuation Boresch 有 {n_clipped} 个力常数被裁剪；raw 值已写入结果 JSON。")
        if n_bad_diag:
            print(f"  [WARN] fluctuation Boresch 有 {n_bad_diag} 个坐标分布偏离高斯或采样不足；请检查 diagnostics。")
        return result


