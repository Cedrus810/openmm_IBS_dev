# ---------------------------------------------------------------------------
# 原生系统缓存读写
# ---------------------------------------------------------------------------
def diagnose_14_scaling(system: openmm.System) -> Optional[Dict[str, Any]]:
    """报告 NonbondedForce 里**真正的 1-4 对**的缩放因子。

    🔑🔑 [2026-09-16] **先前这个诊断是误导性的，不是力场错了。**

    旧实现把 `getNumExceptions()` 里的**每一条**异常都拿来平均 —— 而绝大多数异常是
    1-2/1-3 的**完全排除**（chargeProd=0、epsilon=0）。排除对在分子里比 1-4 对多得多，
    于是平均值被一堆结构性的 0 压向 0：真机 thrombin 打出 fudgeQQ=0.1590 / fudgeLJ=0.2402，
    看着像力场参数错了，实际上只是"把排除对算进了缩放因子的平均"。

    正确口径：**只有显式 1-4 对才有缩放因子可言**。判据不看键图（System 里没有），
    而是看这条异常本身有没有保留相互作用：
      · `chargeProd == 0 且 epsilon == 0`  ⟹ 完全排除（1-2/1-3）⟹ **不参与统计**；
      · 其余                                 ⟹ 显式 pair（1-4）⟹ 参与统计。
    分母为 0 的对（例如两个原子之一 q=0 或 ε=0）同样不参与 —— 那是"这条对上测不出
    缩放因子"，不是"缩放因子是 0"。

    只报中位数与离散度，不报平均：GROMACS 的 fudgeQQ 是全局常数，一旦有离群值，
    要看的是"有几条不一样"，而不是被平均掉的那个数。
    返回统计结果（便于测试与 provenance）；读不到 NonbondedForce 时返回 None。
    """
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            nb = force
            break
    else:
        log.warning("  [WARN] 未找到 NonbondedForce，无法诊断 1-4 缩放")
        return None

    n_excluded = 0
    qq_scales: List[float] = []
    lj_scales: List[float] = []
    n_pair_no_qq_ref = 0
    n_pair_no_lj_ref = 0
    for i in range(nb.getNumExceptions()):
        p1, p2, cp, sigma, eps = nb.getExceptionParameters(i)
        cp_e2 = cp.value_in_unit(unit.elementary_charge**2)
        eps_kj = eps.value_in_unit(unit.kilojoule_per_mole)
        if abs(cp_e2) <= 1.0e-12 and abs(eps_kj) <= 1.0e-12:
            n_excluded += 1          # 完全排除的 1-2/1-3，没有"缩放因子"这回事
            continue
        q1, _sig1, eps1 = nb.getParticleParameters(int(p1))
        q2, _sig2, eps2 = nb.getParticleParameters(int(p2))
        q_prod_raw = (q1 * q2).value_in_unit(unit.elementary_charge**2)
        if abs(q_prod_raw) > 1.0e-10:
            qq_scales.append(cp_e2 / q_prod_raw)
        else:
            n_pair_no_qq_ref += 1
        eps_cross = float(np.sqrt(
            eps1.value_in_unit(unit.kilojoule_per_mole)
            * eps2.value_in_unit(unit.kilojoule_per_mole)
        ))
        if eps_cross > 1.0e-10:
            lj_scales.append(eps_kj / eps_cross)
        else:
            n_pair_no_lj_ref += 1

    def _stat(values: List[float]) -> Dict[str, Any]:
        if not values:
            return {"n": 0, "median": None, "min": None, "max": None}
        arr = np.asarray(values, dtype=float)
        return {
            "n": int(arr.size),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    report = {
        "n_exceptions_total": int(nb.getNumExceptions()),
        "n_fully_excluded": int(n_excluded),
        "n_explicit_pairs": int(nb.getNumExceptions() - n_excluded),
        "fudgeQQ": _stat(qq_scales),
        "fudgeLJ": _stat(lj_scales),
        "n_pairs_without_coulomb_reference": int(n_pair_no_qq_ref),
        "n_pairs_without_lj_reference": int(n_pair_no_lj_ref),
    }
    if report["n_explicit_pairs"] == 0:
        log.info(
            "  1-4 缩放诊断: 全部 %d 条异常都是完全排除，没有显式 1-4 对可统计"
            "（这本身不是错误，例如只有排除表的体系）",
            report["n_exceptions_total"],
        )
        return report
    _q, _l = report["fudgeQQ"], report["fudgeLJ"]
    log.info(
        "  1-4 缩放诊断（只统计 %d 条显式 pair，已剔除 %d 条完全排除）: "
        "fudgeQQ 中位数=%s [%s, %s] (n=%d, 应 ~0.8333), "
        "fudgeLJ 中位数=%s [%s, %s] (n=%d, 应 ~0.5)",
        report["n_explicit_pairs"], report["n_fully_excluded"],
        *( "%.4f" % v if v is not None else "n/a"
           for v in (_q["median"], _q["min"], _q["max"]) ),
        _q["n"],
        *( "%.4f" % v if v is not None else "n/a"
           for v in (_l["median"], _l["min"], _l["max"]) ),
        _l["n"],
    )
    return report

def save_native_system(
    output_dir,
    system,
    topology,
    ligand_indices,
    positions,
    box_vectors,
    cache_identity: Optional[Dict] = None,
    reserved_coion_builder_identity: Optional[Dict] = None,
    charge_treatment: Optional[str] = None,
):
    """持久化原生 System XML 与辅助数据至 output_dir（内置 convert 功能）"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. System XML (强制 Python 所有权)
    native_sys = ensure_owned_system(system)
    xml_path = os.path.join(output_dir, "system_native.xml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(XmlSerializer.serialize(native_sys))
    log.info("  [缓存] System XML 已保存: %s", xml_path)

    # 2. Ligand indices
    lig_path = os.path.join(output_dir, "ligand_indices.json")
    with open(lig_path, "w", encoding="utf-8") as f:
        json.dump({"ligand_indices": [int(i) for i in ligand_indices]}, f, indent=2)
    log.info("  [缓存] 配体索引已保存: %s", lig_path)

    # 3. Box vectors
    if box_vectors is not None:
        box_path = os.path.join(output_dir, "box_vectors.npy")
        if hasattr(box_vectors, 'value_in_unit'):
            box_nm = box_vectors.value_in_unit(unit.nanometer)
        else:
            box_nm = box_vectors
        np.save(box_path, np.asarray(box_nm, dtype=np.float64))
        log.info("  [缓存] 盒子向量已保存: %s", box_path)

    # 4. Topology 缓存 (🔑 改用 mmCIF 格式，彻底解决 PDB >99999 原子序列号溢出与截断问题)
    top_path = os.path.join(output_dir, "topology.cif")
    try:
        app.PDBxFile.writeFile(topology, positions, top_path)
        log.info("  [缓存] 参考拓扑已保存 (mmCIF): %s", top_path)
    except Exception as e:
        log.warning("  [缓存] 拓扑 mmCIF 保存失败: %s", e)

    if cache_identity is None:
        raise ValueError("拒绝写入无输入身份的主 System 缓存")
    manifest = {
        "protocol_version": MAIN_SYSTEM_CACHE_PROTOCOL_VERSION,
        **cache_identity,
        "system_xml_sha256": _sha256_file(xml_path),
        "ligand_indices_sha256": _sha256_file(lig_path),
        "topology_sha256": _sha256_file(top_path) if os.path.isfile(top_path) else None,
        "box_vectors_sha256": (
            _sha256_file(os.path.join(output_dir, "box_vectors.npy"))
            if os.path.isfile(os.path.join(output_dir, "box_vectors.npy"))
            else None
        ),
    }
    if reserved_coion_builder_identity is not None:
        manifest["reserved_coion_builder_identity"] = reserved_coion_builder_identity
    if charge_treatment is not None:
        manifest["charge_treatment"] = str(charge_treatment)
    manifest_path = os.path.join(output_dir, "system_cache_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    log.info("  [缓存] 输入身份 manifest 已保存: %s", manifest_path)

def _select_reserved_coion_water_sites(
    *,
    positions_nm: np.ndarray,
    box_nm: np.ndarray,
    ligand_atom_indices: Sequence[int],
    water_oxygen_indices: Sequence[int],
    count: int,
) -> List[int]:
    """挑 `count` 个"离配体最远、彼此也够远"的水分子氧原子，返回它们的 atom index。

    🔑🔑 [2026-09-16] **这是选点规则的唯一实现。** 两条腿共用：
      · 溶剂腿走 `_insert_reserved_coalchemical_ion_dummies`（Modeller 加删）；
      · 复合物腿走 `_derive_gromacs_inputs_with_reserved_coions`（改写 .gro/.top）。
    先前只有溶剂腿有这段逻辑，复合物腿**根本没有插入步骤**，于是带电配体在
    `co_alchemical_ion_builder_identity_payload` 的数量契约上当场炸（"找到 0 个"）。
    两侧各写一份选点规则就是本仓最贵的那类 bug，所以这里只留一份。

    放置策略（§4.4 "初始位置远离配体且不与周期镜像过近"）：贪心 farthest-first ——
    按离配体质心 minimum-image 距离从远到近扫描，跳过与**已选中**点距离不够
    `COION_COION_MIN_IMAGE_INITIAL_NM` 的候选。

    ⚠️ `count > 1` 时不能只按"离配体最远"独立打分：方盒里"最远的 N 个点"天然挤在
    同一个远角。2026-08-06 用 Ca²⁺(+2) 实测踩到过——两个 dummy 只相距 0.18~0.43 nm，
    λ→0 时两者同号各带 +1e，贴脸的静电排斥把 charging MBAR 拖到不收敛
    （ΔG≈660 kJ/mol，min_overlap<0.02）。
    """
    if count <= 0:
        return []
    lig_indices = [int(i) for i in ligand_atom_indices]
    if not lig_indices:
        raise ValueError("挑 reserved co-ion 位点需要非空 ligand_atom_indices")
    oxygens = [int(i) for i in water_oxygen_indices]
    if len(oxygens) < count:
        raise RuntimeError(
            f"盒子里只有 {len(oxygens)} 个水分子，不够替换出 {count} 个 "
            "reserved co-ion dummy（§4.2：盒子必须留够 bulk-water 壳层，"
            "考虑加大 padding_nm）。"
        )
    pos_nm = np.asarray(positions_nm, dtype=np.float64)
    box_nm = np.asarray(box_nm, dtype=np.float64)

    # A raw arithmetic centroid is wrong when a ligand straddles a periodic
    # boundary (it can land in the opposite side of the box).  Reconstruct the
    # centroid in the same local image used by the co-ion geometry checks.
    lig_origin = pos_nm[lig_indices[0]]
    lig_offsets = minimum_image_displacement_nm(
        pos_nm[lig_indices] - lig_origin, box_nm
    )
    lig_centroid_nm = lig_origin + lig_offsets.mean(axis=0)

    def _mic_distance_between(p: np.ndarray, q: np.ndarray) -> float:
        return float(np.linalg.norm(minimum_image_displacement_nm(p - q, box_nm)))

    scored = sorted(
        ((_mic_distance_between(pos_nm[i], lig_centroid_nm), i) for i in oxygens),
        key=lambda item: item[0],
        reverse=True,
    )
    chosen: List[int] = []
    chosen_pos: List[np.ndarray] = []
    for _dist, idx in scored:
        if len(chosen) == count:
            break
        candidate = pos_nm[idx]
        if all(
            _mic_distance_between(candidate, p) >= COION_COION_MIN_IMAGE_INITIAL_NM
            for p in chosen_pos
        ):
            chosen.append(idx)
            chosen_pos.append(candidate.copy())
    if len(chosen) < count:
        raise RuntimeError(
            f"在这个盒子里找不到 {count} 个彼此 minimum-image 距离 ≥ "
            f"{COION_COION_MIN_IMAGE_INITIAL_NM} nm 的候选水分子来放 reserved "
            f"co-ion dummy（§2.2：多个 dummy 之间也必须留够安全边距，否则它们在 "
            "λ→0 同号带电时会彼此靠得太近、产生非物理静电排斥）。只找到 "
            f"{len(chosen)} 个满足条件的候选，考虑加大 padding_nm/盒子。"
        )
    return chosen


def _insert_reserved_coalchemical_ion_dummies(
    modeller,
    count: int,
    cation: bool,
    ligand_atom_indices: List[int],
) -> List[int]:
    """[B4，§4.1/§2.2] 在 `modeller` 里插入 `count` 个建系时预留的中性 ion-shaped
    dummy 粒子，返回它们在**最终拓扑**里的 atom index。

    只负责"让判据能找到粒子"——真正的身份选择/restraint/charging 电荷映射全部
    交给已有的 `ibs_engine._identify_reserved_neutral_co_ions`（判据：残基名在
    离子集合里、且电荷严格为 0，坐标无关）。这里造出来的粒子电荷仍是模板给的
    ±1，调用方必须在 `createSystem()` 之后显式清零。

    放置策略（§4.4 "初始位置远离配体且不与周期镜像过近"）：贪心 farthest-first
    地摘水分子——按离配体质心 minimum-image 距离从远到近扫描候选，但跳过与
    **已选中**的 dummy 距离不够 `COION_COION_MIN_IMAGE_INITIAL_NM` 的候选，
    把 dummy 放在被摘掉的氧原子原来的位置。零额外体积冲突，且"越远的水优先
    被选中"天然满足配体↔dummy 的安全边距；是否真的够远由下游
    `abfe_core.validate_co_alchemical_ion_placement`（strict=True）兜底 fail
    closed，这里不重复判一遍阈值。

    ⚠️ `count > 1` 时（§2.2 多价配体分摊成多个单价 co-ion）不能只按"离配体最远"
    独立给每个候选打分——方盒里"离配体质心最远的 N 个点"天然会挤在同一个远角，
    2026-08-06 用 Ca²⁺(+2) 实测踩到过：两个 dummy 只相距 0.18~0.43 nm，λ→0 时
    两者同号各带 +1e，几乎贴脸的静电排斥直接把 charging MBAR 拖到不收敛
    （ΔG≈660 kJ/mol，min_overlap<0.02）。所以选点时额外要求任意两个已选 dummy
    之间也满足这个下限，找不满 `count` 个就 fail closed（不静默退化成"离得近也
    凑合用"）。

    ⚠️ 先一次性摘完全部选中的水、再一次性加入全部 dummy——不要在循环里交替
    delete/add：`Modeller.delete()` 会重新编号拓扑，交替做会让本函数早先加入的
    dummy 的 index 被后面的 delete 悄悄移位。
    """
    if count <= 0:
        return []
    element = app.element.sodium if cation else app.element.chlorine
    residue_name = "NA" if cation else "CL"

    pos_nm = np.asarray(
        modeller.positions.value_in_unit(unit.nanometer), dtype=np.float64
    )
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    if box_vecs is None:
        raise RuntimeError("插入 reserved co-ion dummy 需要周期盒向量，但拓扑里没有")
    box_nm = np.array(
        [v.value_in_unit(unit.nanometer) for v in box_vecs], dtype=np.float64
    )
    water_residues = [
        res
        for res in modeller.topology.residues()
        if str(res.name).strip().upper() in WATER_MOLECULE_NAMES
    ]
    oxygen_of_residue = {}
    for res in water_residues:
        atoms = list(res.atoms())
        o_atom = next(
            (a for a in atoms if a.element == app.element.oxygen), atoms[0]
        )
        oxygen_of_residue[int(o_atom.index)] = res
    # 选点规则只有一份实现（复合物腿走同一个函数）。
    chosen_oxygen_indices = _select_reserved_coion_water_sites(
        positions_nm=pos_nm,
        box_nm=box_nm,
        ligand_atom_indices=ligand_atom_indices,
        water_oxygen_indices=sorted(oxygen_of_residue),
        count=count,
    )
    chosen_positions_nm = [pos_nm[i].copy() for i in chosen_oxygen_indices]

    # 摘掉这些水分子（连同它们的 H）——按 residue 删，不是按单个原子删。
    index_to_residue = {
        atom.index: res for res in water_residues for atom in res.atoms()
    }
    residues_to_delete = {index_to_residue[idx] for idx in chosen_oxygen_indices}
    to_delete = [atom for res in residues_to_delete for atom in res.atoms()]
    modeller.delete(to_delete)

    dummy_top = app.Topology()
    dummy_chain = dummy_top.addChain()
    for _ in range(count):
        residue = dummy_top.addResidue(residue_name, dummy_chain)
        dummy_top.addAtom(residue_name, element, residue)
    dummy_positions = [
        Vec3(float(p[0]), float(p[1]), float(p[2])) for p in chosen_positions_nm
    ] * unit.nanometer

    n_before = sum(1 for _ in modeller.topology.atoms())
    modeller.add(dummy_top, dummy_positions)
    return list(range(n_before, n_before + count))



def _gro_atom_line(res_seq: int, res_name: str, atom_name: str, atom_seq: int,
                   xyz_nm) -> str:
    """一行标准 .gro 原子记录（固定列宽，无速度）。"""
    return "%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (
        int(res_seq) % 100000, str(res_name)[:5], str(atom_name)[:5],
        int(atom_seq) % 100000,
        float(xyz_nm[0]), float(xyz_nm[1]), float(xyz_nm[2]),
    )


def _pick_reserved_coion_species(system, topology, *, cation: bool) -> str:
    """从**这个盒子里已有的**离子里挑一个符号正确的物种名当 reserved dummy 的模板。

    不自己发明 moleculetype：新粒子要能被 .top 展开，就必须是这份 .top 里已经
    存在的分子类型。所以判据是"盒子里已经有、残基名在 co-ion 集合里、电荷符号对"，
    找不到就 fail closed —— 造一个 .top 里没有的分子类型只会在解析时炸得更晚更难查。
    """
    nb = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)),
        None,
    )
    if nb is None:
        raise RuntimeError("挑 reserved co-ion 物种需要 NonbondedForce，但体系里没有")
    want = 1.0 if cation else -1.0
    seen = {}
    for residue in topology.residues():
        name = str(residue.name).strip().upper()
        if name not in CO_ALCHEMICAL_ION_RESIDUE_NAMES:
            continue
        atoms = list(residue.atoms())
        if len(atoms) != 1:
            continue          # 多原子的"离子"不是单价 co-ion，跳过
        q = float(
            nb.getParticleParameters(int(atoms[0].index))[0].value_in_unit(
                unit.elementary_charge
            )
        )
        if q * want > 0.0:
            seen.setdefault(str(residue.name).strip(), (q, str(atoms[0].name)))
    if not seen:
        raise RuntimeError(
            "复合物盒里没有符号正确的单价离子可以当 reserved co-ion dummy 的模板"
            f"（需要 {'阳' if cation else '阴'}离子）。charge-transfer 要求在建系时"
            "预留一个中性 ion-shaped dummy；这份 .top 里连同号离子的 moleculetype "
            "都没有，无法派生 —— 请在建系阶段加入相应离子（哪怕只有 1 个）。"
        )
    # 名字确定性地取字典序最小的一个，避免同一份输入两次派生出不同的 .top。
    return sorted(seen)[0]


def _derive_gromacs_inputs_with_reserved_coions(
    *,
    gro_file: str,
    top_file: str,
    output_dir: str,
    system,
    topology,
    positions,
    box_vectors,
    ligand_indices: Sequence[int],
    count: int,
    cation: bool,
) -> Tuple[str, str, Dict[str, Any]]:
    """派生一对带 reserved co-ion dummy 的 GROMACS 输入，返回 `(gro, top, 报告)`。

    🔑🔑 [2026-09-16] **复合物腿缺的就是这一步。** 溶剂腿的盒子是本流程自己建的，
    所以能直接在 `Modeller` 上"摘一个水、加一个 dummy"；复合物腿的体系来自用户的
    `.gro/.top`，`GromacsTopFile` 的粒子表由 `[ molecules ]` 决定 —— 光改 OpenMM
    Topology 不会让 System 多出一个粒子。先前这条路径**根本没有插入步骤**，
    于是带电配体一进 `co_alchemical_ion_builder_identity_payload` 的数量契约就炸
    （"找到 0 个中性 ion-shaped dummy，但配体净电荷 +1 e 需要 1 个"）。

    做法是**派生一对新输入**（原始文件一个字节都不动），与溶剂腿同一套物理：
      ① 用**同一个**选点函数挑离配体最远、彼此也够远的水（`_select_reserved_coion_water_sites`）；
      ② `.gro` 删掉这些水的原子行，在**文件末尾**追加 `count` 个离子原子，
         坐标就是被摘掉的氧原子原位（零额外体积冲突）；
      ③ `.top` 的 `[ molecules ]` 里水数减 `count`，末尾追加一行 `<离子> count`。
    ③ 的"追加在末尾"与 ② 的"追加在末尾"必须同时成立：GROMACS 的原子顺序就是
    `[ molecules ]` 的展开顺序，两边错位就是静默的参数张冠李戴。重复的分子名在
    `[ molecules ]` 里是合法的（OpenMM `_processMolecule` 逐行 append），所以不必
    去动原有那行离子计数、也就不必在文件中间插原子。

    电荷：模板给的是 ±1，调用方必须在建完 System 之后显式清零（§2.2 要求 λ=1 端
    是"中性但保留 LJ 的 ion-shaped dummy"）。这里不碰电荷。
    """
    if count <= 0:
        raise ValueError("count 必须为正")
    species = _pick_reserved_coion_species(system, topology, cation=cation)
    pos_nm = np.asarray(
        positions.value_in_unit(unit.nanometer)
        if hasattr(positions, "value_in_unit") else positions,
        dtype=np.float64,
    )
    box_nm = np.asarray(
        [v.value_in_unit(unit.nanometer) for v in box_vectors]
        if hasattr(box_vectors[0], "value_in_unit") else box_vectors,
        dtype=np.float64,
    )

    water_residues = [
        res for res in topology.residues()
        if str(res.name).strip().upper() in WATER_MOLECULE_NAMES
    ]
    oxygen_to_residue = {}
    for res in water_residues:
        atoms = list(res.atoms())
        o_atom = next((a for a in atoms if a.element == app.element.oxygen), atoms[0])
        oxygen_to_residue[int(o_atom.index)] = res
    chosen_oxygens = _select_reserved_coion_water_sites(
        positions_nm=pos_nm,
        box_nm=box_nm,
        ligand_atom_indices=ligand_indices,
        water_oxygen_indices=sorted(oxygen_to_residue),
        count=count,
    )
    drop_indices = {
        int(a.index)
        for idx in chosen_oxygens
        for a in oxygen_to_residue[idx].atoms()
    }
    water_resname = str(
        next(iter(oxygen_to_residue[chosen_oxygens[0]].atoms())).residue.name
    ).strip()

    # ---- .gro ----
    raw = open(gro_file, encoding="utf-8").read().splitlines()
    n_declared = int(raw[1].split()[0])
    atom_lines = raw[2:2 + n_declared]
    if len(atom_lines) != n_declared:
        raise RuntimeError(
            f"{gro_file} 声明 {n_declared} 个原子，实际只有 {len(atom_lines)} 行"
        )
    if n_declared != topology.getNumAtoms():
        raise RuntimeError(
            f"坐标文件原子数 {n_declared} 与拓扑 {topology.getNumAtoms()} 不一致；"
            "拒绝在两者对不上的输入上派生 co-ion"
        )
    box_line = raw[2 + n_declared]
    if any(len(line) > 50 for line in atom_lines[:8]):
        raise RuntimeError(
            f"{gro_file} 看起来带速度列（行宽 > 50）。派生 co-ion 只支持纯坐标 "
            ".gro —— 带速度时删/加原子还要同步速度块，不做静默截断。"
        )
    kept = [ln for i, ln in enumerate(atom_lines) if i not in drop_indices]
    max_res_seq = max(
        (int(ln[:5]) for ln in atom_lines if ln[:5].strip().isdigit()), default=0
    )
    new_lines = []
    for i, ln in enumerate(kept):
        new_lines.append(ln[:15] + "%5d" % ((i + 1) % 100000) + ln[20:])
    for k, ox in enumerate(chosen_oxygens):
        new_lines.append(
            _gro_atom_line(
                max_res_seq + 1 + k, species, species.capitalize()[:5],
                len(new_lines) + 1, pos_nm[ox],
            )
        )
    derived_dir = os.path.join(output_dir, RESERVED_COION_DERIVED_DIRNAME)
    os.makedirs(derived_dir, exist_ok=True)
    out_gro = os.path.join(derived_dir, os.path.basename(gro_file))
    with open(out_gro, "w", encoding="utf-8") as fh:
        fh.write(raw[0] + "\n")
        fh.write("%d\n" % len(new_lines))
        fh.write("\n".join(new_lines) + "\n")
        fh.write(box_line + "\n")

    # ---- .top ----
    top_lines = open(top_file, encoding="utf-8").read().splitlines()
    mol_start = None
    for i, ln in enumerate(top_lines):
        if re.match(r"^\s*\[\s*molecules\s*\]\s*$", ln, flags=re.IGNORECASE):
            mol_start = i
            break
    if mol_start is None:
        raise RuntimeError(f"{top_file} 里没有 [ molecules ] 段，无法派生 co-ion")
    mol_end = len(top_lines)
    for i in range(mol_start + 1, len(top_lines)):
        if top_lines[i].lstrip().startswith("["):
            mol_end = i
            break
    water_line_idx = None
    for i in range(mol_start + 1, mol_end):
        fields = top_lines[i].split(";")[0].split()
        if len(fields) >= 2 and fields[0].strip().upper() in WATER_MOLECULE_NAMES:
            water_line_idx = i
    if water_line_idx is None:
        raise RuntimeError(
            f"{top_file} 的 [ molecules ] 里找不到水分子行，无法为 co-ion 腾出名额"
        )
    w_fields = top_lines[water_line_idx].split()
    w_count = int(w_fields[1])
    if w_count - count < 0:
        raise RuntimeError(f"水分子只有 {w_count} 个，摘不出 {count} 个")
    top_lines[water_line_idx] = "%-16s %d" % (w_fields[0], w_count - count)
    # 末尾追加，而不是去改已有的离子计数行：原子顺序=[molecules]展开顺序，
    # 追加在末尾正好对上 .gro 里追加在末尾的那些原子。
    insert_at = mol_end
    while insert_at > mol_start + 1 and not top_lines[insert_at - 1].strip():
        insert_at -= 1
    top_lines.insert(
        insert_at,
        "%-16s %d                ; [2026-09-16] reserved co-alchemical ion dummy"
        % (species, count),
    )
    out_top = os.path.join(derived_dir, os.path.basename(top_file))
    with open(out_top, "w", encoding="utf-8") as fh:
        fh.write("\n".join(top_lines) + "\n")

    report = {
        "species": species,
        "count": int(count),
        "cation": bool(cation),
        "water_residue_name": water_resname,
        "removed_water_oxygen_indices": [int(i) for i in chosen_oxygens],
        "removed_atom_count": len(drop_indices),
        "n_atoms_before": int(n_declared),
        "n_atoms_after": len(new_lines),
        "source_gro": os.path.abspath(gro_file),
        "source_top": os.path.abspath(top_file),
        "derived_gro": os.path.abspath(out_gro),
        "derived_top": os.path.abspath(out_top),
    }
    with open(os.path.join(derived_dir, RESERVED_COION_REPORT_BASENAME), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
    return out_gro, out_top, report


def _count_reserved_neutral_coion_candidates(system, topology) -> int:
    """按**数量契约同一把尺子**数一数现成的中性 ion-shaped dummy 有几个。

    判据与 `abfe_core.co_alchemical_ion_builder_identity_payload` /
    `ibs_engine._identify_reserved_neutral_co_ions` 完全一致（残基名在离子集合里
    且电荷严格为 0），所以"这里数出 N 个"等价于"那道契约会看到 N 个"。
    """
    nb = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)),
        None,
    )
    if nb is None:
        return 0
    n = 0
    for atom in topology.atoms():
        if str(atom.residue.name).strip().upper() not in CO_ALCHEMICAL_ION_RESIDUE_NAMES:
            continue
        q = float(
            nb.getParticleParameters(int(atom.index))[0].value_in_unit(
                unit.elementary_charge
            )
        )
        if abs(q) <= TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
            n += 1
    return n


def _total_system_charge_e(system) -> float:
    nb = next(
        (f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)),
        None,
    )
    if nb is None:
        raise RuntimeError("体系里没有 NonbondedForce，算不出总电荷")
    return float(
        sum(
            nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
            for i in range(nb.getNumParticles())
        )
    )


def _ensure_complex_reserved_coions(
    *,
    system,
    topology,
    positions,
    box_vectors,
    ligand_indices,
    gro_file: str,
    top_file: str,
    ligand_resname: str,
    gmx_include_dir: Optional[str],
    output_dir: str,
    ligand_net_charge_e: int,
):
    """让**复合物腿**也走完 reserved co-ion 的构建，返回重建后的五元组 + 报告。

    🔑🔑 [2026-09-16] 先前这条路径直接进数量契约校验，而**没有任何插入步骤** ——
    溶剂腿（`build_and_cache_solvent_leg`）早就有插入 + 清零，复合物腿没有。
    带电配体（thrombin_ligand1，+1 e）因此三个 repeat 都停在同一句
    "找到 0 个中性 ion-shaped dummy，但配体净电荷 +1 e 需要 1 个"。

    三条分支，都不静默：
      · 已经有正好 |q_L| 个 ⟹ **一个字节都不动**（手工预留 dummy 的输入照旧）；
      · 一个都没有 ⟹ 派生一对新 `.gro/.top`、重建、清零、逐项复核；
      · 数量对不上（既不是 0 也不是 |q_L|）⟹ fail closed，不猜哪个才是 dummy。

    复核项（这一步不做，后面炸的地方离原因就很远了）：
      总电荷守恒 / 配体净电荷不变 / 粒子数账目 / dummy 电荷严格为 0 /
      坐标与拓扑等长。
    """
    need = abs(int(ligand_net_charge_e))
    have = _count_reserved_neutral_coion_candidates(system, topology)
    if have == need:
        return system, topology, positions, box_vectors, ligand_indices, None
    if have != 0:
        raise RuntimeError(
            f"复合物体系里已经有 {have} 个中性 ion-shaped dummy，但配体净电荷 "
            f"{ligand_net_charge_e:+d} e 需要 {need} 个。既不是「全都要派生」"
            "也不是「已经齐了」，拒绝猜测哪几个才是 reserved dummy —— 请修输入。"
        )

    q_total_before = _total_system_charge_e(system)
    q_lig_before = float(_compute_ligand_net_charge(system, ligand_indices))
    n_before = system.getNumParticles()
    log.info(
        "  配体净电荷 %+d e：复合物腿缺 %d 个 reserved co-ion dummy，"
        "派生带 dummy 的 GROMACS 输入（原始输入只读）",
        ligand_net_charge_e, need,
    )
    derived_gro, derived_top, report = _derive_gromacs_inputs_with_reserved_coions(
        gro_file=gro_file,
        top_file=top_file,
        output_dir=output_dir,
        system=system,
        topology=topology,
        positions=positions,
        box_vectors=box_vectors,
        ligand_indices=ligand_indices,
        count=need,
        cation=ligand_net_charge_e > 0,
    )
    system, topology, positions, box_vectors, ligand_indices = (
        build_system_from_gromacs(
            derived_gro, derived_top, ligand_resname, gmx_include_dir
        )
    )
    # 派生出来的离子带模板电荷 ±1；§2.2 要求 λ=1 端是"中性但保留 LJ"，显式清零。
    # 它们是 `[ molecules ]` 末尾追加的那几个 ⟹ 就是粒子表最后 need 个。
    nb = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    dummy_indices = list(range(system.getNumParticles() - need,
                               system.getNumParticles()))
    atoms = list(topology.atoms())
    for idx in dummy_indices:
        name = str(atoms[idx].residue.name).strip().upper()
        if name not in CO_ALCHEMICAL_ION_RESIDUE_NAMES:
            raise RuntimeError(
                f"派生后粒子 {idx} 的残基名是 {name!r}，不在 co-ion 集合里 —— "
                "说明 `.gro` 的原子顺序与 `[ molecules ]` 展开顺序对不上，"
                "这会静默把参数张冠李戴，拒绝继续。"
            )
        _, sigma, epsilon = nb.getParticleParameters(idx)
        nb.setParticleParameters(idx, 0.0 * unit.elementary_charge, sigma, epsilon)

    # ---- 逐项复核 ----
    q_total_after = _total_system_charge_e(system)
    if abs(q_total_after - q_total_before) > TOTAL_CHARGE_CONSERVATION_TOLERANCE_E:
        raise RuntimeError(
            f"派生 reserved co-ion 后总电荷从 {q_total_before:+.6f} e 变成 "
            f"{q_total_after:+.6f} e。dummy 在 λ=1 端电荷为 0，本该逐位守恒。"
        )
    q_lig_after = float(_compute_ligand_net_charge(system, ligand_indices))
    if abs(q_lig_after - q_lig_before) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
        raise RuntimeError(
            f"派生后配体净电荷从 {q_lig_before:+.6f} e 变成 {q_lig_after:+.6f} e"
        )
    n_expected = n_before - report["removed_atom_count"] + need
    if system.getNumParticles() != n_expected:
        raise RuntimeError(
            f"派生后粒子数 {system.getNumParticles()} != 预期 {n_expected}"
            f"（原 {n_before} − 摘掉 {report['removed_atom_count']} + {need}）"
        )
    if len(positions) != system.getNumParticles():
        raise RuntimeError(
            f"派生后坐标数 {len(positions)} 与粒子数 "
            f"{system.getNumParticles()} 不一致"
        )
    have_after = _count_reserved_neutral_coion_candidates(system, topology)
    if have_after != need:
        raise RuntimeError(
            f"派生后中性 ion-shaped dummy 数为 {have_after}，期望 {need}"
        )
    report = dict(report, reserved_coion_indices=[int(i) for i in dummy_indices],
                  total_charge_e=q_total_after,
                  ligand_net_charge_e=int(round(q_lig_after)))
    log.info(
        "  [OK] 复合物腿 reserved co-ion 已就位：%d 个 %s 形 dummy（index=%s），"
        "摘掉 %d 个水原子，总电荷 %+.6f e 守恒",
        need, report["species"], dummy_indices, report["removed_atom_count"],
        q_total_after,
    )
    return system, topology, positions, box_vectors, ligand_indices, report


