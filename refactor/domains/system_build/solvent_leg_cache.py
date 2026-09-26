# ================= runabfe.py =================
# 完整替换 build_and_cache_solvent_leg 函数
def build_and_cache_solvent_leg(
    output_dir,
    topology,
    positions,
    ligand_indices,
    ligand_resname,
    ligand_ffxml: Optional[str] = None,
    top_file: Optional[str] = None,
    gmx_include_dir: Optional[str] = None,
    ionic_strength_molar: float = DEFAULT_SOLVENT_IONIC_STRENGTH_MOLAR,
    cache_identity: Optional[Dict] = None,
    padding_nm: float = SOLVENT_PADDING_NM,
    charge_treatment: Optional[str] = None,
):
    """
    🔑 终极纯净版：彻底抛弃 mmCIF，直接从原始 .top 提取配体并自动加水。

    [B4，2026-08-05 落地] `charge_treatment=co_alchemical_charge_transfer` 且配体带
    净电荷时，本 builder 在 `addSolvent()` 产出的普通盐/中和离子之外，额外插入
    `|q_L|` 个建系时预留的**中性 ion-shaped dummy** 粒子（§4.1/§2.2）：电荷强制为 0，
    LJ/mass 取同号物理离子模板（配体净电荷 >0 → Na⁺ 形状，<0 → Cl⁻ 形状），放置在
    离配体最远的 bulk-water 位置（§4.4）。

    这里**只负责让粒子存在**，不重新实现选择/身份/restraint 逻辑——那些已经是
    leg-agnostic 的既有实现：`ibs_engine.select_co_alchemical_ion_once` 靠"残基名在
    离子集合里且电荷严格为 0"这个坐标无关的判据认出它（`_identify_reserved_neutral_co_ions`），
    `abfe_core.build_co_alchemical_ion_identity` 冻结成可落盘身份。复合物腿的输入
    拓扑靠建系脚本预先带上同一种粒子，两条腿殊途同归，不需要两套判据。
    """
    padding_nm = float(padding_nm)
    log.info("正在构建溶剂相 (配体腿) 系统并生成缓存...")
    from openmm.app import Modeller, ForceField
    os.makedirs(output_dir, exist_ok=True)

    if cache_identity is None:
        raise ValueError("拒绝构建无 complex-leg/配体参数身份的溶剂腿缓存")
    topology_ligand_count = sum(
        1 for atom in topology.atoms() if atom.residue.name == ligand_resname
    )
    if topology_ligand_count != len(ligand_indices):
        raise ValueError(
            f"complex topology 中配体原子数 ({topology_ligand_count}) 与索引数 "
            f"({len(ligand_indices)}) 不一致"
        )

    # 1. 提取配体坐标，仅用于确认输入坐标/索引有效。盒子由 OpenMM padding
    # 语义生成，确保配体每一侧都有指定厚度的溶剂。
    if hasattr(positions, "value_in_unit"):
        pos_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    else:
        pos_nm = np.asarray(positions, dtype=np.float64)
    
    lig_coords = pos_nm[ligand_indices]
    if lig_coords.ndim != 2 or lig_coords.shape[1] != 3 or not np.all(np.isfinite(lig_coords)):
        raise ValueError("配体坐标必须是有限的 (N, 3) nm 数组")

    # 2. 🔑 降维打击：彻底抛弃 mmCIF，直接从原始 .top 读取 100% 纯净的配体拓扑
    if not top_file or not os.path.isfile(top_file):
        log.error("[ERR] 构建溶剂相必须提供原始复合物 .top 文件以提取纯净配体键连接")
        return False
        
    log.info("  彻底抛弃 mmCIF，从原始 .top 重建 100% 纯净配体拓扑...")
    # 走唯一入口：溶剂腿也会读复合物 .top，同样可能撞上 `[ pairs ]` funct 2。
    # （首次修 funct-2 时只接了 build_system_from_gromacs，这里就漏了。）
    gmx_top = load_gromacs_topology_for_openmm(top_file, includeDir=gmx_include_dir)
    stable_names = _build_stable_ligand_atom_names(gmx_top, ligand_resname)
    
    # 获取 .top 中的配体原子索引
    top_lig_indices = [a.index for a in gmx_top.topology.atoms() if a.residue.name == ligand_resname]
    if not top_lig_indices:
        log.error(f"[ERR] 在 .top 文件中未找到配体残基 {ligand_resname}")
        return False
        
    # 使用 .top 的绝对正确拓扑 + 全局坐标创建 Modeller
    modeller = Modeller(gmx_top.topology, positions)
    
    # 删掉所有环境原子（蛋白、水、离子），只留配体
    atoms_to_delete = [a for a in modeller.topology.atoms() if a.index not in top_lig_indices]
    modeller.delete(atoms_to_delete)
    log.info("  [OK] 纯净配体拓扑构建完成 (键连接 100% 忠于 .top，无假键)")

    # 3. 用 .top 的稳定名字映射重命名，确保拓扑与 XML 完全同序对齐
    lig_atoms_in_mod = [a for a in modeller.topology.atoms() if a.residue.name == ligand_resname]
    if len(lig_atoms_in_mod) != len(top_lig_indices):
        log.error("[ERR] 配体原子数不匹配：Modeller=%d, Topology=%d", len(lig_atoms_in_mod), len(top_lig_indices))
        return False
    for atom, top_idx in zip(lig_atoms_in_mod, top_lig_indices):
        atom.name = stable_names[int(top_idx)]
    log.info("  [OK] 配体原子名已按 .top 稳定映射重命名并对齐 XML")

    # 4. 复用/生成配体 XML + TIP3P 水
    ffxml_path = resolve_ligand_ffxml(output_dir, ligand_resname, explicit_path=ligand_ffxml)
    if ffxml_path is None:
        ffxml_path = generate_ligand_xml_from_top(
            top_file=top_file,
            ligand_resname=ligand_resname,
            output_dir=output_dir,
            gmx_include_dir=gmx_include_dir,
        )
    if ffxml_path is None:
        log.error("[ERR] 溶剂相构建失败：未找到或无法生成配体力场 XML")
        return False

    try:
        # 🔑 水模型不硬编码，从复合物 .top 反推，保证两腿同模型。
        water_xml, water_itp = resolve_water_model_xml(top_file)
        log.info(
            "  使用 Amber14 + %s (继承自 %s) + 配体 XML 构建溶剂相: %s",
            water_xml,
            water_itp,
            ffxml_path,
        )
        ff = ForceField("amber14-all.xml", water_xml, ffxml_path)
        
        ionic_strength_molar = float(ionic_strength_molar)
        if ionic_strength_molar < 0.0:
            raise ValueError("solvent_ionic_strength_molar 不能为负")
        # 🔑 盒子显式给出，不用 padding=。addSolvent 的 padding 分支曾经产出
        # box = 2*padding 的 3.000 nm 盒（溶质尺寸贡献为 0），而且 OpenMM 7.7+
        # 的 padding 默认给菱形十二面体、不是立方——两者都不是这里想要的。
        box_edge_nm, lig_extent_nm = solvent_box_edge_nm(
            lig_coords, padding_nm=padding_nm
        )
        log.info(
            "  溶剂盒: 配体最长轴 %.4f nm + 2×%.2f nm padding → 立方盒 %.4f nm",
            lig_extent_nm,
            padding_nm,
            box_edge_nm,
        )
        # Match the ionized complex leg's ~0.15 M NaCl instead of silently
        # building a pure-water solvent leg. OpenMM adds salt pairs for the
        # requested ionic strength and any extra counterions needed to neutralize.
        modeller.addSolvent(
            ff,
            boxSize=Vec3(box_edge_nm, box_edge_nm, box_edge_nm) * unit.nanometer,
            positiveIon="Na+",
            negativeIon="Cl-",
            ionicStrength=ionic_strength_molar * unit.molar,
            neutralize=True,
        )

        # fail closed：确认 OpenMM 真的按我们给的盒子建，而不是又自己推了一个。
        realized_vecs = modeller.topology.getPeriodicBoxVectors()
        if realized_vecs is None:
            raise RuntimeError("addSolvent 之后拓扑没有周期盒向量")
        realized_edges = np.linalg.norm(
            np.array(
                [v.value_in_unit(unit.nanometer) for v in realized_vecs], dtype=float
            ),
            axis=1,
        )
        if not np.allclose(realized_edges, box_edge_nm, atol=1.0e-6):
            raise RuntimeError(
                f"溶剂盒构建结果与请求不符：请求 {box_edge_nm:.6f} nm 立方，"
                f"实际边长 {[round(float(x), 6) for x in realized_edges]} nm"
            )

        # [B4] charge-transfer 路线：在**最终** createSystem 之前插入 reserved
        # co-ion dummy——下游只认"残基名在离子集合里且电荷严格为 0"的粒子
        # （`ibs_engine._identify_reserved_neutral_co_ions`），必须真的进最终的
        # NonbondedForce，不能是事后贴上去的。
        reserved_coion_indices: List[int] = []
        reserved_coion_cation: Optional[bool] = None
        reserved_coion_ligand_net_charge_e = 0
        if charge_treatment == CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER:
            probe_system = ff.createSystem(
                modeller.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=SOLVENT_NONBONDED_CUTOFF_NM * unit.nanometer,
                constraints=app.HBonds,
                rigidWater=True,
            )
            probe_nb = next(
                f for f in probe_system.getForces()
                if isinstance(f, openmm.NonbondedForce)
            )
            probe_lig_indices = [
                a.index for a in modeller.topology.atoms()
                if a.residue.name == ligand_resname
            ]
            raw_q = sum(
                probe_nb.getParticleParameters(i)[0].value_in_unit(
                    unit.elementary_charge
                )
                for i in probe_lig_indices
            )
            if not np.isfinite(float(raw_q)):
                raise RuntimeError(
                    f"配体净电荷为 NaN/Inf ({raw_q!r})，拒绝确定 reserved co-ion 数量"
                )
            lig_net_charge = int(round(raw_q))
            if abs(raw_q - lig_net_charge) > LIGAND_NET_CHARGE_INTEGER_TOLERANCE_E:
                raise RuntimeError(
                    f"配体净电荷 {raw_q:+.6f} e 不接近整数（容差 1e-3 e），"
                    "无法确定需要几个 reserved co-ion dummy（§2.2）。"
                )
            del probe_system
            reserved_coion_ligand_net_charge_e = lig_net_charge
            if lig_net_charge != 0:
                reserved_coion_cation = lig_net_charge > 0
                log.info(
                    "  配体净电荷 %+d e：插入 %d 个 %s 形 reserved co-ion dummy（B4，§4.1）",
                    lig_net_charge,
                    abs(lig_net_charge),
                    "Na⁺" if reserved_coion_cation else "Cl⁻",
                )
                reserved_coion_indices = _insert_reserved_coalchemical_ion_dummies(
                    modeller,
                    count=abs(lig_net_charge),
                    cation=reserved_coion_cation,
                    ligand_atom_indices=probe_lig_indices,
                )

        system = ff.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=SOLVENT_NONBONDED_CUTOFF_NM * unit.nanometer,
            constraints=app.HBonds,
            rigidWater=True,
        )

        if reserved_coion_indices:
            # 模板给的是 ±1（Na⁺/Cl⁻ residue template）；charge-transfer 要求 λ=1
            # 端它是"中性但保留 LJ 的 ion-shaped dummy"（§2.2），这里显式清零。
            nb_force = next(
                f for f in system.getForces()
                if isinstance(f, openmm.NonbondedForce)
            )
            for idx in reserved_coion_indices:
                _, sigma, epsilon = nb_force.getParticleParameters(idx)
                nb_force.setParticleParameters(
                    idx, 0.0 * unit.elementary_charge, sigma, epsilon
                )
            log.info(
                "  [OK] %d 个 reserved co-ion dummy 电荷已清零 (indices=%s)",
                len(reserved_coion_indices),
                reserved_coion_indices,
            )
    except Exception as e:
        log.error("[ERR] 溶剂相构建失败 (配体 XML 未正确接入或模板不匹配): %s", e)
        log.error("当前使用的配体 XML: %s", ffxml_path)
        return False

    # 5. 获取溶剂相中的新配体索引并保存缓存
    new_lig_indices = [atom.index for atom in modeller.topology.atoms() if atom.residue.name == ligand_resname]
    residue_names = [str(residue.name).upper() for residue in modeller.topology.residues()]
    na_count = sum(name in {"NA", "NA+", "SOD"} for name in residue_names)
    cl_count = sum(name in {"CL", "CL-", "CLA"} for name in residue_names)
    reserved_coion_builder_identity = co_alchemical_ion_builder_identity_payload(
        system=system,
        topology=modeller.topology,
        charge_treatment=charge_treatment,
        ligand_net_charge_e=reserved_coion_ligand_net_charge_e,
    )
    reserved_na_count = sum(
        1 for idx in reserved_coion_indices
        if str(list(modeller.topology.atoms())[idx].residue.name).upper()
        in {"NA", "NA+", "SOD"}
    )
    reserved_cl_count = sum(
        1 for idx in reserved_coion_indices
        if str(list(modeller.topology.atoms())[idx].residue.name).upper()
        in {"CL", "CL-", "CLA"}
    )
    ordinary_na_count = int(na_count - reserved_na_count)
    ordinary_cl_count = int(cl_count - reserved_cl_count)
    if ionic_strength_molar > 0.0 and (
        ordinary_na_count <= 0 or ordinary_cl_count <= 0
    ):
        log.error(
            "[ERR] 请求 %.3f M NaCl，但生成拓扑中普通盐离子计数异常: "
            "ordinary_Na=%d, ordinary_Cl=%d (total Na=%d, Cl=%d, reserved=%d)",
            ionic_strength_molar,
            ordinary_na_count,
            ordinary_cl_count,
            na_count,
            cl_count,
            len(reserved_coion_indices),
        )
        return False
    
    sol_xml = os.path.join(output_dir, "system_solvent.xml")
    sol_cif = os.path.join(output_dir, "topology_solvent.cif")
    sol_idx = os.path.join(output_dir, "ligand_indices_solvent.json")
    sol_box = os.path.join(output_dir, "box_vectors_solvent.npy")
    sol_manifest = os.path.join(output_dir, "solvent_cache_manifest.json")
    
    with open(sol_xml, "w", encoding="utf-8") as f:
        f.write(XmlSerializer.serialize(ensure_owned_system(system)))
    app.PDBxFile.writeFile(modeller.topology, modeller.positions, sol_cif)
    with open(sol_idx, "w", encoding="utf-8") as f:
        json.dump({"ligand_indices": new_lig_indices}, f)
    
    box_vecs = modeller.topology.getPeriodicBoxVectors()
    if box_vecs:
        np.save(sol_box, np.array([v.value_in_unit(unit.nanometer) for v in box_vecs]))
    manifest_payload = {
                "protocol_version": SOLVENT_CACHE_PROTOCOL_VERSION,
                **cache_identity,
                "padding_nm": padding_nm,
                "ligand_longest_axis_nm": lig_extent_nm,
                "box_edge_nm": box_edge_nm,
                "nonbonded_cutoff_nm": SOLVENT_NONBONDED_CUTOFF_NM,
                "ionic_strength_molar": ionic_strength_molar,
                "positive_ion": "Na+",
                "negative_ion": "Cl-",
                "neutralize": True,
                "na_count": int(na_count),
                "cl_count": int(cl_count),
                "ordinary_na_count": ordinary_na_count,
                "ordinary_cl_count": ordinary_cl_count,
                # [B4，§4.3/§4.5] 三类电荷来源不能混算：na_count/cl_count 是
                # 残基名总数（也包含下面这些 reserved dummy，因为它们复用同一个
                # 残基名模板）；ordinary_* 才是物理 NaCl 计数，reserved_coion_*
                # 是 alchemical co-ion 账目，不能把 dummy 错算成普通盐浓度。
                "charge_treatment": charge_treatment,
                "reserved_coion_ligand_net_charge_e": int(
                    reserved_coion_ligand_net_charge_e
                ),
                "reserved_coion_count": len(reserved_coion_indices),
                "reserved_coion_indices": list(reserved_coion_indices),
                "reserved_coion_cation": reserved_coion_cation,
                "system_xml_sha256": _sha256_file(sol_xml),
                "topology_sha256": _sha256_file(sol_cif),
                "ligand_indices_sha256": _sha256_file(sol_idx),
            }
    if reserved_coion_builder_identity is not None:
        manifest_payload["reserved_coion_builder_identity"] = (
            reserved_coion_builder_identity
        )
    with open(sol_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)
        
    box_lengths_nm = [
        float(np.linalg.norm(v.value_in_unit(unit.nanometer))) for v in box_vecs
    ]
    log.info(
        "[OK] 溶剂相缓存已保存 (padding %.2f nm, 盒长 %s nm, 原子数 %d, Na=%d, Cl=%d, %.3f M"
        "%s)",
        padding_nm,
        [round(x, 3) for x in box_lengths_nm],
        system.getNumParticles(),
        na_count,
        cl_count,
        ionic_strength_molar,
        (
            f", reserved_coion={len(reserved_coion_indices)}"
            if reserved_coion_indices
            else ""
        ),
    )
    return True

def load_native_system(
    output_dir,
    gro_file=None,
    top_file=None,
    gmx_include_dir=None,
    phase="complex",
    prefer_equilibrated: bool = True,
    expected_pre_equilibration_fingerprint: Optional[str] = None,
    require_bonded_topology: bool = False,
    reconstruct_bonds_from_system: bool = False,
):
    """从 output_dir 的缓存文件加载系统（跳过 GROMACS 解析）

    ## ⚠️ `topology.cif` 会丢掉非标准残基的键

    `app.PDBxFile.writeFile` **不写任何键记录**（写入端没有 `struct_conn` /
    `chem_comp_bond`），而读取端只能靠 `createStandardBonds()` 补**标准残基**
    （氨基酸/核酸/水）的键。于是 mmCIF 往返之后：

      * 脂质（`POPC` = `PA`+`PC`+`OL`）、配体（`MOL`）、离子的键**全部丢失**；
      * 而这里原本只校验**原子数**，键数不符照样通过。

    后果是实测出来的：`pre_equilibrate` 之前的「PBC 分子完整性修复」靠 topology 的
    键判断"什么算一个分子"，键丢了就把跨周期边界的脂质**逐段撕开** ——
    最小化后 PE = 4.1e13 kJ/mol、max|F| = 3.7e9（落在脂质尾链氢 `PA334/H8S`），
    几千步后变成一个没有上下文的 `Particle coordinate is NaN`。
    从 `.top` 重建拓扑（有键）时同一体系完全正常（PE ≈ −6.5e5）。

    `require_bonded_topology=True`（膜体系必须传）会：
      1. **优先从 `.top` 重建**拓扑，mmCIF 只在没有 `.top` 时兜底；
      2. 校验键数，覆盖不足即 fail closed。

    可溶路径默认仍走原逻辑（mmCIF 优先）——它的配体 `MOL` 同样丢键，属存量隐患，
    但改动会移动现有生产基线的预平衡起点（§7.7 / R7），单独立项评估。
    """
    paths = _cache_paths(output_dir, phase=phase)
    xml_path = paths["xml"]
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Native XML 缓存不存在: {xml_path}")
    log.info("从原生缓存加载系统 (%s): %s", phase, output_dir)

    # 1. System
    with open(xml_path, "r", encoding="utf-8") as f:
        xml_str = f.read()
    system = ensure_owned_system(XmlSerializer.deserialize(xml_str))
    log.info("  [OK] System 恢复 | 原子数: %d", system.getNumParticles())

    # 2. Ligand indices
    lig_path = paths["idx"]
    with open(lig_path, "r", encoding="utf-8") as f:
        lig_data = json.load(f)
    ligand_indices = lig_data["ligand_indices"]
    log.info("  [OK] 配体索引恢复: %d 原子", len(ligand_indices))

    # 3. Box vectors
    box_path = paths["box"]
    box_vectors = None
    if os.path.exists(box_path):
        box_nm = np.load(box_path)
        box_vectors = [Vec3(float(v[0]), float(v[1]), float(v[2])) for v in box_nm] * unit.nanometer
        log.info("  [OK] 盒子向量恢复")
        try:
            system.setDefaultPeriodicBoxVectors(*box_vectors)
        except Exception as e:
            log.warning("  [WARN] 设置默认盒子失败: %s", e)

    # 4. Topology 恢复
    #
    # 膜体系（require_bonded_topology）**优先从 .top 重建**：mmCIF 不保存非标准残基
    # 的键，而 PBC 分子完整性修复依赖键。详见本函数 docstring。
    top_path = paths["top"]
    topology = None
    cif = None

    if require_bonded_topology and top_file and os.path.exists(top_file):
        try:
            inc_dir = gmx_include_dir or find_gmx_include_dir()
            _rebuilt = load_gromacs_topology_for_openmm(
                top_file, includeDir=inc_dir, periodicBoxVectors=box_vectors
            )
            topology = _rebuilt.topology
            log.info(
                "  [OK] 拓扑从原始 TOP 重建（%d 原子, %d 键）—— 膜体系不用 mmCIF，"
                "因为它不保存非标准残基的键",
                topology.getNumAtoms(), topology.getNumBonds(),
            )
        except Exception as e:
            raise RuntimeError(
                f"膜体系要求从 .top 重建带键拓扑，但失败了：{e}\n"
                "   不能退回 mmCIF —— 它丢掉脂质/配体的键，会让 PBC 分子完整性修复"
                "把跨边界的分子撕开（实测最小化后 PE 到 1e13、随后 NaN）。"
            ) from e

    if topology is None and os.path.exists(top_path):
        try:
            cif = app.PDBxFile(top_path)
            topology = cif.topology
            n_cif = topology.getNumAtoms()
            n_sys = system.getNumParticles()
            if n_cif != n_sys:
                log.warning("  [WARN] mmCIF 拓扑原子数 (%d) 与 System (%d) 不匹配，已丢弃缓存", n_cif, n_sys)
                topology = None
            else:
                # 🔑 [PBC-01] 先删假键，再补缺键。mmCIF 往返两个方向都会错：
                # 链数 > 26 时 `_struct_conn` 解析歧义会**多造**键（brd4/ligand1
                # 实测 3 条，把一个水并进蛋白那个分子），而非标准残基的键会**丢**。
                # 判据与安全方向见 `prune_topology_bonds_unsupported_by_system`。
                # 一条都不用删时原样返回同一个对象 ⟹ 干净体系 topology_sha256 不变。
                topology = prune_topology_bonds_unsupported_by_system(
                    topology, system, log=log.warning
                )
                if reconstruct_bonds_from_system:
                    atoms = list(topology.atoms())
                    existing_pairs = {
                        tuple(sorted((int(bond.atom1.index), int(bond.atom2.index))))
                        for bond in topology.bonds()
                    }
                    recovered_pairs = set()
                    for force in system.getForces():
                        if not isinstance(force, openmm.HarmonicBondForce):
                            continue
                        for bond_index in range(force.getNumBonds()):
                            i, j, _length, _k = force.getBondParameters(bond_index)
                            recovered_pairs.add(tuple(sorted((int(i), int(j)))))
                    for constraint_index in range(system.getNumConstraints()):
                        i, j, _distance = system.getConstraintParameters(constraint_index)
                        recovered_pairs.add(tuple(sorted((int(i), int(j)))))
                    for i, j in sorted(recovered_pairs - existing_pairs):
                        topology.addBond(atoms[i], atoms[j])
                    log.info(
                        "  [OK] 从 System HarmonicBondForce/constraints 恢复 %d 条缺失拓扑连接",
                        len(recovered_pairs - existing_pairs),
                    )
                # 🔑 键数校验：只校验原子数是不够的。mmCIF 不保存非标准残基的键，
                # 读回时只有 createStandardBonds() 能补标准残基，脂质/配体/离子的键
                # 会静默丢失，而 PBC 分子完整性修复正是靠键判断分子边界。
                n_cif_bonds = topology.getNumBonds()
                n_sys_bonds = 0
                for _f in system.getForces():
                    if isinstance(_f, openmm.HarmonicBondForce):
                        n_sys_bonds = max(n_sys_bonds, _f.getNumBonds())
                # System 的键数 = HarmonicBondForce 键数 + 被 constraints 吸收的 X–H 键，
                # 所以 topology 键数应当 ≥ HarmonicBondForce 键数。明显更少就是丢了。
                if n_sys_bonds > 0 and n_cif_bonds < n_sys_bonds:
                    log.warning(
                        "  [WARN] mmCIF 拓扑只有 %d 个键，而 System 的 HarmonicBondForce 有 %d 个"
                        " —— mmCIF 不保存非标准残基（脂质/配体/离子）的键。"
                        "PBC 分子完整性修复依赖键，键丢了会把跨边界分子撕开。"
                        "已丢弃 mmCIF 拓扑，改从 .top 重建。",
                        n_cif_bonds, n_sys_bonds,
                    )
                    topology = None
                else:
                    log.info(
                        "  [OK] 拓扑从 mmCIF 缓存恢复 (原子数 %d, 键数 %d)",
                        n_cif, n_cif_bonds,
                    )
        except Exception as e:
            log.warning("  [WARN] mmCIF 拓扑加载失败: %s", e)

    # 降级方案：从原始 TOP 文件重建拓扑
    if topology is None and phase == "complex" and top_file and os.path.exists(top_file):
        try:
            inc_dir = gmx_include_dir or find_gmx_include_dir()
            top = load_gromacs_topology_for_openmm(top_file, includeDir=inc_dir)
            topology = top.topology
            log.info("  [OK] 拓扑从原始 TOP 文件重建")
        except Exception as e:
            log.warning("  [WARN] TOP 拓扑重建失败: %s", e)

    if topology is None:
        raise RuntimeError(
            "无法恢复拓扑：mmCIF 缓存损坏且无有效 TOP 文件。\n"
            "   解决方案：请确保命令中包含 --top 参数，或删除 output 目录后重新运行。"
        )

    # 🔑 无论拓扑从哪条路径恢复，都必须带上周期盒矢量。
    #
    # `app.DCDFile` 写每帧的 unitcell 时读的是 **topology** 的盒矢量
    # （`dcdfile.py:155`：`boxVectors = self._topology.getPeriodicBoxVectors()`；
    # 为 None 就整段不写，header 的 boxFlag 也是 0）。拓扑没盒矢量 → DCD 没 unitcell
    # → 下游任何要算 APL / 盒序列的分析（§9 膜质量门）直接失效。
    #
    # mmCIF 恢复的拓扑自带 `_cell`，所以历史上没暴露；但从 `.top` 重建时
    # `GromacsTopFile` 只在显式传 `periodicBoxVectors` 时才设盒——这是一处
    # **fail-open**：少传一个参数，产出的轨迹静默缺少盒信息。
    if box_vectors is not None and topology.getPeriodicBoxVectors() is None:
        topology.setPeriodicBoxVectors(box_vectors)
        log.info("  [OK] 已把盒矢量写入拓扑（DCD 的 unitcell 依赖它）")

    # 5. Positions (保持不变)
    positions = None
    equil_dcd = os.path.join(paths["runtime_dir"], "pre_equilibration.dcd")
    # ------------------------------------------------------------------
    # 🔑 [2026-08-31] 这里原来是
    #     equil_cache_trusted = bool(
    #         prefer_equilibrated and expected_...fingerprint is not None
    #         and equilibrium_is_done(paths["runtime_dir"],
    #                                expected_fingerprint=...))
    # 而 [P1-07] 之后 `equilibrium_is_done()` 把"目标 Simulation"变成硬前提：
    # 没传 simulation 就 `log.warning("缺少目标 Simulation ...")` 并 `return False`
    # （见该函数）。六个调用点里只有这一处没传 —— 于是这个表达式**恒为 False**，
    # 下面读 pre_equilibration.dcd 末帧的整段是死代码，而且每次 resume 都会打出
    # 一条与真实情况无关的"缺少目标 Simulation"警告。
    #
    # ⚠️ **不要**用"在这里造一个探针 Simulation"来修它。坐标经
    # `positions_sha256` 进 `_pre_equilibration_fingerprint()`（abfe_pipeline.py，
    # 见其 docstring："The initial coordinates, periodic box and requested step
    # budget are part of the identity"）。一旦这里真的把平衡末帧读进 positions，
    # 后面 6243/6466 处重算的指纹就与落盘的那份（用初始坐标算的）不符，
    # `pre_equilibrate()` 也会判自己的 checkpoint 失效 —— 结果是**每次 resume
    # 都从零重跑预平衡**，比现在什么都不做更糟。
    #
    # 现状是自洽的：坐标始终取身份绑定的初始坐标，预平衡的复用完全交给
    # pipeline 侧那两处带 simulation 的严格判定。所以这里显式写死 False 并留下
    # 依据，既不改行为、也不再发误导性警告。真要复用平衡末帧，先另开一项把坐标
    # 从预平衡指纹里摘出去（那是协议变更，不是这里能顺手做的）。
    # ------------------------------------------------------------------
    equil_cache_trusted = False
    if prefer_equilibrated and expected_pre_equilibration_fingerprint is None:
        log.info(
            "  未提供完整预平衡 fingerprint，不自动读取 DCD；"
            "先恢复身份绑定的初始坐标，由 pipeline 决定是否续用预平衡"
        )
    if equil_cache_trusted and os.path.exists(equil_dcd) and os.path.getsize(equil_dcd) > 212:
        try:
            import mdtraj as md
            md_top = md.Topology.from_openmm(topology)
            traj = md.load(equil_dcd, top=md_top)
            if len(traj) > 0:
                positions = traj.xyz[-1] * unit.nanometer
                if len(traj.unitcell_vectors) > 0:
                    box_vectors = traj.unitcell_vectors[-1] * unit.nanometer
                    system.setDefaultPeriodicBoxVectors(*box_vectors)
                log.info("  [OK] 坐标从预平衡 DCD 最后一帧恢复")
        except Exception as e:
            log.warning("  [WARN] DCD 坐标加载失败: %s", e)

    if positions is None and cif is not None:
        try:
            positions = cif.positions
            log.info("  [OK] 坐标从 mmCIF 缓存恢复")
        except Exception as e:
            log.warning("  [WARN] mmCIF 坐标恢复失败: %s", e)

    if positions is None and phase == "complex" and gro_file and os.path.exists(gro_file):
        gro = app.GromacsGroFile(gro_file)
        positions = gro.positions
        log.info("  [WARN] 坐标回退到 GRO 初始值")

    if positions is None:
        raise RuntimeError("无法恢复坐标，请检查输入文件或预平衡结果")

    log.info("[OK] 原生缓存加载完成")
    return system, topology, positions, box_vectors, ligand_indices

