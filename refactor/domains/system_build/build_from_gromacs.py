# ---------------------------------------------------------------------------
# GROMACS 系统构建（仅首次使用，之后都会被缓存替代）
# ---------------------------------------------------------------------------
def build_system_from_gromacs(
    gro_file: str,
    top_file: str,
    ligand_resname: str,
    gmx_include_dir: Optional[str] = None,
) -> Tuple[openmm.System, app.Topology, list, list, List[int]]:
    """从 GROMACS 文件构建 OpenMM 系统（仅用于首次创建缓存）"""
    if not os.path.exists(gro_file):
        raise FileNotFoundError(f"GRO 文件不存在: {gro_file}")
    if not os.path.exists(top_file):
        raise FileNotFoundError(f"TOP 文件不存在: {top_file}")

    log.info("读取 GROMACS 文件: %s | %s", gro_file, top_file)

    # 解析坐标与盒子
    gro = app.GromacsGroFile(gro_file)
    positions = gro.positions
    box_vectors = gro.getPeriodicBoxVectors()

    # 解析拓扑
    include_dir = gmx_include_dir or find_gmx_include_dir()
    if include_dir:
        log.info("  使用 GROMACS include 目录: %s", include_dir)
    else:
        log.warning("  [WARN] 未找到 include 目录，若拓扑含 #include 可能失败")

    try:
        top = load_gromacs_topology_for_openmm(
            top_file,
            periodicBoxVectors=box_vectors,
            includeDir=include_dir,
        )
    except ValueError as e:
        if "Could not locate #include file" in str(e):
            log.error("拓扑解析失败: %s", e)
            log.error("请通过 --gmx-path 指定包含 .ff 文件夹的父目录")
            sys.exit(1)
        raise

    omm_top = top.topology

    # 创建 System（保留自定义电荷与力常数）
    log.info("  构建 OpenMM System (保留自定义 RESP/Hessian 参数)...")
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )
    if box_vectors is not None:
        system.setDefaultPeriodicBoxVectors(*box_vectors)

    # 提取配体索引（修复：正确使用 atom.index）
    all_atoms = list(omm_top.atoms())
    ligand_info = [(atom.index, atom.residue.index) for atom in all_atoms if atom.residue.name == ligand_resname]
    ligand_indices = [idx for idx, _ in ligand_info]

    if not ligand_indices:
        raise ValueError(f"未在拓扑中找到配体残基: {ligand_resname}")

    # 防御性处理同名残基
    res_indices = set(res_idx for _, res_idx in ligand_info)
    if len(res_indices) > 1:
        target_res = min(res_indices)
        ligand_indices = [idx for idx, res_idx in ligand_info if res_idx == target_res]
        log.warning("[WARN] 发现多个同名残基，仅保留残基索引 %d", target_res)

    log.info("  [OK] 系统构建完成 | 原子数: %d | 配体: %d 原子", system.getNumParticles(), len(ligand_indices))

    return system, omm_top, positions, box_vectors, ligand_indices

def center_system_rigidly(
    positions: unit.Quantity,
    box_vectors: unit.Quantity,
    ligand_indices: List[int],
) -> Tuple[unit.Quantity, unit.Quantity]:
    """
    🔑 绝对刚性居中协议 (No-Wrap Guarantee)
    1. 彻底禁用原子级 PBC Wrap，杜绝跨盒分子被撕裂。
    2. 仅计算配体质心，将整个系统作为刚体平移至盒子几何中心。
    3. 配合 CMMotionRemover，确保生产采样期间体系死死钉在盒子中央。
    """
    import numpy as np
    from openmm import Vec3
    
    pos = np.array(positions.value_in_unit(unit.nanometer))
    box = np.array([v.value_in_unit(unit.nanometer) for v in box_vectors])
    
    # 计算配体质心与盒子中心
    lig_com = np.mean(pos[ligand_indices], axis=0)
    box_center = 0.5 * np.sum(box, axis=0)
    
    # 整体刚性平移
    shift = box_center - lig_com
    pos += shift
    
    log.info("  体系已刚性锁定至盒子中心 (No-Wrap, 仅整体平移)")
    new_pos = [Vec3(float(x), float(y), float(z)) for x, y, z in pos] * unit.nanometer
    return new_pos, box_vectors
