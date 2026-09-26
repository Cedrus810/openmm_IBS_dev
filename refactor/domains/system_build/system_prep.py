

# ============================================================================
# 整分子周期回卷：连通性只信 System，不信 topology 的键
# ============================================================================

# 流式处理的分块预算：按 16 GB 机器给 1 GiB，一次只把这么多字节拉进内存。
# 三处共用（DCD 校验读、回卷后复查、LJ 尾项外积），不要在调用点各搓一套。
STREAMING_CHUNK_BUDGET_BYTES = 1 << 30


def frames_per_chunk(
    bytes_per_frame: int,
    *,
    minimum: int = 1,
    maximum: int = 50000,
    budget_bytes: int = STREAMING_CHUNK_BUDGET_BYTES,
) -> int:
    """一块放多少帧：`budget_bytes // bytes_per_frame`，夹在 [minimum, maximum]。

    `bytes_per_frame` 非正/不可解释时返回 `minimum` —— 宁可慢，不要用一个
    猜出来的大数去赌内存。
    """
    try:
        per_frame = int(bytes_per_frame)
    except (TypeError, ValueError):
        return int(minimum)
    if per_frame <= 0:
        return int(minimum)
    return int(max(minimum, min(maximum, budget_bytes // per_frame)))


def system_molecule_grouping(system):
    """按 **System** 求连通分子：`(molecules, sorted_bonds)`。

    连通性取 `HarmonicBondForce` ∪ `CustomBondForce` ∪ `constraints`，与
    `prune_topology_bonds_unsupported_by_system()` 用的是**同一套**判据。
    `sorted_bonds` 是 BFS 生成树边 `[parent, child]`，语义与
    `mdtraj.Trajectory._sort_bonds()` 的返回值一致，可以直接喂给
    `image_molecules(sorted_bonds=...)`。

    为什么不能用 topology 的键，见 `image_molecules_by_system` 的 docstring。

    ## 为什么取宽（2026-09-09 更正）

    原来只认 `HarmonicBondForce` + constraints，理由是"多一条边会把两个分子并成
    一个"。方向反了：多一条边只是让两块一起平移（无害），**少**一条边会把一个
    分子拆开、被 `image_molecules` 分别平移撕碎。而回卷后的复查只遍历
    `sorted_bonds`（同一份邻接表长出来的），漏掉的边根本不在里面 ⟹ 对这种撕开
    完全失明。`rbfe_core` 的 hybrid builder 就把 core–core 键全放在
    `CustomBondForce` 里，窄判据下整个杂合配体会被当成一堆散原子。
    """
    import openmm as _mm

    n_particles = int(system.getNumParticles())
    adjacency = [[] for _ in range(n_particles)]
    for force in system.getForces():
        if not isinstance(force, (_mm.HarmonicBondForce, _mm.CustomBondForce)):
            continue
        for bond_index in range(force.getNumBonds()):
            i, j = force.getBondParameters(bond_index)[:2]
            adjacency[int(i)].append(int(j))
            adjacency[int(j)].append(int(i))
    for constraint_index in range(system.getNumConstraints()):
        i, j, _distance = system.getConstraintParameters(constraint_index)
        adjacency[int(i)].append(int(j))
        adjacency[int(j)].append(int(i))

    molecules = []
    sorted_bonds = []
    seen = bytearray(n_particles)
    for root in range(n_particles):
        if seen[root]:
            continue
        seen[root] = 1
        component = [root]
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if seen[neighbour]:
                    continue
                seen[neighbour] = 1
                component.append(neighbour)
                queue.append(neighbour)
                # 与 mdtraj._sort_bonds 一样：已定位的原子放在第 0 列，
                # 每个分子因此是一棵生成树，不会有原子被回卷到两个不同镜像。
                sorted_bonds.append((current, neighbour))
        molecules.append(component)
    return molecules, np.asarray(sorted_bonds, dtype=np.int32).reshape(-1, 2)


def prune_topology_bonds_unsupported_by_system(topology, system, *, log=None):
    """删掉 topology 里 **System 完全不认**的键，返回新的 `app.Topology`。

    ## 为什么需要它（PBC-01，2026-09-09）

    `app.PDBxFile` 往返会**凭空多造键**：写入端链 id 按
    `chr(ord('A') + chainIndex % 26)` 循环（`pdbxfile.py:392/473`），读取端
    `_struct_conn` 只按 `(seq_id, asym_id, atom_name)` 解析（`pdbxfile.py:218`），
    链数一超过 26 就有歧义。brd4/ligand1（12549 条链）实测多出 3 条：

        (0, 39543) ACE1.CH3 — HOH12582.H1
        (0, 39544) ACE1.CH3 — HOH12582.H2
        (1, 39542) ACE1.C   — HOH12582.O

    真键一条不缺（读取端 `createStandardBonds()` 已补齐），所以这 3 条是**净多出**。
    它们把一个水并进了蛋白那个"分子"，任何按 topology 的键归组的逻辑都会中招。

    ## 判据故意取得**宽**

    "System 认识的键" = `HarmonicBondForce` ∪ `CustomBondForce` ∪ `constraints`。
    判据宽 = 顶多漏删一条假键 = 无害；判据窄才会误删真键 ⟹ 取宽。

    ⚠️ [2026-09-09 更正] 这里原来写着"比 `system_molecule_grouping()` 宽一条
    `CustomBondForce`，两处的安全方向相反"。那个论证是错的：归组漏一条边会把
    分子拆开、回卷时被撕，比多一条边危险得多。两处现在用**同一套**判据，
    理由见 `system_molecule_grouping()` 的 docstring。

    一条都不用删时**原样返回传入的 topology**（同一个对象），
    所以 mmCIF 干净的体系（链数 ≤ 26，例如溶剂腿）`_topology_hash()` 逐位不变。
    """
    import openmm as _mm

    emit = log if callable(log) else (lambda message: None)

    supported = set()
    for force in system.getForces():
        if isinstance(force, (_mm.HarmonicBondForce, _mm.CustomBondForce)):
            for bond_index in range(force.getNumBonds()):
                i, j = force.getBondParameters(bond_index)[:2]
                supported.add((min(int(i), int(j)), max(int(i), int(j))))
    for constraint_index in range(system.getNumConstraints()):
        i, j, _distance = system.getConstraintParameters(constraint_index)
        supported.add((min(int(i), int(j)), max(int(i), int(j))))

    phantom = [
        (bond[0], bond[1])
        for bond in topology.bonds()
        if (min(bond[0].index, bond[1].index), max(bond[0].index, bond[1].index))
        not in supported
    ]
    if not phantom:
        return topology

    from openmm import app as _app

    pruned = _app.Topology()
    copied = {}
    for chain in topology.chains():
        new_chain = pruned.addChain(chain.id)
        for residue in chain.residues():
            new_residue = pruned.addResidue(
                residue.name, new_chain, residue.id, residue.insertionCode
            )
            for atom in residue.atoms():
                copied[atom.index] = pruned.addAtom(
                    atom.name, atom.element, new_residue, atom.id
                )
    dropped = {(a.index, b.index) for a, b in phantom}
    for atom_a, atom_b in topology.bonds():
        if (atom_a.index, atom_b.index) in dropped:
            continue
        pruned.addBond(copied[atom_a.index], copied[atom_b.index])
    box = topology.getPeriodicBoxVectors()
    if box is not None:
        pruned.setPeriodicBoxVectors(box)

    emit(
        f"  [WARN] 拓扑里有 {len(phantom)} 条 System 不认的键，已删除"
        f"（最先一条：原子 {phantom[0][0].index}"
        f"（{phantom[0][0].name} {phantom[0][0].residue.name}）–"
        f"{phantom[0][1].index}"
        f"（{phantom[0][1].name} {phantom[0][1].residue.name}））。"
        "mmCIF 往返在链数 > 26 时会造假键，见 PBC-01。"
    )
    return pruned


def image_molecules_by_system(traj, system, *, log=None):
    """按 System 的连通性给 `traj` 做整分子回卷，回卷后逐对复查，坏了就 raise。

    ## 为什么不能信 topology 的键（brd4 benchmark，2026-09-09）

    `md.Trajectory.image_molecules()` 默认按 **topology 的键** 判断"什么算一个
    分子"，而 topology 是可以带**假键**的：

      * `app.PDBxFile` 写 `topology.cif` 时链 id 按
        `chr(ord('A') + chainIndex % 26)` 循环（`pdbxfile.py:392/473`）；
      * 读回时 `_struct_conn` 只按 `(seq_id, asym_id, atom_name)` 解析
        （`pdbxfile.py:218`），**不含链序号**。

    链数一超过 26 就有歧义。brd4/ligand1 有 12549 条链（每个水一条），
    第一个残基 `ACE(A,1)` 的三条键被解析到某个同样落在 `(A,1)` 的水上：

        (0, 39543) ACE1.CH3 — HOH12582.H1
        (0, 39544) ACE1.CH3 — HOH12582.H2
        (1, 39542) ACE1.C   — HOH12582.O

    于是那个水被并进蛋白那个"分子"，`make_whole` 顺着假边把 H2 拽到原子 0 的
    最近镜像 —— **一个原子被整整平移了一个盒长**（实测 O–H2 = 7.3564 nm，
    盒长 7.3631 nm）。O–H2 这个 PME 排除对因此跨盒，attachment 腿起点体检
    （`assert_starting_state_is_sane`）直接拒绝开跑。而修复函数自己报的是
    `[OK]` —— 它从不复查自己的输出，所以从 2026-09-09 11:38 那次修复到崩，
    中间没有任何一行日志说过它把体系弄坏了。

    `System` 是唯一权威且必定存在的连通性来源：多一条假键都不会有，
    刚性水的 O–H 也不会缺（它们只以**约束**存在，`topology.bonds()` 里 0 条，
    见 MEM-15）。所以这里既不读 topology 的键，也不再往 topology 上补键。

    ⚠️ 只回卷、不旋转、不缩放，分子内相对坐标一律不变。
    """
    emit = log if callable(log) else (lambda message: None)

    # System 的原子序号下面被直接拿去索引 traj.topology：少一个原子才 IndexError，
    # 多一个是静默错位（anchor 里混进不相干的原子），所以先对一次数。
    if int(traj.n_atoms) != int(system.getNumParticles()):
        raise RuntimeError(
            f"轨迹 {int(traj.n_atoms)} 原子 ≠ System {int(system.getNumParticles())} 原子，"
            "不是同一个体系，拒绝回卷（fail closed）。"
        )

    molecules, sorted_bonds = system_molecule_grouping(system)

    # anchor 的挑法沿用 mdtraj `Topology.guess_anchor_molecules()` 的启发式，
    # 只是喂给它的是 System 的分子划分。分子划分本身不含假键时两者同解，
    # 所以对没触发过这个 bug 的体系，回卷结果与从前逐位相同。
    by_size = sorted(molecules, key=len, reverse=True)
    cutoff = max(len(by_size[int(0.1 * len(by_size))]), int(0.1 * len(by_size[0])))
    md_atoms = list(traj.topology.atoms)
    anchor = [{md_atoms[i] for i in mol} for mol in by_size if len(mol) > cutoff]
    other = [{md_atoms[i] for i in mol} for mol in by_size if len(mol) <= cutoff]
    if not anchor:
        # mdtraj 在这种情况下直接 raise。体系只有一堆同样大的小分子时
        # （纯水盒）挑最大的那个当 anchor 就够，没必要因此让整条流程停下。
        anchor, other = [other[0]], other[1:]

    traj.image_molecules(
        inplace=True,
        anchor_molecules=anchor,
        other_molecules=other,
        sorted_bonds=sorted_bonds,
        make_whole=True,
    )

    # 复查：所有成键/约束对都必须在同一镜像内。半个最短盒边是极宽松的上界
    # （真实键长 ≤ 0.25 nm），正常体系不可能触发，被撕开的必定触发。
    # 这一步就是这个 bug 里唯一缺失的东西：修复报了 OK，却没人问过它对不对。
    if sorted_bonds.size:
        box = np.asarray(traj.unitcell_vectors, dtype=np.float64)
        limit = 0.5 * float(np.min(np.linalg.norm(box, axis=2)))
        # 🔑 [2026-09-09] 逐帧算，不再一次物化 (帧数 × 键数 × 3)。
        #
        # 原来这三行会同时开出五个全尺寸数组：两个 fancy-index 拷贝、它们的
        # float32 差、`astype(np.float64)` 的双宽第四份、再加距离矩阵。
        # `sorted_bonds` 含每个刚性水的约束，所以键数 ≈ 原子数；而最大的调用方是
        # `ibs_engine.compute_u_kn`（`md.join` 了**所有窗口**的轨迹）。
        # 45k 原子 × 11500 帧的量级下这一条语句就是 20 GB 以上的瞬时分配。
        # 这道复查是 load-bearing 的（撕开的分子必须 fail closed），不能删，
        # 只能不要一次全开。逐帧的峰值是 O(键数)，而且大多数情况下第一帧就返回。
        src = sorted_bonds[:, 0]
        dst = sorted_bonds[:, 1]
        n_frames_total = int(traj.xyz.shape[0])
        n_pairs = int(sorted_bonds.shape[0])
        # 一帧的临时量约 = n_pairs × 3 × 8 B × 3 份（两个 gather + 差），再留一份余量。
        block = frames_per_chunk(max(1, n_pairs * 3 * 8 * 4), minimum=1)
        worst_distance = -1.0
        worst_pair_index = 0
        worst_frame = 0
        for start in range(0, n_frames_total, block):
            stop = min(start + block, n_frames_total)
            block_xyz = np.asarray(traj.xyz[start:stop], dtype=np.float64)
            block_distance = np.linalg.norm(
                block_xyz[:, src, :] - block_xyz[:, dst, :], axis=2
            )
            flat = int(np.argmax(block_distance))
            local_frame, local_pair = divmod(flat, n_pairs)
            block_max = float(block_distance[local_frame, local_pair])
            if block_max > worst_distance:
                worst_distance = block_max
                worst_pair_index = local_pair
                worst_frame = start + local_frame
            if worst_distance > limit:
                break
        if worst_distance > limit:
            raise RuntimeError(
                f"整分子回卷之后仍有成键/约束对跨周期镜像："
                f"最远 {worst_distance:.3f} nm > 上限 {limit:.3f} nm"
                f"（第 {worst_frame} 帧），"
                f"最坏的一对是原子 {int(sorted_bonds[worst_pair_index, 0])}–"
                f"{int(sorted_bonds[worst_pair_index, 1])}。\n"
                "    输入构型坏到 image_molecules 也拼不回来（或盒矢量与坐标不匹配）。"
                "拒绝把撕开的分子交给最小化 / PME（fail closed）。"
            )
    emit(f"  分子归组取自 System（{len(molecules)} 个连通分子），不读 topology 的键")
    return molecules


# ============================================================================
# 起始态体检：把"跑了几千步的无上下文 NaN"变成"起点就坏，且坏在哪个原子"
#
# 实测教训（§0.5.7）：一个坏掉的 System 最小化后 PE = 4.1e13 kJ/mol、
# max|F| = 3.7e9（落在脂质尾链氢 PA334/H8S），但代码什么都不报，几千步后只给出
# `Particle coordinate is NaN`，于是花了好几轮去猜。正常时 PE ≈ −6.5e5、
# max|F| ≈ 2.5e3 且落在水上 —— 相差 6 个数量级。
#
# ⚠️ **只有一份实现**：预平衡与 Boresch attachment 腿都调这个函数。
# 同一道门写两遍，迟早会有一处漏改（本仓库的 §0.5.5 就是这个毛病）。
# ============================================================================

# 起始态允许的最大单原子受力（kJ/mol/nm）。1e6 相对正常量级留了约 400 倍余量：
# 正常体系不可能触发，坏体系必定触发。
STARTING_STATE_MAX_FORCE_KJ_PER_MOL_NM = 1.0e6


def _assert_periodic_images_are_consistent(context, *, label: str, log=None):
    """排除对与约束对是否都在**同一周期镜像**内。不是就 raise（MEM-15）。

    ## 为什么必须单独查

    OpenMM 的 PME **要求排除/exception 对比 cutoff 近** —— 排除修正是按两原子的
    实际位移去减倒空间贡献的，一旦这对原子跨了盒，修正就完全算错。
    刚性水的 O–H 又只以**约束**存在（实测 memtest：`topology.bonds()` 里涉及水的
    键数 = 0，而约束 28626 个），所以靠 topology 键归组的 PBC 修复会把跨边界的水
    逐原子回卷、撕开 —— 而这件事对**所有**常规诊断都是隐形的：

      * 水没有键力项 ⟹ 键能与最大键长完全正常；
      * PME 的误差是平滑长程项 ⟹ 势能只是偏移，`max|F|` 也正常（实测 5292）；
      * 崩的是**约束求解器**（要在 5.9–12.4 nm 的 O/H 间满足 0.0957 nm）
        ⟹ 不到 1 ps 就 `Particle coordinate is NaN`，且没有任何上下文。

    所以这道检查不是冗余的：它是唯一能在起点看见这类损坏的量。
    """
    import openmm as _mm

    emit = log if callable(log) else (lambda message: None)
    system = context.getSystem()
    state = context.getState(getPositions=True)
    pos = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(_unit_nanometer()),
        dtype=np.float64,
    )
    box = np.asarray(
        [v.value_in_unit(_unit_nanometer())
         for v in state.getPeriodicBoxVectors()],
        dtype=np.float64,
    )
    lengths = np.diag(box)
    if not np.all(lengths > 0):
        return {"checked": False, "reason": "box_vectors_degenerate"}

    # 允许的上界：cutoff（有 PME 时）否则半个最短盒边。超了就说明跨镜像。
    cutoff = None
    for force in system.getForces():
        if isinstance(force, _mm.NonbondedForce):
            try:
                cutoff = float(
                    force.getCutoffDistance().value_in_unit(_unit_nanometer())
                )
            except Exception:  # noqa: BLE001
                cutoff = None
            break
    limit = float(cutoff) if cutoff else 0.5 * float(lengths.min())

    findings = {}
    for kind, pairs in (
        ("nonbonded_exceptions", _nonbonded_exception_pairs(system)),
        ("constraints", _constraint_pairs(system)),
    ):
        if pairs.size == 0:
            findings[kind] = {"n_pairs": 0, "n_over_limit": 0, "max_nm": 0.0}
            continue
        d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
        over = int(np.count_nonzero(d > limit))
        findings[kind] = {
            "n_pairs": int(pairs.shape[0]),
            "n_over_limit": over,
            "max_nm": float(d.max()),
            "limit_nm": limit,
        }
        if over:
            worst = pairs[int(np.argmax(d))]
            raise RuntimeError(
                f"[{label}] {over} 个 {kind} 对跨了周期镜像"
                f"（最远 {d.max():.3f} nm，上限 {limit:.3f} nm，"
                f"最坏的一对是原子 {int(worst[0])}–{int(worst[1])}）。\n"
                "    OpenMM 的 PME 要求排除对比 cutoff 近；约束对更是必须在同一镜像内，"
                "否则约束求解器无法收敛，几百步内就会给出一个**没有上下文的** "
                "`Particle coordinate is NaN`。\n"
                "    最常见原因：PBC 分子完整性修复按 topology 的**键**归组分子，"
                "而刚性水的 O–H 只以**约束**存在（`topology.bonds()` 里 0 个水键），"
                "于是跨边界的水被逐原子回卷、撕开。\n"
                "    修法见 `ABFEPipeline.repair_pbc_molecule_integrity`（MEM-15）："
                "把 System 的约束补成键再交给 `image_molecules()`。\n"
                "    [WARN] 这类损坏对键能 / 最大键长 / max|F| 全部隐形，只有本检查看得见。"
            )
    emit(
        f"  [OK] 镜像一致性: 排除对 {findings['nonbonded_exceptions']['n_pairs']} 个"
        f"（最远 {findings['nonbonded_exceptions']['max_nm']:.3f} nm）、"
        f"约束对 {findings['constraints']['n_pairs']} 个"
        f"（最远 {findings['constraints']['max_nm']:.3f} nm），上限 {limit:.3f} nm"
    )
    findings["checked"] = True
    return findings


def _unit_nanometer():
    from openmm import unit as _u

    return _u.nanometer


def _nonbonded_exception_pairs(system):
    import openmm as _mm

    for force in system.getForces():
        if isinstance(force, _mm.NonbondedForce):
            n = force.getNumExceptions()
            if n == 0:
                return np.empty((0, 2), dtype=int)
            return np.asarray(
                [force.getExceptionParameters(i)[:2] for i in range(n)], dtype=int
            )
    return np.empty((0, 2), dtype=int)


def _constraint_pairs(system):
    n = system.getNumConstraints()
    if n == 0:
        return np.empty((0, 2), dtype=int)
    return np.asarray(
        [system.getConstraintParameters(i)[:2] for i in range(n)], dtype=int
    )


def assert_starting_state_is_sane(
    context,
    topology,
    *,
    label: str,
    max_force_kj_per_mol_nm: float = STARTING_STATE_MAX_FORCE_KJ_PER_MOL_NM,
    remediation: str = "",
    log=None,
) -> Dict[str, Any]:
    """量一次势能与最大受力，异常即 raise。**只读**，不改坐标/速度/参数。

    `label` 进日志（例如 "最小化后" / "attachment 腿起点 λ=1"）。
    `remediation` 是调用点专属的排查提示，会附在超限报错里。

    返回实测数字，供调用方落盘。
    """
    from openmm import unit as _unit

    emit = log if callable(log) else (lambda message: None)
    state = context.getState(getEnergy=True, getForces=True)
    potential = state.getPotentialEnergy().value_in_unit(_unit.kilojoule_per_mole)
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            _unit.kilojoule_per_mole / _unit.nanometer
        ),
        dtype=float,
    )
    magnitudes = np.linalg.norm(forces, axis=1)
    atoms = _topology_atoms(topology)

    def _atom_label(index: int) -> str:
        atom = atoms[int(index)]
        return f"{atom.residue.name}{atom.residue.index}/{atom.name}"

    finite = np.isfinite(magnitudes)
    if not finite.all():
        raise RuntimeError(
            f"[{label}] 已有 {int((~finite).sum())} 个原子受力为 NaN/Inf —— "
            "起始态就是坏的，继续跑动力学没有意义。"
            "请检查输入坐标与拓扑是否对应（原子顺序、缓存是否串了体系）。"
            + (f"\n{remediation}" if remediation else "")
        )
    worst = int(np.argmax(magnitudes))
    max_force = float(magnitudes.max())
    emit(
        f"  {label}: PE = {potential:.6g} kJ/mol, "
        f"max|F| = {max_force:.4g} kJ/mol/nm "
        f"(idx={worst} {_atom_label(worst)}), "
        f"中位数 {np.median(magnitudes):.4g}"
    )
    if max_force > float(max_force_kj_per_mol_nm):
        order = np.argsort(magnitudes)[::-1][:10]
        worst_list = "\n".join(
            f"      idx={int(i)} {_atom_label(i)}  |F| = {magnitudes[int(i)]:.4g}"
            for i in order
        )
        raise RuntimeError(
            f"[{label}] max|F| = {max_force:.4g} kJ/mol/nm 超过上限 "
            f"{float(max_force_kj_per_mol_nm):.4g}（PE = {potential:.6g} kJ/mol）"
            "——**起始态就是坏的**，继续跑动力学只会得到一个没有上下文的 "
            "`Particle coordinate is NaN`。\n"
            f"    受力最大的 10 个原子：\n{worst_list}\n"
            "    实测参考量级：正常时 PE ≈ −6.5e5 kJ/mol、max|F| ≈ 2.5e3 且落在水上。\n"
            + (f"{remediation}\n" if remediation else "")
        )
    # ---- 镜像一致性（MEM-15）：受力检查看不见的那一类损坏 ----
    #
    # 2026-08-03 实测：243 个刚性水的 O/H 被 PBC 修复放进了不同周期镜像，于是
    # 729 个 PME 排除对跨盒（最远 13.76 nm）。而 **上面那些量全部正常**：
    # PE 只是多了个平滑的长程误差、max|F| = 5292（水没有键力项，所以键能与最大
    # 键长也完全正常）。跑起来后是**约束求解器**先崩 → 不到 1 ps 就 NaN。
    # 也就是说"力看起来正常"根本不能证明起点没坏，必须单独查这一项。
    image_report = _assert_periodic_images_are_consistent(
        context, label=label, log=emit
    )

    return {
        "label": label,
        "potential_energy_kj_per_mol": potential,
        "max_force_kj_per_mol_nm": max_force,
        "periodic_image_consistency": image_report,
        "median_force_kj_per_mol_nm": float(np.median(magnitudes)),
        "max_force_atom_index": worst,
        "max_force_atom": _atom_label(worst),
        "max_force_threshold_kj_per_mol_nm": float(max_force_kj_per_mol_nm),
    }


def _topology_atoms(topology):
    """原子迭代器，兼容 OpenMM（`atoms()` 方法）与 mdtraj（`atoms` 属性）。"""
    attr = topology.atoms
    return list(attr() if callable(attr) else attr)


def _topology_residues(topology):
    attr = topology.residues
    return list(attr() if callable(attr) else attr)


def assign_lipid_leaflets(
    topology,
    positions,
    normal_axis: str = "z",
    lipid_molecules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """按头基参考原子相对膜中面的位置把脂质分到上/下叶（§3.3 / §9）。

    **不假设对半分**：返回实测的每叶计数与组成，以及识别不到头基参考原子的单位
    列表（不静默丢弃）。中面取所有头基参考原子坐标在膜法向上的均值。

    `lipid_molecules`（来自 `classify_system_composition()` 的
    `molecules_by_role["lipid"]`）给出时**按分子**分叶——这是唯一正确的口径：
    Amber Lipid21 把一个 POPC 拆成 `PA` + `PC` + `OL` **三个残基**，按残基数会数出
    3 倍的"脂质数"，APL 直接错 3 倍；而且尾链残基没有磷原子，按残基找头基会
    直接报"找不到头基参考原子"。不给时退回按残基分（只适用于一残基一脂质的体系）。
    """
    axis = {"x": 0, "y": 1, "z": 2}[str(normal_axis).strip().lower()]
    pos_nm = np.asarray(
        positions.value_in_unit(unit.nanometer)
        if hasattr(positions, "value_in_unit")
        else positions,
        dtype=float,
    )

    heads: List[Tuple[int, str, float]] = []
    unassignable: List[Dict[str, Any]] = []
    head_candidates = LIPID_HEAD_REFERENCE_ATOM_NAMES + LIPID_HEAD_FALLBACK_ATOM_NAMES

    if lipid_molecules is not None:
        # 权威口径：按分子。原子区间来自 `.top` 的 [ molecules ] 展开顺序。
        all_atoms = _topology_atoms(topology)
        for entry in lipid_molecules:
            start, stop = int(entry["start"]), int(entry["stop"])
            atom_by_name = {}
            for atom in all_atoms[start:stop]:
                atom_by_name.setdefault(str(atom.name).strip().upper(), int(atom.index))
            ref_index = next(
                (atom_by_name[c] for c in head_candidates if c in atom_by_name), None
            )
            label = str(entry.get("molecule_name", "LIPID")).strip().upper()
            if ref_index is None:
                unassignable.append(
                    {
                        "molecule_name": label,
                        "ordinal": int(entry.get("ordinal", -1)),
                        "atom_range": [start, stop],
                    }
                )
                continue
            heads.append((start, label, float(pos_nm[ref_index][axis])))
    else:
        for residue in _topology_residues(topology):
            name = str(residue.name).strip().upper()
            if name not in KNOWN_LIPID_RESIDUE_NAMES:
                continue
            residue_atoms = residue.atoms
            residue_atoms = list(
                residue_atoms() if callable(residue_atoms) else residue_atoms
            )
            atom_by_name = {
                str(atom.name).strip().upper(): int(atom.index) for atom in residue_atoms
            }
            ref_index = next(
                (atom_by_name[c] for c in head_candidates if c in atom_by_name), None
            )
            if ref_index is None:
                unassignable.append(
                    {"residue_index": int(residue.index), "residue_name": name}
                )
                continue
            heads.append((int(residue.index), name, float(pos_nm[ref_index][axis])))

    if not heads:
        raise ValueError(
            "找不到任何可用的脂质头基参考原子"
            f"（尝试过 {LIPID_HEAD_REFERENCE_ATOM_NAMES + LIPID_HEAD_FALLBACK_ATOM_NAMES}）。"
            "无法判定上下叶——请检查脂质残基/原子命名，不要跳过本检查。"
        )

    coords = np.asarray([h[2] for h in heads], dtype=float)
    midplane = float(np.mean(coords))
    upper: Dict[str, int] = {}
    lower: Dict[str, int] = {}
    for residue_index, name, coordinate in heads:
        bucket = upper if coordinate > midplane else lower
        bucket[name] = bucket.get(name, 0) + 1

    n_upper = int(sum(upper.values()))
    n_lower = int(sum(lower.values()))
    total = n_upper + n_lower
    return {
        "normal_axis": str(normal_axis).strip().lower(),
        "midplane_coordinate_nm": midplane,
        "upper_leaflet_counts": dict(sorted(upper.items())),
        "lower_leaflet_counts": dict(sorted(lower.items())),
        "n_upper": n_upper,
        "n_lower": n_lower,
        "n_total": total,
        "imbalance_fraction": (
            abs(n_upper - n_lower) / total if total else 0.0
        ),
        "unassignable_lipid_residues": unassignable,
        "grouping": "per_molecule" if lipid_molecules is not None else "per_residue",
    }


def verify_membrane_normal_axis(
    topology,
    positions,
    declared_axis: str = "z",
    lipid_molecules: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """从坐标**实测**膜法向，并核对与声明的轴一致（§3.3「蛋白跨膜方向」）。

    为什么必须实测：`MonteCarloMembraneBarostat` 把膜法向**硬编码为 z**——它按
    XY 等比例、Z 独立地缩放。若坐标里的双层其实垂直于 x 或 y，barostat 会沿着膜
    平面内的一个方向单独缩放、同时把法向和一个面内方向绑在一起，膜会被压坏，
    而且**不会报任何错**。只看盒子形状（哪条边最长）是不够的：盒子长边与膜法向
    未必一致。

    判据：沿真正的法向，头基参考原子应当分成**两个清晰的簇**（上下叶），
    两簇计数接近相等且簇心间距 ≈ P–P 膜厚（2–5 nm）；面内两个方向上头基是
    弥散分布。所以取"二分后两簇计数最平衡且簇心间距最大"的轴为法向。
    """
    axis_names = ("x", "y", "z")
    declared = str(declared_axis).strip().lower()
    if declared not in axis_names:
        raise ValueError(f"declared_axis={declared_axis!r} 非法；允许 {list(axis_names)}")

    pos_nm = np.asarray(
        positions.value_in_unit(unit.nanometer)
        if hasattr(positions, "value_in_unit")
        else positions,
        dtype=float,
    )
    head_candidates = LIPID_HEAD_REFERENCE_ATOM_NAMES + LIPID_HEAD_FALLBACK_ATOM_NAMES
    heads: List[int] = []
    if lipid_molecules:
        all_atoms = _topology_atoms(topology)
        for entry in lipid_molecules:
            by_name = {}
            for atom in all_atoms[int(entry["start"]):int(entry["stop"])]:
                by_name.setdefault(str(atom.name).strip().upper(), int(atom.index))
            reference = next((by_name[c] for c in head_candidates if c in by_name), None)
            if reference is not None:
                heads.append(reference)
    else:
        for residue in _topology_residues(topology):
            if str(residue.name).strip().upper() not in KNOWN_LIPID_RESIDUE_NAMES:
                continue
            residue_atoms = residue.atoms
            residue_atoms = list(
                residue_atoms() if callable(residue_atoms) else residue_atoms
            )
            by_name = {
                str(a.name).strip().upper(): int(a.index) for a in residue_atoms
            }
            reference = next((by_name[c] for c in head_candidates if c in by_name), None)
            if reference is not None:
                heads.append(reference)
    if len(heads) < 4:
        raise ValueError(
            f"只找到 {len(heads)} 个脂质头基参考原子，无法实测膜法向。"
        )

    head_array = np.asarray(heads, dtype=int)
    per_axis: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(axis_names):
        values = pos_nm[head_array][:, index]
        split = 0.5 * (float(values.min()) + float(values.max()))
        lower = values[values < split]
        upper = values[values >= split]
        if lower.size == 0 or upper.size == 0:
            per_axis[name] = {
                "n_lower": int(lower.size),
                "n_upper": int(upper.size),
                "separation_nm": 0.0,
                "balance": 0.0,
            }
            continue
        total = float(lower.size + upper.size)
        per_axis[name] = {
            "n_lower": int(lower.size),
            "n_upper": int(upper.size),
            # 1.0 = 完美对半；越小越不像双层。
            "balance": 1.0 - abs(lower.size - upper.size) / total,
            "separation_nm": float(np.mean(upper) - np.mean(lower)),
        }

    # 双层的法向：两簇最平衡；平衡度相同时取簇心间距更大的。
    measured = max(
        axis_names,
        key=lambda name: (per_axis[name]["balance"], per_axis[name]["separation_nm"]),
    )
    report = {
        "declared_axis": declared,
        "measured_axis": measured,
        "n_head_atoms": int(head_array.size),
        "per_axis": per_axis,
        "agrees": measured == declared,
    }
    if measured != declared:
        raise ValueError(
            f"声明膜法向为 {declared!r}，但从坐标实测最像双层法向的是 {measured!r}。"
            f"逐轴证据（二分后两簇计数 / 平衡度 / 簇心间距）：{per_axis}。"
            "OpenMM 的 MonteCarloMembraneBarostat 把法向硬编码为 z——轴错了它会沿"
            "膜平面内单独缩放、把法向与一个面内方向绑死，膜会被压坏且不报错。"
            "请在建系时把膜法向对齐 z。"
        )
    logger.info(
        "膜法向实测确认为 %s（%d 个头基原子；上下叶 %d/%d，簇心间距 %.3f nm）",
        measured, report["n_head_atoms"],
        per_axis[measured]["n_upper"], per_axis[measured]["n_lower"],
        per_axis[measured]["separation_nm"],
    )
    return report


def validate_membrane_input(
    topology,
    positions,
    box_vectors,
    declared: Optional[Dict[str, Any]] = None,
    normal_axis: str = "z",
    composition: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """§3.3 膜输入核对：结构一致性 + 来源可追溯性，任一不符即 fail closed。

    归档（.top / 全部 .itp / 位置限制 / 力场 include 的递归 SHA256）由
    `runabfe._gromacs_dependency_hashes()` 负责，本函数只做结构与声明的交叉核对。
    """
    declared = dict(declared or {})

    missing = [
        field
        for field in MEMBRANE_INPUT_REQUIRED_PROVENANCE_FIELDS
        if not declared.get(field)
    ]
    if missing:
        raise ValueError(
            f"膜输入缺少 §3.3 要求记录的字段 {missing}。"
            "输入必须是已经完成膜构建和主要平衡的体系，且来源可追溯——"
            "不依赖通用预平衡去完成脂质重排或蛋白插膜。"
        )

    # §9/§15：上游平衡的两条合法表述，二者必居其一（见上方常量处的说明）。
    upstream_ns = declared.get(MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD)
    upstream_status = str(declared.get(MEMBRANE_UPSTREAM_STATUS_FIELD) or "").strip().lower()
    if upstream_ns is not None:
        try:
            upstream_ns = float(upstream_ns)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD}={declared[MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD]!r} 不是数值"
            ) from exc
        if not math.isfinite(upstream_ns) or upstream_ns < 0.0:
            raise ValueError(f"{MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD} 必须是非负有限数")
    elif upstream_status == MEMBRANE_UPSTREAM_STATUS_COMPLETED_UNRECORDED:
        if not declared.get("final_equilibration_job"):
            raise ValueError(
                f"{MEMBRANE_UPSTREAM_STATUS_FIELD}="
                f"{MEMBRANE_UPSTREAM_STATUS_COMPLETED_UNRECORDED!r} 时必须给 "
                "final_equilibration_job 指向上游平衡的证据（作业 ID / 路径）。"
            )
    else:
        raise ValueError(
            "膜输入必须说明上游平衡情况，二者必居其一："
            f"给出 {MEMBRANE_UPSTREAM_EQUILIBRATION_FIELD}（正数，计入总平衡时长），"
            f"或声明 {MEMBRANE_UPSTREAM_STATUS_FIELD}="
            f"{MEMBRANE_UPSTREAM_STATUS_COMPLETED_UNRECORDED!r}"
            "（上游生产已完成但时长不可考，需同时给 final_equilibration_job）。"
            "沉默等于让未充分平衡的体系混进来。"
            "[WARN] 无论走哪条，§9 的实测质量门都不受影响。"
        )

    conformational_state = str(declared["conformational_state"]).strip()
    conformational_state_declared = (
        conformational_state.lower() != MEMBRANE_CONFORMATIONAL_STATE_UNSPECIFIED
    )
    if not conformational_state_declared:
        logger.warning(
            "[WARN] conformational_state=%r：构象态未声明，已如实记入 provenance。"
            "§1.1 的本意是防止跨构象态混用——本次运行的构象由输入文件指纹唯一确定，"
            "但若日后要与其它构象态的结果比较，这一项必须补上。",
            MEMBRANE_CONFORMATIONAL_STATE_UNSPECIFIED,
        )

    exposure = str(declared["binding_site_solvent_exposure"]).strip().lower()
    if exposure not in MEMBRANE_BINDING_SITE_EXPOSURE_LEVELS:
        raise ValueError(
            f"binding_site_solvent_exposure={declared['binding_site_solvent_exposure']!r} 非法；"
            f"允许 {list(MEMBRANE_BINDING_SITE_EXPOSURE_LEVELS)}（§3.0 决定迟滞风险等级）。"
        )

    # ---- 坐标 / 拓扑原子数一致 ----
    pos_nm = np.asarray(
        positions.value_in_unit(unit.nanometer)
        if hasattr(positions, "value_in_unit")
        else positions,
        dtype=float,
    )
    n_topology_atoms = int(topology.getNumAtoms())
    if pos_nm.shape[0] != n_topology_atoms:
        raise ValueError(
            f"坐标原子数 {pos_nm.shape[0]} 与拓扑原子数 {n_topology_atoms} 不一致（§3.3）。"
        )
    if declared.get("n_atoms") is not None and int(declared["n_atoms"]) != n_topology_atoms:
        raise ValueError(
            f"声明原子数 {declared['n_atoms']} 与拓扑 {n_topology_atoms} 不一致。"
        )

    # ---- 盒型必须是长方体，膜法向对齐 z（§1.1）----
    box_nm = np.asarray(
        box_vectors.value_in_unit(unit.nanometer)
        if hasattr(box_vectors, "value_in_unit")
        else [
            v.value_in_unit(unit.nanometer) if hasattr(v, "value_in_unit") else v
            for v in box_vectors
        ],
        dtype=float,
    )
    off_diagonal = float(np.max(np.abs(box_nm - np.diag(np.diag(box_nm)))))
    if off_diagonal > 1.0e-6:
        raise ValueError(
            f"膜体系盒型必须是长方体（rectangular），实测最大非对角元 {off_diagonal:.6g} nm。"
            "不得用截角八面体/十二面体（§1.1）。"
        )

    # ---- 上下叶脂质数：实测 + 与声明交叉核对，不假设对半分 ----
    # ⚠️ 顺序有意义：**先**叶片划分再核对法向。
    # 叶片划分会报出"哪些脂质找不到头基参考原子"这条**更具体、更可操作**的错；
    # 而法向实测在头基不足时只会报"无法实测膜法向"——那是前者的下游后果，
    # 先报下游会把人引到错误的地方去查。
    leaflets = assign_lipid_leaflets(
        topology,
        positions,
        normal_axis=normal_axis,
        lipid_molecules=(
            (composition or {}).get("molecules_by_role", {}).get("lipid")
        ),
    )
    if leaflets["unassignable_lipid_residues"]:
        raise ValueError(
            f"有 {len(leaflets['unassignable_lipid_residues'])} 个脂质单位找不到头基参考原子，"
            "无法判定所属叶片。请补充原子命名映射，不要让它们静默不计入。"
        )
    # §3.3：膜法向必须**实测**核对，不能只信盒子形状（长边未必是法向）。
    # 轴错了 MonteCarloMembraneBarostat 会沿膜平面内单独缩放、把法向与一个面内
    # 方向绑死，膜被压坏且不报错。
    normal_axis_report = verify_membrane_normal_axis(
        topology,
        positions,
        declared_axis=normal_axis,
        lipid_molecules=(composition or {}).get("molecules_by_role", {}).get("lipid"),
    )

    for key, measured in (("n_upper", leaflets["n_upper"]), ("n_lower", leaflets["n_lower"])):
        if declared.get(key) is not None and int(declared[key]) != measured:
            raise ValueError(
                f"声明 {key}={declared[key]} 与实测 {measured} 不一致（§9：叶片数必须有依据）。"
            )

    # ---- 水 / 离子计数 ----
    # 有 `.top` 组成时以它为准：残基名扫描会在 `TP3` / `Na+` / `Cl-` 这类命名上
    # **静默数出 0**（实测 memtest 体系正是如此），而 `[ molecules ]` 是权威的。
    n_water = 0
    ion_counts: Dict[str, int] = {}
    if composition:
        roles = composition["roles"]
        for molecule_name, count in composition["molecule_counts"].items():
            role = roles.get(molecule_name)
            if role == "water":
                n_water += int(count)
            elif role == "ion":
                key = str(molecule_name).strip().upper()
                ion_counts[key] = ion_counts.get(key, 0) + int(count)
    else:
        for residue in _topology_residues(topology):
            name = str(residue.name).strip().upper()
            if name in WATER_MOLECULE_NAMES:
                n_water += 1
            elif _normalize_ion_name(name) in MONOATOMIC_ION_NAMES:
                ion_counts[name] = ion_counts.get(name, 0) + 1
    if declared.get("n_water") is not None and int(declared["n_water"]) != n_water:
        raise ValueError(f"声明水分子数 {declared['n_water']} 与实测 {n_water} 不一致。")
    if declared.get("ion_counts"):
        expected_ions = {
            str(k).upper(): int(v) for k, v in dict(declared["ion_counts"]).items()
        }
        if expected_ions != ion_counts:
            raise ValueError(
                f"声明离子计数 {expected_ions} 与实测 {ion_counts} 不一致。"
            )

    return {
        "protocol_version": MEMBRANE_INPUT_PROTOCOL_VERSION,
        "n_atoms": n_topology_atoms,
        "box_lengths_nm": [float(np.linalg.norm(row)) for row in box_nm],
        "box_is_rectangular": True,
        "membrane_normal_axis": normal_axis_report,
        "leaflets": leaflets,
        "n_water": n_water,
        "ion_counts": dict(sorted(ion_counts.items())),
        "declared": declared,
        "binding_site_solvent_exposure": exposure,
        "conformational_state": conformational_state,
        "conformational_state_declared": conformational_state_declared,
        "upstream_equilibration_ns": upstream_ns,
        "upstream_equilibration_status": (
            upstream_status or ("declared_ns" if upstream_ns is not None else None)
        ),
        "nominal_equilibration_precheck_applicable": upstream_ns is not None,
        "composition": (
            {
                "roles": composition["roles"],
                "role_evidence": composition["role_evidence"],
                "molecule_counts": composition["molecule_counts"],
                "n_atoms_total": composition["n_atoms_total"],
            }
            if composition
            else None
        ),
    }
