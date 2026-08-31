"""§9 观测量提取器：轨迹 → 质量门输入（`membrane_observables_from_trajectory`）。

判定层（`evaluate_membrane_quality_gate`）之前是"就绪但拿不到输入"。这个文件测的是
把两者接起来的那一环，用**合成膜轨迹**：一个上下叶各 8 个 POPC 的小双层 + 蛋白 +
配体 + 水 + 离子，几何完全由构造器控制，所以每个观测量都有可独立手算的期望值。

守的重点不是"物理算得多准"，而是：
  1. 直接几何量（APL / 膜厚 / 盒序列）等于手算值；
  2. **宁可 fail closed 也不猜**——缺时间轴、缺键、缺口袋定义、缺头基原子都报错，
     不用帧号冒充 ns、不用距离阈值冒充共价键、不用「离配体最近若干残基」当口袋；
  3. 提取器产出能**直接**喂给判定层，两边的键名对得上（否则各自"完成"却接不起来）。

⚠️ 序参量、疏水核异常水、横向弛豫时间尺度都是"等价结构指标"，本文件只验证口径
自洽与边界行为；**量级是否合理必须用真实膜体系对照文献值**，那一步还没做。

全部 CPU 可跑：合成轨迹在内存里构造，不读盘、不建 Context。
"""

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

pytest.importorskip("openmm")
md = pytest.importorskip("mdtraj")

import abfe_core as core


N_PER_LEAFLET = 8
BOX_XY = 4.0
BOX_Z = 12.0
UPPER_HEAD_Z = 8.0
LOWER_HEAD_Z = 4.0
MIDPLANE = 0.5 * (UPPER_HEAD_Z + LOWER_HEAD_Z)  # 6.0
THICKNESS = UPPER_HEAD_Z - LOWER_HEAD_Z          # 4.0


# 脂质横向扩散系数（nm²/ns）。取 0.00533 使弛豫时间尺度落在 30 ns
# （`0.8² / (4·D)`），与真实 POPC 的 ~0.008 nm²/ns 同量级。
#
# 设成 0 时横向 MSD ≡ 0 → 拟合出的扩散系数为 0 → 弛豫时间尺度趋于无穷。
# 那个判据已于 2026-08-02（MEM-11）从硬门降为诊断，所以现在它不再让门失败，
# 但必须**如实报出**——见 test_a_membrane_that_never_relaxes_laterally_is_reported...
LIPID_LATERAL_D_NM2_PER_NS = 0.8**2 / (4.0 * 30.0)


def _build_membrane_trajectory(
    n_frames=41,
    total_time_ns=40.0,
    with_bonds=True,
    with_time=True,
    apl_slope_per_ns=0.0,
    n_core_waters=0,
    coion_z=10.0,
    lipid_lateral_d_nm2_per_ns=LIPID_LATERAL_D_NM2_PER_NS,
    modular_lipid_residues=False,
):
    """合成双层膜轨迹。

    每个 POPC 残基三个原子：头基 P + 两个尾链碳 C1/C2（C1–C2 成键，用于序参量）。
    尾链沿膜法向排列，所以 C–C 键向量与法向平行 → 序参量应为 +1。

    `modular_lipid_residues=True` 复现 Amber Lipid21 的模块化命名（§0.5.4）：
    同一个 POPC **分子**被拆成两个残基 —— 头基 `PC`（放 P）+ 尾链 `PA`（放 C1/C2）。
    这样"按残基"的判据会数出 2 倍脂质数，且尾链残基没有磷原子会直接报错；
    只有"按分子"（`composition["molecules_by_role"]["lipid"]`）才对。
    memtest 的真实体系是 PA+PC+OL 三残基，机理与此一致。

    脂质做**二维随机行走**（每帧独立高斯步长，每方向方差 2·D·dt），于是
    MSD(Δt) = 4·D·Δt 对单参考帧与时间平均两种口径都成立，提取器可反解出设定的 D。
    横向位移**只**影响 MSD 与蛋白横截面归属：APL 取自盒长，膜厚/序参量/密度分布/
    核内水都只用 z。
    """
    top = md.Topology()
    chain = top.add_chain()
    lipid_head_atoms = []
    lipid_molecule_ranges = []

    for leaflet_z, direction in ((UPPER_HEAD_Z, -1.0), (LOWER_HEAD_Z, +1.0)):
        for i in range(N_PER_LEAFLET):
            start = top.n_atoms
            if modular_lipid_residues:
                head_res = top.add_residue("PC", chain)
                p = top.add_atom("P", md.element.phosphorus, head_res)
                tail_res = top.add_residue("PA", chain)
                c1 = top.add_atom("C1", md.element.carbon, tail_res)
                c2 = top.add_atom("C2", md.element.carbon, tail_res)
            else:
                res = top.add_residue("POPC", chain)
                p = top.add_atom("P", md.element.phosphorus, res)
                c1 = top.add_atom("C1", md.element.carbon, res)
                c2 = top.add_atom("C2", md.element.carbon, res)
            lipid_head_atoms.append((p.index, leaflet_z, i, direction, c1.index, c2.index))
            # 一个脂质**分子**的精确原子区间，供 molecules_by_role 用（不猜、不按名字数）。
            lipid_molecule_ranges.append(
                {"molecule_name": "POPC", "start": start, "stop": top.n_atoms}
            )
            if with_bonds:
                top.add_bond(c1, c2)

    # 蛋白：一段跨膜的 ALA 骨架（N/CA/C/O），沿 z 排列 → 倾角 ≈ 0°。
    protein_atoms = []
    for k in range(6):
        res = top.add_residue("ALA", chain)
        for name, element in (
            ("N", md.element.nitrogen),
            ("CA", md.element.carbon),
            ("C", md.element.carbon),
            ("O", md.element.oxygen),
        ):
            atom = top.add_atom(name, element, res)
            protein_atoms.append((atom.index, 3.0 + 0.15 * len(protein_atoms)))

    ligand_res = top.add_residue("MOL", chain)
    ligand_atoms = [
        top.add_atom("C1", md.element.carbon, ligand_res).index,
        top.add_atom("C2", md.element.carbon, ligand_res).index,
    ]

    water_atoms = []
    for _ in range(6 + n_core_waters):
        res = top.add_residue("HOH", chain)
        water_atoms.append(top.add_atom("O", md.element.oxygen, res).index)
        top.add_atom("H1", md.element.hydrogen, res)
        top.add_atom("H2", md.element.hydrogen, res)

    ion_res = top.add_residue("NA", chain)
    coion_index = top.add_atom("NA", md.element.sodium, ion_res).index

    n_atoms = top.n_atoms
    xyz = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)

    times_ns = np.linspace(0.0, total_time_ns, n_frames)
    # 脂质横向运动必须是**真正的二维随机行走**：每帧独立高斯步长，每个方向方差
    # 2·D·dt，于是 MSD(Δt) = 4·D·Δt 对**单参考帧**与**时间平均**两种口径都成立。
    #
    # ⚠️ 此前用的是"沿固定方向位移 |Δr| = sqrt(4·D·t)"的确定性构造。那只让
    # **单参考帧** MSD 恰好等于 4·D·t，时间平均 MSD 则远小于它（相邻帧的位移
    # 高度相关）。提取器改成时间平均之后（MEM-11），那种构造会让它反解出错的 D ——
    # 也就是说旧 fixture 本身就不是扩散运动，只是恰好配合了旧估计器。
    rng = np.random.default_rng(20260730)
    dt_ns = total_time_ns / max(1, n_frames - 1)
    sigma = float(np.sqrt(2.0 * lipid_lateral_d_nm2_per_ns * dt_ns))
    steps = rng.normal(
        loc=0.0, scale=sigma, size=(n_frames - 1, len(lipid_head_atoms), 2)
    )
    walk = np.concatenate(
        [np.zeros((1, len(lipid_head_atoms), 2)), np.cumsum(steps, axis=0)], axis=0
    )

    for frame in range(n_frames):
        for lipid_i, (p_idx, leaflet_z, i, direction, c1, c2) in enumerate(
            lipid_head_atoms
        ):
            # 每叶 8 个脂质按 4×2 **铺满**横向盒面，不是挤在角落里：
            # APL 的蛋白横截面校正用"最近参考原子归属"，脂质若聚成一团，
            # 远离那团的整片面积都会被判给蛋白，校正量变得不合物理。
            x = (i % 4 + 0.5) * (BOX_XY / 4.0) + walk[frame, lipid_i, 0]
            y = (i // 4 + 0.5) * (BOX_XY / 2.0) + walk[frame, lipid_i, 1]
            xyz[frame, p_idx] = [x, y, leaflet_z]
            # 尾链沿法向伸向膜中心：C–C 向量平行于 z（横向偏移三个原子一致，
            # 所以键向量仍严格平行法向，序参量恒为 1）。
            xyz[frame, c1] = [x, y, leaflet_z + direction * 0.6]
            xyz[frame, c2] = [x, y, leaflet_z + direction * 1.2]
        for atom_index, z in protein_atoms:
            xyz[frame, atom_index] = [2.0, 2.0, z]
        for offset, atom_index in enumerate(ligand_atoms):
            xyz[frame, atom_index] = [2.2, 2.0, 6.0 + 0.1 * offset]
        for w, atom_index in enumerate(water_atoms):
            if w < n_core_waters:
                # 刻意塞进疏水核（|z − 中面| 很小）。
                xyz[frame, atom_index] = [1.0, 1.0, MIDPLANE + 0.1]
            else:
                xyz[frame, atom_index] = [3.5, 3.5, 11.0]
            xyz[frame, atom_index + 1] = xyz[frame, atom_index] + [0.01, 0.0, 0.0]
            xyz[frame, atom_index + 2] = xyz[frame, atom_index] + [0.0, 0.01, 0.0]
        xyz[frame, coion_index] = [3.0, 3.0, coion_z]

    # APL 通过横向盒长变化实现（面积 = a*b）。
    area0 = BOX_XY * BOX_XY
    areas = area0 + apl_slope_per_ns * N_PER_LEAFLET * times_ns
    edge = np.sqrt(areas)
    lengths = np.stack([edge, edge, np.full(n_frames, BOX_Z)], axis=1).astype(np.float32)
    angles = np.tile(np.array([90.0, 90.0, 90.0], dtype=np.float32), (n_frames, 1))

    traj = md.Trajectory(
        xyz,
        top,
        time=(times_ns * 1000.0).astype(np.float32) if with_time else None,
        unitcell_lengths=lengths,
        unitcell_angles=angles,
    )
    if not with_time:
        traj.time = np.zeros(n_frames, dtype=np.float32)
    # 挂在 traj 上而不是加第 4 个返回值：已有 20+ 处按三元组解包，
    # 改签名会把与本项无关的测试全改一遍。取用见 `_composition_for()`。
    traj.lipid_molecule_ranges = lipid_molecule_ranges
    return traj, coion_index, ligand_atoms


def _pocket_indices(traj):
    return [int(i) for i in traj.topology.select("protein and name CA")]


def _composition_for(traj):
    """构造 `classify_system_composition()` 那种形状的组成字典（按分子路径）。

    真实路径的组成来自 `.top` 的 `[ molecules ]` + `[ moleculetype ]`
    （`abfe_core.classify_system_composition`）。这里只需要形状一致：
    脂质给**分子**区间，其余角色给原子索引集合。
    """
    top = traj.topology
    by_role = {"protein": [], "lipid": [], "water": [], "ion": [], "ligand": []}
    lipid_atoms = {
        i
        for entry in traj.lipid_molecule_ranges
        for i in range(entry["start"], entry["stop"])
    }
    for atom in top.atoms:
        index = int(atom.index)
        name = str(atom.residue.name).strip().upper()
        if index in lipid_atoms:
            by_role["lipid"].append(index)
        elif name == "MOL":
            by_role["ligand"].append(index)
        elif name == "HOH":
            by_role["water"].append(index)
        elif name == "NA":
            by_role["ion"].append(index)
        else:
            by_role["protein"].append(index)
    return {
        "atom_indices_by_role": {k: sorted(v) for k, v in by_role.items()},
        "molecules_by_role": {"lipid": list(traj.lipid_molecule_ranges)},
    }


def _extract(traj, **kwargs):
    kwargs.setdefault("pocket_atom_indices", _pocket_indices(traj))
    return core.membrane_observables_from_trajectory(traj, "MOL", **kwargs)


# ---------------------------------------------------------------------------
# 直接几何量：与手算一致
# ---------------------------------------------------------------------------


def test_apl_thickness_and_box_series_match_hand_computed_values():
    traj, _, _ = _build_membrane_trajectory()
    obs, diag = _extract(traj)

    # 上下叶各 8 个 → APL = 面积 / 8。
    expected_apl = (BOX_XY * BOX_XY) / N_PER_LEAFLET
    assert obs["apl_nm2"]["values"][0] == pytest.approx(expected_apl, rel=1e-5)
    assert obs["bilayer_thickness_nm"]["values"][0] == pytest.approx(THICKNESS, rel=1e-5)
    assert obs["box_xy_area_nm2"]["values"][0] == pytest.approx(BOX_XY * BOX_XY, rel=1e-5)
    assert obs["box_z_nm"]["values"][0] == pytest.approx(BOX_Z, rel=1e-5)
    assert obs["box_volume_nm3"]["values"][0] == pytest.approx(
        BOX_XY * BOX_XY * BOX_Z, rel=1e-5
    )
    assert diag["n_upper"] == N_PER_LEAFLET
    assert diag["n_lower"] == N_PER_LEAFLET


def test_apl_is_not_computed_as_total_over_two():
    """§9：叶片数不是随手对半分——不平衡时 APL 必须按各自计数算。

    造一个上叶 8 / 下叶 6 的膜：若按"总数/2 = 7"算，APL 会与按各叶分别算不同。
    """
    traj, _, _ = _build_membrane_trajectory()
    top = traj.topology
    # 删掉下叶两个脂质：直接改 slice 比重建拓扑省事——保留其余所有原子。
    lower_heads = [
        int(a.index)
        for a in top.atoms
        if a.name == "P" and traj.xyz[0][a.index][2] < MIDPLANE
    ]
    drop_residues = {int(top.atom(i).residue.index) for i in lower_heads[:2]}
    keep = [
        int(a.index) for a in top.atoms if int(a.residue.index) not in drop_residues
    ]
    reduced = traj.atom_slice(keep)
    obs, diag = core.membrane_observables_from_trajectory(
        reduced, "MOL", pocket_atom_indices=_pocket_indices(reduced)
    )
    assert diag["n_upper"] == 8
    assert diag["n_lower"] == 6
    area = BOX_XY * BOX_XY
    expected = 0.5 * (area / 8 + area / 6)
    naive_half = area / 7.0
    assert obs["apl_nm2"]["values"][0] == pytest.approx(expected, rel=1e-5)
    assert not np.isclose(expected, naive_half, rtol=1e-5), (
        "构造前提：按各叶分别算与按总数/2 算必须可区分"
    )


def test_order_parameter_is_one_for_tails_parallel_to_the_normal():
    """尾链 C–C 键沿法向 → S = (3·1 − 1)/2 = 1。"""
    traj, _, _ = _build_membrane_trajectory()
    obs, diag = _extract(traj)
    assert obs["lipid_tail_order_parameter"]["values"][0] == pytest.approx(1.0, abs=1e-5)
    # 口径必须写清楚：这不是 S_CD。
    assert "NOT S_CD" in diag["order_parameter_definition"]


def test_tilt_is_zero_for_a_protein_aligned_with_the_normal():
    traj, _, _ = _build_membrane_trajectory()
    obs, _ = _extract(traj)
    assert obs["transmembrane_tilt_deg"]["values"][0] == pytest.approx(0.0, abs=1e-3)


def test_static_protein_and_ligand_give_zero_rmsd_everywhere():
    """蛋白与配体在合成轨迹里不动（只有脂质横向扩散）→ 三项 RMSD 恒为 0。"""
    traj, _, _ = _build_membrane_trajectory()
    obs, _ = _extract(traj)
    for name in (
        "protein_backbone_rmsd_nm",
        "pocket_rmsd_nm",
        "ligand_heavy_atom_rmsd_nm",
    ):
        assert max(obs[name]["values"]) == pytest.approx(0.0, abs=1e-5), name


def test_lateral_relaxation_timescale_is_recovered_from_the_synthetic_msd():
    """构造是二维随机行走（MSD = 4·D·t），提取器应反解出设定的弛豫时间尺度。

    这条同时钉住 §9 那句话的可操作性：弛豫时间尺度是从轨迹**算出来**的，
    不是让用户随口填一个数。

    ⚠️ 容差是 15%，不是此前的 1e-3。真正的扩散运动有统计误差，1e-3 那种精度只有
    "沿固定方向确定性位移 + 单参考帧估计器"这种**非扩散**构造才做得到 ——
    换句话说旧的 1e-3 是 fixture 与估计器互相配合出来的假精度。
    这里用 400 ns / 401 帧把独立时间原点数堆上去，才有资格谈 15%。
    """
    traj, _, _ = _build_membrane_trajectory(n_frames=401, total_time_ns=400.0)
    _, diag = _extract(traj)
    expected_ns = (
        core.LIPID_LATERAL_RELAXATION_REFERENCE_DISPLACEMENT_NM ** 2
        / (4.0 * LIPID_LATERAL_D_NM2_PER_NS)
    )
    assert expected_ns == pytest.approx(30.0, rel=1e-6)
    assert diag["lipid_lateral_relaxation_timescale_ns"] == pytest.approx(
        expected_ns, rel=0.15
    )
    details = diag["lipid_lateral_diffusion"]
    assert details["method"] == "time_averaged_msd_multiple_origins"
    assert details["lateral_diffusion_nm2_per_ns"] == pytest.approx(
        LIPID_LATERAL_D_NM2_PER_NS, rel=0.15
    )
    # 随机行走是纯扩散 → 幂律指数应接近 1（真实脂质是亚扩散，实测 0.80）。
    assert details["msd_power_law_exponent"] == pytest.approx(1.0, abs=0.15)
    # 实际用的拟合窗口必须落盘，否则"换窗口把数调好看"查不出来。
    assert details["fit_lag_window_source"] == "declared"
    assert details["fit_lag_window_ns"][0] == pytest.approx(
        core.LIPID_LATERAL_MSD_FIT_LAG_MIN_NS, rel=0.05
    )


def test_a_membrane_that_never_relaxes_laterally_is_reported_not_silently_dropped():
    """横向扩散为 0 → 弛豫时间尺度趋于无穷。**必须被如实报出。**

    MEM-11 把"预平衡 ≥ 1 × 弛豫时间"从硬门降为诊断（§9 原文只要求记录并论证，
    那个倍数是本实现自加的，且 τ 是方法依赖量）。**降级不等于不记录** ——
    这条测试就是钉住这一点：永不弛豫的膜不会再让 `passed` 变 False，
    但它的 τ 与比值必须原样出现在报告里，任何人一眼能看到。
    """
    traj, _, _ = _build_membrane_trajectory(lipid_lateral_d_nm2_per_ns=0.0)
    obs, diag = _extract(traj, equilibration_length_ns=1000.0)
    assert diag["lipid_lateral_relaxation_timescale_ns"] > 1.0e6

    report = core.evaluate_membrane_quality_gate(obs, diag)
    # 不再是硬门：它不出现在 checks 里。
    assert all(
        c["observable"] != "equilibration_length_ns" for c in report["checks"]
    ), "equilibration_length_ns 已退役为 diagnostics-only，不得塞回 checks"
    assert "equilibration_length_ns" not in report["failed_checks"]
    # 但必须如实落盘。
    rec = report["statistics"]["equilibration_vs_relaxation"]
    assert rec["is_gate"] is False
    assert rec["lipid_lateral_relaxation_timescale_ns"] > 1.0e6
    assert rec["ratio_equilibration_over_relaxation"] < 1.0
    assert "退役" in rec["retired_reason"] or "diagnostics-only" in rec["retired_reason"]


def test_times_are_reported_in_nanoseconds_not_picoseconds():
    traj, _, _ = _build_membrane_trajectory(total_time_ns=40.0)
    obs, _ = _extract(traj)
    assert obs["apl_nm2"]["times_ns"][-1] == pytest.approx(40.0, rel=1e-4)


# ---------------------------------------------------------------------------
# fail closed：宁可报错也不猜
# ---------------------------------------------------------------------------


def test_missing_time_axis_fails_closed_instead_of_using_frame_numbers():
    """§9 的判据定义在"末段 ≥ 20 ns"上，没有真实时间轴就无从判定。"""
    traj, _, _ = _build_membrane_trajectory(with_time=False)
    with pytest.raises(ValueError, match="不允许用帧号冒充 ns"):
        _extract(traj)


def test_missing_bonds_fails_closed_instead_of_faking_them_by_distance():
    """ATT-11 的教训：不用距离阈值冒充共价键。"""
    traj, _, _ = _build_membrane_trajectory(with_bonds=False)
    with pytest.raises(ValueError, match="不会.*用距离阈值冒充共价键"):
        _extract(traj)


def test_missing_pocket_definition_fails_closed():
    """口袋定义决定 pocket_rmsd 这一道门，不接受运行时默契。"""
    traj, _, _ = _build_membrane_trajectory()
    with pytest.raises(ValueError, match="必须显式提供 pocket_atom_indices"):
        core.membrane_observables_from_trajectory(traj, "MOL")


def test_unknown_ligand_resname_fails_closed():
    traj, _, _ = _build_membrane_trajectory()
    with pytest.raises(ValueError, match="找不到配体"):
        core.membrane_observables_from_trajectory(
            traj, "XYZ", pocket_atom_indices=_pocket_indices(traj)
        )


def test_missing_unitcell_fails_closed():
    traj, _, _ = _build_membrane_trajectory()
    traj.unitcell_lengths = None
    with pytest.raises(ValueError, match="没有 unitcell 信息"):
        _extract(traj)


def test_no_lipids_fails_closed():
    traj, _, _ = _build_membrane_trajectory()
    keep = [
        int(a.index)
        for a in traj.topology.atoms
        if str(a.residue.name).upper() != "POPC"
    ]
    reduced = traj.atom_slice(keep)
    with pytest.raises(ValueError, match="这不是膜体系"):
        core.membrane_observables_from_trajectory(
            reduced, "MOL", pocket_atom_indices=_pocket_indices(reduced)
        )


# ---------------------------------------------------------------------------
# 诊断量
# ---------------------------------------------------------------------------


def test_waters_inside_the_hydrophobic_core_are_counted():
    clean, _, _ = _build_membrane_trajectory(n_core_waters=0)
    _, diag_clean = _extract(clean)
    assert diag_clean["anomalous_pocket_water_count"] == 0

    dirty, _, _ = _build_membrane_trajectory(n_core_waters=3)
    _, diag_dirty = _extract(dirty)
    assert diag_dirty["anomalous_pocket_water_count"] == 3


def test_periodic_image_contact_is_flagged_when_the_water_slab_is_thin():
    """膜跨度接近盒高时水层不足 → 记为异常接触（judged as must_be_zero）。"""
    traj, _, _ = _build_membrane_trajectory()
    # 把盒高压到脂质跨度 + 1 nm（< 2.0 nm 阈值）。
    lipid_z = [
        traj.xyz[0][int(a.index)][2]
        for a in traj.topology.atoms
        if str(a.residue.name).upper() == "POPC"
    ]
    span = max(lipid_z) - min(lipid_z)
    lengths = np.array(traj.unitcell_lengths, dtype=np.float32)
    lengths[:, 2] = span + 1.0
    traj.unitcell_lengths = lengths
    _, diag = _extract(traj)
    assert diag["membrane_periodic_image_contacts"] == traj.n_frames


def test_density_profile_has_one_curve_per_component():
    traj, _, _ = _build_membrane_trajectory()
    _, diag = _extract(traj)
    profile = diag["density_profile_along_normal"]
    assert len(profile["bin_centers_nm"]) == 100
    for group in ("water", "lipid", "protein", "ion"):
        assert len(profile[group]) == 100, group
    assert sum(profile["lipid"]) > 0.0


def test_equilibration_length_defaults_to_the_trajectory_span_but_can_be_declared():
    traj, _, _ = _build_membrane_trajectory(total_time_ns=40.0)
    _, diag = _extract(traj)
    assert diag["equilibration_length_ns"] == pytest.approx(40.0, rel=1e-4)
    _, diag2 = _extract(traj, equilibration_length_ns=150.0)
    assert diag2["equilibration_length_ns"] == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# 端到端：提取器产出直接喂判定层
# ---------------------------------------------------------------------------


def test_extractor_output_feeds_the_quality_gate_directly():
    """两边的键名必须对得上——否则各自"完成"却接不起来。"""
    traj, _, _ = _build_membrane_trajectory(total_time_ns=40.0)
    obs, diag = _extract(traj, equilibration_length_ns=150.0)
    # 提供的量必须**恰好覆盖**判定层要求的必需集合。
    assert set(core.REQUIRED_MEMBRANE_QUALITY_OBSERVABLES) <= set(obs)
    for name in core.REQUIRED_MEMBRANE_QUALITY_DIAGNOSTICS:
        assert diag.get(name) is not None, name

    report = core.evaluate_membrane_quality_gate(
        obs,
        diag,
        # 门比的是**校正后**的 APL（含蛋白膜不能拿 raw 比纯脂文献值，MEM-03），
        # 所以这里当"文献值"用的也必须是校正后口径 —— 拿
        # `(BOX_XY**2)/N_PER_LEAFLET`（raw 口径）会因为口径不同而失败，
        # 那是对的行为，不是 bug。
        literature_apl_nm2=float(
            np.mean(obs[core.APL_PROTEIN_CORRECTED_OBSERVABLE]["values"])
        ),
    )
    assert report["passed"] is True, report["failed_checks"]


def test_drifting_apl_is_caught_end_to_end():
    """提取器 + 判定层串起来能抓到 APL 持续漂移。"""
    expected_apl = (BOX_XY * BOX_XY) / N_PER_LEAFLET
    # 每 ns 漂 1% 的 APL，远超 0.2 %/ns。
    traj, _, _ = _build_membrane_trajectory(
        apl_slope_per_ns=0.01 * expected_apl, total_time_ns=40.0
    )
    obs, diag = _extract(traj, equilibration_length_ns=150.0)
    report = core.evaluate_membrane_quality_gate(obs, diag)
    assert report["passed"] is False
    assert "apl_nm2" in report["failed_checks"]


def test_coion_observables_are_produced_when_an_index_is_given():
    traj, coion_index, _ = _build_membrane_trajectory(coion_z=10.5)
    obs, diag = _extract(
        traj, coion_atom_index=coion_index, equilibration_length_ns=150.0
    )
    for name in core.COION_MEMBRANE_QUALITY_OBSERVABLES:
        assert name in obs, name
    assert "coion_z_histogram" in diag
    # co-ion 在 z=10.5、中面 6.0 → |Δz| = 4.5 nm，满足 §13.1 的 ≥ 3.0 nm。
    assert obs["coion_abs_z_from_midplane_nm"]["values"][0] == pytest.approx(4.5, abs=1e-4)

    report = core.evaluate_membrane_quality_gate(obs, diag, require_coion=True)
    assert "coion_abs_z_from_midplane_nm" not in report["failed_checks"]


def test_coion_sitting_inside_the_membrane_is_caught_end_to_end():
    traj, coion_index, _ = _build_membrane_trajectory(coion_z=MIDPLANE + 0.2)
    obs, diag = _extract(
        traj, coion_atom_index=coion_index, equilibration_length_ns=150.0
    )
    report = core.evaluate_membrane_quality_gate(obs, diag, require_coion=True)
    assert report["passed"] is False
    assert "coion_abs_z_from_midplane_nm" in report["failed_checks"]


# ---------------------------------------------------------------------------
# 按分子的叶片划分（`composition["molecules_by_role"]["lipid"]`）
#
# 上面所有测试都**不**传 `composition`，所以走的全是按残基那条分支。分子分支
# 因此零覆盖，`leaflet_composition` 只读残基分支的局部变量 `head_by_residue`
# 这个缺陷一路跑到了真实膜体系上：memtest 的 §9 质量门在 2026-07-31 与 08-02
# 两次运行里都崩在 `UnboundLocalError`，报告落成 `{"evaluated": false}`。
# 下面这组测试钉住两条分支各自的行为，以及"两者不许再分叉"。
# ---------------------------------------------------------------------------


def test_molecule_path_leaflet_composition_is_keyed_by_molecule_name():
    """MEM-01 的直接回归：传 composition 时提取器必须跑完，不许 UnboundLocalError。"""
    traj, _, _ = _build_membrane_trajectory()
    obs, diag = _extract(traj, composition=_composition_for(traj))

    assert diag["lipid_unit"] == "molecule"
    assert diag["n_upper"] == N_PER_LEAFLET
    assert diag["n_lower"] == N_PER_LEAFLET
    # 键是 moleculetype 名，计数按分子——不是构成残基数。
    assert diag["leaflet_composition"] == {
        "upper": {"POPC": N_PER_LEAFLET},
        "lower": {"POPC": N_PER_LEAFLET},
    }
    # APL 仍按每叶分子数算。
    assert obs["apl_nm2"]["values"][0] == pytest.approx(
        (BOX_XY * BOX_XY) / N_PER_LEAFLET, rel=1e-5
    )


def test_residue_path_leaflet_composition_stays_keyed_by_residue_name():
    """不传 composition 的既有行为逐项不变（§7.7 的口径）。"""
    traj, _, _ = _build_membrane_trajectory()
    obs, diag = _extract(traj)

    assert diag["lipid_unit"] == "residue"
    assert diag["leaflet_composition"] == {
        "upper": {"POPC": N_PER_LEAFLET},
        "lower": {"POPC": N_PER_LEAFLET},
    }


def test_both_paths_agree_when_one_molecule_is_one_residue():
    """一分子=一残基时两条分支必须给出同一组数——分叉了就是有一条错。"""
    traj, _, _ = _build_membrane_trajectory()
    obs_res, diag_res = _extract(traj)
    obs_mol, diag_mol = _extract(traj, composition=_composition_for(traj))

    for key in ("n_upper", "n_lower", "leaflet_composition"):
        assert diag_res[key] == diag_mol[key], key
    for key in ("apl_nm2", "bilayer_thickness_nm"):
        assert obs_res[key]["values"] == pytest.approx(obs_mol[key]["values"], rel=1e-6)


def test_modular_lipid_residues_require_the_molecule_path():
    """Amber Lipid21 模块化命名（§0.5.4）：按残基必须报错，按分子才对。

    一个 POPC 分子 = 头基残基 `PC` + 尾链残基 `PA`。按残基分叶时尾链残基
    没有磷原子 → 找不到头基参考原子 → fail closed（**不是**静默不计入）。
    """
    traj, _, _ = _build_membrane_trajectory(modular_lipid_residues=True)

    with pytest.raises(ValueError, match="找不到头基参考原子"):
        _extract(traj)

    obs, diag = _extract(traj, composition=_composition_for(traj))
    assert diag["lipid_unit"] == "molecule"
    # 2 × N_PER_LEAFLET 个脂质**残基**，但只有 N_PER_LEAFLET 个每叶脂质**分子**。
    assert diag["n_upper"] == N_PER_LEAFLET
    assert diag["n_lower"] == N_PER_LEAFLET
    assert diag["leaflet_composition"] == {
        "upper": {"POPC": N_PER_LEAFLET},
        "lower": {"POPC": N_PER_LEAFLET},
    }
    # 按残基会把 APL 算小一倍——这就是 §0.5.4 那条"APL 错 3 倍"的同一机理。
    assert obs["apl_nm2"]["values"][0] == pytest.approx(
        (BOX_XY * BOX_XY) / N_PER_LEAFLET, rel=1e-5
    )


def test_molecule_path_output_still_feeds_the_quality_gate():
    """分子路径的产出必须能直接喂判定层——否则又是"各自完成却接不起来"。"""
    traj, _, _ = _build_membrane_trajectory(modular_lipid_residues=True)
    obs, diag = _extract(
        traj, composition=_composition_for(traj), equilibration_length_ns=150.0
    )
    report = core.evaluate_membrane_quality_gate(obs, diag)
    assert report["passed"] is True, report["failed_checks"]


# ---------------------------------------------------------------------------
# 时间轴（MEM-08）
#
# mdtraj 读 DCD **不传播真实步长**：`traj.time` 是整数帧号 [0, 1, 2, …]。
# 实测 memtest 那条 10 ns / 500 帧（10000 步 × 2 fs = 20 ps/帧）的轨迹，
# `traj.time` 就是 [0…499]，于是时间轴被当成 0.499 ns —— 小 20 倍。
# 原先的守卫只查"存在且单调递增"，帧号完全满足，所以它对 DCD 是 fail-open。
# 时间轴错一个倍数，两道门往**相反**方向坏：末段窗口过严，而
# "预平衡 ≥ 一个脂质横向弛豫时间"过松（MSD 拟合的 D 被同一倍数放大）。
# ---------------------------------------------------------------------------


def test_integer_frame_indices_are_rejected_as_a_time_axis():
    traj, _, _ = _build_membrane_trajectory()
    traj.time = np.arange(traj.n_frames)  # mdtraj 读 DCD 的原样签名：整数帧号
    assert np.issubdtype(np.asarray(traj.time).dtype, np.integer)
    with pytest.raises(ValueError, match="帧号"):
        _extract(traj)


def test_declared_frame_interval_rebuilds_the_time_axis():
    """传了 frame_interval_ps 就以它为准，并把来源写进 diagnostics。"""
    traj, _, _ = _build_membrane_trajectory(n_frames=41, total_time_ns=40.0)
    traj.time = np.arange(traj.n_frames)  # 故意把时间轴弄成帧号
    obs, diag = _extract(traj, frame_interval_ps=1000.0)  # 1 ns/帧

    assert diag["time_axis_source"] == "declared_frame_interval"
    assert diag["frame_interval_ps"] == pytest.approx(1000.0)
    assert obs["apl_nm2"]["times_ns"][:3] == pytest.approx([0.0, 1.0, 2.0])
    assert diag["trajectory_span_ns"] == pytest.approx(40.0)


def test_trajectory_time_field_is_still_used_when_no_interval_is_declared():
    """既有行为不变：真实 float 时间轴照用，来源如实记录。"""
    traj, _, _ = _build_membrane_trajectory()
    obs, diag = _extract(traj)
    assert diag["time_axis_source"] == "trajectory_time_field"
    assert diag["frame_interval_ps"] is None
    assert diag["trajectory_span_ns"] == pytest.approx(40.0, rel=1e-4)


def test_the_memtest_dcd_time_axis_is_off_by_the_reporter_interval():
    """复现 memtest 那条轨迹的具体数字：500 帧 × 20 ps = 10 ns，不是 0.499 ns。"""
    traj, _, _ = _build_membrane_trajectory(n_frames=500, total_time_ns=10.0)
    traj.time = np.arange(traj.n_frames)  # DCD 原样

    with pytest.raises(ValueError, match="帧号"):
        _extract(traj)

    interval_ps = core.pre_equilibration_frame_interval_ps()
    assert interval_ps == pytest.approx(20.0)  # 10000 步 × 2 fs
    _, diag = _extract(traj, frame_interval_ps=interval_ps)
    # 499 帧间隔 × 20 ps = 9.98 ns；被当成帧号时是 0.499 ns，正好差 20 倍。
    assert diag["trajectory_span_ns"] == pytest.approx(9.98)
    assert diag["trajectory_span_ns"] / 0.499 == pytest.approx(20.0, rel=1e-3)


def test_frame_interval_must_be_positive_and_finite():
    traj, _, _ = _build_membrane_trajectory()
    for bad in (0.0, -20.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="frame_interval_ps"):
            _extract(traj, frame_interval_ps=bad)


def test_pre_equilibration_frame_interval_is_shared_with_the_reporter():
    """写轨迹的一侧与判门的一侧必须引用同一组常量，不许各写一个字面量。"""
    import ast
    import inspect
    import abfe_pipeline

    assert core.PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS == 10000
    assert core.PRE_EQUILIBRATION_TIMESTEP_PS == pytest.approx(0.002)
    assert core.pre_equilibration_frame_interval_ps() == pytest.approx(20.0)

    # `pre_equilibrate` 里不许再出现裸的 DCDReporter 间隔字面量。
    source = inspect.getsource(abfe_pipeline.ABFEPipeline.pre_equilibrate)
    tree = ast.parse(inspect.cleandoc(source))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if name != "DCDReporter":
            continue
        interval = node.args[1]
        assert isinstance(interval, ast.Name), (
            "pre_equilibrate 的 DCDReporter 间隔必须引用常量"
            " PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS，不能写字面量——"
            "§9 的时间轴要靠它重建，两边各写一个数就会静默错开倍数。"
        )
        assert interval.id == "PRE_EQUILIBRATION_TRAJ_INTERVAL_STEPS"


# ---------------------------------------------------------------------------
# APL 的蛋白横截面校正（MEM-03 / §13.3）
#
# raw APL = 横向面积 / 每叶脂质数，把跨膜蛋白占掉的面积也摊给了脂质。实测
# memtest（PROA + 90 POPC）raw = 0.826 nm² vs POPC 纯脂文献 ≈ 0.645 —— 差值不是
# 体系有问题，是口径不对。所以绝对值门只能比**校正后**的 APL。
# ---------------------------------------------------------------------------


def test_nearest_owner_area_splits_the_plane_halfway():
    """一个蛋白原子 vs 一个脂质原子：分界线落在两者中间，各得一半面积。

    这就是"没有探针半径"的含义 —— 边界由两类原子的相对位置决定，不由某个可调的
    外扩半径决定。外扩法会沿蛋白周长多算一圈（实测让校正后 APL 偏低 12.6%）。
    """
    area = core._nearest_owner_area_nm2(
        np.array([[1.0, 2.0]]), np.array([[3.0, 2.0]]), (4.0, 4.0)
    )
    assert area == pytest.approx(0.5 * 4.0 * 4.0, rel=0.02)


def test_nearest_owner_area_is_zero_without_protein_atoms():
    area = core._nearest_owner_area_nm2(
        np.empty((0, 2)), np.array([[2.0, 2.0]]), (4.0, 4.0)
    )
    assert area == 0.0


def test_nearest_owner_area_wraps_around_the_periodic_boundary():
    """横向是周期的：贴边的蛋白原子不该因为盒边被截掉归属。"""
    middle = core._nearest_owner_area_nm2(
        np.array([[2.0, 2.0]]), np.array([[2.0, 0.0]]), (4.0, 4.0)
    )
    at_edge = core._nearest_owner_area_nm2(
        np.array([[0.0, 0.0]]), np.array([[0.0, 2.0]]), (4.0, 4.0)
    )
    assert at_edge == pytest.approx(middle, rel=0.02)


def test_nearest_owner_area_has_no_probe_radius_parameter():
    """契约：这个函数**不接受**探针半径。加回来就会重新引入周长偏差。"""
    import inspect

    params = set(inspect.signature(core._nearest_owner_area_nm2).parameters)
    assert "radius_nm" not in params
    assert not hasattr(core, "PROTEIN_CROSS_SECTION_ATOM_RADIUS_NM"), (
        "探针半径常量已退役：外扩定义沿蛋白周长多算一圈，实测让校正后 APL "
        "偏低 12.6%，那会让 §13.3 的绝对值门因方法偏差而不过"
    )


def test_corrected_apl_subtracts_the_protein_cross_section():
    """校正后 APL < raw APL，且差值等于扣掉的蛋白面积 / 每叶脂质数。"""
    traj, _, _ = _build_membrane_trajectory()
    obs, diag = _extract(traj, composition=_composition_for(traj))

    raw = np.asarray(obs["apl_nm2"]["values"])
    corrected = np.asarray(obs["apl_protein_corrected_nm2"]["values"])
    assert np.all(corrected < raw), "有跨膜蛋白时校正后必须更小"

    cross = 0.5 * (
        diag["protein_cross_section_upper_nm2_mean"]
        + diag["protein_cross_section_lower_nm2_mean"]
    )
    assert cross > 0.0
    assert float(np.mean(raw - corrected)) == pytest.approx(
        cross / N_PER_LEAFLET, rel=1e-6
    )


def test_corrected_apl_equals_raw_apl_without_protein_in_the_slab():
    """蛋白完全在膜外时校正应当无效果——不能凭"有蛋白"就一律扣面积。"""
    traj, _, _ = _build_membrane_trajectory()
    # 把蛋白整体搬到水层（z ≈ 11），两个叶片 slab 内都不再有蛋白原子。
    protein = traj.topology.select("protein")
    traj.xyz[:, protein, 2] = 11.0
    obs, diag = _extract(traj, composition=_composition_for(traj))
    assert diag["protein_cross_section_upper_nm2_mean"] == pytest.approx(0.0)
    assert diag["protein_cross_section_lower_nm2_mean"] == pytest.approx(0.0)
    assert obs["apl_protein_corrected_nm2"]["values"] == pytest.approx(
        obs["apl_nm2"]["values"], rel=1e-9
    )


def test_literature_check_uses_the_corrected_apl_not_the_raw_one():
    """§13.3 的绝对值门必须比校正后的值，并在 criterion 里说清楚。

    合成体系的蛋白是 24 个原子叠在同一个 XY 点上，横截面只有约 0.09 nm²，
    raw 与校正后差 0.6% —— 判不出两者的区别。所以这里把校正序列**显式**拉开到
    memtest 的真实量级（raw 0.826 / 校正后 0.645，差 22%），
    这样"门比的是哪一条"才有唯一答案。
    """
    traj, _, _ = _build_membrane_trajectory()
    obs, diag = _extract(
        traj, composition=_composition_for(traj), equilibration_length_ns=150.0
    )
    times = obs["apl_nm2"]["times_ns"]
    obs["apl_nm2"] = {"times_ns": times, "values": [0.826] * len(times)}
    obs["apl_protein_corrected_nm2"] = {
        "times_ns": times,
        "values": [0.645] * len(times),
    }

    report = core.evaluate_membrane_quality_gate(obs, diag, literature_apl_nm2=0.645)
    lit = [
        c
        for c in report["checks"]
        if c["criterion"].startswith("deviation_from_literature_percent")
    ]
    assert len(lit) == 1
    assert lit[0]["observable"] == core.APL_PROTEIN_CORRECTED_OBSERVABLE
    assert lit[0]["criterion"] == "deviation_from_literature_percent"
    assert lit[0]["measured"] == pytest.approx(0.0, abs=1e-9)
    assert lit[0]["passed"] is True

    # 拿 raw APL 当文献值就应当**失败**——否则等于这道门根本没换口径。
    report_raw = core.evaluate_membrane_quality_gate(obs, diag, literature_apl_nm2=0.826)
    assert core.APL_PROTEIN_CORRECTED_OBSERVABLE in report_raw["failed_checks"]


def test_missing_corrected_series_falls_back_but_says_so():
    """老报告/手工观测量没有校正序列时照样判，但 criterion 标明未校正。"""
    traj, _, _ = _build_membrane_trajectory()
    obs, diag = _extract(traj, equilibration_length_ns=150.0)
    obs.pop(core.APL_PROTEIN_CORRECTED_OBSERVABLE)
    report = core.evaluate_membrane_quality_gate(obs, diag, literature_apl_nm2=0.5)
    lit = [
        c
        for c in report["checks"]
        if c["criterion"].startswith("deviation_from_literature_percent")
    ]
    assert len(lit) == 1
    assert lit[0]["criterion"] == "deviation_from_literature_percent_uncorrected"
    assert lit[0]["observable"] == "apl_nm2"


def test_cross_section_sensitivity_is_reported_with_the_correction():
    """方法参数敏感性必须随报告落盘——点估计单看会显得比实际确定。"""
    traj, _, _ = _build_membrane_trajectory()
    _, diag = _extract(traj, composition=_composition_for(traj))
    sens = diag["apl_protein_cross_section_sensitivity"]
    assert sens["method"] == "nearest_reference_atom_partition"
    assert sens["grid_nm"] == core.PROTEIN_CROSS_SECTION_GRID_NM
    assert sens["n_frames_sampled"] >= 2
    assert sens["n_protein_reference_atoms"] > 0
    assert sens["n_lipid_reference_atoms"] > 0
    # 栅格是唯一的方法参数：2× 粗栅格复算的结果必须与细栅格接近，
    # 否则说明离散化误差不可忽略、栅格该调细。
    assert sens["apl_nm2_coarse_grid"] == pytest.approx(
        sens["apl_nm2_fine_grid"], rel=0.05
    )


def test_correction_definition_string_is_recorded():
    traj, _, _ = _build_membrane_trajectory()
    _, diag = _extract(traj, composition=_composition_for(traj))
    text = diag["apl_correction_definition"]
    assert "NO probe radius" in text
    assert "nearest-reference-atom" in text
    assert str(core.PROTEIN_CROSS_SECTION_GRID_NM) in text


def test_cross_section_parameters_reach_provenance():
    """§13：方法参数必须与阈值一起进 provenance，不许只活在代码里。"""
    payload = core.acceptance_thresholds_payload()["membrane_quality_gate"]
    assert payload["protein_cross_section_grid_nm"] == (
        core.PROTEIN_CROSS_SECTION_GRID_NM
    )
    assert payload["protein_cross_section_grid_sensitivity_factor"] == (
        core.PROTEIN_CROSS_SECTION_GRID_SENSITIVITY_FACTOR
    )


# ---------------------------------------------------------------------------
# 坐标污染（MEM-10）：提取器不得修改调用方的轨迹
#
# `mdtraj.Trajectory.superpose()` **原地改 xyz 并返回 self**。原先提取器里那行
# `aligned = traj.superpose(traj, 0, atom_indices=protein_backbone)` 之后，所有读
# `traj.xyz` 的量都在用对齐到蛋白骨架的坐标，而 midplane/upper_z/lower_z 是对齐
# 之前算的。实测（memtest 100 ns）脂质横向弛豫 τ 被从 11.57 放大到 139.36 ns，
# 直接把 §9 质量门判失败；倾角、蛋白横截面、核内水、水层间隙、密度分布同时错配。
# ---------------------------------------------------------------------------


def test_extractor_does_not_mutate_the_caller_trajectory():
    """根因契约：跑完提取器，传入的 traj.xyz 必须逐位不变。

    这条直接钉住 MEM-10 的根因。任何人再写一行原地 `superpose` 都会在这里失败。
    """
    traj, _, _ = _build_membrane_trajectory()
    before = traj.xyz.copy()
    _extract(traj, composition=_composition_for(traj))
    np.testing.assert_array_equal(
        traj.xyz,
        before,
        err_msg=(
            "提取器修改了调用方的坐标。mdtraj 的 superpose() 是原地操作——"
            "要对齐请先 atom_slice 出子集副本（见 MEM-10）"
        ),
    )


def test_rigid_body_motion_of_the_protein_does_not_change_lipid_relaxation():
    """给蛋白加整体刚性平移：脂质弛豫时间尺度不该因此改变。

    污染版本会把蛋白的整体运动通过对齐"转嫁"到脂质坐标上，从而压低脂质 MSD、
    放大 τ（实测放大 12 倍）。修好后两者应当一致。
    """
    traj_a, _, _ = _build_membrane_trajectory()
    traj_b, _, _ = _build_membrane_trajectory()
    protein = traj_b.topology.select("protein")
    drift = np.linspace(0.0, 1.5, traj_b.n_frames)          # 沿 x 漂 1.5 nm
    traj_b.xyz[:, protein, 0] += drift[:, None].astype(np.float32)

    _, diag_a = _extract(traj_a)
    _, diag_b = _extract(traj_b)
    assert diag_b["lipid_lateral_relaxation_timescale_ns"] == pytest.approx(
        diag_a["lipid_lateral_relaxation_timescale_ns"], rel=1e-6
    )


def test_pocket_and_ligand_rmsd_measure_pose_drift_not_internal_conformation():
    """MEM-13：口袋/配体 RMSD 必须是"对齐骨架后不重拟合"的位移。

    原先用 `md.rmsd(..., atom_indices=pocket)`，它会在口袋/配体自身上再做一次最优
    拟合，测到的是内部构象变化而不是相对受体的 pose 漂移（实测配体差 1.7 倍）。
    这里把整个配体**刚性平移**：pose 漂移必须跟着变大，而重拟合口径会给 0。
    """
    traj, _, ligand_atoms = _build_membrane_trajectory()
    shift = 0.20
    traj.xyz[traj.n_frames // 2 :, ligand_atoms, 0] += np.float32(shift)
    obs, _ = _extract(traj)
    ligand_series = np.asarray(obs["ligand_heavy_atom_rmsd_nm"]["values"])
    # 后半程整体平移 0.20 nm，刚体平移下 pose 漂移就等于位移本身。
    assert ligand_series[-1] == pytest.approx(shift, rel=0.05)
    assert ligand_series[0] == pytest.approx(0.0, abs=1e-6)


def test_backbone_rmsd_is_still_the_fitted_rmsd():
    """骨架 RMSD 保持 `md.rmsd` 口径：整体刚性运动不该让它变大。"""
    traj, _, _ = _build_membrane_trajectory()
    protein = traj.topology.select("protein")
    traj.xyz[:, protein, 1] += np.linspace(0.0, 2.0, traj.n_frames)[:, None].astype(
        np.float32
    )
    obs, _ = _extract(traj)
    assert float(np.max(obs["protein_backbone_rmsd_nm"]["values"])) < 1e-3


def test_pure_lipid_apl_reference_is_a_diagnostic_not_a_gate():
    """MEM-12：与纯脂文献 APL 的偏差只落诊断，不判 pass/fail。"""
    traj, _, _ = _build_membrane_trajectory()
    obs, diag = _extract(
        traj, composition=_composition_for(traj), equilibration_length_ns=150.0
    )
    # 故意给一个差很远的参考值：仍然不许影响 passed。
    report = core.evaluate_membrane_quality_gate(
        obs, diag, pure_lipid_reference_apl_nm2=0.10
    )
    rec = report["statistics"]["apl_vs_pure_lipid_literature"]
    assert rec["is_gate"] is False
    assert rec["pure_lipid_reference_apl_nm2"] == pytest.approx(0.10)
    assert rec["deviation_percent"] > 100.0
    assert rec["apl_caliber"] == core.APL_PROTEIN_CORRECTED_OBSERVABLE
    assert report["passed"] is True, report["failed_checks"]
    assert not any(
        c["criterion"].startswith("deviation_from_literature_percent")
        for c in report["checks"]
    ), "诊断参考值不得变成 checks 里的门（那是 literature_apl_nm2 的职责）"


def test_frame_count_reconciliation_is_gone_and_must_not_come_back():
    """MEM-17 已删除（2026-08-03，用户决定）：质量门里不再有帧数对账。

    背景与理由（免得下一个人"顺手补回来"）：
    * resume 的重复帧是**真的** —— 实测那条 100 ns 是 5001 帧而非 5000，
      两次 resume（38,500,000 / 40,500,000 续跑）在 `(40500000,40515000]`
      区间重写了第 40,510,000 步那一帧，重复帧在 DCD 索引 4050/4051。
    * 但对账拦住的是**主线**，代价是重跑 8 h 的预平衡；而根因在
      `abfe_pipeline.pre_equilibrate` 的 `DCDReporter(append=resume_from_chk)`
      没有先把 DCD 截断到 checkpoint 对应的帧边界。要修就修那里。
    * "第 0 帧也算所以 5001 是对的"这个说法不成立：DCD 头 `ISTART=10000`、
      `NSAVC=10000`，monitor 首行是 5000 而不是 0 —— OpenMM 的 reporter 从第
      `interval` 步开始写，不写初始帧。整除不 +1 本来是对的。
    """
    import inspect

    src = inspect.getsource(core.run_membrane_quality_gate)
    assert "预平衡轨迹有" not in src, (
        "帧数对账被加回来了：resume 的重复帧会让它 fail closed 拦住主线，"
        "而根因在 abfe_pipeline.pre_equilibrate 的 DCDReporter(append=True) 不截断。"
    )
    assert "MEM-17 已移除" in src, "删除的缘由注释别一起删掉，否则下个人会再加一遍"
    # 帧数已不再是判据 ⇒ 上游也不该再算这个数喂进来。
    from pathlib import Path

    runabfe_src = (Path(__file__).absolute().parents[1] / "runabfe.py").read_text(
        encoding="utf-8"
    )
    assert '"expected_pre_equilibration_frames":' not in runabfe_src
