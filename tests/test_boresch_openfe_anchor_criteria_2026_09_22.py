"""借 openfe 的锚点判据：逐帧全过 + 按涨落排序（09-24 起为默认档；"orb" 仍可显式选）。

我们原来的判据只在 **frame 0** 上校验几何，然后按 RMSF/距离/序列分散度加权打分。
一帧合法不代表整条轨迹合法——二面角在 frame 0 是 170°、后面绕过 ±π 的候选照收，
事后再靠 wrap 收拾。openfe 在入口就把它排掉。
"""

import numpy as np
import pytest

import abfe_core


pytestmark = pytest.mark.cpu_only


def _estimator():
    est = abfe_core.OrbBoreschEstimator.__new__(abfe_core.OrbBoreschEstimator)
    est.config = dict(abfe_core.OrbBoreschEstimator.DEFAULT_CONFIG) \
        if hasattr(abfe_core.OrbBoreschEstimator, "DEFAULT_CONFIG") else {}
    est.config.setdefault("dist_reject_nm", 0.40)
    est.T = 300.0
    est.gas_constant_kj_per_mol_k = 0.008314462618
    return est


def _traj(frames_xyz):
    class _T:
        xyz = np.asarray(frames_xyz, dtype=np.float32)
    return _T()


def _six_atom_frames(n=50, jitter=0.002, seed=0, h2=(0.0, 0.3, 0.3),
                     h1=(0.0, 0.3, 0.0)):
    """一段轨迹，原子顺序 = openfe_anchor_metrics 的解包顺序 H0,H1,H2,G0,G1,G2。

    默认几何：θA=90°, θB=135°, φA=−90°, φB=0°, φC=121° —— 全部远离 ±180°。
    """
    rng = np.random.default_rng(seed)
    base = np.array([
        [0.0, 0.0, 0.0],   # H0
        list(h1),          # H1
        list(h2),          # H2
        [0.8, 0.0, 0.0],   # G0
        [1.1, 0.3, 0.0],   # G1
        [1.4, 0.3, 0.35],  # G2
    ], dtype=float)
    return np.repeat(base[None, :, :], n, axis=0) + rng.normal(scale=jitter, size=(n, 6, 3))


def test_circular_std_is_wraparound_safe():
    est = _estimator()
    near_pi = np.array([np.pi - 0.01, -np.pi + 0.01, np.pi - 0.02])
    # 线性 std 会算成 ~π（绕边界），圆标准差必须很小
    assert float(np.std(near_pi)) > 1.0
    assert est._circular_std_rad(near_pi) < 0.1


def test_a_dihedral_parked_near_pi_is_rejected():
    est = _estimator()
    # H2 摆到 −x 一侧 ⟹ φA ≈ 176°，紧贴 ±π 缠绕边界
    ok, why, _score, detail = est.openfe_anchor_metrics(
        [0, 1, 2], [3, 4, 5], _traj(_six_atom_frames(h2=(-0.3, 0.3, 0.02))))
    assert not ok and "二面角" in why, (ok, why, detail)


def test_a_well_behaved_anchor_is_accepted_and_scored_by_fluctuation():
    est = _estimator()
    tight = est.openfe_anchor_metrics([0, 1, 2], [3, 4, 5], _traj(_six_atom_frames(jitter=0.002)))
    loose = est.openfe_anchor_metrics([0, 1, 2], [3, 4, 5], _traj(_six_atom_frames(jitter=0.02, seed=3)))
    assert tight[0] is True, tight[1]
    # 涨落越大分越高（排序时取负），松的那组必须比紧的差
    assert loose[2] > tight[2]


def test_flat_angle_is_rejected_only_when_force_constant_is_known():
    est = _estimator()
    # 让 H1-H0-G0 接近 180°（共线）⟹ sinθ 奇点
    # H1 挪到 H0 的另一侧 ⟹ θA = angle(H1,H0,G0) ≈ 180°（共线、sinθ→0）
    frames = _six_atom_frames(h1=(-0.3, 0.0, 0.0))
    ok_nok, why, _s, _d = est.openfe_anchor_metrics(
        [0, 1, 2], [3, 4, 5], _traj(frames),
        k_angle=abfe_core.OrbBoreschEstimator.OPENFE_FIXED_KANGLE_KJ_PER_MOL_RAD2,
        temperature_k=300.0)
    assert not ok_nok and ("平" in why or "二面角" in why), why


def test_boresch_modes_default_to_openfe_route():
    est = _estimator()
    assert est.config["boresch_anchor_mode"] == "openfe"
    assert est.config["boresch_force_constant_mode"] == "openfe_fixed"
    assert abfe_core.DEFAULT_BORESCH_ANCHOR_MODE == "openfe"
    assert abfe_core.DEFAULT_BORESCH_FORCE_CONSTANT_MODE == "openfe_fixed"


# ---------------------------------------------------------------------------
# 配体侧锚点池：优先芳香环（芳香性由 RDKit 判，不是手写几何代理）
# ---------------------------------------------------------------------------

def _mol_traj(smiles="c1ccccc1CCC", n=30, ring_jitter=0.003, tail_jitter=0.20, seed=1):
    """用 RDKit 生成 3D 构象，再搭成 mdtraj 轨迹；环刚性、尾巴乱晃。"""
    import mdtraj as md
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    xyz = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())]) * 0.1  # Å→nm

    top = md.Topology()
    res = top.add_residue("LIG", top.add_chain())
    elem = {"C": md.element.carbon, "H": md.element.hydrogen,
            "N": md.element.nitrogen, "O": md.element.oxygen}
    atoms = [top.add_atom(f"{a.GetSymbol()}{i}", elem[a.GetSymbol()], res)
             for i, a in enumerate(mol.GetAtoms())]
    for b in mol.GetBonds():
        top.add_bond(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])

    aromatic = [i for i, a in enumerate(mol.GetAtoms()) if a.GetIsAromatic()]
    rng = np.random.default_rng(seed)
    frames = np.repeat(xyz[None, :, :], n, axis=0)
    scale = np.full(mol.GetNumAtoms(), tail_jitter)
    scale[aromatic] = ring_jitter
    frames += rng.normal(scale=scale[None, :, None], size=frames.shape)
    return md.Trajectory(frames.astype(np.float32), top), top, aromatic


def test_ligand_pool_is_the_rdkit_aromatic_ring():
    traj, top, aromatic = _mol_traj()
    est = _estimator()
    est.config["rmsf_cutoff_nm"] = 0.15
    pool, note = est.openfe_ligand_anchor_pool(top, traj, "LIG")
    # 池子 = 芳香环上的重原子；柔性丙基尾巴不在里面
    assert set(pool) == set(aromatic), (pool, aromatic, note)
    assert "芳香环" in note


def test_ligand_pool_falls_back_when_the_aromatic_ring_is_too_floppy():
    traj, top, _ = _mol_traj()
    est = _estimator()
    est.config["rmsf_cutoff_nm"] = 0.0005          # 连刚性芳香环也过不了
    pool, note = est.openfe_ligand_anchor_pool(top, traj, "LIG")
    heavy = [int(i) for i in top.select("resname LIG and not element H")]
    assert set(pool) == set(heavy), (pool, note)
    assert "退回全部重原子" in note


# ---------------------------------------------------------------------------
# 生产走的是 source=simple ⟹ GeometricRestraintEstimator，两个键必须接在它上面
# ---------------------------------------------------------------------------

def test_production_estimator_takes_both_mode_switches():
    import inspect
    import runabfe
    est = abfe_core.GeometricRestraintEstimator(
        temperature=300.0, force_constant_mode="openfe_fixed", anchor_mode="openfe")
    assert est.force_constant_mode == "openfe_fixed"
    assert est.anchor_mode == "openfe"
    # 默认档 = openfe 路线（09-24 起）
    assert abfe_core.GeometricRestraintEstimator(temperature=300.0).force_constant_mode == "openfe_fixed"
    assert abfe_core.GeometricRestraintEstimator(temperature=300.0).anchor_mode == "openfe"
    # runabfe 的 simple/fluctuation 分支必须把 config 的两个键传进去
    src = inspect.getsource(runabfe)
    i = src.index("estimator = GeometricRestraintEstimator(")
    call = src[i:i + 600]
    assert 'config.get("boresch_force_constant_mode", DEFAULT_BORESCH_FORCE_CONSTANT_MODE)' in call
    assert 'config.get("boresch_anchor_mode", DEFAULT_BORESCH_ANCHOR_MODE)' in call


def test_fixed_mode_bypasses_the_fitted_clip_floor():
    """固定档必须给 4184 / 334.72，而不是被 clip 到 [10, 1000] 的拟合值。"""
    assert abfe_core.OPENFE_FIXED_KR_KJ_PER_MOL_NM2 == pytest.approx(4184.0)
    assert abfe_core.OPENFE_FIXED_KANGLE_KJ_PER_MOL_RAD2 == pytest.approx(334.72)
    # 两个估算器引用同一份常量，不各写各的
    assert (abfe_core.OrbBoreschEstimator.OPENFE_FIXED_KANGLE_KJ_PER_MOL_RAD2
            == abfe_core.OPENFE_FIXED_KANGLE_KJ_PER_MOL_RAD2)


def test_fixed_force_constants_are_not_re_clipped_at_the_end():
    """固定档必须绕过最终裁剪 —— 那里 kr 的上限是 2000，会把 4184 裁掉。"""
    import inspect
    import runabfe
    src = inspect.getsource(runabfe)
    i = src.index("post_clip_ranges = {")
    head = src[max(0, i - 2500):i]
    assert '_fixed_fc_mode = str(' in head
    assert 'config.get("boresch_force_constant_mode", DEFAULT_BORESCH_FORCE_CONSTANT_MODE)) == "openfe_fixed"' in head
    assert 'if "force_constants" in boresch and not _fixed_fc_mode:' in head
    # 裁剪区间本身没动（拟合档行为不变）
    assert '"kr": [100.0, 2000.0],' in src[i:i + 300]


# ---------------------------------------------------------------------------
# [2026-09-24] openfe 档的候选搜索照 openfe 重写（`_openfe_select_combo`）
# ---------------------------------------------------------------------------

def _rigid_motion(frames, seed=7, shift_nm=0.3, max_rot_rad=0.5):
    """每帧整体转一下、平移一下（形状不变）——局部 RMSF 应当看不见它。"""
    rng = np.random.default_rng(seed)
    out = np.empty_like(frames)
    for f in range(len(frames)):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        a = rng.uniform(-max_rot_rad, max_rot_rad)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        Rm = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
        c = frames[f].mean(axis=0)
        out[f] = (frames[f] - c) @ Rm.T + c + rng.uniform(-shift_nm, shift_nm, size=3)
    return out


def test_ligand_pool_rmsf_ignores_whole_ligand_drift():
    """09-23 复现批：配体在口袋里整体晃，原始坐标 RMSF 把芳香环全判成柔性。"""
    import mdtraj as md
    traj, top, aromatic = _mol_traj(tail_jitter=0.003)
    moved = md.Trajectory(_rigid_motion(traj.xyz.astype(float)).astype(np.float32), top)
    pool, note = abfe_core.openfe_ligand_anchor_pool(top, moved, "LIG", rmsf_cutoff_nm=0.1)
    assert set(aromatic) <= set(pool), note


def _host_guest_traj(n=40, ligand_height_nm=0.9, seed=3):
    """一段锯齿状多肽骨架（N-CA-C…，真实键）+ 正上方的苯环配体。"""
    import mdtraj as md
    from rdkit import Chem
    from rdkit.Chem import AllChem

    top = md.Topology()
    ch = top.add_chain()
    xyz, bb = [], []
    for r in range(8):
        res = top.add_residue("ALA", ch)
        for k, name in enumerate(("N", "CA", "C")):
            i = 3 * r + k
            bb.append(top.add_atom(name, md.element.carbon if name != "N" else md.element.nitrogen, res))
            xyz.append([-1.0 + 0.125 * i, 0.07 * (i % 2), 0.0])
    for a, b in zip(bb[:-1], bb[1:]):
        top.add_bond(a, b)

    mol = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=0xF00D)
    conf = mol.GetConformer()
    lig = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                     conf.GetAtomPosition(i).z] for i in range(mol.GetNumAtoms())]) * 0.1
    lig = lig - lig.mean(axis=0)
    # 环平面转到 xy（压平 z 之后仍是合法的平面环）、抬到骨架上方
    u, s, vt = np.linalg.svd(lig[[a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"]])
    lig = lig @ vt.T
    lig[:, 2] += ligand_height_nm
    res = top.add_residue("LIG", top.add_chain())
    elem = {"C": md.element.carbon, "H": md.element.hydrogen}
    atoms = [top.add_atom(f"{a.GetSymbol()}{i}", elem[a.GetSymbol()], res)
             for i, a in enumerate(mol.GetAtoms())]
    for b in mol.GetBonds():
        top.add_bond(atoms[b.GetBeginAtomIdx()], atoms[b.GetEndAtomIdx()])
    xyz = np.vstack([np.array(xyz), lig])
    rng = np.random.default_rng(seed)
    frames = np.repeat(xyz[None], n, axis=0) + rng.normal(scale=0.004, size=(n,) + xyz.shape)
    return md.Trajectory(frames.astype(np.float32), top), top


def test_openfe_mode_picks_host_atoms_in_the_0p5_to_1p5_nm_shell():
    traj, top = _host_guest_traj()
    est = abfe_core.GeometricRestraintEstimator(
        temperature=300.0, allow_geometric_bond_fallback=False,
        force_constant_mode="openfe_fixed", anchor_mode="openfe")
    res = est.estimate_from_trajectory(traj, "LIG")
    H0, H1, H2 = res["receptor_indices"]
    backbone = set(map(int, top.select("protein and backbone")))
    ring = set(map(int, top.select("resname LIG and element C")))
    assert {H0, H1, H2} <= backbone and set(res["ligand_indices"]) <= ring
    # 老路只在 ≤0.5 nm 找；openfe 要 0.5–1.5 nm
    assert 0.5 < res["equilibrium_values"]["r0"] <= 1.5
    # H0–H1–H2 是真实键连起来的链
    bonds = {frozenset((a.index, b.index)) for a, b in top.bonds}
    assert frozenset((H0, H1)) in bonds and frozenset((H1, H2)) in bonds
    # θ 用"不平"判据，不再叠 [45°,135°]；也不能被后面的 openfe 禁选再筛掉
    assert res["diagnostics"]["n_angle_banned_candidates"] == 0
    assert res["diagnostics"]["openfe_selection"]["n_host_triples_valid"] >= 1


def test_openfe_mode_fails_loudly_when_host_is_only_in_contact_range():
    """骨架全在 0.5 nm 以内 ⟹ openfe 壳层里没人，报清楚是哪一步空了。"""
    traj, _ = _host_guest_traj(ligand_height_nm=0.3)
    # 把骨架压短，最远的原子也在 0.5 nm 内
    traj.xyz[:, :24, 0] *= 0.05
    est = abfe_core.GeometricRestraintEstimator(
        temperature=300.0, allow_geometric_bond_fallback=False, anchor_mode="openfe")
    with pytest.raises(RuntimeError, match="配不上合法的 H0-H1-H2"):
        est.estimate_from_trajectory(traj, "LIG")
