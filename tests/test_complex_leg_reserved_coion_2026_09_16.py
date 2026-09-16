"""复合物腿的 reserved co-ion 构建（2026-09-16）+ 1-4 缩放诊断的口径修正。

真机症状（thrombin_ligand1，配体净电荷 +1 e，三个 repeat 同一处退出）：

    reserved co-ion builder identity 不满足 charge-transfer 数量契约：
    leg system 中找到 0 个中性 ion-shaped dummy，但配体净电荷 +1 e 需要 1 个。

根因不是校验太严，是**复合物腿根本没有插入步骤**：溶剂腿的盒子由本流程
`addSolvent()` 建，所以能在 `Modeller` 上"摘一个水、加一个 dummy"；复合物腿的体系
来自用户的 `.gro/.top`，而 `GromacsTopFile` 的粒子表由 `[ molecules ]` 决定 ——
只改 OpenMM Topology 不会让 System 多出一个粒子。

同一份日志里的 `fudgeQQ=0.1590 / fudgeLJ=0.2402` 看着像力场参数错了，其实是诊断
把 1-2/1-3 **完全排除**对也算进了平均。真机 thrombin 的真实读数是 0.8333 / 0.5000。
"""
import json
import os
import pathlib
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

import runabfe as R  # noqa: E402
from abfe_core import (  # noqa: E402
    CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
    COION_COION_MIN_IMAGE_INITIAL_NM,
    co_alchemical_ion_builder_identity_payload,
    minimum_image_displacement_nm,
)
from ibs_engine import (  # noqa: E402
    _compute_ligand_net_charge,
    _identify_reserved_neutral_co_ions,
)

THROMBIN = pathlib.Path(
    "/home/ruigengji/abfe-benchmark/parameters/thrombin/ligand1"
)


# --------------------------------------------------------------------- 选点规则
#   两条腿共用这一份实现（先前只有溶剂腿有）


def _box(L=4.0):
    return np.diag([L, L, L]).astype(float)


def test_the_farthest_water_wins():
    """离配体质心 minimum-image 最远的那个水被选中。"""
    pos = np.array([[2.0, 2.0, 2.0],      # 0 配体
                    [2.2, 2.0, 2.0],      # 1 近水
                    [0.2, 0.2, 0.2]])     # 2 远水（对角）
    got = R._select_reserved_coion_water_sites(
        positions_nm=pos, box_nm=_box(), ligand_atom_indices=[0],
        water_oxygen_indices=[1, 2], count=1)
    assert got == [2]


def test_minimum_image_is_actually_used():
    """穿过周期边界的"远"水其实很近 —— 用最小影像就不会选它。"""
    pos = np.array([[0.1, 0.1, 0.1],      # 0 配体，贴着原点
                    [3.9, 3.9, 3.9],      # 1 看着很远，MIC 下只有 0.35 nm
                    [2.0, 2.0, 2.0]])     # 2 真正的对角，MIC 3.29 nm
    got = R._select_reserved_coion_water_sites(
        positions_nm=pos, box_nm=_box(), ligand_atom_indices=[0],
        water_oxygen_indices=[1, 2], count=1)
    assert got == [2]


def test_two_dummies_keep_their_distance_from_each_other():
    """|q_L|≥2：候选之间也要留够安全边距，否则同号 co-ion 在 λ→0 会贴脸。

    （2026-08-06 Ca²⁺ 实测：两个 dummy 相距 0.18~0.43 nm ⟹ charging MBAR 不收敛。）
    """
    L = 8.0
    # ⚠️ 别拿"两个对角"当分得开的候选：(0.2,0.2,0.2) 与 (7.8,7.8,7.8) 在周期盒里
    # 只隔 0.69 nm（第一版 fixture 就是这么写错的）。这里第三个点是真的分得开。
    pos = np.array([[4.0, 4.0, 4.0],      # 0 配体，盒心
                    [0.2, 0.2, 0.2],      # 1 远角
                    [0.4, 0.2, 0.2],      # 2 紧挨着 1（只有 0.2 nm）
                    [0.2, 0.2, 3.0]])     # 3 离 1 有 2.8 nm，够远
    got = R._select_reserved_coion_water_sites(
        positions_nm=pos, box_nm=_box(L), ligand_atom_indices=[0],
        water_oxygen_indices=[1, 2, 3], count=2)
    d = np.linalg.norm(
        minimum_image_displacement_nm(pos[got[0]] - pos[got[1]], _box(L)))
    assert d >= COION_COION_MIN_IMAGE_INITIAL_NM, (got, d)


def test_not_enough_room_fails_closed():
    """凑不够彼此够远的候选 ⟹ 报错，不静默退化成"离得近也凑合用"。"""
    pos = np.array([[4.0, 4.0, 4.0], [0.2, 0.2, 0.2], [0.4, 0.2, 0.2]])
    with pytest.raises(RuntimeError, match="minimum-image 距离"):
        R._select_reserved_coion_water_sites(
            positions_nm=pos, box_nm=_box(8.0), ligand_atom_indices=[0],
            water_oxygen_indices=[1, 2], count=2)


def test_too_few_waters_fails_closed():
    pos = np.array([[2.0, 2.0, 2.0], [0.2, 0.2, 0.2]])
    with pytest.raises(RuntimeError, match="不够替换出"):
        R._select_reserved_coion_water_sites(
            positions_nm=pos, box_nm=_box(), ligand_atom_indices=[0],
            water_oxygen_indices=[1], count=2)


# --------------------------------------------------------------------- 1-4 诊断


def _system_with_exceptions(pairs):
    """`pairs` = [(chargeProd_e2, epsilon_kJ), ...]，每条对应一条异常。"""
    system = openmm.System()
    nb = openmm.NonbondedForce()
    for _ in range(2 * len(pairs)):
        system.addParticle(1.0 * unit.dalton)
        nb.addParticle(0.5 * unit.elementary_charge, 0.3 * unit.nanometer,
                       1.0 * unit.kilojoule_per_mole)
    for k, (cp, eps) in enumerate(pairs):
        nb.addException(2 * k, 2 * k + 1,
                        cp * unit.elementary_charge ** 2,
                        0.3 * unit.nanometer,
                        eps * unit.kilojoule_per_mole)
    system.addForce(nb)
    return system


def test_fully_excluded_pairs_no_longer_drag_the_average_down():
    """**要害**：1-2/1-3 完全排除对不参与统计。

    旧实现把它们一起平均，于是真机 thrombin 打出 fudgeQQ=0.1590、fudgeLJ=0.2402
    —— 那不是力场错了，是把结构性的 0 算进了缩放因子。
    """
    # 每个粒子 q=0.5、eps=1.0 ⟹ 未缩放的参照是 q1q2=0.25、sqrt(eps1*eps2)=1.0
    pairs = [(0.0, 0.0)] * 20 + [(0.25 * 0.8333, 0.5)] * 4
    rep = R.diagnose_14_scaling(_system_with_exceptions(pairs))
    assert rep["n_exceptions_total"] == 24
    assert rep["n_fully_excluded"] == 20
    assert rep["n_explicit_pairs"] == 4
    assert rep["fudgeQQ"]["n"] == 4
    assert rep["fudgeQQ"]["median"] == pytest.approx(0.8333, abs=1e-4)
    assert rep["fudgeLJ"]["median"] == pytest.approx(0.5, abs=1e-9)


def test_an_outlier_is_visible_instead_of_averaged_away():
    """只报中位数与极值：有一条不一样时要看得见，不是被平均掉。"""
    pairs = [(0.25 * 0.8333, 0.5)] * 9 + [(0.25 * 0.2, 0.5)]
    rep = R.diagnose_14_scaling(_system_with_exceptions(pairs))
    assert rep["fudgeQQ"]["median"] == pytest.approx(0.8333, abs=1e-4)
    assert rep["fudgeQQ"]["min"] == pytest.approx(0.2, abs=1e-4)


def test_a_box_with_only_exclusions_says_so_instead_of_reporting_zero():
    rep = R.diagnose_14_scaling(_system_with_exceptions([(0.0, 0.0)] * 5))
    assert rep["n_explicit_pairs"] == 0
    assert rep["fudgeQQ"]["median"] is None      # 不是 0.0


# --------------------------------------------------------- 最小 GROMACS 体系派生

MINI_TOP = """[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
 CT     6       12.011   0.0000  A      0.34000  0.45773
 HC     1        1.008   0.0000  A      0.26495  0.06569
 OW     8       15.999   0.0000  A      0.31507  0.63639
 HW     1        1.008   0.0000  A      0.00000  0.00000
 IP    11       22.990   0.0000  A      0.33284  0.01159
 IM    17       35.450   0.0000  A      0.44010  0.41840

[ moleculetype ]
LIG 3

[ atoms ]
     1  CT    1     LIG     C1   1    0.5000  12.011
     2  HC    1     LIG     H1   1    0.5000   1.008

[ bonds ]
    1 2 1 0.10900 284512.0

[ moleculetype ]
HOH 2

[ atoms ]
     1  OW    1     HOH     OW   1   -0.834  15.999
     2  HW    1     HOH     HW1  1    0.417   1.008
     3  HW    1     HOH     HW2  1    0.417   1.008

[ settles ]
    1  1  0.09572  0.15139

[ exclusions ]
    1 2 3
    2 1 3
    3 1 2

[ moleculetype ]
NA 1

[ atoms ]
     1  IP    1     NA      Na   1    1.0000  22.990

[ moleculetype ]
CL 1

[ atoms ]
     1  IM    1     CL      Cl   1   -1.0000  35.450

[ system ]
mini

[ molecules ]
LIG                  1
HOH                  6
NA                   1
CL                   2
"""


def _write_mini_system(tmp_path, box_nm=4.0):
    """一个能被 OpenMM 解析的最小 GROMACS 体系：LIG(+1) + 6 水 + 1 Na + 2 Cl，总电荷 0。"""
    lines = []

    def add(rs, rn, an, ai, x, y, z):
        lines.append("%5d%-5s%5s%5d%8.3f%8.3f%8.3f" % (rs, rn, an, ai, x, y, z))

    n = r = 0
    r += 1
    n += 1
    add(r, "LIG", "C1", n, 2.0, 2.0, 2.0)
    n += 1
    add(r, "LIG", "H1", n, 2.11, 2.0, 2.0)
    for (x, y, z) in [(0.5, 0.5, 0.5), (3.5, 3.5, 3.5), (0.5, 3.5, 0.5),
                      (3.5, 0.5, 3.5), (2.0, 0.6, 2.0), (2.0, 3.4, 2.0)]:
        r += 1
        for name, (dx, dy) in (("OW", (0.0, 0.0)), ("HW1", (0.09, 0.0)),
                               ("HW2", (0.0, 0.09))):
            n += 1
            add(r, "HOH", name, n, x + dx, y + dy, z)
    r += 1
    n += 1
    add(r, "NA", "Na", n, 1.0, 1.0, 3.0)
    for k in range(2):
        r += 1
        n += 1
        add(r, "CL", "Cl", n, 3.0, 1.0 + 0.5 * k, 1.0)
    gro = tmp_path / "mini.gro"
    gro.write_text(
        "mini\n%d\n" % n + "\n".join(lines)
        + "\n%10.5f%10.5f%10.5f\n" % (box_nm, box_nm, box_nm)
    )
    top = tmp_path / "mini.top"
    top.write_text(MINI_TOP)
    return str(gro), str(top)


def _build(gro, top):
    return R.build_system_from_gromacs(gro, top, "LIG", None)


def test_the_contract_really_fails_before_the_fix(tmp_path):
    """先钉住症状本身：没有 dummy 的带电体系进数量契约就是报这句话。"""
    gro, top = _write_mini_system(tmp_path)
    system, topology, _pos, _box_v, lig = _build(gro, top)
    assert int(round(_compute_ligand_net_charge(system, lig))) == 1
    assert R._count_reserved_neutral_coion_candidates(system, topology) == 0
    with pytest.raises(ValueError, match="找到 0 个中性 ion-shaped dummy"):
        co_alchemical_ion_builder_identity_payload(
            system=system, topology=topology,
            charge_treatment=CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
            ligand_net_charge_e=1)


def test_the_complex_leg_now_builds_its_own_reserved_coion(tmp_path):
    """修完之后：复合物腿自己把 dummy 建出来，数量契约与身份识别都过。"""
    gro, top = _write_mini_system(tmp_path)
    system, topology, positions, box_v, lig = _build(gro, top)
    out = tmp_path / "run"
    system, topology, positions, box_v, lig, report = (
        R._ensure_complex_reserved_coions(
            system=system, topology=topology, positions=positions,
            box_vectors=box_v, ligand_indices=lig, gro_file=gro, top_file=top,
            ligand_resname="LIG", gmx_include_dir=None, output_dir=str(out),
            ligand_net_charge_e=1))
    assert report["count"] == 1 and report["species"] == "NA"
    assert report["removed_atom_count"] == 3          # 一个水连 H 一起摘
    assert report["n_atoms_after"] == report["n_atoms_before"] - 3 + 1

    ident = co_alchemical_ion_builder_identity_payload(
        system=system, topology=topology,
        charge_treatment=CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=1)
    assert ident["reserved_coion_count"] == 1
    assert ident["ions"][0]["charge_at_lambda1_e"] == 0.0
    # 保留 LJ（"中性但 ion-shaped"），不是一个什么都没有的鬼影
    assert ident["ions"][0]["sigma_nm"] > 0.0
    assert ident["ions"][0]["epsilon_kj_mol"] > 0.0
    assert ident["ions"][0]["mass_amu"] == pytest.approx(22.99, abs=0.01)

    nb = next(f for f in system.getForces()
              if isinstance(f, openmm.NonbondedForce))
    idx, _meta = _identify_reserved_neutral_co_ions(nb, topology, 1)
    assert idx == [report["reserved_coion_indices"][0]]


def test_the_derivation_never_touches_the_original_inputs(tmp_path):
    gro, top = _write_mini_system(tmp_path)
    before = (pathlib.Path(gro).read_bytes(), pathlib.Path(top).read_bytes())
    system, topology, positions, box_v, lig = _build(gro, top)
    R._ensure_complex_reserved_coions(
        system=system, topology=topology, positions=positions, box_vectors=box_v,
        ligand_indices=lig, gro_file=gro, top_file=top, ligand_resname="LIG",
        gmx_include_dir=None, output_dir=str(tmp_path / "run"),
        ligand_net_charge_e=1)
    assert (pathlib.Path(gro).read_bytes(), pathlib.Path(top).read_bytes()) == before


def test_every_surviving_atom_keeps_its_parameters_and_position(tmp_path):
    """**最贵的那种错**：`.gro` 的原子顺序与 `[ molecules ]` 展开顺序错位。

    错位不会报错，只会把参数张冠李戴。所以逐原子对账：除了被摘掉的水与追加的
    dummy，其余每个粒子的 q/σ/ε/质量/坐标都必须逐位相同。
    """
    gro, top = _write_mini_system(tmp_path)
    s0, t0, p0, _b0, l0 = _build(gro, top)
    out = tmp_path / "run"
    s1, t1, p1, _b1, _l1, report = R._ensure_complex_reserved_coions(
        system=s0, topology=t0, positions=p0, box_vectors=_b0, ligand_indices=l0,
        gro_file=gro, top_file=top, ligand_resname="LIG", gmx_include_dir=None,
        output_dir=str(out), ligand_net_charge_e=1)

    removed = set()
    for res in t0.residues():
        for a in res.atoms():
            if a.index in report["removed_water_oxygen_indices"]:
                removed.update(x.index for x in res.atoms())
    assert len(removed) == 3

    nb0 = next(f for f in s0.getForces() if isinstance(f, openmm.NonbondedForce))
    nb1 = next(f for f in s1.getForces() if isinstance(f, openmm.NonbondedForce))
    a0 = np.asarray(p0.value_in_unit(unit.nanometer))
    a1 = np.asarray(p1.value_in_unit(unit.nanometer))
    new_j = 0
    for old_i in range(s0.getNumParticles()):
        if old_i in removed:
            continue
        for got, want in zip(nb1.getParticleParameters(new_j),
                             nb0.getParticleParameters(old_i)):
            assert got == want, (old_i, new_j)
        assert s1.getParticleMass(new_j) == s0.getParticleMass(old_i)
        assert np.allclose(a1[new_j], a0[old_i], atol=1e-6)
        new_j += 1
    assert new_j + 1 == s1.getNumParticles()


def test_an_input_that_already_reserved_its_dummy_is_left_alone(tmp_path):
    """手工预留过 dummy 的输入：一个字节都不动（`report is None`）。"""
    gro, top = _write_mini_system(tmp_path)
    system, topology, positions, box_v, lig = _build(gro, top)
    # 把现成的 Na⁺ 清零，当作"建系时预留的中性 dummy"
    nb = next(f for f in system.getForces()
              if isinstance(f, openmm.NonbondedForce))
    na = next(a.index for a in topology.atoms()
              if str(a.residue.name).upper() == "NA")
    _q, sigma, eps = nb.getParticleParameters(na)
    nb.setParticleParameters(na, 0.0 * unit.elementary_charge, sigma, eps)
    n_before = system.getNumParticles()
    out = R._ensure_complex_reserved_coions(
        system=system, topology=topology, positions=positions, box_vectors=box_v,
        ligand_indices=lig, gro_file=gro, top_file=top, ligand_resname="LIG",
        gmx_include_dir=None, output_dir=str(tmp_path / "run"),
        ligand_net_charge_e=1)
    assert out[-1] is None
    assert out[0] is system and out[0].getNumParticles() == n_before


def test_a_wrong_number_of_existing_dummies_fails_closed(tmp_path):
    """既不是 0 也不是 |q_L| ⟹ 不猜哪个才是 reserved dummy。"""
    gro, top = _write_mini_system(tmp_path)
    system, topology, positions, box_v, lig = _build(gro, top)
    nb = next(f for f in system.getForces()
              if isinstance(f, openmm.NonbondedForce))
    for name in ("NA", "CL"):
        i = next(a.index for a in topology.atoms()
                 if str(a.residue.name).upper() == name)
        _q, sigma, eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, 0.0 * unit.elementary_charge, sigma, eps)
    with pytest.raises(RuntimeError, match="拒绝猜测"):
        R._ensure_complex_reserved_coions(
            system=system, topology=topology, positions=positions,
            box_vectors=box_v, ligand_indices=lig, gro_file=gro, top_file=top,
            ligand_resname="LIG", gmx_include_dir=None,
            output_dir=str(tmp_path / "run"), ligand_net_charge_e=1)


class _EmptyTopology:
    """一个一个离子都没有的"盒子"。"""

    def residues(self):
        return iter(())


def test_no_same_sign_ion_in_the_box_fails_closed():
    """盒里没有同号离子 ⟹ 没有可用的 moleculetype，如实报错而不是伪造一个。

    `.top` 里不存在的分子类型写进 `[ molecules ]`，只会在解析时炸得更晚更难查。
    """
    with pytest.raises(RuntimeError, match="没有符号正确的单价离子"):
        R._pick_reserved_coion_species(
            _system_with_exceptions([(0.0, 0.0)]), _EmptyTopology(), cation=True)


def test_the_species_comes_from_the_box_and_has_the_right_sign(tmp_path):
    """阳离子配体要 Na⁺ 模板、阴离子配体要 Cl⁻ 模板 —— 都得是这个盒里已有的。"""
    gro, top = _write_mini_system(tmp_path)
    system, topology, _p, _b, _l = _build(gro, top)
    assert R._pick_reserved_coion_species(system, topology, cation=True) == "NA"
    assert R._pick_reserved_coion_species(system, topology, cation=False) == "CL"


# ------------------------------------------------------------ 真机输入：thrombin


@pytest.mark.skipif(
    not (THROMBIN / "complex.gro").exists(),
    reason=f"benchmark 输入不在这台机器上：{THROMBIN}",
)
def test_thrombin_ligand1_end_to_end(tmp_path):
    """**就是报这个错的那个真实体系**：thrombin_ligand1，配体净电荷 +1 e。

    这里不跑采样，只把"从 GROMACS 建系 → 派生 co-ion → 数量契约 → 身份识别 →
    几何门"整条链在真实输入上走一遍。
    """
    from abfe_core import openmm_compatible_gromacs_top

    top_for_openmm, _conv = openmm_compatible_gromacs_top(
        str(THROMBIN / "complex.top"), None,
        compat_dir=str(tmp_path / "gromacs_openmm_compat"))
    system, topology, positions, box_v, lig = R.build_system_from_gromacs(
        str(THROMBIN / "complex.gro"), top_for_openmm, "LIG", None)

    q_lig = _compute_ligand_net_charge(system, lig)
    assert q_lig == pytest.approx(1.0, abs=1e-6), "输入变了？这个体系配体是 +1 e"
    assert R._total_system_charge_e(system) == pytest.approx(0.0, abs=1e-6)
    assert R._count_reserved_neutral_coion_candidates(system, topology) == 0

    # ① 力场没错，是诊断口径错了：真实的 1-4 缩放是 0.8333 / 0.5
    rep14 = R.diagnose_14_scaling(system)
    assert rep14["n_fully_excluded"] > rep14["n_explicit_pairs"]  # 排除对占多数
    assert rep14["fudgeQQ"]["median"] == pytest.approx(0.8333, abs=1e-3)
    assert rep14["fudgeLJ"]["median"] == pytest.approx(0.5, abs=1e-3)

    # ② 复合物腿把 dummy 建出来
    n_before = system.getNumParticles()
    system, topology, positions, box_v, lig, report = (
        R._ensure_complex_reserved_coions(
            system=system, topology=topology, positions=positions,
            box_vectors=box_v, ligand_indices=lig,
            gro_file=str(THROMBIN / "complex.gro"), top_file=top_for_openmm,
            ligand_resname="LIG", gmx_include_dir=None,
            output_dir=str(tmp_path / "run"), ligand_net_charge_e=1))
    assert report["count"] == 1 and report["cation"] is True
    assert system.getNumParticles() == n_before - 3 + 1
    assert len(positions) == system.getNumParticles()
    assert R._total_system_charge_e(system) == pytest.approx(0.0, abs=1e-6)
    assert _compute_ligand_net_charge(system, lig) == pytest.approx(1.0, abs=1e-6)

    # ③ 数量契约 + 坐标无关的身份识别
    ident = co_alchemical_ion_builder_identity_payload(
        system=system, topology=topology,
        charge_treatment=CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=1)
    assert ident["reserved_coion_count"] == 1
    assert ident["ions"][0]["charge_at_lambda1_e"] == 0.0
    nb = next(f for f in system.getForces()
              if isinstance(f, openmm.NonbondedForce))
    idx, _meta = _identify_reserved_neutral_co_ions(nb, topology, 1)
    assert idx == report["reserved_coion_indices"]

    # ④ 几何：dummy 离配体够远（生产路线的 strict 门吃的就是这个量）
    pos = np.asarray(positions.value_in_unit(unit.nanometer))
    box = np.array([v.value_in_unit(unit.nanometer) for v in box_v])
    origin = pos[lig[0]]
    centroid = origin + minimum_image_displacement_nm(
        pos[lig] - origin, box).mean(axis=0)
    d_centroid = float(np.linalg.norm(
        minimum_image_displacement_nm(pos[idx[0]] - centroid, box)))
    d_nearest = min(
        float(np.linalg.norm(minimum_image_displacement_nm(pos[idx[0]] - pos[j], box)))
        for j in lig)
    assert d_nearest >= COION_COION_MIN_IMAGE_INITIAL_NM, d_nearest
    assert d_centroid > 3.0, d_centroid

    # ⑤ 派生产物自带可追溯的报告
    saved = json.loads(
        (tmp_path / "run" / "coion_reserved" / "coion_reserved.json").read_text())
    assert saved["n_atoms_after"] == system.getNumParticles()
    assert pathlib.Path(saved["derived_gro"]).exists()
    assert pathlib.Path(saved["derived_top"]).exists()
