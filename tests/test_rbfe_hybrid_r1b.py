"""RBFE R1b：受限 hybrid builder。

设计依据：`docs/design/PLAN_rbfe_interface_and_implementation.md` §5.3 / §8。

计划 §8 的 R1 验收标准（另一半，映射那半在 `test_rbfe_mapping_r1.py`）：

    **端点相互作用及 dummy 处理有验证；有限差分力与解析力一致。**

本文件按这两条 + §5.3 的「核对每个 λ 的有效总电荷」组织。

## 测试体系为什么是合成的

builder 的验收是**数值**验收：λ=0 的 hybrid 必须逐位还原体系 A、λ=1 还原体系 B、
解析力必须等于 −dU/dx。合成小体系的期望值是精确已知的，真实体系反而会把
"builder 对不对"和"参数化对不对"混在一起。真实体系是 R3 的事，而且它还卡在一个
未解决的前提上：一份**可跑的**配体 B（改基团必然要重新给电荷与成键参数）。

体系构造刻意覆盖了每一类力项：core–core 键力常数 A≠B（必须插值）、core 二面角
k 值 A≠B（必须双项缩放）、A-only 与 B-only 同时存在（两侧都有 dummy）、
1-4 exception、以及两个环境粒子。
"""

from __future__ import annotations

import json

import numpy as np
import openmm
import pytest
from openmm import unit

import rbfe_core as rc


# ---------------------------------------------------------------------------
# 合成 A/B 体系
# ---------------------------------------------------------------------------


def make_system(kind: str, *, core_bond_k: float = None, torsion_k: float = None):
    """配体：core 链 C0-C1-C2-C3；A 在 C3 上挂 H4，B 挂 C4+H5。环境两个粒子。"""
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    bond = openmm.HarmonicBondForce()
    angle = openmm.HarmonicAngleForce()
    torsion = openmm.PeriodicTorsionForce()

    ligand = [
        (12.0, -0.10, 0.34, 0.36),
        (12.0, 0.05, 0.34, 0.36),
        (12.0, 0.05, 0.34, 0.36),
        (12.0, -0.20 if kind == "A" else -0.30, 0.34, 0.36),
    ]
    if kind == "A":
        ligand.append((1.0, 0.20, 0.25, 0.06))                      # H4：A-only
    else:
        ligand.append((12.0, 0.10, 0.34, 0.36))                     # C4：B-only
        ligand.append((1.0, 0.20, 0.25, 0.06))                      # H5：B-only
    n_ligand = len(ligand)

    for mass, charge, sigma, epsilon in ligand:
        system.addParticle(mass * unit.dalton)
        nb.addParticle(
            charge * unit.elementary_charge,
            sigma * unit.nanometer,
            epsilon * unit.kilojoule_per_mole,
        )
    for _ in range(2):
        system.addParticle(16.0 * unit.dalton)
        nb.addParticle(
            -0.5 * unit.elementary_charge,
            0.32 * unit.nanometer,
            0.6 * unit.kilojoule_per_mole,
        )

    k23 = core_bond_k if core_bond_k is not None else (300000.0 if kind == "A" else 320000.0)
    for a, b in ((0, 1), (1, 2)):
        bond.addBond(a, b, 0.152 * unit.nanometer,
                     300000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    bond.addBond(2, 3, 0.152 * unit.nanometer,
                 k23 * unit.kilojoule_per_mole / unit.nanometer**2)
    bond.addBond(3, 4, 0.109 * unit.nanometer,
                 300000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    if kind == "B":
        bond.addBond(4, 5, 0.109 * unit.nanometer,
                     300000.0 * unit.kilojoule_per_mole / unit.nanometer**2)

    for a, b, c, k in ((0, 1, 2, 400.0), (1, 2, 3, 400.0), (2, 3, 4, 300.0)):
        angle.addAngle(a, b, c, 1.911 * unit.radian,
                       k * unit.kilojoule_per_mole / unit.radian**2)

    tk = torsion_k if torsion_k is not None else (2.0 if kind == "A" else 3.5)
    torsion.addTorsion(0, 1, 2, 3, 3, 0.0 * unit.radian, tk * unit.kilojoule_per_mole)

    exclusions = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4)]
    if kind == "B":
        exclusions += [(4, 5), (3, 5)]
    for a, b in exclusions:
        nb.addException(a, b, 0.0 * unit.elementary_charge**2,
                        0.3 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    q0 = nb.getParticleParameters(0)[0]
    q3 = nb.getParticleParameters(3)[0]
    nb.addException(0, 3, 0.8333 * q0 * q3, 0.3 * unit.nanometer,
                    0.15 * unit.kilojoule_per_mole)

    for force in (nb, bond, angle, torsion):
        system.addForce(force)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(3, 0, 0) * unit.nanometer,
        openmm.Vec3(0, 3, 0) * unit.nanometer,
        openmm.Vec3(0, 0, 3) * unit.nanometer,
    )
    return system, list(range(n_ligand))


def make_mapping(n_a: int = 5, n_b: int = 6) -> rc.AtomMapping:
    return rc.AtomMapping(
        protocol_version=rc.RBFE_MAPPING_PROTOCOL_VERSION,
        n_atoms_a=n_a, n_atoms_b=n_b,
        core_pairs=tuple((i, i) for i in range(4)),
        a_only=tuple(range(4, n_a)), b_only=tuple(range(4, n_b)),
        fragment_pairs=(), method="synthetic_fixture",
        symmetry_solution_counts=(), ambiguities=(),
        atom_identity_a=(), atom_identity_b=(),
    )


#: 非退化几何。**不要改成共线**——0-1-2-3 共线时二面角没有定义，
#: `CustomTorsionForce` 的解析导数给 NaN，这条测试踩过一次。
POSITIONS = np.array([
    [1.00, 1.00, 1.00], [1.15, 1.06, 1.02], [1.30, 0.99, 1.05], [1.45, 1.07, 1.01],
    [1.55, 1.13, 1.08],
    [2.20, 1.50, 1.00], [2.40, 1.70, 1.20],
    [1.57, 1.15, 1.06], [1.68, 1.22, 1.12],
])


@pytest.fixture
def pair():
    system_a, ligand_a = make_system("A")
    system_b, ligand_b = make_system("B")
    return system_a, ligand_a, system_b, ligand_b, make_mapping()


@pytest.fixture
def bundle(pair):
    system_a, ligand_a, system_b, ligand_b, mapping = pair
    return rc.build_hybrid_system(
        system_a, ligand_a, system_b, ligand_b, mapping, rc.HybridLambdaSchedule.linear(5)
    )


# ---------------------------------------------------------------------------
# 布局与前置条件
# ---------------------------------------------------------------------------


def test_layout_partitions_all_particles(bundle):
    layout = bundle.layout
    assert layout.n_particles == 9                       # 4 core + 1 A-only + 2 B-only + 2 env
    assert len(layout.core) == 4
    assert len(layout.a_only) == 1
    assert len(layout.b_only) == 2
    assert len(layout.environment) == 2
    everything = set(layout.core) | set(layout.a_only) | set(layout.b_only) | set(layout.environment)
    assert everything == set(range(layout.n_particles))


def test_environment_and_A_keep_their_original_indices(pair, bundle):
    """环境与配体 A 的原子编号不变，B-only 追加在末尾——A 的坐标可以直接复用。"""
    system_a = pair[0]
    for i in range(system_a.getNumParticles()):
        assert bundle.system.getParticleMass(i) == system_a.getParticleMass(i)
    assert all(h >= system_a.getNumParticles() for h in bundle.layout.b_only)


def test_hybrid_system_has_all_three_lambda_parameters(bundle):
    names = set()
    for force in bundle.system.getForces():
        for i in range(getattr(force, "getNumGlobalParameters", lambda: 0)()):
            names.add(force.getGlobalParameterName(i))
    assert set(rc.RBFE_LAMBDA_NAMES) <= names


def test_environment_particle_count_mismatch_is_rejected(pair):
    system_a, ligand_a, system_b, ligand_b, mapping = pair
    system_b.addParticle(1.0 * unit.dalton)
    for force in system_b.getForces():
        if isinstance(force, openmm.NonbondedForce):
            force.addParticle(0.0, 0.3, 0.0)
    with pytest.raises(rc.RBFEHybridBuildError, match="环境原子数不同"):
        rc.build_hybrid_layout(system_a, ligand_a, system_b, ligand_b, mapping)


def test_environment_parameter_mismatch_is_rejected(pair):
    system_a, ligand_a, system_b, ligand_b, mapping = pair
    nb = [f for f in system_b.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    nb.setParticleParameters(6, -0.4 * unit.elementary_charge,
                             0.32 * unit.nanometer, 0.6 * unit.kilojoule_per_mole)
    with pytest.raises(rc.RBFEHybridBuildError, match="非键参数不同"):
        rc.build_hybrid_layout(system_a, ligand_a, system_b, ligand_b, mapping)


def test_environment_bonded_mismatch_is_rejected(pair):
    system_a, ligand_a, system_b, ligand_b, mapping = pair
    bond = [f for f in system_b.getForces() if isinstance(f, openmm.HarmonicBondForce)][0]
    bond.addBond(6, 7, 0.15 * unit.nanometer,
                 1000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    with pytest.raises(rc.RBFEHybridBuildError, match="环境成键项不同"):
        rc.build_hybrid_layout(system_a, ligand_a, system_b, ligand_b, mapping)


def test_core_mass_mismatch_is_rejected(pair):
    """跨 λ 粒子质量必须相同（计划 §5.3）。同位素/HMR 首版不处理。"""
    system_a, ligand_a, system_b, ligand_b, mapping = pair
    system_b.setParticleMass(2, 13.0 * unit.dalton)
    with pytest.raises(rc.RBFEHybridBuildError, match="质量不同"):
        rc.build_hybrid_layout(system_a, ligand_a, system_b, ligand_b, mapping)


def test_unsupported_force_is_rejected(pair):
    system_a, ligand_a, system_b, ligand_b, mapping = pair
    system_a.addForce(openmm.CustomExternalForce("x^2"))
    with pytest.raises(rc.RBFEHybridBuildError, match="未验证的力"):
        rc.build_hybrid_layout(system_a, ligand_a, system_b, ligand_b, mapping)


def test_ligand_indices_length_must_match_mapping(pair):
    system_a, ligand_a, system_b, ligand_b, mapping = pair
    with pytest.raises(rc.RBFEHybridBuildError, match="映射里 A 有"):
        rc.build_hybrid_layout(system_a, ligand_a[:-1], system_b, ligand_b, mapping)


def test_core_bond_topology_mismatch_is_rejected(pair):
    """core 内部成键/断键首版拒绝——`validate_mapping` 的连通性判据抓不到这个。"""
    system_a, ligand_a, system_b, ligand_b, mapping = pair
    bond = [f for f in system_b.getForces() if isinstance(f, openmm.HarmonicBondForce)][0]
    bond.addBond(0, 3, 0.25 * unit.nanometer,
                 100000.0 * unit.kilojoule_per_mole / unit.nanometer**2)
    with pytest.raises(rc.RBFEHybridBuildError, match="核心内部的键连接在 A、B 里不同"):
        rc.build_hybrid_system(system_a, ligand_a, system_b, ligand_b, mapping,
                               rc.HybridLambdaSchedule.linear(3))


# ---------------------------------------------------------------------------
# 成键层的分类
# ---------------------------------------------------------------------------


def test_differing_core_bond_is_interpolated(bundle):
    stats = bundle.provenance["bonded"]
    assert stats["interp_bond"] == 1, "C2-C3 的力常数 A≠B，必须走插值"
    assert stats["plain_bond"] == 2


def test_dummy_bonded_terms_go_to_their_own_force_groups(bundle):
    stats = bundle.provenance["bonded"]
    assert stats["dummy_bond_A"] == 1                     # C3-H4
    assert stats["dummy_bond_B"] == 2                     # C3-C4, C4-H5
    groups = {}
    for force in bundle.system.getForces():
        groups.setdefault(force.getForceGroup(), []).append(type(force).__name__)
    assert rc.RBFE_FORCE_GROUP_DUMMY_A in groups
    assert rc.RBFE_FORCE_GROUP_DUMMY_B in groups


def test_differing_core_torsion_uses_dual_scaling(bundle):
    stats = bundle.provenance["bonded"]
    assert stats["torsion_A_scaled"] == 1 and stats["torsion_B_scaled"] == 1
    assert stats["plain_torsion"] == 0


def test_identical_core_torsion_stays_on_the_native_force(pair):
    """A、B 逐位相同的二面角走 native `PeriodicTorsionForce`。

    不是为了省事：native 力对退化几何（四原子共线）是安全的，
    `CustomTorsionForce` 在那里给 NaN。能用 native 就用 native。
    """
    system_a, ligand_a = make_system("A", torsion_k=2.0)
    system_b, ligand_b = make_system("B", torsion_k=2.0)
    b = rc.build_hybrid_system(system_a, ligand_a, system_b, ligand_b, make_mapping(),
                               rc.HybridLambdaSchedule.linear(3))
    stats = b.provenance["bonded"]
    assert stats["plain_torsion"] == 1
    assert stats["torsion_A_scaled"] == 0 and stats["torsion_B_scaled"] == 0


def test_identical_core_bond_is_not_interpolated(pair):
    system_a, ligand_a = make_system("A", core_bond_k=300000.0)
    system_b, ligand_b = make_system("B", core_bond_k=300000.0)
    b = rc.build_hybrid_system(system_a, ligand_a, system_b, ligand_b, make_mapping(),
                               rc.HybridLambdaSchedule.linear(3))
    assert b.provenance["bonded"]["interp_bond"] == 0
    assert b.provenance["bonded"]["plain_bond"] == 3


# ---------------------------------------------------------------------------
# 非键层
# ---------------------------------------------------------------------------


def _nonbonded(system):
    return [f for f in system.getForces() if type(f) is openmm.NonbondedForce][0]


def test_charge_offsets_are_only_on_alchemical_atoms(bundle):
    nb = _nonbonded(bundle.system)
    offset_atoms = set()
    for i in range(nb.getNumParticleParameterOffsets()):
        name, index, *_ = nb.getParticleParameterOffset(i)
        assert name == rc.LAMBDA_CHARGE
        offset_atoms.add(int(index))
    assert offset_atoms == set(bundle.layout.alchemical)
    assert not (offset_atoms & set(bundle.layout.environment))


def test_charge_offset_interpolates_to_B_not_to_zero(pair, bundle):
    """从 ABFE 抄 offset 机制时最容易错的一处：系数必须是 qᴮ−qᴬ，不是 −qᴬ。

    ABFE 那边写的是 `base=0, scale=q`（插到零，那是 ABFE 的端点）。
    照抄会让 λ=1 的配体没有电荷，而 RBFE 的 λ=1 应该是完整的 B。
    """
    system_a, _, system_b, _, _ = pair
    nb_hybrid = _nonbonded(bundle.system)
    nb_a, nb_b = _nonbonded(system_a), _nonbonded(system_b)
    offsets = {}
    for i in range(nb_hybrid.getNumParticleParameterOffsets()):
        _name, index, charge_scale, *_ = nb_hybrid.getParticleParameterOffset(i)
        # getParticleParameterOffset 返回裸 float（不是 Quantity）
        offsets[int(index)] = float(charge_scale)

    core_atom = 3                                        # 电荷 A=-0.20、B=-0.30
    q_a = nb_a.getParticleParameters(core_atom)[0].value_in_unit(unit.elementary_charge)
    q_b = nb_b.getParticleParameters(core_atom)[0].value_in_unit(unit.elementary_charge)
    assert offsets[core_atom] == pytest.approx(q_b - q_a)
    assert offsets[core_atom] != pytest.approx(-q_a), "这就是照抄 ABFE 系数会得到的错值"


def test_A_only_and_B_only_can_never_interact(bundle):
    """hybrid 拓扑的硬约束：两组 dummy 永远不能相互看见。ABFE 里没有这个概念。"""
    nb = _nonbonded(bundle.system)
    excluded = set()
    for i in range(nb.getNumExceptions()):
        p1, p2, *_ = nb.getExceptionParameters(i)
        excluded.add((min(int(p1), int(p2)), max(int(p1), int(p2))))
    for a in bundle.layout.a_only:
        for b in bundle.layout.b_only:
            assert (min(a, b), max(a, b)) in excluded
    assert bundle.provenance["nonbonded"]["n_forbidden_dummy_pairs"] == 2


def test_alchemical_lj_is_off_the_native_force(bundle):
    """炼金原子的 LJ 全部搬到三个 custom 力上，native 力上 epsilon 清零。"""
    nb = _nonbonded(bundle.system)
    for index in bundle.layout.alchemical:
        _q, _sigma, epsilon = nb.getParticleParameters(index)
        assert epsilon.value_in_unit(unit.kilojoule_per_mole) == 0.0
    for index in bundle.layout.environment:
        _q, _sigma, epsilon = nb.getParticleParameters(index)
        assert epsilon.value_in_unit(unit.kilojoule_per_mole) > 0.0


def test_custom_lj_forces_have_long_range_correction_disabled(bundle):
    """softcore/插值参数下 OpenMM 的解析尾项公式不成立，一律关掉并如实记账。"""
    customs = [f for f in bundle.system.getForces()
               if isinstance(f, openmm.CustomNonbondedForce)]
    assert len(customs) == 3
    assert all(not f.getUseLongRangeCorrection() for f in customs)
    assert bundle.provenance["nonbonded"]["alchemical_lj_lrc_included"] is False


# ---------------------------------------------------------------------------
# R1 验收 ①：端点相互作用恢复 + dummy 可分离
# ---------------------------------------------------------------------------


def test_endpoints_recover_the_physical_systems(pair, bundle):
    """λ=0 还原 A、λ=1 还原 B，扣掉对侧 dummy 力组之后**精确**相等。"""
    system_a, ligand_a, system_b, ligand_b, _ = pair
    report = rc.verify_hybrid_endpoints(bundle, system_a, ligand_a, system_b, ligand_b, POSITIONS)
    assert report["passed"], json.dumps(report, indent=2, ensure_ascii=False)
    for label, row in report["endpoints"].items():
        assert row["energy_deviation_kJ_per_mol"] < 1e-6, label
        assert row["max_force_deviation_kJ_per_mol_nm"] < 1e-6, label


def test_dummy_contribution_is_separable_and_nonzero(pair, bundle):
    """dummy 的成键项确实被隔离在自己的力组里，而且真的非零——

    如果它是零，上面那条端点测试就变成了一句空话。
    """
    system_a, ligand_a, system_b, ligand_b, _ = pair
    report = rc.verify_hybrid_endpoints(bundle, system_a, ligand_a, system_b, ligand_b, POSITIONS)
    for row in report["endpoints"].values():
        assert abs(row["separable_dummy_kJ_per_mol"]) > 1.0


def test_lj_lrc_gap_is_quantified_not_hand_waved(pair, bundle):
    """已知缺口要给出**数字**，不是一句 caveat。"""
    system_a, ligand_a, system_b, ligand_b, _ = pair
    report = rc.verify_hybrid_endpoints(bundle, system_a, ligand_a, system_b, ligand_b, POSITIONS)
    for row in report["endpoints"].values():
        assert "lj_lrc_gap_kJ_per_mol" in row
        assert np.isfinite(row["lj_lrc_gap_kJ_per_mol"])


def test_degenerate_geometry_raises_instead_of_silently_passing(pair, bundle):
    """踩过的坑：`max(0.0, nan)` 在 Python 里返回 0.0，NaN 力偏差会被当成零通过。

    现在非有限值在 `_energy_and_forces` 里就炸掉。**别把这条改成 xfail**。
    """
    system_a, ligand_a, system_b, ligand_b, _ = pair
    collinear = POSITIONS.copy()
    collinear[0:4] = [[1.00, 1.00, 1.00], [1.15, 1.00, 1.00],
                      [1.30, 1.00, 1.00], [1.45, 1.00, 1.00]]
    with pytest.raises(rc.RBFEHybridBuildError, match="非有限值"):
        rc.verify_hybrid_endpoints(bundle, system_a, ligand_a, system_b, ligand_b, collinear)


def test_wrong_position_count_is_rejected(pair, bundle):
    system_a, ligand_a, system_b, ligand_b, _ = pair
    with pytest.raises(rc.RBFEHybridBuildError, match="positions_hybrid"):
        rc.verify_hybrid_endpoints(bundle, system_a, ligand_a, system_b, ligand_b, POSITIONS[:-1])


# ---------------------------------------------------------------------------
# R1 验收 ②：有限差分力与解析力一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", [0, 1, 2, 3, 4])
def test_analytic_forces_match_finite_difference(bundle, state):
    report = rc.verify_hybrid_forces_finite_difference(bundle, POSITIONS, state, n_samples=12)
    assert report["passed"], json.dumps(report, indent=2, ensure_ascii=False)


def test_finite_difference_sampling_is_deterministic(bundle):
    """抽样必须可复现，否则"通过"没有意义。"""
    first = rc.verify_hybrid_forces_finite_difference(bundle, POSITIONS, 2, n_samples=8)
    second = rc.verify_hybrid_forces_finite_difference(bundle, POSITIONS, 2, n_samples=8)
    assert [(r["atom"], r["axis"]) for r in first["samples"]] == \
           [(r["atom"], r["axis"]) for r in second["samples"]]


def test_finite_difference_covers_the_softcore_forces(bundle):
    """抽到的点里必须包含 dummy 原子——softcore 分母是手写的，不测等于没测。"""
    report = rc.verify_hybrid_forces_finite_difference(bundle, POSITIONS, 2, n_samples=40)
    touched = {r["atom"] for r in report["samples"]}
    assert touched & (set(bundle.layout.a_only) | set(bundle.layout.b_only))
    assert report["passed"]


# ---------------------------------------------------------------------------
# R1 验收 ③：逐 λ 的有效总电荷
# ---------------------------------------------------------------------------


def test_charge_ledger_reads_the_built_system(bundle):
    """对账读的是**建好的** NonbondedForce，不是把 builder 的意图重算一遍。"""
    ledger = rc.hybrid_charge_ledger(bundle)
    assert ledger["endpoints_match"]
    assert ledger["constant_across_lambda"]
    assert len(ledger["per_state"]) == bundle.schedule.n_states


def test_charge_stays_constant_even_on_a_staged_schedule(pair):
    """计划 §5.3 警告的是「分段切换导致中间态电荷变化」。

    本 builder 用**一条** `lambda_rbfe_charge` 统一驱动所有炼金原子的电荷，
    所以中间态总电荷结构上恒等于端点值——不是碰巧，是设计。这条测试守住这个性质。
    """
    system_a, ligand_a, system_b, ligand_b, mapping = pair
    b = rc.build_hybrid_system(system_a, ligand_a, system_b, ligand_b, mapping,
                               rc.HybridLambdaSchedule.charge_then_sterics(3, 4))
    ledger = rc.hybrid_charge_ledger(b)
    assert ledger["max_intermediate_excursion_e"] < 1e-12


def test_charge_ledger_rejects_offsets_on_environment_atoms(bundle):
    nb = _nonbonded(bundle.system)
    environment_atom = bundle.layout.environment[0]
    nb.addParticleParameterOffset(rc.LAMBDA_CHARGE, environment_atom,
                                  0.1 * unit.elementary_charge,
                                  0.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole)
    with pytest.raises(rc.RBFEHybridBuildError, match="环境电荷不该随 λ 变"):
        rc.hybrid_charge_ledger(bundle)


# ---------------------------------------------------------------------------
# λ 表与产物身份
# ---------------------------------------------------------------------------


def test_schedule_endpoints_must_be_zero_and_one():
    with pytest.raises(rc.RBFEHybridBuildError, match="端点必须是 0 和 1"):
        rc.HybridLambdaSchedule(name="bad", charge=(0.0, 0.5), sterics=(0.0, 1.0), bonded=(0.0, 1.0))


def test_schedule_must_be_monotonic():
    with pytest.raises(rc.RBFEHybridBuildError, match="单调不减"):
        rc.HybridLambdaSchedule(name="bad", charge=(0.0, 0.7, 0.3, 1.0),
                                sterics=(0.0, 0.3, 0.7, 1.0), bonded=(0.0, 0.3, 0.7, 1.0))


def test_schedule_lengths_must_agree():
    with pytest.raises(rc.RBFEHybridBuildError, match="态数不一致"):
        rc.HybridLambdaSchedule(name="bad", charge=(0.0, 1.0),
                                sterics=(0.0, 0.5, 1.0), bonded=(0.0, 1.0))


def test_linear_schedule_is_uniform():
    schedule = rc.HybridLambdaSchedule.linear(5)
    assert schedule.charge == schedule.sterics == schedule.bonded
    assert schedule.state(0) == {name: 0.0 for name in rc.RBFE_LAMBDA_NAMES}
    assert schedule.state(4) == {name: 1.0 for name in rc.RBFE_LAMBDA_NAMES}


def test_staged_schedule_moves_charge_first():
    schedule = rc.HybridLambdaSchedule.charge_then_sterics(3, 4)
    assert schedule.state(2)[rc.LAMBDA_CHARGE] == 1.0
    assert schedule.state(2)[rc.LAMBDA_STERICS] == 0.0


def test_bundle_fingerprint_is_stable_and_schedule_sensitive(pair):
    system_a, ligand_a, system_b, ligand_b, mapping = pair
    first = rc.build_hybrid_system(system_a, ligand_a, system_b, ligand_b, mapping,
                                   rc.HybridLambdaSchedule.linear(5))
    second = rc.build_hybrid_system(system_a, ligand_a, system_b, ligand_b, mapping,
                                    rc.HybridLambdaSchedule.linear(5))
    third = rc.build_hybrid_system(system_a, ligand_a, system_b, ligand_b, mapping,
                                   rc.HybridLambdaSchedule.linear(7))
    assert first.fingerprint() == second.fingerprint()
    assert first.fingerprint() != third.fingerprint()


def test_bundle_records_the_mapping_it_was_built_from(bundle, pair):
    """换映射 ⇒ 换指纹 ⇒ 旧产物不可复用（计划 §7）。"""
    assert bundle.mapping_fingerprint == pair[4].fingerprint()
    assert bundle.mapping_fingerprint in json.dumps(bundle.to_dict())


def test_bundle_provenance_lists_known_gaps(bundle):
    """已知缺口写进产物，不藏在注释里。"""
    gaps = bundle.provenance["known_gaps"]
    assert any("LRC" in gap for gap in gaps)
    json.dumps(bundle.to_dict())


def test_apply_lambda_state_writes_all_three_parameters(bundle):
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(bundle.system, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(POSITIONS * unit.nanometer)
    written = bundle.apply_lambda_state(context, 3)
    for name, value in written.items():
        assert context.getParameter(name) == pytest.approx(value)
    del context, integrator


# ---------------------------------------------------------------------------
# 回归：空 interaction group 的 CustomNonbondedForce 会算全体粒子对
# ---------------------------------------------------------------------------


def _identity_mapping(n_atoms: int = 5) -> rc.AtomMapping:
    return rc.AtomMapping(
        protocol_version=rc.RBFE_MAPPING_PROTOCOL_VERSION,
        n_atoms_a=n_atoms, n_atoms_b=n_atoms,
        core_pairs=tuple((i, i) for i in range(n_atoms)),
        a_only=(), b_only=(),
        fragment_pairs=(), method="identity",
        symmetry_solution_counts=(), ambiguities=(),
        atom_identity_a=(), atom_identity_b=(),
    )


def _charge_only_mutant():
    """只改电荷、不增删原子的 B——realistic 且**没有任何 dummy 原子**。"""
    system, ligand = make_system("A")
    nb = _nonbonded(system)
    charge, sigma, epsilon = nb.getParticleParameters(3)
    nb.setParticleParameters(3, (charge.value_in_unit(unit.elementary_charge) - 0.12)
                             * unit.elementary_charge, sigma, epsilon)
    return system, ligand


def test_lj_force_is_omitted_when_it_would_have_no_interaction_group():
    """**一个 interaction group 都没有的 `CustomNonbondedForce` 会计算全体粒子对。**

    这是 OpenMM 的默认行为，不是"什么都不算"。所以没有 dummy 原子的边
    （A→A 自边、只改参数的突变）曾经会多挂两个退化成"全体粒子对"的力，
    与 core 力和 native 力重复计数。

    这个 bug 一度被正确答案掩盖：A→A 的 ΔG 照样算出 0，因为 λ 表对称、两个端点上
    softcore lift 都是 0，两个力恰好互为镜像而抵消——**端点对了，中间态全错**。
    别把这条测试改成只验端点。
    """
    system_a, ligand_a = make_system("A")
    system_b, ligand_b = _charge_only_mutant()
    bundle = rc.build_hybrid_system(
        system_a, ligand_a, system_b, ligand_b, _identity_mapping(),
        rc.HybridLambdaSchedule.linear(4),
    )
    assert bundle.layout.a_only == () and bundle.layout.b_only == ()

    customs = [f for f in bundle.system.getForces()
               if isinstance(f, openmm.CustomNonbondedForce)]
    assert len(customs) == 1, "没有 dummy 就只该有 core 那一个 LJ 力"
    assert customs[0].getNumInteractionGroups() > 0
    assert bundle.provenance["nonbonded"]["lj_forces_added"] == ["core"]


def test_every_custom_lj_force_has_at_least_one_interaction_group(bundle):
    """同一条不变量，在有 dummy 的常规边上也守一遍。"""
    for force in bundle.system.getForces():
        if isinstance(force, openmm.CustomNonbondedForce):
            assert force.getNumInteractionGroups() > 0


def test_charge_only_mutation_recovers_both_endpoints():
    """没有 dummy 的边照样要通过端点等价——这条以前会因为重复计数而失败。"""
    system_a, ligand_a = make_system("A")
    system_b, ligand_b = _charge_only_mutant()
    bundle = rc.build_hybrid_system(
        system_a, ligand_a, system_b, ligand_b, _identity_mapping(),
        rc.HybridLambdaSchedule.linear(4),
    )
    report = rc.verify_hybrid_endpoints(
        bundle, system_a, ligand_a, system_b, ligand_b, POSITIONS[:bundle.layout.n_particles]
    )
    assert report["passed"], json.dumps(report, indent=2, ensure_ascii=False)


def test_charge_only_mutation_forces_match_finite_difference():
    system_a, ligand_a = make_system("A")
    system_b, ligand_b = _charge_only_mutant()
    bundle = rc.build_hybrid_system(
        system_a, ligand_a, system_b, ligand_b, _identity_mapping(),
        rc.HybridLambdaSchedule.linear(4),
    )
    positions = POSITIONS[:bundle.layout.n_particles]
    for state in range(4):
        report = rc.verify_hybrid_forces_finite_difference(
            bundle, positions, state, n_samples=8
        )
        assert report["passed"], json.dumps(report, indent=2, ensure_ascii=False)
