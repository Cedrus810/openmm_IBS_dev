"""Shadow-Coulomb builder 不得重复计入配体分子内的 LJ 与 1-4。

## 缺陷

`build_shadow_coul_ibs_system` 的背景处理与 `build_ibs_dual_system` **不同**：

* dual builder 把配体的 σ/ε 压成 `(0.1nm, 0)`、清掉所有沾配体的 exception，
  再由 Group 2 全额重建 —— 主 NB 不出任何 L–L，不会重复。
* shadow builder 只调 `_zero_ligand_environment_charge_in_background()`：
  它清掉配体**粒子电荷**和**跨组** exception 的 chargeProd，但
  **σ/ε 原样保留、L–L exception 原样保留**（这是有意的：VdW 在 shadow 腿全程
  满强度、属于 U_common）。然后又照抄 dual 的写法调 `create_ligand_internal_force`
  并把 `ll_14_force` 一起加进去。

结果：

| 项 | 主 NB | Group 2 | 合计 |
|---|---|---|---|
| L–L 普通库仑 | 0（粒子电荷已清） | 补回 | ×1 ✓ |
| L–L 普通 LJ | **仍在** | **又加一份** | **×2** ✗ |
| L–L 1-4（LJ 与库仑） | **仍在** | `ll_14_force` **又加一份** | **×2** ✗ |

"它与 λ 无关，所以在 ΔF 里相消"**不成立**：它在同一构型的能量差里确实消失，
但它改变构型的 Boltzmann 权重 ⟹ 改变采样系综；翻倍的 r⁻¹² 芯还是不稳定源。

## 为什么现有测试看不见

`test_shadow_coul_ibs_builder.py` 只查 Group 1 的 CV 形态，且它那个体系里唯一的
L–L exception `(0,1)` 的 chargeProd 与 epsilon **都是零** —— 重复加零还是零。

## 本文件怎么测

**把所有电荷置零**，于是整个体系只剩 LJ，`U_common`（group 0+2）必须**逐位等于**
未改动原体系的总能量与力。任何 L–L LJ / 1-4 LJ 的重复都会让它偏大。
再加一组结构断言覆盖 1-4 库仑那一半（零电荷测不到它）。
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

import ibs_engine as ie  # noqa: E402

N_PARTICLES = 9
LIGAND = [0, 1, 2, 3]          # 四个原子，够构造一条 1-4
ENVIRONMENT = [4, 5, 6, 7, 8]
BOX_NM = 4.0
TEMP = 300.0 * unit.kelvin
LAMBDAS = [1.0, 0.5, 0.0]

# 1-4 的 LJ 必须非零，否则"重复加一份"加的是 0，测了等于没测。
LJ14_SIGMA_NM = 0.32
LJ14_EPS_KJ = 0.55


def _system(*, charges):
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(BOX_NM, 0, 0), openmm.Vec3(0, BOX_NM, 0), openmm.Vec3(0, 0, BOX_NM)
    )
    for _ in range(N_PARTICLES):
        system.addParticle(12.0 * unit.dalton)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    for q in charges:
        nb.addParticle(
            q * unit.elementary_charge,
            0.30 * unit.nanometer,
            0.40 * unit.kilojoule_per_mole,
        )
    # 配体内部：1-2 / 1-3 全排除，1-4 保留缩放后的真实参数。
    nb.addException(0, 1, 0.0, 0.30 * unit.nanometer, 0.0)
    nb.addException(1, 2, 0.0, 0.30 * unit.nanometer, 0.0)
    nb.addException(0, 2, 0.0, 0.30 * unit.nanometer, 0.0)
    nb.addException(1, 3, 0.0, 0.30 * unit.nanometer, 0.0)
    nb.addException(
        0, 3,
        charges[0] * charges[3] * 0.8333 * unit.elementary_charge ** 2,
        LJ14_SIGMA_NM * unit.nanometer,
        LJ14_EPS_KJ * unit.kilojoule_per_mole,
    )
    system.addForce(nb)
    return system


def _positions():
    # 配体紧挨在一起（保证 L–L LJ 显著），环境铺开但仍在 cutoff 内。
    coords = [
        (0.50, 0.50, 0.50),
        (0.66, 0.50, 0.50),
        (0.82, 0.52, 0.50),
        (0.97, 0.50, 0.51),
        (1.45, 0.60, 0.50),
        (1.70, 0.45, 0.55),
        (1.95, 0.62, 0.48),
        (2.20, 0.50, 0.52),
        (2.45, 0.55, 0.50),
    ]
    return [openmm.Vec3(*c) for c in coords] * unit.nanometer


def _energy_and_forces(system, groups=None):
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(_positions())
    kwargs = {"getEnergy": True, "getForces": True}
    if groups is not None:
        kwargs["groups"] = groups
    state = context.getState(**kwargs)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = np.asarray(
        state.getForces(asNumpy=True).value_in_unit(
            unit.kilojoule_per_mole / unit.nanometer
        ),
        dtype=np.float64,
    )
    del context, integrator
    return energy, forces


def _build_shadow(system):
    return ie.build_shadow_coul_ibs_system(
        system=system,
        topology=None,
        perturbed_indices=list(LIGAND),
        lambdas_shadow_coul=list(LAMBDAS),
        restraint_params=None,
        prefix="abfe_shadow",
        box_vectors=system.getDefaultPeriodicBoxVectors(),
        temperature=TEMP,
    )


def test_u_common_equals_original_when_all_charges_are_zero():
    """全体零电荷 ⇒ group 0+2 必须等于原体系总能量与力。

    零电荷下整个体系只剩 LJ，而 shadow 对 LJ **什么都不该改**（它只搬静电）。
    所以 `U_common` 与原体系逐项相同。L–L 普通 LJ 或 1-4 LJ 被算两遍，
    这条立刻红。
    """
    zero_charges = [0.0] * N_PARTICLES
    reference_system = _system(charges=zero_charges)
    ref_energy, ref_forces = _energy_and_forces(reference_system)

    built, _wrapper = _build_shadow(_system(charges=zero_charges))
    # group 0 = 背景 NB（builder 不改它的 group），group 2 = 配体内部力。
    common_energy, common_forces = _energy_and_forces(built, groups={0, 2})

    assert common_energy == pytest.approx(ref_energy, abs=1e-6, rel=1e-9), (
        f"U_common(group 0+2) = {common_energy:.9f} kJ/mol，"
        f"原体系总能量 = {ref_energy:.9f} kJ/mol，差 "
        f"{common_energy - ref_energy:+.9f}。\n"
        "  全体电荷为零时两者只含 LJ，必须相等。偏大 = 配体分子内 LJ 或 1-4 LJ "
        "被算了两遍（背景 NB 保留了配体 σ/ε 和 L–L exception，Group 2 又加了一份）。"
    )
    np.testing.assert_allclose(
        common_forces, ref_forces, atol=1e-6, rtol=1e-9,
        err_msg="U_common 的力与原体系不一致——同上，分子内项被重复计入。",
    )


def test_ligand_internal_group2_carries_no_lj_and_no_14_force():
    """结构断言，覆盖零电荷测不到的那一半：1-4 库仑。

    正确划分是：
      * 背景 NB 保留 L–L 普通 LJ 和**全部** L–L exception（含 1-4 的 chargeProd
        与 epsilon）；
      * Group 2 只补 L–L 普通库仑 —— 因此它的配体 ε 必须为 0，且**不得**再挂一个
        内部 1-4 力。
    """
    charges = [0.4, -0.35, 0.25, -0.30, 0.3, -0.3, 0.2, -0.2, 0.0]
    built, _wrapper = _build_shadow(_system(charges=charges))

    group2 = [f for f in built.getForces() if f.getForceGroup() == 2]
    assert group2, "Group 2 里一个力都没有——配体内部库仑没被补回来。"

    bond_like = [f for f in group2 if isinstance(f, openmm.CustomBondForce)]
    assert not bond_like, (
        f"Group 2 里有 {len(bond_like)} 个 CustomBondForce（内部 1-4 力）。"
        "背景 NB 已经保留了全部 L–L exception，再加一份就是把 1-4 的 LJ 和库仑"
        "各算两遍。"
    )

    nonbonded_like = [
        f for f in group2 if isinstance(f, openmm.CustomNonbondedForce)
    ]
    assert len(nonbonded_like) == 1, (
        f"Group 2 应恰有 1 个 CustomNonbondedForce（只补普通库仑），"
        f"实得 {len(nonbonded_like)} 个。"
    )
    internal = nonbonded_like[0]
    for local, global_index in enumerate(LIGAND):
        params = internal.getParticleParameters(global_index)
        # 参数序为 (charge, sigma, epsilon)，与 create_ligand_internal_force 一致。
        assert float(params[2]) == pytest.approx(0.0, abs=0.0), (
            f"配体原子 {global_index} 在 Group 2 的 epsilon = {params[2]} ≠ 0。"
            "Group 2 只该补库仑；LJ 已经在背景 NB 里，非零 ε 就是重复计入。"
        )


def test_background_keeps_ligand_lj_and_all_ligand_ligand_exceptions():
    """背景侧的前提：配体 σ/ε 与 L–L exception 都必须原样保留。

    这条是上面两条的对偶——如果哪天有人"顺手"把背景也按 dual builder 那样清空，
    那 Group 2 只补库仑就会**少算** LJ，方向相反但同样错。
    """
    charges = [0.4, -0.35, 0.25, -0.30, 0.3, -0.3, 0.2, -0.2, 0.0]
    built, _wrapper = _build_shadow(_system(charges=charges))

    background = next(
        f for f in built.getForces()
        if isinstance(f, openmm.NonbondedForce)
    )
    ligand_set = set(LIGAND)
    for index in LIGAND:
        q, sigma, epsilon = background.getParticleParameters(index)
        assert float(q.value_in_unit(unit.elementary_charge)) == pytest.approx(0.0), (
            f"背景 NB 里配体原子 {index} 的电荷未清零——静电会与 shadow CV 双计。"
        )
        assert float(epsilon.value_in_unit(unit.kilojoule_per_mole)) > 0.0, (
            f"背景 NB 里配体原子 {index} 的 epsilon 被清零了。"
            "shadow 腿的 VdW 全程满强度、属于 U_common，不该被清。"
        )

    found_14 = False
    for i in range(background.getNumExceptions()):
        p1, p2, chargeprod, sigma, epsilon = background.getExceptionParameters(i)
        if int(p1) in ligand_set and int(p2) in ligand_set:
            if float(epsilon.value_in_unit(unit.kilojoule_per_mole)) > 0.0:
                found_14 = True
    assert found_14, (
        "背景 NB 里找不到带非零 epsilon 的 L–L exception（1-4）。"
        "它必须留在背景侧；Group 2 不再补 1-4，两边同时没有就是漏算。"
    )
