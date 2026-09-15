"""couple-intramol=no：去电荷腿的配体**分子内**库仑必须逐 λ 恒定。

## 缺陷是什么（2026-09-14 定级，先在真实 run 上量化后修复）

v4 口径（`pme_decharge_v4_..._normal_pairs_annihilated`）把配体内部的普通
≥1-5 对连同配体–环境对一起随粒子 offset 缩放。因为两侧都带 offset，这些对按
**λ²** 走 —— 配体的分子内库仑随 λ 一起湮灭。

当时的理由写在 `configure_pme_ligand_charge_offsets` 的注释里：「两腿的配体内部
Hamiltonian 完全相同，湮灭项在 ΔG_bind 里严格相消」。

**v4 不是「公式写错了」**：它的循环形式上闭合（两腿终态是同一个参考态），完美采样
极限下与 v5 给同一个 ΔG_bind。假的是「**严格相消**」这四个字 —— 哈密顿量相同不等于
自由能相同，这一项的两腿之差是真实的构象重组功，必须靠采样得到。真正的失效是
**条件数**：把一个 ~90 kJ/mol 的物理量做成两个 ~500 kJ/mol 项之差，结合态配体在
500 ps × 8 个 λ 态里完不成构象弛豫 ⟹ 滞后。

实测（cyclod_ligand2 / CypD，400 帧/腿，直接从 DCD + System XML 算）：

    <U_intra> complex = -539.91 kJ/mol   (Rg 0.457 nm)
    <U_intra> solvent = -451.25 kJ/mol   (Rg 0.364 nm)
                   差 = -88.66 kJ/mol
    ΔG_bind 对实验的误差 = -88.05 kJ/mol      <- 1:1，占全部误差

在 4W53/toluene（Σq²=0.499 e²、刚性）上两腿只差 +1.56 kJ/mol，所以唯一验过实验值
的体系对这个失效模式免疫 —— 别再拿它当 stage 1 正确的证据。

## 修法（PME_DECHARGE_MODEL_VERSION -> v5）

`_freeze_ligand_internal_coulomb` 给去电荷系统挂一个 (1-λ²) 前缀的
`CustomBondForce`（`abfe_core.create_ligand_internal_coulomb_force`），逐对展开、
不带 cutoff：主 NB 力给出 λ²·U_intra，补偿项给出 (1-λ²)·U_intra，合计恒为 U_intra。
λ=1 时补偿项**恒等于 0**，物理端点仍逐位等于原始 System（P0-01 的不变量不受影响）。

## 不要这样让本文件变绿

- 放宽容差去掩盖 λ 之间的力差异；
- 把"配体"缩小到没有 ≥1-5 普通对的规模（本 fixture 的 8 原子链有 10 对）；
- 用 NoCutoff 代替 PME —— 缺陷与修复都只在真实 PME 下才有意义。
"""

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, unit  # noqa: E402

import abfe_core as ac  # noqa: E402
import ibs_engine as ie  # noqa: E402

from test_pme_decharge_endpoint_equivalence import (  # noqa: E402
    _N_LIG,
    BOX_NM,
    _build_ligand_positions_nm,
    _build_reference_system,
    _n_normal_ll_pairs,
)

# Reference 平台双精度。λ=1 端点是精确的代数恒等（补偿项恒为 0），只允许求值器噪声。
ENERGY_TOL_KJ_MOL = 1.0e-6
FORCE_TOL_KJ_MOL_NM = 1.0e-6

# 修完之后**仍然残留**的那一项：配体与它自己周期镜像的相互作用。倒空间把它算进去、
# 并且同样按 λ² 走，而逐对补偿只覆盖主镜像；PME 下无法消除（GROMACS 的
# couple-intramol=no 同样留着它），量级 ~1/L³。
#
# 两条量它时的规矩：
#  1. **直接读 E(λ=0)−E(λ=1)，不做任何 Ewald 自能手工扣减** —— 自能项与倒空间
#     自相互作用精确相消，不是可以单独加减的物理量。本仓库为此撤销过一次 `+C·λ²`
#     修正（见 ibs_engine.py `TraditionalMBARAnalyzer.compute_u_kn` 的注释）。
#  2. **别拿本 fixture 的残差量级去推断生产体系**：它是人为拉直的 8 原子链、偶极
#     远大于真实配体，残差因此大 1–2 个量级。生产体系的实测值登记在 docs/STATUS.md，
#     不在这里抄一份副本。
#
# 所以这里**不设绝对容差**（那就是一个魔法数），而是拿摘掉补偿项的同一体系做自校准
# 对照，只要求「修完之后剩的比没修时小 20 倍以上」这个量级关系。
MAX_RESIDUAL_FRACTION = 0.05


def _box_vectors(box_nm):
    return [
        openmm.Vec3(box_nm, 0, 0) * unit.nanometer,
        openmm.Vec3(0, box_nm, 0) * unit.nanometer,
        openmm.Vec3(0, 0, box_nm) * unit.nanometer,
    ]


def _energy_forces(system, positions_nm, box_vectors, lambda_values=None):
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    try:
        context.setPeriodicBoxVectors(*box_vectors)
        context.setPositions(positions_nm * unit.nanometer)
        for name, value in (lambda_values or {}).items():
            context.setParameter(name, float(value))
        state = context.getState(getEnergy=True, getForces=True)
        return (
            float(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)),
            np.asarray(
                state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                ),
                dtype=float,
            ),
        )
    finally:
        del context, integrator


def _ligand_only_system(box_nm, cutoff_nm=1.2):
    """只留配体的周期体系：此时**全部** λ 依赖都是分子内的（含自镜像）。"""
    full, _positions = _build_reference_system()
    system = openmm.System()
    for i in range(_N_LIG):
        system.addParticle(full.getParticleMass(i))
    src_nb = next(f for f in full.getForces() if isinstance(f, NonbondedForce))
    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.PME)
    nb.setCutoffDistance(cutoff_nm * unit.nanometer)
    nb.setUseDispersionCorrection(src_nb.getUseDispersionCorrection())
    for i in range(_N_LIG):
        nb.addParticle(*src_nb.getParticleParameters(i))
    for e in range(src_nb.getNumExceptions()):
        p1, p2, cp, sig, eps = src_nb.getExceptionParameters(e)
        if int(p1) < _N_LIG and int(p2) < _N_LIG:
            nb.addException(int(p1), int(p2), cp, sig, eps)
    system.addForce(nb)
    box = _box_vectors(box_nm)
    system.setDefaultPeriodicBoxVectors(*box)
    return system, box



def _fixture_intra_coulomb_kj_mol() -> float:
    """本 fixture 的 ≥1-5 配体内部库仑（现算）。用来给对照臂定下界，避免写死阈值。"""
    lig_q = np.asarray(
        [nb for nb in __import__(
            "test_pme_decharge_endpoint_equivalence"
        )._LIGAND_Q_E],
        dtype=float,
    )
    pos = _build_ligand_positions_nm()
    ke = ac.ONE_4PI_EPS0_KJ_NM_PER_MOL_E2
    total = 0.0
    for i in range(_N_LIG):
        for j in range(i + 1, _N_LIG):
            if j - i < 4:      # 与 fixture 的拓扑约定一致：<1-5 有 exception
                continue
            total += ke * lig_q[i] * lig_q[j] / float(np.linalg.norm(pos[i] - pos[j]))
    return total


def _lambda_residual(box_nm, compensate: bool):
    """e(λ=0) − e(λ=1)，以及 max|ΔF|。`compensate=False` 复现 v4 的湮灭口径。"""
    system, box = _ligand_only_system(box_nm)
    ie.configure_pme_ligand_charge_offsets(
        system, list(range(_N_LIG)), lambda_name="lambda_coul"
    )
    if not compensate:
        # 把补偿项摘掉 = v4「普通 L–L 对随 λ² 湮灭」。对照组必须由同一条构造路径
        # 产生，只差这一个力，否则比的就不是同一件事。
        removed = [
            i
            for i in range(system.getNumForces())
            if isinstance(system.getForce(i), openmm.CustomBondForce)
        ]
        assert len(removed) == 1, f"预期只有补偿项这一个 CustomBondForce，实际 {len(removed)}"
        system.removeForce(removed[0])
    positions = _build_ligand_positions_nm()
    e1, f1 = _energy_forces(system, positions, box, {"lambda_coul": 1.0})
    e0, f0 = _energy_forces(system, positions, box, {"lambda_coul": 0.0})
    return e0 - e1, float(np.abs(f0 - f1).max())


def test_annihilated_intramolecular_coulomb_is_gone():
    """补偿后的 λ 残差必须远小于 v4 的湮灭项 —— 后者就是灌进 ΔG_bind 的那 88.7。"""
    de_fixed, df_fixed = _lambda_residual(6.0, compensate=True)
    de_v4, df_v4 = _lambda_residual(6.0, compensate=False)

    # 对照臂必须真的有东西可测：拿 fixture 自己的 ≥1-5 内部库仑现算一个下界，
    # 不写死阈值。未补偿时 λ:1→0 至少要暴露出它的一个可观比例。
    floor = 0.1 * abs(_fixture_intra_coulomb_kj_mol())
    assert abs(de_v4) > floor, (
        f"对照组（v4 湮灭口径）的 λ 残差只有 {de_v4:.4g} kJ/mol，低于由 fixture 自身"
        f"内部库仑算出的下界 {floor:.4g} —— 这个 fixture 已测不出缺陷。"
    )
    assert abs(de_fixed) < MAX_RESIDUAL_FRACTION * abs(de_v4), (
        f"补偿后仍残留 {de_fixed:.4g} kJ/mol，未补偿是 {de_v4:.4g} kJ/mol —— "
        "分子内库仑没有被冻结住。"
    )
    assert df_fixed < MAX_RESIDUAL_FRACTION * df_v4, (
        f"补偿后力残差 {df_fixed:.4g} vs 未补偿 {df_v4:.4g} kJ/mol/nm"
    )


def test_remaining_residual_is_the_periodic_self_image_term():
    """剩下的那点残差必须随盒子变大而衰减 —— 证明它是有限尺寸项，不是漏补的分子内项。

    漏补的分子内项与盒子无关；自镜像项 ~1/L³。盒子翻倍时前者不动、后者掉一个量级。
    """
    de_small, _ = _lambda_residual(4.0, compensate=True)
    de_large, _ = _lambda_residual(8.0, compensate=True)
    assert abs(de_large) < 0.5 * abs(de_small), (
        f"box 4→8 nm 残差 {de_small:.4g} → {de_large:.4g} kJ/mol，没有随盒子衰减："
        "剩下的不是周期自镜像项，而是真的漏补了分子内对。"
    )


def test_lambda_one_endpoint_still_equals_the_original_system():
    """P0-01 的不变量：补偿项在 lambda=1 时恒为 0，物理端点不得被改写。"""
    original, positions_q = _build_reference_system()
    positions = np.asarray(positions_q.value_in_unit(unit.nanometer), dtype=float)
    box = _box_vectors(BOX_NM)
    e_ref, f_ref = _energy_forces(original, positions, box)

    prepared, _ = _build_reference_system()
    ie.configure_pme_ligand_charge_offsets(
        prepared, list(range(_N_LIG)), lambda_name="lambda_coul"
    )
    e, f = _energy_forces(prepared, positions, box, {"lambda_coul": 1.0})

    assert abs(e - e_ref) < ENERGY_TOL_KJ_MOL, (
        f"lambda=1 能量偏离原始 System {abs(e - e_ref):.6g} kJ/mol"
    )
    assert float(np.abs(f - f_ref).max()) < FORCE_TOL_KJ_MOL_NM


def test_exclusion_set_is_the_nonbonded_exception_set():
    """两个 stage 共用的那份排除表，必须与 NB 的 L–L exception 逐对一致。

    stage 1 冻结的是 `NonbondedForce` 的 L–L exception；stage 2 的 Group 2 走的是
    键/角/约束/exception 的并集。两者一旦不同，接缝上就会漏算或重复算内部对。
    真实体系上实测两边都是 187 对；这里把这个前提钉死在最小 fixture 上。
    """
    system, _positions = _build_reference_system()
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))
    ligand = list(range(_N_LIG))
    ref_excl = [
        (int(nb.getExceptionParameters(i)[0]), int(nb.getExceptionParameters(i)[1]))
        for i in range(nb.getNumExceptions())
    ]
    collected = ac.collect_ligand_internal_exclusions(nb, ligand, ref_excl, system)
    from_exceptions = {
        (min(p1, p2), max(p1, p2))
        for p1, p2 in ref_excl
        if p1 < _N_LIG and p2 < _N_LIG
    }
    assert collected == from_exceptions

    force = ac.create_ligand_internal_coulomb_force(
        nb, ligand, {i: nb.getParticleParameters(i)[0]._value for i in ligand},
        system=system, reference_exclusions=ref_excl, scale_expr="1",
    )
    n_total = _N_LIG * (_N_LIG - 1) // 2
    assert force is not None
    assert force.getNumBonds() == n_total - len(from_exceptions) > 0
    # 与该 fixture 自己数普通 L–L 对的口径对上（不各写一套）。
    assert force.getNumBonds() == _n_normal_ll_pairs(system)
