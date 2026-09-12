"""`build_shadow_coul_ibs_system` 的第一份测试覆盖。

## 为什么单开一个文件

这个 builder 与它唯一的调用者 `IBSWindowManagerShadowCoul`（`ibs_engine.py`）在
整个 `tests/` 里**零提及** —— EXP-031 独立核查时点出来的（见
`docs/EXP-031_GPU_OPTIMIZATION_2026-09-09.md`；原始 PLAN §11.2 在 CUDA 沙箱，不在本仓）。
它与 `build_ibs_dual_system` 共用同一套 `IBSBiasForce` 多态 log-sum-exp 框架，
只是把 CV 从软核 VdW 换成短程"影子"库仑，所以**任何动到 Group-1 形态的改动都会
同时落到它头上，而此前没有任何东西会发现**。

特别是 EXP-031 S1 的 `IBS_BIAS_OMIT_ZERO_REST_CVS` 开关：它在这条路径上
**一次都没有执行过**。翻默认值之前必须先有这份覆盖，否则等于闭着眼睛改。

## 这里断言什么

1. legacy 形态能建出来（**这个 builder 的第一次测试执行**）；
2. 两种形态的 CV 数目正确（legacy 2K、omit K），且 omit 形态确实没有 `cv_k_rest`；
3. 两种形态在同一构型下的 Group-1 能量**逐比特相同** —— 这是 S1 的核心主张
   （`cv_k_rest` 恒等于零，摘掉它是精确恒等变形）在这条路径上的首次检验；
4. 每个 CV 的排除表按升序（与 `test_exclusion_order_is_ascending.py` 同一条不变量）。
"""
from __future__ import annotations

import pathlib
import sys

import pytest

# 不需要 GPU：只建 System、读排除表 / CV 名单。不标就进不了 `-m cpu_only`
# 那道常规门，防复发的意义会归零。
pytestmark = pytest.mark.cpu_only

REPO = pathlib.Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

import ibs_engine as ie  # noqa: E402

N_PARTICLES = 9
LIGAND = [0, 1]
BOX_NM = 4.0
TEMP = 300.0 * unit.kelvin
LAMBDAS = [1.0, 0.5, 0.0]


def _system():
    """一个带 PME NonbondedForce 与真实电荷的最小周期性体系。"""
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(BOX_NM, 0, 0), openmm.Vec3(0, BOX_NM, 0), openmm.Vec3(0, 0, BOX_NM)
    )
    for _ in range(N_PARTICLES):
        system.addParticle(12.0 * unit.dalton)
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    # 电荷总和为零，且配体带非零电荷 —— 否则"影子库仑"CV 全是 0，测试什么都没验到
    charges = [0.4, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.0]
    for q in charges:
        nb.addParticle(q * unit.elementary_charge, 0.3 * unit.nanometer,
                       0.1 * unit.kilojoule_per_mole)
    nb.addException(0, 1, 0.0, 0.3 * unit.nanometer, 0.0)
    system.addForce(nb)
    return system


def _positions():
    return [openmm.Vec3(0.5 + 0.25 * i, 0.5 + 0.05 * (i % 3), 0.5) for i in range(N_PARTICLES)] * unit.nanometer


def _build(omit):
    saved = ie.IBS_BIAS_OMIT_ZERO_REST_CVS
    ie.IBS_BIAS_OMIT_ZERO_REST_CVS = bool(omit)
    try:
        system = _system()
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
    finally:
        ie.IBS_BIAS_OMIT_ZERO_REST_CVS = saved


def _cv_names(wrapper):
    force = wrapper.get_force()
    return [force.getCollectiveVariableName(i)
            for i in range(force.getNumCollectiveVariables())]


def _group1_energy(system, wrapper):
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(system, integrator,
                             openmm.Platform.getPlatformByName("Reference"))
    context.setPositions(_positions())
    for k in range(len(LAMBDAS)):
        context.setParameter(f"abfe_shadow_f_{k}", 1.5 * k - 2.0)
    context.setParameter("abfe_shadow_bias_scale", 0.73)
    state = context.getState(getEnergy=True, groups={1})
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    del context, integrator
    return energy


def test_shadow_coul_builder_runs_at_all_legacy_form():
    """这个 builder 在 `tests/` 里的**第一次**执行。"""
    built, wrapper = _build(omit=False)
    assert built.getNumParticles() == N_PARTICLES
    names = _cv_names(wrapper)
    assert len(names) == 2 * len(LAMBDAS), f"legacy 形态应有 2K 个 CV，实得 {names}"
    assert all(f"cv_{k}_int" in names for k in range(len(LAMBDAS)))
    assert all(f"cv_{k}_rest" in names for k in range(len(LAMBDAS)))
    assert wrapper.get_force().getForceGroup() == 1


def test_shadow_coul_builder_omit_form_registers_only_int_cvs():
    """EXP-031 S1 的 omit 形态在这条路径上的**第一次**执行。"""
    built, wrapper = _build(omit=True)
    names = _cv_names(wrapper)
    assert len(names) == len(LAMBDAS), f"omit 形态应只有 K 个 CV，实得 {names}"
    assert not any(n.endswith("_rest") for n in names), (
        f"omit 形态不得注册任何 cv_k_rest；实得 {names}"
    )


def test_shadow_coul_both_forms_give_bit_identical_group1_energy():
    """S1 的核心主张在这条路径上的首次检验。

    `cv_k_rest` 是零粒子的 `CustomExternalForce("0")`，对能量与力的贡献严格为 0，
    所以摘掉它是**精确恒等变形**，不是近似。这里要求逐比特相同，不是"接近"。
    """
    legacy_system, legacy_wrap = _build(omit=False)
    omit_system, omit_wrap = _build(omit=True)

    e_legacy = _group1_energy(legacy_system, legacy_wrap)
    e_omit = _group1_energy(omit_system, omit_wrap)

    import math
    assert math.isfinite(e_legacy), "legacy 形态的 Group-1 能量非有限，测例本身有问题"
    assert e_legacy == e_omit, (
        "摘掉恒零的 cv_k_rest 改变了 Shadow-Coulomb 路径的 Group-1 能量："
        f"legacy={e_legacy!r} omit={e_omit!r}（差 {e_omit - e_legacy!r}）。"
        "cv_k_rest 恒等于零，这个差必须精确为 0。"
    )


@pytest.mark.parametrize("omit", [False, True])
def test_shadow_coul_cv_exclusion_lists_are_ascending(omit):
    """与 `test_exclusion_order_is_ascending.py` 同一条不变量，覆盖这条路径。"""
    _built, wrapper = _build(omit=omit)
    force = wrapper.get_force()
    checked = 0
    for i in range(force.getNumCollectiveVariables()):
        cv = force.getCollectiveVariable(i)
        if not isinstance(cv, openmm.CustomNonbondedForce):
            continue
        got = [tuple(sorted(int(x) for x in cv.getExclusionParticles(e)))
               for e in range(cv.getNumExclusions())]
        assert got == sorted(got), (
            f"CV {force.getCollectiveVariableName(i)} 的排除表不是升序（前 8 条 {got[:8]}）；"
            "乱序会让该力每步白烧时间而能量逐比特不变，见 "
            "docs/EXP-031_GPU_OPTIMIZATION_2026-09-09.md（原始 PROPOSAL §2 在 CUDA 沙箱）"
        )
        checked += 1
    assert checked > 0, "没有检查到任何 CustomNonbondedForce CV —— 测例形态变了"
