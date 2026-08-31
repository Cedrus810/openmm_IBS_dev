"""`abfe_core.bake_global_parameter_into_fixed_nonbonded_force` 契约测试。

对应 `docs/experiments/STAGE2_CHARGE_TRANSFER_HANDOFF_PROPOSAL.md`：这个函数是 charging→
vanishing handoff 提案里那条"结构性删除危险活参数"的核心工具，2026-08-11
用户审阅后钉死了六条契约（见函数 docstring），本文件逐条对着测：

1. 只删目标参数，其它 GlobalParameter/offset 原样保留；
2. 同一粒子/exception 上多条挂在目标参数下的 offset 先聚合再烘焙一次；
3. `NonbondedForce` 的非 particle/exception 配置完整保留；
4. 目标参数若被别的 Force 引用，fail closed；
5. charge/sigma/epsilon 都要正确烘焙，不只是 charge；
6. `lambda_value` 只接受精确的 0.0/1.0。

以及"烘焙结果与显式设 Context 参数逐位相同"这条最直接的正确性证明。
"""

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, unit  # noqa: E402

import abfe_core as core  # noqa: E402

BOX_NM = np.diag([5.0, 5.0, 5.0])
POSITIONS = [[1.0, 1.0, 1.0], [1.3, 1.0, 1.0], [1.0, 1.4, 1.0]]


def _simple_system(n_particles=3):
    system = openmm.System()
    for _ in range(n_particles):
        system.addParticle(12.0 * unit.dalton)
    system.setDefaultPeriodicBoxVectors(*(BOX_NM * unit.nanometer))
    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.PME)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    for i in range(n_particles):
        nb.addParticle(
            0.1 * (i + 1) * unit.elementary_charge, 0.3 * unit.nanometer, 0.2 * unit.kilojoule_per_mole
        )
    system.addForce(nb)
    return system, nb


def _find_nb(system):
    return next(f for f in system.getForces() if isinstance(f, NonbondedForce))


def _energy_and_forces(system, positions_nm, global_parameters=None):
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(system, integrator, openmm.Platform.getPlatformByName("Reference"))
    context.setPeriodicBoxVectors(*(BOX_NM * unit.nanometer))
    context.setPositions(np.asarray(positions_nm, dtype=float) * unit.nanometer)
    for name, value in (global_parameters or {}).items():
        context.setParameter(name, float(value))
    state = context.getState(getEnergy=True, getForces=True)
    e = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    f = np.asarray(state.getForces().value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    del context, integrator
    return e, f


# ---------------------------------------------------------------------------
# 正确性核心：烘焙结果必须与"显式设 Context 参数"逐位相同。
# ---------------------------------------------------------------------------


def test_bake_matches_explicit_context_bit_for_bit():
    system, nb = _simple_system()
    nb.addGlobalParameter("lam_test", 1.0)
    for idx in range(3):
        q, sigma, eps = nb.getParticleParameters(idx)
        nb.setParticleParameters(idx, 0.0 * unit.elementary_charge, sigma, eps)
        nb.addParticleParameterOffset(
            "lam_test", idx, q, 0.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole
        )

    want_e, want_f = _energy_and_forces(system, POSITIONS, {"lam_test": 0.0})

    baked = core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 0.0)
    got_e, got_f = _energy_and_forces(baked, POSITIONS)  # 不传任何 global_parameters

    assert got_e == pytest.approx(want_e, rel=1e-12)
    assert np.allclose(got_f, want_f, atol=1e-9)


def test_bake_removes_the_global_parameter_structurally():
    system, nb = _simple_system()
    nb.addGlobalParameter("lam_test", 1.0)
    nb.addParticleParameterOffset(
        "lam_test", 0, 0.5 * unit.elementary_charge, 0.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole
    )
    baked = core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 0.0)
    nb_baked = _find_nb(baked)
    names = [nb_baked.getGlobalParameterName(i) for i in range(nb_baked.getNumGlobalParameters())]
    assert "lam_test" not in names

    # 结构性证明，不只是"应该没有了"：Context 也不认识这个参数了。
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(baked, integrator, openmm.Platform.getPlatformByName("Reference"))
    with pytest.raises(Exception):
        context.setParameter("lam_test", 0.0)
    del context, integrator


# ---------------------------------------------------------------------------
# 契约 1：只删目标参数，其它原样保留。
# ---------------------------------------------------------------------------


def test_bake_preserves_unrelated_global_parameters_and_offsets():
    system, nb = _simple_system()
    nb.addGlobalParameter("lam_test", 1.0)
    nb.addGlobalParameter("lam_other", 0.7)
    nb.addParticleParameterOffset(
        "lam_test", 0, 0.5 * unit.elementary_charge, 0.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole
    )
    nb.addParticleParameterOffset(
        "lam_other", 1, 0.2 * unit.elementary_charge, 0.01 * unit.nanometer, 0.05 * unit.kilojoule_per_mole
    )

    baked = core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 0.0)
    nb_baked = _find_nb(baked)

    names = {
        nb_baked.getGlobalParameterName(i): nb_baked.getGlobalParameterDefaultValue(i)
        for i in range(nb_baked.getNumGlobalParameters())
    }
    assert names == {"lam_other": 0.7}
    assert nb_baked.getNumParticleParameterOffsets() == 1
    pname, particle, q_scale, _s_scale, _e_scale = nb_baked.getParticleParameterOffset(0)
    assert pname == "lam_other"
    assert int(particle) == 1
    # OpenMM 对"恰好是某个值"的 offset scale 读回来的类型不总是稳定
    # （有时是 Quantity，有时是裸 float），两种都接受。
    q_val = q_scale.value_in_unit(unit.elementary_charge) if hasattr(q_scale, "value_in_unit") else float(q_scale)
    assert q_val == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# 契约 2：同一 (parameter, particle/exception) 上出现多条 offset 必须
# fail closed——**不是**先聚合再烘焙。
#
# 2026-08-11 用真实 Context 实测纠正：`parameter = base + Σ(global_i ×
# scale_i)` 里的 Σ 说的是"同一个粒子上不同 GlobalParameter 各自的 offset
# 相加"，不是"同一个 (parameter, particle) 重复挂多条也相加"——对后一种情况，
# OpenMM 的 Context 只认最后一条（0.3+0.2 两条追加，结果对应 0.2，不是
# 0.5；exception 同理）。真实生产代码从不这样重复调用，所以这里按更安全的
# 方式处理：检测到重复就报错，不去复现这个没有文档的"取最后一条"行为。
# ---------------------------------------------------------------------------


def test_bake_fails_closed_on_duplicate_particle_offset_for_the_same_parameter():
    system, nb = _simple_system()
    nb.addGlobalParameter("lam_test", 1.0)
    nb.addParticleParameterOffset(
        "lam_test", 0, 0.3 * unit.elementary_charge, 0.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole
    )
    nb.addParticleParameterOffset(
        "lam_test", 0, 0.2 * unit.elementary_charge, 0.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole
    )
    with pytest.raises(RuntimeError, match="多条"):
        core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 1.0)


def test_bake_fails_closed_on_duplicate_exception_offset_for_the_same_parameter():
    system, nb = _simple_system()
    nb.addGlobalParameter("lam_test", 1.0)
    nb.addException(0, 1, 0.02 * unit.elementary_charge**2, 0.3 * unit.nanometer, 0.15 * unit.kilojoule_per_mole)
    nb.addExceptionParameterOffset(
        "lam_test", 0, 0.01 * unit.elementary_charge**2, 0.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole
    )
    nb.addExceptionParameterOffset(
        "lam_test", 0, 0.02 * unit.elementary_charge**2, 0.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole
    )
    with pytest.raises(RuntimeError, match="多条"):
        core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 1.0)


def test_bake_preserves_unrelated_exception_offset_when_target_offset_is_singular():
    """确认"多条就报错"没有误伤"target 参数只有一条、其它参数也有 offset"
    这个正常场景——即契约 1 在 exception 上的对应版本。"""
    system, nb = _simple_system()
    nb.addGlobalParameter("lam_test", 1.0)
    nb.addGlobalParameter("lam_other", 1.0)
    nb.addException(0, 1, 0.02 * unit.elementary_charge**2, 0.3 * unit.nanometer, 0.15 * unit.kilojoule_per_mole)
    nb.addExceptionParameterOffset(
        "lam_test", 0, 0.01 * unit.elementary_charge**2, 0.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole
    )
    nb.addExceptionParameterOffset(
        "lam_other", 0, 0.5 * unit.elementary_charge**2, 0.0 * unit.nanometer, 0.0 * unit.kilojoule_per_mole
    )

    want_e, want_f = _energy_and_forces(system, POSITIONS, {"lam_test": 1.0, "lam_other": 0.0})
    baked = core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 1.0)
    got_e, got_f = _energy_and_forces(baked, POSITIONS, {"lam_other": 0.0})
    assert got_e == pytest.approx(want_e, rel=1e-12)
    assert np.allclose(got_f, want_f, atol=1e-9)

    nb_baked = _find_nb(baked)
    assert nb_baked.getNumExceptionParameterOffsets() == 1  # 只剩 lam_other 那条
    pname, _exc_idx, _cp, _s, _e = nb_baked.getExceptionParameterOffset(0)
    assert pname == "lam_other"


# ---------------------------------------------------------------------------
# 契约 3：NonbondedForce 的非 particle/exception 配置完整保留。
# ---------------------------------------------------------------------------


def test_bake_preserves_full_nonbonded_force_configuration():
    system, nb = _simple_system()
    nb.setCutoffDistance(1.234 * unit.nanometer)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(1.1 * unit.nanometer)
    nb.setUseDispersionCorrection(True)
    nb.setReactionFieldDielectric(45.0)
    nb.setEwaldErrorTolerance(1e-5)
    nb.setForceGroup(3)
    nb.setName("MyNonbondedForce")
    nb.setExceptionsUsePeriodicBoundaryConditions(True)
    nb.addGlobalParameter("lam_test", 1.0)

    baked = core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 0.0)
    nb_baked = _find_nb(baked)

    assert nb_baked.getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(1.234)
    assert nb_baked.getUseSwitchingFunction() is True
    assert nb_baked.getSwitchingDistance().value_in_unit(unit.nanometer) == pytest.approx(1.1)
    assert nb_baked.getUseDispersionCorrection() is True
    assert nb_baked.getReactionFieldDielectric() == pytest.approx(45.0)
    assert nb_baked.getEwaldErrorTolerance() == pytest.approx(1e-5)
    assert nb_baked.getForceGroup() == 3
    assert nb_baked.getName() == "MyNonbondedForce"
    assert nb_baked.getExceptionsUsePeriodicBoundaryConditions() is True


# ---------------------------------------------------------------------------
# 契约 4：目标参数若被别的 Force 引用，fail closed。
# ---------------------------------------------------------------------------


def test_bake_fails_closed_if_parameter_used_by_another_force():
    system, nb = _simple_system()
    nb.addGlobalParameter("lam_test", 1.0)
    other = openmm.CustomBondForce("lam_test*r")
    other.addGlobalParameter("lam_test", 1.0)
    other.addBond(0, 1, [])
    system.addForce(other)

    with pytest.raises(RuntimeError, match="非 NonbondedForce"):
        core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 0.0)


def test_bake_fails_closed_if_parameter_not_found():
    system, _nb = _simple_system()
    with pytest.raises(RuntimeError, match="无法烘焙"):
        core.bake_global_parameter_into_fixed_nonbonded_force(system, "does_not_exist", 0.0)


def test_bake_fails_closed_if_multiple_nonbonded_forces_share_the_parameter():
    system = openmm.System()
    for _ in range(2):
        system.addParticle(12.0 * unit.dalton)
    system.setDefaultPeriodicBoxVectors(*(BOX_NM * unit.nanometer))
    for _ in range(2):
        nb = NonbondedForce()
        nb.setNonbondedMethod(NonbondedForce.PME)
        nb.setCutoffDistance(1.0 * unit.nanometer)
        for _i in range(2):
            nb.addParticle(0.0 * unit.elementary_charge, 0.3 * unit.nanometer, 0.2 * unit.kilojoule_per_mole)
        nb.addGlobalParameter("lam_test", 1.0)
        system.addForce(nb)

    with pytest.raises(RuntimeError, match="出现在"):
        core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 0.0)


# ---------------------------------------------------------------------------
# 契约 5：charge/sigma/epsilon 都要正确烘焙，不只是 charge。
# ---------------------------------------------------------------------------


def test_bake_handles_sigma_and_epsilon_particle_offsets_too():
    system, nb = _simple_system()
    nb.addGlobalParameter("lam_test", 1.0)
    _q0, sigma0, eps0 = nb.getParticleParameters(0)
    nb.addParticleParameterOffset(
        "lam_test", 0, 0.0 * unit.elementary_charge, 0.05 * unit.nanometer, 0.1 * unit.kilojoule_per_mole
    )

    want_e, want_f = _energy_and_forces(system, POSITIONS, {"lam_test": 1.0})
    baked = core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 1.0)
    got_e, got_f = _energy_and_forces(baked, POSITIONS)
    assert got_e == pytest.approx(want_e, rel=1e-12)
    assert np.allclose(got_f, want_f, atol=1e-9)

    nb_baked = _find_nb(baked)
    _q, sigma_baked, eps_baked = nb_baked.getParticleParameters(0)
    assert sigma_baked.value_in_unit(unit.nanometer) == pytest.approx(
        sigma0.value_in_unit(unit.nanometer) + 0.05
    )
    assert eps_baked.value_in_unit(unit.kilojoule_per_mole) == pytest.approx(
        eps0.value_in_unit(unit.kilojoule_per_mole) + 0.1
    )


# ---------------------------------------------------------------------------
# 契约 6：lambda_value 只接受精确的 0.0/1.0。
# ---------------------------------------------------------------------------


def test_bake_rejects_non_endpoint_lambda():
    system, nb = _simple_system()
    nb.addGlobalParameter("lam_test", 1.0)
    with pytest.raises(ValueError, match="端点"):
        core.bake_global_parameter_into_fixed_nonbonded_force(system, "lam_test", 0.5)
