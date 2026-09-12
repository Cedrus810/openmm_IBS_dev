"""Deterministic, CPU-only contract tests for the existing EM patch.

These tests exercise only the process-local patch and tiny one-particle
OpenMM Systems.  They do not import an EXP-030 runner, touch production
artifacts, or start molecular dynamics beyond OpenMM's minimizer call.
"""

from types import SimpleNamespace

import pytest


openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402


from local_residual import em_no_residual as em_patch  # noqa: E402
import ibs_engine  # noqa: E402



pytestmark = pytest.mark.cpu_only

def _topology_one_atom():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("X", chain)
    topology.addAtom("X", app.element.carbon, residue)
    return topology


def _manager(factory=None):
    manager = object.__new__(ibs_engine.IBSWindowManagerDualLambda)
    manager.residual_basis_force_factory = factory
    manager.topology = _topology_one_atom()
    manager.temperature = 300.0 * unit.kelvin
    return manager


def _periodic_box():
    length = 2.0 * unit.nanometer
    return (
        openmm.Vec3(length, 0, 0),
        openmm.Vec3(0, length, 0),
        openmm.Vec3(0, 0, length),
    )


def _residual_cv_force():
    """真实形状的残差力：`CustomCVForce` 里挂一个叫 `exp025_residual_basis` 的 CV。

    `em_no_residual._system_has_residual_cv()` 认的就是这个 CV 名字。
    """
    inner = openmm.CustomExternalForce("0.5*k*x^2")
    inner.addGlobalParameter("k", 1.0)
    inner.addParticle(0, [])
    cv = openmm.CustomCVForce("exp025_residual_basis")
    cv.addCollectiveVariable("exp025_residual_basis", inner)
    return cv


def _system_with_optional_residual(factory):
    system = openmm.System()
    system.addParticle(12.0 * unit.amu)
    system.setDefaultPeriodicBoxVectors(*_periodic_box())
    if factory is not None:
        system.addForce(factory())
    return system


@pytest.fixture
def installed_patch(monkeypatch):
    em_patch.uninstall()
    original_build = ibs_engine.IBSWindowManagerDualLambda._build_window_system
    original_minimize = app.Simulation.minimizeEnergy
    factory_calls = []

    def residual_factory():
        factory_calls.append("adapter")
        force = openmm.CustomExternalForce("0.5*k*x^2")
        force.addGlobalParameter("k", 1.0)
        force.addParticle(0, [])
        return force

    def fake_build(self, _lc_win, _lv_win, _resolved_box, _positions):
        factory = self.residual_basis_force_factory
        system = _system_with_optional_residual(factory)
        wrapper = SimpleNamespace(
            residual_enabled=factory is not None,
            prefix="test",
        )
        return system, wrapper

    monkeypatch.setattr(
        ibs_engine.IBSWindowManagerDualLambda,
        "_build_window_system",
        fake_build,
    )
    em_patch.install()
    yield {
        "factory": residual_factory,
        "factory_calls": factory_calls,
        "original_build": original_build,
        "original_minimize": original_minimize,
    }
    em_patch.uninstall()


def _simulation(topology, system):
    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    simulation = app.Simulation(
        topology,
        system,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    simulation.context.setPeriodicBoxVectors(*_periodic_box())
    simulation.context.setPositions([[0.2, 0.2, 0.2]] * unit.nanometer)
    return simulation


@pytest.mark.parametrize("order", [("baseline", "candidate"), ("candidate", "baseline")])
def test_em_patch_is_arm_local_and_candidate_uses_twin_once(installed_patch, order):
    candidate_manager = _manager(installed_patch["factory"])
    baseline_manager = _manager(None)
    candidate_built = False
    for arm in order:
        manager = baseline_manager if arm == "baseline" else candidate_manager
        system, wrapper = manager._build_window_system([], [], _periodic_box(), None)
        simulation = _simulation(manager.topology, system)
        simulation.minimizeEnergy(maxIterations=5)

        if arm == "baseline":
            assert wrapper.residual_enabled is False
            assert len(installed_patch["factory_calls"]) == (1 if candidate_built else 0)
            assert system.getNumForces() == 0
        else:
            candidate_built = True
            assert wrapper.residual_enabled is True
            assert installed_patch["factory_calls"] == ["adapter"]
            # The real Context retains the plugin-enabled System after EM;
            # only the temporary minimization Context is residual-free.
            assert system.getNumForces() == 1
            assert simulation.context.getSystem().getNumForces() == 1

        # ⚠️ [2026-09-10] 契约反转：EM 之后 `_STASH` **不再**清空。
        # 清掉它等于每个窗口只保护第一次 minimizeEnergy，而同一窗口后面还有
        # `_production_disaster_rollback` 和约束死锁缓解两次原生最小化，会带着
        # LocalManyBodyResidualForce 跑 EM 且无任何日志。生命周期改由
        # `patched_build_window_system` 入口的无条件 clear 划界。
        if arm == "candidate":
            assert em_patch._STASH != {}, (
                "候选臂 EM 后 stash 被清了 —— 同一窗口后续的原生最小化会失去保护"
            )
        else:
            assert em_patch._STASH == {}, "基线臂不该登记 twin"


def test_stash_survives_repeated_minimizations_inside_one_window(installed_patch):
    """同一个窗口里的**每一次** minimizeEnergy 都要走 twin，不只是第一次。

    这条钉的就是 2026-09-10 那个 bug 的形状：`_production_disaster_rollback`
    与死锁缓解分支各有一次 `sim.minimizeEnergy(...)`，原来它们命中
    `make_em_system is None` 直接走原生路径，静默带着残差力做 EM。
    """
    manager = _manager(installed_patch["factory"])
    system, wrapper = manager._build_window_system([], [], _periodic_box(), None)
    assert wrapper.residual_enabled is True

    simulation = _simulation(manager.topology, system)
    for attempt in range(3):
        simulation.minimizeEnergy(maxIterations=5)
        assert em_patch._STASH.get("make_em_system") is not None, (
            f"第 {attempt + 1} 次最小化之后 twin 构造器没了"
        )
    # 真 Context 始终保留插件力；只有临时最小化 Context 是残差-free 的。
    assert simulation.context.getSystem().getNumForces() == 1


def test_next_window_build_resets_the_stash(installed_patch):
    """生命周期边界：stash 最多活到下一个窗口建起来。"""
    candidate = _manager(installed_patch["factory"])
    candidate._build_window_system([], [], _periodic_box(), None)
    assert em_patch._STASH != {}

    baseline = _manager(None)
    baseline._build_window_system([], [], _periodic_box(), None)
    assert em_patch._STASH == {}, (
        "建下一个窗口没有重设 stash —— 上一个窗口的 twin 会漏到这个窗口"
    )


def test_em_patch_clears_twin_and_restores_factory_after_exception(installed_patch, monkeypatch):
    em_patch.uninstall()
    manager = _manager(installed_patch["factory"])
    original_minimize = app.Simulation.minimizeEnergy

    def fail_on_twin(simulation, *args, **kwargs):
        if simulation.context.getSystem().getNumForces() == 0:
            raise RuntimeError("synthetic EM failure")
        return original_minimize(simulation, *args, **kwargs)

    monkeypatch.setattr(app.Simulation, "minimizeEnergy", fail_on_twin)
    em_patch.install()
    system, wrapper = manager._build_window_system([], [], _periodic_box(), None)
    assert wrapper.residual_enabled is True
    assert installed_patch["factory_calls"] == ["adapter"]
    simulation = _simulation(manager.topology, system)
    with pytest.raises(RuntimeError, match="synthetic EM failure"):
        simulation.minimizeEnergy(maxIterations=5)

    # ⚠️ [2026-09-10] 异常路径同样**不再**清 stash（见上）。异常是从 twin 的
    # minimizeEnergy 里抛出来的，不是"twin 不该用"的证据 —— 同一窗口后续的
    # 原生最小化仍然需要保护。不匹配的 System 由 `real_system_sha256` 那道门拦。
    assert em_patch._STASH.get("make_em_system") is not None
    # 但工厂必须已经还回去 —— 这才是本用例真正要防的泄漏。
    assert manager.residual_basis_force_factory is installed_patch["factory"]
    em_patch.uninstall()


def test_mismatched_system_fails_closed_and_does_clear_the_stash(
    installed_patch, monkeypatch
):
    """stash 不再随 EM 清空之后，那道 sha256 门就是唯一的防走错门机制。

    换一个**仍然含残差 CV** 的 System 去最小化 ⟹ 必须抛，且把 stash 清掉，
    绝不能拿上一个窗口的 twin 去最小化这一个。
    """
    manager = _manager(installed_patch["factory"])
    manager._build_window_system([], [], _periodic_box(), None)
    assert em_patch._STASH != {}

    # 另建一份**真正带残差 CV** 的、与登记时不同的 System。
    #
    # ⚠️ 这里必须用真的 `CustomCVForce("exp025_residual_basis")`，不能用 fixture
    # 那个 `CustomExternalForce` 替身：`_system_has_residual_cv()` 找的就是这个
    # CV 名字，替身过不了它，会走"System 变了但不含残差 ⟹ 原生 EM 是安全的"
    # 那条合法分支，测不到 fail-closed。（合成替身与真实形状不符——本仓反复的坑。）
    other = openmm.System()
    other.addParticle(12.0 * unit.amu)
    other.setDefaultPeriodicBoxVectors(*_periodic_box())
    other.addForce(_residual_cv_force())
    simulation = _simulation(_topology_one_atom(), other)

    with pytest.raises(RuntimeError, match="twin 与当前含残差"):
        simulation.minimizeEnergy(maxIterations=5)
    assert em_patch._STASH == {}, "fail-closed 必须同时清掉 stash"


def test_changed_system_without_the_residual_cv_is_left_alone(installed_patch):
    """反面：System 变了但**不含**残差 CV ⟹ 原生 EM 是安全的，不该抛也不该消耗 twin。

    端点/路径探针会故意从候选窗口的 common XML 派生出残差-free 的 System，
    它们的原生最小化必须放行。
    """
    manager = _manager(installed_patch["factory"])
    manager._build_window_system([], [], _periodic_box(), None)

    probe = openmm.System()
    probe.addParticle(12.0 * unit.amu)
    probe.setDefaultPeriodicBoxVectors(*_periodic_box())
    simulation = _simulation(_topology_one_atom(), probe)

    simulation.minimizeEnergy(maxIterations=5)
    assert em_patch._STASH.get("make_em_system") is not None, (
        "残差-free 探针把 twin 消耗掉了 —— 候选窗口自己的 EM 会失去保护"
    )

