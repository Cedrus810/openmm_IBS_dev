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

        assert em_patch._STASH == {}


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

    assert em_patch._STASH == {}
    assert manager.residual_basis_force_factory is installed_patch["factory"]
    em_patch.uninstall()
