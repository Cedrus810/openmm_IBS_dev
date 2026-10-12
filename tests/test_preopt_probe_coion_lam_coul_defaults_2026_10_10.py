"""带电腿 vanishing pilot 建 Context 时 lam_coul 默认值冲突（2026-10-10）。

真机症状（fragment-opt `MCL-1_ligand-1/rep1`，配体 −1 e，charge-transfer）：

    Two Forces define different default values for the parameter 'lam_coul'

B3 的 couple-intramol=no 补偿项（(1-λ²) 配体内部 ≥1-5 库仑）也注册 lam_coul、
默认 1.0；probe 只把 NonbondedForce 的默认值改成 fixed_lam_coul=0.0（生产
vanishing 传的就是 0.0）。旧测试用单原子配体，没有内部对 ⟹ 补偿项不存在，
所以没暴露。这里用两原子配体，并真的建 Context。

同时钉住：补偿项在 λ_coul=0 时取满，Group 2 的 ll_f 不能再带原始电荷
（否则配体内部库仑算两遍），与生产 vanishing 同口径。
"""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, Vec3, app, unit  # noqa: E402

import abfe_core as core  # noqa: E402
from abfe_pipeline import _create_context_with_local_cpu_fallback  # noqa: E402
from abfe_preoptimizer import build_aces_probe_system_dual_lambda  # noqa: E402
from ibs_engine import LIGAND_INTERNAL_COULOMB_FORCE_NAME  # noqa: E402

pytestmark = pytest.mark.cpu_only


def _system_and_spec():
    topology = app.Topology()
    topology.setPeriodicBoxVectors(
        (Vec3(4.0, 0.0, 0.0), Vec3(0.0, 4.0, 0.0), Vec3(0.0, 0.0, 4.0))
        * unit.nanometer
    )
    chain = topology.addChain()
    lig = topology.addResidue("LIG", chain)
    topology.addAtom("C1", app.element.carbon, lig)
    topology.addAtom("C2", app.element.carbon, lig)
    ion = topology.addResidue("CL", chain)
    topology.addAtom("CL", app.element.chlorine, ion)
    env = topology.addResidue("WAT", chain)
    topology.addAtom("O", app.element.oxygen, env)

    system = openmm.System()
    for mass in (12.0, 12.0, 35.45, 16.0):
        system.addParticle(mass * unit.dalton)
    system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    nb = NonbondedForce()
    nb.setNonbondedMethod(NonbondedForce.PME)
    nb.setCutoffDistance(1.0 * unit.nanometer)
    nb.addParticle(0.6, 0.34, 0.4)
    nb.addParticle(0.4, 0.34, 0.4)   # 两个配体原子之间是普通 ≥1-5 对
    nb.addParticle(0.0, 0.44, 0.2)   # reserved neutral ion-shaped dummy
    nb.addParticle(-1.0, 0.30, 0.5)
    system.addForce(nb)

    positions_nm = np.asarray(
        [[2.0, 2.0, 2.0], [2.5, 2.0, 2.0], [3.6, 3.6, 3.6], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    spec = core.build_co_alchemical_ion_identity(
        system=system, topology=topology, ion_atom_indices=[2],
        ligand_indices=[0, 1], positions_nm=positions_nm,
        box_vectors=np.eye(3, dtype=np.float64) * 4.0, ligand_net_charge_e=1,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        enforce_placement_thresholds=False,
    )
    return system, topology, positions_nm, spec


def _probe(fixed_lam_coul):
    system, topology, positions_nm, spec = _system_and_spec()
    softcore = core.ACESoftcorePotential.from_dict(
        core.ACESoftcorePotential.optimize_alpha(1))
    probe = build_aces_probe_system_dual_lambda(
        system, [0, 1], softcore,
        fixed_lam_coul=fixed_lam_coul, fixed_lam_vdw=1.0,
        topology=topology, positions=positions_nm * unit.nanometer,
        box_vectors=topology.getPeriodicBoxVectors(),
        co_alchemical_ion_spec=spec,
    )
    return probe, positions_nm


def _lam_coul_defaults(system):
    out = []
    for force in system.getForces():
        if not hasattr(force, "getNumGlobalParameters"):
            continue
        for i in range(force.getNumGlobalParameters()):
            if force.getGlobalParameterName(i) == "lam_coul":
                out.append((force.getName(),
                            force.getGlobalParameterDefaultValue(i)))
    return out


@pytest.mark.parametrize("fixed_lam_coul", [0.0, 0.5])
def test_every_lam_coul_default_agrees_and_context_builds(fixed_lam_coul):
    probe, positions_nm = _probe(fixed_lam_coul)
    defaults = _lam_coul_defaults(probe)
    names = [n for n, _ in defaults]
    assert LIGAND_INTERNAL_COULOMB_FORCE_NAME in names, "补偿项没挂上，测试失去意义"
    assert len(defaults) >= 3
    assert all(v == pytest.approx(fixed_lam_coul) for _, v in defaults), defaults

    integ = openmm.VerletIntegrator(0.001)
    ctx = openmm.Context(probe, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions_nm * unit.nanometer)
    assert np.isfinite(ctx.getState(getEnergy=True).getPotentialEnergy()
                       .value_in_unit(unit.kilojoule_per_mole))


def test_group2_ll_force_does_not_recount_internal_coulomb():
    probe, _ = _probe(0.0)
    ll_f = [f for f in probe.getForces()
            if isinstance(f, openmm.CustomNonbondedForce) and f.getForceGroup() == 2]
    assert len(ll_f) == 1
    for idx in (0, 1):
        assert ll_f[0].getParticleParameters(idx)[0] == pytest.approx(0.0)


def test_platform_independent_system_error_is_not_retried_on_cpu():
    """Context 定义错误不判成资源类、不回退 CPU。"""
    system = openmm.System()
    system.addParticle(1.0)
    for default in (0.0, 1.0):
        f = openmm.CustomExternalForce("lam_coul*x")
        f.addGlobalParameter("lam_coul", default)
        f.addParticle(0, [])
        system.addForce(f)
    logs = []
    with pytest.raises(RuntimeError, match="System 定义错误"):
        _create_context_with_local_cpu_fallback(
            system, lambda: openmm.VerletIntegrator(0.001), "Reference", logs.append)
    assert not any("回退到 CPU" in m for m in logs)
