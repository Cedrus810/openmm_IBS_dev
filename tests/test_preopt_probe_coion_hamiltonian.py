"""ACES pilot must consume the real B3 co-ion Hamiltonian."""

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import NonbondedForce, Vec3, app, unit  # noqa: E402

import abfe_core as core  # noqa: E402
from abfe_preoptimizer import build_aces_probe_system_dual_lambda  # noqa: E402


def _system_and_spec():
    topology = app.Topology()
    topology.setPeriodicBoxVectors(
        (
            Vec3(4.0, 0.0, 0.0),
            Vec3(0.0, 4.0, 0.0),
            Vec3(0.0, 0.0, 4.0),
        )
        * unit.nanometer
    )
    chain = topology.addChain()
    ligand_residue = topology.addResidue("LIG", chain)
    topology.addAtom("C", app.element.carbon, ligand_residue)
    ion_residue = topology.addResidue("CL", chain)
    topology.addAtom("CL", app.element.chlorine, ion_residue)
    env_residue = topology.addResidue("WAT", chain)
    topology.addAtom("O", app.element.oxygen, env_residue)

    system = openmm.System()
    for mass in (12.0, 35.45, 16.0):
        system.addParticle(mass * unit.dalton)
    system.setDefaultPeriodicBoxVectors(*topology.getPeriodicBoxVectors())
    nb = NonbondedForce()
    nb.addParticle(1.0, 0.34, 0.4)
    nb.addParticle(0.0, 0.44, 0.2)  # reserved neutral ion-shaped dummy
    nb.addParticle(-0.8, 0.30, 0.5)
    system.addForce(nb)

    positions_nm = np.asarray(
        [[2.0, 2.0, 2.0], [3.0, 2.0, 2.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    spec = core.build_co_alchemical_ion_identity(
        system=system,
        topology=topology,
        ion_atom_indices=[1],
        ligand_indices=[0],
        positions_nm=positions_nm,
        box_vectors=np.eye(3, dtype=np.float64) * 4.0,
        ligand_net_charge_e=1,
        charge_treatment=core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        enforce_placement_thresholds=False,
    )
    return system, topology, positions_nm, spec


def test_charge_transfer_probe_has_real_offsets_endpoint_charge_and_restraint():
    system, topology, positions_nm, spec = _system_and_spec()
    softcore = core.ACESoftcorePotential.from_dict(
        core.ACESoftcorePotential.optimize_alpha(1)
    )
    probe = build_aces_probe_system_dual_lambda(
        system,
        [0],
        softcore,
        fixed_lam_coul=0.5,
        fixed_lam_vdw=1.0,
        topology=topology,
        positions=positions_nm * unit.nanometer,
        box_vectors=topology.getPeriodicBoxVectors(),
        co_alchemical_ion_spec=spec,
    )

    nb = next(force for force in probe.getForces() if isinstance(force, NonbondedForce))
    assert nb.getForceGroup() == 1
    offsets = {
        int(nb.getParticleParameterOffset(i)[1]): float(
            nb.getParticleParameterOffset(i)[2]
        )
        for i in range(nb.getNumParticleParameterOffsets())
    }
    assert offsets == {0: pytest.approx(1.0), 1: pytest.approx(-1.0)}

    base = np.asarray(
        [
            nb.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
            for i in range(nb.getNumParticles())
        ],
        dtype=float,
    )
    for lam in (1.0, 0.5, 0.0):
        charges = base.copy()
        for i, scale in offsets.items():
            charges[i] += lam * scale
        assert charges.sum() == pytest.approx(base.sum())
        assert charges[0] == pytest.approx(lam)
        assert charges[1] == pytest.approx(1.0 - lam)

    restraints = [
        force
        for force in probe.getForces()
        if isinstance(force, openmm.CustomCompoundBondForce)
    ]
    assert len(restraints) == 1
    assert restraints[0].getForceGroup() == spec["ions"][0]["restraint"]["force_group"]
    assert restraints[0].getNumBonds() == 1

    # The ACES custom force must not carry a second ligand Coulomb term after
    # B3's native PME offsets were installed.
    custom_nb = [
        force
        for force in probe.getForces()
        if isinstance(force, openmm.CustomNonbondedForce)
        and force.getForceGroup() == 1
    ][0]
    assert custom_nb.getParticleParameters(0)[0] == pytest.approx(0.0)
