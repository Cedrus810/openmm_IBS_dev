"""Tiny CPU dynamics smoke for GitHub issue #84 closure evidence.

The larger protocol suites validate routing, fingerprints, parameters and
fail-closed behavior.  This file supplies the remaining runtime seam: both
barostat variants must create a Reference Context and survive actual steps.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import app, unit

import abfe_core as core


def _periodic_toy(residue_name: str, n_side: int = 3):
    """Return a small, finite periodic system/topology/position tuple."""
    topology = app.Topology()
    chain = topology.addChain()
    system = openmm.System()
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nonbonded.setCutoffDistance(0.7 * unit.nanometer)

    positions = []
    spacing_nm = 0.65
    for ix in range(n_side):
        for iy in range(n_side):
            for iz in range(n_side):
                residue = topology.addResidue(residue_name, chain)
                topology.addAtom("C", app.element.carbon, residue)
                system.addParticle(12.0 * unit.dalton)
                # Neutral LJ particles are enough to exercise periodic forces
                # and barostat Context integration without PME setup overhead.
                nonbonded.addParticle(
                    0.0 * unit.elementary_charge,
                    0.30 * unit.nanometer,
                    0.10 * unit.kilojoule_per_mole,
                )
                positions.append(
                    openmm.Vec3(
                        0.45 + ix * spacing_nm,
                        0.45 + iy * spacing_nm,
                        0.45 + iz * spacing_nm,
                    )
                )

    system.addForce(nonbonded)
    box_nm = 3.0
    vectors = (
        openmm.Vec3(box_nm, 0.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, box_nm, 0.0) * unit.nanometer,
        openmm.Vec3(0.0, 0.0, box_nm) * unit.nanometer,
    )
    system.setDefaultPeriodicBoxVectors(*vectors)
    topology.setPeriodicBoxVectors(vectors)
    return system, topology, positions * unit.nanometer


@pytest.mark.parametrize(
    "system_type,residue_name,expected_barostat,n_steps",
    [
        ("membrane", "POPC", "MonteCarloMembraneBarostat", 3),
        ("soluble", "SOL", "MonteCarloBarostat", 30),
    ],
)
def test_barostat_protocol_survives_reference_cpu_dynamics(
    system_type, residue_name, expected_barostat, n_steps
):
    system, topology, positions = _periodic_toy(residue_name)
    membrane_config = {"barostat_frequency": 1} if system_type == "membrane" else None
    protocol = core.resolve_membrane_protocol(
        system_type,
        membrane_config=membrane_config,
        topology=topology,
    )
    result = core.ensure_barostat_for_protocol(
        system,
        protocol,
        temperature=300.0,
        pressure=1.0,
    )
    assert result["action"] == "added"
    assert [name for _, name in core.detect_barostats(system)] == [expected_barostat]

    integrator = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picoseconds,
    )
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    try:
        context.setPositions(positions)
        context.setVelocitiesToTemperature(300.0 * unit.kelvin, 20260813)
        integrator.step(n_steps)
        state = context.getState(
            getPositions=True,
            getEnergy=True,
            enforcePeriodicBox=True,
        )
        xyz = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)

        assert np.all(np.isfinite(xyz))
        assert np.all(np.isfinite(box))
        assert np.isfinite(potential)
        assert np.isfinite(kinetic)
        assert abs(float(np.linalg.det(box))) > 0.0
    finally:
        del context
        del integrator
