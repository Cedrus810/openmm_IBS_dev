"""Production contract tests for the pair-specific LJ-matched DEXP entrypoint."""

import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import XmlSerializer, unit

from dexp_NEW import (
    DEXPProductionConfig,
    LEGACY_FIT_KEYS,
    build_production_system,
)
from abfe_core import DEXPSurrogatePotential
from ibs_engine import _create_softcore_force


@pytest.mark.parametrize("legacy_key", sorted(LEGACY_FIT_KEYS))
def test_new_contract_rejects_every_legacy_fit_field(legacy_key):
    with pytest.raises(ValueError, match="legacy/global"):
        DEXPProductionConfig.from_mapping({legacy_key: 1.0})


def test_new_contract_contains_only_pair_specific_controls():
    params = DEXPProductionConfig().to_builder_params()
    assert params == {
        "alpha_vdw": 14.0,
        "beta_vdw": 5.0,
        "sigma_elec": 0.10,
        "switch_width": 0.20,
        "cutoff_distance": 0.70,
    }
    assert not set(params).intersection(LEGACY_FIT_KEYS)


@pytest.mark.parametrize(
    "legacy_params",
    [
        {"r0_vdw": 0.33},
        {"A_fit": 1.0},
        {"B_fit": 0.5},
        {"offset_c0": 0.0},
        {"offset_c1": 0.0},
    ],
)
def test_core_pair_specific_model_rejects_legacy_json(legacy_params):
    with pytest.raises(ValueError, match="旧版全局 DEXP"):
        DEXPSurrogatePotential.from_dict(legacy_params)


def _two_particle_periodic_system():
    system = openmm.System()
    system.addParticle(12.0 * unit.dalton)
    system.addParticle(16.0 * unit.dalton)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(3.0, 0.0, 0.0),
        openmm.Vec3(0.0, 3.0, 0.0),
        openmm.Vec3(0.0, 0.0, 3.0),
    )
    nonbonded = openmm.NonbondedForce()
    nonbonded.setNonbondedMethod(openmm.NonbondedForce.PME)
    nonbonded.setCutoffDistance(1.0 * unit.nanometer)
    # Keep this two-particle analytic test free of the native LJ dispersion
    # finite-size correction; the DEXP interaction itself is tested below.
    nonbonded.setUseDispersionCorrection(False)
    nonbonded.addParticle(0.0, 0.30, 1.0)
    nonbonded.addParticle(0.0, 0.30, 4.0)
    system.addForce(nonbonded)
    return system


def _energy(context):
    return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
        unit.kilojoule_per_mole
    )


def test_new_production_system_has_expected_pair_well_and_lambda_endpoint():
    original = _two_particle_periodic_system()
    produced = build_production_system(original, [0])

    # Builder must not mutate the native input System.
    original_xml = XmlSerializer.serialize(original)
    assert "lam_vdw" not in original_xml

    integrator = openmm.VerletIntegrator(0.001 * unit.picoseconds)
    context = openmm.Context(
        produced,
        integrator,
        openmm.Platform.getPlatformByName("Reference"),
    )
    r0 = (2.0 ** (1.0 / 6.0)) * 0.30
    context.setPositions(
        [
            openmm.Vec3(0.0, 0.0, 0.0),
            openmm.Vec3(r0, 0.0, 0.0),
        ]
        * unit.nanometer
    )
    context.setParameter("lam_coul", 0.0)
    context.setParameter("lam_vdw", 1.0)
    assert _energy(context) == pytest.approx(-2.0, abs=1.0e-6)

    context.setParameter("lam_vdw", 0.0)
    assert _energy(context) == pytest.approx(0.0, abs=1.0e-8)
    del context, integrator


def test_new_production_rejects_box_smaller_than_twice_cutoff():
    system = _two_particle_periodic_system()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(1.0, 0.0, 0.0),
        openmm.Vec3(0.0, 1.0, 0.0),
        openmm.Vec3(0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="Minimum-image violation"):
        build_production_system(system, [0])


def test_ibs_dexp_uses_dexp_cutoff_and_switch_not_softcore_defaults():
    system = _two_particle_periodic_system()
    nonbonded = next(
        force
        for force in system.getForces()
        if isinstance(force, openmm.NonbondedForce)
    )
    force = _create_softcore_force(
        nonbonded,
        perturbed_indices=[0],
        environment_indices=[1],
        lam_coul=0.0,
        lam_vdw=1.0,
        alchemical_params=DEXPProductionConfig().to_builder_params(),
        potential_type="dexp",
    )
    assert force.getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(0.70)
    assert force.getSwitchingDistance().value_in_unit(unit.nanometer) == pytest.approx(0.50)
