"""Production contract tests for the pair-specific LJ-matched DEXP entrypoint."""

import pytest

pytestmark = pytest.mark.cpu_only

openmm = pytest.importorskip("openmm")
from openmm import XmlSerializer, unit

from abfe_core import (
    DEXP_LEGACY_FIT_KEYS,
    DEXP_VDW_CUTOFF_NM,
    DEXP_VDW_SWITCH_WIDTH_NM,
    DEXPSurrogatePotential,
    SurrogateSystemBuilder,
    _validate_minimum_image,
)
from ibs_engine import _create_softcore_force
from runabfe import _load_dexp_params_fail_closed


@pytest.mark.parametrize("legacy_key", sorted(DEXP_LEGACY_FIT_KEYS))
def test_new_contract_rejects_every_legacy_fit_field(legacy_key):
    with pytest.raises(ValueError, match="旧版全局 DEXP"):
        DEXPSurrogatePotential.from_dict({legacy_key: 1.0})


def test_new_contract_contains_only_pair_specific_controls():
    params = DEXPSurrogatePotential().get_parameters_dict()
    assert params == {"alpha_vdw": 14.0, "beta_vdw": 5.0}
    assert not set(params).intersection(DEXP_LEGACY_FIT_KEYS)


def test_new_contract_rejects_unknown_fields():
    with pytest.raises(ValueError, match="未知 DEXP"):
        DEXPSurrogatePotential.from_dict({"not_a_dexp_parameter": 1.0})


def test_dexp_cli_contract_requires_explicit_parameter_file():
    with pytest.raises(ValueError, match="必须显式提供"):
        _load_dexp_params_fail_closed("dexp", None)


def test_dexp_cli_contract_rejects_retired_fit_payload(tmp_path):
    old_params = tmp_path / "dexp_fitted_params.json"
    old_params.write_text('{"fitting_success": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="旧版全局 DEXP"):
        _load_dexp_params_fail_closed("dexp", str(old_params))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_new_contract_rejects_nonfinite_shape_parameters(value):
    with pytest.raises(ValueError, match="有限"):
        DEXPSurrogatePotential(alpha_vdw=value, beta_vdw=5.0)


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
    produced = SurrogateSystemBuilder({}).build_surrogate_system(
        original,
        ligand_indices=[0],
        environment_indices=[1],
    )

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
    energy_r0 = _energy(context)
    assert energy_r0 == pytest.approx(-2.0, abs=1.0e-6)

    h = 1.0e-4
    displaced_energies = []
    for r in (r0 - h, r0 + h):
        context.setPositions(
            [
                openmm.Vec3(0.0, 0.0, 0.0),
                openmm.Vec3(r, 0.0, 0.0),
            ]
            * unit.nanometer
        )
        displaced_energies.append(_energy(context))
    curvature = (displaced_energies[0] - 2.0 * energy_r0 + displaced_energies[1]) / h**2
    assert curvature == pytest.approx(14.0 * 5.0 * 2.0 / r0**2, rel=2.0e-5)

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
        _validate_minimum_image(
            system.getDefaultPeriodicBoxVectors(),
            DEXP_VDW_CUTOFF_NM,
        )


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
        alchemical_params=DEXPSurrogatePotential().get_parameters_dict(),
        potential_type="dexp",
    )
    assert force.getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(
        DEXP_VDW_CUTOFF_NM
    )
    assert force.getSwitchingDistance().value_in_unit(unit.nanometer) == pytest.approx(
        DEXP_VDW_CUTOFF_NM - DEXP_VDW_SWITCH_WIDTH_NM
    )
