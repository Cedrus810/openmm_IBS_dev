"""Executable regressions for the nine 2026-08-31 review findings."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("openmm")
import openmm
from openmm import app, unit

import abfe_core as core
import abfe_pipeline as pl
import abfe_preoptimizer as preopt
import runabfe
from test_top_level_sampling_cache_identity import _fake_pipeline, _system_and_topology
from test_boresch_attachment_leg import _toy_restraint
from test_membrane_observable_extractor import _build_membrane_trajectory, _extract

pytestmark = pytest.mark.cpu_only
SCHEMES = ["dual_lambda", "single_lambda", "2d_diagonal", "2d_geodesic"]
ATTACHMENT = {"converged": True, "attachment_delta_G_kJ_mol": 2.0,
              "attachment_error_kJ_mol": 0.3}


def _pipe(tmp_path, scheme):
    pipe = _fake_pipeline(tmp_path, scheme, None)
    pipe._ligand_conformer_diagnostics = lambda: None
    pipe._commit_ensemble_boresch_equilibrium = lambda params, path: params
    pipe._stage_protocol_key = lambda *a, **kw: {}
    return pipe


def test_remd_model_upgrade_invalidates_trajectory_identity(monkeypatch):
    system, topology = _system_and_topology()
    kwargs = dict(stage_name="decharging", system=system, topology=topology,
                  ligand_indices=[0], lambdas_coul=[1, 0], lambdas_vdw=[1, 1],
                  temperature_K=300, n_steps=10000, exchange_interval=100,
                  boresch_params=None, potential_type="softcore", platform_name="Reference")
    before = pl._remd_sampling_fingerprint(**kwargs)
    monkeypatch.setattr(pl, "PME_DECHARGE_MODEL_VERSION", "next-model")
    assert pl._remd_sampling_fingerprint(**kwargs) != before


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("stage0", [None, dict(ATTACHMENT, converged=False),
                                    {k: v for k, v in ATTACHMENT.items() if k != "converged"}])
def test_final_cycle_rejects_missing_or_unconverged_attachment(tmp_path, scheme, stage0):
    pipe = _pipe(tmp_path, scheme)
    sampling = {"total_delta_G": 10., "total_error": 0.4, "stage0": stage0}
    with pytest.raises(RuntimeError, match="attachment"):
        pl.ABFEPipeline.compute_final_results(
            pipe, sampling, {"uses_analytical_release_formula": True},
            system=pipe.system, decoupling_scheme=scheme,
        )
    assert not (tmp_path / "output" / "final_results.json").exists()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_final_cycle_adds_attachment_and_its_uncertainty_once(tmp_path, monkeypatch, scheme):
    pipe = _pipe(tmp_path, scheme)
    monkeypatch.setattr(pl, "_collect_pipeline_provenance", lambda **kw: {})
    sampling = {"stage0": ATTACHMENT, "total_delta_G": 10., "total_error": 0.4,
                "stage1": {"total_delta_G": 4., "total_error": 0.0},
                "stage2": {"total_delta_G": 6., "total_error": 0.4}}
    result = pl.ABFEPipeline.compute_final_results(
        pipe, sampling, {"uses_analytical_release_formula": True, "delta_g_rest": -1.},
        system=pipe.system, decoupling_scheme=scheme,
    )
    assert result["decoupling_delta_G_kJ_mol"] == pytest.approx(12.)
    assert result["total_delta_G_complex_kJ_mol"] == pytest.approx(11.)
    assert result["total_error_kJ_mol"] == pytest.approx(0.5)
    core.validate_final_leg_result(result)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_each_pipeline_branch_runs_attachment_and_rejects_unconverged_result(tmp_path, monkeypatch, scheme):
    pipe = _pipe(tmp_path, scheme)
    calls = []
    def attachment(*a, **kw):
        calls.append(kw)
        return dict(ATTACHMENT, converged=False)
    monkeypatch.setattr(pl, "run_boresch_attachment_leg", attachment)
    with pytest.raises(RuntimeError, match="converged"):
        pipe.run_full_pipeline(decoupling_scheme=scheme, boresch_params=_toy_restraint(),
                               run_equilibration=False, n_states_per_stage=2)
    assert len(calls) == 1
    assert pipe.stage_calls == []
    assert not (tmp_path / "output" / "checkpoints" / "stage0_attachment.json").exists()
    assert not (tmp_path / "output" / "final_results.json").exists()


@pytest.mark.parametrize("scheme", SCHEMES[1:])
def test_single_and_2d_reuse_valid_attachment_but_rerun_unconverged_cache(tmp_path, monkeypatch, scheme):
    calls = []
    monkeypatch.setattr(pl, "run_boresch_attachment_leg",
                        lambda *a, **kw: calls.append(kw) or dict(ATTACHMENT))
    monkeypatch.setattr(preopt, "optimize_2d_geodesic_path", lambda **kw: [(1., 1.), (0., 0.)])
    for index in range(3):
        pipe = _pipe(tmp_path, scheme)
        def finish(sampling, correction, **kw):
            assert sampling["stage0"] == ATTACHMENT
            return {"test_stage0": sampling["stage0"]}
        pipe.compute_final_results = finish
        pipe.run_full_pipeline(decoupling_scheme=scheme, boresch_params=_toy_restraint(),
                               run_equilibration=False, n_states_per_stage=2,
                               resume=index > 0)
        assert len(calls) == (1 if index < 2 else 2)
        if index == 1:
            path = tmp_path / "output" / "checkpoints" / "stage0_attachment.json"
            doc = json.loads(path.read_text())
            doc["result"]["converged"] = False
            path.write_text(json.dumps(doc))


@pytest.mark.parametrize("shift", [0., 5., 11.])
def test_membrane_water_and_ion_geometry_is_translation_invariant(shift):
    traj, coion, _ = _build_membrane_trajectory(n_core_waters=3, coion_z=10.5)
    traj.xyz[:, :, 2] = (traj.xyz[:, :, 2] + shift) % 12.
    before = traj.xyz.copy()
    obs, diag = _extract(traj, coion_atom_index=coion)
    assert diag["anomalous_pocket_water_count"] == 3
    assert obs["coion_abs_z_from_midplane_nm"]["values"] == pytest.approx([4.5] * len(traj))
    np.testing.assert_array_equal(traj.xyz, before)


@pytest.mark.parametrize("distance", [0.35, 0.45, 0.65])
def test_probe_14_energy_and_force_match_original_openmm(distance):
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(4, 0, 0), openmm.Vec3(0, 4, 0), openmm.Vec3(0, 0, 4))
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(nb.CutoffPeriodic)
    nb.setCutoffDistance(1.0)
    nb.setUseDispersionCorrection(False)
    # Pair 0-1 is ligand 1-4; pair 2-3 verifies environment exceptions survive.
    for _ in range(4):
        system.addParticle(12.)
        nb.addParticle(0., 0.3, 0.)
    nb.addException(0, 1, -0.02, 0.3, 0.2)
    nb.addException(2, 3, 0.01, 0.3, 0.1)
    system.addForce(nb)
    softcore = core.ACESoftcorePotential.from_dict(core.ACESoftcorePotential.optimize_alpha(2))
    probe = preopt.build_aces_probe_system_dual_lambda(system, [0, 1], softcore, fixed_lam_coul=0.5, fixed_lam_vdw=1.0)
    def evaluate(sys):
        integrator = openmm.VerletIntegrator(0.001)
        ctx = openmm.Context(sys, integrator, openmm.Platform.getPlatformByName("Reference"))
        ctx.setPositions([[0.5, 0.5, 0.5], [0.5 + distance, 0.5, 0.5], [2., 2., 2.], [2.5, 2., 2.]])
        state = ctx.getState(getEnergy=True, getForces=True)
        return (state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer))
    expected_e, expected_f = evaluate(system)
    actual_e, actual_f = evaluate(probe)
    assert actual_e == pytest.approx(expected_e, rel=1e-7)
    np.testing.assert_allclose(actual_f, expected_f, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize("failed_stage", ["decharging", "vanishing"])
@pytest.mark.parametrize("converged", [False, None])
def test_traditional_does_not_write_unconverged_final(tmp_path, failed_stage, converged):
    pipe = pl.TraditionalABFEPipeline.__new__(pl.TraditionalABFEPipeline)
    pipe.output_dir = str(tmp_path)
    pipe.run_leg = lambda stage, *a, **kw: {
        "delta_G": 1., "error": 0.1,
        "converged": converged if stage == failed_stage else True}
    with pytest.raises(RuntimeError, match="converged"):
        pipe.run_full(n_lambda=2, n_steps_per_leg=10)
    assert not (tmp_path / "final_results.json").exists()


def test_final_gate_cannot_hide_failed_traditional_stage():
    result = {"delta_G_total_kJ_mol": 10., "error_leg_kJ_mol": 1., "converged": True,
              "stage_decharging": {"converged": False}, "stage_vanishing": {"converged": True}}
    with pytest.raises(core.FinalResultValidationError, match="stage_decharging"):
        core.validate_final_leg_result(result)


def test_inner_equilibration_cannot_override_outer_budget_rejection(tmp_path):
    pipe = _pipe(tmp_path, "single_lambda")
    out = tmp_path / "output"
    (out / "pre_equilibration.dcd").write_bytes(b"x" * 20000)
    (out / "checkpoints" / "pre_equil.chk").write_bytes(b"x" * 1024)
    (out / "pre_equilibration_fingerprint.json").write_text(json.dumps({"fingerprint": "old", "n_steps": 10000}))
    pipe._load_pipeline_state = lambda: {"stages": {"equilibration": {"status": "completed"}}}
    calls = []
    class ReachedStrictConsumer(Exception):
        pass
    def pre_equilibrate(**kw):
        calls.append(kw)
        raise ReachedStrictConsumer
    pipe.pre_equilibrate = pre_equilibrate
    with pytest.raises(ReachedStrictConsumer):
        pipe.run_full_pipeline(decoupling_scheme="single_lambda", run_equilibration=True,
                               n_equil_steps=20000, resume=True)
    assert calls[0]["n_steps"] == 20000
    assert calls[0]["resume"] is True


def test_checkpoint_probe_uses_real_platform_and_matching_equilibration_context(tmp_path):
    system, topology = _system_and_topology()
    nb = openmm.NonbondedForce()
    nb.addParticle(0., 0.3, 0.)
    nb.setNonbondedMethod(nb.CutoffPeriodic)
    system.addForce(nb)
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(3, 0, 0), openmm.Vec3(0, 3, 0), openmm.Vec3(0, 0, 3))
    protocol = core.resolve_membrane_protocol("soluble")
    pipe = SimpleNamespace(system=system, topology=topology, temperature=300*unit.kelvin,
                           pressure=1*unit.bar, platform_name="Reference", barostat_protocol=protocol)
    consumer_system = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(system))
    core.ensure_barostat_for_protocol(consumer_system, protocol, temperature=pipe.temperature, pressure=pipe.pressure)
    consumer = app.Simulation(topology, consumer_system,
        openmm.LangevinMiddleIntegrator(pipe.temperature, 1/unit.picosecond,
                                       core.PRE_EQUILIBRATION_TIMESTEP_PS*unit.picosecond),
        openmm.Platform.getPlatformByName("Reference"))
    consumer.context.setPositions([[1., 1., 1.]])
    consumer.step(2)
    path = tmp_path / "real.chk"
    consumer.saveCheckpoint(str(path))
    probe = runabfe._checkpoint_probe_simulation(pipe)
    assert probe is not None
    assert pl._is_checkpoint_valid(str(path), simulation=probe)
    path.write_bytes(b"garbage checkpoint")
    assert not pl._is_checkpoint_valid(str(path), simulation=probe)


@pytest.mark.parametrize("overlap", [False, True])
def test_analyze_only_applies_existing_conformer_gate(tmp_path, overlap):
    identity = {"comparison_sha256": "same-constraints"}
    for directory, interval in [(tmp_path, [1.3, 1.4]),
                                 (tmp_path / "solvent_leg", [1.2, 1.4] if overlap else [0.6, 0.7])]:
        directory.mkdir(exist_ok=True)
        result = {"total_delta_G_complex_kJ_mol": 10., "total_error_kJ_mol": 1.,
                  "constraint_identity": identity,
                  "ligand_conformer_diagnostics": core.ligand_conformer_summary(
                      {"max_internal_heavy_distance_nm": np.linspace(*interval, 100)}, leg="test")}
        (directory / "final_results.json").write_text(json.dumps(result))
    args = SimpleNamespace(output=str(tmp_path), temperature=300., mode="ibs", decoupling="dual_lambda")
    args.get = lambda key, default=None: getattr(args, key, default)
    if overlap:
        runabfe.run_post_analysis(args)
        result = json.loads((tmp_path / "final_results_postprocess.json").read_text())
        assert result["thermodynamic_cycle_terms"]["ligand_conformer_cross_leg"]["evaluated"]
    else:
        with pytest.raises(ValueError, match="构象"):
            runabfe.run_post_analysis(args)
        assert not (tmp_path / "final_results_postprocess.json").exists()
