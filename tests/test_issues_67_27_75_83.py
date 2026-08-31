"""Behavioral regressions for analysis recovery, migration, segments and real bonds."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import openmm
from openmm import app, unit
import abfe_pipeline as pl
import ibs_engine as ie
import runabfe
from production_artifact_fixtures import identify_window, segments
from test_preopt_protocol_scope import _fake_pipeline, _key

pytestmark = pytest.mark.cpu_only


def _write_json(path, value):
    Path(path).write_text(json.dumps(value))


def _read_json(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def recovery_leg(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    system = openmm.System()
    system.addParticle(12.)
    identity = pl.constraint_identity_fingerprint(system, [0])
    protocols = {}
    paths = {"decharging": ([1., 0.], [[0, 2]]), "vanishing": ([1., .5, 0.], [[0, 2], [1, 3]])}
    for name, (lambdas, ranges) in paths.items():
        payload = dict(pl._stage_analysis_protocol_versions(name), stage_name=name,
                       temperature_K=300., constraint_identity=identity,
                       decharge_method="pme" if name == "decharging" else "n/a",
                       sampling_repair_policy="non_mutating_v1", boresch_params=None,
                       # The public pipeline reads ibs_lse_log_residual_tolerance, not
                       # this similarly named (ignored) kwarg.
                       run_config={"n_steps_per_window": 1000, "kwargs": {"lse_log_residual_tolerance": .01}}, final_gate_thresholds={})
        protocols[name] = pl._protocol_fingerprint(payload)
        _write_json(checkpoints / f"preopt_dual_{name}.json", dict(lambdas_var=lambdas, window_ranges=ranges))
    dec = dict(stage="decharging", total_delta_G=2., total_error=.1, converged=True,
               min_overlap=.8, min_overlap_threshold=.03, diagnostics={})
    pl.ABFEPipeline._populate_stage_diagnostics(dec)
    dec_cache = pl.ABFEPipeline._build_stage_cache_payload("decharging", dec, 2, protocols["decharging"], *paths["decharging"])
    _write_json(checkpoints / "stage1_decharging.json", dec_cache)
    context = dict(stage_protocol_keys=protocols, constraint_identity=identity,
                   charge_treatment="neutral", charge_transfer_reservoir_correction=None,
                   co_alchemical_ion_runtime_identity=None,
                   ligand_conformer_diagnostics={"status": "fixture-evidence"})
    _write_json(checkpoints / "analysis_context.json", pl._protocol_fingerprint(context))
    stage = tmp_path / "vanishing"
    stage.mkdir()
    n = 400
    rng = np.random.default_rng(83)
    common = rng.normal(scale=.01, size=n)
    energy = np.array([common, common+.5])
    for name, array in (("energies", energy), ("bias", common), ("base", common*100)):
        np.save(stage / f"dual_window_0_vdw_{name}.npy", array)
    identify_window(stage, checkpoints, 0, [0., 0.], [1., .5], stage_key=protocols["vanishing"], f_k=[0., .5])
    # The endpoint bank has independent trajectories at both final lambda states.
    bank_dir = checkpoints / "endpoint_bank"
    bank_dir.mkdir()
    _write_json(bank_dir / "manifest.json", {"protocol_version": ie.INDEPENDENT_ENDPOINT_PROTOCOL_VERSION})
    records = {}
    for state in (1, 2):
        e = np.tile([0., .5, 1.], (n, 1)) + rng.normal(scale=.1, size=(n, 1))
        record = dict(u_cv_kj_mol=e, volume_nm3=np.ones(n), energy_column_indices=[0, 1, 2],
                      cavity_waters=np.zeros(n), segments=[dict(n_frames=n, reason="fresh")],
                      global_state=state, init_mode="dry", walker=0)
        records[str(state)] = record
        np.savez(bank_dir / f"state_{state}_dry_w0.npz", u_cv_kj_mol=e)
    kt = float((unit.MOLAR_GAS_CONSTANT_R*300*unit.kelvin).value_in_unit(unit.kilojoule_per_mole))
    solved = ie.solve_independent_endpoint_states(dict(state_indices=[1, 2], records=records), kt)
    assert solved["converged"] is True
    endpoint = dict(solve_all=solved, solve_wet={}, solve_dry=solved,
                    wet_dry_gate=ie.endpoint_wet_dry_hysteresis_gate({}, solved, {}, wet_arm_available=False),
                    diagnostics={"bank_dir": str(bank_dir), "lambda_indices": [1, 2]})
    pl._endpoint_analysis_artifact(str(checkpoints), endpoint, stage_protocol_key=protocols["vanishing"],
                                  lambda_path_fingerprint=pl.ABFEPipeline._lambda_path_fingerprint(*paths["vanishing"]))
    return tmp_path, context, paths


def test_raw_analysis_recovers_hybrid_result_and_identity_without_mutation(recovery_leg):
    base, context, _ = recovery_leg
    before = {str(p): p.read_bytes() for p in base.rglob("*") if p.is_file()}
    result = runabfe._analyze_dual_leg_artifacts(str(base), 300*unit.kelvin)
    assert result["decoupling_delta_G_kJ_mol"] == pytest.approx(3., abs=1e-7)
    assert result["constraint_identity"] == context["constraint_identity"]
    assert result["ligand_conformer_diagnostics"] == context["ligand_conformer_diagnostics"]
    assert result["stage_results"]["vanishing"]["converged"] is True
    assert before == {str(p): p.read_bytes() for p in base.rglob("*") if p.is_file()}


def test_cached_analysis_obeys_same_contract_as_raw(recovery_leg):
    base, context, paths = recovery_leg
    result = runabfe._analyze_dual_leg_artifacts(str(base), 300.)
    stage = result["stage_results"]["vanishing"]
    cache = pl.ABFEPipeline._build_stage_cache_payload("vanishing", stage, 3, context["stage_protocol_keys"]["vanishing"], *paths["vanishing"])
    _write_json(base / "checkpoints/stage2_vanishing.json", cache)
    cached = runabfe._analyze_dual_leg_artifacts(str(base), 300.)
    assert cached["decoupling_delta_G_kJ_mol"] == result["decoupling_delta_G_kJ_mol"]
    cache["coverage_diagnostics"] = None
    _write_json(base / "checkpoints/stage2_vanishing.json", cache)
    with pytest.raises(RuntimeError, match="coverage_diagnostics"):
        runabfe._analyze_dual_leg_artifacts(str(base), 300.)


@pytest.mark.parametrize("damage", ["protocol", "path", "converged", "missing_window", "lambda", "triplet", "frozen", "production_manifest", "segments", "endpoint", "temperature"])
def test_analysis_rejects_invalid_provenance_and_incomplete_data(recovery_leg, damage):
    base, context, paths = recovery_leg
    checkpoints = base / "checkpoints"
    if damage in {"protocol", "path", "converged"}:
        file = checkpoints / "stage1_decharging.json"
        doc = _read_json(file)
        if damage == "protocol":
            doc["protocol_key"] = pl._protocol_fingerprint({"unrelated": "Hamiltonian"})
        elif damage == "path":
            doc["lambda_path_fingerprint"] = pl.ABFEPipeline._lambda_path_fingerprint([1., .2], [[0, 2]])
        else:
            doc["converged"] = False
        _write_json(file, doc)
    elif damage == "missing_window":
        (base / "vanishing/dual_window_0_vdw_energies.npy").unlink()
    elif damage in {"lambda", "segments"}:
        file = base / "vanishing/dual_window_0_vdw_convergence.json"
        doc = _read_json(file)
        doc["lambdas_vdw" if damage == "lambda" else "production_segments"] = []
        _write_json(file, doc)
    elif damage == "triplet":
        np.save(base / "vanishing/dual_window_0_vdw_bias.npy", np.ones(400))
    elif damage == "frozen":
        file = checkpoints / "ibs_state_vdw_window_0.json"
        doc = _read_json(file); doc["f_k"][1] += 1
        _write_json(file, doc)
    elif damage == "production_manifest":
        _, file = ie._production_window_checkpoint_paths(str(checkpoints), "vdw", 0)
        doc = _read_json(file); doc["window_idx"] = 1
        _write_json(file, doc)
    elif damage == "endpoint":
        (checkpoints / "endpoint_bank/state_2_dry_w0.npz").unlink()
    with pytest.raises((RuntimeError, ValueError, FileNotFoundError)):
        runabfe._analyze_dual_leg_artifacts(str(base), 310. if damage == "temperature" else 300.)


@pytest.mark.parametrize("feature_value, compatible", [(None, True), ({"enabled": True}, False)])
def test_missing_inactive_field_migrates_but_never_enables_feature(feature_value, compatible):
    fresh = _key(_fake_pipeline())
    old_payload = copy.deepcopy(fresh["payload"])
    old_payload.pop("charge_transfer_reservoir_correction")
    old = pl._protocol_fingerprint(old_payload)
    fresh_payload = copy.deepcopy(fresh["payload"])
    fresh_payload["charge_transfer_reservoir_correction"] = feature_value
    fresh = pl._protocol_fingerprint(fresh_payload)
    assert pl.ABFEPipeline._preopt_cache_matches_ignoring_code_hash(old, fresh) is compatible
    if compatible:
        migrated = pl._protocol_fingerprint(fresh["payload"])
        assert migrated == fresh
        assert pl.ABFEPipeline._preopt_cache_matches_ignoring_code_hash(migrated, fresh)
    else:
        assert old != fresh


def test_missing_hamiltonian_version_never_inherits_current_version():
    fresh = _key(_fake_pipeline())
    old_payload = copy.deepcopy(fresh["payload"])
    old_payload.pop("preopt_hamiltonian_protocol_version")
    assert not pl.ABFEPipeline._preopt_cache_matches_ignoring_code_hash(pl._protocol_fingerprint(old_payload), fresh)


def test_segmented_autocorrelation_never_crosses_restart_boundary():
    rng = np.random.default_rng(75)
    a = rng.normal(size=(2, 200))
    b = rng.normal(size=(2, 300)) + 1000
    u = np.concatenate([a, b], axis=1)
    expected_a, g_a, _ = ie._decorrelate_by_worst_target_state(a, np.zeros(200), 2.5)
    expected_b, g_b, _ = ie._decorrelate_by_worst_target_state(b, np.zeros(300), 2.5)
    diagnostics = []
    observed, g, _ = ie._decorrelate_by_worst_target_state(u, np.zeros(500), 2.5, segments(500, 200), diagnostics)
    np.testing.assert_array_equal(observed, np.r_[expected_a, expected_b+200])
    assert g == max(g_a, g_b)
    assert [d["n_decorrelated"] for d in diagnostics] == [len(expected_a), len(expected_b)]
    assert [d["start_frame"] for d in diagnostics] == [0, 200]


def test_resume_and_rollback_segments_are_persistable_and_truncated():
    sampler = SimpleNamespace(energy_history=[], bias_history=[], base_energy_history=[])
    ie._start_production_segment(sampler, "fresh")
    for key in ("energy_history", "bias_history", "base_energy_history"):
        setattr(sampler, key, [0.] * 40)
    saved = json.loads(json.dumps(ie._production_segments_snapshot(sampler)))
    # A second process restores the persisted histories and starts a new segment.
    sampler.production_segment_starts = [dict(s) for s in saved]
    ie._start_production_segment(sampler, "cross_process_resume")
    for key in ("energy_history", "bias_history", "base_energy_history"):
        getattr(sampler, key).extend([1.] * 40)
    ie._truncate_production_history(sampler, 60)
    ie._start_production_segment(sampler, "catastrophe_rollback_rebuild")
    for key in ("energy_history", "bias_history", "base_energy_history"):
        getattr(sampler, key).extend([2.] * 30)
    result = ie._production_segments_snapshot(sampler)
    assert [(s["start_frame"], s["end_frame"]) for s in result] == [(0, 40), (40, 60), (60, 90)]
    assert result[-1]["reason"] == "catastrophe_rollback_rebuild"


def test_split_half_rebases_segment_boundaries_and_short_fragments_are_not_counted():
    window = dict(u_kn=np.zeros((2, 100)), bias_energies=np.zeros(100), base_energies=np.zeros(100), production_segments=segments(100, 60))
    sliced = ie._slice_window_frames(window, .5, 1.)
    assert [(s["start_frame"], s["end_frame"]) for s in sliced["production_segments"]] == [(0, 10), (10, 50)]
    idx, _, _ = ie._decorrelate_by_worst_target_state(sliced["u_kn"], sliced["bias_energies"], 2.5, sliced["production_segments"])
    assert np.all(idx >= 10)


def _real_bond_pipeline():
    topology = app.Topology()
    residue = topology.addResidue("LIG", topology.addChain())
    atoms = [topology.addAtom(f"C{i+1}", app.element.carbon, residue) for i in range(4)]
    topology.addBond(atoms[0], atoms[3])  # A wrong distance-inferred mmCIF link.
    system = openmm.System()
    for _ in atoms:
        system.addParticle(12.)
    force = openmm.HarmonicBondForce()
    force.addBond(0, 1, .15, 100.)
    force.addBond(1, 2, .15, 100.)
    system.addForce(force)
    system.addConstraint(2, 3, .15)
    return SimpleNamespace(topology=topology, system=system, ligand_indices=[0, 1, 2, 3], box_vectors=None)


def test_boresch_uses_real_gromacs_bonds_without_geometric_guess(tmp_path):
    top_file = tmp_path / "ligand.top"
    top_file.write_text('''[ defaults ]
1 2 yes 0.5 0.833333
[ atomtypes ]
C 6 12.01 0 A 0.3 0.1
[ moleculetype ]
LIG 3
[ atoms ]
1 C 1 LIG C1 1 0 12.01
2 C 1 LIG C2 1 0 12.01
3 C 1 LIG C3 1 0 12.01
4 C 1 LIG C4 1 0 12.01
[ bonds ]
1 2 1 0.15 100
2 3 1 0.15 100
3 4 1 0.15 100
[ system ]
ligand
[ molecules ]
LIG 1
''')
    pipe = _real_bond_pipeline()
    mdtop = runabfe._boresch_mdtraj_topology(pipe, str(top_file))
    assert {tuple(sorted((a.index, b.index))) for a, b in mdtop.bonds} == {(0, 1), (1, 2), (2, 3)}
    assert {tuple(sorted((a.index, b.index))) for a, b in pipe.topology.bonds()} == {(0, 3)}
    assert [a.name for a in mdtop.atoms] == ["C1", "C2", "C3", "C4"]


def test_cache_only_boresch_uses_system_bonds_and_constraints():
    mdtop = runabfe._boresch_mdtraj_topology(_real_bond_pipeline())
    assert {tuple(sorted((a.index, b.index))) for a, b in mdtop.bonds} == {(0, 1), (1, 2), (2, 3)}


def test_boresch_fails_when_no_authoritative_bonds_exist():
    pipe = _real_bond_pipeline()
    pipe.system.removeForce(0)
    pipe.system.removeConstraint(0)
    with pytest.raises(ValueError, match="真实配体键"):
        runabfe._boresch_mdtraj_topology(pipe)


def test_global_mbar_reports_each_restart_segment_and_base_jump(recovery_leg):
    base, _, paths = recovery_leg
    lambdas, ranges = paths["vanishing"]
    windows = ie.load_ibs_window_outputs_from_dir(
        str(base/"vanishing"), ranges, [0.]*3, lambdas,
        checkpoint_dir=str(base/"checkpoints"), excluded_local_windows={1})
    windows[0]["production_segments"] = segments(400, 200)
    windows[0]["base_energies"][200:] += 1000.
    expected_jump = windows[0]["base_energies"][200] - windows[0]["base_energies"][199]
    result = ie.solve_stage_integrated(windows, 2.4943387854)
    assert result["converged"] is True
    report = result["window_overlap_diagnostics"][0]["production_segments"]
    assert len(report) == 2
    assert sum(s["n_decorrelated"] for s in report) == result["window_overlap_diagnostics"][0]["n_frames_decorrelated"]
    assert report[1]["base_energy_jump_kJ_mol"] == pytest.approx(expected_jump)
    assert result["split_half_diagnostics"]["available"] is True


def test_production_resume_rejects_unknown_boundaries_or_wrong_parent_protocol():
    from test_resume_reuse_contracts import _matching_conv, _gate, LC_WIN, LV_WIN, GOOD_SHAPE, EARLY_STOP_CONFIG, TARGET_STEPS, REPAIR_POLICY, LSE_TOL
    conv = _matching_conv()
    assert _gate(conv)["usable"]
    conv.pop("production_segments")
    assert not _gate(conv)["segment_metadata_match"]
    status = ie._resume_cached_window_gate_status(
        _matching_conv(), GOOD_SHAPE, LC_WIN, LV_WIN, REPAIR_POLICY, LSE_TOL,
        False, EARLY_STOP_CONFIG, TARGET_STEPS, current_stage_protocol_key={"sha256": "different-system"})
    assert status["stage_protocol_match"] is False
    assert status["usable"] is False


def test_decharging_legacy_migration_uses_actual_fixed_pilot_settings():
    pipe = _fake_pipeline({"n_states_per_stage": 12})
    fresh = pipe._preopt_protocol_key("decharging", "softcore", None,
        requested_n_states=12, pilot_n_steps_per_state=10000, pilot_finite_difference_delta="n/a")
    old = copy.deepcopy(fresh["payload"])
    for key in ("requested_n_states", "pilot_n_steps_per_state", "pilot_finite_difference_delta", "charge_transfer_reservoir_correction", "thermodynamic_path_protocol_version"):
        old.pop(key)
    old["preopt_code_sha256"] = "unrelated-code-change"
    old["run_config"] = {"n_states_per_stage": 12, "kwargs": {"pilot_n_steps_per_state": 777, "pilot_finite_difference_delta": .03}}
    assert pl.ABFEPipeline._preopt_cache_matches_ignoring_code_hash(pl._protocol_fingerprint(old), fresh)
    old["preopt_hamiltonian_protocol_version"] -= 1
    assert not pl.ABFEPipeline._preopt_cache_matches_ignoring_code_hash(pl._protocol_fingerprint(old), fresh)


def test_analyze_only_complete_entry_accepts_recovered_leg_identities(recovery_leg, monkeypatch):
    import shutil
    import sys
    base, context, _ = recovery_leg
    solvent = base / "solvent_leg"
    solvent.mkdir()
    for directory in ("checkpoints", "vanishing"):
        shutil.copytree(base/directory, solvent/directory)
    class Args(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)
    args = Args(output=str(base), temperature=300., mode="ibs", decoupling="dual_lambda")
    monkeypatch.setattr(sys, "argv", ["runabfe.py", "--analyze-only"])
    runabfe.run_post_analysis(args)
    result = _read_json(base / "final_results_postprocess.json")
    assert result["delta_G_bind_kJ_mol"] == pytest.approx(0., abs=1e-7)
    assert result["constraint_identity"] == context["constraint_identity"]


def test_extending_budget_keeps_physical_continuation_but_not_completed_reuse(tmp_path):
    from test_resume_reuse_contracts import _matching_conv, LC_WIN, LV_WIN, GOOD_SHAPE, EARLY_STOP_CONFIG, TARGET_STEPS, REPAIR_POLICY, LSE_TOL
    old_payload = dict(system_xml_sha256="same-system", pme_decharge_model_version=1,
                       run_config=dict(n_steps_per_window=TARGET_STEPS, kwargs={}),
                       final_gate_thresholds={"final_min_absolute_ess": 50.})
    new_payload = copy.deepcopy(old_payload)
    new_payload["run_config"]["n_steps_per_window"] *= 2
    new_payload["run_config"]["kwargs"]["final_min_absolute_ess"] = 100.
    new_payload["final_gate_thresholds"]["final_min_absolute_ess"] = 100.
    old_key, new_key = map(pl._protocol_fingerprint, (old_payload, new_payload))
    conv = _matching_conv(stage_protocol_key=old_key)
    status = ie._resume_cached_window_gate_status(
        conv, GOOD_SHAPE, LC_WIN, LV_WIN, REPAIR_POLICY, LSE_TOL, False,
        EARLY_STOP_CONFIG, TARGET_STEPS*2, current_stage_protocol_key=new_key)
    assert status["stage_protocol_match"] is True
    assert status["early_stop_ok"] is False
    assert status["usable"] is False  # A short run is not already complete.
    manifest = ie._build_production_window_checkpoint_manifest("coul", 0, 3, "same-window-system", LC_WIN, LV_WIN, None, None, [0.]*3, 300., "Reference")
    manifest["stage_protocol_key"] = old_key
    checkpoint, metadata = ie._production_window_checkpoint_paths(str(tmp_path), "coul", 0)
    Path(checkpoint).parent.mkdir(parents=True)
    Path(checkpoint).write_bytes(b"existence-only; native loading is a separate gate")
    _write_json(metadata, manifest)
    expected = dict(manifest, stage_protocol_key=new_key)
    assert ie._production_window_checkpoint_is_usable(str(tmp_path), "coul", 0, expected)
    new_payload["system_xml_sha256"] = "different-Hamiltonian"
    expected["stage_protocol_key"] = pl._protocol_fingerprint(new_payload)
    assert not ie._production_window_checkpoint_is_usable(str(tmp_path), "coul", 0, expected)


def test_analysis_threshold_change_reuses_raw_frames_and_endpoint_bank(recovery_leg):
    base, context, _ = recovery_leg
    payload = copy.deepcopy(context["stage_protocol_keys"]["vanishing"]["payload"])
    payload["final_gate_thresholds"] = {"final_min_absolute_ess": 100.}
    payload["run_config"]["kwargs"]["final_min_absolute_ess"] = 100.
    context["stage_protocol_keys"]["vanishing"] = pl._protocol_fingerprint(payload)
    _write_json(base / "checkpoints/analysis_context.json", pl._protocol_fingerprint(context))
    result = runabfe._analyze_dual_leg_artifacts(str(base), 300.)
    assert result["stage_results"]["vanishing"]["converged"] is True
    assert result["decoupling_delta_G_kJ_mol"] == pytest.approx(3., abs=1e-7)
    payload["final_gate_thresholds"]["final_min_decorrelated_samples"] = 10000
    context["stage_protocol_keys"]["vanishing"] = pl._protocol_fingerprint(payload)
    _write_json(base / "checkpoints/analysis_context.json", pl._protocol_fingerprint(context))
    with pytest.raises(RuntimeError):
        runabfe._analyze_dual_leg_artifacts(str(base), 300.)
