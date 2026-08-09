"""回归：single/2D 外层缓存不能绕过 co-ion 身份门。"""

import json

import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

import abfe_pipeline as pipeline  # noqa: E402


COION_A = {
    "schema_version": 1,
    "leg": "complex",
    "charge_treatment": "co_alchemical_charge_transfer",
    "fingerprint": "coion-a",
    "ion_atom_indices": [1],
}
COION_B = dict(COION_A, fingerprint="coion-b")


def _system_and_topology():
    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("LIG", chain)
    topology.addAtom("C", app.element.carbon, residue)
    system = openmm.System()
    system.addParticle(12.0 * unit.dalton)
    return system, topology


def _fake_pipeline(tmp_path, scheme, coion, *, completed=False):
    system, topology = _system_and_topology()
    output_dir = tmp_path / "output"
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    pipe = pipeline.ABFEPipeline.__new__(pipeline.ABFEPipeline)
    pipe.output_dir = str(output_dir)
    pipe.checkpoint_dir = str(checkpoint_dir)
    pipe.log_file = str(output_dir / "pipeline.log")
    pipe.system = system
    pipe.topology = topology
    pipe.positions = [[0.0, 0.0, 0.0]] * unit.nanometer
    pipe.box_vectors = None
    pipe.ligand_indices = [0]
    pipe.temperature = 300.0 * unit.kelvin
    pipe.pressure = 1.0 * unit.bar
    pipe.platform_name = "CPU"
    pipe.results = {}
    pipe._pre_equilibration_done_this_process = False
    pipe._boresch_rebalance_done_this_process = False
    pipe.logs = []
    pipe.stage_calls = []
    pipe._log = lambda message: pipe.logs.append(str(message))
    pipe.ensure_membrane_quality_gate_passed = lambda: None
    pipe.get_device_strategy = lambda **kwargs: {"strategy": "cpu", "devices": []}
    pipe.repair_pbc_molecule_integrity = lambda **kwargs: None
    pipe.co_alchemical_ion_runtime_identity = lambda leg: coion
    pipe._load_pipeline_state = lambda: {
        "stages": (
            {f"sampling_{scheme}": {"status": "completed"}}
            if completed
            else {}
        )
    }
    pipe._update_stage_status = lambda *args, **kwargs: None
    pipe.apply_boresch_correction = lambda *args, **kwargs: {
        "delta_g_rest": 0.0,
        "error": 0.0,
    }
    pipe.compute_final_results = lambda sampling, correction, **kwargs: {
        "total_delta_G": sampling["total_delta_G"],
        "total_error": sampling["total_error"],
    }

    def _run_stage(**kwargs):
        pipe.stage_calls.append(kwargs)
        return {"total_delta_G": 1.0, "total_error": 0.1}

    pipe._run_2d_lambda_stage = _run_stage
    return pipe


@pytest.mark.parametrize("scheme", ["single_lambda", "2d_diagonal"])
def test_outer_sampling_cache_rejects_changed_coion_identity(tmp_path, scheme):
    first = _fake_pipeline(tmp_path, scheme, COION_A)
    first.run_full_pipeline(
        decoupling_scheme=scheme,
        n_states_per_stage=2,
        n_steps_per_window=10,
        steps_per_update=5,
        run_equilibration=False,
        resume=False,
    )

    second = _fake_pipeline(tmp_path, scheme, COION_B, completed=True)
    second.run_full_pipeline(
        decoupling_scheme=scheme,
        n_states_per_stage=2,
        n_steps_per_window=10,
        steps_per_update=5,
        run_equilibration=False,
        resume=True,
    )

    assert len(second.stage_calls) == 1
    sample_file = tmp_path / "output" / "checkpoints" / f"sampling_{scheme}.json"
    saved = json.loads(sample_file.read_text())
    assert saved["protocol_key"]["payload"]["coion_identity"] == COION_B


def test_geodesic_path_cache_reoptimizes_on_changed_coion_identity(tmp_path, monkeypatch):
    path_calls = []

    def _optimize(**kwargs):
        path_calls.append(kwargs)
        return [(1.0, 1.0), (0.0, 0.0)]

    monkeypatch.setattr(
        "abfe_preoptimizer.optimize_2d_geodesic_path", _optimize
    )

    first = _fake_pipeline(tmp_path, "2d_geodesic", COION_A)
    first.run_full_pipeline(
        decoupling_scheme="2d_geodesic",
        n_states_per_stage=2,
        n_steps_per_window=10,
        steps_per_update=5,
        run_equilibration=False,
        resume=False,
    )
    assert len(path_calls) == 1

    second = _fake_pipeline(tmp_path, "2d_geodesic", COION_B, completed=True)
    second.run_full_pipeline(
        decoupling_scheme="2d_geodesic",
        n_states_per_stage=2,
        n_steps_per_window=10,
        steps_per_update=5,
        run_equilibration=False,
        resume=True,
    )

    assert len(path_calls) == 2
    assert len(second.stage_calls) == 1
    path_file = tmp_path / "output" / "checkpoints" / "path_2d_geodesic.json"
    saved_path = json.loads(path_file.read_text())
    assert saved_path["protocol_key"]["payload"]["coion_identity"] == COION_B


def test_old_charged_geodesic_path_without_probe_version_reoptimizes(tmp_path, monkeypatch):
    path_calls = []

    def _optimize(**kwargs):
        path_calls.append(kwargs)
        return [(1.0, 1.0), (0.0, 0.0)]

    monkeypatch.setattr(
        "abfe_preoptimizer.optimize_2d_geodesic_path", _optimize
    )

    first = _fake_pipeline(tmp_path, "2d_geodesic", COION_A)
    first.run_full_pipeline(
        decoupling_scheme="2d_geodesic",
        n_states_per_stage=2,
        n_steps_per_window=10,
        steps_per_update=5,
        run_equilibration=False,
        resume=False,
    )
    assert len(path_calls) == 1

    path_file = tmp_path / "output" / "checkpoints" / "path_2d_geodesic.json"
    cached = json.loads(path_file.read_text())
    cached["protocol_key"]["payload"].pop(
        "coion_probe_hamiltonian_protocol_version"
    )
    path_file.write_text(json.dumps(cached))

    # Keep the same identity.  The only mismatch is the newly required charged
    # pilot-Hamiltonian version field, so the path must be optimized again.
    second = _fake_pipeline(tmp_path, "2d_geodesic", COION_A)
    second.run_full_pipeline(
        decoupling_scheme="2d_geodesic",
        n_states_per_stage=2,
        n_steps_per_window=10,
        steps_per_update=5,
        run_equilibration=False,
        resume=True,
    )
    assert len(path_calls) == 2
