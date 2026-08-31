import inspect
import json

from openmm import unit

from abfe_pipeline import ABFEPipeline


def test_exp019_commits_ensemble_mean_and_only_records_last_frame_diagnostic(tmp_path):
    pipeline = ABFEPipeline.__new__(ABFEPipeline)
    pipeline.checkpoint_dir = str(tmp_path)
    pipeline.temperature = 300.0 * unit.kelvin
    pipeline._log = lambda *args, **kwargs: None
    diagnostic = {
        "status": "DIAGNOSTIC_ONLY",
        "max_deviation_sigma": 0.84,
        "reanchor_applied": False,
    }
    pipeline._diagnose_boresch_last_frame = lambda params: diagnostic

    equilibrium = {
        "r0": 0.61,
        "thetaA0": 1.2,
        "thetaB0": 1.4,
        "phiA0": -0.7,
        "phiB0": 0.4,
        "phiC0": 2.1,
    }
    boresch_simple_path = tmp_path / "boresch_simple.json"
    boresch_simple_path.write_text(
        json.dumps({"equilibrium_values": equilibrium}, indent=2) + "\n",
        encoding="utf-8",
    )
    boresch_simple = json.loads(boresch_simple_path.read_text(encoding="utf-8"))
    params = {
        "equilibrium_values": boresch_simple["equilibrium_values"],
        "receptor_indices": [1, 2, 3],
        "ligand_indices": [4, 5, 6],
        "force_constants": {
            "kr": 100.0,
            "kthetaA": 100.0,
            "kthetaB": 100.0,
            "kphiA": 10.0,
            "kphiB": 10.0,
            "kphiC": 10.0,
        },
    }
    path = tmp_path / "boresch_equilibrium_committed.json"

    committed = pipeline._commit_ensemble_boresch_equilibrium(params, str(path))

    assert committed["equilibrium_values"] == equilibrium
    assert params["equilibrium_values"] == equilibrium
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["equilibrium_values"] == boresch_simple["equilibrium_values"]
    assert payload["last_frame_geometry_diagnostic"] == diagnostic
    assert payload["equilibrium_source"] == "boresch_simple_ensemble_mean"
    assert payload["reanchor_applied"] is False


def test_exp019_run_full_pipeline_does_not_call_last_frame_reanchor():
    source = inspect.getsource(ABFEPipeline.run_full_pipeline)
    assert "update_boresch_from_last_frame" not in source
    assert "_commit_ensemble_boresch_equilibrium" in source
    rebalance_source = inspect.getsource(ABFEPipeline._rebalance_with_boresch)
    assert 'boresch_params["equilibrium_values"]["r0"] =' not in rebalance_source
