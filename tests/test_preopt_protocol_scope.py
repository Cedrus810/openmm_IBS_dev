"""预优化缓存指纹必须只覆盖真实 pilot 输入。"""

import pytest

openmm = pytest.importorskip("openmm")
from openmm import unit  # noqa: E402

import abfe_pipeline as pipeline  # noqa: E402


def _fake_pipeline(run_config=None):
    pipe = pipeline.ABFEPipeline.__new__(pipeline.ABFEPipeline)
    pipe.system = openmm.System()
    pipe.system.addParticle(12.0 * unit.dalton)
    pipe.topology = None
    pipe.ligand_indices = [0]
    pipe.temperature = 300.0 * unit.kelvin
    pipe._last_run_config = dict(run_config or {})
    return pipe


def _boresch(diagnostics=None):
    return {
        "receptor_indices": [1, 2, 3],
        "ligand_indices": [4, 5, 6],
        "equilibrium_values": {
            "r0": 0.4,
            "thetaA0": 2.0,
            "thetaB0": 1.8,
            "phiA0": 0.1,
            "phiB0": -0.2,
            "phiC0": 0.3,
        },
        "force_constants": {
            "kr": 2000.0,
            "kthetaA": 200.0,
            "kthetaB": 200.0,
            "kphiA": 100.0,
            "kphiB": 100.0,
            "kphiC": 100.0,
        },
        "diagnostics": diagnostics or {"score": 1.0},
        "source": "old-run",
    }


def _key(pipe, **kwargs):
    options = {
        "requested_n_states": 12,
        "pilot_n_steps_per_state": 10000,
        "pilot_finite_difference_delta": 0.01,
    }
    options.update(kwargs)
    return pipe._preopt_protocol_key(
        "vanishing",
        "softcore",
        _boresch(),
        **options,
    )


def test_preopt_key_ignores_production_config_and_boresch_diagnostics():
    base = _fake_pipeline(
        {
            "n_states_per_stage": 12,
            "n_steps_per_window": 500000,
            "steps_per_update": 500,
            "kwargs": {
                "warmup_steps": 500000,
                "n_workers": 24,
                "stage2_production_rescue_rounds": 2,
            },
        }
    )
    changed = _fake_pipeline(
        {
            "n_states_per_stage": 12,
            "n_steps_per_window": 900000,
            "steps_per_update": 1000,
            "enable_early_stop": True,
            "kwargs": {
                "warmup_steps": 123,
                "n_workers": 1,
                "stage2_production_rescue_rounds": 99,
            },
        }
    )

    first = _key(base)
    second = changed._preopt_protocol_key(
        "vanishing",
        "softcore",
        _boresch({"score": 999.0, "timestamp": "now"}),
        requested_n_states=12,
        pilot_n_steps_per_state=10000,
        pilot_finite_difference_delta=0.01,
    )

    assert first == second
    payload = first["payload"]
    assert "preopt_code_sha256" not in payload
    assert "run_config" not in payload
    assert payload["preopt_hamiltonian_protocol_version"] == (
        pipeline.PREOPT_HAMILTONIAN_PROTOCOL_VERSION
    )
    assert "coion_probe_hamiltonian_protocol_version" not in payload
    assert "diagnostics" not in payload["boresch_params"]
    assert "source" not in payload["boresch_params"]


@pytest.mark.parametrize(
    "field, value",
    [
        ("requested_n_states", 13),
        ("pilot_n_steps_per_state", 20000),
        ("pilot_finite_difference_delta", 0.02),
    ],
)
def test_preopt_key_changes_for_actual_pilot_inputs(field, value):
    pipe = _fake_pipeline({"n_states_per_stage": 12})
    base_kwargs = {
        "requested_n_states": 12,
        "pilot_n_steps_per_state": 10000,
        "pilot_finite_difference_delta": 0.01,
    }
    changed_kwargs = dict(base_kwargs)
    changed_kwargs[field] = value
    old_key = _key(pipe, **base_kwargs)
    new_key = _key(pipe, **changed_kwargs)
    assert old_key != new_key
    assert any(field in item for item in pipeline._protocol_key_differences(old_key, new_key))


def test_legacy_broad_preopt_cache_migrates_without_source_hash_recompute():
    pipe = _fake_pipeline({"n_states_per_stage": 12})
    fresh = _key(pipe)
    fresh_payload = fresh["payload"]
    old_payload = {
        "kind": fresh_payload["kind"],
        "stage_name": fresh_payload["stage_name"],
        "potential_type": fresh_payload["potential_type"],
        "dexp_params": fresh_payload["dexp_params"],
        "boresch_params": _boresch({"old": True}),
        "decharge_method": "pme",
        "run_config": {
            "n_states_per_stage": 12,
            "n_steps_per_window": 999999,
            "kwargs": {"n_workers": 24, "warmup_steps": 1},
        },
        "temperature_K": fresh_payload["temperature_K"],
        "pressure_bar": 1.0,
        "ligand_indices": fresh_payload["ligand_indices"],
        "system_xml_sha256": fresh_payload["system_xml_sha256"],
        "topology_sha256": fresh_payload["topology_sha256"],
        "preopt_code_sha256": "hash-from-before-b5-helper",
        "aces_softcore_params": fresh_payload["aces_softcore_params"],
        "thermodynamic_path_protocol_version": fresh_payload[
            "thermodynamic_path_protocol_version"
        ],
        "code_sha256": "legacy-wide-code-hash",
        "wca_accounting_version": 1,
        "ibs_bias_protocol_version": 1,
        "final_gate_thresholds": {},
    }
    legacy = pipeline._protocol_fingerprint(old_payload)
    assert pipeline.ABFEPipeline._preopt_cache_matches_ignoring_code_hash(
        legacy, fresh
    )

    # A future Hamiltonian protocol bump must make the same legacy cache
    # ineligible for migration.  The migration projection treats old broad
    # caches as protocol v1; it must not wildcard this field.
    fresh_v2_payload = dict(fresh["payload"])
    fresh_v2_payload["preopt_hamiltonian_protocol_version"] = 2
    fresh_v2 = pipeline._protocol_fingerprint(fresh_v2_payload)
    assert not pipeline.ABFEPipeline._preopt_cache_matches_ignoring_code_hash(
        legacy, fresh_v2
    )


def test_old_charged_preopt_key_without_coion_probe_version_is_rejected():
    pipe = _fake_pipeline({"n_states_per_stage": 12})
    pipe._coion_runtime_identity = {
        "schema_version": 1,
        "leg": "complex",
        "charge_treatment": "co_alchemical_charge_transfer",
        "fingerprint": "same-coion",
        "ion_atom_indices": [7],
    }
    fresh = _key(pipe)
    fresh_payload = fresh["payload"]
    version_field = "coion_probe_hamiltonian_protocol_version"
    assert version_field in fresh_payload

    # This models the old charged cache: identity was already recorded, but
    # the probe implementation predated the native B3 offsets/restraint and
    # therefore had no charged-Hamiltonian version field.
    old_payload = dict(fresh_payload)
    old_payload.pop(version_field)
    old_payload["preopt_code_sha256"] = "legacy-wide-code-hash"
    old_payload["run_config"] = {
        "n_states_per_stage": 12,
        "kwargs": {},
    }
    old = pipeline._protocol_fingerprint(old_payload)
    assert not pipeline.ABFEPipeline._preopt_cache_matches_ignoring_code_hash(
        old, fresh
    )
