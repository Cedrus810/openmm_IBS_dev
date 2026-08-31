"""Production-qualification boundary for merged charge-transfer support."""

import json
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only
pytest.importorskip("openmm")

import abfe_core as core
import runabfe


def _valid_reservoir_correction(temperature_K=300.0):
    leg = {
        "delta_G_release_kJ_mol": 4.0,
        "error_kJ_mol": 0.3,
        "converged": True,
        "restraint_protocol_version": core.CO_ALCHEMICAL_ION_IDENTITY_PROTOCOL_VERSION,
        "box_model": core.CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL,
        "sample_count": 4,
    }
    return {
        "protocol_version": core.CHARGE_TRANSFER_RESERVOIR_CORRECTION_PROTOCOL_VERSION,
        "charge_treatment": core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        "box_model": core.CO_ALCHEMICAL_ION_RESTRAINT_BOX_MODEL,
        "temperature_K": temperature_K,
        "complex": dict(leg),
        "solvent": dict(leg, delta_G_release_kJ_mol=7.0),
        "validation": {
            "c4_passed": True,
            "c5_passed": True,
            "evidence": ["c4-c5-test-artifact"],
            "max_abs_closure_residual_kJ_mol": 0.1,
            "max_abs_box_size_sensitivity_kJ_mol": 0.1,
            "max_abs_anchor_sensitivity_kJ_mol": 0.1,
            "max_abs_restraint_sensitivity_kJ_mol": 0.1,
            "n_box_sizes": 3,
            "n_anchor_choices": 3,
            "n_restraint_settings": 3,
        },
    }


def test_valid_reservoir_correction_is_normalized_and_temperature_bound():
    payload = core.validate_charge_transfer_reservoir_correction(
        _valid_reservoir_correction(310.0), temperature_k=310.0
    )
    assert payload["delta_G_bind_correction_kJ_mol"] == pytest.approx(3.0)
    assert payload["error_kJ_mol"] == pytest.approx((2 * 0.3**2) ** 0.5)
    with pytest.raises(ValueError, match="温度不一致"):
        core.validate_charge_transfer_reservoir_correction(
            _valid_reservoir_correction(300.0), temperature_k=310.0
        )
    inflated = _valid_reservoir_correction(310.0)
    inflated["validation"]["tolerance_kJ_mol"] = 2.0
    inflated["validation"]["max_abs_closure_residual_kJ_mol"] = 1.5
    with pytest.raises(ValueError, match="不能放宽项目闭合门"):
        core.validate_charge_transfer_reservoir_correction(
            inflated, temperature_k=310.0
        )


def test_reservoir_correction_closes_cycle_without_promoting_production():
    payload = core.resolve_charge_treatment(
        core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=1.0,
        charge_transfer_reservoir_correction=_valid_reservoir_correction(),
    )
    assert payload["closes_thermodynamic_cycle"] is True
    assert payload["thermodynamic_cycle_closure_validated"] is True
    assert payload["production_qualified"] is False


def test_charge_transfer_requires_reservoir_correction_before_closure():
    payload = core.resolve_charge_treatment(
        core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=1.0,
    )
    assert payload["closes_thermodynamic_cycle"] is False
    assert payload["thermodynamic_cycle_closure_validated"] is False
    assert payload["incomplete_cycle_reason"] == (
        "tethered_charge_carrier_reservoir_correction_not_validated"
    )
    assert payload["feature_status"] == "experimental"
    assert payload["production_qualified"] is False
    assert payload["c4_passed"] is False
    assert payload["c5_passed"] is False


def test_neutral_path_keeps_legacy_qualification_shape():
    payload = core.resolve_charge_treatment("neutral", ligand_net_charge_e=0.0)
    assert "production_qualified" not in payload
    assert "feature_status" not in payload


def test_coannihilation_remains_method_comparison_only():
    payload = core.resolve_charge_treatment(
        core.CHARGE_TREATMENT_CO_ANNIHILATION_EXPERIMENTAL,
        ligand_net_charge_e=1.0,
        environment_type="soluble",
    )
    assert payload["production_qualified"] is False
    assert payload["feature_status"] == "experimental_method_comparison_only"


def test_run_provenance_records_machine_readable_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr(runabfe, "_collect_pipeline_provenance", lambda **_kwargs: {})
    config = SimpleNamespace(
        as_dict=lambda: {},
        get=lambda _key, default=None: default,
    )
    protocol = core.resolve_charge_treatment(
        core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=1.0,
    )
    written = runabfe._write_run_provenance(
        str(tmp_path), config, charge_protocol=protocol
    )
    assert written["production_qualified"] is False
    assert written["production_qualification"]["c4_passed"] is False
    on_disk = json.loads((tmp_path / "run_provenance.json").read_text())
    assert on_disk["production_qualification"] == written["production_qualification"]


def test_qualification_cannot_be_promoted_by_closed_cycle(monkeypatch):
    monkeypatch.setattr(core, "CHARGE_TRANSFER_C4_PASSED", True)
    monkeypatch.setattr(core, "CHARGE_TRANSFER_C5_PASSED", False)
    payload = core.charge_treatment_qualification_payload(
        core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
    )
    assert payload["c4_passed"] is True
    assert payload["c5_passed"] is False
    assert payload["production_qualified"] is False



def test_postanalysis_qualification_is_resolved_fail_closed():
    protocol = core.resolve_charge_treatment(
        core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER,
        ligand_net_charge_e=1.0,
    )
    resolved = runabfe._resolve_production_qualification_from_sources(
        {}, {}, {"charge_protocol": protocol}
    )
    assert resolved["production_qualified"] is False
    assert resolved["c4_passed"] is False
    assert resolved["c5_passed"] is False


def test_postanalysis_qualification_rejects_mixed_artifacts():
    charge_transfer = core.charge_treatment_qualification_payload(
        core.CHARGE_TREATMENT_CO_ALCHEMICAL_CHARGE_TRANSFER
    )
    tampered = dict(charge_transfer)
    tampered["production_qualified"] = True
    with pytest.raises(RuntimeError, match="qualification mismatch"):
        runabfe._resolve_production_qualification_from_sources(
            {"production_qualification": charge_transfer},
            {"production_qualification": tampered},
        )


def test_postanalysis_neutral_sources_keep_legacy_shape():
    assert runabfe._resolve_production_qualification_from_sources(
        {"charge_treatment": "neutral"},
        {"charge_treatment": "neutral"},
    ) == {}



def test_postanalysis_qualification_rejects_incomplete_payload():
    with pytest.raises(RuntimeError, match="incomplete"):
        runabfe._resolve_production_qualification_from_sources(
            {"production_qualification": {"production_qualified": False}}
        )
