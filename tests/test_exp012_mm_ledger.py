import copy
import json
from pathlib import Path

import numpy as np
import pytest

from exp012_xed.mm_ledger import (
    GAS_CONSTANT_KJ_MOL_K,
    _resolve_ledger_slice_target,
    _validate_source_lambdas,
    analytic_ibs_bias_kj_mol,
    compose_mm_ledger_arrays,
    stable_logsumexp,
)
from exp012_xed.schema import Exp012IntegrityError, Exp012ProtocolError


pytestmark = pytest.mark.cpu_only


def _repository_target():
    path = Path(__file__).resolve().parents[1] / "protocols" / "EXP-012_preregistration.json"
    return json.loads(path.read_text(encoding="utf-8"))["target"]


def test_stable_logsumexp_handles_large_logits():
    assert stable_logsumexp([1000.0, 999.0]) == pytest.approx(
        1000.0 + np.log1p(np.exp(-1.0))
    )


def test_analytic_ibs_bias_uses_softcore_and_fk_not_target_lrc():
    temperature = 300.0
    kT = GAS_CONSTANT_KJ_MOL_K * temperature
    softcore = np.array([10.0, 13.0])
    f_k = np.array([1.0, 2.0])
    expected = -kT * np.log(np.sum(np.exp(-(softcore - f_k) / kT)))
    assert analytic_ibs_bias_kj_mol(softcore, f_k, temperature) == pytest.approx(expected)


def test_complete_ledger_separates_target_lrc_from_sampling_wca_and_applies_beta_once():
    result = compose_mm_ledger_arrays(
        base_energy_kj_mol=[100.0, 110.0],
        softcore_cv_kj_mol=[[1.0, 3.0], [2.0, 5.0]],
        lrc_kj_mol=[[0.5, 0.7], [0.5, 0.7]],
        ibs_bias_kj_mol=[-2.0, -3.0],
        wca_bias_kj_mol=[0.25, 0.5],
        temperature_K=300.0,
    )
    beta = 1.0 / (GAS_CONSTANT_KJ_MOL_K * 300.0)
    np.testing.assert_allclose(result["target_total_kj_mol"], [[101.5, 103.7], [112.5, 115.7]])
    np.testing.assert_allclose(result["sampling_total_kj_mol"], [98.25, 107.5])
    np.testing.assert_allclose(
        result["target_reduced_potential"], beta * result["target_total_kj_mol"]
    )
    np.testing.assert_allclose(
        result["log_importance_unnormalized"],
        beta * (result["sampling_bias_kj_mol"][:, None] - result["target_interaction_kj_mol"]),
    )
    np.testing.assert_allclose(
        result["adjacent_gap_reduced"],
        np.diff(result["target_reduced_potential"], axis=1),
    )


def test_common_base_cancels_from_importance_weights():
    kwargs = dict(
        softcore_cv_kj_mol=[[1.0, 2.0]],
        lrc_kj_mol=[[0.1, 0.2]],
        ibs_bias_kj_mol=[-1.0],
        wca_bias_kj_mol=[0.3],
        temperature_K=300.0,
    )
    first = compose_mm_ledger_arrays(base_energy_kj_mol=[0.0], **kwargs)
    second = compose_mm_ledger_arrays(base_energy_kj_mol=[1.0e6], **kwargs)
    np.testing.assert_allclose(
        first["log_importance_unnormalized"], second["log_importance_unnormalized"]
    )


def test_ledger_slice_resolves_lambdas_by_global_state_id_without_endpoint_inference():
    resolved = _resolve_ledger_slice_target(_repository_target())
    assert resolved["global_state_ids"] == [0, 1, 2, 3, 4]
    assert resolved["schedule_index_range_half_open"] == [0, 5]
    assert resolved["lambda_vdw"][-1] == pytest.approx(0.73187638)
    assert resolved["lambda_vdw"][-1] != 0.0


def test_ledger_slice_missing_global_id_fails_closed():
    target = copy.deepcopy(_repository_target())
    target["ledger_slice"]["global_state_ids"][-1] = 99
    with pytest.raises(Exp012ProtocolError, match="absent from the schedule"):
        _resolve_ledger_slice_target(target)


def test_ledger_slice_out_of_order_global_ids_fail_closed():
    target = copy.deepcopy(_repository_target())
    target["ledger_slice"]["global_state_ids"][1:3] = [2, 1]
    with pytest.raises(Exp012ProtocolError, match="global schedule order"):
        _resolve_ledger_slice_target(target)


def test_source_lambda_mismatch_with_mapped_global_schedule_fails_closed():
    mapped = _resolve_ledger_slice_target(_repository_target())["lambda_vdw"]
    inconsistent = list(mapped)
    inconsistent[2] += 1.0e-4
    with pytest.raises(Exp012IntegrityError, match="global schedule states"):
        _validate_source_lambdas(mapped, (("manifest lambda_vdw", inconsistent),))


def test_source_lambda_extra_precision_is_retained_for_system_reconstruction():
    mapped = _resolve_ledger_slice_target(_repository_target())["lambda_vdw"]
    historical = list(mapped)
    historical[1] = 0.9235285427938671
    validated = _validate_source_lambdas(
        mapped,
        (
            ("manifest lambda_vdw", historical),
            ("bias lambda_vdw", historical),
            ("report lambda_vdw", historical),
        ),
    )
    assert validated == historical


@pytest.mark.parametrize(
    "field,value",
    [
        ("softcore_cv_kj_mol", [[1.0, np.nan]]),
        ("lrc_kj_mol", [[0.0]]),
        ("ibs_bias_kj_mol", [np.inf]),
    ],
)
def test_nonfinite_or_shape_mismatch_fails_closed(field, value):
    kwargs = dict(
        base_energy_kj_mol=[0.0],
        softcore_cv_kj_mol=[[1.0, 2.0]],
        lrc_kj_mol=[[0.0, 0.0]],
        ibs_bias_kj_mol=[-1.0],
        wca_bias_kj_mol=[0.0],
        temperature_K=300.0,
    )
    kwargs[field] = value
    with pytest.raises(Exp012ProtocolError):
        compose_mm_ledger_arrays(**kwargs)
