"""外层 λ 神经基势单文件模块的 WP-1/WP-2 CPU 契约测试。"""

import hashlib
import json
import math

import pytest

from outer_lambda_neural_basis import (
    NeuralPathConfigError,
    NeuralPathIntegrityError,
    OuterLambdaController,
    load_neural_path_config,
    stable_payload_sha256,
)


pytestmark = pytest.mark.cpu_only


def _write_fixture_files(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = tmp_path / "basis.pt"
    indices = tmp_path / "indices.json"
    model.write_bytes(b"frozen-test-model-v1")
    indices.write_text(json.dumps([0, 2, 4]), encoding="utf-8")
    return model, indices, hashlib.sha256(model.read_bytes()).hexdigest()


def _config(tmp_path, *, enabled=True, coefficient=1.0):
    model, indices, model_sha = _write_fixture_files(tmp_path)
    return {
        "neural_path": {
            "enabled": enabled,
            "protocol_version": 1,
            "stage": "vanishing",
            "baseline_potential": "softcore",
            "endpoint_tolerance": 1.0e-12,
            "envelope": {"type": "sin2", "parameters": {}},
            "coefficient_model": {
                "type": "constant",
                "coefficients": [coefficient],
                "max_abs_coefficient": 1.0,
            },
            "bases": [
                {
                    "name": "reorg_basis_0",
                    "backend": "torchforce",
                    "model_path": str(model),
                    "sha256": model_sha,
                    "energy_offset_kj_mol": 2.0,
                    "atom_selection": "fixed_indices",
                    "atom_indices_path": str(indices),
                    "output_unit": "kJ_per_mol",
                    "precision": "single",
                    "periodic": True,
                }
            ],
            "safety": {
                "max_abs_basis_energy_kj_mol": 50.0,
                "max_abs_path_energy_kj_mol": 20.0,
                "max_force_norm_kj_mol_nm": 500.0,
                "fail_on_support_domain_violation": True,
            },
        }
    }


def test_sin2_endpoints_are_exact_zero_and_midpoint_is_one(tmp_path):
    controller = OuterLambdaController.from_mapping(_config(tmp_path))

    assert controller.envelope(0.0) == 0.0
    assert controller.envelope(1.0) == 0.0
    assert controller.envelope(0.5) == pytest.approx(1.0)
    assert controller.coefficient_matrix([0.0, 0.5, 1.0]) == (
        (0.0,),
        (1.0,),
        (0.0,),
    )


def test_same_lambda_is_bitwise_stable_across_windows(tmp_path):
    controller = OuterLambdaController.from_mapping(_config(tmp_path))
    left = controller.coefficient_matrix([0.0, 0.25, 0.5])
    right = controller.coefficient_matrix([0.5, 0.75, 1.0])

    assert left[-1] == right[0]
    assert left[-1][0].hex() == right[0][0].hex()


def test_disabled_path_is_exact_noop(tmp_path):
    controller = OuterLambdaController.from_mapping(
        _config(tmp_path, enabled=False)
    )
    original = (10.0, 20.0, 30.0)

    assert controller.coefficient_matrix([0.0, 0.5, 1.0]) == (
        (0.0,),
        (0.0,),
        (0.0,),
    )
    assert controller.compose_target_state_energies(
        original, [0.0, 0.5, 1.0], [999999.0]
    ) == original


def test_target_energy_contains_centered_path_term(tmp_path):
    controller = OuterLambdaController.from_mapping(_config(tmp_path))
    # U=7, b=2 => Ubar=5；A=[0, 1, 0]。
    path = controller.neural_path_state_energies([0.0, 0.5, 1.0], [7.0])
    target = controller.compose_target_state_energies(
        [10.0, 20.0, 30.0], [0.0, 0.5, 1.0], [7.0]
    )

    assert path == (0.0, 5.0, 0.0)
    assert target == (10.0, 25.0, 30.0)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_coefficients_fail_closed(tmp_path, bad):
    with pytest.raises(NeuralPathConfigError, match="有限"):
        OuterLambdaController.from_mapping(_config(tmp_path, coefficient=bad))


def test_out_of_range_lambda_fails_closed(tmp_path):
    controller = OuterLambdaController.from_mapping(_config(tmp_path))
    with pytest.raises(NeuralPathConfigError, match=r"\[0, 1\]"):
        controller.coefficient_matrix([0.0, 1.01])


def test_coefficient_change_changes_protocol_hash(tmp_path):
    first = OuterLambdaController.from_mapping(
        _config(tmp_path / "first", coefficient=0.5)
    )
    second = OuterLambdaController.from_mapping(
        _config(tmp_path / "second", coefficient=0.75)
    )

    assert first.protocol_sha256(lambdas=[0.0, 0.5, 1.0])
    assert (
        first.protocol_sha256(lambdas=[0.0, 0.5, 1.0])
        != second.protocol_sha256(lambdas=[0.0, 0.5, 1.0])
    )


def test_payload_hash_is_independent_of_mapping_order():
    assert stable_payload_sha256({"a": 1, "b": 2}) == stable_payload_sha256(
        {"b": 2, "a": 1}
    )


def test_model_hash_is_recomputed_and_mismatch_rejected(tmp_path):
    config = _config(tmp_path)
    config["neural_path"]["bases"][0]["sha256"] = "0" * 64
    with pytest.raises(NeuralPathIntegrityError, match="不匹配"):
        OuterLambdaController.from_mapping(config)


def test_model_and_selection_hashes_enter_protocol_payload(tmp_path):
    controller = OuterLambdaController.from_mapping(_config(tmp_path))
    basis = controller.protocol_payload()["bases"][0]

    assert len(basis["sha256"]) == 64
    assert len(basis["atom_indices_sha256"]) == 64


def test_cv_budget_enforces_2k_plus_m_limit(tmp_path):
    controller = OuterLambdaController.from_mapping(_config(tmp_path))
    assert controller.validate_cv_budget(15) == 31
    with pytest.raises(NeuralPathConfigError, match="超过上限"):
        controller.validate_cv_budget(16)


def test_safety_gate_rejects_large_centered_basis_energy(tmp_path):
    controller = OuterLambdaController.from_mapping(_config(tmp_path))
    with pytest.raises(NeuralPathConfigError, match="超过安全上限"):
        controller.neural_path_state_energies([0.5], [100.0])


def test_json_file_loader(tmp_path):
    config = _config(tmp_path)
    path = tmp_path / "neural_path.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_neural_path_config(path)
    assert loaded.enabled is True
    assert loaded.state_coefficients(0.5) == (1.0,)
