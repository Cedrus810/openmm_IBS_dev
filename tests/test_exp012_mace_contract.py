"""C0 tests for read-only MACE model identity and architecture inspection."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from local_residual.mace_contract import (
    MaceContractError,
    MaceModelContract,
    canonical_report_sha256,
    inspect_mace_model_contract,
)


class _ToyLinear:
    def __init__(self, irreps_out):
        self.irreps_out = irreps_out


class _ToyProduct:
    def __init__(self, irreps_out):
        self.linear = _ToyLinear(irreps_out)


class _ToyModel:
    interactions = (object(), object())
    products = (_ToyProduct("128x0e+128x1o"), _ToyProduct("128x0e"))
    r_max = 6.0
    atomic_numbers = (1, 6, 7, 8)

    def forward(self, *args, **kwargs):  # pragma: no cover - must never execute
        raise AssertionError("C0 must not execute a model forward")

    def get_descriptors(self, *args, **kwargs):  # pragma: no cover - forbidden path
        raise AssertionError("C0 must not use NumPy descriptor APIs")


_VERSIONS = {"torch": "2.test", "mace": "0.test", "e3nn": "0.e3"}


def _contract(path: Path) -> MaceModelContract:
    return MaceModelContract(
        model_path=str(path),
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_class=f"{_ToyModel.__module__}.{_ToyModel.__qualname__}",
        expected_torch_version=_VERSIONS["torch"],
        expected_mace_version=_VERSIONS["mace"],
        expected_e3nn_version=_VERSIONS["e3nn"],
        expected_interaction_layer_count=2,
        expected_product_layer_count=2,
        expected_r_max_angstrom=6.0,
        expected_atomic_numbers=(1, 6, 7, 8),
        expected_product_layer_index=1,
        expected_product_layer_irreps=("128x0e+128x1o", "128x0e"),
        expected_node_feats_dimension=640,
        expected_invariant_slice_start=512,
        expected_invariant_slice_stop=640,
        expected_invariant_slice_irreps="128x0e",
    )


def _inspect(contract: MaceModelContract):
    return inspect_mace_model_contract(
        contract,
        model_loader=lambda _path: _ToyModel(),
        version_provider=lambda: _VERSIONS,
    )


def test_toy_report_is_canonical_and_records_forbidden_paths(tmp_path):
    model_path = tmp_path / "toy.model"
    model_path.write_bytes(b"identified toy model")
    first = _inspect(_contract(model_path))
    second = _inspect(_contract(model_path))
    assert first == second
    report_hash = first.pop("report_sha256")
    assert report_hash == canonical_report_sha256(first)
    assert json.dumps(first, sort_keys=True, allow_nan=False)
    assert first["observed"]["node_feats_contract"]["recommended_invariant_slice"] == {
        "start": 512,
        "stop": 640,
        "irreps": "128x0e",
    }
    assert first["observed"]["node_feats_contract"]["product_layer_outputs"] == [
        {
            "index": 0,
            "irreps": "128x0e+128x1o",
            "dimension": 512,
            "concatenated_start": 0,
            "concatenated_stop": 512,
        },
        {
            "index": 1,
            "irreps": "128x0e",
            "dimension": 128,
            "concatenated_start": 512,
            "concatenated_stop": 640,
        },
    ]
    assert first["observed"]["node_feats_contract"]["concatenated_dimension"] == 640
    assert first["policy"] == {
        "model_forward_executed": False,
        "latent_forward_executed": False,
        "numpy_descriptor_path_allowed": False,
        "final_energy_allowed": False,
        "fragment_subtraction_allowed": False,
        "scientific_qualification": False,
    }


def test_sha_mismatch_fails_before_model_loading(tmp_path):
    model_path = tmp_path / "toy.model"
    model_path.write_bytes(b"identified toy model")
    contract = replace(_contract(model_path), expected_sha256="0" * 64)
    loaded = False

    def loader(_path):
        nonlocal loaded
        loaded = True
        return _ToyModel()

    with pytest.raises(MaceContractError, match="sha256 mismatch"):
        inspect_mace_model_contract(contract, model_loader=loader, version_provider=lambda: _VERSIONS)
    assert loaded is False


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    [
        ("expected_class", "wrong.Model", "model_class"),
        ("expected_torch_version", "wrong", "versions"),
        ("expected_interaction_layer_count", 3, "interaction_layer_count"),
        ("expected_r_max_angstrom", 5.0, "r_max_angstrom"),
        ("expected_atomic_numbers", (1, 6, 8), "atomic_numbers"),
        ("expected_product_layer_irreps", ("128x0e+64x1o", "128x0e"), "node_feats_contract"),
        ("expected_node_feats_dimension", 641, "node_feats_contract"),
    ],
)
def test_expected_architecture_mismatch_fails_closed(tmp_path, field, wrong_value, message):
    model_path = tmp_path / "toy.model"
    model_path.write_bytes(b"identified toy model")
    contract = replace(_contract(model_path), **{field: wrong_value})
    with pytest.raises(MaceContractError, match=message):
        _inspect(contract)


def test_contract_requires_all_expected_values(tmp_path):
    model_path = tmp_path / "toy.model"
    model_path.write_bytes(b"identified toy model")
    with pytest.raises(MaceContractError, match="expected_class"):
        replace(_contract(model_path), expected_class="")
    with pytest.raises(MaceContractError, match="atomic_numbers"):
        replace(_contract(model_path), expected_atomic_numbers=(6, 1))


def test_product_layer_count_mismatch_fails_closed(tmp_path):
    model_path = tmp_path / "toy.model"
    model_path.write_bytes(b"identified toy model")
    wrong = replace(
        _contract(model_path),
        expected_product_layer_count=3,
        expected_product_layer_irreps=("128x0e+128x1o", "128x0e", "1x0e"),
        expected_node_feats_dimension=641,
    )
    with pytest.raises(MaceContractError, match="product_layer_count"):
        _inspect(wrong)


def test_wrong_final_slice_zero_to_128_is_rejected_for_concatenated_features(tmp_path):
    model_path = tmp_path / "toy.model"
    model_path.write_bytes(b"identified toy model")
    wrong = replace(
        _contract(model_path),
        expected_invariant_slice_start=0,
        expected_invariant_slice_stop=128,
    )
    with pytest.raises(MaceContractError, match="node_feats_contract"):
        _inspect(wrong)


def test_import_does_not_require_mace_or_e3nn():
    # Importing the contract module above succeeded independently of whether
    # either optional package can be found.  Their presence is reported here
    # only to make the test's scope explicit.
    assert isinstance(importlib.util.find_spec("mace") is not None, bool)
    assert isinstance(importlib.util.find_spec("e3nn") is not None, bool)


def test_real_model_inspection_smoke_when_explicitly_enabled():
    model_name = os.environ.get("EXP012_MACE_MODEL_PATH")
    if not model_name:
        pytest.skip("set EXP012_MACE_MODEL_PATH and explicit EXP012_MACE_EXPECTED_* values")
    required = {
        "sha": "EXP012_MACE_EXPECTED_SHA256",
        "class": "EXP012_MACE_EXPECTED_CLASS",
        "torch": "EXP012_MACE_EXPECTED_TORCH_VERSION",
        "mace": "EXP012_MACE_EXPECTED_MACE_VERSION",
        "e3nn": "EXP012_MACE_EXPECTED_E3NN_VERSION",
    }
    missing = [name for name in required.values() if not os.environ.get(name)]
    if missing:
        pytest.fail("real smoke requested but expected values are missing: " + ", ".join(missing))
    contract = MaceModelContract(
        model_path=model_name,
        expected_sha256=os.environ[required["sha"]],
        expected_class=os.environ[required["class"]],
        expected_torch_version=os.environ[required["torch"]],
        expected_mace_version=os.environ[required["mace"]],
        expected_e3nn_version=os.environ[required["e3nn"]],
        expected_interaction_layer_count=2,
        expected_product_layer_count=2,
        expected_r_max_angstrom=6.0,
        expected_atomic_numbers=(1, 6, 7, 8, 9, 15, 16, 17, 35, 53),
        expected_product_layer_index=1,
        expected_product_layer_irreps=("128x0e+128x1o", "128x0e"),
        expected_node_feats_dimension=640,
        expected_invariant_slice_start=512,
        expected_invariant_slice_stop=640,
        expected_invariant_slice_irreps="128x0e",
    )
    report = inspect_mace_model_contract(contract)
    assert report["status"] == "PASSED_READ_ONLY_ARCHITECTURE_INSPECTION"
