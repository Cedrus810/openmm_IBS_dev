"""Whole-run LORO and trailing contiguous validation split tests."""

import pytest
import importlib.util
from pathlib import Path

np = pytest.importorskip("numpy")

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_exp019_softlift_loro.py"
_SPEC = importlib.util.spec_from_file_location("exp020_softlift_loro", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_rows = _MODULE._rows
_build_seeded_model = _MODULE._build_seeded_model


def test_trailing_validation_split_never_randomizes_or_leaks_runs():
    arrays = {"partition_index": np.repeat(np.array([0, 1, 2]), 5)}
    train = _rows(arrays, [0, 1], trailing_validation=False)
    validation = _rows(arrays, [0, 1], trailing_validation=True)
    test = _rows(arrays, [2], trailing_validation=None)
    assert train.tolist() == [0, 1, 2, 3, 5, 6, 7, 8]
    assert validation.tolist() == [4, 9]
    assert set(train).isdisjoint(set(validation))
    assert set(train).isdisjoint(set(test))
    assert set(validation).isdisjoint(set(test))


def test_seed_is_applied_before_model_construction():
    torch = pytest.importorskip("torch")
    from local_residual.softlift import primary_r1_config

    config = primary_r1_config("0" * 64)
    first = _build_seeded_model(config, context_mode="normal", seed=17)
    repeat = _build_seeded_model(config, context_mode="normal", seed=17)
    different = _build_seeded_model(config, context_mode="normal", seed=18)

    for first_parameter, repeat_parameter in zip(first.parameters(), repeat.parameters()):
        assert torch.equal(first_parameter, repeat_parameter)
    assert any(
        not torch.equal(first_parameter, different_parameter)
        for first_parameter, different_parameter in zip(first.parameters(), different.parameters())
    )
