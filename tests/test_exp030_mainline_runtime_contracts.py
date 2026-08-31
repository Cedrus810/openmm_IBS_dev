import numpy as np
import pytest


pytest.importorskip("openmm")

from abfe_pipeline import _normalize_residual_sampling_runtime
from ibs_engine import (
    IBSWindowManagerDualLambda,
    IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION,
    _load_validated_joint_score_ledgers,
    _sha256_file,
)


SCORE_HASH = "a" * 64


def test_residual_runtime_is_explicit_and_fail_closed():
    disabled = _normalize_residual_sampling_runtime(False, None, None, 0.0, None)
    assert disabled["enabled"] is False
    assert disabled["sampling_score_sha256"] is None

    with pytest.raises(ValueError, match="显式设置 residual_sampling_enabled=True"):
        _normalize_residual_sampling_runtime(False, lambda: object(), None, 0.0, None)

    with pytest.raises(TypeError, match="residual_basis_force_factory"):
        _normalize_residual_sampling_runtime(True, None, lambda *_: [0.0], 0.0, SCORE_HASH)

    enabled = _normalize_residual_sampling_runtime(
        True,
        lambda: object(),
        lambda *_: [0.0],
        1.25,
        SCORE_HASH.upper(),
    )
    assert enabled["enabled"] is True
    assert enabled["energy_offset_kj_mol"] == pytest.approx(1.25)
    assert enabled["sampling_score_sha256"] == SCORE_HASH


def test_joint_ledgers_are_loaded_as_one_validated_frame_chain(tmp_path):
    n_states, n_frames = 2, 4
    sampling = np.arange(n_frames * n_states, dtype=np.float64).reshape(n_frames, n_states)
    residual = np.linspace(-1.0, 1.0, n_frames, dtype=np.float64)
    sampling_path = tmp_path / "dual_window_0_vdw_sampling_states.npy"
    residual_path = tmp_path / "dual_window_0_vdw_residual_basis.npy"
    np.save(sampling_path, sampling)
    np.save(residual_path, residual)
    convergence = {
        "sampling_score_sha256": SCORE_HASH,
        "residual_sampling_protocol_version": IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION,
        "joint_score_window_data": {
            "protocol_version": IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION,
            "sampling_states": {
                "shape": list(sampling.shape),
                "dtype": str(sampling.dtype),
                "sha256": _sha256_file(str(sampling_path)),
            },
            "residual_basis": {
                "shape": list(residual.shape),
                "dtype": str(residual.dtype),
                "sha256": _sha256_file(str(residual_path)),
            },
            "sampling_state_definition": (
                "softcore_U_k_plus_A_k_times_residual_basis_minus_offset"
            ),
            "physical_target_excludes_residual": True,
        },
    }

    loaded = _load_validated_joint_score_ledgers(
        str(tmp_path),
        0,
        "vdw",
        convergence,
        n_states=n_states,
        n_frames=n_frames,
        current_sampling_score_sha256=SCORE_HASH,
    )
    np.testing.assert_array_equal(loaded["sampling_state_energies"], sampling.T)
    np.testing.assert_array_equal(loaded["residual_basis"], residual)

    np.save(residual_path, residual + 1.0)
    with pytest.raises(ValueError, match="ledger hash"):
        _load_validated_joint_score_ledgers(
            str(tmp_path),
            0,
            "vdw",
            convergence,
            n_states=n_states,
            n_frames=n_frames,
            current_sampling_score_sha256=SCORE_HASH,
        )


def test_producer_manifest_loader_round_trip_uses_frames_by_states(tmp_path):
    """Exercise the mainline producer, durable manifest, and canonical loader."""
    n_states, n_frames = 3, 5

    class Sampler:
        energy_history = [np.arange(n_states, dtype=np.float64) + frame for frame in range(n_frames)]
        bias_history = [float(frame) for frame in range(n_frames)]
        base_energy_history = [float(-frame) for frame in range(n_frames)]
        sampling_state_energy_history = [
            np.arange(n_states, dtype=np.float64) + 10.0 * frame
            for frame in range(n_frames)
        ]
        residual_basis_history = [float(frame) / 10.0 for frame in range(n_frames)]

    manager = IBSWindowManagerDualLambda.__new__(IBSWindowManagerDualLambda)
    manager.output_dir = str(tmp_path)
    IBSWindowManagerDualLambda._enqueue_window_snapshot(manager, 0, "vdw", Sampler())

    sampling_path = tmp_path / "dual_window_0_vdw_sampling_states.npy"
    residual_path = tmp_path / "dual_window_0_vdw_residual_basis.npy"
    sampling = np.load(sampling_path, allow_pickle=False)
    residual = np.load(residual_path, allow_pickle=False)
    assert sampling.shape == (n_frames, n_states)
    assert residual.shape == (n_frames,)

    convergence = {
        "sampling_score_sha256": SCORE_HASH,
        "residual_sampling_protocol_version": IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION,
        "joint_score_window_data": {
            "protocol_version": IBS_RESIDUAL_SAMPLING_PROTOCOL_VERSION,
            "sampling_states": {
                "shape": list(sampling.shape),
                "dtype": str(sampling.dtype),
                "sha256": _sha256_file(str(sampling_path)),
            },
            "residual_basis": {
                "shape": list(residual.shape),
                "dtype": str(residual.dtype),
                "sha256": _sha256_file(str(residual_path)),
            },
            "sampling_state_definition": (
                "softcore_U_k_plus_A_k_times_residual_basis_minus_offset"
            ),
            "physical_target_excludes_residual": True,
        },
    }
    loaded = _load_validated_joint_score_ledgers(
        str(tmp_path), 0, "vdw", convergence,
        n_states=n_states, n_frames=n_frames,
        current_sampling_score_sha256=SCORE_HASH,
    )
    np.testing.assert_array_equal(loaded["sampling_state_energies"], sampling.T)
    np.testing.assert_array_equal(loaded["residual_basis"], residual)


def test_legacy_residual_ledger_version_fails_closed_without_axis_guess(tmp_path):
    # Square matrices make a blind transpose impossible to detect from shape.
    sampling = np.arange(4.0).reshape(2, 2)
    residual = np.arange(2.0)
    sampling_path = tmp_path / "dual_window_0_vdw_sampling_states.npy"
    residual_path = tmp_path / "dual_window_0_vdw_residual_basis.npy"
    np.save(sampling_path, sampling)
    np.save(residual_path, residual)
    convergence = {
        "sampling_score_sha256": SCORE_HASH,
        "residual_sampling_protocol_version": 1,
        "joint_score_window_data": {
            "protocol_version": 1,
            "sampling_states": {
                "shape": [2, 2], "dtype": "float64",
                "sha256": _sha256_file(str(sampling_path)),
            },
            "residual_basis": {
                "shape": [2], "dtype": "float64",
                "sha256": _sha256_file(str(residual_path)),
            },
            "sampling_state_definition": (
                "softcore_U_k_plus_A_k_times_residual_basis_minus_offset"
            ),
            "physical_target_excludes_residual": True,
        },
    }
    with pytest.raises(ValueError, match="拒绝加载旧 residual ledger"):
        _load_validated_joint_score_ledgers(
            str(tmp_path), 0, "vdw", convergence,
            n_states=2, n_frames=2,
            current_sampling_score_sha256=SCORE_HASH,
        )
