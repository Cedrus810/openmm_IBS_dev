"""OpenMM-independent EXP-030 joint-score diagnostics and decision math."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from exp030_protocol import DECISION_NO_GAIN, DECISION_PASS, Exp030ProtocolError


MIXTURE_ESS_DEFINITION = "harmonic_mean_per_state_decorrelated_responsibility_ess_v1"


def _statistical_inefficiency(series: np.ndarray, max_lag: int = 1000) -> float:
    values = np.asarray(series, dtype=np.float64).ravel()
    if values.size < 3:
        return 1.0
    centered = values - float(np.mean(values))
    variance = float(np.dot(centered, centered) / values.size)
    if not math.isfinite(variance) or variance <= 1.0e-30:
        return 1.0
    limit = min(values.size - 1, int(max_lag))
    correlation_sum = 0.0
    for lag in range(1, limit + 1):
        covariance = float(np.dot(centered[:-lag], centered[lag:]) / (values.size - lag))
        rho = covariance / variance
        if not math.isfinite(rho) or rho <= 0.0:
            break
        correlation_sum += rho
    return max(1.0, 1.0 + 2.0 * correlation_sum)


def joint_score_diagnostics(
    sampling_state_energies_kj_mol: np.ndarray,
    f_k_kj_mol: Sequence[float],
    temperature_k: float,
) -> dict[str, Any]:
    """Compute residual-aware occupancy, decorrelation and state ESS.

    ``sampling_state_energies`` must be the exact Group-1 discriminants
    before subtracting f_k: softcore U_k + A_k(B_phi-U_offset).  Physical
    target energies are intentionally not accepted here.
    """
    energies = np.asarray(sampling_state_energies_kj_mol, dtype=np.float64)
    f_k = np.asarray(f_k_kj_mol, dtype=np.float64)
    if energies.ndim != 2 or energies.shape[0] < 2 or energies.shape[1] < 1:
        raise Exp030ProtocolError("sampling-state ledger must have shape (states, frames)")
    if f_k.shape != (energies.shape[0],):
        raise Exp030ProtocolError("f_k shape does not match sampling-state ledger")
    if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(f_k)):
        raise Exp030ProtocolError("joint-score diagnostics received NaN/Inf")
    if not math.isfinite(float(temperature_k)) or float(temperature_k) <= 0.0:
        raise Exp030ProtocolError("temperature must be finite and positive")
    kt = 0.00831446261815324 * float(temperature_k)
    logits = -(energies - f_k[:, None]) / kt
    logits -= np.max(logits, axis=0, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=0, keepdims=True)
    n_states, n_frames = probabilities.shape
    rows = []
    state_ess = []
    for state_index in range(n_states):
        series = probabilities[state_index]
        g = _statistical_inefficiency(series)
        stride = max(1, int(math.ceil(g)))
        decorrelated = series[::stride]
        ess = float(np.square(np.sum(decorrelated)) / np.sum(np.square(decorrelated)))
        state_ess.append(ess)
        mean = float(np.mean(series))
        rows.append({
            "state_index": state_index,
            "mean_occupancy": mean,
            "normalized_occupancy_Kp": float(n_states * mean),
            "statistical_inefficiency": float(g),
            "decorrelation_stride": stride,
            "n_frames_raw": n_frames,
            "n_frames_decorrelated": int(decorrelated.size),
            "decorrelated_frame_yield": float(decorrelated.size / n_frames),
            "responsibility_ess": ess,
            "responsibility_ess_ratio_raw_frames": float(ess / n_frames),
        })
    state_ess_array = np.asarray(state_ess)
    mixture_ess = float(n_states / np.sum(1.0 / state_ess_array))
    return {
        "protocol": "exp030-joint-score-diagnostics-v1",
        "mixture_ess_definition": MIXTURE_ESS_DEFINITION,
        "mixture_effective_sample_size": mixture_ess,
        "mixture_ess_ratio_raw_frames": float(mixture_ess / n_frames),
        "min_normalized_occupancy_Kp": min(row["normalized_occupancy_Kp"] for row in rows),
        "min_decorrelated_frames": min(row["n_frames_decorrelated"] for row in rows),
        "per_state": rows,
    }


def paired_utility(
    *, baseline_effective_samples: float, candidate_effective_samples: float,
    baseline_cost_seconds: float, candidate_cost_seconds: float,
) -> dict[str, float]:
    values = (
        baseline_effective_samples, candidate_effective_samples,
        baseline_cost_seconds, candidate_cost_seconds,
    )
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
        raise Exp030ProtocolError("utility inputs must be finite and positive")
    sample_ratio = float(candidate_effective_samples / baseline_effective_samples)
    cost_ratio = float(candidate_cost_seconds / baseline_cost_seconds)
    exp_d = sample_ratio / cost_ratio
    return {
        "decorrelated_information_ratio_r": sample_ratio,
        "itt_cost_ratio_s": cost_ratio,
        "exp_D_equals_r_over_s": exp_d,
        "D_r": float(math.log(exp_d)),
        "fractional_itt_gain": float(exp_d - 1.0),
        "baseline_eta_per_hour": float(baseline_effective_samples / (baseline_cost_seconds / 3600.0)),
        "candidate_eta_per_hour": float(candidate_effective_samples / (candidate_cost_seconds / 3600.0)),
    }


def reduce_paired_decision(repeats: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]) -> dict[str, Any]:
    if len(repeats) != 3:
        raise Exp030ProtocolError("a decision requires all three paired repeats")
    d_values = []
    gains = []
    for expected, row in enumerate(repeats):
        if int(row.get("repeat_index", -1)) != expected:
            raise Exp030ProtocolError("repeat identities must be complete and ordered 0,1,2")
        value = float(row["utility"]["D_r"])
        gain = float(row["utility"]["fractional_itt_gain"])
        if not math.isfinite(value) or not math.isfinite(gain):
            raise Exp030ProtocolError("non-finite paired utility")
        d_values.append(value)
        gains.append(gain)
    n_positive = sum(value > 0.0 for value in d_values)
    median_gain = float(np.median(gains))
    passed = (
        n_positive >= int(gate["min_repeats_candidate_better"])
        and median_gain >= float(gate["median_fractional_itt_utility_gain"])
    )
    return {
        "n_positive_D_r": n_positive,
        "median_fractional_itt_gain": median_gain,
        "all_three_repeats_present": True,
        "decision": DECISION_PASS if passed else DECISION_NO_GAIN,
    }

def residual_path_target(physical_energies, residual_basis, a_k, offset=0.0):
    """Build E*=E+A(B-offset). E already includes LRC; never use sampling_states.

    This is an alternative analysis target, not a change to the frozen sampling
    score or to the preregistered physical-target policy. Inputs are not mutated.
    """
    energy = np.asarray(physical_energies, dtype=float)
    basis = np.asarray(residual_basis, dtype=float)
    amplitude = np.asarray(a_k, dtype=float)
    if energy.ndim != 2 or min(energy.shape) < 1:
        raise Exp030ProtocolError("target ledger must have shape (states, frames)")
    if basis.shape != (energy.shape[1],) or amplitude.shape != (energy.shape[0],):
        raise Exp030ProtocolError("residual basis/A_k shape mismatch")
    if not all(np.all(np.isfinite(x)) for x in (energy, basis, amplitude, offset)):
        raise Exp030ProtocolError("residual target received NaN/Inf")
    if not np.any(amplitude):
        return energy.copy()
    result = energy + amplitude[:, None] * (basis[None, :] - float(offset))
    if not np.all(np.isfinite(result)):
        raise Exp030ProtocolError("residual target overflow")
    return result


def validate_residual_path_chain(specs, expected_windows=6):
    """Validate declared Stage-2 Hamiltonian continuity, not runtime Force identity.

    Only global vdw endpoints are forced to zero residual. Window endpoints in
    the middle of the chain may have nonzero A. Per-window f_k gauges can differ.
    """
    if len(specs) != expected_windows or expected_windows < 1:
        raise Exp030ProtocolError("residual path requires all ordered windows")
    first = specs[0]
    keys = []
    for wi, spec in enumerate(specs):
        spec.validate()
        if spec.window_index != wi or spec.arm != first.arm:
            raise Exp030ProtocolError("residual path window/arm identity mismatch")
        if (dict(spec.phi_identity) != dict(first.phi_identity)
                or spec.residual_energy_offset_kj_mol != first.residual_energy_offset_kj_mol
                or spec.target_policy != first.target_policy):
            raise Exp030ProtocolError("residual model/offset/policy differs across windows")
        if spec.arm == "candidate" and any(not spec.phi_identity.get(k) for k in
                ("model_sha256", "plugin_source_sha256", "plugin_binary_sha256")):
            raise Exp030ProtocolError("missing residual model/plugin identity")
        row = [(round(s.lambda_coul, 14), round(s.lambda_vdw, 14))
               for s in spec.state_lambda_identity]
        if len(row) < 2 or any(lc != 0.0 or not 0.0 <= lv <= 1.0 for lc, lv in row):
            raise Exp030ProtocolError("expected the Stage-2 vdw path at lambda_coul=0")
        if any(row[i][1] <= row[i + 1][1] for i in range(len(row) - 1)):
            raise Exp030ProtocolError("vdw state order must decrease within each window")
        for key, amplitude in zip(row, spec.A_k):
            if key[1] in (0.0, 1.0) and abs(amplitude) > 1e-14:
                raise Exp030ProtocolError("global endpoint residual must be zero")
        keys.append(row)
    if keys[0][0] != (0.0, 1.0) or keys[-1][-1] != (0.0, 0.0):
        raise Exp030ProtocolError("chain must include both global endpoints vdw=1 and vdw=0")
    interfaces = []
    for wi in range(len(specs) - 1):
        shared = set(keys[wi]).intersection(keys[wi + 1])
        if keys[wi][-1] != keys[wi + 1][0] or shared != {keys[wi][-1]}:
            raise Exp030ProtocolError("adjacent windows must share exactly their boundary state")
        if abs(specs[wi].A_k[-1] - specs[wi + 1].A_k[0]) > 1e-14:
            raise Exp030ProtocolError("shared-state A_k mismatch")
        interfaces.append({"left_window": wi, "right_window": wi + 1,
                           "lambda_coul": keys[wi][-1][0],
                           "lambda_vdw": keys[wi][-1][1], "A_k": specs[wi].A_k[-1]})
    return {"declared_chain_continuity": True, "global_endpoint_residual_zero": True,
            "runtime_basis_identity_verified": False, "interfaces": interfaces}


def target_reweighting_summary(energies_target, bias, kt):
    """Single sampled-distribution estimate using actual bias, with raw Kish ESS.

    ESS here is calculated from literal importance weights, not the engine's
    common-mode-removed proxy. No independent-sample or accuracy claim is made.
    """
    energy = np.asarray(energies_target, dtype=float)
    bias = np.asarray(bias, dtype=float)
    if energy.ndim != 2 or energy.shape[0] < 2 or energy.shape[1] < 1:
        raise Exp030ProtocolError("target ledger must contain states and frames")
    if bias.shape != (energy.shape[1],):
        raise Exp030ProtocolError("bias shape mismatch")
    if not (np.all(np.isfinite(energy)) and np.all(np.isfinite(bias))
            and math.isfinite(kt) and kt > 0):
        raise Exp030ProtocolError("target weights require finite energies and positive kT")
    log_w = (bias[None, :] - energy) / kt
    if not np.all(np.isfinite(log_w)):
        raise Exp030ProtocolError("target log weights overflow")
    maxima = np.max(log_w, axis=1)
    lse = maxima + np.log(np.exp(log_w - maxima[:, None]).sum(axis=1))
    normalized = np.exp(log_w - lse[:, None])
    return {
        "delta_G_kJ_mol": float(-kt * (lse[-1] - lse[0])),
        "log_weight_sums": lse.tolist(),
        "raw_kish_ess_by_state": (1.0 / np.sum(normalized ** 2, axis=1)).tolist(),
        "max_normalized_weight_by_state": np.max(normalized, axis=1).tolist(),
        "dominant_frame_within_selection_by_state": np.argmax(normalized, axis=1).tolist(),
        "n_frames": int(energy.shape[1]),
    }


def residual_path_accounting(windows, kt):
    """Exact fixed-frame identity, with interface conversion terms explicit.

    Each window is a (physical E, residual-target E*, actual bias) tuple. The
    caller must first validate the shared-state chain and select the same frames
    for both targets in each window. This is arithmetic attribution, not a
    causal proof or an uncertainty estimate.
    """
    if not windows:
        raise Exp030ProtocolError("cannot account for an empty chain")
    rows = []
    for wi, (physical, retained, bias) in enumerate(windows):
        if np.shape(physical) != np.shape(retained):
            raise Exp030ProtocolError("dual targets must use identical states and frames")
        p = target_reweighting_summary(physical, bias, kt)
        r = target_reweighting_summary(retained, bias, kt)
        correction = -kt * (np.asarray(p["log_weight_sums"]) - np.asarray(r["log_weight_sums"]))
        rows.append({"window_index": wi, "physical": p, "residual_path": r,
                     "physical_minus_residual_free_energy_by_state": correction.tolist()})
    interfaces = [
        {"left_window": wi, "right_window": wi + 1,
         "conversion_mismatch_kJ_mol": (
             rows[wi]["physical_minus_residual_free_energy_by_state"][-1]
             - rows[wi + 1]["physical_minus_residual_free_energy_by_state"][0])}
        for wi in range(len(rows) - 1)]
    physical_total = sum(r["physical"]["delta_G_kJ_mol"] for r in rows)
    residual_total = sum(r["residual_path"]["delta_G_kJ_mol"] for r in rows)
    endpoint = (rows[-1]["physical_minus_residual_free_energy_by_state"][-1]
                - rows[0]["physical_minus_residual_free_energy_by_state"][0])
    interface_sum = sum(r["conversion_mismatch_kJ_mol"] for r in interfaces)
    closure = physical_total - residual_total - endpoint - interface_sum
    if abs(closure) > 1e-8:
        raise Exp030ProtocolError("dual-path arithmetic closure failed")
    return {"physical_total_kJ_mol": physical_total, "residual_path_total_kJ_mol": residual_total,
            "physical_minus_residual_path_kJ_mol": physical_total - residual_total,
            "global_endpoint_conversion_kJ_mol": endpoint,
            "interface_conversion_sum_kJ_mol": interface_sum, "closure_error_kJ_mol": closure,
            "windows": rows, "interfaces": interfaces}
