#!/usr/bin/env python
"""EXP-016: temporal attribution audit for the frozen EXP-012 candidate signals.

This is deliberately an offline audit.  It never changes a production system,
checkpoint, trajectory, or preregistration.  The only event label available in
the current data is an MM-ledger surrogate:

    argmin_k(target_interaction_k - f_k)

An adjacent 0 <-> 1 change in that label is reported as a
``dominant_component_switch``.  It is *not* a physical alchemical-state
trajectory, a replica round trip, or proof of a basin crossing.  The script
therefore reports surrogate-event prediction separately and fail-closed marks
physical crossing prediction as unavailable.

The audit uses the three registered continuous runs, preserves run boundaries,
fits the one-dimensional event predictor on two complete runs, and evaluates
the third complete run (leave-one-run-out).  No random frame split is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finite_array(value: Any, name: str):
    import numpy as np

    array = np.asarray(value)
    if not np.isfinite(array).all():
        raise AuditError(f"{name} contains non-finite values")
    return array


def standardize_fit(train, test):
    import numpy as np

    train = np.asarray(train, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    mean = float(np.mean(train))
    scale = float(np.std(train))
    if not math.isfinite(scale) or scale <= 1.0e-12:
        scale = 1.0
    return (train - mean) / scale, (test - mean) / scale, mean, scale


def autocorrelation_summary(values):
    """Return a conservative positive-sequence autocorrelation summary."""

    import numpy as np

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n < 3 or float(np.std(x)) <= 1.0e-14:
        return {
            "n_raw": n, "variance": float(np.var(x)) if n else 0.0,
            "integrated_autocorrelation_time_frames": 0.5,
            "statistical_inefficiency": 1.0,
            "effective_uncorrelated_samples": float(n),
            "decorrelation_lag_frames": 0,
            "lag_one_autocorrelation": None,
        }
    centered = x - float(np.mean(x))
    variance = float(np.dot(centered, centered) / n)
    max_lag = min(n - 1, max(1, n // 2))
    rho = []
    for lag in range(1, max_lag + 1):
        value = float(np.dot(centered[:-lag], centered[lag:]) / ((n - lag) * variance))
        rho.append(value)
    # Geyer-style initial positive sequence: stop at the first non-positive
    # autocorrelation.  This avoids integrating noisy long-lag tails.
    positive = []
    for value in rho:
        if value <= 0.0:
            break
        positive.append(value)
    g = max(1.0, 1.0 + 2.0 * sum(positive))
    tau = 0.5 * g
    decorrelation = next((i + 1 for i, value in enumerate(rho) if value <= math.exp(-1.0)), max_lag)
    return {
        "n_raw": n,
        "variance": variance,
        "integrated_autocorrelation_time_frames": tau,
        "statistical_inefficiency": g,
        "effective_uncorrelated_samples": float(n / g),
        "decorrelation_lag_frames": int(decorrelation),
        "lag_one_autocorrelation": float(rho[0]) if rho else None,
    }


def auc_score(scores, labels):
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    # Pairwise form is stable for the small 500-frame held-out runs.
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


def average_precision(scores, labels):
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive_count = int(labels.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1)
    return float(np.sum((cumulative / ranks) * sorted_labels) / positive_count)


def fit_logistic_1d(train_x, train_y, *, steps: int = 1000, learning_rate: float = 0.05):
    import numpy as np

    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64)
    intercept = 0.0
    slope = 0.0
    # Deterministic, bounded Newton-free gradient descent.  The feature is
    # standardized using the training runs only; the held-out run is untouched.
    for _ in range(steps):
        logits = np.clip(intercept + slope * x, -40.0, 40.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient_i = float(np.mean(probability - y))
        gradient_s = float(np.mean((probability - y) * x))
        intercept -= learning_rate * gradient_i
        slope -= learning_rate * gradient_s
    return float(intercept), float(slope)


def prediction_metrics(scores, labels, probabilities):
    import numpy as np

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = labels == 1
    negatives = labels == 0
    brier = float(np.mean((probabilities - labels) ** 2))
    calibration = {
        "mean_predicted": float(np.mean(probabilities)),
        "event_rate": float(np.mean(labels)),
        "mean_predicted_positive_labels": float(np.mean(probabilities[positives])) if positives.any() else None,
        "mean_predicted_negative_labels": float(np.mean(probabilities[negatives])) if negatives.any() else None,
    }
    return {
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "event_rate": float(np.mean(labels)),
        "roc_auc": auc_score(scores, labels),
        "average_precision": average_precision(scores, labels),
        "brier_score": brier,
        "calibration": calibration,
    }


def circular_block_bootstrap_prediction(scores, labels, *, block_length: int, n_bootstrap: int, seed: int):
    """Block-bootstrap AUC/AP on one held-out continuous run."""

    import numpy as np

    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(scores)
    block_length = max(1, min(int(block_length), n))
    auc_values = []
    ap_values = []
    for _ in range(int(n_bootstrap)):
        sampled = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            sampled.extend((start + np.arange(block_length)) % n)
        sampled = np.asarray(sampled[:n], dtype=np.int64)
        auc = auc_score(scores[sampled], labels[sampled])
        ap = average_precision(scores[sampled], labels[sampled])
        if auc is not None:
            auc_values.append(auc)
        if ap is not None:
            ap_values.append(ap)
    if not auc_values:
        return {"auc_ci95": None, "average_precision_ci95": None, "n_bootstrap": 0}
    return {
        "auc_ci95": [float(np.percentile(auc_values, 2.5)), float(np.percentile(auc_values, 97.5))],
        "average_precision_ci95": (
            [float(np.percentile(ap_values, 2.5)), float(np.percentile(ap_values, 97.5))]
            if ap_values else None
        ),
        "n_bootstrap": int(len(auc_values)),
    }


def circular_block_bootstrap_difference(values, labels, *, block_length: int, n_bootstrap: int, seed: int):
    """CI for event-vs-control means while resampling contiguous blocks.

    This is an attribution CI, not a claim of independent production repeats.
    Each run is resampled separately and contributes equal weight to the final
    statistic.  The same sampled indices are used for values and labels.
    """

    import numpy as np

    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(values)
    if n == 0 or labels.sum() == 0 or labels.sum() == n:
        return {"difference_mean_event_minus_control": None, "ci95": None, "n_bootstrap": 0}
    block_length = max(1, min(int(block_length), n))
    differences = []
    for _ in range(int(n_bootstrap)):
        sampled = []
        while len(sampled) < n:
            start = int(rng.integers(0, n))
            sampled.extend((start + np.arange(block_length)) % n)
        sampled = np.asarray(sampled[:n], dtype=np.int64)
        sample_values = values[sampled]
        sample_labels = labels[sampled]
        if sample_labels.sum() == 0 or sample_labels.sum() == n:
            continue
        differences.append(float(sample_values[sample_labels == 1].mean() - sample_values[sample_labels == 0].mean()))
    if not differences:
        return {"difference_mean_event_minus_control": None, "ci95": None, "n_bootstrap": 0}
    observed = float(values[labels == 1].mean() - values[labels == 0].mean())
    interval = np.percentile(np.asarray(differences), [2.5, 97.5])
    return {
        "difference_mean_event_minus_control": observed,
        "ci95": [float(interval[0]), float(interval[1])],
        "n_bootstrap": int(len(differences)),
    }


def dihedral_series(trajectory, atom_indices):
    import mdtraj as md

    radians = md.compute_dihedrals(trajectory, [list(atom_indices)], periodic=True)[:, 0]
    return radians.astype("float64")


def load_student_scalar(dataset_path: Path, checkpoint_path: Path):
    """Evaluate the frozen D1 direct-gap checkpoint on the cached geometry."""

    import importlib.util
    import torch

    dataset_report_path = dataset_path.with_name(dataset_path.stem + "_report.json")
    if not dataset_report_path.is_file():
        raise AuditError(f"missing student dataset report: {dataset_report_path}")
    dataset_report = json.loads(dataset_report_path.read_text(encoding="utf-8"))
    if dataset_report.get("status") != "COMPLETED_GEOMETRY_AND_LEDGER_JOIN_NOT_TRAINED":
        raise AuditError("student dataset is not the sealed geometry-only dataset")

    from local_residual.student import build_local_residual_student, reindex_ligand_environment_edges

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise AuditError(f"student checkpoint variant is not direct_gap: {payload.get('variant')!r}")
    model = build_local_residual_student(payload["type_vocabulary"], **payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"])
    model = model.to(torch.float64)
    model.eval()

    import numpy as np

    with np.load(dataset_path) as data:
        arrays = {key: data[key] for key in data.files}
    ligand_indices = [int(v) for v in arrays["ligand_topology_indices"].tolist()]
    ligand_atomic_numbers = [int(v) for v in arrays["ligand_atomic_numbers"].tolist()]
    ligand_type_index = model.atomic_numbers_to_type_index(ligand_atomic_numbers)
    all_atomic = [int(v) for v in arrays["all_topology_atomic_numbers"].tolist()]
    offsets = arrays["edge_offsets"]
    outputs = []
    for row in range(int(arrays["partition_index"].shape[0])):
        start, end = int(offsets[row]), int(offsets[row + 1])
        edge_ligand = torch.tensor(arrays["edge_ligand_topology"][start:end], dtype=torch.int64)
        edge_environment = torch.tensor(arrays["edge_environment_topology"][start:end], dtype=torch.int64)
        distance = torch.tensor(arrays["edge_distance_angstrom"][start:end], dtype=torch.float64)
        reindexed = reindex_ligand_environment_edges(ligand_indices, edge_ligand, edge_environment)
        env_indices = reindexed["environment_topology_indices"].tolist()
        env_types = model.atomic_numbers_to_type_index([all_atomic[int(index)] for index in env_indices])
        with torch.no_grad():
            value = model(
                ligand_type_index, env_types,
                reindexed["edge_ligand_local"], reindexed["edge_environment_local"], distance,
            )
        outputs.append(float(value.item()))
    return np.asarray(outputs, dtype=np.float64), {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_held_out_run_id": payload.get("held_out_run_id"),
        "checkpoint_training_run_ids": payload.get("training_run_ids"),
        "checkpoint_variant": payload.get("variant"),
        "dataset_path": str(dataset_path.resolve()),
        "dataset_sha256": sha256_file(dataset_path),
        "dataset_report_sha256": sha256_file(dataset_report_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default="protocols/EXP-012_preregistration.json")
    parser.add_argument("--latent-cache-dir", default="output/outer_lambda_exp012/teacher_latent_cache")
    parser.add_argument("--ledger-dir", default="output/outer_lambda_exp012/mm_ledger_cuda")
    parser.add_argument("--candidate-screen-root", default="output/outer_lambda_slow_variable_screen")
    parser.add_argument("--student-dataset", default="output/outer_lambda_exp012/student_training_dataset.npz")
    parser.add_argument(
        "--student-checkpoint-dir",
        default="output/outer_lambda_exp012/student_checkpoints",
        help="directory containing one frozen direct_gap seed-0 checkpoint per held-out run",
    )
    parser.add_argument("--output-dir", default="output/outer_lambda_exp016")
    parser.add_argument("--block-length-frames", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--future-horizons-frames", default="1,5,10,25")
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args(argv)

    import numpy as np
    import mdtraj as md

    if args.block_length_frames <= 0 or args.bootstrap_replicates <= 0:
        parser.error("block length and bootstrap replicates must be positive")
    horizons = tuple(int(value) for value in args.future_horizons_frames.split(","))
    if not horizons or any(value <= 0 for value in horizons):
        parser.error("future horizons must be positive comma-separated integers")

    preregistration_path = (ROOT / args.preregistration).resolve()
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    runs = preregistration["inputs"]["runs"]
    if len(runs) != 3:
        raise AuditError(f"EXP-016 expects exactly three registered runs, found {len(runs)}")
    run_ids = [str(run["run_id"]) for run in runs]
    target = preregistration["target"]["ledger_slice"]
    f_k = np.asarray(target["f_k_kj_mol"], dtype=np.float64)
    primary = (4591, 4592, 4593, 4585)
    secondary = (4593, 4585, 4594, 4595)
    val251 = (4020, 4022, 4024, 4026)

    output_dir = (ROOT / args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise AuditError(f"refusing to overwrite non-empty EXP-016 output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    run_data: dict[str, dict[str, Any]] = {}
    manifest_runs = []
    for run in runs:
        run_id = str(run["run_id"])
        trajectory_path = (ROOT / run["trajectory"]["path"]).resolve()
        sample_report_path = (ROOT / run["sample_report"]["path"]).resolve()
        ledger_npz_path = (ROOT / args.ledger_dir / run_id / "ledger_arrays.npz").resolve()
        ledger_report_path = ledger_npz_path.with_name("ledger_report.json")
        latent_path = (ROOT / args.latent_cache_dir / f"latent_cache_{run_id}.npz").resolve()
        latent_report_path = latent_path.with_name(f"latent_cache_{run_id}_report.json")
        required = [trajectory_path, sample_report_path, ledger_npz_path, ledger_report_path, latent_path, latent_report_path]
        if any(not path.is_file() for path in required):
            missing = [str(path) for path in required if not path.is_file()]
            raise AuditError(f"{run_id}: missing inputs: {missing}")
        observed_trajectory_sha = sha256_file(trajectory_path)
        if observed_trajectory_sha != run["trajectory"]["sha256"]:
            raise AuditError(f"{run_id}: trajectory SHA-256 differs from preregistration")
        sample_report = json.loads(sample_report_path.read_text(encoding="utf-8"))
        ledger_report = json.loads(ledger_report_path.read_text(encoding="utf-8"))
        latent_report = json.loads(latent_report_path.read_text(encoding="utf-8"))
        if int(ledger_report["frame_count"]) != int(run["frame_count"]):
            raise AuditError(f"{run_id}: ledger frame count mismatch")
        if int(latent_report["frame_count"]) != int(run["frame_count"]):
            raise AuditError(f"{run_id}: latent frame count mismatch")
        if ledger_report["trajectory_sha256"] != observed_trajectory_sha:
            raise AuditError(f"{run_id}: ledger trajectory hash mismatch")
        trajectory = md.load(str(trajectory_path), top=str(ROOT / preregistration["inputs"]["artifacts"]["topology"]["path"]))
        if trajectory.n_frames != int(run["frame_count"]):
            raise AuditError(f"{run_id}: trajectory frame count mismatch")
        if trajectory.unitcell_vectors is None:
            raise AuditError(f"{run_id}: trajectory has no periodic box")
        # The DCDs contain ~73k atoms.  Extract the cheap coordinate signals
        # now and do not retain all three full trajectories in run_data; this
        # keeps the audit memory bounded without changing any values.
        primary_rad = dihedral_series(trajectory, primary)
        secondary_rad = dihedral_series(trajectory, secondary)
        val_rad = dihedral_series(trajectory, val251)
        cheap_signals = {
            "primary_torsion_sin": np.sin(primary_rad),
            "primary_torsion_cos": np.cos(primary_rad),
            "secondary_torsion_sin": np.sin(secondary_rad),
            "secondary_torsion_cos": np.cos(secondary_rad),
            "VAL251_chi1_sin": np.sin(val_rad),
            "VAL251_chi1_cos": np.cos(val_rad),
        }
        with np.load(ledger_npz_path) as ledger:
            ledger_arrays = {key: ledger[key] for key in ledger.files}
        with np.load(latent_path) as latent:
            pooled_latent = latent["pooled_latent"].astype(np.float64)
            latent_frame_index = latent["frame_index"]
        expected_index = np.arange(int(run["frame_count"]), dtype=latent_frame_index.dtype)
        if not np.array_equal(latent_frame_index, expected_index):
            raise AuditError(f"{run_id}: latent frame index is not 0..N-1")
        if not np.array_equal(ledger_arrays["frame_index"], np.arange(int(run["frame_count"]), dtype=ledger_arrays["frame_index"].dtype)):
            raise AuditError(f"{run_id}: ledger frame index is not 0..N-1")
        if not np.array_equal(ledger_arrays["frame_index"], latent_frame_index):
            raise AuditError(f"{run_id}: latent and ledger frame indices differ")
        for key in ("target_interaction_kj_mol", "adjacent_gap_reduced"):
            if key not in ledger_arrays:
                raise AuditError(f"{run_id}: ledger lacks {key}")
            finite_array(ledger_arrays[key], f"{run_id}/{key}")
        if ledger_arrays["target_interaction_kj_mol"].shape[1] != len(f_k):
            raise AuditError(f"{run_id}: target state count differs from f_k")
        sample_dt_ps = float(sample_report["timestep_ps"]) * float(sample_report["report_interval_steps"])
        if not math.isfinite(sample_dt_ps) or sample_dt_ps <= 0.0:
            raise AuditError(f"{run_id}: invalid sample dt from sample report")
        if trajectory.time is not None and len(trajectory.time) > 1:
            dcd_dt = float(trajectory.time[1] - trajectory.time[0])
        else:
            dcd_dt = None
        run_data[run_id] = {
            "pooled_latent": pooled_latent,
            "ledger": ledger_arrays,
            "cheap_signals": cheap_signals,
            "sample_dt_ps": sample_dt_ps,
            "dcd_time_delta": dcd_dt,
            "sample_report": sample_report,
        }
        manifest_runs.append({
            "run_id": run_id,
            "random_seed": int(run["random_seed"]),
            "frame_count": int(run["frame_count"]),
            "sampling_time_ps": float(sample_report["simulated_sampling_time_ps"]),
            "timestep_ps": float(sample_report["timestep_ps"]),
            "report_interval_steps": int(sample_report["report_interval_steps"]),
            "delta_t_save_ps": sample_dt_ps,
            "dcd_time_delta_raw": dcd_dt,
            "files": {str(path.relative_to(ROOT)): sha256_file(path) for path in required},
        })

    # Unsupervised teacher projection, fit once to all frames.  It uses no
    # event labels or thermodynamic target values.
    pooled_all = np.concatenate([run_data[run_id]["pooled_latent"] for run_id in run_ids], axis=0)
    latent_mean = pooled_all.mean(axis=0)
    _, _, vh = np.linalg.svd(pooled_all - latent_mean[None, :], full_matrices=False)
    pc1_vector = vh[0]
    pivot = int(np.argmax(np.abs(pc1_vector)))
    if pc1_vector[pivot] < 0:
        pc1_vector = -pc1_vector
    teacher_pc1_all = (pooled_all - latent_mean[None, :]) @ pc1_vector
    teacher_norm_all = np.linalg.norm(pooled_all, axis=1)

    # The student signal must obey the same LORO boundary as the event
    # predictor.  The run1 checkpoint is the DEC-045 frozen candidate; run2 and
    # run3 use the already-produced seed-0 checkpoints whose held-out identity
    # is fixed by their filenames/payloads.  No checkpoint is chosen using the
    # temporal-audit outcomes.
    student_identity = {}
    student_scalar_by_run = {}
    checkpoint_dir = (ROOT / args.student_checkpoint_dir).resolve()
    row_offset = 0
    for run in runs:
        run_id = str(run["run_id"])
        checkpoint_path = checkpoint_dir / f"{run_id}__direct_gap__seed0.pt"
        student_scalar_all, identity = load_student_scalar(
            (ROOT / args.student_dataset).resolve(), checkpoint_path
        )
        expected_total = sum(int(item["frame_count"]) for item in runs)
        if len(student_scalar_all) != expected_total:
            raise AuditError("student scalar length does not match the three-run concatenation")
        if identity.get("checkpoint_held_out_run_id") != run_id:
            raise AuditError(
                f"student checkpoint {checkpoint_path} is not the predeclared held-out model for {run_id}"
            )
        frame_count = int(run["frame_count"])
        student_scalar_by_run[run_id] = student_scalar_all[row_offset:row_offset + frame_count]
        student_identity[run_id] = identity
        row_offset += frame_count

    # The topology is shared across runs; compute the independent cheap
    # diagnostics directly from each continuous trajectory.
    signals_by_run: dict[str, dict[str, np.ndarray]] = {}
    offset = 0
    surrogate_labels_by_run: dict[str, np.ndarray] = {}
    for run_id in run_ids:
        data = run_data[run_id]
        ledger = data["ledger"]
        labels = np.argmin(ledger["target_interaction_kj_mol"] - f_k[None, :], axis=1).astype(np.int64)
        surrogate_labels_by_run[run_id] = labels
        n = len(labels)
        signals_by_run[run_id] = {
            "teacher_latent_pc1": teacher_pc1_all[offset:offset + n],
            "teacher_latent_norm": teacher_norm_all[offset:offset + n],
            "student_scalar_direct_gap": student_scalar_by_run[run_id],
            "state1_state0_gap_reduced": ledger["adjacent_gap_reduced"][:, 0].astype(np.float64),
            **data["cheap_signals"],
        }
        offset += n

    # Event protocol is frozen in this script: labels come from MM ledger only;
    # an event is any adjacent 0<->1 label change.  Future labels use horizons
    # that are exact multiples of the measured 1-ps save interval.
    all_event_reports = {}
    all_prediction_reports = {}
    for horizon in horizons:
        horizon_key = f"{horizon}_frames_{horizon * manifest_runs[0]['delta_t_save_ps']:.6g}ps"
        event_reports_for_horizon = {}
        prediction_reports_for_horizon = {}
        future_by_run = {}
        for run_id in run_ids:
            labels = surrogate_labels_by_run[run_id]
            transitions = ((labels[:-1] == 0) & (labels[1:] == 1)) | ((labels[:-1] == 1) & (labels[1:] == 0))
            event_at_frame = np.zeros(len(labels), dtype=np.int64)
            event_at_frame[1:] = transitions.astype(np.int64)
            future = np.zeros(len(labels), dtype=np.int64)
            for index in range(len(labels)):
                stop = min(len(labels), index + horizon + 1)
                future[index] = int(np.any(event_at_frame[index + 1:stop]))
            future_by_run[run_id] = future
            event_reports_for_horizon[run_id] = {
                "n_frames": int(len(labels)),
                "n_label_0": int(np.sum(labels == 0)),
                "n_label_1": int(np.sum(labels == 1)),
                "n_label_other": int(np.sum(~np.isin(labels, [0, 1]))),
                "n_adjacent_0_to_1_or_1_to_0": int(np.sum(transitions)),
                "n_future_event_positive": int(np.sum(future)),
                "event_rate": float(np.mean(future)),
            }

        signal_names = sorted(signals_by_run[run_ids[0]])
        for signal_name in signal_names:
            per_run = {}
            held_out_metrics = []
            for held_out in run_ids:
                train_runs = [run_id for run_id in run_ids if run_id != held_out]
                train_x = np.concatenate([signals_by_run[run_id][signal_name] for run_id in train_runs])
                train_y = np.concatenate([future_by_run[run_id] for run_id in train_runs])
                test_x = signals_by_run[held_out][signal_name]
                test_y = future_by_run[held_out]
                train_z, test_z, mean, scale = standardize_fit(train_x, test_x)
                intercept, slope = fit_logistic_1d(train_z, train_y)
                probabilities = 1.0 / (1.0 + np.exp(-np.clip(intercept + slope * test_z, -40.0, 40.0)))
                metrics = prediction_metrics(test_z * slope, test_y, probabilities)
                metrics["block_bootstrap_prediction"] = circular_block_bootstrap_prediction(
                    test_z * slope, test_y,
                    block_length=args.block_length_frames,
                    n_bootstrap=args.bootstrap_replicates,
                    seed=args.seed + 5000 * (run_ids.index(held_out) + 1) + horizon,
                )
                metrics.update({
                    "held_out_run_id": held_out,
                    "training_run_ids": train_runs,
                    "training_mean": mean,
                    "training_scale": scale,
                    "logistic_intercept": intercept,
                    "logistic_slope": slope,
                })
                held_out_metrics.append(metrics)
                per_run[held_out] = metrics
            # Equal-weight the independent run metrics, rather than pretending
            # 1500 frames are 1500 independent experiments.
            valid_auc = [entry["roc_auc"] for entry in held_out_metrics if entry["roc_auc"] is not None]
            valid_ap = [entry["average_precision"] for entry in held_out_metrics if entry["average_precision"] is not None]
            prediction_reports_for_horizon[signal_name] = {
                "held_out_runs": per_run,
                "mean_roc_auc_across_held_out_runs": float(np.mean(valid_auc)) if valid_auc else None,
                "mean_average_precision_across_held_out_runs": float(np.mean(valid_ap)) if valid_ap else None,
                "mean_brier_across_held_out_runs": float(np.mean([entry["brier_score"] for entry in held_out_metrics])),
                "all_three_runs_have_both_classes": all(entry["roc_auc"] is not None for entry in held_out_metrics),
            }
        all_event_reports[horizon_key] = event_reports_for_horizon
        all_prediction_reports[horizon_key] = prediction_reports_for_horizon

    autocorrelation = {}
    attribution = {}
    # Use the frozen 128-frame block rule from the prior hydration screening
    # (g~82-127), not a block length selected from the current labels.
    block_length = int(args.block_length_frames)
    for signal_name in sorted(signals_by_run[run_ids[0]]):
        autocorrelation[signal_name] = {
            run_id: autocorrelation_summary(signals_by_run[run_id][signal_name])
            for run_id in run_ids
        }
        attribution[signal_name] = {}
        for horizon_key, horizon_events in all_event_reports.items():
            horizon = int(horizon_key.split("_", 1)[0])
            per_run = {}
            for run_number, run_id in enumerate(run_ids):
                labels = np.zeros(len(surrogate_labels_by_run[run_id]), dtype=np.int64)
                source = surrogate_labels_by_run[run_id]
                transitions = ((source[:-1] == 0) & (source[1:] == 1)) | ((source[:-1] == 1) & (source[1:] == 0))
                event_at_frame = np.zeros(len(source), dtype=np.int64)
                event_at_frame[1:] = transitions.astype(np.int64)
                for index in range(len(source)):
                    labels[index] = int(np.any(event_at_frame[index + 1:min(len(source), index + horizon + 1)]))
                values = signals_by_run[run_id][signal_name]
                event_values = values[labels == 1]
                control_values = values[labels == 0]
                bootstrap = circular_block_bootstrap_difference(
                    values, labels, block_length=block_length,
                    n_bootstrap=args.bootstrap_replicates,
                    seed=args.seed + 1000 * (run_number + 1) + horizon,
                )
                per_run[run_id] = {
                    "n_event": int(len(event_values)),
                    "n_control": int(len(control_values)),
                    "mean_event": float(np.mean(event_values)) if len(event_values) else None,
                    "mean_control": float(np.mean(control_values)) if len(control_values) else None,
                    "standardized_mean_difference": (
                        float((np.mean(event_values) - np.mean(control_values)) / np.std(values))
                        if len(event_values) and len(control_values) and np.std(values) > 1.0e-12 else None
                    ),
                    "block_bootstrap": bootstrap,
                }
            attribution[signal_name][horizon_key] = per_run

    # Candidate registry and hard data-feasibility verdict.
    event_total = sum(item["n_adjacent_0_to_1_or_1_to_0"] for item in all_event_reports[next(iter(all_event_reports))].values())
    manifest = {
        "schema_version": "exp016-data-manifest-v1",
        "experiment_id": "EXP-016",
        "status": "SEALED_INPUT_INVENTORY",
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": sha256_file(preregistration_path),
        "registered_run_ids": run_ids,
        "run_count": len(run_ids),
        "raw_frame_count_total": int(sum(entry["frame_count"] for entry in manifest_runs)),
        "run_independence": "three separate scratch trajectories with registered seeds; no frame-level IID claim",
        "trajectory_time": {
            "measured_from": "sample_report.timestep_ps * report_interval_steps",
            "delta_t_save_ps_by_run": {entry["run_id"]: entry["delta_t_save_ps"] for entry in manifest_runs},
            "all_equal": len({entry["delta_t_save_ps"] for entry in manifest_runs}) == 1,
            "dcd_time_field_is_frame_counter": True,
        },
        "physical_crossing_data_available": False,
        "physical_crossing_reason": "IBS has one continuously integrated reference trajectory per run and no discrete alchemical state/replica history; the five-state ledger is an energy ledger, not a physical state trajectory.",
        "surrogate_event": {
            "name": "dominant_component_switch_0_1",
            "label_source": "argmin_k(target_interaction_kj_mol[k] - f_k[k])",
            "event_definition": "adjacent label change 0<->1",
            "lookahead_horizons_frames": list(horizons),
            "lookahead_horizons_ps": [float(value * manifest_runs[0]["delta_t_save_ps"]) for value in horizons],
            "label_leakage_check": "labels use only MM target_interaction and frozen f_k; no teacher latent, student scalar, or candidate signal",
            "is_physical_replica_round_trip": False,
            "total_adjacent_events_at_first_horizon": int(event_total),
        },
        "block_protocol": {
            "type": "circular_contiguous_block_bootstrap_within_run",
            "block_length_frames": block_length,
            "block_length_ps": float(block_length * manifest_runs[0]["delta_t_save_ps"]),
            "replicates": int(args.bootstrap_replicates),
            "seed": int(args.seed),
            "framewise_iid_primary_inference": False,
        },
        "candidate_signals": {
            "teacher_latent_pc1": "pooled_latent PC1, unsupervised projection fit without event/energy labels",
            "teacher_latent_norm": "pooled_latent Euclidean norm",
            "student_scalar_direct_gap": "predeclared seed-0 direct_gap checkpoint trained without the evaluated run (run1 is the DEC-045 candidate)",
            "state1_state0_gap_reduced": "MM ledger adjacent_gap_reduced[:,0], analysis observable rather than learned candidate",
            "primary_torsion": list(primary),
            "secondary_torsion": list(secondary),
            "VAL251_chi1": list(val251),
            "hydration_coordination": "summary-only in existing candidate_screen_v2; no per-frame series was cached, excluded from temporal prediction",
            "force_of_signal": "not available in existing caches; no coordinate gradients were stored for the teacher cache and no force series is inferred",
            "label_independence": {
                "teacher_latent_pc1": "independent of surrogate label values; unsupervised representation projection",
                "teacher_latent_norm": "independent of surrogate label values; representation-only observable",
                "primary_secondary_torsions": "independent coordinate observables",
                "student_scalar_direct_gap": "NOT_INDEPENDENT_FOR_THIS_SURROGATE: checkpoint was trained on adjacent-gap targets, which are derived from the same target interaction ledger used for the label",
                "state1_state0_gap_reduced": "NOT_INDEPENDENT_FOR_THIS_SURROGATE: directly contributes to the event-label construction",
            },
        },
        "runs": manifest_runs,
        "student_identity": student_identity,
    }
    manifest_body = dict(manifest)
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest_body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(output_dir / "EXP-016_data_manifest.json", manifest)

    audit = {
        "schema_version": "exp016-temporal-audit-v1",
        "experiment_id": "EXP-016",
        "status": "COMPLETED_OFFLINE_TEMPORAL_AUDIT",
        "manifest_sha256": manifest["manifest_sha256"],
        "physical_crossing_claim": "UNAVAILABLE",
        "surrogate_event_prediction_claim": "EXPLORATORY_ONLY",
        "data_feasibility": {
            "three_continuous_trajectories_present": True,
            "trajectory_ledger_latent_alignment_passed": True,
            "measured_delta_t_save_ps": manifest_runs[0]["delta_t_save_ps"],
            "all_horizons_are_save_interval_multiples": True,
            "physical_state_history_present": False,
            "independent_physical_crossing_labels_present": False,
            "surrogate_event_count_at_first_horizon": int(event_total),
        },
        "event_protocol": manifest["surrogate_event"],
        "autocorrelation": autocorrelation,
        "attribution": attribution,
        "future_event_prediction": all_prediction_reports,
        "pca": {
            "fit_scope": "all 1500 pooled teacher frames; unsupervised only",
            "component": "PC1",
            "latent_dimension": int(pooled_all.shape[1]),
            "center_l2_norm": float(np.linalg.norm(latent_mean)),
            "pc1_l2_norm": float(np.linalg.norm(pc1_vector)),
        },
        "force_signal_comparison": {
            "status": "NOT_MEASURED",
            "reason": "teacher cache was representation-only/no_grad and the cached student geometry has no coordinate gradient series; measuring force(signal) would require a separate offline autograd pass and is not silently substituted by signal autocorrelation.",
        },
        "decision": {
            "slow_information_gate": "NOT_PASSED",
            "reason": "the only event label is an energy-weighted IBS surrogate, not a physical state/basin crossing; therefore no learned signal may yet be called a physical crossing predictor or production slow information.",
            "online_torchforce_or_mts_promotion": "STOP",
            "next_allowed_experiment": "independent physical/overlap event definition and/or a separately preregistered cheap offline route; no production wiring from this audit",
            "independent_surrogate_prediction_candidates": [
                "teacher_latent_pc1", "teacher_latent_norm",
                "primary_torsion_sin", "primary_torsion_cos",
                "secondary_torsion_sin", "secondary_torsion_cos",
                "VAL251_chi1_sin", "VAL251_chi1_cos",
            ],
            "tautological_or_target_derived_diagnostics": [
                "student_scalar_direct_gap", "state1_state0_gap_reduced",
            ],
        },
    }
    audit["audit_sha256"] = hashlib.sha256(json.dumps(audit, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    atomic_json(output_dir / "EXP-016_temporal_audit.json", audit)

    summary_lines = [
        "# EXP-016 temporal audit summary",
        "",
        "Status: `COMPLETED_OFFLINE_TEMPORAL_AUDIT`; physical crossing claim: **UNAVAILABLE**.",
        "",
        f"- Inputs: 3 continuous runs, {manifest['raw_frame_count_total']} frames total, measured save interval `{manifest_runs[0]['delta_t_save_ps']:.6g} ps`.",
        f"- Alignment: trajectory/ledger/latent frame identity passed; manifest SHA-256 `{manifest['manifest_sha256']}`.",
        "- Event label: MM-ledger `argmin(target_interaction - f_k)` adjacent `0↔1` switch. This is an energy-weighted surrogate, not a physical alchemical state trajectory or replica round trip.",
        f"- Prediction: leave-one-run-out, continuous run splits, horizons `{', '.join(str(h) + ' frames' for h in horizons)}`; surrogate results are exploratory only.",
        "- Hydration: existing files contain only per-run summary statistics, not a per-frame series; it was not reconstructed or promoted silently.",
        "- Force(signal): not measured because no teacher coordinate gradients/force series were cached.",
        "",
        "## Decision",
        "",
        "No candidate is promoted to physical learned slow information. Do not restart the closed real-time TorchForce route or promote MTS from this audit. A future promotion requires an independently defined physical/overlap event and a separately qualified cheap Hamiltonian route.",
        "",
        "Machine-readable outputs: `EXP-016_data_manifest.json` and `EXP-016_temporal_audit.json`.",
    ]
    (output_dir / "EXP-016_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "manifest_sha256": manifest["manifest_sha256"],
        "audit_sha256": audit["audit_sha256"],
        "physical_crossing_claim": audit["physical_crossing_claim"],
        "surrogate_event_count_first_horizon": event_total,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
