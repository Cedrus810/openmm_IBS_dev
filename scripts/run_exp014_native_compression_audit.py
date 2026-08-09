#!/usr/bin/env python3
"""EXP-014 feasibility audit: compress the frozen student into a native pair/radial basis.

This is an offline screening experiment.  It does not build an OpenMM Force and
does not run MD.  For each leave-one-run-out fold, the held-out run's matching
seed-0 direct-gap checkpoint supplies the target scalar.  A linear

    B(R) = sum_{ligand/environment typed pairs} sum_p a[pair,p] phi_p(r)

model is fitted on the other two continuous runs and evaluated on the held-out
run.  The feature map is exactly a typed-pair radial RBF multiplied by the
existing quintic C2 cutoff.  An intercept is used during regression only; the
reported native form omits it because a frame-independent constant does not
contribute to gap variance and has no force.

The audit asks whether the cheap analytic form can preserve the already
observed student gap-variance signal.  It is not evidence for physical
crossing prediction, slow information, or production Hamiltonian correctness.
Those require a separate OpenMM energy/force equivalence and qualification
experiment after this screen passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_dataset(path: Path):
    import numpy as np

    report_path = path.with_name(path.stem + "_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETED_GEOMETRY_AND_LEDGER_JOIN_NOT_TRAINED":
        raise RuntimeError(f"unexpected dataset status: {report.get('status')!r}")
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    return report, arrays, report_path


def _load_model(checkpoint_path: Path):
    import torch

    from local_residual.student import build_local_residual_student

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("variant") != "direct_gap":
        raise RuntimeError(f"{checkpoint_path}: only direct_gap checkpoints are allowed")
    model = build_local_residual_student(payload["type_vocabulary"], **payload["model_kwargs"])
    model = model.to(dtype=torch.float64)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def _run_rows(arrays, partition_labels: list[str], run_id: str):
    import numpy as np

    label = partition_labels.index(run_id)
    return np.nonzero(arrays["partition_index"] == label)[0]


def _student_target_for_rows(model, arrays, rows):
    """Evaluate one frozen student checkpoint without retaining trajectory data."""

    import numpy as np
    import torch

    from local_residual.student import reindex_ligand_environment_edges

    ligand_topology = arrays["ligand_topology_indices"].astype(np.int64)
    ligand_z = arrays["ligand_atomic_numbers"].astype(np.int64)
    all_z = arrays["all_topology_atomic_numbers"].astype(np.int64)
    ligand_type = model.atomic_numbers_to_type_index(ligand_z.tolist())
    offsets = arrays["edge_offsets"]
    edge_ligand = arrays["edge_ligand_topology"]
    edge_environment = arrays["edge_environment_topology"]
    edge_distance = arrays["edge_distance_angstrom"]
    output = np.empty((len(rows),), dtype=np.float64)

    with torch.no_grad():
        for position, row in enumerate(rows.tolist()):
            start, end = int(offsets[row]), int(offsets[row + 1])
            lig = torch.as_tensor(edge_ligand[start:end], dtype=torch.int64)
            env = torch.as_tensor(edge_environment[start:end], dtype=torch.int64)
            distance = torch.as_tensor(edge_distance[start:end], dtype=torch.float64)
            reindexed = reindex_ligand_environment_edges(ligand_topology.tolist(), lig, env)
            env_topology = reindexed["environment_topology_indices"].tolist()
            env_type = model.atomic_numbers_to_type_index(all_z[env_topology].tolist())
            value = model(
                ligand_type,
                env_type,
                reindexed["edge_ligand_local"],
                reindexed["edge_environment_local"],
                distance,
            )
            output[position] = float(value.item())
    return output


def _quintic_cutoff(distance, inner: float, outer: float):
    import numpy as np

    scaled = (distance - inner) / (outer - inner)
    transition = 1.0 - 10.0 * scaled**3 + 15.0 * scaled**4 - 6.0 * scaled**5
    return np.where(distance <= inner, 1.0, np.where(distance >= outer, 0.0, transition))


def _build_pair_radial_features(arrays, rows, *, n_radial_basis: int, inner: float, outer: float):
    """Aggregate edge-level pair/RBF terms into one feature row per frame."""

    import numpy as np

    type_vocabulary = [int(value) for value in arrays["type_vocabulary"].tolist()]
    type_index = {z: i for i, z in enumerate(type_vocabulary)}
    n_types = len(type_vocabulary)
    n_pairs = n_types * n_types
    centers = np.linspace(0.0, outer, n_radial_basis, dtype=np.float64)
    width = outer / max(n_radial_basis - 1, 1)
    offsets = arrays["edge_offsets"]
    edge_ligand = arrays["edge_ligand_topology"]
    edge_environment = arrays["edge_environment_topology"]
    distances = arrays["edge_distance_angstrom"]
    ligand_z_by_topology = np.full(arrays["all_topology_atomic_numbers"].shape, -1, dtype=np.int64)
    ligand_z_by_topology[arrays["ligand_topology_indices"].astype(np.int64)] = arrays["ligand_atomic_numbers"]
    all_z = arrays["all_topology_atomic_numbers"].astype(np.int64)
    feature_count = n_pairs * n_radial_basis
    features = np.zeros((len(rows), feature_count), dtype=np.float64)
    edge_count = np.zeros((len(rows),), dtype=np.int64)

    for out_index, row in enumerate(rows.tolist()):
        start, end = int(offsets[row]), int(offsets[row + 1])
        if end == start:
            continue
        lig_z = ligand_z_by_topology[edge_ligand[start:end]]
        env_z = all_z[edge_environment[start:end]]
        try:
            lig_type = np.fromiter((type_index[int(z)] for z in lig_z), dtype=np.int64, count=len(lig_z))
            env_type = np.fromiter((type_index[int(z)] for z in env_z), dtype=np.int64, count=len(env_z))
        except KeyError as exc:
            raise RuntimeError(f"edge atomic number {exc.args[0]} is outside type_vocabulary") from exc
        pair_id = lig_type * n_types + env_type
        distance = distances[start:end]
        envelope = _quintic_cutoff(distance, inner, outer)
        edge_count[out_index] = len(distance)
        for radial_index, center in enumerate(centers):
            radial = np.exp(-0.5 * ((distance - center) / width) ** 2) * envelope
            features[out_index, radial_index::n_radial_basis] = np.bincount(
                pair_id, weights=radial, minlength=n_pairs
            )
    return features, {
        "type_vocabulary": type_vocabulary,
        "n_types": n_types,
        "n_pairs": n_pairs,
        "n_radial_basis": n_radial_basis,
        "centers_angstrom": centers.tolist(),
        "radial_width_angstrom": float(width),
        "inner_cutoff_angstrom": inner,
        "outer_cutoff_angstrom": outer,
        "edge_count_min": int(edge_count.min()) if len(edge_count) else 0,
        "edge_count_max": int(edge_count.max()) if len(edge_count) else 0,
        "edge_count_mean": float(edge_count.mean()) if len(edge_count) else 0.0,
    }


def _fit_ridge_inner_loro(X_train, y_train, run_labels, alpha_grid):
    import numpy as np

    unique_runs = list(dict.fromkeys(run_labels))
    if len(unique_runs) != 2:
        raise RuntimeError("EXP-014 outer fold must have exactly two training runs")

    def fit(X_fit, y_fit, alpha):
        mean = X_fit.mean(axis=0)
        scale = X_fit.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        Xs = (X_fit - mean) / scale
        yc = y_fit - y_fit.mean()
        gram = Xs.T @ Xs
        rhs = Xs.T @ yc
        coef_std = np.linalg.solve(gram + alpha * np.eye(Xs.shape[1]), rhs)
        coef = coef_std / scale
        intercept = float(y_fit.mean() - mean @ coef)
        return coef, intercept

    scores = []
    for alpha in alpha_grid:
        fold_rmse = []
        for validation_run in unique_runs:
            fit_mask = np.asarray([run != validation_run for run in run_labels], dtype=bool)
            val_mask = ~fit_mask
            coef, intercept = fit(X_train[fit_mask], y_train[fit_mask], alpha)
            prediction = X_train[val_mask] @ coef + intercept
            fold_rmse.append(float(np.sqrt(np.mean((prediction - y_train[val_mask]) ** 2))))
        scores.append({"alpha": float(alpha), "inner_rmse": float(np.mean(fold_rmse)), "fold_rmse": fold_rmse})
    best = min(scores, key=lambda item: (item["inner_rmse"], item["alpha"]))
    return float(best["alpha"]), scores


def _fit_final(X_train, y_train, alpha):
    import numpy as np

    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    Xs = (X_train - mean) / scale
    yc = y_train - y_train.mean()
    coef_std = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(Xs.shape[1]), Xs.T @ yc)
    coef = coef_std / scale
    intercept = float(y_train.mean() - mean @ coef)
    return coef, intercept, mean, scale


def _gap_variance_loss(gaps, basis, log_weights, delta_a):
    import torch

    from local_residual.loss import bidirectional_gap_variance_loss

    return float(
        bidirectional_gap_variance_loss(
            torch.tensor(gaps, dtype=torch.float64),
            torch.tensor(basis, dtype=torch.float64),
            torch.tensor(delta_a, dtype=torch.float64),
            torch.tensor(log_weights, dtype=torch.float64),
            partition_index=None,
            energy_regularization_coefficient=0.0,
            force_regularization_coefficient=0.0,
        )["gap_variance_loss"].item()
    )


def _r2(y_true, y_pred):
    import numpy as np

    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator) if denominator > 0 else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="output/outer_lambda_exp012/student_training_dataset.npz")
    parser.add_argument("--checkpoint-dir", default="output/outer_lambda_exp012/student_checkpoints")
    parser.add_argument("--output", default="output/outer_lambda_exp014_native_compression_audit/EXP-014_native_compression_audit.json")
    parser.add_argument("--n-radial-basis", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--inner-cutoff-angstrom", type=float, default=4.0)
    parser.add_argument("--outer-cutoff-angstrom", type=float, default=5.0)
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=[1e-8, 1e-6, 1e-4, 1e-2, 1.0])
    args = parser.parse_args(argv)

    import numpy as np

    dataset_path = (ROOT / args.dataset).resolve()
    checkpoint_dir = (ROOT / args.checkpoint_dir).resolve()
    output_path = (ROOT / args.output).resolve()
    if output_path.exists():
        raise SystemExit(f"refusing to overwrite existing output: {output_path}")
    if not 0.0 < args.inner_cutoff_angstrom < args.outer_cutoff_angstrom:
        raise SystemExit("cutoffs must satisfy 0 < inner < outer")
    if any(value <= 0 for value in args.n_radial_basis):
        raise SystemExit("n-radial-basis values must be positive")

    dataset_report, arrays, dataset_report_path = _load_dataset(dataset_path)
    partition_labels = list(dataset_report["run_id_by_partition_index"])
    if partition_labels != ["hard_window0_run1", "hard_window0_run2", "hard_window0_run3"]:
        raise RuntimeError(f"unexpected registered run order: {partition_labels}")
    if len(arrays["partition_index"]) != 1500:
        raise RuntimeError("EXP-014 is sealed to the 3x500-frame EXP-012 dataset")

    all_rows = np.arange(len(arrays["partition_index"]), dtype=np.int64)
    features_by_basis = {}
    feature_specs = {}
    for n_radial in args.n_radial_basis:
        features_by_basis[n_radial], feature_specs[n_radial] = _build_pair_radial_features(
            arrays,
            all_rows,
            n_radial_basis=n_radial,
            inner=args.inner_cutoff_angstrom,
            outer=args.outer_cutoff_angstrom,
        )

    fold_reports = []
    checkpoint_records = {}
    for held_out_run in partition_labels:
        checkpoint_path = checkpoint_dir / f"{held_out_run}__direct_gap__seed0.pt"
        model, payload = _load_model(checkpoint_path)
        if payload.get("held_out_run_id") != held_out_run:
            raise RuntimeError(f"{checkpoint_path}: held_out_run_id mismatch")
        train_runs = [run for run in partition_labels if run != held_out_run]
        checkpoint_records[held_out_run] = {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "held_out_run_id": payload["held_out_run_id"],
            "training_run_ids": payload["training_run_ids"],
        }
        targets = _student_target_for_rows(model, arrays, all_rows)
        heldout_mask = np.asarray([run == held_out_run for run in np.asarray(partition_labels, dtype=object)[arrays["partition_index"]]], dtype=bool)
        train_mask = ~heldout_mask
        heldout_rows = all_rows[heldout_mask]
        train_rows = all_rows[train_mask]
        run_label_by_row = np.asarray(partition_labels, dtype=object)[arrays["partition_index"]]
        fold_result = {
            "held_out_run_id": held_out_run,
            "training_run_ids": train_runs,
            "checkpoint": checkpoint_records[held_out_run],
            "student_target_mean": float(targets.mean()),
            "student_target_std": float(targets.std()),
            "radial_candidates": {},
        }
        gaps = arrays["adjacent_gap_reduced"]
        log_weights = arrays["log_importance_unnormalized"]
        delta_a = arrays["delta_A"]
        student_loss = _gap_variance_loss(gaps[heldout_mask], targets[heldout_mask], log_weights[heldout_mask], delta_a)
        baseline_loss = _gap_variance_loss(gaps[heldout_mask], np.zeros(np.sum(heldout_mask)), log_weights[heldout_mask], delta_a)

        for n_radial in args.n_radial_basis:
            X = features_by_basis[n_radial]
            alpha, alpha_scores = _fit_ridge_inner_loro(
                X[train_mask], targets[train_mask], run_label_by_row[train_mask].tolist(), args.ridge_grid
            )
            coef, intercept, mean, scale = _fit_final(X[train_mask], targets[train_mask], alpha)
            affine_prediction = X[heldout_mask] @ coef + intercept
            native_prediction = X[heldout_mask] @ coef
            train_prediction = X[train_mask] @ coef + intercept
            compression_loss = _gap_variance_loss(gaps[heldout_mask], affine_prediction, log_weights[heldout_mask], delta_a)
            native_loss = _gap_variance_loss(gaps[heldout_mask], native_prediction, log_weights[heldout_mask], delta_a)
            student_improvement = baseline_loss - student_loss
            compression_improvement = baseline_loss - compression_loss
            retention = compression_improvement / student_improvement if student_improvement > 0 else None
            fold_result["radial_candidates"][str(n_radial)] = {
                "n_features": int(X.shape[1]),
                "alpha": alpha,
                "alpha_inner_selection": alpha_scores,
                "reconstruction": {
                    "held_out_rmse_affine": float(np.sqrt(np.mean((affine_prediction - targets[heldout_mask]) ** 2))),
                    "held_out_r2_affine": _r2(targets[heldout_mask], affine_prediction),
                    "held_out_max_abs_affine": float(np.max(np.abs(affine_prediction - targets[heldout_mask]))),
                    "held_out_rmse_native_no_intercept": float(np.sqrt(np.mean((native_prediction - targets[heldout_mask]) ** 2))),
                    "train_rmse_affine": float(np.sqrt(np.mean((train_prediction - targets[train_mask]) ** 2))),
                    "intercept_reduced": intercept,
                },
                "gap_variance": {
                    "held_out_baseline_loss": baseline_loss,
                    "held_out_student_loss": student_loss,
                    "held_out_compressed_loss_affine": compression_loss,
                    "held_out_compressed_loss_native_no_intercept": native_loss,
                    "student_improvement_over_baseline": student_improvement,
                    "compressed_improvement_over_baseline": compression_improvement,
                    "retained_student_improvement": retention,
                },
                "coefficient_diagnostics": {
                    "nonzero_abs_gt_1e-8": int(np.count_nonzero(np.abs(coef) > 1e-8)),
                    "l1": float(np.sum(np.abs(coef))),
                    "max_abs": float(np.max(np.abs(coef))),
                    "mean": float(mean.mean()),
                    "scale_min": float(scale.min()),
                    "scale_max": float(scale.max()),
                },
                "native_intercept_omitted_gap_loss_difference": float(native_loss - compression_loss),
            }
        fold_reports.append(fold_result)

    aggregate = {}
    for n_radial in args.n_radial_basis:
        entries = [fold["radial_candidates"][str(n_radial)] for fold in fold_reports]
        retentions = [entry["gap_variance"]["retained_student_improvement"] for entry in entries]
        r2_values = [entry["reconstruction"]["held_out_r2_affine"] for entry in entries]
        aggregate[str(n_radial)] = {
            "mean_retained_student_improvement": float(np.mean(retentions)),
            "min_retained_student_improvement": float(np.min(retentions)),
            "all_retention_ge_0_80": bool(all(value >= 0.80 for value in retentions)),
            "mean_held_out_r2": float(np.mean(r2_values)),
            "min_held_out_r2": float(np.min(r2_values)),
            "all_r2_ge_0_90": bool(all(value >= 0.90 for value in r2_values)),
            "screening_pass": bool(all(value >= 0.80 for value in retentions) and all(value >= 0.90 for value in r2_values)),
        }
    passing = [int(n) for n, value in aggregate.items() if value["screening_pass"]]
    decision = {
        "screening_thresholds_frozen_before_run": {
            "held_out_r2_affine_min": 0.90,
            "retained_student_gap_variance_improvement_min": 0.80,
            "requirement": "one shared radial-basis size passes all three LORO folds",
        },
        "screening_passed": bool(passing),
        "smallest_passing_n_radial_basis": min(passing) if passing else None,
        "openmm_force_qualification": "NOT_STARTED",
        "production_promotion": "STOP",
        "reason": "This is an offline compression screen; OpenMM energy/force equivalence and Hamiltonian qualification remain separate gates.",
    }
    report = {
        "experiment_id": "EXP-014",
        "status": "COMPLETED_OFFLINE_NATIVE_COMPRESSION_FEASIBILITY_AUDIT",
        "schema_version": "exp014-native-compression-audit-v1",
        "protocol": {
            "target": "matching held-out-run seed-0 direct_gap student scalar",
            "fit_split": "leave-one-run-out; fit on two complete runs; inner two-run LORO selects ridge alpha",
            "native_form": "typed ligand/environment pair sum of RBF(distance)*quintic_C2_cutoff(distance); intercept omitted from native form",
            "radial_basis_grid": [int(value) for value in args.n_radial_basis],
            "inner_cutoff_angstrom": args.inner_cutoff_angstrom,
            "outer_cutoff_angstrom": args.outer_cutoff_angstrom,
            "ridge_grid": [float(value) for value in args.ridge_grid],
            "no_md": True,
            "no_openmm_force_qualification": True,
        },
        "inputs": {
            "dataset_path": str(dataset_path),
            "dataset_sha256": _sha256_file(dataset_path),
            "dataset_report_path": str(dataset_report_path),
            "dataset_report_sha256": _sha256_file(dataset_report_path),
            "registered_runs": partition_labels,
            "raw_frame_count": int(len(arrays["partition_index"])),
        },
        "feature_spec": feature_specs,
        "checkpoint_records": checkpoint_records,
        "aggregate": aggregate,
        "folds": fold_reports,
        "decision": decision,
    }
    _write_json(output_path, report)
    print(json.dumps({"output": str(output_path), "screening_passed": decision["screening_passed"], "smallest_passing_n_radial_basis": decision["smallest_passing_n_radial_basis"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
