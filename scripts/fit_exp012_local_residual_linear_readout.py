#!/usr/bin/env python
"""DEC-030(c) step 2: linear/ridge readout, held-out gap-variance vs B=0.

This fits ``basis_reduced = w^T * standardize(pooled_latent)`` -- a plain
linear map, no intercept -- and evaluates it with leave-one-run-out
cross-validation against the uncorrected (``B=0``) baseline, reusing
``local_residual.loss.bidirectional_gap_variance_loss`` directly rather than
re-deriving a parallel closed form.

Per the outer-lambda-neural-basis plan ("第一轮冻结全局 A_k，只训练表示和
readout；禁止同时学习 A_k 造成尺度简并"), ``A_k``/``delta_A`` are frozen
inputs from the join step, never fit here. This is also why there is no
intercept: an additive constant ``b`` enters ``corrected_gap`` only as a
per-edge constant shared by every frame, and ``Var(X + constant) = Var(X)``,
so ``b`` has *zero* effect on ``gap_variance_loss`` -- it is a parameter with
nothing to learn from this objective, not an oversight.

The ridge coefficient is selected by a leakage-free *inner* 2-way split using
only the two training runs of each outer fold (never touching the held-out
run), then the readout is refit on both training runs combined and evaluated
once on the true held-out run. The optimization problem is convex (affine
readout inside a variance objective, plus an L2 ridge term), so LBFGS
converges to the unique global optimum -- no learning-rate tuning, no local
minima to worry about.

Explicit about what this is and is not: it fits a linear readout (a few
thousand scalar weights), evaluated purely offline against cached, frozen
MACE latents. It does not train MACE, does not learn A_k, and does not touch
`LocalResidualStudent` (DEC-030 step d) -- see the policy block in the output
report for machine-checkable flags.
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

from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.loss import bidirectional_gap_variance_loss  # noqa: E402


class ReadoutFitError(ValueError):
    """The joined input or the fit configuration is invalid."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _mask_for_labels(partition_tensor: Any, labels: list[int]) -> Any:
    import torch

    labels_tensor = torch.tensor(labels, dtype=partition_tensor.dtype)
    return (partition_tensor.unsqueeze(-1) == labels_tensor).any(dim=-1)


def _standardize(train_features: Any, *other_features: Any):
    import torch

    mean = train_features.mean(dim=0, keepdim=True)
    std = train_features.std(dim=0, keepdim=True)
    std = torch.where(std > 0, std, torch.ones_like(std))
    standardized_train = (train_features - mean) / std
    standardized_others = [(features - mean) / std for features in other_features]
    return standardized_train, standardized_others


def _fit_linear_readout(
    features: Any, gaps: Any, log_weights: Any, delta_a: Any,
    partition_index: Any, ridge_coefficient: float, *, max_iterations: int = 200,
) -> Any:
    """Fit w minimizing gap_variance_loss + ridge_coefficient * ||w||^2.

    Convex in w (delta_A is fixed), so LBFGS with strong Wolfe line search
    converges to the unique global optimum from a zero start.
    """
    import torch

    weight = torch.zeros(features.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight], max_iter=max_iterations, line_search_fn="strong_wolfe",
        tolerance_grad=1e-12, tolerance_change=1e-14,
    )

    def closure():
        optimizer.zero_grad()
        basis = features @ weight
        result = bidirectional_gap_variance_loss(
            gaps, basis, delta_a, log_weights,
            partition_index=partition_index,
            energy_regularization_coefficient=0.0,
            force_regularization_coefficient=0.0,
        )
        loss = result["gap_variance_loss"] + ridge_coefficient * (weight * weight).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weight.detach()


def _evaluate_gap_variance(
    features: Any, gaps: Any, log_weights: Any, delta_a: Any, weight: Any | None,
) -> float:
    import torch

    with torch.no_grad():
        basis = (
            features @ weight if weight is not None
            else torch.zeros(features.shape[0], dtype=features.dtype)
        )
        result = bidirectional_gap_variance_loss(
            gaps, basis, delta_a, log_weights,
            partition_index=None,
            energy_regularization_coefficient=0.0,
            force_regularization_coefficient=0.0,
        )
    return float(result["gap_variance_loss"].item())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joined", required=True,
        help="output of scripts/join_exp012_teacher_latent_cache_with_ledger.py",
    )
    parser.add_argument(
        "--ridge-grid", type=float, nargs="+",
        default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
        help="candidate ridge coefficients, selected per outer fold by inner 2-way CV",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")
    if len(args.ridge_grid) < 1:
        parser.error("--ridge-grid must contain at least one candidate")

    import numpy as np
    import torch

    joined_path = Path(args.joined)
    joined_report_path = joined_path.with_name(joined_path.stem + "_report.json")
    if not joined_report_path.is_file():
        raise ReadoutFitError(f"cannot find matching join report: {joined_report_path}")
    joined_report = json.loads(joined_report_path.read_text(encoding="utf-8"))
    if joined_report.get("status") != "COMPLETED_JOIN_ONLY_NOT_FIT":
        raise ReadoutFitError("--joined does not point to a completed join report")

    with np.load(joined_path) as joined:
        pooled_latent = joined["pooled_latent"].astype(np.float64)
        adjacent_gap_reduced = joined["adjacent_gap_reduced"]
        log_importance_unnormalized = joined["log_importance_unnormalized"]
        partition_index_np = joined["partition_index"]
        delta_a_np = joined["delta_A"]

    run_ids = joined_report["run_id_by_partition_index"]
    run_count = len(run_ids)
    if run_count < 3:
        raise ReadoutFitError("leave-one-run-out requires at least 3 runs in the join")

    delta_a = torch.tensor(delta_a_np, dtype=torch.float64)
    gaps_all = torch.tensor(adjacent_gap_reduced, dtype=torch.float64)
    log_weights_all = torch.tensor(log_importance_unnormalized, dtype=torch.float64)
    features_all_raw = torch.tensor(pooled_latent, dtype=torch.float64)
    partition_all = torch.tensor(partition_index_np, dtype=torch.int64)

    fold_reports = []
    for held_out_label in range(run_count):
        train_labels = [label for label in range(run_count) if label != held_out_label]
        train_mask = _mask_for_labels(partition_all, train_labels)
        held_out_mask = partition_all == held_out_label
        remap = {label: index for index, label in enumerate(train_labels)}
        train_partition = torch.tensor(
            [remap[int(value)] for value in partition_all[train_mask].tolist()], dtype=torch.int64
        )

        inner_results = []
        best_ridge_coefficient = None
        best_inner_loss = math.inf
        for ridge_coefficient in args.ridge_grid:
            inner_losses = []
            for inner_val_label in train_labels:
                inner_train_labels = [label for label in train_labels if label != inner_val_label]
                inner_train_mask = _mask_for_labels(partition_all, inner_train_labels)
                inner_val_mask = partition_all == inner_val_label

                inner_features_train, (inner_features_val,) = _standardize(
                    features_all_raw[inner_train_mask], features_all_raw[inner_val_mask]
                )
                weight = _fit_linear_readout(
                    inner_features_train, gaps_all[inner_train_mask], log_weights_all[inner_train_mask],
                    delta_a, None, ridge_coefficient,
                )
                inner_losses.append(
                    _evaluate_gap_variance(
                        inner_features_val, gaps_all[inner_val_mask], log_weights_all[inner_val_mask],
                        delta_a, weight,
                    )
                )
            mean_inner_loss = sum(inner_losses) / len(inner_losses)
            inner_results.append(
                {"ridge_coefficient": ridge_coefficient, "mean_inner_validation_gap_variance_loss": mean_inner_loss}
            )
            if mean_inner_loss < best_inner_loss:
                best_inner_loss = mean_inner_loss
                best_ridge_coefficient = ridge_coefficient

        features_train, (features_held_out,) = _standardize(
            features_all_raw[train_mask], features_all_raw[held_out_mask]
        )
        final_weight = _fit_linear_readout(
            features_train, gaps_all[train_mask], log_weights_all[train_mask],
            delta_a, train_partition, best_ridge_coefficient,
        )

        held_out_baseline_loss = _evaluate_gap_variance(
            features_held_out, gaps_all[held_out_mask], log_weights_all[held_out_mask], delta_a, None,
        )
        held_out_fitted_loss = _evaluate_gap_variance(
            features_held_out, gaps_all[held_out_mask], log_weights_all[held_out_mask], delta_a, final_weight,
        )
        relative_improvement = (
            (held_out_baseline_loss - held_out_fitted_loss) / held_out_baseline_loss
            if held_out_baseline_loss > 0 else None
        )
        fold_reports.append(
            {
                "held_out_run_id": run_ids[held_out_label],
                "training_run_ids": [run_ids[label] for label in train_labels],
                "selected_ridge_coefficient": best_ridge_coefficient,
                "inner_cv_grid": inner_results,
                "held_out_baseline_gap_variance_loss": held_out_baseline_loss,
                "held_out_fitted_gap_variance_loss": held_out_fitted_loss,
                "held_out_relative_improvement": relative_improvement,
                "held_out_improved": held_out_fitted_loss < held_out_baseline_loss,
            }
        )

    all_folds_improved = all(fold["held_out_improved"] for fold in fold_reports)
    any_fold_improved = any(fold["held_out_improved"] for fold in fold_reports)
    relative_improvements = [
        fold["held_out_relative_improvement"] for fold in fold_reports
        if fold["held_out_relative_improvement"] is not None
    ]
    mean_relative_improvement = (
        sum(relative_improvements) / len(relative_improvements) if relative_improvements else None
    )

    body = {
        "schema_version": "exp012-local-residual-linear-readout-v1",
        "status": "COMPLETED_HELD_OUT_EVALUATION",
        "readout_type": "linear_no_intercept",
        "feature": "pooled_latent_mean_over_ligand_atoms",
        "ridge_grid": list(args.ridge_grid),
        "folds": fold_reports,
        "all_folds_improved_over_baseline": all_folds_improved,
        "any_fold_improved_over_baseline": any_fold_improved,
        "mean_relative_improvement": mean_relative_improvement,
        "joined_input": {
            "path": str(joined_path.resolve()),
            "sha256": _sha256_file(joined_path),
            "report_sha256": joined_report["report_sha256"],
        },
        "policy": {
            "provisional_not_sealed": True,
            "scientific_qualification": False,
            "decision_reference": "DEC-030(c)",
            "a_k_frozen": True,
            "a_k_learned": False,
            "mace_encoder_trained": False,
            "linear_readout_fit": True,
            "local_residual_student_trained": False,
            "recommendation": "proceed_to_distillation_dec_030_d_only_if_any_fold_improved",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
