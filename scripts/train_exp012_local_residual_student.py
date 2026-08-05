#!/usr/bin/env python
"""DEC-039 / D1: offline `LocalResidualStudent` fitting and go/no-go.

Trains two variants per DEC-037 (d0) item 4's required control, on the exact
same leave-one-run-out folds and early-stopping protocol frozen in DEC-039:

- ``direct_gap``: optimizes `local_residual.loss.bidirectional_gap_variance_loss`
  alone, from real per-frame coordinates (via
  `scripts/build_exp012_student_training_dataset.py`'s cached geometry).
- ``distilled``: the same objective plus an auxiliary MSE term pulling the
  student's `basis_reduced` toward a per-fold refit of the teacher's linear
  readout (DEC-034/035/036's `basis_reduced = w^T * standardize(pooled_latent)`,
  refit fresh on this fold's two training runs only -- never the held-out
  run -- reusing `fit_exp012_local_residual_linear_readout.py`'s own
  `_fit_linear_readout`/`_standardize` rather than a second implementation).
  The teacher is explicitly NOT treated as unconditional ground truth: only
  ``direct_gap``'s real held-out result decides whether the representation
  works at all, and ``distilled`` must not be worse than ``direct_gap`` on the
  same held-out run for distillation itself to be considered useful.

Early stopping (DEC-039): within each fold's two TRAINING runs, model
selection uses only the trailing contiguous time-block frames the dataset
already flagged `is_early_stop_validation` -- never the held-out third run,
which is touched exactly once, after training/selection are both finished, for
the final reported number.

This is D1 only: offline fitting and held-out gap-variance evaluation. No
TorchScript, OpenMM, CUDA deployment, or NVT run happens here (those are
D2/D3/D4, each gated on this stage's result per DEC-037).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.loss import bidirectional_gap_variance_loss  # noqa: E402
from local_residual.student import (  # noqa: E402
    build_local_residual_student,
    count_trainable_parameters,
    reindex_ligand_environment_edges,
)

_READOUT_MODULE_PATH = ROOT / "scripts" / "fit_exp012_local_residual_linear_readout.py"
_READOUT_SPEC = importlib.util.spec_from_file_location(
    "fit_exp012_local_residual_linear_readout", _READOUT_MODULE_PATH
)
_readout_module = importlib.util.module_from_spec(_READOUT_SPEC)
sys.modules[_READOUT_SPEC.name] = _readout_module
_READOUT_SPEC.loader.exec_module(_readout_module)


class TrainError(RuntimeError):
    """A dataset/config input failed a fail-closed check."""


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


def _load_dataset(path: Path):
    import numpy as np

    report_path = path.with_name(path.stem + "_report.json")
    if not report_path.is_file():
        raise TrainError(f"cannot find matching dataset report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETED_GEOMETRY_AND_LEDGER_JOIN_NOT_TRAINED":
        raise TrainError("--dataset does not point to a completed build_exp012_student_training_dataset.py output")

    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files}
    return report, arrays


def _run_row_indices(arrays, partition_labels: list[str], run_id: str):
    import numpy as np

    label_index = partition_labels.index(run_id)
    return np.nonzero(arrays["partition_index"] == label_index)[0]


def _precompute_frames(arrays, row_indices, model):
    """Turn this run's CSR-flattened geometry rows into per-frame model inputs.

    Reindexing and atomic-number lookup depend only on fixed topology/type
    identity, not on any trainable parameter, so this is computed once before
    training (not re-derived every epoch) -- see `local_residual/student.py`'s
    module docstring on the cache/live-recompute split between D1 and D2.
    """

    import torch

    ligand_topology_indices = [int(value) for value in arrays["ligand_topology_indices"].tolist()]
    ligand_atomic_numbers = [int(value) for value in arrays["ligand_atomic_numbers"].tolist()]
    ligand_type_index = model.atomic_numbers_to_type_index(ligand_atomic_numbers)

    offsets = arrays["edge_offsets"]
    edge_ligand_topology_all = arrays["edge_ligand_topology"]
    edge_environment_topology_all = arrays["edge_environment_topology"]
    edge_distance_all = arrays["edge_distance_angstrom"]

    frames = []
    for row in row_indices.tolist():
        start, end = int(offsets[row]), int(offsets[row + 1])
        edge_ligand_topology = torch.tensor(edge_ligand_topology_all[start:end], dtype=torch.int64)
        edge_environment_topology = torch.tensor(edge_environment_topology_all[start:end], dtype=torch.int64)
        distance = torch.tensor(edge_distance_all[start:end], dtype=torch.float64)
        reindexed = reindex_ligand_environment_edges(
            ligand_topology_indices, edge_ligand_topology, edge_environment_topology
        )
        frames.append(
            {
                "ligand_type_index": ligand_type_index,
                "environment_topology_indices": reindexed["environment_topology_indices"],
                "edge_ligand_local": reindexed["edge_ligand_local"],
                "edge_environment_local": reindexed["edge_environment_local"],
                "distance": distance,
            }
        )
    return frames


def _resolve_environment_type_index(model, frame, atomic_number_by_topology_index: dict[int, int]):
    import torch

    if frame["environment_topology_indices"].numel() == 0:
        return torch.empty((0,), dtype=torch.int64)
    numbers = [
        atomic_number_by_topology_index[int(index)]
        for index in frame["environment_topology_indices"].tolist()
    ]
    return model.atomic_numbers_to_type_index(numbers)


def _forward_frames(model, frames, environment_type_index_cache: list):
    import torch

    outputs = []
    for frame, environment_type_index in zip(frames, environment_type_index_cache):
        outputs.append(
            model(
                frame["ligand_type_index"], environment_type_index,
                frame["edge_ligand_local"], frame["edge_environment_local"], frame["distance"],
            )
        )
    return torch.stack(outputs)


def _slice_ledger(arrays, row_indices):
    import torch

    return (
        torch.tensor(arrays["adjacent_gap_reduced"][row_indices], dtype=torch.float64),
        torch.tensor(arrays["log_importance_unnormalized"][row_indices], dtype=torch.float64),
    )


def _fit_teacher_distillation_target(
    *, teacher_arrays, teacher_partition_labels, train_run_ids, ridge_grid,
):
    """Refit the teacher's linear readout on this fold's two training runs only.

    Reuses `_fit_linear_readout`/`_standardize` from
    `fit_exp012_local_residual_linear_readout.py` directly (imported, not
    reimplemented) so the distillation target is exactly the same kind of
    object DEC-034/035/036 already validated -- just refit per-fold here
    instead of reusing a report-only summary (which never persisted the
    fitted weights).
    """

    import torch

    import numpy as np

    features_by_run = {}
    gaps_by_run = {}
    log_weights_by_run = {}
    for run_id in train_run_ids:
        rows = teacher_partition_labels.index(run_id)
        mask = teacher_arrays["partition_index"] == rows
        features_by_run[run_id] = torch.tensor(
            teacher_arrays["pooled_latent"][mask].astype(np.float64), dtype=torch.float64
        )
        gaps_by_run[run_id] = torch.tensor(teacher_arrays["adjacent_gap_reduced"][mask], dtype=torch.float64)
        log_weights_by_run[run_id] = torch.tensor(
            teacher_arrays["log_importance_unnormalized"][mask], dtype=torch.float64
        )
    delta_a = torch.tensor(teacher_arrays["delta_A"], dtype=torch.float64)

    features_train_raw = torch.cat([features_by_run[run_id] for run_id in train_run_ids], dim=0)
    gaps_train = torch.cat([gaps_by_run[run_id] for run_id in train_run_ids], dim=0)
    log_weights_train = torch.cat([log_weights_by_run[run_id] for run_id in train_run_ids], dim=0)
    partition_train = torch.cat(
        [
            torch.full((features_by_run[run_id].shape[0],), label, dtype=torch.int64)
            for label, run_id in enumerate(train_run_ids)
        ]
    )

    best_ridge, best_loss = None, float("inf")
    for ridge_coefficient in ridge_grid:
        losses = []
        for held_out in train_run_ids:
            inner_train = [run_id for run_id in train_run_ids if run_id != held_out]
            inner_features_train_raw = torch.cat([features_by_run[run_id] for run_id in inner_train], dim=0)
            inner_features_train, (inner_features_val,) = _readout_module._standardize(
                inner_features_train_raw, features_by_run[held_out]
            )
            weight = _readout_module._fit_linear_readout(
                inner_features_train,
                torch.cat([gaps_by_run[run_id] for run_id in inner_train], dim=0),
                torch.cat([log_weights_by_run[run_id] for run_id in inner_train], dim=0),
                delta_a, None, ridge_coefficient,
            )
            losses.append(
                _readout_module._evaluate_gap_variance(
                    inner_features_val, gaps_by_run[held_out], log_weights_by_run[held_out], delta_a, weight,
                )
            )
        mean_loss = sum(losses) / len(losses)
        if mean_loss < best_loss:
            best_loss, best_ridge = mean_loss, ridge_coefficient

    features_train, _ = _readout_module._standardize(features_train_raw)
    weight = _readout_module._fit_linear_readout(
        features_train, gaps_train, log_weights_train, delta_a, partition_train, best_ridge,
    )
    mean = features_train_raw.mean(dim=0, keepdim=True)
    std = features_train_raw.std(dim=0, keepdim=True)
    std = torch.where(std > 0, std, torch.ones_like(std))

    return weight, mean, std, best_ridge


def _train_one_model(
    *, variant: str, seed: int, type_vocabulary: list[int], model_kwargs: dict,
    train_frames, train_environment_types,
    val_frames, val_environment_types,
    train_gaps, train_log_weights, train_partition,
    val_gaps, val_log_weights, val_partition,
    delta_a, max_epoch: int, patience: int,
    distillation_target_train=None, distillation_weight: float = 0.0,
    log=lambda message: None,
):
    import copy
    import time
    import torch

    torch.manual_seed(seed)
    model = build_local_residual_student(type_vocabulary, **model_kwargs).to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    started = time.perf_counter()

    for epoch in range(1, max_epoch + 1):
        model.train()
        optimizer.zero_grad()
        basis_train = _forward_frames(model, train_frames, train_environment_types)
        result = bidirectional_gap_variance_loss(
            train_gaps, basis_train, delta_a, train_log_weights,
            partition_index=train_partition,
            energy_regularization_coefficient=1e-4, force_regularization_coefficient=0.0,
        )
        loss = result["loss"]
        if variant == "distilled":
            loss = loss + distillation_weight * (basis_train - distillation_target_train).square().mean()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            basis_val = _forward_frames(model, val_frames, val_environment_types)
            val_loss = bidirectional_gap_variance_loss(
                val_gaps, basis_val, delta_a, val_log_weights,
                partition_index=val_partition,
                energy_regularization_coefficient=0.0, force_regularization_coefficient=0.0,
            )["gap_variance_loss"].item()
        history.append({"epoch": epoch, "train_loss": float(result["gap_variance_loss"].item()), "val_loss": val_loss})

        if val_loss < best_val_loss - 1e-12:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                log(
                    f"      early stop at epoch {epoch} (best epoch {best_epoch}, "
                    f"val_loss={best_val_loss:.6g}, {time.perf_counter() - started:.1f}s)"
                )
                break

        if epoch == 1 or epoch % 10 == 0:
            elapsed = time.perf_counter() - started
            log(
                f"      epoch {epoch}/{max_epoch}: train_loss={history[-1]['train_loss']:.6g} "
                f"val_loss={val_loss:.6g} best={best_val_loss:.6g} "
                f"({elapsed:.1f}s elapsed, {elapsed / epoch:.3f}s/epoch)"
            )

    if len(history) == max_epoch:
        log(f"      reached max_epoch={max_epoch} without early stopping (best epoch {best_epoch})")

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "early_stopped": len(history) < max_epoch,
        "best_validation_gap_variance_loss": best_val_loss,
        "n_trainable_parameters": count_trainable_parameters(model),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="output of build_exp012_student_training_dataset.py")
    parser.add_argument(
        "--teacher-joined", default=None,
        help="output of join_exp012_teacher_latent_cache_with_ledger.py; required unless --skip-distilled",
    )
    parser.add_argument("--skip-distilled", action="store_true", help="only train the direct_gap control")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-epoch", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument(
        "--teacher-ridge-grid", type=float, nargs="+",
        default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-interaction-blocks", type=int, default=2)
    parser.add_argument("--n-radial-basis", type=int, default=16)
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="if set, saves each (fold, variant, seed)'s trained state_dict here as "
             "<held_out_run_id>__<variant>__seed<seed>.pt, needed for D2 (autograd/finite-"
             "difference force check) since this script otherwise discards trained weights "
             "after computing metrics",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if len(args.seeds) < 3:
        parser.error("--seeds must supply at least 3 independent seeds (DEC-039 hard floor)")
    if not args.skip_distilled and args.teacher_joined is None:
        parser.error("--teacher-joined is required unless --skip-distilled is set")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir is not None and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        parser.error(f"--checkpoint-dir already exists and is non-empty, refusing to mix with a new run: {checkpoint_dir}")

    import numpy as np
    import torch

    def _log(message: str) -> None:
        print(message, flush=True)

    dataset_path = Path(args.dataset)
    dataset_report, arrays = _load_dataset(dataset_path)
    partition_labels = dataset_report["run_id_by_partition_index"]
    if len(partition_labels) != 3:
        raise TrainError("D1 requires exactly the 3 registered hard_window0 runs")
    type_vocabulary = [int(value) for value in arrays["type_vocabulary"].tolist()]
    delta_a = torch.tensor(arrays["delta_A"], dtype=torch.float64)
    model_kwargs = {
        "hidden_dim": args.hidden_dim, "n_interaction_blocks": args.n_interaction_blocks,
        "n_radial_basis": args.n_radial_basis,
    }
    # A topology index's element never changes across frames/runs (same
    # shared topology, enforced by build_exp012_student_training_dataset.py),
    # so this lookup and the throwaway model used only for its
    # `atomic_numbers_to_type_index` helper are both built once, not per fold/seed.
    atomic_number_by_topology_index = {
        index: int(number) for index, number in enumerate(arrays["all_topology_atomic_numbers"].tolist())
    }
    lookup_model = build_local_residual_student(type_vocabulary, **model_kwargs)

    teacher_arrays = None
    teacher_partition_labels = None
    teacher_report = None
    if not args.skip_distilled:
        teacher_path = Path(args.teacher_joined)
        teacher_report_path = teacher_path.with_name(teacher_path.stem + "_report.json")
        teacher_report = json.loads(teacher_report_path.read_text(encoding="utf-8"))
        # Whole-document preregistration_sha256 is NOT the right check here:
        # the document legitimately keeps evolving (e.g. the arm-retirement
        # and reseal edits in DEC-039) after an artifact like the teacher
        # join was built, exactly the situation DEC-034 already ran into and
        # deliberately did not gate on. What actually matters is that this
        # dataset and this teacher join describe the *same target window* --
        # check those specific fields instead.
        for field in ("target_f_k_kj_mol", "target_lambda_vdw", "A_k_window", "delta_A"):
            dataset_values = dataset_report[field]
            teacher_values = teacher_report[field]
            if len(dataset_values) != len(teacher_values) or any(
                abs(a - b) > 1e-6 for a, b in zip(dataset_values, teacher_values)
            ):
                raise TrainError(
                    f"--teacher-joined and --dataset disagree on {field} -- they were built "
                    "against different target windows, not just different preregistration edits"
                )
        if dataset_report["global_state_ids"] != teacher_report["global_state_ids"]:
            raise TrainError("--teacher-joined and --dataset disagree on global_state_ids")
        if set(dataset_report["run_id_by_partition_index"]) != set(teacher_report["run_id_by_partition_index"]):
            raise TrainError("--teacher-joined and --dataset do not cover the same set of runs")
        teacher_partition_labels = teacher_report["run_id_by_partition_index"]
        with np.load(teacher_path) as data:
            teacher_arrays = {key: data[key] for key in data.files}

    variants = ["direct_gap"] if args.skip_distilled else ["direct_gap", "distilled"]
    fold_reports = []

    for held_out_run_id in partition_labels:
        train_run_ids = [run_id for run_id in partition_labels if run_id != held_out_run_id]
        _log(f"[fold held_out={held_out_run_id}] training on {train_run_ids}")

        row_indices_by_run = {
            run_id: _run_row_indices(arrays, partition_labels, run_id) for run_id in partition_labels
        }

        # -- split each training run into train/validation via the dataset's
        #    own trailing-time-block flag --
        train_rows, val_rows = [], []
        for run_id in train_run_ids:
            rows = row_indices_by_run[run_id]
            is_val = arrays["is_early_stop_validation"][rows]
            train_rows.append(rows[~is_val])
            val_rows.append(rows[is_val])

        train_gaps, train_log_weights = _slice_ledger(arrays, np.concatenate(train_rows))
        val_gaps, val_log_weights = _slice_ledger(arrays, np.concatenate(val_rows))
        test_gaps, test_log_weights = _slice_ledger(arrays, row_indices_by_run[held_out_run_id])

        train_partition = torch.tensor(
            np.concatenate(
                [np.full(len(rows), label, dtype=np.int64) for label, rows in enumerate(train_rows)]
            ),
            dtype=torch.int64,
        )
        val_partition = torch.tensor(
            np.concatenate(
                [np.full(len(rows), label, dtype=np.int64) for label, rows in enumerate(val_rows)]
            ),
            dtype=torch.int64,
        )

        fold_result = {"held_out_run_id": held_out_run_id, "training_run_ids": train_run_ids, "variants": {}}

        for variant in variants:
            _log(f"  [variant={variant}]")
            per_seed = []
            distillation_weight_for_final_check = None
            teacher_predict = None
            if variant == "distilled":
                weight, mean, std, best_ridge = _fit_teacher_distillation_target(
                    teacher_arrays=teacher_arrays, teacher_partition_labels=teacher_partition_labels,
                    train_run_ids=train_run_ids, ridge_grid=args.teacher_ridge_grid,
                )
                distillation_weight_for_final_check = best_ridge

                def teacher_predict_for_run(run_id, frame_positions_within_run):
                    # `frame_positions_within_run` selects exactly the same
                    # frames (by position 0..N-1 within this run) as the
                    # student-side row selection being aligned against --
                    # both sides' frame 0..N-1 order is fail-closed-verified
                    # against the same ledger, so position i means the same
                    # physical frame in both artifacts.
                    label = teacher_partition_labels.index(run_id)
                    mask = teacher_arrays["partition_index"] == label
                    run_features = teacher_arrays["pooled_latent"][mask].astype(np.float64)[frame_positions_within_run]
                    features = torch.tensor(run_features, dtype=torch.float64)
                    return ((features - mean) / std) @ weight

                teacher_predict = teacher_predict_for_run

            for seed in args.seeds:
                _log(f"    [seed={seed}] precomputing frame features...")
                # Precompute per-frame geometry features once per (fold, run set);
                # cheap enough to redo per seed for code simplicity, and this is
                # D1 (offline), not the performance-sensitive path.
                def build_frames_and_types(rows):
                    frames = _precompute_frames(arrays, rows, lookup_model)
                    environment_types = [
                        _resolve_environment_type_index(lookup_model, frame, atomic_number_by_topology_index)
                        for frame in frames
                    ]
                    return frames, environment_types

                train_frames, train_env_types = build_frames_and_types(np.concatenate(train_rows))
                val_frames, val_env_types = build_frames_and_types(np.concatenate(val_rows))
                test_frames, test_env_types = build_frames_and_types(row_indices_by_run[held_out_run_id])

                distillation_target_train = None
                if variant == "distilled":
                    # train_rows[i] holds this fold's TRAIN-split row indices (into
                    # `arrays`) for train_run_ids[i], in the same order
                    # np.concatenate(train_rows) was built (= train_frames' order).
                    # Convert each to "position within that run" so the teacher
                    # side (which has no notion of the student dataset's global
                    # row numbering) can select the identical physical frames.
                    distillation_target_train = torch.cat(
                        [
                            teacher_predict(run_id, arrays["frame_index_within_run"][rows])
                            for run_id, rows in zip(train_run_ids, train_rows)
                        ],
                        dim=0,
                    )
                    # No validation-time distillation target needed: early
                    # stopping monitors gap_variance_loss alone (see module docstring).

                model, fit_summary = _train_one_model(
                    variant=variant, seed=seed, type_vocabulary=type_vocabulary, model_kwargs=model_kwargs,
                    train_frames=train_frames, train_environment_types=train_env_types,
                    val_frames=val_frames, val_environment_types=val_env_types,
                    train_gaps=train_gaps, train_log_weights=train_log_weights, train_partition=train_partition,
                    val_gaps=val_gaps, val_log_weights=val_log_weights, val_partition=val_partition,
                    delta_a=delta_a, max_epoch=args.max_epoch, patience=args.early_stop_patience,
                    distillation_target_train=distillation_target_train,
                    distillation_weight=args.distillation_weight,
                    log=_log,
                )

                if checkpoint_dir is not None:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = checkpoint_dir / f"{held_out_run_id}__{variant}__seed{seed}.pt"
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "type_vocabulary": type_vocabulary,
                            "model_kwargs": model_kwargs,
                            "held_out_run_id": held_out_run_id,
                            "training_run_ids": train_run_ids,
                            "variant": variant,
                            "seed": seed,
                            "dataset_report_sha256": dataset_report.get("report_sha256"),
                        },
                        checkpoint_path,
                    )
                    _log(f"      saved checkpoint: {checkpoint_path}")

                with torch.no_grad():
                    model.eval()
                    test_baseline = bidirectional_gap_variance_loss(
                        test_gaps, torch.zeros(test_gaps.shape[0], dtype=torch.float64), delta_a, test_log_weights,
                        partition_index=None,
                        energy_regularization_coefficient=0.0, force_regularization_coefficient=0.0,
                    )["gap_variance_loss"].item()
                    basis_test = _forward_frames(model, test_frames, test_env_types)
                    test_fitted = bidirectional_gap_variance_loss(
                        test_gaps, basis_test, delta_a, test_log_weights,
                        partition_index=None,
                        energy_regularization_coefficient=0.0, force_regularization_coefficient=0.0,
                    )["gap_variance_loss"].item()

                relative_improvement = (
                    (test_baseline - test_fitted) / test_baseline if test_baseline > 0 else None
                )
                per_seed.append(
                    {
                        "seed": seed,
                        **fit_summary,
                        "held_out_baseline_gap_variance_loss": test_baseline,
                        "held_out_fitted_gap_variance_loss": test_fitted,
                        "held_out_relative_improvement": relative_improvement,
                        "held_out_improved": test_fitted < test_baseline,
                    }
                )

            improvements = [entry["held_out_relative_improvement"] for entry in per_seed if entry["held_out_relative_improvement"] is not None]
            fold_result["variants"][variant] = {
                "per_seed": per_seed,
                "mean_held_out_relative_improvement": sum(improvements) / len(improvements) if improvements else None,
                "all_seeds_improved": all(entry["held_out_improved"] for entry in per_seed),
                "teacher_distillation_ridge_coefficient": distillation_weight_for_final_check,
            }

        fold_reports.append(fold_result)

    all_folds_direct_gap_improved = all(fold["variants"]["direct_gap"]["all_seeds_improved"] for fold in fold_reports)
    n_folds_direct_gap_improved = sum(
        1 for fold in fold_reports if fold["variants"]["direct_gap"]["all_seeds_improved"]
    )
    mean_improvements = [
        fold["variants"]["direct_gap"]["mean_held_out_relative_improvement"] for fold in fold_reports
        if fold["variants"]["direct_gap"]["mean_held_out_relative_improvement"] is not None
    ]

    body = {
        "schema_version": "exp012-local-residual-student-training-v1",
        "status": "COMPLETED_D1_HELD_OUT_EVALUATION",
        "dataset_report_sha256": dataset_report.get("report_sha256"),
        "teacher_joined_report_sha256": teacher_report.get("report_sha256") if teacher_report else None,
        "variants_trained": variants,
        "seeds": args.seeds,
        "max_epoch": args.max_epoch,
        "early_stop_patience": args.early_stop_patience,
        "checkpoint_dir": str(checkpoint_dir.resolve()) if checkpoint_dir is not None else None,
        "model_kwargs": model_kwargs,
        "folds": fold_reports,
        "direct_gap_all_folds_improved": all_folds_direct_gap_improved,
        "direct_gap_n_folds_improved": n_folds_direct_gap_improved,
        "direct_gap_mean_relative_improvement": (
            sum(mean_improvements) / len(mean_improvements) if mean_improvements else None
        ),
        "policy": {
            "provisional_not_sealed": True,
            "scientific_qualification": False,
            "decision_reference": "DEC-039",
            "a_k_frozen": True,
            "a_k_learned": False,
            "mace_encoder_trained": False,
            "local_residual_student_trained": True,
            "torchforce_used": False,
            "nvt_executed": False,
            "note": "D1 offline fitting only; d2 (autograd/finite-difference force check) is a separate, "
                    "later stage gated on this result per DEC-037",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
