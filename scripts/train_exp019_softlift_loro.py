#!/usr/bin/env python
"""Train the EXP-020 R1/R2/R3 direct-gap LORO screen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.loss import bidirectional_gap_variance_loss  # noqa: E402
from local_residual.softlift import (  # noqa: E402
    PackedSoftLiftBatch,
    SoftLiftConfig,
    build_softlift_model,
    count_trainable_parameters,
    derived_r1_config,
    primary_r1_config,
)


class SoftLiftTrainError(RuntimeError):
    """A dataset, split, or checkpoint contract failed."""


EXPERIMENT_ID = "EXP-020"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_seeded_model(config: SoftLiftConfig, *, context_mode: str, seed: int):
    """Construct a model only after wiring the requested initialization seed."""

    import torch

    torch.manual_seed(int(seed))
    return build_softlift_model(config, context_mode=context_mode)


def _load_dataset(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    report_path = path.with_name(path.stem + "_report.json")
    if not path.is_file() or not report_path.is_file():
        raise SoftLiftTrainError("dataset and matching report are required")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "softlift_dataset_v1" or not report.get("protocol_sha256"):
        raise SoftLiftTrainError("dataset report is not an EXP-019 dataset-v1 artifact")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if arrays["partition_index"].shape[0] != arrays["adjacent_gap_reduced"].shape[0]:
        raise SoftLiftTrainError("dataset frame and ledger dimensions disagree")
    return report, arrays


def _type_index_map(type_vocabulary):
    return {int(value): index for index, value in enumerate(type_vocabulary.tolist())}


def _batch_for_rows(arrays, rows, rung: str) -> PackedSoftLiftBatch:
    import numpy as np
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or rows.size == 0:
        raise SoftLiftTrainError("a training/evaluation split must contain frames")
    type_map = _type_index_map(arrays["type_vocabulary"])
    ligand_topology = arrays["ligand_topology_indices"].astype(np.int64)
    ligand_numbers = arrays["ligand_atomic_numbers"].astype(np.int64)
    ligand_types = np.asarray([type_map[int(value)] for value in ligand_numbers], dtype=np.int64)
    all_numbers = arrays["all_topology_atomic_numbers"].astype(np.int64)
    offsets = arrays["edge_offsets"]
    edge_frame = []
    edge_ligand_local = []
    edge_ligand_type = []
    edge_environment_type = []
    edge_distance = []
    edge_displacement = []
    for local_frame, row in enumerate(rows.tolist()):
        start, end = int(offsets[row]), int(offsets[row + 1])
        ligand_local = arrays["edge_ligand_local"][start:end].astype(np.int64)
        env_topology = arrays["edge_environment_topology"][start:end].astype(np.int64)
        edge_frame.append(np.full(end - start, local_frame, dtype=np.int64))
        edge_ligand_local.append(ligand_local)
        edge_ligand_type.append(np.asarray([type_map[int(ligand_numbers[index])] for index in ligand_local], dtype=np.int64))
        edge_environment_type.append(np.asarray([type_map[int(all_numbers[index])] for index in env_topology], dtype=np.int64))
        edge_distance.append(arrays["edge_distance_angstrom"][start:end].astype(np.float64))
        if rung == "R3":
            edge_displacement.append(arrays["edge_displacement_angstrom"][start:end].astype(np.float64))
    empty_i = np.empty((0,), dtype=np.int64)
    empty_f = np.empty((0,), dtype=np.float64)
    return PackedSoftLiftBatch(
        ligand_type_index=torch.tensor(np.repeat(ligand_types[None, :], rows.size, axis=0), dtype=torch.int64),
        edge_frame=torch.tensor(np.concatenate(edge_frame) if edge_frame else empty_i, dtype=torch.int64),
        edge_ligand_local=torch.tensor(np.concatenate(edge_ligand_local) if edge_ligand_local else empty_i, dtype=torch.int64),
        edge_ligand_type=torch.tensor(np.concatenate(edge_ligand_type) if edge_ligand_type else empty_i, dtype=torch.int64),
        edge_environment_type=torch.tensor(np.concatenate(edge_environment_type) if edge_environment_type else empty_i, dtype=torch.int64),
        edge_distance_angstrom=torch.tensor(np.concatenate(edge_distance) if edge_distance else empty_f, dtype=torch.float64),
        edge_displacement_angstrom=(torch.tensor(np.concatenate(edge_displacement), dtype=torch.float64) if rung == "R3" else None),
    )


def _rows(arrays, labels, *, trailing_validation: bool | None = None):
    import numpy as np

    result = []
    for label in labels:
        candidates = np.flatnonzero(arrays["partition_index"] == label)
        validation_mask = np.zeros(candidates.size, dtype=bool)
        count = max(1, int(round(candidates.size * 0.2))) if candidates.size > 1 else 0
        if count:
            validation_mask[-count:] = True
        if trailing_validation is True:
            candidates = candidates[validation_mask]
        elif trailing_validation is False:
            candidates = candidates[~validation_mask]
        result.append(candidates)
    return np.concatenate(result) if result else np.empty((0,), dtype=np.int64)


def _loss_for_rows(arrays, rows, basis, delta, *, partition_override=None, energy_regularization=0.0):
    import torch

    row_tensor = torch.tensor(rows, dtype=torch.int64)
    gaps = torch.tensor(arrays["adjacent_gap_reduced"][rows], dtype=torch.float64)
    log_weights = torch.tensor(arrays["log_importance_unnormalized"][rows], dtype=torch.float64)
    partitions = torch.tensor(
        arrays["partition_index"][rows] if partition_override is None else partition_override,
        dtype=torch.int64,
    )
    delta_tensor = torch.tensor(delta, dtype=torch.float64)
    return bidirectional_gap_variance_loss(
        gaps, basis, delta_tensor, log_weights, partition_index=partitions,
        energy_regularization_coefficient=energy_regularization,
        force_regularization_coefficient=0.0,
    )


def _config(rung: str, protocol_sha256: str, type_vocabulary, n_ligand: int, budgets, allow_derived: bool = False) -> SoftLiftConfig:
    if rung == "R1":
        config = primary_r1_config(protocol_sha256)
        if tuple(type_vocabulary.tolist()) != config.type_vocabulary or n_ligand != config.n_ligand_atoms:
            # 出厂 R1 config 是 41 原子 Atenolol 的常量，换配体必然走到这里。默认仍然
            # 拒绝（EXP-020 的复现路径不能被悄悄改架构），只有显式要求时才按体系推
            # 出同架构的 config —— local_residual.autofit 走的就是这条。
            if not allow_derived:
                raise SoftLiftTrainError(
                    "dataset type vocabulary/ligand count differs from frozen R1 primary config"
                    "（换配体重训请加 --allow-derived-r1-config）"
                )
            return derived_r1_config(
                protocol_sha256,
                type_vocabulary=type_vocabulary.tolist(),
                n_ligand_atoms=n_ligand,
                max_environment_atoms=int(budgets["unique_environment_atoms_hard"]),
                max_edges=int(budgets["directed_edges_hard"]),
                max_neighbors_per_ligand=int(budgets["neighbors_per_ligand_hard"]),
            )
        return config
    return SoftLiftConfig(
        schema_version="exp019-softlift-v1", rung=rung,
        type_vocabulary=tuple(int(value) for value in type_vocabulary.tolist()),
        n_ligand_atoms=n_ligand, n_radial_basis=16, n_channels=4,
        pair_dim=8, context_dim=8, inner_cutoff_angstrom=4.0,
        outer_cutoff_angstrom=5.0, b_max_reduced=10.0,
        max_environment_atoms=int(budgets["unique_environment_atoms_hard"]),
        max_edges=int(budgets["directed_edges_hard"]),
        max_neighbors_per_ligand=int(budgets["neighbors_per_ligand_hard"]),
        no_contact_output="exact_zero", protocol_sha256=protocol_sha256,
    )


def _train_one(
    *, model, train_batch, val_batch, train_arrays, train_rows, val_arrays, val_rows,
    delta, train_partitions, val_partitions, max_epochs: int, patience: int,
):
    import torch

    model = model.to(torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_basis = model(train_batch)
        train_loss = _loss_for_rows(train_arrays, train_rows, train_basis, delta, partition_override=train_partitions, energy_regularization=1e-4)
        train_loss["loss"].backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_basis = model(val_batch)
            val_result = _loss_for_rows(val_arrays, val_rows, val_basis, delta, partition_override=val_partitions)
            val_loss = float(val_result["gap_variance_loss"].item())
        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "best_validation_gap_variance_loss": best_val}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="output/outer_lambda_exp020_softlift/dataset/softlift_dataset_v1.npz")
    parser.add_argument("--rung", choices=("R1", "R2", "R3"), default="R1")
    parser.add_argument("--context-mode", choices=("normal", "context_zero", "anchor_shuffle"), default="normal")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-root", default="output/outer_lambda_exp020_softlift")
    parser.add_argument("--allow-derived-r1-config", action="store_true",
                        help="换配体重训：R1 的词表/配体原子数/容量按数据集推，架构常量不变")
    args = parser.parse_args(argv)
    if args.max_epochs <= 0 or args.patience <= 0:
        parser.error("--max-epochs and --patience must be positive")

    import numpy as np
    import torch

    dataset_path = Path(args.dataset)
    report, arrays = _load_dataset(dataset_path)
    config = _config(args.rung, report["protocol_sha256"], arrays["type_vocabulary"], int(arrays["ligand_topology_indices"].size), report["budgets"], allow_derived=args.allow_derived_r1_config)
    delta = arrays["delta_A"].astype(np.float64)
    if delta.size != arrays["adjacent_gap_reduced"].shape[1]:
        raise SoftLiftTrainError("delta_A length does not match ledger gap edges")
    run_labels = sorted(int(value) for value in np.unique(arrays["partition_index"]).tolist())
    if len(run_labels) != 3:
        raise SoftLiftTrainError("EXP-019 LORO requires exactly three run partitions")
    output_root = Path(args.output_root) / {"R1": "r1_density", "R2": "r2_context_gate", "R3": "r3_moments"}[args.rung]
    output_root.mkdir(parents=True, exist_ok=True)
    fold_reports = []
    for test_label in run_labels:
        train_labels = [label for label in run_labels if label != test_label]
        train_rows = _rows(arrays, train_labels, trailing_validation=False)
        val_rows = _rows(arrays, train_labels, trailing_validation=True)
        test_rows = _rows(arrays, [test_label], trailing_validation=None)
        train_partitions = arrays["partition_index"][train_rows]
        val_partitions = arrays["partition_index"][val_rows]
        test_partitions = np.zeros(test_rows.size, dtype=np.int64)
        train_batch = _batch_for_rows(arrays, train_rows, args.rung)
        val_batch = _batch_for_rows(arrays, val_rows, args.rung)
        test_batch = _batch_for_rows(arrays, test_rows, args.rung)
        baseline = _loss_for_rows(arrays, test_rows, torch.zeros(test_rows.size, dtype=torch.float64), delta, partition_override=test_partitions)["gap_variance_loss"].item()
        seed_reports = []
        for seed in args.seeds:
            model = _build_seeded_model(config, context_mode=args.context_mode, seed=seed)
            model, fit_report = _train_one(
                model=model, train_batch=train_batch, val_batch=val_batch,
                train_arrays=arrays, train_rows=train_rows, val_arrays=arrays, val_rows=val_rows,
                delta=delta, train_partitions=train_partitions, val_partitions=val_partitions,
                max_epochs=args.max_epochs, patience=args.patience,
            )
            model.eval()
            with torch.no_grad():
                basis_test = model(test_batch)
                candidate = _loss_for_rows(arrays, test_rows, basis_test, delta, partition_override=test_partitions)["gap_variance_loss"].item()
            improvement = (baseline - candidate) / baseline if baseline > 0.0 else 0.0
            variant_name = "direct_gap" if args.context_mode == "normal" else args.context_mode
            checkpoint = output_root / f"{args.rung.lower()}__{variant_name}__fold_test_run{test_label + 1}__seed{seed}.pt"
            if checkpoint.exists():
                raise SoftLiftTrainError(f"refusing to overwrite checkpoint: {checkpoint}")
            torch.save({
                "experiment_id": EXPERIMENT_ID,
                "held_out_run": int(test_label + 1),
                "seed": int(seed),
                "state_dict": model.state_dict(),
                "config": config.__dict__,
                "fit_report": fit_report,
            }, checkpoint)
            checkpoint_sha256 = _sha256_file(checkpoint)
            seed_reports.append({
                "seed": int(seed), "checkpoint": str(checkpoint), "baseline_gap_variance_loss": float(baseline),
                "held_out_gap_variance_loss": float(candidate), "relative_improvement": float(improvement),
                "checkpoint_sha256": checkpoint_sha256,
                "n_trainable_parameters": count_trainable_parameters(model), **fit_report,
            })
        median_improvement = float(np.median([item["relative_improvement"] for item in seed_reports]))
        fold_reports.append({
            "test_partition": int(test_label), "median_relative_improvement": median_improvement,
            "improving_seed_count": int(sum(item["relative_improvement"] > 0.0 for item in seed_reports)),
            "seeds": seed_reports,
        })
    median_values = [item["median_relative_improvement"] for item in fold_reports]
    verdict = bool(
        sum(value > 0.0 for value in median_values) >= 2
        and float(np.mean(median_values)) > 0.0
        and min(median_values) >= -0.10
        and all(item["improving_seed_count"] >= 2 for item in fold_reports)
    )
    body = {
        "schema_version": "exp019-softlift-d1-v1", "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETED_HELD_OUT_EVALUATION",
        "rung": args.rung, "variant": "direct_gap" if args.context_mode == "normal" else args.context_mode,
        "context_mode": args.context_mode, "protocol_sha256": report["protocol_sha256"],
        "dataset_sha256": report["dataset_sha256"], "config": config.__dict__,
        "n_trainable_parameters": count_trainable_parameters(
            _build_seeded_model(config, context_mode=args.context_mode, seed=args.seeds[0])
        ),
        "folds": fold_reports, "mean_relative_improvement": float(np.mean(median_values)),
        "qualification": verdict,
        "training_script": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "seed_wiring": {
            "status": "FIXED_BEFORE_MODEL_CONSTRUCTION",
            "method": "torch.manual_seed(seed)",
            "seeds": [int(seed) for seed in args.seeds],
        },
        "policy": {"test_run_used_for_model_choice": False, "beta_reapplied": False, "A_k_learned": False, "native_started": False},
    }
    body["report_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    variant_name = "direct_gap" if args.context_mode == "normal" else args.context_mode
    report_path = output_root / f"{args.rung.lower()}__{variant_name}__d1_report.json"
    if report_path.exists():
        raise SoftLiftTrainError(f"refusing to overwrite report: {report_path}")
    report_path.write_text(json.dumps(body, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"rung": args.rung, "qualification": verdict, "mean_relative_improvement": body["mean_relative_improvement"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
