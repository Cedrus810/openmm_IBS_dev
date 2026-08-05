#!/usr/bin/env python
"""DEC-039 / D1 step 1: build the `LocalResidualStudent` offline training dataset.

Unlike `scripts/join_exp012_teacher_latent_cache_with_ledger.py` (which joins
the teacher's already-computed `pooled_latent`), this script computes the
STUDENT's own features directly from real per-frame trajectory coordinates --
per DEC-037 (d0) item "(d1) 离线 student 拟合...用真实逐帧坐标计算 student 自己的
特征，不是只读 teacher 的 cached pooled_latent". What is cached here is raw,
parameter-independent geometry (which environment atoms are in range this
frame, and at what distance) via
`local_residual.geometry.ligand_environment_cross_edges` -- the same
DEC-038/039-validated online funnel primitive the student model itself uses at
every step. No embedding, interaction block, or readout runs here; those are
applied fresh during training in `scripts/train_exp012_local_residual_student.py`,
so caching this geometry only saves repeating an expensive-but-parameter-free
O(n_ligand x n_environment) distance computation every training epoch.

Box vectors are read fresh from each frame's own trajectory (`unitcell_vectors`),
never from a separately-cached `box_vectors.npy` -- this sidesteps the
DEC-039-diagnosed staleness bug in the ms/step benchmark entirely, since the
per-frame box actually simulated is exactly what the trajectory file records.

Ledger alignment (`adjacent_gap_reduced`, `log_importance_unnormalized`,
`delta_A`, `A_k`) reuses the exact same fail-closed checks
`join_exp012_teacher_latent_cache_with_ledger.py` established (run_id match,
frame_count match, strict frame_index==0..N-1, target f_k/lambda_vdw
cross-check against the current preregistration, A_k independently
recomputed from `sin^2(pi*lambda_vdw)` rather than trusted) -- duplicated here
as one small, easily-diffed pure function rather than importing across two
independent CLI scripts.

Early-stopping validation split (DEC-039): within each run, the *trailing*
`--early-stop-val-fraction` contiguous time block (default 20%, not a random
sample -- DEC-011's autocorrelation concern applies here exactly as it did to
whole-run splitting) is flagged `is_early_stop_validation=True`. A fold's
held-out third run ignores this flag entirely (it is never used for early
stopping or model selection, only for the final reported evaluation); the
flag only matters for the two TRAINING runs within a given outer fold.
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
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp012_xed.schema import load_preregistration  # noqa: E402
from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.geometry import ligand_environment_cross_edges  # noqa: E402


class DatasetBuildError(RuntimeError):
    """A run's inputs failed a fail-closed identity, alignment, or geometry check."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_npz(path: Path, arrays: dict) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".npz", delete=False
        ) as handle:
            temporary = handle.name
        np.savez(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


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


def _resolve_delta_a(registration: Any) -> tuple[list[float], list[float], list[int]]:
    """Mirrors join_exp012_teacher_latent_cache_with_ledger.py::_resolve_delta_a exactly.

    Small and pure enough to duplicate rather than import across two
    independent standalone CLI scripts; any future edit to this formula must
    be applied to both copies (both are directly reading
    ``target.global_schedule`` and independently re-deriving A_k from
    ``sin^2(pi*lambda_vdw)``, so a divergence between the two would be
    immediately visible as a systematic offset between teacher and student
    targets).
    """

    target = registration.payload["target"]
    global_schedule = target["global_schedule"]
    ledger_slice = target["ledger_slice"]
    if global_schedule.get("A_definition") != "sin_squared_pi_lambda_vdw":
        raise DatasetBuildError("global_schedule.A_definition is not the expected sin^2(pi*lambda_vdw)")
    global_state_ids = [int(value) for value in ledger_slice["global_state_ids"]]
    a_k_full = [float(value) for value in global_schedule["A_k"]]
    lambda_vdw_full = [float(value) for value in global_schedule["lambda_vdw"]]
    a_k_window = [a_k_full[index] for index in global_state_ids]
    lambda_vdw_window = [lambda_vdw_full[index] for index in global_state_ids]

    for index, (declared, lam) in zip(global_state_ids, zip(a_k_window, lambda_vdw_window)):
        recomputed = 0.0 if lam <= 0.0 or lam >= 1.0 else math.sin(math.pi * lam) ** 2
        if abs(recomputed - declared) > 1e-9:
            raise DatasetBuildError(
                f"declared A_k[{index}]={declared} does not match sin^2(pi*lambda_vdw) "
                f"recomputation {recomputed}"
            )

    delta_a = [a_k_window[i + 1] - a_k_window[i] for i in range(len(a_k_window) - 1)]
    return delta_a, a_k_window, global_state_ids


def _load_ledger(
    run_id: str, *, ledger_dir: Path, target_f_k: list[float], target_lambda_vdw: list[float],
):
    import numpy as np

    ledger_npz_path = ledger_dir / run_id / "ledger_arrays.npz"
    ledger_report_path = ledger_dir / run_id / "ledger_report.json"
    for path in (ledger_npz_path, ledger_report_path):
        if not path.is_file():
            raise DatasetBuildError(f"missing required ledger input for {run_id}: {path}")

    ledger_report = json.loads(ledger_report_path.read_text(encoding="utf-8"))
    if ledger_report.get("run_id") != run_id:
        raise DatasetBuildError(f"{run_id}: ledger report run_id mismatch")

    ledger_f_k = [float(value) for value in ledger_report["f_k_kj_mol"]]
    ledger_lambda_vdw = [float(value) for value in ledger_report["lambdas_vdw"]]
    if len(ledger_f_k) != len(target_f_k) or any(abs(a - b) > 1e-6 for a, b in zip(ledger_f_k, target_f_k)):
        raise DatasetBuildError(f"{run_id}: ledger f_k_kj_mol differs from the current preregistration target")
    if len(ledger_lambda_vdw) != len(target_lambda_vdw) or any(
        abs(a - b) > 1e-6 for a, b in zip(ledger_lambda_vdw, target_lambda_vdw)
    ):
        raise DatasetBuildError(f"{run_id}: ledger lambdas_vdw differs from the current preregistration target")

    with np.load(ledger_npz_path) as ledger_data:
        ledger_frame_index = ledger_data["frame_index"]
        adjacent_gap_reduced = ledger_data["adjacent_gap_reduced"]
        log_importance_unnormalized = ledger_data["log_importance_unnormalized"]

    frame_count = int(ledger_report["frame_count"])
    expected_frame_index = np.arange(frame_count, dtype=ledger_frame_index.dtype)
    if not np.array_equal(ledger_frame_index, expected_frame_index):
        raise DatasetBuildError(f"{run_id}: ledger frame_index is not exactly 0..{frame_count - 1}")
    if adjacent_gap_reduced.shape[0] != frame_count or log_importance_unnormalized.shape[0] != frame_count:
        raise DatasetBuildError(f"{run_id}: ledger arrays' frame dimension differs from frame_count")
    if adjacent_gap_reduced.shape[1] + 1 != log_importance_unnormalized.shape[1]:
        raise DatasetBuildError(f"{run_id}: edge count and target-state count are inconsistent")

    return {
        "adjacent_gap_reduced": adjacent_gap_reduced.astype(np.float64),
        "log_importance_unnormalized": log_importance_unnormalized.astype(np.float64),
        "frame_count": frame_count,
        "ledger_npz_sha256": _sha256_file(ledger_npz_path),
        "ledger_report_sha256": _sha256_file(ledger_report_path),
    }


def _build_run_geometry(
    *, run_id: str, topology_path: Path, trajectory_path: Path, ligand_topology_indices: list[int],
    frame_count: int, edge_cutoff_angstrom: float, device: str, log,
):
    """Compute per-frame ligand-environment funnel geometry for one run.

    Returns per-frame edge arrays (as a CSR-style flat concatenation with
    offsets) plus this run's discovered atomic numbers (for the global
    `type_vocabulary`).
    """

    import mdtraj
    import numpy as np
    import torch

    trajectory = mdtraj.load(str(trajectory_path), top=str(topology_path))
    if trajectory.n_frames != frame_count:
        raise DatasetBuildError(
            f"{run_id}: trajectory has {trajectory.n_frames} frames, ledger declares {frame_count}"
        )
    if trajectory.unitcell_vectors is None:
        raise DatasetBuildError(f"{run_id}: trajectory has no periodic box vectors")

    n_atoms = trajectory.topology.n_atoms
    ligand_set = set(ligand_topology_indices)
    if any(index < 0 or index >= n_atoms for index in ligand_topology_indices):
        raise DatasetBuildError(f"{run_id}: a ligand_topology_index is out of range for this topology")
    atomic_numbers_all = [int(atom.element.atomic_number) for atom in trajectory.topology.atoms]
    environment_topology_indices = sorted(set(range(n_atoms)) - ligand_set)
    environment_tensor = torch.tensor(environment_topology_indices, dtype=torch.int64, device=device)
    ligand_tensor = torch.tensor(sorted(ligand_set), dtype=torch.int64, device=device)

    edge_ligand_parts = []
    edge_environment_parts = []
    edge_distance_parts = []
    edge_counts = []
    started = time.perf_counter()
    for frame_index in range(frame_count):
        positions = torch.tensor(
            trajectory.xyz[frame_index] * 10.0, dtype=torch.float64, device=device
        )
        box = torch.tensor(
            trajectory.unitcell_vectors[frame_index] * 10.0, dtype=torch.float64, device=device
        )
        edges = ligand_environment_cross_edges(
            positions, box, ligand_tensor, environment_tensor, outer_cutoff=edge_cutoff_angstrom,
        )
        edge_ligand_parts.append(edges["edge_index"][0].to("cpu").numpy().astype(np.int64))
        edge_environment_parts.append(edges["edge_index"][1].to("cpu").numpy().astype(np.int64))
        edge_distance_parts.append(edges["distance"].to("cpu").numpy().astype(np.float64))
        edge_counts.append(int(edges["distance"].shape[0]))
        if (frame_index + 1) % 100 == 0 or frame_index + 1 == frame_count:
            log(
                f"    {run_id}: {frame_index + 1}/{frame_count} frames "
                f"({time.perf_counter() - started:.1f}s elapsed)"
            )

    edge_ligand_flat = np.concatenate(edge_ligand_parts) if edge_ligand_parts else np.empty((0,), dtype=np.int64)
    edge_environment_flat = (
        np.concatenate(edge_environment_parts) if edge_environment_parts else np.empty((0,), dtype=np.int64)
    )
    edge_distance_flat = (
        np.concatenate(edge_distance_parts) if edge_distance_parts else np.empty((0,), dtype=np.float64)
    )
    edge_offsets = np.concatenate([[0], np.cumsum(edge_counts)]).astype(np.int64)

    return {
        "edge_ligand_flat": edge_ligand_flat,
        "edge_environment_flat": edge_environment_flat,
        "edge_distance_flat": edge_distance_flat,
        "edge_offsets": edge_offsets,
        "atomic_numbers_all": atomic_numbers_all,
        "trajectory_sha256": _sha256_file(trajectory_path),
        "topology_sha256": _sha256_file(topology_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default="protocols/EXP-012_preregistration.json")
    parser.add_argument(
        "--topology", required=True, help="mdtraj-compatible topology shared by every --run-id"
    )
    parser.add_argument(
        "--trajectory", action="append", required=True, metavar="RUN_ID=PATH",
        help="repeatable, e.g. --trajectory hard_window0_run1=output/.../hard_window_screening.dcd",
    )
    parser.add_argument(
        "--ligand-indices", required=True, help="JSON file with a ligand_indices array (shared by every run)"
    )
    parser.add_argument(
        "--ledger-dir", default="output/outer_lambda_exp012/mm_ledger_cuda",
        help="directory with <run_id>/ledger_arrays.npz and ledger_report.json",
    )
    parser.add_argument("--edge-cutoff-angstrom", type=float, default=5.0)
    parser.add_argument(
        "--early-stop-val-fraction", type=float, default=0.2,
        help="trailing contiguous fraction of each run flagged is_early_stop_validation (DEC-039)",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if not (0.0 < args.early_stop_val_fraction < 1.0):
        parser.error("--early-stop-val-fraction must be strictly between 0 and 1")
    if args.edge_cutoff_angstrom <= 0.0:
        parser.error("--edge-cutoff-angstrom must be positive")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen dataset: {args.output}")

    trajectory_by_run: dict[str, Path] = {}
    for item in args.trajectory:
        if "=" not in item:
            parser.error(f"--trajectory must be RUN_ID=PATH, got: {item}")
        run_id, path = item.split("=", 1)
        trajectory_by_run[run_id] = Path(path)

    import numpy as np

    def _log(message: str) -> None:
        print(message, flush=True)

    registration = load_preregistration(
        Path(args.preregistration) if Path(args.preregistration).is_absolute()
        else ROOT / args.preregistration,
        workspace_root=ROOT, verify_files=True,
    )
    delta_a, a_k_window, global_state_ids = _resolve_delta_a(registration)
    target_f_k = [float(value) for value in registration.payload["target"]["ledger_slice"]["f_k_kj_mol"]]
    lambda_vdw_full = [float(value) for value in registration.payload["target"]["global_schedule"]["lambda_vdw"]]
    target_lambda_vdw = [lambda_vdw_full[index] for index in global_state_ids]

    registered_run_ids = [run["run_id"] for run in registration.payload["inputs"]["runs"]]
    missing = set(registered_run_ids) - set(trajectory_by_run)
    if missing:
        parser.error(f"--trajectory missing for run(s) registered in the preregistration: {sorted(missing)}")

    ligand_payload = json.loads(Path(args.ligand_indices).expanduser().read_text(encoding="utf-8"))
    ligand_topology_indices = sorted(int(index) for index in ligand_payload["ligand_indices"])
    topology_path = Path(args.topology)

    per_run: dict[str, dict] = {}
    discovered_atomic_numbers: set[int] = set()
    for run_id in registered_run_ids:
        _log(f"[{run_id}] loading ledger...")
        ledger = _load_ledger(
            run_id, ledger_dir=Path(args.ledger_dir),
            target_f_k=target_f_k, target_lambda_vdw=target_lambda_vdw,
        )
        _log(f"[{run_id}] computing student funnel geometry for {ledger['frame_count']} frames...")
        geometry = _build_run_geometry(
            run_id=run_id, topology_path=topology_path, trajectory_path=trajectory_by_run[run_id],
            ligand_topology_indices=ligand_topology_indices, frame_count=ledger["frame_count"],
            edge_cutoff_angstrom=args.edge_cutoff_angstrom, device=args.device, log=_log,
        )
        discovered_atomic_numbers.update(geometry["atomic_numbers_all"])
        per_run[run_id] = {**ledger, **geometry}

    type_vocabulary = sorted(discovered_atomic_numbers)
    # Sanity: every run must describe the same topology (same atomic numbers
    # for every atom index), otherwise "shared ligand ordering" is meaningless.
    reference_atomic_numbers = per_run[registered_run_ids[0]]["atomic_numbers_all"]
    for run_id in registered_run_ids[1:]:
        if per_run[run_id]["atomic_numbers_all"] != reference_atomic_numbers:
            raise DatasetBuildError(
                f"{run_id}: topology atomic numbers differ from {registered_run_ids[0]}'s -- "
                "runs must share one topology"
            )
    ligand_atomic_numbers = [reference_atomic_numbers[index] for index in ligand_topology_indices]
    # Persist the FULL per-topology-index atomic number array (not just the
    # ligand's small fixed list): the training script must be able to look up
    # any environment atom's element by topology index, and the environment
    # candidate pool spans nearly the whole system (DEC-038/039), not a small
    # fixed manifest -- there is no smaller correct set to store instead.
    all_topology_atomic_numbers = np.asarray(reference_atomic_numbers, dtype=np.int64)

    partition_labels = registered_run_ids
    partition_index_parts = []
    frame_index_within_run_parts = []
    is_early_stop_validation_parts = []
    edge_offsets_global = [0]
    edge_ligand_flat_parts = []
    edge_environment_flat_parts = []
    edge_distance_flat_parts = []
    adjacent_gap_reduced_parts = []
    log_importance_unnormalized_parts = []

    running_edge_total = 0
    for label_index, run_id in enumerate(partition_labels):
        run = per_run[run_id]
        frame_count = run["frame_count"]
        n_validation = int(round(frame_count * args.early_stop_val_fraction))
        n_validation = max(1, min(frame_count - 1, n_validation)) if frame_count > 1 else 0
        is_validation = np.zeros(frame_count, dtype=bool)
        if n_validation > 0:
            is_validation[frame_count - n_validation :] = True

        partition_index_parts.append(np.full(frame_count, label_index, dtype=np.int64))
        frame_index_within_run_parts.append(np.arange(frame_count, dtype=np.int64))
        is_early_stop_validation_parts.append(is_validation)
        adjacent_gap_reduced_parts.append(run["adjacent_gap_reduced"])
        log_importance_unnormalized_parts.append(run["log_importance_unnormalized"])
        edge_ligand_flat_parts.append(run["edge_ligand_flat"])
        edge_environment_flat_parts.append(run["edge_environment_flat"])
        edge_distance_flat_parts.append(run["edge_distance_flat"])
        # This run's local per-frame offsets, shifted into the global flat arrays.
        local_offsets = run["edge_offsets"]
        shifted = local_offsets[1:] + running_edge_total
        edge_offsets_global.extend(shifted.tolist())
        running_edge_total += int(local_offsets[-1])

    output_path = Path(args.output)
    arrays = {
        "partition_index": np.concatenate(partition_index_parts),
        "frame_index_within_run": np.concatenate(frame_index_within_run_parts),
        "is_early_stop_validation": np.concatenate(is_early_stop_validation_parts),
        "adjacent_gap_reduced": np.concatenate(adjacent_gap_reduced_parts, axis=0),
        "log_importance_unnormalized": np.concatenate(log_importance_unnormalized_parts, axis=0),
        "edge_offsets": np.asarray(edge_offsets_global, dtype=np.int64),
        "edge_ligand_topology": np.concatenate(edge_ligand_flat_parts) if edge_ligand_flat_parts else np.empty((0,), dtype=np.int64),
        "edge_environment_topology": np.concatenate(edge_environment_flat_parts) if edge_environment_flat_parts else np.empty((0,), dtype=np.int64),
        "edge_distance_angstrom": np.concatenate(edge_distance_flat_parts) if edge_distance_flat_parts else np.empty((0,), dtype=np.float64),
        "delta_A": np.asarray(delta_a, dtype=np.float64),
        "A_k_window": np.asarray(a_k_window, dtype=np.float64),
        "ligand_topology_indices": np.asarray(ligand_topology_indices, dtype=np.int64),
        "ligand_atomic_numbers": np.asarray(ligand_atomic_numbers, dtype=np.int64),
        "type_vocabulary": np.asarray(type_vocabulary, dtype=np.int64),
        "all_topology_atomic_numbers": all_topology_atomic_numbers,
    }
    _atomic_write_npz(output_path, arrays)
    npz_sha = _sha256_file(output_path)

    body = {
        "schema_version": "exp012-student-training-dataset-v1",
        "status": "COMPLETED_GEOMETRY_AND_LEDGER_JOIN_NOT_TRAINED",
        "preregistration_sha256": registration.payload_sha256,
        "edge_cutoff_angstrom": args.edge_cutoff_angstrom,
        "early_stop_val_fraction": args.early_stop_val_fraction,
        "run_id_by_partition_index": partition_labels,
        "frame_count_by_run": {run_id: per_run[run_id]["frame_count"] for run_id in partition_labels},
        "total_frame_count": int(sum(per_run[run_id]["frame_count"] for run_id in partition_labels)),
        "total_edge_count": int(running_edge_total),
        "global_state_ids": global_state_ids,
        "target_f_k_kj_mol": target_f_k,
        "target_lambda_vdw": target_lambda_vdw,
        "A_k_window": a_k_window,
        "delta_A": delta_a,
        "A_definition": "sin_squared_pi_lambda_vdw",
        "type_vocabulary": type_vocabulary,
        "n_ligand_atoms": len(ligand_topology_indices),
        "npz_path": str(output_path.resolve()),
        "npz_sha256": npz_sha,
        "inputs": {
            run_id: {
                "trajectory_sha256": per_run[run_id]["trajectory_sha256"],
                "topology_sha256": per_run[run_id]["topology_sha256"],
                "ledger_npz_sha256": per_run[run_id]["ledger_npz_sha256"],
                "ledger_report_sha256": per_run[run_id]["ledger_report_sha256"],
            }
            for run_id in partition_labels
        },
        "policy": {
            "feature_source": "real_per_frame_coordinates_not_teacher_pooled_latent",
            "box_vectors_source": "trajectory_unitcell_vectors_per_frame_not_cached_npy",
            "a_k_frozen": True,
            "a_k_recomputed_and_checked": True,
            "student_model_executed": False,
            "training_executed": False,
            "decision_reference": "DEC-039",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(output_path.with_name(output_path.stem + "_report.json"), report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
