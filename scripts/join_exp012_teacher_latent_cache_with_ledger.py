#!/usr/bin/env python
"""DEC-030(c) step 1: join the per-frame teacher latent cache with the MM ledger.

The latent cache (DEC-032/033) is representation-only by design -- it carries
no ledger fields at all, so the thermodynamic target (``adjacent_gap_reduced``,
``log_importance_unnormalized``) and the frozen envelope coefficients
(``A_k``, hence ``delta_A`` per adjacent edge) are joined in here, once,
fail-closed. This script never recomputes or guesses ``delta_A``: it reads
the already-sealed ``A_k`` schedule straight out of
``protocols/EXP-012_preregistration.json``'s ``target.global_schedule`` (a
``sin^2(pi * lambda_vdw)`` envelope, ``A_0 = A_22 = 0`` at the true physical
alchemical endpoints), sliced to this window's five states via
``target.ledger_slice.global_state_ids``, and cross-checks the declared
``A_k`` against an independent recomputation from ``lambda_vdw`` as a
sanity gate. Per the outer-lambda-neural-basis plan (PLAN doc, "第一轮冻结全局
A_k"): A_k must stay frozen while only the readout is fit, to avoid the scale
degeneracy between the envelope and the basis weights.

Identity note: the ledger's own ``preregistration_payload_sha256`` will not
generally equal the current preregistration's payload hash -- the document
has legitimately evolved since the ledger was built (new decision entries,
resolved fields, etc.), and that alone does not mean the target window
changed. Rather than gate on that whole-document hash, this script verifies
the specific fields that matter directly: run_id, frame_count, ``f_k_kj_mol``,
and ``lambda_vdw`` against the *current* preregistration's target section
(with the ledger report's own copies checked too, for redundant confirmation),
plus a strict frame_index alignment check between the cache and the ledger
arrays before concatenating them positionally.

Output is one joined ``.npz`` (features + targets + a per-frame partition
label identifying which run each frame came from) and a report recording
every input's path/SHA-256 and the frozen ``delta_A``/``A_k`` used.
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

from exp012_xed.schema import load_preregistration  # noqa: E402
from local_residual.environment import canonical_json_bytes  # noqa: E402


class JoinError(ValueError):
    """A cache/ledger input failed a fail-closed identity or alignment check."""


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
    target = registration.payload["target"]
    global_schedule = target["global_schedule"]
    ledger_slice = target["ledger_slice"]
    if global_schedule.get("A_definition") != "sin_squared_pi_lambda_vdw":
        raise JoinError("global_schedule.A_definition is not the expected sin^2(pi*lambda_vdw)")
    global_state_ids = [int(value) for value in ledger_slice["global_state_ids"]]
    a_k_full = [float(value) for value in global_schedule["A_k"]]
    lambda_vdw_full = [float(value) for value in global_schedule["lambda_vdw"]]
    a_k_window = [a_k_full[index] for index in global_state_ids]
    lambda_vdw_window = [lambda_vdw_full[index] for index in global_state_ids]

    # Recompute independently from lambda_vdw as a sanity check -- never trust
    # the declared A_k without re-deriving it from the same formula.
    for index, (declared, lam) in zip(global_state_ids, zip(a_k_window, lambda_vdw_window)):
        recomputed = 0.0 if lam <= 0.0 or lam >= 1.0 else math.sin(math.pi * lam) ** 2
        if abs(recomputed - declared) > 1e-9:
            raise JoinError(
                f"declared A_k[{index}]={declared} does not match sin^2(pi*lambda_vdw) "
                f"recomputation {recomputed}"
            )

    delta_a = [a_k_window[i + 1] - a_k_window[i] for i in range(len(a_k_window) - 1)]
    return delta_a, a_k_window, global_state_ids


def _load_run(
    run_id: str,
    *,
    registration: Any,
    latent_cache_dir: Path,
    ledger_dir: Path,
    target_f_k: list[float],
    target_lambda_vdw: list[float],
) -> dict[str, Any]:
    import numpy as np

    cache_npz_path = latent_cache_dir / f"latent_cache_{run_id}.npz"
    cache_report_path = latent_cache_dir / f"latent_cache_{run_id}_report.json"
    ledger_npz_path = ledger_dir / run_id / "ledger_arrays.npz"
    ledger_report_path = ledger_dir / run_id / "ledger_report.json"
    for path in (cache_npz_path, cache_report_path, ledger_npz_path, ledger_report_path):
        if not path.is_file():
            raise JoinError(f"missing required input for {run_id}: {path}")

    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    ledger_report = json.loads(ledger_report_path.read_text(encoding="utf-8"))

    if cache_report.get("run_id") != run_id:
        raise JoinError(f"{run_id}: latent cache report run_id mismatch")
    if cache_report.get("preregistration_sha256") != registration.payload_sha256:
        raise JoinError(
            f"{run_id}: latent cache was built against a different preregistration "
            "than the one currently on disk"
        )
    if ledger_report.get("run_id") != run_id:
        raise JoinError(f"{run_id}: ledger report run_id mismatch")
    if int(ledger_report.get("frame_count", -1)) != int(cache_report.get("frame_count", -2)):
        raise JoinError(f"{run_id}: ledger and latent cache disagree on frame_count")

    # The ledger's whole-document preregistration hash predates unrelated
    # later edits and is not required to match; the fields that actually
    # define this target window must, though.
    ledger_f_k = [float(value) for value in ledger_report["f_k_kj_mol"]]
    ledger_lambda_vdw = [float(value) for value in ledger_report["lambdas_vdw"]]
    if len(ledger_f_k) != len(target_f_k) or any(
        abs(a - b) > 1e-6 for a, b in zip(ledger_f_k, target_f_k)
    ):
        raise JoinError(f"{run_id}: ledger f_k_kj_mol differs from the current preregistration target")
    if len(ledger_lambda_vdw) != len(target_lambda_vdw) or any(
        abs(a - b) > 1e-6 for a, b in zip(ledger_lambda_vdw, target_lambda_vdw)
    ):
        raise JoinError(f"{run_id}: ledger lambdas_vdw differs from the current preregistration target")

    with np.load(cache_npz_path) as cache_data:
        cache_frame_index = cache_data["frame_index"]
        pooled_latent = cache_data["pooled_latent"]
    with np.load(ledger_npz_path) as ledger_data:
        ledger_frame_index = ledger_data["frame_index"]
        adjacent_gap_reduced = ledger_data["adjacent_gap_reduced"]
        log_importance_unnormalized = ledger_data["log_importance_unnormalized"]

    frame_count = int(cache_report["frame_count"])
    expected_frame_index = np.arange(frame_count, dtype=cache_frame_index.dtype)
    if not np.array_equal(cache_frame_index, expected_frame_index):
        raise JoinError(f"{run_id}: latent cache frame_index is not exactly 0..{frame_count - 1}")
    if not np.array_equal(ledger_frame_index, expected_frame_index.astype(ledger_frame_index.dtype)):
        raise JoinError(f"{run_id}: ledger frame_index is not exactly 0..{frame_count - 1}")

    if pooled_latent.shape[0] != frame_count:
        raise JoinError(f"{run_id}: pooled_latent frame dimension differs from frame_count")
    if adjacent_gap_reduced.shape[0] != frame_count or log_importance_unnormalized.shape[0] != frame_count:
        raise JoinError(f"{run_id}: ledger arrays' frame dimension differs from frame_count")
    if adjacent_gap_reduced.shape[1] + 1 != log_importance_unnormalized.shape[1]:
        raise JoinError(f"{run_id}: edge count and target-state count are inconsistent")

    return {
        "pooled_latent": pooled_latent.astype(np.float64),
        "adjacent_gap_reduced": adjacent_gap_reduced.astype(np.float64),
        "log_importance_unnormalized": log_importance_unnormalized.astype(np.float64),
        "frame_count": frame_count,
        "cache_npz_sha256": _sha256_file(cache_npz_path),
        "ledger_npz_sha256": _sha256_file(ledger_npz_path),
        "cache_report_sha256": _sha256_file(cache_report_path),
        "ledger_report_sha256": _sha256_file(ledger_report_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default="protocols/EXP-012_preregistration.json")
    parser.add_argument(
        "--latent-cache-dir", required=True,
        help="directory with latent_cache_<run_id>.npz / _report.json (scripts/build_exp012_teacher_latent_cache.py output)",
    )
    parser.add_argument(
        "--ledger-dir", default="output/outer_lambda_exp012/mm_ledger_cuda",
        help="directory with <run_id>/ledger_arrays.npz and ledger_report.json",
    )
    parser.add_argument(
        "--run-id", action="append", default=None,
        help="repeatable; default is every run_id registered in the preregistration",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen join: {args.output}")

    import numpy as np

    registration = load_preregistration(
        Path(args.preregistration) if Path(args.preregistration).is_absolute()
        else ROOT / args.preregistration,
        workspace_root=ROOT, verify_files=True,
    )
    delta_a, a_k_window, global_state_ids = _resolve_delta_a(registration)
    target_f_k = [float(value) for value in registration.payload["target"]["ledger_slice"]["f_k_kj_mol"]]
    lambda_vdw_full = [float(value) for value in registration.payload["target"]["global_schedule"]["lambda_vdw"]]
    target_lambda_vdw = [lambda_vdw_full[index] for index in global_state_ids]

    run_ids = args.run_id or [run["run_id"] for run in registration.payload["inputs"]["runs"]]
    latent_cache_dir = Path(args.latent_cache_dir)
    ledger_dir = Path(args.ledger_dir)

    per_run = {}
    for run_id in run_ids:
        per_run[run_id] = _load_run(
            run_id, registration=registration,
            latent_cache_dir=latent_cache_dir, ledger_dir=ledger_dir,
            target_f_k=target_f_k, target_lambda_vdw=target_lambda_vdw,
        )

    partition_labels = sorted(per_run)
    pooled_latent = np.concatenate([per_run[run_id]["pooled_latent"] for run_id in partition_labels], axis=0)
    adjacent_gap_reduced = np.concatenate(
        [per_run[run_id]["adjacent_gap_reduced"] for run_id in partition_labels], axis=0
    )
    log_importance_unnormalized = np.concatenate(
        [per_run[run_id]["log_importance_unnormalized"] for run_id in partition_labels], axis=0
    )
    partition_index = np.concatenate(
        [
            np.full(per_run[run_id]["frame_count"], label_index, dtype=np.int64)
            for label_index, run_id in enumerate(partition_labels)
        ]
    )

    output_path = Path(args.output)
    _atomic_write_npz(
        output_path,
        {
            "pooled_latent": pooled_latent.astype(np.float32),
            "adjacent_gap_reduced": adjacent_gap_reduced,
            "log_importance_unnormalized": log_importance_unnormalized,
            "partition_index": partition_index,
            "delta_A": np.asarray(delta_a, dtype=np.float64),
            "A_k_window": np.asarray(a_k_window, dtype=np.float64),
        },
    )
    npz_sha = _sha256_file(output_path)

    body = {
        "schema_version": "exp012-teacher-latent-ledger-join-v1",
        "status": "COMPLETED_JOIN_ONLY_NOT_FIT",
        "preregistration_sha256": registration.payload_sha256,
        "run_id_by_partition_index": partition_labels,
        "frame_count_by_run": {run_id: per_run[run_id]["frame_count"] for run_id in partition_labels},
        "total_frame_count": int(partition_index.shape[0]),
        "global_state_ids": global_state_ids,
        "target_f_k_kj_mol": target_f_k,
        "target_lambda_vdw": target_lambda_vdw,
        "A_k_window": a_k_window,
        "delta_A": delta_a,
        "A_definition": "sin_squared_pi_lambda_vdw",
        "npz_path": str(output_path.resolve()),
        "npz_sha256": npz_sha,
        "inputs": {
            run_id: {
                "cache_npz_sha256": per_run[run_id]["cache_npz_sha256"],
                "cache_report_sha256": per_run[run_id]["cache_report_sha256"],
                "ledger_npz_sha256": per_run[run_id]["ledger_npz_sha256"],
                "ledger_report_sha256": per_run[run_id]["ledger_report_sha256"],
            }
            for run_id in partition_labels
        },
        "policy": {
            "provisional_not_sealed": True,
            "scientific_qualification": False,
            "decision_reference": "DEC-030(c)",
            "a_k_frozen": True,
            "a_k_recomputed_and_checked": True,
            "training_executed": False,
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(output_path.with_name(output_path.stem + "_report.json"), report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
