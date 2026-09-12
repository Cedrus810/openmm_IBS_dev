#!/usr/bin/env python
"""Build and audit the sealed EXP-020 SoftLift ``softlift_dataset_v1`` artifact.

This command is the only D0 writer.  It reads the three frozen EXP-012 runs,
adds per-frame box/displacement/periodic-image data, and refuses to overwrite
any existing EXP-019 output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_residual.softlift_dataset import build_dataset_v1, sha256_file  # noqa: E402


class D0Error(RuntimeError):
    """A sealed protocol or source identity check failed."""


def _canonical_protocol_hash(payload: dict) -> str:
    normalized = json.loads(json.dumps(payload))
    normalized.get("freeze", {}).pop("payload_sha256", None)
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _load_sealed_protocol(path: Path) -> tuple[dict, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment_id") != "EXP-020" or payload.get("freeze", {}).get("status") != "sealed":
        raise D0Error("EXP-020 SoftLift protocol must be sealed")
    expected = payload.get("freeze", {}).get("payload_sha256")
    actual = _canonical_protocol_hash(payload)
    if expected != actual:
        raise D0Error(f"EXP-020 protocol hash mismatch: expected {expected}, calculated {actual}")
    return payload, actual


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _verify_file(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise D0Error(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise D0Error(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def _parent_schedule(parent: dict) -> tuple[list[float], list[float]]:
    target = parent["target"]
    schedule = target["global_schedule"]
    if schedule.get("A_definition") != "sin_squared_pi_lambda_vdw":
        raise D0Error("parent schedule is not sin^2(pi*lambda_vdw)")
    state_ids = [int(value) for value in target["ledger_slice"]["global_state_ids"]]
    lambdas = [float(value) for value in schedule["lambda_vdw"]]
    declared = [float(value) for value in schedule["A_k"]]
    window_lambdas = [lambdas[index] for index in state_ids]
    window_A = [declared[index] for index in state_ids]
    for lam, coefficient in zip(window_lambdas, window_A):
        recomputed = 0.0 if lam <= 0.0 or lam >= 1.0 else math.sin(math.pi * lam) ** 2
        if abs(recomputed - coefficient) > 1e-9:
            raise D0Error("parent A_k schedule failed independent recomputation")
    delta = [window_A[index + 1] - window_A[index] for index in range(len(window_A) - 1)]
    return delta, window_A


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default="protocols/EXP-020_preregistration.json")
    parser.add_argument("--parent-preregistration", default="protocols/EXP-012_preregistration.json")
    parser.add_argument("--topology", default=None, help="override only for an explicitly identical frozen topology")
    parser.add_argument("--ligand-indices", default="output/ligand_indices.json")
    parser.add_argument("--ledger-dir", default=None)
    parser.add_argument("--trajectory", action="append", default=[], metavar="RUN_ID=PATH")
    parser.add_argument("--output", default="output/outer_lambda_exp020_softlift/dataset/softlift_dataset_v1.npz")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    protocol_path = _resolve(ROOT, args.preregistration)
    protocol, protocol_sha256 = _load_sealed_protocol(protocol_path)
    parent_identity_record = protocol["parent_experiment"]["preregistration"]
    parent_identity_path = _resolve(ROOT, parent_identity_record["path"])
    _verify_file(parent_identity_path, parent_identity_record["sha256"], "parent EXP-019 preregistration")
    parent_path = _resolve(ROOT, args.parent_preregistration)
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_exp012_record = protocol["parent_exp012"]
    _verify_file(parent_path, parent_exp012_record["sha256"], "parent EXP-012 preregistration")
    frozen_dataset = protocol["inputs"]["frozen_exp012_dataset"]
    _verify_file(_resolve(ROOT, frozen_dataset["path"]), frozen_dataset["sha256"], "frozen EXP-012 dataset")
    delta_A, A_k_window = _parent_schedule(parent)

    topology_record = parent["inputs"]["artifacts"]["topology"]
    topology_path = _resolve(ROOT, args.topology) if args.topology else _resolve(ROOT, topology_record["path"])
    _verify_file(topology_path, topology_record["sha256"], "topology")
    ligand_payload = json.loads(_resolve(ROOT, args.ligand_indices).read_text(encoding="utf-8"))
    ligand_indices = [int(value) for value in ligand_payload["ligand_indices"]]

    overrides: dict[str, str] = {}
    for item in args.trajectory:
        if "=" not in item:
            parser.error("--trajectory must be RUN_ID=PATH")
        run_id, path = item.split("=", 1)
        overrides[run_id] = path
    ledger_dir = _resolve(ROOT, args.ledger_dir) if args.ledger_dir else ROOT / "output/outer_lambda_exp012/mm_ledger_cuda"
    runs = []
    for record in protocol["inputs"]["runs"]:
        run_id = record["run_id"]
        trajectory_path = _resolve(ROOT, overrides.get(run_id, record["trajectory_path"]))
        ledger_path = _resolve(ROOT, str(Path(args.ledger_dir) / run_id / "ledger_arrays.npz")) if args.ledger_dir else _resolve(ROOT, record["ledger_path"])
        ledger_report_path = _resolve(ROOT, str(Path(args.ledger_dir) / run_id / "ledger_report.json")) if args.ledger_dir else _resolve(ROOT, record["ledger_report_path"])
        _verify_file(trajectory_path, record["trajectory_sha256"], f"{run_id} trajectory")
        _verify_file(ledger_path, record["ledger_sha256"], f"{run_id} ledger")
        _verify_file(ledger_report_path, record["ledger_report_sha256"], f"{run_id} ledger report")
        runs.append({
            "run_id": run_id,
            "trajectory_path": trajectory_path,
            "ledger_path": ledger_path,
            "ledger_report_path": ledger_report_path,
        })

    report = build_dataset_v1(
        runs=runs,
        ligand_topology_indices=ligand_indices,
        topology_path=topology_path,
        output_path=_resolve(ROOT, args.output),
        outer_cutoff_angstrom=float(protocol["scope"]["physical_cutoff_angstrom"]),
        min_distance_support_angstrom=float(protocol["geometry_contract"]["min_distance_support_angstrom"]),
        max_environment_atoms=int(protocol["geometry_contract"]["budget"]["unique_environment_atoms_hard"]),
        max_edges=int(protocol["geometry_contract"]["budget"]["directed_edges_hard"]),
        max_neighbors_per_ligand=int(protocol["geometry_contract"]["budget"]["neighbors_per_ligand_hard"]),
        device=args.device,
        delta_A=delta_A,
        A_k_window=A_k_window,
        protocol_sha256=protocol_sha256,
    )
    print(json.dumps({"status": report["status"], "dataset_sha256": report["dataset_sha256"], "report_sha256": report["report_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
