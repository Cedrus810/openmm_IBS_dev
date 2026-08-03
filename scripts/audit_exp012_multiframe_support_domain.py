#!/usr/bin/env python
"""DEC-030 step (a): report-only multi-frame support-domain audit.

`derived_5a`'s environment manifest fixes its candidate atom set from a single
reference frame (``run1/frame0``).  Building an offline multi-frame latent
cache (DEC-030 step b) with that same fixed manifest assumes the frame0
closure stays adequate for every other frame too -- atoms genuinely move, so
that assumption has never actually been checked against the trajectories the
cache will be built from.

This script checks it, across every frame of the registered EXP-012
target-ledger trajectories (``hard_window0_run1/2/3``).  Per DEC-030 it is
explicit report-only: a frame whose true two-hop cutoff-graph closure reaches
outside the fixed manifest is recorded as a violation, never raised as an
error.  Only malformed inputs (a tampered manifest, a trajectory that no
longer matches its registered SHA-256/frame count) are fail-closed, because
those are audit-integrity failures, not the support-domain question itself.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# DEC-027: exactly two preregistered encoder variants are allowed. Mirrors
# scripts/smoke_exp012_mace_latent.py -- a genuinely new cutoff requires its
# own preregistered variant, not a silent third value here.
ENCODER_VARIANTS = {
    6.0: "original_6a",
    5.0: "derived_5a",
}

from exp012_xed.schema import load_preregistration  # noqa: E402
from local_residual.environment import canonical_json_bytes, load_environment_manifest  # noqa: E402
from local_residual.mace_graph import topology_n_hop_closure  # noqa: E402


class SupportDomainAuditError(ValueError):
    """The audit's own inputs are malformed; this is not a support-domain finding."""


def _default_num_workers() -> int:
    """Respect a cgroup/SLURM CPU affinity mask instead of the whole node.

    ``os.cpu_count()`` reports every core on the physical machine, which
    over-subscribes a shared cluster node when the job was only granted a
    subset via SLURM/cgroups. ``os.sched_getaffinity(0)`` (Linux only)
    reports what this process can actually use.
    """
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count() or 1


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_run(registration: Any, run_id: str) -> dict[str, Any]:
    runs = registration.payload["inputs"]["runs"]
    selected = [run for run in runs if run["run_id"] == run_id]
    if len(selected) != 1:
        raise SupportDomainAuditError(f"unknown or duplicate run_id: {run_id}")
    return selected[0]


def audit_frame(
    positions_angstrom: Any,
    cell_angstrom: Any,
    *,
    ligand_indices: list[int],
    fixed_atom_indices: set[int],
    edge_cutoff_angstrom: float,
    interaction_layers: int,
) -> dict[str, Any]:
    """Report-only: compare one frame's true closure against a fixed atom set.

    Never raises on a violation -- it only measures whether the true
    cutoff-graph closure at this frame's live positions stays inside
    ``fixed_atom_indices``. The caller decides whether/how to aggregate.
    """

    closure, hop = topology_n_hop_closure(
        positions_angstrom,
        cell_angstrom,
        ligand_indices,
        edge_cutoff_angstrom=edge_cutoff_angstrom,
        interaction_layers=interaction_layers,
    )
    reached = {int(index) for index in closure.nonzero(as_tuple=False).flatten().tolist()}
    omitted = sorted(reached - set(fixed_atom_indices))
    hop_list = hop.tolist()
    return {
        "closure_atom_count": len(reached),
        "omitted_atom_count": len(omitted),
        "omitted_atom_indices": omitted,
        "omitted_atom_hops": [hop_list[index] for index in omitted],
    }


def _init_worker() -> None:
    """Pin each worker to one Torch thread to avoid core oversubscription."""
    import torch

    torch.set_num_threads(1)


def _audit_frame_worker(task: tuple) -> dict[str, Any]:
    """Picklable top-level entry point for ProcessPoolExecutor.

    Frames are audited on CPU only: no MACE model runs in this audit, so a
    GPU buys nothing here and one CUDA context per worker process would be
    pure overhead. Cross-process parallelism over frames is the real speedup.
    """
    (
        frame_index,
        positions_angstrom,
        cell_angstrom,
        ligand_indices,
        fixed_atom_indices,
        edge_cutoff_angstrom,
        interaction_layers,
        dtype_name,
    ) = task
    import torch

    torch_dtype = torch.float64 if dtype_name == "float64" else torch.float32
    positions = torch.tensor(positions_angstrom, dtype=torch_dtype)
    cell = torch.tensor(cell_angstrom, dtype=torch_dtype)
    result = audit_frame(
        positions, cell,
        ligand_indices=ligand_indices,
        fixed_atom_indices=fixed_atom_indices,
        edge_cutoff_angstrom=edge_cutoff_angstrom,
        interaction_layers=interaction_layers,
    )
    return {"frame_index": frame_index, **result}


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", default="protocols/EXP-012_preregistration.json")
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--edge-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--interaction-layers", type=int, required=True)
    parser.add_argument(
        "--run-id", action="append", default=None,
        help="repeatable; default is every run_id registered in the preregistration",
    )
    parser.add_argument(
        "--frame-stride", type=int, default=1,
        help="audit every Nth frame instead of all frames (report-only, not a sampling gate)",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--num-workers", type=int, default=_default_num_workers(),
        help=(
            "frames are independent, so this audit fans out across processes; "
            "default is this process's CPU affinity mask (no GPU is used -- no "
            "MACE model runs here, so a GPU buys nothing and per-frame closure "
            "checks are the only cost)"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.edge_cutoff_angstrom not in ENCODER_VARIANTS:
        parser.error(
            "--edge-cutoff-angstrom must be exactly one of the preregistered "
            f"EXP-012 encoder variants: {sorted(ENCODER_VARIANTS)} "
            "(6.0 -> original_6a per DEC-024, 5.0 -> derived_5a per DEC-027)"
        )
    if args.frame_stride < 1:
        parser.error("--frame-stride must be a positive integer")
    if args.num_workers < 1:
        parser.error("--num-workers must be a positive integer")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")

    import mdtraj as md

    registration = load_preregistration(
        Path(args.preregistration) if Path(args.preregistration).is_absolute()
        else ROOT / args.preregistration,
        workspace_root=ROOT,
        verify_files=True,
    )
    manifest = load_environment_manifest(
        args.environment_manifest, workspace_root=ROOT, verify_sources=True,
    )
    payload = manifest["payload"]
    ligand_indices = list(payload["ligand_indices"])
    fixed_atom_indices = set(ligand_indices) | set(payload["environment_candidate_indices"])

    topology_relative = registration.payload["inputs"]["artifacts"]["topology"]["path"]
    topology_path = ROOT / topology_relative

    run_ids = args.run_id or [run["run_id"] for run in registration.payload["inputs"]["runs"]]

    started = time.perf_counter()
    run_reports = []
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=_init_worker) as executor:
        for run_id in run_ids:
            run = _select_run(registration, run_id)
            trajectory_relative = run["trajectory"]["path"]
            trajectory_path = ROOT / trajectory_relative
            observed_sha = _sha256_file(trajectory_path)
            if observed_sha != run["trajectory"]["sha256"]:
                raise SupportDomainAuditError(
                    f"trajectory SHA-256 mismatch for {run_id}: {trajectory_path}"
                )
            expected_frames = int(run["frame_count"])
            trajectory = md.load(str(trajectory_path), top=str(topology_path))
            if trajectory.n_frames != expected_frames:
                raise SupportDomainAuditError(
                    f"{run_id}: trajectory frame count differs from preregistration"
                )
            if trajectory.unitcell_vectors is None:
                raise SupportDomainAuditError(f"{run_id}: trajectory lacks periodic box vectors")

            frame_indices = list(range(0, expected_frames, args.frame_stride))
            run_started = time.perf_counter()
            tasks = (
                (
                    frame_index,
                    trajectory.xyz[frame_index] * 10.0,
                    trajectory.unitcell_vectors[frame_index] * 10.0,
                    ligand_indices,
                    fixed_atom_indices,
                    args.edge_cutoff_angstrom,
                    args.interaction_layers,
                    args.dtype,
                )
                for frame_index in frame_indices
            )
            chunksize = max(1, len(frame_indices) // (4 * args.num_workers))
            violations = []
            max_omitted = 0
            completed = 0
            for result in executor.map(_audit_frame_worker, tasks, chunksize=chunksize):
                completed += 1
                if result["omitted_atom_count"] > 0:
                    max_omitted = max(max_omitted, result["omitted_atom_count"])
                    violations.append(result)
                if completed % 25 == 0 or completed == len(frame_indices):
                    print(
                        f"{run_id}: {completed}/{len(frame_indices)} frames audited, "
                        f"{len(violations)} violation(s) so far",
                        flush=True,
                    )
            run_reports.append(
                {
                    "run_id": run_id,
                    "trajectory": {"path": trajectory_relative, "sha256": observed_sha},
                    "frame_count": expected_frames,
                    "audited_frame_count": len(frame_indices),
                    "frame_stride": args.frame_stride,
                    "violation_frame_count": len(violations),
                    "max_omitted_atom_count": max_omitted,
                    "violations": sorted(violations, key=lambda item: item["frame_index"]),
                    "elapsed_seconds": time.perf_counter() - run_started,
                }
            )

    body = {
        "schema_version": "exp012-multiframe-support-domain-audit-v1",
        "status": "COMPLETED_REPORT_ONLY",
        "encoder_variant": ENCODER_VARIANTS[args.edge_cutoff_angstrom],
        "environment_manifest_sha256": manifest["canonical_sha256"],
        "environment_manifest_path": str(Path(args.environment_manifest).resolve()),
        "preregistration_sha256": registration.payload_sha256,
        "edge_cutoff_angstrom": args.edge_cutoff_angstrom,
        "interaction_layers": args.interaction_layers,
        "fixed_atom_count": len(fixed_atom_indices),
        "num_workers": args.num_workers,
        "runs": run_reports,
        "total_violation_frame_count": sum(run["violation_frame_count"] for run in run_reports),
        "total_audited_frame_count": sum(run["audited_frame_count"] for run in run_reports),
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "provisional_not_sealed": True,
            "scientific_qualification": False,
            "hard_gate": False,
            "raises_on_violation": False,
            "decision_reference": "DEC-030",
            "training_executed": False,
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
