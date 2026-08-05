#!/usr/bin/env python
"""DEC-039 d0-5 gap 2: does the student funnel need a CPU/CUDA membership split?

DEC-038 validated ``local_residual.geometry.ligand_environment_cross_edges``
(the student's online ligand<->environment funnel) against the teacher's
canonical membership, entirely on CPU float64. It has never been run directly
on CUDA float32 -- the precision/device combination the online student would
actually use every MD step. This matters because the exact same class of bug
has already happened once in this project: the teacher's own two-hop closure
independently decided cutoff-boundary membership differently on CPU float64
vs CUDA float32 (DEC-025/026), which is why the teacher's graph construction
was redesigned into a "decide membership once on CPU float64, only move the
already-decided discrete result to the execution device" split (DEC-032/033
Option C, ``local_residual.teacher_graph.compute_canonical_graph_membership``
+ ``build_teacher_graph_from_membership``).

This script answers, with real coordinates rather than by analogy: does
running ``ligand_environment_cross_edges`` directly on CUDA float32 -- with no
such split, the simplest possible online design -- decide the *same* discrete
edge membership as running it on CPU float64? If yes, the funnel needs no
Option-C-style split and DEC-038's design stands as-is for CUDA deployment. If
no, the online design must adopt the same split the teacher already uses,
before D2/D3 -- a real, consequential fork, not a formality.

No student model, no MACE, no TorchForce, no OpenMM integrator, no training,
and no NVT run anywhere in this script -- only
``local_residual.geometry.ligand_environment_cross_edges`` and
``local_residual.geometry.quintic_c2_cutoff`` on two real frames: an ordinary
frame (run1/frame0) and the real worst-edge-count frame identified by
``scripts/audit_exp012_student_environment_funnel_geometry.py``'s 1500-frame
sweep (run2/frame202, 1464 directed edges).
"""

from __future__ import annotations

import argparse
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

from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.geometry import ligand_environment_cross_edges, quintic_c2_cutoff  # noqa: E402


def _sha(path: str | Path) -> str:
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


def _funnel(positions, cell, ligand_tensor, environment_tensor, cutoff):
    result = ligand_environment_cross_edges(positions, cell, ligand_tensor, environment_tensor, outer_cutoff=cutoff)
    edge_index = result["edge_index"]
    distance = result["distance"]
    pairs = {
        (int(edge_index[0, i]), int(edge_index[1, i])): float(distance[i])
        for i in range(edge_index.shape[1])
    }
    return result, pairs


def _one_frame(
    *, run_id: str, frame_index: int, topology: str, trajectory: str,
    ligand_indices: list[int], cutoff: float, inner_cutoff: float,
) -> dict[str, Any]:
    import mdtraj
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this consistency check; none is available")

    frame = mdtraj.load_frame(trajectory, index=frame_index, top=topology)
    if frame.n_frames != 1 or frame.unitcell_vectors is None:
        raise RuntimeError("frame source must provide exactly one frame with triclinic cell vectors")
    positions_angstrom = frame.xyz[0] * 10.0
    cell_angstrom = frame.unitcell_vectors[0] * 10.0
    atom_count = int(positions_angstrom.shape[0])
    ligand_sorted = sorted(ligand_indices)
    environment_sorted = sorted(set(range(atom_count)) - set(ligand_sorted))

    positions_cpu64 = torch.tensor(positions_angstrom, dtype=torch.float64)
    cell_cpu64 = torch.tensor(cell_angstrom, dtype=torch.float64)
    ligand_cpu = torch.tensor(ligand_sorted, dtype=torch.int64)
    environment_cpu = torch.tensor(environment_sorted, dtype=torch.int64)
    funnel_cpu64, pairs_cpu64 = _funnel(positions_cpu64, cell_cpu64, ligand_cpu, environment_cpu, cutoff)

    device = torch.device("cuda")
    positions_cuda32 = torch.tensor(positions_angstrom, dtype=torch.float32, device=device)
    cell_cuda32 = torch.tensor(cell_angstrom, dtype=torch.float32, device=device)
    ligand_cuda = ligand_cpu.to(device)
    environment_cuda = environment_cpu.to(device)
    funnel_cuda32, pairs_cuda32 = _funnel(
        positions_cuda32, cell_cuda32, ligand_cuda, environment_cuda, cutoff
    )

    cpu_keys = set(pairs_cpu64)
    cuda_keys = set(pairs_cuda32)
    cpu_only = sorted(cpu_keys - cuda_keys)
    cuda_only = sorted(cuda_keys - cpu_keys)
    common = sorted(cpu_keys & cuda_keys)
    distance_abs_diffs = [abs(pairs_cpu64[key] - pairs_cuda32[key]) for key in common]

    common_cpu_distance = torch.tensor([pairs_cpu64[key] for key in common], dtype=torch.float64)
    common_cuda_distance = torch.tensor([pairs_cuda32[key] for key in common], dtype=torch.float64)
    weight_cpu64 = quintic_c2_cutoff(common_cpu_distance, inner_cutoff=inner_cutoff, outer_cutoff=cutoff)
    weight_cuda32_from_cuda_distance = quintic_c2_cutoff(
        common_cuda_distance.to(dtype=torch.float32, device=device),
        inner_cutoff=inner_cutoff, outer_cutoff=cutoff,
    ).to(dtype=torch.float64, device="cpu")
    quintic_weight_abs_diffs = (weight_cpu64 - weight_cuda32_from_cuda_distance).abs().tolist()

    sweep = []
    step = 0.05
    distance = cutoff - 0.5
    while distance <= cutoff + 0.5 + 1e-9:
        cpu_value = float(
            quintic_c2_cutoff(
                torch.tensor([distance], dtype=torch.float64), inner_cutoff=inner_cutoff, outer_cutoff=cutoff
            ).item()
        )
        cuda_value = float(
            quintic_c2_cutoff(
                torch.tensor([distance], dtype=torch.float32, device=device),
                inner_cutoff=inner_cutoff, outer_cutoff=cutoff,
            ).item()
        )
        sweep.append(
            {"distance_angstrom": distance, "weight_cpu_float64": cpu_value, "weight_cuda_float32": cuda_value,
             "abs_diff": abs(cpu_value - cuda_value)}
        )
        distance += step

    return {
        "run_id": run_id,
        "frame_index": frame_index,
        "cpu_float64": {
            "s1_atom_count": int(funnel_cpu64["edge_index"][1].unique().numel()) if funnel_cpu64["edge_index"].numel() else 0,
            "edge_count": int(funnel_cpu64["edge_index"].shape[1]),
        },
        "cuda_float32": {
            "s1_atom_count": int(funnel_cuda32["edge_index"][1].unique().numel()) if funnel_cuda32["edge_index"].numel() else 0,
            "edge_count": int(funnel_cuda32["edge_index"].shape[1]),
        },
        "edge_set_identical": (not cpu_only) and (not cuda_only),
        "cpu_only_pairs": [
            {"ligand_topology_index": key[0], "environment_topology_index": key[1], "distance_angstrom": pairs_cpu64[key]}
            for key in cpu_only
        ],
        "cuda_only_pairs": [
            {"ligand_topology_index": key[0], "environment_topology_index": key[1], "distance_angstrom": pairs_cuda32[key]}
            for key in cuda_only
        ],
        "common_pair_count": len(common),
        "common_pair_distance_max_abs_diff": max(distance_abs_diffs) if distance_abs_diffs else 0.0,
        "common_pair_quintic_weight_max_abs_diff": max(quintic_weight_abs_diffs) if quintic_weight_abs_diffs else 0.0,
        "boundary_sweep_inner_cutoff_angstrom": inner_cutoff,
        "boundary_sweep_max_abs_diff": max(point["abs_diff"] for point in sweep),
        "boundary_sweep": sweep,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--ligand-indices", required=True, help="JSON file with a ligand_indices array")
    parser.add_argument(
        "--trajectory", action="append", required=True, nargs=3,
        metavar=("RUN_ID", "TRAJECTORY_PATH", "FRAME_INDEX"),
        help="repeatable; e.g. --trajectory hard_window0_run1 path/to/run1.dcd 0",
    )
    parser.add_argument("--edge-cutoff-angstrom", type=float, required=True)
    parser.add_argument("--boundary-inner-cutoff-angstrom", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    if args.edge_cutoff_angstrom != 5.0:
        parser.error("this smoke is scoped to the DEC-038-frozen student funnel cutoff (5.0)")
    inner_cutoff = (
        args.boundary_inner_cutoff_angstrom
        if args.boundary_inner_cutoff_angstrom is not None
        else args.edge_cutoff_angstrom - 1.0
    )
    if not (0.0 < inner_cutoff < args.edge_cutoff_angstrom):
        parser.error("--boundary-inner-cutoff-angstrom must satisfy 0 < inner < outer_cutoff")
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")

    started_wall = time.perf_counter()
    ligand_payload = json.loads(Path(args.ligand_indices).expanduser().read_text(encoding="utf-8"))
    ligand_indices_raw = ligand_payload.get("ligand_indices")
    if not isinstance(ligand_indices_raw, list) or not ligand_indices_raw:
        raise RuntimeError("--ligand-indices JSON must contain a non-empty ligand_indices array")
    ligand_indices = [int(index) for index in ligand_indices_raw]

    frames = []
    input_paths = {"topology": args.topology, "ligand_indices": args.ligand_indices}
    for run_id, trajectory_path, frame_index_raw in args.trajectory:
        frame_index = int(frame_index_raw)
        frames.append(
            _one_frame(
                run_id=run_id, frame_index=frame_index, topology=args.topology, trajectory=trajectory_path,
                ligand_indices=ligand_indices, cutoff=args.edge_cutoff_angstrom, inner_cutoff=inner_cutoff,
            )
        )
        input_paths[f"trajectory_{run_id}"] = trajectory_path

    all_edge_sets_identical = all(frame["edge_set_identical"] for frame in frames)
    recommendation = (
        "direct CUDA float32 execution of ligand_environment_cross_edges is safe on these frames; "
        "no CPU-float64-decide-then-move split is required for this design"
        if all_edge_sets_identical
        else "boundary disagreement found between CPU float64 and CUDA float32 execution; "
             "the online funnel must adopt the same CPU-float64-decide/GPU-execute split already "
             "used by local_residual.teacher_graph (DEC-032/033 Option C) before D2/D3"
    )

    inputs = {name: {"path": str(Path(path).resolve()), "sha256": _sha(path)} for name, path in input_paths.items()}
    body = {
        "schema_version": "exp012-student-funnel-cuda-consistency-v1",
        "edge_cutoff_angstrom": args.edge_cutoff_angstrom,
        "frames": frames,
        "all_edge_sets_identical": all_edge_sets_identical,
        "recommendation": recommendation,
        "inputs": inputs,
        "elapsed_seconds": time.perf_counter() - started_wall,
        "policy": {
            "training_executed": False,
            "torchforce_used": False,
            "nvt_executed": False,
            "student_model_executed": False,
            "decision_reference": "DEC-039_pending_d0-5_gap_2",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    print(f"all_edge_sets_identical={all_edge_sets_identical}")
    print(recommendation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
