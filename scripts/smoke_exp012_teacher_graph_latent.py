#!/usr/bin/env python
"""DEC-032: real MACE latent/autograd smoke over the per-frame teacher graph.

Unlike ``scripts/smoke_exp012_mace_latent.py`` (which requires a sealed,
fixed environment manifest/atom mapping), this builds ``S_a`` -- the exact
two-hop closure -- fresh for whichever frame(s) are given, via
``local_residual.teacher_graph.build_teacher_graph_for_frame``. There is no
manifest to seal here: the offline teacher never enters OpenMM, so its graph
never needs to stay fixed across frames. This is the tool DEC-032 step 5
calls for -- run CPU/CUDA C1 on the actual worst-case frame identified by
``scripts/audit_exp012_per_frame_teacher_graph_geometry.py``, not a guess.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# DEC-027: exactly two preregistered encoder variants are allowed.
ENCODER_VARIANTS = {
    6.0: "original_6a",
    5.0: "derived_5a",
}

from local_residual.environment import canonical_json_bytes  # noqa: E402
from local_residual.mace_graph import MaceGraphConfig  # noqa: E402
from local_residual.mace_latent import MaceLatentBasisAdapter, load_c0_report  # noqa: E402
from local_residual.teacher_graph import build_teacher_graph_for_frame  # noqa: E402


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _indices(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("frame indices must be comma-separated integers") from exc
    if not result or len(result) > 3 or any(index < 0 for index in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("provide 1--3 unique non-negative frame indices")
    return result


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


def _tensor_stats(tensor) -> dict:
    return {
        "finite": bool(tensor.isfinite().all().item()),
        "minimum": float(tensor.min().item()),
        "maximum": float(tensor.max().item()),
        "norm": float(tensor.norm().item()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--c0-report", required=True)
    parser.add_argument("--ligand-indices", required=True, help="JSON file with a ligand_indices array")
    parser.add_argument("--topology", required=True, help="mdtraj-compatible topology")
    parser.add_argument("--trajectory", required=True, help="mdtraj-compatible frame source")
    parser.add_argument("--frame-indices", required=True, type=_indices)
    parser.add_argument("--edge-cutoff-angstrom", required=True, type=float)
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument(
        "--product-layer-index", type=int, default=1,
        help="zero-based product layer to expose; 1 is a two-interaction/two-hop latent",
    )
    parser.add_argument(
        "--max-node-count", type=int, default=2500,
        help="fail before MACE forward if this frame's exact closure exceeds this node count",
    )
    parser.add_argument(
        "--memory-limit-gb", type=float,
        help="optional Linux process address-space cap applied before Torch/MACE import",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.edge_cutoff_angstrom not in ENCODER_VARIANTS:
        parser.error(
            "--edge-cutoff-angstrom must be exactly one of the preregistered "
            f"EXP-012 encoder variants: {sorted(ENCODER_VARIANTS)}"
        )
    encoder_variant = ENCODER_VARIANTS[args.edge_cutoff_angstrom]
    if Path(args.output).exists():
        parser.error(f"--output already exists, refusing to overwrite a frozen report: {args.output}")
    if args.memory_limit_gb is not None:
        if args.memory_limit_gb <= 0.0:
            parser.error("--memory-limit-gb must be positive")
        try:
            import resource

            requested = int(args.memory_limit_gb * 1024**3)
            _, hard = resource.getrlimit(resource.RLIMIT_AS)
            soft = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
        except (ImportError, OSError, ValueError) as exc:
            parser.error(f"cannot apply --memory-limit-gb: {exc}")

    import mdtraj
    import torch

    started = time.perf_counter()
    ligand_payload = json.loads(Path(args.ligand_indices).expanduser().read_text(encoding="utf-8"))
    ligand_indices = ligand_payload.get("ligand_indices")
    if not isinstance(ligand_indices, list) or not ligand_indices:
        raise RuntimeError("--ligand-indices JSON must contain a non-empty ligand_indices array")
    ligand_indices = sorted(int(index) for index in ligand_indices)

    c0 = load_c0_report(args.c0_report)
    interaction_layers = args.product_layer_index + 1
    graph_config = MaceGraphConfig(
        edge_cutoff_angstrom=args.edge_cutoff_angstrom,
        interaction_layers=interaction_layers,
        geometric_upper_bound_angstrom=args.edge_cutoff_angstrom * interaction_layers,
    )
    torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    topology_object = mdtraj.load(args.topology).topology
    atomic_numbers_by_topology_index = [
        int(atom.element.atomic_number) for atom in topology_object.atoms
    ]

    adapter = None
    frames = []
    for frame_index in args.frame_indices:
        frame_started = time.perf_counter()
        trajectory = mdtraj.load_frame(args.trajectory, index=frame_index, top=args.topology)
        if trajectory.n_frames != 1 or trajectory.unitcell_vectors is None:
            raise RuntimeError("frame source must provide exactly one frame with triclinic cell vectors")
        if trajectory.n_atoms != len(atomic_numbers_by_topology_index):
            raise RuntimeError("frame atom count differs from the topology")
        positions = torch.tensor(
            trajectory.xyz[0] * 10.0, dtype=torch_dtype, device=device, requires_grad=True
        )
        cell = torch.tensor(trajectory.unitcell_vectors[0] * 10.0, dtype=torch_dtype, device=device)
        graph = build_teacher_graph_for_frame(
            positions, cell,
            ligand_indices=ligand_indices,
            atomic_numbers_by_topology_index=atomic_numbers_by_topology_index,
            model_atomic_numbers=c0["expected"]["atomic_numbers"],
            config=graph_config,
        )
        node_count = int(graph["diagnostics"]["node_count"])
        edge_count = int(graph["diagnostics"]["edge_count"])
        print(
            f"frame {frame_index}: exact closure graph ready with {node_count} nodes and "
            f"{edge_count} directed edges",
            flush=True,
        )
        if node_count > args.max_node_count:
            raise RuntimeError(
                f"graph node count {node_count} exceeds --max-node-count "
                f"{args.max_node_count}; refusing MACE forward"
            )
        if adapter is None:
            adapter = MaceLatentBasisAdapter(
                c0_report=c0, model_path=args.model, device=args.device, dtype=args.dtype,
                product_layer_index=args.product_layer_index,
            )
        print(f"frame {frame_index}: starting no-grad repeat reference", flush=True)
        with torch.no_grad():
            repeat_reference = adapter.forward(
                graph, require_coordinate_grad=False
            )["ligand_latent"].detach().clone()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        print(f"frame {frame_index}: starting single autograd forward", flush=True)
        first = adapter(graph)
        latent = first["ligand_latent"]
        repeat_max_abs = float((latent.detach() - repeat_reference).abs().max().item())
        weights = torch.linspace(1.0, 2.0, latent.shape[1], dtype=latent.dtype, device=latent.device)
        scalar_probe = (latent * weights).sum()
        coordinate_gradient = torch.autograd.grad(scalar_probe, positions)[0]
        print(f"frame {frame_index}: autograd completed", flush=True)
        ligand_topology = torch.tensor(ligand_indices, dtype=torch.long, device=device)
        topology_index = graph["topology_indices_by_mace_node_index"]
        environment_topology = topology_index[graph["environment_mask"]]
        parameter_grad_count = sum(
            parameter.grad is not None for parameter in adapter._model.parameters()
        )
        ligand_gradient = coordinate_gradient[ligand_topology]
        environment_gradient = coordinate_gradient[environment_topology]
        if not bool(torch.isfinite(coordinate_gradient).all().item()):
            raise RuntimeError("coordinate autograd produced non-finite values")
        if float(ligand_gradient.norm().item()) <= 0.0:
            raise RuntimeError("autograd smoke produced no ligand-coordinate gradient")
        if float(environment_gradient.norm().item()) <= 0.0:
            raise RuntimeError("autograd smoke produced no environment-coordinate gradient")
        if parameter_grad_count != 0:
            raise RuntimeError("frozen MACE parameters unexpectedly accumulated gradients")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        frames.append(
            {
                "frame_index": frame_index,
                "counts": graph["diagnostics"],
                "box_angstrom": cell.detach().cpu().tolist(),
                "latent": {
                    "shape": list(latent.shape),
                    "dtype": str(latent.dtype),
                    "device": str(latent.device),
                    **_tensor_stats(latent),
                },
                "repeat_forward_max_abs_difference": repeat_max_abs,
                "autograd_smoke_only_scalar_probe": float(scalar_probe.item()),
                "ligand_coordinate_gradient": _tensor_stats(ligand_gradient),
                "environment_coordinate_gradient": _tensor_stats(environment_gradient),
                "mace_parameter_grad_count": parameter_grad_count,
                "elapsed_seconds": time.perf_counter() - frame_started,
            }
        )

    inputs = {
        name: {"path": str(Path(path).resolve()), "sha256": _sha(path)}
        for name, path in (
            ("model", args.model), ("c0_report", args.c0_report),
            ("ligand_indices", args.ligand_indices),
            ("topology", args.topology), ("trajectory", args.trajectory),
        )
    }
    body = {
        "schema_version": "exp012-teacher-graph-latent-autograd-smoke-v1",
        "status": "COMPLETED_AUTOGRAD_SMOKE_ONLY",
        "encoder_variant": encoder_variant,
        "graph_policy": "per_frame_exact_two_hop_closure_no_fixed_manifest",
        "model_r_max_angstrom": float(c0["expected"]["r_max_angstrom"]),
        "graph_cutoff_angstrom": float(args.edge_cutoff_angstrom),
        "inputs": inputs,
        "frame_indices": args.frame_indices,
        "frames": frames,
        "elapsed_seconds": time.perf_counter() - started,
        "policy": {
            "provisional_not_sealed": True,
            "scientific_qualification": False,
            "training_executed": False,
            "full_dataset_scanned": False,
            "energy_fields_used": False,
            "fragment_subtraction_used": False,
            "numpy_descriptor_used": False,
            "latent_detached": False,
            "complete_residue_expansion": False,
            "fixed_environment_manifest": False,
            "decision_reference": "DEC-032",
            "scalar_probe_role": "autograd_smoke_only_not_candidate_B",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
