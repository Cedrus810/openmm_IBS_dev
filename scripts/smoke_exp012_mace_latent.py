#!/usr/bin/env python
"""Run a 1--3 frame EXP-012 real MACE latent/autograd smoke (never training)."""

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

# DEC-027: exactly two preregistered encoder variants are allowed.  This is a
# closed enumeration, not an arbitrary-cutoff escape hatch -- a genuinely new
# cutoff requires its own preregistered variant name and decision entry, not
# a silent third value here.
ENCODER_VARIANTS = {
    6.0: "original_6a",
    5.0: "derived_5a",
}

from local_residual.atom_mapping import load_atom_mapping  # noqa: E402
from local_residual.environment import canonical_json_bytes, load_environment_manifest  # noqa: E402
from local_residual.mace_graph import MaceGraphConfig, build_mace_graph  # noqa: E402
from local_residual.mace_latent import MaceLatentBasisAdapter, load_c0_report  # noqa: E402


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
    parser.add_argument("--environment-manifest", required=True)
    parser.add_argument("--atom-mapping", required=True)
    parser.add_argument("--topology", required=True, help="mdtraj-compatible topology")
    parser.add_argument("--trajectory", required=True, help="mdtraj-compatible frame source")
    parser.add_argument("--frame-indices", required=True, type=_indices)
    parser.add_argument("--edge-cutoff-angstrom", required=True, type=float)
    parser.add_argument(
        "--geometric-upper-bound-angstrom",
        "--candidate-support-radius-angstrom",
        dest="geometric_upper_bound_angstrom",
        required=True,
        type=float,
        help=(
            "cutoff*layers geometric upper bound (the old --candidate-support-radius-angstrom "
            "name remains as a compatibility alias); node completeness uses exact graph closure"
        ),
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument(
        "--product-layer-index",
        type=int,
        default=1,
        help="zero-based product layer to expose; 1 is a two-interaction/two-hop latent",
    )
    parser.add_argument(
        "--max-node-count",
        type=int,
        default=2500,
        help="fail before MACE forward if the fixed graph exceeds this node count",
    )
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        help="optional Linux process address-space cap applied before Torch/MACE import",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.edge_cutoff_angstrom not in ENCODER_VARIANTS:
        parser.error(
            "--edge-cutoff-angstrom must be exactly one of the preregistered "
            f"EXP-012 encoder variants: {sorted(ENCODER_VARIANTS)} "
            "(6.0 -> original_6a per DEC-024, 5.0 -> derived_5a per DEC-027); "
            "a new cutoff value requires its own preregistered variant, not this flag"
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
    manifest = load_environment_manifest(args.environment_manifest)
    mapping = load_atom_mapping(args.atom_mapping, environment_manifest=manifest)
    c0 = load_c0_report(args.c0_report)
    adapter = None
    graph_config = MaceGraphConfig(
        edge_cutoff_angstrom=args.edge_cutoff_angstrom,
        interaction_layers=args.product_layer_index + 1,
        geometric_upper_bound_angstrom=args.geometric_upper_bound_angstrom,
    )
    torch_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)
    frames = []
    for frame_index in args.frame_indices:
        frame_started = time.perf_counter()
        trajectory = mdtraj.load_frame(args.trajectory, index=frame_index, top=args.topology)
        if trajectory.n_frames != 1 or trajectory.unitcell_vectors is None:
            raise RuntimeError("frame source must provide exactly one frame with triclinic cell vectors")
        if trajectory.n_atoms != manifest["payload"]["atom_count"]:
            raise RuntimeError("frame atom count differs from the environment manifest")
        positions = torch.tensor(
            trajectory.xyz[0] * 10.0, dtype=torch_dtype, device=device, requires_grad=True
        )
        cell = torch.tensor(trajectory.unitcell_vectors[0] * 10.0, dtype=torch_dtype, device=device)
        graph = build_mace_graph(
            positions, cell,
            environment_manifest=manifest, atom_mapping=mapping,
            model_atomic_numbers=c0["expected"]["atomic_numbers"], config=graph_config,
        )
        node_count = int(graph["diagnostics"]["node_count"])
        edge_count = int(graph["diagnostics"]["edge_count"])
        print(
            f"frame {frame_index}: graph ready with {node_count} nodes and "
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
                c0_report=c0,
                model_path=args.model,
                device=args.device,
                dtype=args.dtype,
                product_layer_index=args.product_layer_index,
            )
        # Determinism reference does not need an autograd graph.  Running it
        # first and discarding its intermediates avoids retaining two complete
        # extra-large MACE forward graphs at once.
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
        # Fixed, parameter-free diagnostic scalar. This is not candidate B/readout.
        weights = torch.linspace(1.0, 2.0, latent.shape[1], dtype=latent.dtype, device=latent.device)
        scalar_probe = (latent * weights).sum()
        coordinate_gradient = torch.autograd.grad(scalar_probe, positions)[0]
        print(f"frame {frame_index}: autograd completed", flush=True)
        ligand_topology = torch.tensor(
            manifest["payload"]["ligand_indices"], dtype=torch.long, device=device
        )
        environment_topology = torch.tensor(
            manifest["payload"]["environment_candidate_indices"], dtype=torch.long, device=device
        )
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
            ("environment_manifest", args.environment_manifest), ("atom_mapping", args.atom_mapping),
            ("topology", args.topology), ("trajectory", args.trajectory),
        )
    }
    body = {
        "schema_version": "exp012-mace-latent-autograd-smoke-v1",
        "status": "COMPLETED_AUTOGRAD_SMOKE_ONLY",
        "encoder_variant": encoder_variant,
        "model_r_max_angstrom": float(c0["expected"]["r_max_angstrom"]),
        "graph_cutoff_angstrom": float(args.edge_cutoff_angstrom),
        "original_encoder_numerically_preserved": encoder_variant == "original_6a",
        "inputs": inputs,
        "environment_manifest_sha256": manifest["canonical_sha256"],
        "atom_mapping_sha256": mapping["canonical_sha256"],
        "c0_report_sha256": c0["report_sha256"],
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
            "edge_membership_piecewise_fixed": True,
            "edge_displacements_autograd_connected": True,
            "scalar_probe_role": "autograd_smoke_only_not_candidate_B",
        },
    }
    report = {**body, "report_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest()}
    _atomic_json_write(Path(args.output), report)
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
